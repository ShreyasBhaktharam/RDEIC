"""
JPEG2000 robustness comparison script.
Compresses images with OpenJPEG at similar BPP to RDEIC, 
then applies the same bit-flip corruption and measures metrics.

Requires: openjpeg (install via: sudo apt install libopenjpeg-tools or conda install -c conda-forge openjpeg)
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corruptors import bit_flip_bytes, burst_flip_bytes
from utils.file import list_image_files


def compute_metrics(pred: np.ndarray, target: np.ndarray, device: str = "cuda") -> dict:
    """Compute PSNR, SSIM, MS-SSIM, LPIPS."""
    import pyiqa
    
    pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    target_t = torch.from_numpy(target).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    
    metrics = {}
    
    # PSNR
    mse = torch.mean((pred_t - target_t) ** 2)
    metrics["psnr"] = (10 * torch.log10(1.0 / (mse + 1e-10))).item()
    
    # SSIM
    try:
        ssim_fn = pyiqa.create_metric("ssim", device=device)
        metrics["ssim"] = ssim_fn(pred_t, target_t).item()
    except:
        metrics["ssim"] = float("nan")
    
    # MS-SSIM
    try:
        msssim_fn = pyiqa.create_metric("ms_ssim", device=device)
        metrics["ms_ssim"] = msssim_fn(pred_t, target_t).item()
    except:
        metrics["ms_ssim"] = float("nan")
    
    # LPIPS
    try:
        lpips_fn = pyiqa.create_metric("lpips", device=device)
        metrics["lpips"] = lpips_fn(pred_t, target_t).item()
    except:
        metrics["lpips"] = float("nan")
    
    return metrics


def check_openjpeg():
    """Check if OpenJPEG tools are available."""
    try:
        subprocess.run(["opj_compress", "-h"], capture_output=True, check=False)
        subprocess.run(["opj_decompress", "-h"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def compress_jpeg2000(input_ppm_path: str, output_path: str, target_bpp: float, width: int, height: int) -> float:
    """
    Compress PPM image to JPEG2000 at target BPP.
    
    Args:
        input_ppm_path: Path to PPM format image (OpenJPEG always supports PPM)
        output_path: Path for output J2K file
        target_bpp: Target bits per pixel
        width, height: Image dimensions for BPP calculation
    
    Returns actual BPP achieved.
    """
    total_pixels = width * height
    
    # Calculate target rate (compression ratio)
    # JPEG2000 -r flag is compression ratio, not rate
    # For target_bpp of 0.12 on 24-bit RGB: ratio = 24 / 0.12 = 200
    compression_ratio = 24.0 / target_bpp if target_bpp > 0 else 100
    
    # Compress using opj_compress
    cmd = [
        "opj_compress",
        "-i", input_ppm_path,
        "-o", output_path,
        "-r", str(compression_ratio),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Compression failed: {e.stderr}")
        return 0.0
    
    if not os.path.exists(output_path):
        return 0.0
    
    # Calculate actual BPP
    file_size = os.path.getsize(output_path)
    actual_bpp = (file_size * 8) / total_pixels
    
    return actual_bpp


def decompress_jpeg2000(input_path: str, output_path: str) -> bool:
    """Decompress JPEG2000 file to PPM."""
    cmd = [
        "opj_decompress",
        "-i", input_path,
        "-o", output_path,
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_experiment(args: Namespace) -> None:
    """Run JPEG2000 robustness experiment."""
    
    if not check_openjpeg():
        print("ERROR: OpenJPEG tools not found!")
        print("Install with: sudo apt install libopenjpeg-tools")
        print("Or: conda install -c conda-forge openjpeg")
        return
    
    # Parse rates
    rates = [float(r) / 100.0 for r in args.rates.split(",")]
    
    # Get image files
    if os.path.isfile(args.data) and args.data.endswith(".list"):
        with open(args.data) as f:
            image_files = [line.strip() for line in f if line.strip()]
    else:
        image_files = list(list_image_files(args.data, follow_links=True))
    
    print(f"Found {len(image_files)} images")
    print(f"Target BPP: {args.target_bpp}")
    print(f"Error rates: {[r*100 for r in rates]}%")
    print(f"Error type: {args.error_type}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for img_path in tqdm(image_files, desc="Processing images"):
            img_name = Path(img_path).stem
            
            # Load original image and convert to PPM (always supported by OpenJPEG)
            img = Image.open(img_path).convert("RGB")
            target_np = np.array(img)
            width, height = img.size
            
            # Save as PPM for OpenJPEG input
            ppm_path = os.path.join(tmpdir, f"{img_name}.ppm")
            img.save(ppm_path, format="PPM")
            
            # Compress to JPEG2000
            j2k_path = os.path.join(tmpdir, f"{img_name}.j2k")
            actual_bpp = compress_jpeg2000(ppm_path, j2k_path, args.target_bpp, width, height)
            
            if actual_bpp == 0:
                print(f"Failed to compress {img_name}")
                continue
            
            print(f"  {img_name}: compressed at {actual_bpp:.4f} bpp")
            
            # Read compressed bitstream
            with open(j2k_path, "rb") as f:
                clean_data = f.read()
            
            for rate in rates:
                for seed_idx in range(args.num_seeds):
                    seed = args.seed + seed_idx
                    
                    try:
                        # Corrupt bitstream
                        if rate == 0:
                            corrupted_data = clean_data
                        elif args.error_type == "random":
                            corrupted_data = bit_flip_bytes(clean_data, rate, seed)
                        else:  # burst
                            corrupted_data = burst_flip_bytes(clean_data, rate, args.burst_len, seed)
                        
                        # Write corrupted bitstream
                        corrupted_path = os.path.join(tmpdir, f"{img_name}_corrupted.j2k")
                        with open(corrupted_path, "wb") as f:
                            f.write(corrupted_data)
                        
                        # Decompress to PPM
                        decoded_path = os.path.join(tmpdir, f"{img_name}_decoded.ppm")
                        success = decompress_jpeg2000(corrupted_path, decoded_path)
                        
                        if success and os.path.exists(decoded_path):
                            # Load and compute metrics
                            pred_img = Image.open(decoded_path).convert("RGB")
                            pred_np = np.array(pred_img)
                            
                            # Ensure same size
                            if pred_np.shape != target_np.shape:
                                pred_img = pred_img.resize((target_np.shape[1], target_np.shape[0]), Image.LANCZOS)
                                pred_np = np.array(pred_img)
                            
                            metrics = compute_metrics(pred_np, target_np, device=args.device)
                            
                            # Save reconstruction if requested
                            if args.save_recon and rate > 0:
                                recon_dir = os.path.join(args.output_dir, img_name, f"r_{rate*100:.1f}")
                                os.makedirs(recon_dir, exist_ok=True)
                                pred_img.save(os.path.join(recon_dir, f"seed_{seed}.png"))
                            
                            results.append({
                                "codec": "JPEG2000",
                                "image_id": img_name,
                                "error_type": args.error_type,
                                "error_rate": rate * 100,
                                "seed": seed,
                                "bpp": actual_bpp,
                                "psnr": metrics["psnr"],
                                "ssim": metrics["ssim"],
                                "ms_ssim": metrics["ms_ssim"],
                                "lpips": metrics["lpips"],
                                "decode_failed": False
                            })
                        else:
                            results.append({
                                "codec": "JPEG2000",
                                "image_id": img_name,
                                "error_type": args.error_type,
                                "error_rate": rate * 100,
                                "seed": seed,
                                "bpp": actual_bpp,
                                "psnr": 0.0,
                                "ssim": 0.0,
                                "ms_ssim": 0.0,
                                "lpips": 1.0,
                                "decode_failed": True
                            })
                    
                    except Exception as e:
                        print(f"Error processing {img_name} at rate {rate*100:.1f}%: {e}")
                        results.append({
                            "codec": "JPEG2000",
                            "image_id": img_name,
                            "error_type": args.error_type,
                            "error_rate": rate * 100,
                            "seed": seed,
                            "bpp": actual_bpp,
                            "psnr": 0.0,
                            "ssim": 0.0,
                            "ms_ssim": 0.0,
                            "lpips": 1.0,
                            "decode_failed": True
                        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Print summary
    if len(df) > 0:
        print("\n=== Summary ===")
        summary = df.groupby("error_rate").agg({
            "psnr": ["mean", "std"],
            "ssim": ["mean", "std"],
            "ms_ssim": ["mean", "std"],
            "lpips": ["mean", "std"],
            "decode_failed": "sum"
        }).round(4)
        print(summary)
    else:
        print("\nNo results to summarize - all compressions failed.")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run JPEG2000 robustness experiments")
    
    parser.add_argument("--data", type=str, required=True,
                        help="Path to image folder or .list file")
    parser.add_argument("--target_bpp", type=float, default=0.12,
                        help="Target bits per pixel (match RDEIC's BPP)")
    parser.add_argument("--error_type", type=str, choices=["random", "burst"],
                        default="random", help="Type of error injection")
    parser.add_argument("--rates", type=str, default="0,0.1,0.5,1,2,5,10",
                        help="Comma-separated error rates in percent")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of random seeds per rate")
    parser.add_argument("--burst_len", type=float, default=8.0,
                        help="Mean burst length for burst errors")
    
    parser.add_argument("--output_csv", type=str, default="indicators/jpeg2000_robustness.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/jpeg2000_robustness/")
    parser.add_argument("--save_recon", action="store_true")
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
