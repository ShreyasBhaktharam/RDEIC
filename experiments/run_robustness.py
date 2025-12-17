"""
Batch experiment runner for error robustness experiments.

Runs RDEIC under various error conditions and logs metrics to CSV.
"""

import os
import sys
import shutil
import warnings
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldm.xformers_state import disable_xformers
from model.spaced_sampler_relay import SpacedSampler
from model.ddim_sampler_relay import DDIMSampler
from model.rdeic import RDEIC
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.utils import read_body, write_body, filesize

from corruptors import Corruptor, bit_flip_bytes, burst_flip_bytes, latent_corrupt


def compute_metrics(pred: np.ndarray, target: np.ndarray, device: str = "cuda") -> dict:
    """
    Compute image quality metrics between prediction and target.
    
    Args:
        pred: Predicted image (H, W, C), range [0, 255].
        target: Target image (H, W, C), range [0, 255].
        device: Device for metric computation.
        
    Returns:
        Dictionary with metric values.
    """
    import pyiqa
    
    # Convert to tensors (NCHW, [0, 1])
    pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    target_t = torch.from_numpy(target).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    
    metrics = {}
    
    # PSNR
    mse = torch.mean((pred_t - target_t) ** 2)
    if mse > 0:
        metrics["psnr"] = (10 * torch.log10(1.0 / mse)).item()
    else:
        metrics["psnr"] = 100.0  # Perfect match
    
    # SSIM and MS-SSIM
    try:
        ssim_fn = pyiqa.create_metric("ssim", device=device)
        metrics["ssim"] = ssim_fn(pred_t, target_t).item()
    except Exception as e:
        print(f"Warning: SSIM computation failed: {e}")
        metrics["ssim"] = float("nan")
    
    try:
        msssim_fn = pyiqa.create_metric("ms_ssim", device=device)
        metrics["ms_ssim"] = msssim_fn(pred_t, target_t).item()
    except Exception as e:
        print(f"Warning: MS-SSIM computation failed: {e}")
        metrics["ms_ssim"] = float("nan")
    
    # LPIPS
    try:
        lpips_fn = pyiqa.create_metric("lpips", device=device)
        metrics["lpips"] = lpips_fn(pred_t, target_t).item()
    except Exception as e:
        print(f"Warning: LPIPS computation failed: {e}")
        metrics["lpips"] = float("nan")
    
    return metrics


def load_model(args: Namespace) -> RDEIC:
    """Load and prepare the RDEIC model."""
    model: RDEIC = instantiate_from_config(OmegaConf.load(args.config))
    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")["state_dict"]
    ckpt_lc = torch.load(args.ckpt_cc, map_location="cpu")["state_dict"]
    ckpt_sd.update(ckpt_lc)
    load_state_dict(model, ckpt_sd, strict=False)
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)
    return model


def encode_image(
    model: RDEIC,
    img: np.ndarray,
    stream_path: str,
    c_crossattn: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Encode an image and save the bitstream.
    
    Returns:
        (c_latent, guide_hint, bpp)
    """
    control = torch.tensor(img / 255.0, dtype=torch.float32).clamp_(0, 1)
    control = einops.rearrange(control, "h w c -> 1 c h w").contiguous()
    control = control.to(model.device, non_blocking=True)
    
    height, width = control.size(-2), control.size(-1)
    bpp = model.apply_condition_compress(control, stream_path, height, width)
    c_latent, guide_hint = model.apply_condition_decompress(stream_path)
    
    return c_latent, guide_hint, bpp


def decode_from_latent(
    model: RDEIC,
    c_latent: torch.Tensor,
    guide_hint: torch.Tensor,
    c_crossattn: List[torch.Tensor],
    sampler_type: str = "ddpm",
    steps: int = 2,
    guidance_scale: float = 1.0
) -> np.ndarray:
    """
    Decode from a latent tensor.
    
    Returns:
        Reconstructed image (H, W, C), range [0, 255].
    """
    b, c, h, w = c_latent.shape
    
    if sampler_type == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    
    cond = {
        "c_latent": [c_latent],
        "c_crossattn": c_crossattn,
        "guide_hint": guide_hint
    }
    
    shape = (b, 4, h, w)
    t = torch.ones((b,)).long().to(model.device) * model.used_timesteps - 1
    noise = torch.randn(shape, device=model.device, dtype=torch.float32)
    x_T = model.q_sample(x_start=c_latent, t=t, noise=noise)
    
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            unconditional_guidance_scale=guidance_scale,
            x_T=x_T, eta=0
        )
    
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy()
    x_samples = x_samples.clip(0, 255).astype(np.uint8)
    
    return x_samples[0]


def run_experiment(args: Namespace) -> None:
    """Run the robustness experiment."""
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()
    
    # Suppress warnings if requested
    if args.suppress_warnings:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.ERROR)
    
    print("Loading model...")
    model = load_model(args)
    c_crossattn = [model.get_learned_conditioning([""])]
    
    # Parse rates
    rates = [float(r) / 100.0 for r in args.rates.split(",")]
    
    # Get image files
    if os.path.isfile(args.data) and args.data.endswith(".list"):
        with open(args.data) as f:
            image_files = [line.strip() for line in f if line.strip()]
    else:
        image_files = list(list_image_files(args.data, follow_links=True))
    
    print(f"Found {len(image_files)} images")
    print(f"Error rates: {[r*100 for r in rates]}%")
    print(f"Error space: {args.error_space}, Error type: {args.error_type}")
    print(f"Seeds per rate: {args.num_seeds}")
    
    # Prepare output directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.save_recon:
        os.makedirs(args.recon_dir, exist_ok=True)
    
    results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = Path(img_path).stem
        
        # Load and pad image
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (W, H)
        img_np = pad(np.array(img), scale=64)
        padded_h, padded_w = img_np.shape[:2]
        
        # Encode once and cache
        stream_path = os.path.join(args.cache_dir, f"{img_name}")
        latent_path = os.path.join(args.cache_dir, f"{img_name}_latent.pt")
        hint_path = os.path.join(args.cache_dir, f"{img_name}_hint.pt")
        
        if not os.path.exists(stream_path) or args.force_encode:
            os.makedirs(os.path.dirname(stream_path), exist_ok=True)
            c_latent, guide_hint, bpp = encode_image(model, img_np, stream_path, c_crossattn)
            torch.save(c_latent.cpu(), latent_path)
            torch.save(guide_hint.cpu(), hint_path)
        else:
            c_latent = torch.load(latent_path).to(model.device)
            guide_hint = torch.load(hint_path).to(model.device)
            # Recalculate bpp from file size
            size = os.path.getsize(stream_path)
            bpp = float(size) * 8 / (padded_h * padded_w)
        
        # Target image for metrics (original resolution, no padding)
        target_np = np.array(img)
        
        for rate in rates:
            for seed_idx in range(args.num_seeds):
                seed = args.seed + seed_idx
                
                try:
                    if args.error_space == "bitstream":
                        # Read and corrupt bitstream
                        with open(stream_path, "rb") as f:
                            data = f.read()
                        
                        if args.error_type == "random":
                            corrupted_data = bit_flip_bytes(data, rate, seed)
                        else:  # burst
                            corrupted_data = burst_flip_bytes(data, rate, args.burst_len, seed)
                        
                        # Write corrupted bitstream temporarily
                        corrupted_path = os.path.join(args.cache_dir, f"{img_name}_corrupted_{rate}_{seed}")
                        with open(corrupted_path, "wb") as f:
                            f.write(corrupted_data)
                        
                        # Decompress from corrupted bitstream
                        try:
                            c_latent_corrupt, guide_hint_corrupt = model.apply_condition_decompress(corrupted_path)
                        except Exception as e:
                            print(f"Decompression failed for {img_name} at rate {rate*100:.1f}%, seed {seed}: {e}")
                            # Record as catastrophic failure
                            results.append({
                                "image_id": img_name,
                                "error_space": args.error_space,
                                "error_type": args.error_type,
                                "error_rate": rate * 100,
                                "seed": seed,
                                "bpp": bpp,
                                "psnr": 0.0,
                                "ssim": 0.0,
                                "ms_ssim": 0.0,
                                "lpips": 1.0,
                                "decode_failed": True
                            })
                            continue
                        finally:
                            if os.path.exists(corrupted_path):
                                os.remove(corrupted_path)
                    
                    else:  # latent
                        c_latent_corrupt = latent_corrupt(c_latent, args.error_type, rate, seed)
                        guide_hint_corrupt = guide_hint  # Keep guide hint unchanged
                    
                    # Decode
                    pred = decode_from_latent(
                        model, c_latent_corrupt, guide_hint_corrupt, c_crossattn,
                        sampler_type=args.sampler, steps=args.steps, guidance_scale=args.guidance_scale
                    )
                    
                    # Crop padding for metrics
                    pred_cropped = pred[:orig_size[1], :orig_size[0], :]
                    
                    # Compute metrics
                    metrics = compute_metrics(pred_cropped, target_np, device=args.device)
                    
                    # Save reconstruction if requested
                    if args.save_recon:
                        recon_dir = os.path.join(args.recon_dir, img_name, f"r_{rate*100:.1f}", f"seed_{seed}")
                        os.makedirs(recon_dir, exist_ok=True)
                        Image.fromarray(pred_cropped).save(os.path.join(recon_dir, "recon.png"))
                    
                    results.append({
                        "image_id": img_name,
                        "error_space": args.error_space,
                        "error_type": args.error_type,
                        "error_rate": rate * 100,
                        "seed": seed,
                        "bpp": bpp,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                        "ms_ssim": metrics["ms_ssim"],
                        "lpips": metrics["lpips"],
                        "decode_failed": False
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_name} at rate {rate*100:.1f}%, seed {seed}: {e}")
                    results.append({
                        "image_id": img_name,
                        "error_space": args.error_space,
                        "error_type": args.error_type,
                        "error_rate": rate * 100,
                        "seed": seed,
                        "bpp": bpp,
                        "psnr": 0.0,
                        "ssim": 0.0,
                        "ms_ssim": 0.0,
                        "lpips": 1.0,
                        "decode_failed": True
                    })
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Print summary
    print("\n=== Summary ===")
    summary = df.groupby("error_rate").agg({
        "psnr": ["mean", "std"],
        "ssim": ["mean", "std"],
        "ms_ssim": ["mean", "std"],
        "lpips": ["mean", "std"],
        "decode_failed": "sum"
    }).round(4)
    print(summary)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run robustness experiments on RDEIC")
    
    # Model arguments
    parser.add_argument("--ckpt_sd", default="./weight/v2-1_512-ema-pruned.ckpt", type=str)
    parser.add_argument("--ckpt_cc", default="./weight/rdeic_2_step2.ckpt", type=str)
    parser.add_argument("--config", default="configs/model/rdeic.yaml", type=str)
    
    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to image folder or .list file")
    
    # Experiment arguments
    parser.add_argument("--error_space", type=str, choices=["bitstream", "latent"],
                        default="bitstream", help="Where to inject errors")
    parser.add_argument("--error_type", type=str, choices=["random", "burst", "mask_replace", "additive"],
                        default="random", help="Type of error injection")
    parser.add_argument("--rates", type=str, default="0,0.1,0.5,1,2,5,10",
                        help="Comma-separated error rates in percent")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of random seeds per rate")
    parser.add_argument("--burst_len", type=float, default=8.0,
                        help="Mean burst length for burst errors")
    
    # Sampling arguments
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    
    # Output arguments
    parser.add_argument("--output_csv", type=str, default="indicators/robustness_results.csv")
    parser.add_argument("--cache_dir", type=str, default="encodings/")
    parser.add_argument("--recon_dir", type=str, default="outputs/robustness/")
    parser.add_argument("--save_recon", action="store_true", help="Save reconstructed images")
    parser.add_argument("--force_encode", action="store_true", help="Re-encode even if cache exists")
    
    # Runtime arguments
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--suppress_warnings", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

