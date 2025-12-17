"""
Batch experiment runner for out-of-distribution (OOD) domain generalization experiments.

Runs RDEIC on various OOD domains and logs metrics to CSV.
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Dict

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


def compute_noreference_metrics(pred: np.ndarray, device: str = "cuda") -> dict:
    """
    Compute no-reference image quality metrics.
    
    Args:
        pred: Predicted image (H, W, C), range [0, 255].
        device: Device for metric computation.
        
    Returns:
        Dictionary with metric values.
    """
    import pyiqa
    
    pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    pred_t = pred_t.to(device)
    
    metrics = {}
    
    # NIQE - lower is better
    try:
        niqe_fn = pyiqa.create_metric("niqe", device=device)
        metrics["niqe"] = niqe_fn(pred_t).item()
    except Exception as e:
        print(f"Warning: NIQE computation failed: {e}")
        metrics["niqe"] = float("nan")
    
    # BRISQUE - lower is better
    try:
        brisque_fn = pyiqa.create_metric("brisque", device=device)
        metrics["brisque"] = brisque_fn(pred_t).item()
    except Exception as e:
        print(f"Warning: BRISQUE computation failed: {e}")
        metrics["brisque"] = float("nan")
    
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


@torch.no_grad()
def process_image(
    model: RDEIC,
    img: np.ndarray,
    stream_path: str,
    c_crossattn: List[torch.Tensor],
    sampler_type: str = "ddpm",
    steps: int = 2,
    guidance_scale: float = 1.0,
    latent_noise_std: float = 0.0,
    num_samples: int = 1,
    target_for_lpips: Optional[np.ndarray] = None
) -> tuple:
    """
    Process a single image through RDEIC.
    
    Args:
        model: RDEIC model.
        img: Input image (H, W, C), range [0, 255].
        stream_path: Path to save/load compressed stream.
        c_crossattn: Cross-attention conditioning.
        sampler_type: 'ddpm' or 'ddim'.
        steps: Number of sampling steps.
        guidance_scale: Classifier-free guidance scale.
        latent_noise_std: Std of noise to add to latent for test-time augmentation.
        num_samples: Number of noisy samples to generate.
        target_for_lpips: Target image for LPIPS comparison (H, W, C), range [0, 255].
    
    Returns:
        (reconstructed_image, bpp)
    """
    control = torch.tensor(img / 255.0, dtype=torch.float32).clamp_(0, 1)
    control = einops.rearrange(control, "h w c -> 1 c h w").contiguous()
    control = control.to(model.device, non_blocking=True)
    
    height, width = control.size(-2), control.size(-1)
    
    # Compress and decompress
    bpp = model.apply_condition_compress(control, stream_path, height, width)
    c_latent, guide_hint = model.apply_condition_decompress(stream_path)
    
    # Optional: Add latent noise for test-time augmentation
    if latent_noise_std > 0 and num_samples > 1:
        import pyiqa
        
        # Create LPIPS metric if we have a target
        lpips_fn = None
        target_t = None
        if target_for_lpips is not None:
            lpips_fn = pyiqa.create_metric("lpips", device=model.device)
            target_t = torch.from_numpy(target_for_lpips).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            target_t = target_t.to(model.device)
        
        # Sample multiple noisy latents and pick best by LPIPS
        best_sample = None
        best_lpips = float("inf")
        
        for i in range(num_samples):
            noisy_latent = c_latent + torch.randn_like(c_latent) * latent_noise_std
            sample = decode_single(model, noisy_latent, guide_hint, c_crossattn, 
                                   sampler_type, steps, guidance_scale)
            
            if lpips_fn is not None and target_t is not None:
                # Compute LPIPS between sample and target
                sample_t = torch.from_numpy(sample).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                sample_t = sample_t.to(model.device)
                lpips_val = lpips_fn(sample_t, target_t).item()
                
                if lpips_val < best_lpips:
                    best_lpips = lpips_val
                    best_sample = sample
            else:
                # No target available, just use first sample
                if best_sample is None:
                    best_sample = sample
        
        return best_sample, bpp
    
    # Standard decoding
    sample = decode_single(model, c_latent, guide_hint, c_crossattn,
                          sampler_type, steps, guidance_scale)
    
    return sample, bpp


def decode_single(
    model: RDEIC,
    c_latent: torch.Tensor,
    guide_hint: torch.Tensor,
    c_crossattn: List[torch.Tensor],
    sampler_type: str,
    steps: int,
    guidance_scale: float
) -> np.ndarray:
    """Decode from latent to image."""
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
    """Run the OOD experiment."""
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()
    
    if args.suppress_warnings:
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.ERROR)
    
    print("Loading model...")
    model = load_model(args)
    c_crossattn = [model.get_learned_conditioning([""])]
    
    # Parse domains
    domains = [d.strip() for d in args.domains.split(",")]
    
    all_results = []
    
    for domain_path in domains:
        domain_name = Path(domain_path).name
        print(f"\nProcessing domain: {domain_name}")
        
        if not os.path.isdir(domain_path):
            print(f"Warning: {domain_path} is not a directory, skipping")
            continue
        
        image_files = list(list_image_files(domain_path, follow_links=True))
        print(f"Found {len(image_files)} images")
        
        # Prepare output directories
        stream_dir = os.path.join(args.cache_dir, domain_name)
        recon_dir = os.path.join(args.output_dir, domain_name)
        os.makedirs(stream_dir, exist_ok=True)
        if args.save_recon:
            os.makedirs(recon_dir, exist_ok=True)
        
        domain_results = []
        
        for img_path in tqdm(image_files, desc=f"Processing {domain_name}"):
            img_name = Path(img_path).stem
            
            try:
                # Load and pad image
                img = Image.open(img_path).convert("RGB")
                orig_size = img.size  # (W, H)
                img_np = pad(np.array(img), scale=64)
                target_np = np.array(img)  # Keep original for metrics
                
                # For LPIPS selection, we need padded target too
                target_padded = pad(target_np, scale=64) if args.latent_noise_std > 0 else None
                
                # Process
                stream_path = os.path.join(stream_dir, img_name)
                pred, bpp = process_image(
                    model, img_np, stream_path, c_crossattn,
                    sampler_type=args.sampler, steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    latent_noise_std=args.latent_noise_std,
                    num_samples=args.num_noise_samples,
                    target_for_lpips=target_padded
                )
                
                # Crop padding
                pred_cropped = pred[:orig_size[1], :orig_size[0], :]
                
                # Compute metrics (target_np was defined earlier)
                metrics = compute_metrics(pred_cropped, target_np, device=args.device)
                
                # Also compute no-reference metrics
                nr_metrics = compute_noreference_metrics(pred_cropped, device=args.device)
                
                # Save reconstruction if requested
                if args.save_recon:
                    Image.fromarray(pred_cropped).save(os.path.join(recon_dir, f"{img_name}.png"))
                
                result = {
                    "domain": domain_name,
                    "image_id": img_name,
                    "bpp": bpp,
                    **metrics,
                    **nr_metrics
                }
                domain_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save per-domain results
        if domain_results:
            domain_df = pd.DataFrame(domain_results)
            domain_csv = os.path.join(args.output_dir, f"ood_results_{domain_name}.csv")
            domain_df.to_csv(domain_csv, index=False)
            print(f"Domain results saved to {domain_csv}")
            
            # Print domain summary
            print(f"\n=== {domain_name} Summary ===")
            print(f"Mean PSNR: {domain_df['psnr'].mean():.2f} dB")
            print(f"Mean SSIM: {domain_df['ssim'].mean():.4f}")
            print(f"Mean MS-SSIM: {domain_df['ms_ssim'].mean():.4f}")
            print(f"Mean LPIPS: {domain_df['lpips'].mean():.4f}")
            print(f"Mean BPP: {domain_df['bpp'].mean():.4f}")
    
    # Save consolidated results
    if all_results:
        all_df = pd.DataFrame(all_results)
        all_csv = os.path.join(args.output_dir, "ood_results_all.csv")
        all_df.to_csv(all_csv, index=False)
        print(f"\nConsolidated results saved to {all_csv}")
        
        # Print overall summary
        print("\n=== Overall Summary by Domain ===")
        summary = all_df.groupby("domain").agg({
            "psnr": ["mean", "std"],
            "ssim": ["mean", "std"],
            "ms_ssim": ["mean", "std"],
            "lpips": ["mean", "std"],
            "bpp": "mean"
        }).round(4)
        print(summary)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run OOD domain generalization experiments on RDEIC")
    
    # Model arguments
    parser.add_argument("--ckpt_sd", default="./weight/v2-1_512-ema-pruned.ckpt", type=str)
    parser.add_argument("--ckpt_cc", default="./weight/rdeic_2_step2.ckpt", type=str)
    parser.add_argument("--config", default="configs/model/rdeic.yaml", type=str)
    
    # Data arguments
    parser.add_argument("--domains", type=str, required=True,
                        help="Comma-separated list of domain folder paths (e.g., 'ood/sketch,ood/xray')")
    
    # Sampling arguments
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    
    # Test-time augmentation
    parser.add_argument("--latent_noise_std", type=float, default=0.0,
                        help="Standard deviation of noise to add to latent (0 to disable)")
    parser.add_argument("--num_noise_samples", type=int, default=1,
                        help="Number of noisy samples to generate when using latent noise")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="indicators/")
    parser.add_argument("--cache_dir", type=str, default="encodings/ood/")
    parser.add_argument("--save_recon", action="store_true", help="Save reconstructed images")
    
    # Runtime arguments
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--suppress_warnings", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

