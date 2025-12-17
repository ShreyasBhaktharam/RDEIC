"""
Baseline sanity check script.

Run RDEIC on a small set of images to verify the setup works correctly.
Saves reconstructions and basic metrics to confirm everything is functional.
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace

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


def load_model(args: Namespace) -> RDEIC:
    """Load RDEIC model."""
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
def process_single(model, img_np, stream_path, c_crossattn, sampler_type="ddpm", steps=2):
    """Process a single image through RDEIC."""
    control = torch.tensor(img_np / 255.0, dtype=torch.float32).clamp_(0, 1)
    control = einops.rearrange(control, "h w c -> 1 c h w").contiguous()
    control = control.to(model.device)
    
    h, w = control.size(-2), control.size(-1)
    
    # Compress and decompress
    bpp = model.apply_condition_compress(control, stream_path, h, w)
    c_latent, guide_hint = model.apply_condition_decompress(stream_path)
    
    # Sample
    if sampler_type == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    
    cond = {
        "c_latent": [c_latent],
        "c_crossattn": c_crossattn,
        "guide_hint": guide_hint
    }
    
    # Use c_latent shape (already in latent space: h//8, w//8)
    shape = c_latent.shape
    t = torch.ones((1,)).long().to(model.device) * model.used_timesteps - 1
    noise = torch.randn(shape, device=model.device)
    x_T = model.q_sample(x_start=c_latent, t=t, noise=noise)
    
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(steps, shape, cond, 
                                 unconditional_guidance_scale=1.0,
                                 unconditional_conditioning=None,
                                 cond_fn=None, x_T=x_T)
    else:
        samples, _ = sampler.sample(S=steps, batch_size=1, shape=shape[1:],
                                    conditioning=cond, x_T=x_T, eta=0)
    
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy()
    pred = x_samples[0].clip(0, 255).astype(np.uint8)
    
    return pred, bpp


def main(args: Namespace):
    """Run baseline inference."""
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()
    
    print("Loading model...")
    model = load_model(args)
    c_crossattn = [model.get_learned_conditioning([""])]
    
    # Get image files
    if os.path.isdir(args.input):
        image_files = list(list_image_files(args.input, follow_links=True))
    else:
        image_files = [args.input]
    
    print(f"Found {len(image_files)} images")
    if args.max_long_side > 0:
        print(f"Downscaling images with max(H,W) > {args.max_long_side}")
    if args.upsample_to_original:
        print(f"Upsampling outputs to original resolution using {args.upsample_method}")
    
    # Prepare output
    os.makedirs(args.output, exist_ok=True)
    stream_dir = os.path.join(args.output, "data")
    os.makedirs(stream_dir, exist_ok=True)
    
    results = []
    
    # Prepare resample filter for optional upsampling
    if args.upsample_method == "lanczos":
        _resample = Image.LANCZOS
    elif args.upsample_method == "bicubic":
        _resample = Image.BICUBIC
    elif args.upsample_method == "bilinear":
        _resample = Image.BILINEAR
    else:
        _resample = Image.NEAREST
    
    for img_path in tqdm(image_files, desc="Processing"):
        img_name = Path(img_path).stem
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Optionally downscale to limit working size while preserving aspect ratio
        target_max = int(args.max_long_side) if args.max_long_side and args.max_long_side > 0 else 0
        if target_max and max(orig_w, orig_h) > target_max:
            scale = target_max / max(orig_w, orig_h)
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))
            img_scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)
        else:
            scale = 1.0
            scaled_w, scaled_h = orig_w, orig_h
            img_scaled = img
        
        # Pad to multiple of 64
        img_np = pad(np.array(img_scaled), scale=64)
        
        # Process
        stream_path = os.path.join(stream_dir, img_name)
        pred, bpp = process_single(model, img_np, stream_path, c_crossattn,
                                   sampler_type=args.sampler, steps=args.steps)
        
        # Crop padding (use scaled dimensions)
        pred_cropped = pred[:scaled_h, :scaled_w, :]
        
        # Optionally upsample back to original resolution
        if args.upsample_to_original and scale < 1.0:
            pred_img = Image.fromarray(pred_cropped)
            pred_img = pred_img.resize((orig_w, orig_h), _resample)
            pred_final = np.array(pred_img)
        else:
            pred_final = pred_cropped
        
        # Save reconstruction
        save_path = os.path.join(args.output, f"{img_name}.png")
        Image.fromarray(pred_final).save(save_path)
        
        # Compute metrics
        # If upsampled, compare against original; otherwise compare against scaled target
        if args.upsample_to_original and scale < 1.0:
            target_np = np.array(img)  # Original resolution
        else:
            target_np = np.array(img_scaled)  # Scaled resolution
        metrics = compute_metrics(pred_final, target_np, device=args.device)
        
        results.append({
            "image": img_name,
            "bpp": bpp,
            "scale": scale,
            **metrics
        })
        
        scale_info = f" (scale={scale:.2f})" if scale < 1.0 else ""
        print(f"{img_name}{scale_info}: BPP={bpp:.4f}, PSNR={metrics['psnr']:.2f}dB, "
              f"SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output, "baseline_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n=== Summary ===")
    print(f"Mean BPP:    {df['bpp'].mean():.4f}")
    print(f"Mean PSNR:   {df['psnr'].mean():.2f} dB")
    print(f"Mean SSIM:   {df['ssim'].mean():.4f}")
    print(f"Mean MS-SSIM: {df['ms_ssim'].mean():.4f}")
    print(f"Mean LPIPS:  {df['lpips'].mean():.4f}")
    print(f"\nResults saved to {csv_path}")


def parse_args():
    parser = ArgumentParser(description="Baseline RDEIC inference with metrics")
    
    parser.add_argument("--ckpt_sd", default="./weight/v2-1_512-ema-pruned.ckpt", type=str)
    parser.add_argument("--ckpt_cc", default="./weight/rdeic_2_step2.ckpt", type=str)
    parser.add_argument("--config", default="configs/model/rdeic.yaml", type=str)
    
    parser.add_argument("--input", type=str, default="inputs/",
                        help="Input image or directory")
    parser.add_argument("--output", type=str, default="outputs/baseline/",
                        help="Output directory")
    
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=2)
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    # High-resolution image handling
    parser.add_argument("--max_long_side", type=int, default=0,
                        help="downscale so that max(H,W) <= this (keep aspect), 0 disables")
    parser.add_argument("--upsample_to_original", action="store_true",
                        help="upsample outputs back to the original input resolution before saving")
    parser.add_argument("--upsample_method", type=str, default="lanczos",
                        choices=["lanczos", "bicubic", "bilinear", "nearest"],
                        help="resampling method used for upsampling")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

