from typing import List, Tuple, Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace
import time
import warnings
import logging

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from transformers import BlipForConditionalGeneration, BlipProcessor

from ldm.xformers_state import disable_xformers
from model.spaced_sampler_relay import SpacedSampler
from model.ddim_sampler_relay import DDIMSampler
from model.rdeic import RDEIC
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts

_caption_cache = {
    "model_id": None,
    "device": None,
    "model": None,
    "processor": None,
}


def load_captioner(model_id: str, device: str):
    """
    Load (and cache) a BLIP captioning model on the requested device.
    """
    if (
        _caption_cache["model_id"] != model_id
        or _caption_cache["device"] != device
        or _caption_cache["model"] is None
        or _caption_cache["processor"] is None
    ):
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()
        _caption_cache.update(
            {"model_id": model_id, "device": device, "model": model, "processor": processor}
        )
    return _caption_cache["model"], _caption_cache["processor"]


@torch.no_grad()
def generate_caption(image: Image.Image, model_id: str, device: str) -> str:
    """
    Run BLIP on a single image and return a decoded caption string.
    """
    blip_model, blip_processor = load_captioner(model_id, device)
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=64)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption.strip()


@torch.no_grad()
def process(
    model: RDEIC,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_paths: List[str],
    guidance_scale: float,
    c_crossattn: List[torch.Tensor],
    uc_crossattn: Optional[List[torch.Tensor]] = None,
    micro_batch_size: Optional[int] = None,
    use_fp16: bool = False,
    profile_memory: bool = False,
    save_intermediates: bool = False,
    intermediate_prefixes: Optional[List[str]] = None,
    latent_format: str = "pt",
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Apply RDEIC model on a list of images with CFG over text prompts.
    
    Args:
        model (RDEIC): Model.
        imgs (List[np.ndarray]): A list of images (HWC, RGB, range in [0, 255])
        sampler (str): Sampler name.
        steps (int): Sampling steps.
        stream_paths (List[str]): List of savedirs for bitstreams, one per image
        c_crossattn (List[Tensor]): Conditional text embeddings (e.g., BLIP captions).
        uc_crossattn (List[Tensor], optional): Unconditional text embeddings (e.g., empty prompt).
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        bpps (List[float]): Bits-per-pixel for each image
    """
    n_samples = len(imgs)
    assert n_samples == len(stream_paths), "imgs and stream_paths must have the same length"
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    # Keep control on CPU to reduce baseline VRAM; move per-sample slices to GPU as needed
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    height, width = control.size(-2), control.size(-1)
    # Per-sample compression/decompression to allow individual bitstreams
    bpps: List[float] = []
    c_latents: List[torch.Tensor] = []
    guide_hints: List[torch.Tensor] = []
    if profile_memory and torch.cuda.is_available() and model.device.type == "cuda":
        print(f"[mem] before compress: alloc={torch.cuda.memory_allocated() / 1e9:.3f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB")
    for i in range(n_samples):
        control_i = control[i : i + 1].to(model.device, non_blocking=True)
        bpp_i = model.apply_condition_compress(control_i, stream_paths[i], height, width)
        c_latent_i, guide_hint_i = model.apply_condition_decompress(stream_paths[i])
        bpps.append(bpp_i)
        c_latents.append(c_latent_i)
        guide_hints.append(guide_hint_i)
        if save_intermediates:
            assert intermediate_prefixes is not None and len(intermediate_prefixes) == n_samples, "intermediate_prefixes must match batch size"
            prefix = intermediate_prefixes[i]
            # Save latent
            latent_to_save = c_latent_i.detach().cpu()
            if latent_format == "npy":
                np.save(f"{prefix}_latent.npy", latent_to_save.numpy())
            else:
                torch.save(latent_to_save, f"{prefix}_latent.pt")
            # Save a visualization of the guide hint (and a decoded compressed image if possible)
            try:
                gh = guide_hint_i.detach().float().cpu()
                # Normalize per-tensor to 0..1 for visualization
                gh_min = gh.min()
                gh_max = gh.max()
                gh = (gh - gh_min) / (gh_max - gh_min + 1e-8)
                if gh.dim() == 4:
                    # shape (1, C, H, W)
                    gh = gh[0]
                if gh.shape[0] >= 3:
                    gh_vis = gh[:3]
                else:
                    gh_vis = gh[:1].repeat(3, 1, 1)
                gh_vis = einops.rearrange(gh_vis, "c h w -> h w c").numpy()
                gh_vis = (gh_vis * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(gh_vis).save(f"{prefix}_guide.png")
            except Exception:
                pass
            try:
                # Attempt to decode c_latent to an RGB for a "compressed image" preview
                with torch.no_grad():
                    dec = model.decode_first_stage(c_latent_i.to(model.device))
                dec = ((dec + 1) / 2).clamp(0, 1)
                dec_np = (einops.rearrange(dec.detach().cpu(), "b c h w -> b h w c").numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(dec_np[0]).save(f"{prefix}_compressed.png")
            except Exception:
                pass
    c_latent = torch.cat(c_latents, dim=0)
    guide_hint = torch.cat(guide_hints, dim=0)
    cond = {
        "c_latent": [c_latent],
        "c_crossattn": c_crossattn,
        "guide_hint": guide_hint
    }
    
    latent_shape = (n_samples, 4, height // 8, width // 8)
    preds: List[np.ndarray] = []
    # Enable micro-batching for the expensive sampling/decoding stage
    chunk_size = micro_batch_size or n_samples
    if profile_memory and torch.cuda.is_available() and model.device.type == "cuda":
        print(f"[mem] before sampling: alloc={torch.cuda.memory_allocated() / 1e9:.3f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB")
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        bs = end - start
        # Prepare per-chunk conditioning
        c_latent_chunk = c_latent[start:end]
        guide_hint_chunk = guide_hint[start:end]
        # repeat text conditioning to chunk size if needed
        if isinstance(c_crossattn, list) and len(c_crossattn) > 0 and hasattr(c_crossattn[0], "shape"):
            c_crossattn_chunk = [c_crossattn[0][start:end]]
        else:
            c_crossattn_chunk = c_crossattn
        if uc_crossattn is not None:
            if isinstance(uc_crossattn, list) and len(uc_crossattn) > 0 and hasattr(uc_crossattn[0], "shape"):
                uc_crossattn_chunk = [uc_crossattn[0][start:end]]
            else:
                uc_crossattn_chunk = uc_crossattn
        else:
            uc_crossattn_chunk = None
        cond_chunk = {
            "c_latent": [c_latent_chunk],
            "c_crossattn": c_crossattn_chunk,
            "guide_hint": guide_hint_chunk
        }
        uc_chunk = None
        if uc_crossattn_chunk is not None:
            uc_chunk = {
                "c_latent": [c_latent_chunk],
                "c_crossattn": uc_crossattn_chunk,
                "guide_hint": guide_hint_chunk,
            }
        shape = (bs, latent_shape[1], latent_shape[2], latent_shape[3])
        # Create noise only once per chunk
        t = torch.ones((bs,)).long().to(model.device) * model.used_timesteps - 1
        noise = torch.randn(shape, device=model.device, dtype=torch.float32)
        x_T = model.q_sample(x_start=c_latent_chunk, t=t, noise=noise)
        del noise
        if use_fp16 and model.device.type == "cuda":
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            # no-op context manager
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc_val, exc_tb): return False
            amp_ctx = _NullCtx()
        with amp_ctx:
            if isinstance(sampler, SpacedSampler):
                samples_chunk = sampler.sample(
                    steps, shape, cond_chunk,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc_chunk,
                    cond_fn=None, x_T=x_T
                )
            else:
                sampler_ddim: DDIMSampler = sampler
                samples_chunk, _ = sampler_ddim.sample(
                    S=steps, batch_size=shape[0], shape=shape[1:],
                    conditioning=cond_chunk, unconditional_conditioning=uc_chunk,
                    unconditional_guidance_scale=guidance_scale,
                    x_T=x_T, eta=0
                )
            x_samples = model.decode_first_stage(samples_chunk)
        x_samples = ((x_samples + 1) / 2).clamp(0, 1)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        preds.extend([x_samples[i] for i in range(bs)])
        # Free chunk tensors
        del x_T, samples_chunk, x_samples, c_latent_chunk, guide_hint_chunk
        if torch.cuda.is_available() and model.device.type == "cuda":
            torch.cuda.empty_cache()
            if profile_memory:
                print(f"[mem] after chunk {start}-{end}: alloc={torch.cuda.memory_allocated() / 1e9:.3f} GB, reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB")
    
    return preds, bpps


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt_sd", default='./weight/v2-1_512-ema-pruned.ckpt', type=str, help="checkpoint path of stable diffusion")
    parser.add_argument("--ckpt_cc", default='path to checkpoint file of compression and control module', type=str, help="checkpoint path of lfgcm and control module")
    parser.add_argument("--config", default='configs/model/rdeic.yaml', type=str, help="model config path")
    
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=2, type=int)
    parser.add_argument("--guidance_scale", default=1.0, type=float)
    
    parser.add_argument("--output", type=str, default='results/')
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--use_captions", action="store_true", help="enable BLIP captioning + CFG over captions")
    parser.add_argument("--caption_device", type=str, default=None, help="device for BLIP captioning (defaults to --device)")
    parser.add_argument("--caption_model", type=str, default="Salesforce/blip-image-captioning-base", help="BLIP model id for captioning")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images to process together (grouped by resolution)")
    parser.add_argument("--suppress_warnings", action="store_true", help="suppress Python warnings and set logging to ERROR")
    parser.add_argument("--micro_batch_size", type=int, default=0, help="split each batch into smaller chunks during sampling to reduce VRAM")
    parser.add_argument("--fp16", action="store_true", help="use autocast float16 during sampling/decoding (CUDA only)")
    parser.add_argument("--profile_memory", action="store_true", help="print CUDA memory usage before/after key steps")
    parser.add_argument("--save_intermediates", action="store_true", help="save guide hint preview and latent per image")
    parser.add_argument("--latent_format", type=str, default="pt", choices=["pt", "npy"], help="format to save latent tensors")
    parser.add_argument("--max_long_side", type=int, default=0, help="downscale so that max(H,W) <= this (keep aspect), 0 disables")
    parser.add_argument("--enable_resize_guard", action="store_true", help="enable down/upsampling guard for large images")
    parser.add_argument("--upsample_to_original", action="store_true", help="upsample outputs back to the original input resolution before saving")
    parser.add_argument("--upsample_method", type=str, default="lanczos", choices=["lanczos", "bicubic", "bilinear", "nearest"], help="resampling method used for upsampling")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suppress_warnings:
        # Silence Python warnings and lower logging noise
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()

    model: RDEIC = instantiate_from_config(OmegaConf.load(args.config))
    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
    ckpt_lc = torch.load(args.ckpt_cc, map_location="cpu")['state_dict']
    ckpt_sd.update(ckpt_lc)
    load_state_dict(model, ckpt_sd, strict=False)
    # update preprocess model
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)

    bpps = []
    
    assert os.path.isdir(args.input)

    if args.use_captions:
        caption_device = args.caption_device if args.caption_device is not None else args.device
        if caption_device.startswith("cuda") and not torch.cuda.is_available():
            caption_device = "cpu"
    else:
        caption_device = None
    
    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    # Group files by padded resolution so they can be batched safely
    files = sorted(list_image_files(args.input, follow_links=True))
    groups = {}
    image_meta = {}
    for file_path in files:
        img = Image.open(file_path).convert("RGB")
        w, h = img.size
        if args.use_captions:
            caption = generate_caption(img, model_id=args.caption_model, device=caption_device)
        else:
            caption = ""
        # optionally downscale to limit working size while preserving aspect ratio
        if args.enable_resize_guard:
            target_max = int(args.max_long_side) if args.max_long_side and args.max_long_side > 0 else 0
            if target_max and max(w, h) > target_max:
                scale = target_max / max(w, h)
                scaled_w = max(1, int(round(w * scale)))
                scaled_h = max(1, int(round(h * scale)))
            else:
                scale = 1.0
                scaled_w, scaled_h = w, h
        else:
            target_max = 0
            scale = 1.0
            scaled_w, scaled_h = w, h
        # compute padded dims (multiple of 64) after resizing
        pad_h = ((scaled_h + 63) // 64) * 64
        pad_w = ((scaled_w + 63) // 64) * 64
        key = (pad_h, pad_w)
        groups.setdefault(key, []).append(file_path)
        image_meta[file_path] = {
            "orig_h": h, "orig_w": w,                # original file size
            "scaled_h": scaled_h, "scaled_w": scaled_w,  # resized working size
            "scale": scale,
            "caption": caption,
        }
    for (pad_h, pad_w) in sorted(groups.keys()):
        file_paths = sorted(groups[(pad_h, pad_w)])
        bs = max(1, int(args.batch_size))
        for i in range(0, len(file_paths), bs):
            batch_files = file_paths[i:i + bs]
            imgs = []
            save_paths = []
            stream_paths = []
            intermediate_prefixes = []
            orig_sizes = []
            captions = []
            for file_path in batch_files:
                img = Image.open(file_path).convert("RGB")
                meta = image_meta[file_path]
                if img.size != (meta["scaled_w"], meta["scaled_h"]):
                    img = img.resize((meta["scaled_w"], meta["scaled_h"]), Image.LANCZOS)
                x = pad(np.array(img), scale=64)
                imgs.append(x)
                save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
                parent_path, stem, _ = get_file_name_parts(save_path)
                stream_parent_path = os.path.join(parent_path, 'data')
                save_path = os.path.join(parent_path, f"{stem}.png")
                stream_path = os.path.join(stream_parent_path, f"{stem}")
                os.makedirs(parent_path, exist_ok=True)
                os.makedirs(stream_parent_path, exist_ok=True)
                save_paths.append(save_path)
                stream_paths.append(stream_path)
                intermediate_prefixes.append(os.path.join(parent_path, stem))
                # for cropping off padding we want the resized working size
                orig_sizes.append((meta["scaled_h"], meta["scaled_w"]))
                captions.append(meta["caption"])
            if args.use_captions:
                text_cond = model.get_learned_conditioning(captions)
                uc_cond = model.get_learned_conditioning([""] * len(captions))
                c_crossattn = [text_cond]
                uc_crossattn = [uc_cond]
            else:
                c_crossattn = [model.get_learned_conditioning([""] * len(captions))]
                uc_crossattn = None
            start_time = time.time()
            try:
                preds, bpps_batch = process(
                    model, imgs, steps=args.steps, sampler=args.sampler,
                    stream_paths=stream_paths, guidance_scale=args.guidance_scale, c_crossattn=c_crossattn,
                    uc_crossattn=uc_crossattn,
                    micro_batch_size=(args.micro_batch_size if args.micro_batch_size and args.micro_batch_size > 0 else None),
                    use_fp16=bool(args.fp16),
                    profile_memory=bool(args.profile_memory),
                    save_intermediates=bool(args.save_intermediates),
                    intermediate_prefixes=intermediate_prefixes if args.save_intermediates else None,
                    latent_format=str(args.latent_format),
                )
                gen_time = time.time() - start_time
                per_image_time = gen_time / max(1, len(imgs))
                if torch.cuda.is_available() and args.device == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise RuntimeError("CUDA OOM during sampling. Reduce --batch_size or try --micro_batch_size 1 and/or --fp16.") from e
                else:
                    raise
            # choose resample filter for optional upsampling
            if args.upsample_method == "lanczos":
                _resample = Image.LANCZOS
            elif args.upsample_method == "bicubic":
                _resample = Image.BICUBIC
            elif args.upsample_method == "bilinear":
                _resample = Image.BILINEAR
            else:
                _resample = Image.NEAREST
            for pred, (work_h, work_w), file_path, save_path, bpp in zip(preds, orig_sizes, batch_files, save_paths, bpps_batch):
                # remove padding
                pred = pred[:work_h, :work_w, :]
                # optionally upsample back to original file resolution
                meta = image_meta[file_path]
                img_to_save = Image.fromarray(pred)
                if args.enable_resize_guard and args.upsample_to_original and meta["scale"] < 1.0:
                    img_to_save = img_to_save.resize((meta["orig_w"], meta["orig_h"]), _resample)
                img_to_save.save(save_path)
                print(f"save to {save_path}, bpp {bpp:.3f}, time {per_image_time:.2f}s")
                bpps.append(bpp)
            # Aggressively free CPU/GPU memory between batches
            del preds, bpps_batch, imgs, save_paths, stream_paths, intermediate_prefixes, orig_sizes
            if torch.cuda.is_available() and args.device == "cuda":
                torch.cuda.empty_cache()

    avg_bpp = sum(bpps) / len(bpps)
    print(f'avg bpp: {avg_bpp}')
            

if __name__ == "__main__":
    main()
