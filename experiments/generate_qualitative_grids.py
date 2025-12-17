"""
Generate qualitative comparison grids for paper figures.
Creates side-by-side visualizations of:
- Input vs Reconstruction (for OOD experiments)
- Clean vs Corrupted reconstructions (for robustness experiments)
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_image(path: str, size: Tuple[int, int] = None) -> np.ndarray:
    """Load and optionally resize an image."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def add_label(img: np.ndarray, label: str, position: str = "top") -> np.ndarray:
    """Add a text label to an image."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    if position == "top":
        x = (img.shape[1] - text_width) // 2
        y = 5
    else:
        x = (img.shape[1] - text_width) // 2
        y = img.shape[0] - text_height - 10
    
    # Draw background rectangle
    draw.rectangle([x-5, y-2, x + text_width + 5, y + text_height + 2], fill="black")
    draw.text((x, y), label, fill="white", font=font)
    
    return np.array(pil_img)


def create_ood_grid(
    input_dir: str,
    recon_dir: str,
    output_path: str,
    domain_name: str,
    num_samples: int = 6,
    image_size: Tuple[int, int] = (256, 256)
) -> None:
    """
    Create a grid comparing input and reconstructed images for OOD experiments.
    
    Layout: 
    Row 1: Input images
    Row 2: Reconstructed images
    """
    from glob import glob
    
    # Find reconstructions
    recon_files = sorted(glob(os.path.join(recon_dir, "*.png")))[:num_samples]
    
    if len(recon_files) == 0:
        print(f"No reconstructions found in {recon_dir}")
        return
    
    n = len(recon_files)
    
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for idx, recon_path in enumerate(recon_files):
        img_name = Path(recon_path).stem
        
        # Find corresponding input
        input_candidates = [
            os.path.join(input_dir, f"{img_name}.png"),
            os.path.join(input_dir, f"{img_name}.jpg"),
            os.path.join(input_dir, f"{img_name}.jpeg"),
            os.path.join(input_dir, f"{img_name}.PNG"),
            os.path.join(input_dir, f"{img_name}.JPG"),
        ]
        
        input_path = None
        for candidate in input_candidates:
            if os.path.exists(candidate):
                input_path = candidate
                break
        
        # Load images
        recon_img = load_image(recon_path, image_size)
        
        if input_path:
            input_img = load_image(input_path, image_size)
        else:
            input_img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 128
        
        # Plot
        axes[0, idx].imshow(input_img)
        axes[0, idx].axis("off")
        if idx == 0:
            axes[0, idx].set_ylabel("Input", fontsize=14, fontweight='bold')
        
        axes[1, idx].imshow(recon_img)
        axes[1, idx].axis("off")
        if idx == 0:
            axes[1, idx].set_ylabel("RDEIC", fontsize=14, fontweight='bold')
    
    plt.suptitle(f"OOD Domain: {domain_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_robustness_grid(
    original_path: str,
    recon_dirs: List[str],
    error_rates: List[float],
    output_path: str,
    image_size: Tuple[int, int] = (256, 256)
) -> None:
    """
    Create a grid showing degradation at different error rates.
    
    Layout: Original | 0% | 0.1% | 0.5% | 1% | 2% | 5% | 10%
    """
    n = len(error_rates) + 1
    
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    
    # Original
    orig_img = load_image(original_path, image_size)
    axes[0].imshow(orig_img)
    axes[0].set_title("Original", fontsize=11, fontweight='bold')
    axes[0].axis("off")
    
    img_name = Path(original_path).stem
    
    for idx, (recon_dir, rate) in enumerate(zip(recon_dirs, error_rates)):
        # Find reconstruction for this rate
        rate_dir = os.path.join(recon_dir, img_name, f"r_{rate}")
        
        recon_path = None
        if os.path.isdir(rate_dir):
            # Take first seed
            for f in os.listdir(rate_dir):
                if f.endswith(".png"):
                    recon_path = os.path.join(rate_dir, f)
                    break
        
        if recon_path and os.path.exists(recon_path):
            recon_img = load_image(recon_path, image_size)
        else:
            # Show red X for decode failure
            recon_img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 50
            # Add X
            for i in range(min(image_size)):
                if i < image_size[1] and i < image_size[0]:
                    recon_img[i, i] = [255, 0, 0]
                    recon_img[i, image_size[0] - 1 - i] = [255, 0, 0]
        
        axes[idx + 1].imshow(recon_img)
        axes[idx + 1].set_title(f"{rate}%", fontsize=11)
        axes[idx + 1].axis("off")
    
    plt.suptitle(f"RDEIC Robustness: {img_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_grid(
    original_path: str,
    rdeic_recon_path: str,
    jpeg2k_recon_path: str,
    error_rate: float,
    output_path: str,
    image_size: Tuple[int, int] = (256, 256)
) -> None:
    """
    Create a grid comparing RDEIC vs JPEG2000 at the same error rate.
    
    Layout: Original | RDEIC (rate%) | JPEG2000 (rate%)
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    
    # Original
    orig_img = load_image(original_path, image_size)
    axes[0].imshow(orig_img)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # RDEIC
    if os.path.exists(rdeic_recon_path):
        rdeic_img = load_image(rdeic_recon_path, image_size)
    else:
        rdeic_img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 50
    axes[1].imshow(rdeic_img)
    axes[1].set_title(f"RDEIC @ {error_rate}%", fontsize=12, fontweight='bold')
    axes[1].axis("off")
    
    # JPEG2000
    if os.path.exists(jpeg2k_recon_path):
        jpeg2k_img = load_image(jpeg2k_recon_path, image_size)
    else:
        jpeg2k_img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 50
    axes[2].imshow(jpeg2k_img)
    axes[2].set_title(f"JPEG2000 @ {error_rate}%", fontsize=12, fontweight='bold')
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_multi_domain_grid(
    domains: List[Tuple[str, str, str]],  # (name, input_dir, recon_dir)
    output_path: str,
    samples_per_domain: int = 3,
    image_size: Tuple[int, int] = (192, 192)
) -> None:
    """
    Create a grid showing multiple OOD domains.
    
    Layout:
           | Sample1 | Sample2 | Sample3 |
    Domain1| in/out  | in/out  | in/out  |
    Domain2| in/out  | in/out  | in/out  |
    ...
    """
    from glob import glob
    
    n_domains = len(domains)
    n_samples = samples_per_domain
    
    fig = plt.figure(figsize=(3 * n_samples, 2.5 * n_domains))
    gs = gridspec.GridSpec(n_domains, n_samples, hspace=0.3, wspace=0.05)
    
    for d_idx, (domain_name, input_dir, recon_dir) in enumerate(domains):
        recon_files = sorted(glob(os.path.join(recon_dir, "*.png")))[:n_samples]
        
        for s_idx, recon_path in enumerate(recon_files):
            img_name = Path(recon_path).stem
            
            # Find input
            input_candidates = [
                os.path.join(input_dir, f"{img_name}.png"),
                os.path.join(input_dir, f"{img_name}.jpg"),
                os.path.join(input_dir, f"{img_name}.jpeg"),
            ]
            input_path = next((p for p in input_candidates if os.path.exists(p)), None)
            
            # Create subplot with 2 rows (input/recon)
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[d_idx, s_idx], hspace=0.02)
            
            ax_in = fig.add_subplot(inner_gs[0])
            ax_out = fig.add_subplot(inner_gs[1])
            
            # Load and show images
            if input_path:
                input_img = load_image(input_path, image_size)
            else:
                input_img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 128
            recon_img = load_image(recon_path, image_size)
            
            ax_in.imshow(input_img)
            ax_in.axis("off")
            
            ax_out.imshow(recon_img)
            ax_out.axis("off")
            
            # Add domain label on first column
            if s_idx == 0:
                ax_in.set_ylabel(domain_name, fontsize=12, fontweight='bold', rotation=0, 
                                ha='right', va='center', labelpad=40)
    
    plt.suptitle("RDEIC Domain Generalization", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main(args: Namespace):
    """Main function."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "ood":
        # Generate OOD comparison grids
        domains = args.domains.split(",")
        for domain_path in domains:
            domain_name = Path(domain_path).name
            recon_dir = os.path.join(args.recon_base, domain_name)
            
            if os.path.isdir(recon_dir):
                output_path = os.path.join(args.output_dir, f"ood_grid_{domain_name}.png")
                create_ood_grid(domain_path, recon_dir, output_path, domain_name,
                               num_samples=args.num_samples, image_size=(args.image_size, args.image_size))
            else:
                print(f"Reconstruction dir not found: {recon_dir}")
    
    elif args.mode == "robustness":
        # Generate robustness degradation grids
        from glob import glob
        
        input_files = list(glob(os.path.join(args.input_dir, "*.png"))) + \
                     list(glob(os.path.join(args.input_dir, "*.jpg")))
        
        error_rates = [float(r) for r in args.rates.split(",")]
        
        for input_path in input_files[:args.num_samples]:
            img_name = Path(input_path).stem
            output_path = os.path.join(args.output_dir, f"robustness_grid_{img_name}.png")
            
            recon_dirs = [args.recon_base] * len(error_rates)
            create_robustness_grid(input_path, recon_dirs, error_rates, output_path,
                                  image_size=(args.image_size, args.image_size))
    
    elif args.mode == "comparison":
        # Generate RDEIC vs JPEG2000 comparison
        from glob import glob
        
        input_files = list(glob(os.path.join(args.input_dir, "*.png"))) + \
                     list(glob(os.path.join(args.input_dir, "*.jpg")))
        
        for input_path in input_files[:args.num_samples]:
            img_name = Path(input_path).stem
            
            for rate in args.rates.split(","):
                rate_f = float(rate)
                
                rdeic_path = os.path.join(args.rdeic_recon, img_name, f"r_{rate_f}", "seed_231", "recon.png")
                jpeg2k_path = os.path.join(args.jpeg2k_recon, img_name, f"r_{rate_f}", "seed_231.png")
                
                output_path = os.path.join(args.output_dir, f"comparison_{img_name}_{rate}pct.png")
                create_comparison_grid(input_path, rdeic_path, jpeg2k_path, rate_f, output_path,
                                      image_size=(args.image_size, args.image_size))
    
    print("Done!")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate qualitative comparison grids")
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["ood", "robustness", "comparison"],
                        help="Type of grid to generate")
    
    # Common arguments
    parser.add_argument("--output_dir", type=str, default="indicators/grids/",
                        help="Output directory for grids")
    parser.add_argument("--num_samples", type=int, default=6,
                        help="Number of samples to show")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size to resize images to")
    
    # OOD mode arguments
    parser.add_argument("--domains", type=str, default="",
                        help="Comma-separated domain paths for OOD mode")
    parser.add_argument("--recon_base", type=str, default="outputs/ood/",
                        help="Base directory containing reconstructions")
    
    # Robustness mode arguments
    parser.add_argument("--input_dir", type=str, default="inputs/",
                        help="Directory with original images")
    parser.add_argument("--rates", type=str, default="0.0,0.1,0.5,1.0,2.0,5.0,10.0",
                        help="Error rates to show")
    
    # Comparison mode arguments
    parser.add_argument("--rdeic_recon", type=str, default="outputs/robustness/",
                        help="RDEIC reconstruction directory")
    parser.add_argument("--jpeg2k_recon", type=str, default="outputs/jpeg2000_robustness/",
                        help="JPEG2000 reconstruction directory")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

