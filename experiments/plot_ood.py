"""
Plotting script for OOD domain generalization experiment results.

Generates:
- Bar charts comparing metrics across domains
- Qualitative grids of input-output pairs
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_domain_comparison_bars(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create bar charts comparing metrics across domains.
    """
    metrics = ["psnr", "ssim", "ms_ssim", "lpips"]
    metric_labels = {
        "psnr": "PSNR (dB) ↑",
        "ssim": "SSIM ↑",
        "ms_ssim": "MS-SSIM ↑",
        "lpips": "LPIPS ↓"
    }
    
    # Aggregate by domain
    grouped = df.groupby("domain").agg({
        m: ["mean", "std"] for m in metrics
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ["domain"] + [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(grouped))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        domains = grouped["domain"].values
        means = grouped[f"{metric}_mean"].values
        stds = grouped[f"{metric}_std"].values
        
        bars = ax.bar(domains, means, yerr=stds, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel("Domain", fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f"{metric_labels[metric]}", fontsize=14)
        
        # Rotate x labels if needed
        if len(domains) > 4:
            ax.set_xticklabels(domains, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("OOD Domain Generalization Results", fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "ood_domain_bars.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'ood_domain_bars.png')}")


def plot_radar_chart(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create a radar chart comparing domains across all metrics.
    """
    metrics = ["psnr", "ssim", "ms_ssim", "lpips"]
    
    # Aggregate by domain
    grouped = df.groupby("domain")[metrics].mean()
    
    # Normalize metrics to [0, 1], flip LPIPS so higher is better
    normalized = grouped.copy()
    for m in metrics:
        if m == "lpips":
            normalized[m] = 1 - (grouped[m] - grouped[m].min()) / (grouped[m].max() - grouped[m].min() + 1e-8)
        else:
            normalized[m] = (grouped[m] - grouped[m].min()) / (grouped[m].max() - grouped[m].min() + 1e-8)
    
    # Radar chart
    categories = ["PSNR", "SSIM", "MS-SSIM", "1-LPIPS"]
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = sns.color_palette("husl", len(normalized))
    
    for (domain, row), color in zip(normalized.iterrows(), colors):
        values = row.values.tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=domain, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title("OOD Domain Quality Profile", fontsize=16, y=1.08)
    
    plt.savefig(os.path.join(output_dir, "ood_radar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'ood_radar.png')}")


def plot_violin(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create violin plots showing distribution of metrics per domain.
    """
    metrics = ["psnr", "ms_ssim", "lpips"]
    metric_labels = {
        "psnr": "PSNR (dB)",
        "ms_ssim": "MS-SSIM",
        "lpips": "LPIPS"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        sns.violinplot(data=df, x="domain", y=metric, ax=ax, 
                      palette="husl", inner="box")
        
        ax.set_xlabel("Domain", fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f"{metric_labels[metric]} Distribution", fontsize=14)
        
        # Rotate x labels if needed
        if df["domain"].nunique() > 4:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "ood_violin.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'ood_violin.png')}")


def create_qualitative_grid(
    input_dir: str,
    recon_dir: str,
    output_path: str,
    domain: str,
    num_samples: int = 6,
    image_size: tuple = (256, 256)
) -> None:
    """
    Create a side-by-side grid of input and reconstructed images.
    
    Args:
        input_dir: Directory containing original images.
        recon_dir: Directory containing reconstructed images.
        output_path: Path to save the grid.
        domain: Domain name for title.
        num_samples: Number of samples to show.
        image_size: Size to resize images to.
    """
    from glob import glob
    
    # Get list of reconstructions
    recon_files = sorted(glob(os.path.join(recon_dir, "*.png")))[:num_samples]
    
    if len(recon_files) == 0:
        print(f"Warning: No reconstructions found in {recon_dir}")
        return
    
    fig, axes = plt.subplots(len(recon_files), 2, figsize=(8, 3 * len(recon_files)))
    
    if len(recon_files) == 1:
        axes = axes.reshape(1, 2)
    
    for idx, recon_path in enumerate(recon_files):
        img_name = Path(recon_path).stem
        
        # Find corresponding input
        input_candidates = [
            os.path.join(input_dir, f"{img_name}.png"),
            os.path.join(input_dir, f"{img_name}.jpg"),
            os.path.join(input_dir, f"{img_name}.jpeg"),
        ]
        
        input_path = None
        for candidate in input_candidates:
            if os.path.exists(candidate):
                input_path = candidate
                break
        
        # Load and resize images
        recon_img = Image.open(recon_path).convert("RGB")
        recon_img = recon_img.resize(image_size, Image.LANCZOS)
        
        if input_path:
            input_img = Image.open(input_path).convert("RGB")
            input_img = input_img.resize(image_size, Image.LANCZOS)
        else:
            input_img = Image.new("RGB", image_size, color=(128, 128, 128))
        
        axes[idx, 0].imshow(np.array(input_img))
        axes[idx, 0].set_title("Input" if idx == 0 else "")
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(np.array(recon_img))
        axes[idx, 1].set_title("Reconstruction" if idx == 0 else "")
        axes[idx, 1].axis("off")
    
    plt.suptitle(f"Domain: {domain}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create a summary table with metrics per domain.
    """
    metrics = ["psnr", "ssim", "ms_ssim", "lpips", "bpp"]
    
    summary = df.groupby("domain").agg({
        m: ["mean", "std", "median"] for m in metrics if m in df.columns
    }).round(4)
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, "ood_summary_table.csv"))
    print(f"Saved: {os.path.join(output_dir, 'ood_summary_table.csv')}")
    
    # Create a formatted table
    with open(os.path.join(output_dir, "ood_summary.txt"), "w") as f:
        f.write("OOD Domain Generalization Summary\n")
        f.write("=" * 60 + "\n\n")
        
        for domain in df["domain"].unique():
            domain_df = df[df["domain"] == domain]
            f.write(f"Domain: {domain}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Samples: {len(domain_df)}\n")
            f.write(f"  PSNR:    {domain_df['psnr'].mean():.2f} ± {domain_df['psnr'].std():.2f} dB\n")
            f.write(f"  SSIM:    {domain_df['ssim'].mean():.4f} ± {domain_df['ssim'].std():.4f}\n")
            f.write(f"  MS-SSIM: {domain_df['ms_ssim'].mean():.4f} ± {domain_df['ms_ssim'].std():.4f}\n")
            f.write(f"  LPIPS:   {domain_df['lpips'].mean():.4f} ± {domain_df['lpips'].std():.4f}\n")
            if "bpp" in domain_df.columns:
                f.write(f"  BPP:     {domain_df['bpp'].mean():.4f} ± {domain_df['bpp'].std():.4f}\n")
            f.write("\n")
    
    print(f"Saved: {os.path.join(output_dir, 'ood_summary.txt')}")


def plot_bpp_vs_quality(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot rate-distortion curve by domain.
    """
    if "bpp" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", df["domain"].nunique())
    
    for (domain, group), color in zip(df.groupby("domain"), colors):
        ax.scatter(group["bpp"], group["psnr"], 
                  label=domain, color=color, alpha=0.7, s=50)
    
    ax.set_xlabel("BPP (bits per pixel)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion by Domain", fontsize=14)
    ax.legend(title="Domain")
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "ood_rate_distortion.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'ood_rate_distortion.png')}")


def main(args: Namespace) -> None:
    """Main plotting routine."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")
    print(f"Domains: {df['domain'].unique()}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Bar charts
    plot_domain_comparison_bars(df, args.output_dir)
    
    # Radar chart
    if df["domain"].nunique() >= 3:
        plot_radar_chart(df, args.output_dir)
    
    # Violin plots
    plot_violin(df, args.output_dir)
    
    # Rate-distortion
    plot_bpp_vs_quality(df, args.output_dir)
    
    # Summary table
    create_summary_table(df, args.output_dir)
    
    # Qualitative grids
    if args.input_dirs and args.recon_dirs:
        input_dirs = args.input_dirs.split(",")
        recon_dirs = args.recon_dirs.split(",")
        
        for input_dir, recon_dir in zip(input_dirs, recon_dirs):
            domain = Path(input_dir).name
            grid_path = os.path.join(args.output_dir, f"ood_grid_{domain}.png")
            create_qualitative_grid(input_dir, recon_dir, grid_path, domain,
                                   num_samples=args.num_grid_samples)
    
    print("\nDone!")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Plot OOD experiment results")
    
    parser.add_argument("--input_csv", type=str, default="indicators/ood_results_all.csv",
                        help="Path to OOD results CSV")
    parser.add_argument("--output_dir", type=str, default="indicators/",
                        help="Directory to save plots")
    parser.add_argument("--input_dirs", type=str, default=None,
                        help="Comma-separated list of input directories for qualitative grids")
    parser.add_argument("--recon_dirs", type=str, default=None,
                        help="Comma-separated list of reconstruction directories for qualitative grids")
    parser.add_argument("--num_grid_samples", type=int, default=6,
                        help="Number of samples per domain in qualitative grids")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

