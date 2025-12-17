"""
Create overlapping plots for bitstream random and burst errors.

This script loads CSV data and creates overlapping comparison plots
similar to RDEIC vs JPEG2000 comparison.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')


def load_bitstream_data(burst_csv: str, random_csv: str = None, 
                       results_csv: str = None) -> tuple:
    """
    Load bitstream burst and random error data.
    
    Args:
        burst_csv: Path to burst error CSV
        random_csv: Optional path to random error CSV
        results_csv: Optional path to combined results CSV (will extract random from here)
    
    Returns:
        (df_burst, df_random) DataFrames
    """
    # Load burst data
    df_burst = pd.read_csv(burst_csv)
    if "decode_failed" in df_burst.columns:
        df_burst = df_burst[df_burst["decode_failed"] == False]
    print(f"Loaded {len(df_burst)} burst error rows from {burst_csv}")
    
    # Load random data
    df_random = None
    if random_csv and os.path.exists(random_csv):
        df_random = pd.read_csv(random_csv)
        if "decode_failed" in df_random.columns:
            df_random = df_random[df_random["decode_failed"] == False]
        print(f"Loaded {len(df_random)} random error rows from {random_csv}")
    elif results_csv and os.path.exists(results_csv):
        # Extract random bitstream errors from combined results
        df_all = pd.read_csv(results_csv)
        df_random = df_all[
            (df_all["error_space"] == "bitstream") & 
            (df_all["error_type"] == "random")
        ].copy()
        if "decode_failed" in df_random.columns:
            df_random = df_random[df_random["decode_failed"] == False]
        print(f"Extracted {len(df_random)} random error rows from {results_csv}")
    
    if df_random is None or len(df_random) == 0:
        raise ValueError("Could not find random error data. Please provide random_csv or results_csv with random bitstream errors.")
    
    return df_burst, df_random


def plot_overlapping_curves(df_burst: pd.DataFrame, df_random: pd.DataFrame,
                           output_path: str, title: str = "Bitstream Error Comparison") -> None:
    """
    Create overlapping plots for all metrics (PSNR, SSIM, MS-SSIM, LPIPS).
    
    Args:
        df_burst: Burst error DataFrame
        df_random: Random error DataFrame
        output_path: Path to save the plot
        title: Plot title
    """
    metrics = ["psnr", "ssim", "ms_ssim", "lpips"]
    metric_labels = {
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "ms_ssim": "MS-SSIM",
        "lpips": "LPIPS"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot burst errors
        grouped_burst = df_burst.groupby("error_rate").agg({
            metric: ["mean", "std"]
        }).reset_index()
        grouped_burst.columns = ["error_rate", "mean", "std"]
        
        ax.errorbar(grouped_burst["error_rate"], grouped_burst["mean"], 
                   yerr=grouped_burst["std"],
                   marker='o', capsize=3, capthick=1.5, linewidth=2, markersize=6,
                   color='#e74c3c', label='Burst Errors', alpha=0.8)
        
        # Plot random errors
        grouped_random = df_random.groupby("error_rate").agg({
            metric: ["mean", "std"]
        }).reset_index()
        grouped_random.columns = ["error_rate", "mean", "std"]
        
        ax.errorbar(grouped_random["error_rate"], grouped_random["mean"], 
                   yerr=grouped_random["std"],
                   marker='s', capsize=3, capthick=1.5, linewidth=2, markersize=6,
                   color='#3498db', label='Random Errors', alpha=0.8)
        
        ax.set_xlabel("Error Rate (%)", fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f"{metric_labels[metric]}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overlapping curves: {output_path}")


def plot_overlapping_psnr(df_burst: pd.DataFrame, df_random: pd.DataFrame,
                          output_path: str, title: str = "PSNR: Random vs Burst Errors") -> None:
    """
    Create a single PSNR plot with overlapping curves.
    
    Args:
        df_burst: Burst error DataFrame
        df_random: Random error DataFrame
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot burst errors
    grouped_burst = df_burst.groupby("error_rate").agg({
        "psnr": ["mean", "std"]
    }).reset_index()
    grouped_burst.columns = ["error_rate", "mean", "std"]
    
    ax.errorbar(grouped_burst["error_rate"], grouped_burst["mean"], 
               yerr=grouped_burst["std"],
               marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
               color='#e74c3c', label='Burst Errors', alpha=0.8, ecolor='#c0392b')
    
    # Plot random errors
    grouped_random = df_random.groupby("error_rate").agg({
        "psnr": ["mean", "std"]
    }).reset_index()
    grouped_random.columns = ["error_rate", "mean", "std"]
    
    ax.errorbar(grouped_random["error_rate"], grouped_random["mean"], 
               yerr=grouped_random["std"],
               marker='s', capsize=5, capthick=2, linewidth=2, markersize=8,
               color='#3498db', label='Random Errors', alpha=0.8, ecolor='#2980b9')
    
    # Add threshold line
    ax.axhline(y=25, color='gray', linestyle='--', linewidth=1.5, 
              alpha=0.7, label='Threshold (25 dB)')
    
    ax.set_xlabel("Error Rate (%)", fontsize=14)
    ax.set_ylabel("PSNR (dB)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overlapping PSNR plot: {output_path}")


def main():
    """Main function to create overlapping plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create overlapping plots for bitstream errors")
    parser.add_argument("--burst_csv", type=str, 
                       default="indicators/robustness_bitstream_burst.csv",
                       help="Path to burst error CSV")
    parser.add_argument("--random_csv", type=str, default=None,
                       help="Path to random error CSV (optional)")
    parser.add_argument("--results_csv", type=str, 
                       default="indicators/robustness_results.csv",
                       help="Path to combined results CSV (to extract random errors)")
    parser.add_argument("--output_dir", type=str, default="indicators/",
                       help="Output directory for plots")
    parser.add_argument("--psnr_only", action="store_true",
                       help="Only create PSNR comparison plot")
    
    args = parser.parse_args()
    
    # Load data
    df_burst, df_random = load_bitstream_data(
        args.burst_csv, args.random_csv, args.results_csv
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create plots
    if args.psnr_only:
        output_path = os.path.join(args.output_dir, "bitstream_overlapping_psnr.png")
        plot_overlapping_psnr(df_burst, df_random, output_path)
    else:
        # Create full comparison (all metrics)
        output_path = os.path.join(args.output_dir, "bitstream_overlapping_curves.png")
        plot_overlapping_curves(df_burst, df_random, output_path)
        
        # Also create PSNR-only plot
        output_path_psnr = os.path.join(args.output_dir, "bitstream_overlapping_psnr.png")
        plot_overlapping_psnr(df_burst, df_random, output_path_psnr)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
