"""
Plotting script for robustness experiment results.

Generates:
- Metric vs error_rate curves with error bars
- Failure threshold tables
"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_metric_curves(df: pd.DataFrame, output_dir: str, prefix: str = "") -> None:
    """
    Plot metric vs error_rate curves with error bars.
    
    Args:
        df: DataFrame with robustness results.
        output_dir: Directory to save plots.
        prefix: Prefix for output filenames.
    """
    metrics = ["psnr", "ssim", "ms_ssim", "lpips"]
    metric_labels = {
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "ms_ssim": "MS-SSIM",
        "lpips": "LPIPS"
    }
    higher_better = {"psnr": True, "ssim": True, "ms_ssim": True, "lpips": False}
    
    # Group by error_rate and compute stats
    grouped = df.groupby("error_rate").agg({
        m: ["mean", "std"] for m in metrics
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ["error_rate"] + [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        error_rates = grouped["error_rate"].values
        means = grouped[f"{metric}_mean"].values
        stds = grouped[f"{metric}_std"].values
        
        ax.errorbar(error_rates, means, yerr=stds, 
                   marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel("Error Rate (%)", fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f"{metric_labels[metric]} vs Error Rate", fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Mark failure thresholds
        if metric == "ms_ssim":
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Threshold (0.9)')
            ax.legend()
        elif metric == "psnr":
            ax.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Threshold (25 dB)')
            ax.legend()
    
    plt.tight_layout()
    
    filename = f"{prefix}robustness_curves.png" if prefix else "robustness_curves.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename)}")


def plot_single_metric(df: pd.DataFrame, metric: str, output_dir: str, 
                       threshold: float = None, prefix: str = "") -> None:
    """
    Plot a single metric vs error_rate curve.
    """
    metric_labels = {
        "psnr": "PSNR (dB)",
        "ssim": "SSIM",
        "ms_ssim": "MS-SSIM",
        "lpips": "LPIPS"
    }
    
    grouped = df.groupby("error_rate").agg({
        metric: ["mean", "std", "min", "max"]
    }).reset_index()
    grouped.columns = ["error_rate", "mean", "std", "min", "max"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(grouped["error_rate"], grouped["mean"], yerr=grouped["std"],
               marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
               color='#2ecc71', ecolor='#27ae60', label='Mean ± Std')
    
    # Fill between min and max
    ax.fill_between(grouped["error_rate"], grouped["min"], grouped["max"],
                   alpha=0.2, color='#2ecc71', label='Min-Max Range')
    
    if threshold is not None:
        ax.axhline(y=threshold, color='#e74c3c', linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'Threshold ({threshold})')
    
    ax.set_xlabel("Error Rate (%)", fontsize=14)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=14)
    ax.set_title(f"RDEIC Robustness: {metric_labels.get(metric, metric)}", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    filename = f"{prefix}robustness_{metric}.png" if prefix else f"robustness_{metric}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename)}")


def compute_failure_thresholds(df: pd.DataFrame, output_dir: str, prefix: str = "") -> pd.DataFrame:
    """
    Compute failure thresholds for each metric.
    
    Returns a DataFrame with the first error_rate where each metric crosses its threshold.
    """
    thresholds = {
        "psnr": 25.0,  # dB
        "ssim": 0.85,
        "ms_ssim": 0.9,
        "lpips": 0.3  # Higher is worse
    }
    
    higher_better = {"psnr": True, "ssim": True, "ms_ssim": True, "lpips": False}
    
    grouped = df.groupby("error_rate").agg({
        m: "mean" for m in thresholds.keys()
    }).reset_index()
    
    failure_rates = {}
    
    for metric, threshold in thresholds.items():
        if higher_better[metric]:
            # Find first rate where mean drops below threshold
            failed = grouped[grouped[metric] < threshold]
        else:
            # Find first rate where mean exceeds threshold (for LPIPS)
            failed = grouped[grouped[metric] > threshold]
        
        if len(failed) > 0:
            failure_rates[metric] = {
                "threshold": threshold,
                "failure_rate": failed["error_rate"].iloc[0],
                "metric_at_failure": failed[metric].iloc[0]
            }
        else:
            failure_rates[metric] = {
                "threshold": threshold,
                "failure_rate": ">10%",
                "metric_at_failure": grouped[metric].iloc[-1]
            }
    
    # Create DataFrame
    failure_df = pd.DataFrame(failure_rates).T
    failure_df.index.name = "metric"
    failure_df = failure_df.reset_index()
    
    # Save to CSV
    filename = f"{prefix}failure_thresholds.csv" if prefix else "failure_thresholds.csv"
    failure_df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Saved: {os.path.join(output_dir, filename)}")
    
    # Also create a nice text summary
    txt_filename = f"{prefix}failure_thresholds.txt" if prefix else "failure_thresholds.txt"
    with open(os.path.join(output_dir, txt_filename), "w") as f:
        f.write("RDEIC Robustness Failure Thresholds\n")
        f.write("=" * 50 + "\n\n")
        for _, row in failure_df.iterrows():
            f.write(f"{row['metric'].upper()}:\n")
            f.write(f"  Threshold: {row['threshold']}\n")
            f.write(f"  Failure at: {row['failure_rate']}% error rate\n")
            f.write(f"  Value at failure: {row['metric_at_failure']:.4f}\n\n")
    
    print(f"Saved: {os.path.join(output_dir, txt_filename)}")
    
    return failure_df


def plot_heatmap(df: pd.DataFrame, output_dir: str, prefix: str = "") -> None:
    """
    Create a heatmap of metrics across error rates.
    """
    metrics = ["psnr", "ssim", "ms_ssim", "lpips"]
    
    # Pivot table
    pivot = df.groupby("error_rate")[metrics].mean()
    
    # Normalize each metric to [0, 1] for visualization
    pivot_norm = pivot.copy()
    for m in metrics:
        if m == "lpips":
            # Invert LPIPS so higher is better
            pivot_norm[m] = 1 - (pivot[m] - pivot[m].min()) / (pivot[m].max() - pivot[m].min() + 1e-8)
        else:
            pivot_norm[m] = (pivot[m] - pivot[m].min()) / (pivot[m].max() - pivot[m].min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(pivot_norm.T, annot=pivot.T.round(3), fmt='', 
                cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Quality'})
    
    ax.set_xlabel("Error Rate (%)", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_title("RDEIC Quality Degradation Heatmap", fontsize=14)
    
    filename = f"{prefix}robustness_heatmap.png" if prefix else "robustness_heatmap.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename)}")


def plot_comparison(df1: pd.DataFrame, df2: pd.DataFrame, 
                   label1: str, label2: str,
                   output_dir: str, prefix: str = "") -> None:
    """
    Plot comparison between two experiments (e.g., RDEIC vs JPEG2000).
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
        
        for df, label, color in [(df1, label1, '#3498db'), (df2, label2, '#e74c3c')]:
            grouped = df.groupby("error_rate").agg({
                metric: ["mean", "std"]
            }).reset_index()
            grouped.columns = ["error_rate", "mean", "std"]
            
            ax.errorbar(grouped["error_rate"], grouped["mean"], yerr=grouped["std"],
                       marker='o', capsize=3, capthick=1.5, linewidth=2, markersize=6,
                       color=color, label=label)
        
        ax.set_xlabel("Error Rate (%)", fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f"{metric_labels[metric]}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Robustness Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    
    filename = f"{prefix}comparison_curves.png" if prefix else "comparison_curves.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename)}")


def main(args: Namespace) -> None:
    """Main plotting routine."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")
    
    # Filter out failed decodes if present
    if "decode_failed" in df.columns:
        failed_count = df["decode_failed"].sum()
        if failed_count > 0:
            print(f"Warning: {failed_count} decode failures in data")
        df = df[df["decode_failed"] == False]
    
    # Check if this is bitstream data and if we should create comparison plots
    is_bitstream = "error_space" in df.columns and "bitstream" in df["error_space"].values
    has_multiple_error_types = "error_type" in df.columns and df["error_type"].nunique() > 1
    
    # Generate prefix based on error type
    prefix = ""
    if "error_type" in df.columns and df["error_type"].nunique() == 1:
        prefix = f"{df['error_type'].iloc[0]}_"
    if "error_space" in df.columns and df["error_space"].nunique() == 1:
        prefix = f"{df['error_space'].iloc[0]}_{prefix}"
    
    # For bitstream errors, try to find the complementary error type CSV
    df_random = None
    df_burst = None
    
    if is_bitstream:
        # Check if current CSV has both error types
        if has_multiple_error_types and "random" in df["error_type"].values and "burst" in df["error_type"].values:
            df_random = df[df["error_type"] == "random"].copy()
            df_burst = df[df["error_type"] == "burst"].copy()
            print(f"Found both random and burst error types in the same CSV")
        else:
            # Try to find the complementary CSV file
            input_dir = os.path.dirname(args.input_csv) if os.path.dirname(args.input_csv) else "."
            input_basename = os.path.basename(args.input_csv)
            
            # Determine current error type
            current_error_type = None
            if "error_type" in df.columns and df["error_type"].nunique() == 1:
                current_error_type = df["error_type"].iloc[0]
            
            # Look for complementary CSV with multiple naming patterns
            if current_error_type == "random":
                # Look for burst CSV with various naming patterns
                possible_burst_files = [
                    os.path.join(input_dir, input_basename.replace("random", "burst")),
                    os.path.join(input_dir, input_basename.replace("_random", "_burst")),
                    os.path.join(input_dir, "robustness_bitstream_burst.csv"),
                    os.path.join(input_dir, "bitstream_burst_robustness.csv"),
                ]
                for burst_csv in possible_burst_files:
                    if os.path.exists(burst_csv):
                        df_burst = pd.read_csv(burst_csv)
                        if "decode_failed" in df_burst.columns:
                            df_burst = df_burst[df_burst["decode_failed"] == False]
                        df_random = df.copy()
                        print(f"Found complementary burst CSV: {burst_csv}")
                        break
            
            elif current_error_type == "burst":
                # Look for random CSV with various naming patterns
                possible_random_files = [
                    os.path.join(input_dir, input_basename.replace("burst", "random")),
                    os.path.join(input_dir, input_basename.replace("_burst", "_random")),
                    os.path.join(input_dir, "robustness_bitstream_random.csv"),
                    os.path.join(input_dir, "bitstream_random_robustness.csv"),
                ]
                for random_csv in possible_random_files:
                    if os.path.exists(random_csv):
                        df_random = pd.read_csv(random_csv)
                        if "decode_failed" in df_random.columns:
                            df_random = df_random[df_random["decode_failed"] == False]
                        df_burst = df.copy()
                        print(f"Found complementary random CSV: {random_csv}")
                        break
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Main curves
    plot_metric_curves(df, args.output_dir, prefix)
    
    # Individual metric plots
    plot_single_metric(df, "psnr", args.output_dir, threshold=25, prefix=prefix)
    plot_single_metric(df, "ms_ssim", args.output_dir, threshold=0.9, prefix=prefix)
    plot_single_metric(df, "lpips", args.output_dir, threshold=0.3, prefix=prefix)
    
    # Heatmap
    plot_heatmap(df, args.output_dir, prefix)
    
    # Failure thresholds
    print("\nComputing failure thresholds...")
    failure_df = compute_failure_thresholds(df, args.output_dir, prefix)
    print(failure_df)
    
    # Create bitstream random vs burst comparison plot if both are available
    if df_random is not None and df_burst is not None:
        print("\nGenerating bitstream random vs burst comparison plot...")
        # Use a generic prefix for comparison plots
        comparison_prefix = "bitstream_" if is_bitstream else ""
        plot_comparison(df_random, df_burst, "Random Errors", "Burst Errors", 
                       args.output_dir, comparison_prefix)
    
    # Comparison plot if second CSV provided
    if args.compare_csv and os.path.exists(args.compare_csv):
        df2 = pd.read_csv(args.compare_csv)
        if "decode_failed" in df2.columns:
            df2 = df2[df2["decode_failed"] == False]
        plot_comparison(df, df2, args.label1, args.label2, args.output_dir, prefix)
    
    print("\nDone!")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Plot robustness experiment results")
    
    parser.add_argument("--input_csv", type=str, default="indicators/robustness_results.csv",
                        help="Path to robustness results CSV")
    parser.add_argument("--output_dir", type=str, default="indicators/",
                        help="Directory to save plots")
    parser.add_argument("--compare_csv", type=str, default=None,
                        help="Optional: second CSV for comparison plots")
    parser.add_argument("--label1", type=str, default="RDEIC",
                        help="Label for first dataset")
    parser.add_argument("--label2", type=str, default="Baseline",
                        help="Label for second dataset (comparison)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

