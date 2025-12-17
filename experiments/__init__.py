"""
RDEIC Experiment Scripts

This package contains scripts for running robustness and OOD experiments:

- corruptors.py: Error injection utilities (bit flips, latent corruption)
- run_robustness.py: Batch runner for error robustness experiments
- run_ood.py: Batch runner for OOD domain generalization experiments  
- plot_robustness.py: Plotting script for robustness results
- plot_ood.py: Plotting script for OOD results

Example usage:

1. Run robustness experiment:
   python experiments/run_robustness.py --data inputs/ --error_space bitstream --error_type random

2. Run OOD experiment:
   python experiments/run_ood.py --domains ood/sketch,ood/xray,ood/cartoon

3. Generate plots:
   python experiments/plot_robustness.py --input_csv indicators/robustness_results.csv
   python experiments/plot_ood.py --input_csv indicators/ood_results_all.csv
"""

from .corruptors import (
    bit_flip_bytes,
    burst_flip_bytes,
    latent_corrupt,
    Corruptor,
    corrupt_bitstream_file,
    estimate_latent_range
)

__all__ = [
    "bit_flip_bytes",
    "burst_flip_bytes", 
    "latent_corrupt",
    "Corruptor",
    "corrupt_bitstream_file",
    "estimate_latent_range"
]

