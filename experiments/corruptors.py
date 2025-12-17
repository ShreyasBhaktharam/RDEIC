"""
Error injection utilities for robustness experiments.

Supports:
- Bitstream-level corruption (random bit flips, burst errors)
- Latent-level corruption (mask-replace, additive noise)
"""

import numpy as np
import torch
from typing import Tuple, Literal


def bit_flip_bytes(data: bytes, rate: float, seed: int = 42) -> bytes:
    """
    Randomly flip bits in a byte sequence.
    
    Args:
        data: Input byte sequence.
        rate: Fraction of bits to flip (0.0 to 1.0, e.g., 0.01 = 1%).
        seed: Random seed for reproducibility.
        
    Returns:
        Corrupted byte sequence.
    """
    if rate <= 0:
        return data
    
    rng = np.random.RandomState(seed)
    byte_array = bytearray(data)
    total_bits = len(byte_array) * 8
    num_flips = int(total_bits * rate)
    
    if num_flips == 0:
        return data
    
    # Select random bit positions to flip
    flip_positions = rng.choice(total_bits, size=num_flips, replace=False)
    
    for pos in flip_positions:
        byte_idx = pos // 8
        bit_idx = pos % 8
        byte_array[byte_idx] ^= (1 << bit_idx)
    
    return bytes(byte_array)


def burst_flip_bytes(data: bytes, burst_rate: float, mean_burst_len: float = 8.0, seed: int = 42) -> bytes:
    """
    Introduce burst errors: contiguous runs of flipped bits.
    
    Args:
        data: Input byte sequence.
        burst_rate: Expected fraction of bits affected by bursts (0.0 to 1.0).
        mean_burst_len: Average length of each burst (geometric distribution).
        seed: Random seed for reproducibility.
        
    Returns:
        Corrupted byte sequence.
    """
    if burst_rate <= 0:
        return data
    
    rng = np.random.RandomState(seed)
    byte_array = bytearray(data)
    total_bits = len(byte_array) * 8
    target_flipped = int(total_bits * burst_rate)
    
    if target_flipped == 0:
        return data
    
    flipped = 0
    flipped_positions = set()
    
    while flipped < target_flipped:
        # Pick a random starting position
        start = rng.randint(0, total_bits)
        # Sample burst length from geometric distribution
        burst_len = rng.geometric(1.0 / mean_burst_len)
        
        for offset in range(burst_len):
            pos = (start + offset) % total_bits
            if pos not in flipped_positions:
                flipped_positions.add(pos)
                flipped += 1
                if flipped >= target_flipped:
                    break
    
    for pos in flipped_positions:
        byte_idx = pos // 8
        bit_idx = pos % 8
        byte_array[byte_idx] ^= (1 << bit_idx)
    
    return bytes(byte_array)


def latent_corrupt(
    c_latent: torch.Tensor,
    mode: Literal["mask_replace", "additive"],
    rate: float,
    seed: int = 42,
    valid_range: Tuple[float, float] = (-3.0, 3.0)
) -> torch.Tensor:
    """
    Corrupt a latent tensor.
    
    Args:
        c_latent: Input latent tensor (B, C, H, W).
        mode: Corruption mode:
            - "mask_replace": Randomly mask entries and replace with random values.
            - "additive": Add Gaussian noise, then clamp to valid range.
        rate: Corruption rate (fraction of elements for mask_replace, noise std for additive).
        seed: Random seed for reproducibility.
        valid_range: (min, max) range for clamping after corruption.
        
    Returns:
        Corrupted latent tensor.
    """
    if rate <= 0:
        return c_latent.clone()
    
    torch.manual_seed(seed)
    result = c_latent.clone()
    
    if mode == "mask_replace":
        # Create a random mask
        mask = torch.rand_like(c_latent) < rate
        # Generate random replacement values in the valid range
        replacements = torch.rand_like(c_latent) * (valid_range[1] - valid_range[0]) + valid_range[0]
        result[mask] = replacements[mask]
        
    elif mode == "additive":
        # Add Gaussian noise with std = rate
        noise = torch.randn_like(c_latent) * rate
        result = result + noise
        result = result.clamp(valid_range[0], valid_range[1])
        
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")
    
    return result


def corrupt_bitstream_file(
    input_path: str,
    output_path: str,
    error_type: Literal["random", "burst"],
    rate: float,
    seed: int = 42,
    mean_burst_len: float = 8.0
) -> None:
    """
    Read a bitstream file, corrupt it, and write to a new file.
    
    Args:
        input_path: Path to the original bitstream file.
        output_path: Path to save the corrupted bitstream.
        error_type: "random" for uniform bit flips, "burst" for burst errors.
        rate: Error rate (fraction of bits to flip).
        seed: Random seed.
        mean_burst_len: Mean burst length for burst errors.
    """
    with open(input_path, "rb") as f:
        data = f.read()
    
    if error_type == "random":
        corrupted = bit_flip_bytes(data, rate, seed)
    elif error_type == "burst":
        corrupted = burst_flip_bytes(data, rate, mean_burst_len, seed)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    with open(output_path, "wb") as f:
        f.write(corrupted)


def estimate_latent_range(c_latent: torch.Tensor, margin: float = 0.5) -> Tuple[float, float]:
    """
    Estimate a valid range for latent values based on the tensor statistics.
    
    Args:
        c_latent: The latent tensor to analyze.
        margin: Extra margin beyond min/max.
        
    Returns:
        (min_val, max_val) tuple.
    """
    min_val = c_latent.min().item() - margin
    max_val = c_latent.max().item() + margin
    return (min_val, max_val)


# Unified interface
class Corruptor:
    """
    Unified interface for corruption operations.
    """
    
    def __init__(
        self,
        error_space: Literal["bitstream", "latent"],
        error_type: Literal["random", "burst", "mask_replace", "additive"],
        rate: float,
        seed: int = 42,
        mean_burst_len: float = 8.0,
        valid_range: Tuple[float, float] = (-3.0, 3.0)
    ):
        self.error_space = error_space
        self.error_type = error_type
        self.rate = rate
        self.seed = seed
        self.mean_burst_len = mean_burst_len
        self.valid_range = valid_range
    
    def corrupt_bytes(self, data: bytes) -> bytes:
        """Corrupt a byte sequence (for bitstream corruption)."""
        if self.error_space != "bitstream":
            raise ValueError("corrupt_bytes only works with bitstream error_space")
        
        if self.error_type == "random":
            return bit_flip_bytes(data, self.rate, self.seed)
        elif self.error_type == "burst":
            return burst_flip_bytes(data, self.rate, self.mean_burst_len, self.seed)
        else:
            raise ValueError(f"Invalid error_type for bitstream: {self.error_type}")
    
    def corrupt_latent(self, c_latent: torch.Tensor) -> torch.Tensor:
        """Corrupt a latent tensor."""
        if self.error_space != "latent":
            raise ValueError("corrupt_latent only works with latent error_space")
        
        if self.error_type in ["mask_replace", "additive"]:
            return latent_corrupt(c_latent, self.error_type, self.rate, self.seed, self.valid_range)
        else:
            raise ValueError(f"Invalid error_type for latent: {self.error_type}")
    
    def with_seed(self, new_seed: int) -> "Corruptor":
        """Return a new Corruptor with a different seed."""
        return Corruptor(
            error_space=self.error_space,
            error_type=self.error_type,
            rate=self.rate,
            seed=new_seed,
            mean_burst_len=self.mean_burst_len,
            valid_range=self.valid_range
        )


if __name__ == "__main__":
    # Quick test
    print("Testing bit_flip_bytes...")
    original = b"Hello, World!"
    corrupted = bit_flip_bytes(original, 0.1, seed=42)
    print(f"Original: {original}")
    print(f"Corrupted: {corrupted}")
    
    print("\nTesting burst_flip_bytes...")
    corrupted_burst = burst_flip_bytes(original, 0.1, mean_burst_len=4, seed=42)
    print(f"Burst corrupted: {corrupted_burst}")
    
    print("\nTesting latent_corrupt...")
    latent = torch.randn(1, 4, 8, 8)
    corrupted_mask = latent_corrupt(latent, "mask_replace", 0.1, seed=42)
    corrupted_add = latent_corrupt(latent, "additive", 0.5, seed=42)
    print(f"Original latent mean: {latent.mean():.4f}, std: {latent.std():.4f}")
    print(f"Mask-replace mean: {corrupted_mask.mean():.4f}, std: {corrupted_mask.std():.4f}")
    print(f"Additive mean: {corrupted_add.mean():.4f}, std: {corrupted_add.std():.4f}")
    
    print("\nAll tests passed!")

