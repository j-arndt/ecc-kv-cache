"""
ops_cpu.py — Pure PyTorch/NumPy reference implementations of all ECC operations.

These are used exclusively for unit testing on CPU (no GPU or CUDA required).
They are mathematically faithful to the CUDA kernels but not performance-optimized.

IMPORTANT: These functions serve as the ground truth "oracle" for verifying
the CUDA kernels on A100. If a CUDA kernel disagrees with these within tolerance,
the CUDA kernel is wrong.
"""
import torch
import numpy as np
from typing import Tuple


# ─── Lloyd-Max centroid table (N(0,1), 16 levels) ─────────────────────────
# Pre-computed via scipy.stats or offline optimization.
# These match the __constant__ LM_CENTROIDS_NORM in compress_ecc.cu
LM_CENTROIDS_NORM = torch.tensor([
    -2.4008, -1.8435, -1.4371, -1.0993,
    -0.7996, -0.5224, -0.2582,  0.0000,
     0.2582,  0.5224,  0.7996,  1.0993,
     1.4371,  1.8435,  2.4008,  3.1584,
], dtype=torch.float32)


# ─── Hadamard Rotation ────────────────────────────────────────────────────

def hadamard_rotate_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply normalized Walsh-Hadamard Transform to last dimension of x.

    Args:
        x: (..., D) tensor where D is a power of 2

    Returns:
        x_rot: (..., D) same shape, H_n @ x along last dim

    Properties:
        hadamard_rotate(hadamard_rotate(x)) == x  (self-inverse after renorm)
        torch.dot(q, k) == torch.dot(H@q, H@k)    (inner product preserved)
    """
    x = x.float().clone()
    D = x.shape[-1]
    assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"

    # Cooley-Tukey WHT butterfly
    stride = 1
    while stride < D:
        # Reshape to expose pairs
        x_view = x.view(*x.shape[:-1], D // (stride * 2), stride * 2)
        a = x_view[..., :stride].clone()
        b = x_view[..., stride:].clone()
        x_view[..., :stride] = a + b
        x_view[..., stride:] = a - b
        stride *= 2

    # Normalize: H_n = H / sqrt(D) so H_n @ H_n^T = I
    return x / (D ** 0.5)


# ─── Lloyd-Max Quantization ───────────────────────────────────────────────

def lloyd_max_params_cpu(x: torch.Tensor) -> Tuple[float, float]:
    """
    Compute per-block Lloyd-Max scale and zero point.
    Assumes x follows N(mean, std) after Hadamard rotation.

    Returns:
        (scale, zero): scale = std, zero = mean
    """
    x = x.float()
    zero = x.mean().item()
    scale = x.std().item() + 1e-8
    return scale, zero


def lloyd_max_quantize_cpu(
    x: torch.Tensor,
    scale: float,
    zero: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize x to INT4 using Lloyd-Max centroids.

    Args:
        x:     (..., D) float tensor
        scale: per-block std
        zero:  per-block mean

    Returns:
        q:       (..., D) int64 in [0, 15] — quantized indices
        k_tilde: (..., D) float  — dequantized approximation
    """
    x = x.float()

    # Normalize to N(0,1) space
    x_norm = (x - zero) / scale

    # Find nearest centroid (argmin over 16 centroids)
    # Expand for broadcasting: x_norm [..., D, 1] vs centroids [16]
    diffs = (x_norm.unsqueeze(-1) - LM_CENTROIDS_NORM).abs()
    q = diffs.argmin(dim=-1)  # (..., D) in [0, 15]

    # Dequantize
    k_tilde_norm = LM_CENTROIDS_NORM[q]
    k_tilde = k_tilde_norm * scale + zero

    return q, k_tilde


# ─── Rademacher Syndrome ──────────────────────────────────────────────────

def compute_syndrome_cpu(epsilon: torch.Tensor) -> torch.Tensor:
    """
    Compute 1-bit Rademacher syndrome from quantization residual.

    syndrome[i] = 1 if epsilon[i] >= 0, else 0

    Args:
        epsilon: (..., D) float tensor — quantization residual (x - k_tilde)

    Returns:
        syndrome: (..., D) bool tensor — 1=positive, 0=negative
    """
    return (epsilon >= 0).to(torch.bool)


def syndrome_to_float(syndrome: torch.Tensor) -> torch.Tensor:
    """
    Convert binary syndrome to Rademacher {-1, +1} floats.

    s_float[i] = +1.0 if syndrome[i]=True, else -1.0
    """
    return syndrome.float() * 2.0 - 1.0


# ─── Reconstruction ───────────────────────────────────────────────────────

def reconstruct_attention_score_cpu(
    q_rot: torch.Tensor,
    k_tilde: torch.Tensor,
    syndrome: torch.Tensor,
    alpha: float,
    inv_sqrt_d: float = None,
) -> float:
    """
    Compute the ECC-reconstructed attention score: q_rot · k_reconstructed.

    Formula:
        k_reconstructed = k_tilde + alpha * s_float
        score = (q_rot · k_reconstructed) / sqrt(D)

    This decomposes as:
        score = (q · k_tilde + alpha * q · s_float) / sqrt(D)

    Both terms computed in one fused operation in the Triton kernel.
    Here we compute them separately for clarity.

    Args:
        q_rot:   [D] query vector (Hadamard-rotated)
        k_tilde: [D] dequantized key approximation
        syndrome: [D] bool — Rademacher sign bits
        alpha:   scalar compensation (mean absolute residual)
        inv_sqrt_d: 1/sqrt(D), computed if None

    Returns:
        attention score (scalar float)
    """
    q_rot = q_rot.float()
    k_tilde = k_tilde.float()
    s_float = syndrome_to_float(syndrome)

    D = q_rot.shape[-1]
    if inv_sqrt_d is None:
        inv_sqrt_d = D ** -0.5

    # Primary term: INT4 dequantized dot product
    primary = torch.dot(q_rot, k_tilde)

    # ECC correction term: alpha * (q · sign(epsilon))
    correction = alpha * torch.dot(q_rot, s_float)

    return float((primary + correction) * inv_sqrt_d)


def fp16_attention_score_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
) -> float:
    """FP16 reference attention score (ground truth for comparison)."""
    D = q.shape[-1]
    return float(torch.dot(q.float(), k.float()) / (D ** 0.5))


# ─── Full Compress → Reconstruct Pipeline (CPU Oracle) ───────────────────

def compress_cpu(x: torch.Tensor) -> dict:
    """
    Full CPU compression pipeline for one (token, head) vector.

    Steps:
        1. Hadamard rotate
        2. Lloyd-Max quantize
        3. Compute syndrome
        4. Compute alpha

    Args:
        x: [D] FP32 key/value vector (pre-RoPE rotation)

    Returns:
        dict with keys: q, k_tilde, syndrome, alpha, scale, zero, x_rot
    """
    x_rot = hadamard_rotate_cpu(x)
    scale, zero = lloyd_max_params_cpu(x_rot)
    q, k_tilde = lloyd_max_quantize_cpu(x_rot, scale, zero)
    epsilon = x_rot - k_tilde
    syndrome = compute_syndrome_cpu(epsilon)
    alpha = epsilon.abs().mean().item()

    return {
        "x_rot": x_rot,
        "q": q,
        "k_tilde": k_tilde,
        "epsilon": epsilon,
        "syndrome": syndrome,
        "alpha": alpha,
        "scale": scale,
        "zero": zero,
    }


def compression_ratio_bytes() -> dict:
    """Return theoretical compression ratio for D=128."""
    D = 128
    fp16_bytes = D * 2          # 256 bytes
    ecc_bytes  = 128            # 1 ECC_KV_Block
    return {
        "fp16_bytes_per_token_head": fp16_bytes,
        "ecc_bytes_per_token_head":  ecc_bytes,
        "compression_ratio":         fp16_bytes / ecc_bytes,
    }
