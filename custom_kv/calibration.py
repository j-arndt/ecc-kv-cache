"""
calibration.py — Per-layer Lloyd-Max calibration for ECC KV cache.

Runs a forward pass over calibration data to collect KV activation
statistics per layer, then fits Lloyd-Max centroids and estimates
the ECC compensation scalar alpha = E[|epsilon|].

Run ONCE on A100 before benchmarking:
    python scripts/run_calibration.py --n-samples 512 --output calibration_config.json

Output format (calibration_config.json):
    {
      "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
      "n_samples": 512,
      "layers": {
        "0": {"scale": 1.23, "zero": 0.01, "alpha": 0.18},
        "1": {...},
        ...
      }
    }
"""
import torch
import json
import math
from typing import Dict, List, Optional, Any
from pathlib import Path


# ─── Lloyd-Max centroid computation ───────────────────────────────────────

def lloyd_max_gaussian(n_levels: int = 16, sigma: float = 1.0,
                        n_iter: int = 100) -> List[float]:
    """
    Compute Lloyd-Max quantizer centroids for a Gaussian N(0, sigma^2).

    Iterates until convergence:
      1. Compute decision boundaries as midpoints between centroids
      2. Update centroids as conditional expectations within each interval

    Args:
        n_levels: number of quantization levels (16 for INT4)
        sigma:    standard deviation of target distribution
        n_iter:   maximum iterations

    Returns:
        List of n_levels centroid values (sorted ascending)
    """
    import numpy as np
    from scipy import stats

    # Initialize with equal-probability quantiles
    probs = np.linspace(0, 1, n_levels + 1)[1:-1]
    boundaries = stats.norm.ppf(probs, scale=sigma)
    centroids = np.zeros(n_levels)

    for _ in range(n_iter):
        # Update centroids: E[X | boundary[i-1] < X < boundary[i]]
        full_bounds = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_centroids = np.zeros(n_levels)
        for k in range(n_levels):
            lo, hi = full_bounds[k], full_bounds[k + 1]
            # E[X | lo < X < hi] = sigma * (phi(lo/sigma) - phi(hi/sigma)) /
            #                               (Phi(hi/sigma) - Phi(lo/sigma))
            p_lo = stats.norm.pdf(lo / sigma)
            p_hi = stats.norm.pdf(hi / sigma)
            P    = stats.norm.cdf(hi / sigma) - stats.norm.cdf(lo / sigma)
            if P < 1e-10:
                new_centroids[k] = (lo + hi) / 2
            else:
                new_centroids[k] = sigma * (p_lo - p_hi) / P

        # Update boundaries: midpoints between centroids
        new_boundaries = (new_centroids[:-1] + new_centroids[1:]) / 2

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            break

        centroids = new_centroids
        boundaries = new_boundaries

    return centroids.tolist()


# ─── Calibration data collection ──────────────────────────────────────────

class _KVHook:
    """Hook to capture key/value states from a specific attention layer."""

    def __init__(self):
        self.k_stats: List[torch.Tensor] = []
        self.v_stats: List[torch.Tensor] = []

    def hook_fn(self, module, input, output):
        # LlamaAttention output: (attn_output, attn_weights, past_kv)
        # We need to intercept within the forward — use input hook instead
        pass


def _collect_kv_activations(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 2048,
    device: str = "cuda",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Run forward passes and collect KV activations per layer.

    Returns:
        {layer_idx: {"k": tensor[N, D], "v": tensor[N, D]}}
    """
    layer_kvs: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    num_layers = model.config.num_hidden_layers

    for i in range(num_layers):
        layer_kvs[i] = {"k": [], "v": []}

    # Register hooks on each attention layer
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # Capture key/value from attention output
            if isinstance(output, tuple) and len(output) >= 3:
                past_kv = output[2]
                if past_kv is not None:
                    k, v = past_kv
                    # Sample random tokens for efficiency
                    k_flat = k.detach().float().reshape(-1, k.shape[-1])
                    v_flat = v.detach().float().reshape(-1, v.shape[-1])
                    # Random sample up to 1000 tokens
                    n = min(1000, k_flat.shape[0])
                    idx = torch.randperm(k_flat.shape[0])[:n]
                    layer_kvs[layer_idx]["k"].append(k_flat[idx].cpu())
                    layer_kvs[layer_idx]["v"].append(v_flat[idx].cpu())
        return hook

    for i, layer in enumerate(model.model.layers):
        handle = layer.self_attn.register_forward_hook(
            make_hook(i), with_kwargs=True)
        hooks.append(handle)

    # Forward passes
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(device)
            model(**inputs, use_cache=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate collected tensors
    result = {}
    for i in range(num_layers):
        if layer_kvs[i]["k"]:
            result[i] = {
                "k": torch.cat(layer_kvs[i]["k"], dim=0),
                "v": torch.cat(layer_kvs[i]["v"], dim=0),
            }

    return result


# ─── Main calibration function ────────────────────────────────────────────

def calibrate_from_model(
    model,
    tokenizer,
    calibration_data_path: Optional[str] = None,
    texts: Optional[List[str]] = None,
    n_samples: int = 512,
    max_length: int = 2048,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calibrate Lloyd-Max centroids and ECC alpha per layer.

    Either provide calibration_data_path (directory of .txt files) or
    a list of texts directly.

    Args:
        model:                  loaded Llama-3 model on CUDA
        tokenizer:              matching tokenizer
        calibration_data_path:  path to .txt calibration files
        texts:                  list of strings (alternative to file path)
        n_samples:              number of text samples to use
        max_length:             max tokens per sample
        output_path:            save JSON config here if provided

    Returns:
        calibration config dict
    """
    from .ops_cpu import hadamard_rotate_cpu

    print(f"[calibration] Starting with n_samples={n_samples}")

    # Load calibration texts
    if texts is None:
        texts = _load_calibration_texts(calibration_data_path, n_samples)
    texts = texts[:n_samples]
    print(f"[calibration] Loaded {len(texts)} texts")

    # Collect KV activations
    device = next(model.parameters()).device
    layer_kvs = _collect_kv_activations(
        model, tokenizer, texts, max_length, str(device))
    print(f"[calibration] Collected activations for {len(layer_kvs)} layers")

    # Fit per-layer statistics
    layer_configs = {}
    for layer_idx, kv in layer_kvs.items():
        k = kv["k"]  # [N, D]

        # Apply Hadamard rotation (CPU)
        k_rot = hadamard_rotate_cpu(k)

        # Fit scale/zero from distribution
        sigma = k_rot.std().item()
        mean  = k_rot.mean().item()

        # Compute Lloyd-Max centroids
        from .ops_cpu import LM_CENTROIDS_NORM, lloyd_max_quantize_cpu
        scale, zero = sigma, mean
        _, k_tilde = lloyd_max_quantize_cpu(k_rot, scale, zero)
        epsilon = k_rot - k_tilde
        alpha = epsilon.abs().mean().item()

        layer_configs[str(layer_idx)] = {
            "scale": round(scale, 6),
            "zero":  round(mean, 6),
            "alpha": round(alpha, 6),
            "sigma": round(sigma, 6),
        }

        if layer_idx % 8 == 0:
            print(f"  Layer {layer_idx:2d}: scale={scale:.4f}, "
                  f"zero={mean:.4f}, alpha={alpha:.4f}")

    config = {
        "model_id": getattr(model.config, "_name_or_path",
                             "meta-llama/Meta-Llama-3-8B-Instruct"),
        "n_samples": n_samples,
        "max_length": max_length,
        "layers": layer_configs,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[calibration] Config saved to {output_path}")

    return config


def _load_calibration_texts(data_path: str, n_samples: int) -> List[str]:
    """Load text samples from directory or fall back to HF dataset."""
    path = Path(data_path)
    texts = []

    if path.exists():
        for f in sorted(path.glob("*.txt"))[:n_samples]:
            texts.append(f.read_text(encoding="utf-8")[:4000])

    if len(texts) < n_samples:
        # Fall back to wikitext (always available via datasets)
        print(f"[calibration] Only {len(texts)} local files, "
              f"fetching from wikitext...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                              split="train", streaming=True)
            for item in ds:
                if len(item["text"].strip()) > 200:
                    texts.append(item["text"][:4000])
                if len(texts) >= n_samples:
                    break
        except Exception as e:
            print(f"[calibration] Dataset load failed: {e}")

    return texts


def load_calibration_config(path: str) -> Dict[str, Any]:
    """Load a previously saved calibration config."""
    with open(path) as f:
        return json.load(f)
