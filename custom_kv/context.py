"""
context.py — ecc_cache() context manager for clean lifecycle management.

Handles:
  - Cache allocation
  - Model monkey-patching
  - CUDA memory cleanup on exit
  - Exception safety
"""
from contextlib import contextmanager
from typing import Optional, Generator
import torch

from .cache import ErrorCorrectedCache
from .patch import patch_model, unpatch_model


@contextmanager
def ecc_cache(
    model,
    batch_size: int = 1,
    max_cache_len: int = 128_000,
    device: Optional[str] = None,
) -> Generator[ErrorCorrectedCache, None, None]:
    """
    Context manager that patches the model for ECC-cached inference.

    Allocates the ErrorCorrectedCache, installs the SDPA monkey-patch,
    and guarantees cleanup even if an exception occurs.

    Args:
        model:         Llama-3 model (transformers AutoModelForCausalLM)
        batch_size:    number of sequences to process concurrently
        max_cache_len: maximum sequence length to pre-allocate for
        device:        target device (default: "cuda" if available)

    Yields:
        ErrorCorrectedCache instance (pass as past_key_values to model.generate)

    Usage:
        with ecc_cache(model, max_cache_len=128_000) as cache:
            output = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=500,
            )
        # Cache automatically freed here, model restored to normal

    Notes:
        - The context manager is NOT thread-safe (global SDPA patch)
        - For multi-threaded serving, use ErrorCorrectedCache directly
          with per-request patching via patch_model()/unpatch_model()
    """
    cache = ErrorCorrectedCache(
        config=model.config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
    )

    patch_model(model, cache)

    try:
        yield cache
    finally:
        unpatch_model()

        # Explicit deletion + cache clear for deterministic VRAM release
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
