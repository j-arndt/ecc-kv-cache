"""
patch.py — Monkey-patches F.scaled_dot_product_attention to route decode
           queries through the fused Triton ECC decode kernel.

Architecture:
  - Prefill (L > 1): standard SDPA path (attention computed normally,
    cache.update() compresses and stores in background)
  - Decode (L == 1): intercepted here, routed to fused Triton kernel
    which reads from ECC_KV_Block pool without materializing FP16 tensors

Called by ecc_cache() context manager. Not intended for direct use.
"""
import torch
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import ErrorCorrectedCache


# ─── Module-level state ────────────────────────────────────────────────────
_original_sdpa = F.scaled_dot_product_attention
_active_cache: Optional["ErrorCorrectedCache"] = None
_current_layer_idx: int = 0
_is_patched: bool = False


def patch_model(model, cache: "ErrorCorrectedCache") -> None:
    """
    Apply the SDPA monkey-patch to route decode through ECC Triton kernel.

    Also installs forward pre-hooks on each attention layer to track
    the current layer index without modifying model source code.

    Args:
        model: Llama-3 model (transformers AutoModelForCausalLM)
        cache: ErrorCorrectedCache instance to read from during decode
    """
    global _active_cache, _is_patched, _current_layer_idx

    if _is_patched:
        return  # Idempotent

    _active_cache = cache
    _current_layer_idx = 0

    # Replace the global SDPA function
    F.scaled_dot_product_attention = _ecc_sdpa

    # Install layer-tracking hooks
    _install_layer_hooks(model)

    _is_patched = True


def unpatch_model() -> None:
    """Restore the original SDPA and remove layer hooks."""
    global _active_cache, _is_patched, _current_layer_idx, _hook_handles

    F.scaled_dot_product_attention = _original_sdpa
    _active_cache = None
    _is_patched = False
    _current_layer_idx = 0

    # Remove hooks
    for handle in _hook_handles:
        handle.remove()
    _hook_handles.clear()


# ─── Internal state ────────────────────────────────────────────────────────
_hook_handles = []


def _install_layer_hooks(model) -> None:
    """
    Install pre-forward hooks on each LlamaAttention to track layer index.
    The hook fires immediately before the attention module processes its inputs.
    """
    global _current_layer_idx, _hook_handles

    num_layers = model.config.num_hidden_layers

    # Access the layers (works for Llama-3 transformers layout)
    try:
        layers = model.model.layers
    except AttributeError:
        # Fallback for non-standard model layouts
        layers = []
        for name, module in model.named_modules():
            if "self_attn" in name and hasattr(module, "q_proj"):
                layers.append(module)

    def make_hook(layer_idx):
        def pre_hook(module, args):
            global _current_layer_idx
            _current_layer_idx = layer_idx
        return pre_hook

    for i, layer in enumerate(layers):
        attn = layer.self_attn if hasattr(layer, "self_attn") else layer
        handle = attn.register_forward_pre_hook(make_hook(i))
        _hook_handles.append(handle)


# ─── Patched SDPA ─────────────────────────────────────────────────────────

def _ecc_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Patched version of F.scaled_dot_product_attention.

    Routes to ECC Triton decode kernel when:
      1. A cache is active (_active_cache is not None)
      2. This is a decode step (query seq_len == 1)

    Falls back to standard SDPA for:
      - Prefill (query seq_len > 1): KV already compressed in cache.update()
      - Any non-ECC path
    """
    if _active_cache is not None and query.shape[2] == 1:
        # ── DECODE PATH: route to fused Triton kernel ──────────────────
        return _triton_decode(query, _current_layer_idx)

    # ── PREFILL PATH: standard attention ───────────────────────────────
    # During prefill, key/value are passed normally (not from ECC cache).
    # cache.update() handles compression asynchronously.
    return _original_sdpa(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kwargs,
    )


def _triton_decode(
    query_rot: torch.Tensor,  # [B, H, 1, D] — already Hadamard-rotated by cache.update()
    layer_idx: int,
) -> torch.Tensor:
    """
    Dispatch to the fused Triton decode kernel.
    Loads ECC_KV_Block data, dequantizes inline, runs flash-attention online softmax.
    """
    from .fused_decode import fused_ecc_decode_attention

    kv = _active_cache.get_kv_for_layer(layer_idx)
    return fused_ecc_decode_attention(
        query_rot=query_rot,
        k_int4=kv["k_int4"],
        k_syn=kv["k_syn"],
        k_meta=kv["k_meta"],
        v_int4=kv["v_int4"],
        v_syn=kv["v_syn"],
        v_meta=kv["v_meta"],
        seq_len=kv["seq_len"],
    )
