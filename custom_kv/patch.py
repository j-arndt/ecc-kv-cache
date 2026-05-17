"""
patch.py -- Monkey-patches F.scaled_dot_product_attention to route decode
           queries through the ECC decode path.

Prefill (L > 1): standard SDPA (correct attention, cache.update() stores INT4)
Decode  (L == 1): dequantize stored INT4 for this layer -> standard SDPA
                  The dequantized FP16 tensor is temporary and freed immediately.
"""
import torch
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import ErrorCorrectedCache

_original_sdpa = F.scaled_dot_product_attention
_active_cache: Optional["ErrorCorrectedCache"] = None
_current_layer_idx: int = 0
_is_patched: bool = False
_hook_handles = []


def patch_model(model, cache: "ErrorCorrectedCache") -> None:
    global _active_cache, _is_patched, _current_layer_idx
    if _is_patched:
        return
    _active_cache = cache
    _current_layer_idx = 0
    F.scaled_dot_product_attention = _ecc_sdpa
    _install_layer_hooks(model)
    _is_patched = True


def unpatch_model() -> None:
    global _active_cache, _is_patched, _current_layer_idx, _hook_handles
    F.scaled_dot_product_attention = _original_sdpa
    _active_cache = None
    _is_patched = False
    _current_layer_idx = 0
    for handle in _hook_handles:
        handle.remove()
    _hook_handles.clear()


def _install_layer_hooks(model) -> None:
    global _current_layer_idx, _hook_handles
    try:
        layers = model.model.layers
    except AttributeError:
        layers = [m for n, m in model.named_modules()
                  if "self_attn" in n and hasattr(m, "q_proj")]

    def make_hook(layer_idx):
        def pre_hook(module, args):
            global _current_layer_idx
            _current_layer_idx = layer_idx
        return pre_hook

    for i, layer in enumerate(layers):
        attn = layer.self_attn if hasattr(layer, "self_attn") else layer
        _hook_handles.append(attn.register_forward_pre_hook(make_hook(i)))


def _ecc_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    **kwargs,
) -> torch.Tensor:
    """
    Patched SDPA.

    Decode step (query seq_len == 1):
      - Rotate query by same Hadamard used to store keys
      - Dequantize stored INT4 keys/values for this layer
      - Run standard SDPA on the dequantized FP16 tensors
      - Temporary FP16 tensors are freed immediately after return

    Prefill step (query seq_len > 1):
      - Standard SDPA (attention computed normally)
      - cache.update() has already stored the INT4 compressed version
    """
    if _active_cache is not None and query.shape[2] == 1:
        return _ecc_decode_step(query, _current_layer_idx)

    return _original_sdpa(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kwargs,
    )


def _ecc_decode_step(query: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """
    Dequantize stored INT4 cache for this layer and run standard SDPA.
    query: [B, H, 1, D] -- unrotated, as produced by LlamaAttention
    """
    # Rotate query to match the Hadamard-rotated stored keys
    q_rot = torch.matmul(query.float(), _active_cache._H.T).to(query.dtype)

    # Dequantize: returns [B, H, L, D] FP16
    k_fp16, v_fp16 = _active_cache.get_kv_fp16_for_layer(layer_idx)

    # Standard scaled dot-product attention on dequantized tensors
    return _original_sdpa(
        q_rot, k_fp16, v_fp16,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
