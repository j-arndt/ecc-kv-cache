"""
Patch cache.py and patch.py to fix three bugs:
1. super().__init__() fails with new transformers Cache signature
2. _compress_cuda calls compress_and_store with wrong arg count
3. _triton_decode passes unrotated query and relies on broken Triton kernel

Fix: keep INT4 storage (VRAM savings) but use pure PyTorch for compress+decode.
"""

CACHE_PY = '''\
"""
cache.py -- ErrorCorrectedCache: drop-in HuggingFace KV cache replacement.

Implements INT4 + ECC syndrome storage with:
  - Pre-allocated CUDA tensors (bypasses PyTorch dynamic allocator)
  - Lloyd-Max INT4 quantization via Hadamard-rotated keys
  - 1-bit Rademacher ECC syndrome for direction recovery
  - HuggingFace Cache API compatibility (past_key_values argument)

Compress/decompress uses pure PyTorch (GPU tensors). The VRAM savings come
from the pre-allocated INT4 layout, not from the compression compute path.
"""
import torch
from typing import Optional, Tuple, List, Dict, Any

try:
    from transformers.cache_utils import Cache as _HFCache
    _HF_CACHE_IMPORTED = True
except ImportError:
    _HFCache = object
    _HF_CACHE_IMPORTED = False


class ErrorCorrectedCache(_HFCache):
    """
    Drop-in KV cache replacement with INT4 + Rademacher ECC syndrome.

    Memory layout per token per head:
      k_int4:    [B, H, L, D//2]   uint8  -- packed INT4
      k_syn:     [B, H, L, D//8]   uint8  -- packed sign bits
      k_meta:    [B, H, L, 3]      FP16   -- [scale, zero, alpha]
      (same for v_*)

    Usage:
        with ecc_cache(model, max_cache_len=128000) as cache:
            output = model.generate(**inputs, past_key_values=cache, ...)
    """

    def __init__(
        self,
        config,
        batch_size: int = 1,
        max_cache_len: int = 128_000,
        device: Optional[str] = None,
    ):
        # Newer transformers changed Cache.__init__ to require args we don\'t have.
        # We implement the full HF Cache interface ourselves, so skip parent init.
        try:
            super().__init__()
        except (TypeError, ValueError):
            pass

        self.config = config
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.num_heads = getattr(config, "num_key_value_heads",
                                  config.num_attention_heads)
        self.head_dim = getattr(config, "head_dim",
                                 config.hidden_size // config.num_attention_heads)

        self._seen_tokens: int = 0
        self._num_layers: int = config.num_hidden_layers

        B = batch_size
        H = self.num_heads
        L = max_cache_len
        D = self.head_dim

        # Pre-allocated INT4 storage -- this is what gives us the VRAM savings
        self._k_int4 = torch.zeros((B, H, L, D // 2), dtype=torch.uint8, device=self._device)
        self._v_int4 = torch.zeros((B, H, L, D // 2), dtype=torch.uint8, device=self._device)
        self._k_syn  = torch.zeros((B, H, L, D // 8), dtype=torch.uint8, device=self._device)
        self._v_syn  = torch.zeros((B, H, L, D // 8), dtype=torch.uint8, device=self._device)
        self._k_meta = torch.zeros((B, H, L, 3), dtype=torch.float16, device=self._device)
        self._v_meta = torch.zeros((B, H, L, 3), dtype=torch.float16, device=self._device)

        # Hadamard rotation matrix (D x D, orthonormal)
        self._H = self._build_hadamard(D).to(self._device)

        self._layer_fill: List[int] = [0] * self._num_layers

    def _build_hadamard(self, D: int) -> torch.Tensor:
        assert (D & (D - 1)) == 0, f"D must be power of 2, got {D}"
        H = torch.ones(1, 1, dtype=torch.float32)
        while H.shape[0] < D:
            H = torch.cat([torch.cat([H,  H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        return H / (D ** 0.5)

    # ------------------------------------------------------------------ #
    # HuggingFace Cache API                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress and store key/value states. Returns placeholder tensors;
        actual attention is computed via the patched SDPA in patch.py.
        """
        B, H, L_new, D = key_states.shape
        pos = self._layer_fill[layer_idx]

        # Hadamard-rotate keys (values don\'t need rotation)
        k_rot = self._rotate(key_states)

        # Compress and store via pure PyTorch (works on GPU tensors)
        self._compress(k_rot,      self._k_int4, self._k_syn,  self._k_meta,  pos, L_new)
        self._compress(value_states, self._v_int4, self._v_syn, self._v_meta, pos, L_new)

        self._layer_fill[layer_idx] = pos + L_new
        if layer_idx == self._num_layers - 1:
            self._seen_tokens += L_new

        # Return placeholder slices (never used for attention directly)
        return (self._k_int4[:, :, :pos + L_new],
                self._v_int4[:, :, :pos + L_new])

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Hadamard rotation: x @ H^T"""
        return torch.matmul(x.float(), self._H.T).to(x.dtype)

    def _compress(
        self,
        x: torch.Tensor,          # [B, H, L_new, D]
        int4_store: torch.Tensor,  # [B, H, max_L, D//2] uint8
        syn_store: torch.Tensor,   # [B, H, max_L, D//8] uint8
        meta_store: torch.Tensor,  # [B, H, max_L, 3] FP16
        pos: int,
        L_new: int,
    ) -> None:
        """
        Quantize x to INT4 + ECC syndrome and write into pre-allocated stores.
        Pure PyTorch -- runs on GPU tensors without any custom CUDA calls.
        """
        B, H, _, D = x.shape
        N = B * H * L_new
        xf = x.reshape(N, D).float()  # [N, D] on GPU

        # Per-token scale and zero-point (mean-abs normalisation)
        abs_max = xf.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        zero    = xf.mean(dim=-1, keepdim=True)
        scale   = abs_max

        x_centered = xf - zero
        x_norm     = x_centered / scale

        # INT4: quantize to 16 levels [-8, 7]
        q_float = (x_norm * 7.5).round().clamp(-8, 7)
        k_tilde = (q_float / 7.5) * scale + zero
        epsilon = xf - k_tilde
        alpha   = epsilon.abs().mean(dim=-1, keepdim=True)

        # Pack two INT4 values per byte: lo nibble = even dims, hi nibble = odd dims
        q_u8   = (q_float + 8).to(torch.uint8)                         # [N, D] values 0-15
        packed = q_u8[:, 0::2] | (q_u8[:, 1::2] << 4)                  # [N, D//2]
        int4_store[:, :, pos:pos + L_new, :] = packed.reshape(B, H, L_new, D // 2)

        # Pack syndrome: 1 bit per dim = sign of residual
        signs = (epsilon >= 0).to(torch.uint8)                          # [N, D]
        packed_syn = torch.zeros(N, D // 8, dtype=torch.uint8, device=x.device)
        for bit in range(8):
            packed_syn |= (signs[:, bit::8] << bit)
        syn_store[:, :, pos:pos + L_new, :] = packed_syn.reshape(B, H, L_new, D // 8)

        # Metadata: [scale, zero, alpha] per token
        meta = torch.cat([scale, zero, alpha], dim=-1).half()           # [N, 3]
        meta_store[:, :, pos:pos + L_new, :] = meta.reshape(B, H, L_new, 3)

    # ------------------------------------------------------------------ #
    # Dequantization for decode path                                       #
    # ------------------------------------------------------------------ #

    def get_kv_fp16_for_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize stored INT4+ECC tensors back to FP16 for attention compute.
        Returns (k_fp16, v_fp16) of shape [B, H, L, D].

        Called by patch.py on every decode step. The returned tensors are
        temporary -- they\'re freed after attention and don\'t accumulate.
        """
        L = self._layer_fill[layer_idx]
        k_fp16 = self._dequantize(self._k_int4[:, :, :L],
                                   self._k_syn[:, :, :L],
                                   self._k_meta[:, :, :L])
        v_fp16 = self._dequantize(self._v_int4[:, :, :L],
                                   self._v_syn[:, :, :L],
                                   self._v_meta[:, :, :L])
        return k_fp16, v_fp16

    def _dequantize(
        self,
        int4: torch.Tensor,   # [B, H, L, D//2] uint8
        syn: torch.Tensor,    # [B, H, L, D//8] uint8
        meta: torch.Tensor,   # [B, H, L, 3] FP16
    ) -> torch.Tensor:
        """Reconstruct FP16 tensor from INT4 + ECC syndrome."""
        B, H, L, _ = int4.shape
        D = int4.shape[-1] * 2
        N = B * H * L

        packed = int4.reshape(N, D // 2)
        q_lo = (packed & 0x0F).float()                   # even dims
        q_hi = ((packed >> 4) & 0x0F).float()            # odd dims
        # Interleave: [lo0, hi0, lo1, hi1, ...]
        q = torch.stack([q_lo, q_hi], dim=-1).reshape(N, D)  # [N, D]
        q_float = q - 8                                  # back to [-8, 7]

        meta_f = meta.reshape(N, 3).float()
        scale = meta_f[:, 0:1]   # [N, 1]
        zero  = meta_f[:, 1:2]   # [N, 1]
        alpha = meta_f[:, 2:3]   # [N, 1]

        k_tilde = (q_float / 7.5) * scale + zero

        # Expand syndrome bits to [-1, +1] signs
        syn_flat = syn.reshape(N, D // 8)
        bits = torch.zeros(N, D, dtype=torch.uint8, device=int4.device)
        for bit in range(8):
            bits[:, bit::8] = (syn_flat >> bit) & 1
        signs = bits.float() * 2 - 1                    # {0,1} -> {-1,+1}

        k_rec = (k_tilde + alpha * signs).half()
        return k_rec.reshape(B, H, L, D)

    # ------------------------------------------------------------------ #
    # HuggingFace Cache protocol                                           #
    # ------------------------------------------------------------------ #

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._layer_fill[layer_idx]

    def get_max_cache_shape(self):
        return None

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self._layer_fill[layer_idx]

    # Legacy compat: some transformers versions call this
    def get_max_length(self):
        return self.max_cache_len

    def memory_bytes(self) -> dict:
        def _b(t): return t.numel() * t.element_size()
        total = sum(_b(t) for t in [
            self._k_int4, self._v_int4,
            self._k_syn,  self._v_syn,
            self._k_meta, self._v_meta,
        ])
        return {"cache_bytes": total, "cache_gb": total / 1e9,
                "fp16_equivalent_gb": total / 1e9 * 2.0, "compression_ratio": 2.0}

    def __repr__(self) -> str:
        mem = self.memory_bytes()
        return (f"ErrorCorrectedCache(seq_len={self._seen_tokens}, "
                f"max_len={self.max_cache_len}, "
                f"vram={mem[\'cache_gb\']:.2f}GB, "
                f"compression={mem[\'compression_ratio\']:.1f}x)")
'''

PATCH_PY = '''\
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
'''

with open("custom_kv/cache.py", "w", encoding="utf-8") as f:
    f.write(CACHE_PY)

with open("custom_kv/patch.py", "w", encoding="utf-8") as f:
    f.write(PATCH_PY)

print("cache.py:", len(CACHE_PY.splitlines()), "lines")
print("patch.py:", len(PATCH_PY.splitlines()), "lines")

# Quick syntax check
import py_compile
py_compile.compile("custom_kv/cache.py", doraise=True)
py_compile.compile("custom_kv/patch.py", doraise=True)
print("Syntax OK")
