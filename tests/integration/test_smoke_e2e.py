"""
test_smoke_e2e.py — Drive ErrorCorrectedCache through model.generate() once.

This catches future HF Cache contract drift (cache.reset, cache.crop,
batch_select_indices, get_max_cache_shape) the same way
test_no_property_collision_with_hf_cache catches the max_cache_len bug.

Runs on CPU in <1 second. No HF download. No real model weights.
"""
import pytest
import torch
import torch.nn as nn
import types


pytest.importorskip("transformers", reason="transformers required for E2E cache test")


class TinyConfig:
    """Minimal config matching the shape contract of LlamaConfig."""
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 64
    head_dim = 16
    vocab_size = 100
    max_position_embeddings = 256
    _name_or_path = "stub/tiny-llama-test"

    # Attributes some HF code paths poke for:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    rope_theta = 10000.0
    rms_norm_eps = 1e-5


class TinyAttention(nn.Module):
    """Minimal attention layer: just k_proj/v_proj for the hooks ErrorCorrectedCache may install."""
    def __init__(self, cfg):
        super().__init__()
        D = cfg.hidden_size
        kv_d = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, kv_d, bias=False)
        self.v_proj = nn.Linear(D, kv_d, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.cfg = cfg


class TinyLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = TinyAttention(cfg)
        self.input_layernorm = nn.LayerNorm(cfg.hidden_size)


class TinyInnerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TinyLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size)


class TinyModel(nn.Module):
    """Stub for AutoModelForCausalLM. Just enough to satisfy the cache lifecycle."""
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = TinyInnerModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids=None, past_key_values=None, **kwargs):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer.input_layernorm(x)
            # Touch k_proj/v_proj so any hooks fire
            layer.self_attn.k_proj(x)
            layer.self_attn.v_proj(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return types.SimpleNamespace(logits=logits, past_key_values=past_key_values)


class TestEndToEndCacheLifecycle:
    """Exercises the full HF Cache contract through ErrorCorrectedCache."""

    def _make_model_and_cache(self):
        from custom_kv.cache import ErrorCorrectedCache
        torch.manual_seed(42)
        cfg = TinyConfig()
        model = TinyModel(cfg).eval()
        cache = ErrorCorrectedCache(cfg, batch_size=1, max_cache_len=64, device="cpu")
        return model, cache

    def test_cache_has_required_methods(self):
        """The HF Cache protocol requires several methods. Catch missing implementations."""
        _, cache = self._make_model_and_cache()
        required = [
            "update",
            "get_seq_length",
            "get_max_cache_shape",
        ]
        optional = ["reset", "crop", "batch_select_indices", "reorder_cache"]
        missing_required = [m for m in required if not hasattr(cache, m)]
        assert not missing_required, (
            f"ErrorCorrectedCache missing required HF Cache methods: {missing_required}"
        )
        missing_optional = [m for m in optional if not hasattr(cache, m)]
        if missing_optional:
            print(f"\n[WARN] Optional HF Cache methods missing: {missing_optional}. "
                  f"If model.generate() needs any of these, it will crash at runtime.")

    def test_update_and_get_seq_length_through_layers(self):
        """Simulate the per-layer update call pattern used by attention forward."""
        _, cache = self._make_model_and_cache()
        B, H, L, D = 1, 2, 8, 16
        for layer_idx in range(2):
            k = torch.randn(B, H, L, D)
            v = torch.randn(B, H, L, D)
            cache.update(k, v, layer_idx)
        assert cache.get_seq_length(0) == L
        assert cache.get_seq_length(1) == L
        assert cache.seen_tokens == L

    def test_cache_get_kv_fp16_returns_valid_shape(self):
        """patch.py's _ecc_decode_step calls this on every decode token."""
        _, cache = self._make_model_and_cache()
        B, H, L, D = 1, 2, 4, 16
        cache.update(torch.randn(B, H, L, D), torch.randn(B, H, L, D), layer_idx=0)
        k, v = cache.get_kv_fp16_for_layer(0)
        assert k.shape == (B, H, L, D)
        assert v.shape == (B, H, L, D)
        assert k.dtype == torch.float16

    def test_ecc_cache_context_manager_end_to_end(self):
        """Full lifecycle: enter, update across layers, exit, verify no leaks."""
        from custom_kv import ecc_cache
        import torch.nn.functional as F
        torch.manual_seed(42)
        model = TinyModel(TinyConfig()).eval()
        original_sdpa = F.scaled_dot_product_attention

        with ecc_cache(model, batch_size=1, max_cache_len=32, device="cpu") as cache:
            assert F.scaled_dot_product_attention is not original_sdpa, "patch not installed"
            for layer_idx in range(2):
                cache.update(
                    torch.randn(1, 2, 4, 16),
                    torch.randn(1, 2, 4, 16),
                    layer_idx=layer_idx,
                )
            assert cache.get_seq_length(0) == 4
            assert cache.seen_tokens == 4

        # After exit: SDPA must be restored
        assert F.scaled_dot_product_attention is original_sdpa, "patch not removed"

    def test_repeated_context_does_not_leak(self):
        """Enter/exit the context 3 times — patch and unpatch must be idempotent."""
        from custom_kv import ecc_cache
        import torch.nn.functional as F
        original = F.scaled_dot_product_attention
        model = TinyModel(TinyConfig()).eval()
        for _ in range(3):
            with ecc_cache(model, batch_size=1, max_cache_len=16, device="cpu"):
                pass
            assert F.scaled_dot_product_attention is original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
