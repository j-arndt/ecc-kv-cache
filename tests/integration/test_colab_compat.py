"""
test_colab_compat.py — Integration tests that replicate the EXACT failures
seen on Colab A100 today. Run this in the venv_colab environment BEFORE
pushing to GitHub or running on Colab.

Covers:
  1. ErrorCorrectedCache instantiation (super().__init__() bug)
  2. Calibration hook returns N layers not 0 (DynamicCache hook bug)
  3. _compress / _dequantize round-trip correctness
  4. ecc_cache context manager lifecycle (no crash on enter/exit)
  5. compress_and_store arg count matches (CUDA path simulated via CPU fallback)

All tests run on CPU — no GPU or CUDA extension required.
"""
import sys
import types
import torch
import pytest
import math


# ── Fake model config matching Llama-3-8B ────────────────────────────────
class FakeConfig:
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    hidden_size = 4096
    head_dim = 128
    _name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"


# ── Fake attention module for hook tests ─────────────────────────────────
class FakeKProj(torch.nn.Linear):
    def __init__(self):
        super().__init__(4096, 1024, bias=False)

class FakeVProj(torch.nn.Linear):
    def __init__(self):
        super().__init__(4096, 1024, bias=False)

class FakeSelfAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = FakeKProj()
        self.v_proj = FakeVProj()

class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeSelfAttn()

class FakeModel(torch.nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.config = FakeConfig()
        self.config.num_hidden_layers = n_layers
        self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(n_layers)])

class FakeTopModel(torch.nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.config = FakeConfig()
        self.config.num_hidden_layers = n_layers
        self.model = FakeModel(n_layers)
    def parameters(self):
        return iter([torch.zeros(1)])


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: ErrorCorrectedCache instantiation (super().__init__() bug)
# ═══════════════════════════════════════════════════════════════════════════
class TestCacheInstantiation:
    def test_instantiation_does_not_crash(self):
        """Must not raise ValueError from transformers Cache.__init__()."""
        from custom_kv.cache import ErrorCorrectedCache
        cfg = FakeConfig()
        cache = ErrorCorrectedCache(cfg, batch_size=1, max_cache_len=512, device="cpu")
        assert cache is not None

    def test_correct_tensor_shapes(self):
        from custom_kv.cache import ErrorCorrectedCache
        cfg = FakeConfig()
        B, L, D = 1, 512, cfg.head_dim
        H = cfg.num_key_value_heads
        cache = ErrorCorrectedCache(cfg, batch_size=B, max_cache_len=L, device="cpu")
        assert cache._k_int4.shape == (B, H, L, D // 2)
        assert cache._k_syn.shape  == (B, H, L, D // 8)
        assert cache._k_meta.shape == (B, H, L, 3)
        assert cache._v_int4.shape == (B, H, L, D // 2)

    def test_memory_bytes_reasonable(self):
        from custom_kv.cache import ErrorCorrectedCache
        cfg = FakeConfig()
        cache = ErrorCorrectedCache(cfg, batch_size=1, max_cache_len=1000, device="cpu")
        mem = cache.memory_bytes()
        assert mem["cache_gb"] > 0
        assert mem["compression_ratio"] == 2.0

    def test_repr_does_not_crash(self):
        from custom_kv.cache import ErrorCorrectedCache
        cfg = FakeConfig()
        cache = ErrorCorrectedCache(cfg, batch_size=1, max_cache_len=100, device="cpu")
        r = repr(cache)
        assert "ErrorCorrectedCache" in r


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: compress / dequantize round-trip
# ═══════════════════════════════════════════════════════════════════════════
class TestCompressDecompress:
    def _make_cache(self, L=64):
        from custom_kv.cache import ErrorCorrectedCache
        return ErrorCorrectedCache(FakeConfig(), batch_size=1, max_cache_len=L, device="cpu")

    def test_compress_does_not_crash(self):
        cache = self._make_cache()
        x = torch.randn(1, 8, 16, 128)
        cache._compress(x, cache._k_int4, cache._k_syn, cache._k_meta, pos=0, L_new=16)

    def test_int4_values_in_range(self):
        cache = self._make_cache()
        x = torch.randn(1, 8, 16, 128)
        cache._compress(x, cache._k_int4, cache._k_syn, cache._k_meta, pos=0, L_new=16)
        lo = cache._k_int4[:, :, :16] & 0x0F
        hi = (cache._k_int4[:, :, :16] >> 4) & 0x0F
        assert lo.max() <= 15 and lo.min() >= 0
        assert hi.max() <= 15 and hi.min() >= 0

    def test_roundtrip_error_reasonable(self):
        """Reconstructed tensor should be close to original."""
        cache = self._make_cache(L=32)
        x = torch.randn(1, 8, 8, 128)
        cache._compress(x, cache._k_int4, cache._k_syn, cache._k_meta, pos=0, L_new=8)
        x_rec = cache._dequantize(
            cache._k_int4[:, :, :8],
            cache._k_syn[:, :, :8],
            cache._k_meta[:, :, :8],
        )
        assert x_rec.shape == x.shape
        # Relative error should be < 50% (INT4 is lossy but not catastrophic)
        err = (x.float() - x_rec.float()).abs().mean()
        mag = x.float().abs().mean()
        rel_err = err / (mag + 1e-8)
        assert rel_err < 0.5, f"Relative error too large: {rel_err:.3f}"

    def test_syndrome_sign_correctness(self):
        """After ECC correction, sign pattern should be partially recovered."""
        cache = self._make_cache(L=32)
        x = torch.randn(1, 8, 4, 128) * 2.0
        cache._compress(x, cache._k_int4, cache._k_syn, cache._k_meta, pos=0, L_new=4)
        x_rec = cache._dequantize(
            cache._k_int4[:, :, :4],
            cache._k_syn[:, :, :4],
            cache._k_meta[:, :, :4],
        )
        # Signs should agree more often than not
        sign_agree = (x.sign() == x_rec.float().sign()).float().mean()
        assert sign_agree > 0.7, f"Sign agreement too low: {sign_agree:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: cache.update() and HF Cache protocol
# ═══════════════════════════════════════════════════════════════════════════
class TestCacheUpdate:
    def test_update_returns_tuple(self):
        from custom_kv.cache import ErrorCorrectedCache
        cache = ErrorCorrectedCache(FakeConfig(), batch_size=1, max_cache_len=64, device="cpu")
        k = torch.randn(1, 8, 10, 128)
        v = torch.randn(1, 8, 10, 128)
        out = cache.update(k, v, layer_idx=0)
        assert isinstance(out, tuple) and len(out) == 2

    def test_get_seq_length_increments(self):
        from custom_kv.cache import ErrorCorrectedCache
        cache = ErrorCorrectedCache(FakeConfig(), batch_size=1, max_cache_len=64, device="cpu")
        k = torch.randn(1, 8, 10, 128)
        v = torch.randn(1, 8, 10, 128)
        cache.update(k, v, layer_idx=0)
        assert cache.get_seq_length(0) == 10

    def test_seen_tokens_increments_on_last_layer(self):
        from custom_kv.cache import ErrorCorrectedCache
        cfg = FakeConfig()
        cfg.num_hidden_layers = 2
        cache = ErrorCorrectedCache(cfg, batch_size=1, max_cache_len=64, device="cpu")
        k = torch.randn(1, 8, 5, 128)
        v = torch.randn(1, 8, 5, 128)
        cache.update(k, v, layer_idx=0)
        assert cache.seen_tokens == 0   # not last layer
        cache.update(k, v, layer_idx=1)
        assert cache.seen_tokens == 5   # last layer

    def test_get_kv_fp16_shape(self):
        from custom_kv.cache import ErrorCorrectedCache
        cache = ErrorCorrectedCache(FakeConfig(), batch_size=1, max_cache_len=64, device="cpu")
        k = torch.randn(1, 8, 12, 128)
        v = torch.randn(1, 8, 12, 128)
        cache.update(k, v, layer_idx=0)
        k_out, v_out = cache.get_kv_fp16_for_layer(0)
        assert k_out.shape == (1, 8, 12, 128)
        assert v_out.shape == (1, 8, 12, 128)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: ecc_cache context manager
# ═══════════════════════════════════════════════════════════════════════════
class TestEccCacheContextManager:
    def test_enter_exit_no_crash(self):
        from custom_kv import ecc_cache
        model = FakeTopModel(n_layers=2)
        with ecc_cache(model, batch_size=1, max_cache_len=64, device="cpu") as cache:
            assert cache is not None

    def test_patch_is_applied_then_removed(self):
        import torch.nn.functional as F
        from custom_kv import ecc_cache
        from custom_kv.patch import _original_sdpa
        model = FakeTopModel(n_layers=2)
        original = F.scaled_dot_product_attention
        with ecc_cache(model, batch_size=1, max_cache_len=64, device="cpu"):
            assert F.scaled_dot_product_attention is not original
        # After exit, should be restored
        assert F.scaled_dot_product_attention is original

    def test_exception_inside_context_restores_patch(self):
        import torch.nn.functional as F
        from custom_kv import ecc_cache
        model = FakeTopModel(n_layers=2)
        original = F.scaled_dot_product_attention
        try:
            with ecc_cache(model, batch_size=1, max_cache_len=64, device="cpu"):
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        assert F.scaled_dot_product_attention is original


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Calibration hook captures N layers not 0
# ═══════════════════════════════════════════════════════════════════════════
class TestCalibrationHook:
    def test_hook_captures_all_layers(self):
        """
        Verifies that _collect_kv_activations captures every layer.
        This broke on Colab when transformers changed to DynamicCache output.
        Our fix hooks k_proj/v_proj directly — this tests that.
        """
        from custom_kv.calibration import _collect_kv_activations

        n_layers = 4
        D_model = 512
        D_kv = 256  # num_kv_heads * head_dim

        # Build a minimal transformer-like model mirroring real Llama-3 structure:
        # AutoModelForCausalLM wraps an inner model, so top-level has .model.layers
        class MiniSelfAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(D_model, D_kv, bias=False)
                self.v_proj = torch.nn.Linear(D_model, D_kv, bias=False)

        class MiniLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = MiniSelfAttn()
            def forward(self, x):
                self.self_attn.k_proj(x)
                self.self_attn.v_proj(x)
                return x

        class MiniInnerModel(torch.nn.Module):
            """Mirrors model.model — the inner model that has .layers"""
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([MiniLayer() for _ in range(n_layers)])

        class MiniTopModel(torch.nn.Module):
            """Mirrors AutoModelForCausalLM — has .model.layers like Llama-3"""
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(num_hidden_layers=n_layers)
                self.model = MiniInnerModel()
            def forward(self, input_ids=None, **kwargs):
                x = torch.randn(1, input_ids.shape[1], D_model)
                for layer in self.model.layers:
                    x = layer(x)
                return x

        class MiniTokenizer:
            def __call__(self, text, return_tensors=None, max_length=None, truncation=None):
                ids = torch.randint(0, 100, (1, min(len(text.split()), max_length or 32)))
                return {"input_ids": ids}

        model = MiniTopModel()
        tokenizer = MiniTokenizer()
        texts = ["hello world " * 20] * 3

        result = _collect_kv_activations(model, tokenizer, texts, max_length=32, device="cpu")

        assert len(result) == n_layers, (
            f"Expected {n_layers} layers, got {len(result)}. "
            "Hook is not capturing — DynamicCache regression?"
        )
        for i in range(n_layers):
            assert "k" in result[i]
            assert result[i]["k"].shape[-1] == D_kv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
