"""
test_v5_cache_contract.py — Verify ErrorCorrectedCache satisfies transformers v5 Cache contract.

The v5 refactor (PRs #41378, #43168, #43679) made Cache a container of per-layer
CacheLayerMixin objects. ErrorCorrectedCache must provide enough of that contract
that model.generate(past_key_values=cache) doesn't AttributeError on internal calls.
"""
import pytest
import torch


pytest.importorskip("transformers", reason="transformers required for v5 contract test")


class TinyConfig:
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 64
    head_dim = 16


class TestV5CacheContract:
    """The minimum surface generate() needs from past_key_values on transformers >=4.45."""

    def _cache(self):
        from custom_kv.cache import ErrorCorrectedCache
        return ErrorCorrectedCache(TinyConfig(), batch_size=1, max_cache_len=32, device="cpu")

    def test_has_layers_attribute(self):
        """v5 Cache.is_compileable reads self.layers. Missing -> AttributeError in generate()."""
        cache = self._cache()
        assert hasattr(cache, "layers"), "v5 Cache requires self.layers attribute"
        assert isinstance(cache.layers, list)

    def test_is_compileable_returns_false(self):
        """ErrorCorrectedCache uses runtime SDPA patching — not torch.compile compatible."""
        cache = self._cache()
        assert cache.is_compileable is False

    def test_get_mask_sizes_returns_tuple(self):
        """v5 attention mask construction calls cache.get_mask_sizes(query_length, layer_idx)."""
        cache = self._cache()
        cache.update(torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16), layer_idx=0)
        sizes = cache.get_mask_sizes(query_length=1, layer_idx=0)
        assert isinstance(sizes, tuple) and len(sizes) == 2
        assert sizes == (8, 0)

    def test_reset_clears_state(self):
        """generate() calls cache.reset() between multi-batch runs."""
        cache = self._cache()
        cache.update(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16), layer_idx=0)
        cache.update(torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16), layer_idx=1)
        assert cache.get_seq_length(0) == 5
        cache.reset()
        assert cache.get_seq_length(0) == 0
        assert cache.get_seq_length(1) == 0
        assert cache.seen_tokens == 0

    def test_crop_truncates(self):
        """generate() calls cache.crop(N) on context overflow."""
        cache = self._cache()
        cache.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16), layer_idx=0)
        cache.update(torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16), layer_idx=1)
        assert cache.get_seq_length(0) == 10
        cache.crop(7)
        assert cache.get_seq_length(0) == 7
        assert cache.get_seq_length(1) == 7

    def test_subclass_relation(self):
        """generation/utils.py:2149 does `isinstance(cache, Cache)`. Must hold."""
        from transformers.cache_utils import Cache
        cache = self._cache()
        assert isinstance(cache, Cache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
