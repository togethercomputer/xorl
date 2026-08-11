"""Unit tests for DSv4 KV compression shape constraints."""

from types import SimpleNamespace

import pytest
import torch

from xorl.ops.dsv4 import utils
from xorl.ops.dsv4.compressor import DeepSeekV4Compressor
from xorl.ops.dsv4.exact_attention import _legacy_compressor_state_pages
from xorl.ops.dsv4.rope import precompute_freqs_cis


pytestmark = pytest.mark.cpu


class _FakeCPGroup:
    def size(self):
        return 2

    def rank(self):
        return 1


def test_exact_legacy_c4_state_reserves_both_overlap_ring_pages():
    assert _legacy_compressor_state_pages(4) == 2
    assert _legacy_compressor_state_pages(128) == 1
    with pytest.raises(ValueError, match="Unsupported DSV4 compression ratio"):
        _legacy_compressor_state_pages(8)


def _compressor_config(max_position_embeddings=768):
    return SimpleNamespace(
        hidden_size=32,
        qk_rope_head_dim=8,
        rms_norm_eps=1e-6,
        compress_rope_theta=10000.0,
        max_position_embeddings=max_position_embeddings,
        rope_parameters={
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )


def test_context_parallel_compression_ratio_admission_policy(monkeypatch):
    precompute_freqs_cis.cache_clear()

    compressor = DeepSeekV4Compressor(
        _compressor_config(),
        head_dim=16,
        compress_ratio=128,
        rotate=False,
        cp_group=_FakeCPGroup(),
    )
    with torch.no_grad():
        compressor.wkv.weight.fill_(0.01)
        compressor.wgate.weight.zero_()
        compressor.ape.zero_()

    out = compressor.forward_raw(torch.ones(1, 384, 32))

    assert out.shape == (1, 3, 16)

    # Exercise the cache-capacity guard through its production CP consumer.
    short_cache_compressor = DeepSeekV4Compressor(
        _compressor_config(max_position_embeddings=600),
        head_dim=16,
        compress_ratio=128,
        rotate=False,
        cp_group=_FakeCPGroup(),
    )
    with pytest.raises(ValueError, match="RoPE cache is too short"):
        short_cache_compressor.forward_raw(torch.ones(1, 384, 32))

    _assert_c4_context_parallel_keeps_overlap_divisibility_guard(monkeypatch)
    _assert_rotate_activation_fallback_is_orthonormal(monkeypatch)


def _assert_c4_context_parallel_keeps_overlap_divisibility_guard(monkeypatch):
    precompute_freqs_cis.cache_clear()

    compressor = DeepSeekV4Compressor(
        _compressor_config(max_position_embeddings=24),
        head_dim=16,
        compress_ratio=4,
        rotate=False,
        cp_group=_FakeCPGroup(),
    )

    with pytest.raises(AssertionError, match="overlap=True"):
        compressor.forward_raw(torch.ones(1, 12, 32))


def _assert_rotate_activation_fallback_is_orthonormal(monkeypatch):
    monkeypatch.setattr(utils, "_fast_hadamard_transform", None)
    pattern = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(utils.rotate_activation(pattern), torch.full_like(pattern, 0.5))

    torch.manual_seed(0)
    for width in (4, 16, 64, 256):
        values = torch.randn(3, width, dtype=torch.bfloat16)
        rotated = utils.rotate_activation(values)
        torch.testing.assert_close(utils.rotate_activation(rotated), values, atol=8e-3, rtol=8e-3)
        input_norm = values.float().pow(2).sum(dim=-1).sqrt()
        output_norm = rotated.float().pow(2).sum(dim=-1).sqrt()
        torch.testing.assert_close(input_norm, output_norm, atol=1e-2, rtol=1e-2)
