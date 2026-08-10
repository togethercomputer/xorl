"""Unit tests for DSv4 KV compression shape constraints."""

from types import SimpleNamespace

import pytest
import torch

from xorl.ops.dsv4.compressor import DeepSeekV4Compressor
from xorl.ops.dsv4.rope import precompute_freqs_cis


pytestmark = pytest.mark.cpu


class _FakeCPGroup:
    def size(self):
        return 2

    def rank(self):
        return 1


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


def test_c128_context_parallel_only_requires_ratio_divisibility(monkeypatch):
    monkeypatch.delenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", raising=False)
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


def test_c4_context_parallel_keeps_overlap_divisibility_guard(monkeypatch):
    monkeypatch.delenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", raising=False)
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
