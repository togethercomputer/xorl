"""Unit tests for DSv4 RoPE cache sizing."""

from types import SimpleNamespace

import pytest
import torch

from xorl.ops.dsv4.cp_utils import get_freqs_cis_for_cp
from xorl.ops.dsv4.rope import precompute_freqs_cis, wrapped_precompute_freqs_cis


pytestmark = pytest.mark.cpu


def _rope_config(max_position_embeddings=32):
    return SimpleNamespace(
        max_position_embeddings=max_position_embeddings,
        rope_parameters={
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    )


def test_wrapped_precompute_freqs_cis_uses_config_max_position_embeddings(monkeypatch):
    monkeypatch.delenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", raising=False)
    precompute_freqs_cis.cache_clear()

    freqs = wrapped_precompute_freqs_cis(_rope_config(max_position_embeddings=40), rope_head_dim=8, base=10000.0)

    assert freqs.shape == (40, 4)


def test_wrapped_precompute_freqs_cis_env_override_wins(monkeypatch):
    monkeypatch.setenv("XORL_DSV4_ROPE_MAX_SEQ_LEN", "12")
    precompute_freqs_cis.cache_clear()

    freqs = wrapped_precompute_freqs_cis(_rope_config(max_position_embeddings=40), rope_head_dim=8, base=10000.0)

    assert freqs.shape == (12, 4)


def test_get_freqs_cis_for_cp_errors_when_cache_too_short():
    class _FakeGroup:
        def rank(self):
            return 2

    freqs = torch.empty(10, 4)

    with pytest.raises(ValueError, match="RoPE cache is too short"):
        get_freqs_cis_for_cp(freqs, seqlen_local=8, cp_size=4, cp_group=_FakeGroup())
