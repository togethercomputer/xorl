"""Runtime tests for the Ulysses byte-contract surface (CPU-only).

Covers missing Q-head divisibility, the implicit sync-versus-async strategy
choice that can relocate RoPE across the all-to-all, and the GDN backend
fallback under the exact contract.
The bitwise gates themselves are GPU tests (kernel head-bucket gate and the
U1-vs-U8 fixture gate).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import xorl.distributed.parallel_state as parallel_state_module
import xorl.ops.linear_attention.backend as gdn_backend
from xorl.distributed.sequence_parallel.strategy import (
    NoopStrategy,
    UlyssesAsyncStrategy,
    UlyssesSyncStrategy,
    get_cp_strategy,
)
from xorl.ops.linear_attention.modules.bi_contract import gdn_contract


def _fake_state(*, ulysses: bool = True, ring: bool = False, size: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        cp_enabled=ulysses or ring,
        ulysses_enabled=ulysses,
        ringattn_enabled=ring,
        ulysses_size=size,
        ulysses_group=None,
        ringattn_group=None,
    )


@pytest.fixture
def ulysses8_state(monkeypatch):
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_state())


# ---------------------------------------------------------------------------
# Explicit strategy selection (RoPE placement is bit-relevant)
# ---------------------------------------------------------------------------


def test_explicit_variant_overrides_the_kv_heads_heuristic(ulysses8_state):
    # The historical heuristic sends num_kv_heads >= ulysses_size to ASYNC —
    # which applies RoPE after the all-to-all. An exact contract pinning "sync"
    # must win regardless of num_kv_heads.
    assert isinstance(get_cp_strategy(num_kv_heads=64, variant="sync"), UlyssesSyncStrategy)
    assert isinstance(get_cp_strategy(variant="async"), UlyssesAsyncStrategy)


def test_auto_variant_keeps_the_historical_heuristic(ulysses8_state):
    assert isinstance(get_cp_strategy(num_kv_heads=8), UlyssesAsyncStrategy)
    assert isinstance(get_cp_strategy(num_kv_heads=2), UlyssesSyncStrategy)
    assert isinstance(get_cp_strategy(), UlyssesSyncStrategy)


def test_unknown_variant_raises(ulysses8_state):
    with pytest.raises(ValueError, match="Unknown CP strategy variant"):
        get_cp_strategy(variant="bogus")


def test_explicit_variant_rejects_hybrid_ring(monkeypatch):
    monkeypatch.setattr(parallel_state_module, "get_parallel_state", lambda: _fake_state(ring=True))
    with pytest.raises(NotImplementedError, match="hybrid Ulysses\\+Ring"):
        get_cp_strategy(variant="sync")


def test_variant_is_moot_when_cp_disabled(monkeypatch):
    monkeypatch.setattr(
        parallel_state_module,
        "get_parallel_state",
        lambda: _fake_state(ulysses=False, ring=False),
    )
    assert isinstance(get_cp_strategy(variant="sync"), NoopStrategy)


# ---------------------------------------------------------------------------
# Head divisibility fails closed BEFORE any collective
# ---------------------------------------------------------------------------


class _StubAttention:
    def __init__(self, q_heads: int, kv_heads: int, seq_len: int = 16, head_dim: int = 4):
        self._q = torch.zeros(1, seq_len, q_heads, head_dim, dtype=torch.bfloat16)
        self._k = torch.zeros(1, seq_len, kv_heads, head_dim, dtype=torch.bfloat16)
        self._v = torch.zeros(1, seq_len, kv_heads, head_dim, dtype=torch.bfloat16)

    def _project_qkv(self, hidden_states, position_embeddings):
        return self._q, self._k, self._v


def test_uneven_q_head_split_raises_before_comm():
    strategy = UlyssesSyncStrategy(group=None, ulysses_size=8)
    with pytest.raises(ValueError, match="num_attention_heads \\(6\\) to be divisible"):
        strategy.project_qkv(_StubAttention(q_heads=6, kv_heads=8), None, None)


def test_non_divisor_kv_heads_raise_before_comm():
    # Promoted from a bare assert (which vanishes under -O) to a ValueError;
    # ordering: the Q-head check passes first, then GQA replication refuses.
    strategy = UlyssesSyncStrategy(group=None, ulysses_size=8)
    with pytest.raises(ValueError, match="num_key_value_heads \\(3\\) for GQA replication"):
        strategy.project_qkv(_StubAttention(q_heads=8, kv_heads=3), None, None)


# ---------------------------------------------------------------------------
# GDN backend fallback under the exact contract
# ---------------------------------------------------------------------------


def test_gdn_cp_fallback_raises_under_exact_contract():
    with gdn_contract(True):
        with pytest.raises(RuntimeError, match="silently swapping the kernel program is not admitted"):
            gdn_backend.warn_cp_fallback_once()


def test_gdn_cp_fallback_still_warns_outside_the_contract(monkeypatch):
    monkeypatch.setattr(gdn_backend, "_warned_cp_fallback", False)
    with pytest.warns(UserWarning, match="FlashQLA requires 128-dim heads"):
        gdn_backend.warn_cp_fallback_once()
