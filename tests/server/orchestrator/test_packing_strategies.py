"""Tests for the configurable packing strategies and oversized-sample policy.

Covers the server packer/dispatcher redesign:
  - on_oversized = error (default) / skip / truncate
  - strategy = sequential (legacy) / best_fit / balanced_dp
  - dp_size-aware balanced packing (N == k*dp_size, zero dummies, balanced bins)
  - per-token correctness invariants (position_ids document resets, valid-token
    count) preserved across every strategy
  - determinism and datum_order (routed_experts realignment)
"""

import math
import random

import pytest
import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.server.orchestrator.packing import (
    PACKING_STRATEGIES,
    SequentialPacker,
    pack_samples,
)
from xorl.utils.seqlen_pos_transform_utils import pos2culen


pytestmark = [pytest.mark.cpu, pytest.mark.server]


# ============================================================================
# Helpers
# ============================================================================


def _make_data(n, seed=0, lo=200, hi=1800):
    """Already-shifted (xorl_client-style) datums of varied length."""
    rng = random.Random(seed)
    data = []
    for i in range(n):
        length = rng.randint(lo, hi)
        data.append(
            {
                "input_ids": list(range(length)),
                "target_tokens": list(range(length)),
            }
        )
    return data


def _row_lengths(batches):
    return [len(b["input_ids"][0]) for b in batches]


def _total_valid_tokens(batches):
    total = 0
    for b in batches:
        for tok in b["labels"][0]:
            if tok != IGNORE_INDEX:
                total += 1
    return total


def _dispatcher_dummies(num_rows, dp_size):
    """Replicates RunnerDispatcher: rounds*dp_size - num_rows dummy slots."""
    rounds = math.ceil(num_rows / dp_size) if num_rows else 0
    return rounds * dp_size - num_rows


# ============================================================================
# strategy plumbing
# ============================================================================


def test_strategy_admission_correctness_and_determinism_policy():
    with pytest.raises(ValueError, match="Unknown packing strategy"):
        SequentialPacker(strategy="nope")
    with pytest.raises(ValueError, match="Unknown on_oversized"):
        SequentialPacker(on_oversized="nope")

    _assert_oversized_error_is_default()
    _assert_oversized_skip_matches_legacy_drop()
    _assert_oversized_truncate_clips_aligned_fields()
    _assert_oversized_truncate_applies_hf_shift()
    _assert_strategy_correctness_and_utilization_policy()
    _assert_strategy_determinism_and_datum_order_policy()


# ============================================================================
# on_oversized policy (change C — no silent data loss)
# ============================================================================


def _assert_oversized_error_is_default():
    """A sample longer than max_seq_len must fail loud by default."""
    packer = SequentialPacker(log_stats=False, pad_to_multiple_of=1)
    data = [{"input_ids": [1, 2, 3]}, {"input_ids": [1] * 100}]
    with pytest.raises(ValueError, match="exceeding max_seq_len"):
        packer.pack(data, max_seq_len=10)


def _assert_oversized_skip_matches_legacy_drop():
    packer = SequentialPacker(log_stats=False, pad_to_multiple_of=1, on_oversized="skip")
    data = [
        {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
        {"input_ids": [1] * 100, "labels": [1] * 100},
        {"input_ids": [4, 5], "labels": [5, 6]},
    ]
    batches = packer.pack(data, max_seq_len=10)
    assert sum(b["num_samples"] for b in batches) == 2  # oversized dropped


def _assert_oversized_truncate_clips_aligned_fields():
    packer = SequentialPacker(log_stats=False, pad_to_multiple_of=1, on_oversized="truncate")
    data = [
        {
            "input_ids": list(range(100)),
            "target_tokens": list(range(100)),
            "teacher_ids": [7] * 100,
            "teacher_weights": [0.5] * 100,
        }
    ]
    batches = packer.pack(data, max_seq_len=10)
    assert len(batches) == 1
    b = batches[0]
    # Truncated to 10 tokens (no HF shift for already-shifted datums).
    assert len(b["input_ids"][0]) == 10
    assert len(b["labels"][0]) == 10
    assert b["teacher_ids"][0] == [7] * 10
    assert b["teacher_weights"][0] == [0.5] * 10


def _assert_oversized_truncate_applies_hf_shift():
    """HF-format (labels == input length) still shifts after truncation."""
    packer = SequentialPacker(log_stats=False, pad_to_multiple_of=1, on_oversized="truncate")
    data = [{"input_ids": list(range(100)), "labels": list(range(100))}]
    batches = packer.pack(data, max_seq_len=10)
    # truncate to 10, then HF shift drops one -> 9 tokens
    assert len(batches[0]["input_ids"][0]) == 9


# ============================================================================
# sequential strategy is byte-identical to the legacy packer
# ============================================================================


def _assert_sequential_is_legacy_layout():
    """Sequential strategy must reproduce greedy first-fit exactly."""
    data = _make_data(40, seed=3)
    seq = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="sequential").pack(data, max_seq_len=4096)
    # Reproduce the legacy first-fit by hand on raw lengths.
    expected_rows = []
    cur, cur_tok = [], 0
    for i, d in enumerate(data):
        L = len(d["input_ids"])
        if cur and cur_tok + L > 4096:
            expected_rows.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += L
    if cur:
        expected_rows.append(cur)
    assert len(seq) == len(expected_rows)
    # num_samples per row matches
    assert [b["num_samples"] for b in seq] == [len(r) for r in expected_rows]


# ============================================================================
# correctness invariants preserved across ALL strategies
# ============================================================================


def _documents_from_batches(batches):
    """Split packed rows back into per-document (input_ids, labels) tuples using
    position_id resets — the boundaries the model's varlen attention uses."""
    docs = []
    for b in batches:
        ii = b["input_ids"][0]
        lb = b["labels"][0]
        pos = b["position_ids"][0]
        start = 0
        for j in range(1, len(pos) + 1):
            if j == len(pos) or pos[j] == 0:
                docs.append((tuple(ii[start:j]), tuple(lb[start:j])))
                start = j
    return docs


def _assert_strategy_correctness_and_utilization_policy():
    """The K3 precondition: reordering changes only the GROUPING of documents into
    rows, never the per-document token content, labels, or position spans.

    With per-document varlen attention + position resets + global-valid-token loss
    normalization, an identical document multiset means per-token logits/loss are
    identical and the only possible numeric difference is float reduction order
    (within K3's 1e-3/1e-2 tolerance). pad=1 here to avoid padding "documents".
    """
    _assert_sequential_is_legacy_layout()

    data = _make_data(50, seed=23)
    seq = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="sequential").pack(data, max_seq_len=4096)
    for strategy in ("best_fit", "balanced_dp"):
        out = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy=strategy, dp_size=8).pack(
            data, max_seq_len=4096
        )
        assert sorted(_documents_from_batches(out)) == sorted(_documents_from_batches(seq)), strategy

    _assert_valid_token_count_is_strategy_invariant()
    _assert_position_ids_reset_at_document_boundaries()
    _assert_no_row_exceeds_capacity()
    _assert_best_fit_never_uses_more_rows_than_sequential()


def _assert_valid_token_count_is_strategy_invariant():
    """Total valid (non-ignore) tokens must not depend on packing strategy.

    This is the FLOPs-numerator / loss-normalization invariant: reordering which
    sample lands in which row changes the grouping, never the token set.
    """
    data = _make_data(48, seed=7)
    baseline = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="sequential").pack(
        data, max_seq_len=8192
    )
    for strategy in PACKING_STRATEGIES:
        out = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy=strategy, dp_size=8).pack(
            data, max_seq_len=8192
        )
        assert _total_valid_tokens(out) == _total_valid_tokens(baseline), strategy
        assert sum(b["num_samples"] for b in out) == len(data), strategy


def _assert_position_ids_reset_at_document_boundaries():
    """position_ids must reset to 0 at each sample boundary (FLOPs/attention)."""
    data = _make_data(20, seed=9, lo=50, hi=400)
    for strategy in PACKING_STRATEGIES:
        batches = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy=strategy, dp_size=4).pack(
            data, max_seq_len=2048
        )
        for b in batches:
            pos = b["position_ids"][0]
            assert pos[0] == 0, strategy
            for j in range(1, len(pos)):
                assert pos[j] == 0 or pos[j] == pos[j - 1] + 1, strategy
            cu = pos2culen(torch.tensor(pos))
            assert cu[0].item() == 0, strategy
            assert cu[-1].item() == len(pos), strategy
            assert all(cu[k + 1] > cu[k] for k in range(len(cu) - 1)), strategy


def _assert_no_row_exceeds_capacity():
    data = _make_data(50, seed=11)
    pack_len = 4096
    for strategy in PACKING_STRATEGIES:
        batches = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy=strategy, dp_size=8).pack(
            data, max_seq_len=pack_len
        )
        for length in _row_lengths(batches):
            assert length <= pack_len, strategy


# ============================================================================
# best_fit (change B — utilization)
# ============================================================================


def _assert_best_fit_never_uses_more_rows_than_sequential():
    """Best-fit-decreasing minimizes bins -> rows <= sequential for same pack_len."""
    for seed in range(5):
        data = _make_data(60, seed=seed)
        seq = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="sequential").pack(
            data, max_seq_len=4096
        )
        bf = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="best_fit").pack(data, max_seq_len=4096)
        assert len(bf) <= len(seq)


# ============================================================================
# balanced_dp (change A — zero dummies + balanced)
# ============================================================================


def test_balanced_dp_policy():
    """N == k*dp_size and dispatcher needs zero dummy batches."""
    dp_size = 8
    data = _make_data(40, seed=5)
    batches = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=dp_size).pack(
        data, max_seq_len=8192
    )
    assert len(batches) % dp_size == 0
    assert _dispatcher_dummies(len(batches), dp_size) == 0

    _assert_balanced_dp_bins_are_balanced()
    _assert_balanced_dp_full_rows_in_large_batch_regime()
    _assert_balanced_dp_falls_back_when_fewer_samples_than_dp()
    _assert_balanced_dp_size_one_equals_best_fit()


def _assert_balanced_dp_bins_are_balanced():
    """Rank load imbalance (max/mean) close to 1.0."""
    dp_size = 8
    data = _make_data(80, seed=13)
    batches = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=dp_size).pack(
        data, max_seq_len=16384
    )
    lengths = _row_lengths(batches)
    mean = sum(lengths) / len(lengths)
    assert max(lengths) / mean < 1.20  # well-balanced
    assert len(batches) % dp_size == 0


def _assert_balanced_dp_full_rows_in_large_batch_regime():
    """When total_tokens >= dp_size*pack_len, balanced_dp yields N==dp_size FULL rows.

    This is the regime where the change is a real win: every rank busy AND rows
    near the GEMM knee.
    """
    dp_size = 8
    pack_len = 4096
    # ~ dp_size * pack_len total tokens spread over many samples.
    data = _make_data(256, seed=21, lo=900, hi=1100)
    total = sum(len(d["input_ids"]) for d in data)
    assert total >= dp_size * pack_len  # large-batch regime
    batches = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=dp_size).pack(
        data, max_seq_len=pack_len
    )
    assert len(batches) % dp_size == 0
    assert _dispatcher_dummies(len(batches), dp_size) == 0
    util = sum(_row_lengths(batches)) / (len(batches) * pack_len)
    assert util > 0.90  # rows are nearly full


def _assert_balanced_dp_falls_back_when_fewer_samples_than_dp():
    """num_samples < dp_size cannot fill every rank; degrade gracefully (no crash)."""
    dp_size = 32
    data = _make_data(5, seed=1)
    batches = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=dp_size).pack(
        data, max_seq_len=8192
    )
    assert sum(b["num_samples"] for b in batches) == 5
    assert len(batches) >= 1


def _assert_balanced_dp_size_one_equals_best_fit():
    data = _make_data(30, seed=4)
    bd = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=1).pack(
        data, max_seq_len=4096
    )
    bf = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy="best_fit").pack(data, max_seq_len=4096)
    assert [b["num_samples"] for b in bd] == [b["num_samples"] for b in bf]


# ============================================================================
# determinism
# ============================================================================


def _assert_strategy_determinism_and_datum_order_policy():
    data = _make_data(50, seed=17)
    for strategy in PACKING_STRATEGIES:
        kwargs = dict(log_stats=False, pad_to_multiple_of=1, strategy=strategy, dp_size=8)
        a = SequentialPacker(**kwargs).pack(data, max_seq_len=4096)
        b = SequentialPacker(**kwargs).pack(data, max_seq_len=4096)
        assert [x["input_ids"] for x in a] == [x["input_ids"] for x in b], strategy
        assert [x["position_ids"] for x in a] == [x["position_ids"] for x in b], strategy

    _assert_sequential_datum_order_is_identity()
    _assert_reordered_datum_order_matches_build_order()


# ============================================================================
# datum_order (routed_experts realignment after reorder)
# ============================================================================


def _assert_sequential_datum_order_is_identity():
    data = _make_data(20, seed=8)
    batches, order = pack_samples(
        data, max_seq_len=4096, request_id="t", pad_to_multiple_of=1, strategy="sequential", return_datum_order=True
    )
    assert order == list(range(20))


def _assert_reordered_datum_order_matches_build_order():
    data = _make_data(40, seed=6)
    for strategy in ("best_fit", "balanced_dp"):
        packer = SequentialPacker(log_stats=False, pad_to_multiple_of=1, strategy=strategy, dp_size=8)
        batches = packer.pack(data, max_seq_len=4096)
        order = packer.last_datum_order
        assert sorted(order) == list(range(40)), strategy

        packed_document_lengths = []
        for batch in batches:
            boundaries = pos2culen(torch.tensor(batch["position_ids"][0])).tolist()
            packed_document_lengths.extend(end - start for start, end in zip(boundaries, boundaries[1:]))
        expected_lengths = [len(data[index]["input_ids"]) for index in order]
        assert packed_document_lengths == expected_lengths, strategy
