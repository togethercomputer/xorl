"""Direct tests for the Reducer abstraction.

The contract: a reducer is a closure over a caller-supplied denominator, so its
outputs are partial shares — they sum across micro-batches (and across ranks
under all_reduce(SUM)) to the globally-correct value.
"""

import pytest
import torch

from tests.ops.loss.conftest import assert_close
from xorl.ops.loss import SequencePartial, TokenPartial


@pytest.mark.parametrize(
    "scale_fn",
    [
        pytest.param(lambda mask: mask.sum(), id="active_count"),
        pytest.param(lambda mask: torch.tensor(float(mask.numel())), id="numel"),
        pytest.param(lambda mask: torch.tensor(1.0), id="ones_raw_sum"),
    ],
)
def test_token_partial_shares_sum(scale_fn):
    """Per-microbatch TokenPartial shares sum to the single-batch result."""
    torch.manual_seed(0)
    B, S = 4, 6
    values = torch.randn(B, S)
    mask = torch.randint(0, 2, (B, S)).float()

    reducer = TokenPartial(scale=scale_fn(mask))
    single = reducer(values, mask)
    summed = sum(reducer(values[b : b + 1], mask[b : b + 1]) for b in range(B))

    assert_close(summed, single)


def test_token_partial_scale_one_equals_raw_sum():
    """``TokenPartial(scale=1)`` is the raw masked sum (deferred-divide form)."""
    torch.manual_seed(0)
    B, S = 4, 6
    values = torch.randn(B, S)
    mask = torch.randint(0, 2, (B, S)).float()

    out = TokenPartial(scale=torch.tensor(1.0))(values, mask)
    assert_close(out, (values * mask).sum())


def test_sequence_partial_shares_sum():
    """Per-microbatch SequencePartial shares (with sliced cu_seqlens_local) sum to the single-batch result."""
    torch.manual_seed(0)
    B, S = 4, 6
    values = torch.randn(B, S)
    mask = torch.randint(0, 2, (B, S)).float()
    seq_lengths = mask.sum(dim=-1)
    seq_count = torch.tensor(float(B))
    full_cu_seqlens = torch.arange(0, B * S + 1, S)
    mb_cu_seqlens = torch.arange(0, S + 1, S)

    single = SequencePartial(
        scale=seq_count,
        cu_seqlens_local=full_cu_seqlens,
        seq_lengths_global=seq_lengths,
    )(values, mask)
    summed = sum(
        SequencePartial(
            scale=seq_count,
            cu_seqlens_local=mb_cu_seqlens,
            seq_lengths_global=seq_lengths[b : b + 1],
        )(values[b : b + 1], mask[b : b + 1])
        for b in range(B)
    )

    assert_close(summed, single)


def test_sequence_partial_scale_one_equals_sum_of_per_seq_means():
    """``SequencePartial(scale=1)`` is the deferred-outer-divide form of SequencePartial(scale=n_seqs)."""
    torch.manual_seed(0)
    B, S = 4, 6
    values = torch.randn(B, S)
    mask = torch.randint(0, 2, (B, S)).float()
    seq_lengths = mask.sum(dim=-1)
    seq_count = torch.tensor(float(B))
    cu_seqlens_local = torch.arange(0, B * S + 1, S)

    deferred = SequencePartial(
        scale=torch.tensor(1.0),
        cu_seqlens_local=cu_seqlens_local,
        seq_lengths_global=seq_lengths,
    )(values, mask)
    finalized = SequencePartial(
        scale=seq_count,
        cu_seqlens_local=cu_seqlens_local,
        seq_lengths_global=seq_lengths,
    )(values, mask)

    assert_close(deferred / seq_count, finalized)


@pytest.mark.parametrize(
    "reducer",
    [
        pytest.param(TokenPartial(scale=torch.tensor(0.0)), id="token"),
        pytest.param(
            SequencePartial(
                scale=torch.tensor(0.0),
                cu_seqlens_local=torch.tensor([0, 4, 8]),
                seq_lengths_global=torch.zeros(2),
            ),
            id="sequence",
        ),
    ],
)
def test_empty_mask_yields_zero(reducer):
    """Zero denominators clamp to 1 and produce 0, not NaN."""
    values = torch.randn(2, 4)
    mask = torch.zeros(2, 4)
    assert reducer(values, mask) == 0.0


def test_sequence_partial_packed_row_matches_per_segment_mean():
    """One row, three packed segments described by ``cu_seqlens`` — sum of per-segment means / n_seqs."""
    torch.manual_seed(0)
    # Row of 10 tokens packing three segments of lengths [3, 4, 3].
    values = torch.randn(1, 10)
    mask = torch.ones(1, 10)
    cu_seqlens = torch.tensor([0, 3, 7, 10])
    seg_lengths = torch.tensor([3, 4, 3])
    n_seqs = torch.tensor(float(seg_lengths.numel()))

    out = SequencePartial(
        scale=n_seqs,
        cu_seqlens_local=cu_seqlens,
        seq_lengths_global=seg_lengths,
    )(values, mask)

    flat = (values * mask).flatten()
    expected = (flat[0:3].sum() / 3 + flat[3:7].sum() / 4 + flat[7:10].sum() / 3) / n_seqs

    assert_close(out, expected)


def test_sequence_partial_packed_cp_shares_sum():
    """Packed row split across two CP shards: per-shard partials sum to the single-batch result.

    Layout: one row of 10 tokens with three packed segments of pre-shard lengths
    [3, 4, 3]. CP=2 splits the row at column 5:

    - Shard 0 covers columns [0, 5): segment 0 lives wholly here ([0, 3)),
      and the first half of segment 1 ([3, 5), local length 2 of the
      pre-shard 4).
    - Shard 1 covers columns [5, 10): the second half of segment 1
      ([5, 7), local length 2 of 4) and segment 2 wholly ([7, 10)).

    Both shards reference the same ``seq_lengths_global`` for any segment
    they touch. Segment 1's contributions from the two shards each divide
    by 4, then sum to the correct full-segment mean.
    """
    torch.manual_seed(0)
    values = torch.randn(1, 10)
    mask = torch.ones(1, 10)
    seg_lengths_global = torch.tensor([3, 4, 3])
    n_seqs = torch.tensor(float(seg_lengths_global.numel()))

    full = SequencePartial(
        scale=n_seqs,
        cu_seqlens_local=torch.tensor([0, 3, 7, 10]),
        seq_lengths_global=seg_lengths_global,
    )(values, mask)

    shard_0 = SequencePartial(
        scale=n_seqs,
        cu_seqlens_local=torch.tensor([0, 3, 5]),
        seq_lengths_global=seg_lengths_global[:2],
    )(values[:, :5], mask[:, :5])

    shard_1 = SequencePartial(
        scale=n_seqs,
        cu_seqlens_local=torch.tensor([0, 2, 5]),
        seq_lengths_global=seg_lengths_global[1:],
    )(values[:, 5:], mask[:, 5:])

    assert_close(shard_0 + shard_1, full)


def test_token_partial_with_n_seqs_scale_equals_seq_mean_token_sum():
    """``TokenPartial(scale=n_seqs)`` expresses verl's seq-mean-token-sum policy."""
    torch.manual_seed(0)
    B, S = 4, 6
    values = torch.randn(B, S)
    mask = torch.randint(0, 2, (B, S)).float()
    seq_count = torch.tensor(float(B))

    via_token_partial = TokenPartial(scale=seq_count)(values, mask)
    direct = (values * mask).sum(dim=-1).sum() / seq_count

    assert_close(via_token_partial, direct)
