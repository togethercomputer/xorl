"""DSV4-Flash EP combine: variable-row transport + shared canonical fold."""

import pytest
import torch

from xorl.distributed.canonical_moe import canonical_moe_fold_fp64_v3
from xorl.models.layers.moe.dsv4_native_combine import (
    compact_rank_padded_rows,
    exchange_variable_and_canonical_fold,
    validate_dsv4_native_ep_combine_size,
)


def test_dsv4_combine_rejects_unvalidated_ep_sizes():
    with pytest.raises(ValueError, match="admits EP sizes"):
        validate_dsv4_native_ep_combine_size(16)


def test_compact_rank_padded_rows_retains_only_live_prefixes():
    gathered = torch.arange(12).reshape(6, 2)
    compact = compact_rank_padded_rows(gathered, padded_rows=2, row_counts=(2, 0, 1))
    assert torch.equal(compact, torch.stack((gathered[0], gathered[1], gathered[4])))


def test_variable_exchange_applies_the_shared_canonical_fold(monkeypatch):
    """EP8 arrivals in rank order must reduce via canonical_moe_fold_fp64_v3."""

    import xorl.distributed.moe.comm as comm  # noqa: PLC0415

    ep_size = 8
    local_rows = 3
    row_counts = (local_rows,) * ep_size
    partial = torch.randn(sum(row_counts), 4, dtype=torch.bfloat16)
    torch.manual_seed(20260811)
    exchanged = torch.randn(ep_size * local_rows, 4, dtype=torch.bfloat16)

    def fake_apply(group, value, output_splits, input_splits):
        assert group == "group"
        assert output_splits == [local_rows] * ep_size
        assert input_splits == list(row_counts)
        return exchanged

    monkeypatch.setattr(comm._AllToAll, "apply", fake_apply)
    result = exchange_variable_and_canonical_fold(partial, "group", row_counts, source_ordinal=0)
    expected = canonical_moe_fold_fp64_v3(exchanged.reshape(ep_size, local_rows, 4))
    assert torch.equal(result, expected)


def test_canonical_fold_diverges_from_the_retired_nccl_chain(monkeypatch):
    """Witness input where adjacent-pair fold != the old [1..N-1,0] BF16 chain.

    Guards the byte contract of the unification: the combine really is the
    balanced tree, not any left-associative chain. The chain (seed rank 1,
    then 2, 3, 0) forms 1.0+1.0=2.0 first, and each later +2**-7 is a
    round-to-even no-op (half-ulp tie at magnitude 2.0), landing on 2.0.
    The fold pairs (0,1) and (2,3) into 1.0078125 each — exact at magnitude
    1.0 where the ulp is 2**-7 — and lands on 2.015625.
    """

    import xorl.distributed.moe.comm as comm  # noqa: PLC0415
    import xorl.models.layers.moe.dsv4_native_combine as combine_module  # noqa: PLC0415

    monkeypatch.setattr(combine_module, "validate_dsv4_native_ep_combine_size", lambda ep_size: None)

    row_counts = (1, 1, 1, 1)
    partial = torch.zeros(4, 2, dtype=torch.bfloat16)
    blocks = torch.tensor(
        [[2**-7, 2**-7], [1.0, 1.0], [1.0, 1.0], [2**-7, 2**-7]],
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(comm._AllToAll, "apply", lambda *args: blocks)

    result = exchange_variable_and_canonical_fold(partial, "group", row_counts, source_ordinal=0)
    assert torch.equal(result, torch.full((1, 2), 2.015625, dtype=torch.bfloat16))

    chain = blocks[1]
    for source_rank in (2, 3, 0):
        chain = chain + blocks[source_rank]
    assert torch.equal(chain, torch.full((2,), 2.0, dtype=torch.bfloat16))
    assert not torch.equal(result[0], chain)


def test_variable_exchange_returns_empty_for_zero_local_rows(monkeypatch):
    import xorl.distributed.moe.comm as comm  # noqa: PLC0415

    row_counts = (0, 2, 2, 2, 2, 2, 2, 2)
    partial = torch.randn(sum(row_counts), 4, dtype=torch.bfloat16)
    monkeypatch.setattr(comm._AllToAll, "apply", lambda *args: partial.new_zeros((0, 4)))
    result = exchange_variable_and_canonical_fold(partial, "group", row_counts, source_ordinal=0)
    assert result.shape == (0, 4)


def test_variable_exchange_routes_a_ragged_nonzero_owner(monkeypatch):
    import xorl.distributed.moe.comm as comm  # noqa: PLC0415

    row_counts = (1, 3, 0, 2, 4, 1, 2, 1)
    source_ordinal = 3
    local_rows = row_counts[source_ordinal]
    partial = torch.randn(sum(row_counts), 4, dtype=torch.bfloat16)
    arrivals = torch.randn(8 * local_rows, 4, dtype=torch.bfloat16)

    def fake_apply(group, value, output_splits, input_splits):
        assert group == "group"
        assert value is partial
        assert output_splits == [local_rows] * 8
        assert input_splits == list(row_counts)
        return arrivals

    monkeypatch.setattr(comm._AllToAll, "apply", fake_apply)
    result = exchange_variable_and_canonical_fold(
        partial,
        "group",
        row_counts,
        source_ordinal=source_ordinal,
    )
    expected = canonical_moe_fold_fp64_v3(arrivals.reshape(8, local_rows, 4))
    assert torch.equal(result, expected)


def test_variable_exchange_folds_local_arrivals_in_fp32_and_backpropagates(monkeypatch):
    """BF16 transport is unchanged; every source-tree add happens in FP32."""

    import xorl.distributed.moe.comm as comm  # noqa: PLC0415

    row_counts = (1,) * 8
    partial = torch.tensor(
        [[4096.0], [1.0], [-4096.0], [1.0], [0.0], [0.0], [0.0], [0.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def fake_apply(group, value, output_splits, input_splits):
        assert group == "group"
        assert value.dtype is torch.bfloat16
        assert output_splits == [1] * 8
        assert input_splits == [1] * 8
        return value

    monkeypatch.setattr(comm._AllToAll, "apply", fake_apply)
    result = exchange_variable_and_canonical_fold(
        partial,
        "group",
        row_counts,
        source_ordinal=0,
    )

    # The retired BF16-node tree loses both unit contributions.  Promoting the
    # rank-ordered arrivals once and retaining FP32 through the complete tree
    # yields 2.0 before the single BF16 consumer-boundary cast.
    assert result.dtype is torch.bfloat16
    assert torch.equal(result, torch.tensor([[2.0]], dtype=torch.bfloat16))

    result.float().sum().backward()
    assert torch.equal(partial.grad, torch.ones_like(partial))
