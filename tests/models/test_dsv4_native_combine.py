"""DSV4-Flash architecture-scoped EP combine (NCCL-tree contributor order)."""

import pytest
import torch

from xorl.models.layers.moe.dsv4_native_combine import (
    compact_rank_padded_rows,
    dsv4_nccl_tree_chain_order,
    exchange_variable_and_nccl_tree_chain_sum,
    validate_dsv4_native_ep_combine_size,
)


def test_dsv4_chain_order_is_the_captured_nccl_tree_order():
    assert dsv4_nccl_tree_chain_order(8) == (1, 2, 3, 4, 5, 6, 7, 0)


def test_dsv4_combine_rejects_unvalidated_ep_sizes():
    with pytest.raises(ValueError, match="admits EP sizes"):
        validate_dsv4_native_ep_combine_size(16)


def test_compact_rank_padded_rows_retains_only_live_prefixes():
    gathered = torch.arange(12).reshape(6, 2)
    compact = compact_rank_padded_rows(gathered, padded_rows=2, row_counts=(2, 0, 1))
    assert torch.equal(compact, torch.stack((gathered[0], gathered[1], gathered[4])))


def test_variable_exchange_uses_nccl_tree_contributor_order(monkeypatch):
    """The chain must seed at rank 1, add ranks 2..N-1, and add rank 0 LAST.

    This is the captured bitwise behavior of the DSV4 serving contract's
    pinned NCCL tree all-reduce; it intentionally differs from the Qwen/GLM
    canonical adjacent-pair fold and from Qwen's reverse-rank chain.
    """

    import xorl.distributed.moe.comm as comm  # noqa: PLC0415
    import xorl.models.layers.moe.dsv4_native_combine as combine_module  # noqa: PLC0415

    monkeypatch.setattr(combine_module, "validate_dsv4_native_ep_combine_size", lambda ep_size: None)
    monkeypatch.setattr(
        combine_module,
        "dsv4_nccl_tree_chain_order",
        lambda ep_size: (*range(1, ep_size), 0),
    )

    partial = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)

    def fake_apply(group, value, output_splits, input_splits):
        assert group == "group"
        assert value is partial
        assert output_splits == [2, 2, 2]
        assert input_splits == [2, 0, 1]
        return torch.cat((value[:2], value[:2] + 10, value[:2] + 20), dim=0)

    monkeypatch.setattr(comm._AllToAll, "apply", fake_apply)
    result = exchange_variable_and_nccl_tree_chain_sum(partial, "group", (2, 0, 1), local_rank=0)
    # Seed rank 1 (+10), add rank 2 (+20), add rank 0 last.
    assert torch.equal(result, ((partial[:2] + 10) + (partial[:2] + 20)) + partial[:2])
