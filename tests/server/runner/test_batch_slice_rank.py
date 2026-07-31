from types import SimpleNamespace

import pytest

from xorl.server.runner.utils.batch_utils import batch_slice_rank_and_size


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _state(**overrides):
    return SimpleNamespace(
        tp_size=overrides.get("tp_size", 1),
        dp_rank=overrides.get("dp_rank", 0),
        dp_size=overrides.get("dp_size", 1),
        dp_replicate_rank=overrides.get("dp_replicate_rank", 0),
        dp_replicate_size=overrides.get("dp_replicate_size", 1),
        ep_enabled=overrides.get("ep_enabled", False),
        ep_size=overrides.get("ep_size", 1),
        dp_shard_in_ep_size=overrides.get("dp_shard_in_ep_size", 1),
        ep_fsdp_device_mesh=None,
    )


def test_fsdp_shards_share_replicate_slice():
    for rank in range(4):
        assert batch_slice_rank_and_size(rank, 4, _state(dp_rank=rank, dp_size=4), 1, 1) == (0, 1)


def test_tp_ranks_share_slice():
    state = _state(tp_size=2, dp_replicate_size=4)
    assert batch_slice_rank_and_size(0, 8, state, 1, 1) == (0, 4)
    state.dp_replicate_rank = 3
    assert batch_slice_rank_and_size(7, 8, state, 1, 1) == (3, 4)


def test_ep_ranks_are_distinct_by_default():
    state = _state(ep_enabled=True, ep_size=8)
    assert batch_slice_rank_and_size(5, 8, state, 1, 1) == (5, 8)


def test_legacy_ep_duplication_switch(monkeypatch):
    monkeypatch.setenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "1")
    state = _state(ep_enabled=True, ep_size=8)
    assert batch_slice_rank_and_size(5, 8, state, 1, 1) == (0, 1)
