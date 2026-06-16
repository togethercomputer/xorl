from types import SimpleNamespace

import pytest
import torch

import xorl.server.runner.runner_dispatcher as runner_dispatcher_module
from xorl.server.runner.runner_dispatcher import RunnerDispatcher


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeEPMesh:
    def __init__(self, ep_fsdp_rank: int) -> None:
        self.ep_fsdp_rank = ep_fsdp_rank

    def get_local_rank(self, name: str) -> int:
        assert name == "ep_fsdp"
        return self.ep_fsdp_rank


def _dispatcher(rank: int, world_size: int) -> RunnerDispatcher:
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = rank
    dispatcher.world_size = world_size
    return dispatcher


def _batch(batch_id: int, *, num_samples: int = 1) -> dict:
    return {
        "input_ids": [[batch_id, batch_id + 1]],
        "labels": [[batch_id + 2, batch_id + 3]],
        "position_ids": [[0, 1]],
        "num_samples": num_samples,
    }


def _parallel_state(**overrides):
    return SimpleNamespace(
        cp_size=overrides.get("cp_size", 1),
        pp_enabled=overrides.get("pp_enabled", False),
        pp_size=overrides.get("pp_size", 1),
        ep_enabled=overrides.get("ep_enabled", False),
        ep_size=overrides.get("ep_size", 1),
        dp_shard_in_ep_size=overrides.get("dp_shard_in_ep_size", 1),
        ep_fsdp_device_mesh=overrides.get("ep_fsdp_device_mesh"),
    )


def test_select_batches_keeps_existing_dp_distribution_without_ep(monkeypatch):
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    batches = [_batch(10), _batch(20), _batch(30), _batch(40)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=2, world_size=4)._select_and_prepare_batches(
        batches,
        routed_experts=["r0", "r1", "r2", "r3"],
        routed_expert_logits=["l0", "l1", "l2", "l3"],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[30, 31]]))
    assert routed_experts == ["r2"]
    assert routed_logits == ["l2"]


def test_select_batches_broadcasts_one_slice_to_all_ep_ranks(monkeypatch):
    state = _parallel_state(
        ep_enabled=True,
        ep_size=8,
        dp_shard_in_ep_size=1,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=0),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        [_batch(10, num_samples=2)],
        routed_experts=["r0", "r1"],
        routed_expert_logits=["l0", "l1"],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[10, 11]]))
    assert my_batches[0]["num_samples"] == 2
    assert routed_experts == ["r0", "r1"]
    assert routed_logits == ["l0", "l1"]


def test_select_batches_uses_ep_fsdp_rank_for_distinct_ep_batch_slices(monkeypatch):
    state = _parallel_state(
        ep_enabled=True,
        ep_size=4,
        dp_shard_in_ep_size=2,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=1),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    my_batches, routed_experts, routed_logits = _dispatcher(rank=6, world_size=8)._select_and_prepare_batches(
        [_batch(10), _batch(20)],
        routed_experts=["r0", "r1"],
        routed_expert_logits=["l0", "l1"],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[20, 21]]))
    assert routed_experts == ["r1"]
    assert routed_logits == ["l1"]
