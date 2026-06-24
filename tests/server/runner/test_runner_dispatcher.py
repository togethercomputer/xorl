from types import SimpleNamespace

import pytest
import torch

import xorl.server.runner.runner_dispatcher as runner_dispatcher_module
from xorl.server.orchestrator.packing import SequentialPacker
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


def test_select_batches_gives_each_ep_rank_a_distinct_slice(monkeypatch):
    state = _parallel_state(
        ep_enabled=True,
        ep_size=8,
        dp_shard_in_ep_size=1,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=0),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    batches = [_batch(10 * (i + 1)) for i in range(8)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        batches,
        routed_experts=[f"r{i}" for i in range(8)],
        routed_expert_logits=[f"l{i}" for i in range(8)],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[60, 61]]))
    assert routed_experts == ["r5"]
    assert routed_logits == ["l5"]


def test_select_batches_pads_ep_ranks_beyond_real_batches_with_dummies(monkeypatch):
    state = _parallel_state(
        ep_enabled=True,
        ep_size=8,
        dp_shard_in_ep_size=1,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=0),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        [_batch(10, num_samples=2), _batch(20)],
        routed_experts=["r0", "r1", "r2"],
        routed_expert_logits=["l0", "l1", "l2"],
    )

    assert len(my_batches) == 1
    assert my_batches[0]["num_samples"] == 0
    assert torch.all(my_batches[0]["labels"] == -100)
    assert routed_experts == []
    assert routed_logits == []


def test_select_batches_shares_one_slice_across_cp_ranks_under_ep(monkeypatch):
    state = _parallel_state(
        cp_size=2,
        ep_enabled=True,
        ep_size=4,
        dp_shard_in_ep_size=2,
        ep_fsdp_device_mesh=_FakeEPMesh(ep_fsdp_rank=1),
    )
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: state)

    batches = [_batch(10 * (i + 1)) for i in range(4)]
    my_batches, routed_experts, routed_logits = _dispatcher(rank=5, world_size=8)._select_and_prepare_batches(
        batches,
        routed_experts=[f"r{i}" for i in range(4)],
        routed_expert_logits=[f"l{i}" for i in range(4)],
    )

    assert len(my_batches) == 1
    assert torch.equal(my_batches[0]["input_ids"], torch.tensor([[30, 31]]))
    assert routed_experts == ["r2"]
    assert routed_logits == ["l2"]


def test_select_batches_legacy_flag_broadcasts_one_slice_to_all_ep_ranks(monkeypatch):
    monkeypatch.setenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "1")
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


def test_select_batches_legacy_flag_uses_ep_fsdp_rank_for_ep_group_slices(monkeypatch):
    monkeypatch.setenv("XORL_SERVER_EP_DUPLICATE_BATCHES", "1")
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


# ============================================================================
# End-to-end acceptance: balanced_dp packer + dispatcher => zero dummy batches
# ============================================================================


def test_balanced_dp_packing_yields_zero_dispatcher_dummies(monkeypatch):
    """The redesign's primary acceptance criterion (spec section 5.1):

    Packing with strategy='balanced_dp' at the dispatcher's dp_size produces
    N == k*dp_size rows, so EVERY rank gets the same number of REAL batches and
    no rank runs a dummy (num_samples == 0) filler.
    """
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    dp_size = 8
    world_size = 8  # non-EP, cp=pp=1 -> dispatcher dp_size == world_size
    data = [
        {"input_ids": list(range(200 + 37 * i)), "target_tokens": list(range(200 + 37 * i))} for i in range(40)
    ]
    packer = SequentialPacker(
        enable_packing=True, log_stats=False, pad_to_multiple_of=1, strategy="balanced_dp", dp_size=dp_size
    )
    raw_batches = packer.pack(data, max_seq_len=8192, request_id="acc")
    assert len(raw_batches) % dp_size == 0

    round_counts = set()
    total_real = 0
    for rank in range(world_size):
        my_batches, _, _ = _dispatcher(rank=rank, world_size=world_size)._select_and_prepare_batches(raw_batches)
        round_counts.add(len(my_batches))
        # No dummy fillers: every batch this rank runs is real.
        assert all(b["num_samples"] > 0 for b in my_batches)
        total_real += sum(b["num_samples"] for b in my_batches)

    # Lockstep: identical round count across ranks (collective invariant).
    assert len(round_counts) == 1
    # Every datum trained exactly once, nothing dropped or duplicated.
    assert total_real == len(data)


def test_sequential_packing_still_pads_dummies_when_rows_below_dp(monkeypatch):
    """Contrast case: legacy sequential under-fills -> dispatcher still pads."""
    monkeypatch.setattr(runner_dispatcher_module, "get_parallel_state", lambda: _parallel_state())

    dp_size = 8
    # Few large samples -> sequential makes fewer rows than dp_size -> dummies.
    data = [{"input_ids": list(range(7000)), "target_tokens": list(range(7000))} for _ in range(4)]
    packer = SequentialPacker(enable_packing=True, log_stats=False, pad_to_multiple_of=1, strategy="sequential")
    raw_batches = packer.pack(data, max_seq_len=8192, request_id="seq")
    assert len(raw_batches) < dp_size

    saw_dummy = False
    for rank in range(dp_size):
        my_batches, _, _ = _dispatcher(rank=rank, world_size=dp_size)._select_and_prepare_batches(raw_batches)
        if any(b["num_samples"] == 0 for b in my_batches):
            saw_dummy = True
    assert saw_dummy
