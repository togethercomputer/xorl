"""Focused algebra and hot-path regressions for topology-preserving adapters."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Shard

from xorl.server.runner.adapters import gradient_ownership as gradient_ownership_module
from xorl.server.runner.adapters import sharded_state as sharded_state_module
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.adapters.sharded_state import (
    AdapterTensorLayout,
    deterministic_local_initialization,
    discover_adapter_layouts,
    pack_logical_tensor,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _DummyLoRALayer(nn.Module):
    def __init__(self, *, max_rank: int = 4) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(max_rank, 8))
        self.lora_B = nn.Parameter(torch.zeros(8, max_rank))
        self.active_r = max_rank
        self.active_lora_alpha = 16

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        self.active_r = lora_rank
        self.active_lora_alpha = lora_alpha

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        delta_weight = self.lora_B @ self.lora_A
        return hidden_states @ delta_weight.transpose(0, 1)


class _DummyLoRAModel(nn.Module):
    def __init__(self, *, max_rank: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.o_proj = _DummyLoRALayer(max_rank=max_rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0].self_attn.o_proj(hidden_states)


class _ReorderedDummyLoRALayer(_DummyLoRALayer):
    """Same logical factors, registered B-before-A to change FQN iteration order."""

    def __init__(self, *, max_rank: int = 4) -> None:
        nn.Module.__init__(self)
        self.lora_B = nn.Parameter(torch.zeros(8, max_rank))
        self.lora_A = nn.Parameter(torch.randn(max_rank, 8))
        self.active_r = max_rank
        self.active_lora_alpha = 16


def _build_manager(tmp_path, **kwargs):
    max_rank = kwargs.pop("max_rank", 4)
    return LoRAAdapterManager(
        _DummyLoRAModel(max_rank=max_rank),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        **kwargs,
    )


def _session_spec(*, rank: int, alpha: int, optimizer_type: str, lr: float) -> dict:
    return {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": True,
        "lora_config": {"lora_rank": rank, "lora_alpha": alpha},
        "optimizer_config": {
            "type": optimizer_type,
            "learning_rate": lr,
            "weight_decay": 0.0,
            "optimizer_dtype": "bf16",
            "betas": None if optimizer_type == "sgd" else [0.9, 0.95],
            "eps": None if optimizer_type == "sgd" else 1e-8,
            "optimizer_kwargs": {},
        },
    }


def _layout(
    *,
    name: str = "layer.lora_A",
    local_shape: tuple[int, ...] = (4, 4),
    offset: tuple[int, ...] = (0, 0),
    substrate: tuple[int, ...] = (4, 4),
    logical: tuple[int, ...] = (4, 4),
    rank_dim: int = 0,
) -> AdapterTensorLayout:
    slices = tuple(
        slice(max(0, -start), min(size, logical_size - start))
        for start, size, logical_size in zip(offset, local_shape, logical, strict=True)
    )
    storage = tuple(selection.stop - selection.start for selection in slices)
    return AdapterTensorLayout(
        fqn=name,
        dtype=torch.float32,
        rank_dim=rank_dim,
        substrate_shape=substrate,
        logical_shape=logical,
        local_substrate_shape=local_shape,
        local_logical_offset=offset,
        local_logical_shape=local_shape,
        active_local_slices=slices,
        active_storage_shape=storage,
        replica_count=1,
        replica_key=(name, logical, offset, local_shape, "float32"),
    )


def test_layout_pack_unpack_zeroes_inactive_rank_and_supports_empty_intersection():
    layout = _layout(local_shape=(4, 4), substrate=(4, 4), logical=(2, 4))
    local = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    slot = layout.pack_from_local(local)
    assert tuple(slot.shape) == (2, 4)
    restored = layout.unpack_to_local(slot)
    assert torch.equal(restored[:2], local[:2])
    assert torch.count_nonzero(restored[2:]) == 0

    empty = _layout(local_shape=(2, 4), offset=(2, 0), substrate=(4, 4), logical=(2, 4))
    assert empty.active_storage_shape == (0, 4)
    assert empty.pack_from_local(torch.ones(2, 4)).shape == (0, 4)
    assert torch.count_nonzero(empty.unpack_to_local(torch.empty(0, 4))) == 0


def test_layout_discovery_keeps_identical_empty_fsdp_shards_out_of_replica_classes(monkeypatch):
    nonempty = _layout(name="lora_A", local_shape=(1, 4), offset=(0, 0), substrate=(1, 4), logical=(1, 4))
    empty = _layout(name="lora_A", local_shape=(0, 4), offset=(1, 0), substrate=(1, 4), logical=(1, 4))
    gathered = [
        {"layouts": [sharded_state_module._descriptor_from_layout(nonempty)], "group_memberships": {}},
        {"layouts": [sharded_state_module._descriptor_from_layout(empty)], "group_memberships": {}},
        {"layouts": [sharded_state_module._descriptor_from_layout(empty)], "group_memberships": {}},
    ]

    model = nn.Module()
    model.register_parameter("lora_A", nn.Parameter(torch.empty(0, 4)))
    monkeypatch.setattr(sharded_state_module, "_base_layout_for_parameter", lambda *_args, **_kwargs: empty)
    monkeypatch.setattr(sharded_state_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(sharded_state_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sharded_state_module.dist, "get_world_size", lambda group=None: 3)
    monkeypatch.setattr(sharded_state_module.dist, "get_rank", lambda group=None: 1)

    def _all_gather_object(output, local_payload, group=None):
        assert local_payload == gathered[1]
        output[:] = gathered

    monkeypatch.setattr(sharded_state_module.dist, "all_gather_object", _all_gather_object)

    layouts, fingerprint, _ = discover_adapter_layouts(
        model,
        {"lora_A": {"rank_dim": 0}},
        active_rank=1,
    )

    assert layouts["lora_A"].active_storage_shape == (0, 4)
    assert layouts["lora_A"].replica_count == 1
    assert layouts["lora_A"].replica_ranks == (1,)
    assert fingerprint

    model.lora_A.to_local = lambda: model.lora_A
    model.lora_A.placements = ("Shard(0)",)
    plan = gradient_ownership_module.compile_adapter_gradient_ownership(
        layouts=layouts,
        model_parameters={"lora_A": model.lora_A},
        optimizer_parameters={"lora_A": nn.Parameter(torch.empty(0, 4))},
        declarations={
            "lora_A": gradient_ownership_module.ParameterOwnershipDeclaration(
                topology=gradient_ownership_module.TopologyFamily.DENSE_REPLICATED,
                producer=gradient_ownership_module.ProducerFamily.MODULE_MANAGED,
                representation=gradient_ownership_module.GradientRepresentation.FSDP_COMPLETED_LOCAL_SHARD,
                completed_domains=(
                    gradient_ownership_module.ReductionDomainPlan(
                        gradient_ownership_module.ReductionAxis.FSDP_SHARD,
                        gradient_ownership_module.ReductionAuthority.FSDP,
                        gradient_ownership_module.ReductionOperation.SUM,
                        "fsdp_shard",
                    ),
                ),
                pending_domains=(),
                presence=gradient_ownership_module.GradientPresencePolicy.REQUIRED_IF_ACTIVE,
                config_guard_fingerprint="empty-fsdp-shard",
                managed_fsdp_shard=True,
            )
        },
        model_generation="model-generation",
        adapter_generation="adapter-generation",
        rank=1,
    )
    assert plan.parameters[0].norm_replica_divisor == 1
    assert not plan.parameters[0].requires_local_gradient


def test_deterministic_initialization_is_coordinate_and_replica_stable():
    full = _layout(local_shape=(4, 4), offset=(0, 0))
    first = _layout(local_shape=(2, 4), offset=(0, 0))
    second = _layout(local_shape=(2, 4), offset=(2, 0))
    replica = _layout(local_shape=(4, 4), offset=(0, 0))

    full_values = deterministic_local_initialization(full, base_seed=17, session_identity="policy", is_lora_b=False)
    sharded_values = torch.cat(
        [
            deterministic_local_initialization(first, base_seed=17, session_identity="policy", is_lora_b=False),
            deterministic_local_initialization(second, base_seed=17, session_identity="policy", is_lora_b=False),
        ],
        dim=0,
    )
    replica_values = deterministic_local_initialization(
        replica, base_seed=17, session_identity="policy", is_lora_b=False
    )
    assert torch.equal(full_values, sharded_values)
    assert torch.equal(full_values, replica_values)
    assert not torch.equal(
        deterministic_local_initialization(first, base_seed=17, session_identity="expert-0", is_lora_b=False),
        deterministic_local_initialization(second, base_seed=17, session_identity="expert-0", is_lora_b=False),
    )
    assert (
        torch.count_nonzero(
            deterministic_local_initialization(full, base_seed=17, session_identity="p", is_lora_b=True)
        )
        == 0
    )


def test_nonzero_lora_b_initialization_is_coordinate_and_replica_stable():
    full = _layout(local_shape=(4, 4), offset=(0, 0))
    first = _layout(local_shape=(2, 4), offset=(0, 0))
    second = _layout(local_shape=(2, 4), offset=(2, 0))

    full_values = deterministic_local_initialization(
        full, base_seed=1616, session_identity="policy", is_lora_b=True, lora_b_std=1e-3
    )
    sharded_values = torch.cat(
        [
            deterministic_local_initialization(
                first, base_seed=1616, session_identity="policy", is_lora_b=True, lora_b_std=1e-3
            ),
            deterministic_local_initialization(
                second, base_seed=1616, session_identity="policy", is_lora_b=True, lora_b_std=1e-3
            ),
        ],
        dim=0,
    )

    assert torch.equal(full_values, sharded_values)
    assert torch.count_nonzero(full_values) > 0


def test_deterministic_initialization_is_independent_of_fqn_iteration_order(tmp_path):
    ordered = _build_manager(tmp_path / "ordered", optimizer_type="sgd")
    reordered_model = _DummyLoRAModel()
    reordered_model.model.layers[0].self_attn.o_proj = _ReorderedDummyLoRALayer()
    reordered = LoRAAdapterManager(
        reordered_model,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "reordered" / "adapters"),
        auto_save_on_eviction=False,
        optimizer_type="sgd",
    )
    spec = _session_spec(rank=4, alpha=16, optimizer_type="sgd", lr=0.1)
    ordered.register_adapter("policy", session_spec=spec, initialize_fresh=True)
    reordered.register_adapter("policy", session_spec=spec, initialize_fresh=True)
    ordered_state = ordered.get_adapter_state("policy")
    reordered_state = reordered.get_adapter_state("policy")
    assert list(ordered_state.local_params) != list(reordered_state.local_params)
    for name in ordered_state.local_params:
        assert torch.equal(ordered_state.local_params[name], reordered_state.local_params[name]), name


def test_manager_owns_local_active_slots_and_capture_hot_path_has_no_full_tensor(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="sgd", max_rank=4)
    manager.register_adapter(
        "local",
        session_spec=_session_spec(rank=2, alpha=8, optimizer_type="sgd", lr=0.1),
        initialize_fresh=True,
    )
    state = manager.get_adapter_state("local")
    assert all(
        tuple(param.shape) == layout.active_storage_shape
        for name, param in state.local_params.items()
        for layout in [state.tensor_layouts[name]]
    )
    assert state.layout_fingerprint


def test_real_gloo_uneven_dtensor_layout_and_logical_pack():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_SHARDED_LAYOUT_WORKER": "1"},
    )
    result.assert_success("real two-rank Gloo DTensor layout test")


class _FakeEpMesh:
    def __init__(self, rank: int = 1, size: int = 2) -> None:
        self.rank = rank
        self._size = size

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self.rank


class _FakeSpecInfo:
    def __init__(self, placement, ep_mesh) -> None:
        self.placement = placement
        self.ep_mesh = ep_mesh


def test_layout_discovery_composes_explicit_ep_shard_and_ignores_generic_replicate():
    expert_model = nn.Module()
    expert_model.register_parameter("expert_lora_A", nn.Parameter(torch.empty(2, 3)))
    expert_model._fqn2spec_info = {
        "expert_lora_A": _FakeSpecInfo(Shard(0), _FakeEpMesh()),
    }
    expert_layouts, _, _ = discover_adapter_layouts(
        expert_model,
        {"expert_lora_A": {"rank_dim": 0}},
        active_rank=3,
    )
    expert_layout = expert_layouts["expert_lora_A"]
    assert expert_layout.substrate_shape == (4, 3)
    assert expert_layout.local_logical_offset == (2, 0)
    assert expert_layout.active_storage_shape == (1, 3)

    dense_model = nn.Module()
    dense_model.register_parameter("dense_lora_A", nn.Parameter(torch.empty(2, 3)))
    dense_model._fqn2spec_info = {
        "dense_lora_A": _FakeSpecInfo(torch.distributed._tensor.Replicate(), _FakeEpMesh()),
    }
    dense_layouts, _, _ = discover_adapter_layouts(
        dense_model,
        {"dense_lora_A": {"rank_dim": 0}},
        active_rank=2,
    )
    dense_layout = dense_layouts["dense_lora_A"]
    assert dense_layout.substrate_shape == (2, 3)
    assert dense_layout.local_logical_offset == (0, 0)


def _run_gloo_layout_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2
    mesh = DeviceMesh("cpu", torch.arange(world))
    local_rows = 3 if rank == 0 else 2
    local = torch.arange(local_rows * 4, dtype=torch.float32).reshape(local_rows, 4) + rank * 100
    dtensor = DTensor.from_local(local, mesh, [Shard(0)], shape=(5, 4), stride=(4, 1), run_check=False)
    model = nn.Module()
    model.register_parameter("lora_A", nn.Parameter(dtensor))
    layouts, fingerprint, _ = discover_adapter_layouts(
        model,
        {"lora_A": {"rank_dim": 0}},
        active_rank=5,
    )
    layout = layouts["lora_A"]
    assert layout.local_logical_offset == ((0, 0) if rank == 0 else (3, 0))
    assert layout.replica_count == 1
    logical = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    packed = pack_logical_tensor(layout, logical)
    payloads = [None] * world
    dist.all_gather_object(payloads, packed)
    if rank == 0:
        assert torch.equal(torch.cat(payloads, dim=0), logical)
        assert fingerprint
    dist.destroy_process_group()


if os.environ.get("XORL_SHARDED_LAYOUT_WORKER") == "1":
    _run_gloo_layout_worker()
