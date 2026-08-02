"""Focused algebra and hot-path regressions for topology-preserving adapters."""

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Shard

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


class _DummyLoRAModel(nn.Module):
    def __init__(self, *, max_rank: int = 4) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.o_proj = _DummyLoRALayer(max_rank=max_rank)


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


def test_capture_source_contract_excludes_full_tensor():
    source = inspect.getsource(LoRAAdapterManager.capture_gradients)
    assert "full_tensor" not in source


def test_capture_eight_microbatches_matches_one_combined_gradient(tmp_path):
    repeated = _build_manager(tmp_path / "repeated", optimizer_type="sgd")
    combined = _build_manager(tmp_path / "combined", optimizer_type="sgd")
    spec = _session_spec(rank=4, alpha=16, optimizer_type="sgd", lr=0.1)
    repeated.register_adapter("repeated", session_spec=spec, initialize_fresh=True)
    combined.register_adapter("combined", session_spec=spec, initialize_fresh=True)

    repeated_state = repeated.get_adapter_state("repeated")
    combined_state = combined.get_adapter_state("combined")
    repeated_grads = {
        name: torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) + 1
        for name, param in repeated_state.local_params.items()
    }
    for _ in range(8):
        for name, param in repeated.model.named_parameters():
            if name in repeated_grads:
                param.grad = repeated_grads[name].clone()
        repeated.capture_gradients("repeated")

    for name, param in combined.model.named_parameters():
        if name in repeated_grads:
            param.grad = repeated_grads[name] * 8
    combined.capture_gradients("combined")

    for name in repeated_state.local_params:
        assert torch.equal(repeated_state.local_params[name].grad, combined_state.local_params[name].grad)


def test_norm_clipping_disabled_large_norm_and_nonfinite_retry_policy(tmp_path):
    manager = _build_manager(tmp_path / "clip", optimizer_type="sgd")
    spec = _session_spec(rank=4, alpha=16, optimizer_type="sgd", lr=0.1)
    manager.register_adapter("clip", session_spec=spec, initialize_fresh=True)
    state = manager.get_adapter_state("clip")
    parameter = next(param for param in state.local_params.values() if param.numel())
    parameter.data.zero_()
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad.view(-1)[0] = 13.0
    assert manager.optim_step("clip", lr=0.1, gradient_clip=1.0) == pytest.approx(13.0)

    parameter.grad = torch.zeros_like(parameter)
    parameter.grad.view(-1)[0] = 12001.0
    before = parameter.detach().clone()
    assert manager.optim_step("clip", lr=0.1, gradient_clip=None) == pytest.approx(12001.0)
    expected = before.clone()
    expected.view(-1)[0] -= 0.1 * 12001.0
    assert torch.allclose(parameter, expected, atol=1e-3)

    before_weights = {name: param.detach().clone() for name, param in state.local_params.items()}
    state_step = state.global_step
    parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(FloatingPointError, match="all ranks skipped"):
        manager.optim_step("clip", lr=0.1, gradient_clip=1.0)
    assert state.global_step == state_step
    assert all(torch.equal(param, before_weights[name]) for name, param in state.local_params.items())
    assert all(param.grad is None for param in state.local_params.values())


def test_real_gloo_uneven_dtensor_layout_and_logical_pack():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_SHARDED_LAYOUT_WORKER": "1"},
    )
    result.assert_success("real two-rank Gloo DTensor layout test")


def test_real_gloo_collective_finite_and_norm_gate():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_SHARDED_OPT_WORKER": "1"},
    )
    result.assert_success("real two-rank Gloo adapter finite/norm gate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NCCL fused adapter gate requires CUDA")
def test_real_nccl_fused_adapter_optimizer_gate():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_SHARDED_CUDA_WORKER": "1"},
    )
    result.assert_success("real two-rank NCCL fused adapter optimizer gate")


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
    expert_layouts, _ = discover_adapter_layouts(
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
    dense_layouts, _ = discover_adapter_layouts(
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
    layouts, fingerprint = discover_adapter_layouts(
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


def _run_gloo_optimizer_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    expected_world = int(os.environ.get("XORL_EXPECTED_WORLD", "2"))
    assert world == expected_world
    from xorl.distributed.parallel_state import init_parallel_state

    init_parallel_state(dp_size=world, dp_mode="none", device_type="cpu")
    manager = _build_manager(Path("/tmp") / f"xorl-sharded-opt-{rank}", optimizer_type="sgd")
    manager.register_adapter(
        "distributed",
        session_spec=_session_spec(rank=4, alpha=16, optimizer_type="sgd", lr=0.1),
        initialize_fresh=True,
    )
    state = manager.get_adapter_state("distributed")
    for param in state.local_params.values():
        param.grad = torch.full_like(param, float("nan") if rank == 0 else rank + 1.0)
    try:
        manager.optim_step("distributed", lr=0.1, gradient_clip=None)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("collective non-finite gate did not reject the update")
    assert state.global_step == 0
    assert all(param.grad is None for param in state.local_params.values())

    for param in state.local_params.values():
        param.grad = torch.full_like(param, float(rank + 1))
    norm = manager.optim_step("distributed", lr=0.1, gradient_clip=None)
    expected_local_elements = sum(param.numel() for param in state.local_params.values())
    expected = (sum(float(index + 1) ** 2 for index in range(world)) * expected_local_elements / world) ** 0.5
    assert norm == pytest.approx(expected)
    assert state.global_step == 1
    dist.destroy_process_group()


def _run_nccl_fused_worker() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    from xorl.distributed.parallel_state import init_parallel_state

    init_parallel_state(dp_size=2, dp_mode="none", device_type="cuda")
    model = _DummyLoRAModel().to(device="cuda")
    manager = LoRAAdapterManager(
        model,
        device=torch.device("cuda"),
        checkpoint_dir=str(Path("/tmp") / f"xorl-sharded-cuda-{rank}"),
        auto_save_on_eviction=False,
        optimizer_type="adamw",
        optimizer_fused=True,
    )
    manager.register_adapter(
        "fused",
        session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=1e-3),
        initialize_fresh=True,
    )
    state = manager.get_adapter_state("fused")
    for param in state.local_params.values():
        param.grad = torch.ones_like(param)
    norm = manager.optim_step("fused", lr=1e-3, gradient_clip=1.0)
    torch.cuda.synchronize()
    assert norm > 0.0
    assert state.global_step == 1
    peak_memory = torch.tensor([torch.cuda.max_memory_allocated()], dtype=torch.int64, device="cuda")
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"NCCL fused adapter gate: world_size=2 peak_memory_bytes={int(peak_memory.item())}")
    dist.destroy_process_group()


if os.environ.get("XORL_SHARDED_LAYOUT_WORKER") == "1":
    _run_gloo_layout_worker()
if os.environ.get("XORL_SHARDED_OPT_WORKER") == "1":
    _run_gloo_optimizer_worker()
if os.environ.get("XORL_SHARDED_CUDA_WORKER") == "1":
    _run_nccl_fused_worker()
