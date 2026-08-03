"""Synthetic four-rank EP x eFSDP adapter-bank mechanics and checkpoint gate.

This test validates layout/optimizer/checkpoint algebra with real DeviceMeshes
and DTensors. It intentionally does not claim to exercise fully_shard,
MoE dispatch/combine, or the native hybrid-shared autograd kernel; those need a
separate model-level production-topology test.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from xorl.distributed.parallel_plan import SpecInfo
from xorl.distributed.parallel_state import init_parallel_state
from xorl.server.runner.adapters.manager import (
    LoRAAdapterManager,
    save_adapter_optimizer_shards,
)
from xorl.server.runner.adapters.sharded_state import pack_logical_tensor
from xorl.server.session_spec import write_session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _session_spec() -> dict:
    return {
        "base_model": "synthetic-topology-model",
        "is_lora": True,
        "lora_config": {"lora_rank": 2, "lora_alpha": 4, "seed": 17},
        "optimizer_config": {
            "type": "adamw",
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "optimizer_dtype": "fp32",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "optimizer_kwargs": {},
        },
    }


class _TopologyModel(nn.Module):
    """Synthetic post-parallelization model with ordinary, EP, and shared factors."""

    def __init__(self, mesh: DeviceMesh, rank: int) -> None:
        super().__init__()
        self._fqn2spec_info = {}
        self._ep_patterns = {
            "expert.lora_A": Shard(0),
            "expert.lora_B": Shard(0),
            "shared.lora_A": Shard(0),
            "shared.lora_B": Shard(0),
        }
        self.ordinary = nn.Module()
        self.expert = nn.Module()
        self.shared = nn.Module()

        self._register_dtensor(
            self.ordinary,
            "lora_A",
            global_shape=(4, 4),
            placements=(Replicate(), Shard(0)),
            mesh=mesh,
            spec=SpecInfo(mesh, Replicate(), "ordinary.lora_A"),
            rank=rank,
        )
        self._register_dtensor(
            self.ordinary,
            "lora_B",
            global_shape=(4, 4),
            placements=(Replicate(), Shard(1)),
            mesh=mesh,
            spec=SpecInfo(mesh, Replicate(), "ordinary.lora_B"),
            rank=rank,
        )
        # Expert factors are EP-local in dimension 0, then eFSDP-sharded in
        # dimension 1. The DTensor global shape is deliberately EP-local.
        for name in ("lora_A", "lora_B"):
            self._register_dtensor(
                self.expert,
                name,
                global_shape=(2, 4, 4),
                placements=(Replicate(), Shard(1)),
                mesh=mesh,
                spec=SpecInfo(mesh, Shard(0), f"expert.{name}"),
                rank=rank,
            )
        # Shared factors are replicated across EP and eFSDP-sharded over the
        # second tensor dimension. Explicit plan membership makes Replicate()
        # a real EP replica rather than a generic annotation.
        for name in ("lora_A", "lora_B"):
            self._register_dtensor(
                self.shared,
                name,
                global_shape=(1, 4, 4),
                placements=(Replicate(), Shard(1)),
                mesh=mesh,
                spec=SpecInfo(mesh, Replicate(), f"shared.{name}"),
                rank=rank,
            )

    def get_parallel_plan(self):
        return SimpleNamespace(ep_plan=self._ep_patterns)

    @staticmethod
    def _register_dtensor(
        module: nn.Module,
        name: str,
        *,
        global_shape: tuple[int, ...],
        placements: tuple[object, ...],
        mesh: DeviceMesh,
        spec: SpecInfo,
        rank: int,
    ) -> None:
        local_shape = list(global_shape)
        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                local_shape[placement.dim] //= mesh.size(mesh_dim)
        local = torch.arange(torch.tensor(local_shape).prod().item(), dtype=torch.float32).reshape(local_shape)
        local = local + float(rank) * 1000.0
        stride = tuple(torch.empty(global_shape).stride())
        dtensor = DTensor.from_local(
            local,
            mesh,
            list(placements),
            shape=global_shape,
            stride=stride,
            run_check=False,
        )
        module.register_parameter(name, nn.Parameter(dtensor))


def _build_model(mesh: DeviceMesh, rank: int) -> _TopologyModel:
    model = _TopologyModel(mesh, rank)
    model._fqn2spec_info = {}
    for name, param in model.named_parameters():
        if name.startswith("ordinary."):
            model._fqn2spec_info[name] = SpecInfo(mesh, Replicate(), name)
        elif name.startswith("expert."):
            model._fqn2spec_info[name] = SpecInfo(mesh, Shard(0), name)
        else:
            model._fqn2spec_info[name] = SpecInfo(mesh, Replicate(), name)
    return model


def _build_manager(model: nn.Module, root: Path) -> LoRAAdapterManager:
    return LoRAAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(root / "adapters"),
        auto_save_on_eviction=False,
        lora_config={"base_model": "synthetic-topology-model"},
        optimizer_type="adamw",
        optimizer_dtype="fp32",
        optimizer_fused=False,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _logical_grads(manager: LoRAAdapterManager, model_id: str, scale: float) -> dict[str, torch.Tensor]:
    state = manager.get_adapter_state(model_id)
    return {
        name: torch.linspace(0.01, 0.02, int(torch.tensor(layout.logical_shape).prod()))
        .reshape(layout.logical_shape)
        .mul_(scale)
        .to(dtype=param.dtype)
        for name, (layout, param) in {
            name: (state.tensor_layouts[name], state.local_params[name]) for name in state.local_params
        }.items()
    }


def _capture_logical_gradients(
    manager: LoRAAdapterManager, model_id: str, logical_grads: dict[str, torch.Tensor]
) -> None:
    state = manager.get_adapter_state(model_id)
    model_params = dict(manager.model.named_parameters())
    manager.prepare_forward(model_id)
    for name, model_param in model_params.items():
        layout = state.tensor_layouts[name]
        model_dtensor = model_param.data
        local = layout.unpack_to_local(pack_logical_tensor(layout, logical_grads[name]))
        model_param.grad = DTensor.from_local(
            local,
            model_dtensor.device_mesh,
            list(model_dtensor.placements),
            shape=tuple(model_dtensor.shape),
            stride=tuple(model_dtensor.stride()),
            run_check=False,
        )

    original_full_tensor = DTensor.full_tensor
    DTensor.full_tensor = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("capture_gradients must not call DTensor.full_tensor")
    )
    try:
        manager.capture_gradients(model_id)
    finally:
        DTensor.full_tensor = original_full_tensor
    assert all(param.grad is None for param in model_params.values())


def _gather_logical_state(manager: LoRAAdapterManager, model_id: str) -> dict[str, torch.Tensor] | None:
    state = manager.get_adapter_state(model_id)
    payload = {
        name: {
            "logical_shape": layout.logical_shape,
            "offset": layout.active_global_offset,
            "storage_shape": layout.active_storage_shape,
            "replica_key": layout.replica_key,
            "value": param.detach().cpu().contiguous(),
        }
        for name, (layout, param) in {
            name: (state.tensor_layouts[name], state.local_params[name]) for name in state.local_params
        }.items()
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() != 0:
        return None
    logical: dict[str, torch.Tensor] = {}
    replicas: dict[tuple, torch.Tensor] = {}
    for rank_payload in gathered:
        for name, item in rank_payload.items():
            if name not in logical:
                logical[name] = torch.zeros(item["logical_shape"], dtype=item["value"].dtype)
            if item["value"].numel() == 0:
                continue
            slices = tuple(
                slice(offset, offset + size) for offset, size in zip(item["offset"], item["storage_shape"], strict=True)
            )
            key = tuple(item["replica_key"])
            if key in replicas:
                assert torch.equal(replicas[key], item["value"]), f"replica diverged for {name}"
            else:
                replicas[key] = item["value"]
                logical[name][slices].copy_(item["value"])
    return logical


def _gather_logical_optimizer_fields(
    manager: LoRAAdapterManager, model_id: str, field_names: tuple[str, ...] = ("exp_avg", "exp_avg_sq")
) -> dict[str, dict[str, torch.Tensor]] | None:
    state = manager.get_adapter_state(model_id)
    payload = {}
    for name, param in state.local_params.items():
        optimizer_state = state.optimizer.state.get(param, {})
        payload[name] = {
            "logical_shape": state.tensor_layouts[name].logical_shape,
            "offset": state.tensor_layouts[name].active_global_offset,
            "storage_shape": state.tensor_layouts[name].active_storage_shape,
            "replica_key": state.tensor_layouts[name].replica_key,
            "fields": {
                field: optimizer_state[field].detach().cpu().contiguous()
                for field in field_names
                if field in optimizer_state
            },
        }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    if dist.get_rank() != 0:
        return None
    logical: dict[str, dict[str, torch.Tensor]] = {}
    replicas: dict[tuple, dict[str, torch.Tensor]] = {}
    for rank_payload in gathered:
        for name, item in rank_payload.items():
            if name not in logical:
                logical[name] = {
                    field: torch.zeros(item["logical_shape"], dtype=item["fields"][field].dtype)
                    for field in field_names
                    if field in item["fields"]
                }
            slices = tuple(
                slice(offset, offset + size) for offset, size in zip(item["offset"], item["storage_shape"], strict=True)
            )
            key = tuple(item["replica_key"])
            if key in replicas:
                for field, value in item["fields"].items():
                    assert torch.equal(replicas[key][field], value), f"optimizer replica diverged for {name}"
            else:
                replicas[key] = item["fields"]
                for field, value in item["fields"].items():
                    if value.numel():
                        logical[name][field][slices].copy_(value)
    return logical


def _reference_adamw_step(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, nn.Parameter],
    logical_grads: dict[str, torch.Tensor],
    *,
    accumulation_count: int,
    max_norm: float,
) -> float:
    scaled = {name: gradient * accumulation_count for name, gradient in logical_grads.items()}
    total_norm = torch.sqrt(sum(gradient.float().square().sum() for gradient in scaled.values()))
    norm = float(total_norm.item())
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    for name, parameter in parameters.items():
        parameter.grad = scaled[name].to(parameter.dtype).mul(coefficient)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def _collective_assert(condition: bool, message: str) -> None:
    flag = torch.tensor([1 if condition else 0], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise AssertionError(message)


def test_synthetic_four_rank_ep_efsdp_adapter_bank_and_resume():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=4,
        timeout=180,
        extra_env={"XORL_TOPOLOGY_INTEGRATION_WORKER": "1"},
    )
    result.assert_success("synthetic four-rank EP2 x eFSDP2 adapter integration")


def _run_topology_integration_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 4
    init_parallel_state(dp_size=world, dp_mode="none", device_type="cpu")
    mesh = DeviceMesh(
        "cpu",
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        mesh_dim_names=("ep", "ep_fsdp"),
    )
    root = Path("/tmp") / f"xorl-topology-integration-{os.environ.get('MASTER_PORT', 'default')}"
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    dist.barrier()

    os.environ["XORL_VALIDATE_ADAPTER_REPLICAS"] = "1"
    model = _build_model(mesh, rank)
    manager = _build_manager(model, root)
    manager.register_adapter("topology", session_spec=_session_spec(), initialize_fresh=True)
    state = manager.get_adapter_state("topology")
    assert state.tensor_layouts["expert.lora_A"].substrate_shape == (4, 4, 4)
    assert state.tensor_layouts["expert.lora_A"].local_substrate_shape == (2, 2, 4)
    assert state.tensor_layouts["shared.lora_A"].replica_count == 2
    assert state.tensor_layouts["ordinary.lora_A"].replica_count == 2
    assert state.tensor_layouts["ordinary.lora_A"].local_logical_offset[0] in {0, 2}
    local_slot_elements = sum(param.numel() for param in state.local_params.values())
    full_substrate_elements = sum(
        int(torch.tensor(layout.substrate_shape).prod().item()) for layout in state.tensor_layouts.values()
    )
    assert local_slot_elements < full_substrate_elements
    if rank == 0:
        print(
            "Topology adapter accounting: "
            f"local_slot_elements={local_slot_elements} full_substrate_elements={full_substrate_elements}"
        )

    initial = _gather_logical_state(manager, "topology")
    reference_parameters = None
    reference_optimizer = None
    if rank == 0:
        assert initial is not None
        assert not torch.equal(initial["expert.lora_A"][0], initial["expert.lora_A"][2])
        reference_parameters = {name: nn.Parameter(value.clone()) for name, value in initial.items()}
        reference_optimizer = torch.optim.AdamW(
            reference_parameters.values(),
            lr=1e-2,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

    grads_one = _logical_grads(manager, "topology", 1.0)
    _capture_logical_gradients(manager, "topology", grads_one)
    _capture_logical_gradients(manager, "topology", grads_one)
    first_norm = manager.optim_step("topology", lr=1e-2, gradient_clip=0.1)
    _collective_assert(first_norm > 0.0, "first distributed logical norm was not positive")
    if rank == 0:
        assert reference_parameters is not None and reference_optimizer is not None
        reference_first_norm = _reference_adamw_step(
            reference_optimizer,
            reference_parameters,
            grads_one,
            accumulation_count=2,
            max_norm=0.1,
        )
        assert first_norm == pytest.approx(reference_first_norm, rel=1e-6, abs=1e-6)

    grads_two = _logical_grads(manager, "topology", 2.0)
    _capture_logical_gradients(manager, "topology", grads_two)
    second_norm = manager.optim_step("topology", lr=1e-2, gradient_clip=0.1)
    _collective_assert(second_norm > 0.0, "second distributed logical norm was not positive")
    _collective_assert(state.global_step == 2, "not every rank completed two optimizer steps")

    after = _gather_logical_state(manager, "topology")
    optimizer_after = _gather_logical_optimizer_fields(manager, "topology")
    if rank == 0:
        assert after is not None
        assert optimizer_after is not None
        assert not torch.equal(after["expert.lora_A"][0], after["expert.lora_A"][2])
        # The shared logical rectangle is gathered once; the gather helper also
        # checks every duplicate descriptor for exact equality.
        assert after["shared.lora_A"].shape == (1, 4, 2)
        assert reference_parameters is not None and reference_optimizer is not None
        _reference_adamw_step(
            reference_optimizer,
            reference_parameters,
            grads_two,
            accumulation_count=1,
            max_norm=0.1,
        )
        for name, parameter in reference_parameters.items():
            assert torch.allclose(after[name], parameter.detach(), rtol=2e-6, atol=2e-6), name
            for field in ("exp_avg", "exp_avg_sq"):
                assert torch.allclose(
                    optimizer_after[name][field],
                    reference_optimizer.state[parameter][field],
                    rtol=2e-6,
                    atol=2e-6,
                ), f"{name} {field}"

    local_runtime_elements = sum(param.numel() for param in state.local_params.values()) + sum(
        value.numel()
        for parameter_state in state.optimizer.state.values()
        for value in parameter_state.values()
        if isinstance(value, torch.Tensor)
    )
    full_logical_runtime_elements = sum(
        int(torch.tensor(layout.logical_shape).prod().item()) * 3 for layout in state.tensor_layouts.values()
    )
    assert local_runtime_elements < full_logical_runtime_elements
    if rank == 0:
        print(
            "Topology adapter/Adam accounting: "
            f"local_runtime_elements={local_runtime_elements} "
            f"full_logical_runtime_elements={full_logical_runtime_elements}"
        )

    no_clip_state_id = "no_clip"
    manager.register_adapter(no_clip_state_id, session_spec=_session_spec(), initialize_fresh=True)
    manager.load_logical_gradients(
        no_clip_state_id, _logical_grads(manager, no_clip_state_id, 1000.0), accumulate=False
    )
    large_norm = manager.optim_step(no_clip_state_id, lr=1e-2, gradient_clip=None)
    _collective_assert(large_norm > 13.0, "disabled clipping unexpectedly capped a large logical norm")

    for param in state.local_params.values():
        param.grad = torch.full_like(param, float("nan") if rank == 0 else 1.0)
    rejected = False
    try:
        manager.optim_step("topology", lr=1e-2, gradient_clip=None)
    except FloatingPointError:
        rejected = True
    rejection_flags = torch.tensor([int(rejected)], dtype=torch.int64)
    dist.all_reduce(rejection_flags, op=dist.ReduceOp.MIN)
    assert bool(rejection_flags.item())
    _collective_assert(state.global_step == 2, "a rank advanced after the collective non-finite gate")
    assert all(param.grad is None for param in state.local_params.values())

    checkpoint = root / "checkpoint"
    if rank == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    save_adapter_optimizer_shards(state, str(checkpoint))
    dist.barrier()
    if rank == 0:
        assert after is not None
        save_file(
            {f"base_model.model.{name}": value.contiguous() for name, value in after.items()},
            str(checkpoint / "adapter_model.safetensors"),
        )
        (checkpoint / "metadata.json").write_text(
            json.dumps({"global_step": 2, "lr": 1e-2, "save_optimizer": True}),
            encoding="utf-8",
        )
        write_session_spec(checkpoint, _session_spec())
    dist.barrier()

    target_model = _build_model(mesh, rank)
    target_manager = _build_manager(target_model, root)
    target_manager.register_adapter("topology", session_spec=_session_spec(), initialize_fresh=True)
    original_broadcast = dist.broadcast
    dist.broadcast = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("distributed adapter restore must not broadcast heterogeneous raw tensor bytes")
    )
    try:
        target_manager.load_adapter_state("topology", "checkpoint", load_optimizer=True)
    finally:
        dist.broadcast = original_broadcast
    target_state = target_manager.get_adapter_state("topology")
    local_equal = all(
        torch.equal(target_state.local_params[name], state.local_params[name]) for name in state.local_params
    )
    _collective_assert(local_equal, "same-topology logical weight/optimizer checkpoint resume diverged")
    _collective_assert(target_state.global_step == 2, "checkpoint step counter did not round-trip")

    dist.barrier()
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    dist.barrier()
    dist.destroy_process_group()


if os.environ.get("XORL_TOPOLOGY_INTEGRATION_WORKER") == "1":
    _run_topology_integration_worker()
