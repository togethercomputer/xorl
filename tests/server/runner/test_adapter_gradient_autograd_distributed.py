"""Real NCCL/autograd certification for adapter-gradient ownership.

The worker cases use the production ownership compiler, the public
``ModelRunner.forward_backward`` transaction boundary, and the real
``RunnerDispatcher`` optimizer command including broadcast, handler,
publication commit, and command error synchronization. Expected gradients are
reconstructed from the unsharded mathematical objective rather than from
either implementation path.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from xorl.distributed.parallel_plan import ParallelPlan
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.lora.modules.linear import LoraLinear
from xorl.models.layers.moe.lora import MoEExpertsLoRA, MoELoRAConfig
from xorl.qlora.modules.moe_experts import (
    BlockFP8QLoRAMoeExperts,
    NF4QLoRAMoeExperts,
    NvFP4QLoRAMoeExperts,
    QLoRAMoeExperts,
)
from xorl.server.protocol.operations import OptimStepData
from xorl.server.protocol.orchestrator_runner import RunnerDispatchCommand
from xorl.server.runner.adapters.gradient_ownership import (
    GradientScaleState,
    ReductionAxis,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner
from xorl.server.runner.runner_dispatcher import RunnerDispatcher
from xorl.trainers.training_utils import sync_sp_gradients


pytestmark = [pytest.mark.server, pytest.mark.gpu]


def _session_spec(*, rank: int = 2) -> dict:
    return {
        "base_model": "analytical-adapter-fixture",
        "is_lora": True,
        "lora_config": {"lora_rank": rank, "lora_alpha": rank, "seed": 11},
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


def _manager(model: nn.Module, root: Path) -> LoRAAdapterManager:
    return LoRAAdapterManager(
        model,
        device=torch.device("cuda", torch.cuda.current_device()),
        checkpoint_dir=str(root),
        auto_save_on_eviction=False,
        optimizer_fused=False,
        weight_decay=0.0,
    )


def _runner(model: nn.Module, manager: LoRAAdapterManager) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager
    runner.rank = dist.get_rank()
    runner.world_size = dist.get_world_size()
    runner.train_config = {"max_grad_norm": None}
    runner.lora_config = {"merge_lora_interval": 0}
    runner._accumulated_valid_tokens = {"policy": 2}
    runner._accumulated_active_microbatches = {"policy": 1}
    runner._accumulated_active_voter_total = {"policy": 0}
    runner._use_distsignsgd = False
    runner._check_not_sleeping = lambda _operation: None
    runner._sync_registered_lora_session_spec = lambda _model_id: None
    return runner


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        value = value.to_local()
    wait = getattr(value, "wait", None)
    return wait() if wait is not None else value


def _logical_slice(value: torch.Tensor, layout) -> torch.Tensor:
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(layout.active_global_offset, layout.active_storage_shape, strict=True)
    )
    return value[slices].contiguous()


def _assert_first_adam_step(state, initial, logical_gradients, norm: float, clip: float) -> None:
    coefficient = min(1.0, clip / (norm + 1e-6))
    for name, parameter in state.local_params.items():
        gradient = _logical_slice(logical_gradients[name], state.tensor_layouts[name]).float()
        installed = gradient * coefficient
        expected_parameter = initial[name].float() - 1e-2 * installed / (installed.abs() + 1e-8)
        torch.testing.assert_close(parameter.float(), expected_parameter, rtol=5e-4, atol=1e-4)
        optimizer_state = state.optimizer.state[parameter]
        torch.testing.assert_close(optimizer_state["exp_avg"].float(), installed * 0.1, rtol=5e-4, atol=1e-4)
        torch.testing.assert_close(
            optimizer_state["exp_avg_sq"].float(),
            installed.square() * 0.05,
            rtol=5e-4,
            atol=1e-4,
        )


def _init_nccl() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def _root(case: str, rank: int) -> Path:
    root = Path(tempfile.gettempdir()) / f"adapter-gradient-autograd-{case}-{os.environ['MASTER_PORT']}"
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    dist.barrier()
    return root


def _dispatcher(runner: ModelRunner) -> RunnerDispatcher:
    """Build the real production dispatcher boundary around the fixture runner."""

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.trainer = runner
    dispatcher.rank = dist.get_rank()
    dispatcher.world_size = dist.get_world_size()
    dispatcher.cpu_group = dist.new_group(backend="gloo")
    dispatcher._worker_error = None
    dispatcher._running = True
    dispatcher._protocol = None
    dispatcher._adapter_coordinator = SimpleNamespace(
        auto_load_if_evicted=lambda _model_id: (False, None),
    )
    return dispatcher


def _public_optimizer_step(
    dispatcher: RunnerDispatcher,
    *,
    request_id: str,
    lr: float,
    gradient_clip: float,
) -> float | None:
    """Run the actual broadcast, handler, publication, and error-sync command."""

    dispatcher._running = True
    response = None
    request = RunnerDispatchCommand.create(
        "optim_step",
        OptimStepData(lr=lr, gradient_clip=gradient_clip, model_id="policy"),
        request_id=request_id,
    )
    if dispatcher.rank == 0:
        try:
            response = asyncio.run(dispatcher._handle_request_rank0(request))
        finally:
            # Test-only lifecycle command: the optimizer itself already ran
            # through the production rank-0/worker dispatcher paths.
            dist.broadcast_object_list([{"command": "shutdown"}], src=0, group=dispatcher.cpu_group)
            dispatcher._running = False
    else:
        asyncio.run(dispatcher._worker_event_loop())

    dist.barrier(group=dispatcher.cpu_group)
    if dispatcher.rank != 0:
        return None
    assert response is not None
    assert response.success, response.error
    assert response.result["model_id"] == "policy"
    return float(response.result["grad_norm"])


def _set_linear_values(layer: LoraLinear) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        layer.weight.copy_(
            torch.linspace(-0.2, 0.3, layer.weight.numel(), device=layer.weight.device).reshape_as(layer.weight)
        )
        layer.lora_A.copy_(
            torch.linspace(-0.15, 0.25, layer.lora_A.numel(), device=layer.lora_A.device).reshape_as(layer.lora_A)
        )
        layer.lora_B.copy_(
            torch.linspace(0.2, -0.1, layer.lora_B.numel(), device=layer.lora_B.device).reshape_as(layer.lora_B)
        )
    layer.weight.requires_grad_(False)
    return {
        "weight": layer.weight.detach().clone(),
        "lora_A": layer.lora_A.detach().clone(),
        "lora_B": layer.lora_B.detach().clone(),
    }


def _run_dense() -> None:
    rank, world, device = _init_nccl()
    assert world == 2
    try:
        init_parallel_state(dp_size=world, dp_shard_size=world, device_type="cuda")

        class _DenseModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapter = LoraLinear(4, 4, r=2, lora_alpha=2, device=device, dtype=torch.float32)

        model = _DenseModel()
        global_values = _set_linear_values(model.adapter)
        fully_shard(model.adapter, mesh=get_parallel_state().fsdp_mesh)
        model.adapter.set_gradient_divide_factor(1.0)

        manager = _manager(model, _root("dense", rank))
        manager.register_adapter("policy", session_spec=_session_spec(), initialize_fresh=False)
        runner = _runner(model, manager)
        dispatcher = _dispatcher(runner)
        runner._compile_registered_adapter_gradient_ownership("policy")
        manager.prepare_forward("policy")
        state = manager.get_adapter_state("policy")
        initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
        raw_boundary: dict[str, torch.Tensor] = {}
        local_input = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4) / 10.0 + float(rank) * 0.125

        gathered_inputs = [torch.empty_like(local_input) for _ in range(world)]
        dist.all_gather(gathered_inputs, local_input)
        reference_a = global_values["lora_A"].detach().clone().requires_grad_(True)
        reference_b = global_values["lora_B"].detach().clone().requires_grad_(True)
        reference_loss = torch.zeros((), device=device)
        effective = global_values["weight"] + reference_b @ reference_a
        for value in gathered_inputs:
            reference_loss = reference_loss + torch.nn.functional.linear(value, effective).square().sum()
        reference_a_grad, reference_b_grad = torch.autograd.grad(reference_loss, (reference_a, reference_b))
        logical_gradients = {
            "adapter.lora_A": reference_a_grad,
            "adapter.lora_B": reference_b_grad,
        }

        def _capture(_micro_batches, **_kwargs):
            assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
            model.adapter(local_input).square().sum().backward()
            for name, parameter in model.named_parameters():
                if name in state.tensor_layouts:
                    raw_boundary[name] = _local_tensor(parameter.grad).detach().clone()
            manager.stage_gradient_numerators("policy", denominator=2.0, backward_completed=True)
            return {"loss": 0.0}

        runner._forward_backward_impl = _capture
        runner.forward_backward([], model_id="policy")
        runner.commit_forward_backward_completion("policy")

        for name, raw in raw_boundary.items():
            packed = state.tensor_layouts[name].pack_from_local(raw).float()
            torch.testing.assert_close(
                packed,
                _logical_slice(logical_gradients[name], state.tensor_layouts[name]).float(),
                rtol=5e-4,
                atol=1e-4,
            )

        normalized = {name: gradient / 2.0 for name, gradient in logical_gradients.items()}
        expected_norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in normalized.values())))
        actual_norm = _public_optimizer_step(
            dispatcher,
            request_id="dense-optim-step-1",
            lr=1e-2,
            gradient_clip=0.05,
        )
        if rank == 0:
            assert actual_norm == pytest.approx(expected_norm, rel=5e-4, abs=1e-4)
        _assert_first_adam_step(state, initial, normalized, expected_norm, 0.05)
        assert state.publication_eligible
        assert {item.topology for item in state.gradient_ownership_plan.parameters} == {TopologyFamily.DENSE_REPLICATED}
        if rank == 0:
            print("DENSE_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED", flush=True)
    finally:
        dist.destroy_process_group()


def _dtensor_rectangle(value: DTensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape, offset = compute_local_shape_and_global_offset(tuple(value.shape), value.device_mesh, value.placements)
    return tuple(int(item) for item in offset), tuple(int(item) for item in shape)


def _run_sequence_parallel() -> None:
    rank, world, device = _init_nccl()
    assert world == 2
    try:
        init_parallel_state(
            dp_size=1,
            dp_replicate_size=1,
            dp_shard_size=1,
            ulysses_size=world,
            dp_mode="none",
            cp_fsdp_mode="none",
            device_type="cuda",
        )
        ps = get_parallel_state()
        assert ps.sp_grad_sync_group is not None

        class _SequenceParallelModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapter = LoraLinear(4, 4, r=2, lora_alpha=2, device=device, dtype=torch.float32)

        model = _SequenceParallelModel()
        global_values = _set_linear_values(model.adapter)
        manager = _manager(model, _root("sequence-parallel", rank))
        manager.register_adapter("policy", session_spec=_session_spec(), initialize_fresh=False)
        runner = _runner(model, manager)
        dispatcher = _dispatcher(runner)
        runner._compile_registered_adapter_gradient_ownership("policy")
        manager.prepare_forward("policy")
        state = manager.get_adapter_state("policy")
        initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
        local_input = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4) / 10.0 + rank * 0.2
        gathered_inputs = [torch.empty_like(local_input) for _ in range(world)]
        dist.all_gather(gathered_inputs, local_input)

        reference_a = global_values["lora_A"].detach().clone().requires_grad_(True)
        reference_b = global_values["lora_B"].detach().clone().requires_grad_(True)
        effective = global_values["weight"] + reference_b @ reference_a
        reference_loss = sum(torch.nn.functional.linear(value, effective).square().sum() for value in gathered_inputs)
        reference_a_grad, reference_b_grad = torch.autograd.grad(reference_loss, (reference_a, reference_b))
        logical_gradients = {
            "adapter.lora_A": reference_a_grad,
            "adapter.lora_B": reference_b_grad,
        }
        denominator = float(sum(value.shape[0] for value in gathered_inputs))

        def _capture(_micro_batches, **_kwargs):
            assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
            model.adapter(local_input).square().sum().backward()
            before_legacy_sync = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if name in state.tensor_layouts
            }
            manager.stage_gradient_numerators(
                "policy",
                denominator=denominator,
                backward_completed=True,
            )
            exclusions = manager.adapter_sync_exclusions("policy", ReductionAxis.SEQUENCE_PARALLEL)
            assert exclusions == {
                id(parameter) for name, parameter in model.named_parameters() if name in state.tensor_layouts
            }
            sync_sp_gradients(
                model,
                ps.sp_grad_sync_group,
                excluded_parameter_ids=exclusions,
            )
            for name, parameter in model.named_parameters():
                if name in before_legacy_sync:
                    torch.testing.assert_close(parameter.grad, before_legacy_sync[name])
            return {"loss": 0.0}

        runner._forward_backward_impl = _capture
        runner.forward_backward([], model_id="policy")
        runner.commit_forward_backward_completion("policy")

        normalized = {name: gradient / denominator for name, gradient in logical_gradients.items()}
        expected_norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in normalized.values())))
        actual_norm = _public_optimizer_step(
            dispatcher,
            request_id="sequence-parallel-optim-step-1",
            lr=1e-2,
            gradient_clip=0.05,
        )
        if rank == 0:
            assert actual_norm == pytest.approx(expected_norm, rel=5e-4, abs=1e-4), (
                f"sequence-parallel norm mismatch: actual={actual_norm}, expected={expected_norm}, "
                f"replica_divisors={[item.norm_replica_divisor for item in state.gradient_ownership_plan.parameters]}"
            )
        _assert_first_adam_step(state, initial, normalized, expected_norm, 0.05)
        assert state.publication_eligible
        assert all(
            any(domain.axis is ReductionAxis.SEQUENCE_PARALLEL for domain in item.pending_domains)
            for item in state.gradient_ownership_plan.parameters
        )
        assert state.last_transport_stats.collective_count == 1
        if rank == 0:
            print("SEQUENCE_PARALLEL_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED", flush=True)
    finally:
        dist.destroy_process_group()


def _run_direct_output() -> None:
    rank, world, device = _init_nccl()
    assert world == 4
    try:
        init_parallel_state(
            dp_size=world,
            dp_shard_size=world,
            lm_head_tp_size=2,
            device_type="cuda",
        )
        ps = get_parallel_state()

        class _OutputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lm_head = LoraLinear(4, 8, r=2, lora_alpha=2, device=device, dtype=torch.float32)
                self.lm_head.exact_merged_forward = True

        model = _OutputModel()
        global_values = _set_linear_values(model.lm_head)
        fully_shard(model.lm_head, mesh=ps.lm_head_mesh)
        model.lm_head.set_gradient_divide_factor(1.0)

        manager = _manager(model, _root("direct", rank))
        manager.register_adapter("policy", session_spec=_session_spec(), initialize_fresh=False)
        runner = _runner(model, manager)
        dispatcher = _dispatcher(runner)
        runner._compile_registered_adapter_gradient_ownership("policy")
        manager.prepare_forward("policy")
        state = manager.get_adapter_state("policy")
        initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
        completed_boundary: dict[str, torch.Tensor] = {}
        earliest_placements: dict[str, tuple] = {}

        effective_probe = runner._get_effective_lm_head_weight()
        assert isinstance(effective_probe, DTensor)
        rectangle = _dtensor_rectangle(effective_probe)
        coefficient = float(rank + 1) / 4.0
        rectangles: list[tuple[tuple[int, ...], tuple[int, ...], float, int]] = [None] * world
        dist.all_gather_object(
            rectangles,
            (rectangle[0], rectangle[1], coefficient, int(dist.get_rank(ps.lm_head_tp_replica_group))),
        )

        def _reference_gradients(replica: int | None) -> dict[str, torch.Tensor]:
            reference_a = global_values["lora_A"].detach().clone().requires_grad_(True)
            reference_b = global_values["lora_B"].detach().clone().requires_grad_(True)
            effective = global_values["weight"] + reference_b @ reference_a
            objective = torch.zeros((), device=device)
            for offset, shape, scale, replica_ordinal in rectangles:
                if replica is not None and replica_ordinal != replica:
                    continue
                slices = tuple(slice(start, start + size) for start, size in zip(offset, shape, strict=True))
                objective = objective + effective[slices].sum() * scale
            gradients = torch.autograd.grad(objective, (reference_a, reference_b))
            return {"lm_head.lora_A": gradients[0], "lm_head.lora_B": gradients[1]}

        replica_reference = _reference_gradients(int(dist.get_rank(ps.lm_head_tp_replica_group)))
        logical_gradients = _reference_gradients(None)

        def _capture(_micro_batches, **_kwargs):
            assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
            effective = runner._get_effective_lm_head_weight()
            (_local_tensor(effective) * coefficient).sum().backward()
            original_gradients = {}
            original_local_values = {}
            for name, parameter in model.named_parameters():
                if name in state.tensor_layouts:
                    earliest_placements[name] = tuple(getattr(parameter.grad, "placements", ()))
                    original_gradients[name] = parameter.grad
                    original_local_values[name] = _local_tensor(parameter.grad).detach().clone()
            manager.stage_gradient_numerators("policy", denominator=2.0, backward_completed=True)
            for name, parameter in model.named_parameters():
                if name in state.tensor_layouts:
                    assert parameter.grad is original_gradients[name]
                    torch.testing.assert_close(_local_tensor(parameter.grad), original_local_values[name])
                    completed_boundary[name] = state.gradient_scratch.staged_numerators[name].detach().clone()
            return {"loss": 0.0}

        runner._forward_backward_impl = _capture
        runner.forward_backward([], model_id="policy")
        runner.commit_forward_backward_completion("policy")

        assert any(type(placement).__name__ == "Partial" for placement in earliest_placements["lm_head.lora_A"])

        for name, completed in completed_boundary.items():
            torch.testing.assert_close(
                completed,
                _logical_slice(replica_reference[name], state.tensor_layouts[name]).float(),
                rtol=2e-5,
                atol=2e-5,
            )

        normalized = {name: gradient / 2.0 for name, gradient in logical_gradients.items()}
        expected_norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in normalized.values())))
        actual_norm = _public_optimizer_step(
            dispatcher,
            request_id="direct-output-optim-step-1",
            lr=1e-2,
            gradient_clip=0.05,
        )
        if rank == 0:
            assert actual_norm == pytest.approx(expected_norm, rel=2e-5, abs=2e-5)
        _assert_first_adam_step(state, initial, normalized, expected_norm, 0.05)
        assert state.publication_eligible
        assert {item.topology for item in state.gradient_ownership_plan.parameters} == {
            TopologyFamily.DIRECT_OUTPUT_PROJECTION
        }
        assert state.last_transport_stats.collective_count == 1
        if rank == 0:
            print("DIRECT_OUTPUT_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED", flush=True)
    finally:
        dist.destroy_process_group()


def _set_expert_values(module: MoEExpertsLoRA) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(module.named_parameters()):
            sequence = torch.linspace(
                -0.12 + ordinal * 0.01,
                0.16 + ordinal * 0.01,
                parameter.numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            ).reshape_as(parameter)
            parameter.copy_(sequence)
            if "lora_" not in name:
                parameter.requires_grad_(False)
            values[f"experts.{name}"] = parameter.detach().clone()
    return values


def _expert_reference_gradients(
    values: dict[str, torch.Tensor],
    hidden_batches: list[torch.Tensor],
    expert_batches: list[torch.Tensor],
    *,
    owner_range: range | None,
    scaling: float = 1.0,
) -> dict[str, torch.Tensor]:
    factor_names = tuple(name for name in values if "lora_" in name)
    factors = {name: values[name].detach().clone().requires_grad_(True) for name in factor_names}
    gate_up_base = values["experts.gate_up_proj"]
    gate_base, up_base = gate_up_base.split(gate_up_base.shape[-1] // 2, dim=-1)
    down_base = values["experts.down_proj"]
    first_name, first_factor = next(iter(factors.items()))
    rank = first_factor.shape[-1] if first_name.endswith("_A") else first_factor.shape[-2]

    def _factor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        factor = factors.get(name)
        return factor if factor is not None else gate_base.new_zeros(shape)

    expert_count, hidden_size, intermediate_size = gate_base.shape
    gate_a_all = _factor("experts.gate_proj_lora_A", (1, hidden_size, rank))
    gate_b_all = _factor("experts.gate_proj_lora_B", (expert_count, rank, intermediate_size))
    up_a_all = _factor("experts.up_proj_lora_A", (1, hidden_size, rank))
    up_b_all = _factor("experts.up_proj_lora_B", (expert_count, rank, intermediate_size))
    down_a_all = _factor("experts.down_proj_lora_A", (expert_count, intermediate_size, rank))
    down_b_all = _factor("experts.down_proj_lora_B", (1, rank, hidden_size))
    # Keep the mathematical reference structurally connected to every factor.
    # An EP owner can receive no routed tokens for an entire step; its exact
    # reference gradient is a present zero tensor, not an absent gradient.
    objective = sum((factor.sum() * 0.0 for factor in factors.values()), start=torch.zeros((), device=gate_base.device))
    for hidden, selected in zip(hidden_batches, expert_batches, strict=True):
        for token, expert_tensor in zip(hidden, selected.reshape(-1), strict=True):
            expert = int(expert_tensor.item())
            if owner_range is not None and expert not in owner_range:
                continue
            gate_a = gate_a_all[0 if gate_a_all.shape[0] == 1 else expert]
            gate_b = gate_b_all[expert]
            up_a = up_a_all[0 if up_a_all.shape[0] == 1 else expert]
            up_b = up_b_all[expert]
            down_a = down_a_all[expert]
            down_b = down_b_all[0 if down_b_all.shape[0] == 1 else expert]
            compute_dtype = token.dtype
            gate = (
                token @ gate_base[expert].to(compute_dtype)
                + ((token @ gate_a.to(compute_dtype)) @ gate_b.to(compute_dtype)) * scaling
            )
            up = (
                token @ up_base[expert].to(compute_dtype)
                + ((token @ up_a.to(compute_dtype)) @ up_b.to(compute_dtype)) * scaling
            )
            activated = torch.nn.functional.silu(gate) * up
            output = (
                activated @ down_base[expert].to(compute_dtype)
                + ((activated @ down_a.to(compute_dtype)) @ down_b.to(compute_dtype)) * scaling
            )
            objective = objective + output.square().sum()
    gradients = torch.autograd.grad(objective, tuple(factors.values()))
    return dict(zip(factor_names, gradients, strict=True))


def _reference_adam_step(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, nn.Parameter],
    logical_gradients: dict[str, torch.Tensor],
    *,
    gradient_clip: float,
) -> float:
    norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in logical_gradients.values())))
    coefficient = min(1.0, gradient_clip / (norm + 1e-6))
    for name, parameter in parameters.items():
        parameter.grad = logical_gradients[name].to(parameter).mul(coefficient)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def _assert_adam_matches_reference(
    state,
    reference_parameters: dict[str, nn.Parameter],
    reference_optimizer: torch.optim.Optimizer,
    *,
    rtol: float = 5e-4,
    atol: float = 1e-4,
) -> None:
    for name, parameter in state.local_params.items():
        layout = state.tensor_layouts[name]
        reference_parameter = reference_parameters[name]
        torch.testing.assert_close(
            parameter.float(),
            _logical_slice(reference_parameter.detach(), layout).float(),
            rtol=rtol,
            atol=atol,
        )
        actual_optimizer_state = state.optimizer.state[parameter]
        reference_optimizer_state = reference_optimizer.state[reference_parameter]
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual_optimizer_state[field].float(),
                _logical_slice(reference_optimizer_state[field], layout).float(),
                rtol=rtol,
                atol=atol,
            )
        assert float(actual_optimizer_state["step"].item()) == float(reference_optimizer_state["step"].item())


def _run_experts() -> None:
    rank, world, device = _init_nccl()
    assert world in {2, 4}

    try:
        init_parallel_state(dp_size=world, ep_size=2, dp_mode="none", device_type="cuda")
        ps = get_parallel_state()
        backend = os.environ.get("XORL_ADAPTER_EXPERT_BACKEND", "eager")
        ep_dispatch = os.environ.get("XORL_ADAPTER_EXPERT_DISPATCH", "alltoall")
        target_modules = os.environ.get("XORL_ADAPTER_EXPERT_TARGETS", "gate_proj,up_proj,down_proj").split(",")
        hybrid_shared = os.environ.get("XORL_ADAPTER_EXPERT_HYBRID_SHARED", "1") == "1"

        class _ExpertModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = MoEExpertsLoRA(
                    num_experts=4,
                    hidden_dim=128,
                    intermediate_size=128,
                    moe_implementation=backend,
                    lora_config=MoELoRAConfig(
                        r=8,
                        lora_alpha=8,
                        hybrid_shared=hybrid_shared,
                        target_modules=target_modules,
                    ),
                ).to(device)
                self.experts.ep_dispatch = ep_dispatch
                self._parallel_plan = ParallelPlan(
                    {
                        f"experts.{name}": Shard(0)
                        for name in (
                            "gate_up_proj",
                            "down_proj",
                            "gate_proj_lora_A",
                            "gate_proj_lora_B",
                            "up_proj_lora_A",
                            "up_proj_lora_B",
                            "down_proj_lora_A",
                            "down_proj_lora_B",
                        )
                    }
                )

            def get_parallel_plan(self):
                return self._parallel_plan

        model = _ExpertModel()
        global_values = _set_expert_values(model.experts)
        model._fqn2spec_info = model.get_parallel_plan().apply(model, ps.ep_fsdp_device_mesh)
        fully_shard(
            model.experts,
            mesh=ps.ep_fsdp_device_mesh["ep_fsdp"],
            shard_placement_fn=lambda _parameter: Shard(1),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        model.experts.set_gradient_divide_factor(1.0)

        manager = _manager(model, _root(f"experts-{backend}-{ep_dispatch}", rank))
        manager.lora_config["moe_hybrid_shared_lora"] = hybrid_shared
        manager.register_adapter("policy", session_spec=_session_spec(rank=8), initialize_fresh=False)
        runner = _runner(model, manager)
        dispatcher = _dispatcher(runner)
        runner._compile_registered_adapter_gradient_ownership("policy")
        manager.prepare_forward("policy")
        state = manager.get_adapter_state("policy")
        reference_parameters = {
            name: nn.Parameter(value.detach().clone()) for name, value in global_values.items() if "lora_" in name
        }
        reference_optimizer = torch.optim.AdamW(
            reference_parameters.values(),
            lr=1e-2,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        hidden_base = (
            torch.arange(4 * 128, device=device, dtype=torch.float32).reshape(4, 128) / 2048.0 + rank * 0.01
        ).to(torch.bfloat16)
        routing = torch.ones((4, 1), device=device, dtype=torch.bfloat16)
        owner_start = int(ps.ep_rank) * 2
        for step_index, routed_expert in enumerate((0, 2), start=1):
            manager.prepare_forward("policy")
            hidden = hidden_base + (step_index - 1) * 0.07
            selected = torch.full((4, 1), routed_expert, device=device, dtype=torch.int64)
            hidden_batches = [torch.empty_like(hidden) for _ in range(world)]
            expert_batches = [torch.empty_like(selected) for _ in range(world)]
            dist.all_gather(hidden_batches, hidden)
            dist.all_gather(expert_batches, selected)
            reference_values = {
                **global_values,
                **{name: parameter.detach() for name, parameter in reference_parameters.items()},
            }
            owner_reference = _expert_reference_gradients(
                reference_values,
                hidden_batches,
                expert_batches,
                owner_range=range(owner_start, owner_start + 2),
            )
            logical_gradients = _expert_reference_gradients(
                reference_values,
                hidden_batches,
                expert_batches,
                owner_range=None,
            )
            raw_boundary: dict[str, torch.Tensor] = {}
            zero_token_owner = routed_expert not in range(owner_start, owner_start + 2)

            def _capture(_micro_batches, **_kwargs):
                assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
                model.experts(hidden, routing, selected).square().sum().backward()
                for name, parameter in model.named_parameters():
                    if name in state.tensor_layouts:
                        assert parameter.grad is not None, f"missing structural gradient for {name}"
                        raw_boundary[name] = _local_tensor(parameter.grad).detach().clone()
                if zero_token_owner:
                    assert all(torch.count_nonzero(gradient) == 0 for gradient in raw_boundary.values())
                manager.stage_gradient_numerators("policy", denominator=2.0, backward_completed=True)
                return {"loss": 0.0}

            runner._forward_backward_impl = _capture
            runner.forward_backward([], model_id="policy")
            runner.commit_forward_backward_completion("policy")

            assert raw_boundary.keys() == state.tensor_layouts.keys()
            for name, raw in raw_boundary.items():
                packed = state.tensor_layouts[name].pack_from_local(raw).float()
                expected_packed = _logical_slice(owner_reference[name], state.tensor_layouts[name]).float()
                torch.testing.assert_close(
                    packed,
                    expected_packed,
                    rtol=1.5e-2,
                    atol=2e-3,
                )

            normalized = {name: gradient / 2.0 for name, gradient in logical_gradients.items()}
            expected_norm = _reference_adam_step(
                reference_optimizer,
                reference_parameters,
                normalized,
                gradient_clip=0.05,
            )
            actual_norm = _public_optimizer_step(
                dispatcher,
                request_id=f"{backend}-expert-optim-step-{step_index}",
                lr=1e-2,
                gradient_clip=0.05,
            )
            if rank == 0:
                assert actual_norm == pytest.approx(expected_norm, rel=1.5e-2, abs=2e-3)
            _assert_adam_matches_reference(
                state,
                reference_parameters,
                reference_optimizer,
                rtol=1.5e-2,
                atol=2e-3,
            )
            assert state.publication_eligible
            assert state.global_step == step_index

        expected_topologies = {TopologyFamily.OWNER_SHARDED}
        if hybrid_shared:
            expected_topologies.add(TopologyFamily.EP_REPLICATED_SHARED)
        assert {item.topology for item in state.gradient_ownership_plan.parameters} == expected_topologies
        assert state.last_transport_stats.collective_count == (1 if hybrid_shared else 0)
        if rank == 0:
            target_suffix = (
                "" if target_modules == ["gate_proj", "up_proj", "down_proj"] else ":" + ",".join(target_modules)
            )
            layout_suffix = "" if hybrid_shared else ":all_owner"
            print(
                f"EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:"
                f"{backend}:{ep_dispatch}{target_suffix}{layout_suffix}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def _quantized_expert_class(quant_format: str) -> type[QLoRAMoeExperts]:
    try:
        return {
            "block_fp8": BlockFP8QLoRAMoeExperts,
            "nf4": NF4QLoRAMoeExperts,
            "nvfp4": NvFP4QLoRAMoeExperts,
        }[quant_format]
    except KeyError as error:
        raise AssertionError(f"Unsupported quantized expert fixture format {quant_format!r}") from error


def _set_quantized_expert_values(module: QLoRAMoeExperts, *, rank: int) -> dict[str, torch.Tensor]:
    """Install deterministic local quantized bases and global LoRA factors."""

    values: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(module.named_parameters()):
            sequence = torch.linspace(
                -0.08 + ordinal * 0.005,
                0.11 + ordinal * 0.005,
                parameter.numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            ).reshape_as(parameter)
            parameter.copy_(sequence)
            values[f"experts.{name}"] = parameter.detach().clone()

        for ordinal, (projection, in_features, out_features) in enumerate(
            (
                ("gate", module.hidden_size, module.intermediate_size),
                ("up", module.hidden_size, module.intermediate_size),
                ("down", module.intermediate_size, module.hidden_size),
            )
        ):
            base = torch.linspace(
                -0.09 + rank * 0.004 + ordinal * 0.007,
                0.13 + rank * 0.004 + ordinal * 0.007,
                in_features * out_features,
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ).reshape(1, in_features, out_features)
            module._quantize_proj(projection, base)
    module._weights_loaded = True
    return values


def _gather_quantized_expert_bases(module: QLoRAMoeExperts, world: int) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, local in (
        ("experts.gate_proj", module.gate_proj),
        ("experts.up_proj", module.up_proj),
        ("experts.down_proj", module.down_proj),
    ):
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        values[name] = torch.cat(gathered, dim=0)
    return values


def _quantized_expert_reference_gradients(
    values: dict[str, torch.Tensor],
    hidden_batches: list[torch.Tensor],
    expert_batches: list[torch.Tensor],
    *,
    owner_range: range | None,
    scaling: float,
) -> dict[str, torch.Tensor]:
    reference_values = dict(values)
    reference_values["experts.gate_up_proj"] = torch.cat(
        (reference_values.pop("experts.gate_proj"), reference_values.pop("experts.up_proj")),
        dim=-1,
    )
    return _expert_reference_gradients(
        reference_values,
        hidden_batches,
        expert_batches,
        owner_range=owner_range,
        scaling=scaling,
    )


def _run_quantized_experts() -> None:
    rank, world, device = _init_nccl()
    assert world == 2
    try:
        # QLoRA expert factors are intentionally excluded from FSDP.  EP spans
        # the world here so there is no unsupported eFSDP replica dimension.
        init_parallel_state(dp_size=world, ep_size=world, dp_mode="none", device_type="cuda")
        ps = get_parallel_state()
        assert ps.dp_shard_in_ep_size == 1
        backend = os.environ["XORL_ADAPTER_EXPERT_BACKEND"]
        quant_format = os.environ["XORL_ADAPTER_EXPERT_QUANT_FORMAT"]
        ep_dispatch = os.environ.get("XORL_ADAPTER_EXPERT_DISPATCH", "alltoall")
        target_modules = os.environ.get("XORL_ADAPTER_EXPERT_TARGETS", "gate_proj,up_proj,down_proj").split(",")
        qlora_class = _quantized_expert_class(quant_format)
        lora_rank = 8

        class _QuantizedExpertModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = qlora_class(
                    num_local_experts=1,
                    num_experts=world,
                    intermediate_size=128,
                    hidden_size=128,
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    expert_offset=rank,
                    device=device,
                    moe_implementation=backend,
                    hybrid_shared=True,
                    target_modules=target_modules,
                    ep_dispatch=ep_dispatch,
                    deepep_buffer_size_gb=0.1,
                )
                self._parallel_plan = ParallelPlan(
                    {f"experts.{name}": Shard(0) for name, _parameter in self.experts.named_parameters()}
                )

            def get_parallel_plan(self):
                return self._parallel_plan

        model = _QuantizedExpertModel()
        global_values = _set_quantized_expert_values(model.experts, rank=rank)
        global_values.update(_gather_quantized_expert_bases(model.experts, world))
        model._fqn2spec_info = model.get_parallel_plan().apply(model, ps.ep_fsdp_device_mesh)

        manager = _manager(
            model,
            _root(f"quantized-experts-{backend}-{quant_format}-{ep_dispatch}", rank),
        )
        manager.lora_config["moe_hybrid_shared_lora"] = True
        manager.register_adapter(
            "policy",
            session_spec=_session_spec(rank=lora_rank),
            initialize_fresh=False,
        )
        runner = _runner(model, manager)
        dispatcher = _dispatcher(runner)
        runner._compile_registered_adapter_gradient_ownership("policy")
        manager.prepare_forward("policy")
        state = manager.get_adapter_state("policy")
        reference_parameters = {
            name: nn.Parameter(value.detach().clone()) for name, value in global_values.items() if "lora_" in name
        }
        reference_optimizer = torch.optim.AdamW(
            reference_parameters.values(),
            lr=1e-2,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        hidden_base = (
            torch.arange(4 * 128, device=device, dtype=torch.float32).reshape(4, 128) / 2048.0 + rank * 0.01
        ).to(torch.bfloat16)
        routing = torch.ones((4, 1), device=device, dtype=torch.bfloat16)
        for step_index, routed_expert in enumerate((0, world - 1), start=1):
            manager.prepare_forward("policy")
            hidden = hidden_base + (step_index - 1) * 0.015625
            selected = torch.full((4, 1), routed_expert, device=device, dtype=torch.int64)
            hidden_batches = [torch.empty_like(hidden) for _ in range(world)]
            expert_batches = [torch.empty_like(selected) for _ in range(world)]
            dist.all_gather(hidden_batches, hidden)
            dist.all_gather(expert_batches, selected)
            reference_values = {
                **global_values,
                **{name: parameter.detach() for name, parameter in reference_parameters.items()},
            }
            owner_reference = _quantized_expert_reference_gradients(
                reference_values,
                hidden_batches,
                expert_batches,
                owner_range=range(rank, rank + 1),
                scaling=model.experts.scaling,
            )
            logical_gradients = _quantized_expert_reference_gradients(
                reference_values,
                hidden_batches,
                expert_batches,
                owner_range=None,
                scaling=model.experts.scaling,
            )
            raw_boundary: dict[str, torch.Tensor] = {}

            def _capture(_micro_batches, **_kwargs):
                assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
                model.experts(hidden, routing, selected).float().square().sum().backward()
                for name, parameter in model.named_parameters():
                    if name in state.tensor_layouts:
                        assert parameter.grad is not None, f"missing structural gradient for {name}"
                        raw_boundary[name] = _local_tensor(parameter.grad).detach().clone()
                if routed_expert != rank:
                    assert all(torch.count_nonzero(gradient) == 0 for gradient in raw_boundary.values())
                manager.stage_gradient_numerators("policy", denominator=2.0, backward_completed=True)
                return {"loss": 0.0}

            runner._forward_backward_impl = _capture
            runner.forward_backward([], model_id="policy")
            runner.commit_forward_backward_completion("policy")

            assert raw_boundary.keys() == state.tensor_layouts.keys()
            for name, raw in raw_boundary.items():
                packed = state.tensor_layouts[name].pack_from_local(raw).float()
                torch.testing.assert_close(
                    packed,
                    _logical_slice(owner_reference[name], state.tensor_layouts[name]).float(),
                    rtol=1.5e-2,
                    atol=2e-3,
                )

            normalized = {name: gradient / 2.0 for name, gradient in logical_gradients.items()}
            expected_norm = _reference_adam_step(
                reference_optimizer,
                reference_parameters,
                normalized,
                gradient_clip=0.05,
            )
            actual_norm = _public_optimizer_step(
                dispatcher,
                request_id=f"{backend}-{quant_format}-expert-optim-step-{step_index}",
                lr=1e-2,
                gradient_clip=0.05,
            )
            if rank == 0:
                assert actual_norm == pytest.approx(expected_norm, rel=1.5e-2, abs=2e-3)
            _assert_adam_matches_reference(state, reference_parameters, reference_optimizer)
            assert state.publication_eligible
            assert state.global_step == step_index

        assert {item.topology for item in state.gradient_ownership_plan.parameters} == {
            TopologyFamily.EP_REPLICATED_SHARED,
            TopologyFamily.OWNER_SHARDED,
        }
        assert state.last_transport_stats.collective_count == 1
        if rank == 0:
            target_suffix = (
                "" if target_modules == ["gate_proj", "up_proj", "down_proj"] else ":" + ",".join(target_modules)
            )
            print(
                f"QUANTIZED_EXPERT_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:"
                f"{backend}:{quant_format}:{ep_dispatch}{target_suffix}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_dense_real_fsdp_autograd_matches_analytical_optimizer_step() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_ADAPTER_AUTOGRAD_CASE": "dense"},
    )
    result.assert_success("dense real-autograd ownership certification")
    assert "DENSE_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED" in result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
def test_direct_output_real_fsdp_autograd_matches_analytical_optimizer_step() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=4,
        timeout=150,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "direct",
        },
    )
    result.assert_success("direct-output real-autograd ownership certification")
    assert "DIRECT_OUTPUT_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED" in result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_sequence_parallel_real_autograd_matches_analytical_optimizer_step() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_ADAPTER_AUTOGRAD_CASE": "sequence_parallel"},
    )
    result.assert_success("sequence-parallel real-autograd ownership certification")
    assert "SEQUENCE_PARALLEL_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED" in result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
@pytest.mark.parametrize("backend", ("eager", "triton", "native", "quack"))
def test_expert_shared_and_owner_real_fsdp_autograd_match_analytical_optimizer_step(backend: str) -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=4,
        timeout=300,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "experts",
            "XORL_ADAPTER_EXPERT_BACKEND": backend,
        },
    )
    result.assert_success(f"{backend} expert shared/owner real-autograd ownership certification")
    assert (
        f"EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:{backend}:alltoall"
        in result.stdout
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.parametrize("backend", ("eager", "triton", "native", "quack"))
def test_expert_backend_real_autograd_matches_analytical_optimizer_step(backend: str) -> None:
    """Qualify backend math at EP2; the four-rank gate separately adds eFSDP2."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=300,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "experts",
            "XORL_ADAPTER_EXPERT_BACKEND": backend,
        },
    )
    result.assert_success(f"{backend} expert EP2 real-autograd ownership certification")
    assert (
        f"EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:{backend}:alltoall"
        in result.stdout
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.skipif(importlib.util.find_spec("deep_ep") is None, reason="DeepEP is not installed")
@pytest.mark.parametrize("hybrid_shared", (True, False))
def test_unquantized_quack_deepep_real_autograd_matches_analytical_optimizer_step(hybrid_shared: bool) -> None:
    """Qualify the shipped Quack+DeepEP expert-LoRA composition."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=300,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "experts",
            "XORL_ADAPTER_EXPERT_BACKEND": "quack",
            "XORL_ADAPTER_EXPERT_DISPATCH": "deepep",
            "XORL_ADAPTER_EXPERT_HYBRID_SHARED": "1" if hybrid_shared else "0",
        },
    )
    result.assert_success("quack DeepEP expert EP2 real-autograd ownership certification")
    suffix = "" if hybrid_shared else ":all_owner"
    assert (
        "EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:quack:deepep" + suffix
        in result.stdout
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs")
def test_unquantized_quack_all_owner_real_fsdp_autograd_matches_analytical_optimizer_step() -> None:
    """Qualify the restored all-owner Quack layout with an eFSDP dimension."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=4,
        timeout=300,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "experts",
            "XORL_ADAPTER_EXPERT_BACKEND": "quack",
            "XORL_ADAPTER_EXPERT_HYBRID_SHARED": "0",
        },
    )
    result.assert_success("quack all-owner expert eFSDP2 real-autograd ownership certification")
    assert (
        "EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:"
        "quack:alltoall:all_owner" in result.stdout
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_unquantized_expert_projection_subset_matches_analytical_optimizer_step() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=300,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "experts",
            "XORL_ADAPTER_EXPERT_BACKEND": "quack",
            "XORL_ADAPTER_EXPERT_TARGETS": "down_proj",
        },
    )
    result.assert_success("quack exact expert projection-subset certification")
    assert (
        "EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:"
        "quack:alltoall:down_proj" in result.stdout
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.parametrize("backend", ("eager", "triton", "native", "quack"))
def test_unquantized_all_owner_layout_matches_analytical_optimizer_step(backend: str) -> None:
    """Qualify the default non-hybrid layout, including restored Quack recipes."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=300,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "experts",
            "XORL_ADAPTER_EXPERT_BACKEND": backend,
            "XORL_ADAPTER_EXPERT_HYBRID_SHARED": "0",
        },
    )
    result.assert_success(f"{backend} all-owner expert EP2 real-autograd ownership certification")
    assert (
        "EXPERT_SHARED_OWNER_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:"
        f"{backend}:alltoall:all_owner" in result.stdout
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.parametrize("backend", ("triton", "quack", "native"))
@pytest.mark.parametrize("quant_format", ("nf4", "nvfp4", "block_fp8"))
def test_quantized_expert_shared_and_owner_real_autograd_matches_analytical_optimizer_step(
    backend: str,
    quant_format: str,
) -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=240,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "quantized_experts",
            "XORL_ADAPTER_EXPERT_BACKEND": backend,
            "XORL_ADAPTER_EXPERT_QUANT_FORMAT": quant_format,
        },
    )
    result.assert_success(
        f"{backend} {quant_format} quantized expert shared/owner real-autograd ownership certification"
    )
    assert (
        f"QUANTIZED_EXPERT_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:{backend}:{quant_format}:alltoall"
    ) in result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
@pytest.mark.skipif(importlib.util.find_spec("deep_ep") is None, reason="DeepEP is not installed")
def test_quantized_expert_deepep_real_autograd_matches_analytical_optimizer_step() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=240,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "quantized_experts",
            "XORL_ADAPTER_EXPERT_BACKEND": "quack",
            "XORL_ADAPTER_EXPERT_QUANT_FORMAT": "nf4",
            "XORL_ADAPTER_EXPERT_DISPATCH": "deepep",
        },
    )
    result.assert_success("quack NF4 DeepEP quantized expert ownership certification")
    assert (
        "QUANTIZED_EXPERT_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:quack:nf4:deepep"
    ) in result.stdout


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_quantized_expert_projection_subset_matches_analytical_optimizer_step() -> None:
    """The backend's structural zero factors must use EP-local owner dimensions."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=240,
        extra_env={
            "XORL_ADAPTER_AUTOGRAD_CASE": "quantized_experts",
            "XORL_ADAPTER_EXPERT_BACKEND": "quack",
            "XORL_ADAPTER_EXPERT_QUANT_FORMAT": "nf4",
            "XORL_ADAPTER_EXPERT_TARGETS": "down_proj",
        },
    )
    result.assert_success("quack NF4 exact expert projection-subset certification")
    assert (
        "QUANTIZED_EXPERT_TWO_STEP_ZERO_TOKEN_REAL_AUTOGRAD_PUBLIC_OPTIM_CERTIFIED:quack:nf4:alltoall:down_proj"
    ) in result.stdout


if os.environ.get("XORL_ADAPTER_AUTOGRAD_CASE") == "dense":
    _run_dense()
elif os.environ.get("XORL_ADAPTER_AUTOGRAD_CASE") == "sequence_parallel":
    _run_sequence_parallel()
elif os.environ.get("XORL_ADAPTER_AUTOGRAD_CASE") == "direct":
    _run_direct_output()
elif os.environ.get("XORL_ADAPTER_AUTOGRAD_CASE") == "experts":
    _run_experts()
elif os.environ.get("XORL_ADAPTER_AUTOGRAD_CASE") == "quantized_experts":
    _run_quantized_experts()
