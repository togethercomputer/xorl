"""Real-FSDP heterogeneous adapter lifecycle certification.

Defect class: a short active LoRA rank can leave a rank with an empty local
adapter rectangle.  Persistent gradient slots, streamed capture state, or Adam
state from an interleaved adapter must not leak across either session or the
next optimizer step.
"""

from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard

from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.lora.modules.linear import LoraLinear
from xorl.server.runner.adapters.gradient_ownership import GradientScaleState, TopologyFamily
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.server, pytest.mark.gpu]


def _session_spec(rank: int) -> dict:
    return {
        "base_model": "analytical-heterogeneous-adapter-fixture",
        "is_lora": True,
        "lora_config": {"lora_rank": rank, "lora_alpha": rank, "seed": 29},
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


def _logical_slice(value: torch.Tensor, layout) -> torch.Tensor:
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(layout.active_global_offset, layout.active_storage_shape, strict=True)
    )
    return value[slices].contiguous()


def _gather_logical_slots(state) -> dict[str, torch.Tensor]:
    payload = {
        name: {
            "shape": layout.logical_shape,
            "offset": layout.active_global_offset,
            "storage_shape": layout.active_storage_shape,
            "value": parameter.detach().cpu().contiguous(),
        }
        for name, parameter in state.local_params.items()
        for layout in (state.tensor_layouts[name],)
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    result = {name: torch.zeros(item["shape"], dtype=item["value"].dtype) for name, item in payload.items()}
    coverage = {name: torch.zeros_like(value, dtype=torch.int64) for name, value in result.items()}
    for rank_payload in gathered:
        for name, item in rank_payload.items():
            if item["value"].numel() == 0:
                continue
            slices = tuple(
                slice(offset, offset + size) for offset, size in zip(item["offset"], item["storage_shape"], strict=True)
            )
            result[name][slices].copy_(item["value"])
            coverage[name][slices].add_(1)
    assert all(torch.all(mask == 1) for mask in coverage.values())
    return result


def _reference_gradients(
    base_weight: torch.Tensor,
    parameters: dict[str, nn.Parameter],
    inputs: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    lora_a = next(parameter for name, parameter in parameters.items() if name.endswith("lora_A"))
    lora_b = next(parameter for name, parameter in parameters.items() if name.endswith("lora_B"))
    effective = base_weight + lora_b @ lora_a
    objective = sum(torch.nn.functional.linear(value, effective).square().sum() for value in inputs)
    gradients = torch.autograd.grad(objective, tuple(parameters.values()))
    return dict(zip(parameters, gradients, strict=True))


def _reference_step(
    optimizer: torch.optim.Optimizer,
    parameters: dict[str, nn.Parameter],
    gradients: dict[str, torch.Tensor],
    *,
    clip: float,
) -> float:
    norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in gradients.values())))
    coefficient = min(1.0, clip / (norm + 1e-6))
    for name, parameter in parameters.items():
        parameter.grad = gradients[name].mul(coefficient)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def _assert_local_state_matches_reference(state, parameters, optimizer) -> None:
    for name, local_parameter in state.local_params.items():
        layout = state.tensor_layouts[name]
        reference_parameter = parameters[name]
        torch.testing.assert_close(
            local_parameter.float(),
            _logical_slice(reference_parameter.detach(), layout).float(),
            rtol=5e-4,
            atol=2e-4,
            msg=name,
        )
        local_optimizer_state = state.optimizer.state.get(local_parameter, {})
        if local_parameter.numel() == 0:
            assert local_parameter.grad is None
            assert local_optimizer_state == {}
            continue
        reference_optimizer_state = optimizer.state[reference_parameter]
        for field in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                local_optimizer_state[field].float(),
                _logical_slice(reference_optimizer_state[field], layout).float(),
                rtol=5e-4,
                atol=2e-4,
                msg=f"{name}:{field}",
            )
        assert float(local_optimizer_state["step"].item()) == float(reference_optimizer_state["step"].item())


def _run_worker() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl",
        device_id=device,
        timeout=datetime.timedelta(seconds=20),
    )
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    try:
        init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cuda")

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapter = LoraLinear(4, 4, r=4, lora_alpha=4, device=device, dtype=torch.float32)

        model = _Model()
        with torch.no_grad():
            model.adapter.weight.copy_(
                torch.linspace(-0.2, 0.3, model.adapter.weight.numel(), device=device).reshape_as(model.adapter.weight)
            )
        model.adapter.weight.requires_grad_(False)
        base_weight = model.adapter.weight.detach().clone()
        lora_a_id = id(model.adapter.lora_A)
        lora_b_id = id(model.adapter.lora_B)

        def _shard_placement(parameter: nn.Parameter) -> Shard:
            if id(parameter) == lora_a_id:
                return Shard(0)
            if id(parameter) == lora_b_id:
                return Shard(1)
            return Shard(0)

        fully_shard(
            model.adapter,
            mesh=get_parallel_state().fsdp_mesh,
            shard_placement_fn=_shard_placement,
        )
        model.adapter.set_gradient_divide_factor(1.0)

        root = Path(tempfile.gettempdir()) / f"adapter-gradient-heterogeneous-{os.environ['MASTER_PORT']}"
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
        dist.barrier()
        manager = LoRAAdapterManager(
            model,
            device=device,
            checkpoint_dir=str(root),
            auto_save_on_eviction=False,
            optimizer_fused=False,
            optimizer_dtype="fp32",
            weight_decay=0.0,
        )
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner._adapter_manager = manager

        session_ranks = {"short": 1, "wide": 3}
        references = {}
        scratch_pointers = {}
        for model_id, active_rank in session_ranks.items():
            manager.register_adapter(model_id, session_spec=_session_spec(active_rank), initialize_fresh=True)
            runner._compile_registered_adapter_gradient_ownership(model_id)
            state = manager.get_adapter_state(model_id)
            assert {item.topology for item in state.gradient_ownership_plan.parameters} == {
                TopologyFamily.DENSE_REPLICATED
            }
            logical = _gather_logical_slots(state)
            reference_parameters = {name: nn.Parameter(value.to(device)) for name, value in logical.items()}
            references[model_id] = (
                reference_parameters,
                torch.optim.AdamW(
                    reference_parameters.values(),
                    lr=1e-2,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.0,
                ),
            )
            scratch_pointers[model_id] = {
                name: tensor.data_ptr() for name, tensor in state.gradient_scratch.numerators.items()
            }

        short_state = manager.get_adapter_state("short")
        local_short_elements = sum(parameter.numel() for parameter in short_state.local_params.values())
        empty_flag = torch.tensor(int(local_short_elements == 0), device=device)
        dist.all_reduce(empty_flag, op=dist.ReduceOp.SUM)
        assert int(empty_flag.item()) == 1
        assert all(parameter.numel() == 0 for parameter in short_state.local_params.values()) == (rank == 1)

        for round_index in range(2):
            for session_index, model_id in enumerate(("short", "wide")):
                state = manager.get_adapter_state(model_id)
                other_id = "wide" if model_id == "short" else "short"
                other_state = manager.get_adapter_state(other_id)
                other_before = {name: value.detach().clone() for name, value in other_state.local_params.items()}
                reference_parameters, reference_optimizer = references[model_id]
                manager.prepare_forward(model_id)

                local_input = (
                    torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4) / 20.0
                    + rank * 0.11
                    + round_index * 0.07
                    + session_index * 0.03
                )
                inputs = [torch.empty_like(local_input) for _ in range(2)]
                dist.all_gather(inputs, local_input)
                raw_reference = _reference_gradients(base_weight, reference_parameters, inputs)
                denominator = float(sum(value.shape[0] for value in inputs))
                normalized_reference = {name: gradient / denominator for name, gradient in raw_reference.items()}

                def _capture(_micro_batches, **_kwargs):
                    assert manager.begin_gradient_capture(
                        model_id,
                        scale_state=GradientScaleState.RAW_NUMERATOR,
                    )
                    model.adapter(local_input).square().sum().backward()
                    manager.stage_gradient_numerators(
                        model_id,
                        denominator=denominator,
                        backward_completed=True,
                    )
                    return {"loss": 0.0}

                runner._forward_backward_impl = _capture
                runner.forward_backward([], model_id=model_id)
                for name, staged in state.gradient_scratch.staged_numerators.items():
                    torch.testing.assert_close(
                        staged,
                        _logical_slice(raw_reference[name], state.tensor_layouts[name]).float(),
                        rtol=5e-4,
                        atol=2e-4,
                    )
                runner.commit_forward_backward_completion(model_id)
                expected_norm = _reference_step(
                    reference_optimizer,
                    reference_parameters,
                    normalized_reference,
                    clip=0.05,
                )
                actual_norm = manager.optim_step(model_id, lr=1e-2, gradient_clip=0.05)
                assert actual_norm == pytest.approx(expected_norm, rel=5e-4, abs=2e-4)
                dist.barrier()
                manager.commit_optimizer_publication(model_id)
                _assert_local_state_matches_reference(state, reference_parameters, reference_optimizer)
                assert state.global_step == round_index + 1
                assert state.global_forward_backward_step == round_index + 1
                assert state.publication_eligible and not state.publication_pending
                assert all(parameter.grad is None for parameter in state.local_params.values())
                assert all(
                    parameter.grad is None
                    for name, parameter in model.named_parameters()
                    if name in state.tensor_layouts
                )
                assert {
                    name: tensor.data_ptr() for name, tensor in state.gradient_scratch.numerators.items()
                } == scratch_pointers[model_id]
                for name, value in other_state.local_params.items():
                    torch.testing.assert_close(value, other_before[name])

        if rank == 0:
            print("HETEROGENEOUS_EMPTY_RECTANGLE_TWO_STEP_REAL_FSDP_CERTIFIED", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_heterogeneous_sessions_empty_rectangle_two_step_real_fsdp() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=90,
        extra_env={"XORL_HETEROGENEOUS_ADAPTER_WORKER": "1"},
    )
    result.assert_success("heterogeneous empty-rectangle real-FSDP lifecycle certification")
    assert "HETEROGENEOUS_EMPTY_RECTANGLE_TWO_STEP_REAL_FSDP_CERTIFIED" in result.stdout


if os.environ.get("XORL_HETEROGENEOUS_ADAPTER_WORKER") == "1":
    _run_worker()
