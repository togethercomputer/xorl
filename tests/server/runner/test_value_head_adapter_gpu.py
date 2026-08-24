"""Real-FSDP certification of the scalar LoRA value head (critic) lifecycle.

Validates, against analytically reconstructed references on 2 GPUs:

- ownership compile: ``value_head.*`` factors classify as
  DIRECT_OUTPUT_PROJECTION with AUTHORIZED_ZERO presence; trunk factors stay
  DENSE_REPLICATED / REQUIRED_IF_ACTIVE;
- a ``value_loss``-style step (loss consumes the folded value-head weight the
  way the server does) produces correct staged numerators and a correct Adam
  step for BOTH trunk and value-head factors;
- a policy-style step on the SAME session (backward never touches the value
  head) is accepted (AUTHORIZED_ZERO), transports exact-zero value-head
  gradients (no stale-numerator leakage from the previous epoch), and matches
  a zero-grad reference Adam step;
- a forward-only (no-grad) value_prediction pass between training steps
  leaves the adapter layout contract intact (the value head's own FSDP unit
  never unshards, so its factors stay sharded DTensors at all times);
- a second critic session with a smaller active rank slices the value head
  correctly and stays isolated from the first session.
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

from xorl.data.constants import IGNORE_INDEX
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.lora.modules.linear import LoraLinear
from xorl.ops.loss import TokenPartial, value_loss_function, value_prediction_function
from xorl.server.runner.adapters.gradient_ownership import (
    GradientPresencePolicy,
    GradientScaleState,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.server, pytest.mark.gpu]

_HIDDEN = 4


def _session_spec(rank: int, frozen_module_patterns: list[str] | None = None) -> dict:
    lora_config = {"lora_rank": rank, "lora_alpha": rank, "seed": 37}
    if frozen_module_patterns:
        lora_config["frozen_module_patterns"] = frozen_module_patterns
    return {
        "base_model": "value-head-certification-fixture",
        "is_lora": True,
        "lora_config": lora_config,
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


def _critic_objective(
    reference: dict[str, nn.Parameter],
    trunk_base: torch.Tensor,
    inputs: list[torch.Tensor],
    labels: list[torch.Tensor],
    returns: list[torch.Tensor],
    *,
    with_value_head: bool,
) -> torch.Tensor:
    """Unsharded mathematical objective mirroring the worker's loss exactly."""
    eff_trunk = trunk_base + reference["trunk.lora_B"] @ reference["trunk.lora_A"]
    objective = torch.zeros((), device=trunk_base.device)
    for x, y, r in zip(inputs, labels, returns, strict=True):
        h = torch.nn.functional.linear(x, eff_trunk)
        if with_value_head:
            eff_vh = reference["value_head.lora_B"] @ reference["value_head.lora_A"]
            values = torch.nn.functional.linear(h.float(), eff_vh.float()).squeeze(-1)
            valid = (y != IGNORE_INDEX).float()
            objective = objective + (0.5 * ((values - r).square() * valid)).sum()
        else:
            objective = objective + h.square().sum()
    return objective


def _reference_gradients(
    reference: dict[str, nn.Parameter],
    trunk_base: torch.Tensor,
    inputs,
    labels,
    returns,
    *,
    with_value_head: bool,
) -> dict[str, torch.Tensor]:
    objective = _critic_objective(reference, trunk_base, inputs, labels, returns, with_value_head=with_value_head)
    names = list(reference)
    gradients = torch.autograd.grad(objective, [reference[name] for name in names], allow_unused=True)
    return {
        name: (gradient if gradient is not None else torch.zeros_like(reference[name]))
        for name, gradient in zip(names, gradients, strict=True)
    }


def _reference_step(optimizer, parameters, gradients, *, clip: float) -> float:
    norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in gradients.values())))
    coefficient = min(1.0, clip / (norm + 1e-6))
    for name, parameter in parameters.items():
        parameter.grad = gradients[name].mul(coefficient)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def _assert_local_state_matches_reference(state, parameters) -> None:
    for name, local_parameter in state.local_params.items():
        layout = state.tensor_layouts[name]
        torch.testing.assert_close(
            local_parameter.float(),
            _logical_slice(parameters[name].detach(), layout).float(),
            rtol=5e-4,
            atol=2e-4,
            msg=name,
        )


def _run_worker() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=30))
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2
    try:
        init_parallel_state(dp_size=2, dp_shard_size=2, device_type="cuda")

        class _Model(nn.Module):
            """Trunk adapter + scalar LoRA value head. Like production, the
            value head has its own FSDP unit whose forward never runs: its
            factors stay sharded DTensors and the loss consumes the folded
            delta via full_tensor() (direct DTensor lane)."""

            def __init__(self) -> None:
                super().__init__()
                self.trunk = LoraLinear(_HIDDEN, _HIDDEN, r=4, lora_alpha=4, device=device, dtype=torch.float32)
                self.value_head = LoraLinear(_HIDDEN, 1, r=4, lora_alpha=4, device=device, dtype=torch.float32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.trunk(x)

        model = _Model()
        with torch.no_grad():
            model.trunk.weight.copy_(
                torch.linspace(-0.2, 0.3, model.trunk.weight.numel(), device=device).reshape_as(model.trunk.weight)
            )
            model.value_head.weight.zero_()
        model.trunk.weight.requires_grad_(False)
        model.value_head.weight.requires_grad_(False)
        trunk_base = model.trunk.weight.detach().clone()

        lora_b_ids = {id(model.trunk.lora_B), id(model.value_head.lora_B)}

        def _shard_placement(parameter: nn.Parameter) -> Shard:
            return Shard(1) if id(parameter) in lora_b_ids else Shard(0)

        mesh = get_parallel_state().fsdp_mesh
        fully_shard(model.trunk, mesh=mesh, shard_placement_fn=_shard_placement)
        model.trunk.set_gradient_divide_factor(1.0)
        # Production gives the value head its own unit whose forward never
        # runs, so its factors stay sharded DTensors at all times.
        fully_shard(model.value_head, mesh=mesh, shard_placement_fn=_shard_placement)
        model.value_head.set_gradient_divide_factor(1.0)
        fully_shard(model, mesh=mesh, shard_placement_fn=_shard_placement)
        model.set_gradient_divide_factor(1.0)

        root = Path(tempfile.gettempdir()) / f"value-head-adapter-{os.environ['MASTER_PORT']}"
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
        assert set(manager._lora_param_names) == {
            "trunk.lora_A",
            "trunk.lora_B",
            "value_head.lora_A",
            "value_head.lora_B",
        }
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner._adapter_manager = manager

        session_ranks = {"critic": 4, "critic_r2": 2, "critic_frozen": 4}
        session_frozen = {"critic_frozen": ["trunk"]}
        references = {}
        for model_id, active_rank in session_ranks.items():
            manager.register_adapter(
                model_id,
                session_spec=_session_spec(active_rank, session_frozen.get(model_id)),
                initialize_fresh=True,
            )
            runner._compile_registered_adapter_gradient_ownership(model_id)
            state = manager.get_adapter_state(model_id)

            by_name = {item.fqn: item for item in state.gradient_ownership_plan.parameters}
            assert by_name["trunk.lora_A"].topology is TopologyFamily.DENSE_REPLICATED
            assert by_name["value_head.lora_A"].topology is TopologyFamily.DIRECT_OUTPUT_PROJECTION
            assert by_name["value_head.lora_B"].topology is TopologyFamily.DIRECT_OUTPUT_PROJECTION
            assert by_name["value_head.lora_A"].presence is GradientPresencePolicy.AUTHORIZED_ZERO
            assert by_name["value_head.lora_B"].presence is GradientPresencePolicy.AUTHORIZED_ZERO
            trunk_presence = (
                GradientPresencePolicy.AUTHORIZED_ZERO
                if model_id in session_frozen
                else GradientPresencePolicy.REQUIRED_IF_ACTIVE
            )
            assert by_name["trunk.lora_A"].presence is trunk_presence
            assert by_name["trunk.lora_B"].presence is trunk_presence

            logical = _gather_logical_slots(state)
            reference_parameters = {
                name: nn.Parameter(value.to(device), requires_grad=True) for name, value in logical.items()
            }
            references[model_id] = (
                reference_parameters,
                torch.optim.AdamW(
                    reference_parameters.values(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
                ),
            )

        def _run_step(model_id: str, step_tag: float, *, with_value_head: bool) -> None:
            state = manager.get_adapter_state(model_id)
            reference_parameters, reference_optimizer = references[model_id]
            other_ids = [other for other in session_ranks if other != model_id]
            others_before = {
                other: {
                    name: value.detach().clone()
                    for name, value in manager.get_adapter_state(other).local_params.items()
                }
                for other in other_ids
            }
            manager.prepare_forward(model_id)
            assert model.value_head.active_r == session_ranks[model_id]

            local_input = (
                torch.arange(2 * _HIDDEN, device=device, dtype=torch.float32).reshape(2, _HIDDEN) / 17.0
                + rank * 0.13
                + step_tag
            )
            local_labels = torch.tensor([[IGNORE_INDEX, 5]] if rank == 0 else [[7, 9]], device=device)
            local_returns = torch.tensor([[0.3, -0.4]], device=device, dtype=torch.float32) + rank * 0.21 + step_tag

            all_inputs = [torch.empty_like(local_input) for _ in range(world)]
            dist.all_gather(all_inputs, local_input)
            all_labels = [torch.empty_like(local_labels) for _ in range(world)]
            dist.all_gather(all_labels, local_labels)
            all_returns = [torch.empty_like(local_returns) for _ in range(world)]
            dist.all_gather(all_returns, local_returns)

            if with_value_head:
                denominator = float(sum(int((y != IGNORE_INDEX).sum().item()) for y in all_labels))
            else:
                denominator = float(sum(x.shape[0] for x in all_inputs))

            raw_reference = _reference_gradients(
                reference_parameters,
                trunk_base,
                all_inputs,
                [y.squeeze(0) for y in all_labels],
                [r.squeeze(0) for r in all_returns],
                with_value_head=with_value_head,
            )
            frozen_prefixes = tuple(session_frozen.get(model_id, ()))
            for name in raw_reference:
                if any(name.startswith(prefix) for prefix in frozen_prefixes):
                    raw_reference[name] = torch.zeros_like(raw_reference[name])
            normalized_reference = {name: gradient / denominator for name, gradient in raw_reference.items()}

            def _capture(_micro_batches, **_kwargs):
                assert manager.begin_gradient_capture(model_id, scale_state=GradientScaleState.RAW_NUMERATOR)
                hidden = model(local_input)
                if with_value_head:
                    effective_weight = runner._get_effective_value_head_weight()
                    unit = TokenPartial(scale=torch.tensor(1.0, device=device))
                    output = value_loss_function(
                        hidden_states=hidden.view(1, 2, _HIDDEN),
                        weight=effective_weight,
                        labels=local_labels,
                        returns=local_returns,
                        loss_reducer=unit,
                        metric_reducer=unit,
                    )
                    output.loss.backward()
                else:
                    hidden.square().sum().backward()
                manager.stage_gradient_numerators(model_id, denominator=denominator, backward_completed=True)
                return {"loss": 0.0}

            runner._forward_backward_impl = _capture
            runner.forward_backward([], model_id=model_id)
            scratch = manager.get_adapter_state(model_id).gradient_scratch
            for fqn, staged_numerator in scratch.staged_numerators.items():
                layout = state.tensor_layouts[fqn]
                if not layout.has_active_storage:
                    continue
                torch.testing.assert_close(
                    staged_numerator,
                    _logical_slice(raw_reference[fqn], layout).float().reshape(staged_numerator.shape),
                    rtol=5e-4,
                    atol=2e-4,
                    msg=f"staged numerator mismatch: {fqn}",
                )
            staged = set(scratch.staged_parameter_fqns)
            if with_value_head:
                assert {"value_head.lora_A", "value_head.lora_B"} <= staged
            else:
                assert not any(name.startswith("value_head.") for name in staged)
            for prefix in session_frozen.get(model_id, ()):
                assert not any(name.startswith(prefix) for name in staged), staged
            runner.commit_forward_backward_completion(model_id)

            expected_norm = _reference_step(reference_optimizer, reference_parameters, normalized_reference, clip=0.05)
            actual_norm = manager.optim_step(model_id, lr=1e-2, gradient_clip=0.05)
            assert actual_norm == pytest.approx(expected_norm, rel=5e-4, abs=2e-4), (
                f"{model_id} step_tag={step_tag} with_value_head={with_value_head} "
                f"actual={actual_norm} expected={expected_norm}"
            )
            dist.barrier()
            manager.commit_optimizer_publication(model_id)
            _assert_local_state_matches_reference(state, reference_parameters)

            for other, before in others_before.items():
                for name, value in manager.get_adapter_state(other).local_params.items():
                    torch.testing.assert_close(value, before[name], msg=f"{other}:{name}")

        def _run_forward_only(model_id: str) -> None:
            """No-grad value_prediction pass (regression: a forward-only op
            must not disturb the adapter layout contract for later steps)."""
            manager.prepare_forward(model_id)
            with torch.no_grad():
                hidden = model(torch.ones(2, _HIDDEN, device=device))
                effective_weight = runner._get_effective_value_head_weight()
                output = value_prediction_function(
                    hidden_states=hidden.view(1, 2, _HIDDEN),
                    weight=effective_weight,
                    labels=torch.tensor([[5, 9]], device=device),
                )
            assert output.per_token_logprobs.shape == (1, 2)

        # Step 1: value_loss on the full-rank critic (trunk + value-head grads).
        _run_step("critic", 0.00, with_value_head=True)
        # Forward-only op between training steps (this is what a client's
        # value_prediction forward does) — the next step must still validate.
        _run_forward_only("critic")
        # Step 2: policy-style step on the SAME session — value head untouched.
        # AUTHORIZED_ZERO must accept the absent gradients and transport exact
        # zeros (no stale numerators from step 1).
        _run_step("critic", 0.05, with_value_head=False)
        # Step 3: value_loss again on the same session (post-hygiene epoch).
        _run_step("critic", 0.10, with_value_head=True)
        # Step 4: the rank-2 critic slices the value-head substrate and stays
        # isolated from the rank-4 session.
        _run_step("critic_r2", 0.15, with_value_head=True)
        # Step 5: frozen-trunk critic (SAO frozen-attention analogue): trunk
        # factor gradients are skipped at staging, its optimizer never moves
        # them, and only the value head trains.
        frozen_state = manager.get_adapter_state("critic_frozen")
        frozen_trunk_before = {
            name: value.detach().clone()
            for name, value in frozen_state.local_params.items()
            if name.startswith("trunk")
        }
        _run_step("critic_frozen", 0.20, with_value_head=True)
        _run_step("critic_frozen", 0.25, with_value_head=True)
        for name, before in frozen_trunk_before.items():
            torch.testing.assert_close(frozen_state.local_params[name], before, msg=f"frozen factor moved: {name}")

        if rank == 0:
            print("VALUE_HEAD_ADAPTER_REAL_FSDP_CERTIFIED", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_value_head_adapter_lifecycle_real_fsdp() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=180,
        extra_env={"XORL_VALUE_HEAD_ADAPTER_WORKER": "1"},
    )
    result.assert_success("value-head adapter real-FSDP lifecycle certification")
    assert "VALUE_HEAD_ADAPTER_REAL_FSDP_CERTIFIED" in result.stdout


if os.environ.get("XORL_VALUE_HEAD_ADAPTER_WORKER") == "1":
    _run_worker()
