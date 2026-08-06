"""LM-head LoRA folding on the server loss path."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.lora.fold import canonical_lora_fold_linear
from xorl.lora.modules.linear import LoraLinear
from xorl.models.layers.moe.lora import MoEExpertsLoRA, MoELoRAConfig
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts
from xorl.server.runner import model_runner as model_runner_module
from xorl.server.runner.adapters.gradient_ownership import (
    AdapterGradientOwnershipError,
    GradientScaleState,
    ProducerFamily,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner


pytestmark = pytest.mark.cpu


def test_effective_lm_head_uses_canonical_merged_weight_with_adapter_gradients():
    head = LoraLinear(5, 7, r=2, lora_alpha=4, bias=False, dtype=torch.bfloat16)
    head.exact_merged_forward = True
    with torch.no_grad():
        head.weight.copy_(torch.linspace(-1.0, 1.0, head.weight.numel()).reshape_as(head.weight))
        head.lora_A.copy_(torch.linspace(-0.25, 0.25, head.lora_A.numel()).reshape_as(head.lora_A))
        head.lora_B.copy_(torch.linspace(0.125, -0.125, head.lora_B.numel()).reshape_as(head.lora_B))

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(lm_head=head)
    effective = runner._get_effective_lm_head_weight()
    expected = canonical_lora_fold_linear(
        head.weight,
        head.lora_A,
        head.lora_B,
        head._active_scaling(),
    )

    assert torch.equal(effective.detach().view(torch.int16), expected.view(torch.int16))
    effective.float().sum().backward()
    assert head.weight.grad is None
    assert head.lora_A.grad is not None and torch.count_nonzero(head.lora_A.grad) > 0
    assert head.lora_B.grad is not None and torch.count_nonzero(head.lora_B.grad) > 0


def test_effective_lm_head_keeps_legacy_unmerged_formula_without_contract():
    head = LoraLinear(3, 4, r=2, lora_alpha=2, bias=False, dtype=torch.bfloat16)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(lm_head=head)

    expected = head.weight + head.get_delta_weight().to(head.weight.dtype)
    assert torch.equal(runner._get_effective_lm_head_weight(), expected)


def test_runner_compiles_module_and_direct_output_properties_at_registration(tmp_path, monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = LoraLinear(3, 3, r=2, lora_alpha=2)
            self.lm_head = LoraLinear(3, 4, r=2, lora_alpha=2)

    model = _Model()
    model.lm_head.exact_merged_forward = True
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    parallel_state = SimpleNamespace(
        sp_grad_sync_group=None,
        lm_head_tp_replica_group=None,
        ep_size=1,
    )
    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: parallel_state)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    runner._compile_registered_adapter_gradient_ownership("policy")

    plan = manager.get_adapter_state("policy").gradient_ownership_plan
    assert plan is not None
    by_name = {item.fqn: item for item in plan.parameters}
    assert by_name["trunk.lora_A"].producer is ProducerFamily.MODULE_MANAGED
    assert by_name["trunk.lora_A"].topology is TopologyFamily.DENSE_REPLICATED
    assert by_name["lm_head.lora_B"].producer is ProducerFamily.DIRECT_OUTPUT_PROJECTION
    assert by_name["lm_head.lora_B"].topology is TopologyFamily.DIRECT_OUTPUT_PROJECTION
    assert dict(by_name["trunk.lora_A"].config_guard_fields)["merged_forward"] is False
    assert dict(by_name["lm_head.lora_B"].config_guard_fields)["merged_forward"] is True


@pytest.mark.parametrize(
    ("sp_size", "output_size", "replica_count"),
    ((2, 1, 2), (1, 4, 4), (2, 4, 8)),
)
def test_runner_compiler_uses_world_discovered_replica_count_once(
    tmp_path,
    monkeypatch,
    sp_size,
    output_size,
    replica_count,
):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = LoraLinear(3, 3, r=2, lora_alpha=2)
            self.lm_head = LoraLinear(3, 4, r=2, lora_alpha=2)

    model = _Model()
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    state = manager.get_adapter_state("policy")
    trunk_replica_count = sp_size
    state.tensor_layouts = {
        name: replace(
            layout,
            replica_count=replica_count if name.startswith("lm_head.") else trunk_replica_count,
            replica_ranks=tuple(range(replica_count if name.startswith("lm_head.") else trunk_replica_count)),
        )
        for name, layout in state.tensor_layouts.items()
    }
    sp_group = object() if sp_size > 1 else None
    output_group = object() if output_size > 1 else None
    parallel_state = SimpleNamespace(
        sp_grad_sync_group=sp_group,
        lm_head_tp_replica_group=output_group,
        ep_group=None,
        ep_size=1,
        tp_size=1,
    )
    sp_families = tuple(tuple(range(start, start + sp_size)) for start in range(0, replica_count, sp_size))
    output_families = tuple(tuple(range(row, replica_count, sp_size)) for row in range(sp_size))

    def _group_members(group):
        if group is sp_group:
            return sp_families[0]
        if group is output_group:
            return output_families[0]
        raise AssertionError("unexpected process group")

    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(model_runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(model_runner_module.dist, "get_process_group_ranks", _group_members)
    monkeypatch.setattr(
        model_runner_module.dist,
        "get_world_size",
        lambda group=None: sp_size if group is sp_group else output_size if group is output_group else replica_count,
    )
    monkeypatch.setattr(manager, "_agree_gradient_ownership_fingerprint", lambda _plan: None)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    group_memberships = {}
    if sp_group is not None:
        group_memberships["sequence_parallel"] = sp_families
    if output_group is not None:
        group_memberships["output_projection_replica"] = output_families
    runner._compile_registered_adapter_gradient_ownership(
        "policy",
        group_memberships=group_memberships,
    )

    plan = state.gradient_ownership_plan
    assert plan is not None
    by_name = {item.fqn: item for item in plan.parameters}
    assert by_name["trunk.lora_A"].norm_replica_divisor == trunk_replica_count
    assert by_name["lm_head.lora_A"].norm_replica_divisor == replica_count


@pytest.mark.parametrize(
    ("case", "replica_count", "sp_families", "output_families", "error_match"),
    (
        ("unowned", 2, (), (), "exactly cover"),
        ("incomplete", 4, (), ((0, 1), (2, 3)), "exactly cover"),
        ("overlap", 4, ((0, 1), (2, 3)), ((0, 1), (2, 3)), "overlap beyond"),
        ("outside", 4, (), ((0, 1, 4), (2, 3)), "outside the replica class"),
        ("missing", 2, (), ((0, 1),), "identity .* is missing"),
    ),
)
def test_production_runner_compiler_rejects_invalid_replica_coverage(
    tmp_path,
    monkeypatch,
    case,
    replica_count,
    sp_families,
    output_families,
    error_match,
):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = LoraLinear(3, 4, r=2, lora_alpha=2)

    model = _Model()
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    state = manager.get_adapter_state("policy")
    state.tensor_layouts = {
        name: replace(
            layout,
            replica_count=replica_count,
            replica_ranks=tuple(range(replica_count)),
        )
        for name, layout in state.tensor_layouts.items()
    }
    sp_group = object() if sp_families else None
    output_group = object() if output_families else None
    parallel_state = SimpleNamespace(
        sp_grad_sync_group=sp_group,
        lm_head_tp_replica_group=output_group,
        ep_group=None,
        ep_size=1,
        tp_size=1,
    )

    def _members(group):
        families = sp_families if group is sp_group else output_families
        return next(members for members in families if 0 in members)

    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(model_runner_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(model_runner_module.dist, "get_world_size", lambda group=None: replica_count)
    monkeypatch.setattr(model_runner_module.dist, "get_process_group_ranks", _members)
    monkeypatch.setattr(manager, "_agree_gradient_ownership_fingerprint", lambda _plan: None)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    group_memberships = {}
    if case != "missing":
        if sp_families:
            group_memberships["sequence_parallel"] = sp_families
        if output_families:
            group_memberships["output_projection_replica"] = output_families
    with pytest.raises(AdapterGradientOwnershipError, match=error_match):
        runner._compile_registered_adapter_gradient_ownership(
            "policy",
            group_memberships=group_memberships,
        )


def test_production_runner_compiler_rejects_general_model_tensor_parallelism(tmp_path, monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = LoraLinear(3, 3, r=2, lora_alpha=2)

    model = _Model()
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            sp_grad_sync_group=None,
            lm_head_tp_replica_group=None,
            ep_group=None,
            ep_size=1,
            tp_size=2,
        ),
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    with pytest.raises(AdapterGradientOwnershipError, match="tensor parallelism greater than one"):
        runner._compile_registered_adapter_gradient_ownership("policy")


def test_forward_backward_aborts_a_staged_capture_when_the_call_fails(tmp_path):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = LoraLinear(3, 3, r=2, lora_alpha=2)
            self.lm_head = LoraLinear(3, 4, r=2, lora_alpha=2)

    model = _Model()
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    state = manager.get_adapter_state("policy")
    declarations = {
        name: model_runner_module.ParameterOwnershipDeclaration(
            topology=(
                TopologyFamily.DIRECT_OUTPUT_PROJECTION
                if name.startswith("lm_head.")
                else TopologyFamily.DENSE_REPLICATED
            ),
            producer=(
                ProducerFamily.DIRECT_OUTPUT_PROJECTION
                if name.startswith("lm_head.")
                else ProducerFamily.MODULE_MANAGED
            ),
            representation=model_runner_module.GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(),
            config_guard_fingerprint=f"fixed:{name}",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy",
        declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    def _fail_after_stage(_micro_batches, **_kwargs):
        assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
        model.trunk(torch.ones(1, 3)).sum().backward()
        runner._get_effective_lm_head_weight().sum().backward()
        manager.stage_gradient_numerators("policy", denominator=3, backward_completed=True)
        raise RuntimeError("injected post-stage failure")

    runner._forward_backward_impl = _fail_after_stage

    with pytest.raises(RuntimeError, match="post-stage failure"):
        runner.forward_backward([], model_id="policy")

    assert not manager.gradient_capture_is_open("policy")
    assert state.gradient_scratch.next_capture_ordinal == 0
    assert state.gradient_scratch.denominator == 0
    assert state.gradient_scratch.numerators
    assert all(not torch.count_nonzero(tensor) for tensor in state.gradient_scratch.numerators.values())
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name in state.tensor_layouts)


@pytest.mark.parametrize(
    ("implementation", "exact_merged_forward", "expected_producer"),
    (
        ("eager", False, ProducerFamily.MODULE_MANAGED),
        ("eager", True, ProducerFamily.FUSED_MANAGED),
        ("triton", False, ProducerFamily.FUSED_MANAGED),
    ),
)
def test_runner_compiles_certified_unquantized_expert_backends(
    tmp_path, monkeypatch, implementation, exact_merged_forward, expected_producer
):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = MoEExpertsLoRA(
                num_experts=2,
                hidden_dim=8,
                intermediate_size=8,
                moe_implementation=implementation,
                lora_config=MoELoRAConfig(r=2, lora_alpha=2, hybrid_shared=True),
            )

    model = _Model()
    model.experts.exact_merged_forward = exact_merged_forward
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            sp_grad_sync_group=None,
            lm_head_tp_replica_group=None,
            ep_group=None,
            ep_size=1,
            tp_size=1,
        ),
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    runner._compile_registered_adapter_gradient_ownership("policy")

    plan = manager.get_adapter_state("policy").gradient_ownership_plan
    assert plan is not None
    assert len(plan.parameters) == 6
    assert {item.producer for item in plan.parameters} == {expected_producer}
    assert {item.topology for item in plan.parameters} == {TopologyFamily.DENSE_REPLICATED}


@pytest.mark.parametrize("implementation", ("native", "quack"))
def test_runner_rejects_uncertified_unquantized_expert_backends(tmp_path, monkeypatch, implementation):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = MoEExpertsLoRA(
                num_experts=2,
                hidden_dim=8,
                intermediate_size=8,
                moe_implementation=implementation,
                lora_config=MoELoRAConfig(r=2, lora_alpha=2, hybrid_shared=True),
            )

    model = _Model()
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            sp_grad_sync_group=None,
            lm_head_tp_replica_group=None,
            ep_group=None,
            ep_size=1,
            tp_size=1,
        ),
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    with pytest.raises(AdapterGradientOwnershipError, match="certified eager/triton expert backends"):
        runner._compile_registered_adapter_gradient_ownership("policy")


@pytest.mark.parametrize("implementation", ("eager", "triton", "native", "quack"))
def test_runner_rejects_uncertified_quantized_expert_factors(tmp_path, monkeypatch, implementation):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = BlockFP8QLoRAMoeExperts(
                num_local_experts=2,
                num_experts=2,
                intermediate_size=8,
                hidden_size=8,
                r=2,
                lora_alpha=2,
                device=torch.device("cpu"),
                moe_implementation=implementation,
            )

    model = _Model()
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy", lr=0.1)
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_grad_sync_group=None, lm_head_tp_replica_group=None, ep_size=1),
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager

    with pytest.raises(AdapterGradientOwnershipError, match="Quantized expert-factor LoRA is not certified"):
        runner._compile_registered_adapter_gradient_ownership("policy")


def test_direct_output_projection_runs_through_authoritative_analytical_step(tmp_path, monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = LoraLinear(3, 3, r=2, lora_alpha=2)
            self.lm_head = LoraLinear(3, 4, r=2, lora_alpha=2)

    model = _Model()
    model.trunk.exact_merged_forward = True
    model.lm_head.exact_merged_forward = True
    manager = LoRAAdapterManager(
        model,
        torch.device("cpu"),
        checkpoint_dir=str(tmp_path),
        auto_save_on_eviction=False,
        weight_decay=0.0,
    )
    manager.register_adapter("policy", lr=0.1)
    parallel_state = SimpleNamespace(
        sp_grad_sync_group=None,
        lm_head_tp_replica_group=None,
        ep_size=1,
    )
    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: parallel_state)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager
    runner._compile_registered_adapter_gradient_ownership("policy")
    manager.prepare_forward("policy")
    state = manager.get_adapter_state("policy")
    initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)

    hidden = model.trunk(torch.tensor([[0.25, -0.5, 0.75]]))
    loss = F.linear(hidden, runner._get_effective_lm_head_weight()).sum()
    loss.backward()
    raw = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if name in state.tensor_layouts
    }
    manager.capture_gradient_numerators(
        "policy",
        denominator=1,
        backward_completed=True,
    )
    expected_norm = float(torch.sqrt(sum(gradient.float().square().sum() for gradient in raw.values())))

    actual_norm = manager.optim_step("policy", 0.1)

    assert actual_norm == pytest.approx(expected_norm)
    plan = state.gradient_ownership_plan
    assert plan is not None
    assert {item.producer for item in plan.parameters} == {
        ProducerFamily.MODULE_MANAGED,
        ProducerFamily.DIRECT_OUTPUT_PROJECTION,
    }
    for name, parameter in state.local_params.items():
        gradient = raw[name].float()
        expected_parameter = initial[name] - 0.1 * gradient / (gradient.abs() + 1e-8)
        torch.testing.assert_close(parameter, expected_parameter)
        torch.testing.assert_close(state.optimizer.state[parameter]["exp_avg"], gradient * 0.1)
        torch.testing.assert_close(state.optimizer.state[parameter]["exp_avg_sq"], gradient.square() * 0.05)
