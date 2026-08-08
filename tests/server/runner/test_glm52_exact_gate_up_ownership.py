"""Adapter-gradient ownership gate for the exact GLM-5.2 gate/up leaf."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.exact_gate_up_qlora import Glm52ExactTP1FusedGateUpBlockFP8QLoRA
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.server.runner import model_runner as model_runner_module
from xorl.server.runner.adapters.gradient_ownership import (
    AdapterGradientOwnershipError,
    ProducerFamily,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_fused_gate_up_logical_children_compile_module_managed_ownership(tmp_path, monkeypatch) -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up = Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, device=torch.device("cpu"))

    model = _Model()
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
        ep_enabled=False,
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
    assert set(by_name) == {
        "gate_up.gate_proj.lora_A",
        "gate_up.gate_proj.lora_B",
        "gate_up.up_proj.lora_A",
        "gate_up.up_proj.lora_B",
    }
    assert all(item.producer is ProducerFamily.MODULE_MANAGED for item in by_name.values())
    assert all(item.topology is TopologyFamily.DENSE_REPLICATED for item in by_name.values())


def test_exact_dense_mlp_compiles_six_canonical_module_managed_factors(tmp_path, monkeypatch) -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = Glm52ExactTP1DenseMLP(8, 128, device=torch.device("cpu"))

    model = _Model()
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
        ep_enabled=False,
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
    assert set(by_name) == {f"mlp.{name}" for name in model.mlp.logical_factor_names}
    assert all(item.producer is ProducerFamily.MODULE_MANAGED for item in by_name.values())
    assert all(item.topology is TopologyFamily.DENSE_REPLICATED for item in by_name.values())


def test_exact_absorbed_kv_b_compiles_shared_q_v_factors_as_module_managed(tmp_path, monkeypatch) -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kv_b_proj = Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(device=torch.device("cpu"))

    model = _Model()
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
        ep_enabled=False,
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
    assert set(by_name) == {"kv_b_proj.lora_A", "kv_b_proj.lora_B"}
    assert all(item.producer is ProducerFamily.MODULE_MANAGED for item in by_name.values())
    assert all(item.topology is TopologyFamily.DENSE_REPLICATED for item in by_name.values())


def _exact_routed_runner(tmp_path, monkeypatch):
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(128, 128, ep_rank=7, device="cpu")

    model = _Model()
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
        ep_enabled=False,
        ep_size=16,
    )
    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: parallel_state)
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager
    return runner, manager, model


def test_exact_routed_ownership_guard_rejects_unwrapped_unparallelized_experts(tmp_path, monkeypatch) -> None:
    runner, _manager, _model = _exact_routed_runner(tmp_path, monkeypatch)

    with pytest.raises(AdapterGradientOwnershipError, match="not managed by its expert FSDP unit"):
        runner._compile_registered_adapter_gradient_ownership("policy")


def test_exact_routed_ownership_guard_rejects_deepep_mutation(tmp_path, monkeypatch) -> None:
    runner, _manager, model = _exact_routed_runner(tmp_path, monkeypatch)
    model.experts.ep_dispatch = "deepep"

    with pytest.raises(AdapterGradientOwnershipError, match="exact EP16 alltoall routed lane"):
        runner._compile_registered_adapter_gradient_ownership("policy")
