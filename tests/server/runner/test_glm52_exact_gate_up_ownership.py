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


def _compile_dense_plan(name, component, tmp_path, monkeypatch):
    model = nn.Module()
    setattr(model, name, component)
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
    return {item.fqn: item for item in plan.parameters}


def test_exact_dense_and_routed_adapter_gradient_ownership_policy(tmp_path, monkeypatch) -> None:
    with monkeypatch.context() as dense_patch:
        _assert_exact_dense_components_compile_canonical_module_managed_factors(tmp_path, dense_patch)
    with monkeypatch.context() as routed_patch:
        _assert_exact_routed_ownership_guard_rejects_invalid_runtime_ownership(tmp_path, routed_patch)


def _assert_exact_dense_components_compile_canonical_module_managed_factors(tmp_path, monkeypatch) -> None:
    dense_mlp = Glm52ExactTP1DenseMLP(8, 128, device=torch.device("cpu"))
    cases = (
        (
            "gate_up",
            Glm52ExactTP1FusedGateUpBlockFP8QLoRA(8, 128, device=torch.device("cpu")),
            {
                "gate_up.gate_proj.lora_A",
                "gate_up.gate_proj.lora_B",
                "gate_up.up_proj.lora_A",
                "gate_up.up_proj.lora_B",
            },
        ),
        ("mlp", dense_mlp, {f"mlp.{name}" for name in dense_mlp.logical_factor_names}),
        (
            "kv_b_proj",
            Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(device=torch.device("cpu")),
            {"kv_b_proj.lora_A", "kv_b_proj.lora_B"},
        ),
    )

    for name, component, expected_names in cases:
        by_name = _compile_dense_plan(name, component, tmp_path, monkeypatch)
        assert set(by_name) == expected_names
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
        lora_config={"moe_hybrid_shared_lora": True},
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


def _assert_exact_routed_ownership_guard_rejects_invalid_runtime_ownership(tmp_path, monkeypatch) -> None:
    runner, _manager, model = _exact_routed_runner(tmp_path, monkeypatch)

    for ep_dispatch in ("alltoall", "deepep"):
        model.experts.ep_dispatch = ep_dispatch
        with pytest.raises(AdapterGradientOwnershipError, match="requires managed FSDP ownership"):
            runner._compile_registered_adapter_gradient_ownership("policy")
