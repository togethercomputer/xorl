"""Tests for adapter-manager optimizer integration."""

import asyncio
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from xorl.optim import SignSGD
from xorl.server.protocol.operations import AdapterStateData
from xorl.server.runner.adapters import manager as adapter_manager_module
from xorl.server.runner.adapters.adapter_coordinator import AdapterCoordinator
from xorl.server.runner.adapters.gradient_finalizer import (
    AdapterGradientCollectiveFailure,
    AdapterGradientMutationFailure,
)
from xorl.server.runner.adapters.gradient_ownership import (
    AdapterGradientOwnershipError,
    GradientRepresentation,
    GradientScaleState,
    ParameterOwnershipDeclaration,
    ProducerFamily,
    ReductionAuthority,
    ReductionAxis,
    ReductionDomainPlan,
    ReductionOperation,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.adapters.optimizer_reshard import clone_state_to_cpu as _clone_state_to_cpu
from xorl.server.runner.adapters.optimizer_reshard import same_optimizer_value as _same_optimizer_value
from xorl.server.session_spec import normalize_session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server]


@pytest.fixture(autouse=True)
def _trusted_server_artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(tmp_path))


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


class _DummyExactGlmLoRALayer(_DummyLoRALayer):
    _glm52_exact_active_lora_component = True

    def __init__(self, *, rank: int = 4, alpha: int = 16) -> None:
        super().__init__(max_rank=rank)
        self.r = rank
        self.lora_alpha = alpha


class _DummyExactGlmLoRAModel(_DummyLoRAModel):
    def __init__(self, *, rank: int = 4, alpha: int = 16) -> None:
        super().__init__(max_rank=rank)
        self.model.layers[0].self_attn.o_proj = _DummyExactGlmLoRALayer(rank=rank, alpha=alpha)


class _IntegratedTestAdapterManager(LoRAAdapterManager):
    """Test harness that models ModelRunner's mandatory registration compile."""

    def register_adapter(self, *args, **kwargs) -> None:
        super().register_adapter(*args, **kwargs)
        model_id = args[0] if args else kwargs["model_id"]
        state = self.get_adapter_state(model_id)
        if state.gradient_ownership_plan is not None:
            return
        declarations = {
            name: ParameterOwnershipDeclaration(
                topology=TopologyFamily.DENSE_REPLICATED,
                producer=ProducerFamily.MODULE_MANAGED,
                representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
                completed_domains=(),
                pending_domains=(),
                config_guard_fingerprint="integrated-test-module-path-v1",
            )
            for name in state.tensor_layouts
        }
        self.compile_gradient_ownership_plan(
            model_id,
            declarations,
            model_generation="integrated-test-model-generation-v1",
            adapter_generation="integrated-test-adapter-generation-v1",
        )


def _build_manager(tmp_path: Path, **kwargs) -> LoRAAdapterManager:
    max_rank = kwargs.pop("max_rank", 4)
    return _IntegratedTestAdapterManager(
        _DummyLoRAModel(max_rank=max_rank),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        **kwargs,
    )


class _CoordinatorTrainer:
    def __init__(self, adapter_manager: LoRAAdapterManager) -> None:
        self.adapter_manager = adapter_manager
        self.lora_session_specs = {}

    def register_session(self, model_id: str, session_spec: dict, materialize: bool = False, **kwargs):
        self.lora_session_specs[model_id] = session_spec
        if materialize and not self.adapter_manager.has_adapter(model_id):
            self.adapter_manager.register_adapter(
                model_id=model_id,
                session_spec=session_spec,
                initialize_fresh=kwargs.get("initialize_fresh", True),
            )
        return {"registered": True, "model_id": model_id}

    def get_lora_session_spec(self, model_id: str) -> dict:
        if model_id in self.lora_session_specs:
            return self.lora_session_specs[model_id]
        return self.adapter_manager.get_adapter_session_spec(model_id)

    def register_lora_adapter(self, model_id: str, lr=None):
        session_spec = dict(self.get_lora_session_spec(model_id))
        if lr is not None:
            session_spec.setdefault("optimizer_config", {})["learning_rate"] = lr
        self.adapter_manager.register_adapter(model_id=model_id, session_spec=session_spec, initialize_fresh=True)
        return {"registered": True, "model_id": model_id}

    def load_adapter_state(self, model_id: str, path: str, load_optimizer: bool = True, lr=None):
        return self.adapter_manager.load_adapter_state(
            model_id=model_id,
            path=path,
            load_optimizer=load_optimizer,
            lr=lr,
        )


def _session_spec(*, rank: int, alpha: int, optimizer_type: str, lr: float, weight_decay: float = 0.0) -> dict:
    return {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": True,
        "lora_config": {
            "lora_rank": rank,
            "lora_alpha": alpha,
        },
        "optimizer_config": {
            "type": optimizer_type,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer_dtype": "bf16",
            "betas": None if optimizer_type in {"sgd", "signsgd"} else [0.9, 0.95],
            "eps": None if optimizer_type in {"sgd", "signsgd"} else 1e-8,
            "optimizer_kwargs": {},
        },
    }


def test_register_adapter_uses_shared_optimizer_factory_and_checkpoint_dir(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="signsgd", weight_decay=0.25)

    manager.register_adapter("policy-a", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-a")

    assert isinstance(state.optimizer, SignSGD)
    save_result = manager.save_adapter_state("policy-a")
    save_path = Path(save_result["path"])
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))

    assert save_path == tmp_path / "adapters" / "policy-a"
    assert metadata["optimizer"]["type"] == "signsgd"
    assert metadata["optimizer"]["weight_decay"] == pytest.approx(0.25)
    assert metadata["optimizer"]["betas"] == [0.9, 0.95]
    assert metadata["optimizer"]["eps"] == pytest.approx(1e-8)


def test_fresh_adapter_slots_honor_nonzero_lora_b_initialization(tmp_path):
    from xorl.lora.utils import initialize_lora_b_nonzero

    model = _DummyLoRAModel(max_rank=4)
    initialize_lora_b_nonzero(model, std=1e-3, seed=29)
    initialized_b = model.model.layers[0].self_attn.o_proj.lora_B.detach().clone()
    manager = _IntegratedTestAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        optimizer_type="sgd",
        lora_config={"lora_rank": 4, "lora_alpha": 16, "lora_b_init_std": 1e-3, "lora_b_init_seed": 29},
    )

    manager.register_adapter("default", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("default")
    slot = state.local_params["model.layers.0.self_attn.o_proj.lora_B"]
    assert torch.count_nonzero(slot) == slot.numel()
    assert torch.equal(slot, initialized_b)

    model.model.layers[0].self_attn.o_proj.lora_B.data.zero_()
    manager.prepare_forward("default")
    model_b = manager.model.model.layers[0].self_attn.o_proj.lora_B
    assert torch.equal(model_b, slot)


@pytest.mark.parametrize("model_id", ["default", "policy"])
def test_fresh_adapter_slots_initialize_nonzero_lora_b_after_deferred_materialization(tmp_path, model_id):
    model = _DummyLoRAModel(max_rank=4)
    assert torch.count_nonzero(model.model.layers[0].self_attn.o_proj.lora_B) == 0
    manager = _IntegratedTestAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        optimizer_type="sgd",
        lora_config={
            "lora_rank": 4,
            "lora_alpha": 16,
            "lora_b_init_std": 1e-3,
            "lora_b_init_seed": 1616,
        },
    )

    manager.register_adapter(model_id, lr=0.1, initialize_fresh=True)
    slot = manager.get_adapter_state(model_id).local_params["model.layers.0.self_attn.o_proj.lora_B"]
    assert torch.count_nonzero(slot) == slot.numel()


def test_exact_glm_session_rank_alpha_drift_fails_at_registration(tmp_path):
    manager = LoRAAdapterManager(
        _DummyExactGlmLoRAModel(rank=4, alpha=16),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        optimizer_type="sgd",
    )
    incompatible = _session_spec(rank=2, alpha=8, optimizer_type="sgd", lr=0.1)

    with pytest.raises(ValueError, match="cannot mutate construction-time rank/alpha"):
        manager.register_adapter("policy-glm", session_spec=incompatible, initialize_fresh=True)

    assert not manager.has_adapter("policy-glm")


def test_compile_gradient_ownership_plan_is_explicit_and_rejects_pending_reconfiguration(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-owned", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-owned")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(),
            config_guard_fingerprint="fixed-module-path-v1",
        )
        for name in state.tensor_layouts
    }

    plan = manager.compile_gradient_ownership_plan(
        "policy-owned",
        declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
    )

    assert state.gradient_ownership_plan is plan
    assert {item.fqn for item in plan.parameters} == set(state.tensor_layouts)

    parameter = next(iter(state.local_params.values()))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="gradients are pending"):
        manager.compile_gradient_ownership_plan(
            "policy-owned",
            declarations,
            model_generation="model-generation-1",
            adapter_generation="adapter-generation-2",
        )


def test_raw_numerator_capture_preserves_model_grads_and_streams_into_one_fp32_image(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-capture", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-capture")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(),
            config_guard_fingerprint="fixed-module-path-v1",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy-capture",
        declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
    )
    model_parameters = {
        name: parameter for name, parameter in manager.model.named_parameters() if name in state.tensor_layouts
    }

    assert manager.begin_gradient_capture(
        "policy-capture",
        scale_state=GradientScaleState.RAW_NUMERATOR,
    )
    first = {}
    for index, (name, parameter) in enumerate(model_parameters.items(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
        first[name] = parameter.grad.clone()
    assert manager.capture_gradient_numerators(
        "policy-capture",
        denominator=5,
        backward_completed=True,
    ) == (0, 0)
    for name, parameter in model_parameters.items():
        assert parameter.grad is None
        torch.testing.assert_close(state.gradient_scratch.numerators[name], first[name].float())
        assert state.gradient_scratch.numerators[name].dtype is torch.float32

    assert manager.begin_gradient_capture(
        "policy-capture",
        scale_state=GradientScaleState.RAW_NUMERATOR,
    )
    for parameter in model_parameters.values():
        parameter.grad = torch.full_like(parameter, 3)
    assert manager.capture_gradient_numerators(
        "policy-capture",
        denominator=7,
        backward_completed=True,
    ) == (0, 1)

    assert state.gradient_scratch.denominator == 12
    for name in model_parameters:
        torch.testing.assert_close(
            state.gradient_scratch.numerators[name],
            first[name].float() + 3,
        )


def test_uncompiled_manager_fails_closed(tmp_path):
    manager = LoRAAdapterManager(
        _DummyLoRAModel(),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
    )
    manager.register_adapter("policy-no-fallback", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-no-fallback")
    initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}

    with pytest.raises(AdapterGradientOwnershipError, match="requires a compiled plan"):
        manager.begin_gradient_capture(
            "policy-no-fallback",
            scale_state=GradientScaleState.RAW_NUMERATOR,
        )
    with pytest.raises(AdapterGradientOwnershipError, match="requires a compiled plan"):
        manager.optim_step("policy-no-fallback", lr=0.1)

    assert state.global_step == 0
    for name, parameter in state.local_params.items():
        torch.testing.assert_close(parameter, initial[name])


def test_capture_rejects_normalized_or_missing_required_gradients(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-reject", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-reject")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(),
            config_guard_fingerprint="fixed-module-path-v1",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy-reject",
        declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
    )

    with pytest.raises(AdapterGradientOwnershipError, match="unnormalized"):
        manager.begin_gradient_capture(
            "policy-reject",
            scale_state=GradientScaleState.PRE_NORMALIZED,
        )

    assert manager.begin_gradient_capture(
        "policy-reject",
        scale_state=GradientScaleState.RAW_NUMERATOR,
    )
    with pytest.raises(AdapterGradientOwnershipError, match="gradient is absent"):
        manager.capture_gradient_numerators(
            "policy-reject",
            denominator=1,
            backward_completed=True,
        )


def test_capture_adds_no_runtime_control_collective(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-static-capture", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-static-capture")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(),
            config_guard_fingerprint="fixed-module-path-v1",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy-static-capture",
        declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
    )
    assert manager.begin_gradient_capture(
        "policy-static-capture",
        scale_state=GradientScaleState.RAW_NUMERATOR,
    )
    for name, parameter in manager.model.named_parameters():
        if name in state.tensor_layouts:
            parameter.grad = torch.ones_like(parameter)

    def _unexpected_collective(*args, **kwargs):
        raise AssertionError("capture must not execute a control collective")

    monkeypatch.setattr(adapter_manager_module.torch.distributed, "all_reduce", _unexpected_collective)

    assert manager.capture_gradient_numerators(
        "policy-static-capture",
        denominator=1,
        backward_completed=True,
    ) == (0, 0)


def test_authoritative_plan_routes_legacy_sync_exclusions_before_mutation(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-route", lr=0.1, initialize_fresh=True)
    state = manager.get_adapter_state("policy-route")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(
                ReductionDomainPlan(
                    ReductionAxis.SEQUENCE_PARALLEL,
                    ReductionAuthority.ADAPTER_FINALIZER,
                    ReductionOperation.SUM,
                    "sequence_parallel",
                ),
            ),
            config_guard_fingerprint="fixed-module-path-v1",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy-route",
        declarations,
        model_generation="model-generation-1",
        adapter_generation="adapter-generation-1",
        group_memberships={"sequence_parallel": ((0,),)},
    )

    exclusions = manager.adapter_sync_exclusions("policy-route", ReductionAxis.SEQUENCE_PARALLEL)

    expected = {id(parameter) for name, parameter in manager.model.named_parameters() if name in state.tensor_layouts}
    assert exclusions == expected


def _compile_authoritative_dense_plan(
    manager: LoRAAdapterManager,
    model_id: str,
    *,
    adapter_generation: str = "adapter-generation-1",
    merged_forward: bool = False,
):
    state = manager.get_adapter_state(model_id)
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(),
            config_guard_fingerprint=f"fixed-module-path-merged-{merged_forward}",
            config_guard_fields=(("merged_forward", merged_forward),),
        )
        for name in state.tensor_layouts
    }
    return manager.compile_gradient_ownership_plan(
        model_id,
        declarations,
        model_generation="model-generation-1",
        adapter_generation=adapter_generation,
    )


def _capture_authoritative_dense_gradients(
    manager: LoRAAdapterManager,
    model_id: str,
    *,
    denominator: float,
    nonfinite: bool = False,
) -> dict[str, torch.Tensor]:
    state = manager.get_adapter_state(model_id)
    assert manager.begin_gradient_capture(model_id, scale_state=GradientScaleState.RAW_NUMERATOR)
    raw = {}
    for index, (name, parameter) in enumerate(manager.model.named_parameters(), start=1):
        if name not in state.tensor_layouts:
            continue
        parameter.grad = torch.full_like(parameter, float(index))
        if nonfinite and not raw:
            parameter.grad.reshape(-1)[0] = float("nan")
        raw[name] = parameter.grad.detach().clone()
    manager.capture_gradient_numerators(
        model_id,
        denominator=denominator,
        backward_completed=True,
    )
    return raw


def _install_model_gradients(manager: LoRAAdapterManager, model_id: str, value: float) -> dict[str, torch.Tensor]:
    state = manager.get_adapter_state(model_id)
    installed = {}
    for name, parameter in manager.model.named_parameters():
        if name not in state.tensor_layouts:
            continue
        parameter.grad = torch.full_like(parameter, value)
        installed[name] = parameter.grad.detach().clone()
    return installed


def test_capture_commit_uses_preallocated_staged_data_after_model_gradients_are_cleared(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-staged", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-staged")
    state = manager.get_adapter_state("policy-staged")
    storage = {name: tensor.untyped_storage().data_ptr() for name, tensor in state.gradient_scratch.numerators.items()}

    assert manager.begin_gradient_capture("policy-staged", scale_state=GradientScaleState.RAW_NUMERATOR)
    expected = _install_model_gradients(manager, "policy-staged", 3.0)
    manager.stage_gradient_numerators("policy-staged", denominator=2, backward_completed=True)
    for name, parameter in manager.model.named_parameters():
        if name in state.tensor_layouts:
            parameter.grad = None

    manager.commit_gradient_capture("policy-staged")

    assert {
        name: tensor.untyped_storage().data_ptr() for name, tensor in state.gradient_scratch.numerators.items()
    } == storage
    for name, numerator in state.gradient_scratch.numerators.items():
        torch.testing.assert_close(numerator, expected[name].float())


def test_direct_dtensor_capture_returns_completed_tensor_without_replacing_grad(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)

    class _FakeDTensor:
        def __init__(self, local, *, device_mesh="mesh", placements=("partial",)):
            self.local = local
            self.device_mesh = device_mesh
            self.placements = placements

        def redistribute(self, *, device_mesh, placements):
            assert device_mesh == "mesh"
            assert placements == ("shard",)
            return _FakeDTensor(self.local + 1, device_mesh=device_mesh, placements=placements)

        def to_local(self):
            return self.local

    monkeypatch.setattr(adapter_manager_module, "_HAS_DTENSOR", True)
    monkeypatch.setattr(adapter_manager_module, "DTensor", _FakeDTensor)
    original = _FakeDTensor(torch.tensor([2.0]))
    parameter = SimpleNamespace(
        grad=original,
        data=_FakeDTensor(torch.tensor([0.0]), placements=("shard",)),
    )
    item = SimpleNamespace(
        fqn="layer.lora_A",
        representation=GradientRepresentation.DIRECT_DTENSOR_CONTRIBUTION,
        capture_domains=(
            ReductionDomainPlan(
                ReductionAxis.FSDP_SHARD,
                ReductionAuthority.ADAPTER_CAPTURE,
                ReductionOperation.SUM,
                "fsdp",
            ),
        ),
    )
    state = SimpleNamespace(poisoned=False)

    completed = manager._capture_local_gradient(state, item, parameter)

    assert parameter.grad is original
    torch.testing.assert_close(completed, torch.tensor([3.0]))


def test_layout_identity_allows_fsdp2_dtensor_replacement_with_identical_static_contract(monkeypatch):
    class _Shard:
        dim = 0

    class _Mesh:
        mesh_dim_names = ("fsdp",)

    class _FakeDTensor:
        def __init__(self, local):
            self._local = local
            self.device_mesh = _Mesh()
            self.placements = (_Shard(),)
            self.shape = (4, 2)
            self.dtype = local.dtype

        def to_local(self):
            return self._local

    class _FakeParameter:
        def __init__(self, data):
            self.data = data

    original = _FakeParameter(_FakeDTensor(torch.zeros(2, 2)))
    replacement = _FakeParameter(_FakeDTensor(torch.zeros(2, 2)))
    manager = LoRAAdapterManager.__new__(LoRAAdapterManager)
    manager.model = SimpleNamespace(named_parameters=lambda: (("layer.lora_A", replacement),))
    manager._model_param_ids = {"layer.lora_A": id(original)}
    layout = SimpleNamespace(
        dtype=torch.float32,
        substrate_shape=(4, 2),
        local_substrate_shape=(2, 2),
        placement_signature=("dtensor", ("fsdp",), ("_Shard:0",), False),
        is_ep_owned=False,
    )
    state = SimpleNamespace(tensor_layouts={"layer.lora_A": layout})
    monkeypatch.setattr(adapter_manager_module, "_HAS_DTENSOR", True)
    monkeypatch.setattr(adapter_manager_module, "DTensor", _FakeDTensor)
    monkeypatch.setattr(adapter_manager_module, "nn", SimpleNamespace(Parameter=_FakeParameter))

    manager._validate_model_layout_identity(state)


def test_layout_identity_rejects_dtensor_replacement_with_changed_placement(monkeypatch):
    class _Replicate:
        pass

    class _Mesh:
        mesh_dim_names = ("fsdp",)

    class _FakeDTensor:
        shape = (4, 2)
        dtype = torch.float32
        device_mesh = _Mesh()
        placements = (_Replicate(),)

        def to_local(self):
            return torch.zeros(2, 2)

    original = _FakeDTensor()
    replacement = _FakeDTensor()
    manager = LoRAAdapterManager.__new__(LoRAAdapterManager)
    manager.model = SimpleNamespace(named_parameters=lambda: (("layer.lora_A", replacement),))
    manager._model_param_ids = {"layer.lora_A": id(original)}
    layout = SimpleNamespace(
        dtype=torch.float32,
        substrate_shape=(4, 2),
        local_substrate_shape=(2, 2),
        placement_signature=("dtensor", ("fsdp",), ("Shard:0",), False),
        is_ep_owned=False,
    )
    state = SimpleNamespace(tensor_layouts={"layer.lora_A": layout})
    monkeypatch.setattr(adapter_manager_module, "_HAS_DTENSOR", True)
    monkeypatch.setattr(adapter_manager_module, "DTensor", _FakeDTensor)

    with pytest.raises(RuntimeError, match="placement changed"):
        manager._validate_model_layout_identity(state)


def test_layout_identity_allows_fsdp2_dtensor_to_local_materialization() -> None:
    original = nn.Parameter(torch.zeros(2, 2))
    replacement = nn.Parameter(torch.zeros(2, 2))
    manager = LoRAAdapterManager.__new__(LoRAAdapterManager)
    manager.model = SimpleNamespace(named_parameters=lambda: (("layer.lora_A", replacement),))
    manager._model_param_ids = {"layer.lora_A": id(original)}
    layout = SimpleNamespace(
        dtype=torch.float32,
        local_substrate_shape=(2, 2),
        placement_signature=("dtensor", ("fsdp",), ("Shard:0",), False),
        is_ep_owned=False,
    )
    state = SimpleNamespace(tensor_layouts={"layer.lora_A": layout})

    manager._validate_model_layout_identity(state)


def test_layout_identity_rejects_ordinary_local_parameter_replacement() -> None:
    original = nn.Parameter(torch.zeros(2, 2))
    replacement = nn.Parameter(torch.zeros(2, 2))
    manager = LoRAAdapterManager.__new__(LoRAAdapterManager)
    manager.model = SimpleNamespace(named_parameters=lambda: (("layer.lora_A", replacement),))
    manager._model_param_ids = {"layer.lora_A": id(original)}
    layout = SimpleNamespace(
        dtype=torch.float32,
        local_substrate_shape=(2, 2),
        placement_signature=("local", False),
        is_ep_owned=False,
    )
    state = SimpleNamespace(tensor_layouts={"layer.lora_A": layout})

    with pytest.raises(RuntimeError, match="identity changed"):
        manager._validate_model_layout_identity(state)


def test_layout_identity_allows_owned_fsdp_bf16_compute_view_with_fp32_master() -> None:
    original = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
    replacement = nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    manager = LoRAAdapterManager.__new__(LoRAAdapterManager)
    manager.model = SimpleNamespace(named_parameters=lambda: (("layer.lora_A", replacement),))
    manager._model_param_ids = {"layer.lora_A": id(original)}
    manager._model_param_fsdp_managed = {"layer.lora_A": True}
    layout = SimpleNamespace(
        dtype=torch.float32,
        local_substrate_shape=(2, 2),
        placement_signature=("dtensor", ("fsdp",), ("Shard:0",), False),
        is_ep_owned=False,
    )
    state = SimpleNamespace(
        tensor_layouts={"layer.lora_A": layout},
        local_params={"layer.lora_A": nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))},
    )

    manager._validate_model_layout_identity(state)

    manager._model_param_fsdp_managed["layer.lora_A"] = False
    with pytest.raises(RuntimeError, match="dtype changed"):
        manager._validate_model_layout_identity(state)


def test_capture_commit_prevalidation_cannot_partially_mutate_numerators(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-atomic-stage", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-atomic-stage")
    state = manager.get_adapter_state("policy-atomic-stage")

    assert manager.begin_gradient_capture("policy-atomic-stage", scale_state=GradientScaleState.RAW_NUMERATOR)
    _install_model_gradients(manager, "policy-atomic-stage", 4.0)
    manager.stage_gradient_numerators("policy-atomic-stage", denominator=2, backward_completed=True)
    last_fqn = state.gradient_scratch.staged_parameter_fqns[-1]
    state.gradient_scratch.staged_numerators[last_fqn] = torch.ones(1, dtype=torch.float32)

    with pytest.raises(AdapterGradientOwnershipError, match="destination changed"):
        manager.commit_gradient_capture("policy-atomic-stage")

    assert all(not torch.count_nonzero(tensor) for tensor in state.gradient_scratch.numerators.values())
    manager.abort_gradient_capture("policy-atomic-stage")


def test_empty_authoritative_epoch_does_not_mutate_learning_rate_or_publication(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-empty-epoch", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-empty-epoch")
    state = manager.get_adapter_state("policy-empty-epoch")
    group_lrs = [group["lr"] for group in state.optimizer.param_groups]

    with pytest.raises(AdapterGradientOwnershipError, match="stale or empty"):
        manager.optim_step("policy-empty-epoch", lr=0.9)

    assert state.lr == pytest.approx(0.1)
    assert [group["lr"] for group in state.optimizer.param_groups] == group_lrs
    assert state.publication_eligible
    assert not state.publication_pending


def test_abort_gradient_epoch_discards_all_captures_and_is_idempotent(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-abort", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-abort")
    state = manager.get_adapter_state("policy-abort")
    for value in (2.0, 5.0):
        assert manager.begin_gradient_capture("policy-abort", scale_state=GradientScaleState.RAW_NUMERATOR)
        _install_model_gradients(manager, "policy-abort", value)
        manager.capture_gradient_numerators("policy-abort", denominator=3, backward_completed=True)
    manager.increment_forward_backward_step("policy-abort")
    monotonic_successes = state.global_forward_backward_step

    manager.abort_gradient_epoch("policy-abort")
    manager.abort_gradient_epoch("policy-abort")

    assert state.global_forward_backward_step == monotonic_successes
    assert state.global_step == 0
    assert state.publication_eligible
    assert not state.publication_pending
    assert state.gradient_scratch.next_capture_ordinal == 0
    assert state.gradient_scratch.denominator == 0
    assert state.gradient_scratch.source is None
    assert all(not torch.count_nonzero(tensor) for tensor in state.gradient_scratch.numerators.values())
    assert all(parameter.grad is None for parameter in state.local_params.values())


def test_abort_gradient_epoch_rejects_poison_and_publication_pending(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-abort-reject", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-abort-reject")
    state = manager.get_adapter_state("policy-abort-reject")

    state.publication_pending = True
    with pytest.raises(AdapterGradientMutationFailure, match="publication is pending"):
        manager.abort_gradient_epoch("policy-abort-reject")
    state.publication_pending = False
    state.poisoned = True
    with pytest.raises(AdapterGradientMutationFailure, match="cannot be recovered in-process"):
        manager.abort_gradient_epoch("policy-abort-reject")


def test_weight_publication_allows_clean_mid_epoch_but_checkpoint_publication_does_not(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-publication", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-publication")
    state = manager.get_adapter_state("policy-publication")
    _capture_authoritative_dense_gradients(manager, "policy-publication", denominator=2)

    assert not state.publication_eligible
    manager.validate_weight_publication("policy-publication")
    manager.prepare_forward("policy-publication")
    with pytest.raises(RuntimeError, match="checkpoint-publication-eligible"):
        manager.validate_strict_checkpoint_publication("policy-publication")
    with pytest.raises(RuntimeError, match="checkpoint-publication-eligible"):
        manager.save_adapter_state("policy-publication")

    state.publication_pending = True
    with pytest.raises(RuntimeError, match="optimizer command completion"):
        manager.validate_weight_publication("policy-publication")
    state.publication_pending = False
    state.poisoned = True
    with pytest.raises(RuntimeError, match="poisoned"):
        manager.validate_weight_publication("policy-publication")


def test_authoritative_dense_step_matches_analytical_gradient_clip_adamw_and_moments(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-step", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-step")
    state = manager.get_adapter_state("policy-step")
    initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    raw = _capture_authoritative_dense_gradients(manager, "policy-step", denominator=5)
    scratch_storage = {
        name: tensor.untyped_storage().data_ptr() for name, tensor in state.gradient_scratch.numerators.items()
    }
    normalized = {name: gradient.float() / 5 for name, gradient in raw.items()}
    expected_norm = math.sqrt(sum(float(gradient.square().sum()) for gradient in normalized.values()))
    expected_clip = min(1.0, 0.5 / (expected_norm + 1e-6))
    group = state.optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    lr = 0.1
    weight_decay = group["weight_decay"]
    eps = group["eps"]

    actual_norm = manager.optim_step(
        "policy-step",
        lr,
        gradient_clip=0.5,
    )

    assert actual_norm == pytest.approx(expected_norm)
    for name, parameter in state.local_params.items():
        gradient = normalized[name] * expected_clip
        expected_moment = gradient * (1 - beta1)
        expected_variance = gradient.square() * (1 - beta2)
        expected_parameter = initial[name] * (1 - lr * weight_decay) - lr * gradient / (gradient.abs() + eps)
        torch.testing.assert_close(parameter, expected_parameter)
        torch.testing.assert_close(state.optimizer.state[parameter]["exp_avg"], expected_moment)
        torch.testing.assert_close(state.optimizer.state[parameter]["exp_avg_sq"], expected_variance)
    assert state.publication_eligible
    assert not state.poisoned
    assert state.gradient_scratch.next_capture_ordinal == 0
    assert {
        name: tensor.untyped_storage().data_ptr() for name, tensor in state.gradient_scratch.numerators.items()
    } == scratch_storage
    assert all(not torch.count_nonzero(tensor) for tensor in state.gradient_scratch.numerators.values())
    assert state.weight_generation == 1
    assert state.global_step == 1


def test_authoritative_dense_step_adds_only_the_logical_norm_collective(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-collectives", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-collectives")
    _capture_authoritative_dense_gradients(manager, "policy-collectives", denominator=1)
    collectives = []

    monkeypatch.setattr(adapter_manager_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "get_world_size", lambda group=None: 2)

    def _all_reduce(tensor, op=None, group=None):
        collectives.append((tuple(tensor.shape), op, group))
        tensor.mul_(2)

    monkeypatch.setattr(adapter_manager_module.torch.distributed, "all_reduce", _all_reduce)

    manager.optim_step("policy-collectives", lr=0.1)

    assert len(collectives) == 1
    assert collectives[0][0] == ()
    assert collectives[0][1] is torch.distributed.ReduceOp.SUM
    state = manager.get_adapter_state("policy-collectives")
    assert state.publication_pending
    with pytest.raises(AdapterGradientOwnershipError, match="before the distributed optimizer command commits"):
        manager.begin_gradient_capture("policy-collectives", scale_state=GradientScaleState.RAW_NUMERATOR)


def test_exact_lm_head_optimizer_coherence_accepts_scalar_tensor_state(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    parameter = nn.Parameter(torch.ones(1))
    manager._exact_lm_head_replicated_param_names = {"lm_head.lora_A"}
    state = SimpleNamespace(
        local_params={"lm_head.lora_A": parameter},
        optimizer=SimpleNamespace(state={parameter: {"step": torch.tensor(1.0), "exp_avg": torch.ones(1)}}),
    )
    group = object()

    monkeypatch.setattr(adapter_manager_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "get_world_size", lambda group=None: 16)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "xorl.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(lm_head_tp_group=group),
    )

    manager._validate_exact_lm_head_tp_coherence(state, include_optimizer=True)


def test_authoritative_semantic_rejection_is_recoverable_before_mutation(tmp_path):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-semantic", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-semantic")
    state = manager.get_adapter_state("policy-semantic")
    initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    _capture_authoritative_dense_gradients(manager, "policy-semantic", denominator=1, nonfinite=True)

    original_group_lrs = [group["lr"] for group in state.optimizer.param_groups]
    with pytest.raises(AdapterGradientOwnershipError, match="nonfinite"):
        manager.optim_step("policy-semantic", 0.025)

    for name, parameter in state.local_params.items():
        torch.testing.assert_close(parameter, initial[name])
    assert not state.poisoned
    assert state.publication_eligible
    assert state.gradient_scratch.next_capture_ordinal == 0
    assert state.lr == pytest.approx(0.1)
    assert state.session_spec["optimizer_config"]["learning_rate"] == pytest.approx(0.1)
    assert [group["lr"] for group in state.optimizer.param_groups] == original_group_lrs


def test_authoritative_optimizer_failure_poisons_session_and_blocks_publication(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-fatal", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-fatal")
    state = manager.get_adapter_state("policy-fatal")
    committed = manager.save_adapter_state("policy-fatal")
    committed_parameters = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    _capture_authoritative_dense_gradients(manager, "policy-fatal", denominator=1)
    parameter = next(iter(state.local_params.values()))

    def _partial_step():
        parameter.data.add_(1)
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(state.optimizer, "step", _partial_step)
    with pytest.raises(AdapterGradientMutationFailure, match="recover from the last checkpoint"):
        manager.optim_step("policy-fatal", 0.1)

    assert state.poisoned
    assert not state.publication_eligible
    with pytest.raises(RuntimeError, match="poisoned"):
        manager.save_adapter_state("policy-fatal")
    with pytest.raises(RuntimeError, match="cannot be recovered in-process"):
        manager.load_adapter_state("policy-fatal", committed["path"], load_optimizer=True)
    assert any(not torch.equal(parameter, committed_parameters[name]) for name, parameter in state.local_params.items())


def test_authoritative_collective_failure_is_fatal_but_leaves_parameters_unchanged(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path)
    manager.register_adapter("policy-collective", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-collective")
    state = manager.get_adapter_state("policy-collective")
    initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    _capture_authoritative_dense_gradients(manager, "policy-collective", denominator=1)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(adapter_manager_module.torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        adapter_manager_module.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected communicator failure")),
    )

    with pytest.raises(AdapterGradientCollectiveFailure, match="restart the distributed process"):
        manager.optim_step("policy-collective", 0.1)

    for name, parameter in state.local_params.items():
        torch.testing.assert_close(parameter, initial[name])
    assert state.poisoned
    assert not state.publication_eligible


def test_save_adapter_state_preserves_lora_weight_dtype(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="adamw")
    manager.register_adapter("policy-fp32", lr=0.1, initialize_fresh=True)

    save_path = Path(manager.save_adapter_state("policy-fp32")["path"])
    weights = safetensors_load_file(str(save_path / "adapter_model.safetensors"))

    assert weights["base_model.model.model.layers.0.self_attn.o_proj.lora_A"].dtype == torch.float32
    assert weights["base_model.model.model.layers.0.self_attn.o_proj.lora_B"].dtype == torch.float32


def test_save_adapter_state_rejects_path_outside_checkpoint_root(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="adamw")
    manager.register_adapter("policy-contained", lr=0.1, initialize_fresh=True)

    with pytest.raises(ValueError, match="escapes configured root"):
        manager.save_adapter_state("policy-contained", path=str(tmp_path / "outside"))


def test_save_and_validate_adapter_state_binds_strict_target_manifest(tmp_path):
    manifest = {
        "schema_version": 1,
        "target_modules": ["o_proj"],
        "expected_modules": [
            {
                "pattern": "model.layers.*.self_attn.o_proj",
                "count": 1,
                "rank": 4,
            }
        ],
        "allow_unlisted": False,
    }
    manager = _build_manager(
        tmp_path,
        optimizer_type="adamw",
        lora_config={
            "base_model": "Qwen/Qwen3-8B",
            "lora_rank": 4,
            "lora_alpha": 16,
            "lora_target_manifest": manifest,
        },
    )
    manager.register_adapter("policy-strict", lr=0.1, initialize_fresh=True)
    checkpoint = Path(manager.save_adapter_state("policy-strict")["path"])

    assert json.loads((checkpoint / "lora_target_manifest.json").read_text()) == manifest
    manager._validate_checkpoint_adapter_config(str(checkpoint))

    manager.lora_config["lora_target_manifest"] = {
        **manifest,
        "expected_modules": [{**manifest["expected_modules"][0], "count": 2}],
    }
    with pytest.raises(ValueError, match="target manifest does not match"):
        manager._validate_checkpoint_adapter_config(str(checkpoint))


def test_load_adapter_state_uses_checkpoint_optimizer_contract_for_fresh_session(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="signsgd")
    source_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    result = target_manager.load_adapter_state("policy-b", checkpoint_path, load_optimizer=True)

    assert result["model_id"] == "policy-b"
    assert isinstance(target_manager.get_adapter_state("policy-b").optimizer, SignSGD)
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["type"] == "signsgd"


def test_authoritative_checkpoint_restore_resets_lifecycle_and_preserves_plan_contract(tmp_path):
    source_manager = _build_manager(tmp_path / "source")
    source_manager.register_adapter("policy-restore", lr=0.1, initialize_fresh=True)
    source_plan = _compile_authoritative_dense_plan(source_manager, "policy-restore")
    checkpoint_path = source_manager.save_adapter_state("policy-restore")["path"]

    target_manager = _build_manager(tmp_path / "target")
    result = target_manager.load_adapter_state("policy-restore", checkpoint_path, load_optimizer=True)
    state = target_manager.get_adapter_state("policy-restore")

    assert result["gradient_ownership_plan_fingerprint"] == source_plan.fingerprint
    assert not state.poisoned
    assert not state.publication_pending
    assert state.publication_eligible
    assert state.gradient_scratch.next_capture_ordinal == 0
    assert state.gradient_scratch.denominator == 0
    assert all(parameter.grad is None for parameter in state.local_params.values())
    assert all(parameter.grad is None for parameter in target_manager.model.parameters())
    restored_plan = _compile_authoritative_dense_plan(target_manager, "policy-restore")
    assert restored_plan.fingerprint == result["gradient_ownership_plan_fingerprint"]


def test_authoritative_checkpoint_ignores_identity_label_difference_when_direct_contract_matches(tmp_path):
    source_manager = _build_manager(tmp_path / "source")
    source_manager.register_adapter("policy-plan", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(source_manager, "policy-plan")
    checkpoint_path = source_manager.save_adapter_state("policy-plan")["path"]

    target_manager = _build_manager(tmp_path / "target")
    target_manager.register_adapter("policy-plan", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(
        target_manager,
        "policy-plan",
        adapter_generation="incompatible-adapter-generation",
    )
    source_plan = source_manager.get_adapter_state("policy-plan").gradient_ownership_plan
    target_plan = target_manager.get_adapter_state("policy-plan").gradient_ownership_plan
    assert source_plan is not None and target_plan is not None
    assert source_plan.fingerprint != target_plan.fingerprint

    result = target_manager.load_adapter_state("policy-plan", checkpoint_path, load_optimizer=True)

    assert result["gradient_ownership_plan_fingerprint"] == source_plan.fingerprint


def test_authoritative_checkpoint_rejects_direct_plan_mismatch_before_mutation(tmp_path):
    source_manager = _build_manager(tmp_path / "source")
    source_manager.register_adapter("policy-plan", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(source_manager, "policy-plan", merged_forward=False)
    _capture_authoritative_dense_gradients(source_manager, "policy-plan", denominator=1)
    source_manager.optim_step("policy-plan", lr=0.08)
    checkpoint_path = source_manager.save_adapter_state("policy-plan")["path"]

    target_manager = _build_manager(tmp_path / "target")
    target_manager.register_adapter("policy-plan", lr=0.1, initialize_fresh=True)
    _compile_authoritative_dense_plan(target_manager, "policy-plan", merged_forward=True)
    _capture_authoritative_dense_gradients(target_manager, "policy-plan", denominator=1)
    target_manager.optim_step("policy-plan", lr=0.06)
    state = target_manager.get_adapter_state("policy-plan")
    before_parameters = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    before_optimizer = _clone_state_to_cpu(state.optimizer.state_dict())
    before_session_spec = deepcopy(state.session_spec)
    before_step = state.global_step
    before_lr = state.lr

    with pytest.raises(ValueError, match=r"topology/producer contract.*config_guard_fields.merged_forward"):
        target_manager.load_adapter_state("policy-plan", checkpoint_path, load_optimizer=True)

    for name, parameter in state.local_params.items():
        assert torch.equal(parameter, before_parameters[name])
    assert _same_optimizer_value(before_optimizer, state.optimizer.state_dict())
    assert state.session_spec == before_session_spec
    assert state.global_step == before_step
    assert state.lr == before_lr


def test_adapter_coordinator_loads_checkpoint_without_placeholder_spec_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="signsgd")
    source_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)
    source_checkpoint = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    checkpoint_path = Path(target_manager.checkpoint_dir).parent / "weights" / "policy-b"
    checkpoint_path.parent.mkdir(parents=True)
    shutil.copytree(source_checkpoint, checkpoint_path)
    coordinator = AdapterCoordinator(
        trainer=_CoordinatorTrainer(target_manager),
        rank=0,
        world_size=1,
        cpu_group=None,
    )

    result = asyncio.run(
        coordinator.handle_load_adapter_state(
            {
                "payload": AdapterStateData(
                    model_id="policy-b",
                    path=str(checkpoint_path),
                    load_optimizer=True,
                )
            }
        )
    )

    assert result["model_id"] == "policy-b"
    assert isinstance(target_manager.get_adapter_state("policy-b").optimizer, SignSGD)
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["learning_rate"] == pytest.approx(
        0.1
    )


def test_adapter_coordinator_auto_load_evicted_uses_checkpoint_session_spec(tmp_path):
    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    source_manager = _build_manager(tmp_path / "source", optimizer_type="signsgd")
    source_manager.register_adapter("policy-evicted", lr=0.2, initialize_fresh=True)
    checkpoint_path = Path(target_manager.checkpoint_dir) / "evicted" / "policy-evicted"
    source_checkpoint = source_manager.save_adapter_state("policy-evicted")["path"]
    checkpoint_path.parent.mkdir(parents=True)
    shutil.copytree(source_checkpoint, checkpoint_path)

    coordinator = AdapterCoordinator(
        trainer=_CoordinatorTrainer(target_manager),
        rank=0,
        world_size=1,
        cpu_group=None,
    )

    was_loaded, loaded_path = coordinator.auto_load_if_evicted("policy-evicted")

    assert was_loaded is True
    assert loaded_path == str(checkpoint_path)
    assert isinstance(target_manager.get_adapter_state("policy-evicted").optimizer, SignSGD)
    assert target_manager.get_adapter_session_spec("policy-evicted")["optimizer_config"]["learning_rate"] == (
        pytest.approx(0.2)
    )


def test_load_adapter_state_rejects_registered_session_spec_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="signsgd")
    source_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    target_manager.register_adapter("policy-b", lr=0.1, initialize_fresh=True)

    with pytest.raises(ValueError, match="Checkpoint optimizer type is incompatible"):
        target_manager.load_adapter_state("policy-b", checkpoint_path, load_optimizer=True)


def test_load_adapter_state_rejects_checkpoint_outside_trusted_roots(tmp_path, monkeypatch):
    server_root = tmp_path / "server"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(server_root))
    manager = _build_manager(server_root, optimizer_type="adamw")

    with pytest.raises(ValueError, match="escapes configured root"):
        manager.load_adapter_state("policy-b", str(outside), load_optimizer=False)


def test_load_adapter_state_allows_lr_override_for_registered_session(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1, weight_decay=0.01)
    source_manager.register_adapter("policy-lr-override", session_spec=source_spec, initialize_fresh=True)
    checkpoint_path = source_manager.save_adapter_state("policy-lr-override")["path"]

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    target_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)
    target_manager.register_adapter("policy-lr-override", session_spec=target_spec, initialize_fresh=True)

    result = target_manager.load_adapter_state(
        "policy-lr-override",
        checkpoint_path,
        load_optimizer=True,
        lr=0.2,
    )

    target_state = target_manager.get_adapter_state("policy-lr-override")
    assert result["model_id"] == "policy-lr-override"
    assert target_state.lr == pytest.approx(0.2)
    assert target_state.optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    assert target_manager.get_adapter_session_spec("policy-lr-override")["optimizer_config"][
        "learning_rate"
    ] == pytest.approx(0.2)


def test_load_adapter_state_allows_weights_only_optimizer_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="signsgd")
    source_spec = _session_spec(rank=4, alpha=16, optimizer_type="signsgd", lr=0.1)
    source_manager.register_adapter("policy-b", session_spec=source_spec, initialize_fresh=True)
    source_state = source_manager.get_adapter_state("policy-b")
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(1.25)
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(0.5)
    checkpoint_path = source_manager.save_adapter_state("policy-b")["path"]

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    target_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)
    target_manager.register_adapter("policy-b", session_spec=target_spec, initialize_fresh=True)
    target_fresh_a = (
        target_manager.get_adapter_state("policy-b")
        .lora_params["model.layers.0.self_attn.o_proj.lora_A"]
        .detach()
        .clone()
    )
    target_fresh_b = (
        target_manager.get_adapter_state("policy-b")
        .lora_params["model.layers.0.self_attn.o_proj.lora_B"]
        .detach()
        .clone()
    )

    result = target_manager.load_adapter_state("policy-b", checkpoint_path, load_optimizer=False)

    target_state = target_manager.get_adapter_state("policy-b")
    assert result["model_id"] == "policy-b"
    assert isinstance(target_state.optimizer, torch.optim.AdamW)
    assert target_state.lr == pytest.approx(0.05)
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["type"] == "adamw"
    assert target_manager.get_adapter_session_spec("policy-b")["optimizer_config"]["learning_rate"] == pytest.approx(
        0.05
    )
    assert target_state.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"],
        torch.full((4, 8), 1.25),
    )
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"],
        torch.full((8, 4), 0.5),
    )
    assert not torch.equal(target_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"], target_fresh_a)
    assert not torch.equal(target_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"], target_fresh_b)


def test_load_adapter_state_weights_only_restores_checkpoint_lr_for_same_optimizer_contract(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1, weight_decay=0.01)
    source_manager.register_adapter("policy-lr-restore", session_spec=source_spec, initialize_fresh=True)
    _capture_authoritative_dense_gradients(source_manager, "policy-lr-restore", denominator=1)
    source_manager.optim_step("policy-lr-restore", lr=0.25)
    checkpoint_path = source_manager.save_adapter_state("policy-lr-restore")["path"]

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    target_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)
    target_manager.register_adapter("policy-lr-restore", session_spec=target_spec, initialize_fresh=True)

    target_manager.load_adapter_state("policy-lr-restore", checkpoint_path, load_optimizer=False)

    target_state = target_manager.get_adapter_state("policy-lr-restore")
    assert target_state.lr == pytest.approx(0.25)
    assert target_state.optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
    assert target_manager.get_adapter_session_spec("policy-lr-restore")["optimizer_config"][
        "learning_rate"
    ] == pytest.approx(0.25)


def test_load_adapter_state_rejects_checkpoint_target_module_mismatch(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="adamw")
    source_manager.register_adapter("policy-structure", lr=0.1, initialize_fresh=True)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-structure")["path"])

    adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config["target_modules"] = ["q_proj"]
    (checkpoint_path / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    with pytest.raises(ValueError, match="target_modules"):
        target_manager.load_adapter_state("policy-structure", str(checkpoint_path), load_optimizer=True)


def test_load_adapter_state_rejects_checkpoint_with_missing_lora_tensors(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="adamw")
    source_manager.register_adapter("policy-missing", lr=0.1, initialize_fresh=True)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-missing")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    weights.pop("base_model.model.model.layers.0.self_attn.o_proj.lora_B")
    safetensors_save_file(weights, str(weights_path))

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    with pytest.raises(ValueError, match="parameter set does not match"):
        target_manager.load_adapter_state("policy-missing", str(checkpoint_path), load_optimizer=True)


def test_load_adapter_state_rolls_back_freshly_registered_adapter_on_failure(tmp_path):
    source_manager = _build_manager(tmp_path, optimizer_type="adamw")
    source_manager.register_adapter("policy-rollback", lr=0.1, initialize_fresh=True)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-rollback")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    weights.pop("base_model.model.model.layers.0.self_attn.o_proj.lora_B")
    safetensors_save_file(weights, str(weights_path))

    target_manager = _build_manager(tmp_path, optimizer_type="adamw")
    assert "policy-rollback" not in target_manager.adapters

    with pytest.raises(ValueError, match="parameter set does not match"):
        target_manager.load_adapter_state("policy-rollback", str(checkpoint_path), load_optimizer=True)

    assert "policy-rollback" not in target_manager.adapters


def test_load_adapter_state_accepts_weight_suffixed_checkpoint_tensor_names(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_manager.register_adapter("policy-weight-suffix", lr=0.1, initialize_fresh=True)
    source_state = source_manager.get_adapter_state("policy-weight-suffix")
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(1.5)
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(0.75)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-weight-suffix")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    renamed_weights = {}
    for key, value in weights.items():
        renamed_weights[f"{key}.weight"] = value
    safetensors_save_file(renamed_weights, str(weights_path))

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    result = target_manager.load_adapter_state("policy-weight-suffix", str(checkpoint_path), load_optimizer=True)

    target_state = target_manager.get_adapter_state("policy-weight-suffix")
    assert result["model_id"] == "policy-weight-suffix"
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"],
        torch.full((4, 8), 1.5),
    )
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"],
        torch.full((8, 4), 0.75),
    )


def test_load_adapter_state_accepts_indexed_sharded_peft_checkpoint(tmp_path):
    source_manager = _build_manager(tmp_path / "source", optimizer_type="adamw")
    source_manager.register_adapter("policy-sharded", lr=0.1, initialize_fresh=True)
    source_state = source_manager.get_adapter_state("policy-sharded")
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(2.5)
    source_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(1.25)
    checkpoint_path = Path(source_manager.save_adapter_state("policy-sharded")["path"])

    weights_path = checkpoint_path / "adapter_model.safetensors"
    weights = safetensors_load_file(str(weights_path))
    weight_items = sorted(weights.items())
    weight_map = {}
    for index, (key, value) in enumerate(weight_items, start=1):
        shard_name = f"adapter_model-{index:05d}-of-{len(weight_items):05d}.safetensors"
        safetensors_save_file({key: value}, str(checkpoint_path / shard_name))
        weight_map[key] = shard_name
    (checkpoint_path / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    weights_path.unlink()

    target_manager = _build_manager(tmp_path / "target", optimizer_type="adamw")
    result = target_manager.load_adapter_state("policy-sharded", str(checkpoint_path), load_optimizer=True)

    target_state = target_manager.get_adapter_state("policy-sharded")
    assert result["model_id"] == "policy-sharded"
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"],
        torch.full((4, 8), 2.5),
    )
    assert torch.allclose(
        target_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"],
        torch.full((8, 4), 1.25),
    )


def test_load_adapter_state_rejects_checkpoint_rank_exceeding_model_capacity(tmp_path):
    source_manager = _build_manager(
        tmp_path / "source",
        optimizer_type="adamw",
        max_rank=8,
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 8, "lora_alpha": 16},
    )
    source_manager.register_adapter(
        "policy-r8",
        session_spec=_session_spec(rank=8, alpha=16, optimizer_type="adamw", lr=0.1),
        initialize_fresh=True,
    )
    checkpoint_path = source_manager.save_adapter_state("policy-r8")["path"]

    target_manager = _build_manager(
        tmp_path / "target",
        optimizer_type="adamw",
        max_rank=4,
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )
    with pytest.raises(ValueError, match="exceeds live model LoRA capacity"):
        target_manager.load_adapter_state("policy-r8", checkpoint_path, load_optimizer=True)


def test_register_adapter_refuses_to_evict_dirty_adapter(tmp_path):
    manager = _build_manager(tmp_path, max_adapters=1, optimizer_type="adamw")
    manager.register_adapter("policy-a", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1))

    dirty_state = manager.get_adapter_state("policy-a")
    dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].grad = torch.ones_like(
        dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"]
    )

    with pytest.raises(RuntimeError, match="pending gradients"):
        manager.register_adapter(
            "policy-b", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.2)
        )

    assert manager.has_adapter("policy-a")
    assert not manager.has_adapter("policy-b")


def test_register_adapter_evicts_clean_adapter_before_dirty_one(tmp_path):
    manager = _build_manager(tmp_path, max_adapters=2, optimizer_type="adamw")
    manager.register_adapter("policy-a", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1))
    manager.register_adapter("policy-b", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.2))

    dirty_state = manager.get_adapter_state("policy-a")
    clean_state = manager.get_adapter_state("policy-b")
    dirty_state.last_access_time = 1.0
    clean_state.last_access_time = 2.0
    dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].grad = torch.ones_like(
        dirty_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"]
    )

    manager.register_adapter("policy-c", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.3))

    assert manager.has_adapter("policy-a")
    assert not manager.has_adapter("policy-b")
    assert manager.has_adapter("policy-c")


def test_register_adapter_refuses_multi_rank_eviction(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path, max_adapters=1, optimizer_type="adamw")
    manager.register_adapter("policy-a", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1))
    monkeypatch.setattr(adapter_manager_module, "_optimizer_shard_rank_world", lambda: (0, 8))

    with pytest.raises(RuntimeError, match="disabled for multi-rank training"):
        manager.register_adapter(
            "policy-b", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.2)
        )

    assert manager.has_adapter("policy-a")
    assert not manager.has_adapter("policy-b")


def test_register_adapter_keeps_resident_adapter_when_auto_save_fails(tmp_path, monkeypatch):
    manager = _build_manager(tmp_path, max_adapters=1, optimizer_type="adamw")
    manager.auto_save_on_eviction = True
    manager.register_adapter("policy-a", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1))

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(manager, "save_adapter_state", fail_save)
    with pytest.raises(RuntimeError, match="injected save failure"):
        manager.register_adapter(
            "policy-b", session_spec=_session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.2)
        )

    assert manager.has_adapter("policy-a")
    assert not manager.has_adapter("policy-b")


def test_multi_adapter_manager_supports_mixed_ranks_and_optimizers(tmp_path):
    manager = _build_manager(
        tmp_path,
        optimizer_type="adamw",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )

    small_spec = _session_spec(rank=2, alpha=8, optimizer_type="signsgd", lr=0.2)
    large_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.05, weight_decay=0.01)

    manager.register_adapter("policy-small", session_spec=small_spec, initialize_fresh=True)
    manager.register_adapter("policy-large", session_spec=large_spec, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-small")
    _compile_authoritative_dense_plan(manager, "policy-large")

    small_state = manager.get_adapter_state("policy-small")
    large_state = manager.get_adapter_state("policy-large")
    layer = manager.model.model.layers[0].self_attn.o_proj

    assert isinstance(small_state.optimizer, SignSGD)
    assert isinstance(large_state.optimizer, torch.optim.AdamW)
    assert tuple(small_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].shape) == (2, 8)
    assert tuple(small_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].shape) == (8, 2)
    assert tuple(large_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].shape) == (4, 8)
    assert tuple(large_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].shape) == (8, 4)

    small_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(1.5)
    small_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(2.5)
    large_state.lora_params["model.layers.0.self_attn.o_proj.lora_A"].data.fill_(3.5)
    large_state.lora_params["model.layers.0.self_attn.o_proj.lora_B"].data.fill_(4.5)

    manager.prepare_forward("policy-small")
    assert layer.active_r == 2
    assert layer.active_lora_alpha == 8
    assert torch.allclose(layer.lora_A[:2], torch.full((2, 8), 1.5))
    assert torch.count_nonzero(layer.lora_A[2:]) == 0
    assert torch.allclose(layer.lora_B[:, :2], torch.full((8, 2), 2.5))
    assert torch.count_nonzero(layer.lora_B[:, 2:]) == 0

    layer.lora_A.grad = torch.full_like(layer.lora_A, 1.0)
    layer.lora_B.grad = torch.full_like(layer.lora_B, 2.0)
    assert manager.begin_gradient_capture("policy-small", scale_state=GradientScaleState.RAW_NUMERATOR)
    manager.capture_gradient_numerators("policy-small", denominator=1, backward_completed=True)
    small_grad_norm = manager.optim_step("policy-small", lr=0.2)
    assert small_grad_norm > 0
    assert manager.get_global_step("policy-small") == 1

    manager.prepare_forward("policy-large")
    assert layer.active_r == 4
    assert layer.active_lora_alpha == 16
    assert torch.allclose(layer.lora_A, torch.full((4, 8), 3.5))
    assert torch.allclose(layer.lora_B, torch.full((8, 4), 4.5))

    layer.lora_A.grad = torch.full_like(layer.lora_A, 3.0)
    layer.lora_B.grad = torch.full_like(layer.lora_B, 4.0)
    assert manager.begin_gradient_capture("policy-large", scale_state=GradientScaleState.RAW_NUMERATOR)
    manager.capture_gradient_numerators("policy-large", denominator=1, backward_completed=True)
    large_grad_norm = manager.optim_step("policy-large", lr=0.05)
    assert large_grad_norm > 0
    assert manager.get_global_step("policy-large") == 1
    assert large_state.optimizer.state

    small_checkpoint = manager.save_adapter_state("policy-small")["path"]
    large_checkpoint = manager.save_adapter_state("policy-large")["path"]

    reloaded_manager = _build_manager(
        tmp_path,
        optimizer_type="sgd",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )
    reloaded_manager.load_adapter_state("policy-small", small_checkpoint, load_optimizer=True)
    reloaded_manager.load_adapter_state("policy-large", large_checkpoint, load_optimizer=True)

    assert isinstance(reloaded_manager.get_adapter_state("policy-small").optimizer, SignSGD)
    assert isinstance(reloaded_manager.get_adapter_state("policy-large").optimizer, torch.optim.AdamW)
    assert reloaded_manager.get_adapter_session_spec("policy-small")["lora_config"]["lora_rank"] == 2
    assert reloaded_manager.get_adapter_session_spec("policy-large")["lora_config"]["lora_rank"] == 4


def test_save_adapter_state_persists_current_learning_rate(tmp_path):
    manager = _build_manager(tmp_path, optimizer_type="adamw")
    session_spec = _session_spec(rank=4, alpha=16, optimizer_type="adamw", lr=0.1, weight_decay=0.01)
    manager.register_adapter("policy-lr", session_spec=session_spec, initialize_fresh=True)
    _compile_authoritative_dense_plan(manager, "policy-lr")
    _capture_authoritative_dense_gradients(manager, "policy-lr", denominator=1)
    manager.optim_step("policy-lr", lr=0.25)
    checkpoint_path = Path(manager.save_adapter_state("policy-lr")["path"])
    session_spec_json = json.loads((checkpoint_path / "session_spec.json").read_text(encoding="utf-8"))
    metadata_json = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))

    assert manager.get_adapter_session_spec("policy-lr")["optimizer_config"]["learning_rate"] == pytest.approx(0.25)
    assert session_spec_json["optimizer_config"]["learning_rate"] == pytest.approx(0.25)
    assert metadata_json["lr"] == pytest.approx(0.25)
    assert metadata_json["optimizer"]["learning_rate"] == pytest.approx(0.25)
    assert metadata_json["optimizer"]["optimizer_dtype"] == "bf16"
    assert metadata_json["optimizer_state"]["tensor_fields"] == {
        "exp_avg:torch.float32": 2,
        "exp_avg_sq:torch.float32": 2,
        "step:torch.float32": 2,
    }


def test_register_adapter_hoists_common_adam_hparams_out_of_optimizer_kwargs(tmp_path):
    manager = _build_manager(
        tmp_path,
        optimizer_type="adamw",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )

    session_spec = normalize_session_spec(
        base_model="Qwen/Qwen3-8B",
        raw_lora_config={"lora_rank": 4, "lora_alpha": 16},
        raw_optimizer_config={
            "type": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.02,
            "optimizer_kwargs": {
                "betas": [0.8, 0.88],
                "eps": 1e-7,
                "capturable": True,
            },
        },
        default_rank=4,
        default_alpha=16,
        max_lora_rank=16,
        default_optimizer_type="adamw",
        default_learning_rate=1e-5,
        default_weight_decay=0.01,
        default_optimizer_dtype="bf16",
        default_optimizer_kwargs={},
        server_lora_config={"enable_lora": True, "lora_rank": 4, "lora_alpha": 16, "max_lora_rank": 16},
    )

    manager.register_adapter("policy-adam", session_spec=session_spec, initialize_fresh=True)
    state = manager.get_adapter_state("policy-adam")

    assert isinstance(state.optimizer, torch.optim.AdamW)
    assert state.optimizer.defaults["betas"] == (0.8, 0.88)
    assert state.optimizer.defaults["eps"] == pytest.approx(1e-7)
    assert state.optimizer.defaults["capturable"] is True
    assert manager.get_adapter_session_spec("policy-adam")["optimizer_config"]["optimizer_kwargs"] == {
        "capturable": True
    }


def test_muon_set_lr_preserves_muon_param_group_lr(tmp_path):
    manager = _build_manager(
        tmp_path,
        optimizer_type="muon",
        lora_config={"base_model": "Qwen/Qwen3-8B", "lora_rank": 4, "lora_alpha": 16},
    )
    session_spec = _session_spec(rank=4, alpha=16, optimizer_type="muon", lr=1e-4, weight_decay=0.01)
    session_spec["optimizer_config"]["optimizer_kwargs"] = {
        "muon_lr": 0.02,
        "muon_ns_use_quack_kernels": False,
    }

    manager.register_adapter("policy-muon", session_spec=session_spec, initialize_fresh=True)
    state = manager.get_adapter_state("policy-muon")
    muon_groups = [param_group for param_group in state.optimizer.param_groups if param_group.get("use_muon", False)]
    assert muon_groups
    assert all(param_group["lr"] == pytest.approx(0.02) for param_group in muon_groups)

    manager.set_lr("policy-muon", 2e-4)
    assert state.lr == pytest.approx(2e-4)
    assert all(param_group["lr"] == pytest.approx(0.02) for param_group in muon_groups)

    manager.set_lr("policy-muon", 3e-4)
    assert state.lr == pytest.approx(3e-4)
    assert all(param_group["lr"] == pytest.approx(0.02) for param_group in muon_groups)
