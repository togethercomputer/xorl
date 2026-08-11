"""Tests for checkpoint-manager save failure handling."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import xorl.server.runner.checkpoint.manager as checkpoint_manager_module
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.checkpoint.manager import CheckpointManager
from xorl.server.session_spec import normalize_session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeOptimizer:
    def state_dict(self):
        return {"state": {}, "param_groups": []}


class _FakeAdapterState:
    def __init__(self):
        self.global_step = 7
        self.global_forward_backward_step = 11
        self.lr = 2e-5
        self.optimizer = _FakeOptimizer()
        self.local_params = {"adapter.weight.lora_A": nn.Parameter(torch.ones(1, 1))}
        self.tensor_layouts = {}
        self.layout_fingerprint = "f" * 64
        self.gradient_ownership_plan = None
        self.session_spec = {
            "lora_config": {
                "lora_rank": 4,
                "lora_alpha": 16,
            },
            "optimizer_config": {
                "type": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "optimizer_dtype": "bf16",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "optimizer_kwargs": {},
            },
        }


class _FakeAdapterManager:
    def __init__(self):
        self.checkpoint_dir = "/tmp/adapters"
        self.current_adapter_id = "policy-a"
        self.adapters = {"policy-a": _FakeAdapterState()}

    def get_adapter_state(self, model_id: str):
        return self.adapters[model_id]

    def get_global_step(self, model_id: str) -> int:
        return self.adapters[model_id].global_step

    def get_adapter_session_spec(self, model_id: str):
        return self.adapters[model_id].session_spec

    def validate_weight_publication(self, model_id: str) -> None:
        assert model_id in self.adapters

    def validate_strict_checkpoint_publication(self, model_id: str) -> None:
        assert model_id in self.adapters

    def switch_adapter(self, model_id: str, auto_register: bool = False) -> bool:
        return model_id in self.adapters


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


class _ExactActiveLoRAComponent(nn.Module):
    _glm52_exact_active_lora_component = True


def _build_checkpoint_manager() -> CheckpointManager:
    manager = object.__new__(CheckpointManager)
    manager.rank = 0
    manager.local_rank = 0
    manager.lora_config = {"enable_lora": True}
    manager._adapter_manager = _FakeAdapterManager()
    return manager


def _build_fast_save_manager(tmp_path: Path) -> CheckpointManager:
    model = _DummyLoRAModel(max_rank=4)
    adapter_manager = LoRAAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "adapters"),
        auto_save_on_eviction=False,
        lora_config={
            "base_model": "Qwen/Qwen3-8B",
            "lora_rank": 4,
            "lora_alpha": 16,
        },
    )
    adapter_manager.register_adapter(
        "policy-a",
        session_spec=normalize_session_spec(
            base_model="Qwen/Qwen3-8B",
            raw_lora_config={
                "lora_rank": 4,
                "lora_alpha": 16,
            },
            raw_optimizer_config={
                "type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "optimizer_dtype": "bf16",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "optimizer_kwargs": {},
            },
            default_rank=4,
            default_alpha=16,
            max_lora_rank=4,
            default_optimizer_type="adamw",
            default_learning_rate=1e-4,
            default_weight_decay=0.01,
            default_optimizer_dtype="bf16",
            default_optimizer_kwargs={},
        ),
        initialize_fresh=True,
    )

    manager = object.__new__(CheckpointManager)
    manager.rank = 0
    manager.local_rank = 0
    manager.model = model
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager.lora_config = {
        "enable_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    manager._adapter_manager = adapter_manager
    return manager


def test_checkpoint_save_admission_failure_and_live_adapter_state_policy(monkeypatch, tmp_path):
    with monkeypatch.context() as failure_patch:
        _assert_checkpoint_save_failure_and_factor_only_admission_policy(failure_patch, tmp_path)
    with monkeypatch.context() as live_state_patch:
        _assert_lora_save_uses_live_dense_and_moe_adapter_state(live_state_patch, tmp_path)


def _assert_checkpoint_save_failure_and_factor_only_admission_policy(monkeypatch, tmp_path):
    with monkeypatch.context() as case_patch:
        _assert_exact_active_lora_rejects_full_snapshot_paths_before_downstream_work(
            case_patch,
            tmp_path / "factor-only",
        )

    manager = _build_checkpoint_manager()
    manager._save_lora_weights = lambda path, model_id, **kwargs: None
    manager._sync_collective_error = lambda error: error

    def _fail_write(*args, **kwargs):
        raise PermissionError("disk full")

    monkeypatch.setattr(manager, "_write_adapter_training_artifacts", _fail_write)
    monkeypatch.setattr(
        checkpoint_manager_module.dist,
        "barrier",
        lambda: (_ for _ in ()).throw(AssertionError("barrier should not run")),
    )

    with pytest.raises(RuntimeError, match="Adapter state save failed: disk full"):
        manager.save_adapter_state("policy-a", path=str(tmp_path / "adapter-save"), save_optimizer=True)

    manager = _build_checkpoint_manager()
    manager._sync_collective_error = lambda error: error

    def _fail_save(*args, **kwargs):
        raise PermissionError("peft write failed")

    monkeypatch.setattr(manager, "_save_lora_weights", _fail_save)

    with pytest.raises(RuntimeError, match="LoRA-only save failed: peft write failed"):
        manager.save_lora_only(str(tmp_path / "adapter-export"), model_id="policy-a")


def _assert_exact_active_lora_rejects_full_snapshot_paths_before_downstream_work(monkeypatch, tmp_path):
    manager = object.__new__(CheckpointManager)
    manager.model = nn.Sequential(_ExactActiveLoRAComponent())

    monkeypatch.setattr(
        checkpoint_manager_module,
        "get_parallel_state",
        lambda: (_ for _ in ()).throw(AssertionError("parallel-state resolution must not run")),
    )
    monkeypatch.setattr(
        checkpoint_manager_module,
        "ckpt_to_state_dict",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("checkpoint conversion must not run")),
    )

    with pytest.raises(RuntimeError, match="factor-only adapter publication"):
        manager.save_full_weights(str(tmp_path / "full"))
    with pytest.raises(RuntimeError, match="factor-only adapter publication"):
        manager.save_weights_for_sampler(str(tmp_path / "checkpoint"), str(tmp_path / "sampler"))
    ordinary_manager = object.__new__(CheckpointManager)
    ordinary_manager.model = nn.Linear(2, 2)
    ordinary_manager._require_factor_only_exact_active_lora("ordinary save")


def _assert_lora_save_uses_live_dense_and_moe_adapter_state(monkeypatch, tmp_path):
    manager = _build_fast_save_manager(tmp_path)

    export_dir = tmp_path / "adapter-export"
    manager._save_lora_weights(str(export_dir), "policy-a")

    adapter_config = json.loads((export_dir / "adapter_config.json").read_text(encoding="utf-8"))

    assert sorted(adapter_config["target_modules"]) == ["o_proj"]

    with monkeypatch.context() as case_patch:
        _assert_moe_lora_save_uses_collective_state_and_resolved_targets(case_patch, tmp_path / "moe")


def _assert_moe_lora_save_uses_collective_state_and_resolved_targets(monkeypatch, tmp_path):
    manager = _build_checkpoint_manager()
    manager.model = nn.Module()
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager.lora_config = {
        "enable_lora": True,
        "moe_hybrid_shared_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
        "lora_target_modules": ["gate_proj", "up_proj", "down_proj"],
    }
    collective_state = {
        "model.layers.0.mlp.experts.gate_proj_lora_A": torch.arange(64, dtype=torch.float32).reshape(1, 8, 8),
        "model.layers.0.mlp.experts.gate_proj_lora_B": torch.arange(128, dtype=torch.float32).reshape(1, 8, 16),
    }
    manager._gather_adapter_lora_params = lambda model_id: collective_state

    captured = {}

    def _capture_save_lora_checkpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(checkpoint_manager_module, "save_lora_checkpoint", _capture_save_lora_checkpoint)

    manager._save_lora_weights(str(tmp_path / "moe-export"), "policy-a")

    exported_state = captured["lora_state_dict"]
    assert tuple(exported_state["model.layers.0.mlp.experts.gate_proj_lora_A"].shape) == (1, 8, 4)
    assert tuple(exported_state["model.layers.0.mlp.experts.gate_proj_lora_B"].shape) == (1, 4, 16)
    torch.testing.assert_close(
        exported_state["model.layers.0.mlp.experts.gate_proj_lora_A"],
        collective_state["model.layers.0.mlp.experts.gate_proj_lora_A"][..., :4],
    )
    torch.testing.assert_close(
        exported_state["model.layers.0.mlp.experts.gate_proj_lora_B"],
        collective_state["model.layers.0.mlp.experts.gate_proj_lora_B"][:, :4, :],
    )
    assert captured["r"] == 4

    manager = _build_checkpoint_manager()

    class _ModelWithStackedMoELoRA:
        def named_parameters(self):
            yield "model.layers.0.mlp.experts.gate_proj_lora_A", nn.Parameter(torch.ones(1, 8, 4))

    manager.model = _ModelWithStackedMoELoRA()
    manager.model_config = {"model_path": "Qwen/Qwen3-8B"}
    manager.lora_config = {
        "enable_lora": True,
        "lora_rank": 4,
        "lora_alpha": 16,
    }
    manager.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    manager.lora_alpha_value = 16
    collective_state = {"model.layers.0.mlp.experts.gate_proj_lora_A": torch.ones(1, 8, 4)}
    manager._gather_adapter_lora_params = lambda model_id: collective_state

    captured = {}

    def _capture_save_lora_checkpoint(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(checkpoint_manager_module, "save_lora_checkpoint", _capture_save_lora_checkpoint)

    manager._save_lora_weights(str(tmp_path / "moe-export"), "policy-a")

    assert captured["lora_state_dict"].keys() == collective_state.keys()
    torch.testing.assert_close(
        captured["lora_state_dict"]["model.layers.0.mlp.experts.gate_proj_lora_A"],
        collective_state["model.layers.0.mlp.experts.gate_proj_lora_A"],
    )
    assert captured["r"] == 4
