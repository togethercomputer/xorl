"""Tests for LoRA session-registry synchronization in ModelRunner."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import xorl.server.runner.model_runner as model_runner_module
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _session_spec(lr: float) -> dict:
    return {
        "base_model": "Qwen/Qwen3-8B",
        "is_lora": True,
        "lora_config": {"lora_rank": 4, "lora_alpha": 8},
        "optimizer_config": {
            "type": "adamw",
            "learning_rate": lr,
            "weight_decay": 0.01,
            "optimizer_dtype": "bf16",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "optimizer_kwargs": {},
        },
    }


class _FakeAdapterManager:
    def __init__(self, lr: float) -> None:
        self.session_specs = {"policy-a": _session_spec(lr)}
        self.optim_step_calls = []

    def has_adapter(self, model_id: str) -> bool:
        return model_id in self.session_specs

    def get_adapter_session_spec(self, model_id: str) -> dict:
        return deepcopy(self.session_specs[model_id])

    def get_adapter_state(self, model_id: str):
        assert model_id in self.session_specs
        return SimpleNamespace(gradient_ownership_plan=None)

    def get_lr(self, model_id: str) -> float:
        return self.session_specs[model_id]["optimizer_config"]["learning_rate"]

    def optim_step(self, model_id: str, lr: float, clip_value: float) -> float:
        self.optim_step_calls.append((model_id, lr, clip_value))
        self.session_specs[model_id]["optimizer_config"]["learning_rate"] = lr
        return 7.5

    def get_global_step(self, model_id: str) -> int:
        return 3


class _FakeCheckpointManager:
    def __init__(self) -> None:
        self.global_step = 11
        self.global_forward_backward_step = 13
        self.load_calls = []

    def load_adapter_state(self, model_id, path=None, load_optimizer=True, lr=None):
        self.load_calls.append((model_id, path, load_optimizer, lr))
        return {"success": True, "path": path, "model_id": model_id}


class _KillSessionAdapterManager:
    def __init__(self, has_adapter: bool) -> None:
        self._has_adapter = has_adapter
        self.removed = []

    def has_adapter(self, model_id: str) -> bool:
        return self._has_adapter

    def remove_adapter(self, model_id: str) -> None:
        self.removed.append(model_id)


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([2.0]))


class _FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.1}]
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def zero_grad(self, set_to_none=True) -> None:
        self.zero_grad_calls += 1


def _build_runner() -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.is_sleeping = False
    runner.lora_config = {"enable_lora": True, "merge_lora_interval": 0}
    runner.train_config = {}
    runner._accumulated_valid_tokens = {"policy-a": 11}
    runner._lora_session_specs = {"policy-a": _session_spec(0.05)}
    runner.global_step = 0
    runner.global_forward_backward_step = 0
    return runner


def test_lora_session_registry_syncs_after_optimizer_checkpoint_load_and_kill(monkeypatch, tmp_path):
    runner = _build_runner()
    runner._adapter_manager = _FakeAdapterManager(lr=0.05)

    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)

    result = ModelRunner.optim_step(runner, gradient_clip=1.0, lr=0.25, model_id="policy-a")

    assert result["lr"] == pytest.approx(0.25)
    assert runner._adapter_manager.optim_step_calls == [("policy-a", 0.25, 1.0)]
    assert runner._lora_session_specs["policy-a"]["optimizer_config"]["learning_rate"] == pytest.approx(0.25)

    runner = _build_runner()
    runner._adapter_manager = _FakeAdapterManager(lr=0.25)
    runner._checkpoint_mgr = _FakeCheckpointManager()
    compiled_model_ids = []
    runner._compile_registered_adapter_gradient_ownership = compiled_model_ids.append

    result = ModelRunner.load_adapter_state(
        runner,
        "policy-a",
        path="/tmp/checkpoint",
        load_optimizer=False,
        lr=None,
    )

    assert result == {
        "success": True,
        "path": "/tmp/checkpoint",
        "model_id": "policy-a",
    }
    assert runner._checkpoint_mgr.load_calls == [("policy-a", "/tmp/checkpoint", False, None)]
    assert runner.global_step == 11
    assert runner.global_forward_backward_step == 13
    assert runner._lora_session_specs["policy-a"]["optimizer_config"]["learning_rate"] == pytest.approx(0.25)
    assert compiled_model_ids == ["policy-a"]

    _assert_kill_session_nonresident_lora_checkpoint_lifecycle(tmp_path)


def _assert_kill_session_nonresident_lora_checkpoint_lifecycle(tmp_path: Path) -> None:
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.lora_config = {"enable_lora": True}
    runner.train_config = {"output_dir": str(tmp_path)}
    runner._adapter_manager = _KillSessionAdapterManager(has_adapter=False)
    runner._lora_session_specs = {
        "policy-a": {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
        }
    }
    runner._accumulated_valid_tokens = {"policy-a": 17}

    with pytest.raises(FileNotFoundError, match="no evicted checkpoint exists"):
        runner.kill_session("policy-a", save_checkpoint=True)

    assert "policy-a" in runner._lora_session_specs
    assert runner._accumulated_valid_tokens["policy-a"] == 17
    assert runner._adapter_manager.removed == []

    evicted_path = tmp_path / "adapters" / "evicted" / "policy-a"
    evicted_path.mkdir(parents=True)
    (evicted_path / "metadata.json").write_text('{"saved": true}', encoding="utf-8")

    result = runner.kill_session("policy-a", save_checkpoint=True)
    promoted_path = tmp_path / "weights" / "policy-a" / "session_policy-a_final"

    assert result == {
        "success": True,
        "message": "LoRA session 'policy-a' killed successfully.",
        "checkpoint_path": str(promoted_path),
    }
    assert (promoted_path / "metadata.json").read_text(encoding="utf-8") == '{"saved": true}'
    assert "policy-a" not in runner._lora_session_specs
    assert "policy-a" not in runner._accumulated_valid_tokens
    assert runner._adapter_manager.removed == []

    with pytest.raises(ValueError, match="model_id"):
        runner.kill_session("../../outside", save_checkpoint=True)


def test_optim_step_preserves_distsignsgd_scaling_clip_and_cache_policy(monkeypatch):
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.is_sleeping = False
    runner._adapter_manager = None
    runner._use_distsignsgd = True
    runner._accumulated_valid_tokens = {"default": 100}
    runner._accumulated_active_microbatches = {"default": 2}
    runner._accumulated_active_voter_total = {"default": 4}
    runner.train_config = {"max_grad_norm": 1.0}
    runner.lora_config = {"enable_lora": False, "merge_lora_interval": 0}
    runner.model = _TinyModule()
    runner.model.param.grad = torch.tensor([4.0])
    runner.optimizer = _FakeOptimizer()
    runner.pp_enabled = False
    runner.global_step = 0

    captured = {}

    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: type("ParallelState", (), {"fsdp_group": None, "pp_group": None})(),
    )
    monkeypatch.setattr(
        model_runner_module,
        "clip_gradients",
        lambda model, clip_value, pp_enabled=False, pp_group=None: (
            captured.update({"clip_value": clip_value, "grad": model.param.grad.item()}) or 7.0
        ),
    )
    monkeypatch.setattr(model_runner_module, "all_reduce", lambda value, group=None: value)
    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)
    monkeypatch.setattr(model_runner_module, "_maybe_merge_lora_util", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_runner_module.torch.cuda, "empty_cache", lambda: None)

    result = ModelRunner.optim_step(runner, model_id="default")

    assert captured["clip_value"] == float("inf")
    assert captured["grad"] == pytest.approx(1.0)
    assert runner.optimizer.step_calls == 1
    assert runner.optimizer.zero_grad_calls == 1
    assert result["step"] == 1
    assert result["grad_norm"] == pytest.approx(7.0)
    assert result["optim_empty_cache_skipped"] is False

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.is_sleeping = False
    runner._adapter_manager = None
    runner._use_distsignsgd = False
    runner._accumulated_valid_tokens = {"default": 100}
    runner._accumulated_active_microbatches = {"default": 2}
    runner._accumulated_active_voter_total = {"default": 4}
    runner.train_config = {
        "max_grad_norm": 1.0,
        "skip_empty_cache_after_optim_step": True,
    }
    runner.lora_config = {"enable_lora": False, "merge_lora_interval": 0}
    runner.model = _TinyModule()
    runner.model.param.grad = torch.tensor([4.0])
    runner.optimizer = _FakeOptimizer()
    runner.pp_enabled = False
    runner.global_step = 0

    empty_cache_calls = []

    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: type("ParallelState", (), {"fsdp_group": None, "pp_group": None})(),
    )
    monkeypatch.setattr(model_runner_module, "clip_gradients", lambda *args, **kwargs: 7.0)
    monkeypatch.setattr(model_runner_module, "all_reduce", lambda value, group=None: value)
    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)
    monkeypatch.setattr(model_runner_module, "_maybe_merge_lora_util", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_runner_module.torch.cuda, "empty_cache", lambda: empty_cache_calls.append("empty_cache"))

    result = ModelRunner.optim_step(runner, model_id="default")

    assert empty_cache_calls == []
    assert result["optim_empty_cache_skipped"] is True
