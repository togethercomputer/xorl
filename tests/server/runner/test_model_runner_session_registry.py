"""Tests for LoRA session-registry synchronization in ModelRunner."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import torch


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "model_runner.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_model_runner_session_registry", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ModelRunner = _MODULE.ModelRunner


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

    def get_lr(self, model_id: str) -> float:
        return self.session_specs[model_id]["optimizer_config"]["learning_rate"]

    def optim_step(self, model_id: str, lr: float, clip_value: float, *, accumulated_valid_tokens: int = 0) -> float:
        self.optim_step_calls.append((model_id, lr, clip_value, accumulated_valid_tokens))
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


def test_optim_step_syncs_registered_lora_session_spec(monkeypatch):
    runner = _build_runner()
    runner._adapter_manager = _FakeAdapterManager(lr=0.05)

    monkeypatch.setattr(_MODULE, "synchronize", lambda: None)

    result = ModelRunner.optim_step(runner, gradient_clip=1.0, lr=0.25, model_id="policy-a")

    assert result["lr"] == pytest.approx(0.25)
    assert runner._adapter_manager.optim_step_calls == [("policy-a", 0.25, 1.0, 11)]
    assert runner._lora_session_specs["policy-a"]["optimizer_config"]["learning_rate"] == pytest.approx(0.25)


def test_load_adapter_state_syncs_registered_lora_session_spec():
    runner = _build_runner()
    runner._adapter_manager = _FakeAdapterManager(lr=0.25)
    runner._checkpoint_mgr = _FakeCheckpointManager()

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


def test_optim_step_preserves_distsignsgd_scaling_and_clip(monkeypatch):
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
        _MODULE,
        "get_parallel_state",
        lambda: type("ParallelState", (), {"fsdp_group": None, "pp_group": None})(),
    )
    monkeypatch.setattr(
        _MODULE,
        "clip_gradients",
        lambda model, clip_value, pp_enabled=False, pp_group=None: captured.update(
            {"clip_value": clip_value, "grad": model.param.grad.item()}
        )
        or 7.0,
    )
    monkeypatch.setattr(_MODULE, "all_reduce", lambda value, group=None: value)
    monkeypatch.setattr(_MODULE, "synchronize", lambda: None)
    monkeypatch.setattr(_MODULE, "_maybe_merge_lora_util", lambda *args, **kwargs: None)
    monkeypatch.setattr(_MODULE.torch.cuda, "empty_cache", lambda: None)

    result = ModelRunner.optim_step(runner, model_id="default")

    assert captured["clip_value"] == float("inf")
    assert captured["grad"] == pytest.approx(1.0)
    assert runner.optimizer.step_calls == 1
    assert runner.optimizer.zero_grad_calls == 1
    assert result["step"] == 1
    assert result["grad_norm"] == pytest.approx(7.0)
