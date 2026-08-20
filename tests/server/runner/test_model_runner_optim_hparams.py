"""Tests for optimizer betas/eps wiring in server mode.

Covers the silent-hyperparameter bug found in the marin #6279 fidelity audit:
client drivers sent ``AdamParams(beta2=0.999, ...)`` per optim_step, but the
dispatcher dropped betas/eps and ``build_optimizer``'s defaults
(betas=(0.9, 0.95)) bound instead.

Three lanes:
- init path: server config yaml ``adam_betas``/``adam_eps`` reach build_optimizer
- optim_step path: payload beta1/beta2/eps reach the optimizer param groups
- backward compat: omitted values leave the optimizer defaults intact
"""

import asyncio

import pytest
import torch

import xorl.server.runner.model_runner as model_runner_module
from xorl.server.protocol.operations import OptimStepData
from xorl.server.runner.model_runner import ModelRunner
from xorl.server.runner.runner_dispatcher import RunnerDispatcher
from xorl.server.server_arguments import ServerArguments


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


# ---------------------------------------------------------------------------
# Init path: yaml adam_betas / adam_eps -> train_config -> build_optimizer
# ---------------------------------------------------------------------------


def test_optimizer_hyperparameter_lifecycle(monkeypatch):
    """Server arguments feed the initializer without losing Adam policy fields."""
    server_args = ServerArguments(model_path="x", adam_betas=[0.9, 0.999], adam_eps=1e-6)
    train_config = server_args.to_config_dict()["train"]
    assert train_config["adam_betas"] == [0.9, 0.999]
    assert train_config["adam_eps"] == pytest.approx(1e-6)

    defaults = ServerArguments(model_path="x").to_config_dict()["train"]
    assert defaults["adam_betas"] is None
    assert defaults["adam_eps"] is None

    explicit = _init_runner(
        {
            "optimizer": "adamw",
            "lr": 2e-5,
            "weight_decay": 0.01,
            "adam_betas": train_config["adam_betas"],
            "adam_eps": train_config["adam_eps"],
        }
    )
    ModelRunner._initialize_optimizer(explicit)
    groups_with_betas = [g for g in explicit.optimizer.param_groups if "betas" in g]
    assert groups_with_betas, "adamw param groups must carry betas"
    for group in groups_with_betas:
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == pytest.approx(1e-6)

    default_runner = _init_runner({"optimizer": "adamw", "lr": 2e-5, "weight_decay": 0.01})
    ModelRunner._initialize_optimizer(default_runner)
    for group in default_runner.optimizer.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == pytest.approx(1e-8)

    malformed = _init_runner({"optimizer": "adamw", "lr": 2e-5, "adam_betas": [0.9]})
    with pytest.raises(ValueError, match="adam_betas"):
        ModelRunner._initialize_optimizer(malformed)

    with monkeypatch.context() as step_patch:
        _assert_optim_step_applies_full_partial_and_omitted_adam_overrides(step_patch)
    _assert_handle_optim_step_forwards_explicit_and_omitted_adam_hparams()


def _init_runner(train_config: dict) -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.train_config = train_config
    runner.lora_config = {"enable_lora": False}
    runner.pp_enabled = False
    runner.model = _TinyModule()
    runner.get_optimizer_pre_hook = None
    return runner


# ---------------------------------------------------------------------------
# optim_step path: payload beta1/beta2/eps -> optimizer param groups
# ---------------------------------------------------------------------------


def _step_runner(monkeypatch, optimizer: torch.optim.Optimizer) -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.is_sleeping = False
    runner._adapter_manager = None
    runner._use_distsignsgd = False
    runner._accumulated_valid_tokens = {}
    runner.train_config = {"max_grad_norm": 1.0}
    runner.lora_config = {"enable_lora": False, "merge_lora_interval": 0}
    runner.model = _TinyModule()
    runner.optimizer = optimizer
    runner.pp_enabled = False
    runner.global_step = 0

    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: type("ParallelState", (), {"fsdp_group": None, "pp_group": None})(),
    )
    monkeypatch.setattr(
        model_runner_module, "clip_gradients", lambda model, clip_value, pp_enabled=False, pp_group=None: 0.5
    )
    monkeypatch.setattr(model_runner_module, "all_reduce", lambda value, group=None: value)
    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)
    monkeypatch.setattr(model_runner_module, "_maybe_merge_lora_util", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_runner_module.torch.cuda, "empty_cache", lambda: None)
    return runner


def _assert_optim_step_applies_full_partial_and_omitted_adam_overrides(monkeypatch):
    model = _TinyModule()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    runner = _step_runner(monkeypatch, optimizer)

    result = ModelRunner.optim_step(
        runner, gradient_clip=1.0, lr=3e-5, beta1=0.9, beta2=0.999, eps=1e-6, model_id="default"
    )

    assert result["lr"] == pytest.approx(3e-5)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(3e-5)
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == pytest.approx(1e-6)

    default_model = _TinyModule()
    default_optimizer = torch.optim.AdamW(default_model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    default_runner = _step_runner(monkeypatch, default_optimizer)
    ModelRunner.optim_step(default_runner, gradient_clip=1.0, lr=3e-5, model_id="default")
    for group in default_optimizer.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == pytest.approx(1e-8)

    partial_model = _TinyModule()
    partial_optimizer = torch.optim.AdamW(partial_model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    partial_runner = _step_runner(monkeypatch, partial_optimizer)
    ModelRunner.optim_step(partial_runner, gradient_clip=1.0, lr=3e-5, beta2=0.999, model_id="default")
    for group in partial_optimizer.param_groups:
        assert group["betas"] == (0.9, 0.999)

    sgd_model = _TinyModule()
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=1e-5, momentum=0.9)
    sgd_runner = _step_runner(monkeypatch, sgd_optimizer)
    ModelRunner.optim_step(sgd_runner, gradient_clip=1.0, lr=3e-5, beta1=0.9, beta2=0.999, eps=1e-6, model_id="default")
    for group in sgd_optimizer.param_groups:
        assert "betas" not in group
        assert group["momentum"] == pytest.approx(0.9)
        assert group["lr"] == pytest.approx(3e-5)

    _assert_optim_step_multi_adapter_payload_betas_reach_adapter_optimizer(monkeypatch)


class _FakeAdapterState:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer


class _FakeAdapterManager:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._state = _FakeAdapterState(optimizer)
        self.optim_step_calls = []

    def has_adapter(self, model_id):
        return False

    def get_lr(self, model_id):
        return 1e-5

    def get_adapter_state(self, model_id):
        return self._state

    def optim_step(self, model_id, lr, clip_value, *, accumulated_valid_tokens=0):
        self.optim_step_calls.append((model_id, lr, clip_value, accumulated_valid_tokens))
        return 0.5

    def get_global_step(self, model_id):
        return 1


def _assert_optim_step_multi_adapter_payload_betas_reach_adapter_optimizer(monkeypatch):
    model = _TinyModule()
    adapter_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-8)
    runner = _step_runner(monkeypatch, optimizer=None)
    runner._adapter_manager = _FakeAdapterManager(adapter_optimizer)
    runner._lora_session_specs = {}

    ModelRunner.optim_step(runner, gradient_clip=1.0, lr=3e-5, beta1=0.9, beta2=0.999, eps=1e-6, model_id="policy-a")

    assert runner._adapter_manager.optim_step_calls == [("policy-a", 3e-5, 1.0, 0)]
    for group in adapter_optimizer.param_groups:
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# Dispatcher: OptimStepData payload betas/eps are forwarded to the trainer
# ---------------------------------------------------------------------------


class _FakeAdapterCoordinator:
    def auto_load_if_evicted(self, model_id):
        return False, None


class _FakeTrainer:
    def __init__(self):
        self.optim_step_calls = []

    def optim_step(self, **kwargs):
        self.optim_step_calls.append(kwargs)
        return {"step": 1, "grad_norm": 0.5, "lr": kwargs.get("lr")}


def _optim_dispatcher() -> RunnerDispatcher:
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.world_size = 1
    dispatcher.trainer = _FakeTrainer()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()
    return dispatcher


def _assert_handle_optim_step_forwards_explicit_and_omitted_adam_hparams():
    dispatcher = _optim_dispatcher()
    payload = OptimStepData(lr=2e-5, gradient_clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8, model_id="default")

    asyncio.run(dispatcher._handle_optim_step({"payload": payload}))

    assert dispatcher.trainer.optim_step_calls == [
        {
            "gradient_clip": 1.0,
            "lr": 2e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "model_id": "default",
            "sparse_delta_capture": None,
        }
    ]

    dispatcher = _optim_dispatcher()
    payload = OptimStepData(lr=2e-5, gradient_clip=1.0, model_id="default")

    asyncio.run(dispatcher._handle_optim_step({"payload": payload}))

    (call,) = dispatcher.trainer.optim_step_calls
    assert call["beta1"] is None
    assert call["beta2"] is None
    assert call["eps"] is None
