"""Tests for save/session operation handling in RunnerDispatcher."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from xorl.server.protocol.operations import RegisterSessionData, SaveLoraOnlyData, SaveStateData
from xorl.server.protocol.orchestrator_runner import RunnerDispatchCommand


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "runner_dispatcher.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_runner_dispatcher_session_ops", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RunnerDispatcher = _MODULE.RunnerDispatcher


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeAdapterCoordinator:
    def __init__(self):
        self.auto_load_calls = []
        self.register_session_calls = []

    def auto_load_if_evicted(self, model_id: str, *, allow_fresh_materialization: bool = True):
        self.auto_load_calls.append(
            {
                "model_id": model_id,
                "allow_fresh_materialization": allow_fresh_materialization,
            }
        )
        return False, None

    async def handle_register_session(self, command_dict):
        self.register_session_calls.append(command_dict)
        payload = command_dict["payload"]
        return {"registered": True, "model_id": payload.model_id}


class _FakeTrainer:
    def __init__(self):
        self.adapter_manager = object()
        self.lora_config = {"enable_lora": True}
        self.save_state_calls = []
        self.save_lora_only_calls = []

    def save_state(self, checkpoint_path, save_optimizer=True, model_id=None):
        self.save_state_calls.append((checkpoint_path, save_optimizer, model_id))
        return {"success": True}

    def save_lora_only(self, lora_path, model_id="default"):
        self.save_lora_only_calls.append((lora_path, model_id))
        return {"success": True}


def test_handle_save_state_requires_real_checkpoint_for_nonresident_adapter(tmp_path):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.trainer = _FakeTrainer()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    result = asyncio.run(
        dispatcher._handle_save_state(
            {
                "payload": SaveStateData(
                    checkpoint_path=str(tmp_path / "checkpoint"),
                    save_optimizer=True,
                    model_id="policy-a",
                )
            }
        )
    )

    assert result["checkpoint_path"] == str(tmp_path / "checkpoint")
    assert dispatcher._adapter_coordinator.auto_load_calls == [
        {
            "model_id": "policy-a",
            "allow_fresh_materialization": False,
        }
    ]
    assert dispatcher.trainer.save_state_calls == [(str(tmp_path / "checkpoint"), True, "policy-a")]


def test_handle_save_lora_only_requires_real_checkpoint_for_nonresident_adapter(tmp_path):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.trainer = _FakeTrainer()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    result = asyncio.run(
        dispatcher._handle_save_lora_only(
            {
                "payload": SaveLoraOnlyData(
                    lora_path=str(tmp_path / "adapter"),
                    model_id="policy-b",
                )
            }
        )
    )

    assert result["lora_path"] == str(tmp_path / "adapter")
    assert dispatcher._adapter_coordinator.auto_load_calls == [
        {
            "model_id": "policy-b",
            "allow_fresh_materialization": False,
        }
    ]
    assert dispatcher.trainer.save_lora_only_calls == [(str(tmp_path / "adapter"), "policy-b")]


def test_handle_register_session_delegates_to_adapter_coordinator():
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()
    payload = RegisterSessionData(
        model_id="policy-c",
        session_spec={
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
            "lora_config": {"lora_rank": 8, "lora_alpha": 16},
            "optimizer_config": {"type": "adamw", "learning_rate": 1e-4},
        },
        materialize=True,
    )

    result = asyncio.run(dispatcher._handle_register_session({"payload": payload}))

    assert result == {"registered": True, "model_id": "policy-c"}
    assert dispatcher._adapter_coordinator.register_session_calls == [{"payload": payload}]


def test_handle_request_rank0_fails_register_session_on_cross_rank_worker_error(monkeypatch):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.world_size = 2
    dispatcher.cpu_group = object()
    dispatcher._worker_error = None

    async def _handle_register_session(command_dict):
        return {"registered": True, "model_id": command_dict["payload"].model_id}

    dispatcher._handle_register_session = _handle_register_session
    dispatcher._sync_error_state = lambda: "rank 1: Session registration failed: boom"

    monkeypatch.setattr(_MODULE.dist, "broadcast_object_list", lambda *args, **kwargs: None)

    request = RunnerDispatchCommand.create(
        "register_session",
        RegisterSessionData(
            model_id="policy-c",
            session_spec={"base_model": "Qwen/Qwen3-8B", "is_lora": True},
            materialize=False,
        ),
        request_id="req-register-session",
    )

    response = asyncio.run(dispatcher._handle_request_rank0(request))

    assert response.success is False
    assert response.error == "Cross-rank error: rank 1: Session registration failed: boom"
