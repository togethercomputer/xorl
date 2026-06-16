"""Tests for load_state delegation in RunnerDispatcher."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from xorl.server.protocol.operations import AdapterStateData, LoadStateData


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "runner_dispatcher.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_runner_dispatcher_load_state", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RunnerDispatcher = _MODULE.RunnerDispatcher


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeAdapterCoordinator:
    def __init__(self):
        self.calls = []

    async def handle_load_adapter_state(self, command_dict):
        self.calls.append(command_dict)
        return {"success": True, "model_id": command_dict["payload"].model_id, "step": 7}


class _FakeTrainerMultiAdapter:
    def __init__(self):
        self.adapter_manager = object()
        self.step = 99
        self.load_state_calls = []

    def load_state(self, checkpoint_path, load_optimizer=True, model_id=None):
        self.load_state_calls.append((checkpoint_path, load_optimizer, model_id))
        return {"success": True}


class _FakeTrainerSingleTenant:
    def __init__(self):
        self.adapter_manager = None
        self.step = 99
        self.load_state_calls = []

    def load_state(self, checkpoint_path, load_optimizer=True, model_id=None):
        self.load_state_calls.append((checkpoint_path, load_optimizer, model_id))
        return {"success": True, "model_id": model_id}


def test_handle_load_state_uses_adapter_coordinator_for_multi_adapter(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.trainer = _FakeTrainerMultiAdapter()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    result = asyncio.run(
        dispatcher._handle_load_state(
            {
                "payload": LoadStateData(
                    checkpoint_path=str(checkpoint_path),
                    load_optimizer=False,
                    model_id="policy-a",
                )
            }
        )
    )

    assert result == {"success": True, "model_id": "policy-a", "step": 7}
    assert dispatcher.trainer.step == 0
    assert dispatcher.trainer.load_state_calls == []
    assert len(dispatcher._adapter_coordinator.calls) == 1
    payload = dispatcher._adapter_coordinator.calls[0]["payload"]
    assert isinstance(payload, AdapterStateData)
    assert payload.model_id == "policy-a"
    assert payload.path == str(checkpoint_path)
    assert payload.load_optimizer is False


def test_handle_load_state_uses_trainer_load_state_without_adapter_manager(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.trainer = _FakeTrainerSingleTenant()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    result = asyncio.run(
        dispatcher._handle_load_state(
            {
                "payload": LoadStateData(
                    checkpoint_path=str(checkpoint_path),
                    load_optimizer=True,
                    model_id="default",
                )
            }
        )
    )

    assert result == {"success": True, "model_id": "default"}
    assert dispatcher.trainer.step == 0
    assert dispatcher.trainer.load_state_calls == [(str(checkpoint_path), True, "default")]
    assert dispatcher._adapter_coordinator.calls == []
