"""Tests for load_state delegation in RunnerDispatcher."""

import asyncio

import pytest

from xorl.server.protocol.operations import AdapterStateData, LoadStateData
from xorl.server.protocol.orchestrator_runner import RunnerDispatchCommand
from xorl.server.runner.runner_dispatcher import RunnerDispatcher


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


def _assert_load_state_preparation_preserves_errors_and_enforces_output_root(tmp_path, monkeypatch):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0

    async def _prepare_load_state_command(request):
        raise ValueError("checkpoint path rejected")

    dispatcher._prepare_load_state_command = _prepare_load_state_command
    request = RunnerDispatchCommand.create(
        "load_state",
        LoadStateData(
            checkpoint_path="/outside/checkpoint",
            load_optimizer=False,
            model_id="policy-a",
        ),
        request_id="req-load-state",
    )

    response = asyncio.run(dispatcher._handle_request_rank0(request))

    assert response.success is False
    assert response.error == "checkpoint path rejected"

    output_dir = tmp_path / "server-output"
    checkpoint_path = output_dir / "weights" / "policy-a"
    checkpoint_path.mkdir(parents=True)
    unrelated_root = tmp_path / "unrelated-root"
    unrelated_root.mkdir()
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(unrelated_root))

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.output_dir = str(output_dir)
    request = RunnerDispatchCommand.create(
        "load_state",
        LoadStateData(
            checkpoint_path=str(checkpoint_path),
            load_optimizer=False,
            model_id="policy-a",
        ),
        request_id="req-load-state",
    )

    command = asyncio.run(dispatcher._prepare_load_state_command(request))

    assert command["payload"].checkpoint_path == str(checkpoint_path)
    with pytest.raises(ValueError, match="escapes configured root"):
        asyncio.run(
            dispatcher._prepare_load_state_command(
                RunnerDispatchCommand.create(
                    "load_state",
                    LoadStateData(
                        checkpoint_path=str(unrelated_root),
                        load_optimizer=False,
                        model_id="policy-a",
                    ),
                    request_id="req-load-state-outside",
                )
            )
        )


def test_load_state_validation_and_routing_lifecycle(tmp_path, monkeypatch):
    _assert_load_state_preparation_preserves_errors_and_enforces_output_root(tmp_path, monkeypatch)

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir(exist_ok=True)
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(tmp_path))

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

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir(exist_ok=True)
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(tmp_path))

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
