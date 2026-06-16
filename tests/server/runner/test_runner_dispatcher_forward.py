"""Tests for forward-path model_id handling in RunnerDispatcher."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from xorl.server.protocol.operations import ModelPassData


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "runner_dispatcher.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_runner_dispatcher", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RunnerDispatcher = _MODULE.RunnerDispatcher


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeAdapterCoordinator:
    def __init__(self):
        self.calls = []

    def auto_load_if_evicted(self, model_id: str):
        self.calls.append(model_id)
        return True, "/tmp/adapter-state"


def test_handle_compute_rank0_scatter_uses_model_id_for_forward(monkeypatch):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    captured = {}

    def fake_select_and_prepare_batches(batches, routed_experts=None, routed_expert_logits=None):
        return batches, routed_experts, routed_expert_logits

    def fake_execute_and_gather(
        my_batches,
        loss_fn,
        loss_fn_params,
        routed_experts,
        cp_enabled,
        parallel_state,
        *,
        with_backward,
        model_id,
        is_rank0,
        routed_expert_logits=None,
    ):
        captured["model_id"] = model_id
        captured["with_backward"] = with_backward
        captured["is_rank0"] = is_rank0
        return {"total_loss": 0.5}

    dispatcher._select_and_prepare_batches = fake_select_and_prepare_batches
    dispatcher._execute_and_gather = fake_execute_and_gather

    monkeypatch.setattr(_MODULE, "get_parallel_state", lambda: type("PS", (), {"cp_enabled": False})())

    payload = ModelPassData(
        batches=[{"model_input": {"input_ids": [1]}, "loss_fn_inputs": {"labels": [1]}}],
        model_id="adapter-42",
    )

    result = asyncio.run(dispatcher._handle_compute_rank0_scatter({"payload": payload}, with_backward=False))

    assert dispatcher._adapter_coordinator.calls == ["adapter-42"]
    assert captured == {
        "model_id": "adapter-42",
        "with_backward": False,
        "is_rank0": True,
    }
    assert result["auto_loaded"] is True
    assert result["auto_load_path"] == "/tmp/adapter-state"
