"""Focused tests for API training operation responses."""

from __future__ import annotations

import types

import pytest

from xorl.server.api_server.api_types import ForwardRequest, OptimStepRequest
from xorl.server.api_server.server import APIServer


pytestmark = [pytest.mark.cpu, pytest.mark.server, pytest.mark.anyio]


class _FakeOrchestratorClient:
    def __init__(self) -> None:
        self.last_request = None

    async def send_request(self, request):
        self.last_request = request
        return request


def _build_wait_for_response():
    async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
        return types.SimpleNamespace(
            outputs=[
                {
                    "grad_norm": 7.5,
                    "learning_rate": 2e-4,
                    "step": 1,
                }
            ]
        )

    return _wait_for_response


def _build_server():
    server = APIServer(
        engine_input_addr="tcp://127.0.0.1:17000",
        engine_output_addr="tcp://127.0.0.1:17001",
    )
    server.orchestrator_client = _FakeOrchestratorClient()
    server._running = True
    server._wait_for_response = types.MethodType(_build_wait_for_response(), server)
    return server


async def test_optim_step_uses_orchestrator_learning_rate_key():
    server = _build_server()

    response = await server.optim_step(OptimStepRequest(model_id="test-session", learning_rate=2e-4, gradient_clip=1.0))

    assert response.metrics["grad_norm"] == pytest.approx(7.5)
    assert response.metrics["learning_rate"] == pytest.approx(2e-4)
    assert server.orchestrator_client.last_request.payload.lr == pytest.approx(2e-4)


async def test_optim_step_maps_legacy_grad_clip_norm_to_orchestrator_payload():
    server = _build_server()

    response = await server.optim_step(
        OptimStepRequest(
            **{
                "session_id": "legacy-session",
                "adam_params": {"learning_rate": 3e-4, "grad_clip_norm": 2.5},
            }
        )
    )

    assert response.metrics["grad_norm"] == pytest.approx(7.5)
    assert server.orchestrator_client.last_request.payload.lr == pytest.approx(3e-4)
    assert server.orchestrator_client.last_request.payload.gradient_clip == pytest.approx(2.5)


async def test_forward_surfaces_auto_load_info():
    server = _build_server()

    async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
        return types.SimpleNamespace(
            outputs=[
                {
                    "loss": 0.25,
                    "valid_tokens": 2,
                    "execution_time": 0.01,
                    "auto_loaded": True,
                    "auto_load_path": "/tmp/evicted/session-a",
                }
            ]
        )

    server._wait_for_response = types.MethodType(_wait_for_response, server)

    response = await server.forward(
        ForwardRequest(
            model_id="session-a",
            forward_input={
                "data": [
                    {
                        "model_input": {"input_ids": [1, 2]},
                        "loss_fn_inputs": {"labels": [1, 2]},
                    }
                ]
            },
        )
    )

    assert response.metrics["loss:mean"] == pytest.approx(0.25)
    assert response.info == {
        "auto_loaded": True,
        "auto_load_path": "/tmp/evicted/session-a",
    }
