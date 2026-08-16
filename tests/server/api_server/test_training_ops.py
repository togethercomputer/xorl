"""Focused tests for API training operation responses."""

from __future__ import annotations

import types

import pytest

from xorl.server.api_server.api_types import DatumInput, ForwardBackwardRequest, ForwardRequest, OptimStepRequest
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
                    "optim_step_time": 0.125,
                    "optim_empty_cache_skipped": True,
                    "glm52_fullparam_publish": {
                        "published": True,
                        "step": 1,
                        "manifest_checksum": "manifest-1",
                    },
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
    assert response.metrics["optim_step_time"] == pytest.approx(0.125)
    assert response.metrics["optim_empty_cache_skipped"] is True
    assert response.metrics["glm52_fullparam_publish"] == {
        "published": True,
        "step": 1,
        "manifest_checksum": "manifest-1",
    }
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


async def test_forward_backward_surfaces_profile_and_executor_timing_metrics():
    server = _build_server()

    async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
        return types.SimpleNamespace(
            outputs=[
                {
                    "loss": 0.25,
                    "valid_tokens": 2,
                    "execution_time": 1.0,
                    "executor_pack_s": 0.1,
                    "executor_backend_s": 0.8,
                    "executor_build_output_s": 0.02,
                    "executor_total_s": 0.92,
                    "forward_compute_time": 0.35,
                    "backward_compute_time": 0.45,
                }
            ]
        )

    server._wait_for_response = types.MethodType(_wait_for_response, server)

    response = await server.forward_backward(
        ForwardBackwardRequest(
            model_id="session-a",
            forward_backward_input=DatumInput(
                data=[
                    {
                        "model_input": {"input_ids": [1, 2]},
                        "loss_fn_inputs": {"labels": [1, 2]},
                    }
                ],
                loss_fn_params={"profile_phase_timings": True},
            ),
        )
    )

    assert response.metrics["executor_backend_s"] == pytest.approx(0.8)
    assert response.metrics["executor_build_output_s"] == pytest.approx(0.02)
    assert response.metrics["executor_pack_s"] == pytest.approx(0.1)
    assert response.metrics["executor_total_s"] == pytest.approx(0.92)
    assert response.metrics["backward_compute_time"] == pytest.approx(0.45)
    assert response.metrics["forward_compute_time"] == pytest.approx(0.35)
