"""Tests for inference endpoint registration."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xorl.server.api_server.api_types import AddInferenceEndpointRequest, InferenceEndpoint, SyncInferenceWeightsRequest
from xorl.server.api_server.server import APIServer


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class FakeResponse:
    """Minimal fake httpx response for endpoint registration tests."""

    def __init__(self, status_code: int = 200, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_data


def make_async_client(responses: dict[str, FakeResponse], calls: list[str]):
    """Build a fake httpx.AsyncClient that serves pre-baked responses."""

    class FakeAsyncClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str) -> FakeResponse:
            calls.append(url)
            response = responses.get(url)
            if response is None:
                raise RuntimeError(f"Unexpected GET {url}")
            return response

    return FakeAsyncClient


class TestInferenceEndpointRegistration:
    """Test inference endpoint registration behavior."""

    def test_add_inference_endpoint_uses_single_endpoint_port(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(),
            "http://inference.example:30000/server_info": FakeResponse(json_data={"model_path": None, "tp_size": 1}),
        }
        monkeypatch.setattr(
            "xorl.server.api_server.inference_endpoints.httpx.AsyncClient",
            make_async_client(responses, calls),
        )

        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )

        response = asyncio.run(
            server.add_inference_endpoint(
                AddInferenceEndpointRequest(host="inference.example", port=30000),
            )
        )

        assert response.success is True
        assert response.endpoint is not None
        assert response.endpoint.port == 30000
        assert "http://inference.example:29999/health" not in calls

    def test_sync_inference_weights_forwards_single_endpoint(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(host="inference.example", port=30000, world_size=2),
        ]

        captured_request = {}

        async def fake_send_request(engine_request):
            captured_request["request"] = engine_request
            future = asyncio.Future()
            future.set_result(
                SimpleNamespace(
                    error=None,
                    outputs=[
                        {
                            "success": True,
                            "message": "ok",
                            "transfer_time": 0.1,
                            "total_bytes": 123,
                            "num_parameters": 4,
                            "num_buckets": 1,
                            "endpoint_results": [{"host": "inference.example", "port": 30000, "success": True}],
                        }
                    ],
                )
            )
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(master_address="train.example", flush_cache=False),
            )
        )

        assert response.success is True
        assert captured_request["request"].payload.endpoints == [
            {
                "host": "inference.example",
                "port": 30000,
                "world_size": 2,
            }
        ]
