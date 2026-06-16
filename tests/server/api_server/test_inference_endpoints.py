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
        assert response.endpoint.worker_port == 30000
        assert "http://inference.example:29999/health" not in calls

    def test_add_inference_endpoint_checks_explicit_worker_port(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(),
            "http://inference.example:31000/health": FakeResponse(),
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
                AddInferenceEndpointRequest(host="inference.example", port=30000, worker_port=31000),
            )
        )

        assert response.success is True
        assert response.endpoint is not None
        assert response.endpoint.worker_port == 31000
        assert "http://inference.example:30000/health" in calls
        assert "http://inference.example:31000/health" in calls

    def test_add_inference_endpoint_auto_sync_uses_detected_tp_size(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(),
            "http://inference.example:30000/server_info": FakeResponse(json_data={"model_path": None, "tp_size": 4}),
        }
        monkeypatch.setattr(
            "xorl.server.api_server.inference_endpoints.httpx.AsyncClient",
            make_async_client(responses, calls),
        )

        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.orchestrator_client = MagicMock()
        captured_endpoints = None

        async def fake_sync_weights_to_endpoints(endpoints, **_kwargs):
            nonlocal captured_endpoints
            captured_endpoints = endpoints
            return {"success": True, "message": "ok", "transfer_time": 0.1, "total_bytes": 123}

        server._sync_weights_to_endpoints = fake_sync_weights_to_endpoints

        response = asyncio.run(
            server.add_inference_endpoint(
                AddInferenceEndpointRequest(
                    host="inference.example",
                    port=30000,
                    world_size=1,
                    sync_weights=True,
                    master_address="train.example",
                ),
            )
        )

        assert response.success is True
        assert response.endpoint is not None
        assert response.endpoint.world_size == 4
        assert captured_endpoints == [{"host": "inference.example", "port": 30000, "world_size": 4}]

    def test_list_inference_endpoints_accepts_v1_models_health_fallback(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(status_code=404),
            "http://inference.example:30000/v1/models": FakeResponse(json_data={"data": []}),
        }
        monkeypatch.setattr(
            "xorl.server.api_server.inference_endpoints.httpx.AsyncClient",
            make_async_client(responses, calls),
        )

        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server.inference_endpoints = [
            InferenceEndpoint(host="inference.example", port=30000, world_size=1),
        ]

        response = asyncio.run(server.list_inference_endpoints())

        assert response.count == 1
        assert response.endpoints[0].host == "inference.example"
        assert "http://inference.example:30000/health" in calls
        assert "http://inference.example:30000/v1/models" in calls

    def test_lora_adapter_management_uses_worker_port(self, monkeypatch):
        calls: list[str] = []

        def fake_post(url: str, **kwargs):
            calls.append(url)
            return FakeResponse(json_data={"success": True})

        monkeypatch.setattr("xorl.server.api_server.inference_endpoints.requests.post", fake_post)

        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server.inference_endpoints = [
            InferenceEndpoint(host="inference.example", port=30000, worker_port=31000, world_size=1),
        ]

        asyncio.run(server._load_lora_on_inference_endpoints("adapter-001", "/tmp/adapter-001"))
        asyncio.run(server._unload_lora_on_inference_endpoints("adapter-001"))

        assert calls == [
            "http://inference.example:31000/load_lora_adapter",
            "http://inference.example:31000/unload_lora_adapter",
        ]

    def test_loaded_adapter_query_uses_worker_port(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:31000/v1/models": FakeResponse(
                json_data={
                    "data": [
                        {"id": "base-model"},
                        {"id": "adapter-001", "parent": "base-model"},
                    ]
                }
            ),
        }
        monkeypatch.setattr(
            "xorl.server.api_server.inference_endpoints.httpx.AsyncClient",
            make_async_client(responses, calls),
        )

        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        endpoint = InferenceEndpoint(host="inference.example", port=30000, worker_port=31000, world_size=1)

        adapters = asyncio.run(server._get_loaded_adapters_from_endpoint(endpoint))

        assert adapters == ["adapter-001"]
        assert calls == ["http://inference.example:31000/v1/models"]

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
                            "timing_breakdown": {"transfer_s": 0.1, "total_handler_s": 0.2},
                            "p2p_rank_summaries": [{"rank": 0, "is_sender": True, "transfer_wall_s": 0.1}],
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
        assert response.timing_breakdown == {"transfer_s": 0.1, "total_handler_s": 0.2}
        assert response.p2p_rank_summaries == [{"rank": 0, "is_sender": True, "transfer_wall_s": 0.1}]

    def test_add_inference_endpoint_auto_sync_uses_configured_sync_method(self, monkeypatch):
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
            sync_inference_method="p2p",
        )
        server._running = True
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
                        }
                    ],
                )
            )
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.add_inference_endpoint(
                AddInferenceEndpointRequest(
                    host="inference.example",
                    port=30000,
                    sync_weights=True,
                    master_address="train.example",
                ),
            )
        )

        assert response.success is True
        assert response.weights_synced is True
        assert captured_request["request"].payload.sync_method == "p2p"
