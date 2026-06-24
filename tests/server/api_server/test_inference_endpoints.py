"""Tests for inference endpoint registration."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from xorl.server.api_server.api_types import (
    AddInferenceEndpointRequest,
    InferenceEndpoint,
    InferenceEndpointServerInfo,
    SetSyncQuantizationRequest,
    SyncInferenceWeightsRequest,
)
from xorl.server.api_server.inference_endpoints import InferenceEndpointsMixin
from xorl.server.api_server.server import APIServer
from xorl.server.weight_sync.quantization_config import (
    SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY,
    UnsupportedSyncQuantizationError,
    normalize_sync_quantization_config,
)


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

    def test_add_inference_endpoint_records_fp8_kv_cache_server_info(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(),
            "http://inference.example:30000/server_info": FakeResponse(
                json_data={
                    "model_path": None,
                    "tp_size": 1,
                    "kv_cache_dtype": "fp8_e4m3",
                    "requires_fp8_kv_cache_postprocess": "true",
                    "kv_cache_static_scales": "1",
                    "cache_epoch": 0,
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

        response = asyncio.run(
            server.add_inference_endpoint(
                AddInferenceEndpointRequest(
                    host="inference.example",
                    port=30000,
                    receiver_kv_cache_dtype="fp8_e4m3",
                ),
            )
        )

        assert response.success is True
        assert response.endpoint is not None
        assert response.endpoint.server_info is not None
        assert response.endpoint.server_info.kv_cache_dtype == "fp8_e4m3"
        assert response.endpoint.server_info.fp8_kv_cache_enabled is True
        assert response.endpoint.server_info.fp8_kv_cache_requires_postprocess is True
        assert response.endpoint.server_info.fp8_kv_cache_static_scales is True
        assert response.endpoint.server_info.cache_epoch == 0

    def test_add_inference_endpoint_accepts_cache_version_alias_and_infers_fp8_kv_cache(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(),
            "http://inference.example:30000/server_info": FakeResponse(
                json_data={
                    "model_path": None,
                    "tp_size": 1,
                    "kv_cache_dtype": "float8_e4m3fn",
                    "cache_version": "version-1",
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

        response = asyncio.run(
            server.add_inference_endpoint(
                AddInferenceEndpointRequest(
                    host="inference.example",
                    port=30000,
                    receiver_kv_cache_dtype="fp8",
                ),
            )
        )

        assert response.success is True
        assert response.endpoint is not None
        assert response.endpoint.server_info is not None
        assert response.endpoint.server_info.fp8_kv_cache_enabled is True
        assert response.endpoint.server_info.cache_epoch == "version-1"

    def test_add_inference_endpoint_rejects_non_fp8_kv_cache_when_config_requires_fp8(self, monkeypatch):
        calls: list[str] = []
        responses = {
            "http://inference.example:30000/health": FakeResponse(),
            "http://inference.example:30000/server_info": FakeResponse(
                json_data={
                    "model_path": None,
                    "tp_size": 1,
                    "kv_cache_dtype": "bfloat16",
                    "fp8_kv_cache_enabled": False,
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
            train_config={"receiver_kv_cache_dtype": "fp8"},
        )

        response = asyncio.run(
            server.add_inference_endpoint(
                AddInferenceEndpointRequest(host="inference.example", port=30000),
            )
        )

        assert response.success is False
        assert response.endpoint is None
        assert "receiver_kv_cache_dtype='fp8'" in response.message
        assert "kv_cache_dtype='bfloat16'" in response.message

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
                SyncInferenceWeightsRequest(
                    model_id="policy-a",
                    master_address="train.example",
                    flush_cache=False,
                    sparse_delta_paths=["/shared/delta.packed"],
                ),
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
        assert captured_request["request"].payload.model_id == "policy-a"
        assert captured_request["request"].payload.sparse_delta_paths == ["/shared/delta.packed"]
        assert response.timing_breakdown == {"transfer_s": 0.1, "total_handler_s": 0.2}
        assert response.p2p_rank_summaries == [{"rank": 0, "is_sender": True, "transfer_wall_s": 0.1}]

    def test_sync_inference_weights_pools_filter_selects_matching_endpoints(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(host="train-0.example", port=30060, world_size=2),
            InferenceEndpoint(host="train-1.example", port=30060, world_size=2),
            InferenceEndpoint(host="eval-0.example", port=30070, world_size=2, pool="eval"),
        ]
        captured_request = {}

        async def fake_send_request(engine_request):
            captured_request["request"] = engine_request
            future = asyncio.Future()
            future.set_result(SimpleNamespace(error=None, outputs=[{"success": True, "message": "ok"}]))
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        # pools=["eval"] syncs only the eval-pool endpoint.
        response = asyncio.run(
            server.sync_inference_weights(SyncInferenceWeightsRequest(master_address="train.example", pools=["eval"]))
        )
        assert response.success is True
        assert [ep["host"] for ep in captured_request["request"].payload.endpoints] == ["eval-0.example"]

        # pools=["default"] syncs only the (untagged) training endpoints.
        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(master_address="train.example", pools=["default"])
            )
        )
        assert response.success is True
        assert [ep["host"] for ep in captured_request["request"].payload.endpoints] == [
            "train-0.example",
            "train-1.example",
        ]

        # pools=None (default) keeps the historical sync-everything behavior.
        response = asyncio.run(
            server.sync_inference_weights(SyncInferenceWeightsRequest(master_address="train.example"))
        )
        assert response.success is True
        assert [ep["host"] for ep in captured_request["request"].payload.endpoints] == [
            "train-0.example",
            "train-1.example",
            "eval-0.example",
        ]

    def test_sync_inference_weights_pools_filter_no_match_fails_cleanly(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(host="train-0.example", port=30060, world_size=2),
        ]
        server.orchestrator_client = MagicMock()

        response = asyncio.run(
            server.sync_inference_weights(SyncInferenceWeightsRequest(master_address="train.example", pools=["eval"]))
        )
        assert response.success is False
        assert "pools" in response.message
        server.orchestrator_client.send_request.assert_not_called()

    def test_sync_inference_weights_explicit_null_quantization_disables_default(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server._default_sync_quantization = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        server.inference_endpoints = [InferenceEndpoint(host="inference.example", port=30000, world_size=1)]
        captured_request = {}

        async def fake_send_request(engine_request):
            captured_request["request"] = engine_request
            future = asyncio.Future()
            future.set_result(SimpleNamespace(error=None, outputs=[{"success": True, "message": "ok"}]))
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(
                    master_address="train.example",
                    quantization=None,
                )
            )
        )

        assert response.success is True
        assert captured_request["request"].payload.quantization is None

    def test_sync_inference_weights_auto_flushes_for_fp8_weights_and_fp8_kv_cache(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(
                host="inference.example",
                port=30000,
                world_size=1,
                server_info=InferenceEndpointServerInfo(
                    kv_cache_dtype="fp8",
                    fp8_kv_cache_enabled=True,
                    fp8_kv_cache_requires_postprocess=True,
                    fp8_kv_cache_static_scales=True,
                    cache_epoch="epoch-7",
                ),
            )
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
                            "cache_epoch": None,
                            "endpoint_results": [
                                {
                                    "host": "inference.example",
                                    "port": 30000,
                                    "success": True,
                                    "cache_epoch": "epoch-8",
                                    "fp8_kv_cache_postprocess_ran": True,
                                    "fp8_kv_cache_static_scales_updated": True,
                                }
                            ],
                        }
                    ],
                )
            )
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(
                    master_address="train.example",
                    quantization={"quant_method": "fp8", "weight_block_size": [128, 128]},
                )
            )
        )

        payload = captured_request["request"].payload
        assert payload.flush_cache is True
        assert payload.cache_invalidation_mode == "auto"
        assert payload.fp8_kv_cache_enabled is True
        assert payload.fp8_kv_cache_postprocess_required is True
        assert payload.fp8_kv_cache_static_scales is True
        assert response.flush_cache is True
        assert response.fp8_kv_cache_postprocess_requested is True
        assert response.cache_epoch == "epoch-8"
        assert response.endpoints_synced[0].cache_epoch == "epoch-8"
        assert response.endpoints_synced[0].fp8_kv_cache_postprocess_ran is True
        assert response.endpoints_synced[0].fp8_kv_cache_static_scales_updated is True

    def test_sync_inference_weights_accepts_cache_version_alias_and_dtype_only_fp8_kv_cache(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(
                host="inference.example",
                port=30000,
                world_size=1,
                server_info=InferenceEndpointServerInfo(
                    kv_cache_dtype="fp8_e4m3",
                    fp8_kv_cache_requires_postprocess=True,
                    fp8_kv_cache_static_scales=True,
                    cache_epoch="version-1",
                ),
            )
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
                            "endpoint_results": [
                                {
                                    "host": "inference.example",
                                    "port": 30000,
                                    "success": True,
                                    "cache_version": "version-2",
                                    "fp8_kv_cache_postprocess_ran": True,
                                    "fp8_kv_cache_static_scales_updated": True,
                                }
                            ],
                        }
                    ],
                )
            )
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(
                    master_address="train.example",
                    quantization={"quant_method": "fp8", "weight_block_size": [128, 128]},
                )
            )
        )

        payload = captured_request["request"].payload
        assert payload.flush_cache is True
        assert payload.fp8_kv_cache_enabled is True
        assert payload.fp8_kv_cache_postprocess_required is True
        assert response.cache_epoch == "version-2"
        assert response.endpoints_synced[0].cache_epoch == "version-2"
        assert response.endpoints_synced[0].fp8_kv_cache_postprocess_ran is True
        assert response.endpoints_synced[0].fp8_kv_cache_static_scales_updated is True

    def test_sync_inference_weights_does_not_auto_flush_fp8_kv_cache_for_bf16_weight_sync(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(
                host="inference.example",
                port=30000,
                world_size=1,
                server_info=InferenceEndpointServerInfo(
                    kv_cache_dtype="fp8",
                    fp8_kv_cache_enabled=True,
                    fp8_kv_cache_requires_postprocess=True,
                    fp8_kv_cache_static_scales=True,
                ),
            )
        ]
        captured_request = {}

        async def fake_send_request(engine_request):
            captured_request["request"] = engine_request
            future = asyncio.Future()
            future.set_result(SimpleNamespace(error=None, outputs=[{"success": True, "message": "ok"}]))
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(
                    master_address="train.example",
                    quantization=None,
                )
            )
        )

        payload = captured_request["request"].payload
        assert payload.flush_cache is False
        assert payload.fp8_kv_cache_enabled is True
        assert payload.fp8_kv_cache_postprocess_required is False
        assert response.flush_cache is False
        assert response.fp8_kv_cache_postprocess_requested is False

    def test_sync_inference_weights_cache_mode_none_disables_auto_flush_only(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [
            InferenceEndpoint(
                host="inference.example",
                port=30000,
                world_size=1,
                server_info=InferenceEndpointServerInfo(
                    kv_cache_dtype="fp8",
                    fp8_kv_cache_enabled=True,
                    fp8_kv_cache_requires_postprocess=True,
                ),
            )
        ]
        captured_request = {}

        async def fake_send_request(engine_request):
            captured_request["request"] = engine_request
            future = asyncio.Future()
            future.set_result(SimpleNamespace(error=None, outputs=[{"success": True, "message": "ok"}]))
            return future

        server.orchestrator_client = MagicMock(send_request=AsyncMock(side_effect=fake_send_request))

        response = asyncio.run(
            server.sync_inference_weights(
                SyncInferenceWeightsRequest(
                    master_address="train.example",
                    cache_invalidation_mode="none",
                    quantization={"quant_method": "fp8", "weight_block_size": [128, 128]},
                )
            )
        )

        payload = captured_request["request"].payload
        assert payload.cache_invalidation_mode == "none"
        assert payload.flush_cache is False
        assert payload.fp8_kv_cache_postprocess_required is True
        assert response.flush_cache is False
        assert response.fp8_kv_cache_postprocess_requested is True

    def test_sync_inference_weights_rejects_empty_quantization_config(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
        )
        server._running = True
        server.inference_endpoints = [InferenceEndpoint(host="inference.example", port=30000, world_size=1)]
        server.orchestrator_client = MagicMock(send_request=AsyncMock())

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                server.sync_inference_weights(
                    SyncInferenceWeightsRequest(
                        master_address="train.example",
                        quantization={},
                    )
                )
            )

        assert exc_info.value.status_code == 400
        assert "must contain 'quant_method'" in exc_info.value.detail
        server.orchestrator_client.send_request.assert_not_called()

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


class TestQuantizationConfigNormalization:
    """Test FP8 quantization-config auto-detection + name normalization."""

    def test_strips_language_model_infix(self):
        # Multimodal FP8 checkpoint paths use `model.language_model.layers.*`;
        # the trainer's text-only model exposes `model.layers.*`.
        entries = [
            "lm_head",
            "model.embed_tokens",
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.language_model.layers.0.mlp.gate",
            "model.language_model.layers.39.linear_attn.norm",
        ]
        out = InferenceEndpointsMixin._normalize_modules_to_not_convert(entries)
        assert out == [
            "lm_head",
            "model.embed_tokens",
            "model.layers.0.linear_attn.in_proj_a",
            "model.layers.0.mlp.gate",
            "model.layers.39.linear_attn.norm",
        ]

    def test_drops_vision_entries(self):
        entries = [
            "lm_head",
            "model.visual.blocks.0.attn.proj",
            "model.visual.blocks.0.attn.qkv",
            "visual.blocks.0.attn.proj",
            "model.language_model.layers.0.linear_attn.in_proj_a",
        ]
        out = InferenceEndpointsMixin._normalize_modules_to_not_convert(entries)
        assert "model.visual.blocks.0.attn.proj" not in out
        assert "visual.blocks.0.attn.proj" not in out
        assert "lm_head" in out
        assert "model.layers.0.linear_attn.in_proj_a" in out

    def test_detect_quantization_normalizes_skip_list(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "lm_head",
                    "model.language_model.layers.0.linear_attn.in_proj_a",
                    "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                    "model.visual.blocks.0.attn.proj",
                ],
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        detected = InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path))
        assert detected is not None
        assert detected["quant_method"] == "fp8"
        assert detected["fmt"] == "e4m3"
        assert detected["activation_scheme"] == "dynamic"
        assert detected["weight_block_size"] == [128, 128]
        assert detected["modules_to_not_convert"] == [
            "lm_head",
            "model.layers.0.linear_attn.in_proj_a",
            "model.layers.0.linear_attn.in_proj_b",
        ]

    def test_detect_quantization_normalizes_minimal_fp8_config(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "quantization_config": {"quant_method": "fp8"},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        detected = InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path))

        assert detected == {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }

    def test_detect_quantization_marks_mtp_fp8_config_unsupported(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "text_config": {"mtp_num_hidden_layers": 1},
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "model.language_model.layers.0.linear_attn.in_proj_a",
                    "model.language_model.mtp.layers.0.input_layernorm",
                    "model.visual.blocks.0.attn.proj",
                ],
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        detected = InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path))

        assert detected is not None
        assert detected["activation_scheme"] == "dynamic"
        assert detected["modules_to_not_convert"] == [
            "model.layers.0.linear_attn.in_proj_a",
            "model.mtp.layers.0.input_layernorm",
        ]
        assert (
            "MTP/speculative low-precision sync is not implemented"
            in detected[SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY]
        )
        with pytest.raises(UnsupportedSyncQuantizationError, match="MTP/speculative"):
            normalize_sync_quantization_config(detected)

    def test_detect_quantization_marks_unsupported_fp8_receiver_config(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "static",
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        detected = InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path))

        assert detected is not None
        assert detected["quant_method"] == "fp8"
        assert "activation_scheme" in detected[SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY]
        with pytest.raises(UnsupportedSyncQuantizationError, match="activation_scheme"):
            normalize_sync_quantization_config(detected)

    def test_detect_quantization_marks_null_activation_fp8_receiver_config_unsupported(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": None,
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        detected = InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path))

        assert detected is not None
        assert detected["quant_method"] == "fp8"
        assert "activation_scheme" in detected[SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY]
        with pytest.raises(UnsupportedSyncQuantizationError, match="activation_scheme"):
            normalize_sync_quantization_config(detected)

    def test_detect_quantization_marks_ue8m0_scale_receiver_config_unsupported(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0",
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        detected = InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path))

        assert detected is not None
        assert detected["quant_method"] == "fp8"
        assert detected["scale_fmt"] == "ue8m0"
        assert "UE8M0 scale storage" in detected[SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY]
        with pytest.raises(UnsupportedSyncQuantizationError, match="UE8M0 scale storage"):
            normalize_sync_quantization_config(detected)

    def test_detect_quantization_leaves_bf16_mtp_config_unmarked(self, tmp_path):
        cfg = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "text_config": {"mtp_num_hidden_layers": 1},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        assert InferenceEndpointsMixin._detect_quantization_from_hf_config(str(tmp_path)) is None

    def test_enrich_fills_modules_to_not_convert_from_endpoint(self):
        """User passes {quant_method, fmt, block_size} without skip list →
        handler should fill it in from the receiver's auto-detected config."""
        receiver_skip = ["lm_head", "model.layers.0.linear_attn.in_proj_a"]
        endpoint = InferenceEndpoint(
            host="inf",
            port=30000,
            server_info=InferenceEndpointServerInfo(
                quantization_config={
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                    "modules_to_not_convert": receiver_skip,
                },
            ),
        )
        mixin = InferenceEndpointsMixin()
        mixin.inference_endpoints = [endpoint]  # type: ignore[attr-defined]
        mixin._default_sync_quantization = None  # type: ignore[attr-defined]

        user_cfg = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
        enriched = mixin._enrich_quantization_with_receiver_skip_list(user_cfg)
        assert enriched is not None
        assert enriched["modules_to_not_convert"] == receiver_skip
        # User-supplied skip list must not be overwritten.
        with_skip = {**user_cfg, "modules_to_not_convert": ["my-own-skip"]}
        kept = mixin._enrich_quantization_with_receiver_skip_list(with_skip)
        assert kept["modules_to_not_convert"] == ["my-own-skip"]
        # Non-fp8 configs pass through.
        bf16 = {"quant_method": "bf16"}
        assert mixin._enrich_quantization_with_receiver_skip_list(bf16) == bf16
        # None passes through.
        assert mixin._enrich_quantization_with_receiver_skip_list(None) is None

    def test_enrich_propagates_unsupported_receiver_reason(self):
        """A per-call user FP8 config must not bypass an unsupported receiver marker."""
        endpoint = InferenceEndpoint(
            host="inf",
            port=30000,
            server_info=InferenceEndpointServerInfo(
                quantization_config={
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128],
                    "modules_to_not_convert": ["model.mtp.layers.0.input_layernorm"],
                    SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY: (
                        "MTP/speculative low-precision sync is not implemented."
                    ),
                },
            ),
        )
        mixin = InferenceEndpointsMixin()
        mixin.inference_endpoints = [endpoint]  # type: ignore[attr-defined]
        mixin._default_sync_quantization = None  # type: ignore[attr-defined]

        user_cfg = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["my-own-skip"],
        }
        enriched = mixin._enrich_quantization_with_receiver_skip_list(user_cfg)

        assert enriched is not None
        assert enriched["modules_to_not_convert"] == ["my-own-skip"]
        assert (
            "MTP/speculative low-precision sync is not implemented"
            in enriched[SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY]
        )
        with pytest.raises(UnsupportedSyncQuantizationError, match="MTP/speculative"):
            normalize_sync_quantization_config(enriched)

    def test_set_sync_quantization_rejects_unsupported_methods(self):
        mixin = InferenceEndpointsMixin()

        with pytest.raises(HTTPException) as exc_info:
            mixin.set_sync_quantization(SetSyncQuantizationRequest(quantization={"quant_method": "compressed-tensors"}))

        assert exc_info.value.status_code == 400
        assert "INT4/compressed-tensors updates" in exc_info.value.detail

    def test_set_sync_quantization_normalizes_explicit_bf16_to_noop(self):
        mixin = InferenceEndpointsMixin()

        response = mixin.set_sync_quantization(SetSyncQuantizationRequest(quantization={"quant_method": "bf16"}))

        assert response.quantization is None
        assert "bf16" in response.message
