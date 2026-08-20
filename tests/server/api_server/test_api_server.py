"""
Basic tests for APIServer.

Tests focus on server initialization and configuration validation.
Integration tests with Orchestrator are excluded as they require complex infrastructure.
"""

import asyncio
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from xorl.server.api_server.api_types import (
    AdamParams,
    CreateModelRequest,
    CreateSessionRequest,
    OptimStepRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    UntypedAPIFuture,
)
from xorl.server.api_server.endpoints import (
    create_model_endpoint,
    create_session_endpoint,
    save_weights_endpoint,
    session_heartbeat_endpoint,
)
from xorl.server.api_server.server import APIServer, app
from xorl.server.session_spec import build_default_session_spec


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _ImmediateFutureStore:
    def __init__(self) -> None:
        self.last_result = None

    async def create(self, *, model_id, request_type, process_fn, request_data, ttl=None):
        self.last_result = await process_fn(request_data)
        return "future-test-1"


class _FakeOrchestratorClient:
    def __init__(self) -> None:
        self.last_request = None

    async def send_request(self, request):
        self.last_request = request
        return request


def _build_default_session_spec():
    train_config = {
        "optimizer": "adamw",
        "lr": 1e-5,
        "weight_decay": 0.01,
        "optimizer_dtype": "bf16",
        "optimizer_kwargs": {},
    }
    lora_config = {
        "enable_lora": True,
        "lora_rank": 8,
        "max_lora_rank": 16,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "o_proj"],
    }
    default_session_spec = build_default_session_spec(
        base_model="Qwen/Qwen3-8B",
        train_config=train_config,
        lora_config=lora_config,
    )
    return default_session_spec, lora_config


class TestTinkerSessionCompatibility:
    """Test Tinker-compatible session creation and heartbeats."""

    def test_session_publication_creation_and_activity_policy(self, tmp_path, monkeypatch):
        """Returned session IDs should work in follow-up calls that send session_id."""
        app.openapi_schema = None
        schema_paths = app.openapi()["paths"]
        assert "/api/v1/create_session" in schema_paths
        assert "/api/v1/session_heartbeat" in schema_paths

        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17010",
            engine_output_addr="tcp://127.0.0.1:17011",
            output_dir=str(tmp_path),
        )
        seen_model_ids = []

        async def fake_submit_save_weights_async(request):
            seen_model_ids.append(request.model_id)
            return UntypedAPIFuture(request_id="req-1", model_id=request.model_id)

        server.submit_save_weights_async = fake_submit_save_weights_async

        create_response = asyncio.run(create_session_endpoint(CreateSessionRequest(), server=server))
        session_id = create_response.session_id

        assert session_id in server.registered_model_ids
        assert session_id in server.session_last_activity

        save_response = asyncio.run(
            save_weights_endpoint(
                SaveWeightsRequest(session_id=session_id, path="checkpoint-001"),
                server=server,
            )
        )

        assert seen_model_ids == [session_id]
        assert save_response.request_id == "req-1"
        assert save_response.model_id == session_id

        self._assert_session_heartbeat_refreshes_activity(monkeypatch)
        self._assert_create_session_stores_canonical_lora_config()

    def _assert_session_heartbeat_refreshes_activity(self, monkeypatch):
        """Heartbeats should update the activity timestamp for registered sessions."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17012",
            engine_output_addr="tcp://127.0.0.1:17013",
        )

        session_id = "heartbeat-session"
        asyncio.run(create_session_endpoint(CreateSessionRequest(session_id=session_id), server=server))
        initial_activity = server.session_last_activity[session_id]
        monkeypatch.setattr(
            server,
            "_update_session_activity",
            lambda model_id: server.session_last_activity.__setitem__(model_id, initial_activity + 1.0),
        )

        heartbeat_response = asyncio.run(
            session_heartbeat_endpoint(
                SessionHeartbeatRequest(session_id=session_id),
                server=server,
            )
        )

        assert heartbeat_response.session_id == session_id
        assert server.session_last_activity[session_id] == initial_activity + 1.0

    def _assert_create_session_stores_canonical_lora_config(self):
        """Tinker rank/alpha aliases should be canonicalized before server storage."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17022",
            engine_output_addr="tcp://127.0.0.1:17023",
        )

        response = asyncio.run(
            create_session_endpoint(
                CreateSessionRequest(session_id="alias-session", lora_config={"rank": 6, "alpha": 14}),
                server=server,
            )
        )

        assert response.session_id == "alias-session"
        assert server.model_configs["alias-session"]["lora_config"] == {
            "lora_rank": 6,
            "lora_alpha": 14,
        }


class TestTinkerCompatibilityPaths:
    """Exercise HTTP-boundary compatibility paths, not just Pydantic parsing."""

    def _assert_optim_step_uses_legacy_full_session_default(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17004",
            engine_output_addr="tcp://127.0.0.1:17005",
        )
        server.model_configs["full-session"] = {"base_model": "base", "lora_config": {}}
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(outputs=[{"grad_norm": 0.0}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        response = asyncio.run(server.optim_step(OptimStepRequest(model_id="full-session")))

        assert client.last_request.payload.lr == AdamParams().learning_rate
        assert response.metrics["learning_rate"] == AdamParams().learning_rate

    def test_optim_step_legacy_payload_and_learning_rate_fallback_policy(self):
        """Preserve compatibility payloads and the effective learning-rate priority."""
        self._assert_optim_step_supports_tinker_adam_params_payload()
        self._assert_optim_step_uses_registered_session_default_learning_rate()
        self._assert_optim_step_uses_server_train_config_learning_rate()
        self._assert_optim_step_uses_learning_rate_registered_by_create_model()
        self._assert_optim_step_uses_legacy_full_session_default()
        self._assert_optim_step_rejects_missing_learning_rate_without_default()

    def _assert_optim_step_supports_tinker_adam_params_payload(self):
        """Legacy Tinker adam_params should still drive lr, clip, and Adam hyperparameters."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17020",
            engine_output_addr="tcp://127.0.0.1:17021",
        )
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(outputs=[{"grad_norm": 0.5}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        response = asyncio.run(
            server.optim_step(
                OptimStepRequest(
                    **{
                        "session_id": "legacy-session",
                        "adam_params": AdamParams(
                            learning_rate=2e-4,
                            beta1=0.8,
                            beta2=0.88,
                            eps=1e-6,
                            grad_clip_norm=0.75,
                        ),
                    }
                )
            )
        )

        assert client.last_request.operation == "optim_step"
        assert client.last_request.payload.model_id == "legacy-session"
        assert client.last_request.payload.lr == pytest.approx(2e-4)
        assert client.last_request.payload.gradient_clip == pytest.approx(0.75)
        assert client.last_request.payload.beta1 == pytest.approx(0.8)
        assert client.last_request.payload.beta2 == pytest.approx(0.88)
        assert client.last_request.payload.eps == pytest.approx(1e-6)
        assert response.metrics["learning_rate"] == pytest.approx(2e-4)

    def _assert_optim_step_uses_registered_session_default_learning_rate(self):
        """A native request can omit learning_rate when the session has an optimizer default."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17024",
            engine_output_addr="tcp://127.0.0.1:17025",
        )
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client
        server.model_configs["default"] = {
            "base_model": "Qwen/Qwen3-8B",
            "optimizer_config": {"learning_rate": 7e-5},
        }

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(outputs=[{"grad_norm": 0.5}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        response = asyncio.run(server.optim_step(OptimStepRequest(model_id="default")))

        assert client.last_request.payload.lr == pytest.approx(7e-5)
        assert response.metrics["learning_rate"] == pytest.approx(7e-5)

    def _assert_optim_step_uses_server_train_config_learning_rate(self):
        """Full-weight default sessions should inherit the server optimizer LR when request LR is omitted."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17034",
            engine_output_addr="tcp://127.0.0.1:17035",
            train_config={"lr": 6e-5},
            lora_config={"enable_lora": False},
        )
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client
        server.model_configs["default"] = {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": False,
        }

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(outputs=[{"grad_norm": 0.5}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        response = asyncio.run(server.optim_step(OptimStepRequest(model_id="default")))

        assert client.last_request.payload.lr == pytest.approx(6e-5)
        assert response.metrics["learning_rate"] == pytest.approx(6e-5)

    def _assert_optim_step_uses_learning_rate_registered_by_create_model(self):
        """create_model optimizer_config should feed later native optim_step requests."""
        default_session_spec, lora_config = _build_default_session_spec()
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17028",
            engine_output_addr="tcp://127.0.0.1:17029",
            base_model="Qwen/Qwen3-8B",
            default_session_spec=default_session_spec,
            server_lora_config=lora_config,
            max_lora_rank=16,
            skip_initial_checkpoint=True,
        )
        server.future_store = _ImmediateFutureStore()
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client

        async def _wait_for_create_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(
                outputs={"result": {"registered": True, "model_id": response_future.payload.model_id}}
            )

        server._wait_for_response = types.MethodType(_wait_for_create_response, server)

        asyncio.run(
            create_model_endpoint(
                CreateModelRequest(
                    model_id="session-from-create-model",
                    base_model="Qwen/Qwen3-8B",
                    optimizer_config={"learning_rate": 9e-5},
                ),
                server=server,
            )
        )

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(outputs=[{"grad_norm": 0.5}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        response = asyncio.run(server.optim_step(OptimStepRequest(model_id="session-from-create-model")))

        assert client.last_request.payload.lr == pytest.approx(9e-5)
        assert response.metrics["learning_rate"] == pytest.approx(9e-5)

    def _assert_optim_step_rejects_missing_learning_rate_without_default(self):
        """Missing request and session learning rates should fail loudly instead of using a magic number."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17026",
            engine_output_addr="tcp://127.0.0.1:17027",
        )
        server._running = True
        server.orchestrator_client = _FakeOrchestratorClient()

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(server.optim_step(OptimStepRequest(model_id="default")))

        assert exc_info.value.status_code == 400
        assert "no learning_rate" in exc_info.value.detail
