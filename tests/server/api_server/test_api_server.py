"""
Basic tests for APIServer.

Tests focus on server initialization and configuration validation.
Integration tests with Orchestrator are excluded as they require complex infrastructure.
"""

import asyncio
import time
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from xorl.server.api_server.api_types import (
    AdamParams,
    CreateModelRequest,
    CreateSessionRequest,
    Datum,
    DatumInput,
    ForwardBackwardRequest,
    LoadWeightsRequest,
    OptimStepRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    UntypedAPIFuture,
    WeightsInfoRequest,
)
from xorl.server.api_server.endpoints import (
    create_model_endpoint,
    create_session_endpoint,
    save_weights_endpoint,
    session_heartbeat_endpoint,
    weights_info_endpoint,
)
from xorl.server.api_server.server import APIServer, app
from xorl.server.protocol.operations import RegisterSessionData
from xorl.server.session_spec import build_default_session_spec, write_session_spec


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


class TestAPIServerConfiguration:
    """Test APIServer configuration and initialization."""

    def test_server_initialization(self):
        """Test server initialization with defaults and custom parameters."""
        # Defaults
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
        )
        assert server.engine_input_addr == "tcp://127.0.0.1:17000"
        assert server.engine_output_addr == "tcp://127.0.0.1:17001"
        assert server.default_timeout == 120.0
        assert server._running is False
        assert server.orchestrator_client is None

        # Custom timeout
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
            default_timeout=60.0,
        )
        assert server.default_timeout == 60.0

        # Different ports
        server1 = APIServer(engine_input_addr="tcp://127.0.0.1:5000", engine_output_addr="tcp://127.0.0.1:5001")
        server2 = APIServer(engine_input_addr="tcp://127.0.0.1:6000", engine_output_addr="tcp://127.0.0.1:6001")
        assert server1.engine_input_addr != server2.engine_input_addr


class TestAPIRequestCreationAndSerialization:
    """Test API request creation, defaults, and serialization."""

    def test_request_creation_and_serialization(self):
        """Test creating and serializing all request types."""
        # ForwardBackwardRequest
        request = ForwardBackwardRequest(
            model_id="test-model",
            forward_backward_input=DatumInput(
                data=[Datum(model_input={"input_ids": [1, 2, 3, 4]}, loss_fn_inputs={"labels": [2, 3, 4, 5]})],
                loss_fn="causallm_loss",
            ),
        )
        assert request.model_id == "test-model"
        assert len(request.forward_backward_input.data) == 1

        # Multiple samples with default model_id
        request = ForwardBackwardRequest(
            forward_backward_input=DatumInput(
                data=[
                    Datum(model_input={"input_ids": [1, 2]}, loss_fn_inputs={"labels": [2, 3]}),
                    Datum(model_input={"input_ids": [3, 4]}, loss_fn_inputs={"labels": [4, 5]}),
                ]
            ),
        )
        assert len(request.forward_backward_input.data) == 2
        assert request.model_id == "default"

        # Serialization
        data = request.model_dump()
        assert "model_id" in data and "forward_backward_input" in data

        # OptimStepRequest
        request = OptimStepRequest(adam_params=AdamParams(learning_rate=1e-4), gradient_clip=1.0)
        assert request.learning_rate == 1e-4
        assert request.adam_params is not None and request.adam_params.learning_rate == 1e-4
        data = request.model_dump()
        assert data["learning_rate"] == 1e-4
        assert data["adam_params"]["learning_rate"] == 1e-4

        # Defaults
        request = OptimStepRequest()
        assert request.learning_rate is None
        assert request.adam_params is None
        assert request.gradient_clip is None

        # SaveWeightsRequest
        request = SaveWeightsRequest(path="/tmp/checkpoint")
        assert request.path == "/tmp/checkpoint"
        data = request.model_dump()
        assert data["path"] == "/tmp/checkpoint"

        # LoadWeightsRequest
        request = LoadWeightsRequest(path="/tmp/checkpoint", optimizer=True)
        assert request.optimizer is True

    def test_optim_step_learning_rate_resolution(self):
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17004",
            engine_output_addr="tcp://127.0.0.1:17005",
        )
        server.model_configs["full-session"] = {"base_model": "base", "lora_config": {}}
        server.model_configs["lora-session"] = {
            "base_model": "base",
            "lora_config": {"lora_rank": 8},
            "optimizer_config": {"learning_rate": 3e-5},
        }

        assert server._optim_step_learning_rate(OptimStepRequest(model_id="full-session")) == AdamParams().learning_rate
        assert server._optim_step_learning_rate(OptimStepRequest(model_id="lora-session")) == 3e-5
        assert (
            server._optim_step_learning_rate(
                OptimStepRequest(model_id="lora-session", adam_params=AdamParams(learning_rate=2e-4))
            )
            == 2e-4
        )
        assert server._optim_step_learning_rate(OptimStepRequest(model_id="lora-session", learning_rate=7e-5)) == 7e-5

        class FakeClient:
            request = None

            async def send_request(self, request):
                self.request = request
                return object()

        async def fake_wait_for_response(response_future, request_id, timeout, message):
            class Output:
                outputs = [{"grad_norm": 0.0}]

            return Output()

        fake_client = FakeClient()
        server._running = True
        server.orchestrator_client = fake_client
        server._wait_for_response = fake_wait_for_response

        response = asyncio.run(server.optim_step(OptimStepRequest(model_id="full-session")))

        assert fake_client.request.payload.lr == AdamParams().learning_rate
        assert response.metrics["learning_rate"] == AdamParams().learning_rate


class TestTinkerSessionCompatibility:
    """Test Tinker-compatible session creation and heartbeats."""

    def test_tinker_session_endpoints_are_public_in_openapi(self):
        """The repaired Tinker session endpoints should stay in the public schema."""
        app.openapi_schema = None
        schema_paths = app.openapi()["paths"]

        assert "/api/v1/create_session" in schema_paths
        assert "/api/v1/session_heartbeat" in schema_paths

    def test_create_session_registers_usable_session_id(self, tmp_path):
        """Returned session IDs should work in follow-up calls that send session_id."""
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

    def test_session_heartbeat_refreshes_activity(self):
        """Heartbeats should update the activity timestamp for registered sessions."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17012",
            engine_output_addr="tcp://127.0.0.1:17013",
        )

        session_id = "heartbeat-session"
        asyncio.run(create_session_endpoint(CreateSessionRequest(session_id=session_id), server=server))
        initial_activity = server.session_last_activity[session_id]

        time.sleep(0.01)

        heartbeat_response = asyncio.run(
            session_heartbeat_endpoint(
                SessionHeartbeatRequest(session_id=session_id),
                server=server,
            )
        )

        assert heartbeat_response.session_id == session_id
        assert server.session_last_activity[session_id] > initial_activity

    def test_create_session_stores_canonical_lora_config(self):
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

    def test_weights_info_accepts_and_serializes_legacy_lora_rank(self, tmp_path):
        """weights_info should keep Tinker's flat lora_rank response field."""
        default_session_spec, _ = _build_default_session_spec()
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17014",
            engine_output_addr="tcp://127.0.0.1:17015",
            output_dir=str(tmp_path),
            base_model="Qwen/Qwen3-8B",
        )
        checkpoint_path = tmp_path / "weights" / "session-a" / "checkpoint-001"
        write_session_spec(str(checkpoint_path), default_session_spec)

        response = asyncio.run(
            weights_info_endpoint(
                WeightsInfoRequest(xorl_path="xorl://session-a/weights/checkpoint-001"),
                server=server,
            )
        )

        assert response.base_model == "Qwen/Qwen3-8B"
        assert response.is_lora is True
        assert response.lora_rank == 8
        assert "lora_rank" in response.model_dump()

    def test_create_model_stores_dict_lora_config_for_weights_info(self, tmp_path):
        """create_model should not store a typed LoRAConfigRequest where weights_info expects a dict."""
        default_session_spec, lora_config = _build_default_session_spec()
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17016",
            engine_output_addr="tcp://127.0.0.1:17017",
            output_dir=str(tmp_path),
            base_model="Qwen/Qwen3-8B",
            default_session_spec=default_session_spec,
            server_lora_config=lora_config,
            max_lora_rank=16,
            skip_initial_checkpoint=True,
        )
        server.future_store = _ImmediateFutureStore()
        server.orchestrator_client = _FakeOrchestratorClient()
        server._running = True

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(
                outputs={"result": {"registered": True, "model_id": response_future.payload.model_id}}
            )

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        create_response = asyncio.run(
            create_model_endpoint(
                CreateModelRequest(
                    model_id="session-a",
                    base_model="Qwen/Qwen3-8B",
                    lora_config={"rank": 8, "alpha": 16},
                    optimizer_config={"type": "adamw", "learning_rate": 4e-5},
                ),
                server=server,
            )
        )

        assert create_response.request_id == "future-test-1"
        assert server.future_store.last_result == {"model_id": "session-a", "type": "create_model"}
        assert server.model_configs["session-a"]["lora_config"] == {"lora_rank": 8, "lora_alpha": 16}
        assert server.model_configs["session-a"]["optimizer_config"]["type"] == "adamw"
        assert server.model_configs["session-a"]["optimizer_config"]["learning_rate"] == 4e-5
        checkpoint_path = tmp_path / "weights" / "session-a" / "checkpoint-001"
        write_session_spec(str(checkpoint_path), server.model_configs["session-a"])

        info_response = asyncio.run(
            weights_info_endpoint(
                WeightsInfoRequest(xorl_path="xorl://session-a/weights/checkpoint-001"),
                server=server,
            )
        )

        assert info_response.base_model == "Qwen/Qwen3-8B"
        assert info_response.lora_rank == 8
        assert info_response.model_dump()["lora_rank"] == 8

    def test_full_weight_create_model_allows_empty_optional_configs(self):
        """Empty lora/optimizer configs from legacy clients are no-op full-weight overrides."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17032",
            engine_output_addr="tcp://127.0.0.1:17033",
            base_model="Qwen/Qwen3-8B",
            skip_initial_checkpoint=True,
            train_config={"lr": 5e-5},
            lora_config={"enable_lora": False},
        )
        server.future_store = _ImmediateFutureStore()
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="Engine timeout"):
            assert timeout_message == "Register session timeout"
            return SimpleNamespace(error=None, outputs=[{"result": {"registered": True}}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        create_response = asyncio.run(
            create_model_endpoint(
                CreateModelRequest(
                    model_id="default",
                    base_model="Qwen/Qwen3-8B",
                    lora_config={},
                    optimizer_config={},
                ),
                server=server,
            )
        )

        assert create_response.request_id == "future-test-1"
        assert client.last_request.payload.session_spec == {
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": False,
        }
        assert client.last_request.payload.materialize is False

    def test_create_model_registers_lora_session_on_workers(self):
        """LoRA create_model must register a worker session spec before exposing the model_id."""
        train_config = {
            "optimizer": "adamw",
            "lr": 2e-5,
            "weight_decay": 0.02,
            "optimizer_dtype": "bf16",
            "optimizer_kwargs": {},
        }
        lora_config = {
            "enable_lora": True,
            "lora_rank": 4,
            "max_lora_rank": 16,
            "lora_alpha": 8,
        }
        default_session_spec = build_default_session_spec(
            base_model="Qwen/Qwen3-8B",
            train_config=train_config,
            lora_config=lora_config,
        )
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17018",
            engine_output_addr="tcp://127.0.0.1:17019",
            base_model="Qwen/Qwen3-8B",
            skip_initial_checkpoint=True,
            default_session_spec=default_session_spec,
            server_lora_config=lora_config,
            max_lora_rank=16,
            train_config=train_config,
            lora_config=lora_config,
        )
        server.future_store = _ImmediateFutureStore()
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="Engine timeout"):
            assert response_future is client.last_request
            assert request_id == client.last_request.request_id
            assert timeout_message == "Register session timeout"
            return SimpleNamespace(error=None, outputs=[{"result": {"registered": True}}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        create_response = asyncio.run(
            create_model_endpoint(
                CreateModelRequest(
                    model_id="policy-a",
                    base_model="Qwen/Qwen3-8B",
                    lora_config={"rank": 12, "alpha": 24},
                    optimizer_config={"learning_rate": 3e-5},
                ),
                server=server,
            )
        )

        assert create_response.request_id == "future-test-1"
        assert client.last_request.operation == "register_session"
        assert isinstance(client.last_request.payload, RegisterSessionData)
        payload = client.last_request.payload
        assert payload.model_id == "policy-a"
        assert payload.materialize is True
        assert payload.session_spec["lora_config"] == {"lora_rank": 12, "lora_alpha": 24}
        assert payload.session_spec["optimizer_config"]["learning_rate"] == pytest.approx(3e-5)
        assert payload.session_spec["optimizer_config"]["weight_decay"] == pytest.approx(0.02)
        assert server.model_configs["policy-a"]["lora_config"] == {"lora_rank": 12, "lora_alpha": 24}
        assert server.model_configs["policy-a"]["optimizer_config"]["learning_rate"] == pytest.approx(3e-5)

    def test_optim_step_supports_native_payload_without_adam_params(self):
        """Native xorl clients can send learning_rate/gradient_clip without legacy adam_params."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17018",
            engine_output_addr="tcp://127.0.0.1:17019",
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
                    model_id="default",
                    learning_rate=3e-4,
                    gradient_clip=1.25,
                )
            )
        )

        assert client.last_request.operation == "optim_step"
        assert client.last_request.payload.lr == pytest.approx(3e-4)
        assert client.last_request.payload.gradient_clip == pytest.approx(1.25)
        assert client.last_request.payload.beta1 is None
        assert response.metrics["learning_rate"] == pytest.approx(3e-4)

    def test_optim_step_supports_tinker_adam_params_payload(self):
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
                    model_id="default",
                    adam_params=AdamParams(
                        learning_rate=2e-4,
                        beta1=0.8,
                        beta2=0.88,
                        eps=1e-6,
                        grad_clip_norm=0.75,
                    ),
                )
            )
        )

        assert client.last_request.operation == "optim_step"
        assert client.last_request.payload.lr == pytest.approx(2e-4)
        assert client.last_request.payload.gradient_clip == pytest.approx(0.75)
        assert client.last_request.payload.beta1 == pytest.approx(0.8)
        assert client.last_request.payload.beta2 == pytest.approx(0.88)
        assert client.last_request.payload.eps == pytest.approx(1e-6)
        assert response.metrics["learning_rate"] == pytest.approx(2e-4)

    def test_optim_step_uses_registered_session_default_learning_rate(self):
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

    def test_optim_step_uses_server_train_config_learning_rate_for_full_weight_default(self):
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

    def test_optim_step_uses_learning_rate_registered_by_create_model(self):
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

    def test_optim_step_rejects_missing_learning_rate_without_session_default(self):
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

    def test_save_weights_for_sampler_uses_lora_path_for_canonical_lora_config(self, tmp_path):
        """Canonical lora_rank metadata should select LoRA-only sampler export."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17030",
            engine_output_addr="tcp://127.0.0.1:17031",
            output_dir=str(tmp_path),
        )
        client = _FakeOrchestratorClient()
        server._running = True
        server.orchestrator_client = client
        server.model_configs["session-a"] = {
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"lora_rank": 8, "lora_alpha": 16},
        }

        async def _wait_for_response(self, response_future, request_id, timeout, timeout_message="timeout"):
            return SimpleNamespace(outputs=[{"success": True, "lora_path": str(tmp_path / "adapter")}])

        server._wait_for_response = types.MethodType(_wait_for_response, server)

        response = asyncio.run(
            server.save_weights_for_sampler(SaveWeightsForSamplerRequest(model_id="session-a", name="step-1"))
        )

        assert response.path == "xorl://session-a/sampler_weights/step-1"
        assert client.last_request.operation == "save_lora_only"
        assert client.last_request.payload.model_id == "session-a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
