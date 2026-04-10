"""
Basic tests for APIServer.

Tests focus on server initialization and configuration validation.
Integration tests with Orchestrator are excluded as they require complex infrastructure.
"""

import asyncio
import time

import pytest

from xorl.server.api_server.api_types import (
    AdamParams,
    CreateSessionRequest,
    Datum,
    DatumInput,
    ForwardBackwardRequest,
    LoadWeightsRequest,
    OptimStepRequest,
    SaveWeightsRequest,
    SessionHeartbeatRequest,
    UntypedAPIFuture,
)
from xorl.server.api_server.endpoints import create_session_endpoint, save_weights_endpoint, session_heartbeat_endpoint
from xorl.server.api_server.server import APIServer, app


pytestmark = [pytest.mark.cpu, pytest.mark.server]


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
        assert request.adam_params.learning_rate == 1e-4
        data = request.model_dump()
        assert data["adam_params"]["learning_rate"] == 1e-4

        # Defaults
        request = OptimStepRequest()
        assert request.adam_params.learning_rate == 0.0001 and request.gradient_clip is None

        # SaveWeightsRequest
        request = SaveWeightsRequest(path="/tmp/checkpoint")
        assert request.path == "/tmp/checkpoint"
        data = request.model_dump()
        assert data["path"] == "/tmp/checkpoint"

        # LoadWeightsRequest
        request = LoadWeightsRequest(path="/tmp/checkpoint", optimizer=True)
        assert request.optimizer is True

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
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
