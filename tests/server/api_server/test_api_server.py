"""
Basic tests for APIServer.

Tests focus on server initialization and configuration validation.
Integration tests with Orchestrator are excluded as they require complex infrastructure.
"""

import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.server]

from xorl.server.api_server.api_types import (
    AdamParams,
    Datum,
    DatumInput,
    ForwardBackwardRequest,
    LoadWeightsRequest,
    OptimStepRequest,
    SaveWeightsRequest,
)
from xorl.server.api_server.server import APIServer


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
