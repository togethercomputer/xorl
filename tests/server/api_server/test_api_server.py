"""
Basic tests for APIServer.

Tests focus on server initialization and configuration validation.
Integration tests with EngineCore are excluded as they require complex infrastructure.
"""

import pytest

from xorl.server.api_server.api_server import APIServer
from xorl.server.api_server.api_types import (
    Datum,
    DatumInput,
    ForwardBackwardRequest,
    OptimStepRequest,
    SaveWeightsRequest,
    LoadWeightsRequest,
    AdamParams,
)


class TestAPIServerConfiguration:
    """Test APIServer configuration and initialization."""

    def test_server_initialization_with_defaults(self):
        """Test server can be initialized with default parameters."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17000",
            engine_output_addr="tcp://127.0.0.1:17001",
        )

        assert server.engine_input_addr == "tcp://127.0.0.1:17000"
        assert server.engine_output_addr == "tcp://127.0.0.1:17001"
        assert server.default_timeout == 120.0  # Default value
        assert server._running is False
        assert server.engine_client is None

    def test_server_initialization_with_custom_timeout(self):
        """Test server initialization with custom timeout."""
        server = APIServer(
            engine_input_addr="tcp://127.0.0.1:17002",
            engine_output_addr="tcp://127.0.0.1:17003",
            default_timeout=60.0,
        )

        assert server.default_timeout == 60.0

    def test_server_initialization_with_different_ports(self):
        """Test server can be initialized with different port configurations."""
        server1 = APIServer(
            engine_input_addr="tcp://127.0.0.1:5000",
            engine_output_addr="tcp://127.0.0.1:5001",
        )
        
        server2 = APIServer(
            engine_input_addr="tcp://127.0.0.1:6000",
            engine_output_addr="tcp://127.0.0.1:6001",
        )

        assert server1.engine_input_addr != server2.engine_input_addr
        assert server1.engine_output_addr != server2.engine_output_addr


class TestAPIRequestCreation:
    """Test API request object creation and validation."""

    def test_create_forward_backward_request(self):
        """Test creating a valid forward-backward request."""
        request = ForwardBackwardRequest(
            model_id="test-model",
            forward_backward_input=DatumInput(
                data=[
                    Datum(
                        model_input={"input_ids": [1, 2, 3, 4]},
                        loss_fn_inputs={"labels": [2, 3, 4, 5]},
                    )
                ],
                loss_fn="causallm_loss",
            ),
        )

        assert request.model_id == "test-model"
        assert len(request.forward_backward_input.data) == 1
        assert request.forward_backward_input.loss_fn == "causallm_loss"

    def test_create_forward_backward_request_multiple_samples(self):
        """Test creating request with multiple samples."""
        request = ForwardBackwardRequest(
            forward_backward_input=DatumInput(
                data=[
                    Datum(
                        model_input={"input_ids": [1, 2]},
                        loss_fn_inputs={"labels": [2, 3]},
                    ),
                    Datum(
                        model_input={"input_ids": [3, 4]},
                        loss_fn_inputs={"labels": [4, 5]},
                    ),
                ],
            ),
        )

        assert len(request.forward_backward_input.data) == 2
        assert request.model_id == "default"

    def test_create_optim_step_request(self):
        """Test creating an optimizer step request."""
        request = OptimStepRequest(
            adam_params=AdamParams(learning_rate=1e-4),
            gradient_clip=1.0,
        )

        assert request.adam_params.learning_rate == 1e-4
        assert request.gradient_clip == 1.0

    def test_create_optim_step_request_with_defaults(self):
        """Test optimizer step request with default parameters."""
        request = OptimStepRequest()

        assert request.adam_params.learning_rate == 0.0001
        assert request.adam_params.beta1 == 0.9
        assert request.gradient_clip is None

    def test_create_save_weights_request(self):
        """Test creating a save weights request."""
        request = SaveWeightsRequest(
            path="/tmp/checkpoint",
            save_optimizer=True,
        )

        assert request.path == "/tmp/checkpoint"
        assert request.save_optimizer is True

    def test_create_load_weights_request(self):
        """Test creating a load weights request."""
        request = LoadWeightsRequest(
            path="/tmp/checkpoint",
            load_optimizer=True,
        )

        assert request.path == "/tmp/checkpoint"
        assert request.load_optimizer is True


class TestAPIRequestSerialization:
    """Test API request serialization/deserialization."""

    def test_forward_backward_request_to_dict(self):
        """Test serializing forward-backward request to dict."""
        request = ForwardBackwardRequest(
            model_id="test-model",
            forward_backward_input=DatumInput(
                data=[
                    Datum(
                        model_input={"input_ids": [1, 2, 3]},
                        loss_fn_inputs={"labels": [2, 3, 4]},
                    )
                ],
            ),
        )

        data = request.model_dump()
        assert "model_id" in data
        assert "forward_backward_input" in data
        assert data["model_id"] == "test-model"

    def test_optim_step_request_to_dict(self):
        """Test serializing optimizer step request to dict."""
        request = OptimStepRequest(
            adam_params=AdamParams(learning_rate=0.001),
            gradient_clip=1.0,
        )

        data = request.model_dump()
        assert "adam_params" in data
        assert data["adam_params"]["learning_rate"] == 0.001
        assert data["gradient_clip"] == 1.0

    def test_save_weights_request_to_dict(self):
        """Test serializing save weights request to dict."""
        request = SaveWeightsRequest(
            path="/tmp/test_checkpoint",
            save_optimizer=False,
        )

        data = request.model_dump()
        assert data["path"] == "/tmp/test_checkpoint"
        assert data["save_optimizer"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
