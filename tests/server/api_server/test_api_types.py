"""
Tests for API request/response types (Pydantic models).

Tests validation, serialization, and edge cases for the updated API structure.
"""

import pytest
from pydantic import ValidationError

from xorl.server.api_server.api_types import (
    Datum,
    DatumInput,
    LossFnOutput,
    AdamParams,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    OptimStepRequest,
    OptimStepResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    LoadWeightsRequest,
    LoadWeightsResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    HealthCheckResponse,
    ErrorResponse,
)


class TestDatum:
    """Test Datum validation."""

    def test_valid_datum(self):
        """Test creating valid datum."""
        datum = Datum(
            model_input={"input_ids": [1, 2, 3]},
            loss_fn_inputs={"labels": [2, 3, 4]},
        )

        assert "input_ids" in datum.model_input
        assert "labels" in datum.loss_fn_inputs
        assert datum.model_input["input_ids"] == [1, 2, 3]

    def test_datum_with_multiple_inputs(self):
        """Test datum with multiple input types."""
        datum = Datum(
            model_input={
                "input_ids": [1, 2, 3],
                "position_ids": [0, 1, 2],
            },
            loss_fn_inputs={
                "labels": [2, 3, 4],
                "weights": [1.0, 1.0, 1.0],
            },
        )

        assert len(datum.model_input) == 2
        assert len(datum.loss_fn_inputs) == 2

    def test_datum_missing_fields(self):
        """Test datum with missing required fields."""
        with pytest.raises(ValidationError):
            Datum(
                model_input={"input_ids": [1, 2, 3]}
                # Missing loss_fn_inputs
            )

    def test_datum_empty_dicts(self):
        """Test datum with empty dictionaries."""
        datum = Datum(
            model_input={},
            loss_fn_inputs={},
        )

        assert len(datum.model_input) == 0
        assert len(datum.loss_fn_inputs) == 0


class TestDatumInput:
    """Test DatumInput validation."""

    def test_valid_datum_input(self):
        """Test creating valid datum input."""
        datum_input = DatumInput(
            data=[
                Datum(
                    model_input={"input_ids": [1, 2, 3]},
                    loss_fn_inputs={"labels": [2, 3, 4]},
                )
            ],
            loss_fn="causallm_loss",
        )

        assert len(datum_input.data) == 1
        assert datum_input.loss_fn == "causallm_loss"

    def test_datum_input_default_loss_fn(self):
        """Test datum input with default loss function."""
        datum_input = DatumInput(
            data=[
                Datum(
                    model_input={"input_ids": [1, 2, 3]},
                    loss_fn_inputs={"labels": [2, 3, 4]},
                )
            ]
        )

        assert datum_input.loss_fn == "causallm_loss"

    def test_datum_input_multiple_samples(self):
        """Test datum input with multiple samples."""
        datum_input = DatumInput(
            data=[
                Datum(
                    model_input={"input_ids": [1, 2]},
                    loss_fn_inputs={"labels": [2, 3]},
                ),
                Datum(
                    model_input={"input_ids": [3, 4]},
                    loss_fn_inputs={"labels": [4, 5]},
                ),
            ]
        )

        assert len(datum_input.data) == 2


class TestAdamParams:
    """Test AdamParams validation."""

    def test_valid_adam_params(self):
        """Test creating valid Adam parameters."""
        params = AdamParams(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        )

        assert params.learning_rate == 0.001
        assert params.beta1 == 0.9
        assert params.beta2 == 0.999
        assert params.eps == 1e-8

    def test_adam_params_defaults(self):
        """Test Adam parameters with defaults."""
        params = AdamParams()

        assert params.learning_rate == 0.0001
        assert params.beta1 == 0.9
        assert params.beta2 == 0.95
        assert params.eps == 1e-12


class TestForwardBackwardRequest:
    """Test ForwardBackwardRequest validation."""

    def test_valid_forward_backward_request(self):
        """Test creating valid forward-backward request."""
        request = ForwardBackwardRequest(
            model_id="test-model",
            forward_backward_input=DatumInput(
                data=[
                    Datum(
                        model_input={"input_ids": [1, 2, 3]},
                        loss_fn_inputs={"labels": [2, 3, 4]},
                    )
                ],
                loss_fn="causallm_loss",
            ),
        )

        assert request.model_id == "test-model"
        assert len(request.forward_backward_input.data) == 1
        assert request.forward_backward_input.loss_fn == "causallm_loss"

    def test_forward_backward_request_defaults(self):
        """Test forward-backward request with default values."""
        request = ForwardBackwardRequest(
            forward_backward_input=DatumInput(
                data=[
                    Datum(
                        model_input={"input_ids": [1, 2, 3]},
                        loss_fn_inputs={"labels": [2, 3, 4]},
                    )
                ],
            ),
        )

        assert request.model_id == "default"
        assert request.forward_backward_input.loss_fn == "causallm_loss"

    def test_forward_backward_request_multiple_samples(self):
        """Test forward-backward request with multiple samples."""
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


class TestForwardBackwardResponse:
    """Test ForwardBackwardResponse validation."""

    def test_valid_forward_backward_response(self):
        """Test creating valid forward-backward response."""
        response = ForwardBackwardResponse(
            loss_fn_output_type="single_loss",
            loss_fn_outputs=[LossFnOutput(loss=2.345)],
            metrics={"accuracy": 0.95, "perplexity": 3.2},
            info={"grad_norm": 1.23},
        )

        assert response.loss_fn_output_type == "single_loss"
        assert len(response.loss_fn_outputs) == 1
        assert response.loss_fn_outputs[0].loss == 2.345
        assert response.metrics["accuracy"] == 0.95
        assert response.info["grad_norm"] == 1.23

    def test_forward_backward_response_multiple_losses(self):
        """Test response with multiple loss outputs."""
        response = ForwardBackwardResponse(
            loss_fn_output_type="multi_loss",
            loss_fn_outputs=[
                LossFnOutput(loss=2.0),
                LossFnOutput(loss=3.0),
            ],
            metrics={},
            info={},
        )

        assert len(response.loss_fn_outputs) == 2

    def test_forward_backward_response_missing_fields(self):
        """Test response with missing required fields."""
        with pytest.raises(ValidationError):
            ForwardBackwardResponse(
                loss_fn_output_type="single_loss",
                loss_fn_outputs=[LossFnOutput(loss=2.345)]
                # Missing metrics and info
            )


class TestOptimStepRequest:
    """Test OptimStepRequest validation."""

    def test_valid_optim_step_request(self):
        """Test creating valid optimizer step request."""
        request = OptimStepRequest(
            model_id="test-model",
            adam_params=AdamParams(learning_rate=1e-4),
            gradient_clip=1.0,
        )

        assert request.model_id == "test-model"
        assert request.adam_params.learning_rate == 1e-4
        assert request.gradient_clip == 1.0

    def test_optim_step_request_defaults(self):
        """Test optimizer step request with defaults."""
        request = OptimStepRequest()

        assert request.model_id == "default"
        assert request.adam_params.learning_rate == 0.0001
        assert request.gradient_clip is None

    def test_optim_step_request_no_clip(self):
        """Test optimizer step without gradient clipping."""
        request = OptimStepRequest(
            adam_params=AdamParams(learning_rate=0.001)
        )

        assert request.gradient_clip is None


class TestOptimStepResponse:
    """Test OptimStepResponse validation."""

    def test_valid_optim_step_response(self):
        """Test creating valid optimizer step response."""
        response = OptimStepResponse(
            metrics={
                "grad_norm": 1.234,
                "learning_rate": 1e-4,
                "step": 100,
            },
            info={},
        )

        assert response.metrics["grad_norm"] == 1.234
        assert response.metrics["learning_rate"] == 1e-4
        assert response.metrics["step"] == 100

    def test_optim_step_response_empty_metrics(self):
        """Test response with empty metrics."""
        response = OptimStepResponse(metrics={}, info={})

        assert len(response.metrics) == 0


class TestSaveWeightsRequest:
    """Test SaveWeightsRequest validation."""

    def test_valid_save_weights_request(self):
        """Test creating valid save weights request."""
        request = SaveWeightsRequest(
            model_id="test-model",
            path="/tmp/checkpoint",
        )

        assert request.model_id == "test-model"
        assert request.path == "/tmp/checkpoint"

    def test_save_weights_request_defaults(self):
        """Test save weights request with defaults."""
        request = SaveWeightsRequest()

        assert request.model_id == "default"
        assert request.path is None


class TestSaveWeightsResponse:
    """Test SaveWeightsResponse validation."""

    def test_valid_save_weights_response(self):
        """Test creating valid save weights response."""
        response = SaveWeightsResponse(
            path="/tmp/checkpoint/model.pt",
        )

        assert response.path == "/tmp/checkpoint/model.pt"

    def test_save_weights_response_missing_path(self):
        """Test response with missing path."""
        with pytest.raises(ValidationError):
            SaveWeightsResponse()  # path is required


class TestLoadWeightsRequest:
    """Test LoadWeightsRequest validation."""

    def test_valid_load_weights_request(self):
        """Test creating valid load weights request."""
        request = LoadWeightsRequest(
            model_id="test-model",
            path="/tmp/checkpoint",
            optimizer=True,
        )

        assert request.model_id == "test-model"
        assert request.path == "/tmp/checkpoint"
        assert request.optimizer is True

    def test_load_weights_request_defaults(self):
        """Test load weights request with defaults."""
        request = LoadWeightsRequest(path="/tmp/checkpoint")

        assert request.model_id == "default"
        assert request.optimizer is False

    def test_load_weights_request_missing_path(self):
        """Test load weights request without path."""
        with pytest.raises(ValidationError):
            LoadWeightsRequest()  # path is required


class TestLoadWeightsResponse:
    """Test LoadWeightsResponse validation."""

    def test_valid_load_weights_response(self):
        """Test creating valid load weights response."""
        response = LoadWeightsResponse(path="xorl://default/weights/checkpoint-001")

        assert response.path == "xorl://default/weights/checkpoint-001"

    def test_load_weights_response_missing_path(self):
        """Test load weights response requires path."""
        with pytest.raises(ValidationError):
            LoadWeightsResponse()


class TestSaveWeightsForSamplerRequest:
    """Test SaveWeightsForSamplerRequest validation."""

    def test_valid_save_weights_for_sampler_request(self):
        """Test creating valid save weights for sampler request."""
        request = SaveWeightsForSamplerRequest(
            model_id="test-model",
            name="step-100",
        )

        assert request.model_id == "test-model"
        assert request.name == "step-100"

    def test_save_weights_for_sampler_request_defaults(self):
        """Test save weights for sampler request with defaults."""
        request = SaveWeightsForSamplerRequest(name="step-0")

        assert request.model_id == "default"
        assert request.name == "step-0"

    def test_save_weights_for_sampler_request_missing_name(self):
        """Test save weights for sampler request without name."""
        with pytest.raises(ValidationError):
            SaveWeightsForSamplerRequest()  # name is required


class TestSaveWeightsForSamplerResponse:
    """Test SaveWeightsForSamplerResponse validation."""

    def test_valid_save_weights_for_sampler_response(self):
        """Test creating valid save weights for sampler response."""
        response = SaveWeightsForSamplerResponse(
            path="/tmp/sampler_checkpoint/model.pt",
        )

        assert response.path == "/tmp/sampler_checkpoint/model.pt"


class TestHealthCheckResponse:
    """Test HealthCheckResponse validation."""

    def test_valid_health_check_response(self):
        """Test creating valid health check response."""
        response = HealthCheckResponse(
            status="healthy",
            engine_running=True,
            active_requests=5,
            total_requests=100,
        )

        assert response.status == "healthy"
        assert response.engine_running is True
        assert response.active_requests == 5
        assert response.total_requests == 100

    def test_health_check_response_unhealthy(self):
        """Test unhealthy health check response."""
        response = HealthCheckResponse(
            status="unhealthy",
            engine_running=False,
            active_requests=0,
            total_requests=0,
        )

        assert response.status == "unhealthy"
        assert response.engine_running is False


class TestErrorResponse:
    """Test ErrorResponse validation."""

    def test_valid_error_response(self):
        """Test creating valid error response."""
        response = ErrorResponse(
            error="Something went wrong",
            detail="Detailed error information",
        )

        assert response.error == "Something went wrong"
        assert response.detail == "Detailed error information"

    def test_error_response_no_detail(self):
        """Test error response without detail."""
        response = ErrorResponse(error="Error occurred")

        assert response.error == "Error occurred"
        assert response.detail is None


class TestSerialization:
    """Test JSON serialization/deserialization."""

    def test_forward_backward_request_serialization(self):
        """Test request can be serialized to JSON."""
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

        # Serialize to dict
        data = request.model_dump()
        assert "model_id" in data
        assert "forward_backward_input" in data
        assert "data" in data["forward_backward_input"]

        # Deserialize from dict
        request2 = ForwardBackwardRequest(**data)
        assert request2.model_id == request.model_id
        assert len(request2.forward_backward_input.data) == len(request.forward_backward_input.data)

    def test_optim_step_request_serialization(self):
        """Test optimizer step request serialization."""
        request = OptimStepRequest(
            adam_params=AdamParams(learning_rate=0.001),
            gradient_clip=1.0,
        )

        # Serialize to dict
        data = request.model_dump()
        assert "adam_params" in data
        assert data["adam_params"]["learning_rate"] == 0.001

        # Deserialize from dict
        request2 = OptimStepRequest(**data)
        assert request2.adam_params.learning_rate == request.adam_params.learning_rate

    def test_health_check_response_serialization(self):
        """Test response can be serialized to JSON."""
        response = HealthCheckResponse(
            status="healthy",
            engine_running=True,
            active_requests=0,
            total_requests=10,
        )

        # Serialize to dict
        data = response.model_dump()
        assert data["status"] == "healthy"
        assert data["engine_running"] is True

        # Deserialize from dict
        response2 = HealthCheckResponse(**data)
        assert response2.status == response.status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
