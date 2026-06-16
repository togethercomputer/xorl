"""
Tests for API request/response types (Pydantic models).

Tests validation, serialization, and edge cases for the updated API structure.
"""

import pytest
from pydantic import ValidationError

from xorl.server.api_server.api_types import (
    AdamParams,
    CreateModelRequest,
    CreateSessionRequest,
    Datum,
    DatumInput,
    ErrorResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    HealthCheckResponse,
    LoadWeightsRequest,
    LoadWeightsResponse,
    LoRAConfigRequest,
    LossFnOutput,
    OptimizerConfigRequest,
    OptimStepRequest,
    OptimStepResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    WeightsInfoResponse,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class TestDatumAndForwardBackward:
    """Test Datum, DatumInput, ForwardBackwardRequest and ForwardBackwardResponse."""

    def test_datum_forward_backward_request_and_response(self):
        """Test Datum validation, DatumInput defaults, request/response creation and validation."""
        # Valid datum with multiple inputs
        datum = Datum(
            model_input={"input_ids": [1, 2, 3], "position_ids": [0, 1, 2]},
            loss_fn_inputs={"labels": [2, 3, 4], "weights": [1.0, 1.0, 1.0]},
        )
        assert datum.model_input["input_ids"] == [1, 2, 3]
        assert len(datum.model_input) == 2
        assert len(datum.loss_fn_inputs) == 2

        # Missing required field
        with pytest.raises(ValidationError):
            Datum(model_input={"input_ids": [1, 2, 3]})

        # Empty dicts are valid
        datum = Datum(model_input={}, loss_fn_inputs={})
        assert len(datum.model_input) == 0

        # DatumInput defaults
        datum_input = DatumInput(
            data=[Datum(model_input={"input_ids": [1, 2, 3]}, loss_fn_inputs={"labels": [2, 3, 4]})]
        )
        assert datum_input.loss_fn == "causallm_loss"
        assert len(datum_input.data) == 1

        # Multiple samples
        datum_input = DatumInput(
            data=[
                Datum(model_input={"input_ids": [1, 2]}, loss_fn_inputs={"labels": [2, 3]}),
                Datum(model_input={"input_ids": [3, 4]}, loss_fn_inputs={"labels": [4, 5]}),
            ],
            loss_fn="causallm_loss",
        )
        assert len(datum_input.data) == 2

        # ForwardBackwardRequest: explicit, defaults, multiple samples
        request = ForwardBackwardRequest(
            model_id="test-model",
            forward_backward_input=DatumInput(
                data=[Datum(model_input={"input_ids": [1, 2, 3]}, loss_fn_inputs={"labels": [2, 3, 4]})],
                loss_fn="causallm_loss",
            ),
        )
        assert request.model_id == "test-model"
        assert len(request.forward_backward_input.data) == 1
        assert request.forward_backward_input.loss_fn == "causallm_loss"

        request = ForwardBackwardRequest(
            forward_backward_input=DatumInput(
                data=[Datum(model_input={"input_ids": [1, 2, 3]}, loss_fn_inputs={"labels": [2, 3, 4]})],
            ),
        )
        assert request.model_id == "default"
        assert request.forward_backward_input.loss_fn == "causallm_loss"

        request = ForwardBackwardRequest(
            session_id="session-123",
            forward_backward_input=DatumInput(
                data=[Datum(model_input={"input_ids": [1, 2, 3]}, loss_fn_inputs={"labels": [2, 3, 4]})],
            ),
        )
        assert request.model_id == "session-123"

        request = ForwardBackwardRequest(
            forward_backward_input=DatumInput(
                data=[
                    Datum(model_input={"input_ids": [1, 2]}, loss_fn_inputs={"labels": [2, 3]}),
                    Datum(model_input={"input_ids": [3, 4]}, loss_fn_inputs={"labels": [4, 5]}),
                ],
            ),
        )
        assert len(request.forward_backward_input.data) == 2

        # ForwardBackwardResponse: full, multiple losses, missing fields
        response = ForwardBackwardResponse(
            loss_fn_output_type="single_loss",
            loss_fn_outputs=[LossFnOutput(loss=2.345)],
            metrics={"accuracy": 0.95, "perplexity": 3.2},
            info={"grad_norm": 1.23},
        )
        assert response.loss_fn_outputs[0].loss == 2.345
        assert response.metrics["accuracy"] == 0.95

        response = ForwardBackwardResponse(
            loss_fn_output_type="multi_loss",
            loss_fn_outputs=[LossFnOutput(loss=2.0), LossFnOutput(loss=3.0)],
            metrics={},
            info={},
        )
        assert len(response.loss_fn_outputs) == 2

        with pytest.raises(ValidationError):
            ForwardBackwardResponse(
                loss_fn_output_type="single_loss",
                loss_fn_outputs=[LossFnOutput(loss=2.345)],
            )


class TestOptimWeightsHealthAndSerialization:
    """Test OptimStep, Weights, Health, Error types and serialization."""

    def test_optim_step_types(self):
        """Test optimizer/session request types and OptimStepRequest/Response."""
        optimizer = OptimizerConfigRequest(
            type="adamw",
            learning_rate=0.001,
            betas=[0.9, 0.999],
            eps=1e-8,
        )
        assert optimizer.learning_rate == 0.001
        assert optimizer.betas == [0.9, 0.999]

        lora = LoRAConfigRequest(rank=8, alpha=16)
        assert lora.lora_rank == 8
        assert lora.lora_alpha == 16
        assert lora.model_dump(exclude_none=True) == {"lora_rank": 8, "lora_alpha": 16}
        assert set(LoRAConfigRequest.model_json_schema()["properties"]) == {"lora_rank", "lora_alpha"}

        create_request = CreateModelRequest(
            model_id="session-a",
            base_model="Qwen/Qwen3-8B",
            lora_config=LoRAConfigRequest(rank=8, alpha=16),
            optimizer_config=OptimizerConfigRequest(type="signsgd", learning_rate=2e-4),
        )
        assert create_request.lora_config is not None
        assert create_request.lora_config.lora_rank == 8
        assert create_request.optimizer_config is not None
        assert create_request.optimizer_config.type == "signsgd"

        create_session_request = CreateSessionRequest(lora_config={"rank": 4, "alpha": 10})
        assert create_session_request.lora_config is not None
        assert create_session_request.lora_config.model_dump(exclude_none=True) == {
            "lora_rank": 4,
            "lora_alpha": 10,
        }

        request = OptimStepRequest(
            model_id="test-model",
            learning_rate=1e-4,
            gradient_clip=1.0,
        )
        assert request.model_id == "test-model"
        assert request.learning_rate == 1e-4
        assert request.gradient_clip == 1.0

        request = OptimStepRequest()
        assert request.model_id == "default"
        assert request.learning_rate is None
        assert request.gradient_clip is None
        assert request.adam_params is None

        legacy_request = OptimStepRequest(
            **{
                "session_id": "legacy-session",
                "adam_params": {"learning_rate": 2e-4, "grad_clip_norm": 1.5},
            }
        )
        assert legacy_request.model_id == "legacy-session"
        assert legacy_request.learning_rate == pytest.approx(2e-4)
        assert legacy_request.gradient_clip == pytest.approx(1.5)
        assert isinstance(legacy_request.adam_params, AdamParams)
        assert legacy_request.adam_params.learning_rate == pytest.approx(2e-4)

        response = OptimStepResponse(
            metrics={"grad_norm": 1.234, "learning_rate": 1e-4, "step": 100},
            info={},
        )
        assert response.metrics["grad_norm"] == 1.234
        response = OptimStepResponse(metrics={}, info={})
        assert len(response.metrics) == 0

        weights_info = WeightsInfoResponse(
            base_model="Qwen/Qwen3-8B",
            is_lora=True,
            lora_config={"lora_rank": 8, "lora_alpha": 16},
            optimizer_config={
                "type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "optimizer_dtype": "bf16",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "optimizer_kwargs": {},
            },
        )
        assert weights_info.lora_config.lora_rank == 8
        assert weights_info.lora_rank == 8
        assert weights_info.optimizer_config.type == "adamw"

        legacy_weights_info = WeightsInfoResponse(
            base_model="Qwen/Qwen3-8B",
            is_lora=True,
            lora_rank=8,
        )
        assert legacy_weights_info.lora_rank == 8
        assert legacy_weights_info.lora_config is None

        full_weight_info = WeightsInfoResponse(
            base_model="Qwen/Qwen3-8B",
            is_lora=False,
        )
        assert full_weight_info.lora_config is None
        assert full_weight_info.optimizer_config is None

        partial_lora_info = WeightsInfoResponse(
            base_model="Qwen/Qwen3-8B",
            is_lora=True,
        )
        assert partial_lora_info.lora_rank is None

    def test_weights_health_error_and_serialization(self):
        """Test save/load/sampler types, health, error, and roundtrip serialization."""
        # SaveWeightsRequest
        request = SaveWeightsRequest(model_id="test-model", path="/tmp/checkpoint")
        assert request.model_id == "test-model"
        assert request.path == "/tmp/checkpoint"
        request = SaveWeightsRequest(session_id="session-123", path="/tmp/checkpoint")
        assert request.model_id == "session-123"
        request = SaveWeightsRequest()
        assert request.model_id == "default"
        assert request.path is None

        # SaveWeightsResponse
        response = SaveWeightsResponse(path="/tmp/checkpoint/model.pt")
        assert response.path == "/tmp/checkpoint/model.pt"
        with pytest.raises(ValidationError):
            SaveWeightsResponse()

        # LoadWeightsRequest
        request = LoadWeightsRequest(model_id="test-model", path="/tmp/checkpoint", optimizer=True)
        assert request.optimizer is True
        request = LoadWeightsRequest(session_id="session-123", path="/tmp/checkpoint")
        assert request.model_id == "session-123"
        request = LoadWeightsRequest(path="/tmp/checkpoint")
        assert request.model_id == "default"
        assert request.optimizer is True
        with pytest.raises(ValidationError):
            LoadWeightsRequest()

        # CreateModelRequest
        request = CreateModelRequest(
            session_id="session-123",
            base_model="Qwen/Qwen3-8B",
            lora_config={"rank": 64},
        )
        assert request.model_id == "session-123"
        assert request.lora_config is not None
        assert request.lora_config.lora_rank == 64
        assert "rank" not in request.lora_config.model_dump(exclude_none=True)

        # LoadWeightsResponse
        response = LoadWeightsResponse(path="xorl://default/weights/checkpoint-001")
        assert response.path == "xorl://default/weights/checkpoint-001"
        with pytest.raises(ValidationError):
            LoadWeightsResponse()

        # SaveWeightsForSamplerRequest
        request = SaveWeightsForSamplerRequest(model_id="test-model", name="step-100")
        assert request.name == "step-100"
        request = SaveWeightsForSamplerRequest(name="step-0")
        assert request.model_id == "default"
        with pytest.raises(ValidationError):
            SaveWeightsForSamplerRequest()

        # SaveWeightsForSamplerResponse
        response = SaveWeightsForSamplerResponse(path="/tmp/sampler_checkpoint/model.pt")
        assert response.path == "/tmp/sampler_checkpoint/model.pt"

        # HealthCheckResponse
        response = HealthCheckResponse(
            status="healthy",
            engine_running=True,
            active_requests=5,
            total_requests=100,
        )
        assert response.status == "healthy"
        assert response.engine_running is True
        response = HealthCheckResponse(
            status="unhealthy",
            engine_running=False,
            active_requests=0,
            total_requests=0,
        )
        assert response.engine_running is False

        # ErrorResponse
        response = ErrorResponse(error="Something went wrong", detail="Detailed info")
        assert response.error == "Something went wrong"
        assert response.detail == "Detailed info"
        response = ErrorResponse(error="Error occurred")
        assert response.detail is None

        # --- Serialization roundtrips ---
        # ForwardBackwardRequest
        request = ForwardBackwardRequest(
            model_id="test-model",
            forward_backward_input=DatumInput(
                data=[Datum(model_input={"input_ids": [1, 2, 3]}, loss_fn_inputs={"labels": [2, 3, 4]})],
            ),
        )
        data = request.model_dump()
        assert "model_id" in data
        assert "forward_backward_input" in data
        assert "data" in data["forward_backward_input"]
        request2 = ForwardBackwardRequest(**data)
        assert request2.model_id == request.model_id
        assert len(request2.forward_backward_input.data) == len(request.forward_backward_input.data)

        # OptimStepRequest
        request = OptimStepRequest(learning_rate=0.001, gradient_clip=1.0)
        data = request.model_dump()
        assert data["learning_rate"] == 0.001
        request2 = OptimStepRequest(**data)
        assert request2.learning_rate == request.learning_rate

        legacy_data = {"session_id": "legacy-session", "adam_params": {"learning_rate": 3e-4}}
        request3 = OptimStepRequest(**legacy_data)
        assert request3.model_id == "legacy-session"
        assert request3.learning_rate == pytest.approx(3e-4)

        legacy_data_with_clip = {
            "session_id": "legacy-session",
            "adam_params": {"learning_rate": 3e-4, "grad_clip_norm": 0.75},
        }
        request4 = OptimStepRequest(**legacy_data_with_clip)
        assert request4.gradient_clip == pytest.approx(0.75)

        # HealthCheckResponse
        response = HealthCheckResponse(
            status="healthy",
            engine_running=True,
            active_requests=0,
            total_requests=10,
        )
        data = response.model_dump()
        response2 = HealthCheckResponse(**data)
        assert response2.status == response.status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
