"""
Tests for API-Engine Message Protocol.

This module tests the communication protocol between API Server and Engine:
- Message serialization/deserialization (msgpack)
- Typed request construction
- Response builder functions
- Validation and utility functions
"""

import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.server]

from xorl.server.protocol.api_orchestrator import (
    OrchestratorOutputs,
    # Core Message Types
    OrchestratorRequest,
    OutputType,
    # Enums
    RequestType,
    create_error_output,
    # Response Builders
    create_forward_backward_output,
    create_health_check_output,
    create_load_state_output,
    create_optim_step_output,
    create_save_state_output,
    get_operation_from_request,
    is_streaming_output,
    validate_output,
    # Validation and Utilities
    validate_request,
)
from xorl.server.protocol.operations import (
    AbortData,
    LoadStateData,
    ModelPassData,
    OptimStepData,
    SaveStateData,
)


class TestSerializationAndBuilders:
    """Test msgpack roundtrip, request builders, and response builders."""

    def test_roundtrip_request_and_response_builders(self):
        """Test msgpack roundtrip for request/output, all request types, and all response builders."""
        # --- Request roundtrip ---
        original_req = OrchestratorRequest(
            request_id="roundtrip-test",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=[{"model_input": {"input_ids": [1, 2, 3, 4]}, "loss_fn_inputs": {"labels": [2, 3, 4, 5]}}],
                loss_fn="causallm_loss",
            ),
            timestamp=9999.0,
        )
        packed = original_req.to_msgpack()
        assert isinstance(packed, bytes) and len(packed) > 0
        restored_req = OrchestratorRequest.from_msgpack(packed)
        assert restored_req.request_id == original_req.request_id
        assert restored_req.request_type == original_req.request_type
        assert restored_req.operation == original_req.operation
        assert restored_req.payload.data == original_req.payload.data
        assert restored_req.payload.loss_fn == original_req.payload.loss_fn
        assert restored_req.timestamp == original_req.timestamp

        # --- Output roundtrip ---
        original_out = OrchestratorOutputs(
            request_id="roundtrip-output-test",
            output_type=OutputType.FORWARD_BACKWARD,
            outputs=[{"loss": 1.234, "valid_tokens": 512, "grads_norm": 0.987}],
            finished=True,
            error=None,
            timestamp=3333.0,
        )
        packed = original_out.to_msgpack()
        assert isinstance(packed, bytes) and len(packed) > 0
        restored_out = OrchestratorOutputs.from_msgpack(packed)
        assert restored_out.request_id == original_out.request_id
        assert restored_out.output_type == original_out.output_type
        assert restored_out.outputs == original_out.outputs
        assert restored_out.finished == original_out.finished
        assert restored_out.error == original_out.error
        assert restored_out.timestamp == original_out.timestamp

        # --- All request types ---
        # Forward backward with custom ID
        data = [{"model_input": {"input_ids": [1, 2, 3]}, "loss_fn_inputs": {"labels": [2, 3, 4]}}]
        request = OrchestratorRequest(
            operation="forward_backward",
            payload=ModelPassData(data=data, loss_fn="causallm_loss"),
        )
        assert request.request_type == RequestType.ADD
        assert request.operation == "forward_backward"
        assert request.payload.data == data
        custom = OrchestratorRequest(
            request_id="custom-fb-id",
            operation="forward_backward",
            payload=ModelPassData(data=[{"model_input": {"input_ids": [1]}}]),
        )
        assert custom.request_id == "custom-fb-id"

        # Optim step (full and minimal)
        request = OrchestratorRequest(
            operation="optim_step",
            payload=OptimStepData(lr=0.001, gradient_clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8),
        )
        assert request.payload.lr == 0.001
        assert request.payload.gradient_clip == 1.0
        minimal = OrchestratorRequest(operation="optim_step", payload=OptimStepData(lr=0.0001))
        assert minimal.payload.gradient_clip is None
        assert minimal.payload.beta1 is None

        # Save state (with and without optimizer)
        request = OrchestratorRequest(
            operation="save_state",
            payload=SaveStateData(checkpoint_path="/tmp/checkpoint", save_optimizer=True),
        )
        assert request.payload.save_optimizer is True
        no_opt = OrchestratorRequest(
            operation="save_state",
            payload=SaveStateData(checkpoint_path="/tmp/weights_only", save_optimizer=False),
        )
        assert no_opt.payload.save_optimizer is False

        # Load state
        request = OrchestratorRequest(
            operation="load_state",
            payload=LoadStateData(checkpoint_path="/tmp/checkpoint", load_optimizer=True),
        )
        assert request.payload.load_optimizer is True

        # Health check and abort
        health = OrchestratorRequest(request_type=RequestType.UTILITY, operation="health_check")
        assert health.request_type == RequestType.UTILITY
        abort = OrchestratorRequest(
            request_type=RequestType.ABORT,
            operation="abort",
            payload=AbortData(target_request_id="request-to-abort"),
        )
        assert abort.payload.target_request_id == "request-to-abort"

        # --- All response builders ---
        # Forward backward (full, minimal, error, additional metrics)
        output = create_forward_backward_output(
            request_id="fb-req-1",
            loss=2.345,
            valid_tokens=1024,
            grads_norm=1.23,
        )
        assert output.output_type == OutputType.FORWARD_BACKWARD
        assert output.outputs[0]["loss"] == 2.345
        assert output.outputs[0]["valid_tokens"] == 1024

        minimal = create_forward_backward_output(request_id="fb-req-2", loss=1.5)
        assert "valid_tokens" not in minimal.outputs[0]

        with_error = create_forward_backward_output(request_id="fb-err", loss=0.0, error="Forward pass failed")
        assert with_error.error == "Forward pass failed"

        with_metrics = create_forward_backward_output(
            request_id="fb-req-3",
            loss=2.0,
            additional_metrics={"perplexity": 10.5, "accuracy": 0.85},
        )
        assert with_metrics.outputs[0]["perplexity"] == 10.5

        # Optim step (full and minimal)
        output = create_optim_step_output(
            request_id="optim-req-1",
            step=1000,
            learning_rate=0.0001,
            grad_norm=0.95,
        )
        assert output.outputs[0]["step"] == 1000
        assert output.outputs[0]["lr"] == 0.0001
        minimal = create_optim_step_output(request_id="optim-req-2")
        assert minimal.outputs == [{}]

        # Save state (success and error)
        output = create_save_state_output(
            request_id="save-req-1",
            checkpoint_path="/tmp/checkpoint",
            success=True,
        )
        assert output.outputs[0]["success"] is True
        error_out = create_save_state_output(
            request_id="save-err",
            checkpoint_path="/tmp/failed",
            success=False,
            error="Disk full",
        )
        assert error_out.error == "Disk full"

        # Load state
        output = create_load_state_output(
            request_id="load-req-1",
            checkpoint_path="/tmp/checkpoint",
            success=True,
        )
        assert output.outputs[0]["success"] is True

        # Health check
        output = create_health_check_output(
            request_id="health-req-1",
            status="healthy",
            active_requests=5,
            total_requests=1000,
        )
        assert output.outputs[0]["status"] == "healthy"
        with_info = create_health_check_output(
            request_id="health-req-2",
            additional_info={"uptime": 3600, "memory_usage": 0.75},
        )
        assert with_info.outputs[0]["uptime"] == 3600

        # Error
        output = create_error_output(request_id="error-req-1", error_message="Something went wrong")
        assert output.output_type == OutputType.ERROR
        assert output.error == "Something went wrong"
        assert output.outputs == []

        custom_type = create_error_output(
            request_id="error-req-2",
            error_message="Forward failed",
            operation_type=OutputType.FORWARD_BACKWARD,
        )
        assert custom_type.output_type == OutputType.FORWARD_BACKWARD


class TestValidationUtilitiesAndIntegration:
    """Test validation, utility functions, and complete message flows."""

    def test_validation_and_utilities(self):
        """Test request/output validation, get_operation, and is_streaming_output."""
        # Valid ADD
        assert (
            validate_request(
                OrchestratorRequest(
                    request_id="valid-req",
                    request_type=RequestType.ADD,
                    operation="forward_backward",
                )
            )
            is True
        )

        # Valid ABORT
        assert (
            validate_request(
                OrchestratorRequest(
                    request_id="abort-req",
                    request_type=RequestType.ABORT,
                    operation="abort",
                    payload=AbortData(target_request_id="req-to-abort"),
                )
            )
            is True
        )

        # Valid UTILITY
        assert (
            validate_request(
                OrchestratorRequest(
                    request_id="utility-req",
                    request_type=RequestType.UTILITY,
                    operation="health_check",
                )
            )
            is True
        )

        # Missing request_id
        with pytest.raises(ValueError, match="must have request_id"):
            validate_request(
                OrchestratorRequest(
                    request_id="",
                    request_type=RequestType.ADD,
                    operation="test",
                )
            )

        # ADD missing operation
        with pytest.raises(ValueError, match="must have 'operation'"):
            validate_request(OrchestratorRequest(request_id="missing-op", request_type=RequestType.ADD))

        # ABORT missing target
        with pytest.raises(ValueError, match="must have 'target_request_id'"):
            validate_request(
                OrchestratorRequest(
                    request_id="abort-no-target",
                    request_type=RequestType.ABORT,
                    operation="abort",
                )
            )

        # Valid streaming output
        assert (
            validate_output(
                OrchestratorOutputs(
                    request_id="streaming-output",
                    output_type=OutputType.FORWARD_BACKWARD,
                    outputs=[{"partial": "data"}],
                    finished=False,
                )
            )
            is True
        )

        # Valid error output
        assert (
            validate_output(
                OrchestratorOutputs(
                    request_id="error-output",
                    output_type=OutputType.ERROR,
                    outputs=[],
                    finished=True,
                    error="Test error",
                )
            )
            is True
        )

        # Missing request_id
        with pytest.raises(ValueError, match="must have request_id"):
            validate_output(
                OrchestratorOutputs(
                    request_id="",
                    output_type=OutputType.FORWARD_BACKWARD,
                )
            )

        # get_operation_from_request
        assert get_operation_from_request(OrchestratorRequest(operation="forward_backward")) == "forward_backward"
        assert get_operation_from_request(OrchestratorRequest()) is None

        # is_streaming_output
        assert (
            is_streaming_output(
                OrchestratorOutputs(
                    request_id="stream-test",
                    output_type=OutputType.FORWARD_BACKWARD,
                    finished=False,
                    error=None,
                )
            )
            is True
        )
        assert (
            is_streaming_output(
                OrchestratorOutputs(
                    request_id="finished-test",
                    output_type=OutputType.FORWARD_BACKWARD,
                    finished=True,
                )
            )
            is False
        )
        assert (
            is_streaming_output(
                OrchestratorOutputs(
                    request_id="error-test",
                    output_type=OutputType.ERROR,
                    finished=False,
                    error="Test error",
                )
            )
            is False
        )

    def test_complete_request_response_and_error_flow(self):
        """Test complete request-response flow and error handling for all operation types."""
        # Forward backward flow
        data = [{"model_input": {"input_ids": [1, 2, 3, 4, 5]}, "loss_fn_inputs": {"labels": [2, 3, 4, 5, 6]}}]
        request = OrchestratorRequest(
            operation="forward_backward",
            payload=ModelPassData(data=data, loss_fn="causallm_loss"),
        )
        assert validate_request(request) is True
        received_request = OrchestratorRequest.from_msgpack(request.to_msgpack())
        assert received_request.operation == "forward_backward"
        assert received_request.payload.data == data

        output = create_forward_backward_output(request_id=received_request.request_id, loss=2.345, valid_tokens=5)
        assert validate_output(output) is True
        received_output = OrchestratorOutputs.from_msgpack(output.to_msgpack())
        assert received_output.outputs[0]["loss"] == 2.345

        # Optim step flow
        optim_req = OrchestratorRequest(
            operation="optim_step",
            payload=OptimStepData(lr=0.001, gradient_clip=1.0),
        )
        received = OrchestratorRequest.from_msgpack(optim_req.to_msgpack())
        optim_out = create_optim_step_output(
            request_id=received.request_id,
            step=100,
            learning_rate=0.001,
            grad_norm=0.85,
        )
        final = OrchestratorOutputs.from_msgpack(optim_out.to_msgpack())
        assert final.outputs[0]["step"] == 100 and final.outputs[0]["grad_norm"] == 0.85

        # Save state flow
        save_req = OrchestratorRequest(
            operation="save_state",
            payload=SaveStateData(checkpoint_path="/tmp/checkpoint_step_1000", save_optimizer=True),
        )
        received = OrchestratorRequest.from_msgpack(save_req.to_msgpack())
        save_out = create_save_state_output(
            request_id=received.request_id,
            checkpoint_path=received.payload.checkpoint_path,
            success=True,
        )
        final = OrchestratorOutputs.from_msgpack(save_out.to_msgpack())
        assert final.outputs[0]["success"] is True

        # Error handling flow
        request = OrchestratorRequest(
            operation="forward_backward",
            payload=ModelPassData(data=[{"model_input": {"input_ids": [1, 2, 3]}}]),
        )
        error_output = create_error_output(
            request_id=request.request_id,
            error_message="CUDA out of memory",
            operation_type=OutputType.FORWARD_BACKWARD,
        )
        assert validate_output(error_output) is True
        assert error_output.error == "CUDA out of memory"
        assert not is_streaming_output(error_output)

        received = OrchestratorOutputs.from_msgpack(error_output.to_msgpack())
        assert received.error == "CUDA out of memory"
