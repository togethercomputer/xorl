"""
API-Engine Message Protocol.

This module defines the communication protocol between:
- API Server (Frontend) <-> Engine (Backend)

Core types:
- OrchestratorRequest: typed request with operation + payload
- OrchestratorOutputs: response with outputs list

Serialization: msgpack over ZMQ.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import msgpack

from xorl.server.protocol.operations import (
    EmptyData,
    OperationPayload,
    payload_from_dict,
    payload_to_dict,
)


# ============================================================================
# Request Types and Output Types
# ============================================================================


class RequestType(str, Enum):
    """
    Types of requests that can be sent to the engine.

    ADD: Add new training/inference request to engine queue
    ABORT: Abort an existing request by request_id
    UTILITY: Utility operations (health check, metrics, etc.)
    """

    ADD = "add"
    ABORT = "abort"
    UTILITY = "utility"


class OutputType(str, Enum):
    """
    Types of outputs from the engine backend.

    Maps to the operation type in the request data.
    """

    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAVE_STATE = "save_state"
    SAVE_LORA_ONLY = "save_lora_only"
    LOAD_STATE = "load_state"
    SLEEP = "sleep"
    WAKE_UP = "wake_up"
    HEALTH_CHECK = "health_check"
    SYNC_INFERENCE_WEIGHTS = "sync_inference_weights"
    REGISTER_ADAPTER = "register_adapter"
    SAVE_ADAPTER_STATE = "save_adapter_state"
    LOAD_ADAPTER_STATE = "load_adapter_state"
    GET_ADAPTER_INFO = "get_adapter_info"
    KILL_SESSION = "kill_session"
    ERROR = "error"


# ============================================================================
# Core Message Types
# ============================================================================


@dataclass
class OrchestratorRequest:
    """
    Request message sent from API Server to Engine Backend.

    Serialized with msgpack for efficient transmission over ZMQ.

    Fields:
        request_id: Unique identifier for request tracking
        request_type: Type of request (ADD, ABORT, UTILITY)
        operation: Operation name (e.g. "forward_backward", "optim_step")
        payload: Typed operation payload (ModelPassData, OptimStepData, etc.)
        seq_id: Optional sequence ID for request ordering
        timestamp: Optional timestamp for request timing

    Examples:
        # Forward-backward request
        OrchestratorRequest(
            operation="forward_backward",
            payload=ModelPassData(data=[...], loss_fn="causallm_loss"),
            seq_id=1,
        )

        # Health check request
        OrchestratorRequest(
            request_type=RequestType.UTILITY,
            operation="health_check",
        )
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: RequestType = RequestType.ADD
    operation: str = ""
    payload: "OperationPayload" = field(default_factory=lambda: EmptyData())
    seq_id: Optional[int] = None
    timestamp: Optional[float] = None

    def to_msgpack(self) -> bytes:
        """Serialize request to msgpack bytes for ZMQ transmission."""
        data_dict = {
            "request_id": self.request_id,
            "request_type": self.request_type.value,
            "operation": self.operation,
            "payload": payload_to_dict(self.payload),
            "seq_id": self.seq_id,
            "timestamp": self.timestamp or time.time(),
        }
        return msgpack.packb(data_dict, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "OrchestratorRequest":
        """Deserialize request from msgpack bytes."""
        unpacked = msgpack.unpackb(data, raw=False)
        op = unpacked.get("operation", "")
        payload_dict = unpacked.get("payload", {})
        return cls(
            request_id=unpacked["request_id"],
            request_type=RequestType(unpacked["request_type"]),
            operation=op,
            payload=payload_from_dict(op, payload_dict),
            seq_id=unpacked.get("seq_id"),
            timestamp=unpacked.get("timestamp"),
        )

    def __repr__(self) -> str:
        return f"OrchestratorRequest(id={self.request_id[:8]}..., type={self.request_type.value}, operation={self.operation})"


@dataclass
class OrchestratorOutputs:
    """
    Output message sent from Engine Backend to API Server.

    Supports streaming outputs (finished=False) and final results (finished=True).
    Serialized with msgpack for efficient transmission over ZMQ.

    Fields:
        request_id: Matches the request_id from OrchestratorRequest
        output_type: Type of output (FORWARD_BACKWARD, OPTIM_STEP, etc.)
        outputs: List of output dictionaries (operation-specific)
        finished: Whether this is the final output for the request
        error: Error message if operation failed, None otherwise
        timestamp: Optional timestamp for response timing
    """

    request_id: str
    output_type: OutputType
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    error: Optional[str] = None
    timestamp: Optional[float] = None

    def to_msgpack(self) -> bytes:
        """Serialize outputs to msgpack bytes for ZMQ transmission."""
        data_dict = {
            "request_id": self.request_id,
            "output_type": self.output_type.value,
            "outputs": self.outputs,
            "finished": self.finished,
            "error": self.error,
            "timestamp": self.timestamp or time.time(),
        }
        return msgpack.packb(data_dict, use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> "OrchestratorOutputs":
        """Deserialize outputs from msgpack bytes."""
        unpacked = msgpack.unpackb(data, raw=False)
        return cls(
            request_id=unpacked["request_id"],
            output_type=OutputType(unpacked["output_type"]),
            outputs=unpacked.get("outputs", []),
            finished=unpacked.get("finished", False),
            error=unpacked.get("error"),
            timestamp=unpacked.get("timestamp"),
        )

    def __repr__(self) -> str:
        status = "finished" if self.finished else "streaming"
        error_str = f", error='{self.error}'" if self.error else ""
        return (
            f"OrchestratorOutputs(id={self.request_id[:8]}..., type={self.output_type.value}, status={status}{error_str})"
        )


# ============================================================================
# Response Builder Functions (Engine → API Server)
# ============================================================================


def _build_output(
    request_id: str,
    output_type: OutputType,
    error: Optional[str] = None,
    **fields,
) -> OrchestratorOutputs:
    """Build an OrchestratorOutputs with optional fields filtered."""
    outputs_data = {}
    for key, value in fields.items():
        if value is not None:
            outputs_data[key] = value
    return OrchestratorOutputs(
        request_id=request_id,
        output_type=output_type,
        outputs=[outputs_data],
        finished=True,
        error=error,
    )


def create_forward_backward_output(
    request_id: str,
    loss: float,
    valid_tokens: Optional[int] = None,
    grads_norm: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a forward-backward output response."""
    outputs_data = {"loss": loss}
    if valid_tokens is not None:
        outputs_data["valid_tokens"] = valid_tokens
    if grads_norm is not None:
        outputs_data["grads_norm"] = grads_norm
    if additional_metrics:
        outputs_data.update(additional_metrics)
    return OrchestratorOutputs(
        request_id=request_id, output_type=OutputType.FORWARD_BACKWARD,
        outputs=[outputs_data], finished=True, error=error,
    )


def create_optim_step_output(
    request_id: str,
    step: Optional[int] = None,
    learning_rate: Optional[float] = None,
    grad_norm: Optional[float] = None,
    additional_metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create an optimizer step output response."""
    outputs_data = {}
    if step is not None:
        outputs_data["step"] = step
    if learning_rate is not None:
        outputs_data["lr"] = learning_rate
        outputs_data["learning_rate"] = learning_rate
    if grad_norm is not None:
        outputs_data["grad_norm"] = grad_norm
    if additional_metrics:
        outputs_data.update(additional_metrics)
    return OrchestratorOutputs(
        request_id=request_id, output_type=OutputType.OPTIM_STEP,
        outputs=[outputs_data], finished=True, error=error,
    )


def create_save_state_output(
    request_id: str,
    checkpoint_path: str,
    success: bool = True,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a save checkpoint output response."""
    return _build_output(
        request_id, OutputType.SAVE_STATE, error=error,
        success=success, checkpoint_path=checkpoint_path,
    )


def create_save_lora_only_output(
    request_id: str,
    lora_path: str,
    success: bool = True,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a save LoRA-only output response."""
    return _build_output(
        request_id, OutputType.SAVE_LORA_ONLY, error=error,
        success=success, lora_path=lora_path,
    )


def create_load_state_output(
    request_id: str,
    checkpoint_path: str,
    success: bool = True,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a load checkpoint output response."""
    return _build_output(
        request_id, OutputType.LOAD_STATE, error=error,
        success=success, checkpoint_path=checkpoint_path,
    )


def create_save_adapter_state_output(
    request_id: str,
    model_id: str,
    path: str,
    step: int,
    success: bool = True,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a save adapter state output response."""
    return _build_output(
        request_id, OutputType.SAVE_ADAPTER_STATE, error=error,
        success=success, model_id=model_id, path=path, step=step,
    )


def create_load_adapter_state_output(
    request_id: str,
    model_id: str,
    path: str,
    step: int,
    success: bool = True,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a load adapter state output response."""
    return _build_output(
        request_id, OutputType.LOAD_ADAPTER_STATE, error=error,
        success=success, model_id=model_id, path=path, step=step,
    )


def create_health_check_output(
    request_id: str,
    status: str = "healthy",
    active_requests: int = 0,
    total_requests: int = 0,
    additional_info: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a health check output response."""
    outputs_data = {
        "status": status,
        "active_requests": active_requests,
        "total_requests": total_requests,
    }
    if additional_info:
        outputs_data.update(additional_info)
    return OrchestratorOutputs(
        request_id=request_id, output_type=OutputType.HEALTH_CHECK,
        outputs=[outputs_data], finished=True, error=error,
    )


def create_sleep_output(
    request_id: str,
    status: str = "sleeping",
    offload_time: Optional[float] = None,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a sleep output response."""
    return _build_output(
        request_id, OutputType.SLEEP, error=error,
        status=status, offload_time=offload_time,
    )


def create_wake_up_output(
    request_id: str,
    status: str = "awake",
    load_time: Optional[float] = None,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a wake_up output response."""
    return _build_output(
        request_id, OutputType.WAKE_UP, error=error,
        status=status, load_time=load_time,
    )


def create_sync_weights_output(
    request_id: str,
    success: bool,
    message: str,
    transfer_time: float = 0.0,
    total_bytes: int = 0,
    num_parameters: int = 0,
    num_buckets: int = 0,
    endpoint_results: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> OrchestratorOutputs:
    """Create a sync inference weights output response."""
    return _build_output(
        request_id, OutputType.SYNC_INFERENCE_WEIGHTS, error=error,
        success=success, message=message, transfer_time=transfer_time,
        total_bytes=total_bytes, num_parameters=num_parameters,
        num_buckets=num_buckets, endpoint_results=endpoint_results or [],
    )


def create_error_output(
    request_id: str,
    error_message: str,
    operation_type: OutputType = OutputType.ERROR,
) -> OrchestratorOutputs:
    """Create an error output response."""
    return OrchestratorOutputs(
        request_id=request_id, output_type=operation_type,
        outputs=[], finished=True, error=error_message,
    )


# ============================================================================
# Validation and Utilities
# ============================================================================


def validate_request(request: OrchestratorRequest) -> bool:
    """Validate that a request has required fields."""
    if not request.request_id:
        raise ValueError("Request must have request_id")

    if request.request_type not in RequestType:
        raise ValueError(f"Invalid request_type: {request.request_type}")

    if request.request_type == RequestType.ADD:
        if not request.operation:
            raise ValueError("ADD request must have 'operation'")

    if request.request_type == RequestType.ABORT:
        if not getattr(request.payload, "target_request_id", None):
            raise ValueError("ABORT request must have 'target_request_id' in payload")

    return True


def validate_output(output: OrchestratorOutputs) -> bool:
    """Validate that an output has required fields."""
    if not output.request_id:
        raise ValueError("Output must have request_id")

    if output.output_type not in OutputType:
        raise ValueError(f"Invalid output_type: {output.output_type}")

    return True


def get_operation_from_request(request: OrchestratorRequest) -> Optional[str]:
    """Extract operation type from request."""
    return request.operation or None


def is_streaming_output(output: OrchestratorOutputs) -> bool:
    """Check if output is a streaming (incomplete) output."""
    return not output.finished and output.error is None
