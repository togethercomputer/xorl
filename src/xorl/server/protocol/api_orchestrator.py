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
    ABORT_GRADIENT_EPOCH = "abort_gradient_epoch"
    SAVE_STATE = "save_state"
    SAVE_LORA_ONLY = "save_lora_only"
    LOAD_STATE = "load_state"
    SLEEP = "sleep"
    WAKE_UP = "wake_up"
    HEALTH_CHECK = "health_check"
    SYNC_INFERENCE_WEIGHTS = "sync_inference_weights"
    REGISTER_SESSION = "register_session"
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
        return f"OrchestratorOutputs(id={self.request_id[:8]}..., type={self.output_type.value}, status={status}{error_str})"
