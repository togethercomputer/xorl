"""
Executor-Worker Message Protocol (Rank 0 Only Communication).

This module defines messages for communication between:
- Executor (Engine) ← ZMQ → Worker Rank 0

Workers communicate among themselves via NCCL (not defined here).

Message Flow:
=============

1. Initialization:
   Worker Rank 0 → Executor: RunnerReady
   Executor → Worker Rank 0: RunnerAck

2. Request-Response:
   Executor → Worker Rank 0: RunnerDispatchCommand
   Worker Rank 0 → Executor: RunnerAck (immediate)
   Worker Rank 0 processes (coordinates with other ranks via NCCL)
   Worker Rank 0 → Executor: RunnerResponse
"""

import json
import pickle
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from xorl.server.protocol.operations import (
    EmptyData,
    OperationPayload,
)


# ============================================================================
# Message Types
# ============================================================================


class MessageType(str, Enum):
    """Types of messages in the executor-worker protocol."""

    # Worker → Executor
    READY = "ready"
    ACKNOWLEDGEMENT = "ack"
    RESPONSE = "response"

    # Executor → Worker
    REQUEST = "request"

    # Request types (embedded in REQUEST messages)
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAVE_STATE = "save_state"
    SAVE_LORA_ONLY = "save_lora_only"
    LOAD_STATE = "load_state"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    SLEEP = "sleep"
    WAKE_UP = "wake_up"
    HEALTH_CHECK = "health_check"
    SYNC_INFERENCE_WEIGHTS = "sync_inference_weights"
    REGISTER_ADAPTER = "register_adapter"
    SAVE_ADAPTER_STATE = "save_adapter_state"
    LOAD_ADAPTER_STATE = "load_adapter_state"
    GET_ADAPTER_INFO = "get_adapter_info"
    KILL_SESSION = "kill_session"
    SHUTDOWN = "shutdown"
    SAVE_FULL_WEIGHTS = "save_full_weights"


# ============================================================================
# Base Message
# ============================================================================


@dataclass
class BaseMessage:
    """Base class for all messages."""

    message_type: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "BaseMessage":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        # Determine which subclass to instantiate based on message_type
        msg_type = data.get("message_type")

        if msg_type == MessageType.READY:
            return RunnerReady(**data)
        elif msg_type == MessageType.ACKNOWLEDGEMENT:
            return RunnerAck(**data)
        elif msg_type == MessageType.RESPONSE:
            return RunnerResponse(**data)
        elif msg_type == MessageType.REQUEST:
            return RunnerDispatchCommand(**data)
        else:
            # Fallback to base class
            return cls(**data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.message_type}, id={self.message_id[:8]}...)"


# ============================================================================
# Worker → Executor Messages
# ============================================================================


@dataclass
class RunnerReady(BaseMessage):
    """
    Worker sends this on startup to notify executor it's ready.

    Attributes:
        worker_rank: Worker rank (always 0 for rank 0 communication)
        world_size: Total number of workers
        device: Device info (e.g., "cuda:0")
    """

    message_type: str = MessageType.READY
    worker_rank: int = 0
    world_size: int = 1
    device: Optional[str] = None

    def __post_init__(self):
        if self.message_type != MessageType.READY:
            self.message_type = MessageType.READY


@dataclass
class RunnerAck(BaseMessage):
    """
    Worker sends ACK immediately upon receiving a request.

    Attributes:
        request_id: ID of the request being acknowledged
        received_at: Timestamp when request was received
    """

    message_type: str = MessageType.ACKNOWLEDGEMENT
    request_id: str = ""
    received_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.message_type != MessageType.ACKNOWLEDGEMENT:
            self.message_type = MessageType.ACKNOWLEDGEMENT


@dataclass
class RunnerResponse(BaseMessage):
    """
    Worker sends response after completing a request.

    Attributes:
        request_id: ID of the request this is responding to
        success: Whether the operation succeeded
        result: Result data (operation-specific)
        error: Error message if failed
        execution_time: Time taken to execute (seconds)
    """

    message_type: str = MessageType.RESPONSE
    request_id: str = ""
    success: bool = True
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        if self.message_type != MessageType.RESPONSE:
            self.message_type = MessageType.RESPONSE


# ============================================================================
# Executor → Worker Messages
# ============================================================================


@dataclass
class RunnerDispatchCommand(BaseMessage):
    """
    Executor sends request to worker rank 0.

    Attributes:
        operation: Operation name (e.g. "forward_backward", "optim_step")
        payload: Typed operation payload (ModelPassData, OptimStepData, etc.)
        timeout: Optional timeout for this operation (seconds)
    """

    message_type: str = MessageType.REQUEST
    operation: str = ""
    payload: OperationPayload = field(default_factory=lambda: EmptyData())
    timeout: Optional[float] = None

    def __post_init__(self):
        if self.message_type != MessageType.REQUEST:
            self.message_type = MessageType.REQUEST

    @classmethod
    def create(
        cls,
        operation: str,
        payload: OperationPayload,
        request_id: Optional[str] = None,
    ) -> "RunnerDispatchCommand":
        """Create a typed executor request."""
        return cls(
            message_id=request_id or str(uuid.uuid4()),
            operation=operation,
            payload=payload,
        )


# ============================================================================
# Utility Functions
# ============================================================================


def serialize_message(message: BaseMessage) -> bytes:
    """Serialize message to bytes for ZMQ transmission."""
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_message(data: bytes) -> BaseMessage:
    """Deserialize message from bytes."""
    return pickle.loads(data)


def create_ack_for_request(request: RunnerDispatchCommand) -> RunnerAck:
    """Create an acknowledgement for a given request."""
    return RunnerAck(request_id=request.message_id, received_at=time.time())


def create_response_for_request(
    request: RunnerDispatchCommand,
    success: bool,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    execution_time: Optional[float] = None,
) -> RunnerResponse:
    """Create a response for a given request."""
    return RunnerResponse(
        request_id=request.message_id, success=success, result=result or {}, error=error, execution_time=execution_time
    )
