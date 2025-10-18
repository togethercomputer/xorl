"""
Tests for Orchestrator-Runner Message Protocol.

This test suite verifies the message protocol between Orchestrator and Runner Rank 0:
1. Message types and structure
2. Serialization/deserialization
3. RunnerDispatchCommand.create() factory method
4. Message validation
"""

import json
import time

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.server]

from xorl.server.protocol.orchestrator_runner import (
    MessageType,
    BaseMessage,
    RunnerReady,
    RunnerAck,
    RunnerResponse,
    RunnerDispatchCommand,
    serialize_message,
    deserialize_message,
    create_ack_for_request,
)
from xorl.server.protocol.operations import (
    EmptyData,
    LoadStateData,
    ModelPassData,
    OptimStepData,
    SaveStateData,
)


def test_dispatch_command_all_operations():
    """Test RunnerDispatchCommand.create() for all operation types."""
    # forward_backward
    batches = [{"input_ids": [[1, 2, 3]], "labels": [[2, 3, 4]], "position_ids": [[0, 1, 2]]}]
    msg = RunnerDispatchCommand.create(
        operation="forward_backward",
        payload=ModelPassData(batches=batches, loss_fn="causallm_loss"),
        request_id="custom-id",
    )
    assert msg.operation == "forward_backward"
    assert msg.payload.batches == batches
    assert msg.payload.loss_fn == "causallm_loss"
    assert msg.message_id == "custom-id"

    # optim_step (with and without clip)
    msg = RunnerDispatchCommand.create(
        operation="optim_step", payload=OptimStepData(lr=0.001, gradient_clip=1.0), request_id="opt-id",
    )
    assert msg.payload.lr == 0.001 and msg.payload.gradient_clip == 1.0
    msg_no_clip = RunnerDispatchCommand.create(operation="optim_step", payload=OptimStepData(lr=0.001))
    assert msg_no_clip.payload.gradient_clip is None

    # save_state
    msg = RunnerDispatchCommand.create(
        operation="save_state", payload=SaveStateData(checkpoint_path="/tmp/ckpt.pt", save_optimizer=True),
    )
    assert msg.payload.checkpoint_path == "/tmp/ckpt.pt" and msg.payload.save_optimizer is True

    # load_state
    msg = RunnerDispatchCommand.create(
        operation="load_state", payload=LoadStateData(checkpoint_path="/tmp/ckpt.pt", load_optimizer=False),
    )
    assert msg.payload.load_optimizer is False

    # health_check and shutdown
    msg = RunnerDispatchCommand.create(operation="health_check", payload=EmptyData())
    assert msg.operation == "health_check"
    msg = RunnerDispatchCommand.create(operation="shutdown", payload=EmptyData())
    assert msg.operation == "shutdown"

    # RunnerResponse with error
    msg = RunnerResponse(request_id="req-123", success=False, error="Test error message")
    assert msg.success is False and msg.error == "Test error message"


def test_serialization_roundtrip_all_types():
    """Test complete serialization roundtrip for all message types."""
    messages = [
        RunnerReady(worker_rank=0, world_size=8),
        RunnerAck(request_id="req-123"),
        RunnerResponse(request_id="req-123", success=True, result={"loss": 2.5}),
        RunnerDispatchCommand.create("forward_backward", ModelPassData(batches=[], loss_fn="test")),
        RunnerDispatchCommand.create("optim_step", OptimStepData(lr=0.001)),
        RunnerDispatchCommand.create("health_check", EmptyData()),
    ]
    for original_msg in messages:
        serialized = serialize_message(original_msg)
        deserialized = deserialize_message(serialized)
        assert type(deserialized) == type(original_msg)
        assert deserialized.message_type == original_msg.message_type
        assert deserialized.message_id == original_msg.message_id


def test_json_conversion():
    """Test JSON serialization and deserialization of messages."""
    # to_json
    msg = RunnerReady(worker_rank=0, world_size=4)
    data = json.loads(msg.to_json())
    assert data["message_type"] == "ready"
    assert data["worker_rank"] == 0 and data["world_size"] == 4

    # from_json
    json_str = json.dumps({
        "message_type": "ready", "worker_rank": 2, "world_size": 8,
        "device": "cuda:2", "message_id": "msg-123", "timestamp": time.time(),
    })
    msg = BaseMessage.from_json(json_str)
    assert isinstance(msg, RunnerReady)
    assert msg.worker_rank == 2 and msg.world_size == 8 and msg.device == "cuda:2"


def test_complex_data_and_edge_cases():
    """Test large batches, complex results, None values, ID uniqueness, and timestamp accuracy."""
    # Large batches
    batches = [{"input_ids": [list(range(100)) for _ in range(10)],
                "labels": [list(range(100, 200)) for _ in range(10)],
                "position_ids": [list(range(100)) for _ in range(10)],
                "request_id": "req-123", "batch_id": i} for i in range(5)]
    msg = RunnerDispatchCommand.create(
        operation="forward_backward", payload=ModelPassData(batches=batches, loss_fn="causallm_loss"),
    )
    deserialized = deserialize_message(serialize_message(msg))
    assert len(deserialized.payload.batches) == 5
    assert len(deserialized.payload.batches[0]["input_ids"]) == 10

    # Complex result
    result = {"loss": 2.5, "gradients": {"layer1": [0.1, 0.2, 0.3]}, "metrics": {"accuracy": 0.95}}
    msg = RunnerResponse(request_id="req-123", success=True, result=result)
    deserialized = deserialize_message(serialize_message(msg))
    assert deserialized.result["loss"] == 2.5
    assert deserialized.result["gradients"]["layer1"] == [0.1, 0.2, 0.3]

    # None values
    msg = RunnerResponse(request_id="req-123", success=True, result={}, error=None, execution_time=None)
    deserialized = deserialize_message(serialize_message(msg))
    assert deserialized.error is None and deserialized.execution_time is None

    # ID uniqueness
    assert RunnerReady().message_id != RunnerReady().message_id

    # Timestamp accuracy
    before = time.time()
    msg = RunnerReady()
    after = time.time()
    assert before <= msg.timestamp <= after

    # ACK creation
    request = RunnerDispatchCommand.create(
        operation="forward_backward", payload=ModelPassData(batches=[], loss_fn="test"), request_id="req-123",
    )
    ack = create_ack_for_request(request)
    assert isinstance(ack, RunnerAck)
    assert ack.request_id == "req-123"
    assert ack.message_type == MessageType.ACKNOWLEDGEMENT
