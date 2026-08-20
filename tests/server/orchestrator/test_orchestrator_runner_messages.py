"""
Tests for Orchestrator-Runner Message Protocol.

This test suite verifies the message protocol between Orchestrator and Runner Rank 0:
1. Message types and structure
2. Serialization/deserialization
3. RunnerDispatchCommand.create() factory method
4. Message validation
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
import torch

from xorl.server.protocol.operations import (
    EmptyData,
    LoadStateData,
    ModelPassData,
    OptimStepData,
    SaveStateData,
)
from xorl.server.protocol.orchestrator_runner import (
    RunnerAck,
    RunnerDispatchCommand,
    RunnerReady,
    RunnerResponse,
    deserialize_message,
    serialize_message,
)
from xorl.server.runner.utils.rank0_protocol import Rank0Protocol


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _make_protocol_stub():
    protocol = object.__new__(Rank0Protocol)
    protocol.rank = 0
    protocol.world_size = 1
    protocol.device = "cuda:0"
    protocol._request_count = 0
    protocol.request_queue = asyncio.Queue()
    protocol.current_client_id = None
    protocol.channel = AsyncMock()
    return protocol


def test_protocol_roundtrips_and_rank0_ready_handshake_policy():
    batches = [{"input_ids": [[1, 2, 3]], "labels": [[2, 3, 4]], "position_ids": [[0, 1, 2]]}]
    messages = [
        RunnerReady(worker_rank=0, world_size=8, device="cuda:0"),
        RunnerAck(request_id="req-123"),
        RunnerResponse(request_id="req-123", success=True, result={"loss": 2.5}),
        RunnerResponse(request_id="req-failed", success=False, error="Test error message"),
        RunnerDispatchCommand.create(
            "forward_backward",
            ModelPassData(batches=batches, loss_fn="causallm_loss"),
            request_id="custom-id",
        ),
        RunnerDispatchCommand.create("optim_step", OptimStepData(lr=0.001, gradient_clip=1.0)),
        RunnerDispatchCommand.create("save_state", SaveStateData(checkpoint_path="/tmp/ckpt.pt", save_optimizer=True)),
        RunnerDispatchCommand.create("load_state", LoadStateData(checkpoint_path="/tmp/ckpt.pt", load_optimizer=False)),
        RunnerDispatchCommand.create("health_check", EmptyData()),
        RunnerDispatchCommand.create("shutdown", EmptyData()),
    ]
    for original_msg in messages:
        serialized = serialize_message(original_msg)
        deserialized = deserialize_message(serialized)
        assert deserialized == original_msg

    with pytest.raises((ValueError, TypeError)):
        deserialize_message(b"\x80\x04N.")

    original = RunnerResponse(
        request_id="tensor-response",
        result={"values": torch.tensor([[1, 2], [3, 4]])},
    )
    restored = deserialize_message(serialize_message(original))
    assert restored.result["values"].equal(original.result["values"])

    asyncio.run(_assert_rank0_ready_handshake_policy())


async def _assert_rank0_ready_handshake_policy():
    protocol = _make_protocol_stub()
    client_id = b"\x00\x80test-client"
    protocol.channel.recv = AsyncMock(return_value=(client_id, serialize_message(RunnerAck(request_id="ack-123"))))

    await protocol._send_ready(client_id)

    protocol.channel.send.assert_called_once()
    assert isinstance(deserialize_message(protocol.channel.send.call_args[0][1]), RunnerReady)
    assert protocol.request_queue.empty()
    assert protocol._request_count == 0

    protocol = _make_protocol_stub()
    request = RunnerDispatchCommand.create(
        operation="forward_backward",
        payload=ModelPassData(
            batches=[{"input_ids": [1, 2, 3]}],
            loss_fn="importance_sampling",
            loss_fn_params={"eps_clip": 0.2},
            model_id="model-42",
        ),
    )
    protocol.channel.recv = AsyncMock(return_value=(client_id, serialize_message(request)))

    await protocol._send_ready(client_id)

    assert protocol.channel.send.call_count == 2
    assert isinstance(deserialize_message(protocol.channel.send.call_args_list[0][0][1]), RunnerReady)
    ack_msg = deserialize_message(protocol.channel.send.call_args_list[1][0][1])
    assert isinstance(ack_msg, RunnerAck) and ack_msg.request_id == request.message_id
    queued_client_id, queued_request = await protocol.request_queue.get()
    assert queued_client_id == client_id
    assert queued_request.operation == "forward_backward"
    assert queued_request.payload.loss_fn == "importance_sampling"
    assert queued_request.payload.loss_fn_params == {"eps_clip": 0.2}
    assert queued_request.payload.model_id == "model-42"
    assert protocol._request_count == 1

    protocol = _make_protocol_stub()
    old_client = b"\x00\x80old-client"
    new_client = b"\x00\x80new-client"
    protocol.current_client_id = old_client
    request = RunnerDispatchCommand.create(operation="health_check", payload=EmptyData())
    protocol.channel.recv = AsyncMock(return_value=(new_client, serialize_message(request)))

    await protocol._send_ready(new_client)

    queued_client_id, _ = await protocol.request_queue.get()
    assert queued_client_id == new_client and queued_client_id != old_client

    protocol = _make_protocol_stub()
    wrong_msg = RunnerReady(worker_rank=1, world_size=2)
    protocol.channel.recv = AsyncMock(return_value=(b"\x00\x80test", serialize_message(wrong_msg)))
    await protocol._send_ready(b"\x00\x80test")
    assert protocol.channel.send.call_count == 1
    assert protocol.request_queue.empty()
    assert protocol._request_count == 0

    protocol = _make_protocol_stub()
    protocol.channel.recv = AsyncMock(side_effect=RuntimeError("Unexpected frame count"))
    await protocol._send_ready(b"\x00\x80test")
    assert protocol.channel.send.call_count == 1
    assert protocol.request_queue.empty()
    assert not hasattr(protocol, "_pending_request")
