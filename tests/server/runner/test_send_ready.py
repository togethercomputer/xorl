"""
Tests for Rank0Protocol._send_ready pending request handling.

Verifies that when a request arrives before the RunnerAck during
the handshake, it is enqueued directly into the request_queue rather than
being stored in a field that is never checked again.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.server]

from xorl.server.protocol.operations import (
    EmptyData,
    ModelPassData,
)
from xorl.server.protocol.orchestrator_runner import (
    RunnerAck,
    RunnerDispatchCommand,
    RunnerReady,
    deserialize_message,
    serialize_message,
)
from xorl.server.runner.utils.rank0_protocol import Rank0Protocol


def _make_protocol_stub():
    """Create a minimal Rank0Protocol without calling __init__."""
    protocol = object.__new__(Rank0Protocol)
    protocol.rank = 0
    protocol.world_size = 1
    protocol.device = "cuda:0"
    protocol._request_count = 0
    protocol.request_queue = asyncio.Queue()
    protocol.current_client_id = None
    protocol.channel = AsyncMock()
    return protocol


def test_send_ready_normal_and_request_before_ack():
    """Test normal ACK flow and edge case where request arrives before ACK."""

    async def _run():
        # Normal flow: ACK received
        protocol = _make_protocol_stub()
        client_id = b"\x00\x80test-client"
        ack = RunnerAck(request_id="ack-123")
        protocol.channel.recv = AsyncMock(return_value=(client_id, serialize_message(ack)))

        await protocol._send_ready(client_id)

        protocol.channel.send.assert_called_once()
        sent_data = protocol.channel.send.call_args[0][1]
        assert isinstance(deserialize_message(sent_data), RunnerReady)
        assert protocol.request_queue.empty()
        assert protocol._request_count == 0

        # Request before ACK: enqueued with correct data
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

        # Should have sent RunnerReady + ACK
        assert protocol.channel.send.call_count == 2
        assert isinstance(deserialize_message(protocol.channel.send.call_args_list[0][0][1]), RunnerReady)
        ack_msg = deserialize_message(protocol.channel.send.call_args_list[1][0][1])
        assert isinstance(ack_msg, RunnerAck) and ack_msg.request_id == request.message_id

        # Request in queue with all fields preserved
        assert not protocol.request_queue.empty()
        queued_client_id, queued_request = await protocol.request_queue.get()
        assert queued_client_id == client_id
        assert queued_request.operation == "forward_backward"
        assert queued_request.payload.loss_fn == "importance_sampling"
        assert queued_request.payload.loss_fn_params == {"eps_clip": 0.2}
        assert queued_request.payload.model_id == "model-42"
        assert protocol._request_count == 1

    asyncio.run(_run())


def test_send_ready_client_id_and_edge_cases():
    """Test client_id from recv frame, unexpected message type, wrong frame count, and no _pending_request field."""

    async def _run():
        # Client ID from recv frame (not current_client_id)
        protocol = _make_protocol_stub()
        old_client = b"\x00\x80old-client"
        new_client = b"\x00\x80new-client"
        protocol.current_client_id = old_client

        request = RunnerDispatchCommand.create(operation="health_check", payload=EmptyData())
        protocol.channel.recv = AsyncMock(return_value=(new_client, serialize_message(request)))

        await protocol._send_ready(new_client)

        queued_client_id, _ = await protocol.request_queue.get()
        assert queued_client_id == new_client and queued_client_id != old_client

        # Unexpected message type (RunnerReady instead of ACK/request)
        protocol = _make_protocol_stub()
        wrong_msg = RunnerReady(worker_rank=1, world_size=2)
        protocol.channel.recv = AsyncMock(return_value=(b"\x00\x80test", serialize_message(wrong_msg)))

        await protocol._send_ready(b"\x00\x80test")

        assert protocol.channel.send.call_count == 1  # Only RunnerReady sent
        assert protocol.request_queue.empty()
        assert protocol._request_count == 0

        # RuntimeError from channel.recv
        protocol = _make_protocol_stub()
        protocol.channel.recv = AsyncMock(side_effect=RuntimeError("Unexpected frame count"))

        await protocol._send_ready(b"\x00\x80test")

        assert protocol.channel.send.call_count == 1
        assert protocol.request_queue.empty()

    asyncio.run(_run())

    # No _pending_request field
    protocol = _make_protocol_stub()
    assert not hasattr(protocol, "_pending_request")
