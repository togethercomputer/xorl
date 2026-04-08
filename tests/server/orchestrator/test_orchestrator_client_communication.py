"""
Tests for API Server <-> Engine communication.

This test suite verifies the full communication path between OrchestratorClient (API Server)
and Orchestrator (Engine), focusing on the ROUTER-DEALER-PUSH-PULL socket pattern.

The tests use a mock engine that simulates the processing without actual workers.
"""

import asyncio
import logging
import socket as sock

import pytest
import pytest_asyncio
import zmq
import zmq.asyncio

from xorl.server.api_server.orchestrator_client import OrchestratorClient
from xorl.server.protocol.api_orchestrator import (
    OrchestratorOutputs,
    OrchestratorRequest,
    OutputType,
    RequestType,
)
from xorl.server.protocol.operations import (
    ModelPassData,
    OptimStepData,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Mock Engine
# ============================================================================


class MockEngine:
    """
    Mock Engine that simulates the INPUT/OUTPUT socket behavior of Orchestrator.

    This mock implements the same socket pattern as the real engine:
    - INPUT socket (DEALER): connects to API server's ROUTER to receive requests
    - OUTPUT socket (PUSH): binds for API server's PULL to receive outputs
    """

    def __init__(self, input_addr, output_addr, engine_identity=b"engine-0", response_delay=0.0):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.engine_identity = engine_identity
        self.response_delay = response_delay
        self.context = None
        self.input_socket = None
        self.output_socket = None
        self._running = False
        self._task = None
        self.requests_received = []
        self.outputs_sent = []

    async def start(self):
        if self._running:
            return
        self.context = zmq.asyncio.Context()
        self.input_socket = self.context.socket(zmq.DEALER)
        self.input_socket.setsockopt(zmq.IDENTITY, self.engine_identity)
        self.input_socket.setsockopt(zmq.LINGER, 0)
        self.input_socket.connect(self.input_addr)
        self.output_socket = self.context.socket(zmq.PUSH)
        self.output_socket.setsockopt(zmq.LINGER, 0)
        self.output_socket.bind(self.output_addr)
        await asyncio.sleep(0.2)
        self._running = True
        self._task = asyncio.create_task(self._process_requests())

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.input_socket:
            self.input_socket.close()
        if self.output_socket:
            self.output_socket.close()
        if self.context:
            self.context.term()

    async def _process_requests(self):
        while self._running:
            try:
                events = await self.input_socket.poll(timeout=100)
                if not (events & zmq.POLLIN):
                    continue
                frames = await self.input_socket.recv_multipart(copy=False)
                message_bytes = frames[1].bytes if len(frames) >= 2 else frames[0].bytes if len(frames) == 1 else None
                if message_bytes is None:
                    continue
                request = OrchestratorRequest.from_msgpack(message_bytes)
                self.requests_received.append(request)
                if self.response_delay > 0:
                    await asyncio.sleep(self.response_delay)
                output = await self._handle_request(request)
                await self.output_socket.send(output.to_msgpack())
                self.outputs_sent.append(output)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mock engine processing: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _handle_request(self, request):
        op = request.operation
        if op == "forward_backward":
            num_samples = len(request.payload.data) if hasattr(request.payload, "data") else 0
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.FORWARD_BACKWARD,
                outputs=[{"loss": 2.5, "num_samples": num_samples, "status": "success"}],
                finished=True,
            )
        elif op == "optim_step":
            lr = request.payload.lr if hasattr(request.payload, "lr") else 0.0001
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.OPTIM_STEP,
                outputs=[{"grad_norm": 1.23, "learning_rate": lr, "status": "success"}],
                finished=True,
            )
        elif op == "health_check":
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.HEALTH_CHECK,
                outputs=[{"status": "healthy", "workers_discovered": True, "active_workers": 8}],
                finished=True,
            )
        else:
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.ERROR,
                outputs=[],
                finished=True,
                error=f"Unknown operation: {op}",
            )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def zmq_addresses():
    def find_free_port():
        with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            return s.getsockname()[1]

    return {"input": f"tcp://127.0.0.1:{find_free_port()}", "output": f"tcp://127.0.0.1:{find_free_port()}"}


@pytest_asyncio.fixture
async def mock_engine(zmq_addresses):
    engine = MockEngine(input_addr=zmq_addresses["input"], output_addr=zmq_addresses["output"])
    await engine.start()
    yield engine
    await engine.stop()


@pytest_asyncio.fixture
async def orchestrator_client(zmq_addresses):
    client = OrchestratorClient(input_addr=zmq_addresses["input"], output_addr=zmq_addresses["output"])
    await client.start()
    yield client
    await client.stop()


# ============================================================================
# Helper to create requests
# ============================================================================


def _health_check_request():
    return OrchestratorRequest(request_type=RequestType.UTILITY, operation="health_check")


def _forward_backward_request(data):
    return OrchestratorRequest(operation="forward_backward", payload=ModelPassData(data=data))


def _optim_step_request(lr, gradient_clip=None):
    return OrchestratorRequest(operation="optim_step", payload=OptimStepData(lr=lr, gradient_clip=gradient_clip))


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_basic_communication_and_roundtrip(mock_engine, orchestrator_client):
    """Test connection, send, receive, and full roundtrip."""
    await asyncio.sleep(0.4)
    assert mock_engine._running and orchestrator_client._running

    # Send and verify engine receives
    request = _health_check_request()
    future = await orchestrator_client.send_request(request)
    output = await asyncio.wait_for(future, timeout=3.0)
    assert output is not None
    assert output.request_id == request.request_id
    assert output.output_type == OutputType.HEALTH_CHECK
    assert output.finished is True
    assert output.outputs[0]["status"] == "healthy"
    assert len(mock_engine.requests_received) >= 1
    assert len(mock_engine.outputs_sent) >= 1


@pytest.mark.asyncio
async def test_forward_backward_and_optim_step(mock_engine, orchestrator_client):
    """Test forward_backward and optim_step operations through communication."""
    await asyncio.sleep(0.2)

    # Forward backward
    fb_request = _forward_backward_request(
        data=[
            {"model_input": {"input_ids": [1, 2, 3]}, "loss_fn_inputs": {"labels": [2, 3, 4]}},
            {"model_input": {"input_ids": [5, 6, 7]}, "loss_fn_inputs": {"labels": [6, 7, 8]}},
        ]
    )
    future = await orchestrator_client.send_request(fb_request)
    output = await asyncio.wait_for(future, timeout=2.0)
    assert output.output_type == OutputType.FORWARD_BACKWARD
    assert output.outputs[0]["num_samples"] == 2

    # Empty samples
    empty_request = _forward_backward_request(data=[])
    future = await orchestrator_client.send_request(empty_request)
    output = await asyncio.wait_for(future, timeout=2.0)
    assert output.outputs[0]["num_samples"] == 0

    # Optim step
    opt_request = _optim_step_request(lr=0.001, gradient_clip=1.0)
    future = await orchestrator_client.send_request(opt_request)
    output = await asyncio.wait_for(future, timeout=2.0)
    assert output.output_type == OutputType.OPTIM_STEP
    assert output.outputs[0]["learning_rate"] == 0.001


@pytest.mark.asyncio
async def test_multiple_and_interleaved_requests(mock_engine, orchestrator_client):
    """Test multiple sequential requests and interleaved request types."""
    await asyncio.sleep(0.2)

    # Multiple sequential
    requests = [_health_check_request() for _ in range(5)]
    futures = [await orchestrator_client.send_request(req) for req in requests]
    outputs = [await asyncio.wait_for(f, timeout=2.0) for f in futures]
    assert len(outputs) == 5
    assert {req.request_id for req in requests} == {out.request_id for out in outputs}

    # Interleaved types
    mixed_requests = [
        _health_check_request(),
        _forward_backward_request(data=[{"model_input": {"input_ids": [1, 2, 3]}, "loss_fn_inputs": {}}]),
        _optim_step_request(lr=0.001),
        _health_check_request(),
    ]
    futures = [await orchestrator_client.send_request(req) for req in mixed_requests]
    outputs = [await asyncio.wait_for(f, timeout=2.0) for f in futures]
    assert outputs[0].output_type == OutputType.HEALTH_CHECK
    assert outputs[1].output_type == OutputType.FORWARD_BACKWARD
    assert outputs[2].output_type == OutputType.OPTIM_STEP
    assert outputs[3].output_type == OutputType.HEALTH_CHECK


@pytest.mark.asyncio
async def test_edge_cases_and_lifecycle(zmq_addresses, mock_engine, orchestrator_client):
    """Test timeout, delayed response, stats, serialization, errors, and lifecycle."""
    await asyncio.sleep(0.2)

    # Timeout
    output = await orchestrator_client.get_output(timeout=0.5)
    assert output is None

    # Stats
    stats = orchestrator_client.get_stats()
    assert stats["running"] is True

    # Serialization roundtrip
    data = [
        {
            "model_input": {"input_ids": list(range(100)), "attention_mask": [1] * 100},
            "loss_fn_inputs": {"labels": list(range(100, 200))},
        }
    ]
    request = _forward_backward_request(data=data)
    await orchestrator_client.send_request(request)
    await asyncio.sleep(0.2)
    received = mock_engine.requests_received[-1]
    assert received.payload.data == data

    # Error before start
    client = OrchestratorClient(input_addr="tcp://127.0.0.1:50000", output_addr="tcp://127.0.0.1:50001")
    with pytest.raises(RuntimeError, match="not started"):
        await client.send_request(_health_check_request())
    with pytest.raises(RuntimeError, match="not started"):
        await client.get_output(timeout=1.0)

    # Client start/stop - use fresh ports to avoid address conflicts

    def _find_free_port():
        with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            return s.getsockname()[1]

    fresh_input = f"tcp://127.0.0.1:{_find_free_port()}"
    fresh_output = f"tcp://127.0.0.1:{_find_free_port()}"

    client2 = OrchestratorClient(input_addr=fresh_input, output_addr=fresh_output)
    assert not client2._running
    await client2.start()
    assert client2._running
    await client2.stop()
    assert not client2._running

    # Context manager
    fresh_input2 = f"tcp://127.0.0.1:{_find_free_port()}"
    fresh_output2 = f"tcp://127.0.0.1:{_find_free_port()}"
    async with OrchestratorClient(input_addr=fresh_input2, output_addr=fresh_output2) as ctx_client:
        assert ctx_client._running
    assert not ctx_client._running
