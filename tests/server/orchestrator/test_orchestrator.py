"""
Tests for Orchestrator - Main Event Loop and Component Integration.

This test suite verifies the Orchestrator's orchestration of:
1. API socket I/O (input/output threads)
2. Scheduler integration (request lifecycle)
3. RequestProcessor integration (backend dispatch)
4. Event loop processing (main coordination)
5. Request routing and error handling

Test Strategy:
- Uses DummyBackend (no ZMQ workers, no separate processes)
- Tests focus on Orchestrator's scheduling, routing, and output formatting
"""

import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.server]
import socket
import time

import zmq

from xorl.server.backend import DummyBackend
from xorl.server.orchestrator.orchestrator import Orchestrator
from xorl.server.protocol.api_orchestrator import (
    OrchestratorOutputs,
    OrchestratorRequest,
    OutputType,
    RequestType,
)
from xorl.server.protocol.operations import (
    AbortData,
    EmptyData,
    ModelPassData,
    OptimStepData,
)


# ============================================================================
# Fixtures
# ============================================================================


def find_free_port():
    """Find a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def addresses():
    """Get free addresses for sockets."""
    input_port = find_free_port()
    output_port = find_free_port()

    return {
        "input": f"tcp://127.0.0.1:{input_port}",
        "output": f"tcp://127.0.0.1:{output_port}",
    }


@pytest.fixture
def orchestrator(addresses):
    """Create Orchestrator instance with DummyBackend."""
    backend = DummyBackend()
    engine = Orchestrator(
        input_addr=addresses["input"],
        output_addr=addresses["output"],
        sample_packing_sequence_len=100,
        enable_packing=True,
        backend=backend,
    )

    engine.start()
    time.sleep(0.1)

    assert engine.request_processor.is_ready(), "RequestProcessor should be ready after engine.start()"

    yield engine

    engine.stop()


@pytest.fixture
def output_socket(addresses):
    """Create output PULL socket to receive outputs from engine."""
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(addresses["output"])

    yield sock

    sock.close()
    context.term()


# ============================================================================
# Helper Functions
# ============================================================================


def receive_outputs(output_socket, timeout_ms=1000, max_outputs=10):
    """Receive outputs from output socket."""
    import msgpack

    outputs = []

    while len(outputs) < max_outputs:
        if output_socket.poll(timeout=timeout_ms):
            data = output_socket.recv()
            output = msgpack.unpackb(data, raw=False)
            outputs.append(OrchestratorOutputs(**output))
        else:
            break

    return outputs


# ============================================================================
# Tests
# ============================================================================


def test_init_start_stop_and_all_operations(addresses, orchestrator, output_socket):
    """Test Orchestrator lifecycle and event loop processing of all operation types."""
    # --- Init and start/stop lifecycle ---
    backend = DummyBackend()
    engine = Orchestrator(
        input_addr=f"tcp://127.0.0.1:{find_free_port()}",
        output_addr=f"tcp://127.0.0.1:{find_free_port()}",
        backend=backend,
    )
    assert engine.input_addr is not None
    assert engine.output_addr is not None
    assert engine.scheduler is not None
    assert engine.request_processor is not None
    assert engine._running is False

    engine.start()
    time.sleep(0.5)
    assert engine._running is True
    assert engine.event_loop_thread.is_alive()
    assert engine.input_thread.is_alive()
    assert engine.output_thread.is_alive()
    assert engine.worker_thread.is_alive()

    engine.stop()
    time.sleep(0.5)
    assert engine._running is False

    # --- Forward backward ---
    fb_request = OrchestratorRequest(
        request_id="req-fb-001",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(
            data=[
                {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]},
                {"input_ids": [10, 20], "labels": [20, 30]},
            ]
        ),
    )
    orchestrator.input_queue.put(fb_request)
    time.sleep(0.5)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=1)
    assert len(outputs) >= 1
    assert outputs[0].request_id == "req-fb-001"
    assert outputs[0].output_type == OutputType.FORWARD_BACKWARD
    assert "loss" in outputs[0].outputs[0]

    # --- Optim step ---
    opt_request = OrchestratorRequest(
        request_id="req-opt-001",
        request_type=RequestType.ADD,
        operation="optim_step",
        payload=OptimStepData(lr=0.001, gradient_clip=1.0),
    )
    orchestrator.input_queue.put(opt_request)
    time.sleep(0.5)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    opt_outputs = [o for o in outputs if o.request_id == "req-opt-001"]
    assert len(opt_outputs) >= 1
    assert opt_outputs[0].output_type == OutputType.OPTIM_STEP

    # --- Health check ---
    health_request = OrchestratorRequest(
        request_id="req-health-001",
        request_type=RequestType.UTILITY,
        operation="health_check",
        payload=EmptyData(),
    )
    orchestrator.input_queue.put(health_request)
    time.sleep(0.3)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    health_outputs = [o for o in outputs if o.request_id == "req-health-001"]
    assert len(health_outputs) >= 1
    assert health_outputs[0].outputs[0]["status"] == "healthy"


def test_errors_abort_e2e_and_concurrent(orchestrator, output_socket):
    """Test error handling, abort, end-to-end flow, and concurrent requests."""
    # --- Invalid operation ---
    invalid_request = OrchestratorRequest(
        request_id="req-error-001",
        request_type=RequestType.ADD,
        operation="invalid_operation",
        payload=EmptyData(),
    )
    orchestrator.input_queue.put(invalid_request)
    time.sleep(0.5)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    error_outputs = [o for o in outputs if o.request_id == "req-error-001"]
    assert len(error_outputs) >= 1
    assert error_outputs[0].output_type == OutputType.ERROR

    # --- Empty datum list ---
    empty_request = OrchestratorRequest(
        request_id="req-empty-001",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[]),
    )
    orchestrator.input_queue.put(empty_request)
    time.sleep(0.5)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    empty_outputs = [o for o in outputs if o.request_id == "req-empty-001"]
    assert len(empty_outputs) >= 1
    assert empty_outputs[0].output_type == OutputType.ERROR

    # --- Abort ---
    fb_request = OrchestratorRequest(
        request_id="req-abort-001",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2], "labels": [2, 3]}]),
    )
    orchestrator.input_queue.put(fb_request)
    time.sleep(0.1)
    abort_request = OrchestratorRequest(
        request_id="abort-req",
        request_type=RequestType.ABORT,
        operation="abort",
        payload=AbortData(target_request_id="req-abort-001"),
    )
    orchestrator.input_queue.put(abort_request)
    time.sleep(0.3)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    abort_outputs = [o for o in outputs if o.request_id == "req-abort-001"]
    assert len(abort_outputs) >= 1

    # --- End-to-end ---
    request = OrchestratorRequest(
        request_id="req-e2e-001",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2, 3, 4, 5], "labels": [2, 3, 4, 5, 6]}]),
    )
    orchestrator.input_queue.put(request)
    time.sleep(0.5)
    status = orchestrator.scheduler.get_request_status("req-e2e-001")
    assert status in ["processing", "completed"]
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    e2e_outputs = [o for o in outputs if o.request_id == "req-e2e-001"]
    assert len(e2e_outputs) >= 1
    assert e2e_outputs[0].finished is True

    # --- Multiple concurrent requests ---
    requests = []
    for i in range(5):
        req = OrchestratorRequest(
            request_id=f"req-concurrent-{i:03d}",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(data=[{"input_ids": list(range(10)), "labels": list(range(1, 11))}]),
        )
        requests.append(req)
        orchestrator.input_queue.put(req)

    time.sleep(2.0)
    outputs = receive_outputs(output_socket, timeout_ms=2000, max_outputs=10)
    output_ids = {o.request_id for o in outputs}
    for req in requests:
        assert req.request_id in output_ids


def test_stats_throughput_and_health_check(orchestrator):
    """Test engine statistics, throughput tracking, and non-blocking health check."""
    # --- Stats ---
    stats = orchestrator.get_stats()
    assert "running" in stats
    assert "scheduler" in stats
    assert "request_processor" in stats
    assert "queues" in stats
    assert stats["running"] is True

    initial_stats = orchestrator.get_stats()
    for i in range(3):
        req = OrchestratorRequest(
            request_id=f"req-throughput-{i:03d}",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(data=[{"input_ids": [1, 2], "labels": [2, 3]}]),
        )
        orchestrator.input_queue.put(req)
    time.sleep(1.0)
    final_stats = orchestrator.get_stats()
    assert final_stats["scheduler"]["total_requests"] > initial_stats["scheduler"]["total_requests"]

    # --- Non-blocking health check ---
    bare = Orchestrator(
        input_addr="tcp://127.0.0.1:15555",
        output_addr="tcp://127.0.0.1:15556",
        rank0_worker_address="tcp://127.0.0.1:15557",
    )

    # Health check identified correctly
    assert (
        bare._is_health_check_request(
            OrchestratorRequest(
                request_id="test-health",
                request_type=RequestType.UTILITY,
                operation="health_check",
            )
        )
        is True
    )

    # Non-health-check requests
    assert (
        bare._is_health_check_request(
            OrchestratorRequest(
                request_id="test-fb",
                request_type=RequestType.ADD,
                operation="forward_backward",
            )
        )
        is False
    )
    assert (
        bare._is_health_check_request(
            OrchestratorRequest(
                request_id="test-abort",
                request_type=RequestType.ABORT,
                operation="abort",
            )
        )
        is False
    )
    assert (
        bare._is_health_check_request(
            OrchestratorRequest(
                request_id="test-adapter",
                request_type=RequestType.UTILITY,
                operation="get_adapter_info",
            )
        )
        is False
    )

    # Methods exist
    assert callable(bare._is_health_check_request)
    assert callable(bare._handle_health_check_immediate)

    # Stats are readonly
    initial_stats = bare.scheduler.get_stats()
    for _ in range(10):
        bare.scheduler.get_stats()
    final_stats = bare.scheduler.get_stats()
    assert initial_stats["total_requests"] == final_stats["total_requests"]

    initial_proc_stats = bare.request_processor.get_stats()
    for _ in range(10):
        bare.request_processor.get_stats()
    final_proc_stats = bare.request_processor.get_stats()
    assert initial_proc_stats["total_operations"] == final_proc_stats["total_operations"]
