"""
Tests for Request Scheduler.

This test suite verifies the Scheduler and FIFO scheduling policy:
1. Request lifecycle (pending -> processing -> completed/failed)
2. Capacity limits (max_running_requests, max_pending_requests)
3. Request tracking and statistics
4. Abort/cancel operations
5. FIFO policy behavior
"""

import pytest

from xorl.server.orchestrator.scheduler import (
    FIFOPolicy,
    Scheduler,
)
from xorl.server.protocol.api_orchestrator import (
    OrchestratorRequest,
    RequestType,
)
from xorl.server.protocol.operations import EmptyData


pytestmark = [pytest.mark.cpu, pytest.mark.server]


@pytest.fixture
def create_request():
    """Factory for creating test requests."""
    counter = {"value": 0}

    def _create(operation: str = "forward_backward", **kwargs) -> OrchestratorRequest:
        counter["value"] += 1
        return OrchestratorRequest(
            request_id=f"req-{counter['value']}",
            request_type=RequestType.ADD,
            operation=operation,
            payload=EmptyData(),
        )

    return _create


@pytest.fixture
def scheduler():
    """Create a scheduler with default settings."""
    return Scheduler(
        policy=FIFOPolicy(),
        max_running_requests=2,
        max_pending_requests=10,
    )


def test_scheduler_lifecycle_policy(scheduler, create_request):
    """Test adding requests, FIFO dispatch, peek, and capacity limits."""
    # Add single and multiple
    request = create_request()
    assert scheduler.add_request(request) == request.request_id
    assert scheduler.get_pending_count() == 1 and scheduler.has_pending_requests()
    for _ in range(4):
        scheduler.add_request(create_request())
    assert scheduler.get_pending_count() == 5

    # Peek doesn't modify queue
    peeked = scheduler.peek_next_request()
    assert peeked is not None and scheduler.get_pending_count() == 5

    # FIFO order dispatch
    req1 = scheduler.get_next_request()
    assert req1.request_id == request.request_id
    assert req1.status == "processing" and req1.start_time is not None
    scheduler.get_next_request()
    assert scheduler.get_running_count() == 2

    # At capacity
    assert scheduler.get_next_request() is None and scheduler.get_pending_count() == 3

    # Complete frees slot
    scheduler.mark_completed(req1.request_id)
    assert scheduler.get_running_count() == 1
    assert scheduler.get_next_request() is not None

    # Empty queue
    s2 = Scheduler(policy=FIFOPolicy(), max_running_requests=2, max_pending_requests=10)
    assert s2.get_next_request() is None

    # Pending limit
    s3 = Scheduler(policy=FIFOPolicy(), max_running_requests=2, max_pending_requests=10)
    for _ in range(s3.max_pending_requests):
        s3.add_request(create_request())
    with pytest.raises(ValueError, match="Pending queue is full"):
        s3.add_request(create_request())
    assert s3.total_rejected == 1

    terminal_scheduler = Scheduler(
        policy=FIFOPolicy(),
        max_running_requests=2,
        max_pending_requests=10,
    )
    _assert_terminal_states_statistics_history_and_clear(terminal_scheduler, create_request)


def _assert_terminal_states_statistics_history_and_clear(scheduler, create_request):
    """Terminal transitions update scheduler state, statistics, and bounded history."""
    # Complete and fail
    requests = [create_request() for _ in range(5)]
    for req in requests:
        scheduler.add_request(req)
    req1 = scheduler.get_next_request()
    req2 = scheduler.get_next_request()
    scheduler.mark_completed(req1.request_id)
    assert scheduler.total_completed == 1 and scheduler.get_request_status(req1.request_id) == "completed"

    scheduler.mark_failed(req2.request_id, "Test error")
    assert scheduler.total_failed == 1 and scheduler.get_request_status(req2.request_id) == "failed"

    # Unknown IDs
    scheduler.mark_completed("unknown-id")
    scheduler.mark_failed("unknown-id", "error")
    assert scheduler.total_completed == 1 and scheduler.total_failed == 1

    # Abort pending
    req = create_request()
    scheduler.add_request(req)
    assert scheduler.abort_request(req.request_id) is True
    assert scheduler.total_aborted == 1

    # Abort a request that has actually entered the running state.
    running = scheduler.get_next_request()
    assert running is not None and running.status == "processing"
    assert scheduler.abort_request(running.request_id) is True
    assert scheduler.total_aborted == 2

    # Abort nonexistent
    assert scheduler.abort_request("nonexistent") is False
    stats = scheduler.get_stats()
    assert stats["policy"] == "FIFO"
    assert stats["total_completed"] == 1
    assert stats["total_failed"] == 1
    assert stats["total_aborted"] == 2

    # Clear all live and historical request state.
    scheduler.clear()
    assert scheduler.get_pending_count() == 0 and scheduler.get_running_count() == 0
    assert len(scheduler.completed_requests) == 0

    # Completed history remains bounded independently of lifetime counters.
    big = Scheduler()
    for i in range(1500):
        big.add_request(OrchestratorRequest(request_id=f"r-{i}", request_type=RequestType.ADD, operation="test"))
        big.get_next_request()
        big.mark_completed(f"r-{i}")
    assert len(big.completed_requests) == 1000 and big.total_completed == 1500
