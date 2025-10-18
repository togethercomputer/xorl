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

pytestmark = [pytest.mark.cpu, pytest.mark.server]
import time

from xorl.server.orchestrator.scheduler import (
    Scheduler,
    FIFOPolicy,
    ScheduledRequest,
    SchedulingPolicy,
)
from xorl.server.protocol.api_orchestrator import (
    OrchestratorRequest,
    RequestType,
)
from xorl.server.protocol.operations import EmptyData


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


def test_initialization_and_fifo_basics():
    """Test scheduler init (default/custom), FIFO policy, and repr."""
    s = Scheduler()
    assert s.max_running_requests == 16
    assert s.max_pending_requests == 128
    assert s.get_pending_count() == 0 and s.get_running_count() == 0
    assert s.total_requests == 0 and s.total_completed == 0

    s2 = Scheduler(max_running_requests=5, max_pending_requests=50)
    assert s2.max_running_requests == 5

    policy = FIFOPolicy()
    assert policy.get_policy_name() == "FIFO" and policy.is_empty() and policy.size() == 0

    s3 = Scheduler(policy=FIFOPolicy(), max_running_requests=2, max_pending_requests=10)
    assert "Scheduler" in repr(s3) and "FIFO" in repr(s3)


def test_add_dispatch_peek_and_capacity(scheduler, create_request):
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
    req2 = scheduler.get_next_request()
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


def test_completion_failure_abort_and_unknown(scheduler, create_request):
    """Test marking completed/failed, abort pending/running/nonexistent, and unknown IDs."""
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

    # Abort running
    req2 = create_request()
    scheduler.add_request(req2)
    scheduler.get_next_request()
    assert scheduler.abort_request(req2.request_id) is True
    assert scheduler.total_aborted == 2

    # Abort nonexistent
    assert scheduler.abort_request("nonexistent") is False


def test_stats_history_and_clear(create_request):
    """Test statistics, completed history limit, and clear."""
    scheduler = Scheduler(policy=FIFOPolicy(), max_running_requests=2, max_pending_requests=10)
    stats = scheduler.get_stats()
    assert stats["policy"] == "FIFO" and stats["total_requests"] == 0

    for _ in range(5):
        scheduler.add_request(create_request())
    req1 = scheduler.get_next_request()
    scheduler.mark_completed(req1.request_id)
    stats = scheduler.get_stats()
    assert stats["pending_requests"] == 4 and stats["total_completed"] == 1

    # Clear
    scheduler.clear()
    assert scheduler.get_pending_count() == 0 and scheduler.get_running_count() == 0

    # History limit
    big = Scheduler()
    for i in range(1500):
        big.add_request(OrchestratorRequest(request_id=f"r-{i}", request_type=RequestType.ADD, operation="test"))
        big.get_next_request()
        big.mark_completed(f"r-{i}")
    assert len(big.completed_requests) == 1000 and big.total_completed == 1500


def test_scheduled_request_lifecycle(create_request):
    """Test ScheduledRequest creation, age, processing, completion, and failure."""
    req = ScheduledRequest(request=create_request(operation="forward_backward"))
    assert req.status == "pending" and req.start_time is None

    time.sleep(0.1)
    assert req.age_seconds >= 0.1

    req.mark_processing()
    assert req.status == "processing" and req.start_time is not None

    time.sleep(0.05)
    req.mark_completed()
    assert req.status == "completed" and req.processing_time >= 0.05

    # Failed
    req2 = ScheduledRequest(request=create_request())
    req2.mark_processing()
    req2.mark_failed("Test error")
    assert req2.status == "failed" and req2.error == "Test error"


def test_fifo_policy_order_remove_and_clear():
    """Test FIFO ordering, remove, and clear operations."""
    policy = FIFOPolicy()
    for i in range(5):
        policy.add_request(ScheduledRequest(request=OrchestratorRequest(
            request_id=f"req-{i}", request_type=RequestType.ADD, operation="test",
        )))

    for i in range(5):
        assert policy.get_next_request().request_id == f"req-{i}"

    for i in range(5):
        policy.add_request(ScheduledRequest(request=OrchestratorRequest(
            request_id=f"r2-{i}", request_type=RequestType.ADD, operation="test",
        )))
    assert policy.remove_request("r2-0") is True and policy.size() == 4
    assert policy.get_next_request().request_id == "r2-1"

    policy.clear()
    assert policy.is_empty()
