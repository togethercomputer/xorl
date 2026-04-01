"""
Request Scheduler for Orchestrator.

This module provides the scheduling layer for Orchestrator, managing request
ordering, lifecycle tracking, and dispatching. The Scheduler sits between
Orchestrator's input queue and the Executor, determining which requests to
process and when.

Role in Orchestrator Architecture:
================================

The Scheduler is one of three core components in Orchestrator:

┌─────────────────────────────────────────────────────────────────────────┐
│                          ENGINE PROCESS                                  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                        Orchestrator                              │     │
│  │                                                                 │     │
│  │  Thread 1: Input          Thread 2: Main          Thread 3: Output │
│  │  ┌──────────┐            ┌──────────┐            ┌──────────┐ │     │
│  │  │ API Req  │──ADD──────>│Scheduler │──GET──────>│Executor  │ │     │
│  │  │ Listener │            │          │            │          │ │     │
│  │  └──────────┘            │  FIFO    │            │ Workers  │ │     │
│  │      ↑                   │  Policy  │            │  Coord   │ │     │
│  │      │                   └──────────┘            └──────────┘ │     │
│  │      │                        │                        │       │     │
│  │      │                   LIFECYCLE                  RESULTS    │     │
│  │      │                   TRACKING                              │     │
│  │      │                        │                        │       │     │
│  │  API Server            ┌──────────┐            ┌──────────┐    │     │
│  │   (DEALER)             │ Pending  │            │ Results  │    │     │
│  │                        │ Active   │            │  Queue   │    │     │
│  │                        │Completed │            └──────────┘    │     │
│  │                        └──────────┘                 │          │     │
│  │                                                     ↓          │     │
│  │                                              API Server        │     │
│  │                                               (PUSH)            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Request Lifecycle:
==================

Requests flow through the Scheduler with the following state transitions:

1. **PENDING**: Request arrives, added to policy queue
   - Scheduler.add_request() creates ScheduledRequest wrapper
   - Checks if pending queue is full (< max_pending_requests)
   - If full, raises ValueError and rejects the request
   - Policy decides queue position (FIFO: append to end)
   - Request waits until dispatched

2. **PROCESSING**: Request selected for execution
   - Orchestrator calls Scheduler.get_next_request()
   - Checks if at max running capacity (< max_running_requests)
   - If at capacity, returns None (request stays pending)
   - Policy returns next request (FIFO: front of queue)
   - Request moved to active_requests tracking
   - Timestamp recorded for latency tracking

3. **COMPLETED/FAILED**: Execution finishes
   - Orchestrator calls mark_completed() or mark_failed()
   - Request moved to completed_requests history
   - Statistics updated (total_completed, total_failed)
   - Processing time calculated
   - Running slot freed for next request

Example flow (max_running=2):
```
Time    │ Event                           │ Scheduler State
────────┼─────────────────────────────────┼─────────────────────────────
T+0ms   │ Request A arrives               │ Pending: [A]  Running: 0/2
        │ scheduler.add_request(A)        │
────────┼─────────────────────────────────┼─────────────────────────────
T+5ms   │ Request B arrives               │ Pending: [A, B]  Running: 0/2
        │ scheduler.add_request(B)        │
────────┼─────────────────────────────────┼─────────────────────────────
T+10ms  │ Engine ready, gets next request │ Pending: [B]  Running: 1/2
        │ A = scheduler.get_next_request()│ Active: [A]
────────┼─────────────────────────────────┼─────────────────────────────
T+15ms  │ Request C arrives               │ Pending: [B, C]  Running: 1/2
        │ scheduler.add_request(C)        │ Active: [A]
────────┼─────────────────────────────────┼─────────────────────────────
T+20ms  │ Engine gets next request        │ Pending: [C]  Running: 2/2
        │ B = scheduler.get_next_request()│ Active: [A, B]
────────┼─────────────────────────────────┼─────────────────────────────
T+25ms  │ Engine tries next request       │ Pending: [C]  Running: 2/2
        │ None = get_next_request()       │ Active: [A, B] (at capacity!)
        │ (returns None, at max capacity) │
────────┼─────────────────────────────────┼─────────────────────────────
T+50ms  │ A completes execution           │ Pending: [C]  Running: 1/2
        │ scheduler.mark_completed(A)     │ Active: [B]
        │                                 │ Completed: [A]
────────┼─────────────────────────────────┼─────────────────────────────
T+55ms  │ Engine gets next request        │ Pending: []  Running: 2/2
        │ C = scheduler.get_next_request()│ Active: [B, C]
        │                                 │ Completed: [A]
```

Integration with Orchestrator:
============================

The Scheduler is called from Orchestrator's main thread:

```python
# Orchestrator main loop (simplified)
def _process_engine_step(self):
    # 1. Check if scheduler has pending work
    if not self.scheduler.has_pending_requests():
        return

    # 2. Get next request according to policy
    scheduled_req = self.scheduler.get_next_request()
    if not scheduled_req:
        return

    # 3. Extract operation and request
    request = scheduled_req.request
    operation = scheduled_req.operation

    # 4. Delegate execution to appropriate handler
    try:
        if operation == "forward_backward":
            self._execute_forward_backward(request)
        elif operation == "optim_step":
            self._execute_optim_step(request)
        # ... other operations

        # 5. Mark successful completion
        self.scheduler.mark_completed(request.request_id)

    except Exception as e:
        # 6. Mark failure with error
        self.scheduler.mark_failed(request.request_id, str(e))
        self._send_error_output(request.request_id, str(e))
```

The Scheduler does NOT execute requests - it only decides ordering.
Execution is handled by Orchestrator (which delegates to Executor).

Threading Model:
===============

The Scheduler is single-threaded and NOT thread-safe:

- All Scheduler methods are called from Orchestrator's MAIN THREAD only
- No locks needed (single-threaded access)
- Input thread adds to Orchestrator's input queue (NOT directly to Scheduler)
- Orchestrator main thread moves requests: input_queue → Scheduler → Executor

Thread interaction:
```
Thread 1 (Input):
    while running:
        request = zmq_socket.recv()
        orchestrator.input_queue.put(request)  # Thread-safe queue

Thread 2 (Main):
    while running:
        # Pull from input queue
        if not orchestrator.input_queue.empty():
            request = orchestrator.input_queue.get()
            scheduler.add_request(request)  # Single-threaded

        # Process next request
        scheduled_req = scheduler.get_next_request()  # Single-threaded
        if scheduled_req:
            result = executor.execute(scheduled_req)
            scheduler.mark_completed(scheduled_req.request_id)
```

Scheduling Policies:
====================

The Scheduler uses a pluggable policy pattern. Currently implemented:

1. **FIFO (First-In-First-Out)** [DEFAULT]
   - Process requests in arrival order
   - O(1) enqueue and dequeue (collections.deque)
   - No starvation - every request eventually processed
   - Fair and predictable
   - Best for: Standard training workloads

Future policies (architecture supports, not yet implemented):

2. **Priority-Based**
   - Assign priority levels to requests
   - Higher priority processed first
   - Same priority → FIFO order
   - Use cases: Urgent health checks, critical checkpoint saves

3. **Batch Coalescing**
   - Group multiple small forward_backward requests
   - Process as single large batch
   - Improves GPU utilization
   - Use cases: High-throughput inference, small batch training

4. **Preemptive**
   - Abort long-running requests for urgent ones
   - Requires operation checkpointing/rollback
   - Use cases: Emergency checkpoint saves, graceful shutdown

5. **Load-Aware**
   - Consider worker load and queue depth
   - Route requests to least-loaded workers
   - Dynamic load balancing
   - Use cases: Multi-tenant systems, heterogeneous workers

To implement a new policy:
```python
class CustomPolicy(SchedulingPolicy):
    def add_request(self, scheduled_req: ScheduledRequest):
        # Add to internal data structure
        pass

    def get_next_request(self) -> Optional[ScheduledRequest]:
        # Return next request according to policy
        pass

    # ... implement other abstract methods

# Use custom policy
scheduler = Scheduler(policy=CustomPolicy())
```

Statistics and Monitoring:
=========================

The Scheduler tracks detailed metrics for monitoring:

```python
stats = scheduler.get_stats()
# Returns:
{
    "policy": "FIFO",
    "pending_requests": 5,           # Waiting to be processed
    "running_requests": 2,           # Currently being processed
    "active_requests": 2,            # Alias for running_requests
    "max_running_requests": 2,       # Maximum concurrent running requests
    "max_pending_requests": 100,     # Maximum queue size
    "total_requests": 1000,          # Lifetime total
    "total_completed": 980,          # Successfully completed
    "total_failed": 10,              # Failed with errors
    "total_aborted": 4,              # Aborted by client
    "total_rejected": 6,             # Rejected due to full queue
    "avg_pending_age_sec": 0.05,     # Average wait time
    "avg_processing_time_sec": 0.15, # Average execution time
}
```

These stats can be exposed via:
- Health check responses (orchestrator health check)
- Monitoring endpoints (if implemented)
- Debug logging

Usage Example:
=============

```python
from xorl.server.orchestrator.scheduler import Scheduler, FIFOPolicy
from xorl.server.protocol.api_orchestrator import OrchestratorRequest

# Initialize with capacity limits
scheduler = Scheduler(
    policy=FIFOPolicy(),
    max_running_requests=2,    # At most 2 concurrent running requests
    max_pending_requests=100,  # Reject if queue exceeds 100
)

# Add requests (called from Orchestrator input thread -> main thread)
request1 = OrchestratorRequest(
    request_type=RequestType.EXECUTE,
    data={"operation": "forward_backward", "batch_size": 32}
)
request2 = OrchestratorRequest(
    request_type=RequestType.EXECUTE,
    data={"operation": "optim_step", "lr": 0.001}
)

try:
    scheduler.add_request(request1)  # May raise ValueError if queue full
    scheduler.add_request(request2)
except ValueError as e:
    # Queue is full, reject the request
    send_error_to_client(str(e))

# Process requests (called from Orchestrator main thread)
while scheduler.has_pending_requests():
    # Get next request (respects max_running_requests)
    scheduled_req = scheduler.get_next_request()

    if not scheduled_req:
        # At max capacity, wait for a slot to free up
        break

    try:
        # Execute (delegated to Executor)
        result = execute_operation(scheduled_req)

        # Mark success (frees up a running slot)
        scheduler.mark_completed(scheduled_req.request_id)
    except Exception as e:
        # Mark failure (also frees up a slot)
        scheduler.mark_failed(scheduled_req.request_id, str(e))

# Check statistics
stats = scheduler.get_stats()
print(f"Running: {stats['running_requests']}/{stats['max_running_requests']}")
print(f"Pending: {stats['pending_requests']}/{stats['max_pending_requests']}")
```

Key Design Decisions:
====================

1. **Separation from Execution**: Scheduler only orders, doesn't execute
   - Keeps Scheduler simple and focused
   - Executor handles complex distributed coordination
   - Clear separation of concerns

2. **Single-threaded Access**: No locks needed
   - All calls from Orchestrator main thread
   - Simpler, faster, no race conditions
   - Thread-safe queues handle cross-thread communication

3. **Pluggable Policies**: Easy to extend scheduling strategies
   - Abstract base class (SchedulingPolicy)
   - Swap policies without changing Orchestrator
   - Future-proof for complex scheduling needs

4. **Request Metadata**: Rich tracking via ScheduledRequest
   - Arrival time, processing time, retries
   - Enables sophisticated policies (age-based, retry-aware)
   - Provides detailed statistics

5. **Lifecycle Tracking**: Comprehensive state management
   - pending → processing → completed/failed
   - Active requests tracked separately
   - Completed requests kept in history (ring buffer, maxlen=1000)
   - Enables debugging and monitoring
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from xorl.server.protocol.api_orchestrator import (
    OrchestratorRequest,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Request Metadata
# ============================================================================


@dataclass
class ScheduledRequest:
    """
    Wrapper for requests with scheduling metadata.

    Tracks timing and execution state for scheduling decisions.
    """

    request: OrchestratorRequest
    arrival_time: float = field(default_factory=time.time)
    retries: int = 0
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def request_id(self) -> str:
        """Get request ID."""
        return self.request.request_id

    @property
    def operation(self) -> str:
        """Get operation type."""
        return self.request.operation or "unknown"

    @property
    def age_seconds(self) -> float:
        """Get age of request in seconds."""
        return time.time() - self.arrival_time

    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time if started."""
        if self.start_time:
            end_time = self.completion_time or time.time()
            return end_time - self.start_time
        return None

    def mark_processing(self):
        """Mark request as processing."""
        self.status = "processing"
        self.start_time = time.time()

    def mark_completed(self):
        """Mark request as completed."""
        self.status = "completed"
        self.completion_time = time.time()

    def mark_failed(self, error: str):
        """Mark request as failed."""
        self.status = "failed"
        self.completion_time = time.time()
        self.error = error


# ============================================================================
# Scheduling Policies
# ============================================================================


class SchedulingPolicy(ABC):
    """
    Abstract base class for scheduling policies.

    Subclasses implement specific scheduling strategies (FIFO, priority, etc.).
    """

    @abstractmethod
    def add_request(self, scheduled_req: ScheduledRequest):
        """
        Add a request to the scheduler.

        Args:
            scheduled_req: Request with scheduling metadata
        """
        pass

    @abstractmethod
    def get_next_request(self) -> Optional[ScheduledRequest]:
        """
        Get the next request to process according to policy.

        Returns:
            Next ScheduledRequest to process, or None if queue empty
        """
        pass

    @abstractmethod
    def remove_request(self, request_id: str) -> bool:
        """
        Remove a request from the scheduler (for abort operations).

        Args:
            request_id: ID of request to remove

        Returns:
            True if removed, False if not found
        """
        pass

    @abstractmethod
    def peek_next_request(self) -> Optional[ScheduledRequest]:
        """
        Peek at the next request without removing it.

        Returns:
            Next ScheduledRequest, or None if queue empty
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get number of pending requests.

        Returns:
            Number of pending requests in scheduler
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Check if scheduler has any pending requests.

        Returns:
            True if empty, False otherwise
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all pending requests."""
        pass

    @abstractmethod
    def get_policy_name(self) -> str:
        """
        Get name of the scheduling policy.

        Returns:
            Policy name string
        """
        pass


class FIFOPolicy(SchedulingPolicy):
    """
    First-In-First-Out (FIFO) scheduling policy.

    Processes requests in the order they arrive. Simple, fair, and predictable.
    This is the default policy for Orchestrator.

    Features:
    - Arrival-order processing
    - O(1) enqueue and dequeue
    - No starvation (every request eventually processed)
    - Suitable for most training workloads
    """

    def __init__(self):
        """Initialize FIFO policy."""
        self.queue: Deque[ScheduledRequest] = deque()
        self.request_map: Dict[str, ScheduledRequest] = {}
        logger.info("Initialized FIFO scheduling policy")

    def add_request(self, scheduled_req: ScheduledRequest):
        """
        Add request to FIFO queue.

        Args:
            scheduled_req: Request with scheduling metadata
        """
        self.queue.append(scheduled_req)
        self.request_map[scheduled_req.request_id] = scheduled_req
        logger.debug(
            f"Added request {scheduled_req.request_id} to FIFO queue "
            f"(operation={scheduled_req.operation}, queue_size={len(self.queue)})"
        )

    def get_next_request(self) -> Optional[ScheduledRequest]:
        """
        Get next request from front of FIFO queue.

        Returns:
            Next ScheduledRequest, or None if queue empty
        """
        if not self.queue:
            return None

        scheduled_req = self.queue.popleft()
        del self.request_map[scheduled_req.request_id]

        logger.debug(
            f"Dequeued request {scheduled_req.request_id} from FIFO queue "
            f"(operation={scheduled_req.operation}, "
            f"age={scheduled_req.age_seconds:.2f}s, "
            f"remaining={len(self.queue)})"
        )

        return scheduled_req

    def remove_request(self, request_id: str) -> bool:
        """
        Remove request from queue (for abort operations).

        Args:
            request_id: ID of request to remove

        Returns:
            True if removed, False if not found
        """
        if request_id not in self.request_map:
            return False

        scheduled_req = self.request_map[request_id]
        self.queue.remove(scheduled_req)
        del self.request_map[request_id]

        logger.info(f"Removed request {request_id} from FIFO queue")
        return True

    def peek_next_request(self) -> Optional[ScheduledRequest]:
        """
        Peek at next request without removing it.

        Returns:
            Next ScheduledRequest, or None if queue empty
        """
        return self.queue[0] if self.queue else None

    def size(self) -> int:
        """
        Get number of pending requests.

        Returns:
            Queue size
        """
        return len(self.queue)

    def is_empty(self) -> bool:
        """
        Check if queue is empty.

        Returns:
            True if empty
        """
        return len(self.queue) == 0

    def clear(self):
        """Clear all pending requests."""
        count = len(self.queue)
        self.queue.clear()
        self.request_map.clear()
        logger.info(f"Cleared FIFO queue ({count} requests removed)")

    def get_policy_name(self) -> str:
        """Get policy name."""
        return "FIFO"


# ============================================================================
# Scheduler
# ============================================================================


class Scheduler:
    """
    Request scheduler for Orchestrator.

    Manages incoming requests and determines processing order based on
    the configured scheduling policy. Enforces limits on running and pending
    requests to prevent overload.

    Features:
    - Pluggable scheduling policies
    - Request tracking and metadata
    - Statistics and monitoring
    - Abort/cancel support
    - Configurable running/pending request limits

    Example:
        >>> scheduler = Scheduler(
        ...     policy=SeqIdAwareFIFOPolicy(),
        ...     max_running_requests=16,
        ...     max_pending_requests=128
        ... )
        >>> scheduler.add_request(request)  # May raise ValueError if pending queue full
        >>> next_req = scheduler.get_next_request()  # Returns None if at max running
        >>> scheduler.mark_completed(next_req.request_id)
    """

    def __init__(
        self,
        policy: Optional[SchedulingPolicy] = None,
        max_running_requests: int = 16,
        max_pending_requests: int = 128,
    ):
        """
        Initialize scheduler.

        Args:
            policy: Scheduling policy to use (default: FIFOPolicy)
            max_running_requests: Maximum number of concurrent running requests (default: 16)
            max_pending_requests: Maximum number of pending requests in queue (default: 128)
                If exceeded, new requests will be rejected with an error.
        """
        self.policy = policy or FIFOPolicy()
        self.max_running_requests = max_running_requests
        self.max_pending_requests = max_pending_requests

        # Request tracking
        self.active_requests: Dict[str, ScheduledRequest] = {}
        self.completed_requests: Deque[ScheduledRequest] = deque(maxlen=1000)

        # Statistics
        self.total_requests = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_aborted = 0
        self.total_rejected = 0  # New: track rejected requests

        logger.info(
            f"Scheduler initialized with policy: {self.policy.get_policy_name()}, "
            f"max_running_requests={max_running_requests}, "
            f"max_pending_requests={max_pending_requests}"
        )

    # ========================================================================
    # Request Management
    # ========================================================================

    def add_request(self, request: OrchestratorRequest) -> str:
        """
        Add a new request to the scheduler.

        Args:
            request: Request to schedule

        Returns:
            Request ID

        Raises:
            ValueError: If pending queue is full (exceeds max_pending_requests)
        """
        # Check if pending queue is full
        current_pending = self.get_pending_count()
        if current_pending >= self.max_pending_requests:
            self.total_rejected += 1
            error_msg = (
                f"Pending queue is full ({current_pending}/{self.max_pending_requests}). "
                f"Cannot accept new request. Please try again later."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        scheduled_req = ScheduledRequest(request=request)
        self.policy.add_request(scheduled_req)
        self.total_requests += 1

        logger.debug(
            f"Added request {scheduled_req.request_id} to scheduler "
            f"(operation={scheduled_req.operation}, "
            f"pending={current_pending + 1}/{self.max_pending_requests}, "
            f"running={self.get_running_count()}/{self.max_running_requests})"
        )

        return scheduled_req.request_id

    def get_next_request(self) -> Optional[ScheduledRequest]:
        """
        Get next request to process according to policy.

        Moves request from policy queue to active_requests if there is capacity
        (running requests < max_running_requests).

        Returns:
            Next ScheduledRequest to process, or None if queue empty or at capacity
        """
        # Check if we're at max running capacity
        current_running = self.get_running_count()
        if current_running >= self.max_running_requests:
            logger.debug(
                f"Cannot dispatch request: at max running capacity ({current_running}/{self.max_running_requests})"
            )
            return None

        scheduled_req = self.policy.get_next_request()

        if scheduled_req:
            scheduled_req.mark_processing()
            self.active_requests[scheduled_req.request_id] = scheduled_req

            logger.debug(
                f"Dispatching request {scheduled_req.request_id} for processing "
                f"(operation={scheduled_req.operation}, "
                f"age={scheduled_req.age_seconds:.2f}s, "
                f"running={len(self.active_requests)}/{self.max_running_requests}, "
                f"pending={self.get_pending_count()})"
            )

        return scheduled_req

    def peek_next_request(self) -> Optional[ScheduledRequest]:
        """
        Peek at next request without removing it.

        Returns:
            Next ScheduledRequest, or None if queue empty
        """
        return self.policy.peek_next_request()

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a pending or active request.

        Args:
            request_id: ID of request to abort

        Returns:
            True if aborted, False if not found
        """
        # Try to remove from policy queue (pending)
        if self.policy.remove_request(request_id):
            self.total_aborted += 1
            logger.info(f"Aborted pending request {request_id}")
            return True

        # Try to remove from active requests (processing)
        if request_id in self.active_requests:
            scheduled_req = self.active_requests[request_id]
            scheduled_req.mark_failed("Aborted by client")
            del self.active_requests[request_id]
            self.completed_requests.append(scheduled_req)
            self.total_aborted += 1
            logger.info(f"Aborted active request {request_id}")
            return True

        logger.warning(f"Request {request_id} not found for abort")
        return False

    def mark_completed(self, request_id: str):
        """
        Mark a request as completed.

        Args:
            request_id: ID of request that completed
        """
        if request_id not in self.active_requests:
            logger.warning(f"Cannot mark unknown request {request_id} as completed")
            return

        scheduled_req = self.active_requests[request_id]
        scheduled_req.mark_completed()

        del self.active_requests[request_id]
        self.completed_requests.append(scheduled_req)
        self.total_completed += 1

        logger.debug(
            f"Request {request_id} completed "
            f"(operation={scheduled_req.operation}, "
            f"processing_time={scheduled_req.processing_time:.3f}s)"
        )

    def mark_failed(self, request_id: str, error: str):
        """
        Mark a request as failed.

        Args:
            request_id: ID of request that failed
            error: Error message
        """
        if request_id not in self.active_requests:
            logger.warning(f"Cannot mark unknown request {request_id} as failed")
            return

        scheduled_req = self.active_requests[request_id]
        scheduled_req.mark_failed(error)

        del self.active_requests[request_id]
        self.completed_requests.append(scheduled_req)
        self.total_failed += 1

        logger.error(f"Request {request_id} failed (operation={scheduled_req.operation}, error={error})")

    # ========================================================================
    # Query Methods
    # ========================================================================

    def has_pending_requests(self) -> bool:
        """
        Check if there are pending requests to process.

        Returns:
            True if pending requests exist
        """
        return not self.policy.is_empty()

    def get_pending_count(self) -> int:
        """
        Get number of pending requests.

        Returns:
            Number of requests waiting to be processed
        """
        return self.policy.size()

    def get_active_count(self) -> int:
        """
        Get number of active (processing) requests.

        Returns:
            Number of requests currently being processed
        """
        return len(self.active_requests)

    def get_running_count(self) -> int:
        """
        Get number of running (processing) requests.

        Alias for get_active_count() with clearer naming.

        Returns:
            Number of requests currently running/being processed
        """
        return self.get_active_count()

    def get_request_status(self, request_id: str) -> Optional[str]:
        """
        Get status of a request.

        Args:
            request_id: Request ID

        Returns:
            Status string ("pending", "processing", "completed", "failed") or None
        """
        # Check active requests
        if request_id in self.active_requests:
            return self.active_requests[request_id].status

        # Check completed requests
        for req in self.completed_requests:
            if req.request_id == request_id:
                return req.status

        return None

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary with scheduler stats
        """
        pending_ages = []
        active_times = []

        # Collect pending request ages
        current_req = self.policy.peek_next_request()
        if current_req:
            # For FIFO, estimate from queue position
            # For other policies, would need to iterate
            pending_ages.append(current_req.age_seconds)

        # Collect active processing times
        for req in self.active_requests.values():
            if req.processing_time:
                active_times.append(req.processing_time)

        avg_pending_age = sum(pending_ages) / len(pending_ages) if pending_ages else 0.0
        avg_processing_time = sum(active_times) / len(active_times) if active_times else 0.0

        return {
            "policy": self.policy.get_policy_name(),
            "pending_requests": self.get_pending_count(),
            "running_requests": self.get_running_count(),
            "active_requests": self.get_active_count(),  # Alias for running_requests
            "max_running_requests": self.max_running_requests,
            "max_pending_requests": self.max_pending_requests,
            "total_requests": self.total_requests,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_aborted": self.total_aborted,
            "total_rejected": self.total_rejected,
            "avg_pending_age_sec": avg_pending_age,
            "avg_processing_time_sec": avg_processing_time,
        }

    def clear(self):
        """Clear all requests and reset statistics."""
        self.policy.clear()
        self.active_requests.clear()
        self.completed_requests.clear()
        logger.info("Scheduler cleared")

    def __repr__(self) -> str:
        return (
            f"Scheduler(policy={self.policy.get_policy_name()}, "
            f"pending={self.get_pending_count()}, "
            f"active={self.get_active_count()})"
        )


# ============================================================================
# Seq_id-Aware Scheduling Policy
# ============================================================================


class SeqIdAwareFIFOPolicy(SchedulingPolicy):
    """
    Scheduling policy that sorts requests by seq_id per model_id.

    This policy ensures that requests are processed in seq_id order per model_id.
    All requests are immediately added to a sorted structure, and get_next_request()
    returns the request with the lowest seq_id for any model.

    This is critical for ensuring that forward_backward completes before
    optim_step, even if the requests arrive out of network order.

    Features:
    - Sorts requests by seq_id (lowest first) per model_id
    - Falls back to arrival order (FIFO) for requests without seq_id
    - No buffering or expected_seq_id tracking - just sorting
    - Simple and predictable behavior
    """

    def __init__(self):
        """Initialize SeqIdAwareFIFO policy."""
        # Requests with seq_id: model_id -> {seq_id -> ScheduledRequest}
        # Using dict for O(1) lookup by seq_id
        self.seq_id_requests: Dict[str, Dict[int, ScheduledRequest]] = {}

        # Requests without seq_id (FIFO order)
        self.fifo_queue: Deque[ScheduledRequest] = deque()

        # Request map for quick lookup (by request_id)
        self.request_map: Dict[str, ScheduledRequest] = {}

        logger.info("Initialized SeqIdAwareFIFO scheduling policy (sort-based)")

    def add_request(self, scheduled_req: ScheduledRequest):
        """
        Add request to the scheduler.

        Requests with seq_id are stored in a sorted structure per model_id.
        Requests without seq_id go to a FIFO queue.

        Args:
            scheduled_req: Request with scheduling metadata
        """
        request_id = scheduled_req.request_id
        model_id = getattr(scheduled_req.request.payload, "model_id", None) or "default"
        seq_id = scheduled_req.request.seq_id

        self.request_map[request_id] = scheduled_req

        if seq_id is None:
            # No seq_id - use FIFO order
            self.fifo_queue.append(scheduled_req)
            logger.debug(
                f"Added request {request_id} (operation={scheduled_req.operation}) "
                f"to FIFO queue (no seq_id), queue_size={len(self.fifo_queue)}"
            )
        else:
            # Has seq_id - add to sorted structure
            if model_id not in self.seq_id_requests:
                self.seq_id_requests[model_id] = {}

            self.seq_id_requests[model_id][seq_id] = scheduled_req
            logger.debug(
                f"Added request {request_id} (seq_id={seq_id}, operation={scheduled_req.operation}) "
                f"for model {model_id}, pending_for_model={len(self.seq_id_requests[model_id])}"
            )

    def get_next_request(self) -> Optional[ScheduledRequest]:
        """
        Get next request - lowest seq_id across all models, or FIFO if no seq_id requests.

        Returns:
            Next ScheduledRequest, or None if empty
        """
        # First, try to get the request with lowest seq_id across all models
        best_req = None
        best_model_id = None
        best_seq_id = None

        for model_id, seq_map in self.seq_id_requests.items():
            if seq_map:
                # Find minimum seq_id for this model
                min_seq_id = min(seq_map.keys())
                if best_seq_id is None or min_seq_id < best_seq_id:
                    best_seq_id = min_seq_id
                    best_model_id = model_id
                    best_req = seq_map[min_seq_id]

        if best_req is not None:
            # Remove from seq_id_requests
            del self.seq_id_requests[best_model_id][best_seq_id]
            if not self.seq_id_requests[best_model_id]:
                del self.seq_id_requests[best_model_id]
            del self.request_map[best_req.request_id]

            logger.debug(
                f"Dequeued request {best_req.request_id} (seq_id={best_seq_id}, "
                f"operation={best_req.operation}) for model {best_model_id}"
            )
            return best_req

        # No seq_id requests, try FIFO queue
        if self.fifo_queue:
            scheduled_req = self.fifo_queue.popleft()
            del self.request_map[scheduled_req.request_id]

            logger.debug(
                f"Dequeued request {scheduled_req.request_id} (operation={scheduled_req.operation}) "
                f"from FIFO queue, remaining={len(self.fifo_queue)}"
            )
            return scheduled_req

        return None

    def remove_request(self, request_id: str) -> bool:
        """
        Remove request from scheduler (for abort/cancel operations).

        Args:
            request_id: ID of request to remove

        Returns:
            True if removed, False if not found
        """
        if request_id not in self.request_map:
            return False

        scheduled_req = self.request_map[request_id]
        model_id = getattr(scheduled_req.request.payload, "model_id", None) or "default"
        seq_id = scheduled_req.request.seq_id

        del self.request_map[request_id]

        if seq_id is None:
            # Remove from FIFO queue
            if scheduled_req in self.fifo_queue:
                self.fifo_queue.remove(scheduled_req)
                logger.info(f"Removed request {request_id} from FIFO queue")
                return True
        else:
            # Remove from seq_id structure
            if model_id in self.seq_id_requests and seq_id in self.seq_id_requests[model_id]:
                del self.seq_id_requests[model_id][seq_id]
                if not self.seq_id_requests[model_id]:
                    del self.seq_id_requests[model_id]
                logger.info(f"Removed request {request_id} (seq_id={seq_id}) from scheduler")
                return True

        return False

    def peek_next_request(self) -> Optional[ScheduledRequest]:
        """
        Peek at next request without removing it.

        Returns:
            Next ScheduledRequest, or None if empty
        """
        # Check seq_id requests first
        best_req = None
        best_seq_id = None

        for model_id, seq_map in self.seq_id_requests.items():
            if seq_map:
                min_seq_id = min(seq_map.keys())
                if best_seq_id is None or min_seq_id < best_seq_id:
                    best_seq_id = min_seq_id
                    best_req = seq_map[min_seq_id]

        if best_req is not None:
            return best_req

        # Check FIFO queue
        if self.fifo_queue:
            return self.fifo_queue[0]

        return None

    def size(self) -> int:
        """
        Get number of pending requests.

        Returns:
            Total number of pending requests
        """
        seq_id_count = sum(len(seq_map) for seq_map in self.seq_id_requests.values())
        return seq_id_count + len(self.fifo_queue)

    def is_empty(self) -> bool:
        """
        Check if scheduler has any pending requests.

        Returns:
            True if empty
        """
        return len(self.fifo_queue) == 0 and not self.seq_id_requests

    def clear(self):
        """Clear all pending requests."""
        seq_id_count = sum(len(seq_map) for seq_map in self.seq_id_requests.values())
        fifo_count = len(self.fifo_queue)

        self.seq_id_requests.clear()
        self.fifo_queue.clear()
        self.request_map.clear()

        logger.info(f"Cleared SeqIdAwareFIFO queue ({seq_id_count} seq_id + {fifo_count} FIFO requests removed)")

    def get_policy_name(self) -> str:
        """Get policy name."""
        return "SeqIdAwareFIFO"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the scheduler state.

        Returns:
            Dictionary with scheduler statistics
        """
        seq_id_by_model = {model_id: sorted(seq_map.keys()) for model_id, seq_map in self.seq_id_requests.items()}
        return {
            "seq_id_requests": sum(len(seq_map) for seq_map in self.seq_id_requests.values()),
            "fifo_queue_size": len(self.fifo_queue),
            "total_pending": self.size(),
            "seq_ids_by_model": seq_id_by_model,
        }


# ============================================================================
# Future Scheduling Policies (Placeholder Implementations)
# ============================================================================
