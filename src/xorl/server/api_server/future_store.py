"""
FutureStore - Key-value store for two-phase request pattern results.

This module provides a storage system for async request results with:
- Session-based organization (grouped by model_id)
- TTL-based automatic expiration
- Separate status tracking from result storage
- Thread-safe operations

Design:
-------
The store uses two main data structures:
1. _entries: Dict[request_id, FutureEntry] - Main storage for all futures
2. _by_model: Dict[model_id, Set[request_id]] - Index for session-based cleanup

When a session ends (model_id cleanup), all associated futures are removed.
Individual futures expire after their TTL and are cleaned up lazily.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FutureStatus(Enum):
    """Status of a future in the store."""

    PENDING = "pending"  # Waiting to be processed
    PROCESSING = "processing"  # Currently being processed
    COMPLETED = "completed"  # Completed successfully, result available
    FAILED = "failed"  # Failed with error
    EXPIRED = "expired"  # TTL exceeded, entry will be cleaned up


@dataclass
class FutureEntry:
    """A single future entry in the store.

    Attributes:
        request_id: Unique identifier for this future
        model_id: Associated model/session ID
        request_type: Type of request (e.g., "forward_backward", "optim_step")
        status: Current status of the future
        created_at: Timestamp when the future was created
        expires_at: Timestamp when this entry expires
        result: The result data (when completed)
        error: Error message (when failed)
        error_category: Error category ("unknown", "server", "user")
        _completion_event: Event signaled when processing completes (for long polling)
    """

    request_id: str
    model_id: str
    request_type: str
    status: FutureStatus = FutureStatus.PENDING
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # Set by store based on TTL
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_category: str = "unknown"
    _completion_event: Optional[asyncio.Event] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the completion event after dataclass init."""
        if self._completion_event is None:
            self._completion_event = asyncio.Event()

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() > self.expires_at

    def is_terminal(self) -> bool:
        """Check if this entry is in a terminal state (completed, failed, or expired)."""
        return self.status in (FutureStatus.COMPLETED, FutureStatus.FAILED, FutureStatus.EXPIRED)

    def signal_completion(self):
        """Signal that processing is complete (success or failure)."""
        if self._completion_event:
            self._completion_event.set()

    async def wait_for_completion(self, timeout: float) -> bool:
        """Wait for processing to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if completed within timeout, False if timed out
        """
        if self._completion_event is None:
            return self.is_terminal()

        if self.is_terminal():
            return True

        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class FutureStore:
    """Key-value store for async request futures with session support.

    This store manages futures for the two-phase request pattern:
    1. Phase 1: Client submits request, server creates future entry
    2. Phase 2: Client polls for result, server returns status/result

    Features:
    - TTL-based expiration for automatic cleanup
    - Session-based grouping for bulk cleanup when model/session ends
    - Concurrent processing limits
    - Background cleanup task

    Args:
        default_ttl: Default time-to-live for entries in seconds (default: 1 hour)
        max_concurrent: Maximum number of concurrent processing tasks
        cleanup_interval: Interval for background cleanup in seconds

    Example:
        >>> store = FutureStore(default_ttl=3600)
        >>> await store.start()
        >>>
        >>> # Phase 1: Create future
        >>> request_id = await store.create(
        ...     model_id="model-123",
        ...     request_type="forward_backward",
        ...     process_fn=my_process_fn,
        ...     request_data={"data": [...]},
        ... )
        >>>
        >>> # Phase 2: Get result
        >>> entry = store.get(request_id)
        >>> if entry.status == FutureStatus.COMPLETED:
        ...     return entry.result
        >>> elif entry.status == FutureStatus.PENDING:
        ...     return TryAgainResponse(...)
    """

    def __init__(
        self,
        default_ttl: float = 3600.0,  # 1 hour
        max_concurrent: int = 10,
        cleanup_interval: float = 60.0,  # 1 minute
    ):
        self.default_ttl = default_ttl
        self.max_concurrent = max_concurrent
        self.cleanup_interval = cleanup_interval

        # Main storage: request_id -> FutureEntry
        self._entries: Dict[str, FutureEntry] = {}

        # Index by model_id for session-based cleanup
        self._by_model: Dict[str, Set[str]] = {}

        # Locks for thread safety
        self._lock = asyncio.Lock()

        # Semaphore for limiting concurrent processing
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Queue state for rate limiting feedback
        self._queue_state = "active"
        self._queue_state_reason: Optional[str] = None

        logger.info(
            f"FutureStore initialized: default_ttl={default_ttl}s, "
            f"max_concurrent={max_concurrent}"
        )

    async def start(self):
        """Start the store and background cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("FutureStore started")

    async def stop(self):
        """Stop the store and cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("FutureStore stopped")

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"future_{uuid.uuid4().hex[:12]}"

    async def create(
        self,
        model_id: str,
        request_type: str,
        process_fn: Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]],
        request_data: Dict[str, Any],
        ttl: Optional[float] = None,
    ) -> str:
        """Create a new future and start processing.

        Args:
            model_id: Model/session ID this future belongs to
            request_type: Type of request (for logging/debugging)
            process_fn: Async function to process the request
            request_data: Data to pass to process_fn
            ttl: Optional custom TTL (uses default_ttl if not specified)

        Returns:
            Unique request_id for retrieving the result
        """
        request_id = self._generate_request_id()
        ttl = ttl or self.default_ttl

        entry = FutureEntry(
            request_id=request_id,
            model_id=model_id,
            request_type=request_type,
            created_at=time.time(),
            expires_at=time.time() + ttl,
        )

        async with self._lock:
            self._entries[request_id] = entry

            # Add to model index
            if model_id not in self._by_model:
                self._by_model[model_id] = set()
            self._by_model[model_id].add(request_id)

        # Start processing in background
        asyncio.create_task(
            self._process(request_id, process_fn, request_data)
        )

        logger.debug(
            f"Future created: request_id={request_id}, model_id={model_id}, "
            f"type={request_type}, expires_at={entry.expires_at}"
        )

        return request_id

    async def _process(
        self,
        request_id: str,
        process_fn: Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]],
        request_data: Dict[str, Any],
    ):
        """Process a request in the background.

        Uses semaphore to limit concurrent processing.
        Signals completion event when done (for long polling).
        """
        async with self._semaphore:
            entry = self._entries.get(request_id)
            if not entry:
                logger.warning(f"Entry {request_id} not found for processing")
                return

            # Check if already expired before processing
            if entry.is_expired():
                entry.status = FutureStatus.EXPIRED
                entry.signal_completion()
                logger.debug(f"Entry {request_id} expired before processing")
                return

            entry.status = FutureStatus.PROCESSING
            logger.debug(f"Processing {request_id}")

            try:
                result = await process_fn(request_data)
                entry.status = FutureStatus.COMPLETED
                entry.result = result
                logger.debug(f"Completed {request_id}")

            except Exception as e:
                entry.status = FutureStatus.FAILED
                entry.error = str(e)

                # Determine error category
                error_str = str(e).lower()
                if any(x in error_str for x in ["invalid", "bad request", "validation", "missing"]):
                    entry.error_category = "user"
                else:
                    entry.error_category = "server"

                logger.error(f"Failed {request_id}: {e}")

            finally:
                # Always signal completion (success or failure)
                entry.signal_completion()

    def get(self, request_id: str) -> Optional[FutureEntry]:
        """Get a future entry by request_id.

        Returns None if not found or expired.
        Marks expired entries as EXPIRED status.

        Args:
            request_id: The request ID to look up

        Returns:
            FutureEntry if found and not expired, None otherwise
        """
        entry = self._entries.get(request_id)
        if entry is None:
            return None

        # Check expiration
        if entry.is_expired() and entry.status not in (FutureStatus.EXPIRED,):
            entry.status = FutureStatus.EXPIRED
            logger.debug(f"Entry {request_id} marked as expired on access")

        return entry

    async def wait_for_result(
        self,
        request_id: str,
        timeout: float = 45.0,
    ) -> Optional[FutureEntry]:
        """Wait for a future to complete (long polling).

        This method blocks until either:
        - The future completes (success or failure)
        - The timeout is reached
        - The entry is not found

        Args:
            request_id: The request ID to wait for
            timeout: Maximum time to wait in seconds (default: 45s like Tinker)

        Returns:
            FutureEntry if found, None if not found.
            Check entry.is_terminal() to see if it completed or timed out.
        """
        entry = self.get(request_id)
        if entry is None:
            return None

        # If already terminal, return immediately
        if entry.is_terminal():
            return entry

        # Wait for completion with timeout
        completed = await entry.wait_for_completion(timeout)

        # Return the entry regardless of whether it completed
        # Caller checks entry.is_terminal() to determine if result is ready
        return entry

    def get_status(self, request_id: str) -> Optional[FutureStatus]:
        """Get just the status of a future.

        Args:
            request_id: The request ID to look up

        Returns:
            FutureStatus if found, None if not found
        """
        entry = self.get(request_id)
        return entry.status if entry else None

    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed future.

        Args:
            request_id: The request ID to look up

        Returns:
            Result dict if completed, None otherwise
        """
        entry = self.get(request_id)
        if entry and entry.status == FutureStatus.COMPLETED:
            return entry.result
        return None

    def get_error(self, request_id: str) -> Optional[tuple[str, str]]:
        """Get the error info of a failed future.

        Args:
            request_id: The request ID to look up

        Returns:
            Tuple of (error_message, error_category) if failed, None otherwise
        """
        entry = self.get(request_id)
        if entry and entry.status == FutureStatus.FAILED:
            return (entry.error or "Unknown error", entry.error_category)
        return None

    async def delete(self, request_id: str) -> bool:
        """Delete a future entry.

        Args:
            request_id: The request ID to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry:
                # Remove from model index
                if entry.model_id in self._by_model:
                    self._by_model[entry.model_id].discard(request_id)
                    if not self._by_model[entry.model_id]:
                        del self._by_model[entry.model_id]
                return True
            return False

    async def delete_by_model(self, model_id: str) -> int:
        """Delete all futures for a model/session.

        Called when a session ends to clean up all associated futures.

        Args:
            model_id: The model ID whose futures should be deleted

        Returns:
            Number of entries deleted
        """
        async with self._lock:
            request_ids = self._by_model.pop(model_id, set())
            count = len(request_ids)

            for request_id in request_ids:
                self._entries.pop(request_id, None)

            if count > 0:
                logger.info(f"Deleted {count} futures for model_id={model_id}")

            return count

    def list_by_model(self, model_id: str) -> List[FutureEntry]:
        """List all futures for a model/session.

        Args:
            model_id: The model ID to list futures for

        Returns:
            List of FutureEntry objects
        """
        request_ids = self._by_model.get(model_id, set())
        entries = []
        for request_id in request_ids:
            entry = self.get(request_id)
            if entry:
                entries.append(entry)
        return entries

    def get_queue_state(self) -> tuple[str, Optional[str]]:
        """Get current queue state for TryAgainResponse.

        Returns:
            Tuple of (queue_state, queue_state_reason)
        """
        return self._queue_state, self._queue_state_reason

    def set_queue_state(self, state: str, reason: Optional[str] = None):
        """Set queue state for rate limiting feedback.

        Args:
            state: One of "active", "paused_capacity", "paused_rate_limit"
            reason: Optional human-readable reason
        """
        self._queue_state = state
        self._queue_state_reason = reason

    async def _cleanup_loop(self):
        """Background task to periodically clean up expired entries."""
        logger.info(f"Cleanup loop started (interval={self.cleanup_interval}s)")

        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_expired(self):
        """Remove expired entries from the store."""
        now = time.time()
        to_delete = []

        async with self._lock:
            for request_id, entry in list(self._entries.items()):
                # Only clean up terminal states that are expired
                if entry.is_terminal() and entry.is_expired():
                    to_delete.append(request_id)

        # Delete outside lock to minimize lock time
        for request_id in to_delete:
            await self.delete(request_id)

        if to_delete:
            logger.debug(f"Cleaned up {len(to_delete)} expired entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store.

        Returns:
            Dict with counts by status and total
        """
        stats = {
            "total": len(self._entries),
            "by_status": {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "expired": 0,
            },
            "by_model": {
                model_id: len(request_ids)
                for model_id, request_ids in self._by_model.items()
            },
        }

        for entry in self._entries.values():
            status_key = entry.status.value
            if status_key in stats["by_status"]:
                stats["by_status"][status_key] += 1

        return stats


# Convenience function for creating TryAgainResponse-compatible dict
def make_try_again_response(
    request_id: str,
    queue_state: str = "active",
    queue_state_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a TryAgainResponse dict.

    Args:
        request_id: The request ID being polled
        queue_state: Current queue state
        queue_state_reason: Optional reason

    Returns:
        Dict matching TryAgainResponse format
    """
    response = {
        "type": "try_again",
        "request_id": request_id,
        "queue_state": queue_state,
    }
    if queue_state_reason:
        response["queue_state_reason"] = queue_state_reason
    return response


def make_failed_response(
    error: str,
    category: str = "unknown",
) -> Dict[str, Any]:
    """Create a RequestFailedResponse dict.

    Args:
        error: Error message
        category: Error category

    Returns:
        Dict matching RequestFailedResponse format
    """
    return {
        "error": error,
        "category": category,
    }
