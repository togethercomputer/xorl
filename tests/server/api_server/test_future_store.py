"""
Tests for FutureStore in xorl.
"""

import asyncio
import time

import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.server]

from xorl.server.api_server.future_store import (
    FutureEntry,
    FutureStatus,
    FutureStore,
    make_failed_response,
    make_try_again_response,
)


@pytest.fixture
def future_store():
    """Create a FutureStore for testing."""
    return FutureStore(
        default_ttl=60.0,
        max_concurrent=2,
        cleanup_interval=1.0,
    )


class TestFutureStore:
    """Tests for FutureStore class."""

    @pytest.mark.asyncio
    async def test_create_process_failure_concurrent_and_stats(self, future_store):
        """Test creating futures, processing, failure handling, concurrent limits, and stats."""
        await future_store.start()

        try:
            # --- ID generation ---
            id1 = future_store._generate_request_id()
            id2 = future_store._generate_request_id()
            assert id1.startswith("future_") and id2.startswith("future_")
            assert id1 != id2 and len(id1) == 19

            # --- Create and retrieve ---
            async def dummy_process(data):
                return {"result": "success"}

            request_id = await future_store.create(
                model_id="model-1",
                request_type="test",
                process_fn=dummy_process,
                request_data={"test": "data"},
            )
            entry = future_store.get(request_id)
            assert entry is not None and entry.request_type == "test" and entry.model_id == "model-1"

            # --- Processing completes with results ---
            processed = asyncio.Event()

            async def slow_process(data):
                await asyncio.sleep(0.1)
                processed.set()
                return {"result": data["value"] * 2}

            request_id = await future_store.create(
                model_id="model-1",
                request_type="multiply",
                process_fn=slow_process,
                request_data={"value": 5},
            )
            await asyncio.wait_for(processed.wait(), timeout=5.0)
            await asyncio.sleep(0.1)
            entry = future_store.get(request_id)
            assert entry.status == FutureStatus.COMPLETED and entry.result == {"result": 10}

            # --- Failure handling ---
            async def failing_process(data):
                raise ValueError("Invalid input data")

            request_id = await future_store.create(
                model_id="model-1",
                request_type="fail",
                process_fn=failing_process,
                request_data={},
            )
            await asyncio.sleep(0.2)
            entry = future_store.get(request_id)
            assert entry.status == FutureStatus.FAILED
            assert "Invalid input data" in entry.error
            assert entry.error_category == "user"

            # --- Concurrent limit ---
            concurrent_count = 0
            max_concurrent = 0
            lock = asyncio.Lock()

            async def tracking_process(data):
                nonlocal concurrent_count, max_concurrent
                async with lock:
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.2)
                async with lock:
                    concurrent_count -= 1
                return {"done": True}

            request_ids = []
            for i in range(5):
                request_id = await future_store.create(
                    model_id=f"model-{i}",
                    request_type="track",
                    process_fn=tracking_process,
                    request_data={"index": i},
                )
                request_ids.append(request_id)

            await asyncio.sleep(1.0)
            assert max_concurrent <= 2
            for request_id in request_ids:
                assert future_store.get(request_id).status == FutureStatus.COMPLETED

            # --- Stats ---
            completed_event = asyncio.Event()

            async def slow_process2(data):
                await asyncio.sleep(0.1)
                if data.get("index") == 2:
                    completed_event.set()
                return {"done": True}

            for i in range(3):
                await future_store.create(
                    model_id=f"model-{i}",
                    request_type="stats",
                    process_fn=slow_process2,
                    request_data={"index": i},
                )
            await asyncio.wait_for(completed_event.wait(), timeout=5.0)
            await asyncio.sleep(0.1)
            stats = future_store.get_stats()
            assert stats["total"] >= 3
        finally:
            await future_store.stop()

    @pytest.mark.asyncio
    async def test_model_ops_deletion_status_and_expiration(self, future_store):
        """Test listing/deleting by model, status tracking, error info, and TTL expiration."""
        await future_store.start()

        try:

            async def dummy_process(data):
                await asyncio.sleep(0.5)
                return {"done": True}

            # --- List and delete by model ---
            ids_1 = [
                await future_store.create(
                    model_id="model-1", request_type="test", process_fn=dummy_process, request_data={"index": i}
                )
                for i in range(3)
            ]
            ids_2 = [
                await future_store.create(
                    model_id="model-2", request_type="test", process_fn=dummy_process, request_data={"index": i}
                )
                for i in range(2)
            ]

            assert len(future_store.list_by_model("model-1")) == 3
            assert len(future_store.list_by_model("model-2")) == 2
            assert len(future_store.list_by_model("model-3")) == 0

            deleted = await future_store.delete_by_model("model-1")
            assert deleted == 3
            for rid in ids_1:
                assert future_store.get(rid) is None
            for rid in ids_2:
                assert future_store.get(rid) is not None

            # --- Single deletion ---
            async def fast_process(data):
                return {"done": True}

            rid = await future_store.create(
                model_id="model-1", request_type="test", process_fn=fast_process, request_data={}
            )
            assert future_store.get(rid) is not None
            assert await future_store.delete(rid) is True
            assert future_store.get(rid) is None
            assert await future_store.delete(rid) is False

            # Nonexistent entry
            assert future_store.get("nonexistent_id") is None

            # --- Status and result tracking ---
            processed = asyncio.Event()

            async def slow_process(data):
                await asyncio.sleep(0.1)
                processed.set()
                return {"value": 42}

            request_id = await future_store.create(
                model_id="model-1",
                request_type="test",
                process_fn=slow_process,
                request_data={},
            )
            status = future_store.get_status(request_id)
            assert status in (FutureStatus.PENDING, FutureStatus.PROCESSING)
            assert future_store.get_result(request_id) is None

            await asyncio.wait_for(processed.wait(), timeout=5.0)
            await asyncio.sleep(0.1)
            assert future_store.get_status(request_id) == FutureStatus.COMPLETED
            assert future_store.get_result(request_id) == {"value": 42}
            assert future_store.get_status("nonexistent") is None

            # --- Error info ---
            async def failing_process(data):
                raise RuntimeError("Server crashed")

            request_id = await future_store.create(
                model_id="model-1",
                request_type="test",
                process_fn=failing_process,
                request_data={},
            )
            await asyncio.sleep(0.2)
            error_info = future_store.get_error(request_id)
            assert error_info is not None
            error_msg, error_category = error_info
            assert "Server crashed" in error_msg and error_category == "server"
        finally:
            await future_store.stop()

        # --- Expiration (separate store with short TTL) ---
        short_ttl_store = FutureStore(default_ttl=0.1, max_concurrent=2, cleanup_interval=60.0)
        await short_ttl_store.start()
        try:

            async def dummy(data):
                return {"done": True}

            request_id = await short_ttl_store.create(
                model_id="model-1",
                request_type="test",
                process_fn=dummy,
                request_data={},
            )
            await asyncio.sleep(0.15)
            assert short_ttl_store.get(request_id).status == FutureStatus.EXPIRED
        finally:
            await short_ttl_store.stop()

        # --- Custom TTL ---
        await future_store.start()
        try:
            request_id = await future_store.create(
                model_id="model-1",
                request_type="test",
                process_fn=dummy,
                request_data={},
                ttl=3600.0,
            )
            assert future_store.get(request_id).expires_at > time.time() + 3500
        finally:
            await future_store.stop()


class TestFutureEntryAndHelpers:
    """Tests for FutureEntry dataclass and helper functions."""

    def test_entry_queue_state_and_helpers(self):
        """Test FutureEntry creation/expiry/terminal, queue state, and helper functions."""
        # --- Entry creation ---
        entry = FutureEntry(request_id="future_abc", model_id="model-1", request_type="forward_backward")
        assert entry.status == FutureStatus.PENDING
        assert entry.result is None and entry.error is None and entry.error_category == "unknown"

        # Not expired vs expired
        entry_not_expired = FutureEntry(request_id="a", model_id="m", request_type="t", expires_at=time.time() + 3600)
        assert entry_not_expired.is_expired() is False
        entry_expired = FutureEntry(request_id="b", model_id="m", request_type="t", expires_at=time.time() - 1)
        assert entry_expired.is_expired() is True

        # Terminal states
        entry = FutureEntry(request_id="c", model_id="m", request_type="t")
        assert entry.is_terminal() is False
        entry.status = FutureStatus.PROCESSING
        assert entry.is_terminal() is False
        for status in (FutureStatus.COMPLETED, FutureStatus.FAILED, FutureStatus.EXPIRED):
            entry.status = status
            assert entry.is_terminal() is True

        # --- Queue state management ---
        store = FutureStore(default_ttl=60.0, max_concurrent=2, cleanup_interval=1.0)
        state, reason = store.get_queue_state()
        assert state == "active" and reason is None

        store.set_queue_state("paused_capacity", "GPU memory full")
        state, reason = store.get_queue_state()
        assert state == "paused_capacity" and reason == "GPU memory full"

        # --- Helper functions ---
        response = make_try_again_response("future_abc")
        assert response == {"type": "try_again", "request_id": "future_abc", "queue_state": "active"}

        response = make_try_again_response("future_xyz", queue_state="paused_capacity", queue_state_reason="GPU full")
        assert response["queue_state"] == "paused_capacity" and response["queue_state_reason"] == "GPU full"

        response = make_failed_response("Something went wrong")
        assert response == {"error": "Something went wrong", "category": "unknown"}

        response = make_failed_response("Invalid input", category="user")
        assert response["category"] == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
