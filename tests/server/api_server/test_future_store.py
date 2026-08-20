"""
Tests for FutureStore in xorl.
"""

import asyncio
import time

import pytest

from xorl.server.api_server.future_store import (
    FutureStatus,
    FutureStore,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


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
        fresh_store = FutureStore(
            default_ttl=60.0,
            max_concurrent=2,
            cleanup_interval=1.0,
        )
        await self._assert_model_ops_deletion_status_and_expiration(fresh_store)

    async def _assert_model_ops_deletion_status_and_expiration(self, future_store):
        """Test deletion, status/result storage, failure storage, and TTL expiration."""
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
            entry = future_store.get(request_id)
            assert entry.status in (FutureStatus.PENDING, FutureStatus.PROCESSING)
            assert entry.result is None

            await asyncio.wait_for(processed.wait(), timeout=5.0)
            await asyncio.sleep(0.1)
            entry = future_store.get(request_id)
            assert entry.status == FutureStatus.COMPLETED
            assert entry.result == {"value": 42}

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
            entry = future_store.get(request_id)
            assert "Server crashed" in entry.error and entry.error_category == "server"
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
