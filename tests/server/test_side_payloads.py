import base64
import threading

import numpy as np
import pytest
import torch

from xorl.server.side_payloads import (
    R3_ROUTED_EXPERTS,
    SIDE_PAYLOAD_REF_KEY,
    MooncakeSidePayloadStore,
    R3PayloadRollbackError,
    canonicalize_r3_payload_item,
    cleanup_r3_mooncake_payloads,
    load_r3_mooncake_payload_slice,
    put_r3_mooncake_payload_refs,
    r3_payload_count,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class FakeMooncakeClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[str] = []
        self.get_calls: list[str] = []
        self.removed: list[str] = []
        self.remove_calls: list[tuple[str, bool]] = []
        self.remove_statuses: list[int] = []

    def put(self, key: str, value: bytes) -> int:
        self.objects[key] = bytes(value)
        self.put_calls.append(key)
        return 0

    def get(self, key: str) -> bytes:
        self.get_calls.append(key)
        return self.objects.get(key, b"")

    def is_exist(self, key: str) -> int:
        return 1 if key in self.objects else 0

    def remove(self, key: str, force: bool = False) -> int:
        self.remove_calls.append((key, force))
        status = self.remove_statuses.pop(0) if self.remove_statuses else 0
        if status == 0:
            self.objects.pop(key, None)
            self.removed.append(key)
        return status


def _store() -> tuple[MooncakeSidePayloadStore, FakeMooncakeClient]:
    client = FakeMooncakeClient()
    return MooncakeSidePayloadStore(client=client, get_retry_max_wait_s=0.0), client


def test_mooncake_side_payload_store_round_trips_int_and_float_tensors():
    store, client = _store()
    ids = torch.arange(12, dtype=torch.int32).reshape(3, 2, 2)
    weights = torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(3, 2, 2)

    ids_meta = store.put_tensor("r3/ids", ids)
    weights_meta = store.put_tensor("r3/weights", weights)

    assert client.put_calls == ["r3/ids", "r3/weights"]
    assert ids_meta == {"key": "r3/ids", "shape": [3, 2, 2], "dtype": "int32"}
    assert weights_meta == {"key": "r3/weights", "shape": [3, 2, 2], "dtype": "float32"}
    assert torch.equal(store.get_tensor_from_metadata(ids_meta), ids)
    assert torch.equal(store.get_tensor_from_metadata(weights_meta), weights)


def test_mooncake_side_payload_store_retries_empty_transfer_for_existing_object():
    class TransientEmptyGetClient(FakeMooncakeClient):
        def get(self, key: str) -> bytes:
            self.get_calls.append(key)
            if len(self.get_calls) == 1:
                return b""
            return self.objects[key]

    client = TransientEmptyGetClient()
    store = MooncakeSidePayloadStore(
        client=client,
        get_retry_max_wait_s=0.0,
        get_retry_interval_s=0.0,
        get_transfer_max_attempts=2,
    )
    tensor = torch.arange(8, dtype=torch.int32).reshape(2, 2, 2)
    metadata = store.put_tensor("r3/transient-read", tensor)

    assert torch.equal(store.get_tensor_from_metadata(metadata), tensor)
    assert client.get_calls == [metadata["key"], metadata["key"]]


def test_mooncake_side_payload_store_bounds_persistent_empty_transfer():
    class EmptyGetClient(FakeMooncakeClient):
        def get(self, key: str) -> bytes:
            self.get_calls.append(key)
            return b""

    client = EmptyGetClient()
    store = MooncakeSidePayloadStore(
        client=client,
        get_retry_max_wait_s=0.0,
        get_retry_interval_s=0.0,
        get_transfer_max_attempts=2,
    )
    metadata = store.put_tensor("r3/persistent-read-failure", torch.ones((2, 2), dtype=torch.float32))

    with pytest.raises(RuntimeError, match="transfer returned no data.*after 2 attempt"):
        store.get_tensor_from_metadata(metadata)
    assert client.get_calls == [metadata["key"], metadata["key"]]


def test_mooncake_side_payload_store_preserves_valid_empty_tensor():
    store, _ = _store()
    tensor = torch.empty((0, 2), dtype=torch.int32)
    metadata = store.put_tensor("r3/empty", tensor)

    assert torch.equal(store.get_tensor_from_metadata(metadata), tensor)


def test_r3_mooncake_parallel_puts_respect_inflight_byte_limit():
    class BoundedConcurrentClient(FakeMooncakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.active_bytes = 0
            self.max_active_bytes = 0
            self.active_calls = 0
            self.max_active_calls = 0
            self.lock = threading.Lock()
            self.release = threading.Event()

        def put(self, key: str, value: bytes) -> int:
            with self.lock:
                self.active_bytes += len(value)
                self.max_active_bytes = max(self.max_active_bytes, self.active_bytes)
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
                if self.active_calls == 3:
                    self.release.set()
            assert self.release.wait(timeout=2.0)
            try:
                return super().put(key, value)
            finally:
                with self.lock:
                    self.active_bytes -= len(value)
                    self.active_calls -= 1

    client = BoundedConcurrentClient()
    store = MooncakeSidePayloadStore(
        client=client,
        put_workers=8,
        max_put_bytes_inflight=48,
        packed_chunk_bytes=16,
    )
    payloads = [np.full((2, 1, 2), idx, dtype=np.int32) for idx in range(6)]

    expert_ref, _, _ = put_r3_mooncake_payload_refs(
        request_id="bounded-parallel",
        routed_experts=payloads,
        routed_expert_logits=None,
        store=store,
        chunk_ranges=[(idx, 1) for idx in range(len(payloads))],
        max_chunk_bytes=store.packed_chunk_bytes,
    )

    assert expert_ref is not None
    assert client.max_active_calls == 3
    assert client.max_active_bytes == 48
    assert [entry["datum_start"] for entry in expert_ref["items"][R3_ROUTED_EXPERTS]] == list(range(6))


def test_r3_mooncake_parallelizes_expert_and_weight_fields():
    class ConcurrentFieldClient(FakeMooncakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.active_calls = 0
            self.max_active_calls = 0
            self.lock = threading.Lock()
            self.release = threading.Event()

        def put(self, key: str, value: bytes) -> int:
            with self.lock:
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
                if self.active_calls == 2:
                    self.release.set()
            assert self.release.wait(timeout=2.0)
            try:
                return super().put(key, value)
            finally:
                with self.lock:
                    self.active_calls -= 1

    client = ConcurrentFieldClient()
    store = MooncakeSidePayloadStore(
        client=client,
        put_workers=2,
        max_put_bytes_inflight=64,
        packed_chunk_bytes=64,
    )
    experts = [np.zeros((2, 1, 2), dtype=np.int32)]
    weights = [np.zeros((2, 1, 2), dtype=np.float32)]

    expert_ref, logits_ref, _ = put_r3_mooncake_payload_refs(
        request_id="parallel-fields",
        routed_experts=experts,
        routed_expert_logits=weights,
        store=store,
    )

    assert expert_ref is not None
    assert logits_ref is not None
    assert client.max_active_calls == 2


def test_r3_mooncake_parallel_field_failure_rolls_back_other_field():
    class FailingWeightClient(FakeMooncakeClient):
        def put(self, key: str, value: bytes) -> int:
            if "/routed_expert_logits/" in key:
                self.put_calls.append(key)
                return -1
            return super().put(key, value)

    client = FailingWeightClient()
    store = MooncakeSidePayloadStore(
        client=client,
        put_workers=2,
        max_put_bytes_inflight=64,
        packed_chunk_bytes=64,
        remove_retry_max_wait_s=0.0,
    )
    experts = [np.zeros((2, 1, 2), dtype=np.int32)]
    weights = [np.zeros((2, 1, 2), dtype=np.float32)]

    with pytest.raises(RuntimeError, match="put failed"):
        put_r3_mooncake_payload_refs(
            request_id="parallel-field-rollback",
            routed_experts=experts,
            routed_expert_logits=weights,
            store=store,
        )

    assert client.objects == {}
    assert any(force and "/routed_experts/" in key for key, force in client.remove_calls)


def test_r3_mooncake_parallel_put_failure_rolls_back_every_success():
    class FailingChunkClient(FakeMooncakeClient):
        def put(self, key: str, value: bytes) -> int:
            if key.endswith("chunk-000001"):
                self.put_calls.append(key)
                return -1
            return super().put(key, value)

    client = FailingChunkClient()
    store = MooncakeSidePayloadStore(
        client=client,
        put_workers=4,
        max_put_bytes_inflight=64,
        packed_chunk_bytes=16,
        remove_retry_max_wait_s=0.0,
    )
    payloads = [np.full((2, 1, 2), idx, dtype=np.int32) for idx in range(4)]

    with pytest.raises(RuntimeError, match="put failed"):
        put_r3_mooncake_payload_refs(
            request_id="parallel-rollback",
            routed_experts=payloads,
            routed_expert_logits=None,
            store=store,
            chunk_ranges=[(idx, 1) for idx in range(len(payloads))],
            max_chunk_bytes=store.packed_chunk_bytes,
        )

    assert client.objects == {}
    assert sorted(key for key, force in client.remove_calls if force) == sorted(
        key for key in client.put_calls if not key.endswith("chunk-000001")
    )


def test_r3_mooncake_serial_put_failure_rolls_back_every_success():
    class FailingChunkClient(FakeMooncakeClient):
        def put(self, key: str, value: bytes) -> int:
            if key.endswith("chunk-000001"):
                self.put_calls.append(key)
                return -1
            return super().put(key, value)

    client = FailingChunkClient()
    store = MooncakeSidePayloadStore(
        client=client,
        put_workers=1,
        packed_chunk_bytes=16,
        remove_retry_max_wait_s=0.0,
    )
    payloads = [np.full((2, 1, 2), idx, dtype=np.int32) for idx in range(3)]

    with pytest.raises(RuntimeError, match="put failed"):
        put_r3_mooncake_payload_refs(
            request_id="serial-rollback",
            routed_experts=payloads,
            routed_expert_logits=None,
            store=store,
            chunk_ranges=[(idx, 1) for idx in range(len(payloads))],
            max_chunk_bytes=store.packed_chunk_bytes,
        )

    assert client.objects == {}
    assert [key for key, force in client.remove_calls if force] == [client.put_calls[0]]


def test_r3_mooncake_rejects_put_budget_larger_than_local_buffer():
    with pytest.raises(ValueError, match="in-flight put budget exceeds"):
        MooncakeSidePayloadStore(
            client=FakeMooncakeClient(),
            max_put_bytes_inflight=513 * 1024 * 1024,
        )


def test_r3_mooncake_refs_load_only_requested_slice():
    store, client = _store()
    arr0 = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    arr1 = np.arange(8, 16, dtype=np.int32).reshape(2, 2, 2)
    b64_item = {"data": base64.b64encode(arr0.tobytes()).decode("ascii"), "shape": [2, 2, 2]}
    weights0 = np.full((2, 2, 2), 0.25, dtype=np.float32)
    weights1 = np.full((2, 2, 2), 0.75, dtype=np.float32)

    expert_ref, logits_ref, cleanup = put_r3_mooncake_payload_refs(
        request_id="req/1",
        routed_experts=[b64_item, arr1],
        routed_expert_logits=[weights0, weights1.tolist()],
        store=store,
        namespace_prefix="tests/r3",
    )

    assert expert_ref is not None and logits_ref is not None and cleanup is not None
    assert expert_ref[SIDE_PAYLOAD_REF_KEY] is True
    assert expert_ref["backend"] == "mooncake"
    assert expert_ref["version"] == 2
    assert expert_ref["format"] == "packed_rows"
    assert expert_ref["field"] == "routed_experts"
    assert expert_ref["count"] == 2
    assert r3_payload_count(expert_ref) == 2

    loaded_ids = load_r3_mooncake_payload_slice(expert_ref, 1, 1, store=store)
    loaded_weights = load_r3_mooncake_payload_slice(logits_ref, 1, 1, store=store)

    assert len(loaded_ids) == 1
    assert np.array_equal(loaded_ids[0], arr1.astype(np.int32))
    assert np.allclose(loaded_weights[0], weights1)
    assert client.get_calls == [
        expert_ref["items"]["routed_experts"][0]["key"],
        logits_ref["items"]["routed_expert_logits"][0]["key"],
    ]

    cleanup_stats = cleanup_r3_mooncake_payloads(cleanup, force=True)
    assert sorted(client.removed) == sorted(client.put_calls)
    assert all(force for _, force in client.remove_calls)
    assert cleanup_stats.succeeded == len(client.put_calls)
    assert cleanup_stats.failed == 0
    assert cleanup_stats.pending == 0
    assert client.objects == {}


def test_r3_mooncake_cleanup_retries_transient_force_delete_statuses():
    client = FakeMooncakeClient()
    store = MooncakeSidePayloadStore(
        client=client,
        get_retry_max_wait_s=0.0,
        remove_retry_max_wait_s=1.0,
        remove_retry_interval_s=0.0,
    )
    _, _, cleanup = put_r3_mooncake_payload_refs(
        request_id="retry-cleanup",
        routed_experts=[[[[1, 2]]]],
        routed_expert_logits=None,
        store=store,
    )
    assert cleanup is not None
    client.remove_statuses = [-703, -708, 0]

    stats = cleanup_r3_mooncake_payloads(cleanup, force=True)

    assert stats.succeeded == 1
    assert stats.retry_attempts == 2
    assert stats.failed == 0
    assert client.remove_calls == [(client.put_calls[0], True)] * 3


def test_r3_mooncake_cleanup_reports_persistent_active_lease():
    client = FakeMooncakeClient()
    store = MooncakeSidePayloadStore(
        client=client,
        get_retry_max_wait_s=0.0,
        remove_retry_max_wait_s=0.0,
        remove_retry_interval_s=0.0,
    )
    _, _, cleanup = put_r3_mooncake_payload_refs(
        request_id="failed-cleanup",
        routed_experts=[[[[1, 2]]]],
        routed_expert_logits=None,
        store=store,
    )
    assert cleanup is not None
    client.remove_statuses = [-706]

    stats = cleanup_r3_mooncake_payloads(cleanup, force=True)

    assert stats.attempted == 1
    assert stats.succeeded == 0
    assert stats.failed == 1
    assert stats.pending == 1
    assert stats.retained_bytes > 0
    assert "status=-706" in stats.failures[0]
    assert client.objects


def test_r3_mooncake_cleanup_treats_missing_key_as_already_drained():
    store, client = _store()
    _, _, cleanup = put_r3_mooncake_payload_refs(
        request_id="already-absent",
        routed_experts=[[[[1, 2]]]],
        routed_expert_logits=None,
        store=store,
    )
    assert cleanup is not None
    client.objects.clear()
    client.remove_statuses = [-704]

    stats = cleanup_r3_mooncake_payloads(cleanup, force=True)

    assert stats.succeeded == 1
    assert stats.already_absent == 1
    assert stats.failed == 0
    assert stats.pending == 0


def test_r3_mooncake_cleanup_attempts_every_key_after_a_failure():
    store, client = _store()
    _, _, cleanup = put_r3_mooncake_payload_refs(
        request_id="multi-key-cleanup",
        routed_experts=[[[[1, 2]]]],
        routed_expert_logits=[[[[0.25, 0.75]]]],
        store=store,
    )
    assert cleanup is not None
    client.remove_statuses = [-706, 0]

    stats = cleanup_r3_mooncake_payloads(cleanup, force=True)

    assert stats.attempted == 2
    assert stats.succeeded == 1
    assert stats.failed == 1
    assert stats.pending == 1
    assert client.remove_calls == [(key, True) for key in client.put_calls]
    assert set(client.objects) == {client.put_calls[0]}


def test_r3_mooncake_partial_put_failure_force_removes_published_chunks():
    class FailingSecondPutClient(FakeMooncakeClient):
        def put(self, key: str, value: bytes) -> int:
            if len(self.put_calls) == 1:
                self.put_calls.append(key)
                return -1
            return super().put(key, value)

    client = FailingSecondPutClient()
    store = MooncakeSidePayloadStore(
        client=client,
        get_retry_max_wait_s=0.0,
        remove_retry_max_wait_s=0.0,
    )

    with pytest.raises(RuntimeError, match="put failed"):
        put_r3_mooncake_payload_refs(
            request_id="partial-put",
            routed_experts=[[[[1, 2]]]],
            routed_expert_logits=[[[[0.25, 0.75]]]],
            store=store,
        )

    assert client.objects == {}
    assert client.remove_calls == [(client.put_calls[0], True)]


def test_r3_mooncake_partial_put_failure_aggregates_rollback_failures():
    class FailingThirdPutClient(FakeMooncakeClient):
        def put(self, key: str, value: bytes) -> int:
            if len(self.put_calls) == 2:
                self.put_calls.append(key)
                return -1
            return super().put(key, value)

    client = FailingThirdPutClient()
    store = MooncakeSidePayloadStore(
        client=client,
        get_retry_max_wait_s=0.0,
        remove_retry_max_wait_s=0.0,
    )
    client.remove_statuses = [-706, 0]

    with pytest.raises(R3PayloadRollbackError) as exc_info:
        put_r3_mooncake_payload_refs(
            request_id="partial-put-incomplete-rollback",
            routed_experts=[[[[1, 2]]], [[[3, 4]]]],
            routed_expert_logits=[[[[0.25, 0.75]]], [[[0.5, 0.5]]]],
            store=store,
            chunk_ranges=[(0, 1), (1, 1)],
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "put failed" in str(exc_info.value.__cause__)
    assert exc_info.value.cleanup_stats.attempted == 2
    assert exc_info.value.cleanup_stats.failed == 1
    assert exc_info.value.cleanup_stats.pending == 1
    assert client.remove_calls == [(client.put_calls[0], True), (client.put_calls[1], True)]
    assert set(client.objects) == {client.put_calls[0]}


def test_r3_payload_validation_failures_are_loud():
    with pytest.raises(ValueError, match="shape"):
        canonicalize_r3_payload_item([[1, 2], [3, 4]], field=R3_ROUTED_EXPERTS, target_dtype=torch.int32)
    with pytest.raises(ValueError, match="int32/int64"):
        canonicalize_r3_payload_item([[[1.5, 2.0]]], field=R3_ROUTED_EXPERTS, target_dtype=torch.int32)

    store, _ = _store()
    expert_ref, _, _ = put_r3_mooncake_payload_refs(
        request_id="req",
        routed_experts=[[[[1, 2]]]],
        routed_expert_logits=None,
        store=store,
    )
    assert expert_ref is not None
    expert_ref["count"] = 2
    with pytest.raises(ValueError, match="count mismatch"):
        r3_payload_count(expert_ref)


def test_r3_packed_chunks_are_bounded_and_slice_fetches_only_overlap():
    store, client = _store()
    payloads = [np.full((2, 1, 2), idx, dtype=np.int32) for idx in range(4)]
    expert_ref, _, _ = put_r3_mooncake_payload_refs(
        request_id="bounded",
        routed_experts=payloads,
        routed_expert_logits=None,
        store=store,
        chunk_ranges=[(0, 2), (2, 2)],
        max_chunk_bytes=1024,
    )
    assert expert_ref is not None
    chunks = expert_ref["items"]["routed_experts"]
    assert len(chunks) == 2
    assert len(client.put_calls) == 2

    loaded = load_r3_mooncake_payload_slice(expert_ref, 2, 1, store=store)
    assert np.array_equal(loaded[0], payloads[2])
    assert client.get_calls == [chunks[1]["key"]]


def test_r3_v1_per_datum_ref_remains_readable():
    store, _ = _store()
    first = store.put_tensor("legacy/0", torch.tensor([[[1, 2]]], dtype=torch.int32))
    second = store.put_tensor("legacy/1", torch.tensor([[[3, 4]]], dtype=torch.int32))
    ref = {
        SIDE_PAYLOAD_REF_KEY: True,
        "backend": "mooncake",
        "kind": "r3_routing",
        "version": 1,
        "field": "routed_experts",
        "count": 2,
        "items": {"routed_experts": [first, second]},
        "mooncake": store.config.to_metadata(),
    }

    loaded = load_r3_mooncake_payload_slice(ref, 1, 1, store=store)
    assert loaded[0].tolist() == [[[3, 4]]]


def test_r3_original_base64_buffer_can_expose_a_prefix_view():
    store, _ = _store()
    full = np.arange(4 * 2 * 2, dtype=np.int32).reshape(4, 2, 2)
    payload = {
        "data": base64.b64encode(full.tobytes()).decode("ascii"),
        "shape": [4, 2, 2],
        "rows": 3,
    }

    expert_ref, _, _ = put_r3_mooncake_payload_refs(
        request_id="prefix-view",
        routed_experts=[payload],
        routed_expert_logits=None,
        store=store,
    )
    loaded = load_r3_mooncake_payload_slice(expert_ref, 0, 1, store=store)
    assert np.array_equal(loaded[0], full[:3])


def test_mooncake_side_payload_missing_key_raises():
    store, _ = _store()
    with pytest.raises(KeyError):
        store.get_tensor("missing", [1, 1, 1], "float32")
