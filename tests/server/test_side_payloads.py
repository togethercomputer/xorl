import base64

import numpy as np
import pytest
import torch

from xorl.server.side_payloads import (
    R3_ROUTED_EXPERTS,
    SIDE_PAYLOAD_REF_KEY,
    MooncakeSidePayloadStore,
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

    def put(self, key: str, value: bytes) -> int:
        self.objects[key] = bytes(value)
        self.put_calls.append(key)
        return 0

    def get(self, key: str) -> bytes:
        self.get_calls.append(key)
        return self.objects.get(key, b"")

    def is_exist(self, key: str) -> int:
        return 1 if key in self.objects else 0

    def remove(self, key: str) -> int:
        self.objects.pop(key, None)
        self.removed.append(key)
        return 0


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
    assert expert_ref["field"] == "routed_experts"
    assert expert_ref["count"] == 2
    assert r3_payload_count(expert_ref) == 2

    loaded_ids = load_r3_mooncake_payload_slice(expert_ref, 1, 1, store=store)
    loaded_weights = load_r3_mooncake_payload_slice(logits_ref, 1, 1, store=store)

    assert len(loaded_ids) == 1
    assert np.array_equal(loaded_ids[0], arr1.astype(np.int32))
    assert np.allclose(loaded_weights[0], weights1)
    assert client.get_calls == [
        expert_ref["items"]["routed_experts"][1]["key"],
        logits_ref["items"]["routed_expert_logits"][1]["key"],
    ]

    cleanup_r3_mooncake_payloads(cleanup)
    assert sorted(client.removed) == sorted(client.put_calls)
    assert client.objects == {}


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


def test_mooncake_side_payload_missing_key_raises():
    store, _ = _store()
    with pytest.raises(KeyError):
        store.get_tensor("missing", [1, 1, 1], "float32")
