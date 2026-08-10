"""Side-payload transports for large server request metadata.

The normal runner command path uses ``dist.broadcast_object_list``. That is fine
for small Python metadata, but full R3 routing replay payloads can be hundreds of
MB for long-prompt MoE runs. This module provides a small Mooncake-backed tensor
side channel so commands can broadcast metadata refs while workers fetch only
their datum slice.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

import numpy as np
import torch

from xorl.distillation.mooncake_hidden_store import (
    MooncakeStoreConfig,
    _build_mooncake_client,
    bytes_to_tensor,
    dtype_to_str,
    str_to_dtype,
    tensor_to_bytes,
)


logger = logging.getLogger(__name__)

SIDE_PAYLOAD_REF_KEY = "__xorl_side_payload_ref__"
R3_ROUTING_KIND = "r3_routing"
R3_ROUTED_EXPERTS = "routed_experts"
R3_ROUTED_EXPERT_LOGITS = "routed_expert_logits"
R3_ROUTING_FIELDS = (R3_ROUTED_EXPERTS, R3_ROUTED_EXPERT_LOGITS)


class MooncakeByteClient(Protocol):
    def put(self, key: str, value: bytes) -> int: ...

    def get(self, key: str) -> bytes: ...

    def is_exist(self, key: str) -> int: ...

    def remove(self, key: str) -> int: ...


@dataclass(frozen=True)
class R3PayloadCleanup:
    """Producer-side cleanup handle for refs written to Mooncake."""

    refs: tuple[Mapping[str, Any], ...]
    store: "MooncakeSidePayloadStore"


class MooncakeSidePayloadStore:
    """Minimal keyed raw-tensor store for server side payloads."""

    def __init__(
        self,
        config: Optional[MooncakeStoreConfig] = None,
        *,
        client: Optional[MooncakeByteClient] = None,
        get_retry_max_wait_s: float = 30.0,
        get_retry_interval_s: float = 0.2,
    ) -> None:
        self.config = config or MooncakeStoreConfig.from_env()
        self._client: Optional[MooncakeByteClient] = client
        self._owns_client = client is None
        self._get_retry_max_wait_s = float(get_retry_max_wait_s)
        self._get_retry_interval_s = float(get_retry_interval_s)

    @classmethod
    def from_metadata(
        cls,
        metadata: Optional[Mapping[str, Any]],
        *,
        get_retry_max_wait_s: float = 30.0,
        get_retry_interval_s: float = 0.2,
    ) -> "MooncakeSidePayloadStore":
        return cls(
            MooncakeStoreConfig.from_env(metadata or {}),
            get_retry_max_wait_s=get_retry_max_wait_s,
            get_retry_interval_s=get_retry_interval_s,
        )

    @property
    def client(self) -> MooncakeByteClient:
        if self._client is None:
            self._client = _build_mooncake_client(self.config)
        return self._client

    def put_tensor(self, key: str, tensor: torch.Tensor) -> dict[str, Any]:
        if not key:
            raise ValueError("Mooncake side payload key must be non-empty")
        tensor = tensor.detach().to(device="cpu").contiguous()
        ret = self.client.put(key, tensor_to_bytes(tensor))
        if ret is not None and ret != 0:
            raise RuntimeError(f"Mooncake side payload put failed for key {key!r} (error={ret})")
        return {
            "key": key,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": dtype_to_str(tensor.dtype),
        }

    def get_tensor(
        self,
        key: str,
        shape: tuple[int, ...] | list[int],
        dtype: torch.dtype | str,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if not key:
            raise ValueError("Mooncake side payload metadata missing 'key'")
        resolved_shape = tuple(int(dim) for dim in shape)
        resolved_dtype = str_to_dtype(dtype)
        data = self._get_with_retry(key)
        return bytes_to_tensor(data, resolved_shape, resolved_dtype, device=device)

    def get_tensor_from_metadata(
        self,
        metadata: Mapping[str, Any],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        key, shape, dtype = parse_tensor_metadata(metadata)
        return self.get_tensor(key, shape, dtype, device=device)

    def remove(self, key_or_metadata: str | Mapping[str, Any]) -> None:
        key = key_or_metadata if isinstance(key_or_metadata, str) else key_or_metadata.get("key")
        if not key:
            return
        try:
            self.client.remove(str(key))
        except Exception:  # pragma: no cover - cleanup is best-effort
            logger.debug("Failed to remove Mooncake side payload key %s", key, exc_info=True)

    def _get_with_retry(self, key: str) -> bytes:
        deadline = time.monotonic() + self._get_retry_max_wait_s
        while True:
            data = self.client.get(key)
            exists = False
            try:
                exists = bool(self.client.is_exist(key))
            except Exception:
                exists = bool(data)
            if data or exists:
                return bytes(data)
            if time.monotonic() >= deadline:
                raise KeyError(
                    f"Mooncake side payload key {key!r} was not found after {self._get_retry_max_wait_s:.1f}s"
                )
            time.sleep(self._get_retry_interval_s)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
            self._client = None


def parse_tensor_metadata(metadata: Mapping[str, Any]) -> tuple[str, list[int], torch.dtype]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Mooncake side payload tensor metadata must be a mapping, got {type(metadata)!r}")
    key = metadata.get("key")
    if not key:
        raise ValueError(f"Mooncake side payload tensor metadata missing 'key': {metadata!r}")
    shape = metadata.get("shape")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"Mooncake side payload tensor {key!r} has malformed shape {shape!r}")
    if "dtype" not in metadata:
        raise ValueError(f"Mooncake side payload tensor {key!r} missing 'dtype'")
    return str(key), list(shape), str_to_dtype(metadata["dtype"])


def is_side_payload_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value.get(SIDE_PAYLOAD_REF_KEY))


def r3_payload_count(value: Optional[Any]) -> int:
    if value is None:
        return 0
    if not is_side_payload_ref(value):
        return len(value)
    field, items = _parse_r3_ref_items(value)
    count = int(value.get("count", len(items)))
    if count != len(items):
        raise ValueError(f"R3 side payload count mismatch for {field}: ref count={count}, metadata items={len(items)}")
    return count


def put_r3_mooncake_payload_refs(
    *,
    request_id: str,
    routed_experts: Optional[list[Any]],
    routed_expert_logits: Optional[list[Any]],
    store: MooncakeSidePayloadStore,
    namespace_prefix: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[R3PayloadCleanup]]:
    if routed_experts is None and routed_expert_logits is None:
        return routed_experts, routed_expert_logits, None
    if (
        routed_experts is not None
        and routed_expert_logits is not None
        and len(routed_experts) != len(routed_expert_logits)
    ):
        raise ValueError(
            "R3 routed_experts and routed_expert_logits count mismatch: "
            f"{len(routed_experts)} != {len(routed_expert_logits)}"
        )

    safe_request_id = _safe_namespace_component(request_id) or "request"
    namespace = f"{(namespace_prefix or 'xorl/r3').rstrip('/')}/{safe_request_id}/{uuid.uuid4().hex[:12]}"
    written: list[Mapping[str, Any]] = []
    items: dict[str, list[dict[str, Any]]] = {}

    try:
        if routed_experts is not None:
            items[R3_ROUTED_EXPERTS] = _put_r3_field(
                store,
                namespace,
                R3_ROUTED_EXPERTS,
                routed_experts,
                target_dtype=torch.int32,
            )
            written.extend(items[R3_ROUTED_EXPERTS])
        if routed_expert_logits is not None:
            items[R3_ROUTED_EXPERT_LOGITS] = _put_r3_field(
                store,
                namespace,
                R3_ROUTED_EXPERT_LOGITS,
                routed_expert_logits,
                target_dtype=torch.float32,
            )
            written.extend(items[R3_ROUTED_EXPERT_LOGITS])
    except Exception:
        for entry in written:
            store.remove(entry)
        raise

    base_ref: dict[str, Any] = {
        SIDE_PAYLOAD_REF_KEY: True,
        "backend": "mooncake",
        "kind": R3_ROUTING_KIND,
        "version": 1,
        "request_id": str(request_id),
        "namespace": namespace,
        "items": items,
        "mooncake": store.config.to_metadata(),
    }

    def _field_ref(field: str, source: Optional[list[Any]]) -> Optional[dict[str, Any]]:
        if source is None:
            return None
        ref = dict(base_ref)
        ref["field"] = field
        ref["count"] = len(source)
        return ref

    expert_ref = _field_ref(R3_ROUTED_EXPERTS, routed_experts)
    logits_ref = _field_ref(R3_ROUTED_EXPERT_LOGITS, routed_expert_logits)
    refs = tuple(ref for ref in (expert_ref, logits_ref) if ref is not None)
    return expert_ref, logits_ref, R3PayloadCleanup(refs=tuple(refs), store=store)


def load_r3_mooncake_payload_slice(
    ref: Mapping[str, Any],
    start: int,
    count: int,
    *,
    store: Optional[MooncakeSidePayloadStore] = None,
) -> list[np.ndarray]:
    field, items = _parse_r3_ref_items(ref)
    if start < 0 or count < 0 or start + count > len(items):
        raise ValueError(
            f"R3 side payload slice out of range for {field}: start={start}, count={count}, total={len(items)}"
        )
    payload_store = store or MooncakeSidePayloadStore.from_metadata(_require_mapping(ref.get("mooncake"), "mooncake"))
    arrays: list[np.ndarray] = []
    for entry in items[start : start + count]:
        tensor = payload_store.get_tensor_from_metadata(entry)
        arrays.append(tensor.detach().cpu().numpy())
    return arrays


def cleanup_r3_mooncake_payloads(cleanup: R3PayloadCleanup) -> None:
    seen: set[str] = set()
    for ref in cleanup.refs:
        items_by_field = _require_mapping(ref.get("items"), "items")
        for field in R3_ROUTING_FIELDS:
            entries = items_by_field.get(field)
            if entries is None:
                continue
            if not isinstance(entries, list):
                logger.warning("Malformed R3 side payload cleanup entries for %s: %r", field, entries)
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                key = str(entry.get("key", ""))
                if not key or key in seen:
                    continue
                cleanup.store.remove(entry)
                seen.add(key)


def canonicalize_r3_payload_item(item: Any, *, field: str, target_dtype: torch.dtype) -> torch.Tensor:
    if field not in R3_ROUTING_FIELDS:
        raise ValueError(f"Unsupported R3 side payload field {field!r}")
    tensor = _to_routing_tensor(item, field=field, target_dtype=target_dtype)
    if tensor.ndim != 3:
        raise ValueError(f"R3 {field} payload must have shape [num_rows, num_layers, topk], got {tuple(tensor.shape)}")
    if field == R3_ROUTED_EXPERTS:
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"R3 routed_experts payload must be int32/int64, got {tensor.dtype}")
        if tensor.dtype == torch.int64:
            int32_info = torch.iinfo(torch.int32)
            if tensor.numel() and (tensor.min() < int32_info.min or tensor.max() > int32_info.max):
                raise ValueError("R3 routed_experts payload contains values outside int32 range")
        return tensor.to(dtype=torch.int32).contiguous()
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(dtype=torch.float32).contiguous()


def _put_r3_field(
    store: MooncakeSidePayloadStore,
    namespace: str,
    field: str,
    payloads: list[Any],
    *,
    target_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(payloads):
        tensor = canonicalize_r3_payload_item(item, field=field, target_dtype=target_dtype)
        entries.append(store.put_tensor(f"{namespace}/{field}/{idx:06d}", tensor))
    return entries


def _to_routing_tensor(item: Any, *, field: str, target_dtype: torch.dtype) -> torch.Tensor:
    if isinstance(item, torch.Tensor):
        return item.detach().to(device="cpu")
    if isinstance(item, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(item))
    if isinstance(item, Mapping):
        return _decode_sglang_routing_dict(item, field=field, target_dtype=target_dtype)
    if isinstance(item, list):
        return torch.as_tensor(item)
    if isinstance(item, tuple):
        return torch.as_tensor(item)
    if isinstance(item, str):
        raise ValueError(
            f"R3 {field} Mooncake payload requires dict metadata with shape; bare base64 strings are ambiguous"
        )
    raise ValueError(f"Unsupported R3 {field} payload item type {type(item)!r}")


def _decode_sglang_routing_dict(item: Mapping[str, Any], *, field: str, target_dtype: torch.dtype) -> torch.Tensor:
    if "data" not in item:
        raise ValueError(f"R3 {field} dict payload missing 'data'")
    shape = item.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"R3 {field} dict payload must include rank-3 integer 'shape', got {shape!r}")
    try:
        raw = base64.b64decode(item["data"])
    except Exception as exc:
        raise ValueError(f"R3 {field} dict payload has invalid base64 data: {exc}") from exc

    np_dtype = np.int32 if target_dtype in (torch.int32, torch.int64) else np.float32
    try:
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    except Exception as exc:
        raise ValueError(f"R3 {field} dict payload cannot be reshaped to {shape}: {exc}") from exc
    return torch.from_numpy(arr.copy())


def _parse_r3_ref_items(ref: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
    _validate_r3_ref_header(ref)
    field = ref.get("field")
    if field not in R3_ROUTING_FIELDS:
        raise ValueError(f"R3 side payload ref has unsupported field {field!r}")
    items_by_field = _require_mapping(ref.get("items"), "items")
    entries = items_by_field.get(field)
    if not isinstance(entries, list):
        raise ValueError(f"R3 side payload ref missing items[{field!r}] list")
    for idx, entry in enumerate(entries):
        parse_tensor_metadata(_require_mapping(entry, f"items[{field!r}][{idx}]"))
    return str(field), entries


def _validate_r3_ref_header(ref: Mapping[str, Any]) -> None:
    if not is_side_payload_ref(ref):
        raise ValueError(f"R3 side payload ref missing {SIDE_PAYLOAD_REF_KEY!r}")
    if str(ref.get("backend", "")).lower() != "mooncake":
        raise ValueError(f"R3 side payload ref backend must be 'mooncake', got {ref.get('backend')!r}")
    if ref.get("kind") != R3_ROUTING_KIND:
        raise ValueError(f"R3 side payload ref kind must be {R3_ROUTING_KIND!r}, got {ref.get('kind')!r}")
    if int(ref.get("version", 0)) != 1:
        raise ValueError(f"R3 side payload ref version must be 1, got {ref.get('version')!r}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"R3 side payload metadata field {name!r} must be a mapping, got {type(value)!r}")
    return value


def _safe_namespace_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
