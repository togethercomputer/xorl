"""Side-payload transports for large server request metadata.

The normal runner command path uses ``dist.broadcast_object_list``. That is fine
for small Python metadata, but full R3 routing replay payloads can be hundreds of
MB for long-prompt MoE runs. This module provides a small Mooncake-backed tensor
side channel so commands can broadcast metadata refs while workers fetch only
their datum slice.
"""

from __future__ import annotations

import base64
import concurrent.futures
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
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
R3_PACKED_FORMAT = "packed_rows"
# Keep individual Mooncake objects comfortably below the default local buffer
# while reducing the production request from thousands of objects to tens.
DEFAULT_R3_PACKED_CHUNK_BYTES = 256 * 1024 * 1024
MOONCAKE_SUCCESS = 0
MOONCAKE_REPLICA_NOT_READY = -703
MOONCAKE_OBJECT_NOT_FOUND = -704
MOONCAKE_OBJECT_HAS_LEASE = -706
MOONCAKE_OBJECT_HAS_REPLICATION_TASK = -708


class MooncakeByteClient(Protocol):
    def put(self, key: str, value: bytes) -> int: ...

    def get(self, key: str) -> bytes: ...

    def is_exist(self, key: str) -> int: ...

    def remove(self, key: str, force: bool = False) -> int: ...


@dataclass(frozen=True)
class R3PayloadCleanup:
    """Producer-side cleanup handle for refs written to Mooncake."""

    refs: tuple[Mapping[str, Any], ...]
    store: "MooncakeSidePayloadStore"
    created_monotonic_s: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class R3PayloadCleanupStats:
    """Checked cleanup result for one externalized R3 request."""

    total: int
    attempted: int
    succeeded: int
    already_absent: int
    failed: int
    pending: int
    retry_attempts: int
    removed_bytes: int
    retained_bytes: int
    elapsed_s: float
    oldest_key_age_s: float
    failures: tuple[str, ...] = ()


class R3PayloadRollbackError(RuntimeError):
    """A payload put failed and one or more unpublished chunks remain."""

    def __init__(self, original_error: Exception, cleanup_stats: R3PayloadCleanupStats) -> None:
        self.original_error = original_error
        self.cleanup_stats = cleanup_stats
        super().__init__(
            f"R3 Mooncake put failed ({type(original_error).__name__}: {original_error}); "
            f"rollback incomplete: failed={cleanup_stats.failed} pending={cleanup_stats.pending} "
            f"retained_bytes={cleanup_stats.retained_bytes} failures={cleanup_stats.failures}"
        )


class MooncakeSidePayloadStore:
    """Minimal keyed raw-tensor store for server side payloads."""

    def __init__(
        self,
        config: Optional[MooncakeStoreConfig] = None,
        *,
        client: Optional[MooncakeByteClient] = None,
        get_retry_max_wait_s: float = 30.0,
        get_retry_interval_s: float = 0.2,
        get_transfer_max_attempts: int = 2,
        remove_retry_max_wait_s: float = 30.0,
        remove_retry_interval_s: float = 0.2,
        put_workers: Optional[int] = None,
        max_put_bytes_inflight: Optional[int] = None,
        packed_chunk_bytes: Optional[int] = None,
    ) -> None:
        self.config = config or MooncakeStoreConfig.from_env()
        self._client: Optional[MooncakeByteClient] = client
        self._owns_client = client is None
        self._get_retry_max_wait_s = float(get_retry_max_wait_s)
        self._get_retry_interval_s = float(get_retry_interval_s)
        self._get_transfer_max_attempts = max(1, int(get_transfer_max_attempts))
        self._remove_retry_max_wait_s = float(remove_retry_max_wait_s)
        self._remove_retry_interval_s = float(remove_retry_interval_s)
        self.put_workers = max(
            1,
            int(put_workers if put_workers is not None else os.getenv("XORL_R3_MOONCAKE_PUT_WORKERS", "1")),
        )
        self.max_put_bytes_inflight = max(
            1,
            int(
                max_put_bytes_inflight
                if max_put_bytes_inflight is not None
                else os.getenv(
                    "XORL_R3_MOONCAKE_MAX_INFLIGHT_BYTES",
                    str(self.config.local_buffer_size * 3 // 4),
                )
            ),
        )
        self.packed_chunk_bytes = max(
            1,
            int(
                packed_chunk_bytes
                if packed_chunk_bytes is not None
                else os.getenv("XORL_R3_PACKED_CHUNK_BYTES", str(DEFAULT_R3_PACKED_CHUNK_BYTES))
            ),
        )
        if self.packed_chunk_bytes > self.config.local_buffer_size:
            raise ValueError(
                "R3 packed chunk size exceeds the Mooncake local buffer: "
                f"chunk={self.packed_chunk_bytes} buffer={self.config.local_buffer_size}"
            )
        if self.max_put_bytes_inflight > self.config.local_buffer_size:
            raise ValueError(
                "R3 Mooncake in-flight put budget exceeds the local buffer: "
                f"inflight={self.max_put_bytes_inflight} buffer={self.config.local_buffer_size}"
            )
        self._put_bytes_inflight = 0
        self._put_bytes_condition = threading.Condition()
        if self.put_workers > 1:
            logger.info(
                "Configured parallel R3 Mooncake puts workers=%d chunk_bytes=%d max_inflight_bytes=%d "
                "local_buffer_bytes=%d",
                self.put_workers,
                self.packed_chunk_bytes,
                self.max_put_bytes_inflight,
                self.config.local_buffer_size,
            )

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
        data_bytes = tensor.numel() * tensor.element_size()
        if data_bytes > self.config.local_buffer_size:
            raise ValueError(
                f"Mooncake side payload {key!r} is larger than the local buffer: "
                f"payload={data_bytes} buffer={self.config.local_buffer_size}"
            )
        self._acquire_put_bytes(data_bytes)
        try:
            # Serialize only after reserving the byte budget so concurrent
            # host copies are covered by the same bound as active puts.
            data = tensor_to_bytes(tensor)
            ret = self.client.put(key, data)
            if ret is not None and ret != 0:
                raise RuntimeError(f"Mooncake side payload put failed for key {key!r} (error={ret})")
        finally:
            self._release_put_bytes(data_bytes)
        return {
            "key": key,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": dtype_to_str(tensor.dtype),
        }

    def _acquire_put_bytes(self, data_bytes: int) -> None:
        with self._put_bytes_condition:
            while self._put_bytes_inflight and self._put_bytes_inflight + data_bytes > self.max_put_bytes_inflight:
                self._put_bytes_condition.wait()
            self._put_bytes_inflight += data_bytes

    def _release_put_bytes(self, data_bytes: int) -> None:
        with self._put_bytes_condition:
            self._put_bytes_inflight -= data_bytes
            self._put_bytes_condition.notify_all()

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
        expected_bytes = math.prod(resolved_shape) * torch.empty((), dtype=resolved_dtype).element_size()
        data = self._get_with_retry(key, expected_bytes=expected_bytes)
        return bytes_to_tensor(data, resolved_shape, resolved_dtype, device=device)

    def get_tensor_from_metadata(
        self,
        metadata: Mapping[str, Any],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        key, shape, dtype = parse_tensor_metadata(metadata)
        return self.get_tensor(key, shape, dtype, device=device)

    def remove(
        self,
        key_or_metadata: str | Mapping[str, Any],
        *,
        force: bool = False,
        deadline: Optional[float] = None,
    ) -> tuple[int, bool]:
        """Remove one exact key, checking Mooncake's integer status.

        Returns ``(attempts, already_absent)``. R3 cleanup passes
        ``force=True`` only after the distributed backend's all-rank completion
        rendezvous, when every synchronous consumer has returned.
        """
        key = key_or_metadata if isinstance(key_or_metadata, str) else key_or_metadata.get("key")
        if not key:
            return 0, True
        resolved_deadline = time.monotonic() + self._remove_retry_max_wait_s if deadline is None else float(deadline)
        attempts = 0
        last_failure = "unknown"
        while True:
            attempts += 1
            retryable = False
            try:
                status = self.client.remove(str(key), force)
            except Exception as exc:
                last_failure = f"{type(exc).__name__}: {exc}"
                retryable = True
            else:
                if status is None or int(status) == MOONCAKE_SUCCESS:
                    return attempts, False
                if int(status) == MOONCAKE_OBJECT_NOT_FOUND:
                    return attempts, True
                last_failure = f"status={status}"
                try:
                    if not bool(self.client.is_exist(str(key))):
                        return attempts, True
                except Exception:
                    pass
                retryable = (
                    int(status)
                    in {
                        MOONCAKE_REPLICA_NOT_READY,
                        MOONCAKE_OBJECT_HAS_REPLICATION_TASK,
                    }
                    or int(status) <= -900
                )

            remaining_s = resolved_deadline - time.monotonic()
            if not retryable or remaining_s <= 0:
                raise RuntimeError(
                    f"Mooncake side payload removal failed for key {key!r} after {attempts} attempt(s): {last_failure}"
                )
            logger.warning(
                "Retrying Mooncake side payload removal key=%s force=%s attempt=%d failure=%s",
                key,
                force,
                attempts,
                last_failure,
            )
            time.sleep(min(max(self._remove_retry_interval_s, 0.0), remaining_s))

    def _get_with_retry(self, key: str, *, expected_bytes: int) -> bytes:
        missing_deadline = time.monotonic() + self._get_retry_max_wait_s
        transfer_attempts = 0
        while True:
            data = bytes(self.client.get(key))
            if len(data) == expected_bytes and expected_bytes > 0:
                return data
            exists = False
            try:
                exists = bool(self.client.is_exist(key))
            except Exception:
                exists = bool(data)
            if len(data) == expected_bytes and exists:
                return data
            if data:
                # A non-empty size mismatch is metadata corruption, not a
                # transient missing transfer. Let bytes_to_tensor report it.
                return data
            if exists:
                # Mooncake returns b"" for both a missing key and a failed
                # data-plane read. Treating existence as read success would
                # bypass this retry loop and surface a misleading 0-byte
                # tensor. Retry transfer failures independently of the
                # missing-key deadline because one Mooncake get can itself
                # block longer than that deadline.
                transfer_attempts += 1
                if transfer_attempts >= self._get_transfer_max_attempts:
                    raise RuntimeError(
                        f"Mooncake side payload transfer returned no data for existing key {key!r} "
                        f"after {transfer_attempts} attempt(s); expected {expected_bytes} bytes"
                    )
                logger.warning(
                    "Retrying empty Mooncake side payload transfer key=%s attempt=%d/%d expected_bytes=%d",
                    key,
                    transfer_attempts,
                    self._get_transfer_max_attempts,
                    expected_bytes,
                )
                time.sleep(self._get_retry_interval_s)
                continue
            if time.monotonic() >= missing_deadline:
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
    if int(value.get("version", 0)) == 1 and count != len(items):
        raise ValueError(f"R3 side payload count mismatch for {field}: ref count={count}, metadata items={len(items)}")
    if int(value.get("version", 0)) == 2:
        covered = sum(int(item["datum_count"]) for item in items)
        if covered != count:
            raise ValueError(
                f"R3 packed side payload count mismatch for {field}: ref count={count}, chunks cover={covered}"
            )
    return count


def put_r3_mooncake_payload_refs(
    *,
    request_id: str,
    routed_experts: Optional[list[Any]],
    routed_expert_logits: Optional[list[Any]],
    store: MooncakeSidePayloadStore,
    namespace_prefix: Optional[str] = None,
    chunk_ranges: Optional[list[tuple[int, int]]] = None,
    max_chunk_bytes: int = DEFAULT_R3_PACKED_CHUNK_BYTES,
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
    payload_count = len(routed_experts if routed_experts is not None else routed_expert_logits or [])
    resolved_ranges = _validate_chunk_ranges(chunk_ranges, payload_count)
    written: list[Mapping[str, Any]] = []
    items: dict[str, list[dict[str, Any]]] = {}

    field_specs = []
    if routed_experts is not None:
        field_specs.append((R3_ROUTED_EXPERTS, routed_experts, torch.int32))
    if routed_expert_logits is not None:
        field_specs.append((R3_ROUTED_EXPERT_LOGITS, routed_expert_logits, torch.float32))

    def _put_field(field: str, payloads: list[Any], target_dtype: torch.dtype):
        field_written: list[Mapping[str, Any]] = []
        try:
            return _put_r3_field(
                store,
                namespace,
                field,
                payloads,
                target_dtype=target_dtype,
                chunk_ranges=resolved_ranges,
                max_chunk_bytes=max_chunk_bytes,
                written=field_written,
            )
        finally:
            # Preserve every successful put for request-wide rollback even if
            # this field or the other concurrent field ultimately fails.
            written.extend(field_written)

    try:
        if store.put_workers == 1 or len(field_specs) <= 1:
            field_results = [
                (field, _put_field(field, payloads, target_dtype)) for field, payloads, target_dtype in field_specs
            ]
        else:
            # Expert indices and routing weights are independent, equally
            # large fields.  Build and publish them concurrently while the
            # store-wide byte gate remains the single aggregate staging cap.
            # Resolve lazy construction before either field thread enters the
            # Mooncake client binding.
            _ = store.client
            field_results = []
            field_errors: list[BaseException] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(field_specs),
                thread_name_prefix="xorl-r3-mooncake-field",
            ) as executor:
                futures = [
                    (field, executor.submit(_put_field, field, payloads, target_dtype))
                    for field, payloads, target_dtype in field_specs
                ]
                concurrent.futures.wait([future for _, future in futures])
                for field, future in futures:
                    try:
                        entries = future.result()
                    except BaseException as exc:
                        field_errors.append(exc)
                    else:
                        field_results.append((field, entries))
            if field_errors:
                raise field_errors[0]

        for field, entries in field_results:
            items[field] = entries
    except Exception as put_error:
        rollback = R3PayloadCleanup(
            refs=({"items": {R3_ROUTED_EXPERTS: written}},),
            store=store,
        )
        rollback_stats = cleanup_r3_mooncake_payloads(rollback, force=True)
        if rollback_stats.failed or rollback_stats.pending:
            raise R3PayloadRollbackError(put_error, rollback_stats) from put_error
        raise

    base_ref: dict[str, Any] = {
        SIDE_PAYLOAD_REF_KEY: True,
        "backend": "mooncake",
        "kind": R3_ROUTING_KIND,
        "version": 2,
        "format": R3_PACKED_FORMAT,
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
    total = int(ref.get("count", len(items)))
    if start < 0 or count < 0 or start + count > total:
        raise ValueError(f"R3 side payload slice out of range for {field}: start={start}, count={count}, total={total}")
    payload_store = store or MooncakeSidePayloadStore.from_metadata(_require_mapping(ref.get("mooncake"), "mooncake"))
    arrays: list[np.ndarray] = []
    if int(ref.get("version", 0)) == 1:
        for entry in items[start : start + count]:
            tensor = payload_store.get_tensor_from_metadata(entry)
            arrays.append(tensor.detach().cpu().numpy())
        return arrays

    stop = start + count
    for entry in items:
        chunk_start = int(entry["datum_start"])
        chunk_stop = chunk_start + int(entry["datum_count"])
        overlap_start = max(start, chunk_start)
        overlap_stop = min(stop, chunk_stop)
        if overlap_start >= overlap_stop:
            continue
        tensor = payload_store.get_tensor_from_metadata(entry)
        offsets = entry["row_offsets"]
        for datum_idx in range(overlap_start, overlap_stop):
            local_idx = datum_idx - chunk_start
            row_start = int(offsets[local_idx])
            row_stop = int(offsets[local_idx + 1])
            arrays.append(tensor[row_start:row_stop].detach().cpu().numpy())
    if len(arrays) != count:
        raise ValueError(
            f"R3 packed side payload slice coverage mismatch for {field}: requested={count}, loaded={len(arrays)}"
        )
    return arrays


def cleanup_r3_mooncake_payloads(cleanup: R3PayloadCleanup, *, force: bool) -> R3PayloadCleanupStats:
    started_s = time.monotonic()
    seen: set[str] = set()
    unique_entries: list[Mapping[str, Any]] = []
    for ref in cleanup.refs:
        items_by_field = _require_mapping(ref.get("items"), "items")
        for payload_field in R3_ROUTING_FIELDS:
            entries = items_by_field.get(payload_field)
            if entries is None:
                continue
            if not isinstance(entries, list):
                logger.warning(
                    "Malformed R3 side payload cleanup entries for %s: %r",
                    payload_field,
                    entries,
                )
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                key = str(entry.get("key", ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                unique_entries.append(entry)

    total_bytes = sum(_tensor_metadata_nbytes(entry) for entry in unique_entries)
    deadline = started_s + cleanup.store._remove_retry_max_wait_s
    succeeded = 0
    already_absent = 0
    failed = 0
    retry_attempts = 0
    removed_bytes = 0
    failures: list[str] = []
    for entry in unique_entries:
        try:
            attempts, was_absent = cleanup.store.remove(entry, force=force, deadline=deadline)
        except Exception as exc:
            failed += 1
            failures.append(str(exc))
            continue
        succeeded += 1
        already_absent += int(was_absent)
        retry_attempts += max(0, attempts - 1)
        removed_bytes += _tensor_metadata_nbytes(entry)

    finished_s = time.monotonic()
    return R3PayloadCleanupStats(
        total=len(unique_entries),
        attempted=succeeded + failed,
        succeeded=succeeded,
        already_absent=already_absent,
        failed=failed,
        pending=len(unique_entries) - succeeded,
        retry_attempts=retry_attempts,
        removed_bytes=removed_bytes,
        retained_bytes=total_bytes - removed_bytes,
        elapsed_s=finished_s - started_s,
        oldest_key_age_s=max(0.0, finished_s - cleanup.created_monotonic_s),
        failures=tuple(failures),
    )


def _tensor_metadata_nbytes(metadata: Mapping[str, Any]) -> int:
    _, shape, dtype = parse_tensor_metadata(metadata)
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


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
    chunk_ranges: list[tuple[int, int]],
    max_chunk_bytes: int,
    written: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    chunks = list(
        iter_r3_packed_chunks(
            payloads,
            field=field,
            target_dtype=target_dtype,
            chunk_ranges=chunk_ranges,
            max_chunk_bytes=max_chunk_bytes,
        )
    )

    def _put(chunk_index: int, packed: torch.Tensor, chunk_metadata: dict[str, Any]) -> dict[str, Any]:
        metadata = store.put_tensor(f"{namespace}/{field}/chunk-{chunk_index:06d}", packed)
        metadata.update(chunk_metadata)
        return metadata

    if store.put_workers == 1 or len(chunks) <= 1:
        entries = []
        for index, (packed, metadata) in enumerate(chunks):
            entry = _put(index, packed, metadata)
            entries.append(entry)
            # Preserve each completed object immediately so a later chunk
            # failure cannot hide it from request-wide rollback.
            written.append(entry)
        return entries

    # Resolve lazy client construction before worker threads enter the binding.
    # The byte gate in put_tensor keeps aggregate staging below the configured
    # local buffer even when the worker count is larger than safe concurrency.
    _ = store.client
    entries: list[Optional[dict[str, Any]]] = [None] * len(chunks)
    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(store.put_workers, len(chunks)),
        thread_name_prefix="xorl-r3-mooncake-put",
    ) as executor:
        futures = [executor.submit(_put, index, packed, metadata) for index, (packed, metadata) in enumerate(chunks)]
        concurrent.futures.wait(futures)
        for index, future in enumerate(futures):
            try:
                entries[index] = future.result()
            except BaseException as exc:
                errors.append(exc)

    completed = [entry for entry in entries if entry is not None]
    written.extend(completed)
    if errors:
        raise errors[0]
    return completed


def iter_r3_packed_chunks(
    payloads: list[Any],
    *,
    field: str,
    target_dtype: torch.dtype,
    chunk_ranges: Optional[list[tuple[int, int]]] = None,
    max_chunk_bytes: int = DEFAULT_R3_PACKED_CHUNK_BYTES,
) -> Iterator[tuple[torch.Tensor, dict[str, Any]]]:
    """Yield bounded contiguous row tensors plus per-datum row offsets."""
    if max_chunk_bytes <= 0:
        raise ValueError(f"R3 packed max_chunk_bytes must be positive, got {max_chunk_bytes}")
    resolved_ranges = _validate_chunk_ranges(chunk_ranges, len(payloads))
    for range_start, range_count in resolved_ranges:
        pending: list[torch.Tensor] = []
        pending_start = range_start
        pending_bytes = 0
        for idx in range(range_start, range_start + range_count):
            tensor = canonicalize_r3_payload_item(payloads[idx], field=field, target_dtype=target_dtype)
            tensor_bytes = tensor.numel() * tensor.element_size()
            if pending and pending_bytes + tensor_bytes > max_chunk_bytes:
                yield _finalize_r3_packed_chunk(pending, pending_start, field)
                pending = []
                pending_start = idx
                pending_bytes = 0
            pending.append(tensor)
            pending_bytes += tensor_bytes
        if pending:
            yield _finalize_r3_packed_chunk(pending, pending_start, field)


def _finalize_r3_packed_chunk(
    tensors: list[torch.Tensor], datum_start: int, field: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    trailing_shape = tuple(tensors[0].shape[1:])
    if any(tuple(tensor.shape[1:]) != trailing_shape for tensor in tensors):
        raise ValueError(f"R3 {field} payload has inconsistent layer/top-k shapes within a packed chunk")
    row_offsets = [0]
    for tensor in tensors:
        row_offsets.append(row_offsets[-1] + int(tensor.shape[0]))
    packed = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
    return packed, {
        "datum_start": datum_start,
        "datum_count": len(tensors),
        "row_offsets": row_offsets,
    }


def _validate_chunk_ranges(ranges: Optional[list[tuple[int, int]]], count: int) -> list[tuple[int, int]]:
    if ranges is None:
        return [(0, count)] if count else []
    resolved = [(int(start), int(length)) for start, length in ranges if int(length) > 0]
    cursor = 0
    for start, length in resolved:
        if start != cursor or length < 0 or start + length > count:
            raise ValueError(f"R3 packed chunk ranges must cover [0, {count}) contiguously, got {resolved!r}")
        cursor += length
    if cursor != count:
        raise ValueError(f"R3 packed chunk ranges cover [0, {cursor}), expected [0, {count})")
    return resolved


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
    rows = item.get("rows", shape[0])
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0 or rows > shape[0]:
        raise ValueError(f"R3 {field} dict payload has invalid rows view {rows!r} for shape {shape}")
    arr = arr[:rows]
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
    version = int(ref.get("version", 0))
    expected_start = 0
    for idx, entry in enumerate(entries):
        entry = _require_mapping(entry, f"items[{field!r}][{idx}]")
        _, shape, _ = parse_tensor_metadata(entry)
        if version == 2:
            datum_start = int(entry.get("datum_start", -1))
            datum_count = int(entry.get("datum_count", -1))
            offsets = entry.get("row_offsets")
            if datum_start != expected_start or datum_count < 1:
                raise ValueError(f"R3 packed side payload chunk {idx} is not contiguous")
            if (
                not isinstance(offsets, list)
                or len(offsets) != datum_count + 1
                or offsets[0] != 0
                or any(not isinstance(value, int) or value < 0 for value in offsets)
                or any(left > right for left, right in zip(offsets, offsets[1:]))
                or offsets[-1] != shape[0]
            ):
                raise ValueError(f"R3 packed side payload chunk {idx} has invalid row_offsets")
            expected_start += datum_count
    return str(field), entries


def _validate_r3_ref_header(ref: Mapping[str, Any]) -> None:
    if not is_side_payload_ref(ref):
        raise ValueError(f"R3 side payload ref missing {SIDE_PAYLOAD_REF_KEY!r}")
    if str(ref.get("backend", "")).lower() != "mooncake":
        raise ValueError(f"R3 side payload ref backend must be 'mooncake', got {ref.get('backend')!r}")
    if ref.get("kind") != R3_ROUTING_KIND:
        raise ValueError(f"R3 side payload ref kind must be {R3_ROUTING_KIND!r}, got {ref.get('kind')!r}")
    version = int(ref.get("version", 0))
    if version not in (1, 2):
        raise ValueError(f"R3 side payload ref version must be 1 or 2, got {ref.get('version')!r}")
    if version == 2 and ref.get("format") != R3_PACKED_FORMAT:
        raise ValueError(f"R3 side payload ref version 2 must use format {R3_PACKED_FORMAT!r}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"R3 side payload metadata field {name!r} must be a mapping, got {type(value)!r}")
    return value


def _safe_namespace_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
