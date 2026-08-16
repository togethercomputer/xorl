"""
RequestProcessor - Batch Preparation and Backend Dispatch.

This module provides the RequestProcessor class which handles data preparation (packing,
validation) and delegates compute operations to a Backend implementation.

Role in Orchestrator Architecture:
================================

Orchestrator orchestrates three components:
1. **Scheduler**: Orders requests (FIFO policy)
2. **RequestProcessor**: Prepares data and dispatches to Backend (THIS MODULE)
3. **Queues**: Thread-safe communication (input_queue, output_queue)

The RequestProcessor owns:
- Sample packing (datum_list → micro-batches)
- Result formatting (backend result → OrchestratorOutputs)
- Operation statistics

The Backend owns:
- Transport layer (ZMQ, in-process, etc.)
- Worker handshake and lifecycle
- Request serialization and response deserialization

Usage:
=====

```python
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.backend import RemoteBackend

backend = RemoteBackend(worker_address="tcp://127.0.0.1:5556")
executor = RequestProcessor(backend=backend, sample_packing_sequence_len=32000)

await executor.start()
outputs = await executor.execute_forward_backward(request)
await executor.stop()
```
"""

import json
import logging
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.server.backend import Backend
from xorl.server.orchestrator.packing import pack_samples, unpack_per_token_outputs, validate_micro_batches
from xorl.server.protocol.api_orchestrator import OrchestratorOutputs, OrchestratorRequest, OutputType
from xorl.server.protocol.operations import (
    AdapterStateData,
    KillSessionData,
    LoadStateData,
    ModelPassData,
    OptimStepData,
    RegisterAdapterData,
    RegisterSessionData,
    SaveFullWeightsData,
    SaveLoraOnlyData,
    SaveStateData,
    SyncWeightsData,
)
from xorl.server.runner.utils import batch_packed_rows
from xorl.server.side_payloads import (
    DEFAULT_R3_PACKED_CHUNK_BYTES,
    R3_ROUTED_EXPERT_LOGITS,
    R3_ROUTED_EXPERTS,
    MooncakeSidePayloadStore,
    R3PayloadCleanup,
    R3PayloadCleanupStats,
    R3PayloadRollbackError,
    cleanup_r3_mooncake_payloads,
    iter_r3_packed_chunks,
    put_r3_mooncake_payload_refs,
)
from xorl.utils.seqlen_pos_transform_utils import pos2culen


logger = logging.getLogger(__name__)


FORWARD_BACKWARD_RESULT_PREFIXES = ("forward_backward_", "server_profile_", "r3_replay_")
FORWARD_BACKWARD_RESULT_KEYS = {
    "backward_compute_time",
    "forward_compute_time",
}
ROUTING_PAYLOAD_REF_KEY = "__xorl_routing_payload_ref__"
R3_SPANS_SCHEMA = "xorl.r3.spans.v1"
SGLANG_R3_FILE_SCHEMA = "sglang.routed_experts.file.v1"


@dataclass(frozen=True)
class R3SourceFilesCleanup:
    paths: tuple[Path, ...]


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _r3_verbose_logging_enabled() -> bool:
    return _truthy_env("XORL_R3_VERBOSE_LOGGING")


# Metadata and precomputed attention fields that should not be stacked as
# sequence-aligned lists when grouping packed rows.
_ROW_BATCH_METADATA_KEYS = {
    "request_id",
    "batch_id",
    "num_samples",
    "_r3_sample_lengths",
    "_shifted",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
}


def _positive_int_param(value: Any, *, name: str, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1, not a bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def _is_sequence_row(value: Any, row_len: int) -> bool:
    return isinstance(value, list) and len(value) == 1 and isinstance(value[0], list) and len(value[0]) == row_len


def _pad_sequence_value(key: str, seq: list[Any], target_len: int) -> list[Any]:
    pad_len = target_len - len(seq)
    if pad_len <= 0:
        return list(seq)
    if key == "labels":
        return list(seq) + [IGNORE_INDEX] * pad_len
    if key == "position_ids":
        return list(seq) + list(range(pad_len))
    if seq and isinstance(seq[0], list):
        width = len(seq[0])
        return list(seq) + [[0.0] * width for _ in range(pad_len)]
    return list(seq) + [0] * pad_len


# ============================================================================
# RequestProcessor Class
# ============================================================================


class RequestProcessor:
    """
    Batch preparation and backend dispatch coordinator.

    Handles data preparation (packing, validation) and delegates compute
    operations to a Backend implementation. The Backend handles all transport
    concerns (ZMQ, in-process, etc.).
    """

    def __init__(
        self,
        backend: Backend,
        sample_packing_sequence_len: int = 32000,
        enable_packing: bool = True,
        pad_to_multiple_of: int = 128,
        cp_size: int = 1,
        packing_strategy: str = "sequential",
        on_oversized: str = "error",
        dp_size: int = 1,
        r3_payload_transport: str = "inline",
        r3_payload_dir: Optional[str] = None,
        r3_payload_keep: bool = False,
        r3_payload_namespace_prefix: Optional[str] = None,
        routing_payload_dir: Optional[str] = None,
        keep_routing_payloads: Optional[bool] = None,
        routing_payload_store: Optional[MooncakeSidePayloadStore] = None,
    ):
        """
        Initialize RequestProcessor.

        Args:
            backend: Backend implementation for compute operations
            sample_packing_sequence_len: Maximum sequence length for packing (default: 32000)
            enable_packing: Enable sample packing (default: True)
            pad_to_multiple_of: Base padding alignment (default: 128)
            cp_size: Sequence parallel size. Padded length must be divisible
                by cp_size for Ulysses sequence parallelism. The effective
                padding multiple is lcm(pad_to_multiple_of, cp_size).
            packing_strategy: Bin-packing strategy (see packing.PACKING_STRATEGIES).
            on_oversized: How to handle samples longer than the pack length
                (see packing.ON_OVERSIZED_MODES). Default "error" (no silent drop).
            dp_size: Number of distinct dispatcher batch slices
                (world_size // (cp_size·pp_size)); used by the "balanced_dp" strategy.
            r3_payload_transport: "inline", "mooncake", or explicit "filesystem" fallback.
            r3_payload_dir: Shared directory used only by the explicit filesystem fallback.
            r3_payload_keep: If True, do not delete side payloads after the backend call.
            r3_payload_namespace_prefix: Optional Mooncake namespace prefix for R3 payload keys.
            routing_payload_dir: Backward-compatible alias for filesystem transport.
            keep_routing_payloads: Backward-compatible alias for r3_payload_keep.
            routing_payload_store: Optional injected Mooncake side-payload store for tests.
        """
        self.backend = backend
        self.sample_packing_sequence_len = sample_packing_sequence_len
        self.enable_packing = enable_packing
        # Sequence must be divisible by both pad_to_multiple_of and cp_size
        self.pad_to_multiple_of = math.lcm(pad_to_multiple_of, cp_size)
        self.packing_strategy = packing_strategy
        self.on_oversized = on_oversized
        self.dp_size = max(1, int(dp_size))
        if routing_payload_dir is not None:
            if r3_payload_transport != "inline":
                raise ValueError("routing_payload_dir alias cannot be combined with r3_payload_transport")
            r3_payload_transport = "filesystem"
            r3_payload_dir = routing_payload_dir
        if keep_routing_payloads is not None:
            r3_payload_keep = bool(keep_routing_payloads)
        if r3_payload_transport not in {"inline", "mooncake", "filesystem"}:
            raise ValueError(f"Unsupported r3_payload_transport {r3_payload_transport!r}")
        if r3_payload_transport == "inline" and r3_payload_keep:
            raise ValueError("r3_payload_keep requires r3_payload_transport != 'inline'")
        if r3_payload_transport != "filesystem" and r3_payload_dir:
            raise ValueError("r3_payload_dir is only valid with r3_payload_transport='filesystem'")
        if r3_payload_transport != "mooncake" and r3_payload_namespace_prefix:
            raise ValueError("r3_payload_namespace_prefix is only valid with r3_payload_transport='mooncake'")
        self.r3_payload_transport = r3_payload_transport
        self.r3_payload_dir = Path(r3_payload_dir) if r3_payload_dir else None
        self.r3_payload_keep = bool(r3_payload_keep)
        self.r3_payload_namespace_prefix = r3_payload_namespace_prefix
        self._routing_payload_store = routing_payload_store
        self._r3_cleanup_blocked_error: Optional[str] = None

        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0

        logger.info(
            f"RequestProcessor initialized: "
            f"sample_packing_sequence_len={sample_packing_sequence_len}, packing={'enabled' if enable_packing else 'disabled'}, "
            f"pad_to_multiple_of={self.pad_to_multiple_of} (base={pad_to_multiple_of}, cp_size={cp_size}), "
            f"strategy={packing_strategy}, on_oversized={on_oversized}, dp_size={self.dp_size}"
        )
        if self.r3_payload_transport != "inline":
            logger.info(
                "External R3 routing payload transport enabled: transport=%s keep_payloads=%s",
                self.r3_payload_transport,
                self.r3_payload_keep,
            )

    @staticmethod
    def _safe_request_id(request_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(request_id))

    def _externalize_routing_payloads(
        self,
        request_id: str,
        routed_experts: Optional[List[Any]],
        routed_expert_logits: Optional[List[Any]],
        *,
        batches: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[Optional[Any], Optional[Any], Optional[Union[Path, R3PayloadCleanup, R3SourceFilesCleanup]]]:
        if routed_experts is None and routed_expert_logits is None:
            return routed_experts, routed_expert_logits, None
        direct = self._externalize_sglang_routing_spans(routed_experts, routed_expert_logits)
        if direct is not None:
            return direct
        if self.r3_payload_transport == "inline":
            return routed_experts, routed_expert_logits, None
        if self.r3_payload_transport == "mooncake":
            if self._r3_cleanup_blocked_error is not None:
                raise RuntimeError(
                    "Refusing to externalize another R3 Mooncake payload after incomplete cleanup: "
                    f"{self._r3_cleanup_blocked_error}"
                )
            if self._routing_payload_store is None:
                self._routing_payload_store = MooncakeSidePayloadStore()
            store = self._routing_payload_store
            try:
                refs = put_r3_mooncake_payload_refs(
                    request_id=request_id,
                    routed_experts=routed_experts,
                    routed_expert_logits=routed_expert_logits,
                    store=store,
                    namespace_prefix=self.r3_payload_namespace_prefix,
                    chunk_ranges=self._routing_payload_chunk_ranges(
                        batches, len(routed_experts or routed_expert_logits or [])
                    ),
                )
            except R3PayloadRollbackError as exc:
                self._r3_cleanup_blocked_error = str(exc)
                raise
            log_fn = logger.info if _r3_verbose_logging_enabled() else logger.debug
            log_fn(
                "Externalized R3 routing payload request=%s transport=mooncake routed=%s routed_weights=%s",
                request_id,
                len(routed_experts or []),
                len(routed_expert_logits or []),
            )
            return refs

        return self._externalize_routing_payloads_filesystem(
            request_id,
            routed_experts,
            routed_expert_logits,
            chunk_ranges=self._routing_payload_chunk_ranges(batches, len(routed_experts or routed_expert_logits or [])),
        )

    @staticmethod
    def _externalize_sglang_routing_spans(
        routed_experts: Optional[List[Any]],
        routed_expert_logits: Optional[List[Any]],
    ) -> Optional[tuple[Optional[Any], Optional[Any], R3SourceFilesCleanup]]:
        def _normalize(items: Optional[List[Any]], kind: str) -> Optional[List[Any]]:
            if items is None:
                return None
            return [RequestProcessor._normalize_sglang_routing_file(item, kind) for item in items]

        routed_experts = _normalize(routed_experts, R3_ROUTED_EXPERTS)
        routed_expert_logits = _normalize(routed_expert_logits, R3_ROUTED_EXPERT_LOGITS)
        present = [items for items in (routed_experts, routed_expert_logits) if items is not None]
        if not present:
            return None
        has_spans = [
            isinstance(item, dict) and item.get("schema") == R3_SPANS_SCHEMA for items in present for item in items
        ]
        if not any(has_spans):
            return None
        if not all(has_spans):
            raise ValueError("R3 span payloads cannot be mixed with inline routing payloads")

        source_paths: set[Path] = set()
        for items in present:
            for item in items:
                spans = item.get("spans")
                if not isinstance(spans, list):
                    raise ValueError("R3 span payload must contain a spans list")
                if sum(int(span.get("rows", -1)) for span in spans if isinstance(span, dict)) != int(
                    item.get("rows", -1)
                ):
                    raise ValueError("R3 span payload rows do not match its span coverage")
                for span in spans:
                    if not isinstance(span, dict):
                        raise ValueError("R3 span entry must be an object")
                    path = RequestProcessor._validate_r3_source_path(span.get("path"), final=True)
                    source_paths.add(path)
                    error_path = span.get("error_path")
                    if error_path:
                        source_paths.add(RequestProcessor._validate_r3_source_path(error_path, final=False))

        def _ref(field: str, items: Optional[List[Any]]) -> Optional[Dict[str, Any]]:
            if items is None:
                return None
            return {
                ROUTING_PAYLOAD_REF_KEY: True,
                "transport": "sglang_files",
                "version": 1,
                "format": "spans",
                "kind": field,
                "count": len(items),
                "items": items,
            }

        return (
            _ref(R3_ROUTED_EXPERTS, routed_experts),
            _ref(R3_ROUTED_EXPERT_LOGITS, routed_expert_logits),
            R3SourceFilesCleanup(paths=tuple(sorted(source_paths))),
        )

    @staticmethod
    def _normalize_sglang_routing_file(item: Any, kind: str) -> Any:
        if not isinstance(item, dict) or item.get("schema") != SGLANG_R3_FILE_SCHEMA:
            return item
        if item.get("field") != kind:
            raise ValueError(f"SGLang R3 descriptor field does not match {kind}")
        fields = item.get("fields")
        field = fields.get(kind) if isinstance(fields, dict) else None
        if not isinstance(field, dict):
            raise ValueError(f"SGLang R3 descriptor is missing {kind} metadata")
        shape = field.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"SGLang R3 descriptor has invalid {kind} shape")
        rows = int(item.get("rows", -1))
        row_nbytes = math.prod(int(dim) for dim in shape[1:]) * 4
        if rows != int(shape[0]) or int(field.get("nbytes", -1)) != rows * row_nbytes:
            raise ValueError(f"SGLang R3 descriptor has inconsistent {kind} geometry")
        return {
            "schema": R3_SPANS_SCHEMA,
            "rows": rows,
            "shape": [int(dim) for dim in shape],
            "dtype": field.get("dtype"),
            "spans": [
                {
                    "path": item.get("path"),
                    "error_path": item.get("error_path"),
                    "offset": int(field.get("offset", -1)),
                    "source_row": 0,
                    "rows": rows,
                    "row_nbytes": row_nbytes,
                    "source_shape": [int(dim) for dim in shape],
                    "dtype": field.get("dtype"),
                }
            ],
        }

    @staticmethod
    def _validate_r3_source_path(raw: Any, *, final: bool) -> Path:
        path = Path(str(raw or ""))
        if not path.is_absolute() or (final and path.name.startswith(".")):
            raise ValueError(f"R3 source path must be an absolute payload path: {path}")
        configured = os.getenv("XORL_R3_SHARED_ROOTS", "")
        roots = [
            Path(entry).expanduser().resolve(strict=True) for entry in configured.split(os.pathsep) if entry.strip()
        ]
        if not roots:
            raise ValueError("XORL_R3_SHARED_ROOTS must name the trusted SGLang side-channel root")
        parent = path.parent.resolve(strict=True)
        if not any(parent == root or root in parent.parents for root in roots):
            raise ValueError(f"R3 source path is outside XORL_R3_SHARED_ROOTS: {path}")
        return path

    def _routing_payload_chunk_ranges(
        self,
        batches: Optional[List[Dict[str, Any]]],
        datum_count: int,
    ) -> Optional[List[tuple[int, int]]]:
        """Align packed side-payload chunks with the dispatcher's DP datum slices."""
        if not batches or datum_count == 0:
            return None
        num_batches = len(batches)
        base_count = num_batches // self.dp_size
        remainder = num_batches % self.dp_size
        ranges: List[tuple[int, int]] = []
        datum_cursor = 0
        for dp_rank in range(self.dp_size):
            batch_start = dp_rank * base_count + min(dp_rank, remainder)
            batch_count = base_count + (1 if dp_rank < remainder else 0)
            dp_datums = sum(
                int(batches[idx].get("num_samples", 1)) for idx in range(batch_start, batch_start + batch_count)
            )
            if dp_datums:
                ranges.append((datum_cursor, dp_datums))
                datum_cursor += dp_datums
        if datum_cursor != datum_count:
            raise ValueError(f"R3 packed chunk ranges cover {datum_cursor} datums from batches, expected {datum_count}")
        return ranges

    def _externalize_routing_payloads_filesystem(
        self,
        request_id: str,
        routed_experts: Optional[List[Any]],
        routed_expert_logits: Optional[List[Any]],
        *,
        chunk_ranges: Optional[List[tuple[int, int]]] = None,
    ) -> tuple[Optional[Any], Optional[Any], Optional[Path]]:
        if self.r3_payload_dir is None:
            raise ValueError("r3_payload_dir is required for r3_payload_transport='filesystem'")
        safe_request_id = self._safe_request_id(request_id) or "request"
        unique_request_id = f"{safe_request_id}.{uuid.uuid4().hex[:12]}"
        root = self.r3_payload_dir / unique_request_id
        tmp_root = self.r3_payload_dir / f".{unique_request_id}.tmp.{os.getpid()}"

        try:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)

            def _write_chunks(kind: str, items: Optional[List[Any]]) -> Optional[Dict[str, Any]]:
                if items is None:
                    return None
                item_dir = tmp_root / kind
                item_dir.mkdir(parents=True, exist_ok=True)
                if kind == R3_ROUTED_EXPERTS:
                    target_dtype = torch.int32
                    dtype_name = "int32"
                elif kind == R3_ROUTED_EXPERT_LOGITS:
                    target_dtype = torch.float32
                    dtype_name = "float32"
                else:  # pragma: no cover - both callers use the constants above
                    raise ValueError(f"Unsupported R3 filesystem payload kind {kind!r}")

                metadata = []
                for idx, (tensor, chunk_metadata) in enumerate(
                    iter_r3_packed_chunks(
                        items,
                        field=kind,
                        target_dtype=target_dtype,
                        chunk_ranges=chunk_ranges,
                        max_chunk_bytes=DEFAULT_R3_PACKED_CHUNK_BYTES,
                    )
                ):
                    data = tensor.numpy().tobytes(order="C")
                    filename = f"chunk-{idx:06d}.bin"
                    (item_dir / filename).write_bytes(data)
                    chunk_metadata.update(
                        {
                            "file": filename,
                            "shape": [int(dim) for dim in tensor.shape],
                            "dtype": dtype_name,
                            "nbytes": len(data),
                        }
                    )
                    metadata.append(chunk_metadata)
                return {"count": len(items), "chunks": metadata}

            manifest = {
                "format": "xorl-r3-packed",
                "version": 3,
                "request_id": str(request_id),
                R3_ROUTED_EXPERTS: _write_chunks(R3_ROUTED_EXPERTS, routed_experts),
                R3_ROUTED_EXPERT_LOGITS: _write_chunks(R3_ROUTED_EXPERT_LOGITS, routed_expert_logits),
            }
            (tmp_root / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )

            tmp_root.rename(root)
        except Exception:
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise

        def _ref(kind: str, items: Optional[List[Any]]) -> Optional[Dict[str, Any]]:
            if items is None:
                return None
            return {
                ROUTING_PAYLOAD_REF_KEY: True,
                "transport": "filesystem",
                "version": 3,
                "format": "packed_rows",
                "manifest": str(root / "manifest.json"),
                "kind": kind,
                "count": len(items),
            }

        log_fn = logger.info if _r3_verbose_logging_enabled() else logger.debug
        log_fn(
            "Externalized R3 routing payload request=%s dir=%s routed=%s routed_weights=%s",
            request_id,
            root,
            len(routed_experts or []),
            len(routed_expert_logits or []),
        )
        return _ref("routed_experts", routed_experts), _ref("routed_expert_logits", routed_expert_logits), root

    def _cleanup_routing_payloads(
        self,
        cleanup: Optional[Union[Path, R3PayloadCleanup, R3SourceFilesCleanup]],
        *,
        force: bool,
    ) -> Optional[R3PayloadCleanupStats]:
        if cleanup is None or self.r3_payload_keep:
            return None
        if isinstance(cleanup, R3PayloadCleanup):
            try:
                stats = cleanup_r3_mooncake_payloads(cleanup, force=force)
            except Exception as exc:
                message = f"cleanup raised {type(exc).__name__}: {exc}"
                self._r3_cleanup_blocked_error = message
                logger.exception("R3 Mooncake cleanup failed before producing statistics")
                return R3PayloadCleanupStats(
                    total=0,
                    attempted=0,
                    succeeded=0,
                    already_absent=0,
                    failed=1,
                    pending=1,
                    retry_attempts=0,
                    removed_bytes=0,
                    retained_bytes=0,
                    elapsed_s=0.0,
                    oldest_key_age_s=0.0,
                    failures=(message,),
                )
            if stats.failed or stats.pending:
                message = (
                    f"force={force} attempted={stats.attempted}/{stats.total} "
                    f"succeeded={stats.succeeded} failed={stats.failed} pending={stats.pending} "
                    f"retained_bytes={stats.retained_bytes} failures={stats.failures}"
                )
                self._r3_cleanup_blocked_error = message
                logger.error("Incomplete R3 Mooncake cleanup: %s", message)
                return stats
            log_fn = logger.info if _r3_verbose_logging_enabled() else logger.debug
            log_fn(
                "Cleaned external R3 Mooncake routing payload keys force=%s keys=%d bytes=%d retries=%d elapsed=%.3fs",
                force,
                stats.succeeded,
                stats.removed_bytes,
                stats.retry_attempts,
                stats.elapsed_s,
            )
            return stats
        if isinstance(cleanup, R3SourceFilesCleanup):
            for path in cleanup.paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Failed to clean SGLang R3 source file %s: %s", path, exc)
            return None
        self._cleanup_routing_payload_dir(cleanup)
        return None

    def _cleanup_routing_payload_dir(self, root: Path) -> None:
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("Failed to clean external R3 routing payload dir %s: %s", root, exc)
            return
        log_fn = logger.info if _r3_verbose_logging_enabled() else logger.debug
        log_fn("Cleaned external R3 routing payload dir %s", root)

    @staticmethod
    def _packed_row_sequence_keys(batch: Dict[str, Any]) -> set[str] | None:
        input_rows = batch.get("input_ids")
        if not isinstance(input_rows, list) or len(input_rows) != 1 or not isinstance(input_rows[0], list):
            return None
        row_len = len(input_rows[0])
        sequence_keys: set[str] = set()
        for key, value in batch.items():
            if key in _ROW_BATCH_METADATA_KEYS:
                continue
            if _is_sequence_row(value, row_len):
                sequence_keys.add(key)
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                continue
            return None
        required = {"input_ids", "labels", "position_ids"}
        if not required.issubset(sequence_keys):
            return None
        return sequence_keys

    @classmethod
    def _can_batch_packed_rows(cls, rows: list[Dict[str, Any]]) -> tuple[bool, set[str]]:
        if not rows:
            return False, set()
        first_keys = cls._packed_row_sequence_keys(rows[0])
        if first_keys is None:
            return False, set()
        scalar_keys = set(rows[0]) - first_keys - _ROW_BATCH_METADATA_KEYS
        for row in rows[1:]:
            row_keys = cls._packed_row_sequence_keys(row)
            if row_keys != first_keys:
                return False, set()
            if set(row) - row_keys - _ROW_BATCH_METADATA_KEYS != scalar_keys:
                return False, set()
            for key in scalar_keys:
                if row.get(key) != rows[0].get(key):
                    return False, set()
        return True, first_keys

    @classmethod
    def _merge_packed_row_group(
        cls, rows: list[Dict[str, Any]], batch_id: int, sequence_keys: set[str]
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "request_id": rows[0]["request_id"],
            "batch_id": batch_id,
            "num_samples": sum(int(row.get("num_samples", 0)) for row in rows),
            "_r3_sample_lengths": [length for row in rows for length in row.get("_r3_sample_lengths", [])],
        }
        if "_shifted" in rows[0]:
            merged["_shifted"] = all(bool(row.get("_shifted", False)) for row in rows)

        for key in sorted(sequence_keys):
            merged[key] = [[item for row in rows for item in row[key][0]]]

        for key in set(rows[0]) - sequence_keys - _ROW_BATCH_METADATA_KEYS:
            merged[key] = rows[0][key]

        if "position_ids" in merged:
            position_ids_tensor = torch.tensor(merged["position_ids"], dtype=torch.long)
            cu_seqlens = pos2culen(position_ids_tensor)
            merged["cu_seq_lens_q"] = cu_seqlens.tolist()
            merged["cu_seq_lens_k"] = cu_seqlens.tolist()
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            max_length = int(lengths.max().item()) if lengths.numel() else 0
            merged["max_length_q"] = max_length
            merged["max_length_k"] = max_length

        return merged

    @classmethod
    def _batch_packed_rows(cls, batches: list[Dict[str, Any]], row_batch_size: int) -> list[Dict[str, Any]]:
        return batch_packed_rows(batches, row_batch_size)

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def start(self):
        """Start the executor and its backend."""
        logger.info("Starting RequestProcessor...")
        await self.backend.start()
        logger.info("RequestProcessor started successfully")

    async def stop(self):
        """Stop the executor and its backend."""
        logger.info("Stopping RequestProcessor...")
        await self.backend.stop()
        if self._routing_payload_store is not None:
            self._routing_payload_store.close()
        logger.info("RequestProcessor stopped")

    def is_ready(self) -> bool:
        """Check if executor is ready for operations."""
        return self.backend.is_ready()

    # ========================================================================
    # Operation Execution
    # ========================================================================

    async def _execute_model_pass(
        self,
        request: OrchestratorRequest,
        op_name: str,
        output_type: OutputType,
    ) -> OrchestratorOutputs:
        """Shared implementation for forward and forward_backward passes.

        Args:
            request: OrchestratorRequest with data and loss_fn
            op_name: Operation name ("forward" or "forward_backward")
            output_type: OutputType for the response
        """
        logger.debug(f"Executing {op_name} for request {request.request_id}")
        self.total_operations += 1
        t0 = time.perf_counter()

        try:
            # Extract parameters from typed payload
            p: ModelPassData = request.payload
            data = p.data
            loss_fn = p.loss_fn
            loss_fn_params = p.loss_fn_params or {}
            routed_experts = p.routed_experts
            routed_expert_logits = p.routed_expert_logits

            if not data:
                raise ValueError("data or datum_list must be provided")

            self._validate_routing_payload_counts(
                len(data),
                routed_experts,
                routed_expert_logits,
                context="input data",
            )

            if loss_fn == "opd_loss" and loss_fn_params.get("opd_sort_by_teacher", True):
                order = sorted(range(len(data)), key=lambda i: self._teacher_sort_key(data[i]))
                data = [data[i] for i in order]
                if routed_experts is not None:
                    routed_experts = [routed_experts[i] for i in order]
                if routed_expert_logits is not None:
                    routed_expert_logits = [routed_expert_logits[i] for i in order]

            # Pack samples into batches
            logger.debug(f"Packing {len(data)} datum into batches for {op_name} request {request.request_id}")
            batches, datum_order = pack_samples(
                datum_list=data,
                max_seq_len=self.sample_packing_sequence_len,
                enable_packing=self.enable_packing,
                request_id=request.request_id,
                pad_to_multiple_of=self.pad_to_multiple_of,
                strategy=self.packing_strategy,
                on_oversized=self.on_oversized,
                dp_size=self.dp_size,
                return_datum_order=True,
            )

            # Realign per-datum side arrays to the order samples appear in the
            # emitted micro-batches. The dispatcher slices routed_experts /
            # routed_expert_logits by cumulative num_samples in batch order, so
            # any packer reordering (or dropped samples) must be mirrored here.
            if (routed_experts is not None or routed_expert_logits is not None) and datum_order != list(
                range(len(data))
            ):
                if routed_experts is not None:
                    routed_experts = [routed_experts[i] for i in datum_order]
                if routed_expert_logits is not None:
                    routed_expert_logits = [routed_expert_logits[i] for i in datum_order]

            routing_payload_root = None
            if not batches:
                raise ValueError(
                    f"No batches created from {len(data)} samples. The packer did not produce any valid batches."
                )

            if not validate_micro_batches(batches):
                raise ValueError("Invalid batch structure after packing. This may indicate a bug in the packing logic.")

            packed_sample_count = sum(int(batch.get("num_samples", 1)) for batch in batches)
            self._validate_routing_payload_counts(
                packed_sample_count,
                routed_experts,
                routed_expert_logits,
                context="packed batches",
            )

            original_batch_count = len(batches)
            row_batch_size = _positive_int_param(
                loss_fn_params.get("opd_packed_row_batch_size", loss_fn_params.get("packed_row_batch_size")),
                name="opd_packed_row_batch_size",
                default=1,
            )
            if row_batch_size > 1 and (routed_experts is not None or routed_expert_logits is not None):
                raise ValueError("opd_packed_row_batch_size is not supported with routed_experts replay")

            t_before_externalize = time.perf_counter()
            routed_experts, routed_expert_logits, routing_payload_root = self._externalize_routing_payloads(
                request.request_id,
                routed_experts,
                routed_expert_logits,
                batches=batches,
            )
            t_after_externalize = time.perf_counter()

            if row_batch_size > 1:
                row_batch_scope = str(loss_fn_params.get("opd_packed_row_batch_scope", "rank_local")).lower()
                if row_batch_scope in {"global", "executor", "orchestrator"}:
                    batches = self._batch_packed_rows(batches, row_batch_size)
                    if not validate_micro_batches(batches):
                        raise ValueError("Invalid batch structure after packed-row batching.")
                    logger.info(
                        "Executor-global packed-row batching enabled for %s request %s: %d -> %d backend batches "
                        "(row_batch_size=%d)",
                        op_name,
                        request.request_id,
                        original_batch_count,
                        len(batches),
                        row_batch_size,
                    )
                elif row_batch_scope in {"rank_local", "rank-local", "local", "per_rank", "per-rank"}:
                    logger.info(
                        "Deferring packed-row batching to rank-local runner slices for %s request %s "
                        "(executor batches=%d, row_batch_size=%d)",
                        op_name,
                        request.request_id,
                        original_batch_count,
                        row_batch_size,
                    )
                else:
                    raise ValueError(
                        "opd_packed_row_batch_scope must be one of "
                        "rank_local, global, executor, or orchestrator; "
                        f"got {row_batch_scope!r}"
                    )

            t_packed = time.perf_counter()
            logger.debug(f"Packed {len(data)} samples into {len(batches)} batches")

            # Call backend
            backend_method = getattr(self.backend, op_name)
            kwargs = dict(
                batches=batches,
                loss_fn=loss_fn,
                loss_fn_params=loss_fn_params,
                model_id=p.model_id,
                routed_experts=routed_experts,
                routed_expert_logits=routed_expert_logits,
                request_id=request.request_id,
            )

            routing_cleanup_stats = None
            try:
                result = await backend_method(**kwargs)
            except BaseException:
                # A failed/cancelled backend may still have a rank fetching a
                # payload. Normal removal respects that lease; never force it.
                self._cleanup_routing_payloads(routing_payload_root, force=False)
                raise
            else:
                # RunnerDispatcher returns only after its mandatory all-rank
                # completion rendezvous, so every synchronous get has returned.
                routing_cleanup_stats = self._cleanup_routing_payloads(routing_payload_root, force=True)

            t_backend = time.perf_counter()

            # Build output dict
            loss = result.get("total_loss", 0.0)
            tokens = result.get("global_valid_tokens", 0)

            output_dict = {
                "loss": loss,
                "valid_tokens": tokens,
                "success": True,
                "execution_time": result.get(
                    "execution_time", result.get("forward_backward_time", result.get("forward_time", 0.0))
                ),
                "executor_pack_s": t_packed - t0,
                "executor_r3_externalize_s": t_after_externalize - t_before_externalize,
                "executor_backend_s": t_backend - t_packed,
                "executor_build_output_s": 0.0,  # Filled after output construction.
                "executor_total_s": 0.0,  # Filled after output construction.
                "executor_batches": len(batches),
                "executor_original_batches": original_batch_count,
                "executor_packed_row_batch_size": row_batch_size,
                "executor_samples": len(data),
            }
            if routing_cleanup_stats is not None:
                output_dict.update(
                    {
                        "executor_r3_cleanup_total": routing_cleanup_stats.total,
                        "executor_r3_cleanup_attempted": routing_cleanup_stats.attempted,
                        "executor_r3_cleanup_succeeded": routing_cleanup_stats.succeeded,
                        "executor_r3_cleanup_already_absent": routing_cleanup_stats.already_absent,
                        "executor_r3_cleanup_failed": routing_cleanup_stats.failed,
                        "executor_r3_cleanup_pending": routing_cleanup_stats.pending,
                        "executor_r3_cleanup_retry_attempts": routing_cleanup_stats.retry_attempts,
                        "executor_r3_cleanup_removed_bytes": routing_cleanup_stats.removed_bytes,
                        "executor_r3_cleanup_retained_bytes": routing_cleanup_stats.retained_bytes,
                        "executor_r3_cleanup_s": routing_cleanup_stats.elapsed_s,
                        "executor_r3_cleanup_oldest_key_age_s": routing_cleanup_stats.oldest_key_age_s,
                    }
                )

            # Add loss-specific metrics (IS/KL divergence, OPD KL stats, ratio stats, etc.)
            for key in result:
                if key.startswith(("is_", "opd_")):
                    output_dict[key] = result[key]
                elif key.startswith(FORWARD_BACKWARD_RESULT_PREFIXES):
                    output_dict[key] = result[key]
                elif key in FORWARD_BACKWARD_RESULT_KEYS:
                    output_dict[key] = result[key]

            # Pass through expert load summary for MoE models
            if "expert_load_summary" in result:
                output_dict["expert_load_summary"] = result["expert_load_summary"]

            # Pass through auto-load info if adapter was loaded from checkpoint
            if result.get("auto_loaded"):
                output_dict["auto_loaded"] = True
                output_dict["auto_load_path"] = result.get("auto_load_path")

            # Forward-only teacher prefill path writes activation caches to
            # shared storage and returns the cache metadata here.
            for key in (
                "teacher_hidden_cache",
                "teacher_prefill_tokens",
                "teacher_prefill_forward_compute_s",
                "teacher_hidden_cache_write_s",
            ):
                if key in result:
                    output_dict[key] = result[key]

            # Unpack per-token outputs if present (tinker API compatibility)
            if "packed_logprobs" in result and "packed_position_ids" in result:
                output_dict["per_sample_outputs"] = self._unpack_per_sample_outputs(result, batches)

            output = OrchestratorOutputs(
                request_id=request.request_id,
                output_type=output_type,
                outputs=[output_dict],
                finished=True,
            )

            self.successful_operations += 1
            t_done = time.perf_counter()
            output_dict["executor_build_output_s"] = t_done - t_backend
            output_dict["executor_total_s"] = t_done - t0
            logger.info(
                f"[TIMING] executor {op_name}: "
                f"pack={t_packed - t0:.4f}s "
                f"backend={t_backend - t_packed:.4f}s "
                f"build_output={t_done - t_backend:.4f}s "
                f"total={t_done - t0:.4f}s | "
                f"loss={loss:.4f}, tokens={tokens}"
            )
            return output

        except Exception as e:
            self.failed_operations += 1
            error_msg = str(e)
            if "sleep mode" in error_msg.lower():
                logger.error(f"{op_name} failed: {error_msg}")
            else:
                logger.error(f"Error executing {op_name}: {e}", exc_info=True)

            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.ERROR,
                finished=True,
                error=error_msg,
            )

    @staticmethod
    def _validate_routing_payload_counts(
        expected_count: int,
        routed_experts: Optional[List[Any]],
        routed_expert_logits: Optional[List[Any]],
        *,
        context: str,
    ) -> None:
        if routed_experts is not None and len(routed_experts) != expected_count:
            raise ValueError(
                f"R3 routed_experts count mismatch for {context}: expected {expected_count}, got {len(routed_experts)}"
            )
        if routed_expert_logits is not None and len(routed_expert_logits) != expected_count:
            raise ValueError(
                f"R3 routed_expert_logits count mismatch for {context}: "
                f"expected {expected_count}, got {len(routed_expert_logits)}"
            )
        if (
            routed_experts is not None
            and routed_expert_logits is not None
            and len(routed_experts) != len(routed_expert_logits)
        ):
            raise ValueError(
                "R3 routed_experts and routed_expert_logits count mismatch: "
                f"{len(routed_experts)} != {len(routed_expert_logits)}"
            )

    @staticmethod
    def _teacher_sort_key(datum: Dict[str, Any]) -> int:
        flattened: Dict[str, Any] = {}
        if isinstance(datum.get("model_input"), dict):
            flattened.update(datum["model_input"])
        if isinstance(datum.get("loss_fn_inputs"), dict):
            flattened.update(datum["loss_fn_inputs"])
        for key, value in datum.items():
            if key not in ("model_input", "loss_fn_inputs"):
                flattened[key] = value

        teacher_id = flattened.get("teacher_id")
        if teacher_id is None:
            teacher_ids = flattened.get("teacher_ids")
            if hasattr(teacher_ids, "reshape"):
                teacher_ids = teacher_ids.reshape(-1).tolist()
            if isinstance(teacher_ids, list) and teacher_ids:
                while isinstance(teacher_ids[0], list) and teacher_ids[0]:
                    teacher_ids = teacher_ids[0]
                teacher_id = teacher_ids[0]
        return int(teacher_id) if teacher_id is not None else 0

    @staticmethod
    def _unpack_per_sample_outputs(result: Dict, batches: list) -> list:
        """Unpack packed per-token outputs into per-sample lists.

        Handles both cross_entropy (logprobs + losses) and importance_sampling (logprobs only).
        """
        packed_logprobs = result["packed_logprobs"]
        packed_losses = result.get("packed_losses")
        packed_position_ids = result["packed_position_ids"]
        packed_token_diagnostics = result.get("packed_token_diagnostics")
        per_sample_outputs = []

        for i, (logprobs, pos_ids) in enumerate(zip(packed_logprobs, packed_position_ids)):
            logprobs_tensor = torch.tensor(logprobs)
            pos_ids_tensor = torch.tensor(pos_ids)

            sample_logprobs = unpack_per_token_outputs(logprobs_tensor, pos_ids_tensor)
            sample_token_diagnostics = (
                RequestProcessor._unpack_token_diagnostics(packed_token_diagnostics[i], pos_ids_tensor)
                if packed_token_diagnostics is not None and i < len(packed_token_diagnostics)
                else None
            )

            # Limit to real samples: padding tokens create a spurious sequence boundary
            num_real = batches[i].get("num_samples") if i < len(batches) else None
            if num_real is not None:
                sample_logprobs = sample_logprobs[:num_real]
                if sample_token_diagnostics is not None:
                    sample_token_diagnostics = sample_token_diagnostics[:num_real]

            if packed_losses is not None:
                losses_tensor = torch.tensor(packed_losses[i])
                sample_losses = unpack_per_token_outputs(losses_tensor, pos_ids_tensor)
                if num_real is not None:
                    sample_losses = sample_losses[:num_real]
                for sample_idx, (lp, el) in enumerate(zip(sample_logprobs, sample_losses)):
                    output = {"logprobs": lp, "elementwise_loss": el}
                    if sample_token_diagnostics is not None and sample_idx < len(sample_token_diagnostics):
                        output["token_diagnostics"] = sample_token_diagnostics[sample_idx]
                    per_sample_outputs.append(output)
            else:
                for sample_idx, lp in enumerate(sample_logprobs):
                    output = {"logprobs": lp}
                    if sample_token_diagnostics is not None and sample_idx < len(sample_token_diagnostics):
                        output["token_diagnostics"] = sample_token_diagnostics[sample_idx]
                    per_sample_outputs.append(output)

        logger.debug(f"Unpacked {len(per_sample_outputs)} per-sample outputs")
        return per_sample_outputs

    @staticmethod
    def _unpack_token_diagnostics(diagnostics: Dict, position_ids: torch.Tensor) -> list[Dict]:
        """Split packed top-k/rank diagnostics by sample boundaries."""
        if not diagnostics:
            return []

        pos = position_ids.reshape(-1).to(dtype=torch.long)
        if pos.numel() == 0:
            return []
        starts = [0]
        for idx in range(1, pos.numel()):
            if pos[idx].item() <= pos[idx - 1].item():
                starts.append(idx)
        starts.append(pos.numel())

        fields = (
            "target_ids",
            "target_logprobs",
            "target_ranks",
            "topk_ids",
            "topk_logprobs",
            "loss_logprobs",
            "loss_logprob_deltas",
            "reference_target_logprobs",
            "reference_target_ranks",
            "reference_logprob_deltas",
            "hidden_state_summaries",
            "hidden_component_summaries",
        )
        valid_positions = [int(item) for item in diagnostics.get("valid_positions", [])]
        # Verify all per-token fields agree on length with valid_positions so
        # boundary slices stay aligned. A mismatch is a producer-side bug.
        n = len(valid_positions)
        for field in fields:
            values = diagnostics.get(field, [])
            if values and len(values) != n:
                raise ValueError(
                    f"token_diagnostics field '{field}' has length {len(values)} but valid_positions has length {n}"
                )

        if n == 0:
            return []

        out = []
        for start, end in zip(starts[:-1], starts[1:], strict=False):
            selected = [idx for idx, value in enumerate(valid_positions) if start <= value < end]
            item = {"valid_positions": [valid_positions[idx] - start for idx in selected]}
            for field in fields:
                values = diagnostics.get(field, [])
                item[field] = [values[idx] for idx in selected if idx < len(values)]
            out.append(item)
        return out

    async def execute_forward_backward(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute forward-backward pass on workers."""
        return await self._execute_model_pass(
            request,
            "forward_backward",
            OutputType.FORWARD_BACKWARD,
        )

    async def execute_forward(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute forward pass on workers (no gradient computation)."""
        return await self._execute_model_pass(
            request,
            "forward",
            OutputType.FORWARD,
        )

    async def _execute_operation(
        self,
        request: OrchestratorRequest,
        op_name: str,
        backend_coro,
        output_type: OutputType,
        build_output: Callable[[Dict], Union[list, dict]],
    ) -> OrchestratorOutputs:
        """Execute an operation with standard logging, counters, and error handling."""
        logger.info(f"Executing {op_name} for request {request.request_id}")
        self.total_operations += 1
        t0 = time.perf_counter()
        try:
            result = await backend_coro
            t_backend = time.perf_counter()
            outputs = build_output(result)
            self.successful_operations += 1
            t_done = time.perf_counter()
            logger.info(
                f"[TIMING] executor {op_name}: "
                f"backend={t_backend - t0:.4f}s "
                f"build_output={t_done - t_backend:.4f}s "
                f"total={t_done - t0:.4f}s"
            )
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=output_type,
                outputs=outputs,
                finished=True,
            )
        except Exception as e:
            self.failed_operations += 1
            error_msg = str(e)
            if "sleep mode" in error_msg.lower():
                logger.error(f"{op_name} failed: {error_msg}")
            else:
                logger.error(f"Error executing {op_name}: {e}", exc_info=True)
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.ERROR,
                finished=True,
                error=error_msg,
            )

    async def execute_optim_step(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute optimizer step on workers."""
        p: OptimStepData = request.payload
        lr = p.lr

        def build_output(result):
            output_dict = {
                "grad_norm": result.get("grad_norm", 0.0),
                "lr": result.get("lr", result.get("learning_rate", lr)),
                "learning_rate": lr,
                "step": result.get("step", 0),
                "execution_time": result.get("execution_time", 0.0),
            }
            for key in ("optim_step_time", "optim_empty_cache_skipped", "glm52_fullparam_publish"):
                if key in result:
                    output_dict[key] = result[key]
            if result.get("auto_loaded"):
                output_dict["auto_loaded"] = True
                output_dict["auto_load_path"] = result.get("auto_load_path")
            return [output_dict]

        return await self._execute_operation(
            request,
            "optim_step",
            self.backend.optim_step(
                lr=p.lr,
                gradient_clip=p.gradient_clip,
                beta1=p.beta1,
                beta2=p.beta2,
                eps=p.eps,
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.OPTIM_STEP,
            build_output,
        )

    async def execute_abort_gradient_epoch(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Discard all unmutated captures after an ambiguous forward response."""

        p = request.payload
        return await self._execute_operation(
            request,
            "abort_gradient_epoch",
            self.backend.abort_gradient_epoch(
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.ABORT_GRADIENT_EPOCH,
            lambda result: [result],
        )

    async def execute_save_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute checkpoint save on workers."""
        p: SaveStateData = request.payload
        checkpoint_path = p.checkpoint_path

        def build_output(result):
            actual_path = result.get("checkpoint_path", checkpoint_path)
            success = result.get("success", False)
            return [
                {
                    "checkpoint_path": actual_path,
                    "success": success,
                    "execution_time": result.get("execution_time", 0.0),
                    "message": "Checkpoint saved successfully" if success else "Save failed",
                }
            ]

        return await self._execute_operation(
            request,
            "save_state",
            self.backend.save_state(
                checkpoint_path=p.checkpoint_path,
                save_optimizer=p.save_optimizer,
                use_timestamp=p.use_timestamp,
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SAVE_STATE,
            build_output,
        )

    async def execute_save_lora_only(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute LoRA-only checkpoint save on workers."""
        p: SaveLoraOnlyData = request.payload
        lora_path = p.lora_path
        if not lora_path:
            raise ValueError("lora_path is required")

        def build_output(result):
            actual_path = result.get("lora_path", lora_path)
            success = result.get("success", False)
            return [
                {
                    "lora_path": actual_path,
                    "success": success,
                    "execution_time": result.get("execution_time", 0.0),
                    "message": "LoRA adapter saved successfully (PEFT format)" if success else "Save failed",
                }
            ]

        return await self._execute_operation(
            request,
            "save_lora_only",
            self.backend.save_lora_only(
                lora_path=p.lora_path,
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SAVE_LORA_ONLY,
            build_output,
        )

    async def execute_save_full_weights(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute full weights save as safetensors on workers."""
        p: SaveFullWeightsData = request.payload
        output_path = p.output_path
        if not output_path:
            raise ValueError("output_path is required")
        dtype = p.dtype

        def build_output(result):
            success = result.get("success", False)
            num_shards = result.get("num_shards", 1)
            return [
                {
                    "output_path": output_path,
                    "dtype": dtype,
                    "num_shards": num_shards,
                    "success": success,
                    "execution_time": result.get("execution_time", 0.0),
                    "message": f"Full weights saved as safetensors ({num_shards} shards)" if success else "Save failed",
                }
            ]

        return await self._execute_operation(
            request,
            "save_full_weights",
            self.backend.save_full_weights(
                output_path=p.output_path,
                dtype=p.dtype,
                base_model_path=p.base_model_path,
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SAVE_STATE,
            build_output,
        )

    async def execute_load_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute checkpoint load on workers."""
        p: LoadStateData = request.payload
        checkpoint_path = p.checkpoint_path
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        def build_output(result):
            success = result.get("success", False)
            return [
                {
                    "checkpoint_path": checkpoint_path,
                    "success": success,
                    "execution_time": result.get("execution_time", 0.0),
                    "message": "Checkpoint loaded successfully" if success else "Load failed",
                }
            ]

        return await self._execute_operation(
            request,
            "load_state",
            self.backend.load_state(
                checkpoint_path=p.checkpoint_path,
                load_optimizer=p.load_optimizer,
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.LOAD_STATE,
            build_output,
        )

    async def execute_sleep(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute sleep operation (offload model and optimizer to CPU)."""

        def build_output(result):
            return [
                {
                    "status": result.get("status", "sleeping"),
                    "offload_time": result.get("offload_time", 0.0),
                    "execution_time": result.get("execution_time", 0.0),
                }
            ]

        return await self._execute_operation(
            request,
            "sleep",
            self.backend.sleep(request_id=request.request_id),
            OutputType.SLEEP,
            build_output,
        )

    async def execute_wake_up(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute wake_up operation (load model and optimizer to GPU)."""

        def build_output(result):
            return [
                {
                    "status": result.get("status", "awake"),
                    "load_time": result.get("load_time", 0.0),
                    "execution_time": result.get("execution_time", 0.0),
                }
            ]

        return await self._execute_operation(
            request,
            "wake_up",
            self.backend.wake_up(request_id=request.request_id),
            OutputType.WAKE_UP,
            build_output,
        )

    async def execute_sync_inference_weights(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute sync inference weights operation (NCCL transfer to inference endpoints)."""
        p: SyncWeightsData = request.payload
        if not p.endpoints:
            raise ValueError("inference endpoints must be provided")

        group_name = p.group_name
        if p.sync_method == "nccl_broadcast":
            # NCCL weight sync uses abort-based teardown to avoid cooperative
            # shutdown hangs. PyTorch can keep the group name reserved after
            # abort(), so scope NCCL group names to the request.
            request_token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(request.request_id))
            request_token = request_token.strip("_")[:32] or f"{time.time_ns():x}"
            group_name = f"{p.group_name}_{request_token}"

        def build_output(result):
            return [
                {
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                    "transfer_time": result.get("transfer_time", 0.0),
                    "total_bytes": result.get("total_bytes", 0),
                    "num_parameters": result.get("num_parameters", 0),
                    "num_buckets": result.get("num_buckets", 0),
                    "timing_breakdown": result.get("timing_breakdown", {}),
                    "p2p_rank_summaries": result.get("p2p_rank_summaries", []),
                    "endpoint_results": result.get("endpoint_results", []),
                    "execution_time": result.get("execution_time", 0.0),
                }
            ]

        return await self._execute_operation(
            request,
            "sync_inference_weights",
            self.backend.sync_inference_weights(
                endpoints=p.endpoints,
                master_address=p.master_address,
                master_port=p.master_port,
                group_name=group_name,
                buffer_size_mb=p.buffer_size_mb,
                sync_method=p.sync_method,
                flush_cache=p.flush_cache,
                cache_invalidation_mode=p.cache_invalidation_mode,
                pause_mode=p.pause_mode,
                weight_version=p.weight_version,
                quantization=p.quantization,
                model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SYNC_INFERENCE_WEIGHTS,
            build_output,
        )

    async def execute_register_adapter(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute register adapter operation on workers."""
        p: RegisterAdapterData = request.payload

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request,
            "register_adapter",
            self.backend.register_adapter(
                model_id=p.model_id,
                lr=p.lr,
                request_id=request.request_id,
            ),
            OutputType.REGISTER_ADAPTER,
            build_output,
        )

    async def execute_register_session(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Register a normalized session runtime spec on workers."""
        p: RegisterSessionData = request.payload

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request,
            "register_session",
            self.backend.register_session(
                model_id=p.model_id,
                session_spec=p.session_spec,
                materialize=p.materialize,
                request_id=request.request_id,
            ),
            OutputType.REGISTER_SESSION,
            build_output,
        )

    async def execute_save_adapter_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute save adapter state on workers."""
        p: AdapterStateData = request.payload

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request,
            "save_adapter_state",
            self.backend.save_adapter_state(
                model_id=p.model_id,
                path=p.path,
                save_optimizer=p.save_optimizer,
                request_id=request.request_id,
            ),
            OutputType.SAVE_ADAPTER_STATE,
            build_output,
        )

    async def execute_load_adapter_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute load adapter state on workers."""
        p: AdapterStateData = request.payload
        if not p.path:
            raise ValueError("adapter path is required for load_adapter_state")

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request,
            "load_adapter_state",
            self.backend.load_adapter_state(
                model_id=p.model_id,
                path=p.path,
                load_optimizer=p.load_optimizer,
                lr=p.lr,
                request_id=request.request_id,
            ),
            OutputType.LOAD_ADAPTER_STATE,
            build_output,
        )

    async def execute_get_adapter_info(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute get adapter info on workers."""

        def build_output(result):
            return [result]

        return await self._execute_operation(
            request,
            "get_adapter_info",
            self.backend.get_adapter_info(request_id=request.request_id),
            OutputType.GET_ADAPTER_INFO,
            build_output,
        )

    async def execute_kill_session(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute kill session on workers (full-weights training only)."""
        p: KillSessionData = request.payload

        def build_output(result):
            return [
                {
                    "success": result.get("success", False),
                    "message": result.get("message", ""),
                    "checkpoint_path": result.get("checkpoint_path"),
                    "execution_time": result.get("execution_time", 0.0),
                }
            ]

        return await self._execute_operation(
            request,
            "kill_session",
            self.backend.kill_session(
                model_id=p.model_id,
                save_checkpoint=p.save_checkpoint,
                request_id=request.request_id,
            ),
            OutputType.KILL_SESSION,
            build_output,
        )

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "running": self.backend.is_ready(),
            "connected": self.backend.is_ready(),
            "ready": self.backend.is_ready(),
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": (self.successful_operations / self.total_operations if self.total_operations > 0 else 0.0),
        }

    def __repr__(self) -> str:
        return f"RequestProcessor(ready={self.backend.is_ready()}, operations={self.total_operations})"
