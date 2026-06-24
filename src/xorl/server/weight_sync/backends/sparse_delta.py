"""Sparse-delta HTTP transport backend.

This backend is an experimental bridge between xorl's existing streaming
weight-sync pipeline and SGLang's ``/update_weights_from_sparse_delta``
receiver endpoint. It encodes each prepared bucket as a packed sparse file on a
shared filesystem, then asks each inference endpoint to mmap/decode/scatter it.

The packed format is provided by the optional ``delta-encoding`` package. The
dependency is loaded lazily so normal dense weight-sync deployments do not need
it installed.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Tuple

import requests
import torch

from ..sparse_delta_files import prepare_delta_encoding_runtime
from .base import TransportConfig, WeightTransportBackend


logger = logging.getLogger(__name__)

_HTTP_TIMEOUT_SECONDS = 600.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("[SparseDelta] invalid %s=%r; using %.1fs", name, raw, default)
        return default
    if value <= 0:
        logger.warning("[SparseDelta] invalid %s=%r; using %.1fs", name, raw, default)
        return default
    return value


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "weight_sync"


def _endpoint_key(config: TransportConfig, baseline_scope: str) -> Tuple[Any, ...]:
    return (
        baseline_scope,
        config.group_name,
        tuple((ep.host, int(ep.port), int(ep.world_size)) for ep in config.endpoints),
    )


def _sha256_file(path: Path) -> str:
    """sha256 of a packed delta file (matches the sglang receiver's integrity check)."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _endpoint_delta_result(endpoint: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "host": endpoint.host,
        "port": endpoint.port,
        "success": body.get("success", False),
        "message": body.get("message", ""),
    }
    cache_epoch = body["cache_epoch"] if "cache_epoch" in body else body.get("cache_version")
    if cache_epoch is not None:
        result["cache_epoch"] = cache_epoch
    for key in ("fp8_kv_cache_postprocess_ran", "fp8_kv_cache_static_scales_updated"):
        if key in body:
            result[key] = body[key]
    return result


class SparseDeltaTransportBackend(WeightTransportBackend):
    """POST packed sparse-delta files to SGLang sparse-delta receivers."""

    _baseline_cache: ClassVar[Dict[Tuple[Any, ...], Dict[str, torch.Tensor]]] = {}

    def __init__(self, config: TransportConfig, **_: Any) -> None:
        super().__init__(config)
        be_cfg = config.backend_config or {}
        self._output_dir = Path(
            str(be_cfg.get("output_dir") or os.environ.get("XORL_SPARSE_DELTA_OUTPUT_DIR", "/tmp/xorl-sparse-delta"))
        )
        self._keep_files = bool(be_cfg.get("keep_files", _env_bool("XORL_SPARSE_DELTA_KEEP_FILES", False)))
        self._timeout_s = float(
            be_cfg.get("timeout_s", _env_float("XORL_SPARSE_DELTA_HTTP_TIMEOUT_S", _HTTP_TIMEOUT_SECONDS))
        )
        self._run_post_process_weights = bool(
            be_cfg.get(
                "run_post_process_weights",
                _env_bool("XORL_SPARSE_DELTA_RUN_POST_PROCESS_WEIGHTS", False),
            )
        )
        self._post_only = bool(be_cfg.get("post_only", False))
        self._prepacked_only = bool(be_cfg.get("prepacked_only", False))
        self._prime_baseline = bool(
            be_cfg.get(
                "prime_baseline",
                _env_bool("XORL_SPARSE_DELTA_PRIME_BASELINE", False),
            )
        )
        self._base_weight_version = be_cfg.get("base_weight_version") or os.environ.get(
            "XORL_SPARSE_DELTA_BASE_WEIGHT_VERSION"
        )
        self._baseline_scope = str(
            be_cfg.get("baseline_scope") or os.environ.get("XORL_SPARSE_DELTA_BASELINE_SCOPE", "default")
        )
        self._baseline_key = _endpoint_key(config, self._baseline_scope)
        self._initialized = False
        self._sequence = 0
        self._encode_fn: Optional[Callable[[torch.Tensor, torch.Tensor, tuple[int, ...]], Any]] = None
        self._write_packed_file: Optional[Callable[[dict[str, Any], str | Path], Path]] = None
        self._written_files: List[Path] = []
        self._stats: Dict[str, float] = {
            "total_dense_bytes": 0.0,
            "total_dense_elements": 0.0,
            "total_packed_bytes": 0.0,
            "total_changed_values": 0.0,
            "primed_tensors": 0.0,
            "posted_files": 0.0,
            "skipped_unchanged_buckets": 0.0,
            "encode_s": 0.0,
            "baseline_update_s": 0.0,
            "write_s": 0.0,
            "post_s": 0.0,
        }

    @classmethod
    def clear_cached_baselines(cls) -> None:
        """Clear process-local sparse-delta baselines.

        This is mainly for tests and operational reset hooks. Normal syncs keep
        the baseline so later buckets/requests can be encoded sparsely.
        """
        cls._baseline_cache.clear()

    @property
    def _baseline(self) -> Dict[str, torch.Tensor]:
        return self._baseline_cache.setdefault(self._baseline_key, {})

    def initialize(self) -> bool:
        if not self.config.endpoints:
            logger.error("[SparseDelta] No endpoints provided in TransportConfig")
            return False
        if self.config.training_rank != 0:
            logger.error("[SparseDelta] only training rank 0 can initialize the sparse-delta backend")
            return False
        if self._prepacked_only and not self._post_only:
            logger.error(
                "[SparseDelta] prepacked_only=True requires sparse_delta_paths; refusing dense streaming fallback"
            )
            return False

        if not self._post_only:
            try:
                self._load_delta_encoding()
            except Exception as exc:
                logger.error("[SparseDelta] failed to import optional delta-encoding package: %s", exc)
                return False

        if _env_bool("XORL_SPARSE_DELTA_RESET_BASELINE", False):
            self._baseline_cache.pop(self._baseline_key, None)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        logger.info(
            "[SparseDelta] initialized for %d endpoint(s), output_dir=%s, baseline_scope=%s",
            len(self.config.endpoints),
            self._output_dir,
            self._baseline_scope,
        )
        return True

    def _load_delta_encoding(self) -> None:
        be_cfg = self.config.backend_config or {}
        delta_encoding_path = be_cfg.get("delta_encoding_path") or os.environ.get("XORL_DELTA_ENCODING_PATH")
        use_native_extension = bool(
            be_cfg.get(
                "use_native_extension",
                _env_bool("XORL_DELTA_ENCODING_USE_NATIVE_EXTENSION", False),
            )
        )
        prepare_delta_encoding_runtime(
            delta_encoding_path=str(delta_encoding_path) if delta_encoding_path else None,
            use_native_extension=use_native_extension,
        )

        compression = importlib.import_module("delta_encoding.encoding.compression")
        packed = importlib.import_module("delta_encoding.encoding.packed")
        self._encode_fn = compression.encode
        self._write_packed_file = packed.write_packed_file

    def destroy(self, *, complete_receiver: bool = True) -> None:
        del complete_receiver
        self._initialized = False
        if self._keep_files:
            return
        for path in self._written_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("[SparseDelta] failed to remove temporary packed file %s", path, exc_info=True)
        self._written_files.clear()

    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> None:
        if src_rank != 0:
            raise ValueError(f"SparseDeltaTransportBackend only supports src_rank=0, got {src_rank}")
        if self._prepacked_only:
            raise RuntimeError("SparseDelta backend is configured prepacked_only=True; use sparse_delta_paths instead")
        if not self._initialized or self._encode_fn is None or self._write_packed_file is None:
            raise RuntimeError("SparseDelta backend not initialized — call initialize() first")
        if not bucket:
            return

        t_encode = time.perf_counter()
        encoded_tensors: dict[str, Any] = {}
        current_tensors: dict[str, torch.Tensor] = {}
        dense_bytes = 0
        dense_elements = 0
        changed_values = 0

        baseline = self._baseline
        for name, tensor in bucket:
            current = tensor.detach().to("cpu").contiguous()
            current_tensors[name] = current
            dense_bytes += current.numel() * current.element_size()
            dense_elements += current.numel()

            if self._prime_baseline:
                continue

            encoded, nnz = self._encode_changed_values(name, current, baseline.get(name))
            if encoded is not None:
                encoded_tensors[name] = encoded
                changed_values += nnz

        self._stats["encode_s"] += time.perf_counter() - t_encode
        self._stats["total_dense_bytes"] += dense_bytes
        self._stats["total_dense_elements"] += dense_elements
        self._stats["total_changed_values"] += changed_values

        is_final_bucket = flush_cache or weight_version is not None
        if self._prime_baseline:
            t_baseline = time.perf_counter()
            for name, current in current_tensors.items():
                baseline[name] = current
            self._stats["baseline_update_s"] += time.perf_counter() - t_baseline
            self._stats["primed_tensors"] += len(current_tensors)
            if not is_final_bucket:
                self._stats["skipped_unchanged_buckets"] += 1
                return

        if not encoded_tensors:
            if not is_final_bucket:
                self._stats["skipped_unchanged_buckets"] += 1
                return
            for name, current in current_tensors.items():
                encoded_tensors[name] = self._encode_empty(current)

        path = self._next_delta_path()
        t_write = time.perf_counter()
        written = Path(self._write_packed_file(encoded_tensors, path))
        self._stats["write_s"] += time.perf_counter() - t_write
        self._written_files.append(written)
        packed_bytes = written.stat().st_size
        self._stats["total_packed_bytes"] += packed_bytes

        t_post = time.perf_counter()
        try:
            self._post_delta_file(written, flush_cache=flush_cache, weight_version=weight_version)
        except Exception:
            logger.error("[SparseDelta] failed to post packed delta file %s", written, exc_info=True)
            raise
        finally:
            self._stats["post_s"] += time.perf_counter() - t_post

        self._stats["posted_files"] += 1
        t_baseline = time.perf_counter()
        for name, current in current_tensors.items():
            baseline[name] = current
        self._stats["baseline_update_s"] += time.perf_counter() - t_baseline

        if not self._keep_files:
            try:
                written.unlink(missing_ok=True)
                self._written_files.remove(written)
            except (OSError, ValueError):
                logger.debug("[SparseDelta] failed to remove temporary packed file %s", written, exc_info=True)

    def post_packed_delta_paths(
        self,
        delta_paths: List[str],
        *,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> None:
        """POST prepacked sparse-delta files without materializing trainer weights.

        This is the efficient path used when a training loop or external
        ``delta-encoding`` pipeline already emitted packed files. A single path
        is replicated to every TP rank; otherwise the path count must match the
        receiver world size.
        """
        if not self._initialized:
            raise RuntimeError("SparseDelta backend not initialized — call initialize() first")
        if not delta_paths:
            raise ValueError("post_packed_delta_paths requires at least one packed delta path")

        paths = [Path(path) for path in delta_paths]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Sparse-delta packed path(s) do not exist: {missing}")

        unique_paths = {str(path): path for path in paths}
        packed_bytes = sum(path.stat().st_size for path in unique_paths.values())

        t_post = time.perf_counter()
        try:
            self._post_delta_paths(
                [str(path) for path in paths], flush_cache=flush_cache, weight_version=weight_version
            )
        finally:
            self._stats["post_s"] += time.perf_counter() - t_post

        self._stats["posted_files"] += len(unique_paths)
        self._stats["total_packed_bytes"] += packed_bytes

    def _encode_changed_values(
        self,
        name: str,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
    ) -> tuple[Optional[Any], int]:
        del name
        flat = current.reshape(-1)
        self._check_indexable_numel(flat.numel(), current.shape)
        if previous is None or previous.shape != current.shape or previous.dtype != current.dtype:
            flat_indices = torch.arange(flat.numel(), dtype=torch.int32)
        else:
            flat_indices = self._changed_flat_indices(current, previous)

        nnz = int(flat_indices.numel())
        if nnz == 0:
            return None, 0
        values = flat[flat_indices].clone()
        return self._encode_fn(flat_indices, values, tuple(current.shape)), nnz

    def _encode_empty(self, current: torch.Tensor) -> Any:
        flat = current.reshape(-1)
        flat_indices = torch.empty(0, dtype=torch.int32)
        values = flat[:0].clone()
        return self._encode_fn(flat_indices, values, tuple(current.shape))

    @staticmethod
    def _changed_flat_indices(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if current.numel() == 0:
            return torch.empty(0, dtype=torch.int32)
        elem_size = current.element_size()
        current_bytes = current.reshape(-1).view(torch.uint8).reshape(current.numel(), elem_size)
        previous_bytes = previous.reshape(-1).view(torch.uint8).reshape(previous.numel(), elem_size)
        changed = torch.any(current_bytes != previous_bytes, dim=1)
        return changed.nonzero(as_tuple=False).flatten().to(torch.int32)

    @staticmethod
    def _check_indexable_numel(numel: int, shape: torch.Size) -> None:
        if numel > 2**31 - 1:
            raise ValueError(
                f"Sparse-delta packed format requires int32 flat indices, but tensor shape {tuple(shape)} "
                f"has {numel} elements"
            )

    def _next_delta_path(self) -> Path:
        self._sequence += 1
        prefix = _safe_token(self.config.group_name)
        return self._output_dir / (
            f"{prefix}-sparse-delta-r{self.config.training_rank}-p{os.getpid()}-"
            f"{time.time_ns()}-{self._sequence}.packed"
        )

    def _post_delta_file(self, path: Path, *, flush_cache: bool, weight_version: Optional[str]) -> None:
        self._post_delta_paths([str(path)], flush_cache=flush_cache, weight_version=weight_version)

    def _post_delta_paths(self, delta_paths: List[str], *, flush_cache: bool, weight_version: Optional[str]) -> None:
        be_cfg = self.config.backend_config or {}
        endpoint_results: list[dict[str, Any]] = []
        for endpoint in self.config.endpoints:
            world_size = max(1, int(endpoint.world_size))
            endpoint_delta_paths = self._expand_delta_paths(delta_paths, world_size)
            payload: dict[str, Any] = {
                "delta_paths": endpoint_delta_paths,
                "delta_sha256s": [_sha256_file(Path(p)) for p in endpoint_delta_paths],
                "flush_cache": flush_cache,
            }
            if weight_version is not None:
                payload["weight_version"] = weight_version
            if self._base_weight_version is not None:
                payload["base_weight_version"] = self._base_weight_version
            if self._run_post_process_weights:
                payload["run_post_process_weights"] = True
            # Thread FP8 kv-cache post-process knobs to the receiver so it can refresh
            # kv-cache scales after applying the sparse delta (parity with the p2p backend).
            for key in (
                "fp8_kv_cache_enabled",
                "fp8_kv_cache_postprocess_required",
                "fp8_kv_cache_static_scales",
            ):
                if be_cfg.get(key):
                    payload[key] = True

            url = f"http://{endpoint.host}:{endpoint.port}/update_weights_from_sparse_delta"
            response = requests.post(url, json=payload, timeout=self._timeout_s)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Sparse-delta update failed for {endpoint.host}:{endpoint.port}: "
                    f"HTTP {response.status_code}: {response.text[:500]}"
                ) from exc

            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Sparse-delta update returned non-JSON response from {endpoint.host}:{endpoint.port}: "
                    f"{response.text[:500]}"
                ) from exc
            if not body.get("success", False):
                raise RuntimeError(
                    f"Sparse-delta update failed for {endpoint.host}:{endpoint.port}: {body.get('message', body)}"
                )
            endpoint_results.append(_endpoint_delta_result(endpoint, body))
        self.endpoint_results = endpoint_results

    @staticmethod
    def _expand_delta_paths(delta_paths: List[str], world_size: int) -> List[str]:
        if len(delta_paths) == 1:
            return delta_paths * world_size
        if len(delta_paths) == world_size:
            return list(delta_paths)
        raise ValueError(
            f"Expected either one sparse-delta path or {world_size} per-rank paths, got {len(delta_paths)}"
        )

    def stats_summary(self) -> Dict[str, float]:
        summary = dict(self._stats)
        dense_bytes = summary.get("total_dense_bytes", 0.0)
        if dense_bytes > 0:
            summary["packed_to_dense"] = summary.get("total_packed_bytes", 0.0) / dense_bytes
        else:
            summary["packed_to_dense"] = 0.0
        dense_elements = summary.get("total_dense_elements", 0.0)
        if dense_elements > 0:
            summary["changed_density"] = summary.get("total_changed_values", 0.0) / dense_elements
        else:
            summary["changed_density"] = 0.0
        summary["baseline_tensors"] = float(len(self._baseline))
        return summary

    @property
    def sender_ranks(self) -> FrozenSet[int]:
        return frozenset({0})

    @property
    def supports_direct_ep_transfer(self) -> bool:
        return False

    @property
    def supports_direct_pp_transfer(self) -> bool:
        return False
