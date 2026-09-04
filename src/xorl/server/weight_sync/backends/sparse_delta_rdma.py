"""RDMA (Mooncake) distribution for packed sparse-delta payloads.

Replaces the shared-filesystem hop of the sparse-delta transport: instead of
rank 0 writing the packed delta to shared storage and every receiver reading
the whole file back, rank 0 keeps the packed bytes in a Mooncake-registered
host-pinned buffer and RDMA-writes them into a registered staging buffer on
every receiver TP rank via the parallel per-session fan-out (the same
machinery as the P2P full-sync backend — reused, not re-implemented).

Protocol (all HTTP against ``/update_weights_from_sparse_delta``):

1. prepare:  POST ``{"staging_op": "prepare", "staging_nbytes": N}`` to every
   endpoint (unpaused — staging registration touches no weights). Response
   carries per-TP-rank ``{tp_rank, session_id, ptr, nbytes}``.
2. write:    ``batch_transfer_sync_write`` the payload in bounded chunks to
   every (session_id, ptr), fanned out across receiver sessions with
   per-endpoint error isolation (one dead receiver is quarantined; the
   others proceed; the weight-version chain converges the victim later
   via base-version mismatch -> force_prime).
3. apply:    POST ``{"staging_op": "apply", "staging_nbytes": N, ...}`` (with
   sha256, versions, flush_cache) to the healthy endpoints; the receiver
   decodes from its staging buffer.

A receiver that predates the staging protocol rejects the prepare POST with a
non-transport error (e.g. schema 4xx); the caller treats that as
``SparseDeltaRdmaUnsupported`` and falls back to the file transport for the
whole sync, so mixed receiver versions stay consistent.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

from .p2p import (
    _EndpointFailureTracker,
    _endpoint_error_isolation_enabled,
    _resolve_local_hostname,
    _run_sync_transfer_items,
    make_local_mooncake_engine,
)


logger = logging.getLogger(__name__)


class SparseDeltaRdmaUnavailable(RuntimeError):
    """RDMA distribution cannot run at all (no Mooncake engine, or every
    endpoint lacks staging support). The caller should fall back to the file
    transport."""


class SparseDeltaRdmaUnsupported(SparseDeltaRdmaUnavailable):
    """A receiver answered the prepare POST but does not speak the staging
    protocol (old sglang build)."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("[SparseDeltaRDMA] invalid %s; using %d", name, default)
        return default


def _chunk_bytes() -> int:
    return max(1 << 20, _env_int("XORL_SPARSE_DELTA_RDMA_CHUNK_BYTES", 128 * 1024 * 1024))


def _batch_call_budget_bytes() -> int:
    """Max bytes per batch_transfer_sync call.

    Bounded calls avoid making one transfer depend on every local RNIC being
    able to reach the peer.
    """
    return max(_chunk_bytes(), _env_int("XORL_SPARSE_DELTA_RDMA_BATCH_BYTES", 512 * 1024 * 1024))


def _register_region_bytes() -> int:
    """Max bytes per Mooncake memory registration (source + staging are
    registered in sub-ranges of this size; same striping caveat as above)."""
    return max(_chunk_bytes(), _env_int("XORL_SPARSE_DELTA_RDMA_REGION_BYTES", 1024 * 1024 * 1024))


def _rdma_workers(num_sessions: int) -> int:
    default = min(16, max(1, num_sessions))
    return max(1, min(_env_int("XORL_SPARSE_DELTA_RDMA_WORKERS", default), max(1, num_sessions)))


class _EngineCache:
    """Process-level Mooncake engine for the sparse-delta sender.

    The engine handshake + its RPC port are expensive and stateful (receivers
    cache the sender's session), so one engine is kept for the process
    lifetime and shared across syncs. Source-buffer registrations are
    per-payload (registered before the fan-out, deregistered after) — the
    payload tensor changes every fold.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: Optional[Any] = None
        self._key: Optional[Tuple[Any, ...]] = None

    def get(self, *, hostname: Optional[str], gpu_id: int, ib_device: Optional[str]) -> Any:
        key = (hostname, int(gpu_id), ib_device)
        with self._lock:
            if self._engine is not None and self._key == key:
                return self._engine
            if self._engine is not None:
                logger.warning(
                    "[SparseDeltaRDMA] engine config changed %s -> %s; constructing a new engine",
                    self._key,
                    key,
                )
            engine = make_local_mooncake_engine(
                hostname=hostname or _resolve_local_hostname(),
                gpu_id=gpu_id,
                ib_device=ib_device,
            )
            if engine is None:
                raise SparseDeltaRdmaUnavailable(
                    "Mooncake TransferEngine unavailable on the sparse-delta sender"
                )
            self._engine = engine
            self._key = key
            return engine


_ENGINE_CACHE = _EngineCache()


def get_sender_engine(backend_config: Optional[Dict[str, Any]] = None) -> Any:
    be_cfg = backend_config or {}
    return _ENGINE_CACHE.get(
        hostname=be_cfg.get("hostname"),
        gpu_id=int(be_cfg.get("gpu_id", 0)),
        ib_device=be_cfg.get("ib_device"),
    )


class RdmaDeltaDistributor:
    """One sync's worth of RDMA sparse-delta distribution state."""

    def __init__(
        self,
        *,
        engine: Any,
        endpoints: List[Any],  # EndpointConfig-like: .host/.port/.world_size
        http_timeout_s: float,
        post_workers: int,
    ) -> None:
        self._engine = engine
        self._endpoints = list(endpoints)
        self._http_timeout_s = float(http_timeout_s)
        self._post_workers = max(1, int(post_workers))
        # endpoint_idx -> list of per-rank staging infos
        self._staging: Dict[int, List[Dict[str, Any]]] = {}
        self._payload_ptr: Optional[int] = None
        self._payload_nbytes: int = 0
        self._payload_registered = False
        self._registered_ranges: List[int] = []
        self.failure_tracker: Optional[_EndpointFailureTracker] = None
        self.stats: Dict[str, float] = {
            "staging_prepare_s": 0.0,
            "rdma_register_s": 0.0,
            "rdma_write_s": 0.0,
            "rdma_bytes": 0.0,
            "rdma_sessions": 0.0,
            "rdma_quarantined_endpoints": 0.0,
        }

    # ------------------------------------------------------------------
    # Phase 1: prepare staging buffers on every endpoint
    # ------------------------------------------------------------------
    def prepare_staging(self, nbytes: int) -> None:
        t0 = time.perf_counter()
        unsupported: List[str] = []
        failures: Dict[int, str] = {}

        def _prepare_one(item: Tuple[int, Any]) -> None:
            idx, ep = item
            url = f"http://{ep.host}:{ep.port}/update_weights_from_sparse_delta"
            payload = {"staging_op": "prepare", "staging_nbytes": int(nbytes)}
            try:
                resp = requests.post(url, json=payload, timeout=self._http_timeout_s)
            except requests.RequestException as exc:
                # Transport-level failure: the endpoint is down/unreachable.
                # Quarantine it (per-endpoint isolation) instead of failing
                # the entire receiver set.
                failures[idx] = f"staging prepare unreachable: {exc}"
                return
            body: Dict[str, Any] = {}
            try:
                body = resp.json()
            except ValueError:
                pass
            if resp.status_code == 200 and body.get("success") and body.get("staging"):
                self._staging[idx] = list(body["staging"])
                return
            # The endpoint answered but refused: an old build (schema 4xx /
            # "unsupported") means the receiver set cannot do staging -> file
            # fallback for the whole sync keeps mixed versions consistent.
            detail = str(body.get("message") or resp.text[:300])
            unsupported.append(f"{ep.host}:{ep.port}: HTTP {resp.status_code}: {detail}")

        with ThreadPoolExecutor(
            max_workers=min(self._post_workers, len(self._endpoints)),
            thread_name_prefix="sparse-delta-rdma-prep",
        ) as pool:
            list(pool.map(_prepare_one, enumerate(self._endpoints)))

        self.stats["staging_prepare_s"] += time.perf_counter() - t0

        if unsupported:
            raise SparseDeltaRdmaUnsupported(
                f"{len(unsupported)} endpoint(s) do not support sparse-delta staging; "
                f"first: {unsupported[0]}"
            )

        session_to_endpoint: Dict[str, int] = {}
        for idx, infos in self._staging.items():
            for info in infos:
                session_to_endpoint[str(info["session_id"])] = idx
        self.failure_tracker = _EndpointFailureTracker(
            session_to_endpoint=session_to_endpoint,
            num_endpoints=len(self._endpoints),
            isolation_enabled=_endpoint_error_isolation_enabled(),
        )
        for idx, message in failures.items():
            ep = self._endpoints[idx]
            error = RuntimeError(f"[SparseDeltaRDMA] endpoint {ep.host}:{ep.port} {message}")
            if not self.failure_tracker.isolation_enabled:
                raise error
            self.failure_tracker.merge_failed_endpoints({idx: str(error)})
            logger.error(
                "[SparseDeltaRDMA] quarantining endpoint %s:%s at staging prepare: %s",
                ep.host,
                ep.port,
                message,
            )
        self.failure_tracker.raise_if_all_endpoints_failed()
        if not self._staging:
            raise SparseDeltaRdmaUnavailable("no endpoint returned staging buffers")

    # ------------------------------------------------------------------
    # Phase 2: register the source payload + RDMA-write it everywhere
    # ------------------------------------------------------------------
    def distribute(self, payload: torch.Tensor) -> None:
        if payload.dtype != torch.uint8 or payload.device.type != "cpu":
            raise ValueError("RDMA sparse-delta payload must be a CPU uint8 tensor")
        if not payload.is_contiguous():
            payload = payload.contiguous()
        nbytes = int(payload.numel())

        t_reg = time.perf_counter()
        src_ptr = int(payload.data_ptr())
        # The singular register API routes through Mooncake's CPU(pinned)
        # registration path — same call the P2P backend uses for its pinned
        # scratch pools — but in bounded sub-ranges so registration does not
        # depend on every local RNIC being able to reach the peer.
        # XORL_SPARSE_DELTA_RDMA_SOURCE_LOCATION optionally pins the
        # NIC-topology hint (e.g. "cuda:0").
        src_location = os.environ.get("XORL_SPARSE_DELTA_RDMA_SOURCE_LOCATION", "").strip()
        region = _register_region_bytes()
        for off in range(0, nbytes, region):
            length = min(region, nbytes - off)
            try:
                if src_location:
                    ret = self._engine.engine.register_memory(src_ptr + off, length, src_location)
                else:
                    ret = self._engine.engine.register_memory(src_ptr + off, length)
            except TypeError:
                ret = self._engine.engine.register_memory(src_ptr + off, length)
            if ret != 0:
                for prev in range(0, off, region):
                    try:
                        self._engine.engine.unregister_memory(src_ptr + prev)
                    except Exception:  # noqa: BLE001
                        pass
                raise SparseDeltaRdmaUnavailable(
                    f"Mooncake source registration failed: ret={ret} "
                    f"(sub-range {off}..{off + length} of {nbytes / 1e9:.2f} GB, "
                    f"location={src_location or '<auto>'})"
                )
            self._registered_ranges.append(src_ptr + off)
        self._payload_ptr = src_ptr
        self._payload_nbytes = nbytes
        self._payload_registered = True
        self.stats["rdma_register_s"] += time.perf_counter() - t_reg

        chunk = _chunk_bytes()
        # Chunks must not straddle registration sub-range boundaries: clamp
        # the chunk so the region size is a whole multiple of it.
        region = _register_region_bytes()
        while region % chunk != 0:
            chunk >>= 1
        offsets = list(range(0, nbytes, chunk))

        # by_session: session_id -> (src_ptrs, peer_ptrs, lengths, debug)
        by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[Any]]]] = {}
        session_debug_info: Dict[str, Dict[str, Any]] = {}
        for idx, infos in sorted(self._staging.items()):
            ep = self._endpoints[idx]
            for info in infos:
                session_id = str(info["session_id"])
                peer_base = int(info["ptr"])
                peer_capacity = int(info.get("nbytes", 0))
                if peer_capacity and peer_capacity < nbytes:
                    raise RuntimeError(
                        f"[SparseDeltaRDMA] endpoint {ep.host}:{ep.port} rank "
                        f"{info.get('tp_rank')} staging capacity {peer_capacity} "
                        f"< payload {nbytes}"
                    )
                src_ptrs = [src_ptr + off for off in offsets]
                peer_ptrs = [peer_base + off for off in offsets]
                lengths = [min(chunk, nbytes - off) for off in offsets]
                by_session[session_id] = (src_ptrs, peer_ptrs, lengths, None)
                session_debug_info[session_id] = {
                    "endpoint": f"{ep.host}:{ep.port}",
                    "tp_rank": info.get("tp_rank"),
                    "payload_nbytes": nbytes,
                }

        # Bound each batch_transfer_sync call: entry-count cap AND byte
        # budget (multi-GB single calls fail on fabric-less RNIC striping).
        max_batch = max(1, _env_int("XORL_SPARSE_DELTA_RDMA_MAX_BATCH_ENTRIES", 64))
        max_batch = min(max_batch, max(1, _batch_call_budget_bytes() // chunk))
        items: List[Tuple[str, int, int]] = []
        for session_id, (src_ptrs, _peer, _len, _dbg) in by_session.items():
            for i in range(0, len(src_ptrs), max_batch):
                items.append((session_id, i, min(i + max_batch, len(src_ptrs))))

        num_sessions = len(by_session)
        self.stats["rdma_sessions"] = float(num_sessions)
        session_executor: Optional[ThreadPoolExecutor] = None
        n_workers = _rdma_workers(num_sessions)
        if n_workers > 1:
            session_executor = ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="sparse-delta-rdma"
            )
        session_transfer_s: Dict[str, float] = {}
        t_write = time.perf_counter()
        try:
            _run_sync_transfer_items(
                engine_wrapper=self._engine,
                by_session=by_session,
                items=items,
                session_debug_info=session_debug_info,
                session_transfer_s=session_transfer_s,
                bucket_idx=0,
                label="sparse-delta staging write",
                session_executor=session_executor,
                failure_tracker=self.failure_tracker,
            )
        finally:
            if session_executor is not None:
                session_executor.shutdown(wait=True)
            self.stats["rdma_write_s"] += time.perf_counter() - t_write

        assert self.failure_tracker is not None
        self.failure_tracker.raise_if_all_endpoints_failed()
        failed = self.failure_tracker.failed_endpoint_errors()
        self.stats["rdma_quarantined_endpoints"] = float(len(failed))
        healthy_sessions = num_sessions - sum(
            1
            for session_id in by_session
            if self.failure_tracker.is_session_failed(session_id)
        )
        self.stats["rdma_bytes"] += float(nbytes) * healthy_sessions
        if failed:
            logger.warning(
                "[SparseDeltaRDMA] distributed %.1f MB to %d/%d sessions in %.2fs; "
                "quarantined endpoints: %s",
                nbytes / 1e6,
                healthy_sessions,
                num_sessions,
                self.stats["rdma_write_s"],
                {idx: msg[:160] for idx, msg in failed.items()},
            )
        else:
            logger.info(
                "[SparseDeltaRDMA] distributed %.1f MB to %d session(s) in %.2fs "
                "(%.1f GB/s aggregate)",
                nbytes / 1e6,
                num_sessions,
                self.stats["rdma_write_s"],
                (nbytes * num_sessions / 1e9) / max(self.stats["rdma_write_s"], 1e-9),
            )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def healthy_endpoint_indices(self) -> List[int]:
        failed = set(
            self.failure_tracker.failed_endpoint_errors() if self.failure_tracker else {}
        )
        return [
            idx
            for idx in range(len(self._endpoints))
            if idx not in failed and idx in self._staging
        ]

    def failed_endpoint_errors(self) -> Dict[int, str]:
        if self.failure_tracker is None:
            return {}
        return self.failure_tracker.failed_endpoint_errors()

    @property
    def payload_nbytes(self) -> int:
        return self._payload_nbytes

    def close(self) -> None:
        if self._payload_registered:
            for ptr in self._registered_ranges:
                try:
                    self._engine.engine.unregister_memory(ptr)
                except Exception:  # noqa: BLE001 - engine teardown races are non-fatal
                    logger.warning(
                        "[SparseDeltaRDMA] failed to deregister a payload sub-range",
                        exc_info=True,
                    )
        self._registered_ranges = []
        self._payload_registered = False
        self._payload_ptr = None
