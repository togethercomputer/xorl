"""P2P (Mooncake) weight transport backend.

RDMA one-sided writes from training ranks directly into the inference
replica's ``param.data`` slices, modeled on the lmsys/Mooncake P2P weight
update mechanism (https://www.lmsys.org/blog/2026-04-29-p2p-update/).

Compared to the NCCL-broadcast backend:

* No NCCL group rendezvous.
* No rank-0 dist.broadcast bottleneck.
* Each inference TP rank registers its own param memory with Mooncake; the
  trainer issues writes against the per-rank session ids returned by
  SGLang's ``/prepare_weights_update``.

The backend supports the rank-0 dense path and the direct-EP MoE path. PP
stage leaders still funnel through rank 0 in the handler.
"""

import dataclasses
import ipaddress
import logging
import os
import socket
import time
import zlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import requests
import torch
import torch.distributed as dist

from .base import TransportConfig, WeightTransportBackend


@dataclasses.dataclass
class _PendingTransfer:
    """One per-locator transfer pending a stage+coalesce pass in
    ``transfer_bucket``. Held just long enough to sort by peer_ptr, copy
    the src_view into the CPU pinned scratch pool, and coalesce
    contiguous neighbors before issuing the Mooncake call.
    """

    peer_ptr: int
    nbytes: int
    src_view: torch.Tensor
    name: str
    loc: Dict[str, Any]


@dataclasses.dataclass
class _TransferDebugEntry:
    name: str
    peer_ptr: int
    nbytes: int
    dtype: Optional[str]
    memory_handle: Optional[int]
    tp_rank: Any
    ep_rank: Any
    loc_slice: Any


@dataclasses.dataclass
class _StagedTransfer:
    src_ptr: int
    peer_ptr: int
    nbytes: int
    memory_handle: Optional[int]
    name: str
    loc: Dict[str, Any]


@dataclasses.dataclass
class _TransferDebugSample:
    entries: List[_TransferDebugEntry] = dataclasses.field(default_factory=list)
    total: int = 0

    def add(self, name: str, loc: Dict[str, Any], nbytes: int) -> None:
        self.total += 1
        if len(self.entries) < _TRANSFER_DEBUG_SAMPLE_LIMIT:
            self.entries.append(_transfer_debug_entry(name, loc, nbytes))

    def extend(self, other: "_TransferDebugSample") -> None:
        self.total += other.total
        if len(self.entries) >= _TRANSFER_DEBUG_SAMPLE_LIMIT:
            return
        remaining = _TRANSFER_DEBUG_SAMPLE_LIMIT - len(self.entries)
        self.entries.extend(other.entries[:remaining])


@dataclasses.dataclass
class _BucketTiming:
    """Per-bucket wall-time breakdown for the P2P transport."""

    nbytes: int = 0
    prepare_s: float = 0.0
    pool_init_s: float = 0.0
    pool_wait_s: float = 0.0
    stage_s: float = 0.0
    submit_s: float = 0.0
    register_s: float = 0.0
    transfer_s: float = 0.0
    deregister_s: float = 0.0
    num_large_buffers: int = 0
    num_small_buffers: int = 0
    session_bytes: Dict[str, int] = dataclasses.field(default_factory=dict)
    session_transfer_s: Dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return (
            self.prepare_s
            + self.pool_init_s
            + self.pool_wait_s
            + self.stage_s
            + self.submit_s
            + self.register_s
            + self.transfer_s
            + self.deregister_s
        )

    @property
    def main_thread_s(self) -> float:
        return self.prepare_s + self.pool_init_s + self.pool_wait_s + self.stage_s + self.submit_s

    @property
    def throughput_mb_s(self) -> float:
        if self.total_s <= 0:
            return 0.0
        return (self.nbytes / 1e6) / self.total_s


logger = logging.getLogger(__name__)


_HTTP_TIMEOUT_SECONDS = 600
_TRANSFER_DEBUG_SAMPLE_LIMIT = 6


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("[P2P] invalid %s=%r; using %.1fs", name, raw, default)
        return default
    if value <= 0:
        logger.warning("[P2P] invalid %s=%r; using %.1fs", name, raw, default)
        return default
    return value


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[P2P] invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("[P2P] invalid %s=%r; using %d", name, raw, default)
        return default
    return value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("[P2P] invalid %s=%r; using %s", name, raw, default)
    return default


def _prepare_timeout_seconds() -> float:
    return _env_float("XORL_P2P_PREPARE_TIMEOUT_S", 120.0)


def _async_api_min_bytes() -> int:
    return _env_int("XORL_P2P_ASYNC_MIN_BYTES", 128 * 1024 * 1024)


def _small_transfer_chunk() -> int:
    return _env_int("XORL_P2P_SMALL_TRANSFER_CHUNK", 32)


def _persist_small_registration_enabled() -> bool:
    # Opt-in only. On the scaled TP2 layout this was safe, but slower than
    # per-bucket small-source registration because it increased warm-sync tails.
    return _env_flag("XORL_P2P_PERSIST_SMALL_REGISTRATION", False)


def _async_api_enabled(*, cached_prepare: bool) -> bool:
    # The synchronous Mooncake API is the measured sustained path. The async
    # API is kept for experiments because repeated-update tests have shown
    # mixed results and hangs/status failures on some runs.
    mode = os.environ.get("XORL_P2P_USE_ASYNC_API", "0").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"warm", "cached", "cached_prepare"}:
        return cached_prepare
    if mode in {"0", "false", "no", "off"}:
        return False
    logger.warning("[P2P] invalid XORL_P2P_USE_ASYNC_API=%r; using sync transfer API", mode)
    return False


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


# CPU-pinned scratch pool size for source staging. Set big
# enough to hold a full bucket's worth of source bytes — the default
# bucket cap is 2 GB, so we size 4 GB by default for safety. The pool
# is registered with Mooncake once at first use and reused for every
# bucket with no per-bucket register cost. Tunable via env var on
# memory-constrained deployments.
_CPU_SCRATCH_POOL_BYTES = int(os.environ.get("XORL_P2P_CPU_SCRATCH_POOL_BYTES", str(4 * 1024 * 1024 * 1024)))

# Mooncake's CPU-source RDMA path on our cluster fails (ret=-1) for
# very small transfers — observed at 8 KB on a layernorm weight. Small
# entries take the GPU-direct path (per-bucket register/dereg); large
# entries take the CPU pool path. 64 KB threshold matches typical
# layernorm weight size (2048 BF16 = 4 KB; 4× headroom). Setting this
# to 0 forces tiny entries through CPU scratch; that was safe in smoke
# tests, but slower than the default GPU-direct threshold.
_CPU_POOL_MIN_BYTES = int(os.environ.get("XORL_P2P_CPU_POOL_MIN_BYTES", str(64 * 1024)))


class _DirectMooncakeTransferEngine:
    """Minimal xorl-side wrapper for mooncake.engine.TransferEngine.

    SGLang ships a convenience wrapper with the same surface, but the trainer
    environment should only need ``mooncake-transfer-engine`` to construct a
    sender. Keep this fallback small and aligned with the methods used below.
    """

    def __init__(
        self,
        transfer_engine_cls: Any,
        *,
        hostname: str,
        gpu_id: int,
        ib_device: Optional[str],
    ) -> None:
        self.engine = transfer_engine_cls()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "rdma",
            ib_device or "",
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake TransferEngine initialization failed: ret={ret}")
        self.session_id = f"{hostname}:{self.engine.get_rpc_port()}"

    def get_session_id(self) -> str:
        return self.session_id

    def get_ib_device(self) -> Optional[str]:
        return self.ib_device

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        try:
            return self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            if not hasattr(self.engine, "batch_register_memory"):
                raise RuntimeError("Mooncake batch register requires a newer mooncake-transfer-engine")
            return -1

    def batch_deregister(self, ptrs: List[int]) -> int:
        try:
            return self.engine.batch_unregister_memory(ptrs)
        except Exception:
            return -1

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        try:
            return self.engine.batch_transfer_sync_write(session_id, buffers, peer_buffer_addresses, lengths)
        except Exception:
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError("Mooncake batch transfer requires a newer mooncake-transfer-engine")
            return -1


class _CompletedCudaEvent:
    """CPU-test stand-in for ``torch.cuda.Event``."""

    def synchronize(self) -> None:
        return None


def _retry_delay(attempt: int) -> float:
    return min(1.0, 0.05 * (2 ** min(attempt, 4)))


def _locator_memory_handle(loc: Dict[str, Any]) -> Optional[int]:
    raw = loc.get("memory_handle")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _transfer_debug_entry(name: str, loc: Dict[str, Any], nbytes: int) -> _TransferDebugEntry:
    return _TransferDebugEntry(
        name=name,
        peer_ptr=int(loc.get("ptr", 0)),
        nbytes=nbytes,
        dtype=loc.get("dtype"),
        memory_handle=_locator_memory_handle(loc),
        tp_rank=loc.get("tp_rank"),
        ep_rank=loc.get("ep_rank"),
        loc_slice=loc.get("slice"),
    )


def _source_view_key(src_view: torch.Tensor, nbytes: int) -> Tuple[Any, ...]:
    ptr = int(src_view.data_ptr())
    nbytes = int(nbytes)
    if src_view.is_contiguous():
        # For contiguous views, the byte range fully identifies the payload we
        # copy into the CPU pool. Avoid tupleizing shape/stride for the common
        # fanout case where many receivers share the same source slice.
        return (ptr, nbytes)
    return (
        ptr,
        tuple(int(dim) for dim in src_view.shape),
        tuple(int(stride) for stride in src_view.stride()),
        str(src_view.dtype),
        nbytes,
    )


def _format_transfer_debug(debug_entries: Any) -> str:
    if debug_entries is None:
        return "transfer_debug=disabled (set XORL_P2P_TRANSFER_DEBUG=1)"

    total_entries: Optional[int] = None
    if isinstance(debug_entries, _TransferDebugSample):
        total_entries = debug_entries.total
        debug_entries = debug_entries.entries
    else:
        total_entries = len(debug_entries)

    if not debug_entries:
        return "transfer_debug=[]"

    parts: List[str] = []
    for entry in debug_entries[:_TRANSFER_DEBUG_SAMPLE_LIMIT]:
        handle = f"0x{entry.memory_handle:x}" if entry.memory_handle is not None else "None"
        parts.append(
            f"{entry.name}(ptr=0x{entry.peer_ptr:x}, nbytes={entry.nbytes}, "
            f"dtype={entry.dtype}, handle={handle}, tp={entry.tp_rank}, "
            f"ep={entry.ep_rank}, slice={entry.loc_slice})"
        )
    if total_entries > len(debug_entries):
        parts.append(f"... {total_entries - len(debug_entries)} more")
    return "transfer_debug=[" + "; ".join(parts) + "]"


def _chunk_sizes(
    by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[_TransferDebugSample]]]],
    session_id: str,
    i: int,
    end: int,
) -> List[int]:
    return by_session[session_id][2][i:end]


def _chunk_debug_sample(
    by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[_TransferDebugSample]]]],
    session_id: str,
    i: int,
    end: int,
) -> Optional[_TransferDebugSample]:
    debug_entries = by_session[session_id][3]
    if debug_entries is None:
        return None
    debug_lists = debug_entries[i:end]
    sample = _TransferDebugSample()
    for debug in debug_lists:
        sample.extend(debug)
    return sample


def _run_sync_transfer_items(
    *,
    engine_wrapper: Any,
    by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[_TransferDebugSample]]]],
    items: List[Tuple[str, int, int]],
    session_debug_info: Dict[str, Dict[str, Any]],
    session_transfer_s: Dict[str, float],
    bucket_idx: int,
    label: str,
) -> None:
    max_attempts = _env_int("XORL_P2P_TRANSFER_RETRIES", 10)
    for session_id, i, end in items:
        src_ptrs, peer_ptrs, lengths, _ = by_session[session_id]
        t_session = time.perf_counter()
        last_ret = 0
        for attempt in range(max_attempts):
            last_ret = engine_wrapper.batch_transfer_sync(session_id, src_ptrs[i:end], peer_ptrs[i:end], lengths[i:end])
            if last_ret >= 0:
                break
            time.sleep(_retry_delay(attempt))
        if last_ret < 0:
            raise RuntimeError(
                f"[P2P] {label} to {session_id} failed: ret={last_ret} "
                f"(bucket {bucket_idx}, chunk {i}..{end} of {len(src_ptrs)} buffers, "
                f"sizes={_chunk_sizes(by_session, session_id, i, end)}, "
                f"{_format_transfer_debug(_chunk_debug_sample(by_session, session_id, i, end))}, "
                f"session_info={session_debug_info.get(session_id)}, after {max_attempts} attempts)"
            )
        session_transfer_s[session_id] = session_transfer_s.get(session_id, 0.0) + (time.perf_counter() - t_session)


def _run_async_transfer_items(
    *,
    engine_wrapper: Any,
    by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[_TransferDebugSample]]]],
    items: List[Tuple[str, int, int]],
    session_debug_info: Dict[str, Dict[str, Any]],
    session_transfer_s: Dict[str, float],
    bucket_idx: int,
) -> None:
    if not items:
        return

    # Bounded submit/poll. Mooncake's underlying TransferEngine exposes:
    #   batch_transfer_async_write(session, src, dst, lens) -> batch_id (int)
    #   get_batch_transfer_status([batch_id, ...]) -> int (0 success, -1 failure/timeout)
    #
    # Status takes a sequence of batch IDs. The single-transfer status API is
    # not valid for these batch IDs and can wedge the caller.
    raw_engine = engine_wrapper.engine
    max_in_flight = _env_int("XORL_P2P_ASYNC_MAX_IN_FLIGHT", 1)
    status_timeout_s = max(0.001, _env_float("XORL_P2P_ASYNC_STATUS_TIMEOUT_S", 30.0))
    work_items = list(items)
    active: List[Tuple[int, str, int, int, float]] = []
    active_since: Optional[float] = None
    last_status_log_at = 0.0

    while work_items or active:
        while work_items and len(active) < max_in_flight:
            session_id, i, end = work_items.pop(0)
            src_ptrs, peer_ptrs, lengths, _ = by_session[session_id]
            bid = raw_engine.batch_transfer_async_write(session_id, src_ptrs[i:end], peer_ptrs[i:end], lengths[i:end])
            if bid is None or (isinstance(bid, int) and bid < 0):
                raise RuntimeError(
                    f"[P2P] batch_transfer_async_write submit failed: bid={bid} "
                    f"(bucket {bucket_idx}, chunk {i}..{end}, "
                    f"sizes={_chunk_sizes(by_session, session_id, i, end)}, "
                    f"{_format_transfer_debug(_chunk_debug_sample(by_session, session_id, i, end))}, "
                    f"session_info={session_debug_info.get(session_id)})"
                )
            if not active:
                active_since = time.perf_counter()
            active.append((int(bid), session_id, i, end, time.perf_counter()))

        if not active:
            active_since = None
            continue

        bids = [bid for bid, *_ in active]
        status = raw_engine.get_batch_transfer_status(bids)
        if status < 0:
            _, session_id, i, end, _ = active[0]
            raise RuntimeError(
                f"[P2P] get_batch_transfer_status reported failure: status={status} "
                f"(bucket {bucket_idx}, {len(active)} batches in flight, "
                f"first session={session_id}, sizes={_chunk_sizes(by_session, session_id, i, end)}, "
                f"{_format_transfer_debug(_chunk_debug_sample(by_session, session_id, i, end))}, "
                f"session_info={session_debug_info.get(session_id)})"
            )
        if status == 0:
            now = time.perf_counter()
            for _, session_id, _, _, submit_t in active:
                session_transfer_s[session_id] = session_transfer_s.get(session_id, 0.0) + (now - submit_t)
            active = []
            active_since = None
            continue

        now = time.perf_counter()
        waited_s = now - (active_since or now)
        if waited_s > status_timeout_s:
            _, session_id, i, end, _ = active[0]
            raise RuntimeError(
                f"[P2P] async transfer status poll timed out: status={status} "
                f"(bucket {bucket_idx}, waited={waited_s:.3f}s, "
                f"{len(active)} batches in flight, first session={session_id}, "
                f"sizes={_chunk_sizes(by_session, session_id, i, end)}, "
                f"{_format_transfer_debug(_chunk_debug_sample(by_session, session_id, i, end))}, "
                f"session_info={session_debug_info.get(session_id)})"
            )
        if now - last_status_log_at > 5.0:
            logger.warning(
                f"[P2P] async transfer still pending: status={status} "
                f"(bucket {bucket_idx}, waited={waited_s:.3f}s, {len(active)} batches in flight)"
            )
            last_status_log_at = now
        time.sleep(0.0001)


def _transfer_small_entries(
    *,
    engine_wrapper: Any,
    small_session_data: Dict[str, List[Tuple[int, int, int, Optional[_TransferDebugEntry]]]],
    session_debug_info: Dict[str, Dict[str, Any]],
    small_register_ptrs: List[int],
    small_register_lens: List[int],
    session_bytes: Dict[str, int],
    session_transfer_s: Dict[str, float],
    bucket_idx: int,
) -> Tuple[int, int]:
    if small_register_ptrs:
        ret = engine_wrapper.batch_register(small_register_ptrs, small_register_lens)
        if ret != 0:
            raise RuntimeError(f"[P2P] small-entries batch_register failed: ret={ret} (bucket {bucket_idx})")

    total_bytes = 0
    num_buffers = 0
    try:
        chunk = _small_transfer_chunk()
        max_attempts = _env_int("XORL_P2P_TRANSFER_RETRIES", 10)
        for session_id, triples in small_session_data.items():
            t_session = time.perf_counter()
            for i in range(0, len(triples), chunk):
                transfer_chunk = triples[i : i + chunk]
                src_ptrs = [src_ptr for src_ptr, _, _, _ in transfer_chunk]
                peer_ptrs = [peer_ptr for _, peer_ptr, _, _ in transfer_chunk]
                lengths = [nbytes for _, _, nbytes, _ in transfer_chunk]
                chunk_bytes = sum(lengths)
                total_bytes += chunk_bytes
                session_bytes[session_id] = session_bytes.get(session_id, 0) + chunk_bytes
                num_buffers += len(transfer_chunk)
                last_ret = 0
                for attempt in range(max_attempts):
                    last_ret = engine_wrapper.batch_transfer_sync(session_id, src_ptrs, peer_ptrs, lengths)
                    if last_ret >= 0:
                        break
                    time.sleep(_retry_delay(attempt))
                if last_ret < 0:
                    debug_entries = [debug for _, _, _, debug in transfer_chunk if debug is not None]
                    raise RuntimeError(
                        f"[P2P] small-entries transfer to {session_id} "
                        f"failed: ret={last_ret} (bucket {bucket_idx}, "
                        f"chunk {i}..{i + len(transfer_chunk)} of {len(triples)} buffers, "
                        f"sizes={lengths}, "
                        f"{_format_transfer_debug(debug_entries or None)}, "
                        f"session_info={session_debug_info.get(session_id)}, "
                        f"after {max_attempts} attempts)"
                    )
            session_transfer_s[session_id] = session_transfer_s.get(session_id, 0.0) + (time.perf_counter() - t_session)
    finally:
        if small_register_ptrs:
            try:
                engine_wrapper.batch_deregister(small_register_ptrs)
            except Exception as e:
                logger.warning(f"[P2P] small-entries dereg failed (bucket {bucket_idx}): {e}")

    return total_bytes, num_buffers


def _do_async_transfer(
    *,
    engine_wrapper: Any,
    copy_done_event: "torch.cuda.Event",
    by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[_TransferDebugSample]]]],
    small_session_data: Dict[str, List[Tuple[int, int, int, Optional[_TransferDebugEntry]]]],
    session_debug_info: Dict[str, Dict[str, Any]],
    small_register_ptrs: List[int],
    small_register_lens: List[int],
    chunk: int,
    use_async_api: bool,
    timing: _BucketTiming,
    bucket_idx: int,
    slice_holds: List[torch.Tensor],
    src_view_holds: List[torch.Tensor],
    log_bucket_details: bool,
) -> None:
    """Worker-thread Mooncake transfer for one bucket.

    The worker waits for CUDA staging to finish, ships large entries from the
    registered CPU pool, and then handles tiny GPU-direct entries that are too
    small for the CPU-source path. ``slice_holds`` and ``src_view_holds`` keep
    source memory alive while the caller reshards the FSDP module.
    """
    copy_done_event.synchronize()

    t0 = time.perf_counter()
    bucket_bytes = 0
    session_bytes: Dict[str, int] = {}
    session_transfer_s: Dict[str, float] = {}
    num_large_buffers = 0

    all_large_items: List[Tuple[str, int, int]] = []
    async_items: List[Tuple[str, int, int]] = []
    sync_fallback_items: List[Tuple[str, int, int]] = []
    async_min_bytes = _async_api_min_bytes()
    for session_id, (src_ptrs, _, lengths, _) in by_session.items():
        nbytes = sum(lengths)
        bucket_bytes += nbytes
        session_bytes[session_id] = session_bytes.get(session_id, 0) + nbytes
        num_large_buffers += len(lengths)
        for i in range(0, len(src_ptrs), chunk):
            end = min(i + chunk, len(src_ptrs))
            item = (session_id, i, end)
            all_large_items.append(item)
            if sum(lengths[i:end]) < async_min_bytes:
                sync_fallback_items.append(item)
            else:
                async_items.append(item)

    # Mooncake's CPU-source sync path can return ret=-1 under high load, so
    # sync transfers keep bounded retries. The async API is stricter: if submit
    # or status fails, we fail closed because a prior async batch may still be
    # writing receiver memory.
    if use_async_api:
        _run_sync_transfer_items(
            engine_wrapper=engine_wrapper,
            by_session=by_session,
            items=sync_fallback_items,
            session_debug_info=session_debug_info,
            session_transfer_s=session_transfer_s,
            bucket_idx=bucket_idx,
            label="async sync-fallback transfer",
        )
        _run_async_transfer_items(
            engine_wrapper=engine_wrapper,
            by_session=by_session,
            items=async_items,
            session_debug_info=session_debug_info,
            session_transfer_s=session_transfer_s,
            bucket_idx=bucket_idx,
        )
    else:
        _run_sync_transfer_items(
            engine_wrapper=engine_wrapper,
            by_session=by_session,
            items=all_large_items,
            session_debug_info=session_debug_info,
            session_transfer_s=session_transfer_s,
            bucket_idx=bucket_idx,
            label="batch_transfer_sync",
        )

    small_bytes, num_small_buffers = _transfer_small_entries(
        engine_wrapper=engine_wrapper,
        small_session_data=small_session_data,
        session_debug_info=session_debug_info,
        small_register_ptrs=small_register_ptrs,
        small_register_lens=small_register_lens,
        session_bytes=session_bytes,
        session_transfer_s=session_transfer_s,
        bucket_idx=bucket_idx,
    )
    bucket_bytes += small_bytes

    timing.transfer_s = time.perf_counter() - t0
    timing.nbytes = bucket_bytes
    timing.num_large_buffers = num_large_buffers
    timing.num_small_buffers = num_small_buffers
    timing.session_bytes = session_bytes
    timing.session_transfer_s = session_transfer_s
    if log_bucket_details:
        logger.info(
            "[P2P] bucket %d: %.1f MB, register=%.1f ms, transfer=%.1f ms, deregister=%.1f ms, throughput=%.1f MB/s",
            bucket_idx,
            timing.nbytes / 1e6,
            timing.register_s * 1e3,
            timing.transfer_s * 1e3,
            timing.deregister_s * 1e3,
            timing.throughput_mb_s,
        )


class P2PTransportBackend(WeightTransportBackend):
    """RDMA P2P weight transport via the Mooncake TransferEngine.

    See module docstring for the architecture. The backend assumes:

    * The SGLang receiver exposes ``transport="p2p"`` on
      ``/prepare_weights_update`` and returns ``tensor_map`` +
      ``receiver_transfer_engine_infos``.
    * The local training rank can construct a Mooncake TransferEngine
      (mooncake-transfer-engine package installed, IB devices visible).
    """

    def __init__(self, config: TransportConfig, **kwargs: Any) -> None:
        super().__init__(config)
        # backend_config carries optional Mooncake engine setup overrides.
        be_cfg = config.backend_config or {}
        self._engine = None  # MooncakeTransferEngine (lazy-imported)
        # tensor_map[name] -> list of receiver locator dicts.
        self._tensor_map: Dict[str, List[Dict[str, Any]]] = {}
        # receiver_session_ids: list of session_id strings, one per inference TP rank.
        self._receiver_session_ids: List[str] = []
        self._session_debug_info: Dict[str, Dict[str, Any]] = {}
        # Warm-prepare coordination. When every trainer rank already has a
        # cached tensor_map, rank 0 asks SGLang to prepare the update without
        # returning the 100k+ locator JSON again, then broadcasts a tiny reuse
        # marker instead of the full map.
        self._prefer_cached_prepare: bool = False
        self._last_prepare_returned_tensor_map: bool = False
        # Source-side memory regions registered with Mooncake. The current
        # async path registers CPU scratch pools once and registers small GPU
        # entries per bucket in the worker.
        self._registered_source_ptrs: List[int] = []
        self._registered_intervals: List[Tuple[int, int]] = []
        # Ring of CPU-pinned scratch pools for async transfer
        # pipelining. While the worker thread runs Mooncake on pool A's
        # bytes, the main thread stages layer N+1 into pool B (or C, D,
        # etc.) — this hides per-bucket Mooncake latency behind the
        # trainer-side FSDP unshard/extract work. Each pool is
        # registered with Mooncake once at first use and reused for
        # every subsequent bucket.
        #
        # Default 2 pools (ping-pong). Raising XORL_P2P_NUM_POOLS and
        # XORL_P2P_MOONCAKE_WORKERS can hide per-call latency, but on
        # the scaled TP2 layout 3-4 pools/workers regressed due to
        # staging/NIC contention; treat higher values as experiments.
        n_pools = max(1, int(os.environ.get("XORL_P2P_NUM_POOLS", "2")))
        self._n_pools = n_pools
        self._cpu_scratch_pool_bytes: int = int(be_cfg.get("cpu_scratch_pool_bytes", _CPU_SCRATCH_POOL_BYTES))
        self._cpu_pool_min_bytes: int = int(be_cfg.get("cpu_pool_min_bytes", _CPU_POOL_MIN_BYTES))
        self._persist_small_registration: bool = _persist_small_registration_enabled()
        self._cpu_scratch_pools: List[Optional[torch.Tensor]] = [None] * n_pools
        self._cpu_scratch_pool_ptrs: List[int] = [0] * n_pools
        self._cpu_scratch_pool_nbytes: int = 0
        self._cpu_pool_idx: int = 0
        self._cpu_pool_pending_futures: List[Optional[Future]] = [None] * n_pools
        # Worker pool for async Mooncake calls. Default workers is 2; the
        # number can be tuned with XORL_P2P_MOONCAKE_WORKERS.
        self._transfer_executor: Optional[ThreadPoolExecutor] = None
        # Per-bucket timings collected across this sync. Populated by
        # transfer_bucket; read out by the caller (e.g. the e2e harness)
        # for a wall-time breakdown vs. the NCCL backend.
        self._bucket_timings: List[_BucketTiming] = []
        self._log_bucket_details: bool = _env_flag("XORL_P2P_LOG_BUCKET_DETAILS", False)
        self._collect_transfer_debug: bool = _env_flag("XORL_P2P_TRANSFER_DEBUG", False)
        self._transfer_chunk: int = _env_int("XORL_P2P_MOONCAKE_TRANSFER_CHUNK", 1)
        # Stable group name passed back in /complete_weights_update.
        self._group_name = config.group_name
        self._hostname: Optional[str] = be_cfg.get("hostname")
        self._resolved_hostname: Optional[str] = None
        self._gpu_id: int = be_cfg.get("gpu_id", 0)
        self._ib_device: Optional[str] = be_cfg.get("ib_device")
        self._run_post_process_weights: bool = bool(be_cfg.get("run_post_process_weights", False))
        # ---- Direct EP / multi-sender configuration ----
        # When True, this backend is used in a multi-rank training setup
        # where every training rank sends its own slice in parallel
        # (the topology that produces the lmsys article's 7x speedup).
        # The handler is responsible for invoking transfer_bucket on
        # every rank in sender_ranks and only with that rank's params.
        self._direct_ep_transfer: bool = bool(be_cfg.get("direct_ep_transfer", False))
        sender_ranks = be_cfg.get("sender_ranks")
        self._explicit_sender_rank_order: Optional[Tuple[int, ...]] = None
        self._explicit_sender_ranks: Optional[FrozenSet[int]] = None
        if sender_ranks is not None:
            self._explicit_sender_rank_order = tuple(dict.fromkeys(int(rank) for rank in sender_ranks))
            self._explicit_sender_ranks = frozenset(self._explicit_sender_rank_order)
        self._process_group = be_cfg.get("process_group")
        self._direct_ep_size: int = int(be_cfg.get("direct_ep_size", 0) or 0)
        self._direct_ep_dense_sharding: bool = bool(
            be_cfg.get("direct_ep_dense_sharding", _env_flag("XORL_P2P_DIRECT_EP_DENSE_SHARDING"))
        )
        self._sender_ep_ranks: Dict[int, int] = {}
        for item in be_cfg.get("sender_ep_ranks") or ():
            try:
                sender_rank, ep_rank = item
            except (TypeError, ValueError):
                logger.warning("[P2P] ignoring malformed sender_ep_ranks entry %r", item)
                continue
            self._sender_ep_ranks[int(sender_rank)] = int(ep_rank)
        # Optional per-rank predicate. When set, transfer_bucket filters
        # locator entries to only those that belong to *this* rank.
        # Receives the locator dict; returns True if this rank should
        # ship the corresponding slice. Default: ship everything (the
        # single-sender / rank-0-only path).
        self._rank_filter = be_cfg.get("rank_filter")
        # The rank index to claim ownership for in multi-sender mode.
        # Defaults to TransportConfig.training_rank.
        self._rank_index: int = int(be_cfg.get("rank_index", config.training_rank))
        # World size of the trainer side. Only matters when
        # direct_ep_transfer is on (every trainer rank ships its own
        # slice in parallel). Fall back to TransportConfig.training_world_size
        # — this is what the production handler sets — instead of a literal
        # 1, otherwise sender_ranks silently degrades to {0} and the
        # handler routes every non-rank-0 trainer through the gather/
        # broadcast fallback.
        self._world_size: int = int(be_cfg.get("world_size", config.training_world_size or 1))
        self._last_prepare_tensor_map_endpoint_indices: set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _record_matches_endpoint(record: Dict[str, Any], endpoint_idx: int, num_endpoints: int) -> bool:
        record_endpoint_idx = record.get("endpoint_idx")
        if record_endpoint_idx is None:
            return num_endpoints == 1 and endpoint_idx == 0
        try:
            return int(record_endpoint_idx) == endpoint_idx
        except (TypeError, ValueError):
            return False

    @classmethod
    def _drop_endpoint_records(
        cls,
        records: List[Dict[str, Any]],
        endpoint_idx: int,
        num_endpoints: int,
    ) -> List[Dict[str, Any]]:
        return [record for record in records if not cls._record_matches_endpoint(record, endpoint_idx, num_endpoints)]

    @classmethod
    def _drop_endpoint_locators(
        cls,
        tensor_map: Dict[str, List[Dict[str, Any]]],
        endpoint_idx: int,
        num_endpoints: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        updated: Dict[str, List[Dict[str, Any]]] = {}
        for name, locators in tensor_map.items():
            kept = cls._drop_endpoint_records(locators, endpoint_idx, num_endpoints)
            if kept:
                updated[name] = kept
        return updated

    @classmethod
    def _session_ids_for_endpoint(
        cls,
        records: List[Dict[str, Any]],
        endpoint_idx: int,
        num_endpoints: int,
    ) -> set[str]:
        return {
            str(record["session_id"])
            for record in records
            if record.get("session_id") and cls._record_matches_endpoint(record, endpoint_idx, num_endpoints)
        }

    def _receiver_session_infos(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for sid in self._receiver_session_ids:
            info = dict(self._session_debug_info.get(str(sid), {"session_id": sid}))
            info.setdefault("session_id", sid)
            infos.append(info)
        return infos

    @staticmethod
    def _expert_index_from_name(name: str) -> Optional[int]:
        parts = name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part != "experts":
                continue
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
        return None

    @classmethod
    def _experts_per_ep(
        cls,
        tensor_map: Dict[str, List[Dict[str, Any]]],
        ep_size: int,
    ) -> Optional[int]:
        if ep_size <= 1:
            return None
        expert_indices = {
            expert_idx
            for name in tensor_map
            for expert_idx in (cls._expert_index_from_name(name),)
            if expert_idx is not None
        }
        if not expert_indices:
            return None
        total_experts = max(expert_indices) + 1
        if expert_indices != set(range(total_experts)):
            logger.warning("[P2P] expert tensor_map names are not contiguous; keeping full map on each sender")
            return None
        if total_experts % ep_size != 0:
            logger.warning(
                "[P2P] total_experts=%d is not divisible by direct_ep_size=%d; keeping full map on each sender",
                total_experts,
                ep_size,
            )
            return None
        return total_experts // ep_size

    @classmethod
    def _endpoint_indices_for_tensor_map(
        cls,
        tensor_map: Dict[str, List[Dict[str, Any]]],
        num_endpoints: int,
    ) -> set[int]:
        endpoint_indices: set[int] = set()
        for locators in tensor_map.values():
            for loc in locators:
                endpoint_idx = loc.get("endpoint_idx")
                if endpoint_idx is None and num_endpoints == 1:
                    endpoint_indices.add(0)
                    continue
                try:
                    endpoint_indices.add(int(endpoint_idx))
                except (TypeError, ValueError):
                    continue
        return endpoint_indices

    @staticmethod
    def _copy_locator_list_for_scatter(
        locators: List[Dict[str, Any]],
        copy_mode: str,
    ) -> List[Dict[str, Any]]:
        if copy_mode == "deep":
            return [dict(loc) for loc in locators]
        if copy_mode == "none":
            return locators
        # Default: keep an independent list per scatter payload without
        # duplicating every immutable locator dict. Nonzero ranks copy the
        # dicts again when adopting/merging prepared state.
        return list(locators)

    @staticmethod
    def _scatter_locator_copy_mode() -> str:
        if "XORL_P2P_SCATTER_REUSE_LOCATORS" in os.environ:
            if _env_flag("XORL_P2P_SCATTER_REUSE_LOCATORS", False):
                return "none"
            if "XORL_P2P_SCATTER_COPY_MODE" not in os.environ:
                return "list"
        raw_mode = os.environ.get("XORL_P2P_SCATTER_COPY_MODE")
        if raw_mode is None:
            # Fast path: locator lists/dicts are read-only after SGLang prepare,
            # and scatter_object_list serializes each recipient payload anyway.
            return "none"
        raw = raw_mode.strip().lower()
        if raw in {"deep", "dict", "dicts"}:
            return "deep"
        if raw in {"list", "shallow", "lists"}:
            return "list"
        if raw in {"none", "reuse"}:
            return "none"
        logger.warning("[P2P] invalid XORL_P2P_SCATTER_COPY_MODE=%r; using reuse", raw_mode)
        return "none"

    def _filter_tensor_map_for_sender(
        self,
        tensor_map: Dict[str, List[Dict[str, Any]]],
        sender_rank: int,
        *,
        experts_per_ep: Optional[int] = None,
        locator_copy_mode: str = "list",
    ) -> Dict[str, List[Dict[str, Any]]]:
        sender_ep_rank = self._sender_ep_ranks.get(int(sender_rank))
        if experts_per_ep is None:
            experts_per_ep = self._experts_per_ep(tensor_map, self._direct_ep_size)
        if sender_ep_rank is None or experts_per_ep is None:
            return {
                name: self._copy_locator_list_for_scatter(locators, locator_copy_mode)
                for name, locators in tensor_map.items()
            }

        keep_dense = int(sender_rank) == 0
        filtered: Dict[str, List[Dict[str, Any]]] = {}
        for name, locators in tensor_map.items():
            expert_idx = self._expert_index_from_name(name)
            if expert_idx is None:
                if self._direct_ep_dense_sharding:
                    if not self.should_send_dense_param(name, int(sender_rank)):
                        continue
                elif not keep_dense:
                    continue
            elif expert_idx // experts_per_ep != sender_ep_rank:
                continue
            filtered[name] = self._copy_locator_list_for_scatter(locators, locator_copy_mode)
        return filtered

    def should_extract_dense_params_on_rank(self, rank: int) -> bool:
        return self._direct_ep_dense_sharding and int(rank) in self.sender_ranks

    @staticmethod
    def _dense_owner_key(name: str) -> str:
        """Canonicalize equivalent trainer/SGLang dense names for sharding.

        The handler sees fused trainer names before ``_unfuse_for_inference``
        (for example ``qkv_proj`` and ``gate_up_proj``), while the receiver
        tensor map sees the split SGLang names (``q_proj``/``k_proj``/``v_proj``
        and ``gate_proj``/``up_proj``). Hashing the canonical fused key keeps
        extraction, post-unfuse filtering, and tensor-map filtering aligned.
        """
        if P2PTransportBackend._expert_index_from_name(name) is not None:
            return name
        for split_name in (".q_proj.", ".k_proj.", ".v_proj."):
            if split_name in name:
                return name.replace(split_name, ".qkv_proj.", 1)
        for split_name in (".gate_proj.", ".up_proj."):
            if split_name in name:
                return name.replace(split_name, ".gate_up_proj.", 1)
        return name

    def should_send_dense_param(self, name: str, rank: int) -> bool:
        """Return whether ``rank`` owns this dense parameter in direct-EP mode.

        MoE expert names are never treated as dense here. With dense sharding
        disabled, rank 0 remains the sole dense sender, matching the historical
        path. With sharding enabled, names are assigned deterministically across
        the explicit sender rank order using a canonical fused dense name so
        handler extraction and tensor-map filtering make the same decision on
        every rank.
        """
        rank = int(rank)
        if self._expert_index_from_name(name) is not None:
            return False
        if not (self._direct_ep_transfer and self._world_size > 1 and self._direct_ep_dense_sharding):
            return rank == 0
        sender_order = self.sender_rank_order
        if not sender_order:
            return rank == 0
        owner_key = self._dense_owner_key(name)
        owner = sender_order[zlib.crc32(owner_key.encode("utf-8")) % len(sender_order)]
        return rank == int(owner)

    def filter_dense_buffer_for_rank(
        self,
        buffer: List[Tuple[str, torch.Tensor]],
        rank: int,
    ) -> List[Tuple[str, torch.Tensor]]:
        if not (self._direct_ep_transfer and self._world_size > 1):
            return buffer if int(rank) == 0 else []
        if not self._direct_ep_dense_sharding:
            return buffer if int(rank) == 0 else []
        return [(name, tensor) for name, tensor in buffer if self.should_send_dense_param(name, int(rank))]

    def _can_scatter_filtered_tensor_maps(self) -> bool:
        return (
            self._direct_ep_transfer
            and self._world_size > 1
            and self._explicit_sender_rank_order is not None
            and bool(self._sender_ep_ranks)
            and self._direct_ep_size > 1
            and hasattr(dist, "scatter_object_list")
        )

    def _initialize_payloads_for_sender_order(self) -> List[Any]:
        session_infos = self._receiver_session_infos()
        returned_endpoint_indices = set(self._last_prepare_tensor_map_endpoint_indices)
        all_endpoint_indices = set(range(len(self.config.endpoints)))
        if returned_endpoint_indices and returned_endpoint_indices != all_endpoint_indices:
            kind = "merge_tensor_map"
        else:
            kind = "tensor_map_with_infos"

        experts_per_ep = self._experts_per_ep(self._tensor_map, self._direct_ep_size)
        locator_copy_mode = self._scatter_locator_copy_mode()
        payloads: List[Any] = []
        for sender_rank in self.sender_rank_order:
            if int(sender_rank) == 0:
                payloads.append(("rank0_ready",))
                continue
            sender_tensor_map = self._filter_tensor_map_for_sender(
                self._tensor_map,
                sender_rank,
                experts_per_ep=experts_per_ep,
                locator_copy_mode=locator_copy_mode,
            )
            if kind == "merge_tensor_map":
                payloads.append(
                    (
                        kind,
                        sender_tensor_map,
                        session_infos,
                        tuple(sorted(returned_endpoint_indices)),
                    )
                )
            else:
                payloads.append((kind, sender_tensor_map, session_infos))
        return payloads

    def adopt_prepared_state(
        self,
        tensor_map: Dict[str, List[Dict[str, Any]]],
        receiver_session_ids: List[str],
        receiver_session_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Multi-sender hook: take the tensor_map + receiver session ids
        that rank 0 obtained from ``/prepare_weights_update`` and stand up
        the local Mooncake engine without doing the HTTP call again.

        Use this on every non-rank-0 sender. Rank 0 still calls
        :meth:`initialize` to drive the prepare handshake; it then
        broadcasts the tensor_map and session ids (e.g. via
        ``torch.distributed.broadcast_object_list``) and the other
        ranks call this. Each rank ends up with its own local engine
        bound to its own GPU/HCA but a shared view of the receiver's
        registered memory.
        """
        # Reuse a cached engine if this backend is being re-initialized
        # via the handler-side cache.
        if self._engine is None:
            self._engine = self._make_local_engine()
            if self._engine is None:
                return False
        self._tensor_map = dict(tensor_map)
        self._receiver_session_ids = list(receiver_session_ids)
        if receiver_session_infos is not None:
            self._session_debug_info = {
                str(info["session_id"]): dict(info) for info in receiver_session_infos if info.get("session_id")
            }
        else:
            self._session_debug_info = {
                sid: {"session_id": sid, "adopted_from_rank0": True} for sid in self._receiver_session_ids
            }
        return True

    def merge_prepared_state(
        self,
        tensor_map: Dict[str, List[Dict[str, Any]]],
        receiver_session_infos: List[Dict[str, Any]],
        endpoint_indices: Tuple[int, ...],
    ) -> bool:
        if self._engine is None:
            self._engine = self._make_local_engine()
            if self._engine is None:
                return False

        num_endpoints = len(self.config.endpoints)
        updated = {name: [dict(loc) for loc in locators] for name, locators in self._tensor_map.items()}
        indices = set(endpoint_indices) or self._endpoint_indices_for_tensor_map(tensor_map, num_endpoints)
        for endpoint_idx in indices:
            updated = self._drop_endpoint_locators(updated, endpoint_idx, num_endpoints)
        for name, locators in tensor_map.items():
            updated.setdefault(name, []).extend(dict(loc) for loc in locators)
        self._tensor_map = updated
        self._receiver_session_ids = [
            str(info["session_id"]) for info in receiver_session_infos if info.get("session_id")
        ]
        self._session_debug_info = {
            str(info["session_id"]): dict(info) for info in receiver_session_infos if info.get("session_id")
        }
        return True

    def initialize(self) -> bool:
        # Multi-sender: rank 0 drives the HTTP prepare and broadcasts the
        # result to every other sender. Non-zero ranks adopt that shared
        # state and stand up their own local Mooncake engine.
        if self._direct_ep_transfer and self._world_size > 1:
            return self._initialize_multi_sender()
        return self._initialize_single_sender()

    def _initialize_multi_sender(self) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            logger.error(
                "[P2P] direct_ep_transfer set but torch.distributed is not "
                "initialized. Initialize a process group first (the handler "
                "does this for FSDP/EP)."
            )
            return False

        is_rank0 = self._rank_index == 0
        group = self._process_group
        group_world_size = len(self.sender_rank_order)
        has_cached_state = bool(self._tensor_map and self._receiver_session_ids)
        cached_states: List[bool] = [False] * group_world_size
        try:
            dist.all_gather_object(cached_states, has_cached_state, group=group)
        except Exception as e:
            logger.warning(f"[P2P] cached-state all_gather failed; using full prepare: {e}")
            cached_states = [False] * group_world_size
        self._prefer_cached_prepare = all(cached_states)

        if _env_flag("XORL_P2P_PREINIT_NONZERO_ENGINES") and not is_rank0 and self._engine is None:
            logger.info("[P2P] pre-initializing local Mooncake engine while rank 0 prepares receivers")
            self._engine = self._make_local_engine()

        payload: List[Any] = [None]
        scatter_payloads: Optional[List[Any]] = None
        use_scatter_payloads = self._can_scatter_filtered_tensor_maps()
        if is_rank0:
            ok = self._initialize_single_sender()
            if ok:
                if self._last_prepare_returned_tensor_map:
                    if use_scatter_payloads:
                        scatter_payloads = self._initialize_payloads_for_sender_order()
                        payload[0] = ("scatter_payloads",)
                    else:
                        payload[0] = ("tensor_map", self._tensor_map, list(self._receiver_session_ids))
                else:
                    payload[0] = ("reuse_cached", list(self._receiver_session_ids))
            else:
                payload[0] = None
        # All ranks synchronize on init payload delivery. When direct-EP
        # sender mappings are available, scatter per-sender tensor maps instead
        # of broadcasting the full 1M+ locator map to every sender.
        if use_scatter_payloads:
            scatter_input: Optional[List[Any]] = None
            if is_rank0:
                scatter_input = scatter_payloads if scatter_payloads is not None else [payload[0]] * group_world_size
                if len(scatter_input) != group_world_size:
                    logger.error(
                        "[P2P] scatter payload count %d does not match sender group size %d",
                        len(scatter_input),
                        group_world_size,
                    )
                    scatter_input = [None] * group_world_size
            dist.scatter_object_list(payload, scatter_input, src=0, group=group)
        else:
            # payload[0] is None on failure so non-zero ranks can short-circuit
            # cleanly.
            dist.broadcast_object_list(payload, src=0, group=group)
        local_ok = True
        if payload[0] is None:
            if not is_rank0:
                logger.error("[P2P] rank 0 reported initialize() failure")
            local_ok = False
        elif not is_rank0:
            kind = payload[0][0]
            if kind == "reuse_cached":
                if not has_cached_state or self._engine is None:
                    logger.error("[P2P] rank 0 requested cached prepare reuse, but this rank has no cached state")
                    local_ok = False
                elif sids := payload[0][1]:
                    self._receiver_session_ids = list(sids)
                    self._session_debug_info = {
                        sid: {"session_id": sid, "adopted_from_rank0": True, "cached_prepare": True}
                        for sid in self._receiver_session_ids
                    }
            elif kind == "tensor_map":
                _, tmap, sids = payload[0]
                if not self.adopt_prepared_state(tmap, sids):
                    local_ok = False
            elif kind == "tensor_map_with_infos":
                _, tmap, infos = payload[0]
                sids = [str(info["session_id"]) for info in infos if info.get("session_id")]
                if not self.adopt_prepared_state(tmap, sids, receiver_session_infos=infos):
                    local_ok = False
            elif kind == "merge_tensor_map":
                _, tmap, infos, endpoint_indices = payload[0]
                if not self.merge_prepared_state(tmap, infos, tuple(int(idx) for idx in endpoint_indices)):
                    local_ok = False
            else:
                logger.error(f"[P2P] unknown initialize payload kind: {kind!r}")
                local_ok = False

        init_results: List[bool] = [False] * group_world_size
        try:
            dist.all_gather_object(init_results, bool(local_ok), group=group)
        except Exception as e:
            logger.error(f"[P2P] direct-EP initialize result all_gather failed: {e}")
            return False
        if not all(init_results):
            failed_ranks = [idx for idx, ok in enumerate(init_results) if not ok]
            logger.error(f"[P2P] direct-EP initialize failed on ranks {failed_ranks}")
            return False
        return True

    def _initialize_single_sender(self) -> bool:
        cfg = self.config
        if not cfg.endpoints:
            logger.error("[P2P] No endpoints provided in TransportConfig")
            return False

        # If we're being reused via the handler-side cache, the
        # engine is already constructed and the CPU scratch pools are
        # already registered. Skip both — they're the slow steps.
        if self._engine is None:
            self._engine = self._make_local_engine()
            if self._engine is None:
                return False

        sender_session_id = self._engine.get_session_id()
        sender_info = {
            "session_id": sender_session_id,
            "hostname": self._resolved_hostname or self._hostname,
            "gpu_id": self._gpu_id,
            "ib_device": self._engine.get_ib_device(),
            "training_rank": cfg.training_rank,
        }

        # Build prepare buckets once — we send them up front so SGLang
        # can size its registration / state.
        # For the single-sender variant we issue the prepare against each
        # endpoint and merge the returned tensor_maps. Endpoints are
        # independent, so fan out concurrently; on the 16-endpoint TP2 layout
        # this keeps cached prepare from becoming a serialized HTTP/JSON tail.
        has_cached_prepare_state = bool(self._tensor_map and self._receiver_session_ids)
        if not (self._direct_ep_transfer and self._world_size > 1):
            self._prefer_cached_prepare = has_cached_prepare_state
        request_cached_prepare = self._prefer_cached_prepare and has_cached_prepare_state
        self._last_prepare_returned_tensor_map = False
        self._last_prepare_tensor_map_endpoint_indices = set()
        num_endpoints = len(cfg.endpoints)
        prepare_workers = min(
            num_endpoints,
            _env_int("XORL_P2P_PREPARE_WORKERS", min(32, num_endpoints)),
        )

        def _prepare_endpoint(ep_idx: int, ep: Any, cached_prepare: bool) -> Tuple[int, Any, Dict[str, Any]]:
            url = f"http://{ep.host}:{ep.port}/prepare_weights_update"
            payload = {
                "buckets": [],  # buckets are not used in the p2p path
                "num_buckets": 0,
                "group_name": cfg.group_name,
                "transport": "p2p",
                "sender_transfer_engine_info": sender_info,
            }
            if cached_prepare:
                payload["p2p_return_tensor_map"] = False
            try:
                resp = requests.post(url, json=payload, timeout=_prepare_timeout_seconds())
            except requests.RequestException as e:
                raise RuntimeError(f"/prepare_weights_update to {ep.host}:{ep.port} failed: {e}") from e
            if cached_prepare and resp.status_code in (400, 422):
                retry_payload = dict(payload)
                retry_payload.pop("p2p_return_tensor_map", None)
                logger.warning(
                    f"[P2P] cached prepare was rejected by {ep.host}:{ep.port}; "
                    "retrying this endpoint with full tensor_map response"
                )
                try:
                    resp = requests.post(url, json=retry_payload, timeout=_prepare_timeout_seconds())
                except requests.RequestException as e:
                    raise RuntimeError(f"/prepare_weights_update retry to {ep.host}:{ep.port} failed: {e}") from e
            if resp.status_code != 200:
                raise RuntimeError(
                    f"/prepare_weights_update returned {resp.status_code} from {ep.host}:{ep.port}: {resp.text}"
                )
            try:
                body = resp.json()
            except ValueError as e:
                raise RuntimeError(f"/prepare_weights_update from {ep.host}:{ep.port} returned non-JSON") from e
            if not body.get("success", False):
                raise RuntimeError(f"prepare failed at {ep.host}:{ep.port}: {body.get('message')}")
            return ep_idx, ep, body

        tensor_map_endpoint_indices: set[int] = set()
        while True:
            if request_cached_prepare:
                # Cached prepare is the hot warm-sync path. Most warm prepares
                # return no tensor_map from any endpoint, so avoid copying the
                # large locator map unless an endpoint actually refreshes its
                # locators below.
                merged_tensor_map: Dict[str, List[Dict[str, Any]]] = self._tensor_map
            else:
                merged_tensor_map = {}
            merged_receiver_infos: List[Dict[str, Any]] = []
            if request_cached_prepare:
                for sid_idx, sid in enumerate(self._receiver_session_ids):
                    cached_info = dict(self._session_debug_info.get(str(sid), {"session_id": sid}))
                    cached_info.setdefault("session_id", sid)
                    if (
                        "endpoint_idx" not in cached_info
                        and num_endpoints > 1
                        and len(self._receiver_session_ids) == num_endpoints
                    ):
                        cached_info["endpoint_idx"] = sid_idx
                    merged_receiver_infos.append(cached_info)

            restart_full_prepare = False
            tensor_map_endpoint_indices = set()
            try:
                if prepare_workers == 1:
                    endpoint_results = [
                        _prepare_endpoint(ep_idx, ep, request_cached_prepare) for ep_idx, ep in enumerate(cfg.endpoints)
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=prepare_workers, thread_name_prefix="p2p-prepare") as executor:
                        futures = [
                            executor.submit(_prepare_endpoint, ep_idx, ep, request_cached_prepare)
                            for ep_idx, ep in enumerate(cfg.endpoints)
                        ]
                        endpoint_results = [future.result() for future in as_completed(futures)]
            except Exception as e:
                logger.error(f"[P2P] prepare fanout failed: {e}")
                return False

            for ep_idx, ep, body in sorted(endpoint_results, key=lambda item: item[0]):
                ep_tensor_map = body.get("tensor_map") or {}
                ep_receiver_infos = body.get("receiver_transfer_engine_infos") or []
                cached_sessions = self._session_ids_for_endpoint(merged_receiver_infos, ep_idx, num_endpoints)
                returned_sessions = {str(info["session_id"]) for info in ep_receiver_infos if info.get("session_id")}
                if (
                    request_cached_prepare
                    and not ep_tensor_map
                    and cached_sessions
                    and returned_sessions
                    and returned_sessions != cached_sessions
                ):
                    logger.warning(
                        f"[P2P] receiver sessions changed at {ep.host}:{ep.port}; "
                        "restarting prepare for all endpoints with full tensor_map response"
                    )
                    request_cached_prepare = False
                    self._last_prepare_returned_tensor_map = False
                    restart_full_prepare = True
                    break

                if ep_tensor_map:
                    self._last_prepare_returned_tensor_map = True
                    tensor_map_endpoint_indices.add(ep_idx)
                    merged_tensor_map = self._drop_endpoint_locators(merged_tensor_map, ep_idx, num_endpoints)
                for name, locators in ep_tensor_map.items():
                    # Tag each locator with its source endpoint so transfer_bucket
                    # knows where to reach it.
                    for loc in locators:
                        loc = dict(loc)
                        loc["endpoint_idx"] = ep_idx
                        merged_tensor_map.setdefault(name, []).append(loc)

                if ep_receiver_infos:
                    merged_receiver_infos = self._drop_endpoint_records(merged_receiver_infos, ep_idx, num_endpoints)
                for info in ep_receiver_infos:
                    info = dict(info)
                    info["endpoint_idx"] = ep_idx
                    merged_receiver_infos.append(info)

            if restart_full_prepare:
                continue
            break
        self._last_prepare_tensor_map_endpoint_indices = set(tensor_map_endpoint_indices)

        if merged_tensor_map:
            self._tensor_map = merged_tensor_map
        elif not self._tensor_map:
            logger.error("[P2P] prepare returned no tensor_map and no cached map is available")
            return False

        if merged_receiver_infos:
            self._receiver_session_ids = [
                str(info["session_id"]) for info in merged_receiver_infos if info.get("session_id")
            ]
            self._session_debug_info = {
                str(info["session_id"]): dict(info) for info in merged_receiver_infos if info.get("session_id")
            }
        elif not self._receiver_session_ids:
            logger.error("[P2P] prepare returned no receiver session ids and no cached sessions are available")
            return False

        total_locators = sum(len(v) for v in self._tensor_map.values())
        logger.info(
            f"[P2P] prepare ok: {len(self._tensor_map)} hf_names, "
            f"{total_locators} locators across "
            f"{len(self._receiver_session_ids)} receivers "
            f"(cached_prepare={request_cached_prepare and not self._last_prepare_returned_tensor_map}, "
            f"prepare_workers={prepare_workers}, "
            f"tensor_map_endpoints={len(tensor_map_endpoint_indices)}/{num_endpoints})"
        )
        return True

    def _ensure_cpu_scratch_pool(self) -> None:
        """Lazy-init the CPU pinned scratch pools.

        Pools are allocated and registered with Mooncake only when a bucket
        has at least one large entry that uses the CPU-source path. The
        transfer executor is initialized separately so small-only buckets do
        not allocate gigabytes of pinned memory.
        """
        if self._cpu_scratch_pools[0] is not None:
            return
        if self._engine is None:
            raise RuntimeError("[P2P] _ensure_cpu_scratch_pool called before initialize()")
        n = self._cpu_scratch_pool_bytes
        for i in range(self._n_pools):
            # uint8 makes byte-level offset math straightforward.
            pool = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            ptr = int(pool.data_ptr())
            t0 = time.perf_counter()
            # Use the singular API for CPU pinned memory; it routes through a
            # different Mooncake code path than the GPU-oriented batch register.
            ret = self._engine.engine.register_memory(ptr, n)
            dt = time.perf_counter() - t0
            if ret != 0:
                raise RuntimeError(f"[P2P] CPU scratch pool {i} register_memory failed: ret={ret} ({n / 1e9:.2f} GB)")
            self._cpu_scratch_pools[i] = pool
            self._cpu_scratch_pool_ptrs[i] = ptr
            self._registered_source_ptrs.append(ptr)
            self._registered_intervals.append((ptr, ptr + n))
            logger.info(
                f"[P2P] CPU scratch pool {i}/{self._n_pools}: registered {n / 1e9:.2f} GB "
                f"pinned at 0x{ptr:x} in {dt * 1000:.1f} ms"
            )
        self._cpu_scratch_pool_nbytes = n
        self._ensure_transfer_executor()

    def _ensure_transfer_executor(self) -> None:
        """Lazy-init the worker pool that runs Mooncake transfer calls."""
        if self._transfer_executor is not None:
            return
        # Number of concurrent Mooncake calls per rank. More than one worker
        # lets slower per-call paths hide Mooncake latency behind subsequent
        # staged buckets, especially when rank 0 has many small dense buckets.
        # Set XORL_P2P_MOONCAKE_WORKERS to override.
        n_workers = max(1, int(os.environ.get("XORL_P2P_MOONCAKE_WORKERS", "2")))
        self._transfer_executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="p2p-mooncake")
        if n_workers > 1:
            logger.info(f"[P2P] using {n_workers} concurrent Mooncake workers")

    def _wait_all_pending(self) -> None:
        """Block until every outstanding async transfer completes.

        Must be called before reading ``stats_summary`` or tearing down
        the engine — timings and bucket_bytes are only valid after the
        worker has populated them.
        """
        first_error: Optional[Exception] = None
        for i in range(len(self._cpu_pool_pending_futures)):
            fut = self._cpu_pool_pending_futures[i]
            if fut is None:
                continue
            try:
                fut.result()
            except Exception as e:
                logger.error(f"[P2P] async transfer on pool {i} raised: {e}")
                if first_error is None:
                    first_error = e
            finally:
                self._cpu_pool_pending_futures[i] = None
        if first_error is not None:
            raise first_error

    def flush_pending_transfers(self) -> None:
        """Drain async transfers before the handler resumes inference
        or measures wall time. Otherwise generation can read
        partially-updated weights, since RDMA writes land directly in
        ``param.data`` without going through ``/complete_weights_update``.
        """
        self._wait_all_pending()

    def complete_sync(self) -> None:
        """Per-sync teardown: drain in-flight transfers + send the
        receiver-side completion RPC. Leaves engine + CPU scratch pools
        + executor intact so the next sync can reuse them through the handler
        cache.

        Safe to call multiple times after a successful transfer drain. Not safe
        to call for a failed/partial sync: pending transfer errors are raised
        before the receiver-side completion RPC is sent.
        """
        cfg = self.config
        # Drain async transfers before any teardown step; required even
        # in cache-reuse mode because pending futures hold references to
        # CPU pool slots that the next sync will overwrite.
        self._wait_all_pending()

        # In multi-sender mode (direct_ep_transfer + world_size>1), only
        # rank 0 drives the HTTP /complete_weights_update — the receiver
        # services exactly one complete per sync, so non-zero ranks must
        # skip it or they trip the "no update in progress" error.
        skip_complete = self._direct_ep_transfer and self._world_size > 1 and self._rank_index != 0
        if not skip_complete:
            be_cfg = cfg.backend_config or {}
            flush_cache = bool(be_cfg.get("flush_cache", False))
            weight_version = be_cfg.get("weight_version")
            tied_weight_aliases = be_cfg.get("p2p_tied_weight_aliases") or {}
            complete_errors = []
            for ep in cfg.endpoints:
                url = f"http://{ep.host}:{ep.port}/complete_weights_update"
                payload = {
                    "group_name": cfg.group_name,
                    "flush_cache": flush_cache,
                    "transport": "p2p",
                    "run_post_process_weights": self._run_post_process_weights,
                }
                if weight_version is not None:
                    payload["weight_version"] = weight_version
                if tied_weight_aliases:
                    payload["p2p_tied_weight_aliases"] = tied_weight_aliases
                try:
                    resp = requests.post(url, json=payload, timeout=_HTTP_TIMEOUT_SECONDS)
                    if resp.status_code != 200:
                        complete_errors.append(f"{ep.host}:{ep.port} returned HTTP {resp.status_code}: {resp.text}")
                        continue
                    try:
                        body = resp.json()
                    except ValueError:
                        body = {}
                    if body and body.get("success") is False:
                        complete_errors.append(
                            f"{ep.host}:{ep.port} returned success=false: {body.get('message', body)}"
                        )
                except requests.RequestException as e:
                    complete_errors.append(f"{ep.host}:{ep.port} request failed: {e}")
            if complete_errors:
                raise RuntimeError("[P2P] /complete_weights_update failed: " + "; ".join(complete_errors))

        # Keep the receiver tensor_map/session ids with the cached backend.
        # The next sync can ask a warm SGLang receiver to skip returning the
        # huge locator JSON and can reuse this local map instead. If the
        # receiver restarted or rejected the cached prepare path, initialize()
        # replaces this state from the full prepare response.
        self._bucket_timings = []
        # CPU pool ping-pong cursor: reset so the next sync starts on
        # pool 0 (deterministic, makes timing logs easier to read).
        self._cpu_pool_idx = 0

    @property
    def is_alive(self) -> bool:
        """True if engine + scratch pools are still allocated and
        registered. Caller can use this to decide whether to reuse this
        backend on the next sync."""
        return self._engine is not None

    def destroy(self, *, complete_receiver: bool = True) -> None:
        """Full teardown: per-sync complete + release engine, executor,
        and CPU scratch pools. After this the backend cannot be reused."""
        complete_error: Optional[Exception] = None
        if complete_receiver:
            try:
                self.complete_sync()
            except Exception as e:
                complete_error = e
        else:
            try:
                self._wait_all_pending()
            except Exception as e:
                logger.warning(f"[P2P] skipping receiver completion after failed/aborted sync: {e}")
            self._bucket_timings = []
            self._cpu_pool_idx = 0

        if self._transfer_executor is not None:
            self._transfer_executor.shutdown(wait=True)
            self._transfer_executor = None

        # Best-effort source-side deregistration. Includes the CPU
        # pinned scratch pools (registered once at first transfer_bucket
        # call) and any leftover per-bucket registrations from older
        # code paths.
        if self._engine is not None and self._registered_source_ptrs:
            try:
                self._engine.batch_deregister(self._registered_source_ptrs)
            except Exception as e:
                logger.warning(f"[P2P] batch_deregister of source pointers failed: {e}")
            self._registered_source_ptrs = []
            self._registered_intervals = []
        # Release CPU pinned pools. PyTorch frees them on GC; we just
        # drop the handles here for clarity.
        self._cpu_scratch_pools = [None] * self._n_pools
        self._cpu_scratch_pool_ptrs = [0] * self._n_pools
        self._cpu_scratch_pool_nbytes = 0
        self._cpu_pool_idx = 0
        self._cpu_pool_pending_futures = [None] * self._n_pools

        self._engine = None
        if complete_error is not None:
            raise complete_error

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    def transfer_bucket(
        self,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        src_rank: int = 0,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
    ) -> None:
        if not self._direct_ep_transfer and src_rank != 0:
            raise ValueError(
                f"P2PTransportBackend rejects src_rank={src_rank} when "
                "direct_ep_transfer is off (set "
                "TransportConfig.backend_config['direct_ep_transfer']=True "
                "for multi-sender mode)."
            )
        if self._engine is None:
            raise RuntimeError("P2P backend not initialized — call initialize() first")

        # weight_version + flush_cache get applied at /complete_weights_update.
        # The handler signals these on the final bucket; stash them even if
        # this particular rank has no locator-owned transfers.
        if weight_version is not None:
            self.config.backend_config["weight_version"] = weight_version
        self.config.backend_config["flush_cache"] = bool(
            flush_cache or self.config.backend_config.get("flush_cache", False)
        )

        # Two-pass design:
        #
        # Pass 1: collect (session_id, peer_ptr, nbytes, src_view) for
        # every locator in the bucket.
        # Pass 2 per session: sort by peer_ptr, stage src_views into the
        # CPU pool in peer-ptr order so pool offsets match peer-ptr
        # adjacency, then coalesce neighboring entries whose pool & peer
        # ptrs are adjacent into single (src_ptr, peer_ptr, nbytes) tuples.
        # Ship as one batch_transfer_sync per session.
        #
        # Why this matters: receiver locators emit per-expert HF names
        # (e.g. experts.0.gate_proj.weight) that physically live in
        # contiguous slices of receiver-internal tensors (w13_weight).
        # Without coalescing, our trainer ships 96 buffers per layer's
        # MoE batch, which exceeds Mooncake's per-call stability cap
        # (~96-buffer ret=-1 mode observed empirically). After
        # coalescing, a layer's 96 contiguous expert slices collapse
        # to ~2 entries (w13 region, w2 region) per receiver-rank.
        # This yields one entry per receiver-internal name, fully populated in
        # CPU pinned source.
        t_prepare = time.perf_counter()
        pending: Dict[str, List[_PendingTransfer]] = {}
        skipped_errors: List[str] = []
        for name, tensor in bucket:
            locators = self._locators_for_source_name(name)
            if not locators:
                skipped_errors.append(f"{name!r}: no receiver locator")
                continue
            locators_for_rank = 0
            for loc in locators:
                if self._rank_filter is not None and not self._rank_filter(loc):
                    continue
                locators_for_rank += 1
                src_view = self._slice_source_for_locator(name, tensor, loc)
                if src_view is None:
                    skipped_errors.append(f"{name!r}: receiver locator is incompatible with source tensor")
                    continue
                src_nbytes = src_view.numel() * src_view.element_size()
                expected = int(loc.get("nbytes", src_nbytes))
                if src_nbytes != expected:
                    skipped_errors.append(
                        f"[P2P] size mismatch for {name!r} after slicing: "
                        f"source={src_nbytes}, receiver expects={expected}. "
                        f"Check tensor map slice metadata."
                    )
                    continue
                session_id = loc.get("session_id")
                if not session_id:
                    skipped_errors.append(f"{name!r}: receiver locator is missing session_id")
                    continue
                pending.setdefault(session_id, []).append(
                    _PendingTransfer(
                        peer_ptr=int(loc["ptr"]),
                        nbytes=src_nbytes,
                        src_view=src_view,
                        name=name,
                        loc=loc,
                    )
                )
            if locators_for_rank == 0:
                logger.debug("[P2P] no receiver locators owned by this sender for parameter %r", name)

        if skipped_errors:
            preview = "; ".join(skipped_errors[:5])
            if len(skipped_errors) > 5:
                preview += f"; ... {len(skipped_errors) - 5} more"
            raise RuntimeError(f"[P2P] receiver tensor_map is incomplete or incompatible: {preview}")

        if not pending:
            logger.debug("[P2P] transfer_bucket produced no transfers for this sender")
            return

        timing = _BucketTiming()
        self._bucket_timings.append(timing)
        bucket_idx = len(self._bucket_timings)
        timing.prepare_s = time.perf_counter() - t_prepare

        # Split entries into "large" (CPU pool path) and "small"
        # (GPU-direct path with per-bucket register). Mooncake's CPU-source
        # path on raw tiny buffers is unstable on our cluster, so only CUDA
        # tensors below the threshold use GPU-direct. CPU tensors, including
        # trainer-side FP8 scale tensors, are always staged through the
        # pre-registered CPU pool.
        pending_small: Dict[str, List[_PendingTransfer]] = {}
        for sid, entries in list(pending.items()):
            small: List[_PendingTransfer] = []
            large: List[_PendingTransfer] = []
            for e in entries:
                if e.src_view.is_cuda and e.nbytes < self._cpu_pool_min_bytes:
                    small.append(e)
                else:
                    large.append(e)
            if small:
                pending_small[sid] = small
            if large:
                pending[sid] = large
            else:
                pending.pop(sid, None)

        # First-call lazy init. Large entries need the registered CPU pool;
        # small-only buckets only need the transfer executor.
        t_pool_init = time.perf_counter()
        if pending:
            self._ensure_cpu_scratch_pool()
        else:
            self._ensure_transfer_executor()
        timing.pool_init_s = time.perf_counter() - t_pool_init

        # Pick the current pool/future slot. If a prior transfer is still
        # using it, block until it drains before overwriting the slot.
        pool_idx = self._cpu_pool_idx
        prior_future = self._cpu_pool_pending_futures[pool_idx]
        t_pool_wait = time.perf_counter()
        if prior_future is not None:
            prior_future.result()  # surface worker exceptions
            self._cpu_pool_pending_futures[pool_idx] = None
        timing.pool_wait_s = time.perf_counter() - t_pool_wait
        pool = self._cpu_scratch_pools[pool_idx] if pending else None
        pool_ptr = self._cpu_scratch_pool_ptrs[pool_idx] if pending else 0

        # Hold references to staged sub-tensors and src views so the
        # caller can reshard immediately after this method returns
        # without freeing GPU memory the NIC is still reading. Both
        # lists travel into the future closure and stay alive until
        # the worker finishes.
        slice_holds: List[torch.Tensor] = []
        src_view_holds: List[torch.Tensor] = []
        # Per-session transfer lists after coalescing. The debug list tracks
        # which original locators contributed to each emitted Mooncake buffer.
        by_session: Dict[str, Tuple[List[int], List[int], List[int], Optional[List[_TransferDebugSample]]]] = {}
        scratch_offset_bytes = 0
        staged_sources: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...], str, int], int] = {}
        unique_staged_bytes = 0
        reused_staged_bytes = 0
        total_pre_coalesce = 0
        total_post_coalesce = 0
        t_stage = time.perf_counter()

        for session_id, entries in pending.items():
            # Sort by peer_ptr so adjacent receiver-side regions cluster
            # together for the coalesce pass. Source pool offsets are
            # also assigned in this order, so pool adjacency matches
            # peer adjacency exactly.
            entries.sort(key=lambda e: e.peer_ptr)
            total_pre_coalesce += len(entries)

            # Stage all src_views into the CPU pool in sorted order.
            staged: List[_StagedTransfer] = []
            for e in entries:
                if pool is None:
                    raise RuntimeError("[P2P] CPU scratch pool was not initialized")
                source_key = _source_view_key(e.src_view, e.nbytes)
                src_ptr = staged_sources.get(source_key)
                if src_ptr is None:
                    element_size = max(1, int(e.src_view.element_size()))
                    scratch_offset_bytes = _align_up(pool_ptr + scratch_offset_bytes, element_size) - pool_ptr
                    if scratch_offset_bytes + e.nbytes > self._cpu_scratch_pool_nbytes:
                        # The scratch pool must hold the largest staged bucket for
                        # this sender. The default P2P MoE bucket cap is 2 GiB and
                        # the default pool is 4 GiB, so raising the bucket cap may
                        # require raising XORL_P2P_CPU_SCRATCH_POOL_BYTES too.
                        raise RuntimeError(
                            f"[P2P] CPU scratch pool exhausted: bucket needs "
                            f">{scratch_offset_bytes + e.nbytes} bytes but pool "
                            f"is {self._cpu_scratch_pool_nbytes} bytes. Increase "
                            f"XORL_P2P_CPU_SCRATCH_POOL_BYTES."
                        )
                    slot_uint8 = pool[scratch_offset_bytes : scratch_offset_bytes + e.nbytes]
                    slot_view = slot_uint8.view(e.src_view.dtype).view(e.src_view.shape)
                    slot_view.copy_(e.src_view, non_blocking=True)
                    slice_holds.append(slot_view)
                    src_view_holds.append(e.src_view)
                    src_ptr = pool_ptr + scratch_offset_bytes
                    staged_sources[source_key] = src_ptr
                    unique_staged_bytes += e.nbytes
                    scratch_offset_bytes += e.nbytes
                else:
                    reused_staged_bytes += e.nbytes
                    if e.src_view.is_cuda:
                        src_view_holds.append(e.src_view)
                staged.append(
                    _StagedTransfer(
                        src_ptr=src_ptr,
                        peer_ptr=e.peer_ptr,
                        nbytes=e.nbytes,
                        memory_handle=_locator_memory_handle(e.loc),
                        name=e.name,
                        loc=e.loc,
                    )
                )

            # Coalesce: walk staged in sorted-by-peer order, merging
            # adjacent entries whose source pool ptr AND peer ptr are
            # contiguous within the same receiver registration. After this
            # pass, a layer's per-expert slices collapse to a small number
            # of receiver-backed regions per session without asking
            # Mooncake to write across registration boundaries.
            src_ptrs: List[int] = []
            peer_ptrs: List[int] = []
            lens: List[int] = []
            debug_entries: Optional[List[_TransferDebugSample]] = [] if self._collect_transfer_debug else None
            memory_handles: List[Optional[int]] = []
            for staged_entry in staged:
                if (
                    src_ptrs
                    and src_ptrs[-1] + lens[-1] == staged_entry.src_ptr
                    and peer_ptrs[-1] + lens[-1] == staged_entry.peer_ptr
                    and staged_entry.memory_handle is not None
                    and memory_handles[-1] == staged_entry.memory_handle
                ):
                    lens[-1] += staged_entry.nbytes
                    if debug_entries is not None:
                        debug_entries[-1].add(staged_entry.name, staged_entry.loc, staged_entry.nbytes)
                else:
                    src_ptrs.append(staged_entry.src_ptr)
                    peer_ptrs.append(staged_entry.peer_ptr)
                    lens.append(staged_entry.nbytes)
                    memory_handles.append(staged_entry.memory_handle)
                    if debug_entries is not None:
                        debug_sample = _TransferDebugSample()
                        debug_sample.add(staged_entry.name, staged_entry.loc, staged_entry.nbytes)
                        debug_entries.append(debug_sample)
            total_post_coalesce += len(src_ptrs)
            by_session[session_id] = (src_ptrs, peer_ptrs, lens, debug_entries)

        if self._log_bucket_details and total_post_coalesce < total_pre_coalesce:
            logger.info(
                f"[P2P] coalesced {total_pre_coalesce} entries → "
                f"{total_post_coalesce} ({total_pre_coalesce / total_post_coalesce:.1f}x reduction)"
            )
        if self._log_bucket_details and reused_staged_bytes:
            logger.info(
                "[P2P] staged-source reuse saved %.1f MB (unique_staged=%.1f MB, transfer_fanout=%.1f MB)",
                reused_staged_bytes / 1e6,
                unique_staged_bytes / 1e6,
                (unique_staged_bytes + reused_staged_bytes) / 1e6,
            )

        # Build small-entries metadata on the main thread — Mooncake's
        # batch_register doesn't enqueue CUDA work but
        # _intervals_per_cuda_segment calls torch.cuda.memory_snapshot()
        # which we keep on main as a precaution. The actual transfer
        # work happens in the worker.
        small_session_data: Dict[str, List[Tuple[int, int, int, Optional[_TransferDebugEntry]]]] = {}
        small_register_ptrs: List[int] = []
        small_register_lens: List[int] = []
        if pending_small:
            small_persistent_intervals: List[Tuple[int, int]] = []
            small_transient_intervals: List[Tuple[int, int]] = []
            for session_id, entries in pending_small.items():
                triples: List[Tuple[int, int, int, Optional[_TransferDebugEntry]]] = []
                for e in entries:
                    sv = e.src_view.contiguous()
                    src_view_holds.append(sv)
                    storage = sv.untyped_storage()
                    s_start = int(storage.data_ptr())
                    s_end = s_start + int(storage.nbytes())
                    # Persistently register only no-copy contiguous views backed
                    # by stable model-parameter storage. Temporary contiguous
                    # copies must stay per-bucket because they are released after
                    # the async worker drains.
                    can_persist = (
                        self._persist_small_registration
                        and int(sv.data_ptr()) == int(e.src_view.data_ptr())
                        and int(sv.untyped_storage().data_ptr()) == int(e.src_view.untyped_storage().data_ptr())
                    )
                    if can_persist:
                        small_persistent_intervals.append((s_start, s_end))
                    else:
                        small_transient_intervals.append((s_start, s_end))
                    debug_entry = (
                        _transfer_debug_entry(e.name, e.loc, e.nbytes) if self._collect_transfer_debug else None
                    )
                    triples.append((int(sv.data_ptr()), e.peer_ptr, e.nbytes, debug_entry))
                small_session_data[session_id] = triples
            if small_persistent_intervals:
                self._register_persistent_source_intervals(small_persistent_intervals, bucket_idx=bucket_idx)
            if small_transient_intervals:
                small_segs = self._intervals_per_cuda_segment(small_transient_intervals)
                small_register_ptrs = [iv[0] for iv in small_segs]
                small_register_lens = [iv[1] - iv[0] for iv in small_segs]
        timing.stage_s = time.perf_counter() - t_stage

        # The CPU pool is permanently registered; small entries
        # register/dereg in the worker. Both register_s and
        # deregister_s stay zero on the main-thread accounting because
        # all that work moves into transfer_s on the worker.
        timing.register_s = 0.0
        timing.deregister_s = 0.0

        # CUDA event records "all GPU work enqueued so far has
        # completed". The worker waits on this before issuing Mooncake
        # calls because Mooncake's NIC reads bypass CUDA streams entirely.
        # CPU-only protocol tests use a no-op event.
        if torch.cuda.is_available():
            copy_done_event = torch.cuda.Event()
            copy_done_event.record()
        else:
            copy_done_event = _CompletedCudaEvent()

        if self._transfer_executor is None:
            raise RuntimeError("[P2P] transfer executor was not initialized")
        t_submit = time.perf_counter()
        future = self._transfer_executor.submit(
            _do_async_transfer,
            engine_wrapper=self._engine,
            copy_done_event=copy_done_event,
            by_session=by_session,
            small_session_data=small_session_data,
            session_debug_info=self._session_debug_info,
            small_register_ptrs=small_register_ptrs,
            small_register_lens=small_register_lens,
            chunk=self._transfer_chunk,
            use_async_api=_async_api_enabled(cached_prepare=self._prefer_cached_prepare),
            timing=timing,
            bucket_idx=bucket_idx,
            slice_holds=slice_holds,
            src_view_holds=src_view_holds,
            log_bucket_details=self._log_bucket_details,
        )
        timing.submit_s = time.perf_counter() - t_submit
        self._cpu_pool_pending_futures[pool_idx] = future

        # Round-robin to the next pool. With N pools and N workers,
        # the main thread can stage up to N buckets ahead while workers
        # drain them in parallel.
        self._cpu_pool_idx = (pool_idx + 1) % self._n_pools

    # ------------------------------------------------------------------
    # Topology hints
    # ------------------------------------------------------------------

    @property
    def bucket_timings(self) -> List[_BucketTiming]:
        """Per-bucket wall-time breakdown collected during this sync."""
        # Async transfers may still be in flight; their timing fields
        # are filled in by the worker thread. Drain before snapshotting
        # so the caller sees a self-consistent list.
        self._wait_all_pending()
        return list(self._bucket_timings)

    def stats_summary(self) -> Dict[str, Any]:
        """Aggregate timing summary for the most recent sync.

        Returns a dict with: ``num_buckets``, ``total_bytes``, ``total_s``,
        main-thread staging fields, ``register_s``, ``transfer_s``, ``deregister_s``,
        ``transfer_throughput_mb_s`` (transfer-only — excludes
        main-thread/register/deregister), ``effective_throughput_mb_s``
        (bucket wall-time, the number to compare against NCCL).
        """
        self._wait_all_pending()
        timings = self._bucket_timings
        bucket_transfer_s = [t.transfer_s for t in timings]
        bucket_total_s = [t.total_s for t in timings]
        total_bytes = sum(t.nbytes for t in timings)
        prepare_s = sum(t.prepare_s for t in timings)
        pool_init_s = sum(t.pool_init_s for t in timings)
        pool_wait_s = sum(t.pool_wait_s for t in timings)
        stage_s = sum(t.stage_s for t in timings)
        submit_s = sum(t.submit_s for t in timings)
        register_s = sum(t.register_s for t in timings)
        transfer_s = sum(t.transfer_s for t in timings)
        deregister_s = sum(t.deregister_s for t in timings)
        main_thread_s = prepare_s + pool_init_s + pool_wait_s + stage_s + submit_s
        total_s = sum(t.total_s for t in timings)
        session_bytes: Dict[str, int] = {}
        session_transfer_s: Dict[str, float] = {}
        for timing in timings:
            for session_id, nbytes in timing.session_bytes.items():
                session_bytes[session_id] = session_bytes.get(session_id, 0) + nbytes
            for session_id, seconds in timing.session_transfer_s.items():
                session_transfer_s[session_id] = session_transfer_s.get(session_id, 0.0) + seconds

        def _percentile(values: List[float], percentile: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            index = min(len(ordered) - 1, max(0, int((percentile / 100.0) * len(ordered))))
            return ordered[index]

        top_sessions_by_transfer_s = []
        for session_id, seconds in session_transfer_s.items():
            nbytes = session_bytes.get(session_id, 0)
            top_sessions_by_transfer_s.append(
                {
                    "session_id": session_id,
                    "transfer_s": seconds,
                    "total_bytes": nbytes,
                    "throughput_mb_s": ((nbytes / 1e6) / seconds if seconds > 0 else 0.0),
                }
            )
        top_sessions_by_transfer_s.sort(key=lambda row: row["transfer_s"], reverse=True)

        slowest_buckets = [
            {
                "bucket": bucket_idx,
                "total_s": timing.total_s,
                "main_thread_s": timing.main_thread_s,
                "prepare_s": timing.prepare_s,
                "pool_init_s": timing.pool_init_s,
                "pool_wait_s": timing.pool_wait_s,
                "stage_s": timing.stage_s,
                "submit_s": timing.submit_s,
                "transfer_s": timing.transfer_s,
                "total_bytes": timing.nbytes,
                "large_buffers": timing.num_large_buffers,
                "small_buffers": timing.num_small_buffers,
            }
            for bucket_idx, timing in sorted(
                enumerate(timings, start=1),
                key=lambda item: item[1].total_s,
                reverse=True,
            )[:5]
        ]

        return {
            "num_buckets": float(len(timings)),
            "total_bytes": float(total_bytes),
            "prepare_s": prepare_s,
            "pool_init_s": pool_init_s,
            "pool_wait_s": pool_wait_s,
            "stage_s": stage_s,
            "submit_s": submit_s,
            "main_thread_s": main_thread_s,
            "register_s": register_s,
            "transfer_s": transfer_s,
            "deregister_s": deregister_s,
            "total_s": total_s,
            "transfer_throughput_mb_s": ((total_bytes / 1e6) / transfer_s if transfer_s > 0 else 0.0),
            "effective_throughput_mb_s": ((total_bytes / 1e6) / total_s if total_s > 0 else 0.0),
            "max_bucket_transfer_s": max(bucket_transfer_s) if bucket_transfer_s else 0.0,
            "p50_bucket_transfer_s": _percentile(bucket_transfer_s, 50),
            "p95_bucket_transfer_s": _percentile(bucket_transfer_s, 95),
            "max_bucket_total_s": max(bucket_total_s) if bucket_total_s else 0.0,
            "p50_bucket_total_s": _percentile(bucket_total_s, 50),
            "p95_bucket_total_s": _percentile(bucket_total_s, 95),
            "num_large_buffers": float(sum(t.num_large_buffers for t in timings)),
            "num_small_buffers": float(sum(t.num_small_buffers for t in timings)),
            "slowest_buckets": slowest_buckets,
            "top_sessions_by_transfer_s": top_sessions_by_transfer_s[:5],
        }

    @property
    def sender_ranks(self) -> FrozenSet[int]:
        if self._direct_ep_transfer and self._world_size > 1:
            if self._explicit_sender_ranks is not None:
                return self._explicit_sender_ranks
            return frozenset(range(self._world_size))
        return frozenset({0})

    @property
    def sender_rank_order(self) -> Tuple[int, ...]:
        if self._direct_ep_transfer and self._world_size > 1:
            if self._explicit_sender_rank_order is not None:
                return self._explicit_sender_rank_order
            return tuple(range(self._world_size))
        return (0,)

    @property
    def has_explicit_sender_ranks(self) -> bool:
        return self._explicit_sender_ranks is not None

    @property
    def supports_direct_ep_transfer(self) -> bool:
        return self._direct_ep_transfer

    @property
    def supports_direct_pp_transfer(self) -> bool:
        # PP stage leaders still route through rank 0 in the handler. Keep
        # this false until the PP-direct handler path is implemented.
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _intervals_per_cuda_segment(
        self,
        intervals: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Constrain each interval to the CUDA-allocator's *active_allocated*
        blocks.

        ``ibv_reg_mr`` (and Mooncake's wrapper) returns EFAULT when asked
        to register a virtual-address range that crosses two distinct
        physical mappings, or that includes pages currently cached/free
        in PyTorch's caching allocator (those virtual addresses are reserved but not
        backed by physical memory the IB driver can pin).

        Mirrors SGLang's ``register_memory_region_v2`` — walks the
        ``memory_snapshot`` per-segment and within each segment accumulates
        runs of contiguous ``active_allocated`` blocks that overlap the
        candidate intervals, emitting one merged range per run.
        """
        if not intervals:
            return []
        try:
            snapshot = torch.cuda.memory.memory_snapshot()
        except Exception as e:
            logger.warning(
                f"[P2P] torch.cuda.memory.memory_snapshot failed: {e}; "
                "falling back to raw intervals (may EFAULT on registration)."
            )
            return intervals

        # Sort the candidate intervals once for an O(N+M) sweep.
        sorted_candidates = sorted(intervals, key=lambda iv: iv[0])

        def _overlaps_any(start: int, end: int) -> bool:
            for cs, ce in sorted_candidates:
                if ce <= start:
                    continue
                if cs >= end:
                    return False
                return True
            return False

        # Register each active_allocated block separately. We previously
        # tried merging adjacent blocks within a segment, but Mooncake's
        # ibv_reg_mr returned EFAULT on the merged ranges (likely because
        # the merged span crosses an internal allocator boundary that
        # isn't representable as one MR). One-block-at-a-time is more
        # registrations but is the granularity the IB driver expects.
        out: List[Tuple[int, int]] = []
        for seg in snapshot:
            for block in seg.get("blocks", []) or []:
                addr = int(block.get("address", -1))
                size = int(block.get("size", -1))
                state = block.get("state", "")
                if addr < 0 or size <= 0 or state != "active_allocated":
                    continue
                if not _overlaps_any(addr, addr + size):
                    continue
                out.append((addr, addr + size))
        return out

    def _merge_against_registered(
        self,
        candidates: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Return the (merged) intervals from `candidates` that are not
        already covered by any previously-registered range.

        Both sets are merged into one list of disjoint ranges; the diff
        against `self._registered_intervals` is the new coverage we need
        to ask Mooncake to register.
        """
        if not candidates:
            return []

        def merge(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            sorted_iv = sorted(iv for iv in intervals if iv[1] > iv[0])
            merged: List[Tuple[int, int]] = []
            for s, e in sorted_iv:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            return merged

        cand_merged = merge(candidates)
        registered = merge(self._registered_intervals)
        new: List[Tuple[int, int]] = []

        for s, e in cand_merged:
            cur_s = s
            for rs, re in registered:
                if re <= cur_s:
                    continue
                if rs >= e:
                    break
                # rs..re overlaps cur_s..e in some way; carve off the part
                # before this registered range.
                if rs > cur_s:
                    new.append((cur_s, rs))
                cur_s = max(cur_s, re)
                if cur_s >= e:
                    break
            if cur_s < e:
                new.append((cur_s, e))
        return merge(new)

    def _register_persistent_source_intervals(
        self,
        intervals: List[Tuple[int, int]],
        *,
        bucket_idx: int,
    ) -> None:
        """Register stable CUDA source regions once per backend lifetime."""
        if not intervals:
            return
        if self._engine is None:
            raise RuntimeError("[P2P] persistent small registration requires an initialized Mooncake engine")

        # Fast path: if the raw tensor storage interval is already covered by
        # a registered active block, skip the expensive CUDA memory snapshot.
        raw_new = self._merge_against_registered(intervals)
        if not raw_new:
            return

        segments = self._intervals_per_cuda_segment(raw_new)
        new_segments = self._merge_against_registered(segments)
        if not new_segments:
            return

        ptrs = [start for start, _ in new_segments]
        lengths = [end - start for start, end in new_segments]
        ret = self._engine.batch_register(ptrs, lengths)
        if ret != 0:
            raise RuntimeError(
                f"[P2P] persistent small-source batch_register failed: ret={ret} "
                f"(bucket {bucket_idx}, regions={len(ptrs)})"
            )
        self._registered_source_ptrs.extend(ptrs)
        self._registered_intervals.extend(new_segments)

    def _locators_for_source_name(self, name: str) -> Optional[List[Dict[str, Any]]]:
        locators = self._tensor_map.get(name)
        if locators or name.startswith("language_model."):
            return locators
        # Kimi-K2.5 SGLang wrappers expose language model tensors under
        # language_model.*, while XORL trains the unwrapped text model.
        return self._tensor_map.get(f"language_model.{name}")

    @staticmethod
    def _slice_source_for_locator(
        name: str,
        full_tensor: torch.Tensor,
        loc: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Extract the sub-region of the trainer's full HF tensor that
        corresponds to a single receiver locator.

        The trainer holds the *full* HF tensor (training TP=1 means FSDP
        unshard gives the full param). The receiver locator's ``slice``
        field is in full HF coordinates and tells us which rectangle
        belongs at this peer's address.
        """
        full_tensor = P2PTransportBackend._normalize_source_for_locator(name, full_tensor, loc)
        slc = loc.get("slice")
        if slc is None:
            # Replicated / no sharding — use the whole tensor.
            return P2PTransportBackend._normalize_sliced_source_for_locator(name, full_tensor, loc)

        full_shape = loc.get("full_shape")
        if full_shape is not None and list(full_tensor.shape) != list(full_shape):
            local_view = P2PTransportBackend._slice_qwen35_linear_attention_local_param(
                name,
                full_tensor,
                loc,
                full_shape,
                slc,
            )
            if local_view is not None:
                return P2PTransportBackend._normalize_sliced_source_for_locator(name, local_view, loc)
            logger.warning(
                f"[P2P] full_shape mismatch for {name!r}: "
                f"trainer={list(full_tensor.shape)} vs receiver={full_shape}. "
                "Check unfuse / quantization in the handler."
            )
            return None

        index: Tuple[slice, ...] = tuple(slice(int(s[0]), int(s[1])) for s in slc)
        return P2PTransportBackend._normalize_sliced_source_for_locator(name, full_tensor[index], loc)

    @staticmethod
    def _normalize_source_for_locator(
        name: str,
        full_tensor: torch.Tensor,
        loc: Dict[str, Any],
    ) -> torch.Tensor:
        """Normalize trainer-side tensors for receiver-specific HF layouts."""
        full_shape = loc.get("full_shape")
        if (
            ".linear_attn." in name
            and name.endswith(".conv1d.weight")
            and full_shape is not None
            and full_tensor.ndim == len(full_shape) + 1
            and full_tensor.shape[1] == 1
        ):
            squeezed = full_tensor.squeeze(1)
            if list(squeezed.shape) == list(full_shape):
                return squeezed
        return full_tensor

    @staticmethod
    def _normalize_sliced_source_for_locator(
        name: str,
        local_view: torch.Tensor,
        loc: Dict[str, Any],
    ) -> torch.Tensor:
        """Normalize trainer-side local slices for receiver-specific dtypes."""
        if not (
            ".linear_attn." in name
            and (name.endswith(".A_log") or name.endswith(".dt_bias"))
            and torch.is_floating_point(local_view)
        ):
            return local_view

        dtype_name = str(loc.get("dtype") or "")
        target_dtype = {
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "half": torch.float16,
        }.get(dtype_name)
        if target_dtype is None or local_view.dtype == target_dtype:
            return local_view
        return local_view.to(dtype=target_dtype)

    @staticmethod
    def _slice_qwen35_linear_attention_local_param(
        name: str,
        full_tensor: torch.Tensor,
        loc: Dict[str, Any],
        full_shape: Any,
        slc: Any,
    ) -> Optional[torch.Tensor]:
        """Handle Qwen3.5 linear-attention locators that expose TP-local vectors.

        Some receiver builds expose ``A_log``/``dt_bias`` locators with the
        receiver-local shape (for example ``[8]`` on TP=4) instead of the full
        HF shape (``[32]``). The locator still carries ``tp_rank``, so the
        sender can recover the intended full-tensor slice.
        """
        if not (
            ".linear_attn." in name
            and (name.endswith(".A_log") or name.endswith(".dt_bias"))
            and full_tensor.ndim == 1
            and isinstance(full_shape, list)
            and len(full_shape) == 1
        ):
            return None

        local_len = int(full_shape[0])
        if local_len <= 0 or full_tensor.shape[0] <= local_len or full_tensor.shape[0] % local_len != 0:
            return None

        if slc is not None:
            if len(slc) != 1:
                return None
            start, stop = int(slc[0][0]), int(slc[0][1])
            if start != 0 or stop != local_len:
                return None

        try:
            tp_rank = int(loc.get("tp_rank"))
        except (TypeError, ValueError):
            return None

        tp_size = full_tensor.shape[0] // local_len
        if tp_rank < 0 or tp_rank >= tp_size:
            return None

        return full_tensor.narrow(0, tp_rank * local_len, local_len)

    def _make_local_engine(self):
        """Construct the local Mooncake TransferEngine."""
        try:
            # Lazy-import: Mooncake is an optional dep; only pulled when the
            # P2P backend is actually selected.
            from mooncake.engine import TransferEngine  # noqa: PLC0415
        except ImportError as e:
            logger.error(
                "[P2P] mooncake-transfer-engine is not installed. "
                "Install it (see https://kvcache-ai.github.io/Mooncake/getting_started/build.html) "
                "or fall back to sync_inference_method='nccl_broadcast'."
            )
            logger.error(f"[P2P] underlying ImportError: {e}")
            return None

        # Reuse SGLang's Python wrapper so the engine init/handshake is
        # identical to the receiver side. We don't depend on SGLang at
        # runtime in xorl, but if the package is available locally we use
        # it. Otherwise fall back to constructing TransferEngine directly.
        hostname = self._hostname or _resolve_local_hostname()
        self._resolved_hostname = hostname
        logger.info(
            "[P2P] local Mooncake endpoint hostname=%s gpu_id=%s ib_device=%s",
            hostname,
            self._gpu_id,
            self._ib_device or "",
        )
        try:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (  # noqa: PLC0415
                MooncakeTransferEngine,
            )

            engine = MooncakeTransferEngine(
                hostname=hostname,
                gpu_id=self._gpu_id,
                ib_device=self._ib_device,
            )
            return engine
        except ImportError:
            logger.info(
                "[P2P] sglang.srt.distributed.device_communicators.mooncake_transfer_engine "
                "is not importable; using mooncake.engine.TransferEngine directly."
            )
            try:
                return _DirectMooncakeTransferEngine(
                    TransferEngine,
                    hostname=hostname,
                    gpu_id=self._gpu_id,
                    ib_device=self._ib_device,
                )
            except Exception as e:
                logger.error(f"[P2P] Failed to initialize direct Mooncake TransferEngine: {e}")
                return None
        except Exception as e:
            logger.error(f"[P2P] Failed to initialize local MooncakeTransferEngine: {e}")
            return None


def _resolve_local_hostname() -> str:
    """Return a routable host:port string for this rank.

    Mooncake's handshake binds on this hostname; it must be reachable from
    the SGLang receiver.
    """
    explicit = os.environ.get("XORL_P2P_HOSTNAME")
    if explicit and explicit.strip():
        return explicit.strip()

    def _routable_ipv4(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        try:
            addr = ipaddress.ip_address(value.strip())
        except ValueError:
            return None
        if addr.version != 4 or addr.is_loopback or addr.is_unspecified:
            return None
        return str(addr)

    for env_name in ("POD_IP", "HOST_IP", "HOSTNAME_IP"):
        if ip := _routable_ipv4(os.environ.get(env_name)):
            return ip

    # Kubernetes pod hostnames are not necessarily resolvable from peer pods.
    # Prefer the local IP address that socket resolves for this pod, matching
    # the SGLang receiver's advertised session ids.
    try:
        if ip := _routable_ipv4(socket.gethostbyname(socket.gethostname())):
            return ip
    except Exception:
        pass
    try:
        fqdn = socket.getfqdn()
        if ip := _routable_ipv4(socket.gethostbyname(fqdn)):
            return ip
    except Exception:
        pass
    return socket.gethostname()
