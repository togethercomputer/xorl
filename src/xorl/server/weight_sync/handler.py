"""
Weight Sync Handler — manages weight synchronization with inference endpoints.

Entry point: handle_sync_inference_weights (called on all training ranks).

Flow (nccl_broadcast, streaming per-layer):
1. Rank 0: health-check → create NCCL group → pause inference
2. For each PP stage (sequential; non-PP has only stage 0):
   a. Per FSDP module in this stage (intra-stage collective):
      - All stage ranks: unshard() → QLoRA collective ops → reshard()
      - Stage leader extracts params to bf16 buffer
   b. Stage 0: rank 0 quantizes + broadcasts to SGLang directly
   c. Stage 1+: stage leader sends bf16 buffer to rank 0 via per-module
      NCCL broadcast on pp_group (metadata via pickle, tensors via NCCL).
      Rank 0 quantizes and broadcasts to SGLang.
   d. Barrier between stages
3. Rank 0: resume inference → destroy NCCL group
4. All ranks: final barrier

With PP, stages are processed sequentially so rank 0 can stream each stage's
params to SGLang.  FSDP shard groups are independent per stage.  Inter-stage
transfer uses pp_group (one rank per stage at the same dp_shard position).
Only metadata (names/shapes) is pickled; the actual tensor data is transferred
as a flat bf16 NCCL broadcast — same speed as intra-stage all-gather.

This keeps GPU memory to ~1-2 layers (one unsharding + one broadcasting).
"""

import atexit
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp._fully_shard import FSDPModule

from xorl.distributed.parallel_state import get_parallel_state
from xorl.lora.modules.base import LoraModule
from xorl.lora.modules.linear import LoraLinear
from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.layers.moe.lora import MoEExpertsLoRA
from xorl.models.transformers.qwen3_5_shared import (
    has_linear_attention_layers,
    remap_linear_attention_params_for_inference,
)
from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.moe_experts import QLoRAMoeExperts
from xorl.server.protocol.operations import SyncWeightsData
from xorl.server.weight_sync.backends import TransportConfig, create_backend
from xorl.server.weight_sync.backends.base import EndpointConfig
from xorl.server.weight_sync.endpoint_manager import EndpointManager


logger = logging.getLogger(__name__)

_DEFAULT_MOE_BUCKET_BYTES = 256 * 1024 * 1024
_DEFAULT_P2P_MOE_BUCKET_BYTES = 2 * 1024 * 1024 * 1024


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    return value


def _moe_bucket_size_bytes(sync_method: str) -> int:
    """Default MoE bucket sizing is backend-specific; the env var remains an explicit override."""
    default = _DEFAULT_P2P_MOE_BUCKET_BYTES if sync_method == "p2p" else _DEFAULT_MOE_BUCKET_BYTES
    return _env_int("XORL_WEIGHT_SYNC_BUCKET_BYTES", default)


def _prod(shape) -> int:
    """Product of a shape tuple."""
    r = 1
    for d in shape:
        r *= d
    return r


def _p2p_local_rank(rank: int) -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", ""))
    except ValueError:
        if torch.cuda.is_available():
            return rank % max(torch.cuda.device_count(), 1)
        return rank


def _parse_p2p_gpu_to_ib_map(raw: str) -> Dict[str, str]:
    """Parse ``0=mlx5_2,1=mlx5_3`` or ``0:mlx5_2;1:mlx5_3``."""
    mapping: Dict[str, str] = {}
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        sep = "=" if "=" in item else ":"
        if sep not in item:
            logger.warning("Ignoring malformed P2P_TRAINER_GPU_TO_IB_DEVICE_MAP entry %r", item)
            continue
        gpu_idx, ib_device = (part.strip() for part in item.split(sep, 1))
        if gpu_idx and ib_device:
            mapping[gpu_idx] = ib_device
        else:
            logger.warning("Ignoring malformed P2P_TRAINER_GPU_TO_IB_DEVICE_MAP entry %r", item)
    return mapping


def _visible_physical_gpu_indices() -> List[str]:
    """Return physical GPU indices in local-rank order when the launcher provides them."""
    for env_name in ("P2P_TRAINER_VISIBLE_GPU_INDICES", "SELECTED_GPU_INDICES"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            return [idx.strip() for idx in raw.split(",") if idx.strip()]

    # CUDA_VISIBLE_DEVICES is only useful here when it is index-based. In
    # Kubernetes it is commonly UUID-based, so callers should provide
    # P2P_TRAINER_VISIBLE_GPU_INDICES after their dynamic GPU selection.
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        indices = [idx.strip() for idx in raw.split(",") if idx.strip()]
        if indices and all(idx.isdigit() for idx in indices):
            return indices

    return []


def _select_p2p_ib_device(rank: int, world_size: int) -> Optional[str]:
    """Return the Mooncake HCA hint for this trainer rank, if configured."""
    per_rank = os.environ.get("P2P_TRAINER_IB_DEVICES_PER_RANK", "").strip()
    if per_rank:
        devices = [d.strip() for d in per_rank.split(";")]
        local_rank = _p2p_local_rank(rank)
        if len(devices) >= world_size and 0 <= rank < len(devices):
            return devices[rank] or None
        if 0 <= local_rank < len(devices):
            return devices[local_rank] or None
        if 0 <= rank < len(devices):
            return devices[rank] or None
        logger.warning(
            "P2P_TRAINER_IB_DEVICES_PER_RANK has %d entries but no entry for "
            "rank=%d local_rank=%d; falling back to P2P_TRAINER_IB_DEVICE/auto-discovery",
            len(devices),
            rank,
            local_rank,
        )

    gpu_to_ib = _parse_p2p_gpu_to_ib_map(os.environ.get("P2P_TRAINER_GPU_TO_IB_DEVICE_MAP", "").strip())
    if gpu_to_ib:
        local_rank = _p2p_local_rank(rank)
        physical_gpu_indices = _visible_physical_gpu_indices()
        physical_gpu_idx = None
        if 0 <= local_rank < len(physical_gpu_indices):
            physical_gpu_idx = physical_gpu_indices[local_rank]
        elif str(local_rank) in gpu_to_ib:
            physical_gpu_idx = str(local_rank)

        if physical_gpu_idx is not None:
            ib_device = gpu_to_ib.get(physical_gpu_idx)
            if ib_device:
                return ib_device
            logger.warning(
                "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP has no entry for physical GPU %s "
                "(rank=%d local_rank=%d); falling back to P2P_TRAINER_IB_DEVICE/auto-discovery",
                physical_gpu_idx,
                rank,
                local_rank,
            )
        else:
            logger.warning(
                "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP is set, but local_rank=%d cannot be mapped "
                "to a physical GPU index; set P2P_TRAINER_VISIBLE_GPU_INDICES when "
                "CUDA_VISIBLE_DEVICES contains GPU UUIDs",
                local_rank,
            )

    device = os.environ.get("P2P_TRAINER_IB_DEVICE", "").strip()
    return device or None


def _safe_abort_token(value: Optional[str]) -> str:
    raw = str(value) if value else "none"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return safe[:160] or "none"


# ---------------------------------------------------------------------------
# Trainer-side P2P backend cache
#
# Constructing the Mooncake TransferEngine + allocating + Mooncake-registering
# the ~8.6 GB of CPU pinned scratch pools costs ~2-3 s on iter 1 (cold) of
# every sync session. Caching the backend across syncs in the same process
# amortizes that cost — second and subsequent syncs reuse the same engine
# and pools and only re-run the per-sync prepare RPC.
#
# Set ``XORL_P2P_BACKEND_CACHE=0`` to disable.
# ---------------------------------------------------------------------------
_cached_p2p_backend: Optional[Any] = None
_cached_backend_key: Optional[Tuple[Any, ...]] = None


def _atexit_destroy_cached_backend() -> None:
    global _cached_p2p_backend, _cached_backend_key
    if _cached_p2p_backend is not None:
        try:
            _cached_p2p_backend.destroy(complete_receiver=False)
        except Exception:
            pass
        _cached_p2p_backend = None
        _cached_backend_key = None


atexit.register(_atexit_destroy_cached_backend)


class WeightSyncHandler:
    """Handles weight synchronization between training and inference endpoints."""

    def __init__(self, rank: int, world_size: int, trainer) -> None:
        self.rank = rank
        self.world_size = world_size
        self.trainer = trainer
        # Per-sync MoE bucket accumulator. When
        # XORL_WEIGHT_SYNC_BATCH_MOE=1, _direct_ep_transfer_experts
        # appends here instead of flushing at end-of-call. Caller flushes
        # the leftover via _flush_pending_moe_bucket() after the MoE
        # loop completes.
        self._pending_moe_bucket: List[Tuple[str, torch.Tensor]] = []
        self._pending_moe_bucket_bytes: int = 0
        self._pending_moe_cpu_workspace_records: List[Tuple[str, Tuple[Any, ...], int]] = []
        self._fp8_cpu_workspaces: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _sync_abort_path(self, group_name: str, weight_version: Optional[str]) -> str:
        abort_dir = os.environ.get("XORL_WEIGHT_SYNC_ABORT_DIR", "").strip()
        if not abort_dir:
            train_config = getattr(self.trainer, "train_config", {}) or {}
            if isinstance(train_config, dict):
                abort_dir = str(train_config.get("output_dir") or "")
        if not abort_dir:
            abort_dir = "/tmp"
        return os.path.join(
            abort_dir,
            f".xorl_weight_sync_abort_{_safe_abort_token(group_name)}_{_safe_abort_token(weight_version)}",
        )

    def _clear_sync_abort(self, abort_path: str) -> None:
        try:
            os.remove(abort_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("Rank %d: [WeightSync] failed to clear abort marker %s: %s", self.rank, abort_path, e)

    def _mark_sync_abort(self, abort_path: str, err: Exception) -> None:
        try:
            abort_dir = os.path.dirname(abort_path)
            if abort_dir:
                os.makedirs(abort_dir, exist_ok=True)
            with open(abort_path, "w", encoding="utf-8") as f:
                f.write(f"rank={self.rank}: {err}\n")
        except Exception as marker_err:
            logger.warning(
                "Rank %d: [WeightSync] failed to write abort marker %s: %s",
                self.rank,
                abort_path,
                marker_err,
            )

    def _raise_if_sync_aborted(self, abort_path: str) -> None:
        try:
            with open(abort_path, encoding="utf-8") as f:
                reason = f.read().strip()
        except FileNotFoundError:
            return
        except Exception as e:
            logger.debug("Rank %d: [WeightSync] failed to read abort marker %s: %s", self.rank, abort_path, e)
            return

        raise RuntimeError(f"Weight sync aborted by peer rank: {reason or abort_path}")

    def _build_p2p_rank_summary(
        self,
        backend: Any,
        *,
        is_sender: bool,
        transfer_wall_s: float,
        total_bytes: int,
        num_parameters: int,
        num_buckets: int,
        ib_device: Optional[str],
        phase_s: Dict[str, float],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "rank": self.rank,
            "local_rank": _p2p_local_rank(self.rank),
            "is_sender": is_sender,
            "has_transfers": False,
            "transfer_wall_s": transfer_wall_s,
            "total_bytes": int(total_bytes),
            "num_parameters": int(num_parameters),
            "num_buckets": int(num_buckets),
            "ib_device": ib_device,
            "phase_s": phase_s,
        }
        if is_sender and hasattr(backend, "stats_summary"):
            try:
                backend_summary = backend.stats_summary()
                summary["backend"] = backend_summary
                summary["has_transfers"] = float(backend_summary.get("total_bytes", 0.0)) > 0.0
                backend_main_thread_s = float(backend_summary.get("main_thread_s", 0.0))
                summary["backend_main_thread_s"] = backend_main_thread_s
                summary["trainer_overhead_s"] = max(
                    0.0,
                    transfer_wall_s - backend_main_thread_s,
                )
            except Exception as e:
                summary["backend_error"] = str(e)
        return summary

    def _gather_p2p_rank_summaries(self, local_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return [local_summary]
        gathered: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_summary)
        return [item for item in gathered if isinstance(item, dict)]

    def _gather_p2p_transfer_statuses(self, local_error: Optional[Exception]) -> List[Dict[str, Any]]:
        local_status: Dict[str, Any] = {
            "rank": self.rank,
            "ok": local_error is None,
        }
        if local_error is not None:
            local_status["error"] = f"{type(local_error).__name__}: {local_error}"

        if self.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return [local_status]
        gathered: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_status)
        return [item for item in gathered if isinstance(item, dict)]

    @staticmethod
    def _summary_counter(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    @classmethod
    def _aggregate_p2p_transfer_totals(
        cls,
        p2p_rank_summaries: List[Dict[str, Any]],
        *,
        total_bytes: int,
        num_parameters: int,
        num_buckets: int,
    ) -> Tuple[int, int, int]:
        saw_rank_counters = False
        aggregate_bytes = 0
        aggregate_parameters = 0
        aggregate_buckets = 0

        for summary in p2p_rank_summaries:
            if not isinstance(summary, dict):
                continue
            if not any(key in summary for key in ("total_bytes", "num_parameters", "num_buckets")):
                continue
            saw_rank_counters = True
            aggregate_bytes += cls._summary_counter(summary.get("total_bytes", 0))
            aggregate_parameters += cls._summary_counter(summary.get("num_parameters", 0))
            aggregate_buckets += cls._summary_counter(summary.get("num_buckets", 0))

        if not saw_rank_counters:
            return total_bytes, num_parameters, num_buckets
        return aggregate_bytes, aggregate_parameters, aggregate_buckets

    @staticmethod
    def _add_rank_timing_breakdown(
        timing_breakdown: Dict[str, float],
        p2p_rank_summaries: List[Dict[str, Any]],
    ) -> None:
        sender_transfer_times = [
            float(summary["transfer_wall_s"])
            for summary in p2p_rank_summaries
            if summary.get("has_transfers") and isinstance(summary.get("transfer_wall_s"), int | float)
        ]
        if not sender_transfer_times:
            return
        max_transfer_s = max(sender_transfer_times)
        min_transfer_s = min(sender_transfer_times)
        timing_breakdown["max_rank_transfer_s"] = max_transfer_s
        timing_breakdown["min_rank_transfer_s"] = min_transfer_s
        timing_breakdown["rank_transfer_spread_s"] = max_transfer_s - min_transfer_s

    @staticmethod
    def _moe_runtime_lora_views(
        mod: MoEExpertsLoRA | QLoRAMoeExperts,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return the active LoRA slices and scaling for runtime-rank MoE modules."""
        active_rank = getattr(mod, "active_r", None)
        if active_rank is None:
            return lora_A, lora_B, float(mod.scaling)

        active_rank = int(active_rank)
        if active_rank <= 0:
            raise ValueError(f"Active LoRA rank must be positive, got {active_rank}")
        if active_rank > lora_A.shape[-1] or active_rank > lora_B.shape[1]:
            raise ValueError(
                f"Active LoRA rank {active_rank} exceeds available MoE LoRA slices: "
                f"A={tuple(lora_A.shape)}, B={tuple(lora_B.shape)}"
            )

        scaling_fn = getattr(mod, "_active_scaling", None)
        scaling = float(scaling_fn()) if callable(scaling_fn) else float(mod.scaling)
        return lora_A[..., :active_rank], lora_B[:, :active_rank, ...], scaling

    @classmethod
    def _compute_moe_lora_delta(
        cls,
        mod: MoEExpertsLoRA | QLoRAMoeExperts,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        *,
        expert_idx: int,
    ) -> torch.Tensor:
        """Compute one expert's active LoRA delta in GKN format."""
        active_lora_A, active_lora_B, scaling = cls._moe_runtime_lora_views(mod, lora_A, lora_B)
        a_idx = min(expert_idx, active_lora_A.shape[0] - 1)
        b_idx = min(expert_idx, active_lora_B.shape[0] - 1)
        return (active_lora_A[a_idx] @ active_lora_B[b_idx]) * scaling

    # ========================================================================
    # Main entry point
    # ========================================================================

    async def handle_sync_inference_weights(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle sync inference weights request (all ranks participate).

        The ``sync_method`` field selects the transport backend.  Currently
        supported: ``"nccl_broadcast"`` and ``"p2p"``.  New backends (RDMA, multi-rank NCCL,
        etc.) can be added by implementing :class:`WeightTransportBackend` and
        registering in :func:`backends.create_backend`.
        """
        logger.info(f"Rank {self.rank}: [WeightSync] Starting sync_inference_weights")

        p: SyncWeightsData = command_dict.get("payload", SyncWeightsData())

        endpoints = p.endpoints
        master_address = p.master_address
        master_port = p.master_port
        group_name = p.group_name
        buffer_size_mb = p.buffer_size_mb
        sync_method = p.sync_method
        flush_cache = p.flush_cache
        pause_mode = p.pause_mode
        weight_version = p.weight_version
        quantization = p.quantization

        logger.info(
            f"Rank {self.rank}: [WeightSync] sync_method={sync_method}, "
            f"endpoints={len(endpoints)}, flush_cache={flush_cache}, "
            f"weight_version={weight_version}, quantization={quantization}"
        )

        try:
            result = self._sync_weights(
                endpoints=endpoints,
                master_address=master_address,
                master_port=master_port,
                group_name=group_name,
                buffer_size_mb=buffer_size_mb,
                sync_method=sync_method,
                flush_cache=flush_cache,
                pause_mode=pause_mode,
                weight_version=weight_version,
                quantization=quantization,
            )

            return result

        except Exception as e:
            logger.error(f"Rank {self.rank}: sync_inference_weights failed: {e}", exc_info=True)
            return {"success": False, "message": f"Weight sync failed: {str(e)}"}

    # ========================================================================
    # Streaming per-layer weight sync (backend-agnostic)
    # ========================================================================

    def _sync_weights(
        self,
        endpoints: List[Dict[str, Any]],
        master_address: str,
        master_port: int,
        group_name: str,
        buffer_size_mb: int,
        sync_method: str,
        flush_cache: bool,
        pause_mode: str,
        weight_version: Optional[str],
        quantization: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Streaming per-layer weight sync using a pluggable transport backend.

        Instead of materializing the entire model on CPU, this:
        1. Unshards one FSDP module at a time (all-gather to GPU)
        2. Clones params to a buffer on GPU
        3. Reshards to free memory
        4. Sends the buffer via the transport backend
        5. Pipelines: unshard(N+1) overlaps with transfer(N)
        """

        sync_start_time = time.perf_counter()
        timing_breakdown: Dict[str, float] = {}
        model = self.trainer.model
        device = f"cuda:{self.trainer.local_rank}"
        abort_path = self._sync_abort_path(group_name, weight_version) if sync_method == "p2p" else ""
        if abort_path:
            self._clear_sync_abort(abort_path)

        # ------------------------------------------------------------------
        # Step 1: Preparation (all ranks)
        # ------------------------------------------------------------------
        logger.info(f"Rank {self.rank}: [WeightSync] Clearing GPU memory...")
        if hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
            try:
                self.trainer.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.trainer.optimizer.zero_grad()
        torch.cuda.empty_cache()

        if self.rank == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Rank {self.rank}: [WeightSync] GPU memory: {allocated:.2f}GB allocated")

        # ------------------------------------------------------------------
        # Step 2: Discover FSDP modules (all ranks, no communication)
        # ------------------------------------------------------------------
        root_module, layer_modules = self._get_fsdp_modules(model)
        total_modules = (1 if root_module else 0) + len(layer_modules)
        logger.info(
            f"Rank {self.rank}: [WeightSync] Found {total_modules} FSDP modules "
            f"(root={'yes' if root_module else 'no'}, layers={len(layer_modules)})"
        )

        # ------------------------------------------------------------------
        # Step 3: Create backend + endpoint manager, initialize on sender ranks
        # ------------------------------------------------------------------

        # When sync_method=="p2p" AND EP is active on the trainer, default to
        # the multi-sender direct-EP path: each EP rank ships its own local
        # experts directly to the receiver instead of dist.gather'ing through
        # rank 0 then broadcasting. The default NCCL broadcast backend
        # ignores backend_config and stays single-sender.
        _ps_for_cfg = get_parallel_state()
        _backend_config: Dict[str, Any] = {}
        if sync_method == "p2p":
            local_rank = _p2p_local_rank(self.rank)
            _backend_config["gpu_id"] = local_rank
            _backend_config["flush_cache"] = flush_cache
            if weight_version is not None:
                _backend_config["weight_version"] = weight_version
            ib_device = _select_p2p_ib_device(self.rank, self.world_size)
            if ib_device:
                _backend_config["ib_device"] = ib_device
            logger.info(
                "Rank %d: [WeightSync] P2P Mooncake trainer binding: gpu_id=%s, ib_device=%s",
                self.rank,
                local_rank,
                ib_device or "<auto>",
            )
        if sync_method == "p2p" and _ps_for_cfg.ep_enabled and _ps_for_cfg.ep_size > 1:
            _backend_config["direct_ep_transfer"] = True
            # The P2P backend reads world_size out of backend_config; if
            # we don't pass it, it defaults to 1 and sender_ranks
            # silently collapses to {0} so non-rank-0 trainers route
            # back to the gather-and-broadcast fallback.
            _backend_config["world_size"] = self.world_size
            _backend_config["rank_index"] = self.rank

        transport_cfg = TransportConfig(
            endpoints=[
                EndpointConfig(
                    host=ep["host"],
                    port=ep["port"],
                    world_size=ep.get("world_size", 1),
                )
                for ep in endpoints
            ],
            master_address=master_address,
            master_port=master_port,
            group_name=group_name,
            buffer_size_mb=buffer_size_mb,
            device=device,
            training_world_size=self.world_size,
            training_rank=self.rank,
            backend_config=_backend_config,
        )

        # Trainer-side backend cache. The expensive bits — Mooncake
        # TransferEngine handshake + ~8.6 GB CPU pinned pool registration —
        # are amortized across syncs when (sync_method, endpoint set,
        # group_name, master addr) all match the prior call's. The cache
        # is module-level so it survives across handler instances within
        # the same process.
        global _cached_p2p_backend, _cached_backend_key
        cache_enabled = os.environ.get("XORL_P2P_BACKEND_CACHE", "1") == "1" and sync_method == "p2p"
        backend_key: Optional[Tuple[Any, ...]] = None
        if cache_enabled:
            backend_key = (
                sync_method,
                tuple((ep["host"], ep["port"], ep.get("world_size", 1)) for ep in endpoints),
                group_name,
                master_address,
                master_port,
                buffer_size_mb,
                self.world_size,
                self.rank,
                tuple(
                    sorted(
                        (k, v)
                        for k, v in (_backend_config or {}).items()
                        if k not in {"flush_cache", "weight_version"} and isinstance(v, (str, int, bool, float))
                    )
                ),
            )
        if (
            cache_enabled
            and _cached_p2p_backend is not None
            and _cached_backend_key == backend_key
            and getattr(_cached_p2p_backend, "is_alive", False)
        ):
            backend = _cached_p2p_backend
            # Refresh the config in case per-sync params (flush_cache,
            # weight_version) differ from the prior call. The cache-key
            # check above guarantees the structural fields (endpoints,
            # group_name, world_size) match.
            backend.config = transport_cfg
            logger.info(f"Rank {self.rank}: [WeightSync] Reusing cached P2P backend (skips engine + scratch-pool init)")
        else:
            if _cached_p2p_backend is not None:
                try:
                    _cached_p2p_backend.destroy(complete_receiver=False)
                except Exception as e:
                    logger.warning(f"[WeightSync] failed to destroy stale cached backend: {e}")
                _cached_p2p_backend = None
                _cached_backend_key = None
            backend = create_backend(sync_method, transport_cfg)
        _is_sender = self.rank in backend.sender_ranks

        # Endpoint management lives on rank 0 (coordinator).  Future multi-rank
        # backends still designate one rank for HTTP pause/resume coordination.
        endpoint_mgr = EndpointManager(endpoints) if self.rank == 0 else None

        if self.rank == 0:
            if not endpoints:
                return {"success": False, "message": "No endpoints provided"}
            t_health = time.perf_counter()
            endpoint_mgr.health_check()
            timing_breakdown["health_check_s"] = time.perf_counter() - t_health

        # Backend init: all sender ranks participate (collective for NCCL).
        if _is_sender:
            logger.info(f"Rank {self.rank}: [WeightSync] Initializing {sync_method} backend...")
            t_init = time.perf_counter()
            if not backend.initialize():
                return {
                    "success": False,
                    "message": f"Failed to initialize {sync_method} backend",
                }
            timing_breakdown["backend_init_s"] = time.perf_counter() - t_init
            logger.info(f"Rank {self.rank}: [WeightSync] Backend initialized")

        # Pause inference: coordinator only (after backend init).
        if self.rank == 0:
            logger.info(f"Rank {self.rank}: [WeightSync] Pausing inference (mode={pause_mode})...")
            t_pause = time.perf_counter()
            pause_results, all_paused = endpoint_mgr.pause(pause_mode)
            timing_breakdown["pause_s"] = time.perf_counter() - t_pause
            if not all_paused:
                endpoint_mgr.resume()
                if _is_sender:
                    backend.destroy(complete_receiver=False)
                    # Pause failure invalidates the cache; next sync starts fresh.
                    _cached_p2p_backend = None
                    _cached_backend_key = None
                return {
                    "success": False,
                    "message": f"Failed to pause inference endpoints: {pause_results}",
                }

        # ------------------------------------------------------------------
        # Step 4: Stream weights — per-module unshard + broadcast pipeline
        # ------------------------------------------------------------------
        start_time = time.perf_counter()
        total_bytes = 0
        total_params = 0
        num_buckets = 0
        rank_phase_s: Dict[str, float] = {}

        def _add_rank_phase(name: str, start: float) -> None:
            rank_phase_s[name] = rank_phase_s.get(name, 0.0) + (time.perf_counter() - start)

        # Cross-layer MoE batching. When on, _direct_ep_transfer_experts
        # appends to the handler-level _pending_moe_bucket instead of flushing
        # at end-of-call; we ship the leftover once after the module loop.
        # Default off — flip via XORL_WEIGHT_SYNC_BATCH_MOE=1.
        batch_moe = os.environ.get("XORL_WEIGHT_SYNC_BATCH_MOE", "0") == "1"
        moe_bucket_size_bytes = _moe_bucket_size_bytes(sync_method)
        # Reset cross-sync state in case a prior sync raised mid-flush.
        self._pending_moe_bucket = []
        self._pending_moe_bucket_bytes = 0
        self._reset_fp8_cpu_workspace_usage()

        # Build ordered list of FSDP modules to process
        modules_to_sync: List[Tuple[str, FSDPModule]] = []
        if root_module is not None:
            modules_to_sync.append(("(root)", root_module))
        modules_to_sync.extend(layer_modules)

        # Detect EP mode
        _ps = get_parallel_state()
        _ep_enabled = _ps.ep_enabled and _ps.ep_size > 1

        # Detect PP mode (Pipeline Parallelism)
        # With PP, each stage has an independent FSDP shard group.  We process
        # stages sequentially so that rank 0 can stream each stage's params to
        # SGLang.  For remote stages (>0), the stage leader sends each module's
        # params to rank 0 via a per-module NCCL broadcast on pp_group, keeping
        # memory bounded to one module at a time.
        _pp_enabled = _ps.pp_enabled
        _pp_rank = _ps.pp_rank if _pp_enabled else 0
        _pp_size = _ps.pp_size if _pp_enabled else 1

        try:
            for pp_stage in range(_pp_size):
                _is_my_stage = _pp_rank == pp_stage
                _is_remote = _pp_enabled and pp_stage > 0
                # Stage leader: the dp_shard_rank==0 rank within this stage
                _stage_leader = _ps.dp_shard_rank == 0 if _pp_enabled else (self.rank == 0)

                stage_modules = modules_to_sync if _is_my_stage else []

                # For remote stages: tell rank 0 how many modules to expect
                pp_grp = None
                stage_src = 0
                num_stage_modules = len(stage_modules)
                if _is_remote and _ps.dp_shard_rank == 0:
                    pp_grp = _ps.pp_group
                    stage_src = dist.get_global_rank(pp_grp, pp_stage)
                    obj = [num_stage_modules] if _is_my_stage else [None]
                    dist.broadcast_object_list(obj, src=stage_src, group=pp_grp)
                    num_stage_modules = obj[0]

                if _is_my_stage:
                    logger.info(
                        f"Rank {self.rank}: [WeightSync] Processing PP stage {pp_stage} "
                        f"({num_stage_modules} modules, remote={_is_remote})"
                    )

                # Optional fast path: unshard ALL FSDP modules up front so
                # the per-module loop doesn't pay the FSDP allgather barrier
                # latency (~50-100 ms × 50 modules = 2.5-5 s of barrier time
                # collapsed to one batched pass). Memory cost: each rank
                # holds the full model in addition to the sharded copy
                # (~30 GB extra on Qwen3-30B-A3B at FSDP=8). Gate behind
                # XORL_WEIGHT_SYNC_PRE_UNSHARD=1; off by default.
                _pre_unshard = os.environ.get("XORL_WEIGHT_SYNC_PRE_UNSHARD", "0") == "1" and _is_my_stage
                if _pre_unshard:
                    t_pre = time.perf_counter()
                    for _, _fsdp_mod in stage_modules:
                        _fsdp_mod.unshard()
                    # No torch.cuda.synchronize() — unshards queue on
                    # the NCCL stream and the first GPU op in
                    # _extract_params_for_sync will naturally wait via
                    # stream ordering. Skipping the sync lets the
                    # streaming loop start ~1-2 s earlier on rank 0
                    # (which had ~2s of launch latency relative to
                    # other ranks in baseline measurements).
                    logger.info(
                        f"Rank {self.rank}: [WeightSync] Pre-unshard launch done: "
                        f"{len(stage_modules)} modules queued in "
                        f"{(time.perf_counter() - t_pre) * 1000:.1f} ms "
                        f"(allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB)"
                    )

                # XORL_WEIGHT_SYNC_TIMINGS=1 → emit a per-module phase
                # breakdown on rank 0 (unshard / qlora / ep_collect /
                # extract / unfuse / broadcast / direct_ep). Pinpoints
                # which trainer-side phase dominates the streaming wall.
                _ws_timings = os.environ.get("XORL_WEIGHT_SYNC_TIMINGS", "0") == "1"

                for mod_idx in range(num_stage_modules):
                    if abort_path:
                        self._raise_if_sync_aborted(abort_path)
                    is_last_overall = mod_idx == num_stage_modules - 1 and pp_stage == _pp_size - 1

                    # ── FSDP ops (only ranks owning this stage) ──────────
                    current_buffer = None
                    moe_contexts = []
                    ep_moe_contexts = []
                    _t0 = time.perf_counter() if _ws_timings else 0.0
                    _t_unshard = _t_qlora = _t_ep_collect = _t_extract = _t0
                    _t_unfuse = _t_broadcast = _t_direct_ep = _t0

                    if _is_my_stage:
                        mod_name, fsdp_mod = stage_modules[mod_idx]

                        if not _pre_unshard:
                            t_phase = time.perf_counter()
                            fsdp_mod.unshard()
                            _add_rank_phase("unshard_s", t_phase)

                        if _ws_timings:
                            _t_unshard = time.perf_counter()
                        t_phase = time.perf_counter()
                        qlora_linear_buffer, moe_contexts = self._qlora_collective_ops(
                            fsdp_mod,
                            mod_name,
                            collect_results=_stage_leader,
                        )
                        _add_rank_phase("qlora_s", t_phase)
                        if _ws_timings:
                            _t_qlora = time.perf_counter()

                        if _ep_enabled:
                            t_phase = time.perf_counter()
                            ep_moe_contexts = self._collect_ep_moe_data(
                                fsdp_mod,
                                mod_name,
                                _ps,
                                skip_clone=_pre_unshard,
                                phase_s=rank_phase_s,
                            )
                            _add_rank_phase("ep_collect_s", t_phase)
                        if _ws_timings:
                            _t_ep_collect = time.perf_counter()

                        # EP MoE prefixes to skip in extraction
                        ep_moe_prefixes = set()
                        for ctx in ep_moe_contexts:
                            p = ctx["prefix"]
                            if mod_name != "(root)":
                                if p == mod_name:
                                    ep_moe_prefixes.add("")
                                elif p.startswith(mod_name + "."):
                                    ep_moe_prefixes.add(p[len(mod_name) + 1 :])
                            else:
                                ep_moe_prefixes.add(p)

                        if _stage_leader:
                            t_phase = time.perf_counter()
                            if ep_moe_prefixes:
                                logger.info(
                                    f"Rank {self.rank}: [WeightSync] ep_moe_prefixes={ep_moe_prefixes} for {mod_name}"
                                )
                            current_buffer = self._extract_params_for_sync(
                                fsdp_mod,
                                mod_name,
                                DTensor,
                                skip_moe_prefixes=ep_moe_prefixes,
                            )
                            current_buffer.extend(qlora_linear_buffer)
                            _add_rank_phase("extract_s", t_phase)
                        del qlora_linear_buffer
                        if _ws_timings:
                            _t_extract = time.perf_counter()

                        if not _pre_unshard:
                            t_phase = time.perf_counter()
                            fsdp_mod.reshard()
                            _add_rank_phase("reshard_s", t_phase)

                    # ── Transfer / broadcast to SGLang ───────────────────
                    if not _is_remote:
                        # Stage 0: sender rank(s) broadcast directly to SGLang
                        if _is_sender and current_buffer:
                            t_phase = time.perf_counter()
                            current_buffer = self._unfuse_for_inference(
                                current_buffer,
                                model,
                            )
                            if quantization and quantization.get("quant_method") == "fp8":
                                current_buffer = self._quantize_buffer_for_fp8(
                                    current_buffer,
                                    quantization_config=quantization,
                                    target_device=self._fp8_quantization_target_device(backend),
                                    phase_s=rank_phase_s,
                                    phase_prefix="dense_fp8",
                                )
                            _add_rank_phase("unfuse_quantize_s", t_phase)
                            if _ws_timings:
                                _t_unfuse = time.perf_counter()
                            logger.info(f"Rank 0: [WeightSync] Module {mod_name}: {len(current_buffer)} params")
                            t_phase = time.perf_counter()
                            b, p = self._broadcast_buffer(
                                backend,
                                current_buffer,
                                flush_cache=(flush_cache and is_last_overall and not moe_contexts),
                                weight_version=weight_version if is_last_overall and not moe_contexts else None,
                            )
                            _add_rank_phase("broadcast_buffer_s", t_phase)
                            total_bytes += b
                            total_params += p
                            num_buckets += 1
                            del current_buffer
                            if _ws_timings:
                                _t_broadcast = time.perf_counter()

                        # Stage 0 MoE handling. With direct EP/PP transport
                        # (P2P + direct_ep_transfer=True), each EP rank ships
                        # its own local experts in parallel and skips the
                        # rank-0 dist.gather → broadcast funnel. The default
                        # NCCL path still does gather-and-broadcast.
                        if moe_contexts or ep_moe_contexts:
                            if _ep_enabled:
                                use_direct_ep = (
                                    backend.supports_direct_ep_transfer and self.rank in backend.sender_ranks
                                )
                                for ctx in moe_contexts + ep_moe_contexts:
                                    if use_direct_ep:
                                        # batch_moe defers the per-call
                                        # final flush so multiple layers'
                                        # MoE experts coalesce into one
                                        # large bucket (~2 GB instead of
                                        # ~302 MB). flush_cache and
                                        # weight_version migrate to the
                                        # post-loop _flush_pending_moe_bucket
                                        # call below.
                                        t_phase = time.perf_counter()
                                        b, p, n = self._direct_ep_transfer_experts(
                                            backend,
                                            ctx,
                                            flush_cache=(flush_cache and is_last_overall) and not batch_moe,
                                            weight_version=(
                                                weight_version if is_last_overall and not batch_moe else None
                                            ),
                                            bucket_size_bytes=moe_bucket_size_bytes,
                                            quantization=quantization,
                                            ps=_ps,
                                            defer_final_flush=batch_moe,
                                            phase_s=rank_phase_s,
                                        )
                                        _add_rank_phase("direct_ep_s", t_phase)
                                    else:
                                        t_phase = time.perf_counter()
                                        b, p, n = self._gather_and_broadcast_ep_moe_experts(
                                            backend,
                                            ctx,
                                            flush_cache=(flush_cache and is_last_overall),
                                            weight_version=weight_version if is_last_overall else None,
                                            bucket_size_bytes=moe_bucket_size_bytes,
                                            quantization=quantization,
                                            ps=_ps,
                                        )
                                        _add_rank_phase("gather_broadcast_ep_s", t_phase)
                                    total_bytes += b
                                    total_params += p
                                    num_buckets += n
                            elif _is_sender:
                                for ctx in moe_contexts:
                                    t_phase = time.perf_counter()
                                    b, p, n = self._broadcast_moe_experts_bucketed(
                                        backend,
                                        ctx,
                                        flush_cache=(flush_cache and is_last_overall),
                                        weight_version=weight_version if is_last_overall else None,
                                        bucket_size_bytes=moe_bucket_size_bytes,
                                        quantization=quantization,
                                    )
                                    _add_rank_phase("broadcast_moe_s", t_phase)
                                    total_bytes += b
                                    total_params += p
                                    num_buckets += n

                        if _ws_timings and _is_my_stage and self.rank == 0:
                            _t_direct_ep = time.perf_counter()
                            _mn = stage_modules[mod_idx][0] if mod_idx < len(stage_modules) else "?"
                            logger.info(
                                f"Rank 0: [WeightSync timing] {_mn}: "
                                f"unshard={(_t_unshard - _t0) * 1000:.0f}ms "
                                f"qlora={(_t_qlora - _t_unshard) * 1000:.0f}ms "
                                f"ep_collect={(_t_ep_collect - _t_qlora) * 1000:.0f}ms "
                                f"extract={(_t_extract - _t_ep_collect) * 1000:.0f}ms "
                                f"unfuse={(_t_unfuse - _t_extract) * 1000:.0f}ms "
                                f"broadcast={(_t_broadcast - _t_unfuse) * 1000:.0f}ms "
                                f"direct_ep={(_t_direct_ep - _t_broadcast) * 1000:.0f}ms "
                                f"total={(_t_direct_ep - _t0) * 1000:.0f}ms"
                            )
                    else:
                        # Remote stage: per-module NCCL transfer to rank 0
                        if _ps.dp_shard_rank == 0:
                            # Prepare send buffer on stage leader (bf16, not quantized)
                            send_buf = None
                            if _is_my_stage and _stage_leader:
                                send_buf = current_buffer if current_buffer else []
                                send_buf = self._unfuse_for_inference(send_buf, model)
                                # Include MoE experts in the same buffer
                                for ctx in moe_contexts:
                                    expert_items = self._compute_moe_experts_buffer(ctx)
                                    send_buf.extend(expert_items)

                            # NCCL transfer: metadata + flat bf16 tensor
                            received = self._pp_nccl_transfer_buffer(
                                send_buf if (_is_my_stage and _stage_leader) else None,
                                pp_grp,
                                stage_src,
                                device,
                            )

                            # Rank 0: quantize + broadcast to SGLang
                            if self.rank == 0 and received:
                                if quantization and quantization.get("quant_method") == "fp8":
                                    received = self._quantize_buffer_for_fp8(
                                        received,
                                        quantization_config=quantization,
                                        target_device=self._fp8_quantization_target_device(backend),
                                        phase_s=rank_phase_s,
                                        phase_prefix="pp_fp8",
                                    )
                                logger.info(
                                    f"Rank 0: [WeightSync] PP stage {pp_stage} module "
                                    f"{mod_idx}: {len(received)} params via NCCL"
                                )
                                b, p = self._broadcast_buffer(
                                    backend,
                                    received,
                                    flush_cache=(flush_cache and is_last_overall),
                                    weight_version=weight_version if is_last_overall else None,
                                )
                                total_bytes += b
                                total_params += p
                                num_buckets += 1
                                del received

                    if abort_path:
                        self._raise_if_sync_aborted(abort_path)

                # Pre-unshard mode: now that all transfers have been
                # submitted to the worker (transfer_bucket returns after
                # staging), re-shard the modules to free the ~30 GB of
                # extra GPU memory that's been holding the unsharded
                # weights. We do this BEFORE flush_pending_transfers so
                # the reshard work can happen on the compute stream
                # while RDMA reads from the CPU pinned pool on the NIC.
                if _pre_unshard:
                    t_re = time.perf_counter()
                    for _, _fsdp_mod in stage_modules:
                        _fsdp_mod.reshard()
                    logger.info(
                        f"Rank {self.rank}: [WeightSync] Post-streaming reshard "
                        f"in {(time.perf_counter() - t_re) * 1000:.1f} ms"
                    )

                # Barrier between PP stages (all ranks)
                if _pp_enabled:
                    dist.barrier()

            # Cross-layer MoE flush. Ship whatever's left in the
            # accumulator once, instead of per-layer. This is the LAST
            # transfer of the sync, so it carries flush_cache +
            # weight_version (if requested by the caller).
            if abort_path:
                self._raise_if_sync_aborted(abort_path)
            if batch_moe and _is_sender:
                t_phase = time.perf_counter()
                b, p, n = self._flush_pending_moe_bucket(
                    backend,
                    flush_cache=flush_cache,
                    weight_version=weight_version,
                    quantization=quantization,
                    bucket_size_bytes=moe_bucket_size_bytes,
                    phase_s=rank_phase_s,
                )
                _add_rank_phase("moe_final_flush_s", t_phase)
                total_bytes += b
                total_params += p
                num_buckets += n

            # Drain any async transfers (P2P backend submits Mooncake
            # work to a worker thread and returns from transfer_bucket
            # before bytes land). Must complete before the handler
            # resumes inference or the next request can read
            # partially-updated weights.
            pending_transfer_error: Optional[Exception] = None
            if abort_path:
                try:
                    self._raise_if_sync_aborted(abort_path)
                except Exception as abort_err:
                    pending_transfer_error = abort_err
            if _is_sender:
                t_phase = time.perf_counter()
                try:
                    if pending_transfer_error is None:
                        backend.flush_pending_transfers()
                except Exception as flush_err:
                    pending_transfer_error = flush_err
                    if abort_path:
                        self._mark_sync_abort(abort_path, flush_err)
                finally:
                    _add_rank_phase("flush_pending_s", t_phase)

            if sync_method == "p2p":
                transfer_statuses = self._gather_p2p_transfer_statuses(pending_transfer_error)
                failed_statuses = [status for status in transfer_statuses if not status.get("ok", False)]
                if failed_statuses:
                    if pending_transfer_error is not None:
                        raise pending_transfer_error
                    preview = "; ".join(
                        f"rank {status.get('rank')}: {status.get('error', 'unknown error')}"
                        for status in failed_statuses[:4]
                    )
                    if len(failed_statuses) > 4:
                        preview += f"; ... {len(failed_statuses) - 4} more"
                    raise RuntimeError(f"P2P transfer failed on peer rank(s): {preview}")
            elif pending_transfer_error is not None:
                raise pending_transfer_error

            transfer_time = time.perf_counter() - start_time
            timing_breakdown["transfer_s"] = transfer_time
            p2p_rank_summaries: List[Dict[str, Any]] = []
            if sync_method == "p2p":
                t_rank_summary = time.perf_counter()
                local_summary = self._build_p2p_rank_summary(
                    backend,
                    is_sender=_is_sender,
                    transfer_wall_s=transfer_time,
                    total_bytes=total_bytes,
                    num_parameters=total_params,
                    num_buckets=num_buckets,
                    ib_device=_backend_config.get("ib_device"),
                    phase_s=rank_phase_s,
                )
                p2p_rank_summaries = self._gather_p2p_rank_summaries(local_summary)
                total_bytes, total_params, num_buckets = self._aggregate_p2p_transfer_totals(
                    p2p_rank_summaries,
                    total_bytes=total_bytes,
                    num_parameters=total_params,
                    num_buckets=num_buckets,
                )
                if self.rank == 0:
                    timing_breakdown["rank_summary_gather_s"] = time.perf_counter() - t_rank_summary
                    self._add_rank_timing_breakdown(timing_breakdown, p2p_rank_summaries)

            # ------------------------------------------------------------------
            # Step 5: Resume inference, cleanup
            # ------------------------------------------------------------------
            if _is_sender:
                # Finalize receiver-side update before inference resumes.
                # For P2P this sends /complete_weights_update, where SGLang
                # applies weight_version, flush_cache, and post-processing.
                # If completion fails, fail closed and leave inference paused.
                if cache_enabled and backend_key is not None and hasattr(backend, "complete_sync"):
                    t_complete = time.perf_counter()
                    try:
                        backend.complete_sync()
                        _cached_p2p_backend = backend
                        _cached_backend_key = backend_key
                    except Exception as complete_err:
                        logger.warning(
                            f"Rank {self.rank}: [WeightSync] complete_sync failed; "
                            f"falling back to full destroy: {complete_err}"
                        )
                        try:
                            backend.destroy(complete_receiver=False)
                        except Exception:
                            pass
                        _cached_p2p_backend = None
                        _cached_backend_key = None
                        raise
                    finally:
                        timing_breakdown["complete_s"] = time.perf_counter() - t_complete
                else:
                    t_destroy = time.perf_counter()
                    backend.destroy()
                    timing_breakdown["backend_destroy_s"] = time.perf_counter() - t_destroy

            if self.rank == 0:
                throughput = (total_bytes / transfer_time / (1024**3)) if transfer_time > 0 else 0
                logger.info(
                    f"Rank {self.rank}: [WeightSync] Transfer complete: "
                    f"{transfer_time:.2f}s, {throughput:.2f} GB/s, "
                    f"{total_bytes / 1e9:.2f} GB, {total_params} params, "
                    f"{num_buckets} buckets"
                )
                t_resume = time.perf_counter()
                endpoint_mgr.resume()
                timing_breakdown["resume_s"] = time.perf_counter() - t_resume

            timing_breakdown["total_handler_s"] = time.perf_counter() - sync_start_time
            if self.rank == 0:
                ordered = ", ".join(f"{k}={v:.3f}s" for k, v in timing_breakdown.items())
                logger.info(f"Rank {self.rank}: [WeightSync] Timing breakdown: {ordered}")

            return {
                "success": True,
                "message": f"Synced {total_params} params to {len(endpoints)} endpoint(s)",
                "transfer_time": transfer_time,
                "total_bytes": total_bytes,
                "num_parameters": total_params,
                "num_buckets": num_buckets,
                "timing_breakdown": timing_breakdown,
                "p2p_rank_summaries": p2p_rank_summaries,
                "endpoint_results": [{"host": ep["host"], "port": ep["port"], "success": True} for ep in endpoints],
            }

        except Exception as sync_err:
            if abort_path:
                self._mark_sync_abort(abort_path, sync_err)
            if endpoint_mgr is not None:
                if sync_method == "p2p":
                    logger.warning(
                        "Rank 0: [WeightSync] P2P sync failed after streaming began; "
                        "not resuming inference because RDMA may have partially updated receiver weights"
                    )
                else:
                    try:
                        endpoint_mgr.resume()
                    except Exception as resume_err:
                        logger.warning(f"Rank 0: [WeightSync] Failed to resume inference during cleanup: {resume_err}")
            if _is_sender:
                try:
                    backend.destroy(complete_receiver=False)
                except Exception as destroy_err:
                    logger.warning(
                        f"Rank {self.rank}: [WeightSync] Failed to destroy backend during cleanup: {destroy_err}"
                    )
                # Failure path always invalidates the cache so a fresh
                # backend is created next sync.
                _cached_p2p_backend = None
                _cached_backend_key = None
            raise

    # ========================================================================
    # QLoRA collective operations (all ranks must call)
    # ========================================================================

    def _qlora_collective_ops(
        self,
        fsdp_mod,
        mod_name: str,
        collect_results: Optional[bool] = None,
    ) -> Tuple[List[Tuple[str, torch.Tensor]], List[Dict[str, Any]]]:
        """QLoRA collective ops — ALL ranks must call (collective full_tensor).

        Must be called between unshard() and reshard().

        Phase 1 (QLoRALinear): Dequantize base weights + compute LoRA delta.
          Both packed_weight_f32 and lora_A/lora_B are sharded DTensors that
          require collective full_tensor() on all ranks. The stage leader merges
          and returns the bf16 weight; other ranks return empty.

        Phase 2 (QLoRAMoeExperts): Gather LoRA params only.
          LoRA params may be FSDP-managed DTensors (when EP is disabled).
          All ranks call full_tensor() collectively. The stage leader clones the
          full params for later per-expert processing. Quantized buffers are plain
          tensors — they don't need collective ops and can be accessed after
          reshard.

        Args:
            collect_results: if True, this rank collects merged weights/contexts.
                Defaults to self.rank == 0 (backward-compatible).

        Returns:
            (linear_buffer, moe_contexts):
              linear_buffer: list of (name, tensor) for QLoRALinear merged weights
              moe_contexts: list of dicts with info for per-expert processing
        """
        if collect_results is None:
            collect_results = self.rank == 0
        # Discover QLoRA modules (only those directly owned by this FSDP module,
        # not by child FSDP modules — child modules will be processed separately)

        child_fsdp_prefixes = set()
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, FSDPModule) and mname != "":
                child_fsdp_prefixes.add(mname + ".")

        qlora_linears = {}
        qlora_moe = {}
        for mname, mod in fsdp_mod.named_modules():
            # Skip modules that belong to child FSDP modules
            if any(mname.startswith(p) or mname + "." == p for p in child_fsdp_prefixes):
                continue
            if isinstance(mod, QLoRALinear):
                qlora_linears[mname] = mod
            elif isinstance(mod, QLoRAMoeExperts):
                qlora_moe[mname] = mod

        if not qlora_linears and not qlora_moe:
            return [], []

        linear_buffer: List[Tuple[str, torch.Tensor]] = []
        moe_contexts: List[Dict[str, Any]] = []

        # --- Phase 1: QLoRALinear (all ranks participate in collective ops) ---
        for mname, mod in qlora_linears.items():
            full_prefix = f"{mod_name}.{mname}" if mod_name != "(root)" else mname

            # All ranks: dequantize base weight (collective full_tensor on packed_weight_f32)
            # Note: if Hadamard rotation is enabled, dequantized weight is in rotated space
            base_w = mod._dequantize_weight().to(dtype=torch.bfloat16)
            # All ranks: compute delta (collective full_tensor on lora_A, lora_B)
            delta = mod.get_delta_weight().to(dtype=torch.bfloat16)

            if collect_results:
                merged = base_w + delta
                linear_buffer.append((f"{full_prefix}.weight", merged))
            del base_w, delta

        # --- Phase 2: QLoRAMoeExperts (gather lora params, defer heavy work) ---
        ps = get_parallel_state()
        ep_enabled = ps.ep_enabled and ps.ep_size > 1

        for mname, mod in qlora_moe.items():
            full_prefix = f"{mod_name}.{mname}" if mod_name != "(root)" else mname

            # All ranks: gather lora params (collective if DTensor)
            # EP mode: each rank keeps LOCAL lora params (no need for full_tensor,
            #   each rank merges its own local experts with local LoRA shards)
            # Non-EP: rank 0 gathers full params via full_tensor for deferred processing
            lora_params = {}
            for hf_proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("lora_A", "lora_B"):
                    key = f"{hf_proj}_{suffix}"
                    param = getattr(mod, key)
                    if ep_enabled:
                        # EP: use local shard directly (no collective needed)
                        if hasattr(param, "to_local"):
                            lora_params[key] = param.to_local().clone()
                        else:
                            lora_params[key] = param.data.clone()
                    else:
                        # Non-EP: collective full_tensor to get global params on stage leader
                        if hasattr(param, "full_tensor"):
                            full = param.full_tensor()  # collective: all ranks
                        else:
                            full = param.data
                        if collect_results:
                            lora_params[key] = full.clone()
                        del full

            ctx = {
                "module": mod,
                "prefix": full_prefix,
                "lora_params": lora_params,
            }
            if ep_enabled:
                moe_contexts.append(ctx)  # all ranks keep context
            elif collect_results:
                moe_contexts.append(ctx)  # only stage leader

        return linear_buffer, moe_contexts

    # ========================================================================
    # EP MoE data collection (all ranks, during unshard)
    # ========================================================================

    def _collect_ep_moe_data(
        self,
        fsdp_mod,
        mod_name: str,
        ps,
        skip_clone: bool = False,
        phase_s: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Collect local EP-sharded MoE expert data during unshard phase.

        Identifies full-weight MoE modules (MoEExperts, MoEExpertsLoRA) whose
        expert params are EP-sharded DTensors. Clones local expert data for
        later EP gathering after reshard.

        QLoRAMoeExperts are handled separately by _qlora_collective_ops.

        Must be called between unshard() and reshard().
        Returns list of context dicts for _gather_and_broadcast_ep_moe_experts.
        """

        # Skip modules owned by child FSDP modules (processed separately)

        child_fsdp_prefixes = set()
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, FSDPModule) and mname != "":
                child_fsdp_prefixes.add(mname + ".")

        contexts = []
        for mname, mod in fsdp_mod.named_modules():
            if any(mname.startswith(p) or mname + "." == p for p in child_fsdp_prefixes):
                continue
            # Skip QLoRA MoE (handled by _qlora_collective_ops)
            if isinstance(mod, QLoRAMoeExperts):
                continue
            if not isinstance(mod, (MoEExperts, MoEExpertsLoRA)):
                continue

            # Get expert params — after unshard they may be plain tensors or DTensors
            gate_up = getattr(mod, "gate_up_proj", None)
            if isinstance(gate_up, torch.nn.Parameter):
                E_local = gate_up.shape[0]
            else:
                gate = getattr(mod, "gate_proj", None)
                if gate is None or not isinstance(gate, torch.nn.Parameter):
                    continue
                E_local = gate.shape[0]

            if mname:
                full_prefix = f"{mod_name}.{mname}" if mod_name != "(root)" else mname
            else:
                full_prefix = mod_name

            # Clone local expert data for each projection.
            # With EP, each rank's module already holds only local experts [E_local, K, N].
            local_experts = {}
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                param = getattr(mod, proj_name)
                if isinstance(param.data, DTensor):
                    local = param.to_local()
                else:
                    local = param.data

                # Merge LoRA if applicable
                if isinstance(mod, MoEExpertsLoRA):
                    if proj_name in mod.lora_config.target_modules:
                        t_convert = time.perf_counter()
                        delta = mod._compute_proj_delta(proj_name)
                        if isinstance(delta, DTensor):
                            delta = delta.to_local()
                        local = local.to(torch.bfloat16) + delta.to(torch.bfloat16)
                        self._add_phase_time(phase_s, "ep_collect_convert_s", time.perf_counter() - t_convert)
                    else:
                        t_convert = time.perf_counter()
                        local = local.to(torch.bfloat16)
                        self._add_phase_time(phase_s, "ep_collect_convert_s", time.perf_counter() - t_convert)
                else:
                    t_convert = time.perf_counter()
                    local = local.to(torch.bfloat16)
                    self._add_phase_time(phase_s, "ep_collect_convert_s", time.perf_counter() - t_convert)

                # Pre-unshard mode: the unsharded module storage stays
                # alive across the whole streaming loop (we reshard
                # everything at the end), so we can hand out a view
                # instead of cloning. With pre-unshard off, .clone() is
                # required because the per-iteration reshard will free
                # the source memory before transfer reads it.
                t_clone = time.perf_counter()
                local_experts[proj_name] = local if skip_clone else local.clone()
                self._add_phase_time(phase_s, "ep_collect_clone_s", time.perf_counter() - t_clone)

            contexts.append(
                {
                    "type": "full_weight",
                    "prefix": full_prefix,
                    "local_experts": local_experts,  # {proj: [E_local, K, N]}
                    "num_local_experts": E_local,
                }
            )

        return contexts

    # ========================================================================
    # EP-aware MoE expert gathering and broadcasting (all ranks)
    # ========================================================================

    def _direct_ep_transfer_experts(
        self,
        backend,
        ctx: Dict[str, Any],
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
        bucket_size_bytes: int = _DEFAULT_MOE_BUCKET_BYTES,
        quantization: Optional[Dict[str, Any]] = None,
        ps=None,
        defer_final_flush: bool = False,
        phase_s: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, int, int]:
        """Multi-sender EP path: each rank ships its own local experts.

        Compared to :meth:`_gather_and_broadcast_ep_moe_experts`, this
        skips the per-projection ``dist.gather → rank 0 → broadcast``
        funnel. Each EP rank formats its own ``ctx["local_experts"]``
        as HF-named per-expert tensors and calls
        ``backend.transfer_bucket(..., src_rank=self.rank)``. With N EP
        ranks, aggregate trainer→inference bandwidth scales N×.

        Falls back to the gather path for QLoRA contexts — the per-rank
        lora-merge path is similar in shape but model-specific and
        tracked as a follow-up.

        Like the gather path, only the EP-FSDP-rank-0 replica column
        sends; other replicas have identical local shards and would
        duplicate data on the wire.
        """
        # Backend must declare direct-EP support; fall back if not.
        if not backend.supports_direct_ep_transfer:
            return self._gather_and_broadcast_ep_moe_experts(
                backend,
                ctx,
                flush_cache=flush_cache,
                weight_version=weight_version,
                bucket_size_bytes=bucket_size_bytes,
                quantization=quantization,
                ps=ps,
            )

        is_qlora = ctx.get("type") != "full_weight"
        if is_qlora:
            # QLoRA direct-EP needs per-rank dequantize + lora merge into
            # an HF-shaped buffer, mirroring the gather path's lora math
            # but without the gather. Tracked as follow-up; defer to the
            # gather implementation today so QLoRA users still ship.
            return self._gather_and_broadcast_ep_moe_experts(
                backend,
                ctx,
                flush_cache=flush_cache,
                weight_version=weight_version,
                bucket_size_bytes=bucket_size_bytes,
                quantization=quantization,
                ps=ps,
            )

        full_prefix = ctx["prefix"]
        ep_size = ps.ep_size
        ep_rank = ps.ep_rank
        local_experts = ctx["local_experts"]
        E_local = ctx["num_local_experts"]

        ep_fsdp_rank = 0
        if ps.ep_fsdp_device_mesh is not None:
            ep_fsdp_rank = ps.ep_fsdp_device_mesh.get_local_rank("ep_fsdp")
        if ep_fsdp_rank != 0:
            ctx["local_experts"] = None
            return 0, 0, 0

        logger.info(
            f"Rank {self.rank}: [Direct-EP] prefix={full_prefix}, E_local={E_local}, E_total={E_local * ep_size}"
        )

        total_bytes = 0
        total_params = 0
        num_buckets = 0
        fp8_cpu_workspace_pending_source_limit = self._fp8_cpu_workspace_pending_source_bytes(bucket_size_bytes)
        # When batch mode defers the final flush, append to the handler-level
        # bucket so later MoE calls can coalesce into the same transfer.
        if defer_final_flush:
            bucket = self._pending_moe_bucket
            bucket_bytes = self._pending_moe_bucket_bytes
        else:
            bucket = []
            bucket_bytes = 0
        device = f"cuda:{self.rank % torch.cuda.device_count()}"
        fp8_cpu_quantization = (
            quantization is not None
            and quantization.get("quant_method") == "fp8"
            and self._fp8_quantization_target_device(backend) == "cpu"
        )
        fp8_gpu_quantization = (
            fp8_cpu_quantization
            and self._fp8_quantization_execution_device() in {"gpu", "cuda"}
            and quantization.get("fmt", "e4m3") == "e4m3"
        )
        fp8_cpu_workspace = (
            fp8_cpu_quantization
            and not fp8_gpu_quantization
            and defer_final_flush
            and self._fp8_cpu_workspace_enabled()
            and not quantization.get("modules_to_not_convert")
        )

        # local_experts[proj] is [E_local, K, N] (input-major). HF
        # convention is [N, K] per-expert (output-major) — same permute
        # the gather path does before broadcast.
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            logger.debug(f"Rank {self.rank}: [Direct-EP] {full_prefix}.{proj_name} stage=before_permute")
            local_data = local_experts[proj_name]  # [E_local, K, N]
            if fp8_gpu_quantization and local_data.device.type == "cuda":
                entries, original_bytes = self._quantize_ep_expert_projection_for_fp8_gpu_to_cpu(
                    local_data,
                    full_prefix=full_prefix,
                    proj_name=proj_name,
                    ep_rank=ep_rank,
                    quantization_config=quantization,
                    phase_s=phase_s,
                )
                total_bytes += original_bytes
                total_params += E_local
                for entry_name, entry_tensor in entries:
                    entry_bytes = entry_tensor.numel() * entry_tensor.element_size()
                    bucket.append((entry_name, entry_tensor))
                    bucket_bytes += entry_bytes

                    if bucket_bytes >= bucket_size_bytes:
                        t_backend = time.perf_counter()
                        backend.transfer_bucket(
                            bucket,
                            src_rank=self.rank,
                            flush_cache=False,
                        )
                        self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
                        bucket = []
                        bucket_bytes = 0
                        num_buckets += 1
                continue

            if fp8_cpu_workspace:
                records, original_bytes = self._stage_ep_expert_projection_for_fp8_cpu_workspace(
                    local_data,
                    full_prefix=full_prefix,
                    proj_name=proj_name,
                    ep_rank=ep_rank,
                    quantization_config=quantization,
                    phase_s=phase_s,
                )
                total_bytes += original_bytes
                total_params += E_local
                self._pending_moe_cpu_workspace_records.extend(records)
                bucket_bytes += original_bytes
                self._pending_moe_bucket_bytes = bucket_bytes
                if bucket_bytes >= fp8_cpu_workspace_pending_source_limit:
                    _, _, flushed_buckets = self._flush_pending_moe_bucket(
                        backend,
                        flush_cache=False,
                        weight_version=None,
                        quantization=quantization,
                        bucket_size_bytes=bucket_size_bytes,
                        phase_s=phase_s,
                    )
                    num_buckets += flushed_buckets
                    bucket = self._pending_moe_bucket
                    bucket_bytes = self._pending_moe_bucket_bytes
                continue

            if fp8_cpu_quantization:
                entries, original_bytes = self._quantize_ep_expert_projection_for_fp8_cpu(
                    local_data,
                    full_prefix=full_prefix,
                    proj_name=proj_name,
                    ep_rank=ep_rank,
                    quantization_config=quantization,
                    phase_s=phase_s,
                )
                total_bytes += original_bytes
                total_params += E_local
                for entry_name, entry_tensor in entries:
                    entry_bytes = entry_tensor.numel() * entry_tensor.element_size()
                    bucket.append((entry_name, entry_tensor))
                    bucket_bytes += entry_bytes

                    if bucket_bytes >= bucket_size_bytes:
                        t_backend = time.perf_counter()
                        backend.transfer_bucket(
                            bucket,
                            src_rank=self.rank,
                            flush_cache=False,
                        )
                        self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
                        bucket = []
                        bucket_bytes = 0
                        num_buckets += 1
                continue

            t_permute = time.perf_counter()
            local_stack = local_data.permute(0, 2, 1).contiguous().to(device)
            self._add_phase_time(phase_s, "direct_ep_permute_s", time.perf_counter() - t_permute)
            logger.debug(
                f"Rank {self.rank}: [Direct-EP] {full_prefix}.{proj_name} "
                f"stage=after_permute shape={tuple(local_stack.shape)} dtype={local_stack.dtype}"
            )
            for i in range(E_local):
                global_idx = ep_rank * E_local + i
                hf_name = f"{full_prefix}.{global_idx}.{proj_name}.weight"
                tensor = local_stack[i]
                tensor_bytes = tensor.numel() * tensor.element_size()
                bucket.append((hf_name, tensor))
                bucket_bytes += tensor_bytes
                total_bytes += tensor_bytes
                total_params += 1

                if bucket_bytes >= bucket_size_bytes:
                    if quantization and quantization.get("quant_method") == "fp8":
                        bucket = self._quantize_buffer_for_fp8(
                            bucket,
                            quantization_config=quantization,
                            target_device=self._fp8_quantization_target_device(backend),
                            phase_s=phase_s,
                            phase_prefix="direct_ep_fp8",
                        )
                    t_backend = time.perf_counter()
                    backend.transfer_bucket(
                        bucket,
                        src_rank=self.rank,
                        flush_cache=False,
                    )
                    self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
                    bucket = []
                    bucket_bytes = 0
                    num_buckets += 1
            del local_stack

        if defer_final_flush:
            # Hand the partial bucket back to the handler-level state
            # so the next layer's MoE call (or the final flush) picks
            # it up. Skip the per-call final flush entirely.
            self._pending_moe_bucket = bucket
            self._pending_moe_bucket_bytes = bucket_bytes
            logger.debug(
                f"Rank {self.rank}: [Direct-EP] {full_prefix} "
                f"stage=defer_final_flush bucket_bytes={bucket_bytes} bucket_len={len(bucket)}"
            )
        elif bucket:
            logger.debug(
                f"Rank {self.rank}: [Direct-EP] {full_prefix} "
                f"stage=before_final_flush bucket_bytes={bucket_bytes} bucket_len={len(bucket)}"
            )
            if quantization and quantization.get("quant_method") == "fp8":
                bucket = self._quantize_buffer_for_fp8(
                    bucket,
                    quantization_config=quantization,
                    target_device=self._fp8_quantization_target_device(backend),
                    phase_s=phase_s,
                    phase_prefix="direct_ep_fp8",
                )
            t_backend = time.perf_counter()
            backend.transfer_bucket(
                bucket,
                src_rank=self.rank,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
            self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
            num_buckets += 1
            logger.debug(f"Rank {self.rank}: [Direct-EP] {full_prefix} stage=after_final_flush")

        ctx["local_experts"] = None
        logger.info(
            f"Rank {self.rank}: [Direct-EP] {full_prefix} done "
            f"total_bytes={total_bytes} total_params={total_params} num_buckets={num_buckets}"
        )
        return total_bytes, total_params, num_buckets

    def _flush_pending_moe_bucket(
        self,
        backend,
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
        quantization: Optional[Dict[str, Any]] = None,
        bucket_size_bytes: int = _DEFAULT_MOE_BUCKET_BYTES,
        phase_s: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, int, int]:
        """Ship the leftover MoE bucket accumulated across multiple
        ``_direct_ep_transfer_experts(defer_final_flush=True)`` calls.

        Bytes/params already counted upstream by each ctx call (those
        increment as tensors are appended to the bucket, regardless of
        whether the bucket is shipped immediately or deferred), so we
        only return the bucket count here. Returns (0, 0, 1) on a
        non-empty bucket, (0, 0, 0) on empty.
        """
        if self._pending_moe_cpu_workspace_records:
            if not (quantization and quantization.get("quant_method") == "fp8"):
                raise RuntimeError("FP8 CPU workspace records require FP8 quantization config")
            bucket_bytes = self._pending_moe_bucket_bytes
            nparams = len(self._pending_moe_cpu_workspace_records)
            num_buckets = self._quantize_and_transfer_fp8_cpu_workspace_records(
                backend,
                self._pending_moe_cpu_workspace_records,
                quantization_config=quantization,
                bucket_size_bytes=bucket_size_bytes,
                flush_cache=flush_cache,
                weight_version=weight_version,
                phase_s=phase_s,
                phase_prefix="direct_ep_fp8",
            )
            self._pending_moe_bucket = []
            self._pending_moe_bucket_bytes = 0
            self._reset_fp8_cpu_workspace_usage()
            logger.info(
                f"Rank {self.rank}: [WeightSync] Cross-layer MoE CPU workspace flush: "
                f"{bucket_bytes / 1e6:.1f} MB source, {nparams} params, {num_buckets} transfer buckets"
            )
            return 0, 0, num_buckets

        if not self._pending_moe_bucket:
            self._pending_moe_bucket = []
            self._pending_moe_bucket_bytes = 0
            if flush_cache or weight_version is not None:
                backend_config = getattr(getattr(backend, "config", None), "backend_config", None)
                if backend_config is not None:
                    if weight_version is not None:
                        backend_config["weight_version"] = weight_version
                    backend_config["flush_cache"] = bool(flush_cache)
            return 0, 0, 0
        bucket = self._pending_moe_bucket
        bucket_bytes = self._pending_moe_bucket_bytes
        nparams = len(bucket)
        if quantization and quantization.get("quant_method") == "fp8":
            bucket = self._quantize_buffer_for_fp8(
                bucket,
                quantization_config=quantization,
                target_device=self._fp8_quantization_target_device(backend),
                phase_s=phase_s,
                phase_prefix="direct_ep_fp8",
            )
        t_backend = time.perf_counter()
        backend.transfer_bucket(
            bucket,
            src_rank=self.rank,
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
        # Reset state for the next sync.
        self._pending_moe_bucket = []
        self._pending_moe_bucket_bytes = 0
        logger.info(
            f"Rank {self.rank}: [WeightSync] Cross-layer MoE flush: {bucket_bytes / 1e6:.1f} MB, {nparams} params"
        )
        return 0, 0, 1

    def _transfer_bucket_in_chunks(
        self,
        backend,
        bucket: List[Tuple[str, torch.Tensor]],
        *,
        bucket_size_bytes: int,
        flush_cache: bool,
        weight_version: Optional[str],
        phase_s: Optional[Dict[str, float]],
    ) -> int:
        num_buckets = 0
        chunk: List[Tuple[str, torch.Tensor]] = []
        chunk_bytes = 0

        for name, tensor in bucket:
            entry_bytes = tensor.numel() * tensor.element_size()
            if chunk and chunk_bytes + entry_bytes > bucket_size_bytes:
                t_backend = time.perf_counter()
                backend.transfer_bucket(
                    chunk,
                    src_rank=self.rank,
                    flush_cache=False,
                )
                self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
                num_buckets += 1
                chunk = []
                chunk_bytes = 0

            chunk.append((name, tensor))
            chunk_bytes += entry_bytes

        if chunk:
            t_backend = time.perf_counter()
            backend.transfer_bucket(
                chunk,
                src_rank=self.rank,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
            self._add_phase_time(phase_s, "direct_ep_backend_s", time.perf_counter() - t_backend)
            num_buckets += 1

        return num_buckets

    def _quantize_and_transfer_fp8_cpu_workspace_records(
        self,
        backend,
        records: List[Tuple[str, Tuple[Any, ...], int]],
        *,
        quantization_config: Dict[str, Any],
        bucket_size_bytes: int,
        flush_cache: bool,
        weight_version: Optional[str],
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> int:
        stream_bytes = self._fp8_cpu_workspace_stream_bytes(bucket_size_bytes)
        record_chunks = self._chunk_fp8_cpu_workspace_records(records, max_bytes=stream_bytes)
        if not record_chunks:
            return 0

        if len(record_chunks) == 1 or not self._fp8_cpu_workspace_streaming_enabled():
            bucket = self._quantize_fp8_cpu_workspace_records(
                records,
                quantization_config=quantization_config,
                phase_s=phase_s,
                phase_prefix=phase_prefix,
            )
            return self._transfer_bucket_in_chunks(
                backend,
                bucket,
                bucket_size_bytes=bucket_size_bytes,
                flush_cache=flush_cache,
                weight_version=weight_version,
                phase_s=phase_s,
            )

        logger.info(
            "Rank %d: [WeightSync] Streaming FP8 CPU workspace flush in %d chunks (chunk cap %.1f MB)",
            self.rank,
            len(record_chunks),
            stream_bytes / 1e6,
        )

        def transfer_task(bucket: List[Tuple[str, torch.Tensor]], is_final: bool) -> float:
            t_backend = time.perf_counter()
            backend.transfer_bucket(
                bucket,
                src_rank=self.rank,
                flush_cache=(flush_cache if is_final else False),
                weight_version=(weight_version if is_final else None),
            )
            return time.perf_counter() - t_backend

        futures: List[Future[float]] = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"fp8-workspace-transfer-r{self.rank}") as executor:
            for chunk_idx, record_chunk in enumerate(record_chunks):
                bucket = self._quantize_fp8_cpu_workspace_records(
                    record_chunk,
                    quantization_config=quantization_config,
                    phase_s=phase_s,
                    phase_prefix=phase_prefix,
                )
                futures.append(executor.submit(transfer_task, bucket, chunk_idx == len(record_chunks) - 1))

            t_wait = time.perf_counter()
            first_error: Optional[BaseException] = None
            for future in futures:
                try:
                    elapsed = future.result()
                except BaseException as e:
                    if first_error is None:
                        first_error = e
                    continue
                self._add_phase_time(phase_s, "direct_ep_backend_s", elapsed)
            self._add_phase_time(
                phase_s,
                "direct_ep_fp8_workspace_stream_wait_s",
                time.perf_counter() - t_wait,
            )
            if first_error is not None:
                for future in futures:
                    future.cancel()
                raise first_error

        return len(record_chunks)

    def _gather_and_broadcast_ep_moe_experts(
        self,
        backend,
        ctx: Dict[str, Any],
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
        bucket_size_bytes: int = _DEFAULT_MOE_BUCKET_BYTES,
        quantization: Optional[Dict[str, Any]] = None,
        ps=None,
    ) -> Tuple[int, int, int]:
        """EP-aware MoE expert sync using per-projection dist.gather.

        With EP=N, each rank holds E_local = E_total/N experts. For each
        projection (gate, up, down):
        1. ALL ranks: prepare local expert weights [E_local, N, K]
        2. ALL ranks: dist.gather → rank 0 gets [E_total, N, K]
        3. Rank 0: split into per-expert HF weights, quantize, broadcast

        Handles both full-weight MoE (via local_experts in ctx) and
        QLoRA MoE (via module dequant + local LoRA merge).

        Uses 3 gather calls per MoE layer (one per projection).
        Peak memory on rank 0: ~one projection's worth of all experts.

        ALL ranks must call this (collective gather).
        """
        full_prefix = ctx["prefix"]
        ep_group = ps.ep_group
        ep_size = ps.ep_size
        device = f"cuda:{self.rank % torch.cuda.device_count()}"

        is_qlora = ctx.get("type") != "full_weight"
        ep_fsdp_rank = 0
        if ps.ep_fsdp_device_mesh is not None:
            ep_fsdp_rank = ps.ep_fsdp_device_mesh.get_local_rank("ep_fsdp")

        # EP-FSDP replicas hold identical local expert shards after unshard().
        # Only one replica column needs to gather and forward those experts to
        # rank 0; the others would just duplicate the same weights.
        if ep_fsdp_rank != 0:
            if is_qlora:
                ctx["lora_params"] = None
            else:
                ctx["local_experts"] = None
            return 0, 0, 0

        if self.rank == 0:
            logger.info(
                f"Rank 0: [EP-Gather] prefix={full_prefix}, type={'qlora' if is_qlora else 'full_weight'}, "
                f"ep_size={ep_size}"
            )

        if is_qlora:
            mod = ctx["module"]
            lora_params = ctx["lora_params"]
            E_local = mod.num_local_experts
            projections = [
                ("gate", "gate_proj", mod.hidden_size, mod.intermediate_size),
                ("up", "up_proj", mod.hidden_size, mod.intermediate_size),
                ("down", "down_proj", mod.intermediate_size, mod.hidden_size),
            ]
        else:
            local_experts = ctx["local_experts"]
            E_local = ctx["num_local_experts"]
            projections = [
                ("gate", "gate_proj", None, None),
                ("up", "up_proj", None, None),
                ("down", "down_proj", None, None),
            ]

        E_total = E_local * ep_size

        total_bytes = 0
        total_params = 0
        num_buckets = 0
        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        for proj_name, hf_proj, K, N in projections:
            # Each rank: prepare local expert weights in HF format [E_local, N, K]
            if is_qlora:
                lora_A = lora_params[f"{hf_proj}_lora_A"]  # [1 or E_local, K, r]
                lora_B = lora_params[f"{hf_proj}_lora_B"]  # [E_local or 1, r, N]
                local_merged = []
                for i in range(E_local):
                    base_w = mod.dequantize_expert(proj_name, i, K, N)
                    delta = self._compute_moe_lora_delta(mod, lora_A, lora_B, expert_idx=i)
                    merged = base_w.to(torch.bfloat16) + delta.to(torch.bfloat16)
                    local_merged.append(merged.t().contiguous())  # [N, K]
                    del base_w, delta, merged
                local_stack = torch.stack(local_merged, dim=0).to(device)
                del local_merged
            else:
                # Full-weight: already cloned as [E_local, K, N] bf16
                local_data = local_experts[hf_proj]  # [E_local, K, N]
                # Transpose each expert to HF [N, K]
                local_stack = local_data.permute(0, 2, 1).contiguous().to(device)
                del local_data

            # Gather from all EP ranks → rank 0 gets [E_total, N, K]
            if self.rank == 0:
                gathered = [torch.empty_like(local_stack) for _ in range(ep_size)]
            else:
                gathered = None
            dist.gather(local_stack, gathered, dst=0, group=ep_group)
            del local_stack

            # Rank 0: split into per-expert (name, tensor) pairs and bucket
            if self.rank == 0:
                all_experts = torch.cat(gathered, dim=0)  # [E_total, N, K]
                if hf_proj == "gate_proj":
                    logger.info(
                        f"Rank 0: [EP-Gather] {full_prefix}: all_experts={list(all_experts.shape)}, "
                        f"E_local={E_local}, E_total={E_total}, bucket_len={len(bucket)}"
                    )
                del gathered
                for global_idx in range(E_total):
                    hf_name = f"{full_prefix}.{global_idx}.{hf_proj}.weight"
                    expert_weight = all_experts[global_idx]
                    weight_bytes = expert_weight.numel() * expert_weight.element_size()
                    bucket.append((hf_name, expert_weight))
                    bucket_bytes += weight_bytes

                    # Flush bucket when full
                    if bucket_bytes >= bucket_size_bytes:
                        if quantization and quantization.get("quant_method") == "fp8":
                            bucket = self._quantize_buffer_for_fp8(
                                bucket,
                                quantization_config=quantization,
                                target_device=self._fp8_quantization_target_device(backend),
                            )
                        b, p = self._broadcast_buffer(
                            backend,
                            bucket,
                            flush_cache=False,
                        )
                        total_bytes += b
                        total_params += p
                        num_buckets += 1
                        bucket = []
                        bucket_bytes = 0

                del all_experts

        # Flush remaining bucket
        if self.rank == 0 and bucket:
            if quantization and quantization.get("quant_method") == "fp8":
                bucket = self._quantize_buffer_for_fp8(
                    bucket,
                    quantization_config=quantization,
                    target_device=self._fp8_quantization_target_device(backend),
                )
            b, p = self._broadcast_buffer(
                backend,
                bucket,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
            total_bytes += b
            total_params += p
            num_buckets += 1

        # Free cloned data
        if is_qlora:
            ctx["lora_params"] = None
        else:
            ctx["local_experts"] = None

        return total_bytes, total_params, num_buckets

    # ========================================================================
    # QLoRA MoE expert bucketed broadcasting (rank 0 only, non-EP)
    # ========================================================================

    def _broadcast_moe_experts_bucketed(
        self,
        backend,
        ctx: Dict[str, Any],
        flush_cache: bool = False,
        weight_version: Optional[str] = None,
        bucket_size_bytes: int = _DEFAULT_MOE_BUCKET_BYTES,
        quantization: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Process QLoRA MoE experts per-expert and broadcast in buckets.

        Called AFTER reshard on rank 0 only. Dequantizes one expert at a time,
        computes per-expert LoRA delta, merges to bf16, and accumulates into
        a bucket. When the bucket reaches bucket_size_bytes, it's broadcast
        via NCCL and cleared.

        This keeps peak memory to ~bucket_size instead of materializing all
        128×3 expert weights simultaneously.

        Args:
            backend: WeightTransportBackend for transferring weight buckets
            ctx: dict with 'module', 'prefix', 'lora_params'
            flush_cache: whether to flush inference KV cache on last bucket
            bucket_size_bytes: max bucket size before broadcasting

        Returns:
            (total_bytes, total_params, num_buckets)
        """
        mod = ctx["module"]
        full_prefix = ctx["prefix"]
        lora_params = ctx["lora_params"]

        total_bytes = 0
        total_params = 0
        num_buckets = 0
        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        projections = [
            ("gate", "gate_proj", mod.hidden_size, mod.intermediate_size),
            ("up", "up_proj", mod.hidden_size, mod.intermediate_size),
            ("down", "down_proj", mod.intermediate_size, mod.hidden_size),
        ]

        E = mod.num_local_experts

        for proj_name, hf_proj, K, N in projections:
            lora_A = lora_params[f"{hf_proj}_lora_A"]  # [1 or E, K, r]
            lora_B = lora_params[f"{hf_proj}_lora_B"]  # [E or 1, r, N]

            for expert_idx in range(E):
                # Dequantize single expert (plain tensor, no collective needed)
                base_w = mod.dequantize_expert(proj_name, expert_idx, K, N)  # [K, N]

                # Per-expert LoRA delta: A[i] @ B[i] * scaling → [K, N]
                delta = self._compute_moe_lora_delta(mod, lora_A, lora_B, expert_idx=expert_idx)  # [K, N]

                merged = base_w.to(torch.bfloat16) + delta.to(torch.bfloat16)
                del base_w, delta

                # HF format: [N, K] (transpose from GKN)
                hf_name = f"{full_prefix}.{expert_idx}.{hf_proj}.weight"
                expert_weight = merged.t().contiguous()
                del merged

                weight_bytes = expert_weight.numel() * expert_weight.element_size()
                bucket.append((hf_name, expert_weight))
                bucket_bytes += weight_bytes

                # Flush bucket when full
                if bucket_bytes >= bucket_size_bytes:
                    if quantization and quantization.get("quant_method") == "fp8":
                        bucket = self._quantize_buffer_for_fp8(
                            bucket,
                            quantization_config=quantization,
                            target_device=self._fp8_quantization_target_device(backend),
                        )
                    b, p = self._broadcast_buffer(
                        backend,
                        bucket,
                        flush_cache=False,
                    )
                    total_bytes += b
                    total_params += p
                    num_buckets += 1
                    bucket = []
                    bucket_bytes = 0

        # Flush remaining bucket
        if bucket:
            if quantization and quantization.get("quant_method") == "fp8":
                bucket = self._quantize_buffer_for_fp8(
                    bucket,
                    quantization_config=quantization,
                    target_device=self._fp8_quantization_target_device(backend),
                )
            b, p = self._broadcast_buffer(
                backend,
                bucket,
                flush_cache=flush_cache,
                weight_version=weight_version,
            )
            total_bytes += b
            total_params += p
            num_buckets += 1

        # Free cloned lora params
        del lora_params
        ctx["lora_params"] = None

        return total_bytes, total_params, num_buckets

    # ========================================================================
    # PP inter-stage NCCL transfer helpers
    # ========================================================================

    def _pp_nccl_transfer_buffer(
        self,
        send_buffer: Optional[List[Tuple[str, torch.Tensor]]],
        pp_grp,
        src_global: int,
        device: str,
    ) -> Optional[List[Tuple[str, torch.Tensor]]]:
        """Transfer named tensor buffer from one PP stage leader to others via NCCL.

        ALL pp_group members must call this (collective broadcast).

        The source rank (src_global) provides send_buffer: a list of
        (name, bf16_tensor) pairs. Other members pass None.

        Protocol:
          1. broadcast_object_list: metadata [(name, shape), ...] — tiny, pickle OK
          2. dist.broadcast: single flat bf16 tensor — all weights concatenated

        Returns:
            For non-source ranks: list of (name, tensor) on device
            For source rank: None
        """
        is_src = self.rank == src_global

        # Step 1: broadcast metadata (names + shapes)
        if is_src:
            meta = [(n, list(t.shape)) for n, t in send_buffer] if send_buffer else []
            obj = [meta]
        else:
            obj = [None]
        dist.broadcast_object_list(obj, src=src_global, group=pp_grp)
        meta = obj[0]

        if not meta:
            return [] if not is_src else None

        # Step 2: broadcast flat bf16 tensor via NCCL
        if is_src:
            flat = torch.cat([t.to(dtype=torch.bfloat16).reshape(-1) for _, t in send_buffer])
        else:
            total_el = sum(_prod(s) for _, s in meta)
            flat = torch.empty(total_el, dtype=torch.bfloat16, device=device)
        dist.broadcast(flat, src=src_global, group=pp_grp)

        if is_src:
            del flat
            return None

        # Split flat tensor back into named tensors
        result = []
        offset = 0
        for name, shape in meta:
            numel = _prod(shape)
            result.append((name, flat[offset : offset + numel].view(shape)))
            offset += numel
        return result

    def _compute_moe_experts_buffer(
        self,
        ctx: Dict[str, Any],
    ) -> List[Tuple[str, torch.Tensor]]:
        """Compute merged QLoRA MoE expert weights as bf16 GPU tensors.

        Used by PP follower stage leaders to include MoE expert weights in the
        per-module NCCL transfer buffer (bf16, NOT quantized — rank 0 quantizes
        after receiving).

        Args:
            ctx: dict with 'module', 'prefix', 'lora_params' (from _qlora_collective_ops)

        Returns:
            list of (name, bf16_gpu_tensor) pairs
        """
        mod = ctx["module"]
        full_prefix = ctx["prefix"]
        lora_params = ctx["lora_params"]
        E = mod.num_local_experts

        projections = [
            ("gate", "gate_proj", mod.hidden_size, mod.intermediate_size),
            ("up", "up_proj", mod.hidden_size, mod.intermediate_size),
            ("down", "down_proj", mod.intermediate_size, mod.hidden_size),
        ]

        items: List[Tuple[str, torch.Tensor]] = []
        for proj_name, hf_proj, K, N in projections:
            lora_A = lora_params[f"{hf_proj}_lora_A"]
            lora_B = lora_params[f"{hf_proj}_lora_B"]
            for expert_idx in range(E):
                base_w = mod.dequantize_expert(proj_name, expert_idx, K, N)
                delta = self._compute_moe_lora_delta(mod, lora_A, lora_B, expert_idx=expert_idx)
                merged = (base_w.to(torch.bfloat16) + delta.to(torch.bfloat16)).t().contiguous()
                del base_w, delta
                hf_name = f"{full_prefix}.{expert_idx}.{hf_proj}.weight"
                items.append((hf_name, merged))

        logger.info(f"Rank {self.rank}: [WeightSync] MoE {full_prefix}: {len(items)} experts")

        # Free cloned lora params
        del lora_params
        ctx["lora_params"] = None

        return items

    # ========================================================================
    # FP8 quantization
    # ========================================================================

    @staticmethod
    def _fp8_quantization_target_device(backend) -> Optional[str]:
        """Use CPU quantization for P2P, preserving NCCL's device-local path."""
        if backend.__class__.__name__ == "P2PTransportBackend":
            return "cpu"
        return None

    @staticmethod
    def _fp8_quantization_execution_device() -> str:
        return os.environ.get("XORL_P2P_FP8_QUANTIZE_DEVICE", "cpu").strip().lower()

    @staticmethod
    def _fp8_cpu_workspace_enabled() -> bool:
        return os.environ.get("XORL_P2P_FP8_CPU_WORKSPACE", "0") == "1"

    @staticmethod
    def _fp8_cpu_workspace_pin_input() -> bool:
        return os.environ.get("XORL_P2P_FP8_CPU_WORKSPACE_PINNED", "1") != "0"

    @staticmethod
    def _fp8_cpu_workspace_min_capacity() -> int:
        return _env_int("XORL_P2P_FP8_CPU_WORKSPACE_MIN_CAPACITY", 16)

    @staticmethod
    def _fp8_cpu_workspace_streaming_enabled() -> bool:
        return os.environ.get("XORL_P2P_FP8_CPU_WORKSPACE_STREAMING", "1") != "0"

    @staticmethod
    def _fp8_dtype_and_max(quantization_config: Dict[str, Any]) -> Tuple[torch.dtype, float]:
        fmt = quantization_config.get("fmt", "e4m3")
        if fmt == "e5m2":
            fp8_dtype = torch.float8_e5m2
        else:
            fp8_dtype = torch.float8_e4m3fn
        return fp8_dtype, torch.finfo(fp8_dtype).max

    @staticmethod
    def _fp8_block_size(quantization_config: Dict[str, Any]) -> Tuple[int, int]:
        block_size_list = quantization_config.get("weight_block_size", [128, 128])
        block_size_row = block_size_list[0]
        block_size_col = block_size_list[1] if len(block_size_list) > 1 else block_size_list[0]
        return block_size_row, block_size_col

    def _reset_fp8_cpu_workspace_usage(self) -> None:
        self._pending_moe_cpu_workspace_records = []
        for workspace in self._fp8_cpu_workspaces.values():
            workspace["used"] = 0

    @staticmethod
    def _add_phase_time(phase_s: Optional[Dict[str, float]], name: str, elapsed_s: float) -> None:
        if phase_s is not None:
            phase_s[name] = phase_s.get(name, 0.0) + elapsed_s

    @staticmethod
    def _copy_tensor_to_cpu_for_fp8(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cpu":
            return tensor.detach()
        if (
            tensor.device.type == "cuda"
            and torch.cuda.is_available()
            and os.environ.get("XORL_P2P_FP8_PINNED_CPU_COPY", "1") != "0"
        ):
            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor.detach(), non_blocking=True)
            torch.cuda.current_stream(tensor.device).synchronize()
            return cpu_tensor
        return tensor.detach().to("cpu")

    @staticmethod
    def _should_quantize_fp8_weight(
        name: str,
        tensor: torch.Tensor,
        modules_to_not_convert: List[str],
    ) -> bool:
        if not (name.endswith(".weight") and tensor.ndim == 2):
            return False
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return False

        if modules_to_not_convert:
            return not any(
                name == prefix + ".weight" or name.startswith(prefix + ".") for prefix in modules_to_not_convert
            )

        return "_proj.weight" in name or name.endswith("fused_qkv_a_proj_with_mqa.weight")

    @staticmethod
    def _can_group_fp8_tensor(first: torch.Tensor, tensor: torch.Tensor, group_len: int) -> bool:
        if not (first.is_contiguous() and tensor.is_contiguous()):
            return False
        if first.shape != tensor.shape or first.dtype != tensor.dtype or first.device != tensor.device:
            return False
        if first.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
            return False

        rows, cols = first.shape
        return tensor.storage_offset() == first.storage_offset() + group_len * rows * cols

    @staticmethod
    def _can_quantize_fp8_stack_on_gpu(
        stack: torch.Tensor,
        *,
        fp8_dtype: torch.dtype,
        block_size_row: int,
        block_size_col: int,
    ) -> bool:
        return (
            WeightSyncHandler._fp8_quantization_execution_device() in {"gpu", "cuda"}
            and stack.device.type == "cuda"
            and fp8_dtype == torch.float8_e4m3fn
            and block_size_row == block_size_col
            and stack.ndim == 3
            and stack.shape[1] % block_size_row == 0
        )

    @staticmethod
    def _quantize_fp8_stack_on_gpu_to_cpu(
        stack: torch.Tensor,
        *,
        block_size: int,
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from xorl.ops.quantize import block_fp8_quantize_gkn  # noqa: PLC0415

        if stack.ndim != 3:
            raise ValueError(f"Expected a 3D FP8 quantization stack, got shape={tuple(stack.shape)}")
        count, rows, cols = stack.shape
        if rows % block_size != 0:
            raise ValueError(f"GPU FP8 stack quantization requires rows divisible by {block_size}, got rows={rows}")

        t_quant = time.perf_counter()
        work = stack.detach().contiguous()
        flat = work.reshape(count * rows, cols)
        quantized_flat, scale_flat = block_fp8_quantize_gkn(flat, block_size=block_size)
        torch.cuda.current_stream(stack.device).synchronize()
        WeightSyncHandler._add_phase_time(phase_s, f"{phase_prefix}_gpu_quant_s", time.perf_counter() - t_quant)

        scale_cols = (cols + block_size - 1) // block_size
        quantized = quantized_flat.reshape(count, rows, cols)
        scale_inv = scale_flat.reshape(count, rows // block_size, scale_cols)

        t_copy = time.perf_counter()
        quantized_cpu = WeightSyncHandler._copy_tensor_to_cpu_for_fp8(quantized)
        scale_cpu = WeightSyncHandler._copy_tensor_to_cpu_for_fp8(scale_inv)
        WeightSyncHandler._add_phase_time(
            phase_s,
            f"{phase_prefix}_gpu_output_copy_s",
            time.perf_counter() - t_copy,
        )
        return quantized_cpu, scale_cpu

    @staticmethod
    def _quantize_fp8_stack(
        stack: torch.Tensor,
        *,
        fp8_dtype: torch.dtype,
        fp8_max: float,
        block_size_row: int,
        block_size_col: int,
        target_device: Optional[str],
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a [count, rows, cols] tensor stack and return FP8 weights + scales."""
        if stack.ndim != 3:
            raise ValueError(f"Expected a 3D FP8 quantization stack, got shape={tuple(stack.shape)}")

        if (
            target_device is not None
            and torch.device(target_device).type == "cpu"
            and WeightSyncHandler._can_quantize_fp8_stack_on_gpu(
                stack,
                fp8_dtype=fp8_dtype,
                block_size_row=block_size_row,
                block_size_col=block_size_col,
            )
        ):
            return WeightSyncHandler._quantize_fp8_stack_on_gpu_to_cpu(
                stack,
                block_size=block_size_row,
                phase_s=phase_s,
                phase_prefix=phase_prefix,
            )

        work = stack.detach()
        if target_device is not None:
            t_copy = time.perf_counter()
            if torch.device(target_device).type == "cpu":
                work = WeightSyncHandler._copy_tensor_to_cpu_for_fp8(work)
            else:
                work = work.to(target_device)
            WeightSyncHandler._add_phase_time(
                phase_s,
                f"{phase_prefix}_target_copy_s",
                time.perf_counter() - t_copy,
            )

        t_float = time.perf_counter()
        work = work.float()
        WeightSyncHandler._add_phase_time(phase_s, f"{phase_prefix}_float_s", time.perf_counter() - t_float)

        count, rows, cols = work.shape
        pad_rows = (block_size_row - rows % block_size_row) % block_size_row
        pad_cols = (block_size_col - cols % block_size_col) % block_size_col

        if pad_rows > 0 or pad_cols > 0:
            t_pad = time.perf_counter()
            padded = torch.zeros(
                count,
                rows + pad_rows,
                cols + pad_cols,
                dtype=work.dtype,
                device=work.device,
            )
            padded[:, :rows, :cols] = work
            WeightSyncHandler._add_phase_time(phase_s, f"{phase_prefix}_pad_s", time.perf_counter() - t_pad)
        else:
            padded = work

        nr = padded.shape[1] // block_size_row
        nc = padded.shape[2] // block_size_col
        blocks = padded.reshape(count, nr, block_size_row, nc, block_size_col).permute(0, 1, 3, 2, 4)

        t_reduce = time.perf_counter()
        block_max = blocks.abs().reshape(count, nr, nc, -1).max(dim=-1).values
        scale = block_max.clamp(min=1e-12) / fp8_max
        scale_inv = scale.to(torch.float32)
        WeightSyncHandler._add_phase_time(phase_s, f"{phase_prefix}_reduce_s", time.perf_counter() - t_reduce)

        t_cast = time.perf_counter()
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
        quantized_blocks = (blocks / scale_expanded).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        quantized = quantized_blocks.permute(0, 1, 3, 2, 4).reshape(count, padded.shape[1], padded.shape[2])
        WeightSyncHandler._add_phase_time(phase_s, f"{phase_prefix}_cast_s", time.perf_counter() - t_cast)

        if pad_rows > 0 or pad_cols > 0:
            quantized = quantized[:, :rows, :cols].contiguous()

        return quantized, scale_inv

    @staticmethod
    def _quantize_single_fp8_tensor(
        tensor: torch.Tensor,
        *,
        fp8_dtype: torch.dtype,
        fp8_max: float,
        block_size_row: int,
        block_size_col: int,
        target_device: Optional[str],
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, scale_inv = WeightSyncHandler._quantize_fp8_stack(
            tensor.unsqueeze(0),
            fp8_dtype=fp8_dtype,
            fp8_max=fp8_max,
            block_size_row=block_size_row,
            block_size_col=block_size_col,
            target_device=target_device,
            phase_s=phase_s,
            phase_prefix=phase_prefix,
        )
        return quantized[0].contiguous(), scale_inv[0].contiguous()

    @staticmethod
    def _quantize_ep_expert_projection_for_fp8_cpu(
        local_data: torch.Tensor,
        *,
        full_prefix: str,
        proj_name: str,
        ep_rank: int,
        quantization_config: Dict[str, Any],
        phase_s: Optional[Dict[str, float]],
    ) -> Tuple[List[Tuple[str, torch.Tensor]], int]:
        """CPU-quantize one EP-local MoE projection stack.

        ``local_data`` is stored in training layout [E, K, N]. SGLang locators
        expect HF layout [N, K] per expert, plus a matching
        ``weight_scale_inv`` tensor.
        """
        entries, original_bytes = WeightSyncHandler._format_ep_expert_projection_for_fp8_cpu(
            local_data,
            full_prefix=full_prefix,
            proj_name=proj_name,
            ep_rank=ep_rank,
            phase_s=phase_s,
        )
        return (
            WeightSyncHandler._quantize_buffer_for_fp8(
                entries,
                quantization_config=quantization_config,
                target_device=None,
                phase_s=phase_s,
                phase_prefix="direct_ep_fp8",
            ),
            original_bytes,
        )

    @staticmethod
    def _format_ep_expert_projection_for_fp8_cpu(
        local_data: torch.Tensor,
        *,
        full_prefix: str,
        proj_name: str,
        ep_rank: int,
        phase_s: Optional[Dict[str, float]],
    ) -> Tuple[List[Tuple[str, torch.Tensor]], int]:
        """Copy one EP-local MoE projection to CPU HF layout without quantizing."""
        original_bytes = local_data.numel() * local_data.element_size()
        t_copy = time.perf_counter()
        cpu_data = WeightSyncHandler._copy_tensor_to_cpu_for_fp8(local_data)
        WeightSyncHandler._add_phase_time(phase_s, "direct_ep_fp8_source_copy_s", time.perf_counter() - t_copy)

        t_transpose = time.perf_counter()
        hf_stack = cpu_data.permute(0, 2, 1).contiguous()
        WeightSyncHandler._add_phase_time(phase_s, "direct_ep_fp8_cpu_transpose_s", time.perf_counter() - t_transpose)
        del cpu_data

        e_local = hf_stack.shape[0]
        names = [f"{full_prefix}.{ep_rank * e_local + expert_idx}.{proj_name}.weight" for expert_idx in range(e_local)]
        return [(name, hf_stack[idx]) for idx, name in enumerate(names)], original_bytes

    def _ensure_fp8_cpu_workspace(
        self,
        key: Tuple[Any, ...],
        *,
        required: int,
        rows: int,
        cols: int,
        input_dtype: torch.dtype,
        fp8_dtype: torch.dtype,
        block_size_row: int,
        block_size_col: int,
        phase_s: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        workspace = self._fp8_cpu_workspaces.get(key)
        if workspace is not None and workspace["capacity"] >= required:
            return workspace

        t_alloc = time.perf_counter()
        old_workspace = workspace
        old_used = int(old_workspace.get("used", 0)) if old_workspace is not None else 0
        old_capacity = int(old_workspace.get("capacity", 0)) if old_workspace is not None else 0
        min_capacity = self._fp8_cpu_workspace_min_capacity()
        new_capacity = max(required, min_capacity, old_capacity * 2 if old_capacity else 0)
        scale_rows = (rows + block_size_row - 1) // block_size_row
        scale_cols = (cols + block_size_col - 1) // block_size_col
        pin_input = self._fp8_cpu_workspace_pin_input() and torch.cuda.is_available()

        input_workspace = torch.empty(
            (new_capacity, rows, cols),
            dtype=input_dtype,
            device="cpu",
            pin_memory=pin_input,
        )
        if old_workspace is not None and old_used:
            input_workspace[:old_used].copy_(old_workspace["input"][:old_used])

        workspace = {
            "capacity": new_capacity,
            "used": old_used,
            "rows": rows,
            "cols": cols,
            "input_dtype": input_dtype,
            "fp8_dtype": fp8_dtype,
            "block_size_row": block_size_row,
            "block_size_col": block_size_col,
            "input": input_workspace,
            "float": torch.empty((new_capacity, rows, cols), dtype=torch.float32, device="cpu"),
            "abs": torch.empty((new_capacity, rows, cols), dtype=torch.float32, device="cpu"),
            "quantized": torch.empty((new_capacity, rows, cols), dtype=fp8_dtype, device="cpu"),
            "scale": torch.empty((new_capacity, scale_rows, scale_cols), dtype=torch.float32, device="cpu"),
        }
        self._fp8_cpu_workspaces[key] = workspace
        self._add_phase_time(phase_s, "direct_ep_fp8_workspace_alloc_s", time.perf_counter() - t_alloc)
        logger.info(
            "Rank %d: [Direct-EP] FP8 CPU workspace key=%s capacity=%d rows=%d cols=%d pin_input=%s",
            self.rank,
            key,
            new_capacity,
            rows,
            cols,
            pin_input,
        )
        return workspace

    def _stage_ep_expert_projection_for_fp8_cpu_workspace(
        self,
        local_data: torch.Tensor,
        *,
        full_prefix: str,
        proj_name: str,
        ep_rank: int,
        quantization_config: Dict[str, Any],
        phase_s: Optional[Dict[str, float]],
    ) -> Tuple[List[Tuple[str, Tuple[Any, ...], int]], int]:
        """Stage one EP-local projection into reusable CPU HF-layout storage."""
        fp8_dtype, _ = self._fp8_dtype_and_max(quantization_config)
        block_size_row, block_size_col = self._fp8_block_size(quantization_config)
        original_bytes = local_data.numel() * local_data.element_size()
        e_local, cols, rows = local_data.shape
        key = (rows, cols, local_data.dtype, fp8_dtype, block_size_row, block_size_col)

        workspace_used = int(self._fp8_cpu_workspaces.get(key, {}).get("used", 0))
        workspace = self._ensure_fp8_cpu_workspace(
            key,
            required=workspace_used + e_local,
            rows=rows,
            cols=cols,
            input_dtype=local_data.dtype,
            fp8_dtype=fp8_dtype,
            block_size_row=block_size_row,
            block_size_col=block_size_col,
            phase_s=phase_s,
        )
        start_idx = int(workspace_used)
        end_idx = start_idx + e_local

        t_copy = time.perf_counter()
        src = local_data.detach().permute(0, 2, 1)
        dst = workspace["input"][start_idx:end_idx]
        dst.copy_(src, non_blocking=(local_data.device.type == "cuda" and dst.is_pinned()))
        if local_data.device.type == "cuda":
            torch.cuda.current_stream(local_data.device).synchronize()
        self._add_phase_time(phase_s, "direct_ep_fp8_workspace_copy_s", time.perf_counter() - t_copy)

        workspace["used"] = end_idx
        records = [
            (f"{full_prefix}.{ep_rank * e_local + expert_idx}.{proj_name}.weight", key, start_idx + expert_idx)
            for expert_idx in range(e_local)
        ]
        return records, original_bytes

    def _fp8_cpu_workspace_record_bytes(self, key: Tuple[Any, ...]) -> int:
        workspace = self._fp8_cpu_workspaces[key]
        rows = int(workspace["rows"])
        cols = int(workspace["cols"])
        block_size_row = int(workspace["block_size_row"])
        block_size_col = int(workspace["block_size_col"])
        fp8_dtype = workspace["fp8_dtype"]
        scale_rows = (rows + block_size_row - 1) // block_size_row
        scale_cols = (cols + block_size_col - 1) // block_size_col
        weight_bytes = rows * cols * torch.empty((), dtype=fp8_dtype).element_size()
        scale_bytes = scale_rows * scale_cols * torch.empty((), dtype=torch.float32).element_size()
        return weight_bytes + scale_bytes

    def _fp8_cpu_workspace_records_bytes(self, records: List[Tuple[str, Tuple[Any, ...], int]]) -> int:
        return sum(self._fp8_cpu_workspace_record_bytes(key) for _, key, _ in records)

    @staticmethod
    def _fp8_cpu_workspace_stream_bytes(bucket_size_bytes: int) -> int:
        max_stream_bytes = _env_int(
            "XORL_P2P_FP8_CPU_WORKSPACE_STREAM_BYTES",
            bucket_size_bytes,
        )
        return min(max_stream_bytes, bucket_size_bytes)

    @staticmethod
    def _fp8_cpu_workspace_pending_source_bytes(bucket_size_bytes: int) -> int:
        return max(1, _env_int("XORL_P2P_FP8_CPU_WORKSPACE_PENDING_SOURCE_BYTES", bucket_size_bytes))

    def _chunk_fp8_cpu_workspace_records(
        self,
        records: List[Tuple[str, Tuple[Any, ...], int]],
        *,
        max_bytes: int,
    ) -> List[List[Tuple[str, Tuple[Any, ...], int]]]:
        chunks: List[List[Tuple[str, Tuple[Any, ...], int]]] = []
        chunk: List[Tuple[str, Tuple[Any, ...], int]] = []
        chunk_bytes = 0

        for record in records:
            entry_bytes = self._fp8_cpu_workspace_record_bytes(record[1])
            if chunk and chunk_bytes + entry_bytes > max_bytes:
                chunks.append(chunk)
                chunk = []
                chunk_bytes = 0
            chunk.append(record)
            chunk_bytes += entry_bytes

        if chunk:
            chunks.append(chunk)
        return chunks

    def _quantize_fp8_cpu_workspace_range(
        self,
        workspace: Dict[str, Any],
        *,
        start: int,
        end: int,
        fp8_dtype: torch.dtype,
        fp8_max: float,
        block_size_row: int,
        block_size_col: int,
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> None:
        rows = int(workspace["rows"])
        cols = int(workspace["cols"])
        source = workspace["input"][start:end]
        if rows % block_size_row != 0 or cols % block_size_col != 0:
            quantized, scale = self._quantize_fp8_stack(
                source,
                fp8_dtype=fp8_dtype,
                fp8_max=fp8_max,
                block_size_row=block_size_row,
                block_size_col=block_size_col,
                target_device=None,
                phase_s=phase_s,
                phase_prefix=phase_prefix,
            )
            workspace["quantized"][start:end].copy_(quantized)
            workspace["scale"][start:end, : scale.shape[1], : scale.shape[2]].copy_(scale)
            return

        count = end - start
        work = workspace["float"][start:end]
        abs_work = workspace["abs"][start:end]
        scale = workspace["scale"][start:end]
        quantized = workspace["quantized"][start:end]
        nr = rows // block_size_row
        nc = cols // block_size_col

        t_float = time.perf_counter()
        work.copy_(source)
        self._add_phase_time(phase_s, f"{phase_prefix}_float_s", time.perf_counter() - t_float)

        t_reduce = time.perf_counter()
        torch.abs(work, out=abs_work)
        blocks_abs = abs_work.reshape(count, nr, block_size_row, nc, block_size_col)
        torch.amax(blocks_abs, dim=(2, 4), out=scale)
        scale.clamp_(min=1e-12).div_(fp8_max)
        self._add_phase_time(phase_s, f"{phase_prefix}_reduce_s", time.perf_counter() - t_reduce)

        t_cast = time.perf_counter()
        blocks = work.reshape(count, nr, block_size_row, nc, block_size_col)
        blocks.div_(scale.reshape(count, nr, 1, nc, 1))
        work.clamp_(min=-fp8_max, max=fp8_max)
        quantized.copy_(work)
        self._add_phase_time(phase_s, f"{phase_prefix}_cast_s", time.perf_counter() - t_cast)

    def _quantize_fp8_cpu_workspace_record_batch(
        self,
        records: List[Tuple[str, Tuple[Any, ...], int]],
        *,
        quantization_config: Dict[str, Any],
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> None:
        fp8_dtype, fp8_max = self._fp8_dtype_and_max(quantization_config)
        block_size_row, block_size_col = self._fp8_block_size(quantization_config)
        by_key: Dict[Tuple[Any, ...], List[int]] = {}
        for _, key, index in records:
            by_key.setdefault(key, []).append(index)

        for key, indices in by_key.items():
            workspace = self._fp8_cpu_workspaces[key]
            used = int(workspace["used"])
            if workspace["fp8_dtype"] != fp8_dtype:
                raise RuntimeError(f"FP8 workspace dtype mismatch: {workspace['fp8_dtype']} != {fp8_dtype}")
            if int(workspace["block_size_row"]) != block_size_row or int(workspace["block_size_col"]) != block_size_col:
                raise RuntimeError("FP8 workspace block-size mismatch")

            unique_indices = sorted(set(indices))
            if not unique_indices:
                continue
            if unique_indices[-1] >= used:
                raise RuntimeError(f"FP8 workspace record index {unique_indices[-1]} exceeds used count {used}")

            range_start = unique_indices[0]
            range_end = range_start + 1
            for index in unique_indices[1:]:
                if index == range_end:
                    range_end += 1
                    continue
                self._quantize_fp8_cpu_workspace_range(
                    workspace,
                    start=range_start,
                    end=range_end,
                    fp8_dtype=fp8_dtype,
                    fp8_max=fp8_max,
                    block_size_row=block_size_row,
                    block_size_col=block_size_col,
                    phase_s=phase_s,
                    phase_prefix=phase_prefix,
                )
                range_start = index
                range_end = index + 1
            self._quantize_fp8_cpu_workspace_range(
                workspace,
                start=range_start,
                end=range_end,
                fp8_dtype=fp8_dtype,
                fp8_max=fp8_max,
                block_size_row=block_size_row,
                block_size_col=block_size_col,
                phase_s=phase_s,
                phase_prefix=phase_prefix,
            )

    def _quantize_fp8_cpu_workspace_records(
        self,
        records: List[Tuple[str, Tuple[Any, ...], int]],
        *,
        quantization_config: Dict[str, Any],
        phase_s: Optional[Dict[str, float]],
        phase_prefix: str,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Quantize staged workspace tensors while preserving record order."""
        self._quantize_fp8_cpu_workspace_record_batch(
            records,
            quantization_config=quantization_config,
            phase_s=phase_s,
            phase_prefix=phase_prefix,
        )

        result: List[Tuple[str, torch.Tensor]] = []
        for name, key, index in records:
            workspace = self._fp8_cpu_workspaces[key]
            result.append((name, workspace["quantized"][index]))
            result.append((name.replace(".weight", ".weight_scale_inv"), workspace["scale"][index]))
        return result

    @staticmethod
    def _quantize_ep_expert_projection_for_fp8_gpu_to_cpu(
        local_data: torch.Tensor,
        *,
        full_prefix: str,
        proj_name: str,
        ep_rank: int,
        quantization_config: Dict[str, Any],
        phase_s: Optional[Dict[str, float]],
    ) -> Tuple[List[Tuple[str, torch.Tensor]], int]:
        """GPU-quantize one EP-local MoE projection stack and return CPU tensors for P2P."""
        fmt = quantization_config.get("fmt", "e4m3")
        if fmt != "e4m3":
            raise ValueError("GPU FP8 quantization currently supports only e4m3")

        block_size_list = quantization_config.get("weight_block_size", [128, 128])
        block_size_row = block_size_list[0]
        block_size_col = block_size_list[1] if len(block_size_list) > 1 else block_size_list[0]
        if block_size_row != block_size_col:
            raise ValueError("GPU FP8 quantization requires a square block size")

        original_bytes = local_data.numel() * local_data.element_size()

        t_layout = time.perf_counter()
        hf_stack = local_data.detach().permute(0, 2, 1).contiguous()
        torch.cuda.current_stream(local_data.device).synchronize()
        WeightSyncHandler._add_phase_time(phase_s, "direct_ep_fp8_gpu_layout_s", time.perf_counter() - t_layout)

        e_local = hf_stack.shape[0]
        modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
        names = [f"{full_prefix}.{ep_rank * e_local + idx}.{proj_name}.weight" for idx in range(e_local)]
        if not all(
            WeightSyncHandler._should_quantize_fp8_weight(name, hf_stack[idx], modules_to_not_convert)
            for idx, name in enumerate(names)
        ):
            t_copy = time.perf_counter()
            hf_stack_cpu = WeightSyncHandler._copy_tensor_to_cpu_for_fp8(hf_stack)
            WeightSyncHandler._add_phase_time(
                phase_s,
                "direct_ep_fp8_gpu_output_copy_s",
                time.perf_counter() - t_copy,
            )
            entries = [(name, hf_stack_cpu[idx]) for idx, name in enumerate(names)]
            return (
                WeightSyncHandler._quantize_buffer_for_fp8(
                    entries,
                    quantization_config=quantization_config,
                    target_device=None,
                    phase_s=phase_s,
                    phase_prefix="direct_ep_fp8",
                ),
                original_bytes,
            )

        quantized_stack, scale_stack = WeightSyncHandler._quantize_fp8_stack_on_gpu_to_cpu(
            hf_stack,
            block_size=block_size_row,
            phase_s=phase_s,
            phase_prefix="direct_ep_fp8",
        )

        result: List[Tuple[str, torch.Tensor]] = []
        for idx, name in enumerate(names):
            result.append((name, quantized_stack[idx]))
            result.append((name.replace(".weight", ".weight_scale_inv"), scale_stack[idx]))
        return result, original_bytes

    @staticmethod
    def _quantize_buffer_for_fp8(
        buffer: List[Tuple[str, torch.Tensor]],
        quantization_config: Optional[Dict[str, Any]] = None,
        target_device: Optional[str] = None,
        phase_s: Optional[Dict[str, float]] = None,
        phase_prefix: str = "fp8",
    ) -> List[Tuple[str, torch.Tensor]]:
        """Quantize bf16 weight tensors to FP8 with block-wise scales.

        For each weight tensor, produces two entries:
        - (name, fp8_weight): quantized weight in the specified FP8 format
        - (name_scale_inv, scale_inv): per-block scale_inv in float32

        Uses quantization_config (HF format) for:
        - fmt: "e4m3" (default) or "e5m2" → torch.float8_e4m3fn or torch.float8_e5m2
        - weight_block_size: [row_block, col_block], default [128, 128]
        - modules_to_not_convert: list of module name prefixes to skip quantization

        Non-weight params (e.g. layernorm, embedding) are passed through as-is.
        When target_device="cpu", quantized tensors are returned on CPU so the
        P2P backend can stage them through the registered CPU pool and transfer
        fewer bytes to the receiver.
        """
        if quantization_config is None:
            quantization_config = {}

        # FP8 format: e4m3 (default, higher precision) or e5m2 (wider range)
        fp8_dtype, fp8_max = WeightSyncHandler._fp8_dtype_and_max(quantization_config)
        block_size_row, block_size_col = WeightSyncHandler._fp8_block_size(quantization_config)
        modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])

        target_is_cpu = target_device is not None and torch.device(target_device).type == "cpu"

        result = []
        i = 0
        while i < len(buffer):
            name, tensor = buffer[i]
            if not WeightSyncHandler._should_quantize_fp8_weight(name, tensor, modules_to_not_convert):
                result.append((name, tensor))
                i += 1
                continue

            group_end = i + 1
            if tensor.device.type == "cpu" or (target_is_cpu and tensor.device.type != "cpu"):
                while group_end < len(buffer):
                    next_name, next_tensor = buffer[group_end]
                    if not WeightSyncHandler._should_quantize_fp8_weight(
                        next_name,
                        next_tensor,
                        modules_to_not_convert,
                    ):
                        break
                    if not WeightSyncHandler._can_group_fp8_tensor(tensor, next_tensor, group_end - i):
                        break
                    group_end += 1

            if group_end > i + 1:
                rows, cols = tensor.shape
                stack = torch.as_strided(
                    tensor,
                    size=(group_end - i, rows, cols),
                    stride=(rows * cols, cols, 1),
                )
                quantized_stack, scale_stack = WeightSyncHandler._quantize_fp8_stack(
                    stack,
                    fp8_dtype=fp8_dtype,
                    fp8_max=fp8_max,
                    block_size_row=block_size_row,
                    block_size_col=block_size_col,
                    target_device=target_device,
                    phase_s=phase_s,
                    phase_prefix=phase_prefix,
                )
                for group_idx, (group_name, _) in enumerate(buffer[i:group_end]):
                    result.append((group_name, quantized_stack[group_idx]))
                    result.append((group_name.replace(".weight", ".weight_scale_inv"), scale_stack[group_idx]))
                i = group_end
                continue

            quantized, scale_inv = WeightSyncHandler._quantize_single_fp8_tensor(
                tensor,
                fp8_dtype=fp8_dtype,
                fp8_max=fp8_max,
                block_size_row=block_size_row,
                block_size_col=block_size_col,
                target_device=target_device,
                phase_s=phase_s,
                phase_prefix=phase_prefix,
            )
            result.append((name, quantized))
            result.append((name.replace(".weight", ".weight_scale_inv"), scale_inv))
            i += 1

        return result

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _extract_params_for_sync(
        fsdp_mod,
        mod_name: str,
        DTensor,
        skip_moe_prefixes: Optional[set] = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Extract parameters from an unsharded FSDP module for sync.

        Handles LoRA by computing merged weights (base + delta) on-the-fly,
        and skips LoRA-only params (lora_A, lora_B) since inference doesn't
        have them. Also skips DTensor params (belong to child FSDP modules)
        and EP-handled MoE modules (gathered separately).

        Supports both LoraLinear (dense LoRA) and MoEExpertsLoRA (MoE LoRA).
        """
        buffer = []

        # Identify child FSDP module prefixes — their params will be processed
        # when that child module is unsharded separately. Parent unshard may
        # expose child params as plain tensors, so we can't rely on DTensor check.

        child_fsdp_prefixes = set()
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, FSDPModule) and mname != "":
                child_fsdp_prefixes.add(mname + ".")

        # Collect LoRA modules for on-the-fly merging
        lora_modules = {}
        for mname, mod in fsdp_mod.named_modules():
            if isinstance(mod, LoraModule):
                lora_modules[mname] = mod

        # Track which params are LoRA A/B or QLoRA internal so we skip them
        lora_param_names = set()
        for mname, mod in lora_modules.items():
            prefix = f"{mname}." if mname else ""
            if isinstance(mod, QLoRALinear):
                lora_param_names.add(f"{prefix}lora_A")
                lora_param_names.add(f"{prefix}lora_B")
                lora_param_names.add(f"{prefix}packed_weight_f32")
            elif isinstance(mod, LoraLinear):
                lora_param_names.add(f"{prefix}lora_A")
                lora_param_names.add(f"{prefix}lora_B")
            elif isinstance(mod, (QLoRAMoeExperts, MoEExpertsLoRA)):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    lora_param_names.add(f"{prefix}{proj}_lora_A")
                    lora_param_names.add(f"{prefix}{proj}_lora_B")

        for pname, param in fsdp_mod.named_parameters():
            # Skip params owned by child FSDP modules (processed separately)
            if any(pname.startswith(p) for p in child_fsdp_prefixes):
                continue
            # Skip DTensor params (EP-sharded or other distributed tensors)
            if isinstance(param.data, DTensor):
                continue
            # Skip EP-handled MoE expert params (gathered separately)
            if skip_moe_prefixes:
                if "" in skip_moe_prefixes:
                    logger.info(f"[SKIP-EP] Skipping ALL param: {pname} (skip_moe_prefixes={''})")
                    continue  # entire FSDP module is EP MoE — skip all
                if any(pname.startswith(p + ".") or pname == p for p in skip_moe_prefixes):
                    continue
            elif pname in ("gate_proj", "up_proj", "down_proj"):
                logger.info(f"[NO-SKIP] MoE param NOT skipped: {pname}, skip_moe_prefixes={skip_moe_prefixes}")
            # Skip LoRA-only params — they'll be merged into base weight
            if pname in lora_param_names:
                continue

            full_name = f"{mod_name}.{pname}" if mod_name != "(root)" else pname

            # Check if this is a base weight with LoRA to merge
            # Case 1: LoraLinear — pname like "self_attn.q_proj.weight"
            #   parent module "self_attn.q_proj" is the LoraLinear
            parent_name = ".".join(pname.split(".")[:-1])
            param_leaf = pname.split(".")[-1]  # e.g. "weight", "gate_proj"

            if parent_name in lora_modules:
                lora_mod = lora_modules[parent_name]
                if isinstance(lora_mod, LoraLinear):
                    delta = lora_mod.get_delta_weight()
                    merged = param.data.to(dtype=torch.bfloat16) + delta.to(dtype=torch.bfloat16)
                    cloned = merged.clone()
                    buffer.append((full_name, cloned))
                    continue
                elif isinstance(lora_mod, MoEExpertsLoRA):
                    if param_leaf == "gate_up_proj":
                        merged = param.data.to(dtype=torch.bfloat16).clone()
                        half = merged.shape[2] // 2
                        if "gate_proj" in lora_mod.lora_config.target_modules:
                            gate_delta = lora_mod._compute_proj_delta("gate_proj")
                            merged[:, :, :half].add_(gate_delta.to(dtype=torch.bfloat16))
                        if "up_proj" in lora_mod.lora_config.target_modules:
                            up_delta = lora_mod._compute_proj_delta("up_proj")
                            merged[:, :, half:].add_(up_delta.to(dtype=torch.bfloat16))
                        buffer.append((full_name, merged))
                        continue
                    if param_leaf == "down_proj":
                        if "down_proj" in lora_mod.lora_config.target_modules:
                            delta = lora_mod._compute_proj_delta("down_proj")
                            merged = param.data.to(dtype=torch.bfloat16) + delta.to(dtype=torch.bfloat16)
                        else:
                            merged = param.data.to(dtype=torch.bfloat16)
                        buffer.append((full_name, merged.clone()))
                        continue
            # Check if this is a non-LoRA MoEExperts fused tensor
            _is_moe_experts = False
            if parent_name:
                # Walk up to find parent module
                parts_list = parent_name.split(".")
                parent_mod = fsdp_mod
                try:
                    for p in parts_list:
                        parent_mod = getattr(parent_mod, p)
                    if isinstance(parent_mod, (MoEExperts, MoEExpertsLoRA)):
                        if param_leaf in ("gate_up_proj", "down_proj"):
                            _is_moe_experts = True
                except (AttributeError, TypeError):
                    pass

            if _is_moe_experts:
                cloned_moe = param.data.to(dtype=torch.bfloat16).clone()
                buffer.append((full_name, cloned_moe))
            else:
                cloned = param.data.to(dtype=torch.bfloat16).clone()
                buffer.append((full_name, cloned))

        # Handle tied weights (e.g. tie_word_embeddings: lm_head.weight == embed_tokens.weight).
        # named_parameters() deduplicates by identity, so tied weights are only yielded once.
        # We need to emit them under all names so inference engines get all expected params.
        tied_keys = getattr(fsdp_mod, "_tied_weights_keys", None)
        if tied_keys:
            buffer_names = {n for n, _ in buffer}
            for tied_name, source_name in tied_keys.items():
                full_tied = f"{mod_name}.{tied_name}" if mod_name != "(root)" else tied_name
                full_source = f"{mod_name}.{source_name}" if mod_name != "(root)" else source_name
                if full_tied not in buffer_names and full_source in buffer_names:
                    for buf_name, buf_tensor in buffer:
                        if buf_name == full_source:
                            logger.info(
                                f"Rank 0: [WeightSync] Tied weight: emitting {full_tied} (clone of {full_source})"
                            )
                            buffer.append((full_tied, buf_tensor.clone()))
                            break

        return buffer

    @staticmethod
    def _unfuse_for_inference(
        buffer: List[Tuple[str, torch.Tensor]],
        model,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Convert training parameter names to HF/SGLang inference names.

        Handles:
        - qkv_proj → q_proj + k_proj + v_proj (split fused attention)
        - gate_up_proj → gate_proj + up_proj (split fused dense/shared MLP)
        - MoE experts: gate_up_proj/down_proj → per-expert HF gate/up/down weights
        - DeepseekV3 / Kimi-K2.5 MLA: q_a_proj + kv_a_proj_with_mqa →
          fused_qkv_a_proj_with_mqa to match SGLang's inference module
        - Qwen3.5 linear attention: remap split GatedDeltaNet params back to
          HF fused names (q_proj/k_proj/v_proj → in_proj_qkv, etc.)
        """

        config = model.config
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        result = []
        for name, tensor in buffer:
            if ".qkv_proj." in name:
                prefix, suffix = name.rsplit(".qkv_proj.", 1)
                q = tensor[:q_size].clone()
                k = tensor[q_size : q_size + kv_size].clone()
                v = tensor[q_size + kv_size :].clone()
                result.append((f"{prefix}.q_proj.{suffix}", q))
                result.append((f"{prefix}.k_proj.{suffix}", k))
                result.append((f"{prefix}.v_proj.{suffix}", v))
            elif name.endswith(".mlp.experts.gate_up_proj"):
                prefix = name.rsplit(".gate_up_proj", 1)[0]
                half = tensor.shape[2] // 2
                gate = tensor[:, :, :half].transpose(1, 2).contiguous()
                up = tensor[:, :, half:].transpose(1, 2).contiguous()
                for expert_idx in range(tensor.shape[0]):
                    result.append((f"{prefix}.{expert_idx}.gate_proj.weight", gate[expert_idx]))
                    result.append((f"{prefix}.{expert_idx}.up_proj.weight", up[expert_idx]))
            elif name.endswith(".mlp.experts.down_proj"):
                prefix = name.rsplit(".down_proj", 1)[0]
                down = tensor.transpose(1, 2).contiguous()
                for expert_idx in range(tensor.shape[0]):
                    result.append((f"{prefix}.{expert_idx}.down_proj.weight", down[expert_idx]))
            elif ".gate_up_proj." in name:
                prefix, suffix = name.rsplit(".gate_up_proj.", 1)
                half = tensor.shape[0] // 2
                gate = tensor[:half].clone()
                up = tensor[half:].clone()
                result.append((f"{prefix}.gate_proj.{suffix}", gate))
                result.append((f"{prefix}.up_proj.{suffix}", up))
            else:
                result.append((name, tensor))

        result = WeightSyncHandler._remap_deepseek_mla_params_for_inference(result, config)

        if has_linear_attention_layers(config):
            result = remap_linear_attention_params_for_inference(result)

        return result

    @staticmethod
    def _remap_deepseek_mla_params_for_inference(
        buffer: List[Tuple[str, torch.Tensor]],
        config,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Fuse DeepseekV3/Kimi-K2.5 MLA A projections for SGLang receivers."""
        if getattr(config, "q_lora_rank", None) is None:
            return buffer

        result: List[Tuple[str, torch.Tensor]] = []
        pending: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

        for name, tensor in buffer:
            if ".self_attn.q_a_proj." in name:
                prefix, suffix = name.rsplit(".q_a_proj.", 1)
                pending.setdefault((prefix, suffix), {})["q_a_proj"] = tensor
                continue
            if ".self_attn.kv_a_proj_with_mqa." in name:
                prefix, suffix = name.rsplit(".kv_a_proj_with_mqa.", 1)
                pending.setdefault((prefix, suffix), {})["kv_a_proj_with_mqa"] = tensor
                continue
            result.append((name, tensor))

        for (prefix, suffix), parts in pending.items():
            q_a = parts.get("q_a_proj")
            kv_a = parts.get("kv_a_proj_with_mqa")
            if suffix == "weight" and q_a is not None and kv_a is not None:
                result.append(
                    (
                        f"{prefix}.fused_qkv_a_proj_with_mqa.{suffix}",
                        torch.cat([q_a, kv_a], dim=0).contiguous(),
                    )
                )
                continue
            if q_a is not None:
                result.append((f"{prefix}.q_a_proj.{suffix}", q_a))
            if kv_a is not None:
                result.append((f"{prefix}.kv_a_proj_with_mqa.{suffix}", kv_a))

        return result

    def _broadcast_buffer(
        self,
        backend,
        buffer: List[Tuple[str, torch.Tensor]],
        flush_cache: bool,
        weight_version: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Broadcast a buffer of (name, tensor) pairs to inference endpoints.

        Uses the transport backend's transfer_bucket for HTTP coordination
        + data transfer.

        Returns:
            (total_bytes, num_params)
        """
        if not buffer:
            return 0, 0

        bucket_bytes = sum(t.numel() * t.element_size() for _, t in buffer)
        logger.info(f"Rank {self.rank}: [WeightSync] Broadcasting {len(buffer)} params, {bucket_bytes / 1e6:.1f} MB")

        backend.transfer_bucket(
            buffer,
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        return bucket_bytes, len(buffer)

    @staticmethod
    def _get_fsdp_modules(model) -> Tuple[Optional[Any], List[Tuple[str, Any]]]:
        """
        Discover FSDP modules in the model.

        Returns:
            (root_fsdp_module_or_none, [(fqn, fsdp_module), ...])

        Root module holds non-layer params (embed, norm, head).
        Layer modules hold per-layer params.
        """

        root_module = None
        layer_modules = []

        for fqn, mod in model.named_modules():
            if not isinstance(mod, FSDPModule):
                continue
            if fqn == "":
                root_module = mod
            else:
                layer_modules.append((fqn, mod))

        return root_module, layer_modules
