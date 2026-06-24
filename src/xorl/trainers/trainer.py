"""Offline training loop for xorl.

Follows torchtitan's pattern: explicit train_step(), minimal TrainState,
PP/non-PP dispatch via forward_backward_step(). No callback system.

Usage:
    trainer = Trainer(args)
    trainer.train()
"""

import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp_meta
import torch.distributed.tensor._random
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from tqdm import trange

from xorl.arguments import Arguments, save_args
from xorl.checkpoint import build_checkpointer
from xorl.data.constants import IGNORE_INDEX
from xorl.data.data_loader import DataLoaderBuilder
from xorl.data.prepare.prepare_datasets import prepare_datasets
from xorl.distributed.gradient_accumulate_loss import gradient_accumulate_loss
from xorl.distributed.offloading import build_activation_offloading_context
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.pipeline_parallel import build_pipeline_schedule, build_pp_stage
from xorl.distributed.sync_padding import synchronize_micro_batch_padding
from xorl.distributed.torch_parallelize import build_parallelize_model
from xorl.lora.utils import (
    get_lora_state_dict,
    inject_lora_into_model,
    inject_lora_into_model_with_moe,
    save_lora_checkpoint,
)
from xorl.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules
from xorl.models.layers.moe.aux_loss import LoadBalancingBuffer, global_load_balancing_loss_func
from xorl.models.layers.moe.routing_replay import RoutingReplay, set_replay_stage
from xorl.models.module_utils import compute_loss
from xorl.models.transformers.deepseek_v3.support import (
    freeze_deepseek_v3_router_parameters,
    validate_deepseek_v3_training_mode,
)
from xorl.models.transformers.glm5.support import validate_glm5_training_mode
from xorl.optim import build_lr_scheduler, build_optimizer
from xorl.qlora import (
    detect_prequantized_block_fp8,
    detect_prequantized_nvfp4,
    inject_qlora_into_model,
    maybe_load_and_quantize_moe_qlora,
    maybe_load_prequantized_qlora,
    maybe_quantize_qlora,
)
from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp
from xorl.trainers.model_builder import (
    maybe_upcast_trainable_adapter_params,
    resolve_training_model_dtype,
    should_skip_generic_param_upcast,
)
from xorl.trainers.per_component_timer import PerComponentTimer
from xorl.trainers.training_utils import (
    clip_gradients,
    count_active_microbatches,
    count_valid_tokens,
    forward_backward_pp,
    get_distsign_grad_scale_factor,
    get_effective_grad_clip_value,
    make_pp_loss_fn,
    maybe_merge_lora,
    negotiate_pp_seq_len,
    pad_micro_batches_for_pp,
    scale_model_gradients,
    sync_lm_head_tp_gradient,
    sync_sp_gradients,
)
from xorl.utils import helper
from xorl.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_process_group_timeout,
    get_torch_device,
    synchronize,
)
from xorl.utils.dist_utils import all_reduce, distributed_barrier, get_cpu_world_group


logger = helper.create_logger(__name__)
_trainer_cpu_group: Optional[dist.ProcessGroup] = None


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def _memory_trace_rank_enabled(rank: int, ranks: str) -> bool:
    ranks = ranks.strip().lower()
    if ranks in {"", "all", "*"}:
        return True
    return str(rank) in {r.strip() for r in ranks.split(",") if r.strip()}


def _trainer_memory_trace(stage: str, *, force: bool = False) -> None:
    if not torch.cuda.is_available():
        return
    enabled = _env_flag("XORL_TRAINER_MEMORY_TRACE")
    if not enabled and not force:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", "0"))
    ranks = os.environ.get(
        "XORL_TRAINER_MEMORY_TRACE_RANKS",
        os.environ.get("XORL_QUACK_EP_MEMORY_TRACE_RANKS", "all"),
    )
    if not force and not _memory_trace_rank_enabled(rank, ranks):
        return
    min_allocated = int(
        float(
            os.environ.get(
                "XORL_TRAINER_MEMORY_TRACE_MIN_ALLOCATED_GB",
                os.environ.get("XORL_QUACK_EP_MEMORY_TRACE_MIN_ALLOCATED_GB", "0"),
            )
        )
        * (1024**3)
    )
    allocated = torch.cuda.memory_allocated()
    if not force and allocated < min_allocated:
        return
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    free, total = torch.cuda.mem_get_info()
    print(
        f"[TrainerMem r{rank}] {stage}: "
        f"alloc={allocated / (1024**3):.2f}GiB reserved={reserved / (1024**3):.2f}GiB "
        f"peak={peak / (1024**3):.2f}GiB free={free / (1024**3):.2f}GiB "
        f"total={total / (1024**3):.2f}GiB",
        flush=True,
    )


_HOST_INVENTORY_MAX_WORLD_SIZE = int(os.environ.get("XORL_HOST_INVENTORY_MAX_WORLD_SIZE", "64"))
_HOST_INVENTORY_DISABLED = os.environ.get("XORL_DISABLE_HOST_INVENTORY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


_STEP_PHASE_TIMING_ORDER = (
    "count_valid_tokens",
    "count_active_microbatches",
    "optimizer_zero_grad",
    "environ_meter_add",
    "pp_negotiate_seq_len",
    "pp_pad_microbatches",
    "microbatch_to_device",
    "model_forward",
    "fwd_norm/input",
    "fwd_attn/total",
    "fwd_attn/indexer",
    "fwd_norm/post_attn",
    "fwd_mlp_or_moe/total",
    "fwd_moe/gate",
    "fwd_moe/experts",
    "fwd_moe/shared",
    "loss_compute",
    "backward",
    "bwd_norm/input",
    "bwd_attn/total",
    "bwd_attn/indexer",
    "bwd_norm/post_attn",
    "bwd_mlp_or_moe/total",
    "bwd_moe/gate",
    "bwd_moe/experts",
    "bwd_moe/shared",
    "recompute_norm/input",
    "recompute_attn/total",
    "recompute_attn/indexer",
    "recompute_norm/post_attn",
    "recompute_mlp_or_moe/total",
    "recompute_moe/gate",
    "recompute_moe/experts",
    "recompute_moe/shared",
    "loss_item",
    "pp_forward_backward",
    "pp_grad_scale",
    "forward_backward_total",
    "routing_replay_clear",
    "sync_sp_gradients",
    "distsign_grad_scale",
    "clip_gradients",
    "optimizer_step",
    "lr_scheduler_step",
    "clip_and_step_total",
    "maybe_merge_lora",
    "reduce_metrics",
    "train_step_total",
)


def _order_step_phases(phase_keys) -> List[str]:
    """Return phase keys sorted by `_STEP_PHASE_TIMING_ORDER`, with unknown keys appended sorted."""
    keys = list(phase_keys)
    known = [name for name in _STEP_PHASE_TIMING_ORDER if name in keys]
    known.extend(name for name in sorted(keys) if name not in _STEP_PHASE_TIMING_ORDER)
    return known


def _summarize_phase_times_local(phase_times: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Build a {phase: {local, mean, max, min}} summary with every aggregate equal to the local value.

    Used on the exception path so we can log diagnostics without another collective.
    """
    if not phase_times:
        return {}
    return {
        phase: {
            "local": float(phase_times[phase]),
            "mean": float(phase_times[phase]),
            "max": float(phase_times[phase]),
            "min": float(phase_times[phase]),
        }
        for phase in _order_step_phases(phase_times)
    }


def _summarize_memory_stats_local(
    memory_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build a {phase: {metric: {local, mean, max, min}}} summary with aggregates equal to local."""
    if not memory_stats:
        return {}
    return {
        phase: {
            metric: {
                "local": float(value),
                "mean": float(value),
                "max": float(value),
                "min": float(value),
            }
            for metric, value in memory_stats[phase].items()
        }
        for phase in _order_step_phases(memory_stats)
    }


def _reset_cuda_peak_memory_stats() -> None:
    if get_device_type() == "cuda":
        get_torch_device().reset_peak_memory_stats()


def _cuda_max_memory_allocated() -> int:
    if get_device_type() != "cuda":
        return 0
    return get_torch_device().max_memory_allocated()


def _get_trainer_cpu_group() -> Optional[dist.ProcessGroup]:
    """Return a cached CPU/Gloo group for object collectives in trainer bootstrap."""
    global _trainer_cpu_group
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return None
    if _trainer_cpu_group is None:
        _trainer_cpu_group = dist.new_group(backend="gloo")
    return _trainer_cpu_group


def _should_collect_host_inventory(world_size: int) -> bool:
    if _HOST_INVENTORY_DISABLED:
        return False
    return world_size <= _HOST_INVENTORY_MAX_WORLD_SIZE


# ---------------------------------------------------------------------------
# TrainState — checkpointable training state
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Minimal checkpointable state (follows torchtitan's Stateful pattern)."""

    global_step: int = 0
    epoch: int = 0
    start_step: int = 0  # step within current epoch to resume from
    loss_history: List[float] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "start_step": self.start_step,
            "loss_history": self.loss_history,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.global_step = d.get("global_step", 0)
        self.epoch = d.get("epoch", 0)
        self.start_step = d.get("start_step", 0)
        self.loss_history = d.get("loss_history", [])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Offline training loop. Analogous to torchtitan's Trainer.

    Lifecycle::

        trainer = Trainer(args)   # builds model, optimizer, data, etc.
        trainer.train()           # runs epochs; saves final HF weights
    """

    def __init__(self, args: Arguments) -> None:
        self.args = args
        self.state = TrainState()
        self._setup_phase_metrics: Dict[str, float] = {}
        self._startup_metrics: Dict[str, Any] = {}
        self._wandb_initialized = False
        self._current_step_phase_times: Optional[Dict[str, float]] = None
        self._last_step_phase_times: Dict[str, Dict[str, float]] = {}
        self._current_step_memory_stats: Optional[Dict[str, Dict[str, float]]] = None
        self._last_step_memory_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._per_component_timer = PerComponentTimer(
            enabled=args.train.enable_step_phase_timing and args.train.enable_per_component_timing
        )

        # Setup phases (order matters — each depends on previous)
        # Model is built before data: reads weights from disk while I/O is free,
        # before dataset loading competes for bandwidth.
        def _timed(phase_name, fn):
            t0 = time.time()
            fn()
            elapsed = time.time() - t0
            self._setup_phase_metrics[f"startup/{phase_name}_sec"] = elapsed
            logger.info(f"[TIMING] {phase_name} took {elapsed:.1f}s (rank {args.train.local_rank})")
            self._maybe_log_startup_metrics({f"startup/{phase_name}_sec": elapsed}, commit=False)
            self._log_memory_snapshot(f"startup/{phase_name}")

        _timed("bootstrap", self._bootstrap)
        if args.train.prewarm_cuda_blas:
            _timed("prewarm_cuda_blas", self._prewarm_cuda_blas)
        _timed("build_model", self._build_model)
        _timed("parallelize", self._parallelize)
        _timed("build_data", self._build_data)
        _timed("build_optimizer", self._build_optimizer)
        _timed("setup_observability", self._setup_observability)
        _timed("resume_checkpoint", self._resume_checkpoint)
        _timed("build_pp_schedule", self._init_pp_schedule_cache)
        self._write_startup_metrics_file()

    # ===================================================================
    # Setup phases
    # ===================================================================

    def _bootstrap(self) -> None:
        """Initialize distributed, device, seed, parallel state."""
        args = self.args

        get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
        dist.init_process_group(backend=get_nccl_backend(), timeout=get_process_group_timeout())
        logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
        logger.info_rank0(json.dumps(asdict(args), indent=2))

        helper.set_seed(args.train.seed, args.train.enable_full_determinism)

        if args.train.local_rank == 0:
            helper.enable_third_party_logging()

        if args.train.global_rank == 0:
            save_args(args, args.train.output_dir)
            if args.train.use_wandb:
                import wandb  # noqa: PLC0415

                wandb.init(
                    project=args.train.wandb_project,
                    name=args.train.wandb_name,
                    tags=args.train.wandb_tags,
                    config={**vars(args.model), **vars(args.data), **vars(args.train)},
                )
                self._wandb_initialized = True
                config_file = os.path.join(args.train.output_dir, "xorl_cli.yaml")
                if os.path.exists(config_file):
                    wandb.save(config_file, policy="now")
                self._maybe_log_startup_metrics(
                    {
                        "startup/world_size": args.train.world_size,
                        "startup/global_batch_size": args.train.global_batch_size,
                        "startup/micro_batch_size": args.train.micro_batch_size,
                        "startup/gradient_accumulation_steps": args.train.gradient_accumulation_steps,
                        "startup/data_parallel_size": args.train.data_parallel_size,
                        "startup/data_parallel_replicate_size": args.train.data_parallel_replicate_size,
                        "startup/data_parallel_shard_size": args.train.data_parallel_shard_size,
                        "startup/ulysses_parallel_size": args.train.ulysses_parallel_size,
                        "startup/expert_parallel_size": args.train.expert_parallel_size,
                        "startup/pipeline_parallel_size": args.train.pipeline_parallel_size,
                    },
                    commit=False,
                )
        self._log_host_inventory()

        self.Checkpointer = build_checkpointer(
            dist_backend=args.train.data_parallel_mode,
            ckpt_manager=args.train.ckpt_manager,
        )

        init_parallel_state(
            dp_size=args.train.data_parallel_size,
            dp_replicate_size=args.train.data_parallel_replicate_size,
            dp_shard_size=args.train.data_parallel_shard_size,
            tp_size=args.train.tensor_parallel_size,
            ep_size=args.train.expert_parallel_size,
            pp_size=args.train.pipeline_parallel_size,
            ulysses_size=args.train.ulysses_parallel_size,
            ringattn_size=args.train.ringattn_parallel_size,
            lm_head_tp_size=args.train.lm_head_tensor_parallel_size,
            dp_mode=args.train.data_parallel_mode,
            cp_fsdp_mode=args.train.cp_fsdp_mode,
            ep_intranode=args.train.ep_intranode,
        )
        get_cpu_world_group()

        # DTensor RNG tracker (run_state_sync=False to avoid PP deadlock)
        self.ps = get_parallel_state()
        if self.ps.device_mesh is not None:
            torch.distributed.tensor._random.manual_seed(args.train.seed, self.ps.device_mesh)

        # Routing replay is only needed with EP when MoE forward is recomputed
        self._use_routing_replay = self.ps.ep_size > 1 and args.train.moe_recomputed

        # Loss-function kwargs forwarded to causallm_loss_function each step.
        self._causallm_loss_params: Dict[str, Any] = {
            "ce_mode": args.train.ce_mode,
            "num_chunks": args.train.ce_num_chunks,
        }
        if args.train.fsdp_sharded_lm_head_loss:
            self._causallm_loss_params["fsdp_sharded_lm_head_loss_num_chunks"] = (
                args.train.fsdp_sharded_lm_head_loss_num_chunks
            )
        if args.train.softmax_auxiliary_loss:
            if args.train.pipeline_parallel_size > 1:
                raise NotImplementedError(
                    "softmax_auxiliary_loss (Z-loss) is not yet supported with pipeline parallelism. "
                    "PP uses a separate compiled CE loss path (pp_loss_fn) that does not compute logsumexp."
                )
            self._causallm_loss_params["z_loss_coef"] = args.train.auxiliary_loss_multiplier

        # Global-batch MoE load balancing: accumulate expert-selection frequencies across
        # the gradient-accumulation window instead of balancing per micro-batch (Qiu et al.
        # 2025, https://arxiv.org/abs/2501.11873). The PP path does not compute the aux loss.
        self._moe_global_load_balance = args.train.moe_global_load_balancing

    def _prewarm_cuda_blas(self) -> None:
        """Initialize CUDA BLAS handles on the checkpoint recompute path."""
        if get_device_type() != "cuda":
            return

        from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

        args = self.args
        device = torch.device(f"cuda:{args.train.local_rank}")
        dtypes = [torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

        def _linear_gelu(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(torch.nn.functional.linear(inp, weight))

        for dtype in dtypes:
            x = torch.ones((64, 64), device=device, dtype=dtype, requires_grad=True)
            w = torch.ones((64, 64), device=device, dtype=dtype, requires_grad=True)
            y = checkpoint(_linear_gelu, x, w, use_reentrant=False)
            y.float().sum().backward()
            del x, w, y

        torch.cuda.synchronize(device)
        helper.empty_cache()
        logger.info(f"Prewarmed CUDA BLAS handles via checkpointed autograd (rank {args.train.local_rank})")

    def _maybe_log_startup_metrics(self, metrics: Dict[str, Any], commit: bool = False) -> None:
        """Log startup metrics to wandb once rank 0 has initialized it."""
        if not metrics:
            return
        if self.args.train.global_rank == 0:
            self._startup_metrics.update(metrics)
        if self.args.train.global_rank != 0 or not self.args.train.use_wandb or not self._wandb_initialized:
            return
        import wandb  # noqa: PLC0415

        wandb.log(metrics, step=0, commit=commit)

    def _write_startup_metrics_file(self) -> None:
        """Persist startup metrics alongside the run outputs for offline inspection."""
        if self.args.train.global_rank != 0:
            return

        payload = {
            "repo_commit": self.args.train.repo_commit,
            "wandb_project": self.args.train.wandb_project if self.args.train.use_wandb else None,
            "wandb_name": self.args.train.wandb_name if self.args.train.use_wandb else None,
            "metrics": self._startup_metrics,
        }
        output_path = os.path.join(self.args.train.output_dir, "startup_metrics.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _collect_host_inventory(self) -> List[Dict[str, Any]]:
        """Gather rank-to-host mapping across all processes."""
        payload = {
            "global_rank": self.args.train.global_rank,
            "local_rank": self.args.train.local_rank,
            "hostname": socket.gethostname(),
        }
        if not dist.is_available() or not dist.is_initialized() or self.args.train.world_size <= 1:
            return [payload]
        if not _should_collect_host_inventory(self.args.train.world_size):
            return [payload]
        gathered: List[Optional[Dict[str, Any]]] = [None] * self.args.train.world_size
        dist.all_gather_object(gathered, payload, group=_get_trainer_cpu_group())
        return [item for item in gathered if item is not None]

    def _log_host_inventory(self) -> None:
        """Emit host inventory to stdout and wandb config on rank 0."""
        inventory = self._collect_host_inventory()
        if self.args.train.global_rank != 0:
            return

        if self.args.train.world_size > 1 and not _should_collect_host_inventory(self.args.train.world_size):
            logger.info_rank0(
                "Skipping host inventory gather for world_size=%s; "
                "set XORL_HOST_INVENTORY_MAX_WORLD_SIZE or XORL_DISABLE_HOST_INVENTORY to override.",
                self.args.train.world_size,
            )
            self._startup_metrics.update(
                {
                    "startup/master_addr": os.environ.get("MASTER_ADDR"),
                    "startup/master_port": os.environ.get("MASTER_PORT"),
                    "startup/host_inventory_skipped": True,
                    "startup/host_inventory_world_size": self.args.train.world_size,
                }
            )
            return

        unique_hostnames = sorted({item["hostname"] for item in inventory})
        rank_to_hostname = {str(item["global_rank"]): item["hostname"] for item in inventory}
        logger.info_rank0(
            "Host inventory:\n"
            + json.dumps(
                {
                    "master_addr": os.environ.get("MASTER_ADDR"),
                    "master_port": os.environ.get("MASTER_PORT"),
                    "node_count": len(unique_hostnames),
                    "hostnames": unique_hostnames,
                    "ranks": inventory,
                },
                indent=2,
            )
        )
        self._startup_metrics.update(
            {
                "startup/master_addr": os.environ.get("MASTER_ADDR"),
                "startup/master_port": os.environ.get("MASTER_PORT"),
                "startup/node_count": len(unique_hostnames),
                "startup/hostnames": unique_hostnames,
                "startup/rank_to_hostname": rank_to_hostname,
            }
        )
        self._maybe_log_startup_metrics({"startup/node_count": len(unique_hostnames)}, commit=False)
        if self.args.train.use_wandb and self._wandb_initialized:
            import wandb  # noqa: PLC0415

            wandb.config.update(
                {
                    "master_addr": os.environ.get("MASTER_ADDR"),
                    "master_port": os.environ.get("MASTER_PORT"),
                    "hostnames": unique_hostnames,
                    "rank_to_hostname": rank_to_hostname,
                },
                allow_val_change=True,
            )

    def _log_memory_snapshot(self, prefix: str) -> None:
        """Capture a coarse memory snapshot during setup for wandb comparison."""
        if not self.args.train.use_wandb:
            return

        device = get_torch_device()
        allocated_memory = device.memory_allocated()
        reserved_memory = device.memory_reserved()
        max_allocated_memory = device.max_memory_allocated()
        max_reserved_memory = device.max_memory_reserved()

        allocated_memory, reserved_memory, max_allocated_memory, max_reserved_memory = all_reduce(
            (allocated_memory, reserved_memory, max_allocated_memory, max_reserved_memory),
            op="max",
        )
        if self.args.train.global_rank != 0 or not self._wandb_initialized:
            return
        self._maybe_log_startup_metrics(
            {
                f"{prefix}/gpu_allocated_gb": allocated_memory / (1024**3),
                f"{prefix}/gpu_reserved_gb": reserved_memory / (1024**3),
                f"{prefix}/gpu_max_allocated_gb": max_allocated_memory / (1024**3),
                f"{prefix}/gpu_max_reserved_gb": max_reserved_memory / (1024**3),
            },
            commit=False,
        )

    def _build_data(self) -> None:
        """Build tokenizer, datasets, dataloader, compute step counts."""
        args = self.args
        logger.info_rank0("Prepare data")
        self.tokenizer = build_tokenizer(args.model.tokenizer_path)
        train_dataset, _ = prepare_datasets(args, self.tokenizer)

        self.train_dataloader = DataLoaderBuilder(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            gradient_accumulation_steps=args.train.gradient_accumulation_steps,
            num_workers=args.data.dataloader_num_workers,
            drop_last=args.data.dataloader_drop_last,
            pin_memory=args.data.dataloader_pin_memory,
            prefetch_factor=args.data.dataloader_prefetch_factor,
            seed=args.train.seed,
            pad_to_multiple_of=args.data.pad_to_multiple_of,
            fa_max_length_bucket=args.data.fa_max_length_bucket,
        ).build()

        self.train_steps_per_epoch = len(self.train_dataloader)
        self.total_train_steps = self.train_steps_per_epoch * args.train.num_train_epochs
        if args.train.max_steps is not None:
            self.total_train_steps = min(self.total_train_steps, args.train.max_steps)
        logger.info_rank0(
            f"Train steps per epoch: {self.train_steps_per_epoch}, Total train steps: {self.total_train_steps}"
        )

        self.save_epoch_steps = (
            int(args.train.save_epochs * self.train_steps_per_epoch) if args.train.save_epochs else 0
        )
        if self.save_epoch_steps:
            logger.info_rank0(f"Save every {args.train.save_epochs} epoch(s) = every {self.save_epoch_steps} steps")

    def _build_model(self) -> None:
        """Build foundation model and inject LoRA/QLoRA if configured."""
        args = self.args
        logger.info_rank0("Prepare model")
        model_dtype = resolve_training_model_dtype(
            enable_lora=args.lora.enable_lora,
            enable_qlora=args.lora.enable_qlora,
            enable_mixed_precision=args.train.enable_mixed_precision,
            skip_param_upcast=args.train.skip_param_upcast,
        )
        self.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype=model_dtype,
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            ep_dispatch=args.model.ep_dispatch,
            train_router=args.model.train_router,
            record_routing_weights=args.model.record_routing_weights,
            deepep_buffer_size_gb=args.model.deepep_buffer_size_gb,
            deepep_num_sms=args.model.deepep_num_sms,
            deepep_async_combine=args.model.deepep_async_combine,
            router_fp32=args.model.router_fp32,
            lm_head_fp32=args.model.lm_head_fp32,
            alltoall_combine_hidden_chunk_size=args.model.alltoall_combine_hidden_chunk_size,
            rmsnorm_mode=args.model.rmsnorm_mode,
            activation_native=args.model.activation_native,
            rope_native=args.model.rope_native,
            attention_cast_bf16=args.model.attention_cast_bf16,
            sparse_mla_enabled=args.model.sparse_mla_enabled,
            sparse_mla_backend=args.model.sparse_mla_backend,
            flash_attention_deterministic=args.model.flash_attention_deterministic,
            init_device=args.train.init_device,
        )
        self.model_config = self.model.config
        # Normalize _no_split_modules to list — some HF models (e.g. GPT-OSS) define it as a set
        if isinstance(getattr(self.model, "_no_split_modules", None), set):
            self.model._no_split_modules = list(self.model._no_split_modules)
        validate_deepseek_v3_training_mode(
            self.model_config,
            enable_qlora=args.lora.enable_qlora,
            freeze_router=args.model.freeze_router,
            merge_qkv=args.model.merge_qkv,
        )
        validate_glm5_training_mode(
            self.model_config,
            enable_qlora=args.lora.enable_qlora,
            freeze_router=args.model.freeze_router,
            merge_qkv=args.model.merge_qkv,
        )
        helper.print_device_mem_info("VRAM usage after building model")

        # Unfuse QKV for tensor parallelism
        if not args.model.merge_qkv:
            for layer in self.model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "unfuse_for_tp"):
                    layer.self_attn.unfuse_for_tp()
            logger.info_rank0("Unfused QKV projections (merge_qkv=False)")

        if (
            get_parallel_state().tp_enabled
            and hasattr(self.model, "unfuse_for_tp")
            and not getattr(self.model, "_unfused_for_tp", False)
        ):
            logger.info_rank0("Unfusing projections before FP8 training injection for tensor parallelism")
            self.model.unfuse_for_tp()

        # FP8 full-weight / QLoRA / LoRA injection
        self.is_prequantized = False
        self.checkpoint_quant_format = None
        self.exclude_modules = set()
        if args.train.enable_fp8_training and (args.lora.enable_lora or args.lora.enable_qlora):
            raise ValueError("enable_fp8_training is a full-weight mode and cannot be combined with LoRA or QLoRA")
        if args.train.enable_qarl and (args.lora.enable_lora or args.lora.enable_qlora):
            raise ValueError("enable_qarl is a full-weight mode and cannot be combined with LoRA or QLoRA")
        if args.train.enable_qarl and args.train.enable_fp8_training:
            raise ValueError(
                "enable_qarl cannot be combined with enable_fp8_training; choose one low-precision train path"
            )
        if args.train.enable_qarl and args.train.qarl_sync_format != "fp8":
            raise ValueError("Initial QARL supports only qarl_sync_format='fp8'")
        if args.train.enable_qarl and args.train.qarl_calib_size < 0:
            raise ValueError("qarl_calib_size must be non-negative")
        if args.train.enable_qarl and (
            args.train.qarl_quant_sequence_length is not None and args.train.qarl_quant_sequence_length <= 0
        ):
            raise ValueError("qarl_quant_sequence_length must be positive when set")
        if (
            args.train.enable_qarl
            and args.train.qarl_calib_data is None
            and (args.train.qarl_calib_size or args.train.qarl_quant_sequence_length is not None)
        ):
            raise ValueError("qarl_calib_size and qarl_quant_sequence_length require qarl_calib_data")

        if args.train.enable_fp8_training:
            from xorl.fp8_training import (  # noqa: PLC0415
                inject_fp8_training_into_model,
                validate_fp8_blackwell_training_policy,
            )

            validate_fp8_blackwell_training_policy(
                enable_fp8_training=True,
                allow_blackwell=args.train.fp8_training_allow_blackwell,
                validation_artifact=args.train.fp8_training_blackwell_validation_artifact,
            )

            inject_fp8_training_into_model(
                self.model,
                target_modules=args.train.fp8_training_target_modules,
                exclude_modules=args.train.fp8_training_exclude_modules,
                num_first_layers_bf16=args.train.fp8_training_num_first_layers_bf16,
                num_last_layers_bf16=args.train.fp8_training_num_last_layers_bf16,
                block_size=args.train.fp8_training_block_size,
                backward_mode=args.train.fp8_training_backward,
                smoothquant_alpha=args.train.fp8_training_smoothquant_alpha,
                lm_head_smoothquant_alpha=args.train.fp8_training_lm_head_smoothquant_alpha,
                activation_amax_scale=args.train.fp8_training_activation_amax_scale,
                weight_amax_scale=args.train.fp8_training_weight_amax_scale,
                correction_mode=args.train.fp8_training_correction_mode,
                module_overrides=args.train.fp8_training_module_overrides,
                allow_bf16_fallback=args.train.fp8_training_allow_bf16_fallback,
                moe_grouped_backend=args.train.fp8_training_moe_grouped_backend,
            )
            helper.print_device_mem_info("VRAM usage after FP8 training injection")
        elif args.train.enable_qarl:
            from xorl.qarl import calibrate_qarl_model, inject_qarl_into_model  # noqa: PLC0415

            inject_qarl_into_model(
                self.model,
                quant_cfg=args.train.qarl_quant_cfg,
                target_modules=args.train.qarl_target_modules,
                exclude_modules=args.train.qarl_exclude_modules,
            )
            if args.train.qarl_calib_data is not None:
                self.model._qarl_calibration_summary = calibrate_qarl_model(
                    self.model,
                    args.train.qarl_calib_data,
                    calibration_size=args.train.qarl_calib_size,
                    sequence_length=args.train.qarl_quant_sequence_length,
                )
            helper.print_device_mem_info("VRAM usage after QARL fake-quant injection")
        elif args.lora.enable_qlora:
            self._inject_qlora()
        elif args.lora.enable_lora:
            self._inject_lora()

        maybe_upcast_trainable_adapter_params(
            self.model,
            enable_lora=args.lora.enable_lora,
            enable_qlora=args.lora.enable_qlora,
            enable_mixed_precision=args.train.enable_mixed_precision,
        )

        # Save pre-hook before parallelization (some models register optimizer hooks)
        self._optimizer_pre_hook_fn = getattr(self.model, "get_optimizer_pre_hook", None)

    def _inject_qlora(self) -> None:
        """QLoRA injection with pre-quantized checkpoint detection."""
        args = self.args

        if detect_prequantized_nvfp4(args.model.model_path):
            self.is_prequantized = True
            self.checkpoint_quant_format = "nvfp4"
            logger.info_rank0("Detected pre-quantized NVFP4 checkpoint")
        elif detect_prequantized_block_fp8(args.model.model_path):
            self.is_prequantized = True
            self.checkpoint_quant_format = "block_fp8"
            logger.info_rank0("Detected pre-quantized block FP8 checkpoint")

        if args.lora.exclude_modules is not None:
            self.exclude_modules = set(args.lora.exclude_modules)
            logger.info_rank0(f"Using user-specified exclude_modules: {self.exclude_modules}")
        elif self.is_prequantized:
            self.exclude_modules = get_prequantized_exclude_modules(args.model.model_path)
            if self.exclude_modules:
                logger.info_rank0(
                    f"Auto-detected {len(self.exclude_modules)} excluded modules "
                    f"from checkpoint config: {self.exclude_modules}"
                )

        if self.is_prequantized and self.checkpoint_quant_format != args.lora.quant_format:
            logger.info_rank0(
                f"Cross-format conversion: checkpoint={self.checkpoint_quant_format}, "
                f"target={args.lora.quant_format} — will dequantize and re-quantize"
            )

        inject_qlora_into_model(
            self.model,
            r=args.lora.lora_rank,
            lora_alpha=args.lora.lora_alpha,
            quant_format=args.lora.quant_format,
            quant_group_size=args.lora.quant_group_size,
            target_modules=args.lora.lora_target_modules,
            checkpoint_quant_format=self.checkpoint_quant_format,
            merge_qkv=args.model.merge_qkv,
            exclude_modules=self.exclude_modules,
            enable_aqn=args.lora.enable_aqn,
            aqn_alpha=args.lora.aqn_alpha,
        )
        if self.exclude_modules:
            self.model._qlora_exclude_modules = self.exclude_modules
        helper.print_device_mem_info("VRAM usage after QLoRA injection")

    def _inject_lora(self) -> None:
        """Plain LoRA injection (dense + optional MoE-aware)."""
        args = self.args
        is_moe_model = getattr(self.model.config, "num_experts", 0) > 0

        if is_moe_model and args.lora.moe_hybrid_shared_lora:
            logger.info_rank0(f"MoE-aware LoRA injection (hybrid_shared={args.lora.moe_hybrid_shared_lora})")
            inject_lora_into_model_with_moe(
                self.model,
                r=args.lora.lora_rank,
                lora_alpha=args.lora.lora_alpha,
                target_modules=args.lora.lora_target_modules,
                moe_hybrid_shared_lora=args.lora.moe_hybrid_shared_lora,
            )
        else:
            inject_lora_into_model(
                self.model,
                r=args.lora.lora_rank,
                lora_alpha=args.lora.lora_alpha,
                target_modules=args.lora.lora_target_modules,
            )
        helper.print_device_mem_info("VRAM usage after LoRA injection")

    def _parallelize(self) -> None:
        """Apply FSDP2/PP wrapping and deferred QLoRA quantization."""
        args = self.args
        _t0 = time.time()
        logger.info_rank0(
            f"Loading model weights (mode={args.train.load_weights_mode}, init_device={args.train.init_device})..."
        )
        build_result = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            enable_compile=args.train.enable_compile,
            compile_dynamic_shapes=args.train.compile_dynamic_shapes,
            basic_modules=self.model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            gradient_checkpointing_method=args.train.gradient_checkpointing_method,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            load_weights_mode=args.train.load_weights_mode,
            pp_schedule=args.train.pipeline_parallel_schedule if args.train.pipeline_parallel_size > 1 else None,
            reshard_after_forward=args.train.reshard_after_forward,
            fsdp_sharded_lm_head_loss=args.train.fsdp_sharded_lm_head_loss,
            moe_grad_reduce_mode=args.train.moe_grad_reduce_mode,
            fsdp_reduce_dtype=args.train.fsdp_reduce_dtype,
            skip_param_upcast=should_skip_generic_param_upcast(
                enable_lora=args.lora.enable_lora,
                enable_qlora=args.lora.enable_qlora,
                skip_param_upcast=args.train.skip_param_upcast,
            ),
        )

        logger.info_rank0(f"Model weights loaded in {time.time() - _t0:.1f}s")
        helper.print_device_mem_info("VRAM after loading weights")

        # PP returns dict with stages + model_parts; otherwise returns model directly
        self.pp_enabled = isinstance(build_result, dict)
        self.pp_stages = None
        self.model_parts = None
        self.has_first_stage = False
        self.has_last_stage = False
        if self.pp_enabled:
            self.pp_stages = build_result["stages"]
            self.model_parts = build_result["model_parts"]
            self.has_first_stage = build_result["has_first_stage"]
            self.has_last_stage = build_result["has_last_stage"]
            self.model = self.model_parts[0]  # primary model for optimizer etc.
        else:
            self.model = build_result

        # Deferred QLoRA quantization
        if args.lora.enable_qlora:
            self._deferred_qlora_quantize()

        # Freeze non-LoRA params for plain LoRA (QLoRA does this in _deferred_qlora_quantize)
        if args.lora.enable_lora and not args.lora.enable_qlora:
            for name, param in self.model.named_parameters():
                if "lora_A" not in name and "lora_B" not in name:
                    param.requires_grad = False

        if args.model.freeze_router:
            frozen = freeze_deepseek_v3_router_parameters(self.model)
            if frozen == 0:
                for name, param in self.model.named_parameters():
                    if ".gate.weight" in name:
                        param.requires_grad = False
                        frozen += 1
            if frozen > 0:
                logger.info_rank0(f"Froze {frozen} MoE router (gate) parameters")

    def _deferred_qlora_quantize(self) -> None:
        """After FSDP loads weights, quantize them into uint8 buffers."""
        args = self.args
        if self.is_prequantized:
            logger.info("Starting pre-quantized weight loading...")
            helper.print_device_mem_info("VRAM before pre-quantized loading")
            maybe_load_prequantized_qlora(self.model, args.model.model_path)
            logger.info("Done pre-quantized weight loading, freezing non-LoRA params...")
        else:
            logger.info("Starting maybe_quantize_qlora...")
            helper.print_device_mem_info("VRAM before QLoRA quantization")
            maybe_quantize_qlora(self.model)
            logger.info("Done maybe_quantize_qlora, starting MoE weight loading...")
            helper.print_device_mem_info("VRAM after QLoRA linear quantization")
            maybe_load_and_quantize_moe_qlora(self.model, args.model.model_path)
            logger.info("Done MoE weight loading, freezing non-LoRA params...")
            # Deregister packed_weight_f32 from FSDP2 (prevent mixed-precision corruption)
            removed = _deregister_qlora_weights_from_fsdp(
                self.model,
                param_names=("packed_weight_f32",),
            )
            torch.cuda.empty_cache()
            if removed > 0:
                logger.info(f"Deregistered {removed} packed_weight_f32 params from FSDP2")

        # Freeze all non-LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad = False
        helper.print_device_mem_info("VRAM usage after QLoRA quantization")

    def _build_optimizer(self) -> None:
        """Build optimizer and LR scheduler."""
        args = self.args
        self._use_distsignsgd = args.train.optimizer == "distsignsgd"
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            optimizer_dtype=args.train.optimizer_dtype,
            optimizer_kwargs=args.train.optimizer_kwargs,
            cautious_weight_decay=args.train.cautious_weight_decay,
        )
        if self._optimizer_pre_hook_fn is not None:
            hook = self._optimizer_pre_hook_fn(self.model, self.model_config, args.train.data_parallel_mode)
            self.optimizer.register_step_pre_hook(hook)

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=self.total_train_steps,
            lr=args.train.lr,
            lr_min=args.train.lr_min,
            lr_decay_style=args.train.lr_decay_style,
            lr_decay_ratio=args.train.lr_decay_ratio,
            lr_warmup_ratio=args.train.lr_warmup_ratio,
            lr_start=args.train.lr_start,
        )

    def _setup_observability(self) -> None:
        """Initialize wandb, profiler, environ_meter, and save model assets."""
        args = self.args
        self.model_assets = [self.model_config, self.tokenizer]

        if args.train.global_rank == 0:
            save_model_assets(args.train.model_assets_dir, self.model_assets)

        self.profiler = None
        if args.train.profile_this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

        self.environ_meter = helper.EnvironMeter(
            config=self.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            gradient_checkpointing_enabled=args.train.enable_gradient_checkpointing,
            gradient_checkpointing_method=args.train.gradient_checkpointing_method,
            cp_size=args.train.ulysses_parallel_size * args.train.ringattn_parallel_size,
        )
        self._maybe_log_startup_metrics(
            {
                "startup/train_steps_per_epoch": self.train_steps_per_epoch,
                "startup/total_train_steps": self.total_train_steps,
            },
            commit=False,
        )

    def _resume_checkpoint(self) -> None:
        """Load DCP checkpoint if configured."""
        args = self.args
        if not args.train.load_checkpoint_path:
            return

        # When load_weights_mode=skip, parameters are meta-device DTensors after FSDP wrapping.
        # Use to_empty() to materialize them to real CUDA tensors while preserving the DTensor
        # wrapper (unlike manual setattr which would break FSDP2's internal state).
        if args.train.load_weights_mode == "skip":
            logger.info_rank0("Materializing meta parameters to CUDA via to_empty()...")
            self.model.to_empty(device=f"cuda:{args.train.local_rank}")
            logger.info_rank0("Meta parameters materialized.")

        state = {"model": self.model, "extra_state": {}}
        # Only include optimizer if the checkpoint has optimizer state (i.e., resuming training).
        # Model-only DCP checkpoints (from convert_checkpoint.py) won't have optimizer state.
        # load_optimizer=False forces a weights-only resume (optimizer re-initialized fresh / zero
        # momentum) — needed to migrate OLD pre-fix checkpoints whose plain-local Muon momentum
        # buffers are not loadable by the current DTensor-momentum engine.
        ckpt_has_optimizer = os.path.exists(os.path.join(args.train.load_checkpoint_path, ".metadata"))
        if not args.train.load_optimizer:
            logger.info_rank0(
                "load_optimizer=False: resuming weights only, optimizer state re-initialized (zero momentum)."
            )
        elif ckpt_has_optimizer:
            try:
                metadata = dcp_meta.FileSystemReader(args.train.load_checkpoint_path).read_metadata()
                if any(k.startswith("optimizer") for k in metadata.state_dict_metadata.keys()):
                    state["optimizer"] = self.optimizer
            except Exception:
                pass
        self.Checkpointer.load(args.train.load_checkpoint_path, state)

        extra = state.get("extra_state", {})
        self.state.global_step = extra.get("global_step", 0)
        self.state.epoch = self.state.global_step // self.train_steps_per_epoch
        self.state.start_step = self.state.global_step % self.train_steps_per_epoch
        if "lr_scheduler" in extra:
            self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        if "train_dataloader" in extra:
            self.train_dataloader.load_state_dict(extra["train_dataloader"])
        if "environ_meter" in extra:
            self.environ_meter.load_state_dict(extra["environ_meter"])
        if "torch_rng_state" in extra:
            torch.set_rng_state(extra["torch_rng_state"])

        if self.state.start_step == 0:
            iter(self.train_dataloader)  # clear resume state and prefetch

        distributed_barrier()
        logger.info_rank0(f"Loaded checkpoint from {args.train.load_checkpoint_path}")

    def _init_pp_schedule_cache(self) -> None:
        """Initialize PP schedule cache (schedules are built lazily by seq_len)."""
        self._pp_schedule_cache: Dict[int, Any] = {}

    def _build_pp_stage_io(self, example_input_ids: "Optional[torch.Tensor]"):
        """Build (input_args, output_args) meta tensors so PipelineStage skips its
        shape-inference forward (which deadlocks under Ulysses CP). Returns
        (None, None) when no example is available (runtime inference fallback)."""
        if example_input_ids is None:
            return None, None
        mbs = example_input_ids.shape[0]
        s = example_input_ids.shape[-1]
        cfg = self.model.config
        h = cfg.hidden_size
        v = cfg.vocab_size
        dt = torch.bfloat16
        # quack_linear PP loss consumes HIDDEN (lm_head fused into the loss fn),
        # so the last stage outputs hidden [mbs,s,h] instead of logits [mbs,s,v]
        # — this is what avoids the 8GB+ last-stage logits OOM at 248k vocab.
        lm_head_in_loss = self.args.train.ce_mode == "quack_linear"
        if self.ps.is_first_pp_stage:
            input_args = (torch.empty(mbs, s, dtype=example_input_ids.dtype, device="meta"),)
        else:
            input_args = (torch.empty(mbs, s, h, dtype=dt, device="meta"),)
        if self.ps.is_last_pp_stage and not lm_head_in_loss:
            output_args = (torch.empty(mbs, s, v, dtype=dt, device="meta"),)
        else:
            output_args = (torch.empty(mbs, s, h, dtype=dt, device="meta"),)
        return input_args, output_args

    def _get_pp_schedule(self, seq_len: int, example_input_ids: "Optional[torch.Tensor]" = None):
        """Return a cached PP schedule for the given seq_len, building if needed.

        With pp_variable_seq_lengths=True, a new PipelineStage (cheap, no deepcopy)
        is created for each unique seq_len so P2P buffers match the actual shape.
        With static padding, seq_len is always the same so only one entry is cached.

        When ``example_input_ids`` is provided we pass explicit meta input/output
        args to the stage so PipelineStage SKIPS its init-time shape-inference
        forward. This is mandatory under Ulysses CP: the shape-inference dummy
        forward would run the intra-stage CP collectives, which deadlock with the
        cross-stage shape-exchange P2P (observed as a mesh_pp + mesh_ulysses NCCL
        watchdog hang at the first schedule step).
        """
        if seq_len not in self._pp_schedule_cache:
            # quack_linear: last stage returns hidden; the loss fn applies lm_head.
            ce_mode = self.args.train.ce_mode
            self.model_parts[0]._pp_lm_head_in_loss = ce_mode == "quack_linear"
            # Only the last stage computes the loss; pass its lm_head to the loss fn.
            pp_lm_head = getattr(self.model, "lm_head", None) if self.ps.is_last_pp_stage else None
            input_args, output_args = self._build_pp_stage_io(example_input_ids)
            stage = build_pp_stage(
                self.model_parts[0],
                pp_rank=self.ps.pp_rank,
                num_stages=self.ps.pp_size,
                device=get_device_type(),
                pp_group=self.ps.pp_group,
                input_args=input_args,
                output_args=output_args,
            )
            self._pp_schedule_cache[seq_len] = build_pipeline_schedule(
                stages=[stage],
                n_microbatches=self.args.train.gradient_accumulation_steps,
                loss_fn=make_pp_loss_fn(ce_mode, lm_head=pp_lm_head),
                schedule_name=self.args.train.pipeline_parallel_schedule,
            )
        return self._pp_schedule_cache[seq_len]

    # ===================================================================
    # Training loop
    # ===================================================================

    def train(self) -> None:
        """Run the full training loop across all epochs."""
        args = self.args
        ps = self.ps
        state = self.state

        helper.empty_cache()
        model_fwd_context, model_bwd_context = build_activation_offloading_context(
            args.train.enable_activation_offload,
            args.train.enable_gradient_checkpointing,
            args.train.activation_gpu_limit,
            args.train.activation_offload_prefetch_count,
        )
        self._model_fwd_context = model_fwd_context
        self._model_bwd_context = model_bwd_context

        self.model.train()
        logger.info(
            f"rank{args.train.local_rank} Start training, "
            f"train_steps_per_epoch: {self.train_steps_per_epoch}, "
            f"total_train_steps: {self.total_train_steps}, "
            f"epochs: {args.train.num_train_epochs}"
        )
        # Keep all ranks aligned before the first FSDP2 forward. Large
        # full-weight runs can finish setup at different times across nodes,
        # and the first FSDP communicator is initialized lazily in forward.
        distributed_barrier()

        # Per-step ephemeral state for logging in _finalize
        total_loss = 0.0
        grad_norm = 0.0
        lr = args.train.lr

        for epoch in range(state.epoch, args.train.num_train_epochs):
            state.epoch = epoch
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)

            steps_this_epoch = self.train_steps_per_epoch - state.start_step
            if args.train.max_steps is not None:
                steps_this_epoch = min(steps_this_epoch, args.train.max_steps - state.global_step)
            if steps_this_epoch <= 0:
                break

            # Progress bar
            use_tqdm = args.train.log_format == "progress_bar"
            if use_tqdm:
                data_loader_tqdm = trange(
                    steps_this_epoch,
                    desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
                    total=state.start_step + steps_this_epoch,
                    initial=state.start_step,
                    disable=args.train.local_rank != 0,
                )

            data_iterator = iter(self.train_dataloader)
            for _ in range(state.start_step, self.train_steps_per_epoch):
                if args.train.max_steps is not None and state.global_step >= args.train.max_steps:
                    logger.info_rank0(f"Reached max_steps={args.train.max_steps}, stopping training.")
                    break
                state.global_step += 1

                try:
                    micro_batches: List[Dict[str, Any]] = next(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                    break

                # Synchronize padding across DP ranks
                sync_group = ps.fsdp_group if self.pp_enabled else None
                synchronize_micro_batch_padding(micro_batches, group=sync_group)

                # Static PP padding: pad to sample_packing_sequence_len upfront.
                # With pp_variable_seq_lengths, padding is deferred to _forward_backward_pp.
                if self.pp_enabled and not self.args.train.pp_variable_seq_lengths:
                    self._pad_micro_batches_for_pp(micro_batches)

                if state.global_step == 1:
                    helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

                # --- One optimizer step ---
                synchronize()
                start_time = time.time()

                total_loss, grad_norm = self.train_step(micro_batches)
                self._log_step_diagnostics_once()

                synchronize()
                delta_time = time.time() - start_time
                lr = max(self.lr_scheduler.get_last_lr())
                train_metrics = self.environ_meter.step(delta_time, global_step=state.global_step)
                state.loss_history.append(total_loss)

                # Logging
                self._maybe_log(
                    total_loss,
                    grad_norm,
                    lr,
                    train_metrics,
                    delta_time,
                    use_tqdm,
                    data_loader_tqdm if use_tqdm else None,
                )
                self._maybe_profile()
                self._maybe_save_checkpoint()

            if use_tqdm:
                data_loader_tqdm.close()
            state.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self._finalize(total_loss, grad_norm, lr)

    # ===================================================================
    # train_step — one gradient accumulation step
    # ===================================================================

    def _step_phase_timing_active(self) -> bool:
        return self.args.train.enable_step_phase_timing and self._current_step_phase_times is not None

    def _sync_step_phase_timing(self) -> None:
        if self.args.train.step_phase_timing_sync_cuda and get_device_type() != "cpu":
            synchronize()

    def _record_step_phase_time(self, phase_name: str, elapsed: float) -> None:
        if self._current_step_phase_times is None:
            return
        self._current_step_phase_times[phase_name] = self._current_step_phase_times.get(phase_name, 0.0) + elapsed

    def _step_memory_profiling_active(self) -> bool:
        return (
            self.args.train.enable_step_memory_profiling
            and self._current_step_memory_stats is not None
            and get_device_type() == "cuda"
        )

    def _capture_local_memory_stats(self) -> Dict[str, float]:
        if get_device_type() != "cuda":
            return {}

        device = get_torch_device()
        allocated_gb = device.memory_allocated() / (1024**3)
        reserved_gb = device.memory_reserved() / (1024**3)
        max_allocated_gb = device.max_memory_allocated() / (1024**3)
        max_reserved_gb = device.max_memory_reserved() / (1024**3)
        free_bytes, total_bytes = device.mem_get_info()
        device_free_gb = free_bytes / (1024**3)
        device_total_gb = total_bytes / (1024**3)
        device_used_gb = device_total_gb - device_free_gb
        return {
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "max_allocated_gb": max_allocated_gb,
            "max_reserved_gb": max_reserved_gb,
            "device_used_gb": device_used_gb,
            "device_free_gb": device_free_gb,
            "non_pytorch_used_gb": max(device_used_gb - allocated_gb, 0.0),
        }

    def _reset_local_peak_memory_stats(self) -> None:
        if get_device_type() == "cuda":
            get_torch_device().reset_peak_memory_stats()

    def _record_step_phase_memory(
        self,
        phase_name: str,
        before: Optional[Dict[str, float]],
        after: Optional[Dict[str, float]],
    ) -> None:
        if self._current_step_memory_stats is None or not before or not after:
            return

        current = self._current_step_memory_stats.setdefault(phase_name, {})
        current["before_allocated_gb"] = before["allocated_gb"]
        current["after_allocated_gb"] = after["allocated_gb"]
        current["delta_allocated_gb"] = after["allocated_gb"] - before["allocated_gb"]
        current["phase_peak_allocated_gb"] = after["max_allocated_gb"]
        current["before_reserved_gb"] = before["reserved_gb"]
        current["after_reserved_gb"] = after["reserved_gb"]
        current["delta_reserved_gb"] = after["reserved_gb"] - before["reserved_gb"]
        current["phase_peak_reserved_gb"] = after["max_reserved_gb"]
        current["after_device_used_gb"] = after["device_used_gb"]
        current["after_device_free_gb"] = after["device_free_gb"]
        current["after_non_pytorch_used_gb"] = after["non_pytorch_used_gb"]

    def _time_step_phase(self, phase_name: str, fn):
        if not self._step_phase_timing_active():
            return fn()

        self._sync_step_phase_timing()
        memory_before = self._capture_local_memory_stats() if self._step_memory_profiling_active() else None
        if memory_before is not None:
            self._reset_local_peak_memory_stats()
        start = time.perf_counter()
        try:
            return fn()
        finally:
            original_exc_type = sys.exc_info()[0]
            sync_exc: Optional[Exception] = None
            try:
                self._sync_step_phase_timing()
            except Exception as exc:  # noqa: BLE001
                sync_exc = exc
                logger.warning(f"Failed to synchronize after step phase {phase_name!r}: {type(exc).__name__}: {exc}")
            self._record_step_phase_time(phase_name, time.perf_counter() - start)
            memory_after = self._capture_local_memory_stats() if memory_before is not None else None
            self._record_step_phase_memory(phase_name, memory_before, memory_after)
            if sync_exc is not None and original_exc_type is None:
                raise sync_exc

    def _finalize_step_phase_times(self, phase_times: Dict[str, float]) -> None:
        if not phase_times:
            self._last_step_phase_times = {}
            return

        phases = [name for name in _STEP_PHASE_TIMING_ORDER if name in phase_times]
        phases.extend(name for name in sorted(phase_times) if name not in _STEP_PHASE_TIMING_ORDER)
        local_values = [float(phase_times[name]) for name in phases]

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            mean_values = all_reduce(local_values, op="mean")
            max_values = all_reduce(local_values, op="max")
            min_values = all_reduce(local_values, op="min")
        else:
            mean_values = local_values
            max_values = local_values
            min_values = local_values

        if not isinstance(mean_values, list):
            mean_values = [mean_values]
        if not isinstance(max_values, list):
            max_values = [max_values]
        if not isinstance(min_values, list):
            min_values = [min_values]

        self._last_step_phase_times = {
            phase: {
                "local": float(local),
                "mean": float(mean),
                "max": float(maximum),
                "min": float(minimum),
            }
            for phase, local, mean, maximum, minimum in zip(
                phases,
                local_values,
                mean_values,
                max_values,
                min_values,
                strict=False,
            )
        }

    def _finalize_local_step_phase_times(self, phase_times: Dict[str, float]) -> None:
        """Finalize phase timings without distributed collectives.

        This is used only on exception paths. If a step is failing from CUDA OOM
        or NCCL trouble, another collective can hide the useful diagnostic.
        """
        self._last_step_phase_times = _summarize_phase_times_local(phase_times)

    def _finalize_step_memory_stats(self, memory_stats: Dict[str, Dict[str, float]]) -> None:
        if not memory_stats:
            self._last_step_memory_stats = {}
            return

        phases = [name for name in _STEP_PHASE_TIMING_ORDER if name in memory_stats]
        phases.extend(name for name in sorted(memory_stats) if name not in _STEP_PHASE_TIMING_ORDER)
        metric_names = sorted({metric for phase in phases for metric in memory_stats[phase]})
        keys = [(phase, metric) for phase in phases for metric in metric_names if metric in memory_stats[phase]]
        local_values = [float(memory_stats[phase][metric]) for phase, metric in keys]

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            mean_values = all_reduce(local_values, op="mean")
            max_values = all_reduce(local_values, op="max")
            min_values = all_reduce(local_values, op="min")
        else:
            mean_values = local_values
            max_values = local_values
            min_values = local_values

        if not isinstance(mean_values, list):
            mean_values = [mean_values]
        if not isinstance(max_values, list):
            max_values = [max_values]
        if not isinstance(min_values, list):
            min_values = [min_values]

        self._last_step_memory_stats = {}
        for (phase, metric), local, mean, maximum, minimum in zip(
            keys,
            local_values,
            mean_values,
            max_values,
            min_values,
            strict=False,
        ):
            self._last_step_memory_stats.setdefault(phase, {})[metric] = {
                "local": float(local),
                "mean": float(mean),
                "max": float(maximum),
                "min": float(minimum),
            }

    def _finalize_local_step_memory_stats(self, memory_stats: Dict[str, Dict[str, float]]) -> None:
        self._last_step_memory_stats = _summarize_memory_stats_local(memory_stats)

    def _format_local_memory_snapshot(self) -> str:
        if get_device_type() != "cuda":
            return "device=cpu"

        stats = self._capture_local_memory_stats()
        return (
            f"allocated_gb={stats['allocated_gb']:.3f} reserved_gb={stats['reserved_gb']:.3f} "
            f"max_allocated_gb={stats['max_allocated_gb']:.3f} "
            f"max_reserved_gb={stats['max_reserved_gb']:.3f} "
            f"device_used_gb={stats['device_used_gb']:.3f} "
            f"non_pytorch_used_gb={stats['non_pytorch_used_gb']:.3f}"
        )

    def _log_step_exception_diagnostics(
        self,
        phase_times: Optional[Dict[str, float]],
        step_start: float,
        exc: BaseException,
    ) -> None:
        if phase_times is None:
            return

        self._record_step_phase_time("train_step_total", time.perf_counter() - step_start)
        if self._per_component_timer.enabled:
            try:
                for phase_name, elapsed in self._per_component_timer.end_step().items():
                    self._record_step_phase_time(phase_name, elapsed)
            except Exception as timer_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to finalize per-component timing on train_step exception: "
                    f"{type(timer_exc).__name__}: {timer_exc}"
                )

        self._finalize_local_step_phase_times(phase_times)
        timing_log = self._format_step_phase_timing_log(partial=True)
        if timing_log:
            logger.info(timing_log)
        if self._current_step_memory_stats is not None:
            self._finalize_local_step_memory_stats(self._current_step_memory_stats)
            memory_log = self._format_step_memory_profile_log(partial=True)
            if memory_log:
                logger.info(memory_log)
        logger.info(
            "[STEP_EXCEPTION_MEMORY] "
            f"rank={self.args.train.global_rank} local_rank={self.args.train.local_rank} "
            f"step={self.state.global_step} exception={type(exc).__name__}: {exc} "
            f"{self._format_local_memory_snapshot()}"
        )

    def train_step(self, micro_batches: List[Dict[str, Any]]) -> Tuple[float, float]:
        """One complete gradient-accumulation step.

        Returns:
            (total_loss, grad_norm) — all-reduced across DP for logging.
        """
        phase_times: Optional[Dict[str, float]] = {} if self.args.train.enable_step_phase_timing else None
        memory_stats: Optional[Dict[str, Dict[str, float]]] = (
            {} if self.args.train.enable_step_memory_profiling else None
        )
        self._current_step_phase_times = phase_times
        self._last_step_phase_times = {}
        self._current_step_memory_stats = memory_stats
        self._last_step_memory_stats = {}
        self._step_diagnostics_logged = False
        if phase_times is not None:
            self._sync_step_phase_timing()
            step_start = time.perf_counter()
            if memory_stats is not None:
                self._record_step_phase_memory(
                    "step_start",
                    self._capture_local_memory_stats(),
                    self._capture_local_memory_stats(),
                )
            if self._per_component_timer.enabled:
                if not self._per_component_timer._attached:
                    n_layers = self._per_component_timer.attach(self.model)
                    logger.info_rank0(f"Per-component timer attached to {n_layers} decoder layers")
                self._per_component_timer.start_step()
        else:
            step_start = 0.0

        try:
            global_valid_tokens = self._time_step_phase(
                "count_valid_tokens",
                lambda: self._count_valid_tokens(micro_batches),
            )
            if self._use_distsignsgd:
                active_microbatches, active_voter_total = self._time_step_phase(
                    "count_active_microbatches",
                    lambda: self._count_active_microbatches(micro_batches),
                )
            else:
                active_microbatches, active_voter_total = 0, 0
            self._time_step_phase("optimizer_zero_grad", self.optimizer.zero_grad)

            self._forward_peak_bytes = 0
            self._backward_peak_bytes = 0
            self._optim_peak_bytes = 0
            self._fwdbwd_peak_bytes = 0

            def _add_environ_meter_batches() -> None:
                for mb in micro_batches:
                    self.environ_meter.add(mb)

            self._time_step_phase("environ_meter_add", _add_environ_meter_batches)

            # Routing replay: outer lifecycle
            if self._use_routing_replay:
                set_replay_stage("replay_backward")

            if self.pp_enabled:

                def _do_pp_forward_backward() -> float:
                    # The PP schedule interleaves micro-batch forwards and backwards
                    # internally, so fwd/bwd peaks cannot be attributed separately.
                    # Record a single combined fwd+bwd peak instead of falsely
                    # reporting fwd=0.
                    _reset_cuda_peak_memory_stats()
                    total = self._forward_backward_pp(micro_batches, global_valid_tokens)
                    self._fwdbwd_peak_bytes = _cuda_max_memory_allocated()
                    return total

                total_loss = self._time_step_phase(
                    "forward_backward_total",
                    _do_pp_forward_backward,
                )
            else:
                total_loss = self._time_step_phase(
                    "forward_backward_total",
                    lambda: self._forward_backward(micro_batches, global_valid_tokens),
                )

            if self._use_routing_replay:
                self._time_step_phase(
                    "routing_replay_clear",
                    lambda: (set_replay_stage(None), RoutingReplay.clear_all()),
                )

            if os.environ.get("XORL_DEBUG_NONFINITE_GRAD_SCAN", "0") == "1":
                self._scan_nonfinite_grads()

            self._time_step_phase("sync_sp_gradients", self._sync_sp_gradients)
            if getattr(self.ps, "lm_head_tp_size", 1) > 1:
                self._time_step_phase(
                    "sync_lm_head_tp_gradient",
                    lambda: sync_lm_head_tp_gradient(self.model, self.ps.lm_head_tp_replica_group),
                )
            if self._use_distsignsgd and active_microbatches > 0:
                self._time_step_phase(
                    "distsign_grad_scale",
                    lambda: scale_model_gradients(
                        self.model,
                        get_distsign_grad_scale_factor(active_voter_total),
                    ),
                )

            def _clip_and_step_with_memory() -> float:
                _reset_cuda_peak_memory_stats()
                grad_norm = self._clip_and_step()
                self._optim_peak_bytes = _cuda_max_memory_allocated()
                return grad_norm

            grad_norm = self._time_step_phase("clip_and_step_total", _clip_and_step_with_memory)
            self._time_step_phase("maybe_merge_lora", self._maybe_merge_lora)
            total_loss, grad_norm = self._time_step_phase(
                "reduce_metrics",
                lambda: self._reduce_metrics(total_loss, grad_norm),
            )

            if phase_times is not None:
                self._sync_step_phase_timing()
                self._record_step_phase_time("train_step_total", time.perf_counter() - step_start)
                if self._per_component_timer.enabled:
                    for phase_name, elapsed in self._per_component_timer.end_step().items():
                        self._record_step_phase_time(phase_name, elapsed)
                self._finalize_step_phase_times(phase_times)
                if memory_stats is not None:
                    self._finalize_step_memory_stats(memory_stats)

            return total_loss, grad_norm
        except Exception as exc:
            self._log_step_exception_diagnostics(phase_times, step_start, exc)
            raise
        finally:
            self._current_step_phase_times = None
            self._current_step_memory_stats = None

    # ===================================================================
    # train_step helpers
    # ===================================================================

    def _scan_nonfinite_grads(self) -> None:
        """Debug helper (XORL_DEBUG_NONFINITE_GRAD_SCAN=1): after backward, log
        which parameter grads contain nan/inf, aggregated per layer. Backward
        runs last layer -> first, so the HIGHEST layer with bad grads is the
        corruption source; everything below it inherits the bad activation grad."""
        import re  # noqa: PLC0415

        rank = dist.get_rank() if dist.is_initialized() else 0
        bad_names: list[str] = []
        per_layer_bad: dict[str, int] = {}
        per_layer_total: dict[str, int] = {}
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            local = grad._local_tensor if hasattr(grad, "_local_tensor") else grad
            if local.numel() == 0:
                continue
            m = re.search(r"layers\.(\d+)\.", name)
            layer = m.group(1) if m else "non-layer"
            per_layer_total[layer] = per_layer_total.get(layer, 0) + 1
            if bool(torch.isfinite(local.float()).all()):
                continue
            bad_names.append(name)
            per_layer_bad[layer] = per_layer_bad.get(layer, 0) + 1
        if not bad_names:
            logger.info(f"[nonfinite-grad-scan][rank{rank}] all parameter grads finite")
            return
        layer_summary = " ".join(
            f"L{layer}:{per_layer_bad[layer]}/{per_layer_total.get(layer, 0)}"
            for layer in sorted(per_layer_bad, key=lambda x: (x == "non-layer", int(x) if x.isdigit() else 0))
        )
        numeric_layers = [int(x) for x in per_layer_bad if x.isdigit()]
        boundary = max(numeric_layers) if numeric_layers else None
        boundary_params = [n for n in bad_names if boundary is not None and f"layers.{boundary}." in n]
        logger.warning(
            f"[nonfinite-grad-scan][rank{rank}] {len(bad_names)} bad param grads | per-layer: {layer_summary} | "
            f"boundary layer {boundary} bad params: {boundary_params}"
        )

    def _count_valid_tokens(self, micro_batches: List[Dict[str, Any]]) -> torch.Tensor:
        """Count valid (non-IGNORE_INDEX) tokens and all-reduce across DP."""
        return count_valid_tokens(
            micro_batches,
            group=self.ps.loss_group if self.ps.loss_parallel_enabled else None,
        )

    def _count_active_microbatches(self, micro_batches: List[Dict[str, Any]]) -> tuple[int, int]:
        """Return ``(active_microbatches, active_voter_total)`` over the DP group."""
        return count_active_microbatches(
            micro_batches,
            group=self.ps.loss_group if self.ps.loss_parallel_enabled else None,
        )

    def _pad_micro_batches_for_pp(self, micro_batches: List[Dict[str, Any]]) -> None:
        pad_micro_batches_for_pp(
            micro_batches,
            sample_packing_sequence_len=self.args.data.sample_packing_sequence_len or 0,
            sp_size=self.ps.cp_size,
            pad_to_multiple_of=self.args.data.pad_to_multiple_of or 1,
        )

    def _sync_sp_gradients(self) -> None:
        """All-reduce gradients for CP/Ulysses dims not folded into FSDP."""
        sync_sp_gradients(
            self.model,
            self.ps.sp_grad_sync_group,
            skip_dtensor_grads=self._use_distsignsgd,
        )

    def _reduce_metrics(self, total_loss: float, grad_norm: float) -> Tuple[float, float]:
        """All-reduce loss and grad_norm across DP for logging."""
        if self.pp_enabled:
            # PP: the MAX all-reduce in forward_backward_pp syncs loss only across
            # pp_group (stages), not across fsdp_group (DP replicas). Different DP
            # replicas hold different CE_sums, so SUM over fsdp_group then divide
            # gives CE_sum_total / gvt_total — matching the non-PP path.
            total_loss = all_reduce(total_loss, op="sum", group=self.ps.fsdp_group)
            grad_norm = all_reduce(grad_norm, op="mean", group=self.ps.fsdp_group)
        else:
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=self.ps.fsdp_group)
        return total_loss, grad_norm

    # ===================================================================
    # Forward-backward: non-PP and PP paths
    # ===================================================================

    def _forward_backward(
        self,
        micro_batches: List[Dict[str, Any]],
        global_valid_tokens: torch.Tensor,
    ) -> float:
        """Standard gradient accumulation loop (non-PP)."""
        total_loss = 0.0

        # One buffer per optimizer step: accumulates global-batch expert-selection
        # frequencies across this gradient-accumulation window, then is discarded.
        lb_buffer = LoadBalancingBuffer() if self._moe_global_load_balance else None

        # Gradient-accumulation sync deferral (HSDP only): defer ONLY the cross-node replicate-dim
        # all-reduce to the last microbatch. We deliberately do NOT defer the reduce-scatter
        # (set_requires_gradient_sync(False) would keep gradients UNSHARDED across the accumulation
        # window — full param size — and OOM at 35B scale). Keeping reduce-scatter on every microbatch
        # holds grads sharded (cheap intra-node), while set_requires_all_reduce(False) batches the
        # exposed cross-node reduce → ~gradient_accumulation_steps× fewer cross-node syncs. The reduce-
        # scatter sum across microbatches followed by one all-reduce is mathematically equivalent.
        # No-op unless HSDP (data_parallel_replicate_size > 1); experts use a 1-D ep_fsdp mesh with no
        # replicate dim, so their reduce-scatter is unaffected (and is intra-node when ep is intranode).
        from torch.distributed._composable.fsdp.fully_shard import FSDPModule  # noqa: PLC0415

        _n_mb = len(micro_batches)
        _defer_all_reduce = (
            self.args.train.defer_grad_sync_in_accumulation
            and _n_mb > 1
            and self.args.train.data_parallel_replicate_size > 1
            and isinstance(self.model, FSDPModule)
            and hasattr(self.model, "set_requires_all_reduce")
        )

        for _mb_idx, micro_batch in enumerate(micro_batches):
            if _defer_all_reduce:
                self.model.set_requires_all_reduce(_mb_idx == _n_mb - 1)
            micro_batch = self._time_step_phase(
                "microbatch_to_device",
                lambda micro_batch=micro_batch: {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                },
            )

            # Pop labels before forward (model only outputs last_hidden_state)
            labels = micro_batch.pop("labels")

            if self._use_routing_replay:
                set_replay_stage("record")
            _reset_cuda_peak_memory_stats()
            with self._model_fwd_context:

                def _do_model_forward(micro_batch=micro_batch):
                    self._per_component_timer.set_mode("fwd")
                    try:
                        return self.model(**micro_batch, use_cache=False, output_hidden_states=False)
                    finally:
                        self._per_component_timer.set_mode("idle")

                outputs = self._time_step_phase("model_forward", _do_model_forward)

                def _compute_ga_loss(outputs=outputs, labels=labels) -> torch.Tensor:
                    loss_fn_params = self._causallm_loss_params
                    if self.args.train.fsdp_sharded_lm_head_loss:
                        loss_fn_params = dict(loss_fn_params)
                        loss_fn_params["fsdp_sharded_lm_head_loss_global_valid_tokens"] = global_valid_tokens
                    result = compute_loss(
                        self.model.lm_head,
                        outputs.last_hidden_state,
                        loss_fn_name=None,
                        loss_fn_inputs={"labels": labels},
                        loss_fn_params=loss_fn_params,
                        logits_to_keep=0,
                    )
                    loss = result.loss

                    aux_loss = None
                    if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                        raw_aux_loss = global_load_balancing_loss_func(
                            outputs.router_logits,
                            self.model.num_experts,
                            self.model.num_experts_per_tok,
                            dp_group=self.ps.dp_group if self.ps.dp_enabled else None,
                            buffer=lb_buffer,
                        )
                        if raw_aux_loss != 0:
                            aux_loss = self.model.router_aux_loss_coef * raw_aux_loss.to(loss.device)

                    local_valid_tokens = (labels != IGNORE_INDEX).sum()
                    if self.args.train.fsdp_sharded_lm_head_loss:
                        # The sharded-lm-head loss already divides this microbatch's
                        # CE sum by the whole accumulation step token count.
                        ga_loss = loss
                        if aux_loss is not None:
                            aux_ga_loss, _ = gradient_accumulate_loss(
                                aux_loss,
                                local_valid_tokens,
                                global_valid_tokens,
                                group=self.ps.loss_group if self.ps.loss_parallel_enabled else None,
                            )
                            ga_loss = ga_loss + aux_ga_loss
                    else:
                        if aux_loss is not None:
                            loss = loss + aux_loss
                        ga_loss, _ = gradient_accumulate_loss(
                            loss,
                            local_valid_tokens,
                            global_valid_tokens,
                            group=self.ps.loss_group if self.ps.loss_parallel_enabled else None,
                        )
                    return ga_loss

                ga_loss = self._time_step_phase("loss_compute", _compute_ga_loss)
                self._forward_peak_bytes = max(self._forward_peak_bytes, _cuda_max_memory_allocated())
            if self._use_routing_replay:
                set_replay_stage("replay_backward")

            _reset_cuda_peak_memory_stats()
            with self._model_bwd_context:

                def _do_backward(ga_loss=ga_loss) -> None:
                    self._per_component_timer.set_mode("bwd")
                    try:
                        if _env_flag("XORL_TRAINER_MEMORY_TRACE") and torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            _trainer_memory_trace("before ga_loss.backward")
                        try:
                            ga_loss.backward()
                        except torch.OutOfMemoryError:
                            _trainer_memory_trace("OOM during ga_loss.backward", force=True)
                            raise
                        if _env_flag("XORL_TRAINER_MEMORY_TRACE"):
                            _trainer_memory_trace("after ga_loss.backward")
                    finally:
                        self._per_component_timer.set_mode("idle")

                self._time_step_phase("backward", _do_backward)
            self._backward_peak_bytes = max(self._backward_peak_bytes, _cuda_max_memory_allocated())

            total_loss += self._time_step_phase("loss_item", ga_loss.item)
            del micro_batch, labels, outputs, ga_loss

        return total_loss

    def _forward_backward_pp(
        self,
        micro_batches: List[Dict[str, Any]],
        global_valid_tokens: torch.Tensor,
    ) -> float:
        """Pipeline parallel forward-backward step.

        With pp_variable_seq_lengths: negotiates the per-step max seq_len across
        PP ranks, pads micro-batches to that length, and uses a seq_len-keyed
        schedule cache so P2P buffers always match the actual tensor shape.

        Runs the PP schedule (which calls loss_fn.backward() internally), then
        normalizes gradients by global_valid_tokens in-place.  Returns the
        normalized loss for logging.
        """
        if self.args.train.pp_variable_seq_lengths:
            seq_len = self._time_step_phase(
                "pp_negotiate_seq_len",
                lambda: negotiate_pp_seq_len(micro_batches, self.ps.pp_group),
            )
            self._time_step_phase(
                "pp_pad_microbatches",
                lambda: pad_micro_batches_for_pp(
                    micro_batches,
                    sample_packing_sequence_len=seq_len * self.ps.cp_size,
                    sp_size=self.ps.cp_size,
                    pad_to_multiple_of=self.args.data.pad_to_multiple_of or 1,
                ),
            )
        else:
            seq_len = micro_batches[0]["input_ids"].shape[-1]

        raw_loss = self._time_step_phase(
            "pp_forward_backward",
            lambda: forward_backward_pp(
                model_parts=self.model_parts,
                pp_schedule=self._get_pp_schedule(seq_len, example_input_ids=micro_batches[0]["input_ids"]),
                micro_batches=micro_batches,
                has_first_stage=self.has_first_stage,
                has_last_stage=self.has_last_stage,
                pp_group=self.ps.pp_group,
            ),
        )
        gvt = global_valid_tokens.item()
        if gvt > 0 and not self._use_distsignsgd:
            self._time_step_phase(
                "pp_grad_scale",
                lambda: scale_model_gradients(self.model_parts, 1.0 / float(gvt)),
            )
        return raw_loss / gvt if gvt > 0 else 0.0

    def _clip_and_step(self) -> float:
        """Clip gradients, optimizer.step(), lr_scheduler.step().

        Returns grad_norm (scalar).
        """
        clip_value = get_effective_grad_clip_value(
            self.args.train.max_grad_norm,
            use_distsignsgd=self._use_distsignsgd,
        )
        grad_norm = self._time_step_phase(
            "clip_gradients",
            lambda: clip_gradients(
                self.model,
                clip_value,
                pp_enabled=self.pp_enabled,
                pp_group=self.ps.pp_group if self.pp_enabled else None,
            ),
        )

        self._time_step_phase("optimizer_step", self.optimizer.step)
        self._time_step_phase("lr_scheduler_step", self.lr_scheduler.step)

        return grad_norm

    # ===================================================================
    # Periodic actions
    # ===================================================================

    def _maybe_merge_lora(self) -> None:
        """Periodic LoRA merge at merge_lora_interval."""
        maybe_merge_lora(
            self.model,
            enable_lora=self.args.lora.enable_lora,
            enable_qlora=self.args.lora.enable_qlora,
            merge_interval=self.args.lora.merge_lora_interval,
            global_step=self.state.global_step,
            optimizer=self.optimizer,
            reset_optimizer=self.args.lora.reset_optimizer_on_merge,
        )

    def _format_step_phase_timing_log(self, *, partial: bool = False) -> Optional[str]:
        if not self.args.train.enable_step_phase_timing or not self._last_step_phase_times:
            return None

        max_steps_str = self.args.train.max_steps or "?"
        prefix = "STEP_PHASES_PARTIAL" if partial else "STEP_PHASES"
        parts = []
        for phase in _STEP_PHASE_TIMING_ORDER:
            if phase not in self._last_step_phase_times:
                continue
            values = self._last_step_phase_times[phase]
            parts.append(f"{phase}_max_s={values['max']:.6f}")
            parts.append(f"{phase}_mean_s={values['mean']:.6f}")

        for phase in sorted(set(self._last_step_phase_times) - set(_STEP_PHASE_TIMING_ORDER)):
            values = self._last_step_phase_times[phase]
            parts.append(f"{phase}_max_s={values['max']:.6f}")
            parts.append(f"{phase}_mean_s={values['mean']:.6f}")

        return f"[{prefix} {self.state.global_step}/{max_steps_str}] " + " ".join(parts)

    def _format_step_memory_profile_log(self, *, partial: bool = False) -> Optional[str]:
        if not self.args.train.enable_step_memory_profiling or not self._last_step_memory_stats:
            return None

        max_steps_str = self.args.train.max_steps or "?"
        prefix = "STEP_MEMORY_PARTIAL" if partial else "STEP_MEMORY"
        metric_names = (
            "after_allocated_gb",
            "phase_peak_allocated_gb",
            "delta_allocated_gb",
            "after_reserved_gb",
            "phase_peak_reserved_gb",
            "delta_reserved_gb",
            "after_device_used_gb",
            "after_non_pytorch_used_gb",
        )
        phase_order = [name for name in _STEP_PHASE_TIMING_ORDER if name in self._last_step_memory_stats]
        phase_order.extend(
            name for name in sorted(self._last_step_memory_stats) if name not in _STEP_PHASE_TIMING_ORDER
        )
        parts = []
        for phase in phase_order:
            values_by_metric = self._last_step_memory_stats[phase]
            for metric in metric_names:
                if metric not in values_by_metric:
                    continue
                values = values_by_metric[metric]
                safe_phase = phase.replace("/", "_")
                safe_metric = metric.replace("_gb", "")
                parts.append(f"{safe_phase}_{safe_metric}_max_gb={values['max']:.3f}")
                parts.append(f"{safe_phase}_{safe_metric}_mean_gb={values['mean']:.3f}")

        return f"[{prefix} {self.state.global_step}/{max_steps_str}] " + " ".join(parts)

    def _log_step_diagnostics_once(self) -> None:
        if getattr(self, "_step_diagnostics_logged", False):
            return
        phase_log = self._format_step_phase_timing_log()
        if phase_log is not None:
            logger.info_rank0(phase_log)
        memory_log = self._format_step_memory_profile_log()
        if memory_log is not None:
            logger.info_rank0(memory_log)
        self._step_diagnostics_logged = True

    def _consume_activation_offload_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metric_contexts = (
            ("fwd", getattr(self, "_model_fwd_context", None)),
            ("bwd", getattr(self, "_model_bwd_context", None)),
        )
        for prefix, ctx in metric_contexts:
            consume_stats = getattr(ctx, "consume_stats", None)
            if not callable(consume_stats):
                continue
            stats = consume_stats() or {}
            offloaded_gb = float(stats.get("bytes_offloaded", 0)) / (1024**3)
            kept_on_gpu_gb = float(stats.get("bytes_kept_on_gpu", 0)) / (1024**3)
            if offloaded_gb > 0:
                metrics[f"activation_offload/{prefix}_offloaded_max(GB)"] = offloaded_gb
            if kept_on_gpu_gb > 0:
                metrics[f"activation_offload/{prefix}_kept_on_gpu_max(GB)"] = kept_on_gpu_gb

        if metrics and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            keys = list(metrics)
            values = all_reduce([metrics[key] for key in keys], op="max")
            if not isinstance(values, list):
                values = [values]
            metrics.update(dict(zip(keys, values)))
        return metrics

    def _maybe_log(
        self,
        total_loss: float,
        grad_norm: float,
        lr: float,
        train_metrics: Dict[str, Any],
        delta_time: float,
        use_tqdm: bool,
        tqdm_bar: Optional[Any],
    ) -> None:
        """Log metrics to stdout / tqdm / wandb / structured JSON."""
        args = self.args
        tflops_per_gpu = train_metrics.get("efficiency/flops_achieved(T)", 0) / args.train.world_size
        mfu = train_metrics.get("efficiency/mfu", 0)
        tokens_per_sec = train_metrics.get("efficiency/tokens_per_second(K)", 0) * 1e3
        gpu_mem_gb = train_metrics.get("memory/gpu_allocated(GB)", 0)

        offload_metrics = self._consume_activation_offload_metrics()
        offload_gb = offload_metrics.get("activation_offload/fwd_offloaded_max(GB)", 0.0) + offload_metrics.get(
            "activation_offload/bwd_offloaded_max(GB)", 0.0
        )

        fwd_peak_bytes = getattr(self, "_forward_peak_bytes", 0)
        bwd_peak_bytes = getattr(self, "_backward_peak_bytes", 0)
        optim_peak_bytes = getattr(self, "_optim_peak_bytes", 0)
        fwdbwd_peak_bytes = getattr(self, "_fwdbwd_peak_bytes", 0)
        fwd_peak_bytes, bwd_peak_bytes, optim_peak_bytes, fwdbwd_peak_bytes = all_reduce(
            [fwd_peak_bytes, bwd_peak_bytes, optim_peak_bytes, fwdbwd_peak_bytes],
            op="max",
        )
        fwd_peak_gb = fwd_peak_bytes / (1024**3)
        bwd_peak_gb = bwd_peak_bytes / (1024**3)
        optim_peak_gb = optim_peak_bytes / (1024**3)
        fwdbwd_peak_gb = fwdbwd_peak_bytes / (1024**3)
        # Under PP, fwd/bwd are interleaved and reported as a single combined peak.
        pp_combined = self.pp_enabled
        step_peak_gb = max(gpu_mem_gb, fwd_peak_gb, bwd_peak_gb, optim_peak_gb, fwdbwd_peak_gb)
        train_metrics["memory/gpu_allocated(GB)"] = step_peak_gb
        train_metrics["memory/optim_peak(GB)"] = optim_peak_gb
        if pp_combined:
            train_metrics["memory/forward_backward_peak(GB)"] = fwdbwd_peak_gb
        else:
            train_metrics["memory/forward_peak(GB)"] = fwd_peak_gb
            train_metrics["memory/backward_peak(GB)"] = bwd_peak_gb
        if offload_gb > 0:
            train_metrics["memory/activation_offloaded(GB)"] = offload_gb
        train_metrics.update(offload_metrics)

        if use_tqdm and tqdm_bar is not None:
            tqdm_bar.set_postfix_str(f"loss={total_loss:.2f} gn={grad_norm:.2f} lr={lr:.1e} tok/s={tokens_per_sec:.0f}")
            tqdm_bar.update()
        else:
            max_steps_str = args.train.max_steps or "?"
            phase_str = ""
            if pp_combined:
                if fwdbwd_peak_gb > 0 or optim_peak_gb > 0:
                    phase_str = f" fwd+bwd={fwdbwd_peak_gb:.1f}GB optim={optim_peak_gb:.1f}GB"
            elif fwd_peak_gb > 0 or bwd_peak_gb > 0 or optim_peak_gb > 0:
                phase_str = f" fwd={fwd_peak_gb:.1f}GB bwd={bwd_peak_gb:.1f}GB optim={optim_peak_gb:.1f}GB"
            offload_str = f" offload={offload_gb:.2f}GB" if offload_gb > 0 else ""
            logger.info_rank0(
                f"[STEP {self.state.global_step}/{max_steps_str}] "
                f"loss={total_loss:.4f} grad_norm={grad_norm:.4f} lr={lr:.6e} "
                f"tflops={tflops_per_gpu:.1f} mfu={mfu:.4f} "
                f"tokens_per_sec={tokens_per_sec:.0f} time={delta_time:.3f}s "
                f"peak_mem={step_peak_gb:.1f}GB{phase_str}{offload_str}"
            )

        self._log_step_diagnostics_once()

        if (
            args.train.global_rank == 0
            and args.train.use_wandb
            and self.state.global_step % args.train.wandb_log_interval == 0
        ):
            import wandb  # noqa: PLC0415

            epoch_fraction = self.state.epoch + (
                (self.state.global_step - self.state.epoch * self.train_steps_per_epoch)
                / max(self.train_steps_per_epoch, 1)
            )
            train_metrics.update(
                {
                    "training/loss": total_loss,
                    "training/grad_norm": grad_norm,
                    "training/lr": lr,
                    "training/epoch": epoch_fraction,
                    "training/step_time": delta_time,
                    "training/samples_seen": self.state.global_step * args.train.global_batch_size,
                }
            )
            if self._last_step_phase_times:
                for phase, values in self._last_step_phase_times.items():
                    train_metrics[f"step_phase/{phase}_local_s"] = values["local"]
                    train_metrics[f"step_phase/{phase}_mean_s"] = values["mean"]
                    train_metrics[f"step_phase/{phase}_max_s"] = values["max"]
                    train_metrics[f"step_phase/{phase}_min_s"] = values["min"]
            if self._last_step_memory_stats:
                for phase, values_by_metric in self._last_step_memory_stats.items():
                    for metric, values in values_by_metric.items():
                        train_metrics[f"step_memory/{phase}/{metric}_local"] = values["local"]
                        train_metrics[f"step_memory/{phase}/{metric}_mean"] = values["mean"]
                        train_metrics[f"step_memory/{phase}/{metric}_max"] = values["max"]
                        train_metrics[f"step_memory/{phase}/{metric}_min"] = values["min"]
            # Log per-group LRs (e.g., separate Muon vs AdamW LRs)
            for group in self.optimizer.param_groups:
                if group.get("use_muon", False):
                    train_metrics.setdefault("training/lr_muon", group["lr"])
                elif "use_muon" in group:
                    train_metrics.setdefault("training/lr_adamw", group["lr"])
            wandb.log(train_metrics, step=self.state.global_step)

    def _maybe_profile(self) -> None:
        """Advance profiler step."""
        args = self.args
        step = self.state.global_step
        if self.profiler is not None and step <= args.train.profile_end_step:
            self.profiler.step()
            if step == args.train.profile_end_step:
                self.profiler.stop()

    def _maybe_save_checkpoint(self) -> None:
        """Save DCP checkpoint if save_steps or save_epoch_steps triggers."""
        args = self.args
        step = self.state.global_step

        should_save = (args.train.save_steps and step % args.train.save_steps == 0) or (
            self.save_epoch_steps and step % self.save_epoch_steps == 0
        )
        if not should_save:
            return

        helper.empty_cache()
        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{step}")
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {
                "global_step": step,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "train_dataloader": self.train_dataloader.state_dict(),
                "environ_meter": self.environ_meter.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        is_lora_training = args.lora.enable_lora or args.lora.enable_qlora
        _save_lora_only = is_lora_training and args.lora.merge_lora_interval == 0
        self.Checkpointer.save(
            args.train.save_checkpoint_path,
            state,
            global_steps=step,
            save_lora_only=_save_lora_only,
        )
        distributed_barrier()
        logger.info_rank0(f"Checkpoint saved at {save_checkpoint_path}")

    # ===================================================================
    # Finalize
    # ===================================================================

    def _collect_fp8_training_metrics(self) -> Dict[str, Any]:
        """Collect rank-local and world-summed FP8 training runtime counters."""

        from xorl.fp8_training import summarize_fp8_training_model  # noqa: PLC0415

        local_summary = summarize_fp8_training_model(self.model)
        counter_keys = (
            "linear_modules",
            "linear_modules_used_fp8",
            "linear_modules_allow_bf16_fallback",
            "linear_modules_backward_fp8",
            "linear_modules_backward_bf16",
            "moe_modules",
            "moe_fp8_enabled_modules",
            "moe_modules_used_fp8",
            "moe_quack_modules",
        )

        global_counts = {key: int(local_summary[key]) for key in counter_keys}
        if dist.is_initialized() and dist.get_world_size() > 1:
            if get_device_type() == "cuda":
                device = torch.device("cuda", self.args.train.local_rank)
            else:
                device = torch.device("cpu")
            counts = torch.tensor([global_counts[key] for key in counter_keys], dtype=torch.long, device=device)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            global_counts = {key: int(counts[idx].item()) for idx, key in enumerate(counter_keys)}

        return {
            "rank0_linear_modules": int(local_summary["linear_modules"]),
            "rank0_linear_modules_used_fp8": int(local_summary["linear_modules_used_fp8"]),
            "rank0_linear_modules_allow_bf16_fallback": int(local_summary["linear_modules_allow_bf16_fallback"]),
            "rank0_linear_modules_backward_fp8": int(local_summary["linear_modules_backward_fp8"]),
            "rank0_linear_modules_backward_bf16": int(local_summary["linear_modules_backward_bf16"]),
            "rank0_unused_linear_module_names": local_summary["unused_linear_module_names"],
            "rank0_moe_modules": int(local_summary["moe_modules"]),
            "rank0_moe_fp8_enabled_modules": int(local_summary["moe_fp8_enabled_modules"]),
            "rank0_moe_modules_used_fp8": int(local_summary["moe_modules_used_fp8"]),
            "rank0_unused_moe_module_names": local_summary["unused_moe_module_names"],
            "rank0_moe_quack_modules": int(local_summary["moe_quack_modules"]),
            "global_linear_modules": global_counts["linear_modules"],
            "global_linear_modules_used_fp8": global_counts["linear_modules_used_fp8"],
            "global_linear_modules_allow_bf16_fallback": global_counts["linear_modules_allow_bf16_fallback"],
            "global_linear_modules_backward_fp8": global_counts["linear_modules_backward_fp8"],
            "global_linear_modules_backward_bf16": global_counts["linear_modules_backward_bf16"],
            "global_linear_modules_not_used_fp8": (
                global_counts["linear_modules"] - global_counts["linear_modules_used_fp8"]
            ),
            "global_moe_modules": global_counts["moe_modules"],
            "global_moe_fp8_enabled_modules": global_counts["moe_fp8_enabled_modules"],
            "global_moe_modules_used_fp8": global_counts["moe_modules_used_fp8"],
            "global_moe_modules_not_used_fp8": (
                global_counts["moe_fp8_enabled_modules"] - global_counts["moe_modules_used_fp8"]
            ),
            "global_moe_quack_modules": global_counts["moe_quack_modules"],
        }

    def _finalize(self, total_loss: float, grad_norm: float, lr: float) -> None:
        """Post-training: HF save, metrics JSON, barrier, destroy."""
        args = self.args
        state = self.state

        synchronize()

        # Report peak GPU memory on rank 0 — parseable by benchmark scripts.
        if logger.isEnabledFor(10):  # DEBUG
            device = get_torch_device()
            peak_alloc_gb = device.max_memory_allocated() / (1024**3)
            peak_reserved_gb = device.max_memory_reserved() / (1024**3)
            logger.debug_rank0(
                f"[PEAK_MEMORY] peak_alloc_gb={peak_alloc_gb:.3f} peak_reserved_gb={peak_reserved_gb:.3f}"
            )

        # Gather full model state for HF save
        is_lora_training = args.lora.enable_lora or args.lora.enable_qlora
        save_peft_adapter = is_lora_training and args.lora.merge_lora_interval == 0

        hf_model_state_dict = None
        hf_lora_state_dict = None
        if args.train.save_hf_weights and not save_peft_adapter:
            logger.info_rank0("Gathering full model state dict for HF checkpoint via NCCL with CPU offload...")
            hf_model_state_dict = get_model_state_dict(
                self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
        elif args.train.save_hf_weights and save_peft_adapter:
            # Collective: every rank must participate in the DTensor gather inside
            # get_lora_state_dict. Doing this on rank 0 only (inside the save
            # block below) deadlocks because the other ranks have already moved
            # past to dist.barrier().
            logger.info_rank0("Gathering LoRA state dict for HF adapter export via NCCL...")
            hf_lora_state_dict = get_lora_state_dict(self.model)

        del self.optimizer, self.lr_scheduler
        helper.empty_cache()

        # Save HF weights (rank 0)
        if args.train.global_rank == 0 and args.train.save_hf_weights:
            hf_weights_path = os.path.join(args.train.output_dir, f"global_step_{state.global_step}", "hf_ckpt")
            if save_peft_adapter:
                save_lora_checkpoint(
                    self.model,
                    hf_weights_path,
                    base_model_name=args.model.model_path,
                    target_modules=args.lora.lora_target_modules,
                    r=args.lora.lora_rank,
                    lora_alpha=args.lora.lora_alpha,
                    moe_hybrid_shared_lora=args.lora.moe_hybrid_shared_lora,
                    lora_state_dict=hf_lora_state_dict,
                )
                logger.info_rank0(f"PEFT adapter saved at {hf_weights_path}")
            elif hf_model_state_dict is not None:
                checkpoint_handler = (
                    self.model.get_checkpoint_handler() if hasattr(self.model, "get_checkpoint_handler") else None
                )
                save_model_weights(
                    hf_weights_path,
                    hf_model_state_dict,
                    model_assets=self.model_assets,
                    checkpoint_handler=checkpoint_handler,
                )
                del hf_model_state_dict
                logger.info_rank0(f"HF checkpoint saved at {hf_weights_path}")

        # Write training metrics (rank 0)
        training_metrics = {
            "final_loss": total_loss,
            "final_grad_norm": grad_norm,
            "final_lr": lr,
            "global_step": state.global_step,
            "total_train_steps": self.total_train_steps,
            "loss_history": state.loss_history,
        }
        if args.train.enable_fp8_training:
            training_metrics["fp8_training"] = self._collect_fp8_training_metrics()
        if args.train.enable_qarl:
            from xorl.qarl import summarize_qarl_model  # noqa: PLC0415

            training_metrics["qarl"] = summarize_qarl_model(self.model)

        if args.train.global_rank == 0:
            metrics_path = os.path.join(args.train.output_dir, "training_metrics.json")
            os.makedirs(args.train.output_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(training_metrics, f, indent=2)

        distributed_barrier()
        dist.destroy_process_group()
