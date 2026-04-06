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
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import trange

from xorl.arguments import Arguments
from xorl.checkpoint import build_checkpointer
from xorl.data.constants import IGNORE_INDEX
from xorl.data.data_loader import DataLoaderBuilder
from xorl.data.prepare.prepare_datasets import prepare_datasets
from xorl.distributed.gradient_accumulate_loss import gradient_accumulate_loss
from xorl.distributed.offloading import build_activation_offloading_context
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.sync_padding import synchronize_micro_batch_padding
from xorl.distributed.torch_parallelize import build_parallelize_model
from xorl.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from xorl.models.layers.moe.aux_loss import global_load_balancing_loss_func
from xorl.models.layers.moe.routing_replay import RoutingReplay, set_replay_stage
from xorl.models.module_utils import compute_loss
from xorl.optim import build_lr_scheduler, build_optimizer
from xorl.trainers.training_utils import (
    clip_gradients,
    count_valid_tokens,
    forward_backward_pp,
    maybe_merge_lora,
    negotiate_pp_seq_len,
    pad_micro_batches_for_pp,
    pp_loss_fn,
    sync_sp_gradients,
)
from xorl.utils import helper
from xorl.utils.device import get_device_type, get_nccl_backend, get_torch_device, synchronize
from xorl.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


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
        dist.init_process_group(backend=get_nccl_backend())
        logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
        logger.info_rank0(json.dumps(asdict(args), indent=2))

        get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
        helper.set_seed(args.train.seed, args.train.enable_full_determinism)

        if args.train.local_rank == 0:
            helper.enable_third_party_logging()

        if args.train.global_rank == 0:
            from xorl.arguments import save_args

            save_args(args, args.train.output_dir)
            if args.train.use_wandb:
                import wandb

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
            dp_mode=args.train.data_parallel_mode,
            cp_fsdp_mode=args.train.cp_fsdp_mode,
        )

        # DTensor RNG tracker (run_state_sync=False to avoid PP deadlock)
        self.ps = get_parallel_state()
        if self.ps.device_mesh is not None:
            import torch.distributed.tensor._random

            torch.distributed.tensor._random.manual_seed(args.train.seed, self.ps.device_mesh)

        # Routing replay is only needed with EP when MoE forward is recomputed
        self._use_routing_replay = self.ps.ep_size > 1 and args.train.moe_recomputed

    def _maybe_log_startup_metrics(self, metrics: Dict[str, Any], commit: bool = False) -> None:
        """Log startup metrics to wandb once rank 0 has initialized it."""
        if not metrics:
            return
        if self.args.train.global_rank == 0:
            self._startup_metrics.update(metrics)
        if self.args.train.global_rank != 0 or not self.args.train.use_wandb or not self._wandb_initialized:
            return
        import wandb

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
        gathered: List[Optional[Dict[str, Any]]] = [None] * self.args.train.world_size
        dist.all_gather_object(gathered, payload)
        return [item for item in gathered if item is not None]

    def _log_host_inventory(self) -> None:
        """Emit host inventory to stdout and wandb config on rank 0."""
        inventory = self._collect_host_inventory()
        if self.args.train.global_rank != 0:
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
            import wandb

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
        self.model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            ep_dispatch=args.model.ep_dispatch,
            train_router=args.model.train_router,
            deepep_buffer_size_gb=args.model.deepep_buffer_size_gb,
            deepep_num_sms=args.model.deepep_num_sms,
            deepep_async_combine=args.model.deepep_async_combine,
            rmsnorm_mode=args.model.rmsnorm_mode,
            init_device=args.train.init_device,
        )
        self.model_config = self.model.config
        helper.print_device_mem_info("VRAM usage after building model")

        # Unfuse QKV for tensor parallelism
        if not args.model.merge_qkv:
            for layer in self.model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "unfuse_for_tp"):
                    layer.self_attn.unfuse_for_tp()
            logger.info_rank0("Unfused QKV projections (merge_qkv=False)")

        # QLoRA / LoRA injection
        self.is_prequantized = False
        self.checkpoint_quant_format = None
        self.exclude_modules = set()
        if args.lora.enable_qlora:
            self._inject_qlora()
        elif args.lora.enable_lora:
            self._inject_lora()

        # Save pre-hook before parallelization (some models register optimizer hooks)
        self._optimizer_pre_hook_fn = getattr(self.model, "get_optimizer_pre_hook", None)

    def _inject_qlora(self) -> None:
        """QLoRA injection with pre-quantized checkpoint detection."""
        args = self.args
        from xorl.qlora import detect_prequantized_block_fp8, detect_prequantized_nvfp4, inject_qlora_into_model

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
            from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules

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
        """Plain LoRA injection."""
        args = self.args
        from xorl.lora.utils import inject_lora_into_model

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
            basic_modules=self.model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            recompute_modules=args.train.recompute_modules,
            moe_checkpoint_method=args.train.moe_checkpoint_method,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            load_weights_mode=args.train.load_weights_mode,
            pp_schedule=args.train.pipeline_parallel_schedule if args.train.pipeline_parallel_size > 1 else None,
            reshard_after_forward=args.train.reshard_after_forward,
            skip_param_upcast=args.lora.enable_qlora,
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

    def _deferred_qlora_quantize(self) -> None:
        """After FSDP loads weights, quantize them into uint8 buffers."""
        args = self.args
        if self.is_prequantized:
            from xorl.qlora import maybe_load_prequantized_qlora

            logger.info("Starting pre-quantized weight loading...")
            helper.print_device_mem_info("VRAM before pre-quantized loading")
            maybe_load_prequantized_qlora(self.model, args.model.model_path)
            logger.info("Done pre-quantized weight loading, freezing non-LoRA params...")
        else:
            from xorl.qlora import maybe_load_and_quantize_moe_qlora, maybe_quantize_qlora
            from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp

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
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer,
            optimizer_dtype=args.train.optimizer_dtype,
            optimizer_kwargs=args.train.optimizer_kwargs,
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
            gc_enabled=args.train.enable_gradient_checkpointing,
            recompute_modules=args.train.recompute_modules,
            moe_checkpoint_method=args.train.moe_checkpoint_method,
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

        state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}
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

        dist.barrier()
        logger.info_rank0(f"Loaded checkpoint from {args.train.load_checkpoint_path}")

    def _init_pp_schedule_cache(self) -> None:
        """Initialize PP schedule cache (schedules are built lazily by seq_len)."""
        self._pp_schedule_cache: Dict[int, Any] = {}

    def _get_pp_schedule(self, seq_len: int):
        """Return a cached PP schedule for the given seq_len, building if needed.

        With pp_variable_seq_lengths=True, a new PipelineStage (cheap, no deepcopy)
        is created for each unique seq_len so P2P buffers match the actual shape.
        With static padding, seq_len is always the same so only one entry is cached.
        """
        if seq_len not in self._pp_schedule_cache:
            from xorl.distributed.pipeline_parallel import build_pipeline_schedule, build_pp_stage

            stage = build_pp_stage(
                self.model_parts[0],
                pp_rank=self.ps.pp_rank,
                num_stages=self.ps.pp_size,
                device=get_device_type(),
                pp_group=self.ps.pp_group,
            )
            self._pp_schedule_cache[seq_len] = build_pipeline_schedule(
                stages=[stage],
                n_microbatches=self.args.train.gradient_accumulation_steps,
                loss_fn=pp_loss_fn,
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

    def train_step(self, micro_batches: List[Dict[str, Any]]) -> Tuple[float, float]:
        """One complete gradient-accumulation step.

        Returns:
            (total_loss, grad_norm) — all-reduced across DP for logging.
        """
        global_valid_tokens = self._count_valid_tokens(micro_batches)
        self.optimizer.zero_grad()

        for mb in micro_batches:
            self.environ_meter.add(mb)

        # Routing replay: outer lifecycle
        if self._use_routing_replay:
            set_replay_stage("replay_backward")

        if self.pp_enabled:
            total_loss = self._forward_backward_pp(micro_batches, global_valid_tokens)
        else:
            total_loss = self._forward_backward(micro_batches, global_valid_tokens)

        if self._use_routing_replay:
            set_replay_stage(None)
            RoutingReplay.clear_all()

        self._sync_sp_gradients()
        grad_norm = self._clip_and_step()
        self._maybe_merge_lora()
        total_loss, grad_norm = self._reduce_metrics(total_loss, grad_norm)

        return total_loss, grad_norm

    # ===================================================================
    # train_step helpers
    # ===================================================================

    def _count_valid_tokens(self, micro_batches: List[Dict[str, Any]]) -> torch.Tensor:
        """Count valid (non-IGNORE_INDEX) tokens and all-reduce across DP."""
        return count_valid_tokens(
            micro_batches,
            group=self.ps.fsdp_group if self.pp_enabled else None,
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
        sync_sp_gradients(self.model, self.ps.sp_grad_sync_group)

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

        for micro_batch in micro_batches:
            micro_batch = {
                k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in micro_batch.items()
            }

            # Pop labels before forward (model only outputs last_hidden_state)
            labels = micro_batch.pop("labels")

            if self._use_routing_replay:
                set_replay_stage("record")
            with self._model_fwd_context:
                outputs = self.model(**micro_batch, use_cache=False, output_hidden_states=False)
                result = compute_loss(
                    self.model.lm_head,
                    outputs.last_hidden_state,
                    loss_fn_name=None,
                    loss_fn_inputs={"labels": labels},
                    loss_fn_params=None,
                    logits_to_keep=0,
                )
                loss = result.loss

                if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                    aux_loss = global_load_balancing_loss_func(
                        outputs.router_logits,
                        self.model.num_experts,
                        self.model.num_experts_per_tok,
                        dp_group=self.ps.dp_group if self.ps.dp_enabled else None,
                    )
                    if aux_loss != 0:
                        loss = loss + self.model.router_aux_loss_coef * aux_loss.to(loss.device)

                local_valid_tokens = (labels != IGNORE_INDEX).sum()
                ga_loss, _ = gradient_accumulate_loss(loss, local_valid_tokens, global_valid_tokens)
            if self._use_routing_replay:
                set_replay_stage("replay_backward")

            with self._model_bwd_context:
                ga_loss.backward()

            total_loss += ga_loss.item()
            del micro_batch, loss, outputs, ga_loss

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
            seq_len = negotiate_pp_seq_len(micro_batches, self.ps.pp_group)
            pad_micro_batches_for_pp(
                micro_batches,
                sample_packing_sequence_len=seq_len * self.ps.cp_size,
                sp_size=self.ps.cp_size,
                pad_to_multiple_of=self.args.data.pad_to_multiple_of or 1,
            )
        else:
            seq_len = micro_batches[0]["input_ids"].shape[-1]

        raw_loss = forward_backward_pp(
            model_parts=self.model_parts,
            pp_schedule=self._get_pp_schedule(seq_len),
            micro_batches=micro_batches,
            has_first_stage=self.has_first_stage,
            has_last_stage=self.has_last_stage,
            pp_group=self.ps.pp_group,
        )
        gvt = global_valid_tokens.item()
        if gvt > 0:
            scale = 1.0 / gvt
            for model_part in self.model_parts:
                for p in model_part.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
        return raw_loss / gvt if gvt > 0 else 0.0

    def _clip_and_step(self) -> float:
        """Clip gradients, optimizer.step(), lr_scheduler.step().

        Returns grad_norm (scalar).
        """
        grad_norm = clip_gradients(
            self.model,
            self.args.train.max_grad_norm,
            pp_enabled=self.pp_enabled,
            pp_group=self.ps.pp_group if self.pp_enabled else None,
        )

        self.optimizer.step()
        self.lr_scheduler.step()

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
        """Log metrics to stdout / tqdm / wandb."""
        args = self.args
        tflops_per_gpu = train_metrics.get("efficiency/flops_achieved(T)", 0) / args.train.world_size
        mfu = train_metrics.get("efficiency/mfu", 0)
        tokens_per_sec = train_metrics.get("efficiency/tokens_per_second(K)", 0) * 1e3

        if use_tqdm and tqdm_bar is not None:
            tqdm_bar.set_postfix_str(f"loss={total_loss:.2f} gn={grad_norm:.2f} lr={lr:.1e} tok/s={tokens_per_sec:.0f}")
            tqdm_bar.update()
        else:
            max_steps_str = args.train.max_steps or "?"
            logger.info_rank0(
                f"[STEP {self.state.global_step}/{max_steps_str}] "
                f"loss={total_loss:.4f} grad_norm={grad_norm:.4f} lr={lr:.6e} "
                f"tflops={tflops_per_gpu:.1f} mfu={mfu:.4f} "
                f"tokens_per_sec={tokens_per_sec:.0f} time={delta_time:.3f}s"
            )

        if (
            args.train.global_rank == 0
            and args.train.use_wandb
            and self.state.global_step % args.train.wandb_log_interval == 0
        ):
            import wandb

            train_metrics.update(
                {
                    "training/loss": total_loss,
                    "training/grad_norm": grad_norm,
                    "training/lr": lr,
                    "training/epoch": self.state.epoch,
                    "training/step_time": delta_time,
                    "training/samples_seen": self.state.global_step * args.train.global_batch_size,
                }
            )
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
        dist.barrier()
        logger.info_rank0(f"Checkpoint saved at {save_checkpoint_path}")

    # ===================================================================
    # Finalize
    # ===================================================================

    def _finalize(self, total_loss: float, grad_norm: float, lr: float) -> None:
        """Post-training: HF save, metrics JSON, barrier, destroy."""
        args = self.args
        state = self.state

        synchronize()

        # Gather full model state for HF save
        is_lora_training = args.lora.enable_lora or args.lora.enable_qlora
        save_peft_adapter = is_lora_training and args.lora.merge_lora_interval == 0

        hf_model_state_dict = None
        if args.train.save_hf_weights and not save_peft_adapter:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            logger.info_rank0("Gathering full model state dict for HF checkpoint via NCCL with CPU offload...")
            hf_model_state_dict = get_model_state_dict(
                self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )

        del self.optimizer, self.lr_scheduler
        helper.empty_cache()

        # Save HF weights (rank 0)
        if args.train.global_rank == 0 and args.train.save_hf_weights:
            hf_weights_path = os.path.join(args.train.output_dir, f"global_step_{state.global_step}", "hf_ckpt")
            if save_peft_adapter:
                from xorl.lora.utils import save_lora_checkpoint

                save_lora_checkpoint(
                    self.model,
                    hf_weights_path,
                    base_model_name=args.model.model_path,
                    target_modules=args.lora.lora_target_modules,
                    r=args.lora.lora_rank,
                    lora_alpha=args.lora.lora_alpha,
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
        if args.train.global_rank == 0:
            metrics_path = os.path.join(args.train.output_dir, "training_metrics.json")
            os.makedirs(args.train.output_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "final_loss": total_loss,
                        "final_grad_norm": grad_norm,
                        "final_lr": lr,
                        "global_step": state.global_step,
                        "total_train_steps": self.total_train_steps,
                        "loss_history": state.loss_history,
                    },
                    f,
                    indent=2,
                )

        dist.barrier()
        dist.destroy_process_group()
