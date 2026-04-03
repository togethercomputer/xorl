"""
ModelRunner - Handles Model Operations for Distributed Training

This module handles the core model operations:
- Forward and backward passes with gradient accumulation
- Optimizer steps with gradient clipping
- Checkpoint saving and loading (delegated to CheckpointManager)
- R3 routing replay for MoE (delegated to RoutingReplayHandler)
- Uses xorl's training infrastructure (FSDP, optimizer, checkpointing, etc.)

Usage:
    This class is used by dispatcher.py and should not be run directly.
    See xorl.server.runner.runner_dispatcher for the entry point.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from xorl.checkpoint import build_checkpointer
from xorl.data.constants import IGNORE_INDEX
from xorl.distributed.offloading import build_activation_offloading_context
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.pipeline_parallel import build_pipeline_schedule, build_pp_stage
from xorl.distributed.sequence_parallel.data import gather_outputs
from xorl.lora import LoraLinear
from xorl.models.layers.moe.routing_replay import set_replay_stage
from xorl.ops.loss import (
    causallm_loss_function,
    importance_sampling_loss_function,
    policy_loss_function,
)
from xorl.optim import build_optimizer
from xorl.server.runner.adapters import LoRAAdapterManager
from xorl.server.runner.checkpoint import CheckpointManager
from xorl.server.runner.utils import MoeMetricsTracker, RoutingReplayHandler, run_self_test, validate_token_ids
from xorl.trainers.model_builder import build_training_model
from xorl.trainers.training_utils import (
    clip_gradients,
    count_valid_tokens,
    forward_backward_pp,
    negotiate_pp_seq_len,
    pad_micro_batches_for_pp,
    pp_loss_fn,
    sync_sp_gradients,
)
from xorl.trainers.training_utils import (
    maybe_merge_lora as _maybe_merge_lora_util,
)
from xorl.utils import helper
from xorl.utils.device import get_device_id, get_device_type, get_torch_device, synchronize
from xorl.utils.dist_utils import all_reduce


logger = logging.getLogger(__name__)


class RankFilter(logging.Filter):
    """Filter that only allows logging from rank 0 to reduce duplicate messages."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


def configure_rank0_logging(logger_instance, rank):
    """Configure logger to only log from rank 0."""
    # Add filter to all handlers
    for handler in logger_instance.handlers:
        handler.addFilter(RankFilter(rank))
    # If no handlers, add filter to logger itself
    if not logger_instance.handlers:
        logger_instance.addFilter(RankFilter(rank))


def _sp_allreduce_kl_metrics(metrics: Dict[str, Any], sp_group) -> Dict[str, Any]:
    """
    All-reduce KL/ratio metrics across the sequence-parallel (Ulysses) group.

    With Ulysses SP, each rank only sees a shard of the sequence. Rank 0 often
    has only prompt tokens (all target_tokens=-100), so its KL stats are zeros.
    This function aggregates stats across all SP ranks so every rank (especially
    rank 0 which reports metrics) sees the correct global values.

    Mean-type metrics (kl, entropy, ratio_mean, pg_clipfrac) are converted to
    (value * local_n) sums, all-reduced with SUM, then divided by total_n.
    Min/max metrics use MIN/MAX all-reduce with proper identity elements for
    ranks that have no valid tokens.
    """
    device = torch.device("cuda")
    local_n = metrics.get("_n_valid_kl", 0)

    # --- Mean-type metrics: convert to weighted sums, all-reduce, divide ---
    mean_keys = ["kl_sample_train_k3", "entropy_sample", "ratio_mean", "pg_clipfrac"]
    sum_tensors = {}
    for key in mean_keys:
        if key in metrics:
            val = float(metrics[key]) * local_n
            sum_tensors[key] = torch.tensor(val, dtype=torch.float64, device=device)

    # All-reduce the weighted sums
    for t in sum_tensors.values():
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=sp_group)

    # --- Min/max metrics: use identity elements for empty ranks ---
    ratio_min_val = float(metrics.get("ratio_min", 1.0)) if local_n > 0 else float("inf")
    ratio_max_val = float(metrics.get("ratio_max", 1.0)) if local_n > 0 else float("-inf")
    ratio_min_t = torch.tensor(ratio_min_val, dtype=torch.float64, device=device)
    ratio_max_t = torch.tensor(ratio_max_val, dtype=torch.float64, device=device)
    dist.all_reduce(ratio_min_t, op=dist.ReduceOp.MIN, group=sp_group)
    dist.all_reduce(ratio_max_t, op=dist.ReduceOp.MAX, group=sp_group)

    # --- All-reduce total valid token count ---
    n_tensor = torch.tensor(float(local_n), dtype=torch.float64, device=device)
    dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM, group=sp_group)
    total_n = max(n_tensor.item(), 1.0)

    # --- Update metrics with properly reduced values ---
    for key in mean_keys:
        if key in sum_tensors:
            metrics[key] = sum_tensors[key].item() / total_n

    ratio_min_reduced = ratio_min_t.item()
    ratio_max_reduced = ratio_max_t.item()
    metrics["ratio_min"] = ratio_min_reduced if ratio_min_reduced != float("inf") else 1.0
    metrics["ratio_max"] = ratio_max_reduced if ratio_max_reduced != float("-inf") else 1.0
    metrics["valid_tokens"] = total_n

    # Clean up internal key
    metrics.pop("_n_valid_kl", None)

    return metrics


class ModelRunner:
    """
    ModelRunner handles model operations on distributed GPUs using xorl infrastructure.

    Uses the actual training components from xorl (FSDP, optimizer, checkpointing, etc.)
    """

    # --- Exclude keys per loss function for model_inputs filtering ---
    _LOSS_EXCLUDE_KEYS = {
        "causallm_loss": {"labels", "_original_position_ids", "rollout_logprobs"},
        "cross_entropy": {"labels", "_original_position_ids", "rollout_logprobs"},
        "importance_sampling": {
            "labels",
            "target_tokens",
            "logprobs",
            "advantages",
            "_original_position_ids",
            "rollout_logprobs",
        },
        "policy_loss": {"labels", "target_tokens", "logprobs", "advantages", "rollout_logprobs"},
    }

    def __init__(
        self,
        config: Dict[str, Any],
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        """
        Initialize the ModelRunner.

        Args:
            config: Full configuration dictionary (model, train, data args)
            rank: Global rank in distributed training
            world_size: Total number of processes
            local_rank: Local rank on this node
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Configure logger to only log from rank 0 to avoid duplicate messages
        configure_rank0_logging(logger, rank)

        # Extract config sections
        self.model_config = config.get("model", {})
        self.train_config = config.get("train", {})
        self.lora_config = config.get("lora", {})

        # Cross-entropy mode
        self.ce_mode = self.train_config.get("ce_mode", "eager")

        # LM head fp32 flag for loss functions
        self.lm_head_fp32 = self.model_config.get("lm_head_fp32", True)

        # Training state
        self.global_step = 0
        self.global_forward_backward_step = 0
        self.is_sleeping = False  # Track sleep state

        # Deferred gradient normalization: accumulate raw valid token counts
        # across forward_backward calls, normalize once at optim_step.
        self._accumulated_valid_tokens: Dict[str, int] = {}

        # PP schedule cache: keyed by (n_microbatches, seq_len) to avoid rebuilding on every call.
        self._pp_schedule_cache: Dict[tuple, Any] = {}

        # Multi-adapter support (initialized later if LoRA is enabled)
        self._adapter_manager: Optional[LoRAAdapterManager] = None

        # Single-tenant session tracking (for full-weights training mode)
        # When LoRA is disabled, only one training session is allowed at a time
        self._active_session_id: Optional[str] = None

        # PP state (set by _initialize_model if PP enabled)
        self.pp_enabled = False
        self.pp_stages = None
        self.model_parts = None
        self.has_first_stage = False
        self.has_last_stage = False

        # Device setup
        get_torch_device().set_device(f"{get_device_type()}:{local_rank}")
        helper.set_seed(self.train_config.get("seed", 42), False)

        # Initialize distributed parallel state
        self._init_parallel_state()

        # Initialize model, optimizer, etc.
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_checkpointer()
        self._initialize_contexts()

        # Initialize multi-adapter manager if LoRA is enabled
        if self.lora_config.get("enable_lora", False):
            # Adapter manager now takes device instead of optimizer
            # Each adapter has its own optimizer instance
            device = torch.device(f"{get_device_type()}:{self.local_rank}")
            # Only rank 0 should save on eviction to avoid multi-rank file conflicts
            self._adapter_manager = LoRAAdapterManager(
                self.model, device, auto_save_on_eviction=(self.rank == 0), lora_config=self.lora_config
            )
            # Register the "default" adapter with the initial weights and lr
            self._adapter_manager.register_adapter(
                model_id="default",
                lr=self.train_config.get("lr", 1e-5),
                initialize_fresh=False,  # Use current weights as the default adapter
            )
            self._adapter_manager.current_adapter_id = "default"
            logger.info("Multi-adapter manager initialized with default adapter")

        # Initialize tokenizer for sampling (only on rank 0)
        if self.rank == 0:
            model_path = self.model_config.get("model_path")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Tokenizer loaded for sampling: {model_path}")
        else:
            self.tokenizer = None

        # Initialize extracted modules
        self._routing_handler = RoutingReplayHandler(self.model)
        self._checkpoint_mgr = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            checkpointer=self.Checkpointer,
            lora_config=self.lora_config,
            model_config=self.model_config,
            train_config=self.train_config,
            rank=self.rank,
            local_rank=self.local_rank,
            adapter_manager=self._adapter_manager,
        )
        # Sync initial attributes
        self._checkpoint_mgr.lora_target_modules = getattr(self, "lora_target_modules", None)
        self._checkpoint_mgr.lora_alpha_value = getattr(self, "lora_alpha_value", None)

        # Setup MoE metrics collection if this is a Qwen3 MoE model
        self._moe_tracker = MoeMetricsTracker(self.model_config_obj, self.train_config, self.rank)

        logger.info(
            f"ModelRunner initialized on rank {rank}/{world_size} "
            f"(local_rank: {local_rank}, device: {get_device_type()})"
        )

        if self.train_config.get("enable_self_test", False):
            run_self_test(self)

    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA training mode is enabled."""
        return self.lora_config.get("enable_lora", False)

    @property
    def adapter_manager(self):
        """Public access to the adapter manager for multi-tenancy LoRA."""
        return self._adapter_manager

    def _check_not_sleeping(self, operation: str) -> None:
        """Raise if the model is in sleep mode (CPU-offloaded)."""
        if self.is_sleeping:
            raise RuntimeError(f"Cannot perform {operation}: model is in sleep mode. Call wake_up first.")

    def _validate_single_tenant(self, model_id: str) -> None:
        """
        Enforce single-tenant mode when LoRA is disabled (full-weights training).

        In full-weights mode, only one training session is allowed at a time.
        This ensures exclusive access to the model parameters.

        Args:
            model_id: The model identifier for the current request.

        Raises:
            ValueError: If a different session is already active.
        """
        if self.lora_enabled:
            return  # Multi-tenant allowed for LoRA mode

        if self._active_session_id is None:
            self._active_session_id = model_id
            logger.info(f"Full-weights session started: {model_id}")
        elif self._active_session_id != model_id:
            raise ValueError(
                f"Full-weights mode is single-tenant. Active session: {self._active_session_id}, "
                f"requested: {model_id}. Call /api/v1/kill_session first to start a new session."
            )

    def kill_session(self, model_id: str, save_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Kill the active full-weights training session.

        This allows a new session to be started. For LoRA mode, this is a no-op
        since multi-tenancy is supported.

        Args:
            model_id: The session to kill (must match active session).
            save_checkpoint: Whether to save a checkpoint before killing.

        Returns:
            Dictionary with success status and optional checkpoint path.

        Raises:
            ValueError: If model_id doesn't match the active session.
        """
        if self.lora_enabled:
            return {
                "success": True,
                "message": "LoRA mode supports multi-tenancy, no session to kill.",
                "checkpoint_path": None,
            }

        if self._active_session_id is None:
            return {
                "success": True,
                "message": "No active session to kill.",
                "checkpoint_path": None,
            }

        if self._active_session_id != model_id:
            raise ValueError(
                f"Cannot kill session: requested '{model_id}' but active session is '{self._active_session_id}'"
            )

        checkpoint_path = None
        if save_checkpoint:
            # Save checkpoint before killing session
            output_dir = self.train_config.get("output_dir", "outputs")
            checkpoint_name = f"session_{model_id}_final"
            checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint_name)
            try:
                self.save_state(checkpoint_path, save_optimizer=True)
                logger.info(f"Saved final checkpoint before killing session: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint before killing session: {e}")
                checkpoint_path = None

        old_session = self._active_session_id
        self._active_session_id = None
        logger.info(f"Full-weights session killed: {old_session}")

        # Reset optimizer state (clear momentum buffers)
        if self.optimizer:
            self.optimizer.zero_grad()  # Clear pending gradients
            # Clear Adam m, v buffers
            for state in self.optimizer.state.values():
                state.clear()
            logger.info(f"Optimizer state cleared for session: {old_session}")

        # Reset step counters
        self.global_step = 0
        self.global_forward_backward_step = 0
        logger.info(f"Step counters reset to 0 for session: {old_session}")

        # Note: We intentionally do NOT reload base weights here.
        # The caller is responsible for loading the weights they want via load_state
        # after creating a new session. This avoids blocking the event loop for minutes
        # while reading hundreds of GB from disk.

        return {
            "success": True,
            "message": f"Session '{old_session}' killed successfully.",
            "checkpoint_path": checkpoint_path,
        }

    def _init_parallel_state(self):
        """Initialize parallel state for distributed training."""
        # Get parallel sizes from config
        pp_size = self.train_config.get("pipeline_parallel_size", 1)
        ep_size = self.train_config.get("expert_parallel_size", 1)
        ulysses_size = self.train_config.get("ulysses_parallel_size", 1)
        tp_size = self.train_config.get("tensor_parallel_size", 1)
        ringattn_size = self.train_config.get("ringattn_parallel_size", 1)

        # Calculate dp_size (auto-calculated from world_size and other parallel dims)
        dp_size = self.world_size // (pp_size * ulysses_size * tp_size * ringattn_size)

        # Get dp shard/replicate sizes with proper defaults
        dp_replicate_size = self.train_config.get("data_parallel_replicate_size", 1)
        dp_shard_size = self.train_config.get("data_parallel_shard_size", dp_size)

        init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            pp_size=pp_size,
            tp_size=tp_size,
            ep_size=ep_size,
            ulysses_size=ulysses_size,
            ringattn_size=ringattn_size,
            dp_mode=self.train_config.get("data_parallel_mode", "fsdp2"),
            cp_fsdp_mode=self.train_config.get("cp_fsdp_mode", "all"),
        )

    def _initialize_model(self):
        """Initialize the model using the shared build_training_model pipeline."""
        logger.info(f"Loading model: {self.model_config.get('model_path')}")

        lora_enabled = self.lora_config.get("enable_lora", False)
        enable_mixed_precision = self.train_config.get("enable_mixed_precision", False)
        enable_qlora = self.lora_config.get("enable_qlora", False)

        # Resolve target_modules from Tinker-style or flat config
        target_modules = self._resolve_lora_target_modules() if lora_enabled else None

        # Determine model dtype
        if (lora_enabled or enable_qlora) and enable_mixed_precision:
            model_dtype = "bfloat16"
        elif enable_mixed_precision:
            model_dtype = "float32"
        else:
            model_dtype = "bfloat16"

        pp_size = self.train_config.get("pipeline_parallel_size", 1)
        pp_schedule_name = self.train_config.get("pipeline_parallel_schedule", "1F1B") if pp_size > 1 else None

        result = build_training_model(
            config_path=self.model_config.get("config_path"),
            weights_path=self.model_config.get("model_path"),
            torch_dtype=model_dtype,
            attn_implementation=self.model_config.get("attn_implementation", "sdpa"),
            moe_implementation=self.model_config.get("moe_implementation"),
            ep_dispatch=self.model_config.get("ep_dispatch", "alltoall"),
            deepep_buffer_size_gb=self.model_config.get("deepep_buffer_size_gb", 2.0),
            deepep_num_sms=self.model_config.get("deepep_num_sms", 20),
            deepep_async_combine=self.model_config.get("deepep_async_combine", False),
            init_device=self.train_config.get("init_device", "cpu"),
            merge_qkv=self.model_config.get("merge_qkv", True),
            enable_lora=lora_enabled,
            lora_rank=self.lora_config.get("lora_rank", 32),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            lora_target_modules=target_modules,
            moe_shared_lora=self.lora_config.get("moe_shared_lora", False),
            moe_hybrid_shared_lora=self.lora_config.get("moe_hybrid_shared_lora", False),
            enable_qlora=enable_qlora,
            quant_format=self.lora_config.get("quant_format", "nvfp4"),
            quant_group_size=self.lora_config.get("quant_group_size", 16),
            qlora_exclude_modules=self.lora_config.get("exclude_modules"),
            enable_full_shard=self.train_config.get("enable_full_shard", True),
            enable_mixed_precision=enable_mixed_precision,
            enable_gradient_checkpointing=self.train_config.get("enable_gradient_checkpointing", False),
            enable_compile=self.train_config.get("enable_compile", False),
            basic_modules=self.model_config.get("basic_modules", []),
            enable_reentrant=self.train_config.get("enable_reentrant", False),
            enable_forward_prefetch=self.train_config.get("enable_forward_prefetch", True),
            load_weights_mode=self.train_config.get("load_weights_mode", "broadcast"),
            reshard_after_forward=self.train_config.get("reshard_after_forward"),
            pp_schedule=pp_schedule_name,
            freeze_router=self.train_config.get("freeze_router", False),
            router_fp32=self.model_config.get("router_fp32", True),
            lm_head_fp32=self.model_config.get("lm_head_fp32", True),
            rmsnorm_mode=self.model_config.get("rmsnorm_mode", "native"),
            activation_native=self.model_config.get("activation_native", False),
            rope_native=self.model_config.get("rope_native", False),
            attention_cast_bf16=self.model_config.get("attention_cast_bf16", False),
        )

        self.model = result.model
        self.model_config_obj = result.model_config
        self.pp_enabled = result.pp_enabled
        self.pp_stages = result.pp_stages
        self.model_parts = result.model_parts
        self.has_first_stage = result.has_first_stage
        self.has_last_stage = result.has_last_stage
        self.get_optimizer_pre_hook = result.optimizer_pre_hook_fn
        self.is_prequantized = result.is_prequantized
        self.checkpoint_quant_format = result.checkpoint_quant_format
        self.exclude_modules = result.exclude_modules

        # Save LoRA metadata for checkpoint manager
        if lora_enabled or enable_qlora:
            self.lora_target_modules = target_modules
            self.lora_alpha_value = self.lora_config.get("lora_alpha", 16)

        # Log parameter counts
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded and parallelized on rank {self.rank}")
        logger.info(
            f"  - Trainable params: {trainable_params:,} ({100.0 * trainable_params / max(total_params, 1):.2f}%)"
        )
        logger.info(f"  - Total params: {total_params:,}")

    def _resolve_lora_target_modules(self) -> List[str]:
        """Resolve LoRA target modules from flat or Tinker-style config."""
        explicit_target_modules = self.lora_config.get("lora_target_modules", None)
        if explicit_target_modules is not None:
            return explicit_target_modules

        # Legacy Tinker-style: build from boolean flags
        target_modules = []
        if self.lora_config.get("train_attn", True):
            target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
        if self.lora_config.get("train_mlp", True):
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        if self.lora_config.get("train_unembed", True):
            target_modules.append("lm_head")
        if not target_modules:
            raise ValueError("At least one of train_mlp, train_attn, or train_unembed must be True")
        return target_modules

    def _initialize_optimizer(self):
        """Initialize the optimizer."""
        optimizer_type = self.train_config.get("optimizer", "adamw")
        optimizer_kwargs = None
        if optimizer_type == "muon":
            optimizer_kwargs = {
                k: self.train_config[k]
                for k in ("muon_lr", "muon_momentum", "muon_nesterov", "muon_ns_steps", "muon_adjust_lr_fn")
                if k in self.train_config
            }
        self.optimizer = build_optimizer(
            self.model,
            lr=self.train_config.get("lr", 1e-5),
            weight_decay=self.train_config.get("weight_decay", 0.01),
            fused=True,
            optimizer_type=optimizer_type,
            optimizer_dtype=self.train_config.get("optimizer_dtype", "bf16"),
            optimizer_kwargs=optimizer_kwargs,
        )

        # Register optimizer pre-hook if available
        if self.get_optimizer_pre_hook is not None:
            optimizer_pre_hook = self.get_optimizer_pre_hook(
                self.model, self.model_config_obj, self.train_config.get("data_parallel_mode", "fsdp2")
            )
            self.optimizer.register_step_pre_hook(optimizer_pre_hook)

        logger.info(f"Optimizer initialized: {self.train_config.get('optimizer', 'adamw')}")

    def _initialize_checkpointer(self):
        """Initialize checkpointer for save/load."""
        self.Checkpointer = build_checkpointer(
            dist_backend=self.train_config.get("data_parallel_mode", "fsdp2"),
            ckpt_manager=self.train_config.get("ckpt_manager", "torch_dist"),
        )

    def _initialize_contexts(self):
        """Initialize forward/backward contexts for activation offloading."""
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.train_config.get("enable_activation_offload", False),
            self.train_config.get("enable_gradient_checkpointing", False),
            self.train_config.get("activation_gpu_limit", None),
        )

    def register_lora_adapter(self, model_id: str, lr: float) -> Dict[str, Any]:
        """
        Register a new LoRA adapter for a training run.

        Creates fresh LoRA weights and optimizer state for the given model_id.
        If the model_id already has an adapter, it will be replaced.

        Args:
            model_id: Unique identifier for this training run
            lr: Learning rate for this adapter

        Returns:
            Dictionary with registration info

        Raises:
            RuntimeError: If LoRA is not enabled or adapter manager not initialized
        """
        if self._adapter_manager is None:
            raise RuntimeError("Cannot register adapter: LoRA is not enabled or adapter manager not initialized")

        self._adapter_manager.register_adapter(
            model_id=model_id,
            lr=lr,
            initialize_fresh=True,
        )

        return {
            "model_id": model_id,
            "lr": lr,
            "registered": True,
            "total_adapters": len(self._adapter_manager.list_adapters()),
        }

    def register_adapter(self, model_id: str, lr: float) -> Dict[str, Any]:
        """Alias for register_lora_adapter for API consistency."""
        return self.register_lora_adapter(model_id=model_id, lr=lr)

    # =========================================================================
    # Helper methods for forward/backward deduplication
    # =========================================================================

    def _get_effective_lm_head_weight(self):
        """Get lm_head weight, merging LoRA delta on-the-fly if needed."""
        lm_head = self.model.lm_head
        if isinstance(lm_head, LoraLinear):
            return lm_head.weight + lm_head.get_delta_weight().to(lm_head.weight.dtype)
        return lm_head.weight

    def _collect_per_token_outputs(self, per_token_tensors, micro_batch, accumulators):
        """Gather per-token outputs across Ulysses SP group and append to accumulators."""
        ps = get_parallel_state()

        if ps.cp_enabled:
            ulysses_group = ps.ulysses_group
            cp_size = dist.get_world_size(ulysses_group)

            original_position_ids = micro_batch.get("_original_position_ids")
            if original_position_ids is not None:
                original_seq_len = original_position_ids.shape[-1]
                position_ids = original_position_ids
            else:
                first_tensor = next(iter(per_token_tensors.values()))
                original_seq_len = first_tensor.shape[-1] * cp_size
                position_ids = None

            gathered = {}
            for key, tensor in per_token_tensors.items():
                gathered[key] = gather_outputs(
                    tensor,
                    gather_dim=-1,
                    padding_dim=-1,
                    unpad_dim_size=original_seq_len,
                    scale_grad=False,
                    group=ulysses_group,
                )

            if position_ids is not None:
                accumulators["position_ids"].append(position_ids.cpu())
            else:
                generated_pos_ids = torch.arange(original_seq_len, dtype=torch.long).unsqueeze(0)
                accumulators["position_ids"].append(generated_pos_ids)
        else:
            gathered = per_token_tensors
            position_ids = micro_batch.get("position_ids")

            if position_ids is not None:
                accumulators["position_ids"].append(position_ids.cpu())
            else:
                seq_len = gathered["logprobs"].shape[-1] + 1
                generated_pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                accumulators["position_ids"].append(generated_pos_ids)

        accumulators["logprobs"].append(gathered["logprobs"].cpu())
        if gathered.get("loss") is not None:
            accumulators["losses"].append(gathered["loss"].cpu())

    @staticmethod
    def _accumulate_is_metrics(accumulated, new_metrics):
        """Accumulate importance sampling metrics across micro-batches."""
        if not new_metrics:
            return
        new_metrics.pop("_n_valid_kl", None)
        n_tokens = new_metrics.get("valid_tokens", 1)
        for k, v in new_metrics.items():
            if k not in accumulated:
                accumulated[k] = {"sum": 0.0, "count": 0}
            if k == "valid_tokens":
                accumulated[k]["sum"] += v
                accumulated[k]["count"] += 1
            else:
                accumulated[k]["sum"] += v * n_tokens
                accumulated[k]["count"] += n_tokens

    @staticmethod
    def _finalize_is_metrics(accumulated, result):
        """All-reduce IS metrics across DP group, then add averaged values to result dict."""
        if not accumulated:
            return
        ps = get_parallel_state()
        if ps.dp_enabled:
            dp_group = ps.dp_group
            for k, v in accumulated.items():
                if k == "ratio_min":
                    t = torch.tensor(v["sum"] if v["count"] > 0 else float("inf"), dtype=torch.float64, device="cuda")
                    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=dp_group)
                    v["sum"] = t.item() if t.item() != float("inf") else v["sum"]
                    v["count"] = 1
                elif k == "ratio_max":
                    t = torch.tensor(v["sum"] if v["count"] > 0 else float("-inf"), dtype=torch.float64, device="cuda")
                    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dp_group)
                    v["sum"] = t.item() if t.item() != float("-inf") else v["sum"]
                    v["count"] = 1
                else:
                    sum_t = torch.tensor(v["sum"], dtype=torch.float64, device="cuda")
                    count_t = torch.tensor(float(v["count"]), dtype=torch.float64, device="cuda")
                    dist.all_reduce(sum_t, op=dist.ReduceOp.SUM, group=dp_group)
                    dist.all_reduce(count_t, op=dist.ReduceOp.SUM, group=dp_group)
                    v["sum"] = sum_t.item()
                    v["count"] = count_t.item()
        for k, v in accumulated.items():
            if v["count"] > 0:
                result[f"is_{k}"] = v["sum"] / v["count"]

    def _count_global_valid_tokens(self, micro_batches):
        """Count valid tokens across all micro-batches and all-reduce across DP group.

        Uses fsdp_group (not world group) when PP is enabled, so that PP ranks
        processing the same data don't double-count valid tokens.
        """
        group = get_parallel_state().fsdp_group if self.pp_enabled else None
        return count_valid_tokens(micro_batches, group=group)

    # =========================================================================
    # Loss computation dispatch
    # =========================================================================

    def _compute_micro_batch_loss(self, micro_batch, loss_fn, loss_fn_params):
        """Compute loss for a single micro-batch. Returns (loss, per_token_outputs_dict, is_metrics, model_outputs)."""
        params = loss_fn_params or {}
        return_per_token = params.get("return_per_token", True)

        exclude_keys = self._LOSS_EXCLUDE_KEYS.get(loss_fn, set())
        model_inputs = {k: v for k, v in micro_batch.items() if k not in exclude_keys}

        outputs = self.model(**model_inputs, use_cache=False, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state
        effective_weight = self._get_effective_lm_head_weight()

        per_token_outputs = {}
        is_metrics = None

        if loss_fn in ["causallm_loss", "cross_entropy"]:
            labels = micro_batch.get("labels")
            _result = causallm_loss_function(
                hidden_states=hidden_states,
                weight=effective_weight,
                labels=labels,
                return_per_token=return_per_token,
                ce_mode=self.ce_mode,
                lm_head_fp32=self.lm_head_fp32,
            )
            loss = _result.loss
            if return_per_token:
                per_token_outputs["logprobs"] = _result.per_token_logprobs
                per_token_outputs["loss"] = _result.per_token_loss

        elif loss_fn == "importance_sampling":
            target_tokens = micro_batch.get("target_tokens", micro_batch.get("labels"))
            old_logprobs = micro_batch["logprobs"]
            advantages = micro_batch["advantages"]
            compute_kl_stats = params.get("compute_kl_stats", False)

            _result = importance_sampling_loss_function(
                hidden_states=hidden_states,
                weight=effective_weight,
                labels=target_tokens,
                old_logprobs=old_logprobs,
                advantages=advantages,
                ce_mode=self.ce_mode,
                compute_kl_stats=compute_kl_stats,
                lm_head_fp32=self.lm_head_fp32,
            )
            loss = _result.loss
            per_token_outputs["logprobs"] = _result.per_token_logprobs
            is_metrics = _result.metrics

            if compute_kl_stats and get_parallel_state().cp_enabled and is_metrics:
                is_metrics = _sp_allreduce_kl_metrics(is_metrics, get_parallel_state().ulysses_group)

            # Diagnostic top-k extraction (forward-only feature, rarely used)
            diagnostic_topk = params.get("diagnostic_topk", 0)
            if diagnostic_topk > 0:
                with torch.no_grad():
                    H = hidden_states.size(-1)
                    hs_flat = hidden_states.reshape(-1, H)
                    diag_logits = (hs_flat @ effective_weight.t()).float()
                    diag_log_probs = F.log_softmax(diag_logits, dim=-1)

                    valid = target_tokens.reshape(-1) != IGNORE_INDEX
                    valid_indices = valid.nonzero(as_tuple=True)[0]
                    valid_log_probs = diag_log_probs[valid_indices]

                    # Top-k predictions
                    topk_vals, topk_ids = valid_log_probs.topk(diagnostic_topk, dim=-1)

                    # Target token logprob and rank
                    target_ids = target_tokens.reshape(-1)[valid_indices]
                    target_lps = valid_log_probs[
                        torch.arange(len(valid_indices), device=valid_log_probs.device), target_ids
                    ]
                    target_ranks = (valid_log_probs > target_lps.unsqueeze(-1)).sum(dim=-1) + 1

                    # Entropy: -sum(p * log p)
                    diag_probs = diag_log_probs[valid_indices].exp()
                    entropy = -(diag_probs * diag_log_probs[valid_indices]).sum(dim=-1)

                    # Save to file
                    diag_path = params.get("diagnostic_path", "outputs/xorl_diagnostic.pt")
                    ps = get_parallel_state()
                    cp_rank = ps.cp_rank if ps.cp_enabled else 0
                    diag_path_ranked = f"{diag_path}.rank{cp_rank}"
                    os.makedirs(os.path.dirname(diag_path_ranked) or ".", exist_ok=True)
                    torch.save(
                        {
                            "topk_logprobs": topk_vals.cpu(),
                            "topk_ids": topk_ids.cpu(),
                            "target_ids": target_ids.cpu(),
                            "target_logprobs": target_lps.cpu(),
                            "target_ranks": target_ranks.cpu(),
                            "entropy": entropy.cpu(),
                            "valid_positions": valid_indices.cpu(),
                            "cp_rank": cp_rank,
                        },
                        diag_path_ranked,
                    )
                    logger.info(
                        f"Diagnostic top-{diagnostic_topk} saved to {diag_path_ranked} "
                        f"({len(valid_indices)} valid positions, cp_rank={cp_rank})"
                    )

                    del diag_logits, diag_log_probs, diag_probs, valid_log_probs

        elif loss_fn == "policy_loss":
            target_tokens = micro_batch.get("target_tokens", micro_batch.get("labels"))
            old_logprobs = micro_batch["logprobs"]
            advantages = micro_batch["advantages"]
            rollout_logprobs = micro_batch.get("rollout_logprobs")

            eps_clip = params.get("eps_clip", 0.2)
            eps_clip_high = params.get("eps_clip_high", 0.2)
            eps_clip_c = params.get("eps_clip_c", None)
            use_tis = params.get("use_tis", False)
            tis_clip_low = params.get("tis_clip_low", 0.1)
            tis_clip_high = params.get("tis_clip_high", 2.0)
            num_chunks = params.get("num_chunks", 8)
            compute_kl_stats = params.get("compute_kl_stats", False)
            icepop_beta = params.get("icepop_beta", None)

            if use_tis and rollout_logprobs is None:
                logger.warning("use_tis=True but rollout_logprobs not provided.")

            _result = policy_loss_function(
                hidden_states=hidden_states,
                weight=effective_weight,
                labels=target_tokens,
                old_logprobs=old_logprobs,
                advantages=advantages,
                rollout_logprobs=rollout_logprobs,
                eps_clip=eps_clip,
                eps_clip_high=eps_clip_high,
                eps_clip_c=eps_clip_c,
                use_tis=use_tis,
                tis_clip_low=tis_clip_low,
                tis_clip_high=tis_clip_high,
                ce_mode=self.ce_mode,
                num_chunks=num_chunks,
                compute_kl_stats=compute_kl_stats,
                lm_head_fp32=self.lm_head_fp32,
                icepop_beta=icepop_beta,
            )
            loss = _result.loss
            per_token_outputs["logprobs"] = _result.per_token_logprobs
            is_metrics = _result.metrics

            if compute_kl_stats and get_parallel_state().cp_enabled and is_metrics:
                is_metrics = _sp_allreduce_kl_metrics(is_metrics, get_parallel_state().ulysses_group)

        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        return loss, per_token_outputs, is_metrics, outputs

    # =========================================================================
    # Per-sample K3 KL divergence
    # =========================================================================

    @staticmethod
    def _compute_per_sample_k3(
        k3_values: torch.Tensor,
        valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> List[float]:
        """Compute per-sample K3 KL divergence from per-token K3 values.

        Uses scatter_add to aggregate per-token K3 values into per-sample means,
        where sample boundaries are determined from position_ids (position resets
        indicate new samples in packed sequences).

        Args:
            k3_values: Per-token K3 values, shape (T,)
            valid_mask: Valid token mask, shape (T,)
            position_ids: Position IDs, shape (T,). Position resets mark sample boundaries.

        Returns:
            List of per-sample mean K3 values.
        """
        if k3_values.numel() == 0:
            return []

        # Detect sample boundaries from position_ids
        # A new sample starts where position decreases or resets
        pos = position_ids.view(-1)
        sample_starts = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        sample_starts[0] = 0
        if pos.shape[0] > 1:
            resets = (pos[1:] <= pos[:-1]).long()
            sample_starts[1:] = resets
        sample_ids = sample_starts.cumsum(0)
        num_samples = sample_ids.max().item() + 1

        # Aggregate K3 per sample using scatter_add
        masked_k3 = k3_values.masked_fill(~valid_mask, 0.0)
        per_sample_k3_sum = torch.zeros(num_samples, device=k3_values.device, dtype=k3_values.dtype)
        per_sample_k3_sum.scatter_add_(0, sample_ids, masked_k3)

        per_sample_count = torch.zeros(num_samples, device=k3_values.device, dtype=k3_values.dtype)
        per_sample_count.scatter_add_(0, sample_ids, valid_mask.float())

        # Mean K3 per sample (0.0 for samples with no valid tokens)
        per_sample_k3 = per_sample_k3_sum / per_sample_count.clamp(min=1)
        return per_sample_k3.tolist()

    # =========================================================================
    # Unified forward loop
    # =========================================================================

    def _forward_loop(
        self,
        micro_batches,
        loss_fn,
        loss_fn_params,
        *,
        compute_backward=True,
        r3_enabled=False,
        model_id="default",
        abort_callback=None,
    ):
        """Core forward (+ optional backward) loop shared between forward and forward_backward."""
        params = loss_fn_params or {}
        return_per_token = params.get("return_per_token", True)

        # Count valid tokens globally
        global_valid_tokens = self._count_global_valid_tokens(micro_batches)

        if abort_callback and abort_callback():
            raise RuntimeError("Execution aborted by request")

        total_loss = 0.0
        accumulated_is_metrics = {}
        accumulators = {"logprobs": [], "losses": [], "position_ids": []}

        # Per-sample K3 deferred computation
        compute_per_sample_k3 = params.get("compute_per_sample_k3", False)
        deferred_k3: List[Dict] = []  # each entry: {k3_values, valid_mask, position_ids}

        for batch_idx, micro_batch in enumerate(micro_batches):
            if abort_callback and abort_callback():
                raise RuntimeError("Execution aborted by request")

            micro_batch = {
                k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in micro_batch.items()
            }

            labels = micro_batch.get("labels", micro_batch.get("target_tokens"))
            local_valid_tokens = (
                (labels != IGNORE_INDEX).sum() if labels is not None else torch.tensor(0, device=get_device_type())
            )

            # R3: switch to replay_forward so MoEBlock pops pre-populated routing
            if r3_enabled:
                set_replay_stage("replay_forward")

            # Forward pass + loss computation
            with self.model_fwd_context:
                loss, per_token_outputs, is_metrics, outputs = self._compute_micro_batch_loss(
                    micro_batch, loss_fn, params
                )

            logger.debug(
                f"Rank {self.rank}: micro_batch {batch_idx}/{len(micro_batches)} "
                f"loss={loss.item():.6f}, local_valid_tokens={local_valid_tokens.item()}, "
                f"global_valid_tokens={global_valid_tokens.item()}"
            )
            # Note: loss is always finite even when local_valid_tokens=0, because
            # causallm_loss_function uses reduction="none" + manual mean with
            # clamp(min=1) denominator. No need to replace with zeros_like
            # (which would break the autograd graph and cause FSDP2 deadlocks).

            # SP gather per-token outputs
            if per_token_outputs:
                self._collect_per_token_outputs(per_token_outputs, micro_batch, accumulators)

            # Deferred per-sample K3 computation
            if compute_per_sample_k3 and per_token_outputs and "logprobs" in per_token_outputs:
                with torch.no_grad():
                    new_lp = per_token_outputs["logprobs"].view(-1)
                    old_lp = micro_batch.get("logprobs", micro_batch.get("old_logprobs"))
                    if old_lp is not None:
                        old_lp = old_lp.view(-1).to(new_lp.device)
                        _labels = micro_batch.get("labels", micro_batch.get("target_tokens"))
                        _valid = (
                            (_labels.view(-1) != IGNORE_INDEX)
                            if _labels is not None
                            else torch.ones_like(new_lp, dtype=torch.bool)
                        )
                        log_ratio = new_lp - old_lp
                        k3_vals = (torch.exp(log_ratio) - log_ratio - 1.0).masked_fill(~_valid, 0.0)
                        # position_ids and _original_position_ids are both kept
                        # unsharded (full packed sequence length) for cu_seq_lens.
                        # Slice to match local token count (k3_vals length) using
                        # the Ulysses/CP shard boundaries.
                        _pos = micro_batch.get("_original_position_ids", micro_batch.get("position_ids"))
                        if _pos is not None:
                            _pos_flat = _pos.view(-1)
                            local_len = k3_vals.shape[0]
                            if _pos_flat.shape[0] > local_len:
                                # Slice position_ids to this rank's Ulysses shard
                                ps = get_parallel_state()
                                cp_rank = ps.ulysses_rank if ps.ulysses_enabled else 0
                                start = cp_rank * local_len
                                _pos_flat = _pos_flat[start : start + local_len]
                            deferred_k3.append(
                                {
                                    "k3_values": k3_vals.cpu(),
                                    "valid_mask": _valid.cpu(),
                                    "position_ids": _pos_flat.cpu(),
                                }
                            )

            # Gradient accumulation — raw (unnormalized) backward.
            # Normalization by total accumulated valid tokens is deferred to optim_step.
            # FSDP's automatic gradient averaging is disabled (set_gradient_divide_factor(1.0)
            # in torch_parallelize), so no fsdp_size compensation is needed here.
            # When local_valid_tokens=0, this produces 0 gradients while preserving
            # the full autograd graph through all parameters (including lm_head weight),
            # which is critical for FSDP2 reduce-scatter collectives.
            if compute_backward:
                ps = get_parallel_state()
                raw_loss = loss * local_valid_tokens.detach().float()

                if abort_callback and abort_callback():
                    raise RuntimeError("Execution aborted by request")

                # R3: switch to replay_backward so grad ckpt recompute pops same routing
                if r3_enabled:
                    set_replay_stage("replay_backward")

                with self.model_bwd_context:
                    raw_loss.backward()

                # Loss reporting (separately, no grad): compute normalized per-token loss
                with torch.no_grad():
                    loss_report = loss.detach() * local_valid_tokens
                    dist.all_reduce(loss_report, op=dist.ReduceOp.SUM, group=ps.fsdp_group if self.pp_enabled else None)
                    if global_valid_tokens.item() > 0:
                        total_loss += (loss_report / global_valid_tokens).item()
            else:
                # Forward-only: accumulate weighted loss
                if global_valid_tokens.item() > 0:
                    total_loss += loss.item() * (local_valid_tokens.item() / global_valid_tokens.item())

            # Accumulate IS metrics
            self._accumulate_is_metrics(accumulated_is_metrics, is_metrics)

            # Cleanup
            del micro_batch, outputs, loss
            if compute_backward:
                del raw_loss

        # Note: gc.collect() + empty_cache() removed from per-step path.
        # They cost ~250ms + ~50ms per step (profiled on Qwen3-8B 8xH100).
        # Cleanup happens at checkpoint save instead.

        # R3 cleanup
        if r3_enabled:
            self._routing_handler.cleanup()

        # CP/SP gradient sync (backward only)
        if compute_backward:
            sync_sp_gradients(self.model, get_parallel_state().sp_grad_sync_group)
            # Accumulate valid tokens for deferred normalization at optim_step
            self._accumulated_valid_tokens[model_id] = (
                self._accumulated_valid_tokens.get(model_id, 0) + global_valid_tokens.item()
            )

        # Build result
        result = {
            "total_loss": total_loss,
            "global_valid_tokens": global_valid_tokens.item(),
        }

        if accumulators["logprobs"]:
            result["packed_logprobs"] = [t.tolist() for t in accumulators["logprobs"]]
            if accumulators["losses"]:
                result["packed_losses"] = [t.tolist() for t in accumulators["losses"]]
            if accumulators["position_ids"]:
                result["packed_position_ids"] = [t.tolist() for t in accumulators["position_ids"]]

        # Compute deferred per-sample K3
        if deferred_k3:
            all_per_sample_k3 = []
            for entry in deferred_k3:
                per_sample = self._compute_per_sample_k3(entry["k3_values"], entry["valid_mask"], entry["position_ids"])
                all_per_sample_k3.extend(per_sample)
            result["per_sample_k3"] = all_per_sample_k3

        # All-reduce IS metrics across DP and add to result
        self._finalize_is_metrics(accumulated_is_metrics, result)

        synchronize()
        return result

    # =========================================================================
    # Pipeline Parallelism support
    # =========================================================================

    def _get_pp_schedule(self, n_microbatches, seq_len):
        """Return a cached PP schedule keyed by (n_microbatches, seq_len).

        A new PipelineStage (cheap, no deepcopy) is created for each unique
        seq_len so P2P buffers match the actual tensor shape.
        """
        key = (n_microbatches, seq_len)
        if key not in self._pp_schedule_cache:
            ps = get_parallel_state()
            stage = build_pp_stage(
                self.model_parts[0],
                pp_rank=ps.pp_rank,
                num_stages=ps.pp_size,
                device=get_device_type(),
                pp_group=ps.pp_group,
            )
            self._pp_schedule_cache[key] = build_pipeline_schedule(
                stages=[stage],
                n_microbatches=n_microbatches,
                loss_fn=pp_loss_fn,
                schedule_name=self.train_config.get("pipeline_parallel_schedule", "1F1B"),
            )
        return self._pp_schedule_cache[key]

    def _forward_backward_pp(self, micro_batches, global_valid_tokens):
        """Pipeline parallel forward-backward step.

        With pp_variable_seq_lengths: negotiates per-step max seq_len across PP
        ranks and pads micro-batches to that length before running the schedule.

        Returns raw CE_sum (unnormalized); the caller accumulates
        global_valid_tokens into _accumulated_valid_tokens so that
        optim_step can normalize gradients by the full accumulated total.
        """
        ps = get_parallel_state()
        if self.train_config.get("pp_variable_seq_lengths", False):
            seq_len = negotiate_pp_seq_len(micro_batches, ps.pp_group)
            pad_micro_batches_for_pp(
                micro_batches,
                sample_packing_sequence_len=seq_len * ps.cp_size,
                sp_size=ps.cp_size,
                pad_to_multiple_of=self.train_config.get("pad_to_multiple_of", 1),
            )
        else:
            seq_len = micro_batches[0]["input_ids"].shape[-1]

        pp_schedule = self._get_pp_schedule(len(micro_batches), seq_len)
        return forward_backward_pp(
            model_parts=self.model_parts,
            pp_schedule=pp_schedule,
            micro_batches=micro_batches,
            has_first_stage=self.has_first_stage,
            has_last_stage=self.has_last_stage,
            pp_group=ps.pp_group,
        )

    # =========================================================================
    # Forward and backward passes
    # =========================================================================

    def forward_backward(
        self,
        micro_batches: List[Dict[str, Any]],
        loss_fn: str = "causallm_loss",
        loss_fn_params: Optional[Dict[str, Any]] = None,
        abort_callback: Optional[callable] = None,
        model_id: str = "default",
        routed_experts: Optional[List[List[List[List[int]]]]] = None,
        routed_expert_logits: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute forward and backward pass with gradient accumulation.

        Args:
            micro_batches: List of dictionaries containing input tensors (input_ids, attention_mask, labels)
            loss_fn: Loss function to use (e.g., "causallm_loss", "importance_sampling", "policy_loss")
            loss_fn_params: Optional global parameters for the loss function (e.g., eps_clip, use_tis for PPO,
                           return_per_token for tinker-compatible per-token outputs)
            abort_callback: Optional callback to check if execution should be aborted
            model_id: The model_id for multi-adapter training (default: "default")
            routed_experts: Optional R3 routing data from inference.
                           Shape: [batch, num_tokens, num_layers, topk]
                           When provided, MoE layers will replay these routing decisions
                           instead of recomputing top-k, ensuring consistency with inference.
            routed_expert_logits: Optional R3 routing weights from inference.
                           Same format as routed_experts but contains float32 softmax weights.
                           When provided alongside routed_experts, MoE layers use these exact
                           routing weights instead of recomputing softmax.

        Returns:
            Dictionary with loss and other metrics. If return_per_token=True (default), also includes:
                - packed_logprobs: List of per-token log probabilities (one tensor per micro-batch)
                - packed_losses: List of per-token losses (one tensor per micro-batch)
                - packed_position_ids: List of position_ids (one tensor per micro-batch)

        Raises:
            RuntimeError: If execution is aborted via abort_callback or model is in sleep mode
            ValueError: If token IDs or labels are out of vocab range
        """
        self._check_not_sleeping("forward_backward")

        # Validate single-tenant mode for full-weights training
        self._validate_single_tenant(model_id)

        # Switch to the correct adapter for this model_id
        if self._adapter_manager is not None:
            self._adapter_manager.switch_adapter(model_id, auto_register=True)

        # Validate token IDs before processing to catch out-of-vocab errors early
        # This prevents CUDA device-side asserts that can hang the server
        validate_token_ids(micro_batches, self.model.config.vocab_size)

        start_time = time.time()

        # Get return_per_token flag from loss_fn_params (default True for tinker compatibility)
        params = loss_fn_params or {}

        # Reference forward pass: compute Xorl's own logprobs to replace SGLang logprobs
        # This guarantees KL=0 at step 0 since both old and new logprobs come from the same engine
        compute_ref_logprobs = params.get("compute_ref_logprobs", False)
        if compute_ref_logprobs and loss_fn in ["policy_loss", "importance_sampling"]:
            logger.info("Computing reference logprobs via no-grad forward pass")

            # Set up R3 routing for ref pass if needed (separate from main pass)
            ref_r3_enabled = self._routing_handler.setup(micro_batches, routed_experts, routed_expert_logits)

            with torch.no_grad():
                if ref_r3_enabled:
                    set_replay_stage("replay_forward")

                for batch_idx, micro_batch in enumerate(micro_batches):
                    mb = {
                        k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in micro_batch.items()
                    }

                    model_inputs = {
                        k: v
                        for k, v in mb.items()
                        if k not in ["labels", "target_tokens", "logprobs", "advantages", "rollout_logprobs"]
                    }

                    with self.model_fwd_context:
                        outputs = self.model(**model_inputs, use_cache=False, output_hidden_states=False)
                    hidden_states = outputs.last_hidden_state

                    effective_weight = self._get_effective_lm_head_weight()

                    labels = mb.get("target_tokens", mb.get("labels"))

                    # Compute per-token logprobs using same CE path as training
                    _ref_result = causallm_loss_function(
                        hidden_states=hidden_states,
                        weight=effective_weight,
                        labels=labels,
                        return_per_token=True,
                        ce_mode=self.ce_mode,
                        lm_head_fp32=self.lm_head_fp32,
                    )
                    ref_logprobs = _ref_result.per_token_logprobs

                    # Diagnostic: log SGLang vs Xorl logprobs comparison for first micro-batch
                    if batch_idx == 0:
                        orig_lp = micro_batch["logprobs"]
                        valid_mask = labels.view(-1) != IGNORE_INDEX
                        if orig_lp is not None and valid_mask.any():
                            orig_flat = orig_lp.to(ref_logprobs.device).view(-1)
                            ref_flat = ref_logprobs.view(-1)
                            n_show = min(10, valid_mask.sum().item())
                            valid_idx = valid_mask.nonzero(as_tuple=True)[0][:n_show]
                            logger.debug(
                                f"Ref logprobs: SGLang={orig_flat[valid_idx].tolist()}, "
                                f"Xorl={ref_flat[valid_idx].tolist()}"
                            )

                    # Replace SGLang logprobs with Xorl-computed ref logprobs (CPU, matching original format)
                    micro_batches[batch_idx]["logprobs"] = ref_logprobs.cpu()

                    # Clean up to free GPU memory before next micro-batch
                    del mb, outputs, hidden_states, effective_weight, labels, ref_logprobs

            # Clean up ref routing (clears RoutingReplay instances for re-population by main pass)
            if ref_r3_enabled:
                self._routing_handler.cleanup()

            logger.info("Reference logprobs computed, replacing old_logprobs")

        # R3 (Rollout Routing Replay): Pre-populate routing replay from inference data
        r3_enabled = self._routing_handler.setup(micro_batches, routed_experts, routed_expert_logits)

        # PP path
        if self.pp_enabled:
            global_valid_tokens = self._count_global_valid_tokens(micro_batches)
            # Static padding: pad to sample_packing_sequence_len upfront.
            # With pp_variable_seq_lengths, padding is deferred to _forward_backward_pp.
            if not self.train_config.get("pp_variable_seq_lengths", False):
                pad_micro_batches_for_pp(
                    micro_batches,
                    sample_packing_sequence_len=self.train_config.get("sample_packing_sequence_len", 0),
                    sp_size=get_parallel_state().cp_size,
                    pad_to_multiple_of=self.train_config.get("pad_to_multiple_of", 1),
                )
            raw_total_loss = self._forward_backward_pp(micro_batches, global_valid_tokens)
            # raw_total_loss = sum of CE_sum across micro-batches (unnormalized).
            # Normalize for reporting: divide by global_valid_tokens.
            gvt = global_valid_tokens.item()
            reported_loss = raw_total_loss / gvt if gvt > 0 else 0.0
            result = {
                "total_loss": reported_loss,
                "global_valid_tokens": gvt,
            }
            # Accumulate valid tokens for deferred normalization at optim_step
            self._accumulated_valid_tokens[model_id] = self._accumulated_valid_tokens.get(model_id, 0) + gvt
            # R3 cleanup for PP path (stage management handled by _pp_forward)
            if r3_enabled:
                self._routing_handler.cleanup()
        else:
            # Standard forward-backward via unified loop
            result = self._forward_loop(
                micro_batches,
                loss_fn,
                loss_fn_params,
                compute_backward=True,
                r3_enabled=r3_enabled,
                model_id=model_id,
                abort_callback=abort_callback,
            )

        # Capture gradients into adapter's own parameters (for multi-adapter isolation)
        # This must happen AFTER all micro-batches have accumulated gradients in model params,
        # but BEFORE optim_step. Each adapter has its own .grad slots to prevent collision.
        if self._adapter_manager is not None:
            self._adapter_manager.capture_gradients(model_id)

        # Get step counter (use adapter manager if available, else global)
        if self._adapter_manager is not None:
            current_step = self._adapter_manager.get_adapter_state(model_id).global_forward_backward_step
        else:
            current_step = self.global_forward_backward_step

        # Add timing and identity info
        result["step"] = current_step
        result["forward_backward_time"] = time.time() - start_time
        result["model_id"] = model_id

        # Increment step counter (use adapter manager if available, else global)
        if self._adapter_manager is not None:
            self._adapter_manager.increment_forward_backward_step(model_id)
        else:
            self.global_forward_backward_step += 1

        # Synchronize to ensure all async GPU operations complete
        synchronize()

        # Collect and write MoE metrics (only if enabled)
        if self._moe_tracker.enabled:
            moe_metrics = self._moe_tracker.collect(result["step"], result["forward_backward_time"])
            moe_metrics["total_loss"] = result["total_loss"]
            moe_metrics["model_id"] = model_id
            self._moe_tracker.write(moe_metrics)

            # Add summary metrics to result for API response
            if "expert_load" in moe_metrics:
                result["expert_load_summary"] = {
                    "mean_imbalance_ratio": moe_metrics["expert_load"]["mean_imbalance_ratio"],
                    "max_imbalance_ratio": moe_metrics["expert_load"]["max_imbalance_ratio"],
                }

        logger.info(
            f"forward_backward step={result['step']} loss={result['total_loss']:.4f} "
            f"tokens={result.get('global_valid_tokens', 'N/A')} "
            f"time={result['forward_backward_time']:.2f}s"
        )
        logger.debug(
            f"Rank {self.rank}: forward_backward step={result['step']} "
            f"loss={result['total_loss']:.6f}, "
            f"global_valid_tokens={result.get('global_valid_tokens', 'N/A')}, "
            f"n_micro_batches={len(micro_batches)}, loss_fn={loss_fn}, "
            f"model_id={model_id}, time={result['forward_backward_time']:.3f}s"
        )

        return result

    @torch.no_grad()
    def forward(
        self,
        micro_batches: List[Dict[str, Any]],
        loss_fn: str = "causallm_loss",
        loss_fn_params: Optional[Dict[str, Any]] = None,
        routed_experts: Optional[List] = None,
        routed_expert_logits: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute forward pass only (no gradient computation)."""
        self._check_not_sleeping("forward")
        validate_token_ids(micro_batches, self.model.config.vocab_size)

        start_time = time.time()

        r3_enabled = self._routing_handler.setup(micro_batches, routed_experts, routed_expert_logits)

        result = self._forward_loop(
            micro_batches,
            loss_fn,
            loss_fn_params,
            compute_backward=False,
            r3_enabled=r3_enabled,
        )

        result["step"] = self.global_forward_backward_step
        result["forward_time"] = time.time() - start_time

        logger.info(
            f"forward loss={result['total_loss']:.4f} "
            f"tokens={result.get('global_valid_tokens', 'N/A')} "
            f"time={result['forward_time']:.2f}s"
        )
        logger.debug(
            f"Rank {self.rank}: forward loss={result['total_loss']:.6f}, "
            f"global_valid_tokens={result.get('global_valid_tokens', 'N/A')}, "
            f"n_micro_batches={len(micro_batches)}, loss_fn={loss_fn}, "
            f"time={result['forward_time']:.3f}s"
        )
        return result

    # =========================================================================
    # Optimizer step
    # =========================================================================

    def optim_step(
        self,
        gradient_clip: Optional[float] = None,
        lr: Optional[float] = None,
        model_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Execute optimizer step using xorl's optimizer logic.

        For multi-adapter training: Uses the adapter's own optimizer and parameters.
        The gradients were previously captured into the adapter's params via capture_gradients().

        For single-adapter training: Uses the shared optimizer on model parameters.

        Args:
            gradient_clip: Optional gradient clipping value
            lr: Learning rate to set for this step (overrides adapter's lr if provided)
            model_id: The model_id for multi-adapter training (default: "default")

        Returns:
            Dictionary with optimizer metrics

        Raises:
            RuntimeError: If model is in sleep mode
        """
        self._check_not_sleeping("optim_step")

        start_time = time.time()

        # Determine gradient clip value
        # If gradient_clip is provided, use it; otherwise fall back to config
        # Default to a large value (10000.0) to ensure we always use the
        # distributed-aware clip_grad_norm_ path for correct grad norm computation.
        # This effectively means "no clipping" while still computing grad_norm correctly.
        DEFAULT_MAX_GRAD_NORM = 10000.0
        if gradient_clip is not None:
            clip_value = gradient_clip
        else:
            clip_value = self.train_config.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM)

        # Pop accumulated valid tokens for this model_id (deferred normalization)
        accumulated = self._accumulated_valid_tokens.pop(model_id, 0)

        # Multi-adapter path: use adapter's own optimizer on adapter's own parameters
        if self._adapter_manager is not None:
            # Determine learning rate: explicit lr > adapter's stored lr
            if lr is not None:
                effective_lr = lr
            else:
                effective_lr = self._adapter_manager.get_lr(model_id)

            # Step the adapter's optimizer (handles LR update, grad clip, step, zero_grad)
            # Gradients are in the adapter's params (captured by capture_gradients in forward_backward)
            # Pass accumulated_valid_tokens for deferred gradient normalization
            grad_norm = self._adapter_manager.optim_step(
                model_id,
                effective_lr,
                clip_value,
                accumulated_valid_tokens=accumulated,
            )
            current_step = self._adapter_manager.get_global_step(model_id)
            current_lr = effective_lr

        # Single-adapter path: use shared optimizer on model parameters
        else:
            # Deferred gradient normalization: scale raw gradients by 1/accumulated_valid_tokens
            # Use in-place mul_ to preserve DTensor metadata (FSDP2 grads are DTensors).
            if accumulated > 0:
                scale = 1.0 / accumulated
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)

            # Determine learning rate
            if lr is not None:
                effective_lr = lr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = effective_lr

            ps = get_parallel_state()

            grad_norm = clip_gradients(
                self.model,
                clip_value,
                pp_enabled=self.pp_enabled,
                pp_group=ps.pp_group if self.pp_enabled else None,
            )

            # Optimizer step
            self.optimizer.step()
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                # MultiOptimizer (Muon) doesn't support set_to_none kwarg
                self.optimizer.zero_grad()
            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            self.global_step += 1
            current_step = self.global_step
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Collect mean grad_norm across data parallel group for logging
            grad_norm = all_reduce(grad_norm, group=ps.fsdp_group)

        # Periodic LoRA merge (if configured)
        self._maybe_merge_lora()

        result = {
            "step": current_step,
            "grad_norm": grad_norm,
            "lr": current_lr,
            "optim_step_time": time.time() - start_time,
            "model_id": model_id,
        }

        logger.info(
            f"optim_step step={current_step} grad_norm={grad_norm:.4f} "
            f"lr={current_lr:.2e} time={result['optim_step_time']:.2f}s"
        )
        logger.debug(
            f"Rank {self.rank}: optim_step step={current_step}, "
            f"grad_norm={grad_norm:.6f}, lr={current_lr:.2e}, "
            f"clip={clip_value}, accumulated_valid_tokens={accumulated}, "
            f"model_id={model_id}, time={result['optim_step_time']:.3f}s"
        )

        synchronize()

        return result

    def _maybe_merge_lora(self) -> None:
        """Periodic LoRA merge at merge_lora_interval."""
        _maybe_merge_lora_util(
            self.model,
            enable_lora=self.lora_config.get("enable_lora", False),
            enable_qlora=self.lora_config.get("enable_qlora", False),
            merge_interval=self.lora_config.get("merge_lora_interval", 0),
            global_step=self.global_step,
            optimizer=self.optimizer,
            reset_optimizer=self.lora_config.get("reset_optimizer_on_merge", False),
        )

    # =========================================================================
    # Checkpoint delegation (to CheckpointManager)
    # =========================================================================

    def _sync_checkpoint_state(self):
        """Sync mutable state to checkpoint manager before save operations."""
        self._checkpoint_mgr.global_step = self.global_step
        self._checkpoint_mgr.global_forward_backward_step = self.global_forward_backward_step
        self._checkpoint_mgr.lora_target_modules = getattr(self, "lora_target_modules", None)
        self._checkpoint_mgr.lora_alpha_value = getattr(self, "lora_alpha_value", None)

    def _sync_from_checkpoint_state(self):
        """Sync state back from checkpoint manager after load operations."""
        self.global_step = self._checkpoint_mgr.global_step
        self.global_forward_backward_step = self._checkpoint_mgr.global_forward_backward_step

    def save_adapter_state(self, model_id, path=None, save_optimizer=True):
        self._sync_checkpoint_state()
        return self._checkpoint_mgr.save_adapter_state(model_id, path, save_optimizer)

    def load_adapter_state(self, model_id, path=None, load_optimizer=True, lr=None):
        result = self._checkpoint_mgr.load_adapter_state(model_id, path, load_optimizer, lr=lr)
        self._sync_from_checkpoint_state()
        return result

    def save_state(self, checkpoint_path, save_optimizer=True, model_id=None):
        self._sync_checkpoint_state()
        return self._checkpoint_mgr.save_state(checkpoint_path, save_optimizer, model_id)

    def load_state(self, checkpoint_path, load_optimizer=True, model_id=None):
        result = self._checkpoint_mgr.load_state(checkpoint_path, load_optimizer, model_id)
        self._sync_from_checkpoint_state()
        return result

    def save_lora_only(self, lora_path, model_id="default"):
        self._sync_checkpoint_state()
        return self._checkpoint_mgr.save_lora_only(lora_path, model_id)

    def save_full_weights(self, output_path, dtype="bfloat16", base_model_path=None, distributed_write=True):
        return self._checkpoint_mgr.save_full_weights(output_path, dtype, base_model_path, distributed_write)

    def save_weights_for_sampler(self, checkpoint_path, output_path, save_dtype="bfloat16"):
        return self._checkpoint_mgr.save_weights_for_sampler(checkpoint_path, output_path, save_dtype)

    def extract_model_weights(self):
        return self._checkpoint_mgr.extract_model_weights()

    def extract_full_weights_with_ep(self):
        return self._checkpoint_mgr.extract_full_weights_with_ep()

    # =========================================================================
    # Sleep / Wake
    # =========================================================================

    @torch.no_grad()
    def sleep(self) -> Dict[str, Any]:
        """
        Offload model and optimizer to CPU to free GPU memory.

        Returns:
            Dict with operation timing information
        """
        start_time = time.time()

        # Offload model to CPU
        self.model.to("cpu")

        # Offload optimizer state to CPU
        if self.optimizer and self.optimizer.state:
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    state = self.optimizer.state.get(param)
                    if state:
                        for key, value in state.items():
                            if isinstance(value, torch.Tensor):
                                state[key] = value.to("cpu", non_blocking=True)

        # Synchronize to ensure all transfers complete
        synchronize()

        # Free GPU memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Mark as sleeping
        self.is_sleeping = True

        result = {
            "status": "sleeping",
            "offload_time": time.time() - start_time,
        }

        logger.info(f"Model offloaded to CPU in {result['offload_time']:.2f}s")
        return result

    @torch.no_grad()
    def wake_up(self) -> Dict[str, Any]:
        """
        Load model and optimizer back to GPU.

        Returns:
            Dict with operation timing information
        """
        start_time = time.time()

        device_id = get_device_id()

        # Load model to GPU
        self.model.to(device_id)

        # Load optimizer state to GPU
        if self.optimizer and self.optimizer.state:
            for param_group in self.optimizer.param_groups:
                for param in param_group["params"]:
                    state = self.optimizer.state.get(param)
                    if state:
                        for key, value in state.items():
                            if isinstance(value, torch.Tensor):
                                state[key] = value.to(device_id, non_blocking=True)

        # Synchronize to ensure all transfers complete
        synchronize()

        # Mark as awake
        self.is_sleeping = False

        result = {
            "status": "awake",
            "load_time": time.time() - start_time,
        }

        logger.info(f"Model loaded to GPU in {result['load_time']:.2f}s")
        return result
