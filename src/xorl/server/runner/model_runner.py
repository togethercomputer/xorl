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

import gc
import logging
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, PretrainedConfig

from xorl.checkpoint import build_checkpointer
from xorl.data.constants import IGNORE_INDEX
from xorl.distillation import TeacherActivationCache, TeacherHeadManager
from xorl.distributed.offloading import build_activation_offloading_context
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
from xorl.distributed.pipeline_parallel import build_pipeline_schedule, build_pp_stage
from xorl.distributed.sequence_parallel.data import gather_outputs
from xorl.lora import LoraLinear
from xorl.models.layers.moe.routing_replay import set_replay_stage
from xorl.models.transformers.deepseek_v3.support import deepseek_v3_default_lora_targets
from xorl.models.transformers.glm5.support import glm5_default_lora_targets
from xorl.ops.loss import (
    LossOutput,
    OPDLossMetrics,
    TokenPartial,
    causallm_loss_function,
    importance_sampling_loss_function,
    opd_loss_function,
    policy_loss_function,
)
from xorl.optim import build_optimizer
from xorl.server.runner.adapters import LoRAAdapterManager
from xorl.server.runner.checkpoint import CheckpointManager
from xorl.server.runner.utils import (
    MoeMetricsTracker,
    RoutingReplayHandler,
    batch_slice_rank_and_size,
    ep_duplicate_batches_enabled,
    run_self_test,
    validate_token_ids,
)
from xorl.server.session_spec import build_default_session_spec
from xorl.server.weight_sync.source_delta_capture import (
    snapshot_sparse_delta_tensors,
    sparse_delta_capture_enabled,
    write_sparse_source_delta_rank,
)
from xorl.trainers.model_builder import (
    build_training_model,
    resolve_training_model_dtype,
)
from xorl.trainers.training_utils import (
    clip_gradients,
    count_active_microbatches,
    count_valid_tokens,
    forward_backward_pp,
    get_distsign_grad_scale_factor,
    get_effective_grad_clip_value,
    make_pp_loss_fn,
    negotiate_pp_seq_len,
    pad_micro_batches_for_pp,
    scale_model_gradients,
    sync_sp_gradients,
)
from xorl.trainers.training_utils import (
    maybe_merge_lora as _maybe_merge_lora_util,
)
from xorl.utils import helper
from xorl.utils.device import get_device_id, get_device_type, get_torch_device, synchronize
from xorl.utils.dist_utils import all_reduce
from xorl.utils.seqlen_pos_transform_utils import pos2culen


logger = logging.getLogger(__name__)


# Clamp-frac + region/correctness KL-split metric keys emitted by OPDLossMetrics.
# Shared by the per-teacher-group accumulation in _compute_opd_micro_batch_loss and
# the collective key-set seeding in _ensure_opd_loss_metric_accumulators (every rank
# must enter the loss-metric all-reduce with the same key set).
_OPD_SPLIT_METRIC_KEYS = (
    "opd_loss_clamp_frac",
    "opd_kl_prompt_per_valid",
    "opd_kl_buffer_per_valid",
    "opd_kl_answer_per_valid",
    "opd_frac_prompt",
    "opd_frac_buffer",
    "opd_frac_answer",
    "opd_kl_answer_correct_per_valid",
    "opd_kl_answer_wrong_per_valid",
    "opd_frac_answer_correct",
    "opd_frac_answer_wrong",
    "opd_student_entropy_answer_correct_per_valid",
    "opd_student_entropy_answer_wrong_per_valid",
    "opd_teacher_entropy_answer_correct_per_valid",
    "opd_teacher_entropy_answer_wrong_per_valid",
)


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


def _metric_reduce_op(metric_name: str, metric_ops: Optional[Dict[str, str]] = None) -> str:
    if metric_ops and metric_name in metric_ops:
        return metric_ops[metric_name]
    if metric_name in {"ratio_min", "tis_min"}:
        return "min"
    if metric_name in {"ratio_max", "tis_max"}:
        return "max"
    return "mean"


def _sp_allreduce_kl_metrics(
    metrics: Dict[str, Any],
    sp_group,
    metric_ops: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    All-reduce KL/ratio metrics across the sequence-parallel (Ulysses) group.

    With Ulysses SP, each rank only sees a shard of the sequence. Rank 0 often
    has only prompt tokens (all target_tokens=-100), so its KL stats are zeros.
    This function aggregates stats across all SP ranks so every rank (especially
    rank 0 which reports metrics) sees the correct global values.

    Mean-type metrics are raw partial sums, so they SUM-reduce and remain
    unfinalized. ``valid_tokens`` SUM-reduces alongside them; downstream
    accumulation divides mean metrics by the final token count.
    """
    # Backward-compatible argument order for older tests/call sites:
    # _sp_allreduce_kl_metrics(metrics, metric_ops, sp_group).
    if isinstance(sp_group, dict):
        metric_ops, sp_group = sp_group, metric_ops

    device = torch.device(get_device_type())
    local_n = float(metrics.get("valid_tokens", metrics.get("_n_valid_kl", 0)) or 0)
    metrics["valid_tokens"] = local_n

    n_tensor = torch.tensor(local_n, dtype=torch.float64, device=device)
    dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM, group=sp_group)
    total_n = n_tensor.item()

    for key, value in list(metrics.items()):
        if key in {"valid_tokens", "_n_valid_kl"}:
            continue
        op_name = _metric_reduce_op(key, metric_ops)
        if op_name == "min":
            local_value = float(value) if local_n > 0 else float("inf")
            tensor = torch.tensor(local_value, dtype=torch.float64, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=sp_group)
            reduced = tensor.item()
            metrics[key] = tensor if math.isfinite(reduced) else torch.tensor(1.0, dtype=torch.float64, device=device)
        elif op_name == "max":
            local_value = float(value) if local_n > 0 else float("-inf")
            tensor = torch.tensor(local_value, dtype=torch.float64, device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=sp_group)
            reduced = tensor.item()
            metrics[key] = tensor if math.isfinite(reduced) else torch.tensor(1.0, dtype=torch.float64, device=device)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float64, device=device).clone()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=sp_group)
            metrics[key] = tensor

    metrics["valid_tokens"] = int(total_n) if float(total_n).is_integer() else total_n

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
        # target_tokens/weights are loss-side fields (packing folds them into labels);
        # they must never reach the model forward as kwargs.
        "causallm_loss": {"labels", "target_tokens", "weights", "_original_position_ids", "rollout_logprobs"},
        "cross_entropy": {"labels", "target_tokens", "weights", "_original_position_ids", "rollout_logprobs"},
        "importance_sampling": {
            "labels",
            "target_tokens",
            "logprobs",
            "advantages",
            "_original_position_ids",
            "rollout_logprobs",
        },
        "policy_loss": {
            "labels",
            "target_tokens",
            "logprobs",
            "advantages",
            "_original_position_ids",
            "rollout_logprobs",
        },
        "opd_loss": {
            "labels",
            "target_tokens",
            "teacher_id",
            "teacher_ids",
            "teacher_weight",
            "teacher_weights",
            "hidden_match_weights",
            "teacher_cache_indices",
            "teacher_hidden_states",
            # Metrics-only diagnostic tensors (region / sample-correctness KL splits).
            "opd_region_ids",
            "opd_sample_ok",
            # Multi-layer OPRD trainer-side teacher forward: the teacher sequence and
            # its kept-position indices are a SEPARATE (teacher-length) sequence, not
            # student model inputs — exclude so they never reach the model forward.
            "teacher_input_ids",
            "teacher_kept_indices",
            "teacher_position_ids",
            # Packer-emitted per-micro-batch-local view of teacher_cache_indices
            # (OPRD kept-row gather) + the client's per-sample base scalar.
            "teacher_cache_local_indices",
            "teacher_cache_base",
            "_original_position_ids",
        },
        "teacher_hidden_cache": {
            "labels",
            "target_tokens",
            "teacher_id",
            "teacher_ids",
            "teacher_weight",
            "teacher_weights",
            "teacher_cache_indices",
            "teacher_hidden_states",
            "_original_position_ids",
            "num_samples",
            "request_id",
            "batch_id",
            "_shifted",
        },
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
        self._validate_multi_adapter_lora_config()
        if self.train_config.get("load_weights_mode") == "skip" and not self.train_config.get("load_checkpoint_path"):
            raise ValueError(
                "load_weights_mode='skip' skips HF weight loading and requires train.load_checkpoint_path "
                "to materialize parameters from a DCP checkpoint."
            )

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
        self._accumulated_active_microbatches: Dict[str, int] = {}
        self._accumulated_active_voter_total: Dict[str, int] = {}

        # PP schedule cache: keyed by (n_microbatches, seq_len) to avoid rebuilding on every call.
        self._pp_schedule_cache: Dict[tuple, Any] = {}

        # OPD teacher resource caches are initialized lazily from loss_fn_params.
        self._opd_head_manager: Optional[TeacherHeadManager] = None
        self._opd_head_config: Optional[Any] = None
        self._opd_hidden_cache: Optional[TeacherActivationCache] = None
        self._opd_hidden_config: Optional[Any] = None
        # Multi-layer OPRD: separate rank-3 [layers, tokens, d] teacher cache.
        self._opd_layer_cache: Optional[TeacherActivationCache] = None
        self._opd_layer_config: Optional[Any] = None

        # Multi-adapter support (initialized later if LoRA is enabled)
        self._adapter_manager: Optional[LoRAAdapterManager] = None
        self._lora_session_specs: Dict[str, Dict[str, Any]] = {}
        self._default_lora_session_spec: Optional[Dict[str, Any]] = None
        self._checkpoint_mgr: Optional[CheckpointManager] = None

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
        seed = self.train_config.get("seed", 42)
        enable_full_determinism = self.train_config.get("enable_full_determinism", False)
        helper.set_seed(seed, False)

        # Disable TF32 and BF16 reduced-precision accumulation for
        # consistent numerics across parallelism strategies.
        helper.enable_high_precision_for_bf16()

        # Initialize distributed parallel state
        self._init_parallel_state()

        # Initialize model, optimizer, etc.
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_checkpointer()
        self._checkpoint_mgr = self._build_checkpoint_manager()
        self._load_initial_checkpoint()
        if enable_full_determinism:
            # Enabling deterministic algorithms before Kimi DCP/meta materialization
            # makes startup pathologically slow; training and adapter init happen below.
            helper.set_seed(seed, True)
        self._initialize_contexts()

        # Initialize multi-adapter manager if LoRA is enabled
        if self.lora_config.get("enable_lora", False):
            # Adapter manager now takes device instead of optimizer
            # Each adapter has its own optimizer instance
            device = torch.device(f"{get_device_type()}:{self.local_rank}")
            # Only rank 0 should save on eviction to avoid multi-rank file conflicts
            self._adapter_manager = LoRAAdapterManager(
                self.model,
                device,
                checkpoint_dir=self._get_adapter_checkpoint_dir(),
                auto_save_on_eviction=(self.rank == 0),
                lora_config=self.lora_config,
                optimizer_type=self.train_config.get("optimizer", "adamw"),
                optimizer_dtype=self.train_config.get("optimizer_dtype", "bf16"),
                optimizer_kwargs=self._get_optimizer_kwargs(),
                weight_decay=self.train_config.get("weight_decay", 0.01),
            )
            self._default_lora_session_spec = build_default_session_spec(
                base_model=self.model_config.get("model_name") or self.model_config.get("model_path"),
                train_config=self.train_config,
                lora_config=self.lora_config,
            )
            self.register_session(
                model_id="default",
                session_spec=self._default_lora_session_spec,
                materialize=True,
                initialize_fresh=False,
            )
            self._adapter_manager.current_adapter_id = "default"
            self._checkpoint_mgr._adapter_manager = self._adapter_manager
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

    @property
    def lora_session_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return the registered LoRA session specs."""
        return self._lora_session_specs

    def get_lora_session_spec(self, model_id: str) -> Dict[str, Any]:
        """Get the normalized LoRA session spec for a model_id."""
        if model_id not in self._lora_session_specs:
            raise KeyError(f"LoRA session spec not registered for model_id={model_id}")
        return deepcopy(self._lora_session_specs[model_id])

    def _sync_registered_lora_session_spec(self, model_id: str) -> None:
        """Refresh the worker session registry from the live adapter state."""
        if self._adapter_manager is None or not self._adapter_manager.has_adapter(model_id):
            return
        self._lora_session_specs[model_id] = self._adapter_manager.get_adapter_session_spec(model_id)

    def register_session(
        self,
        model_id: str,
        session_spec: Dict[str, Any],
        *,
        materialize: bool = False,
        initialize_fresh: bool = True,
    ) -> Dict[str, Any]:
        """Register a normalized session runtime spec on this worker."""
        if not self.lora_enabled:
            # Full-weight mode remains effectively single-tenant; keep the API
            # tolerant of create_model but don't install heterogeneous runtime state.
            self._validate_single_tenant(model_id)
            return {
                "model_id": model_id,
                "registered": True,
                "materialized": False,
                "message": "Full-weight mode ignores per-session LoRA runtime specs.",
            }

        existing_spec = self._lora_session_specs.get(model_id)
        if existing_spec is not None and existing_spec != session_spec:
            raise ValueError(
                f"Session '{model_id}' is already registered with a different runtime spec. "
                f"existing={existing_spec!r}, requested={session_spec!r}"
            )

        self._lora_session_specs[model_id] = deepcopy(session_spec)

        materialized = False
        if materialize and self._adapter_manager is not None and not self._adapter_manager.has_adapter(model_id):
            self._adapter_manager.register_adapter(
                model_id=model_id,
                session_spec=session_spec,
                initialize_fresh=initialize_fresh,
            )
            materialized = True

        return {
            "model_id": model_id,
            "registered": True,
            "materialized": materialized
            or (self._adapter_manager is not None and self._adapter_manager.has_adapter(model_id)),
            "session_spec": deepcopy(self._lora_session_specs[model_id]),
        }

    def ensure_lora_adapter(self, model_id: str, *, initialize_fresh: bool = True) -> None:
        """Materialize a registered LoRA session into the resident adapter registry."""
        if self._adapter_manager is None:
            raise RuntimeError("Cannot materialize LoRA adapter: adapter manager not initialized")
        session_spec = self.get_lora_session_spec(model_id)
        if not self._adapter_manager.has_adapter(model_id):
            self._adapter_manager.register_adapter(
                model_id=model_id,
                session_spec=session_spec,
                initialize_fresh=initialize_fresh,
            )

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
        Kill an active training session.

        Args:
            model_id: The session to kill.
            save_checkpoint: Whether to save a checkpoint before killing.

        Returns:
            Dictionary with success status and optional checkpoint path.

        Raises:
            ValueError: If model_id doesn't match the active session.
        """
        if self.lora_enabled:
            if model_id == "default":
                return {
                    "success": True,
                    "message": "Default LoRA session is reserved and was not removed.",
                    "checkpoint_path": None,
                }

            if model_id not in self._lora_session_specs and (
                self._adapter_manager is None or not self._adapter_manager.has_adapter(model_id)
            ):
                return {
                    "success": True,
                    "message": f"No active LoRA session '{model_id}' to kill.",
                    "checkpoint_path": None,
                }

            checkpoint_path = None
            if save_checkpoint and self._adapter_manager is not None:
                if self._adapter_manager.has_adapter(model_id):
                    checkpoint_path = os.path.join(
                        self.train_config.get("output_dir", "outputs"),
                        "weights",
                        model_id,
                        f"session_{model_id}_final",
                    )
                    self.save_state(checkpoint_path, save_optimizer=True, model_id=model_id)
                else:
                    evicted_path = os.path.join(self._get_adapter_checkpoint_dir(), "evicted", model_id)
                    if not os.path.exists(evicted_path):
                        raise FileNotFoundError(
                            f"Cannot kill LoRA session '{model_id}' with save_checkpoint=True: "
                            f"adapter is not resident and no evicted checkpoint exists at {evicted_path}."
                        )
                    checkpoint_path = self._promote_evicted_adapter_checkpoint(model_id, evicted_path)

            self._accumulated_valid_tokens.pop(model_id, None)
            if self._adapter_manager is not None and self._adapter_manager.has_adapter(model_id):
                self._adapter_manager.remove_adapter(model_id)
            self._lora_session_specs.pop(model_id, None)

            return {
                "success": True,
                "message": f"LoRA session '{model_id}' killed successfully.",
                "checkpoint_path": checkpoint_path,
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
            lm_head_tp_size=self.train_config.get("lm_head_tensor_parallel_size", 1),
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

        model_dtype = resolve_training_model_dtype(
            enable_lora=lora_enabled,
            enable_qlora=enable_qlora,
            enable_mixed_precision=enable_mixed_precision,
            skip_param_upcast=self.train_config.get("skip_param_upcast", False),
        )

        pp_size = self.train_config.get("pipeline_parallel_size", 1)
        pp_schedule_name = self.train_config.get("pipeline_parallel_schedule", "1F1B") if pp_size > 1 else None

        result = build_training_model(
            config_path=self.model_config.get("config_path"),
            weights_path=self.model_config.get("model_path"),
            torch_dtype=model_dtype,
            attn_implementation=self.model_config.get("attn_implementation", "sdpa"),
            moe_implementation=self.model_config.get("moe_implementation"),
            ep_dispatch=self.model_config.get("ep_dispatch", "alltoall"),
            train_router=self.model_config.get("train_router", False),
            record_routing_weights=self.model_config.get("record_routing_weights", True),
            deepep_buffer_size_gb=self.model_config.get("deepep_buffer_size_gb", 2.0),
            deepep_num_sms=self.model_config.get("deepep_num_sms", 20),
            deepep_async_combine=self.model_config.get("deepep_async_combine", False),
            alltoall_combine_hidden_chunk_size=self.model_config.get("alltoall_combine_hidden_chunk_size", 0),
            init_device=self.train_config.get("init_device", "cpu"),
            merge_qkv=self.model_config.get("merge_qkv", True),
            enable_lora=lora_enabled,
            lora_rank=self.lora_config.get("max_lora_rank", self.lora_config.get("lora_rank", 32)),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            lora_target_modules=target_modules,
            moe_hybrid_shared_lora=self.lora_config.get("moe_hybrid_shared_lora", False),
            enable_qlora=enable_qlora,
            quant_format=self.lora_config.get("quant_format", "nvfp4"),
            quant_group_size=self.lora_config.get("quant_group_size", 16),
            qlora_exclude_modules=self.lora_config.get("exclude_modules"),
            enable_full_shard=self.train_config.get("enable_full_shard", True),
            enable_mixed_precision=enable_mixed_precision,
            fsdp_reduce_dtype=self.train_config.get("fsdp_reduce_dtype", "fp32"),
            skip_param_upcast=self.train_config.get("skip_param_upcast", False),
            enable_fp8_training=self.train_config.get("enable_fp8_training", False),
            enable_qarl=self.train_config.get("enable_qarl", False),
            qarl_quant_cfg=self.train_config.get("qarl_quant_cfg"),
            qarl_calib_data=self.train_config.get("qarl_calib_data"),
            qarl_calib_size=self.train_config.get("qarl_calib_size", 0),
            qarl_quant_sequence_length=self.train_config.get("qarl_quant_sequence_length"),
            qarl_sync_format=self.train_config.get("qarl_sync_format", "fp8"),
            qarl_target_modules=self.train_config.get("qarl_target_modules"),
            qarl_exclude_modules=self.train_config.get("qarl_exclude_modules"),
            fp8_training_num_first_layers_bf16=self.train_config.get("fp8_training_num_first_layers_bf16", 0),
            fp8_training_num_last_layers_bf16=self.train_config.get("fp8_training_num_last_layers_bf16", 0),
            fp8_training_allow_blackwell=self.train_config.get("fp8_training_allow_blackwell", False),
            fp8_training_blackwell_validation_artifact=self.train_config.get(
                "fp8_training_blackwell_validation_artifact"
            ),
            fp8_training_block_size=self.train_config.get("fp8_training_block_size", 128),
            fp8_training_backward=self.train_config.get("fp8_training_backward", "fp8"),
            fp8_training_smoothquant_alpha=self.train_config.get("fp8_training_smoothquant_alpha"),
            fp8_training_lm_head_smoothquant_alpha=self.train_config.get("fp8_training_lm_head_smoothquant_alpha"),
            fp8_training_activation_amax_scale=self.train_config.get("fp8_training_activation_amax_scale", 1.0),
            fp8_training_weight_amax_scale=self.train_config.get("fp8_training_weight_amax_scale", 1.0),
            fp8_training_correction_mode=self.train_config.get("fp8_training_correction_mode", "none"),
            fp8_training_module_overrides=self.train_config.get("fp8_training_module_overrides"),
            fp8_training_moe_grouped_backend=self.train_config.get(
                "fp8_training_moe_grouped_backend",
                "triton_grouped",
            ),
            fp8_training_target_modules=self.train_config.get("fp8_training_target_modules"),
            fp8_training_exclude_modules=self.train_config.get("fp8_training_exclude_modules"),
            fp8_training_allow_bf16_fallback=self.train_config.get("fp8_training_allow_bf16_fallback", False),
            enable_gradient_checkpointing=self.train_config.get("enable_gradient_checkpointing", False),
            gradient_checkpointing_method=self.train_config.get("gradient_checkpointing_method"),
            enable_compile=self.train_config.get("enable_compile", False),
            compile_dynamic_shapes=self.train_config.get("compile_dynamic_shapes", False),
            basic_modules=self.model_config.get("basic_modules", []),
            enable_reentrant=self.train_config.get("enable_reentrant", False),
            enable_forward_prefetch=self.train_config.get("enable_forward_prefetch", True),
            load_weights_mode=self.train_config.get("load_weights_mode", "grouped"),
            reshard_after_forward=self.train_config.get("reshard_after_forward"),
            moe_grad_reduce_mode=self.train_config.get("moe_grad_reduce_mode", "reduce_scatter"),
            pp_schedule=pp_schedule_name,
            freeze_router=self.train_config.get("freeze_router", False),
            router_fp32=self.model_config.get("router_fp32", True),
            lm_head_fp32=self.model_config.get("lm_head_fp32", True),
            rmsnorm_mode=self.model_config.get("rmsnorm_mode", "native"),
            activation_native=self.model_config.get("activation_native", False),
            rope_native=self.model_config.get("rope_native", False),
            attention_cast_bf16=self.model_config.get("attention_cast_bf16", False),
            sparse_mla_enabled=self.model_config.get("sparse_mla_enabled", False),
            sparse_mla_backend=self.model_config.get("sparse_mla_backend", "auto"),
            flash_attention_deterministic=self.model_config.get("flash_attention_deterministic", False),
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

        config_path = self.model_config.get("config_path") or self.model_config.get("model_path")
        model_type = None
        if config_path:
            try:
                config_dict, _ = PretrainedConfig.get_config_dict(config_path)
                model_type = config_dict.get("model_type")
            except Exception:
                model_type = None

        # Legacy Tinker-style: build from boolean flags
        train_attn = self.lora_config.get("train_attn", True)
        train_mlp = self.lora_config.get("train_mlp", True)
        train_unembed = self.lora_config.get("train_unembed", True)

        if model_type in {"deepseek_v3", "kimi_k2", "kimi_k25"}:
            target_modules = deepseek_v3_default_lora_targets(
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
        elif model_type in {"glm_moe_dsa", "xorl_glm5"}:
            target_modules = glm5_default_lora_targets(
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
        else:
            target_modules = []
            if train_attn:
                target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
            if train_mlp:
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])
            if train_unembed:
                target_modules.append("lm_head")
        if not target_modules:
            raise ValueError("At least one of train_mlp, train_attn, or train_unembed must be True")
        return target_modules

    def _validate_multi_adapter_lora_config(self) -> None:
        """Reject LoRA features that are not supported by the multi-adapter server path."""
        if self.lora_config.get("enable_lora", False) and self.lora_config.get("merge_lora_interval", 0) > 0:
            raise ValueError("merge_lora_interval is not supported with multi-adapter LoRA server training")
        if self.lora_config.get("enable_lora", False) and self.train_config.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "pipeline_parallel_size > 1 is not supported with multi-adapter LoRA server training. "
                "Adapter coordination currently assumes identical local LoRA layouts on every rank."
            )
        max_lora_rank = self.lora_config.get("max_lora_rank", self.lora_config.get("lora_rank", 32))
        default_rank = self.lora_config.get("lora_rank", 32)
        if max_lora_rank < default_rank:
            raise ValueError(
                f"max_lora_rank ({max_lora_rank}) must be >= lora_rank ({default_rank}) for multi-adapter LoRA"
            )

    def _get_optimizer_kwargs(self) -> Dict[str, Any]:
        """Collect optimizer kwargs from the server train config."""
        explicit_kwargs = self.train_config.get("optimizer_kwargs")
        if explicit_kwargs is not None:
            return deepcopy(explicit_kwargs)

        optimizer_type = self.train_config.get("optimizer", "adamw")
        kwargs: Dict[str, Any] = {}
        if optimizer_type == "muon":
            for key in (
                "muon_lr",
                "muon_momentum",
                "muon_nesterov",
                "muon_ns_steps",
                "muon_adjust_lr_fn",
                "muon_ns_algorithm",
                "muon_ns_use_quack_kernels",
                "muon_gram_ns_num_restarts",
                "muon_gram_ns_restart_iterations",
            ):
                if key in self.train_config:
                    kwargs[key] = self.train_config[key]

            if self.train_config.get("optimizer_dtype") == "bf16":
                kwargs["muon_momentum_dtype"] = torch.bfloat16

            grad_dtype = self.train_config.get("muon_grad_dtype")
            if grad_dtype == "bf16":
                kwargs["muon_grad_dtype"] = torch.bfloat16
            elif grad_dtype == "fp32":
                kwargs["muon_grad_dtype"] = torch.float32

            update_dtype = self.train_config.get("muon_update_dtype")
            if update_dtype == "bf16":
                kwargs["muon_update_dtype"] = torch.bfloat16
            elif update_dtype == "fp32":
                kwargs["muon_update_dtype"] = torch.float32

            if self.train_config.get("muon_force_momentum_path", False):
                kwargs["muon_force_momentum_path"] = True

        return kwargs

    def _get_adapter_checkpoint_dir(self) -> str:
        """Return the shared adapter checkpoint directory under the server output dir."""
        return os.path.join(self.train_config.get("output_dir", "outputs"), "adapters")

    def _promote_evicted_adapter_checkpoint(self, model_id: str, evicted_path: str) -> str:
        """Copy an evicted adapter checkpoint into the public weights namespace."""
        checkpoint_path = os.path.join(
            self.train_config.get("output_dir", "outputs"),
            "weights",
            model_id,
            f"session_{model_id}_final",
        )
        if self.rank == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            shutil.copytree(evicted_path, checkpoint_path)
        return checkpoint_path

    def _initialize_optimizer(self):
        """Initialize the optimizer."""
        optimizer_type = self.train_config.get("optimizer", "adamw")
        self._use_distsignsgd = optimizer_type == "distsignsgd"
        if self._use_distsignsgd and self.lora_config.get("enable_lora", False):
            raise NotImplementedError("DistSignSGD does not yet support server LoRA adapter-manager training.")
        optimizer_kwargs = self._get_optimizer_kwargs()
        self.optimizer = build_optimizer(
            self.model,
            lr=self.train_config.get("lr", 1e-5),
            weight_decay=self.train_config.get("weight_decay", 0.01),
            fused=True,
            optimizer_type=optimizer_type,
            optimizer_dtype=self.train_config.get("optimizer_dtype", "bf16"),
            optimizer_kwargs=optimizer_kwargs or None,
            cautious_weight_decay=self.train_config.get("cautious_weight_decay", False),
        )

        # Muon runs carry a two-tier lr: matrix groups at muon_lr, fallback groups
        # at the base lr. A client-passed optim_step lr is the new BASE — the muon
        # groups must keep the configured muon_lr/base-lr ratio. Stomping every
        # group with the client value (the old behavior) silently ran the muon
        # matrices at the adamw lr (~10x low): ARITH-012's 0.000 vs the adamw
        # twin's 0.218.
        self._muon_client_lr_scale: Optional[float] = None
        if optimizer_type == "muon":
            base_lr = float(self.train_config.get("lr", 1e-5))
            muon_lr = float((optimizer_kwargs or {}).get("muon_lr", 0.02))
            if base_lr > 0:
                self._muon_client_lr_scale = muon_lr / base_lr
                logger.info(
                    f"Muon client-lr scale: muon groups follow client lr x {self._muon_client_lr_scale:.3g} "
                    f"(configured muon_lr={muon_lr:g} / base lr={base_lr:g})"
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

    def _build_checkpoint_manager(self, adapter_manager=None) -> CheckpointManager:
        return CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            checkpointer=self.Checkpointer,
            lora_config=self.lora_config,
            model_config=self.model_config,
            train_config=self.train_config,
            rank=self.rank,
            local_rank=self.local_rank,
            adapter_manager=adapter_manager,
        )

    def _load_initial_checkpoint(self) -> None:
        checkpoint_path = self.train_config.get("load_checkpoint_path")
        if not checkpoint_path:
            return
        if self._checkpoint_mgr is None:
            raise RuntimeError("Checkpoint manager must be initialized before loading an initial checkpoint.")

        load_optimizer = bool(self.train_config.get("load_optimizer", True))
        logger.info("Loading initial checkpoint from %s (load_optimizer=%s)", checkpoint_path, load_optimizer)
        self._checkpoint_mgr.load_state(checkpoint_path, load_optimizer=load_optimizer)
        self._sync_from_checkpoint_state()

    def register_lora_adapter(self, model_id: str, lr: Optional[float]) -> Dict[str, Any]:
        """
        Materialize a registered LoRA session into the adapter manager.

        Args:
            model_id: Unique identifier for this training run
            lr: Optional learning rate override used for legacy call sites.

        Returns:
            Dictionary with registration info

        Raises:
            RuntimeError: If LoRA is not enabled or adapter manager not initialized
        """
        if self._adapter_manager is None:
            raise RuntimeError("Cannot register adapter: LoRA is not enabled or adapter manager not initialized")

        if model_id not in self._lora_session_specs:
            if model_id == "default" and self._default_lora_session_spec is not None:
                self._lora_session_specs["default"] = deepcopy(self._default_lora_session_spec)
            else:
                raise KeyError(f"LoRA session '{model_id}' is not registered. Call create_model first.")

        if not self._adapter_manager.has_adapter(model_id):
            session_spec = self.get_lora_session_spec(model_id)
            if lr is not None:
                session_spec["optimizer_config"]["learning_rate"] = lr
            self._adapter_manager.register_adapter(
                model_id=model_id,
                session_spec=session_spec,
                initialize_fresh=True,
            )
        self._sync_registered_lora_session_spec(model_id)

        return {
            "model_id": model_id,
            "lr": lr,
            "registered": True,
            "total_adapters": len(self._adapter_manager.list_adapters()),
        }

    def register_adapter(self, model_id: str, lr: Optional[float]) -> Dict[str, Any]:
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

    @staticmethod
    def _get_fp8_lm_head_module(lm_head):
        try:
            from xorl.fp8_training import FP8Linear  # noqa: PLC0415
        except ImportError:  # pragma: no cover - optional in lightweight import contexts.
            return None
        return lm_head if isinstance(lm_head, FP8Linear) else None

    def _collect_per_token_outputs(self, per_token_tensors, micro_batch, accumulators):
        """Gather per-token outputs across the unified SP group and append to accumulators."""
        ps = get_parallel_state()
        token_diagnostics = per_token_tensors.get("token_diagnostics")
        per_token_tensors = {key: value for key, value in per_token_tensors.items() if isinstance(value, torch.Tensor)}
        if not per_token_tensors:
            return

        if ps.cp_enabled:
            sp_group = ps.sp_group
            cp_size = ps.cp_size

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
                    group=sp_group,
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
        if token_diagnostics is not None:
            if ps.cp_enabled:
                logger.warning("token_diagnostics currently skips CP/SP-gathered batches")
            else:
                accumulators["token_diagnostics"].append(token_diagnostics)

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = ModelRunner._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _parse_diagnostic_layer_indices(value: Any) -> list[int]:
        if value is None or value == "":
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            indices: list[int] = []
            for part in value.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    start = int(start_s)
                    end = int(end_s)
                    step = 1 if end >= start else -1
                    indices.extend(range(start, end + step, step))
                else:
                    indices.append(int(part))
            return sorted(set(indices))
        return sorted({int(item) for item in value})

    @staticmethod
    def _parse_diagnostic_sample_indices(value: Any, hidden_dim: int) -> list[int] | None:
        if value is None or value == "":
            return None
        if isinstance(value, str) and value.strip().lower() == "all":
            return list(range(hidden_dim))

        indices = ModelRunner._parse_diagnostic_layer_indices(value)
        invalid = [idx for idx in indices if idx < 0 or idx >= hidden_dim]
        if invalid:
            raise ValueError(
                f"diagnostic_hidden_sample_indices contains out-of-range hidden dims {invalid}; "
                f"valid range is [0, {hidden_dim})"
            )
        return indices

    @staticmethod
    def _build_diagnostic_sample_indices(
        *,
        hidden_dim: int,
        hidden_sample_count: int,
        hidden_sample_indices: Any = None,
        device: torch.device,
    ) -> torch.Tensor:
        explicit_indices = ModelRunner._parse_diagnostic_sample_indices(hidden_sample_indices, hidden_dim)
        if explicit_indices is not None:
            return torch.tensor(explicit_indices, device=device, dtype=torch.long)

        sample_count = max(0, min(int(hidden_sample_count), hidden_dim))
        if sample_count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        return (
            torch.linspace(
                0,
                hidden_dim - 1,
                steps=sample_count,
                device=device,
                dtype=torch.float32,
            )
            .round()
            .to(torch.long)
        )

    def _install_hidden_component_hooks(self, layer_indices: list[int]) -> tuple[list[dict[str, Any]], list[Any]]:
        class _AttributeRestoreHandle:
            _missing = object()

            def __init__(self, obj: Any, name: str):
                self.obj = obj
                self.name = name
                self.value = getattr(obj, "__dict__", {}).get(name, self._missing)

            def remove(self) -> None:
                if self.value is self._missing:
                    if self.name in getattr(self.obj, "__dict__", {}):
                        delattr(self.obj, self.name)
                else:
                    setattr(self.obj, self.name, self.value)

        captures: list[dict[str, Any]] = []
        handles = []
        latest_shared_weighted: dict[int, torch.Tensor] = {}
        latest_shared_gate_logits: dict[int, torch.Tensor] = {}
        model = getattr(self.model, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None:
            logger.warning("diagnostic_hidden_components requested but model has no model.layers")
            return captures, handles

        component_order = {
            "layer_input": 0,
            "input_norm": 1,
            "attention": 2,
            "post_attention_norm": 3,
            "post_attention_residual": 4,
            "experts": 5,
            "shared_expert_input": 6,
            "shared_expert_gate_value": 7,
            "shared_expert": 8,
            "shared_expert_weighted": 9,
            "mlp": 10,
            "layer_output": 11,
        }

        def capture(layer_idx: int, name: str, value: Any) -> None:
            tensor = self._first_tensor(value)
            if tensor is not None:
                captures.append(
                    {
                        "layer": layer_idx,
                        "name": name,
                        "order": component_order[name],
                        "tensor": tensor.detach(),
                    }
                )

        def capture_mlp_output(layer_idx: int, value: Any) -> None:
            tensor = self._first_tensor(value)
            weighted_shared = latest_shared_weighted.get(layer_idx)
            if tensor is not None and weighted_shared is not None and tensor.shape == weighted_shared.shape:
                capture(layer_idx, "experts", tensor - weighted_shared)
            capture(layer_idx, "mlp", value)

        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(layers):
                logger.warning("Skipping diagnostic_hidden_components layer %s outside [0, %s)", layer_idx, len(layers))
                continue
            layer = layers[layer_idx]
            if layer is None:
                continue

            handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, args, layer_idx=layer_idx: capture(
                        layer_idx, "layer_input", args[0] if args else None
                    )
                )
            )
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, layer_idx=layer_idx: capture(layer_idx, "layer_output", output)
                )
            )

            input_norm = getattr(layer, "input_layernorm", None)
            if input_norm is not None:
                handles.append(
                    input_norm.register_forward_hook(
                        lambda _module, _args, output, layer_idx=layer_idx: capture(layer_idx, "input_norm", output)
                    )
                )

            attention = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn", None)
            if attention is not None:
                handles.append(
                    attention.register_forward_hook(
                        lambda _module, _args, output, layer_idx=layer_idx: capture(layer_idx, "attention", output)
                    )
                )

            post_attention_norm = getattr(layer, "post_attention_layernorm", None)
            if post_attention_norm is not None:

                def post_attention_hook(_module, _args, output, layer_idx=layer_idx):
                    capture(layer_idx, "post_attention_norm", output)
                    if isinstance(output, (tuple, list)) and len(output) > 1:
                        capture(layer_idx, "post_attention_residual", output[1])

                handles.append(post_attention_norm.register_forward_hook(post_attention_hook))

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                handles.append(
                    mlp.register_forward_hook(
                        lambda _module, _args, output, layer_idx=layer_idx: capture_mlp_output(layer_idx, output)
                    )
                )
                shared_expert = getattr(mlp, "shared_expert", None)
                if shared_expert is not None:
                    handles.append(
                        shared_expert.register_forward_hook(
                            lambda _module, _args, output, layer_idx=layer_idx: capture(
                                layer_idx, "shared_expert", output
                            )
                        )
                    )

                shared_expert_gate = getattr(mlp, "shared_expert_gate", None)
                if shared_expert_gate is not None:

                    def shared_expert_gate_hook(_module, _args, output, layer_idx=layer_idx):
                        tensor = self._first_tensor(output)
                        if tensor is not None:
                            latest_shared_gate_logits[layer_idx] = tensor.detach()

                    handles.append(shared_expert_gate.register_forward_hook(shared_expert_gate_hook))

                original_shared_expert = getattr(mlp, "_shared_expert", None)
                if callable(original_shared_expert):

                    def wrapped_shared_expert(
                        hidden_states,
                        *args,
                        layer_idx=layer_idx,
                        original_shared_expert=original_shared_expert,
                        **kwargs,
                    ):
                        capture(layer_idx, "shared_expert_input", hidden_states)
                        output = original_shared_expert(hidden_states, *args, **kwargs)
                        tensor = self._first_tensor(output)
                        if tensor is not None:
                            gate_logits = latest_shared_gate_logits.get(layer_idx)
                            if gate_logits is not None and gate_logits.shape[-1] == 1:
                                gate_shape = (*tensor.shape[:-1], 1)
                                if gate_logits.numel() == math.prod(gate_shape):
                                    gate_value = torch.sigmoid(
                                        gate_logits.reshape(gate_shape).to(dtype=tensor.dtype)
                                    ).expand_as(tensor)
                                    capture(layer_idx, "shared_expert_gate_value", gate_value)
                            latest_shared_weighted[layer_idx] = tensor.detach()
                            capture(layer_idx, "shared_expert_weighted", tensor)
                        return output

                    restore_handle = _AttributeRestoreHandle(mlp, "_shared_expert")
                    setattr(mlp, "_shared_expert", wrapped_shared_expert)
                    handles.append(restore_handle)

        return captures, handles

    def _install_opd_selected_layer_hooks(
        self,
        layer_indices: list[int],
        valid_mask: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], list[Any]]:
        """Capture only OPRD-supervised student layer rows.

        ``output_hidden_states=True`` returns full [batch, seq, d] residual streams
        for every requested layer, then OPRD keeps only response-valid rows. That
        retention is exactly what blocks the 8-GPU/prep64 fit probe. These hooks
        capture the same post-decoder-layer tensors but immediately index them down
        to the flattened valid positions, preserving the autograd path for the
        supervised rows while avoiding full-sequence layer outputs.
        """
        captures: dict[int, torch.Tensor] = {}
        handles: list[Any] = []
        model = getattr(self.model, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None:
            logger.warning("OPRD selected-layer capture requested but model has no model.layers")
            return captures, handles

        valid_flat = valid_mask.reshape(-1)
        valid_indices = valid_flat.nonzero(as_tuple=True)[0].to(dtype=torch.long)

        def capture(layer_idx: int, output: Any) -> None:
            tensor = self._first_tensor(output)
            if tensor is None:
                return
            flat = tensor.reshape(-1, tensor.shape[-1])
            captures[layer_idx] = flat.index_select(0, valid_indices.to(device=flat.device))

        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(layers):
                logger.warning("Skipping OPRD selected-layer capture for layer %s outside [0, %s)", layer_idx, len(layers))
                continue
            layer = layers[layer_idx]
            if layer is None:
                continue
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, layer_idx=layer_idx: capture(layer_idx, output)
                )
            )

        return captures, handles

    @staticmethod
    def _write_hidden_component_tensor_dump(
        path: str | os.PathLike[str],
        hidden_components: list[dict[str, Any]],
        *,
        labels: torch.Tensor | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        output_path = Path(f"{path}.rank{rank}.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "__metadata__": {
                "rank": rank,
                "component_count": len(hidden_components),
                **(metadata or {}),
            }
        }
        if labels is not None:
            payload["labels"] = labels.detach().cpu()

        ordered_components = sorted(
            hidden_components,
            key=lambda item: (int(item["layer"]), int(item.get("order", 0)), str(item["name"])),
        )
        for component in ordered_components:
            tensor = component.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                continue
            key = f"model.layers.{int(component['layer'])}.{str(component['name'])}"
            payload[key] = tensor.detach().cpu()

        torch.save(payload, output_path)
        return str(output_path)

    @staticmethod
    def _compute_token_diagnostics(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor | None,
        topk: int,
        lm_head: torch.nn.Module | None = None,
        lm_head_fp32: bool = False,
        per_token_logprobs: torch.Tensor | None = None,
        include_weight_reference: bool = False,
        all_hidden_states: tuple[torch.Tensor, ...] | None = None,
        hidden_components: list[dict[str, Any]] | None = None,
        hidden_sample_count: int = 8,
        hidden_sample_indices: Any = None,
    ) -> dict[str, Any] | None:
        """Return target ranks/top-k logprobs for valid labels in one micro-batch.

        Materializes the full fp32 logits tensor for every valid token (vocab_size
        floats per token). For vocab ~150k and N valid tokens this peaks at ~600 KB·N;
        opt-in via diagnostic_topk > 0 only. When requested, also reports whether
        the loss-path per-token logprob matches this explicit full-vocab
        log-softmax and, for FP8 lm_head module diagnostics, an fp32 raw-weight
        reference for the same target tokens. ``all_hidden_states`` is a heavier
        optional path for K3 repros; it summarizes the selected target rows at
        each layer without returning full hidden vectors.
        """
        if labels is None or topk <= 0:
            return None

        labels_flat = labels.reshape(-1)
        valid = labels_flat != IGNORE_INDEX
        if not valid.any():
            return {
                "valid_positions": [],
                "target_ids": [],
                "target_logprobs": [],
                "target_ranks": [],
                "topk_ids": [],
                "topk_logprobs": [],
            }

        with torch.no_grad():
            hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            valid_indices = valid.nonzero(as_tuple=True)[0]
            valid_hidden = hidden_flat[valid_indices]
            # lm_head_fp32 takes precedence over the FP8 lm_head module: an FP32
            # lm_head must not be FP8-quantized, so compute logits in FP32 from
            # the master weight rather than calling FP8Linear.forward (whose
            # FP8 matmul would otherwise still run, with .float() applied too
            # late). This mirrors compute_per_token_ce.
            if lm_head is not None and not lm_head_fp32:
                logits = lm_head(valid_hidden).float()
            elif lm_head_fp32:
                logits = (valid_hidden.float() @ weight.float().t()).float()
            else:
                logits = (valid_hidden @ weight.t()).float()
            valid_log_probs = F.log_softmax(logits, dim=-1)
            target_ids = labels_flat[valid_indices].to(device=valid_log_probs.device, dtype=torch.long)
            topk = min(int(topk), valid_log_probs.shape[-1])
            topk_vals, topk_ids = valid_log_probs.topk(topk, dim=-1)
            row_indices = torch.arange(target_ids.shape[0], device=valid_log_probs.device)
            target_logprobs = valid_log_probs[row_indices, target_ids]
            target_ranks = (valid_log_probs > target_logprobs.unsqueeze(-1)).sum(dim=-1) + 1

            diagnostics = {
                "valid_positions": valid_indices.cpu().tolist(),
                "target_ids": target_ids.cpu().tolist(),
                "target_logprobs": target_logprobs.cpu().tolist(),
                "target_ranks": target_ranks.cpu().tolist(),
                "topk_ids": topk_ids.cpu().tolist(),
                "topk_logprobs": topk_vals.cpu().tolist(),
            }

            if per_token_logprobs is not None:
                loss_logprobs = per_token_logprobs.reshape(-1)[valid_indices].to(
                    device=target_logprobs.device,
                    dtype=target_logprobs.dtype,
                )
                loss_delta = loss_logprobs - target_logprobs
                diagnostics.update(
                    {
                        "loss_logprobs": loss_logprobs.cpu().tolist(),
                        "loss_logprob_deltas": loss_delta.cpu().tolist(),
                        "loss_logprob_max_abs_delta": float(loss_delta.abs().max().item()),
                    }
                )

            if include_weight_reference:
                reference_logits = (valid_hidden.float() @ weight.float().t()).float()
                reference_log_probs = F.log_softmax(reference_logits, dim=-1)
                reference_target_logprobs = reference_log_probs[row_indices, target_ids]
                reference_target_ranks = (reference_log_probs > reference_target_logprobs.unsqueeze(-1)).sum(dim=-1) + 1
                reference_delta = target_logprobs - reference_target_logprobs
                diagnostics.update(
                    {
                        "reference_target_logprobs": reference_target_logprobs.cpu().tolist(),
                        "reference_target_ranks": reference_target_ranks.cpu().tolist(),
                        "reference_logprob_deltas": reference_delta.cpu().tolist(),
                        "reference_logprob_max_abs_delta": float(reference_delta.abs().max().item()),
                    }
                )

            if all_hidden_states is not None or hidden_components:
                sample_indices = ModelRunner._build_diagnostic_sample_indices(
                    hidden_dim=hidden_flat.shape[-1],
                    hidden_sample_count=hidden_sample_count,
                    hidden_sample_indices=hidden_sample_indices,
                    device=hidden_flat.device,
                )

            if all_hidden_states is not None:
                per_token_summaries = [
                    {
                        "layer_count": len(all_hidden_states),
                        "sample_indices": sample_indices.cpu().tolist(),
                        "layers": [],
                    }
                    for _ in range(valid_indices.shape[0])
                ]
                for layer_index, layer_hidden in enumerate(all_hidden_states):
                    layer_flat = layer_hidden.reshape(-1, layer_hidden.shape[-1])
                    rows = layer_flat[valid_indices].float()
                    row_mean = rows.mean(dim=-1)
                    row_std = rows.std(dim=-1, unbiased=False)
                    row_rms = torch.sqrt(torch.mean(rows * rows, dim=-1))
                    row_max_abs = rows.abs().amax(dim=-1)
                    row_min = rows.amin(dim=-1)
                    row_max = rows.amax(dim=-1)
                    sampled_values = (
                        rows[:, sample_indices] if sample_indices.numel() > 0 else rows.new_empty((rows.shape[0], 0))
                    )
                    for token_index, summary in enumerate(per_token_summaries):
                        summary["layers"].append(
                            {
                                "index": layer_index,
                                "mean": float(row_mean[token_index].item()),
                                "std": float(row_std[token_index].item()),
                                "rms": float(row_rms[token_index].item()),
                                "max_abs": float(row_max_abs[token_index].item()),
                                "min": float(row_min[token_index].item()),
                                "max": float(row_max[token_index].item()),
                                "sample_values": sampled_values[token_index].cpu().tolist(),
                            }
                        )
                diagnostics["hidden_state_summaries"] = per_token_summaries

            if hidden_components:
                per_token_component_summaries = [
                    {
                        "component_count": len(hidden_components),
                        "sample_indices": sample_indices.cpu().tolist(),
                        "components": [],
                    }
                    for _ in range(valid_indices.shape[0])
                ]
                ordered_components = sorted(
                    hidden_components,
                    key=lambda item: (int(item["layer"]), int(item.get("order", 0)), str(item["name"])),
                )
                for component in ordered_components:
                    component_hidden = component["tensor"]
                    component_flat = component_hidden.reshape(-1, component_hidden.shape[-1])
                    rows = component_flat[valid_indices].float()
                    row_mean = rows.mean(dim=-1)
                    row_std = rows.std(dim=-1, unbiased=False)
                    row_rms = torch.sqrt(torch.mean(rows * rows, dim=-1))
                    row_max_abs = rows.abs().amax(dim=-1)
                    row_min = rows.amin(dim=-1)
                    row_max = rows.amax(dim=-1)
                    sampled_values = (
                        rows[:, sample_indices] if sample_indices.numel() > 0 else rows.new_empty((rows.shape[0], 0))
                    )
                    for token_index, summary in enumerate(per_token_component_summaries):
                        summary["components"].append(
                            {
                                "layer": int(component["layer"]),
                                "name": str(component["name"]),
                                "mean": float(row_mean[token_index].item()),
                                "std": float(row_std[token_index].item()),
                                "rms": float(row_rms[token_index].item()),
                                "max_abs": float(row_max_abs[token_index].item()),
                                "min": float(row_min[token_index].item()),
                                "max": float(row_max[token_index].item()),
                                "sample_values": sampled_values[token_index].cpu().tolist(),
                            }
                        )
                diagnostics["hidden_component_summaries"] = per_token_component_summaries

        return diagnostics

    @staticmethod
    def _teacher_cache_dtype(dtype_name: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        key = str(dtype_name).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported teacher_hidden_cache_dtype={dtype_name!r}")
        return mapping[key]

    @staticmethod
    def _position_spans(position_ids: torch.Tensor, num_samples: int) -> List[tuple[int, int]]:
        pos = position_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        if pos.numel() == 0:
            return []
        starts = [0]
        for i in range(1, pos.numel()):
            if pos[i].item() <= pos[i - 1].item():
                starts.append(i)
        starts.append(pos.numel())
        return [(starts[i], starts[i + 1]) for i in range(min(num_samples, len(starts) - 1))]

    @staticmethod
    def _valid_row_length(labels: Optional[torch.Tensor], row: int, fallback: int) -> int:
        if labels is None or labels.numel() == 0:
            return fallback
        row_labels = labels[row].reshape(-1)
        valid = row_labels != IGNORE_INDEX
        if valid.any():
            return int(valid.nonzero(as_tuple=True)[0][-1].item()) + 1
        return 0

    @staticmethod
    def _split_hidden_cache_rows(
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
    ) -> tuple[List[torch.Tensor], List[List[int]]]:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected teacher hidden states [batch, seq, hidden], got {tuple(hidden_states.shape)}")

        chunks: List[torch.Tensor] = []
        cache_indices_by_sample: List[List[int]] = []
        next_index = 0

        def add_chunk(chunk: torch.Tensor) -> None:
            nonlocal next_index
            rows = int(chunk.shape[0])
            if rows <= 0:
                return
            chunks.append(chunk.contiguous())
            cache_indices_by_sample.append(list(range(next_index, next_index + rows)))
            next_index += rows

        if position_ids is not None:
            spans = ModelRunner._position_spans(
                position_ids,
                int(num_samples) if num_samples is not None and int(num_samples) > 0 else position_ids.numel(),
            )
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            flat_labels = labels.reshape(-1) if isinstance(labels, torch.Tensor) else None
            for start, end in spans:
                if end <= start:
                    continue
                span_hidden = flat_hidden[start:end]
                if flat_labels is not None:
                    span_labels = flat_labels[start:end]
                    usable_len = min(span_hidden.shape[0], span_labels.numel())
                    valid = span_labels[:usable_len] != IGNORE_INDEX
                    if valid.any():
                        add_chunk(span_hidden[:usable_len][valid])
                    continue
                add_chunk(span_hidden)
            return chunks, cache_indices_by_sample

        batch_size = hidden_states.shape[0]
        for row in range(batch_size):
            row_hidden = hidden_states[row]
            if isinstance(labels, torch.Tensor):
                row_labels = labels[row].reshape(-1)
                usable_len = min(row_hidden.shape[0], row_labels.numel())
                valid = row_labels[:usable_len] != IGNORE_INDEX
                if valid.any():
                    add_chunk(row_hidden[:usable_len][valid])
            elif row_hidden.shape[0] > 0:
                add_chunk(row_hidden)
        return chunks, cache_indices_by_sample

    @staticmethod
    def _teacher_cache_label_key(micro_batch: Dict[str, Any]) -> Optional[str]:
        if isinstance(micro_batch.get("labels"), torch.Tensor):
            return "labels"
        if isinstance(micro_batch.get("target_tokens"), torch.Tensor):
            return "target_tokens"
        return None

    def _gather_teacher_cache_sequences(
        self,
        hidden_states: torch.Tensor,
        micro_batch: Dict[str, Any],
        ps,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        if not getattr(ps, "cp_enabled", False):
            return hidden_states, micro_batch

        original_position_ids = micro_batch.get("_original_position_ids")
        if isinstance(original_position_ids, torch.Tensor):
            original_seq_len = original_position_ids.shape[-1]
        else:
            original_seq_len = hidden_states.shape[1] * max(int(getattr(ps, "cp_size", 1)), 1)

        sequence_group = getattr(ps, "sp_group", getattr(ps, "ulysses_group", None))
        hidden_states = gather_outputs(
            hidden_states,
            gather_dim=1,
            padding_dim=1,
            unpad_dim_size=original_seq_len,
            scale_grad=False,
            group=sequence_group,
        )

        label_key = self._teacher_cache_label_key(micro_batch)
        if label_key is None:
            return hidden_states, micro_batch

        labels = micro_batch[label_key]
        if labels.shape[-1] == hidden_states.shape[1]:
            return hidden_states, micro_batch

        full_labels = gather_outputs(
            labels,
            gather_dim=-1,
            padding_dim=-1,
            unpad_dim_size=original_seq_len,
            scale_grad=False,
            group=sequence_group,
        )
        micro_batch = dict(micro_batch)
        micro_batch[label_key] = full_labels
        return hidden_states, micro_batch

    def _teacher_cache_write_cp_rank(self, ps, write_rank: int) -> int:
        if not getattr(ps, "cp_enabled", False):
            return -1
        if not (dist.is_available() and dist.is_initialized()):
            return int(getattr(ps, "cp_rank", 0))

        write_cp_rank = torch.tensor(
            [int(getattr(ps, "cp_rank", -1)) if self.rank == write_rank else -1],
            dtype=torch.long,
            device=get_device_type(),
        )
        dist.broadcast(write_cp_rank, src=write_rank)
        return int(write_cp_rank.item())

    def _gather_teacher_cache_chunks(
        self,
        local_chunks: List[torch.Tensor],
        *,
        write_rank: int,
        slice_key: Optional[int] = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        payload = {"rank": int(self.rank), "slice_key": int(slice_key or 0), "chunks": local_chunks}
        if not (dist.is_available() and dist.is_initialized()):
            return [payload] if self.rank == write_rank else None

        gathered = [None for _ in range(dist.get_world_size())] if self.rank == write_rank else None
        dist.gather_object(payload, gathered, dst=write_rank)
        return [item for item in gathered if item is not None] if self.rank == write_rank else None

    def _write_teacher_hidden_cache(
        self,
        gathered_payloads: List[Dict[str, Any]],
        *,
        cache_path: str,
        cache_key: str,
        cache_dtype: torch.dtype,
        now,
    ) -> tuple[Dict[str, Any], float]:
        chunks, cache_indices_by_sample = self._merge_teacher_hidden_cache_payloads(gathered_payloads)

        if not chunks:
            raise ValueError("teacher_hidden_cache produced no hidden-state chunks to write")

        t_write = now()
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        hidden_cache = torch.cat(chunks, dim=0).contiguous()
        tmp_path = f"{cache_path}.tmp-rank{self.rank}"
        save_file({cache_key: hidden_cache}, tmp_path)
        os.replace(tmp_path, cache_path)
        write_s = now() - t_write

        return (
            {
                "path": cache_path,
                "tensor_key": cache_key,
                "dtype": str(cache_dtype).removeprefix("torch."),
                "num_tokens": int(hidden_cache.shape[0]),
                "hidden_size": int(hidden_cache.shape[1]),
                "cache_indices_by_sample": cache_indices_by_sample,
            },
            write_s,
        )

    def _teacher_hidden_chunks_from_batch(
        self,
        hidden_states: torch.Tensor,
        micro_batch: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """Split a teacher forward pass into real per-sample hidden-state chunks."""
        hidden_states = hidden_states.detach()
        position_ids = micro_batch.get("_original_position_ids", micro_batch.get("position_ids"))
        labels = micro_batch.get("labels", micro_batch.get("target_tokens"))

        # Packed batches concatenate samples into batch row 0 and record the
        # number of real samples. Padding is represented as one extra position-id
        # segment, so use num_samples to drop it.
        num_samples = int(micro_batch.get("num_samples", 0) or 0)
        if num_samples > 0:
            if position_ids is None:
                raise ValueError("Packed teacher_hidden_cache batches require position_ids")
            chunks, _ = self._split_hidden_cache_rows(hidden_states, labels, position_ids, num_samples)
            return chunks

        chunks, _ = self._split_hidden_cache_rows(hidden_states, labels)
        return chunks

    def _teacher_hidden_cache_contributor_key(self, ps) -> Optional[int]:
        """Return this rank's logical cache slice key, or None for duplicate shards.

        Must mirror the dispatcher's batch_slice_rank_and_size mapping so cache
        rows merge back in client datum order. Under legacy EP batch duplication
        (XORL_SERVER_EP_DUPLICATE_BATCHES=1) only ep_rank 0 contributes.
        """
        if getattr(ps, "cp_enabled", False) and int(getattr(ps, "cp_rank", 0)) != 0:
            return None

        if getattr(ps, "ep_enabled", False):
            if ep_duplicate_batches_enabled() and int(getattr(ps, "ep_rank", 0)) != 0:
                return None
            cp_size = max(1, int(getattr(ps, "cp_size", 1) or 1)) if getattr(ps, "cp_enabled", False) else 1
            pp_size = max(1, int(getattr(ps, "pp_size", 1)))
            slice_rank, _ = batch_slice_rank_and_size(self.rank, self.world_size, ps, cp_size, pp_size)
            return slice_rank

        return int(getattr(ps, "dp_rank", 0))

    @staticmethod
    def _merge_teacher_hidden_cache_payloads(payloads: List[Optional[Dict[str, Any]]]):
        """Merge per-rank teacher-cache chunks in logical data-slice order."""
        chunks: List[torch.Tensor] = []
        cache_indices_by_sample: List[List[int]] = []
        next_index = 0
        ordered_payloads = sorted(
            (payload for payload in payloads if payload),
            key=lambda payload: (int(payload["slice_key"]), int(payload["rank"])),
        )
        for payload in ordered_payloads:
            for chunk in payload["chunks"]:
                rows = int(chunk.shape[0])
                cache_indices_by_sample.append(list(range(next_index, next_index + rows)))
                next_index += rows
                chunks.append(chunk)
        return chunks, cache_indices_by_sample

    def _forward_teacher_hidden_cache(
        self,
        micro_batches: List[Dict[str, Any]],
        params: Dict[str, Any],
        abort_callback=None,
    ) -> Dict[str, Any]:
        if self.pp_enabled:
            raise NotImplementedError("teacher_hidden_cache does not yet support pipeline parallelism")

        cache_path = (
            params.get("teacher_hidden_cache_path") or params.get("hidden_cache_path") or params.get("output_path")
        )
        if not cache_path:
            raise ValueError(
                "teacher_hidden_cache requires loss_fn_params.teacher_hidden_cache_path "
                "(or hidden_cache_path/output_path)"
            )

        cache_key = params.get("teacher_hidden_cache_key", "hidden_states")
        cache_dtype = self._teacher_cache_dtype(params.get("teacher_hidden_cache_dtype", "bfloat16"))
        write_rank = int(params.get("teacher_hidden_cache_write_rank", 0))
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else self.world_size
        if write_rank < 0 or write_rank >= world_size:
            raise ValueError(f"teacher_hidden_cache_write_rank={write_rank} is outside world_size={world_size}")

        profile_sync_cuda = bool(params.get("opd_profile_sync_cuda", False))

        def _now() -> float:
            if profile_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        ps = get_parallel_state()
        contributor_key = self._teacher_hidden_cache_contributor_key(ps)
        write_cp_rank = self._teacher_cache_write_cp_rank(ps, write_rank)
        is_contributor = contributor_key is not None and (
            not getattr(ps, "cp_enabled", False) or int(getattr(ps, "cp_rank", 0)) == write_cp_rank
        )

        forward_compute_s = 0.0
        local_chunks: List[torch.Tensor] = []

        for micro_batch in micro_batches:
            if abort_callback and abort_callback():
                raise RuntimeError("Execution aborted by request")

            micro_batch = {
                k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in micro_batch.items()
            }

            model_inputs = {
                k: v for k, v in micro_batch.items() if k not in self._LOSS_EXCLUDE_KEYS["teacher_hidden_cache"]
            }

            t_forward = _now()
            with self.model_fwd_context:
                outputs = self.model(**model_inputs, use_cache=False, output_hidden_states=False)
            hidden_states, micro_batch = self._gather_teacher_cache_sequences(
                outputs.last_hidden_state, micro_batch, ps
            )
            forward_compute_s += _now() - t_forward

            if is_contributor:
                for chunk in self._teacher_hidden_chunks_from_batch(hidden_states, micro_batch):
                    local_chunks.append(chunk.to(device="cpu", dtype=cache_dtype))

            del outputs, hidden_states, micro_batch, model_inputs

        gathered_payloads = self._gather_teacher_cache_chunks(
            local_chunks,
            write_rank=write_rank,
            slice_key=contributor_key,
        )

        cache_metadata = None
        write_s = 0.0
        if self.rank == write_rank:
            cache_metadata, write_s = self._write_teacher_hidden_cache(
                gathered_payloads or [],
                cache_path=cache_path,
                cache_key=cache_key,
                cache_dtype=cache_dtype,
                now=_now,
            )

        if dist.is_available() and dist.is_initialized():
            metadata_box = [cache_metadata]
            dist.broadcast_object_list(metadata_box, src=write_rank)
            cache_metadata = metadata_box[0]

        if cache_metadata is None:
            raise RuntimeError("teacher_hidden_cache metadata was not produced")

        result = {
            "total_loss": 0.0,
            "global_valid_tokens": cache_metadata["num_tokens"],
            "teacher_hidden_cache": cache_metadata,
            "teacher_prefill_tokens": cache_metadata["num_tokens"],
            "teacher_prefill_forward_compute_s": forward_compute_s,
            "teacher_hidden_cache_write_s": write_s,
        }
        return result

    @staticmethod
    def _metric_accumulator_key(metric_name: str, loss_fn: str) -> tuple[str, str] | None:
        """Return output metric key and reduction mode for a loss metric."""
        if loss_fn == "opd_loss":
            if metric_name == "valid_tokens":
                # The top-level global_valid_tokens field already reports this.
                return None
            if metric_name == "opd_num_teachers":
                return "opd_num_teachers:max", "max"
            if metric_name.startswith("opd_profile_") and metric_name.endswith("_ms"):
                return metric_name, "sum_max"
            if metric_name.startswith("opd_"):
                return metric_name, "mean"
            return None

        if metric_name == "ratio_min":
            return "is_ratio_min", "min"
        if metric_name == "ratio_max":
            return "is_ratio_max", "max"
        return f"is_{metric_name}", "mean"

    @staticmethod
    def _accumulate_loss_metrics(accumulated, new_metrics, loss_fn: str, metric_ops=None):
        """Accumulate loss-specific metrics across micro-batches.

        RL loss metrics are reducer partials, so mean metrics already carry a
        token-sum share. OPD keeps human-readable per-micro-batch means for its
        namespaced metrics, so those are weighted by valid-token count here.
        """
        if not new_metrics:
            return
        metric_ops = metric_ops or {}
        new_metrics = dict(new_metrics)
        new_metrics.pop("_n_valid_kl", None)
        n_tokens = float(new_metrics.get("valid_tokens", 1))
        device = get_device_type()
        for k, v in new_metrics.items():
            accumulator_key = ModelRunner._metric_accumulator_key(k, loss_fn)
            if accumulator_key is None:
                continue
            output_key, default_op = accumulator_key
            op = metric_ops.get(k, default_op)
            value = torch.as_tensor(v, dtype=torch.float64, device=device)

            if op in ("min", "max"):
                entry = accumulated.get(output_key)
                if entry is None:
                    accumulated[output_key] = {"value": value.clone(), "op": op}
                else:
                    entry["value"] = (
                        torch.minimum(entry["value"], value) if op == "min" else torch.maximum(entry["value"], value)
                    )
                continue

            if op == "sum_max":
                entry = accumulated.get(output_key)
                if entry is None:
                    accumulated[output_key] = {"sum": value.clone(), "op": op}
                else:
                    entry["sum"] = entry["sum"] + value
                continue

            if loss_fn == "opd_loss":
                value_sum = value * n_tokens
                count = n_tokens
            elif k == "valid_tokens":
                value_sum = value
                count = 1.0
            else:
                value_sum = value
                count = n_tokens

            entry = accumulated.get(output_key)
            if entry is None:
                accumulated[output_key] = {"sum": value_sum.clone(), "count": float(count), "op": "mean"}
            else:
                entry["sum"] = entry["sum"] + value_sum
                entry["count"] += float(count)

    @staticmethod
    def _ensure_opd_loss_metric_accumulators(accumulated, include_profile_metrics: bool = False):
        """Ensure all OPD ranks enter the same loss-metric collectives.

        EP/DP packing can leave some ranks with no local micro-batches. Those
        ranks still have to participate in OPD metric reductions, otherwise
        ranks with metrics block in the loss-group all-reduce while empty
        ranks move on to the dispatcher error-sync collective and deadlock.
        Seed zero-valued entries with the canonical key set so every rank
        runs the same `_finalize_loss_metrics` collective.
        """
        device = get_device_type()

        def zero():
            return torch.tensor(0.0, dtype=torch.float64, device=device)

        # Derive the canonical key set from ``OPDLossMetrics.to_dict()`` (mapped
        # through ``_metric_accumulator_key``) so it can NEVER drift from what
        # populated ranks actually emit -- a hardcoded list silently desyncs the
        # day a new metric field is added (e.g. opd_loss_clamp_frac / the split
        # metrics were missing). Ranks that ran at least one OPD micro-batch get
        # these via ``_accumulate_loss_metrics``; empty ranks must seed the
        # identical set or the all-reduce in ``_finalize_loss_metrics`` mismatches
        # in size. opd_num_teachers must be non-None so to_dict() emits it (the
        # dropped key that otherwise desyncs the "max" group).
        for name in OPDLossMetrics(valid_tokens=0, opd_num_teachers=0).to_dict():
            mapped = ModelRunner._metric_accumulator_key(name, "opd_loss")
            if mapped is None:
                continue
            key, op = mapped
            if key in accumulated:
                continue
            if op in ("min", "max"):
                accumulated[key] = {"value": zero(), "op": op}
            elif op == "sum_max":
                accumulated[key] = {"sum": zero(), "op": op}
            else:  # mean
                accumulated[key] = {"sum": zero(), "count": 0.0, "op": "mean"}

        if include_profile_metrics:
            for key in (
                "opd_profile_prefetch_ms",
                "opd_profile_hidden_fetch_ms",
                "opd_profile_head_prepare_ms",
                "opd_profile_kl_compute_ms",
                "opd_profile_total_ms",
                "opd_profile_model_forward_ms",
                "opd_profile_loss_compute_ms",
                "opd_profile_oprd_teacher_forward_ms",
                "opd_profile_oprd_layer_fetch_ms",
            ):
                if key not in accumulated:
                    accumulated[key] = {"sum": zero(), "op": "sum_max"}

    @staticmethod
    def _finalize_loss_metrics(accumulated, result, loss_fn: Optional[str] = None):
        """All-reduce loss metrics, then add reduced values to result dict."""
        if not accumulated:
            return
        ps = get_parallel_state()
        if loss_fn is None and all(str(k).startswith("opd_") for k in accumulated):
            loss_fn = "opd_loss"

        if loss_fn == "opd_loss":
            reduce_group = ps.loss_group if ps.loss_parallel_enabled else None
            should_reduce = ps.loss_parallel_enabled and dist.is_available() and dist.is_initialized()
        else:
            reduce_group = ps.dp_group if ps.dp_enabled else None
            should_reduce = ps.dp_enabled and dist.is_available() and dist.is_initialized()

        groups: Dict[str, list[str]] = {"mean": [], "min": [], "max": [], "sum_max": []}
        for k, entry in accumulated.items():
            groups[entry["op"]].append(k)
        for keys in groups.values():
            keys.sort()

        if groups["mean"]:
            keys = groups["mean"]
            sums = torch.stack([accumulated[k]["sum"] for k in keys])
            counts = torch.tensor(
                [accumulated[k]["count"] for k in keys],
                dtype=torch.float64,
                device=get_device_type(),
            )
            if should_reduce:
                sums_and_counts = torch.cat([sums, counts])
                dist.all_reduce(sums_and_counts, op=dist.ReduceOp.SUM, group=reduce_group)
                sums, counts = sums_and_counts[: len(keys)], sums_and_counts[len(keys) :]
            means = (sums / counts.clamp(min=1.0)).tolist()
            mask = (counts > 0).tolist()
            for i, k in enumerate(keys):
                if mask[i]:
                    result[k] = means[i]

        for op_name, reduce_op in (("min", dist.ReduceOp.MIN), ("max", dist.ReduceOp.MAX)):
            if not groups[op_name]:
                continue
            keys = groups[op_name]
            stacked = torch.stack([accumulated[k]["value"] for k in keys])
            if should_reduce:
                dist.all_reduce(stacked, op=reduce_op, group=reduce_group)
            values = stacked.tolist()
            for k, v in zip(keys, values):
                result[k] = v if math.isfinite(v) else 1.0

        if groups["sum_max"]:
            keys = groups["sum_max"]
            stacked = torch.stack([accumulated[k]["sum"] for k in keys])
            if should_reduce:
                dist.all_reduce(stacked, op=dist.ReduceOp.MAX, group=reduce_group)
            values = stacked.tolist()
            for k, v in zip(keys, values):
                result[k] = v if math.isfinite(v) else 0.0

    @staticmethod
    def _metric_to_float(value):
        """Convert scalar metric values to Python floats before cross-process serialization."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar metric tensor, got shape {tuple(value.shape)}")
            return float(value.detach().cpu().item())
        return float(value)

    @staticmethod
    def _accumulate_is_metrics(accumulated, new_metrics, metric_ops: Optional[Dict[str, str]] = None):
        """Accumulate importance sampling metrics across micro-batches."""
        if not new_metrics:
            return
        metric_ops = metric_ops or {}
        n_tokens = ModelRunner._metric_to_float(new_metrics.get("valid_tokens", 1))
        for k, raw_v in new_metrics.items():
            if k == "_n_valid_kl":
                continue
            v = ModelRunner._metric_to_float(raw_v)
            if k not in accumulated:
                op_name = _metric_reduce_op(k, metric_ops)
                if op_name == "min":
                    accumulated[k] = {"op": op_name, "sum": float("inf"), "count": 0}
                elif op_name == "max":
                    accumulated[k] = {"op": op_name, "sum": float("-inf"), "count": 0}
                else:
                    accumulated[k] = {"op": op_name, "sum": 0.0, "count": 0}
            op_name = accumulated[k].get("op", _metric_reduce_op(k, metric_ops))
            if k == "valid_tokens":
                accumulated[k]["sum"] += v
                accumulated[k]["count"] += 1
            elif op_name == "min":
                if n_tokens > 0:
                    accumulated[k]["sum"] = min(accumulated[k]["sum"], v)
                    accumulated[k]["count"] += 1
            elif op_name == "max":
                if n_tokens > 0:
                    accumulated[k]["sum"] = max(accumulated[k]["sum"], v)
                    accumulated[k]["count"] += 1
            else:
                accumulated[k]["sum"] += v
                accumulated[k]["count"] += n_tokens

    @staticmethod
    def _finalize_is_metrics(accumulated, result):
        """All-reduce IS metrics across DP group, then add averaged values to result dict."""
        if not accumulated:
            return
        ps = get_parallel_state()
        device = torch.device(get_device_type())
        if ps.dp_enabled:
            dp_group = ps.dp_group
            for k, v in accumulated.items():
                op_name = v.get("op", _metric_reduce_op(k))
                if op_name == "min":
                    t = torch.tensor(v["sum"] if v["count"] > 0 else float("inf"), dtype=torch.float64, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=dp_group)
                    v["sum"] = t.item()
                    v["count"] = 1 if math.isfinite(v["sum"]) else 0
                elif op_name == "max":
                    t = torch.tensor(v["sum"] if v["count"] > 0 else float("-inf"), dtype=torch.float64, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dp_group)
                    v["sum"] = t.item()
                    v["count"] = 1 if math.isfinite(v["sum"]) else 0
                else:
                    sum_t = torch.tensor(v["sum"], dtype=torch.float64, device=device)
                    count_t = torch.tensor(float(v["count"]), dtype=torch.float64, device=device)
                    dist.all_reduce(sum_t, op=dist.ReduceOp.SUM, group=dp_group)
                    dist.all_reduce(count_t, op=dist.ReduceOp.SUM, group=dp_group)
                    v["sum"] = sum_t.item()
                    v["count"] = count_t.item()
        for k, v in accumulated.items():
            op_name = v.get("op", _metric_reduce_op(k))
            if op_name in {"min", "max"}:
                result[f"is_{k}"] = v["sum"] if v["count"] > 0 and math.isfinite(v["sum"]) else 1.0
            elif v["count"] > 0:
                result[f"is_{k}"] = v["sum"] / v["count"]

    def _count_global_valid_tokens(self, micro_batches):
        """Count valid tokens across all micro-batches and all-reduce across DP group.

        Uses fsdp_group (not world group) when PP is enabled, so that PP ranks
        processing the same data don't double-count valid tokens.
        """
        group = get_parallel_state().fsdp_group if self.pp_enabled else None
        return count_valid_tokens(micro_batches, group=group)

    def _count_active_microbatches(self, micro_batches) -> tuple[int, int]:
        """Return ``(active_microbatches, active_voter_total)`` over the DP group."""
        group = get_parallel_state().fsdp_group if self.pp_enabled else None
        return count_active_microbatches(micro_batches, group=group)

    @staticmethod
    def _opd_param(params: Dict[str, Any], *names: str, default=None):
        for name in names:
            if name in params:
                return params[name]
        return default

    def _get_opd_head_manager(self, params: Dict[str, Any]) -> TeacherHeadManager:
        teacher_heads = self._opd_param(
            params,
            "teacher_heads",
            "opd_teacher_heads",
            default=self.train_config.get("opd_teacher_heads"),
        )
        if not teacher_heads:
            raise ValueError("opd_loss requires loss_fn_params.teacher_heads (or train.opd_teacher_heads)")
        config_key = repr(teacher_heads)
        if self._opd_head_manager is None or self._opd_head_config != config_key:
            self._opd_head_manager = TeacherHeadManager(teacher_heads)
            self._opd_head_config = config_key
        return self._opd_head_manager

    def _get_opd_hidden_cache(self, params: Dict[str, Any]) -> TeacherActivationCache:
        hidden_caches = self._opd_param(
            params,
            "teacher_hidden_caches",
            "opd_teacher_hidden_caches",
            "teacher_hidden_path",
            "opd_teacher_hidden_path",
            default=self.train_config.get("opd_teacher_hidden_caches")
            or self.train_config.get("opd_teacher_hidden_path"),
        )
        if not hidden_caches:
            raise ValueError(
                "opd_loss requires teacher_hidden_states in the batch or "
                "loss_fn_params.teacher_hidden_caches/teacher_hidden_path"
            )
        config_key = repr(hidden_caches)
        if self._opd_hidden_cache is None or self._opd_hidden_config != config_key:
            self._opd_hidden_cache = TeacherActivationCache(hidden_caches)
            self._opd_hidden_config = config_key
        return self._opd_hidden_cache

    def _get_opd_layer_cache(self, params: Dict[str, Any]) -> Optional[TeacherActivationCache]:
        """Return the rank-3 multi-layer OPRD teacher cache, or None if unconfigured.

        Mirrors ``_get_opd_hidden_cache`` but reads ``teacher_layer_hidden_caches``
        (the SEPARATE per-layer safetensors written by the teacher when
        ``capture_layer_indices`` is set). The underlying ``TeacherActivationCache``
        already returns ``[*indices.shape, L, d]`` for rank-3 caches.
        """
        layer_caches = self._opd_param(
            params,
            "teacher_layer_hidden_caches",
            "opd_teacher_layer_hidden_caches",
            default=self.train_config.get("opd_teacher_layer_hidden_caches"),
        )
        if not layer_caches:
            return None
        config_key = repr(layer_caches)
        if self._opd_layer_cache is None or self._opd_layer_config != config_key:
            self._opd_layer_cache = TeacherActivationCache(layer_caches)
            self._opd_layer_config = config_key
        return self._opd_layer_cache

    def _resolve_opd_oprd_layer_indices(self, params: Dict[str, Any]) -> Optional[List[int]]:
        """Resolve the multi-layer OPRD decoder-layer subset for this forward.

        Accepts an explicit list under ``opd_oprd_layer_indices`` (preferred; the
        client sends the SAME list to the teacher cache and here so they align), or
        the string ``opd_oprd_layers`` ("every4" / comma list). Returns sorted,
        de-duplicated, in-range layer ids, or None if nothing is configured. The
        student's per-layer output_hidden_states is 0-indexed by decoder layer,
        matching the SGLang teacher's capture convention.
        """
        num_layers = int(getattr(self.model.config, "num_hidden_layers", 0) or 0)
        raw = params.get("opd_oprd_layer_indices")
        indices: List[int] = []
        if raw:
            indices = [int(x) for x in raw]
        else:
            spec = str(params.get("opd_oprd_layers", "") or "").strip().lower()
            if not spec:
                return None
            if spec.startswith("every"):
                try:
                    stride = int(spec[len("every") :])
                except ValueError:
                    stride = 0
                if stride > 0 and num_layers > 0:
                    indices = list(range(0, num_layers, stride))
            else:
                indices = [int(tok) for tok in spec.replace(" ", "").split(",") if tok != ""]
        seen = set()
        resolved: List[int] = []
        for i in indices:
            if (num_layers == 0 or 0 <= i < num_layers) and i not in seen:
                seen.add(i)
                resolved.append(i)
        return sorted(resolved) or None

    def _get_opd_teacher_hidden_states(
        self,
        micro_batch: Dict[str, Any],
        teacher_id: int,
        params: Dict[str, Any],
        dtype: torch.dtype,
        teacher_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "teacher_hidden_states" in micro_batch:
            return micro_batch["teacher_hidden_states"].to(get_device_type(), dtype=dtype, non_blocking=True)

        cache_indices = micro_batch.get("teacher_cache_indices")
        if cache_indices is None:
            raise ValueError("opd_loss requires teacher_cache_indices when teacher_hidden_states are not provided")

        if teacher_mask is not None:
            mask = teacher_mask.to(device=cache_indices.device)
            selected_indices = cache_indices[mask]
            if selected_indices.numel() == 0:
                raise ValueError(f"No cache indices available for teacher_id={teacher_id}")
            cache_indices = cache_indices.masked_fill(~mask, selected_indices.reshape(-1)[0])

        cache = self._get_opd_hidden_cache(params)
        return cache.get(teacher_id, cache_indices, device=get_device_type(), dtype=dtype)

    def _get_opd_teacher_layer_hidden_states(
        self,
        micro_batch: Dict[str, Any],
        teacher_id: int,
        layer_cache: TeacherActivationCache,
        dtype: torch.dtype,
        teacher_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the rank-3 teacher layer cache → [group_valid, L, d].

        ``layer_cache.get`` returns ``[*indices.shape, L, d]`` for the rank-3 cache;
        we then keep only the group's valid positions (same order the loss applies
        its internal valid_mask). The cache rows and the rank-2 KL cache share token
        ordering, so the same ``teacher_cache_indices`` index both.
        """
        cache_indices = micro_batch.get("teacher_cache_indices")
        if cache_indices is None:
            raise ValueError("opd_oprd_enabled requires teacher_cache_indices for the rank-3 layer cache")
        mask = teacher_mask.to(device=cache_indices.device)
        selected_indices = cache_indices[mask]
        if selected_indices.numel() == 0:
            raise ValueError(f"No cache indices available for teacher_id={teacher_id}")
        safe_indices = cache_indices.masked_fill(~mask, selected_indices.reshape(-1)[0])
        gathered = layer_cache.get(teacher_id, safe_indices, device=get_device_type(), dtype=dtype)
        if gathered.ndim < 3:
            raise ValueError(
                f"teacher layer cache for teacher_id={teacher_id} is not rank-3 "
                f"([layers, tokens, d]); got gathered shape {tuple(gathered.shape)}"
            )
        # gathered: [batch, seq, L, d] -> [valid, L, d]
        flat = gathered.reshape(-1, *gathered.shape[-2:])
        return flat[valid_mask.reshape(-1)]

    def _trainer_teacher_kept_layers(
        self,
        micro_batch: Dict[str, Any],
        opd_oprd_layer_indices: List[int],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Trainer-side teacher multi-layer hiddens, aligned 1:1 with the student.

        Self-distillation OPRD: the teacher is the SAME (frozen) model weights run on
        the teacher sequence (prompt + CoT + pause + answer for match_cot/insert). We
        run an INDEPENDENT no-grad forward on ``teacher_input_ids`` with
        ``output_hidden_states=True``, select the configured decoder-layer subset, and
        gather the teacher's KEPT positions (the positions whose target_tokens != -100,
        which is exactly what the SGLang rank-3 cache stored, in the same order).

        Alignment guarantee (replicates the SGLang-cache ``layer_cache.get`` semantics
        exactly): the kept positions form the cache-row tensor ``[num_kept, L, d]`` in
        ascending position order — identical to the row ordering the teacher cache
        wrote. We then index it by the student's per-position ``teacher_cache_indices``
        (the client's ``remapped_indices``) at the supervised (valid_mask) positions,
        in student order. Because the client builds remapped_indices so that
        student-supervised-position j maps to teacher kept-row j (the server returns a
        contiguous ``range(num_kept)``), this yields the student's supervised hiddens'
        teacher counterparts in matching order — identical to what
        ``_get_opd_teacher_layer_hidden_states`` produced from the cache.
        """
        teacher_input_ids = micro_batch.get("teacher_input_ids")
        teacher_kept_indices = micro_batch.get("teacher_kept_indices")
        device = get_device_type()
        # COLLECTIVE-UNIFORM: the teacher model forward issues FSDP all-gathers that
        # MUST fire on EVERY rank — including 0-valid / dummy ranks under FSDP data
        # sharding, which early-return from the loss before the per-teacher loop. When
        # a rank lacks teacher_input_ids/teacher_kept_indices, run the forward on its
        # student input_ids purely for the collective side-effect and return None (such
        # a rank has no supervised positions, so the kept tensor is never read).
        is_dummy = teacher_input_ids is None or teacher_kept_indices is None
        fwd_ids = micro_batch.get("input_ids") if is_dummy else teacher_input_ids
        if fwd_ids is None:
            raise ValueError(
                "opd_oprd_enabled (trainer-forward) requires teacher_input_ids or "
                "input_ids in the micro-batch (for the uniform teacher forward)"
            )
        fwd_ids = fwd_ids.to(device=device, dtype=torch.long)
        if fwd_ids.dim() == 1:
            fwd_ids = fwd_ids.unsqueeze(0)

        # Frozen teacher: weights are shared with the student, so a no_grad forward is
        # the only thing that keeps it out of the autograd graph.
        fwd_kwargs = {"input_ids": fwd_ids, "use_cache": False, "output_hidden_states": False}
        # Block-diagonal (per-sample) teacher attention under PACKING: teacher_position_ids
        # resets to 0 at each packed sample (built by the server packer). Derive varlen
        # cu_seqlens so the teacher forward does NOT attend across packed samples — matching
        # the student's packed block-diagonal attention. cu_seq_lens_q routes to both the
        # full-attn (FA3 varlen) and linear-attn (cu_seqlens) layers. Absent (single
        # unpacked sample / dummy fallback) → plain single-sequence causal forward.
        if not is_dummy:
            t_pos = micro_batch.get("teacher_position_ids")
            if t_pos is not None:
                if not torch.is_tensor(t_pos):
                    t_pos = torch.tensor(t_pos, dtype=torch.long)
                t_pos = t_pos.to(device=device, dtype=torch.long).reshape(1, -1)
                if t_pos.shape[-1] == fwd_ids.shape[-1] and int((t_pos == 0).sum()) > 1:
                    cu = pos2culen(t_pos).to(device=device, dtype=torch.int32)
                    seglen = cu[1:] - cu[:-1]
                    max_len = int(seglen.max().item()) if seglen.numel() else int(fwd_ids.shape[-1])
                    fwd_kwargs.update(
                        position_ids=t_pos,
                        cu_seq_lens_q=cu,
                        cu_seq_lens_k=cu,
                        max_length_q=max_len,
                        max_length_k=max_len,
                    )
        if is_dummy:
            with torch.no_grad():
                self.model(**fwd_kwargs)
            # Collective side-effect only; this rank has no supervised positions.
            return None
        teacher_kept_indices = teacher_kept_indices.to(device=device, dtype=torch.long).reshape(-1)
        # Fail loud on the HOST (the device gather assert is async + unattributable):
        # teacher_kept_indices must index into the trainer teacher seq [0, S_teacher).
        n_teacher = int(fwd_ids.numel())
        if teacher_kept_indices.numel():
            kmax = int(teacher_kept_indices.max().item())
            kmin = int(teacher_kept_indices.min().item())
            if kmin < 0 or kmax >= n_teacher:
                raise ValueError(
                    f"OPRD teacher_kept_indices out of bounds: range=[{kmin},{kmax}] vs teacher "
                    f"seq len={n_teacher} (trainer teacher_input_ids len={fwd_ids.shape[-1]} "
                    f"mismatches the client's kept indices; num_kept={teacher_kept_indices.numel()})"
                )
        teacher_keep_mask = torch.zeros(n_teacher, device=device, dtype=torch.bool)
        teacher_keep_mask[teacher_kept_indices] = True
        teacher_keep_mask = teacher_keep_mask.reshape(fwd_ids.shape)
        layer_captures, layer_handles = self._install_opd_selected_layer_hooks(
            opd_oprd_layer_indices,
            teacher_keep_mask,
        )
        try:
            with torch.no_grad():
                self.model(**fwd_kwargs)
        finally:
            for handle in layer_handles:
                handle.remove()
        missing = [idx for idx in opd_oprd_layer_indices if idx not in layer_captures]
        if missing:
            raise ValueError(f"OPRD teacher selected-layer capture did not observe layers: {missing}")
        # [num_kept, L, d] in ascending kept-position order. The per-group
        # student↔teacher index is applied at the call site.
        return torch.stack([layer_captures[idx] for idx in opd_oprd_layer_indices], dim=1).to(dtype=dtype)

    @staticmethod
    def _opd_oprd_last_k_weights(
        group_labels: torch.Tensor,
        base_weights: Optional[torch.Tensor],
        last_k: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """OPRD-only weights that keep just the last-k valid positions per sample.

        Returns a [batch, seq] tensor (multiplier) that is `base_weights` (or 1.0)
        on the last-k valid positions of each sample and 0 elsewhere, so the OPRD
        term is restricted to those positions WITHOUT touching the KL term (which
        uses teacher_weights separately). In packed batches, position-id resets mark
        sample boundaries within a row. Without position_ids, this preserves the
        legacy per-row behavior for unpacked batches.
        """
        device = group_labels.device
        valid = group_labels != IGNORE_INDEX
        keep = torch.zeros_like(group_labels, dtype=torch.bool)
        rows = group_labels.reshape(group_labels.shape[0], -1)
        valid_rows = valid.reshape(rows.shape)
        keep_rows = keep.reshape(rows.shape)
        pos_rows = None
        if position_ids is not None:
            if position_ids.numel() != group_labels.numel():
                raise ValueError(
                    "opd_oprd_last_k position_ids must be labels-aligned, got "
                    f"position_ids={tuple(position_ids.shape)} labels={tuple(group_labels.shape)}"
                )
            pos_rows = position_ids.to(device=device, dtype=torch.long).reshape(rows.shape)
        for r in range(rows.shape[0]):
            idx = valid_rows[r].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if pos_rows is None:
                keep_rows[r, idx[-last_k:]] = True
                continue

            row_pos = pos_rows[r, idx]
            boundaries = (row_pos[1:] <= row_pos[:-1]).nonzero(as_tuple=True)[0] + 1
            starts = [0] + boundaries.tolist()
            ends = boundaries.tolist() + [idx.numel()]
            for start, end in zip(starts, ends):
                sample_idx = idx[start:end]
                if sample_idx.numel() == 0:
                    continue
                keep_rows[r, sample_idx[-last_k:]] = True
        if base_weights is not None:
            weights = base_weights.to(device=device, dtype=torch.float32).reshape(keep.shape)
        else:
            weights = torch.ones_like(group_labels, dtype=torch.float32)
        return weights * keep.to(weights.dtype)

    def _compute_opd_micro_batch_loss(
        self,
        hidden_states: torch.Tensor,
        student_weight: torch.Tensor,
        micro_batch: Dict[str, Any],
        params: Dict[str, Any],
        loss_reducer=None,
        student_lm_head=None,
        student_layer_hidden_states: Optional[torch.Tensor] = None,
        opd_oprd_layer_indices: Optional[List[int]] = None,
    ) -> LossOutput:
        if get_parallel_state().tp_enabled:
            raise NotImplementedError("opd_loss does not yet support tensor parallelism")
        if self.pp_enabled:
            # Mirrors the dispatcher-level guard in _run_forward_backward / _run_forward —
            # belt-and-suspenders so a future direct caller can't accidentally launch
            # OPD under PP without that pathway being thought through.
            raise NotImplementedError("opd_loss does not yet support pipeline parallelism")

        labels = micro_batch.get("labels", micro_batch.get("target_tokens"))
        if labels is None:
            raise ValueError("opd_loss requires labels or target_tokens for its valid-token mask")

        teacher_ids = micro_batch.get("teacher_ids")
        if teacher_ids is None:
            default_teacher_id = int(params.get("teacher_id", 0))
            teacher_ids = torch.full_like(labels, default_teacher_id)
        else:
            teacher_ids = teacher_ids.to(labels.device)

        teacher_weights = micro_batch.get("teacher_weights")
        if teacher_weights is None and "teacher_weight" in params:
            teacher_weights = torch.full(labels.shape, float(params["teacher_weight"]), device=labels.device)
        hidden_match_weights = micro_batch.get("hidden_match_weights")
        # Metrics-only KL-split inputs (region 0/1/2 = prompt/buffer/answer; sample
        # correctness 1/0/-1). Optional — absent on legacy clients and dummy datums.
        diag_region_ids = micro_batch.get("opd_region_ids")
        diag_sample_ok = micro_batch.get("opd_sample_ok")
        profile_timings = bool(params.get("opd_profile_timings", False))
        profile_sync_cuda = bool(params.get("opd_profile_sync_cuda", False))

        def _profile_now() -> float:
            if profile_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        def _profile_elapsed_ms(start: float) -> float:
            return (_profile_now() - start) * 1000.0

        valid_mask = labels != IGNORE_INDEX
        local_valid_tokens = valid_mask.sum()
        lm_head_anchor = self._lm_head_forward_anchor(hidden_states, student_lm_head)
        hidden_anchor = self._fp32_zero_anchor(hidden_states)
        weight_anchor = self._fp32_zero_anchor(student_weight)

        # OPRD multi-layer hiddens: the teacher MODEL forward (FSDP all-gathers) MUST
        # run on EVERY rank with identical collective count, BEFORE the 0-valid
        # early-return below — otherwise dummy / 0-valid ranks (FSDP data sharding)
        # skip it and desync against valid ranks (forward all-gathers vs backward
        # reduce-scatters → both spin at sm=100%/mem=0%). The gate is uniform across
        # ranks: the params flag AND the student-layer capture (capture_student_layers
        # is params-driven, so non-None on all ranks that ran the student forward).
        # The per-group student↔teacher index is applied later, in the per-teacher loop.
        opd_oprd_enabled = bool(params.get("opd_oprd_enabled", False)) and student_layer_hidden_states is not None
        layer_cache = self._get_opd_layer_cache(params) if opd_oprd_enabled else None
        teacher_kept_layers = None
        profile_oprd_teacher_forward_ms = 0.0
        profile_oprd_layer_fetch_ms = 0.0
        if opd_oprd_enabled:
            if opd_oprd_layer_indices is None:
                raise ValueError("opd_oprd_enabled requires opd_oprd_layer_indices (the captured layer subset)")
            if layer_cache is None and getattr(get_parallel_state(), "cp_enabled", False):
                # The trainer-side teacher forward runs the FULL teacher sequence on
                # every rank; incompatible with a sequence-sharded student.
                raise NotImplementedError(
                    "opd_oprd_enabled (trainer-side teacher forward) does not support "
                    "sequence/context parallelism"
                )
            if layer_cache is None:
                profile_start = _profile_now() if profile_timings else 0.0
                teacher_kept_layers = self._trainer_teacher_kept_layers(
                    micro_batch, opd_oprd_layer_indices, dtype=hidden_states.dtype
                )
                if profile_timings:
                    profile_oprd_teacher_forward_ms += _profile_elapsed_ms(profile_start)

        if local_valid_tokens.item() == 0:
            # fp32 to match opd_loss_function's fp32 normal-return path. With
            # ulysses sequence sharding, ranks with no response tokens hit this
            # branch while the rank holding the response runs the fp32 KL kernel.
            # A dtype mismatch corrupts the cross-rank all_reduce in the loss
            # reporter (mixed-dtype byte reinterpretation).
            loss = hidden_anchor + weight_anchor + lm_head_anchor
            metrics = OPDLossMetrics(valid_tokens=0).to_dict()
            if profile_timings:
                metrics["opd_profile_oprd_teacher_forward_ms"] = profile_oprd_teacher_forward_ms
                metrics["opd_profile_oprd_layer_fetch_ms"] = profile_oprd_layer_fetch_ms
            return LossOutput(loss=loss, metrics=metrics)

        head_manager = self._get_opd_head_manager(params)
        unique_teacher_ids = sorted(int(x) for x in torch.unique(teacher_ids[valid_mask]).tolist())
        kl_backend = params.get("opd_kl_backend", params.get("kl_backend", "torch_compile"))
        use_sharded_backend = str(kl_backend).lower() in {"streaming", "tilelang"}
        async_prefetch = bool(params.get("opd_async_prefetch", True))
        teacher_head_fp32 = bool(params.get("teacher_lm_head_fp32", True))
        teacher_head_dtype = torch.float32 if teacher_head_fp32 else student_weight.dtype
        sharded_head_cpu_cache = bool(params.get("opd_sharded_head_cpu_cache", True))
        sharded_head_device_cache = bool(params.get("opd_sharded_head_device_cache", False))
        # OPD loss-mode knobs. Defaults preserve the existing reverse-KL-only behavior.
        opd_loss_mode = str(params.get("opd_loss_mode", "reverse_kl_full"))
        opd_log_prob_min_clamp = params.get("opd_log_prob_min_clamp", None)
        opd_loss_max_clamp = params.get("opd_loss_max_clamp", None)
        opd_emit_full_vocab_diagnostics = bool(params.get("opd_emit_full_vocab_diagnostics", False))
        opd_use_policy_gradient = bool(params.get("opd_use_policy_gradient", False))
        opd_clip_ratio_low = float(params.get("opd_clip_ratio_low", params.get("opd_clip_ratio", 0.2)))
        opd_clip_ratio_high = float(params.get("opd_clip_ratio_high", params.get("opd_clip_ratio", 0.2)))
        opd_use_task_rewards = bool(params.get("opd_use_task_rewards", False))
        opd_distillation_loss_coef = float(params.get("opd_distillation_loss_coef", 1.0))
        opd_hidden_match_coef = float(params.get("opd_hidden_match_coef", 0.0) or 0.0)
        # NB: use an explicit None check (not `or 1.0`) so an intentional 0.0 (hidden-only) survives.
        _opd_klw = params.get("opd_kl_loss_weight", 1.0)
        opd_kl_loss_weight = float(1.0 if _opd_klw is None else _opd_klw)
        opd_hidden_match_mode = str(params.get("opd_hidden_match_mode", "cosine") or "cosine")
        # When PG mode is on, the runner must surface old_logprobs from the micro-batch.
        opd_old_logprobs_full = micro_batch.get("old_logprobs") if opd_use_policy_gradient else None

        # Multi-layer OPRD: active only when the student per-layer subset was captured
        # this forward (single-layer hidden-match path is unchanged when inactive). The
        # teacher multi-layer hiddens come from the TRAINER-SIDE no-grad forward run
        # uniformly above (before the 0-valid early-return), in ``teacher_kept_layers``;
        # ``opd_oprd_enabled`` was resolved there. The legacy ``_get_opd_layer_cache`` /
        # ``_get_opd_teacher_layer_hidden_states`` pair is kept (dead) for callers still
        # wiring teacher_layer_hidden_caches.
        opd_oprd_last_k = int(params.get("opd_oprd_last_k", 0) or 0)

        profile_total_start = _profile_now() if profile_timings else 0.0
        profile_prefetch_ms = 0.0
        profile_hidden_fetch_ms = 0.0
        profile_head_prepare_ms = 0.0
        profile_kl_compute_ms = 0.0

        hidden_cache = None
        if async_prefetch:
            profile_start = _profile_now() if profile_timings else 0.0
            if not (use_sharded_backend and head_manager.has_sharded_head(unique_teacher_ids[0])):
                head_manager.prefetch(unique_teacher_ids[0])
            if "teacher_hidden_states" not in micro_batch:
                hidden_cache = self._get_opd_hidden_cache(params)
                hidden_cache.prefetch(unique_teacher_ids[0])
            if profile_timings:
                profile_prefetch_ms += _profile_elapsed_ms(profile_start)
        if loss_reducer is None:
            loss_reducer = TokenPartial(scale=torch.tensor(1.0, device=hidden_states.device))

        # fp32 — matches opd_loss_function's fp32 return; consistent across ranks.
        total_loss = hidden_anchor + weight_anchor + lm_head_anchor
        weighted_kl_metric = 0.0
        hidden_match_loss_sum = 0.0
        hidden_match_raw_loss_sum = 0.0
        hidden_match_weight_sum = 0.0
        hidden_match_pos_loss_sum = 0.0
        hidden_match_neg_loss_sum = 0.0
        hidden_match_pos_raw_loss_sum = 0.0
        hidden_match_neg_raw_loss_sum = 0.0
        hidden_match_neg_minus_pos_raw_sum = 0.0
        hidden_match_pos_weight_sum = 0.0
        hidden_match_neg_weight_sum = 0.0
        kl_sum = 0.0
        teacher_weight_sum = 0.0
        # Diagnostic accumulators (weighted by group_valid; averaged by valid_count).
        teacher_entropy_sum = 0.0
        student_entropy_sum = 0.0
        top1_agreement_sum = 0.0
        abs_loss_sum = 0.0
        loss_abs_mean_sum = 0.0
        pg_clipfrac_sum = 0.0
        pg_clipfrac_lower_sum = 0.0
        ppo_kl_sum = 0.0
        # Multi-layer OPRD accumulators (loss weighted by group_valid; num_layers is a
        # per-group constant L, so track the max across groups).
        oprd_loss_sum = 0.0
        oprd_raw_loss_sum = 0.0
        oprd_num_layers = 0
        # Clamp-frac + region/correctness split accumulators. All are linear per-valid
        # quantities, so group_valid weighting + denom division recomposes them exactly.
        split_metric_sums = dict.fromkeys(_OPD_SPLIT_METRIC_KEYS, 0.0)
        # min/max are tracked as actual mins/maxes across teacher groups, not weighted means.
        loss_min = float("inf")
        loss_max = float("-inf")
        valid_count = int(local_valid_tokens.item())

        for teacher_index, teacher_id_int in enumerate(unique_teacher_ids):
            if async_prefetch and teacher_index + 1 < len(unique_teacher_ids):
                profile_start = _profile_now() if profile_timings else 0.0
                next_teacher_id = unique_teacher_ids[teacher_index + 1]
                if not (use_sharded_backend and head_manager.has_sharded_head(next_teacher_id)):
                    head_manager.prefetch(next_teacher_id)
                if hidden_cache is not None:
                    hidden_cache.prefetch(next_teacher_id)
                if profile_timings:
                    profile_prefetch_ms += _profile_elapsed_ms(profile_start)
            teacher_mask = valid_mask & (teacher_ids == teacher_id_int)
            if not teacher_mask.any():
                continue

            group_labels = labels.masked_fill(~teacher_mask, IGNORE_INDEX)
            profile_start = _profile_now() if profile_timings else 0.0
            teacher_hidden_states = self._get_opd_teacher_hidden_states(
                micro_batch,
                teacher_id_int,
                params,
                dtype=hidden_states.dtype,
                teacher_mask=teacher_mask,
            )
            if profile_timings:
                profile_hidden_fetch_ms += _profile_elapsed_ms(profile_start)

            # Multi-layer OPRD: build [group_valid, L, d] student/teacher layer
            # tensors (valid_mask order over group_labels) + an OPRD-only weights
            # vector (so last-k restriction never touches the KL term).
            group_teacher_layers = None
            group_student_layers = None
            group_hidden_match_weights = hidden_match_weights
            if opd_oprd_enabled:
                group_valid_mask = group_labels != IGNORE_INDEX
                flat_valid = group_valid_mask.reshape(-1)
                student_layers_flat = student_layer_hidden_states.reshape(
                    -1, *student_layer_hidden_states.shape[-2:]
                )
                # New selected-hook capture provides exactly one row per valid
                # token in labels-valid order: [valid, L, d]. Legacy
                # output_hidden_states capture provides full rows:
                # [batch*seq, L, d]. Support both so candidates can A/B the
                # memory-saving path against the old path.
                if student_layers_flat.shape[0] == int(local_valid_tokens.item()):
                    valid_teacher_ids = teacher_ids.reshape(-1)[valid_mask.reshape(-1)]
                    group_student_layers = student_layers_flat[valid_teacher_ids == teacher_id_int]
                else:
                    group_student_layers = student_layers_flat[flat_valid]
                if layer_cache is not None:
                    profile_start = _profile_now() if profile_timings else 0.0
                    group_teacher_layers = self._get_opd_teacher_layer_hidden_states(
                        micro_batch,
                        teacher_id_int,
                        layer_cache,
                        dtype=hidden_states.dtype,
                        teacher_mask=teacher_mask,
                        valid_mask=group_valid_mask,
                    )
                    if profile_timings:
                        profile_oprd_layer_fetch_ms += _profile_elapsed_ms(profile_start)
                else:
                    # Index the teacher's kept-position hiddens (computed once,
                    # uniformly, in teacher_kept_layers) by the student's per-position
                    # remap at this group's supervised positions, in student order.
                    # Use the packer-emitted PER-MICRO-BATCH-LOCAL view
                    # (teacher_cache_local_indices), NOT teacher_cache_indices: the
                    # latter carries GLOBAL teacher-cache rows for the KL hidden-fetch
                    # and walks out of bounds here (see
                    # docs/notes/oprd_warm_cache_indices_rebase_bug.md).
                    local_cache_indices = micro_batch.get("teacher_cache_local_indices")
                    if local_cache_indices is None:
                        raise ValueError(
                            "opd_oprd_enabled requires teacher_cache_local_indices for student↔teacher "
                            "alignment (packer-emitted local view of teacher_cache_indices)"
                        )
                    if teacher_kept_layers is None:
                        raise ValueError(
                            "opd_oprd_enabled: teacher_kept_layers missing on a rank with supervised tokens"
                        )
                    selected = local_cache_indices.reshape(-1)[flat_valid].to(device=teacher_kept_layers.device)
                    # Fail loud on the HOST: selected must index into [0, num_kept).
                    n_kept = teacher_kept_layers.shape[0]
                    if selected.numel():
                        smax = int(selected.max().item())
                        smin = int(selected.min().item())
                        if smin < 0 or smax >= n_kept:
                            raise ValueError(
                                f"OPRD selected (teacher_cache_local_indices) out of bounds: range=[{smin},{smax}] "
                                f"vs num_kept={n_kept} (per-position remap misaligned with teacher_kept_indices; "
                                f"group supervised positions={int(flat_valid.sum().item())})"
                            )
                    group_teacher_layers = teacher_kept_layers[selected]
                if opd_oprd_last_k > 0:
                    group_hidden_match_weights = self._opd_oprd_last_k_weights(
                        group_labels,
                        base_weights=hidden_match_weights
                        if hidden_match_weights is not None
                        else teacher_weights,
                        last_k=opd_oprd_last_k,
                        position_ids=micro_batch.get("position_ids"),
                    )

            profile_start = _profile_now() if profile_timings else 0.0
            if use_sharded_backend and head_manager.has_sharded_head(teacher_id_int):
                teacher_head = head_manager.sharded_view(
                    teacher_id_int,
                    device=get_device_type(),
                    dtype=teacher_head_dtype,
                    cache_cpu=sharded_head_cpu_cache,
                    cache_device=sharded_head_device_cache,
                )
            else:
                teacher_head = head_manager.get(
                    teacher_id_int,
                    device=get_device_type(),
                    dtype=teacher_head_dtype,
                )
            if profile_timings:
                profile_head_prepare_ms += _profile_elapsed_ms(profile_start)

            profile_start = _profile_now() if profile_timings else 0.0
            result = opd_loss_function(
                hidden_states=hidden_states,
                weight=student_weight,
                labels=group_labels,
                teacher_hidden_states=teacher_hidden_states,
                teacher_lm_head_weight=teacher_head,
                teacher_weights=teacher_weights,
                ignore_index=IGNORE_INDEX,
                num_chunks=params.get("num_chunks", 8),
                lm_head_fp32=self.lm_head_fp32,
                teacher_lm_head_fp32=teacher_head_fp32,
                kl_backend=kl_backend,
                vocab_chunk_size=params.get("opd_vocab_chunk_size", params.get("vocab_chunk_size", 32768)),
                streaming_lowmem=bool(params.get("opd_streaming_lowmem", False)),
                return_per_token=False,
                loss_reducer=loss_reducer,
                loss_mode=opd_loss_mode,
                log_prob_min_clamp=opd_log_prob_min_clamp,
                loss_max_clamp=opd_loss_max_clamp,
                emit_full_vocab_diagnostics=opd_emit_full_vocab_diagnostics,
                use_policy_gradient=opd_use_policy_gradient,
                old_logprobs=opd_old_logprobs_full,
                clip_ratio_low=opd_clip_ratio_low,
                clip_ratio_high=opd_clip_ratio_high,
                use_task_rewards=opd_use_task_rewards,
                distillation_loss_coef=opd_distillation_loss_coef,
                hidden_match_coef=opd_hidden_match_coef,
                kl_loss_weight=opd_kl_loss_weight,
                hidden_match_mode=opd_hidden_match_mode,
                hidden_match_weights=group_hidden_match_weights,
                teacher_layer_hidden_states=group_teacher_layers,
                student_layer_hidden_states=group_student_layers,
                diag_region_ids=diag_region_ids,
                diag_sample_ok=diag_sample_ok,
            )
            if profile_timings:
                profile_kl_compute_ms += _profile_elapsed_ms(profile_start)
            total_loss = total_loss + result.loss

            metrics = result.metrics or {}
            group_valid = int(metrics.get("valid_tokens", teacher_mask.sum().item()))
            kl_sum += float(metrics.get("opd_kl", 0.0)) * group_valid
            teacher_weight_sum += float(metrics.get("opd_teacher_weight_mean", 1.0)) * group_valid
            weighted_kl_metric += float(metrics.get("opd_weighted_kl", 0.0)) * group_valid
            hidden_match_loss_sum += float(metrics.get("opd_hidden_match_loss", 0.0)) * group_valid
            hidden_match_raw_loss_sum += float(metrics.get("opd_hidden_match_raw_loss", 0.0)) * group_valid
            hidden_match_weight_sum += float(metrics.get("opd_hidden_match_weight_mean", 0.0)) * group_valid
            hidden_match_pos_loss_sum += float(metrics.get("opd_hidden_match_pos_loss", 0.0)) * group_valid
            hidden_match_neg_loss_sum += float(metrics.get("opd_hidden_match_neg_loss", 0.0)) * group_valid
            hidden_match_pos_raw_loss_sum += float(metrics.get("opd_hidden_match_pos_raw_loss", 0.0)) * group_valid
            hidden_match_neg_raw_loss_sum += float(metrics.get("opd_hidden_match_neg_raw_loss", 0.0)) * group_valid
            hidden_match_neg_minus_pos_raw_sum += (
                float(metrics.get("opd_hidden_match_neg_minus_pos_raw", 0.0)) * group_valid
            )
            hidden_match_pos_weight_sum += float(metrics.get("opd_hidden_match_pos_weight_mean", 0.0)) * group_valid
            hidden_match_neg_weight_sum += float(metrics.get("opd_hidden_match_neg_weight_mean", 0.0)) * group_valid
            oprd_loss_sum += float(metrics.get("opd_oprd_loss", 0.0)) * group_valid
            oprd_raw_loss_sum += float(metrics.get("opd_oprd_raw_loss", 0.0)) * group_valid
            oprd_num_layers = max(oprd_num_layers, int(metrics.get("opd_oprd_num_layers", 0)))
            # Full-vocab diagnostics: only emitted when emit_full_vocab_diagnostics=True
            # on a full-vocab loss mode. Default 0.0 from OPDLossMetrics on other paths.
            teacher_entropy_sum += float(metrics.get("opd_teacher_entropy", 0.0)) * group_valid
            student_entropy_sum += float(metrics.get("opd_student_entropy", 0.0)) * group_valid
            top1_agreement_sum += float(metrics.get("opd_top1_agreement", 0.0)) * group_valid
            abs_loss_sum += float(metrics.get("opd_abs_loss", 0.0)) * group_valid
            loss_abs_mean_sum += float(metrics.get("opd_loss_abs_mean", 0.0)) * group_valid
            for split_key in _OPD_SPLIT_METRIC_KEYS:
                split_metric_sums[split_key] += float(metrics.get(split_key, 0.0)) * group_valid
            pg_clipfrac_sum += float(metrics.get("opd_pg_clipfrac", 0.0)) * group_valid
            pg_clipfrac_lower_sum += float(metrics.get("opd_pg_clipfrac_lower", 0.0)) * group_valid
            ppo_kl_sum += float(metrics.get("opd_ppo_kl", 0.0)) * group_valid
            if "opd_loss_min" in metrics or "opd_loss_max" in metrics:
                loss_min = min(loss_min, float(metrics.get("opd_loss_min", loss_min)))
                loss_max = max(loss_max, float(metrics.get("opd_loss_max", loss_max)))

        denom = max(valid_count, 1)
        metrics = OPDLossMetrics(
            valid_tokens=valid_count,
            opd_kl=kl_sum / denom,
            opd_weighted_kl=weighted_kl_metric / denom,
            opd_hidden_match_loss=hidden_match_loss_sum / denom,
            opd_hidden_match_raw_loss=hidden_match_raw_loss_sum / denom,
            opd_hidden_match_weight_mean=hidden_match_weight_sum / denom,
            opd_hidden_match_pos_loss=hidden_match_pos_loss_sum / denom,
            opd_hidden_match_neg_loss=hidden_match_neg_loss_sum / denom,
            opd_hidden_match_pos_raw_loss=hidden_match_pos_raw_loss_sum / denom,
            opd_hidden_match_neg_raw_loss=hidden_match_neg_raw_loss_sum / denom,
            opd_hidden_match_neg_minus_pos_raw=hidden_match_neg_minus_pos_raw_sum / denom,
            opd_hidden_match_pos_weight_mean=hidden_match_pos_weight_sum / denom,
            opd_hidden_match_neg_weight_mean=hidden_match_neg_weight_sum / denom,
            opd_teacher_weight_mean=teacher_weight_sum / denom,
            opd_num_teachers=len(unique_teacher_ids),
            opd_teacher_entropy=teacher_entropy_sum / denom,
            opd_student_entropy=student_entropy_sum / denom,
            opd_top1_agreement=top1_agreement_sum / denom,
            opd_abs_loss=abs_loss_sum / denom,
            opd_loss_min=loss_min if loss_min != float("inf") else 0.0,
            opd_loss_max=loss_max if loss_max != float("-inf") else 0.0,
            opd_loss_abs_mean=loss_abs_mean_sum / denom,
            opd_pg_clipfrac=pg_clipfrac_sum / denom,
            opd_pg_clipfrac_lower=pg_clipfrac_lower_sum / denom,
            opd_ppo_kl=ppo_kl_sum / denom,
            opd_oprd_loss=oprd_loss_sum / denom,
            opd_oprd_raw_loss=oprd_raw_loss_sum / denom,
            opd_oprd_num_layers=oprd_num_layers,
            **{key: value / denom for key, value in split_metric_sums.items()},
        ).to_dict()
        if profile_timings:
            metrics.update(
                {
                    "opd_profile_prefetch_ms": profile_prefetch_ms,
                    "opd_profile_hidden_fetch_ms": profile_hidden_fetch_ms,
                    "opd_profile_head_prepare_ms": profile_head_prepare_ms,
                    "opd_profile_kl_compute_ms": profile_kl_compute_ms,
                    "opd_profile_total_ms": _profile_elapsed_ms(profile_total_start),
                    "opd_profile_oprd_teacher_forward_ms": profile_oprd_teacher_forward_ms,
                    "opd_profile_oprd_layer_fetch_ms": profile_oprd_layer_fetch_ms,
                }
            )

        return LossOutput(
            loss=total_loss,
            metrics=metrics,
        )

    # =========================================================================
    # Loss computation dispatch
    # =========================================================================

    @staticmethod
    def _lm_head_forward_anchor(hidden_states: torch.Tensor, lm_head) -> torch.Tensor:
        """Run lm_head.forward with zero loss contribution for FSDP hook ordering."""
        if lm_head is None or hidden_states.numel() == 0:
            return hidden_states.float().sum() * 0.0
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = lm_head(hidden_flat[:1])
        return logits.float().sum() * 0.0

    @staticmethod
    def _fp32_zero_anchor(tensor: torch.Tensor) -> torch.Tensor:
        """Zero-valued fp32 graph edge without materializing a full fp32 tensor copy."""
        if tensor.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=tensor.device)
        return tensor.reshape(-1)[:1].float().sum() * 0.0

    def _compute_micro_batch_loss(self, micro_batch, loss_fn, loss_fn_params):
        """Compute loss for a single micro-batch."""
        params = loss_fn_params or {}
        return_per_token = params.get("return_per_token", True)

        profile_timings = bool(params.get("opd_profile_timings", False))
        profile_sync_cuda = bool(params.get("opd_profile_sync_cuda", False))

        def _profile_now() -> float:
            if profile_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        def _profile_elapsed_ms(start: float) -> float:
            return (_profile_now() - start) * 1000.0

        exclude_keys = self._LOSS_EXCLUDE_KEYS.get(loss_fn, set())
        model_inputs = {k: v for k, v in micro_batch.items() if k not in exclude_keys}

        # Shared-prefix: the dispatcher already repacked input_ids / loss fields and
        # attached a SharedPrefixContext. Move its index tensors to the model device
        # so it flows (via model_inputs) down to the attention backend. Loss runs on
        # the repacked labels (already in micro_batch); outputs are remapped to the
        # original layout in _collect_per_token_outputs.
        if model_inputs.get("shared_prefix_context") is not None:
            model_inputs["shared_prefix_context"] = model_inputs["shared_prefix_context"].to(
                model_inputs["input_ids"].device
            )

        # Multi-layer OPRD (all-layer hidden matching): when enabled for an opd_loss
        # micro-batch, request the per-layer student hidden states so we can match a
        # subset of decoder layers against the teacher's rank-3 cache. Default OFF →
        # output_hidden_states=False → byte-identical to the existing path.
        opd_oprd_enabled = bool(params.get("opd_oprd_enabled", False)) and loss_fn == "opd_loss"
        opd_oprd_layer_indices = self._resolve_opd_oprd_layer_indices(params) if opd_oprd_enabled else None
        capture_student_layers = bool(opd_oprd_enabled and opd_oprd_layer_indices)
        opd_oprd_student_capture = str(params.get("opd_oprd_student_capture", "selected_hooks") or "selected_hooks")
        opd_oprd_student_capture = opd_oprd_student_capture.strip().lower()
        use_opd_selected_layer_hooks = capture_student_layers and opd_oprd_student_capture not in {
            "output_hidden_states",
            "full",
            "legacy",
        }
        opd_selected_layer_captures = None
        opd_selected_layer_handles = []
        if use_opd_selected_layer_hooks:
            labels_for_oprd = micro_batch.get("labels", micro_batch.get("target_tokens"))
            if labels_for_oprd is None:
                raise ValueError("opd_oprd_enabled requires labels or target_tokens for selected-layer capture")
            opd_selected_layer_captures, opd_selected_layer_handles = self._install_opd_selected_layer_hooks(
                opd_oprd_layer_indices,
                labels_for_oprd != IGNORE_INDEX,
            )

        diagnostic_hidden_states = bool(params.get("diagnostic_hidden_states", False))
        diagnostic_hidden_sample_count = int(params.get("diagnostic_hidden_sample_count", 8) or 0)
        diagnostic_hidden_sample_indices = params.get("diagnostic_hidden_sample_indices")
        diagnostic_topk = int(params.get("diagnostic_topk", 0) or 0)
        diagnostic_hidden_components = bool(params.get("diagnostic_hidden_components", False))
        diagnostic_hidden_component_layers = self._parse_diagnostic_layer_indices(
            params.get("diagnostic_hidden_component_layers", [])
        )
        diagnostic_hidden_component_path = params.get("diagnostic_hidden_component_path")
        diagnostic_component_captures = None
        diagnostic_component_handles = []
        if diagnostic_hidden_components and (diagnostic_topk > 0 or diagnostic_hidden_component_path):
            diagnostic_component_captures, diagnostic_component_handles = self._install_hidden_component_hooks(
                diagnostic_hidden_component_layers
            )

        profile_model_start = _profile_now() if profile_timings else 0.0
        try:
            # OPRD can capture just supervised rows via hooks. The legacy
            # output_hidden_states path remains available for A/Bs and diagnostics.
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=(
                    diagnostic_hidden_states or (capture_student_layers and not use_opd_selected_layer_hooks)
                ),
            )
        finally:
            for handle in opd_selected_layer_handles:
                handle.remove()
            for handle in diagnostic_component_handles:
                handle.remove()
        profile_model_forward_ms = _profile_elapsed_ms(profile_model_start) if profile_timings else 0.0
        hidden_states = outputs.last_hidden_state
        diagnostic_all_hidden_states = None
        if diagnostic_hidden_states:
            diagnostic_all_hidden_states = getattr(outputs, "hidden_states", None)
            if diagnostic_all_hidden_states is None and isinstance(outputs, dict):
                diagnostic_all_hidden_states = outputs.get("hidden_states")
            if diagnostic_all_hidden_states is None:
                diagnostic_all_hidden_states = (hidden_states,)
        if diagnostic_component_captures and diagnostic_hidden_component_path:
            saved_path = self._write_hidden_component_tensor_dump(
                diagnostic_hidden_component_path,
                diagnostic_component_captures,
                labels=micro_batch.get("labels"),
                metadata={
                    "diagnostic_hidden_component_layers": diagnostic_hidden_component_layers,
                    "valid_label_count": int((micro_batch.get("labels") != IGNORE_INDEX).sum().item())
                    if micro_batch.get("labels") is not None
                    else None,
                },
            )
            logger.info("Full hidden-component diagnostic tensors saved to %s", saved_path)
        effective_weight = self._get_effective_lm_head_weight()
        fp8_lm_head = self._get_fp8_lm_head_module(getattr(self.model, "lm_head", None))

        # scale=1 → loss_fns return raw masked sums; normalization deferred to
        # optim_step / _finalize_is_metrics.
        token_sum_reducer = TokenPartial(scale=torch.tensor(1.0, device=hidden_states.device))

        per_token_outputs = {}
        is_metrics = None
        is_metric_ops = None

        if loss_fn in ["causallm_loss", "cross_entropy"]:
            labels = micro_batch.get("labels")
            diagnostic_reference_logits = bool(params.get("diagnostic_reference_logits", False))
            _result = causallm_loss_function(
                hidden_states=hidden_states,
                weight=effective_weight,
                labels=labels,
                return_per_token=return_per_token,
                ce_mode=self.ce_mode,
                lm_head_fp32=self.lm_head_fp32,
                loss_reducer=token_sum_reducer,
                lm_head=fp8_lm_head,
            )
            local_loss_sum = _result.loss
            if return_per_token:
                per_token_outputs["logprobs"] = _result.per_token_logprobs
                per_token_outputs["loss"] = _result.per_token_loss
            if diagnostic_topk > 0 and not return_per_token:
                logger.warning(
                    "diagnostic_topk requires return_per_token=True for per-sample unpacking; skipping diagnostics"
                )
            elif diagnostic_topk > 0:
                token_diagnostics = self._compute_token_diagnostics(
                    hidden_states=hidden_states,
                    weight=effective_weight,
                    labels=labels,
                    topk=diagnostic_topk,
                    lm_head=fp8_lm_head,
                    lm_head_fp32=self.lm_head_fp32,
                    per_token_logprobs=_result.per_token_logprobs,
                    include_weight_reference=diagnostic_reference_logits,
                    all_hidden_states=diagnostic_all_hidden_states,
                    hidden_components=diagnostic_component_captures,
                    hidden_sample_count=diagnostic_hidden_sample_count,
                    hidden_sample_indices=diagnostic_hidden_sample_indices,
                )
                if token_diagnostics is not None:
                    per_token_outputs["token_diagnostics"] = token_diagnostics

        elif loss_fn == "importance_sampling":
            target_tokens = micro_batch.get("target_tokens", micro_batch.get("labels"))
            old_logprobs = micro_batch["logprobs"]
            advantages = micro_batch["advantages"]
            compute_kl_stats = params.get("compute_kl_stats", False)
            diagnostic_reference_logits = bool(params.get("diagnostic_reference_logits", False))

            _result = importance_sampling_loss_function(
                hidden_states=hidden_states,
                weight=effective_weight,
                labels=target_tokens,
                old_logprobs=old_logprobs,
                advantages=advantages,
                ce_mode=self.ce_mode,
                compute_kl_stats=compute_kl_stats,
                lm_head_fp32=self.lm_head_fp32,
                loss_reducer=token_sum_reducer,
                metric_reducer=token_sum_reducer,
                lm_head=fp8_lm_head,
            )
            local_loss_sum = _result.loss
            per_token_outputs["logprobs"] = _result.per_token_logprobs
            is_metrics = _result.metrics
            is_metric_ops = _result.metric_ops
            if is_metrics is not None and "valid_tokens" not in is_metrics:
                is_metrics["valid_tokens"] = int((target_tokens != IGNORE_INDEX).sum().item())

            if compute_kl_stats and get_parallel_state().cp_enabled and is_metrics:
                is_metrics = _sp_allreduce_kl_metrics(
                    is_metrics,
                    get_parallel_state().ulysses_group,
                    is_metric_ops,
                )

            token_diagnostics = self._compute_token_diagnostics(
                hidden_states=hidden_states,
                weight=effective_weight,
                labels=target_tokens,
                topk=diagnostic_topk,
                lm_head=fp8_lm_head,
                lm_head_fp32=self.lm_head_fp32,
                per_token_logprobs=_result.per_token_logprobs,
                include_weight_reference=diagnostic_reference_logits,
                all_hidden_states=diagnostic_all_hidden_states,
                hidden_components=diagnostic_component_captures,
                hidden_sample_count=diagnostic_hidden_sample_count,
                hidden_sample_indices=diagnostic_hidden_sample_indices,
            )
            if token_diagnostics is not None:
                per_token_outputs["token_diagnostics"] = token_diagnostics

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
                loss_reducer=token_sum_reducer,
                metric_reducer=token_sum_reducer,
                lm_head=fp8_lm_head,
            )
            local_loss_sum = _result.loss
            per_token_outputs["logprobs"] = _result.per_token_logprobs
            is_metrics = _result.metrics
            is_metric_ops = _result.metric_ops

            if compute_kl_stats and get_parallel_state().cp_enabled and is_metrics:
                is_metrics = _sp_allreduce_kl_metrics(
                    is_metrics,
                    get_parallel_state().ulysses_group,
                    is_metric_ops,
                )

        elif loss_fn == "opd_loss":
            profile_loss_start = _profile_now() if profile_timings else 0.0
            student_layer_hidden_states = None
            if capture_student_layers and use_opd_selected_layer_hooks:
                missing = [idx for idx in opd_oprd_layer_indices if idx not in (opd_selected_layer_captures or {})]
                if missing:
                    raise ValueError(f"OPRD selected-layer capture did not observe layers: {missing}")
                student_layer_hidden_states = torch.stack(
                    [opd_selected_layer_captures[idx] for idx in opd_oprd_layer_indices],
                    dim=1,
                )
            elif capture_student_layers and outputs.hidden_states:
                # Select the configured decoder-layer subset → [batch, seq, L, d].
                student_layer_hidden_states = torch.stack(
                    [outputs.hidden_states[i] for i in opd_oprd_layer_indices], dim=2
                )
            _result = self._compute_opd_micro_batch_loss(
                hidden_states=hidden_states,
                student_weight=effective_weight,
                micro_batch=micro_batch,
                params=params,
                loss_reducer=token_sum_reducer,
                student_lm_head=getattr(self.model, "lm_head", None),
                student_layer_hidden_states=student_layer_hidden_states,
                opd_oprd_layer_indices=opd_oprd_layer_indices if capture_student_layers else None,
            )
            local_loss_sum = _result.loss
            is_metrics = _result.metrics
            if profile_timings:
                is_metrics = dict(is_metrics or {})
                is_metrics.update(
                    {
                        "opd_profile_model_forward_ms": profile_model_forward_ms,
                        "opd_profile_loss_compute_ms": _profile_elapsed_ms(profile_loss_start),
                    }
                )

        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        return local_loss_sum, per_token_outputs, is_metrics, is_metric_ops, outputs

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
        use_distsignsgd = getattr(self, "_use_distsignsgd", False)

        if loss_fn == "teacher_hidden_cache":
            if compute_backward:
                raise ValueError("teacher_hidden_cache is a forward-only operation")
            try:
                return self._forward_teacher_hidden_cache(micro_batches, params, abort_callback=abort_callback)
            finally:
                if r3_enabled:
                    self._routing_handler.cleanup()

        # Count valid tokens globally
        global_valid_tokens = self._count_global_valid_tokens(micro_batches)
        if compute_backward and use_distsignsgd:
            active_microbatches, active_voter_total = self._count_active_microbatches(micro_batches)
        else:
            active_microbatches, active_voter_total = 0, 0

        if abort_callback and abort_callback():
            raise RuntimeError("Execution aborted by request")

        total_loss = 0.0
        accumulated_loss_metrics = {}
        accumulators = {"logprobs": [], "losses": [], "position_ids": [], "token_diagnostics": []}
        profile_phase_timings = bool(params.get("profile_phase_timings", params.get("opd_profile_timings", False)))
        profile_sync_cuda = bool(params.get("opd_profile_sync_cuda", False))
        forward_compute_time = 0.0
        backward_compute_time = 0.0

        def _profile_phase_now() -> float:
            if profile_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        def _profile_phase_elapsed(start: float) -> float:
            return _profile_phase_now() - start

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
            profile_start = _profile_phase_now() if profile_phase_timings else 0.0
            with self.model_fwd_context:
                local_loss_sum, per_token_outputs, is_metrics, is_metric_ops, outputs = self._compute_micro_batch_loss(
                    micro_batch, loss_fn, params
                )
            if profile_phase_timings:
                forward_compute_time += _profile_phase_elapsed(profile_start)

            logger.debug(
                f"Rank {self.rank}: micro_batch {batch_idx}/{len(micro_batches)} "
                f"loss_sum={local_loss_sum.item():.6f}, local_valid_tokens={local_valid_tokens.item()}, "
                f"global_valid_tokens={global_valid_tokens.item()}"
            )
            # Note: loss is always finite even when local_valid_tokens=0, because
            # train losses use reduction="none" + explicit reducers with
            # clamp(min=1) denominators. No need to replace with zeros_like
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

                if abort_callback and abort_callback():
                    raise RuntimeError("Execution aborted by request")

                # R3: switch to replay_backward so grad ckpt recompute pops same routing
                if r3_enabled:
                    set_replay_stage("replay_backward")

                profile_start = _profile_phase_now() if profile_phase_timings else 0.0
                with self.model_bwd_context:
                    local_loss_sum.backward()
                if profile_phase_timings:
                    backward_compute_time += _profile_phase_elapsed(profile_start)

                # Loss reporting (separately, no grad): compute normalized per-token loss
                with torch.no_grad():
                    # Cast to fp32 before the cross-rank reduction: with ulysses SP,
                    # ranks holding only IGNORE_INDEX tokens may hit early-return
                    # paths with a different dtype than the normal-return path.
                    loss_report = local_loss_sum.detach().float()
                    dist.all_reduce(loss_report, op=dist.ReduceOp.SUM, group=ps.fsdp_group if self.pp_enabled else None)
                    if global_valid_tokens.item() > 0:
                        total_loss += (loss_report / global_valid_tokens).item()
            else:
                # Forward-only: accumulate weighted loss
                if global_valid_tokens.item() > 0:
                    total_loss += local_loss_sum.item() / global_valid_tokens.item()

            # Accumulate loss-specific metrics. RL losses keep the historical
            # `is_*` prefix; OPD metrics are already namespaced as `opd_*`.
            self._accumulate_loss_metrics(accumulated_loss_metrics, is_metrics, loss_fn, is_metric_ops)

            # Cleanup
            del micro_batch, outputs, local_loss_sum

        # Note: gc.collect() + empty_cache() removed from per-step path.
        # They cost ~250ms + ~50ms per step (profiled on Qwen3-8B 8xH100).
        # Cleanup happens at checkpoint save instead.
        # Exception: long-row OPD workloads (think-contract rows, ~2-4k tokens)
        # retain ~1-2GB of cyclic garbage per forward_backward call and OOM a
        # 4-GPU 35B full-FT fit before the optimizer step is reached.
        if os.getenv("XORL_FB_PER_CALL_DEFRAG", "0") == "1":
            gc.collect()
            torch.cuda.empty_cache()
            if not getattr(self, "_fb_defrag_announced", False):
                self._fb_defrag_announced = True
                logger.info("XORL_FB_PER_CALL_DEFRAG active: gc+empty_cache per forward_backward call")

        # R3 cleanup
        if r3_enabled:
            self._routing_handler.cleanup()

        # CP/SP gradient sync (backward only)
        if compute_backward:
            sync_sp_gradients(
                self.model,
                get_parallel_state().sp_grad_sync_group,
                skip_dtensor_grads=use_distsignsgd,
            )
            # Accumulate valid tokens for deferred normalization at optim_step
            self._accumulated_valid_tokens[model_id] = (
                self._accumulated_valid_tokens.get(model_id, 0) + global_valid_tokens.item()
            )
            if use_distsignsgd:
                self._accumulated_active_microbatches[model_id] = (
                    self._accumulated_active_microbatches.get(model_id, 0) + active_microbatches
                )
                self._accumulated_active_voter_total[model_id] = (
                    self._accumulated_active_voter_total.get(model_id, 0) + active_voter_total
                )

        if profile_phase_timings and dist.is_available() and dist.is_initialized():
            phase_times = torch.tensor(
                [forward_compute_time, backward_compute_time],
                dtype=torch.float64,
                device=get_device_type(),
            )
            dist.all_reduce(phase_times, op=dist.ReduceOp.MAX)
            forward_compute_time = float(phase_times[0].item())
            backward_compute_time = float(phase_times[1].item())

        # Build result
        result = {
            "total_loss": total_loss,
            "global_valid_tokens": global_valid_tokens.item(),
        }
        if profile_phase_timings:
            result["forward_compute_time"] = forward_compute_time
            result["backward_compute_time"] = backward_compute_time
            if loss_fn == "opd_loss":
                result["opd_profile_forward_compute_s"] = forward_compute_time
                result["opd_profile_backward_compute_s"] = backward_compute_time

        if accumulators["logprobs"]:
            result["packed_logprobs"] = [t.tolist() for t in accumulators["logprobs"]]
            if accumulators["losses"]:
                result["packed_losses"] = [t.tolist() for t in accumulators["losses"]]
            if accumulators["position_ids"]:
                result["packed_position_ids"] = [t.tolist() for t in accumulators["position_ids"]]
            if accumulators["token_diagnostics"]:
                result["packed_token_diagnostics"] = accumulators["token_diagnostics"]

        # Compute deferred per-sample K3
        if deferred_k3:
            all_per_sample_k3 = []
            for entry in deferred_k3:
                per_sample = self._compute_per_sample_k3(entry["k3_values"], entry["valid_mask"], entry["position_ids"])
                all_per_sample_k3.extend(per_sample)
            result["per_sample_k3"] = all_per_sample_k3

        # All-reduce loss-specific metrics across DP and add to result.
        # Seed the canonical key set on every rank first: ranks that ran no
        # valid-token micro-batch (small batch on a large gang, or ulysses
        # sequence sharding) otherwise carry a different metric-key set and the
        # all-reduce inside _finalize_loss_metrics mismatches in size -> NCCL
        # hang. (Call was dropped in the apanda-dev/glm5 rebase; restored.)
        if loss_fn == "opd_loss":
            self._ensure_opd_loss_metric_accumulators(
                accumulated_loss_metrics,
                include_profile_metrics=profile_phase_timings,
            )
        self._finalize_loss_metrics(accumulated_loss_metrics, result, loss_fn)

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
                loss_fn=make_pp_loss_fn(self.ce_mode),
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

        # Defragment GPU memory at the top of every forward_backward call.
        # After weight-sync + optim_step from the previous step the CUDA
        # allocator can have many small free blocks; without this, CUBLAS
        # handle creation or Triton autotuner workspace allocs can fail.
        gc.collect()
        torch.cuda.empty_cache()

        # Validate single-tenant mode for full-weights training
        self._validate_single_tenant(model_id)

        # Switch to the correct adapter for this model_id
        if self._adapter_manager is not None:
            self._adapter_manager.switch_adapter(model_id)

        # Validate token IDs before processing to catch out-of-vocab errors early
        # This prevents CUDA device-side asserts that can hang the server
        validate_token_ids(micro_batches, self.model.config.vocab_size)

        start_time = time.time()

        # Get return_per_token flag from loss_fn_params (default True for tinker compatibility)
        params = loss_fn_params or {}
        use_distsignsgd = getattr(self, "_use_distsignsgd", False)

        if self.pp_enabled and loss_fn == "opd_loss":
            raise NotImplementedError("opd_loss does not yet support pipeline parallelism")

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
                        if k
                        not in [
                            "labels",
                            "target_tokens",
                            "logprobs",
                            "advantages",
                            "_original_position_ids",
                            "rollout_logprobs",
                        ]
                    }

                    with self.model_fwd_context:
                        outputs = self.model(**model_inputs, use_cache=False, output_hidden_states=False)
                    hidden_states = outputs.last_hidden_state

                    effective_weight = self._get_effective_lm_head_weight()
                    fp8_lm_head = self._get_fp8_lm_head_module(getattr(self.model, "lm_head", None))

                    labels = mb.get("target_tokens", mb.get("labels"))

                    # Compute per-token logprobs using same CE path as training
                    _ref_result = causallm_loss_function(
                        hidden_states=hidden_states,
                        weight=effective_weight,
                        labels=labels,
                        return_per_token=True,
                        ce_mode=self.ce_mode,
                        lm_head_fp32=self.lm_head_fp32,
                        lm_head=fp8_lm_head,
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

        # Reclaim fragmented GPU memory before the main forward-backward pass.
        # Without this, the Triton autotuner's workspace allocations can OOM
        # on memory-tight configs (e.g. EP=32, 16 experts/GPU).
        torch.cuda.empty_cache()

        # R3 (Rollout Routing Replay): Pre-populate routing replay from inference data
        r3_enabled = self._routing_handler.setup(micro_batches, routed_experts, routed_expert_logits)

        # PP path
        if self.pp_enabled:
            global_valid_tokens = self._count_global_valid_tokens(micro_batches)
            if use_distsignsgd:
                active_microbatches, active_voter_total = self._count_active_microbatches(micro_batches)
            else:
                active_microbatches, active_voter_total = 0, 0
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
            if use_distsignsgd:
                self._accumulated_active_microbatches[model_id] = (
                    self._accumulated_active_microbatches.get(model_id, 0) + active_microbatches
                )
                self._accumulated_active_voter_total[model_id] = (
                    self._accumulated_active_voter_total.get(model_id, 0) + active_voter_total
                )
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

        if params.get("profile_clear_gradients_after_backward", False):
            clear_start = time.perf_counter()
            synchronize()
            if self.optimizer is not None:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()
            if self.model is not None:
                self.model.zero_grad(set_to_none=True)
            self._accumulated_valid_tokens[model_id] = 0
            self._accumulated_active_microbatches[model_id] = 0
            self._accumulated_active_voter_total[model_id] = 0
            gc.collect()
            torch.cuda.empty_cache()
            result["opd_profile_clear_gradients_ms"] = (time.perf_counter() - clear_start) * 1000.0

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

    # FSDP2 unshard hooks require version counters; no_grad keeps this path
    # forward-only without disabling those counters.
    @torch.no_grad()
    def forward(
        self,
        micro_batches: List[Dict[str, Any]],
        loss_fn: str = "causallm_loss",
        loss_fn_params: Optional[Dict[str, Any]] = None,
        model_id: str = "default",
        routed_experts: Optional[List] = None,
        routed_expert_logits: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute forward pass only (no gradient computation)."""
        self._check_not_sleeping("forward")

        # Validation/eval requests must run against the same tenant adapter as
        # training requests for that model_id.
        self._validate_single_tenant(model_id)
        if self._adapter_manager is not None:
            self._adapter_manager.switch_adapter(model_id)

        validate_token_ids(micro_batches, self.model.config.vocab_size)

        start_time = time.time()

        if self.pp_enabled and loss_fn == "opd_loss":
            raise NotImplementedError("opd_loss does not yet support pipeline parallelism")

        r3_enabled = self._routing_handler.setup(micro_batches, routed_experts, routed_expert_logits)

        result = self._forward_loop(
            micro_batches,
            loss_fn,
            loss_fn_params,
            compute_backward=False,
            r3_enabled=r3_enabled,
            model_id=model_id,
        )

        if self._adapter_manager is not None:
            result["step"] = self._adapter_manager.get_adapter_state(model_id).global_forward_backward_step
        else:
            result["step"] = self.global_forward_backward_step
        result["forward_time"] = time.time() - start_time
        result["model_id"] = model_id

        logger.info(
            f"forward loss={result['total_loss']:.4f} "
            f"tokens={result.get('global_valid_tokens', 'N/A')} "
            f"model_id={model_id} "
            f"time={result['forward_time']:.2f}s"
        )
        logger.debug(
            f"Rank {self.rank}: forward loss={result['total_loss']:.6f}, "
            f"global_valid_tokens={result.get('global_valid_tokens', 'N/A')}, "
            f"n_micro_batches={len(micro_batches)}, loss_fn={loss_fn}, "
            f"model_id={model_id}, "
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
        sparse_delta_capture: Optional[Dict[str, Any]] = None,
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
            sparse_delta_capture: Optional source sparse-delta capture config.

        Returns:
            Dictionary with optimizer metrics

        Raises:
            RuntimeError: If model is in sleep mode
        """
        self._check_not_sleeping("optim_step")

        start_time = time.time()
        capture_config = dict(sparse_delta_capture or {})
        capture_snapshots: dict[str, torch.Tensor] | None = None
        capture_snapshot_s = 0.0
        if sparse_delta_capture_enabled(capture_config):
            if self._adapter_manager is not None:
                raise RuntimeError("sparse_delta_capture is not supported with per-adapter optimizer state")
            t_capture = time.perf_counter()
            capture_snapshots = snapshot_sparse_delta_tensors(self.model, capture_config)
            capture_snapshot_s = time.perf_counter() - t_capture

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
        accumulated_active_microbatches = getattr(self, "_accumulated_active_microbatches", {}).pop(model_id, 0)
        accumulated_active_voter_total = getattr(self, "_accumulated_active_voter_total", {}).pop(model_id, 0)
        use_distsignsgd = getattr(self, "_use_distsignsgd", False)

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
            self._sync_registered_lora_session_spec(model_id)
            current_step = self._adapter_manager.get_global_step(model_id)
            current_lr = effective_lr

        # Single-adapter path: use shared optimizer on model parameters
        else:
            if use_distsignsgd:
                if accumulated_active_voter_total > 0:
                    scale_model_gradients(
                        self.model,
                        get_distsign_grad_scale_factor(accumulated_active_voter_total),
                    )
            elif accumulated > 0:
                scale_model_gradients(self.model, 1.0 / float(accumulated))

            # Determine learning rate. The client lr is the BASE lr; muon matrix
            # groups (marker use_muon) keep their configured muon_lr/base ratio
            # (see _initialize_optimizer).
            if lr is not None:
                effective_lr = lr
                muon_scale = getattr(self, "_muon_client_lr_scale", None)
                for param_group in self.optimizer.param_groups:
                    if muon_scale is not None and param_group.get("use_muon"):
                        param_group["lr"] = effective_lr * muon_scale
                    else:
                        param_group["lr"] = effective_lr

            ps = get_parallel_state()
            clip_value = get_effective_grad_clip_value(
                clip_value,
                use_distsignsgd=use_distsignsgd,
            )

            grad_norm = clip_gradients(
                self.model,
                clip_value,
                pp_enabled=self.pp_enabled,
                pp_group=ps.pp_group if self.pp_enabled else None,
            )

            # Optimizer step
            self.optimizer.step()
            # Fused/foreach optimizer kernels can still be reading gradients when
            # Python reaches zero_grad/empty_cache. Synchronize before releasing
            # grad storage to avoid allocator reuse while those kernels are live.
            synchronize()
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
        if capture_snapshots is not None:
            capture_result = write_sparse_source_delta_rank(
                model=self.model,
                before=capture_snapshots,
                config=capture_config,
                rank=self.rank,
                world_size=self.world_size,
                model_id=model_id,
                step=current_step,
                snapshot_s=capture_snapshot_s,
            )
            result["sparse_delta_capture"] = capture_result

        logger.info(
            f"optim_step step={current_step} grad_norm={grad_norm:.4f} "
            f"lr={current_lr:.2e} time={result['optim_step_time']:.2f}s"
        )
        logger.debug(
            f"Rank {self.rank}: optim_step step={current_step}, "
            f"grad_norm={grad_norm:.6f}, lr={current_lr:.2e}, "
            f"clip={clip_value}, accumulated_valid_tokens={accumulated}, "
            f"accumulated_active_microbatches={accumulated_active_microbatches}, "
            f"accumulated_active_voter_total={accumulated_active_voter_total}, "
            f"model_id={model_id}, time={result['optim_step_time']:.3f}s"
        )

        synchronize()

        return result

    def _maybe_merge_lora(self) -> None:
        """Periodic LoRA merge at merge_lora_interval."""
        if self._adapter_manager is not None:
            merge_interval = self.lora_config.get("merge_lora_interval", 0)
            if merge_interval > 0:
                raise RuntimeError("merge_lora_interval is not supported with multi-adapter LoRA server training")
            return
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
        self._sync_registered_lora_session_spec(model_id)
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
        if self._adapter_manager is not None:
            raise RuntimeError("sleep is not supported with multi-adapter LoRA server training")

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
        if self._adapter_manager is not None:
            raise RuntimeError("wake_up is not supported with multi-adapter LoRA server training")

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
