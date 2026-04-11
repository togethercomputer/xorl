"""
Server Arguments for Xorl Training API Server.

This module provides ServerArguments - a minimal configuration class
containing only fields relevant to the training server (model loading,
device management, parallelism, etc.), excluding client-side training
parameters like batch size, epochs, and optimizer settings.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import yaml


@dataclass
class ServerArguments:
    """
    Server arguments for the Training API.

    Contains only server-side configuration:
    - Model loading and initialization
    - Attention and optimization implementations
    - Parallelism settings (FSDP, Ulysses, etc.)
    - Device and memory management
    - Checkpoint loading
    - Worker configuration

    Does NOT include client-side training parameters like:
    - micro_batch_size, gradient_accumulation_steps
    - num_train_epochs, save_steps
    - optimizer, lr, weight_decay
    - wandb settings

    Usage:
        from xorl.server import ServerArguments
        from xorl.arguments import parse_args

        args = parse_args(ServerArguments)
        model_path = args.model_path
        worker_addr = args.worker.bind_address
    """

    # ========================================================================
    # Model Configuration
    # ========================================================================

    model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pre-trained model (HF Hub or local path)"}
    )

    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model identifier for validation (e.g., 'Qwen/Qwen3-32B'). Defaults to model_path if not specified."
        },
    )

    config_path: Optional[str] = field(default=None, metadata={"help": "Path to model config. Defaults to model_path"})

    tokenizer_path: Optional[str] = field(default=None, metadata={"help": "Path to tokenizer. Defaults to config_path"})

    attn_implementation: Optional[Literal["eager", "sdpa", "native", "flash_attention_3", "flash_attention_4"]] = field(
        default="flash_attention_3",
        metadata={
            "help": "Attention implementation. 'native': PyTorch SDPA+cuDNN (no deps, Hopper+Blackwell). "
            "'flash_attention_3': FA3 (Hopper). 'flash_attention_4': FA4 CUTE (Hopper+Blackwell)."
        },
    )

    moe_implementation: Optional[Literal[None, "eager", "triton", "native", "quack"]] = field(
        default=None,
        metadata={
            "help": "MoE implementation. 'triton' uses Triton group GEMM kernels, 'native' uses torch._grouped_mm, 'quack' uses quack kernels."
        },
    )

    ep_dispatch: str = field(
        default="alltoall",
        metadata={"help": "EP dispatch strategy: 'alltoall' (default) or 'deepep' (NVLink-optimized)."},
    )

    train_router: bool = field(
        default=False,
        metadata={
            "help": "Whether expert computation gradients should flow through routing weights. "
            "Disabled by default and must remain False when ep_dispatch='deepep'."
        },
    )

    deepep_buffer_size_gb: float = field(
        default=2.0, metadata={"help": "DeepEP buffer size in GB (effective when ep_dispatch='deepep')."}
    )

    deepep_num_sms: int = field(
        default=20, metadata={"help": "Number of SMs for DeepEP communication kernels (must be even, default 20)."}
    )

    deepep_async_combine: bool = field(
        default=False, metadata={"help": "Enable async combine for DeepEP (overlap combine with next layer's compute)."}
    )

    # SGLang numerical alignment flags
    router_fp32: bool = field(
        default=True, metadata={"help": "Upcast MoE router gate computation to float32 for numerical stability."}
    )

    lm_head_fp32: bool = field(
        default=True, metadata={"help": "Upcast LM head logits computation to float32 for numerical stability."}
    )

    rmsnorm_mode: Literal["eager", "native", "compile"] = field(
        default="native",
        metadata={
            "help": "RMSNorm implementation mode. 'native' uses torch.nn.functional.rms_norm "
            "and is the default. 'compile' runs that native path through torch.compile. "
            "'eager' uses the plain eager implementation."
        },
    )

    activation_native: bool = field(
        default=False, metadata={"help": "Use native SiLU instead of fused Triton kernel for SGLang alignment."}
    )

    rope_native: bool = field(
        default=False, metadata={"help": "Use naive RoPE implementation instead of flash_attn fused kernel."}
    )

    attention_cast_bf16: bool = field(
        default=False, metadata={"help": "Explicitly cast Q/K to bfloat16 after RoPE for SGLang alignment."}
    )

    # Multimodal model configuration
    foundation: Dict[str, str] = field(default_factory=dict, metadata={"help": "Foundation model extra config"})

    encoders: Dict[Literal["image", "video", "audio"], Dict[str, str]] = field(
        default_factory=dict, metadata={"help": "Multimodal encoder config"}
    )

    # ========================================================================
    # Parallelism Configuration
    # ========================================================================

    data_parallel_mode: Optional[Literal["none", "ddp", "fsdp2"]] = field(
        default="fsdp2",
        metadata={"help": "Data parallelism mode. Use 'none' for single GPU without any parallelization."},
    )

    ulysses_parallel_size: int = field(default=1, metadata={"help": "Ulysses sequence parallelism size"})

    expert_parallel_size: int = field(default=1, metadata={"help": "Expert parallelism size for MoE models"})

    data_parallel_replicate_size: int = field(default=1, metadata={"help": "Data parallel replicate size (HSDP)"})

    data_parallel_shard_size: int = field(default=1, metadata={"help": "Data parallel shard size (FSDP)"})

    pipeline_parallel_size: int = field(default=1, metadata={"help": "Pipeline parallelism size. 1 = disabled."})

    pipeline_parallel_schedule: str = field(
        default="1F1B", metadata={"help": "Pipeline parallelism schedule: '1F1B' or 'GPipe'."}
    )
    pp_variable_seq_lengths: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, negotiate the per-step maximum sequence length across PP ranks "
                "via all-reduce and pad only to that dynamic max, avoiding waste from a "
                "static sample_packing_sequence_len.  Each unique seq_len gets its own "
                "cached PipelineStage so P2P buffers always match the actual shape."
            )
        },
    )

    tensor_parallel_size: int = field(default=1, metadata={"help": "Tensor parallelism size"})

    ringattn_parallel_size: int = field(default=1, metadata={"help": "Ring attention parallel size"})

    cp_fsdp_mode: str = field(
        default="all", metadata={"help": "Sequence parallel FSDP mode: 'all', 'ulysses_only', 'ring_only', 'none'"}
    )

    basic_modules: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "Basic modules to shard in FSDP"}
    )

    merge_qkv: bool = field(
        default=True, metadata={"help": "Whether to merge QKV projections. Set False for tensor parallelism."}
    )

    # ========================================================================
    # Memory & Performance
    # ========================================================================

    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})

    enable_mixed_precision: bool = field(default=True, metadata={"help": "Enable mixed precision training"})

    enable_gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})

    enable_full_shard: bool = field(default=True, metadata={"help": "Enable full parameter sharding (FSDP)"})

    enable_activation_offload: bool = field(default=False, metadata={"help": "Enable activation CPU offloading"})

    enable_compile: bool = field(default=False, metadata={"help": "Enable torch.compile for model forward pass"})

    enable_reentrant: bool = field(
        default=False, metadata={"help": "Use reentrant gradient checkpointing (default: non-reentrant)"}
    )

    enable_forward_prefetch: bool = field(
        default=False, metadata={"help": "Enable FSDP forward prefetch for overlapping compute and communication"}
    )

    reshard_after_forward: bool = field(
        default=True, metadata={"help": "Reshard parameters after forward pass in FSDP2"}
    )

    load_weights_mode: str = field(
        default="auto", metadata={"help": "Weight loading mode: 'auto', 'safetensors', 'dcp'"}
    )

    init_device: Optional[Literal["cpu", "meta", "cuda"]] = field(
        default="meta", metadata={"help": "Device for model initialization"}
    )

    ce_mode: Literal["eager", "compiled"] = field(
        default="compiled",
        metadata={
            "help": "Cross-entropy implementation: 'compiled' (RECOMMENDED, torch.compile) or 'eager' (baseline, may OOM at 32K)"
        },
    )

    # ========================================================================
    # Optimizer
    # ========================================================================

    optimizer: Literal["adamw", "anyprecision_adamw", "sgd", "signsgd", "muon"] = field(
        default="adamw",
        metadata={
            "help": "Optimizer type. 'signsgd' is a state-free sign update; 'muon' uses "
            "Newton-Schulz orthogonalization for 2D+ weight matrices."
        },
    )

    optimizer_dtype: Literal["fp32", "bf16"] = field(
        default="bf16",
        metadata={"help": "Dtype for optimizer states (momentum/variance). 'bf16' halves optimizer memory."},
    )

    muon_lr: float = field(
        default=0.02,
        metadata={
            "help": "Learning rate for Muon parameter groups (2D+ weight matrices). Only used when optimizer='muon'."
        },
    )

    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Momentum coefficient for Muon parameter groups."},
    )

    muon_nesterov: bool = field(
        default=True,
        metadata={"help": "Use Nesterov momentum for Muon parameter groups."},
    )

    muon_ns_steps: int = field(
        default=5,
        metadata={"help": "Number of Newton-Schulz iterations for Muon optimizer."},
    )

    muon_adjust_lr_fn: Optional[str] = field(
        default=None,
        metadata={
            "help": "LR adjustment for Muon. 'original': scale by sqrt(max(1,A/B)). "
            "'match_rms_adamw': scale by 0.2*sqrt(max(A,B)) so Muon can reuse AdamW LR/WD."
        },
    )

    muon_ns_algorithm: Literal["standard_newton_schulz", "gram_newton_schulz"] = field(
        default="standard_newton_schulz",
        metadata={
            "help": "Newton-Schulz backend for Muon. 'standard_newton_schulz' keeps the PyTorch Muon path; "
            "'gram_newton_schulz' uses Dao-AILab's Gram Newton-Schulz formulation."
        },
    )

    muon_ns_use_quack_kernels: bool = field(
        default=True,
        metadata={
            "help": "Allow Muon Gram Newton-Schulz to use Quack symmetric GEMM kernels on supported Hopper/Blackwell GPUs. "
            "Falls back to torch matmuls when unavailable."
        },
    )

    muon_gram_ns_num_restarts: int = field(
        default=1,
        metadata={
            "help": "Number of restart locations to autotune for Muon's Gram Newton-Schulz backend when explicit "
            "muon_gram_ns_restart_iterations are not provided."
        },
    )

    muon_gram_ns_restart_iterations: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Explicit restart iteration indices for Muon's Gram Newton-Schulz backend. "
            "A value of 2 means restart after the second iteration."
        },
    )
    muon_grad_dtype: Optional[Literal["fp32", "bf16"]] = field(
        default=None,
        metadata={
            "help": "Optional dtype cast for the gradient tensor used inside Muon. "
            "Use this to force the Muon optimizer path to fp32 or bf16 independently of momentum state dtype."
        },
    )
    muon_update_dtype: Optional[Literal["fp32", "bf16"]] = field(
        default=None,
        metadata={
            "help": "Optional dtype cast for the transient Muon update tensor passed into Newton-Schulz. "
            "Use this to decouple compute dtype from gradient and momentum-buffer storage dtype."
        },
    )
    muon_force_momentum_path: bool = field(
        default=False,
        metadata={
            "help": "Force Muon to build the update through the momentum-buffer path even when muon_momentum=0. "
            "Intended for debugging and ablations."
        },
    )

    # ========================================================================
    # Checkpointing & Output
    # ========================================================================

    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "Output directory for checkpoints, sampler weights, and logs (must be on shared filesystem for multi-node)"
        },
    )

    storage_limit: str = field(
        default="10TB",
        metadata={
            "help": "Maximum disk usage for output_dir (e.g., '1GB', '500MB', '10GB'). Save operations will fail with StorageLimitError when limit is exceeded. Default: 10TB."
        },
    )

    idle_session_timeout: float = field(
        default=7200.0,
        metadata={
            "help": "Idle session timeout in seconds. Sessions inactive for this duration will be automatically cleaned up. Default: 7200 (2 hours)."
        },
    )

    load_checkpoint_path: str = field(default="", metadata={"help": "Path to checkpoint to load"})

    ckpt_manager: Optional[Literal["torch", "dcp"]] = field(default="dcp", metadata={"help": "Checkpoint manager type"})

    # ========================================================================
    # Logging
    # ========================================================================

    log_level: str = field(default="INFO", metadata={"help": "Logging level (DEBUG, INFO, WARNING, ERROR)"})

    enable_self_test: bool = field(default=False, metadata={"help": "Enable self-test after model initialization"})

    skip_initial_checkpoint: bool = field(
        default=False, metadata={"help": "Skip saving initial checkpoint (000000) on startup"}
    )

    log_gradient_norms: bool = field(
        default=True, metadata={"help": "Log gradient norms by layer type after backward pass"}
    )

    log_router_stats: bool = field(default=True, metadata={"help": "Log MoE router token distribution statistics"})

    # ========================================================================
    # Worker Configuration
    # ========================================================================

    worker_bind_host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "Host for worker ZMQ ROUTER socket to bind. Use '0.0.0.0' for multi-node (accepts connections from any interface)."
        },
    )

    worker_bind_port: int = field(
        default=5556, metadata={"help": "Port for worker ZMQ ROUTER socket to bind (rank 0 worker)"}
    )

    engine_connect_host: Optional[str] = field(
        default=None,
        metadata={
            "help": "Host for Engine to connect to rank 0 worker. If None, auto-discovered (localhost for single-node, file-based for multi-node)."
        },
    )

    worker_bind_address: str = field(
        default="auto",
        metadata={"help": "ZMQ ROUTER socket address to bind (rank 0 worker). 'auto' picks a free port."},
    )

    worker_connection_timeout: float = field(
        default=120.0,
        metadata={"help": "Timeout in seconds for worker-executor connection. Increased for multi-node scenarios."},
    )

    worker_max_retries: int = field(default=3, metadata={"help": "Maximum number of retries for failed operations"})

    # ========================================================================
    # Data Processing Configuration
    # ========================================================================

    sample_packing_sequence_len: int = field(
        default=32000, metadata={"help": "Maximum sequence length for sample packing (default: 32000)"}
    )

    enable_packing: bool = field(
        default=True, metadata={"help": "Enable sample packing to combine multiple samples into one sequence"}
    )

    # ========================================================================
    # LoRA Configuration
    # ========================================================================

    enable_lora: bool = field(default=False, metadata={"help": "Enable LoRA adapters for training"})

    lora_rank: int = field(default=32, metadata={"help": "LoRA rank (r parameter)"})

    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling parameter"})

    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names to apply LoRA to (e.g., ['q_proj', 'k_proj', 'v_proj', 'o_proj']). If None, uses default based on model architecture."
        },
    )

    moe_hybrid_shared_lora: bool = field(
        default=False,
        metadata={
            "help": "Enable hybrid shared LoRA for MoE: share lora_A for gate/up_proj, lora_B for down_proj across experts"
        },
    )

    # ========================================================================
    # QLoRA Configuration
    # ========================================================================

    enable_qlora: bool = field(
        default=False, metadata={"help": "Enable QLoRA (quantized LoRA) for memory-efficient training"}
    )

    quant_format: str = field(
        default="nvfp4", metadata={"help": "Quantization format for QLoRA: 'nvfp4', 'block_fp8', or 'nf4'"}
    )

    quant_group_size: int = field(default=16, metadata={"help": "Quantization group size for QLoRA"})

    qlora_exclude_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Modules to exclude from QLoRA quantization (e.g., ['lm_head'])"}
    )

    merge_lora_interval: int = field(default=0, metadata={"help": "Merge LoRA weights every N steps (0 = never merge)"})
    reset_optimizer_on_merge: bool = field(
        default=False, metadata={"help": "ReLoRA-style partial optimizer reset after each LoRA merge"}
    )

    # ========================================================================
    # MoE Training Configuration
    # ========================================================================

    freeze_router: bool = field(default=True, metadata={"help": "Freeze MoE router weights during training"})

    # ========================================================================
    # Inference Weight Sync Configuration
    # ========================================================================

    sync_inference_method: Literal["nccl_broadcast"] = field(
        default="nccl_broadcast",
        metadata={
            "help": "Method for syncing weights to inference endpoints: "
            "'nccl_broadcast' (rank-0 broadcast via SGLang update_weights_from_distributed)"
        },
    )

    def __post_init__(self):
        """Validate and set defaults."""
        # Set default paths
        if self.config_path is None:
            self.config_path = self.model_path

        if self.tokenizer_path is None:
            self.tokenizer_path = self.config_path

        # Validate model_path
        if self.model_path is None:
            raise ValueError("model_path is required for server configuration")

        # Build worker_bind_address from host and port if explicitly set (not "auto")
        if self.worker_bind_address != "auto":
            if not self.worker_bind_address.startswith("tcp://"):
                raise ValueError("worker_bind_address must be 'auto' or a valid ZMQ TCP address (tcp://host:port)")
        else:
            # "auto" mode: launcher will auto-find a free port
            # If worker_bind_host/port are explicitly configured via YAML,
            # the launcher can still use them via engine_connect_host + worker_bind_port
            pass

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert ServerArguments to the config dict format expected by ModelRunner.

        Returns:
            Dict with 'model', 'train', 'data', and 'lora' sections
        """
        config = {
            "model": {
                "model_path": self.model_path,
                "model_name": self.model_name,
                "config_path": self.config_path,
                "tokenizer_path": self.tokenizer_path,
                "attn_implementation": self.attn_implementation,
                "moe_implementation": self.moe_implementation,
                "ep_dispatch": self.ep_dispatch,
                "train_router": self.train_router,
                "deepep_buffer_size_gb": self.deepep_buffer_size_gb,
                "deepep_num_sms": self.deepep_num_sms,
                "deepep_async_combine": self.deepep_async_combine,
                "foundation": self.foundation,
                "encoders": self.encoders,
                "basic_modules": self.basic_modules,
                "merge_qkv": self.merge_qkv,
                "router_fp32": self.router_fp32,
                "lm_head_fp32": self.lm_head_fp32,
                "rmsnorm_mode": self.rmsnorm_mode,
                "activation_native": self.activation_native,
                "rope_native": self.rope_native,
                "attention_cast_bf16": self.attention_cast_bf16,
            },
            "train": {
                "output_dir": self.output_dir,
                "seed": self.seed,
                "data_parallel_mode": self.data_parallel_mode,
                "ulysses_parallel_size": self.ulysses_parallel_size,
                "expert_parallel_size": self.expert_parallel_size,
                "data_parallel_replicate_size": self.data_parallel_replicate_size,
                "data_parallel_shard_size": self.data_parallel_shard_size,
                "tensor_parallel_size": self.tensor_parallel_size,
                "ringattn_parallel_size": self.ringattn_parallel_size,
                "cp_fsdp_mode": self.cp_fsdp_mode,
                "enable_mixed_precision": self.enable_mixed_precision,
                "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
                "enable_full_shard": self.enable_full_shard,
                "enable_activation_offload": self.enable_activation_offload,
                "enable_compile": self.enable_compile,
                "enable_reentrant": self.enable_reentrant,
                "enable_forward_prefetch": self.enable_forward_prefetch,
                "reshard_after_forward": self.reshard_after_forward,
                "load_weights_mode": self.load_weights_mode,
                "init_device": self.init_device,
                "ce_mode": self.ce_mode,
                "optimizer": self.optimizer,
                "optimizer_dtype": self.optimizer_dtype,
                "muon_lr": self.muon_lr,
                "muon_momentum": self.muon_momentum,
                "muon_nesterov": self.muon_nesterov,
                "muon_ns_steps": self.muon_ns_steps,
                "muon_adjust_lr_fn": self.muon_adjust_lr_fn,
                "muon_ns_algorithm": self.muon_ns_algorithm,
                "muon_ns_use_quack_kernels": self.muon_ns_use_quack_kernels,
                "muon_gram_ns_num_restarts": self.muon_gram_ns_num_restarts,
                "muon_gram_ns_restart_iterations": self.muon_gram_ns_restart_iterations,
                "muon_grad_dtype": self.muon_grad_dtype,
                "muon_update_dtype": self.muon_update_dtype,
                "muon_force_momentum_path": self.muon_force_momentum_path,
                "load_checkpoint_path": self.load_checkpoint_path,
                "ckpt_manager": self.ckpt_manager,
                "enable_self_test": self.enable_self_test,
                "skip_initial_checkpoint": self.skip_initial_checkpoint,
                "log_gradient_norms": self.log_gradient_norms,
                "log_router_stats": self.log_router_stats,
                "freeze_router": self.freeze_router,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "pipeline_parallel_schedule": self.pipeline_parallel_schedule,
                "pp_variable_seq_lengths": self.pp_variable_seq_lengths,
                "log_level": self.log_level,
                "sync_inference_method": self.sync_inference_method,
            },
            "data": {
                # Empty data section - data comes from client at runtime
            },
            "lora": {
                "enable_lora": self.enable_lora,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_target_modules": self.lora_target_modules,
                "moe_hybrid_shared_lora": self.moe_hybrid_shared_lora,
                "enable_qlora": self.enable_qlora,
                "quant_format": self.quant_format,
                "quant_group_size": self.quant_group_size,
                "exclude_modules": self.qlora_exclude_modules,
                "merge_lora_interval": self.merge_lora_interval,
                "reset_optimizer_on_merge": self.reset_optimizer_on_merge,
            },
        }
        return config

    def get_world_size(self) -> int:
        """
        Calculate world size from parallelism configuration.

        Note: EP (Expert Parallel) is NOT included in world_size calculation.
        EP creates a separate 2D mesh: world_size = ep_size * ep_fsdp_size
        where ep_fsdp_size contains all other parallelism dimensions.

        Returns:
            Required world size (number of GPUs)
        """
        return (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.ringattn_parallel_size
            * self.ulysses_parallel_size
            * self.data_parallel_replicate_size
            * self.data_parallel_shard_size
        )

    def get_total_gpus(self) -> int:
        """
        Calculate total number of GPUs required for the parallelism configuration.

        EP and the main parallelism mesh (DP/Ulysses/etc.) share the same GPUs but
        organize them differently:
        - Main mesh: DP x Ulysses
        - EP mesh: EP x ep_fsdp_size (where ep_fsdp_size = world_size / ep_size)

        The total GPUs needed is the MAX of EP and the main mesh dimensions,
        rounded up to be divisible by EP (if EP > 1).

        Returns:
            Total number of GPUs needed
        """
        main_mesh_size = self.get_world_size()
        ep_size = self.expert_parallel_size

        if ep_size <= 1:
            return main_mesh_size

        # Need at least ep_size GPUs, and world_size must be divisible by ep_size
        # Also need to accommodate the main mesh dimensions
        total = max(ep_size, main_mesh_size)

        # Round up to nearest multiple of ep_size
        if total % ep_size != 0:
            total = ((total // ep_size) + 1) * ep_size

        return total

    def get_ep_fsdp_size(self) -> int:
        """
        Calculate ep_fsdp_size (the size of each expert parallel group).

        For non-EP models (ep_size=1), this equals world_size.
        For EP models, ep_fsdp_size = world_size / ep_size.

        Returns:
            Size of the ep_fsdp dimension
        """
        world_size = self.get_world_size()
        if self.expert_parallel_size > 1:
            assert world_size % self.expert_parallel_size == 0, (
                f"world_size ({world_size}) must be divisible by expert_parallel_size ({self.expert_parallel_size})"
            )
            return world_size // self.expert_parallel_size
        return world_size

    def get_dp_size(self) -> int:
        """
        Calculate data parallel size (auto-calculated from other dimensions).

        IMPORTANT: EP (Expert Parallel) is NOT included in this calculation!
        EP creates a separate 2D mesh and doesn't participate in the main mesh.

        dp_size is the remaining parallelism after accounting for ulysses.
        It's then split into dp_replicate_size and dp_shard_size.

        Returns:
            Data parallel size
        """
        world_size = self.get_world_size()
        if world_size % self.ulysses_parallel_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by ulysses_parallel_size ({self.ulysses_parallel_size})"
            )
        return world_size // self.ulysses_parallel_size


def parse_server_args() -> ServerArguments:
    """
    Parse ServerArguments from command line and YAML config.

    This function handles the flat ServerArguments structure by creating
    a temporary nested structure that parse_args can handle, then flattening
    the result.

    Returns:
        ServerArguments with all fields populated from YAML and CLI
    """

    # Read YAML directly to get flat structure
    config_path = None
    for i, arg in enumerate(sys.argv):
        if not arg.startswith("--") and i > 0 and not sys.argv[i - 1].startswith("--"):
            config_path = arg
            break

    if not config_path:
        raise ValueError("Config file path required as first positional argument")

    # Load YAML
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        raise ValueError(f"Empty config file: {config_path}")

    # Process CLI overrides (--bind_address, --connection_timeout, etc.)
    # Supports both --key=value and --key value formats
    cli_overrides = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--") and not arg.startswith("---"):
            key_part = arg[2:]  # Remove '--'

            # Check for --key=value format
            if "=" in key_part:
                key, value = key_part.split("=", 1)
                # Convert dotted notation (worker.bind_address) to flat (worker_bind_address)
                key = key.replace(".", "_")
                # Try to parse as number or boolean
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                    value = float(value) if "." in value else int(value)
                cli_overrides[key] = value
                i += 1
            elif i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                # --key value format
                # Convert dotted notation (worker.bind_address) to flat (worker_bind_address)
                key = key_part.replace(".", "_")
                value = sys.argv[i + 1]
                # Try to parse as number or boolean
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                    value = float(value) if "." in value else int(value)
                cli_overrides[key] = value
                i += 2
            else:
                i += 1
        else:
            i += 1

    # Apply CLI overrides
    config_data.update(cli_overrides)

    # Validate config keys against ServerArguments fields
    valid_fields = {f.name for f in __import__("dataclasses").fields(ServerArguments)}
    unknown_fields = set(config_data.keys()) - valid_fields
    if unknown_fields:
        raise ValueError(
            f"Unrecognized config fields: {sorted(unknown_fields)}. Check your config file for typos or removed fields."
        )

    # Create ServerArguments
    server_args = ServerArguments(**config_data)

    return server_args
