"""
Server Arguments for Xorl Training API Server.

This module provides ServerArguments - a minimal configuration class
containing only fields relevant to the training server (model loading,
device management, parallelism, etc.), excluding client-side training
parameters like batch size, epochs, and optimizer settings.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import yaml

from xorl.ops.loss import CrossEntropyMode


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

    record_routing_weights: bool = field(
        default=True,
        metadata={
            "help": "Cache routing weights on the forward pass so they can override the "
            "regathered weights during checkpoint recompute. Needed only when the "
            "attention forward is non-deterministic across recompute. Disabling skips the "
            "per-layer pinned CPU allocation + D2H/H2D copies on every step."
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

    alltoall_combine_hidden_chunk_size: int = field(
        default=0,
        metadata={
            "help": "Hidden-dimension chunk size for alltoall EP combine. 0 disables chunking. "
            "Useful for long-context MoE runs where the full combine tensor is a memory peak."
        },
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

    flash_attention_deterministic: bool = field(
        default=False,
        metadata={"help": "Request FlashAttention deterministic backward kernels when available."},
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

    enable_full_determinism: bool = field(default=False, metadata={"help": "Enable full deterministic execution."})

    enable_mixed_precision: bool = field(default=True, metadata={"help": "Enable mixed precision training"})

    enable_fp8_training: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable experimental full-weight block-FP8 compute training. Master parameters remain BF16/FP32 "
                "for optimizer/checkpoint compatibility."
            )
        },
    )
    enable_qarl: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable experimental dense full-weight QARL fake quantization. Initial support uses dynamic "
                "E4M3 fake quantization with full-precision master parameters and STE gradients."
            )
        },
    )
    qarl_quant_cfg: Optional[Union[str, Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": (
                "QARL quantization config or alias. Initial support accepts null, 'FP8_DEFAULT_CFG', 'fp8', "
                "or a dict with format/quant_method=e4m3/fp8_e4m3 plus optional weight/activation booleans."
            )
        },
    )
    qarl_calib_data: Optional[str] = field(
        default=None,
        metadata={"help": "Reserved path for future static QARL calibration data. Dynamic QARL leaves this unset."},
    )
    qarl_calib_size: int = field(
        default=0,
        metadata={"help": "Reserved sample count for future static QARL calibration. Dynamic QARL uses 0."},
    )
    qarl_quant_sequence_length: Optional[int] = field(
        default=None,
        metadata={"help": "Reserved sequence length for future static QARL calibration."},
    )
    qarl_sync_format: Literal["fp8"] = field(
        default="fp8",
        metadata={"help": "Target rollout/export sync format for QARL. Initial support is 'fp8' only."},
    )
    qarl_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional short nn.Linear module names to wrap with QARL fake quantization."},
    )
    qarl_exclude_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional short names, FQNs, or globs to keep out of QARL fake quantization."},
    )
    fp8_cfg: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": (
                "Optional compatibility alias for NeMo-style FP8 configs. Supported values are "
                "{enabled: true, fp8: e4m3, fp8_recipe: blockwise, fp8_param: false}; "
                "TransformerEngine-only recipes are rejected."
            )
        },
    )

    fp8_training_num_first_layers_bf16: int = field(
        default=0,
        metadata={"help": "Number of initial decoder layers to keep in BF16 when FP8 training is enabled."},
    )
    fp8_training_num_last_layers_bf16: int = field(
        default=0,
        metadata={"help": "Number of final decoder layers to keep in BF16 when FP8 training is enabled."},
    )
    fp8_training_allow_blackwell: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow native XoRL FP8 training on Blackwell/GB200. Defaults to false; BF16 training plus FP8 "
                "sync/generation is the default policy until a native FP8 recipe is validated on that hardware."
            )
        },
    )
    fp8_training_blackwell_validation_artifact: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Required path or identifier for the validation artifact when enabling native FP8 training "
                "on Blackwell with fp8_training_allow_blackwell=true."
            )
        },
    )

    fp8_training_block_size: int = field(default=128, metadata={"help": "Block size for FP8 training quantization"})

    fp8_training_backward: Literal["bf16", "fp8"] = field(
        default="fp8",
        metadata={"help": "Backward compute mode for FP8 training linear layers"},
    )

    fp8_training_smoothquant_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional SmoothQuant alpha for dense FP8 training matmuls. When set in [0, 1], activation and "
                "weight columns are dynamically balanced before FP8 quantization."
            )
        },
    )
    fp8_training_lm_head_smoothquant_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional SmoothQuant alpha override for an FP8 lm_head. "
                "If unset, fp8_training_smoothquant_alpha is used."
            )
        },
    )

    fp8_training_activation_amax_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplier applied to dense FP8 activation block absmax before deriving scales. "
                "Values below 1.0 clip activation outliers; values above 1.0 add headroom."
            )
        },
    )

    fp8_training_weight_amax_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplier applied to dense FP8 weight block absmax before deriving scales. "
                "Values below 1.0 clip weight outliers; values above 1.0 add headroom."
            )
        },
    )
    fp8_training_correction_mode: Literal["none", "activation", "activation2", "weight", "first_order", "full"] = field(
        default="none",
        metadata={
            "help": (
                "Optional dense FP8 residual-correction mode. 'none' uses one FP8 GEMM, while activation, "
                "activation2, weight, first_order, and full add extra FP8 GEMMs for quantization residuals."
            )
        },
    )
    fp8_training_module_overrides: Optional[Dict[str, Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help": (
                "Optional FQN/short-name glob pattern overrides for dense FP8Linear recipes. "
                "Supported per-pattern keys are block_size, backward_mode, smoothquant_alpha, "
                "activation_amax_scale, weight_amax_scale, and correction_mode."
            )
        },
    )

    fp8_training_moe_grouped_backend: Literal["triton_grouped", "block_loop", "deep_gemm", "scalar_quack"] = field(
        default="triton_grouped",
        metadata={
            "help": (
                "Grouped GEMM backend for FP8 MoE expert compute. 'triton_grouped' is the default grouped "
                "block-FP8 same-NK and same-MN path; 'block_loop', 'deep_gemm', and 'scalar_quack' are opt-in "
                "alternatives."
            )
        },
    )

    fp8_training_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional short nn.Linear module names to replace for FP8 training"},
    )

    fp8_training_exclude_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Optional short names, FQNs, or glob patterns to keep out of FP8 training compute. If unset, "
                "every matched dense Linear uses FP8 compute, including router gates and output heads."
            )
        },
    )

    fp8_training_allow_bf16_fallback: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow FP8 training layers to fall back to regular F.linear when FP8 kernels cannot run. "
                "Defaults to false so full-weight FP8 server runs fail fast instead of silently using BF16."
            )
        },
    )

    fsdp_reduce_dtype: Literal["fp32", "bf16"] = field(
        default="fp32",
        metadata={
            "help": (
                "FSDP2 gradient reduce-scatter buffer dtype. 'fp32' keeps the existing behavior; "
                "'bf16' lowers reduce-scatter memory and bandwidth."
            )
        },
    )

    enable_gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})

    gradient_checkpointing_method: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Gradient checkpointing strategy. One of: recompute_full_layer, "
                "recompute_before_dispatch, no_recompute. None uses model defaults."
            )
        },
    )

    enable_full_shard: bool = field(default=True, metadata={"help": "Enable full parameter sharding (FSDP)"})

    enable_activation_offload: bool = field(default=False, metadata={"help": "Enable activation CPU offloading"})

    activation_gpu_limit: float = field(
        default=0.0,
        metadata={
            "help": (
                "When enabling activation offload, the number of GB of activations allowed to remain on GPU. "
                "Defaults to 0.0, which offloads all eligible activations."
            )
        },
    )

    activation_offload_prefetch_count: int = field(
        default=0,
        metadata={
            "help": (
                "If >0, opt into the stream-overlapped ActivationOffloader and prefetch this many "
                "CPU-resident activations ahead of backward consumption. Default 0 keeps the legacy "
                "`custom_save_on_cpu` path. Mirrors `TrainingArguments.activation_offload_prefetch_count` "
                "for ModelRunner-driven servers."
            )
        },
    )

    enable_compile: bool = field(default=False, metadata={"help": "Enable torch.compile for model forward pass"})
    compile_dynamic_shapes: bool = field(
        default=False,
        metadata={"help": "Pass dynamic=True to torch.compile. Default keeps torch.compile's standard shape behavior."},
    )

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
        default="grouped",
        metadata={"help": ("Weight loading mode: 'grouped' (default, with rank-0 fallback), 'all_ranks', or 'skip'")},
    )

    init_device: Optional[Literal["cpu", "meta", "cuda"]] = field(
        default="meta", metadata={"help": "Device for model initialization"}
    )

    ce_mode: CrossEntropyMode = field(
        default="compiled",
        metadata={
            "help": "Cross-entropy implementation: 'compiled' (RECOMMENDED, torch.compile), "
            "'quack_linear' (Quack scalar loss; return_per_token uses fused "
            "selected-logprob CE), or 'eager' (baseline, may OOM at 32K)"
        },
    )

    use_shared_prefix: bool = field(
        default=False,
        metadata={
            "help": "Shared-prefix attention: when RL rollouts sample multiple responses per shared "
            "prompt, dedup the prompt in the policy-update forward (compute its KV once). Auto-detects "
            "shared-prefix groups from the packed micro-batch; off => standard attention."
        },
    )

    # ========================================================================
    # Optimizer
    # ========================================================================

    optimizer: Literal["adamw", "anyprecision_adamw", "sgd", "signsgd", "distsignsgd", "muon"] = field(
        default="adamw",
        metadata={
            "help": "Optimizer type. 'signsgd' is a local state-free sign update; "
            "'distsignsgd' signs gradients before FSDP2 reduction; 'muon' uses "
            "Newton-Schulz orthogonalization for 2D+ weight matrices."
        },
    )

    lr: float = field(
        default=1e-5,
        metadata={"help": "Default learning rate for the server's implicit/default training session."},
    )

    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Default weight decay for the server's implicit/default training session."},
    )

    optimizer_dtype: Literal["fp32", "bf16"] = field(
        default="bf16",
        metadata={"help": "Dtype for optimizer states (momentum/variance). 'bf16' halves optimizer memory."},
    )

    cautious_weight_decay: bool = field(
        default=False,
        metadata={
            "help": "Apply Cautious Weight Decay (Chen et al., arXiv:2510.12402): "
            "mask the decoupled decay term by I(u_t * x_t >= 0). With optimizer='adamw' "
            "this routes to AnyPrecisionAdamW with fp32 state (no fused kernel)."
        },
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
        default="gram_newton_schulz",
        metadata={
            "help": "Newton-Schulz backend for Muon. 'gram_newton_schulz' (default) batches across "
            "MoE experts via baddbmm and is ~2x faster on Qwen3.5-style MoE; 'standard_newton_schulz' "
            "uses the PyTorch upstream path for bit-exact equivalence with torch.optim._muon."
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
    muon_grouped_gram_ns_fp32_byte_limit: int = field(
        default=512 * 1024**2,
        metadata={
            "help": "Maximum fp32 scratch bytes per grouped Muon Gram Newton-Schulz batch before chunking. "
            "Lower values reduce peak optimizer scratch memory at the cost of more launches."
        },
    )
    muon_fallback_optimizer: Literal["adamw", "sgd"] = field(
        default="adamw",
        metadata={
            "help": "Optimizer used for parameters excluded from Muon. "
            "'adamw' preserves the default mixed Muon/AdamW behavior; 'sgd' uses state-free fallback updates."
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

    muon_distributed_mode: Literal["shard_local", "full_gradient"] = field(
        default="shard_local",
        metadata={
            "help": "How Muon handles Newton-Schulz on FSDP2/EP-sharded DTensor params. "
            "'shard_local': run NS on each rank's local shard (cheap, approximate). "
            "'full_gradient': all-gather post-momentum update, run NS on the full matrix on "
            "every rank in the param's mesh, slice back to the local shard. Implements the "
            "dense path of DeepSeek V4 §3.5.1."
        },
    )

    moe_grad_reduce_mode: Literal["reduce_scatter", "bf16_a2a_fp32_sum"] = field(
        default="reduce_scatter",
        metadata={
            "help": "Reduce-scatter strategy for MoE expert gradients on the ep_fsdp mesh dim. "
            "'reduce_scatter': default NCCL reduce-scatter. "
            "'bf16_a2a_fp32_sum': stochastic-round FP32 grads to BF16, all-to-all across the "
            "ep_fsdp group, then sum the per-rank chunks locally in FP32. Halves comm volume "
            "while preserving FP32 accumulation. Implements the MoE path of DeepSeek V4 §3.5.1."
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
    load_optimizer: bool = field(
        default=True,
        metadata={
            "help": "When resuming from load_checkpoint_path, also load optimizer state. Set False for a "
            "weights-only resume (optimizer re-initialized fresh) — required to resume across a different "
            "expert_parallel_size, since legacy optimizer state may not be reshardable."
        },
    )

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

    sample_packing_strategy: str = field(
        default="sequential",
        metadata={
            "help": (
                "Bin-packing strategy for training micro-batches: 'sequential' (greedy first-fit, "
                "default, unchanged behavior), 'best_fit' (best-fit-decreasing; fuller rows / fewer "
                "rows for the same pack length), or 'balanced_dp' (longest-processing-time partition "
                "into N=k*dp_size balanced rows so the dispatcher needs zero dummy batches). "
                "'balanced_dp' only improves throughput when the batch is large enough to keep rows "
                "near the GEMM knee (~16k tokens/rank)."
            )
        },
    )

    sample_packing_on_oversized: str = field(
        default="error",
        metadata={
            "help": (
                "How to handle a sample longer than sample_packing_sequence_len: 'error' (default; "
                "fail loud so a misconfigured pack length never silently drops training data), "
                "'skip' (drop with a warning; legacy behavior), or 'truncate' (clip the sample and "
                "its token-aligned fields to the pack length)."
            )
        },
    )

    # ========================================================================
    # LoRA Configuration
    # ========================================================================

    enable_lora: bool = field(default=False, metadata={"help": "Enable LoRA adapters for training"})

    lora_rank: int = field(default=32, metadata={"help": "LoRA rank (r parameter)"})

    max_lora_rank: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum LoRA rank allocated in the server model substrate. Defaults to lora_rank. "
            "Per-session ranks must be <= max_lora_rank."
        },
    )

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

    lora_export_format: str = field(
        default="peft",
        metadata={
            "help": "On-disk layout for MoE LoRA export. 'peft' (default) writes per-expert keys in PEFT orientation. 'sglang_shared_outer' writes stacked 3D tensors under experts.w{1,2,3} in SGLang's shared_outer format (requires moe_hybrid_shared_lora=True)."
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
    adapter_state_load_mode: Literal["all_ranks", "rank0_broadcast"] = field(
        default="all_ranks",
        metadata={
            "help": "How to restore multi-adapter LoRA checkpoints. 'all_ranks': each rank loads adapter state locally. "
            "'rank0_broadcast': rank 0 loads once and broadcasts weights, metadata, and optimizer state."
        },
    )

    # ========================================================================
    # MoE Training Configuration
    # ========================================================================

    freeze_router: bool = field(default=True, metadata={"help": "Freeze MoE router weights during training"})

    # ========================================================================
    # Inference Weight Sync Configuration
    # ========================================================================

    sync_inference_method: Literal["nccl_broadcast", "p2p", "sparse_delta"] = field(
        default="nccl_broadcast",
        metadata={
            "help": "Method for syncing weights to inference endpoints: "
            "'nccl_broadcast' (rank-0 broadcast via SGLang update_weights_from_distributed); "
            "'p2p' (RDMA one-sided writes via Mooncake TransferEngine into SGLang's "
            "registered param memory; requires --enable-rdma-weight-updates on the SGLang side); "
            "'sparse_delta' (experimental packed sparse files via SGLang update_weights_from_sparse_delta)"
        },
    )
    receiver_kv_cache_dtype: Optional[Literal["auto", "fp8", "fp8_e4m3"]] = field(
        default=None,
        metadata={
            "help": (
                "Expected KV-cache dtype for registered SGLang receivers. XoRL does not launch SGLang; "
                "set --kv-cache-dtype on the receiver and use this field to validate /server_info metadata. "
                "Use 'fp8' or 'fp8_e4m3' to require an FP8 KV cache."
            )
        },
    )

    @property
    def optimizer_kwargs(self) -> Dict[str, Any]:
        """Collect optimizer-specific kwargs for build_optimizer."""
        kwargs: Dict[str, Any] = {}
        if self.optimizer == "muon":
            kwargs["muon_lr"] = self.muon_lr
            kwargs["muon_momentum"] = self.muon_momentum
            kwargs["muon_nesterov"] = self.muon_nesterov
            kwargs["muon_ns_steps"] = self.muon_ns_steps
            kwargs["muon_adjust_lr_fn"] = self.muon_adjust_lr_fn
            kwargs["muon_ns_algorithm"] = self.muon_ns_algorithm
            kwargs["muon_ns_use_quack_kernels"] = self.muon_ns_use_quack_kernels
            kwargs["muon_gram_ns_num_restarts"] = self.muon_gram_ns_num_restarts
            kwargs["muon_gram_ns_restart_iterations"] = self.muon_gram_ns_restart_iterations
            if self.optimizer_dtype == "bf16":
                kwargs["muon_momentum_dtype"] = torch.bfloat16
            if self.muon_grad_dtype == "bf16":
                kwargs["muon_grad_dtype"] = torch.bfloat16
            elif self.muon_grad_dtype == "fp32":
                kwargs["muon_grad_dtype"] = torch.float32
            if self.muon_update_dtype == "bf16":
                kwargs["muon_update_dtype"] = torch.bfloat16
            elif self.muon_update_dtype == "fp32":
                kwargs["muon_update_dtype"] = torch.float32
            if self.muon_force_momentum_path:
                kwargs["muon_force_momentum_path"] = True
            if self.muon_distributed_mode != "shard_local":
                kwargs["muon_distributed_mode"] = self.muon_distributed_mode
        return kwargs

    def __post_init__(self):
        """Validate and set defaults."""
        from xorl.fp8_training.config_compat import normalize_fp8_training_config  # noqa: PLC0415
        from xorl.qarl import normalize_qarl_quant_cfg, qarl_unsupported_scope_reason  # noqa: PLC0415
        from xorl.server.orchestrator.packing import ON_OVERSIZED_MODES, PACKING_STRATEGIES  # noqa: PLC0415

        if self.sample_packing_strategy not in PACKING_STRATEGIES:
            raise ValueError(
                f"sample_packing_strategy must be one of {PACKING_STRATEGIES}, got {self.sample_packing_strategy!r}"
            )
        if self.sample_packing_on_oversized not in ON_OVERSIZED_MODES:
            raise ValueError(
                f"sample_packing_on_oversized must be one of {ON_OVERSIZED_MODES}, "
                f"got {self.sample_packing_on_oversized!r}"
            )

        normalized_fp8_config = normalize_fp8_training_config(vars(self), context="server.train")
        self.enable_fp8_training = bool(normalized_fp8_config.get("enable_fp8_training", self.enable_fp8_training))
        if self.enable_qarl and self.enable_fp8_training:
            raise ValueError("enable_qarl cannot be combined with enable_fp8_training; choose one low-precision train path")
        if self.enable_fp8_training and (self.enable_lora or self.enable_qlora):
            raise ValueError("enable_fp8_training is a full-weight mode and cannot be combined with LoRA or QLoRA")
        if self.enable_qarl and (self.enable_lora or self.enable_qlora):
            raise ValueError("enable_qarl is a full-weight mode and cannot be combined with LoRA or QLoRA")
        if self.enable_qarl:
            if self.qarl_calib_size < 0:
                raise ValueError("qarl_calib_size must be non-negative")
            if self.qarl_quant_sequence_length is not None and self.qarl_quant_sequence_length <= 0:
                raise ValueError("qarl_quant_sequence_length must be positive when set")
            if self.qarl_calib_data is None and (self.qarl_calib_size or self.qarl_quant_sequence_length is not None):
                raise ValueError("qarl_calib_size and qarl_quant_sequence_length require qarl_calib_data")
            self.qarl_quant_cfg = normalize_qarl_quant_cfg(self.qarl_quant_cfg)
            unsupported_reason = qarl_unsupported_scope_reason(
                model_config=self.foundation,
                config_path=self.config_path or self.model_path,
                module_names=[
                    *(self.qarl_target_modules or []),
                    *(self.qarl_exclude_modules or []),
                ],
            )
            if unsupported_reason is not None:
                raise ValueError(unsupported_reason)

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

        if self.adapter_state_load_mode not in {"all_ranks", "rank0_broadcast"}:
            raise ValueError(
                "adapter_state_load_mode must be 'all_ranks' or 'rank0_broadcast', "
                f"got {self.adapter_state_load_mode!r}"
            )
        if self.enable_lora and self.pipeline_parallel_size > 1:
            raise ValueError(
                "pipeline_parallel_size > 1 is not supported with multi-adapter LoRA server training. "
                "Adapter coordination currently assumes identical local LoRA layouts on every rank."
            )
        if self.enable_lora and self.merge_lora_interval > 0:
            raise ValueError("merge_lora_interval is not supported with multi-adapter LoRA server training")
        if self.max_lora_rank is None:
            self.max_lora_rank = self.lora_rank
        if self.max_lora_rank < self.lora_rank:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be >= lora_rank ({self.lora_rank}) for the default session"
            )

        if self.load_weights_mode not in {"grouped", "all_ranks", "skip"}:
            raise ValueError(
                f"Unsupported load_weights_mode={self.load_weights_mode!r}. Expected one of: grouped, all_ranks, skip."
            )

        if self.load_weights_mode == "skip" and not self.load_checkpoint_path:
            raise ValueError(
                "load_weights_mode='skip' skips HF weight loading and relies on "
                "load_checkpoint_path to materialize parameters from a DCP checkpoint. "
                "Set load_checkpoint_path or choose a different load_weights_mode."
            )
        if self.receiver_kv_cache_dtype is not None:
            receiver_kv_cache_dtype = str(self.receiver_kv_cache_dtype).strip().lower()
            if receiver_kv_cache_dtype in {"", "none", "null"}:
                self.receiver_kv_cache_dtype = None
            elif receiver_kv_cache_dtype not in {"auto", "fp8", "fp8_e4m3"}:
                raise ValueError(
                    "receiver_kv_cache_dtype must be one of: auto, fp8, fp8_e4m3; "
                    f"got {self.receiver_kv_cache_dtype!r}"
                )
            else:
                self.receiver_kv_cache_dtype = receiver_kv_cache_dtype

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
                "record_routing_weights": self.record_routing_weights,
                "deepep_buffer_size_gb": self.deepep_buffer_size_gb,
                "deepep_num_sms": self.deepep_num_sms,
                "deepep_async_combine": self.deepep_async_combine,
                "alltoall_combine_hidden_chunk_size": self.alltoall_combine_hidden_chunk_size,
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
                "flash_attention_deterministic": self.flash_attention_deterministic,
            },
            "train": {
                "output_dir": self.output_dir,
                "seed": self.seed,
                "enable_full_determinism": self.enable_full_determinism,
                "data_parallel_mode": self.data_parallel_mode,
                "ulysses_parallel_size": self.ulysses_parallel_size,
                "expert_parallel_size": self.expert_parallel_size,
                "data_parallel_replicate_size": self.data_parallel_replicate_size,
                "data_parallel_shard_size": self.data_parallel_shard_size,
                "tensor_parallel_size": self.tensor_parallel_size,
                "ringattn_parallel_size": self.ringattn_parallel_size,
                "cp_fsdp_mode": self.cp_fsdp_mode,
                "enable_mixed_precision": self.enable_mixed_precision,
                "enable_fp8_training": self.enable_fp8_training,
                "enable_qarl": self.enable_qarl,
                "qarl_quant_cfg": self.qarl_quant_cfg,
                "qarl_calib_data": self.qarl_calib_data,
                "qarl_calib_size": self.qarl_calib_size,
                "qarl_quant_sequence_length": self.qarl_quant_sequence_length,
                "qarl_sync_format": self.qarl_sync_format,
                "qarl_target_modules": self.qarl_target_modules,
                "qarl_exclude_modules": self.qarl_exclude_modules,
                "fp8_cfg": self.fp8_cfg,
                "fp8_training_num_first_layers_bf16": self.fp8_training_num_first_layers_bf16,
                "fp8_training_num_last_layers_bf16": self.fp8_training_num_last_layers_bf16,
                "fp8_training_allow_blackwell": self.fp8_training_allow_blackwell,
                "fp8_training_blackwell_validation_artifact": self.fp8_training_blackwell_validation_artifact,
                "fp8_training_block_size": self.fp8_training_block_size,
                "fp8_training_backward": self.fp8_training_backward,
                "fp8_training_smoothquant_alpha": self.fp8_training_smoothquant_alpha,
                "fp8_training_lm_head_smoothquant_alpha": self.fp8_training_lm_head_smoothquant_alpha,
                "fp8_training_activation_amax_scale": self.fp8_training_activation_amax_scale,
                "fp8_training_weight_amax_scale": self.fp8_training_weight_amax_scale,
                "fp8_training_correction_mode": self.fp8_training_correction_mode,
                "fp8_training_module_overrides": self.fp8_training_module_overrides,
                "fp8_training_moe_grouped_backend": self.fp8_training_moe_grouped_backend,
                "fp8_training_target_modules": self.fp8_training_target_modules,
                "fp8_training_exclude_modules": self.fp8_training_exclude_modules,
                "fp8_training_allow_bf16_fallback": self.fp8_training_allow_bf16_fallback,
                "fsdp_reduce_dtype": self.fsdp_reduce_dtype,
                "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
                "gradient_checkpointing_method": self.gradient_checkpointing_method,
                "enable_full_shard": self.enable_full_shard,
                "enable_activation_offload": self.enable_activation_offload,
                "activation_gpu_limit": self.activation_gpu_limit,
                "activation_offload_prefetch_count": self.activation_offload_prefetch_count,
                "enable_compile": self.enable_compile,
                "compile_dynamic_shapes": self.compile_dynamic_shapes,
                "enable_reentrant": self.enable_reentrant,
                "enable_forward_prefetch": self.enable_forward_prefetch,
                "reshard_after_forward": self.reshard_after_forward,
                "load_weights_mode": self.load_weights_mode,
                "init_device": self.init_device,
                "ce_mode": self.ce_mode,
                "use_shared_prefix": self.use_shared_prefix,
                "optimizer": self.optimizer,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "optimizer_dtype": self.optimizer_dtype,
                "cautious_weight_decay": self.cautious_weight_decay,
                "muon_lr": self.muon_lr,
                "muon_momentum": self.muon_momentum,
                "muon_nesterov": self.muon_nesterov,
                "muon_ns_steps": self.muon_ns_steps,
                "muon_adjust_lr_fn": self.muon_adjust_lr_fn,
                "muon_ns_algorithm": self.muon_ns_algorithm,
                "muon_ns_use_quack_kernels": self.muon_ns_use_quack_kernels,
                "muon_gram_ns_num_restarts": self.muon_gram_ns_num_restarts,
                "muon_gram_ns_restart_iterations": self.muon_gram_ns_restart_iterations,
                "muon_grouped_gram_ns_fp32_byte_limit": self.muon_grouped_gram_ns_fp32_byte_limit,
                "muon_fallback_optimizer": self.muon_fallback_optimizer,
                "muon_grad_dtype": self.muon_grad_dtype,
                "muon_update_dtype": self.muon_update_dtype,
                "muon_force_momentum_path": self.muon_force_momentum_path,
                "muon_distributed_mode": self.muon_distributed_mode,
                "moe_grad_reduce_mode": self.moe_grad_reduce_mode,
                "optimizer_kwargs": self.optimizer_kwargs,
                "load_checkpoint_path": self.load_checkpoint_path,
                "load_optimizer": self.load_optimizer,
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
                "receiver_kv_cache_dtype": self.receiver_kv_cache_dtype,
            },
            "data": {
                # Empty data section - data comes from client at runtime
            },
            "lora": {
                "enable_lora": self.enable_lora,
                "lora_rank": self.lora_rank,
                "max_lora_rank": self.max_lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_target_modules": self.lora_target_modules,
                "moe_hybrid_shared_lora": self.moe_hybrid_shared_lora,
                "lora_export_format": self.lora_export_format,
                "enable_qlora": self.enable_qlora,
                "quant_format": self.quant_format,
                "quant_group_size": self.quant_group_size,
                "exclude_modules": self.qlora_exclude_modules,
                "merge_lora_interval": self.merge_lora_interval,
                "reset_optimizer_on_merge": self.reset_optimizer_on_merge,
                "adapter_state_load_mode": self.adapter_state_load_mode,
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
