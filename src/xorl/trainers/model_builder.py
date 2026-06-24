"""Shared model build + LoRA/QLoRA injection + FSDP parallelization pipeline.

Both the offline Trainer and the server ModelRunner call build_training_model()
so that every feature (QLoRA, TP, DeepEP, …) is supported in both paths
without reimplementation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Set

import torch
import torch.nn as nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.distributed.torch_parallelize import build_parallelize_model as _parallelize
from xorl.lora import freeze_base_parameters
from xorl.lora.utils import inject_lora_into_model, inject_lora_into_model_with_moe
from xorl.models import build_foundation_model
from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules
from xorl.models.layers.rope import set_rope_native
from xorl.models.transformers.deepseek_v3.support import (
    freeze_deepseek_v3_router_parameters,
    validate_deepseek_v3_training_mode,
)
from xorl.models.transformers.glm5.support import validate_glm5_training_mode
from xorl.qlora import (
    detect_prequantized_block_fp8,
    detect_prequantized_nvfp4,
    inject_qlora_into_model,
    maybe_load_and_quantize_moe_qlora,
    maybe_load_prequantized_qlora,
    maybe_quantize_qlora,
)
from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.moe_experts import QLoRAMoeExperts
from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp
from xorl.utils import helper


logger = helper.create_logger(__name__)


@dataclass
class TrainingModelResult:
    """Return value of :func:`build_training_model`."""

    model: nn.Module
    model_config: Any  # HF PretrainedConfig
    pp_enabled: bool
    pp_stages: Optional[List] = None
    model_parts: Optional[List] = None
    has_first_stage: bool = False
    has_last_stage: bool = False
    optimizer_pre_hook_fn: Optional[Callable] = None
    is_prequantized: bool = False
    checkpoint_quant_format: Optional[str] = None
    exclude_modules: Set[str] = field(default_factory=set)


def resolve_training_model_dtype(
    *,
    enable_lora: bool,
    enable_qlora: bool,
    enable_mixed_precision: bool,
    skip_param_upcast: bool = False,
) -> str:
    """Return the foundation-model dtype for the requested training mode.

    Full-weight mixed-precision training keeps parameters in fp32 before FSDP
    wrapping unless ``skip_param_upcast`` is set. LoRA/QLoRA and skip-upcast
    full-weight runs keep checkpoint-native bf16 weights and only LoRA/QLoRA
    upcasts trainable adapter weights to fp32.
    """
    if (enable_lora or enable_qlora or skip_param_upcast) and enable_mixed_precision:
        return "bfloat16"
    if enable_mixed_precision:
        return "float32"
    return "bfloat16"


def should_skip_generic_param_upcast(
    *,
    enable_lora: bool,
    enable_qlora: bool,
    skip_param_upcast: bool = False,
) -> bool:
    """Whether the generic full-model fp32 upcast should be skipped."""
    return enable_lora or enable_qlora or skip_param_upcast


def maybe_upcast_trainable_adapter_params(
    model: nn.Module,
    *,
    enable_lora: bool,
    enable_qlora: bool,
    enable_mixed_precision: bool,
) -> None:
    """Upcast trainable adapter weights to fp32 while leaving the frozen base in bf16."""
    if not enable_mixed_precision or not (enable_lora or enable_qlora):
        return

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    logger.info_rank0("Upcast trainable LoRA params to float32")


def build_training_model(
    *,
    # --- Model ---
    config_path: str,
    weights_path: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_3",
    moe_implementation: Optional[str] = None,
    ep_dispatch: str = "alltoall",
    train_router: bool = False,
    record_routing_weights: bool = True,
    deepep_buffer_size_gb: float = 2.0,
    deepep_num_sms: int = 20,
    deepep_async_combine: bool = False,
    alltoall_combine_hidden_chunk_size: int = 0,
    init_device: str = "meta",
    merge_qkv: bool = True,
    # --- LoRA ---
    enable_lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_target_modules: Optional[List[str]] = None,
    moe_hybrid_shared_lora: bool = False,
    # --- QLoRA ---
    enable_qlora: bool = False,
    quant_format: str = "nvfp4",
    quant_group_size: int = 16,
    qlora_exclude_modules: Optional[List[str]] = None,
    # --- Parallelization ---
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    gradient_checkpointing_method: Optional[str] = None,
    enable_compile: bool = False,
    compile_dynamic_shapes: bool = False,
    basic_modules: Optional[List[str]] = None,
    enable_reentrant: bool = False,
    enable_forward_prefetch: bool = True,
    load_weights_mode: str = "grouped",
    reshard_after_forward: Optional[bool] = None,
    moe_grad_reduce_mode: str = "reduce_scatter",
    fsdp_reduce_dtype: str = "fp32",
    skip_param_upcast: bool = False,
    enable_fp8_training: bool = False,
    enable_qarl: bool = False,
    qarl_quant_cfg: Optional[dict[str, Any] | str] = None,
    qarl_calib_data: Optional[str] = None,
    qarl_calib_size: int = 0,
    qarl_quant_sequence_length: Optional[int] = None,
    qarl_sync_format: str = "fp8",
    qarl_target_modules: Optional[List[str]] = None,
    qarl_exclude_modules: Optional[List[str]] = None,
    fp8_training_num_first_layers_bf16: int = 0,
    fp8_training_num_last_layers_bf16: int = 0,
    fp8_training_allow_blackwell: bool = False,
    fp8_training_blackwell_validation_artifact: Optional[str] = None,
    fp8_training_block_size: int = 128,
    fp8_training_backward: str = "fp8",
    fp8_training_smoothquant_alpha: Optional[float] = None,
    fp8_training_lm_head_smoothquant_alpha: Optional[float] = None,
    fp8_training_activation_amax_scale: float = 1.0,
    fp8_training_weight_amax_scale: float = 1.0,
    fp8_training_correction_mode: str = "none",
    fp8_training_module_overrides: Optional[dict[str, dict[str, Any]]] = None,
    fp8_training_moe_grouped_backend: str = "triton_grouped",
    fp8_training_target_modules: Optional[List[str]] = None,
    fp8_training_exclude_modules: Optional[List[str]] = None,
    fp8_training_allow_bf16_fallback: bool = False,
    pp_schedule: Optional[str] = None,
    # --- Training flags ---
    freeze_router: bool = False,
    # --- SGLang numerical alignment ---
    router_fp32: bool = True,
    lm_head_fp32: bool = True,
    rmsnorm_mode: str = "native",
    activation_native: bool = False,
    rope_native: bool = False,
    attention_cast_bf16: bool = False,
    sparse_mla_enabled: bool = False,
    sparse_mla_backend: str = "auto",
    flash_attention_deterministic: bool = False,
) -> TrainingModelResult:
    """Build, inject LoRA/QLoRA, and parallelize a training model.

    This is the **single** init pipeline shared by both the offline Trainer and
    the server ModelRunner.  The steps mirror the original Trainer lifecycle::

        1. build_foundation_model()
        2. Unfuse QKV (for TP)
        3. QLoRA or LoRA injection
        4. LoRA + mixed-precision: upcast trainable params to fp32
        5. Save optimizer pre-hook
        6. build_parallelize_model()
        7. Deferred QLoRA quantization
        8. Freeze base params (LoRA/QLoRA) + optional router freeze
        9. model.train()

    Returns a :class:`TrainingModelResult` with model, config, PP state, etc.
    """

    # ------------------------------------------------------------------
    # 1. Build foundation model
    # ------------------------------------------------------------------
    logger.info_rank0("Building foundation model …")
    model = build_foundation_model(
        config_path=config_path,
        weights_path=weights_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        moe_implementation=moe_implementation,
        ep_dispatch=ep_dispatch,
        train_router=train_router,
        record_routing_weights=record_routing_weights,
        deepep_buffer_size_gb=deepep_buffer_size_gb,
        deepep_num_sms=deepep_num_sms,
        deepep_async_combine=deepep_async_combine,
        alltoall_combine_hidden_chunk_size=alltoall_combine_hidden_chunk_size,
        router_fp32=router_fp32,
        lm_head_fp32=lm_head_fp32,
        rmsnorm_mode=rmsnorm_mode,
        activation_native=activation_native,
        rope_native=rope_native,
        attention_cast_bf16=attention_cast_bf16,
        sparse_mla_enabled=sparse_mla_enabled,
        sparse_mla_backend=sparse_mla_backend,
        flash_attention_deterministic=flash_attention_deterministic,
        init_device=init_device,
    )

    # Set module-level flags for rope and activation
    if rope_native:
        set_rope_native(True)
        logger.info_rank0("Using native RoPE (flash_attn fused kernel disabled)")
    if activation_native:
        logger.info_rank0("Using native SiLU activation (fused Triton kernel disabled)")
    model_config = model.config
    validate_deepseek_v3_training_mode(
        model_config,
        enable_qlora=enable_qlora,
        freeze_router=freeze_router,
        merge_qkv=merge_qkv,
    )
    validate_glm5_training_mode(
        model_config,
        enable_qlora=enable_qlora,
        freeze_router=freeze_router,
        merge_qkv=merge_qkv,
    )
    helper.print_device_mem_info("VRAM usage after building model")

    # ------------------------------------------------------------------
    # 2. Unfuse QKV if merge_qkv=False (needed for TP)
    # ------------------------------------------------------------------
    if not merge_qkv:
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "unfuse_for_tp"):
                layer.self_attn.unfuse_for_tp()
        logger.info_rank0("Unfused QKV projections (merge_qkv=False)")

    if get_parallel_state().tp_enabled and hasattr(model, "unfuse_for_tp") and not getattr(model, "_unfused_for_tp", False):
        logger.info_rank0("Unfusing projections before FP8 training injection for tensor parallelism")
        model.unfuse_for_tp()

    # ------------------------------------------------------------------
    # 3. FP8 full-weight / QLoRA / LoRA injection
    # ------------------------------------------------------------------
    is_prequantized = False
    checkpoint_quant_format = None
    exclude_modules: Set[str] = set()

    if enable_fp8_training and (enable_lora or enable_qlora):
        raise ValueError("enable_fp8_training is a full-weight mode and cannot be combined with LoRA or QLoRA")
    if enable_qarl and (enable_lora or enable_qlora):
        raise ValueError("enable_qarl is a full-weight mode and cannot be combined with LoRA or QLoRA")
    if enable_qarl and enable_fp8_training:
        raise ValueError("enable_qarl cannot be combined with enable_fp8_training; choose one low-precision train path")
    if enable_qarl and qarl_sync_format != "fp8":
        raise ValueError("Initial QARL supports only qarl_sync_format='fp8'")
    if enable_qarl and qarl_calib_size < 0:
        raise ValueError("qarl_calib_size must be non-negative")
    if enable_qarl and qarl_quant_sequence_length is not None and qarl_quant_sequence_length <= 0:
        raise ValueError("qarl_quant_sequence_length must be positive when set")
    if enable_qarl and qarl_calib_data is None and (qarl_calib_size or qarl_quant_sequence_length is not None):
        raise ValueError("qarl_calib_size and qarl_quant_sequence_length require qarl_calib_data")

    if enable_fp8_training:
        from xorl.fp8_training import (  # noqa: PLC0415
            inject_fp8_training_into_model,
            validate_fp8_blackwell_training_policy,
        )

        validate_fp8_blackwell_training_policy(
            enable_fp8_training=True,
            allow_blackwell=fp8_training_allow_blackwell,
            validation_artifact=fp8_training_blackwell_validation_artifact,
        )

        inject_fp8_training_into_model(
            model,
            target_modules=fp8_training_target_modules,
            exclude_modules=fp8_training_exclude_modules,
            num_first_layers_bf16=fp8_training_num_first_layers_bf16,
            num_last_layers_bf16=fp8_training_num_last_layers_bf16,
            block_size=fp8_training_block_size,
            backward_mode=fp8_training_backward,
            smoothquant_alpha=fp8_training_smoothquant_alpha,
            lm_head_smoothquant_alpha=fp8_training_lm_head_smoothquant_alpha,
            activation_amax_scale=fp8_training_activation_amax_scale,
            weight_amax_scale=fp8_training_weight_amax_scale,
            correction_mode=fp8_training_correction_mode,
            module_overrides=fp8_training_module_overrides,
            allow_bf16_fallback=fp8_training_allow_bf16_fallback,
            moe_grouped_backend=fp8_training_moe_grouped_backend,
        )
        helper.print_device_mem_info("VRAM usage after FP8 training injection")
    elif enable_qarl:
        from xorl.qarl import calibrate_qarl_model, inject_qarl_into_model  # noqa: PLC0415

        inject_qarl_into_model(
            model,
            quant_cfg=qarl_quant_cfg,
            target_modules=qarl_target_modules,
            exclude_modules=qarl_exclude_modules,
        )
        if qarl_calib_data is not None:
            model._qarl_calibration_summary = calibrate_qarl_model(
                model,
                qarl_calib_data,
                calibration_size=qarl_calib_size,
                sequence_length=qarl_quant_sequence_length,
            )
        helper.print_device_mem_info("VRAM usage after QARL fake-quant injection")
    elif enable_qlora:
        is_prequantized, checkpoint_quant_format, exclude_modules = _inject_qlora(
            model,
            weights_path=weights_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            quant_format=quant_format,
            quant_group_size=quant_group_size,
            merge_qkv=merge_qkv,
            qlora_exclude_modules=qlora_exclude_modules,
        )
    elif enable_lora:
        _inject_lora(
            model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            moe_hybrid_shared_lora=moe_hybrid_shared_lora,
        )

    # ------------------------------------------------------------------
    # 4. LoRA + mixed precision: upcast trainable params to fp32
    # ------------------------------------------------------------------
    maybe_upcast_trainable_adapter_params(
        model,
        enable_lora=enable_lora,
        enable_qlora=enable_qlora,
        enable_mixed_precision=enable_mixed_precision,
    )

    # ------------------------------------------------------------------
    # 5. Save optimizer pre-hook (some models register hooks)
    # ------------------------------------------------------------------
    optimizer_pre_hook_fn = getattr(model, "get_optimizer_pre_hook", None)

    # ------------------------------------------------------------------
    # 6. Parallelize (FSDP2 / PP)
    # ------------------------------------------------------------------
    _basic_modules = list(model._no_split_modules) + (basic_modules or [])
    effective_pp_schedule = pp_schedule if get_parallel_state().pp_enabled else None
    build_result = _parallelize(
        model,
        init_device=init_device,
        weights_path=weights_path,
        enable_full_shard=enable_full_shard,
        enable_mixed_precision=enable_mixed_precision,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        gradient_checkpointing_method=gradient_checkpointing_method,
        enable_compile=enable_compile,
        compile_dynamic_shapes=compile_dynamic_shapes,
        basic_modules=_basic_modules,
        enable_reentrant=enable_reentrant,
        enable_forward_prefetch=enable_forward_prefetch,
        load_weights_mode=load_weights_mode,
        pp_schedule=effective_pp_schedule,
        reshard_after_forward=reshard_after_forward,
        moe_grad_reduce_mode=moe_grad_reduce_mode,
        fsdp_reduce_dtype=fsdp_reduce_dtype,
        skip_param_upcast=should_skip_generic_param_upcast(
            enable_lora=enable_lora,
            enable_qlora=enable_qlora,
            skip_param_upcast=skip_param_upcast,
        ),
    )

    pp_enabled = isinstance(build_result, dict)
    pp_stages = None
    model_parts = None
    has_first_stage = False
    has_last_stage = False
    if pp_enabled:
        pp_stages = build_result["stages"]
        model_parts = build_result["model_parts"]
        has_first_stage = build_result["has_first_stage"]
        has_last_stage = build_result["has_last_stage"]
        model = model_parts[0]
    else:
        model = build_result

    # ------------------------------------------------------------------
    # 7. Deferred QLoRA quantization
    # ------------------------------------------------------------------
    if enable_qlora:
        _deferred_qlora_quantize(model, weights_path, load_weights_mode=load_weights_mode)

    # ------------------------------------------------------------------
    # 8. Freeze base params
    # ------------------------------------------------------------------
    if enable_qlora:
        # After QLoRA quantization, freeze everything except LoRA
        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad = False
        helper.print_device_mem_info("VRAM usage after QLoRA quantization")
    elif enable_lora:
        freeze_base_parameters(model)
        logger.info_rank0("Base model parameters frozen, only LoRA parameters trainable")
    else:
        # Full-weights: all params trainable
        for param in model.parameters():
            param.requires_grad = True

    # Optionally freeze MoE router
    if freeze_router:
        router_frozen_count = freeze_deepseek_v3_router_parameters(model)
        if router_frozen_count == 0:
            for name, param in model.named_parameters():
                if ".gate.weight" in name:
                    param.requires_grad = False
                    router_frozen_count += 1
        if router_frozen_count > 0:
            logger.info_rank0(f"Froze {router_frozen_count} MoE router (gate) parameters")

    # ------------------------------------------------------------------
    # 9. model.train()
    # ------------------------------------------------------------------
    model.train()
    logger.info_rank0("Model built and parallelized")

    return TrainingModelResult(
        model=model,
        model_config=model_config,
        pp_enabled=pp_enabled,
        pp_stages=pp_stages,
        model_parts=model_parts,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
        optimizer_pre_hook_fn=optimizer_pre_hook_fn,
        is_prequantized=is_prequantized,
        checkpoint_quant_format=checkpoint_quant_format,
        exclude_modules=exclude_modules,
    )


# ======================================================================
# Internal helpers
# ======================================================================


def _inject_qlora(
    model: nn.Module,
    *,
    weights_path: str,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: Optional[List[str]],
    quant_format: str,
    quant_group_size: int,
    merge_qkv: bool,
    qlora_exclude_modules: Optional[List[str]],
) -> tuple:
    """QLoRA injection with pre-quantized checkpoint detection.

    Returns (is_prequantized, checkpoint_quant_format, exclude_modules).
    """

    is_prequantized = False
    checkpoint_quant_format = None
    exclude_modules: Set[str] = set()

    if detect_prequantized_nvfp4(weights_path):
        is_prequantized = True
        checkpoint_quant_format = "nvfp4"
        logger.info_rank0("Detected pre-quantized NVFP4 checkpoint")
    elif detect_prequantized_block_fp8(weights_path):
        is_prequantized = True
        checkpoint_quant_format = "block_fp8"
        logger.info_rank0("Detected pre-quantized block FP8 checkpoint")

    if qlora_exclude_modules is not None:
        exclude_modules = set(qlora_exclude_modules)
        logger.info_rank0(f"Using user-specified exclude_modules: {exclude_modules}")
    elif is_prequantized:
        exclude_modules = get_prequantized_exclude_modules(weights_path)
        if exclude_modules:
            logger.info_rank0(
                f"Auto-detected {len(exclude_modules)} excluded modules from checkpoint config: {exclude_modules}"
            )

    # NF4 quantizes bf16 weights on-the-fly — no pre-quantized checkpoint needed.
    # Other formats (nvfp4, block_fp8) require a matching pre-quantized checkpoint.
    if quant_format == "nf4":
        if is_prequantized:
            raise ValueError(
                f"NF4 QLoRA expects a bf16 checkpoint but found a pre-quantized "
                f"'{checkpoint_quant_format}' checkpoint at '{weights_path}'. "
                f"Use a standard bf16 checkpoint for NF4 QLoRA."
            )
        logger.info_rank0("NF4 QLoRA: will quantize bf16 weights on-the-fly")
    else:
        if not is_prequantized:
            raise ValueError(
                f"QLoRA requires a pre-quantized checkpoint. "
                f"The checkpoint at '{weights_path}' is bf16. "
                f"Use a pre-quantized checkpoint (e.g., nvidia/Qwen3-8B-NVFP4 for nvfp4, "
                f"or an HF block-FP8 checkpoint for block_fp8)."
            )
        if checkpoint_quant_format != quant_format:
            raise ValueError(
                f"Checkpoint format '{checkpoint_quant_format}' does not match "
                f"target QLoRA format '{quant_format}'. Cross-format loading is "
                f"not supported — use a checkpoint that matches the target format."
            )

    inject_qlora_into_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        quant_format=quant_format,
        quant_group_size=quant_group_size,
        target_modules=lora_target_modules,
        checkpoint_quant_format=checkpoint_quant_format,
        merge_qkv=merge_qkv,
        exclude_modules=exclude_modules,
    )
    if exclude_modules:
        model._qlora_exclude_modules = exclude_modules
    helper.print_device_mem_info("VRAM usage after QLoRA injection")

    return is_prequantized, checkpoint_quant_format, exclude_modules


def _inject_lora(
    model: nn.Module,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_target_modules: Optional[List[str]],
    moe_hybrid_shared_lora: bool = False,
) -> None:
    """Plain LoRA injection (dense + optional MoE-aware)."""
    is_moe_model = getattr(model.config, "num_experts", 0) > 0

    if is_moe_model and moe_hybrid_shared_lora:
        logger.info_rank0(f"MoE-aware LoRA injection (hybrid_shared={moe_hybrid_shared_lora})")
        inject_lora_into_model_with_moe(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            moe_hybrid_shared_lora=moe_hybrid_shared_lora,
        )
    else:
        inject_lora_into_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
        )

    helper.print_device_mem_info("VRAM usage after LoRA injection")


def _deferred_qlora_quantize(
    model: nn.Module,
    weights_path: str,
    load_weights_mode: str = "grouped",
) -> None:
    """After FSDP loads weights, quantize/load weights into QLoRA modules.

    Handles three cases:
    1. Pre-quantized (nvfp4/block_fp8): load packed data from checkpoint
    2. NF4 linear: bf16 weight already loaded by FSDP → quantize in-place
    3. NF4 MoE: load bf16 experts from checkpoint → quantize
    """

    # 1. Pre-quantized linear/MoE loading (nvfp4/block_fp8)
    needs_prequant_linear = any(
        isinstance(m, QLoRALinear) and m._is_prequantized and not m._inline_loaded for m in model.modules()
    )
    needs_prequant_moe = any(
        isinstance(m, QLoRAMoeExperts) and not m._weights_loaded and m._source_quant_format is not None
        for m in model.modules()
    )

    if needs_prequant_linear or needs_prequant_moe:
        logger.info(f"Starting pre-quantized weight loading (mode={load_weights_mode}) …")
        helper.print_device_mem_info("VRAM before pre-quantized loading")
        maybe_load_prequantized_qlora(model, weights_path, load_mode=load_weights_mode)
        logger.info("Done pre-quantized weight loading")

    # 2. NF4 linear: FSDP loaded bf16 into weight param → quantize to NF4
    needs_bf16_quantize = any(isinstance(m, QLoRALinear) and m.weight is not None for m in model.modules())
    if needs_bf16_quantize:
        logger.info("Quantizing bf16 linear weights to NF4 …")
        maybe_quantize_qlora(model)

    # 3. NF4 MoE: load bf16 experts from checkpoint → quantize
    needs_bf16_moe = any(
        isinstance(m, QLoRAMoeExperts)
        and not m._weights_loaded
        and m._source_quant_format is None  # NF4 (no source quant format)
        for m in model.modules()
    )
    if needs_bf16_moe:
        logger.info("Loading and quantizing bf16 MoE expert weights …")
        maybe_load_and_quantize_moe_qlora(
            model,
            weights_path,
            load_mode=load_weights_mode,
        )

    if not (needs_prequant_linear or needs_prequant_moe or needs_bf16_quantize or needs_bf16_moe):
        logger.info("All QLoRA modules loaded inline, skipping deferred disk I/O")

    # Always deregister packed_weight_f32 from FSDP2 (prevent mixed-precision corruption)
    removed = _deregister_qlora_weights_from_fsdp(model, param_names=("packed_weight_f32",))
    torch.cuda.empty_cache()
    if removed > 0:
        logger.info(f"Deregistered {removed} packed_weight_f32 params from FSDP2")
