"""Shared model build + LoRA/QLoRA injection + FSDP parallelization pipeline.

Both the offline Trainer and the server ModelRunner call build_training_model()
so that every feature (QLoRA, TP, DeepEP, …) is supported in both paths
without reimplementation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Set

import torch
import torch.nn as nn

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


def build_training_model(
    *,
    # --- Model ---
    config_path: str,
    weights_path: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_3",
    moe_implementation: Optional[str] = None,
    ep_dispatch: str = "alltoall",
    deepep_buffer_size_gb: float = 2.0,
    deepep_num_sms: int = 20,
    deepep_async_combine: bool = False,
    init_device: str = "meta",
    merge_qkv: bool = True,
    # --- LoRA ---
    enable_lora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 16,
    lora_target_modules: Optional[List[str]] = None,
    moe_shared_lora: bool = False,
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
    enable_compile: bool = False,
    basic_modules: Optional[List[str]] = None,
    enable_reentrant: bool = False,
    enable_forward_prefetch: bool = True,
    load_weights_mode: str = "broadcast",
    reshard_after_forward: Optional[bool] = None,
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
    from xorl.distributed.torch_parallelize import build_parallelize_model as _parallelize
    from xorl.models import build_foundation_model

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
        deepep_buffer_size_gb=deepep_buffer_size_gb,
        deepep_num_sms=deepep_num_sms,
        deepep_async_combine=deepep_async_combine,
        router_fp32=router_fp32,
        lm_head_fp32=lm_head_fp32,
        rmsnorm_mode=rmsnorm_mode,
        activation_native=activation_native,
        rope_native=rope_native,
        attention_cast_bf16=attention_cast_bf16,
        init_device=init_device,
    )

    # Set module-level flags for rope and activation
    if rope_native:
        from xorl.models.layers.rope import set_rope_native

        set_rope_native(True)
        logger.info_rank0("Using native RoPE (flash_attn fused kernel disabled)")
    if activation_native:
        logger.info_rank0("Using native SiLU activation (fused Triton kernel disabled)")
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    # ------------------------------------------------------------------
    # 2. Unfuse QKV if merge_qkv=False (needed for TP)
    # ------------------------------------------------------------------
    if not merge_qkv:
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "unfuse_for_tp"):
                layer.self_attn.unfuse_for_tp()
        logger.info_rank0("Unfused QKV projections (merge_qkv=False)")

    # ------------------------------------------------------------------
    # 3. QLoRA / LoRA injection
    # ------------------------------------------------------------------
    is_prequantized = False
    checkpoint_quant_format = None
    exclude_modules: Set[str] = set()

    if enable_qlora:
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
            moe_shared_lora=moe_shared_lora,
            moe_hybrid_shared_lora=moe_hybrid_shared_lora,
        )

    # ------------------------------------------------------------------
    # 4. LoRA + mixed precision: upcast trainable params to fp32
    # ------------------------------------------------------------------
    if (enable_lora or enable_qlora) and enable_mixed_precision:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        logger.info_rank0("Upcast trainable LoRA params to float32")

    # ------------------------------------------------------------------
    # 5. Save optimizer pre-hook (some models register hooks)
    # ------------------------------------------------------------------
    optimizer_pre_hook_fn = getattr(model, "get_optimizer_pre_hook", None)

    # ------------------------------------------------------------------
    # 6. Parallelize (FSDP2 / PP)
    # ------------------------------------------------------------------
    _basic_modules = list(model._no_split_modules) + (basic_modules or [])
    build_result = _parallelize(
        model,
        init_device=init_device,
        weights_path=weights_path,
        enable_full_shard=enable_full_shard,
        enable_mixed_precision=enable_mixed_precision,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        enable_compile=enable_compile,
        basic_modules=_basic_modules,
        enable_reentrant=enable_reentrant,
        enable_forward_prefetch=enable_forward_prefetch,
        load_weights_mode=load_weights_mode,
        pp_schedule=pp_schedule,
        reshard_after_forward=reshard_after_forward,
        skip_param_upcast=enable_qlora,
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
        from xorl.lora import freeze_base_parameters

        freeze_base_parameters(model)
        logger.info_rank0("Base model parameters frozen, only LoRA parameters trainable")
    else:
        # Full-weights: all params trainable
        for param in model.parameters():
            param.requires_grad = True

    # Optionally freeze MoE router
    if freeze_router:
        router_frozen_count = 0
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
    from xorl.qlora import (
        detect_prequantized_block_fp8,
        detect_prequantized_nvfp4,
        inject_qlora_into_model,
    )

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
        from xorl.models.checkpoint_handlers.buffers import get_prequantized_exclude_modules

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
    moe_shared_lora: bool = False,
    moe_hybrid_shared_lora: bool = False,
) -> None:
    """Plain LoRA injection (dense + optional MoE-aware)."""
    is_moe_model = getattr(model.config, "num_experts", 0) > 0

    if is_moe_model and (moe_shared_lora or moe_hybrid_shared_lora):
        from xorl.lora.utils import inject_lora_into_model_with_moe

        logger.info_rank0(
            f"MoE-aware LoRA injection (shared={moe_shared_lora}, hybrid_shared={moe_hybrid_shared_lora})"
        )
        inject_lora_into_model_with_moe(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            moe_shared_lora=moe_shared_lora,
            moe_hybrid_shared_lora=moe_hybrid_shared_lora,
        )
    else:
        from xorl.lora.utils import inject_lora_into_model

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
    load_weights_mode: str = "broadcast",
) -> None:
    """After FSDP loads weights, quantize/load weights into QLoRA modules.

    Handles three cases:
    1. Pre-quantized (nvfp4/block_fp8): load packed data from checkpoint
    2. NF4 linear: bf16 weight already loaded by FSDP → quantize in-place
    3. NF4 MoE: load bf16 experts from checkpoint → quantize
    """
    from xorl.qlora.modules.linear import QLoRALinear
    from xorl.qlora.modules.moe_experts import QLoRAMoeExperts
    from xorl.qlora.utils import _deregister_qlora_weights_from_fsdp

    # 1. Pre-quantized linear/MoE loading (nvfp4/block_fp8)
    needs_prequant_linear = any(
        isinstance(m, QLoRALinear) and m._is_prequantized and not m._inline_loaded for m in model.modules()
    )
    needs_prequant_moe = any(
        isinstance(m, QLoRAMoeExperts) and not m._weights_loaded and m._source_quant_format is not None
        for m in model.modules()
    )

    if needs_prequant_linear or needs_prequant_moe:
        from xorl.qlora import maybe_load_prequantized_qlora

        logger.info(f"Starting pre-quantized weight loading (mode={load_weights_mode}) …")
        helper.print_device_mem_info("VRAM before pre-quantized loading")
        maybe_load_prequantized_qlora(model, weights_path, load_mode=load_weights_mode)
        logger.info("Done pre-quantized weight loading")

    # 2. NF4 linear: FSDP loaded bf16 into weight param → quantize to NF4
    needs_bf16_quantize = any(isinstance(m, QLoRALinear) and m.weight is not None for m in model.modules())
    if needs_bf16_quantize:
        from xorl.qlora import maybe_quantize_qlora

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
        from xorl.qlora import maybe_load_and_quantize_moe_qlora

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
