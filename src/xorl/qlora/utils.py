"""
QLoRA utility functions.

Functions for injecting QLoRA into models and managing quantized checkpoints.
"""

import json
import logging
import os
from typing import Collection, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.moe_experts import QLoRAMoeExperts

logger = logging.getLogger(__name__)

# Merge source mapping for fused modules in pre-quantized checkpoints.
# When the model has a fused module (e.g., qkv_proj) but the checkpoint has
# separate projections (q_proj, k_proj, v_proj), this maps fused → source names.
_PREQUANT_MERGE_SOURCES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _get_submodule(model: nn.Module, target: str) -> Tuple[nn.Module, str]:
    """Get parent module and attribute name for a dot-separated target path."""
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_linear_modules(
    model: nn.Module,
    target_modules: List[str],
    exclude_types: tuple = (),
) -> List[str]:
    """Find all nn.Linear module paths matching target names, excluding certain types."""
    matched_paths = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if exclude_types and isinstance(module, exclude_types):
            continue
        module_name = name.split(".")[-1] if name else ""
        if module_name in target_modules:
            matched_paths.append(name)
    return matched_paths


def inject_qlora_into_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 16,
    quant_format: str = "nvfp4",
    quant_group_size: int = 16,
    target_modules: Optional[List[str]] = None,
    is_prequantized: bool = False,
    checkpoint_quant_format: Optional[str] = None,
    merge_qkv: bool = True,
    exclude_modules: Optional[Collection[str]] = None,
    enable_hadamard: bool = False,
    hadamard_block_size: int = 256,
    stochastic_rounding: bool = False,
    clip_ratio: Optional[float] = 1.0,
    enable_aqn: bool = False,
    aqn_alpha: float = 1.0,
) -> nn.Module:
    """
    Inject QLoRA into a model: quantize base weights + add trainable LoRA.

    Replaces nn.Linear with QLoRALinear which stores weights in quantized
    format (~4x weight memory reduction for nvfp4, ~2x for block_fp8).
    Only LoRA params are trainable.

    When ``is_prequantized=True``, the checkpoint already contains quantized weights
    (e.g., NVFP4 modelopt format or block FP8). QLoRALinear modules are created
    without weight parameters — pre-quantized data is loaded later via
    ``maybe_load_prequantized_qlora()``.

    Cross-format conversion: if ``checkpoint_quant_format`` differs from
    ``quant_format``, pre-quantized weights are dequantized to bf16 and
    re-quantized in the target format during loading.

    Args:
        model: Model to inject QLoRA into
        r: LoRA rank
        lora_alpha: LoRA alpha for scaling
        quant_format: Quantization format: "nvfp4" or "block_fp8"
        quant_group_size: Block size for quantization (16 for nvfp4, 128 for block_fp8)
        target_modules: Module names to target. Default: all linear projections.
        is_prequantized: If True, checkpoint has pre-quantized weights (skip quantization).
        checkpoint_quant_format: Format of pre-quantized checkpoint ("nvfp4" or "block_fp8").
            When different from ``quant_format``, triggers cross-format conversion.
        merge_qkv: If True (default), merge q/k/v into fused qkv_proj. If False,
            each projection gets its own QLoRALinear with independent LoRA parameters.
            Requires the model's QKV projections to be unfused before calling this.
        enable_hadamard: If True, apply Hadamard rotation before quantizing to spread outliers.
        hadamard_block_size: Block size for Hadamard rotation (must be power of 2, default 256).
        stochastic_rounding: If True, use stochastic rounding in FP4/FP8 encoding.
        clip_ratio: Scale clipping ratio (0.0-1.0). None = auto-search MSE-optimal ratio.
            Default 1.0 (disabled).
        enable_aqn: If True, add Adaptive Quantization Noise during training forward passes.
        aqn_alpha: Scale factor for AQN noise magnitude (default 1.0).
    """
    if quant_format not in ("nvfp4", "block_fp8"):
        raise ValueError(
            f"Supported QLoRA formats: 'nvfp4', 'block_fp8'. Got quant_format={quant_format!r}"
        )
    if target_modules is None:
        if is_prequantized:
            # Pre-quantized checkpoints: target the fused module names in the model.
            # Separate HF projections (q/k/v, gate/up) are merged during loading.
            target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if is_prequantized:
        # Ensure quant_group_size matches target format
        if quant_format == "block_fp8":
            quant_group_size = 128
        elif quant_format == "nvfp4":
            quant_group_size = 16

    target_paths = _find_linear_modules(model, target_modules, exclude_types=(QLoRALinear,))

    # Filter out modules excluded from quantization in pre-quantized checkpoint.
    # exclude_modules contains short names (e.g., "lm_head", "gate") — match
    # against the last component of each FQN.
    if exclude_modules:
        exclude_modules = set(exclude_modules)  # normalize List/Set/etc. to Set
        all_short_names = {p.split(".")[-1] for p in target_paths}
        before = len(target_paths)
        target_paths = [
            p for p in target_paths
            if p.split(".")[-1] not in exclude_modules
        ]
        excluded = before - len(target_paths)
        if excluded > 0:
            logger.info(
                f"Excluded {excluded} modules from QLoRA injection "
                f"(not quantized in checkpoint): {exclude_modules}"
            )
        # Warn about exclude_modules names that didn't match any target module
        unmatched = exclude_modules - all_short_names
        if unmatched:
            logger.warning(
                f"exclude_modules contains names not matching any target module: {unmatched}. "
                f"Available target short names: {all_short_names}"
            )

    if not target_paths:
        raise ValueError(
            f"No nn.Linear modules found matching target_modules={target_modules}. "
            f"Available: {[n.split('.')[-1] for n, _ in model.named_modules() if n][:20]}..."
        )

    logger.info(
        f"Injecting QLoRA into {len(target_paths)} modules "
        f"(r={r}, alpha={lora_alpha}, {quant_format}, group_size={quant_group_size}"
        f"{', prequantized=True' if is_prequantized else ''})"
    )

    for target_path in target_paths:
        parent, attr_name = _get_submodule(model, target_path)
        original_module = getattr(parent, attr_name)

        if is_prequantized:
            # Pre-quantized: create QLoRALinear directly (no weight parameter).
            # Weights are loaded later via maybe_load_prequantized_qlora().
            qlora_module = QLoRALinear(
                in_features=original_module.in_features,
                out_features=original_module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                quant_format=quant_format,
                quant_group_size=quant_group_size,
                bias=original_module.bias is not None,
                device=original_module.weight.device,
                enable_hadamard=enable_hadamard,
                hadamard_block_size=hadamard_block_size,
                stochastic_rounding=stochastic_rounding,
                clip_ratio=clip_ratio,
                enable_aqn=enable_aqn,
                aqn_alpha=aqn_alpha,
            )
            qlora_module._is_prequantized = True
            qlora_module._source_quant_format = checkpoint_quant_format

            # Determine merge sources and HF source FQN
            merge_sources = _PREQUANT_MERGE_SOURCES.get(attr_name)
            qlora_module._merge_sources = merge_sources
            if merge_sources is not None:
                # Fused module (qkv_proj, gate_up_proj): source FQN is parent path
                qlora_module._source_fqn = target_path.rsplit(".", 1)[0]
            else:
                # Direct module (o_proj, down_proj): source FQN is the full path
                qlora_module._source_fqn = target_path

            # Tell checkpoint loading to silently skip weight key
            qlora_module._qlora_expected_skip_keys = {"weight"}
        else:
            qlora_module = QLoRALinear.from_module(
                original_module,
                r=r,
                lora_alpha=lora_alpha,
                quant_format=quant_format,
                quant_group_size=quant_group_size,
                enable_hadamard=enable_hadamard,
                hadamard_block_size=hadamard_block_size,
                stochastic_rounding=stochastic_rounding,
                clip_ratio=clip_ratio,
                enable_aqn=enable_aqn,
                aqn_alpha=aqn_alpha,
            )

        setattr(parent, attr_name, qlora_module)

    logger.info(f"Replaced {len(target_paths)} modules with QLoRALinear")

    # Also inject QLoRA into MoE experts (replace module.experts, keep original MoeBlock)
    if not any(t in ("gate_proj", "up_proj", "down_proj") for t in target_modules):
        return model

    from xorl.distributed.parallel_state import get_parallel_state
    try:
        parallel_state = get_parallel_state()
        ep_size = parallel_state.ep_size if parallel_state.ep_enabled else 1
        ep_rank = parallel_state.ep_rank if parallel_state.ep_enabled else 0
    except Exception:
        ep_size = 1
        ep_rank = 0

    moe_count = 0
    for name, module in list(model.named_modules()):
        # Match MoE blocks: modules with .gate (nn.Linear) and .experts with 3D weights
        if not (hasattr(module, "gate") and hasattr(module, "experts")):
            continue
        experts = module.experts
        has_3d_weights = (
            hasattr(experts, "gate_proj") and isinstance(experts.gate_proj, nn.Parameter)
            and hasattr(experts, "up_proj") and isinstance(experts.up_proj, nn.Parameter)
            and hasattr(experts, "down_proj") and isinstance(experts.down_proj, nn.Parameter)
            and experts.gate_proj.dim() == 3
            and not isinstance(experts, QLoRAMoeExperts)
        )
        if not has_3d_weights:
            continue

        num_experts = experts.gate_proj.shape[0]
        num_local_experts = num_experts // ep_size
        expert_offset = ep_rank * num_local_experts

        qlora_experts = QLoRAMoeExperts.from_module(
            experts,
            r=r,
            lora_alpha=lora_alpha,
            quant_format=quant_format,
            quant_group_size=quant_group_size,
            num_local_experts=num_local_experts,
            expert_offset=expert_offset,
            hybrid_shared=True,
        )
        # Source FQN and checkpoint format for loading
        experts_fqn = name + ".experts" if not name.endswith(".experts") else name
        qlora_experts._source_fqn = experts_fqn
        qlora_experts._source_quant_format = checkpoint_quant_format

        # Replace only the experts module, keep the original MoeBlock
        module.experts = qlora_experts
        moe_count += 1

    if moe_count > 0:
        logger.info(
            f"Replaced {moe_count} MoE experts with QLoRAMoeExperts "
            f"(ep_size={ep_size}, num_local_experts={num_local_experts})"
        )

    return model


def save_qlora_checkpoint(
    model: nn.Module,
    save_path: str,
) -> str:
    """
    Save QLoRA checkpoint: quantized base weights + LoRA parameters.

    Args:
        model: Model with QLoRALinear layers
        save_path: Directory to save checkpoint
    """
    os.makedirs(save_path, exist_ok=True)

    quantized_state = {}
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, QLoRALinear):
            # Quantized base weights
            q_data = module.get_quantized_state_dict()
            for key, val in q_data.items():
                quantized_state[f"{name}.{key}"] = val.cpu().clone()
            # LoRA weights
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
            # Bias
            if module.bias is not None:
                lora_state[f"{name}.bias"] = module.bias.detach().cpu().clone()

    if quantized_state:
        save_file(quantized_state, os.path.join(save_path, "quantized_weights.safetensors"))
    if lora_state:
        save_file(lora_state, os.path.join(save_path, "lora_weights.safetensors"))

    logger.info(f"Saved QLoRA checkpoint to {save_path}")
    return save_path


def _deregister_qlora_weights_from_fsdp(
    model: nn.Module,
    param_names: tuple = ("weight",),
) -> int:
    """Remove QLoRALinear parameters from all FSDP2 param groups.

    After quantization, the ``weight`` parameter storage is freed — we must remove
    it from FSDP2's tracking to prevent all-gather on empty storage.

    ``packed_weight_f32`` must ALSO be deregistered because FSDP2's mixed-precision
    policy casts it from float32 to ``param_dtype`` (e.g. bfloat16) during forward.
    This lossy cast corrupts the packed uint8 byte patterns.  After deregistration
    the parameter stays as a sharded DTensor; forward code calls ``full_tensor()``
    to all-gather in the original float32 dtype without the mixed-precision cast.

    DCP checkpoint save/load still works because the parameter remains a DTensor
    with mesh placement metadata — DCP reads that directly, not FSDP param groups.

    Args:
        model: Model containing QLoRALinear modules.
        param_names: Parameter names to deregister (default: just ``"weight"``).
    """
    param_name_set = set(param_names)

    # Collect all QLoRALinear modules for matching
    qlora_modules = set()
    for module in model.modules():
        if isinstance(module, QLoRALinear):
            qlora_modules.add(id(module))

    removed = 0
    for module in model.modules():
        state_fn = getattr(module, "_get_fsdp_state", None)
        if state_fn is None:
            continue
        try:
            fsdp_state = state_fn()
            pg = fsdp_state._fsdp_param_group
            if pg is None:
                continue
            to_remove = [
                fp for fp in pg.fsdp_params
                if hasattr(fp, "_module_info")
                and fp._module_info.param_name in param_name_set
                and id(fp._module_info.module) in qlora_modules
            ]
            for fp in to_remove:
                pg.fsdp_params.remove(fp)
                removed += 1
        except Exception:
            pass
    return removed


def maybe_quantize_qlora(model: nn.Module) -> int:
    """
    Quantize deferred QLoRALinear weights after FSDP loads them.

    Call this after build_parallelize_model() when using meta init.
    Each module's weight parameter (loaded by FSDP) is quantized into packed_weight_f32
    (float32 parameter for FSDP2 sharding). Both ``weight`` and ``packed_weight_f32``
    are then deregistered from FSDP2 param groups:

    * ``weight``: storage is freed — must not be all-gathered.
    * ``packed_weight_f32``: must not be cast to ``param_dtype`` (e.g. bf16) by
      FSDP2 mixed precision, as this corrupts packed uint8 byte patterns.
      Stays as a sharded DTensor; forward code uses ``full_tensor()`` to
      all-gather in the original float32 dtype.

    Returns:
        Number of modules quantized.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, QLoRALinear) and module.weight is not None:
            module.quantize_weight()
            count += 1

    if count > 0:
        removed = _deregister_qlora_weights_from_fsdp(
            model, param_names=("weight", "packed_weight_f32"),
        )
        # Now safe to fully delete weight params (deregistered from FSDP2)
        for module in model.modules():
            if isinstance(module, QLoRALinear) and module.weight is not None:
                module.weight = None
        torch.cuda.empty_cache()
        logger.info(
            f"Quantized {count} QLoRALinear modules (deferred quantization), "
            f"deregistered {removed} params from FSDP2"
        )
    return count


def maybe_load_and_quantize_moe_qlora(model: nn.Module, weights_path: str) -> int:
    """
    Load and quantize QLoRAMoeExperts weights directly from checkpoint.

    For large MoE models (e.g., 235B), base expert weights are NOT loaded
    via FSDP (to avoid OOM). Instead, each module loads its weights from
    the checkpoint file, quantizes on GPU, and frees the bf16 copy.

    Args:
        model: Model with QLoRAMoeExperts modules
        weights_path: Path to HF checkpoint directory or hub ID

    Returns:
        Number of MoE modules loaded and quantized.
    """
    # Check if there are any QLoRAMoeExperts that need loading
    moe_modules = [
        m for m in model.modules()
        if isinstance(m, QLoRAMoeExperts) and not m._weights_loaded
    ]
    if not moe_modules:
        return 0

    # Load weight map from index file
    weight_map = None
    try:
        from transformers.utils import cached_file, SAFE_WEIGHTS_INDEX_NAME
        import json
        index_path = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME)
        if index_path:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
    except Exception:
        pass

    moe_count = 0
    for module in moe_modules:
        module.load_and_quantize_weights(weights_path, weight_map=weight_map)
        # Materialize LoRA params from meta device to GPU (they're excluded from FSDP
        # via _skip_fsdp + ignored_params, so FSDP won't materialize them)
        for name, param in module.named_parameters():
            if param.device.type == "meta":
                materialized = nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device="cuda"),
                    requires_grad=param.requires_grad,
                )
                # Re-initialize LoRA A with kaiming, B with zeros
                parts = name.split("_")  # e.g. "gate_lora_A" -> ["gate", "lora", "A"]
                if parts[-1] == "A":
                    for i in range(materialized.shape[0]):
                        nn.init.kaiming_uniform_(materialized.data[i], a=5**0.5)
                # B stays zeros
                setattr(module, name, materialized)
        moe_count += 1

    if moe_count > 0:
        torch.cuda.empty_cache()
        logger.info(
            f"Loaded and quantized {moe_count} QLoRAMoeExperts modules "
            f"(direct from checkpoint, bypassing FSDP)"
        )
    return moe_count


def detect_prequantized_nvfp4(weights_path: str) -> bool:
    """Detect whether a checkpoint contains pre-quantized NVFP4 weights (modelopt format).

    Delegates to :func:`~xorl.models.checkpoint_handlers.buffers.detect_prequantized_checkpoint`.

    Args:
        weights_path: Path to HF model directory

    Returns:
        True if the checkpoint is pre-quantized NVFP4.
    """
    from xorl.models.checkpoint_handlers.buffers import detect_prequantized_checkpoint
    return detect_prequantized_checkpoint(weights_path)


def detect_prequantized_block_fp8(weights_path: str) -> bool:
    """Detect whether a checkpoint contains pre-quantized block FP8 weights.

    Delegates to :func:`~xorl.models.checkpoint_handlers.buffers.detect_prequantized_block_fp8_checkpoint`.

    Args:
        weights_path: Path to HF model directory

    Returns:
        True if the checkpoint is pre-quantized block FP8.
    """
    from xorl.models.checkpoint_handlers.buffers import detect_prequantized_block_fp8_checkpoint
    return detect_prequantized_block_fp8_checkpoint(weights_path)


def maybe_load_prequantized_qlora(model: nn.Module, weights_path: str) -> int:
    """Load pre-quantized weights into QLoRALinear modules from checkpoint.

    For pre-quantized NVFP4 checkpoints (modelopt format), this replaces the
    deferred quantization step. Loads packed weights, block scales, and global
    scales directly into QLoRALinear buffers.

    Also handles QLoRAMoeExperts (auto-detected internally via weight_map probing).

    Args:
        model: Model with QLoRALinear/QLoRAMoeExperts modules
        weights_path: Path to HF model directory

    Returns:
        Number of modules loaded.
    """
    # Load weight map from index file
    weight_map = None
    try:
        from transformers.utils import cached_file, SAFE_WEIGHTS_INDEX_NAME
        index_path = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME)
        if index_path:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
    except Exception:
        pass

    if weight_map is None:
        raise RuntimeError(
            f"Could not load weight index from {weights_path}. "
            "Pre-quantized loading requires model.safetensors.index.json."
        )

    # Shared shard cache across all modules (avoids re-reading same shard files)
    shard_cache = {}

    # Load QLoRALinear modules (attention, dense MLP)
    linear_count = 0
    for fqn, module in model.named_modules():
        if isinstance(module, QLoRALinear) and module._is_prequantized:
            module.load_prequantized_weights(weight_map, shard_cache, weights_path)
            linear_count += 1

    # Load QLoRAMoeExperts (auto-detects prequant via weight_map probing)
    moe_count = 0
    for module in model.modules():
        if isinstance(module, QLoRAMoeExperts) and not module._weights_loaded:
            module.load_and_quantize_weights(weights_path, weight_map=weight_map)
            # Materialize LoRA params from meta device to GPU
            for name, param in module.named_parameters():
                if param.device.type == "meta":
                    materialized = nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype, device="cuda"),
                        requires_grad=param.requires_grad,
                    )
                    parts = name.split("_")
                    if parts[-1] == "A":
                        import math
                        for i in range(materialized.shape[0]):
                            nn.init.kaiming_uniform_(materialized.data[i], a=math.sqrt(5))
                    setattr(module, name, materialized)
            moe_count += 1

    # Free shard cache
    shard_cache.clear()

    total = linear_count + moe_count
    if total > 0:
        # Deregister packed_weight_f32 from FSDP2 to prevent mixed-precision
        # bf16 cast that corrupts packed uint8 byte patterns.
        removed = _deregister_qlora_weights_from_fsdp(
            model, param_names=("packed_weight_f32",),
        )
        torch.cuda.empty_cache()
        logger.info(
            f"Loaded pre-quantized weights: {linear_count} QLoRALinear + "
            f"{moe_count} QLoRAMoeExperts modules (direct from checkpoint), "
            f"deregistered {removed} packed_weight_f32 params from FSDP2"
        )
    return total


def maybe_requant_qlora(model: nn.Module, ema_decay: float = 0.1) -> int:
    """
    Merge LoRA deltas and re-quantize all QLoRA modules unconditionally.

    Call this between optimizer.step() and the next forward pass.
    Under FSDP2, parameters must not be mutated during forward — this function
    performs the merge/re-quantize outside the forward pass.

    The training loop controls when this is called (every N steps).
    EMA-tracked amax is used to inform global_scale for better fp8 precision.

    Args:
        model: Model with QLoRALinear/QLoRAMoeExperts modules.
        ema_decay: EMA decay for amax tracking (0.1 = 10% new, 90% old).

    Returns:
        Number of modules that were re-quantized.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, QLoRALinear):
            module.merge_weights(ema_decay=ema_decay)
            count += 1
        elif isinstance(module, QLoRAMoeExperts):
            module.merge_weights(ema_decay=ema_decay)
            count += 1
    if count > 0:
        logger.info(f"Re-quantized {count} QLoRA modules (LoRA merged into base)")
    return count
