"""Model rewriting utilities for full-weight FP8 compute training."""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Collection, List, Optional, Tuple

import torch.nn as nn

from xorl.fp8_training.config_compat import merge_fp8_bf16_layer_island_excludes
from xorl.fp8_training.linear import FP8BackwardMode, FP8CorrectionMode, FP8Linear


logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE_MODULES: set[str] = set()
DEFAULT_FP8_GROUPED_BACKEND = "triton_grouped"
_FP8_GROUPED_BACKENDS = {"block_loop", "deep_gemm", "scalar_quack", "triton_grouped"}
_FP8_LINEAR_OVERRIDE_KEYS = {
    "activation_amax_scale",
    "backward_mode",
    "block_size",
    "correction_mode",
    "smoothquant_alpha",
    "weight_amax_scale",
}


def validate_fp8_grouped_backend(backend: str) -> str:
    if backend not in _FP8_GROUPED_BACKENDS:
        raise ValueError(
            f"Unsupported FP8 grouped backend: {backend!r}. Expected one of {sorted(_FP8_GROUPED_BACKENDS)}"
        )
    return backend


def _get_submodule(model: nn.Module, target: str) -> Tuple[nn.Module, str]:
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches_any(name: str, short_name: str, patterns: Collection[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(short_name, pattern) for pattern in patterns)


def _linear_recipe_override(
    name: str,
    short_name: str,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not overrides:
        return {}
    merged: dict[str, Any] = {}
    for pattern, values in overrides.items():
        if not _matches_any(name, short_name, [pattern]):
            continue
        if not isinstance(values, dict):
            raise ValueError(f"FP8 module override for {pattern!r} must be a mapping, got {type(values).__name__}")
        unknown = set(values) - _FP8_LINEAR_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                f"Unsupported FP8 module override key(s) for {pattern!r}: {sorted(unknown)}. "
                f"Expected keys from {sorted(_FP8_LINEAR_OVERRIDE_KEYS)}"
            )
        merged.update(values)
    return merged


def _is_moe_experts_module(module: nn.Module) -> bool:
    try:
        from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415
    except ImportError:  # pragma: no cover - defensive for partial import environments
        return hasattr(module, "moe_implementation") and hasattr(module, "last_forward_used_fp8")
    return isinstance(module, MoEExperts)


def inject_fp8_training_into_model(
    model: nn.Module,
    *,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[Collection[str]] = None,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
    block_size: int = 128,
    backward_mode: FP8BackwardMode = "fp8",
    smoothquant_alpha: float | None = None,
    lm_head_smoothquant_alpha: float | None = None,
    activation_amax_scale: float = 1.0,
    weight_amax_scale: float = 1.0,
    correction_mode: FP8CorrectionMode = "none",
    module_overrides: dict[str, dict[str, Any]] | None = None,
    allow_bf16_fallback: bool = True,
    moe_grouped_backend: str = DEFAULT_FP8_GROUPED_BACKEND,
) -> int:
    """Enable trainable FP8 compute modules in-place.

    Ordinary dense ``nn.Linear`` modules are replaced with ``FP8Linear``.
    Stacked MoE expert containers are kept as-is and switched to Quack grouped
    FP8 compute because expert weights are parameters, not child Linear modules.

    Args:
        model: Model to rewrite in-place.
        target_modules: Optional short module names to include. ``None`` means
            every ordinary linear except explicit exclusions.
        exclude_modules: Short names, FQNs, or glob patterns to keep in BF16/FP32.
            ``None`` rewrites every matched dense linear, including router gates
            and output heads.
        num_first_layers_bf16: Number of initial decoder layers to keep out of
            FP8 compute. Expanded into ``model.layers.<idx>.*``-style FQN globs.
        num_last_layers_bf16: Number of final decoder layers to keep out of
            FP8 compute. Overlap with first-layer islands is deduplicated.
        block_size: Block size used by block-FP8 quantization.
        backward_mode: ``"fp8"`` uses FP8 GEMMs in backward where possible;
            ``"bf16"`` uses high-precision backward matmuls.
        smoothquant_alpha: Optional SmoothQuant alpha for dense Linear
            matmuls. If set, activation and weight columns are dynamically
            balanced before FP8 quantization.
        lm_head_smoothquant_alpha: Optional SmoothQuant alpha override for
            modules named ``lm_head``. ``None`` reuses ``smoothquant_alpha``.
        activation_amax_scale: Multiplier applied to activation block absmax
            before deriving FP8 scales. Values below 1.0 clip activation
            outliers; values above 1.0 add headroom.
        weight_amax_scale: Multiplier applied to weight block absmax before
            deriving FP8 scales. Values below 1.0 clip weight outliers; values
            above 1.0 add headroom.
        correction_mode: Optional extra FP8 residual GEMMs for dense Linear
            matmuls. ``"none"`` preserves the baseline one-GEMM path;
            ``"activation2"`` applies two activation-residual FP8 passes.
        module_overrides: Optional FQN/short-name glob pattern overrides for
            dense FP8Linear recipes. Supported keys are ``block_size``,
            ``backward_mode``, ``smoothquant_alpha``,
            ``activation_amax_scale``, ``weight_amax_scale``, and
            ``correction_mode``.
        allow_bf16_fallback: If true, CPU/unsupported inputs fall back to
            ``F.linear``. If false, unsupported FP8 execution raises.
        moe_grouped_backend: Grouped GEMM backend for MoE expert FP8 compute.

    Returns:
        Number of modules or expert containers changed.
    """

    moe_grouped_backend = validate_fp8_grouped_backend(moe_grouped_backend)
    targets = set(target_modules or [])
    merged_excludes, bf16_layer_islands = merge_fp8_bf16_layer_island_excludes(
        model,
        _DEFAULT_EXCLUDE_MODULES if exclude_modules is None else exclude_modules,
        num_first_layers_bf16=num_first_layers_bf16,
        num_last_layers_bf16=num_last_layers_bf16,
    )
    excludes = set(merged_excludes)
    model._fp8_training_bf16_layer_islands = {
        "num_first_layers_bf16": int(num_first_layers_bf16),
        "num_last_layers_bf16": int(num_last_layers_bf16),
        "patterns": list(bf16_layer_islands),
    }
    matched_paths: list[str] = []

    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if targets and short_name not in targets:
            continue
        if _matches_any(name, short_name, excludes):
            continue
        matched_paths.append(name)

    for path in matched_paths:
        parent, attr_name = _get_submodule(model, path)
        original = getattr(parent, attr_name)
        module_smoothquant_alpha = (
            lm_head_smoothquant_alpha
            if attr_name == "lm_head" and lm_head_smoothquant_alpha is not None
            else smoothquant_alpha
        )
        recipe = _linear_recipe_override(path, attr_name, module_overrides)
        module_block_size = int(recipe.get("block_size", block_size))
        module_backward_mode = recipe.get("backward_mode", backward_mode)
        module_smoothquant_alpha = recipe.get("smoothquant_alpha", module_smoothquant_alpha)
        module_activation_amax_scale = recipe.get("activation_amax_scale", activation_amax_scale)
        module_weight_amax_scale = recipe.get("weight_amax_scale", weight_amax_scale)
        module_correction_mode = recipe.get("correction_mode", correction_mode)
        fp8_module = FP8Linear.from_linear(
            original,
            block_size=module_block_size,
            backward_mode=module_backward_mode,
            smoothquant_alpha=module_smoothquant_alpha,
            activation_amax_scale=module_activation_amax_scale,
            weight_amax_scale=module_weight_amax_scale,
            correction_mode=module_correction_mode,
            output_dtype="float32" if attr_name == "lm_head" else "input",
            allow_bf16_fallback=allow_bf16_fallback,
        )
        fp8_module.fp8_module_name = path
        setattr(
            parent,
            attr_name,
            fp8_module,
        )

    moe_paths: list[str] = []
    for name, module in model.named_modules():
        if not _is_moe_experts_module(module):
            continue
        short_name = name.rsplit(".", 1)[-1] if name else ""
        if _matches_any(name, short_name, excludes):
            continue
        if module.moe_implementation != "quack":
            logger.info(
                "Switching MoEExperts module %s from moe_implementation=%s to quack for FP8 grouped compute",
                name or "<root>",
                module.moe_implementation,
            )
            module.moe_implementation = "quack"
        module.fp8_training_enabled = True
        module.fp8_training_grouped_backend = moe_grouped_backend
        module.fp8_training_block_size = int(block_size)
        moe_paths.append(name or "<root>")

    if matched_paths:
        logger.info(
            "Injected full-weight FP8 compute into %d Linear module(s), block_size=%d, backward_mode=%s",
            len(matched_paths),
            block_size,
            backward_mode,
        )
    if moe_paths:
        logger.info(
            "Enabled experimental FP8 grouped compute backend=%s on %d MoEExperts module(s)",
            moe_grouped_backend,
            len(moe_paths),
        )
    if bf16_layer_islands:
        logger.info(
            "Kept %d generated first/last decoder layer island(s) in BF16 for FP8 training: %s",
            len(bf16_layer_islands),
            bf16_layer_islands,
        )
    if not matched_paths and not moe_paths:
        logger.warning(
            "FP8 training was enabled but no Linear or MoEExperts modules were changed "
            "(target_modules=%s, exclude_modules=%s)",
            target_modules,
            sorted(excludes),
        )
    return len(matched_paths) + len(moe_paths)


def summarize_fp8_training_model(model: nn.Module, *, max_unused_names: int = 32) -> dict[str, Any]:
    """Return runtime evidence for FP8 training modules in ``model``.

    ``FP8Linear.last_forward_used_fp8`` records whether a module's most recent
    forward used the block-FP8 path. ``MoEExperts.last_forward_used_fp8`` does
    the same for grouped expert compute. The trainer persists this summary after
    the final step so e2e smokes can prove the real training run exercised FP8
    compute instead of only proving that injection occurred.
    """

    linear_modules = 0
    linear_used_fp8 = 0
    linear_allow_fallback = 0
    linear_backward_fp8 = 0
    linear_backward_bf16 = 0
    unused_linear_names: list[str] = []

    moe_modules = 0
    moe_fp8_enabled = 0
    moe_used_fp8 = 0
    moe_quack = 0
    unused_moe_names: list[str] = []
    bf16_layer_island_info = getattr(model, "_fp8_training_bf16_layer_islands", {})

    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            linear_modules += 1
            if module.last_forward_used_fp8:
                linear_used_fp8 += 1
            elif len(unused_linear_names) < max_unused_names:
                unused_linear_names.append(name or "<root>")
            if module.fp8_allow_bf16_fallback:
                linear_allow_fallback += 1
            if module.fp8_backward_mode == "fp8":
                linear_backward_fp8 += 1
            elif module.fp8_backward_mode == "bf16":
                linear_backward_bf16 += 1

        if _is_moe_experts_module(module):
            moe_modules += 1
            if getattr(module, "fp8_training_enabled", False):
                moe_fp8_enabled += 1
                if getattr(module, "last_forward_used_fp8", False):
                    moe_used_fp8 += 1
                elif len(unused_moe_names) < max_unused_names:
                    unused_moe_names.append(name or "<root>")
            if getattr(module, "moe_implementation", None) == "quack":
                moe_quack += 1

    return {
        "linear_modules": linear_modules,
        "linear_modules_used_fp8": linear_used_fp8,
        "linear_modules_allow_bf16_fallback": linear_allow_fallback,
        "linear_modules_backward_fp8": linear_backward_fp8,
        "linear_modules_backward_bf16": linear_backward_bf16,
        "unused_linear_module_names": unused_linear_names,
        "moe_modules": moe_modules,
        "moe_fp8_enabled_modules": moe_fp8_enabled,
        "moe_modules_used_fp8": moe_used_fp8,
        "unused_moe_module_names": unused_moe_names,
        "moe_quack_modules": moe_quack,
        "bf16_layer_island_patterns": list(bf16_layer_island_info.get("patterns", []))
        if isinstance(bf16_layer_island_info, dict)
        else [],
        "bf16_layer_island_count": len(bf16_layer_island_info.get("patterns", []))
        if isinstance(bf16_layer_island_info, dict)
        else 0,
    }
