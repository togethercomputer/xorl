"""QARL helpers for deriving online SGLang FP8 sync configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from xorl.qarl.fake_quant import QARLLinear
from xorl.server.weight_sync.quantization_config import normalize_sync_quantization_config


def _module_name_from_weight(param_name: str) -> str | None:
    if not param_name.endswith(".weight"):
        return None
    return param_name[: -len(".weight")]


def _qarl_module_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, QARLLinear)]


def _qarl_block_size(model: nn.Module, qarl_modules: list[str]) -> list[int]:
    block_sizes: set[tuple[int, int]] = set()
    modules = dict(model.named_modules())
    for name in qarl_modules:
        module = modules[name]
        assert isinstance(module, QARLLinear)
        block_sizes.add(tuple(module.qarl_weight_block_size))
    if len(block_sizes) != 1:
        raise ValueError(f"QARL online FP8 sync requires one weight_block_size across wrapped modules; got {block_sizes}")
    block_rows, block_cols = next(iter(block_sizes))
    return [int(block_rows), int(block_cols)]


def _non_qarl_2d_weight_modules(model: nn.Module, qarl_modules: list[str]) -> list[str]:
    qarl_set = set(qarl_modules)
    skipped: list[str] = []
    seen: set[str] = set()
    for param_name, param in model.named_parameters():
        module_name = _module_name_from_weight(param_name)
        if module_name is None or module_name in qarl_set:
            continue
        if param.ndim == 2 and module_name not in seen:
            seen.add(module_name)
            skipped.append(module_name)
    return skipped


def qarl_sync_quantization_config(
    model: nn.Module,
    base_quantization: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return an FP8 sync config that exports only QARL-wrapped 2D weights.

    The receiver sees the same block-FP8 E4M3 weight contract used by QARL's
    fake-quant forward path. Non-QARL 2D weights are added to
    ``modules_to_not_convert`` so they remain in the source precision.
    """

    qarl_modules = _qarl_module_names(model)
    if not qarl_modules:
        return None if base_quantization is None else normalize_sync_quantization_config(dict(base_quantization))

    block_size = _qarl_block_size(model, qarl_modules)
    base = dict(base_quantization or {})
    if base.get("quant_method") is None:
        base["quant_method"] = "fp8"
    if base.get("fmt") is None:
        base["fmt"] = "e4m3"
    if base.get("activation_scheme") is None:
        base["activation_scheme"] = "dynamic"
    if "weight_block_size" in base:
        normalized_block_size = normalize_sync_quantization_config(
            {"quant_method": "fp8", "weight_block_size": base["weight_block_size"]}
        )
        assert normalized_block_size is not None
        if normalized_block_size["weight_block_size"] != block_size:
            raise ValueError(
                "Explicit QARL sync weight_block_size must match wrapped modules: "
                f"sync={normalized_block_size['weight_block_size']}, qarl={block_size}"
            )
    base["weight_block_size"] = block_size

    generated_skips = _non_qarl_2d_weight_modules(model, qarl_modules)
    explicit_skips = base.get("modules_to_not_convert") or []
    base["modules_to_not_convert"] = [*explicit_skips, *generated_skips]
    base["xorl_qarl_sync"] = {
        "enabled": True,
        "folded_modules": qarl_modules,
        "source": "qarl_fake_quant",
    }
    return normalize_sync_quantization_config(base, context="qarl sync quantization")
