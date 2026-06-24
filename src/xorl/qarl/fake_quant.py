"""Native dense QARL fake-quant modules.

This initial QARL path is intentionally small: dense full-weight models only,
dynamic E4M3 fake quantization, and full-precision master parameters with
straight-through gradients.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import math
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


_SUPPORTED_QARL_FORMATS = {"fp8", "fp8_e4m3", "e4m3", "fp8_default_cfg"}
_DEFAULT_QARL_QUANT_CFG = {
    "format": "fp8_e4m3",
    "weight": True,
    "activation": True,
    "dynamic": True,
    "weight_block_size": [128, 128],
}
_QARL_MTP_COUNT_FIELDS = ("mtp_num_hidden_layers", "mtp_num_layers", "num_nextn_predict_layers")
_QARL_MTP_FLAG_FIELDS = ("enable_mtp_training",)
_QARL_UNSUPPORTED_MODEL_KEYWORDS = ("mamba",)


def normalize_qarl_quant_cfg(quant_cfg: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the initial XoRL-native QARL quantization config."""

    if quant_cfg is None:
        return dict(_DEFAULT_QARL_QUANT_CFG)
    if isinstance(quant_cfg, str):
        alias = quant_cfg.strip().lower()
        if alias not in _SUPPORTED_QARL_FORMATS:
            raise ValueError(
                f"Unsupported qarl_quant_cfg={quant_cfg!r}. Initial QARL supports only dynamic FP8_DEFAULT_CFG/e4m3."
            )
        return dict(_DEFAULT_QARL_QUANT_CFG)
    if not isinstance(quant_cfg, Mapping):
        raise ValueError(f"qarl_quant_cfg must be a mapping, string alias, or null; got {type(quant_cfg).__name__}")

    unknown = set(quant_cfg) - {"format", "quant_method", "weight", "activation", "dynamic", "weight_block_size"}
    if unknown:
        raise ValueError(f"Unsupported qarl_quant_cfg keys: {sorted(unknown)}")

    fmt = str(quant_cfg.get("format", quant_cfg.get("quant_method", "fp8_e4m3"))).strip().lower()
    if fmt not in _SUPPORTED_QARL_FORMATS:
        raise ValueError(
            f"Unsupported qarl_quant_cfg format={fmt!r}. Initial QARL supports only dynamic FP8_DEFAULT_CFG/e4m3."
        )
    dynamic = bool(quant_cfg.get("dynamic", True))
    if not dynamic:
        raise ValueError("Static/calibrated QARL quantizers are not implemented yet; use dynamic=true")
    raw_block_size = quant_cfg.get("weight_block_size", _DEFAULT_QARL_QUANT_CFG["weight_block_size"])
    if (
        not isinstance(raw_block_size, (list, tuple))
        or len(raw_block_size) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in raw_block_size)
    ):
        raise ValueError("qarl_quant_cfg.weight_block_size must be a pair of positive integers")
    return {
        "format": "fp8_e4m3",
        "weight": bool(quant_cfg.get("weight", True)),
        "activation": bool(quant_cfg.get("activation", True)),
        "dynamic": True,
        "weight_block_size": [int(raw_block_size[0]), int(raw_block_size[1])],
    }


def _positive_int_config_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        try:
            return int(value.strip()) > 0
        except ValueError:
            return False
    return False


def _truthy_config_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _qarl_mtp_evidence_from_config(config: Any, *, prefix: str) -> list[str]:
    if config is None:
        return []

    evidence: list[str] = []
    sections = [(prefix, config)]
    text_config = _config_value(config, "text_config")
    if text_config is not None:
        sections.append((f"{prefix}.text_config", text_config))

    for section_prefix, section in sections:
        for key in _QARL_MTP_COUNT_FIELDS:
            value = _config_value(section, key)
            if _positive_int_config_value(value):
                evidence.append(f"{section_prefix}.{key}={value}")
        for key in _QARL_MTP_FLAG_FIELDS:
            value = _config_value(section, key)
            if _truthy_config_value(value):
                evidence.append(f"{section_prefix}.{key}={value}")
    return evidence


def _qarl_model_family_evidence_from_config(config: Any, *, prefix: str) -> list[str]:
    if config is None:
        return []

    evidence: list[str] = []
    sections = [(prefix, config)]
    text_config = _config_value(config, "text_config")
    if text_config is not None:
        sections.append((f"{prefix}.text_config", text_config))

    for section_prefix, section in sections:
        model_type = _config_value(section, "model_type")
        if isinstance(model_type, str):
            model_type_lower = model_type.strip().lower()
            if any(keyword in model_type_lower for keyword in _QARL_UNSUPPORTED_MODEL_KEYWORDS):
                evidence.append(f"{section_prefix}.model_type={model_type}")

        architectures = _config_value(section, "architectures")
        if isinstance(architectures, str):
            architecture_values = [architectures]
        elif isinstance(architectures, Collection) and not isinstance(architectures, Mapping):
            architecture_values = [entry for entry in architectures if isinstance(entry, str)]
        else:
            architecture_values = []
        for architecture in architecture_values:
            architecture_lower = architecture.strip().lower()
            if any(keyword in architecture_lower for keyword in _QARL_UNSUPPORTED_MODEL_KEYWORDS):
                evidence.append(f"{section_prefix}.architectures includes {architecture}")
                break
    return evidence


def _load_local_json_config(config_path: str | Path | None) -> dict[str, Any] | None:
    if not config_path:
        return None

    path = Path(str(config_path)).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        return None

    try:
        with path.open(encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _looks_like_mtp_module_name(entry: str) -> bool:
    parts = [part.strip().lower() for part in entry.replace("/", ".").split(".") if part.strip()]
    return any(
        part == "mtp"
        or part.startswith("mtp_")
        or part == "nextn"
        or part.startswith("nextn_")
        for part in parts
    )


def _qarl_mtp_evidence_from_module_names(module_names: Collection[str] | None, *, prefix: str) -> list[str]:
    if not module_names:
        return []
    mtp_entries = [entry for entry in module_names if isinstance(entry, str) and _looks_like_mtp_module_name(entry)]
    if not mtp_entries:
        return []
    preview = ", ".join(mtp_entries[:3])
    suffix = ", ..." if len(mtp_entries) > 3 else ""
    return [f"{prefix} includes {preview}{suffix}"]


def qarl_unsupported_scope_reason(
    *,
    model_config: Any = None,
    config_path: str | Path | None = None,
    module_names: Collection[str] | None = None,
) -> str | None:
    """Return the unsupported-scope reason for QARL config evidence, if any."""

    evidence = _qarl_mtp_evidence_from_config(model_config, prefix="model_config")
    evidence.extend(_qarl_model_family_evidence_from_config(model_config, prefix="model_config"))
    local_config = _load_local_json_config(config_path)
    if local_config is not None:
        evidence.extend(_qarl_mtp_evidence_from_config(local_config, prefix="config_json"))
        evidence.extend(_qarl_model_family_evidence_from_config(local_config, prefix="config_json"))
    evidence.extend(_qarl_mtp_evidence_from_module_names(module_names, prefix="qarl module selectors"))

    if not evidence:
        return None
    return (
        "QARL fake quantization currently supports dense full-weight models only; "
        f"MTP/speculative and Mamba model scopes are unsupported. Detected {evidence[0]}."
    )


def _fake_quant_scaled_fp8_e4m3_ste(tensor: torch.Tensor, *, enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled or tensor.numel() == 0:
        return tensor, torch.zeros((), dtype=torch.float32, device=tensor.device)
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    work = tensor.detach().float()
    scale = work.abs().max().clamp(min=1e-12) / fp8_max
    quantized = (work / scale).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype).to(torch.float32) * scale
    quantized = quantized.to(tensor.dtype)
    return tensor + (quantized - tensor).detach(), scale.to(torch.float32)


def _fake_quant_block_fp8_e4m3_ste(
    tensor: torch.Tensor,
    *,
    enabled: bool,
    weight_block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled or tensor.numel() == 0:
        block_rows, block_cols = weight_block_size
        rows, cols = tensor.shape
        scale_shape = (max(1, math.ceil(rows / block_rows)), max(1, math.ceil(cols / block_cols)))
        return tensor, torch.zeros(scale_shape, dtype=torch.float32, device=tensor.device)
    if tensor.ndim != 2:
        raise ValueError(f"Block FP8 QARL weight fake quant expects a 2D tensor, got shape={tuple(tensor.shape)}")

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    block_rows, block_cols = weight_block_size
    work = tensor.detach().float()
    rows, cols = work.shape
    pad_rows = (block_rows - rows % block_rows) % block_rows
    pad_cols = (block_cols - cols % block_cols) % block_cols
    if pad_rows or pad_cols:
        padded = torch.zeros(rows + pad_rows, cols + pad_cols, dtype=torch.float32, device=work.device)
        padded[:rows, :cols] = work
    else:
        padded = work

    block_row_count = padded.shape[0] // block_rows
    block_col_count = padded.shape[1] // block_cols
    blocks = padded.reshape(block_row_count, block_rows, block_col_count, block_cols).permute(0, 2, 1, 3)
    block_max = blocks.abs().reshape(block_row_count, block_col_count, -1).max(dim=-1).values
    scale = block_max.clamp(min=1e-12) / fp8_max
    quantized_blocks = (blocks / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    dequantized = quantized_blocks.to(torch.float32) * scale.unsqueeze(-1).unsqueeze(-1)
    dequantized = dequantized.permute(0, 2, 1, 3).reshape(padded.shape)
    if pad_rows or pad_cols:
        dequantized = dequantized[:rows, :cols].contiguous()
    dequantized = dequantized.to(tensor.dtype)
    return tensor + (dequantized - tensor).detach(), scale.to(torch.float32).contiguous()


class QARLLinear(nn.Module):
    """Linear layer with dynamic FP8 fake quantization and STE gradients."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        quantize_weight: bool = True,
        quantize_activation: bool = True,
        weight_block_size: tuple[int, int] = (128, 128),
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.qarl_format = "fp8_e4m3"
        self.qarl_quantize_weight = bool(quantize_weight)
        self.qarl_quantize_activation = bool(quantize_activation)
        self.qarl_weight_block_size = (int(weight_block_size[0]), int(weight_block_size[1]))
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device)) if bias else None
        self.register_buffer("qarl_input_amax", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("qarl_weight_amax", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer("qarl_input_scale_inv", torch.zeros((), dtype=torch.float32), persistent=True)
        self.register_buffer(
            "qarl_weight_scale_inv",
            torch.zeros(
                (
                    max(1, math.ceil(out_features / self.qarl_weight_block_size[0])),
                    max(1, math.ceil(in_features / self.qarl_weight_block_size[1])),
                ),
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer("qarl_forward_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, module: nn.Linear, *, quant_cfg: Mapping[str, Any]) -> "QARLLinear":
        out = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quantize_weight=bool(quant_cfg.get("weight", True)),
            quantize_activation=bool(quant_cfg.get("activation", True)),
            weight_block_size=tuple(quant_cfg.get("weight_block_size", (128, 128))),
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        out.weight = nn.Parameter(module.weight.detach().clone(), requires_grad=module.weight.requires_grad)
        if module.bias is not None:
            out.bias = nn.Parameter(module.bias.detach().clone(), requires_grad=module.bias.requires_grad)
        return out

    def _record_amax(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            value = torch.zeros((), dtype=torch.float32, device=tensor.device)
        else:
            value = tensor.detach().float().abs().max()
        getattr(self, name).copy_(value.to(device=getattr(self, name).device, dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._record_amax("qarl_input_amax", input)
        self._record_amax("qarl_weight_amax", self.weight)
        self.qarl_forward_count.add_(1)
        fake_input, input_scale = _fake_quant_scaled_fp8_e4m3_ste(input, enabled=self.qarl_quantize_activation)
        fake_weight, weight_scale = _fake_quant_block_fp8_e4m3_ste(
            self.weight,
            enabled=self.qarl_quantize_weight,
            weight_block_size=self.qarl_weight_block_size,
        )
        self.qarl_input_scale_inv.copy_(input_scale.to(device=self.qarl_input_scale_inv.device, dtype=torch.float32))
        self.qarl_weight_scale_inv.copy_(
            weight_scale.to(device=self.qarl_weight_scale_inv.device, dtype=torch.float32)
        )
        return F.linear(fake_input, fake_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={self.qarl_format}, "
            f"weight={self.qarl_quantize_weight}, activation={self.qarl_quantize_activation}, "
            f"weight_block_size={self.qarl_weight_block_size}"
        )


def _get_submodule(model: nn.Module, target: str) -> tuple[nn.Module, str]:
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches_any(name: str, short_name: str, patterns: Collection[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(short_name, pattern) for pattern in patterns)


def _contains_moe_experts(model: nn.Module) -> bool:
    try:
        from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        MoEExperts = ()  # type: ignore[assignment]
    return any(isinstance(module, MoEExperts) for module in model.modules())


def inject_qarl_into_model(
    model: nn.Module,
    *,
    quant_cfg: str | Mapping[str, Any] | None = None,
    target_modules: list[str] | None = None,
    exclude_modules: Collection[str] | None = None,
) -> int:
    """Replace dense ``nn.Linear`` modules with dynamic FP8 fake-quant linears."""

    if int(getattr(getattr(model, "config", None), "num_experts", 0) or 0) > 0 or _contains_moe_experts(model):
        raise ValueError("QARL fake quantization currently supports dense full-weight models only; MoE is unsupported")
    unsupported_reason = qarl_unsupported_scope_reason(
        model_config=getattr(model, "config", None),
        module_names=[*(target_modules or []), *(exclude_modules or [])],
    )
    if unsupported_reason is not None:
        raise ValueError(unsupported_reason)

    normalized_cfg = normalize_qarl_quant_cfg(quant_cfg)
    targets = set(target_modules or [])
    excludes = set(exclude_modules or [])
    matched_paths: list[str] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear) or isinstance(module, QARLLinear):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if targets and not _matches_any(name, short_name, targets):
            continue
        if _matches_any(name, short_name, excludes):
            continue
        matched_paths.append(name)

    for path in matched_paths:
        parent, attr_name = _get_submodule(model, path)
        original = getattr(parent, attr_name)
        qarl_module = QARLLinear.from_linear(original, quant_cfg=normalized_cfg)
        qarl_module.qarl_module_name = path
        setattr(parent, attr_name, qarl_module)

    model._qarl_config = {
        "quant_cfg": dict(normalized_cfg),
        "target_modules": list(target_modules or []),
        "exclude_modules": list(exclude_modules or []),
    }
    if matched_paths:
        logger.info("Injected QARL fake quantization into %d Linear module(s)", len(matched_paths))
    else:
        logger.warning(
            "QARL was enabled but no Linear modules were changed (target_modules=%s, exclude_modules=%s)",
            target_modules,
            sorted(excludes),
        )
    return len(matched_paths)


def summarize_qarl_model(model: nn.Module) -> dict[str, Any]:
    modules: list[str] = []
    forward_counts: dict[str, int] = {}
    input_amax: dict[str, float] = {}
    weight_amax: dict[str, float] = {}
    input_scale_inv: dict[str, float] = {}
    weight_scale_inv_shapes: dict[str, tuple[int, ...]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, QARLLinear):
            continue
        modules.append(name)
        forward_counts[name] = int(module.qarl_forward_count.item())
        input_amax[name] = float(module.qarl_input_amax.item())
        weight_amax[name] = float(module.qarl_weight_amax.item())
        input_scale_inv[name] = float(module.qarl_input_scale_inv.item())
        weight_scale_inv_shapes[name] = tuple(module.qarl_weight_scale_inv.shape)
    return {
        "enabled": bool(modules),
        "linear_count": len(modules),
        "linear_names": modules,
        "forward_counts": forward_counts,
        "input_amax": input_amax,
        "weight_amax": weight_amax,
        "input_scale_inv": input_scale_inv,
        "weight_scale_inv_shapes": weight_scale_inv_shapes,
    }
