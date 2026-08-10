"""Opt-in runtime diagnostics for FP8 training numerics."""

from __future__ import annotations

import atexit
import fnmatch
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


_PROFILE_ENV = "XORL_FP8_LINEAR_ERROR_PROFILE"
_PROFILE_OUTPUT_ENV = "XORL_FP8_LINEAR_ERROR_PROFILE_OUTPUT"
_MAX_CALLS_ENV = "XORL_FP8_LINEAR_ERROR_PROFILE_MAX_CALLS_PER_MODULE"
_MAX_ROWS_ENV = "XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS"
_ROW_INDICES_ENV = "XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES"
_DEFAULT_MAX_CALLS_PER_MODULE = 1
_DEFAULT_MAX_ROWS = 16
_EPS = 1e-6


@dataclass
class _LinearErrorStats:
    calls: int = 0
    sampled_calls: int = 0
    elements: int = 0
    mean_abs_error_sum: float = 0.0
    mean_rel_error_sum: float = 0.0
    rms_error_sum: float = 0.0
    mean_reference_abs_sum: float = 0.0
    mean_output_abs_sum: float = 0.0
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    weight_shape: tuple[int, ...] | None = None
    sampled_row_indices: tuple[int, ...] | None = None
    output_dtype: str | None = None
    metric_sampled_calls: dict[str, int] = field(default_factory=dict)
    metric_mean_abs_error_sum: dict[str, float] = field(default_factory=dict)
    metric_mean_rel_error_sum: dict[str, float] = field(default_factory=dict)
    metric_rms_error_sum: dict[str, float] = field(default_factory=dict)
    metric_max_abs_error: dict[str, float] = field(default_factory=dict)
    metric_max_rel_error: dict[str, float] = field(default_factory=dict)
    sampled_call_summaries: list[dict[str, Any]] = field(default_factory=list)

    def add_sample(
        self,
        output: Tensor,
        reference: Tensor,
        *,
        full_input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        row_indices: tuple[int, ...],
        call_index: int,
    ) -> None:
        output_f = output.detach().float()
        reference_f = reference.detach().float()
        abs_error = (output_f - reference_f).abs()
        rel_error = abs_error / reference_f.abs().clamp_min(_EPS)
        mean_abs_error = float(abs_error.mean().item())
        mean_rel_error = float(rel_error.mean().item())
        rms_error = float(abs_error.square().mean().sqrt().item())
        mean_reference_abs = float(reference_f.abs().mean().item())
        mean_output_abs = float(output_f.abs().mean().item())
        max_abs_error = float(abs_error.max().item())
        max_rel_error = float(rel_error.max().item())

        self.sampled_calls += 1
        self.elements += int(abs_error.numel())
        self.mean_abs_error_sum += mean_abs_error
        self.mean_rel_error_sum += mean_rel_error
        self.rms_error_sum += rms_error
        self.mean_reference_abs_sum += mean_reference_abs
        self.mean_output_abs_sum += mean_output_abs
        self.max_abs_error = max(self.max_abs_error, max_abs_error)
        self.max_rel_error = max(self.max_rel_error, max_rel_error)
        self.input_shape = full_input_shape
        self.output_shape = tuple(output.shape)
        self.weight_shape = weight_shape
        self.sampled_row_indices = row_indices
        self.output_dtype = str(output.dtype).replace("torch.", "")
        self.sampled_call_summaries.append(
            {
                "call_index": call_index,
                "row_indices": list(row_indices),
                "input_shape": list(full_input_shape),
                "output_shape": list(output.shape),
                "mean_abs_error": mean_abs_error,
                "mean_rel_error": mean_rel_error,
                "rms_error": rms_error,
                "mean_reference_abs": mean_reference_abs,
                "mean_output_abs": mean_output_abs,
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
            }
        )

    def add_metric_sample(self, name: str, output: Tensor, reference: Tensor) -> None:
        output_f = output.detach().float()
        reference_f = reference.detach().float()
        abs_error = (output_f - reference_f).abs()
        rel_error = abs_error / reference_f.abs().clamp_min(_EPS)

        self.metric_sampled_calls[name] = self.metric_sampled_calls.get(name, 0) + 1
        self.metric_mean_abs_error_sum[name] = self.metric_mean_abs_error_sum.get(name, 0.0) + float(
            abs_error.mean().item()
        )
        self.metric_mean_rel_error_sum[name] = self.metric_mean_rel_error_sum.get(name, 0.0) + float(
            rel_error.mean().item()
        )
        self.metric_rms_error_sum[name] = self.metric_rms_error_sum.get(name, 0.0) + float(
            abs_error.square().mean().sqrt().item()
        )
        self.metric_max_abs_error[name] = max(
            self.metric_max_abs_error.get(name, 0.0),
            float(abs_error.max().item()),
        )
        self.metric_max_rel_error[name] = max(
            self.metric_max_rel_error.get(name, 0.0),
            float(rel_error.max().item()),
        )

    def as_dict(self) -> dict[str, Any]:
        denom = max(1, self.sampled_calls)
        result = {
            "calls": self.calls,
            "sampled_calls": self.sampled_calls,
            "elements": self.elements,
            "mean_abs_error": self.mean_abs_error_sum / denom,
            "mean_rel_error": self.mean_rel_error_sum / denom,
            "rms_error": self.rms_error_sum / denom,
            "mean_reference_abs": self.mean_reference_abs_sum / denom,
            "mean_output_abs": self.mean_output_abs_sum / denom,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "output_shape": list(self.output_shape) if self.output_shape is not None else None,
            "weight_shape": list(self.weight_shape) if self.weight_shape is not None else None,
            "sampled_row_indices": list(self.sampled_row_indices) if self.sampled_row_indices is not None else None,
            "sampled_call_summaries": self.sampled_call_summaries,
            "output_dtype": self.output_dtype,
        }
        for name, calls in sorted(self.metric_sampled_calls.items()):
            metric_denom = max(1, calls)
            result.update(
                {
                    f"{name}_sampled_calls": calls,
                    f"{name}_mean_abs_error": self.metric_mean_abs_error_sum[name] / metric_denom,
                    f"{name}_mean_rel_error": self.metric_mean_rel_error_sum[name] / metric_denom,
                    f"{name}_rms_error": self.metric_rms_error_sum[name] / metric_denom,
                    f"{name}_max_abs_error": self.metric_max_abs_error[name],
                    f"{name}_max_rel_error": self.metric_max_rel_error[name],
                }
            )
        return result


_lock = threading.Lock()
_stats: dict[str, _LinearErrorStats] = {}
_atexit_registered = False


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, parsed)


def linear_error_profiling_enabled() -> bool:
    return _env_truthy(_PROFILE_ENV)


def _parse_row_indices(value: str, row_count: int) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        parsed: list[int] = []
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in {2, 3}:
                continue
            try:
                start = int(pieces[0])
                end = int(pieces[1])
                step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            except ValueError:
                continue
            if step == 0:
                continue
            parsed = list(range(start, end, step))
        else:
            try:
                parsed = [int(part)]
            except ValueError:
                continue

        for index in parsed:
            if index < 0:
                index = row_count + index
            if 0 <= index < row_count and index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


@dataclass(frozen=True)
class _RowSelector:
    pattern: str
    row_spec: str
    call_index: int | None = None


def _parse_row_selectors(value: str) -> list[_RowSelector] | None:
    if "=" not in value:
        return None

    selectors: list[_RowSelector] = []
    for raw_part in value.split(";"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        raw_target, raw_rows = part.split("=", 1)
        target = raw_target.strip() or "*"
        row_spec = raw_rows.strip()
        call_index = None
        if "@" in target:
            target, raw_call_index = target.rsplit("@", 1)
            target = target.strip() or "*"
            try:
                call_index = int(raw_call_index)
            except ValueError:
                continue
            if call_index <= 0:
                continue
        selectors.append(_RowSelector(pattern=target, row_spec=row_spec, call_index=call_index))
    return selectors


def _selector_matches_name(selector: _RowSelector, module_name: str) -> bool:
    return fnmatch.fnmatchcase(module_name, selector.pattern)


def _selected_row_indices(module_name: str, row_count: int, max_rows: int, call_index: int) -> list[int]:
    row_selector_value = os.environ.get(_ROW_INDICES_ENV, "")
    selectors = _parse_row_selectors(row_selector_value)
    if selectors is not None:
        module_matches = [
            selector
            for selector in selectors
            if selector.pattern != "*" and _selector_matches_name(selector, module_name)
        ]
        wildcard_matches = [selector for selector in selectors if selector.pattern == "*"]
        candidates = module_matches or wildcard_matches
        for selector in candidates:
            if selector.call_index is not None and selector.call_index != call_index:
                continue
            explicit_rows = _parse_row_indices(selector.row_spec, row_count)
            return explicit_rows[:max_rows] if max_rows > 0 else explicit_rows
        return []

    explicit_rows = _parse_row_indices(row_selector_value, row_count)
    if explicit_rows:
        return explicit_rows[:max_rows] if max_rows > 0 else explicit_rows
    if max_rows > 0 and row_count > max_rows:
        return list(range(max_rows))
    return list(range(row_count))


def _sample_rows(
    module_name: str,
    x: Tensor,
    out: Tensor,
    max_rows: int,
    call_index: int,
) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    x_2d = x.detach().reshape(-1, x.shape[-1])
    out_2d = out.detach().reshape(-1, out.shape[-1])
    row_indices = _selected_row_indices(module_name, x_2d.shape[0], max_rows, call_index)
    if not row_indices:
        return x_2d[:0], out_2d[:0], ()
    index = torch.tensor(row_indices, dtype=torch.long, device=x_2d.device)
    return x_2d.index_select(0, index), out_2d.index_select(0, index), tuple(row_indices)


def _fp8_operand_error_breakdown(
    module: torch.nn.Module,
    x: Tensor,
    out_sample: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    row_indices: tuple[int, ...],
) -> dict[str, tuple[Tensor, Tensor]]:
    """Return operand-level error pairs for a sampled CUDA FP8 linear.

    Each value is ``(candidate, reference)``. The references deliberately omit
    bias so that activation, weight, and kernel/output effects can be separated
    from bias addition.
    """

    if not (x.is_cuda and out_sample.is_cuda and weight.is_cuda):
        return {}

    from xorl.fp8_training.linear import _apply_smoothquant, _pad_last_dim  # noqa: PLC0415
    from xorl.ops.quantize import (  # noqa: PLC0415
        block_fp8_dequantize,
        block_fp8_dequantize_gkn_rowwise,
        block_fp8_gemm,
        block_fp8_quantize,
        block_fp8_quantize_gkn_rowwise,
    )

    block_size = int(getattr(module, "fp8_block_size", 128))
    activation_amax_scale = float(getattr(module, "fp8_activation_amax_scale", 1.0))
    weight_amax_scale = float(getattr(module, "fp8_weight_amax_scale", 1.0))

    a_float = x.detach().reshape(-1, x.shape[-1]).float()
    b_float = weight.float()
    smoothquant_alpha = getattr(module, "fp8_smoothquant_alpha", None)
    if smoothquant_alpha is not None:
        a_float, b_float = _apply_smoothquant(a_float, b_float, float(smoothquant_alpha))

    a_padded = _pad_last_dim(a_float, block_size)
    b_padded = _pad_last_dim(b_float, block_size)
    a_fp8_full, a_scales_full = block_fp8_quantize(
        a_padded,
        block_size=block_size,
        amax_scale=activation_amax_scale,
    )
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(
        b_padded,
        block_size=block_size,
        amax_scale=weight_amax_scale,
    )
    row_index_tensor = torch.tensor(row_indices, dtype=torch.long, device=a_padded.device)
    a_sample = a_padded.index_select(0, row_index_tensor).contiguous()
    a_fp8 = a_fp8_full.index_select(0, row_index_tensor).contiguous()
    a_scales = a_scales_full.index_select(0, row_index_tensor).contiguous()
    a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=block_size)
    b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=block_size)

    reference = a_sample @ b_padded.T
    act_only = a_dequant @ b_padded.T
    weight_only = a_sample @ b_dequant.T
    both = a_dequant @ b_dequant.T
    raw_kernel = block_fp8_gemm(
        a_fp8,
        a_scales,
        b_fp8,
        b_scales,
        block_size=block_size,
        weight_scale_layout="row",
        backend="auto",
    )
    output_unbiased = out_sample.float()
    if bias is not None:
        output_unbiased = output_unbiased - bias.float()

    return {
        "activation_quant": (act_only, reference),
        "weight_quant": (weight_only, reference),
        "operand_quant": (both, reference),
        "kernel_accum": (raw_kernel, both),
        "output_cast": (output_unbiased, raw_kernel),
        "kernel_output": (output_unbiased, both),
    }


def record_linear_error(module: torch.nn.Module, x: Tensor, out: Tensor) -> None:
    """Record FP8-vs-reference error for one ``FP8Linear`` forward.

    This function intentionally computes an expensive reference matmul. It is
    only called when ``XORL_FP8_LINEAR_ERROR_PROFILE`` is enabled.
    """

    name = getattr(module, "fp8_module_name", None) or module.__class__.__name__
    with _lock:
        stats = _stats.setdefault(name, _LinearErrorStats())
        stats.calls += 1
        call_index = stats.calls
        max_calls = _env_int(_MAX_CALLS_ENV, _DEFAULT_MAX_CALLS_PER_MODULE)
        if max_calls > 0 and stats.sampled_calls >= max_calls:
            return

    max_rows = _env_int(_MAX_ROWS_ENV, _DEFAULT_MAX_ROWS)
    x_sample, out_sample, row_indices = _sample_rows(name, x, out, max_rows, call_index)
    if not row_indices:
        return
    weight = module.weight.detach()
    bias = module.bias.detach() if getattr(module, "bias", None) is not None else None

    with torch.no_grad():
        reference = F.linear(
            x_sample.float(),
            weight.float(),
            bias.float() if bias is not None else None,
        )
        if getattr(module, "fp8_output_dtype", "input") == "input":
            reference = reference.to(out_sample.dtype)
        output_shape = (*x.shape[:-1], weight.shape[0])
        metric_samples = _fp8_operand_error_breakdown(module, x, out_sample, weight, bias, row_indices)
        output_path = os.environ.get(_PROFILE_OUTPUT_ENV)
        with _lock:
            stats.add_sample(
                out_sample,
                reference,
                full_input_shape=tuple(x.shape),
                weight_shape=tuple(weight.shape),
                row_indices=row_indices,
                call_index=call_index,
            )
            for metric_name, (metric_output, metric_reference) in metric_samples.items():
                stats.add_metric_sample(metric_name, metric_output, metric_reference)
            stats.output_shape = output_shape
            if output_path:
                _register_atexit_dump_locked()
        if output_path:
            write_linear_error_profile(output_path)


def get_linear_error_profile() -> dict[str, Any]:
    with _lock:
        return {
            name: stats.as_dict()
            for name, stats in sorted(
                _stats.items(),
                key=lambda item: item[1].max_abs_error,
                reverse=True,
            )
        }


def clear_linear_error_profile() -> None:
    with _lock:
        _stats.clear()


def write_linear_error_profile(path: str | os.PathLike[str]) -> None:
    profile = get_linear_error_profile()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")


def _register_atexit_dump_locked() -> None:
    global _atexit_registered
    if _atexit_registered or not os.environ.get(_PROFILE_OUTPUT_ENV):
        return
    atexit.register(_dump_profile_at_exit)
    _atexit_registered = True


def _dump_profile_at_exit() -> None:
    path = os.environ.get(_PROFILE_OUTPUT_ENV)
    if not path:
        return
    try:
        write_linear_error_profile(path)
    except Exception:
        pass
