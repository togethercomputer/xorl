"""Validation for online weight-sync quantization contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_NO_QUANTIZATION_METHODS = {"bf16", "bfloat16", "none", "null"}
_SUPPORTED_FP8_FORMATS = {"e4m3"}
SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY = "_xorl_unsupported_reason"


class UnsupportedSyncQuantizationError(ValueError):
    """Raised when a sync quantization config asks for an unsupported workflow."""


def _format_context(context: str) -> str:
    return context.strip() or "sync quantization"


def _normalize_modules_to_not_convert(
    modules_to_not_convert: Any,
    *,
    context: str,
) -> list[str] | None:
    if modules_to_not_convert is None:
        return None
    if not isinstance(modules_to_not_convert, list) or not all(
        isinstance(item, str) for item in modules_to_not_convert
    ):
        raise UnsupportedSyncQuantizationError(f"{context}.modules_to_not_convert must be a list of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_entry in modules_to_not_convert:
        entry = raw_entry.strip()
        if not entry:
            continue
        if entry.endswith(".weight"):
            entry = entry[: -len(".weight")]
        if entry not in seen:
            seen.add(entry)
            normalized.append(entry)
    return normalized


def normalize_sync_quantization_config(
    quantization: Mapping[str, Any] | None,
    *,
    context: str = "sync quantization",
) -> dict[str, Any] | None:
    """Validate and normalize a public sync quantization config.

    Online weight sync currently has two supported contracts:

    - ``None`` or explicit bf16/no-quantization aliases: send BF16/current tensors.
    - ``{"quant_method": "fp8", ...}``: sender-side block-FP8 quantization.

    INT4/compressed-tensors updates, FP8 training semantics, and QAT/fake-quant
    are intentionally rejected here until their sender/receiver contracts are
    implemented and validated.
    """

    context = _format_context(context)
    if quantization is None:
        return None
    if not isinstance(quantization, Mapping):
        raise UnsupportedSyncQuantizationError(f"{context} must be a dict or null; got {type(quantization).__name__}")
    unsupported_reason = quantization.get(SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY)
    if isinstance(unsupported_reason, str) and unsupported_reason.strip():
        raise UnsupportedSyncQuantizationError(f"{context} is unsupported: {unsupported_reason.strip()}")

    if "quant_method" not in quantization:
        raise UnsupportedSyncQuantizationError(
            f"{context} must contain 'quant_method' (supported: 'fp8'; use null for BF16/no quantization)"
        )
    raw_method = quantization["quant_method"]
    if not isinstance(raw_method, str):
        raise UnsupportedSyncQuantizationError(
            f"{context}.quant_method must be a string; got {type(raw_method).__name__}"
        )

    method = raw_method.strip().lower()
    if method in _NO_QUANTIZATION_METHODS:
        return None
    if method != "fp8":
        raise UnsupportedSyncQuantizationError(
            f"Unsupported {context} quant_method={raw_method!r}. Online weight sync currently supports only "
            "BF16/no quantization and sender-side block-FP8. INT4/compressed-tensors updates and QAT/fake-quant "
            "sync are not implemented."
        )

    normalized = dict(quantization)
    normalized["quant_method"] = "fp8"

    raw_fmt = normalized.get("fmt", "e4m3")
    if not isinstance(raw_fmt, str):
        raise UnsupportedSyncQuantizationError(f"{context}.fmt must be a string when provided")
    fmt = raw_fmt.strip().lower()
    if fmt not in _SUPPORTED_FP8_FORMATS:
        raise UnsupportedSyncQuantizationError(
            f"Unsupported {context}.fmt={raw_fmt!r}; online FP8 weight sync supports only Slime/SGLang-compatible "
            "E4M3 block-FP8. Other FP8 dtypes need a separate receiver contract and validation gate."
        )
    normalized["fmt"] = fmt

    raw_activation_scheme = normalized["activation_scheme"] if "activation_scheme" in normalized else "dynamic"
    if not isinstance(raw_activation_scheme, str):
        raise UnsupportedSyncQuantizationError(f"{context}.activation_scheme must be a string when provided")
    activation_scheme = raw_activation_scheme.strip().lower()
    if activation_scheme != "dynamic":
        raise UnsupportedSyncQuantizationError(
            f"Unsupported {context}.activation_scheme={raw_activation_scheme!r}; "
            "online FP8 weight sync supports Slime/SGLang-style dynamic activation scales only"
        )
    normalized["activation_scheme"] = activation_scheme

    raw_scale_fmt = normalized.get("scale_fmt")
    if raw_scale_fmt is not None:
        if not isinstance(raw_scale_fmt, str):
            raise UnsupportedSyncQuantizationError(f"{context}.scale_fmt must be a string when provided")
        scale_fmt = raw_scale_fmt.strip().lower()
        if scale_fmt == "ue8m0":
            raise UnsupportedSyncQuantizationError(
                f"Unsupported {context}.scale_fmt={raw_scale_fmt!r}; "
                "online FP8 weight sync emits standard FP32 block scales in weight_scale_inv tensors. "
                "DeepGEMM/UE8M0 scale storage needs a separate sender/receiver contract and validation gate."
            )
        raise UnsupportedSyncQuantizationError(f"Unsupported {context}.scale_fmt={raw_scale_fmt!r}")

    raw_block_size = normalized.get("weight_block_size", [128, 128])
    if isinstance(raw_block_size, int) and not isinstance(raw_block_size, bool):
        block_size = [raw_block_size, raw_block_size]
    elif (
        isinstance(raw_block_size, Sequence)
        and not isinstance(raw_block_size, (str, bytes))
        and 1 <= len(raw_block_size) <= 2
    ):
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in raw_block_size):
            raise UnsupportedSyncQuantizationError(f"{context}.weight_block_size must contain positive integer values")
        block_size = list(raw_block_size)
        if len(block_size) == 1:
            block_size.append(block_size[0])
    else:
        raise UnsupportedSyncQuantizationError(
            f"{context}.weight_block_size must be a positive int or a one/two-item sequence"
        )
    if any(value <= 0 for value in block_size):
        raise UnsupportedSyncQuantizationError(f"{context}.weight_block_size values must be positive")
    normalized["weight_block_size"] = block_size

    modules_to_not_convert = _normalize_modules_to_not_convert(
        normalized.get("modules_to_not_convert"),
        context=context,
    )
    if modules_to_not_convert is not None:
        normalized["modules_to_not_convert"] = modules_to_not_convert

    return normalized
