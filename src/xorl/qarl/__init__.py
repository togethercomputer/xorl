"""Quantization-aware RL fake-quant modules."""

from xorl.qarl.calibration import calibrate_qarl_model, load_qarl_calibration_batches
from xorl.qarl.fake_quant import (
    QARLLinear,
    inject_qarl_into_model,
    normalize_qarl_quant_cfg,
    qarl_unsupported_scope_reason,
    summarize_qarl_model,
)
from xorl.qarl.sync import qarl_sync_quantization_config


__all__ = [
    "QARLLinear",
    "calibrate_qarl_model",
    "inject_qarl_into_model",
    "load_qarl_calibration_batches",
    "normalize_qarl_quant_cfg",
    "qarl_sync_quantization_config",
    "qarl_unsupported_scope_reason",
    "summarize_qarl_model",
]
