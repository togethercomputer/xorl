"""Timing ledger construction for prediction reports."""

from __future__ import annotations

import math
from typing import Any


try:
    from .schemas import BenchmarkBehaviorPrediction, TimingLedger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from schemas import BenchmarkBehaviorPrediction, TimingLedger


_FORWARD_PHASES = ("model_forward", "forward", "fwd")
_LOSS_PHASES = ("loss_compute", "loss", "cross_entropy", "ce_loss")
_BACKWARD_PHASES = ("backward", "model_backward", "bwd")
_OPTIMIZER_PHASES = ("optimizer_step", "optimizer", "optim", "clip_and_step")
_INPUT_PHASES = ("dataloader", "get_batch", "input", "microbatch_to_device", "collator", "tokenize")
_FORWARD_BACKWARD_PHASES = ("forward_backward", "forward_backward_total", "fwd_bwd", "fwd_bwd_total")


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for key, item in value.items():
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            output[str(key)] = round(numeric, 6)
    return output


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, 6)


def _source_for_observed(calibration_sources: list[str] | None) -> str:
    if not calibration_sources:
        return "observed_logs"
    if len(calibration_sources) == 1:
        return f"observed_logs:{calibration_sources[0]}"
    return f"observed_logs:{len(calibration_sources)} sources"


def _source_for_benchmark(benchmark_behavior: BenchmarkBehaviorPrediction) -> str:
    if benchmark_behavior.matched_label:
        return f"benchmark_behavior:{benchmark_behavior.matched_label}"
    if benchmark_behavior.source:
        return f"benchmark_behavior:{benchmark_behavior.source}"
    return "benchmark_behavior"


def _phase_value(phase_time_sec: dict[str, float], names: tuple[str, ...]) -> float | None:
    lowered = {phase.lower(): value for phase, value in phase_time_sec.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None:
            return value
    return None


def _phase_bucket(phase: str) -> str:
    lowered = phase.lower()
    if any(part in lowered for part in ("dataloader", "get_batch", "input", "tokenize", "collator")):
        return "input"
    if any(part in lowered for part in ("optimizer", "optim", "clip_and_step", "lr_scheduler", "zero_grad")):
        return "optimizer"
    if any(
        part in lowered
        for part in (
            "sync",
            "all_reduce",
            "reduce_scatter",
            "all_gather",
            "nccl",
            "deepep",
            "dispatch",
            "combine",
            "communication",
            "data_movement",
            "a2a",
            "fsdp",
        )
    ):
        return "communication"
    if lowered in {"fwd", "bwd", "fwd+bwd"} or any(
        part in lowered for part in ("forward", "backward", "loss", "recompute", "moe", "attention")
    ):
        return "model_compute"
    return "other"


def _is_composite_phase_for_bottleneck(phase: str, phases: set[str]) -> bool:
    lowered = phase.lower()
    lowered_phases = {item.lower() for item in phases}
    if lowered == "train_step_total":
        return True
    if lowered in set(_FORWARD_BACKWARD_PHASES):
        return bool(
            lowered_phases
            & {
                *_FORWARD_PHASES,
                *_LOSS_PHASES,
                *_BACKWARD_PHASES,
            }
        )
    if lowered == "clip_and_step_total":
        return bool(
            lowered_phases
            & {
                "clip_gradients",
                "optimizer_step",
                "optimizer",
                "optim",
                "lr_scheduler_step",
            }
        )
    return False


def _derive_phase_share(
    phase_time_sec: dict[str, float], phase_time_share: dict[str, float]
) -> tuple[dict[str, float], str | None]:
    if phase_time_share:
        return phase_time_share, None
    denominator = phase_time_sec.get("train_step_total")
    if denominator is None:
        denominator = sum(value for phase, value in phase_time_sec.items() if phase != "train_step_total")
    if not denominator:
        return {}, None
    derived = {
        phase: round(value / denominator, 6) for phase, value in phase_time_sec.items() if phase != "train_step_total"
    }
    return derived, "phase_time_share=derived_from_phase_time_sec"


def _derive_phase_time_sec(
    phase_time_share: dict[str, float],
    step_time_s: float | None,
) -> tuple[dict[str, float], str | None]:
    if not phase_time_share or step_time_s is None or step_time_s <= 0:
        return {}, None
    derived = {
        phase: round(share * step_time_s, 6) for phase, share in phase_time_share.items() if phase != "train_step_total"
    }
    if derived:
        derived["train_step_total"] = round(step_time_s, 6)
        return derived, "phase_time_sec=derived_from_phase_time_share_and_step_time"
    return {}, None


def _phase_bottleneck(phase_time_share: dict[str, float]) -> tuple[str, float] | None:
    visible = {
        phase: share
        for phase, share in phase_time_share.items()
        if not _is_composite_phase_for_bottleneck(phase, set(phase_time_share))
    }
    if not visible:
        visible = {phase: share for phase, share in phase_time_share.items() if phase != "train_step_total"}
    if not visible:
        return None
    phase = max(visible, key=visible.get)
    return phase, visible[phase]


def _forward_backward_time(
    *,
    phase_time_sec: dict[str, float],
    forward_s: float | None,
    loss_s: float | None,
    backward_s: float | None,
) -> tuple[float | None, str | None]:
    direct = _phase_value(phase_time_sec, _FORWARD_BACKWARD_PHASES)
    if direct is not None:
        return direct, None
    if forward_s is None or backward_s is None:
        return None, None
    total = forward_s + backward_s
    if loss_s is not None:
        total += loss_s
    return round(total, 6), "forward_backward_s=summed_phase_components"


def build_timing_ledger(
    observed_summary: dict[str, Any] | None,
    benchmark_behavior: BenchmarkBehaviorPrediction | None,
    *,
    calibration_sources: list[str] | None = None,
) -> TimingLedger:
    """Build a typed timing ledger from observed logs or matched benchmark behavior."""
    notes: list[str] = []
    source = None
    timing_coverage_status = "no_timing_calibration"
    phase_time_sec: dict[str, float] = {}
    phase_time_share: dict[str, float] = {}
    step_time_s: float | None = None

    observed_phase_time_sec = _float_dict((observed_summary or {}).get("phase_time_sec"))
    observed_phase_time_share = _float_dict((observed_summary or {}).get("phase_time_share"))
    observed_step_time = _float_or_none((observed_summary or {}).get("step_time_s_mean"))
    benchmark_phase_time_sec = _float_dict(benchmark_behavior.phase_time_sec if benchmark_behavior else None)
    benchmark_phase_time_share = _float_dict(benchmark_behavior.phase_time_share if benchmark_behavior else None)
    benchmark_step_time = _float_or_none(benchmark_behavior.step_time_sec if benchmark_behavior else None)
    if observed_phase_time_sec:
        source = _source_for_observed(calibration_sources)
        timing_coverage_status = "observed_phase_timing"
        phase_time_sec = observed_phase_time_sec
        phase_time_share = observed_phase_time_share
        step_time_s = observed_step_time
    elif observed_phase_time_share and observed_step_time is not None:
        source = _source_for_observed(calibration_sources)
        timing_coverage_status = "observed_phase_timing"
        phase_time_share = observed_phase_time_share
        step_time_s = observed_step_time
        phase_time_sec, phase_sec_note = _derive_phase_time_sec(phase_time_share, step_time_s)
        if phase_sec_note is not None:
            notes.append(phase_sec_note)
    elif benchmark_phase_time_sec:
        assert benchmark_behavior is not None
        source = _source_for_benchmark(benchmark_behavior)
        timing_coverage_status = "benchmark_phase_timing"
        phase_time_sec = benchmark_phase_time_sec
        phase_time_share = benchmark_phase_time_share
        step_time_s = benchmark_step_time
    elif benchmark_phase_time_share and benchmark_step_time is not None and benchmark_behavior is not None:
        source = _source_for_benchmark(benchmark_behavior)
        timing_coverage_status = "benchmark_phase_timing"
        phase_time_share = benchmark_phase_time_share
        step_time_s = benchmark_step_time
        phase_time_sec, phase_sec_note = _derive_phase_time_sec(phase_time_share, step_time_s)
        if phase_sec_note is not None:
            notes.append(phase_sec_note)
    else:
        if observed_step_time is not None:
            source = _source_for_observed(calibration_sources)
            timing_coverage_status = "observed_total_step_only"
            step_time_s = observed_step_time
        elif benchmark_step_time is not None and benchmark_behavior is not None:
            source = _source_for_benchmark(benchmark_behavior)
            timing_coverage_status = "benchmark_total_step_only"
            step_time_s = benchmark_step_time

    if step_time_s is None:
        step_time_s = phase_time_sec.get("train_step_total")
    share_note = None
    phase_time_share, share_note = _derive_phase_share(phase_time_sec, phase_time_share)
    if share_note is not None:
        notes.append(share_note)
    if timing_coverage_status.endswith("_total_step_only"):
        notes.append("phase_breakdown_unavailable")

    forward_s = _phase_value(phase_time_sec, _FORWARD_PHASES)
    loss_s = _phase_value(phase_time_sec, _LOSS_PHASES)
    backward_s = _phase_value(phase_time_sec, _BACKWARD_PHASES)
    optimizer_s = _phase_value(phase_time_sec, _OPTIMIZER_PHASES)
    input_s = _phase_value(phase_time_sec, _INPUT_PHASES)
    forward_backward_s, forward_backward_note = _forward_backward_time(
        phase_time_sec=phase_time_sec,
        forward_s=forward_s,
        loss_s=loss_s,
        backward_s=backward_s,
    )
    if forward_backward_note is not None:
        notes.append(forward_backward_note)

    bottleneck = _phase_bottleneck(phase_time_share)
    phase_bottleneck_phase = None
    phase_bottleneck_bucket = None
    phase_bottleneck_share = None
    if bottleneck is not None:
        phase_bottleneck_phase, phase_bottleneck_share = bottleneck
        phase_bottleneck_bucket = _phase_bucket(phase_bottleneck_phase)

    return TimingLedger(
        source=source,
        timing_coverage_status=timing_coverage_status,
        forward_backward_s=forward_backward_s,
        forward_s=forward_s,
        loss_s=loss_s,
        backward_s=backward_s,
        optimizer_s=optimizer_s,
        input_s=input_s,
        step_time_s=step_time_s,
        phase_time_sec=phase_time_sec,
        phase_time_share=phase_time_share,
        phase_bottleneck_phase=phase_bottleneck_phase,
        phase_bottleneck_bucket=phase_bottleneck_bucket,
        phase_bottleneck_share=round(phase_bottleneck_share, 6) if phase_bottleneck_share is not None else None,
        notes=notes,
    )
