"""Parse XoRL trainer structured logs into calibration observations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


try:
    from .schemas import MemoryPhaseObservation, ObservedRun, PhaseObservation, StepObservation, to_jsonable
except ImportError:  # pragma: no cover - exercised by direct script execution
    from schemas import MemoryPhaseObservation, ObservedRun, PhaseObservation, StepObservation, to_jsonable


STEP_RE = re.compile(r"\[STEP\s+(?P<step>\d+)/(?P<max>[^\]]+)\]\s+(?P<body>.*)")
PHASE_RE = re.compile(r"\[(?P<prefix>STEP_PHASES(?:_PARTIAL)?)\s+(?P<step>\d+)/(?P<max>[^\]]+)\]\s+(?P<body>.*)")
MEMORY_RE = re.compile(r"\[(?P<prefix>STEP_MEMORY(?:_PARTIAL)?)\s+(?P<step>\d+)/(?P<max>[^\]]+)\]\s+(?P<body>.*)")
KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_+./-]+)=(?P<value>\S+)")


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().rstrip(",")
    for suffix in ("GB", "gb", "s"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_metric_body(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in KV_RE.finditer(body):
        numeric = _float_or_none(match.group("value"))
        if numeric is not None:
            metrics[match.group("key")] = numeric
    return metrics


def _step_from_match(match: re.Match[str], source: str) -> StepObservation:
    metrics = _parse_metric_body(match.group("body"))
    phase_memory: dict[str, float] = {}
    for key in ("fwd", "bwd", "optim", "fwd+bwd", "offload"):
        if key in metrics:
            phase_memory[key] = metrics[key]

    known_keys = {
        "loss",
        "grad_norm",
        "lr",
        "tflops",
        "mfu",
        "tokens_per_sec",
        "time",
        "peak_mem",
        "fwd",
        "bwd",
        "optim",
        "fwd+bwd",
        "offload",
    }
    extra = {key: value for key, value in metrics.items() if key not in known_keys}
    return StepObservation(
        source=source,
        step=int(match.group("step")),
        max_steps=match.group("max"),
        loss=metrics.get("loss"),
        grad_norm=metrics.get("grad_norm"),
        lr=metrics.get("lr"),
        tflops_per_gpu=metrics.get("tflops"),
        mfu=metrics.get("mfu"),
        tokens_per_sec=metrics.get("tokens_per_sec"),
        step_time_s=metrics.get("time"),
        peak_mem_gb=metrics.get("peak_mem"),
        phase_memory_gb=phase_memory,
        extra=extra,
    )


def parse_log_text(text: str, *, source: str = "<text>") -> ObservedRun:
    steps: list[StepObservation] = []
    phases: list[PhaseObservation] = []
    memory_phases: list[MemoryPhaseObservation] = []

    for line in text.splitlines():
        if phase_match := PHASE_RE.search(line):
            phases.append(
                PhaseObservation(
                    source=source,
                    prefix=phase_match.group("prefix"),
                    step=int(phase_match.group("step")),
                    max_steps=phase_match.group("max"),
                    metrics=_parse_metric_body(phase_match.group("body")),
                )
            )
            continue

        if memory_match := MEMORY_RE.search(line):
            memory_phases.append(
                MemoryPhaseObservation(
                    source=source,
                    prefix=memory_match.group("prefix"),
                    step=int(memory_match.group("step")),
                    max_steps=memory_match.group("max"),
                    metrics=_parse_metric_body(memory_match.group("body")),
                )
            )
            continue

        if step_match := STEP_RE.search(line):
            steps.append(_step_from_match(step_match, source))

    return ObservedRun(sources=[source], steps=steps, phases=phases, memory_phases=memory_phases)


def parse_log_path(path: str | Path) -> ObservedRun:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return parse_log_text(text, source=str(log_path))


def merge_observed_runs(runs: Iterable[ObservedRun]) -> ObservedRun:
    sources: list[str] = []
    steps: list[StepObservation] = []
    phases: list[PhaseObservation] = []
    memory_phases: list[MemoryPhaseObservation] = []
    for run in runs:
        sources.extend(run.sources)
        steps.extend(run.steps)
        phases.extend(run.phases)
        memory_phases.extend(run.memory_phases)
    return ObservedRun(sources=sources, steps=steps, phases=phases, memory_phases=memory_phases)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _coefficient_of_variation(values: list[float]) -> float | None:
    mean = _mean(values)
    stdev = _stdev(values)
    if mean in (None, 0.0) or stdev is None:
        return None
    return stdev / mean


def _phase_metric_name(key: str, suffix: str) -> str | None:
    if not key.endswith(suffix):
        return None
    return key[: -len(suffix)]


def _is_composite_phase_for_bottleneck(phase: str, phases: set[str]) -> bool:
    lowered = phase.lower()
    lowered_phases = {item.lower() for item in phases}
    if lowered == "train_step_total":
        return True
    if lowered in {"forward_backward", "forward_backward_total", "fwd_bwd", "fwd_bwd_total"}:
        return bool(
            lowered_phases
            & {
                "model_forward",
                "forward",
                "fwd",
                "loss_compute",
                "loss",
                "backward",
                "model_backward",
                "bwd",
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


def _phase_timing_summary(
    run: ObservedRun,
    *,
    measured_step_keys: set[tuple[str, int]],
    warmup_steps: int,
) -> dict[str, Any]:
    if measured_step_keys:
        phase_rows = [row for row in run.phases if (row.source, row.step) in measured_step_keys]
    else:
        phase_rows = sorted(run.phases, key=lambda row: (row.source, row.step))[warmup_steps:]
    max_values: dict[str, list[float]] = {}
    mean_values: dict[str, list[float]] = {}
    for row in phase_rows:
        for key, value in row.metrics.items():
            if phase := _phase_metric_name(key, "_max_s"):
                max_values.setdefault(phase, []).append(value)
            elif phase := _phase_metric_name(key, "_mean_s"):
                mean_values.setdefault(phase, []).append(value)

    phase_time_sec: dict[str, float] = {}
    phase_time_max_sec: dict[str, float] = {}
    phase_time_rank_mean_sec: dict[str, float] = {}
    for phase in sorted(set(max_values) | set(mean_values)):
        values = max_values.get(phase) or mean_values.get(phase) or []
        if not values:
            continue
        # Median over post-warmup steps (stable profile): straggler steps a 2-step warmup cannot
        # catch (e.g. a 0.586s optimizer step in a 0.20s-steady run) contaminate a mean read and
        # made the phase pins disagree with the stable-profile reader; max stays available below.
        phase_time_sec[phase] = statistics.median(values)
        phase_time_max_sec[phase] = max(values)
        # Cross-rank MEAN companion (same median-over-steps profile): phase_time_sec is the
        # cross-rank MAX convention, so balanced-rank term comparisons need the mean to separate
        # rank asymmetry (routing-imbalance stragglers) from term error.
        rank_mean_rows = mean_values.get(phase) or []
        if rank_mean_rows:
            phase_time_rank_mean_sec[phase] = statistics.median(rank_mean_rows)

    denominator = phase_time_sec.get("train_step_total")
    if denominator is None:
        denominator = sum(value for phase, value in phase_time_sec.items() if phase != "train_step_total")
    phase_time_share = {
        phase: value / denominator
        for phase, value in phase_time_sec.items()
        if denominator and phase != "train_step_total"
    }
    bottleneck_phase = None
    bottleneck_candidates = {
        phase: value
        for phase, value in phase_time_share.items()
        if not _is_composite_phase_for_bottleneck(phase, set(phase_time_share))
    }
    if bottleneck_candidates:
        bottleneck_phase = max(bottleneck_candidates, key=bottleneck_candidates.get)
    elif phase_time_share:
        bottleneck_phase = max(phase_time_share, key=phase_time_share.get)

    return {
        "phase_time_sec": phase_time_sec,
        "phase_time_max_sec": phase_time_max_sec,
        "phase_time_rank_mean_sec": phase_time_rank_mean_sec,
        "phase_time_share": phase_time_share,
        "phase_bottleneck": bottleneck_phase,
    }


def _phase_memory_metric_name(key: str) -> str | None:
    for suffix in (
        "_phase_peak_allocated_max_gb",
        "_phase_peak_reserved_max_gb",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    if "_after_" in key or "_delta_" in key:
        return None
    for suffix in ("_allocated_max_gb", "_reserved_max_gb"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _phase_memory_delta_allows_peak(metrics: dict[str, float], phase: str) -> bool:
    delta_values = [
        value
        for key, value in metrics.items()
        if key.startswith(f"{phase}_delta_") and key.endswith(("_allocated_max_gb", "_reserved_max_gb"))
    ]
    return not delta_values or max(delta_values) > 0.0


def _phase_memory_summary(
    run: ObservedRun,
    *,
    measured_step_keys: set[tuple[str, int]],
    warmup_steps: int,
    peak_mem_gb_max: float | None,
) -> dict[str, Any]:
    if measured_step_keys:
        step_rows = [row for row in run.steps if (row.source, row.step) in measured_step_keys]
        memory_rows = [row for row in run.memory_phases if (row.source, row.step) in measured_step_keys]
    else:
        step_rows = sorted(run.steps, key=lambda row: (row.source, row.step))[warmup_steps:]
        memory_rows = sorted(run.memory_phases, key=lambda row: (row.source, row.step))[warmup_steps:]

    peak_values: dict[str, list[float]] = {}
    for row in step_rows:
        for phase, value in row.phase_memory_gb.items():
            peak_values.setdefault(phase, []).append(value)
    for row in memory_rows:
        for key, value in row.metrics.items():
            if phase := _phase_memory_metric_name(key):
                if not _phase_memory_delta_allows_peak(row.metrics, phase):
                    continue
                peak_values.setdefault(phase, []).append(value)

    phase_memory_peak_gb = {phase: max(values) for phase, values in sorted(peak_values.items()) if values}
    phase_memory_fraction_of_peak = {
        phase: value / peak_mem_gb_max
        for phase, value in phase_memory_peak_gb.items()
        if peak_mem_gb_max not in (None, 0.0)
    }
    memory_bottleneck_phase = None
    memory_bottleneck_candidates = {
        phase: value
        for phase, value in phase_memory_peak_gb.items()
        if not _is_composite_phase_for_bottleneck(phase, set(phase_memory_peak_gb))
    }
    if memory_bottleneck_candidates:
        memory_bottleneck_phase = max(memory_bottleneck_candidates.items(), key=lambda item: (item[1], item[0]))[0]
    elif phase_memory_peak_gb:
        memory_bottleneck_phase = max(phase_memory_peak_gb.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "phase_memory_peak_gb": phase_memory_peak_gb,
        "phase_memory_fraction_of_peak": phase_memory_fraction_of_peak,
        "memory_bottleneck_phase": memory_bottleneck_phase,
    }


def summarize_observed_run(
    run: ObservedRun,
    *,
    warmup_steps: int = 0,
    world_size: int | None = None,
) -> dict[str, Any]:
    ordered_steps = sorted(run.steps, key=lambda row: (row.source, row.step))
    measured = ordered_steps[warmup_steps:]
    measured_step_keys = {(row.source, row.step) for row in measured}
    tps = [row.tokens_per_sec for row in measured if row.tokens_per_sec is not None]
    tflops = [row.tflops_per_gpu for row in measured if row.tflops_per_gpu is not None]
    mfu = [row.mfu for row in measured if row.mfu is not None]
    step_time = [row.step_time_s for row in measured if row.step_time_s is not None]
    peaks = [row.peak_mem_gb for row in measured if row.peak_mem_gb is not None]
    # Realized tokens per step, computed PER STEP (tokens_per_sec x step_time of the SAME step) before
    # aggregating: mean(tps) x median(step) mixes bases and mean(tps) x mean(step) carries the
    # tps/step-time anti-correlation bias (a straggler step drags mean tps down while inflating mean
    # time) — both misprice the realized token load that per-step phase predictions consume.
    tokens_per_step = [
        row.tokens_per_sec * row.step_time_s
        for row in measured
        if row.tokens_per_sec is not None and row.step_time_s is not None
    ]

    summary: dict[str, Any] = {
        "sources": run.sources,
        "parsed_step_count": len(run.steps),
        "parsed_phase_count": len(run.phases),
        "parsed_memory_phase_count": len(run.memory_phases),
        "warmup_excluded": warmup_steps,
        "measured_steps": len(measured),
        "tokens_per_sec_mean": _mean(tps),
        "tokens_per_sec_median": _median(tps),
        "tokens_per_sec_std": _stdev(tps),
        "tokens_per_sec_cv": _coefficient_of_variation(tps),
        "tflops_per_gpu_mean": _mean(tflops),
        "mfu_mean": _mean(mfu),
        "step_time_s_mean": _mean(step_time),
        "step_time_s_std": _stdev(step_time),
        "step_time_s_cv": _coefficient_of_variation(step_time),
        "tokens_per_step_median": _median(tokens_per_step),
        "tokens_per_step_mean": _mean(tokens_per_step),
        "peak_mem_gb_max": max(peaks) if peaks else None,
    }
    summary.update(_phase_timing_summary(run, measured_step_keys=measured_step_keys, warmup_steps=warmup_steps))
    summary.update(
        _phase_memory_summary(
            run,
            measured_step_keys=measured_step_keys,
            warmup_steps=warmup_steps,
            peak_mem_gb_max=summary["peak_mem_gb_max"],
        )
    )
    if world_size and summary["tokens_per_sec_mean"] is not None:
        summary["tokens_per_sec_per_gpu_mean"] = summary["tokens_per_sec_mean"] / world_size
    if measured:
        summary["first_measured_step"] = measured[0].step
        summary["last_measured_step"] = measured[-1].step
        summary["loss_last"] = measured[-1].loss
    return summary


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(child for child in path.rglob("*") if child.is_file()))
        else:
            expanded.append(path)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Log files or directories to parse")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Drop this many parsed [STEP] rows from summary")
    parser.add_argument("--world-size", type=int, default=None, help="Optional GPU count for per-GPU throughput")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to this path")
    parser.add_argument("--include-rows", action="store_true", help="Include parsed row details, not just the summary")
    args = parser.parse_args()

    runs = [parse_log_path(path) for path in _expand_paths(args.paths)]
    observed = merge_observed_runs(runs)
    payload: dict[str, Any] = {
        "summary": summarize_observed_run(observed, warmup_steps=args.warmup_steps, world_size=args.world_size)
    }
    if args.include_rows:
        payload["observed"] = to_jsonable(observed)

    rendered = json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
