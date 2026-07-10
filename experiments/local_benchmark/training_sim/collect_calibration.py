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


def summarize_observed_run(
    run: ObservedRun,
    *,
    warmup_steps: int = 0,
    world_size: int | None = None,
) -> dict[str, Any]:
    ordered_steps = sorted(run.steps, key=lambda row: (row.source, row.step))
    measured = ordered_steps[warmup_steps:]
    tps = [row.tokens_per_sec for row in measured if row.tokens_per_sec is not None]
    tflops = [row.tflops_per_gpu for row in measured if row.tflops_per_gpu is not None]
    mfu = [row.mfu for row in measured if row.mfu is not None]
    step_time = [row.step_time_s for row in measured if row.step_time_s is not None]
    peaks = [row.peak_mem_gb for row in measured if row.peak_mem_gb is not None]

    summary: dict[str, Any] = {
        "sources": run.sources,
        "parsed_step_count": len(run.steps),
        "parsed_phase_count": len(run.phases),
        "parsed_memory_phase_count": len(run.memory_phases),
        "warmup_excluded": warmup_steps,
        "measured_steps": len(measured),
        "tokens_per_sec_mean": _mean(tps),
        "tokens_per_sec_median": _median(tps),
        "tflops_per_gpu_mean": _mean(tflops),
        "mfu_mean": _mean(mfu),
        "step_time_s_mean": _mean(step_time),
        "peak_mem_gb_max": max(peaks) if peaks else None,
    }
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
