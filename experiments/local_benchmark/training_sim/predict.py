"""Emit a static simulator report for one XoRL training config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


try:
    from .benchmark_behavior import load_benchmark_behavior_points, predict_benchmark_behavior
    from .collect_calibration import merge_observed_runs, parse_log_path, summarize_observed_run
    from .config_fingerprint import build_fingerprint, load_training_config
    from .memory_ledger import build_memory_ledger
    from .schemas import PredictionReport, to_jsonable
    from .shape_engine import build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import load_benchmark_behavior_points, predict_benchmark_behavior
    from collect_calibration import merge_observed_runs, parse_log_path, summarize_observed_run
    from config_fingerprint import build_fingerprint, load_training_config
    from memory_ledger import build_memory_ledger
    from schemas import PredictionReport, to_jsonable
    from shape_engine import build_shape_ledger


def build_report(
    config_path: str | Path,
    *,
    world_size: int | None,
    local_world_size: int | None,
    balanced_routing: bool,
    num_experts: int | None,
    top_k: int | None,
    log_paths: list[Path] | None = None,
    warmup_steps: int = 0,
    benchmark_dir: Path | None = None,
) -> PredictionReport:
    fingerprint = build_fingerprint(
        config_path,
        world_size=world_size,
        local_world_size=local_world_size,
        balanced_routing=balanced_routing,
        num_experts=num_experts,
        top_k=top_k,
    )
    raw_config = load_training_config(config_path)
    shape = build_shape_ledger(fingerprint.topology, balanced_routing=balanced_routing)
    observed = None
    observed_summary: dict[str, Any] | None = None
    calibration_sources: list[str] = []
    if log_paths:
        observed = merge_observed_runs(parse_log_path(path) for path in log_paths)
        calibration_sources = observed.sources
        observed_summary = summarize_observed_run(
            observed,
            warmup_steps=warmup_steps,
            world_size=fingerprint.topology.world_size,
        )

    memory = build_memory_ledger(
        raw_config,
        observed,
        topology=fingerprint.topology,
        model_metadata=fingerprint.model_metadata,
    )
    benchmark_behavior = None
    if benchmark_dir is not None:
        behavior_points = load_benchmark_behavior_points(benchmark_dir)
        benchmark_behavior = predict_benchmark_behavior(behavior_points, fingerprint.topology, shape, raw_config)
    warnings = list(shape.warnings)
    if memory.observed_peak_mem_gb_max is None:
        warnings.append("no observed memory calibration was supplied")
    if benchmark_behavior is not None:
        warnings.extend(benchmark_behavior.warnings)

    return PredictionReport(
        fingerprint=fingerprint,
        shape=shape,
        memory=memory,
        benchmark_behavior=benchmark_behavior,
        observed_summary=observed_summary,
        calibration_sources=calibration_sources,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="XoRL YAML config")
    parser.add_argument("--world-size", type=int, default=None, help="Override WORLD_SIZE for config resolution")
    parser.add_argument("--local-world-size", type=int, default=None, help="Override LOCAL_WORLD_SIZE")
    parser.add_argument("--balanced-routing", action="store_true", help="Assume deterministic balanced MoE routing")
    parser.add_argument("--num-experts", type=int, default=None, help="Override model num_experts when config omits it")
    parser.add_argument("--top-k", type=int, default=None, help="Override model top-k routing when config omits it")
    parser.add_argument("--logs", nargs="*", type=Path, default=None, help="Optional trainer logs for calibration")
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Drop this many parsed [STEP] rows from log summary"
    )
    parser.add_argument("--benchmark-dir", type=Path, default=None, help="Optional benchmark recipe directory")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    report = build_report(
        args.config,
        world_size=args.world_size,
        local_world_size=args.local_world_size,
        balanced_routing=args.balanced_routing,
        num_experts=args.num_experts,
        top_k=args.top_k,
        log_paths=args.logs,
        warmup_steps=args.warmup_steps,
        benchmark_dir=args.benchmark_dir,
    )
    rendered = json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
