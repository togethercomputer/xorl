"""Evaluate whether simulator memory feasibility predicts fit versus OOM rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


try:
    from .benchmark_behavior import (
        behavior_point_model_mismatches,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from .calibration_evaluator import _actual_memory_residual, _topology_for_point, _without_point
    from .calibration_packs import resolve_calibration_pack, resolve_pack_inputs
    from .config_fingerprint import load_training_config, resolve_topology
    from .memory_ledger import build_memory_ledger
    from .model_metadata import resolve_model_metadata
    from .scenario_planner import (
        _calibrated_memory_peak_estimate,
        _candidate_from_prediction,
        _communication_ledger,
        _extrapolate_behavior,
        _memory_ownership_notes,
        _topology_label,
    )
    from .schemas import BenchmarkBehaviorPoint, FeasibilityHoldout, FeasibilityReport, to_jsonable
    from .shape_engine import build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import (
        behavior_point_model_mismatches,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from calibration_evaluator import _actual_memory_residual, _topology_for_point, _without_point
    from calibration_packs import resolve_calibration_pack, resolve_pack_inputs
    from config_fingerprint import load_training_config, resolve_topology
    from memory_ledger import build_memory_ledger
    from model_metadata import resolve_model_metadata
    from scenario_planner import (
        _calibrated_memory_peak_estimate,
        _candidate_from_prediction,
        _communication_ledger,
        _extrapolate_behavior,
        _memory_ownership_notes,
        _topology_label,
    )
    from schemas import BenchmarkBehaviorPoint, FeasibilityHoldout, FeasibilityReport, to_jsonable
    from shape_engine import build_shape_ledger


def _actual_outcome(point: BenchmarkBehaviorPoint) -> str | None:
    if point.correctness_status == "oom":
        return "oom"
    if point.tokens_per_sec is not None:
        return "fit"
    return None


def _has_simulator_only_runtime_mismatch(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any]) -> bool:
    return "attention_backend" in behavior_point_workload_mismatches(point, raw_config)


def _predicted_outcome(feasibility_status: str, risk_flags: list[str]) -> str:
    if feasibility_status.startswith("feasible"):
        memory_sensitive_runtime_mismatch = any(
            flag
            in {
                "runtime_mismatch:activation_offload_prefetch_count",
                "runtime_mismatch:deepep_buffer_size_gb",
                "runtime_mismatch:enable_activation_offload",
                "runtime_mismatch:gradient_checkpointing_method",
                "runtime_mismatch:muon_momentum",
                "runtime_mismatch:skip_param_upcast",
            }
            for flag in risk_flags
        )
        if memory_sensitive_runtime_mismatch and (
            feasibility_status.endswith("_high_pressure")
            or feasibility_status.endswith("_moderate_pressure")
            or "memory_extrapolated_overhead" in risk_flags
        ):
            return "unknown"
        if "real_routing_outside_fit_envelope" in risk_flags and (
            feasibility_status.endswith("_high_pressure")
            or feasibility_status.endswith("_moderate_pressure")
            or "memory_extrapolated_overhead" in risk_flags
        ):
            return "unknown"
        return "fit"
    if feasibility_status == "observed_oom" or feasibility_status.endswith("_exceeds_limit"):
        return "blocked"
    if feasibility_status in {"memory_floor_exceeds_limit", "memory_floor_exceeds_safety_margin"}:
        return "blocked"
    return "unknown"


def _max_holdout_by_field(holdouts: list[FeasibilityHoldout], field_name: str) -> FeasibilityHoldout | None:
    return max(
        (holdout for holdout in holdouts if getattr(holdout, field_name) is not None),
        key=lambda holdout: (getattr(holdout, field_name), holdout.label),
        default=None,
    )


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _classify_correct(actual_outcome: str, predicted_outcome: str) -> bool:
    if actual_outcome == "fit":
        return predicted_outcome == "fit"
    if actual_outcome == "oom":
        return predicted_outcome == "blocked"
    return False


def _fit_recall(holdouts: list[FeasibilityHoldout]) -> float | None:
    fit_holdouts = [holdout for holdout in holdouts if holdout.actual_outcome == "fit"]
    if not fit_holdouts:
        return None
    correct = sum(1 for holdout in fit_holdouts if holdout.classified_correctly)
    return round(correct / len(fit_holdouts), 3)


def _oom_recall(holdouts: list[FeasibilityHoldout]) -> float | None:
    oom_holdouts = [holdout for holdout in holdouts if holdout.actual_outcome == "oom"]
    if not oom_holdouts:
        return None
    correct = sum(1 for holdout in oom_holdouts if holdout.classified_correctly)
    return round(correct / len(oom_holdouts), 3)


def _feasibility_holdout(
    heldout: BenchmarkBehaviorPoint,
    *,
    behavior_points: list[BenchmarkBehaviorPoint],
    base_config: dict[str, Any],
    base_topology,
    world_size: int | None,
    local_world_size: int | None,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
) -> tuple[FeasibilityHoldout | None, str | None]:
    actual_outcome = _actual_outcome(heldout)
    if actual_outcome is None:
        return None, "missing tokens_per_sec and not an OOM row"
    raw_config, topology, skip_reason = _topology_for_point(
        base_config,
        base_topology,
        heldout,
        world_size=world_size,
        local_world_size=local_world_size,
        require_tokens=False,
    )
    if raw_config is None or topology is None:
        return None, skip_reason

    training_points = _without_point(behavior_points, heldout)
    shape = build_shape_ledger(topology, balanced_routing=True)
    metadata = resolve_model_metadata(raw_config)
    memory = build_memory_ledger(raw_config, topology=topology, model_metadata=metadata)
    exact_prediction = predict_benchmark_behavior(training_points, topology, shape, raw_config)
    if exact_prediction.status in {"calibrated", "calibrated_failure"}:
        behavior = exact_prediction
        prediction_confidence = exact_prediction.status
        memory_peak_estimate = None
        notes: list[str] = []
    else:
        memory_peak_estimate = _calibrated_memory_peak_estimate(
            training_points,
            base_config,
            raw_config,
            topology,
            shape,
            metadata,
            default_world_size=base_topology.world_size,
            default_local_world_size=base_topology.local_world_size,
            analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
        )
        behavior, notes = _extrapolate_behavior(
            training_points,
            topology,
            shape,
            raw_config=raw_config,
            device_memory_limit_gb=device_memory_limit_gb,
            memory_safety_factor=memory_safety_factor,
            analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
            memory_peak_estimate=memory_peak_estimate,
        )
        prediction_confidence = behavior.status

    candidate = _candidate_from_prediction(
        label=heldout.label,
        config_path=None,
        topology=topology,
        shape=shape,
        behavior=behavior,
        prediction_confidence=prediction_confidence,
        promotable=False,
        behavior_points=training_points,
        raw_config=raw_config,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
        analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
        memory_peak_estimate=memory_peak_estimate,
        memory_ownership_notes=_memory_ownership_notes(memory),
        communication=_communication_ledger(topology),
        notes=notes,
    )
    predicted_outcome = _predicted_outcome(candidate.feasibility_status, candidate.risk_flags)
    actual_memory_residual_gb, actual_memory_residual_fraction = _actual_memory_residual(
        heldout.peak_mem_gb,
        candidate.analytic_peak_floor_gb,
    )
    return (
        FeasibilityHoldout(
            label=heldout.label,
            source=heldout.source,
            topology_label=_topology_label(topology),
            actual_outcome=actual_outcome,
            predicted_outcome=predicted_outcome,
            actual_tokens_per_sec=heldout.tokens_per_sec,
            actual_peak_mem_gb=heldout.peak_mem_gb,
            predicted_tokens_per_sec=behavior.tokens_per_sec,
            predicted_peak_mem_gb=candidate.estimated_peak_mem_gb,
            prediction_status=behavior.status,
            matched_label=behavior.matched_label,
            memory_prediction_basis=candidate.memory_basis,
            analytic_peak_floor_gb=candidate.analytic_peak_floor_gb,
            memory_coverage_status=candidate.memory_coverage_status,
            predicted_memory_residual_gb=candidate.estimated_memory_residual_gb,
            predicted_memory_residual_fraction_of_peak=candidate.estimated_memory_residual_fraction_of_peak,
            actual_memory_residual_gb=actual_memory_residual_gb,
            actual_memory_residual_fraction_of_peak=actual_memory_residual_fraction,
            memory_calibration_source=candidate.memory_calibration_source,
            predicted_feasibility_status=candidate.feasibility_status,
            classified_correctly=_classify_correct(actual_outcome, predicted_outcome),
            calibrated_from_count=len(training_points),
            memory_calibration_notes=candidate.memory_calibration_notes,
            risk_flags=candidate.risk_flags,
            warnings=behavior.warnings,
        ),
        None,
    )


def evaluate_feasibility(
    base_config_path: str | Path,
    *,
    benchmark_dir: str | Path,
    world_size: int | None = None,
    local_world_size: int | None = None,
    device_memory_limit_gb: float = 80.0,
    memory_safety_factor: float = 1.15,
) -> FeasibilityReport:
    base_path = Path(base_config_path)
    benchmark_path = resolve_calibration_pack(benchmark_dir)
    base_config = load_training_config(base_path)
    base_topology = resolve_topology(base_config, world_size=world_size, local_world_size=local_world_size)
    behavior_points = load_benchmark_behavior_points(benchmark_path)
    observed_points = [
        point
        for point in behavior_points
        if _actual_outcome(point) is not None and not _has_simulator_only_runtime_mismatch(point, base_config)
    ]

    holdouts: list[FeasibilityHoldout] = []
    warnings: list[str] = []
    skipped_count = 0
    for heldout in observed_points:
        if behavior_point_model_mismatches(heldout, base_config):
            skipped_count += 1
            warnings.append(f"skipped {heldout.label}: model_ref mismatch")
            continue
        holdout, skip_reason = _feasibility_holdout(
            heldout,
            behavior_points=behavior_points,
            base_config=base_config,
            base_topology=base_topology,
            world_size=world_size,
            local_world_size=local_world_size,
            device_memory_limit_gb=device_memory_limit_gb,
            memory_safety_factor=memory_safety_factor,
        )
        if holdout is None:
            skipped_count += 1
            warnings.append(f"skipped {heldout.label}: {skip_reason}")
            continue
        holdouts.append(holdout)

    correct_count = sum(1 for holdout in holdouts if holdout.classified_correctly)
    actual_fit_count = sum(1 for holdout in holdouts if holdout.actual_outcome == "fit")
    actual_oom_count = sum(1 for holdout in holdouts if holdout.actual_outcome == "oom")
    predicted_fit_count = sum(1 for holdout in holdouts if holdout.predicted_outcome == "fit")
    predicted_blocked_count = sum(1 for holdout in holdouts if holdout.predicted_outcome == "blocked")
    predicted_unknown_count = sum(1 for holdout in holdouts if holdout.predicted_outcome == "unknown")
    false_fit_count = sum(
        1 for holdout in holdouts if holdout.actual_outcome == "oom" and holdout.predicted_outcome == "fit"
    )
    false_blocked_count = sum(
        1 for holdout in holdouts if holdout.actual_outcome == "fit" and holdout.predicted_outcome == "blocked"
    )
    if predicted_unknown_count:
        warnings.append(f"{predicted_unknown_count} feasibility holdouts were predicted unknown")
    predicted_memory_residuals = [
        holdout.predicted_memory_residual_gb for holdout in holdouts if holdout.predicted_memory_residual_gb is not None
    ]
    predicted_memory_residual_fractions = [
        holdout.predicted_memory_residual_fraction_of_peak
        for holdout in holdouts
        if holdout.predicted_memory_residual_fraction_of_peak is not None
    ]
    actual_memory_residuals = [
        holdout.actual_memory_residual_gb for holdout in holdouts if holdout.actual_memory_residual_gb is not None
    ]
    actual_memory_residual_fractions = [
        holdout.actual_memory_residual_fraction_of_peak
        for holdout in holdouts
        if holdout.actual_memory_residual_fraction_of_peak is not None
    ]
    max_predicted_memory_residual_holdout = _max_holdout_by_field(holdouts, "predicted_memory_residual_gb")
    max_predicted_memory_residual_fraction_holdout = _max_holdout_by_field(
        holdouts,
        "predicted_memory_residual_fraction_of_peak",
    )
    max_actual_memory_residual_holdout = _max_holdout_by_field(holdouts, "actual_memory_residual_gb")
    max_actual_memory_residual_fraction_holdout = _max_holdout_by_field(
        holdouts,
        "actual_memory_residual_fraction_of_peak",
    )

    return FeasibilityReport(
        base_config_path=str(base_path),
        benchmark_dir=str(benchmark_path),
        status="ok" if holdouts else "insufficient_data",
        observed_point_count=len(observed_points),
        evaluated_count=len(holdouts),
        skipped_count=skipped_count,
        actual_fit_count=actual_fit_count,
        actual_oom_count=actual_oom_count,
        predicted_fit_count=predicted_fit_count,
        predicted_blocked_count=predicted_blocked_count,
        predicted_unknown_count=predicted_unknown_count,
        correct_count=correct_count,
        false_fit_count=false_fit_count,
        false_blocked_count=false_blocked_count,
        accuracy=round(correct_count / len(holdouts), 3) if holdouts else None,
        fit_recall=_fit_recall(holdouts),
        oom_recall=_oom_recall(holdouts),
        prediction_status_counts=_count_values([holdout.prediction_status for holdout in holdouts]),
        memory_prediction_basis_counts=_count_values([holdout.memory_prediction_basis for holdout in holdouts]),
        memory_coverage_status_counts=_count_values([holdout.memory_coverage_status for holdout in holdouts]),
        feasibility_status_counts=_count_values([holdout.predicted_feasibility_status for holdout in holdouts]),
        max_predicted_memory_residual_gb=(
            round(max(predicted_memory_residuals), 3) if predicted_memory_residuals else None
        ),
        max_predicted_memory_residual_gb_label=(
            max_predicted_memory_residual_holdout.label if max_predicted_memory_residual_holdout is not None else None
        ),
        max_predicted_memory_residual_fraction_of_peak=(
            round(max(predicted_memory_residual_fractions), 3) if predicted_memory_residual_fractions else None
        ),
        max_predicted_memory_residual_fraction_of_peak_label=(
            max_predicted_memory_residual_fraction_holdout.label
            if max_predicted_memory_residual_fraction_holdout is not None
            else None
        ),
        max_actual_memory_residual_gb=round(max(actual_memory_residuals), 3) if actual_memory_residuals else None,
        max_actual_memory_residual_gb_label=(
            max_actual_memory_residual_holdout.label if max_actual_memory_residual_holdout is not None else None
        ),
        max_actual_memory_residual_fraction_of_peak=(
            round(max(actual_memory_residual_fractions), 3) if actual_memory_residual_fractions else None
        ),
        max_actual_memory_residual_fraction_of_peak_label=(
            max_actual_memory_residual_fraction_holdout.label
            if max_actual_memory_residual_fraction_holdout is not None
            else None
        ),
        risk_flag_counts=_count_values([flag for holdout in holdouts for flag in holdout.risk_flags]),
        holdouts=holdouts,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", help="Built-in calibration-pack name")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-world-size", type=int, default=None)
    parser.add_argument("--device-memory-limit-gb", type=float, default=80.0)
    parser.add_argument("--memory-safety-factor", type=float, default=1.15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    args.config, args.benchmark_dir = resolve_pack_inputs(args.pack, args.config, args.benchmark_dir)
    if args.config is None or args.benchmark_dir is None:
        parser.error("provide --pack, or both --config and --benchmark-dir")

    report = evaluate_feasibility(
        args.config,
        benchmark_dir=args.benchmark_dir,
        world_size=args.world_size,
        local_world_size=args.local_world_size,
        device_memory_limit_gb=args.device_memory_limit_gb,
        memory_safety_factor=args.memory_safety_factor,
    )
    rendered = json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
