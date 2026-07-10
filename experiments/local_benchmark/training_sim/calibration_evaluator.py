"""Evaluate scenario-prediction fidelity with leave-one-out benchmark holdouts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


try:
    from .benchmark_behavior import load_benchmark_behavior_points, predict_benchmark_behavior
    from .config_fingerprint import load_training_config, resolve_topology
    from .memory_ledger import build_memory_ledger
    from .model_metadata import resolve_model_metadata
    from .scenario_planner import _extrapolate_behavior, _mutated_config, _topology_label
    from .schemas import (
        BenchmarkBehaviorPoint,
        CalibrationHoldout,
        CalibrationReport,
        Topology,
        to_jsonable,
    )
    from .shape_engine import build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import load_benchmark_behavior_points, predict_benchmark_behavior
    from config_fingerprint import load_training_config, resolve_topology
    from memory_ledger import build_memory_ledger
    from model_metadata import resolve_model_metadata
    from scenario_planner import _extrapolate_behavior, _mutated_config, _topology_label
    from schemas import BenchmarkBehaviorPoint, CalibrationHoldout, CalibrationReport, Topology, to_jsonable
    from shape_engine import build_shape_ledger


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if isinstance(value, dict):
        return value
    raw[name] = {}
    return raw[name]


def _point_parallel_size(value: int | None, fallback: int) -> int:
    return value if value is not None else fallback


def _set_if_known(section: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        section[key] = value


def _apply_point_runtime_signature(raw_config: dict[str, Any], point: BenchmarkBehaviorPoint) -> None:
    model = _section(raw_config, "model")
    train = _section(raw_config, "train")
    _set_if_known(model, "deepep_async_combine", point.deepep_async_combine)
    _set_if_known(model, "deepep_num_sms", point.deepep_num_sms)
    _set_if_known(model, "deepep_buffer_size_gb", point.deepep_buffer_size_gb)
    _set_if_known(train, "enable_compile", point.enable_compile)
    _set_if_known(train, "gradient_checkpointing_method", point.gradient_checkpointing_method)
    _set_if_known(train, "enable_activation_offload", point.enable_activation_offload)
    _set_if_known(train, "activation_offload_prefetch_count", point.activation_offload_prefetch_count)


def _topology_for_point(
    base_config: dict[str, Any],
    base_topology: Topology,
    point: BenchmarkBehaviorPoint,
    *,
    world_size: int | None,
    local_world_size: int | None,
) -> tuple[dict[str, Any] | None, Topology | None, str | None]:
    if point.micro_batch_size is None or point.global_batch_size is None:
        return None, None, "missing micro_batch_size/global_batch_size"
    if point.tokens_per_sec is None:
        return None, None, "missing tokens_per_sec"

    resolved_world_size = point.gpu_count or world_size or base_topology.world_size
    resolved_local_world_size = local_world_size or base_topology.local_world_size
    tensor_parallel = _point_parallel_size(point.tensor_parallel_size, base_topology.tensor_parallel_size)
    pipeline_parallel = _point_parallel_size(point.pipeline_parallel_size, base_topology.pipeline_parallel_size)
    ulysses_parallel = _point_parallel_size(point.ulysses_parallel_size, base_topology.ulysses_parallel_size)
    ringattn_parallel = _point_parallel_size(point.ringattn_parallel_size, base_topology.ringattn_parallel_size)
    expert_parallel = _point_parallel_size(point.expert_parallel_size, base_topology.expert_parallel_size)
    non_dp = tensor_parallel * pipeline_parallel * ulysses_parallel * ringattn_parallel
    if non_dp <= 0 or resolved_world_size % non_dp:
        return None, None, "world_size is not divisible by heldout non-DP topology"
    data_parallel_size = resolved_world_size // non_dp
    denominator = point.micro_batch_size * data_parallel_size
    if denominator <= 0 or point.global_batch_size % denominator:
        return None, None, "global_batch_size is not divisible by micro_batch_size * data_parallel_size"
    gradient_accumulation_steps = point.global_batch_size // denominator

    raw_config = _mutated_config(
        base_config,
        world_size=resolved_world_size,
        micro_batch_size=point.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        expert_parallel_size=expert_parallel,
        tensor_parallel_size=tensor_parallel,
        pipeline_parallel_size=pipeline_parallel,
        ulysses_parallel_size=ulysses_parallel,
        ringattn_parallel_size=ringattn_parallel,
    )
    if point.sample_packing_sequence_len is not None:
        _section(raw_config, "data")["sample_packing_sequence_len"] = point.sample_packing_sequence_len
    _apply_point_runtime_signature(raw_config, point)
    try:
        topology = resolve_topology(
            raw_config,
            world_size=resolved_world_size,
            local_world_size=resolved_local_world_size,
        )
    except ValueError as exc:
        return None, None, str(exc)
    if point.ep_fsdp_size is not None and topology.ep_fsdp_size != point.ep_fsdp_size:
        return None, None, "heldout ep_fsdp_size does not match resolved topology"
    return raw_config, topology, None


def _without_point(
    behavior_points: list[BenchmarkBehaviorPoint],
    heldout: BenchmarkBehaviorPoint,
) -> list[BenchmarkBehaviorPoint]:
    return [point for point in behavior_points if not (point.label == heldout.label and point.source == heldout.source)]


def evaluate_calibration(
    base_config_path: str | Path,
    *,
    benchmark_dir: str | Path,
    world_size: int | None = None,
    local_world_size: int | None = None,
    device_memory_limit_gb: float = 80.0,
    memory_safety_factor: float = 1.15,
) -> CalibrationReport:
    base_path = Path(base_config_path)
    benchmark_path = Path(benchmark_dir)
    base_config = load_training_config(base_path)
    base_topology = resolve_topology(base_config, world_size=world_size, local_world_size=local_world_size)
    metadata = resolve_model_metadata(base_config)
    behavior_points = load_benchmark_behavior_points(benchmark_path)
    measured_points = [point for point in behavior_points if point.tokens_per_sec is not None]

    holdouts: list[CalibrationHoldout] = []
    warnings: list[str] = []
    skipped_count = 0
    for heldout in measured_points:
        raw_config, topology, skip_reason = _topology_for_point(
            base_config,
            base_topology,
            heldout,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        if raw_config is None or topology is None:
            skipped_count += 1
            warnings.append(f"skipped {heldout.label}: {skip_reason}")
            continue

        training_points = _without_point(behavior_points, heldout)
        shape = build_shape_ledger(topology, balanced_routing=True)
        exact_prediction = predict_benchmark_behavior(training_points, topology, shape, raw_config)
        if exact_prediction.status == "calibrated":
            prediction = exact_prediction
        else:
            memory = build_memory_ledger(raw_config, topology=topology, model_metadata=metadata)
            prediction, _ = _extrapolate_behavior(
                training_points,
                topology,
                shape,
                raw_config=raw_config,
                device_memory_limit_gb=device_memory_limit_gb,
                memory_safety_factor=memory_safety_factor,
                analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
            )

        predicted = prediction.tokens_per_sec
        absolute_error = None
        absolute_percentage_error = None
        if predicted is not None:
            absolute_error = abs(predicted - heldout.tokens_per_sec)
            absolute_percentage_error = 100.0 * absolute_error / heldout.tokens_per_sec
        holdouts.append(
            CalibrationHoldout(
                label=heldout.label,
                source=heldout.source,
                topology_label=_topology_label(topology),
                actual_tokens_per_sec=heldout.tokens_per_sec,
                predicted_tokens_per_sec=predicted,
                prediction_status=prediction.status,
                matched_label=prediction.matched_label,
                absolute_error_tokens_per_sec=round(absolute_error, 3) if absolute_error is not None else None,
                absolute_percentage_error=round(absolute_percentage_error, 3)
                if absolute_percentage_error is not None
                else None,
                calibrated_from_count=len(training_points),
                warnings=prediction.warnings,
            )
        )

    errors = [
        holdout.absolute_percentage_error for holdout in holdouts if holdout.absolute_percentage_error is not None
    ]
    status_counts: dict[str, int] = {}
    for holdout in holdouts:
        status_counts[holdout.prediction_status] = status_counts.get(holdout.prediction_status, 0) + 1
    status = "ok" if errors else "insufficient_data"
    if holdouts and not errors:
        warnings.append("all holdouts were unscored")

    return CalibrationReport(
        base_config_path=str(base_path),
        benchmark_dir=str(benchmark_path),
        status=status,
        measured_point_count=len(measured_points),
        evaluated_count=len(holdouts),
        skipped_count=skipped_count,
        mean_absolute_percentage_error=round(statistics.fmean(errors), 3) if errors else None,
        median_absolute_percentage_error=round(statistics.median(errors), 3) if errors else None,
        max_absolute_percentage_error=round(max(errors), 3) if errors else None,
        prediction_status_counts=dict(sorted(status_counts.items())),
        holdouts=holdouts,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-world-size", type=int, default=None)
    parser.add_argument("--device-memory-limit-gb", type=float, default=80.0)
    parser.add_argument("--memory-safety-factor", type=float, default=1.15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_calibration(
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
