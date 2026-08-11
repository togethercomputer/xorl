"""Evaluate scenario-prediction fidelity with leave-one-out benchmark holdouts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


try:
    from .benchmark_behavior import (
        behavior_point_model_mismatches,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from .calibration_packs import resolve_calibration_pack, resolve_pack_inputs
    from .config_fingerprint import load_training_config, resolve_topology
    from .memory_ledger import build_memory_ledger
    from .model_metadata import resolve_model_metadata
    from .runtime_config import runtime_training_config
    from .scenario_planner import (
        _apply_config_override,
        _calibrated_memory_peak_estimate,
        _calibration_distance,
        _calibration_scope,
        _candidate_risk_flags,
        _extrapolate_behavior,
        _measurement_config_filename,
        _memory_coverage_for_candidate,
        _memory_factor,
        _mutated_config,
        _phase_bucket,
        _prediction_interval,
        _prediction_uncertainty_fraction,
        _topology_label,
        _workload_design_variants,
    )
    from .schemas import (
        BenchmarkBehaviorPoint,
        CalibrationHoldout,
        CalibrationReport,
        CalibrationValidationGap,
        ScenarioMeasurementConfig,
        Topology,
        to_jsonable,
    )
    from .shape_engine import build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import (
        behavior_point_model_mismatches,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from calibration_packs import resolve_calibration_pack, resolve_pack_inputs
    from config_fingerprint import load_training_config, resolve_topology
    from memory_ledger import build_memory_ledger
    from model_metadata import resolve_model_metadata
    from runtime_config import runtime_training_config
    from scenario_planner import (
        _apply_config_override,
        _calibrated_memory_peak_estimate,
        _calibration_distance,
        _calibration_scope,
        _candidate_risk_flags,
        _extrapolate_behavior,
        _measurement_config_filename,
        _memory_coverage_for_candidate,
        _memory_factor,
        _mutated_config,
        _phase_bucket,
        _prediction_interval,
        _prediction_uncertainty_fraction,
        _topology_label,
        _workload_design_variants,
    )
    from schemas import (
        BenchmarkBehaviorPoint,
        CalibrationHoldout,
        CalibrationReport,
        CalibrationValidationGap,
        ScenarioMeasurementConfig,
        Topology,
        to_jsonable,
    )
    from shape_engine import build_shape_ledger


_THROUGHPUT_MAPE_TARGET_PERCENT = 8.0
_MEMORY_ABS_ERROR_TARGET_GB = 2.0
_MEMORY_RELATIVE_ERROR_TARGET_FRACTION = 0.03
_PHASE_SHARE_ABS_ERROR_TARGET = 0.10

_CALIBRATION_COMPONENT_TIMING_OVERRIDES = (
    "train.enable_step_phase_timing=true",
    "train.enable_per_component_timing=true",
    "train.step_phase_timing_sync_cuda=true",
)

_CALIBRATION_MEMORY_PROFILE_OVERRIDES = (
    "train.enable_step_phase_timing=true",
    "train.enable_step_memory_profiling=true",
)

_CALIBRATION_REPLAY_OVERRIDES_BY_MEASUREMENT = {
    "collect_phase_timing_for_calibration_holdouts": _CALIBRATION_COMPONENT_TIMING_OVERRIDES,
    "replay_high_error_holdouts_with_component_timing": _CALIBRATION_COMPONENT_TIMING_OVERRIDES,
    "replay_phase_bottleneck_holdouts_with_component_timing": _CALIBRATION_COMPONENT_TIMING_OVERRIDES,
    "replay_phase_top3_holdouts_with_component_timing": _CALIBRATION_COMPONENT_TIMING_OVERRIDES,
    "replay_phase_share_holdouts_with_component_timing": _CALIBRATION_COMPONENT_TIMING_OVERRIDES,
    "collect_peak_memory_for_calibration_holdouts": _CALIBRATION_MEMORY_PROFILE_OVERRIDES,
    "replay_high_memory_error_holdouts_with_memory_profile": _CALIBRATION_MEMORY_PROFILE_OVERRIDES,
    "replay_memory_bottleneck_holdouts_with_phase_memory_profile": _CALIBRATION_MEMORY_PROFILE_OVERRIDES,
}

_CALIBRATION_REPLAY_KIND_BY_MEASUREMENT = {
    "collect_phase_timing_for_calibration_holdouts": "phase_timing",
    "replay_high_error_holdouts_with_component_timing": "component_timing",
    "replay_phase_bottleneck_holdouts_with_component_timing": "component_timing",
    "replay_phase_top3_holdouts_with_component_timing": "component_timing",
    "replay_phase_share_holdouts_with_component_timing": "component_timing",
    "collect_peak_memory_for_calibration_holdouts": "memory_profile",
    "replay_high_memory_error_holdouts_with_memory_profile": "memory_profile",
    "replay_memory_bottleneck_holdouts_with_phase_memory_profile": "memory_profile",
}

_CALIBRATION_HOLDOUT_REPLAY_KIND_BY_MEASUREMENT = {
    "add_scored_same_context_calibration_holdouts": "scored_same_context",
    "add_supported_same_context_calibration_holdouts": "supported_same_context",
}

_CALIBRATION_NEARBY_HOLDOUT_KIND_BY_MEASUREMENT = {
    "add_holdouts_to_recalibrate_prediction_interval": "interval_recalibration",
    "add_nearby_calibration_holdouts_to_tighten_uncertainty": "nearby_holdout",
}


@dataclass(frozen=True)
class _CalibrationMemoryPeakEstimate:
    peak_gb: float
    overhead_gb: float
    source_label: str
    notes: list[str]
    basis: str = "calibration_residual_floor_peak"


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
    simulator = raw_config.setdefault("simulator", {})
    if point.model_ref is not None:
        for key in ("config_path", "model_path", "model_name"):
            if key in model:
                model[key] = point.model_ref
        model.setdefault("model_path", point.model_ref)
    _set_if_known(model, "deepep_async_combine", point.deepep_async_combine)
    _set_if_known(model, "deepep_num_sms", point.deepep_num_sms)
    _set_if_known(model, "deepep_buffer_size_gb", point.deepep_buffer_size_gb)
    _set_if_known(train, "enable_compile", point.enable_compile)
    _set_if_known(train, "gradient_checkpointing_method", point.gradient_checkpointing_method)
    _set_if_known(train, "enable_activation_offload", point.enable_activation_offload)
    _set_if_known(train, "activation_offload_prefetch_count", point.activation_offload_prefetch_count)
    _set_if_known(train, "skip_param_upcast", point.skip_param_upcast)
    _set_if_known(train, "fsdp_reduce_dtype", point.fsdp_reduce_dtype)
    _set_if_known(train, "ce_mode", point.ce_mode)
    _set_if_known(model, "moe_implementation", point.moe_implementation)
    _set_if_known(train, "muon_momentum", point.muon_momentum)
    _set_if_known(train, "muon_update_dtype", point.muon_update_dtype)
    if isinstance(simulator, dict):
        _set_if_known(simulator, "balanced_routing", point.balanced_routing)
        _set_if_known(simulator, "attention_backend", point.attention_backend)


def _topology_for_point(
    base_config: dict[str, Any],
    base_topology: Topology,
    point: BenchmarkBehaviorPoint,
    *,
    world_size: int | None,
    local_world_size: int | None,
    require_tokens: bool = True,
) -> tuple[dict[str, Any] | None, Topology | None, str | None]:
    if point.micro_batch_size is None or point.global_batch_size is None:
        return None, None, "missing micro_batch_size/global_batch_size"
    if require_tokens and point.tokens_per_sec is None:
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
    if denominator <= 0:
        return None, None, "micro_batch_size * data_parallel_size must be positive"
    if point.gradient_accumulation_steps is not None:
        gradient_accumulation_steps = point.gradient_accumulation_steps
        expected_global_batch_size = denominator * gradient_accumulation_steps
        if expected_global_batch_size != point.global_batch_size:
            return None, None, "explicit gradient_accumulation_steps does not match global_batch_size"
    else:
        if point.global_batch_size % denominator:
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
        data_parallel_replicate_size=point.data_parallel_replicate_size,
        data_parallel_shard_size=point.data_parallel_shard_size,
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


def _memory_prediction(
    *,
    prediction_peak_mem_gb: float | None,
    prediction_status: str,
    analytic_peak_floor_gb: float | None,
    memory_peak_estimate: Any | None,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
) -> tuple[float | None, str, str, str, float | None, float | None, str | None, list[str]]:
    predicted_peak_mem_gb = analytic_peak_floor_gb
    memory_basis = "analytic_floor"
    memory_calibration_source = None
    memory_calibration_notes: list[str] = []
    if prediction_peak_mem_gb is not None:
        if prediction_status == "calibrated" and (
            analytic_peak_floor_gb is None or prediction_peak_mem_gb >= analytic_peak_floor_gb
        ):
            predicted_peak_mem_gb = prediction_peak_mem_gb
            memory_basis = "calibrated_peak"
        elif prediction_status != "calibrated" and memory_peak_estimate is not None:
            predicted_peak_mem_gb = memory_peak_estimate.peak_gb
            memory_basis = getattr(memory_peak_estimate, "basis", "calibrated_overhead_peak")
            memory_calibration_source = memory_peak_estimate.source_label
            memory_calibration_notes = memory_peak_estimate.notes
        elif analytic_peak_floor_gb is None or prediction_peak_mem_gb >= analytic_peak_floor_gb:
            predicted_peak_mem_gb = prediction_peak_mem_gb
            memory_basis = "extrapolated_peak"
    elif memory_peak_estimate is not None:
        predicted_peak_mem_gb = memory_peak_estimate.peak_gb
        memory_basis = getattr(memory_peak_estimate, "basis", "calibrated_overhead_peak")
        memory_calibration_source = memory_peak_estimate.source_label
        memory_calibration_notes = memory_peak_estimate.notes

    _, _, feasibility_status = _memory_factor(
        predicted_peak_mem_gb,
        memory_basis=memory_basis,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    memory_coverage_status, predicted_residual_gb, predicted_residual_fraction = _memory_coverage_for_candidate(
        analytic_peak_floor_gb=analytic_peak_floor_gb,
        estimated_peak_mem_gb=predicted_peak_mem_gb,
        memory_basis=memory_basis,
    )
    return (
        predicted_peak_mem_gb,
        memory_basis,
        feasibility_status,
        memory_coverage_status,
        predicted_residual_gb,
        predicted_residual_fraction,
        memory_calibration_source,
        memory_calibration_notes,
    )


def _actual_memory_residual(
    actual_peak_mem_gb: float | None,
    analytic_peak_floor_gb: float | None,
) -> tuple[float | None, float | None]:
    if actual_peak_mem_gb is None or analytic_peak_floor_gb is None or actual_peak_mem_gb <= 0:
        return None, None
    residual = max(actual_peak_mem_gb - analytic_peak_floor_gb, 0.0)
    return round(residual, 3), round(residual / actual_peak_mem_gb, 3)


def _calibration_residual_memory_peak_estimate(
    *,
    training_points: list[BenchmarkBehaviorPoint],
    base_config: dict[str, Any],
    base_topology: Topology,
    raw_config: dict[str, Any],
    target_topology: Topology,
    metadata: Any,
    analytic_peak_floor_gb: float | None,
    world_size: int | None,
    local_world_size: int | None,
) -> _CalibrationMemoryPeakEstimate | None:
    if analytic_peak_floor_gb is None:
        return None

    estimates: list[tuple[tuple[float, float, float, float, float], _CalibrationMemoryPeakEstimate]] = []
    for point in training_points:
        if point.peak_mem_gb is None or point.correctness_status == "oom":
            continue
        if behavior_point_model_mismatches(point, raw_config):
            continue
        reference_config, reference_topology, _ = _topology_for_point(
            base_config,
            base_topology,
            point,
            world_size=world_size,
            local_world_size=local_world_size,
            require_tokens=False,
        )
        if reference_config is None or reference_topology is None:
            continue
        reference_memory = build_memory_ledger(
            reference_config,
            topology=reference_topology,
            model_metadata=metadata,
        )
        reference_floor = reference_memory.analytic_peak_floor_gb
        if reference_floor is None or point.peak_mem_gb < reference_floor:
            continue
        residual_gb = point.peak_mem_gb - reference_floor
        if residual_gb <= 0:
            continue
        sequence_distance = abs(
            math.log(
                (target_topology.sample_packing_sequence_len or 1)
                / (reference_topology.sample_packing_sequence_len or 1)
            )
        )
        parallel_distance = sum(
            abs(math.log2(max(target_value, 1) / max(reference_value, 1)))
            for target_value, reference_value in (
                (target_topology.expert_parallel_size, reference_topology.expert_parallel_size),
                (target_topology.ep_fsdp_size or 1, reference_topology.ep_fsdp_size or 1),
                (target_topology.tensor_parallel_size, reference_topology.tensor_parallel_size),
                (target_topology.pipeline_parallel_size, reference_topology.pipeline_parallel_size),
                (target_topology.sequence_parallel_size, reference_topology.sequence_parallel_size),
            )
        )
        batch_distance = abs(
            math.log2(max(target_topology.micro_batch_size, 1) / max(reference_topology.micro_batch_size, 1))
        )
        workload_mismatches = behavior_point_workload_mismatches(point, raw_config)
        notes = [
            "calibration_residual_prior=minimum_same_model_measured_residual",
            f"memory_residual_reference={point.label}",
            f"reference_peak_gb={point.peak_mem_gb:.3f}",
            f"reference_floor_gb={reference_floor:.3f}",
            f"reference_residual_gb={residual_gb:.3f}",
            f"estimated_residual_gb={residual_gb:.3f}",
            f"sequence_distance_log={sequence_distance:.3f}",
            f"parallel_distance_log2={parallel_distance:.3f}",
            f"batch_distance_log2={batch_distance:.3f}",
        ]
        if workload_mismatches:
            notes.append("reference_workload_mismatches=" + ",".join(workload_mismatches))
        estimates.append(
            (
                (
                    residual_gb,
                    float(len(workload_mismatches)),
                    sequence_distance,
                    parallel_distance,
                    batch_distance,
                ),
                _CalibrationMemoryPeakEstimate(
                    peak_gb=round(analytic_peak_floor_gb + residual_gb, 3),
                    overhead_gb=round(residual_gb, 3),
                    source_label=point.label,
                    notes=notes,
                ),
            )
        )

    if not estimates:
        return None
    return min(estimates, key=lambda item: item[0])[1]


def _phase_bottleneck_details(phase_time_share: dict[str, float]) -> tuple[str | None, str | None, float | None]:
    visible = {phase: share for phase, share in phase_time_share.items() if phase != "train_step_total"}
    if not visible:
        return None, None, None
    phase = max(visible, key=visible.get)
    share = visible[phase]
    return phase, _phase_bucket(phase), round(share, 6)


def _memory_bottleneck_details(
    phase_memory_peak_gb: dict[str, float],
    peak_mem_gb: float | None,
) -> tuple[str | None, str | None, float | None, float | None]:
    visible = {phase: peak for phase, peak in phase_memory_peak_gb.items() if peak > 0}
    if not visible:
        return None, None, None, None
    phase, peak = max(visible.items(), key=lambda item: (item[1], item[0]))
    denominator = peak_mem_gb if peak_mem_gb is not None and peak_mem_gb > 0 else peak
    return phase, _phase_bucket(phase), round(peak, 3), round(peak / denominator, 3)


def _parallel_size_for_attribution(point: BenchmarkBehaviorPoint, field_name: str, topology: Topology) -> int:
    value = getattr(point, field_name)
    if value is not None:
        return int(value)
    return int(getattr(topology, field_name))


def _memory_attribution_distance(point: BenchmarkBehaviorPoint, topology: Topology) -> tuple[float, float, float]:
    point_sequence = point.sample_packing_sequence_len or topology.sample_packing_sequence_len or 1
    target_sequence = topology.sample_packing_sequence_len or point_sequence
    sequence_distance = abs(math.log(max(target_sequence, 1) / max(point_sequence, 1)))
    parallel_pairs = (
        (topology.expert_parallel_size, _parallel_size_for_attribution(point, "expert_parallel_size", topology)),
        (topology.ep_fsdp_size or 1, point.ep_fsdp_size or topology.ep_fsdp_size or 1),
        (topology.tensor_parallel_size, _parallel_size_for_attribution(point, "tensor_parallel_size", topology)),
        (topology.pipeline_parallel_size, _parallel_size_for_attribution(point, "pipeline_parallel_size", topology)),
        (topology.ulysses_parallel_size, _parallel_size_for_attribution(point, "ulysses_parallel_size", topology)),
        (topology.ringattn_parallel_size, _parallel_size_for_attribution(point, "ringattn_parallel_size", topology)),
    )
    parallel_distance = sum(
        abs(math.log2(max(target_value, 1) / max(reference_value, 1)))
        for target_value, reference_value in parallel_pairs
    )
    batch_distance = abs(math.log2(max(topology.micro_batch_size, 1) / max(point.micro_batch_size or 1, 1)))
    return sequence_distance, parallel_distance, batch_distance


def _select_memory_attribution_point(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    raw_config: dict[str, Any],
) -> BenchmarkBehaviorPoint | None:
    usable = [
        point
        for point in behavior_points
        if point.phase_memory_peak_gb
        and point.peak_mem_gb is not None
        and point.peak_mem_gb > 0
        and point.correctness_status != "oom"
        and not behavior_point_model_mismatches(point, raw_config)
    ]
    if not usable:
        return None

    def key(point: BenchmarkBehaviorPoint) -> tuple[int, float, float, float, int, float]:
        runtime_mismatch_count = len(behavior_point_workload_mismatches(point, raw_config))
        sequence_distance, parallel_distance, batch_distance = _memory_attribution_distance(point, topology)
        return (
            -runtime_mismatch_count,
            -sequence_distance,
            -parallel_distance,
            -batch_distance,
            1 if point.tokens_per_sec is not None else 0,
            point.peak_mem_gb or 0.0,
        )

    return max(usable, key=key)


def _select_phase_timing_attribution_point(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    raw_config: dict[str, Any],
) -> BenchmarkBehaviorPoint | None:
    usable = [
        point
        for point in behavior_points
        if point.phase_time_share
        and point.correctness_status != "oom"
        and not behavior_point_model_mismatches(point, raw_config)
    ]
    if not usable:
        return None

    def key(point: BenchmarkBehaviorPoint) -> tuple[int, float, float, float, int, int]:
        runtime_mismatch_count = len(behavior_point_workload_mismatches(point, raw_config))
        sequence_distance, parallel_distance, batch_distance = _memory_attribution_distance(point, topology)
        return (
            -runtime_mismatch_count,
            -sequence_distance,
            -parallel_distance,
            -batch_distance,
            1 if point.tokens_per_sec is not None else 0,
            point.measured_steps or 0,
        )

    return max(usable, key=key)


def _scaled_phase_memory_peak_gb(
    point: BenchmarkBehaviorPoint,
    predicted_peak_mem_gb: float | None,
) -> dict[str, float]:
    scale = 1.0
    if predicted_peak_mem_gb is not None and point.peak_mem_gb is not None and point.peak_mem_gb > 0:
        scale = max(predicted_peak_mem_gb, 0.0) / point.peak_mem_gb
    return {phase: round(max(0.0, peak * scale), 6) for phase, peak in point.phase_memory_peak_gb.items()}


def _phase_top_items(phase_time_share: dict[str, float], *, limit: int = 3) -> list[tuple[str, str, float]]:
    visible = [(phase, share) for phase, share in phase_time_share.items() if phase != "train_step_total"]
    ordered = sorted(visible, key=lambda item: (-item[1], item[0]))
    return [(phase, _phase_bucket(phase), round(share, 6)) for phase, share in ordered[:limit]]


def _ordered_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _top_overlap(actual: list[str], predicted: list[str]) -> tuple[int | None, float | None]:
    actual_unique = _ordered_unique(actual)
    if not actual_unique:
        return None, None
    predicted_unique = set(predicted)
    overlap = sum(1 for value in actual_unique if value in predicted_unique)
    return overlap, round(overlap / len(actual_unique), 3)


def _max_holdout_by_field(holdouts: list[CalibrationHoldout], field_name: str) -> CalibrationHoldout | None:
    return max(
        (holdout for holdout in holdouts if getattr(holdout, field_name) is not None),
        key=lambda holdout: (getattr(holdout, field_name), holdout.label),
        default=None,
    )


def _min_holdout_by_field(holdouts: list[CalibrationHoldout], field_name: str) -> CalibrationHoldout | None:
    return min(
        (holdout for holdout in holdouts if getattr(holdout, field_name) is not None),
        key=lambda holdout: (getattr(holdout, field_name), holdout.label),
        default=None,
    )


def _memory_error_exceeds_target(holdout: CalibrationHoldout) -> bool:
    if holdout.memory_absolute_error_gb is None or holdout.actual_peak_mem_gb is None:
        return False
    threshold = max(_MEMORY_ABS_ERROR_TARGET_GB, holdout.actual_peak_mem_gb * _MEMORY_RELATIVE_ERROR_TARGET_FRACTION)
    return holdout.memory_absolute_error_gb > threshold


def _empirical_required_uncertainty_fraction(holdout: CalibrationHoldout) -> float | None:
    if holdout.predicted_tokens_per_sec is None or holdout.predicted_tokens_per_sec <= 0:
        return None
    if holdout.actual_tokens_per_sec is None:
        return None
    return abs(holdout.actual_tokens_per_sec - holdout.predicted_tokens_per_sec) / holdout.predicted_tokens_per_sec


def _eligible_calibration_point(point: BenchmarkBehaviorPoint) -> bool:
    if point.tokens_per_sec is not None:
        return True
    if point.correctness_status in {"oom", "runtime_failure_after_steps"}:
        return False
    return point.status == "observed_log_metrics_only" and (
        point.peak_mem_gb is not None or bool(point.phase_memory_peak_gb) or bool(point.phase_time_share)
    )


def _load_calibration_support_points(
    support_benchmark_dirs: list[str | Path] | tuple[str | Path, ...],
) -> list[BenchmarkBehaviorPoint]:
    points: list[BenchmarkBehaviorPoint] = []
    for support_dir in support_benchmark_dirs:
        points.extend(load_benchmark_behavior_points(support_dir))
    return points


def _prediction_uncertainty_calibration_status(
    *,
    errors: list[float],
    interval_covered_count: int,
    uncertainty_fractions: list[float],
    empirical_required_uncertainty_fractions: list[float],
) -> str:
    if not errors or not empirical_required_uncertainty_fractions:
        return "no_scored_uncertainty_holdouts"
    if interval_covered_count < len(errors):
        return "prediction_interval_undercoverage"
    max_required = max(empirical_required_uncertainty_fractions)
    if max_required <= _THROUGHPUT_MAPE_TARGET_PERCENT / 100.0:
        if uncertainty_fractions and max(uncertainty_fractions) >= 0.50:
            return "conservative_uncertainty_empirically_supported"
        return "prediction_uncertainty_empirically_supported"
    return "empirical_uncertainty_exceeds_target"


def _calibration_fidelity_support(
    holdouts: list[CalibrationHoldout],
    *,
    errors: list[float],
    interval_covered_count: int,
    uncertainty_fractions: list[float],
    empirical_required_uncertainty_fractions: list[float],
    memory_percentage_errors: list[float],
    memory_bottleneck_evaluated_count: int,
    memory_bottleneck_bucket_match_count: int,
    phase_bottleneck_evaluated_count: int,
    phase_bottleneck_bucket_match_count: int,
    phase_top3_evaluated_count: int,
    phase_bucket_top3_overlap_rates: list[float],
    phase_bottleneck_share_errors: list[float],
) -> tuple[str, list[str]]:
    if not holdouts:
        return "no_calibration_holdouts", ["no_evaluated_holdouts"]
    if not errors:
        return "no_scored_calibration_holdouts", ["no_scored_holdouts"]

    blockers: set[str] = set()
    throughput_holdout_count = sum(1 for holdout in holdouts if holdout.actual_tokens_per_sec is not None)
    if len(errors) < throughput_holdout_count:
        blockers.add("unscored_holdouts")
    if max(errors) > _THROUGHPUT_MAPE_TARGET_PERCENT:
        blockers.add("max_throughput_mape_exceeds_8_percent")
    if interval_covered_count < len(errors):
        blockers.add("prediction_interval_misses")
    high_uncertainty_empirically_supported = (
        max(errors) <= _THROUGHPUT_MAPE_TARGET_PERCENT
        and interval_covered_count == len(errors)
        and bool(empirical_required_uncertainty_fractions)
        and max(empirical_required_uncertainty_fractions) <= _THROUGHPUT_MAPE_TARGET_PERCENT / 100.0
    )
    if uncertainty_fractions and max(uncertainty_fractions) >= 0.50 and not high_uncertainty_empirically_supported:
        blockers.add("high_prediction_uncertainty")

    if not memory_percentage_errors:
        blockers.add("missing_memory_holdouts")
    elif any(_memory_error_exceeds_target(holdout) for holdout in holdouts):
        blockers.add("memory_error_exceeds_3_percent_or_2gb")
    missing_memory_bottleneck_prediction_count = sum(
        1
        for holdout in holdouts
        if holdout.actual_memory_bottleneck_bucket is not None and holdout.predicted_memory_bottleneck_bucket is None
    )
    if missing_memory_bottleneck_prediction_count > 0:
        blockers.add("missing_memory_bottleneck_predictions")
    elif (
        memory_bottleneck_evaluated_count > 0
        and memory_bottleneck_bucket_match_count < memory_bottleneck_evaluated_count
    ):
        blockers.add("memory_bottleneck_bucket_mismatch")

    if phase_bottleneck_evaluated_count == 0:
        blockers.add("missing_phase_bottleneck_holdouts")
    elif phase_bottleneck_bucket_match_count < phase_bottleneck_evaluated_count:
        blockers.add("phase_bottleneck_bucket_mismatch")
    if phase_top3_evaluated_count > 0 and phase_bucket_top3_overlap_rates:
        if min(phase_bucket_top3_overlap_rates) < 1.0:
            blockers.add("phase_bucket_top3_mismatch")
    if phase_bottleneck_share_errors and max(phase_bottleneck_share_errors) > _PHASE_SHARE_ABS_ERROR_TARGET:
        blockers.add("phase_bottleneck_share_error_exceeds_10_percent")

    blocker_list = sorted(blockers)
    if "prediction_interval_misses" in blockers:
        return "calibration_interval_coverage_failed", blocker_list
    if "max_throughput_mape_exceeds_8_percent" in blockers:
        return "calibration_error_attribution_needed", blocker_list
    if "memory_error_exceeds_3_percent_or_2gb" in blockers:
        return "calibration_memory_error_exceeds_target", blocker_list
    if any(blocker.startswith("memory_bottleneck_") for blocker in blockers):
        return "calibration_memory_attribution_mismatch", blocker_list
    if any(blocker.startswith("phase_") for blocker in blockers):
        return "calibration_phase_attribution_mismatch", blocker_list
    missing_blockers = {
        "missing_memory_holdouts",
        "missing_memory_bottleneck_predictions",
        "missing_phase_bottleneck_holdouts",
        "unscored_holdouts",
        "high_prediction_uncertainty",
    }
    if blockers and blockers <= missing_blockers:
        return "partial_calibration_fidelity_missing_attribution", blocker_list
    if blockers:
        return "partial_calibration_fidelity", blocker_list
    return "calibration_fidelity_supported", []


def _calibration_gap_status_counts(gaps: list[CalibrationValidationGap]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gap in gaps:
        counts[gap.gap_status] = counts.get(gap.gap_status, 0) + 1
    return dict(sorted(counts.items()))


def _unique_required_measurements(gaps: list[CalibrationValidationGap]) -> list[str]:
    seen: set[str] = set()
    required: list[str] = []
    for gap in gaps:
        if gap.required_measurement in seen:
            continue
        seen.add(gap.required_measurement)
        required.append(gap.required_measurement)
    return required


def _labels(holdouts: list[CalibrationHoldout]) -> list[str]:
    return [holdout.label for holdout in holdouts]


def _max_percentage_error(holdouts: list[CalibrationHoldout]) -> tuple[float | None, str | None]:
    holdout = _max_holdout_by_field(holdouts, "absolute_percentage_error")
    if holdout is None:
        return None, None
    return holdout.absolute_percentage_error, holdout.label


def _max_memory_error(holdouts: list[CalibrationHoldout]) -> tuple[float | None, str | None]:
    holdout = _max_holdout_by_field(holdouts, "memory_absolute_error_gb")
    if holdout is None:
        return None, None
    return holdout.memory_absolute_error_gb, holdout.label


def _max_phase_share_error(holdouts: list[CalibrationHoldout]) -> tuple[float | None, str | None]:
    holdout = _max_holdout_by_field(holdouts, "phase_bottleneck_share_absolute_error")
    if holdout is None:
        return None, None
    return holdout.phase_bottleneck_share_absolute_error, holdout.label


def _calibration_validation_gap(
    *,
    gap_status: str,
    priority: int,
    required_measurement: str,
    reason: str,
    affected_holdouts: list[CalibrationHoldout],
    blocker_names: list[str],
) -> CalibrationValidationGap:
    max_percentage_error, max_percentage_error_label = _max_percentage_error(affected_holdouts)
    max_memory_error, max_memory_error_label = _max_memory_error(affected_holdouts)
    max_phase_share_error, max_phase_share_error_label = _max_phase_share_error(affected_holdouts)
    return CalibrationValidationGap(
        gap_status=gap_status,
        priority=priority,
        required_measurement=required_measurement,
        reason=reason,
        affected_holdout_count=len(affected_holdouts),
        affected_holdout_labels=_labels(affected_holdouts),
        blocker_names=sorted(set(blocker_names)),
        max_absolute_percentage_error=(round(max_percentage_error, 3) if max_percentage_error is not None else None),
        max_absolute_percentage_error_label=max_percentage_error_label,
        max_memory_absolute_error_gb=round(max_memory_error, 3) if max_memory_error is not None else None,
        max_memory_absolute_error_label=max_memory_error_label,
        max_phase_bottleneck_share_absolute_error=(
            round(max_phase_share_error, 6) if max_phase_share_error is not None else None
        ),
        max_phase_bottleneck_share_absolute_error_label=max_phase_share_error_label,
        missing_memory_count=sum(1 for holdout in affected_holdouts if holdout.memory_absolute_error_gb is None),
        missing_phase_bottleneck_count=sum(
            1 for holdout in affected_holdouts if holdout.phase_bottleneck_bucket_match is None
        ),
    )


def _calibration_validation_gap_portfolio(
    holdouts: list[CalibrationHoldout],
    blockers: list[str],
) -> list[CalibrationValidationGap]:
    blocker_set = set(blockers)
    gaps: list[CalibrationValidationGap] = []
    if "no_evaluated_holdouts" in blocker_set:
        gaps.append(
            _calibration_validation_gap(
                gap_status="no_calibration_holdouts_need_replay",
                priority=130,
                required_measurement="add_calibration_holdouts",
                reason="no measured holdouts were available for calibration validation",
                affected_holdouts=[],
                blocker_names=blockers,
            )
        )
    if "no_scored_holdouts" in blocker_set:
        gaps.append(
            _calibration_validation_gap(
                gap_status="no_scored_holdouts_need_supported_replay",
                priority=125,
                required_measurement="add_scored_same_context_calibration_holdouts",
                reason="measured holdouts exist but none received a scored simulator prediction",
                affected_holdouts=holdouts,
                blocker_names=blockers,
            )
        )
    if "max_throughput_mape_exceeds_8_percent" in blocker_set:
        affected = [
            holdout
            for holdout in holdouts
            if holdout.absolute_percentage_error is not None
            and holdout.absolute_percentage_error > _THROUGHPUT_MAPE_TARGET_PERCENT
        ]
        gaps.append(
            _calibration_validation_gap(
                gap_status="throughput_error_needs_attribution",
                priority=120,
                required_measurement="replay_high_error_holdouts_with_component_timing",
                reason="at least one scored holdout exceeds the throughput mean absolute percentage error target",
                affected_holdouts=affected,
                blocker_names=["max_throughput_mape_exceeds_8_percent"],
            )
        )
    if "prediction_interval_misses" in blocker_set:
        affected = [holdout for holdout in holdouts if holdout.actual_tokens_in_prediction_interval is False]
        gaps.append(
            _calibration_validation_gap(
                gap_status="prediction_interval_miss_needs_uncertainty_recalibration",
                priority=115,
                required_measurement="add_holdouts_to_recalibrate_prediction_interval",
                reason="one or more actual holdout throughputs fall outside the predicted interval",
                affected_holdouts=affected,
                blocker_names=["prediction_interval_misses"],
            )
        )
    if "memory_error_exceeds_3_percent_or_2gb" in blocker_set:
        affected = [holdout for holdout in holdouts if _memory_error_exceeds_target(holdout)]
        gaps.append(
            _calibration_validation_gap(
                gap_status="memory_error_needs_residual_attribution",
                priority=110,
                required_measurement="replay_high_memory_error_holdouts_with_memory_profile",
                reason="one or more memory predictions exceed the absolute or relative memory error target",
                affected_holdouts=affected,
                blocker_names=["memory_error_exceeds_3_percent_or_2gb"],
            )
        )
    if "memory_bottleneck_bucket_mismatch" in blocker_set:
        affected = [holdout for holdout in holdouts if holdout.memory_bottleneck_bucket_match is False]
        gaps.append(
            _calibration_validation_gap(
                gap_status="memory_bottleneck_mismatch_needs_phase_memory_replay",
                priority=105,
                required_measurement="replay_memory_bottleneck_holdouts_with_phase_memory_profile",
                reason="predicted memory bottleneck bucket does not match observed phase memory attribution",
                affected_holdouts=affected,
                blocker_names=["memory_bottleneck_bucket_mismatch"],
            )
        )
    if "missing_memory_bottleneck_predictions" in blocker_set:
        affected = [
            holdout
            for holdout in holdouts
            if holdout.actual_memory_bottleneck_bucket is not None
            and holdout.predicted_memory_bottleneck_bucket is None
        ]
        gaps.append(
            _calibration_validation_gap(
                gap_status="missing_memory_bottleneck_predictions_need_phase_memory_replay",
                priority=105,
                required_measurement="replay_memory_bottleneck_holdouts_with_phase_memory_profile",
                reason="observed phase memory attribution exists but the prediction has no memory bottleneck bucket",
                affected_holdouts=affected,
                blocker_names=["missing_memory_bottleneck_predictions"],
            )
        )
    if "missing_memory_holdouts" in blocker_set:
        affected = [holdout for holdout in holdouts if holdout.memory_absolute_error_gb is None]
        gaps.append(
            _calibration_validation_gap(
                gap_status="missing_memory_holdouts_need_memory_profile",
                priority=100,
                required_measurement="collect_peak_memory_for_calibration_holdouts",
                reason="calibration fidelity cannot validate memory without observed peak memory holdouts",
                affected_holdouts=affected,
                blocker_names=["missing_memory_holdouts"],
            )
        )
    if "phase_bottleneck_bucket_mismatch" in blocker_set:
        affected = [holdout for holdout in holdouts if holdout.phase_bottleneck_bucket_match is False]
        gaps.append(
            _calibration_validation_gap(
                gap_status="phase_bottleneck_mismatch_needs_component_timing_replay",
                priority=95,
                required_measurement="replay_phase_bottleneck_holdouts_with_component_timing",
                reason="predicted phase bottleneck bucket does not match observed phase timing attribution",
                affected_holdouts=affected,
                blocker_names=["phase_bottleneck_bucket_mismatch"],
            )
        )
    if "phase_bucket_top3_mismatch" in blocker_set:
        affected = [
            holdout
            for holdout in holdouts
            if holdout.phase_bucket_top3_overlap_rate is not None and holdout.phase_bucket_top3_overlap_rate < 1.0
        ]
        gaps.append(
            _calibration_validation_gap(
                gap_status="phase_top3_mismatch_needs_component_timing_replay",
                priority=90,
                required_measurement="replay_phase_top3_holdouts_with_component_timing",
                reason="predicted phase bucket top-3 attribution misses observed timing buckets",
                affected_holdouts=affected,
                blocker_names=["phase_bucket_top3_mismatch"],
            )
        )
    if "phase_bottleneck_share_error_exceeds_10_percent" in blocker_set:
        affected = [
            holdout
            for holdout in holdouts
            if holdout.phase_bottleneck_share_absolute_error is not None
            and holdout.phase_bottleneck_share_absolute_error > _PHASE_SHARE_ABS_ERROR_TARGET
        ]
        gaps.append(
            _calibration_validation_gap(
                gap_status="phase_share_error_needs_component_timing_replay",
                priority=85,
                required_measurement="replay_phase_share_holdouts_with_component_timing",
                reason="predicted phase bottleneck share exceeds the attribution error target",
                affected_holdouts=affected,
                blocker_names=["phase_bottleneck_share_error_exceeds_10_percent"],
            )
        )
    if "missing_phase_bottleneck_holdouts" in blocker_set:
        affected = [holdout for holdout in holdouts if holdout.phase_bottleneck_bucket_match is None]
        gaps.append(
            _calibration_validation_gap(
                gap_status="missing_phase_bottleneck_holdouts_need_phase_timing",
                priority=80,
                required_measurement="collect_phase_timing_for_calibration_holdouts",
                reason="calibration fidelity cannot validate phase bottlenecks without observed phase timing holdouts",
                affected_holdouts=affected,
                blocker_names=["missing_phase_bottleneck_holdouts"],
            )
        )
    if "high_prediction_uncertainty" in blocker_set:
        affected = [
            holdout
            for holdout in holdouts
            if holdout.prediction_uncertainty_fraction is not None and holdout.prediction_uncertainty_fraction >= 0.50
        ]
        gaps.append(
            _calibration_validation_gap(
                gap_status="high_prediction_uncertainty_needs_nearby_holdouts",
                priority=70,
                required_measurement="add_nearby_calibration_holdouts_to_tighten_uncertainty",
                reason="predicted uncertainty remains high and is not empirically justified by observed errors",
                affected_holdouts=affected,
                blocker_names=["high_prediction_uncertainty"],
            )
        )
    if "unscored_holdouts" in blocker_set:
        affected = [
            holdout
            for holdout in holdouts
            if holdout.actual_tokens_per_sec is not None and holdout.absolute_percentage_error is None
        ]
        gaps.append(
            _calibration_validation_gap(
                gap_status="unscored_holdouts_need_supported_replay",
                priority=60,
                required_measurement="add_supported_same_context_calibration_holdouts",
                reason="some measured holdouts could not be scored by the simulator",
                affected_holdouts=affected,
                blocker_names=["unscored_holdouts"],
            )
        )
    return sorted(gaps, key=lambda gap: (-gap.priority, gap.gap_status))


def _append_calibration_design_config(
    rendered: list[ScenarioMeasurementConfig],
    seen: set[tuple[str, str]],
    design: ScenarioMeasurementConfig | None,
) -> bool:
    if design is None:
        return False
    required_measurement = design.label.split(":", 3)[1] if ":" in design.label else design.label
    key = (required_measurement, yaml.safe_dump(design.config, sort_keys=True))
    if key in seen:
        return False
    seen.add(key)
    rendered.append(design)
    return True


def _calibration_design_config_from_point(
    *,
    base_config: dict[str, Any],
    base_topology: Topology,
    point: BenchmarkBehaviorPoint,
    required_measurement: str,
    design_kind: str,
    index: int,
    world_size: int | None,
    local_world_size: int | None,
    config_overrides: tuple[str, ...],
) -> ScenarioMeasurementConfig | None:
    raw_config, topology, _ = _topology_for_point(
        base_config,
        base_topology,
        point,
        world_size=world_size,
        local_world_size=local_world_size,
        require_tokens=False,
    )
    if raw_config is None or topology is None:
        return None
    for override in config_overrides:
        _apply_config_override(raw_config, override)
    label = f"design:{required_measurement}:{design_kind}_{index:02d}:{point.label}:{_topology_label(topology)}"
    return ScenarioMeasurementConfig(
        label=label,
        filename=_measurement_config_filename(index, label),
        config=raw_config,
    )


def _calibration_nearby_design_config_from_topology(
    *,
    base_config: dict[str, Any],
    source_point: BenchmarkBehaviorPoint,
    source_topology: Topology,
    required_measurement: str,
    design_kind: str,
    variant_label: str,
    workload_values: dict[str, int],
    index: int,
    config_overrides: tuple[str, ...],
) -> ScenarioMeasurementConfig | None:
    try:
        raw_config = _mutated_config(
            base_config,
            world_size=source_topology.world_size,
            micro_batch_size=workload_values["micro_batch_size"],
            gradient_accumulation_steps=workload_values["gradient_accumulation_steps"],
            expert_parallel_size=source_topology.expert_parallel_size,
            tensor_parallel_size=source_topology.tensor_parallel_size,
            pipeline_parallel_size=source_topology.pipeline_parallel_size,
            ulysses_parallel_size=source_topology.ulysses_parallel_size,
            ringattn_parallel_size=source_topology.ringattn_parallel_size,
            data_parallel_replicate_size=source_topology.data_parallel_replicate_size,
            data_parallel_shard_size=source_topology.data_parallel_shard_size,
        )
        _section(raw_config, "data")["sample_packing_sequence_len"] = workload_values["sample_packing_sequence_len"]
        _apply_point_runtime_signature(raw_config, source_point)
        for override in config_overrides:
            _apply_config_override(raw_config, override)
        topology = resolve_topology(
            raw_config,
            world_size=source_topology.world_size,
            local_world_size=source_topology.local_world_size,
        )
    except (KeyError, TypeError, ValueError):
        return None

    label = (
        f"design:{required_measurement}:{design_kind}_{index:02d}_{variant_label}:"
        f"{source_point.label}:{_topology_label(topology)}"
    )
    return ScenarioMeasurementConfig(
        label=label,
        filename=_measurement_config_filename(index, label),
        config=raw_config,
    )


def _materialize_calibration_measurement_design_configs_from_context(
    *,
    base_config: dict[str, Any],
    base_topology: Topology,
    behavior_points: list[BenchmarkBehaviorPoint],
    calibration_validation_gaps: list[CalibrationValidationGap],
    world_size: int | None,
    local_world_size: int | None,
    max_configs_per_measurement: int,
) -> list[ScenarioMeasurementConfig]:
    points_by_label = {point.label: point for point in behavior_points}
    rendered: list[ScenarioMeasurementConfig] = []
    seen: set[tuple[str, str]] = set()
    index = 1
    for gap in sorted(calibration_validation_gaps, key=lambda item: (-item.priority, item.gap_status)):
        count_for_measurement = 0

        def add_limited(design: ScenarioMeasurementConfig | None) -> None:
            nonlocal count_for_measurement, index
            if count_for_measurement >= max_configs_per_measurement:
                return
            if _append_calibration_design_config(rendered, seen, design):
                index += 1
                count_for_measurement += 1

        overrides = _CALIBRATION_REPLAY_OVERRIDES_BY_MEASUREMENT.get(gap.required_measurement)
        if overrides is not None:
            design_kind = _CALIBRATION_REPLAY_KIND_BY_MEASUREMENT[gap.required_measurement]
            for label in gap.affected_holdout_labels:
                if count_for_measurement >= max_configs_per_measurement:
                    break
                point = points_by_label.get(label)
                if point is None:
                    continue
                add_limited(
                    _calibration_design_config_from_point(
                        base_config=base_config,
                        base_topology=base_topology,
                        point=point,
                        required_measurement=gap.required_measurement,
                        design_kind=design_kind,
                        index=index,
                        world_size=world_size,
                        local_world_size=local_world_size,
                        config_overrides=overrides,
                    )
                )
            continue

        design_kind = _CALIBRATION_HOLDOUT_REPLAY_KIND_BY_MEASUREMENT.get(gap.required_measurement)
        if design_kind is not None:
            for label in gap.affected_holdout_labels:
                if count_for_measurement >= max_configs_per_measurement:
                    break
                point = points_by_label.get(label)
                if point is None:
                    continue
                add_limited(
                    _calibration_design_config_from_point(
                        base_config=base_config,
                        base_topology=base_topology,
                        point=point,
                        required_measurement=gap.required_measurement,
                        design_kind=design_kind,
                        index=index,
                        world_size=world_size,
                        local_world_size=local_world_size,
                        config_overrides=(),
                    )
                )
            continue

        design_kind = _CALIBRATION_NEARBY_HOLDOUT_KIND_BY_MEASUREMENT.get(gap.required_measurement)
        if design_kind is None:
            continue
        for label in gap.affected_holdout_labels:
            if count_for_measurement >= max_configs_per_measurement:
                break
            point = points_by_label.get(label)
            if point is None:
                continue
            _, source_topology, _ = _topology_for_point(
                base_config,
                base_topology,
                point,
                world_size=world_size,
                local_world_size=local_world_size,
                require_tokens=False,
            )
            if source_topology is None:
                continue
            for variant_label, workload_values in _workload_design_variants(source_topology):
                if count_for_measurement >= max_configs_per_measurement:
                    break
                add_limited(
                    _calibration_nearby_design_config_from_topology(
                        base_config=base_config,
                        source_point=point,
                        source_topology=source_topology,
                        required_measurement=gap.required_measurement,
                        design_kind=design_kind,
                        variant_label=variant_label,
                        workload_values=workload_values,
                        index=index,
                        config_overrides=(),
                    )
                )
    return rendered


def materialize_calibration_measurement_design_configs(
    report: CalibrationReport,
    *,
    max_configs_per_measurement: int = 4,
) -> list[ScenarioMeasurementConfig]:
    """Render bounded YAML design rows for calibration replay/profile gaps."""
    base_config = load_training_config(report.base_config_path)
    base_topology = resolve_topology(base_config)
    behavior_points = [
        *load_benchmark_behavior_points(report.benchmark_dir),
        *_load_calibration_support_points(tuple(report.calibration_support_benchmark_dirs)),
    ]
    return _materialize_calibration_measurement_design_configs_from_context(
        base_config=base_config,
        base_topology=base_topology,
        behavior_points=behavior_points,
        calibration_validation_gaps=report.calibration_validation_gaps,
        world_size=None,
        local_world_size=None,
        max_configs_per_measurement=max_configs_per_measurement,
    )


def materialize_calibration_measurement_configs(report: CalibrationReport) -> list[ScenarioMeasurementConfig]:
    """Render calibration design YAML payloads with their on-disk filenames."""
    return [
        ScenarioMeasurementConfig(
            label=item.label,
            filename=f"design_{item.filename}",
            config=item.config,
        )
        for item in materialize_calibration_measurement_design_configs(report)
    ]


def write_measurement_configs(report: CalibrationReport, output_dir: str | Path) -> list[ScenarioMeasurementConfig]:
    """Write calibration replay/profile design configs as YAML files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rendered = materialize_calibration_measurement_configs(report)
    for item in rendered:
        (output_path / item.filename).write_text(
            yaml.safe_dump(runtime_training_config(item.config), sort_keys=False),
            encoding="utf-8",
        )
    return rendered


def evaluate_calibration(
    base_config_path: str | Path,
    *,
    benchmark_dir: str | Path,
    calibration_support_benchmark_dirs: list[str | Path] | tuple[str | Path, ...] = (),
    world_size: int | None = None,
    local_world_size: int | None = None,
    device_memory_limit_gb: float = 80.0,
    memory_safety_factor: float = 1.15,
) -> CalibrationReport:
    base_path = Path(base_config_path)
    benchmark_path = resolve_calibration_pack(benchmark_dir)
    base_config = load_training_config(base_path)
    base_topology = resolve_topology(base_config, world_size=world_size, local_world_size=local_world_size)
    behavior_points = load_benchmark_behavior_points(benchmark_path)
    support_points = _load_calibration_support_points(calibration_support_benchmark_dirs)
    prediction_points = [*behavior_points, *support_points]
    measured_points = [
        point
        for point in behavior_points
        if _eligible_calibration_point(point) and not behavior_point_model_mismatches(point, base_config)
    ]

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
            require_tokens=False,
        )
        if raw_config is None or topology is None:
            skipped_count += 1
            warnings.append(f"skipped {heldout.label}: {skip_reason}")
            continue

        training_points = _without_point(prediction_points, heldout)
        shape = build_shape_ledger(topology, balanced_routing=True)
        metadata = resolve_model_metadata(raw_config)
        memory = build_memory_ledger(raw_config, topology=topology, model_metadata=metadata)
        exact_prediction = predict_benchmark_behavior(training_points, topology, shape, raw_config)
        if exact_prediction.status == "calibrated":
            prediction = exact_prediction
            memory_peak_estimate = None
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
            prediction, _ = _extrapolate_behavior(
                training_points,
                topology,
                shape,
                raw_config=raw_config,
                device_memory_limit_gb=device_memory_limit_gb,
                memory_safety_factor=memory_safety_factor,
                analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
                memory_peak_estimate=memory_peak_estimate,
            )

        if memory_peak_estimate is None and (
            prediction.peak_mem_gb is None
            or (memory.analytic_peak_floor_gb is not None and prediction.peak_mem_gb <= memory.analytic_peak_floor_gb)
        ):
            memory_peak_estimate = _calibration_residual_memory_peak_estimate(
                training_points=training_points,
                base_config=base_config,
                base_topology=base_topology,
                raw_config=raw_config,
                target_topology=topology,
                metadata=metadata,
                analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
                world_size=world_size,
                local_world_size=local_world_size,
            )

        predicted = prediction.tokens_per_sec
        absolute_error = None
        absolute_percentage_error = None
        if predicted is not None and heldout.tokens_per_sec is not None:
            absolute_error = abs(predicted - heldout.tokens_per_sec)
            absolute_percentage_error = 100.0 * absolute_error / heldout.tokens_per_sec
        (
            predicted_peak_mem_gb,
            memory_basis,
            memory_feasibility_status,
            memory_coverage_status,
            predicted_memory_residual_gb,
            predicted_memory_residual_fraction,
            memory_calibration_source,
            memory_calibration_notes,
        ) = _memory_prediction(
            prediction_peak_mem_gb=prediction.peak_mem_gb,
            prediction_status=prediction.status,
            analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
            memory_peak_estimate=memory_peak_estimate,
            device_memory_limit_gb=device_memory_limit_gb,
            memory_safety_factor=memory_safety_factor,
        )
        actual_memory_residual_gb, actual_memory_residual_fraction = _actual_memory_residual(
            heldout.peak_mem_gb,
            memory.analytic_peak_floor_gb,
        )
        calibration_scope = _calibration_scope(
            training_points,
            topology,
            prediction_confidence=prediction.status,
            raw_config=raw_config,
        )
        calibration_distance, _ = _calibration_distance(training_points, topology, prediction)
        risk_flags = _candidate_risk_flags(
            training_points,
            topology,
            prediction,
            raw_config=raw_config,
            calibration_scope=calibration_scope,
            prediction_confidence=prediction.status,
            communication=None,
        )
        if memory_basis in {"calibrated_overhead_peak", "calibration_residual_floor_peak"}:
            risk_flags = sorted({*risk_flags, "memory_extrapolated_overhead"})
        prediction_uncertainty = _prediction_uncertainty_fraction(
            prediction,
            prediction_confidence=prediction.status,
            calibration_scope=calibration_scope,
            calibration_distance=calibration_distance,
            risk_flags=risk_flags,
            memory_coverage_status=memory_coverage_status,
        )
        prediction_interval_lower, prediction_interval_upper = _prediction_interval(predicted, prediction_uncertainty)
        actual_in_prediction_interval = (
            (
                prediction_interval_lower is not None
                and prediction_interval_upper is not None
                and prediction_interval_lower <= heldout.tokens_per_sec <= prediction_interval_upper
            )
            if predicted is not None and heldout.tokens_per_sec is not None
            else None
        )
        memory_absolute_error_gb = None
        memory_absolute_percentage_error = None
        if heldout.peak_mem_gb is not None and predicted_peak_mem_gb is not None and heldout.peak_mem_gb > 0:
            memory_absolute_error_gb = abs(predicted_peak_mem_gb - heldout.peak_mem_gb)
            memory_absolute_percentage_error = 100.0 * memory_absolute_error_gb / heldout.peak_mem_gb
        (
            actual_memory_phase,
            actual_memory_bucket,
            actual_memory_peak_gb,
            actual_memory_fraction,
        ) = _memory_bottleneck_details(heldout.phase_memory_peak_gb, heldout.peak_mem_gb)
        predicted_phase_memory_peak_gb = prediction.phase_memory_peak_gb
        memory_attribution_warnings: list[str] = []
        if not predicted_phase_memory_peak_gb:
            memory_attribution_point = _select_memory_attribution_point(training_points, topology, raw_config)
            if memory_attribution_point is not None:
                predicted_phase_memory_peak_gb = _scaled_phase_memory_peak_gb(
                    memory_attribution_point,
                    predicted_peak_mem_gb,
                )
                memory_attribution_warnings.append(
                    f"memory_bottleneck_attribution_source={memory_attribution_point.label}"
                )
        (
            predicted_memory_phase,
            predicted_memory_bucket,
            predicted_memory_peak_gb,
            predicted_memory_fraction,
        ) = _memory_bottleneck_details(predicted_phase_memory_peak_gb, predicted_peak_mem_gb)
        memory_bottleneck_phase_match = None
        memory_bottleneck_bucket_match = None
        if actual_memory_phase is not None and predicted_memory_phase is not None:
            memory_bottleneck_phase_match = actual_memory_phase == predicted_memory_phase
        if actual_memory_bucket is not None and predicted_memory_bucket is not None:
            memory_bottleneck_bucket_match = actual_memory_bucket == predicted_memory_bucket
        memory_bottleneck_peak_error = (
            round(abs(actual_memory_peak_gb - predicted_memory_peak_gb), 3)
            if actual_memory_peak_gb is not None and predicted_memory_peak_gb is not None
            else None
        )
        memory_bottleneck_fraction_error = (
            round(abs(actual_memory_fraction - predicted_memory_fraction), 3)
            if actual_memory_fraction is not None and predicted_memory_fraction is not None
            else None
        )
        actual_phase, actual_bucket, actual_share = _phase_bottleneck_details(heldout.phase_time_share)
        predicted_phase_time_share = prediction.phase_time_share
        phase_attribution_warnings: list[str] = []
        if not predicted_phase_time_share:
            phase_attribution_point = _select_phase_timing_attribution_point(training_points, topology, raw_config)
            if phase_attribution_point is not None:
                predicted_phase_time_share = phase_attribution_point.phase_time_share
                phase_attribution_warnings.append(f"phase_timing_attribution_source={phase_attribution_point.label}")
        predicted_phase, predicted_bucket, predicted_share = _phase_bottleneck_details(predicted_phase_time_share)
        phase_bottleneck_phase_match = None
        phase_bottleneck_bucket_match = None
        if actual_phase is not None:
            phase_bottleneck_phase_match = actual_phase == predicted_phase
        if actual_bucket is not None:
            phase_bottleneck_bucket_match = actual_bucket == predicted_bucket
        phase_bottleneck_share_error = (
            round(abs(actual_share - predicted_share), 6)
            if actual_share is not None and predicted_share is not None
            else None
        )
        actual_top_items = _phase_top_items(heldout.phase_time_share)
        predicted_top_items = _phase_top_items(predicted_phase_time_share)
        actual_phase_top3 = [phase for phase, _, _ in actual_top_items]
        predicted_phase_top3 = [phase for phase, _, _ in predicted_top_items]
        actual_phase_bucket_top3 = _ordered_unique([bucket for _, bucket, _ in actual_top_items])
        predicted_phase_bucket_top3 = _ordered_unique([bucket for _, bucket, _ in predicted_top_items])
        phase_top3_overlap_count, phase_top3_overlap_rate = _top_overlap(actual_phase_top3, predicted_phase_top3)
        phase_bucket_top3_overlap_count, phase_bucket_top3_overlap_rate = _top_overlap(
            actual_phase_bucket_top3,
            predicted_phase_bucket_top3,
        )
        holdouts.append(
            CalibrationHoldout(
                label=heldout.label,
                source=heldout.source,
                topology_label=_topology_label(topology),
                actual_tokens_per_sec=heldout.tokens_per_sec,
                predicted_tokens_per_sec=predicted,
                prediction_uncertainty_fraction=prediction_uncertainty,
                prediction_interval_lower_tokens_per_sec=prediction_interval_lower,
                prediction_interval_upper_tokens_per_sec=prediction_interval_upper,
                actual_tokens_in_prediction_interval=actual_in_prediction_interval,
                actual_peak_mem_gb=heldout.peak_mem_gb,
                predicted_peak_mem_gb=round(predicted_peak_mem_gb, 3) if predicted_peak_mem_gb is not None else None,
                prediction_status=prediction.status,
                matched_label=prediction.matched_label,
                absolute_error_tokens_per_sec=round(absolute_error, 3) if absolute_error is not None else None,
                absolute_percentage_error=round(absolute_percentage_error, 3)
                if absolute_percentage_error is not None
                else None,
                analytic_peak_floor_gb=(
                    round(memory.analytic_peak_floor_gb, 3) if memory.analytic_peak_floor_gb is not None else None
                ),
                memory_prediction_basis=memory_basis,
                memory_coverage_status=memory_coverage_status,
                memory_feasibility_status=memory_feasibility_status,
                predicted_memory_residual_gb=predicted_memory_residual_gb,
                predicted_memory_residual_fraction_of_peak=predicted_memory_residual_fraction,
                actual_memory_residual_gb=actual_memory_residual_gb,
                actual_memory_residual_fraction_of_peak=actual_memory_residual_fraction,
                memory_absolute_error_gb=round(memory_absolute_error_gb, 3)
                if memory_absolute_error_gb is not None
                else None,
                memory_absolute_percentage_error=round(memory_absolute_percentage_error, 3)
                if memory_absolute_percentage_error is not None
                else None,
                actual_memory_bottleneck_phase=actual_memory_phase,
                actual_memory_bottleneck_bucket=actual_memory_bucket,
                actual_memory_bottleneck_peak_gb=actual_memory_peak_gb,
                actual_memory_bottleneck_fraction_of_peak=actual_memory_fraction,
                predicted_memory_bottleneck_phase=predicted_memory_phase,
                predicted_memory_bottleneck_bucket=predicted_memory_bucket,
                predicted_memory_bottleneck_peak_gb=predicted_memory_peak_gb,
                predicted_memory_bottleneck_fraction_of_peak=predicted_memory_fraction,
                memory_bottleneck_phase_match=memory_bottleneck_phase_match,
                memory_bottleneck_bucket_match=memory_bottleneck_bucket_match,
                memory_bottleneck_peak_absolute_error_gb=memory_bottleneck_peak_error,
                memory_bottleneck_fraction_absolute_error=memory_bottleneck_fraction_error,
                actual_phase_bottleneck_phase=actual_phase,
                actual_phase_bottleneck_bucket=actual_bucket,
                actual_phase_bottleneck_share=actual_share,
                predicted_phase_bottleneck_phase=predicted_phase,
                predicted_phase_bottleneck_bucket=predicted_bucket,
                predicted_phase_bottleneck_share=predicted_share,
                phase_bottleneck_phase_match=phase_bottleneck_phase_match,
                phase_bottleneck_bucket_match=phase_bottleneck_bucket_match,
                phase_bottleneck_share_absolute_error=phase_bottleneck_share_error,
                actual_phase_top3=actual_phase_top3,
                predicted_phase_top3=predicted_phase_top3,
                actual_phase_bucket_top3=actual_phase_bucket_top3,
                predicted_phase_bucket_top3=predicted_phase_bucket_top3,
                phase_top3_overlap_count=phase_top3_overlap_count,
                phase_top3_overlap_rate=phase_top3_overlap_rate,
                phase_bucket_top3_overlap_count=phase_bucket_top3_overlap_count,
                phase_bucket_top3_overlap_rate=phase_bucket_top3_overlap_rate,
                memory_calibration_source=memory_calibration_source,
                calibrated_from_count=len(training_points),
                memory_calibration_notes=memory_calibration_notes,
                warnings=[*prediction.warnings, *memory_attribution_warnings, *phase_attribution_warnings],
            )
        )

    errors = [
        holdout.absolute_percentage_error for holdout in holdouts if holdout.absolute_percentage_error is not None
    ]
    interval_covered_count = sum(1 for holdout in holdouts if holdout.actual_tokens_in_prediction_interval)
    uncertainty_fractions = [
        holdout.prediction_uncertainty_fraction
        for holdout in holdouts
        if holdout.prediction_uncertainty_fraction is not None
    ]
    empirical_required_uncertainty_pairs = [
        (required, holdout)
        for holdout in holdouts
        if (required := _empirical_required_uncertainty_fraction(holdout)) is not None
    ]
    empirical_required_uncertainty_fractions = [required for required, _ in empirical_required_uncertainty_pairs]
    status_counts: dict[str, int] = {}
    memory_basis_counts: dict[str, int] = {}
    memory_coverage_counts: dict[str, int] = {}
    memory_feasibility_counts: dict[str, int] = {}
    for holdout in holdouts:
        status_counts[holdout.prediction_status] = status_counts.get(holdout.prediction_status, 0) + 1
        memory_basis_counts[holdout.memory_prediction_basis] = (
            memory_basis_counts.get(holdout.memory_prediction_basis, 0) + 1
        )
        memory_coverage_counts[holdout.memory_coverage_status] = (
            memory_coverage_counts.get(holdout.memory_coverage_status, 0) + 1
        )
        memory_feasibility_counts[holdout.memory_feasibility_status] = (
            memory_feasibility_counts.get(holdout.memory_feasibility_status, 0) + 1
        )
    memory_absolute_errors = [
        holdout.memory_absolute_error_gb for holdout in holdouts if holdout.memory_absolute_error_gb is not None
    ]
    memory_percentage_errors = [
        holdout.memory_absolute_percentage_error
        for holdout in holdouts
        if holdout.memory_absolute_percentage_error is not None
    ]
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
    memory_bottleneck_holdouts = [holdout for holdout in holdouts if holdout.memory_bottleneck_bucket_match is not None]
    memory_bottleneck_phase_match_count = sum(
        1 for holdout in memory_bottleneck_holdouts if holdout.memory_bottleneck_phase_match
    )
    memory_bottleneck_bucket_match_count = sum(
        1 for holdout in memory_bottleneck_holdouts if holdout.memory_bottleneck_bucket_match
    )
    memory_bottleneck_peak_errors = [
        holdout.memory_bottleneck_peak_absolute_error_gb
        for holdout in memory_bottleneck_holdouts
        if holdout.memory_bottleneck_peak_absolute_error_gb is not None
    ]
    memory_bottleneck_fraction_errors = [
        holdout.memory_bottleneck_fraction_absolute_error
        for holdout in memory_bottleneck_holdouts
        if holdout.memory_bottleneck_fraction_absolute_error is not None
    ]
    phase_bottleneck_holdouts = [holdout for holdout in holdouts if holdout.phase_bottleneck_bucket_match is not None]
    phase_bottleneck_phase_match_count = sum(
        1 for holdout in phase_bottleneck_holdouts if holdout.phase_bottleneck_phase_match
    )
    phase_bottleneck_bucket_match_count = sum(
        1 for holdout in phase_bottleneck_holdouts if holdout.phase_bottleneck_bucket_match
    )
    phase_bottleneck_share_errors = [
        holdout.phase_bottleneck_share_absolute_error
        for holdout in phase_bottleneck_holdouts
        if holdout.phase_bottleneck_share_absolute_error is not None
    ]
    phase_top3_holdouts = [holdout for holdout in holdouts if holdout.phase_top3_overlap_rate is not None]
    phase_top3_overlap_rates = [holdout.phase_top3_overlap_rate for holdout in phase_top3_holdouts]
    phase_bucket_top3_overlap_rates = [
        holdout.phase_bucket_top3_overlap_rate
        for holdout in phase_top3_holdouts
        if holdout.phase_bucket_top3_overlap_rate is not None
    ]
    max_error_holdout = _max_holdout_by_field(holdouts, "absolute_percentage_error")
    max_uncertainty_holdout = _max_holdout_by_field(holdouts, "prediction_uncertainty_fraction")
    max_empirical_required_uncertainty_pair = (
        max(empirical_required_uncertainty_pairs, key=lambda item: (item[0], item[1].label))
        if empirical_required_uncertainty_pairs
        else None
    )
    max_memory_error_holdout = _max_holdout_by_field(holdouts, "memory_absolute_error_gb")
    max_memory_percentage_error_holdout = _max_holdout_by_field(holdouts, "memory_absolute_percentage_error")
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
    max_memory_bottleneck_peak_error_holdout = _max_holdout_by_field(
        holdouts,
        "memory_bottleneck_peak_absolute_error_gb",
    )
    max_memory_bottleneck_fraction_error_holdout = _max_holdout_by_field(
        holdouts,
        "memory_bottleneck_fraction_absolute_error",
    )
    max_phase_bottleneck_share_error_holdout = _max_holdout_by_field(
        holdouts,
        "phase_bottleneck_share_absolute_error",
    )
    min_phase_top3_overlap_holdout = _min_holdout_by_field(holdouts, "phase_top3_overlap_rate")
    min_phase_bucket_top3_overlap_holdout = _min_holdout_by_field(holdouts, "phase_bucket_top3_overlap_rate")
    status = "ok" if errors else "insufficient_data"
    if holdouts and not errors:
        warnings.append("all holdouts were unscored")
    calibration_fidelity_status, calibration_fidelity_blockers = _calibration_fidelity_support(
        holdouts,
        errors=errors,
        interval_covered_count=interval_covered_count,
        uncertainty_fractions=uncertainty_fractions,
        empirical_required_uncertainty_fractions=empirical_required_uncertainty_fractions,
        memory_percentage_errors=memory_percentage_errors,
        memory_bottleneck_evaluated_count=len(memory_bottleneck_holdouts),
        memory_bottleneck_bucket_match_count=memory_bottleneck_bucket_match_count,
        phase_bottleneck_evaluated_count=len(phase_bottleneck_holdouts),
        phase_bottleneck_bucket_match_count=phase_bottleneck_bucket_match_count,
        phase_top3_evaluated_count=len(phase_top3_holdouts),
        phase_bucket_top3_overlap_rates=phase_bucket_top3_overlap_rates,
        phase_bottleneck_share_errors=phase_bottleneck_share_errors,
    )
    calibration_validation_gaps = _calibration_validation_gap_portfolio(holdouts, calibration_fidelity_blockers)
    calibration_design_configs = _materialize_calibration_measurement_design_configs_from_context(
        base_config=base_config,
        base_topology=base_topology,
        behavior_points=behavior_points,
        calibration_validation_gaps=calibration_validation_gaps,
        world_size=world_size,
        local_world_size=local_world_size,
        max_configs_per_measurement=4,
    )

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
        max_absolute_percentage_error_label=max_error_holdout.label if max_error_holdout is not None else None,
        max_absolute_percentage_error_prediction_status=(
            max_error_holdout.prediction_status if max_error_holdout is not None else None
        ),
        max_absolute_percentage_error_in_prediction_interval=(
            max_error_holdout.actual_tokens_in_prediction_interval if max_error_holdout is not None else None
        ),
        prediction_interval_coverage_count=interval_covered_count,
        prediction_interval_coverage_rate=round(interval_covered_count / len(errors), 3) if errors else None,
        mean_prediction_uncertainty_fraction=(
            round(statistics.fmean(uncertainty_fractions), 3) if uncertainty_fractions else None
        ),
        max_prediction_uncertainty_fraction=round(max(uncertainty_fractions), 3) if uncertainty_fractions else None,
        max_prediction_uncertainty_label=(
            max_uncertainty_holdout.label if max_uncertainty_holdout is not None else None
        ),
        memory_evaluated_count=len(memory_percentage_errors),
        mean_memory_absolute_error_gb=round(statistics.fmean(memory_absolute_errors), 3)
        if memory_absolute_errors
        else None,
        max_memory_absolute_error_gb=round(max(memory_absolute_errors), 3) if memory_absolute_errors else None,
        max_memory_absolute_error_label=(
            max_memory_error_holdout.label if max_memory_error_holdout is not None else None
        ),
        mean_memory_absolute_percentage_error=round(statistics.fmean(memory_percentage_errors), 3)
        if memory_percentage_errors
        else None,
        max_memory_absolute_percentage_error=round(max(memory_percentage_errors), 3)
        if memory_percentage_errors
        else None,
        max_memory_absolute_percentage_error_label=(
            max_memory_percentage_error_holdout.label if max_memory_percentage_error_holdout is not None else None
        ),
        memory_prediction_basis_counts=dict(sorted(memory_basis_counts.items())),
        memory_coverage_status_counts=dict(sorted(memory_coverage_counts.items())),
        memory_feasibility_status_counts=dict(sorted(memory_feasibility_counts.items())),
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
        memory_bottleneck_evaluated_count=len(memory_bottleneck_holdouts),
        memory_bottleneck_phase_match_count=memory_bottleneck_phase_match_count,
        memory_bottleneck_phase_match_rate=(
            round(memory_bottleneck_phase_match_count / len(memory_bottleneck_holdouts), 3)
            if memory_bottleneck_holdouts
            else None
        ),
        memory_bottleneck_bucket_match_count=memory_bottleneck_bucket_match_count,
        memory_bottleneck_bucket_match_rate=(
            round(memory_bottleneck_bucket_match_count / len(memory_bottleneck_holdouts), 3)
            if memory_bottleneck_holdouts
            else None
        ),
        mean_memory_bottleneck_peak_absolute_error_gb=(
            round(statistics.fmean(memory_bottleneck_peak_errors), 3) if memory_bottleneck_peak_errors else None
        ),
        max_memory_bottleneck_peak_absolute_error_gb=(
            round(max(memory_bottleneck_peak_errors), 3) if memory_bottleneck_peak_errors else None
        ),
        max_memory_bottleneck_peak_absolute_error_label=(
            max_memory_bottleneck_peak_error_holdout.label
            if max_memory_bottleneck_peak_error_holdout is not None
            else None
        ),
        mean_memory_bottleneck_fraction_absolute_error=(
            round(statistics.fmean(memory_bottleneck_fraction_errors), 3) if memory_bottleneck_fraction_errors else None
        ),
        max_memory_bottleneck_fraction_absolute_error=(
            round(max(memory_bottleneck_fraction_errors), 3) if memory_bottleneck_fraction_errors else None
        ),
        max_memory_bottleneck_fraction_absolute_error_label=(
            max_memory_bottleneck_fraction_error_holdout.label
            if max_memory_bottleneck_fraction_error_holdout is not None
            else None
        ),
        memory_bottleneck_phase_mismatch_labels=[
            holdout.label for holdout in memory_bottleneck_holdouts if holdout.memory_bottleneck_phase_match is False
        ],
        memory_bottleneck_bucket_mismatch_labels=[
            holdout.label for holdout in memory_bottleneck_holdouts if holdout.memory_bottleneck_bucket_match is False
        ],
        phase_bottleneck_evaluated_count=len(phase_bottleneck_holdouts),
        phase_bottleneck_phase_match_count=phase_bottleneck_phase_match_count,
        phase_bottleneck_phase_match_rate=(
            round(phase_bottleneck_phase_match_count / len(phase_bottleneck_holdouts), 3)
            if phase_bottleneck_holdouts
            else None
        ),
        phase_bottleneck_bucket_match_count=phase_bottleneck_bucket_match_count,
        phase_bottleneck_bucket_match_rate=(
            round(phase_bottleneck_bucket_match_count / len(phase_bottleneck_holdouts), 3)
            if phase_bottleneck_holdouts
            else None
        ),
        mean_phase_bottleneck_share_absolute_error=(
            round(statistics.fmean(phase_bottleneck_share_errors), 6) if phase_bottleneck_share_errors else None
        ),
        max_phase_bottleneck_share_absolute_error=(
            round(max(phase_bottleneck_share_errors), 6) if phase_bottleneck_share_errors else None
        ),
        max_phase_bottleneck_share_absolute_error_label=(
            max_phase_bottleneck_share_error_holdout.label
            if max_phase_bottleneck_share_error_holdout is not None
            else None
        ),
        phase_bottleneck_phase_mismatch_labels=[
            holdout.label for holdout in phase_bottleneck_holdouts if holdout.phase_bottleneck_phase_match is False
        ],
        phase_bottleneck_bucket_mismatch_labels=[
            holdout.label for holdout in phase_bottleneck_holdouts if holdout.phase_bottleneck_bucket_match is False
        ],
        phase_top3_evaluated_count=len(phase_top3_holdouts),
        mean_phase_top3_overlap_rate=(
            round(statistics.fmean(phase_top3_overlap_rates), 3) if phase_top3_overlap_rates else None
        ),
        min_phase_top3_overlap_rate=round(min(phase_top3_overlap_rates), 3) if phase_top3_overlap_rates else None,
        min_phase_top3_overlap_rate_label=(
            min_phase_top3_overlap_holdout.label if min_phase_top3_overlap_holdout is not None else None
        ),
        mean_phase_bucket_top3_overlap_rate=(
            round(statistics.fmean(phase_bucket_top3_overlap_rates), 3) if phase_bucket_top3_overlap_rates else None
        ),
        min_phase_bucket_top3_overlap_rate=(
            round(min(phase_bucket_top3_overlap_rates), 3) if phase_bucket_top3_overlap_rates else None
        ),
        min_phase_bucket_top3_overlap_rate_label=(
            min_phase_bucket_top3_overlap_holdout.label if min_phase_bucket_top3_overlap_holdout is not None else None
        ),
        calibration_fidelity_status=calibration_fidelity_status,
        calibration_fidelity_blockers=calibration_fidelity_blockers,
        calibration_validation_gap_count=len(calibration_validation_gaps),
        calibration_validation_gap_status_counts=_calibration_gap_status_counts(calibration_validation_gaps),
        calibration_validation_gap_required_measurements=_unique_required_measurements(calibration_validation_gaps),
        calibration_validation_gaps=calibration_validation_gaps,
        prediction_status_counts=dict(sorted(status_counts.items())),
        holdouts=holdouts,
        warnings=warnings,
        prediction_uncertainty_calibration_status=_prediction_uncertainty_calibration_status(
            errors=errors,
            interval_covered_count=interval_covered_count,
            uncertainty_fractions=uncertainty_fractions,
            empirical_required_uncertainty_fractions=empirical_required_uncertainty_fractions,
        ),
        mean_empirical_required_uncertainty_fraction=(
            round(statistics.fmean(empirical_required_uncertainty_fractions), 3)
            if empirical_required_uncertainty_fractions
            else None
        ),
        max_empirical_required_uncertainty_fraction=(
            round(max(empirical_required_uncertainty_fractions), 3)
            if empirical_required_uncertainty_fractions
            else None
        ),
        max_empirical_required_uncertainty_label=(
            max_empirical_required_uncertainty_pair[1].label
            if max_empirical_required_uncertainty_pair is not None
            else None
        ),
        measurement_design_config_count=len(calibration_design_configs),
        measurement_design_config_labels=[item.label for item in calibration_design_configs],
        measurement_design_config_filenames=[f"design_{item.filename}" for item in calibration_design_configs],
        calibration_support_benchmark_dirs=[str(path) for path in calibration_support_benchmark_dirs],
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
    parser.add_argument(
        "--calibration-support-benchmark-dir",
        dest="calibration_support_benchmark_dirs",
        action="append",
        type=Path,
        default=[],
        help="Additional benchmark dir whose rows can support predictions but are not target holdouts",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--write-measurement-configs",
        type=Path,
        default=None,
        help="Write bounded calibration replay/profile measurement configs in this directory",
    )
    args = parser.parse_args()

    args.config, args.benchmark_dir = resolve_pack_inputs(args.pack, args.config, args.benchmark_dir)
    if args.config is None or args.benchmark_dir is None:
        parser.error("provide --pack, or both --config and --benchmark-dir")

    report = evaluate_calibration(
        args.config,
        benchmark_dir=args.benchmark_dir,
        world_size=args.world_size,
        local_world_size=args.local_world_size,
        calibration_support_benchmark_dirs=tuple(args.calibration_support_benchmark_dirs),
        device_memory_limit_gb=args.device_memory_limit_gb,
        memory_safety_factor=args.memory_safety_factor,
    )
    if args.write_measurement_configs:
        write_measurement_configs(report, args.write_measurement_configs)
    rendered = json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
