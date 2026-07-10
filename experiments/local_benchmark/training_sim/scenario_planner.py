"""Plan and score topology scenarios from a base XoRL training config."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


try:
    from .benchmark_behavior import (
        H100_BF16_PROMISED_TFLOPS_PER_GPU,
        behavior_point_matches_topology,
        behavior_point_matches_workload,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from .config_fingerprint import load_training_config, resolve_topology
    from .memory_ledger import build_memory_ledger
    from .model_metadata import resolve_model_metadata
    from .schemas import (
        BenchmarkBehaviorPoint,
        BenchmarkBehaviorPrediction,
        ModelMetadata,
        ScenarioCandidate,
        ScenarioReport,
        Topology,
        to_jsonable,
    )
    from .shape_engine import ShapeLedger, build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import (
        H100_BF16_PROMISED_TFLOPS_PER_GPU,
        behavior_point_matches_topology,
        behavior_point_matches_workload,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from config_fingerprint import load_training_config, resolve_topology
    from memory_ledger import build_memory_ledger
    from model_metadata import resolve_model_metadata
    from schemas import (
        BenchmarkBehaviorPoint,
        BenchmarkBehaviorPrediction,
        ModelMetadata,
        ScenarioCandidate,
        ScenarioReport,
        Topology,
        to_jsonable,
    )
    from shape_engine import ShapeLedger, build_shape_ledger


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if isinstance(value, dict):
        return value
    raw[name] = {}
    return raw[name]


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw == "auto":
        return None
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _divisors(value: int) -> list[int]:
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def _power_of_two_divisors(value: int, *, max_value: int | None = None) -> list[int]:
    limit = value if max_value is None else min(value, max_value)
    return [candidate for candidate in _divisors(value) if candidate <= limit and candidate & (candidate - 1) == 0]


def _dedupe_sorted(values: list[int] | set[int]) -> list[int]:
    return sorted(value for value in set(values) if value > 0)


def _default_micro_batch_sizes(
    base_topology: Topology,
    behavior_points: list[BenchmarkBehaviorPoint],
) -> list[int]:
    values = {base_topology.micro_batch_size}
    values.update(point.micro_batch_size for point in behavior_points if point.micro_batch_size is not None)
    return sorted(values)


def _default_ep_sizes(base_topology: Topology) -> list[int]:
    if base_topology.num_experts is None:
        return [base_topology.expert_parallel_size]
    ranks_per_pipeline_stage = base_topology.world_size // base_topology.pipeline_parallel_size
    values = {
        value for value in _divisors(ranks_per_pipeline_stage) if value > 0 and base_topology.num_experts % value == 0
    }
    if base_topology.expert_parallel_size in values:
        return [base_topology.expert_parallel_size]
    return sorted(values) or [base_topology.expert_parallel_size]


def _auto_ep_sizes(base_topology: Topology) -> list[int]:
    if base_topology.num_experts is None:
        return [base_topology.expert_parallel_size]
    values = {
        value
        for value in _divisors(base_topology.world_size)
        if base_topology.num_experts % value == 0 and value <= base_topology.world_size
    }
    values.add(base_topology.expert_parallel_size)
    return _dedupe_sorted(values)


def _auto_tensor_parallel_sizes(base_topology: Topology, metadata: ModelMetadata) -> list[int]:
    values = set(_power_of_two_divisors(base_topology.world_size, max_value=base_topology.local_world_size))
    values.add(base_topology.tensor_parallel_size)
    if metadata.hidden_size is not None:
        values = {value for value in values if metadata.hidden_size % value == 0}
    if metadata.num_attention_heads is not None:
        values = {value for value in values if metadata.num_attention_heads % value == 0}
    return _dedupe_sorted(values) or [base_topology.tensor_parallel_size]


def _auto_pipeline_parallel_sizes(base_topology: Topology, metadata: ModelMetadata) -> list[int]:
    values = set(_power_of_two_divisors(base_topology.world_size, max_value=4))
    values.add(base_topology.pipeline_parallel_size)
    if metadata.num_hidden_layers is not None:
        values = {value for value in values if metadata.num_hidden_layers % value == 0}
        values.add(base_topology.pipeline_parallel_size)
    return _dedupe_sorted(values) or [base_topology.pipeline_parallel_size]


def _auto_ulysses_parallel_sizes(base_topology: Topology) -> list[int]:
    values = {base_topology.ulysses_parallel_size, 1}
    seq_len = base_topology.sample_packing_sequence_len or 0
    if seq_len >= 16_384:
        values.update(_power_of_two_divisors(base_topology.world_size, max_value=64))
    return _dedupe_sorted(values)


def _auto_ringattn_parallel_sizes(base_topology: Topology) -> list[int]:
    values = {base_topology.ringattn_parallel_size, 1}
    seq_len = base_topology.sample_packing_sequence_len or 0
    if seq_len >= 64_000:
        values.update(_power_of_two_divisors(base_topology.world_size, max_value=4))
    return _dedupe_sorted(values)


def _known_or_default_parallel_size(point_value: int | None) -> int:
    return point_value if point_value is not None else 1


def _point_matches_topology_parallel_dims(point: BenchmarkBehaviorPoint, topology: Topology) -> bool:
    return (
        _known_or_default_parallel_size(point.tensor_parallel_size) == topology.tensor_parallel_size
        and _known_or_default_parallel_size(point.pipeline_parallel_size) == topology.pipeline_parallel_size
        and _known_or_default_parallel_size(point.ulysses_parallel_size) == topology.ulysses_parallel_size
        and _known_or_default_parallel_size(point.ringattn_parallel_size) == topology.ringattn_parallel_size
    )


def _point_matches_parallel_dims_for_risk(point: BenchmarkBehaviorPoint, topology: Topology) -> bool:
    if point.expert_parallel_size is not None and point.expert_parallel_size != topology.expert_parallel_size:
        return False
    if point.ep_fsdp_size is not None and point.ep_fsdp_size != topology.ep_fsdp_size:
        return False
    return _point_matches_topology_parallel_dims(point, topology)


def _calibration_scope(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    *,
    prediction_confidence: str,
) -> str:
    if prediction_confidence == "calibrated":
        return "exact_calibrated"

    throughput_points = [point for point in behavior_points if point.tokens_per_sec is not None]
    if not throughput_points:
        return "no_calibration"

    same_sequence_points = [
        point
        for point in throughput_points
        if point.sample_packing_sequence_len in (None, topology.sample_packing_sequence_len)
    ]
    if not same_sequence_points:
        return "outside_sequence_calibration_envelope"

    dimensions: tuple[tuple[str, int], ...] = (
        ("micro_batch_size", topology.micro_batch_size),
        ("global_batch_size", topology.global_batch_size),
        ("expert_parallel_size", topology.expert_parallel_size),
        ("ep_fsdp_size", topology.ep_fsdp_size or 0),
        ("tensor_parallel_size", topology.tensor_parallel_size),
        ("pipeline_parallel_size", topology.pipeline_parallel_size),
        ("ulysses_parallel_size", topology.ulysses_parallel_size),
        ("ringattn_parallel_size", topology.ringattn_parallel_size),
    )
    for field_name, topology_value in dimensions:
        observed_values = [
            _known_or_default_parallel_size(getattr(point, field_name))
            if field_name.endswith("_parallel_size")
            else getattr(point, field_name)
            for point in same_sequence_points
            if getattr(point, field_name) is not None
        ]
        if not observed_values:
            continue
        if topology_value < min(observed_values) or topology_value > max(observed_values):
            return "outside_measured_envelope"
    return "inside_measured_envelope"


def _candidate_risk_flags(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    behavior: BenchmarkBehaviorPrediction,
    *,
    raw_config: dict[str, Any] | None,
    calibration_scope: str,
    prediction_confidence: str,
) -> list[str]:
    flags: list[str] = []
    if prediction_confidence != "calibrated":
        flags.append("requires_remeasurement")
    if calibration_scope.startswith("outside"):
        flags.append(calibration_scope)
    if behavior.correctness_status and behavior.correctness_status != "k3_pass":
        flags.append(f"correctness_{behavior.correctness_status}")

    matched_labels = {part.strip() for part in (behavior.matched_label or "").split(",") if part.strip()}
    for point in behavior_points:
        if raw_config is not None and point.label in matched_labels:
            for mismatch in behavior_point_workload_mismatches(point, raw_config):
                flags.append(f"runtime_mismatch:{mismatch}")
        same_sequence = point.sample_packing_sequence_len in (None, topology.sample_packing_sequence_len)
        if not same_sequence and point.sample_packing_sequence_len is not None:
            same_sequence = (
                topology.sample_packing_sequence_len is not None
                and topology.sample_packing_sequence_len >= point.sample_packing_sequence_len
            )
        if not same_sequence or not _point_matches_parallel_dims_for_risk(point, topology):
            continue

        if point.status == "allocator_pressure_slowdown":
            at_or_beyond_mbs = (
                point.micro_batch_size is not None and topology.micro_batch_size >= point.micro_batch_size
            )
            at_or_beyond_global_batch = (
                point.global_batch_size is not None and topology.global_batch_size >= point.global_batch_size
            )
            if point.label in matched_labels:
                flags.append("matched_allocator_pressure_slowdown")
            elif at_or_beyond_mbs or at_or_beyond_global_batch:
                flags.append(f"allocator_pressure_boundary:{point.label}")

        if point.correctness_status == "oom":
            at_or_beyond_sequence = (
                point.sample_packing_sequence_len is not None
                and topology.sample_packing_sequence_len is not None
                and topology.sample_packing_sequence_len >= point.sample_packing_sequence_len
            )
            at_or_beyond_mbs = (
                point.micro_batch_size is not None and topology.micro_batch_size >= point.micro_batch_size
            )
            at_or_beyond_global_batch = (
                point.global_batch_size is not None and topology.global_batch_size >= point.global_batch_size
            )
            if point.label in matched_labels or (
                at_or_beyond_sequence and (at_or_beyond_mbs or at_or_beyond_global_batch)
            ):
                flags.append(f"observed_oom_boundary:{point.label}")

    return sorted(set(flags))


def _risk_adjusted_score(
    score_tokens_per_sec: float | None,
    *,
    calibration_scope: str,
    risk_flags: list[str],
    feasibility_status: str,
) -> float | None:
    if score_tokens_per_sec is None:
        return None

    multiplier = 1.0
    if calibration_scope == "inside_measured_envelope":
        multiplier *= 0.85
    elif calibration_scope == "outside_measured_envelope":
        multiplier *= 0.65
    elif calibration_scope == "outside_sequence_calibration_envelope":
        multiplier *= 0.35
    elif calibration_scope == "no_calibration":
        multiplier *= 0.20

    if "matched_allocator_pressure_slowdown" in risk_flags:
        multiplier *= 0.35
    elif any(flag.startswith("allocator_pressure_boundary:") for flag in risk_flags):
        multiplier *= 0.50
    if any(flag.startswith("observed_oom_boundary:") for flag in risk_flags):
        multiplier *= 0.25

    for flag in risk_flags:
        if flag == "correctness_k3_fail":
            multiplier *= 0.50
        elif flag in {"correctness_not_promoted", "correctness_raw_speed_not_promoted_without_matching_k3_pass"}:
            multiplier *= 0.95
        elif flag == "correctness_not_promoted_extrapolated":
            multiplier *= 0.90
        elif flag == "correctness_runtime_failure_after_steps":
            multiplier *= 0.45
        elif flag == "correctness_missing_calibration":
            multiplier *= 0.50
        elif flag.startswith("correctness_"):
            multiplier *= 0.75

    if feasibility_status.endswith("_high_pressure"):
        multiplier *= 0.85
    elif feasibility_status.endswith("_moderate_pressure"):
        multiplier *= 0.95

    return round(score_tokens_per_sec * multiplier, 3)


def _recommendation(
    *,
    feasible: bool,
    promotable: bool,
    feasibility_status: str,
    risk_flags: list[str],
) -> str:
    if feasibility_status == "observed_oom":
        return "avoid_observed_oom"
    if not feasible:
        return "do_not_launch_unscored"
    if promotable:
        return "promote_candidate"
    if "matched_allocator_pressure_slowdown" in risk_flags or any(
        flag.startswith("allocator_pressure_boundary:") for flag in risk_flags
    ):
        return "measure_allocator_boundary"
    if any(flag.startswith("observed_oom_boundary:") for flag in risk_flags):
        return "remeasure_after_memory_fix"
    if "requires_remeasurement" in risk_flags:
        return "remeasure_before_ranking"
    if "correctness_runtime_failure_after_steps" in risk_flags:
        return "debug_runtime_failure"
    if any(flag.startswith("correctness_") for flag in risk_flags):
        return "correctness_gate_required"
    return "review_candidate"


def _mutated_config(
    base_config: dict[str, Any],
    *,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    expert_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    ulysses_parallel_size: int,
    ringattn_parallel_size: int,
) -> dict[str, Any]:
    raw_config = copy.deepcopy(base_config)
    train = _section(raw_config, "train")
    train["micro_batch_size"] = micro_batch_size
    train["gradient_accumulation_steps"] = gradient_accumulation_steps
    train["expert_parallel_size"] = expert_parallel_size
    train["tensor_parallel_size"] = tensor_parallel_size
    train["pipeline_parallel_size"] = pipeline_parallel_size
    train["ulysses_parallel_size"] = ulysses_parallel_size
    train["ringattn_parallel_size"] = ringattn_parallel_size

    non_dp_size = tensor_parallel_size * pipeline_parallel_size * ulysses_parallel_size * ringattn_parallel_size
    if non_dp_size <= 0 or world_size % non_dp_size != 0:
        raise ValueError("world_size is not divisible by non-DP parallelism product")
    data_parallel_size = world_size // non_dp_size
    train["data_parallel_replicate_size"] = 1
    train["data_parallel_shard_size"] = data_parallel_size
    if pipeline_parallel_size > 1:
        train["gradient_accumulation_steps"] = max(
            int(train.get("gradient_accumulation_steps", 1) or 1), pipeline_parallel_size
        )
    return raw_config


def _topology_label(topology: Topology) -> str:
    return (
        f"mbs{topology.micro_batch_size}-gb{topology.global_batch_size}-"
        f"ep{topology.expert_parallel_size}-efsdp{topology.ep_fsdp_size}-"
        f"tp{topology.tensor_parallel_size}-pp{topology.pipeline_parallel_size}-"
        f"u{topology.ulysses_parallel_size}-r{topology.ringattn_parallel_size}"
    )


def _reference_tokens_per_gpu(point: BenchmarkBehaviorPoint, topology: Topology) -> float | None:
    if point.tokens_per_sec is None:
        return None
    gpu_count = point.gpu_count or topology.world_size
    if gpu_count <= 0:
        return None
    return point.tokens_per_sec / gpu_count


def _select_reference_point(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    raw_config: dict[str, Any] | None = None,
) -> BenchmarkBehaviorPoint | None:
    usable = [
        point
        for point in behavior_points
        if point.tokens_per_sec is not None
        and point.micro_batch_size is not None
        and point.sample_packing_sequence_len in (None, topology.sample_packing_sequence_len)
        and _reference_tokens_per_gpu(point, topology) is not None
    ]
    if not usable:
        return None

    workload_compatible = (
        [point for point in usable if behavior_point_matches_workload(point, raw_config)]
        if raw_config is not None
        else usable
    )
    same_ep = [
        point for point in workload_compatible if point.expert_parallel_size in (None, topology.expert_parallel_size)
    ]
    candidates = same_ep or workload_compatible or usable

    def key(point: BenchmarkBehaviorPoint) -> tuple[float, float, float]:
        mismatch_count = len(behavior_point_workload_mismatches(point, raw_config)) if raw_config is not None else 0
        mbs_distance = abs((point.micro_batch_size or 1) - topology.micro_batch_size)
        per_gpu = _reference_tokens_per_gpu(point, topology) or 0.0
        return (-mismatch_count, -mbs_distance, per_gpu)

    return max(candidates, key=key)


def _parallelism_factor(reference: BenchmarkBehaviorPoint, topology: Topology) -> tuple[float, list[str]]:
    notes: list[str] = []
    factor = 1.0
    if reference.micro_batch_size:
        mbs_ratio = topology.micro_batch_size / reference.micro_batch_size
        factor *= min(1.15, max(0.55, mbs_ratio**0.20))
    if reference.expert_parallel_size and reference.expert_parallel_size != topology.expert_parallel_size:
        ep_ratio = topology.expert_parallel_size / reference.expert_parallel_size
        factor *= max(0.70, 1.0 - 0.04 * abs(math.log2(ep_ratio)))
        notes.append(f"EP extrapolated from {reference.expert_parallel_size} to {topology.expert_parallel_size}")
    reference_tp = _known_or_default_parallel_size(reference.tensor_parallel_size)
    if reference_tp != topology.tensor_parallel_size:
        tp_ratio = topology.tensor_parallel_size / reference_tp
        factor *= 0.90 ** abs(math.log2(tp_ratio))
        notes.append("TP extrapolation uses conservative communication penalty")
    reference_pp = _known_or_default_parallel_size(reference.pipeline_parallel_size)
    if reference_pp != topology.pipeline_parallel_size:
        pp_delta = abs(topology.pipeline_parallel_size - reference_pp)
        factor *= 0.88**pp_delta
        notes.append("PP extrapolation uses conservative bubble penalty")
    reference_cp = _known_or_default_parallel_size(reference.ulysses_parallel_size) * _known_or_default_parallel_size(
        reference.ringattn_parallel_size
    )
    if reference_cp != topology.sequence_parallel_size:
        cp_ratio = topology.sequence_parallel_size / reference_cp
        if topology.sample_packing_sequence_len and topology.sample_packing_sequence_len >= 32768:
            factor *= min(1.10, 1.0 + 0.04 * abs(math.log2(cp_ratio)))
        else:
            factor *= 0.94 ** abs(math.log2(cp_ratio))
            notes.append("SP/CP extrapolation penalized for short-context workload")
    return factor, notes


def _step_time_fit_prediction(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    shape: ShapeLedger,
    raw_config: dict[str, Any] | None = None,
) -> BenchmarkBehaviorPrediction | None:
    if topology.sample_packing_sequence_len is None or shape.global_tokens_per_train_step is None:
        return None
    compatible = [
        point
        for point in behavior_points
        if point.tokens_per_sec is not None
        and point.global_batch_size is not None
        and point.micro_batch_size == topology.micro_batch_size
        and point.sample_packing_sequence_len in (None, topology.sample_packing_sequence_len)
        and point.expert_parallel_size in (None, topology.expert_parallel_size)
        and point.ep_fsdp_size in (None, topology.ep_fsdp_size)
        and _point_matches_topology_parallel_dims(point, topology)
        and (raw_config is None or behavior_point_matches_workload(point, raw_config))
    ]
    best_by_global_batch: dict[int, BenchmarkBehaviorPoint] = {}
    for point in compatible:
        current = best_by_global_batch.get(point.global_batch_size)
        if current is None or (point.tokens_per_sec or 0.0) > (current.tokens_per_sec or 0.0):
            best_by_global_batch[point.global_batch_size] = point
    fit_points = sorted(best_by_global_batch.values(), key=lambda point: point.global_batch_size or 0)
    if len(fit_points) < 2:
        return None

    x_values: list[float] = []
    y_values: list[float] = []
    for point in fit_points:
        tokens = point.global_batch_size * topology.sample_packing_sequence_len
        step_time = point.step_time_sec
        if step_time is None and point.tokens_per_sec:
            step_time = tokens / point.tokens_per_sec
        if step_time is None:
            continue
        x_values.append(float(tokens))
        y_values.append(float(step_time))
    if len(x_values) < 2 or len(set(x_values)) < 2:
        return None

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if denominator == 0:
        return None
    slope = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x_values, y_values, strict=False))
    slope /= denominator
    intercept = y_mean - slope * x_mean
    predicted_step = intercept + slope * shape.global_tokens_per_train_step
    if predicted_step <= 0:
        return None
    tokens_per_sec = shape.global_tokens_per_train_step / predicted_step
    tokens_per_sec_per_gpu = tokens_per_sec / topology.world_size
    labels = ", ".join(point.label for point in fit_points)
    peak_mem_gb = max((point.peak_mem_gb for point in fit_points if point.peak_mem_gb is not None), default=None)
    return BenchmarkBehaviorPrediction(
        status="extrapolated_step_time_fit",
        matched_label=labels,
        source="step_time_fit",
        tokens_per_sec=round(tokens_per_sec, 3),
        tokens_per_sec_per_gpu=round(tokens_per_sec_per_gpu, 3),
        step_time_sec=round(predicted_step, 6),
        mfu_percent=None,
        tflops_per_gpu=None,
        promised_tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU,
        peak_mem_gb=peak_mem_gb,
        allocator_retries=None,
        derived_global_tokens_per_step=shape.global_tokens_per_train_step,
        correctness_status="not_promoted_extrapolated",
        warnings=[
            f"extrapolated step time from calibrated global batches: {labels}",
            f"fit_intercept_sec={intercept:.6f}",
            f"fit_sec_per_token={slope:.12f}",
            "correctness must be re-gated before promotion",
        ],
    )


def _memory_factor(
    memory_estimate_gb: float | None,
    *,
    memory_basis: str,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
) -> tuple[float, float | None, str]:
    if memory_estimate_gb is None:
        return 0.0, None, "unknown_memory_estimate"
    reserved_memory = memory_estimate_gb * memory_safety_factor
    headroom = device_memory_limit_gb - reserved_memory
    status_basis = "floor" if memory_basis == "analytic_floor" else memory_basis
    if headroom < 0:
        if memory_basis == "calibrated_peak" and memory_estimate_gb <= device_memory_limit_gb:
            return 0.75, headroom, f"feasible_{status_basis}_high_pressure"
        if memory_basis == "analytic_floor":
            return 0.0, headroom, "memory_floor_exceeds_limit"
        return 0.0, headroom, f"{status_basis}_exceeds_limit"
    utilization = reserved_memory / device_memory_limit_gb if device_memory_limit_gb else 1.0
    if utilization >= 0.90:
        return 0.75, headroom, f"feasible_{status_basis}_high_pressure"
    if utilization >= 0.80:
        return 0.90, headroom, f"feasible_{status_basis}_moderate_pressure"
    return 1.0, headroom, f"feasible_{status_basis}"


def _extrapolate_behavior(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    shape: ShapeLedger,
    *,
    raw_config: dict[str, Any] | None = None,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
    analytic_peak_floor_gb: float | None,
) -> tuple[BenchmarkBehaviorPrediction, list[str]]:
    step_fit = _step_time_fit_prediction(behavior_points, topology, shape, raw_config=raw_config)
    if step_fit is not None:
        return step_fit, ["step_time_fit_extrapolation"]

    reference = _select_reference_point(behavior_points, topology, raw_config=raw_config)
    if reference is None:
        return (
            BenchmarkBehaviorPrediction(
                status="unscored",
                matched_label=None,
                source=None,
                tokens_per_sec=None,
                tokens_per_sec_per_gpu=None,
                step_time_sec=None,
                mfu_percent=None,
                tflops_per_gpu=None,
                promised_tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU,
                peak_mem_gb=None,
                allocator_retries=None,
                derived_global_tokens_per_step=shape.global_tokens_per_train_step,
                correctness_status="missing_calibration",
                warnings=["no benchmark behavior point is available for extrapolation"],
            ),
            [],
        )

    ref_per_gpu = _reference_tokens_per_gpu(reference, topology) or 0.0
    parallel_factor, notes = _parallelism_factor(reference, topology)
    memory_factor, _, memory_status = _memory_factor(
        analytic_peak_floor_gb,
        memory_basis="analytic_floor",
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    tokens_per_sec_per_gpu = ref_per_gpu * parallel_factor * memory_factor
    tokens_per_sec = tokens_per_sec_per_gpu * topology.world_size
    step_time_sec = None
    if shape.global_tokens_per_train_step and tokens_per_sec:
        step_time_sec = shape.global_tokens_per_train_step / tokens_per_sec
    tflops_per_gpu = reference.tflops_per_gpu
    if tflops_per_gpu is None and reference.mfu_percent is not None:
        tflops_per_gpu = H100_BF16_PROMISED_TFLOPS_PER_GPU * reference.mfu_percent / 100.0
    if tflops_per_gpu is not None and ref_per_gpu:
        tflops_per_gpu *= tokens_per_sec_per_gpu / ref_per_gpu

    warnings = [
        f"extrapolated from {reference.label}; correctness must be re-gated before promotion",
        f"memory feasibility status is {memory_status}",
    ]
    if raw_config is not None:
        mismatches = behavior_point_workload_mismatches(reference, raw_config)
        if mismatches:
            warnings.append(f"reference runtime knobs differ: {', '.join(mismatches)}")
    warnings.extend(notes)
    return (
        BenchmarkBehaviorPrediction(
            status="extrapolated",
            matched_label=reference.label,
            source=reference.source,
            tokens_per_sec=round(tokens_per_sec, 3),
            tokens_per_sec_per_gpu=round(tokens_per_sec_per_gpu, 3),
            step_time_sec=round(step_time_sec, 6) if step_time_sec is not None else None,
            mfu_percent=None,
            tflops_per_gpu=round(tflops_per_gpu, 3) if tflops_per_gpu is not None else None,
            promised_tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU,
            peak_mem_gb=reference.peak_mem_gb,
            allocator_retries=None,
            derived_global_tokens_per_step=shape.global_tokens_per_train_step,
            correctness_status="not_promoted_extrapolated",
            warnings=warnings,
        ),
        notes,
    )


def _candidate_from_prediction(
    *,
    label: str,
    config_path: str | None,
    topology: Topology,
    shape: ShapeLedger,
    behavior: BenchmarkBehaviorPrediction,
    prediction_confidence: str,
    promotable: bool,
    behavior_points: list[BenchmarkBehaviorPoint],
    raw_config: dict[str, Any] | None,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
    analytic_peak_floor_gb: float | None,
    notes: list[str],
) -> ScenarioCandidate:
    estimated_peak_mem_gb = analytic_peak_floor_gb
    memory_basis = "analytic_floor"
    if behavior.peak_mem_gb is not None:
        if analytic_peak_floor_gb is None or behavior.peak_mem_gb >= analytic_peak_floor_gb:
            estimated_peak_mem_gb = behavior.peak_mem_gb
            memory_basis = "calibrated_peak" if prediction_confidence == "calibrated" else "extrapolated_peak"

    _, headroom, feasibility_status = _memory_factor(
        estimated_peak_mem_gb,
        memory_basis=memory_basis,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    if behavior.status == "calibrated_failure" or behavior.correctness_status == "oom":
        feasibility_status = "observed_oom"
    if behavior.tokens_per_sec is None and feasibility_status.startswith("feasible"):
        feasibility_status = "unscored"
    feasible = feasibility_status.startswith("feasible") and behavior.tokens_per_sec is not None
    score_tokens_per_sec = behavior.tokens_per_sec if feasible else None
    score_tokens_per_gpu_per_sec = behavior.tokens_per_sec_per_gpu if feasible else None
    max_ep_slots = max(shape.ep_rank_slots_per_microbatch) if shape.ep_rank_slots_per_microbatch else None
    calibration_scope = _calibration_scope(
        behavior_points,
        topology,
        prediction_confidence=prediction_confidence,
    )
    risk_flags = _candidate_risk_flags(
        behavior_points,
        topology,
        behavior,
        raw_config=raw_config,
        calibration_scope=calibration_scope,
        prediction_confidence=prediction_confidence,
    )
    score_risk_adjusted_tokens_per_sec = _risk_adjusted_score(
        score_tokens_per_sec,
        calibration_scope=calibration_scope,
        risk_flags=risk_flags,
        feasibility_status=feasibility_status,
    )
    recommendation = _recommendation(
        feasible=feasible,
        promotable=promotable and feasible,
        feasibility_status=feasibility_status,
        risk_flags=risk_flags,
    )
    return ScenarioCandidate(
        label=label,
        config_path=config_path,
        topology=topology,
        behavior=behavior,
        prediction_confidence=prediction_confidence,
        promotable=promotable and feasible,
        feasibility_status=feasibility_status,
        score_tokens_per_sec=score_tokens_per_sec,
        score_tokens_per_gpu_per_sec=score_tokens_per_gpu_per_sec,
        score_risk_adjusted_tokens_per_sec=score_risk_adjusted_tokens_per_sec,
        analytic_peak_floor_gb=analytic_peak_floor_gb,
        estimated_peak_mem_gb=estimated_peak_mem_gb,
        memory_basis=memory_basis,
        memory_headroom_gb=round(headroom, 3) if headroom is not None else None,
        max_ep_rank_slots_per_microbatch=max_ep_slots,
        calibration_scope=calibration_scope,
        recommendation=recommendation,
        risk_flags=risk_flags,
        notes=notes,
    )


def _candidate_sort_key(candidate: ScenarioCandidate) -> tuple[float, float]:
    return (
        candidate.score_tokens_per_sec if candidate.score_tokens_per_sec is not None else float("-inf"),
        candidate.score_tokens_per_gpu_per_sec if candidate.score_tokens_per_gpu_per_sec is not None else float("-inf"),
    )


def _risk_adjusted_sort_key(candidate: ScenarioCandidate) -> tuple[float, float]:
    return (
        candidate.score_risk_adjusted_tokens_per_sec
        if candidate.score_risk_adjusted_tokens_per_sec is not None
        else float("-inf"),
        candidate.score_tokens_per_sec if candidate.score_tokens_per_sec is not None else float("-inf"),
    )


def plan_scenario(
    base_config_path: str | Path,
    *,
    benchmark_dir: str | Path | None = None,
    world_size: int | None = None,
    local_world_size: int | None = None,
    micro_batch_sizes: list[int] | None = None,
    gradient_accumulation_steps: list[int] | None = None,
    expert_parallel_sizes: list[int] | None = None,
    tensor_parallel_sizes: list[int] | None = None,
    pipeline_parallel_sizes: list[int] | None = None,
    ulysses_parallel_sizes: list[int] | None = None,
    ringattn_parallel_sizes: list[int] | None = None,
    topology_sweep: str = "base",
    device_memory_limit_gb: float = 80.0,
    memory_safety_factor: float = 1.15,
) -> ScenarioReport:
    if topology_sweep not in {"base", "auto"}:
        raise ValueError("topology_sweep must be 'base' or 'auto'")
    base_path = Path(base_config_path)
    base_config = load_training_config(base_path)
    base_topology = resolve_topology(base_config, world_size=world_size, local_world_size=local_world_size)
    resolved_world_size = world_size or base_topology.world_size
    resolved_local_world_size = local_world_size or base_topology.local_world_size
    behavior_points = load_benchmark_behavior_points(benchmark_dir) if benchmark_dir is not None else []
    metadata = resolve_model_metadata(base_config)

    micro_batch_values = micro_batch_sizes or _default_micro_batch_sizes(base_topology, behavior_points)
    gradient_accumulation_values = gradient_accumulation_steps or [base_topology.gradient_accumulation_steps]
    if topology_sweep == "auto":
        ep_values = expert_parallel_sizes or _auto_ep_sizes(base_topology)
        tp_values = tensor_parallel_sizes or _auto_tensor_parallel_sizes(base_topology, metadata)
        pp_values = pipeline_parallel_sizes or _auto_pipeline_parallel_sizes(base_topology, metadata)
        ulysses_values = ulysses_parallel_sizes or _auto_ulysses_parallel_sizes(base_topology)
        ring_values = ringattn_parallel_sizes or _auto_ringattn_parallel_sizes(base_topology)
    else:
        ep_values = expert_parallel_sizes or [base_topology.expert_parallel_size]
        tp_values = tensor_parallel_sizes or [base_topology.tensor_parallel_size]
        pp_values = pipeline_parallel_sizes or [base_topology.pipeline_parallel_size]
        ulysses_values = ulysses_parallel_sizes or [base_topology.ulysses_parallel_size]
        ring_values = ringattn_parallel_sizes or [base_topology.ringattn_parallel_size]

    candidates: list[ScenarioCandidate] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for pp in pp_values:
        for tp in tp_values:
            for ulysses in ulysses_values:
                for ringattn in ring_values:
                    for ep in ep_values:
                        for micro_batch_size in micro_batch_values:
                            for gradient_accumulation_step in gradient_accumulation_values:
                                try:
                                    raw_config = _mutated_config(
                                        base_config,
                                        world_size=resolved_world_size,
                                        micro_batch_size=micro_batch_size,
                                        gradient_accumulation_steps=gradient_accumulation_step,
                                        expert_parallel_size=ep,
                                        tensor_parallel_size=tp,
                                        pipeline_parallel_size=pp,
                                        ulysses_parallel_size=ulysses,
                                        ringattn_parallel_size=ringattn,
                                    )
                                    topology = resolve_topology(
                                        raw_config,
                                        world_size=resolved_world_size,
                                        local_world_size=resolved_local_world_size,
                                    )
                                except ValueError as exc:
                                    warnings.append(
                                        f"skipped mbs={micro_batch_size}, ga={gradient_accumulation_step}, "
                                        f"ep={ep}, tp={tp}, pp={pp}, u={ulysses}, r={ringattn}: {exc}"
                                    )
                                    continue
                                if topology.ep_fsdp_size is None:
                                    warnings.append(f"skipped {_topology_label(topology)}: ep_fsdp is not integral")
                                    continue
                                if (
                                    topology.num_experts is not None
                                    and topology.num_experts % topology.expert_parallel_size
                                ):
                                    warnings.append(
                                        f"skipped {_topology_label(topology)}: EP does not divide num_experts"
                                    )
                                    continue

                                shape = build_shape_ledger(topology, balanced_routing=True)
                                memory = build_memory_ledger(
                                    raw_config,
                                    topology=topology,
                                    model_metadata=metadata,
                                )
                                exact_points = [
                                    point
                                    for point in behavior_points
                                    if behavior_point_matches_topology(point, topology)
                                    and behavior_point_matches_workload(point, raw_config)
                                ]
                                if exact_points:
                                    for point in exact_points:
                                        behavior = predict_benchmark_behavior([point], topology, shape, raw_config)
                                        label = f"{_topology_label(topology)}:{point.label}"
                                        key = (label, point.source)
                                        if key in seen:
                                            continue
                                        seen.add(key)
                                        candidates.append(
                                            _candidate_from_prediction(
                                                label=label,
                                                config_path=str(base_path),
                                                topology=topology,
                                                shape=shape,
                                                behavior=behavior,
                                                prediction_confidence="calibrated",
                                                promotable=point.correctness_status == "k3_pass",
                                                behavior_points=behavior_points,
                                                raw_config=raw_config,
                                                device_memory_limit_gb=device_memory_limit_gb,
                                                memory_safety_factor=memory_safety_factor,
                                                analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
                                                notes=list(point.notes),
                                            )
                                        )
                                    continue

                                behavior, extrapolation_notes = _extrapolate_behavior(
                                    behavior_points,
                                    topology,
                                    shape,
                                    raw_config=raw_config,
                                    device_memory_limit_gb=device_memory_limit_gb,
                                    memory_safety_factor=memory_safety_factor,
                                    analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
                                )
                                label = f"{_topology_label(topology)}:extrapolated"
                                key = (label, behavior.source or "")
                                if key in seen:
                                    continue
                                seen.add(key)
                                candidates.append(
                                    _candidate_from_prediction(
                                        label=label,
                                        config_path=None,
                                        topology=topology,
                                        shape=shape,
                                        behavior=behavior,
                                        prediction_confidence=behavior.status,
                                        promotable=False,
                                        behavior_points=behavior_points,
                                        raw_config=raw_config,
                                        device_memory_limit_gb=device_memory_limit_gb,
                                        memory_safety_factor=memory_safety_factor,
                                        analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
                                        notes=extrapolation_notes,
                                    )
                                )

    candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
    feasible = [candidate for candidate in candidates if candidate.score_tokens_per_sec is not None]
    best_raw = feasible[0] if feasible else None
    risk_adjusted = [candidate for candidate in feasible if candidate.score_risk_adjusted_tokens_per_sec is not None]
    best_risk_adjusted = max(risk_adjusted, key=_risk_adjusted_sort_key) if risk_adjusted else None
    next_measurement = [candidate for candidate in risk_adjusted if "requires_remeasurement" in candidate.risk_flags]
    best_next_measurement = max(next_measurement, key=_risk_adjusted_sort_key) if next_measurement else None
    promotable = [candidate for candidate in feasible if candidate.promotable]
    best_promotable = promotable[0] if promotable else None
    if best_raw is not None and not best_raw.promotable:
        warnings.append(f"best raw scenario {best_raw.label} is not correctness-promotable")
    if best_raw is not None and best_risk_adjusted is not None and best_raw.label != best_risk_adjusted.label:
        warnings.append(
            f"best raw scenario {best_raw.label} differs from risk-adjusted choice {best_risk_adjusted.label}"
        )
    if best_promotable is None:
        warnings.append("no correctness-promotable scenario found")

    return ScenarioReport(
        base_config_path=str(base_path),
        benchmark_dir=str(benchmark_dir) if benchmark_dir is not None else None,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
        topology_sweep=topology_sweep,
        candidate_count=len(candidates),
        feasible_count=len(feasible),
        best_raw=best_raw,
        best_risk_adjusted=best_risk_adjusted,
        best_next_measurement=best_next_measurement,
        best_promotable=best_promotable,
        candidates=candidates,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-world-size", type=int, default=None)
    parser.add_argument("--micro-batch-sizes", default=None, help="Comma list, or auto when omitted")
    parser.add_argument(
        "--gradient-accumulation-steps", default=None, help="Comma list, or base config GA when omitted"
    )
    parser.add_argument("--expert-parallel-sizes", default=None, help="Comma list, or base config EP when omitted")
    parser.add_argument("--tensor-parallel-sizes", default=None, help="Comma list, or base config TP when omitted")
    parser.add_argument("--pipeline-parallel-sizes", default=None, help="Comma list, or base config PP when omitted")
    parser.add_argument(
        "--ulysses-parallel-sizes", default=None, help="Comma list, or base config Ulysses when omitted"
    )
    parser.add_argument("--ringattn-parallel-sizes", default=None, help="Comma list, or base config Ring when omitted")
    parser.add_argument(
        "--topology-sweep",
        choices=("base", "auto"),
        default="base",
        help="Use base topology dimensions, or derive an automatic TP/PP/CP/EP sweep",
    )
    parser.add_argument("--device-memory-limit-gb", type=float, default=80.0)
    parser.add_argument("--memory-safety-factor", type=float, default=1.15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = plan_scenario(
        args.config,
        benchmark_dir=args.benchmark_dir,
        world_size=args.world_size,
        local_world_size=args.local_world_size,
        micro_batch_sizes=_parse_int_list(args.micro_batch_sizes),
        gradient_accumulation_steps=_parse_int_list(args.gradient_accumulation_steps),
        expert_parallel_sizes=_parse_int_list(args.expert_parallel_sizes),
        tensor_parallel_sizes=_parse_int_list(args.tensor_parallel_sizes),
        pipeline_parallel_sizes=_parse_int_list(args.pipeline_parallel_sizes),
        ulysses_parallel_sizes=_parse_int_list(args.ulysses_parallel_sizes),
        ringattn_parallel_sizes=_parse_int_list(args.ringattn_parallel_sizes),
        topology_sweep=args.topology_sweep,
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
