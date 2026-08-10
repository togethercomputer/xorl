"""Plan and score topology scenarios from a base XoRL training config."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


try:
    from .analytical_ledgers import activation_ledger
    from .benchmark_behavior import (
        H100_BF16_PROMISED_TFLOPS_PER_GPU,
        _config_balanced_routing,
        behavior_point_matches_topology,
        behavior_point_matches_workload,
        behavior_point_model_mismatches,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from .calibration_packs import resolve_pack_inputs
    from .config_fingerprint import load_training_config, resolve_topology
    from .memory_ledger import build_memory_ledger
    from .model_metadata import model_ref_from_config, resolve_model_metadata
    from .runtime_config import runtime_training_config
    from .schemas import (
        BenchmarkBehaviorPoint,
        BenchmarkBehaviorPrediction,
        CommLedger,
        MemoryLedger,
        ModelMetadata,
        ParallelismAxisComparison,
        ParallelismBoundaryAxisCoverage,
        ParallelismBoundaryGroup,
        ScenarioBenchmarkSupport,
        ScenarioCandidate,
        ScenarioCaptureGap,
        ScenarioDecisionSummary,
        ScenarioMeasurementConfig,
        ScenarioParallelismAxisCoverage,
        ScenarioReadiness,
        ScenarioReport,
        ScenarioValidationAction,
        Topology,
        to_jsonable,
    )
    from .shape_engine import ShapeLedger, build_shape_ledger
    from .simulator_support import requested_simulator_surface, resolve_simulator_support
    from .timing_ledger import build_timing_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from analytical_ledgers import activation_ledger
    from benchmark_behavior import (
        H100_BF16_PROMISED_TFLOPS_PER_GPU,
        _config_balanced_routing,
        behavior_point_matches_topology,
        behavior_point_matches_workload,
        behavior_point_model_mismatches,
        behavior_point_workload_mismatches,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from calibration_packs import resolve_pack_inputs
    from config_fingerprint import load_training_config, resolve_topology
    from memory_ledger import build_memory_ledger
    from model_metadata import model_ref_from_config, resolve_model_metadata
    from runtime_config import runtime_training_config
    from schemas import (
        BenchmarkBehaviorPoint,
        BenchmarkBehaviorPrediction,
        CommLedger,
        MemoryLedger,
        ModelMetadata,
        ParallelismAxisComparison,
        ParallelismBoundaryAxisCoverage,
        ParallelismBoundaryGroup,
        ScenarioBenchmarkSupport,
        ScenarioCandidate,
        ScenarioCaptureGap,
        ScenarioDecisionSummary,
        ScenarioMeasurementConfig,
        ScenarioParallelismAxisCoverage,
        ScenarioReadiness,
        ScenarioReport,
        ScenarioValidationAction,
        Topology,
        to_jsonable,
    )
    from shape_engine import ShapeLedger, build_shape_ledger
    from simulator_support import requested_simulator_surface, resolve_simulator_support
    from timing_ledger import build_timing_ledger


_EXACT_CROSS_MODEL_SEQUENCE_RATIO_WINDOW = (0.90, 1.10)
_EXTRAPOLATED_CROSS_MODEL_SEQUENCE_RATIO_WINDOW = (0.45, 2.20)
_MIN_ULYSSES_SEQUENCE_LEN = 16_384
_MIN_RINGATTN_SEQUENCE_LEN = 64_000
_PHASE_BOTTLENECK_HALFSPEED_SCALE = 0.5


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


_SCENARIO_PARALLELISM_DIMENSIONS = (
    "world_size",
    "data_parallel_replicate_size",
    "data_parallel_shard_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
    "ep_fsdp_size",
    "ulysses_parallel_size",
    "ringattn_parallel_size",
)

_PARALLELISM_COMPARISON_DIMENSIONS = (
    "world_size",
    "node_count",
    "data_parallel_size",
    "data_parallel_replicate_size",
    "data_parallel_shard_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
    "ep_fsdp_size",
    "ulysses_parallel_size",
    "ringattn_parallel_size",
)

_PARALLELISM_AXIS_DIMENSIONS = {
    "world_size": (
        "world_size",
        "node_count",
        "data_parallel_size",
        "data_parallel_shard_size",
        "ep_fsdp_size",
    ),
    "dp_replicate": (
        "data_parallel_replicate_size",
        "data_parallel_size",
        "data_parallel_shard_size",
        "ep_fsdp_size",
    ),
    "dp_shard": (
        "data_parallel_shard_size",
        "data_parallel_size",
        "ep_fsdp_size",
    ),
    "tensor_parallel": (
        "tensor_parallel_size",
        "data_parallel_size",
        "data_parallel_shard_size",
        "ep_fsdp_size",
    ),
    "pipeline_parallel": (
        "pipeline_parallel_size",
        "data_parallel_size",
        "data_parallel_shard_size",
        "ep_fsdp_size",
    ),
    "expert_parallel": (
        "expert_parallel_size",
        "ep_fsdp_size",
    ),
    # expert_parallel co-varies with ep_fsdp at fixed dp (ep x ep_fsdp = dp_shard), mirroring the
    # converse declaration above; without it the one physical ep<->ep_fsdp pair is double-counted as
    # a confounded ep_fsdp-axis row even when it is a clean expert_parallel pair.
    "ep_fsdp": (
        "ep_fsdp_size",
        "expert_parallel_size",
    ),
    "ulysses": (
        "ulysses_parallel_size",
        "data_parallel_size",
        "data_parallel_shard_size",
        "ep_fsdp_size",
    ),
    "ringattn": (
        "ringattn_parallel_size",
        "data_parallel_size",
        "data_parallel_shard_size",
        "ep_fsdp_size",
    ),
}

_PARALLELISM_AXIS_PRIMARY_DIMENSIONS = {
    "world_size": ("world_size",),
    "dp_replicate": ("data_parallel_replicate_size",),
    "dp_shard": ("data_parallel_shard_size",),
    "tensor_parallel": ("tensor_parallel_size",),
    "pipeline_parallel": ("pipeline_parallel_size",),
    "expert_parallel": ("expert_parallel_size",),
    "ep_fsdp": ("ep_fsdp_size",),
    "ulysses": ("ulysses_parallel_size",),
    "ringattn": ("ringattn_parallel_size",),
}

_SCENARIO_WORKLOAD_DIMENSIONS = (
    "micro_batch_size",
    "gradient_accumulation_steps",
    "global_batch_size",
    "sample_packing_sequence_len",
)

_SCENARIO_RUNTIME_SIGNATURE_FIELDS = (
    ("train", "gradient_checkpointing_method"),
    ("train", "enable_activation_offload"),
    ("train", "activation_offload_prefetch_count"),
    ("train", "skip_param_upcast"),
    ("train", "fsdp_reduce_dtype"),
    ("train", "ce_mode"),
    ("model", "moe_implementation"),
    ("train", "moe_checkpoint_method"),
    ("train", "muon_momentum"),
    ("train", "muon_update_dtype"),
    ("model", "deepep_async_combine"),
    ("model", "deepep_num_sms"),
    ("model", "deepep_buffer_size_gb"),
    ("train", "enable_compile"),
    ("_simulator", "attention_backend"),
)

_SCENARIO_RUNTIME_DIMENSIONS = tuple(field_name for _, field_name in _SCENARIO_RUNTIME_SIGNATURE_FIELDS)

_RUNTIME_VARIANT_CONFIG_PATHS = {
    field_name: (section_name, field_name) for section_name, field_name in _SCENARIO_RUNTIME_SIGNATURE_FIELDS
}

_RUNTIME_VARIANT_BOOL_DEFAULTS = {
    "enable_activation_offload": False,
    "skip_param_upcast": False,
    "deepep_async_combine": False,
    "enable_compile": False,
}

_SCENARIO_BOUNDARY_WORKLOAD_DIMENSIONS = (
    *_SCENARIO_WORKLOAD_DIMENSIONS,
    "balanced_routing",
)

_SCENARIO_BOUNDARY_SIGNATURE_DIMENSIONS = (
    "micro_batch_size",
    "gradient_accumulation_steps",
    "sample_packing_sequence_len",
    "balanced_routing",
)

_CALIBRATION_DISTANCE_DIMENSIONS = (
    "world_size",
    "micro_batch_size",
    "global_batch_size",
    "sample_packing_sequence_len",
    "expert_parallel_size",
    "ep_fsdp_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "ulysses_parallel_size",
    "ringattn_parallel_size",
)


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


def _auto_ulysses_parallel_sizes(base_topology: Topology, metadata: ModelMetadata) -> list[int]:
    values = {base_topology.ulysses_parallel_size, 1}
    seq_len = base_topology.sample_packing_sequence_len or 0
    if seq_len >= _MIN_ULYSSES_SEQUENCE_LEN:
        max_ulysses = 64
        if metadata.num_key_value_heads is not None:
            max_ulysses = max(1, metadata.num_key_value_heads, base_topology.ulysses_parallel_size)
        values.update(_power_of_two_divisors(base_topology.world_size, max_value=max_ulysses))
    return _dedupe_sorted(values)


def _auto_ringattn_parallel_sizes(base_topology: Topology) -> list[int]:
    values = {base_topology.ringattn_parallel_size, 1}
    seq_len = base_topology.sample_packing_sequence_len or 0
    if seq_len >= _MIN_RINGATTN_SEQUENCE_LEN:
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


def _phase_bottleneck_note(phase_time_share: dict[str, float]) -> str | None:
    bottleneck = _phase_bottleneck(phase_time_share)
    if bottleneck is None:
        return None
    phase, share = bottleneck
    return f"phase_bottleneck={phase}:{share:.1%}"


def _phase_bottleneck_details(
    phase_time_share: dict[str, float],
    phase_time_sec: dict[str, float],
) -> tuple[str, str, float, float | None] | None:
    bottleneck = _phase_bottleneck(phase_time_share)
    if bottleneck is None:
        return None
    phase, share = bottleneck
    time_sec = phase_time_sec.get(phase)
    return phase, _phase_bucket(phase), round(share, 6), round(time_sec, 6) if time_sec is not None else None


def _phase_bottleneck_half_speedup_counterfactual(
    score_tokens_per_sec: float | None,
    phase_bottleneck_share: float | None,
) -> tuple[float | None, float | None]:
    if score_tokens_per_sec is None or score_tokens_per_sec <= 0:
        return None, None
    if phase_bottleneck_share is None or not 0 < phase_bottleneck_share <= 1:
        return None, None
    counterfactual_time_fraction = (
        1.0 - phase_bottleneck_share + (phase_bottleneck_share * _PHASE_BOTTLENECK_HALFSPEED_SCALE)
    )
    if counterfactual_time_fraction <= 0:
        return None, None
    speedup = 1.0 / counterfactual_time_fraction
    return round(score_tokens_per_sec * speedup, 3), round((speedup - 1.0) * 100.0, 3)


def _memory_bottleneck_details(
    phase_memory_peak_gb: dict[str, float],
    peak_mem_gb: float | None,
) -> tuple[str, str, float, float] | None:
    visible = {
        phase: peak
        for phase, peak in phase_memory_peak_gb.items()
        if peak > 0 and not _is_composite_phase_for_bottleneck(phase, set(phase_memory_peak_gb))
    }
    if not visible:
        visible = {phase: peak for phase, peak in phase_memory_peak_gb.items() if peak > 0}
    if not visible:
        return None
    phase, peak = max(visible.items(), key=lambda item: (item[1], item[0]))
    denominator = peak_mem_gb if peak_mem_gb is not None and peak_mem_gb > 0 else peak
    return phase, _phase_bucket(phase), round(peak, 3), round(peak / denominator, 3)


def _scenario_timing_coverage_status(
    behavior: BenchmarkBehaviorPrediction,
    *,
    prediction_confidence: str,
    calibration_scope: str,
    timing_coverage_status: str,
) -> str:
    if timing_coverage_status == "no_timing_calibration":
        return "no_timing_evidence"
    has_phase_timing = timing_coverage_status.endswith("_phase_timing")
    exact_calibrated = prediction_confidence == "calibrated" and calibration_scope == "exact_calibrated"
    if exact_calibrated:
        return "exact_phase_timing" if has_phase_timing else "exact_total_step_only"
    if behavior.status == "extrapolated_step_time_fit":
        return "step_time_fit_phase_timing" if has_phase_timing else "step_time_fit_total_step_only"
    if behavior.status == "cross_model_extrapolated" or calibration_scope == "cross_model_analog":
        return "cross_model_reference_phase_timing" if has_phase_timing else "cross_model_reference_total_step_only"
    return "reference_phase_timing_extrapolated" if has_phase_timing else "reference_total_step_extrapolated"


def _matched_labels(behavior: BenchmarkBehaviorPrediction) -> set[str]:
    return {part.strip() for part in (behavior.matched_label or "").split(",") if part.strip()}


def _point_dimension_value(point: BenchmarkBehaviorPoint, dimension: str) -> int | None:
    if dimension == "world_size":
        return point.gpu_count
    value = getattr(point, dimension)
    if value is None:
        return None
    if dimension.endswith("_parallel_size"):
        return _known_or_default_parallel_size(value)
    return value


def _topology_dimension_value(topology: Topology, dimension: str) -> int | None:
    if dimension == "world_size":
        return topology.world_size
    value = getattr(topology, dimension)
    if value is None:
        return None
    if dimension.endswith("_parallel_size"):
        return _known_or_default_parallel_size(value)
    return value


def _calibration_distance(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    behavior: BenchmarkBehaviorPrediction,
) -> tuple[float | None, list[str]]:
    matched_labels = _matched_labels(behavior)
    if not matched_labels:
        return None, []
    matched_points = [point for point in behavior_points if point.label in matched_labels]
    if not matched_points:
        return None, []

    options: list[tuple[float, str, list[tuple[float, str]]]] = []
    for point in matched_points:
        factors: list[tuple[float, str]] = []
        total = 0.0
        for dimension in _CALIBRATION_DISTANCE_DIMENSIONS:
            reference = _point_dimension_value(point, dimension)
            target = _topology_dimension_value(topology, dimension)
            if reference is None or target is None or reference <= 0 or target <= 0:
                continue
            distance = abs(math.log2(target / reference))
            if distance == 0.0:
                continue
            total += distance
            factors.append((distance, f"{dimension}:{reference}->{target}:log2={distance:.3f}"))
        options.append((total, point.label, factors))

    if not options:
        return None, []
    total, label, factors = min(options, key=lambda option: (option[0], option[1]))
    factor_notes = [f"reference={label}"]
    factor_notes.extend(note for _, note in sorted(factors, key=lambda factor: (-factor[0], factor[1])))
    return round(total, 3), factor_notes


def _calibration_scope(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    *,
    prediction_confidence: str,
    raw_config: dict[str, Any] | None,
) -> str:
    if prediction_confidence == "calibrated":
        return "exact_calibrated"
    if prediction_confidence == "cross_model_extrapolated":
        return "cross_model_analog"

    throughput_points = [
        point
        for point in behavior_points
        if point.tokens_per_sec is not None
        and (raw_config is None or not behavior_point_model_mismatches(point, raw_config))
    ]
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


def _same_or_longer_sequence_for_risk(point: BenchmarkBehaviorPoint, topology: Topology) -> bool:
    if point.sample_packing_sequence_len in (None, topology.sample_packing_sequence_len):
        return True
    return (
        point.sample_packing_sequence_len is not None
        and topology.sample_packing_sequence_len is not None
        and topology.sample_packing_sequence_len >= point.sample_packing_sequence_len
    )


def _real_routing_boundary_flags(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    *,
    raw_config: dict[str, Any] | None,
    prediction_confidence: str,
) -> list[str]:
    if raw_config is None or prediction_confidence == "calibrated" or _config_balanced_routing(raw_config):
        return []

    real_router_fits: list[BenchmarkBehaviorPoint] = []
    balanced_router_fits: list[BenchmarkBehaviorPoint] = []
    for point in behavior_points:
        if point.tokens_per_sec is None or point.micro_batch_size is None:
            continue
        if behavior_point_model_mismatches(point, raw_config):
            continue
        if not _same_or_longer_sequence_for_risk(point, topology):
            continue
        if not _point_matches_parallel_dims_for_risk(point, topology):
            continue

        mismatches = set(behavior_point_workload_mismatches(point, raw_config))
        if not mismatches:
            real_router_fits.append(point)
        elif point.balanced_routing is True and mismatches <= {"balanced_routing"}:
            balanced_router_fits.append(point)

    if not real_router_fits or not balanced_router_fits:
        return []

    max_real_router_mbs = max(point.micro_batch_size or 0 for point in real_router_fits)
    max_real_router_global_batch = max(point.global_batch_size or 0 for point in real_router_fits)
    outside_real_router_fit_envelope = topology.micro_batch_size > max_real_router_mbs or (
        max_real_router_global_batch > 0 and topology.global_batch_size > max_real_router_global_batch
    )
    if not outside_real_router_fit_envelope:
        return []

    balanced_fit_covers_target = any(
        (point.micro_batch_size or 0) >= topology.micro_batch_size
        and (point.global_batch_size or 0) >= topology.global_batch_size
        for point in balanced_router_fits
    )
    if not balanced_fit_covers_target:
        return []
    return ["balanced_routing_only_fit_boundary", "real_routing_outside_fit_envelope"]


def _candidate_risk_flags(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    behavior: BenchmarkBehaviorPrediction,
    *,
    raw_config: dict[str, Any] | None,
    calibration_scope: str,
    prediction_confidence: str,
    communication: CommLedger | None,
) -> list[str]:
    flags: list[str] = []
    if prediction_confidence != "calibrated":
        flags.append("requires_remeasurement")
    if prediction_confidence == "cross_model_extrapolated":
        flags.append("cross_model_analog")
    if calibration_scope.startswith("outside"):
        flags.append(calibration_scope)
    if behavior.correctness_status and behavior.correctness_status != "k3_pass":
        flags.append(f"correctness_{behavior.correctness_status}")
    if communication is not None and prediction_confidence != "calibrated":
        flags.extend(
            _communication_risk_flags(
                behavior_points,
                topology,
                behavior,
                communication,
                prediction_confidence=prediction_confidence,
            )
        )
    flags.extend(
        _real_routing_boundary_flags(
            behavior_points,
            topology,
            raw_config=raw_config,
            prediction_confidence=prediction_confidence,
        )
    )

    for phase, share in behavior.phase_time_share.items():
        bucket = _phase_bucket(phase)
        if bucket == "input" and share >= 0.15:
            flags.append("input_pipeline_bottleneck")
        elif bucket == "optimizer" and share >= 0.25:
            flags.append("optimizer_bottleneck")
        elif bucket == "communication" and share >= 0.20:
            flags.append("communication_bottleneck")

    matched_labels = {part.strip() for part in (behavior.matched_label or "").split(",") if part.strip()}
    for point in behavior_points:
        if raw_config is not None and point.label in matched_labels:
            for mismatch in behavior_point_workload_mismatches(point, raw_config):
                if prediction_confidence == "cross_model_extrapolated" and mismatch == "model_ref":
                    continue
                flags.append(f"runtime_mismatch:{mismatch}")
        if point.label in matched_labels:
            if point.expert_parallel_size is not None and point.expert_parallel_size != topology.expert_parallel_size:
                flags.append("parallelism_extrapolation:ep")
            if point.ep_fsdp_size is not None and point.ep_fsdp_size != topology.ep_fsdp_size:
                flags.append("parallelism_extrapolation:ep_fsdp")
            if _known_or_default_parallel_size(point.tensor_parallel_size) != topology.tensor_parallel_size:
                flags.append("parallelism_extrapolation:tp")
            if _known_or_default_parallel_size(point.pipeline_parallel_size) != topology.pipeline_parallel_size:
                flags.append("parallelism_extrapolation:pp")
            if _known_or_default_parallel_size(point.ulysses_parallel_size) != topology.ulysses_parallel_size:
                flags.append("parallelism_extrapolation:ulysses")
            if _known_or_default_parallel_size(point.ringattn_parallel_size) != topology.ringattn_parallel_size:
                flags.append("parallelism_extrapolation:ring")
            if point.measured_steps is not None and point.measured_steps < 3:
                flags.append("short_measurement_window")
            if point.tokens_per_sec_cv is not None and point.tokens_per_sec_cv >= 0.15:
                flags.append("high_throughput_variance")
        same_sequence = _same_or_longer_sequence_for_risk(point, topology)
        workload_compatible = raw_config is None or behavior_point_matches_workload(point, raw_config)
        if not workload_compatible or not same_sequence or not _point_matches_parallel_dims_for_risk(point, topology):
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
            if point.micro_batch_size is not None:
                at_or_beyond_batch = topology.micro_batch_size >= point.micro_batch_size
            else:
                at_or_beyond_batch = (
                    point.global_batch_size is not None and topology.global_batch_size >= point.global_batch_size
                )
            if point.label in matched_labels or (at_or_beyond_sequence and at_or_beyond_batch):
                flags.append(f"observed_oom_boundary:{point.label}")

    return sorted(set(flags))


def _risk_adjusted_score(
    score_tokens_per_sec: float | None,
    *,
    calibration_scope: str,
    calibration_distance: float | None,
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
    elif calibration_scope == "cross_model_analog":
        multiplier *= 0.25

    if calibration_scope != "exact_calibrated" and calibration_distance is not None and calibration_distance > 0:
        multiplier *= max(0.70, 1.0 - 0.04 * calibration_distance)

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
        elif flag == "short_measurement_window":
            multiplier *= 0.80
        elif flag == "high_throughput_variance":
            multiplier *= 0.75
        elif flag == "input_pipeline_bottleneck":
            multiplier *= 0.70
        elif flag == "optimizer_bottleneck":
            multiplier *= 0.85
        elif flag == "real_routing_outside_fit_envelope":
            multiplier *= 0.55
        elif flag == "balanced_routing_only_fit_boundary":
            multiplier *= 0.80
        elif flag == "cross_model_analog":
            multiplier *= 0.70
        elif flag == "parallelism_extrapolation:tp":
            multiplier *= 0.80
        elif flag == "parallelism_extrapolation:pp":
            multiplier *= 0.75
        elif flag in {"parallelism_extrapolation:ulysses", "parallelism_extrapolation:ring"}:
            multiplier *= 0.85
        elif flag == "parallelism_extrapolation:ep":
            multiplier *= 0.90
        elif flag == "parallelism_extrapolation:ep_fsdp":
            multiplier *= 0.95
        elif flag.startswith("simulator_surface_unsupported:"):
            multiplier *= 0.0
        elif flag.startswith("simulator_surface_partial:"):
            multiplier *= 0.60
        elif flag in {"communication_cross_node:tp", "communication_cross_node:pp", "communication_cross_node:cp"}:
            multiplier *= 0.85
        elif flag in {
            "communication_cross_node:ep",
            "communication_cross_node:fsdp",
            "communication_cross_node:ep_fsdp",
        }:
            multiplier *= 0.92

    if feasibility_status.endswith("_high_pressure"):
        multiplier *= 0.85
    elif feasibility_status.endswith("_moderate_pressure"):
        multiplier *= 0.95

    return round(score_tokens_per_sec * multiplier, 3)


def _prediction_uncertainty_fraction(
    behavior: BenchmarkBehaviorPrediction,
    *,
    prediction_confidence: str,
    calibration_scope: str,
    calibration_distance: float | None,
    risk_flags: list[str],
    memory_coverage_status: str,
) -> float | None:
    if behavior.tokens_per_sec is None:
        return None

    if prediction_confidence == "calibrated":
        fraction = 0.05
    elif prediction_confidence == "cross_model_extrapolated":
        fraction = 0.45
    else:
        fraction = 0.25

    if calibration_scope == "inside_measured_envelope":
        fraction += 0.05
    elif calibration_scope == "outside_measured_envelope":
        fraction += 0.15
    elif calibration_scope == "outside_sequence_calibration_envelope":
        fraction += 0.25
    elif calibration_scope == "no_calibration":
        fraction += 0.35
    elif calibration_scope == "cross_model_analog":
        fraction += 0.25

    if calibration_distance is not None and calibration_distance > 0:
        fraction += min(0.25, 0.03 * calibration_distance)
    if behavior.tokens_per_sec_cv is not None:
        fraction = max(fraction, min(0.60, behavior.tokens_per_sec_cv))

    for flag in risk_flags:
        if flag == "high_throughput_variance":
            fraction += 0.15
        elif flag == "short_measurement_window":
            fraction += 0.10
        elif flag == "cross_model_analog":
            fraction += 0.15
        elif flag.startswith("cross_model_support:"):
            fraction += 0.10
        elif flag == "runtime_mismatch:gradient_checkpointing_method":
            # Checkpointing-method extrapolation is the one runtime channel with measured leave-one-out
            # MISSES (q35 no_recompute rows: 74% and 155% error vs 0.54-0.58 predicted uncertainty);
            # every other runtime mismatch stays at the small generic bump.
            fraction += 0.30
        elif flag.startswith("runtime_mismatch:"):
            fraction += 0.04
        elif flag.startswith("communication_cross_node:"):
            fraction += 0.04
        elif flag.startswith("parallelism_extrapolation:"):
            fraction += 0.05
        elif flag.startswith("simulator_surface_unsupported:"):
            fraction += 0.50
        elif flag.startswith("simulator_surface_partial:"):
            fraction += 0.20
        elif flag == "memory_extrapolated_overhead":
            fraction += 0.08
        elif flag == "allocator_pressure_risk_extrapolated_peak":
            # Replaced the binary 0.5x throughput prior (falsified by the mbs3 prospective holdout):
            # an extrapolated peak crossing 85% reserved widens the interval instead of halving the score.
            fraction += 0.15
        elif flag.startswith("observed_oom_boundary:") or flag.startswith("allocator_pressure_boundary:"):
            fraction += 0.12

    if memory_coverage_status == "analytic_floor_only":
        fraction += 0.10
    elif memory_coverage_status.startswith("calibrated_overhead"):
        fraction += 0.06
    elif memory_coverage_status.startswith("extrapolated"):
        fraction += 0.08

    # Calibrated-class predictions are bounded by measured leave-one-out evidence, not the raw flag
    # stack: across the q35/q235/q36/q397 calibration holdouts every calibrated row's error is <= 11.6%
    # (most 1-6%), while flag stacking pushed several to 0.34-0.38 (20x+ over-conservative), turning
    # real promotable-tie decisions into artificial interval overlaps. Cap at 0.20 (1.7x margin over
    # the observed max); the CV floor above still lifts noisy references.
    if prediction_confidence == "calibrated":
        fraction = min(fraction, 0.20)
    # Step-time-fit extrapolations: after the definition-consistent fit (2026-07-04) the q35 65k
    # ga-family leave-one-out errors are 1.8-5.1% with the ga=1 regime-boundary row at 21.1%; the
    # flag stack had pushed these to 0.72-0.79. Cap at 0.35 (1.65x margin over the observed max).
    elif behavior.status == "extrapolated_step_time_fit":
        fraction = min(fraction, 0.35)

    return round(min(max(fraction, 0.0), 0.95), 3)


def _prediction_interval(score: float | None, uncertainty_fraction: float | None) -> tuple[float | None, float | None]:
    if score is None or uncertainty_fraction is None:
        return None, None
    return round(max(score * (1.0 - uncertainty_fraction), 0.0), 3), round(score * (1.0 + uncertainty_fraction), 3)


def _recommendation(
    *,
    feasible: bool,
    promotable: bool,
    feasibility_status: str,
    risk_flags: list[str],
) -> str:
    if feasibility_status == "observed_oom":
        return "avoid_observed_oom"
    if feasibility_status == "unsupported_simulator_surface":
        return "build_simulator_backend_before_ranking"
    if not feasible:
        return "do_not_launch_unscored"
    if any(flag.startswith("simulator_surface_unsupported:") for flag in risk_flags):
        return "build_simulator_backend_before_ranking"
    if any(flag.startswith("simulator_surface_partial:") for flag in risk_flags):
        return "measure_partial_simulator_surface"
    if promotable:
        return "promote_candidate"
    if "matched_allocator_pressure_slowdown" in risk_flags or any(
        flag.startswith("allocator_pressure_boundary:") for flag in risk_flags
    ):
        return "measure_allocator_boundary"
    if any(flag.startswith("observed_oom_boundary:") for flag in risk_flags):
        return "remeasure_after_memory_fix"
    if "real_routing_outside_fit_envelope" in risk_flags:
        return "measure_real_routing_boundary"
    if "correctness_runtime_failure_after_steps" in risk_flags:
        return "debug_runtime_failure"
    if "short_measurement_window" in risk_flags or "high_throughput_variance" in risk_flags:
        return "remeasure_for_stability"
    if "input_pipeline_bottleneck" in risk_flags:
        return "fix_input_pipeline_before_ranking"
    if "cross_model_analog" in risk_flags:
        return "measure_cross_model_analog"
    if "requires_remeasurement" in risk_flags:
        return "remeasure_before_ranking"
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
    data_parallel_replicate_size: int | None = None,
    data_parallel_shard_size: int | None = None,
) -> dict[str, Any]:
    raw_config = copy.deepcopy(base_config)
    surface = requested_simulator_surface(raw_config)
    if surface == "server_forward_backward":
        nested_server = _section(raw_config, "server")
        train = nested_server if nested_server else raw_config
    else:
        train = raw_config.setdefault("train", {})
        if not isinstance(train, dict):
            train = {}
            raw_config["train"] = train

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
    replicate_size = data_parallel_replicate_size
    shard_size = data_parallel_shard_size
    if replicate_size is None and shard_size is None:
        replicate_size = 1
        shard_size = data_parallel_size
    elif replicate_size is None:
        if shard_size is None or shard_size <= 0 or data_parallel_size % shard_size != 0:
            raise ValueError("data_parallel_shard_size must divide data_parallel_size")
        replicate_size = data_parallel_size // shard_size
    elif shard_size is None:
        if replicate_size <= 0 or data_parallel_size % replicate_size != 0:
            raise ValueError("data_parallel_replicate_size must divide data_parallel_size")
        shard_size = data_parallel_size // replicate_size
    elif replicate_size <= 0 or shard_size <= 0 or replicate_size * shard_size != data_parallel_size:
        raise ValueError("data_parallel_replicate_size * data_parallel_shard_size must equal data_parallel_size")
    train["data_parallel_replicate_size"] = replicate_size
    train["data_parallel_shard_size"] = shard_size
    if pipeline_parallel_size > 1:
        train["gradient_accumulation_steps"] = max(
            int(train.get("gradient_accumulation_steps", 1) or 1), pipeline_parallel_size
        )
    return raw_config


def _set_balanced_routing(raw_config: dict[str, Any], balanced_routing: bool) -> None:
    simulator = raw_config.setdefault("simulator", {})
    if isinstance(simulator, dict):
        simulator["balanced_routing"] = balanced_routing


def _topology_label(topology: Topology) -> str:
    return (
        f"mbs{topology.micro_batch_size}-gb{topology.global_batch_size}-"
        f"ep{topology.expert_parallel_size}-efsdp{topology.ep_fsdp_size}-"
        f"tp{topology.tensor_parallel_size}-pp{topology.pipeline_parallel_size}-"
        f"u{topology.ulysses_parallel_size}-r{topology.ringattn_parallel_size}"
    )


def _candidate_topology_label(topology: Topology, *, include_sequence_len: bool) -> str:
    label = _topology_label(topology)
    if include_sequence_len and topology.sample_packing_sequence_len is not None:
        return f"{label}-seq{topology.sample_packing_sequence_len}"
    return label


def _communication_ledger(topology: Topology) -> CommLedger:
    local_world_size = max(topology.local_world_size, 1)
    sequence_parallel_size = max(topology.sequence_parallel_size, 1)
    ranks_per_pipeline_stage = max(topology.world_size // max(topology.pipeline_parallel_size, 1), 1)
    ep_fsdp_size = topology.ep_fsdp_size or 1

    tensor_cross = topology.tensor_parallel_size > local_world_size
    pipeline_cross = topology.pipeline_parallel_size > local_world_size
    expert_cross = topology.expert_parallel_size > local_world_size
    context_cross = sequence_parallel_size > local_world_size
    fsdp_cross = topology.data_parallel_shard_size > local_world_size
    ep_fsdp_cross = ep_fsdp_size > local_world_size

    dimensions: list[str] = []
    notes = [
        f"node_count={topology.node_count}",
        f"local_world_size={topology.local_world_size}",
        f"ranks_per_pipeline_stage={ranks_per_pipeline_stage}",
    ]
    if tensor_cross:
        dimensions.append("tp")
        notes.append(f"tp_group_size={topology.tensor_parallel_size}:cross_node")
    if pipeline_cross:
        dimensions.append("pp")
        notes.append(f"pp_group_size={topology.pipeline_parallel_size}:cross_node")
    if expert_cross:
        dimensions.append("ep")
        notes.append(f"ep_group_size={topology.expert_parallel_size}:cross_node")
    if context_cross:
        dimensions.append("cp")
        notes.append(f"cp_group_size={sequence_parallel_size}:cross_node")
    if fsdp_cross:
        dimensions.append("fsdp")
        notes.append(f"dp_shard_size={topology.data_parallel_shard_size}:cross_node")
    if ep_fsdp_cross:
        dimensions.append("ep_fsdp")
        notes.append(f"ep_fsdp_size={ep_fsdp_size}:cross_node")

    return CommLedger(
        tensor_parallel_cross_node=tensor_cross,
        pipeline_parallel_cross_node=pipeline_cross,
        expert_parallel_cross_node=expert_cross,
        context_parallel_cross_node=context_cross,
        fsdp_cross_node=fsdp_cross or ep_fsdp_cross,
        cross_node_dimensions=dimensions,
        notes=notes,
    )


def _communication_risk_flags(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    behavior: BenchmarkBehaviorPrediction,
    communication: CommLedger,
    *,
    prediction_confidence: str,
) -> list[str]:
    if prediction_confidence == "calibrated":
        return []
    matched_labels = {part.strip() for part in (behavior.matched_label or "").split(",") if part.strip()}
    if not matched_labels:
        return [f"communication_cross_node:{dimension}" for dimension in communication.cross_node_dimensions]

    local_world_size = max(topology.local_world_size, 1)
    flags: set[str] = set()
    for point in behavior_points:
        if point.label not in matched_labels:
            continue
        reference_cp = _known_or_default_parallel_size(point.ulysses_parallel_size) * _known_or_default_parallel_size(
            point.ringattn_parallel_size
        )
        checks = (
            ("tp", topology.tensor_parallel_size, _known_or_default_parallel_size(point.tensor_parallel_size)),
            ("pp", topology.pipeline_parallel_size, _known_or_default_parallel_size(point.pipeline_parallel_size)),
            ("cp", topology.sequence_parallel_size, reference_cp),
            ("ep", topology.expert_parallel_size, point.expert_parallel_size or 1),
            ("ep_fsdp", topology.ep_fsdp_size or 1, point.ep_fsdp_size or 1),
        )
        for dimension, target_size, reference_size in checks:
            if target_size > local_world_size and reference_size <= local_world_size:
                flags.add(f"communication_cross_node:{dimension}")
    return sorted(flags)


def _reference_tokens_per_gpu(point: BenchmarkBehaviorPoint, topology: Topology) -> float | None:
    if point.tokens_per_sec is None:
        return None
    gpu_count = point.gpu_count or topology.world_size
    if gpu_count <= 0:
        return None
    return point.tokens_per_sec / gpu_count


def _runtime_mismatches_without_model(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any]) -> list[str]:
    return [mismatch for mismatch in behavior_point_workload_mismatches(point, raw_config) if mismatch != "model_ref"]


def _target_runtime_signature(raw_config: dict[str, Any] | None) -> str:
    if raw_config is None:
        return "unknown"
    parts: list[str] = []
    for section_name, field_name in _SCENARIO_RUNTIME_SIGNATURE_FIELDS:
        value = _section(raw_config, section_name).get(field_name)
        if value is not None:
            parts.append(f"{field_name}={value}")
    return ",".join(parts) if parts else "unknown"


def _reference_safety_score(point: BenchmarkBehaviorPoint) -> int:
    if point.correctness_status == "oom":
        return -1
    if point.correctness_status == "k3_pass":
        return 3
    if point.correctness_status in (None, "not_promoted", "raw_speed_not_promoted_without_matching_k3_pass"):
        return 2
    if point.correctness_status == "not_promoted_extrapolated":
        return 1
    return 0


def _is_stable_cross_model_reference(point: BenchmarkBehaviorPoint) -> bool:
    return point.correctness_status in {
        None,
        "k3_pass",
        "not_promoted",
        "raw_speed_not_promoted_without_matching_k3_pass",
    }


def _reference_throughput_quality_tier(point: BenchmarkBehaviorPoint, topology: Topology) -> int:
    per_gpu = _reference_tokens_per_gpu(point, topology)
    if per_gpu is None or per_gpu <= 0:
        return 0
    return int(math.log2(max(per_gpu, 1.0)))


def _sequence_len_ratio(point: BenchmarkBehaviorPoint, topology: Topology) -> float | None:
    if point.sample_packing_sequence_len is None or topology.sample_packing_sequence_len is None:
        return None
    if point.sample_packing_sequence_len <= 0 or topology.sample_packing_sequence_len <= 0:
        return None
    return topology.sample_packing_sequence_len / point.sample_packing_sequence_len


def _sequence_len_matches_for_cross_model(point: BenchmarkBehaviorPoint, topology: Topology) -> bool:
    ratio = _sequence_len_ratio(point, topology)
    if ratio is None:
        return False
    lower, upper = _EXTRAPOLATED_CROSS_MODEL_SEQUENCE_RATIO_WINDOW
    return lower <= ratio <= upper


def _active_param_proxy(metadata: ModelMetadata) -> float | None:
    hidden = metadata.hidden_size
    layers = metadata.num_hidden_layers
    if hidden is None or layers is None:
        return None

    head_dim = metadata.head_dim
    if head_dim is None and metadata.num_attention_heads:
        head_dim = hidden // metadata.num_attention_heads
    if head_dim is None:
        return None

    attention_heads = metadata.num_attention_heads or 1
    key_value_heads = metadata.num_key_value_heads or attention_heads
    q_proj = hidden * attention_heads * head_dim
    k_proj = hidden * key_value_heads * head_dim
    v_proj = hidden * key_value_heads * head_dim
    o_proj = attention_heads * head_dim * hidden
    attention_params = layers * (q_proj + k_proj + v_proj + o_proj)

    dense_mlp_params = 0
    has_routed_experts = metadata.num_experts is not None and metadata.moe_intermediate_size is not None
    if metadata.intermediate_size is not None and not has_routed_experts:
        dense_mlp_params = layers * 3 * hidden * metadata.intermediate_size

    shared_expert_params = 0
    if metadata.shared_expert_intermediate_size is not None:
        shared_expert_params = layers * 3 * hidden * metadata.shared_expert_intermediate_size

    active_expert_params = 0
    if has_routed_experts:
        if metadata.top_k is None or metadata.moe_intermediate_size is None:
            return None
        active_expert_params = layers * metadata.top_k * 3 * hidden * metadata.moe_intermediate_size

    lm_head_params = 0
    if metadata.vocab_size is not None and not metadata.tie_word_embeddings:
        lm_head_params = metadata.vocab_size * hidden

    return float(attention_params + dense_mlp_params + shared_expert_params + active_expert_params + lm_head_params)


def _metadata_for_model_ref(model_ref: str | None) -> ModelMetadata | None:
    if not model_ref:
        return None
    return resolve_model_metadata({"model": {"model_path": model_ref}}, hf_cache_roots=[])


@dataclass(frozen=True)
class _CrossModelScale:
    factor: float
    note: str
    active_param_ratio: float
    reference_active_params_b: float
    target_active_params_b: float


def _cross_model_scale(
    reference: BenchmarkBehaviorPoint,
    raw_config: dict[str, Any],
) -> _CrossModelScale | None:
    target_model_ref = model_ref_from_config(raw_config)
    target_metadata = resolve_model_metadata(raw_config, hf_cache_roots=[])
    reference_metadata = _metadata_for_model_ref(reference.model_ref)
    if target_model_ref is None or reference_metadata is None:
        return None

    target_proxy = _active_param_proxy(target_metadata)
    reference_proxy = _active_param_proxy(reference_metadata)
    if target_proxy is None or reference_proxy is None or target_proxy <= 0 or reference_proxy <= 0:
        return None

    raw_ratio = reference_proxy / target_proxy
    scale = min(1.20, max(0.18, raw_ratio**0.90))
    reference_active_params_b = reference_proxy / 1_000_000_000
    target_active_params_b = target_proxy / 1_000_000_000
    note = (
        f"cross-model active-param scale {reference.model_ref}->{target_model_ref}: "
        f"{reference_active_params_b:.3f}B/{target_active_params_b:.3f}B => {scale:.3f}"
    )
    return _CrossModelScale(
        factor=scale,
        note=note,
        active_param_ratio=raw_ratio,
        reference_active_params_b=reference_active_params_b,
        target_active_params_b=target_active_params_b,
    )


@dataclass(frozen=True)
class _MemoryPeakEstimate:
    peak_gb: float
    overhead_gb: float
    source_label: str
    notes: list[str]


def _point_int_or_default(point: BenchmarkBehaviorPoint, field_name: str, default: int = 1) -> int:
    value = getattr(point, field_name)
    return int(value) if value is not None else default


def _raw_config_for_behavior_point(
    base_config: dict[str, Any],
    point: BenchmarkBehaviorPoint,
    *,
    default_world_size: int,
) -> dict[str, Any] | None:
    if point.micro_batch_size is None or point.global_batch_size is None:
        return None
    world_size = point.gpu_count or default_world_size
    tp = _point_int_or_default(point, "tensor_parallel_size")
    pp = _point_int_or_default(point, "pipeline_parallel_size")
    ulysses = _point_int_or_default(point, "ulysses_parallel_size")
    ring = _point_int_or_default(point, "ringattn_parallel_size")
    ep = _point_int_or_default(point, "expert_parallel_size")
    non_dp = tp * pp * ulysses * ring
    if non_dp <= 0 or world_size % non_dp:
        return None
    dp = world_size // non_dp
    ga_denominator = point.micro_batch_size * dp
    if ga_denominator <= 0 or point.global_batch_size % ga_denominator:
        return None
    gradient_accumulation_steps = point.global_batch_size // ga_denominator
    if gradient_accumulation_steps <= 0:
        return None

    raw_config = _mutated_config(
        base_config,
        world_size=world_size,
        micro_batch_size=point.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        expert_parallel_size=ep,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        ulysses_parallel_size=ulysses,
        ringattn_parallel_size=ring,
        data_parallel_replicate_size=point.data_parallel_replicate_size,
        data_parallel_shard_size=point.data_parallel_shard_size,
    )
    if point.sample_packing_sequence_len is not None:
        _section(raw_config, "data")["sample_packing_sequence_len"] = point.sample_packing_sequence_len
    train = _section(raw_config, "train")
    if point.skip_param_upcast is not None:
        train["skip_param_upcast"] = point.skip_param_upcast
    if point.muon_momentum is not None:
        train["muon_momentum"] = point.muon_momentum
    if point.balanced_routing is not None:
        _set_balanced_routing(raw_config, point.balanced_routing)
    return raw_config


def _memory_overhead_scale(
    reference_topology: Topology,
    reference_shape: ShapeLedger,
    target_topology: Topology,
    target_shape: ShapeLedger,
) -> tuple[float, list[str]]:
    def ratio(target: float | int | None, reference: float | int | None) -> float:
        if target is None or reference in (None, 0):
            return 1.0
        return max(float(target) / float(reference), 0.01)

    token_ratio = ratio(
        target_shape.tokens_per_model_rank_per_microbatch, reference_shape.tokens_per_model_rank_per_microbatch
    )
    target_ep_slots = (
        max(target_shape.ep_rank_slots_per_microbatch) if target_shape.ep_rank_slots_per_microbatch else None
    )
    reference_ep_slots = (
        max(reference_shape.ep_rank_slots_per_microbatch) if reference_shape.ep_rank_slots_per_microbatch else None
    )
    routed_ratio = ratio(target_ep_slots, reference_ep_slots)
    sequence_ratio = ratio(target_topology.sample_packing_sequence_len, reference_topology.sample_packing_sequence_len)

    same_parallel_sequence_shape = (
        reference_topology.expert_parallel_size == target_topology.expert_parallel_size
        and reference_topology.ep_fsdp_size == target_topology.ep_fsdp_size
        and reference_topology.tensor_parallel_size == target_topology.tensor_parallel_size
        and reference_topology.pipeline_parallel_size == target_topology.pipeline_parallel_size
        and reference_topology.sequence_parallel_size == target_topology.sequence_parallel_size
        and reference_topology.sample_packing_sequence_len == target_topology.sample_packing_sequence_len
    )
    if same_parallel_sequence_shape:
        token_component = token_ratio
        routed_component = routed_ratio
        token_weight = 0.50
        routed_weight = 0.40
        sequence_weight = 0.10
        scaling_regime = "linear_same_topology"
    else:
        token_component = math.sqrt(token_ratio)
        routed_component = math.sqrt(routed_ratio)
        token_weight = 0.50
        routed_weight = 0.35
        sequence_weight = 0.15
        scaling_regime = "sqrt_cross_topology"
    sequence_component = sequence_ratio**0.20
    scale = token_weight * token_component + routed_weight * routed_component + sequence_weight * sequence_component

    reference_cp = max(reference_topology.sequence_parallel_size, 1)
    target_cp = max(target_topology.sequence_parallel_size, 1)
    if target_cp > reference_cp:
        scale *= max(0.65, (reference_cp / target_cp) ** 0.15)
    elif target_cp < reference_cp:
        scale *= min(1.35, (reference_cp / target_cp) ** 0.10)

    scale = min(4.0, max(0.20, scale))
    notes = [
        f"memory_overhead_scale={scale:.3f}",
        f"token_ratio={token_ratio:.3f}",
        f"routed_ratio={routed_ratio:.3f}",
        f"sequence_ratio={sequence_ratio:.3f}",
        f"memory_overhead_scaling_regime={scaling_regime}",
    ]
    if target_cp != reference_cp:
        notes.append(f"cp_ratio={target_cp}/{reference_cp}")
    return scale, notes


def _analytic_activation_fit_gb(
    metadata: ModelMetadata,
    topology: Topology,
    train_config: dict[str, Any],
) -> tuple[float | None, float, list[str]]:
    """Analytic activation lower bound (GB) + exact ep_fsdp expert-unshard transient for fit estimates.

    - The activation lower bound comes from the activation ledger. Under pipeline parallelism the
      1F1B in-flight depth (= pipeline_parallel_size) and the per-stage layer split (1/pp of the
      layers) CANCEL on the saved-activation term, so no PP multiplier is applied — the measured
      balanced PP2 row fits at ~64 GB, confirming the cancellation; the real-routing PP2 OOM was the
      mbs2 routing-imbalance effect, not a PP term.
    - With ``ep_fsdp_size > 1`` FSDP re-gathers each layer's full expert group for compute: the exact
      transient is (param + grad) x gathered expert bytes per layer x prefetch depth 2. This is a
      LOWER bound: the measured ep1/efsdp16 65k boundary sits ~13 GB above floor+activation+transient,
      an unattributed re-gather-path residual named in the notes rather than fitted away.
    """
    notes: list[str] = []
    try:
        ledger = activation_ledger(metadata, topology, train_config, seq_len=topology.sample_packing_sequence_len)
    except Exception:  # pragma: no cover - defensive: fall back to heuristic scaling
        return None, 0.0, ["activation_ledger_unavailable"]
    terms = ledger.get("terms") or {}
    total = 0.0
    for name, term in terms.items():
        gb = term.get("gb") if isinstance(term, dict) else None
        if not gb:
            continue
        total += float(gb)
    ep_fsdp = int(topology.ep_fsdp_size or 1)
    transient = 0.0
    if (
        ep_fsdp > 1
        and metadata.num_experts
        and metadata.moe_intermediate_size
        and metadata.hidden_size
        and metadata.num_hidden_layers
    ):
        local_experts = max(int(metadata.num_experts) // max(int(topology.expert_parallel_size), 1), 1)
        gathered_layer_gb = (
            local_experts * 3 * int(metadata.hidden_size) * int(metadata.moe_intermediate_size) * 2 / 1024**3
        )
        transient = round(gathered_layer_gb * 2 * 2, 3)  # (param + grad) x prefetch depth 2
        notes.append(f"ep_fsdp_unshard_transient_gb={transient}")
        notes.append("ep_fsdp_unshard_transient_is_lower_bound_named_residual_at_ep1_65k")
    return round(total, 3), transient, notes


def _calibrated_memory_peak_estimate(
    behavior_points: list[BenchmarkBehaviorPoint],
    base_config: dict[str, Any],
    raw_config: dict[str, Any],
    target_topology: Topology,
    target_shape: ShapeLedger,
    metadata: ModelMetadata,
    *,
    default_world_size: int,
    default_local_world_size: int,
    analytic_peak_floor_gb: float | None,
) -> _MemoryPeakEstimate | None:
    if analytic_peak_floor_gb is None:
        return None

    estimates: list[tuple[tuple[float, ...], _MemoryPeakEstimate]] = []
    for point in behavior_points:
        if point.peak_mem_gb is None or point.correctness_status == "oom":
            continue
        if behavior_point_model_mismatches(point, raw_config) or not behavior_point_matches_workload(point, raw_config):
            continue
        reference_config = _raw_config_for_behavior_point(
            base_config,
            point,
            default_world_size=default_world_size,
        )
        if reference_config is None:
            continue
        point_world_size = point.gpu_count or default_world_size
        point_local_world_size = min(default_local_world_size, point_world_size)
        try:
            reference_topology = resolve_topology(
                reference_config,
                world_size=point_world_size,
                local_world_size=point_local_world_size,
            )
        except ValueError:
            continue
        if point.ep_fsdp_size is not None and point.ep_fsdp_size != reference_topology.ep_fsdp_size:
            continue
        reference_memory = build_memory_ledger(
            reference_config,
            topology=reference_topology,
            model_metadata=metadata,
        )
        reference_floor = reference_memory.analytic_peak_floor_gb
        if reference_floor is None or point.peak_mem_gb < reference_floor:
            continue
        overhead_gb = point.peak_mem_gb - reference_floor
        reference_shape = build_shape_ledger(reference_topology, balanced_routing=True)
        # Analytic-first overhead transfer: scale the measured torch-side residual by the ratio of the
        # EXACT activation lower bounds (seq/topology/PP-aware), and add the exact ep_fsdp unshard
        # transient delta. The old sqrt heuristic under-scaled the measured 65k boundaries (u1's real
        # 4x activation growth became ~2x); it remains only as the fallback when the ledger cannot
        # compute either side.
        target_act, target_transient, target_act_notes = _analytic_activation_fit_gb(
            metadata, target_topology, raw_config.get("train", {})
        )
        reference_act, reference_transient, _ = _analytic_activation_fit_gb(
            metadata, reference_topology, reference_config.get("train", {})
        )

        # The analytic ratio captures topology/sequence-driven activation changes ONLY; when the
        # reference and target differ on runtime fields that move activations through other channels
        # (offload, checkpointing method, reduce dtype, ce mode), the ratio is wrong-by-construction
        # and the heuristic transfer stays in effect (validated on the q235 offload/dtype holdouts).
        def _activation_runtime_signature(config: dict[str, Any]) -> tuple[Any, ...]:
            train_section = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
            return (
                train_section.get("enable_activation_offload"),
                train_section.get("gradient_checkpointing_method"),
                train_section.get("fsdp_reduce_dtype"),
                train_section.get("ce_mode"),
                train_section.get("skip_param_upcast"),
            )

        same_activation_runtime = _activation_runtime_signature(raw_config) == _activation_runtime_signature(
            reference_config
        )
        if target_act and reference_act and same_activation_runtime:
            scale = target_act / reference_act
            scale_notes = [
                f"memory_overhead_scale={scale:.4f}",
                "scaling_regime=analytic_activation_ratio",
                f"target_activation_lower_bound_gb={target_act}",
                f"reference_activation_lower_bound_gb={reference_act}",
                *target_act_notes,
            ]
            transient_delta = max(0.0, target_transient - reference_transient)
        else:
            scale, scale_notes = _memory_overhead_scale(
                reference_topology, reference_shape, target_topology, target_shape
            )
            transient_delta = 0.0
        estimated_overhead = max(0.0, overhead_gb * scale) + transient_delta
        estimated_peak = analytic_peak_floor_gb + estimated_overhead
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
        key = (
            -sequence_distance,
            -parallel_distance,
            -batch_distance,
            float(point.peak_mem_gb),
        )
        notes = [
            f"memory_overhead_reference={point.label}",
            f"reference_peak_gb={point.peak_mem_gb:.3f}",
            f"reference_floor_gb={reference_floor:.3f}",
            f"reference_overhead_gb={overhead_gb:.3f}",
            f"estimated_overhead_gb={estimated_overhead:.3f}",
            *scale_notes,
        ]
        estimates.append(
            (
                key,
                _MemoryPeakEstimate(
                    peak_gb=round(estimated_peak, 3),
                    overhead_gb=round(estimated_overhead, 3),
                    source_label=point.label,
                    notes=notes,
                ),
            )
        )

    if not estimates:
        return None
    return max(estimates, key=lambda item: item[0])[1]


def _memory_ownership_notes(memory: MemoryLedger) -> list[str]:
    ownership_prefixes = (
        "pp_stage=",
        "tp_non_expert_shard_size=",
        "non_expert_total_shard_size=",
        "expert_shard_size=",
        "dp_shard_size=",
        "cp_fsdp_mode=",
        "local_non_expert_params=",
        "local_expert_params=",
    )
    for bucket in memory.top_memory_buckets:
        if bucket.name == "sharded_trainable_params":
            return [note for note in bucket.notes if note.startswith(ownership_prefixes)]
    return []


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
        and (raw_config is None or not behavior_point_model_mismatches(point, raw_config))
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


def _select_cross_model_reference_point(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    raw_config: dict[str, Any] | None,
) -> BenchmarkBehaviorPoint | None:
    if raw_config is None or model_ref_from_config(raw_config) is None:
        return None

    usable = [
        point
        for point in behavior_points
        if point.tokens_per_sec is not None
        and point.micro_batch_size is not None
        and _is_stable_cross_model_reference(point)
        and behavior_point_model_mismatches(point, raw_config)
        and _sequence_len_matches_for_cross_model(point, topology)
        and _reference_tokens_per_gpu(point, topology) is not None
        and _cross_model_scale(point, raw_config) is not None
        and "attention_backend" not in _runtime_mismatches_without_model(point, raw_config)
    ]
    if not usable:
        return None

    def key(point: BenchmarkBehaviorPoint) -> tuple[int, int, float, float, float, float, float]:
        runtime_mismatch_count = len(_runtime_mismatches_without_model(point, raw_config))
        mbs_distance = abs((point.micro_batch_size or 1) - topology.micro_batch_size)
        ep_distance = 0.0
        if point.expert_parallel_size:
            ep_distance = abs(math.log2(topology.expert_parallel_size / point.expert_parallel_size))
        seq_ratio = _sequence_len_ratio(point, topology) or 1.0
        seq_distance = abs(math.log(seq_ratio))
        per_gpu = _reference_tokens_per_gpu(point, topology) or 0.0
        return (
            _reference_safety_score(point),
            _reference_throughput_quality_tier(point, topology),
            -runtime_mismatch_count,
            -seq_distance,
            -ep_distance,
            -mbs_distance,
            per_gpu,
        )

    return max(usable, key=key)


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


def _cross_model_behavior_prediction(
    reference: BenchmarkBehaviorPoint,
    topology: Topology,
    shape: ShapeLedger,
    *,
    raw_config: dict[str, Any],
    device_memory_limit_gb: float,
    memory_safety_factor: float,
    analytic_peak_floor_gb: float | None,
) -> tuple[BenchmarkBehaviorPrediction, list[str]] | None:
    model_scale = _cross_model_scale(reference, raw_config)
    if model_scale is None:
        return None
    model_factor = model_scale.factor
    model_note = model_scale.note
    ref_per_gpu = _reference_tokens_per_gpu(reference, topology) or 0.0
    if ref_per_gpu <= 0:
        return None

    parallel_factor, notes = _parallelism_factor(reference, topology)
    memory_factor, _, memory_status = _memory_factor(
        analytic_peak_floor_gb,
        memory_basis="analytic_floor",
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    seq_ratio = _sequence_len_ratio(reference, topology) or 1.0
    sequence_factor = min(1.08, max(0.88, seq_ratio**-0.12))
    tokens_per_sec_per_gpu = ref_per_gpu * model_factor * parallel_factor * memory_factor * sequence_factor
    tokens_per_sec = tokens_per_sec_per_gpu * topology.world_size
    step_time_sec = None
    if shape.global_tokens_per_train_step and tokens_per_sec:
        step_time_sec = shape.global_tokens_per_train_step / tokens_per_sec

    tflops_per_gpu = reference.tflops_per_gpu
    if tflops_per_gpu is None and reference.mfu_percent is not None:
        tflops_per_gpu = H100_BF16_PROMISED_TFLOPS_PER_GPU * reference.mfu_percent / 100.0
    if tflops_per_gpu is not None and ref_per_gpu:
        tflops_per_gpu *= tokens_per_sec_per_gpu / ref_per_gpu

    target_model_ref = model_ref_from_config(raw_config)
    runtime_mismatches = _runtime_mismatches_without_model(reference, raw_config)
    warnings = [
        f"cross-model analog from {reference.model_ref} to {target_model_ref}; measure before ranking",
        model_note,
        f"sequence_length_factor={sequence_factor:.3f}",
        f"memory feasibility status is {memory_status}",
        "correctness must be re-gated before promotion",
    ]
    exact_sequence_lower, exact_sequence_upper = _EXACT_CROSS_MODEL_SEQUENCE_RATIO_WINDOW
    if not exact_sequence_lower <= seq_ratio <= exact_sequence_upper:
        warnings.append(f"cross-model sequence ratio outside exact-context window: {seq_ratio:.3f}")
    if reference.correctness_status and reference.correctness_status not in {"k3_pass", "not_promoted"}:
        warnings.append(f"reference correctness status is {reference.correctness_status}")
    if runtime_mismatches:
        warnings.append(f"reference runtime knobs differ: {', '.join(runtime_mismatches)}")
    warnings.extend(notes)

    return (
        BenchmarkBehaviorPrediction(
            status="cross_model_extrapolated",
            matched_label=reference.label,
            source=reference.source,
            tokens_per_sec=round(tokens_per_sec, 3),
            tokens_per_sec_per_gpu=round(tokens_per_sec_per_gpu, 3),
            step_time_sec=round(step_time_sec, 6) if step_time_sec is not None else None,
            mfu_percent=None,
            tflops_per_gpu=round(tflops_per_gpu, 3) if tflops_per_gpu is not None else None,
            promised_tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU,
            peak_mem_gb=None,
            allocator_retries=None,
            derived_global_tokens_per_step=shape.global_tokens_per_train_step,
            phase_time_sec=reference.phase_time_sec,
            phase_time_share=reference.phase_time_share,
            phase_memory_peak_gb=reference.phase_memory_peak_gb,
            model_ref=target_model_ref,
            balanced_routing=reference.balanced_routing,
            correctness_status="not_promoted_extrapolated",
            cross_model_active_param_ratio=round(model_scale.active_param_ratio, 3),
            cross_model_active_param_scale=round(model_scale.factor, 3),
            cross_model_reference_active_params_b=round(model_scale.reference_active_params_b, 3),
            cross_model_target_active_params_b=round(model_scale.target_active_params_b, 3),
            cross_model_sequence_length_factor=round(sequence_factor, 3),
            cross_model_parallelism_factor=round(parallel_factor, 3),
            cross_model_memory_factor=round(memory_factor, 3),
            warnings=warnings,
        ),
        ["cross_model_analog", model_note],
    )


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
        # The fit family must share the full sharding regime: rows at the same nominal global batch
        # but a different world size or dp-replicate split have different per-microbatch step
        # structure (measured: the repl=1 gate-lane gbs64 row steps 88.3s vs the repl=2 ga16 row's
        # 101.6s), and best-by-gbs cherry-picking across regimes bent the line by ~20%.
        and point.gpu_count in (None, topology.world_size)
        and (
            point.data_parallel_replicate_size is None
            or point.data_parallel_replicate_size == topology.data_parallel_replicate_size
        )
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
        # Definition-consistent y: the fit converts the predicted step BACK to tokens_per_sec via
        # nominal tokens, so y must be nominal_tokens / tokens_per_sec. The measured step_time_sec
        # field disagrees with that identity by 0.81x-1.07x across the q35 65k ga family (valid-token
        # and window differences), which bent the line by ~20% (LOO APE 20-23% -> 2-5% with the
        # consistent definition).
        step_time = tokens / point.tokens_per_sec if point.tokens_per_sec else point.step_time_sec
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
    model_refs = {point.model_ref for point in fit_points if point.model_ref is not None}
    peak_mem_gb = max((point.peak_mem_gb for point in fit_points if point.peak_mem_gb is not None), default=None)
    phase_memory_peak_gb: dict[str, float] = {}
    for point in fit_points:
        for phase, peak in point.phase_memory_peak_gb.items():
            phase_memory_peak_gb[phase] = max(phase_memory_peak_gb.get(phase, 0.0), peak)
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
        phase_memory_peak_gb=phase_memory_peak_gb,
        model_ref=next(iter(model_refs)) if len(model_refs) == 1 else None,
        balanced_routing=(
            fit_points[0].balanced_routing if len({point.balanced_routing for point in fit_points}) == 1 else None
        ),
        correctness_status="not_promoted_extrapolated",
        warnings=[
            f"extrapolated step time from calibrated global batches: {labels}",
            f"fit_intercept_sec={intercept:.6f}",
            f"fit_sec_per_token={slope:.12f}",
            "correctness must be re-gated before promotion",
        ],
    )


# Measured non-PyTorch device overhead at long context (NCCL buffers + CUDA context + triton cache):
# the 65k 2-node gate-lane OOMs died with 9.0-12.6 GB of non-torch memory resident (e.g. o-ep4mbs2:
# 65.9 GB torch-allocated but 78.3 GB device-used). Device fit must account for it; this is a
# calibrated_residual term (max measured at 65k/2-node) until decomposed.
LONG_CONTEXT_NON_TORCH_DEVICE_OVERHEAD_GB = 12.6
LONG_CONTEXT_NON_TORCH_SEQ_THRESHOLD = 32768


def _device_side_peak_gb(estimated_peak_mem_gb: float | None, topology: Topology | None) -> float | None:
    """Torch-side peak estimate -> device-side estimate for the fit check."""
    if estimated_peak_mem_gb is None:
        return None
    seq_len = int(getattr(topology, "sample_packing_sequence_len", 0) or 0) if topology is not None else 0
    if seq_len >= LONG_CONTEXT_NON_TORCH_SEQ_THRESHOLD:
        return estimated_peak_mem_gb + LONG_CONTEXT_NON_TORCH_DEVICE_OVERHEAD_GB
    return estimated_peak_mem_gb


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
        if memory_basis != "analytic_floor" and memory_estimate_gb <= device_memory_limit_gb:
            return 0.75, headroom, f"feasible_{status_basis}_high_pressure"
        if memory_basis == "analytic_floor":
            if memory_estimate_gb <= device_memory_limit_gb:
                return 0.0, headroom, "memory_floor_exceeds_safety_margin"
            return 0.0, headroom, "memory_floor_exceeds_limit"
        return 0.0, headroom, f"{status_basis}_exceeds_limit"
    utilization = reserved_memory / device_memory_limit_gb if device_memory_limit_gb else 1.0
    if utilization >= 0.90:
        return 0.75, headroom, f"feasible_{status_basis}_high_pressure"
    if utilization >= 0.80:
        return 0.90, headroom, f"feasible_{status_basis}_moderate_pressure"
    return 1.0, headroom, f"feasible_{status_basis}"


def _throughput_memory_factor(
    analytic_peak_floor_gb: float | None,
    *,
    memory_peak_estimate: _MemoryPeakEstimate | None,
    reference: BenchmarkBehaviorPoint,
    topology: Topology,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
) -> tuple[float, str, list[str]]:
    notes: list[str] = []
    factor, _, status = _memory_factor(
        analytic_peak_floor_gb,
        memory_basis="analytic_floor",
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    if memory_peak_estimate is not None and device_memory_limit_gb > 0:
        reserved_utilization = memory_peak_estimate.peak_gb * memory_safety_factor / device_memory_limit_gb
        notes.append(f"throughput_memory_peak_source={memory_peak_estimate.source_label}")
        notes.append(f"throughput_memory_reserved_utilization={reserved_utilization:.3f}")
        expands_microbatch = (
            reference.micro_batch_size is not None and topology.micro_batch_size > reference.micro_batch_size
        )
        # Graded allocator-pressure prior, calibrated by one measured point on each side: the q35
        # mbs3 prospective blind holdout (estimated utilization 0.916) measured NO pressure — the old
        # binary 0.5x at >=0.85 turned a 2.5%-accurate row into a 2x miss — while the q36 mbs10 row
        # (estimated 0.948) measured a real ~2x slowdown. Onset ramps 1.0 -> 0.5 across 0.92 -> 0.94
        # estimated reserved utilization; the risk flag attaches from 0.85 up so intervals widen
        # before the score moves. A reference whose OWN measured peak crosses 0.85 keeps the hard 0.5x.
        reference_reserved = (
            reference.peak_mem_gb * memory_safety_factor / device_memory_limit_gb if reference.peak_mem_gb else None
        )
        if reference.status != "allocator_pressure_slowdown" and expands_microbatch and reserved_utilization >= 0.85:
            if reference_reserved is not None and reference_reserved >= 0.85:
                factor *= 0.50
                notes.append("allocator_pressure_prior=larger_microbatch_measured_reference_peak_ge_85pct_reserved")
            else:
                ramp = min(max((reserved_utilization - 0.92) / 0.02, 0.0), 1.0)
                graded = 1.0 - 0.5 * ramp
                if graded < 1.0:
                    factor *= graded
                    notes.append(
                        f"allocator_pressure_prior=graded_extrapolated_peak_factor_{graded:.3f}"
                        f"_at_{reserved_utilization:.3f}_reserved"
                    )
                notes.append("allocator_pressure_risk=extrapolated_peak_ge_85pct_reserved_unmeasured")
    return factor, status, notes


def _memory_coverage_for_candidate(
    *,
    analytic_peak_floor_gb: float | None,
    estimated_peak_mem_gb: float | None,
    memory_basis: str,
) -> tuple[str, float | None, float | None]:
    if analytic_peak_floor_gb is None:
        return "unresolved_analytic_floor", None, None
    if memory_basis == "analytic_floor":
        return "analytic_floor_only", None, None
    if estimated_peak_mem_gb is None:
        return "unresolved_estimated_peak", None, None
    if estimated_peak_mem_gb <= 0:
        return "invalid_estimated_peak", None, None

    residual = estimated_peak_mem_gb - analytic_peak_floor_gb
    if residual < 0:
        return f"{memory_basis}_below_analytic_floor", 0.0, 0.0
    if residual == 0:
        if memory_basis == "calibrated_peak":
            return "analytic_floor_matches_calibrated_peak", 0.0, 0.0
        return f"{memory_basis}_matches_analytic_floor", 0.0, 0.0

    residual_fraction = residual / estimated_peak_mem_gb
    if memory_basis == "calibrated_peak":
        status = "calibrated_peak_with_unmodeled_residual"
    elif memory_basis == "calibrated_overhead_peak":
        status = "calibrated_overhead_peak_with_scaled_residual"
    elif memory_basis == "extrapolated_peak":
        status = "extrapolated_peak_with_unmodeled_residual"
    else:
        status = f"{memory_basis}_with_unmodeled_residual"
    return status, round(residual, 3), round(residual_fraction, 3)


def _extrapolate_behavior(
    behavior_points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    shape: ShapeLedger,
    *,
    raw_config: dict[str, Any] | None = None,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
    analytic_peak_floor_gb: float | None,
    memory_peak_estimate: _MemoryPeakEstimate | None = None,
) -> tuple[BenchmarkBehaviorPrediction, list[str]]:
    step_fit = _step_time_fit_prediction(behavior_points, topology, shape, raw_config=raw_config)
    if step_fit is not None:
        return step_fit, ["step_time_fit_extrapolation"]

    reference = _select_reference_point(behavior_points, topology, raw_config=raw_config)
    if reference is None:
        cross_model_reference = _select_cross_model_reference_point(behavior_points, topology, raw_config)
        if cross_model_reference is not None and raw_config is not None:
            cross_model_prediction = _cross_model_behavior_prediction(
                cross_model_reference,
                topology,
                shape,
                raw_config=raw_config,
                device_memory_limit_gb=device_memory_limit_gb,
                memory_safety_factor=memory_safety_factor,
                analytic_peak_floor_gb=analytic_peak_floor_gb,
            )
            if cross_model_prediction is not None:
                return cross_model_prediction
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
                balanced_routing=_config_balanced_routing(raw_config) if raw_config is not None else None,
                correctness_status="missing_calibration",
                warnings=["no benchmark behavior point is available for extrapolation"],
            ),
            [],
        )

    ref_per_gpu = _reference_tokens_per_gpu(reference, topology) or 0.0
    parallel_factor, notes = _parallelism_factor(reference, topology)
    memory_factor, memory_status, memory_notes = _throughput_memory_factor(
        analytic_peak_floor_gb,
        memory_peak_estimate=memory_peak_estimate,
        reference=reference,
        topology=topology,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    notes.extend(memory_notes)
    tokens_per_sec_per_gpu = ref_per_gpu * parallel_factor * memory_factor
    tokens_per_sec = tokens_per_sec_per_gpu * topology.world_size
    step_time_sec = None
    if shape.global_tokens_per_train_step and tokens_per_sec:
        step_time_sec = shape.global_tokens_per_train_step / tokens_per_sec
    # ga fixed-tail amortization: a pure-ga extrapolation reuses the reference tok/s verbatim, which
    # drops the per-step fixed cost (clip + optimizer + metrics) amortizing across microbatch chains.
    # step(ga) = ga_ratio x (ref_step - fixed_tail) + fixed_tail from the reference's own phase reads
    # predicted the ga2@mbs2 prospective blind holdout to 0.02% (4.3454 vs measured 4.3455).
    ga_amortization_applicable = (
        step_time_sec is not None
        and parallel_factor == 1.0
        and reference.micro_batch_size == topology.micro_batch_size
        and reference.gradient_accumulation_steps
        and topology.gradient_accumulation_steps != reference.gradient_accumulation_steps
        and reference.step_time_sec
        and reference.phase_time_sec
        and "clip_and_step_total" in reference.phase_time_sec
    )
    if ga_amortization_applicable:
        target_ga = topology.gradient_accumulation_steps
        ga_ratio = target_ga / reference.gradient_accumulation_steps
        fixed_tail = float(reference.phase_time_sec["clip_and_step_total"]) + float(
            reference.phase_time_sec.get("reduce_metrics", 0.0)
        )
        chain_time = max(float(reference.step_time_sec) - fixed_tail, 0.0)
        amortized_step = ga_ratio * chain_time + fixed_tail
        if amortized_step > 0:
            step_time_sec = amortized_step * (step_time_sec / (float(reference.step_time_sec) * ga_ratio))
            tokens_per_sec = shape.global_tokens_per_train_step / step_time_sec
            tokens_per_sec_per_gpu = tokens_per_sec / topology.world_size
            notes.append(
                f"ga_fixed_tail_amortization: ga {reference.gradient_accumulation_steps}->{target_ga}, "
                f"fixed_tail={fixed_tail:.4f}s from reference phase reads"
            )
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
            phase_time_sec=reference.phase_time_sec,
            phase_time_share=reference.phase_time_share,
            phase_memory_peak_gb=reference.phase_memory_peak_gb,
            model_ref=reference.model_ref,
            balanced_routing=reference.balanced_routing,
            correctness_status="not_promoted_extrapolated",
            warnings=warnings,
        ),
        notes,
    )


def _format_factor_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}{suffix}"


def _candidate_decision_factors(
    *,
    behavior: BenchmarkBehaviorPrediction,
    prediction_confidence: str,
    calibration_scope: str,
    calibration_distance: float | None,
    calibration_distance_factors: list[str],
    feasibility_status: str,
    score_tokens_per_sec: float | None,
    score_tokens_per_gpu_per_sec: float | None,
    score_risk_adjusted_tokens_per_sec: float | None,
    score_risk_adjusted_tokens_per_gpu_per_sec: float | None,
    prediction_uncertainty_fraction: float | None,
    prediction_interval_lower_tokens_per_sec: float | None,
    prediction_interval_upper_tokens_per_sec: float | None,
    risk_adjusted_prediction_interval_lower_tokens_per_sec: float | None,
    risk_adjusted_prediction_interval_upper_tokens_per_sec: float | None,
    analytic_peak_floor_gb: float | None,
    estimated_peak_mem_gb: float | None,
    memory_basis: str,
    memory_coverage_status: str,
    memory_residual_gb: float | None,
    memory_residual_fraction: float | None,
    headroom_gb: float | None,
    promotable: bool,
    recommendation: str,
    simulator_surface: str,
    simulator_support_status: str,
    simulator_support_blockers: list[str],
    risk_flags: list[str],
    communication: CommLedger | None,
) -> list[str]:
    factors = [
        f"matched={behavior.matched_label or behavior.status}",
        f"calibration={calibration_scope}/{prediction_confidence}",
        f"feasibility={feasibility_status}",
        f"score_tokens_per_sec={_format_factor_float(score_tokens_per_sec)}",
        f"score_tokens_per_gpu_per_sec={_format_factor_float(score_tokens_per_gpu_per_sec)}",
    ]
    if calibration_distance is not None:
        factors.append(f"calibration_distance={_format_factor_float(calibration_distance)}")
    if calibration_distance_factors:
        factors.append(f"calibration_distance_factors={';'.join(calibration_distance_factors)}")
    if score_risk_adjusted_tokens_per_sec is not None:
        factors.append(f"risk_adjusted_tokens_per_sec={_format_factor_float(score_risk_adjusted_tokens_per_sec)}")
    if score_risk_adjusted_tokens_per_gpu_per_sec is not None:
        factors.append(
            f"risk_adjusted_tokens_per_gpu_per_sec={_format_factor_float(score_risk_adjusted_tokens_per_gpu_per_sec)}"
        )
    if prediction_uncertainty_fraction is not None:
        factors.append(f"prediction_uncertainty_fraction={_format_factor_float(prediction_uncertainty_fraction)}")
    if prediction_interval_lower_tokens_per_sec is not None and prediction_interval_upper_tokens_per_sec is not None:
        factors.append(
            "prediction_interval_tokens_per_sec="
            f"{_format_factor_float(prediction_interval_lower_tokens_per_sec)}.."
            f"{_format_factor_float(prediction_interval_upper_tokens_per_sec)}"
        )
    if (
        risk_adjusted_prediction_interval_lower_tokens_per_sec is not None
        and risk_adjusted_prediction_interval_upper_tokens_per_sec is not None
    ):
        factors.append(
            "risk_adjusted_prediction_interval_tokens_per_sec="
            f"{_format_factor_float(risk_adjusted_prediction_interval_lower_tokens_per_sec)}.."
            f"{_format_factor_float(risk_adjusted_prediction_interval_upper_tokens_per_sec)}"
        )
    factors.append(
        "memory="
        f"{memory_basis}:floor={_format_factor_float(analytic_peak_floor_gb, 'GB')},"
        f"peak={_format_factor_float(estimated_peak_mem_gb, 'GB')},"
        f"headroom={_format_factor_float(headroom_gb, 'GB')}"
    )
    factors.append(
        "memory_coverage="
        f"{memory_coverage_status}:"
        f"residual={_format_factor_float(memory_residual_gb, 'GB')},"
        f"residual_fraction={_format_factor_float(memory_residual_fraction)}"
    )
    if behavior.correctness_status:
        factors.append(f"correctness={behavior.correctness_status}")
    factors.append(f"promotable={str(promotable).lower()}")
    if communication is not None and communication.cross_node_dimensions:
        factors.append(f"cross_node={','.join(communication.cross_node_dimensions)}")
    factors.append(f"simulator_support={simulator_surface}/{simulator_support_status}")
    if simulator_support_blockers:
        factors.append(f"simulator_support_blockers={','.join(simulator_support_blockers)}")
    if risk_flags:
        factors.append(f"risks={','.join(risk_flags)}")
    factors.append(f"recommendation={recommendation}")
    return factors


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
    memory_peak_estimate: _MemoryPeakEstimate | None,
    memory_ownership_notes: list[str],
    communication: CommLedger,
    notes: list[str],
) -> ScenarioCandidate:
    estimated_peak_mem_gb = analytic_peak_floor_gb
    memory_basis = "analytic_floor"
    peak_below_floor_note = None
    memory_calibration_source = None
    memory_calibration_notes: list[str] = []
    if behavior.peak_mem_gb is not None:
        if prediction_confidence == "calibrated" and (
            analytic_peak_floor_gb is None or behavior.peak_mem_gb >= analytic_peak_floor_gb
        ):
            estimated_peak_mem_gb = behavior.peak_mem_gb
            memory_basis = "calibrated_peak"
        elif prediction_confidence != "calibrated" and memory_peak_estimate is not None:
            estimated_peak_mem_gb = memory_peak_estimate.peak_gb
            memory_basis = "calibrated_overhead_peak"
            memory_calibration_source = memory_peak_estimate.source_label
            memory_calibration_notes = memory_peak_estimate.notes
        elif analytic_peak_floor_gb is None or behavior.peak_mem_gb >= analytic_peak_floor_gb:
            estimated_peak_mem_gb = behavior.peak_mem_gb
            memory_basis = "extrapolated_peak"
        else:
            peak_below_floor_note = (
                f"calibrated_peak_below_analytic_floor: peak={behavior.peak_mem_gb:.3f} "
                f"floor={analytic_peak_floor_gb:.3f}; using analytic_floor"
            )
    elif memory_peak_estimate is not None:
        estimated_peak_mem_gb = memory_peak_estimate.peak_gb
        memory_basis = "calibrated_overhead_peak"
        memory_calibration_source = memory_peak_estimate.source_label
        memory_calibration_notes = memory_peak_estimate.notes

    # Fit is a DEVICE-memory question: an ESTIMATED torch-side peak plus the measured long-context
    # non-torch overhead (NCCL/context/triton) is what actually has to fit under the device limit.
    # A MEASURED peak (the candidate's own completed run) is direct device-fit evidence and must not
    # be re-blocked by an overhead estimate.
    fit_check_peak_gb = (
        _device_side_peak_gb(estimated_peak_mem_gb, topology)
        if memory_basis in ("calibrated_overhead_peak", "extrapolated_peak")
        else estimated_peak_mem_gb
    )
    _, headroom, feasibility_status = _memory_factor(
        fit_check_peak_gb,
        memory_basis=memory_basis,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
    )
    (
        memory_coverage_status,
        estimated_memory_residual_gb,
        estimated_memory_residual_fraction,
    ) = _memory_coverage_for_candidate(
        analytic_peak_floor_gb=analytic_peak_floor_gb,
        estimated_peak_mem_gb=estimated_peak_mem_gb,
        memory_basis=memory_basis,
    )
    support = resolve_simulator_support(raw_config or {}, topology=topology)
    if behavior.status == "calibrated_failure" or behavior.correctness_status == "oom":
        feasibility_status = "observed_oom"
    if support.support_status.startswith("unsupported_"):
        feasibility_status = "unsupported_simulator_surface"
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
        raw_config=raw_config,
    )
    risk_flags = _candidate_risk_flags(
        behavior_points,
        topology,
        behavior,
        raw_config=raw_config,
        calibration_scope=calibration_scope,
        prediction_confidence=prediction_confidence,
        communication=communication,
    )
    if peak_below_floor_note is not None:
        risk_flags = sorted({*risk_flags, "calibrated_peak_below_analytic_floor"})
    if memory_basis == "calibrated_overhead_peak":
        risk_flags = sorted({*risk_flags, "memory_extrapolated_overhead"})
    if any("allocator_pressure_risk=extrapolated_peak_ge_85pct_reserved_unmeasured" in w for w in behavior.warnings):
        risk_flags = sorted({*risk_flags, "allocator_pressure_risk_extrapolated_peak"})
    if support.support_status.startswith("unsupported_"):
        risk_flags = sorted({*risk_flags, f"simulator_surface_unsupported:{support.support_status}"})
    elif support.support_status != "supported_local_non_pp":
        risk_flags = sorted({*risk_flags, f"simulator_surface_partial:{support.support_status}"})
    calibration_distance, calibration_distance_factors = _calibration_distance(behavior_points, topology, behavior)
    score_risk_adjusted_tokens_per_sec = _risk_adjusted_score(
        score_tokens_per_sec,
        calibration_scope=calibration_scope,
        calibration_distance=calibration_distance,
        risk_flags=risk_flags,
        feasibility_status=feasibility_status,
    )
    score_risk_adjusted_tokens_per_gpu_per_sec = (
        round(score_risk_adjusted_tokens_per_sec / topology.world_size, 3)
        if score_risk_adjusted_tokens_per_sec is not None and topology.world_size
        else None
    )
    prediction_uncertainty_fraction = _prediction_uncertainty_fraction(
        behavior,
        prediction_confidence=prediction_confidence,
        calibration_scope=calibration_scope,
        calibration_distance=calibration_distance,
        risk_flags=risk_flags,
        memory_coverage_status=memory_coverage_status,
    )
    prediction_interval_lower, prediction_interval_upper = _prediction_interval(
        score_tokens_per_sec,
        prediction_uncertainty_fraction,
    )
    risk_adjusted_interval_lower, risk_adjusted_interval_upper = _prediction_interval(
        score_risk_adjusted_tokens_per_sec,
        prediction_uncertainty_fraction,
    )
    recommendation = _recommendation(
        feasible=feasible,
        promotable=promotable and feasible,
        feasibility_status=feasibility_status,
        risk_flags=risk_flags,
    )
    candidate_notes = list(notes)
    if peak_below_floor_note is not None:
        candidate_notes.append(peak_below_floor_note)
    candidate_notes.extend(memory_calibration_notes)
    timing = build_timing_ledger(None, behavior)
    phase_details = _phase_bottleneck_details(timing.phase_time_share, timing.phase_time_sec)
    phase_bottleneck_share = phase_details[2] if phase_details is not None else None
    (
        phase_bottleneck_half_speedup_tokens_per_sec,
        phase_bottleneck_half_speedup_delta_pct,
    ) = _phase_bottleneck_half_speedup_counterfactual(
        score_tokens_per_sec,
        phase_bottleneck_share,
    )
    (
        phase_bottleneck_half_speedup_risk_adjusted_tokens_per_sec,
        phase_bottleneck_half_speedup_risk_adjusted_delta_pct,
    ) = _phase_bottleneck_half_speedup_counterfactual(
        score_risk_adjusted_tokens_per_sec,
        phase_bottleneck_share,
    )
    memory_details = _memory_bottleneck_details(behavior.phase_memory_peak_gb, estimated_peak_mem_gb)
    if phase_note := _phase_bottleneck_note(timing.phase_time_share):
        candidate_notes.append(phase_note)
    if phase_bottleneck_half_speedup_delta_pct is not None and phase_details is not None:
        candidate_notes.append(
            "fixed_schedule_phase_bottleneck_half_speedup="
            f"{phase_details[0]}:+{phase_bottleneck_half_speedup_delta_pct:.3f}%"
        )
    if memory_details is not None:
        candidate_notes.append(
            f"memory_bottleneck={memory_details[0]}:{memory_details[2]:.3f}GB({memory_details[3]:.3f}x_peak)"
        )
    timing_coverage_status = _scenario_timing_coverage_status(
        behavior,
        prediction_confidence=prediction_confidence,
        calibration_scope=calibration_scope,
        timing_coverage_status=timing.timing_coverage_status,
    )
    decision_factors = _candidate_decision_factors(
        behavior=behavior,
        prediction_confidence=prediction_confidence,
        calibration_scope=calibration_scope,
        calibration_distance=calibration_distance,
        calibration_distance_factors=calibration_distance_factors,
        feasibility_status=feasibility_status,
        score_tokens_per_sec=score_tokens_per_sec,
        score_tokens_per_gpu_per_sec=score_tokens_per_gpu_per_sec,
        score_risk_adjusted_tokens_per_sec=score_risk_adjusted_tokens_per_sec,
        score_risk_adjusted_tokens_per_gpu_per_sec=score_risk_adjusted_tokens_per_gpu_per_sec,
        prediction_uncertainty_fraction=prediction_uncertainty_fraction,
        prediction_interval_lower_tokens_per_sec=prediction_interval_lower,
        prediction_interval_upper_tokens_per_sec=prediction_interval_upper,
        risk_adjusted_prediction_interval_lower_tokens_per_sec=risk_adjusted_interval_lower,
        risk_adjusted_prediction_interval_upper_tokens_per_sec=risk_adjusted_interval_upper,
        analytic_peak_floor_gb=analytic_peak_floor_gb,
        estimated_peak_mem_gb=estimated_peak_mem_gb,
        memory_basis=memory_basis,
        memory_coverage_status=memory_coverage_status,
        memory_residual_gb=estimated_memory_residual_gb,
        memory_residual_fraction=estimated_memory_residual_fraction,
        headroom_gb=headroom,
        promotable=promotable and feasible,
        recommendation=recommendation,
        simulator_surface=support.requested_surface,
        simulator_support_status=support.support_status,
        simulator_support_blockers=support.support_blockers,
        risk_flags=risk_flags,
        communication=communication,
    )
    decision_factors.append(f"timing_coverage={timing_coverage_status}")
    if timing.step_time_s is not None:
        decision_factors.append(f"timing_step_time_s={timing.step_time_s:.6f}")
    if timing.forward_backward_s is not None:
        decision_factors.append(f"timing_forward_backward_s={timing.forward_backward_s:.6f}")
    if phase_bottleneck_half_speedup_delta_pct is not None:
        decision_factors.append(
            f"phase_bottleneck_half_speedup_delta_pct={phase_bottleneck_half_speedup_delta_pct:.3f}"
        )
    if phase_bottleneck_half_speedup_tokens_per_sec is not None:
        decision_factors.append(
            f"phase_bottleneck_half_speedup_tokens_per_sec={phase_bottleneck_half_speedup_tokens_per_sec:.3f}"
        )
    if phase_bottleneck_half_speedup_risk_adjusted_delta_pct is not None:
        decision_factors.append(
            "phase_bottleneck_half_speedup_risk_adjusted_delta_pct="
            f"{phase_bottleneck_half_speedup_risk_adjusted_delta_pct:.3f}"
        )
    if memory_details is not None:
        decision_factors.append(
            "memory_bottleneck="
            f"{memory_details[0]}:{memory_details[2]:.3f}GB,"
            f"bucket={memory_details[1]},fraction_of_peak={memory_details[3]:.3f}"
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
        score_risk_adjusted_tokens_per_gpu_per_sec=score_risk_adjusted_tokens_per_gpu_per_sec,
        prediction_uncertainty_fraction=prediction_uncertainty_fraction,
        prediction_interval_lower_tokens_per_sec=prediction_interval_lower,
        prediction_interval_upper_tokens_per_sec=prediction_interval_upper,
        risk_adjusted_prediction_interval_lower_tokens_per_sec=risk_adjusted_interval_lower,
        risk_adjusted_prediction_interval_upper_tokens_per_sec=risk_adjusted_interval_upper,
        analytic_peak_floor_gb=analytic_peak_floor_gb,
        estimated_peak_mem_gb=estimated_peak_mem_gb,
        memory_basis=memory_basis,
        memory_coverage_status=memory_coverage_status,
        memory_headroom_gb=round(headroom, 3) if headroom is not None else None,
        estimated_memory_residual_gb=estimated_memory_residual_gb,
        estimated_memory_residual_fraction_of_peak=estimated_memory_residual_fraction,
        max_ep_rank_slots_per_microbatch=max_ep_slots,
        phase_bottleneck_phase=phase_details[0] if phase_details is not None else None,
        phase_bottleneck_bucket=phase_details[1] if phase_details is not None else None,
        phase_bottleneck_share=phase_details[2] if phase_details is not None else None,
        phase_bottleneck_time_sec=phase_details[3] if phase_details is not None else None,
        memory_bottleneck_phase=memory_details[0] if memory_details is not None else None,
        memory_bottleneck_bucket=memory_details[1] if memory_details is not None else None,
        memory_bottleneck_peak_gb=memory_details[2] if memory_details is not None else None,
        memory_bottleneck_fraction_of_peak=memory_details[3] if memory_details is not None else None,
        timing_coverage_status=timing_coverage_status,
        timing_source_label=timing.source,
        timing_step_time_s=timing.step_time_s,
        timing_forward_backward_s=timing.forward_backward_s,
        calibration_scope=calibration_scope,
        recommendation=recommendation,
        phase_bottleneck_half_speedup_scale=(
            _PHASE_BOTTLENECK_HALFSPEED_SCALE if phase_bottleneck_half_speedup_delta_pct is not None else None
        ),
        phase_bottleneck_half_speedup_tokens_per_sec=phase_bottleneck_half_speedup_tokens_per_sec,
        phase_bottleneck_half_speedup_delta_pct=phase_bottleneck_half_speedup_delta_pct,
        phase_bottleneck_half_speedup_risk_adjusted_tokens_per_sec=(
            phase_bottleneck_half_speedup_risk_adjusted_tokens_per_sec
        ),
        phase_bottleneck_half_speedup_risk_adjusted_delta_pct=(phase_bottleneck_half_speedup_risk_adjusted_delta_pct),
        simulator_surface=support.requested_surface,
        simulator_support_status=support.support_status,
        simulator_support_blockers=support.support_blockers,
        target_runtime_signature=_target_runtime_signature(raw_config),
        calibration_distance=calibration_distance,
        calibration_distance_factors=calibration_distance_factors,
        memory_calibration_source=memory_calibration_source,
        memory_calibration_notes=memory_calibration_notes,
        memory_ownership_notes=memory_ownership_notes,
        communication=communication,
        decision_factors=decision_factors,
        risk_flags=risk_flags,
        notes=candidate_notes,
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


def _efficiency_sort_key(candidate: ScenarioCandidate) -> tuple[float, float]:
    return (
        candidate.score_tokens_per_gpu_per_sec if candidate.score_tokens_per_gpu_per_sec is not None else float("-inf"),
        candidate.score_tokens_per_sec if candidate.score_tokens_per_sec is not None else float("-inf"),
    )


def _risk_adjusted_efficiency_sort_key(candidate: ScenarioCandidate) -> tuple[float, float]:
    return (
        candidate.score_risk_adjusted_tokens_per_gpu_per_sec
        if candidate.score_risk_adjusted_tokens_per_gpu_per_sec is not None
        else float("-inf"),
        candidate.score_risk_adjusted_tokens_per_sec
        if candidate.score_risk_adjusted_tokens_per_sec is not None
        else float("-inf"),
    )


def _throughput_efficiency_frontier_labels(
    candidates: list[ScenarioCandidate],
    *,
    throughput_attr: str,
    efficiency_attr: str,
) -> list[str]:
    scored: list[tuple[ScenarioCandidate, float, float]] = []
    for candidate in candidates:
        throughput = getattr(candidate, throughput_attr)
        efficiency = getattr(candidate, efficiency_attr)
        if throughput is not None and efficiency is not None:
            scored.append((candidate, throughput, efficiency))

    frontier: list[tuple[ScenarioCandidate, float, float]] = []
    for candidate, throughput, efficiency in scored:
        dominated = any(
            other is not candidate
            and other_throughput >= throughput
            and other_efficiency >= efficiency
            and (other_throughput > throughput or other_efficiency > efficiency)
            for other, other_throughput, other_efficiency in scored
        )
        if not dominated:
            frontier.append((candidate, throughput, efficiency))

    return [
        candidate.label
        for candidate, _, _ in sorted(
            frontier,
            key=lambda item: (item[1], item[2], item[0].label),
            reverse=True,
        )
    ]


def _candidate_dominator(
    candidate: ScenarioCandidate,
    scored: list[tuple[ScenarioCandidate, float, float]],
    *,
    throughput: float,
    efficiency: float,
) -> tuple[ScenarioCandidate, float, float] | None:
    dominators = [
        (other, other_throughput, other_efficiency)
        for other, other_throughput, other_efficiency in scored
        if other is not candidate
        and other_throughput >= throughput
        and other_efficiency >= efficiency
        and (other_throughput > throughput or other_efficiency > efficiency)
    ]
    if not dominators:
        return None
    return max(
        dominators,
        key=lambda item: (
            item[1] - throughput,
            item[2] - efficiency,
            item[1],
            item[2],
            item[0].label,
        ),
    )


def _dominance_updates(
    candidates: list[ScenarioCandidate],
    *,
    throughput_attr: str,
    efficiency_attr: str,
    frontier_member_field: str,
    dominated_by_field: str,
    throughput_margin_field: str,
    efficiency_margin_field: str,
    decision_factor_prefix: str,
) -> dict[str, ScenarioCandidate]:
    scored: list[tuple[ScenarioCandidate, float, float]] = []
    for candidate in candidates:
        throughput = getattr(candidate, throughput_attr)
        efficiency = getattr(candidate, efficiency_attr)
        if throughput is not None and efficiency is not None:
            scored.append((candidate, throughput, efficiency))

    updates: dict[str, ScenarioCandidate] = {}
    for candidate, throughput, efficiency in scored:
        dominator = _candidate_dominator(candidate, scored, throughput=throughput, efficiency=efficiency)
        if dominator is None:
            updates[candidate.label] = replace(candidate, **{frontier_member_field: True})
            continue

        dominator_candidate, dominator_throughput, dominator_efficiency = dominator
        throughput_margin = round(dominator_throughput - throughput, 3)
        efficiency_margin = round(dominator_efficiency - efficiency, 3)
        updates[candidate.label] = replace(
            candidate,
            **{
                frontier_member_field: False,
                dominated_by_field: dominator_candidate.label,
                throughput_margin_field: throughput_margin,
                efficiency_margin_field: efficiency_margin,
                "decision_factors": [
                    *candidate.decision_factors,
                    f"{decision_factor_prefix}_dominated_by={dominator_candidate.label}",
                    f"{decision_factor_prefix}_dominance_margin_tokens_per_sec={throughput_margin:.3f}",
                    f"{decision_factor_prefix}_dominance_margin_tokens_per_gpu_per_sec={efficiency_margin:.3f}",
                ],
                "notes": [
                    *candidate.notes,
                    f"{decision_factor_prefix}_frontier_dominated_by={dominator_candidate.label}",
                ],
            },
        )
    return updates


def _apply_frontier_dominance(candidates: list[ScenarioCandidate]) -> list[ScenarioCandidate]:
    raw_updates = _dominance_updates(
        candidates,
        throughput_attr="score_tokens_per_sec",
        efficiency_attr="score_tokens_per_gpu_per_sec",
        frontier_member_field="raw_frontier_member",
        dominated_by_field="raw_dominated_by_label",
        throughput_margin_field="raw_dominance_margin_tokens_per_sec",
        efficiency_margin_field="raw_dominance_margin_tokens_per_gpu_per_sec",
        decision_factor_prefix="raw",
    )
    raw_candidates = [raw_updates.get(candidate.label, candidate) for candidate in candidates]
    risk_updates = _dominance_updates(
        raw_candidates,
        throughput_attr="score_risk_adjusted_tokens_per_sec",
        efficiency_attr="score_risk_adjusted_tokens_per_gpu_per_sec",
        frontier_member_field="risk_adjusted_frontier_member",
        dominated_by_field="risk_adjusted_dominated_by_label",
        throughput_margin_field="risk_adjusted_dominance_margin_tokens_per_sec",
        efficiency_margin_field="risk_adjusted_dominance_margin_tokens_per_gpu_per_sec",
        decision_factor_prefix="risk_adjusted",
    )
    return [risk_updates.get(candidate.label, candidate) for candidate in raw_candidates]


def _scaling_workload_signature(candidate: ScenarioCandidate) -> tuple[Any, ...]:
    return tuple(getattr(candidate.topology, dimension) for dimension in _SCENARIO_WORKLOAD_DIMENSIONS)


def _apply_same_workload_scaling_metrics(candidates: list[ScenarioCandidate]) -> list[ScenarioCandidate]:
    groups: dict[tuple[Any, ...], list[ScenarioCandidate]] = {}
    for candidate in candidates:
        if candidate.score_tokens_per_sec is None:
            continue
        groups.setdefault(_scaling_workload_signature(candidate), []).append(candidate)

    scaling_updates: dict[str, ScenarioCandidate] = {}
    for group in groups.values():
        if len({candidate.topology.world_size for candidate in group}) < 2:
            continue
        baseline = max(
            (
                candidate
                for candidate in group
                if candidate.topology.world_size == min(c.topology.world_size for c in group)
            ),
            key=_candidate_sort_key,
        )
        if not baseline.topology.world_size or not baseline.score_tokens_per_sec:
            continue

        for candidate in group:
            if candidate.score_tokens_per_sec is None or not candidate.topology.world_size:
                continue
            gpu_ratio = candidate.topology.world_size / baseline.topology.world_size
            if gpu_ratio <= 0:
                continue
            speedup = candidate.score_tokens_per_sec / baseline.score_tokens_per_sec
            scaling_efficiency = speedup / gpu_ratio
            risk_adjusted_speedup = None
            risk_adjusted_scaling_efficiency = None
            if (
                candidate.score_risk_adjusted_tokens_per_sec is not None
                and baseline.score_risk_adjusted_tokens_per_sec is not None
                and baseline.score_risk_adjusted_tokens_per_sec > 0
            ):
                risk_adjusted_speedup = (
                    candidate.score_risk_adjusted_tokens_per_sec / baseline.score_risk_adjusted_tokens_per_sec
                )
                risk_adjusted_scaling_efficiency = risk_adjusted_speedup / gpu_ratio

            decision_factors = [
                *candidate.decision_factors,
                f"scaling_baseline={baseline.label}",
                f"scaling_gpu_ratio={gpu_ratio:.3f}",
                f"scaling_speedup={speedup:.3f}",
                f"scaling_efficiency={scaling_efficiency:.3f}",
            ]
            if risk_adjusted_speedup is not None and risk_adjusted_scaling_efficiency is not None:
                decision_factors.extend(
                    [
                        f"risk_adjusted_scaling_speedup={risk_adjusted_speedup:.3f}",
                        f"risk_adjusted_scaling_efficiency={risk_adjusted_scaling_efficiency:.3f}",
                    ]
                )
            scaling_updates[candidate.label] = replace(
                candidate,
                scaling_baseline_label=baseline.label,
                scaling_baseline_world_size=baseline.topology.world_size,
                scaling_gpu_ratio=round(gpu_ratio, 3),
                scaling_speedup=round(speedup, 3),
                scaling_efficiency=round(scaling_efficiency, 3),
                risk_adjusted_scaling_speedup=(
                    round(risk_adjusted_speedup, 3) if risk_adjusted_speedup is not None else None
                ),
                risk_adjusted_scaling_efficiency=(
                    round(risk_adjusted_scaling_efficiency, 3) if risk_adjusted_scaling_efficiency is not None else None
                ),
                decision_factors=decision_factors,
                notes=[
                    *candidate.notes,
                    f"same_workload_scaling_baseline={baseline.label}",
                ],
            )

    return [scaling_updates.get(candidate.label, candidate) for candidate in candidates]


def _candidate_dimension_value(candidate: ScenarioCandidate, dimension: str) -> Any:
    return getattr(candidate.topology, dimension)


def _varied_candidate_dimensions(candidates: list[ScenarioCandidate], dimensions: tuple[str, ...]) -> list[str]:
    varied: list[str] = []
    for dimension in dimensions:
        values = {_candidate_dimension_value(candidate, dimension) for candidate in candidates}
        if len(values) > 1:
            varied.append(dimension)
    return varied


def _parallelism_strategy_key(candidate: ScenarioCandidate) -> tuple[Any, ...]:
    return tuple(_candidate_dimension_value(candidate, dimension) for dimension in _SCENARIO_PARALLELISM_DIMENSIONS)


def _parallelism_strategy_counts(candidates: list[ScenarioCandidate]) -> tuple[int, int, int, int]:
    unique_strategies = {_parallelism_strategy_key(candidate) for candidate in candidates}
    scored_strategies = {
        _parallelism_strategy_key(candidate) for candidate in candidates if candidate.score_tokens_per_sec is not None
    }
    promotable_strategies = {
        _parallelism_strategy_key(candidate)
        for candidate in candidates
        if candidate.promotable and candidate.score_tokens_per_sec is not None
    }
    remeasurement_strategies = {
        _parallelism_strategy_key(candidate)
        for candidate in candidates
        if candidate.score_tokens_per_sec is not None and "requires_remeasurement" in candidate.risk_flags
    }
    return (
        len(unique_strategies),
        len(scored_strategies),
        len(promotable_strategies),
        len(remeasurement_strategies),
    )


def _signature_for_dimensions(candidate: ScenarioCandidate, dimensions: tuple[str, ...]) -> tuple[tuple[str, Any], ...]:
    return tuple((dimension, getattr(candidate.topology, dimension)) for dimension in dimensions)


def _format_signature(signature: tuple[tuple[str, Any], ...]) -> str:
    return ",".join(f"{dimension}={value}" for dimension, value in signature)


def _candidate_runtime_signature_values(candidate: ScenarioCandidate) -> dict[str, str]:
    if candidate.target_runtime_signature == "unknown":
        return {}
    values: dict[str, str] = {}
    for part in candidate.target_runtime_signature.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def _candidate_runtime_dimension_value(candidate: ScenarioCandidate, dimension: str) -> str:
    return _candidate_runtime_signature_values(candidate).get(dimension, "unknown")


def _varied_candidate_runtime_dimensions(candidates: list[ScenarioCandidate]) -> list[str]:
    varied: list[str] = []
    for dimension in _SCENARIO_RUNTIME_DIMENSIONS:
        values = {_candidate_runtime_dimension_value(candidate, dimension) for candidate in candidates}
        if len(values) > 1:
            varied.append(dimension)
    return varied


def _runtime_dimension_sort_key(dimension: str) -> tuple[int, str]:
    try:
        return _SCENARIO_RUNTIME_DIMENSIONS.index(dimension), dimension
    except ValueError:
        return len(_SCENARIO_RUNTIME_DIMENSIONS), dimension


def _candidate_runtime_mismatch_dimensions(candidates: list[ScenarioCandidate]) -> list[str]:
    dimensions = {
        flag.split(":", 1)[1]
        for candidate in candidates
        for flag in candidate.risk_flags
        if flag.startswith("runtime_mismatch:") and flag.split(":", 1)[1] != "model_ref"
    }
    return sorted(dimensions, key=_runtime_dimension_sort_key)


def _runtime_signature_for_candidate(candidate: ScenarioCandidate) -> tuple[tuple[str, Any], ...]:
    return (("target_runtime_signature", candidate.target_runtime_signature),)


def _benchmark_point_parallel_size(point: BenchmarkBehaviorPoint, field_name: str) -> int:
    value = getattr(point, field_name)
    return int(value) if value is not None else 1


def _benchmark_point_topology_values(
    point: BenchmarkBehaviorPoint,
    base_topology: Topology,
) -> dict[str, Any]:
    world_size = point.gpu_count or base_topology.world_size
    local_world_size = base_topology.local_world_size
    tensor_parallel_size = _benchmark_point_parallel_size(point, "tensor_parallel_size")
    pipeline_parallel_size = _benchmark_point_parallel_size(point, "pipeline_parallel_size")
    expert_parallel_size = _benchmark_point_parallel_size(point, "expert_parallel_size")
    ulysses_parallel_size = _benchmark_point_parallel_size(point, "ulysses_parallel_size")
    ringattn_parallel_size = _benchmark_point_parallel_size(point, "ringattn_parallel_size")
    non_dp_size = tensor_parallel_size * pipeline_parallel_size * ulysses_parallel_size * ringattn_parallel_size
    data_parallel_size = world_size // non_dp_size if non_dp_size and world_size % non_dp_size == 0 else None
    if data_parallel_size is None:
        replicate_size = None
        shard_size = None
    else:
        if point.data_parallel_replicate_size is not None or point.data_parallel_shard_size is not None:
            preferred_replicate_size = point.data_parallel_replicate_size
            preferred_shard_size = point.data_parallel_shard_size
        else:
            preferred_replicate_size = base_topology.data_parallel_replicate_size
            preferred_shard_size = base_topology.data_parallel_shard_size
        replicate_size, shard_size = _valid_dp_split_for_size(
            data_parallel_size,
            preferred_replicate_size=preferred_replicate_size,
            preferred_shard_size=preferred_shard_size,
        )
    ep_fsdp_size = point.ep_fsdp_size
    if ep_fsdp_size is None:
        ranks_per_pipeline_stage = (
            world_size // pipeline_parallel_size if world_size % pipeline_parallel_size == 0 else None
        )
        if ranks_per_pipeline_stage is not None and ranks_per_pipeline_stage % expert_parallel_size == 0:
            ep_fsdp_size = ranks_per_pipeline_stage // expert_parallel_size

    return {
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_count": world_size // local_world_size
        if local_world_size and world_size % local_world_size == 0
        else None,
        "data_parallel_size": data_parallel_size,
        "data_parallel_replicate_size": replicate_size,
        "data_parallel_shard_size": shard_size,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "ep_fsdp_size": ep_fsdp_size,
        "ulysses_parallel_size": ulysses_parallel_size,
        "ringattn_parallel_size": ringattn_parallel_size,
        "micro_batch_size": point.micro_batch_size,
        "gradient_accumulation_steps": (
            point.global_batch_size // point.micro_batch_size // data_parallel_size
            if point.global_batch_size is not None
            and point.micro_batch_size is not None
            and data_parallel_size
            and point.global_batch_size % (point.micro_batch_size * data_parallel_size) == 0
            else None
        ),
        "global_batch_size": point.global_batch_size,
        "sample_packing_sequence_len": point.sample_packing_sequence_len,
    }


def _benchmark_point_dimension_value(
    point: BenchmarkBehaviorPoint,
    dimension: str,
    base_topology: Topology,
) -> Any:
    return _benchmark_point_topology_values(point, base_topology).get(dimension)


def _benchmark_signature_for_dimensions(
    point: BenchmarkBehaviorPoint,
    dimensions: tuple[str, ...],
    base_topology: Topology,
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (dimension, _benchmark_point_dimension_value(point, dimension, base_topology)) for dimension in dimensions
    )


def _benchmark_point_runtime_dimension_value(point: BenchmarkBehaviorPoint, dimension: str) -> str:
    value = getattr(point, dimension, None)
    if value is None:
        return "unknown"
    return str(value)


def _benchmark_runtime_signature_for_point(point: BenchmarkBehaviorPoint) -> tuple[tuple[str, Any], ...]:
    return (
        (
            "target_runtime_signature",
            ",".join(
                f"{dimension}={_benchmark_point_runtime_dimension_value(point, dimension)}"
                for dimension in _SCENARIO_RUNTIME_DIMENSIONS
            ),
        ),
    )


def _benchmark_varied_dimensions(
    points: list[BenchmarkBehaviorPoint],
    dimensions: tuple[str, ...],
    base_topology: Topology,
) -> list[str]:
    varied: list[str] = []
    for dimension in dimensions:
        values = {_benchmark_point_dimension_value(point, dimension, base_topology) for point in points}
        values.discard(None)
        if len(values) > 1:
            varied.append(dimension)
    return varied


def _benchmark_varied_runtime_dimensions(points: list[BenchmarkBehaviorPoint]) -> list[str]:
    varied: list[str] = []
    for dimension in _SCENARIO_RUNTIME_DIMENSIONS:
        values = {_benchmark_point_runtime_dimension_value(point, dimension) for point in points}
        if len(values) > 1:
            varied.append(dimension)
    return varied


def _benchmark_axis_value(
    point: BenchmarkBehaviorPoint,
    dimensions: tuple[str, ...],
    base_topology: Topology,
) -> str:
    return _format_signature(_benchmark_signature_for_dimensions(point, dimensions, base_topology))


def _benchmark_point_is_memory_blocked(point: BenchmarkBehaviorPoint) -> bool:
    return point.correctness_status == "oom"


def _benchmark_parallelism_axis_coverage(
    points: list[BenchmarkBehaviorPoint],
    base_topology: Topology,
) -> list[ScenarioParallelismAxisCoverage]:
    coverage: list[ScenarioParallelismAxisCoverage] = []
    workload_dimensions = _SCENARIO_WORKLOAD_DIMENSIONS
    for axis, axis_dimensions in _PARALLELISM_AXIS_DIMENSIONS.items():
        outside_dimensions = tuple(
            dimension for dimension in _PARALLELISM_COMPARISON_DIMENSIONS if dimension not in axis_dimensions
        )
        groups: dict[tuple[tuple[str, Any], ...], list[BenchmarkBehaviorPoint]] = {}
        for point in points:
            key = (
                *_benchmark_signature_for_dimensions(point, workload_dimensions, base_topology),
                *_benchmark_signature_for_dimensions(point, outside_dimensions, base_topology),
                *_benchmark_runtime_signature_for_point(point),
            )
            groups.setdefault(key, []).append(point)

        varied_groups = [
            group
            for group in groups.values()
            if len({_benchmark_axis_value(point, axis_dimensions, base_topology) for point in group}) > 1
        ]
        if not varied_groups:
            varied_dimensions = _benchmark_varied_dimensions(points, axis_dimensions, base_topology)
            primary_dimensions = [
                dimension for dimension in varied_dimensions if dimension in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
            ]
            co_varied_dimensions = [
                dimension
                for dimension in varied_dimensions
                if dimension not in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
            ]
            if primary_dimensions:
                scored_points = [point for point in points if point.tokens_per_sec is not None]
                blocked_points = [point for point in points if _benchmark_point_is_memory_blocked(point)]
                unscored_points = [
                    point
                    for point in points
                    if point.tokens_per_sec is None and not _benchmark_point_is_memory_blocked(point)
                ]
                if scored_points and blocked_points:
                    status = "confounded_benchmark_single_scored_axis_with_blocked_alternatives"
                elif blocked_points:
                    status = "confounded_benchmark_blocked_parallelism_axis"
                elif scored_points:
                    status = "confounded_benchmark_scored_parallelism_axis"
                else:
                    status = "confounded_benchmark_unscored_parallelism_axis"
                coverage.append(
                    ScenarioParallelismAxisCoverage(
                        axis=axis,
                        status=status,
                        candidate_group_count=0,
                        candidate_count=len(points),
                        scored_count=len(scored_points),
                        blocked_count=len(blocked_points),
                        unscored_count=len(unscored_points),
                        varied_dimensions=varied_dimensions,
                        primary_varied_dimensions=primary_dimensions,
                        co_varied_axis_dimensions=co_varied_dimensions,
                        confounded_runtime_dimensions=_benchmark_varied_runtime_dimensions(points),
                        feasibility_status_counts=_count_values(
                            [
                                "observed_oom" if _benchmark_point_is_memory_blocked(point) else "observed_fit"
                                for point in points
                            ]
                        ),
                    )
                )
                continue
            coverage.append(
                ScenarioParallelismAxisCoverage(
                    axis=axis,
                    status="missing_benchmark_parallelism_axis",
                    candidate_group_count=0,
                    candidate_count=0,
                    scored_count=0,
                    blocked_count=0,
                    unscored_count=0,
                    confounded_runtime_dimensions=[],
                )
            )
            continue

        grouped_points = [point for group in varied_groups for point in group]
        scored_points = [point for point in grouped_points if point.tokens_per_sec is not None]
        blocked_points = [point for point in grouped_points if _benchmark_point_is_memory_blocked(point)]
        unscored_points = [
            point
            for point in grouped_points
            if point.tokens_per_sec is None and not _benchmark_point_is_memory_blocked(point)
        ]
        has_scored_axis_comparison = any(
            len(
                {
                    _benchmark_axis_value(point, axis_dimensions, base_topology)
                    for point in group
                    if point.tokens_per_sec is not None
                }
            )
            > 1
            for group in varied_groups
        )
        has_single_scored_blocked_alternatives = any(
            len([point for point in group if point.tokens_per_sec is not None]) == 1
            and any(point.tokens_per_sec is None for point in group)
            and any(_benchmark_point_is_memory_blocked(point) for point in group)
            for group in varied_groups
        )
        if has_scored_axis_comparison:
            status = "scored_benchmark_parallelism_axis"
        elif has_single_scored_blocked_alternatives:
            status = "benchmark_single_scored_axis_with_blocked_alternatives"
        elif blocked_points:
            status = "blocked_benchmark_parallelism_axis"
        else:
            status = "unscored_benchmark_parallelism_axis"
        varied_dimensions = _benchmark_varied_dimensions(grouped_points, axis_dimensions, base_topology)
        primary_dimensions = [
            dimension for dimension in varied_dimensions if dimension in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
        ]
        co_varied_dimensions = [
            dimension for dimension in varied_dimensions if dimension not in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
        ]
        coverage.append(
            ScenarioParallelismAxisCoverage(
                axis=axis,
                status=status,
                candidate_group_count=len(varied_groups),
                candidate_count=len(grouped_points),
                scored_count=len(scored_points),
                blocked_count=len(blocked_points),
                unscored_count=len(unscored_points),
                varied_dimensions=varied_dimensions,
                primary_varied_dimensions=primary_dimensions,
                co_varied_axis_dimensions=co_varied_dimensions,
                confounded_runtime_dimensions=[],
                feasibility_status_counts=_count_values(
                    [
                        "observed_oom" if _benchmark_point_is_memory_blocked(point) else "observed_fit"
                        for point in grouped_points
                    ]
                ),
            )
        )
    return coverage


def _benchmark_support_status(
    *,
    point_count: int,
    scored_count: int,
    memory_blocked_count: int,
    varied_parallelism_dimensions: list[str],
    varied_workload_dimensions: list[str],
    varied_runtime_dimensions: list[str],
    parallelism_axis_coverage_status_counts: dict[str, int],
) -> tuple[str, list[str]]:
    if point_count == 0:
        return "no_benchmark_support", ["no_benchmark_points"]

    has_parallelism_variation = bool(varied_parallelism_dimensions)
    has_workload_variation = bool(varied_workload_dimensions)
    clean_axis_count = parallelism_axis_coverage_status_counts.get("scored_benchmark_parallelism_axis", 0)
    blockers: set[str] = set()
    if point_count == 1:
        blockers.add("single_benchmark_point")
    if scored_count == 0:
        blockers.add("no_scored_benchmark_points")
    elif scored_count < point_count:
        blockers.add("unscored_benchmark_points")
    if memory_blocked_count == point_count:
        blockers.add("memory_blocked_all_benchmark_points")
    if not has_parallelism_variation:
        blockers.add("missing_parallelism_benchmark_support")
    if not has_workload_variation:
        blockers.add("missing_workload_benchmark_support")
    if varied_runtime_dimensions:
        blockers.add("runtime_variant_benchmark_support")
    if has_parallelism_variation and clean_axis_count == 0:
        blockers.add("no_clean_benchmark_parallelism_axis_coverage")
    if any("blocked" in status for status in parallelism_axis_coverage_status_counts):
        blockers.add("blocked_benchmark_parallelism_axes")
    if any(status.startswith("confounded_") for status in parallelism_axis_coverage_status_counts):
        blockers.add("confounded_benchmark_parallelism_axes")
    if any("unscored" in status for status in parallelism_axis_coverage_status_counts):
        blockers.add("unscored_benchmark_parallelism_axes")

    blocker_list = sorted(blockers)
    if not has_parallelism_variation and not has_workload_variation:
        return "single_shape_benchmark_support", blocker_list
    if has_workload_variation and not has_parallelism_variation:
        return "workload_only_benchmark_support", blocker_list
    if has_parallelism_variation and not has_workload_variation:
        if scored_count == 0:
            return "unscored_parallelism_benchmark_support", blocker_list
        if clean_axis_count == 0:
            return "parallelism_benchmark_support_without_clean_axis", blocker_list
        if blockers - {"missing_workload_benchmark_support"}:
            return "partial_parallelism_benchmark_support", blocker_list
        return "parallelism_only_benchmark_support", blocker_list

    if scored_count == 0:
        return "unscored_broad_benchmark_support", blocker_list
    if clean_axis_count == 0:
        return "confounded_broad_benchmark_support", blocker_list
    if blockers:
        return "partial_broad_benchmark_support", blocker_list
    return "broad_benchmark_support", []


def _scenario_benchmark_support(
    behavior_points: list[BenchmarkBehaviorPoint],
    *,
    base_config: dict[str, Any],
    base_topology: Topology,
) -> ScenarioBenchmarkSupport:
    support_points = [
        point
        for point in behavior_points
        if not behavior_point_model_mismatches(point, base_config)
        and (point.tokens_per_sec is not None or point.correctness_status == "oom")
        and (
            point.sample_packing_sequence_len is None
            or base_topology.sample_packing_sequence_len is None
            or point.sample_packing_sequence_len == base_topology.sample_packing_sequence_len
        )
    ]
    if not support_points:
        return ScenarioBenchmarkSupport()

    parallelism_axis_coverage = _benchmark_parallelism_axis_coverage(support_points, base_topology)
    parallelism_axis_coverage_status_counts = _count_values([coverage.status for coverage in parallelism_axis_coverage])
    varied_parallelism_dimensions = _benchmark_varied_dimensions(
        support_points,
        _SCENARIO_PARALLELISM_DIMENSIONS,
        base_topology,
    )
    varied_workload_dimensions = _benchmark_varied_dimensions(
        support_points,
        _SCENARIO_WORKLOAD_DIMENSIONS,
        base_topology,
    )
    varied_runtime_dimensions = _benchmark_varied_runtime_dimensions(support_points)
    scored_count = sum(1 for point in support_points if point.tokens_per_sec is not None)
    memory_blocked_count = sum(1 for point in support_points if _benchmark_point_is_memory_blocked(point))
    support_status, support_blockers = _benchmark_support_status(
        point_count=len(support_points),
        scored_count=scored_count,
        memory_blocked_count=memory_blocked_count,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
    )
    return ScenarioBenchmarkSupport(
        support_status=support_status,
        support_blockers=support_blockers,
        point_count=len(support_points),
        scored_count=scored_count,
        memory_blocked_count=memory_blocked_count,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
        scored_parallelism_axis_names=[
            coverage.axis
            for coverage in parallelism_axis_coverage
            if coverage.status == "scored_benchmark_parallelism_axis"
        ],
        blocked_parallelism_axis_names=[
            coverage.axis for coverage in parallelism_axis_coverage if "blocked" in coverage.status
        ],
        confounded_parallelism_axis_names=[
            coverage.axis for coverage in parallelism_axis_coverage if coverage.status.startswith("confounded_")
        ],
        unscored_parallelism_axis_names=[
            coverage.axis
            for coverage in parallelism_axis_coverage
            if coverage.status == "unscored_benchmark_parallelism_axis"
        ],
        missing_parallelism_axis_names=[
            coverage.axis
            for coverage in parallelism_axis_coverage
            if coverage.status == "missing_benchmark_parallelism_axis"
        ],
        point_labels=sorted(point.label for point in support_points),
    )


def _axis_value(candidate: ScenarioCandidate, dimensions: tuple[str, ...]) -> str:
    return _format_signature(_signature_for_dimensions(candidate, dimensions))


def _axis_best_worst(
    candidates: list[ScenarioCandidate],
    *,
    score_attr: str,
) -> tuple[ScenarioCandidate | None, ScenarioCandidate | None, float | None, float | None]:
    scored = [candidate for candidate in candidates if getattr(candidate, score_attr) is not None]
    if not scored:
        return None, None, None, None
    best = max(scored, key=lambda candidate: (getattr(candidate, score_attr), candidate.label))
    worst = min(scored, key=lambda candidate: (getattr(candidate, score_attr), candidate.label))
    best_score = getattr(best, score_attr)
    worst_score = getattr(worst, score_attr)
    spread = round(best_score - worst_score, 3)
    ratio = round(best_score / worst_score, 3) if worst_score and worst_score > 0 else None
    return best, worst, spread, ratio


def _axis_comparison_status(
    raw_best: ScenarioCandidate | None,
    raw_spread: float | None,
    risk_adjusted_best: ScenarioCandidate | None,
    risk_adjusted_spread: float | None,
) -> str:
    if raw_best is None and risk_adjusted_best is None:
        return "unscored_axis_comparison"
    if raw_best is None:
        return "risk_adjusted_only_axis_comparison"
    if risk_adjusted_best is None:
        return "raw_only_axis_comparison"
    if raw_spread == 0 and risk_adjusted_spread == 0:
        return "axis_tie"
    if raw_best.label == risk_adjusted_best.label:
        return "raw_and_risk_adjusted_agree"
    return "risk_adjusted_changes_axis_winner"


def _axis_interval_overlap_summary(
    group: list[ScenarioCandidate],
    risk_adjusted_best: ScenarioCandidate | None,
) -> tuple[str, int, list[str], float | None]:
    if risk_adjusted_best is None:
        return "no_scored_interval", 0, [], None
    best_lower = risk_adjusted_best.risk_adjusted_prediction_interval_lower_tokens_per_sec
    best_upper = risk_adjusted_best.risk_adjusted_prediction_interval_upper_tokens_per_sec
    if best_lower is None or best_upper is None:
        return "no_scored_interval", 0, [], None

    overlap_labels: list[str] = []
    other_upper_bounds: list[float] = []
    for candidate in group:
        if candidate.label == risk_adjusted_best.label:
            continue
        lower = candidate.risk_adjusted_prediction_interval_lower_tokens_per_sec
        upper = candidate.risk_adjusted_prediction_interval_upper_tokens_per_sec
        if lower is None or upper is None:
            continue
        other_upper_bounds.append(upper)
        if lower <= best_upper and upper >= best_lower:
            overlap_labels.append(candidate.label)

    if not other_upper_bounds:
        return "single_scored_interval", 0, [], None

    margin = round(best_lower - max(other_upper_bounds), 3)
    if overlap_labels:
        return "overlapping_best_interval", len(overlap_labels), sorted(overlap_labels), margin
    return "clear_best_interval", 0, [], margin


def _parallelism_axis_comparisons(candidates: list[ScenarioCandidate]) -> list[ParallelismAxisComparison]:
    comparisons: list[ParallelismAxisComparison] = []
    workload_dimensions = _SCENARIO_WORKLOAD_DIMENSIONS
    for axis, axis_dimensions in _PARALLELISM_AXIS_DIMENSIONS.items():
        outside_dimensions = tuple(
            dimension for dimension in _PARALLELISM_COMPARISON_DIMENSIONS if dimension not in axis_dimensions
        )
        groups: dict[tuple[tuple[str, Any], ...], list[ScenarioCandidate]] = {}
        for candidate in candidates:
            if candidate.score_tokens_per_sec is None:
                continue
            key = (
                *_signature_for_dimensions(candidate, workload_dimensions),
                *_signature_for_dimensions(candidate, outside_dimensions),
                *_runtime_signature_for_candidate(candidate),
            )
            groups.setdefault(key, []).append(candidate)

        for group_key, group in groups.items():
            axis_values = {_axis_value(candidate, axis_dimensions) for candidate in group}
            if len(group) < 2 or len(axis_values) < 2:
                continue

            raw_best, raw_worst, raw_spread, raw_ratio = _axis_best_worst(
                group,
                score_attr="score_tokens_per_sec",
            )
            risk_best, risk_worst, risk_spread, risk_ratio = _axis_best_worst(
                group,
                score_attr="score_risk_adjusted_tokens_per_sec",
            )
            (
                interval_overlap_status,
                interval_overlap_count,
                interval_overlap_labels,
                interval_margin,
            ) = _axis_interval_overlap_summary(group, risk_best)
            status = _axis_comparison_status(raw_best, raw_spread, risk_best, risk_spread)
            varied_dimensions = _varied_candidate_dimensions(group, axis_dimensions)
            primary_dimensions = [
                dimension for dimension in varied_dimensions if dimension in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
            ]
            co_varied_dimensions = [
                dimension
                for dimension in varied_dimensions
                if dimension not in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
            ]
            comparisons.append(
                ParallelismAxisComparison(
                    axis=axis,
                    varied_dimensions=varied_dimensions,
                    primary_varied_dimensions=primary_dimensions,
                    co_varied_axis_dimensions=co_varied_dimensions,
                    coupling_status=("coupled_axis_comparison" if co_varied_dimensions else "isolated_axis_comparison"),
                    group_key=_format_signature(group_key),
                    candidate_count=len(group),
                    raw_best_label=raw_best.label if raw_best is not None else None,
                    raw_best_axis_value=_axis_value(raw_best, axis_dimensions) if raw_best is not None else None,
                    raw_best_score_tokens_per_sec=(raw_best.score_tokens_per_sec if raw_best is not None else None),
                    raw_worst_label=raw_worst.label if raw_worst is not None else None,
                    raw_worst_axis_value=_axis_value(raw_worst, axis_dimensions) if raw_worst is not None else None,
                    raw_worst_score_tokens_per_sec=(raw_worst.score_tokens_per_sec if raw_worst is not None else None),
                    raw_spread_tokens_per_sec=raw_spread,
                    raw_spread_ratio=raw_ratio,
                    risk_adjusted_best_label=risk_best.label if risk_best is not None else None,
                    risk_adjusted_best_axis_value=(
                        _axis_value(risk_best, axis_dimensions) if risk_best is not None else None
                    ),
                    risk_adjusted_best_score_tokens_per_sec=(
                        risk_best.score_risk_adjusted_tokens_per_sec if risk_best is not None else None
                    ),
                    risk_adjusted_worst_label=risk_worst.label if risk_worst is not None else None,
                    risk_adjusted_worst_axis_value=(
                        _axis_value(risk_worst, axis_dimensions) if risk_worst is not None else None
                    ),
                    risk_adjusted_worst_score_tokens_per_sec=(
                        risk_worst.score_risk_adjusted_tokens_per_sec if risk_worst is not None else None
                    ),
                    risk_adjusted_spread_tokens_per_sec=risk_spread,
                    risk_adjusted_spread_ratio=risk_ratio,
                    risk_adjusted_winner_matches_raw=(
                        raw_best.label == risk_best.label if raw_best is not None and risk_best is not None else None
                    ),
                    comparison_status=status,
                    risk_adjusted_best_interval_lower_tokens_per_sec=(
                        risk_best.risk_adjusted_prediction_interval_lower_tokens_per_sec
                        if risk_best is not None
                        else None
                    ),
                    risk_adjusted_best_interval_upper_tokens_per_sec=(
                        risk_best.risk_adjusted_prediction_interval_upper_tokens_per_sec
                        if risk_best is not None
                        else None
                    ),
                    risk_adjusted_worst_interval_lower_tokens_per_sec=(
                        risk_worst.risk_adjusted_prediction_interval_lower_tokens_per_sec
                        if risk_worst is not None
                        else None
                    ),
                    risk_adjusted_worst_interval_upper_tokens_per_sec=(
                        risk_worst.risk_adjusted_prediction_interval_upper_tokens_per_sec
                        if risk_worst is not None
                        else None
                    ),
                    risk_adjusted_interval_overlap_status=interval_overlap_status,
                    risk_adjusted_interval_overlap_candidate_count=interval_overlap_count,
                    risk_adjusted_interval_overlap_candidate_labels=interval_overlap_labels,
                    risk_adjusted_interval_margin_tokens_per_sec=interval_margin,
                )
            )

    return sorted(
        comparisons,
        key=lambda comparison: (
            comparison.risk_adjusted_spread_tokens_per_sec
            if comparison.risk_adjusted_spread_tokens_per_sec is not None
            else float("-inf"),
            comparison.raw_spread_tokens_per_sec if comparison.raw_spread_tokens_per_sec is not None else float("-inf"),
            comparison.axis,
            comparison.group_key,
        ),
        reverse=True,
    )


def _parallelism_axis_applies_to_scenario_candidates(axis: str, candidates: list[ScenarioCandidate]) -> bool:
    if not candidates:
        return True
    if axis == "ulysses":
        return any(
            candidate.topology.ulysses_parallel_size > 1
            or (candidate.topology.sample_packing_sequence_len or 0) >= _MIN_ULYSSES_SEQUENCE_LEN
            for candidate in candidates
        )
    if axis == "ringattn":
        return any(
            candidate.topology.ringattn_parallel_size > 1
            or (candidate.topology.sample_packing_sequence_len or 0) >= _MIN_RINGATTN_SEQUENCE_LEN
            for candidate in candidates
        )
    return True


def _scenario_parallelism_axis_coverage(candidates: list[ScenarioCandidate]) -> list[ScenarioParallelismAxisCoverage]:
    coverage: list[ScenarioParallelismAxisCoverage] = []
    workload_dimensions = _SCENARIO_WORKLOAD_DIMENSIONS
    for axis, axis_dimensions in _PARALLELISM_AXIS_DIMENSIONS.items():
        if not _parallelism_axis_applies_to_scenario_candidates(axis, candidates):
            coverage.append(
                ScenarioParallelismAxisCoverage(
                    axis=axis,
                    status="not_applicable_parallelism_axis",
                    candidate_group_count=0,
                    candidate_count=len(candidates),
                    scored_count=0,
                    blocked_count=0,
                    unscored_count=0,
                    confounded_runtime_dimensions=[],
                )
            )
            continue
        outside_dimensions = tuple(
            dimension for dimension in _PARALLELISM_COMPARISON_DIMENSIONS if dimension not in axis_dimensions
        )
        groups: dict[tuple[tuple[str, Any], ...], list[ScenarioCandidate]] = {}
        for candidate in candidates:
            key = (
                *_signature_for_dimensions(candidate, workload_dimensions),
                *_signature_for_dimensions(candidate, outside_dimensions),
                *_runtime_signature_for_candidate(candidate),
            )
            groups.setdefault(key, []).append(candidate)

        varied_groups = [
            group
            for group in groups.values()
            if len({_axis_value(candidate, axis_dimensions) for candidate in group}) > 1
        ]
        if not varied_groups:
            varied_dimensions = _varied_candidate_dimensions(candidates, axis_dimensions)
            primary_dimensions = [
                dimension for dimension in varied_dimensions if dimension in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
            ]
            co_varied_dimensions = [
                dimension
                for dimension in varied_dimensions
                if dimension not in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
            ]
            if primary_dimensions:
                scored_candidates = [
                    candidate for candidate in candidates if candidate.score_tokens_per_sec is not None
                ]
                blocked_candidates = [candidate for candidate in candidates if _is_memory_blocked(candidate)]
                unscored_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.score_tokens_per_sec is None and not _is_memory_blocked(candidate)
                ]
                if scored_candidates and blocked_candidates:
                    status = "confounded_single_scored_axis_with_blocked_alternatives"
                elif blocked_candidates:
                    status = "confounded_blocked_parallelism_axis"
                elif scored_candidates:
                    status = "confounded_scored_parallelism_axis"
                else:
                    status = "confounded_unscored_parallelism_axis"
                coverage.append(
                    ScenarioParallelismAxisCoverage(
                        axis=axis,
                        status=status,
                        candidate_group_count=0,
                        candidate_count=len(candidates),
                        scored_count=len(scored_candidates),
                        blocked_count=len(blocked_candidates),
                        unscored_count=len(unscored_candidates),
                        varied_dimensions=varied_dimensions,
                        primary_varied_dimensions=primary_dimensions,
                        co_varied_axis_dimensions=co_varied_dimensions,
                        confounded_runtime_dimensions=_varied_candidate_runtime_dimensions(candidates),
                        feasibility_status_counts=_count_values(
                            [candidate.feasibility_status for candidate in candidates]
                        ),
                    )
                )
                continue
            coverage.append(
                ScenarioParallelismAxisCoverage(
                    axis=axis,
                    status="missing_parallelism_axis",
                    candidate_group_count=0,
                    candidate_count=0,
                    scored_count=0,
                    blocked_count=0,
                    unscored_count=0,
                    confounded_runtime_dimensions=[],
                )
            )
            continue

        grouped_candidates = [candidate for group in varied_groups for candidate in group]
        scored_candidates = [
            candidate for candidate in grouped_candidates if candidate.score_tokens_per_sec is not None
        ]
        blocked_candidates = [candidate for candidate in grouped_candidates if _is_memory_blocked(candidate)]
        unscored_candidates = [
            candidate
            for candidate in grouped_candidates
            if candidate.score_tokens_per_sec is None and not _is_memory_blocked(candidate)
        ]
        has_scored_axis_comparison = any(
            len(
                {
                    _axis_value(candidate, axis_dimensions)
                    for candidate in group
                    if candidate.score_tokens_per_sec is not None
                }
            )
            > 1
            for group in varied_groups
        )
        has_single_scored_blocked_alternatives = any(
            len([candidate for candidate in group if candidate.score_tokens_per_sec is not None]) == 1
            and any(candidate.score_tokens_per_sec is None for candidate in group)
            and any(_is_memory_blocked(candidate) for candidate in group)
            for group in varied_groups
        )
        if has_scored_axis_comparison:
            status = "scored_parallelism_axis"
        elif has_single_scored_blocked_alternatives:
            status = "single_scored_axis_with_blocked_alternatives"
        elif blocked_candidates:
            status = "blocked_parallelism_axis"
        else:
            status = "unscored_parallelism_axis"
        varied_dimensions = _varied_candidate_dimensions(grouped_candidates, axis_dimensions)
        primary_dimensions = [
            dimension for dimension in varied_dimensions if dimension in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
        ]
        co_varied_dimensions = [
            dimension for dimension in varied_dimensions if dimension not in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
        ]
        coverage.append(
            ScenarioParallelismAxisCoverage(
                axis=axis,
                status=status,
                candidate_group_count=len(varied_groups),
                candidate_count=len(grouped_candidates),
                scored_count=len(scored_candidates),
                blocked_count=len(blocked_candidates),
                unscored_count=len(unscored_candidates),
                varied_dimensions=varied_dimensions,
                primary_varied_dimensions=primary_dimensions,
                co_varied_axis_dimensions=co_varied_dimensions,
                confounded_runtime_dimensions=[],
                feasibility_status_counts=_count_values(
                    [candidate.feasibility_status for candidate in grouped_candidates]
                ),
            )
        )
    return coverage


def _scenario_boundary_dimension_value(candidate: ScenarioCandidate, dimension: str) -> Any:
    if dimension == "balanced_routing":
        return candidate.behavior.balanced_routing
    return getattr(candidate.topology, dimension)


def _scenario_boundary_varied_dimensions(
    candidates: list[ScenarioCandidate],
    dimensions: tuple[str, ...],
) -> list[str]:
    varied: list[str] = []
    for dimension in dimensions:
        values = {_scenario_boundary_dimension_value(candidate, dimension) for candidate in candidates}
        if len(values) > 1:
            varied.append(dimension)
    return varied


def _scenario_boundary_signature(candidate: ScenarioCandidate) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (dimension, _scenario_boundary_dimension_value(candidate, dimension))
        for dimension in _SCENARIO_BOUNDARY_SIGNATURE_DIMENSIONS
    )


def _scenario_boundary_outcome(candidate: ScenarioCandidate) -> str | None:
    if _is_memory_blocked(candidate):
        return "failure"
    if candidate.score_tokens_per_sec is not None:
        return "fit"
    return None


def _scenario_parallelism_boundary_groups(candidates: list[ScenarioCandidate]) -> list[ParallelismBoundaryGroup]:
    grouped: dict[tuple[tuple[str, Any], ...], list[ScenarioCandidate]] = {}
    for candidate in candidates:
        if _scenario_boundary_outcome(candidate) is None:
            continue
        grouped.setdefault(_scenario_boundary_signature(candidate), []).append(candidate)

    boundary_groups: list[ParallelismBoundaryGroup] = []
    for signature, group in grouped.items():
        varied_parallelism = _varied_candidate_dimensions(group, _SCENARIO_PARALLELISM_DIMENSIONS)
        if not varied_parallelism:
            continue
        outcomes = [_scenario_boundary_outcome(candidate) for candidate in group]
        if "fit" not in outcomes or "failure" not in outcomes:
            continue
        fits = [candidate for candidate in group if _scenario_boundary_outcome(candidate) == "fit"]
        failures = [candidate for candidate in group if _scenario_boundary_outcome(candidate) == "failure"]
        best_fit = max(fits, key=_candidate_sort_key) if fits else None
        boundary_groups.append(
            ParallelismBoundaryGroup(
                signature=_format_signature(signature),
                candidate_count=len(group),
                fit_count=len(fits),
                failure_count=len(failures),
                best_fit_label=best_fit.label if best_fit is not None else None,
                best_fit_tokens_per_sec=best_fit.score_tokens_per_sec if best_fit is not None else None,
                failure_labels=sorted(candidate.label for candidate in failures),
                varied_parallelism_dimensions=varied_parallelism,
                confounded_workload_dimensions=[
                    dimension
                    for dimension in _scenario_boundary_varied_dimensions(group, _SCENARIO_BOUNDARY_WORKLOAD_DIMENSIONS)
                    if dimension not in _SCENARIO_BOUNDARY_SIGNATURE_DIMENSIONS
                ],
                confounded_runtime_dimensions=_varied_candidate_runtime_dimensions(group),
            )
        )
    return sorted(boundary_groups, key=lambda group: group.signature)


def _scenario_parallelism_boundary_status(
    boundary_groups: list[ParallelismBoundaryGroup],
    candidates: list[ScenarioCandidate],
) -> str:
    observed = [candidate for candidate in candidates if _scenario_boundary_outcome(candidate) is not None]
    if not observed:
        return "insufficient_data"
    if not boundary_groups:
        if len({_parallelism_strategy_key(candidate) for candidate in observed}) > 1:
            return "no_fit_failure_parallelism_boundary"
        return "no_measured_parallelism_variation"
    if any(
        not group.confounded_workload_dimensions and not group.confounded_runtime_dimensions
        for group in boundary_groups
    ):
        return "measured_parallelism_fit_failure_boundary"
    return "confounded_parallelism_fit_failure_boundary"


def _scenario_parallelism_boundary_axis_coverage(
    boundary_groups: list[ParallelismBoundaryGroup],
) -> list[ParallelismBoundaryAxisCoverage]:
    coverage: list[ParallelismBoundaryAxisCoverage] = []
    for axis in _PARALLELISM_AXIS_DIMENSIONS:
        primary_dimensions = _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis]
        axis_groups = [
            group
            for group in boundary_groups
            if any(dimension in group.varied_parallelism_dimensions for dimension in primary_dimensions)
        ]
        if not axis_groups:
            coverage.append(
                ParallelismBoundaryAxisCoverage(
                    axis=axis,
                    status="missing_parallelism_boundary_axis",
                    group_count=0,
                    candidate_count=0,
                    fit_count=0,
                    failure_count=0,
                    varied_parallelism_dimensions=[],
                    co_varied_parallelism_dimensions=[],
                    confounded_workload_dimensions=[],
                    confounded_runtime_dimensions=[],
                )
            )
            continue

        varied_parallelism_dimensions = [
            dimension
            for dimension in _SCENARIO_PARALLELISM_DIMENSIONS
            if any(dimension in group.varied_parallelism_dimensions for group in axis_groups)
        ]
        co_varied_parallelism_dimensions = [
            dimension for dimension in varied_parallelism_dimensions if dimension not in primary_dimensions
        ]
        confounded_workload_dimensions = sorted(
            {dimension for group in axis_groups for dimension in group.confounded_workload_dimensions}
        )
        confounded_runtime_dimensions = sorted(
            {dimension for group in axis_groups for dimension in group.confounded_runtime_dimensions},
            key=_SCENARIO_RUNTIME_DIMENSIONS.index,
        )
        status = (
            "confounded_parallelism_boundary_axis"
            if confounded_workload_dimensions or confounded_runtime_dimensions
            else "measured_parallelism_boundary_axis"
        )
        coverage.append(
            ParallelismBoundaryAxisCoverage(
                axis=axis,
                status=status,
                group_count=len(axis_groups),
                candidate_count=sum(group.candidate_count for group in axis_groups),
                fit_count=sum(group.fit_count for group in axis_groups),
                failure_count=sum(group.failure_count for group in axis_groups),
                varied_parallelism_dimensions=[
                    dimension for dimension in varied_parallelism_dimensions if dimension in primary_dimensions
                ],
                co_varied_parallelism_dimensions=co_varied_parallelism_dimensions,
                confounded_workload_dimensions=confounded_workload_dimensions,
                confounded_runtime_dimensions=confounded_runtime_dimensions,
            )
        )
    return coverage


def _scenario_parallelism_boundary_prediction_support(
    *,
    parallelism_boundary_status: str,
    parallelism_boundary_fit_count: int,
    parallelism_boundary_failure_count: int,
    parallelism_boundary_measured_axis_names: list[str],
    parallelism_boundary_confounded_axis_names: list[str],
    parallelism_boundary_missing_axis_names: list[str],
    parallelism_boundary_confounded_dimensions: list[str],
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if parallelism_boundary_fit_count == 0:
        blockers.append("no_fit_rows")
    if parallelism_boundary_failure_count == 0:
        blockers.append("no_failure_rows")
    if parallelism_boundary_confounded_axis_names:
        blockers.append("confounded_parallelism_boundary_axes")
    if parallelism_boundary_missing_axis_names:
        blockers.append("missing_parallelism_boundary_axes")
    if parallelism_boundary_confounded_dimensions:
        blockers.append("confounded_boundary_dimensions")

    blockers = sorted(set(blockers))
    if parallelism_boundary_status == "insufficient_data":
        return "insufficient_parallelism_boundary_data", sorted({*blockers, "insufficient_data"})
    if parallelism_boundary_status == "no_measured_parallelism_variation":
        return "no_parallelism_boundary_variation", sorted({*blockers, "no_measured_parallelism_variation"})
    if parallelism_boundary_status == "no_fit_failure_parallelism_boundary":
        return "no_fit_failure_boundary_evidence", sorted({*blockers, "no_fit_failure_parallelism_boundary"})
    if parallelism_boundary_status == "confounded_parallelism_fit_failure_boundary":
        return "confounded_parallelism_boundary_prediction", sorted(
            {*blockers, "confounded_parallelism_fit_failure_boundary"}
        )
    if parallelism_boundary_status == "measured_parallelism_fit_failure_boundary":
        if not parallelism_boundary_measured_axis_names:
            return "parallelism_boundary_without_axis_coverage", sorted(
                {*blockers, "no_measured_parallelism_boundary_axes"}
            )
        if blockers:
            return "partial_parallelism_fit_failure_boundary", blockers
        return "validated_parallelism_fit_failure_boundary", []
    return "unknown_parallelism_boundary_prediction", sorted({*blockers, parallelism_boundary_status})


def _parallelism_tradeoff_status(
    *,
    unique_strategy_count: int,
    scored_strategy_count: int,
    promotable_strategy_count: int,
    requires_remeasurement_strategy_count: int,
) -> str:
    if unique_strategy_count == 0:
        return "no_candidates"
    if unique_strategy_count == 1:
        return "single_parallelism_strategy"
    if scored_strategy_count == 0:
        return "unscored_parallelism_tradeoff"
    if scored_strategy_count == 1:
        return "single_scored_strategy_with_blocked_alternatives"
    if requires_remeasurement_strategy_count:
        return "scored_parallelism_tradeoff_requires_remeasurement"
    if promotable_strategy_count >= 2:
        return "promotable_parallelism_tradeoff"
    if promotable_strategy_count == 1:
        return "single_promotable_strategy_with_scored_alternatives"
    return "scored_parallelism_tradeoff_not_promotable"


def _throughput_efficiency_tradeoff_status(
    *,
    best_raw: ScenarioCandidate | None,
    best_risk_adjusted: ScenarioCandidate | None,
    best_efficiency: ScenarioCandidate | None,
    best_risk_adjusted_efficiency: ScenarioCandidate | None,
) -> str:
    if best_raw is None and best_efficiency is None:
        return "no_scored_candidates"
    raw_differs = best_raw is not None and best_efficiency is not None and best_raw.label != best_efficiency.label
    risk_differs = (
        best_risk_adjusted is not None
        and best_risk_adjusted_efficiency is not None
        and best_risk_adjusted.label != best_risk_adjusted_efficiency.label
    )
    if raw_differs and risk_differs:
        return "raw_and_risk_adjusted_efficiency_diverge"
    if raw_differs:
        return "raw_throughput_efficiency_diverge"
    if risk_differs:
        return "risk_adjusted_throughput_efficiency_diverge"
    return "throughput_efficiency_aligned"


def _same_workload_scaling_status(scaling_candidates: list[ScenarioCandidate]) -> str:
    if not scaling_candidates:
        return "no_same_workload_scaling_comparison"
    min_efficiency = min(candidate.scaling_efficiency or 0.0 for candidate in scaling_candidates)
    if min_efficiency < 0.50:
        return "poor_same_workload_scaling"
    if min_efficiency < 0.80:
        return "sublinear_same_workload_scaling"
    return "strong_same_workload_scaling"


def _candidate_score_gap(
    best_candidate: ScenarioCandidate | None,
    fallback_candidate: ScenarioCandidate | None,
    score_attr: str,
) -> float | None:
    if best_candidate is None or fallback_candidate is None:
        return None
    best_score = getattr(best_candidate, score_attr)
    fallback_score = getattr(fallback_candidate, score_attr)
    if best_score is None or fallback_score is None:
        return None
    return round(max(float(best_score) - float(fallback_score), 0.0), 3)


def _candidate_score_gap_percentage(
    best_candidate: ScenarioCandidate | None,
    score_attr: str,
    gap: float | None,
) -> float | None:
    if best_candidate is None or gap is None:
        return None
    best_score = getattr(best_candidate, score_attr)
    if best_score is None or best_score <= 0:
        return None
    return round(gap / float(best_score) * 100.0, 3)


def _scenario_promotion_readiness_status(
    *,
    best_raw: ScenarioCandidate | None,
    best_risk_adjusted: ScenarioCandidate | None,
    best_promotable: ScenarioCandidate | None,
) -> str:
    selected = best_risk_adjusted or best_raw
    if selected is None:
        return "no_scored_candidate"
    if selected.promotable:
        if best_raw is not None and selected.label == best_raw.label:
            return "promote_raw_and_risk_adjusted_winner"
        return "promote_risk_adjusted_winner"
    if "requires_remeasurement" in selected.risk_flags or selected.prediction_confidence != "calibrated":
        return "remeasure_risk_adjusted_winner_before_promotion"
    if selected.recommendation == "debug_runtime_failure":
        return "debug_risk_adjusted_winner_before_promotion"
    if selected.recommendation == "correctness_gate_required":
        return "correctness_gate_risk_adjusted_winner_before_promotion"
    if selected.recommendation.startswith("remeasure"):
        return "remeasure_risk_adjusted_winner_before_promotion"
    if best_promotable is not None:
        return "promote_best_promotable_fallback"
    return "no_promotable_candidate"


def _exact_timing_support_status(candidate: ScenarioCandidate | None) -> str:
    if candidate is None:
        return "missing"
    if candidate.timing_coverage_status == "exact_phase_timing":
        return "exact_phase"
    if candidate.timing_coverage_status == "exact_total_step_only":
        return "exact_total_step_only"
    if candidate.timing_coverage_status == "no_timing_evidence":
        return "missing"
    return "reference_or_extrapolated"


def _parallelism_optimality_support(
    *,
    unique_strategy_count: int,
    scored_strategy_count: int,
    memory_blocked_count: int,
    best_risk_adjusted: ScenarioCandidate | None,
    best_promotable: ScenarioCandidate | None,
    risk_adjusted_interval_overlap_status: str,
    parallelism_tradeoff_status: str,
    parallelism_axis_coverage_status_counts: dict[str, int],
    varied_runtime_dimensions: list[str],
    simulator_support_status_counts: dict[str, int],
    interval_overlap_only_promotable_tie: bool = False,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if unique_strategy_count == 0:
        return "no_candidates", ["no_candidates"]
    if unique_strategy_count == 1:
        return "single_parallelism_strategy_no_tradeoff", ["single_parallelism_strategy"]
    if scored_strategy_count == 0:
        blocker = "memory_blocked_all_candidates" if memory_blocked_count else "no_scored_parallelism_candidates"
        return "unscored_parallelism_tradeoff", [blocker]
    if best_risk_adjusted is None:
        return "no_risk_adjusted_winner", ["no_risk_adjusted_winner"]

    if (
        best_risk_adjusted.prediction_confidence != "calibrated"
        or "requires_remeasurement" in best_risk_adjusted.risk_flags
    ):
        blockers.append("winner_requires_measurement")
    if parallelism_tradeoff_status == "scored_parallelism_tradeoff_requires_remeasurement":
        blockers.append("unmeasured_parallelism_alternatives")
    if not best_risk_adjusted.promotable:
        blockers.append("winner_not_promotable")
    if best_promotable is None:
        blockers.append("no_promotable_candidate")
    elif best_promotable.label != best_risk_adjusted.label:
        blockers.append("promotable_candidate_not_selected_winner")
    if risk_adjusted_interval_overlap_status == "overlapping_best_interval":
        if interval_overlap_only_promotable_tie:
            # A measured tie among K3-promotable strategies IS a resolved tradeoff ("either is
            # optimal"), not selection uncertainty: every overlapping contender and the winner are
            # promotable, so the choice cannot be wrong. Keep the tie visible without blocking.
            blockers.append("risk_adjusted_interval_tie_between_promotable_strategies")
        else:
            blockers.append("risk_adjusted_interval_overlap")
    elif risk_adjusted_interval_overlap_status in {"unknown", "no_scored_interval"}:
        blockers.append("missing_risk_adjusted_interval")
    if parallelism_axis_coverage_status_counts.get("scored_parallelism_axis", 0) == 0:
        blockers.append("no_clean_parallelism_axis_coverage")
    if any(status.startswith("confounded_") for status in parallelism_axis_coverage_status_counts):
        blockers.append("confounded_parallelism_axes")
    if varied_runtime_dimensions:
        blockers.append("runtime_variant_variation")
    if any(status.startswith("unsupported_") for status in simulator_support_status_counts):
        blockers.append("unsupported_simulator_surface")
    partial_surface_count = sum(
        count
        for status, count in simulator_support_status_counts.items()
        if status != "supported_local_non_pp" and not status.startswith("unsupported_")
    )
    if partial_surface_count:
        blockers.append("partial_simulator_surface_support")

    timing_status = _exact_timing_support_status(best_risk_adjusted)
    if timing_status == "missing":
        blockers.append("winner_missing_timing_evidence")
    elif timing_status == "reference_or_extrapolated":
        blockers.append("winner_timing_is_reference_or_extrapolated")
    elif timing_status == "exact_total_step_only":
        blockers.append("winner_missing_phase_timing")

    blockers = sorted(set(blockers))
    if "winner_requires_measurement" in blockers or "unmeasured_parallelism_alternatives" in blockers:
        return "requires_measurement_before_parallelism_optimality", blockers
    if "risk_adjusted_interval_overlap" in blockers:
        return "interval_overlap_parallelism_uncertain", blockers
    if (
        "no_clean_parallelism_axis_coverage" in blockers
        or "confounded_parallelism_axes" in blockers
        or "runtime_variant_variation" in blockers
    ):
        return "confounded_parallelism_winner", blockers
    if "unsupported_simulator_surface" in blockers:
        return "unsupported_surface_parallelism_winner", blockers
    if "partial_simulator_surface_support" in blockers:
        return "partial_surface_parallelism_winner", blockers
    if "winner_not_promotable" in blockers or "no_promotable_candidate" in blockers:
        return "not_promotable_parallelism_winner", blockers
    if "winner_timing_is_reference_or_extrapolated" in blockers or "winner_missing_timing_evidence" in blockers:
        return "timing_unsupported_parallelism_winner", blockers
    if "winner_missing_phase_timing" in blockers:
        return "total_step_only_parallelism_winner", blockers
    if parallelism_tradeoff_status == "promotable_parallelism_tradeoff":
        return "supported_promotable_parallelism_tradeoff", blockers
    return "supported_parallelism_winner", blockers


def _measurement_portfolio_sort_key(candidate: ScenarioCandidate) -> tuple[float, float, str]:
    risk_adjusted = (
        candidate.score_risk_adjusted_tokens_per_sec
        if candidate.score_risk_adjusted_tokens_per_sec is not None
        else float("-inf")
    )
    raw = candidate.score_tokens_per_sec if candidate.score_tokens_per_sec is not None else float("-inf")
    return risk_adjusted, raw, candidate.label


_MEASUREMENT_REASON_WEIGHTS = {
    "best_next_measurement": 4.0,
    "best_risk_adjusted_candidate": 1.5,
    "best_raw_candidate": 1.0,
    "best_promotable_candidate": 0.75,
    "best_gpu_efficiency_candidate": 1.0,
    "throughput_efficiency_frontier": 1.5,
    "risk_adjusted_efficiency_frontier": 1.25,
    "scored_parallelism_tradeoff_requires_remeasurement": 2.5,
    "promotable_parallelism_tradeoff": 1.5,
    "scored_parallelism_tradeoff_not_promotable": 2.0,
    "single_promotable_strategy_with_scored_alternatives": 1.25,
    "poor_same_workload_scaling": 2.5,
    "sublinear_same_workload_scaling": 1.75,
    "partial_simulator_surface_support": 2.0,
}

_PARALLELISM_EXTRAPOLATION_AXIS_BY_FLAG = {
    "parallelism_extrapolation:ep": "expert_parallel",
    "parallelism_extrapolation:ep_fsdp": "ep_fsdp",
    "parallelism_extrapolation:tp": "tensor_parallel",
    "parallelism_extrapolation:pp": "pipeline_parallel",
    "parallelism_extrapolation:ulysses": "ulysses",
    "parallelism_extrapolation:ring": "ringattn",
}

_MEASUREMENT_PARALLELISM_AXIS_GAP_STATUSES = {
    "single_scored_axis_with_blocked_alternatives",
    "blocked_parallelism_axis",
    "unscored_parallelism_axis",
    "confounded_single_scored_axis_with_blocked_alternatives",
    "confounded_blocked_parallelism_axis",
    "confounded_scored_parallelism_axis",
    "confounded_unscored_parallelism_axis",
}

_PHASE_TIMING_MEASUREMENT_CONFIG_OVERRIDES = (
    "train.enable_step_phase_timing=true",
    "train.enable_per_component_timing=true",
    "train.step_phase_timing_sync_cuda=true",
)

_MEMORY_PRESSURE_MEASUREMENT_CONFIG_OVERRIDES = (
    "train.enable_step_phase_timing=true",
    "train.enable_step_memory_profiling=true",
)

_FIT_BOUNDARY_MEASUREMENT_CONFIG_OVERRIDES = (
    "train.enable_step_phase_timing=true",
    "train.enable_step_memory_profiling=true",
    "train.enable_per_component_timing=true",
    "train.step_phase_timing_sync_cuda=true",
)

_PHASE_TIMING_SCENARIO_BLOCKERS = frozenset(
    {
        "missing_phase_timing",
        "missing_phase_bottleneck_evidence",
        "no_timing_evidence",
        "reference_or_extrapolated_timing",
    }
)

_PHASE_TIMING_DESIGN_MEASUREMENTS = frozenset(
    {
        "add_fit_failure_boundary_near_blocked_axes",
        "add_same_parallelism_runtime_workload_variants",
        "add_same_parallelism_workload_runtime_variants",
        "add_same_workload_same_runtime_axis_pairs",
        "add_same_workload_same_runtime_parallelism_axis_variants",
        "add_workload_and_parallelism_variants",
        "score_unscored_capture_candidates",
    }
)


def _reason_priority_weight(reason: str) -> float:
    if reason.startswith("cross_model_analog_support:"):
        return 2.5
    if reason.startswith("cross_model_prediction_interval_top:"):
        return 2.0
    if reason.startswith("parallelism_axis_gap:"):
        if ":confounded_" in reason:
            return 1.5
        if ":blocked_" in reason or ":single_scored_axis_with_blocked_" in reason:
            return 1.25
        if ":unscored_" in reason:
            return 1.0
        return 0.75
    if reason.startswith("phase_timing_gap:"):
        return 1.25
    if reason.startswith("memory_pressure_probe:"):
        return 2.0
    return _MEASUREMENT_REASON_WEIGHTS.get(reason, 0.5)


def _add_measurement_priority(
    factors: list[str],
    score: float,
    *,
    name: str,
    weight: float,
) -> float:
    factors.append(f"{name}={weight:.3f}")
    return score + weight


def _timing_coverage_measurement_weight(timing_coverage_status: str) -> float:
    if timing_coverage_status == "no_timing_evidence":
        return 1.50
    if timing_coverage_status.startswith("reference_") or timing_coverage_status.startswith("cross_model_reference_"):
        return 1.25
    if timing_coverage_status.startswith("step_time_fit_"):
        return 1.00
    if timing_coverage_status == "exact_total_step_only":
        return 0.50
    return 0.0


def _measurement_priority(
    candidate: ScenarioCandidate,
    reasons: set[str],
) -> tuple[float, float, int, list[str]]:
    is_memory_pressure_probe = any(reason.startswith("memory_pressure_probe:") for reason in reasons)
    if candidate.score_tokens_per_sec is None:
        score = 0.25
        factors = ["unscored_candidate=0.250"]
    else:
        score = 1.0
        factors = ["scored_candidate=1.000"]

    for reason in sorted(reasons):
        weight = _reason_priority_weight(reason)
        score = _add_measurement_priority(factors, score, name=f"reason:{reason}", weight=weight)

    if "requires_remeasurement" in candidate.risk_flags:
        score = _add_measurement_priority(factors, score, name="requires_remeasurement", weight=2.0)
    if candidate.prediction_confidence == "cross_model_extrapolated":
        score = _add_measurement_priority(factors, score, name="cross_model_extrapolated", weight=2.0)
    elif candidate.prediction_confidence != "calibrated":
        score = _add_measurement_priority(factors, score, name="extrapolated_prediction", weight=1.25)
    if "cross_model_analog" in candidate.risk_flags:
        score = _add_measurement_priority(factors, score, name="cross_model_analog", weight=1.5)
    if candidate.simulator_support_status != "supported_local_non_pp":
        if candidate.simulator_support_status.startswith("unsupported_"):
            score = _add_measurement_priority(factors, score, name="unsupported_simulator_surface", weight=3.0)
        else:
            score = _add_measurement_priority(factors, score, name="partial_simulator_surface", weight=2.0)

    timing_weight = _timing_coverage_measurement_weight(candidate.timing_coverage_status)
    if timing_weight:
        score = _add_measurement_priority(
            factors,
            score,
            name=f"timing_coverage:{candidate.timing_coverage_status}",
            weight=timing_weight,
        )

    if candidate.calibration_distance is not None:
        if is_memory_pressure_probe:
            closeness_weight = max(0.0, 2.0 - min(2.0, candidate.calibration_distance * 0.25))
            if closeness_weight > 0:
                score = _add_measurement_priority(
                    factors,
                    score,
                    name="calibration_closeness",
                    weight=closeness_weight,
                )
        elif candidate.calibration_distance > 0:
            distance_weight = min(2.0, candidate.calibration_distance * 0.25)
            score = _add_measurement_priority(
                factors,
                score,
                name="calibration_distance",
                weight=distance_weight,
            )

    if candidate.memory_coverage_status == "analytic_floor_only":
        score = _add_measurement_priority(factors, score, name="analytic_memory_floor_only", weight=1.75)
    elif candidate.memory_coverage_status.startswith("calibrated_overhead"):
        score = _add_measurement_priority(factors, score, name="calibrated_overhead_memory", weight=1.0)
    elif candidate.memory_coverage_status.startswith("extrapolated"):
        score = _add_measurement_priority(factors, score, name="extrapolated_memory_peak", weight=1.25)

    if candidate.estimated_memory_residual_fraction_of_peak is not None:
        residual_weight = min(1.5, candidate.estimated_memory_residual_fraction_of_peak * 1.5)
        if residual_weight > 0:
            score = _add_measurement_priority(factors, score, name="memory_residual_fraction", weight=residual_weight)

    if candidate.recommendation == "debug_runtime_failure":
        score = _add_measurement_priority(factors, score, name="debug_runtime_failure", weight=3.0)
    elif candidate.recommendation.startswith("remeasure"):
        score = _add_measurement_priority(factors, score, name="remeasure_recommendation", weight=2.0)
    elif candidate.recommendation == "correctness_gate_required":
        score = _add_measurement_priority(factors, score, name="correctness_gate_required", weight=1.25)

    if candidate.scaling_efficiency is not None and candidate.scaling_gpu_ratio is not None:
        if candidate.scaling_gpu_ratio > 1.0 and candidate.scaling_efficiency < 0.5:
            score = _add_measurement_priority(factors, score, name="poor_scaling_efficiency", weight=2.0)
        elif candidate.scaling_gpu_ratio > 1.0 and candidate.scaling_efficiency < 0.8:
            score = _add_measurement_priority(factors, score, name="sublinear_scaling_efficiency", weight=1.25)

    boundary_prefixes = (
        "allocator_pressure_boundary:",
        "communication_cross_node:",
        "observed_oom_boundary:",
        "runtime_mismatch:",
    )
    boundary_count = sum(1 for flag in candidate.risk_flags if flag.startswith(boundary_prefixes))
    if boundary_count:
        score = _add_measurement_priority(
            factors,
            score,
            name="boundary_or_runtime_mismatch_count",
            weight=min(2.0, boundary_count * 0.5),
        )

    bottleneck_count = sum(1 for flag in candidate.risk_flags if flag.endswith("_bottleneck"))
    if bottleneck_count:
        score = _add_measurement_priority(
            factors,
            score,
            name="phase_bottleneck_count",
            weight=min(1.5, bottleneck_count * 0.5),
        )

    cost_gpus = max(candidate.topology.world_size, 1)
    priority_score = round(score, 3)
    return priority_score, round(priority_score / cost_gpus, 3), cost_gpus, factors


def _allow_unscored_axis_gap_candidate(candidate: ScenarioCandidate) -> bool:
    if candidate.score_tokens_per_sec is not None:
        return True
    if candidate.simulator_support_status.startswith("unsupported_"):
        return False
    return candidate.memory_coverage_status != "analytic_floor_only"


def _candidate_axis_value_tuple(candidate: ScenarioCandidate, axis: str) -> tuple[Any, ...]:
    return tuple(getattr(candidate.topology, dimension) for dimension in _PARALLELISM_AXIS_PRIMARY_DIMENSIONS[axis])


def _candidate_contributes_axis_gap(
    candidate: ScenarioCandidate,
    *,
    axis: str,
    reference_candidate: ScenarioCandidate | None,
    candidates: list[ScenarioCandidate],
) -> bool:
    all_values = {_candidate_axis_value_tuple(row, axis) for row in candidates}
    if len(all_values) <= 1:
        return False
    if reference_candidate is None:
        return True
    return _candidate_axis_value_tuple(candidate, axis) != _candidate_axis_value_tuple(reference_candidate, axis)


def _synthesize_axis_gap_reason(
    candidate: ScenarioCandidate,
    *,
    cross_model_analog_support_status: str,
) -> bool:
    if candidate.score_tokens_per_sec is None:
        return _allow_unscored_axis_gap_candidate(candidate)
    return (
        candidate.prediction_confidence == "cross_model_extrapolated"
        and cross_model_analog_support_status == "single_reference_cannot_rank_parallelism_variants"
    )


def _phase_timing_gap_status(candidate: ScenarioCandidate) -> str | None:
    status = candidate.timing_coverage_status
    if status == "exact_phase_timing" or status.endswith("_phase_timing"):
        return None
    if status == "exact_total_step_only" or status.endswith("_total_step_only"):
        return status
    if status.startswith(("reference_", "cross_model_reference_")):
        return status
    return None


def _allow_memory_pressure_probe_candidate(candidate: ScenarioCandidate) -> bool:
    if candidate.score_tokens_per_sec is not None:
        return False
    if candidate.feasibility_status != "memory_floor_exceeds_safety_margin":
        return False
    if candidate.simulator_support_status != "supported_local_non_pp":
        return False
    return candidate.memory_coverage_status == "analytic_floor_only"


def _measurement_portfolio(
    *,
    candidates: list[ScenarioCandidate],
    throughput_efficiency_frontier_labels: list[str],
    risk_adjusted_efficiency_frontier_labels: list[str],
    parallelism_axis_coverage: list[ScenarioParallelismAxisCoverage],
    parallelism_tradeoff_status: str,
    cross_model_analog_support_status: str,
    cross_model_analog_prediction_interval_selectivity_status: str,
    cross_model_analog_prediction_interval_top_labels: list[str],
    same_workload_scaling_status: str,
    best_raw: ScenarioCandidate | None,
    best_risk_adjusted: ScenarioCandidate | None,
    best_efficiency: ScenarioCandidate | None,
    best_risk_adjusted_efficiency: ScenarioCandidate | None,
    best_next_measurement: ScenarioCandidate | None,
    best_promotable: ScenarioCandidate | None,
    max_candidates: int = 4,
) -> tuple[list[str], dict[str, list[str]], dict[str, float], dict[str, float], dict[str, int], dict[str, list[str]]]:
    by_label = {candidate.label: candidate for candidate in candidates}
    reasons: dict[str, set[str]] = {}

    def add(candidate: ScenarioCandidate | None, reason: str, *, allow_unscored: bool = False) -> None:
        if candidate is None:
            return
        if candidate.score_tokens_per_sec is None and not allow_unscored:
            return
        reasons.setdefault(candidate.label, set()).add(reason)

    def add_label(label: str, reason: str) -> None:
        add(by_label.get(label), reason)

    add(best_next_measurement, "best_next_measurement")
    add(best_risk_adjusted, "best_risk_adjusted_candidate")
    add(best_raw, "best_raw_candidate")
    if best_promotable is not None and best_next_measurement is None:
        add(best_promotable, "best_promotable_candidate")

    if len(throughput_efficiency_frontier_labels) > 1:
        for label in throughput_efficiency_frontier_labels:
            add_label(label, "throughput_efficiency_frontier")
    if len(risk_adjusted_efficiency_frontier_labels) > 1:
        for label in risk_adjusted_efficiency_frontier_labels:
            add_label(label, "risk_adjusted_efficiency_frontier")

    if parallelism_tradeoff_status in {
        "scored_parallelism_tradeoff_requires_remeasurement",
        "promotable_parallelism_tradeoff",
        "scored_parallelism_tradeoff_not_promotable",
        "single_promotable_strategy_with_scored_alternatives",
    }:
        for candidate in sorted(candidates, key=_measurement_portfolio_sort_key, reverse=True):
            if (
                parallelism_tradeoff_status == "scored_parallelism_tradeoff_not_promotable"
                and candidate.score_tokens_per_sec is not None
            ):
                add(candidate, parallelism_tradeoff_status)
            elif "requires_remeasurement" in candidate.risk_flags or candidate.promotable:
                add(candidate, parallelism_tradeoff_status)

    axis_status_by_name = {
        coverage.axis: coverage.status
        for coverage in parallelism_axis_coverage
        if coverage.status in _MEASUREMENT_PARALLELISM_AXIS_GAP_STATUSES
    }
    if axis_status_by_name:
        reference_candidate = best_risk_adjusted or best_raw
        for candidate in sorted(candidates, key=_measurement_portfolio_sort_key, reverse=True):
            for flag in candidate.risk_flags:
                axis = _PARALLELISM_EXTRAPOLATION_AXIS_BY_FLAG.get(flag)
                status = axis_status_by_name.get(axis) if axis is not None else None
                if status is not None:
                    add(
                        candidate,
                        f"parallelism_axis_gap:{axis}:{status}",
                        allow_unscored=_allow_unscored_axis_gap_candidate(candidate),
                    )
            for axis, status in axis_status_by_name.items():
                if not _synthesize_axis_gap_reason(
                    candidate,
                    cross_model_analog_support_status=cross_model_analog_support_status,
                ):
                    continue
                if _candidate_contributes_axis_gap(
                    candidate,
                    axis=axis,
                    reference_candidate=reference_candidate,
                    candidates=candidates,
                ):
                    add(
                        candidate,
                        f"parallelism_axis_gap:{axis}:{status}",
                        allow_unscored=_allow_unscored_axis_gap_candidate(candidate),
                    )

    if cross_model_analog_support_status not in {"not_used", "no_scored_cross_model_candidates"}:
        for candidate in sorted(
            _cross_model_analog_candidates(candidates), key=_measurement_portfolio_sort_key, reverse=True
        ):
            add(candidate, f"cross_model_analog_support:{cross_model_analog_support_status}")

    if cross_model_analog_prediction_interval_selectivity_status in {
        "partial_prediction_interval_top",
        "nonselective_prediction_interval_top",
    }:
        for label in cross_model_analog_prediction_interval_top_labels or []:
            add_label(
                label,
                f"cross_model_prediction_interval_top:{cross_model_analog_prediction_interval_selectivity_status}",
            )

    if same_workload_scaling_status in {"poor_same_workload_scaling", "sublinear_same_workload_scaling"}:
        for candidate in sorted(candidates, key=_measurement_portfolio_sort_key, reverse=True):
            if candidate.scaling_gpu_ratio is not None and candidate.scaling_gpu_ratio > 1.0:
                add(candidate, same_workload_scaling_status)

    for candidate in sorted(candidates, key=_measurement_portfolio_sort_key, reverse=True):
        if candidate.score_tokens_per_sec is None:
            continue
        phase_gap_status = _phase_timing_gap_status(candidate)
        if phase_gap_status is not None:
            add(candidate, f"phase_timing_gap:{phase_gap_status}")

    for candidate in sorted(candidates, key=_measurement_portfolio_sort_key, reverse=True):
        if _allow_memory_pressure_probe_candidate(candidate):
            add(candidate, f"memory_pressure_probe:{candidate.feasibility_status}", allow_unscored=True)

    for candidate in sorted(candidates, key=_measurement_portfolio_sort_key, reverse=True):
        if (
            candidate.simulator_support_status != "supported_local_non_pp"
            and not candidate.simulator_support_status.startswith("unsupported_")
        ):
            add(candidate, "partial_simulator_surface_support", allow_unscored=True)

    for candidate in [best_efficiency, best_risk_adjusted_efficiency]:
        add(candidate, "best_gpu_efficiency_candidate")

    priority_by_label: dict[str, tuple[float, float, int, list[str]]] = {
        label: _measurement_priority(by_label[label], candidate_reasons) for label, candidate_reasons in reasons.items()
    }
    ordered_labels = sorted(
        reasons,
        key=lambda label: (*priority_by_label[label][:2], *_measurement_portfolio_sort_key(by_label[label])),
        reverse=True,
    )
    categories_by_label = {
        label: sorted({_measurement_reason_category(reason) for reason in candidate_reasons})
        for label, candidate_reasons in reasons.items()
    }
    ordered_categories = sorted(
        {category for categories in categories_by_label.values() for category in categories},
        key=lambda category: (-_VALIDATION_ACTION_PRIORITY_BY_CATEGORY.get(category, 10), category),
    )
    selected_labels: list[str] = []
    for category in ordered_categories:
        if len(selected_labels) >= max_candidates:
            break
        label = next(
            (candidate_label for candidate_label in ordered_labels if category in categories_by_label[candidate_label]),
            None,
        )
        if label is not None and label not in selected_labels:
            selected_labels.append(label)
    for label in ordered_labels:
        if len(selected_labels) >= max_candidates:
            break
        if label not in selected_labels:
            selected_labels.append(label)
    return (
        selected_labels,
        {label: sorted(reasons[label]) for label in selected_labels},
        {label: priority_by_label[label][0] for label in selected_labels},
        {label: priority_by_label[label][1] for label in selected_labels},
        {label: priority_by_label[label][2] for label in selected_labels},
        {label: priority_by_label[label][3] for label in selected_labels},
    )


def _measurement_reason_category(reason: str) -> str:
    if reason == "best_next_measurement":
        return "best_next_measurement"
    if reason.startswith(("cross_model_analog_support:", "cross_model_prediction_interval_top:")):
        return "cross_model_analog"
    if reason.startswith("parallelism_axis_gap:"):
        return "parallelism_axis_gap"
    if reason.startswith("phase_timing_gap:"):
        return "phase_timing_gap"
    if reason.startswith("memory_pressure_probe:"):
        return "memory_pressure_probe"
    if reason in {
        "scored_parallelism_tradeoff_requires_remeasurement",
        "promotable_parallelism_tradeoff",
        "scored_parallelism_tradeoff_not_promotable",
        "single_promotable_strategy_with_scored_alternatives",
    }:
        return "parallelism_tradeoff"
    if reason in {
        "best_gpu_efficiency_candidate",
        "throughput_efficiency_frontier",
        "risk_adjusted_efficiency_frontier",
    }:
        return "efficiency_tradeoff"
    if reason in {"poor_same_workload_scaling", "sublinear_same_workload_scaling"}:
        return "same_workload_scaling"
    if reason == "partial_simulator_surface_support":
        return "simulator_surface"
    if reason in {"best_raw_candidate", "best_risk_adjusted_candidate", "best_promotable_candidate"}:
        return "winner_tracking"
    return "other"


def _measurement_portfolio_reason_category_counts(candidate_reasons: dict[str, list[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for reasons in candidate_reasons.values():
        for category in {_measurement_reason_category(reason) for reason in reasons}:
            counts[category] += 1
    return dict(sorted(counts.items()))


def _measurement_portfolio_axis_gap_names(candidate_reasons: dict[str, list[str]]) -> list[str]:
    axes = {
        parts[1]
        for reasons in candidate_reasons.values()
        for reason in reasons
        if reason.startswith("parallelism_axis_gap:")
        for parts in [reason.split(":")]
        if len(parts) >= 3
    }
    return sorted(axes)


def _measurement_portfolio_required_categories(
    *,
    best_next_measurement: ScenarioCandidate | None,
    parallelism_tradeoff_status: str,
    throughput_efficiency_tradeoff_status: str,
    same_workload_scaling_status: str,
    cross_model_analog_support_status: str,
    candidate_reasons: dict[str, list[str]],
) -> set[str]:
    required: set[str] = set()
    if best_next_measurement is not None:
        required.add("best_next_measurement")
    if parallelism_tradeoff_status in {
        "scored_parallelism_tradeoff_requires_remeasurement",
        "promotable_parallelism_tradeoff",
        "scored_parallelism_tradeoff_not_promotable",
        "single_promotable_strategy_with_scored_alternatives",
    }:
        required.add("parallelism_tradeoff")
    if throughput_efficiency_tradeoff_status not in {"throughput_efficiency_aligned", "no_scored_candidates"}:
        required.add("efficiency_tradeoff")
    if same_workload_scaling_status in {"poor_same_workload_scaling", "sublinear_same_workload_scaling"}:
        required.add("same_workload_scaling")
    if cross_model_analog_support_status not in {"not_used", "no_scored_cross_model_candidates"}:
        required.add("cross_model_analog")
    if _measurement_portfolio_axis_gap_names(candidate_reasons):
        required.add("parallelism_axis_gap")
    if any(any(reason.startswith("phase_timing_gap:") for reason in reasons) for reasons in candidate_reasons.values()):
        required.add("phase_timing_gap")
    if any(
        any(reason.startswith("memory_pressure_probe:") for reason in reasons) for reasons in candidate_reasons.values()
    ):
        required.add("memory_pressure_probe")
    if any("partial_simulator_surface_support" in reasons for reasons in candidate_reasons.values()):
        required.add("simulator_surface")
    return required


def _measurement_portfolio_coverage_status(
    *,
    candidate_reasons: dict[str, list[str]],
    best_next_measurement: ScenarioCandidate | None,
    parallelism_tradeoff_status: str,
    throughput_efficiency_tradeoff_status: str,
    same_workload_scaling_status: str,
    cross_model_analog_support_status: str,
) -> tuple[str, list[str], dict[str, int], list[str], int]:
    category_counts = _measurement_portfolio_reason_category_counts(candidate_reasons)
    axis_gap_names = _measurement_portfolio_axis_gap_names(candidate_reasons)
    required = _measurement_portfolio_required_categories(
        best_next_measurement=best_next_measurement,
        parallelism_tradeoff_status=parallelism_tradeoff_status,
        throughput_efficiency_tradeoff_status=throughput_efficiency_tradeoff_status,
        same_workload_scaling_status=same_workload_scaling_status,
        cross_model_analog_support_status=cross_model_analog_support_status,
        candidate_reasons=candidate_reasons,
    )
    present = set(category_counts)
    missing = sorted(required - present)
    blockers = [f"missing_measurement_category:{category}" for category in missing]
    cross_model_count = category_counts.get("cross_model_analog", 0)
    if not candidate_reasons and required:
        return "missing_required_measurement_coverage", blockers, category_counts, axis_gap_names, cross_model_count
    if missing:
        return "partial_required_measurement_gap_coverage", blockers, category_counts, axis_gap_names, cross_model_count
    if required:
        return "covers_required_measurement_gaps", [], category_counts, axis_gap_names, cross_model_count
    if candidate_reasons:
        return "opportunistic_measurement_portfolio", [], category_counts, axis_gap_names, cross_model_count
    return "no_measurement_portfolio_needed", [], category_counts, axis_gap_names, cross_model_count


_VALIDATION_ACTION_PRIORITY_BY_CATEGORY = {
    "memory_pressure_probe": 120,
    "parallelism_axis_gap": 110,
    "cross_model_analog": 105,
    "best_next_measurement": 100,
    "parallelism_tradeoff": 95,
    "phase_timing_gap": 90,
    "same_workload_scaling": 85,
    "efficiency_tradeoff": 80,
    "simulator_surface": 70,
    "winner_tracking": 60,
    "other": 10,
}


def _reason_matches_category(reason: str, category: str) -> bool:
    return _measurement_reason_category(reason) == category


def _reason_suffixes_for_category(reasons: set[str], category: str) -> list[str]:
    suffixes: set[str] = set()
    for reason in reasons:
        if not _reason_matches_category(reason, category):
            continue
        parts = reason.split(":")
        if category == "parallelism_axis_gap" and len(parts) >= 3:
            suffixes.add(parts[2])
        elif category in {"cross_model_analog", "phase_timing_gap", "memory_pressure_probe"} and len(parts) >= 2:
            suffixes.add(parts[1])
        else:
            suffixes.add(reason)
    return sorted(suffixes)


def _parallelism_axis_names_for_reasons(reasons: set[str]) -> list[str]:
    axes = {
        parts[1]
        for reason in reasons
        if reason.startswith("parallelism_axis_gap:")
        for parts in [reason.split(":")]
        if len(parts) >= 3
    }
    return sorted(axes)


def _validation_action_status(category: str, reason_statuses: list[str]) -> str:
    if category == "memory_pressure_probe":
        return "memory_pressure_probe_needed"
    if category == "parallelism_axis_gap":
        if any("confounded" in status and "blocked" in status for status in reason_statuses):
            return "confounded_parallelism_axis_boundary_action"
        if any("confounded" in status for status in reason_statuses):
            return "confounded_parallelism_axis_action"
        if any("blocked" in status for status in reason_statuses):
            return "blocked_parallelism_axis_action"
        if any("unscored" in status for status in reason_statuses):
            return "unscored_parallelism_axis_action"
        return "parallelism_axis_action"
    if category == "cross_model_analog":
        if any(status.startswith("nonselective_") or status.startswith("partial_") for status in reason_statuses):
            return "cross_model_interval_tiebreak_action"
        return "cross_model_analog_action"
    if category == "best_next_measurement":
        return "best_next_measurement_action"
    if category == "parallelism_tradeoff":
        return "parallelism_tradeoff_action"
    if category == "phase_timing_gap":
        return "phase_timing_instrumentation_action"
    if category == "same_workload_scaling":
        return "same_workload_scaling_action"
    if category == "efficiency_tradeoff":
        return "throughput_efficiency_tradeoff_action"
    if category == "simulator_surface":
        return "simulator_surface_action"
    if category == "winner_tracking":
        return "winner_tracking_action"
    return "measurement_followup_action"


def _validation_action_required_measurement(action_status: str) -> str:
    if action_status == "memory_pressure_probe_needed":
        return "launch_memory_pressure_fit_probe_with_memory_profiling"
    if action_status == "confounded_parallelism_axis_boundary_action":
        return "same_workload_same_runtime_axis_pair_with_fit_failure_boundary"
    if action_status == "confounded_parallelism_axis_action":
        return "same_workload_same_runtime_axis_pair"
    if action_status == "blocked_parallelism_axis_action":
        return "same_workload_axis_pair_with_fit_failure_boundary"
    if action_status == "unscored_parallelism_axis_action":
        return "score_unscored_parallelism_axis_candidate"
    if action_status == "parallelism_axis_action":
        return "add_same_workload_same_runtime_axis_pair"
    if action_status == "cross_model_interval_tiebreak_action":
        return "measure_target_cross_model_interval_top_candidates"
    if action_status == "cross_model_analog_action":
        return "measure_target_cross_model_analog_candidate"
    if action_status == "best_next_measurement_action":
        return "remeasure_best_next_candidate_same_workload"
    if action_status == "parallelism_tradeoff_action":
        return "measure_parallelism_tradeoff_candidates"
    if action_status == "phase_timing_instrumentation_action":
        return "rerun_with_phase_timing"
    if action_status == "same_workload_scaling_action":
        return "measure_same_workload_scaling_pair"
    if action_status == "throughput_efficiency_tradeoff_action":
        return "measure_throughput_efficiency_frontier_candidates"
    if action_status == "simulator_surface_action":
        return "add_simulator_surface_support_or_run_probe"
    if action_status == "winner_tracking_action":
        return "rerun_current_winner_with_required_instrumentation"
    return "inspect_measurement_candidate"


def _max_label_for_scores(labels: list[str], scores: dict[str, float]) -> str | None:
    return max(labels, key=lambda label: (scores.get(label, float("-inf")), label), default=None)


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _scenario_validation_actions(
    *,
    candidate_reasons: dict[str, list[str]],
    candidate_priority_scores: dict[str, float],
    candidate_priority_per_gpu: dict[str, float],
    candidate_cost_gpus: dict[str, int],
    candidate_config_overrides: dict[str, list[str]],
) -> list[ScenarioValidationAction]:
    labels_by_category: dict[str, list[str]] = {}
    reasons_by_category: dict[str, set[str]] = {}
    for label, reasons in candidate_reasons.items():
        for category in sorted({_measurement_reason_category(reason) for reason in reasons}):
            labels_by_category.setdefault(category, []).append(label)
            reasons_by_category.setdefault(category, set()).update(
                reason for reason in reasons if _reason_matches_category(reason, category)
            )

    actions: list[ScenarioValidationAction] = []
    for category, labels in labels_by_category.items():
        action_reasons = reasons_by_category.get(category, set())
        reason_statuses = _reason_suffixes_for_category(action_reasons, category)
        action_status = _validation_action_status(category, reason_statuses)
        required_measurement = _validation_action_required_measurement(action_status)
        score_label = _max_label_for_scores(labels, candidate_priority_scores)
        per_gpu_label = _max_label_for_scores(labels, candidate_priority_per_gpu)
        config_overrides = _unique_in_order(
            [override for label in labels for override in candidate_config_overrides.get(label, [])]
        )
        actions.append(
            ScenarioValidationAction(
                action_status=action_status,
                priority=_VALIDATION_ACTION_PRIORITY_BY_CATEGORY.get(category, 10),
                required_measurement=required_measurement,
                reason_category=category,
                candidate_count=len(labels),
                candidate_labels=labels,
                total_gpu_count=sum(candidate_cost_gpus.get(label, 0) for label in labels),
                max_priority_score=candidate_priority_scores.get(score_label) if score_label is not None else None,
                max_priority_label=score_label,
                max_priority_per_gpu=(
                    candidate_priority_per_gpu.get(per_gpu_label) if per_gpu_label is not None else None
                ),
                max_priority_per_gpu_label=per_gpu_label,
                parallelism_axis_names=(
                    _parallelism_axis_names_for_reasons(action_reasons) if category == "parallelism_axis_gap" else []
                ),
                reason_statuses=reason_statuses,
                config_overrides=config_overrides,
            )
        )
    return sorted(
        actions,
        key=lambda action: (
            -action.priority,
            action.reason_category,
            action.required_measurement,
            action.max_priority_label or "",
        ),
    )


def _append_unique(items: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        if value not in items:
            items.append(value)


def _measurement_candidate_config_overrides(candidate_reasons: dict[str, list[str]]) -> dict[str, list[str]]:
    overrides_by_label: dict[str, list[str]] = {}
    for label, reasons in candidate_reasons.items():
        overrides: list[str] = []
        if any(reason.startswith("phase_timing_gap:") for reason in reasons):
            _append_unique(overrides, _PHASE_TIMING_MEASUREMENT_CONFIG_OVERRIDES)
        if "partial_simulator_surface_support" in reasons:
            _append_unique(overrides, _PHASE_TIMING_MEASUREMENT_CONFIG_OVERRIDES)
        if any(reason.startswith("memory_pressure_probe:") for reason in reasons):
            _append_unique(overrides, _MEMORY_PRESSURE_MEASUREMENT_CONFIG_OVERRIDES)
        if any(
            reason.startswith("parallelism_axis_gap:") and "blocked" in reason.split(":", 2)[-1] for reason in reasons
        ):
            _append_unique(overrides, _FIT_BOUNDARY_MEASUREMENT_CONFIG_OVERRIDES)
        if overrides:
            overrides_by_label[label] = overrides
    return overrides_by_label


def _scenario_needs_phase_timing_design_overrides(report: ScenarioReport) -> bool:
    blockers = set(report.decision_summary.scenario_prediction_fidelity_blockers)
    if blockers & _PHASE_TIMING_SCENARIO_BLOCKERS:
        return True
    return any(
        any(reason.startswith("phase_timing_gap:") for reason in reasons)
        for reasons in report.decision_summary.measurement_candidate_reasons.values()
    )


def _measurement_design_config_overrides(
    report: ScenarioReport,
    required_measurement: str,
    *,
    base_overrides: tuple[str, ...] = (),
) -> tuple[str, ...]:
    overrides = list(base_overrides)
    if required_measurement in _PHASE_TIMING_DESIGN_MEASUREMENTS and _scenario_needs_phase_timing_design_overrides(
        report
    ):
        _append_unique(overrides, _PHASE_TIMING_MEASUREMENT_CONFIG_OVERRIDES)
    return tuple(overrides)


def _parse_config_override_value(raw_value: str) -> Any:
    value = yaml.safe_load(raw_value)
    if value is None and raw_value.strip().lower() not in {"null", "none", "~"}:
        return raw_value
    return value


def _apply_config_override(raw_config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"config override must use dotted.path=value syntax: {override!r}")
    dotted_path, raw_value = override.split("=", 1)
    path = [part.strip() for part in dotted_path.split(".") if part.strip()]
    if not path:
        raise ValueError(f"config override has no path: {override!r}")
    target = raw_config
    for part in path[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[path[-1]] = _parse_config_override_value(raw_value)


def _measurement_config_filename(index: int, label: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
    if not stem:
        stem = "candidate"
    return f"{index:02d}_{stem[:180]}.yaml"


def _set_sample_packing_sequence_len(raw_config: dict[str, Any], sequence_len: int) -> None:
    if requested_simulator_surface(raw_config) == "server_forward_backward":
        server = raw_config.get("server")
        target = server if isinstance(server, dict) and server else raw_config
    else:
        target = raw_config.setdefault("data", {})
        if not isinstance(target, dict):
            target = {}
            raw_config["data"] = target
    target["sample_packing_sequence_len"] = sequence_len


def _config_from_candidate(
    report: ScenarioReport,
    base_config: dict[str, Any],
    candidate: ScenarioCandidate,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    topology = candidate.topology
    raw_config = _mutated_config(
        base_config,
        world_size=topology.world_size,
        micro_batch_size=topology.micro_batch_size,
        gradient_accumulation_steps=topology.gradient_accumulation_steps,
        expert_parallel_size=topology.expert_parallel_size,
        tensor_parallel_size=topology.tensor_parallel_size,
        pipeline_parallel_size=topology.pipeline_parallel_size,
        ulysses_parallel_size=topology.ulysses_parallel_size,
        ringattn_parallel_size=topology.ringattn_parallel_size,
        data_parallel_replicate_size=topology.data_parallel_replicate_size,
        data_parallel_shard_size=topology.data_parallel_shard_size,
    )
    if topology.sample_packing_sequence_len is not None:
        _set_sample_packing_sequence_len(raw_config, topology.sample_packing_sequence_len)
    _set_balanced_routing(raw_config, report.balanced_routing)
    for override in overrides or []:
        _apply_config_override(raw_config, override)
    return raw_config


def materialize_measurement_candidate_configs(report: ScenarioReport) -> list[ScenarioMeasurementConfig]:
    """Render runnable YAML payloads for the report's bounded measurement portfolio."""
    base_config = load_training_config(report.base_config_path)
    candidates_by_label = {candidate.label: candidate for candidate in report.candidates}
    rendered: list[ScenarioMeasurementConfig] = []
    for index, label in enumerate(report.decision_summary.measurement_candidate_labels, start=1):
        candidate = candidates_by_label[label]
        raw_config = _config_from_candidate(
            report,
            base_config,
            candidate,
            overrides=report.decision_summary.measurement_candidate_config_overrides.get(label, []),
        )
        rendered.append(
            ScenarioMeasurementConfig(
                label=label,
                filename=_measurement_config_filename(index, label),
                config=raw_config,
            )
        )
    return rendered


def _design_anchor_candidate(report: ScenarioReport) -> ScenarioCandidate | None:
    return (
        report.best_risk_adjusted
        or report.best_raw
        or report.best_next_measurement
        or report.best_promotable
        or (report.candidates[0] if report.candidates else None)
    )


def _runtime_variant_anchor_candidate(
    candidates: list[ScenarioCandidate],
    runtime_mismatch_dimensions: list[str],
) -> ScenarioCandidate | None:
    if not runtime_mismatch_dimensions:
        return None
    mismatch_flags = {f"runtime_mismatch:{dimension}" for dimension in runtime_mismatch_dimensions}
    eligible = [candidate for candidate in candidates if mismatch_flags & set(candidate.risk_flags)]
    if not eligible:
        return None

    calibration_scope_rank = {
        "exact_calibrated": 0,
        "inside_measured_envelope": 1,
        "outside_measured_envelope": 2,
        "outside_sequence_calibration_envelope": 3,
        "cross_model_analog": 4,
        "no_calibration": 5,
    }

    def key(candidate: ScenarioCandidate) -> tuple[int, int, bool, float, float, int, str]:
        support_rank = 0 if candidate.simulator_support_status == "supported_local_non_pp" else 1
        scope_rank = calibration_scope_rank.get(candidate.calibration_scope, len(calibration_scope_rank))
        headroom = candidate.memory_headroom_gb if candidate.memory_headroom_gb is not None else float("-inf")
        peak = candidate.estimated_peak_mem_gb if candidate.estimated_peak_mem_gb is not None else float("inf")
        return (
            scope_rank,
            support_rank,
            _is_memory_blocked(candidate),
            -headroom,
            peak,
            candidate.topology.global_batch_size,
            candidate.label,
        )

    return min(eligible, key=key)


def _matched_behavior_point_for_candidate(
    report: ScenarioReport, candidate: ScenarioCandidate | None
) -> BenchmarkBehaviorPoint | None:
    if candidate is None:
        return None
    matched_labels = {part.strip() for part in (candidate.behavior.matched_label or "").split(",") if part.strip()}
    if not matched_labels:
        return None
    benchmark_dirs = []
    if report.benchmark_dir is not None:
        benchmark_dirs.append(Path(report.benchmark_dir))
    benchmark_dirs.extend(Path(path) for path in report.supplemental_benchmark_dirs)
    for benchmark_dir in benchmark_dirs:
        for point in load_benchmark_behavior_points(benchmark_dir):
            if point.label in matched_labels:
                return point
    return None


def _nearby_positive_values(value: int) -> list[int]:
    candidates = {value + 1, value * 2}
    if value > 1:
        candidates.add(value - 1)
    if value > 2:
        candidates.add(max(1, value // 2))
    return sorted(candidate for candidate in candidates if candidate > 0 and candidate != value)


def _nearby_sequence_lengths(sequence_len: int) -> list[int]:
    candidates = {min(sequence_len * 2, 131_072)}
    if sequence_len > 512:
        candidates.add(max(512, sequence_len // 2))
    return sorted(candidate for candidate in candidates if candidate > 0 and candidate != sequence_len)


def _workload_design_variants(anchor: Topology) -> list[tuple[str, dict[str, int]]]:
    variant_groups = [
        (
            "micro_batch_size",
            "mbs",
            [
                {
                    "micro_batch_size": value,
                    "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
                    "sample_packing_sequence_len": anchor.sample_packing_sequence_len,
                }
                for value in _nearby_positive_values(anchor.micro_batch_size)
            ],
        ),
        (
            "gradient_accumulation_steps",
            "ga",
            [
                {
                    "micro_batch_size": anchor.micro_batch_size,
                    "gradient_accumulation_steps": value,
                    "sample_packing_sequence_len": anchor.sample_packing_sequence_len,
                }
                for value in _nearby_positive_values(anchor.gradient_accumulation_steps)
            ],
        ),
        (
            "sample_packing_sequence_len",
            "seq",
            [
                {
                    "micro_batch_size": anchor.micro_batch_size,
                    "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
                    "sample_packing_sequence_len": value,
                }
                for value in _nearby_sequence_lengths(anchor.sample_packing_sequence_len)
            ],
        ),
    ]
    variants: list[tuple[str, dict[str, int]]] = []
    for varied_field, prefix, group in variant_groups:
        if group:
            variants.append((f"{prefix}{group[0][varied_field]}", group[0]))
    for varied_field, prefix, group in variant_groups:
        for item in group[1:]:
            variants.append((f"{prefix}{item[varied_field]}", item))
    return variants


def _nested_config_value(raw_config: dict[str, Any], path: tuple[str, str], default: Any = None) -> Any:
    section = raw_config.get(path[0])
    if not isinstance(section, dict):
        return default
    return section.get(path[1], default)


def _runtime_variant_design_value(
    *,
    base_config: dict[str, Any],
    dimension: str,
    reference_point: BenchmarkBehaviorPoint | None,
) -> Any:
    if reference_point is not None:
        reference_value = getattr(reference_point, dimension, None)
        if reference_value is not None:
            return reference_value
    if dimension not in _RUNTIME_VARIANT_BOOL_DEFAULTS:
        return None
    path = _RUNTIME_VARIANT_CONFIG_PATHS.get(dimension)
    if path is None:
        return None
    current_value = _nested_config_value(base_config, path, _RUNTIME_VARIANT_BOOL_DEFAULTS[dimension])
    return not bool(current_value)


def _runtime_variant_label_value(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower()).strip("._-") or "value"


def _runtime_design_variants(
    base_config: dict[str, Any],
    runtime_mismatch_dimensions: list[str],
    *,
    reference_point: BenchmarkBehaviorPoint | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    variants: list[tuple[str, tuple[str, ...]]] = []
    for dimension in runtime_mismatch_dimensions:
        path = _RUNTIME_VARIANT_CONFIG_PATHS.get(dimension)
        if path is None:
            continue
        variant_value = _runtime_variant_design_value(
            base_config=base_config,
            dimension=dimension,
            reference_point=reference_point,
        )
        if variant_value is None:
            continue
        override_value = str(variant_value).lower() if isinstance(variant_value, bool) else str(variant_value)
        variants.append(
            (
                f"runtime_{dimension}_{_runtime_variant_label_value(override_value)}",
                (f"{path[0]}.{path[1]}={override_value}",),
            )
        )
    return variants


def _design_config_from_topology(
    *,
    report: ScenarioReport,
    base_config: dict[str, Any],
    required_measurement: str,
    design_kind: str,
    index: int,
    world_size: int,
    local_world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    sample_packing_sequence_len: int,
    expert_parallel_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    ulysses_parallel_size: int,
    ringattn_parallel_size: int,
    data_parallel_replicate_size: int | None = None,
    data_parallel_shard_size: int | None = None,
    config_overrides: tuple[str, ...] = (),
) -> ScenarioMeasurementConfig | None:
    try:
        raw_config = _mutated_config(
            base_config,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            expert_parallel_size=expert_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            ulysses_parallel_size=ulysses_parallel_size,
            ringattn_parallel_size=ringattn_parallel_size,
            data_parallel_replicate_size=data_parallel_replicate_size,
            data_parallel_shard_size=data_parallel_shard_size,
        )
        _set_sample_packing_sequence_len(raw_config, sample_packing_sequence_len)
        _set_balanced_routing(raw_config, report.balanced_routing)
        for override in config_overrides:
            _apply_config_override(raw_config, override)
        topology = resolve_topology(raw_config, world_size=world_size, local_world_size=local_world_size)
    except (TypeError, ValueError):
        return None
    if topology.ep_fsdp_size is None:
        return None

    label = f"design:{required_measurement}:{design_kind}:{_topology_label(topology)}"
    return ScenarioMeasurementConfig(
        label=label,
        filename=_measurement_config_filename(index, label),
        config=raw_config,
    )


def _memory_pressure_design_variants(anchor: Topology) -> list[tuple[str, dict[str, int]]]:
    variants: list[tuple[str, dict[str, int]]] = []
    reduced_mbs = max(1, anchor.micro_batch_size // 2)
    if reduced_mbs == anchor.micro_batch_size and anchor.micro_batch_size > 1:
        reduced_mbs = anchor.micro_batch_size - 1
    reduced_seq = None
    if anchor.sample_packing_sequence_len and anchor.sample_packing_sequence_len > 512:
        reduced_seq = max(512, anchor.sample_packing_sequence_len // 2)

    if reduced_mbs != anchor.micro_batch_size:
        variants.append(
            (
                f"fit_mbs{reduced_mbs}",
                {
                    "micro_batch_size": reduced_mbs,
                    "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
                    "sample_packing_sequence_len": anchor.sample_packing_sequence_len,
                },
            )
        )
    if reduced_seq is not None and reduced_seq != anchor.sample_packing_sequence_len:
        variants.append(
            (
                f"fit_seq{reduced_seq}",
                {
                    "micro_batch_size": anchor.micro_batch_size,
                    "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
                    "sample_packing_sequence_len": reduced_seq,
                },
            )
        )
    if (
        reduced_mbs != anchor.micro_batch_size
        and reduced_seq is not None
        and reduced_seq != anchor.sample_packing_sequence_len
    ):
        variants.append(
            (
                f"fit_mbs{reduced_mbs}_seq{reduced_seq}",
                {
                    "micro_batch_size": reduced_mbs,
                    "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
                    "sample_packing_sequence_len": reduced_seq,
                },
            )
        )
    if not variants and anchor.sample_packing_sequence_len is not None:
        variants.append(
            (
                "profile_current_shape",
                {
                    "micro_batch_size": anchor.micro_batch_size,
                    "gradient_accumulation_steps": anchor.gradient_accumulation_steps,
                    "sample_packing_sequence_len": anchor.sample_packing_sequence_len,
                },
            )
        )
    return variants


def _append_design_config(
    rendered: list[ScenarioMeasurementConfig],
    seen: set[tuple[str, str]],
    design: ScenarioMeasurementConfig | None,
) -> None:
    if design is None:
        return
    required_measurement = design.label.split(":", 3)[1] if ":" in design.label else design.label
    key = (required_measurement, yaml.safe_dump(design.config, sort_keys=True))
    if key in seen:
        return
    seen.add(key)
    rendered.append(design)


def _design_kind_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return slug[:80] or "candidate"


def _fit_boundary_candidate_design_kind(design_kind: str, candidate: ScenarioCandidate) -> str:
    if (
        design_kind == "fit_boundary"
        and candidate.topology.ulysses_parallel_size > 1
        and re.search(r"(?:^|[-_:])u(?:[2-9]|\d{2,})(?:$|[-_:])", candidate.label)
    ):
        return "fit_boundary_ulysses"
    return design_kind


def _action_candidate_axis_family(action: ScenarioValidationAction, candidate: ScenarioCandidate) -> str | None:
    action_axes = set(action.parallelism_axis_names)
    topology = candidate.topology
    if "ulysses" in action_axes and topology.ulysses_parallel_size > 1:
        return "ulysses"
    if "ringattn" in action_axes and topology.ringattn_parallel_size > 1:
        return "ring"
    if {"expert_parallel", "ep_fsdp"} & action_axes:
        return "ep"
    if "tensor_parallel" in action_axes and topology.tensor_parallel_size > 1:
        return "tp"
    if "pipeline_parallel" in action_axes and topology.pipeline_parallel_size > 1:
        return "pp"
    if "dp_replicate" in action_axes and topology.data_parallel_replicate_size > 1:
        return "dp_replicate"
    if {"world_size", "dp_shard"} & action_axes:
        return "world"
    return None


def _action_candidate_design_kind(
    design_kind: str,
    action: ScenarioValidationAction,
    candidate: ScenarioCandidate,
) -> str:
    axis_family = _action_candidate_axis_family(action, candidate)
    if axis_family is None:
        return design_kind
    return f"{design_kind}_{axis_family}"


_GAP_ACTION_DESIGN_MEASUREMENTS = {
    "add_fit_failure_boundary_near_blocked_axes": (
        ("same_workload_same_runtime_axis_pair_with_fit_failure_boundary", "fit_boundary"),
        ("launch_memory_pressure_fit_probe_with_memory_profiling", "memory_fit_probe"),
    ),
    "add_memory_pressure_fit_probe_or_reduce_batch": (
        ("launch_memory_pressure_fit_probe_with_memory_profiling", "memory_fit_probe"),
    ),
    "add_partial_surface_support_or_direct_probe": (("add_simulator_surface_support_or_run_probe", "surface_probe"),),
}


_DIRECT_ACTION_DESIGN_MEASUREMENTS = {
    "add_simulator_surface_support_or_run_probe": "action_surface_probe",
    "launch_memory_pressure_fit_probe_with_memory_profiling": "action_memory_fit_probe",
    "same_workload_same_runtime_axis_pair": "action_axis_pair",
    "same_workload_same_runtime_axis_pair_with_fit_failure_boundary": "action_boundary_axis_pair",
    "measure_parallelism_tradeoff_candidates": "action_parallelism_tradeoff",
    "measure_target_cross_model_interval_top_candidates": "action_cross_model_interval",
    "measure_throughput_efficiency_frontier_candidates": "action_efficiency_frontier",
    "remeasure_best_next_candidate_same_workload": "action_remeasure_best",
    "rerun_current_winner_with_required_instrumentation": "action_winner_instrumented",
    "rerun_with_phase_timing": "action_phase_timing",
}


def _action_backed_design_config(
    *,
    report: ScenarioReport,
    base_config: dict[str, Any],
    required_measurement: str,
    design_kind: str,
    index: int,
    candidate: ScenarioCandidate,
) -> ScenarioMeasurementConfig:
    overrides = list(report.decision_summary.measurement_candidate_config_overrides.get(candidate.label, []))
    if required_measurement == "add_fit_failure_boundary_near_blocked_axes":
        _append_unique(overrides, _MEMORY_PRESSURE_MEASUREMENT_CONFIG_OVERRIDES)
    overrides = list(
        _measurement_design_config_overrides(
            report,
            required_measurement,
            base_overrides=tuple(overrides),
        )
    )
    raw_config = _config_from_candidate(
        report,
        base_config,
        candidate,
        overrides=overrides,
    )
    topology = candidate.topology
    design_kind = _fit_boundary_candidate_design_kind(design_kind, candidate)
    label = (
        f"design:{required_measurement}:{design_kind}_{index:02d}_{_design_kind_slug(candidate.label)}:"
        f"{_topology_label(topology)}"
    )
    return ScenarioMeasurementConfig(
        label=label,
        filename=_measurement_config_filename(index, label),
        config=raw_config,
    )


def _candidate_replay_design_config(
    *,
    report: ScenarioReport,
    base_config: dict[str, Any],
    required_measurement: str,
    index: int,
    candidate: ScenarioCandidate,
) -> ScenarioMeasurementConfig:
    overrides = _measurement_design_config_overrides(
        report,
        required_measurement,
        base_overrides=tuple(report.decision_summary.measurement_candidate_config_overrides.get(candidate.label, [])),
    )
    raw_config = _config_from_candidate(
        report,
        base_config,
        candidate,
        overrides=list(overrides),
    )
    topology = candidate.topology
    label = (
        f"design:{required_measurement}:scored_replay_{index:02d}_{_design_kind_slug(candidate.label)}:"
        f"{_topology_label(topology)}"
    )
    return ScenarioMeasurementConfig(
        label=label,
        filename=_measurement_config_filename(index, label),
        config=raw_config,
    )


def _same_workload_ga_for_topology(
    *,
    target_global_batch_size: int,
    world_size: int,
    micro_batch_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    ulysses_parallel_size: int,
    ringattn_parallel_size: int,
) -> int | None:
    non_dp = tensor_parallel_size * pipeline_parallel_size * ulysses_parallel_size * ringattn_parallel_size
    if non_dp <= 0 or world_size % non_dp:
        return None
    data_parallel_size = world_size // non_dp
    divisor = micro_batch_size * data_parallel_size
    if divisor <= 0 or target_global_batch_size % divisor:
        return None
    return target_global_batch_size // divisor


def _valid_dp_split_for_size(
    data_parallel_size: int,
    *,
    preferred_replicate_size: int | None = None,
    preferred_shard_size: int | None = None,
) -> tuple[int, int]:
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    if (
        preferred_replicate_size is not None
        and preferred_shard_size is not None
        and preferred_replicate_size > 0
        and preferred_shard_size > 0
        and preferred_replicate_size * preferred_shard_size == data_parallel_size
    ):
        return preferred_replicate_size, preferred_shard_size
    if (
        preferred_replicate_size is not None
        and preferred_replicate_size > 0
        and data_parallel_size % preferred_replicate_size == 0
    ):
        return preferred_replicate_size, data_parallel_size // preferred_replicate_size
    if preferred_shard_size is not None and preferred_shard_size > 0 and data_parallel_size % preferred_shard_size == 0:
        return data_parallel_size // preferred_shard_size, preferred_shard_size
    return 1, data_parallel_size


def _topology_values_with_dp_split(
    values: dict[str, int],
    *,
    preferred_replicate_size: int,
    preferred_shard_size: int,
) -> dict[str, int] | None:
    non_dp_size = (
        values["tensor_parallel_size"]
        * values["pipeline_parallel_size"]
        * values["ulysses_parallel_size"]
        * values["ringattn_parallel_size"]
    )
    if non_dp_size <= 0 or values["world_size"] % non_dp_size:
        return None
    data_parallel_size = values["world_size"] // non_dp_size
    replicate_size, shard_size = _valid_dp_split_for_size(
        data_parallel_size,
        preferred_replicate_size=preferred_replicate_size,
        preferred_shard_size=preferred_shard_size,
    )
    return {
        **values,
        "data_parallel_replicate_size": replicate_size,
        "data_parallel_shard_size": shard_size,
    }


def _dp_replicate_design_values(anchor: Topology, base: dict[str, int]) -> list[dict[str, int]]:
    variants: list[dict[str, int]] = []
    candidates = [1 if anchor.data_parallel_replicate_size > 1 else 2]
    candidates.extend([2, 4, 8, anchor.node_count, anchor.data_parallel_size])
    for replicate_size in candidates:
        if replicate_size <= 0 or replicate_size == anchor.data_parallel_replicate_size:
            continue
        if anchor.data_parallel_size % replicate_size:
            continue
        shard_size = anchor.data_parallel_size // replicate_size
        updated = {
            **base,
            "data_parallel_replicate_size": replicate_size,
            "data_parallel_shard_size": shard_size,
        }
        if updated not in variants:
            variants.append(updated)
    return variants


def _parallelism_design_values(
    anchor: Topology,
    metadata: ModelMetadata,
) -> list[tuple[str, dict[str, int]]]:
    base_axes = {
        "world_size": anchor.world_size,
        "expert_parallel_size": anchor.expert_parallel_size,
        "tensor_parallel_size": anchor.tensor_parallel_size,
        "pipeline_parallel_size": anchor.pipeline_parallel_size,
        "ulysses_parallel_size": anchor.ulysses_parallel_size,
        "ringattn_parallel_size": anchor.ringattn_parallel_size,
    }
    base = _topology_values_with_dp_split(
        base_axes,
        preferred_replicate_size=anchor.data_parallel_replicate_size,
        preferred_shard_size=anchor.data_parallel_shard_size,
    )
    if base is None:
        return []
    variants: list[tuple[str, dict[str, int]]] = []

    world_size_values = [anchor.world_size * 2]
    if anchor.world_size > anchor.local_world_size and anchor.world_size % 2 == 0:
        world_size_values.append(anchor.world_size // 2)
    for value in world_size_values:
        if value > 0 and value != anchor.world_size:
            updated = _topology_values_with_dp_split(
                {**base_axes, "world_size": value},
                preferred_replicate_size=anchor.data_parallel_replicate_size,
                preferred_shard_size=anchor.data_parallel_shard_size,
            )
            if updated is not None:
                variants.append(("world", updated))

    for updated in _dp_replicate_design_values(anchor, base):
        variants.append(("dp_replicate", updated))

    for field, axis, values in (
        ("expert_parallel_size", "ep", _auto_ep_sizes(anchor)),
        ("tensor_parallel_size", "tp", _auto_tensor_parallel_sizes(anchor, metadata)),
        ("pipeline_parallel_size", "pp", _auto_pipeline_parallel_sizes(anchor, metadata)),
        ("ulysses_parallel_size", "ulysses", _auto_ulysses_parallel_sizes(anchor, metadata)),
        ("ringattn_parallel_size", "ring", _auto_ringattn_parallel_sizes(anchor)),
    ):
        for value in values:
            if value == base_axes[field]:
                continue
            updated = _topology_values_with_dp_split(
                {**base_axes, field: value},
                preferred_replicate_size=anchor.data_parallel_replicate_size,
                preferred_shard_size=anchor.data_parallel_shard_size,
            )
            if updated is not None:
                variants.append((axis, updated))
    return variants


_SCENARIO_AXIS_DESIGN_PREFERRED_VARIANTS = {
    "world_size": ("world",),
    "dp_replicate": ("dp_replicate", "world"),
    "dp_shard": ("world", "ulysses", "ring"),
    "tensor_parallel": ("tp",),
    "pipeline_parallel": ("pp",),
    "expert_parallel": ("ep",),
    "ep_fsdp": ("ep", "world"),
    "ulysses": ("ulysses",),
    "ringattn": ("ring",),
}

_SCENARIO_PARALLELISM_AXIS_DESIGN_MEASUREMENTS = {
    "add_same_workload_same_runtime_parallelism_axis_variants",
    "add_same_workload_same_runtime_axis_pairs",
}

_SCENARIO_PARALLELISM_AXIS_COMPATIBLE_MEASUREMENTS = {
    *_SCENARIO_PARALLELISM_AXIS_DESIGN_MEASUREMENTS,
    "add_workload_and_parallelism_variants",
}

_SCENARIO_PARALLELISM_AXIS_CONFIG_BUDGET = 6
_SCENARIO_FIT_BOUNDARY_AXIS_CONFIG_BUDGET = 8
_SCENARIO_COMBINED_WORKLOAD_PARALLELISM_CONFIG_BUDGET = 10


def _scenario_gap_parallelism_axes(gap: ScenarioCaptureGap) -> list[str]:
    return _unique_in_order(
        [
            *gap.missing_parallelism_axis_names,
            *gap.blocked_parallelism_axis_names,
            *gap.confounded_parallelism_axis_names,
            *gap.unscored_parallelism_axis_names,
        ]
    )


def _scenario_gap_preferred_design_axes(gap: ScenarioCaptureGap) -> list[str]:
    axis_order: list[str] = []
    for gap_axis in _scenario_gap_parallelism_axes(gap):
        for axis in _SCENARIO_AXIS_DESIGN_PREFERRED_VARIANTS.get(gap_axis, ()):
            if axis not in axis_order:
                axis_order.append(axis)
    return axis_order


def _scenario_gap_design_config_budget(gap: ScenarioCaptureGap, default_budget: int) -> int:
    axis_family_budget = max(_SCENARIO_PARALLELISM_AXIS_CONFIG_BUDGET, len(_scenario_gap_preferred_design_axes(gap)))
    if gap.required_measurement == "add_workload_and_parallelism_variants":
        return max(
            default_budget,
            _SCENARIO_COMBINED_WORKLOAD_PARALLELISM_CONFIG_BUDGET,
            default_budget + axis_family_budget,
        )
    if gap.required_measurement == "add_fit_failure_boundary_near_blocked_axes":
        return max(default_budget, _SCENARIO_FIT_BOUNDARY_AXIS_CONFIG_BUDGET, default_budget + axis_family_budget)
    if gap.required_measurement in _SCENARIO_PARALLELISM_AXIS_DESIGN_MEASUREMENTS:
        return max(default_budget, axis_family_budget)
    return default_budget


def _parallelism_design_values_apply_to_workload(
    axis: str,
    values: dict[str, int],
    workload_values: dict[str, int],
) -> bool:
    seq_len = workload_values["sample_packing_sequence_len"] or 0
    if axis == "ulysses" and values["ulysses_parallel_size"] > 1 and seq_len < _MIN_ULYSSES_SEQUENCE_LEN:
        return False
    if axis == "ring" and values["ringattn_parallel_size"] > 1 and seq_len < _MIN_RINGATTN_SEQUENCE_LEN:
        return False
    return True


def _prioritized_parallelism_design_values_for_gap(
    anchor: Topology,
    metadata: ModelMetadata,
    gap: ScenarioCaptureGap,
) -> list[tuple[str, dict[str, int]]]:
    values = _parallelism_design_values(anchor, metadata)
    grouped: dict[str, list[dict[str, int]]] = {}
    for axis, design_values in values:
        grouped.setdefault(axis, []).append(design_values)

    axis_order: list[str] = []
    for axis in _scenario_gap_preferred_design_axes(gap):
        if axis in grouped and axis not in axis_order:
            axis_order.append(axis)
    for axis in grouped:
        if axis not in axis_order:
            axis_order.append(axis)

    ordered: list[tuple[str, dict[str, int]]] = []
    max_group_size = max((len(grouped[axis]) for axis in axis_order), default=0)
    for offset in range(max_group_size):
        for axis in axis_order:
            variants = grouped[axis]
            if offset < len(variants):
                ordered.append((axis, variants[offset]))
    return ordered


def materialize_measurement_design_configs(
    report: ScenarioReport,
    *,
    max_configs_per_measurement: int = 4,
) -> list[ScenarioMeasurementConfig]:
    """Render bounded YAML design rows for scenario-capture gaps not backed by existing candidates."""
    anchor_candidate = _design_anchor_candidate(report)
    if anchor_candidate is None:
        return []

    base_config = load_training_config(report.base_config_path)
    anchor = anchor_candidate.topology
    metadata = resolve_model_metadata(base_config)
    candidates_by_label = {candidate.label: candidate for candidate in report.candidates}
    validation_actions_by_measurement: dict[str, list[ScenarioValidationAction]] = {}
    for action in report.decision_summary.validation_actions:
        validation_actions_by_measurement.setdefault(action.required_measurement, []).append(action)
    rendered: list[ScenarioMeasurementConfig] = []
    seen: set[tuple[str, str]] = set()
    index = 1

    def add_design(design: ScenarioMeasurementConfig | None) -> None:
        nonlocal index
        before = len(rendered)
        _append_design_config(rendered, seen, design)
        if len(rendered) > before:
            index += 1

    for gap in sorted(
        report.decision_summary.scenario_capture_gaps, key=lambda item: (-item.priority, item.gap_status)
    ):
        count_for_measurement = 0
        design_config_budget = _scenario_gap_design_config_budget(gap, max_configs_per_measurement)

        def add_limited(design: ScenarioMeasurementConfig | None) -> bool:
            nonlocal count_for_measurement
            if count_for_measurement >= design_config_budget:
                return False
            before = len(rendered)
            add_design(design)
            if len(rendered) > before:
                count_for_measurement += 1
                return True
            return False

        for action_measurement, design_kind in _GAP_ACTION_DESIGN_MEASUREMENTS.get(gap.required_measurement, ()):
            for action in validation_actions_by_measurement.get(action_measurement, []):
                for label in action.candidate_labels:
                    if count_for_measurement >= max_configs_per_measurement:
                        break
                    candidate = candidates_by_label.get(label)
                    if candidate is None:
                        continue
                    add_limited(
                        _action_backed_design_config(
                            report=report,
                            base_config=base_config,
                            required_measurement=gap.required_measurement,
                            design_kind=design_kind,
                            index=index,
                            candidate=candidate,
                        )
                    )

        if gap.required_measurement == "add_fit_failure_boundary_near_blocked_axes":
            added_fit_boundary_axis_families: set[str] = set()
            for axis, values in _prioritized_parallelism_design_values_for_gap(anchor, metadata, gap):
                if axis in added_fit_boundary_axis_families:
                    continue
                gradient_accumulation_steps = _same_workload_ga_for_topology(
                    target_global_batch_size=anchor.global_batch_size,
                    world_size=values["world_size"],
                    micro_batch_size=anchor.micro_batch_size,
                    tensor_parallel_size=values["tensor_parallel_size"],
                    pipeline_parallel_size=values["pipeline_parallel_size"],
                    ulysses_parallel_size=values["ulysses_parallel_size"],
                    ringattn_parallel_size=values["ringattn_parallel_size"],
                )
                if gradient_accumulation_steps is None:
                    continue
                if add_limited(
                    _design_config_from_topology(
                        report=report,
                        base_config=base_config,
                        required_measurement=gap.required_measurement,
                        design_kind=f"fit_boundary_{axis}",
                        index=index,
                        world_size=values["world_size"],
                        local_world_size=min(anchor.local_world_size, values["world_size"]),
                        micro_batch_size=anchor.micro_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        sample_packing_sequence_len=anchor.sample_packing_sequence_len,
                        expert_parallel_size=values["expert_parallel_size"],
                        tensor_parallel_size=values["tensor_parallel_size"],
                        pipeline_parallel_size=values["pipeline_parallel_size"],
                        ulysses_parallel_size=values["ulysses_parallel_size"],
                        ringattn_parallel_size=values["ringattn_parallel_size"],
                        data_parallel_replicate_size=values["data_parallel_replicate_size"],
                        data_parallel_shard_size=values["data_parallel_shard_size"],
                        config_overrides=_measurement_design_config_overrides(
                            report,
                            gap.required_measurement,
                            base_overrides=_MEMORY_PRESSURE_MEASUREMENT_CONFIG_OVERRIDES,
                        ),
                    )
                ):
                    added_fit_boundary_axis_families.add(axis)

        if gap.required_measurement == "add_memory_pressure_fit_probe_or_reduce_batch" and count_for_measurement == 0:
            for design_kind, values in _memory_pressure_design_variants(anchor):
                add_limited(
                    _design_config_from_topology(
                        report=report,
                        base_config=base_config,
                        required_measurement=gap.required_measurement,
                        design_kind=design_kind,
                        index=index,
                        world_size=anchor.world_size,
                        local_world_size=anchor.local_world_size,
                        micro_batch_size=values["micro_batch_size"],
                        gradient_accumulation_steps=values["gradient_accumulation_steps"],
                        sample_packing_sequence_len=values["sample_packing_sequence_len"],
                        expert_parallel_size=anchor.expert_parallel_size,
                        tensor_parallel_size=anchor.tensor_parallel_size,
                        pipeline_parallel_size=anchor.pipeline_parallel_size,
                        ulysses_parallel_size=anchor.ulysses_parallel_size,
                        ringattn_parallel_size=anchor.ringattn_parallel_size,
                        data_parallel_replicate_size=anchor.data_parallel_replicate_size,
                        data_parallel_shard_size=anchor.data_parallel_shard_size,
                        config_overrides=_measurement_design_config_overrides(
                            report,
                            gap.required_measurement,
                            base_overrides=_MEMORY_PRESSURE_MEASUREMENT_CONFIG_OVERRIDES,
                        ),
                    )
                )

        if (
            gap.required_measurement
            in {
                "add_scored_measurements_for_existing_sweep",
                "score_unscored_capture_candidates",
            }
            and count_for_measurement == 0
        ):
            for label in report.decision_summary.measurement_candidate_labels:
                candidate = candidates_by_label.get(label)
                if candidate is None or candidate.simulator_support_status.startswith("unsupported_"):
                    continue
                add_limited(
                    _candidate_replay_design_config(
                        report=report,
                        base_config=base_config,
                        required_measurement=gap.required_measurement,
                        index=index,
                        candidate=candidate,
                    )
                )

        if gap.required_measurement in {
            "add_same_parallelism_runtime_workload_variants",
            "add_workload_and_parallelism_variants",
        }:
            workload_only_budget = (
                max_configs_per_measurement
                if gap.required_measurement == "add_workload_and_parallelism_variants"
                else design_config_budget
            )
            for design_kind, values in _workload_design_variants(anchor):
                if count_for_measurement >= workload_only_budget:
                    break
                add_limited(
                    _design_config_from_topology(
                        report=report,
                        base_config=base_config,
                        required_measurement=gap.required_measurement,
                        design_kind=design_kind,
                        index=index,
                        world_size=anchor.world_size,
                        local_world_size=anchor.local_world_size,
                        micro_batch_size=values["micro_batch_size"],
                        gradient_accumulation_steps=values["gradient_accumulation_steps"],
                        sample_packing_sequence_len=values["sample_packing_sequence_len"],
                        expert_parallel_size=anchor.expert_parallel_size,
                        tensor_parallel_size=anchor.tensor_parallel_size,
                        pipeline_parallel_size=anchor.pipeline_parallel_size,
                        ulysses_parallel_size=anchor.ulysses_parallel_size,
                        ringattn_parallel_size=anchor.ringattn_parallel_size,
                        data_parallel_replicate_size=anchor.data_parallel_replicate_size,
                        data_parallel_shard_size=anchor.data_parallel_shard_size,
                        config_overrides=_measurement_design_config_overrides(report, gap.required_measurement),
                    )
                )

        if gap.required_measurement == "add_workload_and_parallelism_variants":
            workload_variants = _workload_design_variants(anchor)
            added_combined_axis_families: set[str] = set()
            for axis, values in _prioritized_parallelism_design_values_for_gap(anchor, metadata, gap):
                if axis in added_combined_axis_families:
                    continue
                for workload_design_kind, workload_values in workload_variants:
                    if not _parallelism_design_values_apply_to_workload(axis, values, workload_values):
                        continue
                    if add_limited(
                        _design_config_from_topology(
                            report=report,
                            base_config=base_config,
                            required_measurement=gap.required_measurement,
                            design_kind=f"combined_{workload_design_kind}_{axis}",
                            index=index,
                            world_size=values["world_size"],
                            local_world_size=min(anchor.local_world_size, values["world_size"]),
                            micro_batch_size=workload_values["micro_batch_size"],
                            gradient_accumulation_steps=workload_values["gradient_accumulation_steps"],
                            sample_packing_sequence_len=workload_values["sample_packing_sequence_len"],
                            expert_parallel_size=values["expert_parallel_size"],
                            tensor_parallel_size=values["tensor_parallel_size"],
                            pipeline_parallel_size=values["pipeline_parallel_size"],
                            ulysses_parallel_size=values["ulysses_parallel_size"],
                            ringattn_parallel_size=values["ringattn_parallel_size"],
                            data_parallel_replicate_size=values["data_parallel_replicate_size"],
                            data_parallel_shard_size=values["data_parallel_shard_size"],
                            config_overrides=_measurement_design_config_overrides(report, gap.required_measurement),
                        )
                    ):
                        added_combined_axis_families.add(axis)
                        break

        if gap.required_measurement == "add_same_parallelism_workload_runtime_variants":
            runtime_anchor = _runtime_variant_anchor_candidate(report.candidates, gap.runtime_mismatch_dimensions)
            runtime_anchor_topology = runtime_anchor.topology if runtime_anchor is not None else anchor
            reference_point = _matched_behavior_point_for_candidate(report, runtime_anchor)
            for design_kind, config_overrides in _runtime_design_variants(
                base_config,
                gap.runtime_mismatch_dimensions,
                reference_point=reference_point,
            ):
                add_limited(
                    _design_config_from_topology(
                        report=report,
                        base_config=base_config,
                        required_measurement=gap.required_measurement,
                        design_kind=design_kind,
                        index=index,
                        world_size=runtime_anchor_topology.world_size,
                        local_world_size=runtime_anchor_topology.local_world_size,
                        micro_batch_size=runtime_anchor_topology.micro_batch_size,
                        gradient_accumulation_steps=runtime_anchor_topology.gradient_accumulation_steps,
                        sample_packing_sequence_len=runtime_anchor_topology.sample_packing_sequence_len,
                        expert_parallel_size=runtime_anchor_topology.expert_parallel_size,
                        tensor_parallel_size=runtime_anchor_topology.tensor_parallel_size,
                        pipeline_parallel_size=runtime_anchor_topology.pipeline_parallel_size,
                        ulysses_parallel_size=runtime_anchor_topology.ulysses_parallel_size,
                        ringattn_parallel_size=runtime_anchor_topology.ringattn_parallel_size,
                        data_parallel_replicate_size=runtime_anchor_topology.data_parallel_replicate_size,
                        data_parallel_shard_size=runtime_anchor_topology.data_parallel_shard_size,
                        config_overrides=_measurement_design_config_overrides(
                            report,
                            gap.required_measurement,
                            base_overrides=config_overrides,
                        ),
                    )
                )

        if gap.required_measurement in _SCENARIO_PARALLELISM_AXIS_COMPATIBLE_MEASUREMENTS:
            added_axis_families: set[str] = set()
            for axis, values in _prioritized_parallelism_design_values_for_gap(anchor, metadata, gap):
                if axis in added_axis_families:
                    continue
                gradient_accumulation_steps = _same_workload_ga_for_topology(
                    target_global_batch_size=anchor.global_batch_size,
                    world_size=values["world_size"],
                    micro_batch_size=anchor.micro_batch_size,
                    tensor_parallel_size=values["tensor_parallel_size"],
                    pipeline_parallel_size=values["pipeline_parallel_size"],
                    ulysses_parallel_size=values["ulysses_parallel_size"],
                    ringattn_parallel_size=values["ringattn_parallel_size"],
                )
                if gradient_accumulation_steps is None:
                    continue
                if add_limited(
                    _design_config_from_topology(
                        report=report,
                        base_config=base_config,
                        required_measurement=gap.required_measurement,
                        design_kind=axis,
                        index=index,
                        world_size=values["world_size"],
                        local_world_size=min(anchor.local_world_size, values["world_size"]),
                        micro_batch_size=anchor.micro_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        sample_packing_sequence_len=anchor.sample_packing_sequence_len,
                        expert_parallel_size=values["expert_parallel_size"],
                        tensor_parallel_size=values["tensor_parallel_size"],
                        pipeline_parallel_size=values["pipeline_parallel_size"],
                        ulysses_parallel_size=values["ulysses_parallel_size"],
                        ringattn_parallel_size=values["ringattn_parallel_size"],
                        data_parallel_replicate_size=values["data_parallel_replicate_size"],
                        data_parallel_shard_size=values["data_parallel_shard_size"],
                        config_overrides=_measurement_design_config_overrides(report, gap.required_measurement),
                    )
                ):
                    added_axis_families.add(axis)

    direct_action_counts_by_measurement = Counter(
        design.label.split(":", 3)[1] for design in rendered if design.label.startswith("design:")
    )
    for action in report.decision_summary.validation_actions:
        direct_design_kind = _DIRECT_ACTION_DESIGN_MEASUREMENTS.get(action.required_measurement)
        if direct_design_kind is None:
            continue
        for label in action.candidate_labels:
            if direct_action_counts_by_measurement[action.required_measurement] >= max_configs_per_measurement:
                break
            candidate = candidates_by_label.get(label)
            if candidate is None:
                continue
            added_before = len(rendered)
            add_design(
                _action_backed_design_config(
                    report=report,
                    base_config=base_config,
                    required_measurement=action.required_measurement,
                    design_kind=_action_candidate_design_kind(direct_design_kind, action, candidate),
                    index=index,
                    candidate=candidate,
                )
            )
            if len(rendered) > added_before:
                direct_action_counts_by_measurement[action.required_measurement] += 1
    return rendered


def materialize_measurement_configs(report: ScenarioReport) -> list[ScenarioMeasurementConfig]:
    """Render candidate and scenario-capture design YAML payloads."""
    candidate_configs = materialize_measurement_candidate_configs(report)
    design_configs = materialize_measurement_design_configs(report)
    return [
        *candidate_configs,
        *[
            ScenarioMeasurementConfig(
                label=item.label,
                filename=f"design_{item.filename}",
                config=item.config,
            )
            for item in design_configs
        ],
    ]


def write_measurement_candidate_configs(
    report: ScenarioReport, output_dir: str | Path
) -> list[ScenarioMeasurementConfig]:
    """Write the report's measurement portfolio as YAML configs and return the rendered payloads."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rendered = materialize_measurement_candidate_configs(report)
    for item in rendered:
        (output_path / item.filename).write_text(
            yaml.safe_dump(runtime_training_config(item.config), sort_keys=False),
            encoding="utf-8",
        )
    return rendered


def write_measurement_configs(report: ScenarioReport, output_dir: str | Path) -> list[ScenarioMeasurementConfig]:
    """Write candidate and design measurement configs as YAML files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rendered = materialize_measurement_configs(report)
    for item in rendered:
        (output_path / item.filename).write_text(
            yaml.safe_dump(runtime_training_config(item.config), sort_keys=False),
            encoding="utf-8",
        )
    return rendered


def _attach_measurement_design_summary(report: ScenarioReport) -> ScenarioReport:
    design_configs = materialize_measurement_design_configs(report)
    labels = [item.label for item in design_configs]
    filenames = [f"design_{item.filename}" for item in design_configs]
    scenario_readiness = replace(
        report.decision_summary.scenario_readiness,
        measurement_design_config_count=len(design_configs),
        measurement_design_config_labels=labels,
        measurement_design_config_filenames=filenames,
    )
    decision_summary = replace(
        report.decision_summary,
        measurement_design_config_count=len(design_configs),
        measurement_design_config_labels=labels,
        measurement_design_config_filenames=filenames,
        scenario_readiness=scenario_readiness,
    )
    return replace(report, decision_summary=decision_summary)


def _cross_model_analog_candidates(candidates: list[ScenarioCandidate]) -> list[ScenarioCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.prediction_confidence == "cross_model_extrapolated" or "cross_model_analog" in candidate.risk_flags
    ]


_CROSS_MODEL_ANALOG_FACTOR_FIELDS = (
    ("active_param_scale", "cross_model_active_param_scale"),
    ("sequence_length_factor", "cross_model_sequence_length_factor"),
    ("parallelism_factor", "cross_model_parallelism_factor"),
    ("memory_factor", "cross_model_memory_factor"),
)


def _scored_cross_model_analog_candidates(candidates: list[ScenarioCandidate]) -> list[ScenarioCandidate]:
    return [
        candidate
        for candidate in _cross_model_analog_candidates(candidates)
        if candidate.score_tokens_per_sec is not None
    ]


def _cross_model_analog_factor_key(candidate: ScenarioCandidate) -> tuple[float | None, ...]:
    return tuple(getattr(candidate.behavior, field_name) for _, field_name in _CROSS_MODEL_ANALOG_FACTOR_FIELDS)


def _cross_model_analog_unique_factor_count(candidates: list[ScenarioCandidate]) -> int:
    return len(
        {_cross_model_analog_factor_key(candidate) for candidate in _scored_cross_model_analog_candidates(candidates)}
    )


def _cross_model_analog_factor_status(candidates: list[ScenarioCandidate]) -> str:
    if not _cross_model_analog_candidates(candidates):
        return "not_used"
    scored = _scored_cross_model_analog_candidates(candidates)
    if not scored:
        return "no_scored_cross_model_factors"
    if _cross_model_analog_unique_factor_count(scored) == 1:
        return "single_cross_model_factor"
    return "multiple_cross_model_factors"


def _cross_model_analog_factor_ranges(candidates: list[ScenarioCandidate]) -> dict[str, list[float]]:
    ranges: dict[str, list[float]] = {}
    scored = _scored_cross_model_analog_candidates(candidates)
    for range_name, field_name in _CROSS_MODEL_ANALOG_FACTOR_FIELDS:
        values = [
            getattr(candidate.behavior, field_name)
            for candidate in scored
            if getattr(candidate.behavior, field_name) is not None
        ]
        if values:
            ranges[range_name] = [round(min(values), 3), round(max(values), 3)]
    return dict(sorted(ranges.items()))


def _cross_model_analog_prediction_interval_top(candidates: list[ScenarioCandidate]) -> list[ScenarioCandidate]:
    scored = _scored_cross_model_analog_candidates(candidates)
    if not scored:
        return []
    predicted_order = sorted(
        scored,
        key=lambda candidate: (candidate.score_tokens_per_sec or float("-inf"), candidate.label),
        reverse=True,
    )
    predicted_best = predicted_order[0]
    best_lower = predicted_best.prediction_interval_lower_tokens_per_sec
    best_upper = predicted_best.prediction_interval_upper_tokens_per_sec
    if best_lower is None or best_upper is None:
        return []
    return [
        candidate
        for candidate in predicted_order
        if candidate.prediction_interval_lower_tokens_per_sec is not None
        and candidate.prediction_interval_upper_tokens_per_sec is not None
        and candidate.prediction_interval_lower_tokens_per_sec <= best_upper
        and candidate.prediction_interval_upper_tokens_per_sec >= best_lower
    ]


def _prediction_interval_selectivity_status(
    *,
    scored_candidate_count: int,
    prediction_interval_top_count: int,
) -> str:
    if scored_candidate_count == 0:
        return "no_scored_cross_model_candidates"
    if prediction_interval_top_count == 0:
        return "no_prediction_interval_top"
    if prediction_interval_top_count == 1:
        return "selective_prediction_interval_top"
    if prediction_interval_top_count >= scored_candidate_count:
        return "nonselective_prediction_interval_top"
    return "partial_prediction_interval_top"


def _cross_model_analog_support(
    candidates: list[ScenarioCandidate],
) -> tuple[str, int, int, int, int, int, int]:
    analog_candidates = _cross_model_analog_candidates(candidates)
    if not analog_candidates:
        return "not_used", 0, 0, 0, 0, 0, 0

    scored = [candidate for candidate in analog_candidates if candidate.score_tokens_per_sec is not None]
    unique_predictions = {round(candidate.score_tokens_per_sec or 0.0, 3) for candidate in scored}
    unique_matched_labels = {
        candidate.behavior.matched_label for candidate in scored if candidate.behavior.matched_label is not None
    }
    unique_target_strategies = {_parallelism_strategy_key(candidate) for candidate in scored}
    unique_target_runtime_signatures = {candidate.target_runtime_signature for candidate in scored}
    if not scored:
        status = "no_scored_cross_model_candidates"
    elif len(scored) == 1:
        status = "single_scored_cross_model_candidate"
    elif len(unique_matched_labels) == 1 and len(unique_target_strategies) > 1:
        status = "single_reference_cannot_rank_parallelism_variants"
    elif len(unique_matched_labels) == 1 and len(unique_target_runtime_signatures) > 1:
        status = "single_reference_cannot_rank_runtime_variants"
    elif len(unique_predictions) == 1 and len(unique_matched_labels) == 1:
        status = "single_reference_tied_tradeoff"
    elif len(unique_predictions) == 1:
        status = "tied_cross_model_frontier"
    else:
        status = "scored_cross_model_frontier"
    return (
        status,
        len(analog_candidates),
        len(scored),
        len(unique_predictions),
        len(unique_matched_labels),
        len(unique_target_strategies),
        len(unique_target_runtime_signatures),
    )


def _model_generalization_support(
    candidates: list[ScenarioCandidate],
    *,
    memory_blocked_count: int,
    cross_model_analog_support_status: str,
    cross_model_analog_scored_count: int,
    cross_model_analog_factor_status: str,
    risk_adjusted_interval_overlap_status: str,
    cross_model_analog_prediction_interval_selectivity_status: str = "unknown",
) -> tuple[str, list[str]]:
    if not candidates:
        return "no_candidates", ["no_candidates"]

    cross_model_candidates = _cross_model_analog_candidates(candidates)
    if cross_model_candidates:
        blockers: list[str] = []
        if cross_model_analog_scored_count == 0:
            blockers.append("no_scored_cross_model_candidates")
        if cross_model_analog_support_status in {
            "single_scored_cross_model_candidate",
            "single_reference_cannot_rank_parallelism_variants",
            "single_reference_cannot_rank_runtime_variants",
            "single_reference_tied_tradeoff",
            "tied_cross_model_frontier",
        }:
            blockers.append(cross_model_analog_support_status)
        if cross_model_analog_factor_status == "single_cross_model_factor" and cross_model_analog_scored_count > 1:
            blockers.append("single_cross_model_factor")
        if (
            cross_model_analog_prediction_interval_selectivity_status == "nonselective_prediction_interval_top"
            and cross_model_analog_scored_count > 1
        ):
            blockers.append("nonselective_cross_model_prediction_interval_top")
        if risk_adjusted_interval_overlap_status == "overlapping_best_interval":
            blockers.append("risk_adjusted_interval_overlap")
        requires_target_measurement = any(
            "requires_remeasurement" in candidate.risk_flags for candidate in cross_model_candidates
        )
        blockers = sorted(set(blockers))
        if not blockers and cross_model_analog_support_status == "scored_cross_model_frontier":
            if requires_target_measurement:
                return (
                    "cross_model_generalization_supported_measurement_prior",
                    ["cross_model_candidates_require_measurement"],
                )
            return "cross_model_generalization_supported_frontier", []
        if requires_target_measurement:
            blockers = sorted({*blockers, "cross_model_candidates_require_measurement"})
        if "no_scored_cross_model_candidates" in blockers:
            return "cross_model_generalization_unscored", blockers
        if (
            "risk_adjusted_interval_overlap" in blockers
            or "nonselective_cross_model_prediction_interval_top" in blockers
        ):
            return "cross_model_generalization_interval_uncertain", blockers
        return "cross_model_generalization_requires_target_measurement", blockers

    if memory_blocked_count == len(candidates):
        return "same_model_unscored_memory_blocked", ["memory_blocked_all_candidates"]
    if any(candidate.prediction_confidence != "calibrated" for candidate in candidates):
        blockers = ["same_model_extrapolated_predictions"]
        if any("requires_remeasurement" in candidate.risk_flags for candidate in candidates):
            blockers.append("same_model_candidates_require_measurement")
        if any(candidate.memory_coverage_status == "analytic_floor_only" for candidate in candidates):
            blockers.append("analytic_floor_only_memory")
        return "same_model_extrapolation_requires_measurement", sorted(set(blockers))
    if any(candidate.behavior.model_ref is None for candidate in candidates):
        return "same_model_calibrated_unknown_model_ref", ["missing_model_ref"]
    return "same_model_calibrated", []


def _scenario_prediction_fidelity_support(
    candidates: list[ScenarioCandidate],
    *,
    memory_blocked_count: int,
    phase_bottleneck_candidate_count: int,
    routing_regime_status: str,
    simulator_support_status_counts: dict[str, int],
) -> tuple[str, list[str]]:
    if not candidates:
        return "no_scenario_candidates", ["no_candidates"]

    scored = [candidate for candidate in candidates if candidate.score_tokens_per_sec is not None]
    if not scored:
        blockers = ["no_scored_candidates"]
        if memory_blocked_count == len(candidates):
            blockers.append("memory_blocked_all_candidates")
        return "no_scored_scenario_fidelity", sorted(set(blockers))

    blockers: set[str] = set()
    # Memory-blocked candidates (observed/announced OOM rows) cannot carry throughput fidelity by
    # definition; they are measured boundary evidence, not fidelity gaps — mirror the capture
    # support's unscored definition and exempt them here too.
    unscored_non_blocked = [
        candidate
        for candidate in candidates
        if candidate.score_tokens_per_sec is None and not _is_memory_blocked(candidate)
    ]
    if unscored_non_blocked:
        blockers.add("unscored_candidates")
    if any(candidate.prediction_confidence == "cross_model_extrapolated" for candidate in scored):
        blockers.add("cross_model_predictions")
    if any(candidate.prediction_confidence not in {"calibrated", "cross_model_extrapolated"} for candidate in scored):
        blockers.add("extrapolated_predictions")
    if any(candidate.calibration_scope != "exact_calibrated" for candidate in scored):
        blockers.add("outside_exact_calibration_scope")
    if any(candidate.memory_coverage_status == "analytic_floor_only" for candidate in scored):
        blockers.add("analytic_floor_only_memory")
    if any(
        candidate.memory_coverage_status.startswith("extrapolated_")
        or candidate.memory_coverage_status == "calibrated_overhead_peak_with_scaled_residual"
        for candidate in scored
    ):
        blockers.add("scaled_or_extrapolated_memory_peak")
    if any(candidate.timing_coverage_status == "no_timing_evidence" for candidate in scored):
        blockers.add("no_timing_evidence")
    if any(
        "extrapolated" in candidate.timing_coverage_status
        or candidate.timing_coverage_status.startswith(("reference_", "cross_model_reference_"))
        for candidate in scored
    ):
        blockers.add("reference_or_extrapolated_timing")
    if any(candidate.timing_coverage_status.endswith("_total_step_only") for candidate in scored):
        blockers.add("missing_phase_timing")
    if phase_bottleneck_candidate_count < len(scored):
        blockers.add("missing_phase_bottleneck_evidence")
    if any("requires_remeasurement" in candidate.risk_flags for candidate in scored):
        blockers.add("requires_remeasurement")
    if any(status.startswith("unsupported_") for status in simulator_support_status_counts):
        blockers.add("unsupported_simulator_surface")
    if any(
        status != "supported_local_non_pp" and not status.startswith("unsupported_")
        for status in simulator_support_status_counts
    ):
        blockers.add("partial_simulator_surface_support")
    if any(
        candidate.prediction_uncertainty_fraction is not None and candidate.prediction_uncertainty_fraction >= 0.50
        for candidate in scored
    ):
        blockers.add("high_prediction_uncertainty")
    if routing_regime_status in {"unknown_routing_regime", "mixed_routing_regime"}:
        blockers.add(routing_regime_status)

    blocker_list = sorted(blockers)
    if "cross_model_predictions" in blockers:
        return "cross_model_fidelity_requires_target_measurement", blocker_list
    if {
        "extrapolated_predictions",
        "outside_exact_calibration_scope",
        "analytic_floor_only_memory",
        "scaled_or_extrapolated_memory_peak",
        "no_timing_evidence",
        "reference_or_extrapolated_timing",
        "requires_remeasurement",
        "unsupported_simulator_surface",
        "high_prediction_uncertainty",
        "unknown_routing_regime",
        "mixed_routing_regime",
    } & blockers:
        return "extrapolated_fidelity_requires_measurement", blocker_list
    if "partial_simulator_surface_support" in blockers:
        return "partial_surface_fidelity_requires_measurement", blocker_list
    total_step_only_blockers = {
        "missing_phase_timing",
        "missing_phase_bottleneck_evidence",
        "unscored_candidates",
    }
    if blockers and blockers <= total_step_only_blockers:
        if "unscored_candidates" in blockers:
            return "partial_calibrated_total_step_fidelity", blocker_list
        return "calibrated_total_step_fidelity", blocker_list
    if "unscored_candidates" in blockers:
        return "partial_calibrated_phase_fidelity", blocker_list
    return "calibrated_phase_fidelity", []


def _scenario_capture_support(
    *,
    candidate_count: int,
    scored_count: int,
    memory_blocked_count: int,
    varied_parallelism_dimensions: list[str],
    varied_workload_dimensions: list[str],
    varied_runtime_dimensions: list[str],
    runtime_mismatch_dimensions: list[str],
    parallelism_axis_coverage_status_counts: dict[str, int],
    simulator_support_status_counts: dict[str, int],
) -> tuple[str, list[str]]:
    if candidate_count == 0:
        return "no_scenario_candidates", ["no_candidates"]

    has_parallelism_variation = bool(varied_parallelism_dimensions)
    has_workload_variation = bool(varied_workload_dimensions)
    clean_axis_count = parallelism_axis_coverage_status_counts.get("scored_parallelism_axis", 0)
    blockers: set[str] = set()
    if candidate_count == 1:
        blockers.add("single_candidate")
    if scored_count == 0:
        blockers.add("no_scored_candidates")
    elif scored_count + memory_blocked_count < candidate_count:
        # Memory-blocked candidates (observed OOM rows) are measured boundary evidence, not capture
        # gaps; only genuinely unscored non-blocked candidates block broad capture.
        blockers.add("unscored_candidates")
    if memory_blocked_count == candidate_count:
        blockers.add("memory_blocked_all_candidates")
    if not has_parallelism_variation:
        blockers.add("missing_parallelism_variation")
    if not has_workload_variation:
        blockers.add("missing_workload_variation")
    if varied_runtime_dimensions:
        blockers.add("runtime_variant_variation")
    if runtime_mismatch_dimensions and (scored_count == 0 or memory_blocked_count == candidate_count):
        blockers.add("runtime_mismatched_measurement_support")
    if has_parallelism_variation and clean_axis_count == 0:
        blockers.add("no_clean_parallelism_axis_coverage")
    if any("blocked" in status for status in parallelism_axis_coverage_status_counts):
        blockers.add("blocked_parallelism_axes")
    if any(status.startswith("confounded_") for status in parallelism_axis_coverage_status_counts):
        blockers.add("confounded_parallelism_axes")
    if any("unscored" in status for status in parallelism_axis_coverage_status_counts):
        blockers.add("unscored_parallelism_axes")
    if any(status.startswith("unsupported_") for status in simulator_support_status_counts):
        blockers.add("unsupported_simulator_surfaces")
    if any(
        status != "supported_local_non_pp" and not status.startswith("unsupported_")
        for status in simulator_support_status_counts
    ):
        blockers.add("partial_simulator_surfaces")

    blocker_list = sorted(blockers)
    if not has_parallelism_variation and not has_workload_variation:
        return "single_shape_capture", blocker_list
    if has_workload_variation and not has_parallelism_variation:
        return "workload_only_capture", blocker_list
    if has_parallelism_variation and not has_workload_variation:
        if scored_count == 0:
            return "unscored_parallelism_capture", blocker_list
        if clean_axis_count == 0:
            return "parallelism_capture_without_clean_axis", blocker_list
        if blockers - {"missing_workload_variation"}:
            return "partial_parallelism_capture", blocker_list
        return "parallelism_only_capture", blocker_list

    if scored_count == 0:
        return "unscored_broad_scenario_capture", blocker_list
    if clean_axis_count == 0:
        return "coupled_broad_scenario_capture", blocker_list
    if blockers:
        return "partial_broad_scenario_capture", blocker_list
    return "broad_scenario_capture", []


def _scenario_readiness_status(
    *,
    candidate_count: int,
    unique_parallelism_strategy_count: int,
    can_capture_scenario: bool,
    can_predict_scenario_fidelity: bool,
    can_select_parallelism_tradeoff: bool,
    can_generalize_model: bool,
    scenario_capture_status: str,
    scenario_prediction_fidelity_status: str,
    model_generalization_status: str,
    parallelism_optimality_status: str,
) -> str:
    if candidate_count == 0:
        return "blocked_by_no_scenario_candidates"
    if not can_capture_scenario:
        if scenario_capture_status == "single_shape_capture":
            return "blocked_by_single_shape_capture"
        if scenario_capture_status.startswith("unscored_"):
            return "blocked_by_unscored_scenario_capture"
        if scenario_capture_status in {
            "coupled_broad_scenario_capture",
            "parallelism_capture_without_clean_axis",
        }:
            return "blocked_by_confounded_scenario_capture"
        return "blocked_by_incomplete_scenario_capture"
    if not can_predict_scenario_fidelity:
        if scenario_prediction_fidelity_status == "no_scored_scenario_fidelity":
            return "blocked_by_no_scored_scenario_fidelity"
        if scenario_prediction_fidelity_status in {
            "cross_model_fidelity_requires_target_measurement",
            "extrapolated_fidelity_requires_measurement",
        }:
            return "blocked_by_unvalidated_prediction_fidelity"
        if scenario_prediction_fidelity_status in {
            "partial_calibrated_total_step_fidelity",
            "calibrated_total_step_fidelity",
            "partial_calibrated_phase_fidelity",
        }:
            return "partial_scenario_prediction_fidelity"
        return "blocked_by_prediction_fidelity_gaps"
    if not can_generalize_model:
        if model_generalization_status.endswith("requires_measurement"):
            return "blocked_by_model_generalization_measurement"
        return "blocked_by_model_generalization_gaps"
    if unique_parallelism_strategy_count > 1 and not can_select_parallelism_tradeoff:
        if parallelism_optimality_status == "confounded_parallelism_winner":
            return "blocked_by_confounded_parallelism_tradeoff"
        if parallelism_optimality_status == "requires_measurement_before_parallelism_optimality":
            return "blocked_by_parallelism_measurement"
        if parallelism_optimality_status == "interval_overlap_parallelism_uncertain":
            return "blocked_by_parallelism_interval_overlap"
        return "blocked_by_parallelism_tradeoff_gaps"
    return "validated_scenario_readiness"


def _scenario_readiness(
    *,
    candidate_count: int,
    scored_count: int,
    unscored_count: int,
    memory_blocked_count: int,
    unique_parallelism_strategy_count: int,
    scored_parallelism_strategy_count: int,
    promotable_parallelism_strategy_count: int,
    scenario_capture_status: str,
    scenario_capture_blockers: list[str],
    scenario_prediction_fidelity_status: str,
    scenario_prediction_fidelity_blockers: list[str],
    parallelism_optimality_status: str,
    parallelism_optimality_blockers: list[str],
    model_generalization_status: str,
    model_generalization_blockers: list[str],
    measurement_readiness_status: str,
    measurement_portfolio_coverage_status: str,
    measurement_portfolio_coverage_blockers: list[str],
    scenario_capture_gaps: list[ScenarioCaptureGap],
    scenario_capture_gap_status_counts: dict[str, int],
    scenario_capture_gap_required_measurements: list[str],
    validation_actions: list[ScenarioValidationAction],
    validation_action_status_counts: dict[str, int],
    validation_action_required_measurements: list[str],
    validation_action_total_gpu_count: int,
    measurement_candidate_labels: list[str],
    measurement_portfolio_total_gpu_count: int,
    measurement_portfolio_parallelism_axis_gap_names: list[str],
    measurement_portfolio_cross_model_analog_count: int,
    varied_parallelism_dimensions: list[str],
    varied_workload_dimensions: list[str],
    varied_runtime_dimensions: list[str],
    runtime_mismatch_dimensions: list[str],
    parallelism_axis_coverage_status_counts: dict[str, int],
    scored_parallelism_axis_names: list[str],
    blocked_parallelism_axis_names: list[str],
    confounded_parallelism_axis_names: list[str],
    unscored_parallelism_axis_names: list[str],
    missing_parallelism_axis_names: list[str],
    simulator_support_status_counts: dict[str, int],
    prediction_confidence_counts: dict[str, int],
    calibration_scope_counts: dict[str, int],
    memory_coverage_status_counts: dict[str, int],
    timing_coverage_status_counts: dict[str, int],
    cross_model_analog_support_status: str,
    cross_model_analog_candidate_count: int,
    cross_model_analog_scored_count: int,
    benchmark_support: ScenarioBenchmarkSupport,
) -> ScenarioReadiness:
    can_capture_scenario = scenario_capture_status == "broad_scenario_capture"
    can_predict_scenario_fidelity = scenario_prediction_fidelity_status == "calibrated_phase_fidelity"
    can_select_parallelism_tradeoff = parallelism_optimality_status in {
        "supported_promotable_parallelism_tradeoff",
        "supported_parallelism_winner",
    }
    can_generalize_model = model_generalization_status in {
        "same_model_calibrated",
        "cross_model_generalization_supported_frontier",
    }
    readiness_status = _scenario_readiness_status(
        candidate_count=candidate_count,
        unique_parallelism_strategy_count=unique_parallelism_strategy_count,
        can_capture_scenario=can_capture_scenario,
        can_predict_scenario_fidelity=can_predict_scenario_fidelity,
        can_select_parallelism_tradeoff=can_select_parallelism_tradeoff,
        can_generalize_model=can_generalize_model,
        scenario_capture_status=scenario_capture_status,
        scenario_prediction_fidelity_status=scenario_prediction_fidelity_status,
        model_generalization_status=model_generalization_status,
        parallelism_optimality_status=parallelism_optimality_status,
    )
    top_capture_gaps = scenario_capture_gaps[:5]
    return ScenarioReadiness(
        readiness_status=readiness_status,
        can_capture_scenario=can_capture_scenario,
        can_predict_scenario_fidelity=can_predict_scenario_fidelity,
        can_select_parallelism_tradeoff=can_select_parallelism_tradeoff,
        can_generalize_model=can_generalize_model,
        scenario_capture_status=scenario_capture_status,
        scenario_capture_blockers=scenario_capture_blockers,
        scenario_prediction_fidelity_status=scenario_prediction_fidelity_status,
        scenario_prediction_fidelity_blockers=scenario_prediction_fidelity_blockers,
        parallelism_optimality_status=parallelism_optimality_status,
        parallelism_optimality_blockers=parallelism_optimality_blockers,
        model_generalization_status=model_generalization_status,
        model_generalization_blockers=model_generalization_blockers,
        measurement_readiness_status=measurement_readiness_status,
        measurement_portfolio_coverage_status=measurement_portfolio_coverage_status,
        measurement_portfolio_coverage_blockers=measurement_portfolio_coverage_blockers,
        required_measurements=_unique_in_order(
            [*scenario_capture_gap_required_measurements, *validation_action_required_measurements]
        ),
        scenario_capture_gap_count=len(scenario_capture_gaps),
        scenario_capture_gap_status_counts=scenario_capture_gap_status_counts,
        scenario_capture_gap_required_measurements=scenario_capture_gap_required_measurements,
        top_scenario_capture_gap_statuses=[gap.gap_status for gap in top_capture_gaps],
        validation_action_count=len(validation_actions),
        validation_action_status_counts=validation_action_status_counts,
        validation_action_required_measurements=validation_action_required_measurements,
        validation_action_total_gpu_count=validation_action_total_gpu_count,
        measurement_candidate_count=len(measurement_candidate_labels),
        measurement_candidate_labels=measurement_candidate_labels,
        measurement_portfolio_total_gpu_count=measurement_portfolio_total_gpu_count,
        measurement_portfolio_parallelism_axis_gap_names=measurement_portfolio_parallelism_axis_gap_names,
        measurement_portfolio_cross_model_analog_count=measurement_portfolio_cross_model_analog_count,
        candidate_count=candidate_count,
        scored_count=scored_count,
        unscored_count=unscored_count,
        memory_blocked_count=memory_blocked_count,
        unique_parallelism_strategy_count=unique_parallelism_strategy_count,
        scored_parallelism_strategy_count=scored_parallelism_strategy_count,
        promotable_parallelism_strategy_count=promotable_parallelism_strategy_count,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        runtime_mismatch_dimensions=runtime_mismatch_dimensions,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
        scored_parallelism_axis_names=scored_parallelism_axis_names,
        blocked_parallelism_axis_names=blocked_parallelism_axis_names,
        confounded_parallelism_axis_names=confounded_parallelism_axis_names,
        unscored_parallelism_axis_names=unscored_parallelism_axis_names,
        missing_parallelism_axis_names=missing_parallelism_axis_names,
        simulator_support_status_counts=simulator_support_status_counts,
        prediction_confidence_counts=prediction_confidence_counts,
        calibration_scope_counts=calibration_scope_counts,
        memory_coverage_status_counts=memory_coverage_status_counts,
        timing_coverage_status_counts=timing_coverage_status_counts,
        cross_model_analog_support_status=cross_model_analog_support_status,
        cross_model_analog_candidate_count=cross_model_analog_candidate_count,
        cross_model_analog_scored_count=cross_model_analog_scored_count,
        benchmark_support=benchmark_support,
    )


def _scenario_capture_gap(
    *,
    gap_status: str,
    priority: int,
    required_measurement: str,
    reason: str,
    blocker_names: list[str],
    candidate_count: int,
    scored_count: int,
    memory_blocked_count: int,
    missing_parallelism_axis_names: list[str],
    blocked_parallelism_axis_names: list[str],
    confounded_parallelism_axis_names: list[str],
    unscored_parallelism_axis_names: list[str],
    varied_parallelism_dimensions: list[str],
    varied_workload_dimensions: list[str],
    varied_runtime_dimensions: list[str],
    runtime_mismatch_dimensions: list[str],
) -> ScenarioCaptureGap:
    return ScenarioCaptureGap(
        gap_status=gap_status,
        priority=priority,
        required_measurement=required_measurement,
        reason=reason,
        blocker_names=sorted(set(blocker_names)),
        candidate_count=candidate_count,
        scored_count=scored_count,
        unscored_count=max(candidate_count - scored_count, 0),
        memory_blocked_count=memory_blocked_count,
        missing_parallelism_axis_names=missing_parallelism_axis_names,
        blocked_parallelism_axis_names=blocked_parallelism_axis_names,
        confounded_parallelism_axis_names=confounded_parallelism_axis_names,
        unscored_parallelism_axis_names=unscored_parallelism_axis_names,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        runtime_mismatch_dimensions=runtime_mismatch_dimensions,
    )


def _scenario_capture_gap_portfolio(
    *,
    scenario_capture_blockers: list[str],
    candidate_count: int,
    scored_count: int,
    memory_blocked_count: int,
    missing_parallelism_axis_names: list[str],
    blocked_parallelism_axis_names: list[str],
    confounded_parallelism_axis_names: list[str],
    unscored_parallelism_axis_names: list[str],
    varied_parallelism_dimensions: list[str],
    varied_workload_dimensions: list[str],
    varied_runtime_dimensions: list[str],
    runtime_mismatch_dimensions: list[str],
) -> list[ScenarioCaptureGap]:
    blockers = set(scenario_capture_blockers)
    common = {
        "candidate_count": candidate_count,
        "scored_count": scored_count,
        "memory_blocked_count": memory_blocked_count,
        "missing_parallelism_axis_names": missing_parallelism_axis_names,
        "blocked_parallelism_axis_names": blocked_parallelism_axis_names,
        "confounded_parallelism_axis_names": confounded_parallelism_axis_names,
        "unscored_parallelism_axis_names": unscored_parallelism_axis_names,
        "varied_parallelism_dimensions": varied_parallelism_dimensions,
        "varied_workload_dimensions": varied_workload_dimensions,
        "varied_runtime_dimensions": varied_runtime_dimensions,
        "runtime_mismatch_dimensions": runtime_mismatch_dimensions,
    }
    gaps: list[ScenarioCaptureGap] = []
    if "no_candidates" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="no_scenario_candidates_need_base_sweep",
                priority=130,
                required_measurement="add_base_scenario_candidates",
                reason="no candidate configurations were generated for this scenario",
                blocker_names=["no_candidates"],
                **common,
            )
        )
    if "single_candidate" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="single_candidate_needs_workload_and_parallelism_variants",
                priority=120,
                required_measurement="add_workload_and_parallelism_variants",
                reason="one candidate cannot prove scenario behavior across workload or topology changes",
                blocker_names=["single_candidate"],
                **common,
            )
        )
    if "missing_parallelism_variation" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="missing_parallelism_variation_needs_axis_sweep",
                priority=110,
                required_measurement="add_same_workload_same_runtime_parallelism_axis_variants",
                reason="scenario capture has no topology variation to test parallelism tradeoffs",
                blocker_names=["missing_parallelism_variation"],
                **common,
            )
        )
    if "confounded_parallelism_axes" in blockers or "no_clean_parallelism_axis_coverage" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="confounded_parallelism_capture_needs_like_for_like_axis_pairs",
                priority=105,
                required_measurement="add_same_workload_same_runtime_axis_pairs",
                reason="parallelism varies, but no clean like-for-like axis comparison proves the scenario",
                blocker_names=[
                    blocker
                    for blocker in ["confounded_parallelism_axes", "no_clean_parallelism_axis_coverage"]
                    if blocker in blockers
                ],
                **common,
            )
        )
    if "blocked_parallelism_axes" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="blocked_parallelism_capture_needs_fit_boundary",
                priority=100,
                required_measurement="add_fit_failure_boundary_near_blocked_axes",
                reason="some topology axes are represented only by memory-blocked candidates",
                blocker_names=["blocked_parallelism_axes"],
                **common,
            )
        )
    if "missing_workload_variation" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="missing_workload_variation_needs_workload_sweep",
                priority=95,
                required_measurement="add_same_parallelism_runtime_workload_variants",
                reason="scenario capture has no workload variation to test scaling over batch or sequence shape",
                blocker_names=["missing_workload_variation"],
                **common,
            )
        )
    if "unsupported_simulator_surfaces" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="unsupported_surface_capture_needs_support_or_probe",
                priority=90,
                required_measurement="add_simulator_support_or_direct_probe_for_surface",
                reason="candidate surface is outside the simulator implementation",
                blocker_names=["unsupported_simulator_surfaces"],
                **common,
            )
        )
    if "memory_blocked_all_candidates" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="memory_blocked_capture_needs_fit_probe",
                priority=90,
                required_measurement="add_memory_pressure_fit_probe_or_reduce_batch",
                reason="every candidate is memory blocked, so no scored scenario behavior is captured",
                blocker_names=["memory_blocked_all_candidates"],
                **common,
            )
        )
    if "runtime_mismatched_measurement_support" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="runtime_mismatch_capture_needs_runtime_variant",
                priority=88,
                required_measurement="add_same_parallelism_workload_runtime_variants",
                reason="supporting measurements differ from target runtime knobs",
                blocker_names=["runtime_mismatched_measurement_support"],
                **common,
            )
        )
    if "no_scored_candidates" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="no_scored_capture_needs_supported_measurement",
                priority=85,
                required_measurement="add_scored_measurements_for_existing_sweep",
                reason="generated candidates do not include any scored throughput measurement",
                blocker_names=["no_scored_candidates"],
                **common,
            )
        )
    if "unscored_candidates" in blockers or "unscored_parallelism_axes" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="unscored_capture_needs_candidate_replay",
                priority=80,
                required_measurement="score_unscored_capture_candidates",
                reason="some generated candidates cannot contribute measured scenario behavior",
                blocker_names=[
                    blocker for blocker in ["unscored_candidates", "unscored_parallelism_axes"] if blocker in blockers
                ],
                **common,
            )
        )
    if "runtime_variant_variation" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="runtime_variant_capture_needs_runtime_isolation",
                priority=75,
                required_measurement="replay_with_runtime_dimensions_fixed",
                reason="runtime knobs vary with the scenario and confound capture",
                blocker_names=["runtime_variant_variation"],
                **common,
            )
        )
    if "partial_simulator_surfaces" in blockers:
        gaps.append(
            _scenario_capture_gap(
                gap_status="partial_surface_capture_needs_support_or_probe",
                priority=65,
                required_measurement="add_partial_surface_support_or_direct_probe",
                reason="some candidates are only partially covered by the simulator surface",
                blocker_names=["partial_simulator_surfaces"],
                **common,
            )
        )
    return sorted(gaps, key=lambda gap: (-gap.priority, gap.gap_status))


_CROSS_MODEL_ANALOG_SUPPORT_FACTORS = {
    "single_scored_cross_model_candidate": 0.85,
    "single_reference_cannot_rank_parallelism_variants": 0.60,
    "single_reference_cannot_rank_runtime_variants": 0.60,
    "single_reference_tied_tradeoff": 0.70,
    "tied_cross_model_frontier": 0.80,
}


def _replace_decision_factor(factors: list[str], prefix: str, value: str) -> list[str]:
    replacement = f"{prefix}{value}"
    updated: list[str] = []
    replaced = False
    for factor in factors:
        if factor.startswith(prefix):
            if not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(factor)
    if not replaced:
        updated.append(replacement)
    return updated


def _apply_cross_model_analog_support_risk(candidates: list[ScenarioCandidate]) -> list[ScenarioCandidate]:
    support_status, *_ = _cross_model_analog_support(candidates)
    factor = _CROSS_MODEL_ANALOG_SUPPORT_FACTORS.get(support_status, 1.0)
    if factor >= 1.0:
        return candidates

    adjusted_candidates: list[ScenarioCandidate] = []
    support_flag = f"cross_model_support:{support_status}"
    for candidate in candidates:
        is_analog_candidate = (
            candidate.prediction_confidence == "cross_model_extrapolated"
            or "cross_model_analog" in candidate.risk_flags
        )
        if not is_analog_candidate:
            adjusted_candidates.append(candidate)
            continue
        adjusted_uncertainty = (
            round(min(candidate.prediction_uncertainty_fraction + 0.10, 0.95), 3)
            if candidate.prediction_uncertainty_fraction is not None
            else None
        )
        raw_interval_lower, raw_interval_upper = _prediction_interval(
            candidate.score_tokens_per_sec,
            adjusted_uncertainty,
        )
        if candidate.score_risk_adjusted_tokens_per_sec is None:
            decision_factors = _replace_decision_factor(
                candidate.decision_factors,
                "prediction_uncertainty_fraction=",
                _format_factor_float(adjusted_uncertainty),
            )
            if raw_interval_lower is not None and raw_interval_upper is not None:
                decision_factors = _replace_decision_factor(
                    decision_factors,
                    "prediction_interval_tokens_per_sec=",
                    f"{_format_factor_float(raw_interval_lower)}..{_format_factor_float(raw_interval_upper)}",
                )
            decision_factors.extend(
                [
                    f"cross_model_support={support_status}",
                    f"cross_model_support_factor={factor:.3f}",
                    f"prediction_uncertainty_after_cross_model_support={_format_factor_float(adjusted_uncertainty)}",
                ]
            )
            adjusted_candidates.append(
                replace(
                    candidate,
                    prediction_uncertainty_fraction=adjusted_uncertainty,
                    prediction_interval_lower_tokens_per_sec=raw_interval_lower,
                    prediction_interval_upper_tokens_per_sec=raw_interval_upper,
                    risk_flags=sorted({*candidate.risk_flags, support_flag}),
                    decision_factors=decision_factors,
                )
            )
            continue

        adjusted_score = round(candidate.score_risk_adjusted_tokens_per_sec * factor, 3)
        adjusted_efficiency_score = (
            round(adjusted_score / candidate.topology.world_size, 3) if candidate.topology.world_size else None
        )
        risk_interval_lower, risk_interval_upper = _prediction_interval(adjusted_score, adjusted_uncertainty)
        decision_factors = _replace_decision_factor(
            candidate.decision_factors,
            "risk_adjusted_tokens_per_sec=",
            _format_factor_float(adjusted_score),
        )
        if adjusted_efficiency_score is not None:
            decision_factors = _replace_decision_factor(
                decision_factors,
                "risk_adjusted_tokens_per_gpu_per_sec=",
                _format_factor_float(adjusted_efficiency_score),
            )
        decision_factors = _replace_decision_factor(
            decision_factors,
            "prediction_uncertainty_fraction=",
            _format_factor_float(adjusted_uncertainty),
        )
        if raw_interval_lower is not None and raw_interval_upper is not None:
            decision_factors = _replace_decision_factor(
                decision_factors,
                "prediction_interval_tokens_per_sec=",
                f"{_format_factor_float(raw_interval_lower)}..{_format_factor_float(raw_interval_upper)}",
            )
        if risk_interval_lower is not None and risk_interval_upper is not None:
            decision_factors = _replace_decision_factor(
                decision_factors,
                "risk_adjusted_prediction_interval_tokens_per_sec=",
                f"{_format_factor_float(risk_interval_lower)}..{_format_factor_float(risk_interval_upper)}",
            )
        decision_factors.extend(
            [
                f"cross_model_support={support_status}",
                f"cross_model_support_factor={factor:.3f}",
                f"risk_adjusted_after_cross_model_support={_format_factor_float(adjusted_score)}",
                f"prediction_uncertainty_after_cross_model_support={_format_factor_float(adjusted_uncertainty)}",
            ]
        )
        adjusted_candidates.append(
            replace(
                candidate,
                score_risk_adjusted_tokens_per_sec=adjusted_score,
                score_risk_adjusted_tokens_per_gpu_per_sec=adjusted_efficiency_score,
                prediction_uncertainty_fraction=adjusted_uncertainty,
                prediction_interval_lower_tokens_per_sec=raw_interval_lower,
                prediction_interval_upper_tokens_per_sec=raw_interval_upper,
                risk_adjusted_prediction_interval_lower_tokens_per_sec=risk_interval_lower,
                risk_adjusted_prediction_interval_upper_tokens_per_sec=risk_interval_upper,
                risk_flags=sorted({*candidate.risk_flags, support_flag}),
                decision_factors=decision_factors,
                notes=[
                    *candidate.notes,
                    f"cross_model_support_factor={factor:.3f}:{support_status}",
                ],
            )
        )
    return adjusted_candidates


def _format_count_summary(counts: dict[str, int]) -> str:
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _routing_regime(behavior: BenchmarkBehaviorPrediction) -> str:
    if behavior.balanced_routing is True:
        return "balanced_synthetic_routing"
    if behavior.balanced_routing is False:
        return "real_routing"
    return "unknown_routing"


def _routing_regime_status(counts: dict[str, int]) -> str:
    if not counts:
        return "no_candidates"
    known = [regime for regime in counts if regime != "unknown_routing"]
    if not known:
        return "unknown_routing_regime"
    if len(known) == 1 and ("unknown_routing" not in counts):
        return "single_routing_regime"
    return "mixed_routing_regime"


def _scenario_measurement_guidance(
    *,
    candidates: list[ScenarioCandidate],
    scored_count: int,
    memory_blocked_count: int,
    parallelism_tradeoff_status: str,
    parallelism_optimality_status: str,
    parallelism_optimality_blockers: list[str],
    parallelism_boundary_status: str,
    parallelism_boundary_prediction_status: str,
    parallelism_boundary_prediction_blockers: list[str],
    throughput_efficiency_tradeoff_status: str,
    throughput_efficiency_frontier_count: int,
    risk_adjusted_efficiency_frontier_count: int,
    raw_dominated_candidate_count: int,
    risk_adjusted_dominated_candidate_count: int,
    parallelism_axis_comparison_count: int,
    isolated_parallelism_axis_comparison_count: int,
    coupled_parallelism_axis_comparison_count: int,
    parallelism_axis_interval_overlap_count: int,
    blocked_parallelism_axis_names: list[str],
    confounded_parallelism_axis_names: list[str],
    same_workload_scaling_status: str,
    same_workload_scaling_candidate_count: int,
    min_scaling_efficiency: float | None,
    memory_coverage_status_counts: dict[str, int],
    timing_coverage_status_counts: dict[str, int],
    max_memory_residual_gb: float | None,
    phase_bottleneck_bucket_counts: dict[str, int],
    max_phase_bottleneck_share: float | None,
    max_phase_bottleneck_half_speedup_delta_pct: float | None,
    memory_bottleneck_bucket_counts: dict[str, int],
    max_memory_bottleneck_fraction_of_peak: float | None,
    high_uncertainty_candidate_count: int,
    max_prediction_uncertainty_fraction: float | None,
    risk_adjusted_interval_overlap_status: str,
    risk_adjusted_interval_overlap_contender_count: int,
    risk_adjusted_interval_best_vs_next_margin_tokens_per_sec: float | None,
    routing_regime_status: str,
    routing_regime_counts: dict[str, int],
    cross_model_analog_support_status: str,
    cross_model_analog_factor_status: str,
    cross_model_analog_unique_factor_count: int,
    cross_model_analog_unique_target_runtime_signature_count: int,
    cross_model_analog_scored_varied_parallelism_dimensions: list[str],
    cross_model_analog_scored_varied_workload_dimensions: list[str],
    model_generalization_status: str,
    model_generalization_blockers: list[str],
    promotion_readiness_status: str,
    promotable_raw_gap_tokens_per_sec: float | None,
    promotable_risk_adjusted_gap_tokens_per_sec: float | None,
    best_risk_adjusted: ScenarioCandidate | None,
    best_next_measurement: ScenarioCandidate | None,
    best_promotable: ScenarioCandidate | None,
) -> tuple[str, list[str]]:
    focus = best_next_measurement or best_promotable or best_risk_adjusted
    rationale = [f"parallelism_tradeoff={parallelism_tradeoff_status}"]
    rationale.append(f"parallelism_optimality={parallelism_optimality_status}")
    if parallelism_optimality_blockers:
        rationale.append(f"parallelism_optimality_blockers={','.join(parallelism_optimality_blockers)}")
    rationale.append(f"parallelism_boundary={parallelism_boundary_status}")
    rationale.append(f"parallelism_boundary_prediction={parallelism_boundary_prediction_status}")
    if parallelism_boundary_prediction_blockers:
        rationale.append(
            f"parallelism_boundary_prediction_blockers={','.join(parallelism_boundary_prediction_blockers)}"
        )
    rationale.append(f"throughput_efficiency_tradeoff={throughput_efficiency_tradeoff_status}")
    rationale.append(f"throughput_efficiency_frontier_count={throughput_efficiency_frontier_count}")
    rationale.append(f"risk_adjusted_efficiency_frontier_count={risk_adjusted_efficiency_frontier_count}")
    if raw_dominated_candidate_count:
        rationale.append(f"raw_dominated_candidate_count={raw_dominated_candidate_count}")
    if risk_adjusted_dominated_candidate_count:
        rationale.append(f"risk_adjusted_dominated_candidate_count={risk_adjusted_dominated_candidate_count}")
    if parallelism_axis_comparison_count:
        rationale.append(f"parallelism_axis_comparison_count={parallelism_axis_comparison_count}")
        rationale.append(f"isolated_parallelism_axis_comparison_count={isolated_parallelism_axis_comparison_count}")
        rationale.append(f"coupled_parallelism_axis_comparison_count={coupled_parallelism_axis_comparison_count}")
    if parallelism_axis_interval_overlap_count:
        rationale.append(f"parallelism_axis_interval_overlap_count={parallelism_axis_interval_overlap_count}")
    if blocked_parallelism_axis_names:
        rationale.append(f"blocked_parallelism_axes={','.join(blocked_parallelism_axis_names)}")
    if confounded_parallelism_axis_names:
        rationale.append(f"confounded_parallelism_axes={','.join(confounded_parallelism_axis_names)}")
    rationale.append(f"same_workload_scaling={same_workload_scaling_status}")
    if same_workload_scaling_candidate_count:
        rationale.append(f"same_workload_scaling_candidate_count={same_workload_scaling_candidate_count}")
    if min_scaling_efficiency is not None:
        rationale.append(f"min_scaling_efficiency={min_scaling_efficiency:.3f}")
    if memory_coverage_status_counts:
        rationale.append(f"memory_coverage={_format_count_summary(memory_coverage_status_counts)}")
    if timing_coverage_status_counts:
        rationale.append(f"timing_coverage={_format_count_summary(timing_coverage_status_counts)}")
    if max_memory_residual_gb is not None:
        rationale.append(f"max_memory_residual_gb={max_memory_residual_gb:.3f}")
    if phase_bottleneck_bucket_counts:
        rationale.append(f"phase_bottlenecks={_format_count_summary(phase_bottleneck_bucket_counts)}")
    if max_phase_bottleneck_share is not None:
        rationale.append(f"max_phase_bottleneck_share={max_phase_bottleneck_share:.3f}")
    if max_phase_bottleneck_half_speedup_delta_pct is not None:
        rationale.append(
            f"max_phase_bottleneck_half_speedup_delta_pct={max_phase_bottleneck_half_speedup_delta_pct:.3f}"
        )
    if memory_bottleneck_bucket_counts:
        rationale.append(f"memory_bottlenecks={_format_count_summary(memory_bottleneck_bucket_counts)}")
    if max_memory_bottleneck_fraction_of_peak is not None:
        rationale.append(f"max_memory_bottleneck_fraction_of_peak={max_memory_bottleneck_fraction_of_peak:.3f}")
    if high_uncertainty_candidate_count:
        rationale.append(f"high_uncertainty_candidate_count={high_uncertainty_candidate_count}")
    if max_prediction_uncertainty_fraction is not None:
        rationale.append(f"max_prediction_uncertainty_fraction={max_prediction_uncertainty_fraction:.3f}")
    if risk_adjusted_interval_overlap_status != "unknown":
        rationale.append(f"risk_adjusted_interval_overlap={risk_adjusted_interval_overlap_status}")
    if risk_adjusted_interval_overlap_contender_count:
        rationale.append(
            f"risk_adjusted_interval_overlap_contender_count={risk_adjusted_interval_overlap_contender_count}"
        )
    if risk_adjusted_interval_best_vs_next_margin_tokens_per_sec is not None:
        rationale.append(
            "risk_adjusted_interval_best_vs_next_margin_tokens_per_sec="
            f"{risk_adjusted_interval_best_vs_next_margin_tokens_per_sec:.3f}"
        )
    if routing_regime_status == "mixed_routing_regime":
        rationale.append(f"routing_regime={routing_regime_status}")
        rationale.append(f"routing_regimes={_format_count_summary(routing_regime_counts)}")
    if cross_model_analog_support_status != "not_used":
        rationale.append(f"cross_model_analog_support={cross_model_analog_support_status}")
    if cross_model_analog_factor_status not in {"not_used", "no_scored_cross_model_factors"}:
        rationale.append(f"cross_model_analog_factor_status={cross_model_analog_factor_status}")
        rationale.append(f"cross_model_analog_unique_factor_count={cross_model_analog_unique_factor_count}")
    if cross_model_analog_unique_target_runtime_signature_count > 1:
        rationale.append(
            "cross_model_analog_unique_target_runtime_signature_count="
            f"{cross_model_analog_unique_target_runtime_signature_count}"
        )
    if cross_model_analog_scored_varied_parallelism_dimensions:
        rationale.append(
            f"cross_model_analog_varied_parallelism={','.join(cross_model_analog_scored_varied_parallelism_dimensions)}"
        )
    if cross_model_analog_scored_varied_workload_dimensions:
        rationale.append(
            f"cross_model_analog_varied_workload={','.join(cross_model_analog_scored_varied_workload_dimensions)}"
        )
    rationale.append(f"model_generalization={model_generalization_status}")
    if model_generalization_blockers:
        rationale.append(f"model_generalization_blockers={','.join(model_generalization_blockers)}")
    if best_risk_adjusted is not None:
        rationale.append(f"best_risk_adjusted={best_risk_adjusted.label}")
    if best_promotable is not None:
        rationale.append(f"best_promotable={best_promotable.label}")
    rationale.append(f"promotion_readiness={promotion_readiness_status}")
    if promotable_raw_gap_tokens_per_sec is not None:
        rationale.append(f"promotable_raw_gap_tokens_per_sec={promotable_raw_gap_tokens_per_sec:.3f}")
    if promotable_risk_adjusted_gap_tokens_per_sec is not None:
        rationale.append(
            f"promotable_risk_adjusted_gap_tokens_per_sec={promotable_risk_adjusted_gap_tokens_per_sec:.3f}"
        )
    if best_next_measurement is not None:
        rationale.append(f"best_next_measurement={best_next_measurement.label}")
    if focus is not None:
        rationale.append(f"focused_recommendation={focus.recommendation}")
        if focus.risk_flags:
            rationale.append(f"focused_risks={','.join(focus.risk_flags)}")

    if not candidates:
        return "no_candidates", rationale
    if best_next_measurement is not None:
        if parallelism_tradeoff_status == "scored_parallelism_tradeoff_requires_remeasurement":
            return "measure_parallelism_tradeoff_candidate", rationale
        if "cross_model_analog" in best_next_measurement.risk_flags:
            return "measure_cross_model_analog_candidate", rationale
        if best_next_measurement.memory_coverage_status == "analytic_floor_only":
            return "measure_analytic_floor_candidate", rationale
        return "measure_best_next_candidate", rationale
    if best_promotable is not None:
        if parallelism_tradeoff_status == "promotable_parallelism_tradeoff":
            return "promote_parallelism_tradeoff_winner", rationale
        return "promote_best_promotable_candidate", rationale
    if best_risk_adjusted is not None:
        if best_risk_adjusted.recommendation == "debug_runtime_failure":
            return "debug_best_risk_adjusted_candidate", rationale
        if best_risk_adjusted.recommendation == "correctness_gate_required":
            return "correctness_gate_best_risk_adjusted_candidate", rationale
        if best_risk_adjusted.recommendation.startswith("remeasure"):
            return "measure_best_risk_adjusted_candidate", rationale
        return "review_best_risk_adjusted_candidate", rationale
    if memory_blocked_count == len(candidates):
        if any(candidate.feasibility_status == "memory_floor_exceeds_safety_margin" for candidate in candidates):
            return "measure_memory_safety_margin_candidate", rationale
        if memory_coverage_status_counts == {"analytic_floor_only": len(candidates)}:
            return "blocked_by_analytic_memory_floor", rationale
        return "blocked_by_memory_model", rationale
    if scored_count == 0:
        return "no_scored_candidate", rationale
    return "review_scenario", rationale


def _count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _count_int_values(values: list[int | None]) -> dict[int, int]:
    return dict(sorted(Counter(value for value in values if value is not None).items()))


def _count_candidate_model_refs(candidates: list[ScenarioCandidate]) -> dict[str, int]:
    return _count_values([candidate.behavior.model_ref or "unknown_model_ref" for candidate in candidates])


def _count_candidate_runtime_signatures(candidates: list[ScenarioCandidate]) -> dict[str, int]:
    return _count_values([candidate.target_runtime_signature for candidate in candidates])


def _count_candidate_topology_values(candidates: list[ScenarioCandidate], dimension: str) -> dict[int, int]:
    return _count_int_values([_candidate_dimension_value(candidate, dimension) for candidate in candidates])


def _cross_node_dimension_counts(candidates: list[ScenarioCandidate]) -> dict[str, int]:
    dimensions = [
        dimension
        for candidate in candidates
        if candidate.communication is not None
        for dimension in candidate.communication.cross_node_dimensions
    ]
    return _count_values(dimensions)


def _is_memory_blocked(candidate: ScenarioCandidate) -> bool:
    status = candidate.feasibility_status
    return (
        status == "observed_oom" or status.endswith("_exceeds_limit") or status == "memory_floor_exceeds_safety_margin"
    )


def _risk_adjusted_interval_overlap_summary(
    feasible: list[ScenarioCandidate],
    best_risk_adjusted: ScenarioCandidate | None,
) -> tuple[str, int, list[str], float | None]:
    if best_risk_adjusted is None:
        return "no_scored_interval", 0, [], None
    best_lower = best_risk_adjusted.risk_adjusted_prediction_interval_lower_tokens_per_sec
    best_upper = best_risk_adjusted.risk_adjusted_prediction_interval_upper_tokens_per_sec
    if best_lower is None or best_upper is None:
        return "no_scored_interval", 0, [], None

    overlapping_labels: list[str] = []
    contender_upper_bounds: list[float] = []
    best_strategy = _parallelism_strategy_key(best_risk_adjusted)
    for candidate in feasible:
        if candidate.label == best_risk_adjusted.label:
            continue
        # Same-strategy candidates are re-measurements of the SAME parallelism choice (winner reruns,
        # instrumented reruns, historical twins); their interval overlap cannot make the strategy
        # SELECTION uncertain. Only overlap with a genuinely different strategy blocks the choice.
        if _parallelism_strategy_key(candidate) == best_strategy:
            continue
        contender_lower = candidate.risk_adjusted_prediction_interval_lower_tokens_per_sec
        contender_upper = candidate.risk_adjusted_prediction_interval_upper_tokens_per_sec
        if contender_lower is None or contender_upper is None:
            continue
        contender_upper_bounds.append(contender_upper)
        if contender_lower <= best_upper and contender_upper >= best_lower:
            overlapping_labels.append(candidate.label)

    if not contender_upper_bounds:
        return "single_scored_interval", 0, [], None

    margin = round(best_lower - max(contender_upper_bounds), 3)
    if overlapping_labels:
        return "overlapping_best_interval", len(overlapping_labels), sorted(overlapping_labels), margin
    return "clear_best_interval", 0, [], margin


def _scenario_decision_summary(
    candidates: list[ScenarioCandidate],
    feasible: list[ScenarioCandidate],
    best_raw: ScenarioCandidate | None,
    best_risk_adjusted: ScenarioCandidate | None,
    best_efficiency: ScenarioCandidate | None,
    best_risk_adjusted_efficiency: ScenarioCandidate | None,
    best_next_measurement: ScenarioCandidate | None,
    best_promotable: ScenarioCandidate | None,
    benchmark_support: ScenarioBenchmarkSupport,
) -> ScenarioDecisionSummary:
    risk_flags = [flag for candidate in candidates for flag in candidate.risk_flags]
    distances = [
        candidate.calibration_distance for candidate in candidates if candidate.calibration_distance is not None
    ]
    scored_distances = [
        candidate.calibration_distance for candidate in feasible if candidate.calibration_distance is not None
    ]
    uncertainty_fractions = [
        candidate.prediction_uncertainty_fraction
        for candidate in candidates
        if candidate.prediction_uncertainty_fraction is not None
    ]
    scored_uncertainty_fractions = [
        candidate.prediction_uncertainty_fraction
        for candidate in feasible
        if candidate.prediction_uncertainty_fraction is not None
    ]
    high_uncertainty_candidate_count = sum(
        1
        for candidate in candidates
        if candidate.prediction_uncertainty_fraction is not None and candidate.prediction_uncertainty_fraction >= 0.50
    )
    max_prediction_uncertainty_fraction = round(max(uncertainty_fractions), 3) if uncertainty_fractions else None
    (
        risk_adjusted_interval_overlap_status,
        risk_adjusted_interval_overlap_contender_count,
        risk_adjusted_interval_overlap_contender_labels,
        risk_adjusted_interval_best_vs_next_margin,
    ) = _risk_adjusted_interval_overlap_summary(feasible, best_risk_adjusted)
    memory_residuals = [
        candidate.estimated_memory_residual_gb
        for candidate in candidates
        if candidate.estimated_memory_residual_gb is not None
    ]
    memory_residual_fractions = [
        candidate.estimated_memory_residual_fraction_of_peak
        for candidate in candidates
        if candidate.estimated_memory_residual_fraction_of_peak is not None
    ]
    phase_bottleneck_candidates = [
        candidate for candidate in candidates if candidate.phase_bottleneck_bucket is not None
    ]
    phase_bottleneck_shares = [
        candidate.phase_bottleneck_share
        for candidate in phase_bottleneck_candidates
        if candidate.phase_bottleneck_share is not None
    ]
    phase_bottleneck_bucket_counts = _count_values(
        [
            candidate.phase_bottleneck_bucket
            for candidate in phase_bottleneck_candidates
            if candidate.phase_bottleneck_bucket
        ]
    )
    phase_bottleneck_phase_counts = _count_values(
        [
            candidate.phase_bottleneck_phase
            for candidate in phase_bottleneck_candidates
            if candidate.phase_bottleneck_phase
        ]
    )
    max_phase_bottleneck_share = round(max(phase_bottleneck_shares), 3) if phase_bottleneck_shares else None
    max_phase_bottleneck_candidate = _max_positive_candidate_by_field(candidates, "phase_bottleneck_share")
    phase_bottleneck_half_speedup_candidates = [
        candidate for candidate in candidates if candidate.phase_bottleneck_half_speedup_delta_pct is not None
    ]
    phase_bottleneck_half_speedup_deltas = [
        candidate.phase_bottleneck_half_speedup_delta_pct
        for candidate in phase_bottleneck_half_speedup_candidates
        if candidate.phase_bottleneck_half_speedup_delta_pct is not None
    ]
    max_phase_bottleneck_half_speedup_delta_pct = (
        round(max(phase_bottleneck_half_speedup_deltas), 3) if phase_bottleneck_half_speedup_deltas else None
    )
    max_phase_bottleneck_half_speedup_candidate = _max_positive_candidate_by_field(
        candidates,
        "phase_bottleneck_half_speedup_delta_pct",
    )
    memory_bottleneck_candidates = [
        candidate for candidate in candidates if candidate.memory_bottleneck_bucket is not None
    ]
    memory_bottleneck_fractions = [
        candidate.memory_bottleneck_fraction_of_peak
        for candidate in memory_bottleneck_candidates
        if candidate.memory_bottleneck_fraction_of_peak is not None
    ]
    memory_bottleneck_bucket_counts = _count_values(
        [
            candidate.memory_bottleneck_bucket
            for candidate in memory_bottleneck_candidates
            if candidate.memory_bottleneck_bucket
        ]
    )
    memory_bottleneck_phase_counts = _count_values(
        [
            candidate.memory_bottleneck_phase
            for candidate in memory_bottleneck_candidates
            if candidate.memory_bottleneck_phase
        ]
    )
    max_memory_bottleneck_fraction = round(max(memory_bottleneck_fractions), 3) if memory_bottleneck_fractions else None
    max_memory_bottleneck_candidate = _max_positive_candidate_by_field(
        candidates,
        "memory_bottleneck_fraction_of_peak",
    )
    (
        unique_strategy_count,
        scored_strategy_count,
        promotable_strategy_count,
        requires_remeasurement_strategy_count,
    ) = _parallelism_strategy_counts(candidates)
    scored_count = len(feasible)
    memory_blocked_count = sum(1 for candidate in candidates if _is_memory_blocked(candidate))
    memory_coverage_status_counts = _count_values([candidate.memory_coverage_status for candidate in candidates])
    timing_coverage_status_counts = _count_values([candidate.timing_coverage_status for candidate in candidates])
    simulator_support_status_counts = _count_values([candidate.simulator_support_status for candidate in candidates])
    simulator_support_blocker_counts = _count_values(
        [blocker for candidate in candidates for blocker in candidate.simulator_support_blockers]
    )
    max_memory_residual_gb = round(max(memory_residuals), 3) if memory_residuals else None
    routing_regime_counts = _count_values([_routing_regime(candidate.behavior) for candidate in candidates])
    routing_regime_status = _routing_regime_status(routing_regime_counts)
    parallelism_tradeoff_status = _parallelism_tradeoff_status(
        unique_strategy_count=unique_strategy_count,
        scored_strategy_count=scored_strategy_count,
        promotable_strategy_count=promotable_strategy_count,
        requires_remeasurement_strategy_count=requires_remeasurement_strategy_count,
    )
    (
        cross_model_analog_support_status,
        cross_model_analog_candidate_count,
        cross_model_analog_scored_count,
        cross_model_analog_unique_prediction_count,
        cross_model_analog_unique_matched_label_count,
        cross_model_analog_unique_target_strategy_count,
        cross_model_analog_unique_target_runtime_signature_count,
    ) = _cross_model_analog_support(candidates)
    cross_model_analog_scored_candidates = [
        candidate
        for candidate in _cross_model_analog_candidates(candidates)
        if candidate.score_tokens_per_sec is not None
    ]
    cross_model_analog_scored_varied_parallelism_dimensions = _varied_candidate_dimensions(
        cross_model_analog_scored_candidates, _SCENARIO_PARALLELISM_DIMENSIONS
    )
    cross_model_analog_scored_varied_workload_dimensions = _varied_candidate_dimensions(
        cross_model_analog_scored_candidates, _SCENARIO_WORKLOAD_DIMENSIONS
    )
    cross_model_analog_factor_status = _cross_model_analog_factor_status(candidates)
    cross_model_analog_unique_factor_count = _cross_model_analog_unique_factor_count(candidates)
    cross_model_analog_factor_ranges = _cross_model_analog_factor_ranges(candidates)
    cross_model_analog_prediction_interval_top = _cross_model_analog_prediction_interval_top(candidates)
    cross_model_analog_prediction_interval_top_count = len(cross_model_analog_prediction_interval_top)
    cross_model_analog_prediction_interval_top_fraction = (
        round(cross_model_analog_prediction_interval_top_count / cross_model_analog_scored_count, 3)
        if cross_model_analog_scored_count
        else None
    )
    if cross_model_analog_candidate_count == 0:
        cross_model_analog_prediction_interval_selectivity_status = "not_used"
    else:
        cross_model_analog_prediction_interval_selectivity_status = _prediction_interval_selectivity_status(
            scored_candidate_count=cross_model_analog_scored_count,
            prediction_interval_top_count=cross_model_analog_prediction_interval_top_count,
        )
    cross_model_analog_prediction_interval_top_labels = [
        candidate.label for candidate in cross_model_analog_prediction_interval_top
    ]
    model_generalization_status, model_generalization_blockers = _model_generalization_support(
        candidates,
        memory_blocked_count=memory_blocked_count,
        cross_model_analog_support_status=cross_model_analog_support_status,
        cross_model_analog_scored_count=cross_model_analog_scored_count,
        cross_model_analog_factor_status=cross_model_analog_factor_status,
        risk_adjusted_interval_overlap_status=risk_adjusted_interval_overlap_status,
        cross_model_analog_prediction_interval_selectivity_status=(
            cross_model_analog_prediction_interval_selectivity_status
        ),
    )
    scenario_prediction_fidelity_status, scenario_prediction_fidelity_blockers = _scenario_prediction_fidelity_support(
        candidates,
        memory_blocked_count=memory_blocked_count,
        phase_bottleneck_candidate_count=len(phase_bottleneck_candidates),
        routing_regime_status=routing_regime_status,
        simulator_support_status_counts=simulator_support_status_counts,
    )
    throughput_efficiency_tradeoff_status = _throughput_efficiency_tradeoff_status(
        best_raw=best_raw,
        best_risk_adjusted=best_risk_adjusted,
        best_efficiency=best_efficiency,
        best_risk_adjusted_efficiency=best_risk_adjusted_efficiency,
    )
    throughput_efficiency_frontier_labels = _throughput_efficiency_frontier_labels(
        feasible,
        throughput_attr="score_tokens_per_sec",
        efficiency_attr="score_tokens_per_gpu_per_sec",
    )
    risk_adjusted_efficiency_frontier_labels = _throughput_efficiency_frontier_labels(
        feasible,
        throughput_attr="score_risk_adjusted_tokens_per_sec",
        efficiency_attr="score_risk_adjusted_tokens_per_gpu_per_sec",
    )
    raw_dominated_candidate_count = sum(1 for candidate in feasible if candidate.raw_dominated_by_label is not None)
    risk_adjusted_dominated_candidate_count = sum(
        1 for candidate in feasible if candidate.risk_adjusted_dominated_by_label is not None
    )
    parallelism_axis_comparisons = _parallelism_axis_comparisons(feasible)
    parallelism_axis_coverage = _scenario_parallelism_axis_coverage(candidates)
    scored_parallelism_axis_names = [
        coverage.axis for coverage in parallelism_axis_coverage if coverage.status == "scored_parallelism_axis"
    ]
    blocked_parallelism_axis_names = [
        coverage.axis for coverage in parallelism_axis_coverage if "blocked" in coverage.status
    ]
    confounded_parallelism_axis_names = [
        coverage.axis for coverage in parallelism_axis_coverage if coverage.status.startswith("confounded_")
    ]
    unscored_parallelism_axis_names = [
        coverage.axis for coverage in parallelism_axis_coverage if coverage.status == "unscored_parallelism_axis"
    ]
    missing_parallelism_axis_names = [
        coverage.axis for coverage in parallelism_axis_coverage if coverage.status == "missing_parallelism_axis"
    ]
    parallelism_axis_coverage_status_counts = _count_values([coverage.status for coverage in parallelism_axis_coverage])
    varied_parallelism_dimensions = _varied_candidate_dimensions(candidates, _SCENARIO_PARALLELISM_DIMENSIONS)
    varied_workload_dimensions = _varied_candidate_dimensions(candidates, _SCENARIO_WORKLOAD_DIMENSIONS)
    varied_runtime_dimensions = _varied_candidate_runtime_dimensions(candidates)
    runtime_mismatch_dimensions = _candidate_runtime_mismatch_dimensions(candidates)
    promotable_labels = {candidate.label for candidate in feasible if candidate.promotable}
    interval_overlap_only_promotable_tie = (
        risk_adjusted_interval_overlap_status == "overlapping_best_interval"
        and best_risk_adjusted is not None
        and best_risk_adjusted.promotable
        and bool(risk_adjusted_interval_overlap_contender_labels)
        and set(risk_adjusted_interval_overlap_contender_labels) <= promotable_labels
    )
    parallelism_optimality_status, parallelism_optimality_blockers = _parallelism_optimality_support(
        unique_strategy_count=unique_strategy_count,
        scored_strategy_count=scored_strategy_count,
        memory_blocked_count=memory_blocked_count,
        best_risk_adjusted=best_risk_adjusted,
        best_promotable=best_promotable,
        risk_adjusted_interval_overlap_status=risk_adjusted_interval_overlap_status,
        interval_overlap_only_promotable_tie=interval_overlap_only_promotable_tie,
        parallelism_tradeoff_status=parallelism_tradeoff_status,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
        varied_runtime_dimensions=varied_runtime_dimensions,
        simulator_support_status_counts=simulator_support_status_counts,
    )
    scenario_capture_status, scenario_capture_blockers = _scenario_capture_support(
        candidate_count=len(candidates),
        scored_count=scored_count,
        memory_blocked_count=memory_blocked_count,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        runtime_mismatch_dimensions=runtime_mismatch_dimensions,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
        simulator_support_status_counts=simulator_support_status_counts,
    )
    scenario_capture_gaps = _scenario_capture_gap_portfolio(
        scenario_capture_blockers=scenario_capture_blockers,
        candidate_count=len(candidates),
        scored_count=scored_count,
        memory_blocked_count=memory_blocked_count,
        missing_parallelism_axis_names=missing_parallelism_axis_names,
        blocked_parallelism_axis_names=blocked_parallelism_axis_names,
        confounded_parallelism_axis_names=confounded_parallelism_axis_names,
        unscored_parallelism_axis_names=unscored_parallelism_axis_names,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        runtime_mismatch_dimensions=runtime_mismatch_dimensions,
    )
    scenario_capture_gap_status_counts = _count_values([gap.gap_status for gap in scenario_capture_gaps])
    scenario_capture_gap_required_measurements = _unique_in_order(
        [gap.required_measurement for gap in scenario_capture_gaps]
    )
    parallelism_boundary_groups = _scenario_parallelism_boundary_groups(candidates)
    parallelism_boundary_status = _scenario_parallelism_boundary_status(parallelism_boundary_groups, candidates)
    parallelism_boundary_axis_coverage = _scenario_parallelism_boundary_axis_coverage(parallelism_boundary_groups)
    parallelism_boundary_measured_axis_names = [
        coverage.axis
        for coverage in parallelism_boundary_axis_coverage
        if coverage.status == "measured_parallelism_boundary_axis"
    ]
    parallelism_boundary_confounded_axis_names = [
        coverage.axis
        for coverage in parallelism_boundary_axis_coverage
        if coverage.status == "confounded_parallelism_boundary_axis"
    ]
    parallelism_boundary_missing_axis_names = [
        coverage.axis
        for coverage in parallelism_boundary_axis_coverage
        if coverage.status == "missing_parallelism_boundary_axis"
    ]
    parallelism_boundary_axis_coverage_status_counts = _count_values(
        [coverage.status for coverage in parallelism_boundary_axis_coverage]
    )
    parallelism_boundary_candidates = [
        candidate for candidate in candidates if _scenario_boundary_outcome(candidate) is not None
    ]
    parallelism_boundary_fit_count = sum(
        1 for candidate in parallelism_boundary_candidates if _scenario_boundary_outcome(candidate) == "fit"
    )
    parallelism_boundary_failure_count = sum(
        1 for candidate in parallelism_boundary_candidates if _scenario_boundary_outcome(candidate) == "failure"
    )
    parallelism_boundary_best_fit = max(
        (group for group in parallelism_boundary_groups if group.best_fit_tokens_per_sec is not None),
        key=lambda group: (group.best_fit_tokens_per_sec or float("-inf"), group.best_fit_label or ""),
        default=None,
    )
    parallelism_boundary_confounded_dimensions = sorted(
        {
            dimension
            for group in parallelism_boundary_groups
            for dimension in [*group.confounded_workload_dimensions, *group.confounded_runtime_dimensions]
        }
    )
    (
        parallelism_boundary_prediction_status,
        parallelism_boundary_prediction_blockers,
    ) = _scenario_parallelism_boundary_prediction_support(
        parallelism_boundary_status=parallelism_boundary_status,
        parallelism_boundary_fit_count=parallelism_boundary_fit_count,
        parallelism_boundary_failure_count=parallelism_boundary_failure_count,
        parallelism_boundary_measured_axis_names=parallelism_boundary_measured_axis_names,
        parallelism_boundary_confounded_axis_names=parallelism_boundary_confounded_axis_names,
        parallelism_boundary_missing_axis_names=parallelism_boundary_missing_axis_names,
        parallelism_boundary_confounded_dimensions=parallelism_boundary_confounded_dimensions,
    )
    isolated_parallelism_axis_comparison_count = sum(
        1 for comparison in parallelism_axis_comparisons if comparison.coupling_status == "isolated_axis_comparison"
    )
    coupled_parallelism_axis_comparison_count = sum(
        1 for comparison in parallelism_axis_comparisons if comparison.coupling_status == "coupled_axis_comparison"
    )
    parallelism_axis_interval_overlap_count = sum(
        1
        for comparison in parallelism_axis_comparisons
        if comparison.risk_adjusted_interval_overlap_status == "overlapping_best_interval"
    )
    scaling_candidates = [
        candidate
        for candidate in feasible
        if candidate.scaling_efficiency is not None
        and candidate.scaling_gpu_ratio is not None
        and candidate.scaling_gpu_ratio > 1.0
    ]
    risk_adjusted_scaling_candidates = [
        candidate for candidate in scaling_candidates if candidate.risk_adjusted_scaling_efficiency is not None
    ]
    best_scaling_efficiency = (
        max(scaling_candidates, key=lambda candidate: (candidate.scaling_efficiency or float("-inf"), candidate.label))
        if scaling_candidates
        else None
    )
    best_risk_adjusted_scaling_efficiency = (
        max(
            risk_adjusted_scaling_candidates,
            key=lambda candidate: (candidate.risk_adjusted_scaling_efficiency or float("-inf"), candidate.label),
        )
        if risk_adjusted_scaling_candidates
        else None
    )
    scaling_efficiencies = [
        candidate.scaling_efficiency for candidate in scaling_candidates if candidate.scaling_efficiency is not None
    ]
    risk_adjusted_scaling_efficiencies = [
        candidate.risk_adjusted_scaling_efficiency
        for candidate in risk_adjusted_scaling_candidates
        if candidate.risk_adjusted_scaling_efficiency is not None
    ]
    same_workload_scaling_status = _same_workload_scaling_status(scaling_candidates)
    promotion_readiness_status = _scenario_promotion_readiness_status(
        best_raw=best_raw,
        best_risk_adjusted=best_risk_adjusted,
        best_promotable=best_promotable,
    )
    promotable_raw_gap_tokens_per_sec = _candidate_score_gap(
        best_raw,
        best_promotable,
        "score_tokens_per_sec",
    )
    promotable_raw_gap_percentage = _candidate_score_gap_percentage(
        best_raw,
        "score_tokens_per_sec",
        promotable_raw_gap_tokens_per_sec,
    )
    promotable_risk_adjusted_gap_tokens_per_sec = _candidate_score_gap(
        best_risk_adjusted,
        best_promotable,
        "score_risk_adjusted_tokens_per_sec",
    )
    promotable_risk_adjusted_gap_percentage = _candidate_score_gap_percentage(
        best_risk_adjusted,
        "score_risk_adjusted_tokens_per_sec",
        promotable_risk_adjusted_gap_tokens_per_sec,
    )
    measurement_readiness_status, measurement_rationale = _scenario_measurement_guidance(
        candidates=candidates,
        scored_count=scored_count,
        memory_blocked_count=memory_blocked_count,
        parallelism_tradeoff_status=parallelism_tradeoff_status,
        parallelism_optimality_status=parallelism_optimality_status,
        parallelism_optimality_blockers=parallelism_optimality_blockers,
        parallelism_boundary_status=parallelism_boundary_status,
        parallelism_boundary_prediction_status=parallelism_boundary_prediction_status,
        parallelism_boundary_prediction_blockers=parallelism_boundary_prediction_blockers,
        throughput_efficiency_tradeoff_status=throughput_efficiency_tradeoff_status,
        throughput_efficiency_frontier_count=len(throughput_efficiency_frontier_labels),
        risk_adjusted_efficiency_frontier_count=len(risk_adjusted_efficiency_frontier_labels),
        raw_dominated_candidate_count=raw_dominated_candidate_count,
        risk_adjusted_dominated_candidate_count=risk_adjusted_dominated_candidate_count,
        parallelism_axis_comparison_count=len(parallelism_axis_comparisons),
        isolated_parallelism_axis_comparison_count=isolated_parallelism_axis_comparison_count,
        coupled_parallelism_axis_comparison_count=coupled_parallelism_axis_comparison_count,
        parallelism_axis_interval_overlap_count=parallelism_axis_interval_overlap_count,
        blocked_parallelism_axis_names=blocked_parallelism_axis_names,
        confounded_parallelism_axis_names=confounded_parallelism_axis_names,
        same_workload_scaling_status=same_workload_scaling_status,
        same_workload_scaling_candidate_count=len(scaling_candidates),
        min_scaling_efficiency=round(min(scaling_efficiencies), 3) if scaling_efficiencies else None,
        memory_coverage_status_counts=memory_coverage_status_counts,
        timing_coverage_status_counts=timing_coverage_status_counts,
        max_memory_residual_gb=max_memory_residual_gb,
        phase_bottleneck_bucket_counts=phase_bottleneck_bucket_counts,
        max_phase_bottleneck_share=max_phase_bottleneck_share,
        max_phase_bottleneck_half_speedup_delta_pct=max_phase_bottleneck_half_speedup_delta_pct,
        memory_bottleneck_bucket_counts=memory_bottleneck_bucket_counts,
        max_memory_bottleneck_fraction_of_peak=max_memory_bottleneck_fraction,
        high_uncertainty_candidate_count=high_uncertainty_candidate_count,
        max_prediction_uncertainty_fraction=max_prediction_uncertainty_fraction,
        risk_adjusted_interval_overlap_status=risk_adjusted_interval_overlap_status,
        risk_adjusted_interval_overlap_contender_count=risk_adjusted_interval_overlap_contender_count,
        risk_adjusted_interval_best_vs_next_margin_tokens_per_sec=risk_adjusted_interval_best_vs_next_margin,
        routing_regime_status=routing_regime_status,
        routing_regime_counts=routing_regime_counts,
        cross_model_analog_support_status=cross_model_analog_support_status,
        cross_model_analog_factor_status=cross_model_analog_factor_status,
        cross_model_analog_unique_factor_count=cross_model_analog_unique_factor_count,
        cross_model_analog_unique_target_runtime_signature_count=(
            cross_model_analog_unique_target_runtime_signature_count
        ),
        cross_model_analog_scored_varied_parallelism_dimensions=(
            cross_model_analog_scored_varied_parallelism_dimensions
        ),
        cross_model_analog_scored_varied_workload_dimensions=cross_model_analog_scored_varied_workload_dimensions,
        model_generalization_status=model_generalization_status,
        model_generalization_blockers=model_generalization_blockers,
        promotion_readiness_status=promotion_readiness_status,
        promotable_raw_gap_tokens_per_sec=promotable_raw_gap_tokens_per_sec,
        promotable_risk_adjusted_gap_tokens_per_sec=promotable_risk_adjusted_gap_tokens_per_sec,
        best_risk_adjusted=best_risk_adjusted,
        best_next_measurement=best_next_measurement,
        best_promotable=best_promotable,
    )
    (
        measurement_candidate_labels,
        measurement_candidate_reasons,
        measurement_candidate_priority_scores,
        measurement_candidate_priority_per_gpu,
        measurement_candidate_cost_gpus,
        measurement_candidate_priority_factors,
    ) = _measurement_portfolio(
        candidates=candidates,
        throughput_efficiency_frontier_labels=throughput_efficiency_frontier_labels,
        risk_adjusted_efficiency_frontier_labels=risk_adjusted_efficiency_frontier_labels,
        parallelism_axis_coverage=parallelism_axis_coverage,
        parallelism_tradeoff_status=parallelism_tradeoff_status,
        cross_model_analog_support_status=cross_model_analog_support_status,
        cross_model_analog_prediction_interval_selectivity_status=(
            cross_model_analog_prediction_interval_selectivity_status
        ),
        cross_model_analog_prediction_interval_top_labels=cross_model_analog_prediction_interval_top_labels,
        same_workload_scaling_status=same_workload_scaling_status,
        best_raw=best_raw,
        best_risk_adjusted=best_risk_adjusted,
        best_efficiency=best_efficiency,
        best_risk_adjusted_efficiency=best_risk_adjusted_efficiency,
        best_next_measurement=best_next_measurement,
        best_promotable=best_promotable,
    )
    measurement_candidate_config_overrides = _measurement_candidate_config_overrides(measurement_candidate_reasons)
    (
        measurement_portfolio_coverage_status,
        measurement_portfolio_coverage_blockers,
        measurement_portfolio_reason_category_counts,
        measurement_portfolio_parallelism_axis_gap_names,
        measurement_portfolio_cross_model_analog_count,
    ) = _measurement_portfolio_coverage_status(
        candidate_reasons=measurement_candidate_reasons,
        best_next_measurement=best_next_measurement,
        parallelism_tradeoff_status=parallelism_tradeoff_status,
        throughput_efficiency_tradeoff_status=throughput_efficiency_tradeoff_status,
        same_workload_scaling_status=same_workload_scaling_status,
        cross_model_analog_support_status=cross_model_analog_support_status,
    )
    validation_actions = _scenario_validation_actions(
        candidate_reasons=measurement_candidate_reasons,
        candidate_priority_scores=measurement_candidate_priority_scores,
        candidate_priority_per_gpu=measurement_candidate_priority_per_gpu,
        candidate_cost_gpus=measurement_candidate_cost_gpus,
        candidate_config_overrides=measurement_candidate_config_overrides,
    )
    validation_action_status_counts = _count_values([action.action_status for action in validation_actions])
    validation_action_required_measurements = _unique_in_order(
        [action.required_measurement for action in validation_actions]
    )
    measurement_rationale = [
        *measurement_rationale,
        f"scenario_capture={scenario_capture_status}",
        f"benchmark_support={benchmark_support.support_status}",
        f"scenario_prediction_fidelity={scenario_prediction_fidelity_status}",
        f"measurement_portfolio_coverage={measurement_portfolio_coverage_status}",
        "simulator_support_status_counts="
        + ",".join(f"{status}:{count}" for status, count in simulator_support_status_counts.items()),
    ]
    if benchmark_support.support_blockers:
        measurement_rationale.append(f"benchmark_support_blockers={','.join(benchmark_support.support_blockers)}")
    if scenario_capture_blockers:
        measurement_rationale.append(f"scenario_capture_blockers={','.join(scenario_capture_blockers)}")
    if scenario_capture_gaps:
        measurement_rationale.append(
            "scenario_capture_gaps="
            + ",".join(f"{status}:{count}" for status, count in scenario_capture_gap_status_counts.items())
        )
        measurement_rationale.append(
            "scenario_capture_required_measurements=" + ",".join(scenario_capture_gap_required_measurements)
        )
    if varied_runtime_dimensions:
        measurement_rationale.append(f"varied_runtime_dimensions={','.join(varied_runtime_dimensions)}")
    if runtime_mismatch_dimensions:
        measurement_rationale.append(f"runtime_mismatch_dimensions={','.join(runtime_mismatch_dimensions)}")
    if cross_model_analog_prediction_interval_selectivity_status not in {
        "not_used",
        "no_scored_cross_model_candidates",
    }:
        measurement_rationale.append(
            "cross_model_analog_prediction_interval_selectivity="
            f"{cross_model_analog_prediction_interval_selectivity_status}"
        )
    if scenario_prediction_fidelity_blockers:
        measurement_rationale.append(
            f"scenario_prediction_fidelity_blockers={','.join(scenario_prediction_fidelity_blockers)}"
        )
    if measurement_portfolio_coverage_blockers:
        measurement_rationale.append(
            f"measurement_portfolio_coverage_blockers={','.join(measurement_portfolio_coverage_blockers)}"
        )
    if measurement_portfolio_parallelism_axis_gap_names:
        measurement_rationale.append(
            f"measurement_portfolio_parallelism_axis_gaps={','.join(measurement_portfolio_parallelism_axis_gap_names)}"
        )
    if validation_actions:
        measurement_rationale.append(
            "validation_actions="
            + ",".join(f"{status}:{count}" for status, count in validation_action_status_counts.items())
        )
        measurement_rationale.append(
            "validation_action_required_measurements=" + ",".join(validation_action_required_measurements)
        )
    measurement_portfolio_total_gpu_count = sum(measurement_candidate_cost_gpus.values())
    prediction_confidence_counts = _count_values([candidate.prediction_confidence for candidate in candidates])
    calibration_scope_counts = _count_values([candidate.calibration_scope for candidate in candidates])
    scenario_readiness = _scenario_readiness(
        candidate_count=len(candidates),
        scored_count=scored_count,
        unscored_count=len(candidates) - scored_count,
        memory_blocked_count=memory_blocked_count,
        unique_parallelism_strategy_count=unique_strategy_count,
        scored_parallelism_strategy_count=scored_strategy_count,
        promotable_parallelism_strategy_count=promotable_strategy_count,
        scenario_capture_status=scenario_capture_status,
        scenario_capture_blockers=scenario_capture_blockers,
        scenario_prediction_fidelity_status=scenario_prediction_fidelity_status,
        scenario_prediction_fidelity_blockers=scenario_prediction_fidelity_blockers,
        parallelism_optimality_status=parallelism_optimality_status,
        parallelism_optimality_blockers=parallelism_optimality_blockers,
        model_generalization_status=model_generalization_status,
        model_generalization_blockers=model_generalization_blockers,
        measurement_readiness_status=measurement_readiness_status,
        measurement_portfolio_coverage_status=measurement_portfolio_coverage_status,
        measurement_portfolio_coverage_blockers=measurement_portfolio_coverage_blockers,
        scenario_capture_gaps=scenario_capture_gaps,
        scenario_capture_gap_status_counts=scenario_capture_gap_status_counts,
        scenario_capture_gap_required_measurements=scenario_capture_gap_required_measurements,
        validation_actions=validation_actions,
        validation_action_status_counts=validation_action_status_counts,
        validation_action_required_measurements=validation_action_required_measurements,
        validation_action_total_gpu_count=measurement_portfolio_total_gpu_count,
        measurement_candidate_labels=measurement_candidate_labels,
        measurement_portfolio_total_gpu_count=measurement_portfolio_total_gpu_count,
        measurement_portfolio_parallelism_axis_gap_names=measurement_portfolio_parallelism_axis_gap_names,
        measurement_portfolio_cross_model_analog_count=measurement_portfolio_cross_model_analog_count,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        runtime_mismatch_dimensions=runtime_mismatch_dimensions,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
        scored_parallelism_axis_names=scored_parallelism_axis_names,
        blocked_parallelism_axis_names=blocked_parallelism_axis_names,
        confounded_parallelism_axis_names=confounded_parallelism_axis_names,
        unscored_parallelism_axis_names=unscored_parallelism_axis_names,
        missing_parallelism_axis_names=missing_parallelism_axis_names,
        simulator_support_status_counts=simulator_support_status_counts,
        prediction_confidence_counts=prediction_confidence_counts,
        calibration_scope_counts=calibration_scope_counts,
        memory_coverage_status_counts=memory_coverage_status_counts,
        timing_coverage_status_counts=timing_coverage_status_counts,
        cross_model_analog_support_status=cross_model_analog_support_status,
        cross_model_analog_candidate_count=cross_model_analog_candidate_count,
        cross_model_analog_scored_count=cross_model_analog_scored_count,
        benchmark_support=benchmark_support,
    )
    return ScenarioDecisionSummary(
        candidate_count=len(candidates),
        scored_count=scored_count,
        unscored_count=len(candidates) - scored_count,
        feasible_count=scored_count,
        promotable_count=sum(1 for candidate in feasible if candidate.promotable),
        requires_remeasurement_count=sum(
            1 for candidate in candidates if "requires_remeasurement" in candidate.risk_flags
        ),
        memory_blocked_count=memory_blocked_count,
        unique_parallelism_strategy_count=unique_strategy_count,
        best_raw_label=best_raw.label if best_raw is not None else None,
        best_raw_score_tokens_per_sec=best_raw.score_tokens_per_sec if best_raw is not None else None,
        best_risk_adjusted_label=best_risk_adjusted.label if best_risk_adjusted is not None else None,
        best_risk_adjusted_score_tokens_per_sec=(
            best_risk_adjusted.score_risk_adjusted_tokens_per_sec if best_risk_adjusted is not None else None
        ),
        best_efficiency_label=best_efficiency.label if best_efficiency is not None else None,
        best_efficiency_score_tokens_per_gpu_per_sec=(
            best_efficiency.score_tokens_per_gpu_per_sec if best_efficiency is not None else None
        ),
        best_risk_adjusted_efficiency_label=(
            best_risk_adjusted_efficiency.label if best_risk_adjusted_efficiency is not None else None
        ),
        best_risk_adjusted_efficiency_score_tokens_per_gpu_per_sec=(
            best_risk_adjusted_efficiency.score_risk_adjusted_tokens_per_gpu_per_sec
            if best_risk_adjusted_efficiency is not None
            else None
        ),
        best_next_measurement_label=best_next_measurement.label if best_next_measurement is not None else None,
        best_next_measurement_score_tokens_per_sec=(
            best_next_measurement.score_tokens_per_sec if best_next_measurement is not None else None
        ),
        best_promotable_label=best_promotable.label if best_promotable is not None else None,
        candidate_model_ref_counts=_count_candidate_model_refs(candidates),
        scored_model_ref_counts=_count_candidate_model_refs(feasible),
        candidate_world_size_counts=_count_candidate_topology_values(candidates, "world_size"),
        scored_world_size_counts=_count_candidate_topology_values(feasible, "world_size"),
        candidate_sequence_length_counts=_count_candidate_topology_values(candidates, "sample_packing_sequence_len"),
        scored_sequence_length_counts=_count_candidate_topology_values(feasible, "sample_packing_sequence_len"),
        candidate_global_batch_size_counts=_count_candidate_topology_values(candidates, "global_batch_size"),
        scored_global_batch_size_counts=_count_candidate_topology_values(feasible, "global_batch_size"),
        candidate_runtime_signature_counts=_count_candidate_runtime_signatures(candidates),
        scored_runtime_signature_counts=_count_candidate_runtime_signatures(feasible),
        best_promotable_score_tokens_per_sec=(
            best_promotable.score_tokens_per_sec if best_promotable is not None else None
        ),
        best_promotable_score_risk_adjusted_tokens_per_sec=(
            best_promotable.score_risk_adjusted_tokens_per_sec if best_promotable is not None else None
        ),
        promotable_raw_gap_tokens_per_sec=promotable_raw_gap_tokens_per_sec,
        promotable_raw_gap_percentage=promotable_raw_gap_percentage,
        promotable_risk_adjusted_gap_tokens_per_sec=promotable_risk_adjusted_gap_tokens_per_sec,
        promotable_risk_adjusted_gap_percentage=promotable_risk_adjusted_gap_percentage,
        promotion_readiness_status=promotion_readiness_status,
        throughput_efficiency_frontier_labels=throughput_efficiency_frontier_labels,
        risk_adjusted_efficiency_frontier_labels=risk_adjusted_efficiency_frontier_labels,
        throughput_efficiency_frontier_count=len(throughput_efficiency_frontier_labels),
        risk_adjusted_efficiency_frontier_count=len(risk_adjusted_efficiency_frontier_labels),
        raw_dominated_candidate_count=raw_dominated_candidate_count,
        risk_adjusted_dominated_candidate_count=risk_adjusted_dominated_candidate_count,
        throughput_efficiency_tradeoff_status=throughput_efficiency_tradeoff_status,
        same_workload_scaling_status=same_workload_scaling_status,
        same_workload_scaling_group_count=len(
            {candidate.scaling_baseline_label for candidate in scaling_candidates if candidate.scaling_baseline_label}
        ),
        same_workload_scaling_candidate_count=len(scaling_candidates),
        best_scaling_efficiency_label=(best_scaling_efficiency.label if best_scaling_efficiency is not None else None),
        best_scaling_efficiency=(
            best_scaling_efficiency.scaling_efficiency if best_scaling_efficiency is not None else None
        ),
        mean_scaling_efficiency=(
            round(sum(scaling_efficiencies) / len(scaling_efficiencies), 3) if scaling_efficiencies else None
        ),
        min_scaling_efficiency=round(min(scaling_efficiencies), 3) if scaling_efficiencies else None,
        best_risk_adjusted_scaling_efficiency_label=(
            best_risk_adjusted_scaling_efficiency.label if best_risk_adjusted_scaling_efficiency is not None else None
        ),
        best_risk_adjusted_scaling_efficiency=(
            best_risk_adjusted_scaling_efficiency.risk_adjusted_scaling_efficiency
            if best_risk_adjusted_scaling_efficiency is not None
            else None
        ),
        mean_risk_adjusted_scaling_efficiency=(
            round(sum(risk_adjusted_scaling_efficiencies) / len(risk_adjusted_scaling_efficiencies), 3)
            if risk_adjusted_scaling_efficiencies
            else None
        ),
        min_risk_adjusted_scaling_efficiency=(
            round(min(risk_adjusted_scaling_efficiencies), 3) if risk_adjusted_scaling_efficiencies else None
        ),
        measurement_readiness_status=measurement_readiness_status,
        measurement_rationale=measurement_rationale,
        measurement_candidate_count=len(measurement_candidate_labels),
        measurement_candidate_labels=measurement_candidate_labels,
        measurement_candidate_reasons=measurement_candidate_reasons,
        measurement_candidate_priority_scores=measurement_candidate_priority_scores,
        measurement_candidate_priority_per_gpu=measurement_candidate_priority_per_gpu,
        measurement_candidate_cost_gpus=measurement_candidate_cost_gpus,
        measurement_candidate_priority_factors=measurement_candidate_priority_factors,
        measurement_candidate_config_overrides=measurement_candidate_config_overrides,
        measurement_portfolio_total_gpu_count=measurement_portfolio_total_gpu_count,
        measurement_portfolio_max_priority_score=(
            max(measurement_candidate_priority_scores.values()) if measurement_candidate_priority_scores else None
        ),
        measurement_portfolio_max_priority_label=_max_score_label(measurement_candidate_priority_scores),
        measurement_portfolio_max_priority_per_gpu=(
            max(measurement_candidate_priority_per_gpu.values()) if measurement_candidate_priority_per_gpu else None
        ),
        measurement_portfolio_max_priority_per_gpu_label=_max_score_label(measurement_candidate_priority_per_gpu),
        measurement_portfolio_coverage_status=measurement_portfolio_coverage_status,
        measurement_portfolio_coverage_blockers=measurement_portfolio_coverage_blockers,
        measurement_portfolio_reason_category_counts=measurement_portfolio_reason_category_counts,
        measurement_portfolio_parallelism_axis_gap_names=measurement_portfolio_parallelism_axis_gap_names,
        measurement_portfolio_cross_model_analog_count=measurement_portfolio_cross_model_analog_count,
        validation_action_count=len(validation_actions),
        validation_action_status_counts=validation_action_status_counts,
        validation_action_required_measurements=validation_action_required_measurements,
        validation_action_total_gpu_count=measurement_portfolio_total_gpu_count,
        validation_actions=validation_actions,
        max_calibration_distance=round(max(distances), 3) if distances else None,
        max_calibration_distance_label=_max_positive_candidate_label(candidates, "calibration_distance"),
        mean_scored_calibration_distance=(
            round(sum(scored_distances) / len(scored_distances), 3) if scored_distances else None
        ),
        high_uncertainty_candidate_count=high_uncertainty_candidate_count,
        max_prediction_uncertainty_fraction=max_prediction_uncertainty_fraction,
        max_prediction_uncertainty_fraction_label=_max_positive_candidate_label(
            candidates,
            "prediction_uncertainty_fraction",
        ),
        mean_scored_prediction_uncertainty_fraction=(
            round(sum(scored_uncertainty_fractions) / len(scored_uncertainty_fractions), 3)
            if scored_uncertainty_fractions
            else None
        ),
        risk_adjusted_interval_overlap_status=risk_adjusted_interval_overlap_status,
        risk_adjusted_interval_overlap_contender_count=risk_adjusted_interval_overlap_contender_count,
        risk_adjusted_interval_overlap_contender_labels=risk_adjusted_interval_overlap_contender_labels,
        risk_adjusted_interval_best_vs_next_margin_tokens_per_sec=risk_adjusted_interval_best_vs_next_margin,
        parallelism_tradeoff_status=parallelism_tradeoff_status,
        parallelism_optimality_status=parallelism_optimality_status,
        parallelism_optimality_blockers=parallelism_optimality_blockers,
        scored_parallelism_strategy_count=scored_strategy_count,
        promotable_parallelism_strategy_count=promotable_strategy_count,
        requires_remeasurement_parallelism_strategy_count=requires_remeasurement_strategy_count,
        parallelism_axis_comparison_count=len(parallelism_axis_comparisons),
        isolated_parallelism_axis_comparison_count=isolated_parallelism_axis_comparison_count,
        coupled_parallelism_axis_comparison_count=coupled_parallelism_axis_comparison_count,
        parallelism_axis_interval_overlap_count=parallelism_axis_interval_overlap_count,
        parallelism_axis_comparisons=parallelism_axis_comparisons,
        scored_parallelism_axis_names=scored_parallelism_axis_names,
        blocked_parallelism_axis_names=blocked_parallelism_axis_names,
        confounded_parallelism_axis_names=confounded_parallelism_axis_names,
        unscored_parallelism_axis_names=unscored_parallelism_axis_names,
        missing_parallelism_axis_names=missing_parallelism_axis_names,
        parallelism_axis_coverage_status_counts=parallelism_axis_coverage_status_counts,
        parallelism_axis_coverage=parallelism_axis_coverage,
        parallelism_boundary_status=parallelism_boundary_status,
        parallelism_boundary_prediction_status=parallelism_boundary_prediction_status,
        parallelism_boundary_prediction_blockers=parallelism_boundary_prediction_blockers,
        parallelism_boundary_group_count=len(parallelism_boundary_groups),
        parallelism_boundary_candidate_count=len(parallelism_boundary_candidates),
        parallelism_boundary_fit_count=parallelism_boundary_fit_count,
        parallelism_boundary_failure_count=parallelism_boundary_failure_count,
        parallelism_boundary_best_fit_label=(
            parallelism_boundary_best_fit.best_fit_label if parallelism_boundary_best_fit is not None else None
        ),
        parallelism_boundary_confounded_dimensions=parallelism_boundary_confounded_dimensions,
        parallelism_boundary_measured_axis_names=parallelism_boundary_measured_axis_names,
        parallelism_boundary_confounded_axis_names=parallelism_boundary_confounded_axis_names,
        parallelism_boundary_missing_axis_names=parallelism_boundary_missing_axis_names,
        parallelism_boundary_axis_coverage_status_counts=parallelism_boundary_axis_coverage_status_counts,
        parallelism_boundary_axis_coverage=parallelism_boundary_axis_coverage,
        parallelism_boundary_groups=parallelism_boundary_groups,
        cross_model_analog_support_status=cross_model_analog_support_status,
        cross_model_analog_candidate_count=cross_model_analog_candidate_count,
        cross_model_analog_scored_count=cross_model_analog_scored_count,
        cross_model_analog_unique_prediction_count=cross_model_analog_unique_prediction_count,
        cross_model_analog_unique_matched_label_count=cross_model_analog_unique_matched_label_count,
        cross_model_analog_unique_target_strategy_count=cross_model_analog_unique_target_strategy_count,
        cross_model_analog_unique_target_runtime_signature_count=(
            cross_model_analog_unique_target_runtime_signature_count
        ),
        cross_model_analog_scored_varied_parallelism_dimensions=(
            cross_model_analog_scored_varied_parallelism_dimensions
        ),
        cross_model_analog_scored_varied_workload_dimensions=cross_model_analog_scored_varied_workload_dimensions,
        cross_model_analog_factor_status=cross_model_analog_factor_status,
        cross_model_analog_unique_factor_count=cross_model_analog_unique_factor_count,
        cross_model_analog_factor_ranges=cross_model_analog_factor_ranges,
        cross_model_analog_prediction_interval_top_count=cross_model_analog_prediction_interval_top_count,
        cross_model_analog_prediction_interval_top_fraction=cross_model_analog_prediction_interval_top_fraction,
        cross_model_analog_prediction_interval_top_labels=cross_model_analog_prediction_interval_top_labels,
        cross_model_analog_prediction_interval_selectivity_status=(
            cross_model_analog_prediction_interval_selectivity_status
        ),
        model_generalization_status=model_generalization_status,
        model_generalization_blockers=model_generalization_blockers,
        scenario_capture_status=scenario_capture_status,
        scenario_capture_blockers=scenario_capture_blockers,
        scenario_capture_gap_count=len(scenario_capture_gaps),
        scenario_capture_gap_status_counts=scenario_capture_gap_status_counts,
        scenario_capture_gap_required_measurements=scenario_capture_gap_required_measurements,
        scenario_capture_gaps=scenario_capture_gaps,
        benchmark_support=benchmark_support,
        scenario_prediction_fidelity_status=scenario_prediction_fidelity_status,
        scenario_prediction_fidelity_blockers=scenario_prediction_fidelity_blockers,
        varied_parallelism_dimensions=varied_parallelism_dimensions,
        varied_workload_dimensions=varied_workload_dimensions,
        varied_runtime_dimensions=varied_runtime_dimensions,
        runtime_mismatch_dimensions=runtime_mismatch_dimensions,
        prediction_confidence_counts=prediction_confidence_counts,
        calibration_scope_counts=calibration_scope_counts,
        memory_basis_counts=_count_values([candidate.memory_basis for candidate in candidates]),
        memory_coverage_status_counts=memory_coverage_status_counts,
        simulator_support_status_counts=simulator_support_status_counts,
        simulator_support_blocker_counts=simulator_support_blocker_counts,
        timing_coverage_status_counts=timing_coverage_status_counts,
        max_estimated_memory_residual_gb=max_memory_residual_gb,
        max_estimated_memory_residual_gb_label=_max_positive_candidate_label(
            candidates,
            "estimated_memory_residual_gb",
        ),
        max_estimated_memory_residual_fraction_of_peak=(
            round(max(memory_residual_fractions), 3) if memory_residual_fractions else None
        ),
        max_estimated_memory_residual_fraction_of_peak_label=_max_positive_candidate_label(
            candidates,
            "estimated_memory_residual_fraction_of_peak",
        ),
        phase_bottleneck_candidate_count=len(phase_bottleneck_candidates),
        phase_bottleneck_bucket_counts=phase_bottleneck_bucket_counts,
        phase_bottleneck_phase_counts=phase_bottleneck_phase_counts,
        max_phase_bottleneck_share=max_phase_bottleneck_share,
        max_phase_bottleneck_share_label=(
            max_phase_bottleneck_candidate.label if max_phase_bottleneck_candidate is not None else None
        ),
        max_phase_bottleneck_phase=(
            max_phase_bottleneck_candidate.phase_bottleneck_phase
            if max_phase_bottleneck_candidate is not None
            else None
        ),
        max_phase_bottleneck_bucket=(
            max_phase_bottleneck_candidate.phase_bottleneck_bucket
            if max_phase_bottleneck_candidate is not None
            else None
        ),
        phase_bottleneck_half_speedup_candidate_count=len(phase_bottleneck_half_speedup_candidates),
        max_phase_bottleneck_half_speedup_delta_pct=max_phase_bottleneck_half_speedup_delta_pct,
        max_phase_bottleneck_half_speedup_delta_label=(
            max_phase_bottleneck_half_speedup_candidate.label
            if max_phase_bottleneck_half_speedup_candidate is not None
            else None
        ),
        max_phase_bottleneck_half_speedup_phase=(
            max_phase_bottleneck_half_speedup_candidate.phase_bottleneck_phase
            if max_phase_bottleneck_half_speedup_candidate is not None
            else None
        ),
        max_phase_bottleneck_half_speedup_bucket=(
            max_phase_bottleneck_half_speedup_candidate.phase_bottleneck_bucket
            if max_phase_bottleneck_half_speedup_candidate is not None
            else None
        ),
        memory_bottleneck_candidate_count=len(memory_bottleneck_candidates),
        memory_bottleneck_bucket_counts=memory_bottleneck_bucket_counts,
        memory_bottleneck_phase_counts=memory_bottleneck_phase_counts,
        max_memory_bottleneck_fraction_of_peak=max_memory_bottleneck_fraction,
        max_memory_bottleneck_fraction_label=(
            max_memory_bottleneck_candidate.label if max_memory_bottleneck_candidate is not None else None
        ),
        max_memory_bottleneck_phase=(
            max_memory_bottleneck_candidate.memory_bottleneck_phase
            if max_memory_bottleneck_candidate is not None
            else None
        ),
        max_memory_bottleneck_bucket=(
            max_memory_bottleneck_candidate.memory_bottleneck_bucket
            if max_memory_bottleneck_candidate is not None
            else None
        ),
        cross_node_dimension_counts=_cross_node_dimension_counts(candidates),
        feasibility_status_counts=_count_values([candidate.feasibility_status for candidate in candidates]),
        routing_regime_status=routing_regime_status,
        routing_regime_counts=routing_regime_counts,
        recommendation_counts=_count_values([candidate.recommendation for candidate in candidates]),
        risk_flag_counts=_count_values(risk_flags),
        scenario_readiness=scenario_readiness,
    )


def _max_positive_candidate_by_field(
    candidates: list[ScenarioCandidate],
    field_name: str,
) -> ScenarioCandidate | None:
    max_row: tuple[float, str] | None = None
    max_candidate: ScenarioCandidate | None = None
    for candidate in candidates:
        value = getattr(candidate, field_name)
        if not isinstance(value, int | float) or value <= 0:
            continue
        row = (float(value), candidate.label)
        if max_row is None or row > max_row:
            max_row = row
            max_candidate = candidate
    return max_candidate


def _max_score_label(scores: dict[str, float]) -> str | None:
    return max(scores.items(), key=lambda item: (item[1], item[0]), default=(None, 0.0))[0]


def _max_positive_candidate_label(candidates: list[ScenarioCandidate], field_name: str) -> str | None:
    max_candidate = _max_positive_candidate_by_field(candidates, field_name)
    return max_candidate.label if max_candidate is not None else None


def _csv(values: list[int] | None) -> str | None:
    if values is None:
        return None
    return ",".join(str(value) for value in values)


def _append_cli_option(args: list[str], option: str, value: object | None) -> None:
    if value is None:
        return
    args.extend([option, str(value)])


def _scenario_planner_context(
    *,
    base_path: Path,
    benchmark_dir: str | Path | None,
    supplemental_benchmark_dirs: list[str | Path] | None,
    analog_benchmark_dirs: list[str | Path] | None,
    requested_world_size: int | None,
    candidate_world_sizes: list[int],
    resolved_local_world_size: int,
    micro_batch_values: list[int],
    gradient_accumulation_values: list[int],
    sample_packing_sequence_values: list[int],
    expert_parallel_sizes: list[int] | None,
    tensor_parallel_sizes: list[int] | None,
    pipeline_parallel_sizes: list[int] | None,
    ulysses_parallel_sizes: list[int] | None,
    ringattn_parallel_sizes: list[int] | None,
    topology_sweep: str,
    balanced_routing: bool,
    device_memory_limit_gb: float,
    memory_safety_factor: float,
) -> dict[str, Any]:
    cli_args = [
        "xorl-sim-plan",
        "--config",
        str(base_path),
    ]
    _append_cli_option(cli_args, "--benchmark-dir", benchmark_dir)
    for supplemental_dir in supplemental_benchmark_dirs or []:
        _append_cli_option(cli_args, "--supplemental-benchmark-dir", supplemental_dir)
    for analog_dir in analog_benchmark_dirs or []:
        _append_cli_option(cli_args, "--analog-benchmark-dir", analog_dir)
    _append_cli_option(cli_args, "--world-size", requested_world_size)
    _append_cli_option(cli_args, "--world-sizes", _csv(candidate_world_sizes))
    _append_cli_option(cli_args, "--local-world-size", resolved_local_world_size)
    _append_cli_option(cli_args, "--micro-batch-sizes", _csv(micro_batch_values))
    _append_cli_option(cli_args, "--gradient-accumulation-steps", _csv(gradient_accumulation_values))
    _append_cli_option(cli_args, "--sample-packing-sequence-lengths", _csv(sample_packing_sequence_values))
    _append_cli_option(cli_args, "--expert-parallel-sizes", _csv(expert_parallel_sizes))
    _append_cli_option(cli_args, "--tensor-parallel-sizes", _csv(tensor_parallel_sizes))
    _append_cli_option(cli_args, "--pipeline-parallel-sizes", _csv(pipeline_parallel_sizes))
    _append_cli_option(cli_args, "--ulysses-parallel-sizes", _csv(ulysses_parallel_sizes))
    _append_cli_option(cli_args, "--ringattn-parallel-sizes", _csv(ringattn_parallel_sizes))
    _append_cli_option(cli_args, "--topology-sweep", topology_sweep)
    if balanced_routing:
        cli_args.append("--balanced-routing")
    _append_cli_option(cli_args, "--device-memory-limit-gb", device_memory_limit_gb)
    _append_cli_option(cli_args, "--memory-safety-factor", memory_safety_factor)
    write_args = [*cli_args, "--write-measurement-configs", "<output-dir>"]
    return {
        "requested_args": {
            "benchmark_dir": str(benchmark_dir) if benchmark_dir is not None else None,
            "supplemental_benchmark_dirs": [
                str(supplemental_dir) for supplemental_dir in supplemental_benchmark_dirs or []
            ],
            "analog_benchmark_dirs": [str(analog_dir) for analog_dir in analog_benchmark_dirs or []],
            "world_size": requested_world_size,
            "world_sizes": candidate_world_sizes,
            "local_world_size": resolved_local_world_size,
            "micro_batch_sizes": micro_batch_values,
            "gradient_accumulation_steps": gradient_accumulation_values,
            "sample_packing_sequence_lengths": sample_packing_sequence_values,
            "expert_parallel_sizes": expert_parallel_sizes,
            "tensor_parallel_sizes": tensor_parallel_sizes,
            "pipeline_parallel_sizes": pipeline_parallel_sizes,
            "ulysses_parallel_sizes": ulysses_parallel_sizes,
            "ringattn_parallel_sizes": ringattn_parallel_sizes,
            "topology_sweep": topology_sweep,
            "balanced_routing": balanced_routing,
            "device_memory_limit_gb": device_memory_limit_gb,
            "memory_safety_factor": memory_safety_factor,
        },
        "measurement_config_command": write_args,
    }


def plan_scenario(
    base_config_path: str | Path,
    *,
    benchmark_dir: str | Path | None = None,
    supplemental_benchmark_dirs: list[str | Path] | None = None,
    analog_benchmark_dirs: list[str | Path] | None = None,
    world_size: int | None = None,
    world_sizes: list[int] | None = None,
    local_world_size: int | None = None,
    micro_batch_sizes: list[int] | None = None,
    gradient_accumulation_steps: list[int] | None = None,
    sample_packing_sequence_lengths: list[int] | None = None,
    expert_parallel_sizes: list[int] | None = None,
    tensor_parallel_sizes: list[int] | None = None,
    pipeline_parallel_sizes: list[int] | None = None,
    ulysses_parallel_sizes: list[int] | None = None,
    ringattn_parallel_sizes: list[int] | None = None,
    topology_sweep: str = "base",
    balanced_routing: bool = False,
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
    candidate_world_sizes = _dedupe_sorted(world_sizes or [resolved_world_size])
    primary_behavior_points = load_benchmark_behavior_points(benchmark_dir) if benchmark_dir is not None else []
    raw_supplemental_behavior_points: list[BenchmarkBehaviorPoint] = []
    for supplemental_dir in supplemental_benchmark_dirs or []:
        raw_supplemental_behavior_points.extend(load_benchmark_behavior_points(supplemental_dir))
    supplemental_behavior_points = [
        point for point in raw_supplemental_behavior_points if not behavior_point_model_mismatches(point, base_config)
    ]
    supplemental_model_mismatch_count = len(raw_supplemental_behavior_points) - len(supplemental_behavior_points)
    analog_behavior_points: list[BenchmarkBehaviorPoint] = []
    for analog_dir in analog_benchmark_dirs or []:
        analog_behavior_points.extend(load_benchmark_behavior_points(analog_dir))
    same_model_behavior_points = primary_behavior_points + supplemental_behavior_points
    behavior_points = same_model_behavior_points + analog_behavior_points
    metadata = resolve_model_metadata(base_config)
    benchmark_support = _scenario_benchmark_support(
        same_model_behavior_points,
        base_config=base_config,
        base_topology=base_topology,
    )

    default_behavior_points = same_model_behavior_points or behavior_points
    micro_batch_values = micro_batch_sizes or _default_micro_batch_sizes(base_topology, default_behavior_points)
    gradient_accumulation_values = gradient_accumulation_steps or [base_topology.gradient_accumulation_steps]
    sample_packing_sequence_values = sample_packing_sequence_lengths or (
        [base_topology.sample_packing_sequence_len] if base_topology.sample_packing_sequence_len is not None else []
    )

    candidates: list[ScenarioCandidate] = []
    warnings: list[str] = []
    if not sample_packing_sequence_values:
        warnings.append("skipped all scenarios: data.sample_packing_sequence_len is not set")
    include_sequence_len_in_label = len(sample_packing_sequence_values) > 1
    seen: set[tuple[str, str]] = set()
    for candidate_world_size in candidate_world_sizes:
        candidate_local_world_size = min(resolved_local_world_size, candidate_world_size)
        for sample_packing_sequence_len in sample_packing_sequence_values:
            try:
                candidate_base_values = _topology_values_with_dp_split(
                    {
                        "world_size": candidate_world_size,
                        "expert_parallel_size": base_topology.expert_parallel_size,
                        "tensor_parallel_size": base_topology.tensor_parallel_size,
                        "pipeline_parallel_size": base_topology.pipeline_parallel_size,
                        "ulysses_parallel_size": base_topology.ulysses_parallel_size,
                        "ringattn_parallel_size": base_topology.ringattn_parallel_size,
                    },
                    preferred_replicate_size=base_topology.data_parallel_replicate_size,
                    preferred_shard_size=base_topology.data_parallel_shard_size,
                )
                if candidate_base_values is None:
                    raise ValueError("candidate base topology has invalid DP split")
                candidate_base_config = _mutated_config(
                    base_config,
                    world_size=candidate_world_size,
                    micro_batch_size=base_topology.micro_batch_size,
                    gradient_accumulation_steps=base_topology.gradient_accumulation_steps,
                    expert_parallel_size=base_topology.expert_parallel_size,
                    tensor_parallel_size=base_topology.tensor_parallel_size,
                    pipeline_parallel_size=base_topology.pipeline_parallel_size,
                    ulysses_parallel_size=base_topology.ulysses_parallel_size,
                    ringattn_parallel_size=base_topology.ringattn_parallel_size,
                    data_parallel_replicate_size=candidate_base_values["data_parallel_replicate_size"],
                    data_parallel_shard_size=candidate_base_values["data_parallel_shard_size"],
                )
                _set_sample_packing_sequence_len(candidate_base_config, sample_packing_sequence_len)
                candidate_base_topology = resolve_topology(
                    candidate_base_config,
                    world_size=candidate_world_size,
                    local_world_size=candidate_local_world_size,
                )
            except ValueError as exc:
                warnings.append(
                    f"skipped world_size={candidate_world_size}, "
                    f"sample_packing_sequence_len={sample_packing_sequence_len}: {exc}"
                )
                continue

            if topology_sweep == "auto":
                ep_values = expert_parallel_sizes or _auto_ep_sizes(candidate_base_topology)
                tp_values = tensor_parallel_sizes or _auto_tensor_parallel_sizes(candidate_base_topology, metadata)
                pp_values = pipeline_parallel_sizes or _auto_pipeline_parallel_sizes(candidate_base_topology, metadata)
                ulysses_values = ulysses_parallel_sizes or _auto_ulysses_parallel_sizes(
                    candidate_base_topology, metadata
                )
                ring_values = ringattn_parallel_sizes or _auto_ringattn_parallel_sizes(candidate_base_topology)
            else:
                ep_values = expert_parallel_sizes or [candidate_base_topology.expert_parallel_size]
                tp_values = tensor_parallel_sizes or [candidate_base_topology.tensor_parallel_size]
                pp_values = pipeline_parallel_sizes or [candidate_base_topology.pipeline_parallel_size]
                ulysses_values = ulysses_parallel_sizes or [candidate_base_topology.ulysses_parallel_size]
                ring_values = ringattn_parallel_sizes or [candidate_base_topology.ringattn_parallel_size]

            for pp in pp_values:
                for tp in tp_values:
                    for ulysses in ulysses_values:
                        for ringattn in ring_values:
                            for ep in ep_values:
                                for micro_batch_size in micro_batch_values:
                                    for gradient_accumulation_step in gradient_accumulation_values:
                                        try:
                                            candidate_values = _topology_values_with_dp_split(
                                                {
                                                    "world_size": candidate_world_size,
                                                    "expert_parallel_size": ep,
                                                    "tensor_parallel_size": tp,
                                                    "pipeline_parallel_size": pp,
                                                    "ulysses_parallel_size": ulysses,
                                                    "ringattn_parallel_size": ringattn,
                                                },
                                                preferred_replicate_size=base_topology.data_parallel_replicate_size,
                                                preferred_shard_size=base_topology.data_parallel_shard_size,
                                            )
                                            if candidate_values is None:
                                                raise ValueError("candidate topology has invalid DP split")
                                            raw_config = _mutated_config(
                                                base_config,
                                                world_size=candidate_world_size,
                                                micro_batch_size=micro_batch_size,
                                                gradient_accumulation_steps=gradient_accumulation_step,
                                                expert_parallel_size=ep,
                                                tensor_parallel_size=tp,
                                                pipeline_parallel_size=pp,
                                                ulysses_parallel_size=ulysses,
                                                ringattn_parallel_size=ringattn,
                                                data_parallel_replicate_size=candidate_values[
                                                    "data_parallel_replicate_size"
                                                ],
                                                data_parallel_shard_size=candidate_values["data_parallel_shard_size"],
                                            )
                                            _set_sample_packing_sequence_len(raw_config, sample_packing_sequence_len)
                                            _set_balanced_routing(raw_config, balanced_routing)
                                            topology = resolve_topology(
                                                raw_config,
                                                world_size=candidate_world_size,
                                                local_world_size=candidate_local_world_size,
                                            )
                                        except ValueError as exc:
                                            warnings.append(
                                                f"skipped world={candidate_world_size}, "
                                                f"seq={sample_packing_sequence_len}, mbs={micro_batch_size}, "
                                                f"ga={gradient_accumulation_step}, ep={ep}, tp={tp}, pp={pp}, "
                                                f"u={ulysses}, r={ringattn}: {exc}"
                                            )
                                            continue
                                        if topology.ep_fsdp_size is None:
                                            warnings.append(
                                                f"skipped {_topology_label(topology)}: ep_fsdp is not integral"
                                            )
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
                                        communication = _communication_ledger(topology)
                                        memory_peak_estimate = _calibrated_memory_peak_estimate(
                                            behavior_points,
                                            base_config,
                                            raw_config,
                                            topology,
                                            shape,
                                            metadata,
                                            default_world_size=candidate_world_size,
                                            default_local_world_size=candidate_local_world_size,
                                            analytic_peak_floor_gb=memory.analytic_peak_floor_gb,
                                        )
                                        exact_points = [
                                            point
                                            for point in behavior_points
                                            if behavior_point_matches_topology(point, topology)
                                            and behavior_point_matches_workload(point, raw_config)
                                        ]
                                        label_topology = _candidate_topology_label(
                                            topology,
                                            include_sequence_len=include_sequence_len_in_label,
                                        )
                                        if exact_points:
                                            for point in exact_points:
                                                behavior = predict_benchmark_behavior(
                                                    [point], topology, shape, raw_config
                                                )
                                                label = f"{label_topology}:{point.label}"
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
                                                        memory_peak_estimate=memory_peak_estimate,
                                                        memory_ownership_notes=_memory_ownership_notes(memory),
                                                        communication=communication,
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
                                            memory_peak_estimate=memory_peak_estimate,
                                        )
                                        label_suffix = (
                                            "cross_model_extrapolated"
                                            if behavior.status == "cross_model_extrapolated"
                                            else "extrapolated"
                                        )
                                        label = f"{label_topology}:{label_suffix}"
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
                                                memory_peak_estimate=memory_peak_estimate,
                                                memory_ownership_notes=_memory_ownership_notes(memory),
                                                communication=communication,
                                                notes=extrapolation_notes,
                                            )
                                        )

    candidates = _apply_cross_model_analog_support_risk(candidates)
    candidates = _apply_same_workload_scaling_metrics(candidates)
    candidates = _apply_frontier_dominance(candidates)
    candidates = sorted(candidates, key=_candidate_sort_key, reverse=True)
    feasible = [candidate for candidate in candidates if candidate.score_tokens_per_sec is not None]
    best_raw = feasible[0] if feasible else None
    risk_adjusted = [candidate for candidate in feasible if candidate.score_risk_adjusted_tokens_per_sec is not None]
    best_risk_adjusted = max(risk_adjusted, key=_risk_adjusted_sort_key) if risk_adjusted else None
    efficiency = [candidate for candidate in feasible if candidate.score_tokens_per_gpu_per_sec is not None]
    best_efficiency = max(efficiency, key=_efficiency_sort_key) if efficiency else None
    risk_adjusted_efficiency = [
        candidate for candidate in feasible if candidate.score_risk_adjusted_tokens_per_gpu_per_sec is not None
    ]
    best_risk_adjusted_efficiency = (
        max(risk_adjusted_efficiency, key=_risk_adjusted_efficiency_sort_key) if risk_adjusted_efficiency else None
    )
    next_measurement = [candidate for candidate in risk_adjusted if "requires_remeasurement" in candidate.risk_flags]
    best_next_measurement = max(next_measurement, key=_risk_adjusted_sort_key) if next_measurement else None
    promotable = [candidate for candidate in feasible if candidate.promotable]
    best_promotable = promotable[0] if promotable else None
    decision_summary = _scenario_decision_summary(
        candidates,
        feasible,
        best_raw,
        best_risk_adjusted,
        best_efficiency,
        best_risk_adjusted_efficiency,
        best_next_measurement,
        best_promotable,
        benchmark_support,
    )
    if best_raw is not None and not best_raw.promotable:
        warnings.append(f"best raw scenario {best_raw.label} is not correctness-promotable")
    if best_raw is not None and best_risk_adjusted is not None and best_raw.label != best_risk_adjusted.label:
        warnings.append(
            f"best raw scenario {best_raw.label} differs from risk-adjusted choice {best_risk_adjusted.label}"
        )
    if best_promotable is None:
        warnings.append("no correctness-promotable scenario found")
    if supplemental_behavior_points:
        warnings.append(f"loaded {len(supplemental_behavior_points)} supplemental benchmark behavior points")
    if supplemental_model_mismatch_count:
        warnings.append(
            f"ignored {supplemental_model_mismatch_count} supplemental benchmark behavior points with model mismatch"
        )
    if analog_behavior_points:
        warnings.append(f"loaded {len(analog_behavior_points)} analog benchmark behavior points")

    report = ScenarioReport(
        base_config_path=str(base_path),
        benchmark_dir=str(benchmark_dir) if benchmark_dir is not None else None,
        device_memory_limit_gb=device_memory_limit_gb,
        memory_safety_factor=memory_safety_factor,
        topology_sweep=topology_sweep,
        balanced_routing=balanced_routing,
        world_sizes=candidate_world_sizes,
        candidate_count=len(candidates),
        feasible_count=len(feasible),
        best_raw=best_raw,
        best_risk_adjusted=best_risk_adjusted,
        best_efficiency=best_efficiency,
        best_risk_adjusted_efficiency=best_risk_adjusted_efficiency,
        best_next_measurement=best_next_measurement,
        best_promotable=best_promotable,
        decision_summary=decision_summary,
        candidates=candidates,
        planner_context=_scenario_planner_context(
            base_path=base_path,
            benchmark_dir=benchmark_dir,
            supplemental_benchmark_dirs=supplemental_benchmark_dirs,
            analog_benchmark_dirs=analog_benchmark_dirs,
            requested_world_size=world_size,
            candidate_world_sizes=candidate_world_sizes,
            resolved_local_world_size=resolved_local_world_size,
            micro_batch_values=micro_batch_values,
            gradient_accumulation_values=gradient_accumulation_values,
            sample_packing_sequence_values=sample_packing_sequence_values,
            expert_parallel_sizes=expert_parallel_sizes,
            tensor_parallel_sizes=tensor_parallel_sizes,
            pipeline_parallel_sizes=pipeline_parallel_sizes,
            ulysses_parallel_sizes=ulysses_parallel_sizes,
            ringattn_parallel_sizes=ringattn_parallel_sizes,
            topology_sweep=topology_sweep,
            balanced_routing=balanced_routing,
            device_memory_limit_gb=device_memory_limit_gb,
            memory_safety_factor=memory_safety_factor,
        ),
        warnings=warnings,
        supplemental_benchmark_dirs=[str(path) for path in supplemental_benchmark_dirs or []],
        analog_benchmark_dirs=[str(path) for path in analog_benchmark_dirs or []],
        primary_behavior_point_count=len(primary_behavior_points),
        supplemental_behavior_point_count=len(supplemental_behavior_points),
        analog_behavior_point_count=len(analog_behavior_points),
        total_behavior_point_count=len(behavior_points),
    )
    return _attach_measurement_design_summary(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", help="Built-in calibration-pack name")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument(
        "--supplemental-benchmark-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional same-model benchmark directory included as scenario calibration and support evidence",
    )
    parser.add_argument(
        "--analog-benchmark-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional benchmark directory used only as low-confidence analog evidence when same-model data is absent",
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--world-sizes", default=None, help="Comma list of world sizes to compare")
    parser.add_argument("--local-world-size", type=int, default=None)
    parser.add_argument("--micro-batch-sizes", default=None, help="Comma list, or auto when omitted")
    parser.add_argument(
        "--gradient-accumulation-steps", default=None, help="Comma list, or base config GA when omitted"
    )
    parser.add_argument(
        "--sample-packing-sequence-lengths",
        default=None,
        help="Comma list, or base config data.sample_packing_sequence_len when omitted",
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
    parser.add_argument(
        "--balanced-routing",
        action="store_true",
        help="Match measured scenarios that used XORL_MOE_SYNTHETIC_ROUTING=balanced",
    )
    parser.add_argument("--device-memory-limit-gb", type=float, default=80.0)
    parser.add_argument("--memory-safety-factor", type=float, default=1.15)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--write-measurement-configs",
        type=Path,
        default=None,
        help="Write the bounded measurement portfolio as runnable YAML configs in this directory",
    )
    args = parser.parse_args()

    args.config, args.benchmark_dir = resolve_pack_inputs(args.pack, args.config, args.benchmark_dir)
    if args.config is None:
        parser.error("provide --pack or --config")

    report = plan_scenario(
        args.config,
        benchmark_dir=args.benchmark_dir,
        supplemental_benchmark_dirs=args.supplemental_benchmark_dir,
        analog_benchmark_dirs=args.analog_benchmark_dir,
        world_size=args.world_size,
        world_sizes=_parse_int_list(args.world_sizes),
        local_world_size=args.local_world_size,
        micro_batch_sizes=_parse_int_list(args.micro_batch_sizes),
        gradient_accumulation_steps=_parse_int_list(args.gradient_accumulation_steps),
        sample_packing_sequence_lengths=_parse_int_list(args.sample_packing_sequence_lengths),
        expert_parallel_sizes=_parse_int_list(args.expert_parallel_sizes),
        tensor_parallel_sizes=_parse_int_list(args.tensor_parallel_sizes),
        pipeline_parallel_sizes=_parse_int_list(args.pipeline_parallel_sizes),
        ulysses_parallel_sizes=_parse_int_list(args.ulysses_parallel_sizes),
        ringattn_parallel_sizes=_parse_int_list(args.ringattn_parallel_sizes),
        topology_sweep=args.topology_sweep,
        balanced_routing=args.balanced_routing,
        device_memory_limit_gb=args.device_memory_limit_gb,
        memory_safety_factor=args.memory_safety_factor,
    )
    rendered = json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.write_measurement_configs is not None:
        write_measurement_configs(report, args.write_measurement_configs)


if __name__ == "__main__":
    main()
