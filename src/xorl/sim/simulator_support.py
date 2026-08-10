"""Declare which training-engine surface a static simulator report covers."""

from __future__ import annotations

from typing import Any


try:
    from .schemas import MemoryLedger, SimulatorSupportLedger, TimingLedger, Topology
except ImportError:  # pragma: no cover - exercised by direct script execution
    from schemas import MemoryLedger, SimulatorSupportLedger, TimingLedger, Topology


_SERVER_FLAT_KEYS = {
    "ce_mode",
    "data_parallel_mode",
    "enable_packing",
    "sample_packing_sequence_len",
    "skip_initial_checkpoint",
    "sync_inference_method",
    "worker_connection_timeout",
    "worker_max_retries",
}


def _section(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw_config.get(name, {})
    return value if isinstance(value, dict) else {}


def _looks_like_flat_server_config(raw_config: dict[str, Any]) -> bool:
    if _section(raw_config, "train"):
        return False
    if "model_path" not in raw_config and "config_path" not in raw_config:
        return False
    return any(key in raw_config for key in _SERVER_FLAT_KEYS)


def requested_simulator_surface(raw_config: dict[str, Any]) -> str:
    if _section(raw_config, "server") or _looks_like_flat_server_config(raw_config):
        return "server_forward_backward"
    if _section(raw_config, "train"):
        return "local_training"
    return "unknown_config_surface"


def _configured_int(
    raw_config: dict[str, Any],
    key: str,
    *,
    surface: str,
    topology_value: int | None = None,
    default: int = 1,
) -> int:
    sections: list[dict[str, Any]]
    if surface == "server_forward_backward":
        sections = [_section(raw_config, "server"), raw_config, _section(raw_config, "train")]
    else:
        sections = [_section(raw_config, "train"), raw_config, _section(raw_config, "server")]

    for section in sections:
        value = section.get(key)
        if value is not None:
            return int(value)
    if topology_value is not None:
        return int(topology_value)
    return default


def _runtime_output_support(memory: MemoryLedger | None, timing: TimingLedger | None) -> tuple[list[str], list[str]]:
    supported: list[str] = []
    unsupported: list[str] = []

    if memory is None:
        unsupported.append("memory_ledger_not_built")
    else:
        if memory.analytic_peak_floor_gb is not None:
            supported.append("analytic_memory_floor")
        else:
            unsupported.append("analytic_memory_floor")
        if memory.calibrated_peak_mem_gb is not None or memory.observed_peak_mem_gb_max is not None:
            supported.append("calibrated_peak_memory")
        else:
            unsupported.append("calibrated_peak_memory_without_observed_or_benchmark_anchor")

    if timing is None:
        unsupported.append("timing_ledger_not_built")
    else:
        if timing.step_time_s is not None:
            supported.append("step_time_prediction")
        else:
            unsupported.append("step_time_prediction_without_observed_or_benchmark_anchor")
        if timing.phase_time_sec:
            supported.append("phase_timing")
        else:
            unsupported.append("phase_timing_without_observed_or_benchmark_anchor")

    return supported, unsupported


def resolve_simulator_support(
    raw_config: dict[str, Any],
    *,
    topology: Topology | None = None,
    memory: MemoryLedger | None = None,
    timing: TimingLedger | None = None,
) -> SimulatorSupportLedger:
    surface = requested_simulator_surface(raw_config)
    pipeline_parallel_size = _configured_int(
        raw_config,
        "pipeline_parallel_size",
        surface=surface,
        topology_value=topology.pipeline_parallel_size if topology is not None else None,
    )
    runtime_supported, runtime_unsupported = _runtime_output_support(memory, timing)

    if surface == "server_forward_backward":
        supported = ["config_surface_detection", "model_metadata_resolution"]
        if topology is not None:
            supported.append("server_parallelism_topology")
        blockers = [
            "server_forward_backward_backend_missing",
            "opd_loss_path_missing",
            "hidden_cache_prefetch_timing_missing",
            "server_phase_memory_attribution_missing",
        ]
        unsupported = [
            "server_forward_backward_timing",
            "opd_loss_path_timing",
            "hidden_cache_fetch_prefetch_timing",
            "server_phase_memory_attribution",
            *runtime_unsupported,
        ]
        notes = [
            "server configs use a flat YAML surface and need a server-specific forward/backward backend",
            "server and local forward/backward surfaces must be calibrated separately",
        ]
        if pipeline_parallel_size > 1:
            blockers.extend(
                [
                    "pp_schedule_event_model_missing",
                    "pp_forward_backward_peak_not_separated",
                ]
            )
            unsupported.extend(["pp_schedule_timing", "separate_forward_backward_peak"])
        return SimulatorSupportLedger(
            requested_surface=surface,
            support_status="unsupported_server_forward_backward",
            support_blockers=sorted(set(blockers)),
            supported_outputs=sorted(set(supported)),
            unsupported_outputs=sorted(set(unsupported)),
            notes=notes,
        )

    if surface != "local_training":
        return SimulatorSupportLedger(
            requested_surface=surface,
            support_status="unsupported_unknown_config_surface",
            support_blockers=["config_surface_not_recognized"],
            supported_outputs=["config_surface_detection"],
            unsupported_outputs=sorted(
                {
                    "local_training_topology",
                    "server_forward_backward_timing",
                    *runtime_unsupported,
                }
            ),
            notes=["local training configs must use nested train/data/model sections"],
        )

    supported = [
        "config_surface_detection",
        "model_metadata_resolution",
        "local_training_topology",
        "shape_ledger",
        *runtime_supported,
    ]
    unsupported = list(runtime_unsupported)
    if pipeline_parallel_size > 1:
        blockers = [
            "pp_schedule_event_model_missing",
            "pp_forward_backward_peak_not_separated",
            "pp_phase_timing_not_separated",
        ]
        unsupported.extend(
            [
                "pp_schedule_timing",
                "separate_forward_backward_peak",
                "pp_activation_liveness_peak",
            ]
        )
        return SimulatorSupportLedger(
            requested_surface=surface,
            support_status="partial_local_pp_memory_only",
            support_blockers=blockers,
            supported_outputs=sorted({*supported, "pp_stage_parameter_ownership"}),
            unsupported_outputs=sorted(set(unsupported)),
            notes=[
                "PP reports include topology and analytic ownership, but not a schedule-level event model",
                "XoRL reports a combined fwd+bwd peak for PP until schedule attribution is added",
            ],
        )

    return SimulatorSupportLedger(
        requested_surface=surface,
        support_status="supported_local_non_pp",
        support_blockers=[],
        supported_outputs=sorted(set(supported)),
        unsupported_outputs=sorted(set(unsupported)),
        notes=[],
    )
