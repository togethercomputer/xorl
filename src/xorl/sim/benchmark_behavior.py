"""Empirical benchmark behavior calibration for checked-in benchmark recipes."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml


try:
    from .calibration_packs import resolve_calibration_pack
    from .collect_calibration import parse_log_text, summarize_observed_run
    from .config_fingerprint import load_training_config, resolve_topology
    from .model_metadata import model_ref_from_config
    from .runtime_config import runtime_training_config
    from .schemas import BenchmarkBehaviorPoint, BenchmarkBehaviorPrediction, ShapeLedger, Topology, to_jsonable
except ImportError:  # pragma: no cover - exercised by direct script execution
    from calibration_packs import resolve_calibration_pack
    from collect_calibration import parse_log_text, summarize_observed_run
    from config_fingerprint import load_training_config, resolve_topology
    from model_metadata import model_ref_from_config
    from runtime_config import runtime_training_config
    from schemas import BenchmarkBehaviorPoint, BenchmarkBehaviorPrediction, ShapeLedger, Topology, to_jsonable


H100_BF16_PROMISED_TFLOPS_PER_GPU = 989.0
BENCHMARK_SOURCE_MANIFESTS = ("benchmark_sources.yaml", "benchmark_sources.yml", "benchmark_sources.json")
BENCHMARK_BEHAVIOR_OVERRIDE_MANIFESTS = (
    "benchmark_behavior_overrides.yaml",
    "benchmark_behavior_overrides.yml",
    "benchmark_behavior_overrides.json",
)


def _gpu_count_from_text(text: str) -> int | None:
    match = re.search(r"(?P<nodes>\d+)\s+nodes?\s+x\s+(?P<gpus>\d+)\s+H100", text, re.IGNORECASE)
    if match:
        return int(match.group("nodes")) * int(match.group("gpus"))
    match = re.search(r"(?P<gpus>\d+)\s*[x×]\s*H100", text, re.IGNORECASE)
    if match:
        return int(match.group("gpus"))
    match = re.search(r"(?P<gpus>\d+)\s+GPUs?", text, re.IGNORECASE)
    return int(match.group("gpus")) if match else None


def human_number(value: str) -> float:
    cleaned = value.strip().replace(",", "").lstrip("~")
    multiplier = 1.0
    if cleaned.endswith(("K", "k")):
        cleaned = cleaned[:-1]
        multiplier = 1_000.0
    elif cleaned.endswith(("M", "m")):
        cleaned = cleaned[:-1]
        multiplier = 1_000_000.0
    return float(cleaned) * multiplier


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return result


def _model_ref_from_text_or_path(text: str, source: str | Path) -> str | None:
    lowered = f"{text}\n{source}".lower()
    if "qwen3.6-35b-a3b-fp8" in lowered:
        return "Qwen/Qwen3.6-35B-A3B-FP8"
    if "qwen3.6-35b-a3b" in lowered or "qwen36" in lowered:
        return "Qwen/Qwen3.6-35B-A3B"
    if "qwen3.5-397b-a17b" in lowered or "q397b" in lowered:
        return "Qwen/Qwen3.5-397B-A17B"
    if "qwen3-235b-a22b-instruct-2507" in lowered:
        return "Qwen/Qwen3-235B-A22B-Instruct-2507"
    if "qwen3-235b-a22b" in lowered or "q235" in lowered:
        return "Qwen/Qwen3-235B-A22B"
    if "qwen3.5-35b-a3b" in lowered or "q35_2node_005_050" in lowered:
        return "Qwen/Qwen3.5-35B-A3B"
    if "qwen3-coder-30b-a3b" in lowered:
        return "Qwen/Qwen3-Coder-30B-A3B"
    # Non-Coder Qwen3-30B-A3B. Checked after the Coder variant so a coder label/path never
    # falls through here; "qwen3-coder-30b-a3b" does not contain the substring "qwen3-30b-a3b".
    if "qwen3-30b-a3b" in lowered:
        return "Qwen/Qwen3-30B-A3B"
    if "qwen3-8b" in lowered:
        return "Qwen/Qwen3-8B"
    return None


def _model_family_key(model_ref: str | None) -> str | None:
    if not model_ref:
        return None
    lowered = model_ref.strip().lower()
    if "qwen3.6-35b-a3b-fp8" in lowered:
        return "qwen3.6-35b-a3b-fp8"
    if "qwen3.6-35b-a3b" in lowered:
        return "qwen3.6-35b-a3b"
    if "qwen3.5-397b-a17b" in lowered:
        return "qwen3.5-397b-a17b"
    if "qwen3.5-35b-a3b" in lowered:
        return "qwen3.5-35b-a3b"
    if "qwen3-235b-a22b" in lowered:
        return "qwen3-235b-a22b"
    return lowered.rstrip("/")


def behavior_point_model_mismatches(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any]) -> bool:
    point_key = _model_family_key(point.model_ref)
    config_key = _model_family_key(model_ref_from_config(raw_config))
    return point_key is not None and config_key is not None and point_key != config_key


def _readme_point(readme_text: str, *, source: str, model_ref: str | None) -> BenchmarkBehaviorPoint | None:
    tps_match = re.search(r"\|\s*tokens/sec\s*\|\s*(?P<value>~?[0-9.]+[KkMm]?)\s*\|", readme_text)
    step_match = re.search(r"\|\s*step time\s*\|\s*(?P<value>~?[0-9.]+)s\s*\|", readme_text)
    mfu_match = re.search(r"\|\s*MFU\s*\|\s*(?P<value>~?[0-9.]+)%", readme_text)
    memory_match = re.search(r"\|\s*allocated memory\s*\|\s*(?P<value>~?[0-9.]+)GB\s*\|", readme_text)
    retries_match = re.search(r"\|\s*allocator retries\s*\|\s*(?P<value>\d+)\s*\|", readme_text)
    mbs_match = re.search(r"micro_batch_size:\s*(?P<value>\d+)", readme_text)
    global_batch_match = re.search(r"global_batch_size:\s*(?P<value>\d+)", readme_text)
    gradient_accumulation_match = re.search(r"gradient_accumulation_steps:\s*(?P<value>\d+)", readme_text)
    if not tps_match:
        return None
    return BenchmarkBehaviorPoint(
        label="readme_reference_mbs8",
        source=source,
        micro_batch_size=int(mbs_match.group("value")) if mbs_match else None,
        global_batch_size=int(global_batch_match.group("value")) if global_batch_match else None,
        gradient_accumulation_steps=(
            int(gradient_accumulation_match.group("value")) if gradient_accumulation_match else None
        ),
        tokens_per_sec=human_number(tps_match.group("value")),
        step_time_sec=float(step_match.group("value").lstrip("~")) if step_match else None,
        phase_time_sec=_float_dict({}),
        phase_time_share=_float_dict({}),
        mfu_percent=float(mfu_match.group("value").lstrip("~")) if mfu_match else None,
        tflops_per_gpu=None,
        peak_mem_gb=float(memory_match.group("value").lstrip("~")) if memory_match else None,
        allocator_retries=int(retries_match.group("value")) if retries_match else None,
        gpu_count=_gpu_count_from_text(readme_text),
        model_ref=model_ref,
        sample_packing_sequence_len=_seq_len_from_readme(readme_text),
        tensor_parallel_size=_readme_parallel_int(
            readme_text,
            "tensor_parallel_size",
            (r"\btp[=_ ](?P<value>\d+)\b", r"\bTP(?P<value>\d+)\b"),
        ),
        pipeline_parallel_size=_readme_parallel_int(
            readme_text,
            "pipeline_parallel_size",
            (r"\bpp[=_ ](?P<value>\d+)\b", r"\bPP(?P<value>\d+)\b"),
        ),
        ulysses_parallel_size=_readme_parallel_int(
            readme_text,
            "ulysses_parallel_size",
            (r"\bulysses[=_ ](?P<value>\d+)\b", r"\bU(?P<value>\d+)\b"),
        ),
        ringattn_parallel_size=_readme_parallel_int(
            readme_text,
            "ringattn_parallel_size",
            (r"\bring[=_ ](?P<value>\d+)\b", r"\bR(?P<value>\d+)\b"),
        ),
        expert_parallel_size=_readme_parallel_int(
            readme_text,
            "expert_parallel_size",
            (r"\bep[=_ ](?P<value>\d+)\b", r"\bEP(?P<value>\d+)\b"),
        ),
        ep_fsdp_size=_readme_parallel_int(
            readme_text,
            "ep_fsdp_size",
            (r"\bep_fsdp[=_ ](?P<value>\d+)\b", r"\beFSDP(?P<value>\d+)\b"),
        ),
        deepep_async_combine=_readme_bool_from_text(readme_text, "deepep_async_combine"),
        deepep_num_sms=_readme_parallel_int(
            readme_text,
            "deepep_num_sms",
            (r"\bdeepep_num_sms[=: ]+(?P<value>\d+)\b", r"\bSMS(?P<value>\d+)\b"),
        ),
        deepep_buffer_size_gb=_readme_float_from_text(readme_text, "deepep_buffer_size_gb"),
        enable_compile=_readme_bool_from_text(readme_text, "enable_compile"),
        gradient_checkpointing_method=_checkpointing_method_from_text(readme_text),
        status="reference_speed",
        correctness_status="raw_speed_not_promoted_without_matching_k3_pass",
        notes=["current-main logical FLOPs accounting", "balanced synthetic routing", "deepep_async_combine true"],
    )


def _readme_adjacent_mbs10_point(
    readme_text: str, *, source: str, seq_len: int | None, model_ref: str | None
) -> BenchmarkBehaviorPoint | None:
    match = re.search(r"`mbs=10`[^~]+~(?P<value>[0-9.]+)K tok/s", readme_text)
    if not match:
        return None
    tokens_per_sec = human_number(match.group("value") + "K")
    global_batch_size = 320
    step_time_sec = (global_batch_size * seq_len / tokens_per_sec) if seq_len else None
    return BenchmarkBehaviorPoint(
        label="readme_adjacent_mbs10_allocator_pressure",
        source=source,
        micro_batch_size=10,
        global_batch_size=global_batch_size,
        tokens_per_sec=tokens_per_sec,
        step_time_sec=step_time_sec,
        phase_time_sec=_float_dict({}),
        phase_time_share=_float_dict({}),
        mfu_percent=None,
        tflops_per_gpu=None,
        peak_mem_gb=None,
        allocator_retries=None,
        gpu_count=_gpu_count_from_text(readme_text),
        model_ref=model_ref,
        sample_packing_sequence_len=seq_len,
        tensor_parallel_size=_readme_parallel_int(
            readme_text,
            "tensor_parallel_size",
            (r"\btp[=_ ](?P<value>\d+)\b", r"\bTP(?P<value>\d+)\b"),
        ),
        pipeline_parallel_size=_readme_parallel_int(
            readme_text,
            "pipeline_parallel_size",
            (r"\bpp[=_ ](?P<value>\d+)\b", r"\bPP(?P<value>\d+)\b"),
        ),
        ulysses_parallel_size=_readme_parallel_int(
            readme_text,
            "ulysses_parallel_size",
            (r"\bulysses[=_ ](?P<value>\d+)\b", r"\bU(?P<value>\d+)\b"),
        ),
        ringattn_parallel_size=_readme_parallel_int(
            readme_text,
            "ringattn_parallel_size",
            (r"\bring[=_ ](?P<value>\d+)\b", r"\bR(?P<value>\d+)\b"),
        ),
        expert_parallel_size=_readme_parallel_int(
            readme_text,
            "expert_parallel_size",
            (r"\bep[=_ ](?P<value>\d+)\b", r"\bEP(?P<value>\d+)\b"),
        ),
        ep_fsdp_size=_readme_parallel_int(
            readme_text,
            "ep_fsdp_size",
            (r"\bep_fsdp[=_ ](?P<value>\d+)\b", r"\beFSDP(?P<value>\d+)\b"),
        ),
        deepep_async_combine=_readme_bool_from_text(readme_text, "deepep_async_combine"),
        deepep_num_sms=_readme_parallel_int(
            readme_text,
            "deepep_num_sms",
            (r"\bdeepep_num_sms[=: ]+(?P<value>\d+)\b", r"\bSMS(?P<value>\d+)\b"),
        ),
        deepep_buffer_size_gb=_readme_float_from_text(readme_text, "deepep_buffer_size_gb"),
        enable_compile=_readme_bool_from_text(readme_text, "enable_compile"),
        gradient_checkpointing_method=_checkpointing_method_from_text(readme_text),
        status="allocator_pressure_slowdown",
        correctness_status="not_promoted",
        notes=["fit but slowed with allocator retries"],
    )


def _result_throughput_point(
    result_path: Path,
    result: dict[str, Any],
    *,
    topology_defaults: dict[str, int | float | bool | str],
) -> BenchmarkBehaviorPoint:
    throughput = result["throughput"]
    candidate = (
        throughput.get("candidate")
        or result.get("candidate")
        or (result.get("replay_candidate", {}) if isinstance(result.get("replay_candidate"), dict) else {}).get(
            "candidate"
        )
        or "throughput"
    )
    return BenchmarkBehaviorPoint(
        label=f"{result_path.stem}:{candidate}",
        source=str(result_path),
        micro_batch_size=throughput.get("micro_batch_size"),
        global_batch_size=throughput.get("global_batch_size"),
        gradient_accumulation_steps=_first_non_none(
            throughput.get("gradient_accumulation_steps"), topology_defaults.get("gradient_accumulation_steps")
        ),
        tokens_per_sec=throughput.get("tokens_per_sec"),
        step_time_sec=throughput.get("step_time_sec"),
        tokens_per_sec_std=_first_non_none(
            throughput.get("tokens_per_sec_std"), throughput.get("tokens_per_sec_stdev")
        ),
        tokens_per_sec_cv=throughput.get("tokens_per_sec_cv"),
        step_time_sec_std=_first_non_none(throughput.get("step_time_sec_std"), throughput.get("step_time_s_std")),
        step_time_sec_cv=_first_non_none(throughput.get("step_time_sec_cv"), throughput.get("step_time_s_cv")),
        phase_time_sec=_float_dict(throughput.get("phase_time_sec")),
        phase_time_share=_float_dict(throughput.get("phase_time_share")),
        phase_memory_peak_gb=_float_dict(throughput.get("phase_memory_peak_gb")),
        mfu_percent=throughput.get("mfu_percent"),
        tflops_per_gpu=throughput.get("mean_tflops_per_gpu"),
        peak_mem_gb=throughput.get("gpu_alloc_gb"),
        allocator_retries=None,
        measured_steps=throughput.get("measured_steps"),
        warmup_steps=throughput.get("warmup_steps"),
        gpu_count=throughput.get("gpus"),
        model_ref=_first_non_none(throughput.get("model_ref"), topology_defaults.get("model_ref")),
        sample_packing_sequence_len=_first_non_none(
            throughput.get("sample_packing_sequence_len"), topology_defaults.get("sample_packing_sequence_len")
        ),
        data_parallel_replicate_size=_first_non_none(
            throughput.get("data_parallel_replicate_size"), topology_defaults.get("data_parallel_replicate_size")
        ),
        data_parallel_shard_size=_first_non_none(
            throughput.get("data_parallel_shard_size"), topology_defaults.get("data_parallel_shard_size")
        ),
        tensor_parallel_size=_first_non_none(
            throughput.get("tensor_parallel_size"), topology_defaults.get("tensor_parallel_size")
        ),
        pipeline_parallel_size=_first_non_none(
            throughput.get("pipeline_parallel_size"), topology_defaults.get("pipeline_parallel_size")
        ),
        ulysses_parallel_size=_first_non_none(
            throughput.get("ulysses_parallel_size"), topology_defaults.get("ulysses_parallel_size")
        ),
        ringattn_parallel_size=_first_non_none(
            throughput.get("ringattn_parallel_size"), topology_defaults.get("ringattn_parallel_size")
        ),
        expert_parallel_size=_first_non_none(
            throughput.get("expert_parallel_size"), topology_defaults.get("expert_parallel_size")
        ),
        ep_fsdp_size=_first_non_none(
            throughput.get("ep_fsdp"), throughput.get("ep_fsdp_size"), topology_defaults.get("ep_fsdp_size")
        ),
        deepep_async_combine=_first_non_none(
            throughput.get("deepep_async_combine"), topology_defaults.get("deepep_async_combine")
        ),
        deepep_num_sms=_first_non_none(throughput.get("deepep_num_sms"), topology_defaults.get("deepep_num_sms")),
        deepep_buffer_size_gb=_first_non_none(
            throughput.get("deepep_buffer_size_gb"), topology_defaults.get("deepep_buffer_size_gb")
        ),
        enable_compile=_first_non_none(throughput.get("enable_compile"), topology_defaults.get("enable_compile")),
        gradient_checkpointing_method=_first_non_none(
            throughput.get("gradient_checkpointing_method"), topology_defaults.get("gradient_checkpointing_method")
        ),
        enable_activation_offload=_first_non_none(
            throughput.get("enable_activation_offload"), topology_defaults.get("enable_activation_offload")
        ),
        activation_offload_prefetch_count=_first_non_none(
            throughput.get("activation_offload_prefetch_count"),
            topology_defaults.get("activation_offload_prefetch_count"),
        ),
        fsdp_reduce_dtype=_first_non_none(
            throughput.get("fsdp_reduce_dtype"), topology_defaults.get("fsdp_reduce_dtype")
        ),
        ce_mode=_first_non_none(throughput.get("ce_mode"), topology_defaults.get("ce_mode")),
        moe_implementation=_first_non_none(
            throughput.get("moe_implementation"), topology_defaults.get("moe_implementation")
        ),
        moe_checkpoint_method=_first_non_none(
            throughput.get("moe_checkpoint_method"), topology_defaults.get("moe_checkpoint_method")
        ),
        muon_update_dtype=_first_non_none(
            throughput.get("muon_update_dtype"), topology_defaults.get("muon_update_dtype")
        ),
        attention_backend=_first_non_none(
            throughput.get("attention_backend"), topology_defaults.get("attention_backend")
        ),
        balanced_routing=_first_non_none(
            throughput.get("balanced_routing"),
            throughput.get("synthetic_balanced_routing"),
            topology_defaults.get("balanced_routing"),
        ),
        status="historical_throughput_artifact",
        correctness_status=None,
        notes=[f"commit={throughput.get('commit')}"] if throughput.get("commit") else [],
    )


def _with_k3_status(point: BenchmarkBehaviorPoint, result: dict[str, Any]) -> BenchmarkBehaviorPoint:
    k3_gate = result.get("k3_gate", {})
    if not k3_gate or k3_gate.get("candidate") not in (None, point.label.split(":", 1)[-1]):
        return point
    notes = list(point.notes)
    if k3_gate.get("primary_failure"):
        notes.append(f"k3_primary_failure={k3_gate['primary_failure']}")
    return replace(
        point,
        correctness_status=f"k3_{k3_gate.get('status')}",
        notes=notes,
    )


def _seq_len_from_readme(readme_text: str) -> int | None:
    match = re.search(r"sample_packing_sequence_len:\s*(?P<seq>\d+)", readme_text)
    if match:
        return int(match.group("seq"))
    match = re.search(r"max_seq_len[=:]\s*(?P<seq>\d+)", readme_text)
    return int(match.group("seq")) if match else None


def _config_int_from_text(text: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}:\s*(?P<value>\d+)", text)
    return int(match.group("value")) if match else None


def _readme_float_from_text(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}:\s*(?P<value>\d+(?:\.\d+)?)", text)
    return float(match.group("value")) if match else None


def _readme_bool_from_text(text: str, key: str) -> bool | None:
    match = re.search(rf"{re.escape(key)}:\s*(?P<value>true|false)", text, re.IGNORECASE)
    if not match:
        return None
    return match.group("value").lower() == "true"


def _checkpointing_method_from_text(text: str) -> str | None:
    lowered = text.lower()
    if "recompute_before_dispatch" in lowered or "before_dispatch" in lowered:
        return "recompute_before_dispatch"
    if "recompute_full_layer" in lowered or "full-layer recompute" in lowered or "fullrecompute" in lowered:
        return "recompute_full_layer"
    if "no_recompute" in lowered or "no recompute" in lowered:
        return "no_recompute"
    return None


def _trial_checkpointing_method(trial: str) -> str | None:
    return _checkpointing_method_from_text(trial)


def _trial_activation_offload(trial: str) -> bool | None:
    if "noactivationoffload" in trial:
        return False
    if "activationoffload" in trial:
        return True
    return None


def _trial_prefetch_count(trial: str) -> int | None:
    match = re.search(r"prefetch(?P<value>\d+)", trial)
    return int(match.group("value")) if match else None


def _trial_compile_enabled(trial: str) -> bool | None:
    if "nocompile" in trial:
        return False
    if "compile" in trial:
        return True
    return None


def _trial_deepep_async_combine(trial: str) -> bool | None:
    if "noasync" in trial:
        return False
    if "async" in trial:
        return True
    return None


def _trial_sms_count(trial: str) -> int | None:
    match = re.search(r"sms(?P<value>\d+)", trial)
    return int(match.group("value")) if match else None


def _trial_buffer_size_gb(trial: str) -> float | None:
    match = re.search(r"buf(?P<value>\d+)", trial)
    if not match:
        return None
    raw = match.group("value")
    if len(raw) == 1:
        return float(raw)
    return float(f"{raw[:-1]}.{raw[-1]}")


def _trial_ce_mode(trial: str) -> str | None:
    lowered = trial.lower()
    if "quackce" in lowered or "quack_linear" in lowered:
        return "quack_linear"
    if "compiledce" in lowered:
        return "compiled"
    return None


def _trial_moe_implementation(trial: str) -> str | None:
    lowered = trial.lower()
    if "untunedquack" in lowered or "quack" in lowered:
        return "quack"
    if "tritonmoe" in lowered or "triton_moe" in lowered:
        return "triton"
    return None


def _trial_muon_update_dtype(trial: str) -> str | None:
    lowered = trial.lower()
    if "bf16update" in lowered or "updatebf16" in lowered:
        return "bf16"
    if "fp32update" in lowered or "updatefp32" in lowered:
        return "fp32"
    return None


def _last_regex_int(line: str, patterns: tuple[str, ...]) -> int | None:
    value = None
    for pattern in patterns:
        for match in re.finditer(pattern, line, re.IGNORECASE):
            groupdict = match.groupdict()
            for key in ("value", "tp", "pp", "u", "ring"):
                if groupdict.get(key) is not None:
                    value = int(groupdict[key])
                    break
    return value


def _readme_parallel_int(readme_text: str, config_key: str, patterns: tuple[str, ...]) -> int | None:
    if value := _config_int_from_text(readme_text, config_key):
        return value
    for line in readme_text.splitlines():
        if value := _last_regex_int(line, patterns):
            return value
    return None


def _readme_default_balanced_routing(readme_text: str) -> bool | None:
    in_fixed_block = False
    for line in readme_text.splitlines():
        lowered = line.strip().lower()
        if not lowered or lowered.startswith("#"):
            in_fixed_block = False
            continue
        if re.search(r"\bfixed\s+(path|flags?)\b", lowered):
            in_fixed_block = True
        elif lowered.startswith(("-", "*")):
            in_fixed_block = False
        if not in_fixed_block:
            continue
        if (
            "balanced synthetic routing" in lowered
            or "synthetic balanced routing" in lowered
            or "xorl_moe_synthetic_routing=balanced" in lowered
        ):
            return True
        if "real imbalanced routing" in lowered or "real routing" in lowered:
            return False
    return None


def _readme_topology_defaults(readme_text: str) -> dict[str, int | float | bool | str]:
    defaults: dict[str, int | float | bool | str] = {}
    field_patterns = {
        "data_parallel_replicate_size": (
            r"\bdata_parallel_replicate_size[=: ]+(?P<value>\d+)\b",
            r"\bdp[_-]?rep(?:lica(?:te)?)?[=_ ](?P<value>\d+)\b",
        ),
        "data_parallel_shard_size": (
            r"\bdata_parallel_shard_size[=: ]+(?P<value>\d+)\b",
            r"\bdp[_-]?shard[=_ ](?P<value>\d+)\b",
        ),
        "tensor_parallel_size": (r"\btp[=_ ](?P<value>\d+)\b", r"\bTP(?P<value>\d+)\b"),
        "pipeline_parallel_size": (r"\bpp[=_ ](?P<value>\d+)\b", r"\bPP(?P<value>\d+)\b"),
        "ulysses_parallel_size": (r"\bulysses[=_ ](?P<value>\d+)\b", r"\bU(?P<value>\d+)\b"),
        "ringattn_parallel_size": (r"\bring[=_ ](?P<value>\d+)\b", r"\bR(?P<value>\d+)\b"),
        "expert_parallel_size": (r"\bep[=_ ](?P<value>\d+)\b", r"\bEP(?P<value>\d+)\b"),
        "ep_fsdp_size": (r"\bep_fsdp[=_ ](?P<value>\d+)\b", r"\beFSDP(?P<value>\d+)\b"),
    }
    for field, patterns in field_patterns.items():
        if value := _readme_parallel_int(readme_text, field, patterns):
            defaults[field] = value
    if value := _readme_parallel_int(
        readme_text,
        "gradient_accumulation_steps",
        (r"\bgradient_accumulation_steps[=: ]+(?P<value>\d+)\b", r"\bga[=_ ](?P<value>\d+)\b"),
    ):
        defaults["gradient_accumulation_steps"] = value
    if (value := _readme_bool_from_text(readme_text, "deepep_async_combine")) is not None:
        defaults["deepep_async_combine"] = value
    if value := _readme_parallel_int(
        readme_text,
        "deepep_num_sms",
        (r"\bdeepep_num_sms[=: ]+(?P<value>\d+)\b", r"\bSMS(?P<value>\d+)\b"),
    ):
        defaults["deepep_num_sms"] = value
    if (value := _readme_float_from_text(readme_text, "deepep_buffer_size_gb")) is not None:
        defaults["deepep_buffer_size_gb"] = value
    if (value := _readme_bool_from_text(readme_text, "enable_compile")) is not None:
        defaults["enable_compile"] = value
    if value := _checkpointing_method_from_text(readme_text):
        defaults["gradient_checkpointing_method"] = value
    if (value := _readme_default_balanced_routing(readme_text)) is not None:
        defaults["balanced_routing"] = value
    return defaults


def _first_markdown_number(value: str) -> float | None:
    match = re.search(r"~?\s*(?P<value>[0-9][0-9,.]*)(?P<suffix>[KkMm]?)", value.replace("*", ""))
    if not match:
        return None
    return human_number(match.group("value") + match.group("suffix"))


def _markdown_value(values: dict[str, str], key_substring: str) -> str:
    for key, value in values.items():
        if key_substring in key:
            return value
    return ""


def _markdown_peak_gb(value: str) -> float | None:
    if "oom" in value.lower():
        return None
    return _first_markdown_number(value)


def _q235_fp32_master_64k_points(
    readme_text: str, *, source: str, model_ref: str | None
) -> list[BenchmarkBehaviorPoint]:
    if "MEMORY-DOWN WITH FP32 MASTER" not in readme_text or "pr83_opt_mom0_bf16grad" not in readme_text:
        return []

    base_kwargs: dict[str, Any] = {
        "source": source,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "gpu_count": 64,
        "model_ref": model_ref,
        "sample_packing_sequence_len": 64_000,
        "data_parallel_replicate_size": 8,
        "data_parallel_shard_size": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "ulysses_parallel_size": 8,
        "ringattn_parallel_size": 1,
        "expert_parallel_size": 8,
        "ep_fsdp_size": 8,
        "deepep_async_combine": False,
        "deepep_num_sms": 48,
        "enable_compile": False,
        "gradient_checkpointing_method": "recompute_full_layer",
        "skip_param_upcast": False,
        "ce_mode": "compiled",
        "muon_momentum": 0.0,
        "balanced_routing": False,
        "status": "historical_q235_64k_fp32_master",
    }
    return [
        BenchmarkBehaviorPoint(
            label="q235_markdown:pr83_mom0_fp32grad_64k",
            tokens_per_sec=28_500.0,
            step_time_sec=18.4,
            mfu_percent=18.8,
            tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU * 0.188,
            peak_mem_gb=60.8,
            allocator_retries=None,
            deepep_buffer_size_gb=1.0,
            enable_activation_offload=False,
            fsdp_reduce_dtype="fp32",
            correctness_status="not_promoted",
            notes=["8-node 64K fp32-master mom0 control", "not K3-gated"],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q235_markdown:pr83_mom0_activation_offload_64k",
            tokens_per_sec=27_600.0,
            step_time_sec=19.0,
            mfu_percent=18.2,
            tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU * 0.182,
            peak_mem_gb=55.0,
            allocator_retries=None,
            deepep_buffer_size_gb=1.0,
            enable_activation_offload=True,
            fsdp_reduce_dtype="fp32",
            correctness_status="not_promoted",
            notes=["8-node 64K fp32-master mom0 with activation offload", "not K3-gated"],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q235_markdown:pr83_mom0_bf16grad_64k",
            tokens_per_sec=33_300.0,
            step_time_sec=15.7,
            mfu_percent=22.0,
            tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU * 0.22,
            peak_mem_gb=62.3,
            allocator_retries=None,
            deepep_buffer_size_gb=2.0,
            enable_activation_offload=False,
            fsdp_reduce_dtype="bf16",
            correctness_status="requires_k3",
            notes=["8-node 64K fp32-master mom0 with bf16 grad-reduce", "numeric change requires K3 before promotion"],
            **base_kwargs,
        ),
    ]


def _q235_markdown_points(readme_text: str, *, source: str, model_ref: str | None) -> list[BenchmarkBehaviorPoint]:
    if "Qwen3-235B" not in readme_text or "tok/s tot" not in readme_text:
        return []

    points: list[BenchmarkBehaviorPoint] = _q235_fp32_master_64k_points(readme_text, source=source, model_ref=model_ref)
    current_header: list[str] | None = None
    current_gpu_count: int | None = None
    current_ep_size: int | None = None
    current_ep_fsdp_size: int | None = None
    current_tensor_parallel_size = 1
    current_pipeline_parallel_size = 1
    current_ulysses_parallel_size = 1
    current_ringattn_parallel_size = 1
    for line in readme_text.splitlines():
        if gpu_count := _gpu_count_from_text(line):
            current_gpu_count = gpu_count
        ep_matches = list(re.finditer(r"\bEP(?P<ep>\d+)\b", line))
        if ep_matches:
            current_ep_size = int(ep_matches[-1].group("ep"))
        efsdp_matches = list(re.finditer(r"(?:ep_fsdp|eFSDP)(?:[= ]|)(?P<efsdp>\d+)", line))
        if efsdp_matches:
            current_ep_fsdp_size = int(efsdp_matches[-1].group("efsdp"))
        if tp := _last_regex_int(
            line,
            (
                r"\bTP(?P<tp>\d+)\b",
                r"\btensor_parallel_size[:= ]+(?P<value>\d+)\b",
                r"\btp[=_](?P<tp>\d+)\b",
            ),
        ):
            current_tensor_parallel_size = tp
        if pp := _last_regex_int(
            line,
            (
                r"\bPP(?P<pp>\d+)\b",
                r"\bpipeline_parallel_size[:= ]+(?P<value>\d+)\b",
                r"\bpp[=_](?P<pp>\d+)\b",
            ),
        ):
            current_pipeline_parallel_size = pp
        if ulysses := _last_regex_int(
            line,
            (
                r"\bU(?P<u>\d+)\b",
                r"\bul[y]?sses_parallel_size[:= ]+(?P<value>\d+)\b",
                r"\bu[=_]?(?P<u>\d+)\b",
            ),
        ):
            current_ulysses_parallel_size = ulysses
        if ringattn := _last_regex_int(
            line,
            (
                r"\bR(?P<ring>\d+)\b",
                r"\bringattn_parallel_size[:= ]+(?P<value>\d+)\b",
                r"\bring[=_]?(?P<ring>\d+)\b",
            ),
        ):
            current_ringattn_parallel_size = ringattn
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        lowered = [cell.lower() for cell in cells]
        if "run" in lowered and "tok/s tot" in lowered:
            current_header = lowered
            continue
        if current_header is None or set(cells) == {"---"} or not cells:
            continue
        values = dict(zip(current_header, cells, strict=False))
        run = values.get("run", "").replace("*", "").strip("` ")
        if not run or run.lower() in {"run", "-----"}:
            continue
        status_text = _markdown_value(values, "status")
        is_failure = "oom" in status_text.lower() or "fail" in status_text.lower()
        tokens_per_sec = _first_markdown_number(_markdown_value(values, "tok/s tot"))
        tok_step = _first_markdown_number(_markdown_value(values, "tok/step"))
        pack = _first_markdown_number(_markdown_value(values, "pack"))
        if tok_step is None or pack in (None, 0):
            continue
        if tokens_per_sec is None and not is_failure:
            continue
        global_batch_size = int(round(tok_step / pack))
        step_time_sec = _first_markdown_number(_markdown_value(values, "step s"))
        mfu_percent = _first_markdown_number(_markdown_value(values, "mfu"))
        peak_mem_gb = _markdown_peak_gb(_markdown_value(values, "peak gb"))
        skip_param_upcast = "no skipupcast" not in run.lower()
        deepep_buffer_size_gb = 4.0 if "buffer 4" in status_text.lower() or "pk8192_fix" in run.lower() else 2.0
        points.append(
            BenchmarkBehaviorPoint(
                label=f"q235_markdown:{run}",
                source=source,
                micro_batch_size=int(_first_markdown_number(values.get("mbs", "")) or 1),
                global_batch_size=global_batch_size,
                tokens_per_sec=tokens_per_sec,
                step_time_sec=step_time_sec,
                phase_time_sec=_float_dict({}),
                phase_time_share=_float_dict({}),
                mfu_percent=mfu_percent,
                peak_mem_gb=peak_mem_gb,
                allocator_retries=None,
                gpu_count=current_gpu_count,
                model_ref=model_ref,
                sample_packing_sequence_len=int(pack),
                tensor_parallel_size=current_tensor_parallel_size,
                pipeline_parallel_size=current_pipeline_parallel_size,
                ulysses_parallel_size=current_ulysses_parallel_size,
                ringattn_parallel_size=current_ringattn_parallel_size,
                expert_parallel_size=current_ep_size,
                ep_fsdp_size=current_ep_fsdp_size,
                deepep_async_combine=False,
                deepep_num_sms=24,
                deepep_buffer_size_gb=deepep_buffer_size_gb,
                enable_compile=False,
                gradient_checkpointing_method="recompute_before_dispatch",
                skip_param_upcast=skip_param_upcast,
                fsdp_reduce_dtype="fp32",
                ce_mode="quack_linear",
                moe_implementation="quack",
                muon_momentum=0.95,
                balanced_routing=False,
                status="historical_q235_markdown_oom" if is_failure else "historical_q235_markdown",
                correctness_status="oom" if is_failure else "not_promoted",
                notes=[status_text] if status_text else [],
            )
        )
    return points


def _result_metric_value(readme_text: str, metric: str) -> str | None:
    pattern = rf"\|\s*(?:\*\*)?{re.escape(metric)}(?:\*\*)?[^|]*\|\s*(?P<value>[^|]+)\|"
    match = re.search(pattern, readme_text, re.IGNORECASE)
    return match.group("value").strip() if match else None


def _first_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", value.replace(",", ""))
    return int(match.group(0)) if match else None


def _q35_headroom_phase_shares(readme_text: str) -> dict[str, float]:
    lowered = readme_text.lower()
    if "comm/data-movement-bound" not in lowered:
        return {}
    comm_match = re.search(r"~(?P<value>[0-9.]+)%\s*:\s*moe a2a", lowered)
    gemm_match = re.search(r"only\s*~(?P<value>[0-9.]+)%\s*gemm", lowered)
    shares: dict[str, float] = {}
    if comm_match:
        shares["communication_data_movement"] = round(float(comm_match.group("value")) / 100.0, 3)
    if gemm_match:
        shares["model_forward_gemm"] = round(float(gemm_match.group("value")) / 100.0, 3)
    attributed = sum(shares.values())
    if shares and attributed < 1.0:
        shares["other_unattributed"] = round(1.0 - attributed, 3)
    return shares


def _q35_markdown_points(readme_text: str, *, source: str, model_ref: str | None) -> list[BenchmarkBehaviorPoint]:
    if not readme_text.lstrip().startswith("# Qwen3.5-35B-A3B"):
        return []

    gpu_count = _gpu_count_from_text(readme_text) or 16
    seq_len = 65_536
    headroom_phase_share = _q35_headroom_phase_shares(readme_text)
    baseline_step_time_sec = float(
        _first_markdown_number(_result_metric_value(readme_text, "mean step time") or "71.8") or 71.8
    )
    base_kwargs: dict[str, Any] = {
        "gpu_count": gpu_count,
        "model_ref": model_ref,
        "sample_packing_sequence_len": seq_len,
        "data_parallel_replicate_size": 2,
        "data_parallel_shard_size": 2,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "ulysses_parallel_size": 4,
        "ringattn_parallel_size": 1,
        "expert_parallel_size": 8,
        "ep_fsdp_size": 2,
        "deepep_async_combine": False,
        "deepep_num_sms": 24,
        "deepep_buffer_size_gb": 2.0,
        "enable_compile": True,
        "ce_mode": "quack_linear",
        "moe_implementation": "quack",
    }
    points = [
        BenchmarkBehaviorPoint(
            label="q35_markdown:n2_u4_ga16_compile",
            source=source,
            micro_batch_size=1,
            global_batch_size=64,
            tokens_per_sec=float(
                _first_markdown_number(_result_metric_value(readme_text, "mean tokens/s") or "47352") or 47_352.0
            ),
            step_time_sec=baseline_step_time_sec,
            phase_time_share=headroom_phase_share,
            mfu_percent=float(_first_markdown_number(_result_metric_value(readme_text, "mean MFU") or "8.97") or 8.97),
            tflops_per_gpu=float(
                _first_markdown_number(_result_metric_value(readme_text, "mean TFLOPS/GPU") or "88.7") or 88.7
            ),
            peak_mem_gb=float(
                _first_markdown_number(_result_metric_value(readme_text, "peak memory / rank") or "51.2") or 51.2
            ),
            allocator_retries=None,
            measured_steps=_first_int(_result_metric_value(readme_text, "measured steps")) or 11,
            warmup_steps=_first_int(_result_metric_value(readme_text, "warmup excluded")) or 3,
            gradient_checkpointing_method="recompute_before_dispatch",
            fsdp_reduce_dtype="fp32",
            balanced_routing=False,
            correctness_status="not_promoted",
            status="historical_q35_markdown",
            notes=[
                "real CoderForge 65k packs",
                "clean completed 2-node reproduction",
                "markdown headroom analysis: ~60% communication/data movement, ~18% GEMM",
            ],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q35_markdown:no_recompute",
            source=source,
            micro_batch_size=1,
            global_batch_size=64,
            tokens_per_sec=46_358.0,
            step_time_sec=(64 * seq_len / 46_358.0),
            phase_time_share=headroom_phase_share,
            mfu_percent=None,
            tflops_per_gpu=87.0,
            peak_mem_gb=51.2,
            allocator_retries=None,
            gradient_checkpointing_method="no_recompute",
            fsdp_reduce_dtype="fp32",
            balanced_routing=False,
            correctness_status="not_promoted",
            status="historical_q35_headroom",
            notes=[
                "no_recompute was throughput-neutral versus before_dispatch",
                "markdown headroom analysis: ~60% communication/data movement, ~18% GEMM",
            ],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q35_markdown:mbs2_ga8_oom",
            source=source,
            micro_batch_size=2,
            global_batch_size=64,
            tokens_per_sec=None,
            step_time_sec=None,
            mfu_percent=None,
            tflops_per_gpu=None,
            peak_mem_gb=None,
            allocator_retries=None,
            gradient_checkpointing_method="recompute_before_dispatch",
            fsdp_reduce_dtype="fp32",
            balanced_routing=False,
            correctness_status="oom",
            status="historical_q35_markdown_oom",
            notes=["ragged real pack spikes past 80GB under mbs2"],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q35_markdown:bf16red_not_loss_safe",
            source=source,
            micro_batch_size=1,
            global_batch_size=64,
            tokens_per_sec=48_488.0,
            step_time_sec=(64 * seq_len / 48_488.0),
            phase_time_share=headroom_phase_share,
            mfu_percent=None,
            tflops_per_gpu=90.9,
            peak_mem_gb=50.5,
            allocator_retries=None,
            gradient_checkpointing_method="recompute_before_dispatch",
            fsdp_reduce_dtype="bf16",
            balanced_routing=False,
            correctness_status="not_loss_safe",
            status="historical_q35_headroom_not_loss_safe",
            notes=[
                "fsdp_reduce_dtype=bf16 underflowed grad_norm; throughput curiosity only",
                "markdown headroom analysis: ~60% communication/data movement, ~18% GEMM",
            ],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q35_markdown:balanced_mbs1",
            source=source,
            micro_batch_size=1,
            global_batch_size=64,
            tokens_per_sec=53_414.0,
            step_time_sec=(64 * seq_len / 53_414.0),
            mfu_percent=10.1,
            tflops_per_gpu=100.2,
            peak_mem_gb=39.8,
            allocator_retries=None,
            gradient_checkpointing_method="recompute_before_dispatch",
            fsdp_reduce_dtype="fp32",
            balanced_routing=True,
            correctness_status="synthetic_routing_not_loss_valid",
            status="historical_q35_balanced_routing",
            notes=["XORL_MOE_SYNTHETIC_ROUTING=balanced; throughput ceiling only"],
            **base_kwargs,
        ),
        BenchmarkBehaviorPoint(
            label="q35_markdown:balanced_mbs2",
            source=source,
            micro_batch_size=2,
            global_batch_size=64,
            tokens_per_sec=52_416.0,
            step_time_sec=(64 * seq_len / 52_416.0),
            mfu_percent=10.0,
            tflops_per_gpu=99.0,
            peak_mem_gb=52.6,
            allocator_retries=None,
            gradient_checkpointing_method="recompute_before_dispatch",
            fsdp_reduce_dtype="fp32",
            balanced_routing=True,
            correctness_status="synthetic_routing_not_loss_valid",
            status="historical_q35_balanced_routing",
            notes=["balanced routing makes mbs2 fit; throughput-neutral versus balanced mbs1"],
            **base_kwargs,
        ),
    ]
    return points


def _best_by_mfu_point(
    result_path: Path,
    result: dict[str, Any],
    row: dict[str, Any],
    *,
    topology_defaults: dict[str, int | float | bool | str],
) -> BenchmarkBehaviorPoint:
    trial = str(row["trial"])
    caveat = row.get("caveat")
    k3_gate = row.get("k3_gate")
    notes = []
    if caveat:
        notes.append(str(caveat))
    if k3_gate:
        notes.append(str(k3_gate))
    correctness_status = None
    if k3_gate and str(k3_gate).startswith("pass"):
        correctness_status = "k3_pass"
    elif _first_non_none(row.get("deepep_async_combine"), _trial_deepep_async_combine(trial)):
        correctness_status = "raw_speed_not_promoted_without_matching_k3_pass"
    return BenchmarkBehaviorPoint(
        label=f"best_by_mfu:{trial}",
        source=str(result_path),
        micro_batch_size=row.get("micro_batch_size"),
        global_batch_size=row.get("global_batch_size"),
        tokens_per_sec=row.get("tokens_per_sec"),
        step_time_sec=row.get("step_time_sec"),
        tokens_per_sec_std=_first_non_none(row.get("tokens_per_sec_std"), row.get("tokens_per_sec_stdev")),
        tokens_per_sec_cv=row.get("tokens_per_sec_cv"),
        step_time_sec_std=_first_non_none(row.get("step_time_sec_std"), row.get("step_time_s_std")),
        step_time_sec_cv=_first_non_none(row.get("step_time_sec_cv"), row.get("step_time_s_cv")),
        phase_time_sec=_float_dict(row.get("phase_time_sec")),
        phase_time_rank_mean_sec=_float_dict(row.get("phase_time_rank_mean_sec")),
        phase_time_share=_float_dict(row.get("phase_time_share")),
        mfu_percent=row.get("mfu_percent"),
        tflops_per_gpu=row.get("mean_tflops_per_gpu"),
        peak_mem_gb=None,
        allocator_retries=None,
        measured_steps=row.get("measured_steps"),
        warmup_steps=row.get("warmup_steps"),
        gpu_count=row.get("gpus") or _gpu_count_from_text(str(result.get("workload", ""))),
        model_ref=_first_non_none(row.get("model_ref"), topology_defaults.get("model_ref")),
        sample_packing_sequence_len=_first_non_none(
            row.get("sample_packing_sequence_len"), topology_defaults.get("sample_packing_sequence_len")
        ),
        data_parallel_replicate_size=_first_non_none(
            row.get("data_parallel_replicate_size"), topology_defaults.get("data_parallel_replicate_size")
        ),
        data_parallel_shard_size=_first_non_none(
            row.get("data_parallel_shard_size"), topology_defaults.get("data_parallel_shard_size")
        ),
        tensor_parallel_size=_first_non_none(
            row.get("tensor_parallel_size"), topology_defaults.get("tensor_parallel_size")
        ),
        pipeline_parallel_size=_first_non_none(
            row.get("pipeline_parallel_size"), topology_defaults.get("pipeline_parallel_size")
        ),
        ulysses_parallel_size=_first_non_none(
            row.get("ulysses_parallel_size"), topology_defaults.get("ulysses_parallel_size")
        ),
        ringattn_parallel_size=_first_non_none(
            row.get("ringattn_parallel_size"), topology_defaults.get("ringattn_parallel_size")
        ),
        expert_parallel_size=_first_non_none(
            row.get("expert_parallel_size"), topology_defaults.get("expert_parallel_size")
        ),
        ep_fsdp_size=_first_non_none(row.get("ep_fsdp"), topology_defaults.get("ep_fsdp_size")),
        deepep_async_combine=_first_non_none(
            row.get("deepep_async_combine"),
            _trial_deepep_async_combine(trial),
            topology_defaults.get("deepep_async_combine"),
        ),
        deepep_num_sms=_first_non_none(
            row.get("deepep_num_sms"), _trial_sms_count(trial), topology_defaults.get("deepep_num_sms")
        ),
        deepep_buffer_size_gb=_first_non_none(
            row.get("deepep_buffer_size_gb"),
            _trial_buffer_size_gb(trial),
            topology_defaults.get("deepep_buffer_size_gb"),
        ),
        enable_compile=_first_non_none(
            row.get("enable_compile"), _trial_compile_enabled(trial), topology_defaults.get("enable_compile")
        ),
        gradient_checkpointing_method=_first_non_none(
            row.get("gradient_checkpointing_method"),
            _trial_checkpointing_method(trial),
            topology_defaults.get("gradient_checkpointing_method"),
        ),
        enable_activation_offload=_first_non_none(
            row.get("enable_activation_offload"),
            _trial_activation_offload(trial),
            topology_defaults.get("enable_activation_offload"),
        ),
        activation_offload_prefetch_count=_first_non_none(
            row.get("activation_offload_prefetch_count"),
            _trial_prefetch_count(trial),
            topology_defaults.get("activation_offload_prefetch_count"),
        ),
        fsdp_reduce_dtype=_first_non_none(row.get("fsdp_reduce_dtype"), topology_defaults.get("fsdp_reduce_dtype")),
        ce_mode=_first_non_none(row.get("ce_mode"), _trial_ce_mode(trial), topology_defaults.get("ce_mode")),
        moe_implementation=_first_non_none(
            row.get("moe_implementation"),
            _trial_moe_implementation(trial),
            topology_defaults.get("moe_implementation"),
        ),
        moe_checkpoint_method=_first_non_none(
            row.get("moe_checkpoint_method"), topology_defaults.get("moe_checkpoint_method")
        ),
        muon_update_dtype=_first_non_none(
            row.get("muon_update_dtype"),
            _trial_muon_update_dtype(trial),
            topology_defaults.get("muon_update_dtype"),
        ),
        attention_backend=_first_non_none(row.get("attention_backend"), topology_defaults.get("attention_backend")),
        balanced_routing=_first_non_none(
            row.get("balanced_routing"),
            row.get("synthetic_balanced_routing"),
            topology_defaults.get("balanced_routing"),
        ),
        status="autotune_result",
        correctness_status=correctness_status,
        notes=notes,
    )


def _load_startup_metrics(run_dir: Path) -> dict[str, Any]:
    startup_path = run_dir / "startup_metrics.json"
    if not startup_path.is_file():
        return {}
    return json.loads(startup_path.read_text(encoding="utf-8"))


def _startup_master_log_path(benchmark_path: Path, startup_metrics: dict[str, Any]) -> Path | None:
    metrics = startup_metrics.get("metrics", {})
    master_addr = metrics.get("startup/master_addr")
    if not isinstance(master_addr, str) or not master_addr:
        return None
    run_name = master_addr.removesuffix("-master")
    return benchmark_path / run_name / "node-0.log"


def _resolved_run_log_path(
    benchmark_path: Path,
    run_dir: Path,
    startup_metrics: dict[str, Any],
    *,
    log_paths: list[Path] | None = None,
) -> Path | None:
    paths = _resolved_run_log_paths(benchmark_path, run_dir, startup_metrics, log_paths=log_paths)
    return paths[0] if paths else None


def _resolved_run_log_paths(
    benchmark_path: Path,
    run_dir: Path,
    startup_metrics: dict[str, Any],
    *,
    log_paths: list[Path] | None = None,
) -> list[Path]:
    explicit_paths = [path for path in log_paths or [] if path.is_file()]
    if explicit_paths:
        return explicit_paths
    candidates = [
        run_dir / "node-0.log",
        _startup_master_log_path(benchmark_path, startup_metrics),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return [candidate]
    return []


def _read_benchmark_source_manifest(benchmark_path: Path) -> tuple[Path, dict[str, Any]] | None:
    for filename in BENCHMARK_SOURCE_MANIFESTS:
        manifest_path = benchmark_path / filename
        if not manifest_path.is_file():
            continue
        if manifest_path.suffix == ".json":
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return manifest_path, {}
        return manifest_path, payload
    return None


def _read_manifest_from_candidates(
    benchmark_path: Path, filenames: tuple[str, ...]
) -> tuple[Path, dict[str, Any]] | None:
    for filename in filenames:
        manifest_path = benchmark_path / filename
        if not manifest_path.is_file():
            continue
        if manifest_path.suffix == ".json":
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return manifest_path, {}
        return manifest_path, payload
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _resolve_manifest_path(base_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (base_path / path).resolve(strict=False)


def _expand_manifest_path_patterns(base_path: Path, value: Any) -> list[Path]:
    paths: list[Path] = []
    for item in _as_list(value):
        path = _resolve_manifest_path(base_path, item)
        if path is None:
            continue
        pattern = str(path)
        if any(token in pattern for token in ("*", "?", "[")):
            paths.extend(Path(match) for match in sorted(glob.glob(pattern, recursive=True)))
        else:
            paths.append(path)
    return paths


def _resolved_run_label(config_path: Path, label_root: Path, label_prefix: str | None) -> str:
    try:
        relative = config_path.parent.relative_to(label_root)
    except ValueError:
        relative = config_path.parent.name
    if label_prefix:
        return f"resolved_run:{label_prefix}/{relative}"
    return f"resolved_run:{relative}"


def _log_paths_for_manifest_run(
    manifest_dir: Path,
    root: Path,
    config_path: Path,
    source: dict[str, Any],
) -> list[Path]:
    log_root = _resolve_manifest_path(manifest_dir, source.get("log_root"))
    log_base = log_root if log_root is not None else manifest_dir
    log_paths_by_run = source.get("log_paths_by_run", source.get("logs_by_run", {}))
    if not isinstance(log_paths_by_run, dict):
        return []
    try:
        relative_key = config_path.parent.relative_to(root).as_posix()
    except ValueError:
        relative_key = config_path.parent.name
    run_keys = tuple(dict.fromkeys((relative_key, config_path.parent.name)))
    paths: list[Path] = []
    for run_key in run_keys:
        if run_key in log_paths_by_run:
            paths.extend(_expand_manifest_path_patterns(log_base, log_paths_by_run[run_key]))
    return paths


def _manifest_run_keys(root: Path, config_path: Path) -> tuple[str, str]:
    try:
        relative_key = config_path.parent.relative_to(root).as_posix()
    except ValueError:
        relative_key = config_path.parent.name
    return relative_key, config_path.parent.name


def _manifest_run_int(
    root: Path,
    config_path: Path,
    source: dict[str, Any],
    *,
    field_name: str,
    by_run_field_name: str,
) -> int | None:
    by_run = source.get(by_run_field_name, {})
    if isinstance(by_run, dict):
        for run_key in _manifest_run_keys(root, config_path):
            value = by_run.get(run_key)
            if value is not None:
                return int(value)
    value = source.get(field_name)
    return int(value) if value is not None else None


def _manifest_run_value(
    root: Path,
    config_path: Path,
    source: dict[str, Any],
    *,
    field_name: str,
    by_run_field_name: str,
) -> Any | None:
    by_run = source.get(by_run_field_name, {})
    if isinstance(by_run, dict):
        for run_key in _manifest_run_keys(root, config_path):
            if run_key in by_run:
                return by_run[run_key]
    return source.get(field_name)


def _manifest_run_metrics_only_reason(
    root: Path,
    config_path: Path,
    source: dict[str, Any],
) -> str | None:
    value = _manifest_run_value(
        root,
        config_path,
        source,
        field_name="metrics_only",
        by_run_field_name="metrics_only_by_run",
    )
    if value is None:
        value = _manifest_run_value(
            root,
            config_path,
            source,
            field_name="exclude_throughput",
            by_run_field_name="exclude_throughput_by_run",
        )
    if value in (None, False):
        return None
    if value is True:
        return "metrics_only"
    return str(value)


def _log_failure_status(text: str) -> str | None:
    lowered = text.lower()
    if "outofmemoryerror" in lowered or "cuda out of memory" in lowered:
        return "oom"
    if (
        "childfailederror" in lowered
        or "traceback" in lowered
        or "distbackenderror" in lowered
        or "watchdog caught collective operation timeout" in lowered
        or "indentationerror" in lowered
    ):
        return "runtime_failure_after_steps"
    return None


def _oom_peak_mem_gb(text: str) -> float | None:
    values = [
        float(match.group("value"))
        for match in re.finditer(r"process has (?P<value>\d+(?:\.\d+)?)\s+GiB memory in use", text)
    ]
    return max(values) if values else None


def _round_or_none(value: Any, ndigits: int) -> float | None:
    return round(float(value), ndigits) if value is not None else None


RESOLVED_CONFIG_RE = re.compile(r"(?m)^resolved_config=(?P<path>\S+)\s*$")


def _same_resolved_config_path(raw_path: str, config_path: Path) -> bool:
    marker_path = Path(raw_path).expanduser()
    if marker_path == config_path:
        return True
    try:
        if marker_path.resolve() == config_path.resolve():
            return True
    except OSError:
        pass
    if marker_path.is_file() and config_path.is_file():
        try:
            if runtime_training_config(load_training_config(marker_path)) == runtime_training_config(
                load_training_config(config_path)
            ):
                return True
        except (OSError, ValueError):
            pass
    marker_stem = _normalized_config_token(marker_path.stem)
    run_dir_name = _normalized_config_token(config_path.parent.name)
    if marker_stem and run_dir_name and (run_dir_name in marker_stem or marker_stem in run_dir_name):
        return True
    return str(marker_path) == str(config_path)


def _normalized_config_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _log_segment_for_resolved_config(log_text: str, config_path: Path) -> tuple[str, list[str]]:
    markers = list(RESOLVED_CONFIG_RE.finditer(log_text))
    if not markers:
        return log_text, []

    matching_indexes = [
        index for index, marker in enumerate(markers) if _same_resolved_config_path(marker.group("path"), config_path)
    ]
    if not matching_indexes:
        return "", [f"log_segments_total={len(markers)}", "log_segment_selected=none_for_resolved_config"]

    selected_index = matching_indexes[-1]
    segment_end = markers[selected_index + 1].start() if selected_index + 1 < len(markers) else len(log_text)
    notes = [
        f"log_segments_total={len(markers)}",
        f"log_matching_segments={len(matching_indexes)}",
        f"log_segment_selected={matching_indexes.index(selected_index) + 1}_of_{len(matching_indexes)}",
    ]
    return log_text[markers[selected_index].start() : segment_end], notes


def _summary_values(summaries: list[dict[str, Any]], key: str) -> list[float]:
    return [float(summary[key]) for summary in summaries if isinstance(summary.get(key), (int, float))]


def _summary_mean(summaries: list[dict[str, Any]], key: str) -> float | None:
    values = _summary_values(summaries, key)
    return statistics.fmean(values) if values else None


def _summary_max(summaries: list[dict[str, Any]], key: str) -> float | None:
    values = _summary_values(summaries, key)
    return max(values) if values else None


def _summary_min(summaries: list[dict[str, Any]], key: str) -> int | None:
    values = [int(summary[key]) for summary in summaries if isinstance(summary.get(key), int)]
    return min(values) if values else None


def _summary_dict_metric(summaries: list[dict[str, Any]], key: str, *, method: str = "mean") -> dict[str, float]:
    values_by_name: dict[str, list[float]] = {}
    for summary in summaries:
        values = summary.get(key)
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, (int, float)):
                values_by_name.setdefault(str(name), []).append(float(value))
    result: dict[str, float] = {}
    for name, values in sorted(values_by_name.items()):
        if not values:
            continue
        result[name] = max(values) if method == "max" else statistics.fmean(values)
    return result


def _aggregate_resolved_run_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    if len(summaries) == 1:
        return summaries[0]
    return {
        "tokens_per_sec_mean": _summary_mean(summaries, "tokens_per_sec_mean"),
        "step_time_s_mean": _summary_mean(summaries, "step_time_s_mean"),
        "tokens_per_sec_std": _summary_mean(summaries, "tokens_per_sec_std"),
        "tokens_per_sec_cv": _summary_mean(summaries, "tokens_per_sec_cv"),
        "step_time_s_std": _summary_mean(summaries, "step_time_s_std"),
        "step_time_s_cv": _summary_mean(summaries, "step_time_s_cv"),
        "tokens_per_step_median": _summary_mean(summaries, "tokens_per_step_median"),
        "phase_time_sec": _summary_dict_metric(summaries, "phase_time_sec"),
        "phase_time_rank_mean_sec": _summary_dict_metric(summaries, "phase_time_rank_mean_sec"),
        "phase_time_share": _summary_dict_metric(summaries, "phase_time_share"),
        "phase_memory_peak_gb": _summary_dict_metric(summaries, "phase_memory_peak_gb", method="max"),
        "peak_mem_gb_max": _summary_max(summaries, "peak_mem_gb_max"),
        "mfu_mean": _summary_mean(summaries, "mfu_mean"),
        "tflops_per_gpu_mean": _summary_mean(summaries, "tflops_per_gpu_mean"),
        "measured_steps": _summary_min(summaries, "measured_steps"),
        "warmup_excluded": _summary_min(summaries, "warmup_excluded"),
        "parsed_step_count": _summary_min(summaries, "parsed_step_count"),
    }


def _combined_failure_status(log_texts: list[str]) -> str | None:
    statuses = [status for text in log_texts if (status := _log_failure_status(text)) is not None]
    if "oom" in statuses:
        return "oom"
    return statuses[0] if statuses else None


def _resolved_run_behavior_point(
    benchmark_path: Path,
    config_path: Path,
    *,
    label_root: Path | None = None,
    label_prefix: str | None = None,
    label: str | None = None,
    log_paths: list[Path] | None = None,
    warmup_steps: int | None = None,
    metrics_only_reason: str | None = None,
    notes: list[str] | None = None,
) -> BenchmarkBehaviorPoint | None:
    run_dir = config_path.parent
    raw_config = load_training_config(config_path)
    try:
        topology = resolve_topology(raw_config)
    except ValueError:
        return None

    startup_metrics = _load_startup_metrics(run_dir)
    resolved_log_paths = _resolved_run_log_paths(benchmark_path, run_dir, startup_metrics, log_paths=log_paths)
    log_texts: list[str] = []
    log_segment_notes: list[str] = []
    observed_summaries: list[dict[str, Any]] = []
    for log_path in resolved_log_paths:
        full_log_text = log_path.read_text(encoding="utf-8", errors="replace")
        log_text, segment_notes = _log_segment_for_resolved_config(full_log_text, config_path)
        log_texts.append(log_text)
        log_segment_notes.extend(segment_notes)
        observed = parse_log_text(log_text, source=str(log_path))
        selected_warmup_steps = warmup_steps if warmup_steps is not None else 2 if len(observed.steps) > 2 else 0
        observed_summaries.append(
            summarize_observed_run(
                observed,
                warmup_steps=selected_warmup_steps,
                world_size=topology.world_size,
            )
        )
    failure_status = _combined_failure_status(log_texts)
    observed_summary = _aggregate_resolved_run_summaries(observed_summaries)

    tokens_per_sec = _round_or_none(observed_summary.get("tokens_per_sec_mean"), 3)
    step_time_sec = _round_or_none(observed_summary.get("step_time_s_mean"), 6)
    tokens_per_step = _round_or_none(observed_summary.get("tokens_per_step_median"), 1)
    tokens_per_sec_std = _round_or_none(observed_summary.get("tokens_per_sec_std"), 3)
    tokens_per_sec_cv = _round_or_none(observed_summary.get("tokens_per_sec_cv"), 6)
    step_time_sec_std = _round_or_none(observed_summary.get("step_time_s_std"), 6)
    step_time_sec_cv = _round_or_none(observed_summary.get("step_time_s_cv"), 6)
    phase_time_sec = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_time_sec")).items()
    }
    phase_time_rank_mean_sec = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_time_rank_mean_sec")).items()
    }
    phase_time_share = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_time_share")).items()
    }
    phase_memory_peak_gb = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_memory_peak_gb")).items()
    }
    peak_mem_gb = _round_or_none(observed_summary.get("peak_mem_gb_max"), 3)
    if failure_status == "oom":
        oom_peak_mem_gb = _round_or_none(max((_oom_peak_mem_gb(text) or 0.0) for text in log_texts), 3)
        if oom_peak_mem_gb is not None:
            peak_mem_gb = max(value for value in (peak_mem_gb, oom_peak_mem_gb) if value is not None)
    observed_has_metrics = peak_mem_gb is not None or bool(phase_time_sec) or bool(phase_memory_peak_gb)
    if metrics_only_reason is not None and failure_status is None:
        tokens_per_sec = None
        step_time_sec = None
        tokens_per_sec_std = None
        tokens_per_sec_cv = None
        step_time_sec_std = None
        step_time_sec_cv = None
        tokens_per_step = None
    if (
        tokens_per_sec is None
        and failure_status is None
        and not (metrics_only_reason is not None and observed_has_metrics)
    ):
        return None

    if failure_status == "oom" and tokens_per_sec is None:
        status = "observed_log_oom"
        correctness_status = "oom"
    elif failure_status is not None:
        status = "observed_log_partial_failure"
        correctness_status = failure_status
    elif metrics_only_reason is not None:
        status = "observed_log_metrics_only"
        correctness_status = "not_promoted"
    else:
        status = "observed_log_summary"
        correctness_status = "not_promoted"

    metrics = startup_metrics.get("metrics", {})
    source_paths = [str(path) for path in resolved_log_paths]
    point_notes = [
        f"warmup_excluded={observed_summary.get('warmup_excluded', 0)}",
        f"parsed_steps={observed_summary.get('parsed_step_count', 0)}",
        f"resolved_log_count={len(resolved_log_paths)}",
        *dict.fromkeys(log_segment_notes),
        *(notes or []),
    ]
    if startup_metrics.get("repo_commit"):
        point_notes.append(f"commit={startup_metrics['repo_commit']}")
    if isinstance(metrics.get("startup/master_addr"), str):
        point_notes.append(f"master_addr={metrics['startup/master_addr']}")
    if failure_status is not None:
        point_notes.append(f"log_failure_status={failure_status}")
    if metrics_only_reason is not None and failure_status is None:
        point_notes.append(f"metrics_only={metrics_only_reason}")

    return BenchmarkBehaviorPoint(
        label=label or _resolved_run_label(config_path, label_root or benchmark_path, label_prefix),
        source=";".join(source_paths) if source_paths else str(config_path),
        micro_batch_size=topology.micro_batch_size,
        global_batch_size=topology.global_batch_size,
        gradient_accumulation_steps=topology.gradient_accumulation_steps,
        tokens_per_sec=tokens_per_sec,
        step_time_sec=step_time_sec,
        tokens_per_sec_std=tokens_per_sec_std,
        tokens_per_sec_cv=tokens_per_sec_cv,
        step_time_sec_std=step_time_sec_std,
        step_time_sec_cv=step_time_sec_cv,
        tokens_per_step=tokens_per_step,
        phase_time_sec=phase_time_sec,
        phase_time_rank_mean_sec=phase_time_rank_mean_sec,
        phase_time_share=phase_time_share,
        phase_memory_peak_gb=phase_memory_peak_gb,
        mfu_percent=_round_or_none((observed_summary.get("mfu_mean") or 0.0) * 100.0, 3)
        if observed_summary.get("mfu_mean") is not None and metrics_only_reason is None
        else None,
        tflops_per_gpu=_round_or_none(observed_summary.get("tflops_per_gpu_mean"), 3)
        if metrics_only_reason is None
        else None,
        peak_mem_gb=peak_mem_gb,
        allocator_retries=None,
        measured_steps=observed_summary.get("measured_steps"),
        warmup_steps=observed_summary.get("warmup_excluded"),
        gpu_count=topology.world_size,
        model_ref=_first_non_none(
            _model_ref_from_text_or_path(json.dumps(raw_config, sort_keys=True), config_path),
            model_ref_from_config(raw_config),
        ),
        sample_packing_sequence_len=topology.sample_packing_sequence_len,
        data_parallel_replicate_size=topology.data_parallel_replicate_size,
        data_parallel_shard_size=topology.data_parallel_shard_size,
        tensor_parallel_size=topology.tensor_parallel_size,
        pipeline_parallel_size=topology.pipeline_parallel_size,
        ulysses_parallel_size=topology.ulysses_parallel_size,
        ringattn_parallel_size=topology.ringattn_parallel_size,
        expert_parallel_size=topology.expert_parallel_size,
        ep_fsdp_size=topology.ep_fsdp_size,
        deepep_async_combine=_config_bool(raw_config, "model", "deepep_async_combine", False),
        deepep_num_sms=_config_int(raw_config, "model", "deepep_num_sms"),
        deepep_buffer_size_gb=_config_float(raw_config, "model", "deepep_buffer_size_gb"),
        enable_compile=_config_bool(raw_config, "train", "enable_compile", False),
        gradient_checkpointing_method=_config_str(raw_config, "train", "gradient_checkpointing_method"),
        enable_activation_offload=_config_bool(raw_config, "train", "enable_activation_offload", False),
        activation_offload_prefetch_count=_config_int(raw_config, "train", "activation_offload_prefetch_count"),
        skip_param_upcast=_config_bool(raw_config, "train", "skip_param_upcast", False),
        fsdp_reduce_dtype=_config_fsdp_reduce_dtype(raw_config),
        ce_mode=_config_ce_mode(raw_config),
        moe_implementation=_config_str(raw_config, "model", "moe_implementation"),
        moe_checkpoint_method=_config_str(raw_config, "train", "moe_checkpoint_method"),
        muon_momentum=_config_float(raw_config, "train", "muon_momentum"),
        muon_update_dtype=_config_str(raw_config, "train", "muon_update_dtype"),
        attention_backend=_config_attention_backend(raw_config),
        balanced_routing=_config_balanced_routing(raw_config),
        status=status,
        correctness_status=correctness_status,
        notes=point_notes,
    )


def _resolved_run_points(benchmark_path: Path) -> list[BenchmarkBehaviorPoint]:
    points: list[BenchmarkBehaviorPoint] = []
    for config_path in sorted(benchmark_path.rglob("xorl_cli.yaml")):
        if not config_path.is_file():
            continue
        point = _resolved_run_behavior_point(benchmark_path, config_path)
        if point is not None:
            points.append(point)
    return points


def _logged_training_config(log_text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    starts = [match.start("brace") for match in re.finditer(r"xorl\.trainers\.trainer\s+-\s+(?P<brace>\{)", log_text)]
    for start in reversed(starts):
        try:
            parsed, _ = decoder.raw_decode(log_text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and {"model", "train", "data"}.issubset(parsed):
            return parsed
    return None


def _standalone_log_label(log_path: Path, benchmark_path: Path) -> str:
    try:
        relative = log_path.parent.relative_to(benchmark_path)
    except ValueError:
        relative = log_path.parent.name
    return f"observed_log:{relative.as_posix() if isinstance(relative, Path) else relative}"


def _standalone_log_behavior_point(
    benchmark_path: Path,
    log_path: Path,
) -> BenchmarkBehaviorPoint | None:
    full_log_text = log_path.read_text(encoding="utf-8", errors="replace")
    raw_config = _logged_training_config(full_log_text)
    if raw_config is None:
        return None
    try:
        topology = resolve_topology(raw_config)
    except ValueError:
        return None

    failure_status = _log_failure_status(full_log_text)
    observed = parse_log_text(full_log_text, source=str(log_path))
    warmup_steps = 2 if len(observed.steps) > 2 else 0
    observed_summary = summarize_observed_run(
        observed,
        warmup_steps=warmup_steps,
        world_size=topology.world_size,
    )
    tokens_per_sec = _round_or_none(observed_summary.get("tokens_per_sec_mean"), 3)
    peak_mem_gb = _round_or_none(observed_summary.get("peak_mem_gb_max"), 3)
    if failure_status == "oom":
        oom_peak_mem_gb = _round_or_none(_oom_peak_mem_gb(full_log_text), 3)
        if oom_peak_mem_gb is not None:
            peak_mem_gb = max(value for value in (peak_mem_gb, oom_peak_mem_gb) if value is not None)
    if tokens_per_sec is None and failure_status is None and peak_mem_gb is None:
        return None

    if failure_status == "oom" and tokens_per_sec is None:
        status = "observed_log_oom"
        correctness_status = "oom"
    elif failure_status is not None:
        status = "observed_log_partial_failure"
        correctness_status = failure_status
    else:
        status = "observed_log_summary"
        correctness_status = "not_promoted"

    phase_time_sec = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_time_sec")).items()
    }
    phase_time_rank_mean_sec = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_time_rank_mean_sec")).items()
    }
    phase_time_share = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_time_share")).items()
    }
    phase_memory_peak_gb = {
        key: round(value, 6) for key, value in _float_dict(observed_summary.get("phase_memory_peak_gb")).items()
    }
    notes = [
        "source=standalone_node_log",
        f"warmup_excluded={observed_summary.get('warmup_excluded', 0)}",
        f"parsed_steps={observed_summary.get('parsed_step_count', 0)}",
    ]
    resolved_config_match = RESOLVED_CONFIG_RE.search(full_log_text)
    if resolved_config_match:
        notes.append(f"resolved_config={resolved_config_match.group('path')}")
    git_match = re.search(r"(?m)^git\s+(?P<commit>[0-9a-f]{7,40})\s*$", full_log_text)
    if git_match:
        notes.append(f"commit={git_match.group('commit')}")
    if failure_status is not None:
        notes.append(f"log_failure_status={failure_status}")

    return BenchmarkBehaviorPoint(
        label=_standalone_log_label(log_path, benchmark_path),
        source=str(log_path),
        micro_batch_size=topology.micro_batch_size,
        global_batch_size=topology.global_batch_size,
        gradient_accumulation_steps=topology.gradient_accumulation_steps,
        tokens_per_sec=tokens_per_sec,
        step_time_sec=_round_or_none(observed_summary.get("step_time_s_mean"), 6),
        tokens_per_sec_std=_round_or_none(observed_summary.get("tokens_per_sec_std"), 3),
        tokens_per_sec_cv=_round_or_none(observed_summary.get("tokens_per_sec_cv"), 6),
        step_time_sec_std=_round_or_none(observed_summary.get("step_time_s_std"), 6),
        step_time_sec_cv=_round_or_none(observed_summary.get("step_time_s_cv"), 6),
        phase_time_sec=phase_time_sec,
        phase_time_rank_mean_sec=phase_time_rank_mean_sec,
        phase_time_share=phase_time_share,
        phase_memory_peak_gb=phase_memory_peak_gb,
        mfu_percent=_round_or_none((observed_summary.get("mfu_mean") or 0.0) * 100.0, 3)
        if observed_summary.get("mfu_mean") is not None
        else None,
        tflops_per_gpu=_round_or_none(observed_summary.get("tflops_per_gpu_mean"), 3),
        peak_mem_gb=peak_mem_gb,
        allocator_retries=None,
        measured_steps=observed_summary.get("measured_steps"),
        warmup_steps=observed_summary.get("warmup_excluded"),
        gpu_count=topology.world_size,
        model_ref=_first_non_none(
            _model_ref_from_text_or_path(json.dumps(raw_config, sort_keys=True), log_path),
            model_ref_from_config(raw_config),
        ),
        sample_packing_sequence_len=topology.sample_packing_sequence_len,
        data_parallel_replicate_size=topology.data_parallel_replicate_size,
        data_parallel_shard_size=topology.data_parallel_shard_size,
        tensor_parallel_size=topology.tensor_parallel_size,
        pipeline_parallel_size=topology.pipeline_parallel_size,
        ulysses_parallel_size=topology.ulysses_parallel_size,
        ringattn_parallel_size=topology.ringattn_parallel_size,
        expert_parallel_size=topology.expert_parallel_size,
        ep_fsdp_size=topology.ep_fsdp_size,
        deepep_async_combine=_config_bool(raw_config, "model", "deepep_async_combine", False),
        deepep_num_sms=_config_int(raw_config, "model", "deepep_num_sms"),
        deepep_buffer_size_gb=_config_float(raw_config, "model", "deepep_buffer_size_gb"),
        enable_compile=_config_bool(raw_config, "train", "enable_compile", False),
        gradient_checkpointing_method=_config_str(raw_config, "train", "gradient_checkpointing_method"),
        enable_activation_offload=_config_bool(raw_config, "train", "enable_activation_offload", False),
        activation_offload_prefetch_count=_config_int(raw_config, "train", "activation_offload_prefetch_count"),
        skip_param_upcast=_config_bool(raw_config, "train", "skip_param_upcast", False),
        fsdp_reduce_dtype=_config_fsdp_reduce_dtype(raw_config),
        ce_mode=_config_ce_mode(raw_config),
        moe_implementation=_config_str(raw_config, "model", "moe_implementation"),
        moe_checkpoint_method=_config_str(raw_config, "train", "moe_checkpoint_method"),
        muon_momentum=_config_float(raw_config, "train", "muon_momentum"),
        muon_update_dtype=_config_str(raw_config, "train", "muon_update_dtype"),
        attention_backend=_config_attention_backend(raw_config),
        balanced_routing=_config_balanced_routing(raw_config),
        status=status,
        correctness_status=correctness_status,
        notes=notes,
    )


def _standalone_log_points(
    benchmark_path: Path,
    *,
    used_sources: set[str],
) -> list[BenchmarkBehaviorPoint]:
    points: list[BenchmarkBehaviorPoint] = []
    for log_path in sorted(benchmark_path.rglob("node-0.log")):
        if str(log_path) in used_sources or str(log_path.resolve(strict=False)) in used_sources:
            continue
        point = _standalone_log_behavior_point(benchmark_path, log_path)
        if point is not None:
            points.append(point)
    return points


_FLASHQLA_PROFILE_PHASE_KEYS = {
    "train.forward_loss": "forward",
    "train.backward": "backward",
    "train.optimizer_step": "optimizer",
    "train.reduce_metrics": "reduce_metrics",
}


def _flashqla_profile_phase_time_sec(profile: Any) -> dict[str, float]:
    if not isinstance(profile, dict):
        return {}
    phases: dict[str, float] = {}
    for profile_key, phase_name in _FLASHQLA_PROFILE_PHASE_KEYS.items():
        row = profile.get(profile_key)
        if not isinstance(row, dict):
            continue
        value_ms = _first_non_none(row.get("max_ms_rank"), row.get("mean_ms_per_rank"))
        if value_ms is None:
            continue
        phases[phase_name] = round(float(value_ms) / 1000.0, 6)
    return phases


def _phase_time_share_from_step(phase_time_sec: dict[str, float], step_time_sec: float | None) -> dict[str, float]:
    if not phase_time_sec or step_time_sec is None or step_time_sec <= 0:
        return {}
    return {phase: round(value / step_time_sec, 6) for phase, value in phase_time_sec.items()}


def _flashqla_summary_config_path(summary_path: Path, backend: str, source: dict[str, Any]) -> Path:
    config_paths_by_backend = source.get("config_paths_by_backend", source.get("configs_by_backend", {}))
    if isinstance(config_paths_by_backend, dict) and backend in config_paths_by_backend:
        resolved = _resolve_manifest_path(summary_path.parent, config_paths_by_backend[backend])
        if resolved is not None:
            return resolved
    config_root = _resolve_manifest_path(summary_path.parent, source.get("config_root"))
    base = config_root if config_root is not None else summary_path.parent
    return base / f"out_{backend}" / "xorl_cli.yaml"


def _flashqla_summary_point(
    *,
    summary_path: Path,
    item: dict[str, Any],
    backend: str,
    backend_entry: dict[str, Any],
    source: dict[str, Any],
    source_notes: list[str],
) -> BenchmarkBehaviorPoint | None:
    steps = backend_entry.get("steps")
    if not isinstance(steps, dict):
        return None
    tokens_per_sec = steps.get("mean_tokens_per_sec")
    if tokens_per_sec is None:
        return None

    config_path = _flashqla_summary_config_path(summary_path, backend, source)
    if not config_path.is_file():
        return None
    raw_config = load_training_config(config_path)
    try:
        topology = resolve_topology(raw_config)
    except ValueError:
        return None

    run_name = str(item.get("run") or summary_path.parent.name)
    label_prefix = source.get("label_prefix")
    prefix = f"{label_prefix}/" if label_prefix else ""
    step_time_sec = _round_or_none(steps.get("mean_step_time_s"), 6)
    phase_time_sec = _flashqla_profile_phase_time_sec(backend_entry.get("profile"))
    measured_steps = steps.get("measured_steps")
    total_steps = steps.get("steps")
    warmup_steps = None
    if total_steps is not None and measured_steps is not None:
        warmup_steps = max(0, int(total_steps) - int(measured_steps))

    notes = [
        *source_notes,
        f"summary_run={run_name}",
        f"attention_backend={backend}",
    ]
    if total_steps is not None:
        notes.append(f"summary_steps={total_steps}")

    return BenchmarkBehaviorPoint(
        label=f"flashqla_summary:{prefix}{run_name}/{backend}",
        source=str(summary_path),
        micro_batch_size=topology.micro_batch_size,
        global_batch_size=topology.global_batch_size,
        gradient_accumulation_steps=topology.gradient_accumulation_steps,
        tokens_per_sec=_round_or_none(tokens_per_sec, 3),
        step_time_sec=step_time_sec,
        phase_time_sec=phase_time_sec,
        phase_time_share=_phase_time_share_from_step(phase_time_sec, step_time_sec),
        mfu_percent=None,
        tflops_per_gpu=_round_or_none(steps.get("mean_tflops_per_gpu"), 3),
        peak_mem_gb=None,
        allocator_retries=None,
        measured_steps=int(measured_steps) if measured_steps is not None else None,
        warmup_steps=warmup_steps,
        gpu_count=topology.world_size,
        model_ref=_first_non_none(
            _model_ref_from_text_or_path(json.dumps(raw_config, sort_keys=True), config_path),
            model_ref_from_config(raw_config),
        ),
        sample_packing_sequence_len=topology.sample_packing_sequence_len,
        data_parallel_replicate_size=topology.data_parallel_replicate_size,
        data_parallel_shard_size=topology.data_parallel_shard_size,
        tensor_parallel_size=topology.tensor_parallel_size,
        pipeline_parallel_size=topology.pipeline_parallel_size,
        ulysses_parallel_size=topology.ulysses_parallel_size,
        ringattn_parallel_size=topology.ringattn_parallel_size,
        expert_parallel_size=topology.expert_parallel_size,
        ep_fsdp_size=topology.ep_fsdp_size,
        deepep_async_combine=_config_bool(raw_config, "model", "deepep_async_combine", False),
        deepep_num_sms=_config_int(raw_config, "model", "deepep_num_sms"),
        deepep_buffer_size_gb=_config_float(raw_config, "model", "deepep_buffer_size_gb"),
        enable_compile=_config_bool(raw_config, "train", "enable_compile", False),
        gradient_checkpointing_method=_config_str(raw_config, "train", "gradient_checkpointing_method"),
        enable_activation_offload=_config_bool(raw_config, "train", "enable_activation_offload", False),
        activation_offload_prefetch_count=_config_int(raw_config, "train", "activation_offload_prefetch_count"),
        skip_param_upcast=_config_bool(raw_config, "train", "skip_param_upcast", False),
        fsdp_reduce_dtype=_config_fsdp_reduce_dtype(raw_config),
        ce_mode=_config_ce_mode(raw_config),
        moe_implementation=_config_str(raw_config, "model", "moe_implementation"),
        moe_checkpoint_method=_config_str(raw_config, "train", "moe_checkpoint_method"),
        muon_momentum=_config_float(raw_config, "train", "muon_momentum"),
        muon_update_dtype=_config_str(raw_config, "train", "muon_update_dtype"),
        attention_backend=backend,
        balanced_routing=_config_balanced_routing(raw_config),
        status="historical_flashqla_profile_summary",
        correctness_status="not_promoted",
        notes=notes,
    )


def _manifest_flashqla_summary_points(benchmark_path: Path) -> list[BenchmarkBehaviorPoint]:
    manifest = _read_benchmark_source_manifest(benchmark_path)
    if manifest is None:
        return []
    manifest_path, payload = manifest
    manifest_dir = manifest_path.parent
    points: list[BenchmarkBehaviorPoint] = []

    for source in _as_list(payload.get("flashqla_profile_summaries")):
        if isinstance(source, str):
            source = {"path": source}
        if not isinstance(source, dict):
            continue
        summary_paths = _expand_manifest_path_patterns(manifest_dir, source.get("path", source.get("summary")))
        selected_backends = {str(item) for item in _as_list(source.get("backends"))}
        source_notes = [f"benchmark_source_manifest={manifest_path.name}", "source=flashqla_profile_summaries"]
        label_prefix = source.get("label_prefix")
        if label_prefix:
            source_notes.append(f"source_label_prefix={label_prefix}")
        for summary_path in summary_paths:
            if not summary_path.is_file():
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                backends = item.get("backends", {})
                if not isinstance(backends, dict):
                    continue
                for backend, backend_entry in sorted(backends.items()):
                    if selected_backends and backend not in selected_backends:
                        continue
                    if not isinstance(backend_entry, dict):
                        continue
                    point = _flashqla_summary_point(
                        summary_path=summary_path,
                        item=item,
                        backend=str(backend),
                        backend_entry=backend_entry,
                        source=source,
                        source_notes=source_notes,
                    )
                    if point is not None:
                        points.append(point)
    return points


def _manifest_resolved_run_points(benchmark_path: Path) -> list[BenchmarkBehaviorPoint]:
    manifest = _read_benchmark_source_manifest(benchmark_path)
    if manifest is None:
        return []
    manifest_path, payload = manifest
    manifest_dir = manifest_path.parent
    points: list[BenchmarkBehaviorPoint] = []

    for source in _as_list(payload.get("resolved_run_roots")):
        if isinstance(source, str):
            source = {"path": source}
        if not isinstance(source, dict):
            continue
        root = _resolve_manifest_path(manifest_dir, source.get("path"))
        if root is None or not root.is_dir():
            continue
        label_prefix = source.get("label_prefix")
        label_prefix = str(label_prefix) if label_prefix is not None else None
        source_notes = [f"benchmark_source_manifest={manifest_path.name}"]
        if label_prefix:
            source_notes.append(f"source_label_prefix={label_prefix}")
        for config_path in sorted(root.rglob("xorl_cli.yaml")):
            if not config_path.is_file():
                continue
            point = _resolved_run_behavior_point(
                benchmark_path,
                config_path,
                label_root=root,
                label_prefix=label_prefix,
                log_paths=_log_paths_for_manifest_run(manifest_dir, root, config_path, source),
                warmup_steps=_manifest_run_int(
                    root,
                    config_path,
                    source,
                    field_name="warmup_steps",
                    by_run_field_name="warmup_steps_by_run",
                ),
                metrics_only_reason=_manifest_run_metrics_only_reason(root, config_path, source),
                notes=source_notes,
            )
            if point is not None:
                points.append(point)

    for source in _as_list(payload.get("resolved_runs")):
        if not isinstance(source, dict):
            continue
        config_path = _resolve_manifest_path(manifest_dir, source.get("config"))
        if config_path is None or not config_path.is_file():
            continue
        label = source.get("label")
        if isinstance(label, str) and not label.startswith("resolved_run:"):
            label = f"resolved_run:{label}"
        elif label is not None:
            label = str(label)
        log_root = _resolve_manifest_path(manifest_dir, source.get("log_root"))
        log_base = log_root if log_root is not None else manifest_dir
        log_paths = _expand_manifest_path_patterns(log_base, source.get("logs"))
        source_notes = [f"benchmark_source_manifest={manifest_path.name}", "source=resolved_runs"]
        point = _resolved_run_behavior_point(
            benchmark_path,
            config_path,
            label=label,
            log_paths=log_paths,
            warmup_steps=int(source["warmup_steps"]) if source.get("warmup_steps") is not None else None,
            metrics_only_reason=(
                None
                if source.get("metrics_only") in (None, False)
                else "metrics_only"
                if source.get("metrics_only") is True
                else str(source.get("metrics_only"))
            ),
            notes=source_notes,
        )
        if point is not None:
            points.append(point)
    return points


def _behavior_override_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_points = payload.get("points", payload.get("overrides", {}))
    if isinstance(raw_points, dict):
        return {str(label): dict(value) for label, value in raw_points.items() if isinstance(value, dict)}
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(raw_points, list):
        for item in raw_points:
            if not isinstance(item, dict) or not item.get("label"):
                continue
            label = str(item["label"])
            entry = dict(item)
            entry.pop("label", None)
            entries[label] = entry
    return entries


def _coerce_behavior_override_field(field_name: str, value: Any) -> Any:
    if field_name in {"phase_time_sec", "phase_time_rank_mean_sec", "phase_time_share", "phase_memory_peak_gb"}:
        return _float_dict(value)
    if field_name == "notes":
        return [str(item) for item in _as_list(value)]
    return value


def _apply_behavior_point_overrides(
    benchmark_path: Path,
    points: list[BenchmarkBehaviorPoint],
) -> list[BenchmarkBehaviorPoint]:
    manifest = _read_manifest_from_candidates(benchmark_path, BENCHMARK_BEHAVIOR_OVERRIDE_MANIFESTS)
    if manifest is None:
        return points
    manifest_path, payload = manifest
    overrides = _behavior_override_entries(payload)
    if not overrides:
        return points

    valid_fields = {field.name for field in fields(BenchmarkBehaviorPoint)}
    result: list[BenchmarkBehaviorPoint] = []
    for point in points:
        override = overrides.get(point.label)
        if override is None:
            result.append(point)
            continue
        replace_kwargs: dict[str, Any] = {}
        for field_name, value in override.items():
            if field_name in {"label", "notes_append"} or field_name not in valid_fields:
                continue
            replace_kwargs[field_name] = _coerce_behavior_override_field(field_name, value)
        notes = list(replace_kwargs.pop("notes", point.notes))
        notes.append(f"behavior_override_manifest={manifest_path.name}")
        for item in _as_list(override.get("notes_append")):
            notes.append(str(item))
        result.append(replace(point, notes=notes, **replace_kwargs))
    return result


def load_benchmark_behavior_points(benchmark_dir: str | Path) -> list[BenchmarkBehaviorPoint]:
    benchmark_path = resolve_calibration_pack(benchmark_dir)
    points: list[BenchmarkBehaviorPoint] = []
    topology_defaults: dict[str, int | float | bool | str] = {}

    for readme_path in (benchmark_path / "README.md", benchmark_path / "RESULTS.md"):
        if not readme_path.is_file():
            continue
        readme_text = readme_path.read_text(encoding="utf-8")
        model_ref = _model_ref_from_text_or_path(readme_text, readme_path)
        if model_ref is not None:
            topology_defaults["model_ref"] = model_ref
        topology_defaults.update(_readme_topology_defaults(readme_text))
        seq_len = _seq_len_from_readme(readme_text)
        if seq_len is not None:
            topology_defaults["sample_packing_sequence_len"] = seq_len
        readme_reference = _readme_point(readme_text, source=str(readme_path), model_ref=model_ref)
        if readme_reference is not None:
            points.append(readme_reference)
        adjacent_mbs10 = _readme_adjacent_mbs10_point(
            readme_text, source=str(readme_path), seq_len=seq_len, model_ref=model_ref
        )
        if adjacent_mbs10 is not None:
            points.append(adjacent_mbs10)
        points.extend(_q235_markdown_points(readme_text, source=str(readme_path), model_ref=model_ref))
        points.extend(_q35_markdown_points(readme_text, source=str(readme_path), model_ref=model_ref))

    for result_path in sorted((benchmark_path / "results").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_model_ref = _model_ref_from_text_or_path(json.dumps(result, sort_keys=True), result_path)
        result_defaults = dict(topology_defaults)
        if result_model_ref is not None:
            result_defaults["model_ref"] = result_model_ref
        for row in result.get("best_by_mfu", []):
            if isinstance(row, dict) and row.get("trial"):
                points.append(_best_by_mfu_point(result_path, result, row, topology_defaults=result_defaults))
        throughput = result.get("throughput")
        if isinstance(throughput, dict):
            points.append(
                _with_k3_status(
                    _result_throughput_point(result_path, result, topology_defaults=result_defaults), result
                )
            )

    points.extend(_manifest_resolved_run_points(benchmark_path))
    points.extend(_manifest_flashqla_summary_points(benchmark_path))
    points.extend(_resolved_run_points(benchmark_path))
    used_sources = {point.source for point in points}
    used_sources.update(str(Path(point.source).resolve(strict=False)) for point in points if point.source)
    points.extend(_standalone_log_points(benchmark_path, used_sources=used_sources))
    return _apply_behavior_point_overrides(benchmark_path, points)


def behavior_point_matches_topology(point: BenchmarkBehaviorPoint, topology: Topology) -> bool:
    if point.micro_batch_size != topology.micro_batch_size or point.global_batch_size != topology.global_batch_size:
        return False
    if (
        point.gradient_accumulation_steps is not None
        and point.gradient_accumulation_steps != topology.gradient_accumulation_steps
    ):
        return False
    if not _point_parallel_size_matches(point.tensor_parallel_size, topology.tensor_parallel_size):
        return False
    if not _point_parallel_size_matches(point.pipeline_parallel_size, topology.pipeline_parallel_size):
        return False
    if not _point_parallel_size_matches(point.ulysses_parallel_size, topology.ulysses_parallel_size):
        return False
    if not _point_parallel_size_matches(point.ringattn_parallel_size, topology.ringattn_parallel_size):
        return False
    if point.expert_parallel_size is None:
        if topology.expert_parallel_size != 1:
            return False
    elif point.expert_parallel_size != topology.expert_parallel_size:
        return False
    if point.ep_fsdp_size is not None and point.ep_fsdp_size != topology.ep_fsdp_size:
        return False
    if (
        point.sample_packing_sequence_len is not None
        and topology.sample_packing_sequence_len is not None
        and point.sample_packing_sequence_len != topology.sample_packing_sequence_len
    ):
        return False
    return True


def _section(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw_config.get(name, {})
    return value if isinstance(value, dict) else {}


def _config_bool(raw_config: dict[str, Any], section_name: str, key: str, default: bool | None = None) -> bool | None:
    section = _section(raw_config, section_name)
    value = section.get(key, default)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_int(raw_config: dict[str, Any], section_name: str, key: str) -> int | None:
    section = _section(raw_config, section_name)
    value = section.get(key)
    return int(value) if value is not None else None


def _config_float(raw_config: dict[str, Any], section_name: str, key: str) -> float | None:
    section = _section(raw_config, section_name)
    value = section.get(key)
    return float(value) if value is not None else None


def _config_str(raw_config: dict[str, Any], section_name: str, key: str) -> str | None:
    section = _section(raw_config, section_name)
    value = section.get(key)
    return str(value) if value is not None else None


def _boolish(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "balanced"}
    return bool(value)


def _config_balanced_routing(raw_config: dict[str, Any]) -> bool:
    for section_name in ("simulator", "_simulator", "train", "model", "data"):
        section = _section(raw_config, section_name)
        if "balanced_routing" in section:
            value = _boolish(section.get("balanced_routing"))
            if value is not None:
                return value
        for key in ("synthetic_routing", "synthetic_routing_mode", "moe_synthetic_routing"):
            value = section.get(key)
            if isinstance(value, str) and value.strip().lower() == "balanced":
                return True
    return False


def _config_attention_backend(raw_config: dict[str, Any]) -> str | None:
    for section_name in ("simulator", "_simulator"):
        section = _section(raw_config, section_name)
        value = section.get("attention_backend")
        if value is not None:
            return str(value)
    return None


def _config_fsdp_reduce_dtype(raw_config: dict[str, Any]) -> str:
    return _config_str(raw_config, "train", "fsdp_reduce_dtype") or "fp32"


def _config_ce_mode(raw_config: dict[str, Any]) -> str | None:
    return _config_str(raw_config, "train", "ce_mode") or "compiled"


def behavior_point_workload_mismatches(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any]) -> list[str]:
    config_activation_offload = _config_bool(raw_config, "train", "enable_activation_offload", False)
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("balanced_routing", point.balanced_routing, _config_balanced_routing(raw_config)),
        ("attention_backend", point.attention_backend, _config_attention_backend(raw_config)),
        (
            "deepep_async_combine",
            point.deepep_async_combine,
            _config_bool(raw_config, "model", "deepep_async_combine", False),
        ),
        ("deepep_num_sms", point.deepep_num_sms, _config_int(raw_config, "model", "deepep_num_sms")),
        (
            "deepep_buffer_size_gb",
            point.deepep_buffer_size_gb,
            _config_float(raw_config, "model", "deepep_buffer_size_gb"),
        ),
        ("enable_compile", point.enable_compile, _config_bool(raw_config, "train", "enable_compile", False)),
        (
            "gradient_checkpointing_method",
            point.gradient_checkpointing_method,
            _config_str(raw_config, "train", "gradient_checkpointing_method"),
        ),
        (
            "enable_activation_offload",
            point.enable_activation_offload,
            config_activation_offload,
        ),
        (
            "activation_offload_prefetch_count",
            point.activation_offload_prefetch_count,
            _config_int(raw_config, "train", "activation_offload_prefetch_count"),
        ),
        ("skip_param_upcast", point.skip_param_upcast, _config_bool(raw_config, "train", "skip_param_upcast", False)),
        ("fsdp_reduce_dtype", point.fsdp_reduce_dtype, _config_fsdp_reduce_dtype(raw_config)),
        ("ce_mode", point.ce_mode, _config_ce_mode(raw_config)),
        ("moe_implementation", point.moe_implementation, _config_str(raw_config, "model", "moe_implementation")),
        (
            "moe_checkpoint_method",
            point.moe_checkpoint_method,
            _config_str(raw_config, "train", "moe_checkpoint_method"),
        ),
        ("muon_momentum", point.muon_momentum, _config_float(raw_config, "train", "muon_momentum")),
        ("muon_update_dtype", point.muon_update_dtype, _config_str(raw_config, "train", "muon_update_dtype")),
    )
    mismatches: list[str] = []
    if behavior_point_model_mismatches(point, raw_config):
        mismatches.append("model_ref")
    for field_name, point_value, config_value in checks:
        if point_value is None:
            continue
        if (
            field_name == "activation_offload_prefetch_count"
            and point.enable_activation_offload is False
            and config_activation_offload is False
        ):
            continue
        if isinstance(point_value, float):
            if config_value is None or abs(float(point_value) - float(config_value)) > 1e-9:
                mismatches.append(field_name)
        elif point_value != config_value:
            mismatches.append(field_name)
    return mismatches


def behavior_point_matches_workload(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any]) -> bool:
    return not behavior_point_workload_mismatches(point, raw_config)


def _point_parallel_size_matches(point_value: int | None, topology_value: int) -> bool:
    if point_value is None:
        return topology_value == 1
    return point_value == topology_value


def _fsdp_reduce_dtype_selection_penalty(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any] | None) -> int:
    if raw_config is None or point.fsdp_reduce_dtype is None:
        return 0
    config_value = _config_str(raw_config, "train", "fsdp_reduce_dtype") or "fp32"
    return 0 if point.fsdp_reduce_dtype == config_value else 1


def _behavior_point_evidence_rank(point: BenchmarkBehaviorPoint) -> int:
    if point.label.startswith("resolved_run:") and point.status.startswith("observed_log"):
        return 0
    return 1


def _select_behavior_point_match(
    matches: list[BenchmarkBehaviorPoint],
    raw_config: dict[str, Any] | None,
) -> BenchmarkBehaviorPoint:
    indexed_matches = list(enumerate(matches))
    _, point = min(
        indexed_matches,
        key=lambda item: (
            _fsdp_reduce_dtype_selection_penalty(item[1], raw_config),
            _behavior_point_evidence_rank(item[1]),
            item[0],
        ),
    )
    return point


def predict_benchmark_behavior(
    points: list[BenchmarkBehaviorPoint],
    topology: Topology,
    shape: ShapeLedger,
    raw_config: dict[str, Any] | None = None,
) -> BenchmarkBehaviorPrediction:
    matches = [
        point
        for point in points
        if behavior_point_matches_topology(point, topology)
        and (raw_config is None or behavior_point_matches_workload(point, raw_config))
        and point.status != "observed_log_metrics_only"
    ]
    warnings: list[str] = []
    if not matches:
        known = ", ".join(
            f"{point.label}(mbs={point.micro_batch_size},gb={point.global_batch_size})" for point in points
        )
        return BenchmarkBehaviorPrediction(
            status="no_calibrated_match",
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
            warnings=[
                f"no empirical behavior point for mbs={topology.micro_batch_size}, gb={topology.global_batch_size}; known: {known}"
            ],
        )

    point = _select_behavior_point_match(matches, raw_config)
    tokens_per_sec_per_gpu = None
    if point.tokens_per_sec is not None and topology.world_size:
        tokens_per_sec_per_gpu = point.tokens_per_sec / topology.world_size
    step_time_sec = point.step_time_sec
    if step_time_sec is None and shape.global_tokens_per_train_step and point.tokens_per_sec:
        step_time_sec = shape.global_tokens_per_train_step / point.tokens_per_sec
        warnings.append(
            "step_time_sec derived as NOMINAL global tokens (gbs x seq) / measured tokens_per_sec; "
            "real-dataset packs may not fill bins (q35 65k realized ~0.81x nominal), which would "
            "overstate the derived step time by the same factor"
        )
    tflops_per_gpu = None
    if point.tflops_per_gpu is not None:
        tflops_per_gpu = point.tflops_per_gpu
    elif point.mfu_percent is not None:
        tflops_per_gpu = H100_BF16_PROMISED_TFLOPS_PER_GPU * point.mfu_percent / 100.0

    if point.status == "allocator_pressure_slowdown":
        warnings.append("matched behavior point is an allocator-pressure slowdown, not a promotable speed target")
    if point.correctness_status and point.correctness_status != "k3_pass":
        warnings.append(f"correctness status is {point.correctness_status}")

    prediction_status = "calibrated_failure" if point.correctness_status == "oom" else "calibrated"

    return BenchmarkBehaviorPrediction(
        status=prediction_status,
        matched_label=point.label,
        source=point.source,
        tokens_per_sec=point.tokens_per_sec,
        tokens_per_sec_per_gpu=tokens_per_sec_per_gpu,
        step_time_sec=step_time_sec,
        tokens_per_sec_std=point.tokens_per_sec_std,
        tokens_per_sec_cv=point.tokens_per_sec_cv,
        step_time_sec_std=point.step_time_sec_std,
        step_time_sec_cv=point.step_time_sec_cv,
        phase_time_sec=point.phase_time_sec,
        phase_time_share=point.phase_time_share,
        phase_memory_peak_gb=point.phase_memory_peak_gb,
        measured_steps=point.measured_steps,
        warmup_steps=point.warmup_steps,
        model_ref=point.model_ref,
        balanced_routing=point.balanced_routing,
        mfu_percent=point.mfu_percent,
        tflops_per_gpu=tflops_per_gpu,
        promised_tflops_per_gpu=H100_BF16_PROMISED_TFLOPS_PER_GPU,
        peak_mem_gb=point.peak_mem_gb,
        allocator_retries=point.allocator_retries,
        derived_global_tokens_per_step=shape.global_tokens_per_train_step,
        correctness_status=point.correctness_status,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", help="Path or built-in calibration-pack name")
    args = parser.parse_args()
    points = load_benchmark_behavior_points(args.benchmark_dir)
    print(json.dumps(to_jsonable({"points": points}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
