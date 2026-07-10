"""Empirical benchmark behavior calibration for checked-in benchmark recipes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


try:
    from .collect_calibration import parse_log_path, summarize_observed_run
    from .config_fingerprint import load_training_config, resolve_topology
    from .schemas import BenchmarkBehaviorPoint, BenchmarkBehaviorPrediction, ShapeLedger, Topology, to_jsonable
except ImportError:  # pragma: no cover - exercised by direct script execution
    from collect_calibration import parse_log_path, summarize_observed_run
    from config_fingerprint import load_training_config, resolve_topology
    from schemas import BenchmarkBehaviorPoint, BenchmarkBehaviorPrediction, ShapeLedger, Topology, to_jsonable


H100_BF16_PROMISED_TFLOPS_PER_GPU = 989.0


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


def _readme_point(readme_text: str, *, source: str) -> BenchmarkBehaviorPoint | None:
    tps_match = re.search(r"\|\s*tokens/sec\s*\|\s*(?P<value>~?[0-9.]+[KkMm]?)\s*\|", readme_text)
    step_match = re.search(r"\|\s*step time\s*\|\s*(?P<value>~?[0-9.]+)s\s*\|", readme_text)
    mfu_match = re.search(r"\|\s*MFU\s*\|\s*(?P<value>~?[0-9.]+)%", readme_text)
    memory_match = re.search(r"\|\s*allocated memory\s*\|\s*(?P<value>~?[0-9.]+)GB\s*\|", readme_text)
    retries_match = re.search(r"\|\s*allocator retries\s*\|\s*(?P<value>\d+)\s*\|", readme_text)
    mbs_match = re.search(r"micro_batch_size:\s*(?P<value>\d+)", readme_text)
    global_batch_match = re.search(r"global_batch_size:\s*(?P<value>\d+)", readme_text)
    if not tps_match:
        return None
    return BenchmarkBehaviorPoint(
        label="readme_reference_mbs8",
        source=source,
        micro_batch_size=int(mbs_match.group("value")) if mbs_match else None,
        global_batch_size=int(global_batch_match.group("value")) if global_batch_match else None,
        tokens_per_sec=human_number(tps_match.group("value")),
        step_time_sec=float(step_match.group("value").lstrip("~")) if step_match else None,
        mfu_percent=float(mfu_match.group("value").lstrip("~")) if mfu_match else None,
        tflops_per_gpu=None,
        peak_mem_gb=float(memory_match.group("value").lstrip("~")) if memory_match else None,
        allocator_retries=int(retries_match.group("value")) if retries_match else None,
        gpu_count=_gpu_count_from_text(readme_text),
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
    readme_text: str, *, source: str, seq_len: int | None
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
        mfu_percent=None,
        tflops_per_gpu=None,
        peak_mem_gb=None,
        allocator_retries=None,
        gpu_count=_gpu_count_from_text(readme_text),
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
        tokens_per_sec=throughput.get("tokens_per_sec"),
        step_time_sec=throughput.get("step_time_sec"),
        mfu_percent=throughput.get("mfu_percent"),
        tflops_per_gpu=throughput.get("mean_tflops_per_gpu"),
        peak_mem_gb=throughput.get("gpu_alloc_gb"),
        allocator_retries=None,
        measured_steps=throughput.get("measured_steps"),
        warmup_steps=throughput.get("warmup_steps"),
        gpu_count=throughput.get("gpus"),
        sample_packing_sequence_len=throughput.get("sample_packing_sequence_len"),
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
    return BenchmarkBehaviorPoint(
        label=point.label,
        source=point.source,
        micro_batch_size=point.micro_batch_size,
        global_batch_size=point.global_batch_size,
        tokens_per_sec=point.tokens_per_sec,
        step_time_sec=point.step_time_sec,
        mfu_percent=point.mfu_percent,
        tflops_per_gpu=point.tflops_per_gpu,
        peak_mem_gb=point.peak_mem_gb,
        allocator_retries=point.allocator_retries,
        measured_steps=point.measured_steps,
        warmup_steps=point.warmup_steps,
        gpu_count=point.gpu_count,
        sample_packing_sequence_len=point.sample_packing_sequence_len,
        tensor_parallel_size=point.tensor_parallel_size,
        pipeline_parallel_size=point.pipeline_parallel_size,
        ulysses_parallel_size=point.ulysses_parallel_size,
        ringattn_parallel_size=point.ringattn_parallel_size,
        expert_parallel_size=point.expert_parallel_size,
        ep_fsdp_size=point.ep_fsdp_size,
        deepep_async_combine=point.deepep_async_combine,
        deepep_num_sms=point.deepep_num_sms,
        deepep_buffer_size_gb=point.deepep_buffer_size_gb,
        enable_compile=point.enable_compile,
        gradient_checkpointing_method=point.gradient_checkpointing_method,
        enable_activation_offload=point.enable_activation_offload,
        activation_offload_prefetch_count=point.activation_offload_prefetch_count,
        status=point.status,
        correctness_status=f"k3_{k3_gate.get('status')}",
        notes=notes,
    )


def _seq_len_from_readme(readme_text: str) -> int | None:
    match = re.search(r"sample_packing_sequence_len:\s*(?P<seq>\d+)", readme_text)
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


def _readme_topology_defaults(readme_text: str) -> dict[str, int | float | bool | str]:
    defaults: dict[str, int | float | bool | str] = {}
    field_patterns = {
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


def _q235_markdown_points(readme_text: str, *, source: str) -> list[BenchmarkBehaviorPoint]:
    if "Qwen3-235B" not in readme_text or "tok/s tot" not in readme_text:
        return []

    points: list[BenchmarkBehaviorPoint] = []
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
        points.append(
            BenchmarkBehaviorPoint(
                label=f"q235_markdown:{run}",
                source=source,
                micro_batch_size=int(_first_markdown_number(values.get("mbs", "")) or 1),
                global_batch_size=global_batch_size,
                tokens_per_sec=tokens_per_sec,
                step_time_sec=step_time_sec,
                mfu_percent=mfu_percent,
                peak_mem_gb=peak_mem_gb,
                allocator_retries=None,
                gpu_count=current_gpu_count,
                sample_packing_sequence_len=int(pack),
                tensor_parallel_size=current_tensor_parallel_size,
                pipeline_parallel_size=current_pipeline_parallel_size,
                ulysses_parallel_size=current_ulysses_parallel_size,
                ringattn_parallel_size=current_ringattn_parallel_size,
                expert_parallel_size=current_ep_size,
                ep_fsdp_size=current_ep_fsdp_size,
                status="historical_q235_markdown_oom" if is_failure else "historical_q235_markdown",
                correctness_status="oom" if is_failure else "not_promoted",
                notes=[status_text] if status_text else [],
            )
        )
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
        mfu_percent=row.get("mfu_percent"),
        tflops_per_gpu=row.get("mean_tflops_per_gpu"),
        peak_mem_gb=None,
        allocator_retries=None,
        measured_steps=row.get("measured_steps"),
        warmup_steps=row.get("warmup_steps"),
        gpu_count=row.get("gpus") or _gpu_count_from_text(str(result.get("workload", ""))),
        sample_packing_sequence_len=row.get("sample_packing_sequence_len"),
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


def _resolved_run_log_path(benchmark_path: Path, run_dir: Path, startup_metrics: dict[str, Any]) -> Path | None:
    candidates = [
        run_dir / "node-0.log",
        _startup_master_log_path(benchmark_path, startup_metrics),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _log_failure_status(text: str) -> str | None:
    lowered = text.lower()
    if "outofmemoryerror" in lowered or "cuda out of memory" in lowered:
        return "oom"
    if "childfailederror" in lowered or "traceback" in lowered:
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


def _resolved_run_behavior_point(benchmark_path: Path, config_path: Path) -> BenchmarkBehaviorPoint | None:
    run_dir = config_path.parent
    raw_config = load_training_config(config_path)
    try:
        topology = resolve_topology(raw_config)
    except ValueError:
        return None

    startup_metrics = _load_startup_metrics(run_dir)
    log_path = _resolved_run_log_path(benchmark_path, run_dir, startup_metrics)
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path is not None else ""
    failure_status = _log_failure_status(log_text)
    observed_summary: dict[str, Any] = {}
    if log_path is not None:
        observed = parse_log_path(log_path)
        warmup_steps = 2 if len(observed.steps) > 2 else 0
        observed_summary = summarize_observed_run(observed, warmup_steps=warmup_steps, world_size=topology.world_size)

    tokens_per_sec = _round_or_none(observed_summary.get("tokens_per_sec_mean"), 3)
    peak_mem_gb = _round_or_none(observed_summary.get("peak_mem_gb_max"), 3)
    if peak_mem_gb is None and failure_status == "oom":
        peak_mem_gb = _round_or_none(_oom_peak_mem_gb(log_text), 3)
    if tokens_per_sec is None and failure_status is None:
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

    metrics = startup_metrics.get("metrics", {})
    notes = [
        f"warmup_excluded={observed_summary.get('warmup_excluded', 0)}",
        f"parsed_steps={observed_summary.get('parsed_step_count', 0)}",
    ]
    if startup_metrics.get("repo_commit"):
        notes.append(f"commit={startup_metrics['repo_commit']}")
    if isinstance(metrics.get("startup/master_addr"), str):
        notes.append(f"master_addr={metrics['startup/master_addr']}")
    if failure_status is not None:
        notes.append(f"log_failure_status={failure_status}")

    return BenchmarkBehaviorPoint(
        label=f"resolved_run:{config_path.parent.relative_to(benchmark_path)}",
        source=str(log_path or config_path),
        micro_batch_size=topology.micro_batch_size,
        global_batch_size=topology.global_batch_size,
        tokens_per_sec=tokens_per_sec,
        step_time_sec=_round_or_none(observed_summary.get("step_time_s_mean"), 6),
        mfu_percent=_round_or_none((observed_summary.get("mfu_mean") or 0.0) * 100.0, 3)
        if observed_summary.get("mfu_mean") is not None
        else None,
        tflops_per_gpu=_round_or_none(observed_summary.get("tflops_per_gpu_mean"), 3),
        peak_mem_gb=peak_mem_gb,
        allocator_retries=None,
        measured_steps=observed_summary.get("measured_steps"),
        warmup_steps=observed_summary.get("warmup_excluded"),
        gpu_count=topology.world_size,
        sample_packing_sequence_len=topology.sample_packing_sequence_len,
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
        status=status,
        correctness_status=correctness_status,
        notes=notes,
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


def load_benchmark_behavior_points(benchmark_dir: str | Path) -> list[BenchmarkBehaviorPoint]:
    benchmark_path = Path(benchmark_dir)
    points: list[BenchmarkBehaviorPoint] = []
    topology_defaults: dict[str, int | float | bool | str] = {}

    for readme_path in (benchmark_path / "README.md", benchmark_path / "RESULTS.md"):
        if not readme_path.is_file():
            continue
        readme_text = readme_path.read_text(encoding="utf-8")
        topology_defaults.update(_readme_topology_defaults(readme_text))
        seq_len = _seq_len_from_readme(readme_text)
        readme_reference = _readme_point(readme_text, source=str(readme_path))
        if readme_reference is not None:
            points.append(readme_reference)
        adjacent_mbs10 = _readme_adjacent_mbs10_point(readme_text, source=str(readme_path), seq_len=seq_len)
        if adjacent_mbs10 is not None:
            points.append(adjacent_mbs10)
        points.extend(_q235_markdown_points(readme_text, source=str(readme_path)))

    for result_path in sorted((benchmark_path / "results").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for row in result.get("best_by_mfu", []):
            if isinstance(row, dict) and row.get("trial"):
                points.append(_best_by_mfu_point(result_path, result, row, topology_defaults=topology_defaults))
        throughput = result.get("throughput")
        if isinstance(throughput, dict):
            points.append(
                _with_k3_status(
                    _result_throughput_point(result_path, result, topology_defaults=topology_defaults), result
                )
            )

    points.extend(_resolved_run_points(benchmark_path))
    return points


def behavior_point_matches_topology(point: BenchmarkBehaviorPoint, topology: Topology) -> bool:
    if point.micro_batch_size != topology.micro_batch_size or point.global_batch_size != topology.global_batch_size:
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


def behavior_point_workload_mismatches(point: BenchmarkBehaviorPoint, raw_config: dict[str, Any]) -> list[str]:
    checks: tuple[tuple[str, Any, Any], ...] = (
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
            _config_bool(raw_config, "train", "enable_activation_offload", False),
        ),
        (
            "activation_offload_prefetch_count",
            point.activation_offload_prefetch_count,
            _config_int(raw_config, "train", "activation_offload_prefetch_count"),
        ),
    )
    mismatches: list[str] = []
    for field_name, point_value, config_value in checks:
        if point_value is None:
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
            warnings=[
                f"no empirical behavior point for mbs={topology.micro_batch_size}, gb={topology.global_batch_size}; known: {known}"
            ],
        )

    point = matches[0]
    tokens_per_sec_per_gpu = None
    if point.tokens_per_sec is not None and topology.world_size:
        tokens_per_sec_per_gpu = point.tokens_per_sec / topology.world_size
    step_time_sec = point.step_time_sec
    if step_time_sec is None and shape.global_tokens_per_train_step and point.tokens_per_sec:
        step_time_sec = shape.global_tokens_per_train_step / point.tokens_per_sec
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
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args()
    points = load_benchmark_behavior_points(args.benchmark_dir)
    print(json.dumps(to_jsonable({"points": points}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
