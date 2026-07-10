"""Validate simulator output against checked-in throughput benchmark recipes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any


try:
    from .benchmark_behavior import (
        H100_BF16_PROMISED_TFLOPS_PER_GPU,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from .config_fingerprint import load_training_config
    from .predict import build_report
    from .schemas import to_jsonable
    from .shape_engine import build_shape_ledger
except ImportError:  # pragma: no cover - exercised by direct script execution
    from benchmark_behavior import (
        H100_BF16_PROMISED_TFLOPS_PER_GPU,
        load_benchmark_behavior_points,
        predict_benchmark_behavior,
    )
    from config_fingerprint import load_training_config
    from predict import build_report
    from schemas import to_jsonable
    from shape_engine import build_shape_ledger


def _human_number(value: str) -> float:
    cleaned = value.strip().replace(",", "").lstrip("~")
    multiplier = 1.0
    if cleaned.endswith(("K", "k")):
        cleaned = cleaned[:-1]
        multiplier = 1_000.0
    elif cleaned.endswith(("M", "m")):
        cleaned = cleaned[:-1]
        multiplier = 1_000_000.0
    return float(cleaned) * multiplier


def _parse_readme_metrics(readme_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if match := re.search(r"Hardware:\s*(?P<nodes>\d+)\s*nodes?\s*x\s*(?P<gpus>\d+)\s*H100", readme_text):
        metrics["nodes"] = int(match.group("nodes"))
        metrics["gpus_per_node"] = int(match.group("gpus"))
        metrics["world_size"] = metrics["nodes"] * metrics["gpus_per_node"]
    if match := re.search(r"sample_packing_sequence_len:\s*(?P<seq>\d+)", readme_text):
        metrics["sample_packing_sequence_len"] = int(match.group("seq"))
    if match := re.search(r"\|\s*tokens/sec\s*\|\s*(?P<value>~?[0-9.]+[KkMm]?)\s*\|", readme_text):
        metrics["tokens_per_sec"] = _human_number(match.group("value"))
    if match := re.search(r"\|\s*step time\s*\|\s*(?P<value>~?[0-9.]+)s\s*\|", readme_text):
        metrics["step_time_sec"] = float(match.group("value").lstrip("~"))
    if match := re.search(r"\|\s*MFU\s*\|\s*(?P<value>~?[0-9.]+)%", readme_text):
        metrics["mfu_percent"] = float(match.group("value").lstrip("~"))
    if match := re.search(r"\|\s*allocated memory\s*\|\s*(?P<value>~?[0-9.]+)GB\s*\|", readme_text):
        metrics["allocated_memory_gb"] = float(match.group("value").lstrip("~"))
    if match := re.search(r"`mbs=10`[^~]+~(?P<value>[0-9.]+)K tok/s", readme_text):
        metrics["mbs10_tokens_per_sec"] = _human_number(match.group("value") + "K")
    return metrics


def _check(name: str, expected: Any, actual: Any, *, tolerance: float | None = None) -> dict[str, Any]:
    if tolerance is None:
        passed = expected == actual
    else:
        passed = expected is not None and actual is not None and abs(float(expected) - float(actual)) <= tolerance
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
    }


def _benchmark_config_path(benchmark_dir: Path) -> Path:
    configs = sorted((benchmark_dir / "configs").glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"no benchmark configs found under {benchmark_dir / 'configs'}")
    if len(configs) > 1:
        raise ValueError(f"expected one config under {benchmark_dir / 'configs'}, found {len(configs)}")
    return configs[0]


def _render_script_text(benchmark_dir: Path) -> str:
    script_path = benchmark_dir / "scripts" / "render_k8s_manifest.sh"
    return script_path.read_text(encoding="utf-8") if script_path.is_file() else ""


def _render_manifest_text(benchmark_dir: Path) -> str:
    script_path = benchmark_dir / "scripts" / "render_k8s_manifest.sh"
    if not script_path.is_file():
        return ""
    with tempfile.TemporaryDirectory(prefix="xorl-training-sim-") as tmpdir:
        output_path = Path(tmpdir) / "manifest.yaml"
        env = os.environ.copy()
        env["OUTPUT"] = str(output_path)
        result = subprocess.run(
            [str(script_path)],
            check=False,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return result.stdout + result.stderr
        return output_path.read_text(encoding="utf-8")


def _validate_config_behavior(
    benchmark_dir: Path,
    readme_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config_path = _benchmark_config_path(benchmark_dir)
    raw_config = load_training_config(config_path)
    world_size = readme_metrics.get("world_size")
    local_world_size = readme_metrics.get("gpus_per_node")
    report = build_report(
        config_path,
        world_size=world_size,
        local_world_size=local_world_size,
        balanced_routing=True,
        num_experts=None,
        top_k=None,
        benchmark_dir=benchmark_dir,
    )
    topology = report.fingerprint.topology
    shape = report.shape
    model = raw_config.get("model", {})
    data = raw_config.get("data", {})
    train = raw_config.get("train", {})

    checks = [
        _check("readme_reference_tokens_per_sec", 261000.0, readme_metrics.get("tokens_per_sec")),
        _check("readme_reference_step_time_sec", 8.04, readme_metrics.get("step_time_sec")),
        _check("readme_reference_mfu_percent", 16.2, readme_metrics.get("mfu_percent")),
        _check("readme_reference_allocated_memory_gb", 56.4, readme_metrics.get("allocated_memory_gb")),
        _check("readme_mbs10_tokens_per_sec", 133700.0, readme_metrics.get("mbs10_tokens_per_sec")),
        _check(
            "readme_mbs10_allocator_pressure_slowdown",
            True,
            readme_metrics.get("mbs10_tokens_per_sec", 0) < readme_metrics.get("tokens_per_sec", 0) * 0.6,
        ),
        _check("world_size", world_size, topology.world_size),
        _check("local_world_size", local_world_size, topology.local_world_size),
        _check("pipeline_parallel_size", 1, topology.pipeline_parallel_size),
        _check("tensor_parallel_size", 1, topology.tensor_parallel_size),
        _check("ringattn_parallel_size", 1, topology.ringattn_parallel_size),
        _check("ulysses_parallel_size", 1, topology.ulysses_parallel_size),
        _check(
            "sample_packing_sequence_len",
            readme_metrics.get("sample_packing_sequence_len"),
            topology.sample_packing_sequence_len,
        ),
        _check("micro_batch_size", 8, topology.micro_batch_size),
        _check("global_batch_size", 256, topology.global_batch_size),
        _check("data_parallel_replicate_size", 1, topology.data_parallel_replicate_size),
        _check("expert_parallel_size", 8, topology.expert_parallel_size),
        _check("ep_fsdp_size", 4, topology.ep_fsdp_size),
        _check("data_parallel_shard_size", 32, topology.data_parallel_shard_size),
        _check("num_experts", 256, topology.num_experts),
        _check("top_k", 8, topology.top_k),
        _check("dataset_path", "dummy", data.get("datasets", [{}])[0].get("path")),
        _check("dataset_type", "tokenized", data.get("datasets", [{}])[0].get("type")),
        _check("sample_packing_method", "sequential", data.get("sample_packing_method")),
        _check("moe_implementation", "quack", model.get("moe_implementation")),
        _check("ep_dispatch", "deepep", model.get("ep_dispatch")),
        _check("train_router", False, model.get("train_router")),
        _check("deepep_buffer_size_gb", 2.0, model.get("deepep_buffer_size_gb")),
        _check("deepep_num_sms", 72, model.get("deepep_num_sms")),
        _check("deepep_async_combine", True, model.get("deepep_async_combine")),
        _check("data_parallel_mode", "fsdp2", train.get("data_parallel_mode")),
        _check("gradient_checkpointing_method", "recompute_full_layer", train.get("gradient_checkpointing_method")),
        _check("optimizer", "adamw", train.get("optimizer")),
        _check("enable_mixed_precision", True, train.get("enable_mixed_precision")),
        _check("enable_full_shard", True, train.get("enable_full_shard")),
        _check("init_device", "meta", train.get("init_device")),
        _check("load_weights_mode", "grouped", train.get("load_weights_mode")),
        _check("enable_compile", True, train.get("enable_compile")),
        _check("empty_cache_steps", 10, train.get("empty_cache_steps")),
        _check("gc_steps", 10, train.get("gc_steps")),
        _check("save_steps", 0, train.get("save_steps")),
        _check("save_epochs", 0, train.get("save_epochs")),
        _check("log_format", "structured", train.get("log_format")),
        _check("global_tokens_per_train_step", 2_097_408, shape.global_tokens_per_train_step),
    ]
    behavior_points = load_benchmark_behavior_points(benchmark_dir)
    behavior_prediction = predict_benchmark_behavior(behavior_points, topology, shape, raw_config)
    behavior_labels = sorted(point.label for point in behavior_points)
    behavior_predictions = _predict_all_behavior_points(behavior_points, topology)
    checks.extend(
        [
            _check(
                "benchmark_behavior_points",
                [
                    "qwen36_static_k3_summary_20260519:q36-main-af98064-deepepenv-05190533",
                    "readme_adjacent_mbs10_allocator_pressure",
                    "readme_reference_mbs8",
                ],
                behavior_labels,
            ),
            _check("benchmark_behavior_prediction_status", "calibrated", behavior_prediction.status),
            _check("benchmark_behavior_prediction_label", "readme_reference_mbs8", behavior_prediction.matched_label),
            _check(
                "benchmark_behavior_tokens_per_sec",
                readme_metrics.get("tokens_per_sec"),
                behavior_prediction.tokens_per_sec,
            ),
            _check(
                "benchmark_behavior_step_time_sec",
                readme_metrics.get("step_time_sec"),
                behavior_prediction.step_time_sec,
            ),
            _check(
                "benchmark_behavior_peak_mem_gb",
                readme_metrics.get("allocated_memory_gb"),
                behavior_prediction.peak_mem_gb,
            ),
            _check("benchmark_behavior_allocator_retries", 0, behavior_prediction.allocator_retries),
            _check(
                "benchmark_behavior_promised_tflops_per_gpu",
                H100_BF16_PROMISED_TFLOPS_PER_GPU,
                behavior_prediction.promised_tflops_per_gpu,
            ),
            _check("benchmark_behavior_tflops_per_gpu", 160.218, behavior_prediction.tflops_per_gpu, tolerance=0.001),
        ]
    )
    for label, prediction in behavior_predictions.items():
        point = next(point for point in behavior_points if point.label == label)
        variant_shape = prediction["shape"]
        variant_behavior = prediction["behavior"]
        checks.extend(
            [
                _check(f"behavior_matrix:{label}:prediction_status", "calibrated", variant_behavior["status"]),
                _check(f"behavior_matrix:{label}:prediction_label", label, variant_behavior["matched_label"]),
                _check(
                    f"behavior_matrix:{label}:tokens_per_sec", point.tokens_per_sec, variant_behavior["tokens_per_sec"]
                ),
                _check(
                    f"behavior_matrix:{label}:global_tokens_per_step",
                    (point.global_batch_size or 0) * (topology.sample_packing_sequence_len or 0),
                    variant_shape["global_tokens_per_train_step"],
                ),
            ]
        )
        if point.step_time_sec is not None:
            checks.append(
                _check(
                    f"behavior_matrix:{label}:step_time_sec",
                    point.step_time_sec,
                    variant_behavior["step_time_sec"],
                    tolerance=0.01,
                )
            )
        if point.mfu_percent is not None:
            checks.extend(
                [
                    _check(
                        f"behavior_matrix:{label}:mfu_percent",
                        point.mfu_percent,
                        variant_behavior["mfu_percent"],
                    ),
                    _check(
                        f"behavior_matrix:{label}:promised_tflops_per_gpu",
                        H100_BF16_PROMISED_TFLOPS_PER_GPU,
                        variant_behavior["promised_tflops_per_gpu"],
                    ),
                    _check(
                        f"behavior_matrix:{label}:tflops_per_gpu",
                        H100_BF16_PROMISED_TFLOPS_PER_GPU * point.mfu_percent / 100.0,
                        variant_behavior["tflops_per_gpu"],
                        tolerance=0.001,
                    ),
                ]
            )
        if point.tokens_per_sec and variant_shape["global_tokens_per_train_step"]:
            checks.append(
                _check(
                    f"behavior_matrix:{label}:tokens_imply_step_time",
                    variant_behavior["step_time_sec"],
                    variant_shape["global_tokens_per_train_step"] / point.tokens_per_sec,
                    tolerance=0.05,
                )
            )
    if readme_metrics.get("tokens_per_sec") and shape.global_tokens_per_train_step:
        derived_step_time = shape.global_tokens_per_train_step / readme_metrics["tokens_per_sec"]
        checks.append(
            _check(
                "readme_tokens_per_sec_implies_step_time",
                readme_metrics.get("step_time_sec"),
                derived_step_time,
                tolerance=0.05,
            )
        )
    if report.shape.balanced_routing is not None:
        counts = report.shape.balanced_routing.counts_by_expert
        checks.extend(
            [
                _check("balanced_routing_imbalance_slots", 1, report.shape.balanced_routing.imbalance_slots),
                _check("balanced_routing_count_sum", report.shape.balanced_routing.total_slots, sum(counts)),
                _check("experts_per_ep_rank", 32, topology.num_experts // topology.expert_parallel_size),
            ]
        )
    script_text = _render_script_text(benchmark_dir)
    manifest_text = _render_manifest_text(benchmark_dir)
    checks.append(
        _check(
            "render_script_sets_balanced_synthetic_routing_env",
            True,
            "XORL_MOE_SYNTHETIC_ROUTING" in script_text and "balanced" in script_text,
        )
    )
    checks.extend(
        [
            _check("rendered_manifest_sets_team_turbo", True, "team: turbo" in manifest_text),
            _check(
                "rendered_manifest_sets_balanced_synthetic_routing_env",
                True,
                "name: XORL_MOE_SYNTHETIC_ROUTING" in manifest_text and 'value: "balanced"' in manifest_text,
            ),
            _check(
                "rendered_manifest_sets_nccl_socket_ifname",
                True,
                "name: NCCL_SOCKET_IFNAME" in manifest_text and 'value: "bond0"' in manifest_text,
            ),
            _check("rendered_manifest_sets_runtime_class", True, "runtimeClassName: nvidia" in manifest_text),
        ]
    )
    return to_jsonable(report), checks, behavior_predictions


def _predict_all_behavior_points(behavior_points, topology) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for point in behavior_points:
        if not point.micro_batch_size or not point.global_batch_size:
            continue
        denom = point.micro_batch_size * topology.data_parallel_size
        if point.global_batch_size % denom != 0:
            continue
        variant_topology = replace(
            topology,
            micro_batch_size=point.micro_batch_size,
            gradient_accumulation_steps=point.global_batch_size // denom,
            global_batch_size=point.global_batch_size,
        )
        variant_shape = build_shape_ledger(variant_topology, balanced_routing=True)
        predictions[point.label] = {
            "shape": to_jsonable(variant_shape),
            "behavior": to_jsonable(predict_benchmark_behavior(behavior_points, variant_topology, variant_shape)),
        }
    return predictions


def _validate_result_json(benchmark_dir: Path, readme_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    seq_len = int(readme_metrics.get("sample_packing_sequence_len") or 0)
    for result_path in sorted((benchmark_dir / "results").glob("*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        throughput = result.get("throughput", {})
        if throughput and seq_len:
            global_tokens = throughput.get("global_batch_size", 0) * seq_len
            derived_step_time = (
                global_tokens / throughput["tokens_per_sec"] if throughput.get("tokens_per_sec") else None
            )
            checks.extend(
                [
                    _check(
                        f"{result_path.name}:throughput_candidate",
                        "q36-main-af98064-deepepenv-05190533",
                        throughput.get("candidate"),
                    ),
                    _check(f"{result_path.name}:throughput_gpus", 32, throughput.get("gpus")),
                    _check(f"{result_path.name}:throughput_tokens_per_sec", 254600.0, throughput.get("tokens_per_sec")),
                    _check(
                        f"{result_path.name}:derived_step_time_sec",
                        throughput.get("step_time_sec"),
                        derived_step_time,
                        tolerance=0.01,
                    ),
                ]
            )
        k3_gate = result.get("k3_gate", {})
        if k3_gate:
            checks.extend(
                [
                    _check(f"{result_path.name}:k3_gate_status", "fail", k3_gate.get("status")),
                    _check(f"{result_path.name}:k3_total_tokens", 192, k3_gate.get("k3", {}).get("total_tokens")),
                    _check(
                        f"{result_path.name}:k3_primary_failure", "k3.mean <= 0.001", k3_gate.get("primary_failure")
                    ),
                ]
            )
        diagnostics = result.get("diagnostic_replays", [])
        if diagnostics:
            checks.append(_check(f"{result_path.name}:diagnostic_replay_count", 3, len(diagnostics)))
            checks.append(
                _check(
                    f"{result_path.name}:diagnostic_low_k3_rows",
                    2,
                    sum(1 for row in diagnostics if row.get("status") == "diagnostic_low_k3"),
                )
            )
    return checks


def validate_benchmark_dir(benchmark_dir: Path) -> dict[str, Any]:
    readme_path = benchmark_dir / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"missing README.md in {benchmark_dir}")
    readme_metrics = _parse_readme_metrics(readme_path.read_text(encoding="utf-8"))
    report, config_checks, behavior_predictions = _validate_config_behavior(benchmark_dir, readme_metrics)
    result_checks = _validate_result_json(benchmark_dir, readme_metrics)
    checks = config_checks + result_checks
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {
        "benchmark_dir": str(benchmark_dir),
        "status": status,
        "readme_metrics": readme_metrics,
        "simulator_report": report,
        "behavior_points": to_jsonable(load_benchmark_behavior_points(benchmark_dir)),
        "behavior_predictions": behavior_predictions,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks-root", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Benchmark model subdirectory to validate")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-fail-on-error", action="store_true", help="Always exit 0 after writing the report")
    args = parser.parse_args()

    payload = validate_benchmark_dir(args.benchmarks_root / args.model)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if payload["status"] != "pass" and not args.no_fail_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
