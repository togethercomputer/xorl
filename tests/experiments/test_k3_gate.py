import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_BENCHMARK_DIR = REPO_ROOT / "experiments" / "local_benchmark"
if str(LOCAL_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_BENCHMARK_DIR))


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, LOCAL_BENCHMARK_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


k3_gate = _load_module("k3_gate")
k3_leaderboard = _load_module("k3_leaderboard")
run_local_benchmark = _load_module("run_local_benchmark")


def _write_k3(path: Path, mean: float = 1e-4, p95: float = 2e-4):
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "total_prompts": 2,
                    "total_tokens": 6,
                    "k3": {"mean": mean, "p95": p95, "max": p95},
                    "per_sample_k3": {"mean": mean},
                }
            }
        )
    )


def _write_k3_with_samples(path: Path):
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "total_prompts": 1,
                    "total_tokens": 2,
                    "k3": {"mean": 4.0, "p95": 7.0, "max": 8.0},
                    "per_sample_k3": {"mean": 4.0},
                },
                "samples": [
                    {
                        "trace_id": "trace-a",
                        "prompt_len": 3,
                        "gen_len": 2,
                        "sample_k3_mean": 4.0,
                        "per_token": [
                            {"position": 0, "token_id": 101, "k3": 0.1, "sglang_logprob": -1.0, "xorl_logprob": -1.2},
                            {"position": 1, "token_id": 102, "k3": 8.0, "sglang_logprob": -0.5, "xorl_logprob": -6.5},
                        ],
                    }
                ],
            }
        )
    )


def _write_gate_summary(
    path: Path,
    *,
    candidate: str,
    status: str,
    tokens_per_sec: float,
    tokens_per_sec_per_gpu: float,
    k3_mean: float | None = None,
    k3_p95: float | None = None,
    failure_reasons: list[str] | None = None,
    k3_failure_reason: str | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate": candidate,
                "status": status,
                "failure_reasons": failure_reasons or [],
                "throughput": {
                    "tokens_per_sec_mean": tokens_per_sec,
                    "tokens_per_sec_per_gpu_mean": tokens_per_sec_per_gpu,
                    "mfu_mean": 0.15,
                    "step_time_sec_mean": 6.0,
                },
                "k3": {
                    "status": "pass" if status == "pass" else "failed",
                    "mean": k3_mean,
                    "p95": k3_p95,
                    "max": k3_p95,
                    "failure_reason": k3_failure_reason,
                },
                "artifacts": {"static_traces_file": "traces.json"},
            }
        )
    )


def test_gate_passes_for_summary_and_low_k3(tmp_path):
    summary = tmp_path / "benchmark_summary.json"
    summary.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path / "run-a"),
                "measured_metrics": {
                    "tokens_per_sec": {"mean": 256000.0, "max": 260000.0},
                    "tflops_per_gpu": {"mean": 120.0},
                    "step_time_sec": {"mean": 6.1},
                },
                "measured_steps": [{"mfu": 0.15}, {"mfu": 0.17}],
            }
        )
    )
    k3_path = tmp_path / "k3.json"
    _write_k3(k3_path)

    throughput = k3_gate.resolve_throughput(
        summary,
        candidate=None,
        trial=None,
        warmup_steps=0,
        gpus=32,
        tokens_per_step=None,
        manual_tokens_per_sec=None,
        manual_tflops_per_gpu=None,
        manual_mfu=None,
        manual_step_time_sec=None,
    )
    result = k3_gate.build_gate_result(
        throughput,
        k3_gate.parse_k3_result(k3_path),
        max_mean_k3=1e-3,
        max_p95_k3=1e-2,
        min_tokens_per_sec_per_gpu=7000.0,
        min_tokens_per_sec=None,
        artifacts={"static_traces_file": "traces.json"},
    )

    assert result["status"] == "pass"
    assert result["throughput"]["tokens_per_sec_per_gpu_mean"] == 8000.0
    assert result["throughput"]["mfu_mean"] == 0.16
    assert result["artifacts"]["static_traces_file"] == "traces.json"


def test_gate_artifacts_include_model_diagnoses(tmp_path):
    args = type(
        "Args",
        (),
        {
            "throughput_source": tmp_path / "summary.json",
            "k3_result": tmp_path / "k3.json",
            "static_traces_file": tmp_path / "traces.json",
            "k3_diagnosis": tmp_path / "diagnosis.json",
            "k3_repro_traces_file": None,
            "model_diagnosis": [tmp_path / "gdn-layer0.json", tmp_path / "gdn-layer38.json"],
            "resolved_config": None,
            "manifest": None,
        },
    )()

    artifacts = k3_gate.build_artifacts_from_args(args)

    assert artifacts["model_diagnoses"] == [
        str(tmp_path / "gdn-layer0.json"),
        str(tmp_path / "gdn-layer38.json"),
    ]
    assert artifacts["k3_diagnosis"] == str(tmp_path / "diagnosis.json")


def test_gate_fails_for_high_k3(tmp_path):
    k3_path = tmp_path / "k3.json"
    _write_k3(k3_path, mean=0.02, p95=0.03)

    throughput = k3_gate.ThroughputMetrics(
        candidate="candidate-a",
        source="manual",
        status="manual",
        measured_steps=0,
        tokens_per_sec_mean=250000.0,
    )
    result = k3_gate.build_gate_result(
        throughput,
        k3_gate.parse_k3_result(k3_path),
        max_mean_k3=1e-3,
        max_p95_k3=1e-2,
        min_tokens_per_sec=None,
        min_tokens_per_sec_per_gpu=None,
    )

    assert result["status"] == "fail"
    assert "k3.mean <= 0.001" in result["failure_reasons"]
    assert "k3.p95 <= 0.01" in result["failure_reasons"]


def test_parse_k3_result_builds_diagnostics_from_samples(tmp_path):
    k3_path = tmp_path / "k3.json"
    _write_k3_with_samples(k3_path)

    parsed = k3_gate.parse_k3_result(k3_path)

    assert parsed.diagnostics["worst_tokens"][0]["trace_id"] == "trace-a"
    assert parsed.diagnostics["worst_tokens"][0]["position"] == 1
    assert parsed.diagnostics["worst_samples"][0]["max_token_id"] == 102


def test_gate_records_failed_k3_replay_without_json():
    throughput = k3_gate.ThroughputMetrics(
        candidate="candidate-a",
        source="manual",
        status="manual",
        measured_steps=0,
        tokens_per_sec_mean=250000.0,
    )
    result = k3_gate.build_gate_result(
        throughput,
        k3_gate.failed_k3_result("CUDA illegal memory access during static replay"),
        max_mean_k3=1e-3,
        max_p95_k3=1e-2,
        min_tokens_per_sec=None,
        min_tokens_per_sec_per_gpu=None,
    )

    assert result["status"] == "fail"
    assert result["k3"]["status"] == "failed"
    assert "CUDA illegal memory access" in "; ".join(result["failure_reasons"])


def test_static_trace_coverage_counts_json_and_jsonl(tmp_path):
    json_path = tmp_path / "traces.json"
    json_path.write_text(
        json.dumps(
            {
                "traces": [
                    {"trace_id": "a", "output_ids": [1, 2, 3]},
                    {"trace_id": "b", "gen_len": 4},
                ]
            }
        )
    )
    jsonl_path = tmp_path / "traces.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"trace_id": "a", "output_ids": [1]}),
                json.dumps({"trace_id": "b", "output_ids": [2, 3]}),
            ]
        )
    )

    assert k3_gate.static_trace_coverage(json_path) == {"prompts": 2, "tokens": 7}
    assert k3_gate.static_trace_coverage(jsonl_path) == {"prompts": 2, "tokens": 3}


def test_gate_can_require_full_static_trace_coverage(tmp_path):
    traces = tmp_path / "traces.json"
    traces.write_text(
        json.dumps(
            {
                "traces": [
                    {"trace_id": "a", "output_ids": [1, 2, 3]},
                    {"trace_id": "b", "output_ids": [4, 5, 6]},
                ]
            }
        )
    )
    k3_path = tmp_path / "k3.json"
    _write_k3(k3_path, mean=1e-4, p95=2e-4)

    throughput = k3_gate.ThroughputMetrics(
        candidate="candidate-a",
        source="manual",
        status="manual",
        measured_steps=0,
        tokens_per_sec_mean=250000.0,
    )
    k3 = k3_gate.attach_static_trace_coverage(k3_gate.parse_k3_result(k3_path), traces)

    passing = k3_gate.build_gate_result(
        throughput,
        k3,
        max_mean_k3=1e-3,
        max_p95_k3=1e-2,
        min_tokens_per_sec=None,
        min_tokens_per_sec_per_gpu=None,
        require_full_k3_coverage=True,
    )
    assert passing["status"] == "pass"
    assert passing["k3"]["expected_prompts"] == 2
    assert passing["k3"]["expected_tokens"] == 6

    k3.total_prompts = 1
    k3.total_tokens = 3
    failing = k3_gate.build_gate_result(
        throughput,
        k3,
        max_mean_k3=1e-3,
        max_p95_k3=1e-2,
        min_tokens_per_sec=None,
        min_tokens_per_sec_per_gpu=None,
        require_full_k3_coverage=True,
    )
    assert failing["status"] == "fail"
    assert "k3.total_prompts >= 2" in failing["failure_reasons"]
    assert "k3.total_tokens >= 6" in failing["failure_reasons"]


def test_gate_can_require_minimum_k3_replay_size(tmp_path):
    k3_path = tmp_path / "k3.json"
    _write_k3(k3_path, mean=1e-4, p95=2e-4)
    throughput = k3_gate.ThroughputMetrics(
        candidate="candidate-a",
        source="manual",
        status="manual",
        measured_steps=0,
        tokens_per_sec_mean=250000.0,
    )

    result = k3_gate.build_gate_result(
        throughput,
        k3_gate.parse_k3_result(k3_path),
        max_mean_k3=1e-3,
        max_p95_k3=1e-2,
        min_tokens_per_sec=None,
        min_tokens_per_sec_per_gpu=None,
        min_k3_prompts=4,
        min_k3_tokens=8,
    )

    assert result["status"] == "fail"
    assert "k3.total_prompts >= 4" in result["failure_reasons"]
    assert "k3.total_tokens >= 8" in result["failure_reasons"]


def test_k3_leaderboard_promotes_fastest_passing_candidate(tmp_path):
    _write_gate_summary(
        tmp_path / "fast-failed" / "k3_gated_summary.json",
        candidate="fast-failed",
        status="fail",
        tokens_per_sec=320000.0,
        tokens_per_sec_per_gpu=10000.0,
        failure_reasons=["static replay failed"],
        k3_failure_reason="CUDA illegal memory access",
    )
    _write_gate_summary(
        tmp_path / "slow-passing" / "k3_gated_summary.json",
        candidate="slow-passing",
        status="pass",
        tokens_per_sec=250000.0,
        tokens_per_sec_per_gpu=7812.5,
        k3_mean=1e-4,
        k3_p95=2e-4,
    )

    leaderboard = k3_leaderboard.build_leaderboard([tmp_path])

    assert leaderboard["summary"]["total"] == 2
    assert leaderboard["summary"]["passing"] == 1
    assert leaderboard["best_passing"]["candidate"] == "slow-passing"
    assert leaderboard["fastest_overall"]["candidate"] == "fast-failed"
    assert leaderboard["rows"][0]["candidate"] == "slow-passing"
    assert leaderboard["rows"][1]["candidate"] == "fast-failed"
    assert leaderboard["rows"][1]["primary_failure"] == "CUDA illegal memory access"


def test_k3_leaderboard_discovers_files_once(tmp_path):
    gate_file = tmp_path / "candidate-a" / "k3_gated_summary.json"
    _write_gate_summary(
        gate_file,
        candidate="candidate-a",
        status="pass",
        tokens_per_sec=123.0,
        tokens_per_sec_per_gpu=12.3,
        k3_mean=1e-5,
        k3_p95=2e-5,
    )

    discovered = k3_leaderboard.discover_gate_files([tmp_path, gate_file])

    assert discovered == [gate_file]


def test_parse_trials_csv_selects_trial_and_computes_per_gpu(tmp_path):
    path = tmp_path / "trials.csv"
    path.write_text(
        "\n".join(
            [
                "trial,status,world,mean_tokens_per_sec,mean_tflops_per_gpu,mean_step_time_s,measured_steps",
                "r1,complete,32,1000,10,7,4",
                "r2,complete,64,6400,20,5,6",
            ]
        )
    )

    throughput = k3_gate.parse_trials_csv(path, "r2", None, None)

    assert throughput.candidate == "r2"
    assert throughput.tokens_per_sec_mean == 6400.0
    assert throughput.tokens_per_sec_per_gpu_mean == 100.0
    assert throughput.measured_steps == 6


def test_parse_tqdm_log_uses_warmup_and_tokens_per_step(tmp_path):
    path = tmp_path / "node-0.log"
    path.write_text("Epoch 1/1: loss=1 tok/s=100\nEpoch 1/1: loss=1 tok/s=200\nEpoch 1/1: loss=1 tok/s=400\n")

    throughput = k3_gate.parse_log_source(
        path,
        warmup_steps=1,
        tokens_per_step=800.0,
        candidate="run",
        gpus=2,
    )

    assert throughput.measured_steps == 2
    assert throughput.tokens_per_sec_mean == 300.0
    assert throughput.tokens_per_sec_per_gpu_mean == 150.0
    assert throughput.step_time_sec_mean == 3.0


def test_local_benchmark_writes_k3_gate_artifact(tmp_path):
    summary = tmp_path / "benchmark_summary.json"
    summary.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path),
                "measured_metrics": {
                    "tokens_per_sec": {"mean": 3200.0, "max": 3300.0},
                    "tflops_per_gpu": {"mean": 40.0},
                    "step_time_sec": {"mean": 2.0},
                },
                "measured_steps": [{"mfu": 0.12}],
            }
        )
    )
    k3_path = tmp_path / "k3.json"
    _write_k3(k3_path)

    result = run_local_benchmark._write_k3_gate_artifact(
        summary_path=summary,
        output_dir=tmp_path,
        candidate="run-a",
        nproc_per_node=4,
        options=run_local_benchmark.K3GateOptions(
            k3_result=k3_path,
            k3_failure_reason=None,
            static_traces_file=None,
            require_k3=True,
            max_mean_k3=1e-3,
            max_p95_k3=1e-2,
            min_k3_prompts=None,
            min_k3_tokens=None,
            require_full_k3_coverage=False,
            min_tokens_per_sec=None,
            min_tokens_per_sec_per_gpu=700.0,
        ),
    )

    output = tmp_path / "k3_gated_summary.json"
    assert result["status"] == "pass"
    assert result["tokens_per_sec_per_gpu_mean"] == 800.0
    assert json.loads(output.read_text())["status"] == "pass"


def test_local_benchmark_required_k3_fails_without_input(tmp_path):
    with pytest.raises(RuntimeError, match="K3 gate is required"):
        run_local_benchmark._write_k3_gate_artifact(
            summary_path=tmp_path / "benchmark_summary.json",
            output_dir=tmp_path,
            candidate="run-a",
            nproc_per_node=4,
            options=run_local_benchmark.K3GateOptions(
                k3_result=None,
                k3_failure_reason=None,
                static_traces_file=None,
                require_k3=True,
                max_mean_k3=1e-3,
                max_p95_k3=1e-2,
                min_k3_prompts=None,
                min_k3_tokens=None,
                require_full_k3_coverage=False,
                min_tokens_per_sec=None,
                min_tokens_per_sec_per_gpu=None,
            ),
        )


def test_local_benchmark_passes_k3_coverage_options(tmp_path):
    summary = tmp_path / "benchmark_summary.json"
    summary.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path),
                "measured_metrics": {
                    "tokens_per_sec": {"mean": 3200.0, "max": 3300.0},
                    "tflops_per_gpu": {"mean": 40.0},
                    "step_time_sec": {"mean": 2.0},
                },
                "measured_steps": [{"mfu": 0.12}],
            }
        )
    )
    k3_path = tmp_path / "k3.json"
    _write_k3(k3_path)
    traces = tmp_path / "traces.json"
    traces.write_text(
        json.dumps(
            {
                "traces": [
                    {"trace_id": "a", "output_ids": [1, 2, 3]},
                    {"trace_id": "b", "output_ids": [4, 5, 6]},
                ]
            }
        )
    )

    result = run_local_benchmark._write_k3_gate_artifact(
        summary_path=summary,
        output_dir=tmp_path,
        candidate="run-a",
        nproc_per_node=4,
        options=run_local_benchmark.K3GateOptions(
            k3_result=k3_path,
            k3_failure_reason=None,
            static_traces_file=traces,
            require_k3=True,
            max_mean_k3=1e-3,
            max_p95_k3=1e-2,
            min_k3_prompts=2,
            min_k3_tokens=6,
            require_full_k3_coverage=True,
            min_tokens_per_sec=None,
            min_tokens_per_sec_per_gpu=700.0,
        ),
    )

    gate = json.loads((tmp_path / "k3_gated_summary.json").read_text())
    assert result["status"] == "pass"
    assert result["k3_expected_prompts"] == 2
    assert gate["k3"]["expected_tokens"] == 6
    assert gate["artifacts"]["static_traces_file"] == str(traces)
