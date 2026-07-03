import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parents[2]
_K3_SKILL = _REPO_ROOT / "skills" / "xorl-k3-correctness-check" / "SKILL.md"
_BENCHMARKS = _REPO_ROOT / "skills" / "xorl-throughput-tuner" / "benchmarks"
_K3_SCRIPTS = _REPO_ROOT / "experiments" / "k3_tests"
_LOCAL_BENCHMARK = _REPO_ROOT / "experiments" / "local_benchmark"


def _load_module(path: Path, name: str):
    if not path.exists():  # pragma: no cover - experiments/ outside src+tests scope
        pytest.skip(f"k3 artifact '{path.name}' absent (experiments/ is outside the src+tests merge scope)")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_k3_skill_documents_static_trace_gate():
    if not _K3_SKILL.exists():  # pragma: no cover - skills/ outside src+tests scope
        pytest.skip("xorl-k3-correctness-check SKILL.md absent (skills/ is outside the src+tests merge scope)")
    text = _K3_SKILL.read_text(encoding="utf-8")

    for expected in (
        "make_static_traces.py",
        "compare_static_traces.py",
        "k3_gate.py",
        "k3_leaderboard.py",
        "--require-full-k3-coverage",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen3.6 trace bundle is not evidence for Qwen3.5-397B",
        "FLASHINFER_WORKSPACE_BASE",
        "Stale FlashInfer build files",
    ):
        assert expected in text

    for script in (
        _K3_SCRIPTS / "static_trace_utils.py",
        _K3_SCRIPTS / "make_static_traces.py",
        _K3_SCRIPTS / "refresh_static_traces.py",
        _K3_SCRIPTS / "compare_static_traces.py",
        _K3_SCRIPTS / "diagnose_static_k3.py",
        _K3_SCRIPTS / "extract_k3_repro_traces.py",
        _K3_SCRIPTS / "slice_static_trace_token.py",
        _LOCAL_BENCHMARK / "k3_gate.py",
        _LOCAL_BENCHMARK / "k3_leaderboard.py",
    ):
        if not script.exists():  # pragma: no cover - experiments/ outside src+tests scope
            pytest.skip(f"k3 artifact '{script.name}' absent (experiments/ is outside the src+tests merge scope)")


def test_static_trace_helpers_round_trip_and_gate(tmp_path):
    static_trace_utils = _load_module(_K3_SCRIPTS / "static_trace_utils.py", "static_trace_utils")
    k3_gate = _load_module(_LOCAL_BENCHMARK / "k3_gate.py", "k3_gate")

    traces_file = tmp_path / "traces.json"
    trace = {
        "trace_id": "t0",
        "prompt_ids": [1, 2, 3],
        "output_ids": [4, 5],
        "sglang_logprobs": [-0.1, -0.2],
    }
    static_trace_utils.write_static_trace_file(traces_file, {"model_name": "m"}, [trace])
    metadata, traces = static_trace_utils.load_static_trace_file(traces_file)
    normalized = static_trace_utils.normalize_trace(traces[0])

    assert metadata["model_name"] == "m"
    assert normalized["full_ids"] == [1, 2, 3, 4, 5]
    assert static_trace_utils.xorl_input_ids_for_trace(normalized) == [1, 2, 3, 4]
    assert static_trace_utils.labels_for_trace(normalized) == [-100, -100, 4, 5]

    k3_file = tmp_path / "k3.json"
    k3_file.write_text(
        json.dumps(
            {
                "aggregate": {
                    "total_prompts": 1,
                    "total_tokens": 2,
                    "k3": {"mean": 1e-4, "p95": 2e-4, "max": 2e-4},
                    "per_sample_k3": {"mean": 1e-4},
                }
            }
        ),
        encoding="utf-8",
    )
    throughput = k3_gate.ThroughputMetrics(
        candidate="candidate",
        source="manual",
        status="manual",
        measured_steps=0,
        tokens_per_sec_mean=1000.0,
        tokens_per_sec_per_gpu_mean=125.0,
    )
    k3 = k3_gate.attach_static_trace_coverage(k3_gate.parse_k3_result(k3_file), traces_file)
    result = k3_gate.build_gate_result(
        throughput,
        k3,
        max_mean_k3=1e-2,
        max_p95_k3=3e-2,
        min_tokens_per_sec=None,
        min_tokens_per_sec_per_gpu=None,
        require_full_k3_coverage=True,
    )

    assert result["status"] == "pass"
    assert result["k3"]["expected_prompts"] == 1
    assert result["k3"]["expected_tokens"] == 2


def test_launch_k3_dry_run_helper_uses_server_side_apply(monkeypatch, tmp_path):
    launcher = _load_module(_K3_SCRIPTS / "launch_k3_test.py", "launch_k3_test")
    if not hasattr(launcher, "dry_run_pod_manifest"):  # pragma: no cover - experiments/ outside src+tests scope
        pytest.skip("launch_k3_test.dry_run_pod_manifest absent (experiments/ is outside the src+tests merge scope)")
    calls = []

    def fake_kubectl(cmd, check=True):
        calls.append((cmd, check))
        return "pod/k3-xo-test serverside-applied (server dry run)"

    monkeypatch.setattr(launcher, "kubectl", fake_kubectl)
    yaml_path = tmp_path / "pod.yaml"

    launcher.dry_run_pod_manifest("k3-xo-test", "apiVersion: v1\nkind: Pod\n", str(yaml_path))

    assert yaml_path.read_text(encoding="utf-8") == "apiVersion: v1\nkind: Pod\n"
    assert calls == [("apply --server-side --dry-run=server -f " + str(yaml_path), True)]


def test_k3_diagnosis_marks_numeric_like_outlier_tokens():
    diagnosis = _load_module(_K3_SCRIPTS / "diagnose_static_k3.py", "diagnose_static_k3")

    assert diagnosis._looks_numeric_like(" 2026")
    assert diagnosis._looks_numeric_like("3.1415")
    assert diagnosis._looks_numeric_like("12/31")
    assert not diagnosis._looks_numeric_like("token")
    assert not diagnosis._looks_numeric_like("abc123")


def test_qwen36_summary_records_failed_gate_and_model_specific_traces():
    path = _BENCHMARKS / "qwen3_6_35b_a3b" / "results" / "qwen36_static_k3_summary_20260519.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert data["static_traces"]["num_traces"] == 3
    assert data["static_traces"]["total_tokens"] == 192
    assert "must not be reused for Qwen3.5-397B" in data["static_traces"]["note"]
    assert data["k3_gate"]["status"] == "fail"
    assert data["k3_gate"]["k3"]["mean"] > 1.0
    assert any(row["k3_mean"] < 0.01 for row in data["diagnostic_replays"])


def test_qwen35_mfu_summary_has_trace_status_and_nonasync_candidate():
    path = _BENCHMARKS / "qwen3_5_397b_a17b" / "results" / "shortctx_8node_mfu_summary_20260519.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["model"] == "Qwen/Qwen3.5-397B-A17B"
    assert "R73 synchronous-combine row passed" in data["k3_status"]
    assert "R75 remains raw-speed-only" in data["k3_status"]
    rows = data["best_by_mfu"]
    assert rows[0]["mfu_percent"] == pytest.approx(9.44)
    assert rows[0]["deepep_async_combine"] is True
    assert any(
        row["trial"].startswith("r73-")
        and row["deepep_async_combine"] is False
        and row.get("k3_gate", "").startswith("pass:")
        for row in rows
    )

    status_path = _BENCHMARKS / "qwen3_5_397b_a17b" / "results" / "static_trace_k3_status_20260520.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "static_traces_replayed_r73_passed"
    assert status["static_traces"]["num_traces"] == 3
    assert status["static_traces"]["total_output_tokens"] == 129
    assert status["replay_candidate"]["replay_status"] == "passed"
    assert status["replay_candidate"]["k3"]["p95"] < 0.003
    assert status["promotion_gate"]["status"] == "pass"

    summary_path = _BENCHMARKS / "qwen3_5_397b_a17b" / "results" / "r73_static_k3_summary_20260520.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["throughput"]["mfu_percent"] == pytest.approx(9.44)
    assert summary["k3_gate"]["total_tokens"] == 129
    assert summary["diagnosis"]["worst_token"]["target_token_text"] == "<|im_end|>"
