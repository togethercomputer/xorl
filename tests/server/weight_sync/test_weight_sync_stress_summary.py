import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_stress_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "weight_sync_stress.py"
    spec = importlib.util.spec_from_file_location("weight_sync_stress", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_results_includes_p2p_breakdowns() -> None:
    stress = _load_stress_module()
    results = [
        {
            "sync_wall_s": 2.0,
            "sync": {
                "success": True,
                "transfer_time": 1.8,
                "total_bytes": 1_800_000_000,
                "timing_breakdown": {
                    "health_check_s": 0.1,
                    "transfer_s": 1.8,
                    "max_rank_transfer_s": 1.8,
                    "total_handler_s": 2.0,
                },
                "p2p_rank_summaries": [
                    {
                        "rank": 0,
                        "has_transfers": True,
                        "transfer_wall_s": 1.7,
                        "backend": {
                            "main_thread_s": 0.05,
                            "stage_s": 0.10,
                            "prepare_s": 0.20,
                            "pool_wait_s": 0.0,
                            "transfer_s": 1.4,
                        },
                        "phase_s": {
                            "direct_ep_s": 0.25,
                            "direct_ep_fp8_target_copy_s": 0.30,
                            "direct_ep_fp8_reduce_s": 0.40,
                        },
                    },
                    {
                        "rank": 1,
                        "has_transfers": True,
                        "transfer_wall_s": 1.8,
                        "backend": {
                            "main_thread_s": 0.06,
                            "stage_s": 0.11,
                            "prepare_s": 0.21,
                            "pool_wait_s": 0.0,
                            "transfer_s": 1.5,
                        },
                        "phase_s": {
                            "direct_ep_s": 0.35,
                            "direct_ep_fp8_target_copy_s": 0.50,
                            "direct_ep_fp8_reduce_s": 0.60,
                        },
                    },
                ],
            },
            "samples": [{"latency_s": 1.0}, {"latency_s": 2.0}],
        },
        {
            "sync_wall_s": 4.0,
            "sync": {
                "success": True,
                "transfer_time": 3.0,
                "total_bytes": 1_800_000_000,
                "timing_breakdown": {
                    "health_check_s": 0.2,
                    "transfer_s": 3.0,
                    "max_rank_transfer_s": 3.2,
                    "total_handler_s": 4.0,
                },
                "p2p_rank_summaries": [
                    {
                        "rank": 0,
                        "has_transfers": True,
                        "transfer_wall_s": 3.2,
                        "backend": {
                            "main_thread_s": 0.07,
                            "stage_s": 0.12,
                            "prepare_s": 0.22,
                            "pool_wait_s": 0.0,
                            "transfer_s": 2.9,
                        },
                        "phase_s": {
                            "direct_ep_s": 0.45,
                            "direct_ep_fp8_target_copy_s": 0.70,
                            "direct_ep_fp8_reduce_s": 0.80,
                        },
                    },
                    {
                        "rank": 1,
                        "has_transfers": True,
                        "transfer_wall_s": 2.8,
                        "backend": {
                            "main_thread_s": 0.08,
                            "stage_s": 0.13,
                            "prepare_s": 0.23,
                            "pool_wait_s": 0.0,
                            "transfer_s": 2.5,
                        },
                        "phase_s": {
                            "direct_ep_s": 0.55,
                            "direct_ep_fp8_target_copy_s": 0.90,
                            "direct_ep_fp8_reduce_s": 1.00,
                        },
                    },
                ],
            },
            "samples": [{"latency_s": 3.0}],
        },
        {"sync": {"success": False}, "error": "failed"},
    ]

    summary = stress.summarize_results(results)

    assert summary["iterations"] == 3
    assert summary["successful_syncs"] == 2
    assert summary["failed_syncs"] == 1
    assert summary["sync_wall_s"] == pytest.approx({"mean": 3.0, "p50": 2.0, "p95": 4.0})
    assert summary["transfer_time_s"] == pytest.approx({"mean": 2.4, "p50": 1.8, "p95": 3.0})
    assert summary["wall_transfer_gap_s"] == pytest.approx({"mean": 0.6, "p50": 0.2, "p95": 1.0})
    assert summary["sample_latency_s"] == pytest.approx({"mean": 2.0, "p50": 2.0, "p95": 3.0})
    assert summary["transfer_size_bytes"] == 1_800_000_000
    assert summary["mean_transfer_throughput_gb_s"] == pytest.approx(0.75)
    assert summary["handler_timing_means_s"]["health_check"] == pytest.approx(0.15)
    assert summary["p2p_max_rank_transfer_s"] == pytest.approx({"mean": 2.5, "p50": 1.8, "p95": 3.2})
    assert summary["p2p_wall_max_rank_gap_s"] == pytest.approx({"mean": 0.5, "p50": 0.2, "p95": 0.8})
    assert summary["p2p_per_rank_transfer_mean_s"] == pytest.approx({"0": 2.45, "1": 2.3})
    assert summary["p2p_backend_active_rank_means_s"]["worker_transfer"] == pytest.approx(2.075)
    assert summary["p2p_phase_active_rank_means_s"]["direct_ep"] == pytest.approx(0.4)
    assert summary["p2p_phase_active_rank_means_s"]["direct_ep_fp8_target_copy"] == pytest.approx(0.6)
    assert summary["p2p_phase_active_rank_means_s"]["direct_ep_fp8_reduce"] == pytest.approx(0.7)
    json.dumps(summary)


def test_parse_args_supports_skip_sample(monkeypatch) -> None:
    stress = _load_stress_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["weight_sync_stress.py", "--infer-url", "http://localhost:30000", "--skip-sample"],
    )

    args = stress.parse_args()

    assert args.skip_sample is True
