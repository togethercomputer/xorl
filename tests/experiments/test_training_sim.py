from pathlib import Path

import pytest
import yaml

from xorl.sim.analytical_ledgers import activation_ledger, communication_ledger, flops_ledger
from xorl.sim.benchmark_behavior import load_benchmark_behavior_points
from xorl.sim.calibration_evaluator import evaluate_calibration
from xorl.sim.calibration_packs import (
    list_calibration_packs,
    load_calibration_pack,
    resolve_calibration_pack,
    validate_calibration_pack,
)
from xorl.sim.collect_calibration import parse_log_text, summarize_observed_run
from xorl.sim.config_fingerprint import build_fingerprint, load_training_config, resolve_topology
from xorl.sim.feasibility_evaluator import evaluate_feasibility
from xorl.sim.kernel_variants import compare_kernel_variants, rank_kernel_variants
from xorl.sim.model_metadata import resolve_model_metadata
from xorl.sim.predict import build_report
from xorl.sim.scenario_planner import plan_scenario
from xorl.sim.shape_engine import balanced_counts, build_shape_ledger
from xorl.sim.tradeoff_ranker import rank_benchmark_tradeoffs
from xorl.sim.validate import validate_simulator


def test_balanced_counts_round_robin_distribution() -> None:
    assert balanced_counts(20, 6) == [4, 4, 3, 3, 3, 3]


def test_resolve_topology_matches_training_arguments_dp_formula() -> None:
    raw_config = {
        "train": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "ulysses_parallel_size": 4,
            "tensor_parallel_size": 1,
            "ringattn_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 8,
            "data_parallel_replicate_size": 2,
        },
        "data": {"sample_packing_sequence_len": 2048},
        "model": {"num_experts": 16, "num_experts_per_tok": 4},
    }

    topology = resolve_topology(raw_config, world_size=16, local_world_size=8)

    assert topology.data_parallel_size == 4
    assert topology.data_parallel_replicate_size == 2
    assert topology.data_parallel_shard_size == 2
    assert topology.global_batch_size == 64
    assert topology.sequence_parallel_size == 4
    assert topology.ep_fsdp_size == 2
    assert topology.num_experts == 16
    assert topology.top_k == 4


def test_shape_ledger_uses_sequence_parallel_local_tokens() -> None:
    raw_config = {
        "train": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "ulysses_parallel_size": 4,
            "expert_parallel_size": 4,
        },
        "data": {"sample_packing_sequence_len": 2048},
        "model": {"num_experts": 16, "num_experts_per_tok": 4},
    }
    topology = resolve_topology(raw_config, world_size=16, local_world_size=8)

    ledger = build_shape_ledger(topology, balanced_routing=True)

    assert ledger.microbatch_tokens_per_dp_rank == 2048
    assert ledger.global_tokens_per_microbatch == 8192
    assert ledger.global_tokens_per_train_step == 16384
    assert ledger.tokens_per_gpu_per_train_step == 1024
    assert ledger.tokens_per_model_rank_per_microbatch == 512
    assert ledger.routed_slots_per_model_rank_microbatch == 2048
    assert ledger.routed_slots_per_train_step_model_rank == 4096
    assert ledger.balanced_routing is not None
    assert ledger.balanced_routing.counts_by_expert == [128] * 16
    assert ledger.ep_rank_slots_per_microbatch == [512, 512, 512, 512]


def test_parse_structured_step_phase_and_memory_logs() -> None:
    log_text = """
    [STEP 4/9] loss=1.0 grad_norm=0.1 lr=1.0e-5 tflops=100.2 mfu=0.1010 tokens_per_sec=53414 time=72.100s peak_mem=39.8GB fwd=20.1GB bwd=39.8GB optim=10.0GB
    [STEP_PHASES 4/9] dataloader_max_s=0.100000 dataloader_mean_s=0.050000 model_forward_max_s=10.000000 model_forward_mean_s=9.000000
    [STEP_MEMORY 4/9] model_forward_after_allocated_max_gb=20.100 model_forward_phase_peak_allocated_max_gb=39.800
    """

    observed = parse_log_text(log_text, source="sample.log")
    summary = summarize_observed_run(observed, warmup_steps=0, world_size=16)

    assert len(observed.steps) == 1
    assert observed.steps[0].tokens_per_sec == 53414
    assert observed.steps[0].phase_memory_gb == {"fwd": 20.1, "bwd": 39.8, "optim": 10.0}
    assert observed.phases[0].metrics["model_forward_max_s"] == 10.0
    assert observed.memory_phases[0].metrics["model_forward_phase_peak_allocated_max_gb"] == 39.8
    assert summary["tokens_per_sec_per_gpu_mean"] == 3338.375


def _write_resolved_run_fixture(root: Path) -> Path:
    config = {
        "model": {
            "model_path": "Qwen/Qwen3.6-35B-A3B",
            "deepep_async_combine": False,
            "deepep_num_sms": 24,
            "deepep_buffer_size_gb": 1.0,
        },
        "data": {"sample_packing_sequence_len": 1024},
        "train": {
            "data_parallel_mode": "fsdp2",
            "data_parallel_replicate_size": 1,
            "data_parallel_shard_size": 4,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "ulysses_parallel_size": 1,
            "ringattn_parallel_size": 1,
            "expert_parallel_size": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 3,
            "enable_compile": True,
            "gradient_checkpointing_method": "recompute_full_layer",
            "enable_activation_offload": True,
            "activation_offload_prefetch_count": 4,
            "optimizer": "muon",
            "optimizer_dtype": "bf16",
            "muon_momentum": 0.0,
        },
    }
    run_dir = root / "resolved" / "fit"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "xorl_cli.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "startup_metrics.json").write_text(
        """
        {
          "repo_commit": "abc123",
          "metrics": {
            "startup/master_addr": "fit-master",
            "startup/node_count": 1,
            "startup/total_train_steps": 4
          }
        }
        """,
        encoding="utf-8",
    )
    log_dir = root / "fit"
    log_dir.mkdir()
    (log_dir / "node-0.log").write_text(
        """
        [STEP 1/4] loss=1.0 grad_norm=0.1 lr=1.0e-4 tflops=1.0 mfu=0.001 tokens_per_sec=100 time=10.0s peak_mem=60.0GB
        [STEP 2/4] loss=1.0 grad_norm=0.1 lr=1.0e-4 tflops=10.0 mfu=0.010 tokens_per_sec=1000 time=8.0s peak_mem=68.0GB
        [STEP 3/4] loss=1.0 grad_norm=0.1 lr=1.0e-4 tflops=12.0 mfu=0.012 tokens_per_sec=1200 time=7.0s peak_mem=70.0GB
        [STEP 4/4] loss=1.0 grad_norm=0.1 lr=1.0e-4 tflops=10.0 mfu=0.010 tokens_per_sec=1000 time=9.0s peak_mem=69.0GB
        """,
        encoding="utf-8",
    )

    oom_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    oom_config["train"]["micro_batch_size"] = 3
    oom_dir = root / "resolved" / "oom"
    oom_dir.mkdir()
    (oom_dir / "xorl_cli.yaml").write_text(yaml.safe_dump(oom_config), encoding="utf-8")
    (oom_dir / "startup_metrics.json").write_text(
        """
        {
          "repo_commit": "abc123",
          "metrics": {
            "startup/master_addr": "oom-master",
            "startup/node_count": 1
          }
        }
        """,
        encoding="utf-8",
    )
    oom_log_dir = root / "oom"
    oom_log_dir.mkdir()
    (oom_log_dir / "node-0.log").write_text(
        """
        [rank0]: Traceback (most recent call last):
        [rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 79.18 GiB of which 59.62 MiB is free. Including non-PyTorch memory, this process has 79.09 GiB memory in use.
        torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
        """,
        encoding="utf-8",
    )
    return config_path


def test_benchmark_behavior_loader_ingests_resolved_run_logs_and_ooms(tmp_path: Path) -> None:
    config_path = _write_resolved_run_fixture(tmp_path)

    points = load_benchmark_behavior_points(tmp_path)
    by_label = {point.label: point for point in points}

    fit = by_label[f"resolved_run:{config_path.parent.relative_to(tmp_path)}"]
    assert fit.status == "observed_log_summary"
    assert fit.correctness_status == "not_promoted"
    assert fit.tokens_per_sec == 1_100.0
    assert fit.step_time_sec == 8.0
    assert fit.tflops_per_gpu == 11.0
    assert fit.mfu_percent == 1.1
    assert fit.peak_mem_gb == 70.0
    assert fit.measured_steps == 2
    assert fit.warmup_steps == 2
    assert fit.gpu_count == 4
    assert fit.global_batch_size == 24
    assert fit.expert_parallel_size == 2
    assert fit.ep_fsdp_size == 2
    assert fit.deepep_num_sms == 24
    assert fit.enable_activation_offload is True
    assert fit.activation_offload_prefetch_count == 4

    oom = by_label["resolved_run:resolved/oom"]
    assert oom.status == "observed_log_oom"
    assert oom.correctness_status == "oom"
    assert oom.tokens_per_sec is None
    assert oom.peak_mem_gb == 79.09
    assert oom.micro_batch_size == 3


def test_scenario_planner_keeps_observed_fit_feasible_when_safety_margin_is_tight(tmp_path: Path) -> None:
    config_path = _write_resolved_run_fixture(tmp_path)

    report = plan_scenario(
        config_path,
        benchmark_dir=tmp_path,
        micro_batch_sizes=[2],
        expert_parallel_sizes=[2],
        device_memory_limit_gb=80.0,
        memory_safety_factor=1.15,
    )

    assert report.candidate_count == 1
    assert report.feasible_count == 1
    assert report.best_raw is not None
    assert report.best_raw.label == "mbs2-gb24-ep2-efsdp2-tp1-pp1-u1-r1:resolved_run:resolved/fit"
    assert report.best_raw.score_tokens_per_sec == 1_100.0
    assert report.best_raw.feasibility_status == "feasible_calibrated_peak_high_pressure"
    assert report.best_raw.memory_headroom_gb == -0.5
    assert report.best_raw.recommendation == "remeasure_for_stability"


def test_build_fingerprint_reads_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = {
        "train": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 3,
            "ulysses_parallel_size": 2,
            "data_parallel_shard_size": 4,
        },
        "data": {"sample_packing_sequence_len": 1024},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    fingerprint = build_fingerprint(
        config_path,
        world_size=8,
        local_world_size=8,
        balanced_routing=True,
        num_experts=8,
        top_k=2,
    )

    assert fingerprint.config_name == "config.yaml"
    assert fingerprint.balanced_routing is True
    assert fingerprint.topology.data_parallel_size == 4
    assert fingerprint.topology.data_parallel_replicate_size == 1
    assert fingerprint.topology.data_parallel_shard_size == 4
    assert fingerprint.topology.global_batch_size == 24
    assert len(fingerprint.config_sha256) == 64


def test_resolve_model_metadata_from_hf_cache(tmp_path: Path) -> None:
    config_dir = tmp_path / "models--Example--MoE" / "snapshots" / "abc123"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        """
        {
          "model_type": "example",
          "text_config": {
            "num_experts": 12,
            "num_experts_per_tok": 3,
            "num_hidden_layers": 7,
            "hidden_size": 128,
            "moe_intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 4096,
            "tie_word_embeddings": false
          }
        }
        """,
        encoding="utf-8",
    )

    metadata = resolve_model_metadata({"model": {"model_path": "Example/MoE"}}, hf_cache_roots=[tmp_path])

    assert metadata.source == "hf_config"
    assert metadata.num_experts == 12
    assert metadata.top_k == 3
    assert metadata.num_hidden_layers == 7
    assert metadata.moe_intermediate_size == 32
    assert metadata.num_attention_heads == 4
    assert metadata.num_key_value_heads == 2
    assert metadata.head_dim == 32
    assert metadata.tie_word_embeddings is False
    assert metadata.config_path is not None


def test_resolve_known_qwen235_metadata_without_hf_cache() -> None:
    metadata = resolve_model_metadata(
        {"model": {"model_path": "Qwen/Qwen3-235B-A22B"}},
        hf_cache_roots=[],
    )

    assert metadata.source == "known_model"
    assert metadata.num_experts == 128
    assert metadata.top_k == 8
    assert metadata.num_hidden_layers == 94
    assert metadata.hidden_size == 4096
    assert metadata.moe_intermediate_size == 1536
    assert metadata.num_attention_heads == 64
    assert metadata.num_key_value_heads == 4
    assert metadata.head_dim == 128
    assert metadata.vocab_size == 151936


def _write_q235_results_fixture(benchmark_dir: Path) -> None:
    benchmark_dir.mkdir()
    (benchmark_dir / "RESULTS.md").write_text(
        """
# Qwen3-235B-A22B @ 2k context

Measured: 4 nodes / 32xH100, u1/dp_shard32/EP8/ep_fsdp4.

| run | gcm | pack | mbs | tok/step | step s | MFU | tok/s tot | tok/s/GPU | peak GB | status |
|-----|-----|-----:|----:|---------:|-------:|----:|----------:|----------:|--------:|--------|
| n4_ep8_bd_pk4096 | before_dispatch | 4096 | 1 | 131,072 | ~18.4 | **~3.0%** | **~6,800** | ~213 | 68.3 | OK |
| n4_ep8_bd_pk4096_ga2 | before_dispatch | 4096 | 1 | 262,144 | ~31.3 | **~3.7%** | **~8,400** | ~263 | 68.3 | NEW BEST |
| n4_ep8_bd_pk16k | before_dispatch | 16384 | 1 | 524,288 | -- | -- | -- | -- | OOM | FAIL |
""",
        encoding="utf-8",
    )


def test_qwen235_markdown_loader_extracts_pack_and_ga_rows(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)

    points = load_benchmark_behavior_points(benchmark_dir)
    by_label = {point.label: point for point in points}

    assert set(by_label) == {
        "q235_markdown:n4_ep8_bd_pk4096",
        "q235_markdown:n4_ep8_bd_pk4096_ga2",
        "q235_markdown:n4_ep8_bd_pk16k",
    }
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].global_batch_size == 32
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].tokens_per_sec == 6_800.0
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].mfu_percent == 3.0
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].gpu_count == 32
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].sample_packing_sequence_len == 4096
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].tensor_parallel_size == 1
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].pipeline_parallel_size == 1
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].ulysses_parallel_size == 1
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].ringattn_parallel_size == 1
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].expert_parallel_size == 8
    assert by_label["q235_markdown:n4_ep8_bd_pk4096"].ep_fsdp_size == 4
    assert by_label["q235_markdown:n4_ep8_bd_pk4096_ga2"].global_batch_size == 64
    assert by_label["q235_markdown:n4_ep8_bd_pk4096_ga2"].tokens_per_sec == 8_400.0
    assert by_label["q235_markdown:n4_ep8_bd_pk16k"].sample_packing_sequence_len == 16384
    assert by_label["q235_markdown:n4_ep8_bd_pk16k"].tokens_per_sec is None
    assert by_label["q235_markdown:n4_ep8_bd_pk16k"].correctness_status == "oom"


def _write_q235_config_fixture(config_path: Path) -> None:
    config = {
        "model": {
            "model_path": "Qwen/Qwen3-235B-A22B",
            "ep_dispatch": "deepep",
            "moe_implementation": "quack",
            "deepep_buffer_size_gb": 2.0,
            "deepep_num_sms": 24,
            "deepep_async_combine": False,
        },
        "data": {
            "sample_packing_sequence_len": 4096,
        },
        "train": {
            "data_parallel_mode": "fsdp2",
            "ulysses_parallel_size": 1,
            "ringattn_parallel_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 8,
            "data_parallel_replicate_size": 1,
            "data_parallel_shard_size": 32,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "muon",
            "optimizer_dtype": "bf16",
            "muon_momentum": 0.95,
            "enable_mixed_precision": True,
            "skip_param_upcast": True,
            "fsdp_reduce_dtype": "fp32",
            "ce_mode": "quack_linear",
            "gradient_checkpointing_method": "recompute_before_dispatch",
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_qwen235_scenario_planner_uses_markdown_calibration_for_ga_tradeoff(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)

    report = plan_scenario(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
        micro_batch_sizes=[1],
        gradient_accumulation_steps=[1, 2],
        expert_parallel_sizes=[8],
    )

    assert report.candidate_count == 2
    assert report.best_raw is not None
    assert report.best_raw.label == "mbs1-gb64-ep8-efsdp4-tp1-pp1-u1-r1:q235_markdown:n4_ep8_bd_pk4096_ga2"
    assert report.best_raw.prediction_confidence == "calibrated"
    assert report.best_raw.score_tokens_per_sec == 8_400.0
    assert report.best_raw.behavior.tokens_per_sec_per_gpu == 262.5
    assert report.best_raw.analytic_peak_floor_gb == 56.812
    assert report.best_raw.estimated_peak_mem_gb == 68.3
    assert report.best_raw.memory_basis == "calibrated_peak"
    assert report.best_raw.feasibility_status == "feasible_calibrated_peak_high_pressure"
    assert report.best_promotable is None


def test_qwen235_scenario_planner_extrapolates_ga_asymptote_from_step_time_fit(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)

    report = plan_scenario(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
        micro_batch_sizes=[1],
        gradient_accumulation_steps=[1, 2, 4, 8],
        expert_parallel_sizes=[8],
    )

    assert report.candidate_count == 4
    assert report.best_raw is not None
    assert report.best_raw.label == "mbs1-gb256-ep8-efsdp4-tp1-pp1-u1-r1:extrapolated"
    assert report.best_raw.prediction_confidence == "extrapolated_step_time_fit"
    assert report.best_raw.score_tokens_per_sec == 10_200.0
    assert report.best_raw.behavior.step_time_sec == 102.801569
    assert report.best_raw.estimated_peak_mem_gb == 68.3
    assert report.best_raw.memory_basis == "calibrated_overhead_peak"
    assert report.best_raw.feasibility_status == "feasible_calibrated_overhead_peak_high_pressure"
    assert report.best_raw.calibration_scope == "outside_measured_envelope"
    assert report.best_raw.score_risk_adjusted_tokens_per_sec == 4666.194
    assert report.best_raw.recommendation == "remeasure_before_ranking"
    assert "outside_measured_envelope" in report.best_raw.risk_flags
    assert "requires_remeasurement" in report.best_raw.risk_flags
    assert report.best_raw.promotable is False
    assert report.best_risk_adjusted is not None
    assert report.best_risk_adjusted.label == ("mbs1-gb64-ep8-efsdp4-tp1-pp1-u1-r1:q235_markdown:n4_ep8_bd_pk4096_ga2")
    assert report.best_risk_adjusted.score_risk_adjusted_tokens_per_sec == 6783.0
    assert report.best_next_measurement is not None
    assert report.best_next_measurement.label == "mbs1-gb256-ep8-efsdp4-tp1-pp1-u1-r1:extrapolated"
    by_label = {candidate.label: candidate for candidate in report.candidates}
    ga4 = by_label["mbs1-gb128-ep8-efsdp4-tp1-pp1-u1-r1:extrapolated"]
    assert ga4.prediction_confidence == "extrapolated_step_time_fit"
    assert ga4.score_tokens_per_sec == 9_520.0


def test_qwen235_calibration_evaluator_reports_leave_one_out_ga_error(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)

    report = evaluate_calibration(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
    )

    assert report.status == "ok"
    assert report.measured_point_count == 2
    assert report.evaluated_count == 2
    assert report.skipped_count == 0
    assert report.prediction_status_counts == {"extrapolated": 2}
    assert report.mean_absolute_percentage_error == 19.16
    by_label = {holdout.label: holdout for holdout in report.holdouts}
    ga1 = by_label["q235_markdown:n4_ep8_bd_pk4096"]
    ga2 = by_label["q235_markdown:n4_ep8_bd_pk4096_ga2"]
    assert ga1.topology_label == "mbs1-gb32-ep8-efsdp4-tp1-pp1-u1-r1"
    assert ga1.predicted_tokens_per_sec == 7_560.0
    assert ga1.absolute_percentage_error == 11.176
    assert ga2.predicted_tokens_per_sec == 6_120.0
    assert ga2.absolute_percentage_error == 27.143


def test_qwen235_scenario_planner_does_not_exact_match_observed_row_to_tp_what_if(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)

    report = plan_scenario(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
        micro_batch_sizes=[1],
        gradient_accumulation_steps=[2],
        expert_parallel_sizes=[8],
        tensor_parallel_sizes=[1, 2],
    )

    by_label = {candidate.label: candidate for candidate in report.candidates}
    exact = by_label["mbs1-gb64-ep8-efsdp4-tp1-pp1-u1-r1:q235_markdown:n4_ep8_bd_pk4096_ga2"]
    tp2 = by_label["mbs1-gb32-ep8-efsdp4-tp2-pp1-u1-r1:extrapolated"]
    assert exact.prediction_confidence == "calibrated"
    assert exact.score_tokens_per_sec == 8_400.0
    assert tp2.prediction_confidence == "extrapolated"
    assert tp2.behavior.matched_label == "q235_markdown:n4_ep8_bd_pk4096_ga2"
    assert "TP extrapolation uses conservative communication penalty" in tp2.behavior.warnings
    assert tp2.score_tokens_per_sec == 6_804.0


def test_qwen235_scenario_planner_auto_sweeps_parallelism_strategy_space(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)

    report = plan_scenario(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
        micro_batch_sizes=[1],
        gradient_accumulation_steps=[2],
        topology_sweep="auto",
    )

    assert report.topology_sweep == "auto"
    assert {1, 2, 4, 8}.issubset({candidate.topology.tensor_parallel_size for candidate in report.candidates})
    assert 2 in {candidate.topology.pipeline_parallel_size for candidate in report.candidates}
    assert {8, 16, 32}.issubset({candidate.topology.expert_parallel_size for candidate in report.candidates})
    assert report.best_raw is not None
    assert report.best_raw.label == "mbs1-gb64-ep8-efsdp4-tp1-pp1-u1-r1:q235_markdown:n4_ep8_bd_pk4096_ga2"
    tp_candidates = [
        candidate
        for candidate in report.candidates
        if candidate.topology.tensor_parallel_size == 2
        and candidate.topology.pipeline_parallel_size == 1
        and candidate.topology.expert_parallel_size == 8
    ]
    assert tp_candidates
    assert all(candidate.prediction_confidence == "extrapolated" for candidate in tp_candidates)


def test_qwen235_auto_sweep_includes_long_context_cp_without_cross_seq_calibration(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["data"]["sample_packing_sequence_len"] = 64_000
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    report = plan_scenario(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
        micro_batch_sizes=[1],
        gradient_accumulation_steps=[1],
        expert_parallel_sizes=[8],
        tensor_parallel_sizes=[1],
        pipeline_parallel_sizes=[1],
        topology_sweep="auto",
    )

    cp_pairs = {
        (candidate.topology.ulysses_parallel_size, candidate.topology.ringattn_parallel_size)
        for candidate in report.candidates
    }
    assert {(2, 1), (4, 1), (1, 2)}.issubset(cp_pairs)
    assert report.feasible_count == 0
    assert {candidate.prediction_confidence for candidate in report.candidates} == {"unscored"}
    assert {candidate.calibration_scope for candidate in report.candidates} == {"outside_sequence_calibration_envelope"}
    base_cp = next(
        candidate
        for candidate in report.candidates
        if candidate.topology.ulysses_parallel_size == 1 and candidate.topology.ringattn_parallel_size == 1
    )
    assert "observed_oom_boundary:q235_markdown:n4_ep8_bd_pk16k" in base_cp.risk_flags


def test_qwen235_scenario_planner_marks_matching_oom_pack_infeasible(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "q235"
    _write_q235_results_fixture(benchmark_dir)
    config_path = tmp_path / "q235.yaml"
    _write_q235_config_fixture(config_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["data"]["sample_packing_sequence_len"] = 16384
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    report = plan_scenario(
        config_path,
        benchmark_dir=benchmark_dir,
        world_size=32,
        local_world_size=8,
        micro_batch_sizes=[1],
        gradient_accumulation_steps=[1],
        expert_parallel_sizes=[8],
    )

    assert report.candidate_count == 1
    assert report.feasible_count == 0
    candidate = report.candidates[0]
    assert candidate.label == "mbs1-gb32-ep8-efsdp4-tp1-pp1-u1-r1:q235_markdown:n4_ep8_bd_pk16k"
    assert candidate.behavior.status == "calibrated_failure"
    assert candidate.feasibility_status == "observed_oom"
    assert candidate.score_tokens_per_sec is None
    assert candidate.calibration_scope == "exact_calibrated"
    assert "observed_oom_boundary:q235_markdown:n4_ep8_bd_pk16k" in candidate.risk_flags


def test_builtin_calibration_packs_are_sanitized_and_versioned() -> None:
    assert list_calibration_packs() == ["qwen3_235b_a22b", "qwen3_5_397b_a17b", "qwen3_6_35b_a3b"]
    for name in list_calibration_packs():
        pack = load_calibration_pack(name)
        validation = validate_calibration_pack(pack.path)
        assert pack.manifest["schema_version"] == 1
        assert pack.default_config.is_file()
        assert validation["status"] == "pass"


def test_builtin_pack_prefix_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="unknown built-in calibration pack"):
        resolve_calibration_pack("builtin:../qwen3_6_35b_a3b")


def test_model_metadata_restricts_local_config_reads_to_approved_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    blocked_config = tmp_path / "blocked" / "config.json"
    allowed_root.mkdir()
    blocked_config.parent.mkdir()
    blocked_config.write_text('{"num_experts": 999}', encoding="utf-8")

    metadata = resolve_model_metadata(
        {"model": {"model_path": str(blocked_config)}},
        hf_cache_roots=[allowed_root],
    )

    assert metadata.config_path is None
    assert metadata.num_experts is None


def test_builtin_qwen35_pack_preserves_raw_and_promotable_winners() -> None:
    pack = load_calibration_pack("qwen3_5_397b_a17b")
    points = load_benchmark_behavior_points(pack.path)
    report = rank_benchmark_tradeoffs(pack.path)

    assert len(points) == 6
    assert report.best_raw is not None
    assert report.best_raw.score_tokens_per_sec == 59_217.0
    assert report.best_raw.promotable is False
    assert report.best_promotable is not None
    assert report.best_promotable.score_tokens_per_sec == 59_188.0
    assert report.best_promotable.promotable is True


def test_builtin_qwen36_pack_matches_default_config_but_remains_ungated() -> None:
    pack = load_calibration_pack("qwen3_6_35b_a3b")
    report = build_report(
        pack.default_config,
        world_size=None,
        local_world_size=None,
        balanced_routing=True,
        num_experts=None,
        top_k=None,
        benchmark_dir=pack.path,
    )

    assert report.benchmark_behavior is not None
    assert report.benchmark_behavior.matched_label == "readme_reference_mbs8"
    assert report.benchmark_behavior.tokens_per_sec == 261_000.0
    assert report.benchmark_behavior.correctness_status == "raw_speed_not_promoted_without_matching_k3_pass"
    assert report.support.support_status == "supported_local_non_pp"
    assert report.timing.timing_coverage_status == "benchmark_total_step_only"


def test_builtin_qwen235_pack_replays_fit_and_oom_boundaries() -> None:
    pack = load_calibration_pack("qwen3_235b_a22b")
    report = evaluate_feasibility(pack.default_config, benchmark_dir=pack.path)

    assert report.status == "ok"
    assert report.evaluated_count == 3
    assert report.accuracy == 1.0
    assert report.fit_recall == 1.0
    assert report.oom_recall == 1.0
    assert {holdout.actual_outcome for holdout in report.holdouts} == {"fit", "oom"}


def test_portable_analytical_core_covers_flops_activations_and_communication() -> None:
    pack = load_calibration_pack("qwen3_5_397b_a17b")
    raw_config = load_training_config(pack.default_config)
    topology = resolve_topology(raw_config)
    metadata = resolve_model_metadata(raw_config)

    flops = flops_ledger(metadata, topology)
    activations = activation_ledger(metadata, topology, raw_config["train"])
    communication = communication_ledger(metadata, topology, raw_config["train"])

    assert metadata.full_attention_interval == 4
    assert metadata.linear_num_value_heads == 64
    assert flops["status"] == "exact_analytic"
    assert flops["total_flops"] > 0
    assert activations["status"] == "exact_analytic_lower_bound"
    assert activations["analytic_activation_lower_bound_gb"] > 0
    assert communication["status"] == "exact_analytic_bytes"
    assert communication["total_per_rank_gb"] > 0


def test_kernel_variant_ranking_requires_a_correctness_gate() -> None:
    rows = [
        {
            "family": "attention",
            "variant": "fast-ungated",
            "workload": "qwen35-seq4096",
            "latency_ms": 8.0,
            "correctness_status": "not_promoted",
        },
        {
            "family": "attention",
            "variant": "validated",
            "workload": "qwen35-seq4096",
            "latency_ms": 10.0,
            "correctness_status": "pass",
        },
    ]

    report = rank_kernel_variants(rows)
    comparison = compare_kernel_variants(rows[1], rows[0])

    assert report["status"] == "ok"
    assert report["best"]["variant"] == "validated"
    assert report["measurements"][0]["variant"] == "fast-ungated"
    assert comparison["speedup"] == 1.25
    assert comparison["candidate_promotable"] is False


def test_consolidated_validator_covers_all_builtin_packs() -> None:
    report = validate_simulator()

    assert report["status"] == "pass"
    assert report["pack_count"] == 3
    assert report["check_count"] >= 200
    assert report["failed_check_count"] == 0
