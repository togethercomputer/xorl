import importlib.util
import json
import re
import sys
from argparse import Namespace
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL_DIR = _REPO_ROOT / "skills" / "xorl-throughput-tuner"


def _load_script(name: str):
    script_path = _SKILL_DIR / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"xorl_skill_{name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _topology_args(**overrides):
    args = {
        "model": "qwen3-235b-a22b",
        "nodes": 4,
        "gpus_per_node": 8,
        "world_size": None,
        "seq_len": 8192,
        "num_experts": None,
        "kv_heads": None,
        "max_ulysses": 1,
        "pp_sizes": "1",
        "tp_sizes": "1",
        "ringattn_sizes": "1",
        "ulysses_sizes": "1",
        "ep_sizes": "8,16",
        "cp_fsdp_mode": "all",
        "min_ep_fsdp": 1,
        "require_ep_divides_experts": True,
        "top": 20,
        "format": "table",
    }
    args.update(overrides)
    return Namespace(**args)


def _render_args(**overrides):
    args = {
        "name": "qwen36-repro",
        "config_path": "skills/xorl-throughput-tuner/benchmarks/qwen3_6_35b_a3b/configs/qwen3_6_35b_a3b_8k_4node_ep8_mbs8_fullrecompute_deepep.yaml",
        "nodes": 2,
        "gpus_per_node": 8,
        "output": Path("/tmp/unused.yaml"),
        "image": "xorl:test",
        "repo_root": "/workspace/xorl-internal",
        "home_mount": "/workspace",
        "shared_mount": "/shared",
        "home_pvc": "home-pvc",
        "shared_pvc": "shared-pvc",
        "torchrun_bin": "/workspace/.venv/bin/torchrun",
        "master_port": 29500,
        "result_root": None,
        "node_names": "node-a.example.com,node-b.example.com",
        "node_selector": ["accelerator=nvidia-h100"],
        "node_selector_key": "node-group",
        "node_selector_values": "default,nccl",
        "priority_class": "normal",
        "runtime_class": "nvidia",
        "source_shell_env": True,
        "privileged": False,
        "nccl_bond0_preset": True,
        "env": ["NCCL_NET_GDR_LEVEL=PHB", "XORL_MOE_SYNTHETIC_ROUTING=balanced"],
        "toleration": True,
        "toleration_key": "node-group",
        "toleration_value": "nccl",
        "toleration_effect": "NoSchedule",
        "extra_toleration": ["nvidia.com/weka-error=mount-failure"],
    }
    args.update(overrides)
    return Namespace(**args)


def test_topology_candidates_keep_ep_fsdp_legal_for_qwen_moe():
    topology = _load_script("xorl_topology_candidates")

    candidates = topology.generate_candidates(_topology_args())

    assert candidates
    assert all(candidate.world_size == 32 for candidate in candidates)
    assert all(
        candidate.world_size == candidate.pp * candidate.tp * candidate.ringattn * candidate.ulysses * candidate.dp_size
        for candidate in candidates
    )
    ep8 = next(candidate for candidate in candidates if candidate.ep == 8)
    assert ep8.ep_fsdp == 4
    assert ep8.experts_per_ep_rank == 16
    assert "--train.expert_parallel_size 8" in ep8.overrides


def test_render_k8s_manifest_dedupes_env_and_pins_nodes():
    renderer = _load_script("render_k8s_torchrun_jobs")

    manifest = renderer.render(_render_args())

    assert manifest.count("kind: Job") == 2
    assert manifest.count("nodeSelector:") == 2
    assert "accelerator: nvidia-h100" in manifest
    assert "node-group: default" in manifest
    assert "node-group: nccl" in manifest
    assert "nodeName: node-a.example.com" in manifest
    assert "nodeName: node-b.example.com" in manifest
    assert "claimName: home-pvc" in manifest
    assert "claimName: shared-pvc" in manifest
    nccl_gdr = re.search(r'- name: NCCL_NET_GDR_LEVEL\s+value: "([^"]+)"', manifest)
    assert nccl_gdr is not None, "NCCL_NET_GDR_LEVEL env entry missing"
    assert nccl_gdr.group(1) == "PHB"
    assert 'value: "balanced"' in manifest
    assert "git_commit" in manifest
    assert "--node_rank=0" in manifest
    assert "--node_rank=1" in manifest
    assert "securityContext:" not in manifest
    assert "privileged: true" not in manifest


def test_render_k8s_manifest_requires_one_node_name_per_node():
    renderer = _load_script("render_k8s_torchrun_jobs")

    with pytest.raises(ValueError, match="--node-names"):
        renderer.render(_render_args(node_names="node-a.example.com"))


def test_render_k8s_manifest_requires_one_node_selector_value_per_node():
    renderer = _load_script("render_k8s_torchrun_jobs")

    with pytest.raises(ValueError, match="--node-selector-values"):
        renderer.render(_render_args(node_selector_values="default"))


def test_collect_xorl_metrics_parses_structured_logs_after_warmup(tmp_path):
    collector = _load_script("collect_xorl_metrics")
    log_path = tmp_path / "node-0.log"
    log_path.write_text(
        "\n".join(
            [
                "[STEP 1/3] loss=1.0 grad_norm=0.1 lr=1e-5 tflops=10.0 mfu=0.1 tokens_per_sec=100.0 time=2.0s peak_mem=40.0GB",
                "[STEP 2/3] loss=1.0 grad_norm=0.1 lr=1e-5 tflops=20.0 mfu=0.2 tokens_per_sec=200.0 time=1.0s peak_mem=41.0GB",
                "[STEP 3/3] loss=1.0 grad_norm=0.1 lr=1e-5 tflops=30.0 mfu=0.3 tokens_per_sec=300.0 time=0.5s peak_mem=42.0GB",
            ]
        ),
        encoding="utf-8",
    )

    rows = collector.collect([tmp_path], warmup=1)

    assert len(rows) == 1
    row = rows[0]
    assert row.steps == 2
    assert row.tokens_per_sec_mean == pytest.approx(250.0)
    assert row.tokens_per_sec_max == pytest.approx(300.0)
    assert row.tflops_per_gpu_mean == pytest.approx(25.0)
    assert row.step_time_sec_mean == pytest.approx(0.75)
    assert row.peak_mem_gb_max == pytest.approx(42.0)
    assert row.status == "ok"


def test_qwen36_benchmark_config_keeps_repro_guardrails():
    config = (
        _SKILL_DIR
        / "benchmarks"
        / "qwen3_6_35b_a3b"
        / "configs"
        / "qwen3_6_35b_a3b_8k_4node_ep8_mbs8_fullrecompute_deepep.yaml"
    ).read_text(encoding="utf-8")

    for expected in (
        "deepep_num_sms: 72",
        "deepep_async_combine: true",
        "data_parallel_shard_size: 32",
        "expert_parallel_size: 8",
        "micro_batch_size: 8",
        "empty_cache_steps: 10",
        "gc_steps: 10",
        "gradient_checkpointing_method: recompute_full_layer",
        "enable_compile: true",
        "save_steps: 0",
        "save_epochs: 0",
        "log_format: structured",
        "use_wandb: false",
    ):
        assert expected in config


def test_topology_candidates_resolve_qwen36_preset():
    topology = _load_script("xorl_topology_candidates")
    preset = topology.resolve_preset("Qwen/Qwen3.6-35B-A3B")
    assert preset is not None, "qwen3.6-35b-a3b must resolve so EP-divides-experts guardrail engages"
    assert preset["num_experts"] == 128
    assert preset["kv_heads"] is not None


def test_collect_xorl_metrics_parse_summary_includes_peak_mem(tmp_path):
    """Regression: producer + parse_summary used to silently drop peak_mem_gb."""
    collector = _load_script("collect_xorl_metrics")
    summary_path = tmp_path / "run-a" / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "output_dir": str(summary_path.parent),
                "measured_metrics": {
                    "tokens_per_sec": {"mean": 200.0, "max": 300.0},
                    "tflops_per_gpu": {"mean": 25.0},
                    "step_time_sec": {"mean": 1.0},
                },
                "measured_steps": [
                    {"step": 2, "tokens_per_sec": 200.0, "peak_mem_gb": 41.0},
                    {"step": 3, "tokens_per_sec": 300.0, "peak_mem_gb": 42.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    row = collector.parse_summary(summary_path)

    assert row.steps == 2
    assert row.tokens_per_sec_mean == pytest.approx(200.0)
    assert row.peak_mem_gb_max == pytest.approx(42.0)
    assert row.status == "ok"


def test_collect_xorl_metrics_discover_keeps_log_only_siblings(tmp_path):
    """Regression: discover() used to drop log-only runs when any summary existed."""
    collector = _load_script("collect_xorl_metrics")

    run_with_summary = tmp_path / "run-a"
    run_with_summary.mkdir()
    (run_with_summary / "benchmark_summary.json").write_text("{}", encoding="utf-8")
    (run_with_summary / "benchmark_stdout.log").write_text("", encoding="utf-8")

    run_log_only = tmp_path / "run-b"
    run_log_only.mkdir()
    (run_log_only / "benchmark_stdout.log").write_text("", encoding="utf-8")

    discovered = collector.discover([tmp_path])

    discovered_names = {p.name for p in discovered}
    discovered_parents = {p.parent.name for p in discovered}

    assert "benchmark_summary.json" in discovered_names
    assert "benchmark_stdout.log" in discovered_names
    assert "run-b" in discovered_parents
    summary_dir_logs = [p for p in discovered if p.parent == run_with_summary and p.suffix == ".log"]
    assert summary_dir_logs == [], "log alongside benchmark_summary.json should be skipped"


def test_render_k8s_manifest_no_privileged_drops_security_context():
    renderer = _load_script("render_k8s_torchrun_jobs")
    manifest = renderer.render(_render_args(privileged=False))
    assert "privileged: true" not in manifest
    assert "securityContext:" not in manifest


def test_render_k8s_manifest_privileged_is_opt_in():
    renderer = _load_script("render_k8s_torchrun_jobs")
    manifest = renderer.render(_render_args(privileged=True))
    assert "securityContext:" in manifest
    assert "privileged: true" in manifest


def test_throughput_skill_routes_correctness_to_k3_skill():
    text = (_SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")

    for expected in (
        "xorl-k3-correctness-check",
        "experiments/local_benchmark/k3_gate.py",
        "A missing K3 artifact is incomplete",
        "Qwen3.6 trace bundle is not valid evidence",
    ):
        assert expected in text
