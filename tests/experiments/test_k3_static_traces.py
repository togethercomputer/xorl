import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from xorl.models.layers.moe.experts import (
    _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS,
    _acquire_deepep_parity_diagnostic_record,
    _diff_summary,
    _dispatch_context_summary,
    _evenly_spaced_int64_indices,
    _safe_expert_output_reference_comparison,
    _safe_result_reference_comparison,
)
from xorl.ops.moe.activations import apply_moe_activation
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.runner.model_runner import ModelRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
K3_DIR = REPO_ROOT / "experiments" / "k3_tests"
if str(K3_DIR) not in sys.path:
    sys.path.insert(0, str(K3_DIR))


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_module(name: str):
    path = K3_DIR / f"{name}.py"
    if not path.exists():  # pragma: no cover - experiments/ excluded from src+tests merges
        pytest.skip(
            f"k3 experiment harness script '{name}.py' is absent from this tree "
            "(experiments/k3_tests/ is excluded from the src+tests merge scope); the static-trace "
            "parity test depends on it",
            allow_module_level=True,
        )
    return _load_module_from_path(name, path)


static_trace_utils = _load_module("static_trace_utils")
compare_logprobs = _load_module("compare_logprobs")
compare_static_traces = _load_module("compare_static_traces")
compare_hidden_component_artifacts = _load_module("compare_hidden_component_artifacts")
diagnose_component_source_leaderboard = _load_module("diagnose_component_source_leaderboard")
sglang_debug_dump_to_artifact = _load_module("sglang_debug_dump_to_artifact")
sglang_return_hidden_to_artifact = _load_module("sglang_return_hidden_to_artifact")
diagnose_final_norm_boundary = _load_module("diagnose_final_norm_boundary")
compare_component_tensor_dumps = _load_module("compare_component_tensor_dumps")
diagnose_tensor_source_terms = _load_module("diagnose_tensor_source_terms")
diagnose_tensor_input_sensitivity = _load_module("diagnose_tensor_input_sensitivity")
diagnose_tensor_rmsnorm_amplification = _load_module("diagnose_tensor_rmsnorm_amplification")
diagnose_tensor_layer_chain = _load_module("diagnose_tensor_layer_chain")
diagnose_qwen36_gdn_input_sensitivity = _load_module("diagnose_qwen36_gdn_input_sensitivity")
diagnose_qwen36_full_attention_sensitivity = _load_module("diagnose_qwen36_full_attention_sensitivity")
prepare_sglang_component_tensor_dump = _load_module("prepare_sglang_component_tensor_dump")
slime_megatron_activation_hook = _load_module("slime_megatron_activation_hook")
static_trace_to_slime_rollout = _load_module("static_trace_to_slime_rollout")
diagnose_sglang_post_attention_norm = _load_module("diagnose_sglang_post_attention_norm")
diagnose_component_norm_boundary = _load_module("diagnose_component_norm_boundary")
diagnose_component_residual_sources = _load_module("diagnose_component_residual_sources")
diagnose_component_window_bridge = _load_module("diagnose_component_window_bridge")
diagnose_residual_delta_flow = _load_module("diagnose_residual_delta_flow")
diagnose_residual_delta_flow_windows = _load_module("diagnose_residual_delta_flow_windows")
diagnose_component_three_way = _load_module("diagnose_component_three_way")
diagnose_shared_expert_gate = _load_module("diagnose_shared_expert_gate")
diagnose_shared_expert_input_sources = _load_module("diagnose_shared_expert_input_sources")
diagnose_shared_expert_raw_compute = _load_module("diagnose_shared_expert_raw_compute")
diagnose_shared_expert_reference_input = _load_module("diagnose_shared_expert_reference_input")
diagnose_tensor_shared_expert_reference_input = _load_module("diagnose_tensor_shared_expert_reference_input")
diagnose_tensor_moe_routing = _load_module("diagnose_tensor_moe_routing")
diagnose_tensor_post_attention_routing_sensitivity = _load_module("diagnose_tensor_post_attention_routing_sensitivity")
diagnose_residual_add_policy = _load_module("diagnose_residual_add_policy")
compare_qwen36_gdn_parity = _load_module("compare_qwen36_gdn_parity")
diagnose_sglang_gdn_rank_local_closure = _load_module("diagnose_sglang_gdn_rank_local_closure")
diagnose_static_k3 = _load_module("diagnose_static_k3")
extract_k3_repro_traces = _load_module("extract_k3_repro_traces")
slice_static_trace_token = _load_module("slice_static_trace_token")
make_static_traces = _load_module("make_static_traces")
refresh_static_traces = _load_module("refresh_static_traces")
parse_deepep_parity_diagnostics = _load_module("parse_deepep_parity_diagnostics")
launch_k3_test = _load_module("launch_k3_test")
launch_slime_megatron_trace = _load_module("launch_slime_megatron_trace")
fp8_sync_logprob_gate = _load_module_from_path(
    "fp8_sync_logprob_gate",
    REPO_ROOT / "scripts" / "fp8_sync_logprob_gate.py",
)


def test_launch_k3_default_namespace_matches_cluster_team_namespace():
    assert launch_k3_test.NAMESPACE == os.environ.get("K8S_NAMESPACE", "apanda")


def test_launch_k3_default_pvc_matches_cluster_home_pvc(monkeypatch):
    monkeypatch.delenv("K8S_PVC_NAME", raising=False)
    assert launch_k3_test.default_pvc_name() == "home-apanda"

    monkeypatch.setenv("K8S_PVC_NAME", "custom-pvc")
    assert launch_k3_test.default_pvc_name() == "custom-pvc"


def test_launch_slime_megatron_trace_defaults_deepep_to_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch_slime_megatron_trace.py"])
    args = launch_slime_megatron_trace.parse_args()

    assert args.enable_deepep is False
    assert args.image == launch_slime_megatron_trace.DEFAULT_IMAGE
    assert args.sglang_repo == launch_slime_megatron_trace.DEFAULT_SGLANG_REPO


def test_launch_slime_megatron_trace_can_select_slime_image_installed_sglang(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_slime_megatron_trace.py", "--use-slime-image", "--use-installed-sglang"],
    )
    args = launch_slime_megatron_trace.parse_args()

    assert args.image == launch_slime_megatron_trace.DEFAULT_SLIME_IMAGE
    assert args.sglang_repo == ""


def test_launch_slime_megatron_trace_rejects_cuda_visible_devices_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch_slime_megatron_trace.py", "--cuda-visible-devices", "0"],
    )

    with pytest.raises(SystemExit):
        launch_slime_megatron_trace.parse_args()


def test_launch_slime_megatron_trace_manifest_wires_hook_and_turbo_label():
    args = SimpleNamespace(
        pod_name="slime-q36-hook-test",
        node_name=None,
        home_dir="/home/apanda",
        slime_repo="/home/apanda/slime",
        sglang_repo="/home/apanda/xorl-sglang-internal/python",
        flashqla_repo="/home/apanda/FlashQLA",
        xorl_repo="/home/apanda/xorl-slime-parity-low-precision",
        megatron_repo="/shared/qywu/WorkingProjects/Megatron-LM",
        hf_checkpoint="/shared/huggingface/qwen",
        ref_load="/home/apanda/slime_megatron_artifacts/qwen-td",
        trace_id="trace-a",
        num_gpus=8,
        qwen_gdn_backend="flashqla",
        model_type="qwen3.5-35B-A3B",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        max_tokens_per_gpu=8192,
        sequence_parallel=True,
        install_deps=True,
        install_full_requirements=False,
        convert_if_missing=True,
        enable_deepep=True,
        slime_extra_args="--mock-flag value",
        component_layers="10-11",
        hidden_sample_indices="all",
        capture_ranks="0",
        pvc_name="home-apanda",
    )

    manifest = launch_slime_megatron_trace.render_slime_manifest(
        args,
        trace_configmap_name="slime-q36-hook-test-trace",
        run_dir="/home/apanda/slime_megatron_artifacts/slime-q36-hook-test",
    )

    assert "team: turbo" in manifest
    assert "privileged: true" not in manifest
    assert "export CUDA_VISIBLE_DEVICES=" not in manifest
    assert 'nvidia.com/gpu: "8"' in manifest
    assert "claimName: home-apanda" in manifest
    assert "name: slime-q36-hook-test-trace" in manifest
    assert "mountPath: /trace-input" in manifest
    assert "numpy<2" in manifest
    assert "SGLANG_REPO=/home/apanda/xorl-sglang-internal/python" in manifest
    assert "FLASHQLA_REPO=/home/apanda/FlashQLA" in manifest
    assert "REPLAY_BATCH_SIZE=2" in manifest
    assert "QWEN_GDN_BACKEND=flashqla" in manifest
    assert (
        "export PYTHONPATH=/home/apanda/FlashQLA:/home/apanda/xorl-sglang-internal/python:/shared/qywu/WorkingProjects/Megatron-LM:/home/apanda/slime"
        in manifest
    )
    assert '"SGLANG_REPO"' in manifest
    assert '"FLASHQLA_REPO"' in manifest
    assert '"sglang"' in manifest
    assert '"flash_qla"' in manifest
    assert "tilelang==0.1.8" in manifest
    assert "openai==2.6.1" in manifest
    assert "partial-json-parser" in manifest
    assert "pybase64" in manifest
    assert "sentencepiece" in manifest
    assert "torch_memory_saver==0.0.9" in manifest
    assert "import slime.ray.rollout" in manifest
    assert "import slime.backends.sglang_utils.sglang_engine" in manifest
    assert "import slime.backends.megatron_utils.actor" in manifest
    assert "import sglang.srt.server_args" in manifest
    assert "slime/sglang recursive import preflight: OK" in manifest
    assert "mbridge" in manifest
    assert "github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" in manifest
    assert "github.com/fla-org/flash-linear-attention.git@9714c595" in manifest
    assert "CONVERT_MODEL_ARGS=()" in manifest
    assert '"${CONVERT_MODEL_ARGS[@]}"' in manifest
    assert '--target-sample-count "2"' in manifest
    assert "XORL_SLIME_MEGATRON_COMPONENT_LAYERS=10-11" in manifest
    assert "XORL_SLIME_MEGATRON_HIDDEN_SAMPLE_INDICES=all" in manifest
    assert "XORL_SLIME_MEGATRON_SKIP_ACTOR_RESTORE=1" in manifest
    assert "XORL_SLIME_MEGATRON_PATCH_TILELANG_PRELOWER_CHECK=1" in manifest
    assert "XORL_SLIME_MEGATRON_PATCH_TRITON_GET_INT_DTYPE=1" in manifest
    assert "XORL_SLIME_MEGATRON_TRITON_CACHE_DIR=" in manifest
    assert '--load-debug-rollout-data "${ROLLOUT_PT}"' in manifest
    assert (
        "--custom-megatron-init-path experiments.k3_tests.slime_megatron_activation_hook.patch_actor_restore_for_slime_replay"
        in manifest
    )
    assert '--qwen-gdn-backend "flashqla"' in manifest
    assert "--loss-type sft_loss" in manifest
    assert '--rollout-batch-size "2"' in manifest
    assert '--global-batch-size "2"' in manifest
    assert "--tensor-model-parallel-size 2" in manifest
    assert "--sequence-parallel" in manifest
    assert "--context-parallel-size 2" in manifest
    assert "--expert-model-parallel-size 8" in manifest
    assert "--max-tokens-per-gpu 8192" in manifest
    assert (
        "--custom-megatron-before-log-prob-hook-path experiments.k3_tests.slime_megatron_activation_hook.capture_megatron_logprob_components"
        in manifest
    )
    assert "train.py failed after activation artifact was written" in manifest
    assert "TRAIN_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep)" in manifest
    assert "CONVERT_IF_MISSING=1" in manifest
    assert "SLIME_EXTRA_ARGS='--mock-flag value'" in manifest


def test_launch_slime_megatron_trace_manifest_can_use_image_installed_sglang():
    args = SimpleNamespace(
        pod_name="slime-q36-hook-installed-sglang",
        image=launch_slime_megatron_trace.DEFAULT_SLIME_IMAGE,
        node_name=None,
        home_dir="/home/apanda",
        slime_repo="/home/apanda/slime",
        sglang_repo="",
        flashqla_repo="/home/apanda/FlashQLA",
        xorl_repo="/home/apanda/xorl-slime-parity-low-precision",
        megatron_repo="/shared/qywu/WorkingProjects/Megatron-LM",
        hf_checkpoint="/shared/huggingface/qwen",
        ref_load="/home/apanda/slime_megatron_artifacts/qwen-td",
        trace_id="trace-a",
        num_gpus=4,
        qwen_gdn_backend="fla",
        model_type="qwen3.5-35B-A3B",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        max_tokens_per_gpu=4096,
        sequence_parallel=False,
        install_deps=False,
        install_full_requirements=False,
        convert_if_missing=False,
        enable_deepep=False,
        slime_extra_args="",
        component_layers="10-11",
        hidden_sample_indices="all",
        capture_ranks="0",
        pvc_name="home-apanda",
    )

    manifest = launch_slime_megatron_trace.render_slime_manifest(
        args,
        trace_configmap_name="slime-q36-hook-installed-sglang-trace",
        run_dir="/home/apanda/slime_megatron_artifacts/slime-q36-hook-installed-sglang",
    )

    assert f"image: {launch_slime_megatron_trace.DEFAULT_SLIME_IMAGE}" in manifest
    assert "team: turbo" in manifest
    assert "privileged: true" not in manifest
    assert "export CUDA_VISIBLE_DEVICES=" not in manifest
    assert "export SGLANG_REPO=" in manifest
    assert "SGLANG_REPO=/home/apanda/xorl-sglang-internal/python" not in manifest
    assert "/home/apanda/xorl-sglang-internal/python" not in manifest
    assert (
        "export PYTHONPATH=/home/apanda/FlashQLA:/shared/qywu/WorkingProjects/Megatron-LM:/home/apanda/slime:"
        "/home/apanda/xorl-slime-parity-low-precision:/home/apanda/xorl-slime-parity-low-precision/experiments/k3_tests:"
        in manifest
    )
    assert 'echo "SGLANG_REPO=${SGLANG_REPO:-<installed>}"' in manifest


def test_launch_slime_megatron_trace_can_render_smaller_diagnostic_topology():
    args = SimpleNamespace(
        pod_name="slime-q36-hook-4gpu",
        node_name=None,
        home_dir="/home/apanda",
        slime_repo="/home/apanda/slime",
        sglang_repo="/home/apanda/xorl-sglang-internal/python",
        flashqla_repo="/home/apanda/FlashQLA",
        xorl_repo="/home/apanda/xorl-slime-parity-low-precision",
        megatron_repo="/shared/qywu/WorkingProjects/Megatron-LM",
        hf_checkpoint="/shared/huggingface/qwen",
        ref_load=None,
        trace_id="trace-a",
        num_gpus=4,
        qwen_gdn_backend="flashqla",
        model_type="qwen3.5-35B-A3B",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        max_tokens_per_gpu=4096,
        sequence_parallel=False,
        install_deps=False,
        install_full_requirements=False,
        convert_if_missing=False,
        enable_deepep=False,
        slime_extra_args="",
        component_layers="10-11",
        hidden_sample_indices="all",
        capture_ranks="0",
        pvc_name="home-apanda",
    )
    args.ref_load = launch_slime_megatron_trace.default_ref_load_for_topology(args)

    manifest = launch_slime_megatron_trace.render_slime_manifest(
        args,
        trace_configmap_name="slime-q36-hook-4gpu-trace",
        run_dir="/home/apanda/slime_megatron_artifacts/slime-q36-hook-4gpu",
    )

    assert args.ref_load.endswith("_tp1_pp1_cp1_ep4_etp1")
    assert 'nvidia.com/gpu: "4"' in manifest
    assert "team: turbo" in manifest
    assert "privileged: true" not in manifest
    assert "export CUDA_VISIBLE_DEVICES=" not in manifest
    assert "REPLAY_BATCH_SIZE=4" in manifest
    assert '--target-sample-count "4"' in manifest
    assert '--rollout-batch-size "4"' in manifest
    assert '--global-batch-size "4"' in manifest
    assert "--tensor-model-parallel-size 1" in manifest
    assert "--sequence-parallel" not in manifest
    assert "--context-parallel-size 1" in manifest
    assert "--expert-model-parallel-size 4" in manifest
    assert "--max-tokens-per-gpu 4096" in manifest
    assert "export ENABLE_DEEPEP=0" in manifest


def test_copy_pod_file_artifact_captures_profile_json(monkeypatch, tmp_path):
    calls = []

    class Result:
        returncode = 0
        stdout = '{"profile": true}\n'
        stderr = ""

    monkeypatch.setattr(launch_k3_test, "pod_exists", lambda name: name == "k3-xo-test")

    def fake_run(cmd, check=True, capture=False):
        calls.append((cmd, check, capture))
        return Result()

    monkeypatch.setattr(launch_k3_test, "run", fake_run)

    copied = launch_k3_test.copy_pod_file_artifact(
        "k3-xo-test",
        "/tmp/fp8-profile.json",
        tmp_path,
        artifact_name="profile.json",
    )

    assert copied == tmp_path / "profile.json"
    assert json.loads(copied.read_text(encoding="utf-8")) == {"profile": True}
    assert calls == [
        (
            "kubectl -n apanda exec k3-xo-test -- cat /tmp/fp8-profile.json",
            False,
            True,
        )
    ]


def test_tokenized_row_to_prompt_completion_uses_supervised_boundary_suffix():
    row = {
        "input_ids": list(range(20)),
        "labels": [-100] * 12 + list(range(12, 20)),
    }

    extracted = static_trace_utils.tokenized_row_to_prompt_completion(
        row,
        prompt_tokens=5,
        completion_tokens=3,
        prompt_window="suffix",
        min_prompt_tokens=2,
        use_dataset_completion=True,
    )

    assert extracted == ([7, 8, 9, 10, 11], [12, 13, 14])


def test_static_trace_json_round_trip(tmp_path):
    path = tmp_path / "traces.json"
    trace = {
        "trace_id": "t0",
        "prompt_ids": [1, 2, 3],
        "output_ids": [4, 5],
        "sglang_logprobs": [-0.1, -0.2],
    }

    static_trace_utils.write_static_trace_file(path, {"model_name": "m"}, [trace])
    metadata, traces = static_trace_utils.load_static_trace_file(path)
    normalized = static_trace_utils.normalize_trace(traces[0])

    assert metadata["model_name"] == "m"
    assert normalized["full_ids"] == [1, 2, 3, 4, 5]
    assert normalized["prompt_len"] == 3
    assert normalized["gen_len"] == 2
    assert static_trace_utils.xorl_input_ids_for_trace(normalized) == [1, 2, 3, 4]
    assert static_trace_utils.labels_for_trace(normalized) == [-100, -100, 4, 5]


def test_static_trace_preserves_optional_generation_logprobs(tmp_path):
    path = tmp_path / "traces.json"
    trace = {
        "trace_id": "t0",
        "prompt_ids": [1, 2],
        "output_ids": [3],
        "sglang_logprobs": [-0.3],
        "sglang_generation_logprobs": [-0.31],
    }

    static_trace_utils.write_static_trace_file(path, {}, [trace])
    _, traces = static_trace_utils.load_static_trace_file(path)
    normalized = static_trace_utils.normalize_trace(traces[0])

    assert normalized["sglang_generation_logprobs"] == [-0.31]


def test_static_trace_jsonl_round_trip(tmp_path):
    path = tmp_path / "traces.jsonl"
    trace = {
        "trace_id": "t0",
        "prompt_ids": [1],
        "output_ids": [2],
        "sglang_logprobs": [-1.0],
    }

    static_trace_utils.write_static_trace_file(path, {}, [trace])
    metadata, traces = static_trace_utils.load_static_trace_file(path)

    assert metadata["format"] == "jsonl"
    assert len(traces) == 1
    assert static_trace_utils.normalize_trace(traces[0])["full_ids"] == [1, 2]


def test_filter_traces_by_id_preserves_requested_order():
    traces = [{"trace_id": "a"}, {"trace_id": "b"}, {"trace_id": "c"}]

    filtered = static_trace_utils.filter_traces_by_id(traces, ["c", "a"])

    assert [trace["trace_id"] for trace in filtered] == ["c", "a"]


def test_static_trace_to_slime_rollout_writes_replay_payload(tmp_path):
    traces_path = tmp_path / "traces.json"
    output_pt = tmp_path / "rollout_0.pt"
    static_trace_utils.write_static_trace_file(
        traces_path,
        {"model_name": "Qwen/Qwen3.6-35B-A3B"},
        [
            {
                "trace_id": "trace-a",
                "prompt_ids": [10, 20, 30],
                "output_ids": [40, 50],
                "sglang_logprobs": [-0.4, -0.5],
            },
            {
                "trace_id": "trace-b",
                "prompt_ids": [1],
                "output_ids": [2],
                "sglang_logprobs": [-0.2],
            },
        ],
    )

    metadata, traces = static_trace_utils.load_static_trace_file(traces_path)
    assert metadata["model_name"] == "Qwen/Qwen3.6-35B-A3B"
    dump = static_trace_to_slime_rollout.build_slime_rollout_dump(
        traces=static_trace_utils.filter_traces_by_id(traces, ["trace-a"]),
        rollout_id=7,
        reward=1.5,
        target_only=True,
    )
    torch.save(dump, output_pt)

    loaded = torch.load(output_pt, weights_only=False)
    sample = loaded["samples"][0]
    assert loaded["rollout_id"] == 7
    assert sample["tokens"] == [10, 20, 30, 40, 50]
    assert sample["response_length"] == 2
    assert sample["loss_mask"] == [0, 1]
    assert sample["reward"] == pytest.approx(1.5)
    assert sample["status"] == "completed"
    assert sample["metadata"]["trace_id"] == "trace-a"
    assert sample["metadata"]["sglang_logprobs"] == pytest.approx([-0.4, -0.5])


def test_static_trace_to_slime_rollout_can_repeat_trace_for_dp_batch():
    dump = static_trace_to_slime_rollout.build_slime_rollout_dump(
        traces=[
            {
                "trace_id": "trace-a",
                "prompt_ids": [10, 20],
                "output_ids": [30],
                "sglang_logprobs": [-0.4],
            }
        ],
        target_sample_count=4,
    )

    assert [sample["index"] for sample in dump["samples"]] == [0, 1, 2, 3]
    assert [sample["group_id"] for sample in dump["samples"]] == [0, 1, 2, 3]
    assert [sample["metadata"]["trace_id"] for sample in dump["samples"]] == ["trace-a"] * 4
    assert all(sample["tokens"] == [10, 20, 30] for sample in dump["samples"])


def test_live_compare_sample_preserves_static_trace_fields(monkeypatch):
    class _Tokenizer:
        vocab_size = 100

        def encode(self, text: str) -> list[int]:
            assert text == "prompt"
            return [10, 20]

    captured = {}
    monkeypatch.setattr(
        compare_logprobs,
        "sglang_generate",
        lambda *args, **kwargs: {
            "output_ids": [30, 40],
            "text": "generated",
            "meta_info": {"output_token_logprobs": [[-0.15, 30, "a"], [-0.25, 40, "b"]]},
        },
    )
    monkeypatch.setattr(compare_logprobs, "sglang_score", lambda *args, **kwargs: ([-0.1, -0.2], None, None))

    def fake_xorl_forward(xorl_url, input_ids, labels, *args, **kwargs):
        captured["input_ids"] = input_ids
        captured["labels"] = labels
        return {"ok": True}

    monkeypatch.setattr(compare_logprobs, "xorl_forward", fake_xorl_forward)
    monkeypatch.setattr(compare_logprobs, "extract_xorl_logprobs", lambda result, gen_len: [-0.1, -0.2])

    result = compare_logprobs.process_prompt(
        "prompt",
        0,
        1,
        _Tokenizer(),
        "http://sglang",
        "http://xorl",
        "default",
        max_new_tokens=2,
        top_logprobs_num=0,
        timeout=1.0,
    )

    assert result["trace_id"] == "prompt-00000"
    assert result["trace_mode"] == "sglang_generation"
    assert result["prompt_text"] == "prompt"
    assert result["prompt_ids"] == [10, 20]
    assert result["output_ids"] == [30, 40]
    assert result["full_ids"] == [10, 20, 30, 40]
    assert result["sglang_logprobs"] == [-0.1, -0.2]
    assert result["sglang_generation_logprobs"] == [-0.15, -0.25]
    assert captured["input_ids"] == [10, 20, 30]
    assert captured["labels"] == [-100, 30, 40]
    static_trace_utils.normalize_trace(result)


class _RoutingResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_sglang_score_captures_routed_experts_when_requested(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):  # noqa: A002 - matches requests.post signature
        captured["json"] = json
        return _RoutingResp(
            {
                "meta_info": {
                    "input_token_logprobs": [[-0.5, 7, "x"], [-0.1, 8, "y"]],
                    "routed_experts": "BASE64ROUTING",
                }
            }
        )

    monkeypatch.setattr(compare_logprobs.requests, "post", fake_post)

    lps, top, routed = compare_logprobs.sglang_score(
        "http://sg", [1, 2, 3], gen_len=1, top_logprobs_num=0, return_routed_experts=True
    )

    assert lps == [-0.1]
    assert top is None
    assert routed == "BASE64ROUTING"
    assert captured["json"]["return_routed_experts"] is True
    assert captured["json"]["routed_experts_start_len"] == 0


def test_sglang_score_omits_routing_request_by_default(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):  # noqa: A002
        captured["json"] = json
        return _RoutingResp({"meta_info": {"input_token_logprobs": [[-0.1, 8, "y"]]}})

    monkeypatch.setattr(compare_logprobs.requests, "post", fake_post)

    lps, top, routed = compare_logprobs.sglang_score("http://sg", [1, 2], gen_len=1)

    assert routed is None
    assert "return_routed_experts" not in captured["json"]
    assert "routed_experts_start_len" not in captured["json"]


def test_xorl_forward_injects_routed_experts(monkeypatch):
    bodies = []

    def fake_post(url, json, timeout):  # noqa: A002
        bodies.append((url, json))
        if url.endswith("/api/v1/forward"):
            return _RoutingResp({"request_id": "r1"})
        return _RoutingResp({"result": "ok"})

    monkeypatch.setattr(compare_logprobs.requests, "post", fake_post)

    routing = [{"data": "B64", "shape": [3, 2, 8]}]
    result = compare_logprobs.xorl_forward("http://xo", [1, 2, 3], [-100, 2, 3], routed_experts=routing)

    assert result == {"result": "ok"}
    submit_body = next(body for url, body in bodies if url.endswith("/api/v1/forward"))
    assert submit_body["forward_input"]["routed_experts"] == routing


def test_xorl_forward_omits_routed_experts_when_none(monkeypatch):
    bodies = []

    def fake_post(url, json, timeout):  # noqa: A002
        bodies.append((url, json))
        if url.endswith("/api/v1/forward"):
            return _RoutingResp({"request_id": "r1"})
        return _RoutingResp({"result": "ok"})

    monkeypatch.setattr(compare_logprobs.requests, "post", fake_post)

    compare_logprobs.xorl_forward("http://xo", [1, 2, 3], [-100, 2, 3])

    submit_body = next(body for url, body in bodies if url.endswith("/api/v1/forward"))
    assert "routed_experts" not in submit_body["forward_input"]


def test_refresh_trace_captures_routed_experts(monkeypatch):
    trace = {"trace_id": "t0", "prompt_ids": [1, 2], "output_ids": [3], "sglang_logprobs": [-0.5]}

    def fake_score(*args, return_routed_experts=False, **kwargs):
        return ([-0.4], None, "ROUTING_B64" if return_routed_experts else None)

    monkeypatch.setattr(refresh_static_traces, "sglang_score", fake_score)

    refreshed = refresh_static_traces.refresh_trace(
        trace,
        sglang_url="http://sg",
        max_new_tokens=None,
        top_logprobs_num=0,
        regenerate_outputs=False,
        check_generation=False,
        capture_routing=True,
    )

    assert refreshed["routed_experts"] == "ROUTING_B64"


def test_score_trace_with_xorl_replays_routing(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-0.3],
            "routed_experts": "ROUTING_B64",
        }
    )
    captured = {}

    def fake_xorl_forward(*args, **kwargs):
        captured["routed_experts"] = kwargs.get("routed_experts")
        return {"ok": True}

    monkeypatch.setattr(compare_static_traces, "xorl_forward", fake_xorl_forward)
    monkeypatch.setattr(compare_static_traces, "extract_xorl_logprobs", lambda result, gen_len: [-0.2])

    compare_static_traces.score_trace_with_xorl(
        trace,
        xorl_url="http://xo",
        xorl_model_id="default",
        timeout=1.0,
        local_forward=False,
        local_device="cpu",
        model_name="",
        reference_logprobs="prefill",
        replay_routing=True,
    )

    assert captured["routed_experts"] == ["ROUTING_B64"]


def test_score_trace_with_xorl_replay_routing_requires_capture(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {"trace_id": "t0", "prompt_ids": [1, 2], "output_ids": [3], "sglang_logprobs": [-0.3]}
    )
    monkeypatch.setattr(compare_static_traces, "xorl_forward", lambda *args, **kwargs: {"ok": True})

    with pytest.raises(ValueError, match="no routed_experts"):
        compare_static_traces.score_trace_with_xorl(
            trace,
            xorl_url="http://xo",
            xorl_model_id="default",
            timeout=1.0,
            local_forward=False,
            local_device="cpu",
            model_name="",
            reference_logprobs="prefill",
            replay_routing=True,
        )


def test_aggregate_results_reports_selection_flip_tail():
    results = [
        {
            "gen_len": 4,
            "sample_k3_mean": 0.05,
            "per_token": [{"k3": 0.01}, {"k3": 0.02}, {"k3": 0.9}, {"k3": 0.03}],
        }
    ]

    agg = compare_logprobs.aggregate_results(results, flip_threshold=0.1)

    flip = agg["selection_flip"]
    assert flip["threshold"] == 0.1
    assert flip["flip_count"] == 1
    assert flip["flip_rate"] == 0.25
    assert flip["non_flip_count"] == 3
    assert abs(flip["non_flip_mean_k3"] - 0.02) < 1e-9
    # median is robust to the single flip token (0.9): (0.02 + 0.03) / 2
    assert abs(agg["k3"]["median"] - 0.025) < 1e-9


def test_threshold_failed_median_gate_is_robust_to_flip_tail():
    agg = {
        "k3": {"mean": 0.5, "median": 0.02, "p95": 0.9},
        "selection_flip": {"flip_rate": 0.05, "threshold": 0.1},
    }
    # Median passes even though raw mean (flip-tail-dominated) is high.
    assert compare_static_traces._threshold_failed(agg, None, None, max_median_k3=0.03) is None
    msg = compare_static_traces._threshold_failed(agg, None, None, max_median_k3=0.01)
    assert msg is not None and "median K3" in msg


def test_threshold_failed_flip_rate_gate():
    agg = {"k3": {"mean": 0.5, "median": 0.02}, "selection_flip": {"flip_rate": 0.08, "threshold": 0.1}}
    assert compare_static_traces._threshold_failed(agg, None, None, max_flip_rate=0.1) is None
    msg = compare_static_traces._threshold_failed(agg, None, None, max_flip_rate=0.05)
    assert msg is not None and "flip_rate" in msg


def _routing_capture_tokenizer():
    class _Tok:
        vocab_size = 100

        def encode(self, text):
            return [10, 20]

    return _Tok()


def _patch_process_prompt_deps(monkeypatch, routed_value):
    monkeypatch.setattr(
        compare_logprobs,
        "sglang_generate",
        lambda *a, **k: {
            "output_ids": [30, 40],
            "text": "g",
            "meta_info": {"output_token_logprobs": [[-0.1, 30, "a"], [-0.2, 40, "b"]]},
        },
    )

    def fake_score(url, full_ids, gen_len, top_logprobs_num=0, return_routed_experts=False):
        return ([-0.1, -0.2], None, (routed_value if return_routed_experts else None))

    monkeypatch.setattr(compare_logprobs, "sglang_score", fake_score)
    monkeypatch.setattr(compare_logprobs, "xorl_forward", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(compare_logprobs, "extract_xorl_logprobs", lambda r, g: [-0.1, -0.2])


def test_process_prompt_captures_routing_when_enabled(monkeypatch):
    _patch_process_prompt_deps(monkeypatch, "ROUTE_B64")

    sample = compare_logprobs.process_prompt(
        "p",
        0,
        1,
        _routing_capture_tokenizer(),
        "http://s",
        "http://x",
        "default",
        max_new_tokens=2,
        top_logprobs_num=0,
        timeout=1.0,
        capture_routing=True,
    )

    assert sample["routed_experts"] == "ROUTE_B64"
    trace = compare_logprobs._static_trace_from_sample(sample)
    assert trace["routed_experts"] == "ROUTE_B64"


def test_process_prompt_omits_routing_by_default(monkeypatch):
    _patch_process_prompt_deps(monkeypatch, "ROUTE_B64")

    sample = compare_logprobs.process_prompt(
        "p",
        0,
        1,
        _routing_capture_tokenizer(),
        "http://s",
        "http://x",
        "default",
        max_new_tokens=2,
        top_logprobs_num=0,
        timeout=1.0,
    )

    assert "routed_experts" not in sample
    trace = compare_logprobs._static_trace_from_sample(sample)
    assert "routed_experts" not in trace


def test_compare_logprobs_writes_static_trace_bundle(tmp_path):
    output = tmp_path / "static_traces.json"
    sample = {
        "trace_id": "prompt-00000",
        "trace_mode": "sglang_generation",
        "prompt_text": "prompt",
        "prompt_ids": [10, 20],
        "output_ids": [30],
        "full_ids": [10, 20, 30],
        "prompt_len": 2,
        "gen_len": 1,
        "generated_text": "generated",
        "sglang_logprobs": [-0.1],
        "sglang_generation_logprobs": [-0.15],
        "per_token": [
            {
                "position": 0,
                "token_id": 30,
                "sglang_logprob": -0.1,
                "sglang_top_logprobs": [[-0.1, 30, "generated"]],
            }
        ],
    }

    compare_logprobs.write_static_traces_from_results(
        output,
        {"model_name": "Qwen/Test", "sglang_url": "http://sg", "max_new_tokens": 1, "top_logprobs_num": 1},
        [sample],
    )
    metadata, traces = static_trace_utils.load_static_trace_file(output)
    normalized = static_trace_utils.normalize_trace(traces[0])

    assert metadata["model_name"] == "Qwen/Test"
    assert metadata["source"] == "compare_logprobs"
    assert normalized["trace_mode"] == "sglang_generation"
    assert normalized["full_ids"] == [10, 20, 30]
    assert normalized["sglang_generation_logprobs"] == [-0.15]
    assert normalized["sglang_top_logprobs"][0][0][1] == 30


def test_score_trace_with_xorl_computes_zero_k3_for_matching_logprobs(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3, 4],
            "sglang_logprobs": [-0.3, -0.4],
        }
    )

    captured = {}

    def fake_xorl_forward(xorl_url, input_ids, labels, **kwargs):
        captured["input_ids"] = input_ids
        captured["labels"] = labels
        return {"ok": True}

    monkeypatch.setattr(compare_static_traces, "xorl_forward", fake_xorl_forward)
    monkeypatch.setattr(compare_static_traces, "extract_xorl_logprobs", lambda result, gen_len: [-0.3, -0.4])

    result = compare_static_traces.score_trace_with_xorl(
        trace,
        xorl_url="http://xorl",
        xorl_model_id="default",
        timeout=1.0,
        local_forward=False,
        local_device="cpu",
        model_name="",
        reference_logprobs="auto",
    )

    assert result["sample_k3_mean"] == pytest.approx(0.0)
    assert result["per_token"][0]["xorl_logprob"] == pytest.approx(-0.3)
    assert captured["input_ids"] == [1, 2, 3]
    assert captured["labels"] == [-100, 3, 4]


def test_fp8_sync_gate_records_pass_artifact(monkeypatch, tmp_path):
    score_calls = []

    def fake_score(url, prompts, *, max_new_tokens, timeout, **kwargs):
        score_calls.append(
            {
                "url": url,
                "prompts": prompts,
                "max_new_tokens": max_new_tokens,
                "timeout": timeout,
            }
        )
        return [[(-1.0, 10), (-0.25, 11)]]

    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_generated_logprobs", fake_score)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=2,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-test",
        buffer_size_mb=128,
        weight_version="v1",
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    output_path = tmp_path / "fp8-sync.json"
    fp8_sync_logprob_gate._write_output_json(str(output_path), result)
    written = json.loads(output_path.read_text())

    assert result["status"] == "pass"
    assert result["comparison"]["compared"] == 2
    assert result["config"]["max_new_tokens"] == 2
    assert result["config"]["flush_cache"] is True
    assert result["sync_payload"]["master_address"] == "10.0.0.1"
    assert result["sync_payload"]["weight_version"] == "v1"
    assert result["sync_payload"]["flush_cache"] is True
    assert len(score_calls) == 2
    assert all(call["max_new_tokens"] == 2 for call in score_calls)
    assert any(call[0].endswith("/api/v1/set_sync_quantization") for call in post_calls)
    assert written["status"] == "pass"


def test_fp8_sync_gate_scores_token_id_prompts(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        return {
            "meta_info": {
                "output_token_logprobs": [
                    {"logprob": -0.5, "token_id": 7},
                ],
            },
        }

    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    scores = fp8_sync_logprob_gate._score_generated_logprobs(
        "http://sglang",
        [[1, 2, 3]],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.5, 7)]]
    assert post_calls[0][0] == "http://sglang/generate"
    assert post_calls[0][1]["input_ids"] == [1, 2, 3]
    assert "text" not in post_calls[0][1]


def test_fp8_sync_gate_scores_generated_placeholder_from_top_logprobs(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        return {
            "meta_info": {
                "output_token_logprobs": [
                    [None, 0, None],
                ],
                "output_top_logprobs": [
                    [
                        [-0.125, 0, None],
                        [-2.5, 1, None],
                    ],
                ],
            },
        }

    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    scores = fp8_sync_logprob_gate._score_generated_logprobs(
        "http://sglang",
        ["The capital of France is"],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.125, 0)]]
    assert post_calls[0][0] == "http://sglang/generate"
    assert post_calls[0][1]["top_logprobs_num"] == 5


def test_fp8_sync_gate_scores_generated_placeholder_from_first_top_logprob(monkeypatch):
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_post_json",
        lambda *args, **kwargs: {
            "meta_info": {
                "output_token_logprobs": [
                    [None, 0, None],
                ],
                "output_top_logprobs": [
                    [
                        [-0.25, 42, None],
                        [-1.5, 43, None],
                    ],
                ],
            },
        },
    )

    scores = fp8_sync_logprob_gate._score_generated_logprobs(
        "http://sglang",
        ["The capital of France is"],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.25, 42)]]


def test_fp8_sync_gate_scores_fixed_input_logprobs(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        return {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 1, None],
                    [-0.25, 0, ""],
                    [-0.5, 1, ""],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [[-0.25, 0, ""]],
                    [[-0.75, 0, ""]],
                ],
            },
        }

    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    scores = fp8_sync_logprob_gate._score_input_logprobs(
        "http://sglang",
        [[1, 1, 1, 1, 0]],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.25, 0)]]
    assert post_calls[0][0] == "http://sglang/generate"
    assert post_calls[0][1]["input_ids"] == [1, 1, 1, 1, 0, 1]
    assert post_calls[0][1]["sampling_params"]["max_new_tokens"] == 1
    assert post_calls[0][1]["logprob_start_len"] == 3
    assert post_calls[0][1]["token_ids_logprob"] == [0]


def test_fp8_sync_gate_scores_input_logprobs_from_exact_target_position(monkeypatch):
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_post_json",
        lambda *args, **kwargs: {
            "meta_info": {
                "input_token_ids_logprobs": [
                    None,
                    [[-0.25, 0, None]],
                    [[-0.9, 0, None]],
                ],
            },
        },
    )

    scores = fp8_sync_logprob_gate._score_input_logprobs(
        "http://sglang",
        [[1, 1, 1, 1, 0]],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.25, 0)]]


def test_fp8_sync_gate_prefers_positional_input_logprobs_for_target_token(monkeypatch):
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_post_json",
        lambda *args, **kwargs: {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 1, None],
                    [-0.25, 0, None],
                    [-0.5, 1, None],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [[-9.0, 0, None]],
                    [[-8.0, 0, None]],
                ],
            },
        },
    )

    scores = fp8_sync_logprob_gate._score_input_logprobs(
        "http://sglang",
        [[1, 1, 1, 1, 0]],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.25, 0)]]


def test_fp8_sync_gate_falls_back_to_requested_input_logprob_for_placeholder_position(monkeypatch):
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_post_json",
        lambda *args, **kwargs: {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 1, None],
                    [None, 5, None],
                    [-0.5, 1, None],
                ],
                "input_token_ids_logprobs": [
                    None,
                    [[-0.125, 5, None]],
                    [[-0.9, 5, None]],
                ],
            },
        },
    )

    scores = fp8_sync_logprob_gate._score_input_logprobs(
        "http://sglang",
        [[1, 2, 3, 4, 5]],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.125, 5)]]


def test_fp8_sync_gate_falls_back_to_nearest_requested_input_logprob_when_position_shifted(monkeypatch):
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_post_json",
        lambda *args, **kwargs: {
            "meta_info": {
                "input_token_logprobs": [
                    [None, 4, None],
                    [None, 0, None],
                    [None, 0, None],
                ],
                "input_token_ids_logprobs": [
                    [[-0.25, 5, None]],
                    [[-9.0, 4, None]],
                    [[-0.5, 5, None]],
                ],
            },
        },
    )

    scores = fp8_sync_logprob_gate._score_input_logprobs(
        "http://sglang",
        [[1, 2, 3, 4, 5]],
        max_new_tokens=1,
        timeout=2.0,
    )

    assert scores == [[(-0.25, 5)]]


def test_fp8_sync_gate_flushes_before_target_token_scoring(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        if url.endswith("/generate"):
            return {
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 1, None],
                        [-0.25, 0, ""],
                        [-0.5, 1, ""],
                    ],
                    "input_token_ids_logprobs": [
                        None,
                        [[-0.25, 0, ""]],
                        [[-0.75, 0, ""]],
                    ],
                },
            }
        return {"success": True}

    flush_calls = []

    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_flush_sglang_cache",
        lambda url, *, timeout: flush_calls.append((url, timeout)),
    )

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-target-score",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "pass"
    assert result["config"]["flush_cache_before_score"] is True
    assert flush_calls == [("http://sglang", 3.0)] * 4


def test_fp8_sync_gate_none_quantization_sends_null_default(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        return {"success": True}

    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: [[(-1.0, 10)]],
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="bf16-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="none",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    set_sync_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/set_sync_quantization"))
    sync_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/sync_inference_weights"))

    assert result["status"] == "pass"
    assert result["config"]["sync_quantization"] == "none"
    assert result["config"]["quantization"] is None
    assert set_sync_payload == {"quantization": None}
    assert sync_payload["flush_cache"] is True
    assert "quantization" not in sync_payload


def test_fp8_sync_gate_qarl_quantization_uses_server_derived_config(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        return {"success": True}

    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: [[(-1.0, 10)]],
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="qarl-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="qarl",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    set_sync_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/set_sync_quantization"))
    sync_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/sync_inference_weights"))

    assert result["status"] == "pass"
    assert result["config"]["sync_quantization"] == "qarl"
    assert result["config"]["qarl_sync"] is True
    assert result["config"]["quantization"] is None
    assert set_sync_payload == {"quantization": None}
    assert sync_payload["flush_cache"] is True
    assert "quantization" not in sync_payload


def test_fp8_sync_gate_fp8_kv_cache_gate_records_epoch_change(monkeypatch):
    server_info_sequence = [
        {
            "model_path": "Qwen/Qwen3-0.6B-FP8",
            "quantization": "fp8",
            "kv_cache_dtype": "fp8_e4m3",
            "requires_fp8_kv_cache_postprocess": "true",
            "kv_cache_static_scales": "1",
            "cache_epoch": 7,
        },
        {
            "model_path": "Qwen/Qwen3-0.6B-FP8",
            "quantization": "fp8",
            "kv_cache_dtype": "fp8_e4m3",
            "requires_fp8_kv_cache_postprocess": "true",
            "kv_cache_static_scales": "1",
            "cache_epoch": 8,
        },
    ]

    def fake_get_json(url, *, timeout):
        assert url == "http://sglang/server_info"
        assert timeout == 3.0
        return server_info_sequence.pop(0)

    def fake_post_json(url, payload, *, timeout):
        if url.endswith("/api/v1/sync_inference_weights"):
            return {
                "success": True,
                "message": "ok",
                "fp8_kv_cache_enabled": True,
                "fp8_kv_cache_postprocess_requested": True,
                "fp8_kv_cache_static_scales": True,
                "cache_epoch": 8,
                "endpoints_synced": [
                    {
                        "host": "sglang",
                        "port": 30000,
                        "success": True,
                        "fp8_kv_cache_postprocess_ran": True,
                        "fp8_kv_cache_static_scales_updated": True,
                        "cache_epoch": 8,
                    }
                ],
            }
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_get_json", fake_get_json)
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: [[(-1.0, 10)]],
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-kv-cache-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        cache_invalidation_mode="auto",
        fp8_kv_cache_gate=True,
        require_cache_epoch_change=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "pass"
    assert result["config"]["fp8_kv_cache_gate"] is True
    assert result["config"]["require_cache_epoch_change"] is True
    assert result["server_info_before"]["kv_cache_dtype"] == "fp8_e4m3"
    assert result["server_info_before"]["fp8_kv_cache_enabled"] is True
    assert result["server_info_before"]["cache_epoch"] == 7
    assert result["server_info_after_sync"]["cache_epoch"] == 8
    assert result["sync_result"]["fp8_kv_cache_enabled"] is True
    assert result["sync_result"]["fp8_kv_cache_postprocess_requested"] is True
    assert result["sync_result"]["endpoints_synced"][0]["fp8_kv_cache_postprocess_ran"] is True


def test_fp8_sync_gate_same_weight_initial_alignment_checks_second_sync_epoch(monkeypatch):
    server_info_sequence = [
        {"quantization": "fp8", "kv_cache_dtype": "fp8_e4m3", "cache_epoch": 1},
        {"quantization": "fp8", "kv_cache_dtype": "fp8_e4m3", "cache_epoch": 2},
        {"quantization": "fp8", "kv_cache_dtype": "fp8_e4m3", "cache_epoch": 3},
    ]
    score_sequence = [
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
    ]
    sync_epochs = iter([2, 3])
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            epoch = next(sync_epochs)
            return {
                "success": True,
                "message": "ok",
                "fp8_kv_cache_enabled": True,
                "fp8_kv_cache_postprocess_requested": True,
                "fp8_kv_cache_static_scales": True,
                "cache_epoch": epoch,
                "endpoints_synced": [
                    {
                        "host": "sglang",
                        "port": 30000,
                        "success": True,
                        "fp8_kv_cache_postprocess_ran": True,
                        "fp8_kv_cache_static_scales_updated": True,
                        "cache_epoch": epoch,
                    }
                ],
            }
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_get_json", lambda *args, **kwargs: server_info_sequence.pop(0))
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: score_sequence.pop(0),
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-kv-cache-second-sync-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        cache_invalidation_mode="auto",
        fp8_kv_cache_gate=True,
        require_cache_epoch_change=True,
        same_weight_initial_alignment_sync=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    sync_calls = [call for call in post_calls if call[0].endswith("/api/v1/sync_inference_weights")]

    assert result["status"] == "pass"
    assert result["mode"] == "same_weight"
    assert result["config"]["same_weight_initial_alignment_sync"] is True
    assert result["comparison_phase"] == "after_initial_sync_vs_second_sync"
    assert result["server_info_before"]["cache_epoch"] == 1
    assert result["server_info_after_initial_sync"]["cache_epoch"] == 2
    assert result["server_info_after_sync"]["cache_epoch"] == 3
    assert result["initial_sync_result"]["fp8_kv_cache_enabled"] is True
    assert result["sync_result"]["fp8_kv_cache_enabled"] is True
    assert len(sync_calls) == 2
    assert not server_info_sequence
    assert not score_sequence


def test_fp8_sync_gate_trained_update_fp8_kv_cache_gate_records_resync_epoch(monkeypatch):
    server_info_sequence = [
        {"kv_cache_dtype": "fp8_e4m3", "cache_epoch": 1},
        {"kv_cache_dtype": "fp8_e4m3", "cache_epoch": 2},
        {"kv_cache_dtype": "fp8_e4m3", "cache_epoch": 3},
        {"kv_cache_dtype": "fp8_e4m3", "cache_epoch": 4},
    ]
    score_sequence = [
        [[(-10.0, 10)]],  # baseline after initial alignment sync
        [[(-10.0, 10)]],  # stale receiver after training, before trained sync
        [[(-1.0, 10)]],  # trained update after sync
        [[(-1.0, 10)]],  # second no-op sync remains stable
    ]
    post_calls = []
    future_results = {
        "fb-0": {"loss_fn_outputs": [{"loss": 1.0}], "metrics": {"grad_norm": 2.0}, "info": {}},
        "opt-0": {"metrics": {"learning_rate": 0.2}, "info": {}},
    }
    sync_epochs = iter([2, 3, 4])

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            epoch = next(sync_epochs)
            return {
                "success": True,
                "message": "ok",
                "fp8_kv_cache_enabled": True,
                "fp8_kv_cache_postprocess_requested": True,
                "fp8_kv_cache_static_scales": True,
                "cache_epoch": epoch,
                "endpoints_synced": [
                    {
                        "host": "sglang",
                        "port": 30000,
                        "success": True,
                        "fp8_kv_cache_postprocess_ran": True,
                        "fp8_kv_cache_static_scales_updated": True,
                        "cache_epoch": epoch,
                    }
                ],
            }
        if url.endswith("/api/v1/forward_backward"):
            return {"request_id": "fb-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/optim_step"):
            return {"request_id": "opt-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/retrieve_future"):
            return future_results[payload["request_id"]]
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_get_json", lambda *args, **kwargs: server_info_sequence.pop(0))
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: score_sequence.pop(0),
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-kv-cache-trained-update-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        cache_invalidation_mode="auto",
        fp8_kv_cache_gate=True,
        require_cache_epoch_change=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "pass"
    assert result["mode"] == "trained_update"
    assert result["comparison_phase"] == "after_train_sync_vs_second_sync"
    assert result["server_info_before"]["cache_epoch"] == 1
    assert result["server_info_after_initial_sync"]["cache_epoch"] == 2
    assert result["server_info_after_sync"]["cache_epoch"] == 3
    assert result["server_info_after_resync"]["cache_epoch"] == 4
    assert result["sync_result"]["fp8_kv_cache_postprocess_requested"] is True
    assert result["post_train_resync_result"]["fp8_kv_cache_enabled"] is True
    assert result["stale_receiver_comparison"]["passed"] is True
    assert result["trained_update_change"]["changed"] is True
    assert not server_info_sequence
    assert len([call for call in post_calls if call[0].endswith("/api/v1/sync_inference_weights")]) == 3


def test_fp8_sync_gate_fp8_kv_cache_gate_fails_unchanged_epoch(monkeypatch):
    server_info_sequence = [
        {"kv_cache_dtype": "fp8", "cache_epoch": "epoch-7"},
        {"kv_cache_dtype": "fp8", "cache_epoch": "epoch-7"},
    ]

    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_get_json",
        lambda *args, **kwargs: server_info_sequence.pop(0),
    )
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: [[(-1.0, 10)]],
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_post_json",
        lambda url, payload, *, timeout: (
            {
                "success": True,
                "message": "ok",
                "fp8_kv_cache_enabled": True,
            }
            if url.endswith("/api/v1/sync_inference_weights")
            else {"success": True}
        ),
    )

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-kv-cache-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        cache_invalidation_mode="auto",
        fp8_kv_cache_gate=True,
        require_cache_epoch_change=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert "Cache epoch did not change" in result["failure_reason"]


def test_fp8_sync_gate_can_opt_out_of_receiver_cache_flush(monkeypatch):
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        return {"success": True}

    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: [[(-1.0, 10)]],
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-no-flush-test",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    sync_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/sync_inference_weights"))

    assert result["status"] == "pass"
    assert result["config"]["flush_cache"] is False
    assert sync_payload["flush_cache"] is False


def test_fp8_sync_gate_trained_update_runs_training_and_stability_sync(monkeypatch):
    score_sequence = [
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
        [[(-0.5, 10)]],
        [[(-0.5, 10)]],
    ]
    score_calls = []

    def fake_score(url, prompts, *, max_new_tokens, timeout, **kwargs):
        score_calls.append((url, list(prompts), max_new_tokens, timeout))
        return score_sequence.pop(0)

    post_calls = []
    future_results = {
        "fb-0": {"loss_fn_outputs": [{"loss": 1.0}], "metrics": {"grad_norm": 2.0}, "info": {}},
        "opt-0": {"metrics": {"learning_rate": 0.2}, "info": {}},
    }

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        if url.endswith("/api/v1/forward_backward"):
            return {"request_id": "fb-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/optim_step"):
            return {"request_id": "opt-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/retrieve_future"):
            return future_results[payload["request_id"]]
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_generated_logprobs", fake_score)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    sync_calls = [call for call in post_calls if call[0].endswith("/api/v1/sync_inference_weights")]
    forward_backward_payload = next(
        payload for url, payload, _ in post_calls if url.endswith("/api/v1/forward_backward")
    )
    optim_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/optim_step"))

    assert result["status"] == "pass"
    assert result["mode"] == "trained_update"
    assert result["comparison_phase"] == "after_train_sync_vs_second_sync"
    assert result["stale_receiver_comparison"]["passed"] is True
    assert result["trained_update_change"]["changed"] is True
    assert len(sync_calls) == 3
    assert [payload["model_id"] for _, payload, _ in sync_calls] == ["default", "default", "default"]
    assert len(score_calls) == 4
    assert forward_backward_payload["seq_id"] == 42
    assert forward_backward_payload["forward_backward_input"]["data"][0]["model_input"]["input_ids"] == [11, 12, 13]
    assert optim_payload["seq_id"] == 43
    assert optim_payload["learning_rate"] == pytest.approx(0.2)
    assert optim_payload["gradient_clip"] == pytest.approx(0.7)


def test_fp8_sync_gate_qarl_trained_update_uses_server_derived_config(monkeypatch):
    score_sequence = [
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
        [[(-0.5, 10)]],
        [[(-0.5, 10)]],
    ]

    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: score_sequence.pop(0),
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)

    post_calls = []
    future_results = {
        "fb-0": {"loss_fn_outputs": [{"loss": 1.0}], "metrics": {"grad_norm": 2.0}, "info": {}},
        "opt-0": {"metrics": {"learning_rate": 0.2}, "info": {}},
    }

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        if url.endswith("/api/v1/forward_backward"):
            return {"request_id": "fb-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/optim_step"):
            return {"request_id": "opt-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/retrieve_future"):
            return future_results[payload["request_id"]]
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="qarl-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="qarl",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)
    set_sync_payload = next(payload for url, payload, _ in post_calls if url.endswith("/api/v1/set_sync_quantization"))
    sync_payloads = [payload for url, payload, _ in post_calls if url.endswith("/api/v1/sync_inference_weights")]

    assert result["status"] == "pass"
    assert result["mode"] == "trained_update"
    assert result["config"]["sync_quantization"] == "qarl"
    assert result["config"]["qarl_sync"] is True
    assert result["config"]["quantization"] is None
    assert result["comparison_phase"] == "after_train_sync_vs_second_sync"
    assert result["stale_receiver_comparison"]["passed"] is True
    assert result["trained_update_change"]["changed"] is True
    assert set_sync_payload == {"quantization": None}
    assert len(sync_payloads) == 3
    assert all("quantization" not in payload for payload in sync_payloads)
    assert [payload["model_id"] for payload in sync_payloads] == ["default", "default", "default"]


def test_fp8_sync_gate_stabilizes_target_token_baseline_before_training(monkeypatch):
    score_sequence = [
        [[(-10.0, 0)]],  # warmup
        [[(-9.0, 0)]],  # first baseline probe is unstable
        [[(-10.0, 0)]],
        [[(-10.0, 0)]],  # stable old-weight baseline
        [[(-10.0, 0)]],  # stale receiver after training, before sync
        [[(-2.0, 0)]],  # first post-sync probe is transient
        [[(-1.0, 0)]],
        [[(-1.0, 0)]],  # stable trained-weight post-sync baseline
        [[(-1.0, 0)]],
        [[(-1.0, 0)]],  # stable second-sync baseline
    ]
    score_calls = []

    def fake_score(url, prompts, *, max_new_tokens, timeout, **kwargs):
        score_calls.append((url, list(prompts), max_new_tokens, timeout))
        return score_sequence.pop(0)

    post_calls = []
    future_results = {
        "fb-0": {"loss_fn_outputs": [{"loss": 1.0}], "metrics": {"grad_norm": 2.0}, "info": {}},
        "opt-0": {"metrics": {"learning_rate": 0.2}, "info": {}},
    }

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        if url.endswith("/api/v1/forward_backward"):
            return {"request_id": "fb-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/optim_step"):
            return {"request_id": "opt-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/retrieve_future"):
            return future_results[payload["request_id"]]
        return {"success": True}

    flush_calls = []
    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_input_logprobs", fake_score)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_flush_sglang_cache",
        lambda url, *, timeout: flush_calls.append((url, timeout)),
    )

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        pre_train_score_stability_attempts=4,
        post_sync_score_stability_attempts=4,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "pass"
    assert result["pre_train_baseline_stability"]["passed"] is True
    assert result["pre_train_baseline_stability"]["attempt_count"] == 3
    assert result["pre_train_baseline_stability"]["selected_attempt"] == 2
    assert result["post_train_sync_stability"]["passed"] is True
    assert result["post_train_sync_stability"]["attempt_count"] == 3
    assert result["post_train_sync_stability"]["selected_attempt"] == 2
    assert result["post_train_resync_stability"]["passed"] is True
    assert result["post_train_resync_stability"]["attempt_count"] == 2
    assert result["stale_receiver_comparison"]["passed"] is True
    assert result["trained_update_change"]["changed"] is True
    assert len(score_calls) == 10
    assert flush_calls == [("http://sglang", 3.0)] * 10


def test_fp8_sync_gate_fails_unstable_target_token_post_sync_scores(monkeypatch):
    score_sequence = [
        [[(-10.0, 0)]],  # warmup
        [[(-10.0, 0)]],
        [[(-10.0, 0)]],  # stable old-weight baseline
        [[(-10.0, 0)]],  # stale receiver after training, before sync
        [[(-1.0, 0)]],
        [[(-2.0, 0)]],
        [[(-1.0, 0)]],  # never gets two consecutive matching post-sync scores
    ]
    post_calls = []
    future_results = {
        "fb-0": {"loss_fn_outputs": [{"loss": 1.0}], "metrics": {"grad_norm": 2.0}, "info": {}},
        "opt-0": {"metrics": {"learning_rate": 0.2}, "info": {}},
    }

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        if url.endswith("/api/v1/forward_backward"):
            return {"request_id": "fb-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/optim_step"):
            return {"request_id": "opt-0", "model_id": payload["model_id"]}
        if url.endswith("/api/v1/retrieve_future"):
            return future_results[payload["request_id"]]
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_input_logprobs", lambda *args, **kwargs: score_sequence.pop(0))
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_flush_sglang_cache", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        pre_train_score_stability_attempts=3,
        post_sync_score_stability_attempts=3,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert result["comparison_phase"] == "post_train_sync_stability"
    assert result["post_train_sync_stability"]["passed"] is False
    assert result["post_train_sync_stability"]["attempt_count"] == 3
    assert result["trained_update_change"]["changed"] is True
    assert "Post-sync receiver target-token scoring was unstable" in result["failure_reason"]
    sync_calls = [call for call in post_calls if call[0].endswith("/api/v1/sync_inference_weights")]
    assert len(sync_calls) == 2


def test_fp8_sync_gate_fails_unstable_target_token_scores_before_initial_sync(monkeypatch):
    score_sequence = [
        [[(-10.0, 0)]],
        [[(-9.0, 0)]],
        [[(-10.0, 0)]],
    ]
    post_calls = []
    register_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_input_logprobs", lambda *args, **kwargs: score_sequence.pop(0))
    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_register_inference_endpoint",
        lambda *args, **kwargs: register_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_flush_sglang_cache", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        pre_initial_score_stability_attempts=3,
        pre_train_score_stability_attempts=3,
        post_sync_score_stability_attempts=3,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert result["comparison_phase"] == "pre_initial_score_stability"
    assert result["pre_initial_score_stability"]["passed"] is False
    assert "Pre-initial-sync receiver target-token scoring was unstable" in result["failure_reason"]
    assert register_calls == []
    assert post_calls == []


def test_fp8_sync_gate_preserves_pre_initial_stability_when_sync_fails(monkeypatch):
    score_sequence = [
        [[(-10.0, 0)]],
        [[(-10.0, 0)]],
        [[(-10.0, 0)]],
    ]
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": False, "message": "p2p init failed"}
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_input_logprobs", lambda *args, **kwargs: score_sequence.pop(0))
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_flush_sglang_cache", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="none",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=0,
        pre_initial_score_stability_attempts=2,
        pre_train_score_stability_attempts=3,
        post_sync_score_stability_attempts=3,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert result["comparison_phase"] == "same_weight_sync_failure"
    assert result["pre_initial_score_stability"]["passed"] is True
    assert result["pre_initial_score_stability"]["attempt_count"] == 2
    assert result["pre_initial"] == [[{"logprob": -10.0, "token_id": 0}]]
    assert "p2p init failed" in result["failure_reason"]
    assert any(url.endswith("/api/v1/set_sync_quantization") for url, _, _ in post_calls)
    assert any(url.endswith("/api/v1/sync_inference_weights") for url, _, _ in post_calls)


def test_fp8_sync_gate_preserves_sync_context_when_post_sync_scoring_fails(monkeypatch):
    score_sequence = [
        [[(-10.0, 0)]],
        [[(-10.0, 0)]],
        RuntimeError("Logprob entry does not contain a numeric score: [None, 15, None]"),
    ]
    post_calls = []

    def fake_score(*args, **kwargs):
        item = score_sequence.pop(0)
        if isinstance(item, RuntimeError):
            raise item
        return item

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "Synced 53 params"}
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_input_logprobs", fake_score)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_flush_sglang_cache", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        pre_initial_score_stability_attempts=2,
        pre_train_score_stability_attempts=3,
        post_sync_score_stability_attempts=3,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert result["comparison_phase"] == "pre_train_warmup_scoring_failure"
    assert result["pre_initial_score_stability"]["passed"] is True
    assert result["initial_sync_result"]["success"] is True
    assert result["set_sync_quantization"]["success"] is True
    assert "Logprob entry does not contain a numeric score" in result["failure_reason"]
    assert "before_warmup" not in result
    assert any(url.endswith("/api/v1/sync_inference_weights") for url, _, _ in post_calls)


def test_fp8_sync_gate_fails_unstable_target_token_baseline_before_training(monkeypatch):
    score_sequence = [
        [[(-10.0, 0)]],  # warmup
        [[(-9.0, 0)]],
        [[(-10.0, 0)]],
        [[(-9.0, 0)]],
    ]
    post_calls = []

    def fake_post_json(url, payload, *, timeout):
        post_calls.append((url, payload, timeout))
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_score_input_logprobs", lambda *args, **kwargs: score_sequence.pop(0))
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)
    monkeypatch.setattr(fp8_sync_logprob_gate, "_flush_sglang_cache", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=[],
        prompt_input_ids=[[1, 1, 1, 1, 0]],
        prompt_file=None,
        max_new_tokens=1,
        score_input_logprobs=True,
        flush_cache_before_score=True,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        flush_cache=True,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        pre_train_score_stability_attempts=3,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert result["comparison_phase"] == "pre_train_baseline_stability"
    assert result["pre_train_baseline_stability"]["passed"] is False
    assert "Pre-train receiver target-token scoring was unstable" in result["failure_reason"]
    assert not any(url.endswith("/api/v1/forward_backward") for url, _, _ in post_calls)


def test_fp8_sync_gate_trained_update_requires_receiver_change(monkeypatch):
    score_sequence = [
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
    ]

    monkeypatch.setattr(
        fp8_sync_logprob_gate,
        "_score_generated_logprobs",
        lambda *args, **kwargs: score_sequence.pop(0),
    )
    monkeypatch.setattr(fp8_sync_logprob_gate, "_register_inference_endpoint", lambda *args, **kwargs: None)

    def fake_post_json(url, payload, *, timeout):
        if url.endswith("/api/v1/sync_inference_weights"):
            return {"success": True, "message": "ok"}
        if url.endswith("/api/v1/forward_backward"):
            return {"request_id": "fb-0"}
        if url.endswith("/api/v1/optim_step"):
            return {"request_id": "opt-0"}
        if url.endswith("/api/v1/retrieve_future"):
            return {"metrics": {}, "info": {}}
        return {"success": True}

    monkeypatch.setattr(fp8_sync_logprob_gate, "_post_json", fake_post_json)

    args = argparse.Namespace(
        xorl_url="http://xorl",
        sglang_url="http://sglang",
        reference_sglang_url=None,
        prompt=["hello"],
        prompt_file=None,
        max_new_tokens=1,
        master_address="10.0.0.1",
        master_port=0,
        group_name="fp8-trained-update",
        buffer_size_mb=128,
        weight_version=None,
        sync_quantization="fp8",
        fmt="e4m3",
        weight_block_size=[128, 128],
        skip_module=[],
        per_call=False,
        sparse_delta=False,
        delta_encoding_path=None,
        sparse_delta_output_dir=None,
        sparse_delta_keep_files=False,
        sparse_delta_skip_post_process=False,
        sparse_delta_timeout_s=None,
        pre_sync_train_steps=1,
        train_model_id="default",
        train_input_ids=[11, 12, 13],
        train_labels=[12, 13, 14],
        train_learning_rate=0.2,
        train_gradient_clip=0.7,
        train_seq_id_base=42,
        future_poll_interval=0.0,
        train_change_atol=1e-6,
        allow_missing_train_change=False,
        atol=0.05,
        timeout=3.0,
        output_json=None,
    )

    result = fp8_sync_logprob_gate.run_gate(args)

    assert result["status"] == "fail"
    assert result["trained_update_change"]["changed"] is False
    assert "Trained update did not change receiver" in result["failure_reason"]


def test_fp8_sync_gate_reports_drift_failures():
    result = fp8_sync_logprob_gate._compare_scores(
        [[(-1.0, 10)]],
        [[(-1.2, 10)]],
        atol=0.05,
    )

    assert result["passed"] is False
    assert result["compared"] == 1
    assert result["abs_diff"]["max"] == pytest.approx(0.2)
    assert result["failures"][0]["reason"] == "abs_diff_exceeded"


def test_fp8_sync_gate_exhaustive_stability_requires_final_stable_run():
    score_sequence = [
        [[(-1.0, 10)]],
        [[(-1.0, 10)]],
        [[(-1.3, 10)]],
        [[(-1.3, 10)]],
        [[(-1.6, 10)]],
    ]

    def fake_score_receiver(url):
        assert url == "http://sglang"
        return score_sequence.pop(0)

    score, stability = fp8_sync_logprob_gate._score_stable_baseline(
        score_receiver=fake_score_receiver,
        sglang_url="http://sglang",
        attempts=5,
        atol=0.05,
        required_consecutive_matches=3,
        exhaustive=True,
    )

    assert score == [[(-1.6, 10)]]
    assert stability["passed"] is False
    assert stability["attempt_count"] == 5
    assert stability["required_consecutive_matches"] == 3
    assert stability["exhaustive"] is True
    assert stability["max_stable_run_length"] == 2
    assert stability["final_stable_run_length"] == 1
    assert stability["final_comparison"]["passed"] is False


def test_launch_k3_builds_fp8_sync_gate_command():
    args = argparse.Namespace(
        fp8_sync_master_address="",
        fp8_sync_master_port=0,
        fp8_sync_group_name="fp8-group",
        fp8_sync_buffer_size_mb=256,
        fp8_sync_quantization="fp8",
        fp8_sync_fmt="e4m3",
        fp8_sync_weight_block_size=[64, 128],
        fp8_sync_max_new_tokens=3,
        fp8_sync_atol=0.01,
        compare_timeout=12.0,
        fp8_sync_per_call=True,
        fp8_sync_weight_version="sync-v1",
        fp8_sync_prompt_file="/tmp/prompts.json",
        fp8_sync_prompt=["prompt-a", "prompt-b"],
        fp8_sync_prompt_input_ids=[[1, 2, 3]],
        fp8_sync_score_input_logprobs=True,
        fp8_sync_pre_initial_score_stability_attempts=2,
        fp8_sync_pre_train_score_stability_attempts=5,
        fp8_sync_post_sync_score_stability_attempts=6,
        fp8_sync_score_stability_required_consecutive_matches=4,
        fp8_sync_score_stability_exhaustive=True,
        fp8_sync_no_flush_cache_before_score=True,
        fp8_sync_no_flush_cache=True,
        fp8_sync_same_weight_initial_alignment_sync=True,
        fp8_sync_fp8_kv_cache_gate=True,
        fp8_sync_require_cache_epoch_change=True,
        fp8_sync_skip_module=["lm_head"],
    )

    cmd = launch_k3_test.build_fp8_sync_gate_command(
        python_bin=Path("/venv/bin/python"),
        xorl_url="http://xorl:8000",
        sglang_url="http://sglang:30000",
        output_json="/tmp/out.json",
        master_address="10.0.0.5",
        args=args,
    )

    assert cmd[:2] == ["/venv/bin/python", str(REPO_ROOT / "scripts" / "fp8_sync_logprob_gate.py")]
    assert cmd[cmd.index("--master-address") + 1] == "10.0.0.5"
    assert cmd[cmd.index("--weight-block-size") + 1 : cmd.index("--weight-block-size") + 3] == ["64", "128"]
    assert cmd[cmd.index("--sync-quantization") + 1] == "fp8"
    assert cmd[cmd.index("--max-new-tokens") + 1] == "3"
    assert "--per-call" in cmd
    assert "--score-input-logprobs" in cmd
    assert cmd[cmd.index("--pre-initial-score-stability-attempts") + 1] == "2"
    assert cmd[cmd.index("--pre-train-score-stability-attempts") + 1] == "5"
    assert cmd[cmd.index("--post-sync-score-stability-attempts") + 1] == "6"
    assert cmd[cmd.index("--score-stability-required-consecutive-matches") + 1] == "4"
    assert "--score-stability-exhaustive" in cmd
    assert "--no-flush-cache-before-score" in cmd
    assert "--no-flush-cache" in cmd
    assert "--same-weight-initial-alignment-sync" in cmd
    assert "--fp8-kv-cache-gate" in cmd
    assert "--require-cache-epoch-change" in cmd
    assert cmd.count("--prompt") == 2
    assert cmd[cmd.index("--prompt-input-ids") + 1 : cmd.index("--prompt-input-ids") + 4] == ["1", "2", "3"]
    assert cmd[cmd.index("--skip-module") + 1] == "lm_head"


def test_fp8_sync_gate_registers_endpoint_with_receiver_kv_cache_dtype(monkeypatch):
    post_calls = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError(f"unexpected HTTP error in test: {self.status_code}")

        def json(self):
            return self._payload

    def fake_post(url, *, json, timeout):
        post_calls.append((url, json, timeout))
        if url.endswith("/api/v1/add_inference_endpoint"):
            return FakeResponse(404, {"detail": "not found"})
        return FakeResponse(200, {"success": True})

    monkeypatch.setattr(fp8_sync_logprob_gate.requests, "post", fake_post)

    fp8_sync_logprob_gate._register_inference_endpoint(
        "http://xorl:8000",
        "http://sglang:30000",
        timeout=3.0,
        receiver_kv_cache_dtype="fp8",
    )

    assert post_calls[0][1] == {
        "host": "sglang",
        "port": 30000,
        "world_size": 1,
        "receiver_kv_cache_dtype": "fp8",
    }
    assert post_calls[1][1] == post_calls[0][1]


def test_launch_k3_builds_qarl_fp8_sync_gate_command():
    args = argparse.Namespace(
        fp8_sync_master_address="",
        fp8_sync_master_port=0,
        fp8_sync_group_name="fp8-group",
        fp8_sync_buffer_size_mb=256,
        fp8_sync_quantization="qarl",
        fp8_sync_fmt="e4m3",
        fp8_sync_weight_block_size=[64, 128],
        fp8_sync_max_new_tokens=3,
        fp8_sync_atol=0.01,
        compare_timeout=12.0,
        fp8_sync_per_call=False,
        fp8_sync_weight_version=None,
        fp8_sync_prompt_file=None,
        fp8_sync_prompt=[],
        fp8_sync_prompt_input_ids=[[1, 2, 3]],
        fp8_sync_score_input_logprobs=True,
        fp8_sync_pre_initial_score_stability_attempts=0,
        fp8_sync_pre_train_score_stability_attempts=4,
        fp8_sync_post_sync_score_stability_attempts=4,
        fp8_sync_score_stability_required_consecutive_matches=2,
        fp8_sync_score_stability_exhaustive=False,
        fp8_sync_no_flush_cache_before_score=False,
        fp8_sync_no_flush_cache=False,
        fp8_sync_skip_module=[],
    )

    cmd = launch_k3_test.build_fp8_sync_gate_command(
        python_bin=Path("/venv/bin/python"),
        xorl_url="http://xorl:8000",
        sglang_url="http://sglang:30000",
        output_json="/tmp/out.json",
        master_address="10.0.0.5",
        args=args,
    )

    assert cmd[cmd.index("--sync-quantization") + 1] == "qarl"


def test_launch_k3_builds_trained_update_fp8_sync_gate_command():
    args = argparse.Namespace(
        fp8_sync_master_address="",
        fp8_sync_master_port=0,
        fp8_sync_group_name="fp8-group",
        fp8_sync_buffer_size_mb=256,
        fp8_sync_quantization="fp8",
        fp8_sync_fmt="e4m3",
        fp8_sync_weight_block_size=[64, 128],
        fp8_sync_max_new_tokens=3,
        fp8_sync_atol=0.01,
        compare_timeout=12.0,
        fp8_sync_per_call=False,
        fp8_sync_weight_version=None,
        fp8_sync_prompt_file=None,
        fp8_sync_prompt=[],
        fp8_sync_no_flush_cache_before_score=False,
        fp8_sync_no_flush_cache=False,
        fp8_sync_skip_module=[],
        fp8_sync_train_steps=2,
        fp8_sync_train_model_id="default",
        fp8_sync_train_input_ids=[11, 12, 13],
        fp8_sync_train_labels=[12, 13, 14],
        fp8_sync_train_learning_rate=0.2,
        fp8_sync_train_gradient_clip=0.7,
        fp8_sync_train_seq_id_base=42,
        fp8_sync_future_poll_interval=0.1,
        fp8_sync_train_change_atol=1e-5,
        fp8_sync_allow_missing_train_change=True,
    )

    cmd = launch_k3_test.build_fp8_sync_gate_command(
        python_bin=Path("/venv/bin/python"),
        xorl_url="http://xorl:8000",
        sglang_url="http://sglang:30000",
        output_json="/tmp/out.json",
        master_address="10.0.0.5",
        args=args,
    )

    assert cmd[cmd.index("--pre-sync-train-steps") + 1] == "2"
    assert cmd[cmd.index("--train-model-id") + 1] == "default"
    assert cmd[cmd.index("--train-learning-rate") + 1] == "0.2"
    assert cmd[cmd.index("--train-gradient-clip") + 1] == "0.7"
    assert cmd[cmd.index("--train-seq-id-base") + 1] == "42"
    assert cmd[cmd.index("--future-poll-interval") + 1] == "0.1"
    assert cmd[cmd.index("--train-change-atol") + 1] == "1e-05"
    assert cmd[cmd.index("--train-input-ids") + 1 : cmd.index("--train-input-ids") + 4] == ["11", "12", "13"]
    assert cmd[cmd.index("--train-labels") + 1 : cmd.index("--train-labels") + 4] == ["12", "13", "14"]
    assert "--allow-missing-train-change" in cmd


def test_score_trace_prefers_generation_logprobs_for_generated_traces(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {
            "trace_id": "t0",
            "trace_mode": "sglang_generation",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-9.0],
            "sglang_generation_logprobs": [-0.5],
        }
    )

    monkeypatch.setattr(compare_static_traces, "xorl_forward", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(compare_static_traces, "extract_xorl_logprobs", lambda result, gen_len: [-0.5])

    result = compare_static_traces.score_trace_with_xorl(
        trace,
        xorl_url="http://xorl",
        xorl_model_id="default",
        timeout=1.0,
        local_forward=False,
        local_device="cpu",
        model_name="",
        reference_logprobs="auto",
    )

    token = result["per_token"][0]
    assert result["sample_k3_mean"] == pytest.approx(0.0)
    assert token["sglang_logprob"] == pytest.approx(-0.5)
    assert token["sglang_logprob_source"] == "generation"
    assert token["sglang_prefill_logprob"] == pytest.approx(-9.0)


def test_extract_xorl_token_diagnostics_slices_generated_tokens():
    result = {
        "loss_fn_outputs": [
            {
                "token_diagnostics": {
                    "valid_positions": [2, 3, 4],
                    "target_ids": [10, 11, 12],
                    "target_logprobs": [-1.0, -2.0, -3.0],
                    "target_ranks": [1, 5, 9],
                    "topk_ids": [[10, 1], [7, 11], [8, 9]],
                    "topk_logprobs": [[-1.0, -4.0], [-0.5, -2.0], [-0.25, -0.75]],
                    "loss_logprobs": [-1.0, -2.0, -3.0],
                    "loss_logprob_deltas": [0.0, 0.0, 0.0],
                    "reference_target_logprobs": [-0.9, -1.9, -2.9],
                    "reference_target_ranks": [1, 4, 8],
                    "reference_logprob_deltas": [-0.1, -0.1, -0.1],
                    "hidden_state_summaries": [
                        {"layer_count": 2, "layers": [{"index": 0, "rms": 1.0}]},
                        {"layer_count": 2, "layers": [{"index": 0, "rms": 2.0}]},
                        {"layer_count": 2, "layers": [{"index": 0, "rms": 3.0}]},
                    ],
                    "hidden_component_summaries": [
                        {"component_count": 1, "components": [{"layer": 33, "name": "layer_input", "rms": 1.0}]},
                        {"component_count": 1, "components": [{"layer": 34, "name": "mlp", "rms": 2.0}]},
                        {"component_count": 1, "components": [{"layer": 35, "name": "layer_output", "rms": 3.0}]},
                    ],
                }
            }
        ]
    }

    rows = compare_static_traces.extract_xorl_token_diagnostics(result, gen_len=2)

    assert rows[0]["target_id"] == 11
    assert rows[0]["target_rank"] == 5
    assert rows[0]["top"][1]["token_id"] == 11
    assert rows[0]["loss_logprob_delta"] == pytest.approx(0.0)
    assert rows[0]["reference_target_logprob"] == pytest.approx(-1.9)
    assert rows[0]["reference_target_rank"] == 4
    assert rows[0]["reference_logprob_delta"] == pytest.approx(-0.1)
    assert rows[0]["hidden_state_summary"]["layers"][0]["rms"] == 2.0
    assert rows[0]["hidden_component_summary"]["components"][0]["name"] == "mlp"
    assert rows[1]["target_logprob"] == pytest.approx(-3.0)


def test_score_trace_can_attach_xorl_topk_diagnostics(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-0.25],
        }
    )
    captured = {}

    def fake_xorl_forward(*args, **kwargs):
        captured["loss_fn_params"] = kwargs.get("loss_fn_params")
        return {
            "loss_fn_outputs": [
                {
                    "logprobs": {"data": [-0.5], "dtype": "float32", "shape": [1]},
                    "token_diagnostics": {
                        "valid_positions": [2],
                        "target_ids": [3],
                        "target_logprobs": [-0.5],
                        "target_ranks": [4],
                        "topk_ids": [[9, 8]],
                        "topk_logprobs": [[-0.1, -0.2]],
                        "loss_logprobs": [-0.5],
                        "loss_logprob_deltas": [0.0],
                        "reference_target_logprobs": [-0.45],
                        "reference_target_ranks": [3],
                        "reference_logprob_deltas": [-0.05],
                        "hidden_state_summaries": [
                            {
                                "layer_count": 2,
                                "sample_indices": [0, 2],
                                "layers": [{"index": 0, "rms": 12.0}],
                            }
                        ],
                        "hidden_component_summaries": [
                            {
                                "component_count": 1,
                                "sample_indices": [0, 2],
                                "components": [{"layer": 34, "name": "mlp", "rms": 13.0}],
                            }
                        ],
                    },
                }
            ]
        }

    monkeypatch.setattr(compare_static_traces, "xorl_forward", fake_xorl_forward)

    result = compare_static_traces.score_trace_with_xorl(
        trace,
        xorl_url="http://xorl",
        xorl_model_id="default",
        timeout=1.0,
        local_forward=False,
        local_device="cpu",
        model_name="",
        reference_logprobs="auto",
        xorl_diagnostic_topk=2,
        xorl_diagnostic_reference_logits=True,
        xorl_diagnostic_hidden_states=True,
        xorl_diagnostic_hidden_sample_count=2,
        xorl_diagnostic_hidden_sample_indices="1,2",
        xorl_diagnostic_hidden_components=True,
        xorl_diagnostic_hidden_component_layers="34,38",
        xorl_diagnostic_hidden_component_path="/home/apanda/k3_artifacts/full-components",
    )

    token = result["per_token"][0]
    assert captured["loss_fn_params"] == {
        "diagnostic_topk": 2,
        "diagnostic_reference_logits": True,
        "diagnostic_hidden_states": True,
        "diagnostic_hidden_sample_count": 2,
        "diagnostic_hidden_sample_indices": "1,2",
        "diagnostic_hidden_components": True,
        "diagnostic_hidden_component_layers": "34,38",
        "diagnostic_hidden_component_path": "/home/apanda/k3_artifacts/full-components",
    }
    assert token["xorl_target_rank"] == 4
    assert token["xorl_top_logprobs"][0]["token_id"] == 9
    assert token["xorl_loss_logprob_delta"] == pytest.approx(0.0)
    assert token["xorl_reference_target_logprob"] == pytest.approx(-0.45)
    assert token["xorl_reference_target_rank"] == 3
    assert token["xorl_reference_logprob_delta"] == pytest.approx(-0.05)
    assert token["xorl_hidden_state_summary"]["layers"][0]["rms"] == pytest.approx(12.0)
    assert token["xorl_hidden_component_summary"]["components"][0]["rms"] == pytest.approx(13.0)


def test_score_trace_can_request_xorl_hidden_component_file_without_topk(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-0.25],
        }
    )
    captured = {}

    def fake_xorl_forward(*args, **kwargs):
        captured["loss_fn_params"] = kwargs.get("loss_fn_params")
        return {
            "loss_fn_outputs": [
                {
                    "logprobs": {"data": [-0.5], "dtype": "float32", "shape": [1]},
                }
            ]
        }

    monkeypatch.setattr(compare_static_traces, "xorl_forward", fake_xorl_forward)

    result = compare_static_traces.score_trace_with_xorl(
        trace,
        xorl_url="http://xorl",
        xorl_model_id="default",
        timeout=1.0,
        local_forward=False,
        local_device="cpu",
        model_name="",
        reference_logprobs="auto",
        xorl_diagnostic_topk=0,
        xorl_diagnostic_hidden_components=False,
        xorl_diagnostic_hidden_component_layers="10-11",
        xorl_diagnostic_hidden_component_path="/home/apanda/k3_artifacts/l10-full",
    )

    assert result["per_token"][0]["xorl_logprob"] == pytest.approx(-0.5)
    assert captured["loss_fn_params"] == {
        "diagnostic_hidden_components": True,
        "diagnostic_hidden_component_layers": "10-11",
        "diagnostic_hidden_component_path": "/home/apanda/k3_artifacts/l10-full",
    }


def test_compare_hidden_component_artifacts_reports_component_deltas():
    def artifact(rms: float, sample_values: list[float], source_prefix: str):
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_logprob": -1.0,
                            "xorl_target_rank": 3,
                            "xorl_hidden_state_summary": {
                                "layer_count": 2,
                                "sample_indices": [0, 2, 4],
                                "layers": [
                                    {
                                        "index": 0,
                                        "rms": rms + 2,
                                        "max_abs": 6.0,
                                        "mean": 0.125,
                                        "sample_values": [4.0, 5.0, 6.0],
                                    },
                                    {
                                        "index": 1,
                                        "rms": rms + 3,
                                        "max_abs": 7.0,
                                        "mean": 0.0625,
                                        "sample_values": sample_values,
                                    },
                                ],
                            },
                            "xorl_hidden_component_summary": {
                                "component_count": 3,
                                "sample_indices": [0, 2, 4],
                                "components": [
                                    {
                                        "layer": 33,
                                        "name": "layer_input",
                                        "source_module": f"{source_prefix}:layer_input",
                                        "rms": rms,
                                        "max_abs": 4.0,
                                        "mean": 0.5,
                                        "sample_values": sample_values,
                                    },
                                    {
                                        "layer": 33,
                                        "name": "mlp",
                                        "source_module": f"{source_prefix}:mlp",
                                        "rms": rms + 1,
                                        "max_abs": 5.0,
                                        "mean": 0.25,
                                        "sample_values": [0.0, 1.0, 2.0],
                                    },
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    report = compare_hidden_component_artifacts.compare_hidden_component_artifacts(
        reference=artifact(1.0, [1.0, 2.0, 3.0], "reference"),
        candidate=artifact(1.25, [1.0, 1.5, 4.0], "candidate"),
        top_n=1,
        include_all=True,
    )

    assert report["summary"]["matched_component_token_count"] == 1
    assert report["summary"]["matched_component_count"] == 2
    assert report["summary"]["sample_max_abs_delta"]["max"] == pytest.approx(1.0)
    assert report["summary"]["abs_rms_delta"]["max"] == pytest.approx(0.25)
    assert report["summary"]["matched_hidden_state_token_count"] == 1
    assert report["summary"]["matched_hidden_state_layer_count"] == 2
    assert report["summary"]["hidden_state_sample_max_abs_delta"]["max"] == pytest.approx(1.0)
    assert report["summary"]["hidden_state_abs_rms_delta"]["max"] == pytest.approx(0.25)

    top = report["top_component_deltas"][0]
    assert top["trace_id"] == "trace-a"
    assert top["position"] == 0
    assert top["token_id"] == 7472
    assert top["layer"] == 33
    assert top["name"] == "layer_input"
    assert top["candidate_source_module"] == "candidate:layer_input"
    assert top["reference_source_module"] == "reference:layer_input"
    assert top["candidate_minus_reference_rms"] == pytest.approx(0.25)
    assert top["sample_max_abs_delta"] == pytest.approx(1.0)
    assert top["sample_deltas"] == pytest.approx([0.0, -0.5, 1.0])
    assert top["top_sample_deltas"][0]["hidden_index"] == 4
    assert top["top_sample_deltas"][0]["delta"] == pytest.approx(1.0)

    top_hidden = report["top_hidden_state_deltas"][0]
    assert top_hidden["trace_id"] == "trace-a"
    assert top_hidden["position"] == 0
    assert top_hidden["token_id"] == 7472
    assert top_hidden["layer_index"] == 1
    assert top_hidden["candidate_minus_reference_rms"] == pytest.approx(0.25)
    assert top_hidden["sample_max_abs_delta"] == pytest.approx(1.0)
    assert top_hidden["sample_deltas"] == pytest.approx([0.0, -0.5, 1.0])
    assert top_hidden["top_sample_deltas"][0]["hidden_index"] == 4
    assert top_hidden["top_sample_deltas"][0]["delta"] == pytest.approx(1.0)
    assert "hidden_state_deltas" in report


def test_component_source_leaderboard_ranks_cross_window_sources():
    def component_row(layer: int, name: str, delta: float, *, hidden_index: int = 4):
        return {
            "trace_id": "trace-a",
            "position": 0,
            "token_id": 7472,
            "layer": layer,
            "name": name,
            "sample_count": 3,
            "sample_max_abs_delta": abs(delta),
            "sample_mean_abs_delta": abs(delta) / 3,
            "candidate_minus_reference_rms": delta / 10,
            "candidate_logprob": -2.0,
            "reference_logprob": -1.5,
            "candidate_target_rank": 6,
            "reference_target_rank": 1,
            "top_sample_deltas": [
                {
                    "sample_offset": 0,
                    "hidden_index": hidden_index,
                    "candidate_value": 1.0 + delta,
                    "reference_value": 1.0,
                    "delta": delta,
                    "abs_delta": abs(delta),
                }
            ],
        }

    def hidden_row(layer: int, delta: float):
        return {
            "trace_id": "trace-a",
            "position": 0,
            "token_id": 7472,
            "layer_index": layer,
            "sample_count": 3,
            "sample_max_abs_delta": abs(delta),
            "sample_mean_abs_delta": abs(delta) / 3,
            "candidate_minus_reference_rms": delta / 10,
            "top_sample_deltas": [
                {
                    "sample_offset": 0,
                    "hidden_index": 4,
                    "candidate_value": 1.0 + delta,
                    "reference_value": 1.0,
                    "delta": delta,
                    "abs_delta": abs(delta),
                }
            ],
        }

    window_10 = {
        "component_deltas": [
            component_row(10, "layer_input", 0.2),
            component_row(10, "attention", 0.05),
            component_row(10, "post_attention_residual", 0.25),
            component_row(10, "mlp", 0.1),
            component_row(10, "layer_output", 0.35),
        ],
        "hidden_state_deltas": [hidden_row(10, 0.35)],
    }
    top_only_window = {
        "top_component_deltas": [component_row(11, "input_norm", 0.7)],
        "top_hidden_state_deltas": [hidden_row(11, 0.5)],
    }

    report = diagnose_component_source_leaderboard.build_component_source_leaderboard(
        [
            ("l10", "/tmp/l10.json", window_10),
            ("l11", "/tmp/l11.json", top_only_window),
        ],
        top_n=4,
        top_samples=1,
    )

    assert report["summary"]["comparison_count"] == 2
    assert report["summary"]["component_row_count"] == 6
    assert report["summary"]["hidden_state_row_count"] == 2
    assert report["summary"]["top_component_max_abs_delta"] == pytest.approx(0.7)
    assert report["top_component_rows"][0]["comparison_label"] == "l11"
    assert report["top_component_rows"][0]["name"] == "input_norm"

    by_name = {row["name"]: row for row in report["summary"]["max_by_component_name"]}
    assert by_name["input_norm"]["max_sample_max_abs_delta"] == pytest.approx(0.7)
    assert by_name["layer_output"]["max_sample_max_abs_delta"] == pytest.approx(0.35)

    source_rows = {
        (row["comparison_label"], row["layer"], row["equation"]): row for row in report["top_source_equation_rows"]
    }
    layer_output = source_rows[("l10", 10, "layer_output")]
    assert layer_output["output_max_abs_delta"] == pytest.approx(0.35)
    assert layer_output["dominant_term_component"] == "post_attention_residual"
    assert layer_output["dominant_term_max_abs_delta"] == pytest.approx(0.25)
    assert layer_output["term_deltas_at_top_hidden_index"] == {
        "post_attention_residual": pytest.approx(0.25),
        "mlp": pytest.approx(0.1),
    }
    assert layer_output["signed_equation_residual_at_top_hidden_index"] == pytest.approx(0.0)


def test_residual_delta_flow_splits_inherited_and_local_terms():
    def artifact(layer_input: list[float], attention: list[float], post_attention_residual: list[float]):
        components = [
            {
                "layer": 10,
                "name": "layer_input",
                "rms": 1.0,
                "max_abs": max(abs(v) for v in layer_input),
                "mean": sum(layer_input) / len(layer_input),
                "sample_values": layer_input,
            },
            {
                "layer": 10,
                "name": "attention",
                "rms": 1.0,
                "max_abs": max(abs(v) for v in attention),
                "mean": sum(attention) / len(attention),
                "sample_values": attention,
            },
            {
                "layer": 10,
                "name": "post_attention_residual",
                "rms": 1.0,
                "max_abs": max(abs(v) for v in post_attention_residual),
                "mean": sum(post_attention_residual) / len(post_attention_residual),
                "sample_values": post_attention_residual,
            },
        ]
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": len(components),
                                "sample_indices": [0, 1],
                                "components": components,
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_residual_delta_flow.diagnose_residual_delta_flow(
        reference=artifact(
            layer_input=[1.0, 2.0],
            attention=[0.5, -0.5],
            post_attention_residual=[1.5, 1.5],
        ),
        candidate=artifact(
            layer_input=[1.25, 1.75],
            attention=[0.75, -0.25],
            post_attention_residual=[2.0, 1.5],
        ),
        layers=[10],
        equations=["attention_residual"],
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["captured_output_delta_max_abs"] == pytest.approx(0.5)
    assert report["summary"]["primary_delta_max_abs"] == pytest.approx(0.25)
    assert report["summary"]["secondary_delta_sum_max_abs"] == pytest.approx(0.25)
    record = report["flow_records"][0]
    assert record["primary_component"] == "layer_input"
    assert record["secondary_components"] == ["attention"]
    assert record["secondary_reinforcement_fraction"] == pytest.approx(0.5)
    top = record["top_output_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["output_delta"] == pytest.approx(0.5)
    assert top["primary_delta"] == pytest.approx(0.25)
    assert top["secondary_delta_sum"] == pytest.approx(0.25)
    assert top["secondary_reinforces_primary"] is True
    assert top["bf16_inherited_only_delta"] == pytest.approx(0.25)
    assert top["bf16_local_only_delta"] == pytest.approx(0.25)
    assert top["captured_minus_bf16_recomputed_delta"] == pytest.approx(0.0)


def test_residual_delta_flow_windows_ranks_local_sensitivity(tmp_path):
    def artifact(path: Path, layer: int, values: dict[str, list[float]]):
        components = [
            {
                "layer": layer,
                "name": name,
                "rms": 1.0,
                "max_abs": max(abs(v) for v in sample_values),
                "mean": sum(sample_values) / len(sample_values),
                "sample_values": sample_values,
            }
            for name, sample_values in values.items()
        ]
        path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "trace_id": "trace-a",
                            "per_token": [
                                {
                                    "position": 0,
                                    "token_id": 7472,
                                    "xorl_hidden_component_summary": {
                                        "component_count": len(components),
                                        "sample_indices": [0, 1],
                                        "components": components,
                                    },
                                }
                            ],
                        }
                    ]
                }
            )
        )

    def comparison(path: Path, reference: Path, candidate: Path, layer: int):
        rows = [{"layer": layer, "name": name} for name in ["layer_input", "attention", "post_attention_residual"]]
        path.write_text(
            json.dumps(
                {
                    "config": {"reference": str(reference), "candidate": str(candidate)},
                    "component_deltas": rows,
                }
            )
        )

    ref0 = tmp_path / "ref0.json"
    cand0 = tmp_path / "cand0.json"
    cmp0 = tmp_path / "cmp0.json"
    artifact(
        ref0,
        0,
        {
            "layer_input": [1.0, 1.0],
            "attention": [0.0, 0.0],
            "post_attention_residual": [1.0, 1.0],
        },
    )
    artifact(
        cand0,
        0,
        {
            "layer_input": [1.5, 1.0],
            "attention": [0.0, 0.0],
            "post_attention_residual": [1.5, 1.0],
        },
    )
    comparison(cmp0, ref0, cand0, 0)

    ref1 = tmp_path / "ref1.json"
    cand1 = tmp_path / "cand1.json"
    cmp1 = tmp_path / "cmp1.json"
    artifact(
        ref1,
        1,
        {
            "layer_input": [1.0, 1.0],
            "attention": [0.0, 0.0],
            "post_attention_residual": [1.0, 1.0],
        },
    )
    artifact(
        cand1,
        1,
        {
            "layer_input": [1.0, 1.0],
            "attention": [0.25, 0.0],
            "post_attention_residual": [1.25, 1.0],
        },
    )
    comparison(cmp1, ref1, cand1, 1)

    report = diagnose_residual_delta_flow_windows.diagnose_residual_delta_flow_windows(
        [cmp0, cmp1],
        equations=["attention_residual"],
        top_n=2,
    )

    assert report["summary"]["window_count"] == 2
    assert report["summary"]["record_count"] == 2
    by_layer = report["summary"]["max_local_only_by_layer"]
    assert by_layer[0]["layer"] == 1
    assert by_layer[0]["local_only_max_abs"] == pytest.approx(0.25)
    assert by_layer[1]["layer"] == 0
    assert by_layer[1]["inherited_only_max_abs"] == pytest.approx(0.5)


def test_component_norm_boundary_diagnostic_recomputes_zero_centered_norm(tmp_path):
    def zero_centered_norm(values: list[float], weight: torch.Tensor) -> list[float]:
        vector = torch.tensor(values, dtype=torch.float32)
        output = vector * torch.rsqrt(vector.pow(2).mean(-1, keepdim=True) + 1e-6)
        output = output * (1.0 + weight.float())
        return output.to(dtype=torch.bfloat16).float().tolist()

    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(layer_input: list[float], input_norm: list[float]) -> dict:
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": 2,
                                "sample_indices": [0, 1, 2, 3],
                                "components": [
                                    component(10, "layer_input", layer_input),
                                    component(10, "input_norm", input_norm),
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    weight_name = "model.language_model.layers.10.input_layernorm.weight"
    weight = torch.tensor([0.0, 0.25, -0.5, 0.5], dtype=torch.float32)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file({weight_name: weight}, model_dir / "model-00001-of-00001.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {weight_name: "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    reference_input = [1.0, -2.0, 0.5, 4.0]
    candidate_input = [1.5, -1.0, 0.25, 4.0]
    report = diagnose_component_norm_boundary.diagnose_component_norm_boundaries(
        reference=artifact(reference_input, zero_centered_norm(reference_input, weight)),
        candidate=artifact(candidate_input, zero_centered_norm(candidate_input, weight)),
        model_path=model_dir,
        layers=[10],
        boundaries=["input"],
        norm_type="zero-centered",
        top_n=2,
    )

    assert report["summary"]["boundary_count"] == 1
    boundary = report["boundaries"][0]
    assert boundary["norm_weight_key"] == weight_name
    assert boundary["source_delta"]["max_abs"] == pytest.approx(1.0)
    assert boundary["candidate_recompute_diff"]["max_abs"] == pytest.approx(0.0)
    assert boundary["reference_recompute_diff"]["max_abs"] == pytest.approx(0.0)
    assert boundary["captured_vs_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)
    assert boundary["top_source_deltas"][0]["hidden_index"] == 1
    assert boundary["top_source_deltas"][0]["abs_delta"] == pytest.approx(1.0)
    assert boundary["top_reference_recompute_diffs"][0]["abs_delta"] == pytest.approx(0.0)


def test_component_residual_source_diagnostic_decomposes_layer_boundaries():
    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(values: dict[tuple[int, str], list[float]]) -> dict:
        components = [
            component(layer, name, component_values)
            for (layer, name), component_values in sorted(values.items(), key=lambda item: (item[0][0], item[0][1]))
        ]
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": len(components),
                                "sample_indices": [0, 1, 2],
                                "components": components,
                            },
                        }
                    ],
                }
            ]
        }

    reference = {
        (10, "layer_input"): [1.0, 2.0, 3.0],
        (10, "attention"): [0.5, 0.5, 0.5],
        (10, "post_attention_residual"): [1.5, 2.5, 3.5],
        (10, "experts"): [0.05, 0.10, 0.15],
        (10, "shared_expert_weighted"): [0.05, 0.10, 0.15],
        (10, "mlp"): [0.10, 0.20, 0.30],
        (10, "layer_output"): [1.6, 2.7, 3.8],
        (11, "layer_input"): [1.6, 2.7, 3.8],
    }
    candidate = {
        (10, "layer_input"): [1.5, 2.0, 3.0],
        (10, "attention"): [0.75, 0.25, 0.5],
        (10, "post_attention_residual"): [2.25, 2.25, 3.5],
        (10, "experts"): [0.0, 0.20, 0.15],
        (10, "shared_expert_weighted"): [0.0, 0.30, 0.15],
        (10, "mlp"): [0.0, 0.50, 0.30],
        (10, "layer_output"): [2.25, 2.75, 3.8],
        (11, "layer_input"): [2.25, 2.75, 3.8],
    }

    report = diagnose_component_residual_sources.diagnose_component_residual_sources(
        reference=artifact(reference),
        candidate=artifact(candidate),
        layers=[10],
        equations=["attention_residual", "layer_output", "mlp"],
        include_continuity=False,
        top_n=2,
    )

    assert report["summary"]["record_count"] == 3
    by_equation = {record["equation"]: record for record in report["records"]}
    attention_residual = by_equation["attention_residual"]
    assert attention_residual["output_delta"]["max_abs"] == pytest.approx(0.75)
    assert attention_residual["captured_vs_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)
    assert attention_residual["captured_vs_bf16_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)
    assert attention_residual["candidate_bf16_recompute_diff"]["max_abs"] == pytest.approx(0.0)

    layer_output = by_equation["layer_output"]
    assert layer_output["output_delta"]["max_abs"] == pytest.approx(0.65)
    assert layer_output["captured_vs_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)
    top = layer_output["top_output_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["delta"] == pytest.approx(0.65)
    assert [term["component"] for term in top["term_deltas"]] == ["post_attention_residual", "mlp"]
    assert top["term_deltas"][0]["delta"] == pytest.approx(0.75)
    assert top["term_deltas"][1]["delta"] == pytest.approx(-0.10)

    mlp = by_equation["mlp"]
    assert mlp["output_delta"]["max_abs"] == pytest.approx(0.30)
    assert [term["component"] for term in mlp["term_deltas"]] == ["experts", "shared_expert_weighted"]
    assert mlp["captured_vs_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)

    continuity = diagnose_component_residual_sources.diagnose_component_residual_sources(
        reference=artifact(reference),
        candidate=artifact(candidate),
        layers=[10, 11],
        equations=[],
        include_continuity=True,
        top_n=2,
    )
    assert continuity["summary"]["record_count"] == 1
    record = continuity["records"][0]
    assert record["kind"] == "continuity"
    assert record["output_delta"]["max_abs"] == pytest.approx(0.65)
    assert record["candidate_internal_diff"]["max_abs"] == pytest.approx(0.0)
    assert record["reference_internal_diff"]["max_abs"] == pytest.approx(0.0)
    assert record["captured_vs_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)
    assert record["captured_vs_bf16_recomputed_delta_diff"]["max_abs"] == pytest.approx(0.0)


def test_component_window_bridge_compares_adjacent_artifacts():
    def artifact(layer: int, component: str, values: list[float]) -> dict:
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7,
                            "xorl_hidden_component_summary": {
                                "sample_indices": [5, 8, 9],
                                "components": [
                                    {
                                        "layer": layer,
                                        "name": component,
                                        "sample_values": values,
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_component_window_bridge.diagnose_component_window_bridge(
        left_reference=artifact(5, "layer_output", [1.0, 2.0, 3.0]),
        left_candidate=artifact(5, "layer_output", [1.5, 2.0, 2.5]),
        right_reference=artifact(6, "layer_input", [1.0, 2.0, 3.0]),
        right_candidate=artifact(6, "layer_input", [1.5, 2.0, 2.25]),
        left_layer=5,
        left_component="layer_output",
        right_layer=6,
        right_component="layer_input",
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["left_delta_max_abs"] == pytest.approx(0.5)
    assert report["summary"]["right_delta_max_abs"] == pytest.approx(0.75)
    assert report["summary"]["candidate_bridge_diff_max_abs"] == pytest.approx(0.25)
    assert report["summary"]["reference_bridge_diff_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["delta_equivalence_diff_max_abs"] == pytest.approx(0.25)
    row = report["records"][0]["top_bridge_rows"][0]
    assert row["hidden_index"] == 9
    assert row["left_delta"] == pytest.approx(-0.5)
    assert row["right_delta"] == pytest.approx(-0.75)
    assert row["delta_equivalence_diff"] == pytest.approx(0.25)
    assert row["candidate_bridge_diff"] == pytest.approx(0.25)
    assert row["reference_bridge_diff"] == pytest.approx(0.0)


def test_component_three_way_diagnostic_classifies_reference_closest_terms():
    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(values: dict[tuple[int, str], list[float]]) -> dict:
        components = [
            component(layer, name, component_values)
            for (layer, name), component_values in sorted(values.items(), key=lambda item: (item[0][0], item[0][1]))
        ]
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": len(components),
                                "sample_indices": [10, 20, 30],
                                "components": components,
                            },
                        }
                    ],
                }
            ]
        }

    reference = {
        (10, "layer_input"): [1.0, 2.0, 3.0],
        (10, "attention"): [0.5, 0.5, 0.5],
        (10, "post_attention_residual"): [1.5, 2.5, 3.5],
        (10, "experts"): [0.0, 0.1, 0.2],
        (10, "shared_expert_weighted"): [0.1, 0.2, 0.3],
        (10, "mlp"): [0.1, 0.3, 0.5],
        (10, "layer_output"): [1.6, 2.8, 4.0],
    }
    xorl = {
        (10, "layer_input"): [1.4, 2.0, 3.0],
        (10, "attention"): [0.6, 0.5, 0.5],
        (10, "post_attention_residual"): [2.0, 2.5, 3.5],
        (10, "experts"): [0.0, 0.1, 0.2],
        (10, "shared_expert_weighted"): [0.3, 0.2, 0.3],
        (10, "mlp"): [0.3, 0.3, 0.5],
        (10, "layer_output"): [2.3, 2.8, 4.0],
    }
    slime = {
        (10, "layer_input"): [1.1, 2.0, 3.0],
        (10, "attention"): [0.55, 0.5, 0.5],
        (10, "post_attention_residual"): [1.65, 2.5, 3.5],
        (10, "experts"): [0.0, 0.1, 0.2],
        (10, "shared_expert_weighted"): [0.05, 0.2, 0.3],
        (10, "mlp"): [0.05, 0.3, 0.5],
        (10, "layer_output"): [1.7, 2.8, 4.0],
    }

    report = diagnose_component_three_way.diagnose_component_three_way(
        reference=artifact(reference),
        candidates={"xorl": artifact(xorl), "slime": artifact(slime)},
        layers=[10],
        equations=["layer_output", "mlp"],
        top_n=2,
    )

    assert report["config"]["matched_token_count"] == 1
    assert report["summary"]["component_record_count"] == 7
    assert report["summary"]["equation_record_count"] == 2
    assert report["summary"]["max_component_delta_by_candidate"]["xorl"] == pytest.approx(0.7)
    assert report["summary"]["max_component_delta_by_candidate"]["slime"] == pytest.approx(0.15)

    by_component = {(record["layer"], record["name"]): record for record in report["component_records"]}
    layer_output = by_component[(10, "layer_output")]
    assert layer_output["closest_candidate_by_max_abs"] == "slime"
    assert layer_output["candidate_deltas"]["xorl"]["max_abs"] == pytest.approx(0.7)
    assert layer_output["candidate_deltas"]["slime"]["max_abs"] == pytest.approx(0.1)
    assert layer_output["top_deltas"][0]["hidden_index"] == 10
    assert layer_output["top_deltas"][0]["closest_candidate"] == "slime"
    assert layer_output["top_deltas"][0]["candidate_deltas"]["xorl"] == pytest.approx(0.7)

    by_equation = {record["equation"]: record for record in report["equation_records"]}
    equation = by_equation["layer_output"]
    assert equation["closest_candidate_by_output_max_abs"] == "slime"
    assert equation["candidate_output_deltas"]["xorl"]["max_abs"] == pytest.approx(0.7)
    assert equation["candidate_output_deltas"]["slime"]["max_abs"] == pytest.approx(0.1)
    assert equation["top_output_deltas"][0]["hidden_index"] == 10
    assert equation["top_output_deltas"][0]["candidate_term_deltas"]["xorl"] == {
        "post_attention_residual": pytest.approx(0.5),
        "mlp": pytest.approx(0.2),
    }
    assert equation["top_output_deltas"][0]["candidate_term_deltas"]["slime"] == {
        "post_attention_residual": pytest.approx(0.15),
        "mlp": pytest.approx(-0.05),
    }


def test_shared_expert_gate_diagnostic_decomposes_weighted_delta():
    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(raw_values: list[float], weighted_values: list[float], gate_values: list[float]) -> dict:
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": 2,
                                "sample_indices": [0, 1, 2],
                                "components": [
                                    component(10, "shared_expert", raw_values),
                                    component(10, "shared_expert_gate_value", gate_values),
                                    component(10, "shared_expert_weighted", weighted_values),
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_shared_expert_gate.diagnose_shared_expert_gate(
        reference=artifact([2.0, -4.0, 0.5], [0.5, -1.0, 0.125], [0.25, 0.25, 0.25]),
        candidate=artifact([3.0, -2.0, 0.5], [1.5, -1.0, 0.25], [0.5, 0.5, 0.5]),
        layers=[10],
        top_n=3,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["direct_gate_record_count"] == 1
    record = report["records"][0]
    assert record["reference_gate"]["gate"] == pytest.approx(0.25)
    assert record["candidate_gate"]["gate"] == pytest.approx(0.5)
    assert record["gate_delta"] == pytest.approx(0.25)
    assert record["direct_gate"]["delta"]["max_abs"] == pytest.approx(0.25)
    assert record["direct_gate"]["candidate_direct_minus_inferred"]["max_abs"] == pytest.approx(0.0)
    assert record["direct_gate"]["captured_vs_recomputed_weighted_delta_diff"]["max_abs"] == pytest.approx(0.0)
    assert record["raw_delta"]["max_abs"] == pytest.approx(2.0)
    assert record["weighted_delta"]["max_abs"] == pytest.approx(1.0)
    assert record["captured_vs_recomputed_weighted_delta_diff"]["max_abs"] == pytest.approx(0.0)

    top = record["top_weighted_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["raw_delta"] == pytest.approx(1.0)
    assert top["weighted_delta"] == pytest.approx(1.0)
    assert top["reference_gate_times_raw_delta"] == pytest.approx(0.25)
    assert top["gate_delta_times_reference_raw"] == pytest.approx(0.5)
    assert top["interaction"] == pytest.approx(0.25)
    assert top["recomputed_delta"] == pytest.approx(1.0)
    assert top["captured_vs_recomputed_delta_diff"] == pytest.approx(0.0)


def test_shared_expert_raw_compute_diagnostic_recomputes_from_checkpoint(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    weight_keys = {
        "gate_proj": "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "up_proj": "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "down_proj": "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
    }
    weights = {
        weight_keys["gate_proj"]: torch.eye(2, dtype=torch.float32),
        weight_keys["up_proj"]: torch.eye(2, dtype=torch.float32),
        weight_keys["down_proj"]: torch.eye(2, dtype=torch.float32),
    }
    save_file(weights, model_path / "model-00001-of-00001.safetensors")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weight_keys.values(), "model-00001-of-00001.safetensors")}),
        encoding="utf-8",
    )

    def shared_raw(values: list[float]) -> list[float]:
        x = torch.tensor(values, dtype=torch.float32)
        return (torch.nn.functional.silu(x) * x).tolist()

    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(input_values: list[float]) -> dict:
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": 2,
                                "sample_indices": [0, 1],
                                "components": [
                                    component(0, "shared_expert_input", input_values),
                                    component(0, "shared_expert", shared_raw(input_values)),
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_shared_expert_raw_compute.diagnose_shared_expert_raw_compute(
        reference=artifact([1.0, 2.0]),
        candidate=artifact([1.5, 2.0]),
        model_path=model_path,
        layers=[0],
        policies=("fp32",),
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["fp32_candidate_captured_minus_policy_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["fp32_reference_captured_minus_policy_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["fp32_captured_delta_minus_same_policy_delta_max_abs"] == pytest.approx(0.0)
    record = report["records"][0]
    assert record["weight_keys"] == weight_keys
    assert record["input_delta"]["max_abs"] == pytest.approx(0.5)
    assert record["top_captured_raw_deltas"][0]["hidden_index"] == 0


def test_shared_expert_reference_input_diagnostic_conditions_on_reference_input(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    weight_keys = {
        "gate_proj": "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "up_proj": "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "down_proj": "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
    }
    weights = {
        weight_keys["gate_proj"]: torch.eye(2, dtype=torch.float32),
        weight_keys["up_proj"]: torch.eye(2, dtype=torch.float32),
        weight_keys["down_proj"]: torch.eye(2, dtype=torch.float32),
    }
    save_file(weights, model_path / "model-00001-of-00001.safetensors")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weight_keys.values(), "model-00001-of-00001.safetensors")}),
        encoding="utf-8",
    )

    def shared_raw(values: list[float]) -> list[float]:
        x = torch.tensor(values, dtype=torch.float32)
        return (torch.nn.functional.silu(x) * x).tolist()

    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(input_name: str, input_values: list[float], gate_values: list[float]) -> dict:
        raw = torch.tensor(shared_raw(input_values), dtype=torch.float32)
        gate = torch.tensor(gate_values, dtype=torch.float32)
        weighted = (raw * gate).tolist()
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": 7,
                                "sample_indices": [0, 1],
                                "components": [
                                    component(0, input_name, input_values),
                                    component(0, "shared_expert_gate_value", gate_values),
                                    component(0, "shared_expert", raw.tolist()),
                                    component(0, "shared_expert_weighted", weighted),
                                    component(0, "experts", [0.0, 0.0]),
                                    component(0, "mlp", weighted),
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_shared_expert_reference_input.diagnose_shared_expert_reference_input(
        reference=artifact("post_attention_norm", [1.0, 0.1], [0.25, 0.25]),
        candidate=artifact("shared_expert_input", [1.5, 0.1], [0.5, 0.5]),
        model_path=model_path,
        layers=[0],
        policy="fp32",
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["candidate_raw_minus_policy_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["reference_raw_minus_policy_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["captured_weighted_delta_minus_policy_delta_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["reference_input_reference_gate_delta_vs_reference_captured_max_abs"] == pytest.approx(0.0)
    record = report["records"][0]
    assert record["input_components"] == {"candidate": "shared_expert_input", "reference": "post_attention_norm"}
    assert record["gate_components"] == {
        "candidate": "shared_expert_gate_value",
        "reference": "shared_expert_gate_value",
    }
    assert record["input_delta"]["max_abs"] == pytest.approx(0.5)
    assert record["mlp_delta"]["max_abs"] == pytest.approx(record["captured_weighted_delta"]["max_abs"])
    top = record["top_captured_weighted_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["input_only_weighted_delta"] > 0
    assert top["gate_only_weighted_delta"] > 0
    assert top["mlp_delta"] == pytest.approx(top["captured_weighted_delta"])
    assert top["experts_delta"] == pytest.approx(0.0)
    top_mlp = record["top_mlp_deltas"][0]
    assert top_mlp["hidden_index"] == 0
    assert top_mlp["mlp_delta"] == pytest.approx(top_mlp["captured_weighted_delta"])
    assert top_mlp["experts_delta"] == pytest.approx(0.0)


def test_tensor_shared_expert_reference_input_diagnostic_conditions_on_reference_input(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    weight_keys = {
        "gate_proj": "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "up_proj": "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "down_proj": "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
    }
    weights = {
        weight_keys["gate_proj"]: torch.eye(2, dtype=torch.float32),
        weight_keys["up_proj"]: torch.eye(2, dtype=torch.float32),
        weight_keys["down_proj"]: torch.eye(2, dtype=torch.float32),
    }
    save_file(weights, model_path / "model-00001-of-00001.safetensors")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weight_keys.values(), "model-00001-of-00001.safetensors")}),
        encoding="utf-8",
    )

    def shared_raw(values: list[float]) -> torch.Tensor:
        x = torch.tensor(values, dtype=torch.float32)
        return torch.nn.functional.silu(x) * x

    def payload(input_component: str, input_values: list[float], gate_values: list[float]) -> dict:
        raw = shared_raw(input_values)
        gate = torch.tensor(gate_values, dtype=torch.float32)
        weighted = raw * gate
        return {
            f"model.layers.0.{input_component}": torch.tensor([[input_values]], dtype=torch.float32),
            "model.layers.0.shared_expert_gate_value": gate.reshape(1, 1, -1),
            "model.layers.0.shared_expert": raw.reshape(1, -1),
            "model.layers.0.shared_expert_weighted": weighted.reshape(1, 1, -1),
            "model.layers.0.experts": torch.zeros(1, 1, 2),
            "model.layers.0.mlp": weighted.reshape(1, 1, -1),
            "labels": torch.tensor([[7472]]),
        }

    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(payload("post_attention_norm", [1.0, 0.1], [0.25, 0.25]), reference_path)
    torch.save(payload("shared_expert_input", [1.5, 0.1], [0.5, 0.5]), candidate_path)

    report = diagnose_tensor_shared_expert_reference_input.diagnose_tensor_shared_expert_reference_input(
        reference_path=reference_path,
        candidate_path=candidate_path,
        model_path=model_path,
        layers=[0],
        row_selector="labels",
        policy="fp32",
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["missing_record_count"] == 0
    assert report["summary"]["candidate_raw_minus_policy_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["reference_raw_minus_policy_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["captured_weighted_delta_minus_policy_delta_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["reference_input_reference_gate_delta_vs_reference_captured_max_abs"] == pytest.approx(0.0)
    record = report["records"][0]
    assert record["input_components"] == {"candidate": "shared_expert_input", "reference": "post_attention_norm"}
    assert record["weight_keys"] == weight_keys
    assert record["input_delta"]["max_abs"] == pytest.approx(0.5)
    assert record["mlp_delta"]["max_abs"] == pytest.approx(record["captured_weighted_delta"]["max_abs"])
    top = record["top_captured_weighted_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["input_only_weighted_delta"] > 0
    assert top["gate_only_weighted_delta"] > 0
    assert top["mlp_delta"] == pytest.approx(top["captured_weighted_delta"])
    assert top["experts_delta"] == pytest.approx(0.0)
    top_mlp = record["top_mlp_deltas"][0]
    assert top_mlp["hidden_index"] == 0
    assert top_mlp["mlp_delta"] == pytest.approx(top_mlp["captured_weighted_delta"])
    assert top_mlp["experts_delta"] == pytest.approx(0.0)


def test_tensor_moe_routing_diagnostic_compares_reference_topk(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    gate_key = "model.language_model.layers.0.mlp.gate.weight"
    expert_gate_up_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    expert_down_key = "model.language_model.layers.0.mlp.experts.down_proj"
    gate_up_internal = torch.zeros(3, 3, 2, dtype=torch.float32)
    gate_up_internal[:, 0, 0] = 1.0
    gate_up_internal[:, 1, 1] = 1.0
    down_internal = torch.zeros(3, 1, 3, dtype=torch.float32)
    down_internal[:, 0, 0] = 1.0
    save_file(
        {
            gate_key: torch.eye(3, dtype=torch.float32),
            expert_gate_up_key: gate_up_internal.transpose(1, 2).contiguous(),
            expert_down_key: down_internal.transpose(1, 2).contiguous(),
        },
        model_path / "model-00001-of-00001.safetensors",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    gate_key: "model-00001-of-00001.safetensors",
                    expert_gate_up_key: "model-00001-of-00001.safetensors",
                    expert_down_key: "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    reference_input = torch.tensor([[3.0, 2.0, 0.0], [3.0, 2.0, 0.0]], dtype=torch.float32)
    candidate_input = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]], dtype=torch.float32)
    labels = torch.tensor([[11, 12]])

    def payload(values: torch.Tensor, experts: torch.Tensor) -> dict:
        return {
            "model.layers.0.post_attention_norm": values.reshape(1, 2, 3),
            "model.layers.0.experts": experts.reshape(1, 2, 3),
            "labels": labels,
        }

    def normalized_topk(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), k=2, dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True), ids

    reference_weights, reference_ids = normalized_topk(reference_input)
    candidate_weights, candidate_ids = normalized_topk(candidate_input)
    expert_weights = diagnose_tensor_moe_routing._load_expert_weights(model_path, 0)
    reference_experts = torch.stack(
        [
            diagnose_tensor_moe_routing._recompute_routed_experts(
                reference_input[row],
                expert_ids=reference_ids[row],
                expert_weights=reference_weights[row],
                weights=expert_weights,
                hidden_act="silu",
                policy="bf16",
            )
            for row in range(reference_input.shape[0])
        ]
    )
    candidate_experts = torch.stack(
        [
            diagnose_tensor_moe_routing._recompute_routed_experts(
                candidate_input[row],
                expert_ids=candidate_ids[row],
                expert_weights=candidate_weights[row],
                weights=expert_weights,
                hidden_act="silu",
                policy="bf16",
            )
            for row in range(candidate_input.shape[0])
        ]
    )
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    reference_topk_path = tmp_path / "reference-topk.pt"
    torch.save(payload(reference_input, reference_experts), reference_path)
    torch.save(payload(candidate_input, candidate_experts), candidate_path)
    torch.save(
        {
            "model.layers.0.mlp.topk": [
                reference_weights,
                reference_ids.to(torch.int32),
                reference_input.to(torch.bfloat16),
            ]
        },
        reference_topk_path,
    )

    report = diagnose_tensor_moe_routing.diagnose_tensor_moe_routing(
        reference_path=reference_path,
        candidate_path=candidate_path,
        reference_topk_path=reference_topk_path,
        model_path=model_path,
        layers=[0],
        row_selector="0,1",
        policy="bf16",
        top_n=2,
        recompute_experts=True,
    )

    assert report["summary"]["record_count"] == 2
    assert report["summary"]["missing_record_count"] == 0
    assert report["summary"]["candidate_topk_min_set_overlap"] == 1
    assert report["summary"]["candidate_topk_exact_order_match_count"] == 1
    assert report["summary"]["reference_topk_exact_order_match_count"] == 2
    first, second = report["records"]
    assert first["candidate_policy_vs_reference_captured_topk"]["exact_order_match"] is True
    assert second["candidate_policy_vs_reference_captured_topk"]["exact_order_match"] is False
    assert second["candidate_policy_vs_reference_captured_topk"]["set_overlap_count"] == 1
    assert second["experts_delta"]["max_abs"] > 0.0
    assert second["expert_recompute"]["candidate_policy_delta_minus_captured_experts_delta"]["max_abs"] == 0.0
    assert report["summary"]["expert_recompute_candidate_policy_delta_minus_captured_experts_delta_max_abs"] == 0.0
    assert second["gate_weight_key"] == gate_key
    assert second["expert_recompute"]["gate_up_key"] == expert_gate_up_key
    assert second["expert_recompute"]["down_key"] == expert_down_key


def test_post_attention_routing_sensitivity_chains_norm_topk_and_experts(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    norm_key = "model.language_model.layers.0.post_attention_layernorm.weight"
    gate_key = "model.language_model.layers.0.mlp.gate.weight"
    expert_gate_up_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    expert_down_key = "model.language_model.layers.0.mlp.experts.down_proj"
    gate_up_internal = torch.zeros(3, 3, 2, dtype=torch.float32)
    gate_up_internal[:, 0, 0] = 1.0
    gate_up_internal[:, 1, 1] = 1.0
    down_internal = torch.zeros(3, 1, 3, dtype=torch.float32)
    down_internal[:, 0, 0] = 1.0
    save_file(
        {
            norm_key: torch.zeros(3, dtype=torch.float32),
            gate_key: torch.eye(3, dtype=torch.float32),
            expert_gate_up_key: gate_up_internal.transpose(1, 2).contiguous(),
            expert_down_key: down_internal.transpose(1, 2).contiguous(),
        },
        model_path / "model-00001-of-00001.safetensors",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    norm_key: "model-00001-of-00001.safetensors",
                    gate_key: "model-00001-of-00001.safetensors",
                    expert_gate_up_key: "model-00001-of-00001.safetensors",
                    expert_down_key: "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    norm_eps = 1e-6
    multiplier = diagnose_tensor_rmsnorm_amplification._weight_multiplier(torch.zeros(3), "zero-centered")
    reference_residual = torch.tensor([[3.0, 2.0, 0.0], [3.0, 2.0, 0.0]], dtype=torch.float32)
    candidate_residual = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]], dtype=torch.float32)
    reference_norm = torch.stack(
        [diagnose_tensor_rmsnorm_amplification._bf16_rms_norm(row, multiplier, norm_eps) for row in reference_residual]
    )
    candidate_norm = torch.stack(
        [diagnose_tensor_rmsnorm_amplification._bf16_rms_norm(row, multiplier, norm_eps) for row in candidate_residual]
    )

    def normalized_topk(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), k=2, dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True), ids

    reference_weights, reference_ids = normalized_topk(reference_norm)
    candidate_weights, candidate_ids = normalized_topk(candidate_norm)
    expert_weights = diagnose_tensor_moe_routing._load_expert_weights(model_path, 0)
    reference_experts = torch.stack(
        [
            diagnose_tensor_moe_routing._recompute_routed_experts(
                reference_norm[row],
                expert_ids=reference_ids[row],
                expert_weights=reference_weights[row],
                weights=expert_weights,
                hidden_act="silu",
                policy="bf16",
            )
            for row in range(reference_norm.shape[0])
        ]
    )
    candidate_experts = torch.stack(
        [
            diagnose_tensor_moe_routing._recompute_routed_experts(
                candidate_norm[row],
                expert_ids=candidate_ids[row],
                expert_weights=candidate_weights[row],
                weights=expert_weights,
                hidden_act="silu",
                policy="bf16",
            )
            for row in range(candidate_norm.shape[0])
        ]
    )

    labels = torch.tensor([[11, 12]])

    def payload(residual: torch.Tensor, norm: torch.Tensor, experts: torch.Tensor) -> dict:
        zeros = torch.zeros_like(residual)
        return {
            "labels": labels,
            "model.layers.0.layer_input": zeros.reshape(1, 2, 3),
            "model.layers.0.attention": residual.reshape(1, 2, 3),
            "model.layers.0.post_attention_residual": residual.reshape(1, 2, 3),
            "model.layers.0.post_attention_norm": norm.reshape(1, 2, 3),
            "model.layers.0.experts": experts.reshape(1, 2, 3),
        }

    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    reference_topk_path = tmp_path / "reference-topk.pt"
    torch.save(payload(reference_residual, reference_norm, reference_experts), reference_path)
    torch.save(payload(candidate_residual, candidate_norm, candidate_experts), candidate_path)
    torch.save(
        {
            "model.layers.0.mlp.topk": [
                reference_weights,
                reference_ids.to(torch.int32),
                reference_norm.to(torch.bfloat16),
            ]
        },
        reference_topk_path,
    )

    report = diagnose_tensor_post_attention_routing_sensitivity.diagnose_tensor_post_attention_routing_sensitivity(
        reference_path=reference_path,
        candidate_path=candidate_path,
        reference_topk_path=reference_topk_path,
        model_path=model_path,
        layers=[0],
        row_selector="0,1",
        norm_eps=norm_eps,
        top_n=2,
        recompute_experts=True,
    )

    assert report["summary"]["record_count"] == 2
    assert report["summary"]["missing_record_count"] == 0
    assert report["summary"]["captured_topk_min_set_overlap"] == 1
    second = report["records"][1]
    assert second["captured_residual_delta_minus_bf16_add_delta"]["max_abs"] == pytest.approx(0.0)
    assert second["captured_norm_delta_minus_bf16_norm_delta"]["max_abs"] == pytest.approx(0.0)
    assert second["captured_norm_candidate_vs_reference_policy_topk"]["exact_order_match"] is False
    assert second["bf16_norm_candidate_vs_reference_policy_topk"]["exact_order_match"] is False
    assert second["expert_recompute"]["captured_norm"]["candidate_policy_delta_minus_captured_experts_delta"][
        "max_abs"
    ] == pytest.approx(0.0)
    assert second["expert_recompute"]["bf16_norm"]["candidate_policy_delta_minus_captured_experts_delta"][
        "max_abs"
    ] == pytest.approx(0.0)


def test_shared_expert_input_source_diagnostic_decomposes_norm_source(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    norm_key = "model.language_model.layers.0.post_attention_layernorm.weight"
    save_file({norm_key: torch.zeros(2, dtype=torch.float32)}, model_path / "model-00001-of-00001.safetensors")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {norm_key: "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def norm(values: list[float]) -> list[float]:
        vector = torch.tensor(values, dtype=torch.float32)
        output = vector * torch.rsqrt(vector.pow(2).mean(-1, keepdim=True) + 1e-6)
        return output.to(dtype=torch.bfloat16).float().tolist()

    def artifact(layer_input: list[float], attention: list[float]) -> dict:
        source = (torch.tensor(layer_input).to(torch.bfloat16) + torch.tensor(attention).to(torch.bfloat16)).to(
            torch.bfloat16
        )
        source_values = source.float().tolist()
        norm_values = norm(source_values)
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": 5,
                                "sample_indices": [0, 1],
                                "components": [
                                    component(0, "layer_input", layer_input),
                                    component(0, "attention", attention),
                                    component(0, "post_attention_residual", source_values),
                                    component(0, "post_attention_norm", norm_values),
                                    component(0, "shared_expert_input", norm_values),
                                ],
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_shared_expert_input_sources.diagnose_shared_expert_input_sources(
        reference=artifact([1.0, 2.0], [0.5, -0.5]),
        candidate=artifact([1.25, 2.0], [0.5, -0.25]),
        model_path=model_path,
        layers=[0],
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["captured_source_minus_bf16_terms_candidate_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["captured_norm_minus_recomputed_candidate_max_abs"] == pytest.approx(0.0)
    record = report["records"][0]
    assert record["shared_input_vs_post_attention_norm"]["candidate"]["max_abs"] == pytest.approx(0.0)
    assert record["post_attention_residual_delta"]["max_abs"] == pytest.approx(0.25)
    top = record["top_shared_expert_input_deltas"][0]
    assert top["hidden_index"] in {0, 1}
    assert {term["component"] for term in top["term_deltas"]} == {"layer_input", "attention"}


def test_residual_add_policy_diagnostic_reports_counterfactuals():
    def component(layer: int, name: str, values: list[float]) -> dict:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "layer": layer,
            "name": name,
            "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
            "max_abs": float(tensor.abs().max().item()),
            "mean": float(tensor.mean().item()),
            "sample_values": values,
        }

    def artifact(values: dict[tuple[int, str], list[float]]) -> dict:
        components = [
            component(layer, name, component_values)
            for (layer, name), component_values in sorted(values.items(), key=lambda item: (item[0][0], item[0][1]))
        ]
        return {
            "samples": [
                {
                    "trace_id": "trace-a",
                    "per_token": [
                        {
                            "position": 0,
                            "token_id": 7472,
                            "xorl_hidden_component_summary": {
                                "component_count": len(components),
                                "sample_indices": [0, 1],
                                "components": components,
                            },
                        }
                    ],
                }
            ]
        }

    report = diagnose_residual_add_policy.diagnose_residual_add_policy(
        reference=artifact(
            {
                (10, "post_attention_residual"): [1.0, 2.0],
                (10, "mlp"): [0.25, -0.125],
                (10, "layer_output"): [1.25, 1.875],
            }
        ),
        candidate=artifact(
            {
                (10, "post_attention_residual"): [1.5, 2.0],
                (10, "mlp"): [0.25, -0.5],
                (10, "layer_output"): [1.75, 1.5],
            }
        ),
        layers=[10],
        equations=["layer_output"],
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    record = report["records"][0]
    assert record["captured_delta"]["max_abs"] == pytest.approx(0.5)
    assert record["policy_stats"]["candidate_policy_vs_reference_captured"]["fp32"]["max_abs"] == pytest.approx(0.5)
    assert record["policy_stats"]["same_policy_delta"]["bf16_pairwise"]["max_abs"] == pytest.approx(0.5)
    assert record["policy_stats"]["candidate_captured_minus_policy"]["bf16_pairwise"]["max_abs"] == pytest.approx(0.0)
    top = record["top_captured_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["captured_delta"] == pytest.approx(0.5)
    assert top["candidate_policy_vs_reference_captured_delta"]["fp32"] == pytest.approx(0.5)
    assert top["same_policy_delta"]["fp32_to_bf16"] == pytest.approx(0.5)
    assert [term["component"] for term in top["term_deltas"]] == ["post_attention_residual", "mlp"]


def test_sglang_debug_dump_to_artifact_builds_hidden_component_schema(tmp_path):
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    dump = {
        "model.embed_tokens": hidden,
        "model.layers.0.layer_input": hidden + 10,
        "model.layers.0.input_layernorm": hidden + 20,
        "model.layers.0.linear_attn.out_proj": hidden + 30,
        "model.layers.0.post_attention_layernorm": hidden + 40,
        "model.layers.0.mlp": hidden + 50,
        "model.layers.0.mlp.experts": hidden + 52,
        "model.layers.0.mlp.shared_expert": hidden + 54,
        "model.layers.0.mlp.shared_expert_gate": torch.zeros(5, 1),
        "model.layers.0": [hidden + 55, hidden + 60],
        "model.norm": hidden + 70,
    }
    dump_dir = tmp_path / "dump" / "TP0_PP0_Rank0_pid123"
    dump_dir.mkdir(parents=True)
    dump_file = dump_dir / "Pass00000.pt"
    torch.save(dump, dump_file)

    trace = {
        "trace_id": "trace-a",
        "prompt_ids": [1, 2, 3],
        "prompt_len": 3,
        "output_ids": [7472],
        "gen_len": 1,
        "full_ids": [1, 2, 3, 7472],
        "sglang_logprobs": [-2.5],
    }
    loaded_file, loaded_dump = sglang_debug_dump_to_artifact.load_sglang_dump(tmp_path / "dump")
    artifact = sglang_debug_dump_to_artifact.build_sglang_debug_artifact(
        traces=[trace],
        dump=loaded_dump,
        dump_file=loaded_file,
        hidden_sample_indices="1,3",
        component_layers="0",
        num_layers=2,
        hidden_dim=4,
        request_metadata={"trace_id": "trace-a", "returned_logprobs": [-2.25]},
    )

    assert loaded_file == dump_file
    assert artifact["config"]["request_metadata"]["trace_id"] == "trace-a"
    token = artifact["samples"][0]["per_token"][0]
    assert token["diagnostic_row_index"] == 2
    assert token["xorl_logprob"] == pytest.approx(-2.25)
    assert token["xorl_hidden_state_summary"]["sample_indices"] == [1, 3]
    assert token["xorl_hidden_state_summary"]["expected_layer_count"] == 3
    assert token["xorl_hidden_state_summary"]["layers"][0]["index"] == 0
    assert token["xorl_hidden_state_summary"]["layers"][0]["sample_values"] == pytest.approx([9.0, 11.0])
    assert token["xorl_hidden_state_summary"]["layers"][1]["index"] == 1
    assert token["xorl_hidden_state_summary"]["layers"][1]["sample_values"] == pytest.approx([133.0, 137.0])
    assert token["xorl_hidden_state_summary"]["layers"][2]["index"] == 2
    assert token["xorl_hidden_state_summary"]["layers"][2]["sample_values"] == pytest.approx([79.0, 81.0])

    components = token["xorl_hidden_component_summary"]["components"]
    by_name = {component["name"]: component for component in components}
    assert set(by_name) == {
        "layer_input",
        "input_norm",
        "attention",
        "post_attention_norm",
        "raw_mlp",
        "mlp",
        "experts",
        "shared_expert_input",
        "shared_expert_gate_value",
        "shared_expert",
        "shared_expert_weighted",
        "layer_output",
    }
    assert by_name["layer_input"]["sample_values"] == pytest.approx([9.0, 11.0])
    assert by_name["input_norm"]["sample_values"] == pytest.approx([29.0, 31.0])
    assert by_name["raw_mlp"]["sample_values"] == pytest.approx([59.0, 61.0])
    assert by_name["mlp"]["sample_values"] == pytest.approx([92.5, 95.5])
    assert by_name["experts"]["sample_values"] == pytest.approx([61.0, 63.0])
    assert by_name["shared_expert_input"]["sample_values"] == pytest.approx([49.0, 51.0])
    assert by_name["shared_expert_gate_value"]["sample_values"] == pytest.approx([0.5, 0.5])
    assert by_name["shared_expert"]["sample_values"] == pytest.approx([63.0, 65.0])
    assert by_name["shared_expert_weighted"]["sample_values"] == pytest.approx([31.5, 32.5])
    assert by_name["layer_output"]["sample_values"] == pytest.approx([133.0, 137.0])

    report = compare_hidden_component_artifacts.compare_hidden_component_artifacts(
        reference=artifact,
        candidate=artifact,
        top_n=1,
    )
    assert report["summary"]["matched_component_count"] == 12
    assert report["summary"]["matched_hidden_state_layer_count"] == 3
    assert report["summary"]["sample_max_abs_delta"]["max"] == pytest.approx(0.0)
    assert report["summary"]["hidden_state_sample_max_abs_delta"]["max"] == pytest.approx(0.0)


def test_sglang_layer_input_prefers_returned_input_layernorm_residual():
    hidden = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    forward_residual = torch.tensor([[10.0, 20.0]], dtype=torch.bfloat16)
    returned_residual = torch.tensor([[100.0, 200.0]], dtype=torch.bfloat16)
    dump = {
        "model.layers.1.layer_input": hidden,
        "model.layers.1.forward_input.residual": forward_residual,
        "model.layers.1.input_layernorm.residual": returned_residual,
    }

    tensor = sglang_debug_dump_to_artifact._layer_input_tensor(
        dump,
        "model.layers.1.layer_input",
        hidden_dim=2,
    )

    torch.testing.assert_close(tensor, returned_residual)


def test_prepare_sglang_component_tensor_dump_tp_merges_full_components(tmp_path):
    root = tmp_path / "dump"
    rank0 = root / "TP0_PP0_Rank0_pid1"
    rank1 = root / "TP1_PP0_Rank1_pid2"
    rank0.mkdir(parents=True)
    rank1.mkdir(parents=True)
    dump0 = {
        "model.embed_tokens": torch.zeros(1, 4),
        "model.layers.10.layer_input": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "model.layers.10.forward_input.residual": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "model.layers.10.linear_attn": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "model.layers.10.post_attention_layernorm": torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        "model.layers.10.mlp.experts": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "model.layers.10.mlp.shared_expert": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "model.layers.10.mlp.shared_expert_gate": torch.zeros(1, 1),
    }
    dump1 = {
        "model.embed_tokens": torch.zeros(1, 4),
        "model.layers.10.layer_input": torch.tensor([[0.5, 1.0, 1.5, 2.0]]),
        "model.layers.10.forward_input.residual": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "model.layers.10.linear_attn": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        "model.layers.10.post_attention_layernorm": torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        "model.layers.10.mlp.experts": torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
        "model.layers.10.mlp.shared_expert": torch.tensor([[0.0, 0.0, 3.0, 0.0]]),
        "model.layers.10.mlp.shared_expert_gate": torch.zeros(1, 1),
    }
    torch.save(dump0, rank0 / "Pass00000.pt")
    torch.save(dump1, rank1 / "Pass00000.pt")

    output = tmp_path / "sglang-components.pt"
    metadata = prepare_sglang_component_tensor_dump.prepare_sglang_component_tensor_dump(
        dump_path=root,
        output_path=output,
        layers=[10],
        components=[
            "layer_input",
            "attention",
            "shared_expert",
            "shared_expert_gate_value",
            "shared_expert_weighted",
            "experts",
        ],
        hidden_dim=4,
    )

    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert metadata["hidden_dim"] == 4
    assert metadata["missing"] == []
    torch.testing.assert_close(payload["model.layers.10.layer_input"], torch.tensor([[11.5, 23.0, 34.5, 46.0]]))
    torch.testing.assert_close(payload["model.layers.10.attention"], torch.tensor([[1.1, 1.2, 1.3, 1.4]]))
    torch.testing.assert_close(payload["model.layers.10.experts"], torch.tensor([[1.0, 2.0, 0.0, 0.0]]))
    torch.testing.assert_close(payload["model.layers.10.shared_expert"], torch.tensor([[0.0, 1.0, 3.0, 0.0]]))
    torch.testing.assert_close(payload["model.layers.10.shared_expert_gate_value"], torch.full((1, 4), 0.5))
    torch.testing.assert_close(
        payload["model.layers.10.shared_expert_weighted"],
        torch.tensor([[0.0, 0.5, 1.5, 0.0]]),
    )


def test_compare_component_tensor_dumps_reports_label_rows_and_extra_keys(tmp_path):
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.10.layer_input": torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        "model.layers.10.mlp": torch.zeros(3, 3),
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[-100, 7472, -100, -100]]),
        "model.layers.10.layer_input": torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.5, 5.0, 6.0],
                    [7.0, 10.0, 9.0],
                    [100.0, 100.0, 100.0],
                ]
            ]
        ),
        "model.layers.10.mlp": torch.zeros(1, 4, 3),
        "model.layers.10.shared_expert_weighted": torch.ones(1, 4, 3),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)

    report = compare_component_tensor_dumps.compare_component_tensor_dumps(
        reference_path=reference_path,
        candidate_path=candidate_path,
        row_selector="labels,label+1",
        top_n=3,
    )

    assert report["summary"]["common_key_count"] == 2
    assert report["summary"]["extra_candidate_keys"] == ["model.layers.10.shared_expert_weighted"]
    assert report["summary"]["label_rows"] == [1]
    assert report["summary"]["label_tokens"] == [7472]
    top_component = report["summary"]["top_components_by_max_abs"][0]
    assert top_component["key"] == "model.layers.10.layer_input"
    assert top_component["max_abs"] == pytest.approx(2.0)
    assert top_component["compared_rows"] == 3
    assert top_component["label_rows"][0]["row"] == 1
    assert top_component["label_rows"][0]["max_abs"] == pytest.approx(0.5)
    assert report["summary"]["selected_rows_by_max_abs"][0]["row"] == 2
    assert report["summary"]["selected_rows_by_max_abs"][0]["max_abs"] == pytest.approx(2.0)


def test_compare_component_tensor_dumps_can_ignore_padded_tail_after_last_label(tmp_path):
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.0.layer_input": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[-100, 7472, -100, -100]]),
        "model.layers.0.layer_input": torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.25, 4.0],
                    [500.0, 600.0],
                    [700.0, 800.0],
                ]
            ]
        ),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)

    unrestricted = compare_component_tensor_dumps.compare_component_tensor_dumps(
        reference_path=reference_path,
        candidate_path=candidate_path,
        row_selector="labels,label+1",
        top_n=2,
    )
    causal = compare_component_tensor_dumps.compare_component_tensor_dumps(
        reference_path=reference_path,
        candidate_path=candidate_path,
        row_selector="labels,label+1",
        top_n=2,
        causal_prefix_only=True,
    )

    unrestricted_component = unrestricted["summary"]["top_components_by_max_abs"][0]
    causal_component = causal["summary"]["top_components_by_max_abs"][0]
    assert unrestricted_component["compared_rows"] == 4
    assert unrestricted_component["max_abs"] == pytest.approx(792.0)
    assert causal["config"]["causal_prefix_only"] is True
    assert causal["config"]["causal_prefix_row_count"] == 2
    assert causal_component["compared_rows"] == 2
    assert causal_component["max_abs"] == pytest.approx(0.25)
    assert causal_component["label_rows"][0]["row"] == 1
    assert causal_component["selected_rows"] == [{"row": 1, "max_abs": 0.25, "mean_abs": 0.125, "hidden_index": 0}]


def test_diagnose_tensor_source_terms_reports_scored_row_block_flow(tmp_path):
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.10.layer_input": torch.tensor([[1.0, 2.0], [10.0, 20.0]]),
        "model.layers.10.attention": torch.tensor([[0.5, -0.5], [1.0, 2.0]]),
        "model.layers.10.post_attention_residual": torch.tensor([[1.5, 1.5], [11.0, 22.0]]),
        "model.layers.10.mlp": torch.tensor([[0.25, 0.25], [0.5, 0.5]]),
        "model.layers.10.layer_output": torch.tensor([[1.75, 1.75], [11.5, 22.5]]),
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[7472, -100]]),
        "model.layers.10.layer_input": torch.tensor([[[1.25, 1.75], [10.0, 20.0]]]),
        "model.layers.10.attention": torch.tensor([[[0.75, -0.25], [1.0, 2.0]]]),
        "model.layers.10.post_attention_residual": torch.tensor([[[2.0, 1.5], [11.0, 22.0]]]),
        "model.layers.10.mlp": torch.tensor([[[0.35, 0.15], [0.5, 0.5]]]),
        "model.layers.10.layer_output": torch.tensor([[[2.35, 1.65], [11.5, 22.5]]]),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)

    report = diagnose_tensor_source_terms.diagnose_tensor_source_terms(
        reference_path=reference_path,
        candidate_path=candidate_path,
        layers=[10],
        equations=["attention_residual", "layer_output", "block_output"],
        row_selector="labels",
        top_n=2,
    )

    assert report["summary"]["label_rows"] == [0]
    assert report["summary"]["label_tokens"] == [7472]
    assert report["summary"]["record_count"] == 3
    assert report["summary"]["missing_record_count"] == 0
    block = next(record for record in report["records"] if record["equation"] == "block_output")
    assert block["row"] == 0
    assert block["captured_output_delta"]["max_abs"] == pytest.approx(0.6)
    assert block["primary_delta"]["max_abs"] == pytest.approx(0.25)
    assert block["secondary_delta_sum"]["max_abs"] == pytest.approx(0.35)
    assert block["bf16_inherited_only_delta"]["max_abs"] == pytest.approx(0.25)
    assert block["bf16_local_only_delta"]["max_abs"] == pytest.approx(0.34375)
    top = block["top_output_deltas"][0]
    assert top["hidden_index"] == 0
    assert top["output_delta"] == pytest.approx(0.6)
    assert top["primary_delta"] == pytest.approx(0.25)
    assert top["secondary_delta_sum"] == pytest.approx(0.35)
    assert [term["component"] for term in top["term_deltas"]] == ["layer_input", "attention", "mlp"]


def test_diagnose_tensor_input_sensitivity_reports_norm_and_consumer_pairs(tmp_path):
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.10.layer_input": torch.tensor([[1.0, 2.0], [10.0, 20.0]]),
        "model.layers.10.input_norm": torch.tensor([[2.0, 4.0], [20.0, 40.0]]),
        "model.layers.10.attention": torch.tensor([[0.5, -0.5], [1.0, 2.0]]),
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[7472, -100]]),
        "model.layers.10.layer_input": torch.tensor([[[1.25, 1.75], [10.0, 20.0]]]),
        "model.layers.10.input_norm": torch.tensor([[[3.0, 3.5], [20.0, 40.0]]]),
        "model.layers.10.attention": torch.tensor([[[0.75, -0.25], [1.0, 2.0]]]),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)

    report = diagnose_tensor_input_sensitivity.diagnose_tensor_input_sensitivity(
        reference_path=reference_path,
        candidate_path=candidate_path,
        layers=[10],
        pairs=["input_norm", "attention"],
        row_selector="labels",
        top_n=2,
    )

    assert report["summary"]["label_rows"] == [0]
    assert report["summary"]["label_tokens"] == [7472]
    assert report["summary"]["record_count"] == 2
    assert report["summary"]["missing_record_count"] == 0
    input_norm = next(record for record in report["records"] if record["pair"] == "input_norm")
    assert input_norm["input_component"] == "layer_input"
    assert input_norm["output_component"] == "input_norm"
    assert input_norm["input_delta"]["max_abs"] == pytest.approx(0.25)
    assert input_norm["output_delta"]["max_abs"] == pytest.approx(1.0)
    assert input_norm["aggregate_amplification"]["max_abs_ratio"] == pytest.approx(4.0)
    assert input_norm["top_output_deltas"][0]["hidden_index"] == 0
    assert input_norm["top_output_deltas"][0]["same_coordinate_abs_output_to_input_ratio"] == pytest.approx(4.0)
    attention = next(record for record in report["records"] if record["pair"] == "attention")
    assert attention["input_component"] == "input_norm"
    assert attention["output_component"] == "attention"
    assert attention["input_delta"]["max_abs"] == pytest.approx(1.0)
    assert attention["output_delta"]["max_abs"] == pytest.approx(0.25)


def test_diagnose_tensor_layer_chain_reports_nonlocal_attention_source(tmp_path):
    reference_prev_input = torch.tensor([[1.0, 2.0, 3.0]])
    reference_prev_attention = torch.tensor([[0.1, 0.2, 0.3]])
    reference_prev_mlp = torch.tensor([[0.01, 0.02, 0.03]])
    reference_prev_output = (reference_prev_input + reference_prev_attention + reference_prev_mlp).to(torch.bfloat16)
    candidate_prev_input = torch.tensor([[[1.25, 2.0, 3.0]]])
    candidate_prev_attention = torch.tensor([[[0.35, 0.2, 0.3]]])
    candidate_prev_mlp = torch.tensor([[[0.01, 0.02, 0.13]]])
    candidate_prev_output = (candidate_prev_input + candidate_prev_attention + candidate_prev_mlp).to(torch.bfloat16)
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.8.layer_input": reference_prev_input,
        "model.layers.8.attention": reference_prev_attention,
        "model.layers.8.mlp": reference_prev_mlp,
        "model.layers.8.layer_output": reference_prev_output,
        "model.layers.9.layer_input": reference_prev_output,
        "model.layers.9.input_norm": torch.tensor([[10.0, 20.0, 30.0]]),
        "model.layers.9.attention": torch.tensor([[0.0, 0.0, 0.0]]),
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[7472]]),
        "model.layers.8.layer_input": candidate_prev_input,
        "model.layers.8.attention": candidate_prev_attention,
        "model.layers.8.mlp": candidate_prev_mlp,
        "model.layers.8.layer_output": candidate_prev_output,
        "model.layers.9.layer_input": candidate_prev_output,
        "model.layers.9.input_norm": torch.tensor([[[15.0, 20.0, 31.0]]]),
        "model.layers.9.attention": torch.tensor([[[0.1, 0.5, 0.2]]]),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)

    report = diagnose_tensor_layer_chain.diagnose_tensor_layer_chain(
        reference_path=reference_path,
        candidate_path=candidate_path,
        source_layer=8,
        target_layer=9,
        row_selector="labels",
        top_n=3,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["missing_record_count"] == 0
    assert report["summary"]["label_rows"] == [0]
    record = report["records"][0]
    assert record["handoff"]["candidate_prev_output_minus_target_input"]["max_abs"] == pytest.approx(0.0)
    assert record["handoff"]["reference_prev_output_minus_target_input"]["max_abs"] == pytest.approx(0.0)
    assert record["handoff"]["delta_equivalence_prev_output_minus_target_input"]["max_abs"] == pytest.approx(0.0)
    assert record["source_layer_block_output"]["captured_output_delta"]["max_abs"] == pytest.approx(0.5)
    assert record["target_input_norm"]["output_delta"]["max_abs"] == pytest.approx(5.0)
    assert record["target_attention"]["output_delta"]["max_abs"] == pytest.approx(0.5)
    top_attention = record["target_attention"]["top_output_deltas"][0]
    assert top_attention["hidden_index"] == 1
    assert top_attention["target_input_norm_delta"] == pytest.approx(0.0)
    assert top_attention["same_coordinate_input_norm_abs_below_threshold"] is True
    assert top_attention["same_coordinate_abs_attention_to_input_norm_ratio"] is None
    assert report["summary"]["top_attention_records"][0]["top_hidden_index"] == 1
    assert report["summary"]["top_attention_records"][0]["top_same_coordinate_input_norm_delta"] == pytest.approx(0.0)


def test_diagnose_qwen36_gdn_input_sensitivity_reports_nonlocal_projection(monkeypatch, tmp_path):
    out_proj = torch.eye(4)
    out_proj[2] = torch.tensor([3.0, 0.0, 0.0, 0.0])

    def fake_components(hidden_states, **kwargs):
        module = kwargs.get("module")
        weights = kwargs.get("weights")
        weight = module.out_proj_weight if module is not None else weights.out_proj
        final = hidden_states.float().matmul(weight.t()).to(hidden_states.dtype)
        components = {name: hidden_states.clone() for name in compare_qwen36_gdn_parity.GDN_COMPONENT_NAMES}
        components["normed"] = hidden_states.clone()
        components["final"] = final
        return components

    monkeypatch.setattr(
        diagnose_qwen36_gdn_input_sensitivity,
        "load_gdn_config",
        lambda _path: SimpleNamespace(layer_types=["linear_attention"] * 4),
    )
    monkeypatch.setattr(
        diagnose_qwen36_gdn_input_sensitivity, "select_linear_attention_layer", lambda _config, layer: layer
    )
    monkeypatch.setattr(
        diagnose_qwen36_gdn_input_sensitivity,
        "load_gdn_weights",
        lambda _path, _layer: SimpleNamespace(out_proj=out_proj),
    )
    monkeypatch.setattr(diagnose_qwen36_gdn_input_sensitivity, "_to_device", lambda weights, **_kwargs: weights)
    monkeypatch.setattr(
        diagnose_qwen36_gdn_input_sensitivity,
        "build_xorl_gdn",
        lambda _config, weights, **_kwargs: SimpleNamespace(out_proj_weight=weights.out_proj),
    )
    monkeypatch.setattr(diagnose_qwen36_gdn_input_sensitivity, "import_sglang_fla_helpers", lambda _path: object())
    monkeypatch.setattr(diagnose_qwen36_gdn_input_sensitivity, "xorl_style_components", fake_components)
    monkeypatch.setattr(diagnose_qwen36_gdn_input_sensitivity, "sglang_style_components", fake_components)

    reference_input_norm = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    candidate_input_norm = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    reference_attention = reference_input_norm.float().matmul(out_proj.t()).to(torch.bfloat16)
    candidate_attention = candidate_input_norm.float().matmul(out_proj.t()).to(torch.bfloat16)
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.2.input_norm": reference_input_norm,
        "model.layers.2.attention": reference_attention,
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[-100, 7472, -100]]),
        "model.layers.2.input_norm": candidate_input_norm.unsqueeze(0),
        "model.layers.2.attention": candidate_attention.unsqueeze(0),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)
    rank_dump_path = tmp_path / "sglang_dump"
    rank0_dir = rank_dump_path / "TP0_PP0_Rank0_pid1"
    rank1_dir = rank_dump_path / "TP1_PP0_Rank1_pid2"
    rank0_dir.mkdir(parents=True)
    rank1_dir.mkdir(parents=True)
    reference_hidden = reference_input_norm.unsqueeze(0)
    partials = diagnose_qwen36_gdn_input_sensitivity._tp_sharded_out_proj_partials(
        reference_hidden,
        out_proj_weight=out_proj,
        tp_size=2,
    )
    torch.save({"model.layers.2.linear_attn.out_proj": partials[0][0]}, rank0_dir / "Pass00001.pt")
    torch.save({"model.layers.2.linear_attn.out_proj": partials[1][0]}, rank1_dir / "Pass00001.pt")

    report = diagnose_qwen36_gdn_input_sensitivity.diagnose_qwen36_gdn_input_sensitivity(
        reference_path=reference_path,
        candidate_path=candidate_path,
        layers=[2],
        model_path="/unused",
        sglang_path="/unused",
        row_selector="labels",
        device="cpu",
        top_n=2,
        tp_out_proj_size=2,
        sglang_rank_dump_path=rank_dump_path,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["missing_record_count"] == 0
    assert report["summary"]["label_rows"] == [1]
    assert report["summary"]["captured_attention_delta_max_abs"] == pytest.approx(3.0)
    top_record = report["summary"]["top_records_by_captured_attention_delta"][0]
    assert top_record["top_captured_hidden_index"] == 2
    assert top_record["xorl_final_minus_captured_max_abs"] == pytest.approx(0.0)
    record = report["layers"][0]["records"][0]
    assert record["xorl_candidate_minus_sglang_reference_final_delta"]["max_abs"] == pytest.approx(3.0)
    assert record["xorl_candidate_minus_sglang_reference_final_delta_minus_captured_attention_delta"][
        "max_abs"
    ] == pytest.approx(0.0)
    assert record["xorl_candidate_final_minus_candidate_captured_attention"]["max_abs"] == pytest.approx(0.0)
    assert record["xorl_reference_final_minus_reference_captured_attention"]["max_abs"] == pytest.approx(0.0)
    assert record["sglang_candidate_final_minus_candidate_captured_attention"]["max_abs"] == pytest.approx(0.0)
    assert record["sglang_reference_final_minus_reference_captured_attention"]["max_abs"] == pytest.approx(0.0)
    assert record["xorl_tp2_candidate_final_minus_candidate_captured_attention"]["max_abs"] == pytest.approx(0.0)
    assert record["xorl_tp2_reference_final_minus_reference_captured_attention"]["max_abs"] == pytest.approx(0.0)
    assert report["summary"]["xorl_tp2_candidate_final_minus_candidate_captured_attention_max_abs"] == pytest.approx(
        0.0
    )
    assert report["summary"]["rank_local_tp_recomputed_partial_minus_dump_out_proj_max_abs"] == pytest.approx(0.0)
    assert report["summary"]["rank_local_tp_dump_out_proj_sum_minus_reference_captured_attention_max_abs"] == (
        pytest.approx(0.0)
    )
    rank_local = report["layers"][0]["rank_local_tp_report"]
    assert rank_local["rank_count"] == 2
    assert rank_local["rows"][0]["ranks"][0]["recomputed_partial_minus_dump_out_proj"]["max_abs"] == pytest.approx(0.0)
    xorl_row = report["layers"][0]["style_row_reports"]["xorl"]["1"]
    assert xorl_row["final_delta"]["max_abs"] == pytest.approx(3.0)
    top_final = xorl_row["top_final_deltas"][0]
    assert top_final["output_hidden_index"] == 2
    assert top_final["same_coordinate_input_norm_delta"] == pytest.approx(0.0)
    assert top_final["same_coordinate_input_norm_abs_below_threshold"] is True
    contributors = top_final["out_proj_normed_delta_contributors"]["top_contributors"]
    assert contributors[0]["flat_normed_index"] == 0
    assert contributors[0]["contribution"] == pytest.approx(3.0)


def test_diagnose_qwen36_full_attention_sensitivity_reports_nonlocal_projection(monkeypatch, tmp_path):
    out_proj = torch.eye(4)
    out_proj[2] = torch.tensor([3.0, 0.0, 0.0, 0.0])

    def fake_components(hidden_states, **kwargs):
        weights = kwargs["weights"]
        seq_len = hidden_states.shape[1]
        final = hidden_states.float().matmul(weights.o_proj.t()).to(hidden_states.dtype)
        components = {
            name: hidden_states.clone()
            for name in diagnose_qwen36_full_attention_sensitivity.FULL_ATTENTION_SEQUENCE_COMPONENTS
        }
        components["gated"] = hidden_states.clone()
        components["final"] = final
        components["attention_scores"] = torch.zeros((1, 1, seq_len, seq_len), dtype=hidden_states.dtype)
        components["attention_probs"] = torch.zeros((1, 1, seq_len, seq_len), dtype=hidden_states.dtype)
        return components

    monkeypatch.setattr(
        diagnose_qwen36_full_attention_sensitivity,
        "load_full_attention_config",
        lambda _path: SimpleNamespace(layer_types=["full_attention"] * 4),
    )
    monkeypatch.setattr(
        diagnose_qwen36_full_attention_sensitivity, "select_full_attention_layer", lambda _config, layer: layer
    )
    monkeypatch.setattr(
        diagnose_qwen36_full_attention_sensitivity,
        "load_full_attention_weights",
        lambda _path, _layer: SimpleNamespace(o_proj=out_proj),
    )
    monkeypatch.setattr(
        diagnose_qwen36_full_attention_sensitivity,
        "move_full_attention_weights",
        lambda weights, **_kwargs: weights,
    )
    monkeypatch.setattr(
        diagnose_qwen36_full_attention_sensitivity,
        "RotaryEmbedding",
        lambda *_args, **_kwargs: SimpleNamespace(to=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setattr(diagnose_qwen36_full_attention_sensitivity, "full_attention_components", fake_components)

    reference_input_norm = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    candidate_input_norm = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    reference_attention = reference_input_norm.float().matmul(out_proj.t()).to(torch.bfloat16)
    candidate_attention = candidate_input_norm.float().matmul(out_proj.t()).to(torch.bfloat16)
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.2.input_norm": reference_input_norm,
        "model.layers.2.attention": reference_attention,
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[-100, 7472, -100]]),
        "model.layers.2.input_norm": candidate_input_norm.unsqueeze(0),
        "model.layers.2.attention": candidate_attention.unsqueeze(0),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)

    report = diagnose_qwen36_full_attention_sensitivity.diagnose_qwen36_full_attention_sensitivity(
        reference_path=reference_path,
        candidate_path=candidate_path,
        layers=[2],
        model_path="/unused",
        row_selector="labels",
        device="cpu",
        top_n=2,
    )

    assert report["summary"]["record_count"] == 1
    assert report["summary"]["missing_record_count"] == 0
    assert report["summary"]["label_rows"] == [1]
    assert report["summary"]["captured_attention_delta_max_abs"] == pytest.approx(3.0)
    top_record = report["summary"]["top_records_by_captured_attention_delta"][0]
    assert top_record["top_captured_hidden_index"] == 2
    assert top_record["computed_final_minus_captured_max_abs"] == pytest.approx(0.0)
    row = report["layers"][0]["row_reports"]["1"]
    assert row["final_delta"]["max_abs"] == pytest.approx(3.0)
    top_final = row["top_final_deltas"][0]
    assert top_final["output_hidden_index"] == 2
    assert top_final["same_coordinate_input_norm_delta"] == pytest.approx(0.0)
    assert top_final["same_coordinate_input_norm_abs_below_threshold"] is True
    contributors = top_final["out_proj_gated_delta_contributors"]["top_contributors"]
    assert contributors[0]["flat_gated_index"] == 0
    assert contributors[0]["contribution"] == pytest.approx(3.0)


def test_diagnose_tensor_rmsnorm_amplification_reports_coordinate_scale_and_closure(tmp_path):
    weight = torch.tensor([0.0, 0.0])
    multiplier = diagnose_tensor_rmsnorm_amplification._weight_multiplier(weight, "zero-centered")
    norm_eps = 1e-6
    reference_source = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    candidate_source = torch.tensor([[1.5, 2.0], [10.0, 20.0]])
    reference_norm = torch.stack(
        [diagnose_tensor_rmsnorm_amplification._bf16_rms_norm(row, multiplier, norm_eps) for row in reference_source]
    )
    candidate_norm = torch.stack(
        [diagnose_tensor_rmsnorm_amplification._bf16_rms_norm(row, multiplier, norm_eps) for row in candidate_source]
    )
    reference = {
        "__metadata__": {"source": "reference"},
        "model.layers.10.layer_input": reference_source,
        "model.layers.10.input_norm": reference_norm,
    }
    candidate = {
        "__metadata__": {"rank": 0},
        "labels": torch.tensor([[7472, -100]]),
        "model.layers.10.layer_input": candidate_source.unsqueeze(0),
        "model.layers.10.input_norm": candidate_norm.unsqueeze(0),
    }
    reference_path = tmp_path / "reference.pt"
    candidate_path = tmp_path / "candidate.pt"
    model_path = tmp_path / "model.pt"
    torch.save(reference, reference_path)
    torch.save(candidate, candidate_path)
    torch.save({"model.layers.10.input_layernorm.weight": weight}, model_path)

    report = diagnose_tensor_rmsnorm_amplification.diagnose_tensor_rmsnorm_amplification(
        reference_path=reference_path,
        candidate_path=candidate_path,
        model_path=model_path,
        layers=[10],
        boundaries=["input_norm"],
        row_selector="labels",
        norm_eps=norm_eps,
        top_n=2,
    )

    assert report["summary"]["label_rows"] == [0]
    assert report["summary"]["label_tokens"] == [7472]
    assert report["summary"]["record_count"] == 1
    assert report["summary"]["missing_record_count"] == 0
    record = report["records"][0]
    assert record["boundary"] == "input_norm"
    assert record["source_component"] == "layer_input"
    assert record["output_component"] == "input_norm"
    assert record["source_delta"]["max_abs"] == pytest.approx(0.5)
    assert record["coordinate_only_delta"]["max_abs"] > 0.0
    assert record["scale_only_delta"]["max_abs"] > 0.0
    assert record["candidate_captured_minus_bf16_recomputed"]["max_abs"] == pytest.approx(0.0)
    assert record["reference_captured_minus_bf16_recomputed"]["max_abs"] == pytest.approx(0.0)
    assert record["captured_delta_minus_bf16_recomputed_delta"]["max_abs"] == pytest.approx(0.0)
    assert record["fp32_direct_minus_decomposition"]["max_abs"] < 1e-6
    top = record["top_captured_norm_deltas"][0]
    assert top["hidden_index"] in {0, 1}
    assert top["dominant_decomposition_term"] in {"coordinate_only", "scale_only", "interaction"}


def test_sglang_debug_dump_tp_sum_only_merges_tp_partial_submodule_tensors(tmp_path):
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    dump0 = {
        "model.embed_tokens": hidden,
        "model.layers.0.mlp": hidden + 10,
        "model.layers.0": [hidden + 20, hidden + 30],
        "model.norm": hidden + 40,
    }
    dump1 = {
        "model.embed_tokens": hidden,
        "model.layers.0.mlp": hidden + 1,
        "model.layers.0": [hidden + 2, hidden + 300],
        "model.norm": hidden + 40,
    }
    dump_root = tmp_path / "dump"
    rank0 = dump_root / "TP0_PP0_Rank0_pid123"
    rank1 = dump_root / "TP1_PP0_Rank1_pid456"
    rank0.mkdir(parents=True)
    rank1.mkdir(parents=True)
    torch.save(dump0, rank0 / "Pass00001.pt")
    torch.save(dump1, rank1 / "Pass00001.pt")

    trace = {
        "trace_id": "trace-a",
        "prompt_ids": [1, 2, 3],
        "prompt_len": 3,
        "output_ids": [7472],
        "gen_len": 1,
        "full_ids": [1, 2, 3, 7472],
        "sglang_logprobs": [-2.5],
    }
    loaded = sglang_debug_dump_to_artifact.load_sglang_rank_dumps(
        dump_root,
        pass_index=1,
        rank_glob="TP*_*",
    )
    artifact = sglang_debug_dump_to_artifact.build_sglang_debug_artifact(
        traces=[trace],
        dump=dump0,
        dump_file=rank0 / "Pass00001.pt",
        tp_sum_rank_dumps=[dump for _path, dump in loaded],
        tp_sum_dump_files=[path for path, _dump in loaded],
        hidden_sample_indices="1,3",
        component_layers="0",
        num_layers=2,
        hidden_dim=4,
    )

    token = artifact["samples"][0]["per_token"][0]
    hidden_layers = {layer["index"]: layer for layer in token["xorl_hidden_state_summary"]["layers"]}
    components = {component["name"]: component for component in token["xorl_hidden_component_summary"]["components"]}

    assert len(loaded) == 2
    assert artifact["config"]["tp_sum_dump_files"] == [
        str(rank0 / "Pass00001.pt"),
        str(rank1 / "Pass00001.pt"),
    ]
    assert hidden_layers[0]["sample_values"] == pytest.approx([9.0, 11.0])
    assert hidden_layers[1]["sample_values"] == pytest.approx([79.0, 85.0])
    assert hidden_layers[2]["sample_values"] == pytest.approx([49.0, 51.0])
    assert components["mlp"]["sample_values"] == pytest.approx([29.0, 33.0])
    assert components["layer_output"]["sample_values"] == pytest.approx([79.0, 85.0])


def test_sglang_debug_dump_tp_sum_preserves_replicated_parent_mlp_output():
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    replicated_mlp = hidden + 100
    dump0 = {
        "model.layers.0.mlp": replicated_mlp,
        "model.layers.0.mlp.experts": hidden + 10,
        "model.layers.0.mlp.shared_expert": hidden + 20,
        "model.layers.0.mlp.shared_expert_gate": torch.zeros(5, 1),
        "model.layers.0.layer_output": replicated_mlp + 1000,
    }
    dump1 = {
        "model.layers.0.mlp": replicated_mlp.clone(),
        "model.layers.0.mlp.experts": hidden + 1,
        "model.layers.0.mlp.shared_expert": hidden + 2,
        "model.layers.0.mlp.shared_expert_gate": torch.zeros(5, 1),
        "model.layers.0.layer_output": replicated_mlp + 1000,
    }
    trace = {
        "trace_id": "trace-a",
        "prompt_ids": [1, 2, 3],
        "prompt_len": 3,
        "output_ids": [7472],
        "gen_len": 1,
        "full_ids": [1, 2, 3, 7472],
        "sglang_logprobs": [-2.5],
    }

    artifact = sglang_debug_dump_to_artifact.build_sglang_debug_artifact(
        traces=[trace],
        dump=dump0,
        tp_sum_rank_dumps=[dump0, dump1],
        hidden_sample_indices="1,3",
        component_layers="0",
        num_layers=1,
        hidden_dim=4,
    )

    token = artifact["samples"][0]["per_token"][0]
    components = {component["name"]: component for component in token["xorl_hidden_component_summary"]["components"]}

    assert components["raw_mlp"]["sample_values"] == pytest.approx([109.0, 111.0])
    assert components["mlp"]["sample_values"] == pytest.approx([49.0, 55.0])
    assert components["experts"]["sample_values"] == pytest.approx([29.0, 33.0])
    assert components["shared_expert"]["sample_values"] == pytest.approx([40.0, 44.0])
    assert components["shared_expert_weighted"]["sample_values"] == pytest.approx([20.0, 22.0])
    assert components["layer_output"]["sample_values"] == pytest.approx([1109.0, 1111.0])


def test_sglang_debug_dump_tp_sum_layer_input_hidden_contribution_then_adds_residual():
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    dump0 = {
        "model.layers.35.layer_input": hidden + 10,
        "model.layers.35.forward_input.hidden_states": hidden + 20,
        "model.layers.35.forward_input.residual": hidden + 100,
    }
    dump1 = {
        "model.layers.35.layer_input": hidden + 1,
        "model.layers.35.forward_input.hidden_states": hidden + 2,
        "model.layers.35.forward_input.residual": hidden + 999,
    }

    merged = sglang_debug_dump_to_artifact._merge_tp_layer_dumps([dump0, dump1], hidden_dim=4)
    layer_input = sglang_debug_dump_to_artifact._tensor_for_key(
        merged,
        ["model.layers.35.layer_input"],
        hidden_dim=4,
    )
    forward_input = sglang_debug_dump_to_artifact._tensor_for_key(
        merged,
        ["model.layers.35.forward_input.hidden_states"],
        hidden_dim=4,
    )

    assert layer_input is not None
    assert forward_input is not None
    assert layer_input[2, [1, 3]].tolist() == pytest.approx([138.0, 144.0])
    assert forward_input[2, [1, 3]].tolist() == pytest.approx([149.0, 155.0])


def test_slime_megatron_init_hook_patches_tilelang_018_prelower_check(monkeypatch):
    def old_engine_lower(_mod):
        return "old-engine-lower"

    def old_phase_check(_mod):
        return "old-phase"

    def old_lower_check(_mod):
        return "old-lower"

    tilelang_module = ModuleType("tilelang")
    engine_module = ModuleType("tilelang.engine")
    phase_module = ModuleType("tilelang.engine.phase")
    lower_module = ModuleType("tilelang.engine.lower")
    phase_module.PreLowerSemanticCheck = old_phase_check
    lower_module.PreLowerSemanticCheck = old_lower_check
    engine_module.phase = phase_module
    engine_module.lower = old_engine_lower

    monkeypatch.setitem(sys.modules, "tilelang", tilelang_module)
    monkeypatch.setitem(sys.modules, "tilelang.engine", engine_module)
    monkeypatch.setitem(sys.modules, "tilelang.engine.phase", phase_module)
    monkeypatch.setitem(sys.modules, "tilelang.engine.lower", lower_module)
    monkeypatch.setenv("XORL_SLIME_MEGATRON_PATCH_TILELANG_PRELOWER_CHECK", "1")
    monkeypatch.setattr(slime_megatron_activation_hook, "_TILELANG_PRELOWER_CHECK_PATCHED", False)

    slime_megatron_activation_hook._patch_tilelang_prelower_semantic_check()

    assert phase_module.PreLowerSemanticCheck is not old_phase_check
    assert lower_module.PreLowerSemanticCheck is not old_lower_check
    assert phase_module.PreLowerSemanticCheck(object()) is None
    assert lower_module.PreLowerSemanticCheck(object()) is None


def test_slime_megatron_init_hook_patches_te_triton_get_int_dtype(monkeypatch):
    triton = pytest.importorskip("triton")
    triton_core = pytest.importorskip("triton.language.core")
    original_get_int_dtype = triton_core.get_int_dtype
    monkeypatch.setattr(triton_core, "get_int_dtype", original_get_int_dtype)

    te_module = ModuleType("transformer_engine")
    common_module = ModuleType("transformer_engine.common")
    triton_module = ModuleType("transformer_engine.common.triton")
    permutation_module = ModuleType("transformer_engine.common.triton.permutation")
    permutation_module.get_int_dtype = original_get_int_dtype

    monkeypatch.setitem(sys.modules, "transformer_engine", te_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", common_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.triton", triton_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.triton.permutation", permutation_module)
    monkeypatch.setenv("XORL_SLIME_MEGATRON_PATCH_TRITON_GET_INT_DTYPE", "1")
    monkeypatch.setattr(slime_megatron_activation_hook, "_TRITON_GET_INT_DTYPE_PATCHED", False)

    slime_megatron_activation_hook._patch_triton_get_int_dtype_references()

    assert triton_core.get_int_dtype is not original_get_int_dtype
    assert hasattr(triton_core.get_int_dtype, "cache_key")
    assert permutation_module.get_int_dtype is triton_core.get_int_dtype
    assert hasattr(permutation_module.get_int_dtype, "cache_key")
    assert permutation_module.get_int_dtype.cache_key == triton.constexpr_function(original_get_int_dtype).cache_key


def test_slime_megatron_activation_hook_writes_component_artifact(monkeypatch, tmp_path):
    class AddModule(torch.nn.Module):
        def __init__(self, delta: float):
            super().__init__()
            self.delta = delta

        def forward(self, x):
            return x + self.delta

    class SharedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_weight = torch.nn.Parameter(torch.zeros(1, 4))

        def forward(self, x):
            raw = x + 60.0
            gate = torch.sigmoid(x @ self.gate_weight.t())
            return raw * gate

    class FakeMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = AddModule(50.0)
            self.shared_experts = SharedExperts()

        def forward(self, x):
            experts = self.experts(x)
            shared = self.shared_experts(x)
            return experts + shared

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = AddModule(20.0)
            self.self_attention = AddModule(30.0)
            self.pre_mlp_layernorm = AddModule(40.0)
            self.mlp = FakeMlp()

        def forward(self, x):
            input_norm = self.input_layernorm(x)
            attention = self.self_attention(input_norm)
            residual = x + attention
            post_attention_norm = self.pre_mlp_layernorm(residual)
            mlp = self.mlp(post_attention_norm)
            return residual + mlp

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class FakeMegatronModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def forward(self, x):
            return self.decoder(x)

    trace_path = tmp_path / "trace.json"
    static_trace_utils.write_static_trace_file(
        trace_path,
        {},
        [
            {
                "trace_id": "trace-a",
                "prompt_ids": [1, 2, 3],
                "output_ids": [7472],
                "sglang_logprobs": [-2.5],
            }
        ],
    )
    output_path = tmp_path / "slime-artifact-rank{rank}.json"
    monkeypatch.setenv("XORL_SLIME_MEGATRON_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("XORL_SLIME_MEGATRON_ACTIVATION_ARTIFACT", str(output_path))
    monkeypatch.setenv("XORL_SLIME_MEGATRON_COMPONENT_LAYERS", "0")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_HIDDEN_SAMPLE_INDICES", "1,3")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_RANKS", "all")

    empty_chunk = torch.nn.Identity()
    model = FakeMegatronModel()
    slime_megatron_activation_hook.capture_megatron_logprob_components(
        argparse.Namespace(),
        [empty_chunk, model],
        "actor_",
    )
    _empty_output = empty_chunk(torch.arange(20, dtype=torch.float32).reshape(5, 4))
    assert not (tmp_path / "slime-artifact-rank0.json").exists()
    _output = model(torch.arange(20, dtype=torch.float32).reshape(5, 4))

    artifact = json.loads((tmp_path / "slime-artifact-rank0.json").read_text(encoding="utf-8"))
    token = artifact["samples"][0]["per_token"][0]
    components = {component["name"]: component for component in token["xorl_hidden_component_summary"]["components"]}

    assert artifact["config"]["artifact_source"] == "slime_megatron_activation_hook"
    assert token["diagnostic_row_index"] == 2
    assert set(components) == {
        "layer_input",
        "input_norm",
        "attention",
        "post_attention_residual",
        "post_attention_norm",
        "shared_expert_input",
        "shared_expert_gate_value",
        "shared_expert_weighted",
        "experts",
        "mlp",
        "layer_output",
    }
    assert components["layer_input"]["sample_values"] == pytest.approx([9.0, 11.0])
    assert components["attention"]["sample_values"] == pytest.approx([59.0, 61.0])
    assert components["post_attention_residual"]["sample_values"] == pytest.approx([68.0, 72.0])
    assert components["post_attention_norm"]["sample_values"] == pytest.approx([108.0, 112.0])
    assert components["shared_expert_gate_value"]["sample_values"] == pytest.approx([0.5, 0.5])
    assert components["shared_expert_weighted"]["source_module"].endswith("mlp.shared_experts")
    assert components["shared_expert_weighted"]["sample_values"] == pytest.approx([84.0, 86.0])
    assert components["experts"]["source_module"] == "derived:mlp-shared_expert_weighted"
    assert components["mlp"]["sample_values"] == pytest.approx([242.0, 248.0])
    assert components["layer_output"]["sample_values"] == pytest.approx([310.0, 320.0])

    report = compare_hidden_component_artifacts.compare_hidden_component_artifacts(
        reference=artifact,
        candidate=artifact,
        top_n=1,
    )
    assert report["summary"]["matched_component_count"] == 11


def test_slime_megatron_activation_hook_captures_layer_input_from_hidden_states_kwarg(monkeypatch, tmp_path):
    class AddModule(torch.nn.Module):
        def __init__(self, delta: float):
            super().__init__()
            self.delta = delta

        def forward(self, x):
            return x + self.delta

    class KeywordLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = AddModule(10.0)
            self.self_attention = AddModule(20.0)

        def forward(self, *, hidden_states, attention_mask=None):
            del attention_mask
            input_norm = self.input_layernorm(hidden_states)
            attention = self.self_attention(input_norm)
            return hidden_states + attention

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([KeywordLayer()])

        def forward(self, x):
            return self.layers[0](hidden_states=x)

    class FakeMegatronModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def forward(self, x):
            return self.decoder(x)

    trace_path = tmp_path / "trace.json"
    static_trace_utils.write_static_trace_file(
        trace_path,
        {},
        [
            {
                "trace_id": "trace-a",
                "prompt_ids": [1],
                "output_ids": [7472],
                "sglang_logprobs": [-2.5],
            }
        ],
    )
    monkeypatch.setenv("XORL_SLIME_MEGATRON_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("XORL_SLIME_MEGATRON_ACTIVATION_ARTIFACT", str(tmp_path / "slime-artifact-rank{rank}.json"))
    monkeypatch.setenv("XORL_SLIME_MEGATRON_COMPONENT_LAYERS", "0")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_HIDDEN_SAMPLE_INDICES", "0,2")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_RANKS", "all")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_PATCH_TILELANG_PRELOWER_CHECK", "0")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_PATCH_TRITON_GET_INT_DTYPE", "0")

    model = FakeMegatronModel()
    slime_megatron_activation_hook.capture_megatron_logprob_components(argparse.Namespace(), model, "actor_")
    _output = model(torch.arange(12, dtype=torch.float32).reshape(3, 4))

    artifact = json.loads((tmp_path / "slime-artifact-rank0.json").read_text(encoding="utf-8"))
    token = artifact["samples"][0]["per_token"][0]
    components = {component["name"]: component for component in token["xorl_hidden_component_summary"]["components"]}

    assert components["layer_input"]["source_module"] == "chunk0.decoder.layers.0"
    assert components["layer_input"]["sample_values"] == pytest.approx([0.0, 2.0])
    assert components["attention"]["sample_values"] == pytest.approx([30.0, 32.0])
    assert components["post_attention_residual"]["sample_values"] == pytest.approx([30.0, 34.0])


def test_slime_megatron_activation_hook_derives_fused_input_norm(monkeypatch, tmp_path):
    class FusedLinearQkv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm_weight = torch.nn.Parameter(torch.zeros(4))

    class FusedAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = FusedLinearQkv()

        def forward(self, x):
            return x + 10.0

    class FusedFullAttentionLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(layernorm_epsilon=0.0, layernorm_zero_centered_gamma=True)
            self.input_layernorm = torch.nn.Identity()
            self.self_attention = FusedAttention()

        def forward(self, hidden_states):
            norm_or_identity = self.input_layernorm(hidden_states)
            attention = self.self_attention(norm_or_identity)
            return hidden_states + attention

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FusedFullAttentionLayer()])

        def forward(self, x):
            return self.layers[0](x)

    class FakeMegatronModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def forward(self, x):
            return self.decoder(x)

    trace_path = tmp_path / "trace.json"
    static_trace_utils.write_static_trace_file(
        trace_path,
        {},
        [
            {
                "trace_id": "trace-a",
                "prompt_ids": [1],
                "output_ids": [7472],
                "sglang_logprobs": [-2.5],
            }
        ],
    )
    monkeypatch.setenv("XORL_SLIME_MEGATRON_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("XORL_SLIME_MEGATRON_ACTIVATION_ARTIFACT", str(tmp_path / "slime-artifact-rank{rank}.json"))
    monkeypatch.setenv("XORL_SLIME_MEGATRON_COMPONENT_LAYERS", "0")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_HIDDEN_SAMPLE_INDICES", "0,1,2,3")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_RANKS", "all")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_PATCH_TILELANG_PRELOWER_CHECK", "0")
    monkeypatch.setenv("XORL_SLIME_MEGATRON_PATCH_TRITON_GET_INT_DTYPE", "0")

    model = FakeMegatronModel()
    slime_megatron_activation_hook.capture_megatron_logprob_components(argparse.Namespace(), model, "actor_")
    _output = model(torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float32))

    artifact = json.loads((tmp_path / "slime-artifact-rank0.json").read_text(encoding="utf-8"))
    token = artifact["samples"][0]["per_token"][0]
    components = {component["name"]: component for component in token["xorl_hidden_component_summary"]["components"]}

    assert components["layer_input"]["sample_values"] == pytest.approx([3.0, 4.0, 0.0, 0.0])
    assert components["input_norm"]["source_module"] == "derived:fused_input_layernorm"
    assert components["input_norm"]["sample_values"] == pytest.approx([1.2, 1.6, 0.0, 0.0])


def test_post_attention_norm_diagnostic_tp_sums_attention_and_checks_zero_centered_norm(tmp_path):
    def zero_centered_norm(tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = tensor.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + 1e-6)
        return output * (1.0 + weight.float())

    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    weight33 = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float32)
    weight35 = torch.tensor([0.2, -0.1, 0.0, 0.4], dtype=torch.float32)
    residual33 = hidden + 10
    attention33_rank0 = hidden + 1
    attention33_rank1 = hidden + 2
    post_residual33 = attention33_rank0 + attention33_rank1 + residual33
    residual35 = hidden + 20
    attention35_rank0 = hidden + 3
    attention35_rank1 = hidden + 4
    post_residual35 = attention35_rank0 + attention35_rank1 + residual35
    dump0 = {
        "model.layers.33.linear_attn": attention33_rank0,
        "model.layers.33.input_layernorm.residual": residual33,
        "model.layers.33.post_attention_residual": post_residual33,
        "model.layers.33.post_attention_layernorm": zero_centered_norm(post_residual33, weight33),
        "model.layers.35.o_proj": attention35_rank0,
        "model.layers.35.input_layernorm.residual": residual35,
        "model.layers.35.post_attention_residual": post_residual35,
        "model.layers.35.post_attention_layernorm": zero_centered_norm(post_residual35, weight35),
    }
    dump1 = {
        "model.layers.33.linear_attn": attention33_rank1,
        "model.layers.33.input_layernorm.residual": residual33 + 1000,
        "model.layers.33.post_attention_residual": post_residual33 + 1000,
        "model.layers.33.post_attention_layernorm": zero_centered_norm(post_residual33 + 1000, weight33),
        "model.layers.35.o_proj": attention35_rank1,
        "model.layers.35.input_layernorm.residual": residual35 + 1000,
        "model.layers.35.post_attention_residual": post_residual35 + 1000,
        "model.layers.35.post_attention_layernorm": zero_centered_norm(post_residual35 + 1000, weight35),
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.save(
        {
            "model.layers.33.post_attention_layernorm.weight": weight33,
            "model.layers.35.post_attention_layernorm.weight": weight35,
        },
        model_dir / "pytorch_model.bin",
    )

    report = diagnose_sglang_post_attention_norm.diagnose_post_attention_norm(
        dump=dump0,
        model_path=model_dir,
        layers=[33, 35],
        hidden_dim=4,
        tp_sum_rank_dumps=[dump0, dump1],
        row_indices=[1],
    )

    by_layer = {layer["layer"]: layer for layer in report["layers"]}
    assert by_layer[33]["attention_key"] == "model.layers.33.linear_attn"
    assert by_layer[35]["attention_key"] == "model.layers.35.o_proj"
    assert by_layer[33]["residual_add_diff"]["max_abs"] == pytest.approx(0.0)
    assert by_layer[35]["residual_add_diff"]["max_abs"] == pytest.approx(0.0)
    assert by_layer[33]["norm_from_captured_residual_diff"]["max_abs"] == pytest.approx(0.0)
    assert by_layer[35]["norm_from_captured_residual_diff"]["max_abs"] == pytest.approx(0.0)
    assert by_layer[35]["row_diffs"]["residual_add_diff"]["1"]["max_abs"] == pytest.approx(0.0)
    assert report["summary"]["layer_count"] == 2
    assert report["summary"]["norm_from_derived_residual_diff_max_abs"] == pytest.approx(0.0)


def test_sglang_debug_dump_request_scores_full_trace(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 1],
                        [-0.7, 3],
                        [-0.25, 4],
                    ]
                }
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(sglang_debug_dump_to_artifact.requests, "post", fake_post)

    metadata = sglang_debug_dump_to_artifact.request_sglang_dump(
        sglang_url="http://sglang",
        trace={
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3, 4],
            "sglang_logprobs": [-0.7, -0.25],
        },
        top_logprobs_num=5,
        timeout=12.0,
        validate_logprobs_atol=1e-6,
    )

    assert captured["url"] == "http://sglang/generate"
    assert captured["timeout"] == 12.0
    assert captured["json"]["input_ids"] == [1, 2, 3, 4]
    assert captured["json"]["sampling_params"]["max_new_tokens"] == 0
    assert captured["json"]["return_logprob"] is True
    assert captured["json"]["top_logprobs_num"] == 5
    assert metadata["trace_id"] == "t0"
    assert metadata["request_input_len"] == 4
    assert metadata["returned_logprobs"] == [-0.7, -0.25]
    assert metadata["max_abs_logprob_delta"] == pytest.approx(0.0)


def test_sglang_debug_dump_request_can_use_generation_path(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_ids": [3, 4],
                "meta_info": {
                    "output_token_logprobs": [
                        [-0.7, 3, "a"],
                        [-0.25, 4, "b"],
                    ]
                },
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(sglang_debug_dump_to_artifact.requests, "post", fake_post)

    metadata = sglang_debug_dump_to_artifact.request_sglang_dump(
        sglang_url="http://sglang",
        trace={
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3, 4],
            "sglang_logprobs": [-0.7, -0.25],
        },
        request_mode="generation",
        top_logprobs_num=5,
        timeout=12.0,
        validate_logprobs_atol=1e-6,
    )

    assert captured["url"] == "http://sglang/generate"
    assert captured["timeout"] == 12.0
    assert captured["json"]["input_ids"] == [1, 2]
    assert captured["json"]["sampling_params"]["max_new_tokens"] == 2
    assert captured["json"]["logprob_start_len"] == -1
    assert captured["json"]["return_logprob"] is True
    assert captured["json"]["top_logprobs_num"] == 5
    assert metadata["trace_id"] == "t0"
    assert metadata["request_mode"] == "generation"
    assert metadata["request_input_len"] == 2
    assert metadata["request_full_len"] == 4
    assert metadata["returned_logprobs"] == [-0.7, -0.25]
    assert metadata["generation_matches_trace_output_ids"] is True
    assert metadata["generation_first_mismatch"] is None
    assert metadata["max_abs_logprob_delta"] == pytest.approx(0.0)


def test_sglang_return_hidden_artifact_extracts_tail_hidden_and_compares_xorl(tmp_path, monkeypatch):
    traces_file = tmp_path / "traces.json"
    output_json = tmp_path / "hidden.json"
    xorl_result = tmp_path / "xorl.json"
    static_trace_utils.write_static_trace_file(
        traces_file,
        {"model_name": "m"},
        [
            {
                "trace_id": "t0",
                "prompt_ids": [1, 2],
                "output_ids": [3],
                "sglang_logprobs": [-0.25],
            }
        ],
    )
    xorl_result.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "per_token": [
                            {
                                "xorl_hidden_state_summary": {
                                    "layers": [
                                        {"index": 40, "sample_values": [0.75, 1.0]},
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "meta_info": {
                    "input_token_logprobs": [[None, 1], [-0.5, 2], [-0.25, 3]],
                    "input_top_logprobs": [[], [], [[-0.25, 3, None]]],
                    "hidden_states": [
                        [
                            [0.0, 0.25],
                            [0.5, 0.75],
                            [1.0, 1.25],
                        ]
                    ],
                }
            }

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(sglang_return_hidden_to_artifact.requests, "post", fake_post)

    artifact = sglang_return_hidden_to_artifact.request_hidden_artifact(
        traces_file=traces_file,
        sglang_url="http://sglang",
        output_json=output_json,
        top_logprobs_num=5,
        request_timeout=12.0,
        trace_ids=[],
        xorl_result=xorl_result,
        xorl_layer_index=40,
    )

    assert captured["url"] == "http://sglang/generate"
    assert captured["timeout"] == 12.0
    assert captured["json"] == sglang_return_hidden_to_artifact.build_hidden_request(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-0.25],
        },
        top_logprobs_num=5,
    )
    assert artifact["score"]["returned_logprobs"] == [-0.25]
    assert artifact["hidden"]["kind"] == "final"
    assert artifact["hidden"]["row_count"] == 1
    assert artifact["request"]["hidden_row_selection"] == "score"
    assert artifact["request"]["hidden_row_indices"] == [1]
    assert artifact["hidden"]["rows"][0]["source_row_index"] == 1
    assert artifact["hidden"]["rows"][0]["values"] == [0.5, 0.75]
    assert artifact["hidden"]["rows"][0]["summary"]["rms"] == pytest.approx(0.6373774391990981)
    assert artifact["xorl_comparison"]["xorl_reference"] == {"kind": "hidden_state", "layer_index": 40}
    assert artifact["xorl_comparison"]["summary"]["max_abs_delta"] == pytest.approx(0.25)
    assert json.loads(output_json.read_text(encoding="utf-8"))["trace_id"] == "t0"


def test_sglang_return_hidden_artifact_can_capture_prompt_tail_row(tmp_path, monkeypatch):
    traces_file = tmp_path / "traces.json"
    output_json = tmp_path / "hidden.json"
    static_trace_utils.write_static_trace_file(
        traces_file,
        {"model_name": "m"},
        [
            {
                "trace_id": "t0",
                "prompt_ids": [1, 2],
                "output_ids": [3],
                "sglang_logprobs": [-0.25],
            }
        ],
    )
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "meta_info": {
                    "input_token_logprobs": [[None, 1], [-0.5, 2]],
                    "input_top_logprobs": [[], [[-0.5, 2, None]]],
                    "hidden_states": [[[0.0, 0.25], [0.5, 0.75]]],
                }
            }

    def fake_post(url, json, timeout):
        captured["json"] = json
        return Response()

    monkeypatch.setattr(sglang_return_hidden_to_artifact.requests, "post", fake_post)

    artifact = sglang_return_hidden_to_artifact.request_hidden_artifact(
        traces_file=traces_file,
        sglang_url="http://sglang",
        output_json=output_json,
        top_logprobs_num=5,
        request_timeout=12.0,
        trace_ids=[],
        input_mode="prompt",
    )

    assert captured["json"] == sglang_return_hidden_to_artifact.build_hidden_request(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-0.25],
        },
        top_logprobs_num=5,
        input_mode="prompt",
    )
    assert captured["json"]["input_ids"] == [1, 2]
    assert artifact["request"]["input_mode"] == "prompt"
    assert artifact["request"]["input_len"] == 2
    assert artifact["request"]["hidden_row_selection"] == "score"
    assert artifact["request"]["hidden_row_indices"] == [1]
    assert artifact["score"]["returned_logprobs"] == [-0.5]
    assert artifact["score"]["expected_logprobs"] is None
    assert artifact["score"]["alignment"] == "prompt_tail_input_logprobs"
    assert artifact["hidden"]["rows"][0]["values"] == [0.5, 0.75]
    assert artifact["hidden"]["rows"][0]["source_row_index"] == 1


def test_sglang_return_hidden_artifact_can_compare_pre_final_component(tmp_path, monkeypatch):
    traces_file = tmp_path / "traces.json"
    output_json = tmp_path / "hidden.json"
    xorl_result = tmp_path / "xorl.json"
    static_trace_utils.write_static_trace_file(
        traces_file,
        {"model_name": "m"},
        [
            {
                "trace_id": "t0",
                "prompt_ids": [1, 2],
                "output_ids": [3],
                "sglang_logprobs": [-0.25],
            }
        ],
    )
    xorl_result.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "per_token": [
                            {
                                "xorl_hidden_component_summary": {
                                    "components": [
                                        {"layer": 39, "name": "layer_output", "sample_values": [1.25, 1.0]},
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "meta_info": {
                    "input_token_logprobs": [[None, 1], [-0.5, 2], [-0.25, 3]],
                    "hidden_states": [[[0.0, 0.25], [0.5, 0.75], [1.0, 1.25]]],
                }
            }

    monkeypatch.setattr(sglang_return_hidden_to_artifact.requests, "post", lambda *args, **kwargs: Response())

    artifact = sglang_return_hidden_to_artifact.request_hidden_artifact(
        traces_file=traces_file,
        sglang_url="http://sglang",
        output_json=output_json,
        top_logprobs_num=5,
        request_timeout=12.0,
        trace_ids=[],
        hidden_kind="pre_final_norm",
        xorl_result=xorl_result,
        xorl_component_layer=39,
        xorl_component_name="layer_output",
    )

    assert artifact["request"]["hidden_kind"] == "pre_final_norm"
    assert artifact["hidden"]["kind"] == "pre_final_norm"
    assert artifact["xorl_comparison"]["xorl_reference"] == {
        "kind": "component",
        "layer_index": 39,
        "component_name": "layer_output",
    }
    assert artifact["request"]["hidden_row_indices"] == [1]
    assert artifact["hidden"]["rows"][0]["values"] == [0.5, 0.75]
    assert artifact["xorl_comparison"]["summary"]["max_abs_delta"] == pytest.approx(0.75)


def test_diagnose_final_norm_boundary_recomputes_xorl_and_compares_sglang(tmp_path, monkeypatch):
    xorl_result = tmp_path / "xorl.json"
    sglang_hidden = tmp_path / "sglang-hidden.json"
    output_json = tmp_path / "report.json"
    model_path = tmp_path / "model"
    model_path.mkdir()

    final_input = torch.tensor([1.0, 2.0], dtype=torch.float32)
    weight = torch.tensor([0.0, 1.0], dtype=torch.float32)
    recomputed = diagnose_final_norm_boundary._apply_final_norm(
        final_input,
        weight,
        eps=1e-6,
        norm_type="zero-centered",
    )
    sglang_vector = [float(recomputed[0].item()) + 0.5, float(recomputed[1].item())]
    xorl_result.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "trace_id": "trace-a",
                        "per_token": [
                            {
                                "position": 0,
                                "token_id": 7,
                                "sglang_logprob": -1.0,
                                "xorl_logprob": -0.5,
                                "xorl_hidden_state_summary": {
                                    "sample_indices": [0, 1],
                                    "layers": [
                                        {
                                            "index": 2,
                                            "sample_values": [float(recomputed[0].item()), float(recomputed[1].item())],
                                        }
                                    ],
                                },
                                "xorl_hidden_component_summary": {
                                    "sample_indices": [0, 1],
                                    "components": [
                                        {
                                            "layer": 1,
                                            "name": "layer_output",
                                            "sample_values": [1.0, 2.0],
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sglang_hidden.write_text(
        json.dumps(
            {
                "trace_id": "trace-a",
                "score": {"returned_logprobs": [-1.0]},
                "hidden": {
                    "rows": [
                        {
                            "tail_index": -1,
                            "values": sglang_vector,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_load_checkpoint_tensor(path, key):
        assert path == model_path
        assert key == "model.norm.weight"
        return weight, model_path / "model.safetensors"

    monkeypatch.setattr(diagnose_final_norm_boundary, "load_checkpoint_tensor", fake_load_checkpoint_tensor)

    report = diagnose_final_norm_boundary.diagnose_final_norm_boundary(
        xorl_result_path=xorl_result,
        sglang_hidden_path=sglang_hidden,
        model_path=model_path,
        component_layer=1,
        final_hidden_layer=2,
        norm_eps=1e-6,
        norm_type="zero-centered",
        top_n=2,
    )
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert report["xorl"]["closure"]["recomputed_minus_captured"]["max_abs"] == pytest.approx(0.0)
    assert report["comparisons"]["xorl_final_minus_sglang_final"]["max_abs"] == pytest.approx(0.5)
    assert report["comparisons"]["top_xorl_final_minus_sglang_final"][0]["hidden_index"] == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["source"] == "diagnose_final_norm_boundary"


def test_sglang_debug_dump_generation_request_validates_output_ids(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_ids": [99],
                "meta_info": {"output_token_logprobs": [[-0.7, 99, "x"]]},
            }

    monkeypatch.setattr(sglang_debug_dump_to_artifact.requests, "post", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="did not reproduce trace output_ids"):
        sglang_debug_dump_to_artifact.request_sglang_dump(
            sglang_url="http://sglang",
            trace={
                "trace_id": "t0",
                "prompt_ids": [1, 2],
                "output_ids": [3],
                "sglang_logprobs": [-0.7],
            },
            request_mode="generation",
            validate_logprobs_atol=1e-6,
        )


def test_sglang_debug_dump_wait_selects_new_dump_with_required_rows(tmp_path):
    dump_root = tmp_path / "dump"
    rank_dir = dump_root / "TP0_PP0_Rank0_pid123"
    rank_dir.mkdir(parents=True)
    health_dump = rank_dir / "Pass00000.pt"
    torch.save({"model.layers.33.layer_output": torch.zeros(1, 4)}, health_dump)
    existing = set(sglang_debug_dump_to_artifact.list_sglang_dump_files(dump_root))
    torch.save({"model.layers.33.layer_output": torch.ones(3, 4)}, rank_dir / "Pass00001.pt")
    trace_dump = rank_dir / "Pass00002.pt"
    torch.save({"model.layers.33.layer_output": torch.ones(8, 4)}, trace_dump)

    dump_file, dump = sglang_debug_dump_to_artifact.wait_for_sglang_dump_with_min_rows(
        dump_root,
        min_rows=5,
        existing_files=existing,
        timeout=0.1,
        poll_interval=0.01,
        hidden_dim=4,
    )

    assert dump_file == trace_dump
    assert dump["model.layers.33.layer_output"].shape == (8, 4)


def test_threshold_failure_message():
    agg = {"k3": {"mean": 0.02, "p95": 0.03}}

    assert "mean K3" in compare_static_traces._threshold_failed(agg, 0.01, None)
    assert "p95 K3" in compare_static_traces._threshold_failed(agg, None, 0.01)
    assert compare_static_traces._threshold_failed(agg, 0.03, 0.04) is None


def test_build_k3_diagnostics_reports_worst_token_and_sample():
    diagnostics = compare_static_traces.build_k3_diagnostics(
        [
            {
                "trace_id": "t0",
                "prompt_len": 4,
                "gen_len": 2,
                "sample_k3_mean": 5.0,
                "per_token": [
                    {"position": 0, "token_id": 10, "k3": 0.1, "sglang_logprob": -1.0, "xorl_logprob": -1.1},
                    {"position": 1, "token_id": 11, "k3": 9.9, "sglang_logprob": -0.2, "xorl_logprob": -5.2},
                ],
            }
        ],
        top_n=1,
    )

    assert diagnostics["worst_tokens"][0]["trace_id"] == "t0"
    assert diagnostics["worst_tokens"][0]["position"] == 1
    assert diagnostics["worst_tokens"][0]["absolute_position"] == 5
    assert diagnostics["worst_samples"][0]["max_token_id"] == 11
    assert diagnostics["abs_logprob_diff_stats"]["max"] == pytest.approx(5.0)


def test_build_k3_diagnostics_summarizes_top_logprobs():
    diagnostics = compare_static_traces.build_k3_diagnostics(
        [
            {
                "trace_id": "t0",
                "prompt_len": 4,
                "gen_len": 1,
                "sample_k3_mean": 9.0,
                "per_token": [
                    {
                        "position": 0,
                        "token_id": 11,
                        "k3": 9.0,
                        "sglang_logprob": -0.2,
                        "xorl_logprob": -5.2,
                        "sglang_top_logprobs": [[-0.1, 10, "x"], [-0.2, 11, "y"]],
                    },
                ],
            }
        ],
        top_n=1,
    )

    top = diagnostics["worst_tokens"][0]["sglang_top_logprobs"]
    assert top["target_rank"] == 2
    assert top["target_logprob"] == pytest.approx(-0.2)


def test_shift_scan_detects_better_shifted_alignment():
    samples = [
        {
            "trace_id": "t0",
            "per_token": [
                {"sglang_logprob": -1.0, "xorl_logprob": -9.0},
                {"sglang_logprob": -2.0, "xorl_logprob": -1.0},
                {"sglang_logprob": -3.0, "xorl_logprob": -2.0},
            ],
        }
    ]

    scan = diagnose_static_k3.compute_shift_scan(samples, max_shift=1)
    best = diagnose_static_k3.best_shift(scan)

    assert best["xorl_shift"] == 1
    assert best["k3"]["mean"] == pytest.approx(0.0)


def test_diagnose_report_can_annotate_worst_token_with_trace_context():
    k3_result = {
        "aggregate": {"total_tokens": 1, "k3": {"mean": 10.0, "p95": 10.0, "max": 10.0}},
        "diagnostics": {
            "worst_tokens": [
                {
                    "trace_id": "t0",
                    "position": 1,
                    "absolute_position": 3,
                    "token_id": 4,
                    "k3": 10.0,
                    "sglang_logprob": -1.0,
                    "xorl_logprob": -5.0,
                }
            ],
            "worst_samples": [],
        },
        "samples": [
            {
                "trace_id": "t0",
                "per_token": [
                    {"sglang_logprob": -1.0, "xorl_logprob": -5.0},
                    {"sglang_logprob": -2.0, "xorl_logprob": -2.0},
                ],
            }
        ],
    }
    trace_map = {
        "t0": {
            "trace_id": "t0",
            "prompt_len": 2,
            "full_ids": [1, 2, 3, 4, 5],
        }
    }

    report = diagnose_static_k3.build_report(
        k3_result=k3_result,
        trace_metadata={},
        trace_map=trace_map,
        model_name=None,
        top_n=1,
        max_shift=1,
        context_radius=1,
    )

    assert report["diagnostics"]["worst_tokens"][0]["trace_id"] == "t0"
    assert "decoded_context" not in report["diagnostics"]["worst_tokens"][0]


def test_sglang_self_consistency_reports_generation_prefill_diffs():
    result = diagnose_static_k3.compute_sglang_self_consistency(
        [
            {
                "trace_id": "t0",
                "prompt_len": 2,
                "output_ids": [10, 11],
                "sglang_generation_logprobs": [-1.0, -2.0],
                "sglang_logprobs": [-1.1, -4.5],
            }
        ],
        top_n=1,
    )

    assert result["checked_tokens"] == 2
    assert result["abs_logprob_diff"]["max"] == pytest.approx(2.5)
    assert result["worst_tokens"][0]["token_id"] == 11


def test_reference_mode_comparison_splits_current_prefill_generation():
    samples = [
        {
            "trace_id": "t0",
            "prompt_len": 3,
            "gen_len": 2,
            "per_token": [
                {
                    "position": 0,
                    "token_id": 10,
                    "sglang_logprob": -1.0,
                    "sglang_prefill_logprob": -8.0,
                    "sglang_generation_logprob": -1.0,
                    "xorl_logprob": -1.0,
                },
                {
                    "position": 1,
                    "token_id": 11,
                    "sglang_logprob": -2.0,
                    "sglang_prefill_logprob": -4.0,
                    "sglang_generation_logprob": -2.0,
                    "xorl_logprob": -4.0,
                },
            ],
        }
    ]

    result = diagnose_static_k3.compute_reference_mode_comparison(samples, top_n=1)

    assert result["current"]["checked_tokens"] == 2
    assert result["prefill"]["checked_tokens"] == 2
    assert result["generation"]["checked_tokens"] == 2
    assert result["prefill"]["worst_tokens"][0]["position"] == 0
    assert result["generation"]["worst_tokens"][0]["position"] == 1
    assert result["current"]["k3"]["max"] == pytest.approx(result["generation"]["k3"]["max"])


def test_compare_k3_results_reports_recurring_bad_token_position():
    run_a = {
        "aggregate": {"total_tokens": 2, "k3": {"mean": 5.0, "p95": 9.0, "max": 9.0}},
        "samples": [
            {
                "trace_id": "t0",
                "prompt_len": 4,
                "gen_len": 2,
                "per_token": [
                    {"position": 0, "token_id": 10, "k3": 0.1, "sglang_logprob": -1.0, "xorl_logprob": -1.1},
                    {"position": 1, "token_id": 11, "k3": 9.0, "sglang_logprob": -2.0, "xorl_logprob": -8.0},
                ],
            }
        ],
    }
    run_b = {
        "aggregate": {"total_tokens": 2, "k3": {"mean": 20.0, "p95": 40.0, "max": 40.0}},
        "samples": [
            {
                "trace_id": "t0",
                "prompt_len": 4,
                "gen_len": 2,
                "per_token": [
                    {"position": 0, "token_id": 10, "k3": 0.2, "sglang_logprob": -1.0, "xorl_logprob": -0.9},
                    {"position": 1, "token_id": 11, "k3": 40.0, "sglang_logprob": -2.0, "xorl_logprob": -12.0},
                ],
            }
        ],
    }

    result = diagnose_static_k3.compare_k3_results([("baseline", run_a), ("native", run_b)], top_n=1)
    token = result["recurring_tokens"][0]

    assert token["trace_id"] == "t0"
    assert token["position"] == 1
    assert token["absolute_position"] == 5
    assert token["token_id"] == 11
    assert token["artifact_count"] == 2
    assert token["top_n_artifact_count"] == 2
    assert token["k3"]["max"] == pytest.approx(40.0)
    assert token["xorl_minus_sglang_logprob"]["min"] == pytest.approx(-10.0)
    assert [run["artifact_label"] for run in token["runs"]] == ["native", "baseline"]


def test_gdn_parity_selects_first_linear_attention_layer():
    config = compare_qwen36_gdn_parity.GDNConfig(
        hidden_size=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=3,
        linear_value_head_dim=5,
        linear_conv_kernel_dim=4,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        layer_types=["full_attention", "linear_attention", "linear_attention"],
    )

    assert compare_qwen36_gdn_parity.select_linear_attention_layer(config, None) == 1
    assert compare_qwen36_gdn_parity.select_linear_attention_layer(config, 2) == 2
    with pytest.raises(ValueError, match="not linear_attention"):
        compare_qwen36_gdn_parity.select_linear_attention_layer(config, 0)


def test_gdn_parity_splits_qkv_and_runs_causal_conv():
    config = compare_qwen36_gdn_parity.GDNConfig(
        hidden_size=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=2,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    weights = compare_qwen36_gdn_parity.GDNWeights(
        in_proj_qkv=torch.arange((2 + 2 + 3) * 2, dtype=torch.float32).reshape(7, 2),
        in_proj_z=torch.empty(3, 2),
        in_proj_b=torch.empty(1, 2),
        in_proj_a=torch.empty(1, 2),
        conv1d=torch.ones(7, 1, 2),
        out_proj=torch.empty(2, 3),
        norm=torch.empty(3),
        dt_bias=torch.empty(1),
        A_log=torch.empty(1),
    )

    q, k, v = compare_qwen36_gdn_parity.split_qkv_weight(weights, config)
    conv = compare_qwen36_gdn_parity.causal_depthwise_conv_1d(
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        torch.ones(1, 1, 2),
        activation="identity",
    )

    assert q.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert k.tolist() == [[4.0, 5.0], [6.0, 7.0]]
    assert v.tolist() == [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]
    assert conv.squeeze(-1).tolist() == [[1.0, 3.0, 5.0]]


def test_gdn_parity_checkpoint_key_candidates_include_language_model_prefix():
    candidates = compare_qwen36_gdn_parity._checkpoint_key_candidates("model.layers.0.linear_attn.A_log")

    assert candidates == [
        "model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.A_log",
    ]


def test_gdn_parity_reports_flattened_row_diff_stats():
    actual = torch.zeros(2, 3, 2)
    expected = actual.clone()
    expected[0, 0] = torch.tensor([1.0, 3.0])
    expected[1, 1] = torch.tensor([2.0, 4.0])

    rows = compare_qwen36_gdn_parity.diff_stats_for_flat_rows(actual, expected, [0, 4])

    assert rows["0"]["flat_row"] == 0
    assert rows["0"]["batch_index"] == 0
    assert rows["0"]["sequence_index"] == 0
    assert rows["0"]["shape"] == [2]
    assert rows["0"]["max_abs"] == pytest.approx(3.0)
    assert rows["0"]["mean_abs"] == pytest.approx(2.0)
    assert rows["4"]["flat_row"] == 4
    assert rows["4"]["batch_index"] == 1
    assert rows["4"]["sequence_index"] == 1
    assert rows["4"]["max_abs"] == pytest.approx(4.0)
    assert rows["4"]["mean_abs"] == pytest.approx(3.0)


def test_gdn_parity_reports_top_abs_diff_locations():
    actual = torch.zeros(2, 3, 2)
    expected = actual.clone()
    actual[1, 2, 0] = -4.0
    expected[0, 1, 1] = 3.0
    expected[1, 2, 0] = 1.0

    entries = compare_qwen36_gdn_parity.top_abs_diff_entries(actual, expected, top_n=2)

    assert entries[0]["rank"] == 0
    assert entries[0]["indices"] == [1, 2, 0]
    assert entries[0]["actual_value"] == pytest.approx(-4.0)
    assert entries[0]["expected_value"] == pytest.approx(1.0)
    assert entries[0]["delta"] == pytest.approx(-5.0)
    assert entries[0]["abs_delta"] == pytest.approx(5.0)
    assert entries[1]["indices"] == [0, 1, 1]
    assert entries[1]["abs_delta"] == pytest.approx(3.0)


def test_gdn_parity_loads_sglang_dump_tensor_and_tp_sums_rank_outputs(tmp_path):
    rank0 = tmp_path / "TP0_PP0_Rank0"
    rank1 = tmp_path / "TP1_PP0_Rank1"
    rank0.mkdir()
    rank1.mkdir()
    torch.save({"layer.output": torch.ones(2, 3)}, rank0 / "Pass00001.pt")
    torch.save({"layer.output": torch.full((2, 3), 2.0)}, rank1 / "Pass00001.pt")

    single, single_source = compare_qwen36_gdn_parity.load_dump_tensor(
        tmp_path,
        key="layer.output",
        pass_index=1,
        rank_glob="TP*_*",
        tp_sum=False,
        hidden_dim=3,
    )
    summed, summed_source = compare_qwen36_gdn_parity.load_dump_tensor(
        tmp_path,
        key="layer.output",
        pass_index=1,
        rank_glob="TP*_*",
        tp_sum=True,
        hidden_dim=3,
    )

    assert single.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert len(single_source.files) == 1
    assert summed.tolist() == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    assert len(summed_source.files) == 2


def test_sglang_gdn_rank_local_closure_closes_projection_core_norm_and_out_proj(monkeypatch, tmp_path):
    config = compare_qwen36_gdn_parity.GDNConfig(
        hidden_size=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=1,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=1,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    weights = compare_qwen36_gdn_parity.GDNWeights(
        in_proj_qkv=torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4) / 16.0,
        in_proj_z=torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) / 13.0,
        in_proj_b=torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) / 11.0,
        in_proj_a=torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4) / 7.0,
        conv1d=torch.ones(16, 1, dtype=torch.float32),
        out_proj=torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) / 5.0,
        norm=torch.tensor([0.5, 1.5], dtype=torch.float32),
        dt_bias=torch.zeros(4, dtype=torch.float32),
        A_log=torch.zeros(4, dtype=torch.float32),
    )
    hidden = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 9.0

    def fake_chunk_gated_delta_rule(**kwargs):
        return kwargs["v"], None, None

    for tp_rank in range(2):
        rank_dir = tmp_path / f"TP{tp_rank}_PP0_Rank{tp_rank}"
        rank_dir.mkdir()
        qkvz_weight = diagnose_sglang_gdn_rank_local_closure._qkvz_weight_shard(
            weights, config, tp_rank=tp_rank, tp_size=2
        )
        ba_weight = diagnose_sglang_gdn_rank_local_closure._ba_weight_shard(weights, tp_rank=tp_rank, tp_size=2)
        out_proj_weight = diagnose_sglang_gdn_rank_local_closure._column_shard(
            weights.out_proj, tp_rank=tp_rank, tp_size=2, name="out_proj"
        )
        qkvz = torch.nn.functional.linear(hidden, qkvz_weight)
        ba = torch.nn.functional.linear(hidden, ba_weight)
        mixed_qkv, z, _b, _a = diagnose_sglang_gdn_rank_local_closure._split_rank_qkvz_ba(
            qkvz,
            ba,
            config=config,
            tp_size=2,
        )
        mixed_qkv_conv = compare_qwen36_gdn_parity.causal_depthwise_conv_1d(
            mixed_qkv.unsqueeze(0),
            diagnose_sglang_gdn_rank_local_closure._conv_weight_shard(weights, config, tp_rank=tp_rank, tp_size=2),
            activation="silu",
        ).squeeze(0)
        _q, _k, value = mixed_qkv_conv.split([1, 1, 4], dim=-1)
        attn = value.reshape(1, 3, 2, 2)
        norm = compare_qwen36_gdn_parity.rms_norm_gated_reference(
            attn,
            z.reshape(1, 3, 2, 2),
            weights.norm,
            config.rms_norm_eps,
        ).squeeze(0)
        out_proj = torch.nn.functional.linear(norm.reshape(3, 4), out_proj_weight)
        torch.save(
            {
                "model.layers.0.input_layernorm": hidden,
                "model.layers.0.linear_attn.in_proj_qkvz": qkvz,
                "model.layers.0.linear_attn.in_proj_ba": ba,
                "model.layers.0.linear_attn.attn": attn,
                "model.layers.0.linear_attn.norm": norm.reshape(6, 2),
                "model.layers.0.linear_attn.out_proj": out_proj,
                "model.layers.0.linear_attn": out_proj.clone(),
            },
            rank_dir / "Pass00001.pt",
        )

    monkeypatch.setattr(diagnose_sglang_gdn_rank_local_closure, "load_gdn_config", lambda _model_path: config)
    monkeypatch.setattr(diagnose_sglang_gdn_rank_local_closure, "load_gdn_weights", lambda _model_path, _layer: weights)
    monkeypatch.setattr(
        diagnose_sglang_gdn_rank_local_closure,
        "import_sglang_fla_helpers",
        lambda _sglang_path: SimpleNamespace(chunk_gated_delta_rule=fake_chunk_gated_delta_rule),
    )

    result = diagnose_sglang_gdn_rank_local_closure.analyze_sglang_rank_local_closure(
        sglang_dump=tmp_path,
        model_path="/unused",
        pass_index=1,
        rank_glob="TP*_*",
        layer=0,
        rows=[0, 2],
        device="cpu",
        dtype=torch.float32,
        top_n=2,
    )

    assert result["summary"]["rank_count"] == 2
    assert result["summary"]["record_count"] == 4
    assert result["summary"]["qkvz_projection_max_abs"] == pytest.approx(0.0)
    assert result["summary"]["ba_projection_max_abs"] == pytest.approx(0.0)
    assert result["summary"]["core_to_attn_max_abs"] == pytest.approx(0.0)
    assert result["summary"]["computed_core_to_norm_max_abs"] == pytest.approx(0.0)
    assert result["summary"]["captured_attn_to_norm_max_abs"] == pytest.approx(0.0)
    assert result["summary"]["norm_to_out_proj_max_abs"] <= 2e-5
    assert result["summary"]["out_proj_vs_linear_attn_max_abs"] <= 2e-5


def test_gdn_parity_normalizes_captured_token_rows_to_batched_hidden_states():
    token_rows = torch.zeros(4, 8)
    batched = torch.zeros(2, 4, 8)

    assert compare_qwen36_gdn_parity._normalize_hidden_states(token_rows, hidden_dim=8, name="hidden").shape == (
        1,
        4,
        8,
    )
    assert compare_qwen36_gdn_parity._normalize_hidden_states(batched, hidden_dim=8, name="hidden").shape == (2, 4, 8)
    with pytest.raises(ValueError, match="must have shape"):
        compare_qwen36_gdn_parity._normalize_hidden_states(torch.zeros(4, 7), hidden_dim=8, name="hidden")


def test_gdn_parity_rejects_invalid_flattened_row_index():
    tensor = torch.zeros(1, 2, 3)

    with pytest.raises(ValueError, match=r"outside flattened batch\*sequence range"):
        compare_qwen36_gdn_parity.diff_stats_for_flat_rows(tensor, tensor, [2])


@pytest.mark.skipif(
    not hasattr(ModelRunner, "_compute_token_diagnostics"),
    reason="ModelRunner.token_diagnostics support not in this build (see PR #306).",
)
def test_model_runner_token_diagnostics_reports_exact_rank():
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    labels = torch.tensor([[-100, 2]])

    diagnostics = ModelRunner._compute_token_diagnostics(hidden, weight, labels, topk=2)

    assert diagnostics["valid_positions"] == [1]
    assert diagnostics["target_ids"] == [2]
    assert diagnostics["target_ranks"] == [2]
    assert diagnostics["topk_ids"][0][0] == 1


@pytest.mark.skipif(
    not hasattr(RequestProcessor, "_unpack_token_diagnostics"),
    reason="RequestProcessor.token_diagnostics support not in this build (see PR #306).",
)
def test_request_processor_unpacks_token_diagnostics_by_position_reset():
    diagnostics = {
        "valid_positions": [1, 3, 4],
        "target_ids": [10, 20, 21],
        "target_logprobs": [-1.0, -2.0, -3.0],
        "target_ranks": [1, 2, 3],
        "topk_ids": [[10], [20], [21]],
        "topk_logprobs": [[-1.0], [-2.0], [-3.0]],
    }

    unpacked = RequestProcessor._unpack_token_diagnostics(diagnostics, torch.tensor([[0, 1, 2, 0, 1]]))

    assert unpacked[0]["valid_positions"] == [1]
    assert unpacked[0]["target_ids"] == [10]
    assert unpacked[1]["valid_positions"] == [0, 1]
    assert unpacked[1]["target_ids"] == [20, 21]


def test_extract_generation_logprobs_handles_sglang_tuple_format():
    gen_result = {"meta_info": {"output_token_logprobs": [[-0.1, 1, "a"], [-0.2, 2, "b"]]}}

    assert make_static_traces.extract_generation_logprobs(gen_result, 2) == [-0.1, -0.2]


def test_launch_profile_respects_explicit_xorl_gpu_override():
    args = argparse.Namespace(
        model="qwen3.6-35b",
        model_path=None,
        xorl_config="custom.yaml",
        sglang_tp=None,
        sglang_gpus=None,
        xorl_gpus=4,
    )

    model_path, sglang_tp, sglang_gpus, xorl_gpus, xorl_config = launch_k3_test.resolve_launch_settings(args)

    assert model_path == "Qwen/Qwen3.6-35B-A3B"
    assert sglang_tp == 2
    assert sglang_gpus == 2
    assert xorl_gpus == 4
    assert xorl_config == "custom.yaml"


@pytest.mark.parametrize(
    ("profile", "expected_config"),
    [
        (
            "qwen3-8b-qlora-nvfp4-fp8-sync",
            "experiments/k3_tests/configs/qwen3-8b_qlora-nvfp4-fp8-sync-smoke.yaml",
        ),
        (
            "qwen3-8b-qlora-block-fp8-sync",
            "experiments/k3_tests/configs/qwen3-8b_qlora-block-fp8-sync-smoke.yaml",
        ),
    ],
)
def test_prequantized_qlora_fp8_sync_profiles_launch_fp8_receiver(profile, expected_config):
    args = argparse.Namespace(
        model=profile,
        model_path=None,
        xorl_config=None,
        sglang_tp=None,
        sglang_gpus=None,
        xorl_gpus=None,
        sglang_quantization="",
        sglang_load_format="",
        sglang_kv_cache_dtype="",
    )

    model_path, sglang_tp, sglang_gpus, xorl_gpus, xorl_config = launch_k3_test.resolve_launch_settings(args)
    launch_k3_test.apply_launch_profile_runtime_defaults(args)

    assert model_path == "Qwen/Qwen3-8B-FP8"
    assert sglang_tp == 1
    assert sglang_gpus == 1
    assert xorl_gpus == 1
    assert xorl_config == expected_config
    assert args.sglang_quantization == "fp8"
    assert args.sglang_load_format == "flash_rl"


def test_launcher_command_records_reproducible_invocation():
    command = launch_k3_test.launcher_command(["launch_k3_test.py", "--capacity-check-only"])

    assert command[0] == sys.executable
    assert command[1:] == ["launch_k3_test.py", "--capacity-check-only"]


def test_launch_k3_rejects_cuda_visible_devices_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch_k3_test.py", "--xorl-cuda-visible-devices", "0"])

    with pytest.raises(SystemExit):
        launch_k3_test.main()


def test_launch_k3_rejects_node_name_and_hostname_selector(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_k3_test.py",
            "--xorl-node-name",
            "research-common-h100-073.cloud.together.ai",
            "--xorl-node-selector-name",
            "research-common-h100-073.cloud.together.ai",
        ],
    )

    with pytest.raises(SystemExit):
        launch_k3_test.main()


def test_launch_k3_rejects_capacity_selector_with_explicit_hostname(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_k3_test.py",
            "--xorl-node-selector-name",
            "research-common-h100-073.cloud.together.ai",
            "--xorl-node-selector-from-capacity",
        ],
    )

    with pytest.raises(SystemExit):
        launch_k3_test.main()


def test_launch_k3_rejects_sglang_capacity_selector_with_explicit_hostname(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_k3_test.py",
            "--sglang-node-selector-name",
            "research-common-h100-073.cloud.together.ai",
            "--sglang-node-selector-from-capacity",
        ],
    )

    with pytest.raises(SystemExit):
        launch_k3_test.main()


def test_launch_k3_rejects_non_positive_capacity_poll_interval(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch_k3_test.py", "--gpu-capacity-poll-interval-sec", "0"])

    with pytest.raises(SystemExit):
        launch_k3_test.main()


def test_launch_k3_rejects_incomplete_sglang_return_hidden_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch_k3_test.py", "--sglang-return-hidden-only"])

    with pytest.raises(SystemExit):
        launch_k3_test.main()


def test_static_trace_sglang_debug_dump_mode_launches_sglang_without_xorl():
    args = argparse.Namespace(
        static_traces_file="/tmp/traces.json",
        sglang_debug_dump=True,
        sglang_debug_dump_output_folder="",
    )

    assert launch_k3_test.sglang_debug_dump_enabled(args) is True
    assert launch_k3_test.should_launch_sglang(args) is True
    requests = launch_k3_test.launch_gpu_requests(
        sglang_pod="k3-sg-debug",
        sglang_gpus=2,
        sglang_node_name=None,
        xorl_pod=None,
        xorl_gpus=8,
        xorl_node_name=None,
    )

    assert requests == [("k3-sg-debug", 2, None)]


def test_static_trace_sglang_return_hidden_mode_launches_sglang():
    args = argparse.Namespace(
        static_traces_file="/tmp/traces.json",
        sglang_debug_dump=False,
        sglang_debug_dump_output_folder="",
        sglang_return_hidden_artifact_json="/tmp/hidden.json",
    )

    assert launch_k3_test.sglang_return_hidden_enabled(args) is True
    assert launch_k3_test.should_launch_sglang(args) is True


def test_build_sglang_debug_dump_command_threads_reference_compare_args():
    args = argparse.Namespace(
        static_traces_file="/tmp/traces.json",
        sglang_debug_dump_output_folder="/home/apanda/k3_sglang_debug/run",
        sglang_debug_dump_artifact_json="/tmp/sglang-artifact.json",
        sglang_debug_dump_pass_index=0,
        sglang_debug_dump_rank_glob="TP0_*",
        sglang_debug_dump_request_mode="generation",
        sglang_debug_dump_tp_sum_rank_glob="TP*_*",
        sglang_debug_dump_request_timeout=11.0,
        sglang_debug_dump_timeout=12.0,
        sglang_debug_dump_hidden_sample_count=4,
        sglang_debug_dump_request_top_logprobs_num=5,
        sglang_debug_dump_validate_logprobs_atol=1e-6,
        sglang_debug_dump_hidden_sample_indices="0,127,1024,2047",
        sglang_debug_dump_component_layers="33-39",
        sglang_debug_dump_num_layers=41,
        sglang_debug_dump_hidden_dim=2048,
        sglang_debug_dump_reference_artifact="/tmp/xorl.json",
        sglang_debug_dump_compare_json="/tmp/compare.json",
        sglang_debug_dump_compare_top_n=7,
        sglang_debug_dump_compare_include_all=True,
        trace_id=["trace-a"],
    )

    cmd = launch_k3_test.build_sglang_debug_dump_command(
        python_bin=Path("/venv/bin/python"),
        sglang_url="http://sglang:30000",
        args=args,
    )

    assert cmd[:2] == ["/venv/bin/python", str(launch_k3_test.SGLANG_DEBUG_DUMP_SCRIPT)]
    assert "--sglang-url" in cmd
    assert cmd[cmd.index("--sglang-url") + 1] == "http://sglang:30000"
    assert cmd[cmd.index("--dump-path") + 1] == "/home/apanda/k3_sglang_debug/run"
    assert cmd[cmd.index("--request-mode") + 1] == "generation"
    assert cmd[cmd.index("--tp-sum-rank-glob") + 1] == "TP*_*"
    assert cmd[cmd.index("--hidden-sample-indices") + 1] == "0,127,1024,2047"
    assert cmd[cmd.index("--component-layers") + 1] == "33-39"
    assert cmd[cmd.index("--reference-artifact") + 1] == "/tmp/xorl.json"
    assert cmd[cmd.index("--compare-output-json") + 1] == "/tmp/compare.json"
    assert "--compare-include-all" in cmd
    assert cmd[cmd.index("--trace-id") + 1] == "trace-a"


def test_build_sglang_return_hidden_command_threads_compare_args():
    args = argparse.Namespace(
        static_traces_file="/tmp/traces.json",
        sglang_return_hidden_artifact_json="/tmp/hidden.json",
        sglang_return_hidden_top_logprobs_num=5,
        sglang_return_hidden_request_timeout=11.0,
        sglang_return_hidden_input_mode="full",
        sglang_return_pre_final_norm_hidden=False,
        sglang_return_hidden_xorl_result="/tmp/xorl.json",
        sglang_return_hidden_xorl_layer_index=40,
        sglang_return_hidden_xorl_component_layer=None,
        sglang_return_hidden_xorl_component_name="layer_output",
        trace_id=["trace-a"],
    )

    cmd = launch_k3_test.build_sglang_return_hidden_command(
        python_bin=Path("/venv/bin/python"),
        sglang_url="http://sglang:30000",
        args=args,
    )

    assert cmd[:2] == ["/venv/bin/python", str(launch_k3_test.SGLANG_RETURN_HIDDEN_SCRIPT)]
    assert cmd[cmd.index("--sglang-url") + 1] == "http://sglang:30000"
    assert cmd[cmd.index("--output-json") + 1] == "/tmp/hidden.json"
    assert cmd[cmd.index("--hidden-kind") + 1] == "final"
    assert cmd[cmd.index("--input-mode") + 1] == "full"
    assert cmd[cmd.index("--xorl-result") + 1] == "/tmp/xorl.json"
    assert cmd[cmd.index("--xorl-layer-index") + 1] == "40"
    assert cmd[cmd.index("--trace-id") + 1] == "trace-a"


def test_build_sglang_return_hidden_command_can_request_pre_final_component_compare():
    args = argparse.Namespace(
        static_traces_file="/tmp/traces.json",
        sglang_return_hidden_artifact_json="/tmp/hidden.json",
        sglang_return_hidden_top_logprobs_num=5,
        sglang_return_hidden_request_timeout=11.0,
        sglang_return_hidden_input_mode="prompt",
        sglang_return_pre_final_norm_hidden=True,
        sglang_return_hidden_xorl_result="/tmp/xorl.json",
        sglang_return_hidden_xorl_layer_index=40,
        sglang_return_hidden_xorl_component_layer=39,
        sglang_return_hidden_xorl_component_name="layer_output",
        trace_id=[],
    )

    cmd = launch_k3_test.build_sglang_return_hidden_command(
        python_bin=Path("/venv/bin/python"),
        sglang_url="http://sglang:30000",
        args=args,
    )

    assert cmd[cmd.index("--hidden-kind") + 1] == "pre_final_norm"
    assert cmd[cmd.index("--input-mode") + 1] == "prompt"
    assert cmd[cmd.index("--xorl-result") + 1] == "/tmp/xorl.json"
    assert "--xorl-layer-index" not in cmd
    assert cmd[cmd.index("--xorl-component-layer") + 1] == "39"
    assert cmd[cmd.index("--xorl-component-name") + 1] == "layer_output"


def test_launch_gpu_capacity_rejects_ep8_when_no_node_has_eight_free_gpus():
    capacity = [
        {"name": "h100-a", "allocatable": 8, "used": 2, "free": 6},
        {"name": "h100-b", "allocatable": 8, "used": 2, "free": 6},
        {"name": "h100-c", "allocatable": 8, "used": 4, "free": 4},
    ]

    ok, reason = launch_k3_test.fit_gpu_requests(
        capacity,
        [("k3-sg-q36", 2, None), ("k3-xo-q36", 8, None)],
    )

    assert ok is False
    assert "k3-xo-q36" in reason
    assert "requesting 8 GPU" in reason
    assert "h100-a free=6" in reason


def test_launch_gpu_capacity_fits_ep8_gate_when_receiver_has_separate_room():
    capacity = [
        {"name": "h100-xorl", "allocatable": 8, "used": 0, "free": 8},
        {"name": "h100-sglang", "allocatable": 8, "used": 6, "free": 2},
    ]

    ok, reason = launch_k3_test.fit_gpu_requests(
        capacity,
        [("k3-sg-q36", 2, None), ("k3-xo-q36", 8, None)],
    )

    assert ok is True
    assert "k3-xo-q36->h100-xorl" in reason
    assert "k3-sg-q36->h100-sglang" in reason


def test_launch_gpu_capacity_returns_assignments_for_auto_selectors():
    capacity = [
        {"name": "h100-xorl", "allocatable": 8, "used": 0, "free": 8},
        {"name": "h100-sglang", "allocatable": 8, "used": 6, "free": 2},
    ]

    ok, reason, assignments = launch_k3_test.fit_gpu_request_assignments(
        capacity,
        [("k3-sg-q36", 2, None), ("k3-xo-q36", 8, None)],
    )

    assert ok is True
    assert "k3-xo-q36->h100-xorl" in reason
    assert assignments == {"k3-xo-q36": "h100-xorl", "k3-sg-q36": "h100-sglang"}


def test_gpu_capacity_wait_retries_until_capacity_fits(monkeypatch):
    snapshots = [
        [{"name": "h100-a", "allocatable": 8, "used": 5, "free": 3}],
        [{"name": "h100-a", "allocatable": 8, "used": 0, "free": 8}],
    ]
    sleeps = []
    clock = [0.0]

    def fake_collect_gpu_capacity(*, require_nccl_bench_passed: bool = False):
        assert require_nccl_bench_passed is True
        return snapshots.pop(0)

    def fake_sleep(seconds: float):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(launch_k3_test, "collect_gpu_capacity", fake_collect_gpu_capacity)

    ok, message, assignments, capacity, attempts = launch_k3_test.wait_for_gpu_capacity(
        [("k3-xo-q36", 8, None)],
        require_nccl_bench_passed=True,
        timeout_sec=60.0,
        poll_interval_sec=5.0,
        sleep_fn=fake_sleep,
        monotonic_fn=lambda: clock[0],
    )

    assert ok is True
    assert "k3-xo-q36->h100-a" in message
    assert assignments == {"k3-xo-q36": "h100-a"}
    assert capacity == [{"name": "h100-a", "allocatable": 8, "used": 0, "free": 8}]
    assert sleeps == [5.0]
    assert [attempt["status"] for attempt in attempts] == ["failed", "passed"]


def test_gpu_capacity_wait_times_out_with_final_snapshot(monkeypatch):
    clock = [0.0]
    sleeps = []

    def fake_collect_gpu_capacity(*, require_nccl_bench_passed: bool = False):
        del require_nccl_bench_passed
        return [{"name": "h100-a", "allocatable": 8, "used": 5, "free": 3}]

    def fake_sleep(seconds: float):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(launch_k3_test, "collect_gpu_capacity", fake_collect_gpu_capacity)

    ok, message, assignments, capacity, attempts = launch_k3_test.wait_for_gpu_capacity(
        [("k3-xo-q36", 8, None)],
        require_nccl_bench_passed=False,
        timeout_sec=10.0,
        poll_interval_sec=30.0,
        sleep_fn=fake_sleep,
        monotonic_fn=lambda: clock[0],
    )

    assert ok is False
    assert "cannot fit pod 'k3-xo-q36'" in message
    assert assignments == {}
    assert capacity == [{"name": "h100-a", "allocatable": 8, "used": 5, "free": 3}]
    assert sleeps == [10.0]
    assert [attempt["status"] for attempt in attempts] == ["failed", "failed"]


def test_capacity_selected_hostname_selector_uses_capacity_assignment():
    selector = launch_k3_test.capacity_selected_hostname_selector(
        enabled=True,
        pod_name="k3-xo-q36",
        explicit_node_name=None,
        explicit_selector_name=None,
        assignments={"k3-xo-q36": "h100-xorl"},
    )

    assert selector == "h100-xorl"


def test_capacity_selected_hostname_selector_preserves_explicit_selector():
    selector = launch_k3_test.capacity_selected_hostname_selector(
        enabled=True,
        pod_name="k3-xo-q36",
        explicit_node_name=None,
        explicit_selector_name="h100-explicit",
        assignments={"k3-xo-q36": "h100-xorl"},
    )

    assert selector == "h100-explicit"


def test_launch_gpu_capacity_prefers_healthier_equal_free_node():
    capacity = sorted(
        [
            {
                "name": "h100-unvetted",
                "allocatable": 8,
                "used": 0,
                "free": 8,
                "cuda_health": "ok",
                "nccl_bench_status": "",
            },
            {
                "name": "h100-passed",
                "allocatable": 8,
                "used": 0,
                "free": 8,
                "cuda_health": "ok",
                "nccl_bench_status": "passed",
            },
        ],
        key=launch_k3_test._capacity_sort_key,
        reverse=True,
    )

    ok, reason = launch_k3_test.fit_gpu_requests(
        capacity,
        [("k3-xo-q36", 8, None)],
    )

    assert ok is True
    assert "k3-xo-q36->h100-passed" in reason


def test_collect_gpu_capacity_can_require_nccl_passed(monkeypatch):
    nodes = {
        "items": [
            {
                "metadata": {
                    "name": "h100-unvetted",
                    "labels": {
                        "node-group": "default",
                        "nvidia.com/cuda-health": "ok",
                    },
                },
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
            {
                "metadata": {
                    "name": "h100-passed",
                    "labels": {
                        "node-group": "default",
                        "nvidia.com/cuda-health": "ok",
                        "nvidia.com/nccl-bench-status": "passed",
                    },
                },
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
            {
                "metadata": {
                    "name": "h100-unschedulable",
                    "labels": {
                        "node-group": "default",
                        "nvidia.com/cuda-health": "ok",
                        "nvidia.com/nccl-bench-status": "passed",
                    },
                },
                "spec": {"unschedulable": True},
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8"},
                    "conditions": [{"type": "Ready", "status": "True"}],
                },
            },
        ]
    }
    pods = {
        "items": [
            {
                "spec": {
                    "nodeName": "h100-passed",
                    "containers": [{"resources": {"requests": {"nvidia.com/gpu": "2"}}}],
                },
                "status": {"phase": "Running"},
            }
        ]
    }

    def fake_run(command, *, capture=False, **kwargs):
        del capture, kwargs
        if command == "kubectl get nodes -o json":
            return SimpleNamespace(stdout=json.dumps(nodes))
        if command == "kubectl get pods -A -o json":
            return SimpleNamespace(stdout=json.dumps(pods))
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(launch_k3_test, "run", fake_run)

    capacity = launch_k3_test.collect_gpu_capacity(require_nccl_bench_passed=True)

    assert capacity == [
        {
            "name": "h100-passed",
            "allocatable": 8,
            "used": 2,
            "free": 6,
            "cuda_health": "ok",
            "nccl_bench_status": "passed",
        }
    ]


def test_capacity_debug_artifact_records_ep8_preflight_snapshot(tmp_path):
    capacity = [
        {"name": "h100-a", "allocatable": 8, "used": 2, "free": 6},
        {"name": "h100-b", "allocatable": 8, "used": 6, "free": 2},
    ]
    requests = launch_k3_test.launch_gpu_requests(
        sglang_pod="k3-sg-q36",
        sglang_gpus=2,
        sglang_node_name=None,
        xorl_pod="k3-xo-q36",
        xorl_gpus=8,
        xorl_node_name=None,
    )
    ok, message = launch_k3_test.fit_gpu_requests(capacity, requests)

    launch_k3_test.write_capacity_debug_artifact(
        tmp_path,
        capacity=capacity,
        requests=requests,
        ok=ok,
        message=message,
    )

    artifact = json.loads((tmp_path / "capacity.json").read_text(encoding="utf-8"))

    assert artifact["status"] == "failed"
    assert artifact["assignments"] == {}
    assert artifact["attempts"] == []
    assert artifact["requests"] == [
        {"pod": "k3-sg-q36", "gpus": 2, "node_name": None},
        {"pod": "k3-xo-q36", "gpus": 8, "node_name": None},
    ]
    assert artifact["capacity"] == capacity
    assert "k3-xo-q36" in artifact["message"]


def test_capacity_debug_artifact_records_wait_attempts(tmp_path):
    attempts = [
        {
            "status": "failed",
            "message": "cannot fit pod 'k3-xo-q36'",
            "assignments": {},
            "capacity": [{"name": "h100-a", "allocatable": 8, "used": 5, "free": 3}],
        },
        {
            "status": "passed",
            "message": "capacity check passed: k3-xo-q36->h100-a",
            "assignments": {"k3-xo-q36": "h100-a"},
            "capacity": [{"name": "h100-a", "allocatable": 8, "used": 0, "free": 8}],
        },
    ]

    launch_k3_test.write_capacity_debug_artifact(
        tmp_path,
        capacity=attempts[-1]["capacity"],
        requests=[("k3-xo-q36", 8, None)],
        ok=True,
        message=attempts[-1]["message"],
        assignments=attempts[-1]["assignments"],
        attempts=attempts,
    )

    artifact = json.loads((tmp_path / "capacity.json").read_text(encoding="utf-8"))

    assert artifact["status"] == "passed"
    assert artifact["assignments"] == {"k3-xo-q36": "h100-a"}
    assert artifact["attempts"] == attempts


def _deepep_parity_record(phase: str, **overrides):
    record = {
        "tag": "xorl_deepep_parity_diagnostic",
        "record_id": 0,
        "phase": phase,
        "rank": 0,
        "ep_rank": 0,
        "ep_size": 2,
        "ep_dispatch": "deepep",
        "moe_implementation": "quack",
        "quack_deepep_no_permute": False,
        "fp8_training_enabled": False,
        "num_experts": 4,
        "num_local_experts": 2,
        "expected_num_local_experts": 2,
        "local_expert_global_range": [0, 2],
        "expected_local_expert_global_range": [0, 2],
        "num_local_experts_matches_expected": True,
        "cumsum_length": 2,
        "cumsum_length_matches_num_local_experts": True,
        "cumsum_length_matches_expected_num_local_experts": True,
        "hidden_states": {"shape": [2, 6], "dtype": "bfloat16"},
        "selected_experts": {
            "shape": [2, 2],
            "top_global_experts": [{"expert": 0, "count": 1}, {"expert": 1, "count": 1}],
        },
        "gate_up_proj": {"shape": [2, 6, 4], "dtype": "bfloat16"},
        "down_proj": {"shape": [2, 2, 6], "dtype": "bfloat16"},
        "permute_tokens": {"shape": [4, 6], "dtype": "bfloat16"},
        "cumsum": {"shape": [2], "last": 4, "top_counts": [{"local_expert": 1, "count": 3}]},
        "dispatch_ctx": {"type": "DispatchContext", "num_recv_tokens": 3, "num_valid": 4},
        "expert_output": None,
        "result": None,
    }
    record.update(overrides)
    return record


def test_deepep_parity_parser_summarizes_records_and_invariants(tmp_path):
    log_path = tmp_path / "xorl.log"
    records = [
        _deepep_parity_record("post_dispatch"),
        _deepep_parity_record("post_compute", expert_output={"shape": [4, 6], "dtype": "bfloat16"}),
        _deepep_parity_record(
            "post_combine",
            expert_output={"shape": [4, 6], "dtype": "bfloat16"},
            result={"shape": [2, 6], "dtype": "bfloat16"},
        ),
    ]
    log_path.write_text(
        "\n".join(
            f"2026-06-04T00:00:0{idx}Z [DEEPEP PARITY] {json.dumps(record)}" for idx, record in enumerate(records)
        ),
        encoding="utf-8",
    )

    output = tmp_path / "summary.json"
    summary = parse_deepep_parity_diagnostics.write_deepep_parity_summary([tmp_path], output)

    assert output.exists()
    assert summary["status"] == "pass"
    assert summary["failure_reasons"] == []
    assert summary["record_count"] == 3
    assert summary["by_phase"] == {"post_combine": 1, "post_compute": 1, "post_dispatch": 1}
    assert summary["layout_mismatch_count"] == 0
    assert summary["cumsum_mismatch_count"] == 0
    assert summary["result_shape_mismatch_count"] == 0
    assert summary["incomplete_phase_group_count"] == 0


def test_deepep_parity_parser_handles_concatenated_records_on_one_log_line(tmp_path):
    log_path = tmp_path / "xorl.log"
    first = _deepep_parity_record("post_dispatch", rank=0, ep_rank=0)
    second = _deepep_parity_record("post_dispatch", rank=1, ep_rank=1)
    log_path.write_text(
        f"prefix [DEEPEP PARITY] {json.dumps(first)}[DEEPEP PARITY] {json.dumps(second)}\n",
        encoding="utf-8",
    )

    records, errors = parse_deepep_parity_diagnostics.parse_deepep_parity_logs([log_path])

    assert errors == []
    assert [record["rank"] for record in records] == [0, 1]
    assert [record["_line_occurrence"] for record in records] == [0, 1]


def test_deepep_parity_parser_flags_cross_rank_result_fingerprint_mismatch(tmp_path):
    log_path = tmp_path / "xorl.log"
    records = []
    for rank, result_sha in [(0, "rank0"), (1, "rank1")]:
        hidden_states = {
            "shape": [2, 6],
            "dtype": "bfloat16",
            "fingerprint": {"sha256": "same-input"},
        }
        result = {
            "shape": [2, 6],
            "dtype": "bfloat16",
            "fingerprint": {"sha256": result_sha},
        }
        records.extend(
            [
                _deepep_parity_record("post_dispatch", rank=rank, ep_rank=rank, hidden_states=hidden_states),
                _deepep_parity_record(
                    "post_compute",
                    rank=rank,
                    ep_rank=rank,
                    hidden_states=hidden_states,
                    expert_output={"shape": [4, 6], "dtype": "bfloat16"},
                ),
                _deepep_parity_record(
                    "post_combine",
                    rank=rank,
                    ep_rank=rank,
                    hidden_states=hidden_states,
                    expert_output={"shape": [4, 6], "dtype": "bfloat16"},
                    result=result,
                ),
            ]
        )
    log_path.write_text("\n".join(f"[DEEPEP PARITY] {json.dumps(record)}" for record in records), encoding="utf-8")

    summary = parse_deepep_parity_diagnostics.write_deepep_parity_summary([log_path], tmp_path / "summary.json")

    assert summary["status"] == "failed"
    assert summary["failure_reasons"] == ["cross_rank_result_fingerprint_mismatches"]
    assert summary["cross_rank_result_fingerprint_mismatch_count"] == 1
    assert summary["cross_rank_result_fingerprint_mismatches"] == [
        {
            "record_id": 0,
            "phase": "post_combine",
            "hidden_states_fingerprint": "same-input",
            "result_fingerprints": ["rank0", "rank1"],
            "ranks": [0, 1],
        }
    ]


def test_deepep_parity_parser_flags_expert_output_reference_failure(tmp_path):
    log_path = tmp_path / "xorl.log"
    records = [
        _deepep_parity_record("post_dispatch"),
        _deepep_parity_record(
            "post_compute",
            expert_output={"shape": [4, 6], "dtype": "bfloat16"},
            expert_output_reference={
                "status": "failed",
                "reference_dtype": "float32",
                "compare_rows": 4,
                "total_rows": 4,
                "thresholds": {"max_abs": 0.01},
                "thresholds_exceeded": ["max_abs"],
                "diff": {"max_abs": 0.5, "mean_abs": 0.05, "p95_abs": 0.4, "nonfinite_diff_count": 0},
            },
        ),
        _deepep_parity_record(
            "post_combine",
            expert_output={"shape": [4, 6], "dtype": "bfloat16"},
            result={"shape": [2, 6], "dtype": "bfloat16"},
        ),
    ]
    log_path.write_text("\n".join(f"[DEEPEP PARITY] {json.dumps(record)}" for record in records), encoding="utf-8")

    summary = parse_deepep_parity_diagnostics.write_deepep_parity_summary([log_path], tmp_path / "summary.json")

    assert summary["status"] == "failed"
    assert summary["failure_reasons"] == ["expert_output_reference_failures"]
    assert summary["expert_output_reference_record_count"] == 1
    assert summary["expert_output_reference_statuses"] == {"failed": 1}
    assert summary["expert_output_reference_failure_count"] == 1
    assert summary["expert_output_reference_failures"][0]["max_abs"] == 0.5
    assert summary["expert_output_reference_failures"][0]["thresholds_exceeded"] == ["max_abs"]
    assert summary["expert_output_reference_worst"]["max_abs"]["max_abs"] == 0.5
    assert summary["expert_output_reference_worst"]["mean_abs"]["mean_abs"] == 0.05
    assert summary["expert_output_reference_worst"]["p95_abs"]["p95_abs"] == 0.4


def test_deepep_parity_parser_flags_result_reference_failure(tmp_path):
    log_path = tmp_path / "xorl.log"
    records = [
        _deepep_parity_record("post_dispatch"),
        _deepep_parity_record("post_compute", expert_output={"shape": [4, 6], "dtype": "bfloat16"}),
        _deepep_parity_record(
            "post_combine",
            expert_output={"shape": [4, 6], "dtype": "bfloat16"},
            result={"shape": [2, 6], "dtype": "bfloat16"},
            result_reference={
                "status": "failed",
                "reference_dtype": "float32",
                "compare_rows": 2,
                "total_rows": 2,
                "thresholds": {"max_abs": 0.01},
                "thresholds_exceeded": ["max_abs"],
                "local_contribution_count": 3,
                "global_contribution_count": 6,
                "diff": {"max_abs": 0.75, "mean_abs": 0.07, "p95_abs": 0.5, "nonfinite_diff_count": 0},
            },
        ),
    ]
    log_path.write_text("\n".join(f"[DEEPEP PARITY] {json.dumps(record)}" for record in records), encoding="utf-8")

    summary = parse_deepep_parity_diagnostics.write_deepep_parity_summary([log_path], tmp_path / "summary.json")

    assert summary["status"] == "failed"
    assert summary["failure_reasons"] == ["result_reference_failures"]
    assert summary["result_reference_record_count"] == 1
    assert summary["result_reference_statuses"] == {"failed": 1}
    assert summary["result_reference_failure_count"] == 1
    assert summary["result_reference_failures"][0]["max_abs"] == 0.75
    assert summary["result_reference_failures"][0]["global_contribution_count"] == 6
    assert summary["result_reference_worst"]["max_abs"]["max_abs"] == 0.75
    assert summary["result_reference_worst"]["mean_abs"]["mean_abs"] == 0.07
    assert summary["result_reference_worst"]["p95_abs"]["p95_abs"] == 0.5


def test_deepep_parity_reference_compare_matches_plain_expert_loop(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE", "1")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE", "fp32")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", "2")

    permute_tokens = torch.randn(3, 4, dtype=torch.float32)
    cumsum = torch.tensor([2, 3], dtype=torch.int32)
    gate_up_proj = torch.randn(2, 4, 6, dtype=torch.float32)
    down_proj = torch.randn(2, 3, 4, dtype=torch.float32)
    expert_scores = torch.tensor([0.5, 1.0, 0.25], dtype=torch.float32)

    expected_parts = []
    start = 0
    for expert_idx, end in enumerate(cumsum.tolist()):
        tokens = permute_tokens[start:end]
        gate_up = tokens.matmul(gate_up_proj[expert_idx])
        gate, up = gate_up.split(3, dim=-1)
        expected_parts.append(apply_moe_activation("silu", gate, up).matmul(down_proj[expert_idx]))
        start = end
    expert_output = torch.cat(expected_parts, dim=0) * expert_scores.unsqueeze(-1)

    summary = _safe_expert_output_reference_comparison(
        permute_tokens=permute_tokens,
        cumsum=cumsum,
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        intermediate_size=3,
        expert_scores=expert_scores,
        hidden_act="silu",
        gate_up_bias=None,
        down_bias=None,
        expert_output=expert_output,
    )

    assert summary["status"] == "observed"
    assert summary["reference_dtype"] == "float32"
    assert summary["compare_rows"] == 3
    assert summary["diff"]["max_abs"] == 0.0
    assert summary["diff"]["mean_abs"] == 0.0


def test_deepep_parity_result_reference_compare_matches_plain_moe_loop(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE", "1")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE", "fp32")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", "2")

    hidden_states = torch.randn(3, 4, dtype=torch.float32)
    selected_experts = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.long)
    routing_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32)
    gate_up_proj = torch.randn(2, 4, 6, dtype=torch.float32)
    down_proj = torch.randn(2, 3, 4, dtype=torch.float32)

    expected = torch.zeros_like(hidden_states)
    for expert_idx in range(2):
        mask = selected_experts == expert_idx
        rows, topk = mask.nonzero(as_tuple=True)
        gate_up = hidden_states.index_select(0, rows).matmul(gate_up_proj[expert_idx])
        gate, up = gate_up.split(3, dim=-1)
        out = apply_moe_activation("silu", gate, up).matmul(down_proj[expert_idx])
        out = out * routing_weights[rows, topk].unsqueeze(-1)
        expected.index_add_(0, rows, out)

    summary = _safe_result_reference_comparison(
        hidden_states=hidden_states,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        intermediate_size=3,
        hidden_act="silu",
        gate_up_bias=None,
        down_bias=None,
        result=expected,
        ep_group=None,
    )

    assert summary["status"] == "observed"
    assert summary["reference_dtype"] == "float32"
    assert summary["compare_rows"] == 3
    assert summary["local_contribution_count"] == 6
    assert summary["global_contribution_count"] == 6
    assert summary["diff"]["max_abs"] == 0.0
    assert summary["diff"]["mean_abs"] == 0.0


def test_deepep_parity_reference_compare_samples_across_rows(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE", "1")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE", "fp32")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS", "2")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES", "2")

    permute_tokens = torch.randn(6, 4, dtype=torch.float32)
    cumsum = torch.tensor([3, 6], dtype=torch.int32)
    gate_up_proj = torch.randn(2, 4, 6, dtype=torch.float32)
    down_proj = torch.randn(2, 3, 4, dtype=torch.float32)

    expected_parts = []
    start = 0
    for expert_idx, end in enumerate(cumsum.tolist()):
        tokens = permute_tokens[start:end]
        gate_up = tokens.matmul(gate_up_proj[expert_idx])
        gate, up = gate_up.split(3, dim=-1)
        expected_parts.append(apply_moe_activation("silu", gate, up).matmul(down_proj[expert_idx]))
        start = end
    expert_output = torch.cat(expected_parts, dim=0)
    expert_output[-1, 0] += 1.0

    summary = _safe_expert_output_reference_comparison(
        permute_tokens=permute_tokens,
        cumsum=cumsum,
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        intermediate_size=3,
        expert_scores=None,
        hidden_act="silu",
        gate_up_bias=None,
        down_bias=None,
        expert_output=expert_output,
    )

    assert summary["status"] == "observed"
    assert summary["compare_rows"] == 2
    assert summary["total_rows"] == 6
    assert summary["row_limited"] is True
    assert summary["row_sample_strategy"] == "evenly_spaced"
    assert summary["row_indices_head"] == [0, 5]
    assert summary["row_indices_tail"] == [0, 5]
    assert summary["diff"]["max_abs"] == pytest.approx(1.0)


def test_deepep_parity_diff_summary_samples_large_p95(monkeypatch):
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_P95_MAX_ELEMS", "16")
    actual = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    expected = torch.zeros_like(actual)

    summary = _diff_summary(actual, expected)

    assert summary["compared_elements"] == 64
    assert summary["max_abs"] == 63.0
    assert summary["mean_abs"] == 31.5
    assert summary["p95_sampled"] is True
    assert summary["p95_sample_size"] == 16
    assert summary["p95_abs"] is not None


def test_deepep_parity_evenly_spaced_indices_stay_in_bounds_for_large_numel():
    indices = _evenly_spaced_int64_indices(64_000_003, 1024, torch.device("cpu"))

    assert indices.dtype == torch.long
    assert int(indices[0].item()) == 0
    assert int(indices[-1].item()) == 64_000_002
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) < 64_000_003


def test_deepep_parity_record_start_skips_early_records(monkeypatch):
    _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS.clear()
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC", "1")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS", "0")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START", "3")
    monkeypatch.setenv("XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS", "2")

    observed = [_acquire_deepep_parity_diagnostic_record() for _ in range(6)]

    assert observed == [None, None, None, 3, 4, None]
    _DEEPEP_PARITY_DIAGNOSTIC_RECORD_COUNTS.clear()


def test_ep_parity_dispatch_context_summary_covers_alltoall_context():
    ctx = SimpleNamespace(
        input_splits=[1, 2],
        output_splits=[3, 4],
        num_tokens_per_expert=torch.tensor([2, 3], dtype=torch.int32),
        routing_map=torch.tensor([[True, False], [False, True]]),
        perm_mapping=torch.tensor([1, 0]),
        expert_scores=torch.tensor([0.25, 0.75]),
        orig_shape=torch.Size([2, 4]),
        num_experts=4,
    )

    summary = _dispatch_context_summary(ctx)

    assert summary["type"] == "SimpleNamespace"
    assert summary["input_splits"] == [1, 2]
    assert summary["output_splits"] == [3, 4]
    assert summary["orig_shape"] == [2, 4]
    assert summary["num_experts"] == 4
    assert summary["num_tokens_per_expert"]["shape"] == [2]
    assert summary["routing_map"]["shape"] == [2, 2]
    assert summary["perm_mapping"]["shape"] == [2]
    assert summary["expert_scores"]["shape"] == [2]


def test_deepep_parity_parser_flags_layout_and_shape_mismatches(tmp_path):
    log_path = tmp_path / "xorl.log"
    bad = _deepep_parity_record(
        "post_combine",
        num_local_experts=4,
        cumsum={"shape": [4], "last": 5},
        cumsum_length=4,
        result={"shape": [3, 6], "dtype": "bfloat16"},
    )
    log_path.write_text(f"prefix [DEEPEP PARITY] {json.dumps(bad)}\n", encoding="utf-8")

    summary = parse_deepep_parity_diagnostics.write_deepep_parity_summary([log_path], tmp_path / "summary.json")

    assert summary["status"] == "failed"
    assert summary["failure_reasons"] == [
        "layout_mismatches",
        "cumsum_mismatches",
        "result_shape_mismatches",
        "incomplete_phase_groups",
    ]
    assert summary["record_count"] == 1
    assert summary["layout_mismatch_count"] == 1
    assert summary["cumsum_mismatch_count"] == 3
    assert {mismatch["kind"] for mismatch in summary["cumsum_mismatches"]} == {
        "cumsum_length_vs_expected_num_local_experts",
        "cumsum_vs_num_valid",
        "cumsum_vs_permute_tokens",
    }
    assert summary["result_shape_mismatch_count"] == 1
    assert summary["incomplete_phase_group_count"] == 1


def test_deepep_parity_parser_flags_missing_expected_ranks(tmp_path):
    log_path = tmp_path / "xorl.log"
    records = [
        _deepep_parity_record("post_dispatch"),
        _deepep_parity_record("post_compute", expert_output={"shape": [4, 6], "dtype": "bfloat16"}),
        _deepep_parity_record(
            "post_combine",
            expert_output={"shape": [4, 6], "dtype": "bfloat16"},
            result={"shape": [2, 6], "dtype": "bfloat16"},
        ),
    ]
    log_path.write_text(
        "\n".join(f"[DEEPEP PARITY] {json.dumps(record)}" for record in records),
        encoding="utf-8",
    )

    summary = parse_deepep_parity_diagnostics.write_deepep_parity_summary(
        [log_path],
        tmp_path / "summary.json",
        expected_rank_count=2,
    )

    assert summary["status"] == "failed"
    assert summary["failure_reasons"] == ["rank_coverage_mismatch"]
    assert summary["rank_coverage"] == {
        "expected_rank_count": 2,
        "expected_ranks": [0, 1],
        "observed_ranks": [0],
        "missing_ranks": [1],
        "unexpected_ranks": [],
        "unparseable_ranks": [],
    }


def test_launch_k3_writes_deepep_parity_summary_when_logs_contain_records(tmp_path):
    record = _deepep_parity_record("post_dispatch")
    (tmp_path / "k3-xo-test.log").write_text(f"[DEEPEP PARITY] {json.dumps(record)}\n", encoding="utf-8")

    launch_k3_test.write_deepep_parity_summary_if_present(tmp_path)

    summary = json.loads((tmp_path / "deepep_parity_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failure_reasons"] == ["incomplete_phase_groups"]
    assert summary["record_count"] == 1
    assert summary["by_phase"] == {"post_dispatch": 1}


def test_launch_k3_deepep_expected_rank_count_only_for_all_ranks():
    args = argparse.Namespace(xorl_deepep_parity_diagnostic="1", xorl_deepep_parity_diagnostic_ranks="all")
    assert launch_k3_test.deepep_expected_rank_count(args, xorl_gpus=8) == 8

    args.xorl_deepep_parity_diagnostic_ranks = "*"
    assert launch_k3_test.deepep_expected_rank_count(args, xorl_gpus=4) == 4

    args.xorl_deepep_parity_diagnostic_ranks = "0,3"
    assert launch_k3_test.deepep_expected_rank_count(args, xorl_gpus=8) is None

    args.xorl_deepep_parity_diagnostic = ""
    args.xorl_deepep_parity_diagnostic_ranks = "all"
    assert launch_k3_test.deepep_expected_rank_count(args, xorl_gpus=8) is None


def test_xorl_pod_template_threads_fp8_linear_error_profiler_env():
    manifest = launch_k3_test.render_template(
        launch_k3_test.K8S_DIR / "xorl-pod.yaml.tmpl",
        {
            "POD_NAME": "k3-xo-test",
            "NAMESPACE": "apanda",
            "MODEL_LABEL": "qwen3-8b",
            "XORL_CONFIG": "config.yaml",
            "NUM_GPUS": "1",
            "XORL_PORT": "8000",
            "MASTER_PORT": "29510",
            "XORL_VENV": "/venv",
            "XORL_MOE_SYNTHETIC_ROUTING_VALUE": "balanced",
            "XORL_WEIGHT_SYNC_DENSE_BUCKET_BYTES_VALUE": "",
            "XORL_WEIGHT_SYNC_TIMINGS_VALUE": "",
            "XORL_WEIGHT_SYNC_FP8_TARGET_DEVICE_VALUE": "",
            "XORL_P2P_FP8_QUANTIZE_DEVICE_VALUE": "",
            "XORL_WEIGHT_SYNC_FP8_CPU_H2D_CHUNK_BYTES_VALUE": "",
            "XORL_WEIGHT_SYNC_FP8_CPU_H2D_ASYNC_VALUE": "",
            "XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT_VALUE": "",
            "XORL_WEIGHT_SYNC_REINIT_PER_BUCKET_VALUE": "",
            "XORL_WEIGHT_SYNC_WAIT_AFTER_RECEIVER_VALUE": "",
            "XORL_WEIGHT_SYNC_NCCL_TWO_PHASE_VALUE": "1",
            "XORL_WEIGHT_SYNC_NCCL_CHUNK_BYTES_VALUE": "",
            "XORL_WEIGHT_SYNC_DEBUG_TENSOR_STATS_VALUE": "",
            "XORL_WEIGHT_SYNC_SKIP_PARAM_PATTERNS_VALUE": "model.embed_tokens.weight,lm_head.weight",
            "XORL_FP8_LINEAR_ERROR_PROFILE_VALUE": "1",
            "XORL_FP8_LINEAR_ERROR_PROFILE_OUTPUT_VALUE": "/tmp/fp8-profile.json",
            "XORL_FP8_LINEAR_ERROR_PROFILE_MAX_CALLS_PER_MODULE_VALUE": "2",
            "XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS_VALUE": "8",
            "XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES_VALUE": "1444,1445",
            "XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE_VALUE": "0",
            "XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE_VALUE": "2048",
            "XORL_QUACK_DEEPEP_SCATTER_CHUNK_TOKENS_VALUE": "8192",
            "XORL_QUACK_DEEPEP_FORCE_GENERIC_VALUE": "1",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_VALUE": "1",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS_VALUE": "0,3",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS_VALUE": "4",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START_VALUE": "32",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES_VALUE": "5",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK_VALUE": "6",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_FINGERPRINT_MAX_ELEMS_VALUE": "4096",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE_VALUE": "1",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE_VALUE": "fp32",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS_VALUE": "1024",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ABS_VALUE": "0.01",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MEAN_ABS_VALUE": "0.001",
            "NCCL_P2P_DISABLE_VALUE": "",
            "NCCL_SHM_DISABLE_VALUE": "",
            "CUDA_LAUNCH_BLOCKING_VALUE": "1",
            "NODE_SELECTOR_BLOCK": "",
            "NODE_NAME_BLOCK": "",
            "HOME_DIR": "/home/apanda",
            "REPO_DIR": "/repo",
            "PVC_NAME": "home-apanda",
        },
    )

    assert "team: turbo" in manifest
    assert "privileged: true" not in manifest
    assert "export CUDA_VISIBLE_DEVICES=" not in manifest
    assert "export XORL_MOE_SYNTHETIC_ROUTING=balanced" in manifest
    assert 'for module in ("deep_ep", "torch")' in manifest
    assert "export NCCL_NSOCKS_PERTHREAD=8" in manifest
    assert "export NCCL_SOCKET_NTHREADS=4" in manifest
    assert "export NCCL_BUFFSIZE=8388608" in manifest
    assert "export NCCL_NET_GDR_LEVEL=PHB" in manifest
    assert "export NCCL_SOCKET_IFNAME=bond0" in manifest
    assert "export XORL_WEIGHT_SYNC_NCCL_TWO_PHASE=1" in manifest
    assert 'export XORL_WEIGHT_SYNC_SKIP_PARAM_PATTERNS="model.embed_tokens.weight,lm_head.weight"' in manifest
    assert "export XORL_FP8_LINEAR_ERROR_PROFILE=1" in manifest
    assert "export XORL_FP8_LINEAR_ERROR_PROFILE_OUTPUT=/tmp/fp8-profile.json" in manifest
    assert "export XORL_FP8_LINEAR_ERROR_PROFILE_MAX_CALLS_PER_MODULE=2" in manifest
    assert "export XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS=8" in manifest
    assert 'export XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES="1444,1445"' in manifest
    assert "export XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE=0" in manifest
    assert "export XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE=2048" in manifest
    assert "export XORL_QUACK_DEEPEP_SCATTER_CHUNK_TOKENS=8192" in manifest
    assert "export XORL_QUACK_DEEPEP_FORCE_GENERIC=1" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC=1" in manifest
    assert 'export XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS="0,3"' in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS=4" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START=32" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES=5" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK=6" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_FINGERPRINT_MAX_ELEMS=4096" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE=1" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE=fp32" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS=1024" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ABS=0.01" in manifest
    assert "export XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MEAN_ABS=0.001" in manifest
    assert "export CUDA_LAUNCH_BLOCKING=1" in manifest
    assert '"XORL_WEIGHT_SYNC_SKIP_PARAM_PATTERNS"' in manifest
    assert '"XORL_FP8_LINEAR_ERROR_PROFILE_OUTPUT"' in manifest
    assert '"XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES"' in manifest
    assert '"XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE"' in manifest
    assert '"XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE"' in manifest
    assert '"XORL_QUACK_DEEPEP_SCATTER_CHUNK_TOKENS"' in manifest
    assert '"XORL_QUACK_DEEPEP_FORCE_GENERIC"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_FINGERPRINT_MAX_ELEMS"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ABS"' in manifest
    assert '"XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MEAN_ABS"' in manifest
    assert '"CUDA_LAUNCH_BLOCKING"' in manifest


def test_xorl_pod_template_can_pin_with_hostname_selector_without_node_name():
    node_name = "research-common-h100-073.cloud.together.ai"
    manifest = launch_k3_test.render_template(
        launch_k3_test.K8S_DIR / "xorl-pod.yaml.tmpl",
        {
            "POD_NAME": "k3-xo-test",
            "NAMESPACE": "apanda",
            "MODEL_LABEL": "qwen3-8b",
            "XORL_CONFIG": "config.yaml",
            "NUM_GPUS": "8",
            "XORL_PORT": "8000",
            "MASTER_PORT": "29510",
            "XORL_VENV": "/venv",
            "XORL_MOE_SYNTHETIC_ROUTING_VALUE": "",
            "XORL_WEIGHT_SYNC_DENSE_BUCKET_BYTES_VALUE": "",
            "XORL_WEIGHT_SYNC_TIMINGS_VALUE": "",
            "XORL_WEIGHT_SYNC_FP8_TARGET_DEVICE_VALUE": "",
            "XORL_P2P_FP8_QUANTIZE_DEVICE_VALUE": "",
            "P2P_TRAINER_IB_DEVICES_PER_RANK_VALUE": "",
            "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP_VALUE": "",
            "P2P_TRAINER_IB_DEVICE_VALUE": "",
            "P2P_TRAINER_VISIBLE_GPU_INDICES_VALUE": "",
            "XORL_WEIGHT_SYNC_FP8_CPU_H2D_CHUNK_BYTES_VALUE": "",
            "XORL_WEIGHT_SYNC_FP8_CPU_H2D_ASYNC_VALUE": "",
            "XORL_WEIGHT_SYNC_SGLANG_LOAD_FORMAT_VALUE": "",
            "XORL_WEIGHT_SYNC_REINIT_PER_BUCKET_VALUE": "",
            "XORL_WEIGHT_SYNC_WAIT_AFTER_RECEIVER_VALUE": "",
            "XORL_WEIGHT_SYNC_NCCL_TWO_PHASE_VALUE": "",
            "XORL_WEIGHT_SYNC_NCCL_CHUNK_BYTES_VALUE": "",
            "XORL_WEIGHT_SYNC_DEBUG_TENSOR_STATS_VALUE": "",
            "XORL_WEIGHT_SYNC_SKIP_PARAM_PATTERNS_VALUE": "",
            "XORL_FP8_LINEAR_ERROR_PROFILE_VALUE": "",
            "XORL_FP8_LINEAR_ERROR_PROFILE_OUTPUT_VALUE": "",
            "XORL_FP8_LINEAR_ERROR_PROFILE_MAX_CALLS_PER_MODULE_VALUE": "",
            "XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS_VALUE": "",
            "XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES_VALUE": "",
            "XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE_VALUE": "",
            "XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE_VALUE": "",
            "XORL_QUACK_DEEPEP_SCATTER_CHUNK_TOKENS_VALUE": "",
            "XORL_QUACK_DEEPEP_FORCE_GENERIC_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_RANKS_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_RECORDS_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_RECORD_START_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_MAX_VALUES_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_HIST_TOPK_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_FINGERPRINT_MAX_ELEMS_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_COMPARE_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_DTYPE_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ROWS_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MAX_ABS_VALUE": "",
            "XORL_DEEPEP_PARITY_DIAGNOSTIC_REFERENCE_MEAN_ABS_VALUE": "",
            "NCCL_P2P_DISABLE_VALUE": "",
            "NCCL_SHM_DISABLE_VALUE": "",
            "CUDA_LAUNCH_BLOCKING_VALUE": "",
            "NODE_SELECTOR_BLOCK": launch_k3_test.hostname_node_selector_block(node_name),
            "NODE_NAME_BLOCK": "",
            "HOME_DIR": "/home/apanda",
            "REPO_DIR": "/repo",
            "PVC_NAME": "home-apanda",
        },
    )

    assert "node-group: default" in manifest
    assert f"kubernetes.io/hostname: {node_name}" in manifest
    assert "nodeName:" not in manifest


def test_sglang_pod_template_threads_optional_fp8_receiver_args():
    node_name = "research-common-h100-014.cloud.together.ai"
    manifest = launch_k3_test.render_template(
        launch_k3_test.K8S_DIR / "sglang-pod.yaml.tmpl",
        {
            "POD_NAME": "k3-sg-test",
            "NAMESPACE": "apanda",
            "MODEL_LABEL": "qwen3-0.6b",
            "MODEL_PATH": "Qwen/Qwen3-0.6B",
            "TP_SIZE": "1",
            "NUM_GPUS": "1",
            "SGLANG_PORT": "30000",
            "SGLANG_PYTHON": "/venv/bin/python",
            "SGLANG_REPO": "/sglang/python",
            "SGLANG_QUANTIZATION_VALUE": "fp8",
            "SGLANG_LOAD_FORMAT_VALUE": "flash_rl",
            "SGLANG_KV_CACHE_DTYPE_VALUE": "fp8_e4m3",
            "SGLANG_MOE_RUNNER_BACKEND_VALUE": "cutlass",
            "SGLANG_ENABLE_RDMA_WEIGHT_UPDATES_VALUE": "1",
            "SGLANG_MOONCAKE_IB_DEVICE_VALUE": "mlx5_0",
            "SGLANG_DISABLE_RADIX_CACHE_VALUE": "1",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE_VALUE": "1",
            "SGLANG_ENABLE_FP32_ROUTER_VALUE": "1",
            "SGLANG_ENABLE_RETURN_HIDDEN_STATES_VALUE": "1",
            "SGLANG_RETURN_PRE_FINAL_NORM_HIDDEN_VALUE": "1",
            "SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER_VALUE": "/home/apanda/k3_sglang_debug/test",
            "SGLANG_DEBUG_TENSOR_DUMP_LAYERS_VALUE": "33 34",
            "TENSOR_DUMP_CAPTURE_PARENT_MODULES_VALUE": "1",
            "TENSOR_DUMP_CAPTURE_MODULE_INPUTS_VALUE": "1",
            "XORL_WEIGHT_SYNC_NCCL_CHUNK_BYTES_VALUE": "1048576",
            "XORL_WEIGHT_SYNC_DEBUG_TENSOR_STATS_VALUE": "1",
            "XORL_FP8_MOE_RUNTIME_FINGERPRINT_VALUE": "1",
            "XORL_FP8_MOE_INNER_FINGERPRINT_VALUE": "1",
            "XORL_FP8_MOE_RUNTIME_FINGERPRINT_MAX_ELEMS_VALUE": "4096",
            "XORL_FP8_MOE_ZERO_CUTLASS_OUTPUT_VALUE": "1",
            "XORL_FP8_MOE_SYNC_AFTER_CUTLASS_VALUE": "1",
            "XORL_FP8_MOE_FRESH_CUTLASS_BUFFERS_VALUE": "1",
            "XORL_WEIGHT_SYNC_SKIP_RECEIVER_SYNC_AFTER_RECV_VALUE": "",
            "NCCL_P2P_DISABLE_VALUE": "",
            "NCCL_SHM_DISABLE_VALUE": "",
            "CUDA_LAUNCH_BLOCKING_VALUE": "",
            "NODE_SELECTOR_BLOCK": launch_k3_test.hostname_node_selector_block(node_name),
            "NODE_NAME_BLOCK": "",
            "HOME_DIR": "/home/apanda",
            "PVC_NAME": "home-apanda",
        },
    )

    assert "team: turbo" in manifest
    assert f"kubernetes.io/hostname: {node_name}" in manifest
    assert "nodeName:" not in manifest
    assert "privileged: true" not in manifest
    assert "export CUDA_VISIBLE_DEVICES=" not in manifest
    assert 'SGLANG_QUANTIZATION_ARG="fp8"' in manifest
    assert 'SGLANG_LOAD_FORMAT_ARG="flash_rl"' in manifest
    assert 'SGLANG_KV_CACHE_DTYPE_ARG="fp8_e4m3"' in manifest
    assert 'SGLANG_MOE_RUNNER_BACKEND_ARG="cutlass"' in manifest
    assert 'SGLANG_ENABLE_RDMA_WEIGHT_UPDATES_ARG="1"' in manifest
    assert 'SGLANG_MOONCAKE_IB_DEVICE_ARG="mlx5_0"' in manifest
    assert 'SGLANG_DISABLE_RADIX_CACHE_ARG="1"' in manifest
    assert 'SGLANG_ENABLE_DETERMINISTIC_INFERENCE_ARG="1"' in manifest
    assert 'SGLANG_ENABLE_FP32_ROUTER_ARG="1"' in manifest
    assert "export SGLANG_RETURN_PRE_FINAL_NORM_HIDDEN=1" in manifest
    assert 'echo "SGLANG_RETURN_PRE_FINAL_NORM_HIDDEN=${SGLANG_RETURN_PRE_FINAL_NORM_HIDDEN:-<unset>}"' in manifest
    assert 'SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER_ARG="/home/apanda/k3_sglang_debug/test"' in manifest
    assert 'SGLANG_DEBUG_TENSOR_DUMP_LAYERS_ARG="33 34"' in manifest
    assert "export TENSOR_DUMP_CAPTURE_PARENT_MODULES=1" in manifest
    assert "export TENSOR_DUMP_CAPTURE_MODULE_INPUTS=1" in manifest
    assert 'echo "TENSOR_DUMP_CAPTURE_PARENT_MODULES=${TENSOR_DUMP_CAPTURE_PARENT_MODULES:-<unset>}"' in manifest
    assert 'echo "TENSOR_DUMP_CAPTURE_MODULE_INPUTS=${TENSOR_DUMP_CAPTURE_MODULE_INPUTS:-<unset>}"' in manifest
    assert 'SGLANG_EXTRA_ARGS+=(--quantization "${SGLANG_QUANTIZATION_ARG}")' in manifest
    assert 'SGLANG_EXTRA_ARGS+=(--load-format "${SGLANG_LOAD_FORMAT_ARG}")' in manifest
    assert 'SGLANG_EXTRA_ARGS+=(--kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE_ARG}")' in manifest
    assert 'SGLANG_EXTRA_ARGS+=(--moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND_ARG}")' in manifest
    assert "SGLANG_EXTRA_ARGS+=(--enable-rdma-weight-updates)" in manifest
    assert 'SGLANG_EXTRA_ARGS+=(--mooncake-ib-device "${SGLANG_MOONCAKE_IB_DEVICE_ARG}")' in manifest
    assert "SGLANG_EXTRA_ARGS+=(--disable-radix-cache)" in manifest
    assert "SGLANG_EXTRA_ARGS+=(--enable-deterministic-inference)" in manifest
    assert "SGLANG_EXTRA_ARGS+=(--enable-fp32-router)" in manifest
    assert "SGLANG_EXTRA_ARGS+=(--enable-return-hidden-states)" in manifest
    assert (
        'SGLANG_EXTRA_ARGS+=(--debug-tensor-dump-output-folder "${SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER_ARG}")'
    ) in manifest
    assert 'SGLANG_EXTRA_ARGS+=(--debug-tensor-dump-layers "${SGLANG_DEBUG_TENSOR_DUMP_LAYER_ITEMS[@]}")' in manifest
    assert 'echo "SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION_ARG:-<unset>}"' in manifest
    assert 'echo "SGLANG_LOAD_FORMAT=${SGLANG_LOAD_FORMAT_ARG:-<unset>}"' in manifest
    assert 'echo "SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE_ARG:-<unset>}"' in manifest
    assert 'echo "SGLANG_MOE_RUNNER_BACKEND=${SGLANG_MOE_RUNNER_BACKEND_ARG:-<unset>}"' in manifest
    assert 'echo "SGLANG_ENABLE_RDMA_WEIGHT_UPDATES=${SGLANG_ENABLE_RDMA_WEIGHT_UPDATES_ARG:-<unset>}"' in manifest
    assert 'echo "SGLANG_MOONCAKE_IB_DEVICE=${SGLANG_MOONCAKE_IB_DEVICE_ARG:-<unset>}"' in manifest
    assert 'echo "SGLANG_DISABLE_RADIX_CACHE=${SGLANG_DISABLE_RADIX_CACHE_ARG:-<unset>}"' in manifest
    assert (
        'echo "SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE_ARG:-<unset>}"'
    ) in manifest
    assert 'echo "SGLANG_ENABLE_FP32_ROUTER=${SGLANG_ENABLE_FP32_ROUTER_ARG:-<unset>}"' in manifest
    assert ('echo "SGLANG_ENABLE_RETURN_HIDDEN_STATES=${SGLANG_ENABLE_RETURN_HIDDEN_STATES_ARG:-<unset>}"') in manifest
    assert (
        'echo "SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER=${SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER_ARG:-<unset>}"'
    ) in manifest
    assert 'echo "SGLANG_DEBUG_TENSOR_DUMP_LAYERS=${SGLANG_DEBUG_TENSOR_DUMP_LAYERS_ARG:-<unset>}"' in manifest
    assert 'rm -rf "${SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER_ARG}"' in manifest
    assert "export XORL_FP8_MOE_RUNTIME_FINGERPRINT=1" in manifest
    assert 'echo "XORL_FP8_MOE_RUNTIME_FINGERPRINT=${XORL_FP8_MOE_RUNTIME_FINGERPRINT:-<unset>}"' in manifest
    assert "export XORL_FP8_MOE_INNER_FINGERPRINT=1" in manifest
    assert 'echo "XORL_FP8_MOE_INNER_FINGERPRINT=${XORL_FP8_MOE_INNER_FINGERPRINT:-<unset>}"' in manifest
    assert "export XORL_FP8_MOE_RUNTIME_FINGERPRINT_MAX_ELEMS=4096" in manifest
    assert (
        'echo "XORL_FP8_MOE_RUNTIME_FINGERPRINT_MAX_ELEMS=${XORL_FP8_MOE_RUNTIME_FINGERPRINT_MAX_ELEMS:-<unset>}"'
    ) in manifest
    assert "export XORL_FP8_MOE_ZERO_CUTLASS_OUTPUT=1" in manifest
    assert 'echo "XORL_FP8_MOE_ZERO_CUTLASS_OUTPUT=${XORL_FP8_MOE_ZERO_CUTLASS_OUTPUT:-<unset>}"' in manifest
    assert "export XORL_FP8_MOE_SYNC_AFTER_CUTLASS=1" in manifest
    assert 'echo "XORL_FP8_MOE_SYNC_AFTER_CUTLASS=${XORL_FP8_MOE_SYNC_AFTER_CUTLASS:-<unset>}"' in manifest
    assert "export XORL_FP8_MOE_FRESH_CUTLASS_BUFFERS=1" in manifest
    assert 'echo "XORL_FP8_MOE_FRESH_CUTLASS_BUFFERS=${XORL_FP8_MOE_FRESH_CUTLASS_BUFFERS:-<unset>}"' in manifest
    assert '"${SGLANG_EXTRA_ARGS[@]}"' in manifest
    assert "--fp8-gemm-backend triton" in manifest


def test_resolve_live_static_traces_output_defaults_to_log_dir(tmp_path):
    assert launch_k3_test.resolve_live_static_traces_output(None, None, tmp_path) == str(
        tmp_path / "static_traces.json"
    )
    assert launch_k3_test.resolve_live_static_traces_output(None, "/tmp/custom.json", tmp_path) == "/tmp/custom.json"
    assert launch_k3_test.resolve_live_static_traces_output("/tmp/input.json", "/tmp/custom.json", tmp_path) is None


def test_refresh_trace_can_regenerate_and_attach_reference_metadata(monkeypatch):
    trace = static_trace_utils.normalize_trace(
        {
            "trace_id": "t0",
            "prompt_ids": [1, 2],
            "output_ids": [3],
            "sglang_logprobs": [-1.0],
        }
    )

    monkeypatch.setattr(
        refresh_static_traces,
        "sglang_generate",
        lambda *args, **kwargs: {
            "output_ids": [4],
            "text": "new",
            "meta_info": {"output_token_logprobs": [[-0.25, 4, "new"]]},
        },
    )
    monkeypatch.setattr(
        refresh_static_traces,
        "sglang_score",
        lambda *args, **kwargs: ([-0.2], [[[-0.2, 4, "new"]]], None),
    )

    refreshed = refresh_static_traces.refresh_trace(
        trace,
        sglang_url="http://sg",
        max_new_tokens=1,
        top_logprobs_num=1,
        regenerate_outputs=True,
        check_generation=False,
    )

    assert refreshed["output_ids"] == [4]
    assert refreshed["sglang_logprobs"] == [-0.2]
    assert refreshed["sglang_generation_logprobs"] == [-0.25]
    assert refreshed["reference_refresh"]["output_ids_changed"] is True
    assert refreshed["reference_refresh"]["previous_prefill_abs_diff"]["max"] == pytest.approx(0.8)


def test_extract_k3_repro_traces_selects_worst_unique_trace_ids(tmp_path):
    traces_file = tmp_path / "traces.json"
    static_trace_utils.write_static_trace_file(
        traces_file,
        {"model_name": "m"},
        [
            {"trace_id": "bad", "prompt_ids": [1], "output_ids": [2], "sglang_logprobs": [-1.0]},
            {"trace_id": "also-bad", "prompt_ids": [3], "output_ids": [4], "sglang_logprobs": [-2.0]},
            {"trace_id": "ok", "prompt_ids": [5], "output_ids": [6], "sglang_logprobs": [-3.0]},
        ],
    )
    k3_result = tmp_path / "k3.json"
    k3_result.write_text(
        """
        {
          "diagnostics": {
            "worst_samples": [
              {"trace_id": "bad", "sample_k3_mean": 10.0},
              {"trace_id": "also-bad", "sample_k3_mean": 5.0}
            ],
            "worst_tokens": [
              {"trace_id": "bad", "position": 0, "k3": 99.0},
              {"trace_id": "ok", "position": 0, "k3": 7.0}
            ]
          }
        }
        """
    )
    output = tmp_path / "repro.json"

    result = extract_k3_repro_traces.extract_traces(
        traces_file=traces_file,
        k3_result=k3_result,
        output=output,
        top_traces=1,
        top_tokens=2,
        explicit_trace_ids=[],
    )
    metadata, traces = static_trace_utils.load_static_trace_file(output)

    assert result["selected_trace_ids"] == ["bad", "ok"]
    assert [trace["trace_id"] for trace in traces] == ["bad", "ok"]
    assert metadata["repro_selection"]["selected_trace_ids"] == ["bad", "ok"]


def test_slice_static_trace_token_builds_one_target_full_prefix():
    trace = {
        "trace_id": "row11",
        "trace_mode": "sglang_generation",
        "prompt_ids": [1, 2, 3],
        "output_ids": [4, 5, 6],
        "full_ids": [1, 2, 3, 4, 5, 6],
        "sglang_logprobs": [-0.4, -0.5, -0.6],
        "sglang_generation_logprobs": [-0.41, -0.51, -0.61],
        "sglang_top_logprobs": [[[-0.4, 4, "a"]], [[-0.5, 5, "b"]], [[-0.6, 6, "c"]]],
    }

    sliced = slice_static_trace_token.slice_trace_token(trace, position=1)

    assert sliced["trace_id"] == "row11:pos1:target"
    assert sliced["prompt_ids"] == [1, 2, 3, 4]
    assert sliced["prompt_len"] == 4
    assert sliced["output_ids"] == [5]
    assert sliced["full_ids"] == [1, 2, 3, 4, 5]
    assert sliced["sglang_logprobs"] == [-0.5]
    assert sliced["sglang_generation_logprobs"] == [-0.51]
    assert sliced["sglang_top_logprobs"] == [[[-0.5, 5, "b"]]]
    assert sliced["source_trace_slice"]["source_absolute_position"] == 4
    assert sliced["source_trace_slice"]["source_prefix_start"] == 0
    assert sliced["source_trace_slice"]["reference_context_preserved"] is True
    assert static_trace_utils.xorl_input_ids_for_trace(sliced) == [1, 2, 3, 4]
    assert static_trace_utils.labels_for_trace(sliced) == [-100, -100, -100, 5]


def test_slice_static_trace_token_can_limit_prefix_and_write_bundle(tmp_path):
    traces_file = tmp_path / "traces.json"
    output = tmp_path / "one-token.json"
    static_trace_utils.write_static_trace_file(
        traces_file,
        {"model_name": "m"},
        [
            {
                "trace_id": "row11",
                "prompt_ids": [1, 2, 3],
                "output_ids": [4, 5, 6],
                "sglang_logprobs": [-0.4, -0.5, -0.6],
            }
        ],
    )

    result = slice_static_trace_token.extract_token_slice(
        traces_file=traces_file,
        output=output,
        trace_id="row11",
        position=1,
        prefix_tokens=2,
    )
    metadata, traces = static_trace_utils.load_static_trace_file(output)
    sliced = static_trace_utils.normalize_trace(traces[0])

    assert result["output"] == str(output)
    assert result["prompt_len"] == 2
    assert result["output_ids"] == [5]
    assert metadata["token_slice"]["source_trace_id"] == "row11"
    assert metadata["token_slice"]["source_position"] == 1
    assert metadata["token_slice"]["source_prefix_start"] == 2
    assert metadata["token_slice"]["prefix_tokens"] == 2
    assert metadata["token_slice"]["reference_context_preserved"] is False
    assert "refresh or rescore" in metadata["token_slice"]["reference_note"]
    assert sliced["prompt_ids"] == [3, 4]
    assert sliced["output_ids"] == [5]
