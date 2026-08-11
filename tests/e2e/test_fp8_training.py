"""E2E smokes for full-weight FP8 training through the CLI trainer."""

import math
import os
from ctypes import CDLL
from pathlib import Path

import pytest

from tests.e2e.e2e_utils import (
    generate_training_config,
    run_training,
    skip_if_gpu_count_less_than,
    skip_if_no_flash_attn,
    skip_if_no_quack,
)


pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _prepend_library_path(path: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if path not in existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path


def _install_nvidia_ml_library_path() -> None:
    try:
        CDLL("libnvidia-ml.so.1")
        return
    except OSError:
        pass

    for stub in (
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"),
        Path("/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"),
    ):
        if not stub.exists():
            continue
        stub_dir = Path("/tmp/xorl-nvidia-ml-stub")
        stub_dir.mkdir(exist_ok=True)
        soname = stub_dir / "libnvidia-ml.so.1"
        if not soname.exists():
            soname.symlink_to(stub)
        _prepend_library_path(str(stub_dir))
        return


def _install_nvshmem_library_path() -> None:
    import nvidia.nvshmem  # noqa: PLC0415

    nvshmem_lib = os.path.join(list(nvidia.nvshmem.__path__)[0], "lib")
    _prepend_library_path(nvshmem_lib)


def _assert_fp8_training_smoke_succeeded(result, expected_steps: int, *, expect_moe: bool) -> None:
    result.assert_success()
    assert result.global_step == expected_steps
    assert result.loss_history is not None and len(result.loss_history) == expected_steps
    assert all(math.isfinite(loss) for loss in result.loss_history)
    assert result.final_loss is not None and math.isfinite(result.final_loss)
    assert result.final_grad_norm is not None and math.isfinite(result.final_grad_norm)
    _assert_fp8_metrics_all_used(result, expect_moe=expect_moe)


def _assert_fp8_metrics_all_used(result, *, expect_moe: bool) -> None:
    fp8_metrics = result.metrics.get("fp8_training")
    assert fp8_metrics is not None
    assert fp8_metrics["global_linear_modules"] > 0
    rank0_linear_modules = fp8_metrics["rank0_linear_modules"]
    assert rank0_linear_modules > 0
    reporting_ranks = fp8_metrics["global_linear_modules"] // rank0_linear_modules
    assert fp8_metrics["global_linear_modules"] == rank0_linear_modules * reporting_ranks
    assert fp8_metrics["global_linear_modules_used_fp8"] == (fp8_metrics["global_linear_modules"] - reporting_ranks)
    assert fp8_metrics["global_linear_modules_not_used_fp8"] == reporting_ranks
    assert fp8_metrics["global_linear_modules_allow_bf16_fallback"] == 0
    assert fp8_metrics["global_linear_modules_backward_fp8"] == fp8_metrics["global_linear_modules"]
    # Qwen3's resolved numerical program intentionally keeps the output head
    # in FP32, so every injected FP8Linear except lm_head must use FP8.
    assert fp8_metrics["rank0_unused_linear_module_names"] == ["lm_head"]

    if expect_moe:
        assert fp8_metrics["global_moe_modules"] > 0
        assert fp8_metrics["global_moe_fp8_enabled_modules"] == fp8_metrics["global_moe_modules"]
        assert fp8_metrics["global_moe_modules_used_fp8"] == fp8_metrics["global_moe_modules"]
        assert fp8_metrics["global_moe_modules_not_used_fp8"] == 0
        assert fp8_metrics["rank0_unused_moe_module_names"] == []
        assert fp8_metrics["global_moe_quack_modules"] == fp8_metrics["global_moe_modules"]
    else:
        assert fp8_metrics["global_moe_modules"] == 0
        assert fp8_metrics["global_moe_fp8_enabled_modules"] == 0
        assert fp8_metrics["global_moe_modules_used_fp8"] == 0
        assert fp8_metrics["global_moe_modules_not_used_fp8"] == 0
        assert fp8_metrics["global_moe_quack_modules"] == 0


class TestFP8FullWeightTrainer:
    @skip_if_gpu_count_less_than(1)
    def test_dense_full_weight_fp8_checkpoint_save_and_resume(self, tiny_dense_model_dir_with_weights):
        """Trainer: FP8 compute training can save a DCP checkpoint and resume."""
        phase1_steps = 2
        output_dir_1 = os.path.join(tiny_dense_model_dir_with_weights, "output_fp8_dense_ckpt_phase1")
        config_path_1 = generate_training_config(
            model_dir=tiny_dense_model_dir_with_weights,
            model_path=tiny_dense_model_dir_with_weights,
            output_dir=output_dir_1,
            attn_implementation="eager",
            num_gpus=1,
            seq_len=32,
            packing_seq_len=64,
            max_steps=phase1_steps,
            save_steps=1,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_allow_bf16_fallback": False,
            },
        )

        result_1 = run_training(config_path_1, num_gpus=1, timeout=900)

        _assert_fp8_training_smoke_succeeded(result_1, expected_steps=phase1_steps, expect_moe=False)
        ckpt_path = os.path.join(output_dir_1, "checkpoints", "global_step_1")
        assert os.path.isdir(ckpt_path), f"Phase 1 checkpoint missing: {ckpt_path}"

        output_dir_2 = os.path.join(tiny_dense_model_dir_with_weights, "output_fp8_dense_ckpt_phase2")
        config_path_2 = generate_training_config(
            model_dir=tiny_dense_model_dir_with_weights,
            model_path=tiny_dense_model_dir_with_weights,
            output_dir=output_dir_2,
            attn_implementation="eager",
            num_gpus=1,
            seq_len=32,
            packing_seq_len=64,
            max_steps=3,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_allow_bf16_fallback": False,
                "load_checkpoint_path": ckpt_path,
            },
        )

        result_2 = run_training(config_path_2, num_gpus=1, timeout=900)

        result_2.assert_success()
        assert f"Loaded checkpoint from {ckpt_path}" in result_2.stdout
        assert result_2.global_step == 3
        assert result_2.final_loss is not None and math.isfinite(result_2.final_loss)
        assert result_2.final_grad_norm is not None and math.isfinite(result_2.final_grad_norm)
        assert result_2.loss_history is not None and result_2.loss_history
        assert all(math.isfinite(loss) for loss in result_2.loss_history)
        _assert_fp8_metrics_all_used(result_2, expect_moe=False)

    @skip_if_gpu_count_less_than(2)
    def test_dense_full_weight_fp8_tensor_parallel_cli_training_runs(self, tiny_dense_model_dir_with_weights):
        """Trainer: tiny dense Qwen3 keeps lm_head on the FP8 path under tensor parallelism."""
        max_steps = 2
        output_dir = os.path.join(tiny_dense_model_dir_with_weights, "output_fp8_dense_tp")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir_with_weights,
            model_path=tiny_dense_model_dir_with_weights,
            output_dir=output_dir,
            attn_implementation="eager",
            num_gpus=2,
            tp_size=2,
            seq_len=32,
            packing_seq_len=64,
            max_steps=max_steps,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_allow_bf16_fallback": False,
            },
        )

        result = run_training(config_path, num_gpus=2, timeout=900)

        _assert_fp8_training_smoke_succeeded(result, expected_steps=max_steps, expect_moe=False)

    @skip_if_gpu_count_less_than(2)
    def test_dense_full_weight_fp8_ulysses_long_packed_cli_training_runs(self, tiny_dense_model_dir_with_weights):
        """Trainer: FP8 dense compute remains active through longer packed Ulysses CP shapes."""
        max_steps = 2
        output_dir = os.path.join(tiny_dense_model_dir_with_weights, "output_fp8_dense_ulysses_long_packed")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir_with_weights,
            model_path=tiny_dense_model_dir_with_weights,
            output_dir=output_dir,
            attn_implementation="eager",
            num_gpus=2,
            ulysses_size=2,
            dp_shard_size=1,
            seq_len=256,
            packing_seq_len=512,
            max_steps=max_steps,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_allow_bf16_fallback": False,
                "cp_fsdp_mode": "none",
            },
        )

        result = run_training(config_path, num_gpus=2, timeout=1200)

        _assert_fp8_training_smoke_succeeded(result, expected_steps=max_steps, expect_moe=False)

    @skip_if_gpu_count_less_than(2)
    @skip_if_no_flash_attn
    def test_dense_full_weight_fp8_ring_attention_cli_training_runs(self, tiny_dense_model_dir_with_weights):
        """Trainer: FP8 dense compute composes with Ring context parallel attention."""
        pytest.importorskip("flash_attn_interface")

        max_steps = 2
        output_dir = os.path.join(tiny_dense_model_dir_with_weights, "output_fp8_dense_ring_attention")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir_with_weights,
            model_path=tiny_dense_model_dir_with_weights,
            output_dir=output_dir,
            attn_implementation="flash_attention_3",
            num_gpus=2,
            dp_shard_size=1,
            seq_len=256,
            packing_seq_len=512,
            max_steps=max_steps,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_data={"dataset_num_proc": 1},
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_allow_bf16_fallback": False,
                "ringattn_parallel_size": 2,
                "cp_fsdp_mode": "none",
            },
        )

        result = run_training(config_path, num_gpus=2, timeout=1200)

        _assert_fp8_training_smoke_succeeded(result, expected_steps=max_steps, expect_moe=False)

    @skip_if_gpu_count_less_than(1)
    @skip_if_no_quack
    def test_moe_full_weight_fp8_cli_training_runs(self, tiny_moe_model_dir):
        """Trainer: tiny Qwen3-MoE uses FP8 dense Linear and Quack grouped expert compute."""
        max_steps = 2
        output_dir = os.path.join(tiny_moe_model_dir, "output_fp8_moe")
        config_path = generate_training_config(
            model_dir=tiny_moe_model_dir,
            output_dir=output_dir,
            attn_implementation="eager",
            moe_implementation="triton",
            num_gpus=1,
            seq_len=32,
            packing_seq_len=64,
            max_steps=max_steps,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_moe_grouped_backend": "triton_grouped",
                "fp8_training_allow_bf16_fallback": False,
            },
        )

        result = run_training(config_path, num_gpus=1, timeout=600)

        _assert_fp8_training_smoke_succeeded(result, expected_steps=max_steps, expect_moe=True)

    @skip_if_gpu_count_less_than(4)
    @skip_if_no_quack
    def test_moe_full_weight_fp8_deepep_ep_efsdp_cli_training_runs(self, small_moe_model_dir_with_weights):
        """Trainer: tiny Qwen3-MoE uses FP8 expert compute with DeepEP, EP, and eFSDP."""
        pytest.importorskip("deep_ep")
        pytest.importorskip("nvidia.nvshmem")
        _install_nvidia_ml_library_path()
        _install_nvshmem_library_path()

        max_steps = 2
        output_dir = os.path.join(small_moe_model_dir_with_weights, "output_fp8_moe_deepep_ep_efsdp")
        config_path = generate_training_config(
            model_dir=small_moe_model_dir_with_weights,
            model_path=small_moe_model_dir_with_weights,
            output_dir=output_dir,
            attn_implementation="eager",
            moe_implementation="quack",
            num_gpus=4,
            ep_size=2,
            dp_shard_size=4,
            seq_len=32,
            packing_seq_len=64,
            max_steps=max_steps,
            lr=1e-3,
            enable_gradient_checkpointing=False,
            extra_model={
                "ep_dispatch": "deepep",
                "deepep_buffer_size_gb": 0.25,
                "deepep_num_sms": 20,
            },
            extra_train={
                "enable_fp8_training": True,
                "fp8_training_backward": "fp8",
                "fp8_training_moe_grouped_backend": "triton_grouped",
                "fp8_training_allow_bf16_fallback": False,
            },
        )

        result = run_training(
            config_path,
            num_gpus=4,
            timeout=1200,
            extra_env={
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "XORL_MOE_SYNTHETIC_ROUTING": "balanced",
            },
        )

        _assert_fp8_training_smoke_succeeded(result, expected_steps=max_steps, expect_moe=True)
