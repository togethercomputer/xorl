"""E2E tests for LoRA fine-tuning with Qwen3-8B."""

import os

import pytest

from tests.e2e.e2e_utils import (
    QWEN3_8B_ID,
    generate_training_config,
    run_training,
    skip_if_gpu_count_less_than,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.slow]


class TestLoRA1GPU:

    @skip_if_gpu_count_less_than(1)
    def test_lora_loss_converges(self, tmp_workspace):
        """Qwen3-8B LoRA training shows strong loss convergence over 20 steps."""
        output_dir = os.path.join(tmp_workspace, "output_lora_converge")
        config_path = generate_training_config(
            model_dir=QWEN3_8B_ID,
            model_path=QWEN3_8B_ID,
            output_dir=output_dir,
            num_gpus=1,
            max_steps=20,
            lr=1e-3,
            enable_lora=True,
            lora_rank=32,
            lora_alpha=32,
            merge_qkv=False,
        )
        result = run_training(config_path, num_gpus=1, timeout=600)

        result.assert_success()
        result.assert_loss_converged(max_final_loss=8.0, min_drop_ratio=0.30)


class TestLoRA2GPU:

    @skip_if_gpu_count_less_than(2)
    def test_lora_fsdp2(self, tmp_workspace):
        """Qwen3-8B LoRA + FSDP2 on 2 GPUs converges."""
        output_dir = os.path.join(tmp_workspace, "output_lora_fsdp2")
        config_path = generate_training_config(
            model_dir=QWEN3_8B_ID,
            model_path=QWEN3_8B_ID,
            output_dir=output_dir,
            num_gpus=2,
            dp_shard_size=2,
            max_steps=20,
            lr=1e-3,
            enable_lora=True,
            lora_rank=32,
            lora_alpha=32,
            merge_qkv=False,
        )
        result = run_training(config_path, num_gpus=2, timeout=600)

        result.assert_success()
        result.assert_loss_converged(max_final_loss=8.0, min_drop_ratio=0.30)

    @skip_if_gpu_count_less_than(2)
    def test_lora_checkpoint_save_and_resume(self, tmp_workspace):
        """Qwen3-8B LoRA checkpoint save and resume round-trip."""
        # Phase 1: Train 5 steps, save at step 3
        output_dir_1 = os.path.join(tmp_workspace, "output_lora_ckpt_p1")
        config_path_1 = generate_training_config(
            model_dir=QWEN3_8B_ID,
            model_path=QWEN3_8B_ID,
            output_dir=output_dir_1,
            num_gpus=2,
            dp_shard_size=2,
            max_steps=5,
            save_steps=3,
            lr=1e-3,
            enable_lora=True,
            lora_rank=32,
            lora_alpha=32,
            merge_qkv=False,
        )
        result_1 = run_training(config_path_1, num_gpus=2, timeout=600)
        result_1.assert_success()

        ckpt_path = os.path.join(output_dir_1, "checkpoints", "global_step_3")
        assert os.path.isdir(ckpt_path), f"Phase 1 checkpoint missing: {ckpt_path}"

        # Phase 2: Resume from step 3, train to step 10
        output_dir_2 = os.path.join(tmp_workspace, "output_lora_ckpt_p2")
        config_path_2 = generate_training_config(
            model_dir=QWEN3_8B_ID,
            model_path=QWEN3_8B_ID,
            output_dir=output_dir_2,
            num_gpus=2,
            dp_shard_size=2,
            max_steps=10,
            lr=1e-3,
            enable_lora=True,
            lora_rank=32,
            lora_alpha=32,
            merge_qkv=False,
            extra_train={"load_checkpoint_path": ckpt_path},
        )
        result_2 = run_training(config_path_2, num_gpus=2, timeout=600)

        result_2.assert_success()
        assert result_2.global_step == 10
