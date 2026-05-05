"""E2E tests for DistSignSGD with tiny Qwen3 dense models."""

import math
import os

import pytest

from tests.e2e.e2e_utils import (
    generate_training_config,
    run_training,
    skip_if_gpu_count_less_than,
)


pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.slow]


class TestDistSignSGD2GPU:
    @skip_if_gpu_count_less_than(2)
    def test_distsignsgd_fsdp2_runs_with_finite_loss(self, tiny_dense_model_dir):
        """Trainer: DistSignSGD on 2-GPU FSDP2 completes and reports finite metrics."""
        output_dir = os.path.join(tiny_dense_model_dir, "output_distsignsgd_fsdp2")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir,
            output_dir=output_dir,
            num_gpus=2,
            dp_shard_size=2,
            gradient_accumulation_steps=2,
            packing_seq_len=256,
            optimizer="distsignsgd",
            max_steps=5,
            lr=1e-3,
        )

        result = run_training(config_path, num_gpus=2, timeout=600)

        result.assert_success()
        assert result.global_step == 5
        assert result.final_loss is not None and not math.isnan(result.final_loss)
        assert result.final_loss < 12.0
        assert result.loss_history is not None and len(result.loss_history) == 5

    @skip_if_gpu_count_less_than(4)
    def test_distsignsgd_with_ulysses_outside_fsdp_and_gradient_accumulation_runs(
        self,
        tiny_dense_model_dir_with_weights,
    ):
        """Trainer: DistSignSGD keeps Ulysses exact-sum outside FSDP and accumulates mean sign votes."""
        output_dir = os.path.join(tiny_dense_model_dir_with_weights, "output_distsignsgd_fsdp2_u2_dp2_gas2")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir_with_weights,
            model_path=tiny_dense_model_dir_with_weights,
            output_dir=output_dir,
            num_gpus=4,
            dp_shard_size=2,
            ulysses_size=2,
            gradient_accumulation_steps=2,
            packing_seq_len=256,
            optimizer="distsignsgd",
            max_steps=5,
            lr=1e-3,
            extra_train={"cp_fsdp_mode": "none"},
        )

        result = run_training(config_path, num_gpus=4, timeout=900)

        result.assert_success()
        assert result.global_step == 5
        assert result.final_loss is not None and not math.isnan(result.final_loss)
        assert result.final_loss < 12.0
        assert result.loss_history is not None and len(result.loss_history) == 5
