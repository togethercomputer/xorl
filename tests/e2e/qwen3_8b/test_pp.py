"""E2E tests for pipeline parallelism with Qwen3-8B (dense).

Tests the PP fixes: micro-batch padding, correct loss normalization (fsdp_size=1),
and metadata queue guard during no_grad shape inference.

Tests both the direct trainer path and the server (ModelRunner) path.

GPU layouts:
    TestPP2GPU:        PP=2, FSDP=1  (2 GPUs) — minimal PP correctness
    TestPP8GPU:        PP=2, FSDP=4  (8 GPUs) — PP + FSDP combination
    TestPP2GPUServer:  PP=2, FSDP=1  (2 GPUs) — server PP path
    TestPP8GPUServer:  PP=2, FSDP=4  (8 GPUs) — server PP + FSDP
"""

import os

import pytest

from tests.e2e.e2e_utils import (
    generate_training_config,
    run_training,
    skip_if_gpu_count_less_than,
)
from tests.e2e.server_utils import (
    ServerProcess,
    _create_full_weight_client,
    _get_free_port,
    _start_server_or_fail,
    assert_loss_decreases,
    generate_random_sft_data,
    generate_server_config,
    run_sft_steps,
)


pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


class TestPP2GPU:
    @skip_if_gpu_count_less_than(2)
    def test_pp2_loss_converges(self, tiny_dense_model_dir):
        """Trainer: PP=2 on 2 GPUs — loss must be finite and decreasing."""
        output_dir = os.path.join(tiny_dense_model_dir, "output_pp2")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir,
            output_dir=output_dir,
            num_gpus=2,
            pp_size=2,
            dp_shard_size=1,
            gradient_accumulation_steps=2,
            packing_seq_len=256,
            max_steps=5,
            lr=1e-3,
        )
        result = run_training(config_path, num_gpus=2, timeout=600)
        result.assert_success()
        result.assert_loss_converged(max_final_loss=12.0, min_drop_ratio=0.001)


class TestPP8GPU:
    @skip_if_gpu_count_less_than(8)
    def test_pp2_fsdp4_loss_converges(self, tiny_dense_model_dir):
        """Trainer: PP=2 + FSDP=4 on 8 GPUs — validates fsdp_size=1 fix."""
        output_dir = os.path.join(tiny_dense_model_dir, "output_pp2_fsdp4")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir,
            output_dir=output_dir,
            num_gpus=8,
            pp_size=2,
            dp_shard_size=4,
            gradient_accumulation_steps=2,
            packing_seq_len=256,
            max_steps=5,
            lr=1e-3,
        )
        result = run_training(config_path, num_gpus=8, timeout=600)
        result.assert_success()
        result.assert_loss_converged(max_final_loss=12.0, min_drop_ratio=0.001)

    @skip_if_gpu_count_less_than(8)
    def test_pp2_fsdp4_muon_loss_converges(self, tiny_dense_model_dir):
        """Trainer: PP=2 + FSDP=4 + Muon on 8 GPUs."""
        output_dir = os.path.join(tiny_dense_model_dir, "output_pp2_fsdp4_muon")
        config_path = generate_training_config(
            model_dir=tiny_dense_model_dir,
            output_dir=output_dir,
            num_gpus=8,
            pp_size=2,
            dp_shard_size=4,
            gradient_accumulation_steps=2,
            packing_seq_len=256,
            optimizer="muon",
            max_steps=10,
            lr=5e-4,
            extra_train={"muon_lr": 2e-3, "muon_momentum": 0.95},
        )
        result = run_training(config_path, num_gpus=8, timeout=600)
        result.assert_success()
        result.assert_loss_converged(max_final_loss=12.0, min_drop_ratio=0.001)


# ---------------------------------------------------------------------------
# Server tests
# ---------------------------------------------------------------------------


class TestPP2GPUServer:
    @skip_if_gpu_count_less_than(2)
    def test_pp2_server_loss_decreases(self, tiny_dense_model_dir_with_weights):
        """Server: PP=2 on 2 GPUs — validates server PP padding + loss normalization fix."""
        output_dir = os.path.join(tiny_dense_model_dir_with_weights, "output_server_pp2")
        api_port = _get_free_port()

        config_path = generate_server_config(
            model_dir=tiny_dense_model_dir_with_weights,
            output_dir=output_dir,
            num_gpus=2,
            pp_size=2,
            dp_shard_size=1,
            sample_packing_sequence_len=256,
            enable_lora=False,
        )

        server = ServerProcess(config_path, num_gpus=2, api_port=api_port, output_dir=output_dir)
        try:
            _start_server_or_fail(server, timeout=300)
            _, training_client = _create_full_weight_client(server.base_url, tiny_dense_model_dir_with_weights)
            data = generate_random_sft_data(num_samples=4, seq_len=64, vocab_size=32000)
            losses = run_sft_steps(training_client, data, num_steps=10, lr=1e-2)
            assert_loss_decreases(losses, msg="PP=2 server loss should decrease")
        finally:
            server.stop()


class TestPP8GPUServer:
    @skip_if_gpu_count_less_than(8)
    def test_pp2_fsdp4_server_loss_decreases(self, tiny_dense_model_dir_with_weights):
        """Server: PP=2 + FSDP=4 on 8 GPUs — validates reported_loss fix (no /fsdp_size)."""
        output_dir = os.path.join(tiny_dense_model_dir_with_weights, "output_server_pp2_fsdp4")
        api_port = _get_free_port()

        config_path = generate_server_config(
            model_dir=tiny_dense_model_dir_with_weights,
            output_dir=output_dir,
            num_gpus=8,
            pp_size=2,
            dp_shard_size=4,
            sample_packing_sequence_len=256,
            enable_lora=False,
        )

        server = ServerProcess(config_path, num_gpus=8, api_port=api_port, output_dir=output_dir)
        try:
            _start_server_or_fail(server, timeout=300)
            _, training_client = _create_full_weight_client(server.base_url, tiny_dense_model_dir_with_weights)
            data = generate_random_sft_data(num_samples=4, seq_len=64, vocab_size=32000)
            losses = run_sft_steps(training_client, data, num_steps=10, lr=1e-2)
            assert_loss_decreases(losses, msg="PP=2+FSDP=4 server loss should decrease")
        finally:
            server.stop()
