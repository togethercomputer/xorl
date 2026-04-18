"""E2E tests for pipeline parallelism with Qwen3-30B-A3B (MoE).

Tests PP with EP=4 + CP=4 (ring attention) where EP and CP are folded onto
the same GPU axis, requiring only 8 GPUs (PP=2 * 4 folded EP/CP).

Tests both the direct trainer path and the server (ModelRunner) path.
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

# EP=4 and CP=4 are folded onto the same GPU axis: PP=2 * 4 = 8 GPUs total.
NUM_GPUS = 8
PP_SIZE = 2
EP_SIZE = 4
RINGATTN_SIZE = 4


class TestPP30BTrainer:
    @skip_if_gpu_count_less_than(NUM_GPUS)
    def test_pp2_ep4_cp4_muon_loss_converges(self, tiny_moe_model_dir):
        """Trainer: PP=2 + EP=4 + CP=4 (folded, 8 GPUs) with Muon."""
        output_dir = os.path.join(tiny_moe_model_dir, "output_trainer_pp2_ep4_cp4")
        config_path = generate_training_config(
            model_dir=tiny_moe_model_dir,
            output_dir=output_dir,
            moe_implementation="triton",
            num_gpus=NUM_GPUS,
            pp_size=PP_SIZE,
            ep_size=EP_SIZE,
            dp_shard_size=1,
            gradient_accumulation_steps=2,
            seq_len=64,
            packing_seq_len=128,
            optimizer="muon",
            max_steps=10,
            lr=1e-3,
            extra_train={
                "ringattn_parallel_size": RINGATTN_SIZE,
                "muon_lr": 2e-3,
                "muon_momentum": 0.95,
                "reshard_after_forward": True,
            },
        )
        result = run_training(config_path, num_gpus=NUM_GPUS, timeout=600)
        result.assert_success()
        result.assert_loss_converged(max_final_loss=12.0, min_drop_ratio=0.001)


class TestPP30BServer:
    @skip_if_gpu_count_less_than(NUM_GPUS)
    def test_pp2_ep4_cp4_server_loss_decreases(self, tiny_moe_model_dir_with_weights):
        """Server: PP=2 + EP=4 + CP=4 (folded, 8 GPUs) — loss must be finite and decreasing."""
        output_dir = os.path.join(tiny_moe_model_dir_with_weights, "output_server_pp2_ep4_cp4")
        api_port = _get_free_port()

        config_path = generate_server_config(
            model_dir=tiny_moe_model_dir_with_weights,
            output_dir=output_dir,
            moe_implementation="triton",
            num_gpus=NUM_GPUS,
            pp_size=PP_SIZE,
            ep_size=EP_SIZE,
            dp_shard_size=1,
            sample_packing_sequence_len=128,
            enable_lora=False,
            extra_config={
                "ringattn_parallel_size": RINGATTN_SIZE,
                "reshard_after_forward": True,
            },
        )

        server = ServerProcess(config_path, num_gpus=NUM_GPUS, api_port=api_port, output_dir=output_dir)
        try:
            _start_server_or_fail(server, timeout=300)
            _, training_client = _create_full_weight_client(server.base_url, tiny_moe_model_dir_with_weights)
            data = generate_random_sft_data(num_samples=4, seq_len=33, vocab_size=32000)
            losses = run_sft_steps(training_client, data, num_steps=10, lr=1e-2)
            assert_loss_decreases(losses, msg="PP=2+EP=4+CP=4 server loss should decrease")
        finally:
            server.stop()
