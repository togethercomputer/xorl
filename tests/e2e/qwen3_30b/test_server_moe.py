"""E2E server tests for Qwen3-30B-A3B MoE with real model weights."""

import math
import os
import re

import pytest

from tests.e2e.e2e_utils import skip_if_gpu_count_less_than
from tests.e2e.server_utils import (
    ServerProcess,
    _create_lora_client,
    _get_free_port,
    _start_server_or_fail,
    generate_random_sft_data,
    generate_server_config,
    run_sft_steps,
)


pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.server, pytest.mark.slow]


class TestQwen3_30B_MoE:
    """Real Qwen3-30B-A3B MoE server tests."""

    @pytest.fixture
    def tmp_workspace(self, tmp_path):
        return str(tmp_path)

    @skip_if_gpu_count_less_than(4)
    @pytest.mark.slow
    def test_moe_ep2_efsd2_hybrid_shared_lora_sft(self, tmp_workspace, monkeypatch):
        """Qwen3-30B-A3B native MoE EP=2 + eFSDP=2 with hybrid shared LoRA."""
        # Exercise the expensive correctness diagnostics in the production
        # topology gate: shared gradients/weights/moments must agree across EP
        # replicas, while expert-local rectangles remain single-owner. The
        # positive reducer telemetry assertion proves this is the shared-factor
        # path rather than the vacuous disjoint-expert path.
        monkeypatch.setenv("XORL_VALIDATE_ADAPTER_REPLICAS", "1")
        model_dir = "Qwen/Qwen3-30B-A3B"
        output_dir = os.path.join(tmp_workspace, "output_moe_ep2_efsd2")
        api_port = _get_free_port()

        config_path = generate_server_config(
            model_dir=model_dir,
            output_dir=output_dir,
            num_gpus=4,
            dp_shard_size=4,
            ep_size=2,
            moe_implementation="native",
            enable_lora=True,
            lora_rank=8,
            moe_hybrid_shared_lora=True,
        )

        server = ServerProcess(config_path, num_gpus=4, api_port=api_port, output_dir=output_dir)
        try:
            _start_server_or_fail(server, timeout=600)

            _, training_client = _create_lora_client(server.base_url, model_dir, model_id="test-moe-ep2-efsd2")
            data = generate_random_sft_data(num_samples=8, seq_len=64, vocab_size=151936)
            losses = run_sft_steps(training_client, data, num_steps=2)
            assert all(not math.isnan(l) for l in losses), f"NaN in losses: {losses}"
            assert all(l > 0 for l in losses), f"Non-positive loss: {losses}"
            log = server.get_log()
            assert "hybrid_shared=True" in log
            assert re.search(
                r"EP replicated gradient sync: configured_parameters=[1-9][0-9]* "
                r"participating_parameters=[1-9][0-9]* buckets=[1-9][0-9]* "
                r"gradient_bytes=[1-9][0-9]* reduced_bytes=[1-9][0-9]*",
                log,
            ), "shared-factor reducer telemetry was not emitted"
        finally:
            server.stop()
