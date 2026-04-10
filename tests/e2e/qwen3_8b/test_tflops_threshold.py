"""TFLOPS threshold tests for Qwen3-8B on Hopper (H100) GPUs.

Runs LoRA SFT benchmarks with the real Qwen3-8B model and asserts
that achieved TFLOPS meets minimum thresholds. Thresholds are set at
80% of corrected baselines on H100 80GB HBM3 with flash_attention_3.

Corrected baselines (Qwen3-8B LoRA rank=8, seq_len=4096, 4 samples, 10 steps):
    1 GPU:  ~416 TFLOPS  -> threshold 333
    2 GPUs: ~391 TFLOPS  -> threshold 313
    4 GPUs: ~383 TFLOPS  -> threshold 306

These tests are slow (~2-3 min each) and require real Qwen3-8B weights.
Run with: pytest tests/e2e/qwen3_8b/test_tflops_threshold.py -v -s
"""

import os
import time

import pytest
from transformers import AutoConfig


try:
    import xorl_client
except ModuleNotFoundError:
    xorl_client = None

from tests.e2e.e2e_utils import skip_if_gpu_count_less_than
from tests.e2e.server_utils import (
    ServerProcess,
    _create_lora_client,
    _get_free_port,
    _start_server_or_fail,
    extract_loss,
    generate_random_sft_data,
    generate_server_config,
)
from xorl.utils.count_flops import XorlFlopsCounter, get_device_flops


pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.server, pytest.mark.benchmark]

# ---------------------------------------------------------------------------
# Model and thresholds
# ---------------------------------------------------------------------------

QWEN3_8B_DIR = "/data/shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
VOCAB_SIZE = 151936

# Minimum TFLOPS per GPU (80% of corrected H100 baseline, flash_attention_3, seq_len=4096)
MIN_TFLOPS = {
    1: 333,
    2: 313,
    4: 306,
}

# Benchmark parameters
SEQ_LEN = 4096
NUM_SAMPLES = 4
NUM_STEPS = 10
NUM_WARMUP = 2
LORA_RANK = 8
PACKING_SEQ_LEN = 16384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_no_qwen3_8b():
    return pytest.mark.skipif(
        not os.path.isdir(QWEN3_8B_DIR),
        reason=f"Qwen3-8B not found at {QWEN3_8B_DIR}",
    )


def _run_tflops_benchmark(num_gpus: int, dp_shard_size: int, tmp_path: str):
    """Run a Qwen3-8B LoRA SFT benchmark and return TFLOPS."""
    if xorl_client is None:
        pytest.skip("xorl_client not installed")

    output_dir = os.path.join(tmp_path, f"bench_{num_gpus}gpu")
    api_port = _get_free_port()

    config_path = generate_server_config(
        model_dir=QWEN3_8B_DIR,
        output_dir=output_dir,
        num_gpus=num_gpus,
        dp_shard_size=dp_shard_size,
        enable_lora=True,
        lora_rank=LORA_RANK,
        sample_packing_sequence_len=PACKING_SEQ_LEN,
    )

    server = ServerProcess(config_path, num_gpus=num_gpus, api_port=api_port, output_dir=output_dir)
    try:
        _start_server_or_fail(server, timeout=300)

        _, training_client = _create_lora_client(server.base_url, QWEN3_8B_DIR, model_id=f"bench-{num_gpus}gpu")

        data = generate_random_sft_data(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
        adam_params = xorl_client.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8)

        # Warmup
        for _ in range(NUM_WARMUP):
            training_client.forward_backward(data, loss_fn="causallm_loss").result()
            training_client.optim_step(adam_params).result()

        # Measured steps
        step_times = []
        losses = []
        for step in range(NUM_STEPS):
            t0 = time.perf_counter()
            fwd_bwd = training_client.forward_backward(data, loss_fn="causallm_loss")
            optim = training_client.optim_step(adam_params)
            result = fwd_bwd.result()
            optim.result()
            t1 = time.perf_counter()
            step_times.append(t1 - t0)
            losses.append(extract_loss(result))

        avg_step = sum(step_times) / len(step_times)
        total_tokens = NUM_SAMPLES * SEQ_LEN * NUM_STEPS
        tokens_per_sec = total_tokens / sum(step_times)

        # Estimate TFLOPS
        model_config = AutoConfig.from_pretrained(QWEN3_8B_DIR)
        flops_counter = XorlFlopsCounter(model_config)
        batch_seqlens = [SEQ_LEN] * NUM_SAMPLES
        flops_achieved, _ = flops_counter.estimate_flops(batch_seqlens, avg_step)
        device_peak_tflops = get_device_flops(unit="T")
        mfu = flops_achieved / device_peak_tflops if device_peak_tflops > 0 else 0.0

        print(f"\n{'─' * 60}")
        print(f"Qwen3-8B LoRA SFT — {num_gpus} GPU(s)")
        print(f"{'─' * 60}")
        print(f"  Loss:       {losses[0]:.4f} → {losses[-1]:.4f}")
        print(f"  Avg step:   {avg_step:.3f}s")
        print(f"  Tokens/sec: {tokens_per_sec:.0f}")
        print(f"  TFLOPS:     {flops_achieved:.2f} / {device_peak_tflops:.0f} peak")
        print(f"  MFU:        {mfu:.2%}")
        print(f"{'─' * 60}")

        return {
            "tflops": flops_achieved,
            "mfu": mfu,
            "tokens_per_sec": tokens_per_sec,
            "avg_step_time": avg_step,
            "losses": losses,
        }
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwen3_8B_TFLOPS:
    """Assert Qwen3-8B LoRA training meets TFLOPS thresholds on H100."""

    @skip_if_gpu_count_less_than(1)
    @_skip_if_no_qwen3_8b()
    def test_1gpu_tflops(self, tmp_path):
        """1-GPU Qwen3-8B LoRA must achieve >= 333 TFLOPS on H100."""
        result = _run_tflops_benchmark(num_gpus=1, dp_shard_size=1, tmp_path=str(tmp_path))
        assert result["tflops"] >= MIN_TFLOPS[1], f"1-GPU TFLOPS {result['tflops']:.1f} below threshold {MIN_TFLOPS[1]}"
        assert result["losses"][-1] < result["losses"][0], "Loss should decrease"

    @skip_if_gpu_count_less_than(2)
    @_skip_if_no_qwen3_8b()
    def test_2gpu_tflops(self, tmp_path):
        """2-GPU Qwen3-8B LoRA FSDP2 must achieve >= 313 TFLOPS on H100."""
        result = _run_tflops_benchmark(num_gpus=2, dp_shard_size=2, tmp_path=str(tmp_path))
        assert result["tflops"] >= MIN_TFLOPS[2], f"2-GPU TFLOPS {result['tflops']:.1f} below threshold {MIN_TFLOPS[2]}"
        assert result["losses"][-1] < result["losses"][0], "Loss should decrease"

    @skip_if_gpu_count_less_than(4)
    @_skip_if_no_qwen3_8b()
    def test_4gpu_tflops(self, tmp_path):
        """4-GPU Qwen3-8B LoRA FSDP2 must achieve >= 306 TFLOPS on H100."""
        result = _run_tflops_benchmark(num_gpus=4, dp_shard_size=4, tmp_path=str(tmp_path))
        assert result["tflops"] >= MIN_TFLOPS[4], f"4-GPU TFLOPS {result['tflops']:.1f} below threshold {MIN_TFLOPS[4]}"
        assert result["losses"][-1] < result["losses"][0], "Loss should decrease"
