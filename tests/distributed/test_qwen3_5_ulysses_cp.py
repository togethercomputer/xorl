"""Distributed smoke tests for Qwen3.5 Ulysses CP integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import init_parallel_state
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils.device import get_nccl_backend

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

pytestmark = [pytest.mark.distributed]


def _build_tiny_qwen35_config(*, layer_types: list[str]) -> Qwen3_5Config:
    config = Qwen3_5Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=8,
        num_key_value_heads=8,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=layer_types,
        max_position_embeddings=128,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return config


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _setup_dist(*, ulysses_size: int, ringattn_size: int) -> torch.device:
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        ulysses_size=ulysses_size,
        ringattn_size=ringattn_size,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    return torch.device("cuda", local_rank)


def _run_positive_smoke() -> None:
    world_size = _world_size()
    device = _setup_dist(ulysses_size=world_size, ringattn_size=1)
    rank = dist.get_rank()

    config = _build_tiny_qwen35_config(layer_types=["linear_attention", "full_attention"])
    model = Qwen3_5Model(config).to(device)
    model.train()

    total_tokens = 16 * world_size
    local_seq_len = total_tokens // world_size
    first_seq_len = local_seq_len - 6
    full_input_ids = (torch.arange(total_tokens, device=device).unsqueeze(0) % config.vocab_size).long()
    local_input_ids = full_input_ids[:, rank * local_seq_len : (rank + 1) * local_seq_len].contiguous()
    position_ids = torch.arange(total_tokens, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, first_seq_len, total_tokens], dtype=torch.int32, device=device)
    cp_context = build_linear_attention_cp_context(cu_seqlens, conv1d_kernel_size=config.linear_conv_kernel_dim)
    assert cp_context is not None

    outputs = model(
        input_ids=local_input_ids,
        position_ids=position_ids,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=total_tokens,
        max_length_k=total_tokens,
    )
    loss = outputs.last_hidden_state.float().sum()
    loss.backward()

    assert outputs.last_hidden_state.shape == (1, local_seq_len, config.hidden_size)
    assert torch.isfinite(outputs.last_hidden_state).all()
    if rank == 0:
        print("ulysses positive smoke passed")


def _run_negative_smoke() -> None:
    world_size = _world_size()
    device = _setup_dist(ulysses_size=1, ringattn_size=world_size)
    rank = dist.get_rank()

    config = _build_tiny_qwen35_config(layer_types=["linear_attention", "full_attention"])
    model = Qwen3_5Model(config).to(device)
    model.eval()

    full_seq_len = 16 * world_size
    local_seq_len = full_seq_len // world_size
    full_input_ids = (torch.arange(full_seq_len, device=device).unsqueeze(0) % config.vocab_size).long()
    local_input_ids = full_input_ids[:, rank * local_seq_len : (rank + 1) * local_seq_len].contiguous()
    position_ids = torch.arange(full_seq_len, device=device).unsqueeze(0)
    assert build_linear_attention_cp_context() is None

    try:
        model(input_ids=local_input_ids, position_ids=position_ids)
    except ValueError as exc:
        assert "temporarily disabled" in str(exc)
        if rank == 0:
            print("ring+FLA negative smoke passed")
        return

    raise AssertionError("Expected ring+FLA configuration to raise ValueError")


def _main() -> None:
    mode = os.environ.get("QWEN35_CP_MODE", "positive")
    try:
        if mode == "positive":
            _run_positive_smoke()
        elif mode == "negative":
            _run_negative_smoke()
        else:
            raise ValueError(f"Unsupported QWEN35_CP_MODE: {mode}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ != "__main__":
    @skip_if_gpu_count_less_than(2)
    def test_qwen35_ulysses_positive_smoke():
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"QWEN35_CP_MODE": "positive"},
        )
        result.assert_success("Qwen3.5 positive Ulysses smoke should pass")


    @skip_if_gpu_count_less_than(2)
    def test_qwen35_ring_fla_negative_smoke():
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"QWEN35_CP_MODE": "negative"},
        )
        result.assert_success("Qwen3.5 ring+FLA negative smoke should fail fast")


if __name__ == "__main__":
    _main()
