"""Distributed numerical equivalence checks for native linear-attention CP."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import init_parallel_state
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils.device import get_nccl_backend

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

pytestmark = [pytest.mark.distributed]


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _setup_dist(*, ulysses_size: int) -> torch.device:
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
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    return torch.device("cuda", local_rank)


def _build_test_layer(device: torch.device) -> GatedDeltaNet:
    layer = GatedDeltaNet(
        hidden_size=128,
        expand_v=1.0,
        head_dim=16,
        num_heads=8,
        num_v_heads=8,
        mode="chunk",
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        norm_eps=1e-6,
    ).to(device=device, dtype=torch.bfloat16)
    layer.eval()
    return layer


def _gather_sequence_shards(local_tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(local_tensor) for _ in range(_world_size())]
    dist.all_gather(gathered, local_tensor.contiguous())
    return torch.cat(gathered, dim=1)


def _run_cp_equivalence() -> None:
    world_size = _world_size()
    device = _setup_dist(ulysses_size=world_size)
    rank = dist.get_rank()

    total_tokens = 128
    hidden_size = 128
    assert total_tokens % world_size == 0
    local_seq_len = total_tokens // world_size

    # Make one sequence cross the rank boundary so conv-prefix/state exchange is exercised.
    first_seq_len = local_seq_len - 10
    cu_seqlens = torch.tensor([0, first_seq_len, total_tokens], dtype=torch.int32, device=device)

    torch.manual_seed(0)
    reference_layer = _build_test_layer(device)
    cp_layer = _build_test_layer(device)
    cp_layer.load_state_dict(reference_layer.state_dict())

    torch.manual_seed(1)
    full_hidden_states = torch.randn(1, total_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    reference_input = full_hidden_states.clone().detach().requires_grad_(True)
    cp_input = (
        full_hidden_states[:, rank * local_seq_len : (rank + 1) * local_seq_len]
        .contiguous()
        .clone()
        .detach()
        .requires_grad_(True)
    )

    cp_context = build_linear_attention_cp_context(
        cu_seqlens,
        conv1d_kernel_size=cp_layer.conv_size,
    )
    assert cp_context is not None
    assert cp_context.cu_seqlens is not None

    reference_output, _, _ = reference_layer(
        hidden_states=reference_input,
        use_cache=False,
        cu_seqlens=cu_seqlens,
    )
    cp_output, _, _ = cp_layer(
        hidden_states=cp_input,
        use_cache=False,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    )

    gathered_cp_output = _gather_sequence_shards(cp_output.detach())

    full_output_numel = reference_output.numel()
    reference_loss = reference_output.float().square().sum() / full_output_numel
    cp_loss = cp_output.float().square().sum() / full_output_numel
    reference_loss.backward()
    cp_loss.backward()

    assert cp_input.grad is not None
    gathered_cp_input_grad = _gather_sequence_shards(cp_input.grad.detach())

    if rank == 0:
        torch.testing.assert_close(
            gathered_cp_output.float(),
            reference_output.detach().float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            gathered_cp_input_grad.float(),
            reference_input.grad.detach().float(),
            atol=2e-2,
            rtol=2e-2,
        )
        print("linear-attention CP equivalence passed")


def _main() -> None:
    try:
        _run_cp_equivalence()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ != "__main__":
    @skip_if_gpu_count_less_than(2)
    def test_linear_attention_cp_matches_single_gpu_reference():
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=240,
        )
        result.assert_success("Native linear-attention CP should match single-GPU reference")


if __name__ == "__main__":
    _main()
