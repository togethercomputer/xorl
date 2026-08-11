"""Distributed exact-convolution equivalence across a Ulysses shard boundary."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import init_parallel_state
from xorl.ops.linear_attention.modules import ShortConvolution, causal_conv1d_qkv_contract
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed]


def _setup_dist() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        ulysses_size=world_size,
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    return torch.device("cuda", local_rank)


def _make_convs(device: torch.device) -> list[ShortConvolution]:
    torch.manual_seed(7)
    return [
        ShortConvolution(dim, kernel_size=4, bias=False, activation="silu").to(
            device=device,
            dtype=torch.bfloat16,
        )
        for dim in (32, 32, 64)
    ]


def _gather_sequence(local: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local.contiguous())
    return torch.cat(gathered, dim=1)


def _run_equivalence() -> None:
    device = _setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_tokens = 32 * world_size
    local_tokens = total_tokens // world_size
    # The second document starts three tokens before rank 1, so every exchanged
    # prefix element is live and the packed boundary after it must still reset.
    cu_seqlens = torch.tensor([0, local_tokens - 3, total_tokens], device=device, dtype=torch.int32)

    reference_convs = _make_convs(device)
    cp_convs = _make_convs(device)
    for cp_conv, reference_conv in zip(cp_convs, reference_convs, strict=True):
        cp_conv.load_state_dict(reference_conv.state_dict())

    torch.manual_seed(11)
    full_inputs = [
        torch.randn(1, total_tokens, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        for dim in (32, 32, 64)
    ]
    local_inputs = [
        value[:, rank * local_tokens : (rank + 1) * local_tokens].detach().clone().requires_grad_(True)
        for value in full_inputs
    ]
    cp_context = build_linear_attention_cp_context(cu_seqlens, conv1d_kernel_size=4)
    assert cp_context is not None

    reference = causal_conv1d_qkv_contract(
        *full_inputs,
        *reference_convs,
        cu_seqlens=cu_seqlens,
    )
    local = causal_conv1d_qkv_contract(
        *local_inputs,
        *cp_convs,
        cu_seqlens=cp_context.cu_seqlens,
        cp_context=cp_context,
    )
    for local_value, reference_value in zip(local, reference, strict=True):
        gathered = _gather_sequence(local_value.detach())
        if rank == 0:
            assert torch.equal(gathered, reference_value.detach())

    torch.manual_seed(13)
    grad_outputs = [torch.randn_like(value) for value in reference]
    torch.autograd.backward(reference, grad_outputs)
    local_grad_outputs = [
        value[:, rank * local_tokens : (rank + 1) * local_tokens].contiguous() for value in grad_outputs
    ]
    torch.autograd.backward(local, local_grad_outputs)

    for local_input, reference_input in zip(local_inputs, full_inputs, strict=True):
        assert local_input.grad is not None
        assert reference_input.grad is not None
        gathered_grad = _gather_sequence(local_input.grad)
        if rank == 0:
            torch.testing.assert_close(gathered_grad, reference_input.grad, rtol=2e-2, atol=2e-2)

    for cp_conv, reference_conv in zip(cp_convs, reference_convs, strict=True):
        assert cp_conv.weight.grad is not None
        assert reference_conv.weight.grad is not None
        dist.all_reduce(cp_conv.weight.grad)
        if rank == 0:
            torch.testing.assert_close(cp_conv.weight.grad, reference_conv.weight.grad, rtol=2e-2, atol=2e-2)

    if rank == 0:
        print("exact GDN convolution CP equivalence passed")


def _main() -> None:
    try:
        _run_equivalence()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_exact_gdn_conv_contract_matches_unsharded_reference():
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success("Exact GDN convolution CP should match the unsharded serving-bit reference")


if __name__ == "__main__":
    _main()
