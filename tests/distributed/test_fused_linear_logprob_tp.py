"""Tensor-parallel tests for the fused selected-token log-probability kernel.

Validates that the chunked fused path reproduces the full-vocabulary selected-token
log-probability while reducing only three ``[N]`` scalar vectors (row max, shifted
sum-exp, selected logit) across TP ranks — and that the backward pass all-reduces
``grad_h`` only when it is requested.

Can be run two ways:
    1. pytest tests/distributed/test_fused_linear_logprob_tp.py -v   (launches torchrun)
    2. torchrun --nproc_per_node=2 tests/distributed/test_fused_linear_logprob_tp.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.ops.loss.fused_linear_logprob import fused_selected_logprob_ce
from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def _shard_bounds(V, world_size, rank):
    """Even split with the remainder placed on the last ranks (uneven shards)."""
    base = V // world_size
    rem = V % world_size
    sizes = [base + (1 if r >= world_size - rem else 0) for r in range(world_size)]
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def check_correctness(rank, world_size, tp_group):
    """Fused TP per-token CE == full-gather reference, for even & uneven shards."""
    torch.manual_seed(42)
    for V, has_bias, temperature in [(1024, False, 1.0), (1023, True, 0.8)]:
        N, H = 128, 256
        hidden = torch.randn(N, H, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, V, (N,), device="cuda")
        labels[::5] = -100

        full_weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) / (H**0.5)
        full_bias = torch.randn(V, device="cuda", dtype=torch.bfloat16) if has_bias else None
        lo, hi = _shard_bounds(V, world_size, rank)
        local_weight = full_weight[lo:hi].contiguous()
        local_bias = full_bias[lo:hi].contiguous() if has_bias else None

        # Reference: full gather, standard CE
        full_logits = (hidden @ full_weight.t()).float()
        if has_bias:
            full_logits = full_logits + full_bias.float()[None, :]
        full_logits = full_logits / temperature
        ref_ce = F.cross_entropy(full_logits, labels, reduction="none", ignore_index=-100)

        fused_ce = fused_selected_logprob_ce(
            hidden, local_weight, labels, tp_group, bias=local_bias, ignore_index=-100, temperature=temperature
        )
        err = (fused_ce - ref_ce).abs().max().item()
        if rank == 0:
            print(f"[correctness V={V} bias={has_bias} T={temperature}] max abs error: {err:.2e}")
        assert err < 5e-2, f"fused TP CE error too large: {err}"

        # also matches the existing chunked vocab-parallel path (T=1, no bias)
        if not has_bias and temperature == 1.0:
            par_ce = vocab_parallel_cross_entropy(hidden, local_weight, labels, tp_group, ignore_index=-100)
            err2 = (fused_ce - par_ce).abs().max().item()
            if rank == 0:
                print(f"[correctness V={V}] fused vs vocab_parallel: {err2:.2e}")
            assert err2 < 5e-2, f"fused vs vocab_parallel mismatch: {err2}"


def check_backward(rank, world_size, tp_group):
    """grad_h (all-reduced) and grad_W shard match the full-gather reference."""
    torch.manual_seed(42)
    N, H, V = 64, 128, 512
    lo, hi = _shard_bounds(V, world_size, rank)

    full_weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) / (H**0.5)
    labels = torch.randint(0, V, (N,), device="cuda")
    labels[::3] = -100
    valid = (labels != -100).sum().clamp(min=1).float()

    h_ref = torch.randn(N, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w_ref = full_weight.clone().requires_grad_(True)
    ref_ce = F.cross_entropy((h_ref @ w_ref.t()).float(), labels, reduction="none", ignore_index=-100)
    (ref_ce.sum() / valid).backward()

    h_par = h_ref.detach().clone().requires_grad_(True)
    w_par = full_weight[lo:hi].contiguous().requires_grad_(True)
    fused_ce = fused_selected_logprob_ce(h_par, w_par, labels, tp_group, ignore_index=-100)
    (fused_ce.sum() / valid).backward()

    grad_h_err = (h_par.grad - h_ref.grad).abs().max().item()
    grad_w_err = (w_par.grad - w_ref.grad[lo:hi]).abs().max().item()
    if rank == 0:
        print(f"[backward] grad_h err: {grad_h_err:.2e}  grad_W shard err: {grad_w_err:.2e}")
    assert grad_h_err < 5e-2, f"grad_h error too large: {grad_h_err}"
    assert grad_w_err < 5e-2, f"grad_W error too large: {grad_w_err}"


def check_needs_input_grad(rank, world_size, tp_group):
    """LoRA RL in TP: frozen weight -> grad_W None, grad_h still all-reduced & correct."""
    torch.manual_seed(7)
    N, H, V = 64, 128, 512
    lo, hi = _shard_bounds(V, world_size, rank)
    full_weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16) / (H**0.5)
    labels = torch.randint(0, V, (N,), device="cuda")

    h_ref = torch.randn(N, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w_ref = full_weight.clone().requires_grad_(True)
    F.cross_entropy((h_ref @ w_ref.t()).float(), labels, reduction="none").sum().backward()

    h_par = h_ref.detach().clone().requires_grad_(True)
    w_par = full_weight[lo:hi].contiguous().requires_grad_(False)  # frozen head
    fused_ce = fused_selected_logprob_ce(h_par, w_par, labels, tp_group, ignore_index=-100)
    fused_ce.sum().backward()

    assert w_par.grad is None, "frozen weight must get no gradient in TP"
    grad_h_err = (h_par.grad - h_ref.grad).abs().max().item()
    if rank == 0:
        print(f"[needs_input_grad] grad_W None=True, grad_h err (all-reduced): {grad_h_err:.2e}")
    assert grad_h_err < 5e-2, f"grad_h must match with frozen head: {grad_h_err}"


def main():
    rank, world_size = setup()
    tp_group = dist.group.WORLD
    if rank == 0:
        print(f"=== test_fused_linear_logprob_tp (tp={world_size}) ===")

    check_correctness(rank, world_size, tp_group)
    dist.barrier()
    check_backward(rank, world_size, tp_group)
    dist.barrier()
    check_needs_input_grad(rank, world_size, tp_group)
    dist.barrier()

    if rank == 0:
        print("All TP tests passed!")
    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest

    from tests.distributed.distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

    SCRIPT_PATH = os.path.abspath(__file__)

    @pytest.mark.gpu
    @pytest.mark.distributed
    @skip_if_gpu_count_less_than(2)
    def test_fused_linear_logprob_tp_2gpu():
        """Fused selected-logprob TP correctness + backward with 2 GPUs."""
        result = run_distributed_script(SCRIPT_PATH, num_gpus=2, timeout=240)
        result.assert_success()


if __name__ == "__main__":
    main()
