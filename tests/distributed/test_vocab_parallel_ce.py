"""Test vocab-parallel cross-entropy with funcol.

Can be run two ways:
    1. pytest tests/distributed/test_vocab_parallel_ce.py -v   (launches torchrun internally)
    2. torchrun --nproc_per_node=2 tests/distributed/test_vocab_parallel_ce.py  (direct)
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


# ============================================================================
# Distributed test functions (run inside torchrun)
# ============================================================================


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def check_correctness(rank, world_size, tp_group):
    """Compare vocab-parallel CE against full-gather reference."""
    torch.manual_seed(42)
    BT = 128
    H = 256
    V = 1024

    assert V % world_size == 0
    local_V = V // world_size

    hidden_states = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (BT,), device="cuda")
    labels[::5] = -100

    full_weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)
    local_weight = full_weight[rank * local_V : (rank + 1) * local_V].contiguous()

    # Reference: full gather, standard CE
    full_logits = (hidden_states @ full_weight.t()).float()
    ref_ce = F.cross_entropy(full_logits, labels, reduction="none", ignore_index=-100)

    for use_compile in [False, True]:
        par_ce = vocab_parallel_cross_entropy(
            hidden_states,
            local_weight,
            labels,
            tp_group,
            ignore_index=-100,
            use_compile=use_compile,
        )
        err = (par_ce - ref_ce).abs().max().item()
        mode = "compiled" if use_compile else "eager"
        if rank == 0:
            print(f"[correctness/{mode}] max abs error: {err:.2e}")
        assert err < 1e-3, f"Error too large ({mode}): {err}"


def check_backward(rank, world_size, tp_group):
    """Test gradients against full-gather reference."""
    torch.manual_seed(42)
    BT = 64
    H = 128
    V = 512
    local_V = V // world_size

    full_weight = torch.randn(V, H, device="cuda", dtype=torch.bfloat16)

    # --- Reference backward (full gather) ---
    h_ref = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w_ref = full_weight.clone().requires_grad_(True)
    labels = torch.randint(0, V, (BT,), device="cuda")
    labels[::3] = -100

    full_logits = (h_ref @ w_ref.t()).float()
    ref_ce = F.cross_entropy(full_logits, labels, reduction="none", ignore_index=-100)
    valid = (labels != -100).sum().clamp(min=1).float()
    (ref_ce.sum() / valid).backward()

    for use_compile in [False, True]:
        # --- Vocab-parallel backward ---
        h_par = h_ref.detach().clone().requires_grad_(True)
        w_par = full_weight[rank * local_V : (rank + 1) * local_V].contiguous().requires_grad_(True)

        par_ce = vocab_parallel_cross_entropy(
            h_par,
            w_par,
            labels,
            tp_group,
            ignore_index=-100,
            use_compile=use_compile,
        )
        (par_ce.sum() / valid).backward()

        grad_h_err = (h_par.grad - h_ref.grad).abs().max().item()
        ref_w_grad_shard = w_ref.grad[rank * local_V : (rank + 1) * local_V]
        grad_w_err = (w_par.grad - ref_w_grad_shard).abs().max().item()

        mode = "compiled" if use_compile else "eager"
        if rank == 0:
            print(f"[backward/{mode}] grad_hidden err: {grad_h_err:.2e}")
            print(f"[backward/{mode}] grad_weight err: {grad_w_err:.2e}")

        assert grad_h_err < 1e-2, f"grad_h error too large ({mode}): {grad_h_err}"
        assert grad_w_err < 1e-2, f"grad_w error too large ({mode}): {grad_w_err}"


def main():
    rank, world_size = setup()
    tp_group = dist.group.WORLD

    if rank == 0:
        print(f"=== test_vocab_parallel_ce (tp={world_size}) ===\n")

    check_correctness(rank, world_size, tp_group)
    dist.barrier()
    if rank == 0:
        print()

    check_backward(rank, world_size, tp_group)
    dist.barrier()
    if rank == 0:
        print()

    if rank == 0:
        print("\nAll tests passed!")

    dist.destroy_process_group()


# ============================================================================
# Pytest wrappers (launch torchrun internally)
# ============================================================================

if __name__ != "__main__":
    # Only define pytest tests when imported by pytest (not when run via torchrun)
    import pytest

    from tests.distributed.distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

    SCRIPT_PATH = os.path.abspath(__file__)

    @pytest.mark.gpu
    @pytest.mark.distributed
    @skip_if_gpu_count_less_than(2)
    def test_vocab_parallel_ce_2gpu():
        """Vocab-parallel cross-entropy correctness + backward with 2 GPUs."""
        result = run_distributed_script(SCRIPT_PATH, num_gpus=2, timeout=180)
        result.assert_success()


if __name__ == "__main__":
    main()
