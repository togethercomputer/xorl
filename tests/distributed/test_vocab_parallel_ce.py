"""Test vocab-parallel cross-entropy with funcol.

Can be run two ways:
    1. pytest tests/distributed/test_vocab_parallel_ce.py -v   (launches torchrun internally)
    2. torchrun --nproc_per_node=2 tests/distributed/test_vocab_parallel_ce.py  (direct)
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.ops.exact_sampling_transforms import TOP_K_ALL, exact_selected_logprob
from xorl.ops.loss.per_token_ce import compute_per_token_ce
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


def check_sampling_transform_replay(rank, world_size, tp_group):
    """Body-TP top-k/top-p/min-p + per-row temperature replay via the shared core."""
    torch.manual_seed(7)
    BT = 48
    H = 128
    V = 512
    local_V = V // world_size

    # FP32 operands keep the support decision identical between the dense
    # reference GEMM and the sharded/row-chunked GEMMs; bf16 rounding could
    # legitimately flip a token exactly on the top-p boundary.
    hidden = torch.randn(BT, H, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, V, (BT,), device="cuda")
    labels[::7] = -100
    full_weight = torch.randn(V, H, device="cuda", dtype=torch.float32)
    temperature = torch.linspace(0.7, 1.3, BT, device="cuda", dtype=torch.float32)
    top_ks = torch.randint(2, V, (BT,), device="cuda", dtype=torch.int64)
    top_ks[0] = TOP_K_ALL
    top_ps = torch.linspace(0.6, 1.0, BT, device="cuda", dtype=torch.float32)
    min_ps = torch.linspace(0.0, 0.15, BT, device="cuda", dtype=torch.float32)

    # Reference: dense single-rank program on the full weight.
    h_ref = hidden.clone().requires_grad_(True)
    w_ref = full_weight.clone().requires_grad_(True)
    valid = labels != -100
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    ref_logits = (h_ref @ w_ref.t()).float() / temperature.unsqueeze(1)
    ref_logprob, _, _, _ = exact_selected_logprob(ref_logits, safe, top_ks, top_ps, min_ps)
    ref_ce = torch.where(valid, -ref_logprob, torch.zeros_like(ref_logprob))
    (ref_ce[valid].mean()).backward()

    h_par = hidden.clone().requires_grad_(True)
    w_par = full_weight[rank * local_V : (rank + 1) * local_V].contiguous().requires_grad_(True)
    par_ce = compute_per_token_ce(
        h_par,
        w_par,
        labels,
        ignore_index=-100,
        ce_mode="compiled",
        tp_group=tp_group,
        logprob_temperature=temperature,
        logprob_top_k=top_ks,
        logprob_top_p=top_ps,
        logprob_min_p=min_ps,
    )
    (par_ce[valid].mean()).backward()

    # A replayed token outside its row's support scores +inf CE in both
    # implementations; compare infinities structurally and values elsewhere.
    assert torch.equal(torch.isinf(par_ce), torch.isinf(ref_ce)), "support decisions diverged"
    finite = ~torch.isinf(ref_ce)
    ce_err = (par_ce - ref_ce)[finite].abs().max().item()
    grad_h_err = (h_par.grad.float() - h_ref.grad.float()).abs().max().item()
    ref_w_shard = w_ref.grad[rank * local_V : (rank + 1) * local_V]
    grad_w_err = (w_par.grad.float() - ref_w_shard.float()).abs().max().item()
    if rank == 0:
        print(f"[transform-replay] ce err: {ce_err:.2e}, grad_h err: {grad_h_err:.2e}, grad_w err: {grad_w_err:.2e}")
    assert ce_err < 1e-3, f"transform-replay CE error too large: {ce_err}"
    assert grad_h_err < 1e-2, f"transform-replay grad_h error too large: {grad_h_err}"
    assert grad_w_err < 1e-2, f"transform-replay grad_w error too large: {grad_w_err}"


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

    check_sampling_transform_replay(rank, world_size, tp_group)
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
