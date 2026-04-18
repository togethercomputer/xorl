"""Test vocab-parallel cross-entropy with funcol.

Can be run two ways:
    1. pytest tests/distributed/test_vocab_parallel_ce.py -v   (launches torchrun internally)
    2. torchrun --nproc_per_node=2 tests/distributed/test_vocab_parallel_ce.py  (direct)
"""

import os
import time

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


def _bench_one(hidden_states, local_weight, labels, tp_group, use_compile, n_warmup=5, n_iter=20, fwd_only=False):
    """Run warmup + timed iterations, return (ms/iter, peak_memory_MB)."""
    for _ in range(n_warmup):
        ce = vocab_parallel_cross_entropy(hidden_states, local_weight, labels, tp_group, use_compile=use_compile)
        if not fwd_only:
            ce.sum().backward()
            hidden_states.grad = None
            local_weight.grad = None
    torch.cuda.synchronize()

    # Reset peak memory stats before timed run
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    start = time.perf_counter()
    for _ in range(n_iter):
        ce = vocab_parallel_cross_entropy(hidden_states, local_weight, labels, tp_group, use_compile=use_compile)
        if not fwd_only:
            ce.sum().backward()
            hidden_states.grad = None
            local_weight.grad = None
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / n_iter * 1000

    peak_mem = torch.cuda.max_memory_allocated()
    # Peak activation memory = peak total - memory before (which includes weights + NCCL buffers)
    peak_activation_mb = (peak_mem - mem_before) / 1024 / 1024

    return ms, peak_activation_mb, peak_mem / 1024 / 1024


def bench_perf(rank, world_size, tp_group):
    """Benchmark eager vs compiled vocab-parallel CE with memory tracking."""
    torch.manual_seed(42)
    BT = 4096
    H = 4096
    V = 152064  # Qwen3 vocab
    local_V = V // world_size

    hidden_states = torch.randn(BT, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    local_weight = torch.randn(local_V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (BT,), device="cuda")

    if rank == 0:
        print(f"[perf] BT={BT}, H={H}, V={V}, tp={world_size}")
        print(f"[perf] local_V={local_V}, num_chunks=8")
        weight_mb = local_weight.nelement() * local_weight.element_size() / 1024 / 1024
        hidden_mb = hidden_states.nelement() * hidden_states.element_size() / 1024 / 1024
        print(f"[perf] weight: {weight_mb:.1f} MB, hidden: {hidden_mb:.1f} MB")
        print()

    # --- Forward-only benchmark ---
    if rank == 0:
        print("--- Forward only ---")
        print(f"{'Mode':<12} {'Time (ms)':<12} {'Peak Act (MB)':<16} {'Peak Total (MB)'}")
    for use_compile in [False, True]:
        mode = "compiled" if use_compile else "eager"
        ms, peak_act_mb, peak_total_mb = _bench_one(
            hidden_states,
            local_weight,
            labels,
            tp_group,
            use_compile=use_compile,
            fwd_only=True,
        )
        if rank == 0:
            print(f"{mode:<12} {ms:<12.2f} {peak_act_mb:<16.1f} {peak_total_mb:.1f}")

    if rank == 0:
        print()

    # --- Forward + Backward benchmark ---
    if rank == 0:
        print("--- Forward + Backward ---")
        print(f"{'Mode':<12} {'Time (ms)':<12} {'Peak Act (MB)':<16} {'Peak Total (MB)'}")
    for use_compile in [False, True]:
        mode = "compiled" if use_compile else "eager"
        ms, peak_act_mb, peak_total_mb = _bench_one(
            hidden_states,
            local_weight,
            labels,
            tp_group,
            use_compile=use_compile,
            fwd_only=False,
        )
        if rank == 0:
            print(f"{mode:<12} {ms:<12.2f} {peak_act_mb:<16.1f} {peak_total_mb:.1f}")

    if rank == 0:
        print()


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

    bench_perf(rank, world_size, tp_group)
    dist.barrier()

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

    @pytest.mark.gpu
    @pytest.mark.distributed
    @skip_if_gpu_count_less_than(4)
    def test_vocab_parallel_ce_4gpu():
        """Vocab-parallel cross-entropy correctness + backward with 4 GPUs."""
        result = run_distributed_script(SCRIPT_PATH, num_gpus=4, timeout=180)
        result.assert_success()

    @pytest.mark.gpu
    @pytest.mark.distributed
    @skip_if_gpu_count_less_than(8)
    def test_vocab_parallel_ce_8gpu():
        """Vocab-parallel cross-entropy correctness + backward with 8 GPUs."""
        result = run_distributed_script(SCRIPT_PATH, num_gpus=8, timeout=180)
        result.assert_success()


if __name__ == "__main__":
    main()
