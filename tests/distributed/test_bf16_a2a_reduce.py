"""Distributed correctness tests for ``BF16StochasticAllToAllReduceScatter``.

Verifies the custom reduce-scatter (stochastic-round FP32→BF16, all-to-all,
local FP32 sum) produces results numerically close to native FP32
reduce-scatter, with bias-in-expectation near zero.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp

from xorl.distributed.fsdp2 import BF16StochasticAllToAllReduceScatter
from xorl.distributed.fsdp2.bf16_a2a_reduce import _canonical_reduce_op
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than


pytestmark = [pytest.mark.distributed]


@pytest.mark.cpu
def test_canonical_reduce_op_accepts_fsdp_wrapped_ops():
    assert _canonical_reduce_op(ReduceOp(ReduceOp.SUM)) == dist.ReduceOp.SUM
    assert _canonical_reduce_op(ReduceOp(ReduceOp.AVG)) == dist.ReduceOp.AVG
    assert _canonical_reduce_op(dist.ReduceOp.SUM) == dist.ReduceOp.SUM
    assert _canonical_reduce_op(dist._make_nccl_premul_sum(1.0)) == dist.ReduceOp.SUM


def _world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _setup_dist() -> torch.device:
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    return torch.device("cuda", local_rank)


def _run() -> None:
    device = _setup_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Each rank generates a chunk of FP32 grad. We want to reduce-scatter:
    # the input on every rank is the FULL flat unsharded grad, viewed as
    # ``world * chunk`` elements; reduce-scatter sums across ranks and
    # gives each rank its ``chunk_numel`` slice of the global sum.
    chunk_numel = 4096
    total_numel = chunk_numel * world

    torch.manual_seed(0xCAFE + rank)
    # Per-rank gradient (independent across ranks).
    local_grad = torch.randn(total_numel, dtype=torch.float32, device=device)

    # ---- Reference: native FP32 reduce-scatter ----
    ref_out = torch.empty(chunk_numel, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(ref_out, local_grad.clone(), op=dist.ReduceOp.SUM)

    # ---- Test: BF16 stochastic-rounded a2a + FP32 local sum ----
    comm = BF16StochasticAllToAllReduceScatter()
    test_out = comm.allocate((chunk_numel,), dtype=torch.float32, device=device)
    comm(test_out, local_grad.clone(), group=dist.group.WORLD, op=dist.ReduceOp.SUM)

    # ---- Bound the per-element error ----
    # Each rank's contribution is stochastically rounded FP32→BF16 with at most
    # one ulp of noise. After summing ``world`` such contributions, the error
    # is bounded by sum of |x_r| * 2^-7 in the worst case. Compute this bound.
    abs_input = local_grad.abs()
    err_bound_local = abs_input * (2**-7)  # per-rank max error envelope
    # Get this rank's slice of the global error bound — match what reduce-scatter does.
    err_bound_full_sum = err_bound_local.clone()
    dist.all_reduce(err_bound_full_sum, op=dist.ReduceOp.SUM)
    # Slice this rank's chunk of the bound.
    bound_chunks = err_bound_full_sum.view(world, chunk_numel)
    bound_for_my_chunk = bound_chunks[rank]

    abs_err = (test_out - ref_out).abs()
    # Max element-wise should be <= our bound, with some headroom for FP32
    # rounding in the local sum. Use 4x headroom.
    max_err = abs_err.max().item()
    max_bound = bound_for_my_chunk.max().item() * 4 + 1e-6
    assert max_err < max_bound, f"[rank {rank}] BF16 a2a max err {max_err:.4e} exceeds bound {max_bound:.4e}"

    # Bias-in-expectation: average over many trials should approach the FP32 reference.
    # Use the same input tensor; only the stochastic rounding noise differs.
    n_trials = 200
    accum = torch.zeros_like(test_out)
    for _ in range(n_trials):
        out = comm.allocate((chunk_numel,), dtype=torch.float32, device=device)
        comm(out, local_grad.clone(), group=dist.group.WORLD, op=dist.ReduceOp.SUM)
        accum += out
    mean = accum / n_trials
    mean_err = (mean - ref_out).abs().max().item()
    # Standard error of the mean ~ bound / sqrt(n_trials). For n=200 and BF16
    # bound ~|x|/128, SEM ~ |x| * 1e-3. Allow generous 5x headroom.
    sem_bound = max_bound / (n_trials**0.5) * 5
    assert mean_err < sem_bound, (
        f"[rank {rank}] BF16 a2a is biased: mean err over {n_trials} trials = "
        f"{mean_err:.4e}, expected < {sem_bound:.4e}"
    )

    if rank == 0:
        print(f"[rank 0] BF16 a2a max err = {max_err:.4e}, bound = {max_bound:.4e}")
        print(f"[rank 0] BF16 a2a unbiased mean err over {n_trials} trials = {mean_err:.4e}")

    dist.barrier()
    dist.destroy_process_group()


def _main() -> None:
    _run()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(4)
    def test_bf16_a2a_reduce_scatter_matches_fp32_within_bound():
        result = run_distributed_script(__file__, num_gpus=4, timeout=180)
        result.assert_success("BF16 a2a reduce-scatter should match FP32 within BF16 ulp bound")


if __name__ == "__main__":
    _main()
