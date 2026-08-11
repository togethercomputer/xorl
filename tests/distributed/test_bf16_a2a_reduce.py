"""Default-runtime contract for ``BF16StochasticAllToAllReduceScatter``.

One report covers the stochastic FP32-to-BF16 primitive and the real two-rank
Gloo all-to-all/FP32 accumulation transaction. This avoids treating a four-GPU
admission gate and an isolated rounding unit as independent confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.fsdp2 import BF16StochasticAllToAllReduceScatter
from xorl.optim.stochastic_round import stochastic_round_to_bf16


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]


def _setup_dist() -> torch.device:
    dist.init_process_group(backend="gloo")
    return torch.device("cpu")


def _assert_stochastic_round_distribution_and_admission() -> None:
    values = torch.randn(7, 13, dtype=torch.float32)
    generator_one = torch.Generator().manual_seed(42)
    generator_two = torch.Generator().manual_seed(42)
    rounded = stochastic_round_to_bf16(values, generator=generator_one)

    assert rounded.dtype is torch.bfloat16
    assert rounded.shape == values.shape
    assert torch.equal(rounded, stochastic_round_to_bf16(values, generator=generator_two))
    with pytest.raises(ValueError, match="requires fp32 input"):
        stochastic_round_to_bf16(values.to(torch.bfloat16))

    sample_count = 1 << 16
    lower_bits = 0x3F800000
    fractional_bits = 0x4000
    samples = torch.full((sample_count,), lower_bits + fractional_bits, dtype=torch.int32).view(torch.float32)
    rounded_samples = stochastic_round_to_bf16(samples, generator=torch.Generator().manual_seed(0)).float()
    lower = torch.tensor(lower_bits, dtype=torch.int32).view(torch.float32)
    upper = torch.tensor(lower_bits + 0x10000, dtype=torch.int32).view(torch.float32)

    assert ((rounded_samples == lower) | (rounded_samples == upper)).all()
    assert (rounded_samples == upper).float().mean().item() == pytest.approx(0.25, abs=0.01)
    assert rounded_samples.mean().item() == pytest.approx(samples[0].item(), abs=1e-4)


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

    # Reference: native FP32 reduce-scatter.
    ref_out = torch.empty(chunk_numel, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(ref_out, local_grad.clone(), op=dist.ReduceOp.SUM)

    # Test: BF16 stochastic-rounded a2a + FP32 local sum.
    comm = BF16StochasticAllToAllReduceScatter()
    test_out = comm.allocate((chunk_numel,), dtype=torch.float32, device=device)
    comm(test_out, local_grad.clone(), group=dist.group.WORLD, op=dist.ReduceOp.SUM)

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

    if rank == 0:
        print(f"[rank 0] BF16 a2a max err = {max_err:.4e}, bound = {max_bound:.4e}")

    dist.barrier()
    dist.destroy_process_group()


def _main() -> None:
    _run()


if __name__ != "__main__":

    def test_bf16_stochastic_a2a_reduce_scatter_default_runtime_contract():
        _assert_stochastic_round_distribution_and_admission()
        result = run_distributed_script(
            __file__,
            num_gpus=2,
            timeout=180,
            extra_env={"CUDA_VISIBLE_DEVICES": ""},
        )
        result.assert_success("BF16 a2a reduce-scatter should match FP32 on two CPU/Gloo ranks")


if __name__ == "__main__":
    _main()
