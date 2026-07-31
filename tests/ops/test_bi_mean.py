"""Regression gate: BI-mode aten::mean.

aten::mean full-reduce dispatches to the mean.dim override with dim=[]; the
empty n_elems product made it return the SUM. Locks: full-reduce == mean,
dim reductions bit-identical to the certified mean_dim kernel path.
"""

import pytest
import torch

from xorl.ops.batch_invariant_ops import mean_dim, set_batch_invariant_mode


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(512,), (4, 8, 16), (33, 127)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_full_reduce_mean_is_mean_not_sum(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = x.double().mean()
    with set_batch_invariant_mode(True):
        got = x.mean()
    assert got.shape == torch.Size([])
    tol = 1e-2 if dtype is torch.bfloat16 else 1e-5
    assert abs(got.double().item() - ref.item()) < tol
    # the bug returned the full sum
    assert abs(got.double().item() - x.double().sum().item()) > tol or x.numel() == 1


@requires_cuda
@pytest.mark.gpu
def test_full_reduce_dtype_kwarg():
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True):
        got = x.mean(dtype=torch.float32)
    assert got.dtype == torch.float32
    assert abs(got.item() - x.double().mean().item()) < 1e-2


@requires_cuda
@pytest.mark.gpu
def test_dim_reductions_unchanged_bitwise():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True):
        got_1d = x.mean(-1)
        got_keep = x.mean(-1, keepdim=True)
        got_2d = x.mean(dim=(0, 1))
    assert torch.equal(got_1d, mean_dim(x, 2))
    assert torch.equal(got_keep, mean_dim(x, 2, keepdim=True))
    n = x.shape[0] * x.shape[1]
    assert torch.equal(got_2d, torch.sum(x, dim=(0, 1), dtype=torch.float32) / n)
