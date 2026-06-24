import pytest
import torch
import torch.nn.functional as F

from xorl.ops.fused_silu_and_mul import fused_silu_and_mul, silu_and_mul_backward


def _native_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def test_fused_silu_and_mul_cpu_falls_back_to_native_autograd():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 10, dtype=torch.float32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    grad = torch.randn(2, 3, 5)

    out = fused_silu_and_mul(x)
    ref = _native_silu_and_mul(ref_x)
    torch.testing.assert_close(out, ref)

    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(x.grad, ref_x.grad)


def test_silu_and_mul_backward_cpu_matches_native_autograd():
    torch.manual_seed(1)
    x = torch.randn(4, 14, dtype=torch.float32, requires_grad=True)
    grad = torch.randn(4, 7)

    ref = _native_silu_and_mul(x)
    ref.backward(grad)

    got = silu_and_mul_backward(grad, x.detach())
    torch.testing.assert_close(got, x.grad)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton fused SwiGLU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 128), (5, 2048), (2, 4864)])
def test_fused_silu_and_mul_cuda_matches_native_forward(dtype, shape):
    torch.manual_seed(2)
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 3.0).contiguous()

    got = fused_silu_and_mul(x)
    ref = _native_silu_and_mul(x)

    torch.testing.assert_close(got, ref, rtol=0.02, atol=0.015625)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton fused SwiGLU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_silu_and_mul_cuda_backward_matches_native(dtype):
    torch.manual_seed(3)
    x = (torch.randn(4, 1536, device="cuda", dtype=dtype) * 2.0).contiguous().requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    grad = torch.randn(4, 768, device="cuda", dtype=dtype)

    fused_silu_and_mul(x).backward(grad)
    _native_silu_and_mul(ref_x).backward(grad)

    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0.025, atol=0.015625)
