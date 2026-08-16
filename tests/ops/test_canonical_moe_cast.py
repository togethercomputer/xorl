"""Focused CUDA gate for the canonical MoE FP64 output boundary."""

import pytest
import torch

from xorl.ops.canonical_moe_cast import canonical_moe_fp64_to_lowp_rne


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("output_dtype", "midpoint_witness", "expected_bits", "double_rounded_bits"),
    [
        (torch.bfloat16, 338.9999990463257, 0x43A9, 0x43AA),
        (torch.float16, 338.3749990463257, 0x5D49, 0x5D4A),
    ],
)
def test_cuda_direct_rne_result_and_backward(
    output_dtype: torch.dtype,
    midpoint_witness: float,
    expected_bits: int,
    double_rounded_bits: int,
):
    value = torch.tensor([midpoint_witness], device="cuda", dtype=torch.float64, requires_grad=True)

    output = canonical_moe_fp64_to_lowp_rne(value, output_dtype)

    assert int(output.view(torch.uint16).item()) == expected_bits
    assert int(value.detach().to(output_dtype).view(torch.uint16).item()) == double_rounded_bits
    output.backward(torch.tensor([1.25], device="cuda", dtype=output_dtype))
    assert torch.equal(value.grad, torch.tensor([1.25], device="cuda", dtype=torch.float64))
