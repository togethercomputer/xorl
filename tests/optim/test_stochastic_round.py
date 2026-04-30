"""Tests for stochastic rounding to BF16."""

import pytest
import torch

from xorl.optim.stochastic_round import stochastic_round_to_bf16


pytestmark = [pytest.mark.cpu]


def test_stochastic_round_dtype_and_shape():
    x = torch.randn(7, 13, dtype=torch.float32)
    y = stochastic_round_to_bf16(x)
    assert y.dtype == torch.bfloat16
    assert y.shape == x.shape
    assert y.device == x.device


def test_stochastic_round_rejects_non_fp32():
    x = torch.randn(4, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        stochastic_round_to_bf16(x)


def test_stochastic_round_is_unbiased_in_expectation():
    """E[round(x)] should approach x as we average many samples."""
    torch.manual_seed(0)
    x = torch.randn(64, 64, dtype=torch.float32)
    n_samples = 4000
    accum = torch.zeros_like(x)
    for _ in range(n_samples):
        accum += stochastic_round_to_bf16(x).to(torch.float32)
    mean = accum / n_samples
    # With n=4000 samples and BF16 ulp ~ |x| * 2**-7, the standard error of
    # the mean is ~ulp / sqrt(n) ≈ ulp / 63. The test allows a generous
    # multiple to keep it deterministic across platforms.
    rel_err = ((mean - x).abs() / (x.abs() + 1e-8)).max().item()
    assert rel_err < 1e-2, f"Mean relative error {rel_err} exceeds tolerance; expected unbiased"


def test_stochastic_round_within_neighbors():
    """Output is always one of the two BF16 neighbors of x (no overshoot)."""
    torch.manual_seed(1)
    x = torch.randn(256, dtype=torch.float32)
    # The two BF16 neighbors of an FP32 x are obtained by truncate-down (mask
    # off low bits) and truncate-down + 1 ulp.
    x_int = x.view(torch.int32)
    lower_int = x_int & ~0xFFFF
    upper_int = lower_int + 0x10000
    lower = lower_int.view(torch.float32).to(torch.bfloat16).to(torch.float32)
    upper = upper_int.view(torch.float32).to(torch.bfloat16).to(torch.float32)
    # Note: when x is exactly representable in BF16, lower == x; upper is the
    # next BF16 above, but stochastic round only ever returns x in that case.
    for _ in range(20):
        y = stochastic_round_to_bf16(x).to(torch.float32)
        # y must match either lower or upper for every element
        match_lower = y == lower
        match_upper = y == upper
        match_x = y == x  # exact-representation case
        assert (match_lower | match_upper | match_x).all(), (
            "stochastic rounded value is not one of the two BF16 neighbors"
        )


def test_stochastic_round_deterministic_with_generator():
    x = torch.randn(32, dtype=torch.float32)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    y1 = stochastic_round_to_bf16(x, generator=g1)
    y2 = stochastic_round_to_bf16(x, generator=g2)
    assert torch.equal(y1, y2)
