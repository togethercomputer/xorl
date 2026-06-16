"""Stochastic rounding to BF16.

Stochastic rounding produces an unbiased estimator of an FP32 input:
``E[stochastic_round_to_bf16(x).to(fp32)] == x``. Round-to-nearest is biased
when accumulating many small values; stochastic rounding distributes the
rounding decision in proportion to where ``x`` falls between its two BF16
neighbors, which is what makes BF16-transit reductions safe in expectation.

The implementation manipulates the FP32 bit pattern: BF16 is the FP32 bit
pattern with the low 16 mantissa bits truncated. Adding a uniform random
integer in ``[0, 2**16)`` to those low bits before truncation gives, for an
FP32 value at fractional position ``f`` between BF16 neighbors, probability
``f`` of rounding up and ``1-f`` of rounding down. This is the standard
implementation used in TransformerEngine and torchao.
"""

from typing import Optional

import torch


def stochastic_round_to_bf16(
    x: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Stochastically round an FP32 tensor to BF16.

    Args:
        x: FP32 tensor.
        generator: Optional ``torch.Generator`` for reproducibility.

    Returns:
        BF16 tensor with the same shape and device as ``x``.
    """
    if x.dtype != torch.float32:
        raise ValueError(f"stochastic_round_to_bf16 requires fp32 input, got {x.dtype}")

    x_int = x.contiguous().view(torch.int32)
    noise = torch.empty_like(x_int)
    noise.random_(0, 1 << 16, generator=generator)
    rounded_int = (x_int + noise) & ~0xFFFF
    return rounded_int.view(torch.float32).to(torch.bfloat16)
