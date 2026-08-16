"""Exact final cast for CUDA canonical-MoE FP64 accumulators."""

from __future__ import annotations

import torch


try:  # Triton is optional in CPU-only development environments.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _TRITON_AVAILABLE = False


def validate_canonical_moe_fp64_cast_input(
    value: torch.Tensor,
    output_dtype: torch.dtype,
) -> None:
    if value.dtype is not torch.float64:
        raise TypeError(f"Canonical MoE direct cast requires FP64 input, got {value.dtype}")
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Canonical MoE direct cast requires BF16 or FP16 output, got {output_dtype}")
    if not value.is_cuda:
        raise ValueError("Canonical MoE direct cast requires a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError("Canonical MoE direct cast requires contiguous input")


if _TRITON_AVAILABLE:

    @triton.jit
    def _canonical_moe_fp64_to_lowp_rne_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # Store the FP64 register directly through the BF16 pointer.  An
        # explicit FP32 conversion here would introduce a double-rounding
        # boundary and can select the opposite BF16 neighbor at a midpoint.
        tl.store(output_ptr + offsets, value, mask=mask)


def _canonical_moe_fp64_to_lowp_cuda(
    value: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:  # pragma: no cover - guarded by CUDA environments
        raise RuntimeError("Canonical MoE CUDA direct cast requires Triton")
    output = torch.empty_like(value, dtype=output_dtype)
    if output.numel() == 0:
        return output
    block_size = 256
    _canonical_moe_fp64_to_lowp_rne_kernel[(triton.cdiv(output.numel(), block_size),)](
        value,
        output,
        output.numel(),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


@torch.library.custom_op("xorl_k3::_canonical_moe_fp64_to_bf16_rne", mutates_args=())
def _canonical_moe_fp64_to_bf16_rne_op(value: torch.Tensor) -> torch.Tensor:
    """Round FP64 directly to BF16."""
    validate_canonical_moe_fp64_cast_input(value, torch.bfloat16)
    return _canonical_moe_fp64_to_lowp_cuda(value, torch.bfloat16)


@_canonical_moe_fp64_to_bf16_rne_op.register_fake
def _canonical_moe_fp64_to_bf16_rne_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value, dtype=torch.bfloat16)


@torch.library.custom_op("xorl_k3::_canonical_moe_fp64_to_fp16_rne", mutates_args=())
def _canonical_moe_fp64_to_fp16_rne_op(value: torch.Tensor) -> torch.Tensor:
    """Round FP64 directly to FP16."""
    validate_canonical_moe_fp64_cast_input(value, torch.float16)
    return _canonical_moe_fp64_to_lowp_cuda(value, torch.float16)


@_canonical_moe_fp64_to_fp16_rne_op.register_fake
def _canonical_moe_fp64_to_fp16_rne_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value, dtype=torch.float16)


def _canonical_moe_fp64_to_lowp_backward(ctx, grad_output: torch.Tensor):
    del ctx
    # Match torch's straight-through cast derivative.  The surrounding FP64
    # tree then distributes this FP64 gradient to every contributor before its
    # original BF16 leaf cast handles the final backward conversion.
    return grad_output.to(torch.float64)


_canonical_moe_fp64_to_bf16_rne_op.register_autograd(
    _canonical_moe_fp64_to_lowp_backward,
)
_canonical_moe_fp64_to_fp16_rne_op.register_autograd(
    _canonical_moe_fp64_to_lowp_backward,
)


def canonical_moe_fp64_to_lowp_rne(
    value: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Validate and directly round FP64 to the canonical transport dtype."""
    validate_canonical_moe_fp64_cast_input(value, output_dtype)
    if output_dtype is torch.bfloat16:
        return _canonical_moe_fp64_to_bf16_rne_op(value)
    return _canonical_moe_fp64_to_fp16_rne_op(value)


__all__ = [
    "canonical_moe_fp64_to_lowp_rne",
    "validate_canonical_moe_fp64_cast_input",
]
