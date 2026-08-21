"""Compile-stable one-round arithmetic for canonical MoE contributor leaves."""

from __future__ import annotations

import torch


try:  # Triton is optional in CPU-only development environments.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _TRITON_AVAILABLE = False


def validate_canonical_moe_leaf_operands(
    shared: torch.Tensor,
    routed: torch.Tensor,
) -> None:
    if shared.shape != routed.shape:
        raise ValueError(
            f"Canonical MoE leaf operands must have equal shapes, got {tuple(shared.shape)} and {tuple(routed.shape)}"
        )
    if shared.dtype != routed.dtype or shared.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise TypeError(
            f"Canonical MoE leaf operands must have the same BF16 or FP16 dtype, got {shared.dtype} and {routed.dtype}"
        )
    if shared.device != routed.device:
        raise ValueError(f"Canonical MoE leaf operands must share a device, got {shared.device} and {routed.device}")
    if shared.device.type not in ("cpu", "cuda"):
        raise ValueError(f"Canonical MoE leaf operands must be CPU or CUDA tensors, got {shared.device.type}")
    if not shared.is_contiguous() or not routed.is_contiguous():
        raise ValueError("Canonical MoE leaf operands must be contiguous")


if _TRITON_AVAILABLE:

    @triton.jit
    def _fp32_fma_rn(multiplicand, multiplier, addend):
        return tl.inline_asm_elementwise(
            asm="fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=(multiplicand, multiplier, addend),
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _canonical_moe_leaf_fp32_kernel(
        shared_ptr,
        routed_ptr,
        output_ptr,
        n_elements,
        routed_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        routed = tl.load(routed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        leaf = _fp32_fma_rn(routed, routed_scale.to(tl.float32), shared)
        # The accumulator stays in a register. This store is the sole cast to
        # the declared BF16/FP16 transport dtype.
        tl.store(output_ptr + offsets, leaf, mask=mask)


def _canonical_moe_leaf_cuda(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:  # pragma: no cover - guarded by CUDA environments
        raise RuntimeError("Canonical MoE CUDA leaf construction requires Triton")
    output = torch.empty_like(shared)
    if output.numel() == 0:
        return output
    block_size = 256
    _canonical_moe_leaf_fp32_kernel[(triton.cdiv(output.numel(), block_size),)](
        shared,
        routed,
        output,
        output.numel(),
        float(routed_scale),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def _canonical_moe_leaf_cpu(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
) -> torch.Tensor:
    # BF16/FP16 operands and an FP32 scale have at most 35 product bits, so the
    # product is exact in FP64. The FP64 add followed by one FP32 round is the
    # independent CPU reference (apart from a theoretical double-rounding
    # corner); the CUDA identity tests pin the production FMA bytes. The final
    # conversion is the sole transport cast.
    scale_fp32 = torch.tensor(routed_scale, dtype=torch.float32)
    pre_cast = shared.double() + routed.double() * scale_fp32.double()
    return pre_cast.float().to(shared.dtype)


@torch.library.custom_op("xorl_k3::_canonical_moe_leaf_fp32_v1", mutates_args=())
def _canonical_moe_leaf_fp32_v1_op(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
) -> torch.Tensor:
    """Opaque one-FMA leaf primitive; output is the transport dtype."""
    validate_canonical_moe_leaf_operands(shared, routed)
    if shared.is_cuda:
        return _canonical_moe_leaf_cuda(shared, routed, routed_scale)
    return _canonical_moe_leaf_cpu(shared, routed, routed_scale)


@_canonical_moe_leaf_fp32_v1_op.register_fake
def _canonical_moe_leaf_fp32_v1_fake(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
) -> torch.Tensor:
    del routed, routed_scale
    return torch.empty_like(shared)


def _canonical_moe_leaf_setup_context(ctx, inputs, output) -> None:
    del output
    shared, _routed, routed_scale = inputs
    ctx.input_dtype = shared.dtype
    ctx.routed_scale = float(routed_scale)


def _canonical_moe_leaf_backward(ctx, grad_output: torch.Tensor):
    grad_fp32 = grad_output.float()
    grad_shared = grad_fp32.to(ctx.input_dtype)
    grad_routed = (grad_fp32 * ctx.routed_scale).to(ctx.input_dtype)
    return grad_shared, grad_routed, None


_canonical_moe_leaf_fp32_v1_op.register_autograd(
    _canonical_moe_leaf_backward,
    setup_context=_canonical_moe_leaf_setup_context,
)


def canonical_moe_leaf_fp32_v1_op(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
) -> torch.Tensor:
    """Validate and execute the compile-opaque canonical contributor leaf."""
    validate_canonical_moe_leaf_operands(shared, routed)
    return _canonical_moe_leaf_fp32_v1_op(shared, routed, float(routed_scale))


__all__ = ["canonical_moe_leaf_fp32_v1_op", "validate_canonical_moe_leaf_operands"]
