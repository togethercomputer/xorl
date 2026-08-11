"""Fused SiLU-and-multiply (SwiGLU activation) using Triton kernels.

Computes: output = SiLU(input[:, :N]) * input[:, N:]
Used by both dense MLP (SwiGLU) and MoE expert layers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,  # intermediate_size (half of input dim)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU activation and element-wise multiplication.

    Computes: output = SiLU(input[:, :N]) * input[:, N:]

    Args:
        input_ptr: Input tensor of shape [num_tokens, 2*N]
        output_ptr: Output tensor of shape [num_tokens, N]
        N: intermediate_size (half of input dimension)
        BLOCK_SIZE: Block size for processing
    """
    row_idx = tl.program_id(0)

    # Process in blocks along the N dimension
    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load gate (first half) and up (second half)
        gate_ptr = input_ptr + row_idx * 2 * N + col_offsets
        up_ptr = input_ptr + row_idx * 2 * N + N + col_offsets

        gate = tl.load(gate_ptr, mask=mask, other=0.0)
        up = tl.load(up_ptr, mask=mask, other=0.0)

        # Compute SiLU(gate) * up
        gate_f32 = gate.to(tl.float32)
        silu_gate = gate_f32 * tl.sigmoid(gate_f32)
        result = silu_gate.to(gate.dtype) * up

        # Store result
        out_ptr = output_ptr + row_idx * N + col_offsets
        tl.store(out_ptr, result, mask=mask)


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """Fused SiLU activation and element-wise multiplication.

    Computes: output = SiLU(input[:, :N]) * input[:, N:]
    where N = input.shape[-1] // 2

    Args:
        input_tensor: Input tensor of shape [..., 2*N]

    Returns:
        Output tensor of shape [..., N]
    """
    assert input_tensor.shape[-1] % 2 == 0, "Last dimension must be even"

    original_shape = input_tensor.shape
    input_2d = input_tensor.view(-1, original_shape[-1])

    num_tokens = input_2d.shape[0]
    N = input_2d.shape[1] // 2

    output = torch.empty(
        (num_tokens, N),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    BLOCK_SIZE = 1024
    grid = (num_tokens,)

    _silu_and_mul_kernel[grid](
        input_2d,
        output,
        N,
        BLOCK_SIZE,
    )

    # Reshape to match input shape (except last dim is halved)
    output_shape = list(original_shape)
    output_shape[-1] = N
    return output.view(output_shape)


@triton.jit
def _silu_and_mul_backward_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for fused SiLU and multiply.

    Given: y = SiLU(gate) * up
    Computes:
        d_gate = grad_output * up * SiLU_grad(gate)
        d_up = grad_output * SiLU(gate)
    """
    row_idx = tl.program_id(0)

    for block_start in range(0, N, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # Load grad_output
        grad_out = tl.load(grad_output_ptr + row_idx * N + col_offsets, mask=mask, other=0.0)

        # Load gate and up
        gate = tl.load(input_ptr + row_idx * 2 * N + col_offsets, mask=mask, other=0.0)
        up = tl.load(input_ptr + row_idx * 2 * N + N + col_offsets, mask=mask, other=0.0)

        # Compute SiLU and its gradient
        gate_f32 = gate.to(tl.float32)
        sigmoid_gate = tl.sigmoid(gate_f32)
        silu_gate = gate_f32 * sigmoid_gate
        silu_grad = sigmoid_gate + gate_f32 * sigmoid_gate * (1.0 - sigmoid_gate)

        # Compute gradients
        grad_out_f32 = grad_out.to(tl.float32)
        up_f32 = up.to(tl.float32)

        d_gate = grad_out_f32 * up_f32 * silu_grad
        d_up = grad_out_f32 * silu_gate

        # Store gradients
        tl.store(grad_input_ptr + row_idx * 2 * N + col_offsets, d_gate.to(gate.dtype), mask=mask)
        tl.store(grad_input_ptr + row_idx * 2 * N + N + col_offsets, d_up.to(up.dtype), mask=mask)


def silu_and_mul_backward(grad_output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """Backward pass for fused SiLU and multiply.

    Args:
        grad_output: Gradient of output, shape [..., N]
        input_tensor: Original input, shape [..., 2*N]

    Returns:
        Gradient of input, shape [..., 2*N]
    """
    original_shape = input_tensor.shape
    input_2d = input_tensor.view(-1, original_shape[-1])
    grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

    num_tokens = input_2d.shape[0]
    N = input_2d.shape[1] // 2

    grad_input = torch.empty_like(input_2d)

    BLOCK_SIZE = 1024
    grid = (num_tokens,)

    _silu_and_mul_backward_kernel[grid](
        grad_output_2d,
        input_2d,
        grad_input,
        N,
        BLOCK_SIZE,
    )

    return grad_input.view(original_shape)


class SiluAndMulFunction(torch.autograd.Function):
    """Autograd function for fused SiLU and multiply."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return silu_and_mul(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = ctx.saved_tensors
        return silu_and_mul_backward(grad_output, input_tensor)


def fused_silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """Fused SiLU and multiply with autograd support.

    Args:
        input_tensor: Input tensor of shape [..., 2*N]

    Returns:
        Output tensor of shape [..., N]
    """
    return SiluAndMulFunction.apply(input_tensor)
