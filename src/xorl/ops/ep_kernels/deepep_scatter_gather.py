"""Optimized Triton kernels for DeepEP scatter/gather operations."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_by_index_kernel(
    output_ptr,
    input_ptr,
    indices_ptr,
    num_gather: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_gather:
        return
    src_idx = tl.load(indices_ptr + pid)
    for h_start in range(0, hidden_dim, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_dim
        val = tl.load(input_ptr + src_idx * hidden_dim + h_offs, mask=mask, other=0.0)
        tl.store(output_ptr + pid * hidden_dim + h_offs, val, mask=mask)


@triton.jit
def _weighted_scatter_add_no_atomic_kernel(
    output_ptr,
    expert_output_ptr,
    sorted_token_idx_ptr,
    weights_ptr,
    num_expert_outputs: tl.constexpr,
    num_tokens: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_expert_outputs:
        return
    token_idx = tl.load(sorted_token_idx_ptr + pid)
    weight = tl.load(weights_ptr + pid).to(tl.float32)
    for h_start in range(0, hidden_dim, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_dim
        val = tl.load(expert_output_ptr + pid * hidden_dim + h_offs, mask=mask, other=0.0).to(tl.float32)
        weighted_val = val * weight
        out_ptr = output_ptr + token_idx * hidden_dim + h_offs
        tl.atomic_add(out_ptr, weighted_val, mask=mask)


class DeepEPScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, sorted_indices):
        num_gather = sorted_indices.shape[0]
        hidden_dim = tokens.shape[1]
        output = torch.empty(num_gather, hidden_dim, dtype=tokens.dtype, device=tokens.device)
        if num_gather == 0:
            ctx.save_for_backward(sorted_indices)
            ctx.num_tokens = tokens.shape[0]
            ctx.hidden_dim = hidden_dim
            return output
        BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
        grid = (num_gather,)
        _gather_by_index_kernel[grid](output, tokens, sorted_indices, num_gather, hidden_dim, BLOCK_H=BLOCK_H)
        ctx.save_for_backward(sorted_indices)
        ctx.num_tokens = tokens.shape[0]
        ctx.hidden_dim = hidden_dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (sorted_indices,) = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        hidden_dim = ctx.hidden_dim
        grad_tokens = torch.zeros(num_tokens, hidden_dim, dtype=torch.float32, device=grad_output.device)
        if sorted_indices.shape[0] > 0:
            expanded_indices = sorted_indices.unsqueeze(1).expand(-1, hidden_dim)
            grad_tokens.scatter_add_(0, expanded_indices, grad_output.float())
        if grad_output.dtype != torch.float32:
            grad_tokens = grad_tokens.to(grad_output.dtype)
        return grad_tokens, None


class DeepEPWeightedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, sorted_token_idx, sorted_k_idx, weights, num_tokens):
        num_expert_tokens = expert_output.shape[0]
        hidden_dim = expert_output.shape[1]
        dtype = expert_output.dtype
        device = expert_output.device
        output = torch.zeros(num_tokens, hidden_dim, dtype=torch.float32, device=device)
        if num_expert_tokens == 0:
            ctx.save_for_backward(expert_output, sorted_token_idx, sorted_k_idx, weights)
            ctx.num_tokens = num_tokens
            return output.to(dtype)
        gathered_weights = weights[sorted_token_idx, sorted_k_idx]
        BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
        grid = (num_expert_tokens,)
        _weighted_scatter_add_no_atomic_kernel[grid](
            output,
            expert_output,
            sorted_token_idx,
            gathered_weights,
            num_expert_tokens,
            num_tokens,
            hidden_dim,
            BLOCK_H=BLOCK_H,
        )
        ctx.save_for_backward(expert_output, sorted_token_idx, sorted_k_idx, weights)
        ctx.num_tokens = num_tokens
        return output.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        expert_output, sorted_token_idx, sorted_k_idx, weights = ctx.saved_tensors
        num_expert_tokens = expert_output.shape[0]
        topk = weights.shape[1]
        grad_output = grad_output.contiguous()
        gathered_weights = weights[sorted_token_idx, sorted_k_idx]
        grad_expert_output = grad_output[sorted_token_idx] * gathered_weights.unsqueeze(1)
        grad_weights = torch.zeros_like(weights)
        if num_expert_tokens > 0:
            dot_products = (grad_output[sorted_token_idx] * expert_output).sum(dim=1)
            flat_idx = sorted_token_idx * topk + sorted_k_idx
            grad_weights.view(-1).scatter_add_(0, flat_idx, dot_products)
        return grad_expert_output, None, None, grad_weights, None


def deepep_scatter(tokens: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
    return DeepEPScatter.apply(tokens, sorted_indices)


def deepep_weighted_gather(
    expert_output: torch.Tensor,
    sorted_token_idx: torch.Tensor,
    sorted_k_idx: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return DeepEPWeightedGather.apply(expert_output, sorted_token_idx, sorted_k_idx, weights, num_tokens)
