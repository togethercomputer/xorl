"""Optimized sorting and grouping operations for DeepEP."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_sorted_indices_kernel(
    sorted_token_idx_ptr,
    sorted_k_idx_ptr,
    sorted_flat_indices_ptr,
    num_valid: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_valid
    flat_idx = tl.load(sorted_flat_indices_ptr + idx, mask=mask, other=0)
    token_idx = flat_idx // topk
    k_idx = flat_idx % topk
    tl.store(sorted_token_idx_ptr + idx, token_idx, mask=mask)
    tl.store(sorted_k_idx_ptr + idx, k_idx, mask=mask)


def build_sorted_indices(
    sorted_flat_indices: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_valid = sorted_flat_indices.shape[0]
    device = sorted_flat_indices.device
    if num_valid == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
        )

    sorted_token_idx = torch.empty(num_valid, dtype=torch.int64, device=device)
    sorted_k_idx = torch.empty(num_valid, dtype=torch.int64, device=device)
    BLOCK_SIZE = 1024
    grid = ((num_valid + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _build_sorted_indices_kernel[grid](
        sorted_token_idx,
        sorted_k_idx,
        sorted_flat_indices,
        num_valid,
        topk,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return sorted_token_idx, sorted_k_idx


def group_tokens_by_expert_v2(
    expert_ids: torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    del num_experts
    device = expert_ids.device
    n = expert_ids.numel()
    flat_expert_ids = expert_ids.view(-1).long()
    flat_idx = torch.arange(n, device=device, dtype=torch.int64)
    composite_keys = torch.where(
        flat_expert_ids >= 0,
        flat_expert_ids * n + flat_idx,
        torch.iinfo(torch.int64).max,
    )
    _, sorted_indices = torch.sort(composite_keys, stable=False)
    valid_mask = flat_expert_ids >= 0
    num_valid = valid_mask.sum().item()
    if num_valid == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            0,
        )
    sorted_flat_indices = sorted_indices[:num_valid]
    sorted_token_idx = sorted_flat_indices // topk
    sorted_k_idx = sorted_flat_indices % topk
    return sorted_token_idx, sorted_k_idx, num_valid


def group_tokens_by_expert(
    expert_ids: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = expert_ids.device
    flat_expert_ids = expert_ids.view(-1).long()
    valid_mask = flat_expert_ids >= 0
    num_valid = valid_mask.sum().item()
    if num_valid == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            0,
        )
    valid_expert_ids = flat_expert_ids[valid_mask]
    valid_flat_indices = valid_mask.nonzero(as_tuple=True)[0]
    sorted_order = torch.argsort(valid_expert_ids, stable=True)
    sorted_flat_indices = valid_flat_indices[sorted_order]
    sorted_token_idx, sorted_k_idx = build_sorted_indices(sorted_flat_indices, topk)
    return sorted_token_idx, sorted_k_idx, num_valid
