from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .comm import all_to_all
from .utils import generate_weights_idx, permute, sort_chunks_by_idxs, unpermute


def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))

    # [ep_size] represent the number of sum tokens in each rank
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()

    # gather all the number of tokens per expert from all ep ranks
    # [ep_size, num_experts]
    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    # [ep_size, num_local_experts]
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    # [ep_size]
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    # [num_local_expert]
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        torch.device("cpu"), non_blocking=True
    )

    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(
        torch.device("cpu"), non_blocking=True
    )

    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = permute(hidden_states, routing_map)

    # Validate split sizes match permuted tokens before all_to_all
    expected_tokens = sum(input_splits) if isinstance(input_splits, list) else input_splits.sum().item()
    actual_tokens = local_permuted_hidden_states.shape[0]
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"R3 EP split mismatch: input_splits sum ({expected_tokens}) != "
            f"permuted tokens ({actual_tokens}). "
            f"hidden_states: {hidden_states.shape}, expert_mask: {expert_mask.shape}"
        )

    global_permuted_hidden_states = all_to_all(ep_group, local_permuted_hidden_states, output_splits, input_splits)

    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    global_permuted_hidden_states = sort_chunks_by_idxs(
        global_permuted_hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )

    return global_permuted_hidden_states, routing_map, local_input_permutation_mapping, org_hidden_states_shape


def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: int,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    # [tokens, experts]
    weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)

    unpermute_outputs = unpermute(
        unpermute_outputs,
        weights_idx,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
    )

    return unpermute_outputs


# =============================================================================
# Unified EP Dispatch/Combine Interface
# =============================================================================
# Mirrors xorl-moe's dispatch/alltoall.py unified interface.
# Wraps existing preprocess() + token_pre_all2all() + tokens_post_all2all()
# so backends only need to provide a compute function.


@dataclass
class AllToAllDispatchContext:
    """State from dispatch needed for combine.

    Carries all information between ``alltoall_pre_dispatch()`` and
    ``alltoall_post_combine()`` so the EP compute step is stateless.
    """
    input_splits: List[int]
    output_splits: List[int]
    num_tokens_per_expert: torch.Tensor
    routing_map: torch.Tensor
    perm_mapping: torch.Tensor
    orig_shape: torch.Size
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor
    num_experts: int


def alltoall_pre_dispatch(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, AllToAllDispatchContext]:
    """Dispatch tokens to expert-owning ranks via all-to-all.

    Wraps ``preprocess()`` + ``token_pre_all2all()`` into a single call
    that returns a context object for ``alltoall_post_combine()``.

    Args:
        hidden_states: Input hidden states ``[num_tokens, hidden_dim]``.
        routing_weights: Routing weights ``[num_tokens, topk]``.
        selected_experts: Selected expert indices ``[num_tokens, topk]``.
        num_experts: Total number of experts.
        ep_group: Expert parallel process group.

    Returns:
        Tuple of:
        - expert_input: Permuted tokens ready for expert compute ``[N_local, hidden_dim]``.
        - cumsum: Cumulative sum of tokens per local expert ``[num_local_experts]``.
        - ctx: :class:`AllToAllDispatchContext` for ``alltoall_post_combine()``.
    """
    expert_mask = F.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    input_splits, output_splits, num_tokens_per_expert, sum_tokens = preprocess(
        expert_mask=expert_mask,
        num_experts=num_experts,
        ep_group=ep_group,
    )

    permuted_tokens, routing_map, perm_mapping, orig_shape = token_pre_all2all(
        hidden_states=hidden_states,
        expert_mask=expert_mask,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=num_tokens_per_expert,
        ep_group=ep_group,
    )

    cumsum = torch.cumsum(sum_tokens, dim=0).to(permuted_tokens.device)

    ctx = AllToAllDispatchContext(
        input_splits=input_splits,
        output_splits=output_splits,
        num_tokens_per_expert=num_tokens_per_expert,
        routing_map=routing_map,
        perm_mapping=perm_mapping,
        orig_shape=orig_shape,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        num_experts=num_experts,
    )

    return permuted_tokens, cumsum, ctx


def alltoall_post_combine(
    expert_output: torch.Tensor,
    ctx: AllToAllDispatchContext,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Combine expert outputs back to original ranks.

    Wraps ``tokens_post_all2all()`` using the context from
    ``alltoall_pre_dispatch()``.

    Args:
        expert_output: Expert outputs ``[N_local, hidden_dim]``.
        ctx: :class:`AllToAllDispatchContext` from ``alltoall_pre_dispatch()``.
        ep_group: Expert parallel process group.

    Returns:
        Output tensor ``[num_tokens, hidden_dim]``.
    """
    return tokens_post_all2all(
        expert_outputs=expert_output,
        routing_weights=ctx.routing_weights,
        selected_experts=ctx.selected_experts,
        num_experts=ctx.num_experts,
        input_splits=ctx.input_splits,
        output_splits=ctx.output_splits,
        num_global_tokens_per_local_expert=ctx.num_tokens_per_expert,
        routing_map=ctx.routing_map,
        local_input_permutation_mapping=ctx.perm_mapping,
        org_hidden_states_shape=ctx.orig_shape,
        ep_group=ep_group,
    )
