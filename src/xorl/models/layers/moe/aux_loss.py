"""Globally-reduced load balancing loss for MoE training."""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F


def global_load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer with
    optional global reduction across data-parallel ranks.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
    This function implements the loss function presented in equations (4) - (6)
    of the paper. It aims at penalizing cases where the routing between experts
    is too unbalanced.

    When ``dp_group`` is provided, ``tokens_per_expert`` is all-reduced across
    the group so the loss reflects the global routing distribution rather than
    just the local micro-batch. ``router_prob_per_expert`` stays local so that
    gradients flow back to the local gate parameters.

    Args:
        gate_logits:
            Logits from the ``gate``, should be a tuple of tensors of
            shape ``[batch_size * sequence_length, num_experts]`` (one per MoE layer).
        num_experts:
            Number of experts.
        top_k:
            The number of experts to route per-token.
        attention_mask:
            Optional attention mask of shape ``[batch_size, sequence_length]``.
        dp_group:
            Data-parallel process group for global reduction. ``None`` skips
            the all-reduce (single-rank or local-only mode).

    Returns:
        The auxiliary loss (scalar tensor), or ``0`` if MoE is not active.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Globally reduce tokens_per_expert across DP ranks
        if dp_group is not None:
            dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.AVG, group=dp_group)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each expert
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Globally reduce tokens_per_expert across DP ranks
        if dp_group is not None:
            dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.AVG, group=dp_group)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
