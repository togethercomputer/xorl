"""Load balancing loss for MoE training (micro-batch and global-batch)."""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F


class LoadBalancingBuffer:
    """Accumulates synchronized expert-selection counts over a gradient-accumulation window.

    Implements the buffer of Algorithm 1 in Qiu et al. 2025, "Demons in the Detail:
    On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert
    Models" (https://arxiv.org/abs/2501.11873). One buffer is created per optimizer
    step (one full gradient-accumulation window) and discarded after the step.

    At every micro-batch, the local per-expert selection counts and token totals are
    all-reduced (sum) across the data-parallel group and added to the running buffer.
    The current load-balancing frequency ``f_i`` is then derived from the accumulated
    buffer. As more micro-batches are accumulated, ``f_i`` approaches the true
    global-batch frequency ``f̄_i``, while the routing probability ``P_i`` used in the
    loss stays local to the micro-batch so gradients still flow to the local gate.
    """

    def __init__(self) -> None:
        self.counts: Optional[torch.Tensor] = None
        self.denom: Optional[torch.Tensor] = None

    def accumulate(
        self,
        local_counts: torch.Tensor,
        local_denom: torch.Tensor,
        dp_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Synchronize counts across DP, add them to the buffer, return current ``f_i``.

        Args:
            local_counts: Per-expert selection counts for this micro-batch,
                shape ``[top_k, num_experts]`` (detached; carries no gradient).
            local_denom: Per-expert token totals for this micro-batch, same shape.
            dp_group: Data-parallel process group to sum over. ``None`` skips the
                all-reduce (single-rank or local-only mode).

        Returns:
            ``buffer.counts / buffer.denom`` — the global-batch frequency estimate so
            far, shape ``[top_k, num_experts]``, detached.
        """
        counts = local_counts.detach()
        denom = local_denom.detach()
        if dp_group is not None:
            # Single collective over a tiny [2, top_k, num_experts] tensor.
            packed = torch.stack([counts, denom])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=dp_group)
            counts, denom = packed[0], packed[1]

        if self.counts is None:
            self.counts = counts.clone()
            self.denom = denom.clone()
        else:
            self.counts += counts
            self.denom += denom

        return self.counts / self.denom


def global_load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
    buffer: Optional[LoadBalancingBuffer] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer with
    optional global reduction across data-parallel ranks.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
    This function implements the loss function presented in equations (4) - (6)
    of the paper. It aims at penalizing cases where the routing between experts
    is too unbalanced.

    The token-fraction term ``f_i`` (``tokens_per_expert``) determines the *scope*
    over which routing is balanced; the routing-probability term ``P_i``
    (``router_prob_per_expert``) always stays local so gradients flow to the local
    gate parameters. There are two scopes:

    * **Micro-batch** (``buffer`` is ``None``): ``f_i`` is the per-micro-batch
      fraction, optionally averaged across ``dp_group``. This pushes the router to
      balance experts *within each micro-batch* (often near sequence level), an
      overly strict constraint that inhibits expert specialization.
    * **Global-batch** (``buffer`` provided): ``f_i`` is accumulated across the
      gradient-accumulation window and summed across ``dp_group`` via ``buffer``,
      so routing is balanced over the whole global batch. This relaxes the
      constraint and improves performance and expert specialization, per Qiu
      et al. 2025 (https://arxiv.org/abs/2501.11873, Alg. 1).

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
        buffer:
            Optional :class:`LoadBalancingBuffer` enabling global-batch balancing.
            When provided, ``f_i`` is accumulated across micro-batches (and summed
            across ``dp_group``) instead of computed per micro-batch.

    Returns:
        The auxiliary loss (scalar tensor), or ``0`` if MoE is not active.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    gate_logits = tuple(layer_gate for layer_gate in gate_logits if layer_gate is not None)
    if not gate_logits:
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        if buffer is not None:
            # Global-batch: accumulate counts / token totals across the GA window.
            counts = torch.sum(expert_mask.float(), dim=0)
            denom = torch.full_like(counts, float(expert_mask.shape[0]))
            tokens_per_expert = buffer.accumulate(counts, denom, dp_group)
        else:
            # Micro-batch: percentage of tokens routed to each expert.
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

        if buffer is not None:
            # Global-batch: accumulate masked counts / token totals across the GA window.
            counts = torch.sum(expert_mask.float() * expert_attention_mask, dim=0)
            denom = torch.sum(expert_attention_mask, dim=0)
            tokens_per_expert = buffer.accumulate(counts, denom, dp_group)
        else:
            # Micro-batch: percentage of tokens routed to each expert.
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
