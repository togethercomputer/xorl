"""Token-choice top-k router for MoE layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Top-K routing: softmax -> topk -> optional renormalization.

    This module is *stateless* — it does **not** own the gate ``nn.Linear``.
    The gate stays on ``MoEBlock.gate`` to preserve the checkpoint path
    ``mlp.gate.weight``.

    Args:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token.
        norm_topk_prob: Whether to renormalize top-k routing weights.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def forward(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        """Compute routing weights and expert selection.

        Args:
            router_logits: Raw logits from the gate ``(num_tokens, num_experts)``.
            input_dtype: Dtype to cast final routing weights to.

        Returns:
            routing_weights: ``(num_tokens, top_k)`` weights per selected expert.
            selected_experts: ``(num_tokens, top_k)`` expert indices.
        """
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(input_dtype)
        return routing_weights, selected_experts

    @classmethod
    def from_config(cls, config):
        """Create from a model config (e.g. ``Qwen3MoeConfig``)."""
        return cls(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )
