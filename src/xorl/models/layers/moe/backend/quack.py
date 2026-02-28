"""Quack MoE expert backend — quack group GEMM kernels."""

import torch

from xorl.ops import quack_moe_forward


def quack_expert_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    **kwargs,
) -> torch.Tensor:
    """Forward pass using quack group GEMM kernels.

    EP is handled centrally by ``MoEExperts._ep_forward()``.

    Args:
        hidden_states: Input tensor ``(num_tokens, hidden_dim)``.
        routing_weights: Routing weights ``(num_tokens, top_k)``.
        selected_experts: Selected expert indices ``(num_tokens, top_k)``.
        gate_proj: Gate projection weights ``[num_experts, hidden, intermediate]``.
        up_proj: Up projection weights ``[num_experts, hidden, intermediate]``.
        down_proj: Down projection weights ``[num_experts, intermediate, hidden]``.
        num_experts: Total number of experts.
        **kwargs: Extra arguments (ignored).

    Returns:
        Output tensor ``(num_tokens, hidden_dim)``.
    """
    return quack_moe_forward(
        module=None,
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
    )
