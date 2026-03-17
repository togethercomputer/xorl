"""Quack MoE expert backend — moe_act variant (activation recompute)."""

import torch

from xorl.ops.moe.quack import quack_moe_forward_moe_act


def quack_expert_forward_moe_act(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    **kwargs,
) -> torch.Tensor:
    """Forward pass using quack group GEMM with moe_act (activation recompute)."""
    return quack_moe_forward_moe_act(
        module=None,
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
    )
