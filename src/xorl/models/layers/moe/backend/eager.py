"""Eager MoE expert backend — per-expert matmul. For debugging/testing."""

import torch

from xorl.distributed.parallel_state import get_parallel_state


def eager_expert_forward(
    hidden_states: torch.Tensor,
    expert_idx: int,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn,
) -> torch.Tensor:
    """Forward pass for a single expert (eager mode).

    Called in a loop by ``MoEBlock._eager_forward()``.

    Args:
        hidden_states: Input tensor of shape ``(num_tokens, hidden_dim)``.
        expert_idx: Index of the expert to use.
        gate_proj: Gate projection weights ``[num_experts, hidden, intermediate]``.
        up_proj: Up projection weights ``[num_experts, hidden, intermediate]``.
        down_proj: Down projection weights ``[num_experts, intermediate, hidden]``.
        act_fn: Activation function (e.g. ``torch.nn.SiLU``).

    Returns:
        Output tensor of shape ``(num_tokens, hidden_dim)``.
    """
    assert not get_parallel_state().ep_enabled, "_moe_implementation='eager' does not support EP"
    gate_proj_out = torch.matmul(hidden_states, gate_proj[expert_idx])
    up_proj_out = torch.matmul(hidden_states, up_proj[expert_idx])
    out = act_fn(gate_proj_out) * up_proj_out
    out = torch.matmul(out, down_proj[expert_idx])
    return out
