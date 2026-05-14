"""Eager MoE expert backend — per-expert matmul. For debugging/testing."""

import torch

from xorl.distributed.parallel_state import get_parallel_state
from xorl.ops.moe.activations import apply_moe_activation


def eager_expert_forward(
    hidden_states: torch.Tensor,
    expert_idx: int,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_act: str = "silu",
    gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward pass for a single expert (eager mode).

    Called in a loop by ``MoEBlock._eager_forward()``.

    Args:
        hidden_states: Input tensor of shape ``(num_tokens, hidden_dim)``.
        expert_idx: Index of the expert to use.
        gate_proj: Gate projection weights ``[num_experts, hidden, intermediate]``.
        up_proj: Up projection weights ``[num_experts, hidden, intermediate]``.
        down_proj: Down projection weights ``[num_experts, intermediate, hidden]``.
        hidden_act: Activation kind (e.g. ``"silu"``, ``"gelu_tanh"``,
            ``"clamped_swiglu"``) — dispatched via ``apply_moe_activation``.
        gate_up_bias: Optional per-expert bias ``[num_experts, 2*intermediate]``,
            split as ``[gate_bias | up_bias]`` along the last dim.
        down_bias: Optional per-expert bias ``[num_experts, hidden_dim]``.
    """
    assert not get_parallel_state().ep_enabled, "_moe_implementation='eager' does not support EP"
    gate_proj_out = torch.matmul(hidden_states, gate_proj[expert_idx])
    up_proj_out = torch.matmul(hidden_states, up_proj[expert_idx])
    if gate_up_bias is not None:
        intermediate_size = gate_proj_out.shape[-1]
        gate_proj_out = gate_proj_out + gate_up_bias[expert_idx, :intermediate_size]
        up_proj_out = up_proj_out + gate_up_bias[expert_idx, intermediate_size:]
    out = apply_moe_activation(hidden_act, gate_proj_out, up_proj_out)
    out = torch.matmul(out, down_proj[expert_idx])
    if down_bias is not None:
        out = out + down_bias[expert_idx]
    return out
