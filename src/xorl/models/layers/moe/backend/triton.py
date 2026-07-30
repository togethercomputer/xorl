"""Triton MoE expert backend — custom Triton group GEMM kernels."""

import torch

from xorl.ops.moe.triton import triton_moe_forward


def triton_expert_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    hidden_act: str = "silu",
    gate_up_proj: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    gated: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Forward pass using custom Triton group GEMM kernels.

    Uses ``xorl.ops.moe_experts_forward`` which dispatches to Triton kernels for
    scatter/gather and group GEMM (``group_gemm_same_nk``).

    Args:
        hidden_states: Input tensor ``(num_tokens, hidden_dim)``.
        routing_weights: Routing weights ``(num_tokens, top_k)``.
        selected_experts: Selected expert indices ``(num_tokens, top_k)``.
        gate_proj: Gate projection weights ``[num_experts, hidden, intermediate]``.
        up_proj: Up projection weights ``[num_experts, hidden, intermediate]``.
        down_proj: Down projection weights ``[num_experts, intermediate, hidden]``.
        num_experts: Total number of experts.
        hidden_act: Activation kind ("silu" or "gelu_tanh"; "relu2" when ``gated=False``).
        gate_up_proj: Pre-fused ``[num_experts, hidden, 2*intermediate]`` weight
            used by the fused-GEMM path (required by ``TritonMoeExpertsFunction``).
            Holds the plain up projection ``[num_experts, in_dim, intermediate]``
            when ``gated=False``.
        swiglu_limit: Optional gate pre-activation clamp (DeepSeek-V4 SwiGLU limit).
        gated: Whether the experts use a gated (GLU) first projection.
        **kwargs: Forwarded for forward compatibility; currently unused.

    Returns:
        Output tensor ``(num_tokens, hidden_dim)``.
    """
    del kwargs
    return triton_moe_forward(
        module=None,
        num_experts=num_experts,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        gate_up_proj=gate_up_proj,
        hidden_act=hidden_act,
        swiglu_limit=swiglu_limit,
        gated=gated,
    )
