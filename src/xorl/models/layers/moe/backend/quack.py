"""Quack MoE expert backend — quack group GEMM kernels."""

import torch

from xorl.ops.moe.quack import quack_moe_forward


def quack_expert_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    hidden_act: str = "silu",
    gate_up_proj: torch.Tensor | None = None,
    activation_native: bool = False,
    gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    fp8_compute: bool = False,
    fp8_grouped_backend: str = "triton_grouped",
    fp8_block_size: int = 128,
    swiglu_limit: float = 0.0,
    gated: bool = True,
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
        hidden_act: Activation kind ("silu", "gelu_tanh", or "clamped_swiglu").
        gate_up_proj: Optional pre-fused ``[num_experts, hidden, 2*intermediate]`` weight
            used by the fused Quack local path.
        gate_up_bias: Optional per-expert fused gate/up bias.
        down_bias: Optional per-expert down projection bias.
        fp8_compute: Use the experimental FP8 grouped GEMM path.
        fp8_grouped_backend: FP8 grouped GEMM backend name.
        fp8_block_size: FP8 quantization block size.
        **kwargs: Forwarded for forward compatibility; currently unused.

    Returns:
        Output tensor ``(num_tokens, hidden_dim)``.
    """
    del kwargs
    if not gated:
        raise NotImplementedError("quack backend does not support non-gated experts")
    return quack_moe_forward(
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
        activation_native=activation_native,
        fp8_compute=fp8_compute,
        fp8_grouped_backend=fp8_grouped_backend,
        fp8_block_size=fp8_block_size,
        gate_up_bias=gate_up_bias,
        down_bias=down_bias,
        swiglu_limit=swiglu_limit,
    )
