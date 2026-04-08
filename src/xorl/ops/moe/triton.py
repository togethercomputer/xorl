import torch

from xorl.utils.import_utils import is_fused_moe_available


if is_fused_moe_available():
    from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
    from xorl.ops.group_gemm.kernel.moe import (
        expert_histogram,
        moe_gather,
        moe_index_compute,
        moe_scatter,
    )


class TritonEPGroupGemm(torch.autograd.Function):
    """EP expert MLP with fused gate+up GEMM. Zero-copy weight references.

    Forward: single ``x @ gate_up_proj`` GEMM → split → SwiGLU → down GEMM.
    Backward: fused dgrad/wgrad for gate+up (2x fewer GEMMs than split version).
    ``save_for_backward`` stores the original ``gate_up_proj`` parameter
    reference (zero extra memory) plus the fused activation output.
    """

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores=None):
        max_M = permute_tokens.shape[0]
        I = intermediate_size
        ctx.has_expert_scores = expert_scores is not None

        gate_up_output = group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_output = gate_activation * up_output
        if expert_scores is not None:
            gated_output = gated_output * expert_scores.to(gated_output.dtype).unsqueeze(-1)
        del gate_activation

        down_output = group_gemm_same_nk(
            a=gated_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        del gated_output

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(permute_tokens, cumsum, gate_up_proj, down_proj, gate_up_output, expert_scores)
        ctx.intermediate_size = I

        return down_output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, cumsum, gate_up_proj, down_proj, gate_up_output, expert_scores = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = grad_output.shape[0]

        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_output = gate_activation * up_output
        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)
        gated_weighted = gated_output * expert_scores.unsqueeze(-1)

        # dgrad FC2
        grad_gated_weighted = group_gemm_same_nk(
            a=grad_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )

        # wgrad FC2
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_weighted,
                b=grad_output,
                c=grad_down_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        grad_expert_scores = None
        if ctx.has_expert_scores:
            grad_expert_scores = (grad_gated_weighted * gated_output).sum(dim=-1).to(expert_scores_dtype)
        del gated_output, gated_weighted

        grad_gated_output = grad_gated_weighted * expert_scores.unsqueeze(-1)
        del grad_gated_weighted

        # Activation backward
        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output, gate_up_output

        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # Fused dgrad FC1: cat grads → single GEMM with gate_up_proj (no .contiguous() copies)
        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output
        grad_permute_tokens = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )

        # Fused wgrad FC1: single GEMM produces grad_gate_up_proj directly
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            group_gemm_same_mn(
                a=permute_tokens,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_gate_up_act

        return (
            grad_permute_tokens,
            None,  # cumsum
            grad_gate_up_proj,
            grad_down_proj,
            None,  # intermediate_size
            grad_expert_scores,
        )


class TritonEPGroupGemmMoeAct(torch.autograd.Function):
    """EP expert MLP with moe_act and fused gate+up GEMM.

    Saves only inputs + weights (no activation outputs), recomputes via
    fused GEMM in backward. Zero extra memory from weight copies.
    """

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores=None):
        max_M = permute_tokens.shape[0]
        I = intermediate_size
        ctx.has_expert_scores = expert_scores is not None

        gate_up_output = group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )

        gate_activation = torch.ops.aten.silu(gate_up_output[..., :I])
        gated_output = gate_activation * gate_up_output[..., I:]
        if expert_scores is not None:
            gated_output = gated_output * expert_scores.to(gated_output.dtype).unsqueeze(-1)
        del gate_activation, gate_up_output

        down_output = group_gemm_same_nk(
            a=gated_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        del gated_output

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(permute_tokens, cumsum, gate_up_proj, down_proj, expert_scores)
        ctx.intermediate_size = I
        return down_output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, cumsum, gate_up_proj, down_proj, expert_scores = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = grad_output.shape[0]

        # Recompute via fused GEMM
        gate_up_output = group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_output = gate_activation * up_output
        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)
        gated_weighted = gated_output * expert_scores.unsqueeze(-1)

        # dgrad FC2
        grad_gated_weighted = group_gemm_same_nk(
            a=grad_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )

        # wgrad FC2
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_weighted,
                b=grad_output,
                c=grad_down_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        grad_expert_scores = None
        if ctx.has_expert_scores:
            grad_expert_scores = (grad_gated_weighted * gated_output).sum(dim=-1).to(expert_scores_dtype)
        del gated_output, gated_weighted

        grad_gated_output = grad_gated_weighted * expert_scores.unsqueeze(-1)
        del grad_gated_weighted

        # Activation backward
        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output, gate_up_output

        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # Fused dgrad FC1: cat grads → single GEMM with gate_up_proj (no .contiguous() copies)
        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output
        grad_permute_tokens = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )

        # Fused wgrad FC1: single GEMM produces grad_gate_up_proj directly
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            group_gemm_same_mn(
                a=permute_tokens,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_gate_up_act

        return (
            grad_permute_tokens,
            None,  # cumsum
            grad_gate_up_proj,
            grad_down_proj,
            None,  # intermediate_size
            grad_expert_scores,
        )


class TritonMoeExpertsFunction(torch.autograd.Function):
    """MoE expert computation with custom autograd for efficient backward pass.

    Memory-optimized: uses separate gate/up GEMMs, recomputes cheap
    intermediates in backward, and uses explicit `del` + in-place add
    to free dead tensors immediately.
    """

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_weight=None,  # Ignored (kept for API compat)
    ):
        # Token dispatch: compute histogram, scatter index, and scatter tokens
        splits = expert_histogram(expert_index, num_experts)
        cumsum_t = torch.cumsum(splits, dim=0)
        scatter_index = moe_index_compute(expert_index, cumsum_t)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        max_M = scatter_output.shape[0]

        # Separate gate and up GEMMs (avoids allocating concatenated weight tensor)
        gate_output = group_gemm_same_nk(
            a=scatter_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        up_output = group_gemm_same_nk(
            a=scatter_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        del scatter_output

        # SiLU activation + element-wise multiply (bf16 like native backend)
        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        del gate_activation

        # Apply routing weights in scattered layout
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        gated_weighted = gated_activation * scattered_gate_weight
        del gated_activation

        # Down projection
        down_output = group_gemm_same_nk(
            a=gated_weighted,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        del gated_weighted

        # Gather and reshape
        output = moe_gather(down_output, scatter_index).reshape(hidden_states.shape)
        del down_output

        # Save gate_output + up_output for backward (cheap intermediates like
        # scatter_output, gate_activation, gated_weighted are recomputed instead).
        ctx.save_for_backward(
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            gate_output,
            up_output,
            scattered_gate_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]

        # Recompute cheap intermediates (avoids saving them)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        gated_weighted = gated_activation * scattered_gate_weight

        # Scatter grad to expert-sorted layout
        grad_down_output = moe_scatter(grad_output, scatter_index)

        # FC2 dgrad: grad @ down_proj^T
        grad_gated_weighted = group_gemm_same_nk(
            a=grad_down_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        # FC2 wgrad: gated_weighted^T @ grad
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_down_output, gated_weighted

        # Routing weight gradient
        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        del gated_activation, grad_gated_weighted

        # Activation backward (separate ops, matching TritonEPGroupGemm pattern)
        grad_up_output = gate_activation * grad_gated_activation
        grad_gate_activation = grad_gated_activation * up_output
        del grad_gated_activation, gate_activation, up_output
        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # FC1 dgrad: separate GEMMs, in-place add (avoids allocating sum tensor)
        grad_scatter_output = group_gemm_same_nk(
            a=grad_gate_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )
        grad_scatter_output += group_gemm_same_nk(
            a=grad_up_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        # FC1 wgrad: separate GEMMs (avoids concatenated grad weight alloc + .contiguous() copies)
        grad_gate_proj = None
        if gate_proj.requires_grad:
            grad_gate_proj = torch.empty_like(gate_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_output,
                c=grad_gate_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_gate_output
        grad_up_proj = None
        if up_proj.requires_grad:
            grad_up_proj = torch.empty_like(up_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_up_output,
                c=grad_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_up_output, scatter_output

        # Gather gradient for hidden_states
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_gate_proj,  # gate_proj
            grad_up_proj,  # up_proj
            grad_down_proj,  # down_proj
            None,  # gate_up_weight
        )


def triton_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_weight: torch.Tensor = None,
    **kwargs,
):
    """Forward pass for MoE experts using Triton group GEMM (local, single-GPU).

    EP is handled centrally by ``MoEExperts._ep_forward()``.

    Args:
        module: The parent module (unused, for compatibility).
        num_experts: Total number of experts.
        routing_weights: Routing weights from the router, shape [num_tokens, topk].
        selected_experts: Expert indices from the router, shape [num_tokens, topk].
        hidden_states: Input hidden states, shape [num_tokens, hidden_dim].
        gate_proj: Gate projection weights, shape [num_experts, hidden_dim, intermediate_size].
        up_proj: Up projection weights, shape [num_experts, hidden_dim, intermediate_size].
        down_proj: Down projection weights, shape [num_experts, intermediate_size, hidden_dim].
        gate_up_weight: Optional pre-concatenated weights [num_experts, hidden_dim, 2*intermediate_size].

    Returns:
        Output hidden states, shape [num_tokens, hidden_dim].
    """
    return TritonMoeExpertsFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_weight,
    )


class TritonMoeExpertsFunctionMoeAct(torch.autograd.Function):
    """Local MoE expert computation with moe_act: drops gate_output/up_output
    from save_for_backward and recomputes them in backward."""

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_weight=None,
    ):
        splits = expert_histogram(expert_index, num_experts)
        cumsum_t = torch.cumsum(splits, dim=0)
        scatter_index = moe_index_compute(expert_index, cumsum_t)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        max_M = scatter_output.shape[0]

        gate_output = group_gemm_same_nk(
            a=scatter_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        up_output = group_gemm_same_nk(
            a=scatter_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        del scatter_output

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        del gate_activation

        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        gated_weighted = gated_activation * scattered_gate_weight
        del gated_activation

        down_output = group_gemm_same_nk(
            a=gated_weighted,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        del gated_weighted

        output = moe_gather(down_output, scatter_index).reshape(hidden_states.shape)
        del down_output

        # moe_act: save 8 tensors (drop gate_output, up_output vs 10 in standard)
        ctx.save_for_backward(
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            scattered_gate_weight,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            gate_proj,
            up_proj,
            down_proj,
            hidden_states,
            scatter_index,
            cumsum_t,
            scattered_gate_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]

        # Recompute scatter_output, gate_output, up_output from saved inputs + weights
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_output = group_gemm_same_nk(
            a=scatter_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        up_output = group_gemm_same_nk(
            a=scatter_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )

        gate_activation = torch.ops.aten.silu(gate_output)
        gated_activation = gate_activation * up_output
        gated_weighted = gated_activation * scattered_gate_weight

        # Scatter grad to expert-sorted layout
        grad_down_output = moe_scatter(grad_output, scatter_index)

        # FC2 dgrad
        grad_gated_weighted = group_gemm_same_nk(
            a=grad_down_output,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        # FC2 wgrad
        grad_down_proj = None
        if down_proj.requires_grad:
            grad_down_proj = torch.empty_like(down_proj)
            group_gemm_same_mn(
                a=gated_weighted,
                b=grad_down_output,
                c=grad_down_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_down_output, gated_weighted

        # Routing weight gradient
        grad_gated_activation = grad_gated_weighted * scattered_gate_weight
        grad_gate_weight = torch.sum(gated_activation * grad_gated_weighted, dim=-1)[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        del gated_activation, grad_gated_weighted

        # Activation backward
        grad_up_output = gate_activation * grad_gated_activation
        grad_gate_activation = grad_gated_activation * up_output
        del grad_gated_activation, gate_activation, up_output
        grad_gate_output = torch.ops.aten.silu_backward(grad_gate_activation, gate_output)
        del grad_gate_activation, gate_output

        # FC1 dgrad
        grad_scatter_output = group_gemm_same_nk(
            a=grad_gate_output,
            b=gate_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )
        grad_scatter_output += group_gemm_same_nk(
            a=grad_up_output,
            b=up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )

        # FC1 wgrad
        grad_gate_proj = None
        if gate_proj.requires_grad:
            grad_gate_proj = torch.empty_like(gate_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_output,
                c=grad_gate_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_gate_output
        grad_up_proj = None
        if up_proj.requires_grad:
            grad_up_proj = torch.empty_like(up_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_up_output,
                c=grad_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
                transpose_b=False,
            )
        del grad_up_output, scatter_output

        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index).reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,
            None,  # expert_index
            grad_hidden_states,
            grad_gate_proj,
            grad_up_proj,
            grad_down_proj,
            None,  # gate_up_weight
        )


def triton_moe_forward_moe_act(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_weight: torch.Tensor = None,
    **kwargs,
):
    """Forward pass for MoE experts using Triton group GEMM with moe_act
    (activation recompute, no EP recompute)."""
    return TritonMoeExpertsFunctionMoeAct.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_weight,
    )
