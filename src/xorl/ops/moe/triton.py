import torch
import torch.nn.functional as F

from xorl.utils.import_utils import is_fused_moe_available


if is_fused_moe_available():
    from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
    from xorl.ops.group_gemm.kernel.moe import (
        expert_histogram,
        moe_index_compute,
        moe_scatter,
    )


# Canonical activation kinds understood by MoE ops. Upstream `hidden_act`
# strings (e.g. "gelu_pytorch_tanh") are normalized to one of these.
SUPPORTED_HIDDEN_ACTS: frozenset[str] = frozenset({"silu", "gelu_tanh"})


def normalize_hidden_act(hidden_act: str | None) -> str:
    """Normalize a HF-style ``hidden_act`` string to an MoE act kind."""
    if hidden_act is None or hidden_act == "silu":
        return "silu"
    if hidden_act in ("gelu_tanh", "gelu_pytorch_tanh"):
        return "gelu_tanh"
    raise ValueError(f"Unsupported hidden_act={hidden_act!r}. Supported: {sorted(SUPPORTED_HIDDEN_ACTS)}")


def check_hidden_act_supported(hidden_act: str, backend: str, supported: frozenset[str]) -> None:
    """Raise if ``hidden_act`` is not in the backend's supported set."""
    if hidden_act not in supported:
        raise ValueError(
            f"MoE backend {backend!r} does not support hidden_act={hidden_act!r}. Supported: {sorted(supported)}"
        )


def _moe_gate_activation(gate_output: torch.Tensor, hidden_act: str = "silu") -> torch.Tensor:
    """Apply gate activation by kind."""
    if hidden_act == "gelu_tanh":
        return F.gelu(gate_output, approximate="tanh")
    return torch.ops.aten.silu(gate_output)


def _moe_gate_activation_backward(
    grad: torch.Tensor, gate_output: torch.Tensor, hidden_act: str = "silu"
) -> torch.Tensor:
    """Backward for gate activation."""
    if hidden_act == "gelu_tanh":
        with torch.enable_grad():
            g = gate_output.detach().requires_grad_(True)
            a = F.gelu(g, approximate="tanh")
        return torch.autograd.grad(a, g, grad)[0]
    return torch.ops.aten.silu_backward(grad, gate_output)


class TritonEPGroupGemm(torch.autograd.Function):
    """EP expert MLP with fused gate+up GEMM. Zero-copy weight references.

    Forward: single ``x @ gate_up_proj`` GEMM → split → GLU activation → down GEMM.
    Backward: fused dgrad/wgrad for gate+up (2x fewer GEMMs than split version).
    """

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh"})

    @staticmethod
    def forward(
        ctx, permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores=None, hidden_act="silu"
    ):
        check_hidden_act_supported(hidden_act, "triton", TritonEPGroupGemm.SUPPORTED_HIDDEN_ACTS)
        max_M = permute_tokens.shape[0]
        I = intermediate_size
        ctx.has_expert_scores = expert_scores is not None
        ctx.hidden_act = hidden_act

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

        gate_activation = _moe_gate_activation(gate_output, getattr(ctx, "hidden_act", "silu"))
        gated_output = gate_activation * up_output
        del gate_activation

        # Down projection (NO expert_scores inside GEMM — apply after)
        down_output = group_gemm_same_nk(
            a=gated_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
        )
        del gated_output

        if expert_scores is not None:
            down_output = down_output * expert_scores.to(down_output.dtype).unsqueeze(-1)

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

        gate_activation = _moe_gate_activation(gate_output, getattr(ctx, "hidden_act", "silu"))
        gated_output = gate_activation * up_output
        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)

        # Forward was: out = down_GEMM(gated_output) * expert_scores
        grad_expert_scores = None
        if ctx.has_expert_scores:
            down_output = group_gemm_same_nk(a=gated_output, b=down_proj, cumsum_M=cumsum, max_M=max_M)
            grad_expert_scores = (down_output * grad_output).sum(dim=-1).to(expert_scores_dtype)
            del down_output

        grad_scaled = grad_output * expert_scores.unsqueeze(-1)

        # dgrad FC2
        grad_gated_output = group_gemm_same_nk(
            a=grad_scaled,
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
                a=gated_output,
                b=grad_scaled,
                c=grad_down_proj,
                cumsum_K=cumsum,
                max_K=max_M,
                transpose_a=True,
            )
        del gated_output, grad_scaled

        # Activation backward
        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output, gate_up_output

        grad_gate_output = _moe_gate_activation_backward(
            grad_gate_activation, gate_output, getattr(ctx, "hidden_act", "silu")
        )
        del grad_gate_activation, gate_output

        # Fused dgrad FC1
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
            None,  # hidden_act
        )


class TritonMoeExpertsFunction(torch.autograd.Function):
    """MoE expert computation with custom autograd for efficient backward pass.

    Memory-optimized: uses separate gate/up GEMMs, recomputes cheap
    intermediates in backward, and uses explicit `del` + in-place add
    to free dead tensors immediately.
    """

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh"})

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
        gate_up_proj=None,
        hidden_act="silu",
    ):
        check_hidden_act_supported(hidden_act, "triton", TritonMoeExpertsFunction.SUPPORTED_HIDDEN_ACTS)
        ctx.hidden_act = hidden_act
        num_tokens = hidden_states.shape[0]
        top_k = expert_index.shape[1]

        # Token dispatch: sort by expert
        splits = expert_histogram(expert_index, num_experts)
        cumsum_t = torch.cumsum(splits, dim=0)
        scatter_index = moe_index_compute(expert_index, cumsum_t)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        max_M = scatter_output.shape[0]

        assert gate_up_proj is not None, "TritonMoeExpertsFunction requires a fused gate_up_proj"
        gate_up_output = group_gemm_same_nk(
            a=scatter_output,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
        )
        I = gate_up_output.shape[-1] // 2
        gate_output = gate_up_output[..., :I]
        up_output = gate_up_output[..., I:]

        # Activation + GLU
        gate_activation = _moe_gate_activation(gate_output, getattr(ctx, "hidden_act", "silu"))
        gated_activation = gate_activation * up_output
        del gate_activation

        # Down projection (NO routing weights inside GEMM — apply after)
        down_output = group_gemm_same_nk(
            a=gated_activation,
            b=down_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
        )
        del gated_activation

        # Unsort, apply routing weights, reshape+sum (deterministic accumulation)
        per_slot = down_output[scatter_index.flatten()].reshape(num_tokens, top_k, -1)
        output = (per_slot * gate_weights.unsqueeze(-1)).sum(dim=1)
        del down_output, per_slot

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
            gate_up_proj,
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
            gate_up_proj,
        ) = ctx.saved_tensors
        # Recompute scattered routing weights for backward
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        max_M = grad_output.shape[0]

        # Recompute cheap intermediates
        scatter_output = moe_scatter(hidden_states, scatter_index)
        gate_activation = _moe_gate_activation(gate_output, getattr(ctx, "hidden_act", "silu"))
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
        grad_gate_output = _moe_gate_activation_backward(
            grad_gate_activation, gate_output, getattr(ctx, "hidden_act", "silu")
        )
        del grad_gate_activation, gate_output

        # FC1 dgrad + wgrad — fused via gate_up_proj
        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output
        grad_scatter_output = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
        )
        grad_gate_up_proj = None
        if gate_up_proj.requires_grad:
            grad_gate_up_proj = torch.empty_like(gate_up_proj)
            group_gemm_same_mn(
                a=scatter_output,
                b=grad_gate_up_act,
                c=grad_gate_up_proj,
                cumsum_K=cumsum_t,
                max_K=max_M,
                transpose_a=True,
            )
        del grad_gate_up_act, scatter_output

        # Unsort grad + reshape+sum (deterministic, matching forward)
        grad_hidden_states = (
            grad_scatter_output[scatter_index.flatten()]
            .reshape(hidden_states.shape[0], scatter_index.shape[1], -1)
            .sum(dim=1)
        )

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            None,  # gate_proj (unused — fused into gate_up_proj)
            None,  # up_proj   (unused — fused into gate_up_proj)
            grad_down_proj,  # down_proj
            grad_gate_up_proj,  # gate_up_proj
            None,  # hidden_act
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
    gate_up_proj: torch.Tensor = None,
    hidden_act: str = "silu",
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
        gate_up_proj: Pre-fused weights [num_experts, hidden_dim, 2*intermediate_size].
        hidden_act: Activation kind ("silu" or "gelu_tanh").

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
        gate_up_proj,
        hidden_act,
    )
