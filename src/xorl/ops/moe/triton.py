import os

import torch
import torch.nn.functional as F

# Re-export canonical activation utilities so callers that historically
# imported from ``xorl.ops.moe.triton`` keep working.
from xorl.ops.moe.activations import (  # noqa: F401
    CLAMPED_SWIGLU_ALPHA,
    CLAMPED_SWIGLU_LIMIT,
    SUPPORTED_HIDDEN_ACTS,
    UNGATED_HIDDEN_ACTS,
    check_hidden_act_supported,
    normalize_hidden_act,
)
from xorl.utils.import_utils import is_fused_moe_available


if is_fused_moe_available():
    from xorl.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
    from xorl.ops.group_gemm.kernel.moe import (
        expert_histogram,
        moe_index_compute,
        moe_scatter,
    )


# Fold expert_scores into the down-GEMM input (before-down) rather than scaling the
# down-GEMM output (after-down, the historical default). Mathematically identical (a
# per-row scalar commutes through the linear down projection) but a different bf16
# rounding point, and a cheaper backward when router gradients are needed: after-down
# recomputes a full down GEMM just for grad_expert_scores and materializes [M, H]
# transients, while before-down harvests grad_expert_scores from the FC2 dgrad with
# only an [M, I] extra. Off by default; K3/parity lanes must keep it off (it changes
# the trainer's reduction tree). Covered by test_moe_routing_weight_position.py.
_ROUTING_WEIGHTS_BEFORE_DOWN_CONFIG = False


def set_routing_weights_before_down(enabled: bool) -> None:
    """Set the config-driven routing-weight position (called at model init)."""
    global _ROUTING_WEIGHTS_BEFORE_DOWN_CONFIG
    _ROUTING_WEIGHTS_BEFORE_DOWN_CONFIG = bool(enabled)


def routing_weights_before_down() -> bool:
    """Whether expert_scores fold into the down-GEMM input instead of its output.

    The ``XORL_MOE_ROUTING_WEIGHTS_BEFORE_DOWN=1`` env var force-enables the
    before-down position regardless of config; it is read lazily (per forward)
    so it keeps working when set after import.
    """
    return _ROUTING_WEIGHTS_BEFORE_DOWN_CONFIG or os.environ.get("XORL_MOE_ROUTING_WEIGHTS_BEFORE_DOWN", "0") == "1"


def resolve_routing_weights_before_down(setting: bool | str, *, train_router: bool, ep_dispatch: str) -> bool:
    """Resolve the ``moe_routing_weights_before_down`` knob (``auto``/true/false) to a position.

    Explicit true/false overrides are honored as-is. ``auto`` (the default) enables
    the before-down position only in the regime where it is a measured win — router
    gradients needed (``train_router=True``) with alltoall EP dispatch — and never
    while the ``XORL_MOE_SGLANG_FUSED_EXPERTS`` parity opt-in is active, so
    K3/parity lanes stay anchored on the historical after-down reduction tree.
    """
    if isinstance(setting, bool):
        return setting
    normalized = str(setting).strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    if normalized != "auto":
        raise ValueError(f"Invalid moe_routing_weights_before_down={setting!r}; expected 'auto', true, or false.")
    from xorl.models.layers.moe.experts import moe_sglang_fused_experts_enabled  # noqa: PLC0415

    return train_router and ep_dispatch == "alltoall" and not moe_sglang_fused_experts_enabled()


def _maybe_fake_quant_down_input(tensor: torch.Tensor, act_quant_group_size: int) -> torch.Tensor:
    """STE-fake-quantize the down-GEMM input (the SwiGLU intermediate) for W4A4.

    No-op when ``act_quant_group_size`` is 0 (the default bf16 down input). Used by
    both ``TritonEPGroupGemm.forward`` and its backward so the down_proj weight
    gradient is taken against the exact same quantized intermediate that forward fed
    to the down GEMM, under either routing order.
    """
    if not act_quant_group_size:
        return tensor
    from xorl.ops.quantize.nvfp4_fake_quant import fake_quantize_activation_nvfp4  # noqa: PLC0415

    return fake_quantize_activation_nvfp4(tensor, act_quant_group_size)


def _moe_gate_activation(gate_output: torch.Tensor, hidden_act: str = "silu") -> torch.Tensor:
    """Apply gate activation by kind."""
    if hidden_act == "gelu_tanh":
        return F.gelu(gate_output, approximate="tanh")
    if hidden_act == "clamped_swiglu":
        gate_clamped = gate_output.clamp(max=CLAMPED_SWIGLU_LIMIT)
        return gate_clamped * torch.sigmoid(CLAMPED_SWIGLU_ALPHA * gate_clamped)
    if hidden_act == "relu2":
        # Squared ReLU: relu2(x) == relu(x) * x, so the gated path with
        # gate ≡ up reduces the gate activation to a plain relu.
        return F.relu(gate_output)
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
    if hidden_act == "clamped_swiglu":
        with torch.enable_grad():
            g = gate_output.detach().requires_grad_(True)
            g_clamped = g.clamp(max=CLAMPED_SWIGLU_LIMIT)
            a = g_clamped * torch.sigmoid(CLAMPED_SWIGLU_ALPHA * g_clamped)
        return torch.autograd.grad(a, g, grad)[0]
    if hidden_act == "relu2":
        return torch.ops.aten.threshold_backward(grad, gate_output, 0)
    return torch.ops.aten.silu_backward(grad, gate_output)


def _maybe_clamp_swiglu_gate(gate_output: torch.Tensor, swiglu_limit: float = 0.0) -> torch.Tensor:
    """Clamp the gate pre-activation for DeepSeek-V4's SwiGLU stability limit."""
    if swiglu_limit > 0:
        return gate_output.clamp(-swiglu_limit, swiglu_limit)
    return gate_output


def _apply_swiglu_clamp_backward(
    grad_gate_output: torch.Tensor, unclamped_gate_output: torch.Tensor, swiglu_limit: float = 0.0
) -> torch.Tensor:
    """Apply the derivative of ``clamp(-limit, limit)`` to gate gradients."""
    if swiglu_limit <= 0:
        return grad_gate_output
    mask = (unclamped_gate_output >= -swiglu_limit) & (unclamped_gate_output <= swiglu_limit)
    return grad_gate_output * mask.to(grad_gate_output.dtype)


class TritonEPGroupGemm(torch.autograd.Function):
    """EP expert MLP with fused gate+up GEMM. Zero-copy weight references.

    Forward: single ``x @ gate_up_proj`` GEMM → split → GLU activation → down GEMM.
    Backward: fused dgrad/wgrad for gate+up (2x fewer GEMMs than split version).

    With ``gated=False`` the first weight is a plain up projection
    ``[E, in_dim, I]``: the GEMM output is not split and serves as both gate
    and up branch (e.g. ``relu2(x) == relu(x) * x``).
    """

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh"})  # noqa: F811

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        gate_up_proj,
        down_proj,
        intermediate_size,
        expert_scores=None,
        hidden_act="silu",
        swiglu_limit=0.0,
        gated=True,
        act_quant_group_size=0,
    ):
        if gated:
            check_hidden_act_supported(hidden_act, "triton", TritonEPGroupGemm.SUPPORTED_HIDDEN_ACTS)
        else:
            check_hidden_act_supported(hidden_act, "triton (non-gated)", UNGATED_HIDDEN_ACTS)
        max_M = permute_tokens.shape[0]
        I = intermediate_size
        ctx.has_expert_scores = expert_scores is not None
        ctx.hidden_act = hidden_act
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        ctx.gated = gated
        ctx.act_quant_group_size = int(act_quant_group_size or 0)

        gate_up_output = group_gemm_same_nk(
            a=permute_tokens,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_a=False,
            transpose_b=False,
        )
        if gated:
            gate_output = gate_up_output[..., :I]
            up_output = gate_up_output[..., I:]
        else:
            gate_output = up_output = gate_up_output

        before_down = routing_weights_before_down()
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
        gate_activation = _moe_gate_activation(gate_for_activation, getattr(ctx, "hidden_act", "silu"))
        gated_output = gate_activation * up_output
        if before_down and expert_scores is not None:
            gated_output.mul_(expert_scores.to(gated_output.dtype).unsqueeze(-1))
        del gate_activation, gate_for_activation

        # W4A4: fake-quantize the down-GEMM input (the SwiGLU intermediate) with NVFP4
        # STE. Sits right before the down GEMM so it covers the exact down input under
        # both routing orders (expert_scores already folded in above when routing-before-
        # down). The backward treats the quant as identity for the upstream activation
        # gradient and re-quantizes the recomputed intermediate for the down_proj weight
        # gradient. Default off (act_quant_group_size=0) -> bf16 down input.
        gated_output = _maybe_fake_quant_down_input(gated_output, ctx.act_quant_group_size)

        # Down projection (NO expert_scores inside GEMM — apply after)
        down_output = group_gemm_same_nk(
            a=gated_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
        )
        del gated_output

        if expert_scores is not None and not before_down:
            down_output.mul_(expert_scores.to(down_output.dtype).unsqueeze(-1))

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(permute_tokens, cumsum, gate_up_proj, down_proj, gate_up_output, expert_scores)
        ctx.intermediate_size = I
        ctx.routing_weights_before_down = before_down

        return down_output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, cumsum, gate_up_proj, down_proj, gate_up_output, expert_scores = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = grad_output.shape[0]

        if ctx.gated:
            gate_output = gate_up_output[..., :I]
            up_output = gate_up_output[..., I:]
        else:
            gate_output = up_output = gate_up_output

        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, getattr(ctx, "swiglu_limit", 0.0))
        gate_activation = _moe_gate_activation(gate_for_activation, getattr(ctx, "hidden_act", "silu"))
        gated_output = gate_activation * up_output
        # W4A4: the forward fed quant(<down-GEMM input>) to the down GEMM, so the down_proj
        # weight gradient must be taken against the SAME quantized intermediate (true STE
        # wgrad). The exact down input differs by routing order — quant(gated_output) when
        # routing-after-down (else branch), quant(gated_output * expert_scores) when
        # routing-before-down — so re-quantize per branch below. The upstream activation
        # gradient (grad_gated_output) flows as identity (no quant). No-op when act-quant off.
        act_quant_gs = getattr(ctx, "act_quant_group_size", 0)
        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)
        routing_weights_before_down = getattr(ctx, "routing_weights_before_down", False)

        grad_expert_scores = None
        grad_down_proj = None
        if routing_weights_before_down:
            if ctx.has_expert_scores and ctx.needs_input_grad[5]:
                gated_for_down = gated_output * expert_scores.unsqueeze(-1)
            elif ctx.has_expert_scores:
                gated_output.mul_(expert_scores.unsqueeze(-1))
                gated_for_down = gated_output
            else:
                gated_for_down = gated_output

            grad_gated_for_down = group_gemm_same_nk(
                a=grad_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

            if down_proj.requires_grad:
                grad_down_proj = torch.empty_like(down_proj)
                # Forward fed quant(gated_output * expert_scores) to the down GEMM, so the
                # wgrad is taken against the quantized score-folded intermediate (STE wgrad).
                gated_for_down_wgrad = _maybe_fake_quant_down_input(gated_for_down, act_quant_gs)
                group_gemm_same_mn(
                    a=gated_for_down_wgrad,
                    b=grad_output,
                    c=grad_down_proj,
                    cumsum_K=cumsum,
                    max_K=max_M,
                    transpose_a=True,
                )
                if gated_for_down_wgrad is not gated_for_down:
                    del gated_for_down_wgrad
            if gated_for_down is not gated_output:
                del gated_for_down

            if ctx.has_expert_scores and ctx.needs_input_grad[5]:
                grad_expert_scores = (grad_gated_for_down * gated_output).sum(dim=-1).to(expert_scores_dtype)
            del gated_output

            grad_gated_output = (
                grad_gated_for_down * expert_scores.unsqueeze(-1) if ctx.has_expert_scores else grad_gated_for_down
            )
            del grad_gated_for_down
        else:
            # Forward was: out = down_GEMM(quant(gated_output)) * expert_scores.
            # Re-quantize the (un-scaled) intermediate the down GEMM consumed (STE wgrad).
            gated_output_for_wgrad = _maybe_fake_quant_down_input(gated_output, act_quant_gs)
            # Skip the extra down-GEMM when expert_scores doesn't require a gradient.
            if ctx.has_expert_scores and ctx.needs_input_grad[5]:
                down_output = group_gemm_same_nk(a=gated_output_for_wgrad, b=down_proj, cumsum_M=cumsum, max_M=max_M)
                grad_expert_scores = (down_output * grad_output).sum(dim=-1).to(expert_scores_dtype)
                del down_output

            grad_scaled = grad_output * expert_scores.unsqueeze(-1)

            grad_gated_output = group_gemm_same_nk(
                a=grad_scaled,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

            if down_proj.requires_grad:
                grad_down_proj = torch.empty_like(down_proj)
                group_gemm_same_mn(
                    a=gated_output_for_wgrad,
                    b=grad_scaled,
                    c=grad_down_proj,
                    cumsum_K=cumsum,
                    max_K=max_M,
                    transpose_a=True,
                )
            if gated_output_for_wgrad is not gated_output:
                del gated_output_for_wgrad
            del gated_output, grad_scaled

        # Activation backward
        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output, gate_up_output

        grad_gate_output = _moe_gate_activation_backward(
            grad_gate_activation, gate_for_activation, getattr(ctx, "hidden_act", "silu")
        )
        grad_gate_output = _apply_swiglu_clamp_backward(
            grad_gate_output, gate_output, getattr(ctx, "swiglu_limit", 0.0)
        )
        del grad_gate_activation, gate_output, gate_for_activation

        # Fused dgrad FC1. Non-gated: gate and up are the same GEMM output,
        # so the two branch gradients sum instead of concatenating.
        if ctx.gated:
            grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        else:
            grad_gate_up_act = grad_gate_output.add_(grad_up_output)
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
            None,  # swiglu_limit
            None,  # gated
            None,  # act_quant_group_size
        )


class TritonEPGroupGemmMoeAct(torch.autograd.Function):
    """EP expert GEMM with moe_act: drop gate/up activations and recompute them in backward."""

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh"})  # noqa: F811

    @staticmethod
    def forward(
        ctx, permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores=None, hidden_act="silu"
    ):
        check_hidden_act_supported(hidden_act, "triton", TritonEPGroupGemmMoeAct.SUPPORTED_HIDDEN_ACTS)
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
        before_down = routing_weights_before_down()
        gate_activation = _moe_gate_activation(gate_up_output[..., :I], getattr(ctx, "hidden_act", "silu"))
        gated_output = gate_activation * gate_up_output[..., I:]
        if before_down and expert_scores is not None:
            gated_output.mul_(expert_scores.to(gated_output.dtype).unsqueeze(-1))
        del gate_activation, gate_up_output

        down_output = group_gemm_same_nk(
            a=gated_output,
            b=down_proj,
            cumsum_M=cumsum,
            max_M=max_M,
        )
        del gated_output

        if expert_scores is not None and not before_down:
            down_output.mul_(expert_scores.to(down_output.dtype).unsqueeze(-1))

        if expert_scores is None:
            expert_scores = permute_tokens.new_ones(permute_tokens.shape[0])
        ctx.save_for_backward(permute_tokens, cumsum, gate_up_proj, down_proj, expert_scores)
        ctx.intermediate_size = I
        ctx.routing_weights_before_down = before_down
        return down_output

    @staticmethod
    def backward(ctx, grad_output):
        permute_tokens, cumsum, gate_up_proj, down_proj, expert_scores = ctx.saved_tensors
        I = ctx.intermediate_size
        max_M = grad_output.shape[0]

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

        expert_scores_dtype = expert_scores.dtype
        expert_scores = expert_scores.to(gated_output.dtype)
        routing_weights_before_down = getattr(ctx, "routing_weights_before_down", False)

        grad_expert_scores = None
        if routing_weights_before_down:
            if ctx.has_expert_scores and ctx.needs_input_grad[5]:
                gated_for_down = gated_output * expert_scores.unsqueeze(-1)
            elif ctx.has_expert_scores:
                gated_output.mul_(expert_scores.unsqueeze(-1))
                gated_for_down = gated_output
            else:
                gated_for_down = gated_output

            grad_gated_for_down = group_gemm_same_nk(
                a=grad_output,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

            grad_down_proj = None
            if down_proj.requires_grad:
                grad_down_proj = torch.empty_like(down_proj)
                group_gemm_same_mn(
                    a=gated_for_down,
                    b=grad_output,
                    c=grad_down_proj,
                    cumsum_K=cumsum,
                    max_K=max_M,
                    transpose_a=True,
                )
            if gated_for_down is not gated_output:
                del gated_for_down

            if ctx.has_expert_scores and ctx.needs_input_grad[5]:
                grad_expert_scores = (grad_gated_for_down * gated_output).sum(dim=-1).to(expert_scores_dtype)
            del gated_output

            grad_gated_output = (
                grad_gated_for_down * expert_scores.unsqueeze(-1) if ctx.has_expert_scores else grad_gated_for_down
            )
            del grad_gated_for_down
        else:
            if ctx.has_expert_scores and ctx.needs_input_grad[5]:
                down_output = group_gemm_same_nk(a=gated_output, b=down_proj, cumsum_M=cumsum, max_M=max_M)
                grad_expert_scores = (down_output * grad_output).sum(dim=-1).to(expert_scores_dtype)
                del down_output

            grad_scaled = grad_output * expert_scores.unsqueeze(-1)

            grad_gated_output = group_gemm_same_nk(
                a=grad_scaled,
                b=down_proj,
                cumsum_M=cumsum,
                max_M=max_M,
                transpose_b=True,
            )

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

        grad_up_output = gate_activation * grad_gated_output
        grad_gate_activation = grad_gated_output * up_output
        del grad_gated_output, gate_activation, up_output, gate_up_output

        grad_gate_output = _moe_gate_activation_backward(
            grad_gate_activation, gate_output, getattr(ctx, "hidden_act", "silu")
        )
        del grad_gate_activation, gate_output

        grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        del grad_gate_output, grad_up_output

        # Fused wgrad FC1 first: allocate the large parameter gradient before
        # materializing the smaller input gradient to keep the peak lower.
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

        # Fused dgrad FC1
        grad_permute_tokens = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum,
            max_M=max_M,
            transpose_b=True,
        )
        del grad_gate_up_act

        return (
            grad_permute_tokens,
            None,
            grad_gate_up_proj,
            grad_down_proj,
            None,
            grad_expert_scores,
            None,
        )


class TritonMoeExpertsFunction(torch.autograd.Function):
    """MoE expert computation with custom autograd for efficient backward pass.

    Memory-optimized: uses separate gate/up GEMMs, recomputes cheap
    intermediates in backward, and uses explicit `del` + in-place add
    to free dead tensors immediately.
    """

    SUPPORTED_HIDDEN_ACTS = frozenset({"silu", "gelu_tanh"})  # noqa: F811

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
        swiglu_limit=0.0,
        gated=True,
    ):
        if gated:
            check_hidden_act_supported(hidden_act, "triton", TritonMoeExpertsFunction.SUPPORTED_HIDDEN_ACTS)
        else:
            check_hidden_act_supported(hidden_act, "triton (non-gated)", UNGATED_HIDDEN_ACTS)
        ctx.hidden_act = hidden_act
        ctx.swiglu_limit = float(swiglu_limit or 0.0)
        ctx.gated = gated
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
        if gated:
            I = gate_up_output.shape[-1] // 2
            gate_output = gate_up_output[..., :I]
            up_output = gate_up_output[..., I:]
        else:
            gate_output = up_output = gate_up_output

        # Activation + GLU
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, ctx.swiglu_limit)
        gate_activation = _moe_gate_activation(gate_for_activation, getattr(ctx, "hidden_act", "silu"))
        gated_activation = gate_activation * up_output
        del gate_activation, gate_for_activation

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
        gate_for_activation = _maybe_clamp_swiglu_gate(gate_output, getattr(ctx, "swiglu_limit", 0.0))
        gate_activation = _moe_gate_activation(gate_for_activation, getattr(ctx, "hidden_act", "silu"))
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
            grad_gate_activation, gate_for_activation, getattr(ctx, "hidden_act", "silu")
        )
        grad_gate_output = _apply_swiglu_clamp_backward(
            grad_gate_output, gate_output, getattr(ctx, "swiglu_limit", 0.0)
        )
        del grad_gate_activation, gate_output, gate_for_activation

        # FC1 dgrad + wgrad — fused via gate_up_proj. Non-gated: gate and up
        # are the same GEMM output, so the branch gradients sum.
        if ctx.gated:
            grad_gate_up_act = torch.cat([grad_gate_output, grad_up_output], dim=-1)
        else:
            grad_gate_up_act = grad_gate_output.add_(grad_up_output)
        del grad_gate_output, grad_up_output

        # FC1 wgrad first so the large parameter gradient is allocated before
        # the input-gradient output extends the live tensor set.
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

        # FC1 dgrad
        grad_scatter_output = group_gemm_same_nk(
            a=grad_gate_up_act,
            b=gate_up_proj,
            cumsum_M=cumsum_t,
            max_M=max_M,
            transpose_b=True,
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
            None,  # swiglu_limit
            None,  # gated
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
    swiglu_limit: float = 0.0,
    gated: bool = True,
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
        gate_up_proj: Pre-fused weights [num_experts, hidden_dim, 2*intermediate_size]
            (or the plain up projection [num_experts, in_dim, intermediate_size]
            when ``gated=False``).
        hidden_act: Activation kind ("silu" or "gelu_tanh"; "relu2" when ``gated=False``).
        swiglu_limit: Optional gate pre-activation clamp (DeepSeek-V4 SwiGLU limit).
        gated: Whether the experts use a gated (GLU) first projection.

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
        swiglu_limit,
        gated,
    )
