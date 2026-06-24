"""Eager MoE expert backend — per-expert matmul. For debugging/testing."""

import torch

from xorl.distributed.parallel_state import get_parallel_state
from xorl.models.layers.moe.common import split_gate_up_proj
from xorl.ops.moe.activations import UNGATED_HIDDEN_ACTS, apply_moe_activation, check_hidden_act_supported


def eager_expert_forward(
    hidden_states: torch.Tensor,
    expert_idx: int,
    gate_proj: torch.Tensor | None,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_act: str = "silu",
    swiglu_limit: float = 0.0,
    gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    gated: bool = True,
) -> torch.Tensor:
    """Forward pass for a single expert (eager mode).

    Called in a loop by ``MoEBlock._eager_forward()``.

    Args:
        hidden_states: Input tensor of shape ``(num_tokens, hidden_dim)``.
        expert_idx: Index of the expert to use.
        gate_proj: Gate projection weights ``[num_experts, hidden, intermediate]``
            (``None`` when ``gated=False``).
        up_proj: Up projection weights ``[num_experts, hidden, intermediate]``.
        down_proj: Down projection weights ``[num_experts, intermediate, hidden]``.
        hidden_act: Activation kind (e.g. ``"silu"``, ``"gelu_tanh"``,
            ``"clamped_swiglu"``) — dispatched via ``apply_moe_activation``.
        swiglu_limit: Optional gate pre-activation clamp (DeepSeek-V4 SwiGLU limit).
        gate_up_bias: Optional per-expert bias ``[num_experts, 2*intermediate]``,
            split as ``[gate_bias | up_bias]`` along the last dim
            (``[num_experts, intermediate]`` when ``gated=False``).
        down_bias: Optional per-expert bias ``[num_experts, hidden_dim]``.
        gated: Whether the experts use a gated (GLU) first projection. When
            False, the activation is applied directly to the single up GEMM
            output (passed as both activation arguments — e.g.
            ``relu2(x) == relu(x) * x``).
    """
    assert not get_parallel_state().ep_enabled, "_moe_implementation='eager' does not support EP"
    if not gated:
        check_hidden_act_supported(hidden_act, "eager (non-gated)", UNGATED_HIDDEN_ACTS)
        up_proj_out = torch.matmul(hidden_states, up_proj[expert_idx])
        if gate_up_bias is not None:
            up_proj_out = up_proj_out + gate_up_bias[expert_idx]
        out = apply_moe_activation(hidden_act, up_proj_out, up_proj_out)
    else:
        gate_proj_out = torch.matmul(hidden_states, gate_proj[expert_idx])
        up_proj_out = torch.matmul(hidden_states, up_proj[expert_idx])
        if gate_up_bias is not None:
            intermediate_size = gate_proj_out.shape[-1]
            gate_proj_out = gate_proj_out + gate_up_bias[expert_idx, :intermediate_size]
            up_proj_out = up_proj_out + gate_up_bias[expert_idx, intermediate_size:]
        if swiglu_limit > 0:
            gate_proj_out = gate_proj_out.clamp(-swiglu_limit, swiglu_limit)
        out = apply_moe_activation(hidden_act, gate_proj_out, up_proj_out)
    out = torch.matmul(out, down_proj[expert_idx])
    if down_bias is not None:
        out = out + down_bias[expert_idx]
    return out


def _counts_from_cumsum(cumsum: torch.Tensor, num_experts: int) -> list[int]:
    """Convert inclusive cumsum token counts to per-expert counts."""
    counts = []
    prev = 0
    for end in cumsum.detach().cpu().tolist()[:num_experts]:
        end = int(end)
        counts.append(end - prev)
        prev = end
    if len(counts) < num_experts:
        counts.extend([0] * (num_experts - len(counts)))
    return counts


def eager_ep_compute(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    intermediate_size: int,
    expert_scores: torch.Tensor | None = None,
    hidden_act: str = "silu",
    activation_native: bool = False,
    gate_up_bias: torch.Tensor | None = None,
    down_bias: torch.Tensor | None = None,
    swiglu_limit: float = 0.0,
    gated: bool = True,
    **_extras,
) -> torch.Tensor:
    """EP expert compute using eager per-local-expert matmuls.

    ``permute_tokens`` is already all-to-all dispatched and sorted by local
    expert. ``cumsum`` therefore describes local expert ids ``0..E_local-1``;
    global expert ids must not be used at this stage.

    With ``gated=False``, ``gate_up_proj`` holds the plain up projection
    ``[num_local_experts, in_dim, intermediate_size]`` and the activation is
    applied directly to the single first GEMM output.
    """
    del activation_native
    if permute_tokens.shape[0] == 0:
        return permute_tokens
    if any(v is not None for v in _extras.values()):
        raise NotImplementedError(f"Unsupported eager EP expert extras: {sorted(_extras)}")

    if gated:
        gate_proj, up_proj = split_gate_up_proj(gate_up_proj, intermediate_size)
    else:
        check_hidden_act_supported(hidden_act, "eager EP (non-gated)", UNGATED_HIDDEN_ACTS)
        gate_proj, up_proj = None, gate_up_proj
    num_local_experts = up_proj.shape[0]
    counts = _counts_from_cumsum(cumsum, num_local_experts)

    outputs = []
    start = 0
    for expert_idx, count in enumerate(counts):
        end = start + count
        current = permute_tokens[start:end]
        if count == 0:
            outputs.append(permute_tokens.new_empty((0, down_proj.shape[-1])))
        else:
            if not gated:
                up_proj_out = torch.matmul(current, up_proj[expert_idx])
                if gate_up_bias is not None:
                    up_proj_out = up_proj_out + gate_up_bias[expert_idx]
                activated = apply_moe_activation(hidden_act, up_proj_out, up_proj_out)
            else:
                gate_proj_out = torch.matmul(current, gate_proj[expert_idx])
                up_proj_out = torch.matmul(current, up_proj[expert_idx])
                if gate_up_bias is not None:
                    gate_proj_out = gate_proj_out + gate_up_bias[expert_idx, :intermediate_size]
                    up_proj_out = up_proj_out + gate_up_bias[expert_idx, intermediate_size:]
                if swiglu_limit > 0:
                    gate_proj_out = gate_proj_out.clamp(-swiglu_limit, swiglu_limit)
                activated = apply_moe_activation(hidden_act, gate_proj_out, up_proj_out)
            out = torch.matmul(activated, down_proj[expert_idx])
            if down_bias is not None:
                out = out + down_bias[expert_idx]
            if expert_scores is not None:
                out = out * expert_scores[start:end, None].to(out.dtype)
            outputs.append(out)
        start = end

    return torch.cat(outputs, dim=0) if outputs else permute_tokens.new_empty((0, down_proj.shape[-1]))


def eager_ep_compute_lora(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_proj_lora_A: torch.Tensor,
    gate_proj_lora_B: torch.Tensor,
    up_proj_lora_A: torch.Tensor,
    up_proj_lora_B: torch.Tensor,
    down_proj_lora_A: torch.Tensor,
    down_proj_lora_B: torch.Tensor,
    scaling: float,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """EP expert compute with LoRA using eager per-local-expert matmuls."""
    if permute_tokens.shape[0] == 0:
        return permute_tokens

    num_local_experts = gate_proj.shape[0]
    counts = _counts_from_cumsum(cumsum, num_local_experts)
    hidden_dim = down_proj.shape[-1]
    outputs = []
    start = 0
    for expert_idx, count in enumerate(counts):
        end = start + count
        current = permute_tokens[start:end]
        if count == 0:
            outputs.append(permute_tokens.new_empty((0, hidden_dim)))
            start = end
            continue

        compute_dtype = current.dtype
        gate_proj_out = torch.matmul(current, gate_proj[expert_idx])
        up_proj_out = torch.matmul(current, up_proj[expert_idx])

        gate_A = gate_proj_lora_A[min(expert_idx, gate_proj_lora_A.shape[0] - 1)].to(compute_dtype)
        gate_B = gate_proj_lora_B[expert_idx].to(compute_dtype)
        gate_proj_out = gate_proj_out + torch.matmul(torch.matmul(current, gate_A), gate_B) * scaling
        if swiglu_limit > 0:
            gate_proj_out = gate_proj_out.clamp(-swiglu_limit, swiglu_limit)

        up_A = up_proj_lora_A[min(expert_idx, up_proj_lora_A.shape[0] - 1)].to(compute_dtype)
        up_B = up_proj_lora_B[expert_idx].to(compute_dtype)
        up_proj_out = up_proj_out + torch.matmul(torch.matmul(current, up_A), up_B) * scaling

        out = apply_moe_activation("silu", gate_proj_out, up_proj_out)

        down_out = torch.matmul(out, down_proj[expert_idx])
        down_A = down_proj_lora_A[expert_idx].to(compute_dtype)
        down_B = down_proj_lora_B[min(expert_idx, down_proj_lora_B.shape[0] - 1)].to(compute_dtype)
        down_out = down_out + torch.matmul(torch.matmul(out, down_A), down_B) * scaling
        outputs.append(down_out)
        start = end

    return torch.cat(outputs, dim=0) if outputs else permute_tokens.new_empty((0, hidden_dim))
