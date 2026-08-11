"""Native-EP canonical combine for the exact Qwen3.5-MoE program.

The TRAINING half of the EP<n>/a2a-none BI-ops serving pairing: serving computes,
on every rank, its contiguous expert slice's weighted contribution for ALL tokens
plus a TP slice of the shared expert, adds them in bf16, transports the raw
partials, and applies the version-1 adjacent-pair BF16 fold. Logical expert-slice
identity, rather than collective history, determines the contributor order.

The trainer mirrors this on its real EP group (trainer EP gives rank r exactly
serving-rank-r's contiguous expert slice):

1. gather the full token batch over the EP group (backward: reduce-scatter sum);
2. every rank computes ITS routed partial for ALL tokens through the trainable
   masked serving-kernel Function (T1: compact-valid-slots backward) plus ITS
   shared-expert TP slice (trainable BI GEMMs and torch-native bf16 silu*mul),
   joined with the routed partial by serving's fused gate/sigmoid/mul/add;
3. partials are exchanged RAW (all-to-all) — NEVER via an NCCL-summed
   collective (the fold order must stay ours) — and every rank applies the
   shared adjacent-pair BF16 tree to its own tokens' logical contributions;
4. backward reverses the exchange (grad all-to-all), then the masked backward
   per rank. Grad REDUCTIONS (reduce-scatter of d(x), DP grad sync) stay stock
   NCCL — only forward bits are contracted.

The exact Qwen3.5-MoE model path selects this combine structurally when trainer
EP is active. EP8 remains the only admitted Qwen topology; other contributor
counts require their own end-to-end topology qualification even though the
shared arithmetic primitive admits them.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


QWEN35_NATIVE_EP_COMBINE_SIZES = frozenset({8})


def validate_qwen35_native_ep_combine_size(ep_size: int) -> None:
    """Reject contributor counts not validated against Qwen serving."""
    if ep_size not in QWEN35_NATIVE_EP_COMBINE_SIZES:
        raise ValueError(
            f"Qwen3.5-MoE canonical combine requires EP8 (expert_parallel_size=8); received {ep_size}. "
            "A different EP size requires a serving capture and a new "
            "byte-exact reduction-order validation."
        )


class _AllGatherSumBackward(torch.autograd.Function):
    """All-gather rows over the EP group; backward reduce-scatters the SUM.

    Every rank's partial depends on every gathered row, so the gather's
    backward must SUM the per-rank contributions (unlike the CP helpers, whose
    backward slices). The reduce-scatter is stock NCCL — backward carries no
    bitwise contract.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, padded_rows: int) -> torch.Tensor:
        ctx.group = group
        ctx.local_rows = x.shape[0]
        ctx.padded_rows = int(padded_rows)
        if ctx.padded_rows < ctx.local_rows:
            raise ValueError(
                f"EP native combine padded_rows={ctx.padded_rows} is smaller than local_rows={ctx.local_rows}"
            )
        x = _pad_rows(x, ctx.padded_rows).contiguous()
        world_size = dist.get_world_size(group)
        out = torch.empty((world_size * ctx.padded_rows, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_local_padded = torch.empty(
            (ctx.padded_rows, *grad_output.shape[1:]), dtype=grad_output.dtype, device=grad_output.device
        )
        dist.reduce_scatter_tensor(grad_local_padded, grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_local_padded[: ctx.local_rows], None, None


def _pad_rows(x: torch.Tensor, padded_rows: int, *, value: int | float = 0) -> torch.Tensor:
    """Right-pad the leading row dimension without changing equal-row inputs."""
    if x.shape[0] == padded_rows:
        return x
    pad = x.new_full((padded_rows - x.shape[0], *x.shape[1:]), value)
    return torch.cat((x, pad), dim=0)


def max_rows_for_ep_combine(local_rows: int, device: torch.device, group) -> int:
    """Negotiate a common EP row count for variable-length packed DP slices."""
    rows = torch.tensor([local_rows], dtype=torch.int64, device=device)
    dist.all_reduce(rows, op=dist.ReduceOp.MAX, group=group)
    return int(rows.item())


def gather_tokens_for_ep_combine(x: torch.Tensor, group, padded_rows: int | None = None) -> torch.Tensor:
    """Autograd token gather with EP-uniform row padding.

    Dispatcher DP slices can contain different packed sequence lengths. Native
    EP still needs every expert rank to see every slice, so shorter ranks are
    padded to the negotiated maximum before the equal-count NCCL gather. The
    caller slices the final folded result back to its original row count.
    """
    if padded_rows is None:
        padded_rows = max_rows_for_ep_combine(x.shape[0], x.device, group)
    return _AllGatherSumBackward.apply(x, group, padded_rows)


def gather_ids_for_ep_combine(ids: torch.Tensor, group, padded_rows: int | None = None) -> torch.Tensor:
    """Plain gather for integer routing ids, padding invalid rows with ``-1``."""
    if padded_rows is None:
        padded_rows = max_rows_for_ep_combine(ids.shape[0], ids.device, group)
    if padded_rows < ids.shape[0]:
        raise ValueError(f"EP native combine padded_rows={padded_rows} is smaller than local_rows={ids.shape[0]}")
    ids = _pad_rows(ids, padded_rows, value=-1).contiguous()
    world_size = dist.get_world_size(group)
    out = torch.empty((world_size * padded_rows, *ids.shape[1:]), dtype=ids.dtype, device=ids.device)
    dist.all_gather_into_tensor(out, ids, group=group)
    return out


def _run_sglang_fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Call the installed sampler's exact shared-gate forward kernel."""
    from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add  # noqa: PLC0415

    fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, final_hidden_states)


class _SGLangFusedGateSigmoidMulAdd(torch.autograd.Function):
    """Serving-exact shared-gate forward with an ordinary differentiable backward."""

    @staticmethod
    def forward(ctx, hidden_states, gate_weight, shared_output, routed_output):
        ctx.save_for_backward(hidden_states, gate_weight, shared_output)
        ctx.routed_dtype = routed_output.dtype
        result = routed_output.contiguous().clone()
        _run_sglang_fused_gate_sigmoid_mul_add(
            hidden_states.contiguous(),
            gate_weight.contiguous(),
            shared_output.contiguous(),
            result,
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, gate_weight, shared_output = ctx.saved_tensors
        hidden_fp32 = hidden_states.float()
        weight_fp32 = gate_weight.float()
        shared_fp32 = shared_output.float()
        grad_fp32 = grad_output.float()

        gate = torch.sigmoid((hidden_fp32 * weight_fp32).sum(dim=-1))
        grad_gate_logits = (grad_fp32 * shared_fp32).sum(dim=-1) * gate * (1.0 - gate)
        grad_hidden = (grad_gate_logits.unsqueeze(-1) * weight_fp32).to(hidden_states.dtype)
        grad_weight = (grad_gate_logits.unsqueeze(-1) * hidden_fp32).sum(dim=0).to(gate_weight.dtype)
        grad_shared = (grad_fp32 * gate.unsqueeze(-1)).to(shared_output.dtype)
        grad_routed = grad_output.to(ctx.routed_dtype)
        return grad_hidden, grad_weight, grad_shared, grad_routed


def sglang_fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
) -> torch.Tensor:
    """Apply the sampler's forward operation without giving up trainer gradients."""
    return _SGLangFusedGateSigmoidMulAdd.apply(hidden_states, gate_weight, shared_output, routed_output)


def exchange_and_canonical_fold(partial: torch.Tensor, group, ep_size: int) -> torch.Tensor:
    """RAW partial exchange followed by the version-1 canonical BF16 fold.

    ``partial`` is this rank's [n*T, H] contribution for ALL gathered tokens.
    The all-to-all hands rank r the n per-rank partials for ITS OWN T rows;
    group-rank order is also Qwen's logical expert-slice ordinal order, so the
    shared fold consumes the reshaped arrivals directly.
    """
    from xorl.distributed.canonical_moe import canonical_moe_fold_v1  # noqa: PLC0415
    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    validate_qwen35_native_ep_combine_size(ep_size)
    if partial.ndim < 2:
        raise ValueError("Qwen3.5-MoE canonical combine requires row and payload dimensions")
    if partial.dtype is not torch.bfloat16:
        raise TypeError("Qwen3.5-MoE canonical combine requires BF16 partials")
    if partial.shape[0] % ep_size:
        raise ValueError("Qwen3.5-MoE canonical combine rows must be divisible by EP size")
    exchanged = _AllToAll.apply(group, partial.contiguous(), None, None)  # [n*T, H], segment s from rank s
    rows = exchanged.shape[0] // ep_size
    logical_sources = exchanged.reshape(ep_size, rows, *exchanged.shape[1:])
    return canonical_moe_fold_v1(logical_sources)
