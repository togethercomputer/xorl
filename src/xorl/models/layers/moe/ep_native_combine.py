"""Native-EP ordered combine for the exact Qwen3.5-MoE program.

The TRAINING half of the EP<n>/a2a-none BI-ops serving pairing: serving computes,
on every rank, its contiguous expert slice's weighted contribution for ALL tokens
plus a TP slice of the shared expert, adds them in bf16, and all-reduces — and
NCCL's tree all-reduce over n intranode ranks is bitwise a SEQUENTIAL bf16 chain
sum in rank order (n-1) -> 0 (identified against captured per-rank partials; see
the exact-path component tests). The contributor order is part of the exact
Qwen3.5 numerical program.

The trainer mirrors this on its real EP group (trainer EP gives rank r exactly
serving-rank-r's contiguous expert slice):

1. gather the full token batch over the EP group (backward: reduce-scatter sum);
2. every rank computes ITS routed partial for ALL tokens through the trainable
   masked serving-kernel Function (T1: compact-valid-slots backward) plus ITS
   shared-expert TP slice (trainable BI GEMMs and torch-native bf16 silu*mul),
   joined with the routed partial by serving's fused gate/sigmoid/mul/add;
3. partials are exchanged RAW (all-to-all) — NEVER via an NCCL-summed
   collective (the sum order must stay ours) — and every rank chain-sums its
   own tokens' n partials locally in bf16, serving rank order (n-1) -> 0;
4. backward reverses the exchange (grad all-to-all), then the masked backward
   per rank. Grad REDUCTIONS (reduce-scatter of d(x), DP grad sync) stay stock
   NCCL — only forward bits are contracted.

The exact Qwen3.5-MoE model path selects this combine structurally when trainer
EP is active. EP8 is the admitted topology because it is the topology whose
serving reduction was captured; a different contributor count is a different
numerical program and is rejected before collectives begin.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


QWEN35_NATIVE_EP_COMBINE_SIZES = frozenset({8})


def validate_qwen35_native_ep_combine_size(ep_size: int) -> None:
    """Reject contributor counts not validated against Qwen serving."""
    if ep_size not in QWEN35_NATIVE_EP_COMBINE_SIZES:
        raise ValueError(
            f"Qwen3.5-MoE exact ordered combine requires EP8 (expert_parallel_size=8); received {ep_size}. "
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


def row_counts_for_ep_combine(local_rows: int, device: torch.device, group) -> tuple[int, ...]:
    """Gather the live (unpadded) row count in EP rank order."""

    local = torch.tensor([local_rows], dtype=torch.int64, device=device)
    world_size = dist.get_world_size(group)
    gathered = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    counts = tuple(int(value) for value in gathered.cpu().tolist())
    if any(value < 0 for value in counts):
        raise RuntimeError(f"EP native combine received negative row counts: {counts}")
    return counts


def compact_rank_padded_rows(
    gathered: torch.Tensor,
    *,
    padded_rows: int,
    row_counts: tuple[int, ...],
) -> torch.Tensor:
    """Remove per-rank right padding from an equal-count all-gather."""

    if padded_rows < 0 or any(count > padded_rows for count in row_counts):
        raise ValueError(f"Invalid compact-row geometry: padded_rows={padded_rows}, row_counts={row_counts}")
    expected_rows = len(row_counts) * padded_rows
    if gathered.shape[0] != expected_rows:
        raise ValueError(
            f"Gathered row count {gathered.shape[0]} does not match {len(row_counts)}*{padded_rows}"
        )
    pieces = [
        gathered[rank * padded_rows : rank * padded_rows + count]
        for rank, count in enumerate(row_counts)
        if count
    ]
    if pieces:
        return torch.cat(pieces, dim=0)
    return gathered[:0]


def gather_tokens_for_ep_combine(x: torch.Tensor, group, padded_rows: int | None = None) -> torch.Tensor:
    """Autograd token gather with EP-uniform row padding.

    Dispatcher DP slices can contain different packed sequence lengths. Native
    EP still needs every expert rank to see every slice, so shorter ranks are
    padded to the negotiated maximum before the equal-count NCCL gather. The
    caller slices the final chain-summed result back to its original row count.
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


def exchange_and_chain_sum(partial: torch.Tensor, group, ep_size: int) -> torch.Tensor:
    """RAW partial exchange + the serving-order bf16 chain sum.

    ``partial`` is this rank's [n*T, H] contribution for ALL gathered tokens.
    The all-to-all hands rank r the n per-rank partials for ITS OWN T rows;
    the chain then sums them SEQUENTIALLY in serving rank order (n-1) -> 0 —
    bitwise NCCL's intranode tree all-reduce (the K3 contract; never replace
    with a summed collective or a stacked .sum()).
    """
    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    exchanged = _AllToAll.apply(group, partial.contiguous(), None, None)  # [n*T, H], segment s from rank s
    rows = exchanged.shape[0] // ep_size
    acc = exchanged[(ep_size - 1) * rows : ep_size * rows]
    for s in range(ep_size - 2, -1, -1):
        acc = acc + exchanged[s * rows : (s + 1) * rows]
    return acc


def exchange_variable_and_chain_sum(
    partial: torch.Tensor,
    group,
    row_counts: tuple[int, ...],
    local_rank: int,
    chain_order: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Variable-row equivalent of :func:`exchange_and_chain_sum`.

    ``partial`` contains every rank's live rows in destination-rank order.
    Raw all-to-all returns this destination's rows from each source rank; the
    final BF16 chain retains the serving contributor order.

    ``chain_order`` is the serving-captured contributor sequence (first entry
    is the accumulator seed, the rest are added left-associatively in order).
    Default is the Qwen3.5-captured reverse-rank chain [N-1 .. 0]; the DSV4
    exact lane passes the NCCL-tree-captured order [1, 2, .., N-1, 0].
    """

    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    ep_size = len(row_counts)
    if not 0 <= local_rank < ep_size:
        raise ValueError(f"EP local rank {local_rank} is outside [0, {ep_size})")
    if chain_order is None:
        chain_order = tuple(range(ep_size - 1, -1, -1))
    if sorted(chain_order) != list(range(ep_size)):
        raise ValueError(f"chain_order {chain_order!r} must be a permutation of range({ep_size})")
    total_rows = sum(row_counts)
    if partial.shape[0] != total_rows:
        raise ValueError(f"Partial rows {partial.shape[0]} do not match live row total {total_rows}")
    local_rows = row_counts[local_rank]
    exchanged = _AllToAll.apply(
        group,
        partial.contiguous(),
        [local_rows] * ep_size,
        list(row_counts),
    )
    if local_rows == 0:
        return exchanged[:0]
    seed = chain_order[0]
    acc = exchanged[seed * local_rows : (seed + 1) * local_rows]
    for source_rank in chain_order[1:]:
        acc = acc + exchanged[source_rank * local_rows : (source_rank + 1) * local_rows]
    return acc
