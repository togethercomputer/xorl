"""Native-EP ordered combine for the serving-kernel MoE parity lane (P5).

The TRAINING half of the EP<n>/a2a-none BI-ops serving pairing: serving computes,
on every rank, its contiguous expert slice's weighted contribution for ALL tokens
plus a TP slice of the shared expert, adds them in bf16, and all-reduces — and
NCCL's tree all-reduce over n intranode ranks is bitwise a SEQUENTIAL bf16 chain
sum in rank order (n-1) -> 0 (identified against captured per-rank partials; see
`experiments/k3_tests/moe_ep_combine_sim_gate.py` and PR #478).

The trainer mirrors this on its real EP group (trainer EP gives rank r exactly
serving-rank-r's contiguous expert slice):

1. gather the full token batch over the EP group (backward: reduce-scatter sum);
2. every rank computes ITS routed partial for ALL tokens through the trainable
   masked serving-kernel Function (T1: compact-valid-slots backward) plus ITS
   shared-expert TP slice (trainable BI GEMMs, torch-native bf16 silu*mul,
   sigmoid gate), added in bf16;
3. partials are exchanged RAW (all-to-all) — NEVER via an NCCL-summed
   collective (the sum order must stay ours) — and every rank chain-sums its
   own tokens' n partials locally in bf16, serving rank order (n-1) -> 0;
4. backward reverses the exchange (grad all-to-all), then the masked backward
   per rank. Grad REDUCTIONS (reduce-scatter of d(x), DP grad sync) stay stock
   NCCL — only forward bits are contracted.

FRAGILITY NOTE (doubly so here — this lane trains): the chain order is an NCCL
implementation detail (version/topology). Deployments must re-validate the
combine against one captured serving all-reduce at bring-up (the step-0 gate,
`experiments/k3_tests/moe_ep_native_combine_gate.py`) before trusting a new
serving environment.

Enable with ``XORL_MOE_SGLANG_EP_COMBINE=native`` (opt-in; requires trainer EP
mirroring the serving EP size). The single-GPU simulation
(``XORL_MOE_SGLANG_EP_COMBINE_SIM``) stays the offline verification harness.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


_MOE_SGLANG_EP_COMBINE_ENV = "XORL_MOE_SGLANG_EP_COMBINE"


def moe_sglang_ep_combine_native_enabled() -> bool:
    """True when ``XORL_MOE_SGLANG_EP_COMBINE=native`` arms the training lane."""
    raw = os.environ.get(_MOE_SGLANG_EP_COMBINE_ENV, "").strip().lower()
    if not raw or raw in {"0", "false", "off", "no"}:
        return False
    if raw != "native":
        raise ValueError(f"{_MOE_SGLANG_EP_COMBINE_ENV}={raw!r} is not supported; expected 'native' (or unset)")
    return True


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
