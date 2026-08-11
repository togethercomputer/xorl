"""DSV4-Flash native-EP combine: NCCL-tree-order contributor reproduction.

ARCHITECTURE-SCOPED ON PURPOSE. The Qwen3.5/GLM exact lanes reduce MoE
partials with the canonical adjacent-pair BF16 fold (``canonical_moe_fold_v1``
in ``xorl.distributed.canonical_moe``); the DSV4-Flash serving contract
instead pins ``NCCL_ALGO=allreduce:tree`` and the trainer reproduces the
observed bitwise contributor order ``[1, 2, .., N-1, 0]`` as a
left-associative BF16 chain. These are different byte programs: folding
DSV4 through the generic adjacent-pair tree invalidates its 64-decision
qualification (togethercomputer/xorl#45) and requires full requalification.

The variable-row gather primitives are deliberately duplicated here rather
than shared with the Qwen combine module so each architecture's byte
contract can evolve (or be requalified) independently.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


DSV4_NATIVE_EP_COMBINE_SIZES = frozenset({8})


def validate_dsv4_native_ep_combine_size(ep_size: int) -> None:
    """Reject contributor counts not validated against DSV4 serving."""
    if ep_size not in DSV4_NATIVE_EP_COMBINE_SIZES:
        raise ValueError(
            f"DSV4-Flash native EP combine admits EP sizes {sorted(DSV4_NATIVE_EP_COMBINE_SIZES)}; "
            f"received {ep_size}. New sizes require byte-exact reduction-order validation."
        )


class _AllGatherSumBackward(torch.autograd.Function):
    """Equal-count all-gather whose backward SUMS per-rank contributions.

    The reduce-scatter is stock NCCL — backward carries no bitwise contract
    (forward bits are the contract; backward is trainer-owned numerics).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, padded_rows: int) -> torch.Tensor:
        ctx.group = group
        ctx.local_rows = x.shape[0]
        ctx.padded_rows = int(padded_rows)
        if ctx.padded_rows < ctx.local_rows:
            raise ValueError(
                f"DSV4 EP combine padded_rows={ctx.padded_rows} is smaller than local_rows={ctx.local_rows}"
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
        raise RuntimeError(f"DSV4 EP combine received negative row counts: {counts}")
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
    """Autograd token gather with EP-uniform row padding."""
    if padded_rows is None:
        padded_rows = max_rows_for_ep_combine(x.shape[0], x.device, group)
    return _AllGatherSumBackward.apply(x, group, padded_rows)


def gather_ids_for_ep_combine(ids: torch.Tensor, group, padded_rows: int | None = None) -> torch.Tensor:
    """Plain gather for integer routing ids, padding invalid rows with ``-1``."""
    if padded_rows is None:
        padded_rows = max_rows_for_ep_combine(ids.shape[0], ids.device, group)
    if padded_rows < ids.shape[0]:
        raise ValueError(f"DSV4 EP combine padded_rows={padded_rows} is smaller than local_rows={ids.shape[0]}")
    ids = _pad_rows(ids, padded_rows, value=-1).contiguous()
    world_size = dist.get_world_size(group)
    out = torch.empty((world_size * padded_rows, *ids.shape[1:]), dtype=ids.dtype, device=ids.device)
    dist.all_gather_into_tensor(out, ids, group=group)
    return out


def dsv4_nccl_tree_chain_order(ep_size: int) -> tuple[int, ...]:
    """The DSV4-captured bitwise contributor order of NCCL_ALGO=allreduce:tree.

    Captured 2026-08-11 on the EP8 qualification lane: the tree's result is a
    left-associative BF16 chain seeded at rank 1, adding ranks 2..N-1, with
    rank 0 last. Byte-verified against rank-0 serving dumps (40919/40960
    columns matched with only expert-partial ulps outstanding at capture
    time; end-to-end K3 = 0 after the remaining repairs).
    """

    validate_dsv4_native_ep_combine_size(ep_size)
    return (*range(1, ep_size), 0)


def exchange_variable_and_nccl_tree_chain_sum(
    partial: torch.Tensor,
    group,
    row_counts: tuple[int, ...],
    local_rank: int,
) -> torch.Tensor:
    """Variable-row partial exchange + the DSV4 NCCL-tree-order BF16 chain.

    ``partial`` contains every rank's live rows in destination-rank order.
    Raw all-to-all returns this destination's rows from each source rank; the
    final BF16 chain reproduces the serving contributor order (see
    :func:`dsv4_nccl_tree_chain_order`).
    """

    from xorl.distributed.moe.comm import _AllToAll  # noqa: PLC0415

    ep_size = len(row_counts)
    if not 0 <= local_rank < ep_size:
        raise ValueError(f"EP local rank {local_rank} is outside [0, {ep_size})")
    chain_order = dsv4_nccl_tree_chain_order(ep_size)
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


__all__ = [
    "DSV4_NATIVE_EP_COMBINE_SIZES",
    "compact_rank_padded_rows",
    "dsv4_nccl_tree_chain_order",
    "exchange_variable_and_nccl_tree_chain_sum",
    "gather_ids_for_ep_combine",
    "gather_tokens_for_ep_combine",
    "max_rows_for_ep_combine",
    "row_counts_for_ep_combine",
    "validate_dsv4_native_ep_combine_size",
]
