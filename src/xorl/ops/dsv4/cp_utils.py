"""Utility functions for DeepSeek V4 context-parallel support."""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


@lru_cache(1)
def _get_window_topk_idxs_ref(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Reference (single-device, no-CP) window topk index builder. Used only as
    an equality oracle by :func:`get_window_topk_idxs_cp` when ``cp_size == 1``;
    the call site compares ``result.cpu()`` against ``ref.cpu()``, so the ref
    is built on CPU regardless of the live path's device.
    """

    def _inner():
        if start_pos >= window_size - 1:
            return torch.arange(window_size)
        elif start_pos > 0:
            return F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
        else:
            base = torch.arange(seqlen).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
            matrix = torch.where(matrix > base, -1, matrix)
            return matrix

    return _inner().unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def _get_compress_topk_idxs_ref(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    """Reference (single-device, no-CP) compress topk index builder. Used only as
    an equality oracle by :func:`get_compress_topk_idxs_cp` when ``cp_size == 1``;
    built on CPU like :func:`_get_window_topk_idxs_ref`.
    """

    def _inner():
        if start_pos > 0:
            return torch.arange(0, (start_pos + 1) // ratio) + offset
        else:
            matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
            mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            matrix = torch.where(mask, -1, matrix + offset)
            return matrix

    return _inner().unsqueeze(0).expand(bsz, -1, -1)


class _AllGatherCP(torch.autograd.Function):
    """Differentiable CP gather for arbitrary process-group rank ranges.

    ``torch.distributed.nn.functional.all_gather`` implements its Gloo VJP via
    subgroup-unsafe global ``scatter`` sources.  A DP2/CP4 owner plane therefore
    fails on its second CP group (global ranks 4..7).  The mathematical VJP is a
    reduce-scatter of the gathered-axis gradient; encode that directly so the
    same path works for every DP and PP-local CP group on Gloo and NCCL.
    """

    @staticmethod
    def forward(ctx, tensor: Tensor, dim: int, cp_group: torch.distributed.ProcessGroup) -> Tensor:
        dim = dim if dim >= 0 else tensor.ndim + dim
        if dim < 0 or dim >= tensor.ndim:
            raise IndexError(f"CP all-gather dimension {dim} is invalid for rank-{tensor.ndim} tensor")
        world_size = cp_group.size()
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor.contiguous(), group=cp_group)
        ctx.dim = dim
        ctx.cp_group = cp_group
        ctx.world_size = world_size
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # reduce_scatter_tensor splits its input on dimension 0.  Move the
        # gathered sequence axis there without changing its rank-order chunks.
        grad_front = grad_output.movedim(ctx.dim, 0).contiguous()
        if grad_front.shape[0] % ctx.world_size:
            raise RuntimeError(
                "CP all-gather backward cannot evenly reduce-scatter the gathered axis: "
                f"length={grad_front.shape[0]}, world_size={ctx.world_size}"
            )
        local_front = torch.empty(
            (grad_front.shape[0] // ctx.world_size, *grad_front.shape[1:]),
            dtype=grad_front.dtype,
            device=grad_front.device,
        )
        torch.distributed.reduce_scatter_tensor(
            local_front,
            grad_front,
            op=torch.distributed.ReduceOp.SUM,
            group=ctx.cp_group,
        )
        return local_front.movedim(0, ctx.dim), None, None


def all_gather_cp(tensor: Tensor, dim: int, cp_group: torch.distributed.ProcessGroup) -> Tensor:
    """All-gather CP shards in group-rank order with a reduce-scatter VJP.

    The trainer's CP collator owns contiguous sequence shards, so group-rank
    order is the request's natural logical token order.
    """

    return _AllGatherCP.apply(tensor, dim, cp_group)


@dataclass(frozen=True)
class Dsv4ExactCPLayout:
    """Immutable per-microbatch row plan for exact DSV4 attention.

    Local tensors are in compact local-query order and padded to
    ``compute_rows`` where noted. ``gather_order`` selects live rows from a
    rank-major gather of those padded tensors and restores original packed-row
    order. Per-request index tuples keep attention and compression isolated.
    """

    local_storage_indices: Tensor
    local_logical_rows: Tensor
    local_request_ids: Tensor
    local_request_positions: Tensor
    local_live_count: int
    compute_rows: int
    gather_order: Tensor
    global_logical_rows: Tensor
    global_request_ids: Tensor
    global_request_positions: Tensor
    request_ids: tuple[int, ...]
    local_request_row_indices: tuple[Tensor, ...]
    global_request_row_indices: tuple[Tensor, ...]


def _all_gather_metadata(tensor: Tensor, group: torch.distributed.ProcessGroup | None) -> list[Tensor]:
    if group is None:
        return [tensor]
    gathered = [torch.empty_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def build_dsv4_exact_cp_layout(
    local_logical_rows: Tensor,
    local_request_ids: Tensor,
    local_request_positions: Tensor,
    local_live_mask: Tensor,
    *,
    compute_rows: int,
    cp_group: torch.distributed.ProcessGroup | None,
) -> Dsv4ExactCPLayout:
    """Build the variable-row CP plan once at the model boundary."""

    logical = local_logical_rows.reshape(-1).to(dtype=torch.int64)
    request_ids = local_request_ids.reshape(-1).to(device=logical.device, dtype=torch.int64)
    request_positions = local_request_positions.reshape(-1).to(device=logical.device, dtype=torch.int64)
    live_mask = local_live_mask.reshape(-1).to(device=logical.device, dtype=torch.bool)
    if not (logical.numel() == request_ids.numel() == request_positions.numel() == live_mask.numel()):
        raise ValueError("DSV4 exact CP row metadata must have identical local storage lengths")

    storage_indices = live_mask.nonzero(as_tuple=False).reshape(-1)
    live_count = int(storage_indices.numel())
    if compute_rows < live_count:
        raise ValueError(f"DSV4 exact CP compute rows {compute_rows} are shorter than {live_count} local live rows")
    compact_logical = logical.index_select(0, storage_indices)
    compact_request_ids = request_ids.index_select(0, storage_indices)
    compact_request_positions = request_positions.index_select(0, storage_indices)

    def _pad(values: Tensor, value: int) -> Tensor:
        if values.numel() == compute_rows:
            return values.contiguous()
        return F.pad(values, (0, compute_rows - values.numel()), value=value).contiguous()

    padded_logical = _pad(compact_logical, -1)
    padded_request_ids = _pad(compact_request_ids, -1)
    padded_request_positions = _pad(compact_request_positions, 0)
    count_tensor = torch.tensor([live_count], dtype=torch.int64, device=logical.device)
    counts = torch.cat(_all_gather_metadata(count_tensor, cp_group))
    gathered_logical = torch.cat(_all_gather_metadata(padded_logical, cp_group))
    gathered_request_ids = torch.cat(_all_gather_metadata(padded_request_ids, cp_group))
    gathered_request_positions = torch.cat(_all_gather_metadata(padded_request_positions, cp_group))

    valid_rank_major = torch.cat(
        [
            torch.arange(rank * compute_rows, rank * compute_rows + int(count.item()), device=logical.device)
            for rank, count in enumerate(counts)
        ]
    )
    live_logical_rank_major = gathered_logical.index_select(0, valid_rank_major)
    logical_sort = torch.argsort(live_logical_rank_major, stable=True)
    gather_order = valid_rank_major.index_select(0, logical_sort)
    global_logical = gathered_logical.index_select(0, gather_order)
    expected_logical = torch.arange(global_logical.numel(), dtype=torch.int64, device=logical.device)
    if not torch.equal(global_logical, expected_logical):
        raise RuntimeError(
            "DSV4 exact CP live rows do not cover the packed logical stream exactly once: "
            f"rows={global_logical.numel()}, unique={torch.unique(global_logical).numel()}"
        )
    global_request_ids = gathered_request_ids.index_select(0, gather_order)
    global_request_positions = gathered_request_positions.index_select(0, gather_order)
    unique_request_ids = tuple(int(value) for value in torch.unique(global_request_ids, sorted=True).tolist())
    if any(request_id < 0 for request_id in unique_request_ids):
        raise RuntimeError("DSV4 exact CP marked a padding row as live")

    local_request_rows = []
    global_request_rows = []
    for request_id in unique_request_ids:
        local_rows = (padded_request_ids == request_id).nonzero(as_tuple=False).reshape(-1)
        global_rows = (global_request_ids == request_id).nonzero(as_tuple=False).reshape(-1)
        positions = global_request_positions.index_select(0, global_rows)
        expected_positions = torch.arange(positions.numel(), dtype=torch.int64, device=positions.device)
        if not torch.equal(positions, expected_positions):
            raise RuntimeError(
                f"DSV4 exact CP request {request_id} positions are not a complete serving stream: "
                f"rows={positions.numel()}"
            )
        local_request_rows.append(local_rows)
        global_request_rows.append(global_rows)

    return Dsv4ExactCPLayout(
        local_storage_indices=storage_indices,
        local_logical_rows=padded_logical,
        local_request_ids=padded_request_ids,
        local_request_positions=padded_request_positions,
        local_live_count=live_count,
        compute_rows=compute_rows,
        gather_order=gather_order,
        global_logical_rows=global_logical,
        global_request_ids=global_request_ids,
        global_request_positions=global_request_positions,
        request_ids=unique_request_ids,
        local_request_row_indices=tuple(local_request_rows),
        global_request_row_indices=tuple(global_request_rows),
    )


def gather_dsv4_exact_cp_rows(
    tensor: Tensor,
    *,
    dim: int,
    layout: Dsv4ExactCPLayout,
    cp_group: torch.distributed.ProcessGroup | None,
) -> Tensor:
    """Differentiably gather padded CP rows and compact them in logical order."""

    gathered = tensor if cp_group is None else all_gather_cp(tensor, dim=dim, cp_group=cp_group)
    return gathered.index_select(dim, layout.gather_order)


def get_q_positions_for_cp(
    seqlen_local: int,
    *,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    device,
) -> Tensor:
    """Get global positions for local q tokens (contiguous CP)."""
    if cp_size <= 1 or cp_group is None:
        return torch.arange(0, seqlen_local, device=device)
    cp_rank = cp_group.rank()
    start = cp_rank * seqlen_local
    return torch.arange(start, start + seqlen_local, device=device)


def get_window_topk_idxs_cp(
    q_positions: Tensor,
    *,
    window_size: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """Get window topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    base = q_positions.unsqueeze(1)
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen_global, window_size), device=device)
    topk_idxs = torch.where(k_pos > base, -1, k_pos)
    result = topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = _get_window_topk_idxs_ref(window_size, bsz, seqlen_local, start_pos=0)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_window_topk_idxs_cp mismatch with ref"

    return result


def get_compress_topk_idxs_cp(
    q_positions: Tensor,
    *,
    ratio: int,
    cp_size: int,
    bsz: int,
) -> Tensor:
    """Get static compress topk indices (CP-aware)."""
    device = q_positions.device
    seqlen_local = q_positions.shape[0]
    seqlen_global = seqlen_local * cp_size
    offset = seqlen_global
    k_group_idx = torch.arange(seqlen_global // ratio, device=device).repeat(seqlen_local, 1)
    q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
    invalid_mask = k_group_idx >= q_first_invalid_group
    compress_topk_idxs = torch.where(invalid_mask, -1, k_group_idx + offset)
    result = compress_topk_idxs.unsqueeze(0).expand(bsz, -1, -1)

    if cp_size == 1:
        ref_result = _get_compress_topk_idxs_ref(ratio, bsz, seqlen_local, start_pos=0, offset=offset)
        assert torch.equal(result.cpu(), ref_result.cpu()), "get_compress_topk_idxs_cp mismatch with ref"

    return result


def get_freqs_cis_for_cp(
    freqs_cis: Tensor,
    seqlen_local: int,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    stride: int = 1,
) -> Tensor:
    """Get freqs_cis for this CP rank (contiguous slice)."""
    expected = (seqlen_local + stride - 1) // stride
    if cp_size == 1 or cp_group is None:
        result = freqs_cis[:seqlen_local:stride]
        start = 0
        stop = seqlen_local
    else:
        cp_rank = cp_group.rank()
        start = cp_rank * seqlen_local
        stop = start + seqlen_local
        result = freqs_cis[start:stop:stride]
    if result.size(0) != expected:
        raise ValueError(
            "DSv4 RoPE cache is too short for this context-parallel slice: "
            f"need positions [{start}, {stop}) with stride {stride}, "
            f"but freqs_cis only has {freqs_cis.size(0)} positions. "
            "Increase XORL_DSV4_ROPE_MAX_SEQ_LEN or config.max_position_embeddings."
        )
    return result
