"""
Utility functions for DeepSeek V4 Context Parallelism support.
"""

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
