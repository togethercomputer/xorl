"""Ring attention using flash_attn lower-level API.

Implements distributed attention across ring-parallel ranks via P2P ring
communication with double-buffered CUDA streams for compute-comm overlap.

Optimizations over naive all-gather approach:
1. P2P ring with 2 CUDA streams: overlaps KV transfer with attention compute
2. Zigzag load balancing: all ranks compute all steps for causal attn
3. Memory-efficient backward: saves only local K,V, re-gathers in backward

Supports both batched [B, S, H, D] and varlen [total, H, D] tensor layouts.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn_interface import _flash_attn_backward, _flash_attn_forward
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _merge_attn_outputs(
    out_acc: Tensor,
    lse_acc: Tensor,
    out_new: Tensor,
    lse_new: Tensor,
    is_varlen: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Merge two partial attention outputs using numerically stable log-sum-exp.

    Uses the identity:
        merged_out = sigma * out_acc + (1 - sigma) * out_new
        where sigma = sigmoid(lse_acc - lse_new)  (computed via softplus for stability)

    Args:
        out_acc: accumulated output [B, S, H, D] (batched) or [total, H, D] (varlen)
        lse_acc: accumulated LSE [B, H, S] (batched) or [H, total] (varlen)
        out_new: new step's output, same shape as out_acc
        lse_new: new step's LSE, same shape as lse_acc
        is_varlen: if True, use varlen tensor layout

    Returns:
        (merged_out, merged_lse) with same shapes as inputs.
    """
    if is_varlen:
        # lse: [H, total], out: [total, H, D]
        # Transpose lse to [total, H, 1] for broadcasting
        lse_acc_expanded = lse_acc.T.unsqueeze(-1)  # [total, H, 1]
        lse_new_expanded = lse_new.T.unsqueeze(-1)  # [total, H, 1]
    else:
        # lse: [B, H, S], out: [B, S, H, D]
        # Transpose lse to [B, S, H, 1] for broadcasting
        lse_acc_expanded = lse_acc.transpose(1, 2).unsqueeze(-1)  # [B, S, H, 1]
        lse_new_expanded = lse_new.transpose(1, 2).unsqueeze(-1)  # [B, S, H, 1]

    # Numerically stable merge via softplus
    # log(sigma) = -softplus(lse_new - lse_acc)
    # log(1 - sigma) = -softplus(lse_acc - lse_new)
    diff = lse_new_expanded - lse_acc_expanded
    log_sigma = -F.softplus(diff)
    log_one_minus_sigma = -F.softplus(-diff)

    out = out_acc * torch.exp(log_sigma) + out_new * torch.exp(log_one_minus_sigma)

    # Merged LSE: lse = lse_acc - log_sigma
    if is_varlen:
        lse = lse_acc - log_sigma.squeeze(-1).T  # [H, total]
    else:
        lse = lse_acc - log_sigma.squeeze(-1).transpose(1, 2)  # [B, H, S]

    return out, lse


def _all_gather_kv(
    k: Tensor,
    v: Tensor,
    ringattn_group: dist.ProcessGroup,
) -> Tuple[List[Tensor], List[Tensor]]:
    """All-gather KV from all ring ranks.

    Stacks K and V into a single buffer, performs one all_gather, then splits.

    Args:
        k: local key tensor [B, S_local, H_kv, D] or [total_local, H_kv, D]
        v: local value tensor, same shape as k
        ringattn_group: ring attention process group

    Returns:
        (k_list, v_list): lists of ringattn_size tensors, each same shape as k/v.
    """
    ringattn_size = dist.get_world_size(ringattn_group)

    # Pack K,V into single buffer: [2, *k.shape]
    kv_local = torch.stack([k, v], dim=0).contiguous()

    # All-gather: [2 * ringattn_size, *k.shape]
    kv_gathered = torch.empty(
        (ringattn_size * kv_local.shape[0], *kv_local.shape[1:]),
        dtype=kv_local.dtype,
        device=kv_local.device,
    )
    dist.all_gather_into_tensor(kv_gathered, kv_local, group=ringattn_group)

    # Split into per-rank [2, *k.shape] chunks
    kv_chunks = kv_gathered.chunk(ringattn_size, dim=0)
    k_list = [chunk[0].contiguous() for chunk in kv_chunks]
    v_list = [chunk[1].contiguous() for chunk in kv_chunks]
    return k_list, v_list


def _reduce_scatter_grads(
    grad_list: List[Tensor],
    ringattn_group: dist.ProcessGroup,
) -> Tensor:
    """Reduce-scatter gradients back to owning ring ranks.

    Args:
        grad_list: list of ringattn_size tensors (one per rank), each same shape.
        ringattn_group: ring attention process group.

    Returns:
        Local gradient tensor (reduced sum of contributions from all ranks).
    """
    # Stack into [ringattn_size, *shape] for reduce_scatter
    stacked = torch.stack(grad_list, dim=0).contiguous()

    # Output: single chunk [*shape]
    output = torch.empty_like(grad_list[0])
    dist.reduce_scatter_tensor(output, stacked, op=dist.ReduceOp.SUM, group=ringattn_group)
    return output


# ---------------------------------------------------------------------------
# P2P Communication
# ---------------------------------------------------------------------------


def _p2p_communicate(
    rank: int,
    send_buf: Tensor,
    send_dst: int,
    recv_buf: Tensor,
    recv_src: int,
    ringattn_group: dist.ProcessGroup,
) -> List:
    """Async P2P send/recv with deadlock avoidance.

    Even ranks send first then recv; odd ranks recv first then send.
    Uses batch_isend_irecv for efficiency.

    Returns:
        List of async request handles (call .wait() on each to synchronize).
    """
    ops = []
    if rank % 2 == 0:
        ops.append(dist.P2POp(dist.isend, send_buf, send_dst, ringattn_group))
        ops.append(dist.P2POp(dist.irecv, recv_buf, recv_src, ringattn_group))
    else:
        ops.append(dist.P2POp(dist.irecv, recv_buf, recv_src, ringattn_group))
        ops.append(dist.P2POp(dist.isend, send_buf, send_dst, ringattn_group))
    return dist.batch_isend_irecv(ops)


# ---------------------------------------------------------------------------
# Zigzag Load Balancing
# ---------------------------------------------------------------------------
# Data arrives pre-zigzagged from the collator: each ring rank holds
# [early_chunk, late_chunk] per document. The section logic below determines
# which sub-tensors to use at each ring step.


def _zigzag_min_chunk_id(rank: int, ringattn_size: int) -> int:
    """Get the minimum global chunk position for a rank after zigzag.

    After zigzag reorder, rank r holds sub-chunks [r, 2*ringattn_size-1-r].
    The minimum is simply min(r, 2*ringattn_size-1-r).
    """
    return min(rank, 2 * ringattn_size - 1 - rank)


def _get_zigzag_step_section(ringattn_rank: int, ringattn_size: int, step: int) -> str:
    """Determine the section type for a load-balanced ring step.

    After zigzag, each rank has chunks at positions [min_r, max_r].
    The section determines which sub-tensors to use for attention.

    Returns:
        "diagonal", "lower", or "upper"
    """
    if step == 0:
        return "diagonal"
    source = (ringattn_rank - step) % ringattn_size
    min_q = _zigzag_min_chunk_id(ringattn_rank, ringattn_size)
    min_kv = _zigzag_min_chunk_id(source, ringattn_size)
    if min_q > min_kv:
        return "lower"  # Q's early chunk is later → all Q attends to KV first half
    else:
        return "upper"  # Q's early chunk is earlier → Q second half attends to all KV


def _compute_half_indices(cu_seqlens: Tensor) -> Tuple[Tensor, Tensor]:
    """Precompute index tensors for extracting early/late halves of each doc.

    Each doc in a zigzag-reordered packed sequence has [early_half, late_half].
    This function returns index tensors to extract all early halves or all late
    halves as contiguous tensors for flash attention.

    Args:
        cu_seqlens: [num_docs + 1] cumulative sequence lengths

    Returns:
        (early_idx, late_idx): index tensors for index_select on dim=0
    """
    num_docs = cu_seqlens.shape[0] - 1
    early_parts = []
    late_parts = []
    for d in range(num_docs):
        start = cu_seqlens[d].item()
        end = cu_seqlens[d + 1].item()
        mid = (start + end) // 2
        early_parts.append(torch.arange(start, mid, device=cu_seqlens.device))
        late_parts.append(torch.arange(mid, end, device=cu_seqlens.device))
    return torch.cat(early_parts), torch.cat(late_parts)


# ---------------------------------------------------------------------------
# Flash attention call helpers
# ---------------------------------------------------------------------------


def _flash_fwd(
    q,
    k,
    v,
    softmax_scale,
    dropout_p,
    causal,
    is_varlen,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    """Flash attention forward (FA3). Handles both batched and varlen via cu_seqlens kwargs."""
    kwargs = dict(softmax_scale=softmax_scale, causal=causal)
    if is_varlen:
        kwargs.update(
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k
        )
    out, lse, _, _ = _flash_attn_forward(q, k, v, **kwargs)
    return out, lse


def _flash_bwd(
    grad_out,
    q,
    k,
    v,
    out,
    lse,
    dq,
    dk,
    dv,
    softmax_scale,
    dropout_p,
    causal,
    is_varlen,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    """Flash attention backward (FA3). Handles both batched and varlen via cu_seqlens kwargs."""
    kwargs = dict(softmax_scale=softmax_scale, is_causal=causal, dq=dq, dk=dk, dv=dv)
    if is_varlen:
        kwargs.update(
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k
        )
    _flash_attn_backward(grad_out, q, k, v, out, lse, **kwargs)


# ---------------------------------------------------------------------------
# P2P Ring Attention with Double-Buffered Streams
# ---------------------------------------------------------------------------


class RingAttentionP2PFunc(torch.autograd.Function):
    """Ring attention via P2P KV rotation with double-buffered CUDA streams.

    Forward:
        1. Pack local KV into contiguous buffer
        2. P2P ring loop: alternate between two streams for compute-comm overlap
        3. For causal: data arrives pre-zigzagged from collator

    Backward:
        1. Re-gather KV via all-gather (memory-efficient: only local K,V saved)
        2. Gradient loop over computed steps (section-based for zigzag)
        3. Reduce-scatter dk, dv back to owning ranks
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        softmax_scale: float,
        dropout_p: float,
        causal: bool,
        ringattn_group: dist.ProcessGroup,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> Tensor:
        ringattn_rank = dist.get_rank(ringattn_group)
        ringattn_size = dist.get_world_size(ringattn_group)
        is_varlen = cu_seqlens_q is not None

        # Zigzag load balancing: data arrives pre-zigzagged from the collator.
        # Each rank holds [early_chunk, late_chunk] per document.
        zigzag = causal and ringattn_size > 1
        seq_dim = 0 if is_varlen else 1  # varlen: [total, H, D], batched: [B, S, H, D]

        # P2P ring destinations (convert group-local to global ranks)
        global_ranks = dist.get_process_group_ranks(ringattn_group)
        send_dst = global_ranks[(ringattn_rank + 1) % ringattn_size]
        recv_src = global_ranks[(ringattn_rank - 1) % ringattn_size]

        # Pack KV into contiguous buffer for P2P
        k_numel = k.numel()
        kv_buf = torch.cat([k.reshape(-1), v.reshape(-1)])

        # Double-buffered P2P
        kv_bufs = [kv_buf, torch.empty_like(kv_buf)]
        send_recv_reqs: List = []

        # Compute on default stream, P2P on comm stream for overlap.
        # All compute and output merging stays on compute_stream to avoid
        # cross-stream races on shared accumulators (out, lse).
        compute_stream = torch.cuda.current_stream(q.device)
        comm_stream = torch.cuda.Stream(device=q.device)

        out = None
        lse = None
        computed_steps = []

        # Precompute half-sequence info for zigzag varlen sections
        early_idx = late_idx = None
        cu_half_q = cu_half_k = None
        max_half_q = max_half_k = None
        if zigzag and is_varlen:
            early_idx, late_idx = _compute_half_indices(cu_seqlens_q)
            cu_half_q = cu_seqlens_q // 2
            cu_half_k = cu_seqlens_k // 2
            max_half_q = max_seqlen_q // 2 if max_seqlen_q else None
            max_half_k = max_seqlen_k // 2 if max_seqlen_k else None

        for i in range(ringattn_size):
            # Wait for P2P recv of kv_bufs[i%2] to complete
            for req in send_recv_reqs:
                req.wait()  # syncs compute_stream with NCCL completion
            send_recv_reqs = []

            # Post P2P for NEXT step on comm_stream (overlaps with compute below)
            if i < ringattn_size - 1:
                with torch.cuda.stream(comm_stream):
                    # Wait for compute to finish reading kv_bufs[(i+1)%2]
                    comm_stream.wait_stream(compute_stream)
                    send_recv_reqs = _p2p_communicate(
                        ringattn_rank,
                        kv_bufs[i % 2],
                        send_dst,
                        kv_bufs[(i + 1) % 2],
                        recv_src,
                        ringattn_group,
                    )

            # Extract K,V from current buffer (on compute_stream)
            k_step = kv_bufs[i % 2][:k_numel].view(k.shape)
            v_step = kv_bufs[i % 2][k_numel:].view(v.shape)

            # Determine section and compute attention
            if zigzag:
                section = _get_zigzag_step_section(ringattn_rank, ringattn_size, i)

                if section == "lower":
                    # All Q attends to early half of each doc in KV (no mask)
                    if is_varlen:
                        k_half = k_step.index_select(0, early_idx).contiguous()
                        v_half = v_step.index_select(0, early_idx).contiguous()
                        out_step, lse_step = _flash_fwd(
                            q,
                            k_half,
                            v_half,
                            softmax_scale,
                            dropout_p,
                            False,
                            True,
                            cu_seqlens_q,
                            cu_half_k,
                            max_seqlen_q,
                            max_half_k,
                        )
                    else:
                        half = k_step.shape[seq_dim] // 2
                        k_half = k_step.narrow(seq_dim, 0, half).contiguous()
                        v_half = v_step.narrow(seq_dim, 0, half).contiguous()
                        out_step, lse_step = _flash_fwd(
                            q,
                            k_half,
                            v_half,
                            softmax_scale,
                            dropout_p,
                            False,
                            False,
                        )
                elif section == "upper":
                    # Late half of each doc in Q attends to all KV (no mask)
                    if is_varlen:
                        q_half = q.index_select(0, late_idx).contiguous()
                        out_step, lse_step = _flash_fwd(
                            q_half,
                            k_step,
                            v_step,
                            softmax_scale,
                            dropout_p,
                            False,
                            True,
                            cu_half_q,
                            cu_seqlens_k,
                            max_half_q,
                            max_seqlen_k,
                        )
                    else:
                        half = q.shape[seq_dim] // 2
                        q_half = q.narrow(seq_dim, half, half).contiguous()
                        out_step, lse_step = _flash_fwd(
                            q_half,
                            k_step,
                            v_step,
                            softmax_scale,
                            dropout_p,
                            False,
                            False,
                        )
                else:
                    # Diagonal: causal on full sequence
                    out_step, lse_step = _flash_fwd(
                        q,
                        k_step,
                        v_step,
                        softmax_scale,
                        dropout_p,
                        True,
                        is_varlen,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                    )

                computed_steps.append((i, section))
            else:
                # Non-causal or ringattn_size==1: no zigzag needed
                step_causal = False
                out_step, lse_step = _flash_fwd(
                    q,
                    k_step,
                    v_step,
                    softmax_scale,
                    dropout_p,
                    step_causal,
                    is_varlen,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                )
                computed_steps.append((i, "full"))

            # Merge output (all on compute_stream)
            if out is None:
                if zigzag and section == "upper":
                    # Upper-triangle: initialize full output, set late half
                    out = torch.zeros_like(q, dtype=out_step.dtype)
                    if is_varlen:
                        out[late_idx] = out_step
                        lse = torch.full(
                            (q.shape[1], q.shape[0]),
                            float("-inf"),
                            dtype=torch.float32,
                            device=q.device,
                        )
                        lse[:, late_idx] = lse_step
                    else:
                        out.narrow(seq_dim, half, half).copy_(out_step)
                        lse = torch.full(
                            (q.shape[0], q.shape[2], q.shape[1]),
                            float("-inf"),
                            dtype=torch.float32,
                            device=q.device,
                        )
                        lse[:, :, half:] = lse_step
                else:
                    out = out_step.clone()
                    lse = lse_step.clone()
            elif zigzag and section == "upper":
                # Merge only late half of each doc
                if is_varlen:
                    out_late = out[late_idx]
                    lse_late = lse[:, late_idx]
                    merged_out, merged_lse = _merge_attn_outputs(
                        out_late,
                        lse_late,
                        out_step,
                        lse_step,
                        True,
                    )
                    out[late_idx] = merged_out.to(out.dtype)
                    lse[:, late_idx] = merged_lse
                else:
                    half = q.shape[seq_dim] // 2
                    out_second = out.narrow(seq_dim, half, half)
                    lse_second = lse[:, :, half:]
                    merged_out, merged_lse = _merge_attn_outputs(
                        out_second,
                        lse_second,
                        out_step,
                        lse_step,
                        False,
                    )
                    out.narrow(seq_dim, half, half).copy_(merged_out)
                    lse[:, :, half:] = merged_lse
            else:
                out, lse = _merge_attn_outputs(
                    out,
                    lse,
                    out_step,
                    lse_step,
                    is_varlen,
                )

        # Cast output back to input dtype
        if out is not None:
            out = out.to(q.dtype)

        # Handle edge case: no steps computed (shouldn't happen with zigzag)
        if out is None:
            out = torch.zeros_like(q)
            if is_varlen:
                lse = torch.full(
                    (q.shape[1], q.shape[0]),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q.device,
                )
            else:
                lse = torch.full(
                    (q.shape[0], q.shape[2], q.shape[1]),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q.device,
                )

        # Save for backward
        save_tensors = [q, k, v, out, lse]
        if cu_seqlens_q is not None:
            save_tensors.extend([cu_seqlens_q, cu_seqlens_k])

        ctx.save_for_backward(*save_tensors)
        ctx.ringattn_group = ringattn_group
        ctx.ringattn_rank = ringattn_rank
        ctx.ringattn_size = ringattn_size
        ctx.softmax_scale = softmax_scale
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.is_varlen = is_varlen
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.computed_steps = computed_steps
        ctx.zigzag = zigzag
        ctx.early_idx = early_idx
        ctx.late_idx = late_idx

        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        ringattn_group = ctx.ringattn_group
        ringattn_rank = ctx.ringattn_rank
        ringattn_size = ctx.ringattn_size
        is_varlen = ctx.is_varlen
        zigzag = ctx.zigzag

        # Unpack saved tensors
        saved = ctx.saved_tensors
        q = saved[0]
        k_saved = saved[1]
        v_saved = saved[2]
        out = saved[3]
        lse = saved[4]

        cu_seqlens_q = cu_seqlens_k = None
        if is_varlen:
            cu_seqlens_q = saved[5]
            cu_seqlens_k = saved[6]

        seq_dim = 0 if is_varlen else 1  # varlen: [total, H, D], batched: [B, S, H, D]
        early_idx = ctx.early_idx
        late_idx = ctx.late_idx

        # Re-gather KV from all ring ranks
        k_all, v_all = _all_gather_kv(k_saved, v_saved, ringattn_group)

        # Precompute half cu_seqlens for varlen zigzag
        cu_half_q = cu_half_k = None
        max_half_q = max_half_k = None
        if zigzag and is_varlen:
            cu_half_q = cu_seqlens_q // 2
            cu_half_k = cu_seqlens_k // 2
            max_half_q = ctx.max_seqlen_q // 2 if ctx.max_seqlen_q else None
            max_half_k = ctx.max_seqlen_k // 2 if ctx.max_seqlen_k else None

        # Initialize gradient accumulators
        dq = torch.zeros_like(q)
        dk_all = [torch.zeros_like(k_all[i]) for i in range(ringattn_size)]
        dv_all = [torch.zeros_like(v_all[i]) for i in range(ringattn_size)]

        # Backward ring loop
        for step_i, section in ctx.computed_steps:
            kv_source = (ringattn_rank - step_i) % ringattn_size
            k_step = k_all[kv_source]
            v_step = v_all[kv_source]

            if zigzag:
                if section == "lower":
                    # All Q, early half KV (no mask)
                    if is_varlen:
                        k_half = k_step.index_select(0, early_idx).contiguous()
                        v_half = v_step.index_select(0, early_idx).contiguous()
                        dq_step = torch.empty_like(q)
                        dk_step = torch.empty_like(k_half)
                        dv_step = torch.empty_like(v_half)
                        _flash_bwd(
                            grad_output,
                            q,
                            k_half,
                            v_half,
                            out,
                            lse,
                            dq_step,
                            dk_step,
                            dv_step,
                            ctx.softmax_scale,
                            ctx.dropout_p,
                            False,
                            True,
                            cu_seqlens_q,
                            cu_half_k,
                            ctx.max_seqlen_q,
                            max_half_k,
                        )
                        dq.add_(dq_step)
                        # Scatter dk/dv back to early positions
                        dk_all[kv_source][early_idx] += dk_step
                        dv_all[kv_source][early_idx] += dv_step
                    else:
                        half = q.shape[seq_dim] // 2
                        k_half = k_step.narrow(seq_dim, 0, half).contiguous()
                        v_half = v_step.narrow(seq_dim, 0, half).contiguous()
                        dq_step = torch.empty_like(q)
                        dk_step = torch.empty_like(k_half)
                        dv_step = torch.empty_like(v_half)
                        _flash_bwd(
                            grad_output,
                            q,
                            k_half,
                            v_half,
                            out,
                            lse,
                            dq_step,
                            dk_step,
                            dv_step,
                            ctx.softmax_scale,
                            ctx.dropout_p,
                            False,
                            False,
                        )
                        dq.add_(dq_step)
                        dk_all[kv_source].narrow(seq_dim, 0, half).add_(dk_step)
                        dv_all[kv_source].narrow(seq_dim, 0, half).add_(dv_step)

                elif section == "upper":
                    # Late half Q, all KV (no mask)
                    if is_varlen:
                        q_half = q.index_select(0, late_idx).contiguous()
                        grad_half = grad_output.index_select(0, late_idx).contiguous()
                        out_half = out.index_select(0, late_idx).contiguous()
                        lse_half = lse[:, late_idx].contiguous()
                        dq_step = torch.empty_like(q_half)
                        dk_step = torch.empty_like(k_step)
                        dv_step = torch.empty_like(v_step)
                        _flash_bwd(
                            grad_half,
                            q_half,
                            k_step,
                            v_step,
                            out_half,
                            lse_half,
                            dq_step,
                            dk_step,
                            dv_step,
                            ctx.softmax_scale,
                            ctx.dropout_p,
                            False,
                            True,
                            cu_half_q,
                            cu_seqlens_k,
                            max_half_q,
                            ctx.max_seqlen_k,
                        )
                        dq[late_idx] += dq_step
                        dk_all[kv_source].add_(dk_step)
                        dv_all[kv_source].add_(dv_step)
                    else:
                        half = q.shape[seq_dim] // 2
                        q_half = q.narrow(seq_dim, half, half).contiguous()
                        grad_half = grad_output.narrow(seq_dim, half, half).contiguous()
                        out_half = out.narrow(seq_dim, half, half).contiguous()
                        lse_half = lse[:, :, half:].contiguous()
                        dq_step = torch.empty_like(q_half)
                        dk_step = torch.empty_like(k_step)
                        dv_step = torch.empty_like(v_step)
                        _flash_bwd(
                            grad_half,
                            q_half,
                            k_step,
                            v_step,
                            out_half,
                            lse_half,
                            dq_step,
                            dk_step,
                            dv_step,
                            ctx.softmax_scale,
                            ctx.dropout_p,
                            False,
                            False,
                        )
                        dq.narrow(seq_dim, half, half).add_(dq_step)
                        dk_all[kv_source].add_(dk_step)
                        dv_all[kv_source].add_(dv_step)

                else:
                    # Diagonal: full causal
                    dq_step = torch.empty_like(q)
                    dk_step = torch.empty_like(k_step)
                    dv_step = torch.empty_like(v_step)
                    _flash_bwd(
                        grad_output,
                        q,
                        k_step,
                        v_step,
                        out,
                        lse,
                        dq_step,
                        dk_step,
                        dv_step,
                        ctx.softmax_scale,
                        ctx.dropout_p,
                        True,
                        is_varlen,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_k,
                    )
                    dq.add_(dq_step)
                    dk_all[kv_source].add_(dk_step)
                    dv_all[kv_source].add_(dv_step)
            else:
                step_causal = section == "causal"
                dq_step = torch.empty_like(q)
                dk_step = torch.empty_like(k_step)
                dv_step = torch.empty_like(v_step)
                _flash_bwd(
                    grad_output,
                    q,
                    k_step,
                    v_step,
                    out,
                    lse,
                    dq_step,
                    dk_step,
                    dv_step,
                    ctx.softmax_scale,
                    ctx.dropout_p,
                    step_causal,
                    is_varlen,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                )
                dq.add_(dq_step)
                dk_all[kv_source].add_(dk_step)
                dv_all[kv_source].add_(dv_step)

        # Reduce-scatter dk, dv back to owning ranks
        dk = _reduce_scatter_grads(dk_all, ringattn_group)
        dv = _reduce_scatter_grads(dv_all, ringattn_group)

        return dq, dk, dv, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ring_flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    ringattn_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = True,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> Tensor:
    """Ring flash attention forward pass.

    Each ring rank holds a shard of the sequence. KV is rotated across the
    ring group via P2P, and flash attention is computed per-step with online
    LSE merging. Uses double-buffered CUDA streams for compute-comm overlap.

    For causal batched attention, zigzag redistribution ensures all
    ranks compute all steps for balanced work distribution.

    Args:
        q: query tensor [B, S_local, H_q, D] or [total_local, H_q, D] for varlen
        k: key tensor [B, S_local, H_kv, D] or [total_local, H_kv, D]
        v: value tensor, same shape as k
        ringattn_group: ring attention process group
        softmax_scale: attention scale (defaults to 1/sqrt(head_dim))
        dropout_p: dropout probability
        causal: whether to use causal attention
        cu_seqlens_q: cumulative sequence lengths for queries (varlen only)
        cu_seqlens_k: cumulative sequence lengths for keys (varlen only)
        max_seqlen_q: max query sequence length (varlen only)
        max_seqlen_k: max key sequence length (varlen only)

    Returns:
        Attention output, same shape as q.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # Handle batch dimension for varlen: the model pipeline produces 4D
    # [1, total, H, D] but _flash_attn_varlen_forward expects 3D [total, H, D].
    squeezed_batch = False
    if cu_seqlens_q is not None and q.ndim == 4 and q.shape[0] == 1:
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        squeezed_batch = True

    out = RingAttentionP2PFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        dropout_p,
        causal,
        ringattn_group,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )

    if squeezed_batch:
        out = out.unsqueeze(0)

    return out
