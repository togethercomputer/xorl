"""tilelang-backed lightning indexer autograd Function.

Mirrors miles' `lighting_indexer` shape: `index_q [seq_len, head_num,
dim]`, `index_k [seq_len, dim]`, `weights [seq_len, head_num]` (after
miles' `.squeeze(-1)`). Returns `(top-k scores, top-k indices)` with
`indices` already masked to ``-1`` where the corresponding indexer logit
was ``-inf``.
"""

import torch

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface


def _extract_topk_scores(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe_indices)
    return torch.where(valid_mask, scores, float("-inf"))


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        _, head_num, _ = index_q.shape
        logits = indexer_fwd_interface(
            index_q,
            index_k,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=True,
        )
        if topk_indices is None:
            index_score, topk_indices = torch.topk(logits, topk, dim=-1)
            topk_indices = topk_indices.to(torch.int32)
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)

        index_score = _extract_topk_scores(logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        ctx.topk = topk
        ctx.head_num = head_num
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    return IndexerFunction.apply(index_q, index_k, weights.squeeze(-1), cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices)


def generate_varlen_mask_params(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    return starts, ends
