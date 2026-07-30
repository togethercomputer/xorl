import torch


# Kernel modules are imported lazily inside ``DeepSeekV4SparseAttention`` so
# that hosts without ``tilelang`` (e.g. CPU CI) can still import this module
# to use the pure-torch references ``sparse_attn_torch`` / ``dense_attn_torch``.


def sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """
    Args:
        q: (b, m, h, d)
        kv: (b, n, d)
        attn_sink: (h,)
        topk_idxs: (b, m, topk)
        sm_scale: float
    Returns:
        o: (b, m, h, d) cast back to ``q``'s original dtype.
    """
    out_dtype = q.dtype
    q = q.float()
    kv = kv.float()

    b, m, h, d = q.shape
    k_len = kv.shape[1]
    _, _, topk = topk_idxs.shape

    assert (topk_idxs < k_len).all(), f"topk_idxs should be smaller than length of k: {k_len}, but got {topk_idxs}"

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    mask = topk_idxs != -1
    safe_idxs = topk_idxs.masked_fill(~mask, 0)

    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1)

    kv_gathered = kv[batch_idx, safe_idxs]

    scores = torch.einsum("bmhd,bmkd->bmhk", q, kv_gathered)

    scores = scores * sm_scale
    mask_expanded = mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~mask_expanded, float("-inf"))

    scores = scores.to(torch.float32)
    scores_max = scores.max(dim=-1).values
    # Numerical safety: when every key at a position is masked (all
    # ``topk_idxs == -1``), ``scores_max`` is ``-inf`` and the standard
    # ``scores - scores_max`` shift produces ``-inf - (-inf) = nan``. Replace
    # ``-inf`` rows with 0 so ``exp_scores`` is all zero at that position; the
    # ``attn_sink`` term carries the softmax denominator and the final output
    # is 0 (the only correct value when nothing was attended to).
    scores_max = torch.where(torch.isfinite(scores_max), scores_max, torch.zeros_like(scores_max))
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))

    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, kv_gathered.to(torch.float32))

    sum_exp = exp_scores.sum(dim=-1)
    # Promote attn_sink to fp32 to match the rest of the math (it's nominally
    # fp32 in the model but the bf16-cast smoke path can land it in bf16).
    sink_term = torch.exp(attn_sink.float().view(1, 1, h) - scores_max)
    denominator = sum_exp + sink_term

    o = numerator / denominator.unsqueeze(-1)

    return o.to(out_dtype)


def dense_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """
    Dense GEMM implementation: converts topk_idxs to a sparse mask and computes
    full Q @ K^T attention, then applies mask. No gather operations.
    """
    b, m, h, d = q.shape
    n = kv.shape[1]
    _, _, topk = topk_idxs.shape

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    attn_mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)

    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    seq_idx = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)

    valid_mask = topk_idxs != -1

    valid_batch = batch_idx[valid_mask]
    valid_seq = seq_idx[valid_mask]
    valid_kv_idx = topk_idxs[valid_mask].long()
    attn_mask[valid_batch, valid_seq, valid_kv_idx] = True

    scores = torch.einsum("bmhd,bnd->bmhn", q, kv).to(torch.float32) * sm_scale

    attn_mask_expanded = attn_mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~attn_mask_expanded, float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = scores_max.clamp(min=-1e30)

    exp_scores = torch.exp(scores - scores_max)

    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kv.float())

    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, h) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term

    o = numerator / denominator.unsqueeze(-1)

    return o.to(q.dtype)


class DeepSeekV4SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None):
        from .kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd  # noqa: PLC0415

        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx, do):
        from .kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd  # noqa: PLC0415

        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
        )

        return dq, dkv, d_attn_sink, None, None


def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)
