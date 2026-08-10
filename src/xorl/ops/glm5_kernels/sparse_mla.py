"""tilelang-backed sparse-MLA autograd Function.

Mirrors miles' `SparseMLA.apply` shape: `q [seq_len, heads, dim+tail]`,
`kv [seq_len_kv, kv_group, dim+tail]`, `indices [seq_len, kv_group, topk]`,
returns `out [seq_len, heads, dim]`. Caller flattens the batch dim before
calling and unflattens after.

This module imports tilelang at import time, so it should only be loaded
behind the dispatch layer's lazy guard.
"""

import torch

from .tilelang_sparse_mla_bwd import sparse_mla_bwd
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface


class SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)
        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling
        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling)
        return tl_dq, tl_dkv, None, None
