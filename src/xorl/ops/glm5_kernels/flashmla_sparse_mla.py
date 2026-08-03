"""Serving-exact FlashMLA forward with a separately gated training backward.

The GLM-5 zero-K3 contract is defined by the sampler's sparse-attention
forward bytes.  This autograd function therefore calls the same public
``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` entry point as serving and
saves its output and base-2 LSE.  FlashMLA does not expose a backward, so
training uses the existing TileLang sparse-MLA derivative kernel.  That
backward implements the derivative of the same sparse softmax attention,
but it is a separate trainability contract: its gradients are checked for
finiteness/correctness rather than included in the forward bitwise claim.

All-invalid query rows are compacted out before the TileLang call.  This
avoids its all-``-1`` NaN path and makes padding gradients exactly zero.
"""

from __future__ import annotations

import torch


def _run_flashmla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load FlashMLA lazily so CPU-only installs can still import XoRL."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd  # noqa: PLC0415

    return flash_mla_sparse_fwd(q, kv, indices, sm_scale=scaling, d_v=512)


def _run_tilelang_sparse_mla_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load TileLang only when a gradient is actually requested."""
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd  # noqa: PLC0415

    return sparse_mla_bwd(
        q,
        kv,
        out,
        grad_out,
        indices,
        lse,
        sm_scale=scaling,
    )


class FlashMLASparseWithTileLangBackward(torch.autograd.Function):
    """FlashMLA sparse prefill forward plus TileLang sparse-MLA backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        q = q.contiguous()
        kv = kv.contiguous()
        indices = indices.to(dtype=torch.int32).contiguous()

        out, _max_logits, lse = _run_flashmla_sparse_fwd(q, kv, indices, scaling)
        lse = lse.contiguous()
        valid_rows = indices.ge(0).any(dim=(1, 2))

        # FlashMLA already returns exact zeros for all-invalid rows.  Keep
        # that contract explicit here so a padded row can never expose a
        # backend-specific sentinel while valid rows remain untouched.
        out = torch.where(valid_rows[:, None, None], out, torch.zeros((), dtype=out.dtype, device=out.device))

        ctx.scaling = scaling
        ctx.save_for_backward(q, kv, indices, out, lse, valid_rows)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        q, kv, indices, out, lse, valid_rows = ctx.saved_tensors
        valid_row_indices = torch.nonzero(valid_rows, as_tuple=False).flatten()

        # TileLang's backward does not define the all-invalid softmax row.
        # Avoid invoking it when the entire input is padding.
        if valid_row_indices.numel() == 0:
            return torch.zeros_like(q), torch.zeros_like(kv), None, None

        q_valid = q.index_select(0, valid_row_indices).contiguous()
        indices_valid = indices.index_select(0, valid_row_indices).contiguous()
        out_valid = out.index_select(0, valid_row_indices).contiguous()
        lse_valid = lse.index_select(0, valid_row_indices).contiguous()
        grad_out_valid = grad_out.index_select(0, valid_row_indices).contiguous()

        dq_valid, dkv = _run_tilelang_sparse_mla_bwd(
            q_valid,
            kv,
            out_valid,
            grad_out_valid,
            indices_valid,
            lse_valid,
            ctx.scaling,
        )
        dq = torch.zeros_like(q)
        dq.index_copy_(0, valid_row_indices, dq_valid)
        return dq, dkv, None, None


__all__ = ["FlashMLASparseWithTileLangBackward"]
