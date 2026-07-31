"""Trainable wrapper for the batch-invariant fused LM-head logprob contract.

Forward scores per-token cross-entropy through
:func:`xorl.ops.batch_invariant_ops.bi_lm_head_selected_logprob` — the K3
lm-head contract vendored identically in SGLang, so trainer and serving
logprobs are bitwise identical from bit-exact hidden states. The bf16 weight
stays resident (no fp32 lm-head copy) and only chunk-sized logits tiles ever
materialize. An optional scalar temperature scores ``log softmax(z/T)`` with
the ``1/T`` divide inside the contract kernel, matching serving's
temp-scaled input logprobs bitwise.

Backward is the closed-form CE gradient computed against the saved forward
``lse`` with chunked cuBLAS recompute (stock-numerics class, like the other
fused CE backwards — the contract governs the forward bits only). All three
recompute GEMMs run on the resident bf16 tensors with fp32 accumulation
(``out_dtype=float32``), so no fp32 copy of hidden or weight ever
materializes.
"""

import torch

from xorl.ops.batch_invariant_ops import BI_LM_HEAD_VOCAB_CHUNK, bi_lm_head_selected_logprob
from xorl.ops.bi_families_v2 import families_v2_enabled, head_v2_selected_logprob


class _BiFusedLmHeadPerTokenCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels_safe, valid_mask, temp_row, vocab_chunk):
        if families_v2_enabled():
            # head v2 (families-v2 migration): same GEMM K-chain, epilogue-stats
            # online LSE; logits never materialize; backward only consumes lse.
            logprob, lse, _ = head_v2_selected_logprob(hidden, weight, labels_safe, temperature=temp_row)
        else:
            logprob, lse, _ = bi_lm_head_selected_logprob(
                hidden, weight, labels_safe, temperature=temp_row, vocab_chunk=vocab_chunk
            )
        ctx.save_for_backward(hidden, weight, labels_safe, valid_mask, lse, temp_row)
        ctx.vocab_chunk = vocab_chunk
        return torch.where(valid_mask, -logprob, torch.zeros_like(logprob))

    @staticmethod
    def backward(ctx, grad_ce):
        hidden, weight, labels, valid_mask, lse, temp_row = ctx.saved_tensors
        vocab_chunk = ctx.vocab_chunk
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        need_h = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]

        g = (grad_ce * valid_mask).float()  # dCE/dy = g * (softmax(y) - onehot); y = z/T
        g_col = g.unsqueeze(1)
        lse_col = lse.unsqueeze(1)
        inv_t = None if temp_row is None else (1.0 / temp_row).unsqueeze(1)
        grad_h = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device) if need_h else None
        grad_w = torch.empty_like(weight) if need_w else None
        rows = torch.arange(n_tokens, device=hidden.device)

        for col_start in range(0, vocab, vocab_chunk):
            col_end = min(col_start + vocab_chunk, vocab)
            w_c = weight[col_start:col_end]
            # bf16 tensor-core GEMM, fp32 accumulate + fp32 out (no fp32 copies)
            logits_c = torch.mm(hidden, w_c.t(), out_dtype=torch.float32)
            if inv_t is not None:
                logits_c *= inv_t
            grad_z = logits_c.sub_(lse_col).exp_().mul_(g_col)
            in_chunk = (labels >= col_start) & (labels < col_end)
            grad_z[rows[in_chunk], labels[in_chunk] - col_start] -= g[in_chunk]
            if inv_t is not None:
                grad_z *= inv_t  # dy/dz = 1/T
            grad_z16 = grad_z.to(hidden.dtype)
            if need_h:
                torch.addmm(grad_h, grad_z16, w_c, out_dtype=torch.float32, out=grad_h)
            if need_w:
                grad_w[col_start:col_end] = torch.mm(grad_z16.t(), hidden, out_dtype=torch.float32).to(weight.dtype)

        return (
            grad_h.to(hidden.dtype) if need_h else None,
            grad_w if need_w else None,
            None,
            None,
            None,
            None,
        )


def bi_fused_per_token_ce(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    temperature: float = 1.0,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> torch.Tensor:
    """Per-token CE (``-log p(labels)``; 0 at ignored positions) through the
    batch-invariant lm-head contract. Requires CUDA bf16 hidden/weight; the
    fp32-class numerics come from the contract itself, so ``lm_head_fp32`` is
    implied rather than materialized. ``temperature != 1.0`` scores
    ``log softmax(z/T)`` via the in-kernel per-row scale (serving passes the
    same per-row fp32 temperatures, keeping the contract bitwise)."""
    if not hidden_states.is_cuda:
        raise ValueError("ce_mode='bi_fused' requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("ce_mode='bi_fused' requires bf16 hidden states and lm-head weight")
    if temperature <= 0:
        raise ValueError("ce_mode='bi_fused' requires temperature > 0")
    valid_mask = labels != ignore_index
    labels_safe = torch.where(valid_mask, labels, torch.zeros_like(labels))
    temp_row = None
    if temperature != 1.0:
        temp_row = torch.full(
            (hidden_states.shape[0],), float(temperature), dtype=torch.float32, device=hidden_states.device
        )
    return _BiFusedLmHeadPerTokenCE.apply(hidden_states, weight, labels_safe, valid_mask, temp_row, vocab_chunk)
