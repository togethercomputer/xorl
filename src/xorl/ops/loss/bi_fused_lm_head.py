"""Trainable wrapper for the batch-invariant fused LM-head logprob contract.

Forward scores per-token cross-entropy through
:func:`xorl.ops.batch_invariant_ops.bi_lm_head_selected_logprob` — the K3
lm-head contract vendored identically in SGLang, so trainer and serving
logprobs are bitwise identical from bit-exact hidden states. The bf16 weight
stays resident (no fp32 lm-head copy) and only chunk-sized logits tiles ever
materialize.

Backward is the closed-form CE gradient computed against the saved forward
``lse`` with chunked cuBLAS recompute (stock-numerics class, like the other
fused CE backwards — the contract governs the forward bits only).
"""

import torch

from xorl.ops.batch_invariant_ops import BI_LM_HEAD_VOCAB_CHUNK, bi_lm_head_selected_logprob
from xorl.ops.bi_families_v2 import families_v2_enabled, head_v2_selected_logprob


class _BiFusedLmHeadPerTokenCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels_safe, valid_mask, vocab_chunk):
        if families_v2_enabled():
            # Current path: projection and fixed-order vocabulary statistics
            # share one launch. The v1 chunked path is the kill-switch rollback
            # and uses the same pinned GEMM K chain. The switch is the shared
            # families-v2 flag pair, so one setting rolls back the trainer and
            # the sampler together; rolling back only one is invalid.
            logprob, lse, _ = head_v2_selected_logprob(hidden, weight, labels_safe)
        else:
            logprob, lse, _ = bi_lm_head_selected_logprob(hidden, weight, labels_safe, vocab_chunk=vocab_chunk)
        ctx.save_for_backward(hidden, weight, labels_safe, valid_mask, lse)
        ctx.vocab_chunk = vocab_chunk
        return torch.where(valid_mask, -logprob, torch.zeros_like(logprob))

    @staticmethod
    def backward(ctx, grad_ce):
        hidden, weight, labels, valid_mask, lse = ctx.saved_tensors
        vocab_chunk = ctx.vocab_chunk
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        need_h = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]

        h32 = hidden.float()
        g = (grad_ce * valid_mask).float()  # dCE/dz = g * (softmax(z) - onehot(y))
        grad_h = torch.zeros_like(h32) if need_h else None
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        rows = torch.arange(n_tokens, device=hidden.device)

        for col_start in range(0, vocab, vocab_chunk):
            col_end = min(col_start + vocab_chunk, vocab)
            w32_c = weight[col_start:col_end].float()
            grad_z = torch.exp(h32 @ w32_c.t() - lse.unsqueeze(1)) * g.unsqueeze(1)
            in_chunk = (labels >= col_start) & (labels < col_end)
            grad_z[rows[in_chunk], labels[in_chunk] - col_start] -= g[in_chunk]
            if need_h:
                grad_h += grad_z @ w32_c
            if need_w:
                grad_w[col_start:col_end] = grad_z.t() @ h32

        return (
            grad_h.to(hidden.dtype) if need_h else None,
            grad_w.to(weight.dtype) if need_w else None,
            None,
            None,
            None,
        )


def bi_fused_per_token_ce(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> torch.Tensor:
    """Per-token CE (``-log p(labels)``; 0 at ignored positions) through the
    batch-invariant lm-head contract. Requires CUDA bf16 hidden/weight; the
    fp32-class numerics come from the contract itself, so ``lm_head_fp32`` is
    implied rather than materialized."""
    if not hidden_states.is_cuda:
        raise ValueError("ce_mode='bi_fused' requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("ce_mode='bi_fused' requires bf16 hidden states and lm-head weight")
    valid_mask = labels != ignore_index
    labels_safe = torch.where(valid_mask, labels, torch.zeros_like(labels))
    return _BiFusedLmHeadPerTokenCE.apply(hidden_states, weight, labels_safe, valid_mask, vocab_chunk)
