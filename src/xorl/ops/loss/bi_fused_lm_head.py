"""Trainable wrapper for the batch-invariant fused LM-head logprob contract.

Forward scores per-token cross-entropy through
:func:`xorl.ops.batch_invariant_ops.bi_lm_head_selected_logprob` — the K3
lm-head contract vendored identically in SGLang, so trainer and serving
logprobs are bitwise identical from bit-exact hidden states. The bf16 weight
stays resident (no fp32 lm-head copy). Per-row temperature materializes the
same FP32 ``z * (1/T)`` tensor that serving samples and scores; the scalar-one
call keeps the original non-materialized path.

Backward is the closed-form CE gradient computed against the saved forward
``lse`` with chunked cuBLAS recompute (stock-numerics class, like the other
fused CE backwards — the contract governs the forward bits only). All three
recompute GEMMs run on the resident bf16 tensors with fp32 accumulation
(``out_dtype=float32``), so no fp32 copy of hidden or weight ever
materializes.
"""

import math

import torch

from xorl.ops.batch_invariant_ops import (
    BI_LM_HEAD_VOCAB_CHUNK,
    bi_lm_head_full_logits,
    bi_lm_head_selected_logprob,
    bi_lm_head_selected_logprob_from_logits,
)
from xorl.ops.bi_families_v2 import (
    exact_temperature_scale_fp32_logits,
    families_v2_enabled,
    head_v2_full_logits_with_lse,
    head_v2_selected_logprob,
    head_v2_selected_logprob_from_logits,
)
from xorl.ops.exact_sampling_transforms import exact_selected_logprob


_TEMPERATURE_MATERIALIZE_ROW_CHUNK = 32


class _BiFusedLmHeadPerTokenCE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        labels_safe,
        valid_mask,
        temp_row,
        top_ks,
        top_ps,
        min_ps,
        vocab_chunk,
    ):
        use_v2 = families_v2_enabled()
        has_sampling_filter = top_ks is not None
        support_chunks = []
        if temp_row is None and not has_sampling_filter and use_v2:
            # head v2 (families-v2 migration): same GEMM K-chain, epilogue-stats
            # online LSE; logits never materialize; backward only consumes lse.
            logprob, lse, _ = head_v2_selected_logprob(hidden, weight, labels_safe, temperature=temp_row)
        elif temp_row is None and not has_sampling_filter:
            logprob, lse, _ = bi_lm_head_selected_logprob(
                hidden, weight, labels_safe, temperature=temp_row, vocab_chunk=vocab_chunk
            )
        else:
            logprob_chunks = []
            lse_chunks = []
            for start in range(0, hidden.shape[0], _TEMPERATURE_MATERIALIZE_ROW_CHUNK):
                end = min(start + _TEMPERATURE_MATERIALIZE_ROW_CHUNK, hidden.shape[0])
                hidden_chunk = hidden[start:end]
                labels_chunk = labels_safe[start:end]
                temperature_chunk = None if temp_row is None else temp_row[start:end]
                if use_v2:
                    logits, _ = head_v2_full_logits_with_lse(hidden_chunk, weight, temperature=None)
                    transformed_logits = (
                        logits
                        if temperature_chunk is None
                        else exact_temperature_scale_fp32_logits(logits, temperature_chunk)
                    )
                    if has_sampling_filter:
                        logprob_chunk, lse_chunk, _, support_chunk = exact_selected_logprob(
                            transformed_logits,
                            labels_chunk,
                            top_ks[start:end],
                            top_ps[start:end],
                            min_ps[start:end],
                        )
                        support_chunks.append(support_chunk)
                    else:
                        logprob_chunk, lse_chunk, _ = head_v2_selected_logprob_from_logits(
                            transformed_logits,
                            labels_chunk,
                            temperature=None,
                        )
                else:
                    logits = bi_lm_head_full_logits(hidden_chunk, weight, vocab_chunk=vocab_chunk)
                    transformed_logits = (
                        logits
                        if temperature_chunk is None
                        else exact_temperature_scale_fp32_logits(logits, temperature_chunk)
                    )
                    if has_sampling_filter:
                        logprob_chunk, lse_chunk, _, support_chunk = exact_selected_logprob(
                            transformed_logits,
                            labels_chunk,
                            top_ks[start:end],
                            top_ps[start:end],
                            min_ps[start:end],
                        )
                        support_chunks.append(support_chunk)
                    else:
                        logprob_chunk, lse_chunk, _ = bi_lm_head_selected_logprob_from_logits(
                            transformed_logits,
                            labels_chunk,
                            temperature=None,
                            vocab_chunk=vocab_chunk,
                        )
                logprob_chunks.append(logprob_chunk)
                lse_chunks.append(lse_chunk)
            if logprob_chunks:
                logprob = torch.cat(logprob_chunks, dim=0)
                lse = torch.cat(lse_chunks, dim=0)
            else:
                logprob = torch.empty((0,), dtype=torch.float32, device=hidden.device)
                lse = logprob.clone()
        support = (
            torch.cat(support_chunks, dim=0)
            if support_chunks
            else torch.empty((0, 0), dtype=torch.bool, device=hidden.device)
        )
        ctx.save_for_backward(hidden, weight, labels_safe, valid_mask, lse, temp_row, support)
        ctx.vocab_chunk = vocab_chunk
        return torch.where(valid_mask, -logprob, torch.zeros_like(logprob))

    @staticmethod
    def backward(ctx, grad_ce):
        hidden, weight, labels, valid_mask, lse, temp_row, support = ctx.saved_tensors
        vocab_chunk = ctx.vocab_chunk
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        need_h = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]

        selected_support = (
            torch.ones_like(valid_mask) if support.numel() == 0 else support.gather(1, labels.unsqueeze(1)).squeeze(1)
        )
        g = (grad_ce * valid_mask * selected_support).float()
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
            if support.numel() != 0:
                grad_z *= support[:, col_start:col_end]
            in_chunk = selected_support & (labels >= col_start) & (labels < col_end)
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
            None,
            None,
            None,
        )


def bi_fused_per_token_ce(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    temperature: float | torch.Tensor = 1.0,
    top_ks: torch.Tensor | None = None,
    top_ps: torch.Tensor | None = None,
    min_ps: torch.Tensor | None = None,
    vocab_chunk: int = BI_LM_HEAD_VOCAB_CHUNK,
) -> torch.Tensor:
    """Per-token CE (``-log p(labels)``; 0 at ignored positions) through the
    batch-invariant lm-head contract. Requires CUDA bf16 hidden/weight; the
    fp32-class numerics come from the contract itself, so ``lm_head_fp32`` is
    implied rather than materialized. Per-row temperature scores the same
    materialized FP32 ``z * (1/T)`` tensor as serving."""
    if not hidden_states.is_cuda:
        raise ValueError("ce_mode='bi_fused' requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("ce_mode='bi_fused' requires bf16 hidden states and lm-head weight")
    valid_mask = labels != ignore_index
    labels_safe = torch.where(valid_mask, labels, torch.zeros_like(labels))
    if isinstance(temperature, torch.Tensor):
        if temperature.dtype is not torch.float32:
            raise TypeError("ce_mode='bi_fused' requires per-row FP32 temperature")
        if temperature.device != hidden_states.device or tuple(temperature.shape) != (hidden_states.shape[0],):
            raise ValueError("ce_mode='bi_fused' requires temperature aligned one-to-one with hidden-state rows")
        if not temperature.is_contiguous() or temperature.requires_grad:
            raise ValueError("ce_mode='bi_fused' requires contiguous, non-differentiable temperature metadata")
        torch._assert_async(
            (torch.isfinite(temperature) & (temperature > 0)).all(),
            "ce_mode='bi_fused' requires finite temperature > 0",
        )
        temp_row = temperature
    elif temperature == 1.0:
        temp_row = None
    else:
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("ce_mode='bi_fused' requires finite temperature > 0")
        temp_row = torch.full((hidden_states.shape[0],), temperature, dtype=torch.float32, device=hidden_states.device)
    if (top_ks is None, top_ps is None, min_ps is None).count(True) not in (0, 3):
        raise ValueError("ce_mode='bi_fused' requires all or none of top-k/top-p/min-p row metadata")
    return _BiFusedLmHeadPerTokenCE.apply(
        hidden_states,
        weight,
        labels_safe,
        valid_mask,
        temp_row,
        top_ks,
        top_ps,
        min_ps,
        vocab_chunk,
    )
