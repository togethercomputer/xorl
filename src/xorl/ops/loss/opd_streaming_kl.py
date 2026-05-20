from __future__ import annotations

import math

import torch


def _iter_weight_chunks(teacher_weight, vocab_size: int, chunk_rows: int):
    if hasattr(teacher_weight, "iter_device_chunks"):
        yield from teacher_weight.iter_device_chunks(chunk_rows)
        return
    for start, end in _iter_ranges(vocab_size, chunk_rows):
        yield start, end, teacher_weight[start:end]


def _chunk_size(vocab_size: int, requested: int) -> int:
    if requested <= 0 or requested >= vocab_size:
        return vocab_size
    return requested


def _iter_ranges(vocab_size: int, requested_chunk_size: int):
    chunk_size = _chunk_size(vocab_size, requested_chunk_size)
    for start in range(0, vocab_size, chunk_size):
        yield start, min(start + chunk_size, vocab_size)


def _update_online_logsumexp(
    running_max: torch.Tensor,
    running_sumexp: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_max = logits.max(dim=-1, keepdim=True).values
    new_max = torch.maximum(running_max, chunk_max)
    prev_scale = torch.where(
        torch.isfinite(running_max),
        (running_max - new_max).exp(),
        torch.zeros_like(running_sumexp),
    )
    chunk_sumexp = (logits - new_max).exp().sum(dim=-1, keepdim=True)
    return new_max, running_sumexp * prev_scale + chunk_sumexp


class _StreamingReverseKL(torch.autograd.Function):
    """Exact KL(student || teacher) over vocab chunks.

    This is the TileLang-facing OPD path: it exposes the same execution shape a
    native kernel will use (stream vocab blocks, save only per-token statistics,
    recompute logits in backward) while keeping a pure PyTorch implementation as
    the portable fallback.
    """

    @staticmethod
    def forward(
        ctx,
        student_hidden_states: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_hidden_states: torch.Tensor,
        labels: torch.Tensor,
        teacher_weight,
        ignore_index: int,
        vocab_chunk_size: int,
    ) -> torch.Tensor:
        teacher_shape = tuple(int(x) for x in teacher_weight.shape)
        if student_weight.shape[0] != teacher_shape[0]:
            raise ValueError(
                f"student vocab size ({student_weight.shape[0]}) must match teacher vocab size ({teacher_shape[0]})"
            )

        vocab_size = int(student_weight.shape[0])
        token_count = int(student_hidden_states.shape[0])
        valid = labels != ignore_index
        neg_inf = -float("inf")

        s_max = torch.full((token_count, 1), neg_inf, device=student_hidden_states.device, dtype=torch.float32)
        t_max = torch.full((token_count, 1), neg_inf, device=student_hidden_states.device, dtype=torch.float32)
        s_sumexp = torch.zeros((token_count, 1), device=student_hidden_states.device, dtype=torch.float32)
        t_sumexp = torch.zeros((token_count, 1), device=student_hidden_states.device, dtype=torch.float32)

        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
            s_logits = (student_hidden_states @ student_weight[start:end].t()).float()
            t_logits = (teacher_hidden_states @ t_weight.t()).float()
            s_max, s_sumexp = _update_online_logsumexp(s_max, s_sumexp, s_logits)
            t_max, t_sumexp = _update_online_logsumexp(t_max, t_sumexp, t_logits)

        s_logz = s_sumexp.log() + s_max
        t_logz = t_sumexp.log() + t_max
        kl = torch.zeros(token_count, device=student_hidden_states.device, dtype=torch.float32)
        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
            s_logits = (student_hidden_states @ student_weight[start:end].t()).float()
            t_logits = (teacher_hidden_states @ t_weight.t()).float()
            s_log_probs = s_logits - s_logz
            t_log_probs = t_logits - t_logz
            s_probs = s_log_probs.exp()
            kl = kl + (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)

        kl = kl * valid.to(kl.dtype)
        ctx.save_for_backward(
            student_hidden_states,
            student_weight,
            teacher_hidden_states,
            labels,
            s_logz,
            t_logz,
            kl,
        )
        ctx.teacher_weight = teacher_weight
        ctx.ignore_index = ignore_index
        ctx.vocab_chunk_size = vocab_chunk_size
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            student_hidden_states,
            student_weight,
            teacher_hidden_states,
            labels,
            s_logz,
            t_logz,
            kl,
        ) = ctx.saved_tensors
        teacher_weight = ctx.teacher_weight

        valid = (labels != ctx.ignore_index).to(dtype=torch.float32, device=grad_output.device)
        scale = grad_output.to(dtype=torch.float32) * valid
        vocab_size = int(student_weight.shape[0])

        grad_hidden = None
        if ctx.needs_input_grad[0]:
            grad_hidden = torch.zeros_like(student_hidden_states, dtype=torch.float32)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(student_weight, dtype=torch.float32)

        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, ctx.vocab_chunk_size):
            s_weight = student_weight[start:end]
            s_logits = (student_hidden_states @ s_weight.t()).float()
            t_logits = (teacher_hidden_states @ t_weight.t()).float()
            s_log_probs = s_logits - s_logz
            t_log_probs = t_logits - t_logz
            s_probs = s_log_probs.exp()

            # d KL(p_s || p_t) / d student_logits_i =
            # p_s_i * (log p_s_i - log p_t_i - KL)
            grad_logits = s_probs * (s_log_probs - t_log_probs - kl.unsqueeze(1))
            grad_logits = grad_logits * scale.unsqueeze(1)

            if grad_hidden is not None:
                grad_hidden = grad_hidden + grad_logits @ s_weight.float()
            if grad_weight is not None:
                grad_weight[start:end] = grad_logits.t() @ student_hidden_states.float()

        if grad_hidden is not None:
            grad_hidden = grad_hidden.to(student_hidden_states.dtype)
        if grad_weight is not None:
            grad_weight = grad_weight.to(student_weight.dtype)

        if hasattr(teacher_weight, "clear_device_cache"):
            teacher_weight.clear_device_cache()

        return grad_hidden, grad_weight, None, None, None, None, None


def streaming_reverse_kl_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    vocab_chunk_size: int = 32768,
) -> torch.Tensor:
    """Compute per-token reverse KL without materializing full-vocab logits."""
    if vocab_chunk_size <= 0:
        vocab_chunk_size = int(student_weight.shape[0])
    if not math.isfinite(float(vocab_chunk_size)):
        raise ValueError(f"Invalid vocab_chunk_size={vocab_chunk_size}")
    return _StreamingReverseKL.apply(
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        labels,
        teacher_weight.detach() if torch.is_tensor(teacher_weight) else teacher_weight,
        int(ignore_index),
        int(vocab_chunk_size),
    )
