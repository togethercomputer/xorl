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

        # Single fused pass over vocab chunks. Reverse-KL decomposes as
        #   KL = sum_v p_s(v)*(s_v - t_v) - logZ_s + logZ_t
        # and sum_v p_s(v)*(s_v - t_v) = A / Z_s with
        #   A = sum_v exp(s_v - s_max) * (s_v - t_v),
        # which is online-accumulable (flash-attention style) alongside the
        # student/teacher log-sum-exp normalizers. This halves the forward GEMM
        # vs the prior two-pass form (a logsumexp pass + a probability pass)
        # while still never materializing the full logits. The backward
        # (recomputes logits from the saved normalizers) is unchanged, so the
        # gradients are identical up to fp32 summation order.
        s_max = torch.full((token_count, 1), neg_inf, device=student_hidden_states.device, dtype=torch.float32)
        t_max = torch.full((token_count, 1), neg_inf, device=student_hidden_states.device, dtype=torch.float32)
        s_sumexp = torch.zeros((token_count, 1), device=student_hidden_states.device, dtype=torch.float32)
        t_sumexp = torch.zeros((token_count, 1), device=student_hidden_states.device, dtype=torch.float32)
        weighted_diff = torch.zeros((token_count, 1), device=student_hidden_states.device, dtype=torch.float32)

        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
            s_logits = (student_hidden_states @ student_weight[start:end].t()).float()
            t_logits = (teacher_hidden_states @ t_weight.t()).float()
            # Student: online log-sum-exp + online sum_v exp(s_v - s_max)*(s_v - t_v).
            chunk_s_max = s_logits.max(dim=-1, keepdim=True).values
            new_s_max = torch.maximum(s_max, chunk_s_max)
            s_scale = torch.where(
                torch.isfinite(s_max),
                (s_max - new_s_max).exp(),
                torch.zeros_like(s_sumexp),
            )
            exp_s = (s_logits - new_s_max).exp()
            s_sumexp = s_sumexp * s_scale + exp_s.sum(dim=-1, keepdim=True)
            weighted_diff = weighted_diff * s_scale + (exp_s * (s_logits - t_logits)).sum(dim=-1, keepdim=True)
            s_max = new_s_max
            # Teacher: online log-sum-exp normalizer only.
            t_max, t_sumexp = _update_online_logsumexp(t_max, t_sumexp, t_logits)

        s_logz = s_sumexp.log() + s_max
        t_logz = t_sumexp.log() + t_max
        kl = (weighted_diff / s_sumexp - s_logz + t_logz).squeeze(-1)
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


@torch.no_grad()
def streaming_full_vocab_diagnostics(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight,
    vocab_chunk_size: int | None = 32768,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token (teacher_entropy, student_entropy, top1_agreement) diagnostics.

    Streams vocab chunks exactly like `_StreamingReverseKL` (one logsumexp pass,
    one probability pass) so the full logits are never materialized. No-grad and
    opt-in: this is the streaming-backend counterpart of the compile backend's
    `*_with_diag` kernels, costing one extra full-vocab pass per micro-batch.
    """
    if vocab_chunk_size is None:
        vocab_chunk_size = 32768
    elif vocab_chunk_size <= 0:
        vocab_chunk_size = int(student_weight.shape[0])
    vocab_size = int(student_weight.shape[0])
    token_count = int(student_hidden_states.shape[0])
    device = student_hidden_states.device
    neg_inf = -float("inf")

    s_max = torch.full((token_count, 1), neg_inf, device=device, dtype=torch.float32)
    t_max = torch.full((token_count, 1), neg_inf, device=device, dtype=torch.float32)
    s_sumexp = torch.zeros((token_count, 1), device=device, dtype=torch.float32)
    t_sumexp = torch.zeros((token_count, 1), device=device, dtype=torch.float32)
    s_best_idx = torch.zeros(token_count, device=device, dtype=torch.long)
    t_best_idx = torch.zeros(token_count, device=device, dtype=torch.long)
    s_best_val = torch.full((token_count,), neg_inf, device=device, dtype=torch.float32)
    t_best_val = torch.full((token_count,), neg_inf, device=device, dtype=torch.float32)

    for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
        s_weight = student_weight[start:end].to(student_hidden_states.dtype)
        t_weight = t_weight.to(teacher_hidden_states.dtype)
        s_logits = (student_hidden_states @ s_weight.t()).float()
        t_logits = (teacher_hidden_states @ t_weight.t()).float()
        s_max, s_sumexp = _update_online_logsumexp(s_max, s_sumexp, s_logits)
        t_max, t_sumexp = _update_online_logsumexp(t_max, t_sumexp, t_logits)
        s_chunk_val, s_chunk_idx = s_logits.max(dim=-1)
        t_chunk_val, t_chunk_idx = t_logits.max(dim=-1)
        s_better = s_chunk_val > s_best_val
        t_better = t_chunk_val > t_best_val
        s_best_idx = torch.where(s_better, s_chunk_idx + start, s_best_idx)
        s_best_val = torch.maximum(s_best_val, s_chunk_val)
        t_best_idx = torch.where(t_better, t_chunk_idx + start, t_best_idx)
        t_best_val = torch.maximum(t_best_val, t_chunk_val)

    s_logz = s_sumexp.log() + s_max
    t_logz = t_sumexp.log() + t_max
    s_plogp = torch.zeros(token_count, device=device, dtype=torch.float32)
    t_plogp = torch.zeros(token_count, device=device, dtype=torch.float32)
    for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
        s_weight = student_weight[start:end].to(student_hidden_states.dtype)
        t_weight = t_weight.to(teacher_hidden_states.dtype)
        s_logits = (student_hidden_states @ s_weight.t()).float()
        t_logits = (teacher_hidden_states @ t_weight.t()).float()
        s_log_probs = s_logits - s_logz
        t_log_probs = t_logits - t_logz
        s_plogp = s_plogp + (s_log_probs.exp() * s_log_probs).sum(dim=-1)
        t_plogp = t_plogp + (t_log_probs.exp() * t_log_probs).sum(dim=-1)

    if hasattr(teacher_weight, "clear_device_cache"):
        teacher_weight.clear_device_cache()

    teacher_entropy = -t_plogp
    student_entropy = -s_plogp
    top1_agreement = (s_best_idx == t_best_idx).float()
    return teacher_entropy, student_entropy, top1_agreement


def streaming_reverse_kl_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    vocab_chunk_size: int | None = 32768,
) -> torch.Tensor:
    """Compute per-token reverse KL without materializing full-vocab logits."""
    if vocab_chunk_size is None:
        vocab_chunk_size = 32768
    elif vocab_chunk_size <= 0:
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


class _StreamingReverseKLLowMem(torch.autograd.Function):
    """Memory-lean reverse KL(student||teacher), bit-exact with _StreamingReverseKL
    under an fp32 lm-head, but without holding full fp32 copies of the lm-head
    weights or a second full [V,H] grad buffer.

    Two memory levers, both gradient-identical to the current OPD fp32 path:

      1. Per-chunk fp32 upcast. The student/teacher lm-head weights stay in their
         native (bf16) dtype; each [chunk, H] vocab slice is upcast to
         `compute_dtype` (fp32) inside the loop, right before the matmul. Slicing
         commutes with the elementwise upcast, so the per-chunk fp32 matmul is
         identical to multiplying by a whole pre-upcast fp32 weight -- but the
         two ~2 GB fp32 weight copies (`weight.float()`) are never resident.

      2. In-place weight grad. The weight gradient is streamed straight into the
         leaf `student_weight.grad` (created lazily, in the weight's native
         dtype), one vocab chunk at a time, instead of allocating a full [V,H]
         buffer that autograd then adds into `.grad`. Vocab chunks partition the
         grad rows disjointly, so this is exact; it removes the full extra buffer
         (and its grad-accumulation doubling). `student_weight` must be the leaf
         parameter when `inplace_weight_grad=True`.

    This is the AMDAHL-029..033 1-node unblock: it removes ~2-4 GB of resident
    lm-head memory that tipped the streaming-KL backward over on one node.
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
        compute_dtype: torch.dtype,
        inplace_weight_grad: bool,
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
        dev = student_hidden_states.device
        sh = student_hidden_states.to(compute_dtype)
        th = teacher_hidden_states.to(compute_dtype)

        s_max = torch.full((token_count, 1), neg_inf, device=dev, dtype=torch.float32)
        t_max = torch.full((token_count, 1), neg_inf, device=dev, dtype=torch.float32)
        s_sumexp = torch.zeros((token_count, 1), device=dev, dtype=torch.float32)
        t_sumexp = torch.zeros((token_count, 1), device=dev, dtype=torch.float32)
        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
            s_logits = (sh @ student_weight[start:end].to(compute_dtype).t()).float()
            t_logits = (th @ t_weight.to(compute_dtype).t()).float()
            s_max, s_sumexp = _update_online_logsumexp(s_max, s_sumexp, s_logits)
            t_max, t_sumexp = _update_online_logsumexp(t_max, t_sumexp, t_logits)

        s_logz = s_sumexp.log() + s_max
        t_logz = t_sumexp.log() + t_max
        kl = torch.zeros(token_count, device=dev, dtype=torch.float32)
        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, vocab_chunk_size):
            s_logits = (sh @ student_weight[start:end].to(compute_dtype).t()).float()
            t_logits = (th @ t_weight.to(compute_dtype).t()).float()
            s_log_probs = s_logits - s_logz
            t_log_probs = t_logits - t_logz
            s_probs = s_log_probs.exp()
            kl = kl + (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)

        kl = kl * valid.to(kl.dtype)
        ctx.save_for_backward(student_hidden_states, student_weight, teacher_hidden_states, labels, s_logz, t_logz, kl)
        ctx.teacher_weight = teacher_weight
        ctx.ignore_index = ignore_index
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.compute_dtype = compute_dtype
        ctx.inplace_weight_grad = inplace_weight_grad
        ctx.student_weight_ref = student_weight
        return kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (student_hidden_states, student_weight, teacher_hidden_states, labels, s_logz, t_logz, kl) = ctx.saved_tensors
        teacher_weight = ctx.teacher_weight
        cdt = ctx.compute_dtype
        valid = (labels != ctx.ignore_index).to(dtype=torch.float32, device=grad_output.device)
        scale = grad_output.to(dtype=torch.float32) * valid
        vocab_size = int(student_weight.shape[0])
        sh = student_hidden_states.to(cdt)
        th = teacher_hidden_states.to(cdt)

        grad_hidden = None
        if ctx.needs_input_grad[0]:
            grad_hidden = torch.zeros_like(student_hidden_states, dtype=torch.float32)

        # Weight grad: either stream in place into the leaf .grad (no second full
        # buffer), or build one buffer in the weight's native dtype and return it.
        accumulate_weight = ctx.needs_input_grad[1] or ctx.inplace_weight_grad
        inplace = ctx.inplace_weight_grad
        grad_weight = None
        if accumulate_weight:
            if inplace:
                wparam = ctx.student_weight_ref
                if wparam.grad is None:
                    wparam.grad = torch.zeros_like(wparam)
            else:
                grad_weight = torch.zeros_like(student_weight)

        for start, end, t_weight in _iter_weight_chunks(teacher_weight, vocab_size, ctx.vocab_chunk_size):
            s_weight = student_weight[start:end].to(cdt)
            s_logits = (sh @ s_weight.t()).float()
            t_logits = (th @ t_weight.to(cdt).t()).float()
            s_log_probs = s_logits - s_logz
            t_log_probs = t_logits - t_logz
            s_probs = s_log_probs.exp()
            grad_logits = s_probs * (s_log_probs - t_log_probs - kl.unsqueeze(1))
            grad_logits = grad_logits * scale.unsqueeze(1)
            if grad_hidden is not None:
                grad_hidden = grad_hidden + grad_logits @ s_weight
            if accumulate_weight:
                chunk = (grad_logits.t() @ sh).to(student_weight.dtype)
                if inplace:
                    ctx.student_weight_ref.grad[start:end].add_(chunk)
                else:
                    grad_weight[start:end] = chunk

        if grad_hidden is not None:
            grad_hidden = grad_hidden.to(student_hidden_states.dtype)
        if hasattr(teacher_weight, "clear_device_cache"):
            teacher_weight.clear_device_cache()

        return grad_hidden, grad_weight, None, None, None, None, None, None, None


def streaming_reverse_kl_lowmem_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    vocab_chunk_size: int | None = 32768,
    compute_dtype: torch.dtype = torch.float32,
    inplace_weight_grad: bool = False,
) -> torch.Tensor:
    """Memory-lean per-token reverse KL: native-dtype lm-head weights upcast per
    vocab chunk to `compute_dtype`, gradient-identical to the fp32 path.

    Pass the lm-head weights in their stored (bf16) dtype -- do NOT pre-cast them
    to fp32. `student_hidden_states` may be fp32 (cheap; [N,H]). Set
    `inplace_weight_grad=True` only when `student_weight` is the leaf parameter
    whose `.grad` the optimizer reads (saves the second full [V,H] buffer).
    """
    if vocab_chunk_size is None:
        vocab_chunk_size = 32768
    elif vocab_chunk_size <= 0:
        vocab_chunk_size = int(student_weight.shape[0])
    if not math.isfinite(float(vocab_chunk_size)):
        raise ValueError(f"Invalid vocab_chunk_size={vocab_chunk_size}")
    return _StreamingReverseKLLowMem.apply(
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        labels,
        teacher_weight.detach() if torch.is_tensor(teacher_weight) else teacher_weight,
        int(ignore_index),
        int(vocab_chunk_size),
        compute_dtype,
        bool(inplace_weight_grad),
    )
