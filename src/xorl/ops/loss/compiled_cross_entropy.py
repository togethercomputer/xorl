"""
Compiled Cross-Entropy Computation.

This module provides memory-efficient cross-entropy computation using
torch.compile with auto_chunker to avoid materializing large logits tensors.
"""

from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Default token-dim chunk count for the compiled/chunked CE. Shared so every path that chunks
# CE uses the same value: the auto_chunker loss path (causallm_loss_function's num_chunks
# default) and the whole-step traceable unroll (traceable_chunked_cross_entropy).
DEFAULT_NUM_CHUNKS = 8


# Cache for compiled cross-entropy functions
_compiled_ce_cache: Dict[int, Callable] = {}

# Cache for compiled CE+LSE^2 (Z-loss) functions
_compiled_ce_and_lse_sq_cache: Dict[int, Callable] = {}

# Cache for compiled OPD reverse-KL functions
_compiled_reverse_kl_cache: Dict[int, Callable] = {}

# Check if auto_chunker is available
_AUTO_CHUNKER_AVAILABLE = None


def _check_auto_chunker_available() -> bool:
    """Check if torch.compile auto_chunker option is available."""
    global _AUTO_CHUNKER_AVAILABLE
    if _AUTO_CHUNKER_AVAILABLE is not None:
        return _AUTO_CHUNKER_AVAILABLE

    try:

        @torch.compile(options={"auto_chunker.enable": True, "auto_chunker.num_chunk": 2})
        def _test_fn(x):
            return x * 2

        # Don't actually run, just check if compilation setup works
        _AUTO_CHUNKER_AVAILABLE = True
    except (RuntimeError, TypeError):
        _AUTO_CHUNKER_AVAILABLE = False

    return _AUTO_CHUNKER_AVAILABLE


def compiled_cross_entropy_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 64,
    reduction: str = "none",
    lm_head_fp32: bool = False,
) -> torch.Tensor:
    """
    Compute memory-efficient cross-entropy using torch.compile.

    Uses torch.compile (with auto_chunker if available) to avoid materializing
    the full [batch_size, vocab_size] logits tensor, reducing memory usage.

    Args:
        hidden_states: Flattened hidden states, shape (batch * seq_len, hidden_dim)
        weight: LM head weight matrix, shape (vocab_size, hidden_dim)
        labels: Flattened target labels, shape (batch * seq_len,)
        ignore_index: Index to ignore in loss computation (default: -100)
        num_chunks: Number of chunks for auto_chunker (default: 64). Higher values use
                   less memory but may have more overhead.
        reduction: Loss reduction mode - "none", "mean", or "sum" (default: "none")

    Returns:
        If reduction="none": per-token cross-entropy loss, shape (batch * seq_len,)
        If reduction="mean": scalar mean loss
        If reduction="sum": scalar sum loss
    """
    if lm_head_fp32:
        hidden_states = hidden_states.float()
        weight = weight.float()
    compute_ce_fn = _get_compiled_ce_fn(num_chunks, reduction)
    return compute_ce_fn(hidden_states, weight, labels, ignore_index)


def compiled_ce_and_lse_sq_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 64,
    lm_head_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token cross-entropy AND per-token logsumexp(logits)^2 in one fused pass.

    Used for the Z-loss auxiliary term. Without auto_chunker we'd have to
    materialize the [batch*seq, vocab] logits tensor twice (once for CE, once
    for LSE), so we co-compute them inside the same compiled region.

    Returns:
        (per_token_ce, per_token_lse_sq) — both shape (batch * seq_len,).
        per_token_lse_sq is zero at ignored-index positions.
    """
    if lm_head_fp32:
        hidden_states = hidden_states.float()
        weight = weight.float()
    fn = _get_compiled_ce_and_lse_sq_fn(num_chunks)
    return fn(hidden_states, weight, labels, ignore_index)


def compiled_reverse_kl_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 64,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
) -> torch.Tensor:
    """Compute per-token KL(student || teacher) without materializing full logits twice.

    Args:
        student_hidden_states: Flattened student hidden states, shape [tokens, student_hidden_dim].
        student_weight: Student LM head, shape [vocab_size, student_hidden_dim].
        teacher_hidden_states: Flattened teacher hidden states, shape [tokens, teacher_hidden_dim].
        teacher_weight: Teacher LM head, shape [vocab_size, teacher_hidden_dim].
        labels: Flattened labels, shape [tokens]. Only used to zero ignored positions.
        ignore_index: Label value to mask out of the returned per-token KL.
        num_chunks: Number of token chunks for torch.compile auto_chunker. 0 disables chunking.
        lm_head_fp32: Cast student hidden/head tensors to fp32 before matmul.
        teacher_lm_head_fp32: Cast teacher hidden/head tensors to fp32 before matmul.

    Returns:
        Per-token reverse KL, shape [tokens], with zero at ignored-index positions.
    """
    if lm_head_fp32:
        student_hidden_states = student_hidden_states.float()
        student_weight = student_weight.float()
    if teacher_lm_head_fp32:
        teacher_hidden_states = teacher_hidden_states.float()
        teacher_weight = teacher_weight.float()
    if student_hidden_states.device.type == "cpu":
        return _compute_reverse_kl(
            student_hidden_states,
            student_weight,
            teacher_hidden_states.detach(),
            teacher_weight.detach(),
            labels,
            ignore_index,
        )
    fn = _get_compiled_reverse_kl_fn(num_chunks)
    return fn(
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        teacher_weight.detach(),
        labels,
        ignore_index,
    )


def _compute_reverse_kl(
    student_hidden_states,
    student_weight,
    teacher_hidden_states,
    teacher_weight,
    labels,
    ignore_index,
):
    student_logits = (student_hidden_states @ student_weight.t()).float()
    teacher_logits = (teacher_hidden_states @ teacher_weight.t()).float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    token_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    valid = (labels != ignore_index).to(token_kl.dtype)
    return token_kl * valid


def _get_compiled_ce_fn(num_chunks: int, reduction: str = "none") -> Callable:
    """
    Get or create a compiled cross-entropy function.

    Uses torch.compile with auto_chunker to avoid materializing the full
    [batch_size, vocab_size] logits tensor, significantly reducing memory usage.

    If auto_chunker is not available (PyTorch < nightly), falls back to
    standard torch.compile without chunking.

    Args:
        num_chunks: Number of chunks for auto_chunker. Higher values use less
                   memory but may have more overhead. 0 disables chunking.
        reduction: Loss reduction mode - "none", "mean", or "sum".

    Returns:
        Compiled function that computes cross-entropy.
    """
    cache_key = (num_chunks, reduction)
    if cache_key not in _compiled_ce_cache:

        def _compute_ce(hidden_states, weight, labels, ignore_index):
            logits = (hidden_states @ weight.t()).float()
            return F.cross_entropy(logits, labels, reduction=reduction, ignore_index=ignore_index)

        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_ce_cache[cache_key] = torch.compile(
                _compute_ce,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_ce_cache[cache_key] = torch.compile(_compute_ce)
    return _compiled_ce_cache[cache_key]


def _get_compiled_ce_and_lse_sq_fn(num_chunks: int) -> Callable:
    """Get or create a compiled CE+LSE^2 function (chunked along the token dim)."""
    cache_key = num_chunks
    if cache_key not in _compiled_ce_and_lse_sq_cache:

        def _compute_ce_and_lse_sq(hidden_states, weight, labels, ignore_index):
            logits = (hidden_states @ weight.t()).float()
            per_token_ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
            lse = torch.logsumexp(logits, dim=-1)
            valid = (labels != ignore_index).to(lse.dtype)
            per_token_lse_sq = (lse * lse) * valid
            return per_token_ce, per_token_lse_sq

        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_ce_and_lse_sq_cache[cache_key] = torch.compile(
                _compute_ce_and_lse_sq,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_ce_and_lse_sq_cache[cache_key] = torch.compile(_compute_ce_and_lse_sq)
    return _compiled_ce_and_lse_sq_cache[cache_key]


def _get_compiled_reverse_kl_fn(num_chunks: int) -> Callable:
    """Get or create a compiled reverse-KL function (chunked along the token dim)."""
    cache_key = num_chunks
    if cache_key not in _compiled_reverse_kl_cache:
        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_reverse_kl_cache[cache_key] = torch.compile(
                _compute_reverse_kl,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_reverse_kl_cache[cache_key] = torch.compile(_compute_reverse_kl)
    return _compiled_reverse_kl_cache[cache_key]


def traceable_chunked_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    reduction: str = "sum",
) -> torch.Tensor:
    """Chunked linear cross-entropy with **no inner torch.compile**.

    Unlike ``compiled_cross_entropy_function`` (which wraps an ``auto_chunker`` inductor
    pass in its own ``torch.compile`` and therefore can't be nested inside another compiled
    region), this is plain, traceable PyTorch: a **static Python unroll** over ``num_chunks``
    token chunks. Each chunk computes ``logits = (hidden_chunk @ weightᵀ).float()`` →
    ``F.cross_entropy`` and is wrapped in activation checkpointing so the backward
    **recomputes** the chunk's logits instead of stashing them (matching what ``auto_chunker``
    does internally). Peak logits stay at one chunk — ``[ceil(N/num_chunks), vocab]`` — in
    *both* forward and backward.

    Because there is no inner ``torch.compile`` and (under static shapes) the loop unrolls at
    trace time, Dynamo folds this into an enclosing compiled region — e.g. the whole-step
    graph (backbone + lm_head + CE) — so the chunked CE no longer has to live in a separate
    graph the way the ``auto_chunker`` path does.

    Args:
        hidden_states: Flattened hidden states, shape ``[N, hidden]`` (``N = batch * seq``).
        weight: LM-head weight, shape ``[vocab, hidden]``.
        labels: Flattened next-token labels, shape ``[N]``.
        ignore_index: Label value excluded from the loss (default -100).
        num_chunks: Number of token chunks to unroll over. ``<= 1`` means no chunking.
        reduction: ``"sum"`` (default) or ``"mean"`` → scalar; ``"none"`` → per-token ``[N]``.
            ``"mean"`` divides by the number of valid (non-ignored) tokens, matching
            ``F.cross_entropy(reduction="mean")``.

    Returns:
        Scalar loss for ``"sum"``/``"mean"``, or per-token loss ``[N]`` for ``"none"``.
    """
    n = int(hidden_states.shape[0])
    chunks = max(1, num_chunks)
    # ceil division → static Python int so the loop below unrolls under torch.compile.
    chunk_size = (n + chunks - 1) // chunks

    def _chunk_loss(h_chunk: torch.Tensor, l_chunk: torch.Tensor) -> torch.Tensor:
        logits = (h_chunk @ weight.t()).float()
        return F.cross_entropy(logits, l_chunk, reduction="none", ignore_index=ignore_index)

    # Recompute per chunk in backward (instead of saving every chunk's logits) only when there
    # is a grad to compute; in inference there's nothing to recompute, so skip the overhead.
    use_checkpoint = torch.is_grad_enabled() and (
        hidden_states.requires_grad or weight.requires_grad
    )

    parts = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        h_chunk = hidden_states[start:end]
        l_chunk = labels[start:end]
        if use_checkpoint:
            ce_chunk = checkpoint(_chunk_loss, h_chunk, l_chunk, use_reentrant=False)
        else:
            ce_chunk = _chunk_loss(h_chunk, l_chunk)
        parts.append(ce_chunk)

    per_token = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    if reduction == "none":
        return per_token
    if reduction == "sum":
        return per_token.sum()
    if reduction == "mean":
        valid = (labels != ignore_index).sum().clamp(min=1)
        return per_token.sum() / valid
    raise ValueError(f"reduction must be 'none', 'sum', or 'mean', got {reduction!r}")
