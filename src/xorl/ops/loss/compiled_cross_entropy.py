"""
Compiled Cross-Entropy Computation.

This module provides memory-efficient cross-entropy computation using
torch.compile with auto_chunker to avoid materializing large logits tensors.
"""

from typing import Callable, Dict

import torch
import torch.nn.functional as F


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
