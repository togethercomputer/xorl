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

# Cache for compiled OPD reverse-KL + full-vocab diagnostic functions
_compiled_reverse_kl_with_diag_cache: Dict[int, Callable] = {}

# Cache for compiled OPD forward-KL (full-vocab) functions
_compiled_forward_kl_full_cache: Dict[int, Callable] = {}

# Cache for compiled OPD forward-KL (full-vocab) + diagnostic functions
_compiled_forward_kl_full_with_diag_cache: Dict[int, Callable] = {}

# Cache for compiled OPD sampled-token logprob extractor (for KL estimators)
_compiled_sampled_logp_cache: Dict[int, Callable] = {}

# Sentinel passed to clamp_min that effectively disables the clamp. Using a finite
# value (rather than None) keeps the compiled function signature stable.
_NO_CLAMP_SENTINEL = -1.0e30

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


def _full_vocab_diagnostics(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid: torch.Tensor,
):
    """Per-token full-vocab diagnostics.

    Returns per-token tensors:
        teacher_entropy: H(p_T) = -Σ p_T * log p_T. Higher == teacher less confident.
        student_entropy: H(p_S) = -Σ p_S * log p_S. Higher == student less confident.
        top1_agreement: 1.0 if argmax(p_S) == argmax(p_T), else 0.0. The full-vocab
            replacement for the top-k overlap metric — it's well-defined regardless
            of any truncation choice and answers "do they pick the same token?".

    All outputs are zeroed at ignore-index positions via `valid`.
    """
    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1) * valid
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1) * valid
    teacher_top1 = teacher_log_probs.argmax(dim=-1)
    student_top1 = student_log_probs.argmax(dim=-1)
    top1_agreement = (teacher_top1 == student_top1).to(teacher_entropy.dtype) * valid
    return teacher_entropy, student_entropy, top1_agreement


def _compute_reverse_kl_with_diag(
    student_hidden_states,
    student_weight,
    teacher_hidden_states,
    teacher_weight,
    labels,
    ignore_index,
):
    """Reverse KL(student||teacher) + full-vocab diagnostics in one pass.

    Returns (token_kl, teacher_entropy, student_entropy, top1_agreement).
    """
    student_logits = (student_hidden_states @ student_weight.t()).float()
    teacher_logits = (teacher_hidden_states @ teacher_weight.t()).float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    token_kl = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    valid = (labels != ignore_index).to(token_kl.dtype)
    token_kl = token_kl * valid
    teacher_entropy, student_entropy, top1_agreement = _full_vocab_diagnostics(
        student_log_probs, teacher_log_probs, valid
    )
    return token_kl, teacher_entropy, student_entropy, top1_agreement


def _compute_forward_kl_full(
    student_hidden_states,
    student_weight,
    teacher_hidden_states,
    teacher_weight,
    labels,
    ignore_index,
    log_prob_min_clamp: float,
):
    """Full-vocabulary forward KL: Σ_v p_T(v) · (log p_T(v) − log p_S(v)).

    This is the formula VERL would have shipped as kl_penalty_forward("full") if
    their data pipeline had access to the full teacher distribution; in xorl, we
    project teacher hidden states through the cached teacher LM head locally so
    the full p_T is available at loss time.

    `log_prob_min_clamp` of `_NO_CLAMP_SENTINEL` (= -1e30) effectively disables
    the clamp.
    """
    student_logits = (student_hidden_states @ student_weight.t()).float()
    teacher_logits = (teacher_hidden_states @ teacher_weight.t()).float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs_clamped = student_log_probs.clamp_min(log_prob_min_clamp)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs_clamped)).sum(dim=-1)
    valid = (labels != ignore_index).to(token_kl.dtype)
    return token_kl * valid


def _compute_forward_kl_full_with_diag(
    student_hidden_states,
    student_weight,
    teacher_hidden_states,
    teacher_weight,
    labels,
    ignore_index,
    log_prob_min_clamp: float,
):
    """Forward KL (full-vocab) + diagnostics in one pass.

    Returns (token_kl, teacher_entropy, student_entropy, top1_agreement).
    """
    student_logits = (student_hidden_states @ student_weight.t()).float()
    teacher_logits = (teacher_hidden_states @ teacher_weight.t()).float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs_clamped = student_log_probs.clamp_min(log_prob_min_clamp)
    teacher_probs = teacher_log_probs.exp()
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs_clamped)).sum(dim=-1)
    valid = (labels != ignore_index).to(token_kl.dtype)
    token_kl = token_kl * valid
    teacher_entropy, student_entropy, top1_agreement = _full_vocab_diagnostics(
        student_log_probs, teacher_log_probs, valid
    )
    return token_kl, teacher_entropy, student_entropy, top1_agreement


def _compute_sampled_token_logprobs(
    student_hidden_states,
    student_weight,
    teacher_hidden_states,
    teacher_weight,
    labels,
    ignore_index,
):
    """Per-token student & teacher log-probs of the *sampled* token (= labels position).

    Used by single-sample KL estimators (k1/k2/k3/abs/mse/low_var_kl). Returns
    (student_logp_at_label, teacher_logp_at_label), both zeroed at ignore-index.

    Note: labels is used as both the mask AND the gather index. Callers must
    pass safe gather indices (negative values would crash gather); the standard
    pattern is to pass `labels.clamp_min(0)` and rely on the `valid` mask.
    """
    student_logits = (student_hidden_states @ student_weight.t()).float()
    teacher_logits = (teacher_hidden_states @ teacher_weight.t()).float()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    gather_idx = labels.clamp_min(0).unsqueeze(-1)
    student_lp = torch.gather(student_log_probs, dim=-1, index=gather_idx).squeeze(-1)
    teacher_lp = torch.gather(teacher_log_probs, dim=-1, index=gather_idx).squeeze(-1)
    valid = (labels != ignore_index).to(student_lp.dtype)
    return student_lp * valid, teacher_lp * valid


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


def _get_compiled_reverse_kl_with_diag_fn(num_chunks: int) -> Callable:
    """Get or create a compiled reverse-KL-with-full-vocab-diag function."""
    cache_key = num_chunks
    if cache_key not in _compiled_reverse_kl_with_diag_cache:
        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_reverse_kl_with_diag_cache[cache_key] = torch.compile(
                _compute_reverse_kl_with_diag,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_reverse_kl_with_diag_cache[cache_key] = torch.compile(_compute_reverse_kl_with_diag)
    return _compiled_reverse_kl_with_diag_cache[cache_key]


def _get_compiled_forward_kl_full_fn(num_chunks: int) -> Callable:
    """Get or create a compiled forward-KL (full-vocab) function."""
    cache_key = num_chunks
    if cache_key not in _compiled_forward_kl_full_cache:
        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_forward_kl_full_cache[cache_key] = torch.compile(
                _compute_forward_kl_full,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_forward_kl_full_cache[cache_key] = torch.compile(_compute_forward_kl_full)
    return _compiled_forward_kl_full_cache[cache_key]


def _get_compiled_forward_kl_full_with_diag_fn(num_chunks: int) -> Callable:
    """Get or create a compiled forward-KL (full-vocab) + diag function."""
    cache_key = num_chunks
    if cache_key not in _compiled_forward_kl_full_with_diag_cache:
        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_forward_kl_full_with_diag_cache[cache_key] = torch.compile(
                _compute_forward_kl_full_with_diag,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_forward_kl_full_with_diag_cache[cache_key] = torch.compile(_compute_forward_kl_full_with_diag)
    return _compiled_forward_kl_full_with_diag_cache[cache_key]


def _get_compiled_sampled_logp_fn(num_chunks: int) -> Callable:
    """Get or create a compiled sampled-token logprob extractor (for KL estimators)."""
    cache_key = num_chunks
    if cache_key not in _compiled_sampled_logp_cache:
        if num_chunks > 0 and _check_auto_chunker_available():
            _compiled_sampled_logp_cache[cache_key] = torch.compile(
                _compute_sampled_token_logprobs,
                options={"auto_chunker.enable": True, "auto_chunker.num_chunk": num_chunks},
            )
        else:
            _compiled_sampled_logp_cache[cache_key] = torch.compile(_compute_sampled_token_logprobs)
    return _compiled_sampled_logp_cache[cache_key]


def compiled_reverse_kl_with_diag_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 64,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
):
    """Reverse KL(student||teacher) + per-token full-vocab diagnostics.

    Returns a 4-tuple: (token_kl, teacher_entropy, student_entropy, top1_agreement).
    """
    if lm_head_fp32:
        student_hidden_states = student_hidden_states.float()
        student_weight = student_weight.float()
    if teacher_lm_head_fp32:
        teacher_hidden_states = teacher_hidden_states.float()
        teacher_weight = teacher_weight.float()
    args = (
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        teacher_weight.detach(),
        labels,
        ignore_index,
    )
    if student_hidden_states.device.type == "cpu":
        return _compute_reverse_kl_with_diag(*args)
    return _get_compiled_reverse_kl_with_diag_fn(num_chunks)(*args)


def compiled_forward_kl_full_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    log_prob_min_clamp: float | None = None,
    num_chunks: int = 64,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
):
    """Full-vocab forward KL: Σ_v p_T(v) · (log p_T(v) − log p_S(v)).

    The full-distribution counterpart of VERL's `forward_kl_topk` — possible
    because xorl ships teacher hidden states (not truncated logprobs) and
    projects through the cached teacher LM head locally at loss time.
    """
    if lm_head_fp32:
        student_hidden_states = student_hidden_states.float()
        student_weight = student_weight.float()
    if teacher_lm_head_fp32:
        teacher_hidden_states = teacher_hidden_states.float()
        teacher_weight = teacher_weight.float()
    clamp = _NO_CLAMP_SENTINEL if log_prob_min_clamp is None else float(log_prob_min_clamp)
    args = (
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        teacher_weight.detach(),
        labels,
        ignore_index,
        clamp,
    )
    if student_hidden_states.device.type == "cpu":
        return _compute_forward_kl_full(*args)
    return _get_compiled_forward_kl_full_fn(num_chunks)(*args)


def compiled_forward_kl_full_with_diag_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    log_prob_min_clamp: float | None = None,
    num_chunks: int = 64,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
):
    """Forward KL (full-vocab) + diagnostics in one pass.

    Returns (token_kl, teacher_entropy, student_entropy, top1_agreement).
    """
    if lm_head_fp32:
        student_hidden_states = student_hidden_states.float()
        student_weight = student_weight.float()
    if teacher_lm_head_fp32:
        teacher_hidden_states = teacher_hidden_states.float()
        teacher_weight = teacher_weight.float()
    clamp = _NO_CLAMP_SENTINEL if log_prob_min_clamp is None else float(log_prob_min_clamp)
    args = (
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        teacher_weight.detach(),
        labels,
        ignore_index,
        clamp,
    )
    if student_hidden_states.device.type == "cpu":
        return _compute_forward_kl_full_with_diag(*args)
    return _get_compiled_forward_kl_full_with_diag_fn(num_chunks)(*args)


def compiled_sampled_token_logprobs_function(
    student_hidden_states: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 64,
    lm_head_fp32: bool = False,
    teacher_lm_head_fp32: bool = True,
):
    """Per-token (student_logp, teacher_logp) at the sampled-token (label) index.

    Used by single-sample KL estimators. Outputs are zero at ignore-index positions.
    """
    if lm_head_fp32:
        student_hidden_states = student_hidden_states.float()
        student_weight = student_weight.float()
    if teacher_lm_head_fp32:
        teacher_hidden_states = teacher_hidden_states.float()
        teacher_weight = teacher_weight.float()
    args = (
        student_hidden_states,
        student_weight,
        teacher_hidden_states.detach(),
        teacher_weight.detach(),
        labels,
        ignore_index,
    )
    if student_hidden_states.device.type == "cpu":
        return _compute_sampled_token_logprobs(*args)
    return _get_compiled_sampled_logp_fn(num_chunks)(*args)
