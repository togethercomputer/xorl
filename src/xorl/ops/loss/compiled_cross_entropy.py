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
