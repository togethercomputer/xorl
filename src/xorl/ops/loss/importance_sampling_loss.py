from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .compiled_cross_entropy import compiled_cross_entropy_function


def _compute_per_token_ce(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int,
    ce_mode: str,
    num_chunks: int = 8,
) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss based on the specified mode.

    Args:
        hidden_states_flat: Flattened hidden states, shape (BT, H)
        weight: LM head weight matrix, shape (V, H)
        labels_flat: Flattened labels, shape (BT,)
        ignore_index: Index to ignore in loss computation
        ce_mode: Cross-entropy computation mode ("compiled" or "eager")
        num_chunks: Number of chunks for compiled mode

    Returns:
        per_token_ce: Per-token cross-entropy loss, shape (BT,)
    """
    if ce_mode == "compiled":
        # Uses torch.compile to avoid materializing full [BT, V] logits
        per_token_ce = compiled_cross_entropy_function(hidden_states_flat, weight, labels_flat, ignore_index, num_chunks)
    else:
        # eager mode
        logits_flat = (hidden_states_flat @ weight.t()).float()
        per_token_ce = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)

    return per_token_ce


def importance_sampling_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 8,
    ce_mode: str = "compiled",
    return_per_token: bool = False,
) -> Tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute importance sampling loss for GRPO/RL training.

    This implements the Tinker-style importance sampling loss:
        prob_ratio = exp(new_logprobs - old_logprobs)
        loss = -(prob_ratio * advantages).mean()

    Supports multiple computation modes:
    - "compiled": RECOMMENDED. torch.compile (1.6x speed, 16% memory)
    - "eager": Simple F.cross_entropy baseline (may OOM at 32K)

    Args:
        hidden_states: Model hidden states, shape (batch, seq_len, hidden_dim)
        weight: LM head weight matrix, shape (vocab_size, hidden_dim)
        labels: Target token IDs, shape (batch, seq_len). Already next-token aligned.
        old_logprobs: Old policy log probabilities from sampling, shape (batch, seq_len)
        advantages: Per-token advantages, shape (batch, seq_len)
        ignore_index: Index to ignore in loss computation (default: -100)
        num_chunks: Number of chunks for compiled mode (default: 8).
        ce_mode: Cross-entropy mode - "compiled" (default) or "eager"
        return_per_token: If True, returns per-token logprobs and per-token CE loss.
                         Useful for custom loss computations.

    Returns:
        Tuple of (loss, None, per_token_logprobs, per_token_loss, metrics)
        - loss: Scalar importance sampling loss = -(ratio * advantages).mean()
        - None: Placeholder for compatibility
        - per_token_logprobs: Per-token log probabilities, shape (batch, seq_len)
        - per_token_loss: Per-token policy gradient loss (-(ratio * advantages)), shape (batch, seq_len)
        - metrics: Dictionary with ratio statistics (ratio_mean, ratio_min, ratio_max)
    """
    original_shape = labels.shape
    H = hidden_states.size(-1)

    # Flatten tensors
    labels_flat = labels.reshape(-1)
    hidden_states_flat = hidden_states.reshape(-1, H)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)

    # Valid/action mask
    valid_mask = (labels_flat != ignore_index)
    n_valid = valid_mask.sum().clamp(min=1).float()

    # ---- Cross-entropy computation ----
    per_token_ce = _compute_per_token_ce(
        hidden_states_flat, weight, labels_flat, ignore_index, ce_mode, num_chunks
    )

    # new logprobs = log p(target) = -CE
    new_logprobs_flat = -per_token_ce.detach()

    # ---- ratio computation (no sanitization) ----
    delta = (new_logprobs_flat - old_logprobs_flat)
    delta = delta.masked_fill(~valid_mask, 0.0)
    delta = delta.clamp(min=-20.0, max=20.0)
    ratio = torch.exp(delta)

    # ---- Per-token policy gradient loss: -(ratio * advantages) ----
    per_token_pg = -(ratio * advantages_flat)
    per_token_pg = per_token_pg.masked_fill(~valid_mask, 0.0)

    # ---- Option B: value from true PG, grad from weighted CE surrogate ----
    true_pg = per_token_pg.sum() / n_valid

    w = (ratio.detach() * advantages_flat).masked_fill(~valid_mask, 0.0)
    surrogate = (w * per_token_ce).sum() / n_valid

    loss = true_pg.detach() + surrogate - surrogate.detach()

    # Compute metrics for logging (convert to Python floats for JSON serialization)
    valid_ratio = ratio[valid_mask] if valid_mask.any() else ratio
    metrics = {
        "ratio_mean": valid_ratio.mean().detach().item(),
        "ratio_min": valid_ratio.min().detach().item(),
        "ratio_max": valid_ratio.max().detach().item(),
    }

    # Reshape per-token outputs
    per_token_logprobs = new_logprobs_flat.view(original_shape)
    per_token_loss = per_token_pg.view(original_shape)

    return loss, None, per_token_logprobs, per_token_loss, metrics
