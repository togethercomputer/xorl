from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce


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
    tp_group: Optional[dist.ProcessGroup] = None,
    compute_kl_stats: bool = False,
) -> "LossOutput":
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
        compute_kl_stats: If True, compute and return KL statistics in metrics dict:
                         - kl_sample_train_k3: Schulman's K3 estimator: mean(exp(log_ratio) - log_ratio - 1)
                           where log_ratio = new_logprobs - old_logprobs. Non-negative, unbiased, lower variance.
                         - entropy_sample: -mean(old_logprobs) over valid tokens
                         - valid_tokens: Count of valid tokens

    Returns:
        LossOutput with loss, per_token_logprobs, per_token_loss, and metrics.
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
    per_token_ce = compute_per_token_ce(
        hidden_states_flat, weight, labels_flat, ignore_index, ce_mode, num_chunks,
        tp_group=tp_group,
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

    # Optionally compute KL statistics
    if compute_kl_stats:
        with torch.no_grad():
            _n_valid_kl = valid_mask.sum().item()  # TRUE count, no clamp
            if valid_mask.any():
                valid_old = old_logprobs_flat[valid_mask]
                valid_new = new_logprobs_flat[valid_mask]
                log_ratio = valid_new - valid_old

                # K3 estimator (Schulman): exp(log_ratio) - log_ratio - 1
                # Non-negative, unbiased, lower variance than K1/K2
                k3 = (torch.exp(log_ratio) - log_ratio - 1.0).mean().item()
                metrics["kl_sample_train_k3"] = k3
                metrics["entropy_sample"] = -valid_old.mean().item()
                metrics["valid_tokens"] = _n_valid_kl
                metrics["_n_valid_kl"] = _n_valid_kl
            else:
                metrics["kl_sample_train_k3"] = 0.0
                metrics["entropy_sample"] = 0.0
                metrics["valid_tokens"] = 0
                metrics["_n_valid_kl"] = 0

    # Reshape per-token outputs
    per_token_logprobs = new_logprobs_flat.view(original_shape)
    per_token_loss = per_token_pg.view(original_shape)

    return LossOutput(
        loss=loss,
        per_token_logprobs=per_token_logprobs,
        per_token_loss=per_token_loss,
        metrics=metrics,
    )
