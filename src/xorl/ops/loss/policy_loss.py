"""
Policy Loss with PPO Clipping and TIS Correction.

This module provides the policy loss functions including:
- PPO-style clipped policy gradient loss
- Temporal Importance Sampling (TIS) correction
- Combined policy_loss_function
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True)
def compute_ppo_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.2,
    eps_clip_c: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PPO-style clipped policy loss.

    Args:
        ppo_kl: KL divergence tensor (old_log_probs - new_log_probs)
        advantages: Per-token advantages
        eps_clip: Lower clip ratio (default: 0.2)
        eps_clip_high: Upper clip ratio (default: 0.2)
        eps_clip_c: Dual-clip ratio for negative advantages (optional)

    Returns:
        pg_losses: Clipped policy gradient losses
        is_clipped: Per-token boolean mask of clipped tokens
        ratio: Importance sampling ratio exp(-ppo_kl)
    """
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    is_clipped = torch.gt(pg_losses2, pg_losses1)

    # Optional dual-clip for negative advantages
    if eps_clip_c is not None:
        assert eps_clip_c > 1.0, f"eps_clip_c must be > 1.0, got {eps_clip_c}"
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        # Also mark dual-clipped tokens
        is_dual_clipped = (advantages < 0) & torch.lt(pg_losses3, clip_pg_losses1)
        is_clipped = is_clipped | is_dual_clipped
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, is_clipped, ratio


def apply_tis_correction(
    pg_loss: torch.Tensor,
    train_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    tis_clip_low: float = 0.1,
    tis_clip_high: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply Temporal Importance Sampling (TIS) correction.

    TIS corrects for the distribution shift between rollout time and training time.
    The TIS weight is: exp(train_log_probs - rollout_log_probs)

    Args:
        pg_loss: Policy gradient loss tensor
        train_log_probs: Log probabilities from current training step
        rollout_log_probs: Log probabilities from rollout/inference
        valid_mask: Mask for valid tokens
        tis_clip_low: Lower bound for TIS clipping (default: 0.1)
        tis_clip_high: Upper bound for TIS clipping (default: 2.0)

    Returns:
        Tuple of (corrected_loss, metrics_dict)
    """
    # Compute TIS weights: ratio of train vs rollout distributions
    tis = torch.exp(train_log_probs - rollout_log_probs)

    # Clip TIS weights to prevent extreme values
    tis_clipped = torch.clamp(tis, min=tis_clip_low, max=tis_clip_high)

    # Apply TIS correction to loss
    corrected_loss = pg_loss * tis_clipped

    # Compute metrics
    tis_metrics = {
        "tis_mean": tis[valid_mask].mean() if valid_mask.any() else torch.tensor(1.0),
        "tis_min": tis[valid_mask].min() if valid_mask.any() else torch.tensor(1.0),
        "tis_max": tis[valid_mask].max() if valid_mask.any() else torch.tensor(1.0),
        "tis_clipfrac": (tis_clipped != tis)[valid_mask].float().mean() if valid_mask.any() else torch.tensor(0.0),
    }

    return corrected_loss, tis_metrics


def policy_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    rollout_logprobs: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.2,
    eps_clip_c: Optional[float] = None,
    tis_clip_low: float = 0.1,
    tis_clip_high: float = 2.0,
    use_tis: bool = False,
    use_liger: bool = True,
    num_chunks: int = 8,
    ce_mode: str = "compiled",
    compute_kl_stats: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    lm_head_fp32: bool = False,
    icepop_beta: Optional[float] = None,
) -> "LossOutput":
    """
    Policy loss with PPO clipping, optional IcePop masking, and optional TIS correction.

    This implements the loss function which includes:
    1. PPO-style clipping on the importance sampling ratio
    2. Optional IcePop hard masking (GLM-5): zeros gradient for tokens where ratio is outside [1/β, β]
    3. Optional Temporal Importance Sampling (TIS) correction for off-policy data

    Supports multiple computation modes:
    - "compiled": RECOMMENDED. torch.compile (1.6x speed, 16% memory)
    - "eager": Simple F.cross_entropy baseline (may OOM at 32K)

    Args:
        hidden_states: Model hidden states, shape (batch, seq_len, hidden_dim)
        weight: LM head weight matrix, shape (vocab_size, hidden_dim)
        labels: Target token IDs, shape (batch, seq_len). Already next-token aligned.
        old_logprobs: Old policy log probabilities from sampling, shape (batch, seq_len)
        advantages: Per-token advantages, shape (batch, seq_len)
        rollout_logprobs: Optional rollout log probabilities for TIS correction, shape (batch, seq_len)
        ignore_index: Index to ignore in loss computation (default: -100)
        eps_clip: Lower clip ratio for PPO (default: 0.2)
        eps_clip_high: Upper clip ratio for PPO (default: 0.2)
        eps_clip_c: Dual-clip ratio for negative advantages (optional)
        tis_clip_low: Lower bound for TIS clipping (default: 0.1)
        tis_clip_high: Upper bound for TIS clipping (default: 2.0)
        use_tis: Whether to apply TIS correction (default: False)
        use_liger: Kept for API compatibility (ignored)
        num_chunks: Number of chunks for auto_chunker (default: 8). Only used when ce_mode="compiled".
        ce_mode: Cross-entropy mode - "compiled" (recommended) or "eager"
        tp_group: TP process group for vocab-parallel cross-entropy (default: None)
        compute_kl_stats: If True, compute and return full KL statistics in metrics dict
                         (kl_sample_train_k3, entropy_sample, ratio stats).
                         If False (default), only return valid_tokens and pg_clipfrac.

    Returns:
        LossOutput with loss, per_token_logprobs (new logprobs), and metrics.
    """

    # Store original shape
    original_shape = labels.shape

    # Flatten tensors
    labels_flat = labels.view(-1)
    hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
    old_logprobs_flat = old_logprobs.view(-1)
    advantages_flat = advantages.view(-1)

    # Create mask for valid tokens (use labels != ignore_index)
    valid_mask = (labels_flat != ignore_index)
    n_valid = valid_mask.sum().clamp(min=1)

    # Compute cross-entropy (supports vocab-parallel TP via tp_group)
    per_token_ce = compute_per_token_ce(
        hidden_states_flat, weight, labels_flat, ignore_index, ce_mode, num_chunks,
        tp_group=tp_group, lm_head_fp32=lm_head_fp32,
    )

    new_logprobs_flat = -per_token_ce.detach()

    # Compute PPO KL: old_log_probs - new_log_probs
    ppo_kl = old_logprobs_flat - new_logprobs_flat

    # Mask invalid positions
    ppo_kl = ppo_kl.masked_fill(~valid_mask, 0.0)
    advantages_masked = advantages_flat.masked_fill(~valid_mask, 0.0)

    # Compute KL stats BEFORE compute_ppo_loss to avoid torch.compile interference
    _kl_stats = None
    if compute_kl_stats:
        with torch.no_grad():
            _n_valid_kl = valid_mask.sum().item()  # TRUE count, no clamp
            if valid_mask.any():
                _valid_old = old_logprobs_flat[valid_mask]
                _valid_new = new_logprobs_flat[valid_mask]
                _log_ratio = _valid_new - _valid_old
                _ratio_valid = torch.exp(_log_ratio)
                _kl_stats = {
                    "kl_sample_train_k3": (_ratio_valid - _log_ratio - 1.0).mean().item(),
                    "entropy_sample": -_valid_old.mean().item(),
                    "ratio_mean": _ratio_valid.mean().item(),
                    "ratio_min": _ratio_valid.min().item(),
                    "ratio_max": _ratio_valid.max().item(),
                    "_n_valid_kl": _n_valid_kl,
                }
            else:
                _kl_stats = {
                    "kl_sample_train_k3": 0.0,
                    "entropy_sample": 0.0,
                    "ratio_mean": 1.0,
                    "ratio_min": 1.0,
                    "ratio_max": 1.0,
                    "_n_valid_kl": 0,
                }

    # Compute PPO-style clipped loss (returns per-token losses, clip mask, and ratio)
    pg_losses, is_clipped, ratio = compute_ppo_loss(
        ppo_kl=ppo_kl,
        advantages=advantages_masked,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
        eps_clip_c=eps_clip_c,
    )

    # IcePop hard masking (GLM-5, arXiv:2602.15763):
    # Zero gradient for tokens where ratio is outside [1/β, β]
    icepop_mask = None
    if icepop_beta is not None:
        if use_tis:
            logger.warning("IcePop and TIS are both enabled. IcePop makes TIS redundant "
                           "when using inference logprobs as old_logprobs.")
        ratio_d = ratio.detach()
        icepop_mask = (ratio_d >= 1.0 / icepop_beta) & (ratio_d <= icepop_beta)

    # Apply TIS correction if enabled and rollout_logprobs provided
    tis_metrics = {}
    if use_tis and rollout_logprobs is not None:
        rollout_logprobs_flat = rollout_logprobs.view(-1)
        pg_losses, tis_metrics = apply_tis_correction(
            pg_loss=pg_losses,
            train_log_probs=new_logprobs_flat,
            rollout_log_probs=rollout_logprobs_flat,
            valid_mask=valid_mask,
            tis_clip_low=tis_clip_low,
            tis_clip_high=tis_clip_high,
        )

    # Surrogate loss for gradient flow through CE
    # True loss value (for logging): pg_losses averaged over valid tokens
    true_loss = pg_losses.masked_fill(~valid_mask, 0.0).sum() / n_valid

    # Gradient-active mask: tokens that are not clipped, not IcePop-masked, and valid
    gradient_active = ~is_clipped & valid_mask
    if icepop_mask is not None:
        gradient_active = gradient_active & icepop_mask

    # Surrogate: gradient weight = ratio * advantages, zeroed for inactive tokens
    gradient_weight = (ratio.detach() * advantages_flat).masked_fill(~gradient_active, 0.0)
    surrogate = (gradient_weight * per_token_ce).sum() / n_valid

    # Combine: forward value from true_loss, gradient from surrogate
    loss_with_grad = true_loss.detach() + surrogate - surrogate.detach()

    # Return training logprobs reshaped
    new_logprobs = new_logprobs_flat.view(original_shape)

    # Compute metrics
    with torch.no_grad():
        metrics = {
            "valid_tokens": valid_mask.sum().item(),
            "pg_clipfrac": is_clipped[valid_mask].float().mean().item() if valid_mask.any() else 0.0,
        }

        if icepop_mask is not None:
            metrics["icepop_maskfrac"] = (~icepop_mask)[valid_mask].float().mean().item() if valid_mask.any() else 0.0

        # Use pre-computed KL statistics (computed before torch.compile'd compute_ppo_loss)
        if _kl_stats is not None:
            metrics.update(_kl_stats)

        # Add TIS metrics if available
        for k, v in tis_metrics.items():
            metrics[k] = v.item() if torch.is_tensor(v) else v

    return LossOutput(
        loss=loss_with_grad,
        per_token_logprobs=new_logprobs,
        metrics=metrics,
    )
