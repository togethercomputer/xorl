"""
Policy Loss with PPO Clipping and TIS Correction.

This module provides the policy loss functions including:
- PPO-style clipped policy gradient loss
- Temporal Importance Sampling (TIS) correction
- Combined policy_loss_function
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce
from xorl.ops.loss.reducers import Reducer, TokenPartial


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
    metric_reducer: Reducer,
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
        metric_reducer: Reducer applied to per-token mean metrics (tis_mean,
            tis_clipfrac). min/max are local reductions and bypass it.
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

    valid_mask_f = valid_mask.float()
    tis_clipfrac_per_token = (tis_clipped != tis).float()
    # ±inf identity on empty ranks lets cross-rank MIN/MAX-allreduce ignore empty contributors.
    if valid_mask.any():
        tis_min = tis.masked_fill(~valid_mask, float("inf")).min()
        tis_max = tis.masked_fill(~valid_mask, float("-inf")).max()
    else:
        tis_min = tis.new_tensor(float("inf"))
        tis_max = tis.new_tensor(float("-inf"))
    tis_metrics = {
        "tis_mean": metric_reducer(tis, valid_mask_f),
        "tis_min": tis_min,
        "tis_max": tis_max,
        "tis_clipfrac": metric_reducer(tis_clipfrac_per_token, valid_mask_f),
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
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
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
        loss_reducer: Reduces per-token loss to a scalar partial share. None =>
            ``TokenPartial(scale=valid_mask.sum())`` (legacy local token-mean; does
            not compose across micro-batches/ranks). Pass a shared global-scale
            reducer to make summed partial shares recover the global loss.
        metric_reducer: Reduces per-token /mean metrics (pg_clipfrac, icepop_maskfrac,
            tis_mean, tis_clipfrac, kl_sample_train_k3, entropy_sample, ratio_mean)
            the same way. ratio_min/ratio_max/tis_min/tis_max stay local scalars.

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
    valid_mask = labels_flat != ignore_index
    valid_mask_f = valid_mask.float()
    valid_count = valid_mask.sum()

    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=valid_count.float())
    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=valid_count.float())

    # Compute cross-entropy (supports vocab-parallel TP via tp_group)
    per_token_ce = compute_per_token_ce(
        hidden_states_flat,
        weight,
        labels_flat,
        ignore_index,
        ce_mode,
        num_chunks,
        tp_group=tp_group,
        lm_head_fp32=lm_head_fp32,
    )

    new_logprobs_flat = -per_token_ce.detach()

    # Compute PPO KL: old_log_probs - new_log_probs
    ppo_kl = old_logprobs_flat - new_logprobs_flat

    # Mask invalid positions
    ppo_kl = ppo_kl.masked_fill(~valid_mask, 0.0)
    advantages_masked = advantages_flat.masked_fill(~valid_mask, 0.0)

    # Computed BEFORE compute_ppo_loss to avoid torch.compile interference.
    _kl_stats = None
    if compute_kl_stats:
        with torch.no_grad():
            _log_ratio_full = (new_logprobs_flat - old_logprobs_flat).masked_fill(~valid_mask, 0.0)
            _ratio_full = torch.exp(_log_ratio_full)
            _per_token_k3 = _ratio_full - _log_ratio_full - 1.0
            # ±inf identity on empty ranks lets cross-rank MIN/MAX-allreduce ignore empty contributors.
            if valid_mask.any():
                _ratio_min = _ratio_full.masked_fill(~valid_mask, float("inf")).min()
                _ratio_max = _ratio_full.masked_fill(~valid_mask, float("-inf")).max()
            else:
                _ratio_min = _ratio_full.new_tensor(float("inf"))
                _ratio_max = _ratio_full.new_tensor(float("-inf"))
            _kl_stats = {
                "kl_sample_train_k3": metric_reducer(_per_token_k3, valid_mask_f),
                "entropy_sample": metric_reducer(-old_logprobs_flat, valid_mask_f),
                "ratio_mean": metric_reducer(_ratio_full, valid_mask_f),
                "ratio_min": _ratio_min,
                "ratio_max": _ratio_max,
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
            logger.warning(
                "IcePop and TIS are both enabled. IcePop makes TIS redundant "
                "when using inference logprobs as old_logprobs."
            )
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
            metric_reducer=metric_reducer,
            tis_clip_low=tis_clip_low,
            tis_clip_high=tis_clip_high,
        )

    # True loss value (for logging): partial share under loss_reducer.
    true_loss = loss_reducer(pg_losses, valid_mask_f)

    # Gradient-active mask: tokens that are not clipped, not IcePop-masked, and valid
    gradient_active = ~is_clipped & valid_mask
    if icepop_mask is not None:
        gradient_active = gradient_active & icepop_mask

    # Surrogate: gradient weight = ratio * advantages, zeroed for inactive tokens
    gradient_weight = (ratio.detach() * advantages_flat).masked_fill(~gradient_active, 0.0)
    surrogate = loss_reducer(gradient_weight * per_token_ce, valid_mask_f)

    # Combine: forward value from true_loss, gradient from surrogate
    loss_with_grad = true_loss.detach() + surrogate - surrogate.detach()

    # Return training logprobs reshaped
    new_logprobs = new_logprobs_flat.view(original_shape)

    with torch.no_grad():
        metrics: Dict[str, Any] = {
            "valid_tokens": valid_count.item(),
            "pg_clipfrac": metric_reducer(is_clipped.float(), valid_mask_f),
        }

        if icepop_mask is not None:
            metrics["icepop_maskfrac"] = metric_reducer((~icepop_mask).float(), valid_mask_f)

        if _kl_stats is not None:
            metrics.update(_kl_stats)

        metrics.update(tis_metrics)

    metric_ops: Dict[str, str] = {}
    if _kl_stats is not None:
        metric_ops["ratio_min"] = "min"
        metric_ops["ratio_max"] = "max"
    if tis_metrics:
        metric_ops["tis_min"] = "min"
        metric_ops["tis_max"] = "max"

    return LossOutput(
        loss=loss_with_grad,
        per_token_logprobs=new_logprobs,
        metrics=metrics,
        metric_ops=metric_ops or None,
    )
