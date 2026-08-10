from __future__ import annotations

from typing import Literal

import torch


KLEstimator = Literal["k1", "k2", "k3", "low_var_kl"]


def compute_kl_estimate(
    policy_logprobs: torch.Tensor,
    base_logprobs: torch.Tensor,
    kind: KLEstimator,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Slime-compatible sampled-token KL estimates.

    ``policy_logprobs`` are the current policy log-probabilities and
    ``base_logprobs`` are reference/base log-probabilities. ``importance_ratio``
    is optional ``pi_current / pi_old`` weighting, matching Slime's unbiased KL
    mode.
    """
    log_ratio = policy_logprobs.float() - base_logprobs.float()

    if kind == "k1":
        kl = log_ratio
    elif kind == "k2":
        kl = 0.5 * log_ratio.square()
    elif kind in ("k3", "low_var_kl"):
        neg_log_ratio = -log_ratio
        kl = torch.exp(neg_log_ratio) - 1.0 - neg_log_ratio
    else:
        raise ValueError(f"Unknown KL estimator: {kind}")

    if importance_ratio is not None:
        kl = importance_ratio.float() * kl

    if kind == "low_var_kl":
        kl = torch.clamp(kl, min=-10.0, max=10.0)

    return kl


def compute_sequence_kl(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    masks: torch.Tensor,
    *,
    expand: bool = False,
) -> torch.Tensor:
    """Compute per-sequence PPO KL ``mean(old - current)`` over valid tokens.

    When ``expand=True``, the per-sequence value is expanded back to token shape,
    which is the policy-path GSPO convention used by Slime before PPO clipping.
    """
    mask_f = masks.float()
    seq_lengths = mask_f.sum(dim=-1).clamp(min=1.0)
    seq_kl = ((old_logprobs.float() - current_logprobs.float()) * mask_f).sum(dim=-1) / seq_lengths
    if expand:
        return seq_kl.unsqueeze(-1).expand_as(current_logprobs)
    return seq_kl
