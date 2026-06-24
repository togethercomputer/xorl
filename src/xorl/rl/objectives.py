from __future__ import annotations

import torch

from xorl.rl.kl import compute_sequence_kl


def compute_policy_clip_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the PPO clipped policy loss used by Slime and XoRL.

    ``ppo_kl`` is ``old_logprobs - current_logprobs``. The returned clip
    fraction tensor marks the first PPO clip and intentionally does not include
    dual-clip events, matching Slime's reporting behavior.
    """
    ratio = torch.exp(-ppo_kl)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        if eps_clip_c <= 1.0:
            raise ValueError(f"eps_clip_c must be > 1.0, got {eps_clip_c}")
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac, ratio


def compute_gspo_kl(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute GSPO sequence-level KL and expand it to token shape."""
    return compute_sequence_kl(current_logprobs, old_logprobs, masks, expand=True)


def compute_opsm_mask(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    masks: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Slime-style Off-Policy Sequence Masking.

    Tokens with negative advantages are masked out when their sequence-level
    ``mean(old - current)`` KL exceeds ``delta``. ``opsm_clipfrac`` follows
    Slime's reported value: sum of each sequence's masked-token fraction.
    """
    mask_f = masks.float()
    seq_kl = compute_sequence_kl(current_logprobs, old_logprobs, mask_f, expand=False)
    masked = (advantages < 0) & (seq_kl.unsqueeze(-1) > delta) & masks.bool()
    opsm_mask = torch.ones_like(mask_f).masked_fill(masked, 0.0)
    per_sequence_fraction = (masked.float() * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)
    opsm_clipfrac = per_sequence_fraction.sum()
    return opsm_mask, opsm_clipfrac
