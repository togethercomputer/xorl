"""Reusable RL objective primitives.

These helpers intentionally stop at train-time tensor math. Rollout collection,
reward execution, and request construction live outside the core training
engine.
"""

from xorl.rl.kl import compute_kl_estimate, compute_sequence_kl
from xorl.rl.normalization import reduce_token_or_sample_mean
from xorl.rl.objectives import compute_gspo_kl, compute_opsm_mask, compute_policy_clip_loss


__all__ = [
    "compute_gspo_kl",
    "compute_kl_estimate",
    "compute_opsm_mask",
    "compute_policy_clip_loss",
    "compute_sequence_kl",
    "reduce_token_or_sample_mean",
]
