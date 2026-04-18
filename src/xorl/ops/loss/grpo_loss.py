"""
DR-GRPO: "Done Right" GRPO Loss for RL Training.

Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
https://arxiv.org/abs/2503.20783
"""

from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed as dist

from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce


AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
KLType = Literal["k1", "k2", "k3"]
RatioType = Literal["token", "sequence"]


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked mean: sum(values * mask) / divisor."""
    masked_sum = (values * mask).sum()
    if loss_scale is not None:
        divisor = loss_scale.clamp(min=1.0)
    else:
        divisor = mask.sum().clamp(min=1.0)
    return masked_sum / divisor


def compute_ratio(
    logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    mask: torch.Tensor,
    ratio_type: RatioType = "token",
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, torch.Tensor]]]:
    """Importance sampling ratio r = π_θ/π_old.

    token:    r_t = exp(logprobs_t - generator_logprobs_t)
    sequence: r_seq = exp(mean_t[logprobs - generator_logprobs]), uses reparameterization.
    """
    if ratio_type == "token":
        log_ratio = logprobs - generator_logprobs.detach()
        ratio = torch.exp(log_ratio)
    elif ratio_type == "sequence":
        token_log_ratio = logprobs - generator_logprobs.detach()
        seq_lengths = mask.sum(dim=-1).clamp(min=1)
        seq_log_ratio = (token_log_ratio * mask).sum(dim=-1) / seq_lengths

        # Reparameterization: forward uses seq ratio, backward uses token grads
        log_ratio = logprobs - logprobs.detach() + seq_log_ratio.detach().unsqueeze(-1)
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(f"Unknown ratio_type: {ratio_type}")

    with torch.no_grad():
        metrics = [
            ("loss/ratio/mean", masked_mean(ratio, mask)),
            ("loss/kl_policy/mean", masked_mean(-log_ratio, mask)),
        ]

    return ratio, log_ratio, metrics


def compute_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    kl_type: KLType = "k3",
) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
    """KL divergence using Schulman's estimators (k1, k2, k3)."""
    log_ratio = policy_logprobs - ref_logprobs.detach()

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio.square()
    elif kl_type == "k3":
        neg_log_ratio = torch.clamp(-log_ratio, min=-10.0, max=10.0)
        ratio = torch.exp(neg_log_ratio)
        kl = ratio - neg_log_ratio - 1
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    with torch.no_grad():
        metrics = [("loss/kl_ref/mean", masked_mean(kl, mask))]

    return kl, metrics


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
    """PPO clipped surrogate: L = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A)."""
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        mask_bool = mask.bool()
        clipped_high = (ratio > 1 + clip_high) & mask_bool
        clipped_low = (ratio < 1 - clip_low) & mask_bool
        pos_adv = advantages > 0
        neg_adv = advantages < 0

        metrics = [
            ("loss/clip/clipped_ratio/mean", masked_mean(clipped_ratio, mask)),
            ("loss/clip/high_fraction", masked_mean((clipped_high & pos_adv).float(), mask)),
            ("loss/clip/low_fraction", masked_mean((clipped_low & neg_adv).float(), mask)),
        ]

    return pg_loss, metrics


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    agg_type: AggType = "token_mean",
    loss_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
    """Aggregate per-token loss: token_mean, fixed_horizon, or sequence_mean."""
    if agg_type == "token_mean":
        loss = masked_mean(per_token_loss, mask, loss_scale)
    elif agg_type == "fixed_horizon":
        loss = (per_token_loss * mask).sum() / max(mask.numel(), 1)
    elif agg_type == "sequence_mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_means = (per_token_loss * mask).sum(dim=-1) / seq_lengths
        loss = seq_means.sum() / max(seq_means.numel(), 1)
    else:
        raise ValueError(f"Unknown agg_type: {agg_type}")

    with torch.no_grad():
        metrics = [("loss/aggregate/active_fraction", mask.mean())]

    return loss, metrics


def drgrpo_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    beta: float = 0.1,
    agg_type: AggType = "fixed_horizon",
    ratio_type: RatioType = "token",
    kl_type: KLType = "k3",
    ce_mode: str = "compiled",
    num_chunks: int = 8,
    tp_group: Optional[dist.ProcessGroup] = None,
    lm_head_fp32: bool = False,
    loss_scale: Optional[torch.Tensor] = None,
) -> LossOutput:
    """DR-GRPO loss for RL training.

    Per-token: L_t = max(-r*A, -clip(r, 1-ε, 1+ε)*A) + β*KL
    Aggregated: Depends on agg_type (default: fixed_horizon)

    Args:
        hidden_states: (B, S, H) model hidden states.
        weight: (V, H) or (V/tp, H) LM head weight.
        labels: (B, S) target token IDs, already next-token aligned.
        old_logprobs: (B, S) log probs from generation policy.
        advantages: (B, S) per-token advantages.
        ref_logprobs: (B, S) reference model log probs for KL (required if beta > 0).
        ignore_index: Token ID to ignore (default: -100).
        clip_low: Lower clip bound (default: 0.2).
        clip_high: Upper clip bound (default: 0.28).
        beta: KL penalty coefficient (default: 0.1).
        agg_type: Aggregation type (default: "fixed_horizon").
        ratio_type: Ratio type: "token" or "sequence" (default: "token").
        kl_type: KL estimator: "k1", "k2", "k3" (default: "k3").
        ce_mode: Cross-entropy mode: "compiled" or "eager".
        num_chunks: Chunks for compiled mode.
        tp_group: TP process group for vocab-parallel CE.
        lm_head_fp32: Compute LM head in FP32.
        loss_scale: For distributed token_mean aggregation.

    Returns:
        LossOutput with loss, per_token_logprobs, per_token_loss, and metrics.
    """
    if beta > 0 and ref_logprobs is None:
        raise ValueError("ref_logprobs required when beta > 0")

    B, S = labels.shape
    H = hidden_states.size(-1)

    labels_flat = labels.reshape(-1)
    hidden_states_flat = hidden_states.reshape(-1, H)

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
    logprobs = -per_token_ce.view(B, S)

    loss_mask = (labels != ignore_index).float()

    ratio, _, ratio_m = compute_ratio(logprobs, old_logprobs, loss_mask, ratio_type)

    pg_loss, clip_m = pg_ppo_clip(ratio, advantages, loss_mask, clip_low, clip_high)

    kl_m: List[Tuple[str, torch.Tensor]] = []
    if beta > 0:
        kl, kl_m = compute_kl(logprobs, ref_logprobs, loss_mask, kl_type)
        pg_loss = pg_loss + beta * kl

    loss, agg_m = aggregate(pg_loss, loss_mask, agg_type, loss_scale)

    all_metrics = ratio_m + clip_m + kl_m + agg_m
    metrics = {k: v.item() for k, v in all_metrics}

    return LossOutput(
        loss=loss,
        per_token_logprobs=logprobs.detach(),
        per_token_loss=pg_loss.detach(),
        metrics=metrics,
    )
