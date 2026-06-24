"""
DR-GRPO: "Done Right" GRPO Loss for RL Training.

Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
https://arxiv.org/abs/2503.20783
"""

from typing import List, Literal, Tuple

import torch
import torch.distributed as dist

from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce
from xorl.ops.loss.reducers import Reducer, TokenPartial


KLType = Literal["k1", "k2", "k3"]
RatioType = Literal["token", "sequence"]


def compute_ratio(
    logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    mask: torch.Tensor,
    metric_reducer: Reducer,
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
            ("loss/ratio/mean", metric_reducer(ratio, mask)),
            ("loss/kl_policy/mean", metric_reducer(-log_ratio, mask)),
        ]

    return ratio, log_ratio, metrics


def compute_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    metric_reducer: Reducer,
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
        metrics = [("loss/kl_ref/mean", metric_reducer(kl, mask))]

    return kl, metrics


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    metric_reducer: Reducer,
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
            ("loss/clip/clipped_ratio/mean", metric_reducer(clipped_ratio, mask)),
            ("loss/clip/high_fraction", metric_reducer((clipped_high & pos_adv).float(), mask)),
            ("loss/clip/low_fraction", metric_reducer((clipped_low & neg_adv).float(), mask)),
        ]

    return pg_loss, metrics


def drgrpo_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: torch.Tensor | None = None,
    ignore_index: int = -100,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    beta: float = 0.1,
    ratio_type: RatioType = "token",
    kl_type: KLType = "k3",
    ce_mode: str = "compiled",
    num_chunks: int = 8,
    tp_group: dist.ProcessGroup | None = None,
    lm_head_fp32: bool = False,
    loss_reducer: Reducer | None = None,
    metric_reducer: Reducer | None = None,
    lm_head: torch.nn.Module | None = None,
) -> LossOutput:
    """DR-GRPO loss for RL training.

    Per-token: L_t = max(-r*A, -clip(r, 1-ε, 1+ε)*A) + β*KL
    Aggregated: ``loss_reducer(per_token_loss, mask)``. Defaults to
    ``TokenPartial(scale=loss_mask.sum())`` — the local active-token mean.

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
        ratio_type: Ratio type: "token" or "sequence" (default: "token").
        kl_type: KL estimator: "k1", "k2", "k3" (default: "k3").
        ce_mode: Cross-entropy mode: "compiled" or "eager".
        num_chunks: Chunks for compiled mode.
        tp_group: TP process group for vocab-parallel CE.
        lm_head_fp32: Compute LM head in FP32.
        loss_reducer / metric_reducer: Both default to
            ``TokenPartial(scale=loss_mask.sum())`` (legacy local active-token
            mean; does not compose across mbs/ranks). Pass shared global-scale
            reducers to make summed partial shares recover the global value.

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
        lm_head=lm_head,
    )
    logprobs = -per_token_ce.view(B, S)

    loss_mask = (labels != ignore_index).float()

    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=loss_mask.sum())
    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=loss_mask.sum())

    ratio, _, ratio_m = compute_ratio(logprobs, old_logprobs, loss_mask, metric_reducer, ratio_type)

    pg_loss, clip_m = pg_ppo_clip(ratio, advantages, loss_mask, metric_reducer, clip_low, clip_high)

    kl_m: List[Tuple[str, torch.Tensor]] = []
    if beta > 0:
        kl, kl_m = compute_kl(logprobs, ref_logprobs, loss_mask, metric_reducer, kl_type)
        pg_loss = pg_loss + beta * kl

    loss = loss_reducer(pg_loss, loss_mask)

    metrics = dict(ratio_m + clip_m + kl_m)

    return LossOutput(
        loss=loss,
        per_token_logprobs=logprobs.detach(),
        per_token_loss=pg_loss.detach(),
        metrics=metrics,
    )
