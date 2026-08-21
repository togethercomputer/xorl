from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from xorl.objectives.loss_output import LossOutput
from xorl.objectives.reducers import Reducer, TokenPartial
from xorl.ops.exact_sampling_transforms import TOP_K_ALL
from xorl.ops.loss.per_token_ce import compute_per_token_ce


K3_DEBUG_THRESHOLDS = (
    ("1e_minus_6", 1e-6),
    ("1e_minus_4", 1e-4),
    ("1e_minus_3", 1e-3),
    ("1e_minus_2", 1e-2),
    ("1e_minus_1", 1e-1),
    ("1", 1.0),
)


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
    lm_head_fp32: bool = False,
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
    lm_head: Optional[torch.nn.Module] = None,
    logprob_temperature: float = 1.0,
    logprob_top_k: int | torch.Tensor = TOP_K_ALL,
    logprob_top_p: float | torch.Tensor = 1.0,
    logprob_min_p: float | torch.Tensor = 0.0,
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
        loss_reducer: Reduces per-token loss to a scalar partial share. None =>
            ``TokenPartial(scale=valid_mask.sum())`` (legacy local token-mean; does
            not compose across micro-batches/ranks). Pass a shared global-scale
            reducer to make summed partial shares recover the global loss.
        metric_reducer: Reduces per-token /mean metrics (ratio_mean,
            kl_sample_train_k3, entropy_sample). ratio_min/ratio_max stay local
            scalars and bypass it.
        logprob_temperature: Temperature applied to trainer logits before
            selected-token logprob calculation. ``1.0`` is raw policy logprobs;
            setting this to the rollout temperature yields behavior-policy
            semantics for the sampled-token ratio.

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
    valid_mask = labels_flat != ignore_index
    valid_mask_f = valid_mask.float()
    valid_count = valid_mask.sum()

    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=valid_count.float())
    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=valid_count.float())

    # ---- Cross-entropy computation ----
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
        logprob_temperature=logprob_temperature,
        logprob_top_k=logprob_top_k,
        logprob_top_p=logprob_top_p,
        logprob_min_p=logprob_min_p,
    )

    current_support = torch.isfinite(per_token_ce)
    # new logprobs = log p(target) = -CE
    new_logprobs_flat = -per_token_ce.detach()

    # ---- ratio computation (no sanitization) ----
    delta = (new_logprobs_flat - old_logprobs_flat).masked_fill(~valid_mask, 0.0)
    delta = torch.where(current_support, delta.clamp(min=-20.0, max=20.0), torch.full_like(delta, -torch.inf))
    ratio = torch.exp(delta)

    # ---- Per-token policy gradient loss: -(ratio * advantages) ----
    per_token_pg = -(ratio * advantages_flat)
    per_token_pg = per_token_pg.masked_fill(~valid_mask, 0.0)

    # ---- Option B: value from true PG, grad from weighted CE surrogate ----
    true_pg = loss_reducer(per_token_pg, valid_mask_f)

    w = (ratio.detach() * advantages_flat).masked_fill(~valid_mask | ~current_support, 0.0)
    safe_per_token_ce = torch.where(current_support, per_token_ce, torch.zeros_like(per_token_ce))
    surrogate = loss_reducer(w * safe_per_token_ce, valid_mask_f)

    loss = true_pg.detach() + surrogate - surrogate.detach()

    # ±inf identity on empty ranks lets cross-rank MIN/MAX-allreduce ignore empty contributors.
    if valid_mask.any():
        ratio_min = ratio.masked_fill(~valid_mask, float("inf")).min()
        ratio_max = ratio.masked_fill(~valid_mask, float("-inf")).max()
    else:
        ratio_min = ratio.new_tensor(float("inf"))
        ratio_max = ratio.new_tensor(float("-inf"))
    metrics: Dict[str, Any] = {
        "ratio_mean": metric_reducer(ratio, valid_mask_f).detach(),
        "ratio_min": ratio_min.detach(),
        "ratio_max": ratio_max.detach(),
    }

    if compute_kl_stats:
        with torch.no_grad():
            raw_log_ratio = (new_logprobs_flat - old_logprobs_flat).masked_fill(~valid_mask, 0.0)
            log_ratio_full = torch.where(
                current_support,
                raw_log_ratio,
                torch.full_like(raw_log_ratio, -20.0),
            )
            ratio_full = torch.where(current_support, torch.exp(raw_log_ratio), torch.zeros_like(raw_log_ratio))
            per_token_k3 = ratio_full - log_ratio_full - 1.0
            if valid_mask.any():
                k3_max = per_token_k3.masked_fill(~valid_mask, float("-inf")).max()
                logratio_min = log_ratio_full.masked_fill(~valid_mask, float("inf")).min()
                logratio_max = log_ratio_full.masked_fill(~valid_mask, float("-inf")).max()
                abs_logratio_max = log_ratio_full.abs().masked_fill(~valid_mask, float("-inf")).max()
            else:
                k3_max = per_token_k3.new_tensor(float("-inf"))
                logratio_min = log_ratio_full.new_tensor(float("inf"))
                logratio_max = log_ratio_full.new_tensor(float("-inf"))
                abs_logratio_max = log_ratio_full.new_tensor(float("-inf"))
            metrics["kl_sample_train_k3"] = metric_reducer(per_token_k3, valid_mask_f)
            metrics["kl_k3_debug_mean"] = metric_reducer(per_token_k3, valid_mask_f)
            metrics["kl_k3_debug_max"] = k3_max
            metrics["kl_k3_debug_abs_logratio_mean"] = metric_reducer(log_ratio_full.abs(), valid_mask_f)
            metrics["kl_k3_debug_abs_logratio_max"] = abs_logratio_max
            metrics["kl_k3_debug_logratio_mean"] = metric_reducer(log_ratio_full, valid_mask_f)
            metrics["kl_k3_debug_logratio_min"] = logratio_min
            metrics["kl_k3_debug_logratio_max"] = logratio_max
            metrics["kl_k3_debug_frac_logratio_positive"] = metric_reducer((log_ratio_full > 0).float(), valid_mask_f)
            for suffix, threshold in K3_DEBUG_THRESHOLDS:
                metrics[f"kl_k3_debug_frac_gt_{suffix}"] = metric_reducer(
                    (per_token_k3 > threshold).float(), valid_mask_f
                )
            metrics["entropy_sample"] = metric_reducer(-old_logprobs_flat, valid_mask_f)
            metrics["valid_tokens"] = valid_count.item()
            metrics["current_support_fraction"] = metric_reducer(current_support.float(), valid_mask_f)

    # Reshape per-token outputs
    per_token_logprobs = new_logprobs_flat.view(original_shape)
    per_token_loss = per_token_pg.view(original_shape)

    metric_ops = {"ratio_min": "min", "ratio_max": "max"}
    if compute_kl_stats:
        metric_ops.update(
            {
                "kl_k3_debug_max": "max",
                "kl_k3_debug_abs_logratio_max": "max",
                "kl_k3_debug_logratio_min": "min",
                "kl_k3_debug_logratio_max": "max",
            }
        )

    return LossOutput(
        loss=loss,
        per_token_logprobs=per_token_logprobs,
        per_token_loss=per_token_loss,
        metrics=metrics,
        metric_ops=metric_ops,
    )
