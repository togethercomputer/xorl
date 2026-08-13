from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from xorl.ops.loss.importance_sampling_loss import K3_DEBUG_THRESHOLDS
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce
from xorl.ops.loss.reducers import Reducer, TokenPartial


def cispo_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
    clip_low_threshold: float = 0.0,
    clip_high_threshold: float = 4.0,
    num_chunks: int = 8,
    ce_mode: str = "compiled",
    tp_group: Optional[dist.ProcessGroup] = None,
    compute_kl_stats: bool = False,
    lm_head_fp32: bool = False,
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
    lm_head: Optional[torch.nn.Module] = None,
    logprob_temperature: float = 1.0,
) -> LossOutput:
    """Compute Tinker-compatible CISPO.

    CISPO clips the importance ratio and uses it as a detached coefficient on
    ``log p_theta``. Unlike PPO objective clipping, this retains a gradient for
    every valid token::

        ratio = exp(target_logprobs - sampling_logprobs)
        clipped = clamp(ratio, clip_low_threshold, clip_high_threshold)
        loss = -(clipped.detach() * target_logprobs * advantages).mean()

    The default absolute ratio bounds, ``[0, 4]``, match Tinker's one-sided
    CISPO default and the MiniMax-M1 prescription of disabling the lower bound.
    """
    if clip_low_threshold < 0.0:
        raise ValueError("clip_low_threshold must be non-negative")
    if clip_high_threshold < clip_low_threshold:
        raise ValueError("clip_high_threshold must be >= clip_low_threshold")

    original_shape = labels.shape
    hidden_size = hidden_states.size(-1)

    labels_flat = labels.reshape(-1)
    hidden_states_flat = hidden_states.reshape(-1, hidden_size)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)

    valid_mask = labels_flat != ignore_index
    valid_mask_f = valid_mask.float()
    valid_count = valid_mask.sum()

    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=valid_count.float())
    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=valid_count.float())

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
    )

    new_logprobs_flat = -per_token_ce.detach()
    log_ratio = (new_logprobs_flat - old_logprobs_flat).masked_fill(~valid_mask, 0.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, clip_low_threshold, clip_high_threshold)

    coefficient = (clipped_ratio.detach() * advantages_flat).masked_fill(~valid_mask, 0.0)
    per_token_loss_flat = coefficient * per_token_ce
    loss = loss_reducer(per_token_loss_flat, valid_mask_f)

    is_clipped = (clipped_ratio != ratio) & valid_mask
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
        "clip_fraction": metric_reducer(is_clipped.float(), valid_mask_f).detach(),
    }

    metric_ops = {"ratio_min": "min", "ratio_max": "max"}
    if compute_kl_stats:
        with torch.no_grad():
            per_token_k3 = ratio - log_ratio - 1.0
            if valid_mask.any():
                k3_max = per_token_k3.masked_fill(~valid_mask, float("-inf")).max()
                logratio_min = log_ratio.masked_fill(~valid_mask, float("inf")).min()
                logratio_max = log_ratio.masked_fill(~valid_mask, float("-inf")).max()
                abs_logratio_max = log_ratio.abs().masked_fill(~valid_mask, float("-inf")).max()
            else:
                k3_max = per_token_k3.new_tensor(float("-inf"))
                logratio_min = log_ratio.new_tensor(float("inf"))
                logratio_max = log_ratio.new_tensor(float("-inf"))
                abs_logratio_max = log_ratio.new_tensor(float("-inf"))
            metrics["kl_sample_train_k3"] = metric_reducer(per_token_k3, valid_mask_f)
            metrics["kl_k3_debug_mean"] = metric_reducer(per_token_k3, valid_mask_f)
            metrics["kl_k3_debug_max"] = k3_max
            metrics["kl_k3_debug_abs_logratio_mean"] = metric_reducer(log_ratio.abs(), valid_mask_f)
            metrics["kl_k3_debug_abs_logratio_max"] = abs_logratio_max
            metrics["kl_k3_debug_logratio_mean"] = metric_reducer(log_ratio, valid_mask_f)
            metrics["kl_k3_debug_logratio_min"] = logratio_min
            metrics["kl_k3_debug_logratio_max"] = logratio_max
            metrics["kl_k3_debug_frac_logratio_positive"] = metric_reducer((log_ratio > 0).float(), valid_mask_f)
            for suffix, threshold in K3_DEBUG_THRESHOLDS:
                metrics[f"kl_k3_debug_frac_gt_{suffix}"] = metric_reducer(
                    (per_token_k3 > threshold).float(), valid_mask_f
                )
            metrics["entropy_sample"] = metric_reducer(-old_logprobs_flat, valid_mask_f)
            metrics["valid_tokens"] = valid_count.item()
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
        per_token_logprobs=new_logprobs_flat.view(original_shape),
        per_token_loss=per_token_loss_flat.detach().view(original_shape),
        metrics=metrics,
        metric_ops=metric_ops,
    )
