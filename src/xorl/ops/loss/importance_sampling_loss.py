from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.per_token_ce import compute_per_token_ce
from xorl.ops.loss.reducers import Reducer, TokenPartial


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
    )

    # new logprobs = log p(target) = -CE
    new_logprobs_flat = -per_token_ce.detach()

    # ---- ratio computation (no sanitization) ----
    delta = new_logprobs_flat - old_logprobs_flat
    delta = delta.masked_fill(~valid_mask, 0.0)
    delta = delta.clamp(min=-20.0, max=20.0)
    ratio = torch.exp(delta)

    # ---- Per-token policy gradient loss: -(ratio * advantages) ----
    per_token_pg = -(ratio * advantages_flat)
    per_token_pg = per_token_pg.masked_fill(~valid_mask, 0.0)

    # ---- Option B: value from true PG, grad from weighted CE surrogate ----
    true_pg = loss_reducer(per_token_pg, valid_mask_f)

    w = (ratio.detach() * advantages_flat).masked_fill(~valid_mask, 0.0)
    surrogate = loss_reducer(w * per_token_ce, valid_mask_f)

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
            log_ratio_full = (new_logprobs_flat - old_logprobs_flat).masked_fill(~valid_mask, 0.0)
            ratio_full = torch.exp(log_ratio_full)
            per_token_k3 = ratio_full - log_ratio_full - 1.0
            metrics["kl_sample_train_k3"] = metric_reducer(per_token_k3, valid_mask_f)
            metrics["entropy_sample"] = metric_reducer(-old_logprobs_flat, valid_mask_f)
            metrics["valid_tokens"] = valid_count.item()

    # Reshape per-token outputs
    per_token_logprobs = new_logprobs_flat.view(original_shape)
    per_token_loss = per_token_pg.view(original_shape)

    return LossOutput(
        loss=loss,
        per_token_logprobs=per_token_logprobs,
        per_token_loss=per_token_loss,
        metrics=metrics,
        metric_ops={"ratio_min": "min", "ratio_max": "max"},
    )
