"""Value-function (critic) losses for RL training with a scalar value head.

The value head is a ``hidden_size -> 1`` projection consumed the same way the
lm_head is consumed by the CE-based losses: the loss receives the head's
effective (LoRA-folded) weight and applies it to the trunk's hidden states,
so gradients reach the head's adapter factors and the trunk in one backward.

Both functions follow the server loss contract: per-token values are reduced
through the injected ``Reducer`` (the server passes raw-sum ``TokenPartial``
reducers and defers normalization to ``optim_step``), and per-token outputs
ride the standard per-token channels of :class:`LossOutput`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.reducers import Reducer, TokenPartial


def _project_values(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project flattened hidden states through the scalar value head in fp32."""
    if weight.dim() != 2 or weight.size(0) != 1:
        raise ValueError(f"Value head weight must have shape (1, hidden_size), got {tuple(weight.shape)}")
    hidden_flat = hidden_states.reshape(-1, hidden_states.size(-1))
    return F.linear(hidden_flat.float(), weight.float()).squeeze(-1)


def value_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_range: float = 0.0,
    vf_coef: float = 1.0,
    ignore_index: int = -100,
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
) -> LossOutput:
    """Masked squared-error value loss against per-token returns.

    Args:
        hidden_states: Trunk hidden states, shape (batch, seq_len, hidden).
        weight: Effective value-head weight, shape (1, hidden). For a LoRA
            value head, pass the straight-through folded weight so gradients
            reach the adapter factors.
        labels: Target-aligned token ids, shape (batch, seq_len). Tokens equal
            to ``ignore_index`` are masked out (the packer folds the client's
            ``weights`` mask into labels).
        returns: Per-token return targets R_t, shape (batch, seq_len),
            target-aligned like ``advantages``.
        old_values: Optional per-token values from before the update, enabling
            PPO-style value clipping.
        clip_range: If > 0 and ``old_values`` is provided, uses the clipped
            value objective ``max((v-R)^2, (clip(v)-R)^2)``.
        vf_coef: Multiplier on the final loss.
        loss_reducer / metric_reducer: Reduction policies; ``None`` falls back
            to the local token mean (does not compose across ranks).

    Returns:
        LossOutput with ``loss = vf_coef * 0.5 * reduce(vf_error_sq)``,
        per-token values in ``per_token_logprobs`` (the generic per-token
        channel), and per-token squared errors in ``per_token_loss``.
    """
    original_shape = labels.shape
    labels_flat = labels.reshape(-1)
    returns_flat = returns.reshape(-1).float()

    valid_mask = labels_flat != ignore_index
    valid_mask_f = valid_mask.float()
    valid_count = valid_mask.sum()

    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=valid_count.float())
    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=valid_count.float())

    values = _project_values(hidden_states, weight)

    error_sq = (values - returns_flat).square()
    clip_fraction = None
    if clip_range > 0.0 and old_values is not None:
        old_values_flat = old_values.reshape(-1).float()
        values_clipped = old_values_flat + (values - old_values_flat).clamp(-clip_range, clip_range)
        clipped_error_sq = (values_clipped - returns_flat).square()
        clip_fraction = metric_reducer((clipped_error_sq > error_sq).float(), valid_mask_f).detach()
        vf_error_sq = torch.maximum(error_sq, clipped_error_sq)
    else:
        vf_error_sq = error_sq

    vf_error_sq = vf_error_sq * valid_mask_f
    loss = vf_coef * 0.5 * loss_reducer(vf_error_sq, valid_mask_f)

    with torch.no_grad():
        # Sum-composable moments: downstream normalization by the global valid
        # token count turns these into global means, from which explained
        # variance can be derived (EV = 1 - E[(R-V)^2] / (E[R^2] - E[R]^2)).
        metrics: Dict[str, Any] = {
            "value_mean": metric_reducer(values, valid_mask_f),
            "value_sq_mean": metric_reducer(values.square(), valid_mask_f),
            "return_mean": metric_reducer(returns_flat, valid_mask_f),
            "return_sq_mean": metric_reducer(returns_flat.square(), valid_mask_f),
            "value_error_sq_mean": metric_reducer(error_sq, valid_mask_f),
            "valid_tokens": int(valid_count.item()),
        }
        if clip_fraction is not None:
            metrics["value_clip_fraction"] = clip_fraction

    return LossOutput(
        loss=loss,
        per_token_logprobs=values.detach().view(original_shape),
        per_token_loss=(vf_error_sq.detach()).view(original_shape),
        metrics=metrics,
        metric_ops={},
    )


def value_prediction_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    loss_reducer: Optional[Reducer] = None,
    metric_reducer: Optional[Reducer] = None,
) -> LossOutput:
    """Forward-only per-token value prediction V(s_t).

    Intended for the no-grad ``forward`` op: the client reads the per-token
    values from the per-token output channel and computes advantages (GAE)
    itself. The loss is an exact zero that keeps a valid autograd graph, so an
    accidental ``forward_backward`` produces zero gradients instead of failing.

    ``labels`` is used only for mask-aware metrics; values are returned for
    every position so the client can apply its own action mask.
    """
    original_shape = labels.shape
    labels_flat = labels.reshape(-1)
    valid_mask_f = (labels_flat != ignore_index).float()
    valid_count = valid_mask_f.sum()

    if metric_reducer is None:
        metric_reducer = TokenPartial(scale=valid_count.float())

    values = _project_values(hidden_states, weight)
    loss = (values * 0.0).sum()

    with torch.no_grad():
        metrics = {
            "value_mean": metric_reducer(values, valid_mask_f),
            "value_sq_mean": metric_reducer(values.square(), valid_mask_f),
            "valid_tokens": int(valid_count.item()),
        }

    return LossOutput(
        loss=loss,
        per_token_logprobs=values.detach().view(original_shape),
        per_token_loss=None,
        metrics=metrics,
        metric_ops={},
    )
