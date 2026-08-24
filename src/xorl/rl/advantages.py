"""Skip-observation token-level GAE (SAO, arXiv:2607.07508, Eq. 4-5).

Agentic trajectories interleave model actions with environment observations
the model did not generate. Standard GAE bootstraps across every adjacent
token, which propagates critic noise through observation tokens. The
skip-observation estimator chains the Bellman recursion across *action tokens
only*: the TD target of an action token bootstraps from the value of the next
action token, bridging any observation gap in between.

Pure Python (no torch/numpy) so it can be lifted verbatim into xorl-client,
which has no torch dependency. Sequences follow the server's target-aligned
convention: index t describes the t-th target token, matching the per-token
values returned by the ``value_prediction`` loss and the ``advantages`` /
``returns`` fields of a training datum.

Note: the server's packer masks tokens where ``advantages == 0.0``; that is
the desired behavior for non-action tokens (this module emits exactly 0.0
there), but a true action-token advantage of exactly 0.0 would also be
masked. Nudge such values by a tiny epsilon if that matters for your reward
scale.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def compute_skip_observation_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    action_mask: Optional[Sequence[int]] = None,
    gamma: float = 1.0,
    lam: float = 1.0,
    bootstrap_value: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """Compute per-token advantages and value targets across action tokens.

    Args:
        rewards: Per-token rewards, length T (typically zero everywhere except
            the final action token of the trajectory).
        values: Per-token value predictions V(s_t), length T (e.g. from a
            ``forward`` call with ``loss_fn="value_prediction"``).
        action_mask: Per-token 0/1 mask, length T; 1 marks model-generated
            (action) tokens. ``None`` treats every token as an action token,
            which reduces to standard token-level GAE.
        gamma: Discount factor.
        lam: GAE lambda. For length-adaptive GAE (VAPO), pass
            ``1 - 1 / (alpha * num_action_tokens)``.
        bootstrap_value: Value bootstrapped after the last action token
            (0.0 for terminated trajectories).

    Returns:
        ``(advantages, returns)`` lists of length T. Non-action tokens carry
        0.0 in both; action tokens carry the skip-observation GAE advantage
        and the corresponding value target ``R_t = A_t + V(s_t)``.
    """
    n = len(rewards)
    if len(values) != n:
        raise ValueError(f"rewards ({n}) and values ({len(values)}) must have the same length")
    if action_mask is None:
        action_indices = list(range(n))
    else:
        if len(action_mask) != n:
            raise ValueError(f"action_mask ({len(action_mask)}) must match rewards ({n})")
        action_indices = [t for t in range(n) if action_mask[t]]

    advantages = [0.0] * n
    returns = [0.0] * n

    next_advantage = 0.0
    next_value = float(bootstrap_value)
    for t in reversed(action_indices):
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        advantage = delta + gamma * lam * next_advantage
        advantages[t] = advantage
        returns[t] = advantage + float(values[t])
        next_advantage = advantage
        next_value = float(values[t])

    return advantages, returns


def explained_variance(
    value_error_sq_mean: float,
    return_mean: float,
    return_sq_mean: float,
) -> float:
    """Critic explained variance from the ``value_loss`` moment metrics.

    ``EV = 1 - E[(R - V)^2] / Var(R)`` with ``Var(R) = E[R^2] - E[R]^2``.
    The three inputs are exactly the (globally normalized) ``value_error_sq_mean``,
    ``return_mean``, and ``return_sq_mean`` metrics a ``value_loss``
    forward_backward reports, so EV composes correctly across micro-batches
    and ranks. EV is the paper's key critic-health diagnostic (SAO Fig. 4a):
    it should climb toward 1.0 as the critic converges; near 0 the critic is
    no better than predicting the mean return.

    Returns NaN when the return distribution is (numerically) constant, where
    explained variance is undefined.
    """
    return_variance = return_sq_mean - return_mean * return_mean
    if not math.isfinite(return_variance) or return_variance <= 1e-12:
        return float("nan")
    return 1.0 - value_error_sq_mean / return_variance
