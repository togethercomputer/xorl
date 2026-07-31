from __future__ import annotations

from typing import Literal

import torch


ReductionMode = Literal["token_mean", "sample_mean", "slime_sum_of_sample_mean", "token_sum"]


def reduce_token_or_sample_mean(
    values: torch.Tensor,
    masks: torch.Tensor,
    mode: ReductionMode,
) -> torch.Tensor:
    """Reduce token-aligned values with an explicit normalization contract.

    Modes:
    - ``token_mean``: global token-weighted mean over all valid tokens.
    - ``sample_mean``: arithmetic mean of non-empty per-sample means.
    - ``slime_sum_of_sample_mean``: sum of per-sample means, matching Slime's
      train-time reducer before its later global-batch divisor.
    - ``token_sum``: masked sum, useful when normalization is deferred.
    """
    mask_f = masks.float()
    masked_sum = (values * mask_f).sum()

    if mode == "token_sum":
        return masked_sum
    if mode == "token_mean":
        return masked_sum / mask_f.sum().clamp(min=1.0)

    sample_denoms = mask_f.sum(dim=-1).clamp(min=1.0)
    sample_means = (values * mask_f).sum(dim=-1) / sample_denoms
    nonempty = mask_f.sum(dim=-1) > 0

    if mode == "slime_sum_of_sample_mean":
        return sample_means.masked_fill(~nonempty, 0.0).sum()
    if mode == "sample_mean":
        return sample_means.masked_select(nonempty).sum() / nonempty.sum().clamp(min=1)

    raise ValueError(f"Unknown reduction mode: {mode}")
