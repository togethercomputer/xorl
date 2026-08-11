"""Reducer protocol and canonical denominator policies for loss aggregation.

A ``Reducer`` collapses a ``(B, S)`` tensor to a scalar over a
caller-supplied denominator policy. Partial shares sum across micro-batches
and ``all_reduce(SUM)`` across ranks to the globally-correct value.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Reducer(Protocol):
    """``(values, mask) -> scalar`` partial share over a pre-computed
    denominator. Partial shares sum across micro-batches and ``all_reduce(SUM)``
    across ranks.
    """

    def __call__(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class TokenPartial:
    """Flat masked sum divided by a caller-supplied ``scale``."""

    scale: torch.Tensor

    def __call__(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / self.scale.clamp(min=1.0)


__all__ = [
    "Reducer",
    "TokenPartial",
]
