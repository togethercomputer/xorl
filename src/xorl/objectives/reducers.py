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


@dataclass(frozen=True)
class SequencePartial:
    """Sum of per-segment token-means, divided by a caller-supplied ``scale``.

    Segment boundaries are flat across ``(values * mask).reshape(-1)``:

    - ``cu_seqlens_local: (N+1,)`` — shard-local segment extents. Under CP each
      rank's slice sums to its segment's local contribution.
    - ``seq_lengths_global: (N,)`` — pre-CP-shard token count per segment, used
      as the per-segment denominator so partial shares from each CP rank sum
      to the correct per-segment mean.
    """

    scale: torch.Tensor
    cu_seqlens_local: torch.Tensor
    seq_lengths_global: torch.Tensor

    def __call__(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flat = (values * mask).reshape(-1)
        seg_lengths_local = self.cu_seqlens_local.diff()
        n_segments = seg_lengths_local.numel()
        seg_ids = torch.repeat_interleave(
            torch.arange(n_segments, device=flat.device),
            seg_lengths_local,
        )
        seg_sums = torch.zeros(n_segments, dtype=flat.dtype, device=flat.device).index_add(0, seg_ids, flat)
        seg_means = seg_sums / self.seq_lengths_global.clamp(min=1.0)
        return seg_means.sum() / self.scale.clamp(min=1.0)


__all__ = [
    "Reducer",
    "SequencePartial",
    "TokenPartial",
]
