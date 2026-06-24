"""Shared fakes for sparse-delta tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FakeEncoded:
    """Stand-in for ``delta_encoding.encoding.types.EncodedDelta``.

    ``flat_deltas`` aliases ``flat_indices`` so this fake satisfies both the
    ``EncodedDelta`` and ``MmapPackedFile.flat_deltas_view`` shapes the
    sparse-delta code relies on.
    """

    flat_indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]

    @property
    def flat_deltas(self) -> torch.Tensor:
        return self.flat_indices
