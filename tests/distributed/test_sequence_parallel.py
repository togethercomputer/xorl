"""Tests for sequence parallel utilities (non-distributed).

Distributed tests removed -- run with torchrun separately.
"""

import pytest
import torch

from xorl.distributed.sequence_parallel import (
    slice_position_embedding,
)
from xorl.distributed.sequence_parallel.utils import pad_tensor, unpad_tensor

pytestmark = [pytest.mark.distributed]


class TestPaddingUtilities:
    """Test padding, unpadding, and roundtrip for pad_tensor/unpad_tensor."""

    def test_pad_unpad_roundtrip_and_dims(self):
        """Pad on dim=1 and dim=0, verify shapes and values, then roundtrip."""
        # Pad on dim=1
        x = torch.randn(2, 5, 4)
        padded = pad_tensor(x, dim=1, padding_size=3, padding_value=0)
        assert padded.shape == (2, 8, 4)
        assert torch.allclose(padded[:, :5, :], x)
        assert torch.allclose(padded[:, 5:, :], torch.zeros(2, 3, 4))

        # Pad on dim=0 with custom value
        x0 = torch.randn(3, 4, 5)
        padded0 = pad_tensor(x0, dim=0, padding_size=2, padding_value=-1)
        assert padded0.shape == (5, 4, 5)
        assert torch.allclose(padded0[:3, :, :], x0)
        assert torch.allclose(padded0[3:, :, :], torch.full((2, 4, 5), -1.0))

        # Unpad
        unpadded = unpad_tensor(padded, dim=1, padding_size=3)
        assert unpadded.shape == (2, 5, 4)
        assert torch.allclose(unpadded, padded[:, :5, :])

        # Roundtrip
        xr = torch.randn(3, 7, 5)
        assert torch.allclose(xr, unpad_tensor(pad_tensor(xr, dim=1, padding_size=3, padding_value=0), dim=1, padding_size=3))


class TestSlicePositionEmbedding:
    """Test position embedding slicing."""

    def test_slice_position_embedding_no_group(self):
        """No SP group: slicing is a no-op."""
        cos = torch.randn(1, 8, 1)
        sin = torch.randn(1, 8, 1)
        result_cos, result_sin = slice_position_embedding((cos, sin), dim=1, sp_group=None)
        assert torch.equal(result_cos, cos)
        assert torch.equal(result_sin, sin)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
