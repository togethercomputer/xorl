"""Pure-torch FWHT fallback for ``rotate_activation``.

The on-disk Flash routed-expert weights need ``rotate_activation`` to
work on every C4 layer (DSA indexer with ``rotate=True`` compressor).
The kernel ships in the ``fast_hadamard_transform`` package; we maintain
a pure-torch fallback so CPU CI and lean dev images don't need that
package to import the model.
"""

import pytest
import torch


pytestmark = pytest.mark.cpu


def test_fwht_round_trip_orthonormal():
    """Orthonormal Hadamard squared = identity: H @ H @ x == x (modulo
    rounding) when ``scale = 1 / sqrt(D)`` is applied at each FWHT call.
    """
    from xorl.ops.dsv4.utils import _fwht_torch

    torch.manual_seed(0)
    for D in (4, 16, 64, 256):
        x = torch.randn(3, D, dtype=torch.bfloat16)
        scale = D**-0.5
        once = _fwht_torch(x, scale)
        twice = _fwht_torch(once, scale)
        # Two ortho-Hadamards = identity, but bf16 accumulates ~5e-3 error.
        torch.testing.assert_close(twice, x, atol=8e-3, rtol=8e-3)


def test_fwht_known_pattern():
    """``H_4 @ [1, 0, 0, 0] = [1, 1, 1, 1]`` (unnormalized; scale=1)."""
    from xorl.ops.dsv4.utils import _fwht_torch

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    out = _fwht_torch(x, scale=1.0)
    expected = torch.ones(1, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_fwht_preserves_l2_norm_when_orthonormal():
    """An orthonormal transform preserves L2 norm."""
    from xorl.ops.dsv4.utils import _fwht_torch

    torch.manual_seed(1)
    D = 128
    x = torch.randn(2, 5, D, dtype=torch.bfloat16)
    out = _fwht_torch(x, scale=D**-0.5)
    n_x = x.float().pow(2).sum(dim=-1).sqrt()
    n_out = out.float().pow(2).sum(dim=-1).sqrt()
    torch.testing.assert_close(n_x, n_out, atol=1e-2, rtol=1e-2)


def test_fwht_rejects_non_power_of_two():
    from xorl.ops.dsv4.utils import _fwht_torch

    x = torch.randn(3, 6, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power-of-2"):
        _fwht_torch(x, 1.0)


def test_rotate_activation_dispatches_to_fallback_when_kernel_missing(monkeypatch):
    """When ``fast_hadamard_transform`` isn't installed, ``rotate_activation``
    must transparently use the torch FWHT (no AssertionError)."""
    from xorl.ops.dsv4 import utils

    monkeypatch.setattr(utils, "_fast_hadamard_transform", None)
    torch.manual_seed(2)
    x = torch.randn(2, 4, 64, dtype=torch.bfloat16)  # last dim = 64 (power of 2)
    out = utils.rotate_activation(x)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
