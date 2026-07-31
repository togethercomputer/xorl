"""Reachability gate for the families-v2 norm adoption.

The structure gates in tests/ops/test_bi_families_v2_norm.py prove the two
realizations of the v2 tree agree bitwise. This file proves the trainer's norm
entry points actually reach those kernels, and that the kill switch takes them
back to v1. Unreachable kernels gate nothing.
"""

import pytest
import torch

import xorl.models.layers.normalization as normalization
from xorl.ops.bi_families_v2 import families_v2_enabled, rms_norm_v2


EPS = 1e-6
H = 3840

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
pytestmark = pytest.mark.gpu


def _payload(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16).cuda()


@requires_cuda
def test_norm_dispatcher_routes_to_families_v2(monkeypatch):
    """The trainer's norm entry points must reach the v2 kernels, and the kill
    switch must take them back to v1. Unreachable kernels gate nothing."""
    monkeypatch.delenv("XORL_FAMILIES_V2", raising=False)
    monkeypatch.delenv("SGLANG_FAMILIES_V2", raising=False)
    assert families_v2_enabled() is True, "families v2 is default on"

    x, residual, weight = _payload((64, H), 15), _payload((64, H), 16), _payload((H,), 17)
    expected, expected_residual = rms_norm_v2(x, weight, EPS, residual=residual)
    expected_plain = rms_norm_v2(x, weight, EPS)

    assert torch.equal(normalization.fast_sglang_rms_norm(x, weight, EPS), expected_plain)
    assert torch.equal(normalization.fast_batch_invariant_rms_norm(x, weight, EPS), expected_plain)

    fused_out, fused_residual = normalization.fast_sglang_residual_rms_norm(x, residual, weight, EPS)
    assert torch.equal(fused_out, expected)
    assert torch.equal(fused_residual, expected_residual)

    # Rollback: assert on the kernel selected, not on the bits. The two trees
    # agree on most values, so a bit comparison would be a weak signal.
    import xorl.ops.bi_families_v2 as module

    reached = []
    original = module.rms_norm_v2
    monkeypatch.setattr(normalization, "rms_norm_v2", lambda *a, **k: (reached.append(1), original(*a, **k))[1])
    monkeypatch.setenv("XORL_FAMILIES_V2", "0")
    assert families_v2_enabled() is False, "the kill switch must roll this engine back"
    normalization.fast_batch_invariant_rms_norm(x, weight, EPS)
    normalization.fast_sglang_residual_rms_norm(x, residual, weight, EPS)
    assert not reached, "the kill switch left the dispatcher on the v2 kernels"


def test_kill_switch_rolls_back_on_either_engine_variable(monkeypatch):
    """One flag, both engines. A setting that rolls back only one of the two
    would put the trainer and the sampler on different trees."""
    for variable in ("XORL_FAMILIES_V2", "SGLANG_FAMILIES_V2"):
        monkeypatch.delenv("XORL_FAMILIES_V2", raising=False)
        monkeypatch.delenv("SGLANG_FAMILIES_V2", raising=False)
        assert families_v2_enabled() is True
        monkeypatch.setenv(variable, "0")
        assert families_v2_enabled() is False, f"{variable}=0 must roll back"
