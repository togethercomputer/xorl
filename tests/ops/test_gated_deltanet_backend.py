import warnings

import pytest
import torch

import xorl.ops.linear_attention.layers.gated_deltanet as gated_deltanet
from xorl.ops.linear_attention import GatedDeltaNet


pytestmark = [pytest.mark.cpu]


def _tiny_gdn() -> GatedDeltaNet:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = GatedDeltaNet(
            hidden_size=128,
            expand_v=1.0,
            head_dim=128,
            num_heads=1,
            num_v_heads=1,
            mode="chunk",
            use_gate=False,
            use_short_conv=False,
        )
    layer.train()
    return layer


def test_gdn_backend_env_dispatches_to_flashqla(monkeypatch):
    calls = []

    def fake_flashqla_chunk(**kwargs):
        calls.append(kwargs)
        assert "cp_context" not in kwargs
        return kwargs["v"], None

    monkeypatch.setenv("XORL_GDN_BACKEND", "flashqla")
    monkeypatch.setattr(gated_deltanet, "flashqla_chunk_gated_delta_rule", fake_flashqla_chunk)

    out, _, _ = _tiny_gdn()(torch.randn(1, 8, 128))

    assert len(calls) == 1
    assert calls[0]["q"].shape == (1, 8, 1, 128)
    assert out.shape == (1, 8, 128)
