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
    monkeypatch.delenv("XORL_GDN_PACKED_SEGMENT_LOOP", raising=False)
    monkeypatch.setattr(gated_deltanet, "flashqla_chunk_gated_delta_rule", fake_flashqla_chunk)

    out, _, _ = _tiny_gdn()(torch.randn(1, 8, 128))

    assert len(calls) == 1
    assert calls[0]["q"].shape == (1, 8, 1, 128)
    assert out.shape == (1, 8, 128)


def test_gdn_packed_segment_loop_splits_varlen_kernel_calls(monkeypatch):
    segment_lengths = []

    def fake_fla_chunk(**kwargs):
        segment_lengths.append(kwargs["q"].shape[1])
        assert kwargs["cu_seqlens"] is None
        return kwargs["v"], None

    monkeypatch.setenv("XORL_GDN_BACKEND", "fla")
    monkeypatch.setenv("XORL_GDN_PACKED_SEGMENT_LOOP", "1")
    monkeypatch.setattr(gated_deltanet, "chunk_gated_delta_rule", fake_fla_chunk)

    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    out, _, _ = _tiny_gdn()(torch.randn(1, 8, 128), cu_seqlens=cu_seqlens)

    assert segment_lengths == [3, 5]
    assert out.shape == (1, 8, 128)


def test_gdn_fused_projection_env_matches_split_projection_path(monkeypatch):
    calls = []

    def fake_fla_chunk(**kwargs):
        calls.append(
            {
                "q": kwargs["q"].detach().clone(),
                "k": kwargs["k"].detach().clone(),
                "v": kwargs["v"].detach().clone(),
                "g": kwargs["g"].detach().clone(),
                "beta": kwargs["beta"].detach().clone(),
            }
        )
        return kwargs["v"], None

    layer = GatedDeltaNet(
        hidden_size=16,
        expand_v=1.0,
        head_dim=4,
        num_heads=2,
        num_v_heads=2,
        mode="chunk",
        use_gate=True,
        use_short_conv=False,
    )
    layer.train()
    hidden_states = torch.randn(1, 5, 16)

    monkeypatch.setenv("XORL_GDN_BACKEND", "fla")
    monkeypatch.delenv("XORL_GDN_FUSED_PROJECTIONS", raising=False)
    monkeypatch.setattr(gated_deltanet, "chunk_gated_delta_rule", fake_fla_chunk)
    split_out, _, _ = layer(hidden_states)
    split_call = calls[-1]

    monkeypatch.setenv("XORL_GDN_FUSED_PROJECTIONS", "1")
    fused_out, _, _ = layer(hidden_states)
    fused_call = calls[-1]

    torch.testing.assert_close(fused_out, split_out, atol=1e-6, rtol=1e-6)
    for key in ("q", "k", "v", "g", "beta"):
        torch.testing.assert_close(fused_call[key], split_call[key], atol=1e-6, rtol=1e-6)
