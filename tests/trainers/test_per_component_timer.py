"""Smoke tests for the per-component timing infra.

These tests use a fake decoder-layer-shaped nn.Module so they don't require
the heavy GLM-5 / Qwen3 dependencies — what we're checking is the hook
attachment + event recording logic, which is model-agnostic.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from xorl.trainers.per_component_timer import PerComponentTimer


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = nn.Linear(64, 64)
        self.q = nn.Linear(64, 64)

    def forward(self, x):
        return self.q(x) + self.indexer(x)


class _FakeMoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 64)
        self.experts = nn.Linear(64, 64)
        self.shared_experts = nn.Linear(64, 64)

    def forward(self, x):
        # gate output is used to mimic GLM-5's routing weight pathway, so
        # that backward propagates through the gate module.
        weights = self.gate(x).sigmoid()
        return weights * self.experts(x) + self.shared_experts(x)


class _FakeGlmDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(64)
        self.self_attn = _FakeAttn()
        self.post_attention_layernorm = nn.LayerNorm(64)
        self.mlp = _FakeMoEBlock()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _FakeQwen3DecoderLayer(nn.Module):
    """Qwen3-style: no indexer, no shared_experts."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(64)

        class _FakeAttnNoIndexer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(64, 64)

            def forward(self, x):
                return self.q(x)

        class _FakeMoENoShared(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(64, 64)
                self.experts = nn.Linear(64, 64)

            def forward(self, x):
                weights = self.gate(x).sigmoid()
                return weights * self.experts(x)

        self.self_attn = _FakeAttnNoIndexer()
        self.post_attention_layernorm = nn.LayerNorm(64)
        self.mlp = _FakeMoENoShared()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def _make_model(layer_cls, n_layers=3):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([layer_cls() for _ in range(n_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return _Model()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for event-based timing")
def test_model_style_hooks_capture_present_and_skip_missing_phases():
    model = _make_model(_FakeGlmDecoderLayer).cuda()
    timer = PerComponentTimer(enabled=True)
    n = timer.attach(model)
    assert n == 3

    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    timer.start_step()
    timer.set_mode("fwd")
    out = model(x)
    loss = out.sum()
    timer.set_mode("bwd")
    loss.backward()
    timer.set_mode("idle")
    result = timer.end_step()

    expected_fwd = {
        "fwd_norm/input",
        "fwd_attn/total",
        "fwd_attn/indexer",
        "fwd_norm/post_attn",
        "fwd_mlp_or_moe/total",
        "fwd_moe/gate",
        "fwd_moe/experts",
        "fwd_moe/shared",
    }
    expected_bwd = {p.replace("fwd_", "bwd_", 1) for p in expected_fwd}
    assert expected_fwd.issubset(result.keys()), f"missing fwd phases: {expected_fwd - result.keys()}"
    assert expected_bwd.issubset(result.keys()), f"missing bwd phases: {expected_bwd - result.keys()}"
    # All recorded times should be non-negative.
    for phase, secs in result.items():
        assert secs >= 0.0, f"{phase} reported negative time {secs}"

    model = _make_model(_FakeQwen3DecoderLayer).cuda()
    timer = PerComponentTimer(enabled=True)
    n = timer.attach(model)
    assert n == 3

    x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
    timer.start_step()
    timer.set_mode("fwd")
    out = model(x)
    loss = out.sum()
    timer.set_mode("bwd")
    loss.backward()
    timer.set_mode("idle")
    result = timer.end_step()

    # Indexer and shared_experts are absent on Qwen3-style layers — no phases.
    assert "fwd_attn/indexer" not in result
    assert "bwd_attn/indexer" not in result
    assert "fwd_moe/shared" not in result
    assert "bwd_moe/shared" not in result
    # Common phases should still be present.
    assert "fwd_attn/total" in result
    assert "fwd_moe/experts" in result
