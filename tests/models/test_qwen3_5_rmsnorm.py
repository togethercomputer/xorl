import os

import pytest
import torch

from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5 import modeling_qwen3_5
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel
from xorl.ops.batch_invariant_ops import set_trunk_linear_contract


def test_qwen3_5_gdn_contract_defaults_to_certified_v1_family(monkeypatch):
    monkeypatch.setenv("XORL_BI_GDN", "1")
    monkeypatch.delenv("XORL_FAMILIES_V2", raising=False)
    modeling_qwen3_5._default_qwen35_gdn_contract_family()
    assert os.environ["XORL_FAMILIES_V2"] == "0"

    monkeypatch.setenv("XORL_FAMILIES_V2", "1")
    modeling_qwen3_5._default_qwen35_gdn_contract_family()
    assert os.environ["XORL_FAMILIES_V2"] == "1"


def test_qwen3_5_sglang_fused_rmsnorm_routes_serving_families(monkeypatch):
    calls = []

    def fake_native(hidden_states, weight, variance_epsilon):
        calls.append("native")
        return hidden_states + 1

    def fake_residual(hidden_states, weight, variance_epsilon):
        calls.append("residual")
        return hidden_states + 3

    def fake_family1(hidden_states, weight, variance_epsilon):
        calls.append("family1")
        return hidden_states + 5

    monkeypatch.setattr(modeling_qwen3_5, "native_zero_centered_rms_norm", fake_native)
    monkeypatch.setattr(
        modeling_qwen3_5,
        "native_zero_centered_rms_norm_without_batch_invariant",
        fake_residual,
    )
    monkeypatch.setattr(
        modeling_qwen3_5,
        "fast_zero_centered_batch_invariant_rms_norm",
        fake_family1,
    )

    set_rmsnorm_mode("sglang_fused")
    try:
        norm = modeling_qwen3_5.Qwen3_5RMSNorm(4)
        x = torch.ones(2, 4)

        assert torch.equal(norm(x), x + 1)
        assert calls[-1] == "native"
        assert torch.equal(norm(x, force_sglang_residual=True), x + 3)
        assert calls[-1] == "residual"

        set_trunk_linear_contract(True)
        try:
            assert torch.equal(norm(x), x + 5)
            assert calls[-1] == "family1"
            assert torch.equal(norm(x, force_sglang_residual=True), x + 3)
            assert calls[-1] == "residual"
        finally:
            set_trunk_linear_contract(False)
    finally:
        set_rmsnorm_mode("native")


class CaptureNorm(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.force_values = []

    def forward(self, hidden_states, *, force_sglang_residual=False, **kwargs):
        self.force_values.append(force_sglang_residual)
        if kwargs.get("prenorm"):
            return hidden_states, kwargs.get("residual")
        return hidden_states


class IdentityAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


def _tiny_config(**overrides) -> Qwen3_5Config:
    kwargs = dict(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        layer_types=["full_attention", "full_attention"],
        _attn_implementation="eager",
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return Qwen3_5Config(**kwargs)


@pytest.mark.parametrize(
    ("layer_idx", "mode", "expected"),
    [(0, "sglang_fused", False), (1, "native", False), (1, "sglang", True), (1, "sglang_fused", True)],
)
def test_qwen3_5_layer_input_norm_selects_residual_family(layer_idx, mode, expected):
    layer = Qwen3_5DecoderLayer(_tiny_config(), layer_idx=layer_idx)
    norm = CaptureNorm(mode)
    layer.input_layernorm = norm
    layer.self_attn = IdentityAttention()
    layer.post_attention_layernorm = CaptureNorm(mode)
    layer.mlp = torch.nn.Identity()

    hidden = torch.ones(1, 2, 8)
    layer(hidden, position_embeddings=(hidden, hidden))

    assert norm.force_values == [expected]


@pytest.mark.parametrize(("mode", "expected"), [("native", False), ("sglang", True), ("sglang_fused", True)])
def test_qwen3_5_final_norm_selects_residual_family(mode, expected):
    class StubLayer(torch.nn.Module):
        layer_type = "full_attention"

        def forward(self, hidden_states, *args, **kwargs):
            return (hidden_states,)

    model = Qwen3_5TextModel(_tiny_config())
    model.layers = torch.nn.ModuleList([StubLayer()])
    norm = CaptureNorm(mode)
    model.norm = norm

    model(input_ids=torch.tensor([[0, 1]]))

    assert norm.force_values == [expected]
