import pytest
import torch

from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.transformers.qwen3_5 import modeling_qwen3_5
from xorl.models.transformers.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextModel


def test_qwen3_5_exact_norm_selection_is_structural(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modeling_qwen3_5,
        "fast_zero_centered_batch_invariant_residual_rms_norm",
        lambda hidden, _weight, _eps: calls.append("exact") or hidden + 1,
    )
    monkeypatch.setattr(
        modeling_qwen3_5,
        "native_zero_centered_rms_norm_without_batch_invariant",
        lambda hidden, _weight, _eps: calls.append("legacy") or hidden + 2,
    )

    set_rmsnorm_mode("sglang")
    try:
        norm = modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=True)
        x = torch.ones(2, 4)
        assert torch.equal(norm(x, force_sglang_residual=True), x + 1)
        assert calls == ["exact"]
    finally:
        set_rmsnorm_mode("native")


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
    monkeypatch.setattr(
        modeling_qwen3_5,
        "fast_zero_centered_batch_invariant_residual_rms_norm",
        fake_residual,
    )

    set_rmsnorm_mode("sglang_fused")
    try:
        norm = modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=False)
        exact_norm = modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=True)
        x = torch.ones(2, 4)

        assert torch.equal(norm(x), x + 1)
        assert calls[-1] == "native"
        assert torch.equal(norm(x, force_sglang_residual=True), x + 3)
        assert calls[-1] == "residual"

        assert torch.equal(exact_norm(x), x + 5)
        assert calls[-1] == "family1"
        assert torch.equal(exact_norm(x, force_sglang_residual=True), x + 3)
        assert calls[-1] == "residual"

        # Exact and ordinary models may coexist in one process. Running the
        # exact module must not alter the ordinary module's dispatch.
        assert torch.equal(norm(x), x + 1)
        assert calls[-1] == "native"
    finally:
        set_rmsnorm_mode("native")


def test_qwen3_5_v2_candidate_is_explicit_and_fail_loud(monkeypatch):
    calls = []

    def fake_v2(hidden, _weight, _eps, *, residual=None):
        calls.append(residual is not None)
        if residual is None:
            return hidden + 7
        residual_out = hidden + residual
        return residual_out + 7, residual_out

    monkeypatch.setattr(modeling_qwen3_5, "fast_zero_centered_families_v2_rms_norm", fake_v2)
    x = torch.ones(2, 4)
    residual = torch.full_like(x, 3)

    set_rmsnorm_mode("sglang_fused")
    try:
        v1 = modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=True)
        v2 = modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=True, rmsnorm_family="v2")
        assert v1.rmsnorm_family == "v1"
        assert torch.equal(v2(x), x + 7)
        out, residual_out = v2(x, residual=residual, prenorm=True)
        assert torch.equal(residual_out, x + residual)
        assert torch.equal(out, x + residual + 7)
        assert calls == [False, True]
    finally:
        set_rmsnorm_mode("native")

    with pytest.raises(RuntimeError, match="only in the exact training lane"):
        modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=False, rmsnorm_family="v2")

    set_rmsnorm_mode("native")
    try:
        rejected = modeling_qwen3_5.Qwen3_5RMSNorm(4, exact_contract=True, rmsnorm_family="v2")
        with pytest.raises(RuntimeError, match="requires rmsnorm_mode='sglang_fused'"):
            rejected(x)
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


def test_qwen3_5_v2_resolves_every_zero_centered_norm_site():
    config = _tiny_config()
    config._qwen35_exact_contract = True
    config._qwen35_rmsnorm_family = "v2"
    set_rmsnorm_mode("sglang_fused")
    try:
        model = Qwen3_5TextModel(config)
    finally:
        set_rmsnorm_mode("native")

    resolved = {
        name: module.rmsnorm_family
        for name, module in model.named_modules()
        if isinstance(module, modeling_qwen3_5.Qwen3_5RMSNorm)
    }
    assert resolved
    assert set(resolved.values()) == {"v2"}
    assert "norm" in resolved
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        assert resolved[f"{prefix}.input_layernorm"] == "v2"
        assert resolved[f"{prefix}.post_attention_layernorm"] == "v2"
        assert resolved[f"{prefix}.self_attn.q_norm"] == "v2"
        assert resolved[f"{prefix}.self_attn.k_norm"] == "v2"


def test_qwen3_5_gdn_gated_norm_remains_a_separate_exact_surface():
    config = _tiny_config(layer_types=["linear_attention", "full_attention"])
    config._qwen35_exact_contract = True
    config._qwen35_rmsnorm_family = "v2"
    set_rmsnorm_mode("sglang_fused")
    try:
        layer = Qwen3_5DecoderLayer(config, layer_idx=0)
    finally:
        set_rmsnorm_mode("native")

    assert layer.linear_attn is not None
    assert type(layer.linear_attn.o_norm).__name__ == "FusedRMSNormGated"
    assert not hasattr(layer.linear_attn.o_norm, "rmsnorm_family")


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
