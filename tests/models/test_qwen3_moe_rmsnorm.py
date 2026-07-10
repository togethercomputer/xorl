import pytest
import torch

from xorl.models.layers.normalization import (
    RMS_NORM_FAMILY_NO_RESIDUAL,
    RMS_NORM_FAMILY_RESIDUAL_TREE,
)
from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    _materialize_moe_tp_shards_with_residual,
    _materialize_o_proj_partial_residual,
)


class CaptureInputNorm(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.family_values = []

    def forward(self, hidden_states, *, force_sglang_residual=False, family=None):
        assert force_sglang_residual is False, "call sites declare family, not force flags"
        self.family_values.append(family)
        return hidden_states


class CaptureDelayedInputNorm(torch.nn.Module):
    mode = "native"

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        hidden_states,
        residual=None,
        prenorm=False,
        *,
        force_sglang_residual=False,
        force_sglang_residual_kernel=False,
        family=None,
    ):
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "residual": None if residual is None else residual.detach().clone(),
                "prenorm": prenorm,
                "force_sglang_residual": force_sglang_residual,
                "force_sglang_residual_kernel": force_sglang_residual_kernel,
                "family": family,
            }
        )
        if residual is None:
            return hidden_states + 10.0
        return hidden_states + 10.0, hidden_states + residual


class IdentityAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class PartialOutputAttention(torch.nn.Module):
    def __init__(self, partials):
        super().__init__()
        self.partials = partials

    def forward(self, hidden_states, **kwargs):
        del hidden_states, kwargs
        output = self.partials[0] + self.partials[1]
        output._xorl_o_proj_tp_partials = self.partials
        return output, None


class IdentityPostAttentionNorm(torch.nn.Module):
    def forward(self, hidden_states, residual=None, prenorm=False, **kwargs):
        return hidden_states, residual


def _small_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_experts=0,
        use_qk_norm=False,
        _attn_implementation="eager",
    )


@pytest.mark.parametrize(
    ("layer_idx", "mode", "expected_family"),
    [
        (0, "sglang", RMS_NORM_FAMILY_NO_RESIDUAL),
        (1, "native", RMS_NORM_FAMILY_RESIDUAL_TREE),
        (1, "sglang", RMS_NORM_FAMILY_RESIDUAL_TREE),
        (0, "sglang_fused", RMS_NORM_FAMILY_NO_RESIDUAL),
        (1, "sglang_fused", RMS_NORM_FAMILY_RESIDUAL_TREE),
    ],
)
def test_qwen3_moe_layer_input_norm_declares_family_by_layer(layer_idx, mode, expected_family):
    """The input-norm call site declares the serving family explicitly: layer-0 is a
    no-residual site, layer>0 is a pre-summed residual-tree site (the 2026-07-04
    norm-seed contract). The declaration is mode-independent; RMSNorm keeps the
    residual-tree dispatch confined to the sglang modes."""
    layer = Qwen3MoeDecoderLayer(_small_config(), layer_idx=layer_idx)
    assert layer.layer_idx == layer_idx

    input_norm = CaptureInputNorm(mode)
    layer.input_layernorm = input_norm
    layer.self_attn = IdentityAttention()
    layer.post_attention_layernorm = IdentityPostAttentionNorm()

    hidden_states = torch.ones(1, 2, 4)
    layer._pre_mlp_forward(hidden_states, position_embeddings=(hidden_states, hidden_states))

    assert input_norm.family_values == [expected_family]


def test_qwen3_moe_model_final_norm_declares_residual_tree_family():
    from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

    config = _small_config()
    config.pad_token_id = 0
    model = Qwen3MoeModel(config)

    assert model.norm.family == RMS_NORM_FAMILY_RESIDUAL_TREE

    class StubLayer(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return (hidden_states,)

    model.layers = torch.nn.ModuleList([StubLayer()])
    final_norm = CaptureInputNorm("sglang_fused")
    model.norm = final_norm

    model(input_ids=torch.tensor([[0, 1]]))

    # The call is bare: the family lives on the module (declared at construction).
    assert final_norm.family_values == [None]


def test_qwen3_moe_layer_consumes_delayed_residual_pair_at_input_norm():
    layer = Qwen3MoeDecoderLayer(_small_config(), layer_idx=1)

    input_norm = CaptureDelayedInputNorm()
    layer.input_layernorm = input_norm
    layer.self_attn = IdentityAttention()
    layer.post_attention_layernorm = IdentityPostAttentionNorm()
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    hidden_delta = torch.full((1, 2, 4), 2.0)
    residual = torch.full((1, 2, 4), 3.0)
    hidden_states, post_attention_residual = layer._pre_mlp_forward(
        (hidden_delta, residual),
        position_embeddings=(hidden_delta, hidden_delta),
    )

    assert len(input_norm.calls) == 1
    call = input_norm.calls[0]
    assert call["prenorm"] is True
    assert call["force_sglang_residual"] is False
    torch.testing.assert_close(call["hidden_states"], hidden_delta)
    torch.testing.assert_close(call["residual"], residual)
    torch.testing.assert_close(hidden_states, hidden_delta + 10.0)
    torch.testing.assert_close(post_attention_residual, hidden_delta + residual)
    torch.testing.assert_close(captures["delayed_pair_delta"], hidden_delta)
    torch.testing.assert_close(captures["delayed_pair_residual"], residual)
    assert "delayed_pair_shard_sum" not in captures
    assert "delayed_pair_shard_materialized" not in captures
    torch.testing.assert_close(captures["materialized_layer_input"], hidden_delta + residual)
    torch.testing.assert_close(captures["input_norm_residual"], hidden_delta + residual)
    torch.testing.assert_close(captures["input_norm"], hidden_delta + 10.0)
    torch.testing.assert_close(captures["post_attention_norm_input"], hidden_delta + 10.0)
    torch.testing.assert_close(captures["post_attention_norm_residual"], hidden_delta + residual)
    torch.testing.assert_close(captures["post_attention_norm"], hidden_delta + 10.0)
    torch.testing.assert_close(captures["post_attention_residual"], hidden_delta + residual)


def test_qwen3_moe_layer_consumes_delayed_tp_shards_at_input_norm(monkeypatch):
    monkeypatch.setenv("XORL_QWEN3_MOE_DELAYED_RESIDUAL_PAIR_TP_SHARD_CARRY", "1")
    layer = Qwen3MoeDecoderLayer(_small_config(), layer_idx=1)

    input_norm = CaptureDelayedInputNorm()
    layer.input_layernorm = input_norm
    layer.self_attn = IdentityAttention()
    layer.post_attention_layernorm = IdentityPostAttentionNorm()
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    hidden_delta = torch.full((1, 2, 4), 99.0)
    residual = torch.full((1, 2, 4), 3.0)
    shard0 = torch.full((1, 2, 4), 2.0)
    shard1 = torch.full((1, 2, 4), 5.0)
    hidden_delta._xorl_sglang_moe_tp_shards = (shard0, shard1)

    hidden_states, post_attention_residual = layer._pre_mlp_forward(
        (hidden_delta, residual),
        position_embeddings=(hidden_delta, hidden_delta),
    )

    expected_materialized = residual + shard0 + shard1
    assert len(input_norm.calls) == 1
    call = input_norm.calls[0]
    assert call["prenorm"] is False
    assert call["residual"] is None
    torch.testing.assert_close(call["hidden_states"], expected_materialized)
    torch.testing.assert_close(hidden_states, expected_materialized + 10.0)
    torch.testing.assert_close(post_attention_residual, expected_materialized)
    torch.testing.assert_close(captures["delayed_pair_delta"], hidden_delta)
    torch.testing.assert_close(captures["delayed_pair_residual"], residual)
    torch.testing.assert_close(captures["delayed_pair_shard_sum"], shard0 + shard1)
    torch.testing.assert_close(captures["delayed_pair_shard_materialized"], expected_materialized)
    torch.testing.assert_close(captures["materialized_layer_input"], expected_materialized)
    torch.testing.assert_close(captures["input_norm_residual"], expected_materialized)
    torch.testing.assert_close(captures["input_norm"], expected_materialized + 10.0)
    torch.testing.assert_close(captures["post_attention_norm_input"], expected_materialized + 10.0)
    torch.testing.assert_close(captures["post_attention_norm_residual"], expected_materialized)
    torch.testing.assert_close(captures["post_attention_residual"], expected_materialized)


def test_qwen3_moe_tp_shard_carry_reduces_shards_before_residual(monkeypatch):
    monkeypatch.setenv("XORL_QWEN3_MOE_DELAYED_RESIDUAL_PAIR_TP_SHARD_CARRY", "1")

    residual = torch.tensor([1.0], dtype=torch.bfloat16)
    hidden_delta = torch.tensor([99.0], dtype=torch.bfloat16)
    shard0 = torch.tensor([0.002], dtype=torch.bfloat16)
    shard1 = torch.tensor([0.002], dtype=torch.bfloat16)
    hidden_delta._xorl_sglang_moe_tp_shards = (shard0, shard1)

    actual = _materialize_moe_tp_shards_with_residual(hidden_delta, residual)
    expected = residual + (shard0 + shard1)
    residual_first = (residual + shard0) + shard1

    torch.testing.assert_close(actual, expected)
    assert actual.item() != residual_first.item()


def test_qwen3_moe_layer_can_consume_o_proj_partials_at_post_attention_boundary(monkeypatch):
    monkeypatch.setenv("XORL_QWEN3_MOE_POST_ATTENTION_O_PROJ_PARTIAL_RESIDUAL", "1")

    layer = Qwen3MoeDecoderLayer(_small_config(), layer_idx=1)

    input_norm = CaptureDelayedInputNorm()
    post_norm = CaptureDelayedInputNorm()
    partial0 = torch.full((1, 2, 4), 2.0)
    partial1 = torch.full((1, 2, 4), 5.0)
    layer.input_layernorm = input_norm
    layer.self_attn = PartialOutputAttention((partial0, partial1))
    layer.post_attention_layernorm = post_norm
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    hidden_states = torch.full((1, 2, 4), 3.0)
    output, post_attention_residual = layer._pre_mlp_forward(
        hidden_states,
        position_embeddings=(hidden_states, hidden_states),
    )

    expected_residual = hidden_states + partial0 + partial1
    assert len(post_norm.calls) == 1
    call = post_norm.calls[0]
    assert call["residual"] is None
    assert call["prenorm"] is False
    torch.testing.assert_close(call["hidden_states"], expected_residual)
    torch.testing.assert_close(output, expected_residual + 10.0)
    torch.testing.assert_close(post_attention_residual, expected_residual)
    torch.testing.assert_close(captures["post_attention_o_proj_partial_sum"], partial0 + partial1)
    torch.testing.assert_close(captures["post_attention_partial_residual"], expected_residual)
    torch.testing.assert_close(captures["post_attention_norm_input"], expected_residual)
    torch.testing.assert_close(captures["post_attention_residual"], expected_residual)


def test_qwen3_moe_can_capture_o_proj_partial_residual_candidates_without_applying(monkeypatch):
    monkeypatch.setenv("XORL_QWEN3_MOE_CAPTURE_O_PROJ_PARTIAL_RESIDUAL_CANDIDATES", "1")
    monkeypatch.setenv("XORL_QWEN3_MOE_CAPTURE_O_PROJ_PARTIAL_RESIDUAL_CANDIDATES_LAYERS", "1")

    layer = Qwen3MoeDecoderLayer(_small_config(), layer_idx=1)

    input_norm = CaptureDelayedInputNorm()
    post_norm = CaptureDelayedInputNorm()
    partial0 = torch.full((1, 2, 4), 0.002, dtype=torch.bfloat16)
    partial1 = torch.full((1, 2, 4), 0.002, dtype=torch.bfloat16)
    layer.input_layernorm = input_norm
    layer.self_attn = PartialOutputAttention((partial0, partial1))
    layer.post_attention_layernorm = post_norm
    captures = {}
    layer._diagnostic_capture_component = lambda name, tensor: captures.setdefault(name, tensor.detach().clone())

    hidden_states = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    output, post_attention_residual = layer._pre_mlp_forward(
        hidden_states,
        position_embeddings=(hidden_states, hidden_states),
    )

    expected_sum_then_residual = hidden_states + (partial0 + partial1)
    expected_residual_then_partials = (hidden_states + partial0) + partial1
    assert "post_attention_o_proj_partial_sum_sum_then_residual" in captures
    assert "post_attention_partial_residual_sum_then_residual" in captures
    assert "post_attention_partial_residual_residual_then_partials" in captures
    assert "post_attention_partial_residual_fp32_sum_then_residual" in captures
    torch.testing.assert_close(
        captures["post_attention_partial_residual_sum_then_residual"], expected_sum_then_residual
    )
    torch.testing.assert_close(
        captures["post_attention_partial_residual_residual_then_partials"],
        expected_residual_then_partials,
    )
    assert expected_sum_then_residual.flatten()[0].item() != expected_residual_then_partials.flatten()[0].item()

    assert len(post_norm.calls) == 1
    torch.testing.assert_close(post_norm.calls[0]["hidden_states"], partial0 + partial1)
    torch.testing.assert_close(output, partial0 + partial1 + 10.0)
    torch.testing.assert_close(post_attention_residual, hidden_states + partial0 + partial1)


def test_qwen3_moe_o_proj_partial_residual_modes(monkeypatch):
    hidden_states = torch.tensor([1.0], dtype=torch.bfloat16)
    residual = torch.tensor([1.0], dtype=torch.bfloat16)
    partial0 = torch.tensor([0.002], dtype=torch.bfloat16)
    partial1 = torch.tensor([0.002], dtype=torch.bfloat16)

    monkeypatch.setenv("XORL_QWEN3_MOE_POST_ATTENTION_O_PROJ_PARTIAL_RESIDUAL_MODE", "sum_then_residual")
    _, sum_then_residual = _materialize_o_proj_partial_residual(hidden_states, residual, (partial0, partial1))
    expected_sum_then_residual = residual + (partial0 + partial1)
    torch.testing.assert_close(sum_then_residual, expected_sum_then_residual)

    monkeypatch.setenv(
        "XORL_QWEN3_MOE_POST_ATTENTION_O_PROJ_PARTIAL_RESIDUAL_MODE",
        "residual_then_partials",
    )
    _, residual_then_partials = _materialize_o_proj_partial_residual(hidden_states, residual, (partial0, partial1))
    expected_residual_then_partials = (residual + partial0) + partial1
    torch.testing.assert_close(residual_then_partials, expected_residual_then_partials)
    assert sum_then_residual.item() != residual_then_partials.item()
