"""Regression for issue #87: merge_qkv=False must keep the checkpoint handler
in sync with the unfused module structure.

The builder's old per-attention unfuse never set ``model._unfused_for_tp``, so
the checkpoint handler kept merging q/k/v into ``qkv_proj`` keys that the
freshly created (empty) unfused modules could never receive — every attention
weight silently loaded as garbage, which full-weight weight-sync then shipped
to samplers verbatim.
"""

import pytest
import torch

from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM


pytestmark = pytest.mark.cpu


def _tiny_config():
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=64,
        pad_token_id=0,
    )


def _make_model():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(_tiny_config())


def test_fused_model_handler_merges_qkv():
    model = _make_model()
    layer = model.model.layers[0]
    assert hasattr(layer.self_attn, "qkv_proj"), "model should construct fused"
    handler = model.get_checkpoint_handler(weights_path=None)
    q = torch.randn(64, 64)
    k = torch.randn(32, 64)
    v = torch.randn(32, 64)
    out = []
    for name, tensor in (
        ("model.layers.0.self_attn.q_proj.weight", q),
        ("model.layers.0.self_attn.k_proj.weight", k),
        ("model.layers.0.self_attn.v_proj.weight", v),
    ):
        out.extend(handler.on_load_weight(name, tensor))
    assert [name for name, _ in out] == ["model.layers.0.self_attn.qkv_proj.weight"]
    assert torch.equal(out[0][1], torch.cat([q, k, v], dim=0))


def test_model_level_unfuse_disables_handler_merges():
    """After model.unfuse_for_tp(), separate checkpoint keys must pass through
    untouched: the unfused modules are the ONLY place those weights can land."""
    model = _make_model()
    model.unfuse_for_tp()
    layer = model.model.layers[0]
    assert hasattr(layer.self_attn, "q_proj") and not hasattr(layer.self_attn, "qkv_proj")
    assert hasattr(layer.mlp, "gate_proj") and not hasattr(layer.mlp, "gate_up_proj")
    assert getattr(model, "_unfused_for_tp", False) is True

    handler = model.get_checkpoint_handler(weights_path=None)
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ):
        tensor = torch.randn(8, 8)
        result = handler.on_load_weight(name, tensor)
        assert result == [(name, tensor)], f"{name} must pass through unmerged"


def test_builder_unfuse_block_sets_handler_flag():
    """The exact code path build_training_model runs for merge_qkv=False:
    the model-level unfuse must be preferred so the flag reaches the handler.

    (The old buggy path — looping layer.self_attn.unfuse_for_tp() — left
    _unfused_for_tp unset, and the handler kept merging.)"""
    model = _make_model()
    # Mirror the fixed builder block.
    if hasattr(model, "unfuse_for_tp"):
        model.unfuse_for_tp()
    else:  # pragma: no cover - qwen3 implements it
        for layer in model.model.layers:
            layer.self_attn.unfuse_for_tp()
        model._unfused_for_tp = True

    handler = model.get_checkpoint_handler(weights_path=None)
    name = "model.layers.1.self_attn.q_proj.weight"
    tensor = torch.randn(8, 8)
    assert handler.on_load_weight(name, tensor) == [(name, tensor)]
