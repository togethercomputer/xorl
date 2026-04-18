import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as HFQwen2ForCausalLM

from xorl.models.auto import build_foundation_model
from xorl.models.transformers.qwen2.configuration_qwen2 import Qwen2Config as XQwen2Config
from xorl.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM


pytestmark = [pytest.mark.cpu]


def _make_hf_qwen2_config():
    return HFQwen2Config(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_sliding_window=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _make_xorl_qwen2_config():
    config = XQwen2Config(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_sliding_window=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    config._activation_native = True
    return config


def test_build_foundation_model_accepts_hf_qwen2_config_object():
    hf_config = _make_hf_qwen2_config()

    model = build_foundation_model(hf_config, init_device="meta", attn_implementation="eager")

    assert isinstance(model, Qwen2ForCausalLM)
    assert model.config.model_type == "qwen2"
    assert not hasattr(model.model.layers[0].self_attn, "q_norm")
    assert not hasattr(model.model.layers[0].self_attn, "k_norm")
    assert model.model.layers[0].self_attn.qkv_proj.bias is not None
    assert model.model.layers[0].self_attn.o_proj.bias is None


def test_qwen2_unfuse_for_tp_matches_hf_parameter_layout():
    model = Qwen2ForCausalLM(_make_xorl_qwen2_config())

    model.unfuse_for_tp()

    layer = model.model.layers[0]
    assert not hasattr(layer.self_attn, "qkv_proj")
    assert hasattr(layer.self_attn, "q_proj")
    assert hasattr(layer.self_attn, "k_proj")
    assert hasattr(layer.self_attn, "v_proj")
    assert layer.self_attn.q_proj.bias is not None
    assert layer.self_attn.k_proj.bias is not None
    assert layer.self_attn.v_proj.bias is not None
    assert layer.self_attn.o_proj.bias is None
    assert not hasattr(layer.mlp, "gate_up_proj")
    assert hasattr(layer.mlp, "gate_proj")
    assert hasattr(layer.mlp, "up_proj")
    assert model.get_checkpoint_handler() is None


def test_qwen2_checkpoint_handler_exports_hf_compatible_attention_keys():
    model = Qwen2ForCausalLM(_make_xorl_qwen2_config())
    handler = model.get_checkpoint_handler()

    transformed = {}
    for name, tensor in model.state_dict().items():
        for out_name, out_tensor in handler.on_save_weight(name, tensor):
            transformed[out_name] = out_tensor

    assert "model.layers.0.self_attn.q_proj.weight" in transformed
    assert "model.layers.0.self_attn.q_proj.bias" in transformed
    assert "model.layers.0.self_attn.k_proj.weight" in transformed
    assert "model.layers.0.self_attn.v_proj.weight" in transformed
    assert "model.layers.0.self_attn.o_proj.weight" in transformed
    assert "model.layers.0.mlp.gate_proj.weight" in transformed
    assert "model.layers.0.mlp.up_proj.weight" in transformed
    assert "model.layers.0.mlp.down_proj.weight" in transformed
    assert "model.layers.0.self_attn.o_proj.bias" not in transformed
    assert "model.layers.0.self_attn.q_norm.weight" not in transformed
    assert "model.layers.0.self_attn.k_norm.weight" not in transformed
    assert "model.layers.0.self_attn.qkv_proj.weight" not in transformed
    assert "model.layers.0.mlp.gate_up_proj.weight" not in transformed


def test_qwen2_checkpoint_handler_loads_hf_weights_into_fused_model():
    hf_config = _make_hf_qwen2_config()
    hf_config._attn_implementation = "eager"
    xorl_config = _make_xorl_qwen2_config()

    hf_model = HFQwen2ForCausalLM(hf_config)
    xorl_model = Qwen2ForCausalLM(xorl_config)

    handler = xorl_model.get_checkpoint_handler()
    transformed = {}
    for name, tensor in hf_model.state_dict().items():
        for out_name, out_tensor in handler.on_load_weight(name, tensor):
            transformed[out_name] = out_tensor
    for out_name, out_tensor in handler.on_load_complete():
        transformed[out_name] = out_tensor

    assert set(transformed) == set(xorl_model.state_dict())
    assert "model.layers.0.self_attn.qkv_proj.weight" in transformed
    assert "model.layers.0.self_attn.qkv_proj.bias" in transformed
    assert "model.layers.0.mlp.gate_up_proj.weight" in transformed
    assert "model.layers.0.self_attn.q_norm.weight" not in transformed
    assert "model.layers.0.self_attn.k_norm.weight" not in transformed

    load_result = xorl_model.load_state_dict(transformed, strict=False)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    input_ids = torch.tensor([[1, 2, 3, 4]])
    hf_model.eval()
    xorl_model.eval()

    with torch.no_grad():
        hf_hidden_states = hf_model.model(input_ids=input_ids).last_hidden_state
        xorl_hidden_states = xorl_model(input_ids=input_ids).last_hidden_state
        hf_logits = hf_model.lm_head(hf_hidden_states)
        xorl_logits = xorl_model.lm_head(xorl_hidden_states)

    torch.testing.assert_close(xorl_hidden_states, hf_hidden_states, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(xorl_logits, hf_logits, atol=5e-5, rtol=5e-5)
