import pytest
import torch
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config as HFOlmo2Config
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM as HFOlmo2ForCausalLM

from xorl.models.auto import build_foundation_model
from xorl.models.transformers.olmo2.configuration_olmo2 import Olmo2Config as XOlmo2Config
from xorl.models.transformers.olmo2.modeling_olmo2 import Olmo2ForCausalLM


pytestmark = [pytest.mark.cpu]


_COMMON_KWARGS = dict(
    architectures=["Olmo2ForCausalLM"],
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=32,
    rope_theta=500000.0,
    # OLMo-2 is post-norm: RMSNorm is applied to attn/MLP output. With the
    # default 0.02 init the activations are tiny and RMSNorm amplifies fp32
    # rounding so much that HF and xorl drift apart numerically. A larger
    # init keeps the comparison in the regime where rms ≫ eps.
    initializer_range=0.5,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    use_cache=False,
)


def _make_hf_olmo2_config():
    return HFOlmo2Config(**_COMMON_KWARGS)


def _make_xorl_olmo2_config():
    config = XOlmo2Config(**_COMMON_KWARGS)
    config._attn_implementation = "eager"
    config._activation_native = True
    return config


def test_build_foundation_model_accepts_hf_olmo2_config_object():
    hf_config = _make_hf_olmo2_config()

    model = build_foundation_model(hf_config, init_device="meta", attn_implementation="eager")

    assert isinstance(model, Olmo2ForCausalLM)
    assert model.config.model_type == "olmo2"
    layer = model.model.layers[0]
    # OLMo-2 uses post-norm: no input_layernorm, has post-attn and post-feedforward norms.
    assert not hasattr(layer, "input_layernorm")
    assert hasattr(layer, "post_attention_layernorm")
    assert hasattr(layer, "post_feedforward_layernorm")
    # Full-axis QK norms (head_dim * num_heads), not per-head.
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    assert layer.self_attn.q_norm.weight.shape == (hf_config.num_attention_heads * head_dim,)
    assert layer.self_attn.k_norm.weight.shape == (hf_config.num_key_value_heads * head_dim,)
    assert layer.self_attn.qkv_proj.bias is None
    assert layer.self_attn.o_proj.bias is None

    _assert_olmo2_unfuse_for_tp_matches_hf_parameter_layout()
    _assert_olmo2_checkpoint_handler_bidirectional_policy()


def _assert_olmo2_unfuse_for_tp_matches_hf_parameter_layout():
    model = Olmo2ForCausalLM(_make_xorl_olmo2_config())

    model.unfuse_for_tp()

    layer = model.model.layers[0]
    assert not hasattr(layer.self_attn, "qkv_proj")
    assert hasattr(layer.self_attn, "q_proj")
    assert hasattr(layer.self_attn, "k_proj")
    assert hasattr(layer.self_attn, "v_proj")
    assert layer.self_attn.q_proj.bias is None
    assert layer.self_attn.k_proj.bias is None
    assert layer.self_attn.v_proj.bias is None
    assert layer.self_attn.o_proj.bias is None
    assert not hasattr(layer.mlp, "gate_up_proj")
    assert hasattr(layer.mlp, "gate_proj")
    assert hasattr(layer.mlp, "up_proj")
    # Unfused: the handler is still returned, with both merges disabled. It used to be
    # dropped entirely, which also discarded everything else the handler does.
    handler = model.get_checkpoint_handler()
    assert handler is not None
    assert handler._qkv_buffer is None
    assert handler._gate_up_buffer is None
    # HF's already-split keys therefore pass straight through to matching parameters.
    key = "model.layers.0.self_attn.q_proj.weight"
    passthrough = handler.on_load_weight(key, layer.self_attn.q_proj.weight.detach())
    assert [name for name, _ in passthrough] == [key]


def _assert_olmo2_checkpoint_handler_bidirectional_policy():
    model = Olmo2ForCausalLM(_make_xorl_olmo2_config())
    handler = model.get_checkpoint_handler()

    transformed = {}
    for name, tensor in model.state_dict().items():
        for out_name, out_tensor in handler.on_save_weight(name, tensor):
            transformed[out_name] = out_tensor

    assert "model.layers.0.self_attn.q_proj.weight" in transformed
    assert "model.layers.0.self_attn.k_proj.weight" in transformed
    assert "model.layers.0.self_attn.v_proj.weight" in transformed
    assert "model.layers.0.self_attn.o_proj.weight" in transformed
    assert "model.layers.0.self_attn.q_norm.weight" in transformed
    assert "model.layers.0.self_attn.k_norm.weight" in transformed
    assert "model.layers.0.post_attention_layernorm.weight" in transformed
    assert "model.layers.0.post_feedforward_layernorm.weight" in transformed
    assert "model.layers.0.mlp.gate_proj.weight" in transformed
    assert "model.layers.0.mlp.up_proj.weight" in transformed
    assert "model.layers.0.mlp.down_proj.weight" in transformed
    assert "model.layers.0.self_attn.qkv_proj.weight" not in transformed
    assert "model.layers.0.mlp.gate_up_proj.weight" not in transformed

    _assert_olmo2_checkpoint_handler_loads_hf_weights_into_fused_model()


def _assert_olmo2_checkpoint_handler_loads_hf_weights_into_fused_model():
    hf_config = _make_hf_olmo2_config()
    hf_config._attn_implementation = "eager"
    xorl_config = _make_xorl_olmo2_config()

    hf_model = HFOlmo2ForCausalLM(hf_config)
    xorl_model = Olmo2ForCausalLM(xorl_config)

    handler = xorl_model.get_checkpoint_handler()
    transformed = {}
    for name, tensor in hf_model.state_dict().items():
        for out_name, out_tensor in handler.on_load_weight(name, tensor):
            transformed[out_name] = out_tensor
    for out_name, out_tensor in handler.on_load_complete():
        transformed[out_name] = out_tensor

    assert set(transformed) == set(xorl_model.state_dict())
    assert "model.layers.0.self_attn.qkv_proj.weight" in transformed
    assert "model.layers.0.mlp.gate_up_proj.weight" in transformed
    assert "model.layers.0.self_attn.q_norm.weight" in transformed
    assert "model.layers.0.self_attn.k_norm.weight" in transformed

    load_result = xorl_model.load_state_dict(transformed, strict=False)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    # Avoid pad_token_id (1) so the embedded sequence has real activations
    # at every position; otherwise tiny attn outputs amplify in post-norm.
    input_ids = torch.tensor([[2, 3, 4, 5]])
    hf_model.eval()
    xorl_model.eval()

    with torch.no_grad():
        hf_hidden_states = hf_model.model(input_ids=input_ids).last_hidden_state
        xorl_hidden_states = xorl_model(input_ids=input_ids).last_hidden_state
        hf_logits = hf_model.lm_head(hf_hidden_states)
        xorl_logits = xorl_model.lm_head(xorl_hidden_states)

    torch.testing.assert_close(xorl_hidden_states, hf_hidden_states, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(xorl_logits, hf_logits, atol=2e-4, rtol=5e-4)
