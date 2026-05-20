import pytest
import torch
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config as HFOlmo2Config
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM as HFOlmo2ForCausalLM

from xorl.models.auto import build_foundation_model
from xorl.models.transformers.olmo2.configuration_olmo2 import Olmo2Config as XOlmo2Config
from xorl.models.transformers.olmo2.modeling_olmo2 import Olmo2ForCausalLM
from xorl.models.transformers.olmo2.parallelize import MODEL_TP_PLAN, TP_PLAN
from xorl.models.transformers.olmo2.tp_styles import LocalAxisRMSNormShard


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


def test_olmo2_tp_plan_uses_local_axis_qk_norm():
    # OLMo-2's full-axis q_norm/k_norm doesn't compose with SequenceParallel
    # or stock ColwiseParallel — under colwise q/k_proj the input arrives
    # hidden-sharded and a full-hidden weight can't be applied directly.
    # LocalAxisRMSNormShard shards the 1-D weight on dim 0 so each rank's
    # slice matches its local q/k slice. This is the actual root cause of
    # what was originally reported (the issue thought it was post-norm; the trace
    # was at q_norm in _project_qkv).
    assert isinstance(TP_PLAN["layers.*.self_attn.q_norm"], LocalAxisRMSNormShard)
    assert isinstance(TP_PLAN["layers.*.self_attn.k_norm"], LocalAxisRMSNormShard)

    # Post-norms see a Replicate input after the rowwise all-reduce in
    # o_proj/down_proj — they should NOT be in the plan (no TP wrapping).
    assert "layers.*.post_attention_layernorm" not in TP_PLAN
    assert "layers.*.post_feedforward_layernorm" not in TP_PLAN
    assert "norm" not in TP_PLAN

    # Standard colwise/rowwise everywhere else.
    assert isinstance(TP_PLAN["layers.*.self_attn.q_proj"], ColwiseParallel)
    assert isinstance(TP_PLAN["layers.*.self_attn.o_proj"], RowwiseParallel)
    assert isinstance(TP_PLAN["layers.*.mlp.gate_proj"], ColwiseParallel)
    assert isinstance(TP_PLAN["layers.*.mlp.down_proj"], RowwiseParallel)

    # lm_head: vanilla colwise (Replicate input from the model's final norm,
    # vocab-parallel output for vocab_parallel_cross_entropy).
    assert MODEL_TP_PLAN["lm_head"] == "colwise" or isinstance(MODEL_TP_PLAN["lm_head"], ColwiseParallel)


def test_olmo2_unfuse_for_tp_matches_hf_parameter_layout():
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
    assert model.get_checkpoint_handler() is None


def test_olmo2_checkpoint_handler_exports_hf_compatible_attention_keys():
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


def test_olmo2_checkpoint_handler_loads_hf_weights_into_fused_model():
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
