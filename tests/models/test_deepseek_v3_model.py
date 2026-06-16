import pytest
import torch

from xorl.lora.modules import LoraLinear
from xorl.lora.utils import inject_lora_into_model
from xorl.models.layers.moe import MoEExpertsLoRA
from xorl.models.layers.moe.routing_replay import RoutingReplay, set_replay_stage
from xorl.models.transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from xorl.models.transformers.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from xorl.models.transformers.deepseek_v3.support import freeze_deepseek_v3_router_parameters


pytestmark = [pytest.mark.cpu]


def _tiny_config() -> DeepseekV3Config:
    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_dropout=0.0,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
    )
    config._attn_implementation = "eager"
    config._moe_implementation = "eager"
    config._activation_native = True
    return config


def test_deepseek_v3_tiny_forward_backward_and_freeze_router():
    model = DeepseekV3ForCausalLM(_tiny_config())
    model.train()

    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        output_router_logits=True,
    )

    assert tuple(outputs.last_hidden_state.shape) == (2, 5, model.config.hidden_size)
    assert len(outputs.router_logits) == model.config.num_hidden_layers

    loss = outputs.last_hidden_state.float().sum()
    loss.backward()

    assert model.model.layers[0].self_attn.q_a_proj.weight.grad is not None
    assert model.model.layers[0].mlp.shared_experts.down_proj.weight.grad is not None

    frozen = freeze_deepseek_v3_router_parameters(model)

    assert frozen == model.config.num_hidden_layers
    for name, param in model.named_parameters():
        if ".gate.weight" in name:
            assert param.requires_grad is False


def test_deepseek_v3_default_lora_targets_cover_mla_and_moe():
    model = DeepseekV3ForCausalLM(_tiny_config())

    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=None)

    attn = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp

    assert isinstance(attn.q_a_proj, LoraLinear)
    assert isinstance(attn.q_b_proj, LoraLinear)
    assert isinstance(attn.kv_a_proj_with_mqa, LoraLinear)
    assert isinstance(attn.kv_b_proj, LoraLinear)
    assert isinstance(attn.o_proj, LoraLinear)
    assert isinstance(mlp.shared_experts.gate_proj, LoraLinear)
    assert isinstance(mlp.shared_experts.up_proj, LoraLinear)
    assert isinstance(mlp.shared_experts.down_proj, LoraLinear)
    assert isinstance(mlp.experts, MoEExpertsLoRA)


def test_deepseek_v3_forward_emits_router_logits_when_aux_loss_is_enabled_by_config():
    config = _tiny_config()
    config.output_router_logits = False
    config.router_aux_loss_coef = 0.001

    model = DeepseekV3ForCausalLM(config)
    outputs = model(
        input_ids=torch.randint(0, model.config.vocab_size, (2, 5)),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    assert outputs.router_logits is not None
    assert len(outputs.router_logits) == model.config.num_hidden_layers


def test_deepseek_v3_router_logits_skip_dense_layers_for_aux_loss():
    config = _tiny_config()
    config.first_k_dense_replace = 1
    config.output_router_logits = False
    config.router_aux_loss_coef = 0.001

    model = DeepseekV3ForCausalLM(config)
    outputs = model(
        input_ids=torch.randint(0, model.config.vocab_size, (2, 5)),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    assert outputs.router_logits is not None
    assert len(outputs.router_logits) == model.config.num_hidden_layers - config.first_k_dense_replace
    assert all(router_logits is not None for router_logits in outputs.router_logits)


def test_deepseek_v3_routing_replay_records_weights():
    model = DeepseekV3ForCausalLM(_tiny_config())
    block = model.model.layers[0].mlp
    replay = RoutingReplay()
    block._routing_replay = replay
    hidden_states = torch.randn(2, 3, model.config.hidden_size)

    try:
        set_replay_stage("record")
        block(hidden_states)
    finally:
        set_replay_stage(None)

    assert len(replay.top_indices_list) == 1
    assert len(replay.top_weights_list) == 1
    assert tuple(replay.top_weights_list[0].shape) == (
        hidden_states.shape[0] * hidden_states.shape[1],
        model.config.num_experts_per_tok,
    )
