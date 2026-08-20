import torch

from xorl.models.layers.attention.backend.flash_attention import _flash_attention_causal_flag
from xorl.models.layers.moe import MoEBlock
from xorl.models.layers.moe.routing_replay import RoutingReplay, set_replay_stage
from xorl.models.transformers.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from xorl.models.transformers.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def _tiny_qwen3_moe_config() -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        _moe_implementation="eager",
        hidden_act="gelu_pytorch_tanh",
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def test_qwen3_moe_decode_cache_matches_full_forward_with_r3_replay():
    _assert_flash_attention_decode_cache_uses_prefix_attention()

    torch.manual_seed(0)
    set_replay_stage(None)
    RoutingReplay.clear_all()
    RoutingReplay._instances.clear()
    natural_model = Qwen3MoeForCausalLM(_tiny_qwen3_moe_config()).eval()
    natural_input_ids = torch.tensor([[1, 7, 13, 21, 34, 55]])
    natural_attention_mask = torch.ones_like(natural_input_ids)
    natural_prefill_len = 3
    with torch.no_grad():
        natural_full = natural_model(
            input_ids=natural_input_ids,
            attention_mask=natural_attention_mask,
        ).last_hidden_state
        natural_past_key_values = []
        natural_outputs = natural_model(
            input_ids=natural_input_ids[:, :natural_prefill_len],
            attention_mask=natural_attention_mask[:, :natural_prefill_len],
            past_key_values=natural_past_key_values,
        )
        assert natural_outputs.past_key_values is not None
        natural_past_key_values = list(natural_outputs.past_key_values)
        natural_decoded = []
        for position in range(natural_prefill_len, natural_input_ids.shape[1]):
            natural_outputs = natural_model(
                input_ids=natural_input_ids[:, position : position + 1],
                attention_mask=natural_attention_mask[:, : position + 1],
                position_ids=torch.tensor([[position]]),
                past_key_values=natural_past_key_values,
            )
            assert natural_outputs.past_key_values is not None
            natural_past_key_values = list(natural_outputs.past_key_values)
            natural_decoded.append(natural_outputs.last_hidden_state)
    natural_decoded_hidden = torch.cat(natural_decoded, dim=1)
    torch.testing.assert_close(
        natural_decoded_hidden,
        natural_full[:, natural_prefill_len:],
        atol=1e-5,
        rtol=1e-5,
    )

    torch.manual_seed(0)
    set_replay_stage(None)
    RoutingReplay.clear_all()
    RoutingReplay._instances.clear()
    original_target_device = RoutingReplay._target_device
    RoutingReplay._target_device = lambda self: torch.device("cpu")
    model = Qwen3MoeForCausalLM(_tiny_qwen3_moe_config()).eval()
    model.enable_routing_replay()
    input_ids = torch.tensor([[1, 7, 13, 21, 34, 55]])
    attention_mask = torch.ones_like(input_ids)
    prefill_len = 3
    labels = torch.tensor([[-100, -100, 21, 34, 55, 8]])
    segments = [(0, prefill_len), *[(pos, pos + 1) for pos in range(prefill_len, input_ids.shape[1])]]

    moe_blocks = [module.mlp for module in model.modules() if isinstance(getattr(module, "mlp", None), MoEBlock)]

    try:
        with torch.no_grad():
            set_replay_stage("record")
            model(input_ids=input_ids, attention_mask=attention_mask)
            set_replay_stage(None)

            recorded_indices = [block._routing_replay.top_indices_list[0].clone() for block in moe_blocks]
            recorded_weights = [block._routing_replay.top_weights_list[0].clone() for block in moe_blocks]

            RoutingReplay.clear_all()
            for block, indices, weights in zip(moe_blocks, recorded_indices, recorded_weights, strict=True):
                block._routing_replay.top_indices_list.append(indices)
                block._routing_replay.top_weights_list.append(weights)
            set_replay_stage("replay_forward")
            full = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            set_replay_stage(None)

            RoutingReplay.clear_all()
            for block, indices, weights in zip(moe_blocks, recorded_indices, recorded_weights, strict=True):
                for start, end in segments:
                    block._routing_replay.top_indices_list.append(indices[start:end])
                    block._routing_replay.top_weights_list.append(weights[start:end])

            past_key_values = []
            decoded = []
            set_replay_stage("replay_forward")
            for start, end in segments:
                outputs = model(
                    input_ids=input_ids[:, start:end],
                    attention_mask=attention_mask[:, :end],
                    position_ids=torch.arange(start, end).unsqueeze(0),
                    past_key_values=past_key_values,
                    diagnostic_decode_cache=True,
                )
                assert outputs.past_key_values is not None
                past_key_values = list(outputs.past_key_values)
                decoded.append(outputs.last_hidden_state)

        cached_hidden = torch.cat(decoded, dim=1)
        valid = labels != -100
        torch.testing.assert_close(cached_hidden[valid], full[valid], atol=1e-5, rtol=1e-5)
    finally:
        set_replay_stage(None)
        RoutingReplay.clear_all()
        RoutingReplay._target_device = original_target_device


def _assert_flash_attention_decode_cache_uses_prefix_attention():
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 5, 2, 4)

    assert (
        _flash_attention_causal_flag(
            module_causal=True,
            query=query,
            key=key,
            diagnostic_decode_cache=True,
        )
        is False
    )
    assert (
        _flash_attention_causal_flag(
            module_causal=True,
            query=query,
            key=key,
            diagnostic_decode_cache=False,
        )
        is True
    )
