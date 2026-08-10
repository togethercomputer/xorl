from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.exact_shared_expert_qlora import Glm52ExactTP16SharedExpertBlockFP8QLoRA
from xorl.models.transformers.glm5.modeling_glm5 import Glm5Attention, Glm5ForCausalLM
from xorl.models.transformers.glm5.native_fp8 import (
    Glm52NativeBlockFP8Experts,
    NativeBlockFP8ExpertPairBuffer,
    NativeBlockFP8PairBuffer,
    native_fp8_dense_source_map,
    validate_glm52_native_fp8_config,
)
from xorl.ops.block_fp8_native import NativeBlockFP8Linear, unpack_float32_as_fp8


OFFICIAL_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": ["model.embed_tokens", "lm_head", "model.layers.78.shared_head.norm"],
}


def _fp8_values(shape):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int16)
    return (((values % 31) - 15).to(torch.float32).reshape(shape)).to(torch.float8_e4m3fn)


def test_hf_config_preserves_quantization_metadata_and_roundtrips():
    hf_config = SimpleNamespace(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=78,
        num_attention_heads=8,
        quantization_config=OFFICIAL_QUANT_CONFIG,
    )

    config = Glm5Config.from_hf_config(hf_config)
    restored = Glm5Config.from_dict(config.to_dict())

    assert config.quantization_config == OFFICIAL_QUANT_CONFIG
    assert restored.quantization_config == OFFICIAL_QUANT_CONFIG
    assert validate_glm52_native_fp8_config(restored.quantization_config)["weight_block_size"] == [128, 128]


@pytest.mark.parametrize(
    "field,value",
    [
        ("quant_method", "int8"),
        ("fmt", "e5m2"),
        ("activation_scheme", "static"),
        ("weight_block_size", [64, 128]),
    ],
)
def test_native_config_rejects_nonofficial_contract(field, value):
    config = dict(OFFICIAL_QUANT_CONFIG)
    config[field] = value
    with pytest.raises(ValueError, match="Unsupported"):
        validate_glm52_native_fp8_config(config)


class _TinyNativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = NativeBlockFP8Linear(256, 384)


def test_pair_buffer_emits_exact_dcp_visible_parameter_names_and_bytes():
    model = _TinyNativeModel()
    buffer = NativeBlockFP8PairBuffer(model, {"official.proj": "proj"})
    weight = _fp8_values((384, 256))
    scale = torch.arange(6, dtype=torch.float32).reshape(3, 2) / 17

    assert buffer.try_consume("official.proj.weight_scale_inv", scale) == []
    pairs = buffer.try_consume("official.proj.weight", weight)
    buffer.validate_complete()

    assert pairs is not None
    assert [name for name, _ in pairs] == ["proj.packed_weight_f32", "proj.weight_scale_inv"]
    assert torch.equal(pairs[0][1].view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(pairs[1][1].view(torch.uint8), scale.view(torch.uint8))
    assert set(model.state_dict()) == {"proj.packed_weight_f32", "proj.weight_scale_inv"}


def test_pair_buffer_fails_closed_on_missing_duplicate_and_bad_scale():
    model = _TinyNativeModel()
    weight = _fp8_values((384, 256))
    scale = torch.ones(3, 2, dtype=torch.float32)

    missing = NativeBlockFP8PairBuffer(model, {"official.proj": "proj"})
    assert missing.try_consume("official.proj.weight", weight) == []
    with pytest.raises(ValueError, match="Incomplete"):
        missing.validate_complete()

    duplicate = NativeBlockFP8PairBuffer(model, {"official.proj": "proj"})
    assert duplicate.try_consume("official.proj.weight", weight) == []
    with pytest.raises(ValueError, match="Duplicate"):
        duplicate.try_consume("official.proj.weight", weight)

    bad_scale = NativeBlockFP8PairBuffer(model, {"official.proj": "proj"})
    assert bad_scale.try_consume("official.proj.weight", weight) == []
    with pytest.raises(ValueError, match="must be FP32"):
        bad_scale.try_consume("official.proj.weight_scale_inv", scale.to(torch.bfloat16))


def test_pair_buffer_rejects_noninjective_target_mapping():
    model = _TinyNativeModel()
    with pytest.raises(ValueError, match="duplicate targets"):
        NativeBlockFP8PairBuffer(
            model,
            {"official.a": "proj", "official.b": "proj"},
        )

    model.alias = model.proj
    with pytest.raises(ValueError, match="same module"):
        NativeBlockFP8PairBuffer(
            model,
            {"official.a": "proj", "official.b": "alias"},
        )


def _tiny_native_config():
    exclusions = ["model.embed_tokens", "lm_head"]
    exclusions.extend(f"model.layers.{layer}.self_attn.indexers_proj" for layer in range(2))
    quantization_config = dict(OFFICIAL_QUANT_CONFIG)
    quantization_config["modules_to_not_convert"] = exclusions
    return Glm5Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=64,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=4,
        pad_token_id=0,
        quantization_config=quantization_config,
    )


def test_model_replaces_only_quantized_dense_shared_and_expert_modules():
    model = Glm5ForCausalLM(_tiny_native_config())
    modules = dict(model.named_modules())

    assert isinstance(modules["model.layers.0.self_attn.q_a_proj"], NativeBlockFP8Linear)
    assert isinstance(modules["model.layers.0.mlp.gate_proj"], NativeBlockFP8Linear)
    assert isinstance(modules["model.layers.1.mlp.shared_experts.gate_proj"], NativeBlockFP8Linear)
    assert isinstance(modules["model.layers.1.mlp.experts"], Glm52NativeBlockFP8Experts)
    assert isinstance(modules["model.layers.0.self_attn.indexer.weights_proj"], nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.get_ignore_modules_in_mixed_precision() == (
        NativeBlockFP8Linear,
        Glm52ExactTP1BlockFP8QLoRALinear,
        Glm52ExactTP16SharedExpertBlockFP8QLoRA,
    )


def test_sparse_mla_native_kv_weight_uses_module_forward_and_preserves_layout():
    class MaterializingNativeLinear(NativeBlockFP8Linear):
        def __init__(self):
            super().__init__(4, 16)
            self.expected = torch.arange(64, dtype=torch.bfloat16).reshape(16, 4)

        def forward(self, input=None, *, return_dequantized_weight=False):
            assert input is None
            assert return_dequantized_weight is True
            return self.expected

    projection = MaterializingNativeLinear()
    pre_hook_calls = []
    projection.register_forward_pre_hook(lambda *_: pre_hook_calls.append(True))
    attention = SimpleNamespace(
        kv_b_proj=projection,
        num_heads=2,
        qk_nope_head_dim=4,
        v_head_dim=4,
        kv_lora_rank=4,
    )

    w_kc, w_vc = Glm5Attention._split_kv_b_weight(attention)

    expected = projection.expected.view(2, 8, 4)
    assert pre_hook_calls == [True]
    assert torch.equal(w_kc, expected[:, :4])
    assert torch.equal(w_vc, expected[:, 4:])


def test_canonical_router_uses_serving_dispatch_and_defers_routed_scale(monkeypatch):
    model = Glm5ForCausalLM(_tiny_native_config())
    block = model.model.layers[1].mlp
    hidden = torch.linspace(-1, 1, steps=2 * block.config.hidden_size, dtype=torch.bfloat16).reshape(
        2, block.config.hidden_size
    )
    logits = torch.linspace(-1, 1, steps=2 * block.num_experts, dtype=torch.float32).reshape(2, block.num_experts)
    expected_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32)
    expected_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    calls = []

    def fake_serving_topk(hidden_states, router_logits, correction_bias, **kwargs):
        calls.append((hidden_states, router_logits, correction_bias, kwargs))
        return expected_weights, expected_ids

    monkeypatch.setattr(
        "xorl.models.transformers.glm5.modeling_glm5._glm52_serving_grouped_topk",
        fake_serving_topk,
    )

    block.config.indexer_types = ["full"]
    block.config._glm52_exact_contract = True
    block.canonical_contract_version = "test"
    canonical_weights, canonical_ids = block._route_tokens_to_experts(
        logits,
        torch.bfloat16,
        hidden_states=hidden,
    )

    assert torch.equal(canonical_weights, expected_weights)
    assert torch.equal(canonical_ids, expected_ids)
    assert len(calls) == 1
    assert calls[0][0] is hidden
    assert calls[0][1] is logits
    assert calls[0][2] is block.gate.e_score_correction_bias
    assert calls[0][3] == {
        "top_k": block.top_k,
        "num_expert_group": block.n_group,
        "topk_group": block.topk_group,
        "routed_scaling_factor": 2.5,
    }


class _TinyExpertModel(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = Glm52NativeBlockFP8Experts(num_experts, 256, 128)


def test_glm_expert_state_is_frozen_exact_and_scoring_only():
    module = Glm52NativeBlockFP8Experts(2, 256, 128)
    gate_up = _fp8_values((2, 256, 256))
    gate_up_scale = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) / 11
    down = _fp8_values((2, 128, 256))
    down_scale = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2) / 19
    module.load_prequantized(gate_up, gate_up_scale, down, down_scale)

    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert torch.equal(module.gate_up_proj.view(torch.uint8), gate_up.view(torch.uint8))
    assert torch.equal(module.down_proj.view(torch.uint8), down.view(torch.uint8))
    hidden = torch.zeros(2, 256, dtype=torch.bfloat16, requires_grad=True)
    routing = torch.zeros(2, 1, dtype=torch.bfloat16)
    local_ids = torch.zeros(2, 1, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="scoring-only"):
        module(hidden, routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(TypeError, match="FP32 routing weights"):
        module(hidden.detach(), routing, sglang_ep_native_local_ids=local_ids)
    with pytest.raises(RuntimeError, match="canonical rank-local"):
        module(hidden.detach(), routing)


def test_expert_pair_buffer_fuses_local_gate_up_down_bytes_and_scales():
    model = _TinyExpertModel()
    buffer = NativeBlockFP8ExpertPairBuffer(model, ep_rank=1, ep_size=2, num_experts=4)
    pieces = {}
    for expert in (2, 3):
        for proj, shape, scale_shape in (
            ("gate", (128, 256), (1, 2)),
            ("up", (128, 256), (1, 2)),
            ("down", (256, 128), (2, 1)),
        ):
            weight = _fp8_values(shape)
            scale = torch.full(scale_shape, expert + {"gate": 1, "up": 2, "down": 3}[proj], dtype=torch.float32)
            pieces[(expert, proj)] = (weight, scale)

    emitted = []
    for suffix in ("weight_scale_inv", "weight"):
        for expert in (3, 2):
            for proj in ("down", "up", "gate"):
                emitted.extend(
                    buffer.try_consume(
                        f"model.layers.0.mlp.experts.{expert}.{proj}_proj.{suffix}",
                        pieces[(expert, proj)][1 if suffix == "weight_scale_inv" else 0],
                    )
                    or []
                )
    buffer.validate_complete()
    result = dict(emitted)

    assert set(result) == {
        "model.layers.0.mlp.experts.gate_up_packed_weight_f32",
        "model.layers.0.mlp.experts.gate_up_weight_scale_inv",
        "model.layers.0.mlp.experts.down_packed_weight_f32",
        "model.layers.0.mlp.experts.down_weight_scale_inv",
    }
    expected_gate_up = torch.stack(
        [torch.cat((pieces[(expert, "gate")][0], pieces[(expert, "up")][0]), dim=0).T for expert in (2, 3)]
    ).contiguous()
    restored_gate_up = unpack_float32_as_fp8(
        result[next(k for k in result if k.endswith("gate_up_packed_weight_f32"))], expected_gate_up.shape
    )
    assert torch.equal(restored_gate_up.view(torch.uint8), expected_gate_up.view(torch.uint8))
    assert result[next(k for k in result if k.endswith("gate_up_weight_scale_inv"))].dtype is torch.float32


def test_expert_pair_buffer_rejects_bad_rank_and_target_count():
    with pytest.raises(ValueError, match="Invalid"):
        NativeBlockFP8ExpertPairBuffer(_TinyExpertModel(), ep_rank=2, ep_size=2, num_experts=4)
    with pytest.raises(ValueError, match="declares 2 experts"):
        NativeBlockFP8ExpertPairBuffer(_TinyExpertModel(num_experts=2), ep_rank=0, ep_size=2, num_experts=4)


def test_grouped_dense_and_expert_handlers_own_disjoint_native_pair_families():
    model = Glm5ForCausalLM(_tiny_native_config())
    dense_handler = model.get_checkpoint_handler(
        ep_rank=0,
        ep_size=1,
        load_family="dense",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    expert_handler = model.get_checkpoint_handler(
        ep_rank=0,
        ep_size=2,
        load_family="expert",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert dense_handler._native_weight_buffer is not None
    assert dense_handler._native_expert_buffer is None
    assert dense_handler._expert_buffer is None
    assert expert_handler._native_weight_buffer is None
    assert expert_handler._native_expert_buffer is not None

    dense_results = []
    modules = dict(model.named_modules())
    for source, target in native_fp8_dense_source_map(model).items():
        module = modules[target]
        dense_results.extend(
            dense_handler.on_load_weight(
                f"{source}.weight_scale_inv",
                torch.ones_like(module.weight_scale_inv, dtype=torch.float32),
            )
        )
        dense_results.extend(
            dense_handler.on_load_weight(
                f"{source}.weight",
                torch.zeros(module.out_features, module.in_features, dtype=torch.float8_e4m3fn),
            )
        )

    expert_results = []
    for expert in (0, 1):
        for proj, weight_shape, scale_shape in (
            ("gate", (128, 256), (1, 2)),
            ("up", (128, 256), (1, 2)),
            ("down", (256, 128), (2, 1)),
        ):
            prefix = f"model.layers.1.mlp.experts.{expert}.{proj}_proj"
            expert_results.extend(
                expert_handler.on_load_weight(
                    f"{prefix}.weight_scale_inv",
                    torch.ones(scale_shape, dtype=torch.float32),
                )
            )
            expert_results.extend(
                expert_handler.on_load_weight(
                    f"{prefix}.weight",
                    torch.zeros(weight_shape, dtype=torch.float8_e4m3fn),
                )
            )

    dense_handler.on_load_complete()
    expert_handler.on_load_complete()
    assert len(dense_results) == 2 * len(native_fp8_dense_source_map(model))
    assert {name for name, _ in expert_results} == {
        "model.layers.1.mlp.experts.gate_up_packed_weight_f32",
        "model.layers.1.mlp.experts.gate_up_weight_scale_inv",
        "model.layers.1.mlp.experts.down_packed_weight_f32",
        "model.layers.1.mlp.experts.down_weight_scale_inv",
    }
    expert_skip = expert_handler.get_skip_key_fn()
    assert expert_skip is not None
    assert expert_skip("model.layers.1.mlp.experts.2.gate_proj.weight")
    assert not expert_skip("model.layers.1.mlp.experts.1.gate_proj.weight")
