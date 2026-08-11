from types import SimpleNamespace

import pytest
import torch

from xorl.models.transformers.deepseek_v4.checkpoint_handler import (
    _fuse_native_mxfp4_bank,
    _native_dense_payload_results,
)
from xorl.models.transformers.deepseek_v4.native_payload import (
    Dsv4NativeBlockFp8Linear,
    Dsv4NativeBlockFp8Payload,
    Dsv4NativeMxfp4ExpertPayload,
    _dequantize_native_block_fp8,
    _validate_single_adapter_batch_info,
    attach_dsv4_native_payloads,
)


def test_dense_native_payload_round_trips_exact_e4m3_and_e8m0_bytes():
    weight = torch.arange(256 * 128, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).reshape(256, 128)
    scale = torch.tensor([[1], [2]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    payload = Dsv4NativeBlockFp8Payload(256, 128)
    payload.load_prequantized(weight, scale)

    assert torch.equal(payload.fp8_weight().view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(payload.e8m0_scale().view(torch.uint8), scale.view(torch.uint8))
    materialized_weight, materialized_scale = payload()
    assert torch.equal(materialized_weight.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(materialized_scale.view(torch.uint8), scale.view(torch.uint8))
    assert materialized_weight.data_ptr() != payload.fp8_weight().data_ptr()
    assert materialized_scale.data_ptr() != payload.e8m0_scale().data_ptr()


def test_dense_dequantization_enters_payload_forward_boundary():
    payload = Dsv4NativeBlockFp8Payload(128, 128)
    weight = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)
    payload.load_prequantized(weight, scale)
    calls = []
    payload.register_forward_pre_hook(lambda *_args: calls.append("pre_forward"))

    dequantized = _dequantize_native_block_fp8(payload)

    assert calls == ["pre_forward"]
    assert dequantized.dtype is torch.bfloat16
    assert tuple(dequantized.shape) == (128, 128)
    payload.to(torch.bfloat16)
    assert payload.packed_weight_f32.dtype is torch.float32
    assert payload.packed_scale_f32.dtype is torch.float32
    assert torch.equal(payload.fp8_weight().view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(payload.e8m0_scale().view(torch.uint8), scale.view(torch.uint8))


def test_native_payload_attachment_preserves_official_bf16_linear_families():
    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(8, 8, bias=False)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            self.model.compressor = torch.nn.Module()
            self.model.compressor.wkv = torch.nn.Linear(8, 8, bias=False)
            self.model.indexer = torch.nn.Module()
            self.model.indexer.linear_weights_proj = torch.nn.Linear(8, 8, bias=False)
            self.model.mlp = torch.nn.Module()
            self.model.mlp.gate = torch.nn.Linear(8, 8, bias=False)
            self.model.scaled_projection = torch.nn.Linear(8, 8, bias=False)

    model = Holder()
    original_weights = {
        name: module.weight for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    }
    attach_dsv4_native_payloads(model, SimpleNamespace(num_hidden_layers=0))

    for name in (
        "lm_head",
        "model.compressor.wkv",
        "model.indexer.linear_weights_proj",
        "model.mlp.gate",
    ):
        module = model.get_submodule(name)
        assert module.weight is original_weights[name]
        assert not hasattr(module, "native_base_payload")
    assert isinstance(model.model.scaled_projection, Dsv4NativeBlockFp8Linear)
    assert hasattr(model.model.scaled_projection, "native_base_payload")


def test_dense_handler_payload_names_and_padded_scale_bytes_are_deterministic():
    weight = torch.zeros(129, 129, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, 2, dtype=torch.float8_e8m0fnu)
    results = _native_dense_payload_results("model.layers.0.self_attn.wq_a.weight", weight, scale)
    assert [name for name, _ in results] == [
        "model.layers.0.self_attn.wq_a.native_base_payload.packed_weight_f32",
        "model.layers.0.self_attn.wq_a.native_base_payload.packed_scale_f32",
    ]
    assert results[0][1].dtype is torch.float32
    assert results[1][1].dtype is torch.float32


def test_native_mxfp4_fusion_matches_marlin_gate_then_up_layout():
    num_experts, hidden, intermediate = 2, 64, 32
    by_projection = {name: {} for name in ("w1", "w2", "w3")}
    for expert in range(num_experts):
        for slot, marker, shape, scale_shape in (
            ("w1", 10 + expert, (intermediate, hidden // 2), (intermediate, hidden // 32)),
            ("w2", 20 + expert, (hidden, intermediate // 2), (hidden, intermediate // 32)),
            ("w3", 30 + expert, (intermediate, hidden // 2), (intermediate, hidden // 32)),
        ):
            weight = torch.full(shape, marker, dtype=torch.int8)
            scale = torch.full(scale_shape, marker, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            by_projection[slot][expert] = (weight, scale)

    w13, w2, s13, s2 = _fuse_native_mxfp4_bank(by_projection, num_experts)
    assert tuple(w13.shape) == (2, 64, 32)
    assert tuple(w2.shape) == (2, 64, 16)
    assert torch.all(w13[0, :intermediate] == 10)
    assert torch.all(w13[0, intermediate:] == 30)
    assert torch.all(w2[1] == 21)
    assert torch.all(s13[1, :intermediate].view(torch.uint8) == 11)
    assert torch.all(s13[1, intermediate:].view(torch.uint8) == 31)
    assert torch.all(s2[0].view(torch.uint8) == 20)


def test_native_mxfp4_payload_scale_dtype_survives_model_cast():
    payload = Dsv4NativeMxfp4ExpertPayload(2, 64, 32)
    payload.to(torch.bfloat16)
    assert all(parameter.dtype is torch.float32 for parameter in payload.parameters())
    assert payload.w13_weight.dtype is torch.int8
    assert payload.w2_weight.dtype is torch.int8
    assert payload.w13_weight_scale_inv.dtype is torch.float8_e8m0fnu
    assert payload.w2_weight_scale_inv.dtype is torch.float8_e8m0fnu


def test_native_mxfp4_payload_round_trips_padded_expert_rows():
    from xorl.models.transformers.deepseek_v4.native_payload import (
        pack_expert_rows_as_float32,
    )

    payload = Dsv4NativeMxfp4ExpertPayload(2, 64, 32)
    originals = (
        torch.arange(2 * 64 * 32, dtype=torch.int32).to(torch.int8).reshape(2, 64, 32),
        torch.arange(2 * 64 * 16, dtype=torch.int32).to(torch.int8).reshape(2, 64, 16),
        torch.arange(2 * 64 * 2, dtype=torch.int32).to(torch.uint8).view(torch.float8_e8m0fnu).reshape(2, 64, 2),
        torch.arange(2 * 64, dtype=torch.int32).to(torch.uint8).view(torch.float8_e8m0fnu).reshape(2, 64, 1),
    )
    destinations = (
        payload.packed_w13_weight_f32,
        payload.packed_w2_weight_f32,
        payload.packed_w13_scale_f32,
        payload.packed_w2_scale_f32,
    )
    with torch.no_grad():
        for destination, original in zip(destinations, originals, strict=True):
            destination.copy_(pack_expert_rows_as_float32(original))

    recovered = (
        payload.w13_weight,
        payload.w2_weight,
        payload.w13_weight_scale_inv,
        payload.w2_weight_scale_inv,
    )
    for actual, expected in zip(recovered, originals, strict=True):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_native_mxfp4_cache_identity_follows_owned_parameters_not_typed_views():
    payload = Dsv4NativeMxfp4ExpertPayload(2, 64, 32)
    parameters = (
        payload.packed_w13_weight_f32,
        payload.packed_w2_weight_f32,
        payload.packed_w13_scale_f32,
        payload.packed_w2_scale_f32,
    )

    def cache_key():
        return tuple((parameter._version, tuple(parameter.shape), parameter.dtype) for parameter in parameters)

    initial = cache_key()
    first_views = (
        payload.w13_weight,
        payload.w2_weight,
        payload.w13_weight_scale_inv,
        payload.w2_weight_scale_inv,
    )
    second_views = (
        payload.w13_weight,
        payload.w2_weight,
        payload.w13_weight_scale_inv,
        payload.w2_weight_scale_inv,
    )
    assert all(first is not second for first, second in zip(first_views, second_views, strict=True))
    assert cache_key() == initial

    with torch.no_grad():
        payload.packed_w13_weight_f32.copy_(payload.packed_w13_weight_f32)
    assert cache_key() != initial


def test_cached_single_adapter_metadata_corruption_fails_at_named_boundary():
    info = SimpleNamespace(
        seg_indptr=torch.tensor([0, 10], dtype=torch.int32),
        weight_indices=torch.tensor([0], dtype=torch.int32),
        lora_ranks=torch.tensor([1], dtype=torch.int32),
        seg_lens=torch.tensor([11], dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="layers.1.self_attn.wo_a"):
        _validate_single_adapter_batch_info(
            info,
            10,
            where="model.layers.1.self_attn.wo_a",
        )


def test_dense_payload_ownership_survives_lora_replacement():
    from xorl.lora.modules.linear import LoraLinear

    linear = torch.nn.Linear(64, 32, bias=False)
    payload = Dsv4NativeBlockFp8Payload(32, 64)
    linear.add_module("native_base_payload", payload)
    adapted = LoraLinear.from_module(linear, r=1, lora_alpha=1)
    assert adapted.native_base_payload is payload
    assert dict(adapted.named_parameters())["native_base_payload.packed_weight_f32"] is payload.packed_weight_f32


def test_routed_payload_ownership_survives_dsv4_lora_replacement():
    from xorl.lora.expert_adapter_contract import (
        DSV4_CLAMPED_SWIGLU_LORA_PROGRAM,
    )
    from xorl.models.layers.moe import MoEExperts
    from xorl.models.layers.moe.lora import MoEExpertsLoRA

    experts = MoEExperts(
        num_experts=2,
        hidden_dim=64,
        intermediate_size=32,
        moe_implementation="eager",
        swiglu_limit=10,
    )
    experts.expert_lora_semantics = DSV4_CLAMPED_SWIGLU_LORA_PROGRAM
    payload = Dsv4NativeMxfp4ExpertPayload(2, 64, 32)
    experts.add_module("native_mxfp4_payload", payload)
    adapted = MoEExpertsLoRA.from_module(
        experts,
        r=1,
        lora_alpha=1,
    )
    assert adapted.native_mxfp4_payload is payload
    assert adapted.fsdp_requires_full_precision is True
    assert adapted.expert_lora_semantics == DSV4_CLAMPED_SWIGLU_LORA_PROGRAM
