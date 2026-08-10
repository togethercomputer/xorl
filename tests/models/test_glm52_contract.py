from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xorl.distributed.canonical_moe import (
    CanonicalMoEGraphMetadata,
    CanonicalMoETransport,
    ParallelPlan,
    canonical_moe_reduce_reference,
)
from xorl.models.transformers.glm5 import indexer as indexer_module
from xorl.models.transformers.glm5 import sparse_selector as sparse_selector_module
from xorl.models.transformers.glm5.checkpoint_handler import Glm5CheckpointHandler
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.index_share import (
    CanonicalLogicalIndices,
    IndexShareContextManager,
    IndexShareLifecycle,
    IndexShareMode,
)
from xorl.models.transformers.glm5.indexer import (
    Glm5DsaIndexer,
    _fused_bf16_indexer_projection,
    _fused_sampler_index_k_prepare,
    _mix_sampler_index_k_preparation,
    _scale_fused_bf16_indexer_head_gates,
)
from xorl.models.transformers.glm5.layer_plan import Glm52LayerPlan
from xorl.models.transformers.glm5.modeling_glm5 import (
    _GLM52_CANONICAL_TRAINER_TOPOLOGIES,
    Glm5Attention,
    Glm5ForCausalLM,
    Glm5MoEBlock,
    Glm5TopkRouter,
)
from xorl.models.transformers.glm5.sparse_selector import (
    GLM52_SELECTOR_VERSION,
    gather_selected_logical_values,
    physical_cache_to_logical_indices,
    quantize_e4m3_dynamic,
    quantize_e4m3_ue8m0,
    quantize_sparse_key_cache,
    quantize_sparse_query,
    rotate_sparse_selector_activation,
    select_glm52_logical_indices,
)


GLM52_FULL_INDEX_LAYERS = (0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74)


def _map_tensors(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_tensors(fn, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(fn, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(fn, item) for item in value)
    return value


@pytest.mark.cpu
def test_canonical_trainer_admits_only_certified_world16_ep16_cp16():
    assert _GLM52_CANONICAL_TRAINER_TOPOLOGIES == ((16, 1, 1, 1),)
    plan = ParallelPlan.glm52_trainer()
    assert (plan.world_size, plan.pp_size, plan.tp_size, plan.dp_size, plan.ep_size, plan.cp_size) == (
        16,
        1,
        1,
        1,
        16,
        16,
    )
    with pytest.raises(ValueError, match="Unsupported GLM-5.2 trainer topology"):
        ParallelPlan.glm52_trainer(world_size=32, dp_size=2, contributor_count=16)


def _fake_fp8_mqa_logits(q, kv, weights, starts, ends, *, clean_logits):
    assert clean_logits is False
    key, key_scale = kv
    dequantized_key = key.float() * key_scale.float().unsqueeze(-1)
    per_head = torch.einsum("mhd,nd->mhn", q.float(), dequantized_key).relu()
    scores = torch.einsum("mhn,mh->mn", per_head, weights.float())
    columns = torch.arange(key.shape[0], device=q.device)
    legal = (columns.unsqueeze(0) >= starts.unsqueeze(1)) & (columns.unsqueeze(0) < ends.unsqueeze(1))
    return scores.masked_fill(~legal, float("nan"))


def _reference_select(scores, lengths, topk, *, row_starts=None, validate=True):
    del row_starts, validate
    ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :topk]
    valid_counts = lengths.clamp_max(topk)
    valid = torch.arange(topk, device=scores.device).unsqueeze(0) < valid_counts.unsqueeze(1)
    ranked = ranked.to(torch.int32).masked_fill(~valid, -1)
    sentinel = torch.iinfo(torch.int32).max
    ranked = torch.sort(ranked.masked_fill(~valid, sentinel), dim=-1).values
    return ranked.masked_fill(~valid, -1)


def _official_schedule_config() -> SimpleNamespace:
    full = set(GLM52_FULL_INDEX_LAYERS)
    return SimpleNamespace(
        num_hidden_layers=78,
        indexer_types=["full" if layer in full else "shared" for layer in range(78)],
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
        mlp_layer_types=["dense" if layer < 3 else "sparse" for layer in range(78)],
    )


@pytest.mark.cpu
def test_official_layer_plan_counts_producers_and_38_40_split():
    plan = Glm52LayerPlan.from_config(
        _official_schedule_config(),
        pipeline_layer_ranges=((0, 38), (38, 78)),
    )
    assert plan.full_indexer_layers == GLM52_FULL_INDEX_LAYERS
    assert len(plan.full_indexer_layers) == 21
    assert len(plan.shared_indexer_layers) == 57
    assert plan.dense_layers == (0, 1, 2)
    assert len(plan.sparse_layers) == 75
    assert plan.layers[37].index_producer_layer == 34
    assert plan.layers[38].index_producer_layer == 38
    assert plan.layers[77].index_producer_layer == 74


@pytest.mark.cpu
def test_layer_plan_rejects_malformed_schedules_and_shared_stage_start():
    config = _official_schedule_config()
    config.indexer_types = config.indexer_types[:-1]
    with pytest.raises(ValueError, match="indexer_types has length"):
        Glm52LayerPlan.from_config(config)

    config = _official_schedule_config()
    config.indexer_types[38] = "shared"
    config.index_topk_pattern = [int(value == "full") for value in config.indexer_types]
    with pytest.raises(ValueError, match="starts with shared-index"):
        Glm52LayerPlan.from_config(config, pipeline_layer_ranges=((0, 38), (38, 78)))

    config = _official_schedule_config()
    config.mlp_layer_types[12] = "mystery"
    with pytest.raises(ValueError, match="Unknown mlp_layer_types"):
        Glm52LayerPlan.from_config(config)


def _small_plan() -> Glm52LayerPlan:
    config = SimpleNamespace(
        num_hidden_layers=4,
        indexer_types=["full", "shared", "shared", "full"],
        index_topk_freq=3,
        index_skip_topk_offset=0,
        index_topk_pattern=[1, 0, 0, 1],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    )
    return Glm52LayerPlan.from_config(config)


@pytest.mark.cpu
def test_index_share_context_lifecycle_reuse_exception_cleanup_and_concurrency_guard():
    plan = _small_plan()
    manager = IndexShareContextManager(plan, (0, 4))
    payload = CanonicalLogicalIndices(torch.tensor([[[0, 1, -1]]], dtype=torch.int32))

    first = manager.begin(mode=IndexShareMode.TRAINING_WITH_BACKWARD)
    published = first.get_or_publish(producer_layer_index=0, layer_plan=plan, produce_payload=lambda: payload)
    assert torch.equal(published.values, payload.values)
    assert first.require(producer_layer_index=0, layer_plan=plan) is published
    with pytest.raises(RuntimeError, match="one live"):
        manager.begin(mode=IndexShareMode.FORWARD_ONLY)
    manager.end(first)
    manager.end(first)
    assert first.lifecycle is IndexShareLifecycle.CLOSED
    assert manager.active is None

    with pytest.raises(RuntimeError, match="body failed"):
        with manager.invocation(mode=IndexShareMode.FORWARD_ONLY) as second:
            second.get_or_publish(producer_layer_index=0, layer_plan=plan, produce_payload=lambda: payload)
            raise RuntimeError("body failed")
    assert manager.active is None

    with manager.invocation(mode=IndexShareMode.FORWARD_ONLY) as third:
        with pytest.raises(RuntimeError, match="has not published"):
            third.require(producer_layer_index=0, layer_plan=plan)


@pytest.mark.cpu
def test_index_share_context_identity_survives_fsdp_tensor_transform():
    plan = _small_plan()
    manager = IndexShareContextManager(plan, (0, 4))
    payload = CanonicalLogicalIndices(torch.tensor([[[0, 1, -1]]], dtype=torch.int32))

    with manager.invocation(mode=IndexShareMode.FORWARD_ONLY) as context:
        producer_context = _map_tensors(lambda tensor: tensor, {"context": context})["context"]
        assert producer_context is context
        producer_context.get_or_publish(producer_layer_index=0, layer_plan=plan, produce_payload=lambda: payload)

        consumer_context = _map_tensors(lambda tensor: tensor, {"context": context})["context"]
        assert consumer_context is context
        assert consumer_context.require(producer_layer_index=0, layer_plan=plan) is producer_context.get_or_publish(
            producer_layer_index=0,
            layer_plan=plan,
            produce_payload=lambda: pytest.fail("retained producer payload was recomputed"),
        )


@pytest.mark.cpu
@torch.no_grad()
def test_index_share_survives_fsdp_cast_across_dense_producer_and_shared_consumer(monkeypatch):
    monkeypatch.setattr(
        indexer_module,
        "bi_bf16_fp32_linear",
        lambda hidden, weight: F.linear(hidden.float(), weight.float()),
    )
    config = _small_glm_config()
    config.indexer_types = ["full", "full", "full", "shared"]
    config.index_topk_freq = 4
    config.index_skip_topk_offset = 3
    config.index_topk_pattern = [1, 1, 1, 0]
    config.mlp_layer_types = ["dense", "dense", "dense", "sparse"]
    config._sparse_mla_enabled = True
    config._sparse_mla_backend = "auto"
    config._activation_native = True
    # The production Class-B rotary path is CUDA-only and preserves BF16.
    # Cast after the CPU reference fallback so this identity-only fixture does
    # not exercise an unsupported mixed-dtype eager attention matmul.
    config._attention_cast_bf16 = True

    model = Glm5ForCausalLM(config).model.to(torch.bfloat16).eval()
    # This CPU-only regression exercises IndexShare identity across FSDP's
    # tensor transform. Native sparse-selector scoring is intentionally CUDA-only.
    for layer in model.layers[:3]:
        layer.self_attn.indexer.selector_version = "legacy_torch_or_tilelang"
    # The regression concerns the attention IndexShare boundary. Avoid running
    # the unrelated EP-only MoE implementation in the first shared layer.
    model.layers[3].mlp = nn.Identity()

    context_ids: list[int] = []

    def emulate_fsdp_mixed_precision_input_cast(_module, args, kwargs):
        def cast(tensor):
            return tensor.to(torch.bfloat16) if torch.is_floating_point(tensor) else tensor

        transformed = _map_tensors(cast, kwargs)
        context_ids.append(id(transformed["index_share_context"]))
        return args, transformed

    handles = [
        layer.register_forward_pre_hook(emulate_fsdp_mixed_precision_input_cast, with_kwargs=True)
        for layer in model.layers
    ]
    try:
        output = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            index_share_mode=IndexShareMode.FORWARD_ONLY,
        ).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (1, 4, config.hidden_size)
    assert len(context_ids) == 4
    assert len(set(context_ids)) == 1
    assert model._index_share_context_managers[(0, 4)].active is None


def _small_glm_config() -> Glm5Config:
    return Glm5Config(
        vocab_size=32,
        pad_token_id=0,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=3,
        indexer_types=["full", "shared", "shared", "full"],
        index_topk_freq=3,
        index_skip_topk_offset=0,
        index_topk_pattern=[1, 0, 0, 1],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    )


@pytest.mark.cpu
def test_only_full_layers_allocate_indexer_parameters_and_strict_state_dict_round_trip():
    config = _small_glm_config()
    plan = Glm52LayerPlan.from_config(config)
    attentions = nn.ModuleList([Glm5Attention(config, layer, layer_plan=plan) for layer in range(4)])
    assert attentions[0].indexer is not None
    assert attentions[1].indexer is None
    assert attentions[2].indexer is None
    assert attentions[3].indexer is not None

    state = attentions.state_dict()
    assert any(key.startswith("0.indexer.") for key in state)
    assert not any(key.startswith("1.indexer.") or key.startswith("2.indexer.") for key in state)
    clone = nn.ModuleList([Glm5Attention(config, layer, layer_plan=plan) for layer in range(4)])
    clone.load_state_dict(state, strict=True)


@pytest.mark.cpu
def test_sparse_selector_ties_short_rows_dead_rows_and_logical_cache_mapping():
    query = torch.zeros((1, 3, 2, 128), dtype=torch.bfloat16)
    key = torch.zeros((1, 5, 128), dtype=torch.bfloat16)
    weights = torch.ones((1, 3, 2), dtype=torch.float32)
    allowed = torch.tensor(
        [
            [
                [True, True, True, True, True],
                [True, False, False, False, False],
                [False, False, False, False, False],
            ]
        ],
        dtype=torch.bool,
    )
    result = select_glm52_logical_indices(
        query,
        key,
        weights,
        allowed,
        topk=3,
        _native_kernel_for_testing=_fake_fp8_mqa_logits,
        _selector_for_testing=_reference_select,
    )
    assert result.selector_version == GLM52_SELECTOR_VERSION
    assert result.logical_indices.values.tolist() == [[[0, 1, 2], [0, -1, -1], [-1, -1, -1]]]
    assert result.valid_counts.tolist() == [[3, 1, 0]]

    values = torch.randn((1, 5, 4), dtype=torch.bfloat16)
    gathered = gather_selected_logical_values(values, result.logical_indices)
    assert torch.count_nonzero(gathered[0, 2]) == 0
    assert bool(torch.all(torch.isfinite(gathered)))

    physical = torch.tensor([[[2, 0, -1]]], dtype=torch.int32)
    page_map = torch.tensor([4, 1, 3], dtype=torch.int32)
    logical = physical_cache_to_logical_indices(physical, page_map)
    assert logical.values.tolist() == [[[3, 4, -1]]]


@pytest.mark.cpu
def test_glm52_sparse_shared_selector_handles_production_boundary_ties_and_dead_tail():
    query = torch.zeros((1, 1, 1, 128), dtype=torch.bfloat16)
    key = torch.zeros((1, 4112, 128), dtype=torch.bfloat16)
    weights = torch.ones((1, 1, 1), dtype=torch.float32)
    allowed = torch.zeros((1, 1, 4112), dtype=torch.bool)
    allowed[..., :4099] = True

    result = select_glm52_logical_indices(
        query,
        key,
        weights,
        allowed,
        topk=2048,
        _native_kernel_for_testing=_fake_fp8_mqa_logits,
        _selector_for_testing=_reference_select,
    )

    assert result.logical_indices.values.tolist() == [[list(range(2048))]]
    assert result.valid_counts.tolist() == [[2048]]


@pytest.mark.cpu
def test_glm52_sparse_hadamard_transport_is_normalized_and_self_inverse():
    source = torch.arange(128, dtype=torch.float32).sub(63.5).to(torch.bfloat16).reshape(1, 1, 128)
    rotated = rotate_sparse_selector_activation(source)
    restored = rotate_sparse_selector_activation(rotated)

    assert rotated.dtype is torch.bfloat16
    assert not torch.equal(rotated.view(torch.uint8), source.view(torch.uint8))
    torch.testing.assert_close(restored.float(), source.float(), atol=1.0, rtol=0.0)


@pytest.mark.cpu
def test_sparse_selector_applies_hadamard_before_fp8_quantization(monkeypatch):
    torch.manual_seed(1)
    query = torch.randn((1, 1, 2, 128), dtype=torch.bfloat16)
    key = torch.randn((1, 128, 128), dtype=torch.bfloat16)
    weights = torch.randn((1, 1, 2), dtype=torch.float32)
    allowed = torch.ones((1, 1, 128), dtype=torch.bool)

    expected_query = rotate_sparse_selector_activation(query)
    expected_key = rotate_sparse_selector_activation(key)
    observed = {}
    quantize_query = sparse_selector_module.quantize_sparse_query
    quantize_key = sparse_selector_module.quantize_sparse_key_cache

    def record_query(tensor, **kwargs):
        observed["query"] = tensor.clone()
        return quantize_query(tensor, **kwargs)

    def record_key(tensor, **kwargs):
        observed["key"] = tensor.clone()
        return quantize_key(tensor, **kwargs)

    monkeypatch.setattr(sparse_selector_module, "quantize_sparse_query", record_query)
    monkeypatch.setattr(sparse_selector_module, "quantize_sparse_key_cache", record_key)

    select_glm52_logical_indices(
        query,
        key,
        weights,
        allowed,
        topk=16,
        _native_kernel_for_testing=_fake_fp8_mqa_logits,
        _selector_for_testing=_reference_select,
    )

    assert torch.equal(observed["query"].view(torch.uint8), expected_query.view(torch.uint8))
    assert torch.equal(observed["key"].view(torch.uint8), expected_key.view(torch.uint8))


@pytest.mark.cpu
def test_sparse_selector_fused_contract_skips_hadamard_before_fp8_quantization(monkeypatch):
    torch.manual_seed(2)
    query = torch.randn((1, 1, 2, 128), dtype=torch.bfloat16)
    key = torch.randn((1, 128, 128), dtype=torch.bfloat16)
    weights = torch.randn((1, 1, 2), dtype=torch.bfloat16)
    allowed = torch.ones((1, 1, 128), dtype=torch.bool)
    observed = {}
    quantize_query = sparse_selector_module.quantize_sparse_query
    quantize_key = sparse_selector_module.quantize_sparse_key_cache

    def record_query(tensor, **kwargs):
        observed["query"] = tensor.clone()
        return quantize_query(tensor, **kwargs)

    def record_key(tensor, **kwargs):
        observed["key"] = tensor.clone()
        return quantize_key(tensor, **kwargs)

    monkeypatch.setattr(sparse_selector_module, "quantize_sparse_query", record_query)
    monkeypatch.setattr(sparse_selector_module, "quantize_sparse_key_cache", record_key)

    select_glm52_logical_indices(
        query,
        key,
        weights,
        allowed,
        topk=16,
        apply_hadamard=False,
        _native_kernel_for_testing=_fake_fp8_mqa_logits,
        _selector_for_testing=_reference_select,
    )

    assert torch.equal(observed["query"].view(torch.uint8), query.view(torch.uint8))
    assert torch.equal(observed["key"].view(torch.uint8), key.view(torch.uint8))


@pytest.mark.cpu
def test_fused_bf16_indexer_projection_matches_sampler_row_order(monkeypatch):
    torch.manual_seed(3)
    hidden = torch.randn((1, 5, 16), dtype=torch.bfloat16)
    wk = torch.randn((128, 16), dtype=torch.bfloat16)
    weights_proj = torch.randn((2, 16), dtype=torch.bfloat16)
    fused_weight = torch.cat((wk, weights_proj), dim=0).contiguous()
    expected = F.linear(hidden, fused_weight)
    calls = []
    linear = F.linear

    def recording_linear(input, weight, bias=None):
        calls.append((tuple(input.shape), weight.detach().clone()))
        return linear(input, weight, bias)

    monkeypatch.setattr(indexer_module.F, "linear", recording_linear)
    index_k, raw_gate = _fused_bf16_indexer_projection(
        hidden,
        wk,
        weights_proj,
        index_head_dim=128,
        index_n_heads=2,
    )

    assert len(calls) == 1
    assert calls[0][0] == (5, 16)
    assert torch.equal(calls[0][1].view(torch.uint8), fused_weight.view(torch.uint8))
    assert torch.equal(index_k.view(torch.uint8), expected[..., :128].view(torch.uint8))
    assert torch.equal(raw_gate.view(torch.uint8), expected[..., 128:].view(torch.uint8))


@pytest.mark.cpu
def test_fused_bf16_indexer_head_gate_scaling_promotes_before_head_scale():
    raw_gate = torch.tensor([-6.4375, 11.0, -7.65625, -8.125], dtype=torch.bfloat16)

    scaled = _scale_fused_bf16_indexer_head_gates(raw_gate, index_n_heads=32)
    expected = raw_gate.float() * 32**-0.5
    bf16_intermediate = (raw_gate * 32**-0.5).float()

    assert scaled.dtype is torch.float32
    assert torch.equal(scaled, expected)
    assert not torch.equal(scaled, bf16_intermediate)

    query_scale = torch.tensor([0.03125, 0.0625, 0.0625, 0.03125], dtype=torch.float32)
    native_score_weights = scaled * query_scale * 128**-0.5
    sampler_formula = raw_gate.float() * 32**-0.5 * query_scale * 128**-0.5
    assert torch.equal(native_score_weights, sampler_formula)


@pytest.mark.cpu
def test_fused_sampler_index_k_prepare_preserves_projection_stride_and_builds_literal_rope_cache():
    backing = torch.arange(3 * 160, dtype=torch.int32).reshape(1, 3, 160).to(torch.bfloat16)
    raw_key = backing[..., :128]
    norm_weight = torch.ones((128,), dtype=torch.float32)
    norm_bias = torch.zeros((128,), dtype=torch.float32)
    cos_half = torch.arange(3 * 32, dtype=torch.float32).reshape(1, 3, 32)
    sin_half = cos_half.neg()
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    calls = []

    def recording_kernel(key, weight, bias, eps, cos_sin_cache, positions):
        calls.append(
            {
                "key_stride": key.stride(),
                "weight": weight.clone(),
                "bias": bias.clone(),
                "eps": eps,
                "cos_sin_cache": cos_sin_cache.clone(),
                "positions": positions.clone(),
            }
        )
        return key.contiguous()

    prepared = _fused_sampler_index_k_prepare(
        raw_key,
        norm_weight,
        norm_bias,
        1e-6,
        (cos, sin),
        interleaved=True,
        _native_kernel_for_testing=recording_kernel,
    )

    assert len(calls) == 1
    assert calls[0]["key_stride"] == (160, 1)
    assert torch.equal(calls[0]["weight"], norm_weight)
    assert torch.equal(calls[0]["bias"], norm_bias)
    assert calls[0]["eps"] == 1e-6
    assert torch.equal(calls[0]["cos_sin_cache"], torch.cat((cos_half.reshape(3, 32), sin_half.reshape(3, 32)), -1))
    assert torch.equal(calls[0]["positions"], torch.arange(3, dtype=torch.int64))
    assert torch.equal(prepared, raw_key)


@pytest.mark.cpu
def test_sampler_index_k_preparation_uses_split_prompt_and_fused_decode_suffix():
    split = torch.full((1, 8, 128), 7, dtype=torch.bfloat16)
    backing = torch.arange(8 * 160, dtype=torch.int32).reshape(1, 8, 160).to(torch.bfloat16)
    raw = backing[..., :128]
    weight = torch.ones((128,), dtype=torch.float32)
    bias = torch.zeros((128,), dtype=torch.float32)
    cos = torch.ones((1, 8, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    calls = []

    def mark_fused_suffix(key, _weight, _bias, _eps, _cache, _positions):
        calls.append((tuple(key.shape), key.stride()))
        return torch.full_like(key, 11)

    mixed = _mix_sampler_index_k_preparation(
        split,
        raw,
        weight,
        bias,
        1e-6,
        (cos, sin),
        torch.tensor([6], dtype=torch.int64),
        query_offset=2,
        interleaved=True,
        _native_kernel_for_testing=mark_fused_suffix,
    )

    assert calls == [((4, 128), (160, 1))]
    assert torch.equal(mixed[:, :4], split[:, :4])
    assert torch.equal(mixed[:, 4:], torch.full_like(mixed[:, 4:], 11))


@pytest.mark.cpu
def test_sampler_index_k_preparation_maps_4096_boundary_across_cp16():
    local_length = 260
    split = torch.zeros((1, local_length, 128), dtype=torch.bfloat16)
    raw = torch.zeros_like(split)
    weight = torch.ones((128,), dtype=torch.float32)
    bias = torch.zeros((128,), dtype=torch.float32)
    cos = torch.ones((1, local_length, 64), dtype=torch.float32)
    sin = torch.zeros_like(cos)

    for cp_rank in range(16):
        calls = []

        def record_suffix(key, _weight, _bias, _eps, _cache, _positions):
            calls.append(key.shape[0])
            return torch.ones_like(key)

        mixed = _mix_sampler_index_k_preparation(
            split,
            raw,
            weight,
            bias,
            1e-6,
            (cos, sin),
            torch.tensor([4096], dtype=torch.int64),
            query_offset=cp_rank * local_length,
            interleaved=True,
            _native_kernel_for_testing=record_suffix,
        )

        if cp_rank < 15:
            assert calls == []
            assert torch.equal(mixed, split)
        else:
            assert calls == [64]
            assert torch.equal(mixed[:, :196], split[:, :196])
            assert torch.equal(mixed[:, 196:], torch.ones_like(mixed[:, 196:]))


@pytest.mark.cpu
def test_glm52_sparse_query_and_key_codecs_have_distinct_scale_contracts():
    source = torch.linspace(-3.0, 2.0, 128, dtype=torch.float32).to(torch.bfloat16).reshape(1, 128)
    _, query_scale = quantize_e4m3_ue8m0(source)
    _, key_scale = quantize_e4m3_dynamic(source)

    assert query_scale.item() == 0.0078125
    expected_key_scale = source.float().abs().amax(dim=-1, keepdim=True) / 448.0
    assert torch.equal(key_scale, expected_key_scale)
    assert key_scale.item() != query_scale.item()


@pytest.mark.cpu
def test_glm52_sparse_key_cache_unpack_preserves_sglang_page_layout():
    page_size = 64
    block_size = 128
    cache = torch.zeros((2, page_size * (block_size + 4)), dtype=torch.uint8)
    expected_bytes = torch.arange(65 * block_size, dtype=torch.int64).remainder(256).to(torch.uint8)
    cache[0, : page_size * block_size] = expected_bytes[: page_size * block_size]
    cache[1, :block_size] = expected_bytes[page_size * block_size :]
    expected_scales = torch.arange(65, dtype=torch.float32).add_(0.25)
    cache[0, page_size * block_size :].view(torch.float32).copy_(expected_scales[:page_size])
    cache[1, page_size * block_size :].view(torch.float32)[0] = expected_scales[page_size]

    quantized, scales = sparse_selector_module._unpack_sparse_key_cache(cache, 65)

    assert torch.equal(quantized.view(torch.uint8).reshape(-1), expected_bytes)
    assert torch.equal(scales, expected_scales)


@pytest.mark.gpu
@torch.no_grad()
def test_glm52_sparse_native_sampler_codecs_are_bitwise_at_production_shapes():
    pytest.importorskip(
        "sglang",
        reason=(
            "xorl-sglang is not installed by plain `uv sync` (the repositories "
            "pin different PyTorch constraints); install the paired xorl-sglang "
            "build documented in the PR body to run the GPU exact tests"
        ),
    )
    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
    from sglang.kernels.ops.attention.fused_store_index_cache import fused_store_index_k_cache

    torch.manual_seed(520052)
    query = torch.randn((4, 32, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((4160, 128), device="cuda", dtype=torch.bfloat16)

    query_fp8, query_scale = quantize_sparse_query(query)
    expected_query_fp8, expected_query_scale = act_quant(query.contiguous(), 128, "ue8m0")
    assert torch.equal(query_fp8.view(torch.uint8), expected_query_fp8.view(torch.uint8))
    assert torch.equal(query_scale.view(torch.uint8), expected_query_scale.view(torch.uint8))

    key_fp8, key_scale = quantize_sparse_key_cache(key)
    cache = torch.zeros((65, 64 * (128 + 4)), device="cuda", dtype=torch.uint8)
    locations = torch.arange(4160, device="cuda", dtype=torch.int64)
    fused_store_index_k_cache(key, cache, locations, page_size=64)
    expected_key_fp8, expected_key_scale = sparse_selector_module._unpack_sparse_key_cache(cache, 4160)
    assert torch.equal(key_fp8.view(torch.uint8), expected_key_fp8.view(torch.uint8))
    assert torch.equal(key_scale.view(torch.uint8).reshape(-1), expected_key_scale.view(torch.uint8).reshape(-1))

    repeated_key_fp8, repeated_key_scale = quantize_sparse_key_cache(key)
    assert torch.equal(key_fp8.view(torch.uint8), repeated_key_fp8.view(torch.uint8))
    assert torch.equal(key_scale.view(torch.uint8), repeated_key_scale.view(torch.uint8))


@pytest.mark.cpu
def test_glm52_sparse_native_dispatch_flattens_batches_and_masks_unwritten_cells():
    query = torch.zeros((2, 2, 2, 128), dtype=torch.bfloat16)
    key = torch.zeros((2, 3, 128), dtype=torch.bfloat16)
    weights = torch.ones((2, 2, 2), dtype=torch.float32)
    allowed = torch.tensor(
        [
            [[True, False, False], [True, True, False]],
            [[False, False, False], [True, True, True]],
        ],
        dtype=torch.bool,
    )
    calls = []

    def recording_kernel(q, kv, native_weights, starts, ends, *, clean_logits):
        calls.append((q.shape, kv[0].shape, kv[1].shape, native_weights.shape, starts.tolist(), ends.tolist()))
        return _fake_fp8_mqa_logits(q, kv, native_weights, starts, ends, clean_logits=clean_logits)

    selected = select_glm52_logical_indices(
        query,
        key,
        weights,
        allowed,
        topk=3,
        _native_kernel_for_testing=recording_kernel,
        _selector_for_testing=_reference_select,
    )
    assert calls == [
        (torch.Size([4, 2, 128]), torch.Size([6, 128]), torch.Size([6]), torch.Size([4, 2]), [0, 0, 3, 3], [1, 2, 3, 6])
    ]
    assert selected.logical_indices.values.tolist() == [
        [[0, -1, -1], [0, 1, -1]],
        [[-1, -1, -1], [0, 1, 2]],
    ]


@pytest.mark.cpu
def test_glm52_sparse_native_selector_fails_closed_without_cuda_and_on_nonprefix_mask():
    query = torch.zeros((1, 1, 2, 128), dtype=torch.bfloat16)
    key = torch.zeros((1, 2, 128), dtype=torch.bfloat16)
    weights = torch.ones((1, 1, 2), dtype=torch.float32)
    allowed = torch.ones((1, 1, 2), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="requires CUDA DeepGEMM"):
        select_glm52_logical_indices(query, key, weights, allowed, topk=1)

    allowed[0, 0] = torch.tensor([False, True])
    with pytest.raises(RuntimeError, match="contiguous request-local key prefixes"):
        select_glm52_logical_indices(
            query,
            key,
            weights,
            allowed,
            topk=1,
            _native_kernel_for_testing=_fake_fp8_mqa_logits,
        )


@pytest.mark.cpu
def test_glm52_sparse_deepgemm_loader_requires_score_capability(monkeypatch):
    monkeypatch.setattr(sparse_selector_module.importlib, "import_module", lambda _name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="deep_gemm.fp8_mqa_logits"):
        sparse_selector_module._load_sparse_score_kernel()


@pytest.mark.cpu
def test_glm52_sparse_selector_loader_imports_the_shared_kernel(monkeypatch):
    """Imports are the compatibility mechanism; there is no version handshake.

    A genuine API break fails naturally at import; residual numerical drift
    on either side reads as nonzero behavior_k3 in the first training steps
    and in the qualification replays.
    """

    def selector(scores, lengths, topk):
        del lengths
        return torch.empty((scores.shape[0], topk), dtype=torch.int32)

    module = SimpleNamespace()
    monkeypatch.setattr(sparse_selector_module.importlib, "import_module", lambda _name: module)
    with pytest.raises(RuntimeError, match="select_canonical_logical_topk"):
        sparse_selector_module._load_sparse_selection()

    module.select_canonical_logical_topk = selector
    assert sparse_selector_module._load_sparse_selection() is selector


@pytest.mark.cpu
def test_correction_bias_stays_fp32_and_checkpoint_ingestion_fails_closed():
    config = SimpleNamespace(n_routed_experts=4, hidden_size=3, _router_fp32=False)
    router = Glm5TopkRouter(config)
    official_values = torch.tensor([34.12345, -0.00314159, 0.33333334, 17.00013], dtype=torch.float32)
    router.e_score_correction_bias.copy_(official_values)

    source_bytes = router.e_score_correction_bias.view(torch.uint8).clone()

    router.to(dtype=torch.bfloat16)
    assert router.weight.dtype is torch.bfloat16
    assert router.e_score_correction_bias.dtype is torch.float32
    assert torch.equal(router.e_score_correction_bias, official_values)
    router.to(dtype=torch.float32)
    assert torch.equal(router.e_score_correction_bias.view(torch.uint8), source_bytes)

    meta_router = Glm5TopkRouter(config).to(device="meta")
    meta_router.to_empty(device="cpu")
    assert meta_router.weight.device.type == "cpu"
    assert meta_router.e_score_correction_bias.device.type == "cpu"
    assert meta_router.e_score_correction_bias.dtype is torch.float32
    assert meta_router.e_score_correction_bias.shape == official_values.shape

    clone = Glm5TopkRouter(config).to(dtype=torch.bfloat16)
    clone.load_state_dict(router.state_dict(), strict=True)
    assert clone.e_score_correction_bias.dtype is torch.float32
    assert torch.equal(clone.e_score_correction_bias, official_values)

    handler = Glm5CheckpointHandler(num_experts=4, checkpoint_has_per_expert=False)
    key = "model.layers.3.mlp.gate.e_score_correction_bias"
    assert handler.on_load_weight(key, official_values) == [(key, official_values)]
    with pytest.raises(TypeError, match="official FP32"):
        handler.on_load_weight(key, official_values.to(torch.bfloat16))
    with pytest.raises(ValueError, match="finite vector"):
        handler.on_load_weight(key, torch.tensor([0.0, 1.0, float("inf"), 3.0]))


@pytest.mark.cpu
def test_canonical_moe_rejects_routing_replay_configuration():
    config = _small_glm_config()
    config._glm52_exact_contract = True
    block = Glm5MoEBlock(config, layer_idx=1)
    block._routing_replay = object()
    with pytest.raises(RuntimeError, match="forbids routing replay"):
        block.route(torch.zeros((1, 1, config.hidden_size)))


@pytest.mark.cpu
def test_canonical_moe_transport_resolves_internally_with_no_public_knob():
    """There is no user-facing transport menu: the model resolves the best
    certified transport for the geometry internally."""
    config = _small_glm_config()
    config._glm52_exact_contract = True
    assert Glm5MoEBlock(config, layer_idx=1).canonical_moe_transport is CanonicalMoETransport.AUTO
    assert "canonical_moe_transport" not in config.to_dict()


@pytest.mark.cpu
def test_canonical_glm_router_and_indexer_are_exact_without_environment(monkeypatch):
    calls = []

    def router_gemm(hidden, weight):
        calls.append("router")
        return F.linear(hidden.float(), weight.float())

    def indexer_gemm(hidden, weight):
        calls.append("indexer")
        return F.linear(hidden.float(), weight.float())

    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5._BIRouterGemm.apply", router_gemm)
    monkeypatch.setattr(indexer_module, "bi_bf16_fp32_linear", indexer_gemm)

    config = _small_glm_config()
    config._glm52_exact_contract = True
    router = Glm5TopkRouter(config).to(torch.bfloat16)
    router(torch.zeros((2, config.hidden_size), dtype=torch.bfloat16))

    indexer = Glm5DsaIndexer(config).to(torch.bfloat16)
    hidden = torch.zeros((1, 2, config.hidden_size), dtype=torch.bfloat16)
    compressed = torch.zeros((1, 2, config.q_lora_rank), dtype=torch.bfloat16)
    cos = torch.ones((1, 2, config.qk_rope_head_dim), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    indexer.project(hidden, compressed, (cos, sin))
    assert calls == ["router", "indexer"]


@pytest.mark.cpu
def test_noncanonical_glm_retains_ordinary_router(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("noncanonical GLM unexpectedly used the exact router kernel")

    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5._BIRouterGemm.apply", forbidden)
    config = _small_glm_config()
    config.indexer_types = None
    config._router_fp32 = False
    router = Glm5TopkRouter(config).to(torch.bfloat16)
    output = router(torch.zeros((2, config.hidden_size), dtype=torch.bfloat16))
    assert output.dtype is torch.bfloat16


def _semantic_model_config(num_moe_layers: int) -> Glm5Config:
    config = Glm5Config(
        vocab_size=32,
        pad_token_id=0,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=num_moe_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=3,
        indexer_types=["full"] * num_moe_layers,
        index_topk_freq=1,
        index_skip_topk_offset=0,
        index_topk_pattern=[1] * num_moe_layers,
        mlp_layer_types=["sparse"] * num_moe_layers,
    )
    config._attn_implementation = "eager"
    config._glm52_exact_contract = True
    config._dsa_mask_disabled = True
    config._activation_native = True
    # See the Class-B CPU fallback note in the IndexShare fixture above.
    config._attention_cast_bf16 = True
    return config


def _semantic_rank_partials(block, hidden_states, routing_weights, selected_experts):
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    routed = block._eager_forward(flat, routing_weights, selected_experts.to(torch.long)).reshape_as(hidden_states)
    shared = block.shared_experts(hidden_states)
    base = ((routed + shared).to(torch.bfloat16) / 8).to(torch.bfloat16)
    offsets = (16.0, -16.0, 0.125, -0.125, 0.5, -0.5, 2.0, -2.0)
    return torch.stack([(base + offset).to(torch.bfloat16) for offset in offsets])


def _bind_semantic_canonicalizers(model, boundaries, *, serving: bool, skip_layer: int | None = None):
    for layer_id, layer in enumerate(model.model.layers):
        block = layer.mlp

        def canonical_forward(
            self,
            hidden_states,
            routing_weights,
            selected_experts,
            absolute_positions,
            *,
            _layer_id=layer_id,
        ):
            partials = _semantic_rank_partials(self, hidden_states, routing_weights, selected_experts)
            rows = hidden_states.shape[0] * hidden_states.shape[1]
            metadata = CanonicalMoEGraphMetadata.build(
                torch.arange(rows),
                absolute_positions.reshape(-1),
                capacity=rows,
            )
            flattened = partials.reshape(8, rows, hidden_states.shape[-1])
            if serving:
                level = tuple(flattened.unbind(0))
                while len(level) > 1:
                    level = tuple(
                        (level[index] + level[index + 1]).to(torch.bfloat16) for index in range(0, len(level), 2)
                    )
                canonical = level[0]
            else:
                canonical = canonical_moe_reduce_reference(flattened, metadata)
            if _layer_id == skip_layer:
                canonical = flattened[0]
            canonical = canonical.reshape_as(hidden_states)
            boundaries.append(canonical.detach().clone())
            return canonical

        block._canonical_ep_forward = MethodType(canonical_forward, block)


def _semantic_logprobs(model, input_ids):
    positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    hidden = model.model(
        input_ids=input_ids,
        position_ids=positions,
        index_share_mode=IndexShareMode.FORWARD_ONLY,
    ).last_hidden_state
    return F.log_softmax(model.lm_head(hidden).float(), dim=-1)


@pytest.mark.cpu
@pytest.mark.parametrize("num_moe_layers", [1, 4])
def test_semantic_moe_stack_boundary_logprob_engagement_permutation_and_composition(num_moe_layers, monkeypatch):
    def rowwise_router(hidden, weight):
        weight_fp32 = weight.float()
        return torch.stack(
            [torch.sum(row.float().unsqueeze(0) * weight_fp32, dim=1) for row in hidden],
            dim=0,
        )

    def semantic_serving_topk(
        hidden_states,
        router_logits,
        correction_bias,
        *,
        top_k,
        num_expert_group,
        topk_group,
        routed_scaling_factor,
    ):
        del hidden_states, routed_scaling_factor
        scores = router_logits.sigmoid()
        num_tokens, num_experts = scores.shape
        experts_per_group = num_experts // num_expert_group
        choice = scores + correction_bias
        group_scores = choice.view(num_tokens, num_expert_group, experts_per_group).topk(2, dim=-1)[0].sum(-1)
        groups = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, groups, True)
        score_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(num_tokens, num_experts)
        selected = torch.topk(choice.masked_fill(~score_mask, float("-inf")), k=top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, selected)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(torch.float32), selected.to(torch.int32)

    monkeypatch.setattr(
        "xorl.models.transformers.glm5.modeling_glm5._BIRouterGemm.apply",
        rowwise_router,
    )
    monkeypatch.setattr(
        "xorl.models.transformers.glm5.modeling_glm5._glm52_serving_grouped_topk",
        semantic_serving_topk,
    )
    torch.manual_seed(1234)
    trainer = Glm5ForCausalLM(_semantic_model_config(num_moe_layers)).to(torch.bfloat16).eval()
    sampler = Glm5ForCausalLM(_semantic_model_config(num_moe_layers)).to(torch.bfloat16).eval()
    sampler.load_state_dict(trainer.state_dict(), strict=True)
    trainer_boundaries = []
    sampler_boundaries = []
    _bind_semantic_canonicalizers(trainer, trainer_boundaries, serving=False)
    _bind_semantic_canonicalizers(sampler, sampler_boundaries, serving=True)

    batch = torch.tensor([[1, 2, 3], [4, 5, 6]])
    trainer_logprobs = _semantic_logprobs(trainer, batch)
    sampler_logprobs = _semantic_logprobs(sampler, batch)
    assert len(trainer_boundaries) == len(sampler_boundaries) == num_moe_layers
    assert all(torch.equal(left, right) for left, right in zip(trainer_boundaries, sampler_boundaries, strict=True))
    assert torch.equal(trainer_logprobs.view(torch.uint8), sampler_logprobs.view(torch.uint8))

    permuted = _semantic_logprobs(trainer, batch.flip(0)).flip(0)
    assert torch.equal(permuted.view(torch.uint8), trainer_logprobs.view(torch.uint8))
    solo = torch.cat([_semantic_logprobs(trainer, row.unsqueeze(0)) for row in batch], dim=0)
    assert torch.equal(solo.view(torch.uint8), trainer_logprobs.view(torch.uint8))

    if num_moe_layers == 4:
        faulty = Glm5ForCausalLM(_semantic_model_config(num_moe_layers)).to(torch.bfloat16).eval()
        faulty.load_state_dict(trainer.state_dict(), strict=True)
        _bind_semantic_canonicalizers(faulty, [], serving=False, skip_layer=0)
        faulty_logprobs = _semantic_logprobs(faulty, batch)
        assert not torch.equal(faulty_logprobs.view(torch.uint8), trainer_logprobs.view(torch.uint8))
