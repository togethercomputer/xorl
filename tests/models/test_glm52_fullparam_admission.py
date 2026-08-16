"""Model-level admission gates for GLM-5.2 full-param training.

The admitted forward topology for full-param mode is the canonical WORLD16
row (the EP16 combine path passes rank-local ids to the bank; its combine is
gated in tests/distributed/test_glm52_fullparam_ep16_combine.py).  These
tests gate the ADMISSION itself: component installation with byte-preserving
seeding, the trainable-target allowlist enforced by exhaustive walk,
fail-closed geometry/config validation, engagement logging, and the
step-boundary refresh/publication orchestration.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import xorl.models.transformers.glm5.exact_fullparam_admission as admission_module
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.exact_fullparam_admission import (
    Glm52FullParamTopkRouter,
    assert_glm52_fullparam_allowlist,
    install_glm52_fullparam_components,
    iter_glm52_fullparam_publication,
    prepare_glm52_fullparam_training,
    refresh_glm52_fullparam_caches,
)
from xorl.models.transformers.glm5.exact_fullparam_experts import (
    Glm52FullParamBlockFP8RoutedExperts,
)
from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    Glm52FullParamDenseMLP,
)
from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM


_OFFICIAL_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


def _tiny_config() -> Glm5Config:
    exclusions = ["model.embed_tokens", "lm_head"]
    exclusions.extend(f"model.layers.{layer}.self_attn.indexers_proj" for layer in range(2))
    quantization_config = dict(_OFFICIAL_QUANT_CONFIG)
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
        n_routed_experts=8,
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
        index_topk_freq=4,
        index_skip_topk_offset=1,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        pad_token_id=0,
        hidden_act="silu",
        quantization_config=quantization_config,
    )


def test_official_geometry_validation_fails_closed_on_reduced_and_misconfigured_rows() -> None:
    tiny = _tiny_config()
    with pytest.raises(ValueError, match="official model geometry"):
        prepare_glm52_fullparam_training(nn.Module(), tiny, trainable_expert_layers=(3,))

    official_fields = {
        "vocab_size": 154880,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "first_k_dense_replace": 3,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "hidden_act": "silu",
        "quantization_config": dict(_OFFICIAL_QUANT_CONFIG, modules_to_not_convert=["lm_head"]),
    }
    stub = SimpleNamespace(**official_fields, _ep_dispatch="deepep")
    with pytest.raises(ValueError, match="alltoall"):
        prepare_glm52_fullparam_training(nn.Module(), stub, trainable_expert_layers=(3,))
    stub = SimpleNamespace(**official_fields, _ep_dispatch="alltoall", _glm52_exact_contract=True)
    with pytest.raises(ValueError, match="scoring-only exact contract"):
        prepare_glm52_fullparam_training(nn.Module(), stub, trainable_expert_layers=(3,))


def test_expert_scope_validation_fails_closed() -> None:
    """The expert scope is validated before any installation."""

    tiny = _tiny_config()

    # Production entrypoint: an explicit non-empty scope is mandatory.
    with pytest.raises(TypeError, match="explicit trainable_expert_layers"):
        prepare_glm52_fullparam_training(nn.Module(), tiny, trainable_expert_layers=None)
    with pytest.raises(ValueError, match="non-empty trainable_expert_layers"):
        prepare_glm52_fullparam_training(nn.Module(), tiny, trainable_expert_layers=())
    with pytest.raises(TypeError, match="explicit trainable_expert_layers"):
        prepare_glm52_fullparam_training(nn.Module(), tiny, trainable_expert_layers="3")

    # Component-level scope validation (before get_submodule, so no model needed).
    with pytest.raises(ValueError, match="not a sparse"):
        install_glm52_fullparam_components(
            nn.Module(), tiny, trainable_expert_layers=(0,), _skip_geometry_validation=True
        )
    with pytest.raises(ValueError, match="not a sparse"):
        install_glm52_fullparam_components(
            nn.Module(), tiny, trainable_expert_layers=(7,), _skip_geometry_validation=True
        )
    with pytest.raises(ValueError, match="more than once"):
        install_glm52_fullparam_components(
            nn.Module(), tiny, trainable_expert_layers=(1, 1), _skip_geometry_validation=True
        )
    with pytest.raises(TypeError, match="sequence of layer indices"):
        install_glm52_fullparam_components(
            nn.Module(), tiny, trainable_expert_layers="1", _skip_geometry_validation=True
        )


def _hopper_or_skip() -> torch.device:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    return torch.device("cuda")


def _seed_native_bytes(model: nn.Module) -> None:
    """Fill every native FP8 module and the router with finite deterministic bytes.

    A checkpoint-less ``Glm5ForCausalLM`` holds uninitialized packed storage,
    whose garbage bytes can decode to FP8 NaN; the admission's master seeding
    would (correctly) refuse it.  This stands in for checkpoint loading.
    """

    from xorl.models.transformers.glm5.modeling_glm5 import Glm5TopkRouter
    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
    from xorl.ops.block_fp8_native import NativeBlockFP8Linear

    def _pattern(*shape: int, offset: int) -> torch.Tensor:
        values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
        return (((values * 3 + offset) % 29) - 14).reshape(shape).div(16.0)

    offset = 1
    for module in model.modules():
        if isinstance(module, NativeBlockFP8Linear):
            device = module.packed_weight_f32.device
            module.load_prequantized(
                _pattern(module.out_features, module.in_features, offset=offset).to(device).to(torch.float8_e4m3fn),
                torch.ones_like(module.weight_scale_inv),
            )
            offset += 1
        elif isinstance(module, Glm52NativeBlockFP8Experts):
            device = module.gate_up_packed_weight_f32.device
            experts = int(module.gate_up_packed_weight_f32.shape[0])
            module.load_prequantized(
                _pattern(experts, module.hidden_size, 2 * module.intermediate_size, offset=offset)
                .to(device)
                .to(torch.float8_e4m3fn),
                torch.ones_like(module.gate_up_weight_scale_inv),
                _pattern(experts, module.intermediate_size, module.hidden_size, offset=offset + 1)
                .to(device)
                .to(torch.float8_e4m3fn),
                torch.ones_like(module.down_weight_scale_inv),
            )
            offset += 2
        elif isinstance(module, Glm5TopkRouter):
            with torch.no_grad():
                module.weight.copy_(
                    _pattern(*module.weight.shape, offset=offset).to(module.weight.device, module.weight.dtype)
                )
                module.e_score_correction_bias.zero_()
            offset += 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_admission_installs_components_preserves_bytes_and_enforces_allowlist(caplog) -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    admission_module._engagement_logged = False

    config = _tiny_config()
    torch.manual_seed(3)
    model = Glm5ForCausalLM(config).to(torch.bfloat16).to(device)
    _seed_native_bytes(model)

    # Retain the pre-admission serving bytes (step-0 identity across the walk).
    dense_gate = model.get_submodule("model.layers.0.mlp.gate_proj")
    original_gate_bytes = dense_gate.fp8_weight().view(torch.uint8).clone()
    original_gate_scale = dense_gate.weight_scale_inv.detach().clone()
    bank = model.get_submodule("model.layers.1.mlp.experts")
    original_bank_bytes = bank.gate_up_proj.view(torch.uint8).clone()
    original_router_weight = model.get_submodule("model.layers.1.mlp.gate").weight.detach().clone()

    with caplog.at_level(logging.INFO, logger="xorl.models.transformers.glm5.exact_fullparam_admission"):
        report = install_glm52_fullparam_components(model, config, _skip_geometry_validation=True)
        install_second_model = Glm5ForCausalLM(config).to(torch.bfloat16).to(device)
        _seed_native_bytes(install_second_model)
        install_glm52_fullparam_components(install_second_model, config, _skip_geometry_validation=True)
    engagement_lines = [record for record in caplog.records if "full-param admission engaged" in record.message]
    assert len(engagement_lines) == 1  # logs once per process, not per model

    assert report.dense_mlp_layers == (0,)
    assert report.routed_expert_layers == (1,)
    assert report.router_layers == (1,)
    dense = model.get_submodule("model.layers.0.mlp")
    assert isinstance(dense, Glm52FullParamDenseMLP)
    new_bank = model.get_submodule("model.layers.1.mlp.experts")
    assert isinstance(new_bank, Glm52FullParamBlockFP8RoutedExperts)
    router = model.get_submodule("model.layers.1.mlp.gate")
    assert isinstance(router, Glm52FullParamTopkRouter)

    # Step-0 byte preservation through the admission walk.
    gate_view, gate_scale, _up_view, _up_scale = dense.publishable_split_projections()
    assert torch.equal(gate_view.view(torch.uint8), original_gate_bytes)
    assert torch.equal(gate_scale, original_gate_scale)
    published_bank = new_bank.publishable_expert_bytes()
    assert torch.equal(published_bank[0].view(torch.uint8), original_bank_bytes)
    assert torch.equal(router.full_param.publishable_weight_bytes(), original_router_weight)
    assert router.e_score_correction_bias.dtype is torch.float32

    # Allowlist: exactly the masters train; indexer/kv_b/lm_head/embeddings frozen.
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "model.layers.0.mlp.gate_up_proj.weight_master",
        "model.layers.0.mlp.down_proj.weight_master",
        "model.layers.1.mlp.experts.gate_up_weight_master",
        "model.layers.1.mlp.experts.down_weight_master",
        "model.layers.1.mlp.gate.full_param.weight_master",
    }
    for frozen_name in (
        "model.layers.0.self_attn.kv_b_proj.packed_weight_f32",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ):
        parameter = dict(model.named_parameters())[frozen_name]
        assert not parameter.requires_grad, frozen_name
    assert report.trained_parameter_count == 5
    assert "dsa_indexer" in report.frozen_group_sizes
    assert "attention_mla_projections" in report.frozen_group_sizes

    assert_glm52_fullparam_allowlist(model)
    dict(model.named_parameters())["model.layers.0.self_attn.indexer.weights_proj.weight"].requires_grad_(True)
    with pytest.raises(RuntimeError, match="allowlist violated"):
        assert_glm52_fullparam_allowlist(model)
    dict(model.named_parameters())["model.layers.0.self_attn.indexer.weights_proj.weight"].requires_grad_(False)

    # Publication enumeration covers exactly the admitted components and the
    # payload protocol accepts them end to end.
    from xorl.server.weight_sync.glm52_fullparam_payload import (
        publish_glm52_fullparam_payload,
        verify_glm52_fullparam_payload,
    )

    components = iter_glm52_fullparam_publication(model)
    assert [name for name, _ in components] == [
        "model.layers.0.mlp",
        *[f"model.layers.1.mlp.experts.{expert}" for expert in range(8)],
        "model.layers.1.mlp.gate.full_param",
    ]
    payload = publish_glm52_fullparam_payload(components, weight_version="admission-step-0")
    verify_glm52_fullparam_payload(payload)
    # Checkpoint-form kinds: the loader's split consumption, per publication unit.
    assert [item.kind for item in payload.items] == (
        ["block_fp8_dense_mlp"] + ["block_fp8_expert"] * 8 + ["bf16_router"]
    )
    # Global expert indexing was assigned by the admission (full bank -> offset 0).
    assert new_bank.global_expert_ids == tuple(range(8))
    # The disk-route mapping is mechanical over checkpoint FQNs.
    from xorl.server.weight_sync.glm52_fullparam_payload import glm52_fullparam_hf_name_mapping

    mapping = glm52_fullparam_hf_name_mapping(payload)
    assert mapping["model.layers.0.mlp"]["gate"] == "model.layers.0.mlp.gate_proj.weight"
    assert (
        mapping["model.layers.1.mlp.experts.5"]["down_scale_inv"]
        == "model.layers.1.mlp.experts.5.down_proj.weight_scale_inv"
    )
    assert mapping["model.layers.1.mlp.gate.full_param"]["weight"] == "model.layers.1.mlp.gate.weight"

    # Step-boundary orchestration: mutation without refresh fails closed at
    # publication; refresh_glm52_fullparam_caches restores the invariant.
    with torch.no_grad():
        dense.gate_up_proj.weight_master.add_(1.0)
        new_bank.gate_up_weight_master.mul_(1.5)
        router.full_param.weight_master.mul_(0.5)
    with pytest.raises(RuntimeError, match="stale"):
        publish_glm52_fullparam_payload(components, weight_version="admission-step-1")
    assert refresh_glm52_fullparam_caches(model) == 3
    payload = publish_glm52_fullparam_payload(components, weight_version="admission-step-1")
    verify_glm52_fullparam_payload(payload)

    # Admission is deliberately not idempotent.
    with pytest.raises(RuntimeError, match="not idempotent"):
        install_glm52_fullparam_components(model, config, _skip_geometry_validation=True)


def test_ep_expert_start_fails_closed_without_an_ep_group() -> None:
    """A partial (EP-local) bank without an initialized EP process group must
    raise the admission's own refusing-to-guess diagnostic — not a raw mesh
    error from the parallel-state property (ep_group RAISES when no mesh is
    initialized; getattr-with-default cannot catch that)."""

    assert admission_module._ep_expert_start(8, 8) == 0  # full bank: rank 0 by construction
    with pytest.raises(RuntimeError, match="refusing to guess"):
        admission_module._ep_expert_start(8, 16)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_scoped_admission_trains_only_named_expert_layers_with_real_envelopes() -> None:
    """trainable_expert_layers scopes expert banks; out-of-scope
    sparse layers keep frozen native banks (step-0-identical bytes) while
    their routers still train; the report carries REAL per-scope memory
    envelopes measured from the installed masters."""

    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts

    config = _tiny_config()
    config.num_hidden_layers = 3
    config.mlp_layer_types = ["dense", "sparse", "sparse"]
    config.indexer_types = ["full", "shared", "shared"]
    torch.manual_seed(5)
    model = Glm5ForCausalLM(config).to(torch.bfloat16).to(device)
    _seed_native_bytes(model)
    frozen_bank_bytes = model.get_submodule("model.layers.2.mlp.experts").gate_up_proj.view(torch.uint8).clone()

    report = install_glm52_fullparam_components(
        model, config, trainable_expert_layers=(1,), _skip_geometry_validation=True
    )
    assert report.dense_mlp_layers == (0,)
    assert report.routed_expert_layers == (1,)
    assert report.router_layers == (1, 2)

    # Scoped bank is full-param; out-of-scope bank stays the frozen native
    # bank with untouched step-0 bytes; BOTH routers train.
    assert isinstance(model.get_submodule("model.layers.1.mlp.experts"), Glm52FullParamBlockFP8RoutedExperts)
    out_of_scope = model.get_submodule("model.layers.2.mlp.experts")
    assert isinstance(out_of_scope, Glm52NativeBlockFP8Experts)
    assert not isinstance(out_of_scope, Glm52FullParamBlockFP8RoutedExperts)
    assert torch.equal(out_of_scope.gate_up_proj.view(torch.uint8), frozen_bank_bytes)
    assert isinstance(model.get_submodule("model.layers.1.mlp.gate"), Glm52FullParamTopkRouter)
    assert isinstance(model.get_submodule("model.layers.2.mlp.gate"), Glm52FullParamTopkRouter)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "model.layers.0.mlp.gate_up_proj.weight_master",
        "model.layers.0.mlp.down_proj.weight_master",
        "model.layers.1.mlp.experts.gate_up_weight_master",
        "model.layers.1.mlp.experts.down_weight_master",
        "model.layers.1.mlp.gate.full_param.weight_master",
        "model.layers.2.mlp.gate.full_param.weight_master",
    }

    # The memory envelope is measured from the INSTALLED masters, not config
    # arithmetic: recompute per group from the live parameters.
    expected_bytes = {"dense_mlp_masters": 0, "routed_expert_masters": 0, "router_masters": 0}
    for name in trainable:
        parameter = dict(model.named_parameters())[name]
        if ".experts." in name:
            group = "routed_expert_masters"
        elif ".gate.full_param." in name:
            group = "router_masters"
        else:
            group = "dense_mlp_masters"
        expected_bytes[group] += parameter.numel() * parameter.element_size()
    assert report.trained_master_bytes_by_group == expected_bytes
    assert report.adamw_step_bytes_by_group == {name: 4 * value for name, value in expected_bytes.items()}
    assert report.trained_master_bytes_by_group["routed_expert_masters"] > 0

    # Publication enumerates the scoped bank's experts, both routers, the
    # dense composite — and NOTHING from the out-of-scope bank.
    targets = [name for name, _ in iter_glm52_fullparam_publication(model)]
    assert targets == [
        "model.layers.0.mlp",
        *[f"model.layers.1.mlp.experts.{expert}" for expert in range(8)],
        "model.layers.1.mlp.gate.full_param",
        "model.layers.2.mlp.gate.full_param",
    ]

    # Step-boundary refresh covers dense + scoped bank + both routers.
    with torch.no_grad():
        model.get_submodule("model.layers.1.mlp.experts").gate_up_weight_master.mul_(1.5)
    assert refresh_glm52_fullparam_caches(model) == 4


def test_rebind_master_identities_restores_freshness_without_touching_cache_bytes() -> None:
    """Post-FSDP2-wrap semantics: wrapping
    replaces master storage, so pre-wrap freshness bindings go permanently
    stale; the rebind re-stamps identity ONLY — step-0 cache bytes (the
    checkpoint bytes and seeding order) must come through untouched."""

    import torch
    from torch import nn

    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        rebind_glm52_fullparam_master_identities,
    )
    from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
        Glm52ExactTP1BlockFP8FullParamLinear,
    )

    module = Glm52ExactTP1BlockFP8FullParamLinear(128, 256, device=torch.device("cpu"))
    with torch.no_grad():
        module.weight_master.zero_()
        raw = (torch.arange(module.quantized_weight_f32.numel() * 4, dtype=torch.int64) % 251).to(torch.uint8)
        module.quantized_weight_f32.copy_(raw.view(torch.float32).reshape(module.quantized_weight_f32.shape))
        module.weight_scale_inv.fill_(0.5)
    module._record_master_identity()
    seeded_weight, seeded_scale = module.publishable_weight_bytes()
    seeded_weight = seeded_weight.clone()
    seeded_scale = seeded_scale.clone()

    root = nn.Module()
    root.linear = module

    # Simulate the wrap: same VALUES, new storage (identity change).
    module.weight_master = nn.Parameter(module.weight_master.detach().clone())
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module.publishable_weight_bytes()

    ordinal_before = module._refresh_ordinal
    assert rebind_glm52_fullparam_master_identities(root) == 1
    assert module._refresh_ordinal == ordinal_before + 1

    weight_after, scale_after = module.publishable_weight_bytes()
    assert torch.equal(weight_after.view(torch.uint8), seeded_weight.view(torch.uint8))
    assert torch.equal(scale_after, seeded_scale)

    # A REAL post-rebind master mutation still trips the gate: the rebind
    # cannot mask genuine staleness introduced afterwards.
    with torch.no_grad():
        module.weight_master.add_(1.0)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module.publishable_weight_bytes()

    # No admitted components: refuse.
    with pytest.raises(RuntimeError, match="no admitted full-param components"):
        rebind_glm52_fullparam_master_identities(nn.Module())


def test_dcp_restore_rebinds_linear_and_router_without_requantizing_cache_bytes(monkeypatch) -> None:
    """A strict full-model restore writes masters and caches coherently but
    bumps master versions.  The restore-specific hook blesses exactly those
    restored pairs and leaves their checkpoint cache bytes untouched."""

    from types import SimpleNamespace

    from xorl.checkpoint import checkpointer
    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        restore_glm52_fullparam_master_identities,
    )
    from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
        Glm52ExactFullParamRouterWeight,
        Glm52ExactTP1BlockFP8FullParamLinear,
    )

    monkeypatch.setattr(checkpointer, "get_parallel_state", lambda: SimpleNamespace(dp_mode="none"))

    model = nn.Module()
    model.linear = Glm52ExactTP1BlockFP8FullParamLinear(128, 256, device="cpu")
    model.router = Glm52ExactFullParamRouterWeight(8, 128, device="cpu")
    with torch.no_grad():
        model.linear.weight_master.zero_()
        model.linear.quantized_weight_f32.zero_()
        model.linear.weight_scale_inv.fill_(0.25)
        model.router.weight_master.zero_()
        model.router._effective_weight.zero_()
    model.linear._record_master_identity()
    model.router._record_master_identity()

    restored_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    restored_state["linear.weight_master"].fill_(3.0)
    raw = (torch.arange(restored_state["linear.quantized_weight_f32"].numel() * 4, dtype=torch.int64) % 251).to(
        torch.uint8
    )
    restored_state["linear.quantized_weight_f32"].copy_(
        raw.view(torch.float32).reshape(restored_state["linear.quantized_weight_f32"].shape)
    )
    restored_state["linear.weight_scale_inv"].fill_(0.75)
    restored_state["router.weight_master"].fill_(5.0)
    restored_state["router._effective_weight"].fill_(2.0)

    checkpointer.ModelState(model).load_state_dict(restored_state)
    assert model.linear._master_is_stale()
    assert model.router._master_is_stale()
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        model.linear.publishable_weight_bytes()
    with pytest.raises(RuntimeError, match="stale BF16 view"):
        model.router.publishable_weight_bytes()

    weight_bytes = model.linear.quantized_weight_f32.detach().clone()
    scale_bytes = model.linear.weight_scale_inv.detach().clone()
    router_bytes = model.router._effective_weight.detach().clone()

    assert restore_glm52_fullparam_master_identities(model) == 2
    assert not model.linear._master_is_stale()
    assert not model.router._master_is_stale()
    assert torch.equal(model.linear.publishable_weight_bytes()[0].view(torch.uint8), weight_bytes.view(torch.uint8))
    assert torch.equal(model.linear.publishable_weight_bytes()[1].view(torch.uint8), scale_bytes.view(torch.uint8))
    assert torch.equal(model.router.publishable_weight_bytes().view(torch.uint8), router_bytes.view(torch.uint8))
