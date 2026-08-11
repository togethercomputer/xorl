from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import pytest
import torch
from torch import nn
from torch.distributed._tensor import Replicate, Shard

from tests.models.test_glm52_qlora import _meta_model, _official_config
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.exact_lm_head_qlora import (
    Glm52ExactTP16LmHeadLoraLinear,
    Glm52ExactTP16LmHeadSelectedLogprob,
    glm52_lm_head_shard,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.modeling_glm5 import Glm5MoEBlock
from xorl.models.transformers.glm5.parallelize import get_ep_plan
from xorl.models.transformers.glm5.qlora import GLM52_QLORA_FACTOR_COUNT, prepare_glm52_block_fp8_qlora
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts
from xorl.server.runner.adapters.sharded_state import discover_adapter_layouts


_ORDINARY_ATTENTION_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")
_ALL_ATTENTION_PROJECTIONS = (*_ORDINARY_ATTENTION_PROJECTIONS, "kv_b_proj")


@dataclass(frozen=True)
class _EPState:
    ep_enabled: bool
    ep_size: int
    ep_rank: int
    tp_size: int = 1
    pp_size: int = 1
    lm_head_tp_size: int = 1
    lm_head_tp_group: object | None = None
    lm_head_mesh: object | None = None


class _FakeEPMesh:
    ndim = 1

    def size(self, mesh_dim: int | None = None) -> int:
        assert mesh_dim in (None, -1, 0)
        return 16

    def get_local_rank(self, mesh_dim: int | None = None) -> int:
        assert mesh_dim in (None, 0)
        return 7


class _FakeEPFSDPMesh:
    def __init__(self) -> None:
        self.ep = _FakeEPMesh()

    def __getitem__(self, mesh_dim_name: str) -> _FakeEPMesh:
        assert mesh_dim_name == "ep"
        return self.ep


def _exact_moe_config():
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True
    config._glm52_exact_active_lora_attention_component = True
    config._glm52_exact_active_lora_shared_expert_component = True
    config._glm52_exact_active_lora_routed_expert_component = True
    config._sparse_mla_enabled = True
    config._ep_dispatch = "alltoall"
    return config


def _patch_ep16_rank7(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xorl.models.transformers.glm5.qlora.get_parallel_state",
        lambda: _EPState(ep_enabled=True, ep_size=16, ep_rank=7),
    )


def _patch_exact_world16_rank7(monkeypatch: pytest.MonkeyPatch) -> object:
    group = object()
    state = _EPState(
        ep_enabled=True,
        ep_size=16,
        ep_rank=7,
        lm_head_tp_size=16,
        lm_head_tp_group=group,
        lm_head_mesh=object(),
    )
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.get_parallel_state", lambda: state)
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.dist.is_initialized", lambda: True)
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.dist.get_world_size", lambda requested=None: 16)
    monkeypatch.setattr(
        "xorl.models.transformers.glm5.qlora.dist.get_process_group_ranks",
        lambda requested: list(range(16)),
    )
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.dist.get_backend", lambda requested: "nccl")
    monkeypatch.setattr(
        "xorl.models.transformers.glm5.qlora.dist.get_rank",
        lambda requested=None: 7,
    )
    return group


def _empty_moe_block() -> Glm5MoEBlock:
    block = Glm5MoEBlock.__new__(Glm5MoEBlock)
    nn.Module.__init__(block)
    block.routed_scaling_factor = 2.5
    return block


def _assert_canonical_moe_routed_and_shared_boundary_policy() -> None:
    block = _empty_moe_block()
    experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(128, 128, ep_rank=7, device="cpu")
    captured = {}

    def forward(self, hidden, routing, selected_experts=None, **kwargs):
        captured.update(
            hidden=hidden,
            routing=routing,
            selected_experts=selected_experts,
            local_ids=kwargs["sglang_ep_native_local_ids"],
            routed_scaling_factor=kwargs["routed_scaling_factor"],
        )
        return torch.ones_like(hidden)

    experts.forward = MethodType(forward, experts)
    block.experts = experts
    hidden = torch.zeros((3, 128), dtype=torch.bfloat16)
    routing = torch.arange(24, dtype=torch.float32).reshape(3, 8).div_(32)
    global_ids = torch.arange(24, dtype=torch.int64).reshape(3, 8).add_(112)
    local_ids = torch.arange(24, dtype=torch.int32).reshape(3, 8).remainder_(16)

    output = block._canonical_routed_local_partial(hidden, routing, global_ids, local_ids)

    assert torch.equal(output, torch.ones_like(hidden))
    assert captured["hidden"] is hidden
    assert captured["routing"] is routing
    assert captured["selected_experts"] is global_ids
    assert captured["local_ids"] is local_ids
    assert captured["routed_scaling_factor"] == 2.5

    shared = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta")
    captured.clear()

    def shared_forward(self, shared_hidden, *, contributor_ordinal):
        captured.update(hidden=shared_hidden, contributor_ordinal=contributor_ordinal)
        return torch.full_like(shared_hidden, 0.5)

    shared.forward = MethodType(shared_forward, shared)
    block.shared_experts = shared
    hidden = torch.zeros((3, 6144), dtype=torch.bfloat16)

    output = block._canonical_shared_local_partial(
        hidden,
        contributor_ordinal=7,
        contributor_count=16,
    )

    assert torch.equal(output, torch.full_like(hidden, 0.5))
    assert captured["hidden"] is hidden
    assert captured["contributor_ordinal"] == 7


def test_glm52_exact_moe_construction_preserves_complete_global_inventory_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ep16_rank7(monkeypatch)
    config = _exact_moe_config()
    model = _meta_model(config)

    inventory = prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    expected_shared = {f"model.layers.{layer_idx}.mlp.shared_experts" for layer_idx in range(3, 78)}
    expected_routed = {f"model.layers.{layer_idx}.mlp.experts" for layer_idx in range(3, 78)}
    exact_shared = {
        name for name, module in model.named_modules() if type(module) is Glm52ExactTP16SharedExpertBlockFP8QLoRA
    }
    exact_routed = {
        name for name, module in model.named_modules() if type(module) is Glm52ExactEP16BlockFP8QLoRARoutedExperts
    }
    assert exact_shared == expected_shared
    assert exact_routed == expected_routed
    assert len(exact_shared) == len(exact_routed) == 75

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert len(inventory.factors) == GLM52_QLORA_FACTOR_COUNT == 1700
    assert len(inventory.factor_names) == GLM52_QLORA_FACTOR_COUNT
    assert set(trainable) == inventory.factor_names
    assert len({id(parameter) for parameter in trainable.values()}) == GLM52_QLORA_FACTOR_COUNT
    assert all(parameter.dtype is torch.float32 for parameter in trainable.values())
    assert all(factor.dtype is torch.float32 for factor in inventory.factors)

    _assert_complete_attention_and_dense_inventory(model, inventory, trainable)

    assert not any(
        isinstance(module, BlockFP8QLoRALinear)
        for name, module in model.named_modules()
        if any(name.startswith(f"{root}.") for root in expected_shared)
    )
    assert not any(isinstance(module, BlockFP8QLoRAMoeExperts) for module in model.modules())

    for layer_idx in range(3, 78):
        shared_fqn = f"model.layers.{layer_idx}.mlp.shared_experts"
        shared = model.get_submodule(shared_fqn)
        assert type(shared) is Glm52ExactTP16SharedExpertBlockFP8QLoRA
        assert shared._checkpoint_source_prefix == shared_fqn
        assert {name for name in trainable if name.startswith(f"{shared_fqn}.")} == {
            f"{shared_fqn}.{projection}.lora_{factor}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for factor in ("A", "B")
        }
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(shared, projection_name)
            assert projection._source_fqn == f"{shared_fqn}.{projection_name}"
            assert projection._source_quant_format == "block_fp8"
            assert projection._is_prequantized is True
            assert projection._merge_sources is None
            assert projection._qlora_expected_skip_keys == {"weight", "weight_scale_inv"}

        routed_fqn = f"model.layers.{layer_idx}.mlp.experts"
        routed = model.get_submodule(routed_fqn)
        assert type(routed) is Glm52ExactEP16BlockFP8QLoRARoutedExperts
        assert routed._source_fqn == routed_fqn
        assert routed._source_quant_format == "block_fp8"
        assert (routed.num_experts, routed.ep_size, routed.num_local_experts) == (256, 16, 16)
        assert (routed.ep_rank, routed.expert_offset, routed.moe_tp_size, routed.ep_dispatch) == (
            7,
            112,
            1,
            "alltoall",
        )
        assert tuple(routed.gate_proj_lora_A.shape) == (1, 6144, 1)
        assert tuple(routed.gate_proj_lora_B.shape) == (256, 1, 2048)
        assert tuple(routed.up_proj_lora_A.shape) == (1, 6144, 1)
        assert tuple(routed.up_proj_lora_B.shape) == (256, 1, 2048)
        assert tuple(routed.down_proj_lora_A.shape) == (256, 2048, 1)
        assert tuple(routed.down_proj_lora_B.shape) == (1, 1, 6144)
        assert {name for name in trainable if name.startswith(f"{routed_fqn}.")} == {
            f"{routed_fqn}.{factor_name}" for factor_name in routed.logical_factor_names
        }

    with monkeypatch.context() as admission_patch:
        _assert_glm52_exact_moe_construction_admission_policy(admission_patch)
    with monkeypatch.context() as layout_patch:
        _assert_glm52_exact_moe_post_ep_layout_preserves_factor_fqns_and_owner_logical_shapes(layout_patch)
    with monkeypatch.context() as head_patch:
        _assert_glm52_complete_exact_construction_attaches_only_selected_logprob_lm_head(head_patch)
    _assert_canonical_moe_routed_and_shared_boundary_policy()


def _assert_complete_attention_and_dense_inventory(model, inventory, trainable) -> None:
    expected_attention_factors = {
        f"model.layers.{layer_idx}.self_attn.{projection}.lora_{factor}"
        for layer_idx in range(78)
        for projection in _ALL_ATTENTION_PROJECTIONS
        for factor in ("A", "B")
    }
    actual_attention_factors = {factor.name for factor in inventory.factors if factor.role.startswith("attention.")}
    assert actual_attention_factors == expected_attention_factors
    assert len(actual_attention_factors) == 78 * 5 * 2 == 780

    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        prefix = f"model.layers.{layer_idx}.self_attn"
        for projection in _ORDINARY_ATTENTION_PROJECTIONS:
            module = getattr(attention, projection)
            assert type(module) is Glm52ExactTP1BlockFP8QLoRALinear
            assert module._source_fqn == f"{prefix}.{projection}"
        assert type(attention.kv_b_proj) is Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA
        assert attention.kv_b_proj._source_fqn == f"{prefix}.kv_b_proj"
        assert {name for name in trainable if name.startswith(f"{prefix}.")} == {
            f"{prefix}.{projection}.lora_{factor}" for projection in _ALL_ATTENTION_PROJECTIONS for factor in ("A", "B")
        }

    assert {name for name, module in model.named_modules() if isinstance(module, Glm52ExactTP1DenseMLP)} == {
        f"model.layers.{layer_idx}.mlp" for layer_idx in range(3)
    }


def _assert_glm52_exact_moe_post_ep_layout_preserves_factor_fqns_and_owner_logical_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ep16_rank7(monkeypatch)
    config = _exact_moe_config()
    model = _meta_model(config)
    inventory = prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)
    pre_ep_factor_names = inventory.factor_names

    specs = get_ep_plan().apply(model, _FakeEPFSDPMesh(), already_local=True)
    model._fqn2spec_info = specs
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert len(trainable) == GLM52_QLORA_FACTOR_COUNT == 1700
    assert set(trainable) == pre_ep_factor_names

    parameter_metadata = {}
    for name, parameter in trainable.items():
        if name.endswith("lora_A"):
            rank_dim = 0 if parameter.ndim == 2 else 2
        else:
            assert name.endswith("lora_B")
            rank_dim = 1
        parameter_metadata[name] = {
            "shape": tuple(parameter.shape),
            "dtype": parameter.dtype,
            "rank_dim": rank_dim,
        }
    layouts, _fingerprint, _memberships = discover_adapter_layouts(
        model,
        parameter_metadata,
        active_rank=1,
        local_group_memberships={},
    )
    assert set(layouts) == pre_ep_factor_names

    shared_factors = {"gate_proj_lora_A", "up_proj_lora_A", "down_proj_lora_B"}
    owner_factors = {"gate_proj_lora_B", "up_proj_lora_B", "down_proj_lora_A"}
    for layer_idx in range(3, 78):
        routed_fqn = f"model.layers.{layer_idx}.mlp.experts"
        for local_name in shared_factors:
            full_name = f"{routed_fqn}.{local_name}"
            assert tuple(trainable[full_name].shape[:1]) == (1,)
            assert isinstance(specs[full_name].placement, Replicate)
            assert layouts[full_name].logical_shape[0] == 1
            assert layouts[full_name].local_logical_shape[0] == 1
        for local_name in owner_factors:
            full_name = f"{routed_fqn}.{local_name}"
            assert tuple(trainable[full_name].shape[:1]) == (16,)
            assert isinstance(specs[full_name].placement, Shard)
            assert specs[full_name].placement.dim == 0
            assert layouts[full_name].logical_shape[0] == 256
            assert layouts[full_name].local_logical_shape[0] == 16
            assert layouts[full_name].local_logical_offset[0] == 112


def _assert_glm52_complete_exact_construction_attaches_only_selected_logprob_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _patch_exact_world16_rank7(monkeypatch)
    config = _exact_moe_config()
    config._glm52_exact_active_lora_lm_head_component = True
    model = _meta_model(config)

    inventory = prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    lm_head = model.lm_head
    assert type(lm_head) is Glm52ExactTP16LmHeadLoraLinear
    assert lm_head._glm52_exact_tp16_lm_head is True
    assert lm_head._glm52_exact_replicated_parameter_names == ("lora_A",)
    assert type(lm_head._glm52_exact_selected_logprob) is Glm52ExactTP16LmHeadSelectedLogprob
    assert lm_head._glm52_exact_selected_logprob.tp_group is group
    assert lm_head._glm52_exact_selected_logprob.shard == glm52_lm_head_shard(7)
    assert tuple(lm_head.weight.shape) == (154_880, 6_144)
    assert tuple(lm_head.lora_A.shape) == (1, 6_144)
    assert tuple(lm_head.lora_B.shape) == (154_880, 1)
    assert lm_head.lora_A.dtype is lm_head.lora_B.dtype is torch.float32
    assert len(inventory.factors) == GLM52_QLORA_FACTOR_COUNT
    assert {factor.name for factor in inventory.factors if factor.target_name == "lm_head"} == {
        "lm_head.lora_A",
        "lm_head.lora_B",
    }
    with pytest.raises(RuntimeError, match="cannot materialize or execute"):
        lm_head.get_delta_weight()
    with pytest.raises(RuntimeError, match="cannot materialize or execute"):
        lm_head(torch.empty((1, 6_144), device="meta", dtype=torch.bfloat16))


def _assert_glm52_exact_moe_construction_rejects_incomplete_dependency_flags(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    message: str,
) -> None:
    _patch_ep16_rank7(monkeypatch)
    config = _exact_moe_config()
    for name, value in override.items():
        setattr(config, name, value)
    model = _meta_model(config)

    with pytest.raises(ValueError, match=message):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    assert not any("lora_" in name for name, _ in model.named_parameters())


def _assert_glm52_exact_moe_construction_admission_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for override, message in (
        ({"_glm52_exact_active_lora_dense_component": False}, "requires the exact active-LoRA dense component"),
        (
            {"_glm52_exact_active_lora_attention_component": False},
            "requires the exact active-LoRA attention component",
        ),
        (
            {"_glm52_exact_active_lora_shared_expert_component": False},
            "requires the exact active-LoRA shared-expert component",
        ),
        (
            {
                "_glm52_exact_active_lora_routed_expert_component": False,
                "_glm52_exact_active_lora_lm_head_component": True,
            },
            "requires the exact active-LoRA routed-expert component",
        ),
        ({"_sparse_mla_enabled": False}, "requires sparse_mla_enabled=true"),
        ({"_ep_dispatch": "deepep"}, "requires ep_dispatch='alltoall'"),
    ):
        _assert_glm52_exact_moe_construction_rejects_incomplete_dependency_flags(monkeypatch, override, message)

    for state in (
        _EPState(ep_enabled=False, ep_size=16, ep_rank=0),
        _EPState(ep_enabled=True, ep_size=8, ep_rank=7),
    ):
        _assert_glm52_exact_moe_construction_rejects_non_ep16(monkeypatch, state)

    group = object()
    state = _EPState(
        ep_enabled=True,
        ep_size=16,
        ep_rank=7,
        lm_head_tp_size=8,
        lm_head_tp_group=group,
        lm_head_mesh=object(),
    )
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.get_parallel_state", lambda: state)
    config = _exact_moe_config()
    config._glm52_exact_active_lora_lm_head_component = True
    model = _meta_model(config)

    with pytest.raises(RuntimeError, match="requires initialized lm-head TP16"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    assert not any("lora_" in name for name, _ in model.named_parameters())

    for rank, alpha in ((1, 2), (2, 1)):
        _assert_glm52_exact_moe_construction_rejects_non_rank1_alpha1(monkeypatch, rank, alpha)


def _assert_glm52_exact_moe_construction_rejects_non_ep16(
    monkeypatch: pytest.MonkeyPatch,
    state: _EPState,
) -> None:
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.get_parallel_state", lambda: state)
    config = _exact_moe_config()
    model = _meta_model(config)

    with pytest.raises(RuntimeError, match="require initialized EP16"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    assert not any("lora_" in name for name, _ in model.named_parameters())


def _assert_glm52_exact_moe_construction_rejects_non_rank1_alpha1(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    alpha: int,
) -> None:
    _patch_ep16_rank7(monkeypatch)
    config = _exact_moe_config()
    model = _meta_model(config)

    with pytest.raises(ValueError, match="requires adapter_rank=1 and adapter_alpha=1"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=rank, adapter_alpha=alpha)

    assert not any("lora_" in name for name, _ in model.named_parameters())
