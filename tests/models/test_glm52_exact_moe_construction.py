from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.distributed._tensor import Replicate, Shard

from tests.models.test_glm52_qlora import _meta_model, _official_config
from xorl.models.transformers.glm5.exact_lm_head_qlora import (
    Glm52ExactTP16LmHeadLoraLinear,
    Glm52ExactTP16LmHeadSelectedLogprob,
    glm52_lm_head_shard,
)
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.parallelize import get_ep_plan
from xorl.models.transformers.glm5.qlora import GLM52_QLORA_FACTOR_COUNT, prepare_glm52_block_fp8_qlora
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts
from xorl.server.runner.adapters.sharded_state import discover_adapter_layouts


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


def test_glm52_exact_moe_post_ep_layout_preserves_factor_fqns_and_owner_logical_shapes(
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


def test_glm52_complete_exact_construction_attaches_only_selected_logprob_lm_head(
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


@pytest.mark.parametrize(
    ("override", "message"),
    (
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
    ),
)
def test_glm52_exact_moe_construction_rejects_incomplete_dependency_flags_before_mutation(
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


@pytest.mark.parametrize(
    "state",
    (
        _EPState(ep_enabled=False, ep_size=16, ep_rank=0),
        _EPState(ep_enabled=True, ep_size=8, ep_rank=7),
    ),
)
def test_glm52_exact_moe_construction_rejects_non_ep16_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    state: _EPState,
) -> None:
    monkeypatch.setattr("xorl.models.transformers.glm5.qlora.get_parallel_state", lambda: state)
    config = _exact_moe_config()
    model = _meta_model(config)

    with pytest.raises(RuntimeError, match="require initialized EP16"):
        prepare_glm52_block_fp8_qlora(model, config, adapter_rank=1, adapter_alpha=1)

    assert not any("lora_" in name for name, _ in model.named_parameters())


def test_glm52_exact_lm_head_rejects_non_world16_before_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.mark.parametrize(("rank", "alpha"), ((16, 16), (1, 2), (2, 1)))
def test_glm52_exact_moe_construction_rejects_non_rank1_alpha1_before_mutation(
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
