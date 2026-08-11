from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.test_glm52_qlora import _official_config
from xorl.distributed.canonical_moe import CANONICAL_MOE_REDUCE_VERSION
from xorl.models.auto import _is_exact_glm52, build_foundation_model
from xorl.models.exact_contract import (
    GLM52_EXACT_ACTIVE_LORA_FLAGS,
    glm52_exact_active_lora_enabled,
    glm52_exact_forward_enabled,
    set_glm52_exact_active_lora,
)
from xorl.models.layers.moe.moe_block import _moe_bi_router_enabled
from xorl.models.transformers.glm5.indexer import Glm5DsaIndexer
from xorl.models.transformers.glm5.modeling_glm5 import Glm5MoEBlock
from xorl.models.transformers.glm5.sparse_selector import GLM52_SELECTOR_VERSION


def _config_with_all_active_lora_flags():
    config = _official_config()
    config._glm52_exact_contract = False
    for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS:
        setattr(config, flag, True)
    return config


def test_complete_active_lora_flags_enable_exact_forward_without_scoring_marker() -> None:
    config = _config_with_all_active_lora_flags()

    assert glm52_exact_active_lora_enabled(config)
    assert glm52_exact_forward_enabled(config)
    assert _is_exact_glm52(config)
    assert _moe_bi_router_enabled(config)


def test_active_lora_component_flags_are_set_and_cleared_atomically() -> None:
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True

    set_glm52_exact_active_lora(config, enabled=False)
    assert all(getattr(config, flag) is False for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS)

    set_glm52_exact_active_lora(config, enabled=True)
    assert all(getattr(config, flag) is True for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS)


def _exact_world16_state(*, lm_head_tp_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        world_size=16,
        global_rank=0,
        pp_size=1,
        tp_size=1,
        dp_size=1,
        ep_size=16,
        cp_size=16,
        ringattn_size=1,
        ringattn_enabled=False,
        ulysses_size=16,
        lm_head_tp_size=lm_head_tp_size,
        cp_enabled=True,
    )


def _build_exact_active_lora(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lora_rank: int = 1,
    lora_alpha: int = 1,
    lm_head_tp_size: int = 16,
):
    class _Loader:
        def load_model(self, *, init_kwargs, **_kwargs):
            return SimpleNamespace(config=init_kwargs["config"])

    monkeypatch.setattr(
        "xorl.models.auto.get_parallel_state", lambda: _exact_world16_state(lm_head_tp_size=lm_head_tp_size)
    )
    monkeypatch.setattr("xorl.models.auto.get_loader", lambda _config: _Loader())
    monkeypatch.setattr("xorl.models.auto.get_attention_fn", lambda _implementation: object())
    config = _official_config()
    config._glm52_exact_active_lora_dense_component = True
    config._glm52_exact_active_lora_attention_component = False
    return build_foundation_model(
        config,
        moe_implementation="triton",
        ep_dispatch="alltoall",
        server_training=True,
        block_fp8_qlora_training=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        init_device="meta",
    )


def test_rank1_server_training_derives_the_complete_family_without_private_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_exact_active_lora(monkeypatch)

    assert glm52_exact_active_lora_enabled(model.config)
    assert model.config._glm52_exact_contract is False
    assert all(getattr(model.config, flag) is True for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS)


def test_rank16_server_training_derives_the_complete_family_without_private_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_exact_active_lora(monkeypatch, lora_rank=16, lora_alpha=32)

    assert glm52_exact_active_lora_enabled(model.config)
    assert model.config._glm52_exact_contract is False
    assert all(getattr(model.config, flag) is True for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS)


def test_rank1_server_training_rejects_non_tp16_lm_head_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="lm-head-TP16"):
        _build_exact_active_lora(monkeypatch, lm_head_tp_size=1)


@pytest.mark.parametrize("missing_flag", GLM52_EXACT_ACTIVE_LORA_FLAGS)
def test_every_active_lora_component_is_required(missing_flag: str) -> None:
    config = _config_with_all_active_lora_flags()
    setattr(config, missing_flag, False)

    assert not glm52_exact_active_lora_enabled(config)
    assert not glm52_exact_forward_enabled(config)
    assert not _is_exact_glm52(config)
    assert not _moe_bi_router_enabled(config)


def test_scoring_only_marker_remains_an_independent_exact_forward_admission() -> None:
    config = _official_config()
    config._glm52_exact_contract = True

    assert not glm52_exact_active_lora_enabled(config)
    assert glm52_exact_forward_enabled(config)


def test_cached_indexer_and_moe_surfaces_activate_only_for_complete_composite() -> None:
    complete = _config_with_all_active_lora_flags()
    partial = _config_with_all_active_lora_flags()
    partial._glm52_exact_active_lora_lm_head_component = False

    with torch.device("meta"):
        complete_indexer = Glm5DsaIndexer(complete)
        partial_indexer = Glm5DsaIndexer(partial)
        complete_moe = Glm5MoEBlock(complete)
        partial_moe = Glm5MoEBlock(partial)

    assert complete_indexer.selector_version == GLM52_SELECTOR_VERSION
    assert partial_indexer.selector_version == "legacy_torch_or_tilelang"
    assert complete_moe.canonical_contract_version == CANONICAL_MOE_REDUCE_VERSION
    assert partial_moe.canonical_contract_version is None
