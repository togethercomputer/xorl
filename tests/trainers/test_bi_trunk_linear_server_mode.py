"""Server-training model-program selection before FSDP2."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.ops import bi_families_v2
from xorl.ops.batch_invariant_ops import is_trunk_linear_contract_enabled, set_trunk_linear_contract
from xorl.trainers.model_builder import build_training_model


pytestmark = [pytest.mark.cpu]


class TinyTrunkModel(nn.Module):
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q_proj": nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                    "o_proj": nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                    "gate_proj": nn.Linear(16, 32, bias=False, dtype=torch.bfloat16),
                    "up_proj": nn.Linear(16, 32, bias=False, dtype=torch.bfloat16),
                    "down_proj": nn.Linear(32, 16, bias=False, dtype=torch.bfloat16),
                }
            )
            for _ in range(2)
        )
        self.lm_head = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)


class ExactTinyTrunkModel(TinyTrunkModel):
    def _apply_qwen35_gdn_exact(self):
        from xorl.ops.batch_invariant_ops import wrap_trunk_linears_batch_invariant

        return wrap_trunk_linears_batch_invariant(self)


@pytest.fixture(autouse=True)
def _reset_contract_state():
    yield
    set_trunk_linear_contract(False)
    bi_families_v2._select_nonexact_families()


def _build(monkeypatch, captured, *, model=None, server_training=False, freeze_router=False):
    def fake_parallelize(model, **_kwargs):
        captured["wrapped_at_parallelize"] = sum(
            1 for _, m in model.named_modules() if getattr(m, "_xorl_bi_trunk_wrapped", False)
        )
        return model

    model = TinyTrunkModel() if model is None else model

    def fake_build_foundation_model(**kwargs):
        captured["server_training_at_build"] = kwargs["server_training"]
        return model

    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", fake_build_foundation_model)
    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", fake_parallelize)
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)

    return build_training_model(
        config_path="unused",
        weights_path="unused",
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
        server_training=server_training,
        freeze_router=freeze_router,
    )


def test_exact_server_trunk_and_numerical_family_selection_policy(monkeypatch):
    with monkeypatch.context() as trunk_patch:
        _assert_trunk_linear_contract_engages_only_for_exact_server_models(trunk_patch)
    with monkeypatch.context() as family_patch:
        _assert_model_structure_selects_and_restores_v2_numerical_families(family_patch)


def _assert_trunk_linear_contract_engages_only_for_exact_server_models(monkeypatch):
    captured = {}
    result = _build(monkeypatch, captured, model=ExactTinyTrunkModel(), server_training=True)

    assert captured["server_training_at_build"] is True
    assert captured["wrapped_at_parallelize"] == 10, "wrap must land before parallelization (pre-FSDP2)"
    assert not getattr(result.model.lm_head, "_xorl_bi_trunk_wrapped", False)
    assert is_trunk_linear_contract_enabled()

    set_trunk_linear_contract(False)
    captured = {}
    _build(monkeypatch, captured, server_training=True)

    assert captured["server_training_at_build"] is True
    assert captured["wrapped_at_parallelize"] == 0
    assert not is_trunk_linear_contract_enabled()

    captured = {}
    _build(monkeypatch, captured, model=ExactTinyTrunkModel(), server_training=False)

    assert captured["server_training_at_build"] is False
    assert captured["wrapped_at_parallelize"] == 0


def _assert_model_structure_selects_and_restores_v2_numerical_families(monkeypatch):
    bi_families_v2._select_nonexact_families()
    assert bi_families_v2.families_v2_enabled() is True

    captured = {}
    model = TinyTrunkModel()
    model.config = SimpleNamespace(
        model_type="xorl_glm5",
        indexer_types=["full"],
        _glm52_exact_contract=True,
    )
    _build(monkeypatch, captured, model=model, freeze_router=True)

    assert bi_families_v2.families_v2_enabled() is True
    assert bi_families_v2._EXACT_FAMILIES_VERSION == "v2"

    captured = {}
    _build(monkeypatch, captured)

    assert bi_families_v2.families_v2_enabled() is True
    assert bi_families_v2._EXACT_FAMILIES_VERSION is None
