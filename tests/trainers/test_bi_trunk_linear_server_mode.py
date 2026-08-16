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


def _build(
    monkeypatch,
    captured,
    *,
    model=None,
    server_training=False,
    freeze_router=False,
    enable_lora=False,
):
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
        enable_lora=enable_lora,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def test_exact_server_model_wraps_trunk_linears_before_parallelize(monkeypatch):
    captured = {}
    result = _build(monkeypatch, captured, model=ExactTinyTrunkModel(), server_training=True)

    assert captured["server_training_at_build"] is True
    assert captured["wrapped_at_parallelize"] == 10, "wrap must land before parallelization (pre-FSDP2)"
    assert not getattr(result.model.lm_head, "_xorl_bi_trunk_wrapped", False)
    assert is_trunk_linear_contract_enabled()


def test_non_exact_server_model_does_not_wrap(monkeypatch):
    captured = {}
    _build(monkeypatch, captured, server_training=True)

    assert captured["server_training_at_build"] is True
    assert captured["wrapped_at_parallelize"] == 0
    assert not is_trunk_linear_contract_enabled()


def test_exact_model_is_not_implicitly_engaged_outside_server_training(monkeypatch):
    captured = {}
    _build(monkeypatch, captured, model=ExactTinyTrunkModel(), server_training=False)

    assert captured["server_training_at_build"] is False
    assert captured["wrapped_at_parallelize"] == 0


def test_glm52_selects_v2_family_structurally(monkeypatch):
    captured = {}
    model = TinyTrunkModel()
    model.config = SimpleNamespace(
        model_type="xorl_glm5",
        indexer_types=["full"],
        _glm52_exact_contract=True,
    )
    monkeypatch.setenv("XORL_FAMILIES_V2", "0")

    _build(monkeypatch, captured, model=model, freeze_router=True)

    assert bi_families_v2.families_v2_enabled() is True


def test_dense_qwen_exact_program_is_reinstalled_after_lora_replacement(monkeypatch):
    from xorl.lora.modules.linear import LoraLinear
    from xorl.ops.batch_invariant_ops import wrap_trunk_linears_batch_invariant

    captured = {}
    model = TinyTrunkModel()
    model.config._qwen3_dense_exact_contract = True
    monkeypatch.setenv("XORL_FAMILIES_V2", "0")

    # Match the foundation-model lifecycle: these monkey patches belong to
    # the original nn.Linear objects and LoRA injection replaces them.
    wrap_trunk_linears_batch_invariant(model)
    result = _build(monkeypatch, captured, model=model, server_training=True, enable_lora=True)

    assert captured["wrapped_at_parallelize"] == 10
    assert bi_families_v2.families_v2_enabled() is True
    for layer in result.model.layers:
        for name in ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            module = layer[name]
            assert type(module) is LoraLinear
            assert module.exact_merged_forward is True
            assert module._xorl_bi_trunk_wrapped is True


def test_ordinary_model_restores_nonexact_family_selection(monkeypatch):
    captured = {}
    bi_families_v2._select_glm52_families_v2()
    monkeypatch.setenv("XORL_FAMILIES_V2", "0")

    _build(monkeypatch, captured)

    assert bi_families_v2.families_v2_enabled() is False
