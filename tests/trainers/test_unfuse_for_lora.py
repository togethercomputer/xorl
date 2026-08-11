"""``unfuse_for_lora``: split fused projections before LoRA selects modules by name.

The ordering test is the load-bearing one. ``unfuse_for_tp`` allocates fresh ``nn.Linear``s
without copying from the fused weight, so it has to run before LoRA injection *and* before
the checkpoint is read (which happens inside ``_parallelize``). Getting that order wrong
produces a model whose new projections are never filled.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from xorl.trainers.model_builder import build_training_model, maybe_unfuse_projections


pytestmark = [pytest.mark.cpu]


class _Unfusable(nn.Module):
    def __init__(self, calls=None):
        super().__init__()
        self.calls = calls if calls is not None else []
        self.config = SimpleNamespace(num_experts=0, model_type="qwen3")
        self._no_split_modules = []
        self.proj = nn.Linear(4, 4, bias=False)

    def unfuse_for_tp(self):
        self.calls.append("unfuse")


class _NotUnfusable(nn.Module):
    pass


class TestMaybeUnfuseProjections:
    def test_noop_when_disabled(self):
        model = _Unfusable()

        maybe_unfuse_projections(model, enabled=False)

        assert model.calls == []

    def test_unfuses_when_enabled(self):
        model = _Unfusable()

        maybe_unfuse_projections(model, enabled=True)

        assert model.calls == ["unfuse"]

    def test_raises_rather_than_silently_skipping(self):
        """A silent skip is the exact failure mode this work exists to remove: the
        targets would train unadapted while the config claims otherwise."""
        with pytest.raises(NotImplementedError, match="cannot unfuse"):
            maybe_unfuse_projections(_NotUnfusable(), enabled=True)

    def test_unsupported_architecture_is_fine_while_disabled(self):
        maybe_unfuse_projections(_NotUnfusable(), enabled=False)


class TestBuildTrainingModelOrdering:
    @staticmethod
    def _patch(monkeypatch, model, calls):
        monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **kwargs: model)
        monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *a, **k: None)
        monkeypatch.setattr(
            "xorl.trainers.model_builder.inject_lora_into_model",
            lambda *a, **k: calls.append("inject"),
        )

        def _fake_parallelize(m, **kwargs):
            calls.append("parallelize")
            return m

        monkeypatch.setattr("xorl.trainers.model_builder._parallelize", _fake_parallelize)

    @staticmethod
    def _build(**overrides):
        kwargs = {
            "config_path": "unused",
            "weights_path": "unused",
            "enable_lora": True,
            "lora_target_modules": ["q_proj"],
            "enable_mixed_precision": False,
            "enable_gradient_checkpointing": False,
        }
        kwargs.update(overrides)
        return build_training_model(**kwargs)

    def test_unfuse_precedes_injection_and_weight_loading(self, monkeypatch):
        calls = []
        self._patch(monkeypatch, _Unfusable(calls), calls)

        self._build(unfuse_for_lora=True)

        assert calls == ["unfuse", "inject", "parallelize"]

    def test_not_unfused_by_default(self, monkeypatch):
        """Opt-in only: ``model_runner`` shares this builder and must keep its layout."""
        calls = []
        self._patch(monkeypatch, _Unfusable(calls), calls)

        self._build()

        assert calls == ["inject", "parallelize"]

    def test_merge_qkv_false_does_not_double_unfuse(self, monkeypatch):
        """``merge_qkv=False`` unfuses attention per-layer without setting the
        ``_unfused_for_tp`` flag; running it after a full unfuse would be redundant
        at best and would re-split already-split projections at worst."""
        calls = []
        self._patch(monkeypatch, _Unfusable(calls), calls)

        self._build(unfuse_for_lora=True, merge_qkv=False)

        assert calls == ["unfuse", "inject", "parallelize"]
