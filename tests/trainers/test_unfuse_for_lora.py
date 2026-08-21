"""``unfuse_for_lora``: split fused projections before LoRA selects modules by name.

The ordering tests are the load-bearing ones. ``unfuse_for_tp`` allocates fresh
``nn.Linear``s without copying from the fused weight, so it has to run before LoRA
injection *and* before the checkpoint is read (which happens inside ``_parallelize``).
Getting that order wrong produces a model whose new projections are never filled.

Both build paths are covered: ``build_training_model`` (used by the inference-server
ModelRunner and by out-of-repo callers) and ``Trainer._build_model`` (used by
``xorl.cli.train``). They carry hand-duplicated copies of the same sequence, so each
needs its own guard.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from xorl.trainers.model_builder import build_training_model, maybe_unfuse_projections
from xorl.trainers.trainer import Trainer


pytestmark = [pytest.mark.cpu]


class _Unfusable(nn.Module):
    def __init__(self, calls: list[str] | None = None) -> None:
        super().__init__()
        self.calls = calls if calls is not None else []
        self.config = SimpleNamespace(num_experts=0, model_type="qwen3")
        self.config._resolved_numerical_program = {
            "attn_implementation": "eager",
            "router_fp32": False,
            "lm_head_fp32": False,
            "rmsnorm_mode": "native",
            "qwen35_rmsnorm_family": None,
            "activation_native": False,
            "rope_native": False,
            "rope_fp32_single_round": False,
            "attention_cast_bf16": False,
            "sparse_mla_enabled": False,
            "sparse_mla_backend": None,
        }
        self._no_split_modules = []
        self.proj = nn.Linear(4, 4, bias=False)

    def unfuse_for_tp(self):
        self.calls.append("unfuse")


class _NotUnfusable(nn.Module):
    pass


class _PlanWritingModel(_Unfusable):
    """Mirrors the real ``unfuse_for_tp``, which also rewrites ``base_model_tp_plan``."""

    def unfuse_for_tp(self):
        super().unfuse_for_tp()
        self.config.base_model_tp_plan = {"layers.*.self_attn.q_proj": "colwise"}


class TestMaybeUnfuseProjections:
    def test_noop_when_disabled(self):
        model = _Unfusable()

        maybe_unfuse_projections(model, unfuse_for_lora=False, enable_lora=True, enable_qlora=False)

        assert model.calls == []

    def test_unfuses_when_enabled(self):
        model = _Unfusable()

        maybe_unfuse_projections(model, unfuse_for_lora=True, enable_lora=True, enable_qlora=False)

        assert model.calls == ["unfuse"]

    def test_rejects_qlora(self):
        """Reject the QLoRA combination.

        QLoRA targets the fused names, so unfusing would leave q/k/v and gate/up both
        unquantized and unadapted while injection still reports success.
        """
        with pytest.raises(ValueError, match="not supported with QLoRA"):
            maybe_unfuse_projections(_Unfusable(), unfuse_for_lora=True, enable_lora=True, enable_qlora=True)

    def test_rejects_unfusing_without_lora(self):
        """Reject unfusing without LoRA.

        Splitting the GEMMs buys nothing if no adapter is going to use the names.
        """
        with pytest.raises(ValueError, match="requires enable_lora"):
            maybe_unfuse_projections(_Unfusable(), unfuse_for_lora=True, enable_lora=False, enable_qlora=False)

    def test_raises_rather_than_silently_skipping(self):
        """Raise on an architecture that cannot unfuse.

        A silent skip is the exact failure mode this work exists to remove: the targets
        would train unadapted while the config claims otherwise.
        """
        with pytest.raises(NotImplementedError, match="cannot unfuse"):
            maybe_unfuse_projections(_NotUnfusable(), unfuse_for_lora=True, enable_lora=True, enable_qlora=False)

    def test_leaves_no_tp_plan_on_the_config(self):
        """The Trainer holds a reference to this config and save_pretrained's it into
        every exported checkpoint, so a TP plan left behind by ``unfuse_for_tp`` would
        ship in config.json — and it is always spurious here, since TP + LoRA is rejected.
        """
        model = _PlanWritingModel()

        maybe_unfuse_projections(model, unfuse_for_lora=True, enable_lora=True, enable_qlora=False)

        assert "base_model_tp_plan" not in model.config.__dict__

    def test_restores_a_pre_existing_tp_plan(self):
        """A plan the config already carried is the caller's, not ours to drop."""
        model = _PlanWritingModel()
        original = {"layers.*.mlp.down_proj": "rowwise"}
        model.config.base_model_tp_plan = original

        maybe_unfuse_projections(model, unfuse_for_lora=True, enable_lora=True, enable_qlora=False)

        assert model.config.base_model_tp_plan == original

    def test_unsupported_architecture_is_fine_while_disabled(self):
        """The disabled check must precede the capability check: an architecture that
        cannot unfuse is only a problem when unfusing is actually requested.
        """
        model = _NotUnfusable()

        maybe_unfuse_projections(model, unfuse_for_lora=False, enable_lora=True, enable_qlora=False)

        assert not hasattr(model, "unfuse_for_tp")


def _patch_builder(monkeypatch, model, calls):
    monkeypatch.setattr("xorl.trainers.model_builder.build_foundation_model", lambda **kwargs: model)
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "xorl.trainers.model_builder.inject_lora_into_model",
        lambda *args, **kwargs: calls.append("inject"),
    )

    def _fake_parallelize(model_arg, **kwargs):
        calls.append("parallelize")
        return model_arg

    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", _fake_parallelize)


class TestBuildTrainingModelOrdering:
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
        _patch_builder(monkeypatch, _Unfusable(calls), calls)

        self._build(unfuse_for_lora=True)

        assert calls == ["unfuse", "inject", "parallelize"]

    def test_not_unfused_by_default(self, monkeypatch):
        """Stay fused unless asked.

        The inference-server ModelRunner shares this builder and must keep its fused
        layout, which weight sync canonicalizes against.
        """
        calls = []
        _patch_builder(monkeypatch, _Unfusable(calls), calls)

        self._build()

        assert calls == ["inject", "parallelize"]

    def test_merge_qkv_false_yields_to_unfuse_for_lora(self, monkeypatch):
        """With both set, only the full unfuse runs.

        ``merge_qkv=False`` unfuses attention per-layer, and running that after a full
        unfuse would raise AttributeError, since unfuse_for_tp reads the fused weight
        before deleting it.
        """
        calls = []
        _patch_builder(monkeypatch, _Unfusable(calls), calls)

        self._build(unfuse_for_lora=True, merge_qkv=False)

        assert calls == ["unfuse", "inject", "parallelize"]


class TestTrainerOrdering:
    """``Trainer._build_model`` does not go through ``build_training_model``; it carries
    its own copy of the unfuse -> inject -> parallelize sequence.
    """

    @staticmethod
    def _trainer(monkeypatch, calls, model):
        monkeypatch.setattr("xorl.trainers.trainer.build_foundation_model", lambda **kwargs: model)
        monkeypatch.setattr("xorl.trainers.trainer.helper.print_device_mem_info", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "xorl.trainers.trainer.maybe_upcast_trainable_adapter_params",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(Trainer, "_inject_lora", lambda self: calls.append("inject"))
        monkeypatch.setattr(Trainer, "_inject_qlora", lambda self: calls.append("inject_qlora"))

        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            model=SimpleNamespace(
                config_path="unused",
                model_path="unused",
                attn_implementation="eager",
                moe_implementation=None,
                moe_routing_weights_before_down=False,
                ep_dispatch="alltoall",
                train_router=False,
                record_routing_weights=False,
                deepep_buffer_size_gb=2.0,
                deepep_num_sms=20,
                deepep_async_combine=False,
                router_fp32=False,
                lm_head_fp32=False,
                alltoall_combine_hidden_chunk_size=0,
                rmsnorm_mode="native",
                qwen35_rmsnorm_family=None,
                activation_native=False,
                rope_native=False,
                rope_fp32_single_round=False,
                attention_cast_bf16=False,
                sparse_mla_enabled=False,
                sparse_mla_backend=None,
                flash_attention_deterministic=False,
                freeze_router=False,
                merge_qkv=True,
                encoders=None,
            ),
            lora=SimpleNamespace(
                enable_lora=True,
                enable_qlora=False,
                unfuse_for_lora=True,
                lora_rank=2,
                lora_alpha=2,
            ),
            train=SimpleNamespace(
                enable_mixed_precision=False,
                init_device="meta",
                enable_vision_encoder=False,
                skip_param_upcast=False,
                ce_mode="native",
                enable_fp8_training=False,
                enable_qarl=False,
                pipeline_parallel_virtual_stages=1,
                pipeline_parallel_input_weight=1,
                pipeline_parallel_output_weight=1,
                pipeline_parallel_num_layers_in_first_stage=None,
                pipeline_parallel_num_layers_in_last_stage=None,
            ),
        )
        trainer._causallm_loss_params = {}
        return trainer

    def test_unfuse_precedes_lora_injection(self, monkeypatch):
        calls = []
        trainer = self._trainer(monkeypatch, calls, _Unfusable(calls))

        trainer._build_model()

        assert calls == ["unfuse", "inject"]

    def test_merge_qkv_false_yields_to_unfuse_for_lora(self, monkeypatch):
        """The live path carries its own copy of the guard clause."""
        calls = []
        trainer = self._trainer(monkeypatch, calls, _Unfusable(calls))
        trainer.args.model.merge_qkv = False

        trainer._build_model()

        assert calls == ["unfuse", "inject"]

    def test_not_unfused_when_flag_is_off(self, monkeypatch):
        calls = []
        trainer = self._trainer(monkeypatch, calls, _Unfusable(calls))
        trainer.args.lora.unfuse_for_lora = False

        trainer._build_model()

        assert calls == ["inject"]
