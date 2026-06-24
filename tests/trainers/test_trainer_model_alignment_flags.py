from types import SimpleNamespace

import pytest
import torch.nn as nn

from xorl.trainers.trainer import Trainer


pytestmark = pytest.mark.cpu


class TinyModel(nn.Module):
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")


def _trainer_args():
    return SimpleNamespace(
        model=SimpleNamespace(
            config_path="unused-config",
            model_path="unused-weights",
            attn_implementation="eager",
            moe_implementation=None,
            ep_dispatch="alltoall",
            train_router=False,
            record_routing_weights=True,
            deepep_buffer_size_gb=2.0,
            deepep_num_sms=20,
            deepep_async_combine=False,
            router_fp32=False,
            lm_head_fp32=False,
            alltoall_combine_hidden_chunk_size=0,
            rmsnorm_mode="eager",
            activation_native=True,
            rope_native=True,
            attention_cast_bf16=True,
            sparse_mla_enabled=False,
            sparse_mla_backend="auto",
            flash_attention_deterministic=True,
            merge_qkv=True,
            freeze_router=False,
        ),
        train=SimpleNamespace(
            enable_mixed_precision=True,
            skip_param_upcast=False,
            init_device="meta",
            enable_fp8_training=False,
            enable_qarl=False,
            qarl_sync_format="fp8",
            qarl_calib_size=0,
            qarl_quant_sequence_length=None,
            qarl_calib_data=None,
        ),
        lora=SimpleNamespace(enable_lora=False, enable_qlora=False),
    )


def test_local_trainer_forwards_model_numeric_alignment_flags(monkeypatch):
    captured = {}

    def fake_build_foundation_model(**kwargs):
        captured.update(kwargs)
        return TinyModel()

    trainer = Trainer.__new__(Trainer)
    trainer.args = _trainer_args()

    monkeypatch.setattr("xorl.trainers.trainer.build_foundation_model", fake_build_foundation_model)
    monkeypatch.setattr("xorl.trainers.trainer.get_parallel_state", lambda: SimpleNamespace(tp_enabled=False))
    monkeypatch.setattr("xorl.trainers.trainer.helper.print_device_mem_info", lambda *args, **kwargs: None)

    trainer._build_model()

    assert captured["router_fp32"] is False
    assert captured["lm_head_fp32"] is False
    assert captured["activation_native"] is True
    assert captured["rope_native"] is True
    assert captured["attention_cast_bf16"] is True
