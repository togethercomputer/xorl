from types import SimpleNamespace

import pytest

from xorl.distributed.torch_parallelize import build_parallelize_model
from xorl.models.auto import build_foundation_model
from xorl.models.transformers.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from xorl.models.transformers.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from xorl.models.transformers.deepseek_v3.support import validate_deepseek_v3_tensor_parallelism
from xorl.trainers.model_builder import build_training_model


pytestmark = [pytest.mark.cpu]


def _tiny_config() -> DeepseekV3Config:
    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
    )
    config._attn_implementation = "eager"
    config._moe_implementation = "eager"
    config._activation_native = True
    return config


def _patch_tiny_model(monkeypatch):
    monkeypatch.setattr(
        "xorl.trainers.model_builder.build_foundation_model",
        lambda **kwargs: DeepseekV3ForCausalLM(_tiny_config()),
    )
    monkeypatch.setattr("xorl.trainers.model_builder.helper.print_device_mem_info", lambda *args, **kwargs: None)


def test_build_foundation_model_rejects_train_router_for_deepseek():
    with pytest.raises(ValueError, match="train_router=True"):
        build_foundation_model(_tiny_config(), train_router=True)


@pytest.mark.parametrize(
    ("override_kwargs", "match"),
    [
        ({"freeze_router": False}, "requires freeze_router=True"),
        ({"freeze_router": True, "enable_qlora": True}, "enable_qlora=True"),
        ({"freeze_router": True, "merge_qkv": False}, "merge_qkv=False"),
    ],
)
def test_build_training_model_rejects_unsupported_deepseek_modes(monkeypatch, override_kwargs, match):
    _patch_tiny_model(monkeypatch)

    kwargs = {
        "config_path": "unused",
        "weights_path": "unused",
        "freeze_router": True,
        "enable_mixed_precision": False,
        "enable_gradient_checkpointing": False,
    }
    kwargs.update(override_kwargs)

    with pytest.raises(ValueError, match=match):
        build_training_model(**kwargs)


def test_build_training_model_freezes_router_when_requested(monkeypatch):
    _patch_tiny_model(monkeypatch)
    monkeypatch.setattr("xorl.trainers.model_builder._parallelize", lambda model, **kwargs: model)

    result = build_training_model(
        config_path="unused",
        weights_path="unused",
        freeze_router=True,
        enable_mixed_precision=False,
        enable_gradient_checkpointing=False,
    )

    router_params = [param for name, param in result.model.named_parameters() if ".gate.weight" in name]
    assert router_params
    assert all(param.requires_grad is False for param in router_params)
    assert result.model.model.layers[0].self_attn.q_a_proj.weight.requires_grad is True


def test_validate_deepseek_v3_tensor_parallelism_rejects_tp(monkeypatch):
    monkeypatch.setattr(
        "xorl.models.transformers.deepseek_v3.support.get_parallel_state",
        lambda: SimpleNamespace(tp_enabled=True),
    )

    with pytest.raises(ValueError, match="tensor parallelism is not supported yet"):
        validate_deepseek_v3_tensor_parallelism(_tiny_config())


def test_build_parallelize_model_preserves_deepseek_tp_guard(monkeypatch):
    model = DeepseekV3ForCausalLM(_tiny_config())
    monkeypatch.setattr(
        "xorl.distributed.torch_parallelize.get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=False, tp_enabled=True),
    )
    monkeypatch.setattr(
        "xorl.models.transformers.deepseek_v3.support.get_parallel_state",
        lambda: SimpleNamespace(tp_enabled=True),
    )

    with pytest.raises(ValueError, match="tensor parallelism is not supported yet"):
        build_parallelize_model(
            model,
            init_device="cuda",
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
        )
