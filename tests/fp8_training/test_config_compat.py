from types import SimpleNamespace

import pytest
import torch.nn as nn

from xorl.fp8_training import (
    FP8Linear,
    UnsupportedFP8ConfigError,
    inject_fp8_training_into_model,
    normalize_fp8_training_config,
    resolve_fp8_bf16_layer_islands,
    summarize_fp8_training_model,
    validate_external_fp8_runtime_config,
    validate_fp8_blackwell_training_policy,
)


pytestmark = pytest.mark.cpu


class TinyDecoderStack(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                "q_proj": nn.Linear(8, 8),
                                "o_proj": nn.Linear(8, 8),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "gate_up_proj": nn.Linear(8, 16),
                                "down_proj": nn.Linear(16, 8),
                            }
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_pp_module_config(self):
        return {
            "layer_prefix": "model.layers",
            "num_layers": self.config.num_hidden_layers,
        }


def test_nemo_fp8_cfg_blockwise_alias_enables_native_fp8_training():
    normalized = normalize_fp8_training_config(
        {
            "enable_fp8_training": False,
            "fp8_cfg": {"enabled": True, "fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": False},
        }
    )

    assert normalized["enable_fp8_training"] is True


@pytest.mark.parametrize(
    "fp8_cfg, match",
    [
        ({"enabled": True, "fp8": "hybrid"}, "hybrid"),
        ({"enabled": True, "fp8_recipe": "tensorwise"}, "tensorwise"),
        ({"enabled": True, "fp8_recipe": "mxfp8"}, "MXFP8|mxfp8"),
        ({"enabled": True, "fp8_param": True}, "fp8_param"),
    ],
)
def test_nemo_fp8_cfg_rejects_transformer_engine_only_recipes(fp8_cfg, match):
    with pytest.raises(UnsupportedFP8ConfigError, match=match):
        normalize_fp8_training_config({"fp8_cfg": fp8_cfg})


def test_vllm_fp8_runtime_knobs_fail_before_silent_translation():
    with pytest.raises(UnsupportedFP8ConfigError, match="vLLM FP8 receiver"):
        validate_external_fp8_runtime_config({"generation": {"vllm_cfg": {"precision": "fp8"}}})

    with pytest.raises(UnsupportedFP8ConfigError, match="vLLM FP8 receiver"):
        validate_external_fp8_runtime_config({"generation": {"vllm_cfg": {"quantization": "fp8"}}})

    with pytest.raises(UnsupportedFP8ConfigError, match="num_first_layers_in_bf16"):
        validate_external_fp8_runtime_config({"generation": {"vllm_cfg": {"num_first_layers_in_bf16": 1}}})

    with pytest.raises(UnsupportedFP8ConfigError, match="num_last_layers_in_bf16"):
        validate_external_fp8_runtime_config({"generation": {"vllm_cfg": {"num_last_layers_in_bf16": 1}}})

    with pytest.raises(UnsupportedFP8ConfigError, match="quantization_ignored_layer_kws"):
        validate_external_fp8_runtime_config(
            {"generation": {"vllm_cfg": {"quantization_ignored_layer_kws": ["a_proj"]}}}
        )

    with pytest.raises(UnsupportedFP8ConfigError, match="vLLM DeepGEMM"):
        validate_external_fp8_runtime_config({"generation": {"vllm_cfg": {"use_deep_gemm": True}}})

    with pytest.raises(UnsupportedFP8ConfigError, match="pow2_weight_scaling_factors"):
        validate_external_fp8_runtime_config(
            {"generation": {"vllm_cfg": {"pow2_weight_scaling_factors": True}}}
        )

    with pytest.raises(UnsupportedFP8ConfigError, match="pow2_activation_scaling_factors"):
        validate_external_fp8_runtime_config(
            {"generation": {"vllm_cfg": {"pow2_activation_scaling_factors": True}}}
        )

    with pytest.raises(UnsupportedFP8ConfigError, match="receiver_kv_cache_dtype"):
        validate_external_fp8_runtime_config({"generation": {"vllm_cfg": {"kv_cache_dtype": "fp8_e4m3"}}})

    with pytest.raises(UnsupportedFP8ConfigError, match="receiver_kv_cache_dtype"):
        validate_external_fp8_runtime_config({"kv_cache_dtype": "fp8"})


@pytest.mark.parametrize(
    "config",
    [
        {"policy": {"quant_cfg": "FP8_DEFAULT_CFG"}},
        {"policy": {"generation": {"quant_cfg": {"format": "fp8_e4m3"}}}},
        {"generation": {"quant_cfg": "NVFP4_DEFAULT_CFG"}},
    ],
)
def test_nemo_modelopt_qarl_configs_fail_before_silent_translation(config):
    with pytest.raises(UnsupportedFP8ConfigError, match="ModelOpt QARL"):
        validate_external_fp8_runtime_config(config)


def test_blackwell_policy_rejects_native_fp8_training_without_override():
    with pytest.raises(UnsupportedFP8ConfigError, match="guarded on Blackwell"):
        validate_fp8_blackwell_training_policy(
            enable_fp8_training=True,
            allow_blackwell=False,
            device_name="NVIDIA GB200",
            capability=(10, 0),
        )


def test_blackwell_policy_rejects_override_without_validation_artifact():
    with pytest.raises(UnsupportedFP8ConfigError, match="requires fp8_training_blackwell_validation_artifact"):
        validate_fp8_blackwell_training_policy(
            enable_fp8_training=True,
            allow_blackwell=True,
            validation_artifact=None,
            device_name="NVIDIA B200",
            capability=(10, 0),
        )


def test_blackwell_policy_allows_explicit_override_with_artifact():
    validate_fp8_blackwell_training_policy(
        enable_fp8_training=True,
        allow_blackwell=True,
        validation_artifact="/tmp/fp8-blackwell-validation.json",
        device_name="NVIDIA GB200",
        capability=(10, 0),
    )


def test_resolve_fp8_bf16_layer_islands_covers_first_last_and_overlap():
    model = TinyDecoderStack(num_layers=4)

    assert resolve_fp8_bf16_layer_islands(model, num_first_layers_bf16=2) == [
        "model.layers.0.*",
        "model.layers.1.*",
    ]
    assert resolve_fp8_bf16_layer_islands(model, num_last_layers_bf16=2) == [
        "model.layers.2.*",
        "model.layers.3.*",
    ]
    assert resolve_fp8_bf16_layer_islands(model, num_first_layers_bf16=1, num_last_layers_bf16=2) == [
        "model.layers.0.*",
        "model.layers.2.*",
        "model.layers.3.*",
    ]
    assert resolve_fp8_bf16_layer_islands(model, num_first_layers_bf16=3, num_last_layers_bf16=3) == [
        "model.layers.0.*",
        "model.layers.1.*",
        "model.layers.2.*",
        "model.layers.3.*",
    ]


def test_resolve_fp8_bf16_layer_islands_rejects_too_large_and_nonstandard_layout():
    with pytest.raises(UnsupportedFP8ConfigError, match="exceeds model layer count"):
        resolve_fp8_bf16_layer_islands(TinyDecoderStack(num_layers=2), num_first_layers_bf16=3)

    with pytest.raises(UnsupportedFP8ConfigError, match="standard model.layers"):
        resolve_fp8_bf16_layer_islands(nn.Sequential(nn.Linear(8, 8)), num_first_layers_bf16=1)


def test_fp8_injection_keeps_generated_first_last_layer_islands_bf16():
    model = TinyDecoderStack(num_layers=4)

    replaced = inject_fp8_training_into_model(
        model,
        num_first_layers_bf16=1,
        num_last_layers_bf16=1,
    )

    assert replaced == 9
    assert isinstance(model.model.layers[0]["self_attn"]["q_proj"], nn.Linear)
    assert not isinstance(model.model.layers[0]["self_attn"]["q_proj"], FP8Linear)
    assert isinstance(model.model.layers[1]["self_attn"]["q_proj"], FP8Linear)
    assert isinstance(model.model.layers[2]["mlp"]["down_proj"], FP8Linear)
    assert isinstance(model.model.layers[3]["mlp"]["down_proj"], nn.Linear)
    assert not isinstance(model.model.layers[3]["mlp"]["down_proj"], FP8Linear)
    assert isinstance(model.lm_head, FP8Linear)

    summary = summarize_fp8_training_model(model)
    assert summary["bf16_layer_island_patterns"] == ["model.layers.0.*", "model.layers.3.*"]
    assert summary["bf16_layer_island_count"] == 2
