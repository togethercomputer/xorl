import pytest

from xorl.server.weight_sync.quantization_config import (
    SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY,
    UnsupportedSyncQuantizationError,
    normalize_sync_quantization_config,
)


def test_none_quantization_is_bf16_noop():
    assert normalize_sync_quantization_config(None) is None


@pytest.mark.parametrize("method", ["bf16", "bfloat16", "none", "null"])
def test_explicit_no_quantization_aliases_disable_quantization(method):
    assert normalize_sync_quantization_config({"quant_method": method}) is None


def test_fp8_quantization_config_is_normalized():
    config = normalize_sync_quantization_config(
        {
            "quant_method": "FP8",
            "fmt": "E4M3",
            "activation_scheme": "Dynamic",
            "weight_block_size": [64],
            "modules_to_not_convert": ["lm_head"],
        }
    )

    assert config == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [64, 64],
        "modules_to_not_convert": ["lm_head"],
    }


def test_fp8_quantization_defaults_to_slime_dynamic_block_contract():
    config = normalize_sync_quantization_config({"quant_method": "fp8"})

    assert config == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }


def test_modules_to_not_convert_normalizes_weight_suffixes_and_duplicates():
    config = normalize_sync_quantization_config(
        {
            "quant_method": "fp8",
            "modules_to_not_convert": [
                " lm_head.weight ",
                "lm_head",
                "model.layers.0.linear_attn.in_proj_a.weight",
                "",
            ],
        }
    )

    assert config is not None
    assert config["modules_to_not_convert"] == ["lm_head", "model.layers.0.linear_attn.in_proj_a"]


@pytest.mark.parametrize("method", ["int4", "compressed-tensors", "awq", "qat"])
def test_unsupported_sync_quantization_methods_fail_explicitly(method):
    with pytest.raises(UnsupportedSyncQuantizationError, match="Online weight sync currently supports only"):
        normalize_sync_quantization_config({"quant_method": method})


def test_invalid_fp8_format_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="Unsupported .*fmt"):
        normalize_sync_quantization_config({"quant_method": "fp8", "fmt": "ue8m0"})


def test_unsupported_fp8_e5m2_format_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="E4M3"):
        normalize_sync_quantization_config({"quant_method": "fp8", "fmt": "e5m2"})


def test_unsupported_fp8_activation_scheme_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="activation_scheme"):
        normalize_sync_quantization_config({"quant_method": "fp8", "activation_scheme": "static"})


def test_null_fp8_activation_scheme_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="activation_scheme"):
        normalize_sync_quantization_config({"quant_method": "fp8", "activation_scheme": None})


def test_unsupported_fp8_ue8m0_scale_format_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="UE8M0 scale storage"):
        normalize_sync_quantization_config({"quant_method": "fp8", "scale_fmt": "ue8m0"})


def test_invalid_modules_to_not_convert_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="modules_to_not_convert"):
        normalize_sync_quantization_config({"quant_method": "fp8", "modules_to_not_convert": "lm_head"})


def test_internal_unsupported_quantization_marker_fails_explicitly():
    with pytest.raises(UnsupportedSyncQuantizationError, match="MTP/speculative low-precision sync"):
        normalize_sync_quantization_config(
            {
                "quant_method": "fp8",
                SYNC_QUANTIZATION_UNSUPPORTED_REASON_KEY: "MTP/speculative low-precision sync is not implemented.",
            },
            context="sync_inference_weights.quantization",
        )
