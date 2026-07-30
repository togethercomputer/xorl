import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.fp8_training import (
    FP8Linear,
    clear_linear_error_profile,
    get_linear_error_profile,
    inject_fp8_training_into_model,
)


def _manual_padded_fp8_linear_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int,
    smoothquant_alpha: float | None = None,
    activation_amax_scale: float = 1.0,
    weight_amax_scale: float = 1.0,
) -> torch.Tensor:
    from xorl.fp8_training.linear import _apply_smoothquant, _pad_last_dim  # noqa: PLC0415
    from xorl.ops.quantize import (  # noqa: PLC0415
        block_fp8_dequantize,
        block_fp8_dequantize_gkn_rowwise,
        block_fp8_quantize,
        block_fp8_quantize_gkn_rowwise,
    )

    a_2d = a.reshape(-1, a.shape[-1])
    a_float = a_2d.float()
    b_float = b.float()
    if smoothquant_alpha is not None:
        a_float, b_float = _apply_smoothquant(a_float, b_float, smoothquant_alpha)
    a_padded = _pad_last_dim(a_float, block_size)
    b_padded = _pad_last_dim(b_float, block_size)
    a_fp8, a_scales = block_fp8_quantize(
        a_padded,
        block_size=block_size,
        amax_scale=activation_amax_scale,
    )
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(
        b_padded,
        block_size=block_size,
        amax_scale=weight_amax_scale,
    )
    a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=block_size)
    b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=block_size)
    return a_dequant @ b_dequant.T


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)
        self.gate = nn.Linear(16, 1, bias=False)
        self.lm_head = nn.Linear(32, 8, bias=False)


class TinyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                "qkv_proj": nn.Linear(16, 16),
                                "o_proj": nn.Linear(16, 16),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "gate_up_proj": nn.Linear(16, 32),
                                "down_proj": nn.Linear(32, 16),
                            }
                        ),
                    }
                )
            ]
        )
        self.lm_head = nn.Linear(16, 8, bias=False)


def test_inject_fp8_training_replaces_all_linears_by_default_and_preserves_parameters():
    model = TinyModel()
    original_weight = model.proj.weight
    replaced = inject_fp8_training_into_model(model)

    assert replaced == 3
    assert isinstance(model.proj, FP8Linear)
    assert model.proj.fp8_output_dtype == "input"
    assert model.proj.weight is original_weight
    assert isinstance(model.gate, FP8Linear)
    assert isinstance(model.lm_head, FP8Linear)
    assert model.lm_head.fp8_output_dtype == "float32"
    assert "proj.weight" in model.state_dict()


def test_inject_fp8_training_tags_replaced_modules_with_fqns():
    model = TinyTransformerBlock()

    inject_fp8_training_into_model(model)

    assert model.model.layers[0]["self_attn"]["qkv_proj"].fp8_module_name == "model.layers.0.self_attn.qkv_proj"
    assert model.model.layers[0]["mlp"]["down_proj"].fp8_module_name == "model.layers.0.mlp.down_proj"
    assert model.lm_head.fp8_module_name == "lm_head"


def test_inject_fp8_training_threads_amax_scale_recipe():
    model = TinyModel()

    inject_fp8_training_into_model(model, activation_amax_scale=0.875, weight_amax_scale=1.125)

    assert model.proj.fp8_activation_amax_scale == 0.875
    assert model.proj.fp8_weight_amax_scale == 1.125
    assert model.lm_head.fp8_activation_amax_scale == 0.875
    assert model.lm_head.fp8_weight_amax_scale == 1.125


def test_inject_fp8_training_applies_fqn_module_recipe_overrides():
    model = TinyTransformerBlock()

    inject_fp8_training_into_model(
        model,
        block_size=64,
        smoothquant_alpha=0.4,
        module_overrides={
            "model.layers.*.self_attn.o_proj": {
                "block_size": 32,
                "smoothquant_alpha": 0.25,
                "correction_mode": "activation2",
            },
            "lm_head": {
                "activation_amax_scale": 1.125,
                "weight_amax_scale": 0.875,
                "correction_mode": "full",
            },
        },
    )

    qkv = model.model.layers[0]["self_attn"]["qkv_proj"]
    o_proj = model.model.layers[0]["self_attn"]["o_proj"]
    assert qkv.fp8_block_size == 64
    assert qkv.fp8_smoothquant_alpha == 0.4
    assert o_proj.fp8_block_size == 32
    assert o_proj.fp8_smoothquant_alpha == 0.25
    assert o_proj.fp8_correction_mode == "activation2"
    assert model.lm_head.fp8_activation_amax_scale == 1.125
    assert model.lm_head.fp8_weight_amax_scale == 0.875
    assert model.lm_head.fp8_correction_mode == "full"
    assert model.lm_head.fp8_output_dtype == "float32"


def test_inject_fp8_training_rejects_unknown_module_recipe_override_key():
    model = TinyModel()

    with pytest.raises(ValueError, match="Unsupported FP8 module override key"):
        inject_fp8_training_into_model(
            model,
            module_overrides={"proj": {"unknown": 1}},
        )


def test_inject_fp8_training_can_keep_explicit_exclusions_in_bf16():
    model = TinyModel()

    replaced = inject_fp8_training_into_model(model, exclude_modules=["gate", "lm_head"])

    assert replaced == 1
    assert isinstance(model.proj, FP8Linear)
    assert isinstance(model.gate, nn.Linear)
    assert not isinstance(model.gate, FP8Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, FP8Linear)


def test_inject_fp8_training_honors_fqn_glob_exclusions():
    model = TinyTransformerBlock()

    replaced = inject_fp8_training_into_model(model, exclude_modules=["model.layers.*.self_attn.*"])

    assert replaced == 3
    assert isinstance(model.model.layers[0]["self_attn"]["qkv_proj"], nn.Linear)
    assert not isinstance(model.model.layers[0]["self_attn"]["qkv_proj"], FP8Linear)
    assert isinstance(model.model.layers[0]["self_attn"]["o_proj"], nn.Linear)
    assert not isinstance(model.model.layers[0]["self_attn"]["o_proj"], FP8Linear)
    assert isinstance(model.model.layers[0]["mlp"]["gate_up_proj"], FP8Linear)
    assert isinstance(model.model.layers[0]["mlp"]["down_proj"], FP8Linear)
    assert isinstance(model.lm_head, FP8Linear)


def test_fp8_linear_cpu_fallback_matches_linear():
    torch.manual_seed(0)
    linear = nn.Linear(16, 32, dtype=torch.float32)
    fp8 = FP8Linear.from_linear(linear)
    x = torch.randn(4, 16)

    got = fp8(x)
    expected = F.linear(x, linear.weight, linear.bias)

    assert fp8.last_forward_used_fp8 is False
    assert torch.allclose(got, expected)


def test_fp8_linear_cpu_fallback_honors_float32_output_dtype():
    linear = nn.Linear(16, 32, dtype=torch.bfloat16)
    fp8 = FP8Linear.from_linear(linear, output_dtype="float32")
    x = torch.randn(4, 16, dtype=torch.bfloat16)

    got = fp8(x)

    assert fp8.last_forward_used_fp8 is False
    assert got.dtype == torch.float32


def test_fp8_linear_can_fail_fast_without_fallback():
    linear = nn.Linear(16, 32)
    fp8 = FP8Linear.from_linear(linear, allow_bf16_fallback=False)

    with pytest.raises(RuntimeError, match="cannot use FP8 compute"):
        fp8(torch.randn(2, 16))


def test_fp8_linear_error_profiler_records_sampled_module_stats(monkeypatch, tmp_path):
    from xorl.fp8_training.profiler import record_linear_error, write_linear_error_profile  # noqa: PLC0415

    clear_linear_error_profile()
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_MAX_CALLS_PER_MODULE", "1")
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS", "2")
    output_path = tmp_path / "fp8-profile.json"
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_OUTPUT", str(output_path))
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, dtype=torch.float32)
    fp8 = FP8Linear.from_linear(linear)
    fp8.fp8_module_name = "tiny.proj"
    x = torch.randn(5, 4)
    reference = F.linear(x, fp8.weight, fp8.bias)
    out = reference + 0.25

    record_linear_error(fp8, x, out)
    record_linear_error(fp8, x, out + 10.0)

    assert json.loads(output_path.read_text())["tiny.proj"]["sampled_calls"] == 1
    profile = get_linear_error_profile()
    assert set(profile) == {"tiny.proj"}
    stats = profile["tiny.proj"]
    assert stats["calls"] == 2
    assert stats["sampled_calls"] == 1
    assert stats["elements"] == 6
    assert stats["input_shape"] == [5, 4]
    assert stats["output_shape"] == [5, 3]
    assert stats["weight_shape"] == [3, 4]
    assert stats["sampled_row_indices"] == [0, 1]
    assert stats["sampled_call_summaries"] == [
        {
            "call_index": 1,
            "row_indices": [0, 1],
            "input_shape": [5, 4],
            "output_shape": [2, 3],
            "mean_abs_error": pytest.approx(0.25),
            "mean_rel_error": pytest.approx(stats["sampled_call_summaries"][0]["mean_rel_error"]),
            "rms_error": pytest.approx(0.25),
            "mean_reference_abs": pytest.approx(stats["sampled_call_summaries"][0]["mean_reference_abs"]),
            "mean_output_abs": pytest.approx(stats["sampled_call_summaries"][0]["mean_output_abs"]),
            "max_abs_error": pytest.approx(0.25),
            "max_rel_error": pytest.approx(stats["sampled_call_summaries"][0]["max_rel_error"]),
        }
    ]
    assert stats["mean_abs_error"] == pytest.approx(0.25)
    assert stats["max_abs_error"] == pytest.approx(0.25)

    write_linear_error_profile(output_path)
    assert json.loads(output_path.read_text())["tiny.proj"]["calls"] == 2
    clear_linear_error_profile()


def test_fp8_linear_error_profiler_samples_explicit_flattened_rows(monkeypatch):
    from xorl.fp8_training.profiler import record_linear_error  # noqa: PLC0415

    clear_linear_error_profile()
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS", "3")
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES", "1,4,99,-1,4")

    linear = nn.Linear(4, 3, dtype=torch.float32)
    fp8 = FP8Linear.from_linear(linear)
    fp8.fp8_module_name = "tiny.row_targeted"
    x = torch.randn(3, 2, 4)
    reference = F.linear(x, fp8.weight, fp8.bias)
    per_row_error = torch.arange(6, dtype=torch.float32).view(3, 2, 1)
    out = reference + per_row_error

    record_linear_error(fp8, x, out)

    stats = get_linear_error_profile()["tiny.row_targeted"]
    assert stats["sampled_calls"] == 1
    assert stats["elements"] == 9
    assert stats["sampled_row_indices"] == [1, 4, 5]
    assert stats["sampled_call_summaries"][0]["call_index"] == 1
    assert stats["sampled_call_summaries"][0]["row_indices"] == [1, 4, 5]
    assert stats["sampled_call_summaries"][0]["input_shape"] == [3, 2, 4]
    assert stats["sampled_call_summaries"][0]["output_shape"] == [3, 3]
    assert stats["input_shape"] == [3, 2, 4]
    assert stats["output_shape"] == [3, 2, 3]
    assert stats["mean_abs_error"] == pytest.approx((1 + 4 + 5) / 3)
    assert stats["max_abs_error"] == pytest.approx(5.0)
    clear_linear_error_profile()


def test_fp8_linear_error_profiler_samples_module_specific_call_rows(monkeypatch):
    from xorl.fp8_training.profiler import record_linear_error  # noqa: PLC0415

    clear_linear_error_profile()
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS", "2")
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_ROW_INDICES", "tiny.call@3=2;*=0,1")

    linear = nn.Linear(4, 3, dtype=torch.float32)
    fp8 = FP8Linear.from_linear(linear)
    fp8.fp8_module_name = "tiny.call"
    x = torch.randn(4, 4)
    reference = F.linear(x, fp8.weight, fp8.bias)
    per_row_error = torch.arange(4, dtype=torch.float32).view(4, 1)

    record_linear_error(fp8, x, reference + per_row_error)
    record_linear_error(fp8, x, reference + per_row_error)
    assert get_linear_error_profile()["tiny.call"]["sampled_calls"] == 0

    record_linear_error(fp8, x, reference + per_row_error)
    stats = get_linear_error_profile()["tiny.call"]
    assert stats["calls"] == 3
    assert stats["sampled_calls"] == 1
    assert stats["sampled_row_indices"] == [2]
    assert stats["sampled_call_summaries"][0]["call_index"] == 3
    assert stats["mean_abs_error"] == pytest.approx(2.0)

    other = FP8Linear.from_linear(linear)
    other.fp8_module_name = "other.call"
    record_linear_error(other, x, reference + per_row_error)
    other_stats = get_linear_error_profile()["other.call"]
    assert other_stats["sampled_row_indices"] == [0, 1]
    assert other_stats["sampled_call_summaries"][0]["call_index"] == 1
    clear_linear_error_profile()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_linear_error_profiler_records_cuda_operand_breakdown(monkeypatch):
    clear_linear_error_profile()
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE", "1")
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_MAX_CALLS_PER_MODULE", "1")
    monkeypatch.setenv("XORL_FP8_LINEAR_ERROR_PROFILE_MAX_ROWS", "4")

    torch.manual_seed(0)
    linear = nn.Linear(128, 64, device="cuda", dtype=torch.bfloat16)
    fp8 = FP8Linear.from_linear(
        linear,
        block_size=64,
        smoothquant_alpha=0.4,
        output_dtype="float32",
        allow_bf16_fallback=False,
    )
    fp8.fp8_module_name = "cuda.proj"
    x = (torch.randn(6, 128, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    out = fp8(x)
    torch.cuda.synchronize()

    assert fp8.last_forward_used_fp8 is True
    assert out.shape == (6, 64)
    assert out.dtype == torch.float32
    stats = get_linear_error_profile()["cuda.proj"]
    assert stats["sampled_calls"] == 1
    assert stats["elements"] == 4 * 64
    for metric in (
        "activation_quant",
        "weight_quant",
        "operand_quant",
        "kernel_accum",
        "output_cast",
        "kernel_output",
    ):
        assert stats[f"{metric}_sampled_calls"] == 1
        assert torch.isfinite(torch.tensor(stats[f"{metric}_mean_abs_error"]))
        assert torch.isfinite(torch.tensor(stats[f"{metric}_max_abs_error"]))
        assert stats[f"{metric}_mean_abs_error"] >= 0.0
        assert stats[f"{metric}_max_abs_error"] >= 0.0
    assert stats["output_cast_max_abs_error"] < 1e-4
    clear_linear_error_profile()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_size", [64, 128])
def test_block_fp8_gemm_matches_explicit_dequantized_reference(block_size):
    from xorl.ops.quantize import (  # noqa: PLC0415
        block_fp8_dequantize,
        block_fp8_dequantize_gkn,
        block_fp8_gemm,
        block_fp8_quantize,
        block_fp8_quantize_gkn,
    )

    torch.manual_seed(0)
    a = (torch.randn(17, 256, device="cuda", dtype=torch.float32) * 0.25).contiguous()
    b = (torch.randn(193, 256, device="cuda", dtype=torch.float32) * 0.25).contiguous()

    a_fp8, a_scales = block_fp8_quantize(a, block_size=block_size)
    b_fp8, b_scales = block_fp8_quantize_gkn(b, block_size=block_size)

    got = block_fp8_gemm(a_fp8, a_scales, b_fp8, b_scales, block_size=block_size)
    a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=block_size)
    b_dequant = block_fp8_dequantize_gkn(b_fp8, b_scales, block_size=block_size)
    expected = a_dequant @ b_dequant.T

    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_size", [64, 128])
def test_block_fp8_gemm_rowwise_weight_scales_match_explicit_dequantized_reference(block_size):
    from xorl.ops.quantize import (  # noqa: PLC0415
        block_fp8_dequantize,
        block_fp8_dequantize_gkn_rowwise,
        block_fp8_gemm,
        block_fp8_quantize,
        block_fp8_quantize_gkn_rowwise,
    )

    torch.manual_seed(3)
    a = (torch.randn(17, 256, device="cuda", dtype=torch.float32) * 0.25).contiguous()
    b = (torch.randn(193, 256, device="cuda", dtype=torch.float32) * 0.25).contiguous()

    a_fp8, a_scales = block_fp8_quantize(a, block_size=block_size)
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(b, block_size=block_size)

    got = block_fp8_gemm(
        a_fp8,
        a_scales,
        b_fp8,
        b_scales,
        block_size=block_size,
        weight_scale_layout="row",
    )
    a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=block_size)
    b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=block_size)
    expected = a_dequant @ b_dequant.T

    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_fp8_gemm_torch_scaled_mm_backend_matches_explicit_dequantized_reference():
    from xorl.ops.quantize import (  # noqa: PLC0415
        block_fp8_dequantize,
        block_fp8_dequantize_gkn_rowwise,
        block_fp8_gemm,
        block_fp8_quantize,
        block_fp8_quantize_gkn_rowwise,
    )

    torch.manual_seed(4)
    a = (torch.randn(128, 256, device="cuda", dtype=torch.float32) * 0.25).contiguous()
    b = (torch.randn(192, 256, device="cuda", dtype=torch.float32) * 0.25).contiguous()

    a_fp8, a_scales = block_fp8_quantize(a, block_size=128)
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(b, block_size=128)

    got = block_fp8_gemm(
        a_fp8,
        a_scales,
        b_fp8,
        b_scales,
        block_size=128,
        weight_scale_layout="row",
        backend="torch_scaled_mm",
    )
    a_dequant = block_fp8_dequantize(a_fp8, a_scales, block_size=128)
    b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=128)
    expected = a_dequant @ b_dequant.T

    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("block_size", "smoothquant_alpha", "activation_amax_scale", "weight_amax_scale"),
    [
        (64, None, 1.0, 1.0),
        (128, None, 1.0, 1.0),
        (128, 0.5, 1.0, 1.0),
        (64, 0.4, 0.875, 1.0),
        (64, 0.4, 1.0, 1.125),
    ],
)
def test_fp8_linear_matmul_padding_matches_explicit_dequantized_reference(
    block_size,
    smoothquant_alpha,
    activation_amax_scale,
    weight_amax_scale,
):
    from xorl.fp8_training.linear import _fp8_matmul  # noqa: PLC0415

    torch.manual_seed(1)
    a = (torch.randn(2, 7, 160, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
    b = (torch.randn(49, 160, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()

    got = _fp8_matmul(
        a,
        b,
        block_size=block_size,
        smoothquant_alpha=smoothquant_alpha,
        activation_amax_scale=activation_amax_scale,
        weight_amax_scale=weight_amax_scale,
    )
    expected = _manual_padded_fp8_linear_reference(
        a,
        b,
        block_size,
        smoothquant_alpha=smoothquant_alpha,
        activation_amax_scale=activation_amax_scale,
        weight_amax_scale=weight_amax_scale,
    ).reshape(
        2,
        7,
        49,
    )

    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_linear_full_residual_correction_reduces_forward_error():
    from xorl.fp8_training.linear import _fp8_matmul  # noqa: PLC0415

    torch.manual_seed(11)
    a = (torch.randn(19, 160, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
    b = (torch.randn(97, 160, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
    expected = a.float() @ b.float().T

    base = _fp8_matmul(
        a,
        b,
        block_size=64,
        smoothquant_alpha=0.4,
        correction_mode="none",
    )
    corrected = _fp8_matmul(
        a,
        b,
        block_size=64,
        smoothquant_alpha=0.4,
        correction_mode="full",
    )
    base_error = (base - expected).abs().mean()
    corrected_error = (corrected - expected).abs().mean()

    assert torch.isfinite(corrected).all()
    assert corrected_error < base_error * 0.25


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_linear_activation2_reduces_activation_quantization_error():
    from xorl.fp8_training.linear import _apply_smoothquant, _fp8_matmul, _pad_last_dim  # noqa: PLC0415
    from xorl.ops.quantize import block_fp8_dequantize_gkn_rowwise, block_fp8_quantize_gkn_rowwise  # noqa: PLC0415

    torch.manual_seed(12)
    a = (torch.randn(23, 160, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
    b = (torch.randn(83, 160, device="cuda", dtype=torch.bfloat16) * 0.7).contiguous()
    block_size = 64
    a_float, b_float = _apply_smoothquant(a.float(), b.float(), 0.4)
    a_padded = _pad_last_dim(a_float, block_size)
    b_padded = _pad_last_dim(b_float, block_size)
    b_fp8, b_scales = block_fp8_quantize_gkn_rowwise(b_padded, block_size=block_size)
    b_dequant = block_fp8_dequantize_gkn_rowwise(b_fp8, b_scales, block_size=block_size)
    expected_weight_quantized = a_padded @ b_dequant.T

    activation = _fp8_matmul(
        a,
        b,
        block_size=block_size,
        smoothquant_alpha=0.4,
        correction_mode="activation",
    )
    activation2 = _fp8_matmul(
        a,
        b,
        block_size=block_size,
        smoothquant_alpha=0.4,
        correction_mode="activation2",
    )
    activation_error = (activation - expected_weight_quantized).abs().mean()
    activation2_error = (activation2 - expected_weight_quantized).abs().mean()

    assert torch.isfinite(activation2).all()
    assert activation2_error < activation_error * 0.5


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_linear_cuda_train_step_updates_master_weight():
    torch.manual_seed(0)
    linear = nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16)
    fp8 = FP8Linear.from_linear(linear, backward_mode="fp8", allow_bf16_fallback=False)
    opt = torch.optim.AdamW(fp8.parameters(), lr=1e-2)
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    before = fp8.weight.detach().clone()

    out = fp8(x)
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    opt.step()

    assert fp8.last_forward_used_fp8 is True
    assert fp8.weight.grad is not None
    assert torch.isfinite(fp8.weight.grad.float()).all()
    assert not torch.equal(fp8.weight.detach(), before)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_linear_cuda_float32_output_dtype_still_uses_fp8():
    linear = nn.Linear(128, 128, device="cuda", dtype=torch.bfloat16)
    fp8 = FP8Linear.from_linear(linear, output_dtype="float32", allow_bf16_fallback=False)
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)

    out = fp8(x)

    assert fp8.last_forward_used_fp8 is True
    assert out.dtype == torch.float32
