import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.qwen3_5_shared import (
    map_qwen3_5_linear_attention_weight,
    remap_linear_attention_params_for_inference,
)
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.layers.gated_deltanet import _sglang_compatible_beta_gate
from xorl.ops.linear_attention.modules.short_conv import ShortConvolution


pytestmark = pytest.mark.cpu


def _causal_depthwise_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = F.conv1d(
        x.transpose(1, 2),
        weight,
        padding=weight.shape[-1] - 1,
        groups=weight.shape[0],
    )
    y = y[:, :, : x.shape[1]]
    return F.silu(y.transpose(1, 2))


def test_qwen35_linear_attention_checkpoint_mapping_round_trips_to_fused_names():
    key_dim = 4
    value_dim = 6
    hidden_size = 5
    layer = 2

    fused_tensors = {
        f"model.layers.{layer}.linear_attn.in_proj_qkv.weight": torch.arange(
            (2 * key_dim + value_dim) * hidden_size, dtype=torch.float32
        ).reshape(2 * key_dim + value_dim, hidden_size),
        f"model.layers.{layer}.linear_attn.conv1d.weight": torch.arange(
            (2 * key_dim + value_dim) * 3, dtype=torch.float32
        ).reshape(2 * key_dim + value_dim, 1, 3),
        f"model.layers.{layer}.linear_attn.in_proj_z.weight": torch.full((value_dim, hidden_size), 1.0),
        f"model.layers.{layer}.linear_attn.in_proj_b.weight": torch.full((2, hidden_size), 2.0),
        f"model.layers.{layer}.linear_attn.in_proj_a.weight": torch.full((2, hidden_size), 3.0),
        f"model.layers.{layer}.linear_attn.out_proj.weight": torch.full((hidden_size, value_dim), 4.0),
        f"model.layers.{layer}.linear_attn.norm.weight": torch.full((value_dim,), 5.0),
        f"model.layers.{layer}.linear_attn.dt_bias": torch.full((2,), 6.0),
        f"model.layers.{layer}.linear_attn.A_log": torch.full((2,), 7.0),
    }

    split_buffer: list[tuple[str, torch.Tensor]] = []
    for name, tensor in fused_tensors.items():
        mapped = map_qwen3_5_linear_attention_weight(name, tensor, key_dim, value_dim)
        assert mapped is not None
        split_buffer.extend(mapped)

    split_by_name = dict(split_buffer)
    torch.testing.assert_close(
        split_by_name[f"model.layers.{layer}.linear_attn.q_proj.weight"],
        fused_tensors[f"model.layers.{layer}.linear_attn.in_proj_qkv.weight"][:key_dim],
    )
    torch.testing.assert_close(
        split_by_name[f"model.layers.{layer}.linear_attn.k_proj.weight"],
        fused_tensors[f"model.layers.{layer}.linear_attn.in_proj_qkv.weight"][key_dim : 2 * key_dim],
    )
    torch.testing.assert_close(
        split_by_name[f"model.layers.{layer}.linear_attn.v_proj.weight"],
        fused_tensors[f"model.layers.{layer}.linear_attn.in_proj_qkv.weight"][2 * key_dim :],
    )
    torch.testing.assert_close(
        split_by_name[f"model.layers.{layer}.linear_attn.q_conv1d.weight"],
        fused_tensors[f"model.layers.{layer}.linear_attn.conv1d.weight"][:key_dim],
    )
    torch.testing.assert_close(
        split_by_name[f"model.layers.{layer}.linear_attn.k_conv1d.weight"],
        fused_tensors[f"model.layers.{layer}.linear_attn.conv1d.weight"][key_dim : 2 * key_dim],
    )
    torch.testing.assert_close(
        split_by_name[f"model.layers.{layer}.linear_attn.v_conv1d.weight"],
        fused_tensors[f"model.layers.{layer}.linear_attn.conv1d.weight"][2 * key_dim :],
    )

    fused_again = dict(remap_linear_attention_params_for_inference(split_buffer))

    assert set(fused_again) == set(fused_tensors)
    for name, expected in fused_tensors.items():
        torch.testing.assert_close(fused_again[name], expected)


def test_qwen35_split_projection_and_short_conv_match_fused_packed_path():
    torch.manual_seed(0)
    key_dim = 4
    value_dim = 6
    conv_dim = 2 * key_dim + value_dim
    hidden_size = 5
    kernel_size = 3
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.long)

    hidden_states = torch.randn(1, 7, hidden_size)
    qkv_weight = torch.randn(conv_dim, hidden_size)
    conv_weight = torch.randn(conv_dim, 1, kernel_size)

    fused_projection = F.linear(hidden_states, qkv_weight)
    fused_segments = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        fused_segments.append(_causal_depthwise_conv(fused_projection[:, start:end], conv_weight))
    fused_output = torch.cat(fused_segments, dim=1)

    q_proj = F.linear(hidden_states, qkv_weight[:key_dim])
    k_proj = F.linear(hidden_states, qkv_weight[key_dim : 2 * key_dim])
    v_proj = F.linear(hidden_states, qkv_weight[2 * key_dim :])

    q_conv = ShortConvolution(key_dim, kernel_size, bias=False, activation="silu")
    k_conv = ShortConvolution(key_dim, kernel_size, bias=False, activation="silu")
    v_conv = ShortConvolution(value_dim, kernel_size, bias=False, activation="silu")
    with torch.no_grad():
        q_conv.weight.copy_(conv_weight[:key_dim])
        k_conv.weight.copy_(conv_weight[key_dim : 2 * key_dim])
        v_conv.weight.copy_(conv_weight[2 * key_dim :])

    q_out, _ = q_conv(q_proj, cu_seqlens=cu_seqlens)
    k_out, _ = k_conv(k_proj, cu_seqlens=cu_seqlens)
    v_out, _ = v_conv(v_proj, cu_seqlens=cu_seqlens)
    split_output = torch.cat([q_out, k_out, v_out], dim=-1)

    torch.testing.assert_close(split_output, fused_output, atol=1e-6, rtol=1e-6)


def test_gated_deltanet_keeps_decay_parameters_fp32_when_model_casts_to_bf16():
    module = GatedDeltaNet(
        hidden_size=8,
        expand_v=1.0,
        head_dim=4,
        num_heads=2,
        num_v_heads=2,
        conv_size=3,
    )

    module.to(torch.bfloat16)

    assert module.q_proj.weight.dtype is torch.bfloat16
    assert module.A_log.dtype is torch.float32
    assert module.dt_bias.dtype is torch.float32


def test_gated_deltanet_beta_gate_rounds_through_projection_dtype():
    b_input = torch.tensor([[-3.5, -0.25, 0.0, 2.0]], dtype=torch.bfloat16)

    beta = _sglang_compatible_beta_gate(b_input)

    expected = b_input.float().sigmoid().to(dtype=torch.bfloat16).float()
    torch.testing.assert_close(beta, expected, atol=0.0, rtol=0.0)
    assert beta.dtype is torch.float32
