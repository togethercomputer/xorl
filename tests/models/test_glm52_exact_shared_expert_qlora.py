from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    GLM52_EXACT_TP16_SHARED_EXPERT_QLORA_CONTRACT_VERSION,
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)


def _pattern(
    shape: tuple[int, ...],
    *,
    modulus: int,
    center: int,
    divisor: int,
    device: torch.device,
) -> torch.Tensor:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return (
        torch.arange(elements, dtype=torch.float32, device=device)
        .remainder_(modulus)
        .sub_(center)
        .div_(divisor)
        .reshape(shape)
    )


def _fill_factors(module: Glm52ExactTP16SharedExpertBlockFP8QLoRA) -> None:
    device = module.gate_proj.lora_A.device
    with torch.no_grad():
        module.gate_proj.lora_A.copy_(_pattern((1, 6144), modulus=37, center=18, divisor=1024, device=device))
        module.gate_proj.lora_B.copy_(_pattern((2048, 1), modulus=47, center=23, divisor=2048, device=device))
        module.up_proj.lora_A.copy_(_pattern((1, 6144), modulus=43, center=21, divisor=1536, device=device))
        module.up_proj.lora_B.copy_(_pattern((2048, 1), modulus=53, center=26, divisor=1792, device=device))
        module.down_proj.lora_A.copy_(_pattern((1, 2048), modulus=59, center=29, divisor=2304, device=device))
        module.down_proj.lora_B.copy_(_pattern((6144, 1), modulus=61, center=30, divisor=2560, device=device))


def test_shared_expert_construction_and_runtime_admission_policy() -> None:
    for kwargs, message in (
        ({"hidden_size": 4096}, "hidden_size=6144"),
        ({"intermediate_size": 4096}, "intermediate_size=2048"),
        ({"tp_size": 8}, "requires TP16"),
        ({"r": 0}, "positive integer rank"),
        ({"lora_alpha": 0}, "positive integer alpha"),
        ({"bias": True}, "bias-free"),
        ({"enable_aqn": True}, "rejects adaptive quantization noise"),
    ):
        with pytest.raises(ValueError, match=message):
            Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta", **kwargs)

    module = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="cpu")
    with pytest.raises(TypeError, match="contributor_ordinal must be an integer"):
        module(torch.zeros(1, 6144, dtype=torch.bfloat16), contributor_ordinal=True)
    with pytest.raises(ValueError, match=r"must be in \[0, 16\)"):
        module(torch.zeros(1, 6144, dtype=torch.bfloat16), contributor_ordinal=16)
    with pytest.raises(TypeError, match="requires BF16 activations"):
        module(torch.zeros(1, 6144), contributor_ordinal=0)
    with pytest.raises(ValueError, match="input width"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16), contributor_ordinal=0)
    with pytest.raises(ValueError, match="contiguous sampler-layout"):
        module(torch.zeros(6144, 2, dtype=torch.bfloat16).transpose(0, 1), contributor_ordinal=0)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module(torch.zeros(1, 6144, dtype=torch.bfloat16), contributor_ordinal=0)
    with pytest.raises(RuntimeError, match="cannot run independently"):
        module.gate_proj(torch.zeros(1, 6144, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="cannot bypass active LoRA"):
        module.gate_proj.forward_partition(
            torch.zeros(1, 6144, dtype=torch.bfloat16),
            output_range=(0, 128),
        )

    module.tp_size = 8
    with pytest.raises(RuntimeError, match="runtime contract was mutated"):
        module(torch.zeros(1, 6144, dtype=torch.bfloat16), contributor_ordinal=0)
    module.tp_size = 16
    module.gate_proj.lora_A = nn.Parameter(module.gate_proj.lora_A.to(torch.bfloat16))
    with pytest.raises(TypeError, match="gate_proj.lora_A must remain FP32"):
        module(torch.zeros(1, 6144, dtype=torch.bfloat16), contributor_ordinal=0)

    _assert_shared_expert_logical_and_checkpoint_state_policy()
    _assert_shared_expert_native_base_views_use_output_rows_and_input_columns()


def _assert_shared_expert_logical_and_checkpoint_state_policy() -> None:
    module = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="cpu")
    assert module.contract_version == GLM52_EXACT_TP16_SHARED_EXPERT_QLORA_CONTRACT_VERSION
    parameters = dict(module.named_parameters())
    expected_shapes = {
        "gate_proj.lora_A": (1, 6144),
        "gate_proj.lora_B": (2048, 1),
        "up_proj.lora_A": (1, 6144),
        "up_proj.lora_B": (2048, 1),
        "down_proj.lora_A": (1, 2048),
        "down_proj.lora_B": (6144, 1),
    }
    assert module.logical_factor_names == tuple(expected_shapes)
    identities = {name: id(parameters[name]) for name in expected_shapes}
    for name, shape in expected_shapes.items():
        assert parameters[name].dtype is torch.float32
        assert tuple(parameters[name].shape) == shape
    assert not any(
        parameter.requires_grad
        for name, parameter in parameters.items()
        if name.endswith(("packed_weight_f32", "weight_scale_inv"))
    )

    module.to(dtype=torch.bfloat16)
    after = dict(module.named_parameters())
    for name in expected_shapes:
        assert after[name].dtype is torch.float32
        assert id(after[name]) == identities[name]
    assert all(
        parameter.dtype is torch.float32
        for name, parameter in after.items()
        if name.endswith(("packed_weight_f32", "weight_scale_inv"))
    )

    module = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta")
    prefix = "model.layers.3.mlp.shared_experts"

    module.bind_checkpoint_sources(prefix)

    assert module._checkpoint_source_prefix == prefix
    for projection_name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(module, projection_name)
        assert projection._source_fqn == f"{prefix}.{projection_name}"
        assert projection._source_quant_format == "block_fp8"
        assert projection._is_prequantized is True
        assert projection._merge_sources is None
        assert projection._qlora_expected_skip_keys == {"weight"}
    module.bind_checkpoint_sources(prefix)
    with pytest.raises(RuntimeError, match="immutable once bound"):
        module.bind_checkpoint_sources("model.layers.4.mlp.shared_experts")
    with pytest.raises(ValueError, match="Invalid"):
        Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta").bind_checkpoint_sources("")


def test_shared_expert_physical_factor_views_match_pinned_sglang_tp_slices() -> None:
    pytest.importorskip("sglang")
    from sglang.srt.lora.layers import MergedColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA

    module = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="cpu")
    _fill_factors(module)
    ordinal = 11
    effective = tuple(
        factor.to(torch.bfloat16).contiguous()
        for factor in (
            module.gate_proj.lora_A,
            module.gate_proj.lora_B,
            module.up_proj.lora_A,
            module.up_proj.lora_B,
            module.down_proj.lora_A,
            module.down_proj.lora_B,
        )
    )
    actual = module._physical_factor_views_from_effective(*effective, ordinal)

    gate_up_A = torch.cat(
        (
            module.gate_proj.lora_A.to(torch.bfloat16),
            module.up_proj.lora_A.to(torch.bfloat16),
        ),
        dim=0,
    )
    gate_up_B = torch.cat(
        (
            module.gate_proj.lora_B.to(torch.bfloat16),
            module.up_proj.lora_B.to(torch.bfloat16),
        ),
        dim=0,
    )
    merged_stub = SimpleNamespace(
        base_layer=SimpleNamespace(
            tp_rank=ordinal,
            output_partition_sizes=(128, 128),
            output_sizes=(2048, 2048),
        )
    )
    expected_gate_up_A = MergedColumnParallelLinearWithLoRA.slice_lora_a_weights(merged_stub, gate_up_A)
    expected_gate_up_B = MergedColumnParallelLinearWithLoRA.slice_lora_b_weights(merged_stub, gate_up_B)

    row_stub = SimpleNamespace(
        base_layer=SimpleNamespace(
            tp_rank=ordinal,
            input_size_per_partition=128,
        )
    )
    expected_down_A = RowParallelLinearWithLoRA.slice_lora_a_weights(
        row_stub,
        module.down_proj.lora_A.to(torch.bfloat16),
    )
    expected_down_B = RowParallelLinearWithLoRA.slice_lora_b_weights(
        row_stub,
        module.down_proj.lora_B.to(torch.bfloat16),
    )

    assert actual.gate_up_A.shape == (1, 2, 6144)
    assert actual.gate_up_B.shape == (1, 256, 1)
    assert actual.down_A.shape == (1, 1, 128)
    assert actual.down_B.shape == (1, 6144, 1)
    assert torch.equal(actual.gate_up_A[0], expected_gate_up_A)
    assert torch.equal(actual.gate_up_B[0], expected_gate_up_B)
    assert torch.equal(actual.down_A[0], expected_down_A)
    assert torch.equal(actual.down_B[0], expected_down_B)
    assert all(tensor.is_contiguous() for tensor in (actual.gate_up_A, actual.gate_up_B, actual.down_A, actual.down_B))


def _assert_shared_expert_native_base_views_use_output_rows_and_input_columns() -> None:
    module = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="cpu")
    ordinal = 5
    gate_weight = torch.empty((2048, 6144), dtype=torch.float8_e4m3fn)
    up_weight = torch.empty_like(gate_weight)
    down_weight = torch.empty((6144, 2048), dtype=torch.float8_e4m3fn)
    gate_scales = torch.empty((16, 48), dtype=torch.float32)
    up_scales = torch.empty_like(gate_scales)
    down_scales = torch.empty((48, 16), dtype=torch.float32)
    for rank in range(16):
        start, end = rank * 128, (rank + 1) * 128
        gate_weight[start:end].fill_((rank + 1) / 32)
        up_weight[start:end].fill_(-(rank + 1) / 32)
        down_weight[:, start:end].fill_((rank + 1) / 64)
        gate_scales[rank].fill_(rank + 0.25)
        up_scales[rank].fill_(rank + 0.5)
        down_scales[:, rank].fill_(rank + 0.75)
    module.load_prequantized(
        gate_weight,
        gate_scales,
        up_weight,
        up_scales,
        down_weight,
        down_scales,
    )

    actual = module._physical_base_views(ordinal)
    start, end = ordinal * 128, (ordinal + 1) * 128
    assert actual.gate_up_weight.shape == (256, 6144)
    assert actual.gate_up_scales.shape == (2, 48)
    assert actual.down_weight.shape == (6144, 128)
    assert actual.down_scales.shape == (48, 1)
    assert torch.equal(actual.gate_up_weight[:128].view(torch.uint8), gate_weight[start:end].view(torch.uint8))
    assert torch.equal(actual.gate_up_weight[128:].view(torch.uint8), up_weight[start:end].view(torch.uint8))
    assert torch.equal(actual.gate_up_scales[0], gate_scales[ordinal])
    assert torch.equal(actual.gate_up_scales[1], up_scales[ordinal])
    assert torch.equal(actual.down_weight.view(torch.uint8), down_weight[:, start:end].contiguous().view(torch.uint8))
    assert torch.equal(actual.down_scales[:, 0], down_scales[:, ordinal])
