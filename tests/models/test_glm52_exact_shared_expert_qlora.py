from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xorl.distributed.canonical_moe import CanonicalMoEGraphMetadata, canonical_moe_reduce_reference
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


def _load_base(module: Glm52ExactTP16SharedExpertBlockFP8QLoRA) -> None:
    device = module.gate_proj.packed_weight_f32.device
    gate_weight = torch.full((2048, 6144), 0.25, dtype=torch.float8_e4m3fn, device=device)
    up_weight = torch.full((2048, 6144), -0.125, dtype=torch.float8_e4m3fn, device=device)
    down_weight = torch.full((6144, 2048), 0.0625, dtype=torch.float8_e4m3fn, device=device)
    gate_scales = torch.full((16, 48), 0.03125, dtype=torch.float32, device=device)
    up_scales = torch.full((16, 48), 0.0625, dtype=torch.float32, device=device)
    down_scales = torch.full((48, 16), 0.046875, dtype=torch.float32, device=device)
    module.load_prequantized(
        gate_weight,
        gate_scales,
        up_weight,
        up_scales,
        down_weight,
        down_scales,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 4096}, "hidden_size=6144"),
        ({"intermediate_size": 4096}, "intermediate_size=2048"),
        ({"tp_size": 8}, "requires TP16"),
        ({"r": 2}, "rank=1 and alpha=1"),
        ({"lora_alpha": 2}, "rank=1 and alpha=1"),
        ({"bias": True}, "bias-free"),
        ({"enable_aqn": True}, "rejects adaptive quantization noise"),
    ],
)
def test_shared_expert_construction_fails_closed(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Glm52ExactTP16SharedExpertBlockFP8QLoRA(device="meta", **kwargs)


def test_shared_expert_registers_one_logical_state_and_preserves_fp32_masters() -> None:
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


def test_shared_expert_checkpoint_sources_are_canonical_and_immutable() -> None:
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
        assert projection._qlora_expected_skip_keys == {"weight", "weight_scale_inv"}
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
    actual = module.physical_factor_views(ordinal)

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


def test_shared_expert_native_base_views_use_output_rows_and_input_columns() -> None:
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


def test_shared_expert_runtime_contract_fails_before_sglang_kernel_import() -> None:
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


def _manual_local_vjp(
    module: Glm52ExactTP16SharedExpertBlockFP8QLoRA,
    input: torch.Tensor,
    exact_gate_up: torch.Tensor,
    exact_activated: torch.Tensor,
    grad_output: torch.Tensor,
    ordinal: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    start, end = ordinal * 128, (ordinal + 1) * 128
    effective_gate_A = module.gate_proj.lora_A.detach().to(torch.bfloat16)
    effective_gate_B = module.gate_proj.lora_B.detach().to(torch.bfloat16)
    effective_up_A = module.up_proj.lora_A.detach().to(torch.bfloat16)
    effective_up_B = module.up_proj.lora_B.detach().to(torch.bfloat16)
    effective_down_A = module.down_proj.lora_A.detach().to(torch.bfloat16)
    effective_down_B = module.down_proj.lora_B.detach().to(torch.bfloat16)

    def projection_vjp(
        projection,
        projection_input,
        factor_A,
        factor_B,
        projection_grad,
        *,
        output_range=None,
        input_range=None,
        A_input_range=None,
        B_output_range=None,
    ):
        with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
            base_input = projection_input.detach().requires_grad_(True)
            base_weight = module._dequantized_partition_weight(
                projection,
                output_range=output_range,
                input_range=input_range,
            ).to(base_input.dtype)
            base_output = F.linear(base_input, base_weight)
            (base_input_grad,) = torch.autograd.grad(
                base_output,
                base_input,
                grad_outputs=projection_grad.to(base_output.dtype),
            )

            lora_input = projection_input.float().detach().requires_grad_(True)
            reference_A = factor_A.float().detach().requires_grad_(True)
            reference_B = factor_B.float().detach().requires_grad_(True)
            physical_A = reference_A if A_input_range is None else reference_A[:, A_input_range[0] : A_input_range[1]]
            physical_B = reference_B if B_output_range is None else reference_B[B_output_range[0] : B_output_range[1]]
            lora_output = F.linear(F.linear(lora_input, physical_A), physical_B)
            lora_input_grad, factor_A_grad, factor_B_grad = torch.autograd.grad(
                lora_output,
                (lora_input, reference_A, reference_B),
                grad_outputs=projection_grad.float(),
            )
        return base_input_grad.float() + lora_input_grad, factor_A_grad, factor_B_grad

    down_input_grad, down_A_grad, down_B_grad = projection_vjp(
        module.down_proj,
        exact_activated,
        effective_down_A,
        effective_down_B,
        grad_output,
        input_range=(start, end),
        A_input_range=(start, end),
    )
    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        gate_up_input = exact_gate_up.detach().requires_grad_(True)
        activation = F.silu(gate_up_input[:, :128]) * gate_up_input[:, 128:]
        (gate_up_grad,) = torch.autograd.grad(
            activation,
            gate_up_input,
            grad_outputs=down_input_grad.to(activation.dtype),
        )
    gate_grad, up_grad = gate_up_grad.split(128, dim=-1)
    gate_input_grad, gate_A_grad, gate_B_grad = projection_vjp(
        module.gate_proj,
        input,
        effective_gate_A,
        effective_gate_B,
        gate_grad,
        output_range=(start, end),
        B_output_range=(start, end),
    )
    up_input_grad, up_A_grad, up_B_grad = projection_vjp(
        module.up_proj,
        input,
        effective_up_A,
        effective_up_B,
        up_grad,
        output_range=(start, end),
        B_output_range=(start, end),
    )
    return gate_input_grad.to(input.dtype) + up_input_grad.to(input.dtype), {
        "gate_proj.lora_A": gate_A_grad,
        "gate_proj.lora_B": gate_B_grad,
        "up_proj.lora_A": up_A_grad,
        "up_proj.lora_B": up_B_grad,
        "down_proj.lora_A": down_A_grad,
        "down_proj.lora_B": down_B_grad,
    }


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_official_shared_expert_actual_operands_fold_and_surrogate_vjp() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability(0)[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 shared-expert component requires Hopper")
    from sglang.kernels.ops.gemm.gate_up_lora_b import gate_up_lora_b_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
    from sglang.srt.distributed.canonical_moe import (
        CanonicalRowSlots,
    )
    from sglang.srt.distributed.canonical_moe import (
        canonical_moe_reference as sampler_canonical_moe_reference,
    )
    from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
    from sglang.srt.lora.utils import LoRABatchInfo

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    module = Glm52ExactTP16SharedExpertBlockFP8QLoRA(device=device)
    _load_base(module)
    _fill_factors(module)
    rows, ordinal = 3, 11
    input = _pattern((rows, 6144), modulus=127, center=63, divisor=64, device=device).to(torch.bfloat16)
    effective = tuple(
        factor.detach().to(torch.bfloat16).contiguous()
        for factor in (
            module.gate_proj.lora_A,
            module.gate_proj.lora_B,
            module.up_proj.lora_A,
            module.up_proj.lora_B,
            module.down_proj.lora_A,
            module.down_proj.lora_B,
        )
    )
    actual_witness = module._exact_forward_value(input, *effective, contributor_ordinal=ordinal)
    cold_output = module(input, contributor_ordinal=ordinal)
    warm_output = module(input, contributor_ordinal=ordinal)

    factors = module.physical_factor_views(ordinal)
    base = module._physical_base_views(ordinal)
    batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.ones(1, dtype=torch.int32, device=device),
        scalings=torch.ones(1, dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )
    raw_gate_up_base = triton_w8a8_block_fp8_linear(
        input,
        base.gate_up_weight,
        [128, 128],
        base.gate_up_scales,
    )
    raw_gate_up_A = sgemm_lora_a_fwd(input, factors.gate_up_A, batch_info, stack_num=2)
    raw_gate_up = gate_up_lora_b_fwd(
        raw_gate_up_A,
        factors.gate_up_B,
        batch_info,
        128,
        base_output=raw_gate_up_base.clone(),
    )
    raw_activated = F.silu(raw_gate_up[:, :128]) * raw_gate_up[:, 128:]
    raw_down_base = triton_w8a8_block_fp8_linear(
        raw_activated,
        base.down_weight,
        [128, 128],
        base.down_scales,
    )
    raw_down_A = sgemm_lora_a_fwd(raw_activated, factors.down_A, batch_info)
    raw_output = sgemm_lora_b_fwd(
        raw_down_A,
        factors.down_B,
        batch_info,
        base_output=raw_down_base.clone(),
    )

    byte_pairs = {
        "gate_up_base": (actual_witness.gate_up_base, raw_gate_up_base),
        "gate_up_A": (actual_witness.gate_up_A_output, raw_gate_up_A),
        "gate_up_post_add": (actual_witness.gate_up, raw_gate_up),
        "activation": (actual_witness.activated, raw_activated),
        "down_base": (actual_witness.down_base, raw_down_base),
        "down_A": (actual_witness.down_A_output, raw_down_A),
        "local_partial": (actual_witness.output, raw_output),
        "cold": (cold_output, raw_output),
        "warm": (warm_output, raw_output),
    }
    for name, (actual, expected) in byte_pairs.items():
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name

    # One shared logical state emits sixteen physical partials. The existing
    # public canonical owner—not a new shared-expert reduction—performs the
    # adjacent-pairwise fold in sampler contributor order.
    with torch.no_grad():
        fold_input = input[:1].contiguous()
        partials = torch.stack(
            [module(fold_input, contributor_ordinal=rank) for rank in range(16)],
            dim=0,
        )
    metadata = CanonicalMoEGraphMetadata.build(
        torch.tensor([0], dtype=torch.int64, device=device),
        torch.tensor([0], dtype=torch.int64, device=device),
        capacity=1,
    )
    canonical = canonical_moe_reduce_reference(partials, metadata)
    sampler_slots = CanonicalRowSlots.from_positions(
        torch.tensor([0], dtype=torch.int64, device=device),
        capacity=1,
    )
    sampler_canonical = sampler_canonical_moe_reference(partials, sampler_slots)
    assert torch.equal(canonical.view(torch.uint8), sampler_canonical.view(torch.uint8))

    # Validate one physical producer's custom VJP against an independent,
    # staged QLoRA reference using the same effective BF16 factor bytes.
    grad_input = input.detach().clone().requires_grad_(True)
    grad_witness = module._exact_forward_value(grad_input.detach(), *effective, contributor_ordinal=ordinal)
    grad_output = _pattern(
        (rows, 6144),
        modulus=67,
        center=33,
        divisor=71,
        device=device,
    ).to(torch.bfloat16)
    expected_input_grad, expected_factor_grads = _manual_local_vjp(
        module,
        grad_input.detach(),
        grad_witness.gate_up,
        grad_witness.activated,
        grad_output,
        ordinal,
    )
    module(grad_input, contributor_ordinal=ordinal).backward(grad_output)

    assert torch.equal(grad_input.grad, expected_input_grad.to(torch.bfloat16))
    parameters = dict(module.named_parameters())
    for name, expected in expected_factor_grads.items():
        actual = parameters[name].grad
        assert actual is not None, name
        assert actual.dtype is torch.float32, name
        assert torch.equal(actual, expected), name
    start, end = ordinal * 128, (ordinal + 1) * 128
    assert not torch.count_nonzero(module.gate_proj.lora_B.grad[:start])
    assert not torch.count_nonzero(module.gate_proj.lora_B.grad[end:])
    assert not torch.count_nonzero(module.up_proj.lora_B.grad[:start])
    assert not torch.count_nonzero(module.up_proj.lora_B.grad[end:])
    assert not torch.count_nonzero(module.down_proj.lora_A.grad[:, :start])
    assert not torch.count_nonzero(module.down_proj.lora_A.grad[:, end:])
