"""Tests for the GLM-5.2 full-parameter block-FP8 exact-lane components.

CPU tests cover admission, fail-closed combinations, the staleness gate, and
the autograd composition on monkeypatched value programs (mirroring the
existing GLM-5.2 exact component tests).  Hopper CUDA tests cover the
load-bearing byte contracts: quantize/publish identity, step-0 checkpoint
byte preservation, forward-byte equality against the frozen serving-program
consumer of the same published bytes, and the straight-through gradients.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION,
    GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION,
    Glm52ExactFullParamRouterWeight,
    Glm52ExactTP1BlockFP8FullParamLinear,
    Glm52FullParamDenseMLP,
    quantize_expert_masters_to_serving_bytes,
    quantize_master_to_serving_bytes,
)
from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8DenseMLP
from xorl.ops.block_fp8_native import NativeBlockFP8Linear
from xorl.ops.fused_silu_and_mul import exact_fp32_silu_and_mul


def _linear(in_features: int = 8, out_features: int = 16) -> Glm52ExactTP1BlockFP8FullParamLinear:
    module = Glm52ExactTP1BlockFP8FullParamLinear(in_features, out_features, device=torch.device("cpu"))
    with torch.no_grad():
        module.weight_master.copy_(
            torch.arange(out_features * in_features, dtype=torch.float32)
            .reshape(out_features, in_features)
            .sub_(37)
            .div_(53)
        )
    return module


def _router(num_experts: int = 4, hidden_size: int = 8) -> Glm52ExactFullParamRouterWeight:
    module = Glm52ExactFullParamRouterWeight(num_experts, hidden_size, device=torch.device("cpu"))
    with torch.no_grad():
        module.weight_master.copy_(
            torch.arange(num_experts * hidden_size, dtype=torch.float32)
            .reshape(num_experts, hidden_size)
            .sub_(11)
            .div_(29)
        )
    return module


# ---------------------------------------------------------------------------
# Construction admission (CPU)
# ---------------------------------------------------------------------------


def test_linear_construction_fails_closed_on_unsupported_combinations() -> None:
    with pytest.raises(ValueError, match="bias-free"):
        Glm52ExactTP1BlockFP8FullParamLinear(8, 16, bias=True)
    with pytest.raises(ValueError, match=r"\(128, 128\)"):
        Glm52ExactTP1BlockFP8FullParamLinear(8, 16, block_size=(64, 128))
    with pytest.raises(ValueError, match="divisible by four"):
        Glm52ExactTP1BlockFP8FullParamLinear(6, 16)
    with pytest.raises(ValueError, match="positive"):
        Glm52ExactTP1BlockFP8FullParamLinear(0, 16)

    module = _linear(576, 256)
    assert module.contract_version == GLM52_EXACT_TP1_FULLPARAM_FP8_CONTRACT_VERSION
    assert module.fsdp_requires_full_precision is True
    assert module.weight_master.dtype is torch.float32
    assert module.weight_master.requires_grad
    assert module.quantized_weight_f32.shape == (256, 144)
    assert module.weight_scale_inv.shape == (math.ceil(256 / 128), math.ceil(576 / 128))
    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert trainable == {"weight_master"}
    # The byte cache and scales are buffers, never trainable parameters.
    assert {name for name, _ in module.named_buffers()} == {"quantized_weight_f32", "weight_scale_inv"}

    with pytest.raises(TypeError, match="Expected nn.Linear"):
        Glm52ExactTP1BlockFP8FullParamLinear.from_linear(torch.nn.Conv1d(4, 4, 1))
    biased = torch.nn.Linear(8, 16, bias=True)
    with pytest.raises(ValueError, match="bias-free"):
        Glm52ExactTP1BlockFP8FullParamLinear.from_linear(biased)


def test_router_construction_fails_closed_and_declares_contract() -> None:
    with pytest.raises(ValueError, match="positive"):
        Glm52ExactFullParamRouterWeight(0, 8)
    module = _router()
    assert module.contract_version == GLM52_EXACT_FULLPARAM_ROUTER_CONTRACT_VERSION
    assert module.fsdp_requires_full_precision is True
    assert module.weight_master.dtype is torch.float32
    assert module._effective_weight.dtype is torch.bfloat16
    with pytest.raises(TypeError, match="must be BF16"):
        module.load_from_bf16(torch.zeros(4, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match="does not match"):
        module.load_from_bf16(torch.zeros(4, 4, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# Staleness gate (CPU)
# ---------------------------------------------------------------------------


def test_linear_forward_and_publication_fail_closed_before_seed_and_after_mutation(monkeypatch) -> None:
    module = _linear()
    input = torch.zeros(2, 8, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="before the quantized cache was seeded"):
        module(input)
    with pytest.raises(RuntimeError, match="before the quantized cache was seeded"):
        module.publishable_weight_bytes()

    # Simulate a completed refresh without CUDA, then verify the gate trips on
    # any master mutation (in-place update bumps the version counter) and on a
    # .data swap (data_ptr changes).
    module._record_master_identity()
    monkeypatch.setattr(module, "_exact_forward_value", lambda value: torch.zeros(2, 16, dtype=torch.bfloat16))
    module(input)

    with torch.no_grad():
        module.weight_master.add_(1.0)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(input)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module.publishable_weight_bytes()

    module._record_master_identity()
    module(input)
    module.weight_master.data = module.weight_master.data.clone()
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(input)


def test_linear_cache_refresh_and_quantizer_require_cuda() -> None:
    module = _linear()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module.refresh_quantized_cache()
    with pytest.raises(RuntimeError, match="requires CUDA"):
        quantize_master_to_serving_bytes(torch.zeros(8, 8, dtype=torch.float32))
    with pytest.raises(TypeError, match="FP32 master"):
        quantize_master_to_serving_bytes(torch.zeros(8, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="2D master"):
        quantize_master_to_serving_bytes(torch.zeros(8, dtype=torch.float32))


def test_router_forward_and_publication_fail_closed_before_seed_and_after_mutation(monkeypatch) -> None:
    module = _router()
    hidden = torch.zeros(3, 8, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="before the BF16 view was seeded"):
        module(hidden)
    with pytest.raises(RuntimeError, match="before the BF16 view was seeded"):
        module.publishable_weight_bytes()

    module._record_master_identity()
    monkeypatch.setattr(module, "_router_value", lambda value: torch.zeros(3, 4, dtype=torch.float32))
    module(hidden)

    with torch.no_grad():
        module.weight_master.mul_(2.0)
    with pytest.raises(RuntimeError, match="stale BF16 view"):
        module(hidden)
    with pytest.raises(RuntimeError, match="stale BF16 view"):
        module.publishable_weight_bytes()


# ---------------------------------------------------------------------------
# Engaged-contract admission (CPU)
# ---------------------------------------------------------------------------


def test_linear_engaged_contract_rejects_bad_inputs_before_any_kernel(monkeypatch) -> None:
    module = _linear()
    module._record_master_identity()
    monkeypatch.setattr(module, "_exact_forward_value", lambda value: torch.zeros(2, 16, dtype=torch.bfloat16))

    with pytest.raises(TypeError, match="BF16 activations"):
        module(torch.zeros(2, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match="input width"):
        module(torch.zeros(2, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        module(torch.zeros(8, 4, dtype=torch.bfloat16).t())
    with pytest.raises(ValueError, match="empty"):
        module(torch.zeros(0, 8, dtype=torch.bfloat16))

    module.weight_master.requires_grad_(False)
    with pytest.raises(RuntimeError, match="must be trainable"):
        module(torch.zeros(2, 8, dtype=torch.bfloat16))
    module.weight_master.requires_grad_(True)

    # A frozen-master configuration belongs to the native frozen lane, not here.
    module(torch.zeros(2, 8, dtype=torch.bfloat16))


def test_router_engaged_contract_rejects_bad_inputs_before_any_kernel(monkeypatch) -> None:
    module = _router()
    module._record_master_identity()
    monkeypatch.setattr(module, "_router_value", lambda value: torch.zeros(3, 4, dtype=torch.float32))

    with pytest.raises(TypeError, match="BF16 hidden states"):
        module(torch.zeros(3, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match=r"\[tokens, hidden\]"):
        module(torch.zeros(3, 2, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="empty"):
        module(torch.zeros(0, 8, dtype=torch.bfloat16))
    module.weight_master.requires_grad_(False)
    with pytest.raises(RuntimeError, match="must be trainable"):
        module(torch.zeros(3, 8, dtype=torch.bfloat16))


def test_load_prequantized_validates_bytes_before_touching_state() -> None:
    module = _linear(128, 128)
    good_weight = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
    good_scale = torch.ones(1, 1, dtype=torch.float32)

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        module.load_prequantized(torch.zeros(128, 128, dtype=torch.bfloat16), good_scale)
    with pytest.raises(ValueError, match="weight shape"):
        module.load_prequantized(torch.zeros(128, 64, dtype=torch.float8_e4m3fn), good_scale)
    with pytest.raises(TypeError, match="FP32"):
        module.load_prequantized(good_weight, good_scale.to(torch.float64))
    with pytest.raises(ValueError, match="scale shape"):
        module.load_prequantized(good_weight, torch.ones(2, 2, dtype=torch.float32))
    with pytest.raises(ValueError, match="non-finite"):
        module.load_prequantized(good_weight, torch.full((1, 1), float("inf")))
    # Validation passes on CPU; seeding itself requires the pinned CUDA dequant.
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module.load_prequantized(good_weight, good_scale)
    assert module._master_identity is None


# ---------------------------------------------------------------------------
# Autograd composition on monkeypatched value programs (CPU)
# ---------------------------------------------------------------------------


def test_linear_straight_through_backward_produces_fp32_master_grad_and_bf16_program_dgrad(monkeypatch) -> None:
    module = _linear()
    module._record_master_identity()

    dequantized = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8).sub_(63).div_(97)

    def forward_value(value: torch.Tensor) -> torch.Tensor:
        return F.linear(value.float(), dequantized).to(torch.bfloat16)

    monkeypatch.setattr(module, "_exact_forward_value", forward_value)
    monkeypatch.setattr(module, "_dequantized_cached_weight", lambda: dequantized)

    input = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8).sub_(11).div_(13).to(torch.bfloat16).requires_grad_()
    output = module(input)
    assert output.dtype is torch.bfloat16
    grad_output = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16).sub_(23).div_(41).to(torch.bfloat16)
    output.backward(grad_output)

    # Master wgrad: straight-through FP32 GEMM, no BF16 rounding.
    expected_master_grad = grad_output.float().t().matmul(input.detach().float())
    assert module.weight_master.grad is not None
    assert module.weight_master.grad.dtype is torch.float32
    assert torch.equal(module.weight_master.grad, expected_master_grad)

    # Activation dgrad: the declared BF16 linear program on the dequantized bytes.
    reference_input = input.detach().clone().requires_grad_()
    F.linear(reference_input, dequantized.to(torch.bfloat16)).backward(grad_output)
    assert input.grad is not None
    assert torch.equal(input.grad, reference_input.grad)


def test_linear_backward_raises_if_master_mutates_between_forward_and_backward(monkeypatch) -> None:
    module = _linear()
    module._record_master_identity()
    monkeypatch.setattr(
        module,
        "_exact_forward_value",
        lambda value: F.linear(value.float(), module.weight_master.detach()).to(torch.bfloat16),
    )
    input = torch.zeros(2, 8, dtype=torch.bfloat16, requires_grad=True)
    output = module(input)
    with torch.no_grad():
        module.weight_master.add_(1.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.float().sum().backward()


def test_router_backward_produces_fp32_master_grad(monkeypatch) -> None:
    module = _router()
    module.load_from_bf16(module.weight_master.detach().to(torch.bfloat16))

    def router_value(hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float().matmul(module._effective_weight.float().t())

    monkeypatch.setattr(module, "_router_value", router_value)
    hidden = (
        torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8).sub_(17).div_(19).to(torch.bfloat16).requires_grad_()
    )
    logits = module(hidden)
    assert logits.dtype is torch.float32
    grad_logits = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4).sub_(7).div_(31)
    logits.backward(grad_logits)

    expected_master_grad = grad_logits.t().matmul(hidden.detach().float())
    assert module.weight_master.grad is not None
    assert module.weight_master.grad.dtype is torch.float32
    assert torch.equal(module.weight_master.grad, expected_master_grad)
    expected_hidden_grad = grad_logits.matmul(module._effective_weight.float()).to(torch.bfloat16)
    assert torch.equal(hidden.grad, expected_hidden_grad)


def test_router_load_from_bf16_seeds_master_exactly_and_refresh_reproduces_bytes() -> None:
    module = _router()
    checkpoint = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8).sub_(13).div_(7).to(torch.bfloat16)
    module.load_from_bf16(checkpoint)
    assert torch.equal(module.publishable_weight_bytes(), checkpoint)
    # BF16 -> FP32 widening is exact, so a refresh regenerates identical bytes.
    module.refresh_effective_view()
    assert torch.equal(module.publishable_weight_bytes(), checkpoint)
    assert module.publishable_weight_bytes().data_ptr() == module._effective_weight.data_ptr()


def _dense_mlp_gate_up(input: torch.Tensor) -> torch.Tensor:
    return torch.cat((input, torch.flip(input, dims=(-1,))), dim=-1).contiguous()


def test_dense_mlp_routes_to_one_round_and_discriminates_the_old_program(monkeypatch) -> None:
    """Both full-param and serving composites select the one-round program.

    The same fixture must distinguish the retired two-round program so the
    route and cross-composite equality assertions cannot pass vacuously.
    """

    module = Glm52FullParamDenseMLP(128, 128, device="cpu")
    monkeypatch.setattr(module.gate_up_proj, "forward", _dense_mlp_gate_up)
    monkeypatch.setattr(module.down_proj, "forward", lambda activated: activated)
    serving = Glm52NativeBlockFP8DenseMLP(128, 128, device="cpu")
    monkeypatch.setattr(serving.gate_up_proj, "forward", _dense_mlp_gate_up)
    monkeypatch.setattr(serving.down_proj, "forward", lambda activated: activated)

    generator = torch.Generator().manual_seed(9173)
    input = (torch.randn(32, 128, generator=generator) * 1.5).to(torch.bfloat16)
    gate_up = _dense_mlp_gate_up(input)
    expected = exact_fp32_silu_and_mul(gate_up)
    split = gate_up.shape[-1] // 2
    retired = F.silu(gate_up[..., :split]) * gate_up[..., split:]

    actual = module(input)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(serving(input).view(torch.uint8), actual.view(torch.uint8))
    differing = int((expected.view(torch.uint16) != retired.view(torch.uint16)).sum())
    assert differing > 0, "one-round sensitivity fixture no longer distinguishes the retired program"
    assert not torch.equal(actual.view(torch.uint8), retired.view(torch.uint8))


def test_dense_mlp_one_round_gradient_matches_the_selected_program(monkeypatch) -> None:
    module = Glm52FullParamDenseMLP(128, 128, device="cpu")
    monkeypatch.setattr(module.gate_up_proj, "forward", _dense_mlp_gate_up)
    monkeypatch.setattr(module.down_proj, "forward", lambda activated: activated)

    generator = torch.Generator().manual_seed(31)
    input = torch.randn(7, 128, generator=generator).to(torch.bfloat16).requires_grad_(True)
    grad_output = torch.randn(7, 128, generator=generator).to(torch.bfloat16)
    module(input).backward(grad_output)

    reference_input = input.detach().clone().requires_grad_(True)
    exact_fp32_silu_and_mul(_dense_mlp_gate_up(reference_input)).backward(grad_output)
    assert input.grad is not None
    assert torch.equal(input.grad, reference_input.grad)
    assert bool(torch.all(torch.isfinite(input.grad.float())))


# ---------------------------------------------------------------------------
# Hopper CUDA byte contracts
# ---------------------------------------------------------------------------


def _hopper_or_skip() -> torch.device:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the qualified exact GLM-5.2 component requires Hopper")
    return torch.device("cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    ((6144, 576), (512, 384)),
    ids=("kv-a-partial-edge", "aligned"),
)
def test_cuda_publication_returns_the_exact_bytes_the_forward_consumed(in_features: int, out_features: int) -> None:
    device = _hopper_or_skip()
    torch.manual_seed(0)
    base = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32, device=device)
    module = Glm52ExactTP1BlockFP8FullParamLinear.from_linear(base)

    published_weight, published_scale = module.publishable_weight_bytes()
    # Publication is a view of the consumed cache, not a copy.
    assert published_weight.data_ptr() == module.quantized_weight_f32.data_ptr()
    assert published_scale.data_ptr() == module.weight_scale_inv.data_ptr()

    # An independent run of the quantization program on the same master
    # reproduces the cache: Q is a pure function of the master bytes.
    reference_weight, reference_scale = quantize_master_to_serving_bytes(module.weight_master)
    assert torch.equal(published_weight.view(torch.uint8), reference_weight.view(torch.uint8))
    assert torch.equal(published_scale, reference_scale)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_step0_checkpoint_bytes_are_preserved_not_requantized() -> None:
    device = _hopper_or_skip()
    out_features, in_features = 384, 512
    weight_values = torch.arange(out_features * in_features, device=device, dtype=torch.int32)
    checkpoint_weight = ((weight_values % 31) - 15).reshape(out_features, in_features).float().to(torch.float8_e4m3fn)
    scale_shape = ((out_features + 127) // 128, (in_features + 127) // 128)
    checkpoint_scale = (
        torch.arange(scale_shape[0] * scale_shape[1], device=device, dtype=torch.float32)
        .reshape(scale_shape)
        .remainder_(31)
        .add_(1)
        .div_(32)
        .contiguous()
    )

    module = Glm52ExactTP1BlockFP8FullParamLinear(in_features, out_features, device=device)
    module.load_prequantized(checkpoint_weight, checkpoint_scale)

    published_weight, published_scale = module.publishable_weight_bytes()
    assert torch.equal(published_weight.view(torch.uint8), checkpoint_weight.view(torch.uint8))
    assert torch.equal(published_scale, checkpoint_scale)

    # Demonstrate the hazard the seeding mechanism exists for: requantizing
    # the dequantized checkpoint is NOT a byte round-trip in general.  (If
    # this ever starts holding bitwise for adversarial scales, the seeding
    # mechanism becomes belt-and-braces rather than load-bearing.)
    requantized_weight, requantized_scale = quantize_master_to_serving_bytes(module.weight_master)
    roundtrip_identical = torch.equal(
        requantized_weight.view(torch.uint8), checkpoint_weight.view(torch.uint8)
    ) and torch.equal(requantized_scale, checkpoint_scale)
    assert not roundtrip_identical, (
        "Q(dequant(checkpoint)) reproduced the checkpoint bytes for this fixture; "
        "the step-0 seeding mechanism is still required in general, but this "
        "fixture no longer demonstrates the hazard - strengthen it."
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    ((6144, 576), (512, 384)),
    ids=("kv-a-partial-edge", "aligned"),
)
def test_cuda_forward_bytes_match_frozen_serving_program_on_published_bytes(
    in_features: int, out_features: int
) -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    torch.manual_seed(1)
    base = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32, device=device)
    module = Glm52ExactTP1BlockFP8FullParamLinear.from_linear(base)
    published_weight, published_scale = module.publishable_weight_bytes()

    # The frozen native module executes the sampler's exact serving-value
    # program; loading the published bytes into it stands in for the sampler
    # after a weight sync.
    sampler_stand_in = NativeBlockFP8Linear(in_features, out_features, device=device)
    sampler_stand_in.load_prequantized(published_weight.contiguous(), published_scale.contiguous())

    rows = 17
    input = (
        torch.arange(rows * in_features, device=device, dtype=torch.float32)
        .remainder_(127)
        .sub_(63)
        .div_(64)
        .reshape(rows, in_features)
        .to(torch.bfloat16)
    )
    trainer_bytes = module(input.clone().requires_grad_(True))
    sampler_bytes = sampler_stand_in(input)
    assert trainer_bytes.dtype is torch.bfloat16
    assert torch.equal(trainer_bytes.detach(), sampler_bytes)

    # Co-batching must not change a row's bytes (lane batch-invariance bar).
    single = module(input[:1].contiguous())
    assert torch.equal(trainer_bytes.detach()[:1], single.detach())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_optimizer_step_trips_staleness_and_refresh_republishes_new_bytes() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    torch.manual_seed(2)
    base = torch.nn.Linear(512, 384, bias=False, dtype=torch.float32, device=device)
    module = Glm52ExactTP1BlockFP8FullParamLinear.from_linear(base)
    before = module.publishable_weight_bytes()[0].view(torch.uint8).clone()

    optimizer = torch.optim.SGD([module.weight_master], lr=10.0)
    input = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    module(input).float().sum().backward()
    optimizer.step()

    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module(input)
    with pytest.raises(RuntimeError, match="stale quantized cache"):
        module.publishable_weight_bytes()

    module.refresh_quantized_cache()
    after_weight, _after_scale = module.publishable_weight_bytes()
    module(input)
    reference_weight, _ = quantize_master_to_serving_bytes(module.weight_master)
    assert torch.equal(after_weight.view(torch.uint8), reference_weight.view(torch.uint8))
    # lr=10 guarantees the master moved far beyond one FP8 quantization step.
    assert not torch.equal(after_weight.view(torch.uint8), before)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_straight_through_gradients_match_reference_programs() -> None:
    device = _hopper_or_skip()
    pytest.importorskip("sglang")
    torch.manual_seed(3)
    base = torch.nn.Linear(512, 384, bias=False, dtype=torch.float32, device=device)
    module = Glm52ExactTP1BlockFP8FullParamLinear.from_linear(base)

    rows = 9
    input = torch.randn(rows, 512, device=device, dtype=torch.bfloat16).requires_grad_(True)
    grad_output = torch.randn(rows, 384, device=device, dtype=torch.bfloat16)
    module(input).backward(grad_output)

    expected_master_grad = grad_output.float().t().matmul(input.detach().float())
    assert torch.equal(module.weight_master.grad, expected_master_grad)

    reference_input = input.detach().clone().requires_grad_(True)
    dequantized = module._dequantized_cached_weight().to(torch.bfloat16)
    F.linear(reference_input, dequantized).backward(grad_output)
    assert torch.equal(input.grad, reference_input.grad)


def test_expert_quantization_admission_fails_closed_on_ragged_geometry() -> None:
    with pytest.raises(ValueError, match=r"\[E, out, in\]"):
        quantize_expert_masters_to_serving_bytes(torch.zeros(8, 8, dtype=torch.float32))
    with pytest.raises(ValueError, match="128-aligned"):
        quantize_expert_masters_to_serving_bytes(torch.zeros(2, 96, 128, dtype=torch.float32))
    with pytest.raises(ValueError, match="128-aligned"):
        quantize_expert_masters_to_serving_bytes(torch.zeros(2, 128, 192, dtype=torch.float32))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_expert_quantization_preserves_expert_boundaries_and_fuse_after_quantize() -> None:
    device = _hopper_or_skip()
    torch.manual_seed(5)
    num_experts, intermediate, hidden = 4, 256, 384
    gate = torch.randn(num_experts, intermediate, hidden, device=device, dtype=torch.float32)
    up = torch.randn(num_experts, intermediate, hidden, device=device, dtype=torch.float32)
    fused = torch.cat((gate, up), dim=1).contiguous()

    bank_weight, bank_scale = quantize_expert_masters_to_serving_bytes(fused)
    assert bank_weight.shape == (num_experts, 2 * intermediate, hidden)
    assert bank_scale.shape == (num_experts, 2 * intermediate // 128, hidden // 128)

    for expert_index in range(num_experts):
        # Expert-boundary preservation: the stacked helper output equals an
        # independent per-expert run of the same Q program.
        expert_weight, expert_scale = quantize_master_to_serving_bytes(fused[expert_index].contiguous())
        assert torch.equal(bank_weight[expert_index].view(torch.uint8), expert_weight.view(torch.uint8))
        assert torch.equal(bank_scale[expert_index], expert_scale)

        # Fuse-after-quantize equivalence on 128-aligned rows: quantizing the
        # fused [gate; up] matrix equals concatenating per-projection
        # quantizations, bytes and scales alike.
        gate_weight, gate_scale = quantize_master_to_serving_bytes(gate[expert_index].contiguous())
        up_weight, up_scale = quantize_master_to_serving_bytes(up[expert_index].contiguous())
        assert torch.equal(
            bank_weight[expert_index].view(torch.uint8),
            torch.cat((gate_weight, up_weight), dim=0).view(torch.uint8),
        )
        assert torch.equal(bank_scale[expert_index], torch.cat((gate_scale, up_scale), dim=0))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_cuda_router_forward_bytes_match_frozen_router_program_on_published_bytes() -> None:
    device = _hopper_or_skip()
    from xorl.ops.batch_invariant_ops import bi_router_gemm

    torch.manual_seed(4)
    num_experts, hidden_size = 256, 512
    module = Glm52ExactFullParamRouterWeight(num_experts, hidden_size, device=device)
    checkpoint = torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16)
    module.load_from_bf16(checkpoint)

    hidden = torch.randn(13, hidden_size, device=device, dtype=torch.bfloat16).requires_grad_(True)
    trainer_logits = module(hidden)
    frozen_logits = bi_router_gemm(hidden.detach(), module.publishable_weight_bytes())
    assert trainer_logits.dtype is torch.float32
    assert torch.equal(trainer_logits.detach(), frozen_logits)
    assert torch.equal(module.publishable_weight_bytes(), checkpoint)

    grad_logits = torch.randn(13, num_experts, device=device, dtype=torch.float32)
    trainer_logits.backward(grad_logits)
    expected_master_grad = grad_logits.t().matmul(hidden.detach().float())
    assert torch.equal(module.weight_master.grad, expected_master_grad)

    # After an optimizer-style update the stale view must fail closed, and a
    # refresh publishes the BF16 rounding of the new master.
    with torch.no_grad():
        module.weight_master.add_(module.weight_master.grad, alpha=-1.0)
    with pytest.raises(RuntimeError, match="stale BF16 view"):
        module(hidden.detach())
    module.refresh_effective_view()
    assert torch.equal(
        module.publishable_weight_bytes(),
        module.weight_master.detach().to(torch.bfloat16),
    )
