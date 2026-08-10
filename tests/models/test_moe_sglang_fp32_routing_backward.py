"""FP32 routing-gradient contracts for SGLang fused-MoE training wrappers."""

import importlib

import pytest
import torch

from xorl.models.layers.moe.experts import (
    _group_gemm_same_nk_fp32_accumulator,
    _scale_moe_grad_by_fp32_routing,
    _SglangFusedExpertsEPTrainFunction,
    _SglangFusedExpertsTrainFunction,
)


pytestmark = [pytest.mark.cpu]


def test_fp32_routing_scale_rounds_only_after_multiply():
    grad_output = torch.tensor([[1.234375]], dtype=torch.bfloat16)
    routing = torch.tensor([2.99991e-5], dtype=torch.float32)

    expected = (grad_output.float() * routing[:, None]).to(torch.bfloat16)
    old_bf16_score_path = grad_output * routing.to(torch.bfloat16)[:, None]
    actual = _scale_moe_grad_by_fp32_routing(grad_output, routing)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, old_bf16_score_path), "stimulus must detect a pre-multiply BF16 score cast"


def test_fp32_accumulator_helper_requests_fresh_fp32_output():
    seen = {}

    def fake_group_gemm_same_nk(*, a, b, cumsum_M, max_M, output_dtype):
        seen.update(cumsum_M=cumsum_M, max_M=max_M, output_dtype=output_dtype)
        return (a.float() @ b[0].float()).to(output_dtype)

    a = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    b = torch.tensor([[[1.0], [2**-8]]], dtype=torch.bfloat16)
    cumsum = torch.tensor([1], dtype=torch.int32)
    actual = _group_gemm_same_nk_fp32_accumulator(
        fake_group_gemm_same_nk,
        a=a,
        b=b,
        cumsum_M=cumsum,
        max_M=1,
    )

    assert seen["output_dtype"] is torch.float32
    assert actual.dtype is torch.float32
    assert actual.item() == 1.00390625
    assert actual.item() != actual.to(torch.bfloat16).item()


@pytest.fixture()
def eager_grouped_moe(monkeypatch):
    """Replace CUDA bookkeeping/GEMMs with deterministic eager equivalents.

    This lets the custom Functions run on CPU while an independent, direct
    token/slot oracle supplies the forward and autograd reference.
    """
    group_gemm = importlib.import_module("xorl.ops.group_gemm.kernel.group_gemm")
    moe_kernels = importlib.import_module("xorl.ops.group_gemm.kernel.moe")
    triton_moe = importlib.import_module("xorl.ops.moe.triton")

    def expert_histogram(expert_index, num_experts):
        return torch.bincount(expert_index.reshape(-1).to(torch.int64), minlength=num_experts).to(torch.int32)

    def moe_index_compute(expert_index, cumsum):
        starts = torch.cat([cumsum.new_zeros(1), cumsum[:-1]]).to(torch.int64)
        cursors = starts.tolist()
        result = torch.empty_like(expert_index, dtype=torch.int64)
        for flat_index, expert in enumerate(expert_index.reshape(-1).tolist()):
            result.reshape(-1)[flat_index] = cursors[expert]
            cursors[expert] += 1
        return result

    def moe_scatter(x, index, out_dtype=None):
        topk = index.shape[1]
        output = torch.empty(index.numel(), x.shape[-1], dtype=out_dtype or x.dtype, device=x.device)
        for token in range(index.shape[0]):
            for slot in range(topk):
                output[index[token, slot]] = x[token]
        return output

    def group_gemm_same_nk(
        a,
        b,
        cumsum_M,
        max_M,
        transpose_a=False,
        transpose_b=False,
        c=None,
        output_dtype=None,
        **_,
    ):
        assert not transpose_a
        del max_M
        output_width = b.shape[1] if transpose_b else b.shape[2]
        output = (
            c
            if c is not None
            else torch.empty(
                a.shape[0],
                output_width,
                dtype=a.dtype if output_dtype is None else output_dtype,
                device=a.device,
            )
        )
        start = 0
        for expert, end_tensor in enumerate(cumsum_M):
            end = int(end_tensor)
            matrix = b[expert].transpose(0, 1) if transpose_b else b[expert]
            value = a[start:end].float() @ matrix.float()
            if c is None:
                output[start:end] = value.to(output.dtype)
            else:
                output[start:end].add_(value.to(output.dtype))
            start = end
        return output

    def group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=False, transpose_b=False, **_):
        assert transpose_a and not transpose_b
        del max_K
        start = 0
        for expert, end_tensor in enumerate(cumsum_K):
            end = int(end_tensor)
            value = a[start:end].float().transpose(0, 1) @ b[start:end].float()
            c[expert].copy_(value.to(c.dtype))
            start = end
        return c

    monkeypatch.setattr(moe_kernels, "expert_histogram", expert_histogram)
    monkeypatch.setattr(moe_kernels, "moe_index_compute", moe_index_compute)
    monkeypatch.setattr(moe_kernels, "moe_scatter", moe_scatter)
    monkeypatch.setattr(group_gemm, "group_gemm_same_nk", group_gemm_same_nk)
    monkeypatch.setattr(group_gemm, "group_gemm_same_mn", group_gemm_same_mn)
    monkeypatch.setattr(triton_moe, "_maybe_clamp_swiglu_gate", lambda gate, _limit: gate)
    monkeypatch.setattr(triton_moe, "_apply_swiglu_clamp_backward", lambda grad, _gate, _limit: grad)
    monkeypatch.setattr(triton_moe, "_moe_gate_activation", lambda gate, _hidden_act: gate)
    monkeypatch.setattr(triton_moe, "_moe_gate_activation_backward", lambda grad, _gate, _hidden_act: grad)


def _explicit_serving_forward(hidden, routing, selected, gate_up, down, *, filter_expert=False):
    """Direct token/slot oracle: FP32 down accumulator times FP32 routing."""
    rows = []
    for token in range(hidden.shape[0]):
        slots = []
        for slot in range(selected.shape[1]):
            expert = int(selected[token, slot])
            if filter_expert and expert < 0:
                slots.append(torch.zeros_like(hidden[token]))
                continue
            gate_up_row = (hidden[token].float() @ gate_up[expert].float()).to(hidden.dtype)
            intermediate = gate_up_row.shape[0] // 2
            gated = gate_up_row[:intermediate] * gate_up_row[intermediate:]
            down_accumulator = gated.float() @ down[expert].float()
            slots.append((down_accumulator * routing[token, slot].float()).to(hidden.dtype))
        rows.append(torch.stack(slots).float().sum(dim=0).to(hidden.dtype))
    return torch.stack(rows)


def _fake_sglang_impl(hidden, w13, w2, routing, selected, **kwargs):
    return _explicit_serving_forward(
        hidden,
        routing,
        selected,
        w13.transpose(1, 2),
        w2.transpose(1, 2),
        filter_expert=kwargs["filter_expert"],
    )


def _assert_gradients_equal(actual, expected):
    for label, left, right in zip(("dX", "dRouting", "dW13", "dW2"), actual, expected):
        assert left is not None and right is not None
        assert torch.equal(left, right), f"{label} mismatch:\nactual={left}\nexpected={right}"


@pytest.mark.parametrize("filter_expert", [False, True])
def test_local_backward_matches_fp32_routing_oracle_with_int32_ids(eager_grouped_moe, filter_expert):
    hidden = torch.ones(2, 1, dtype=torch.bfloat16)
    routing = torch.tensor([[2.99991e-5, 0.5009], [0.33331, 0.77771]], dtype=torch.float32)
    selected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    if filter_expert:
        selected[0, 1] = -1
    gate_up = torch.ones(2, 1, 2, dtype=torch.bfloat16)
    down = torch.ones(2, 1, 1, dtype=torch.bfloat16)
    grad_output = torch.tensor([[1.234375], [-0.703125]], dtype=torch.bfloat16)

    custom = [value.clone().requires_grad_(True) for value in (hidden, routing, gate_up, down)]
    output = _SglangFusedExpertsTrainFunction.apply(
        custom[0],
        custom[1],
        selected,
        custom[2],
        custom[3],
        _fake_sglang_impl,
        "silu",
        "silu",
        0.0,
        2,
        None,
        filter_expert,
    )
    output.backward(grad_output)

    reference = [value.clone().requires_grad_(True) for value in (hidden, routing, gate_up, down)]
    reference_output = _explicit_serving_forward(
        reference[0], reference[1], selected, reference[2], reference[3], filter_expert=filter_expert
    )
    reference_output.backward(grad_output)

    _assert_gradients_equal([value.grad for value in custom], [value.grad for value in reference])
    if filter_expert:
        assert custom[1].grad[0, 1].item() == 0.0


def test_ep_backward_matches_fp32_routing_oracle(eager_grouped_moe):
    hidden = torch.ones(4, 1, dtype=torch.bfloat16)
    routing = torch.tensor([2.99991e-5, 0.5009, 0.33331, 0.77771], dtype=torch.float32)
    selected = torch.tensor([[0], [0], [1], [1]], dtype=torch.int32)
    cumsum = torch.tensor([2, 4], dtype=torch.int32)
    gate_up = torch.ones(2, 1, 2, dtype=torch.bfloat16)
    down = torch.ones(2, 1, 1, dtype=torch.bfloat16)
    grad_output = torch.tensor([[1.234375], [-0.703125], [0.8125], [-1.125]], dtype=torch.bfloat16)

    custom = [value.clone().requires_grad_(True) for value in (hidden, routing, gate_up, down)]
    output = _SglangFusedExpertsEPTrainFunction.apply(
        custom[0], custom[1], custom[2], custom[3], cumsum, _fake_sglang_impl, "silu", "silu", 0.0, True
    )
    output.backward(grad_output)

    reference = [value.clone().requires_grad_(True) for value in (hidden, routing, gate_up, down)]
    reference_output = _explicit_serving_forward(
        reference[0], reference[1].reshape(-1, 1), selected, reference[2], reference[3]
    )
    reference_output.backward(grad_output)

    _assert_gradients_equal([value.grad for value in custom], [value.grad for value in reference])
