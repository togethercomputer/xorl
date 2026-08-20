from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import xorl.models.transformers.deepseek_v4.exact_lm_head as exact_head
from xorl.models.transformers.deepseek_v4.exact_lm_head import (
    _Dsv4ExactDistributedHeadFunction,
    _rank_order_variable_row_all_gather,
    _temperature_scale_bf16_logits,
)
from xorl.ops.exact_sampling_transforms import (
    selected_logprob_reference_grad as _selected_logprob_reference_grad,
)
from xorl.ops.exact_sampling_transforms import (
    selected_logprob_reference_grad_filtered as _selected_logprob_reference_grad_filtered,
)
from xorl.ops.exact_sampling_transforms import (
    selected_logprob_reference_grad_partitioned as _selected_logprob_reference_grad_partitioned,
)
from xorl.ops.loss.per_token_ce import compute_per_token_ce


def test_dsv4_temperature_scaling_preserves_bf16_store_contract() -> None:
    logits = torch.tensor(
        [[1.25, -0.5, 0.75], [-1.0, 2.0, 0.25]],
        dtype=torch.bfloat16,
    )
    ones = torch.ones(2, dtype=torch.float32)
    mixed = torch.tensor([0.7, 1.3], dtype=torch.float32)

    assert _temperature_scale_bf16_logits(logits, None) is logits
    assert torch.equal(
        _temperature_scale_bf16_logits(logits, ones).view(torch.uint8),
        logits.view(torch.uint8),
    )
    assert torch.equal(
        _temperature_scale_bf16_logits(logits, mixed).view(torch.uint8),
        logits.bfloat16().div(mixed.unsqueeze(1)).bfloat16().view(torch.uint8),
    )


def test_dsv4_temperature_reference_gradient_contains_per_row_inverse() -> None:
    logits = torch.tensor(
        [[1.25, -0.5, 0.75], [-1.0, 2.0, 0.25]],
        dtype=torch.float32,
    )
    token_ids = torch.tensor([2, 1], dtype=torch.int64)
    grad_logprob = torch.tensor([0.5, -0.75], dtype=torch.float32)
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)

    actual = _selected_logprob_reference_grad(
        logits,
        token_ids,
        grad_logprob,
        temperature,
    )

    reference_logits = logits.detach().requires_grad_(True)
    selected = (
        F.log_softmax(
            reference_logits * (1.0 / temperature).unsqueeze(1),
            dim=-1,
        )
        .gather(1, token_ids.unsqueeze(1))
        .squeeze(1)
    )
    (expected,) = torch.autograd.grad(selected, reference_logits, grad_outputs=grad_logprob)
    assert torch.equal(actual, expected)


def test_dsv4_identity_reference_gradient_is_unchanged_beside_filtered_row() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0], [2.0, 0.0, -1.0]], dtype=torch.float32)
    token_ids = torch.tensor([0, 0], dtype=torch.int64)
    grad_logprob = torch.tensor([0.5, -0.75], dtype=torch.float32)
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)
    support = torch.tensor([[True, True, True], [True, False, False]])
    identity_rows = torch.tensor([True, False])

    actual = _selected_logprob_reference_grad_partitioned(
        logits,
        token_ids,
        grad_logprob,
        temperature,
        support,
        identity_rows,
    )
    native = _selected_logprob_reference_grad(logits[:1], token_ids[:1], grad_logprob[:1], temperature[:1])
    filtered = _selected_logprob_reference_grad_filtered(
        logits[1:], token_ids[1:], grad_logprob[1:], temperature[1:], support[1:]
    )

    assert torch.equal(actual[0], native[0])
    assert torch.equal(actual[1], filtered[0])


def test_dsv4_variable_row_gather_keeps_temperature_in_logical_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_counts = (1, 0, 2, 0, 1, 0, 0, 1)
    padded_rows = 2

    def fake_all_gather(output, _value, *, group):
        assert group == "tp8"
        for rank in range(8):
            output[rank * padded_rows : (rank + 1) * padded_rows] = torch.tensor(
                [rank + 0.25, rank + 0.75],
                dtype=torch.float32,
            )

    monkeypatch.setattr(exact_head.dist, "all_gather_into_tensor", fake_all_gather)
    gathered = _rank_order_variable_row_all_gather(
        torch.tensor([0.7], dtype=torch.float32),
        "tp8",
        row_counts=row_counts,
        padded_rows=padded_rows,
    )

    assert torch.equal(
        gathered,
        torch.tensor([0.25, 2.25, 2.75, 4.25, 7.25], dtype=torch.float32),
    )


def test_dsv4_custom_boundary_carries_temperature_through_adapter_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = {}
    group = object()

    class FakeComponent:
        source_ordinal = 0

        @staticmethod
        def _validate_tp_group():
            return group

        @staticmethod
        def _exact_forward_value(hidden, weight, effective_a, effective_b, token_ids, temperature):
            captures["a"] = effective_a.clone()
            captures["b"] = effective_b.clone()
            captures["forward_temperature"] = temperature.clone()
            return hidden.float().sum(dim=-1) + weight.float().sum() * 0 + token_ids.float() * 0

        @staticmethod
        def _surrogate_vjp(
            hidden,
            weight,
            effective_a,
            effective_b,
            token_ids,
            grad_logprob,
            temperature,
            *,
            needs_input_grad,
        ):
            del weight, token_ids, needs_input_grad
            captures["backward_temperature"] = temperature.clone()
            return (
                grad_logprob.unsqueeze(1).expand_as(hidden).float(),
                torch.ones_like(effective_a, dtype=torch.float32),
                torch.ones_like(effective_b, dtype=torch.float32),
            )

    monkeypatch.setattr(exact_head, "_rank_order_row_counts", lambda *_args: (2, 0, 0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(exact_head, "_rank_order_variable_row_all_gather", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(exact_head.dist, "get_rank", lambda _group: 0)

    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16).requires_grad_(True)
    weight = torch.zeros(5, 3, dtype=torch.bfloat16)
    lora_a = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32, requires_grad=True)
    lora_b = torch.tensor([[0.1], [-0.2], [0.3], [-0.4], [0.5]], dtype=torch.float32, requires_grad=True)
    token_ids = torch.tensor([0, 4], dtype=torch.int64)
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)

    logprob = _Dsv4ExactDistributedHeadFunction.apply(
        hidden,
        weight,
        lora_a,
        lora_b,
        token_ids,
        temperature,
        (None, None, None),
        FakeComponent(),
    )
    logprob.sum().backward()

    assert torch.equal(captures["a"], lora_a.detach().to(torch.bfloat16))
    assert torch.equal(captures["b"], lora_b.detach().to(torch.bfloat16))
    assert torch.equal(captures["forward_temperature"], temperature)
    assert torch.equal(captures["backward_temperature"], temperature)
    assert torch.equal(hidden.grad, torch.ones_like(hidden))
    assert torch.equal(lora_a.grad, torch.ones_like(lora_a))
    assert torch.equal(lora_b.grad, torch.ones_like(lora_b))


def test_dsv4_loss_route_preserves_per_row_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    captures = {}

    def fake_exact(_hidden, _weight, _labels, **kwargs):
        captures.update(kwargs)
        return torch.zeros(2, dtype=torch.float32)

    monkeypatch.setattr(exact_head, "dsv4_exact_lm_head_per_token_ce", fake_exact)
    lm_head = nn.Module()
    lm_head._dsv4_exact_tp8_lm_head = True
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)

    compute_per_token_ce(
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((6, 4), dtype=torch.bfloat16),
        torch.tensor([1, 2], dtype=torch.int64),
        -100,
        "compiled",
        tp_group=object(),
        lm_head_fp32=False,
        lm_head=lm_head,
        logprob_temperature=temperature,
    )

    assert captures["logprob_temperature"] is temperature
