from __future__ import annotations

import importlib
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import xorl.models.transformers.glm5.exact_lm_head_qlora as exact_lm_head_impl
from xorl.distributed.torch_parallelize import _exact_lm_head_replicated_params
from xorl.models.module_utils import get_lm_head_weight
from xorl.models.transformers.glm5.exact_lm_head_qlora import Glm52ExactTP16LmHeadLoraLinear
from xorl.ops.loss.per_token_ce import compute_per_token_ce
from xorl.server.runner.model_runner import ModelRunner
from xorl.trainers.training_utils import make_pp_loss_fn


def _tiny_exact_head() -> Glm52ExactTP16LmHeadLoraLinear:
    base = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    return Glm52ExactTP16LmHeadLoraLinear.from_module(base, r=1, lora_alpha=1)


def test_per_token_ce_routes_exact_head_before_generic_tp_and_fp32_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captures = {}

    def _fake_exact(hidden, weight, labels, **kwargs):
        captures.update(kwargs)
        captures["hidden"] = hidden
        captures["weight"] = weight
        captures["labels"] = labels
        return torch.tensor([1.25, 2.5], dtype=torch.float32)

    monkeypatch.setattr(exact_lm_head_impl, "glm52_exact_lm_head_per_token_ce", _fake_exact)
    lm_head = nn.Module()
    lm_head._glm52_exact_tp16_lm_head = True
    tp_group = object()
    hidden = torch.zeros((2, 4), dtype=torch.bfloat16)
    weight = torch.zeros((6, 4), dtype=torch.bfloat16)
    labels = torch.tensor([1, -100], dtype=torch.int64)

    actual = compute_per_token_ce(
        hidden,
        weight,
        labels,
        -100,
        "bi_fused",
        tp_group=tp_group,
        lm_head_fp32=True,
        lm_head=lm_head,
    )

    assert torch.equal(actual, torch.tensor([1.25, 2.5]))
    assert captures["lm_head"] is lm_head
    assert captures["tp_group"] is tp_group
    assert captures["ignore_index"] == -100
    assert captures["ce_mode"] == "bi_fused"
    assert captures["lm_head_fp32"] is True
    assert captures["logprob_temperature"] == 1.0
    assert captures["hidden"] is hidden
    assert captures["weight"] is weight
    assert captures["labels"] is labels


def test_per_token_ce_preserves_exact_per_row_temperature_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captures = {}

    def _fake_exact(_hidden, _weight, _labels, **kwargs):
        captures.update(kwargs)
        return torch.zeros(2, dtype=torch.float32)

    monkeypatch.setattr(exact_lm_head_impl, "glm52_exact_lm_head_per_token_ce", _fake_exact)
    lm_head = nn.Module()
    lm_head._glm52_exact_tp16_lm_head = True
    temperature = torch.tensor([0.7, 1.3], dtype=torch.float32)

    compute_per_token_ce(
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((6, 4), dtype=torch.bfloat16),
        torch.tensor([1, 2], dtype=torch.int64),
        -100,
        "bi_fused",
        tp_group=object(),
        lm_head_fp32=True,
        lm_head=lm_head,
        logprob_temperature=temperature,
    )

    assert captures["logprob_temperature"] is temperature


def test_causallm_exact_head_admits_its_tp_group_and_rejects_z_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    causallm_impl = importlib.import_module("xorl.ops.loss.causallm_loss")
    lm_head = nn.Module()
    lm_head._glm52_exact_tp16_lm_head = True
    hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).to(torch.bfloat16).requires_grad_(True)
    weight = torch.zeros((6, 4), dtype=torch.bfloat16)
    labels = torch.tensor([[1, 2]], dtype=torch.int64)
    monkeypatch.setattr(
        causallm_impl,
        "compute_per_token_ce",
        lambda hidden_states_flat, *args, **kwargs: hidden_states_flat.float().sum(dim=-1),
    )

    result = causallm_impl.causallm_loss_function(
        hidden,
        weight,
        labels,
        ce_mode="bi_fused",
        tp_group=object(),
        lm_head_fp32=True,
        lm_head=lm_head,
        return_per_token=True,
    )
    result.loss.backward()
    assert result.per_token_logprobs.shape == labels.shape
    assert hidden.grad is not None

    with pytest.raises(NotImplementedError, match="does not support Z-loss"):
        causallm_impl.causallm_loss_function(
            hidden.detach(),
            weight,
            labels,
            ce_mode="bi_fused",
            tp_group=object(),
            lm_head_fp32=True,
            lm_head=lm_head,
            z_loss_coef=1e-4,
        )


def test_exact_head_weight_and_server_loss_selector_never_materialize_delta() -> None:
    lm_head = _tiny_exact_head()
    lm_head._xorl_fsdp_sharded_lm_head_loss = True
    assert get_lm_head_weight(lm_head, fsdp_sharded_loss=True) is lm_head.weight

    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(lm_head=lm_head)
    assert runner._get_effective_lm_head_weight() is lm_head.weight
    assert runner._get_loss_lm_head_module(lm_head) is lm_head


def test_exact_head_fsdp_ignores_only_replicated_a() -> None:
    lm_head = _tiny_exact_head()
    lm_head._glm52_exact_replicated_parameter_names = ("lora_A",)

    assert _exact_lm_head_replicated_params(lm_head) == {lm_head.lora_A}
    assert lm_head.lora_B not in _exact_lm_head_replicated_params(lm_head)

    lm_head._glm52_exact_replicated_parameter_names = ("lora_A", "lora_B")
    with pytest.raises(RuntimeError, match="declare only lora_A"):
        _exact_lm_head_replicated_params(lm_head)


def test_pp_exact_head_loss_matches_dispatcher_value_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    def _differentiable_exact(hidden, weight, labels, *, lm_head, ignore_index, **_kwargs):
        safe_labels = labels.clamp_min(0)
        base = (hidden.float() * weight.index_select(0, safe_labels).float()).sum(dim=-1)
        low_rank = (hidden.float() @ lm_head.lora_A.float().t()).squeeze(-1)
        selected_b = lm_head.lora_B.float().index_select(0, safe_labels).squeeze(-1)
        return (base + low_rank * selected_b) * (labels != ignore_index)

    monkeypatch.setattr(exact_lm_head_impl, "glm52_exact_lm_head_per_token_ce", _differentiable_exact)
    direct_head = _tiny_exact_head()
    pp_head = _tiny_exact_head()
    pp_head.load_state_dict(direct_head.state_dict())
    direct_hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    pp_hidden = direct_hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, -100, 3], [2, 4, -100]], dtype=torch.int64)
    tp_group = object()

    direct = compute_per_token_ce(
        direct_hidden.reshape(-1, 4),
        get_lm_head_weight(direct_head),
        labels.reshape(-1),
        ignore_index=-100,
        ce_mode="bi_fused",
        tp_group=tp_group,
        lm_head_fp32=True,
        lm_head=direct_head,
    ).sum()
    pp = make_pp_loss_fn(
        "bi_fused",
        lm_head=pp_head,
        tp_group=tp_group,
        lm_head_fp32=True,
    )(pp_hidden, labels)

    direct.backward()
    pp.backward()
    torch.testing.assert_close(pp, direct, rtol=0, atol=0)
    torch.testing.assert_close(pp_hidden.grad, direct_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(pp_head.lora_A.grad, direct_head.lora_A.grad, rtol=0, atol=0)
    torch.testing.assert_close(pp_head.lora_B.grad, direct_head.lora_B.grad, rtol=0, atol=0)


def test_pp_exact_head_loss_relays_mixed_temperatures_through_real_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = {}

    def _temperature_aware_exact(hidden, weight, labels, *, logprob_temperature, ignore_index, **_kwargs):
        captures["temperature"] = logprob_temperature
        safe_labels = labels.clamp_min(0)
        selected = (hidden.float() * weight.index_select(0, safe_labels).float()).sum(dim=-1)
        return selected / logprob_temperature * (labels != ignore_index)

    # Leave compute_per_token_ce itself unmocked: this regression must exercise
    # its tensor validation and exact-head dispatch rather than only the PP
    # queue plumbing.  Only the official-geometry CUDA lowerer is projected to
    # a tiny differentiable CPU seam.
    monkeypatch.setattr(exact_lm_head_impl, "glm52_exact_lm_head_per_token_ce", _temperature_aware_exact)
    lm_head = _tiny_exact_head()
    hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[1, -100, 3], [2, 4, -100]], dtype=torch.int64)
    temperatures = torch.tensor([[0.5, 1.0, 2.0], [0.8, 1.25, 1.0]], dtype=torch.float32)
    owner = SimpleNamespace(_pp_loss_temperatures=deque([temperatures]))

    actual = make_pp_loss_fn(
        "bi_fused",
        lm_head=lm_head,
        tp_group=object(),
        lm_head_fp32=True,
        loss_owner=owner,
    )(hidden, labels)
    safe_labels = labels.reshape(-1).clamp_min(0)
    expected = (
        (hidden.reshape(-1, 4).float() * lm_head.weight.index_select(0, safe_labels).float()).sum(dim=-1)
        / temperatures.reshape(-1)
        * (labels.reshape(-1) != -100)
    ).sum()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not owner._pp_loss_temperatures
    assert captures["temperature"].dtype == torch.float32
    assert captures["temperature"].is_contiguous()
    assert captures["temperature"].shape == (labels.numel(),)


def test_pp_exact_head_loss_keeps_sharded_weight_and_microbatch_temperature_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_utils = importlib.import_module("xorl.models.module_utils")
    per_token_ce_impl = importlib.import_module("xorl.ops.loss.per_token_ce")
    lm_head = _tiny_exact_head()
    lm_head._xorl_fsdp_sharded_lm_head_loss = True
    sharded_weight = SimpleNamespace(to_local=lambda: lm_head.weight)
    temperatures_seen = []

    def _get_weight(module, *, fsdp_sharded_loss):
        assert module is lm_head
        assert fsdp_sharded_loss is True
        return sharded_weight

    def _capture_ce(hidden, weight, labels, *, logprob_temperature, **_kwargs):
        assert weight is sharded_weight
        temperatures_seen.append(logprob_temperature.clone())
        return hidden.float().sum(dim=-1) * 0

    monkeypatch.setattr(module_utils, "get_lm_head_weight", _get_weight)
    monkeypatch.setattr(per_token_ce_impl, "compute_per_token_ce", _capture_ce)
    first_temperature = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    second_temperature = torch.tensor([[1.25, 2.0]], dtype=torch.float32)
    owner = SimpleNamespace(_pp_loss_temperatures=deque([first_temperature, second_temperature]))
    loss_fn = make_pp_loss_fn("bi_fused", lm_head=lm_head, loss_owner=owner)
    labels = torch.tensor([[1, 2]], dtype=torch.int64)

    loss_fn(torch.randn(1, 2, 4, dtype=torch.bfloat16), labels)
    loss_fn(torch.randn(1, 2, 4, dtype=torch.bfloat16), labels)

    assert not owner._pp_loss_temperatures
    torch.testing.assert_close(temperatures_seen[0], first_temperature.reshape(-1))
    torch.testing.assert_close(temperatures_seen[1], second_temperature.reshape(-1))


def test_cached_pp_exact_loss_reads_the_current_step_scalar_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    per_token_ce_impl = importlib.import_module("xorl.ops.loss.per_token_ce")
    lm_head = _tiny_exact_head()
    owner = SimpleNamespace(_pp_loss_temperatures=deque([None]))
    temperatures_seen = []

    def _capture_ce(hidden, _weight, _labels, *, logprob_temperature, **_kwargs):
        temperatures_seen.append(logprob_temperature)
        return hidden.float().sum(dim=-1) * 0

    monkeypatch.setattr(per_token_ce_impl, "compute_per_token_ce", _capture_ce)
    cached_loss_fn = make_pp_loss_fn("bi_fused", lm_head=lm_head, loss_owner=owner)
    hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
    labels = torch.tensor([[1, 2]], dtype=torch.int64)

    owner._pp_loss_scalar_temperature = 0.7
    cached_loss_fn(hidden, labels)
    owner._pp_loss_temperatures = deque([None])
    owner._pp_loss_scalar_temperature = 1.3
    cached_loss_fn(hidden, labels)

    assert temperatures_seen == [0.7, 1.3]
