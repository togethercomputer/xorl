from __future__ import annotations

import importlib
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
