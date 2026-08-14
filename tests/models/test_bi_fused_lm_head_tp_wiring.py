from types import SimpleNamespace

import pytest
import torch
from torch import nn

import xorl.models.module_utils as module_utils
import xorl.ops.loss.causallm_loss as causallm_loss_impl
import xorl.ops.loss.per_token_ce as per_token_ce_impl
import xorl.trainers.training_utils as training_utils
from xorl.ops.loss.loss_output import LossOutput


def test_causallm_routes_bi_fused_tp_before_ordinary_vocab_ce(monkeypatch):
    tp_group = object()
    replica_group = object()
    captured = {}
    reduced_groups = []

    def fake_per_token_ce(hidden, weight, labels, ignore_index, ce_mode, *args, **kwargs):
        captured.update(
            hidden=hidden,
            weight=weight,
            labels=labels,
            ignore_index=ignore_index,
            ce_mode=ce_mode,
            tp_group=kwargs["tp_group"],
            bi_fused_vocab_parallel=kwargs["bi_fused_vocab_parallel"],
        )
        return torch.arange(labels.numel(), dtype=torch.float32)

    monkeypatch.setattr(causallm_loss_impl, "compute_per_token_ce", fake_per_token_ce)
    def fake_all_reduce(value, *, group, **_kwargs):
        reduced_groups.append(group)
        value.mul_(2)

    monkeypatch.setattr(causallm_loss_impl.dist, "all_reduce", fake_all_reduce)
    hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    labels = torch.tensor([[1, -100, 3]])

    result = causallm_loss_impl.causallm_loss_function(
        hidden,
        weight,
        labels,
        ce_mode="bi_fused",
        tp_group=tp_group,
        bi_fused_vocab_parallel=True,
        bi_fused_loss_reduce_group=replica_group,
        lm_head_fp32=True,
        return_per_token=True,
    )

    assert captured["tp_group"] is tp_group
    assert captured["bi_fused_vocab_parallel"] is True
    assert captured["ce_mode"] == "bi_fused"
    assert captured["hidden"].shape == (3, 4)
    assert reduced_groups == [tp_group, replica_group, tp_group, replica_group]
    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    assert torch.equal(result.per_token_loss, torch.tensor([[0.0, 1.0, 2.0]]))


def test_causallm_rejects_body_tp_for_bi_fused():
    with pytest.raises(NotImplementedError, match="dedicated vocabulary-sharded LM-head TP"):
        causallm_loss_impl.causallm_loss_function(
            torch.randn(1, 3, 4, dtype=torch.bfloat16),
            torch.randn(6, 4, dtype=torch.bfloat16),
            torch.tensor([[1, -100, 3]]),
            ce_mode="bi_fused",
            tp_group=object(),
            lm_head_fp32=True,
        )


def test_compute_loss_keeps_sharded_bi_fused_tp_and_global_token_scale(monkeypatch):
    tp_group = object()
    replica_group = object()
    ps = SimpleNamespace(
        lm_head_tp_size=2,
        lm_head_tp_group=tp_group,
        lm_head_tp_replica_group=replica_group,
        tp_enabled=False,
    )
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: ps)

    lm_head = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    lm_head._xorl_fsdp_sharded_lm_head_loss = True
    captured = {}

    def fake_get_weight(module, *, fsdp_sharded_loss):
        captured["fsdp_sharded_loss"] = fsdp_sharded_loss
        return module.weight

    def fake_loss(**kwargs):
        captured["loss_kwargs"] = kwargs
        values = kwargs["hidden_states"].new_ones(kwargs["labels"].shape, dtype=torch.float32)
        mask = (kwargs["labels"] != -100).float()
        return LossOutput(loss=kwargs["loss_reducer"](values, mask))

    monkeypatch.setattr(module_utils, "get_lm_head_weight", fake_get_weight)
    monkeypatch.setattr(module_utils, "get_loss_function", lambda _name: fake_loss)

    hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    labels = torch.tensor([[1, -100, 3]])
    result = module_utils.compute_loss(
        lm_head,
        hidden,
        loss_fn_name="causallm_loss",
        loss_fn_inputs={"labels": labels},
        loss_fn_params={
            "ce_mode": "bi_fused",
            "lm_head_fp32": True,
            "fsdp_sharded_lm_head_loss_num_chunks": 4,
            "fsdp_sharded_lm_head_loss_global_valid_tokens": torch.tensor(8),
        },
    )

    assert captured["fsdp_sharded_loss"] is False
    kwargs = captured["loss_kwargs"]
    assert kwargs["tp_group"] is tp_group
    assert kwargs["bi_fused_vocab_parallel"] is True
    assert kwargs["bi_fused_loss_reduce_group"] is replica_group
    assert "fsdp_sharded_lm_head_loss_num_chunks" not in kwargs
    assert "fsdp_sharded_lm_head_loss_global_valid_tokens" not in kwargs
    torch.testing.assert_close(kwargs["loss_reducer"].scale, torch.tensor(8.0))
    torch.testing.assert_close(result.loss, torch.tensor(0.25))


def test_compute_loss_rejects_unsharded_bi_fused_lm_head_tp(monkeypatch):
    ps = SimpleNamespace(
        lm_head_tp_size=2,
        lm_head_tp_group=object(),
        tp_enabled=False,
    )
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: ps)

    lm_head = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires a vocabulary-sharded lm_head"):
        module_utils.compute_loss(
            lm_head,
            torch.randn(1, 3, 4, dtype=torch.bfloat16),
            loss_fn_name="causallm_loss",
            loss_fn_inputs={"labels": torch.tensor([[1, -100, 3]])},
            loss_fn_params={"ce_mode": "bi_fused", "lm_head_fp32": True},
        )


def test_pp_routes_only_sharded_dedicated_bi_fused_lm_head_tp(monkeypatch):
    lm_head = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    labels = torch.tensor([[1, -100, 3]])
    tp_group = object()

    with pytest.raises(RuntimeError, match="requires a vocabulary-sharded lm_head"):
        training_utils._pp_lm_head_ce_sum(
            hidden,
            labels,
            lm_head=lm_head,
            ce_mode="bi_fused",
            tp_group=tp_group,
            lm_head_fp32=True,
        )

    lm_head._xorl_fsdp_sharded_lm_head_loss = True
    captured = {}

    def fake_get_weight(module, *, fsdp_sharded_loss):
        captured["fsdp_sharded_loss"] = fsdp_sharded_loss
        return module.weight

    def fake_per_token_ce(*_args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([1.0, 0.0, 2.0])

    monkeypatch.setattr(module_utils, "get_lm_head_weight", fake_get_weight)
    monkeypatch.setattr(per_token_ce_impl, "compute_per_token_ce", fake_per_token_ce)
    loss_sum = training_utils._pp_lm_head_ce_sum(
        hidden,
        labels,
        lm_head=lm_head,
        ce_mode="bi_fused",
        tp_group=tp_group,
        lm_head_fp32=True,
    )

    assert captured["fsdp_sharded_loss"] is True
    assert captured["tp_group"] is tp_group
    assert captured["bi_fused_vocab_parallel"] is True
    torch.testing.assert_close(loss_sum, torch.tensor(3.0))
