import torch
from torch import nn

import xorl.models.module_utils as module_utils
from xorl.models.module_utils import compute_loss
from xorl.ops.loss.loss_output import LossOutput


class _ParallelState:
    sp_group = object()
    fsdp_group = object()
    lm_head_tp_group = object()
    lm_head_tp_replica_group = object()
    loss_group = object()
    lm_head_tp_size = 2
    tp_enabled = False
    tp_group = object()


def test_sharded_lm_head_loss_empty_cache_by_default(monkeypatch):
    calls = []
    captured = {}

    monkeypatch.setattr(module_utils, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: _ParallelState())

    def fake_sharded_loss(**kwargs):
        captured.update(kwargs)
        return LossOutput(loss=torch.tensor(1.0))

    monkeypatch.setattr(module_utils, "fsdp_sharded_causallm_loss_function", fake_sharded_loss)

    lm_head = nn.Linear(4, 8, bias=False)
    lm_head._xorl_fsdp_sharded_lm_head_loss = True
    last_hidden_state = torch.randn(1, 2, 4)
    labels = torch.tensor([[1, 2]])

    result = compute_loss(
        lm_head,
        last_hidden_state,
        loss_fn_name=None,
        loss_fn_inputs={"labels": labels},
        loss_fn_params={
            "ce_mode": "compiled",
            "fsdp_sharded_lm_head_loss_num_chunks": 4,
            "fsdp_sharded_lm_head_loss_global_valid_tokens": torch.tensor(2.0),
        },
    )

    assert result.loss.item() == 1.0
    assert calls == ["empty_cache"]
    assert captured["num_chunks"] == 4
    assert captured["sequence_group"] is _ParallelState.lm_head_tp_group
    assert captured["vocab_group"] is _ParallelState.lm_head_tp_group
    # The per-cp_replica losses are summed (divisor=1) over the replica dim, not the
    # full SP group; the matching weight-grad sum is sync_lm_head_tp_gradient's job.
    assert captured["loss_reduce_group"] is _ParallelState.lm_head_tp_replica_group
    assert captured["loss_reduce_divisor"] == 1.0


def test_sharded_lm_head_loss_can_disable_empty_cache(monkeypatch):
    calls = []

    monkeypatch.setattr(module_utils, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(module_utils, "get_parallel_state", lambda: _ParallelState())
    monkeypatch.setattr(
        module_utils,
        "fsdp_sharded_causallm_loss_function",
        lambda **_: LossOutput(loss=torch.tensor(1.0)),
    )

    lm_head = nn.Linear(4, 8, bias=False)
    lm_head._xorl_fsdp_sharded_lm_head_loss = True

    compute_loss(
        lm_head,
        torch.randn(1, 2, 4),
        loss_fn_name=None,
        loss_fn_inputs={"labels": torch.tensor([[1, 2]])},
        loss_fn_params={
            "fsdp_sharded_lm_head_loss_global_valid_tokens": torch.tensor(2.0),
            "empty_cache_before_sharded_lm_head_loss": False,
        },
    )

    assert calls == []
