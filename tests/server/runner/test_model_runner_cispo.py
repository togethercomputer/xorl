from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from xorl.server.runner import model_runner as model_runner_module
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=8)
        self.embed = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _NoopRoutingHandler:
    def __init__(self):
        self.calls = []

    def setup(self, micro_batches, routed_experts, routed_expert_logits):
        self.calls.append((micro_batches, routed_experts, routed_expert_logits))
        return False


def _make_runner(*, pp_enabled=False):
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model = _TinyModel()
    runner.pp_enabled = pp_enabled
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False
    runner.model_fwd_context = nullcontext()
    runner._adapter_manager = None
    runner._routing_handler = _NoopRoutingHandler()
    runner._moe_tracker = SimpleNamespace(enabled=False)
    runner._check_not_sleeping = lambda *_args, **_kwargs: None
    runner._validate_single_tenant = lambda *_args, **_kwargs: None
    runner._index_share_forward_kwargs = lambda *_args, **_kwargs: {}
    runner._get_loss_tp_group = lambda: None
    runner._compute_token_diagnostics = lambda **_kwargs: None
    runner.global_forward_backward_step = 0
    return runner


def _micro_batch():
    return {
        "input_ids": torch.tensor([[1, 2]]),
        "target_tokens": torch.tensor([[2, 3]]),
        "logprobs": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
    }


def test_cispo_compute_ref_logprobs_replaces_behavior_logprobs(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)
    monkeypatch.setattr(model_runner_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=False),
    )

    captured = {}

    def fake_forward_loop(micro_batches, loss_fn, loss_fn_params, **_kwargs):
        captured["old_logprobs"] = micro_batches[0]["logprobs"].clone()
        loss, outputs, metrics, _metric_ops, _model_outputs = runner._compute_micro_batch_loss(
            micro_batches[0],
            loss_fn,
            loss_fn_params,
        )
        captured["new_logprobs"] = outputs["logprobs"]
        captured["metrics"] = metrics
        return {"total_loss": float(loss.detach() / 2), "global_valid_tokens": 2}

    runner._forward_loop = fake_forward_loop
    original_logprobs = _micro_batch()["logprobs"].clone()
    micro_batches = [_micro_batch()]

    runner._forward_backward_impl(
        micro_batches,
        loss_fn="cispo",
        loss_fn_params={
            "compute_ref_logprobs": True,
            "compute_kl_stats": True,
            "forward_backward_defrag": False,
        },
    )

    assert not torch.equal(captured["old_logprobs"], original_logprobs)
    torch.testing.assert_close(captured["new_logprobs"], captured["old_logprobs"], rtol=0, atol=0)
    assert captured["metrics"]["ratio_mean"] == 2
    assert captured["metrics"]["kl_sample_train_k3"] == 0


def test_cispo_reduces_always_on_metrics_across_cp_without_kl_stats(monkeypatch):
    runner = _make_runner()
    ulysses_group = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(cp_enabled=True, ulysses_group=ulysses_group),
    )
    calls = []

    def fake_sp_reduce(metrics, group, metric_ops):
        calls.append((dict(metrics), group, dict(metric_ops)))
        return metrics

    monkeypatch.setattr(model_runner_module, "_sp_allreduce_kl_metrics", fake_sp_reduce)

    runner._compute_micro_batch_loss(
        _micro_batch(),
        "cispo",
        {"compute_kl_stats": False},
    )

    assert len(calls) == 1
    metrics, group, metric_ops = calls[0]
    assert group is ulysses_group
    assert {"ratio_mean", "ratio_min", "ratio_max", "clip_fraction", "valid_tokens"} <= metrics.keys()
    assert metric_ops == {"ratio_min": "min", "ratio_max": "max"}
