import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.objectives.causallm_loss import causallm_loss_function
from xorl.objectives.importance_sampling_loss import importance_sampling_loss_function
from xorl.ops.loss.per_token_ce import compute_per_token_ce


vpce = importlib.import_module("xorl.ops.loss.vocab_parallel_cross_entropy")


class CountingHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.linear(x, self.weight)


def test_compute_per_token_ce_selects_lm_head_path_locally_and_under_tp(monkeypatch):
    collective_calls = _patch_identity_tp_collectives(monkeypatch)
    torch.manual_seed(0)
    hidden = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    module_weight = torch.randn(3, 4)
    raw_weight = torch.randn(3, 4)
    lm_head = CountingHead(module_weight)

    module_ce = compute_per_token_ce(
        hidden,
        raw_weight,
        labels,
        ignore_index=-100,
        ce_mode="compiled",
        num_chunks=1,
        lm_head=lm_head,
    )
    expected_module_ce = F.cross_entropy(F.linear(hidden, module_weight), labels, reduction="none")
    assert lm_head.calls == 1
    torch.testing.assert_close(module_ce, expected_module_ce)

    master_ce = compute_per_token_ce(
        hidden,
        raw_weight,
        labels,
        ignore_index=-100,
        ce_mode="eager",
        num_chunks=1,
        lm_head=lm_head,
        lm_head_fp32=True,
    )
    expected_master_ce = F.cross_entropy(hidden.float() @ raw_weight.float().t(), labels, reduction="none")
    assert lm_head.calls == 1
    torch.testing.assert_close(master_ce, expected_master_ce)

    tp_module_ce = compute_per_token_ce(
        hidden,
        torch.zeros_like(raw_weight),
        labels,
        ignore_index=-100,
        ce_mode="compiled",
        num_chunks=1,
        tp_group=object(),
        lm_head=lm_head,
    )
    assert lm_head.calls == 2
    assert (2, 4) not in collective_calls
    torch.testing.assert_close(tp_module_ce, expected_module_ce)

    tp_master_ce = compute_per_token_ce(
        hidden,
        raw_weight,
        labels,
        ignore_index=-100,
        ce_mode="eager",
        num_chunks=1,
        tp_group=object(),
        lm_head=lm_head,
        lm_head_fp32=True,
    )
    assert lm_head.calls == 2
    assert torch.isfinite(tp_master_ce).all()

    _assert_logprob_temperature_threads_through_per_token_and_causallm_losses()
    _assert_loss_dispatchers_lm_head_module_policy(monkeypatch)


def _assert_logprob_temperature_threads_through_per_token_and_causallm_losses():
    torch.manual_seed(2)
    hidden = torch.randn(5, 4)
    labels = torch.tensor([0, 1, -100, 2, 3])
    weight = torch.randn(6, 4)

    per_token_ce = compute_per_token_ce(
        hidden,
        weight,
        labels,
        ignore_index=-100,
        ce_mode="eager",
        logprob_temperature=0.7,
    )
    expected_ce = F.cross_entropy(
        (hidden @ weight.t()).float() / 0.7,
        labels,
        reduction="none",
        ignore_index=-100,
    )

    torch.testing.assert_close(per_token_ce, expected_ce)

    result = causallm_loss_function(
        hidden.unsqueeze(0),
        weight,
        labels.unsqueeze(0),
        ignore_index=-100,
        ce_mode="eager",
        return_per_token=True,
        logprob_temperature=0.7,
    )
    expected_ce = F.cross_entropy(
        (hidden @ weight.t()).float() / 0.7,
        labels,
        reduction="none",
        ignore_index=-100,
    )

    torch.testing.assert_close(result.per_token_logprobs.squeeze(0), -expected_ce)


def _assert_importance_sampling_loss_threads_lm_head_module_to_ce():
    torch.manual_seed(1)
    hidden = torch.randn(1, 5, 4)
    labels = torch.tensor([[0, 1, -100, 2, 3]])
    module_weight = torch.randn(6, 4)
    raw_weight = torch.zeros_like(module_weight)
    lm_head = CountingHead(module_weight)
    old_logprobs = torch.zeros_like(labels, dtype=torch.float32)
    advantages = torch.ones_like(labels, dtype=torch.float32)

    result = importance_sampling_loss_function(
        hidden_states=hidden,
        weight=raw_weight,
        labels=labels,
        old_logprobs=old_logprobs,
        advantages=advantages,
        ce_mode="compiled",
        num_chunks=2,
        return_per_token=True,
        lm_head=lm_head,
    )
    expected_ce = F.cross_entropy(
        F.linear(hidden.reshape(-1, 4), module_weight),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(labels)

    assert lm_head.calls == 1
    torch.testing.assert_close(result.per_token_logprobs, -expected_ce)


def _patch_identity_tp_collectives(monkeypatch):
    calls = []

    def fake_all_reduce(tensor, *args, **kwargs):
        del args, kwargs
        calls.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(vpce.funcol, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(vpce, "_get_vocab_shard_offset", lambda *args, **kwargs: 0)
    return calls


def _assert_causallm_loss_lm_head_fp32_bypasses_module():
    # causallm_loss_function has its OWN use_lm_head_module path (not via
    # compute_per_token_ce); lm_head_fp32 must bypass the FP8 module here too so
    # the per-token logprobs (which drive the K3 metric) come from fp32 weights.
    torch.manual_seed(5)
    hidden = torch.randn(1, 5, 4)
    labels = torch.tensor([[0, 1, -100, 2, 3]])
    module_weight = torch.randn(6, 4)
    raw_weight = torch.randn(6, 4)
    lm_head = CountingHead(module_weight)

    result = causallm_loss_function(
        hidden_states=hidden,
        weight=raw_weight,
        labels=labels,
        ce_mode="eager",
        num_chunks=2,
        return_per_token=True,
        lm_head=lm_head,
        lm_head_fp32=True,
    )
    expected_ce = F.cross_entropy(
        (hidden.reshape(-1, 4).float() @ raw_weight.float().t()).float(),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )

    assert lm_head.calls == 0  # FP8 module bypassed in the causallm per-token path
    torch.testing.assert_close(result.per_token_logprobs.reshape(-1), -expected_ce)


def _assert_loss_dispatchers_lm_head_module_policy(monkeypatch):
    _assert_importance_sampling_loss_threads_lm_head_module_to_ce()
    _assert_causallm_loss_lm_head_fp32_bypasses_module()

    calls = _patch_identity_tp_collectives(monkeypatch)
    torch.manual_seed(3)
    hidden = torch.randn(1, 2, 4)
    labels = torch.tensor([[0, 1]])
    weight = torch.randn(3, 4, requires_grad=True)
    lm_head = CountingHead(weight)
    hidden = hidden.detach().requires_grad_(True)

    result = causallm_loss_function(
        hidden_states=hidden,
        weight=torch.zeros_like(weight),
        labels=labels,
        ce_mode="compiled",
        num_chunks=1,
        tp_group=object(),
        lm_head=lm_head,
    )
    expected = F.cross_entropy(F.linear(hidden.reshape(-1, 4), weight), labels.reshape(-1))
    result.loss.backward()

    assert lm_head.calls == 1
    assert (2, 4) in calls
    torch.testing.assert_close(result.loss, expected)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert lm_head.weight.grad is not None and torch.isfinite(lm_head.weight.grad).all()
