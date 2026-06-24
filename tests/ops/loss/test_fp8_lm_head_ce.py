import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.ops.loss.causallm_loss import causallm_loss_function
from xorl.ops.loss.importance_sampling_loss import importance_sampling_loss_function
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


def test_compute_per_token_ce_uses_lm_head_module_when_provided():
    torch.manual_seed(0)
    hidden = torch.randn(5, 4)
    labels = torch.tensor([0, 1, -100, 2, 3])
    module_weight = torch.randn(6, 4)
    raw_weight = torch.zeros_like(module_weight)
    lm_head = CountingHead(module_weight)

    got = compute_per_token_ce(
        hidden,
        raw_weight,
        labels,
        ignore_index=-100,
        ce_mode="compiled",
        num_chunks=2,
        lm_head=lm_head,
    )
    expected = F.cross_entropy(F.linear(hidden, module_weight), labels, reduction="none", ignore_index=-100)

    assert lm_head.calls == 1
    torch.testing.assert_close(got, expected)


def test_compute_per_token_ce_lm_head_fp32_bypasses_module():
    # lm_head_fp32 takes precedence over an FP8 lm_head module: the module must
    # NOT be called, and logits come from the fp32 (master) raw weight.
    torch.manual_seed(0)
    hidden = torch.randn(5, 4)
    labels = torch.tensor([0, 1, -100, 2, 3])
    module_weight = torch.randn(6, 4)
    raw_weight = torch.randn(6, 4)  # distinct master weight
    lm_head = CountingHead(module_weight)

    got = compute_per_token_ce(
        hidden,
        raw_weight,
        labels,
        ignore_index=-100,
        ce_mode="eager",
        num_chunks=2,
        lm_head=lm_head,
        lm_head_fp32=True,
    )
    expected = F.cross_entropy(
        (hidden.float() @ raw_weight.float().t()).float(),
        labels,
        reduction="none",
        ignore_index=-100,
    )

    assert lm_head.calls == 0  # FP8 module bypassed
    torch.testing.assert_close(got, expected)


def test_compute_per_token_ce_lm_head_fp32_bypasses_module_with_tp_group(monkeypatch):
    _patch_identity_tp_collectives(monkeypatch)
    torch.manual_seed(4)
    hidden = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    module_weight = torch.randn(3, 4)
    raw_weight = torch.randn(3, 4, requires_grad=True)
    lm_head = CountingHead(module_weight)

    got = compute_per_token_ce(
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

    assert lm_head.calls == 0  # FP8 module bypassed even under TP
    assert torch.isfinite(got).all()


def test_importance_sampling_loss_threads_lm_head_module_to_ce():
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


def test_compute_per_token_ce_uses_lm_head_module_with_tp_group(monkeypatch):
    calls = _patch_identity_tp_collectives(monkeypatch)
    torch.manual_seed(2)
    hidden = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    weight = torch.randn(3, 4, requires_grad=True)
    lm_head = CountingHead(weight)

    got = compute_per_token_ce(
        hidden,
        torch.zeros_like(weight),
        labels,
        ignore_index=-100,
        ce_mode="compiled",
        num_chunks=1,
        tp_group=object(),
        lm_head=lm_head,
    )
    expected = F.cross_entropy(F.linear(hidden, weight), labels, reduction="none", ignore_index=-100)

    assert lm_head.calls == 1
    assert (2, 4) not in calls
    torch.testing.assert_close(got, expected)


def test_causallm_loss_lm_head_fp32_bypasses_module():
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


def test_causallm_loss_uses_lm_head_module_with_tp_group_and_reduces_hidden_grad(monkeypatch):
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
