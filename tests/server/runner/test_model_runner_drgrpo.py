from types import SimpleNamespace

import pytest
import torch

from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]

if "drgrpo" not in ModelRunner._LOSS_EXCLUDE_KEYS:  # pragma: no cover - upstream WIP gap
    pytest.skip(
        "model_runner has no Dr.GRPO loss dispatch branch or _LOSS_EXCLUDE_KEYS['drgrpo'] entry upstream; "
        "the drgrpo loss path is not yet implemented, so these tests cannot exercise it",
        allow_module_level=True,
    )


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def forward(self, input_ids, **kwargs):
        assert set(kwargs) == {"use_cache", "output_hidden_states"}
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _NoopRoutingHandler:
    def __init__(self):
        self.calls = []

    def setup(self, micro_batches, routed_experts, routed_expert_logits):
        self.calls.append((micro_batches, routed_experts, routed_expert_logits))
        return False


def test_compute_micro_batch_loss_dispatches_drgrpo_and_filters_loss_inputs():
    runner = object.__new__(ModelRunner)
    runner.model = _TinyModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False

    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "target_tokens": torch.tensor([[2, 3, 4]]),
        "old_logprobs": torch.tensor([[-2.0, -2.1, -1.9]]),
        "advantages": torch.tensor([[0.5, -0.25, 0.75]]),
        "ref_logprobs": torch.tensor([[-2.2, -2.0, -2.4]]),
    }

    loss, per_token_outputs, metrics, metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch,
        "drgrpo",
        {
            "beta": 0.05,
            "clip_low": 0.1,
            "clip_high": 0.2,
            "ratio_type": "sequence",
            "kl_type": "low_var_kl",
        },
    )

    assert loss.isfinite()
    assert per_token_outputs["logprobs"].shape == micro_batch["target_tokens"].shape
    assert metrics["valid_tokens"] == 3
    assert "loss/kl_ref/mean" in metrics
    assert metric_ops is None


def test_compute_micro_batch_loss_drgrpo_accepts_legacy_logprobs_key():
    runner = object.__new__(ModelRunner)
    runner.model = _TinyModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False

    micro_batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "target_tokens": torch.tensor([[2, 3]]),
        "logprobs": torch.tensor([[-2.0, -2.1]]),
        "advantages": torch.tensor([[1.0, 1.0]]),
    }

    loss, per_token_outputs, metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch,
        "drgrpo",
        {"beta": 0.0},
    )

    assert loss.isfinite()
    assert per_token_outputs["logprobs"].shape == micro_batch["target_tokens"].shape
    assert metrics["valid_tokens"] == 2


def test_forward_backward_dispatches_drgrpo_through_standard_loop(monkeypatch):
    monkeypatch.setattr("xorl.server.runner.model_runner.synchronize", lambda: None)

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=16))
    runner.pp_enabled = False
    runner._allocator_dirty = False
    runner._adapter_manager = None
    runner._routing_handler = _NoopRoutingHandler()
    runner._moe_tracker = SimpleNamespace(enabled=False)
    runner._check_not_sleeping = lambda *_args, **_kwargs: None
    runner._validate_single_tenant = lambda *_args, **_kwargs: None
    runner.global_forward_backward_step = 7
    captured = {}

    def fake_forward_loop(micro_batches, loss_fn, loss_fn_params, **kwargs):
        captured["micro_batches"] = micro_batches
        captured["loss_fn"] = loss_fn
        captured["loss_fn_params"] = loss_fn_params
        captured["kwargs"] = kwargs
        return {"total_loss": 0.25, "global_valid_tokens": 2}

    runner._forward_loop = fake_forward_loop
    micro_batches = [
        {
            "input_ids": torch.tensor([[1, 2]]),
            "target_tokens": torch.tensor([[2, 3]]),
            "old_logprobs": torch.tensor([[-2.0, -2.1]]),
            "advantages": torch.tensor([[1.0, 1.0]]),
        }
    ]
    params = {"beta": 0.0, "ratio_type": "sequence"}

    result = runner.forward_backward(micro_batches, loss_fn="drgrpo", loss_fn_params=params, model_id="policy-a")

    assert captured["micro_batches"] is micro_batches
    assert captured["loss_fn"] == "drgrpo"
    assert captured["loss_fn_params"] is params
    assert captured["kwargs"]["compute_backward"] is True
    assert captured["kwargs"]["r3_enabled"] is False
    assert captured["kwargs"]["model_id"] == "policy-a"
    assert result["total_loss"] == pytest.approx(0.25)
    assert result["global_valid_tokens"] == 2
    assert result["step"] == 7
    assert result["model_id"] == "policy-a"
    assert runner.global_forward_backward_step == 8
    assert runner._routing_handler.calls == [(micro_batches, None, None)]
