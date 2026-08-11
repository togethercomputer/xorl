import math
from types import SimpleNamespace

import pytest
import torch

from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def forward(self, input_ids, **kwargs):
        assert set(kwargs) == {"use_cache", "output_hidden_states"}
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


def _loss_runner(model=None):
    runner = object.__new__(ModelRunner)
    runner.model = model or _TinyModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False
    return runner


def _legacy_micro_batch():
    return {
        "input_ids": torch.tensor([[1, 2]]),
        "target_tokens": torch.tensor([[2, 3]]),
        "logprobs": torch.tensor([[-2.0, -2.1]]),
        "advantages": torch.tensor([[1.0, 1.0]]),
    }


def test_compute_micro_batch_loss_dispatches_drgrpo_and_honors_options():
    _assert_causallm_loss_returns_raw_token_sum()
    runner = _loss_runner()

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

    # Legacy field names and output controls use the same dispatch boundary.
    micro_batch = _legacy_micro_batch()

    loss, raw_outputs, metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch, "drgrpo", {"beta": 0.0}
    )
    assert loss.isfinite()
    assert raw_outputs["logprobs"].shape == micro_batch["target_tokens"].shape
    assert metrics["valid_tokens"] == 2

    _loss, temp_outputs, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch, "drgrpo", {"beta": 0.0, "logprob_temperature": 0.7}
    )
    assert not torch.allclose(raw_outputs["logprobs"], temp_outputs["logprobs"])

    _loss, disabled_outputs, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch, "drgrpo", {"beta": 0.0, "return_per_token": False}
    )
    assert disabled_outputs == {}

    _loss, k3_outputs, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch,
        "drgrpo",
        {"beta": 0.0, "return_per_token": False, "compute_per_sample_k3": True},
    )
    assert k3_outputs["logprobs"].shape == micro_batch["target_tokens"].shape


def _assert_causallm_loss_returns_raw_token_sum():
    model = _TinyModel()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    runner = _loss_runner(model)

    loss, per_token_outputs, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        {
            "input_ids": torch.tensor([[0, 0]]),
            "labels": torch.tensor([[0, 1]]),
        },
        "causallm_loss",
        {},
    )

    expected = math.log(model.lm_head.out_features)
    assert loss.item() == pytest.approx(2 * expected)
    assert per_token_outputs["loss"].reshape(-1).tolist() == pytest.approx([expected, expected])
