from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import xorl.objectives.causallm_loss as causallm_loss_impl
import xorl.ops.loss.bi_fused_lm_head as bi_fused_lm_head_impl
import xorl.ops.loss.per_token_ce as per_token_ce_impl
import xorl.server.runner.model_runner as model_runner_module
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


class _TinyMarkedBiFusedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.lm_head._xorl_fsdp_sharded_lm_head_loss = True

    def forward(self, input_ids, **kwargs):
        assert set(kwargs) == {"use_cache", "output_hidden_states"}
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


def _make_bi_fused_lm_head_tp_runner(monkeypatch):
    tp_group = object()
    ps = SimpleNamespace(
        lm_head_tp_size=2,
        lm_head_tp_group=tp_group,
        lm_head_tp_replica_group=None,
        tp_enabled=False,
    )
    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(per_token_ce_impl, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(causallm_loss_impl.dist, "all_reduce", lambda *_args, **_kwargs: None)
    routed_groups = []

    def fake_vocab_parallel_ce(
        hidden,
        weight,
        labels,
        group,
        ignore_index=-100,
        *,
        temperature=1.0,
        **_kwargs,
    ):
        routed_groups.append(group)
        logits = hidden.float() @ weight.float().t()
        if isinstance(temperature, torch.Tensor):
            logits = logits / temperature[:, None]
        elif float(temperature) != 1.0:
            logits = logits / float(temperature)
        return F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)

    monkeypatch.setattr(
        bi_fused_lm_head_impl,
        "bi_fused_vocab_parallel_per_token_ce",
        fake_vocab_parallel_ce,
    )
    runner = object.__new__(ModelRunner)
    runner.model = _TinyMarkedBiFusedModel()
    runner.ce_mode = "bi_fused"
    runner.lm_head_fp32 = True
    return runner, tp_group, routed_groups


def test_compute_micro_batch_loss_routes_marked_bi_fused_causallm(monkeypatch):
    runner, tp_group, routed_groups = _make_bi_fused_lm_head_tp_runner(monkeypatch)

    loss, per_token_outputs, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[2, 3, 4]]),
        },
        "causallm_loss",
        {"return_per_token": True},
    )
    loss.backward()

    assert routed_groups == [tp_group]
    assert per_token_outputs["logprobs"].shape == (1, 3)
    assert runner.model.lm_head.weight.grad is not None


def test_marked_ordinary_head_is_exposed_only_for_bi_fused_loss_metadata():
    runner = object.__new__(ModelRunner)
    runner.model = _TinyMarkedBiFusedModel()
    runner.ce_mode = "eager"
    assert runner._get_loss_lm_head_module(runner.model.lm_head) is None

    runner.ce_mode = "bi_fused"
    assert runner._get_loss_lm_head_module(runner.model.lm_head) is runner.model.lm_head


def test_compute_micro_batch_loss_routes_marked_bi_fused_drgrpo_backward(monkeypatch):
    runner, tp_group, routed_groups = _make_bi_fused_lm_head_tp_runner(monkeypatch)

    loss, per_token_outputs, metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "target_tokens": torch.tensor([[2, 3, 4]]),
            "old_logprobs": torch.tensor([[-2.0, -2.1, -1.9]]),
            "advantages": torch.tensor([[0.5, -0.25, 0.75]]),
        },
        "drgrpo",
        {"beta": 0.0},
    )
    loss.backward()

    assert routed_groups == [tp_group]
    assert per_token_outputs["logprobs"].shape == (1, 3)
    assert metrics["valid_tokens"] == 3
    assert runner.model.lm_head.weight.grad is not None


@pytest.mark.parametrize("loss_parallel_enabled", [False, True])
def test_loss_reporting_reduces_once_over_loss_owner_group(monkeypatch, loss_parallel_enabled):
    loss_group = object()
    ps = SimpleNamespace(loss_parallel_enabled=loss_parallel_enabled, loss_group=loss_group)
    monkeypatch.setattr(model_runner_module, "get_parallel_state", lambda: ps)
    reduced_groups = []

    def fake_all_reduce(value, *, group, **_kwargs):
        reduced_groups.append(group)
        value.add_(5.0)

    monkeypatch.setattr(model_runner_module.dist, "all_reduce", fake_all_reduce)
    local_loss = torch.tensor(2.0, requires_grad=True)

    report = ModelRunner._reduce_loss_report(local_loss)

    assert report.item() == (7.0 if loss_parallel_enabled else 2.0)
    assert reduced_groups == ([loss_group] if loss_parallel_enabled else [])
    assert local_loss.item() == 2.0
    assert local_loss.grad is None


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


def test_compute_micro_batch_loss_drgrpo_forwards_logprob_temperature():
    runner = object.__new__(ModelRunner)
    runner.model = _TinyModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False

    micro_batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "target_tokens": torch.tensor([[2, 3]]),
        "logprobs": torch.tensor([[-2.0, -2.1]]),
        "advantages": torch.zeros((1, 2)),
    }

    _loss, raw_outputs, _raw_metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch,
        "drgrpo",
        {"beta": 0.0},
    )
    _loss, temp_outputs, _temp_metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch,
        "drgrpo",
        {"beta": 0.0, "logprob_temperature": 0.7},
    )

    assert not torch.allclose(raw_outputs["logprobs"], temp_outputs["logprobs"])


def test_compute_micro_batch_loss_forwards_per_row_temperatures(monkeypatch):
    runner = object.__new__(ModelRunner)
    runner.model = _TinyModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False
    temperatures = torch.tensor([[0.7, 1.3]], dtype=torch.float32)
    captured = {}

    def fake_drgrpo_loss_function(*, hidden_states, labels, logprob_temperature, **_kwargs):
        captured["temperature"] = logprob_temperature
        zero = hidden_states.float().sum() * 0.0
        return SimpleNamespace(
            loss=zero,
            per_token_logprobs=torch.zeros_like(labels, dtype=torch.float32),
            per_token_loss=torch.zeros_like(labels, dtype=torch.float32),
            metrics={"valid_tokens": labels.numel()},
            metric_ops=None,
        )

    monkeypatch.setattr(model_runner_module, "drgrpo_loss_function", fake_drgrpo_loss_function)
    micro_batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "target_tokens": torch.tensor([[2, 3]]),
        "logprobs": torch.tensor([[-2.0, -2.1]]),
        "advantages": torch.ones((1, 2)),
        "logprob_temperatures": temperatures,
    }

    runner._compute_micro_batch_loss(micro_batch, "drgrpo", {"beta": 0.0})

    assert captured["temperature"] is temperatures


def test_per_row_temperatures_reject_nonunit_scalar_override():
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelRunner._resolve_logprob_temperature(
            {"logprob_temperatures": torch.ones(2, dtype=torch.float32)},
            {"logprob_temperature": 0.7},
        )


def test_compute_micro_batch_loss_forwards_per_row_sampling_transforms(monkeypatch):
    runner = object.__new__(ModelRunner)
    runner.model = _TinyModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False
    top_ks = torch.tensor([[4, 8]], dtype=torch.int64)
    top_ps = torch.tensor([[0.8, 0.9]], dtype=torch.float32)
    min_ps = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    captured = {}

    def fake_drgrpo_loss_function(*, hidden_states, labels, logprob_top_k, logprob_top_p, logprob_min_p, **_kwargs):
        captured.update(top_k=logprob_top_k, top_p=logprob_top_p, min_p=logprob_min_p)
        zero = hidden_states.float().sum() * 0.0
        return SimpleNamespace(
            loss=zero,
            per_token_logprobs=torch.zeros_like(labels, dtype=torch.float32),
            per_token_loss=torch.zeros_like(labels, dtype=torch.float32),
            metrics={"valid_tokens": labels.numel()},
            metric_ops=None,
        )

    monkeypatch.setattr(model_runner_module, "drgrpo_loss_function", fake_drgrpo_loss_function)
    runner._compute_micro_batch_loss(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "target_tokens": torch.tensor([[2, 3]]),
            "logprobs": torch.tensor([[-2.0, -2.1]]),
            "advantages": torch.ones((1, 2)),
            "logprob_top_ks": top_ks,
            "logprob_top_ps": top_ps,
            "logprob_min_ps": min_ps,
        },
        "drgrpo",
        {"beta": 0.0},
    )
    assert captured == {"top_k": top_ks, "top_p": top_ps, "min_p": min_ps}


def test_per_row_sampling_transforms_reject_nonidentity_scalar_override():
    with pytest.raises(ValueError, match="cannot be combined"):
        ModelRunner._resolve_logprob_sampling_transforms(
            {"logprob_top_ks": torch.ones(2, dtype=torch.int64)},
            {"logprob_top_k": 4},
        )


def test_compute_micro_batch_loss_drgrpo_skips_returned_logprobs_when_disabled():
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
        {"beta": 0.0, "return_per_token": False},
    )

    assert loss.isfinite()
    assert per_token_outputs == {}
    assert metrics["valid_tokens"] == 2


def test_compute_micro_batch_loss_drgrpo_keeps_logprobs_for_per_sample_k3():
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
        {"beta": 0.0, "return_per_token": False, "compute_per_sample_k3": True},
    )

    assert loss.isfinite()
    assert per_token_outputs["logprobs"].shape == micro_batch["target_tokens"].shape
    assert metrics["valid_tokens"] == 2


def test_compute_micro_batch_loss_forwards_sampler_prefill_lengths_to_model():
    class _BoundaryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.forward_kwargs = None

        def forward(self, input_ids, **kwargs):
            self.forward_kwargs = kwargs
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    runner = object.__new__(ModelRunner)
    runner.model = _BoundaryModel()
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False
    input_ids = torch.tensor([[1, 2]])

    runner._compute_micro_batch_loss(
        {
            "input_ids": input_ids,
            "labels": torch.tensor([[2, 3]]),
        },
        "causallm_loss",
        {"sampler_prefill_lengths": [4096]},
    )

    boundary = runner.model.forward_kwargs["sampler_prefill_lengths"]
    assert boundary.dtype is torch.int64
    assert boundary.device == input_ids.device
    assert boundary.tolist() == [4096]

    micro_batch_boundary = torch.tensor([1536], dtype=torch.int64)
    runner._compute_micro_batch_loss(
        {
            "input_ids": input_ids,
            "labels": torch.tensor([[2, 3]]),
            "sampler_prefill_lengths": micro_batch_boundary,
        },
        "causallm_loss",
        {},
    )

    assert runner.model.forward_kwargs["sampler_prefill_lengths"] is micro_batch_boundary


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
    runner.commit_forward_backward_completion("policy-a")

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
