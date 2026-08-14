from types import SimpleNamespace

import pytest
import torch

import xorl.server.runner.model_runner as model_runner_module
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def _runner() -> ModelRunner:
    runner = object.__new__(ModelRunner)
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False
    runner.pp_enabled = True
    runner.pp_num_stages = 1
    runner.train_config = {}
    part = torch.nn.Module()
    part.lm_head = torch.nn.Linear(8, 16, bias=False)
    runner.model = part
    runner.model_parts = [part]
    runner.pp_stages = [SimpleNamespace(stage_index=0)]
    return runner


def _micro_batch(*, per_row_temperature: bool = True) -> dict[str, torch.Tensor]:
    labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.long)
    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": labels,
        "target_tokens": labels.clone(),
        "logprobs": torch.zeros_like(labels, dtype=torch.float32),
        "old_logprobs": torch.zeros_like(labels, dtype=torch.float32),
        "ref_logprobs": torch.zeros_like(labels, dtype=torch.float32),
        "rollout_logprobs": torch.zeros_like(labels, dtype=torch.float32),
        "advantages": torch.tensor([[0.0, 1.0, -0.5, 0.25]], dtype=torch.float32),
    }
    if per_row_temperature:
        micro_batch["logprob_temperatures"] = torch.ones_like(labels, dtype=torch.float32)
    return micro_batch


@pytest.mark.parametrize(
    ("loss_fn", "params"),
    [
        ("causallm_loss", {"return_per_token": True}),
        ("cross_entropy", {"return_per_token": True}),
        ("importance_sampling", {"compute_kl_stats": True}),
        ("cispo", {"compute_kl_stats": True, "clip_low_threshold": 0.2, "clip_high_threshold": 3.0}),
        ("drgrpo", {"return_per_token": True, "beta": 0.1}),
        ("policy_loss", {"return_per_token": True, "compute_kl_stats": True}),
    ],
)
def test_physical_pp_dispatcher_runs_requested_real_objective_and_backward(monkeypatch, loss_fn, params):
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            tp_enabled=False,
            cp_enabled=False,
            lm_head_tp_group=None,
        ),
    )
    runner = _runner()
    dispatcher = runner._make_pp_train_loss_fn()
    assert dispatcher is runner._make_pp_train_loss_fn()
    dispatcher.begin_step(
        [_micro_batch(per_row_temperature=False)],
        loss_fn=loss_fn,
        loss_fn_params=params,
        model_id="policy-a",
        assert_consumed=True,
    )
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    loss = dispatcher(hidden, torch.tensor([0], dtype=torch.long))
    assert loss.isfinite()
    loss.backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()
    records = dispatcher.end_step()
    assert len(records) == 1
    assert records[0]["microbatch_id"] == 0
    assert isinstance(records[0]["per_token_outputs"].get("logprobs"), torch.Tensor)
    assert not dispatcher.active


def test_physical_pp_dispatcher_resolves_objective_per_step_and_consumes_ids_once(monkeypatch):
    runner = _runner()
    observed = []

    def fake_compute(hidden, metadata):
        observed.append((metadata.loss_fn, metadata.model_id, tuple(metadata.micro_batch["labels"].shape)))
        return hidden.sum(), {}, {"valid_tokens": 1}, None

    monkeypatch.setattr(runner, "_compute_pp_terminal_objective", fake_compute)
    dispatcher = runner._make_pp_train_loss_fn()
    for objective, model_id in (("policy_loss", "a"), ("cispo", "b")):
        dispatcher.begin_step(
            [_micro_batch()],
            loss_fn=objective,
            loss_fn_params={"eps_clip": 0.1},
            model_id=model_id,
            assert_consumed=True,
        )
        dispatcher(torch.ones(1, 4, 8, requires_grad=True), torch.tensor([0]))
        with pytest.raises(RuntimeError, match="more than once"):
            dispatcher(torch.ones(1, 4, 8), torch.tensor([0]))
        assert len(dispatcher.end_step()) == 1

    assert observed == [("policy_loss", "a", (1, 4)), ("cispo", "b", (1, 4))]


def test_physical_pp_dispatcher_fails_when_schedule_skips_or_mixes_ids():
    runner = _runner()
    dispatcher = runner._make_pp_train_loss_fn()
    dispatcher.begin_step(
        [_micro_batch(), _micro_batch()],
        loss_fn="causallm_loss",
        loss_fn_params={},
        model_id="a",
        assert_consumed=True,
    )
    with pytest.raises(RuntimeError, match="mixed multiple"):
        dispatcher(torch.ones(1, 4, 8), torch.tensor([0, 1]))
    with pytest.raises(RuntimeError, match="exactly once"):
        dispatcher.end_step()
    assert not dispatcher.active


def test_physical_pp_rejects_only_intermediate_capture_objectives():
    ModelRunner._validate_physical_pp_objective("opd_loss", {})
    with pytest.raises(NotImplementedError, match="OPRD/intermediate-layer"):
        ModelRunner._validate_physical_pp_objective("opd_loss", {"opd_oprd_enabled": True})
    with pytest.raises(NotImplementedError, match="intermediate activation"):
        ModelRunner._validate_physical_pp_objective("teacher_hidden_cache", {})


def test_physical_pp_sampling_transform_fallback_uses_frozen_identity():
    runner = _runner()
    assert runner._pp_sampling_transform_kwargs({}, {"logprob_top_k": 1 << 30}) == {}
    with pytest.raises(RuntimeError, match="before the exact sampling-transform"):
        runner._pp_sampling_transform_kwargs({}, {"logprob_top_k": 2**31 - 1})


def test_cached_physical_pp_schedule_captures_only_stable_dispatcher(monkeypatch):
    runner = _runner()
    runner._pp_schedule_cache = {}
    built_loss_fns = []
    schedule = object()
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(pp_group="pp"),
    )
    monkeypatch.setattr(model_runner_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(model_runner_module, "build_pp_stage", lambda *_args, **_kwargs: object())

    def fake_build_pipeline_schedule(*, stages, n_microbatches, loss_fn, schedule_name):
        assert len(stages) == 1
        assert n_microbatches == 1
        assert schedule_name == "1F1B"
        built_loss_fns.append(loss_fn)
        return schedule

    monkeypatch.setattr(model_runner_module, "build_pipeline_schedule", fake_build_pipeline_schedule)
    dispatcher = runner._make_pp_train_loss_fn()

    first = runner._get_pp_schedule(1, 4, loss_fn=dispatcher)
    dispatcher.begin_step(
        [_micro_batch()],
        loss_fn="policy_loss",
        loss_fn_params={},
        model_id="policy-a",
        assert_consumed=False,
    )
    dispatcher.abort_step()
    second = runner._get_pp_schedule(1, 4, loss_fn=dispatcher)

    assert first is schedule and second is schedule
    assert built_loss_fns == [dispatcher]


def test_physical_pp_objective_composes_future_per_row_sampling_resolver(monkeypatch):
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(tp_enabled=False, cp_enabled=False, lm_head_tp_group=None),
    )
    runner = _runner()
    micro_batch = _micro_batch(per_row_temperature=False)
    micro_batch.update(
        logprob_top_ks=torch.tensor([[4, 8, 16, 1 << 30]], dtype=torch.int64),
        logprob_top_ps=torch.tensor([[0.8, 0.9, 1.0, 1.0]], dtype=torch.float32),
        logprob_min_ps=torch.tensor([[0.0, 0.01, 0.02, 0.0]], dtype=torch.float32),
    )
    captured = {}

    def resolve_sampling(mb, _params):
        return {
            "logprob_top_k": mb["logprob_top_ks"],
            "logprob_top_p": mb["logprob_top_ps"],
            "logprob_min_p": mb["logprob_min_ps"],
        }

    def fake_importance_sampling_loss_function(*, hidden_states, labels, **kwargs):
        captured.update(
            top_k=kwargs["logprob_top_k"].detach().cpu(),
            top_p=kwargs["logprob_top_p"].detach().cpu(),
            min_p=kwargs["logprob_min_p"].detach().cpu(),
        )
        return SimpleNamespace(
            loss=hidden_states.sum(),
            per_token_logprobs=torch.zeros_like(labels, dtype=torch.float32),
            metrics={"valid_tokens": int(labels.numel())},
            metric_ops={},
        )

    runner._resolve_logprob_sampling_transforms = resolve_sampling
    monkeypatch.setattr(
        model_runner_module,
        "importance_sampling_loss_function",
        fake_importance_sampling_loss_function,
    )
    dispatcher = runner._make_pp_train_loss_fn()
    dispatcher.begin_step(
        [micro_batch],
        loss_fn="importance_sampling",
        loss_fn_params={},
        model_id="policy-a",
        assert_consumed=True,
    )
    dispatcher(torch.ones(1, 4, 8, requires_grad=True), torch.tensor([0]))
    dispatcher.end_step()

    assert torch.equal(captured["top_k"], micro_batch["logprob_top_ks"])
    assert torch.equal(captured["top_p"], micro_batch["logprob_top_ps"])
    assert torch.equal(captured["min_p"], micro_batch["logprob_min_ps"])
