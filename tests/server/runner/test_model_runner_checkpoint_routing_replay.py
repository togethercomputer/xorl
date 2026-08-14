from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import xorl.server.runner.model_runner as model_runner_module
from xorl.models.layers.moe.routing_replay import (
    RoutingReplay,
    get_replay_stage,
    set_r3_mode,
    set_replay_stage,
)
from xorl.server.runner.model_runner import ModelRunner


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _RoutingHandler:
    def __init__(self, *, r3_enabled: bool = False):
        self.r3_enabled = r3_enabled
        self.cleanup_calls = 0

    def setup(self, *_args, **_kwargs):
        if self.r3_enabled:
            set_r3_mode(True)
            set_replay_stage("replay_backward")
        return self.r3_enabled

    def cleanup(self):
        self.cleanup_calls += 1
        set_replay_stage(None)
        RoutingReplay.clear_all()
        set_r3_mode(False)


class _CheckpointedLossModel(torch.nn.Module):
    def __init__(self, replay, observed_stages, observed_routes):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.replay = replay
        self.observed_stages = observed_stages
        self.observed_routes = observed_routes
        self.selected_experts = torch.tensor([[2, 6]], dtype=torch.long)

    def _checkpointed_moe_body(self, hidden_states):
        stage = get_replay_stage()
        self.observed_stages.append(stage)
        if stage == "record":
            self.replay.record(self.selected_experts)
            route = self.selected_experts
        elif stage == "replay_backward":
            route = self.replay.pop_backward()
        else:
            raise AssertionError(f"loss model ran in unexpected routing stage {stage!r}")
        self.observed_routes.append(route.clone())
        return torch.sin(hidden_states) * hidden_states

    def forward(self, input_ids, **kwargs):
        assert set(kwargs) == {"use_cache", "output_hidden_states"}
        hidden_states = checkpoint(
            self._checkpointed_moe_body,
            self.embed(input_ids),
            use_reentrant=False,
        )
        return SimpleNamespace(last_hidden_state=hidden_states)


@pytest.fixture(autouse=True)
def _isolate_routing_replay_state():
    set_replay_stage(None)
    set_r3_mode(False)
    RoutingReplay.clear_all()
    yield
    set_replay_stage(None)
    set_r3_mode(False)
    RoutingReplay.clear_all()


def _checkpoint_runner(monkeypatch, *, r3_enabled: bool, fail_recompute: bool = False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_runner_module, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(model_runner_module, "should_defer_hsdp_all_reduce", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(model_runner_module, "sync_sp_gradients", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_runner_module, "synchronize", lambda: None)
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(sp_grad_sync_group=None, lm_head_tp_size=1),
    )

    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.model = torch.nn.Identity()
    runner.train_config = {}
    runner.model_fwd_context = nullcontext()
    runner.model_bwd_context = nullcontext()
    runner._adapter_manager = None
    runner._use_routing_replay = True
    runner._routing_handler = _RoutingHandler(r3_enabled=r3_enabled)
    runner._accumulated_valid_tokens = {}
    runner._release_index_share_contexts = lambda: None
    runner._count_global_valid_tokens = lambda _micro_batches: torch.tensor(1)
    runner._reduce_loss_report = lambda loss: loss.detach()
    runner._accumulate_loss_metrics = lambda *_args, **_kwargs: None
    runner._finalize_loss_metrics = lambda *_args, **_kwargs: None

    replay = RoutingReplay()
    selected_experts = torch.tensor([[3, 5]], dtype=torch.long)
    observed_stages = []
    observed_routes = []
    if r3_enabled:
        replay.record(selected_experts)
        set_r3_mode(True)
        set_replay_stage("replay_backward")

    def checkpointed(value):
        stage = get_replay_stage()
        observed_stages.append(stage)
        if stage == "record":
            replay.record(selected_experts)
            route = selected_experts
        elif stage == "replay_forward":
            route = replay.pop_forward()
        elif stage == "replay_backward":
            route = replay.pop_backward()
            if fail_recompute:
                raise RuntimeError("injected checkpoint recompute failure")
        else:
            raise AssertionError(f"checkpointed MoE body ran in unexpected routing stage {stage!r}")
        observed_routes.append(route.clone())
        return torch.sin(value) * value

    def compute_micro_batch_loss(micro_batch, loss_fn, _params, **_kwargs):
        assert loss_fn == "drgrpo"
        output = checkpoint(checkpointed, micro_batch["payload"], use_reentrant=False)
        return output.sum(), {}, {}, None, output

    runner._compute_micro_batch_loss = compute_micro_batch_loss
    micro_batches = [
        {
            "payload": torch.tensor([0.25, -0.5], requires_grad=True),
            "target_tokens": torch.tensor([[1]], dtype=torch.long),
        }
    ]
    return runner, replay, micro_batches, observed_stages, observed_routes, selected_experts


def test_standard_checkpoint_replay_records_then_replays_and_cleans_up(monkeypatch):
    runner, replay, micro_batches, observed_stages, observed_routes, selected_experts = _checkpoint_runner(
        monkeypatch,
        r3_enabled=False,
    )

    runner._forward_loop(micro_batches, "drgrpo", {"beta": 0.0})

    assert observed_stages == ["record", "replay_backward"]
    assert len(observed_routes) == 2
    torch.testing.assert_close(observed_routes[0], selected_experts)
    torch.testing.assert_close(observed_routes[1], selected_experts)
    assert get_replay_stage() is None
    assert replay.top_indices_list == []
    assert replay.forward_index == replay.backward_index == 0


def test_standard_checkpoint_replay_cleans_up_when_recompute_fails(monkeypatch):
    runner, replay, micro_batches, observed_stages, _observed_routes, _selected_experts = _checkpoint_runner(
        monkeypatch,
        r3_enabled=False,
        fail_recompute=True,
    )

    with pytest.raises(RuntimeError, match="injected checkpoint recompute failure"):
        runner._forward_loop(micro_batches, "drgrpo", {"beta": 0.0})

    assert observed_stages == ["record", "replay_backward"]
    assert get_replay_stage() is None
    assert replay.top_indices_list == []
    assert replay.forward_index == replay.backward_index == 0


@pytest.mark.parametrize("loss_fn", ["causallm_loss", "drgrpo"])
def test_real_causal_and_drgrpo_backward_recompute_with_recorded_routing(monkeypatch, loss_fn):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel_state",
        lambda: SimpleNamespace(tp_enabled=False, lm_head_tp_group=None),
    )
    replay = RoutingReplay()
    observed_stages = []
    observed_routes = []
    model = _CheckpointedLossModel(replay, observed_stages, observed_routes)
    runner = object.__new__(ModelRunner)
    runner.model = model
    runner.ce_mode = "eager"
    runner.lm_head_fp32 = False

    micro_batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
    }
    if loss_fn == "drgrpo":
        micro_batch.update(
            {
                "target_tokens": micro_batch.pop("labels"),
                "old_logprobs": torch.tensor([[-2.0, -2.1, -1.9]]),
                "advantages": torch.tensor([[0.5, -0.25, 0.75]]),
            }
        )

    set_replay_stage("record")
    loss, _per_token, _metrics, _metric_ops, _outputs = runner._compute_micro_batch_loss(
        micro_batch,
        loss_fn,
        {"return_per_token": False, "beta": 0.0},
    )
    set_replay_stage("replay_backward")
    loss.backward()

    assert observed_stages == ["record", "replay_backward"]
    assert len(observed_routes) == 2
    torch.testing.assert_close(observed_routes[0], model.selected_experts)
    torch.testing.assert_close(observed_routes[1], model.selected_experts)
    assert model.embed.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_r3_checkpoint_replay_keeps_forward_then_backward_stages(monkeypatch):
    runner, replay, micro_batches, observed_stages, observed_routes, selected_experts = _checkpoint_runner(
        monkeypatch,
        r3_enabled=True,
    )

    runner._forward_loop(micro_batches, "drgrpo", {"beta": 0.0}, r3_enabled=True)

    assert observed_stages == ["replay_forward", "replay_backward"]
    assert len(observed_routes) == 2
    torch.testing.assert_close(observed_routes[0], selected_experts)
    torch.testing.assert_close(observed_routes[1], selected_experts)
    assert runner._routing_handler.cleanup_calls == 1
    assert get_replay_stage() is None
    assert replay.top_indices_list == []


def test_pp_checkpoint_replay_initializes_outer_stage_and_cleans_up_on_failure():
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=8))
    runner.pp_enabled = True
    runner.train_config = {"pp_variable_seq_lengths": True}
    runner._adapter_manager = None
    runner._use_routing_replay = True
    runner._routing_handler = _RoutingHandler()
    runner._check_not_sleeping = lambda *_args, **_kwargs: None
    runner._validate_single_tenant = lambda *_args, **_kwargs: None
    runner._count_global_valid_tokens = lambda _micro_batches: torch.tensor(1)
    observed_stages = []

    def fail_at_schedule(*_args, **_kwargs):
        observed_stages.append(get_replay_stage())
        raise RuntimeError("stop at PP schedule boundary")

    runner._forward_backward_pp = fail_at_schedule
    micro_batches = [
        {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "target_tokens": torch.tensor([[2]], dtype=torch.long),
        }
    ]

    with pytest.raises(RuntimeError, match="stop at PP schedule boundary"):
        runner._forward_backward_impl(
            micro_batches,
            loss_fn="drgrpo",
            loss_fn_params={"forward_backward_defrag": False},
        )

    assert observed_stages == ["replay_backward"]
    assert runner._routing_handler.cleanup_calls == 1
    assert get_replay_stage() is None
