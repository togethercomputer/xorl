"""Tests for save/session operation handling in RunnerDispatcher."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from xorl.server.protocol.operations import (
    AbortGradientEpochData,
    OptimStepData,
    RegisterSessionData,
    SaveLoraOnlyData,
    SaveStateData,
)
from xorl.server.protocol.orchestrator_runner import RunnerDispatchCommand


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "xorl" / "server" / "runner" / "runner_dispatcher.py"
_SPEC = importlib.util.spec_from_file_location("xorl_test_runner_dispatcher_session_ops", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RunnerDispatcher = _MODULE.RunnerDispatcher


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeAdapterCoordinator:
    def __init__(self):
        self.auto_load_calls = []
        self.register_session_calls = []

    def auto_load_if_evicted(self, model_id: str, *, allow_fresh_materialization: bool = True):
        self.auto_load_calls.append(
            {
                "model_id": model_id,
                "allow_fresh_materialization": allow_fresh_materialization,
            }
        )
        return False, None

    async def handle_register_session(self, command_dict):
        self.register_session_calls.append(command_dict)
        payload = command_dict["payload"]
        return {"registered": True, "model_id": payload.model_id}


class _FakeTrainer:
    def __init__(self):
        self.adapter_manager = object()
        self.lora_config = {"enable_lora": True}
        self.save_state_calls = []
        self.save_lora_only_calls = []
        self.abort_gradient_epoch_calls = []

    def save_state(self, checkpoint_path, save_optimizer=True, model_id=None):
        self.save_state_calls.append((checkpoint_path, save_optimizer, model_id))
        return {"success": True}

    def save_lora_only(self, lora_path, model_id="default"):
        self.save_lora_only_calls.append((lora_path, model_id))
        return {"success": True}

    def abort_gradient_epoch(self, model_id="default"):
        self.abort_gradient_epoch_calls.append(model_id)
        return {"success": True, "model_id": model_id, "step": 2, "forward_backward_step": 7}


class _FakePublicationManager:
    def __init__(self):
        self.commits = []

    def commit_optimizer_publication(self, model_id: str) -> None:
        self.commits.append(model_id)


class _FakeAdapterState:
    def __init__(self):
        self.poisoned = False
        self.publication_eligible = True


class _FailingPublicationManager(_FakePublicationManager):
    def __init__(self):
        super().__init__()
        self.state = _FakeAdapterState()

    def get_adapter_state(self, model_id: str):
        assert model_id == "policy"
        return self.state

    def commit_optimizer_publication(self, model_id: str) -> None:
        super().commit_optimizer_publication(model_id)
        raise RuntimeError("injected publication tail failure")


def test_handle_save_state_requires_real_checkpoint_for_nonresident_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(tmp_path))
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.trainer = _FakeTrainer()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    result = asyncio.run(
        dispatcher._handle_save_state(
            {
                "payload": SaveStateData(
                    checkpoint_path=str(tmp_path / "checkpoint"),
                    save_optimizer=True,
                    model_id="policy-a",
                )
            }
        )
    )

    assert result["checkpoint_path"] == str(tmp_path / "checkpoint")
    assert dispatcher._adapter_coordinator.auto_load_calls == [
        {
            "model_id": "policy-a",
            "allow_fresh_materialization": False,
        }
    ]
    assert dispatcher.trainer.save_state_calls == [(str(tmp_path / "checkpoint"), True, "policy-a")]


def test_handle_save_lora_only_requires_real_checkpoint_for_nonresident_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(tmp_path))
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.trainer = _FakeTrainer()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()

    result = asyncio.run(
        dispatcher._handle_save_lora_only(
            {
                "payload": SaveLoraOnlyData(
                    lora_path=str(tmp_path / "adapter"),
                    model_id="policy-b",
                )
            }
        )
    )

    assert result["lora_path"] == str(tmp_path / "adapter")
    assert dispatcher._adapter_coordinator.auto_load_calls == [
        {
            "model_id": "policy-b",
            "allow_fresh_materialization": False,
        }
    ]
    assert dispatcher.trainer.save_lora_only_calls == [(str(tmp_path / "adapter"), "policy-b")]


def test_handle_register_session_delegates_to_adapter_coordinator():
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()
    payload = RegisterSessionData(
        model_id="policy-c",
        session_spec={
            "base_model": "Qwen/Qwen3-8B",
            "is_lora": True,
            "lora_config": {"lora_rank": 8, "lora_alpha": 16},
            "optimizer_config": {"type": "adamw", "learning_rate": 1e-4},
        },
        materialize=True,
    )

    result = asyncio.run(dispatcher._handle_register_session({"payload": payload}))

    assert result == {"registered": True, "model_id": "policy-c"}
    assert dispatcher._adapter_coordinator.register_session_calls == [{"payload": payload}]


def test_handle_request_rank0_fails_register_session_on_cross_rank_worker_error(monkeypatch):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.world_size = 2
    dispatcher.cpu_group = object()
    dispatcher._worker_error = None

    async def _handle_register_session(command_dict):
        return {"registered": True, "model_id": command_dict["payload"].model_id}

    dispatcher._handle_register_session = _handle_register_session
    dispatcher._sync_error_state = lambda: "rank 1: Session registration failed: boom"

    monkeypatch.setattr(_MODULE.dist, "broadcast_object_list", lambda *args, **kwargs: None)

    request = RunnerDispatchCommand.create(
        "register_session",
        RegisterSessionData(
            model_id="policy-c",
            session_spec={"base_model": "Qwen/Qwen3-8B", "is_lora": True},
            materialize=False,
        ),
        request_id="req-register-session",
    )

    response = asyncio.run(dispatcher._handle_request_rank0(request))

    assert response.success is False
    assert response.error == "Cross-rank error: rank 1: Session registration failed: boom"


def test_handle_request_rank0_terminates_process_after_optimizer_mutation_failure(monkeypatch):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.world_size = 1
    dispatcher.cpu_group = None
    dispatcher._worker_error = None

    async def _handle_optim_step(_command_dict):
        raise _MODULE.AdapterGradientMutationFailure("injected asymmetric optimizer failure")

    def _exit(exit_code):
        raise SystemExit(exit_code)

    dispatcher._handle_optim_step = _handle_optim_step
    monkeypatch.setattr(_MODULE.os, "_exit", _exit)
    request = RunnerDispatchCommand.create(
        "optim_step",
        OptimStepData(lr=1e-4, model_id="policy"),
        request_id="req-fatal-optim-step",
    )

    with pytest.raises(SystemExit) as exited:
        asyncio.run(dispatcher._handle_request_rank0(request))

    assert exited.value.code == _MODULE._ADAPTER_GRADIENT_FATAL_EXIT_CODE


def test_existing_command_completion_commits_adapter_publication_without_another_collective():
    dispatcher = object.__new__(RunnerDispatcher)
    manager = _FakePublicationManager()
    dispatcher.trainer = type("Trainer", (), {"adapter_manager": manager})()
    payload = OptimStepData(lr=1e-4, model_id="policy")

    dispatcher._commit_adapter_optimizer_publication({"payload": payload})

    assert manager.commits == ["policy"]


def test_publication_commit_failure_after_mutation_is_fatal_and_poisons_adapter():
    dispatcher = object.__new__(RunnerDispatcher)
    manager = _FailingPublicationManager()
    dispatcher.trainer = type("Trainer", (), {"adapter_manager": manager})()
    payload = OptimStepData(lr=1e-4, model_id="policy")

    with pytest.raises(_MODULE.AdapterGradientMutationFailure, match="publication commit failed") as raised:
        dispatcher._commit_adapter_optimizer_publication({"payload": payload})

    assert str(raised.value.__cause__) == "injected publication tail failure"
    assert manager.commits == ["policy"]
    assert manager.state.poisoned is True
    assert manager.state.publication_eligible is False


def test_optimizer_handler_tail_failure_after_mutation_is_fatal_and_poisons_adapter():
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher.world_size = 1
    manager = _FailingPublicationManager()

    class _ExplodingResult(dict):
        def get(self, key, default=None):
            raise RuntimeError("injected handler tail failure")

    class _Trainer:
        adapter_manager = manager

        def optim_step(self, **_kwargs):
            return _ExplodingResult()

    dispatcher.trainer = _Trainer()
    dispatcher._adapter_coordinator = _FakeAdapterCoordinator()
    payload = OptimStepData(
        lr=1e-4,
        model_id="policy",
        sparse_delta_capture={"enabled": True},
    )

    with pytest.raises(_MODULE.AdapterGradientMutationFailure, match="handler tail failed") as raised:
        asyncio.run(dispatcher._handle_optim_step({"payload": payload}))

    assert str(raised.value.__cause__) == "injected handler tail failure"
    assert manager.state.poisoned is True
    assert manager.state.publication_eligible is False


def test_abort_gradient_epoch_uses_the_normal_distributed_handler():
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.trainer = _FakeTrainer()

    result = asyncio.run(
        dispatcher._handle_abort_gradient_epoch({"payload": AbortGradientEpochData(model_id="policy-abort")})
    )

    assert result == {
        "success": True,
        "model_id": "policy-abort",
        "step": 2,
        "forward_backward_step": 7,
    }
    assert dispatcher.trainer.abort_gradient_epoch_calls == ["policy-abort"]
    assert "abort_gradient_epoch" in RunnerDispatcher._ERROR_SYNC_OPS


def test_forward_backward_commits_after_completion_rendezvous_before_rank0_merge():
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    events = []

    class _Trainer:
        def commit_forward_backward_completion(self, model_id):
            events.append(("commit", model_id))

    dispatcher.trainer = _Trainer()
    dispatcher._shard_and_slice_batches = lambda batches, routed, logits, cp, ps: (batches, routed, logits)
    dispatcher._maybe_dump_microbatch_diagnostic = lambda *args, **kwargs: None
    dispatcher._execute_compute = lambda *args, **kwargs: {"packed_logprobs": [[-1.0]]}
    dispatcher._gather_is_metrics = lambda *args, **kwargs: events.append(("metrics", None))

    def _rendezvous(*args, **kwargs):
        events.append(("rendezvous", None))
        return [{"rank": 0, "slice_rank": 0, "packed_logprobs": [[-1.0]]}]

    dispatcher._completion_rendezvous = _rendezvous
    dispatcher._merge_completion_payloads = lambda result, gathered: events.append(("merge", None))

    result = dispatcher._execute_and_gather(
        [],
        "causallm_loss",
        {},
        None,
        False,
        object(),
        with_backward=True,
        model_id="policy",
        is_rank0=True,
    )

    assert result == {"packed_logprobs": [[-1.0]]}
    assert events == [
        ("metrics", None),
        ("rendezvous", None),
        ("commit", "policy"),
        ("merge", None),
    ]
    assert "forward_backward" not in RunnerDispatcher._ERROR_SYNC_OPS


@pytest.mark.parametrize("rank", (0, 1))
def test_forward_backward_preserves_uniform_rejection_and_promotes_other_failures_to_fatal(rank):
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = rank

    async def _uniform(*_args, **_kwargs):
        raise _MODULE.AdapterGradientUniformRejection("ADAPTER_GRADIENT_ZERO_DENOMINATOR: injected")

    target = "_handle_compute_rank0_scatter" if rank == 0 else "_handle_compute_worker_receive"
    setattr(dispatcher, target, _uniform)
    with pytest.raises(_MODULE.AdapterGradientUniformRejection, match="ZERO_DENOMINATOR"):
        asyncio.run(dispatcher._handle_forward_backward({}))

    async def _asymmetric(*_args, **_kwargs):
        raise RuntimeError("injected rank-local failure")

    setattr(dispatcher, target, _asymmetric)
    with pytest.raises(_MODULE.AdapterGradientCollectiveFailure, match="Rank-asymmetric"):
        asyncio.run(dispatcher._handle_forward_backward({}))
