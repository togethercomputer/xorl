"""Asymmetric post-mutation adapter failure must terminate the torchrun gang."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from xorl.distributed.parallel_state import init_parallel_state
from xorl.server.protocol.operations import OptimStepData
from xorl.server.protocol.orchestrator_runner import RunnerDispatchCommand
from xorl.server.runner.adapters.gradient_finalizer import (
    AdapterGradientCollectiveFailure,
    AdapterGradientMutationFailure,
)
from xorl.server.runner.adapters.gradient_ownership import (
    GradientRepresentation,
    GradientScaleState,
    ParameterOwnershipDeclaration,
    ProducerFamily,
    ReductionAuthority,
    ReductionAxis,
    ReductionDomainPlan,
    ReductionOperation,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner
from xorl.server.runner.runner_dispatcher import RunnerDispatcher


pytestmark = pytest.mark.server


class _AdapterLayer(nn.Module):
    adapter_gradient_producer_family = "module_managed"

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 2, device=device))
        self.lora_B = nn.Parameter(torch.ones(2, 2, device=device))


class _AdapterModel(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.layer = _AdapterLayer(device)


def _session_spec() -> dict:
    return {
        "base_model": "synthetic-fatal-adapter",
        "is_lora": True,
        "lora_config": {"lora_rank": 2, "lora_alpha": 2, "seed": 3},
        "optimizer_config": {
            "type": "adamw",
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "optimizer_dtype": "fp32",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "optimizer_kwargs": {},
        },
    }


def _run_worker() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    init_parallel_state(
        dp_size=1,
        ulysses_size=dist.get_world_size(),
        dp_mode="none",
        cp_fsdp_mode="none",
        device_type="cuda",
    )
    model = _AdapterModel(device)
    root = Path(tempfile.gettempdir()) / f"adapter-gradient-fatal-{os.environ['MASTER_PORT']}"
    manager = LoRAAdapterManager(
        model,
        device=device,
        checkpoint_dir=str(root),
        auto_save_on_eviction=False,
        optimizer_fused=False,
        weight_decay=0.0,
    )
    manager.register_adapter("policy", session_spec=_session_spec(), initialize_fresh=True)
    state = manager.get_adapter_state("policy")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(
                ReductionDomainPlan(
                    ReductionAxis.SEQUENCE_PARALLEL,
                    ReductionAuthority.ADAPTER_FINALIZER,
                    ReductionOperation.SUM,
                    "sequence_parallel",
                ),
            ),
            config_guard_fingerprint="fixed-module-producer-v1",
        )
        for name, layout in state.tensor_layouts.items()
    }
    manager.compile_gradient_ownership_plan(
        "policy",
        declarations,
        model_generation="fatal-model-generation-1",
        adapter_generation="fatal-adapter-generation-1",
        group_memberships={"sequence_parallel": (tuple(range(dist.get_world_size())),)},
        rank=rank,
    )
    manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float(rank + 1))
    manager.capture_gradient_numerators("policy", denominator=1, backward_completed=True)

    if rank == 1:
        parameter = next(iter(state.local_params.values()))

        def _partial_step() -> None:
            parameter.data.add_(1)
            raise RuntimeError("injected asymmetric post-mutation failure")

        state.optimizer.step = _partial_step

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = rank
    try:
        manager.optim_step("policy", lr=1e-2)
    except AdapterGradientMutationFailure as error:
        print("ASYMMETRIC_MUTATION_FAILURE_REACHED_FATAL_BOUNDARY", flush=True)
        dispatcher._terminate_after_adapter_gradient_failure(error)

    if state.publication_eligible:
        print("PUBLICATION_ELIGIBLE_AFTER_ASYMMETRIC_FAILURE", flush=True)
    else:
        print("SUCCESSFUL_RANK_AWAITED_COMMAND_PUBLICATION_COMMIT", flush=True)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least two GPUs")
def test_asymmetric_post_mutation_failure_terminates_gang() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=90,
        extra_env={"XORL_ADAPTER_FATAL_WORKER": "1"},
    )

    assert result.exit_code not in {0, -1}, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "ASYMMETRIC_MUTATION_FAILURE_REACHED_FATAL_BOUNDARY" in combined
    assert "PUBLICATION_ELIGIBLE_AFTER_ASYMMETRIC_FAILURE" not in combined
    assert "SUCCESSFUL_RANK_AWAITED_COMMAND_PUBLICATION_COMMIT" in combined


def _run_pre_rendezvous_worker() -> None:
    """Defect class: rank-asymmetric pre-rendezvous failure must forbid peer commit."""

    dist.init_process_group("gloo", timeout=timedelta(seconds=3))
    rank = dist.get_rank()
    init_parallel_state(
        dp_size=dist.get_world_size(),
        dp_shard_size=dist.get_world_size(),
        device_type="cpu",
    )
    root = Path(tempfile.gettempdir()) / f"adapter-gradient-rendezvous-{os.environ['MASTER_PORT']}"
    model = _AdapterModel(torch.device("cpu"))
    manager = LoRAAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(root),
        auto_save_on_eviction=False,
        optimizer_fused=False,
        weight_decay=0.0,
    )
    manager.register_adapter("policy", session_spec=_session_spec(), initialize_fresh=True)
    state = manager.get_adapter_state("policy")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(
                ReductionDomainPlan(
                    ReductionAxis.SEQUENCE_PARALLEL,
                    ReductionAuthority.ADAPTER_FINALIZER,
                    ReductionOperation.SUM,
                    "failure_fixture_replica",
                ),
            ),
            config_guard_fingerprint="fixed-module-producer-v1",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy",
        declarations,
        model_generation="rendezvous-model-generation-1",
        adapter_generation="rendezvous-adapter-generation-1",
        group_memberships={"failure_fixture_replica": ((0, 1),)},
        rank=rank,
    )
    initial = {name: parameter.detach().clone() for name, parameter in state.local_params.items()}
    assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float(rank + 1))
    manager.stage_gradient_numerators("policy", denominator=2, backward_completed=True)
    print(f"PRE_RENDEZVOUS_CAPTURE_STAGED_RANK_{rank}", flush=True)
    dist.barrier()

    if rank == 1:
        try:
            raise RuntimeError("injected rank-local failure before completion rendezvous")
        except RuntimeError:
            print("RANK_LOCAL_FAILURE_BEFORE_RENDEZVOUS", flush=True)
        time.sleep(5)
        os._exit(73)

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = rank
    dispatcher.world_size = 2
    dispatcher.cpu_group = dist.group.WORLD
    dispatcher._batch_parallel_rank_and_size = lambda *_args: (rank, 2)
    try:
        dispatcher._completion_rendezvous(
            {},
            [],
            SimpleNamespace(cp_size=1, pp_enabled=False),
            is_rank0=True,
        )
    except AdapterGradientCollectiveFailure:
        assert state.gradient_scratch.capture_staged
        assert state.gradient_scratch.next_capture_ordinal == 0
        assert all(not torch.count_nonzero(value) for value in state.gradient_scratch.numerators.values())
        assert state.global_step == 0
        assert not state.publication_pending
        assert not state.publication_eligible
        for name, parameter in state.local_params.items():
            torch.testing.assert_close(parameter, initial[name])
        print("PEER_STAGED_CAPTURE_NEVER_COMMITTED", flush=True)
        os._exit(74)
    raise AssertionError("completion rendezvous unexpectedly returned without the failed rank")


def _build_post_mutation_runner(label: str) -> tuple[ModelRunner, LoRAAdapterManager]:
    """Build a real manager/ModelRunner pair with one committed raw-gradient epoch."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = _AdapterModel(torch.device("cpu"))
    root = Path(tempfile.gettempdir()) / f"adapter-gradient-{label}-{os.environ['MASTER_PORT']}"
    manager = LoRAAdapterManager(
        model,
        device=torch.device("cpu"),
        checkpoint_dir=str(root),
        auto_save_on_eviction=False,
        optimizer_fused=False,
        weight_decay=0.0,
    )
    manager.register_adapter("policy", session_spec=_session_spec(), initialize_fresh=True)
    state = manager.get_adapter_state("policy")
    declarations = {
        name: ParameterOwnershipDeclaration(
            topology=TopologyFamily.DENSE_REPLICATED,
            producer=ProducerFamily.MODULE_MANAGED,
            representation=GradientRepresentation.FULL_LOGICAL_CONTRIBUTION,
            completed_domains=(),
            pending_domains=(
                ReductionDomainPlan(
                    ReductionAxis.SEQUENCE_PARALLEL,
                    ReductionAuthority.ADAPTER_FINALIZER,
                    ReductionOperation.SUM,
                    "sequence_parallel",
                ),
            ),
            config_guard_fingerprint="fixed-module-producer-v1",
        )
        for name in state.tensor_layouts
    }
    manager.compile_gradient_ownership_plan(
        "policy",
        declarations,
        model_generation=f"{label}-model-generation-1",
        adapter_generation=f"{label}-adapter-generation-1",
        group_memberships={"sequence_parallel": (tuple(range(world_size)),)},
        rank=rank,
    )
    assert manager.begin_gradient_capture("policy", scale_state=GradientScaleState.RAW_NUMERATOR)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float(rank + 1))
    manager.capture_gradient_numerators("policy", denominator=2, backward_completed=True)

    runner = object.__new__(ModelRunner)
    runner.model = model
    runner._adapter_manager = manager
    runner.rank = rank
    runner.world_size = world_size
    runner.train_config = {"max_grad_norm": None}
    runner.lora_config = {"merge_lora_interval": 0}
    runner._accumulated_valid_tokens = {"policy": 2}
    runner._accumulated_active_microbatches = {"policy": 1}
    runner._accumulated_active_voter_total = {"policy": 0}
    runner._use_distsignsgd = False
    runner._check_not_sleeping = lambda _operation: None
    runner._sync_registered_lora_session_spec = lambda _model_id: None
    return runner, manager


def _build_post_mutation_dispatcher(runner: ModelRunner) -> RunnerDispatcher:
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.trainer = runner
    dispatcher.rank = dist.get_rank()
    dispatcher.world_size = dist.get_world_size()
    dispatcher.cpu_group = dist.group.WORLD
    dispatcher._worker_error = None
    dispatcher._running = True
    dispatcher._adapter_coordinator = SimpleNamespace(
        auto_load_if_evicted=lambda _model_id: (False, None),
    )
    return dispatcher


def _run_post_mutation_request(label: str, inject_failure) -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=8))
    rank = dist.get_rank()
    init_parallel_state(
        dp_size=1,
        ulysses_size=dist.get_world_size(),
        dp_mode="none",
        cp_fsdp_mode="none",
        device_type="cpu",
    )
    runner, manager = _build_post_mutation_runner(label)
    state = manager.get_adapter_state("policy")
    inject_failure(rank, runner, manager)
    dispatcher = _build_post_mutation_dispatcher(runner)

    def _terminate(error: BaseException) -> None:
        print(f"{label.upper()}_FATAL_RANK_{rank}:{type(error).__name__}", flush=True)
        if state.poisoned and not state.publication_eligible:
            print(f"{label.upper()}_FAILED_RANK_PUBLICATION_BLOCKED", flush=True)
        os._exit(81 + rank)

    dispatcher._terminate_after_adapter_gradient_failure = _terminate
    request = RunnerDispatchCommand.create(
        "optim_step",
        OptimStepData(lr=1e-2, model_id="policy"),
        request_id=f"{label}-request",
    )
    if rank == 0:
        response = asyncio.run(dispatcher._handle_request_rank0(request))
        print(f"{label.upper()}_EXTERNAL_OPTIM_RESPONSE:{response.success}", flush=True)
        os._exit(90)
    asyncio.run(dispatcher._worker_event_loop())
    os._exit(91)


def _run_model_runner_tail_worker() -> None:
    def _inject(rank: int, runner: ModelRunner, _manager: LoRAAdapterManager) -> None:
        if rank == 1:
            runner._sync_registered_lora_session_spec = lambda _model_id: (_ for _ in ()).throw(
                RuntimeError("injected asymmetric ModelRunner tail failure")
            )

    _run_post_mutation_request("model_runner_tail", _inject)


def _run_publication_commit_worker() -> None:
    def _inject(rank: int, _runner: ModelRunner, manager: LoRAAdapterManager) -> None:
        if rank == 1:
            manager.commit_optimizer_publication = lambda _model_id: (_ for _ in ()).throw(
                RuntimeError("injected asymmetric publication commit failure")
            )

    _run_post_mutation_request("publication_commit", _inject)


@pytest.mark.cpu
def test_pre_rendezvous_rank_failure_prevents_peer_capture_commit_with_bounded_timeout() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=45,
        extra_env={"XORL_ADAPTER_PRE_RENDEZVOUS_WORKER": "1"},
    )

    assert result.exit_code not in {0, -1}, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "PRE_RENDEZVOUS_CAPTURE_STAGED_RANK_0" in combined
    assert "PRE_RENDEZVOUS_CAPTURE_STAGED_RANK_1" in combined
    assert "RANK_LOCAL_FAILURE_BEFORE_RENDEZVOUS" in combined
    assert "PEER_STAGED_CAPTURE_NEVER_COMMITTED" in combined


@pytest.mark.cpu
def test_asymmetric_model_runner_tail_failure_prevents_external_publication() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=45,
        extra_env={"XORL_ADAPTER_MODEL_RUNNER_TAIL_WORKER": "1"},
    )

    assert result.exit_code not in {0, -1}, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "MODEL_RUNNER_TAIL_FATAL_RANK_1:AdapterGradientMutationFailure" in combined
    assert "MODEL_RUNNER_TAIL_FAILED_RANK_PUBLICATION_BLOCKED" in combined
    assert "MODEL_RUNNER_TAIL_EXTERNAL_OPTIM_RESPONSE" not in combined


@pytest.mark.cpu
def test_asymmetric_publication_commit_failure_prevents_external_publication() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=45,
        extra_env={"XORL_ADAPTER_PUBLICATION_COMMIT_WORKER": "1"},
    )

    assert result.exit_code not in {0, -1}, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "PUBLICATION_COMMIT_FATAL_RANK_1:AdapterGradientMutationFailure" in combined
    assert "PUBLICATION_COMMIT_FAILED_RANK_PUBLICATION_BLOCKED" in combined
    assert "PUBLICATION_COMMIT_EXTERNAL_OPTIM_RESPONSE" not in combined


if os.environ.get("XORL_ADAPTER_FATAL_WORKER") == "1":
    _run_worker()
elif os.environ.get("XORL_ADAPTER_PRE_RENDEZVOUS_WORKER") == "1":
    _run_pre_rendezvous_worker()
elif os.environ.get("XORL_ADAPTER_MODEL_RUNNER_TAIL_WORKER") == "1":
    _run_model_runner_tail_worker()
elif os.environ.get("XORL_ADAPTER_PUBLICATION_COMMIT_WORKER") == "1":
    _run_publication_commit_worker()
