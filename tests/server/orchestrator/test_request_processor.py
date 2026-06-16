"""
Tests for RequestProcessor with DummyBackend.

This test suite verifies the RequestProcessor's data preparation and result formatting:
1. Sample packing (datum_list -> micro-batches)
2. Operation execution (forward_backward, optim_step, etc.)
3. Output formatting (OrchestratorOutputs)
4. Error handling
5. Statistics tracking

Test Strategy:
- Use DummyBackend (in-process mock, no ZMQ)
- Verify RequestProcessor correctly packs data and formats outputs
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from xorl.server.backend import DummyBackend
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.protocol.api_orchestrator import (
    OrchestratorOutputs,
    OrchestratorRequest,
    OutputType,
    RequestType,
)
from xorl.server.protocol.operations import (
    EmptyData,
    LoadStateData,
    ModelPassData,
    OptimStepData,
    RegisterSessionData,
    SaveStateData,
    SyncWeightsData,
)
from xorl.server.runner.runner_dispatcher import RunnerDispatcher


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def processor():
    """Create and start processor with DummyBackend."""
    backend = DummyBackend()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        enable_packing=True,
    )
    await exec.start()
    assert exec.is_ready()
    yield exec
    await exec.stop()


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_lifecycle_and_ready_state():
    """Test processor start, stop, and ready state."""
    backend = DummyBackend()
    exec = RequestProcessor(backend=backend)
    await exec.start()
    assert exec.is_ready()
    stats = exec.get_stats()
    assert stats["connected"] is True and stats["ready"] is True
    await exec.stop()
    assert not exec.is_ready()


@pytest.mark.asyncio
async def test_forward_backward_operations(processor):
    """Test forward_backward with datum list and forward-only pass."""
    # Forward backward with multiple samples
    datum_list = [
        {"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]},
        {"input_ids": [10, 20], "labels": [20, 30]},
        {"input_ids": [100, 200, 300], "labels": [200, 300, 400]},
    ]
    request = OrchestratorRequest(
        request_id="req-001",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=datum_list, loss_fn="causallm_loss"),
    )
    output = await processor.execute_forward_backward(request)
    assert isinstance(output, OrchestratorOutputs)
    assert output.request_id == "req-001"
    assert output.output_type == OutputType.FORWARD_BACKWARD
    assert output.finished is True
    assert "loss" in output.outputs[0]
    assert "valid_tokens" in output.outputs[0]
    assert output.outputs[0]["success"] is True
    assert output.outputs[0]["loss"] >= 0

    # Forward only (no gradients)
    request = OrchestratorRequest(
        request_id="req-fwd",
        request_type=RequestType.ADD,
        operation="forward",
        payload=ModelPassData(data=[{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}]),
    )
    output = await processor.execute_forward(request)
    assert output.output_type == OutputType.FORWARD
    assert "loss" in output.outputs[0]


def test_teacher_sort_key_reads_nested_loss_inputs():
    assert RequestProcessor._teacher_sort_key({"loss_fn_inputs": {"teacher_id": 3}}) == 3
    assert RequestProcessor._teacher_sort_key({"loss_fn_inputs": {"teacher_ids": [[2, 2, 2]]}}) == 2
    assert RequestProcessor._teacher_sort_key({"teacher_id": 1, "loss_fn_inputs": {"teacher_id": 4}}) == 1


@pytest.mark.asyncio
async def test_nccl_sync_uses_request_scoped_group_name():
    class CapturingBackend(DummyBackend):
        def __init__(self):
            super().__init__()
            self.group_names = []

        async def sync_inference_weights(self, *args, **kwargs):
            self.group_names.append(kwargs["group_name"])
            return await super().sync_inference_weights(*args, **kwargs)

    backend = CapturingBackend()
    exec = RequestProcessor(backend=backend)
    await exec.start()
    try:
        request = OrchestratorRequest(
            request_id="sync-req-0001",
            request_type=RequestType.ADD,
            operation="sync_inference_weights",
            payload=SyncWeightsData(
                endpoints=[{"host": "127.0.0.1", "port": 30000, "world_size": 1}],
                group_name="weight_sync_group",
                sync_method="nccl_broadcast",
            ),
        )
        output = await exec.execute_sync_inference_weights(request)
    finally:
        await exec.stop()

    assert output.output_type == OutputType.SYNC_INFERENCE_WEIGHTS
    assert backend.group_names == ["weight_sync_group_sync_req_0001"]


@pytest.mark.asyncio
async def test_model_pass_replay_fields_reach_backend(processor):
    """Both routing replay tensors should be forwarded for forward and forward_backward."""
    routed_experts = [[[1, 2], [3, 4]]]
    routed_expert_logits = [[[0.1, 0.9], [0.7, 0.3]]]
    result = {"total_loss": 1.25, "global_valid_tokens": 3}
    processor.backend.forward_backward = AsyncMock(return_value=result)
    processor.backend.forward = AsyncMock(return_value=result)

    fb_request = OrchestratorRequest(
        request_id="req-r3-fb",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(
            data=[{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}],
            model_id="session-a",
            routed_experts=routed_experts,
            routed_expert_logits=routed_expert_logits,
        ),
    )
    await processor.execute_forward_backward(fb_request)
    fb_kwargs = processor.backend.forward_backward.await_args.kwargs
    assert fb_kwargs["model_id"] == "session-a"
    assert fb_kwargs["routed_experts"] == routed_experts
    assert fb_kwargs["routed_expert_logits"] == routed_expert_logits

    fwd_request = OrchestratorRequest(
        request_id="req-r3-fwd",
        request_type=RequestType.ADD,
        operation="forward",
        payload=ModelPassData(
            data=[{"input_ids": [4, 5, 6], "labels": [5, 6, 7]}],
            model_id="session-b",
            routed_experts=routed_experts,
            routed_expert_logits=routed_expert_logits,
        ),
    )
    await processor.execute_forward(fwd_request)
    fwd_kwargs = processor.backend.forward.await_args.kwargs
    assert fwd_kwargs["model_id"] == "session-b"
    assert fwd_kwargs["routed_experts"] == routed_experts
    assert fwd_kwargs["routed_expert_logits"] == routed_expert_logits


def test_runner_dispatcher_forward_compute_preserves_model_id():
    """Forward-only runner execution should switch/use the requested session adapter."""

    class FakeTrainer:
        def __init__(self):
            self.forward_kwargs = None

        def forward(
            self,
            my_batches,
            loss_fn,
            loss_fn_params,
            *,
            model_id="default",
            routed_experts=None,
            routed_expert_logits=None,
        ):
            self.forward_kwargs = {
                "my_batches": my_batches,
                "loss_fn": loss_fn,
                "loss_fn_params": loss_fn_params,
                "model_id": model_id,
                "routed_experts": routed_experts,
                "routed_expert_logits": routed_expert_logits,
            }
            return {"success": True, "model_id": model_id}

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.trainer = FakeTrainer()
    routed_experts = [[[1, 2]]]
    routed_expert_logits = [[[0.25, 0.75]]]

    result = RunnerDispatcher._execute_compute(
        dispatcher,
        [{"input_ids": [1, 2], "labels": [2, 3]}],
        "causallm_loss",
        {"return_per_token": False},
        routed_experts,
        with_backward=False,
        model_id="session-a",
        routed_expert_logits=routed_expert_logits,
    )

    assert result["model_id"] == "session-a"
    assert dispatcher.trainer.forward_kwargs["model_id"] == "session-a"
    assert dispatcher.trainer.forward_kwargs["routed_experts"] == routed_experts
    assert dispatcher.trainer.forward_kwargs["routed_expert_logits"] == routed_expert_logits


@pytest.mark.asyncio
async def test_runner_dispatcher_forward_rank0_scatter_preserves_model_id():
    """The rank-0 forward handler must not drop model_id before compute execution."""

    class FakeCoordinator:
        def auto_load_if_evicted(self, model_id):
            captured["auto_load_model_id"] = model_id
            return False, None

    captured = {}
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher._adapter_coordinator = FakeCoordinator()

    routed_experts = [[[1, 2]]]
    routed_expert_logits = [[[0.25, 0.75]]]

    def select_batches(batches, routed_experts=None, routed_expert_logits=None):
        return batches, routed_experts, routed_expert_logits

    def execute_and_gather(
        my_batches,
        loss_fn,
        loss_fn_params,
        routed_experts,
        cp_enabled,
        parallel_state,
        *,
        with_backward,
        model_id,
        is_rank0,
        routed_expert_logits=None,
    ):
        captured.update(
            {
                "model_id": model_id,
                "with_backward": with_backward,
                "is_rank0": is_rank0,
                "routed_experts": routed_experts,
                "routed_expert_logits": routed_expert_logits,
            }
        )
        return {"success": True, "model_id": model_id}

    dispatcher._select_and_prepare_batches = select_batches
    dispatcher._execute_and_gather = execute_and_gather

    result = await RunnerDispatcher._handle_compute_rank0_scatter(
        dispatcher,
        {
            "payload": ModelPassData(
                batches=[{"input_ids": [1, 2], "labels": [2, 3]}],
                model_id="session-a",
                routed_experts=routed_experts,
                routed_expert_logits=routed_expert_logits,
            )
        },
        with_backward=False,
    )

    assert result["model_id"] == "session-a"
    assert captured["model_id"] == "session-a"
    assert captured["auto_load_model_id"] == "session-a"
    assert captured["with_backward"] is False
    assert captured["is_rank0"] is True
    assert captured["routed_experts"] == routed_experts
    assert captured["routed_expert_logits"] == routed_expert_logits


@pytest.mark.asyncio
async def test_optim_and_checkpoint_operations(processor):
    """Test optim_step, save_state, load_state, sleep, and wake_up."""
    # Optim step
    request = OrchestratorRequest(
        request_id="req-004",
        request_type=RequestType.ADD,
        operation="optim_step",
        payload=OptimStepData(lr=0.001, gradient_clip=1.0),
    )
    output = await processor.execute_optim_step(request)
    assert output.output_type == OutputType.OPTIM_STEP
    assert "grad_norm" in output.outputs[0]
    assert output.outputs[0]["learning_rate"] == 0.001

    # Save state
    request = OrchestratorRequest(
        request_id="req-save",
        request_type=RequestType.ADD,
        operation="save_state",
        payload=SaveStateData(checkpoint_path="/tmp/ckpt"),
    )
    output = await processor.execute_save_state(request)
    assert output.output_type == OutputType.SAVE_STATE
    assert output.outputs[0]["success"] is True

    # Load state
    request = OrchestratorRequest(
        request_id="req-load",
        request_type=RequestType.ADD,
        operation="load_state",
        payload=LoadStateData(checkpoint_path="/tmp/ckpt"),
    )
    output = await processor.execute_load_state(request)
    assert output.output_type == OutputType.LOAD_STATE
    assert output.outputs[0]["success"] is True

    # Sleep
    request = OrchestratorRequest(
        request_id="req-sleep",
        request_type=RequestType.ADD,
        operation="sleep",
        payload=EmptyData(),
    )
    output = await processor.execute_sleep(request)
    assert output.output_type == OutputType.SLEEP

    # Wake up
    request = OrchestratorRequest(
        request_id="req-wake",
        request_type=RequestType.ADD,
        operation="wake_up",
        payload=EmptyData(),
    )
    output = await processor.execute_wake_up(request)
    assert output.output_type == OutputType.WAKE_UP


@pytest.mark.asyncio
async def test_register_session_operation_reaches_backend(processor):
    """register_session should flow through the processor to the backend."""
    session_spec = {
        "base_model": "Qwen/Qwen3-8B",
        "lora_config": {"lora_rank": 4, "lora_alpha": 8},
        "optimizer_config": {"type": "adamw", "learning_rate": 1e-4},
    }
    request = OrchestratorRequest(
        request_id="req-register-session",
        request_type=RequestType.ADD,
        operation="register_session",
        payload=RegisterSessionData(model_id="session-a", session_spec=session_spec, materialize=True),
    )

    output = await processor.execute_register_session(request)

    assert output.output_type == OutputType.REGISTER_SESSION
    result = output.outputs["result"]
    assert result["registered"] is True
    assert result["model_id"] == "session-a"
    assert result["session_spec"] == session_spec
    assert result["materialize"] is True


@pytest.mark.asyncio
async def test_runner_dispatcher_register_session_handler_materializes_adapter():
    """Remote register_session should be a real runner operation, not an unknown command."""

    class FakeCoordinator:
        def __init__(self):
            self.command_dict = None

        async def handle_register_session(self, command_dict):
            self.command_dict = command_dict
            payload = command_dict["payload"]
            lr = payload.session_spec["optimizer_config"]["learning_rate"]
            return {
                "registered": True,
                "model_id": payload.model_id,
                "lr": lr,
                "session_spec": payload.session_spec,
                "materialize": payload.materialize,
            }

    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher.rank = 0
    dispatcher._adapter_coordinator = FakeCoordinator()
    session_spec = {
        "optimizer_config": {"learning_rate": 2e-4},
        "lora_config": {"lora_rank": 4, "lora_alpha": 8},
    }

    result = await RunnerDispatcher._handle_register_session(
        dispatcher,
        {
            "payload": RegisterSessionData(
                model_id="session-a",
                session_spec=session_spec,
                materialize=True,
            )
        },
    )

    assert RunnerDispatcher._COMMAND_HANDLERS["register_session"] == "_handle_register_session"
    assert result["registered"] is True
    assert result["model_id"] == "session-a"
    assert result["lr"] == pytest.approx(2e-4)
    assert result["session_spec"] == session_spec
    assert result["materialize"] is True
    forwarded_payload = dispatcher._adapter_coordinator.command_dict["payload"]
    assert forwarded_payload.session_spec["optimizer_config"]["learning_rate"] == pytest.approx(2e-4)
    assert forwarded_payload.materialize is True


@pytest.mark.asyncio
async def test_statistics_tracking(processor):
    """Test that statistics track operations correctly."""
    initial = processor.total_operations

    request = OrchestratorRequest(
        request_id="req-stat",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2], "labels": [2, 3]}]),
    )
    await processor.execute_forward_backward(request)

    assert processor.total_operations == initial + 1
    assert processor.successful_operations >= 1

    stats = processor.get_stats()
    assert "connected" in stats and "total_operations" in stats


@pytest.mark.asyncio
async def test_error_handling(processor):
    """Test error handling for empty datum list, missing labels, and sequential ops."""
    # Empty datum list
    request = OrchestratorRequest(
        request_id="req-008",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[]),
    )
    output = await processor.execute_forward_backward(request)
    assert output.output_type == OutputType.ERROR

    # Without labels (no valid tokens)
    request = OrchestratorRequest(
        request_id="req-009",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2, 3, 4, 5]}]),
    )
    output = await processor.execute_forward_backward(request)
    assert output.output_type == OutputType.ERROR

    # Multiple sequential operations
    for i in range(5):
        request = OrchestratorRequest(
            request_id=f"req-seq-{i}",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(data=[{"input_ids": list(range(10)), "labels": list(range(1, 11))}]),
        )
        output = await processor.execute_forward_backward(request)
        assert output.finished is True

    assert processor.total_operations >= 5
    assert processor.successful_operations >= 5
