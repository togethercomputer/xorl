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

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
import torch

from xorl.server.backend import DummyBackend
from xorl.server.orchestrator import request_processor as request_processor_module
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
from xorl.server.side_payloads import (
    SIDE_PAYLOAD_REF_KEY,
    MooncakeSidePayloadStore,
    load_r3_mooncake_payload_slice,
)


# ============================================================================
# Fixtures
# ============================================================================


class FakeMooncakeClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.removed: list[str] = []

    def put(self, key: str, value: bytes) -> int:
        self.objects[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        return self.objects.get(key, b"")

    def is_exist(self, key: str) -> int:
        return 1 if key in self.objects else 0

    def remove(self, key: str) -> int:
        self.objects.pop(key, None)
        self.removed.append(key)
        return 0


def _mooncake_store() -> tuple[MooncakeSidePayloadStore, FakeMooncakeClient]:
    client = FakeMooncakeClient()
    return MooncakeSidePayloadStore(client=client, get_retry_max_wait_s=0.0), client


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
async def test_forward_backward_operation_and_lifecycle_policy(processor):
    """A live processor tracks successful model passes and shuts down cleanly."""
    assert processor.is_ready()
    initial = processor.total_operations
    stats = processor.get_stats()
    assert stats["connected"] is True and stats["ready"] is True

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

    result = {
        "backward_compute_time": 0.75,
        "forward_compute_time": 0.5,
        "total_loss": 0.5,
        "global_valid_tokens": 2,
        "forward_backward_time": 1.25,
    }
    processor.backend.forward_backward = AsyncMock(return_value=result)
    timed_request = OrchestratorRequest(
        request_id="req-fb-result",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2], "labels": [2, 3]}], loss_fn="causallm_loss"),
    )
    timed_output = await processor.execute_forward_backward(timed_request)
    assert timed_output.outputs[0]["backward_compute_time"] == pytest.approx(0.75)
    assert timed_output.outputs[0]["forward_compute_time"] == pytest.approx(0.5)
    assert timed_output.outputs[0]["forward_backward_time"] == pytest.approx(1.25)

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

    assert processor.total_operations == initial + 3
    assert processor.successful_operations >= 3
    stats = processor.get_stats()
    assert stats["total_operations"] == initial + 3

    await _assert_forward_backward_rejects_batches_without_valid_targets(processor)
    await _assert_packed_row_batching_policy(processor)

    await processor.stop()
    assert not processor.is_ready()
    await _assert_runner_dispatcher_forward_lifecycle_preserves_model_id()


async def _assert_nccl_sync_uses_request_scoped_group_name():
    class CapturingBackend(DummyBackend):
        def __init__(self):
            super().__init__()
            self.group_names = []
            self.cache_invalidation_modes = []

        async def sync_inference_weights(self, *args, **kwargs):
            self.group_names.append(kwargs["group_name"])
            self.cache_invalidation_modes.append(kwargs["cache_invalidation_mode"])
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
                cache_invalidation_mode="none",
            ),
        )
        output = await exec.execute_sync_inference_weights(request)
    finally:
        await exec.stop()

    assert output.output_type == OutputType.SYNC_INFERENCE_WEIGHTS
    assert backend.group_names == ["weight_sync_group_sync_req_0001"]
    assert backend.cache_invalidation_modes == ["none"]


@pytest.mark.asyncio
async def test_model_pass_r3_payload_lifecycle(tmp_path, monkeypatch):
    await _assert_model_pass_writes_r3_payloads_to_mooncake_refs()
    await _assert_model_pass_routing_payload_cleanup_policy(tmp_path / "cleanup")
    with monkeypatch.context() as case_patch:
        await _assert_opd_sort_and_mooncake_payloads_follow_packer_datum_order(case_patch)
    _assert_unpack_token_diagnostics_policy()


async def _assert_model_pass_writes_r3_payloads_to_mooncake_refs():
    backend = DummyBackend()
    store, _ = _mooncake_store()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        r3_payload_transport="mooncake",
        r3_payload_keep=True,
        routing_payload_store=store,
    )
    await exec.start()
    try:
        result = {"total_loss": 1.25, "global_valid_tokens": 3}
        exec.backend.forward_backward = AsyncMock(return_value=result)
        routed_experts = [
            [[[1, 2]], [[3, 4]]],
            [[[5, 6]], [[7, 8]]],
        ]
        routed_expert_logits = [
            [[[0.25, 0.75]], [[0.5, 0.5]]],
            [[[0.1, 0.9]], [[0.6, 0.4]]],
        ]

        request = OrchestratorRequest(
            request_id="req-r3-ref",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=[
                    {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
                    {"input_ids": [4, 5, 6], "labels": [5, 6, 7]},
                ],
                routed_experts=routed_experts,
                routed_expert_logits=routed_expert_logits,
            ),
        )
        await exec.execute_forward_backward(request)
        kwargs = exec.backend.forward_backward.await_args.kwargs
        expert_ref = kwargs["routed_experts"]
        logits_ref = kwargs["routed_expert_logits"]
        assert expert_ref[SIDE_PAYLOAD_REF_KEY] is True
        assert logits_ref[SIDE_PAYLOAD_REF_KEY] is True
        assert expert_ref["backend"] == "mooncake"
        assert logits_ref["backend"] == "mooncake"
        assert expert_ref["field"] == "routed_experts"
        assert logits_ref["field"] == "routed_expert_logits"
        assert expert_ref["count"] == 2
        assert load_r3_mooncake_payload_slice(expert_ref, 1, 1, store=store)[0].tolist() == routed_experts[1]
        assert np.allclose(load_r3_mooncake_payload_slice(logits_ref, 1, 1, store=store)[0], routed_expert_logits[1])
    finally:
        await exec.stop()


async def _assert_model_pass_routing_payload_cleanup_policy(tmp_path):
    backend = DummyBackend()
    store, client = _mooncake_store()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        r3_payload_transport="mooncake",
        routing_payload_store=store,
    )
    await exec.start()
    seen = {}

    async def _forward_backward(**kwargs):
        seen["keys"] = [
            kwargs["routed_experts"]["items"]["routed_experts"][0]["key"],
            kwargs["routed_expert_logits"]["items"]["routed_expert_logits"][0]["key"],
        ]
        assert all(key in client.objects for key in seen["keys"])
        return {"total_loss": 1.25, "global_valid_tokens": 3}

    try:
        exec.backend.forward_backward = AsyncMock(side_effect=_forward_backward)
        request = OrchestratorRequest(
            request_id="req-r3-mooncake-clean",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=[{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}],
                routed_experts=[[[[1, 2]]]],
                routed_expert_logits=[[[[0.25, 0.75]]]],
            ),
        )
        output = await exec.execute_forward_backward(request)
    finally:
        await exec.stop()

    assert output.output_type == OutputType.FORWARD_BACKWARD
    assert seen["keys"]
    assert client.objects == {}
    assert sorted(client.removed) == sorted(seen["keys"])

    await _assert_mooncake_payloads_cleaned_on_backend_exception()
    await _assert_externalized_payloads_cleaned_by_default(tmp_path)


async def _assert_mooncake_payloads_cleaned_on_backend_exception():
    backend = DummyBackend()
    store, client = _mooncake_store()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        r3_payload_transport="mooncake",
        routing_payload_store=store,
    )
    await exec.start()

    async def _forward_backward(**kwargs):
        assert kwargs["routed_experts"][SIDE_PAYLOAD_REF_KEY] is True
        raise RuntimeError("backend failed")

    try:
        exec.backend.forward_backward = AsyncMock(side_effect=_forward_backward)
        request = OrchestratorRequest(
            request_id="req-r3-mooncake-error",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=[{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}],
                routed_experts=[[[[1, 2]]]],
                routed_expert_logits=[[[[0.25, 0.75]]]],
            ),
        )
        output = await exec.execute_forward_backward(request)
    finally:
        await exec.stop()

    assert output.output_type == OutputType.ERROR
    assert "backend failed" in output.error
    assert client.objects == {}
    assert len(client.removed) == 2


async def _assert_opd_sort_and_mooncake_payloads_follow_packer_datum_order(monkeypatch):
    backend = DummyBackend()
    store, _ = _mooncake_store()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        r3_payload_transport="mooncake",
        r3_payload_keep=True,
        routing_payload_store=store,
    )

    def _pack_samples(*args, **kwargs):
        del args
        assert [datum["input_ids"] for datum in kwargs["datum_list"]] == [[3, 4], [5, 6], [1, 2]]
        return (
            [
                {
                    "input_ids": [[1, 2, 3, 4]],
                    "labels": [[-100, 2, 3, 4]],
                    "position_ids": [[0, 1, 2, 3]],
                    "request_id": "req-r3-order",
                    "batch_id": 0,
                    "num_samples": 2,
                }
            ],
            [2, 0],
        )

    monkeypatch.setattr(request_processor_module, "pack_samples", _pack_samples)
    await exec.start()
    try:
        exec.backend.forward_backward = AsyncMock(return_value={"total_loss": 1.25, "global_valid_tokens": 3})
        routed_experts = [
            [[[10, 11]]],
            [[[20, 21]]],
            [[[30, 31]]],
        ]
        request = OrchestratorRequest(
            request_id="req-r3-order",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=[
                    {"input_ids": [1, 2], "labels": [2, 3], "loss_fn_inputs": {"teacher_id": 3}},
                    {"input_ids": [3, 4], "labels": [4, 5], "loss_fn_inputs": {"teacher_ids": [[1, 1]]}},
                    {
                        "input_ids": [5, 6],
                        "labels": [6, 7],
                        "teacher_id": 2,
                        "loss_fn_inputs": {"teacher_id": 4},
                    },
                ],
                loss_fn="opd_loss",
                routed_experts=routed_experts,
            ),
        )
        await exec.execute_forward_backward(request)
        expert_ref = exec.backend.forward_backward.await_args.kwargs["routed_experts"]
        loaded = load_r3_mooncake_payload_slice(expert_ref, 0, 2, store=store)
        assert [item.tolist() for item in loaded] == [routed_experts[0], routed_experts[1]]
    finally:
        await exec.stop()


async def _assert_externalized_payloads_cleaned_by_default(tmp_path):
    backend = DummyBackend()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        r3_payload_transport="filesystem",
        r3_payload_dir=str(tmp_path / "routing"),
    )
    await exec.start()
    seen = {}

    async def _forward_backward(**kwargs):
        expert_ref = kwargs["routed_experts"]
        manifest_path = Path(expert_ref["manifest"])
        assert manifest_path.exists()
        seen["root"] = manifest_path.parent
        dispatcher = object.__new__(RunnerDispatcher)
        dispatcher.rank = 0
        loaded = dispatcher._load_routing_payload_slice(expert_ref, 1, 1)
        assert loaded is not None
        assert torch.equal(loaded[0], torch.tensor([[[20, 21]]], dtype=torch.int32))
        return {"total_loss": 1.25, "global_valid_tokens": 3}

    try:
        exec.backend.forward_backward = AsyncMock(side_effect=_forward_backward)
        request = OrchestratorRequest(
            request_id="req-r3-clean",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=[
                    {"input_ids": [1, 2, 3], "labels": [2, 3, 4]},
                    {"input_ids": [4, 5, 6], "labels": [5, 6, 7]},
                ],
                routed_experts=[[[[10, 11]]], [[[20, 21]]]],
                routed_expert_logits=[[[[0.1, 0.9]]], [[[0.2, 0.8]]]],
            ),
        )
        await exec.execute_forward_backward(request)
    finally:
        await exec.stop()

    assert "root" in seen
    assert not seen["root"].exists()


async def _assert_runner_dispatcher_forward_lifecycle_preserves_model_id():
    """A rank-0 forward request reaches the trainer with its session and R3 data."""

    class FakeCoordinator:
        def auto_load_if_evicted(self, model_id):
            captured["auto_load_model_id"] = model_id
            return False, None

    class FakeTrainer:
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
            captured.update(
                {
                    "batches": my_batches,
                    "loss_fn": loss_fn,
                    "loss_fn_params": loss_fn_params,
                    "model_id": model_id,
                    "routed_experts": routed_experts,
                    "routed_expert_logits": routed_expert_logits,
                }
            )
            return {"success": True, "model_id": model_id}

    captured = {}
    dispatcher = object.__new__(RunnerDispatcher)
    dispatcher._adapter_coordinator = FakeCoordinator()
    dispatcher.trainer = FakeTrainer()

    routed_experts = [[[1, 2]]]
    routed_expert_logits = [[[0.25, 0.75]]]

    def select_batches(batches, loss_fn_params=None, routed_experts=None, routed_expert_logits=None):
        return batches, routed_experts, routed_expert_logits

    dispatcher._select_and_prepare_batches = select_batches
    dispatcher._shard_and_slice_batches = lambda batches, experts, logits, _cp, _state: (batches, experts, logits)
    dispatcher._maybe_dump_microbatch_diagnostic = lambda *args, **kwargs: None
    dispatcher._gather_is_metrics = lambda *args, **kwargs: None
    dispatcher._completion_rendezvous = lambda *args, **kwargs: []
    dispatcher._merge_completion_payloads = lambda *args, **kwargs: None

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
    assert captured["routed_experts"] == routed_experts
    assert captured["routed_expert_logits"] == routed_expert_logits


@pytest.mark.asyncio
async def test_control_optim_checkpoint_sync_and_lifecycle_operations(processor):
    """Test optim_step, save_state, load_state, sleep, and wake_up."""
    original_optim_step = processor.backend.optim_step

    async def _optim_step_with_cleanup_metrics(**kwargs):
        return {
            "grad_norm": 0.5,
            "step": 3,
            "learning_rate": kwargs["lr"],
            "optim_step_time": 0.125,
            "optim_empty_cache_skipped": True,
        }

    processor.backend.optim_step = _optim_step_with_cleanup_metrics

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
    assert output.outputs[0]["optim_step_time"] == pytest.approx(0.125)
    assert output.outputs[0]["optim_empty_cache_skipped"] is True

    processor.backend.optim_step = original_optim_step

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

    await _assert_nccl_sync_uses_request_scoped_group_name()
    await _assert_register_session_operation_reaches_backend(processor)


async def _assert_register_session_operation_reaches_backend(processor):
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


async def _assert_forward_backward_rejects_batches_without_valid_targets(processor):
    """Reject a nonempty batch that contains no valid targets."""
    request = OrchestratorRequest(
        request_id="req-009",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2, 3, 4, 5]}]),
    )
    output = await processor.execute_forward_backward(request)
    assert output.output_type == OutputType.ERROR


# ============================================================================
# _unpack_token_diagnostics
# ============================================================================


def _assert_unpack_token_diagnostics_policy():
    """Two packed samples with a position-id reset between them: diagnostics
    split correctly, valid_positions rebased per sample, fields aligned."""
    # Packed sequence: sample A spans positions 0..3, sample B spans 0..2.
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
    # Valid (non-IGNORE) positions in micro-batch coords: A has 1,3; B has 4,6.
    diagnostics = {
        "valid_positions": [1, 3, 4, 6],
        "target_ids": [11, 13, 24, 26],
        "target_logprobs": [-1.0, -1.5, -2.0, -2.5],
        "target_ranks": [1, 2, 3, 4],
        "topk_ids": [[11, 12], [13, 14], [24, 25], [26, 27]],
        "topk_logprobs": [[-1.0, -1.1], [-1.5, -1.6], [-2.0, -2.1], [-2.5, -2.6]],
        "loss_logprobs": [-1.0, -1.5, -2.0, -2.5],
        "loss_logprob_deltas": [0.0, 0.0, 0.0, 0.0],
        "reference_target_logprobs": [-0.9, -1.4, -1.9, -2.4],
        "reference_target_ranks": [1, 2, 2, 3],
        "reference_logprob_deltas": [-0.1, -0.1, -0.1, -0.1],
        "hidden_state_summaries": [
            {"layer_count": 2, "layers": [{"index": 0, "rms": 1.0}]},
            {"layer_count": 2, "layers": [{"index": 0, "rms": 1.3}]},
            {"layer_count": 2, "layers": [{"index": 0, "rms": 2.4}]},
            {"layer_count": 2, "layers": [{"index": 0, "rms": 2.6}]},
        ],
        "hidden_component_summaries": [
            {"component_count": 2, "components": [{"layer": 34, "name": "layer_input", "rms": 1.0}]},
            {"component_count": 2, "components": [{"layer": 34, "name": "mlp", "rms": 1.3}]},
            {"component_count": 2, "components": [{"layer": 38, "name": "layer_input", "rms": 2.4}]},
            {"component_count": 2, "components": [{"layer": 38, "name": "mlp", "rms": 2.6}]},
        ],
    }
    samples = RequestProcessor._unpack_token_diagnostics(diagnostics, position_ids)
    assert len(samples) == 2

    a, b = samples
    # Sample A: positions 1,3 → rebased identical (sample starts at 0)
    assert a["valid_positions"] == [1, 3]
    assert a["target_ids"] == [11, 13]
    assert a["target_ranks"] == [1, 2]
    assert a["topk_ids"] == [[11, 12], [13, 14]]
    assert a["loss_logprobs"] == [-1.0, -1.5]
    assert a["reference_target_logprobs"] == [-0.9, -1.4]
    assert a["reference_target_ranks"] == [1, 2]
    assert a["hidden_state_summaries"][1]["layers"][0]["rms"] == 1.3
    assert a["hidden_component_summaries"][1]["components"][0]["name"] == "mlp"

    # Sample B: positions 4,6 → rebased to 0,2 (sample starts at 4)
    assert b["valid_positions"] == [0, 2]
    assert b["target_ids"] == [24, 26]
    assert b["target_ranks"] == [3, 4]
    assert b["topk_ids"] == [[24, 25], [26, 27]]
    assert b["loss_logprob_deltas"] == [0.0, 0.0]
    assert b["reference_logprob_deltas"] == [-0.1, -0.1]
    assert b["hidden_state_summaries"][0]["layers"][0]["rms"] == 2.4
    assert b["hidden_component_summaries"][0]["components"][0]["layer"] == 38

    _assert_empty_token_diagnostics_return_empty()
    _assert_token_diagnostic_length_mismatch_raises()


def _assert_empty_token_diagnostics_return_empty():
    """Empty/falsy diagnostics short-circuits to []."""
    assert RequestProcessor._unpack_token_diagnostics({}, torch.tensor([[0, 1, 2]])) == []
    assert RequestProcessor._unpack_token_diagnostics(None, torch.tensor([[0, 1, 2]])) == []
    # Has shape but no valid positions
    empty_diag = {
        "valid_positions": [],
        "target_ids": [],
        "target_logprobs": [],
        "target_ranks": [],
        "topk_ids": [],
        "topk_logprobs": [],
    }
    assert RequestProcessor._unpack_token_diagnostics(empty_diag, torch.tensor([[0, 1, 2]])) == []


def _assert_token_diagnostic_length_mismatch_raises():
    """A producer-side bug (field shorter than valid_positions) raises rather than
    silently truncating."""
    diagnostics = {
        "valid_positions": [0, 1, 2],
        "target_ids": [10, 11],  # one short
        "target_logprobs": [-1.0, -1.1, -1.2],
        "target_ranks": [1, 1, 1],
        "topk_ids": [[10], [11], [12]],
        "topk_logprobs": [[-1.0], [-1.1], [-1.2]],
    }
    with pytest.raises(ValueError, match="target_ids"):
        RequestProcessor._unpack_token_diagnostics(diagnostics, torch.tensor([[0, 1, 2]]))


async def _assert_packed_row_batching_policy(processor):
    backend = DummyBackend()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=6,
        enable_packing=True,
        pad_to_multiple_of=1,
    )
    await exec.start()
    try:
        backend.forward_backward = AsyncMock(return_value={"total_loss": 1.0, "global_valid_tokens": 9})
        data = [
            {"input_ids": [1, 2, 3], "target_tokens": [2, 3, 4]},
            {"input_ids": [4, 5, 6], "target_tokens": [5, 6, 7]},
            {"input_ids": [7, 8, 9], "target_tokens": [8, 9, 10]},
        ]
        request = OrchestratorRequest(
            request_id="req-rowbatch",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=data,
                loss_fn="opd_loss",
                loss_fn_params={"opd_packed_row_batch_size": 2, "opd_packed_row_batch_scope": "global"},
            ),
        )
        output = await exec.execute_forward_backward(request)
    finally:
        await exec.stop()

    batches = backend.forward_backward.await_args.kwargs["batches"]
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch["input_ids"]) == 1
    assert batch["input_ids"][0] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert batch["position_ids"][0] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert batch["cu_seq_lens_q"] == [0, 3, 6, 9]
    assert batch["num_samples"] == 3
    assert output.outputs[0]["executor_original_batches"] == 2
    assert output.outputs[0]["executor_batches"] == 1
    assert output.outputs[0]["executor_packed_row_batch_size"] == 2

    await _assert_packed_row_batching_defers_to_rank_local_runner()
    await _assert_packed_row_batching_rejects_routed_replay(processor)


async def _assert_packed_row_batching_defers_to_rank_local_runner():
    backend = DummyBackend()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=6,
        enable_packing=True,
        pad_to_multiple_of=1,
    )
    await exec.start()
    try:
        backend.forward_backward = AsyncMock(return_value={"total_loss": 1.0, "global_valid_tokens": 9})
        data = [
            {"input_ids": [1, 2, 3], "target_tokens": [2, 3, 4]},
            {"input_ids": [4, 5, 6], "target_tokens": [5, 6, 7]},
            {"input_ids": [7, 8, 9], "target_tokens": [8, 9, 10]},
        ]
        request = OrchestratorRequest(
            request_id="req-rowbatch-rank-local",
            request_type=RequestType.ADD,
            operation="forward_backward",
            payload=ModelPassData(
                data=data,
                loss_fn="opd_loss",
                loss_fn_params={"opd_packed_row_batch_size": 2},
            ),
        )
        output = await exec.execute_forward_backward(request)
    finally:
        await exec.stop()

    batches = backend.forward_backward.await_args.kwargs["batches"]
    assert len(batches) == 2
    assert [batch["num_samples"] for batch in batches] == [2, 1]
    assert output.outputs[0]["executor_original_batches"] == 2
    assert output.outputs[0]["executor_batches"] == 2
    assert output.outputs[0]["executor_packed_row_batch_size"] == 2


async def _assert_packed_row_batching_rejects_routed_replay(processor):
    request = OrchestratorRequest(
        request_id="req-rowbatch-r3",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(
            data=[{"input_ids": [1, 2, 3], "labels": [2, 3, 4]}],
            loss_fn="opd_loss",
            loss_fn_params={"opd_packed_row_batch_size": 2},
            routed_experts=[[[1, 2], [3, 4]]],
        ),
    )
    output = await processor.execute_forward_backward(request)
    assert output.output_type == OutputType.ERROR
    assert "routed_experts replay" in output.error
