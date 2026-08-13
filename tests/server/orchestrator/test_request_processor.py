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

import json
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


@pytest.mark.asyncio
async def test_forward_backward_preserves_runner_result_fields(processor):
    result = {
        "backward_compute_time": 0.75,
        "forward_compute_time": 0.5,
        "total_loss": 0.5,
        "global_valid_tokens": 2,
        "forward_backward_time": 1.25,
    }
    processor.backend.forward_backward = AsyncMock(return_value=result)

    request = OrchestratorRequest(
        request_id="req-fb-result",
        request_type=RequestType.ADD,
        operation="forward_backward",
        payload=ModelPassData(data=[{"input_ids": [1, 2], "labels": [2, 3]}], loss_fn="causallm_loss"),
    )

    output = await processor.execute_forward_backward(request)

    assert output.output_type == OutputType.FORWARD_BACKWARD
    assert output.outputs[0]["backward_compute_time"] == pytest.approx(0.75)
    assert output.outputs[0]["forward_compute_time"] == pytest.approx(0.5)
    assert output.outputs[0]["forward_backward_time"] == pytest.approx(1.25)


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


@pytest.mark.asyncio
async def test_model_pass_writes_r3_payloads_to_mooncake_refs():
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


@pytest.mark.asyncio
async def test_model_pass_cleans_mooncake_routing_payloads_by_default():
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


def test_mooncake_chunk_ranges_follow_dispatcher_dp_slices():
    processor = RequestProcessor(backend=DummyBackend(), dp_size=3)
    batches = [
        {"num_samples": 2},
        {"num_samples": 1},
        {"num_samples": 3},
        {"num_samples": 4},
    ]

    assert processor._routing_payload_chunk_ranges(batches, 10) == [(0, 3), (3, 3), (6, 4)]


@pytest.mark.asyncio
async def test_model_pass_cleans_mooncake_routing_payloads_on_backend_exception():
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


@pytest.mark.asyncio
async def test_model_pass_mooncake_payloads_follow_packer_datum_order(monkeypatch):
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
        del args, kwargs
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
                    {"input_ids": [1, 2], "labels": [2, 3]},
                    {"input_ids": [3, 4], "labels": [4, 5]},
                    {"input_ids": [5, 6], "labels": [6, 7]},
                ],
                routed_experts=routed_experts,
            ),
        )
        await exec.execute_forward_backward(request)
        expert_ref = exec.backend.forward_backward.await_args.kwargs["routed_experts"]
        loaded = load_r3_mooncake_payload_slice(expert_ref, 0, 2, store=store)
        assert [item.tolist() for item in loaded] == [routed_experts[2], routed_experts[0]]
    finally:
        await exec.stop()


@pytest.mark.asyncio
async def test_model_pass_cleans_externalized_routing_payloads_by_default(tmp_path):
    backend = DummyBackend()
    exec = RequestProcessor(
        backend=backend,
        sample_packing_sequence_len=100,
        routing_payload_dir=str(tmp_path / "routing"),
    )
    await exec.start()
    seen = {}

    async def _forward_backward(**kwargs):
        expert_ref = kwargs["routed_experts"]
        manifest_path = Path(expert_ref["manifest"])
        assert manifest_path.exists()
        assert expert_ref["version"] == 3
        assert expert_ref["format"] == "packed_rows"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["format"] == "xorl-r3-packed"
        assert len(manifest["routed_experts"]["chunks"]) == 1
        assert sorted(path.name for path in (manifest_path.parent / "routed_experts").iterdir()) == ["chunk-000000.bin"]
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

    def select_batches(batches, loss_fn_params=None, routed_experts=None, routed_expert_logits=None):
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
    original_optim_step = processor.backend.optim_step

    async def _optim_step_with_cleanup_metrics(**kwargs):
        return {
            "grad_norm": 0.5,
            "step": 3,
            "learning_rate": kwargs["lr"],
            "optim_step_time": 0.125,
            "optim_empty_cache_skipped": True,
            "glm52_fullparam_publish": {
                "published": True,
                "step": 3,
                "manifest_checksum": "manifest-3",
            },
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
    assert output.outputs[0]["glm52_fullparam_publish"] == {
        "published": True,
        "step": 3,
        "manifest_checksum": "manifest-3",
    }

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


# ============================================================================
# _unpack_token_diagnostics
# ============================================================================


def test_unpack_token_diagnostics_splits_two_samples():
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


def test_unpack_token_diagnostics_empty_diagnostics_returns_empty():
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


def test_unpack_token_diagnostics_field_length_mismatch_raises():
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


@pytest.mark.asyncio
async def test_packed_row_batching_groups_single_row_packed_batches():
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


@pytest.mark.asyncio
async def test_packed_row_batching_defers_to_rank_local_runner_by_default():
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


@pytest.mark.asyncio
async def test_packed_row_batching_rejects_routed_replay(processor):
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


def test_sglang_span_payloads_bypass_repacking_and_cleanup_sources(tmp_path, monkeypatch):
    source = tmp_path / "routing.bin"
    source.write_bytes(b"\0" * 32)
    item = {
        "schema": "xorl.r3.spans.v1",
        "rows": 2,
        "shape": [2, 2, 2],
        "dtype": "int32",
        "spans": [
            {
                "path": str(source),
                "error_path": str(tmp_path / ".routing.error.json"),
                "offset": 0,
                "source_row": 0,
                "rows": 2,
                "row_nbytes": 16,
                "source_shape": [2, 2, 2],
                "dtype": "int32",
            }
        ],
    }
    monkeypatch.setenv("XORL_R3_SHARED_ROOTS", str(tmp_path))
    processor = RequestProcessor(backend=DummyBackend())

    routed, logits, cleanup = processor._externalize_routing_payloads(
        "request", [item], None
    )

    assert logits is None
    assert routed["transport"] == "sglang_files"
    assert routed["items"][0] is item
    assert source.exists()
    processor._cleanup_routing_payloads(cleanup)
    assert not source.exists()
