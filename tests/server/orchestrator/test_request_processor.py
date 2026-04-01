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
    SaveStateData,
)


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
