"""
API Server with integrated OrchestratorClient.

This server combines FastAPI REST endpoints with the OrchestratorClient
for direct communication with the Engine backend.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                      API SERVER (Frontend)                               │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    OrchestratorClient                                     │ │
│  │                                                                      │ │
│  │  INPUT SOCKET  ZMQ ROUTER (bind)                                   │ │
│  │       ↓ send requests                                               │ │
│  │       • OrchestratorRequest (ADD, ABORT, UTILITY)                    │ │
│  │       • Msgpack serialization                                       │ │
│  │                                                                      │ │
│  │  OUTPUT SOCKET ZMQ PULL (connect)                                  │ │
│  │       ↑ receive outputs                                             │ │
│  │       • OrchestratorOutputs                                           │ │
│  │       • Msgpack deserialization                                     │ │
│  │       • Async queue for buffering                                   │ │
│  │                                                                      │ │
│  │  Background Tasks:                                                  │ │
│  │    - process_outputs_socket() [asyncio.Task]                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

The APIServer class is composed from domain-specific mixins:
- TrainingOpsMixin: forward/backward, optim_step, two-phase async pattern
- WeightsMixin: save/load weights, checkpoint management
- InferenceEndpointsMixin: inference endpoints, LoRA adapter management, sampling
- HealthMixin: health check, sleep, wake-up
"""

import asyncio
import logging
import os
import time
import uvicorn
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status

from xorl.server.api_server.endpoints import router
from xorl.server.api_server.health import HealthMixin
from xorl.server.api_server.inference_endpoints import InferenceEndpointsMixin
from xorl.server.api_server.training_ops import TrainingOpsMixin
from xorl.server.api_server.utils import (
    MODEL_ID_NOT_REGISTERED_ERROR,
    validate_model_id,
)
from xorl.server.api_server.weights import WeightsMixin
from xorl.server.api_server.api_types import (
    InferenceEndpoint,
    LossFnOutput,
    TensorData,
)
from xorl.server.api_server.orchestrator_client import OrchestratorClient
from xorl.server.api_server.future_store import FutureStore
from xorl.server.protocol.api_orchestrator import OrchestratorRequest
from xorl.server.protocol.operations import KillSessionData


logger = logging.getLogger(__name__)


# ============================================================================
# API Server with OrchestratorClient
# ============================================================================


class APIServer(TrainingOpsMixin, WeightsMixin, InferenceEndpointsMixin, HealthMixin):
    """
    API server with integrated OrchestratorClient.

    Combines FastAPI REST endpoints with OrchestratorClient for direct
    communication with the Engine backend.
    """

    def __init__(
        self,
        engine_input_addr: str = "tcp://127.0.0.1:6000",
        engine_output_addr: str = "tcp://127.0.0.1:6001",
        default_timeout: float = 120.0,
        output_dir: str = "outputs",
        base_model: Optional[str] = None,
        storage_limit: str = "10TB",
        max_sampling_loras: int = 3,
        idle_session_timeout: float = 7200.0,
        skip_initial_checkpoint: bool = False,
        sync_inference_method: str = "nccl_broadcast",
    ):
        """
        Initialize Unified API Server.

        Args:
            engine_input_addr: Engine input address (ROUTER binds here)
            engine_output_addr: Engine output address (PULL connects here)
            default_timeout: Default timeout for engine operations
            output_dir: Output directory for checkpoints and sampler weights (must be on shared filesystem)
            base_model: Base model name that this server is configured for (e.g., 'Qwen/Qwen2.5-3B-Instruct').
                        Used to validate create_model requests.
            storage_limit: Maximum disk usage for output_dir (e.g., '1GB', '500MB'). Save operations
                          will fail with StorageLimitError when limit is exceeded. Default: 10TB.
            max_sampling_loras: Maximum number of LoRA adapters to keep loaded for sampling (default: 3).
                               When this limit is reached, the oldest adapter will be unloaded.
            idle_session_timeout: Idle session timeout in seconds (default: 7200.0 = 2 hours).
                                 Sessions inactive for this duration will be automatically cleaned up.
            skip_initial_checkpoint: If True, skip auto-saving initial checkpoint on first create_model call.
                                    This is useful for full-weight mode to avoid memory issues during save.
            sync_inference_method: Method for syncing weights to inference endpoints.
                               Currently only 'nccl_broadcast' is supported.
        """
        self.engine_input_addr = engine_input_addr
        self.engine_output_addr = engine_output_addr
        self.default_timeout = default_timeout
        self.output_dir = output_dir
        self.base_model = base_model
        self.storage_limit = storage_limit
        self.max_sampling_loras = max_sampling_loras
        self.sync_inference_method = sync_inference_method

        # OrchestratorClient
        self.orchestrator_client: Optional[OrchestratorClient] = None

        # Running state
        self._running = False

        # Inference endpoints registry (unified)
        # List of inference endpoints for both LoRA loading and weight synchronization
        self.inference_endpoints: List[InferenceEndpoint] = []
        self._default_sync_quantization: Optional[Dict[str, Any]] = None

        # Weight synchronizers for each endpoint (persistent NCCL connections)
        # Maps endpoint_key -> WeightSynchronizer
        # endpoint_key = f"{host}:{port}"
        self.weight_synchronizers: Dict[str, Any] = {}

        # Model registry for tracking created models
        # Set of model_ids that have been registered via create_model endpoint
        # "default" is pre-registered to allow direct API usage without create_model
        self.registered_model_ids: set = {"default"}

        # Model config registry for storing LoRA configs
        # Maps model_id -> {"base_model": str, "lora_config": dict}
        # Used by /api/v1/weights_info to return checkpoint metadata
        self.model_configs: Dict[str, Dict[str, Any]] = {}

        # Sampling session LoRA tracking (per-model_id, LRU order - oldest first)
        # Maps model_id -> List of (lora_name, model_path) tuples for loaded adapters
        # Each model_id has its own set of tracked adapters to support parallel training runs
        self.loaded_sampling_loras: Dict[str, List[tuple]] = {}

        # Maximum number of adapters per model_id for sampling
        self.max_adapters_per_model: int = 3

        # Session activity tracking for idle cleanup
        # Maps model_id -> last activity timestamp (time.time())
        self.session_last_activity: Dict[str, float] = {}

        # Idle session timeout in seconds
        self.idle_session_timeout: float = idle_session_timeout

        # Background task for idle session cleanup
        self._idle_cleanup_task: Optional[asyncio.Task] = None

        # FutureStore for two-phase request pattern
        # Stores async request results with TTL-based expiration
        # Organized by model_id for session-based cleanup
        self.future_store: Optional[FutureStore] = None

        # Flag to track if initial checkpoint "000000" has been saved
        # This should only happen once when the first model is created after server start
        # Set to True if skip_initial_checkpoint is True to prevent auto-save in create_model
        self._initial_checkpoint_saved: bool = skip_initial_checkpoint

        logger.info(
            f"APIServer initialized: "
            f"engine_input={engine_input_addr}, engine_output={engine_output_addr}, "
            f"base_model={base_model}, storage_limit={storage_limit}, "
            f"idle_session_timeout={self.idle_session_timeout}s"
        )

    def validate_model_id(self, model_id: str) -> None:
        """
        Validate that a model_id has been registered via create_model.

        The "default" model_id is always valid (pre-registered).
        Other model_ids must be registered via create_model endpoint.

        Args:
            model_id: The model identifier to validate

        Raises:
            HTTPException: If model_id has not been registered
        """
        if model_id not in self.registered_model_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=MODEL_ID_NOT_REGISTERED_ERROR.format(model_id=model_id)
            )

    def _update_session_activity(self, model_id: str) -> None:
        """
        Update the last activity time for a session.

        Called on every request that uses a model_id to track session activity.
        Used by the idle cleanup task to detect dead sessions.

        Args:
            model_id: The model identifier to update activity for
        """
        self.session_last_activity[model_id] = time.time()

    def _require_engine(self) -> None:
        """Raise 503 if the engine is not running."""
        if not self._running or not self.orchestrator_client:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API server not running")

    @staticmethod
    def _flatten_api_data(data_list) -> List[Dict[str, Any]]:
        """Convert API datum list to flat engine data dicts."""
        result = []
        for datum in data_list:
            plain = datum.to_plain_dict()
            d: Dict[str, Any] = {}
            if plain.get("model_input"):
                d.update(plain["model_input"])
            if plain.get("loss_fn_inputs"):
                d.update(plain["loss_fn_inputs"])
            result.append(d)
        return result

    @staticmethod
    def _build_loss_fn_outputs(result: Dict[str, Any]):
        """Build (loss_fn_outputs, loss_fn_output_type) from engine result."""
        per_sample_outputs = result.get("per_sample_outputs", [])
        per_sample_k3 = result.get("per_sample_k3", [])

        if per_sample_outputs:
            outputs = []
            for i, sample in enumerate(per_sample_outputs):
                logprobs = sample.get("logprobs", [])
                elementwise_loss = sample.get("elementwise_loss", [])
                k3_val = per_sample_k3[i] if i < len(per_sample_k3) else None
                outputs.append(
                    LossFnOutput(
                        logprobs=TensorData(data=logprobs, dtype="float32", shape=[len(logprobs)]),
                        elementwise_loss=TensorData(data=elementwise_loss, dtype="float32", shape=[len(elementwise_loss)]),
                        k3=k3_val,
                    )
                )
            return outputs, "CrossEntropyLossReturn"

        # When no per-sample outputs, but we have per_sample_k3, create one output per sample
        if per_sample_k3:
            outputs = [LossFnOutput(loss=result.get("loss", 0.0), k3=k3) for k3 in per_sample_k3]
            return outputs, "single_loss"

        return [LossFnOutput(loss=result.get("loss", 0.0))], "single_loss"

    @staticmethod
    def _build_info(result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract auto-load info from engine result."""
        if result.get("auto_loaded"):
            return {"auto_loaded": True, "auto_load_path": result.get("auto_load_path")}
        return {}

    async def _cleanup_session(self, model_id: str) -> None:
        """
        Clean up all server-side state for a session.

        This removes the session from all tracking structures and unloads
        adapters from inference endpoints. For full-weights training mode,
        it also sends a kill_session command to workers to reset their state.

        Args:
            model_id: The model identifier to clean up
        """
        logger.info(f"Cleaning up session: {model_id}")

        # For full-weights training mode, send kill_session to workers to reset their state
        # This ensures workers don't reject new sessions due to stale active session
        try:
            engine_request = OrchestratorRequest(
                operation="kill_session",
                payload=KillSessionData(
                    model_id=model_id,
                    save_checkpoint=False,  # Don't save checkpoint on idle cleanup
                ),
            )

            response_future = await self.orchestrator_client.send_request(engine_request)
            output = await self._wait_for_response(
                response_future,
                engine_request.request_id,
                timeout=60.0,  # Shorter timeout for cleanup
                timeout_message=f"Kill session timeout during cleanup for {model_id}",
            )

            result = output.outputs[0] if output.outputs else {}
            if result.get("success"):
                logger.info(f"Workers acknowledged session cleanup for {model_id}")
            else:
                logger.warning(f"Workers returned non-success for session cleanup: {result.get('message', 'unknown')}")

        except Exception as e:
            # Log but don't fail - we still want to clean up local state
            logger.warning(f"Failed to notify workers of session cleanup for {model_id}: {e}")

        # Unload sampling adapters from SGL inference endpoints BEFORE removing tracking
        # This ensures we actually send unload requests to SGL
        try:
            unloaded = await self._unload_adapters_for_model(model_id)
            if unloaded > 0:
                logger.info(f"Unloaded {unloaded} sampling adapter(s) from inference endpoints for {model_id}")
        except Exception as e:
            logger.warning(f"Failed to unload sampling adapters for {model_id}: {e}")

        # Remove from tracking structures
        # Note: _unload_adapters_for_model clears loaded_sampling_loras[model_id] to [],
        # so we pop to fully remove the entry
        self.registered_model_ids.discard(model_id)
        self.model_configs.pop(model_id, None)
        self.session_last_activity.pop(model_id, None)
        self.loaded_sampling_loras.pop(model_id, None)

        # Clean up any pending futures for this model/session
        if self.future_store:
            deleted_count = await self.future_store.delete_by_model(model_id)
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} futures for session {model_id}")

        # Training adapters on workers are managed via LRU eviction
        # No explicit unload needed - workers will evict when memory pressure occurs

        logger.info(f"Session {model_id} cleaned up successfully")

    async def _cleanup_idle_sessions(self) -> None:
        """
        Background task to periodically clean up idle sessions.

        Runs every 60 seconds and cleans up sessions that haven't been active
        for longer than idle_session_timeout (default: 30 minutes).
        """
        logger.info(f"Starting idle session cleanup task (timeout={self.idle_session_timeout}s)")

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                current_time = time.time()
                idle_model_ids = [
                    (model_id, last_activity)
                    for model_id, last_activity in list(self.session_last_activity.items())
                    if current_time - last_activity > self.idle_session_timeout
                ]

                for model_id, last_activity in idle_model_ids:
                    logger.info(
                        f"Cleaning up idle session: {model_id} "
                        f"(idle for {current_time - last_activity:.0f}s)"
                    )
                    await self._cleanup_session(model_id)

            except asyncio.CancelledError:
                logger.info("Idle session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in idle session cleanup: {e}")
                # Continue running despite errors

    async def _wait_for_response(
        self, response_future: asyncio.Future, request_id: str, timeout: float, timeout_message: str = "Engine timeout"
    ):
        """
        Wait for engine response with timeout and proper cleanup.

        Args:
            response_future: Future from orchestrator_client.send_request()
            request_id: Request ID for cleanup on timeout
            timeout: Timeout in seconds
            timeout_message: Custom timeout message for error

        Returns:
            OrchestratorOutputs from the engine

        Raises:
            HTTPException: On timeout (504) or engine error (500)
        """
        try:
            output = await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            # Clean up the pending request to prevent memory leak and notify engine
            await self.orchestrator_client.cancel_request(request_id, send_abort=True)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"{timeout_message} after {timeout}s"
            )

        if output.error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Engine error: {output.error}"
            )

        return output

    async def start(self):
        """Start the API server and OrchestratorClient."""
        if self._running:
            logger.warning("APIServer already started")
            return

        logger.info("Starting APIServer...")

        # Initialize OrchestratorClient
        self.orchestrator_client = OrchestratorClient(
            input_addr=self.engine_input_addr,
            output_addr=self.engine_output_addr,
            output_queue_maxsize=1000,
        )
        await self.orchestrator_client.start()

        # Initialize and start future store for two-phase pattern
        max_concurrent = int(os.environ.get("XORL_FUTURE_STORE_MAX_CONCURRENT", "128"))
        self.future_store = FutureStore(
            default_ttl=3600.0,  # 1 hour
            max_concurrent=max_concurrent,
            cleanup_interval=60.0,  # 1 minute
        )
        await self.future_store.start()

        # Start background task for idle session cleanup
        self._idle_cleanup_task = asyncio.create_task(self._cleanup_idle_sessions())

        self._running = True
        logger.info("APIServer started successfully")

    async def stop(self):
        """Stop the API server and OrchestratorClient."""
        if not self._running:
            return

        logger.info("Stopping APIServer...")

        # Cancel the idle cleanup task
        if self._idle_cleanup_task:
            self._idle_cleanup_task.cancel()
            try:
                await self._idle_cleanup_task
            except asyncio.CancelledError:
                pass
            self._idle_cleanup_task = None

        # Stop future store
        if self.future_store:
            await self.future_store.stop()

        if self.orchestrator_client:
            await self.orchestrator_client.stop()

        self._running = False
        logger.info("APIServer stopped")


# ============================================================================
# Global API Server Instance
# ============================================================================

import xorl.server.api_server._state as _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""

    # Startup
    logger.info("Starting Unified API Server...")
    _state.api_server = APIServer(
        engine_input_addr="tcp://127.0.0.1:6000",
        engine_output_addr="tcp://127.0.0.1:6001",
        default_timeout=120.0,
    )
    await _state.api_server.start()

    yield

    # Shutdown
    logger.info("Shutting down Unified API Server...")
    if _state.api_server:
        await _state.api_server.stop()
        _state.api_server = None


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Unified Training API",
    description="REST API with integrated OrchestratorClient for distributed LLM training",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Run the API server with uvicorn."""
    uvicorn.run(
        "xorl.server.api_server_unified:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
