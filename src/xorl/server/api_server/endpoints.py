"""FastAPI endpoint functions using APIRouter."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

import xorl.server.api_server._state as _state
from xorl.server.api_server._state import require_api_server
from xorl.server.api_server.api_types import (
    AddInferenceEndpointRequest,
    AddInferenceEndpointResponse,
    CreateModelRequest,
    CreateModelResponse,
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    DeleteCheckpointRequest,
    DeleteCheckpointResponse,
    ErrorResponse,
    ForwardBackwardRequest,
    ForwardRequest,
    FutureRetrieveRequest,
    HealthCheckResponse,
    KillSessionRequest,
    KillSessionResponse,
    ListCheckpointsRequest,
    ListCheckpointsResponse,
    ListInferenceEndpointsResponse,
    LoadWeightsRequest,
    OptimStepRequest,
    RemoveInferenceEndpointRequest,
    RemoveInferenceEndpointResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    SessionInfoResponse,
    SetSyncQuantizationRequest,
    SetSyncQuantizationResponse,
    SyncInferenceWeightsRequest,
    SyncInferenceWeightsResponse,
    TrainingRunsResponse,
    UnloadModelRequest,
    UnloadModelResponse,
    UntypedAPIFuture,
    WeightsInfoRequest,
    WeightsInfoResponse,
)
from xorl.server.protocol.api_orchestrator import OrchestratorRequest
from xorl.server.protocol.operations import KillSessionData

logger = logging.getLogger(__name__)

router = APIRouter()



# ============================================================================
# Training Endpoints (Two-Phase Pattern)
# ============================================================================


@router.post(
    "/api/v1/retrieve_future",
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Async Operations"],
)
async def retrieve_future_endpoint(request: FutureRetrieveRequest, server=Depends(require_api_server)):
    """
    Retrieve the result of a previously submitted async request (Phase 2).

    This is the polling endpoint for the two-phase async pattern. Call this
    with the request_id returned from Phase 1 endpoints (forward_backward,
    optim_step, save_weights, etc.).

    Returns:
    - The actual result dict if ready (e.g., ForwardBackwardResponse, OptimStepResponse)
    - TryAgainResponse if still in progress
    - RequestFailedResponse if the operation failed
    """
    return await server.retrieve_future(request)


@router.post(
    "/api/v1/forward_backward",
    response_model=UntypedAPIFuture,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Training Operations"],
)
async def forward_backward_endpoint(request: ForwardBackwardRequest, server=Depends(require_api_server)):
    """
    Execute forward and backward pass through the model (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the ForwardBackwardResponse result.
    """
    return await server.submit_forward_backward_async(request)


@router.post(
    "/api/v1/forward",
    response_model=UntypedAPIFuture,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Training Operations"],
)
async def forward_endpoint(request: ForwardRequest, server=Depends(require_api_server)):
    """
    Execute forward pass through the model (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the ForwardResponse result.
    """
    return await server.submit_forward_async(request)


@router.post(
    "/api/v1/optim_step",
    response_model=UntypedAPIFuture,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Training Operations"],
)
async def optim_step_endpoint(request: OptimStepRequest, server=Depends(require_api_server)):
    """
    Perform optimization step using AdamW optimizer (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the OptimStepResponse result.
    """
    return await server.submit_optim_step_async(request)


@router.post(
    "/api/v1/save_weights",
    response_model=UntypedAPIFuture,
    responses={
        409: {"model": ErrorResponse, "description": "Checkpoint already exists"},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        507: {"model": ErrorResponse, "description": "Storage limit exceeded"},
    },
    tags=["Weights Management"],
)
async def save_weights_endpoint(request: SaveWeightsRequest, server=Depends(require_api_server)):
    """
    Save model weights (and optimizer state) to persistent storage (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the SaveWeightsResponse result with xorl:// URI that can be used with load_weights.

    If checkpoint already exists, returns success with a warning message (no save performed).
    """
    server.validate_model_id(request.model_id)
    return await server.submit_save_weights_async(request)


@router.post(
    "/api/v1/load_weights",
    response_model=UntypedAPIFuture,
    responses={
        404: {"model": ErrorResponse, "description": "Checkpoint not found"},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Weights Management"],
)
async def load_weights_endpoint(request: LoadWeightsRequest, server=Depends(require_api_server)):
    """
    Load model weights from a saved checkpoint (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the LoadWeightsResponse result.

    Returns 404 Not Found if checkpoint does not exist - use /api/v1/list_checkpoints to see available checkpoints.
    """
    server.validate_model_id(request.model_id)
    return await server.submit_load_weights_async(request)


@router.post(
    "/api/v1/list_checkpoints",
    response_model=ListCheckpointsResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Weights Management"],
)
async def list_checkpoints_endpoint(request: ListCheckpointsRequest, server=Depends(require_api_server)):
    """
    List all available checkpoints.

    Returns both training checkpoints (weights/) and sampler checkpoints (sampler_weights/).
    Checkpoints are sorted by creation time (newest first).
    """
    return await server.list_checkpoints(request)


@router.get(
    "/api/v1/training_runs",
    response_model=TrainingRunsResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Training Runs"],
)
async def list_training_runs_endpoint(
    limit: int = 20,
    offset: int = 0,
    server=Depends(require_api_server),
):
    """
    List training runs with pagination support.

    Note: In xorl_client, there is typically only the "default" training run.
    Additional training runs are created via /api/v1/create_model.

    This endpoint provides API compatibility with tinker's list_training_runs.
    """
    return server.list_training_runs(limit=limit, offset=offset)


@router.post(
    "/api/v1/delete_checkpoint",
    response_model=DeleteCheckpointResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse, "description": "Cannot delete reserved checkpoint (000000)"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Weights Management"],
)
async def delete_checkpoint_endpoint(request: DeleteCheckpointRequest, server=Depends(require_api_server)):
    """
    Delete a checkpoint.

    The checkpoint_id should be in the format:
    - 'weights/{model_id}/{name}' for training checkpoints
    - 'sampler_weights/{name}' for sampler checkpoints (flat, no model_id)

    Note: The initial checkpoint '000000' is reserved and cannot be deleted.
    It preserves the original model state before any training operations.
    """
    return await server.delete_checkpoint(request)


@router.post(
    "/api/v1/weights_info",
    response_model=WeightsInfoResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Weights Management"],
)
async def weights_info_endpoint(request: WeightsInfoRequest, server=Depends(require_api_server)):
    """
    Get checkpoint metadata for resuming training.

    This endpoint returns the base_model and lora_rank needed to create
    a TrainingClient that can load the checkpoint. It mirrors tinker's
    /api/v1/weights_info endpoint for API compatibility.

    The xorl_path should be a xorl:// URI (e.g., "xorl://default/weights/checkpoint-001").
    """
    # Parse the xorl:// URI to get model_id
    # Format: xorl://model_id/weights/checkpoint_name
    xorl_path = request.xorl_path
    if not xorl_path.startswith("xorl://"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid xorl_path format: {xorl_path}. Expected xorl://model_id/weights/checkpoint_name",
        )

    parts = xorl_path[7:].split("/")  # Remove "xorl://"
    if len(parts) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid xorl_path format: {xorl_path}. Could not extract model_id",
        )

    model_id = parts[0]

    # Look up model config
    model_config = server.model_configs.get(model_id)

    if model_config is None:
        # Fall back to server's base_model if no specific config is stored
        if server.base_model is not None:
            logger.warning(
                f"No model config found for model_id '{model_id}', using server's base_model: {server.base_model}"
            )
            return WeightsInfoResponse(
                base_model=server.base_model,
                is_lora=True,
                lora_rank=None,  # Unknown rank
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No model config found for model_id '{model_id}' and server has no default base_model configured",
            )

    # Extract lora_rank from lora_config
    lora_config = model_config.get("lora_config", {})
    lora_rank = lora_config.get("lora_rank")

    return WeightsInfoResponse(
        base_model=model_config["base_model"],
        is_lora=True,
        lora_rank=lora_rank,
    )


@router.post(
    "/api/v1/create_model",
    response_model=UntypedAPIFuture,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Model Management"],
)
async def create_model_endpoint(request: CreateModelRequest, server=Depends(require_api_server)):
    """
    Create a new model for training (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the CreateModelResponse result.

    This initializes the model on the training server but doesn't actually
    do anything yet in the current implementation. It's a placeholder for
    future multi-model support.

    The base_model in the request must match the server's configured base model.
    """
    if not server.future_store:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Future store not initialized")

    model_id = request.model_id

    async def process_create_model(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process create_model request and return result dict."""
        req = CreateModelRequest(**request_data)

        logger.info(f"Creating model: {req.model_id}, base_model={req.base_model}")

        # Register the model_id so subsequent /api/v1/* calls can use it
        server.registered_model_ids.add(req.model_id)

        # Store the model config for /api/v1/weights_info
        server.model_configs[req.model_id] = {
            "base_model": req.base_model,
            "lora_config": req.lora_config,
        }

        # Initialize session activity tracking
        server._update_session_activity(req.model_id)

        logger.info(f"Registered model_id: {req.model_id} with lora_config: {req.lora_config}")

        # Auto-save initial checkpoint "000000" only once per server lifetime
        # This preserves the initial model state (base LoRA weights) before any training
        # Subsequent create_model calls skip this since the base model hasn't changed
        if not server._initial_checkpoint_saved:
            try:
                save_request = SaveWeightsRequest(
                    model_id=req.model_id,
                    path=server.RESERVED_CHECKPOINT_NAME,  # "000000"
                )
                save_response = await server.save_weights(save_request)
                server._initial_checkpoint_saved = True
                logger.info(f"Auto-saved initial checkpoint for model_id={req.model_id}: {save_response.path}")
            except Exception as e:
                logger.warning(f"Failed to auto-save initial checkpoint for model_id={req.model_id}: {e}")
                # Don't fail create_model if checkpoint save fails - it's not critical
        else:
            logger.debug(f"Skipping initial checkpoint save for model_id={req.model_id} (already saved)")

        return CreateModelResponse(model_id=req.model_id).model_dump()

    # Submit to future store
    request_id = await server.future_store.create(
        model_id=model_id,
        request_type="create_model",
        process_fn=process_create_model,
        request_data=request.model_dump(),
    )

    return UntypedAPIFuture(
        request_id=request_id,
        model_id=model_id,
    )


@router.post(
    "/api/v1/unload_model",
    response_model=UntypedAPIFuture,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Session Management"],
)
async def unload_model_endpoint(request: UnloadModelRequest, server=Depends(require_api_server)):
    """
    Unload a model and end the training session (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the UnloadModelResponse result.

    This unloads the model weights and releases all resources associated with the session.
    It cleans up server-side state and sends a message to workers to unload the training
    adapter, freeing GPU memory.

    This is optional - sessions will also be automatically cleaned up after 30 minutes
    of inactivity.

    Args:
        request: UnloadModelRequest with model_id to unload

    Returns:
        UntypedAPIFuture with request_id for polling
    """
    if not server.future_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Future store not initialized"
        )

    model_id = request.model_id

    # Check if the session exists
    if model_id not in server.registered_model_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    async def process_unload_model(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process unload_model request and return result dict."""
        req_model_id = request_data["model_id"]
        logger.info(f"Unloading model: {req_model_id}")

        try:
            await server._cleanup_session(req_model_id)
            return UnloadModelResponse(
                model_id=req_model_id,
                type="unload_model",
            ).model_dump()
        except Exception as e:
            logger.error(f"Error unloading model {req_model_id}: {e}")
            raise

    # Submit to future store
    request_id = await server.future_store.create(
        model_id=model_id,
        request_type="unload_model",
        process_fn=process_unload_model,
        request_data=request.model_dump(),
    )

    return UntypedAPIFuture(
        request_id=request_id,
        model_id=model_id,
    )


@router.post(
    "/api/v1/kill_session",
    response_model=KillSessionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Session Management"],
)
async def kill_session_endpoint(request: KillSessionRequest, server=Depends(require_api_server)):
    """
    Kill the active full-weights training session.

    In full-weights training mode (enable_lora=False), the server operates in
    single-tenant mode where only one training session is allowed at a time.
    This endpoint kills the active session to allow starting a new one.

    For LoRA mode, this is a no-op since multi-tenancy is supported.

    Args:
        request: KillSessionRequest with model_id to kill and save_checkpoint flag

    Returns:
        KillSessionResponse with kill status and optional checkpoint path
    """
    model_id = request.model_id
    save_checkpoint = request.save_checkpoint

    logger.info(f"Killing session: {model_id}, save_checkpoint={save_checkpoint}")

    try:
        engine_request = OrchestratorRequest(
            operation="kill_session",
            payload=KillSessionData(
                model_id=model_id,
                save_checkpoint=save_checkpoint,
            ),
        )

        response_future = await server.orchestrator_client.send_request(engine_request)
        output = await server._wait_for_response(
            response_future,
            engine_request.request_id,
            timeout=120.0,
            timeout_message="Kill session timeout",
        )

        result = output.outputs[0] if output.outputs else {}

        return KillSessionResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            checkpoint_path=result.get("checkpoint_path"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error killing session {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to kill session: {e}"
        )


@router.get(
    "/api/v1/session_info",
    response_model=SessionInfoResponse,
    responses={
        503: {"model": ErrorResponse},
    },
    tags=["Session Management"],
)
async def session_info_endpoint(server=Depends(require_api_server)):
    """
    Get information about active training sessions.

    Returns information about all registered sessions, their activity status,
    and loaded adapters. Useful for monitoring and debugging.

    Returns:
        SessionInfoResponse with session information
    """
    # Build activity map (absolute timestamps)
    session_activity = dict(server.session_last_activity)

    # Build adapter count map for sampling
    loaded_adapters = {
        model_id: len(adapters)
        for model_id, adapters in server.loaded_sampling_loras.items()
    }

    # Query worker for training adapter info
    loaded_training_adapters: List[str] = []
    max_training_adapters = 0
    current_training_adapter: Optional[str] = None

    if server._running and server.orchestrator_client:
        try:
            engine_request = OrchestratorRequest(operation="get_adapter_info")
            response_future = await server.orchestrator_client.send_request(engine_request)
            output = await server._wait_for_response(
                response_future, engine_request.request_id, 10.0, "Get adapter info timeout"
            )

            if output.outputs:
                result = output.outputs[0] if isinstance(output.outputs, list) else output.outputs
                loaded_training_adapters = result.get("loaded_adapters", [])
                max_training_adapters = result.get("max_adapters", 0)
                current_training_adapter = result.get("current_adapter_id")
        except Exception as e:
            logger.warning(f"Failed to get adapter info from worker: {e}")

    return SessionInfoResponse(
        registered_models=list(server.registered_model_ids),
        active_sessions=len(server.session_last_activity),
        session_activity=session_activity,
        idle_timeout_seconds=server.idle_session_timeout,
        loaded_sampling_adapters=loaded_adapters,
        loaded_training_adapters=loaded_training_adapters,
        max_training_adapters=max_training_adapters,
        current_training_adapter=current_training_adapter,
    )


@router.post(
    "/api/v1/save_weights_for_sampler",
    response_model=UntypedAPIFuture,
    responses={
        409: {"model": ErrorResponse, "description": "Sampler weights already exist"},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        507: {"model": ErrorResponse, "description": "Storage limit exceeded"},
    },
    tags=["State Management"],
)
async def save_weights_for_sampler_endpoint(request: SaveWeightsForSamplerRequest, server=Depends(require_api_server)):
    """
    Save weights for sampling/inference (two-phase pattern).

    Returns UntypedAPIFuture immediately. Poll /api/v1/retrieve_future to get
    the SaveWeightsForSamplerResponse result.

    To load weights on inference workers, call /api/v1/create_sampling_session with the returned model_path.
    """
    server.validate_model_id(request.model_id)
    return await server.submit_save_weights_for_sampler_async(request)


@router.post(
    "/add_inference_endpoint",
    response_model=AddInferenceEndpointResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Inference Endpoint Management"],
)
async def add_inference_endpoint_endpoint(request: AddInferenceEndpointRequest, server=Depends(require_api_server)):
    """
    Add an inference endpoint for weights transfer.

    This endpoint registers an inference server that can receive model weights
    from the training server. Before adding, it performs a health check by
    sending a GET request to /health on the inference endpoint.

    If the endpoint is healthy and sync_weights is set to True, weights will
    be automatically synced to the new endpoint via NCCL broadcast.

    Parameters for auto-sync:
    - sync_weights: Whether to sync weights after adding (default: False)
    - master_address: Training server hostname for NCCL rendezvous (auto-detected if not provided)
    - master_port: Port for NCCL rendezvous (default: 29600)
    - group_name: NCCL process group name (default: weight_sync_group)
    - buffer_size_mb: Transfer bucket size in MB (default: 1024)
    """
    return await server.add_inference_endpoint(request)


@router.get(
    "/list_inference_endpoints",
    response_model=ListInferenceEndpointsResponse,
    tags=["Inference Endpoint Management"],
)
async def list_inference_endpoints_endpoint(server=Depends(require_api_server)):
    """
    List all registered inference endpoints.

    Performs health checks on all registered endpoints and removes any
    that are no longer healthy. Returns the list of healthy endpoints.
    """
    return await server.list_inference_endpoints()


@router.post(
    "/remove_inference_endpoint",
    response_model=RemoveInferenceEndpointResponse,
    responses={
        503: {"model": ErrorResponse},
    },
    tags=["Inference Endpoint Management"],
)
async def remove_inference_endpoint_endpoint(request: RemoveInferenceEndpointRequest, server=Depends(require_api_server)):
    """
    Remove an inference endpoint from the registry.

    Removes the specified endpoint from the list of registered inference
    endpoints. Returns success even if the endpoint was not found.
    """
    return server.remove_inference_endpoint(request)


@router.post(
    "/sync_inference_weights",
    response_model=SyncInferenceWeightsResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["Inference Endpoint Management"],
    include_in_schema=False,
)
@router.post(
    "/api/v1/sync_inference_weights",
    response_model=SyncInferenceWeightsResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["Inference Endpoint Management"],
)
async def sync_inference_weights_endpoint(request: SyncInferenceWeightsRequest, server=Depends(require_api_server)):
    """
    Synchronize full model weights to all registered inference endpoints via NCCL.

    Transfers the current model weights (with LoRA merged if applicable) to all
    registered inference endpoints. The sync method is configured via server arguments.

    NCCL groups are lazily created on first sync and reused across syncs.
    """
    return await server.sync_inference_weights(request)


@router.post(
    "/api/v1/set_sync_quantization",
    response_model=SetSyncQuantizationResponse,
    tags=["Inference Endpoint Management"],
)
async def set_sync_quantization_endpoint(request: SetSyncQuantizationRequest, server=Depends(require_api_server)):
    """
    Set the default quantization config for weight sync to inference endpoints.

    Accepts HF quantization_config dict, e.g.:
        {"quant_method": "fp8", "weight_block_size": [128, 128]}
    Optionally with modules_to_not_convert:
        {"quant_method": "fp8", "weight_block_size": [128, 128],
         "modules_to_not_convert": ["lm_head", "model.layers.0.input_layernorm", ...]}

    Set to null for bf16 (no quantization).
    This setting persists until changed and is used when sync_inference_weights
    doesn't specify an explicit quantization config.
    """
    return server.set_sync_quantization(request)


# ============================================================================
# Sampling Session Management Endpoints
# ============================================================================


@router.post(
    "/api/v1/create_sampling_session",
    response_model=CreateSamplingSessionResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Sampling Session Management"],
)
async def create_sampling_session_endpoint(request: CreateSamplingSessionRequest, server=Depends(require_api_server)):
    """
    Create a sampling session by loading a LoRA adapter on inference workers.

    This endpoint:
    1. Validates the model_path format
    2. Checks if the checkpoint exists on disk
    3. If already loaded, returns early (moves to MRU)
    4. If at max capacity, unloads the oldest adapter
    5. Loads the new adapter on all registered inference endpoints via /load_lora_adapter

    Example model_path formats:
    - "sampler_weights/adapter-001"
    - "adapter-001" (just the adapter name)

    Note: Sampler weights are stored flat under output_dir/sampler_weights/{name}/
    without model_id subdirectories (inference endpoints don't know about model_id).
    """
    return await server.create_sampling_session(request)


# ============================================================================
# Tinker Compatibility Endpoints
# ============================================================================


@router.post("/api/v1/create_session", tags=["Tinker Compatibility"])
async def create_session_endpoint():
    """Stub for tinker SDK session creation."""
    return {
        "type": "create_session",
        "session_id": str(uuid.uuid4()),
        "info_message": None,
        "warning_message": None,
        "error_message": None,
    }


@router.post("/api/v1/session_heartbeat", tags=["Tinker Compatibility"])
async def session_heartbeat_endpoint():
    """Stub for tinker SDK session heartbeat."""
    return {"type": "session_heartbeat"}


@router.get("/api/v1/healthz", tags=["Tinker Compatibility"])
async def healthz_endpoint():
    """Tinker-compatible health check."""
    if _state.api_server is None:
        return {"status": "not_initialized"}
    result = await _state.api_server.health_check()
    return {"status": "ok" if result.engine_running else "not_ready"}


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
)
async def health_check_endpoint():
    """
    Check system health and engine status.

    Queries Engine backend via OrchestratorClient.
    """
    if _state.api_server is None:
        return HealthCheckResponse(
            status="not_initialized",
            engine_running=False,
            active_requests=0,
            total_requests=0,
        )
    return await _state.api_server.health_check()


@router.post(
    "/sleep",
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["System"],
)
async def sleep_endpoint(server=Depends(require_api_server)):
    """
    Offload model and optimizer to CPU to free GPU memory.

    Sends request to Engine backend via OrchestratorClient.
    """
    return await server.sleep()


@router.post(
    "/wake_up",
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["System"],
)
async def wake_up_endpoint(server=Depends(require_api_server)):
    """
    Load model and optimizer back to GPU.

    Sends request to Engine backend via OrchestratorClient.
    """
    return await server.wake_up()


@router.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Unified Training API",
        "version": "2.0.0",
        "architecture": "FastAPI + OrchestratorClient",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "forward_backward": "/api/v1/forward_backward",
            "optim_step": "/api/v1/optim_step",
            "save_weights": "/api/v1/save_weights",
            "load_weights": "/api/v1/load_weights",
            "save_weights_for_sampler": "/api/v1/save_weights_for_sampler",
            "sleep": "/sleep",
            "wake_up": "/wake_up",
        },
    }
