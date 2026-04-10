"""
API Request/Response Types for REST API.

Pydantic type definitions for FastAPI endpoints in the unified API server.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


# ============================================================================
# Two-Phase Async Response Types
# ============================================================================


# ============================================================================
# Type Aliases
# ============================================================================


class TensorData(BaseModel):
    """Tensor data with dtype and shape metadata.

    This format is used by xorl_client/tinker to pass tensor data with type information.
    Example: {"data": [0, 0, 1, 1], "dtype": "float32", "shape": [4]}
    """

    data: List[Union[int, float, str]] = Field(..., description="Flattened tensor data")
    dtype: str = Field(default="float32", description="Data type (e.g., 'float32', 'int64')")
    shape: List[int] = Field(..., description="Tensor shape")

    def tolist(self) -> List[Union[int, float, str]]:
        """Return the data as a list (tinker API compatibility)."""
        return self.data


InputType = Union[List[int], List[float], List[str], TensorData]


# ============================================================================
# Data Structures and Helper Classes
# ============================================================================


class Datum(BaseModel):
    """Single training example with model inputs and loss function inputs."""

    model_input: Dict[str, InputType] = Field(..., description="Model input tensors (input_ids, position_ids, etc.)")
    loss_fn_inputs: Dict[str, InputType] = Field(..., description="Loss function input tensors (e.g., labels)")

    @model_validator(mode="before")
    @classmethod
    def _convert_tinker_format(cls, data):
        """Accept tinker's {chunks: [{tokens: [...]}]} and convert to {input_ids: [...]}."""
        if isinstance(data, dict):
            mi = data.get("model_input")
            if isinstance(mi, dict) and "chunks" in mi:
                tokens = []
                for chunk in mi["chunks"]:
                    if isinstance(chunk, dict) and "tokens" in chunk:
                        tokens.extend(chunk["tokens"])
                data["model_input"] = {"input_ids": tokens}
        return data

    def to_plain_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary with TensorData objects converted to lists.

        This is used by the data processor which expects plain lists, not TensorData objects.
        """

        def convert_value(v: InputType) -> List[Union[int, float, str]]:
            if isinstance(v, TensorData):
                return v.data
            return v

        return {
            "model_input": {k: convert_value(v) for k, v in self.model_input.items()},
            "loss_fn_inputs": {k: convert_value(v) for k, v in self.loss_fn_inputs.items()},
        }


class DatumInput(BaseModel):
    """Input containing list of data examples and loss function."""

    model_config = ConfigDict(extra="ignore")

    data: List[Datum] = Field(..., description="List of training/inference examples")
    loss_fn: str = Field(default="causallm_loss", description="Loss function type")
    loss_fn_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Global loss function parameters (e.g., eps_clip, use_tis for PPO)"
    )
    routed_experts: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Per-datum MOE routing data for R3 (Rollout Routing Replay). "
            "Each element can be either:\n"
            "  - Nested list with shape [num_tokens, num_layers, topk]\n"
            "  - Base64-encoded int32 numpy array (from SGLang) with shape info in meta_info\n"
            "When provided, training will replay these expert selections instead of recomputing "
            "top-k routing, ensuring consistency with inference routing decisions."
        ),
    )
    routed_expert_logits: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Per-datum MOE routing weights for R3. Same format as routed_experts but contains "
            "float32 softmax weights instead of int32 expert indices. When provided alongside "
            "routed_experts, training uses these exact routing weights from inference instead of "
            "recomputing softmax, ensuring exact numerical parity."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _map_tinker_fields(cls, data):
        """Map tinker's loss_fn_config to loss_fn_params."""
        if isinstance(data, dict) and "loss_fn_config" in data and "loss_fn_params" not in data:
            data["loss_fn_params"] = data.pop("loss_fn_config")
        return data


class LossFnOutput(BaseModel):
    """Single loss function output.

    For standard loss functions, only 'loss' is populated.
    For cross_entropy with return_per_token=True (tinker API compatibility),
    'logprobs' and 'elementwise_loss' contain per-token TensorData.

    Note: This model excludes None values from serialization to match tinker's
    Dict[str, TensorData] format which only contains populated fields.
    """

    loss: Optional[float] = Field(default=None, description="Loss value (for backward compatibility)")
    logprobs: Optional[TensorData] = Field(default=None, description="Per-token log probabilities")
    elementwise_loss: Optional[TensorData] = Field(default=None, description="Per-token cross entropy loss")
    k3: Optional[float] = Field(default=None, description="Per-sample K3 KL divergence estimate")

    @field_serializer("loss")
    def _serialize_loss_as_tensor_data(self, v, _info):
        """Wrap scalar loss as TensorData dict for tinker SDK compatibility.

        Tinker SDK expects LossFnOutput = Dict[str, TensorData], so scalar
        loss values must be serialized as TensorData on the wire.
        """
        if v is not None:
            return {"data": [v], "dtype": "float32", "shape": [1]}
        return v

    def model_dump(self, **kwargs):
        """Override to always exclude None values for tinker compatibility."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        """Override to always exclude None values for JSON serialization."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)


class AdamParams(BaseModel):
    """AdamW optimizer parameters."""

    learning_rate: float = Field(default=0.0001, description="Learning rate")
    beta1: float = Field(default=0.9, description="First moment coefficient")
    beta2: float = Field(default=0.95, description="Second moment coefficient")
    eps: float = Field(default=1e-12, description="Numerical stability term")
    weight_decay: float = Field(default=0.0, description="Weight decay (decoupled)")
    grad_clip_norm: float = Field(default=0.0, description="Gradient clipping norm (0.0 = no clipping)")


# ============================================================================
# Inference Operations
# ============================================================================


class ForwardRequest(BaseModel):
    """API request for forward operation."""

    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(
        default="default", description="Model identifier (must be created via /api/v1/create_model first)"
    )
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for request ordering")
    forward_input: DatumInput = Field(..., description="Forward input data")


class ForwardResponse(BaseModel):
    """API response for forward operation."""

    loss_fn_output_type: str = Field(..., description="Type of loss function output")
    loss_fn_outputs: List[LossFnOutput] = Field(..., description="Loss function outputs")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")
    info: Dict[str, Any] = Field(..., description="Additional information")


# ============================================================================
# Training Operations
# ============================================================================


class ForwardBackwardRequest(BaseModel):
    """API request for forward-backward operation."""

    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(
        default="default", description="Model identifier (must be created via /api/v1/create_model first)"
    )
    seq_id: Optional[int] = Field(
        default=None,
        description="Sequence ID for request ordering (ensures forward_backward executes before optim_step)",
    )
    forward_backward_input: DatumInput = Field(..., description="Forward-backward input data")


class ForwardBackwardResponse(BaseModel):
    """API response for forward-backward operation."""

    loss_fn_output_type: str = Field(..., description="Type of loss function output")
    loss_fn_outputs: List[LossFnOutput] = Field(..., description="Loss function outputs")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")
    info: Dict[str, Any] = Field(..., description="Additional information")


class OptimStepRequest(BaseModel):
    """API request for optimizer step."""

    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(
        default="default", description="Model identifier (must be created via /api/v1/create_model first)"
    )
    seq_id: Optional[int] = Field(
        default=None,
        description="Sequence ID for request ordering (ensures forward_backward executes before optim_step)",
    )
    adam_params: AdamParams = Field(default_factory=AdamParams, description="AdamW optimizer parameters")
    gradient_clip: Optional[float] = Field(default=None, description="Gradient clipping value")


class OptimStepResponse(BaseModel):
    """API response for optimizer step."""

    metrics: Dict[str, Any] = Field(..., description="Optimization metrics (grad_norm, learning_rate)")
    info: Dict[str, Any] = Field(..., description="Additional information")


# ============================================================================
# Generation Operations
# ============================================================================


# ============================================================================
# Weights Management
# ============================================================================


class SaveWeightsRequest(BaseModel):
    """API request for saving weights (checkpoint).

    Endpoint: POST /api/v1/save_weights
    """

    model_id: str = Field(default="default", description="Model identifier")
    path: Optional[str] = Field(
        default=None, description="Checkpoint name (e.g., 'checkpoint-001'). Auto-generated if not specified."
    )
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for request ordering")
    # Note: save_optimizer is always True - we always save full state for checkpointing


class SaveWeightsResponse(BaseModel):
    """API response for saving weights.

    Returns a xorl:// URI pointing to the saved checkpoint.
    """

    path: str = Field(..., description="Xorl URI (e.g., xorl://default/weights/checkpoint-001)")
    warning: Optional[str] = Field(default=None, description="Warning message if checkpoint already existed")


class LoadWeightsRequest(BaseModel):
    """API request for loading weights.

    Endpoint: POST /api/v1/load_weights
    """

    model_id: str = Field(default="default", description="Model identifier")
    path: str = Field(..., description="Xorl URI to load from (e.g., xorl://default/weights/checkpoint-001)")
    optimizer: bool = Field(default=True, description="Whether to load optimizer state")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for request ordering")


class LoadWeightsResponse(BaseModel):
    """API response for loading weights."""

    path: str = Field(..., description="Xorl URI that was loaded")


class WeightsInfoRequest(BaseModel):
    """API request for getting checkpoint metadata.

    Endpoint: POST /api/v1/weights_info

    This mirrors tinker's weights_info endpoint for compatibility.
    The xorl_path is the checkpoint URI (e.g., "xorl://default/weights/checkpoint-001").
    """

    xorl_path: str = Field(..., description="Xorl URI to the checkpoint")


class WeightsInfoResponse(BaseModel):
    """API response for checkpoint metadata.

    Returns minimal information needed to resume training from a checkpoint.
    """

    base_model: str = Field(..., description="Base model name (e.g., 'Qwen/Qwen2.5-3B-Instruct')")
    is_lora: bool = Field(default=True, description="Whether this is a LoRA checkpoint")
    lora_rank: Optional[int] = Field(default=None, description="LoRA rank (if is_lora=True)")


class CreateModelRequest(BaseModel):
    """API request for creating a new model."""

    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(default="default", description="Model identifier")
    base_model: str = Field(..., description="Base model name (e.g., 'Qwen/Qwen2.5-3B-Instruct')")
    lora_config: Dict[str, Any] = Field(default_factory=dict, description="LoRA configuration (rank, alpha, etc.)")

    @model_validator(mode="before")
    @classmethod
    def _map_tinker_fields(cls, data):
        """Map tinker's session_id to model_id if model_id not provided."""
        if isinstance(data, dict):
            if "session_id" in data and "model_id" not in data:
                data["model_id"] = data["session_id"]
            # Tinker sends lora_config with "rank" key; normalize to also include lora_rank
            lora_cfg = data.get("lora_config")
            if isinstance(lora_cfg, dict) and "rank" in lora_cfg and "lora_rank" not in lora_cfg:
                lora_cfg["lora_rank"] = lora_cfg["rank"]
        return data


class CreateModelResponse(BaseModel):
    """API response for creating a model (Tinker-compatible)."""

    model_id: str = Field(..., description="Model identifier")
    type: Literal["create_model"] = Field(default="create_model", description="Response type identifier")


class UnloadModelRequest(BaseModel):
    """API request for unloading a model (Tinker-compatible)."""

    model_id: str = Field(..., description="Model identifier to unload")
    type: Literal["unload_model"] = Field(default="unload_model", description="Request type identifier")


class UnloadModelResponse(BaseModel):
    """API response for unloading a model (Tinker-compatible)."""

    model_id: str = Field(..., description="Model identifier that was unloaded")
    type: Optional[Literal["unload_model"]] = Field(default=None, description="Response type identifier")


class SessionInfoResponse(BaseModel):
    """API response with session information for monitoring."""

    registered_models: List[str] = Field(..., description="List of registered model IDs (all ever registered)")
    active_sessions: int = Field(..., description="Number of active sessions")
    session_activity: Dict[str, float] = Field(
        default_factory=dict, description="Map of model_id to last activity timestamp"
    )
    idle_timeout_seconds: float = Field(..., description="Idle session timeout in seconds")
    loaded_sampling_adapters: Dict[str, int] = Field(
        default_factory=dict, description="Map of model_id to number of loaded sampling adapters"
    )
    # Training adapter info (from worker's adapter manager)
    loaded_training_adapters: List[str] = Field(
        default_factory=list, description="List of training adapters currently loaded in GPU memory"
    )
    max_training_adapters: int = Field(
        default=0, description="Maximum number of training adapters that can be loaded (LRU eviction threshold)"
    )
    current_training_adapter: Optional[str] = Field(
        default=None, description="Currently active training adapter (if any)"
    )


# ============================================================================
# Adapter State Save/Load
# ============================================================================


class SaveAdapterStateRequest(BaseModel):
    """API request for saving adapter state (weights + optimizer)."""

    model_id: str = Field(..., description="Adapter/session identifier to save")
    path: Optional[str] = Field(
        default=None, description="Directory to save adapter state to. Auto-generated if not specified."
    )
    save_optimizer: bool = Field(default=True, description="Whether to save optimizer state for resuming training")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for request ordering")


class SaveAdapterStateResponse(BaseModel):
    """API response for saving adapter state."""

    success: bool = Field(..., description="Whether save was successful")
    path: str = Field(..., description="Directory where adapter state was saved")
    model_id: str = Field(..., description="Model identifier that was saved")
    step: int = Field(..., description="Global step at save time")


class RegisterWorkersRequest(BaseModel):
    """API request for registering inference workers for a model."""

    model_path: str = Field(..., description="Model path (e.g., 'xorl://model-123/step-100')")
    worker_urls: List[str] = Field(..., description="List of worker URLs")


class RegisterWorkersResponse(BaseModel):
    """API response for registering workers."""

    status: str = Field(..., description="Registration status")
    num_workers: int = Field(..., description="Number of workers registered")


class SaveWeightsForSamplerRequest(BaseModel):
    """API request for saving weights in inference-compatible format."""

    model_id: str = Field(
        default="default", description="Model identifier (must be created via /api/v1/create_model first)"
    )
    name: str = Field(..., description="Checkpoint name (e.g., 'step-100')")


class SaveWeightsForSamplerResponse(BaseModel):
    """API response for saving weights for sampler."""

    path: str = Field(
        ..., description="Xorl URI for the saved checkpoint (e.g., 'xorl://model-0/sampler_weights/step-100')"
    )


# ============================================================================
# Checkpoint Management Operations
# ============================================================================


class CheckpointInfo(BaseModel):
    """Information about a single checkpoint."""

    checkpoint_id: str = Field(
        ..., description="The checkpoint ID (e.g., 'weights/model_id/name' or 'sampler_weights/name')"
    )
    checkpoint_type: Literal["training", "sampler"] = Field(..., description="The type of checkpoint")
    time: str = Field(..., description="ISO format timestamp when the checkpoint was created")
    path: str = Field(..., description="The xorl:// path to the checkpoint")
    size_bytes: Optional[int] = Field(default=None, description="The size of the checkpoint in bytes")


class ListCheckpointsRequest(BaseModel):
    """API request for listing checkpoints."""

    model_id: str = Field(default="default", description="Model identifier")


class ListCheckpointsResponse(BaseModel):
    """API response for listing checkpoints.

    Returns all available checkpoints for the specified model.
    """

    checkpoints: List[CheckpointInfo] = Field(default_factory=list, description="List of available checkpoints")


class DeleteCheckpointRequest(BaseModel):
    """API request for deleting a checkpoint.

    The checkpoint_id should be in the format:
    - 'weights/{model_id}/{name}' for training checkpoints
    - 'sampler_weights/{name}' for sampler checkpoints (flat, no model_id)
    """

    model_id: str = Field(default="default", description="Model identifier (used for training checkpoints)")
    checkpoint_id: str = Field(
        ..., description="Checkpoint ID to delete (e.g., 'weights/model_id/ckpt-001' or 'sampler_weights/adapter-001')"
    )


class DeleteCheckpointResponse(BaseModel):
    """API response for deleting a checkpoint."""

    success: bool = Field(..., description="Whether the deletion was successful")
    deleted_path: Optional[str] = Field(default=None, description="The xorl:// path that was deleted")
    error: Optional[str] = Field(default=None, description="Error message if deletion failed")


# ============================================================================
# Full-Weights Training Session Management
# ============================================================================


class KillSessionRequest(BaseModel):
    """API request for killing a full-weights training session.

    In full-weights training mode (enable_lora=False), the server operates in
    single-tenant mode. This endpoint allows killing the active session to
    start a new one.
    """

    model_id: str = Field(..., description="Session to kill (must match active session)")
    save_checkpoint: bool = Field(default=True, description="Save checkpoint before killing")
    reset_weights: bool = Field(default=False, description="Reload base model weights after killing session")


class KillSessionResponse(BaseModel):
    """API response for killing a full-weights training session."""

    success: bool = Field(..., description="Whether the session was killed successfully")
    message: str = Field(..., description="Status message")
    checkpoint_path: Optional[str] = Field(
        default=None, description="Path to saved checkpoint (if save_checkpoint=True)"
    )


# ============================================================================
# System Operations
# ============================================================================


class HealthCheckResponse(BaseModel):
    """API response for health check."""

    status: str = Field(..., description="System status")
    engine_running: bool = Field(..., description="Whether engine is running")
    active_requests: int = Field(..., description="Number of active requests")
    total_requests: int = Field(..., description="Total requests processed")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


# ============================================================================
# Inference Endpoint Management
# ============================================================================


class InferenceEndpointServerInfo(BaseModel):
    """Server info from SGLang inference endpoint (/server_info)."""

    model_path: Optional[str] = Field(default=None, description="Model path loaded on the server")
    served_model_name: Optional[str] = Field(default=None, description="Served model name")
    tp_size: Optional[int] = Field(default=None, description="Tensor parallelism size")
    quantization: Optional[str] = Field(
        default=None, description="Quantization method reported by SGLang (e.g., 'fp8')"
    )
    quantization_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full HF quantization_config dict auto-detected from model's config.json",
    )
    dtype: Optional[str] = Field(default=None, description="Model dtype (e.g., 'auto', 'bfloat16')")
    enable_lora: Optional[bool] = Field(default=None, description="Whether LoRA is enabled")
    max_lora_rank: Optional[int] = Field(default=None, description="Maximum LoRA rank supported")
    version: Optional[str] = Field(default=None, description="SGLang version")


class InferenceEndpoint(BaseModel):
    """Represents a single inference endpoint."""

    host: str = Field(..., description="Hostname or IP address of the inference endpoint")
    port: int = Field(..., description="Port number of the SGLang server")
    worker_port: int = Field(..., description="Port number of the inference worker (HTTP API wrapper)")
    world_size: int = Field(default=1, description="Number of workers at this endpoint")
    healthy: bool = Field(default=True, description="Whether the endpoint is healthy")
    server_info: Optional[InferenceEndpointServerInfo] = Field(
        default=None, description="Server info from the inference endpoint"
    )

    @property
    def worker_url(self) -> str:
        """Compute the inference worker URL for /api/update_weights calls."""
        return f"http://{self.host}:{self.worker_port}"


class AddInferenceEndpointRequest(BaseModel):
    """API request for adding an inference endpoint."""

    host: str = Field(..., description="Hostname or IP address of the inference endpoint")
    port: int = Field(..., description="Port number of the SGLang server")
    worker_port: Optional[int] = Field(
        default=None,
        description="Port number of the inference worker (if None, defaults to port - 1 for backwards compatibility)",
    )
    world_size: int = Field(default=1, description="Number of workers at this endpoint")
    # Auto-sync configuration
    sync_weights: bool = Field(
        default=False, description="Whether to automatically sync weights to this endpoint after adding"
    )
    master_address: Optional[str] = Field(
        default=None, description="Master address for NCCL rendezvous (auto-detected if not provided)"
    )
    master_port: int = Field(default=0, description="Master port for NCCL rendezvous (0 selects an ephemeral port)")
    group_name: str = Field(default="weight_sync_group", description="Name of the NCCL process group")
    buffer_size_mb: int = Field(default=1024, description="Size of each transfer bucket in MB (to avoid OOM)")


class AddInferenceEndpointResponse(BaseModel):
    """API response for adding an inference endpoint."""

    success: bool = Field(..., description="Whether the endpoint was added successfully")
    message: str = Field(..., description="Status message")
    endpoint: Optional[InferenceEndpoint] = Field(default=None, description="The added endpoint info")
    weights_synced: bool = Field(default=False, description="Whether weights were synced to this endpoint")
    sync_message: Optional[str] = Field(default=None, description="Weight sync status message")


class ListInferenceEndpointsResponse(BaseModel):
    """API response for listing inference endpoints."""

    endpoints: List[InferenceEndpoint] = Field(..., description="List of registered inference endpoints")
    count: int = Field(..., description="Number of registered endpoints")


class RemoveInferenceEndpointRequest(BaseModel):
    """API request for removing an inference endpoint."""

    host: str = Field(..., description="Hostname or IP address of the inference endpoint")
    port: int = Field(..., description="Port number of the inference endpoint")


class RemoveInferenceEndpointResponse(BaseModel):
    """API response for removing an inference endpoint."""

    success: bool = Field(..., description="Whether the endpoint was removed successfully")
    message: str = Field(..., description="Status message")


# ============================================================================
# Weight Synchronization
# ============================================================================


class SyncInferenceWeightsRequest(BaseModel):
    """API request for synchronizing full model weights to inference endpoints."""

    master_address: str = Field(
        default="", description="Master address for NCCL rendezvous (training server address). Auto-detected if empty."
    )
    master_port: int = Field(default=0, description="Master port for NCCL rendezvous (0 selects an ephemeral port)")
    group_name: str = Field(default="weight_sync_group", description="Name of the NCCL process group")
    buffer_size_mb: int = Field(default=1024, description="Size of each transfer bucket in MB (to avoid OOM)")
    flush_cache: bool = Field(
        default=True,
        description="Whether to flush inference KV cache after weight sync. "
        "Set to False to keep cached KV when weights haven't changed significantly.",
    )
    pause_mode: Literal["retract", "abort", "in_place"] = Field(
        default="retract",
        description="How to pause inference during weight sync. "
        "'retract' (default): retract running requests to waiting queue, re-execute after resume. "
        "'abort': abort and return all in-flight requests. "
        "'in_place': keep requests in place with KV cache (flush_cache must be False).",
    )
    weight_version: Optional[str] = Field(
        default=None,
        description="Version string for weight tracking",
    )
    quantization: Optional[Dict[str, Any]] = Field(
        default=None,
        description="HF quantization_config dict for weight sync. "
        "If None, uses the default set via set_sync_quantization or auto-detected from endpoint.",
    )


class EndpointSyncResult(BaseModel):
    """Result of syncing weights to a single endpoint."""

    host: str = Field(..., description="Endpoint host")
    port: int = Field(..., description="Endpoint port")
    success: bool = Field(..., description="Whether sync succeeded")
    message: str = Field(default="", description="Status or error message")


class SyncInferenceWeightsResponse(BaseModel):
    """API response for weight synchronization."""

    success: bool = Field(..., description="Whether all endpoints synced successfully")
    message: str = Field(..., description="Overall status message")
    transfer_time: float = Field(default=0.0, description="Total transfer time in seconds")
    total_bytes: int = Field(default=0, description="Total bytes transferred")
    num_parameters: int = Field(default=0, description="Number of parameters transferred")
    num_buckets: int = Field(default=0, description="Number of transfer buckets used")
    endpoints_synced: List[EndpointSyncResult] = Field(
        default_factory=list, description="Sync results for each endpoint"
    )


class SetSyncQuantizationRequest(BaseModel):
    """API request to set the default quantization format for weight sync.

    Accepts HF quantization_config format, e.g.:
        {"quant_method": "fp8", "weight_block_size": [128, 128]}
    or with modules_to_not_convert:
        {"quant_method": "fp8", "weight_block_size": [128, 128],
         "modules_to_not_convert": ["lm_head", "model.layers.0.input_layernorm", ...]}

    Set to null to disable quantization (bf16).
    """

    quantization: Optional[Dict[str, Any]] = Field(
        default=None,
        description="HF quantization_config dict. Must contain 'quant_method' (e.g. 'fp8'). "
        "Optional fields: 'weight_block_size' (default [128, 128]), "
        "'modules_to_not_convert' (list of module name prefixes to skip), "
        "'fmt' (e.g. 'e4m3'), 'activation_scheme'. "
        "Set to null for bf16 (no quantization).",
    )


class SetSyncQuantizationResponse(BaseModel):
    """API response for setting sync quantization format."""

    quantization: Optional[Dict[str, Any]] = Field(..., description="Current quantization config after update")
    message: str = Field(..., description="Status message")


# ============================================================================
# Sampling Session Management
# ============================================================================


class CreateSamplingSessionRequest(BaseModel):
    """API request for creating a sampling session.

    This loads the specified LoRA adapter on all inference workers.
    The model_path can be:
    - 'sampler_weights/adapter_name'
    - 'adapter_name' (just the name)
    """

    model_path: str = Field(
        ...,
        description="Path to the LoRA adapter (e.g., 'sampler_weights/adapter-001' or just 'adapter-001')",
    )


class CreateSamplingSessionResponse(BaseModel):
    """API response for creating a sampling session.

    Returns information about the loaded LoRA adapter.
    """

    success: bool = Field(..., description="Whether the session was created successfully")
    model_path: str = Field(..., description="The model path that was loaded")
    lora_name: str = Field(..., description="The name of the LoRA adapter that was loaded")
    message: Optional[str] = Field(default=None, description="Status or error message")


# ============================================================================
# Training Runs
# ============================================================================


class Cursor(BaseModel):
    """Pagination cursor information."""

    offset: int = Field(..., description="The offset used for pagination")
    limit: int = Field(..., description="The maximum number of items requested")
    total_count: int = Field(..., description="The total number of items available")


class TrainingRun(BaseModel):
    """Information about a training run.

    Note: In xorl_client, there is only a single training run with model_id="default".
    """

    training_run_id: str = Field(..., description="The unique identifier for the training run")
    base_model: str = Field(..., description="The base model name this model is derived from")
    model_owner: str = Field(default="local", description="The owner/creator of this model")
    is_lora: bool = Field(default=True, description="Whether this model uses LoRA")
    corrupted: bool = Field(default=False, description="Whether the model is in a corrupted state")
    lora_rank: Optional[int] = Field(default=None, description="The LoRA rank if this is a LoRA model")
    last_request_time: str = Field(..., description="ISO timestamp of the last request made to this model")
    last_checkpoint: Optional[CheckpointInfo] = Field(default=None, description="The most recent training checkpoint")
    last_sampler_checkpoint: Optional[CheckpointInfo] = Field(
        default=None, description="The most recent sampler checkpoint"
    )
    user_metadata: Optional[Dict[str, str]] = Field(
        default=None, description="Optional metadata about this training run"
    )


class TrainingRunsResponse(BaseModel):
    """Response from list_training_runs operation."""

    training_runs: List[TrainingRun] = Field(..., description="List of training runs")
    cursor: Cursor = Field(..., description="Pagination cursor information")


# ============================================================================
# Two-Phase Request Pattern Types
# ============================================================================


class UntypedAPIFuture(BaseModel):
    """Server response containing a request_id for async result retrieval.

    This is returned by endpoints like /api/v1/forward_backward in Phase 1
    of the two-phase request pattern.
    """

    request_id: str = Field(..., description="Unique identifier for this async request")
    model_id: Optional[str] = Field(default=None, description="Model identifier associated with this request")


class TryAgainResponse(BaseModel):
    """Response indicating the request is still being processed.

    The client should continue polling /api/v1/retrieve_future until
    a different response type is received.
    """

    type: Literal["try_again"] = Field(default="try_again", description="Response type identifier")
    request_id: str = Field(..., description="The request ID being polled")
    queue_state: Literal["active", "paused_capacity", "paused_rate_limit"] = Field(
        default="active", description="Current state of the request in the queue"
    )
    queue_state_reason: Optional[str] = Field(default=None, description="Reason for the current queue state")


class RequestFailedResponse(BaseModel):
    """Response indicating the request failed.

    Contains the error message and category to help the client decide
    whether to retry or report the error to the user.
    """

    error: str = Field(..., description="Human-readable error message")
    category: str = Field(default="unknown", description="Error category (unknown, server, user)")


class FutureRetrieveRequest(BaseModel):
    """Request to retrieve the result of an async operation.

    Sent to /api/v1/retrieve_future endpoint with the request_id obtained
    from the initial request (UntypedAPIFuture).
    """

    request_id: str = Field(..., description="The ID of the request to retrieve results for")


# Union of all possible responses from /api/v1/retrieve_future
FutureRetrieveResponse = Union[
    TryAgainResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    OptimStepResponse,
    SaveWeightsResponse,
    LoadWeightsResponse,
    SaveWeightsForSamplerResponse,
    CreateModelResponse,
    UnloadModelResponse,
    RequestFailedResponse,
]
