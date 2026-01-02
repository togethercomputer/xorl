"""
API Request/Response Types for REST API.

Pydantic type definitions for FastAPI endpoints in the unified API server.
"""

from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field


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
    model_input: Dict[str, InputType] = Field(
        ...,
        description="Model input tensors (input_ids, position_ids, etc.)"
    )
    loss_fn_inputs: Dict[str, InputType] = Field(
        ...,
        description="Loss function input tensors (e.g., labels)"
    )

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
    data: List[Datum] = Field(
        ...,
        description="List of training/inference examples"
    )
    loss_fn: str = Field(default="causallm_loss", description="Loss function type")
    loss_fn_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Global loss function parameters (e.g., eps_clip, use_tis for PPO)"
    )


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
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    forward_input: DatumInput = Field(
        ...,
        description="Forward input data"
    )


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
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for request ordering (ensures forward_backward executes before optim_step)")
    forward_backward_input: DatumInput = Field(
        ...,
        description="Forward-backward input data"
    )


class ForwardBackwardResponse(BaseModel):
    """API response for forward-backward operation."""
    loss_fn_output_type: str = Field(..., description="Type of loss function output")
    loss_fn_outputs: List[LossFnOutput] = Field(..., description="Loss function outputs")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")
    info: Dict[str, Any] = Field(..., description="Additional information")


class OptimStepRequest(BaseModel):
    """API request for optimizer step."""
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for request ordering (ensures forward_backward executes before optim_step)")
    adam_params: AdamParams = Field(default_factory=AdamParams, description="AdamW optimizer parameters")
    gradient_clip: Optional[float] = Field(default=None, description="Gradient clipping value")


class OptimStepResponse(BaseModel):
    """API response for optimizer step."""
    metrics: Dict[str, Any] = Field(..., description="Optimization metrics (grad_norm, learning_rate)")
    info: Dict[str, Any] = Field(..., description="Additional information")


# ============================================================================
# Generation Operations
# ============================================================================

class Prompt(BaseModel):
    """Single prompt for generation."""
    input_ids: List[int] = Field(..., description="Input token IDs")


class SampleRequest(BaseModel):
    """API request for sampling/generation."""
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    prompts: List[Prompt] = Field(..., description="List of prompts to generate from")
    max_new_tokens: int = Field(default=256, description="Maximum number of tokens to generate")
    temperature: float = Field(default=1.0, description="Sampling temperature (higher = more random)")
    top_p: float = Field(default=1.0, description="Nucleus sampling threshold")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling threshold (None = disabled)")
    do_sample: bool = Field(default=True, description="Whether to use sampling vs greedy decoding")


class GeneratedOutput(BaseModel):
    """Single generated output."""
    prompt_length: int = Field(..., description="Length of input prompt in tokens")
    generated_tokens: List[int] = Field(..., description="Generated token IDs")
    generated_length: int = Field(..., description="Number of tokens generated")


class SampleResponse(BaseModel):
    """API response for sampling/generation."""
    outputs: List[GeneratedOutput] = Field(..., description="Generated outputs for each prompt")
    info: Dict[str, Any] = Field(..., description="Additional information (sample_time, etc.)")


# ============================================================================
# Weights Management
# ============================================================================

class SaveWeightsRequest(BaseModel):
    """API request for saving weights (checkpoint).

    Endpoint: POST /api/v1/save_weights
    """
    model_id: str = Field(default="default", description="Model identifier")
    path: Optional[str] = Field(default=None, description="Checkpoint name (e.g., 'checkpoint-001'). Auto-generated if not specified.")
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
    optimizer: bool = Field(default=False, description="Whether to load optimizer state")
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
    model_id: str = Field(default="default", description="Model identifier")
    base_model: str = Field(..., description="Base model name (e.g., 'Qwen/Qwen2.5-3B-Instruct')")
    lora_config: Dict[str, Any] = Field(..., description="LoRA configuration (rank, alpha, etc.)")


class CreateModelResponse(BaseModel):
    """API response for creating a model."""
    model_id: str = Field(..., description="Model identifier")
    status: str = Field(..., description="Creation status")


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
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    name: str = Field(..., description="Checkpoint name (e.g., 'step-100')")


class SaveWeightsForSamplerResponse(BaseModel):
    """API response for saving weights for sampler."""
    path: str = Field(..., description="Checkpoint path (local path)")
    model_path: str = Field(..., description="Model path for routing (e.g., 'xorl://model-123/step-100')")


class SaveLoRAOnlyRequest(BaseModel):
    """API request for saving only LoRA adapter weights."""
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    lora_path: Optional[str] = Field(default=None, description="LoRA save path (auto-generated if not specified)")


class SaveLoRAOnlyResponse(BaseModel):
    """API response for saving LoRA weights."""
    lora_path: str = Field(..., description="LoRA adapter path")


# ============================================================================
# Checkpoint Management Operations
# ============================================================================

class CheckpointInfo(BaseModel):
    """Information about a single checkpoint."""
    checkpoint_id: str = Field(..., description="The checkpoint ID (e.g., 'weights/000' or 'sampler_weights/step-100')")
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

    The checkpoint_id should be in the format 'weights/{name}' or 'sampler_weights/{name}'.
    """
    model_id: str = Field(default="default", description="Model identifier")
    checkpoint_id: str = Field(..., description="Checkpoint ID to delete (e.g., 'weights/000' or 'sampler_weights/step-100')")


class DeleteCheckpointResponse(BaseModel):
    """API response for deleting a checkpoint."""
    success: bool = Field(..., description="Whether the deletion was successful")
    deleted_path: Optional[str] = Field(default=None, description="The xorl:// path that was deleted")
    error: Optional[str] = Field(default=None, description="Error message if deletion failed")


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
# Dedicated Endpoint Operations
# ============================================================================

class UpdateDedicatedEndpointRequest(BaseModel):
    """
    API request for saving checkpoint and updating a Together dedicated endpoint.

    This endpoint handles the full workflow:
    1. Save LoRA checkpoint locally
    2. Upload to HuggingFace
    3. Upload to Together via models.upload API
    4. Wait for model to be ready
    5. Probe endpoint to verify adapter is loaded
    """
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    name: str = Field(..., description="Checkpoint name (e.g., 'step-000001')")
    endpoint_id: str = Field(..., description="Together dedicated endpoint ID")
    together_api_key: str = Field(..., description="Together API key")
    hf_token: str = Field(..., description="HuggingFace token for uploading")
    hf_org: str = Field(default="xorl-org", description="HuggingFace organization")
    together_model_id: Optional[str] = Field(
        default=None,
        description="Optional Together model ID (auto-generated with timestamp if not provided)"
    )


class UpdateDedicatedEndpointResponse(BaseModel):
    """API response for updating dedicated endpoint."""
    success: bool = Field(..., description="Whether the update was successful")
    checkpoint_path: str = Field(..., description="Local checkpoint path")
    hf_path: str = Field(..., description="HuggingFace repository path")
    together_model_id: str = Field(
        ...,
        description="Together model ID (use this for sampling from the dedicated endpoint)"
    )
    upload_time: float = Field(..., description="Time to upload to HuggingFace (seconds)")
    together_upload_time: float = Field(..., description="Time for Together API upload (seconds)")
    probe_time: float = Field(..., description="Time to probe endpoint (seconds)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# Serverless Inference Operations
# ============================================================================

class UpdateServerlessWeightsRequest(BaseModel):
    """API request for updating weights on Together AI serverless inference."""
    model_id: str = Field(default="default", description="Model identifier (must be created via /api/v1/create_model first)")
    name: str = Field(..., description="Checkpoint name (e.g., 'step-001')")
    together_api_key: str = Field(..., description="Together AI API key")
    hf_token: str = Field(..., description="HuggingFace token with write access")
    together_base_model: str = Field(
        ...,
        description="Together AI base model name (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct-Reference')"
    )
    hf_repo_prefix: str = Field(
        default="xorl-adapter",
        description="Prefix for HuggingFace repository names"
    )
    private_repo: bool = Field(
        default=True,
        description="Whether to create private HuggingFace repository"
    )
    wait_for_completion: bool = Field(
        default=True,
        description="Whether to wait for Together AI upload to complete"
    )


class UpdateServerlessWeightsResponse(BaseModel):
    """API response for updating weights on Together AI."""
    together_model_id: str = Field(..., description="Together AI model name for inference")
    hf_repo_url: str = Field(..., description="HuggingFace repository URL")
    checkpoint_path: str = Field(..., description="Local checkpoint path")
    status: str = Field(..., description="Upload status ('Complete', 'submitted', etc.)")


# ============================================================================
# Inference Endpoint Management
# ============================================================================

class InferenceEndpointServerInfo(BaseModel):
    """Server info from SGLang inference endpoint (/server_info)."""
    model_path: Optional[str] = Field(default=None, description="Model path loaded on the server")
    served_model_name: Optional[str] = Field(default=None, description="Served model name")
    tp_size: Optional[int] = Field(default=None, description="Tensor parallelism size")
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
    worker_port: Optional[int] = Field(default=None, description="Port number of the inference worker (if None, defaults to port - 1 for backwards compatibility)")
    world_size: int = Field(default=1, description="Number of workers at this endpoint")
    # Auto-sync configuration
    sync_weights: bool = Field(
        default=False,
        description="Whether to automatically sync weights to this endpoint after adding"
    )
    master_address: Optional[str] = Field(
        default=None,
        description="Master address for NCCL rendezvous (auto-detected if not provided)"
    )
    master_port: int = Field(
        default=29600,
        description="Master port for NCCL rendezvous"
    )
    group_name: str = Field(
        default="weight_sync_group",
        description="Name of the NCCL process group"
    )
    buffer_size_mb: int = Field(
        default=1024,
        description="Size of each transfer bucket in MB (to avoid OOM)"
    )


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
    """API request for synchronizing weights to inference endpoints via NCCL."""
    master_address: str = Field(
        default="localhost",
        description="Master address for NCCL rendezvous (training server address)"
    )
    master_port: int = Field(
        default=29600,
        description="Master port for NCCL rendezvous"
    )
    group_name: str = Field(
        default="weight_sync_group",
        description="Name of the NCCL process group"
    )
    buffer_size_mb: int = Field(
        default=1024,
        description="Size of each transfer bucket in MB (to avoid OOM)"
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
        default_factory=list,
        description="Sync results for each endpoint"
    )


# ============================================================================
# Sampling Session Management
# ============================================================================

class CreateSamplingSessionRequest(BaseModel):
    """API request for creating a sampling session.

    This loads the specified LoRA adapter on all inference workers.
    The model_path must start with 'sampler_weights/' or 'xorl://'.
    """
    model_path: str = Field(
        ...,
        description="Path to the LoRA adapter (e.g., 'xorl://default/sampler_weights/step-100' or 'sampler_weights/step-100')"
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
    last_sampler_checkpoint: Optional[CheckpointInfo] = Field(default=None, description="The most recent sampler checkpoint")
    user_metadata: Optional[Dict[str, str]] = Field(default=None, description="Optional metadata about this training run")


class TrainingRunsResponse(BaseModel):
    """Response from list_training_runs operation."""
    training_runs: List[TrainingRun] = Field(..., description="List of training runs")
    cursor: Cursor = Field(..., description="Pagination cursor information")
