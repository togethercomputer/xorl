"""
API Request/Response Types for REST API.

Pydantic type definitions for FastAPI endpoints in the unified API server.
"""

from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Type Aliases
# ============================================================================

InputType = Union[List[int], List[float], List[str]]


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


class DatumInput(BaseModel):
    """Input containing list of data examples and loss function."""
    data: List[Datum] = Field(
        ...,
        description="List of training/inference examples"
    )
    loss_fn: str = Field(default="causallm_loss", description="Loss function type")


class LossFnOutput(BaseModel):
    """Single loss function output."""
    loss: float = Field(..., description="Loss value")


class AdamParams(BaseModel):
    """AdamW optimizer parameters."""
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    beta1: float = Field(default=0.9, description="First moment coefficient")
    beta2: float = Field(default=0.95, description="Second moment coefficient")
    eps: float = Field(default=1e-12, description="Numerical stability term")


# ============================================================================
# Inference Operations
# ============================================================================

class ForwardRequest(BaseModel):
    """API request for forward operation."""
    model_id: str = Field(default="default", description="Model identifier")
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
    model_id: str = Field(default="default", description="Model identifier")
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
    model_id: str = Field(default="default", description="Model identifier")
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
    model_id: str = Field(default="default", description="Model identifier")
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
    """API request for saving weights."""
    model_id: str = Field(default="default", description="Model identifier")
    path: Optional[str] = Field(default=None, description="Checkpoint name/path (auto-generated if not specified)")
    save_optimizer: bool = Field(default=True, description="Whether to save optimizer state")


class SaveWeightsResponse(BaseModel):
    """API response for saving weights."""
    path: str = Field(..., description="Checkpoint path (local path, TODO: s3/r2 path)")


class LoadWeightsRequest(BaseModel):
    """API request for loading weights."""
    model_id: str = Field(default="default", description="Model identifier")
    path: str = Field(..., description="Checkpoint path (local path, TODO: s3/r2 path)")
    load_optimizer: bool = Field(default=True, description="Whether to load optimizer state")


class LoadWeightsResponse(BaseModel):
    """API response for loading weights."""
    success: bool = Field(..., description="Whether the load was successful")


class CreateModelRequest(BaseModel):
    """API request for creating a new model."""
    model_id: str = Field(..., description="Model identifier")
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
    model_id: str = Field(default="default", description="Model identifier")
    name: str = Field(..., description="Checkpoint name (e.g., 'step-100')")


class SaveWeightsForSamplerResponse(BaseModel):
    """API response for saving weights for sampler."""
    path: str = Field(..., description="Checkpoint path (local path)")
    model_path: str = Field(..., description="Model path for routing (e.g., 'xorl://model-123/step-100')")


class SaveLoRAOnlyRequest(BaseModel):
    """API request for saving only LoRA adapter weights."""
    model_id: str = Field(default="default", description="Model identifier")
    lora_path: Optional[str] = Field(default=None, description="LoRA save path (auto-generated if not specified)")


class SaveLoRAOnlyResponse(BaseModel):
    """API response for saving LoRA weights."""
    lora_path: str = Field(..., description="LoRA adapter path")


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
    model_id: str = Field(default="default", description="Model identifier")
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
    model_id: str = Field(default="default", description="Model identifier")
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
