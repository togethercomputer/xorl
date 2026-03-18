"""
Typed Operation Payloads for the Engine Protocol.

This module defines typed dataclasses for all operation payloads, replacing
the untyped Dict[str, Any] that was previously used in OrchestratorRequest.data
and RunnerDispatchCommand.data.

These types are shared by both protocol layers:
- API→Engine (msgpack): serialized via payload_to_dict(), deserialized via payload_from_dict()
- Executor→Worker (pickle): serialized directly as typed objects
- Worker broadcast (Gloo): pickle internally, typed objects work identically to dicts
"""

import os
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional, Union

# ============================================================================
# Timeout Constants (shared by engine/executor and backend/remote)
# ============================================================================

# Default: 30 minutes (1800 seconds) for both save and load operations
SAVE_STATE_TIMEOUT = float(os.environ.get("XORL_SAVE_STATE_TIMEOUT", 1800.0))
LOAD_STATE_TIMEOUT = float(os.environ.get("XORL_LOAD_STATE_TIMEOUT", 1800.0))


# ============================================================================
# Operation Payload Dataclasses
# ============================================================================


@dataclass
class ModelPassData:
    """Payload for forward / forward_backward operations."""

    data: List[Dict[str, Any]] = field(default_factory=list)  # datum list (API→Engine)
    batches: Optional[List[Any]] = None  # packed batches (set by Executor)
    loss_fn: str = "causallm_loss"
    loss_fn_params: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    routed_experts: Optional[List[Any]] = None
    routed_expert_logits: Optional[List[Any]] = None


@dataclass
class OptimStepData:
    """Payload for optim_step operations."""

    lr: float = 1e-4
    gradient_clip: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    eps: Optional[float] = None
    model_id: Optional[str] = None


@dataclass
class SaveStateData:
    """Payload for save_state / save_weights / save_weights_for_sampler operations."""

    checkpoint_path: Optional[str] = None
    save_optimizer: bool = True
    use_timestamp: bool = False
    model_id: Optional[str] = None


@dataclass
class SaveLoraOnlyData:
    """Payload for save_lora_only operations."""

    lora_path: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class LoadStateData:
    """Payload for load_state / load_weights operations."""

    checkpoint_path: Optional[str] = None
    load_optimizer: bool = True
    model_id: Optional[str] = None


@dataclass
class SaveFullWeightsData:
    """Payload for save_full_weights operations."""

    output_path: Optional[str] = None
    dtype: str = "bfloat16"
    base_model_path: Optional[str] = None
    model_id: Optional[str] = None


@dataclass
class SyncWeightsData:
    """Payload for sync_inference_weights operations."""

    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    master_address: str = "localhost"
    master_port: int = 29600
    group_name: str = "weight_sync_group"
    buffer_size_mb: int = 1024
    sync_method: str = "nccl_broadcast"
    flush_cache: bool = True
    pause_mode: str = "retract"
    weight_version: Optional[str] = None
    quantization: Optional[Dict[str, Any]] = None


@dataclass
class RegisterAdapterData:
    """Payload for register_adapter operations."""

    model_id: str = "default"
    lr: float = 1e-5


@dataclass
class AdapterStateData:
    """Payload for save_adapter_state / load_adapter_state operations."""

    model_id: str = "default"
    path: Optional[str] = None
    save_optimizer: bool = True
    load_optimizer: bool = True
    lr: Optional[float] = None


@dataclass
class KillSessionData:
    """Payload for kill_session operations."""

    model_id: str = "default"
    save_checkpoint: bool = True
    reset_weights: bool = False


@dataclass
class AbortData:
    """Payload for abort operations."""

    target_request_id: str = ""


@dataclass
class EmptyData:
    """Payload for operations with no data (health_check, sleep, wake_up, get_adapter_info, shutdown)."""

    pass


# ============================================================================
# Type Union and Registry
# ============================================================================

OperationPayload = Union[
    ModelPassData,
    OptimStepData,
    SaveStateData,
    SaveLoraOnlyData,
    LoadStateData,
    SaveFullWeightsData,
    SyncWeightsData,
    RegisterAdapterData,
    AdapterStateData,
    KillSessionData,
    AbortData,
    EmptyData,
]

# Maps operation string → payload dataclass type
_PAYLOAD_TYPE_MAP: Dict[str, type] = {
    "forward": ModelPassData,
    "forward_backward": ModelPassData,
    "optim_step": OptimStepData,
    "save_state": SaveStateData,
    "save_lora_only": SaveLoraOnlyData,
    "load_state": LoadStateData,
    "load_weights": LoadStateData,
    "save_full_weights": SaveFullWeightsData,
    "sync_inference_weights": SyncWeightsData,
    "register_adapter": RegisterAdapterData,
    "save_adapter_state": AdapterStateData,
    "load_adapter_state": AdapterStateData,
    "kill_session": KillSessionData,
    "health_check": EmptyData,
    "sleep": EmptyData,
    "wake_up": EmptyData,
    "get_adapter_info": EmptyData,
    "shutdown": EmptyData,
    "save_weights_for_sampler": SaveStateData,
}


# ============================================================================
# Serialization Helpers
# ============================================================================


def payload_to_dict(payload: OperationPayload) -> Dict[str, Any]:
    """Serialize payload to dict for msgpack. Filters None values."""
    return {k: v for k, v in asdict(payload).items() if v is not None}


def payload_from_dict(operation: str, data: Dict[str, Any]) -> OperationPayload:
    """Deserialize payload from dict (msgpack). Tolerant of unknown keys."""
    cls = _PAYLOAD_TYPE_MAP.get(operation, EmptyData)
    known = {f.name for f in dc_fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)
