"""
Abstract Backend interface for compute operations.

A clean interface (~15 methods) that the Executor calls directly,
with implementations handling the transport layer.

Implementations:
- RemoteBackend: ZMQ DEALER-ROUTER communication with RunnerDispatcher
- DummyBackend: In-process mock for testing (no ZMQ, no threads)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Backend(ABC):
    """Abstract compute backend for training operations."""

    # ========================================================================
    # Model Pass Operations
    # ========================================================================

    @abstractmethod
    async def forward_backward(
        self,
        batches: List[Dict[str, Any]],
        loss_fn: str = "causallm_loss",
        loss_fn_params: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        routed_experts: Optional[List[Any]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run forward + backward pass. Returns {total_loss, global_valid_tokens, ...}."""

    @abstractmethod
    async def forward(
        self,
        batches: List[Dict[str, Any]],
        loss_fn: str = "causallm_loss",
        loss_fn_params: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward-only pass (no gradients). Returns {total_loss, global_valid_tokens, ...}."""

    # ========================================================================
    # Optimizer
    # ========================================================================

    @abstractmethod
    async def optim_step(
        self,
        lr: float,
        gradient_clip: Optional[float] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        eps: Optional[float] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Optimizer step. Returns {grad_norm, step, ...}."""

    # ========================================================================
    # Checkpoint Operations
    # ========================================================================

    @abstractmethod
    async def save_state(
        self,
        checkpoint_path: Optional[str] = None,
        save_optimizer: bool = True,
        use_timestamp: bool = False,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save checkpoint. Returns {checkpoint_path, success, ...}."""

    @abstractmethod
    async def load_state(
        self,
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint. Returns {checkpoint_path, success, ...}."""

    @abstractmethod
    async def save_lora_only(
        self,
        lora_path: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save LoRA-only checkpoint (PEFT format). Returns {lora_path, success, ...}."""

    @abstractmethod
    async def save_full_weights(
        self,
        output_path: Optional[str] = None,
        dtype: str = "bfloat16",
        base_model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save full weights as safetensors. Returns {output_path, success, num_shards, ...}."""

    # ========================================================================
    # Sleep / Wake
    # ========================================================================

    @abstractmethod
    async def sleep(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Offload model to CPU. Returns {status, offload_time, ...}."""

    @abstractmethod
    async def wake_up(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Load model back to GPU. Returns {status, load_time, ...}."""

    # ========================================================================
    # Weight Sync
    # ========================================================================

    @abstractmethod
    async def sync_inference_weights(
        self,
        endpoints: List[Dict[str, Any]],
        master_address: str = "localhost",
        master_port: int = 29600,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Sync weights to inference endpoints via NCCL. Returns {success, transfer_time, ...}."""

    # ========================================================================
    # Adapter Operations
    # ========================================================================

    @abstractmethod
    async def register_adapter(
        self,
        model_id: str = "default",
        lr: float = 1e-5,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new LoRA adapter. Returns {result, ...}."""

    @abstractmethod
    async def save_adapter_state(
        self,
        model_id: str = "default",
        path: Optional[str] = None,
        save_optimizer: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save adapter state. Returns {result, ...}."""

    @abstractmethod
    async def load_adapter_state(
        self,
        model_id: str = "default",
        path: Optional[str] = None,
        load_optimizer: bool = True,
        lr: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load adapter state. Returns {result, ...}."""

    @abstractmethod
    async def get_adapter_info(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Get adapter info. Returns adapter metadata dict."""

    # ========================================================================
    # Session Management
    # ========================================================================

    @abstractmethod
    async def kill_session(
        self,
        model_id: str = "default",
        save_checkpoint: bool = True,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Kill training session. Returns {success, message, checkpoint_path, ...}."""

    @abstractmethod
    async def health_check(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Health check. Returns {status, ...}."""

    # ========================================================================
    # Lifecycle
    # ========================================================================

    @abstractmethod
    async def start(self) -> None:
        """Start the backend (connect, handshake, etc.)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend and release resources."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the backend is ready for operations."""
