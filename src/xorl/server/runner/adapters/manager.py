"""
LoRA Adapter Manager - Manages multiple LoRA adapters for parallel training runs.

Each model_id has exactly one active adapter. Multiple model_ids can coexist,
enabling different training runs to interleave on the same base model and GPUs.

Design (Revised - Per-Adapter Parameters + Optimizer):
- Base model stays loaded on GPUs (frozen weights)
- Each adapter has its OWN nn.Parameter objects (separate .grad slots)
- Each adapter has its OWN optimizer instance
- Model params are "scratch space" - load weights before forward, capture grads after backward
- No gradient collision because each adapter's gradients live in its own Parameters
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file


try:
    from torch.distributed._tensor import DTensor
    from torch.distributed._tensor.placement_types import Shard

    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False


logger = logging.getLogger(__name__)


@dataclass
class AdapterState:
    """Complete isolated state for one training run.

    Key insight: Each adapter owns its own nn.Parameter objects, which have
    their own .grad slots. This prevents gradient collision when multiple
    adapters' forward_backward calls interleave.
    """

    model_id: str
    lora_params: Dict[str, nn.Parameter]  # Actual Parameters with own .grad
    optimizer: torch.optim.Optimizer  # Per-adapter optimizer
    global_step: int = 0
    global_forward_backward_step: int = 0
    lr: float = 1e-5
    last_access_time: float = field(default_factory=time.time)  # For LRU eviction


class LoRAAdapterManager:
    """
    Manages multiple LoRA adapters - one per model_id.

    Design: Each model_id has its own nn.Parameter objects and optimizer.
    The model's LoRA params are used as "scratch space" for forward/backward.

    Flow:
    1. prepare_forward(model_id): Copy adapter weights into model
    2. Forward + backward (gradients go to model's params)
    3. capture_gradients(model_id): Copy model's grads to adapter's params
    4. optim_step(model_id): Adapter's optimizer steps on adapter's params
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_adapters: int = 10,
        checkpoint_dir: Optional[str] = None,
        auto_save_on_eviction: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter manager.

        Args:
            model: The model with LoRA layers injected
            device: Device to create adapter parameters on
            max_adapters: Maximum number of adapters to keep in memory (LRU eviction)
            checkpoint_dir: Directory for saving adapter checkpoints (default: outputs/adapters)
            auto_save_on_eviction: If True, save adapter state before LRU eviction
            lora_config: LoRA configuration dict (for saving adapter_config.json)
        """
        self.model = model
        self.device = device
        self.max_adapters = max_adapters
        self.checkpoint_dir = checkpoint_dir or "outputs/adapters"
        self.auto_save_on_eviction = auto_save_on_eviction
        self.lora_config = lora_config or {}
        self.adapters: Dict[str, AdapterState] = {}
        self.current_adapter_id: Optional[str] = None

        # Cache the list of LoRA parameter names for efficient lookups
        self._lora_param_names: List[str] = []
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self._lora_param_names.append(name)

        logger.info(
            f"LoRAAdapterManager initialized with {len(self._lora_param_names)} LoRA parameters, "
            f"max_adapters={max_adapters}, auto_save_on_eviction={auto_save_on_eviction}"
        )

    def _maybe_evict(self) -> Optional[str]:
        """
        Evict the least recently used adapter if at capacity.

        If auto_save_on_eviction is enabled, saves the adapter state before evicting.

        Returns:
            The model_id of the evicted adapter, or None if no eviction was needed.
        """
        if len(self.adapters) >= self.max_adapters:
            if not self.adapters:
                return None
            # Find LRU adapter - all adapters can be evicted
            lru_id = min(self.adapters.keys(), key=lambda k: self.adapters[k].last_access_time)
            logger.info(f"Evicting LRU adapter: {lru_id} (capacity {len(self.adapters)}/{self.max_adapters})")

            # Auto-save before eviction if enabled
            if self.auto_save_on_eviction:
                try:
                    eviction_path = os.path.join(self.checkpoint_dir, "evicted", lru_id)
                    self.save_adapter_state(lru_id, eviction_path)
                    logger.info(f"Auto-saved adapter {lru_id} before eviction to {eviction_path}")
                except Exception as e:
                    logger.warning(f"Failed to auto-save adapter {lru_id} before eviction: {e}")

            self.remove_adapter(lru_id)
            return lru_id
        return None

    def register_adapter(
        self,
        model_id: str,
        lr: float,
        initialize_fresh: bool = True,
    ) -> None:
        """
        Register a new LoRA adapter for a model_id.

        Creates new nn.Parameter objects and a new optimizer for this adapter.
        If at capacity, evicts the least recently used adapter first.

        Args:
            model_id: Unique identifier for this training run
            lr: Learning rate for this adapter's optimizer
            initialize_fresh: If True, initialize with fresh random weights.
                            If False, use the current model's LoRA weights.
        """
        # Evict LRU adapter if at capacity and this is a new adapter
        if model_id not in self.adapters:
            self._maybe_evict()
        else:
            logger.info(f"Replacing existing adapter for model_id={model_id}")

        # Create Parameter objects for this adapter
        # IMPORTANT: Must create regular tensors, not DTensors, because the adapter's
        # optimizer needs to work on regular tensors (DTensors cause issues with fused ops)
        lora_params: Dict[str, nn.Parameter] = {}
        for name, param in self.model.named_parameters():
            if name in self._lora_param_names:
                # Get shape and dtype from the parameter
                # IMPORTANT: For DTensors, use .shape (global shape) and .dtype directly
                # DO NOT call full_tensor() here as it's a collective operation that
                # requires all ranks to participate, which can cause deadlock when
                # called from load_adapter_state (only rank 0 does the load)
                if _HAS_DTENSOR and isinstance(param, DTensor):
                    # DTensor.shape gives the global (unsharded) shape
                    param_shape = param.shape
                    param_dtype = param.dtype
                else:
                    param_shape = param.data.shape
                    param_dtype = param.data.dtype

                if initialize_fresh:
                    # Fresh initialization - create regular tensor on device
                    if "lora_A" in name:
                        new_tensor = torch.empty(
                            param_shape,
                            dtype=param_dtype,
                            device=self.device,
                        )
                        nn.init.kaiming_uniform_(new_tensor, a=math.sqrt(5))
                    else:  # lora_B
                        new_tensor = torch.zeros(
                            param_shape,
                            dtype=param_dtype,
                            device=self.device,
                        )
                else:
                    # Copy current model weights as regular tensor
                    # NOTE: This path requires full_tensor() for DTensors, so it should
                    # only be called when all ranks are participating (e.g., from
                    # register_adapter endpoint, not from load_adapter_state)
                    if _HAS_DTENSOR and isinstance(param, DTensor):
                        param_data = param.full_tensor()
                    else:
                        param_data = param.data
                    new_tensor = param_data.detach().clone().to(self.device)

                # Create as nn.Parameter so it has its own .grad slot
                lora_params[name] = nn.Parameter(new_tensor, requires_grad=True)

        # Create optimizer for this adapter's params
        optimizer = torch.optim.AdamW(
            list(lora_params.values()),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

        self.adapters[model_id] = AdapterState(
            model_id=model_id,
            lora_params=lora_params,
            optimizer=optimizer,
            global_step=0,
            global_forward_backward_step=0,
            lr=lr,
        )

        logger.info(
            f"Registered adapter for model_id={model_id} "
            f"(lr={lr}, fresh_weights={initialize_fresh}, num_params={len(lora_params)})"
        )

    def prepare_forward(self, model_id: str) -> None:
        """
        Load adapter weights into model before forward pass.

        This must be called before forward() to ensure the model uses
        the correct adapter's weights.

        For FSDP2/DTensor: The model's params are sharded DTensors, but the
        adapter's params are full regular tensors. We need to copy only the
        local shard from the adapter to the model.

        Args:
            model_id: The adapter to prepare for

        Raises:
            KeyError: If the adapter is not registered. Use register_adapter() first.
        """
        if model_id not in self.adapters:
            raise KeyError(
                f"Adapter for model_id={model_id} not registered. "
                "Call register_adapter() first or ensure the session is valid."
            )

        state = self.adapters[model_id]
        # Update last access time for LRU tracking
        state.last_access_time = time.time()

        # Copy adapter weights into model's params (for forward to use)
        # Use no_grad to avoid autograd issues with DTensor views
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state.lora_params:
                    adapter_data = state.lora_params[name].data

                    if _HAS_DTENSOR and isinstance(param, DTensor):
                        # For DTensor: copy to the local tensor (the shard)
                        # The adapter has full weights, but model param is sharded.
                        # We need to extract the correct shard from adapter weights.
                        local_tensor = param.to_local()
                        placements = param.placements
                        device_mesh = param.device_mesh

                        sliced_data = adapter_data
                        skip_copy = False
                        for dim, placement in enumerate(placements):
                            if isinstance(placement, Shard):
                                shard_dim = placement.dim
                                mesh_dim_size = device_mesh.size(dim)
                                local_rank = device_mesh.get_local_rank(mesh_dim=dim)
                                total_size = sliced_data.shape[shard_dim]
                                shard_size = (total_size + mesh_dim_size - 1) // mesh_dim_size
                                start = local_rank * shard_size
                                end = min(start + shard_size, total_size)
                                length = max(end - start, 0)
                                if start >= total_size or length == 0:
                                    skip_copy = True
                                    break
                                sliced_data = sliced_data.narrow(shard_dim, start, length)

                        if not skip_copy and local_tensor.numel() > 0:
                            local_tensor.copy_(sliced_data)
                    else:
                        # Regular tensor: direct copy
                        param.data.copy_(adapter_data)

        self.current_adapter_id = model_id

    def capture_gradients(self, model_id: str) -> None:
        """
        Copy gradients from model params to adapter params after backward.

        This captures the gradients computed during backward() and stores
        them in the adapter's own Parameter objects (which have their own
        .grad slots). This prevents gradient collision when multiple adapters
        interleave.

        For FSDP2/DTensor: If gradients are DTensors (sharded), we call
        .full_tensor() to get the full unsharded gradient before copying.

        Args:
            model_id: The adapter to capture gradients for
        """
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]
        grad_count = 0

        for name, param in self.model.named_parameters():
            if name in state.lora_params:
                adapter_param = state.lora_params[name]
                if param.grad is not None:
                    # Handle DTensor (FSDP2 sharded gradients)
                    grad = param.grad
                    if _HAS_DTENSOR and isinstance(grad, DTensor):
                        grad = grad.full_tensor()

                    # Copy gradient to adapter's param (accumulate for grad accumulation)
                    if adapter_param.grad is None:
                        adapter_param.grad = grad.clone()
                    else:
                        adapter_param.grad.add_(grad)
                    grad_count += 1
                    # Clear model's grad to prevent accumulation across adapters
                    param.grad = None

    def optim_step(
        self,
        model_id: str,
        lr: float,
        gradient_clip: Optional[float] = None,
        accumulated_valid_tokens: int = 0,
    ) -> float:
        """
        Run optimizer step on adapter's own parameters.

        This uses the adapter's own optimizer on the adapter's own Parameters,
        which have their own gradients from capture_gradients().

        Args:
            model_id: The adapter to step
            lr: Learning rate to use
            gradient_clip: Optional gradient clipping value
            accumulated_valid_tokens: Total valid tokens accumulated across
                forward_backward calls. If > 0, gradients are scaled by
                1/accumulated_valid_tokens (deferred normalization).

        Returns:
            The gradient norm before clipping
        """
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]

        # Update learning rate
        state.lr = lr
        for pg in state.optimizer.param_groups:
            pg["lr"] = lr

        # Deferred gradient normalization: scale raw gradients by 1/accumulated_valid_tokens
        if accumulated_valid_tokens > 0:
            scale = 1.0 / accumulated_valid_tokens
            for p in state.lora_params.values():
                if p.grad is not None:
                    p.grad.data = p.grad.data.float() * scale

        # Always use clip_grad_norm_ for correct grad norm computation
        # Using a large clip value (10000.0) effectively means no clipping
        clip_value = gradient_clip if (gradient_clip is not None and gradient_clip > 0) else 10000.0
        grad_norm = torch.nn.utils.clip_grad_norm_(list(state.lora_params.values()), clip_value)
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()

        # Step the adapter's optimizer
        state.optimizer.step()
        state.optimizer.zero_grad()

        # Increment step counter
        state.global_step += 1

        return grad_norm

    def sync_weights_to_model(self, model_id: str) -> None:
        """
        Sync adapter weights to model (for save_lora_only, inference, etc).

        For FSDP2/DTensor: Same as prepare_forward, handles sharded params.

        Args:
            model_id: The adapter whose weights to sync
        """
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        # Just delegate to prepare_forward which handles DTensor properly
        self.prepare_forward(model_id)

    def get_adapter_state(self, model_id: str) -> AdapterState:
        """Get the adapter state for a model_id."""
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")
        return self.adapters[model_id]

    def get_current_adapter(self) -> Optional[AdapterState]:
        """Get the currently active adapter state."""
        if self.current_adapter_id is None:
            return None
        return self.adapters.get(self.current_adapter_id)

    def increment_forward_backward_step(self, model_id: str) -> int:
        """Increment and return the forward_backward step counter for an adapter."""
        state = self.adapters[model_id]
        state.global_forward_backward_step += 1
        return state.global_forward_backward_step

    def get_global_step(self, model_id: str) -> int:
        """Get the global step counter for an adapter."""
        return self.adapters[model_id].global_step

    def get_forward_backward_step(self, model_id: str) -> int:
        """Get the forward_backward step counter for an adapter."""
        return self.adapters[model_id].global_forward_backward_step

    def get_lr(self, model_id: str) -> float:
        """Get the learning rate for an adapter."""
        return self.adapters[model_id].lr

    def set_lr(self, model_id: str, lr: float) -> None:
        """Set the learning rate for an adapter."""
        state = self.adapters[model_id]
        state.lr = lr
        for param_group in state.optimizer.param_groups:
            param_group["lr"] = lr

    def has_adapter(self, model_id: str) -> bool:
        """Check if an adapter is registered for a model_id."""
        return model_id in self.adapters

    def list_adapters(self) -> List[str]:
        """List all registered model_ids."""
        return list(self.adapters.keys())

    def remove_adapter(self, model_id: str) -> None:
        """Remove an adapter for a model_id."""
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        if self.current_adapter_id == model_id:
            self.current_adapter_id = None

        del self.adapters[model_id]
        logger.info(f"Removed adapter for model_id={model_id}")

    # Legacy compatibility methods (for gradual migration)
    def switch_adapter(self, model_id: str, auto_register: bool = False) -> bool:
        """Legacy method - now just calls prepare_forward.

        Args:
            model_id: The adapter to switch to
            auto_register: If True, auto-register adapter if not found (default: False)

        Returns:
            True if successful

        Raises:
            KeyError: If adapter not registered and auto_register is False
        """
        if model_id not in self.adapters:
            if auto_register:
                logger.warning(f"Auto-registering adapter for model_id={model_id} (deprecated)")
                self.register_adapter(model_id, lr=1e-5, initialize_fresh=True)
            else:
                raise KeyError(f"Adapter for model_id={model_id} not registered")

        self.prepare_forward(model_id)
        return True

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Return memory usage per adapter in bytes.

        Includes both parameters and optimizer state (AdamW stores ~2x params).

        Returns:
            Dict mapping model_id to memory usage in bytes.
        """
        usage = {}
        for model_id, state in self.adapters.items():
            # Calculate parameter memory
            param_bytes = sum(p.numel() * p.element_size() for p in state.lora_params.values())

            # Calculate optimizer state memory (AdamW stores exp_avg and exp_avg_sq)
            optim_bytes = 0
            for param_state in state.optimizer.state.values():
                for v in param_state.values():
                    if isinstance(v, torch.Tensor):
                        optim_bytes += v.numel() * v.element_size()

            usage[model_id] = param_bytes + optim_bytes
        return usage

    def get_adapter_count(self) -> int:
        """Return the number of currently loaded adapters."""
        return len(self.adapters)

    def save_adapter_state(
        self,
        model_id: str,
        path: Optional[str] = None,
        save_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Save a specific adapter's state to disk.

        Saves LoRA weights in PEFT-compatible format (adapter_model.safetensors),
        plus optimizer state and metadata for full training resume.

        Args:
            model_id: The adapter to save
            path: Directory to save to (default: {checkpoint_dir}/{model_id})
            save_optimizer: Whether to save optimizer state

        Returns:
            Dict with path, model_id, step, and save_time
        """
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]

        # Use default path if not provided
        if path is None:
            path = os.path.join(self.checkpoint_dir, model_id)

        # Create directory
        os.makedirs(path, exist_ok=True)

        start_time = time.time()

        # 1. Save LoRA weights in safetensors format (PEFT-compatible)
        # Convert parameter names to PEFT format: base_model.model.{name}
        weights_dict = {}
        for name, param in state.lora_params.items():
            peft_name = f"base_model.model.{name}"
            weights_dict[peft_name] = param.data.cpu().to(torch.bfloat16)

        weights_path = os.path.join(path, "adapter_model.safetensors")
        safetensors_save_file(weights_dict, weights_path)

        # 2. Save optimizer state
        if save_optimizer:
            optimizer_path = os.path.join(path, "optimizer.pt")
            torch.save(state.optimizer.state_dict(), optimizer_path)

        # 3. Save metadata
        metadata = {
            "model_id": model_id,
            "global_step": state.global_step,
            "global_forward_backward_step": state.global_forward_backward_step,
            "lr": state.lr,
            "timestamp": time.time(),
            "save_optimizer": save_optimizer,
        }
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Save adapter config (PEFT-compatible)
        # Extract LoRA config from parameter shapes
        lora_r = None
        target_modules = set()
        for name, param in state.lora_params.items():
            if "lora_A" in name:
                lora_r = param.shape[0]  # lora_A is [r, in_features]
                # Extract module name (e.g., "model.layers.0.self_attn.q_proj" from full name)
                parts = name.replace(".lora_A.weight", "").split(".")
                if len(parts) >= 1:
                    target_modules.add(parts[-1])  # e.g., "q_proj"

        adapter_config = {
            "r": lora_r,
            "lora_alpha": lora_r,  # Assume alpha = r (common default)
            "target_modules": list(target_modules),
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "moe_hybrid_shared_lora": self.lora_config.get("moe_hybrid_shared_lora", False),
        }
        config_path = os.path.join(path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)

        save_time = time.time() - start_time
        logger.info(
            f"Saved adapter state for model_id={model_id} to {path} "
            f"(step={state.global_step}, save_optimizer={save_optimizer}, time={save_time:.2f}s)"
        )

        return {
            "path": path,
            "model_id": model_id,
            "step": state.global_step,
            "save_time": save_time,
        }

    def load_adapter_state(
        self,
        model_id: str,
        path: str,
        load_optimizer: bool = True,
        lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Load adapter state from checkpoint.

        Can load into a new model_id (different from the one saved).
        Creates/registers the adapter if it doesn't exist.

        Args:
            model_id: Target model_id to load into (can differ from saved)
            path: Directory to load from
            load_optimizer: Whether to load optimizer state
            lr: Learning rate override (uses saved lr if None)

        Returns:
            Dict with path, model_id, step, and load_time
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

        start_time = time.time()

        # 1. Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Determine learning rate
        effective_lr = lr if lr is not None else metadata.get("lr", 1e-5)

        # 2. Register adapter if not exists (this will evict if needed)
        if model_id not in self.adapters:
            self.register_adapter(model_id, lr=effective_lr, initialize_fresh=True)

        state = self.adapters[model_id]

        # 3. Load LoRA weights
        weights_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(weights_path):
            loaded_weights = safetensors_load_file(weights_path)

            # Convert from PEFT format back to internal format
            for peft_name, weight in loaded_weights.items():
                # Remove "base_model.model." prefix
                internal_name = peft_name.replace("base_model.model.", "")
                if internal_name in state.lora_params:
                    state.lora_params[internal_name].data.copy_(weight.to(self.device))
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # 4. Load optimizer state
        optimizer_path = os.path.join(path, "optimizer.pt")
        if load_optimizer and os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=self.device, weights_only=True)
            state.optimizer.load_state_dict(optimizer_state)
            logger.debug(f"Loaded optimizer state from {optimizer_path}")

        # 5. Restore metadata
        state.global_step = metadata.get("global_step", 0)
        state.global_forward_backward_step = metadata.get("global_forward_backward_step", 0)
        if lr is not None:
            state.lr = lr
            for pg in state.optimizer.param_groups:
                pg["lr"] = lr
        elif "lr" in metadata:
            state.lr = metadata["lr"]

        # Update last access time
        state.last_access_time = time.time()

        load_time = time.time() - start_time
        logger.info(
            f"Loaded adapter state for model_id={model_id} from {path} "
            f"(step={state.global_step}, load_optimizer={load_optimizer}, time={load_time:.2f}s)"
        )

        return {
            "path": path,
            "model_id": model_id,
            "step": state.global_step,
            "load_time": load_time,
        }
