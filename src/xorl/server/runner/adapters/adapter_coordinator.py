"""
AdapterCoordinator - Multi-rank adapter lifecycle management.

Extracted from dispatcher.py to separate adapter coordination
concerns (broadcast, auto-load, register, save/load state, info, kill session)
from the core worker communication logic.

All methods that require multi-rank coordination for LoRA adapter management
live here. The RunnerDispatcher delegates to this class.
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file as safetensors_load_file

from xorl.lora.utils import convert_peft_lora_state_dict, get_lora_tensor_shard_specs
from xorl.server.protocol.operations import (
    AdapterStateData,
    KillSessionData,
    RegisterAdapterData,
    RegisterSessionData,
)
from xorl.server.session_spec import load_session_spec_from_checkpoint


if TYPE_CHECKING:
    from xorl.server.runner.model_runner import ModelRunner


logger = logging.getLogger(__name__)

_ADAPTER_STATE_LOAD_MODES = {"all_ranks", "rank0_broadcast"}


class AdapterCoordinator:
    """Coordinates multi-rank LoRA adapter operations.

    Handles broadcasting adapter state, auto-loading evicted adapters,
    registering new adapters, saving/loading adapter state, querying
    adapter info, and killing sessions.
    """

    def __init__(
        self,
        trainer: ModelRunner,
        rank: int,
        world_size: int,
        cpu_group: Optional[dist.ProcessGroup],
    ):
        self.trainer = trainer
        self.rank = rank
        self.world_size = world_size
        self.cpu_group = cpu_group

    def _validate_pipeline_parallel_broadcast_safe(self) -> None:
        """Reject pipeline-parallel topologies for broadcast-based adapter coordination."""
        pipeline_parallel_size = int(getattr(self.trainer, "train_config", {}).get("pipeline_parallel_size", 1))
        if pipeline_parallel_size > 1 and self.world_size > 1:
            raise RuntimeError(
                "pipeline_parallel_size > 1 is not supported with multi-adapter LoRA server training. "
                "Adapter coordination currently assumes identical local LoRA layouts on every rank."
            )

    # ========================================================================
    # Adapter Broadcast
    # ========================================================================

    def broadcast_adapter_state(self, model_id: str, default_lr: float) -> None:
        """
        Broadcast adapter weights and metadata from rank 0 to all other ranks.

        Args:
            model_id: The adapter/session ID to broadcast
            default_lr: Default learning rate if not available from adapter state
        """
        if self.world_size <= 1:
            return
        self._validate_pipeline_parallel_broadcast_safe()

        adapter_state = self.trainer.adapter_manager.get_adapter_state(model_id)

        # Broadcast each parameter
        for name, param in adapter_state.lora_params.items():
            dist.broadcast(param.data, src=0)

        # Broadcast metadata
        metadata = [None]
        if self.rank == 0:
            metadata = [
                {
                    "global_step": adapter_state.global_step,
                    "global_forward_backward_step": adapter_state.global_forward_backward_step,
                    "lr": adapter_state.lr,
                }
            ]
        dist.broadcast_object_list(metadata, src=0, group=self.cpu_group)

        # Update metadata on non-rank-0 workers
        if self.rank != 0 and metadata[0]:
            adapter_state.global_step = metadata[0].get("global_step", 0)
            adapter_state.global_forward_backward_step = metadata[0].get("global_forward_backward_step", 0)
            self.trainer.adapter_manager.set_lr(model_id, metadata[0].get("lr", default_lr))
        adapter_state.last_access_time = time.time()

        logger.debug(f"Rank {self.rank}: Broadcast adapter state for model_id={model_id}")

    @staticmethod
    def _optimizer_state_to_cpu(value: Any) -> Any:
        """Recursively move optimizer state dict tensors to CPU for object broadcast."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: AdapterCoordinator._optimizer_state_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AdapterCoordinator._optimizer_state_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(AdapterCoordinator._optimizer_state_to_cpu(v) for v in value)
        return value

    def broadcast_adapter_optimizer_state(self, model_id: str) -> None:
        """Broadcast adapter optimizer state from rank 0 to all other ranks."""
        if self.world_size <= 1:
            return
        self._validate_pipeline_parallel_broadcast_safe()

        adapter_state = self.trainer.adapter_manager.get_adapter_state(model_id)
        optimizer_state = [None]
        if self.rank == 0:
            optimizer_state[0] = self._optimizer_state_to_cpu(adapter_state.optimizer.state_dict())

        dist.broadcast_object_list(optimizer_state, src=0, group=self.cpu_group)

        if self.rank != 0 and optimizer_state[0] is not None:
            adapter_state.optimizer.load_state_dict(optimizer_state[0])

    def _has_ep_sharded_adapter_params(self, model_id: str) -> bool:
        """Return whether the resident adapter has LoRA tensors sharded by EP."""
        adapter_manager = self.trainer.adapter_manager
        model = getattr(adapter_manager, "model", getattr(self.trainer, "model", None))
        if adapter_manager is None or model is None or not adapter_manager.has_adapter(model_id):
            return False

        canonical_name = getattr(adapter_manager, "_canonical_lora_param_name", lambda name: name)
        state = adapter_manager.get_adapter_state(model_id)
        requested_names = {canonical_name(name) for name in state.lora_params}
        return bool(get_lora_tensor_shard_specs(model, names=requested_names))

    def _expected_adapter_param_maps(self, model_id: str) -> Tuple[Dict[str, str], Dict[str, torch.Size]]:
        """Build canonical-name maps for the live adapter tensors."""
        adapter_manager = self.trainer.adapter_manager
        state = adapter_manager.get_adapter_state(model_id)
        expected_param_map: Dict[str, str] = {}
        expected_shapes: Dict[str, torch.Size] = {}

        for actual_name, param in state.lora_params.items():
            canonical_name = adapter_manager._canonical_lora_param_name(actual_name)
            if canonical_name in expected_param_map and expected_param_map[canonical_name] != actual_name:
                raise ValueError(
                    f"Live adapter contains duplicate LoRA tensors after canonicalization. param={canonical_name!r}"
                )
            expected_param_map[canonical_name] = actual_name
            expected_shapes[canonical_name] = param.shape

        return expected_param_map, expected_shapes

    @staticmethod
    def _strip_optimizer_config(session_spec: Dict[str, Any]) -> Dict[str, Any]:
        stripped = deepcopy(session_spec)
        stripped.pop("optimizer_config", None)
        return stripped

    @staticmethod
    def _strip_optimizer_learning_rate(session_spec: Dict[str, Any]) -> Dict[str, Any]:
        stripped = deepcopy(session_spec)
        optimizer_config = stripped.get("optimizer_config")
        if isinstance(optimizer_config, dict):
            optimizer_config.pop("learning_rate", None)
        return stripped

    def _validate_broadcast_checkpoint_session_spec(
        self,
        model_id: str,
        checkpoint_session_spec: Optional[Dict[str, Any]],
        *,
        load_optimizer: bool,
        lr: Optional[float],
    ) -> None:
        """Validate rank-0 broadcast checkpoint metadata before applying tensors."""
        if not isinstance(checkpoint_session_spec, dict):
            raise ValueError("Rank-0 broadcast adapter payload did not include a valid checkpoint session spec")

        registered_session_spec = self.trainer.get_lora_session_spec(model_id)
        checkpoint_spec_for_compare = checkpoint_session_spec
        registered_spec_for_compare = registered_session_spec
        if lr is not None:
            checkpoint_spec_for_compare = self._strip_optimizer_learning_rate(checkpoint_spec_for_compare)
            registered_spec_for_compare = self._strip_optimizer_learning_rate(registered_spec_for_compare)

        if load_optimizer:
            specs_match = checkpoint_spec_for_compare == registered_spec_for_compare
            mismatch_context = "registered multi-adapter session"
        else:
            specs_match = self._strip_optimizer_config(checkpoint_spec_for_compare) == self._strip_optimizer_config(
                registered_spec_for_compare
            )
            mismatch_context = "registered multi-adapter session for weights-only restore"

        if not specs_match:
            raise ValueError(
                "Checkpoint session spec does not match the "
                f"{mismatch_context}. checkpoint={checkpoint_session_spec!r}, "
                f"current={registered_session_spec!r}"
            )

    def _rank0_load_adapter_checkpoint_payload(self, model_id: str, path: str, load_optimizer: bool) -> Dict[str, Any]:
        """Load adapter checkpoint tensors on rank 0 and broadcast them as a CPU payload."""
        payload = [None]

        if self.rank == 0:
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

                validate_checkpoint = getattr(self.trainer.adapter_manager, "_validate_checkpoint_adapter_config", None)
                if validate_checkpoint is not None:
                    validate_checkpoint(path)

                registered_session_spec = self.trainer.get_lora_session_spec(model_id)
                checkpoint_session_spec = load_session_spec_from_checkpoint(
                    path,
                    fallback_base_model=registered_session_spec.get("base_model"),
                    fallback_session_spec=registered_session_spec,
                )

                metadata_path = os.path.join(path, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {}

                weights_path = os.path.join(path, "adapter_model.safetensors")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"Weights file not found: {weights_path}")

                loaded_weights = safetensors_load_file(weights_path)
                payload[0] = {
                    "error": None,
                    "session_spec": checkpoint_session_spec,
                    "metadata": metadata,
                    "weights": {name: tensor.cpu() for name, tensor in loaded_weights.items()},
                    "optimizer_present": load_optimizer and os.path.exists(os.path.join(path, "optimizer.pt")),
                }
            except Exception as e:
                payload[0] = {"error": str(e)}

        dist.broadcast_object_list(payload, src=0, group=self.cpu_group)
        if payload[0].get("error"):
            raise RuntimeError(payload[0]["error"])
        return payload[0]

    def _apply_broadcast_adapter_checkpoint_payload(
        self,
        model_id: str,
        payload: Dict[str, Any],
        *,
        load_optimizer: bool,
        lr: Optional[float],
    ) -> None:
        """Convert a rank0-broadcast checkpoint payload into this rank's local adapter tensors."""
        adapter_manager = self.trainer.adapter_manager
        state = adapter_manager.get_adapter_state(model_id)
        self._validate_broadcast_checkpoint_session_spec(
            model_id,
            payload.get("session_spec"),
            load_optimizer=load_optimizer,
            lr=lr,
        )
        expected_param_map, expected_shapes = self._expected_adapter_param_maps(model_id)
        expected_shard_specs = get_lora_tensor_shard_specs(adapter_manager.model, names=expected_shapes.keys())

        converted_weights = convert_peft_lora_state_dict(
            payload["weights"],
            expected_shapes=expected_shapes,
            expected_shard_specs=expected_shard_specs,
        )

        checkpoint_tensors: Dict[str, torch.Tensor] = {}
        for converted_name, weight in converted_weights.items():
            canonical_name = adapter_manager._canonical_lora_param_name(converted_name)
            if canonical_name in checkpoint_tensors:
                raise ValueError(
                    f"Checkpoint contains duplicate LoRA tensors after canonicalization. param={canonical_name!r}"
                )
            checkpoint_tensors[canonical_name] = weight

        expected_param_names = set(expected_param_map)
        checkpoint_param_names = set(checkpoint_tensors)
        missing_param_names = sorted(expected_param_names - checkpoint_param_names)
        unexpected_param_names = sorted(checkpoint_param_names - expected_param_names)
        if missing_param_names or unexpected_param_names:
            raise ValueError(
                "Checkpoint LoRA parameter set does not match the live adapter structure. "
                f"missing={missing_param_names!r}, unexpected={unexpected_param_names!r}"
            )

        for internal_name, tensor in checkpoint_tensors.items():
            target_param = state.lora_params[expected_param_map[internal_name]]
            if tuple(tensor.shape) != tuple(target_param.shape):
                raise ValueError(
                    "Checkpoint tensor shape does not match the live adapter shape. "
                    f"param={internal_name!r}, checkpoint={tuple(tensor.shape)!r}, "
                    f"live={tuple(target_param.shape)!r}"
                )

        for internal_name, tensor in checkpoint_tensors.items():
            target_param = state.lora_params[expected_param_map[internal_name]]
            target_param.data.copy_(tensor.to(device=target_param.device, dtype=target_param.dtype))

        metadata = payload.get("metadata", {})
        state.global_step = metadata.get("global_step", 0)
        state.global_forward_backward_step = metadata.get("global_forward_backward_step", 0)
        if lr is not None:
            adapter_manager.set_lr(model_id, lr)
        elif "lr" in metadata:
            adapter_manager.set_lr(model_id, metadata["lr"])
        state.last_access_time = time.time()

    def _restore_ep_sharded_rank0_broadcast_adapter_state(
        self,
        model_id: str,
        path: str,
        *,
        load_optimizer: bool,
        lr: Optional[float],
    ) -> Dict[str, Any]:
        """Restore an EP-sharded LoRA adapter without broadcasting rank 0's local expert slice."""
        start_time = time.time()
        payload = self._rank0_load_adapter_checkpoint_payload(model_id, path, load_optimizer)
        self._apply_broadcast_adapter_checkpoint_payload(
            model_id,
            payload,
            load_optimizer=load_optimizer,
            lr=lr,
        )

        if payload.get("optimizer_present") and self.rank == 0:
            logger.warning(
                "Skipping optimizer restore for EP-sharded rank0_broadcast adapter load because optimizer.pt "
                "contains rank-local optimizer tensors. Adapter weights and metadata were restored safely."
            )

        state = self.trainer.adapter_manager.get_adapter_state(model_id)
        return {
            "path": path,
            "model_id": model_id,
            "step": state.global_step,
            "load_time": time.time() - start_time,
            "success": True,
        }

    def _get_adapter_state_load_mode(self) -> str:
        """Return the configured adapter-state restore mode."""
        mode = getattr(self.trainer, "lora_config", {}).get("adapter_state_load_mode", "all_ranks")
        if mode not in _ADAPTER_STATE_LOAD_MODES:
            raise ValueError(
                f"Unsupported adapter_state_load_mode: {mode!r}. "
                f"Supported: {', '.join(sorted(_ADAPTER_STATE_LOAD_MODES))}."
            )
        return mode

    def _sync_collective_error(self, local_error: Optional[str]) -> Optional[str]:
        """Synchronize restore/registration failures before collective broadcast."""
        if self.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return local_error

        group = self.cpu_group
        if group is not None:
            backend = dist.get_backend(group)
        else:
            backend = dist.get_backend()
        device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if backend == "nccl" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        has_error = torch.tensor([1 if local_error else 0], dtype=torch.int64, device=device)
        dist.all_reduce(has_error, op=dist.ReduceOp.MAX, group=group)

        if has_error.item() == 0:
            return None

        error_strings = [None] * self.world_size
        dist.all_gather_object(error_strings, local_error or "", group=group)
        errors = {i: msg for i, msg in enumerate(error_strings) if msg}
        if errors:
            return "; ".join(f"rank {i}: {msg}" for i, msg in errors.items())
        return local_error

    def _rollback_created_adapter(self, model_id: str, created_adapter: bool) -> None:
        """Remove a newly materialized adapter after a failed restore attempt."""
        if not created_adapter or self.trainer.adapter_manager is None:
            return
        if not self.trainer.adapter_manager.has_adapter(model_id):
            return
        try:
            self.trainer.adapter_manager.remove_adapter(model_id)
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to roll back adapter '{model_id}' after restore error: {e}")

    def _rollback_session_registration(self, model_id: str, *, had_session_spec: bool, had_adapter: bool) -> None:
        """Remove newly installed adapter/session state after a failed registration."""
        if (
            not had_adapter
            and self.trainer.adapter_manager is not None
            and self.trainer.adapter_manager.has_adapter(model_id)
        ):
            try:
                self.trainer.adapter_manager.remove_adapter(model_id)
            except Exception as e:
                logger.warning(
                    f"Rank {self.rank}: Failed to roll back adapter '{model_id}' after registration error: {e}"
                )

        if not had_session_spec and model_id in self.trainer.lora_session_specs:
            self.trainer.lora_session_specs.pop(model_id, None)

    def _ensure_adapter_materialized_for_restore(self, model_id: str, lr: float) -> bool:
        """Materialize a nonresident adapter and fail collectively if any rank cannot."""
        created_adapter = False
        local_error = None
        if not self.trainer.adapter_manager.has_adapter(model_id):
            try:
                self.trainer.register_lora_adapter(model_id, lr)
                created_adapter = True
            except Exception as e:
                local_error = f"Failed to register adapter for restore: {e}"

        synced_error = self._sync_collective_error(local_error)
        if synced_error:
            self._rollback_created_adapter(model_id, created_adapter)
            raise RuntimeError(synced_error)

        return created_adapter

    def _ensure_fresh_adapter_materialized(self, model_id: str) -> float:
        """Materialize a fresh nonresident adapter and sync failures before broadcast."""
        created_adapter = False
        local_error = None
        default_lr = None
        try:
            session_spec = self.trainer.get_lora_session_spec(model_id)
            default_lr = session_spec["optimizer_config"]["learning_rate"]
            if not self.trainer.adapter_manager.has_adapter(model_id):
                self.trainer.register_lora_adapter(model_id, default_lr)
                created_adapter = True
        except Exception as e:
            local_error = f"Failed to register fresh adapter for model_id={model_id}: {e}"

        synced_error = self._sync_collective_error(local_error)
        if synced_error:
            self._rollback_created_adapter(model_id, created_adapter)
            raise RuntimeError(synced_error)

        if default_lr is None:
            default_lr = self.trainer.adapter_manager.get_adapter_state(model_id).lr
        return default_lr

    def _restore_adapter_state(
        self,
        model_id: str,
        path: str,
        *,
        load_optimizer: bool,
        lr: Optional[float],
        default_lr: float,
        created_adapter: bool = False,
    ) -> Dict[str, Any]:
        """Restore adapter state using the configured rank loading strategy."""
        mode = self._get_adapter_state_load_mode()
        result = None
        local_error = None

        try:
            if mode == "rank0_broadcast" and self.world_size > 1 and self._has_ep_sharded_adapter_params(model_id):
                result = self._restore_ep_sharded_rank0_broadcast_adapter_state(
                    model_id=model_id,
                    path=path,
                    load_optimizer=load_optimizer,
                    lr=lr,
                )
            elif mode == "all_ranks" or self.world_size <= 1:
                result = self.trainer.load_adapter_state(
                    model_id=model_id,
                    path=path,
                    load_optimizer=load_optimizer,
                    lr=lr,
                )
            elif self.rank == 0:
                result = self.trainer.load_adapter_state(
                    model_id=model_id,
                    path=path,
                    load_optimizer=load_optimizer,
                    lr=lr,
                )
        except Exception as e:
            local_error = f"Adapter state restore failed for model_id={model_id}: {e}"

        synced_error = self._sync_collective_error(local_error)
        if synced_error:
            self._rollback_created_adapter(model_id, created_adapter)
            raise RuntimeError(synced_error)

        # In all_ranks mode every rank has loaded its own local tensor contents.
        # The EP-sharded rank0_broadcast path also materializes each rank's local
        # expert slice directly from the full checkpoint tensors.
        if mode == "rank0_broadcast" and not self._has_ep_sharded_adapter_params(model_id):
            self.broadcast_adapter_state(model_id, default_lr)
            if load_optimizer and self.world_size > 1:
                self.broadcast_adapter_optimizer_state(model_id)

        if result is None:
            adapter_state = self.trainer.adapter_manager.get_adapter_state(model_id)
            result = {
                "success": True,
                "model_id": model_id,
                "step": adapter_state.global_step,
            }
        return result

    # ========================================================================
    # Auto-Load Evicted Adapters
    # ========================================================================

    def _find_evicted_checkpoint(self, model_id: str) -> Optional[str]:
        """Look for an evicted adapter checkpoint on disk.

        Args:
            model_id: The adapter/session ID to find

        Returns:
            Path to the evicted checkpoint, or None if not found
        """
        evicted_path = os.path.join(
            self.trainer.adapter_manager.checkpoint_dir,
            "evicted",
            model_id,
        )
        if os.path.exists(evicted_path):
            return evicted_path
        return None

    def _resolve_evicted_checkpoint(self, model_id: str) -> Optional[str]:
        """Resolve the evicted checkpoint path using the configured load mode."""
        if (
            self.world_size > 1
            and self.cpu_group is not None
            and self._get_adapter_state_load_mode() == "rank0_broadcast"
        ):
            checkpoint_ref = [None]
            if self.rank == 0:
                checkpoint_ref[0] = self._find_evicted_checkpoint(model_id)
            dist.broadcast_object_list(checkpoint_ref, src=0, group=self.cpu_group)
            return checkpoint_ref[0]
        return self._find_evicted_checkpoint(model_id)

    def _register_fresh_adapter(self, model_id: str, lr: float = 1e-5) -> None:
        """Register a new adapter with fresh weights. Raises on failure.

        Args:
            model_id: The adapter/session ID to register
            lr: Learning rate for the new adapter
        """
        try:
            self.trainer.register_lora_adapter(model_id, lr)
        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Failed to auto-register adapter '{model_id}': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Adapter for model_id={model_id} not registered. "
                f"Current adapters: {len(self.trainer.adapter_manager.adapters)}"
                f"/{self.trainer.adapter_manager.max_adapters}. "
                f"The adapter may have been evicted (LRU) or was never registered. "
                f"Call /api/v1/register_adapter first."
            )

    def auto_load_if_evicted(
        self,
        model_id: str,
        *,
        allow_fresh_materialization: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an adapter was evicted and auto-load from checkpoint if available.

        This enables transparent "swap to disk" behavior where adapters can be
        evicted under memory pressure and automatically reloaded when needed.

        ALL ranks must call this method together (collective operation).

        Args:
            model_id: The adapter/session ID to check
            allow_fresh_materialization: When False, require a real evicted checkpoint
                for nonresident adapters instead of creating fresh step-0 state.

        Returns:
            Tuple of (was_auto_loaded, checkpoint_path)
            - was_auto_loaded: True if adapter was loaded from checkpoint
            - checkpoint_path: Path the adapter was loaded from, or None
        """
        if self.trainer.adapter_manager is None:
            return False, None

        if self.trainer.adapter_manager.has_adapter(model_id):
            return False, None

        # Look for evicted checkpoint
        checkpoint_path = self._resolve_evicted_checkpoint(model_id)

        if checkpoint_path is None:
            if not allow_fresh_materialization:
                raise FileNotFoundError(
                    f"Adapter '{model_id}' is not resident and no evicted checkpoint was found under "
                    f"{os.path.join(self.trainer.adapter_manager.checkpoint_dir, 'evicted', model_id)}. "
                    "Refusing to recreate fresh state for this operation because that would discard trained weights."
                )

            # No checkpoint — register fresh adapter
            logger.debug(f"Rank {self.rank}: Auto-registering new adapter '{model_id}' (no previous checkpoint found)")
            default_lr = self._ensure_fresh_adapter_materialized(model_id)
            self.broadcast_adapter_state(model_id, default_lr)
            return True, None

        # Auto-load from checkpoint
        logger.debug(f"Rank {self.rank}: Auto-loading evicted adapter '{model_id}' from checkpoint: {checkpoint_path}")

        try:
            had_session_spec = model_id in self.trainer.lora_session_specs
            if not had_session_spec:
                local_error = None
                try:
                    session_spec = load_session_spec_from_checkpoint(checkpoint_path)
                    self.trainer.register_session(model_id=model_id, session_spec=session_spec, materialize=False)
                except Exception as e:
                    local_error = f"Failed to register session spec from checkpoint for model_id={model_id}: {e}"

                synced_error = self._sync_collective_error(local_error)
                if synced_error:
                    raise RuntimeError(synced_error)

            session_spec = self.trainer.get_lora_session_spec(model_id)
            effective_lr = session_spec["optimizer_config"]["learning_rate"]
            created_adapter = self._ensure_adapter_materialized_for_restore(model_id, effective_lr)
            self._restore_adapter_state(
                model_id=model_id,
                path=checkpoint_path,
                load_optimizer=True,
                lr=None,
                default_lr=effective_lr,
                created_adapter=created_adapter,
            )

            adapter_state = self.trainer.adapter_manager.get_adapter_state(model_id)
            logger.debug(
                f"Rank {self.rank}: Auto-loaded adapter '{model_id}' from {checkpoint_path} "
                f"(step={adapter_state.global_step})"
            )
            return True, checkpoint_path

        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Failed to auto-load adapter '{model_id}' from {checkpoint_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to auto-load adapter '{model_id}' from {checkpoint_path}: {e}") from e

    # ========================================================================
    # Adapter Registration Handler
    # ========================================================================

    async def handle_register_session(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle register session request (all ranks participate)."""
        p: RegisterSessionData = command_dict.get("payload", RegisterSessionData())
        model_id = p.model_id
        session_spec = p.session_spec
        materialize = p.materialize
        had_session_spec = model_id in self.trainer.lora_session_specs
        had_adapter = self.trainer.adapter_manager is not None and self.trainer.adapter_manager.has_adapter(model_id)
        local_error = None
        result = None

        logger.debug(
            f"Rank {self.rank}: Registering session: model_id={model_id}, "
            f"materialize={materialize}, session_spec={session_spec}"
        )

        try:
            result = self.trainer.register_session(
                model_id=model_id,
                session_spec=session_spec,
                materialize=materialize,
            )
        except Exception as e:
            logger.error(f"Rank {self.rank}: register_session failed: {e}", exc_info=True)
            local_error = str(e)

        synced_error = self._sync_collective_error(local_error)
        if synced_error:
            self._rollback_session_registration(
                model_id,
                had_session_spec=had_session_spec,
                had_adapter=had_adapter,
            )
            raise RuntimeError(f"Session registration failed: {synced_error}")

        if (
            materialize
            and self.trainer.adapter_manager is not None
            and self.trainer.adapter_manager.has_adapter(model_id)
        ):
            default_lr = session_spec["optimizer_config"]["learning_rate"]
            self.broadcast_adapter_state(model_id, default_lr)

        if self.rank == 0:
            return result
        return {}

    async def handle_register_adapter(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle register adapter request (all ranks participate).

        Args:
            command_dict: Command dictionary with model_id and lr

        Returns:
            Dict with registration result
        """
        p: RegisterAdapterData = command_dict.get("payload", RegisterAdapterData())
        model_id = p.model_id
        lr = p.lr
        had_session_spec = model_id in self.trainer.lora_session_specs
        had_adapter = self.trainer.adapter_manager is not None and self.trainer.adapter_manager.has_adapter(model_id)
        local_error = None
        result = None

        logger.debug(f"Rank {self.rank}: Registering adapter: model_id={model_id}, lr={lr}")

        try:
            result = self.trainer.register_adapter(model_id=model_id, lr=lr)
        except Exception as e:
            logger.error(f"Rank {self.rank}: register_adapter failed: {e}", exc_info=True)
            local_error = str(e)

        synced_error = self._sync_collective_error(local_error)
        if synced_error:
            self._rollback_session_registration(
                model_id,
                had_session_spec=had_session_spec,
                had_adapter=had_adapter,
            )
            raise RuntimeError(f"Adapter registration failed: {synced_error}")

        self.broadcast_adapter_state(model_id, lr)

        logger.debug(f"Rank {self.rank}: register_adapter completed: model_id={model_id}")

        if self.rank == 0:
            return result
        return {}

    # ========================================================================
    # Adapter State Save/Load Handlers
    # ========================================================================

    async def handle_save_adapter_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle save adapter state request.

        ALL ranks must call this method because it uses full_tensor()
        which is a collective operation. Only rank 0 writes files.

        Args:
            command_dict: Command dictionary with model_id, path, save_optimizer

        Returns:
            Dict with save result (from rank 0), empty dict for other ranks
        """
        try:
            p: AdapterStateData = command_dict.get("payload", AdapterStateData())
            model_id = p.model_id
            path = p.path
            save_optimizer = p.save_optimizer

            logger.debug(f"Rank {self.rank}: Participating in save_adapter_state: model_id={model_id}, path={path}")

            if self.trainer.adapter_manager is not None:
                self.auto_load_if_evicted(model_id, allow_fresh_materialization=False)

            result = self.trainer.save_adapter_state(
                model_id=model_id,
                path=path,
                save_optimizer=save_optimizer,
            )

            logger.debug(f"Rank {self.rank}: save_adapter_state completed: model_id={model_id}")

            if self.rank == 0:
                return result
            else:
                return {}

        except Exception as e:
            logger.error(f"Rank {self.rank}: save_adapter_state failed: {e}", exc_info=True)
            raise RuntimeError(f"Adapter state save failed: {str(e)}") from e

    async def handle_load_adapter_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle load adapter state request.

        Restores adapter state according to `lora.adapter_state_load_mode`:
        either all ranks read the checkpoint locally, or rank 0 reads it and
        broadcasts weights, metadata, and optimizer state.

        Args:
            command_dict: Command dictionary with model_id, path, load_optimizer, lr

        Returns:
            Dict with load result (from rank 0 only)
        """
        had_session_spec = False
        had_adapter = False
        model_id = ""
        try:
            p: AdapterStateData = command_dict.get("payload", AdapterStateData())
            model_id = p.model_id
            path = p.path
            load_optimizer = p.load_optimizer
            lr = p.lr
            had_session_spec = model_id in self.trainer.lora_session_specs
            had_adapter = self.trainer.adapter_manager is not None and self.trainer.adapter_manager.has_adapter(
                model_id
            )

            if not path:
                raise ValueError("path is required for load_adapter_state")

            local_error = None
            if not had_session_spec:
                try:
                    session_spec = load_session_spec_from_checkpoint(path)
                    self.trainer.register_session(model_id=model_id, session_spec=session_spec, materialize=False)
                except Exception as e:
                    local_error = f"Failed to register session spec from checkpoint for model_id={model_id}: {e}"

            synced_error = self._sync_collective_error(local_error)
            if synced_error:
                raise RuntimeError(synced_error)

            session_spec = self.trainer.get_lora_session_spec(model_id)
            # Step 1: ALL ranks register the adapter with fresh weights
            effective_lr = lr if lr is not None else session_spec["optimizer_config"]["learning_rate"]
            created_adapter = self._ensure_adapter_materialized_for_restore(model_id, effective_lr)

            # Step 2: Restore weights + metadata, and optimizer state if requested,
            # using the configured adapter_state_load_mode.
            logger.debug(
                f"Rank {self.rank}: Restoring adapter state from disk: "
                f"model_id={model_id}, path={path}, mode={self._get_adapter_state_load_mode()}"
            )
            result = self._restore_adapter_state(
                model_id=model_id,
                path=path,
                load_optimizer=load_optimizer,
                lr=lr,
                default_lr=effective_lr,
                created_adapter=created_adapter,
            )

            logger.debug(
                f"Rank {self.rank}: load_adapter_state completed: model_id={model_id}, step={result.get('step', 0)}"
            )

            if self.rank == 0:
                return result
            else:
                return {"success": True, "model_id": model_id}

        except Exception as e:
            self._rollback_session_registration(
                model_id,
                had_session_spec=had_session_spec,
                had_adapter=had_adapter,
            )
            logger.error(f"Rank {self.rank}: load_adapter_state failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Adapter state load failed: {str(e)}",
            }

    # ========================================================================
    # Adapter Info Handler
    # ========================================================================

    async def handle_get_adapter_info(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get adapter info request.

        Returns information about loaded adapters in GPU memory.
        Rank-0 only — no multi-rank coordination needed.

        Args:
            command_dict: Command dictionary (no data needed)

        Returns:
            Dict with loaded_adapters, max_adapters, current_adapter_id
        """
        try:
            if self.trainer.adapter_manager is None:
                return {
                    "loaded_adapters": [],
                    "max_adapters": 0,
                    "current_adapter_id": None,
                }

            adapter_manager = self.trainer.adapter_manager

            return {
                "loaded_adapters": adapter_manager.list_adapters(),
                "max_adapters": adapter_manager.max_adapters,
                "current_adapter_id": adapter_manager.current_adapter_id,
            }

        except Exception as e:
            logger.error(f"Rank {self.rank}: get_adapter_info failed: {e}", exc_info=True)
            return {
                "loaded_adapters": [],
                "max_adapters": 0,
                "current_adapter_id": None,
                "error": str(e),
            }

    # ========================================================================
    # Kill Session Handler
    # ========================================================================

    async def handle_kill_session(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle kill session request (full-weights training only).

        In LoRA mode this removes the resident adapter and the registered
        session spec from workers. In full-weights mode it resets the
        single active session.

        Args:
            command_dict: Command dictionary with model_id and save_checkpoint

        Returns:
            Dict with success status and optional checkpoint path
        """
        p: KillSessionData = command_dict.get("payload", KillSessionData())
        model_id = p.model_id
        save_checkpoint = p.save_checkpoint

        logger.debug(
            f"Rank {self.rank}: Handling kill_session for model_id={model_id}, save_checkpoint={save_checkpoint}"
        )

        local_error = None
        result = None
        try:
            result = self.trainer.kill_session(model_id=model_id, save_checkpoint=save_checkpoint)
        except Exception as e:
            logger.error(f"Rank {self.rank}: kill_session failed: {e}", exc_info=True)
            local_error = str(e)

        synced_error = self._sync_collective_error(local_error)
        if synced_error:
            raise RuntimeError(f"Kill session failed: {synced_error}")

        if self.world_size > 1:
            dist.barrier()

        logger.debug(f"Rank {self.rank}: kill_session completed: {result}")
        return result
