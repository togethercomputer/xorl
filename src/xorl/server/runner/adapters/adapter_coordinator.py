"""
AdapterCoordinator - Multi-rank adapter lifecycle management.

Extracted from dispatcher.py to separate adapter coordination
concerns (broadcast, auto-load, register, save/load state, info, kill session)
from the core worker communication logic.

All methods that require multi-rank coordination for LoRA adapter management
live here. The RunnerDispatcher delegates to this class.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch.distributed as dist

from xorl.server.protocol.operations import (
    AdapterStateData,
    KillSessionData,
    RegisterAdapterData,
)
from xorl.server.runner.model_runner import ModelRunner

logger = logging.getLogger(__name__)


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

        adapter_state = self.trainer.adapter_manager.get_adapter_state(model_id)

        # Broadcast each parameter
        for name, param in adapter_state.lora_params.items():
            dist.broadcast(param.data, src=0)

        # Broadcast metadata
        metadata = [None]
        if self.rank == 0:
            metadata = [{
                "global_step": adapter_state.global_step,
                "global_forward_backward_step": adapter_state.global_forward_backward_step,
                "lr": adapter_state.lr,
            }]
        dist.broadcast_object_list(metadata, src=0, group=self.cpu_group)

        # Update metadata on non-rank-0 workers
        if self.rank != 0 and metadata[0]:
            adapter_state.global_step = metadata[0].get("global_step", 0)
            adapter_state.global_forward_backward_step = metadata[0].get("global_forward_backward_step", 0)
            adapter_state.lr = metadata[0].get("lr", default_lr)

        logger.debug(f"Rank {self.rank}: Broadcast adapter state for model_id={model_id}")

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

    def auto_load_if_evicted(self, model_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if an adapter was evicted and auto-load from checkpoint if available.

        This enables transparent "swap to disk" behavior where adapters can be
        evicted under memory pressure and automatically reloaded when needed.

        ALL ranks must call this method together (collective operation).

        Args:
            model_id: The adapter/session ID to check

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
        checkpoint_path = self._find_evicted_checkpoint(model_id)

        if checkpoint_path is None:
            # No checkpoint — register fresh adapter
            logger.debug(
                f"Rank {self.rank}: Auto-registering new adapter '{model_id}' "
                f"(no previous checkpoint found)"
            )
            self._register_fresh_adapter(model_id)
            return True, None

        # Auto-load from checkpoint
        logger.debug(
            f"Rank {self.rank}: Auto-loading evicted adapter '{model_id}' "
            f"from checkpoint: {checkpoint_path}"
        )

        try:
            effective_lr = 1e-5  # Default, will be overwritten from checkpoint
            self.trainer.register_lora_adapter(model_id, effective_lr)

            if self.rank == 0:
                self.trainer.load_adapter_state(
                    model_id=model_id,
                    path=checkpoint_path,
                    load_optimizer=True,
                )

            self.broadcast_adapter_state(model_id, effective_lr)

            adapter_state = self.trainer.adapter_manager.get_adapter_state(model_id)
            logger.debug(
                f"Rank {self.rank}: Auto-loaded adapter '{model_id}' from {checkpoint_path} "
                f"(step={adapter_state.global_step})"
            )
            return True, checkpoint_path

        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Failed to auto-load adapter '{model_id}' "
                f"from {checkpoint_path}: {e}",
                exc_info=True,
            )
            return False, None

    # ========================================================================
    # Adapter Registration Handler
    # ========================================================================

    async def handle_register_adapter(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle register adapter request (all ranks participate).

        Args:
            command_dict: Command dictionary with model_id and lr

        Returns:
            Dict with registration result
        """
        try:
            p: RegisterAdapterData = command_dict.get("payload", RegisterAdapterData())
            model_id = p.model_id
            lr = p.lr

            logger.debug(f"Rank {self.rank}: Registering adapter: model_id={model_id}, lr={lr}")

            result = self.trainer.register_adapter(model_id=model_id, lr=lr)

            logger.debug(f"Rank {self.rank}: register_adapter completed: model_id={model_id}")

            if self.rank == 0:
                return result
            else:
                return {}

        except Exception as e:
            logger.error(f"Rank {self.rank}: register_adapter failed: {e}", exc_info=True)
            return {
                "registered": False,
                "error": f"Adapter registration failed: {str(e)}",
            }

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
            return {
                "success": False,
                "error": f"Adapter state save failed: {str(e)}",
            }

    async def handle_load_adapter_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle load adapter state request.

        ALL ranks must register the adapter, but only rank 0 loads weights
        from disk, then broadcasts to other ranks.

        Args:
            command_dict: Command dictionary with model_id, path, load_optimizer, lr

        Returns:
            Dict with load result (from rank 0 only)
        """
        try:
            p: AdapterStateData = command_dict.get("payload", AdapterStateData())
            model_id = p.model_id
            path = p.path
            load_optimizer = p.load_optimizer
            lr = p.lr

            if not path:
                raise ValueError("path is required for load_adapter_state")

            # Step 1: ALL ranks register the adapter with fresh weights
            effective_lr = lr if lr is not None else 1e-5
            if not self.trainer.adapter_manager.has_adapter(model_id):
                logger.debug(f"Rank {self.rank}: Registering adapter for model_id={model_id}")
                self.trainer.register_lora_adapter(model_id, effective_lr)

            # Step 2: Only rank 0 loads weights and optimizer from disk
            result = None
            if self.rank == 0:
                logger.debug(f"Rank {self.rank}: Loading adapter weights from disk: model_id={model_id}, path={path}")

                result = self.trainer.load_adapter_state(
                    model_id=model_id,
                    path=path,
                    load_optimizer=load_optimizer,
                    lr=lr,
                )

                logger.debug(f"Rank {self.rank}: load_adapter_state completed: model_id={model_id}, step={result.get('step', 0)}")

            # Step 3: Broadcast loaded weights from rank 0 to all other ranks
            self.broadcast_adapter_state(model_id, effective_lr)

            if self.rank == 0:
                return result
            else:
                return {"success": True, "model_id": model_id}

        except Exception as e:
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

        In full-weights training mode (enable_lora=False), the server operates in
        single-tenant mode. This kills the active session to allow starting a new one.
        For LoRA mode, this is a no-op since multi-tenancy is supported.

        Args:
            command_dict: Command dictionary with model_id and save_checkpoint

        Returns:
            Dict with success status and optional checkpoint path
        """
        p: KillSessionData = command_dict.get("payload", KillSessionData())
        model_id = p.model_id
        save_checkpoint = p.save_checkpoint

        logger.debug(f"Rank {self.rank}: Handling kill_session for model_id={model_id}, save_checkpoint={save_checkpoint}")

        try:
            result = self.trainer.kill_session(model_id=model_id, save_checkpoint=save_checkpoint)

            if self.world_size > 1:
                dist.barrier()

            logger.debug(f"Rank {self.rank}: kill_session completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Rank {self.rank}: kill_session failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Kill session failed: {str(e)}",
                "checkpoint_path": None,
            }
