"""
RunnerDispatcher - ROUTER Socket Communication with Executor.

This module handles ZMQ communication between the Executor and ModelRunner for
distributed training. Uses ROUTER-DEALER socket pair for reliable communication
that supports reconnections.

Architecture:
-------------
- Rank 0: Communicates with Executor via ZMQ, broadcasts commands, scatters batches
- Other ranks: Receive commands via broadcast, receive batches via scatter
- All ranks: Execute commands and participate in distributed operations

Protocol:
---------
1. Initialization:
   Worker (rank 0) → Executor: RunnerReady
   Executor → Worker (rank 0): RunnerAck

2. Request-Response Loop:
   Executor → Worker (rank 0): RunnerDispatchCommand
   Worker (rank 0) → Executor: RunnerAck (immediate)
   Worker (rank 0) broadcasts command type to all ranks
   Worker (rank 0) scatters batches to all ranks
   All ranks process their portion
   All ranks gather results to rank 0
   Worker (rank 0) → Executor: RunnerResponse

Message Types:
--------------
- FORWARD_BACKWARD: Execute forward/backward pass, returns loss and metrics
- OPTIM_STEP: Execute optimizer step, returns grad_norm
- SAVE_STATE: Save checkpoint
- LOAD_STATE: Load checkpoint
- HEALTH_CHECK: Returns worker health status
- SHUTDOWN: Gracefully shuts down worker

Usage:
------
# Single GPU:
python -m xorl.server.runner.runner_dispatcher \\
    sft.yaml \\
    --worker.bind_address tcp://127.0.0.1:5556

# Multi-GPU (DDP with torchrun):
torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    --master-addr=127.0.0.1 \
    --master-port=29501 \
    -m xorl.server.runner.runner_dispatcher \
    examples/server/sft.yaml \
    --worker.bind_address tcp://127.0.0.1:5556

Note: Only rank 0 communicates with Executor via ZMQ.
      Commands are broadcast to all ranks, batches are scattered.
"""

import asyncio
import datetime
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import zmq.asyncio

from xorl.data.collators import TextSequenceShardCollator
from xorl.distributed.parallel_state import get_parallel_state
from xorl.server.protocol.orchestrator_runner import (
    RunnerDispatchCommand,
    RunnerResponse,
)
from xorl.server.protocol.operations import (
    ModelPassData,
    OptimStepData,
    SaveStateData,
    SaveLoraOnlyData,
    LoadStateData,
    SaveFullWeightsData,
    EmptyData,
)
from xorl.server.runner.utils import Rank0Protocol
from xorl.server.runner.utils import (
    apply_sequence_sharding,
    convert_batch_to_tensors,
    simple_sequence_shard,
    validate_batch_shapes,
)
from xorl.server.runner.adapters import AdapterCoordinator
from xorl.server.runner.model_runner import ModelRunner
from xorl.server.weight_sync.handler import WeightSyncHandler


logger = logging.getLogger(__name__)


# ============================================================================
# Distributed Model Worker
# ============================================================================


class RunnerDispatcher:
    """
    Distributed model worker that communicates with Executor via PAIR socket.

    This worker wraps ModelRunner and handles:
    - ZMQ communication with Executor (rank 0 only)
    - Broadcasting commands to all ranks
    - Scattering batches to all ranks
    - Gathering results from all ranks
    - Response formatting

    For multi-GPU setups:
    - Rank 0 handles ZMQ communication
    - Commands are broadcast to all ranks via Gloo (CPU-based, doesn't block GPU)
    - Model operations use NCCL for GPU tensors
    - All ranks execute commands with their portion of data
    """

    def __init__(
        self,
        trainer: ModelRunner,
        rank: int,
        world_size: int,
        bind_address: str = "tcp://127.0.0.1:5556",
        device: str = "cuda:0",
        cpu_group: Optional[dist.ProcessGroup] = None,
        output_dir: str = "outputs",
    ):
        """
        Initialize distributed model worker.

        Args:
            trainer: ModelRunner instance for model operations
            rank: Worker rank in distributed setup
            world_size: Total number of workers
            bind_address: ZMQ PAIR socket address to bind (rank 0 only)
            device: Device string for logging
            cpu_group: Gloo process group for CPU-based communication (broadcast_object_list)
            output_dir: Output directory for checkpoints and address file (should be on shared filesystem for multi-node)
        """
        self.trainer = trainer
        self.rank = rank
        self.world_size = world_size
        self.bind_address = bind_address
        self.device = device
        self.output_dir = output_dir

        # Gloo process group for CPU-based communication (broadcast_object_list)
        # Using Gloo instead of NCCL for command broadcasting:
        # 1. Doesn't hold GPU resources while waiting for commands
        # 2. More efficient for small CPU data (command dicts)
        # 3. NCCL is reserved for actual tensor operations
        self.cpu_group = cpu_group

        # ZMQ context (rank 0 only)
        self.context: Optional[zmq.asyncio.Context] = None

        # Protocol handler (rank 0 only)
        self._protocol: Optional[Rank0Protocol] = None

        # Initialize sequence shard collator for Ulysses sequence parallelism
        # Each rank will apply sharding based on its own cp_rank after receiving full batches
        self._sequence_shard_collator: Optional[TextSequenceShardCollator] = None
        parallel_state = get_parallel_state()
        if parallel_state.cp_enabled:
            self._sequence_shard_collator = TextSequenceShardCollator()
            logger.info(
                f"Rank {rank}: Initialized TextSequenceShardCollator for sequence parallelism "
                f"(cp_size={parallel_state.cp_size}, cp_rank={parallel_state.cp_rank})"
            )

        # State
        self._running = False

        # Weight sync handler (owns synchronizers, RDMA caches, streaming state)
        self._weight_sync_handler = WeightSyncHandler(rank, world_size, trainer)

        # Adapter coordinator (multi-rank adapter lifecycle: broadcast, eviction, save/load)
        self._adapter_coordinator = AdapterCoordinator(trainer, rank, world_size, cpu_group)

        logger.info(
            f"RunnerDispatcher initialized (rank={rank}/{world_size}, "
            f"bind_address={bind_address}, device={device})"
        )

    async def start(self):
        """Start the worker. Rank 0 handles ZMQ, all ranks participate in distributed ops."""
        if self._running:
            logger.warning(f"Rank {self.rank}: Worker already running")
            return

        logger.info(f"Rank {self.rank}: Starting RunnerDispatcher...")
        logger.info(f"Rank {self.rank}:   Bind Address: {self.bind_address}")
        logger.info(f"Rank {self.rank}:   Rank: {self.rank}, World Size: {self.world_size}")
        logger.info(f"Rank {self.rank}:   Device: {self.device}")

        self._running = True

        try:
            if self.rank == 0:
                # Rank 0: Handle ZMQ communication via protocol handler
                self.context = zmq.asyncio.Context()
                self._protocol = Rank0Protocol(
                    bind_address=self.bind_address,
                    output_dir=self.output_dir,
                    rank=self.rank,
                    world_size=self.world_size,
                    device=self.device,
                    cpu_group=self.cpu_group,
                    request_handler=self._handle_request_rank0,
                    context=self.context,
                )
                await self._protocol.run()  # blocks until shutdown
            else:
                # Other ranks: Wait for broadcast commands from rank 0
                logger.info(f"Rank {self.rank}: Worker ready (waiting for broadcasts from rank 0)")
                await self._worker_event_loop()
        except Exception as e:
            logger.error(f"Rank {self.rank}: CRITICAL ERROR in start(): {e}", exc_info=True)
            raise

    async def stop(self):
        """Stop the worker."""
        if not self._running:
            return

        logger.info(f"Rank {self.rank}: Stopping RunnerDispatcher...")
        self._running = False

        # Only rank 0 has ZMQ resources to clean up
        if self.rank == 0:
            if self._protocol:
                await self._protocol.stop()

            if self.context:
                self.context.term()

            # Print statistics
            if self._protocol:
                logger.info(f"Rank {self.rank}: Final Statistics:")
                logger.info(f"Rank {self.rank}:   Total Requests: {self._protocol.request_count}")
                logger.info(f"Rank {self.rank}:   Successful: {self._protocol.success_count}")
                logger.info(f"Rank {self.rank}:   Failed: {self._protocol.failure_count}")

        logger.info(f"Rank {self.rank}: RunnerDispatcher stopped")

    async def _worker_event_loop(self):
        """Event loop for worker ranks (non-rank-0) - wait for broadcast commands."""
        logger.info(f"Rank {self.rank}: Starting worker event loop...")

        while self._running:
            try:
                # Wait for broadcast command from rank 0
                # Using Gloo (CPU-based) instead of NCCL to avoid holding GPU resources
                command_obj = [None]
                if self.world_size > 1:
                    try:
                        # Add small sleep before broadcast to reduce CPU spinning when idle
                        await asyncio.sleep(0.01)
                        dist.broadcast_object_list(command_obj, src=0, group=self.cpu_group)
                    except Exception as broadcast_error:
                        logger.error(f"Rank {self.rank}: Broadcast error: {broadcast_error}", exc_info=True)
                        await asyncio.sleep(0.1)
                        continue
                else:
                    # Single worker mode, just sleep
                    await asyncio.sleep(0.1)
                    continue

                command_dict = command_obj[0]
                if command_dict is None:
                    # No command received, sleep a bit longer before next poll
                    await asyncio.sleep(0.05)
                    continue

                command_type = command_dict.get("command")

                # Handle heartbeat - just a keepalive, no action needed
                if command_type == "heartbeat":
                    logger.debug(f"Rank {self.rank}: Received heartbeat from rank 0")
                    continue

                logger.debug(f"Rank {self.rank}: Received broadcast command: {command_type}")

                if command_type == "shutdown":
                    logger.info(f"Rank {self.rank}: Shutting down")
                    self._running = False
                    break

                # Execute command based on type
                try:
                    await self._handle_worker_command(command_dict)
                    logger.debug(f"Rank {self.rank}: Command {command_type} completed successfully")
                except Exception as cmd_error:
                    # Log gracefully - only include traceback for unexpected errors
                    error_msg = str(cmd_error)
                    if "sleep mode" in error_msg.lower():
                        # Expected error - log without traceback
                        logger.error(f"Rank {self.rank}: {error_msg}")
                    else:
                        # Unexpected error - log with traceback
                        logger.error(
                            f"Rank {self.rank}: Error executing command {command_type}: {cmd_error}", exc_info=True
                        )
                    # Continue to next command instead of crashing - DON'T break or set _running=False

            except asyncio.CancelledError:
                logger.info(f"Rank {self.rank}: Worker event loop cancelled")
                break
            except Exception as e:
                logger.error(f"Rank {self.rank}: Unexpected error in worker event loop: {e}", exc_info=True)
                # Don't break the loop, just continue
                await asyncio.sleep(0.1)

        logger.info(f"Rank {self.rank}: Worker event loop stopped")

    # Maps operation string -> custom prepare method name (operations not listed use default pass-through)
    _PREPARE_HANDLERS = {
        "save_state": "_prepare_save_state_command",
        "save_lora_only": "_prepare_save_lora_only_command",
        "save_full_weights": "_prepare_save_full_weights_command",
        "load_state": "_prepare_load_state_command",
        "save_weights_for_sampler": "_prepare_save_weights_for_sampler_command",
        "sleep": "_prepare_sleep_command",
        "wake_up": "_prepare_wake_up_command",
    }

    # Maps command string -> handler method name (used by both rank 0 execute and worker dispatch)
    _COMMAND_HANDLERS = {
        "forward": "_handle_forward",
        "forward_backward": "_handle_forward_backward",
        "optim_step": "_handle_optim_step",
        "save_state": "_handle_save_state",
        "save_lora_only": "_handle_save_lora_only",
        "save_full_weights": "_handle_save_full_weights",
        "load_state": "_handle_load_state",
        "save_weights_for_sampler": "_handle_save_weights_for_sampler",
        "sleep": "_handle_sleep",
        "wake_up": "_handle_wake_up",
        "health_check": "_handle_health_check",
        "sync_inference_weights": "_handle_sync_inference_weights",
        "register_adapter": "_handle_register_adapter",
        "save_adapter_state": "_handle_save_adapter_state",
        "load_adapter_state": "_handle_load_adapter_state",
        "get_adapter_info": "_handle_get_adapter_info",
        "kill_session": "_handle_kill_session",
    }

    # Commands handled on rank 0 only (no broadcast needed)
    _NO_BROADCAST = {"health_check", "get_adapter_info"}

    async def _handle_request_rank0(self, request: RunnerDispatchCommand) -> RunnerResponse:
        """
        Handle a request on rank 0 - prepare data, broadcast to all ranks, then execute.

        Args:
            request: RunnerDispatchCommand to process

        Returns:
            RunnerResponse with results or error
        """
        start_time = time.time()

        try:
            # Handle rank-0-only commands (no broadcast)
            if request.operation in self._NO_BROADCAST:
                handler = getattr(self, self._COMMAND_HANDLERS.get(request.operation, ""), None)
                if handler:
                    result = await handler({})
                    return RunnerResponse(
                        request_id=request.message_id, success=True, result=result, execution_time=time.time() - start_time
                    )

            # Prepare command dict (custom prepare or default pass-through)
            prepare_method_name = self._PREPARE_HANDLERS.get(request.operation)
            if prepare_method_name:
                command_dict = await getattr(self, prepare_method_name)(request)
            else:
                command_dict = {
                    "command": request.operation,
                    "request_id": request.message_id,
                    "payload": request.payload,
                }

            # Broadcast command to all ranks using Gloo (CPU-based, doesn't block GPU)
            if self.world_size > 1 and command_dict:
                logger.debug(f"Rank {self.rank}: Broadcasting command: {request.operation}")
                command_obj = [command_dict]
                dist.broadcast_object_list(command_obj, src=0, group=self.cpu_group)
                logger.debug(f"Rank {self.rank}: Broadcast completed: {request.operation}")

            # Execute command on rank 0
            command_type = command_dict.get("command", "")
            handler_name = self._COMMAND_HANDLERS.get(command_type)
            if handler_name is None:
                return RunnerResponse(
                    request_id=request.message_id,
                    success=False,
                    error=f"Unknown operation: {request.operation}",
                    execution_time=time.time() - start_time,
                )
            result = await getattr(self, handler_name)(command_dict)

            return RunnerResponse(
                request_id=request.message_id, success=True, result=result, execution_time=time.time() - start_time
            )

        except Exception as e:
            # Log gracefully - only include traceback for unexpected errors
            error_msg = str(e)
            if "sleep mode" in error_msg.lower():
                logger.error(f"Rank {self.rank}: {error_msg}")
            else:
                logger.error(f"Rank {self.rank}: Error handling request: {e}", exc_info=True)

            return RunnerResponse(
                request_id=request.message_id, success=False, error=str(e), execution_time=time.time() - start_time
            )

    async def _handle_worker_command(self, command_dict: Dict[str, Any]):
        """
        Handle a command on worker ranks (non-rank-0).

        Args:
            command_dict: Command dictionary broadcast from rank 0
        """
        command_type = command_dict.get("command")
        handler_name = self._COMMAND_HANDLERS.get(command_type)
        if handler_name is None:
            logger.warning(f"Rank {self.rank}: Unknown command type: {command_type}")
            return
        await getattr(self, handler_name)(command_dict)

    def _convert_batch_to_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert batch data from lists to torch tensors, with padding if needed."""
        return convert_batch_to_tensors(batch, rank=self.rank)

    def _validate_batch_shapes(self, batch: Dict[str, Any], batch_idx: int = 0) -> bool:
        """Validate that all sequence tensors in a batch have consistent shapes."""
        return validate_batch_shapes(batch, rank=self.rank, batch_idx=batch_idx)

    def _simple_sequence_shard(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Simple sequence sharding for non-packed batches (batch_size > 1)."""
        return simple_sequence_shard(batch, rank=self.rank)

    def _apply_sequence_sharding(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply appropriate sequence sharding based on batch format."""
        return apply_sequence_sharding(batch, rank=self.rank, sequence_shard_collator=self._sequence_shard_collator)

    # ========================================================================
    # Forward / Forward-Backward Handlers
    # ========================================================================

    async def _handle_forward_backward(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle forward_backward on all ranks."""
        if self.rank == 0:
            return await self._handle_compute_rank0_scatter(command_dict, with_backward=True)
        await self._handle_compute_worker_receive(command_dict, with_backward=True)
        return {}

    async def _handle_forward(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle forward on all ranks (no gradient computation)."""
        if self.rank == 0:
            return await self._handle_compute_rank0_scatter(command_dict, with_backward=False)
        await self._handle_compute_worker_receive(command_dict, with_backward=False)
        return {}

    async def _handle_compute_rank0_scatter(
        self, command_dict: Dict[str, Any], with_backward: bool
    ) -> Dict[str, Any]:
        """Rank 0: convert batches, scatter to all ranks, run compute, gather metrics.

        Args:
            command_dict: Command payload containing batches, loss_fn, etc.
            with_backward: If True, run forward+backward (training); if False, forward-only.
        """
        p: ModelPassData = command_dict.get("payload", ModelPassData())
        batches = p.batches or []
        loss_fn = p.loss_fn
        loss_fn_params = p.loss_fn_params
        routed_experts = p.routed_experts
        model_id = p.model_id if with_backward else None

        # Auto-load adapter if it was evicted (all ranks must call this together)
        was_auto_loaded, auto_load_path = False, None
        if with_backward:
            was_auto_loaded, auto_load_path = self._adapter_coordinator.auto_load_if_evicted(model_id)

        # Convert batches to tensors
        converted_batches = [self._convert_batch_to_tensors(batch) for batch in batches]

        # Get sequence parallelism info
        parallel_state = get_parallel_state()
        cp_size = parallel_state.cp_size
        cp_enabled = parallel_state.cp_enabled

        # Scatter batches to all ranks
        # NOTE: For FSDP, ALL ranks must participate in forward/backward with the same number of batches
        # With SP enabled, all ranks in an SP group must process the SAME batch (each rank shards locally)
        # With PP, all pipeline stages process the SAME microbatches (data flows through the pipeline).
        num_batches = len(converted_batches)

        if self.world_size > 1:
            # Calculate effective DP size (accounting for sequence parallelism and PP)
            # Each SP group acts as one DP unit - all ranks in SP group process same batch
            # Each PP pipeline processes the same microbatches - don't split data across PP stages
            pp_size = parallel_state.pp_size if parallel_state.pp_enabled else 1
            dp_size = self.world_size // (cp_size * pp_size)

            # Calculate batches per DP group (not per rank)
            # With PP, each rank needs at least pp_size microbatches for the pipeline schedule
            batches_per_dp_group = (num_batches + dp_size - 1) // dp_size
            if pp_size > 1:
                batches_per_dp_group = max(batches_per_dp_group, pp_size)
            total_needed = batches_per_dp_group * dp_size

            # Pad with dummy batches that have valid inputs but labels=-100.
            # We clone input fields (input_ids, attention_mask, position_ids) from a
            # real batch so the model forward pass produces valid outputs (no NaN).
            # Setting labels=-100 means loss=0 and gradient_accumulate_loss produces
            # grad_scale=0 (local_valid_tokens=0), so dummy ranks contribute nothing
            # to the gradient while still participating in FSDP collectives correctly.
            if num_batches < total_needed:
                num_dummy = total_needed - num_batches
                logger.debug(
                    f"Rank {self.rank}: Padding with {num_dummy} dummy batches "
                    f"(valid inputs, labels=-100) to fill {total_needed} DP slots"
                )
                for i in range(num_dummy):
                    src_batch = converted_batches[i % num_batches]
                    converted_batches.append(self._create_dummy_batch(src_batch))

            # Distribute batches across ranks, accounting for sequence parallelism and PP
            batches_per_rank = self._distribute_batches_to_ranks(
                converted_batches, dp_size, cp_size, cp_enabled, batches_per_dp_group,
                pp_size=pp_size,
            )

            # Compute per-rank R3 datum offset so each rank can slice routed_experts
            if routed_experts is not None:
                self._assign_r3_datum_offsets(
                    batches_per_rank, converted_batches, dp_size, cp_size,
                    cp_enabled, batches_per_dp_group, num_batches
                )

            # Scatter using object list
            output_list = [None]
            dist.scatter_object_list(output_list, batches_per_rank, src=0)
            my_batches = output_list[0]
        else:
            # Single rank mode
            my_batches = converted_batches

        result = self._execute_and_gather(
            my_batches, loss_fn, loss_fn_params, routed_experts,
            cp_enabled, parallel_state,
            with_backward=with_backward, model_id=model_id, is_rank0=True,
        )
        del converted_batches

        # Add auto-load info to result if adapter was loaded from checkpoint
        if was_auto_loaded:
            result["auto_loaded"] = True
            result["auto_load_path"] = auto_load_path

        return result

    async def _handle_compute_worker_receive(
        self, command_dict: Dict[str, Any], with_backward: bool
    ) -> None:
        """Worker ranks: receive scattered batches, run compute, participate in collective metrics.

        Args:
            command_dict: Command payload containing loss_fn, etc.
            with_backward: If True, run forward+backward (training); if False, forward-only.
        """
        p: ModelPassData = command_dict.get("payload", ModelPassData())
        loss_fn = p.loss_fn
        loss_fn_params = p.loss_fn_params
        routed_experts = p.routed_experts
        model_id = p.model_id if with_backward else None

        # Auto-load adapter if it was evicted (all ranks must call this together)
        if with_backward:
            self._adapter_coordinator.auto_load_if_evicted(model_id)

        # Receive scattered batches from rank 0
        output_list = [None]
        dist.scatter_object_list(output_list, None, src=0)
        my_batches = output_list[0]

        # Get parallel state for SP info
        parallel_state = get_parallel_state()
        cp_enabled = parallel_state.cp_enabled

        self._execute_and_gather(
            my_batches, loss_fn, loss_fn_params, routed_experts,
            cp_enabled, parallel_state,
            with_backward=with_backward, model_id=model_id, is_rank0=False,
        )

    # -- Helpers for compute handlers --

    def _execute_and_gather(self, my_batches, loss_fn, loss_fn_params, routed_experts,
                            cp_enabled, parallel_state, *, with_backward, model_id, is_rank0):
        """Shard batches, execute compute, gather IS metrics. Shared by rank-0 and workers."""
        my_batches, routed_experts = self._shard_and_slice_batches(
            my_batches, routed_experts, cp_enabled, parallel_state
        )
        result = self._execute_compute(
            my_batches, loss_fn, loss_fn_params, routed_experts,
            with_backward=with_backward, model_id=model_id,
        )
        del my_batches
        self._gather_is_metrics(result, cp_enabled, is_rank0=is_rank0)
        return result

    @staticmethod
    def _create_dummy_batch(src_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Create a dummy batch with valid inputs but labels=-100 (no gradient contribution).

        Clones input fields from src_batch so the model forward pass produces valid
        outputs (no NaN from zero attention masks or invalid inputs). Labels and
        target_tokens are set to -100 (IGNORE_INDEX) so cross-entropy loss = 0 and
        gradient_accumulate_loss produces grad_scale = 0 (local_valid_tokens = 0).
        """
        _LABEL_KEYS = {"labels", "target_tokens"}
        dummy_batch = {}
        for key, value in src_batch.items():
            if key in _LABEL_KEYS:
                # Set all labels to IGNORE_INDEX so loss=0, gradient=0
                if isinstance(value, torch.Tensor):
                    dummy_batch[key] = torch.full_like(value, -100)
                else:
                    dummy_batch[key] = value
            elif isinstance(value, torch.Tensor):
                # Clone input tensors (input_ids, attention_mask, position_ids, etc.)
                dummy_batch[key] = value.clone()
            else:
                dummy_batch[key] = value
        dummy_batch["num_samples"] = 0
        return dummy_batch

    def _distribute_batches_to_ranks(
        self,
        converted_batches: List[Dict[str, Any]],
        dp_size: int,
        cp_size: int,
        cp_enabled: bool,
        batches_per_dp_group: int,
        pp_size: int = 1,
    ) -> List[List[Dict[str, Any]]]:
        """Distribute batches across ranks, accounting for sequence parallelism and PP.

        With SP enabled: all ranks in same SP group get the SAME full batch.
        Each rank applies its own sequence sharding locally based on its cp_rank.

        With PP enabled: all ranks in the same pipeline get the SAME microbatches.
        Data is only split across DP groups (dp_shard * dp_replicate), not across PP stages.
        The device mesh ordering is [pp, ..., dp_shard], so:
            global_rank = pp_stage * (world_size // pp_size) + dp_rank
        """
        batches_per_rank: List[List[Dict[str, Any]]] = [[] for _ in range(self.world_size)]
        ranks_per_stage = self.world_size // pp_size

        if cp_enabled:
            logger.debug(
                f"Rank {self.rank}: SP enabled (cp_size={cp_size}, dp_size={dp_size}), "
                f"broadcasting full batches to SP groups (each rank shards locally)"
            )
            for dp_rank in range(dp_size):
                start_idx = dp_rank * batches_per_dp_group
                end_idx = start_idx + batches_per_dp_group
                dp_batches = converted_batches[start_idx:end_idx]

                for cp_rank in range(cp_size):
                    for pp_stage in range(pp_size):
                        global_rank = pp_stage * ranks_per_stage + dp_rank * cp_size + cp_rank
                        batches_per_rank[global_rank] = [
                            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                            for b in dp_batches
                        ]
        else:
            for dp_rank in range(dp_size):
                start_idx = dp_rank * batches_per_dp_group
                end_idx = start_idx + batches_per_dp_group
                dp_batches = converted_batches[start_idx:end_idx]
                for pp_stage in range(pp_size):
                    global_rank = pp_stage * ranks_per_stage + dp_rank
                    batches_per_rank[global_rank] = dp_batches

        return batches_per_rank

    @staticmethod
    def _assign_r3_datum_offsets(
        batches_per_rank: List[List[Dict[str, Any]]],
        converted_batches: List[Dict[str, Any]],
        dp_size: int,
        cp_size: int,
        cp_enabled: bool,
        batches_per_dp_group: int,
        num_real_batches: int,
    ) -> None:
        """Assign R3 datum offsets to each rank's batches for routed_experts slicing."""
        datum_offset = 0
        for dp_rank in range(dp_size):
            start_idx = dp_rank * batches_per_dp_group
            end_idx = min(start_idx + batches_per_dp_group, num_real_batches)
            dp_datum_count = sum(
                converted_batches[bi].get("num_samples", 1)
                for bi in range(start_idx, end_idx)
            )
            if cp_enabled:
                for cp_rank in range(cp_size):
                    global_rank = dp_rank * cp_size + cp_rank
                    if batches_per_rank[global_rank]:
                        batches_per_rank[global_rank][0]["_r3_datum_offset"] = datum_offset
                        batches_per_rank[global_rank][0]["_r3_datum_count"] = dp_datum_count
            else:
                if batches_per_rank[dp_rank]:
                    batches_per_rank[dp_rank][0]["_r3_datum_offset"] = datum_offset
                    batches_per_rank[dp_rank][0]["_r3_datum_count"] = dp_datum_count
            datum_offset += dp_datum_count

    def _shard_and_slice_batches(
        self,
        my_batches: List[Dict[str, Any]],
        routed_experts: Optional[List[Any]],
        cp_enabled: bool,
        parallel_state: Any,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
        """Validate, apply SP sharding, and slice R3 routing data for this rank's batches.

        Returns:
            Tuple of (sharded_batches, sliced_routed_experts).
        """
        # Log and validate batch shapes before applying sharding
        for i, batch in enumerate(my_batches):
            shapes = {
                k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
                for k, v in batch.items()
                if k in ["input_ids", "labels", "position_ids", "attention_mask"]
            }
            logger.debug(f"Rank {self.rank}: Batch {i} shapes before sharding: {shapes}")
            self._validate_batch_shapes(batch, i)

        # Apply sequence sharding locally if SP is enabled
        if cp_enabled:
            sharded_batches = []
            for i, batch in enumerate(my_batches):
                try:
                    # Store original position_ids for unpacking per-token outputs later
                    if "position_ids" in batch:
                        original_pos_ids = batch["position_ids"]
                        if isinstance(original_pos_ids, torch.Tensor):
                            batch["_original_position_ids"] = original_pos_ids.clone()
                        else:
                            batch["_original_position_ids"] = torch.tensor(
                                original_pos_ids, dtype=torch.long
                            )

                    sharded_batch = self._apply_sequence_sharding(batch)
                    sharded_batches.append(sharded_batch)
                except Exception as e:
                    shapes = {
                        k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
                        for k, v in batch.items()
                    }
                    logger.error(
                        f"Rank {self.rank}: Sharding failed on batch {i}. "
                        f"Shapes: {shapes}. Error: {e}"
                    )
                    raise
            my_batches = sharded_batches
            logger.debug(
                f"Rank {self.rank}: Applied sequence sharding locally "
                f"(cp_rank={parallel_state.cp_rank})"
            )

        # Slice routed_experts for this rank's datum subset
        if routed_experts is not None and my_batches:
            r3_offset = my_batches[0].pop("_r3_datum_offset", None)
            r3_count = my_batches[0].pop("_r3_datum_count", None)
            if r3_offset is not None and r3_count is not None:
                routed_experts = routed_experts[r3_offset:r3_offset + r3_count]

        return my_batches, routed_experts

    def _execute_compute(
        self,
        my_batches: List[Dict[str, Any]],
        loss_fn: str,
        loss_fn_params: Optional[Dict],
        routed_experts: Optional[List[Any]],
        *,
        with_backward: bool,
        model_id: Optional[str],
    ) -> Dict[str, Any]:
        """Execute forward or forward+backward on the model runner."""
        if with_backward:
            return self.trainer.forward_backward(
                my_batches, loss_fn, loss_fn_params,
                model_id=model_id, routed_experts=routed_experts,
            )
        return self.trainer.forward(
            my_batches, loss_fn, loss_fn_params, routed_experts=routed_experts,
        )

    def _gather_is_metrics(
        self, result: Dict[str, Any], cp_enabled: bool, *, is_rank0: bool
    ) -> None:
        """Gather importance-sampling metrics across ranks via all_gather.

        All ranks must call this together when SP is enabled.
        Rank 0 merges IS metrics into its result dict; workers just participate.
        """
        if not (self.world_size > 1 and cp_enabled):
            return

        logger.debug(f"Rank {self.rank}: Gathering results from all ranks to merge IS metrics...")
        all_results = [None] * self.world_size
        dist.all_gather_object(all_results, result)

        if is_rank0:
            # Merge IS metrics (take from any rank that has them)
            for i, rank_result in enumerate(all_results):
                if rank_result:
                    for key in list(rank_result.keys()):
                        if key.startswith("is_") and key not in result:
                            result[key] = rank_result[key]
                            logger.debug(
                                f"Rank {self.rank}: Copied IS metric '{key}' from rank {i}"
                            )
            logger.debug(f"Rank {self.rank}: Final result keys: {list(result.keys())}")

        del all_results

    # ========================================================================
    # Optim Step Handlers
    # ========================================================================

    async def _handle_optim_step(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optim_step on all ranks (unified handler)."""
        p: OptimStepData = command_dict.get("payload", OptimStepData())
        gradient_clip = p.gradient_clip
        lr = p.lr
        model_id = p.model_id or "default"

        # Auto-load adapter if it was evicted (all ranks must call this together)
        # This can happen if forward_backward was done on this adapter, then another adapter
        # was used causing this one to be evicted, and now optim_step is called
        was_auto_loaded, auto_load_path = self._adapter_coordinator.auto_load_if_evicted(model_id)

        # All ranks execute optim_step (synchronized via DDP)
        result = self.trainer.optim_step(gradient_clip=gradient_clip, lr=lr, model_id=model_id)

        # Add auto-load info to result if adapter was loaded from checkpoint
        if was_auto_loaded and self.rank == 0:
            result["auto_loaded"] = True
            result["auto_load_path"] = auto_load_path

        return result if self.rank == 0 else {}

    # ========================================================================
    # Save/Load State Handlers
    # ========================================================================

    async def _prepare_save_state_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare save_state command with computed checkpoint path."""
        p: SaveStateData = request.payload
        checkpoint_path = p.checkpoint_path
        save_optimizer = p.save_optimizer
        use_timestamp = p.use_timestamp
        model_id = p.model_id or "default"

        # If no checkpoint path provided, generate one with timestamp_step_{step} format
        if checkpoint_path is None:
            # Default to checkpoints/ directory in current working directory
            base_dir = os.path.join(os.getcwd(), "checkpoints")
            os.makedirs(base_dir, exist_ok=True)

            current_step = getattr(self.trainer, "step", 0)

            if use_timestamp:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(base_dir, f"{timestamp}_step_{current_step}")
            else:
                checkpoint_path = os.path.join(base_dir, f"checkpoint_step_{current_step}")
        else:
            # User provided a path - use it with optional timestamp suffix
            if use_timestamp:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                current_step = getattr(self.trainer, "step", 0)
                checkpoint_path = checkpoint_path.rstrip("/")
                checkpoint_path = f"{checkpoint_path}_{timestamp}_step_{current_step}"

        return {
            "command": "save_state",
            "request_id": request.message_id,
            "payload": SaveStateData(
                checkpoint_path=checkpoint_path,
                save_optimizer=save_optimizer,
                model_id=model_id,
            ),
        }

    async def _prepare_load_state_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare load_state command."""
        p: LoadStateData = request.payload
        return {
            "command": "load_state",
            "request_id": request.message_id,
            "payload": LoadStateData(
                checkpoint_path=p.checkpoint_path,
                load_optimizer=p.load_optimizer,
                model_id=p.model_id or "default",
            ),
        }

    async def _handle_save_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle save_state on all ranks (unified handler)."""
        p: SaveStateData = command_dict.get("payload", SaveStateData())
        checkpoint_path = p.checkpoint_path
        save_optimizer = p.save_optimizer
        model_id = p.model_id or "default"

        logger.debug(
            f"Rank {self.rank}: Saving state to {checkpoint_path}, "
            f"model_id={model_id}, save_optimizer={save_optimizer}"
        )

        # For LoRA models with save_optimizer=False (sampler weights), use fast PEFT-compatible save
        # This creates adapter_model.safetensors + adapter_config.json that SGLang can load
        is_lora_enabled = self.trainer.lora_config.get("enable_lora", False)
        if is_lora_enabled and not save_optimizer:
            logger.debug(f"Rank {self.rank}: Using save_lora_only for sampler weights (PEFT format)")
            result = self.trainer.save_lora_only(checkpoint_path, model_id=model_id)
        else:
            # NOTE: Cannot use thread pool because trainer.save_state() calls dist.barrier()
            # which requires all ranks to call from the same thread (main thread).
            # This will block the event loop but that's unavoidable for collective operations.
            result = self.trainer.save_state(checkpoint_path, save_optimizer, model_id=model_id)

        logger.debug(f"Rank {self.rank}: save_state completed for model_id={model_id}")

        # Update result with actual checkpoint path used (rank 0 returns this)
        if self.rank == 0 and isinstance(result, dict):
            result["checkpoint_path"] = checkpoint_path

        return result if self.rank == 0 else {}

    async def _prepare_save_lora_only_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare save_lora_only command."""
        p: SaveLoraOnlyData = request.payload
        return {
            "command": "save_lora_only",
            "request_id": request.message_id,
            "payload": SaveLoraOnlyData(
                lora_path=p.lora_path,
                model_id=p.model_id or "default",
            ),
        }

    async def _prepare_save_full_weights_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare save_full_weights command."""
        p: SaveFullWeightsData = request.payload
        return {
            "command": "save_full_weights",
            "request_id": request.message_id,
            "payload": SaveFullWeightsData(
                output_path=p.output_path,
                dtype=p.dtype,
                base_model_path=p.base_model_path,
            ),
        }

    async def _handle_save_lora_only(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle save_lora_only on all ranks (unified handler).

        Saves only LoRA adapter weights in PEFT-compatible format.
        """
        p: SaveLoraOnlyData = command_dict.get("payload", SaveLoraOnlyData())
        lora_path = p.lora_path
        model_id = p.model_id or "default"

        logger.debug(f"Rank {self.rank}: Saving LoRA adapter to {lora_path} for model_id={model_id}")

        # NOTE: Cannot use thread pool because trainer.save_lora_only() calls dist.barrier()
        # which requires all ranks to call from the same thread (main thread).
        result = self.trainer.save_lora_only(lora_path, model_id=model_id)

        logger.debug(f"Rank {self.rank}: save_lora_only completed for model_id={model_id}")

        # Update result with actual lora path used (rank 0 returns this)
        if self.rank == 0 and isinstance(result, dict):
            result["lora_path"] = lora_path
            result["success"] = True

        return result if self.rank == 0 else {}

    async def _handle_save_full_weights(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle save_full_weights on all ranks.

        Saves full model weights as safetensors with config files for SGLang loading.
        """
        p: SaveFullWeightsData = command_dict.get("payload", SaveFullWeightsData())
        output_path = p.output_path
        dtype = p.dtype
        base_model_path = p.base_model_path

        logger.debug(f"Rank {self.rank}: Saving full weights to {output_path} (dtype={dtype})")

        # NOTE: Cannot use thread pool because trainer.save_full_weights()
        # calls dist.barrier() which requires all ranks to call from the same thread.
        result = self.trainer.save_full_weights(
            output_path=output_path,
            dtype=dtype,
            base_model_path=base_model_path,
        )

        logger.debug(f"Rank {self.rank}: save_full_weights completed")

        if self.rank == 0 and isinstance(result, dict):
            result["success"] = True

        return result if self.rank == 0 else {}

    async def _handle_load_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle load_state on all ranks (unified handler)."""
        p: LoadStateData = command_dict.get("payload", LoadStateData())
        checkpoint_path = p.checkpoint_path
        load_optimizer = p.load_optimizer
        model_id = p.model_id or "default"

        logger.debug(
            f"Rank {self.rank}: Loading state from {checkpoint_path}, "
            f"model_id={model_id}, load_optimizer={load_optimizer}"
        )

        # Check if checkpoint path exists
        if not os.path.exists(checkpoint_path):
            error_msg = f"Checkpoint path does not exist: {checkpoint_path}"
            logger.error(f"Rank {self.rank}: {error_msg}")
            raise FileNotFoundError(error_msg)

        logger.debug(f"Rank {self.rank}: Checkpoint path exists: {checkpoint_path}")

        try:
            # NOTE: Cannot use thread pool because trainer.load_state() calls dist.barrier()
            # which requires all ranks to call from the same thread (main thread).
            # This will block the event loop but that's unavoidable for collective operations.
            logger.debug(f"Rank {self.rank}: About to call trainer.load_state()...")

            # Flush logs before the potentially crashing call
            sys.stdout.flush()
            sys.stderr.flush()

            result = self.trainer.load_state(checkpoint_path, load_optimizer, model_id=model_id)
            logger.debug(f"Rank {self.rank}: trainer.load_state() returned successfully")

            # Reset step to 0 after loading state
            self.trainer.step = 0
            logger.debug(f"Rank {self.rank}: load_state completed, reset step to 0")

            return result if self.rank == 0 else {}
        except SystemExit as se:
            logger.error(f"Rank {self.rank}: SystemExit caught during load_state: {se}", exc_info=True)
            raise
        except KeyboardInterrupt as ki:
            logger.error(f"Rank {self.rank}: KeyboardInterrupt caught during load_state: {ki}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Rank {self.rank}: Failed to load state: {e}", exc_info=True)
            raise

    async def _prepare_save_weights_for_sampler_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare save_weights_for_sampler command."""
        p: SaveStateData = request.payload
        checkpoint_path = p.checkpoint_path
        output_path = getattr(p, "output_path", None)
        save_dtype = getattr(p, "save_dtype", "bfloat16")

        # If no output path provided, default to checkpoints/hf_weights/
        if output_path is None:
            base_dir = os.path.join(os.getcwd(), "checkpoints", "hf_weights")
            os.makedirs(base_dir, exist_ok=True)
            output_path = base_dir

        return {
            "command": "save_weights_for_sampler",
            "request_id": request.message_id,
            "payload": SaveStateData(
                checkpoint_path=checkpoint_path,
            ),
            "output_path": output_path,
            "save_dtype": save_dtype,
        }

    async def _handle_save_weights_for_sampler(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle save_weights_for_sampler on all ranks (unified handler)."""
        p: SaveStateData = command_dict.get("payload", SaveStateData())
        checkpoint_path = p.checkpoint_path
        output_path = command_dict.get("output_path")
        save_dtype = command_dict.get("save_dtype", "bfloat16")

        logger.debug(f"Rank {self.rank}: Saving HF weights from {checkpoint_path} to {output_path}, dtype={save_dtype}")

        # NOTE: Cannot use thread pool because it calls dist.barrier()
        result = self.trainer.save_weights_for_sampler(checkpoint_path, output_path, save_dtype)

        logger.debug(f"Rank {self.rank}: save_weights_for_sampler completed")

        return result if self.rank == 0 else {}

    # ========================================================================
    # Sleep/Wake Up Handlers
    # ========================================================================

    async def _prepare_sleep_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare sleep command."""
        return {"command": "sleep", "request_id": request.message_id, "payload": EmptyData()}

    async def _prepare_wake_up_command(self, request: RunnerDispatchCommand) -> Dict[str, Any]:
        """Prepare wake_up command."""
        return {"command": "wake_up", "request_id": request.message_id, "payload": EmptyData()}

    async def _handle_sleep(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sleep on all ranks (unified handler) - offload to CPU."""
        logger.debug(f"Rank {self.rank}: Offloading model and optimizer to CPU...")

        # NOTE: Cannot use thread pool because it calls dist.barrier()
        result = self.trainer.sleep()

        logger.debug(f"Rank {self.rank}: sleep completed")

        return result if self.rank == 0 else {}

    async def _handle_wake_up(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Handle wake_up on all ranks (unified handler) - load to GPU."""
        logger.debug(f"Rank {self.rank}: Loading model and optimizer to GPU...")

        # NOTE: Cannot use thread pool because it calls dist.barrier()
        result = self.trainer.wake_up()

        logger.debug(f"Rank {self.rank}: wake_up completed")

        return result if self.rank == 0 else {}

    # ========================================================================
    # Health Check Handlers
    # ========================================================================

    async def _handle_health_check(self) -> Dict[str, Any]:
        """Handle health check on all ranks (unified handler)."""
        logger.debug(f"Rank {self.rank}: Health check")

        # Only rank 0 returns data
        if self.rank == 0:
            return {
                "status": "healthy",
                "rank": self.rank,
                "world_size": self.world_size,
                "device": self.device,
                "requests_processed": self._protocol.request_count if self._protocol else 0,
                "success_count": self._protocol.success_count if self._protocol else 0,
                "failure_count": self._protocol.failure_count if self._protocol else 0,
            }
        return {}

    # ========================================================================
    # Weight Sync (delegated to WeightSyncHandler)
    # ========================================================================

    async def _handle_sync_inference_weights(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await self._weight_sync_handler.handle_sync_inference_weights(command_dict)


    # ========================================================================
    # Adapter Operations (delegated to AdapterCoordinator)
    # ========================================================================

    def _broadcast_adapter_state(self, model_id: str, default_lr: float) -> None:
        self._adapter_coordinator.broadcast_adapter_state(model_id, default_lr)

    def _auto_load_adapter_if_evicted(self, model_id: str) -> tuple[bool, str | None]:
        return self._adapter_coordinator.auto_load_if_evicted(model_id)

    async def _handle_register_adapter(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await self._adapter_coordinator.handle_register_adapter(command_dict)

    async def _handle_save_adapter_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await self._adapter_coordinator.handle_save_adapter_state(command_dict)

    async def _handle_load_adapter_state(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await self._adapter_coordinator.handle_load_adapter_state(command_dict)

    async def _handle_get_adapter_info(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await self._adapter_coordinator.handle_get_adapter_info(command_dict)

    async def _handle_kill_session(self, command_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await self._adapter_coordinator.handle_kill_session(command_dict)


if __name__ == "__main__":
    from xorl.server.runner.setup import main

    main()
