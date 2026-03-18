"""
RequestProcessor - Batch Preparation and Backend Dispatch.

This module provides the RequestProcessor class which handles data preparation (packing,
validation) and delegates compute operations to a Backend implementation.

Role in Orchestrator Architecture:
================================

Orchestrator orchestrates three components:
1. **Scheduler**: Orders requests (FIFO policy)
2. **RequestProcessor**: Prepares data and dispatches to Backend (THIS MODULE)
3. **Queues**: Thread-safe communication (input_queue, output_queue)

The RequestProcessor owns:
- Sample packing (datum_list → micro-batches)
- Result formatting (backend result → OrchestratorOutputs)
- Operation statistics

The Backend owns:
- Transport layer (ZMQ, in-process, etc.)
- Worker handshake and lifecycle
- Request serialization and response deserialization

Usage:
=====

```python
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.backend import RemoteBackend

backend = RemoteBackend(worker_address="tcp://127.0.0.1:5556")
executor = RequestProcessor(backend=backend, sample_packing_sequence_len=32000)

await executor.start()
outputs = await executor.execute_forward_backward(request)
await executor.stop()
```
"""

import logging
import math
import time
from typing import Any, Callable, Dict, Optional, Union

import torch

from xorl.server.backend import Backend
from xorl.server.orchestrator.packing import pack_samples, unpack_per_token_outputs, validate_micro_batches
from xorl.server.protocol.api_orchestrator import OrchestratorOutputs, OrchestratorRequest, OutputType
from xorl.server.protocol.operations import (
    LOAD_STATE_TIMEOUT,
    SAVE_STATE_TIMEOUT,
    AdapterStateData,
    KillSessionData,
    LoadStateData,
    ModelPassData,
    OptimStepData,
    RegisterAdapterData,
    SaveFullWeightsData,
    SaveLoraOnlyData,
    SaveStateData,
    SyncWeightsData,
)


logger = logging.getLogger(__name__)


# ============================================================================
# RequestProcessor Class
# ============================================================================


class RequestProcessor:
    """
    Batch preparation and backend dispatch coordinator.

    Handles data preparation (packing, validation) and delegates compute
    operations to a Backend implementation. The Backend handles all transport
    concerns (ZMQ, in-process, etc.).
    """

    def __init__(
        self,
        backend: Backend,
        sample_packing_sequence_len: int = 32000,
        enable_packing: bool = True,
        pad_to_multiple_of: int = 128,
        cp_size: int = 1,
    ):
        """
        Initialize RequestProcessor.

        Args:
            backend: Backend implementation for compute operations
            sample_packing_sequence_len: Maximum sequence length for packing (default: 32000)
            enable_packing: Enable sample packing (default: True)
            pad_to_multiple_of: Base padding alignment (default: 128)
            cp_size: Sequence parallel size. Padded length must be divisible
                by cp_size for Ulysses sequence parallelism. The effective
                padding multiple is lcm(pad_to_multiple_of, cp_size).
        """
        self.backend = backend
        self.sample_packing_sequence_len = sample_packing_sequence_len
        self.enable_packing = enable_packing
        # Sequence must be divisible by both pad_to_multiple_of and cp_size
        self.pad_to_multiple_of = math.lcm(pad_to_multiple_of, cp_size)

        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0

        logger.info(
            f"RequestProcessor initialized: "
            f"sample_packing_sequence_len={sample_packing_sequence_len}, packing={'enabled' if enable_packing else 'disabled'}, "
            f"pad_to_multiple_of={self.pad_to_multiple_of} (base={pad_to_multiple_of}, cp_size={cp_size})"
        )

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def start(self):
        """Start the executor and its backend."""
        logger.info("Starting RequestProcessor...")
        await self.backend.start()
        logger.info("RequestProcessor started successfully")

    async def stop(self):
        """Stop the executor and its backend."""
        logger.info("Stopping RequestProcessor...")
        await self.backend.stop()
        logger.info("RequestProcessor stopped")

    def is_ready(self) -> bool:
        """Check if executor is ready for operations."""
        return self.backend.is_ready()

    # ========================================================================
    # Operation Execution
    # ========================================================================

    async def _execute_model_pass(
        self,
        request: OrchestratorRequest,
        op_name: str,
        output_type: OutputType,
    ) -> OrchestratorOutputs:
        """Shared implementation for forward and forward_backward passes.

        Args:
            request: OrchestratorRequest with data and loss_fn
            op_name: Operation name ("forward" or "forward_backward")
            output_type: OutputType for the response
        """
        logger.debug(f"Executing {op_name} for request {request.request_id}")
        self.total_operations += 1
        t0 = time.perf_counter()

        try:
            # Extract parameters from typed payload
            p: ModelPassData = request.payload
            data = p.data
            loss_fn = p.loss_fn
            loss_fn_params = p.loss_fn_params or {}

            if not data:
                raise ValueError("data or datum_list must be provided")

            # Pack samples into batches
            logger.debug(f"Packing {len(data)} datum into batches for {op_name} request {request.request_id}")
            batches = pack_samples(
                datum_list=data,
                max_seq_len=self.sample_packing_sequence_len,
                enable_packing=self.enable_packing,
                request_id=request.request_id,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

            if not batches:
                raise ValueError(
                    f"No batches created from {len(data)} samples. "
                    "The packer did not produce any valid batches."
                )

            if not validate_micro_batches(batches):
                raise ValueError(
                    "Invalid batch structure after packing. This may indicate a bug in the packing logic."
                )

            t_packed = time.perf_counter()
            logger.debug(f"Packed {len(data)} samples into {len(batches)} batches")

            # Call backend
            backend_method = getattr(self.backend, op_name)
            kwargs = dict(
                batches=batches,
                loss_fn=loss_fn,
                loss_fn_params=loss_fn_params,
                model_id=p.model_id,
                request_id=request.request_id,
            )
            if op_name == "forward_backward":
                kwargs["routed_experts"] = p.routed_experts

            result = await backend_method(**kwargs)

            t_backend = time.perf_counter()

            # Build output dict
            loss = result.get("total_loss", 0.0)
            tokens = result.get("global_valid_tokens", 0)

            output_dict = {
                "loss": loss,
                "valid_tokens": tokens,
                "success": True,
                "execution_time": result.get("execution_time", 0.0),
            }

            # Add IS metrics (KL divergence, ratio stats, etc.)
            for key in result:
                if key.startswith("is_"):
                    output_dict[key] = result[key]

            # Pass through expert load summary for MoE models
            if "expert_load_summary" in result:
                output_dict["expert_load_summary"] = result["expert_load_summary"]

            # Pass through auto-load info if adapter was loaded from checkpoint
            if result.get("auto_loaded"):
                output_dict["auto_loaded"] = True
                output_dict["auto_load_path"] = result.get("auto_load_path")

            # Unpack per-token outputs if present (tinker API compatibility)
            if "packed_logprobs" in result and "packed_position_ids" in result:
                output_dict["per_sample_outputs"] = self._unpack_per_sample_outputs(result, batches)

            output = OrchestratorOutputs(
                request_id=request.request_id,
                output_type=output_type,
                outputs=[output_dict],
                finished=True,
            )

            self.successful_operations += 1
            t_done = time.perf_counter()
            logger.info(
                f"[TIMING] executor {op_name}: "
                f"pack={t_packed - t0:.4f}s "
                f"backend={t_backend - t_packed:.4f}s "
                f"build_output={t_done - t_backend:.4f}s "
                f"total={t_done - t0:.4f}s | "
                f"loss={loss:.4f}, tokens={tokens}"
            )
            return output

        except Exception as e:
            self.failed_operations += 1
            error_msg = str(e)
            if "sleep mode" in error_msg.lower():
                logger.error(f"{op_name} failed: {error_msg}")
            else:
                logger.error(f"Error executing {op_name}: {e}", exc_info=True)

            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.ERROR,
                finished=True,
                error=error_msg,
            )

    @staticmethod
    def _unpack_per_sample_outputs(result: Dict, batches: list) -> list:
        """Unpack packed per-token outputs into per-sample lists.

        Handles both cross_entropy (logprobs + losses) and importance_sampling (logprobs only).
        """
        packed_logprobs = result["packed_logprobs"]
        packed_losses = result.get("packed_losses")
        packed_position_ids = result["packed_position_ids"]
        per_sample_outputs = []

        for i, (logprobs, pos_ids) in enumerate(zip(packed_logprobs, packed_position_ids)):
            logprobs_tensor = torch.tensor(logprobs)
            pos_ids_tensor = torch.tensor(pos_ids)

            sample_logprobs = unpack_per_token_outputs(logprobs_tensor, pos_ids_tensor)

            # Limit to real samples: padding tokens create a spurious sequence boundary
            num_real = batches[i].get("num_samples") if i < len(batches) else None
            if num_real is not None:
                sample_logprobs = sample_logprobs[:num_real]

            if packed_losses is not None:
                losses_tensor = torch.tensor(packed_losses[i])
                sample_losses = unpack_per_token_outputs(losses_tensor, pos_ids_tensor)
                if num_real is not None:
                    sample_losses = sample_losses[:num_real]
                for lp, el in zip(sample_logprobs, sample_losses):
                    per_sample_outputs.append({"logprobs": lp, "elementwise_loss": el})
            else:
                for lp in sample_logprobs:
                    per_sample_outputs.append({"logprobs": lp})

        logger.debug(f"Unpacked {len(per_sample_outputs)} per-sample outputs")
        return per_sample_outputs

    async def execute_forward_backward(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute forward-backward pass on workers."""
        return await self._execute_model_pass(
            request, "forward_backward", OutputType.FORWARD_BACKWARD,
        )

    async def execute_forward(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute forward pass on workers (no gradient computation)."""
        return await self._execute_model_pass(
            request, "forward", OutputType.FORWARD,
        )

    async def _execute_operation(
        self,
        request: OrchestratorRequest,
        op_name: str,
        backend_coro,
        output_type: OutputType,
        build_output: Callable[[Dict], Union[list, dict]],
    ) -> OrchestratorOutputs:
        """Execute an operation with standard logging, counters, and error handling."""
        logger.info(f"Executing {op_name} for request {request.request_id}")
        self.total_operations += 1
        t0 = time.perf_counter()
        try:
            result = await backend_coro
            t_backend = time.perf_counter()
            outputs = build_output(result)
            self.successful_operations += 1
            t_done = time.perf_counter()
            logger.info(
                f"[TIMING] executor {op_name}: "
                f"backend={t_backend - t0:.4f}s "
                f"build_output={t_done - t_backend:.4f}s "
                f"total={t_done - t0:.4f}s"
            )
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=output_type,
                outputs=outputs,
                finished=True,
            )
        except Exception as e:
            self.failed_operations += 1
            error_msg = str(e)
            if "sleep mode" in error_msg.lower():
                logger.error(f"{op_name} failed: {error_msg}")
            else:
                logger.error(f"Error executing {op_name}: {e}", exc_info=True)
            return OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.ERROR,
                finished=True,
                error=error_msg,
            )

    async def execute_optim_step(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute optimizer step on workers."""
        p: OptimStepData = request.payload
        lr = p.lr

        def build_output(result):
            output_dict = {
                "grad_norm": result.get("grad_norm", 0.0),
                "learning_rate": lr,
                "step": result.get("step", 0),
                "execution_time": result.get("execution_time", 0.0),
            }
            if result.get("auto_loaded"):
                output_dict["auto_loaded"] = True
                output_dict["auto_load_path"] = result.get("auto_load_path")
            return [output_dict]

        return await self._execute_operation(
            request, "optim_step",
            self.backend.optim_step(
                lr=p.lr, gradient_clip=p.gradient_clip,
                beta1=p.beta1, beta2=p.beta2, eps=p.eps,
                model_id=p.model_id, request_id=request.request_id,
            ),
            OutputType.OPTIM_STEP, build_output,
        )

    async def execute_save_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute checkpoint save on workers."""
        p: SaveStateData = request.payload
        checkpoint_path = p.checkpoint_path

        def build_output(result):
            actual_path = result.get("checkpoint_path", checkpoint_path)
            success = result.get("success", False)
            return [{
                "checkpoint_path": actual_path, "success": success,
                "execution_time": result.get("execution_time", 0.0),
                "message": "Checkpoint saved successfully" if success else "Save failed",
            }]

        return await self._execute_operation(
            request, "save_state",
            self.backend.save_state(
                checkpoint_path=p.checkpoint_path, save_optimizer=p.save_optimizer,
                use_timestamp=p.use_timestamp, model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SAVE_STATE, build_output,
        )

    async def execute_save_lora_only(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute LoRA-only checkpoint save on workers."""
        p: SaveLoraOnlyData = request.payload
        lora_path = p.lora_path
        if not lora_path:
            raise ValueError("lora_path is required")

        def build_output(result):
            actual_path = result.get("lora_path", lora_path)
            success = result.get("success", False)
            return [{
                "lora_path": actual_path, "success": success,
                "execution_time": result.get("execution_time", 0.0),
                "message": "LoRA adapter saved successfully (PEFT format)" if success else "Save failed",
            }]

        return await self._execute_operation(
            request, "save_lora_only",
            self.backend.save_lora_only(
                lora_path=p.lora_path, model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SAVE_LORA_ONLY, build_output,
        )

    async def execute_save_full_weights(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute full weights save as safetensors on workers."""
        p: SaveFullWeightsData = request.payload
        output_path = p.output_path
        if not output_path:
            raise ValueError("output_path is required")
        dtype = p.dtype

        def build_output(result):
            success = result.get("success", False)
            num_shards = result.get("num_shards", 1)
            return [{
                "output_path": output_path, "dtype": dtype,
                "num_shards": num_shards, "success": success,
                "execution_time": result.get("execution_time", 0.0),
                "message": f"Full weights saved as safetensors ({num_shards} shards)" if success else "Save failed",
            }]

        return await self._execute_operation(
            request, "save_full_weights",
            self.backend.save_full_weights(
                output_path=p.output_path, dtype=p.dtype,
                base_model_path=p.base_model_path, model_id=p.model_id,
                request_id=request.request_id,
            ),
            OutputType.SAVE_STATE, build_output,
        )

    async def execute_load_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute checkpoint load on workers."""
        p: LoadStateData = request.payload
        checkpoint_path = p.checkpoint_path
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        def build_output(result):
            success = result.get("success", False)
            return [{
                "checkpoint_path": checkpoint_path, "success": success,
                "execution_time": result.get("execution_time", 0.0),
                "message": "Checkpoint loaded successfully" if success else "Load failed",
            }]

        return await self._execute_operation(
            request, "load_state",
            self.backend.load_state(
                checkpoint_path=p.checkpoint_path, load_optimizer=p.load_optimizer,
                model_id=p.model_id, request_id=request.request_id,
            ),
            OutputType.LOAD_STATE, build_output,
        )

    async def execute_sleep(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute sleep operation (offload model and optimizer to CPU)."""
        def build_output(result):
            return [{"status": result.get("status", "sleeping"),
                     "offload_time": result.get("offload_time", 0.0),
                     "execution_time": result.get("execution_time", 0.0)}]

        return await self._execute_operation(
            request, "sleep",
            self.backend.sleep(request_id=request.request_id),
            OutputType.SLEEP, build_output,
        )

    async def execute_wake_up(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute wake_up operation (load model and optimizer to GPU)."""
        def build_output(result):
            return [{"status": result.get("status", "awake"),
                     "load_time": result.get("load_time", 0.0),
                     "execution_time": result.get("execution_time", 0.0)}]

        return await self._execute_operation(
            request, "wake_up",
            self.backend.wake_up(request_id=request.request_id),
            OutputType.WAKE_UP, build_output,
        )

    async def execute_sync_inference_weights(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute sync inference weights operation (NCCL transfer to inference endpoints)."""
        p: SyncWeightsData = request.payload
        if not p.endpoints:
            raise ValueError("inference endpoints must be provided")

        def build_output(result):
            return [{
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "transfer_time": result.get("transfer_time", 0.0),
                "total_bytes": result.get("total_bytes", 0),
                "num_parameters": result.get("num_parameters", 0),
                "num_buckets": result.get("num_buckets", 0),
                "endpoint_results": result.get("endpoint_results", []),
                "execution_time": result.get("execution_time", 0.0),
            }]

        return await self._execute_operation(
            request, "sync_inference_weights",
            self.backend.sync_inference_weights(
                endpoints=p.endpoints, master_address=p.master_address,
                master_port=p.master_port, group_name=p.group_name,
                buffer_size_mb=p.buffer_size_mb, sync_method=p.sync_method,
                flush_cache=p.flush_cache, pause_mode=p.pause_mode,
                weight_version=p.weight_version, quantization=p.quantization,
                request_id=request.request_id,
            ),
            OutputType.SYNC_INFERENCE_WEIGHTS, build_output,
        )

    async def execute_register_adapter(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute register adapter operation on workers."""
        p: RegisterAdapterData = request.payload

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request, "register_adapter",
            self.backend.register_adapter(
                model_id=p.model_id, lr=p.lr, request_id=request.request_id,
            ),
            OutputType.REGISTER_ADAPTER, build_output,
        )

    async def execute_save_adapter_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute save adapter state on workers."""
        p: AdapterStateData = request.payload

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request, "save_adapter_state",
            self.backend.save_adapter_state(
                model_id=p.model_id, path=p.path,
                save_optimizer=p.save_optimizer, request_id=request.request_id,
            ),
            OutputType.SAVE_ADAPTER_STATE, build_output,
        )

    async def execute_load_adapter_state(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute load adapter state on workers."""
        p: AdapterStateData = request.payload
        if not p.path:
            raise ValueError("adapter path is required for load_adapter_state")

        def build_output(result):
            return {"result": result}

        return await self._execute_operation(
            request, "load_adapter_state",
            self.backend.load_adapter_state(
                model_id=p.model_id, path=p.path,
                load_optimizer=p.load_optimizer, lr=p.lr,
                request_id=request.request_id,
            ),
            OutputType.LOAD_ADAPTER_STATE, build_output,
        )

    async def execute_get_adapter_info(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute get adapter info on workers."""
        def build_output(result):
            return [result]

        return await self._execute_operation(
            request, "get_adapter_info",
            self.backend.get_adapter_info(request_id=request.request_id),
            OutputType.GET_ADAPTER_INFO, build_output,
        )

    async def execute_kill_session(self, request: OrchestratorRequest) -> OrchestratorOutputs:
        """Execute kill session on workers (full-weights training only)."""
        p: KillSessionData = request.payload

        def build_output(result):
            return [{
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "checkpoint_path": result.get("checkpoint_path"),
                "execution_time": result.get("execution_time", 0.0),
            }]

        return await self._execute_operation(
            request, "kill_session",
            self.backend.kill_session(
                model_id=p.model_id, save_checkpoint=p.save_checkpoint,
                request_id=request.request_id,
            ),
            OutputType.KILL_SESSION, build_output,
        )

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "running": self.backend.is_ready(),
            "connected": self.backend.is_ready(),
            "ready": self.backend.is_ready(),
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": (self.successful_operations / self.total_operations if self.total_operations > 0 else 0.0),
        }

    def __repr__(self) -> str:
        return f"RequestProcessor(ready={self.backend.is_ready()}, operations={self.total_operations})"
