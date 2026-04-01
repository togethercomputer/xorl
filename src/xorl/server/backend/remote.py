"""
Remote Backend - ZMQ DEALER-ROUTER communication with RunnerDispatcher.

Extracted from Executor: handles ZMQ socket setup, handshake, and the
request-ACK-response protocol for communicating with worker rank 0.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from xorl.server.backend.base import Backend
from xorl.server.protocol.operations import (
    LOAD_STATE_TIMEOUT,
    SAVE_STATE_TIMEOUT,
    AdapterStateData,
    EmptyData,
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
from xorl.server.protocol.orchestrator_runner import (
    RunnerAck,
    RunnerDispatchCommand,
    RunnerReady,
    RunnerResponse,
    deserialize_message,
    serialize_message,
)
from xorl.server.utils.zmq_channels import AsyncDealerChannel


logger = logging.getLogger(__name__)


class RemoteBackend(Backend):
    """Backend that communicates with RunnerDispatcher via ZMQ DEALER-ROUTER."""

    def __init__(
        self,
        worker_address: str = "tcp://127.0.0.1:5556",
        operation_timeout: float = 3600.0,
        connection_timeout: float = 120.0,
        ack_timeout: float = 10.0,
    ):
        self.worker_address = worker_address
        self.operation_timeout = operation_timeout
        self.connection_timeout = connection_timeout
        self.ack_timeout = ack_timeout

        # ZMQ channel
        self.channel: Optional[AsyncDealerChannel] = None

        # State
        self._running = False
        self._connected = False
        self._ready = False

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def start(self) -> None:
        """Connect to worker and perform handshake."""
        if self._running:
            logger.warning("RemoteBackend already started")
            return

        logger.info(f"Starting RemoteBackend (worker={self.worker_address})...")

        self.channel = AsyncDealerChannel(self.worker_address, identity=b"executor-0")
        try:
            self.channel.connect()
            self._connected = True
            logger.info(f"Connected to rank 0 worker at {self.worker_address}")
        except Exception as e:
            logger.error(f"Failed to connect to rank 0 worker: {e}")
            raise

        self._running = True

        # Perform handshake
        ready = await self._wait_for_worker_ready()
        if not ready:
            raise RuntimeError("Worker rank 0 failed to become ready")

        logger.info("RemoteBackend started successfully")

    async def stop(self) -> None:
        """Close ZMQ socket and context."""
        if not self._running:
            return

        logger.info("Stopping RemoteBackend...")
        self._running = False
        self._connected = False
        self._ready = False

        if self.channel:
            self.channel.close()
            self.channel = None

        logger.info("RemoteBackend stopped")

    def is_ready(self) -> bool:
        return self._running and self._connected and self._ready

    # ========================================================================
    # Handshake
    # ========================================================================

    async def _wait_for_worker_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait for worker rank 0 to be ready (RunnerReady handshake)."""
        if not self._running or not self.channel:
            raise RuntimeError("RemoteBackend not started")

        total_timeout = timeout or self.connection_timeout
        logger.info(f"Waiting for rank 0 ready (timeout={total_timeout}s)...")

        try:
            # Send connection request
            connect_request = RunnerDispatchCommand.create("health_check", EmptyData(), request_id="connect-handshake")
            await self.channel.send(serialize_message(connect_request))

            # Wait for RunnerReady
            ready_bytes = await self.channel.recv(timeout=total_timeout)
            ready_msg = deserialize_message(ready_bytes)

            if not isinstance(ready_msg, RunnerReady):
                logger.warning(f"Expected RunnerReady, got {type(ready_msg).__name__}")
                return False

            logger.info(
                f"Rank 0 ready: rank={ready_msg.worker_rank}, "
                f"world_size={ready_msg.world_size}, device={ready_msg.device}"
            )

            # Send acknowledgement
            ack = RunnerAck(request_id=ready_msg.message_id)
            await self.channel.send(serialize_message(ack))

            self._ready = True
            return True

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for rank 0 ready after {total_timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error waiting for rank 0 ready: {e}", exc_info=True)
            return False

    # ========================================================================
    # ZMQ Communication
    # ========================================================================

    async def _send_and_receive(
        self,
        request: RunnerDispatchCommand,
        timeout: Optional[float] = None,
    ) -> RunnerResponse:
        """Send request, wait for ACK, then wait for response."""
        if not self._connected or not self.channel:
            raise RuntimeError("Not connected to rank 0 worker")

        # Send request
        logger.debug(f"Sending request {request.message_id} (operation={request.operation})")
        await self.channel.send(serialize_message(request))

        # Wait for ACK
        try:
            ack_bytes = await self.channel.recv(timeout=self.ack_timeout)
            ack = deserialize_message(ack_bytes)
            if not isinstance(ack, RunnerAck):
                raise RuntimeError(f"Expected RunnerAck, got {type(ack).__name__}")
            if ack.request_id != request.message_id:
                raise RuntimeError(f"ACK request_id mismatch: expected {request.message_id}, got {ack.request_id}")
            logger.debug(f"Received ACK for request {request.message_id}")
        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout waiting for ACK (request {request.message_id})")

        # Wait for response
        response_timeout = timeout or self.operation_timeout
        try:
            response_bytes = await self.channel.recv(timeout=response_timeout)
            response = deserialize_message(response_bytes)
            if not isinstance(response, RunnerResponse):
                raise RuntimeError(f"Expected RunnerResponse, got {type(response).__name__}")
            if response.request_id != request.message_id:
                raise RuntimeError(
                    f"Response request_id mismatch: expected {request.message_id}, got {response.request_id}"
                )
            logger.debug(
                f"Received response for request {request.message_id}: "
                f"success={response.success}, time={response.execution_time:.3f}s"
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timeout waiting for response (request {request.message_id}, timeout={response_timeout}s)"
            )

        if not response.success:
            raise RuntimeError(f"Operation failed: {response.error or 'Unknown error'}")

        return response

    # ========================================================================
    # Backend Operations
    # ========================================================================

    async def _execute(
        self, operation: str, payload, request_id: Optional[str] = None, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generic execute: create request, send, return result dict."""
        request = RunnerDispatchCommand.create(operation, payload, request_id=request_id)
        response = await self._send_and_receive(request, timeout=timeout)
        result = response.result
        if response.execution_time is not None:
            result["execution_time"] = response.execution_time
        return result

    async def forward_backward(
        self, batches, loss_fn="causallm_loss", loss_fn_params=None, model_id=None, routed_experts=None, request_id=None
    ):
        return await self._execute(
            "forward_backward",
            ModelPassData(
                batches=batches,
                loss_fn=loss_fn,
                loss_fn_params=loss_fn_params,
                model_id=model_id,
                routed_experts=routed_experts,
            ),
            request_id=request_id,
        )

    async def forward(self, batches, loss_fn="causallm_loss", loss_fn_params=None, model_id=None, request_id=None):
        return await self._execute(
            "forward",
            ModelPassData(
                batches=batches,
                loss_fn=loss_fn,
                loss_fn_params=loss_fn_params,
                model_id=model_id,
            ),
            request_id=request_id,
        )

    async def optim_step(
        self, lr, gradient_clip=None, beta1=None, beta2=None, eps=None, model_id=None, request_id=None
    ):
        return await self._execute(
            "optim_step",
            OptimStepData(
                lr=lr,
                gradient_clip=gradient_clip,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                model_id=model_id,
            ),
            request_id=request_id,
        )

    async def save_state(
        self, checkpoint_path=None, save_optimizer=True, use_timestamp=False, model_id=None, request_id=None
    ):
        return await self._execute(
            "save_state",
            SaveStateData(
                checkpoint_path=checkpoint_path,
                save_optimizer=save_optimizer,
                use_timestamp=use_timestamp,
                model_id=model_id,
            ),
            request_id=request_id,
            timeout=SAVE_STATE_TIMEOUT,
        )

    async def load_state(self, checkpoint_path=None, load_optimizer=True, model_id=None, request_id=None):
        return await self._execute(
            "load_state",
            LoadStateData(
                checkpoint_path=checkpoint_path,
                load_optimizer=load_optimizer,
                model_id=model_id,
            ),
            request_id=request_id,
            timeout=LOAD_STATE_TIMEOUT,
        )

    async def save_lora_only(self, lora_path=None, model_id=None, request_id=None):
        return await self._execute(
            "save_lora_only",
            SaveLoraOnlyData(
                lora_path=lora_path,
                model_id=model_id,
            ),
            request_id=request_id,
            timeout=SAVE_STATE_TIMEOUT,
        )

    async def save_full_weights(
        self, output_path=None, dtype="bfloat16", base_model_path=None, model_id=None, request_id=None
    ):
        return await self._execute(
            "save_full_weights",
            SaveFullWeightsData(
                output_path=output_path,
                dtype=dtype,
                base_model_path=base_model_path,
                model_id=model_id,
            ),
            request_id=request_id,
            timeout=SAVE_STATE_TIMEOUT,
        )

    async def sleep(self, request_id=None):
        return await self._execute("sleep", EmptyData(), request_id=request_id, timeout=30.0)

    async def wake_up(self, request_id=None):
        return await self._execute("wake_up", EmptyData(), request_id=request_id, timeout=30.0)

    async def sync_inference_weights(
        self,
        endpoints,
        master_address="localhost",
        master_port=29600,
        group_name="weight_sync_group",
        buffer_size_mb=1024,
        sync_method="nccl_broadcast",
        flush_cache=False,
        pause_mode="retract",
        weight_version=None,
        quantization=None,
        request_id=None,
    ):
        return await self._execute(
            "sync_inference_weights",
            SyncWeightsData(
                endpoints=endpoints,
                master_address=master_address,
                master_port=master_port,
                group_name=group_name,
                buffer_size_mb=buffer_size_mb,
                sync_method=sync_method,
                flush_cache=flush_cache,
                pause_mode=pause_mode,
                weight_version=weight_version,
                quantization=quantization,
            ),
            request_id=request_id,
            timeout=600.0,
        )

    async def register_adapter(self, model_id="default", lr=1e-5, request_id=None):
        return await self._execute(
            "register_adapter",
            RegisterAdapterData(
                model_id=model_id,
                lr=lr,
            ),
            request_id=request_id,
            timeout=60.0,
        )

    async def save_adapter_state(self, model_id="default", path=None, save_optimizer=True, request_id=None):
        return await self._execute(
            "save_adapter_state",
            AdapterStateData(
                model_id=model_id,
                path=path,
                save_optimizer=save_optimizer,
            ),
            request_id=request_id,
            timeout=600.0,
        )

    async def load_adapter_state(self, model_id="default", path=None, load_optimizer=True, lr=None, request_id=None):
        return await self._execute(
            "load_adapter_state",
            AdapterStateData(
                model_id=model_id,
                path=path,
                load_optimizer=load_optimizer,
                lr=lr,
            ),
            request_id=request_id,
            timeout=600.0,
        )

    async def get_adapter_info(self, request_id=None):
        return await self._execute("get_adapter_info", EmptyData(), request_id=request_id, timeout=60.0)

    async def kill_session(self, model_id="default", save_checkpoint=True, request_id=None):
        return await self._execute(
            "kill_session",
            KillSessionData(
                model_id=model_id,
                save_checkpoint=save_checkpoint,
            ),
            request_id=request_id,
            timeout=120.0,
        )

    async def health_check(self, request_id=None):
        return await self._execute("health_check", EmptyData(), request_id=request_id, timeout=5.0)
