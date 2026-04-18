"""
Rank 0 ZMQ Protocol Handler.

Manages the ROUTER channel lifecycle, handshake, request recv/ACK/enqueue,
response send, heartbeats, and shutdown broadcast for rank 0 of the
RunnerDispatcher.

This separates ZMQ protocol orchestration from compute logic, so
RunnerDispatcher can focus on model operations.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import torch.distributed as dist
import zmq

from xorl.server.protocol.orchestrator_runner import (
    RunnerAck,
    RunnerDispatchCommand,
    RunnerReady,
    RunnerResponse,
    deserialize_message,
    serialize_message,
)
from xorl.server.utils.network import get_local_ip, parse_zmq_address, write_address_file
from xorl.server.utils.zmq_channels import AsyncRouterChannel


logger = logging.getLogger(__name__)


class Rank0Protocol:
    """ZMQ ROUTER protocol handler for rank 0.

    Manages: channel lifecycle, address file discovery, handshake,
    request recv/ACK/enqueue, response send, heartbeats, shutdown broadcast.

    The protocol delegates actual request processing to a callback:
        request_handler(request: RunnerDispatchCommand) -> RunnerResponse
    """

    def __init__(
        self,
        bind_address: str,
        output_dir: str,
        rank: int,
        world_size: int,
        device: str,
        cpu_group: Optional[Any],
        request_handler: Callable[[RunnerDispatchCommand], Awaitable[RunnerResponse]],
        context: Any,
    ):
        self.bind_address = bind_address
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.cpu_group = cpu_group
        self._request_handler = request_handler
        self.context = context

        # Channel
        self.channel: Optional[AsyncRouterChannel] = None
        self.current_client_id: Optional[bytes] = None

        # Request queue (receiver loop → event loop)
        self.request_queue: asyncio.Queue = asyncio.Queue()

        # State
        self._running = False
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def running(self) -> bool:
        return self._running

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def run(self) -> None:
        """Main entry point — bind channel, run receiver + event loop, cleanup.

        Blocks until shutdown is received or stop() is called.
        """
        logger.info(f"Rank {self.rank}: Starting connection loop, ready for executor connections...")
        self._running = True

        # Bind ROUTER channel
        try:
            self.channel = AsyncRouterChannel(self.bind_address, context=self.context)
            self.channel.bind()
        except zmq.ZMQError as e:
            logger.error(f"Rank {self.rank}: Failed to bind channel: {e}")
            raise

        # Write address file for Engine discovery (multi-node support)
        try:
            bind_host, bind_port = parse_zmq_address(self.bind_address)
            if bind_host == "0.0.0.0":
                connect_host = get_local_ip()
            else:
                connect_host = bind_host
            connect_address = f"tcp://{connect_host}:{bind_port}"
            address_file_path = write_address_file(connect_address, self.output_dir)
            logger.info(f"Rank {self.rank}: Wrote connect address {connect_address} to {address_file_path}")
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Could not write address file (OK for single-node): {e}")

        self.current_client_id = None

        # Start event loop task (processes requests from queue)
        event_loop_task = asyncio.create_task(self._event_loop())
        logger.info(f"Rank {self.rank}: Event loop task started")

        try:
            # Run receiver loop (blocks until shutdown)
            await self._receiver_loop()
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error in connection loop: {e}", exc_info=True)
        finally:
            # Cancel event loop task
            if not event_loop_task.done():
                event_loop_task.cancel()
                try:
                    await event_loop_task
                except asyncio.CancelledError:
                    pass

            # Clean up channel
            if self.channel:
                self.channel.close()
                self.channel = None
                logger.debug(f"Rank {self.rank}: ROUTER channel closed")

        logger.info(f"Rank {self.rank}: Connection loop stopped")

    async def stop(self) -> None:
        """Signal shutdown."""
        self._running = False

    # ========================================================================
    # Handshake
    # ========================================================================

    async def _send_ready(self, client_id: bytes):
        """Send RunnerReady message to executor."""
        logger.debug(f"Rank {self.rank}: Sending RunnerReady message to client {client_id.hex()[:8]}...")

        ready_msg = RunnerReady(worker_rank=self.rank, world_size=self.world_size, device=self.device)
        await self.channel.send(client_id, serialize_message(ready_msg))
        logger.debug(
            f"Rank {self.rank}: Sent RunnerReady: rank={self.rank}, world_size={self.world_size}, device={self.device}"
        )

        # Wait for acknowledgement (but might receive request first)
        try:
            recv_client_id, msg_bytes = await self.channel.recv()
        except RuntimeError:
            logger.warning(f"Rank {self.rank}: Unexpected frame count in handshake recv")
            return

        ack = deserialize_message(msg_bytes)

        if isinstance(ack, RunnerAck):
            logger.debug(f"Rank {self.rank}: Received ACK for ready: {ack.request_id}")
        elif isinstance(ack, RunnerDispatchCommand):
            # Received request before ACK - this is OK, send ACK immediately
            logger.debug(
                f"Rank {self.rank}: Received RunnerDispatchCommand before ACK - sending ACK and queuing for processing"
            )
            request_ack = RunnerAck(request_id=ack.message_id)
            await self.channel.send(recv_client_id, serialize_message(request_ack))
            logger.debug(f"Rank {self.rank}: Sent ACK for request {ack.message_id[:8]}")
            self._request_count += 1
            await self.request_queue.put((recv_client_id, ack))
        else:
            logger.warning(f"Rank {self.rank}: Expected ACK or RunnerDispatchCommand, got {type(ack).__name__}")

    # ========================================================================
    # Receiver Loop
    # ========================================================================

    async def _receiver_loop(self):
        """Receive requests from executor and enqueue them.

        Works with ROUTER socket — handles multipart messages with client identity.
        """
        logger.info(f"Rank {self.rank}: Starting receiver loop...")

        while self._running:
            try:
                if not await self.channel.poll(timeout_ms=100):
                    continue

                try:
                    client_id, request_bytes = await self.channel.recv()
                except RuntimeError:
                    logger.warning(f"Rank {self.rank}: Unexpected frame count in receiver loop")
                    continue

                # New client connection
                if self.current_client_id is None or client_id != self.current_client_id:
                    logger.info(f"Rank {self.rank}: New client connected: {client_id.hex()[:8]}")
                    self.current_client_id = client_id
                    await self._send_ready(client_id)
                    continue

                # Empty message means connection closed
                if not request_bytes:
                    logger.info(f"Rank {self.rank}: Received empty message from client {client_id.hex()[:8]}")
                    self.current_client_id = None
                    continue

                request = deserialize_message(request_bytes)

                if not isinstance(request, RunnerDispatchCommand):
                    logger.warning(f"Rank {self.rank}: Received non-request message: {type(request).__name__}")
                    continue

                self._request_count += 1

                logger.debug(f"Rank {self.rank}: Received request #{self._request_count}: {request.operation}")
                logger.debug(f"Rank {self.rank}: Request payload type: {type(request.payload).__name__}")

                # Send immediate ACK
                ack = RunnerAck(request_id=request.message_id)
                try:
                    await self.channel.send(client_id, serialize_message(ack))
                    logger.debug(f"Rank {self.rank}: Sent ACK for request {request.message_id[:8]}")
                except zmq.ZMQError as e:
                    logger.warning(f"Rank {self.rank}: Failed to send ACK, connection lost: {e}")
                    self.current_client_id = None
                    continue

                # Enqueue request for processing
                await self.request_queue.put((client_id, request))
                logger.debug(f"Rank {self.rank}: Enqueued request {request.message_id[:8]} to queue")

            except asyncio.CancelledError:
                logger.info(f"Rank {self.rank}: Receiver loop cancelled")
                break
            except zmq.ZMQError as e:
                logger.info(f"Rank {self.rank}: ZMQ error in receiver loop: {e}")
                self.current_client_id = None
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in receiver loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info(f"Rank {self.rank}: Receiver loop stopped")

    # ========================================================================
    # Event Loop
    # ========================================================================

    async def _event_loop(self):
        """Process requests from queue, send heartbeats, dispatch to handler, send responses."""
        logger.info(f"Rank {self.rank}: Starting rank 0 event loop...")

        heartbeat_interval = 5.0
        last_heartbeat_time = time.time()

        while self._running:
            try:
                # Get (client_id, request) from queue (non-blocking)
                try:
                    queue_item = self.request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Send heartbeat to workers if needed
                    current_time = time.time()
                    if self.world_size > 1 and (current_time - last_heartbeat_time) >= heartbeat_interval:
                        heartbeat_obj = [{"command": "heartbeat"}]
                        dist.broadcast_object_list(heartbeat_obj, src=0, group=self.cpu_group)
                        last_heartbeat_time = current_time
                        logger.debug(f"Rank {self.rank}: Sent heartbeat to workers")

                    await asyncio.sleep(0.01)
                    continue

                client_id, request = queue_item
                last_heartbeat_time = time.time()

                logger.debug(f"Rank {self.rank}: Processing request from queue: {request.operation}")

                # Handle shutdown specially
                if request.operation == "shutdown":
                    logger.info(f"Rank {self.rank}: Received SHUTDOWN request")

                    if self.world_size > 1:
                        command_obj = [{"command": "shutdown"}]
                        dist.broadcast_object_list(command_obj, src=0, group=self.cpu_group)

                    response = RunnerResponse(
                        request_id=request.message_id,
                        success=True,
                        result={"status": "shutting_down"},
                        execution_time=0.0,
                    )
                    await self.channel.send(client_id, serialize_message(response))
                    self._running = False
                    break

                # Dispatch to worker's request handler
                response = await self._request_handler(request)
                await self.channel.send(client_id, serialize_message(response))

                if response.success:
                    self._success_count += 1
                    logger.debug(f"Rank {self.rank}: Sent successful response for request {request.message_id[:8]}")
                else:
                    self._failure_count += 1
                    logger.warning(
                        f"Rank {self.rank}: Sent failure response for request {request.message_id[:8]}: {response.error}"
                    )

            except asyncio.CancelledError:
                logger.info(f"Rank {self.rank}: Event loop cancelled")
                break
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error in event loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info(f"Rank {self.rank}: Event loop stopped")
