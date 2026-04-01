"""
OrchestratorClient - Async client for engine communication.

This is the OrchestratorClient component in the system design.

Architecture:
- INPUT SOCKET: ZMQ ROUTER (bind) - sends requests to engine
- OUTPUT SOCKET: ZMQ PULL (connect) - receives outputs from engine
- Background task: process_outputs_socket() - continuously reads from PULL socket
- Asyncio queue: output_queue - buffers outputs for async retrieval

Usage:
    client = OrchestratorClient(
        input_addr="tcp://127.0.0.1:5555",
        output_addr="tcp://127.0.0.1:5556"
    )
    await client.start()

    # Send request
    request = create_forward_backward_request(data)
    await client.send_request(request)

    # Get outputs
    output = await client.get_output(timeout=10.0)

    await client.stop()
"""

import asyncio
import logging
import time
from typing import Optional

import zmq.asyncio

from xorl.server.protocol.api_orchestrator import OrchestratorOutputs, OrchestratorRequest, RequestType
from xorl.server.protocol.operations import AbortData
from xorl.server.utils.zmq_channels import AsyncPullChannel, AsyncRouterChannel


logger = logging.getLogger(__name__)


class OrchestratorClient:
    """
    Async client for engine communication.

    Manages ROUTER (input) and PULL (output) sockets for bidirectional
    communication with the engine backend.
    """

    def __init__(
        self,
        input_addr: str,
        output_addr: str,
        output_queue_maxsize: int = 1000,
    ):
        """
        Initialize OrchestratorClient.

        Args:
            input_addr: ZMQ address for input ROUTER socket (e.g., "tcp://127.0.0.1:5555")
            output_addr: ZMQ address for output PULL socket (e.g., "tcp://127.0.0.1:5556")
            output_queue_maxsize: Maximum size of output queue
        """
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.output_queue_maxsize = output_queue_maxsize

        # ZMQ context and channels
        self.context: Optional[zmq.asyncio.Context] = None
        self.input_channel: Optional[AsyncRouterChannel] = None  # ROUTER for sending
        self.output_channel: Optional[AsyncPullChannel] = None  # PULL for receiving

        # Output queue and background task
        self.output_queue: Optional[asyncio.Queue] = None
        self.output_task: Optional[asyncio.Task] = None

        # Request tracking - map request_id to asyncio.Future for response
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._pending_requests_lock = asyncio.Lock()

        # State
        self._running = False
        self._request_count = 0

        # Connection health
        self._last_request_time = 0.0
        self._last_output_time = 0.0

        logger.info(f"OrchestratorClient initialized: input={input_addr}, output={output_addr}")

    async def start(self):
        """Start the client and connect sockets."""
        if self._running:
            logger.warning("OrchestratorClient already started")
            return

        logger.info("Starting OrchestratorClient...")

        # Initialize ZMQ context (shared by both channels)
        self.context = zmq.asyncio.Context()

        # Create INPUT channel (ROUTER for sending requests)
        self.input_channel = AsyncRouterChannel(self.input_addr, context=self.context)
        self.input_channel.bind()

        # Create OUTPUT channel (PULL for receiving outputs)
        self.output_channel = AsyncPullChannel(self.output_addr, context=self.context)
        self.output_channel.connect()

        # Create output queue
        self.output_queue = asyncio.Queue(maxsize=self.output_queue_maxsize)

        # Start background task for processing outputs
        self.output_task = asyncio.create_task(self.process_outputs_socket())

        self._running = True
        logger.info("OrchestratorClient started successfully")

    async def stop(self):
        """Stop the client and close sockets."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping OrchestratorClient...")

        # Cancel background task
        if self.output_task:
            self.output_task.cancel()
            try:
                await self.output_task
            except asyncio.CancelledError:
                pass

        # Close channels (sockets only, context owned by self)
        if self.input_channel:
            self.input_channel.close()
        if self.output_channel:
            self.output_channel.close()

        # Terminate shared context
        if self.context:
            self.context.term()

        logger.info("OrchestratorClient stopped")

    async def send_request(
        self,
        request: OrchestratorRequest,
        engine_identity: bytes = b"engine-0",
    ) -> asyncio.Future:
        """
        Send a request to the engine and return a Future for the response.

        Args:
            request: OrchestratorRequest to send
            engine_identity: Engine identity for ROUTER routing

        Returns:
            asyncio.Future that will be resolved with the OrchestratorOutputs response
        """
        if not self._running or not self.input_channel:
            raise RuntimeError("AsyncMPClient not started")

        # Create future for this request
        future = asyncio.get_event_loop().create_future()

        async with self._pending_requests_lock:
            self._pending_requests[request.request_id] = future

        # Serialize request
        request_bytes = request.to_msgpack()

        # Send via ROUTER channel
        try:
            await self.input_channel.send(engine_identity, request_bytes)

            self._request_count += 1
            self._last_request_time = time.time()

            logger.debug(
                f"Sent request {request.request_id} to engine {engine_identity} "
                f"(type={request.request_type.value}, size={len(request_bytes)} bytes)"
            )

            return future

        except zmq.error.ZMQError as e:
            # Remove from pending requests on error
            async with self._pending_requests_lock:
                self._pending_requests.pop(request.request_id, None)
            logger.error(f"ZMQ error sending request {request.request_id}: {e}")
            raise
        except Exception as e:
            # Remove from pending requests on error
            async with self._pending_requests_lock:
                self._pending_requests.pop(request.request_id, None)
            logger.error(f"Error sending request {request.request_id}: {e}")
            raise

    async def get_output(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[OrchestratorOutputs]:
        """
        Get an output from the output queue.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            OrchestratorOutputs or None if timeout
        """
        if not self._running or not self.output_channel:
            raise RuntimeError("AsyncMPClient not started")

        try:
            if timeout is not None:
                output = await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
            else:
                output = await self.output_queue.get()

            logger.debug(
                f"Retrieved output {output.request_id} from queue "
                f"(type={output.output_type.value}, finished={output.finished})"
            )

            return output

        except asyncio.TimeoutError:
            logger.debug(f"Timeout waiting for output (timeout={timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Error getting output: {e}")
            raise

    async def process_outputs_socket(self):
        """
        Background task to continuously read from OUTPUT socket.

        Reads outputs from PULL socket and resolves the corresponding request Future.
        """
        logger.info("Starting output socket processing task...")

        while self._running:
            try:
                # Poll for outputs with short timeout
                if not self.output_channel:
                    break

                # Check if data is available
                if not await self.output_channel.poll(timeout_ms=100):
                    continue

                # Receive output
                output_bytes = await self.output_channel.recv()

                # Deserialize
                output = OrchestratorOutputs.from_msgpack(output_bytes)

                self._last_output_time = time.time()

                logger.debug(
                    f"Received output {output.request_id} from PULL socket "
                    f"(type={output.output_type.value}, finished={output.finished})"
                )

                # Find and resolve the corresponding request Future
                async with self._pending_requests_lock:
                    future = self._pending_requests.pop(output.request_id, None)

                if future:
                    if not future.done():
                        future.set_result(output)
                        logger.debug(f"Resolved future for request {output.request_id}")
                    else:
                        logger.warning(f"Future for request {output.request_id} already done")
                else:
                    logger.warning(
                        f"Received output for unknown request {output.request_id} (may have timed out or was cancelled)"
                    )
                    # Still put in output queue as fallback
                    await self.output_queue.put(output)

            except asyncio.CancelledError:
                logger.info("Output processing task cancelled")
                break
            except zmq.error.ZMQError as e:
                logger.error(f"ZMQ error in output processing: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in output processing: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Output socket processing task stopped")

    async def cancel_request(self, request_id: str, send_abort: bool = True) -> bool:
        """
        Cancel a pending request.

        This removes the request from _pending_requests and optionally sends
        an ABORT request to the engine.

        Args:
            request_id: ID of the request to cancel
            send_abort: Whether to send an ABORT request to the engine (default: True)

        Returns:
            True if the request was found and cancelled, False otherwise
        """
        async with self._pending_requests_lock:
            future = self._pending_requests.pop(request_id, None)

        if future is None:
            logger.debug(f"Request {request_id} not found in pending requests (already completed?)")
            return False

        # Cancel the future if not already done
        if not future.done():
            future.cancel()
            logger.info(f"Cancelled pending request {request_id}")

        # Optionally send ABORT to engine to stop processing
        if send_abort:
            try:
                abort_request = OrchestratorRequest(
                    request_type=RequestType.ABORT,
                    operation="abort",
                    payload=AbortData(target_request_id=request_id),
                )
                await self.input_channel.send(b"engine-0", abort_request.to_msgpack())
                logger.info(f"Sent ABORT request for {request_id} to engine")
            except Exception as e:
                logger.warning(f"Failed to send ABORT for {request_id}: {e}")

        return True

    async def clear_pending_requests(self):
        """
        Clear all pending requests (e.g., when client session resets).

        Cancels all pending futures without sending aborts (since the engine
        scheduler will handle stale requests).
        """
        async with self._pending_requests_lock:
            count = len(self._pending_requests)
            for request_id, future in self._pending_requests.items():
                if not future.done():
                    future.cancel()
            self._pending_requests.clear()

        if count > 0:
            logger.info(f"Cleared {count} pending requests from engine client")

    def get_stats(self) -> dict:
        """
        Get client statistics.

        Returns:
            Dictionary with client stats
        """
        return {
            "running": self._running,
            "input_addr": self.input_addr,
            "output_addr": self.output_addr,
            "request_count": self._request_count,
            "pending_requests": len(self._pending_requests),
            "last_request_time": self._last_request_time,
            "last_output_time": self._last_output_time,
            "output_queue_size": self.output_queue.qsize() if self.output_queue else 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
