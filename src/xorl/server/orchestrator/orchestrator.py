"""
Orchestrator - Unified Engine with Scheduler and RequestProcessor.

This module provides the main Orchestrator class that orchestrates request handling,
scheduling, and distributed execution for the training engine.

Components:
===========
1. **Orchestrator**: Main coordinator (queue management, threading, lifecycle)
2. **Scheduler**: Request ordering and lifecycle tracking (scheduler.py)
3. **RequestProcessor**: Worker management and distributed execution (request_processor.py)

Architecture Overview:
=====================

┌───────────────────────────────────────────────────────────────────────────┐
│                              API Server                                   │
│                         (External Component)                              │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │ OrchestratorRequest (msgpack)
                                  ↓
┌───────────────────────────────────────────────────────────────────────────┐
│                            Orchestrator                                     │
│  Main Thread: event_loop()                                                │
│  Background Threads:                                                      │
│    - input_thread:  process_input_sockets()                               │
│    - output_thread: process_output_sockets()                              │
│    - worker_thread: request_processor asyncio event loop                           │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │               API Server Communication                              │  │
│  │                                                                     │  │
│  │  INPUT Socket (DEALER connect)    OUTPUT Socket (PUSH bind)         │  │
│  │       ↓                                  ↑                          │  │
│  │  input_queue (Queue)             output_queue (Queue)               │  │
│  │       ↓                                  ↑                          │  │
│  │  [OrchestratorRequest]             [OrchestratorOutputs]                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        Scheduler                                    │  │
│  │  (Request ordering and lifecycle tracking)                          │  │
│  │                                                                     │  │
│  │  Policy: FIFOPolicy (extensible)                                    │  │
│  │  ├─ add_request()        → enqueue                                  │  │
│  │  ├─ get_next_request()   → dequeue (pending → processing)           │  │
│  │  ├─ mark_completed()     → processing → completed                   │  │
│  │  ├─ mark_failed()        → processing → failed                      │  │
│  │  └─ abort_request()      → remove from queue                        │  │
│  │                                                                     │  │
│  │  Tracks: arrival_time, processing_time, retries, status             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         RequestProcessor                                    │  │
│  │  (Rank 0 worker communication and distributed execution)            │  │
│  │                                                                     │  │
│  │  PAIR Socket (connect) ←→ Worker Rank 0 (PAIR bind)                 │  │
│  │                                                                       │ │
│  │  Connection Management:                                              │ │
│  │  ├─ wait_for_worker0_ready()     → handshake with rank 0            │ │
│  │  ├─ check_connection()           → health check                     │ │
│  │  └─ ACK mechanism                → reliable delivery                │ │
│  │                                                                       │ │
│  │  Operation Execution:                                                │ │
│  │  ├─ execute_forward_backward()   → send to rank 0 → receive         │ │
│  │  ├─ execute_optim_step()         → send to rank 0 → receive         │ │
│  │  ├─ execute_save_state()         → send to rank 0 → receive         │ │
│  │  └─ execute_load_state()         → send to rank 0 → receive         │ │
│  │                                                                       │ │
│  │  Protocol: Request → ACK → Response (rank 0 coordinates via NCCL)   │ │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓ RunnerDispatchCommand (JSON/msgpack)
┌───────────────────────────────────────────────────────────────────────────┐
│                    Distributed Workers (Rank 0 PAIR)                       │
│  worker-0 (rank 0) ← PAIR ─ RequestProcessor                                       │
│  worker-1, ..., worker-N ← NCCL ─ Rank 0                                   │
│  Each: ModelRunner + PyTorch Distributed (NCCL)                           │
└───────────────────────────────────────────────────────────────────────────┘

Message Flow:
============

1. **Request Ingestion:**
   ```
   API Server → input_socket (DEALER) → input_queue → Scheduler.add_request()
   ```

2. **Scheduling:**
   ```
   Scheduler.get_next_request() → [pending → processing]
   ```

3. **Execution:**
   ```
   Orchestrator.event_loop()
     → _execute_*()
     → RequestProcessor.execute_*() (async in worker_thread)
       → send RunnerDispatchCommand to rank 0 (PAIR)
       → receive RunnerAck (PAIR)
       → receive RunnerResponse (PAIR)
       → rank 0 coordinates with other workers via NCCL
     → return OrchestratorOutputs
   ```

4. **Response:**
   ```
   output_queue → output_socket (PUSH) → API Server
   Scheduler.mark_completed() [processing → completed]
   ```

Threading Model:
===============

```
Main Thread (event_loop):
  while running:
    ├─ _process_input_queue()      # input_queue → Scheduler
    ├─ _process_engine_step()      # Scheduler → RequestProcessor → output_queue
    │    ├─ scheduled_req = scheduler.get_next_request()
    │    ├─ output = _call_async(request_processor.execute_*())
    │    ├─ _send_output(output)
    │    └─ scheduler.mark_completed()
    └─ sleep(0.01)

Background Threads:
  ├─ input_thread:  input_socket → input_queue (blocking zmq.poll)
  ├─ output_thread: output_queue → output_socket (blocking queue.get)
  └─ worker_thread: asyncio event loop for request processor (async ZMQ operations)
```

Component Responsibilities:
==========================

**Orchestrator:**
- API socket I/O (DEALER input, PUSH output)
- Queue management (input_queue, output_queue)
- Threading coordination
- Component orchestration (Scheduler + RequestProcessor)
- Request routing to operations

**Scheduler (scheduler.py):**
- Request ordering (FIFO, future: Priority, Batch)
- Lifecycle tracking (pending → processing → completed/failed)
- Abort/cancel support
- Request metadata (arrival_time, processing_time)
- Statistics (total, completed, failed, aborted)

**RequestProcessor (request_processor.py):**
- Worker discovery and health monitoring
- Message broadcasting (ROUTER pattern)
- Response collection (with policies)
- Result aggregation (loss, gradients, etc.)
- Distributed operation coordination

Key Design Principles:
=====================
1. **Separation of Concerns**: Each component has a single, clear responsibility
2. **Thread Safety**: Queues for inter-thread communication
3. **Async Workers**: Request processor uses asyncio for efficient ZMQ operations
4. **Extensibility**: Pluggable scheduling policies, response policies
5. **Observability**: Comprehensive statistics at each layer
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

import zmq

from xorl.server.backend import Backend, RemoteBackend
from xorl.server.orchestrator.request_processor import RequestProcessor
from xorl.server.orchestrator.scheduler import Scheduler, SeqIdAwareFIFOPolicy
from xorl.server.protocol.api_orchestrator import (
    OrchestratorOutputs,
    OrchestratorRequest,
    OutputType,
    RequestType,
)
from xorl.server.utils.zmq_channels import SyncDealerChannel, SyncPushChannel


logger = logging.getLogger(__name__)


# ============================================================================
# Main Orchestrator Class
# ============================================================================


class Orchestrator:
    """
    Unified Engine Core with integrated worker scheduling.

    Handles both API server communication and worker management in a single component.
    """

    def __init__(
        self,
        # API Server communication addresses
        input_addr: str,  # DEALER socket to receive from API server
        output_addr: str,  # PUSH socket to send to API server
        engine_identity: bytes = b"engine-0",
        # Worker communication addresses
        rank0_worker_address: str = "tcp://127.0.0.1:5556",
        # Configuration
        num_workers: int = 16,
        operation_timeout: float = 120.0,
        connection_timeout: float = 10.0,
        ack_timeout: float = 30.0,  # 30 seconds
        sample_packing_sequence_len: int = 32000,
        enable_packing: bool = True,
        pad_to_multiple_of: int = 128,
        cp_size: int = 1,
        input_queue_maxsize: int = 1000,
        output_queue_maxsize: int = 1000,
        train_config: Optional[Dict[str, Any]] = None,
        backend: Optional["Backend"] = None,
    ):
        """
        Initialize Orchestrator.

        Args:
            input_addr: ZMQ address for input DEALER socket (API server requests)
            output_addr: ZMQ address for output PUSH socket (API server responses)
            engine_identity: Identity for DEALER socket
            rank0_worker_address: ZMQ address of rank 0 worker (PAIR socket)
            num_workers: Number of workers for distributed execution
            operation_timeout: Timeout for operation completion (seconds)
            connection_timeout: Timeout for initial connection (seconds)
            ack_timeout: Timeout for ACK receipt (seconds)
            sample_packing_sequence_len: Maximum sequence length for packing (default: 32000)
            enable_packing: Enable sample packing (default: True)
            pad_to_multiple_of: Base padding alignment (default: 128)
            cp_size: Sequence parallel size for Ulysses SP (default: 1)
            input_queue_maxsize: Maximum size of input queue
            output_queue_maxsize: Maximum size of output queue
            train_config: Training configuration for data processing
            backend: Optional Backend instance. If None, creates a RemoteBackend.
        """
        # API Server communication
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.engine_identity = engine_identity

        # Worker communication
        self.rank0_worker_address = rank0_worker_address
        self.num_workers = num_workers
        self.train_config = train_config or {}

        # ZMQ channels for API server communication
        self.input_channel: Optional[SyncDealerChannel] = None  # DEALER for API server
        self.output_channel: Optional[SyncPushChannel] = None  # PUSH for API server

        # Create backend if not provided
        if backend is None:
            backend = RemoteBackend(
                worker_address=rank0_worker_address,
                operation_timeout=operation_timeout,
                connection_timeout=connection_timeout,
                ack_timeout=ack_timeout,
            )

        # RequestProcessor for worker management and distributed execution
        self.request_processor = RequestProcessor(
            backend=backend,
            sample_packing_sequence_len=sample_packing_sequence_len,
            enable_packing=enable_packing,
            pad_to_multiple_of=pad_to_multiple_of,
            cp_size=cp_size,
        )

        # Threading queues
        self.input_queue = queue.Queue(maxsize=input_queue_maxsize)
        self.output_queue = queue.Queue(maxsize=output_queue_maxsize)

        # Scheduler for request management
        # Use SeqIdAwareFIFOPolicy to enforce request ordering via seq_id
        # This ensures forward_backward completes before optim_step even if
        # requests arrive out of network order
        self.scheduler = Scheduler(policy=SeqIdAwareFIFOPolicy())

        # Background threads
        self.input_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None
        self.event_loop_thread: Optional[threading.Thread] = None

        # Asyncio loop for worker communication
        self.worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self.worker_thread: Optional[threading.Thread] = None

        # State
        self._running = False
        self._stop_event = threading.Event()

        logger.info(
            f"Orchestrator initialized: "
            f"input={input_addr}, output={output_addr}, "
            f"rank0_worker={rank0_worker_address}, num_workers={num_workers}"
        )

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    def start(self):
        """Start the engine core process."""
        if self._running:
            logger.warning("Orchestrator already started")
            return

        logger.info("Starting Orchestrator...")

        self._running = True
        self._stop_event.clear()

        # Start worker communication event loop thread
        self.worker_loop = asyncio.new_event_loop()
        self.worker_thread = threading.Thread(target=self._run_worker_loop, name="WorkerLoopThread", daemon=True)
        self.worker_thread.start()

        # Wait for worker loop to be ready
        time.sleep(0.2)

        # Initialize request processor (async) — backend.start() handles handshake
        logger.info("Starting request processor (backend handshake)...")
        future = asyncio.run_coroutine_threadsafe(self.request_processor.start(), self.worker_loop)
        future.result()  # Wait for completion
        logger.info("Request processor started, backend ready")

        # Start API server socket I/O threads
        self.input_thread = threading.Thread(target=self.process_input_sockets, name="EngineInputThread", daemon=True)
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, name="EngineOutputThread", daemon=True
        )

        self.input_thread.start()
        self.output_thread.start()

        # Give socket threads time to initialize
        time.sleep(0.2)

        # Start main event loop thread
        self.event_loop_thread = threading.Thread(target=self.event_loop, name="EngineEventLoopThread", daemon=True)
        self.event_loop_thread.start()

        logger.info("Orchestrator started successfully")

    def stop(self):
        """Stop the engine core process."""
        if not self._running:
            return

        logger.info("Stopping Orchestrator...")

        self._running = False
        self._stop_event.set()

        # Stop request processor
        if self.worker_loop:
            future = asyncio.run_coroutine_threadsafe(self.request_processor.stop(), self.worker_loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Error stopping request processor: {e}")

            self.worker_loop.call_soon_threadsafe(self.worker_loop.stop)

        # Drain output queue to unblock the output thread
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for threads to finish before closing sockets
        for thread in [self.event_loop_thread, self.input_thread, self.output_thread, self.worker_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)

        # Close input channel (output channel closed by its own thread)
        try:
            if self.input_channel:
                self.input_channel.close()
        except Exception as e:
            logger.warning(f"Error closing channels: {e}")

        logger.info("Orchestrator stopped")

    def _run_worker_loop(self):
        """Run asyncio event loop for request processor."""
        logger.info("Starting request processor event loop thread...")
        asyncio.set_event_loop(self.worker_loop)
        self.worker_loop.run_forever()
        logger.info("Request processor event loop thread stopped")

    # ========================================================================
    # API Server Communication (Input/Output Sockets)
    # ========================================================================

    def process_input_sockets(self):
        """Background thread: Process input socket from API server."""
        logger.info("Starting input socket thread...")

        self.input_channel = SyncDealerChannel(self.input_addr, identity=self.engine_identity)
        self.input_channel.connect()

        while self._running:
            try:
                if not self.input_channel.poll(timeout_ms=100):
                    continue

                message_bytes = self.input_channel.recv()

                # Deserialize request
                request = OrchestratorRequest.from_msgpack(message_bytes)

                # Handle health check requests directly in input thread (non-blocking)
                # This prevents health checks from being blocked by long-running GPU operations
                if self._is_health_check_request(request):
                    self._handle_health_check_immediate(request)
                    continue

                try:
                    self.input_queue.put(request, block=True, timeout=1.0)
                    logger.debug(f"Received request {request.request_id} (type={request.request_type.value})")
                except queue.Full:
                    logger.warning("Input queue full, dropping request")

            except zmq.error.ZMQError as e:
                if not self._running:
                    break
                logger.error(f"ZMQ error in input thread: {e}")
                time.sleep(0.1)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Error in input thread: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Input socket thread stopped")

    def _is_health_check_request(self, request: OrchestratorRequest) -> bool:
        """Check if request is a health check that can be handled immediately."""
        if request.request_type != RequestType.UTILITY:
            return False
        return request.operation == "health_check"

    def _handle_health_check_immediate(self, request: OrchestratorRequest):
        """
        Handle health check request immediately in the input thread.

        This is non-blocking because get_stats() methods are read-only operations
        that don't require coordination with the event loop or request processor.
        """
        try:
            # Get stats (read-only, thread-safe)
            processor_stats = self.request_processor.get_stats()
            scheduler_stats = self.scheduler.get_stats()

            output = OrchestratorOutputs(
                request_id=request.request_id,
                output_type=OutputType.HEALTH_CHECK,
                outputs=[
                    {
                        "status": "healthy",
                        "active_requests": scheduler_stats["active_requests"],
                        "total_requests": scheduler_stats["total_requests"],
                        "pending_requests": scheduler_stats["pending_requests"],
                        "scheduler_policy": scheduler_stats["policy"],
                        "processor_connected": processor_stats["connected"],
                        "processor_ready": processor_stats["ready"],
                        "total_operations": processor_stats["total_operations"],
                    }
                ],
                finished=True,
            )

            # Send directly to output queue (bypasses event loop)
            self._send_output(output)
            logger.debug(f"Health check {request.request_id} handled immediately (non-blocking)")

        except Exception as e:
            logger.error(f"Error handling immediate health check: {e}")
            # Fall back to queuing the request for event loop handling
            try:
                self.input_queue.put(request, block=True, timeout=1.0)
            except queue.Full:
                logger.warning("Input queue full, dropping health check request")

    def process_output_sockets(self):
        """Background thread: Process output socket to API server."""
        logger.info("Starting output socket thread...")

        self.output_channel = SyncPushChannel(self.output_addr)
        self.output_channel.bind()

        while self._running:
            try:
                try:
                    output = self.output_queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue

                if not self._running:
                    break

                output_bytes = output.to_msgpack()
                self.output_channel.send(output_bytes)

                logger.debug(
                    f"Sent output {output.request_id} via PUSH socket "
                    f"(type={output.output_type.value}, finished={output.finished})"
                )

            except zmq.error.ZMQError as e:
                if not self._running:
                    break
                logger.error(f"ZMQ error in output thread: {e}")
                time.sleep(0.1)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Error in output thread: {e}", exc_info=True)
                time.sleep(0.1)

        # Close output channel from this thread to avoid cross-thread races
        if self.output_channel:
            self.output_channel.close()
            self.output_channel = None

        logger.info("Output socket thread stopped")

    # ========================================================================
    # Main Event Loop
    # ========================================================================

    def event_loop(self):
        """Main event loop: Process requests and execute on workers."""
        logger.info("Starting event loop...")

        # Track when we last logged state (to avoid log spam)
        last_state_log_time = 0.0
        state_log_interval = 5.0  # Log state every 5 seconds when idle

        while self._running:
            try:
                # Process input queue and add to scheduler
                self._process_input_queue()

                # Process next request from scheduler (sorted by seq_id)
                processed = self._process_engine_step()
                if not processed:
                    # Log scheduler state periodically when idle (for debugging)
                    current_time = time.time()
                    if current_time - last_state_log_time > state_log_interval:
                        scheduler_stats = self.scheduler.get_stats()
                        policy_stats = {}
                        if hasattr(self.scheduler.policy, "get_stats"):
                            policy_stats = self.scheduler.policy.get_stats()
                        if scheduler_stats["pending_requests"] > 0 or scheduler_stats["running_requests"] > 0:
                            logger.debug(
                                f"Event loop idle - scheduler state: "
                                f"pending={scheduler_stats['pending_requests']}, "
                                f"running={scheduler_stats['running_requests']}, "
                                f"policy_stats={policy_stats}"
                            )
                        last_state_log_time = current_time
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in event loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Event loop stopped")

    def _process_input_queue(self):
        """Process all available requests from input queue and add to scheduler."""
        while not self.input_queue.empty():
            try:
                request = self.input_queue.get_nowait()

                if request.request_type == RequestType.ADD:
                    self._add_request(request)
                elif request.request_type == RequestType.ABORT:
                    self._abort_request(request)
                elif request.request_type == RequestType.UTILITY:
                    self._handle_utility_request(request)

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing input request: {e}", exc_info=True)

    def _add_request(self, request: OrchestratorRequest):
        """Add a new training request to scheduler."""
        # Add to scheduler (scheduler will track it)
        request_id = self.scheduler.add_request(request)

        logger.debug(f"Added request {request_id} to scheduler (operation={request.operation})")

    def _abort_request(self, request: OrchestratorRequest):
        """Abort an existing request."""
        target_id = request.payload.target_request_id

        # Try to abort via scheduler
        if self.scheduler.abort_request(target_id):
            logger.info(f"Aborted request {target_id}")

            output = OrchestratorOutputs(
                request_id=target_id,
                output_type=OutputType.ERROR,
                finished=True,
                error="Request aborted",
            )
            self._send_output(output)
        else:
            logger.warning(f"Request {target_id} not found for abort")

    def _handle_utility_request(self, request: OrchestratorRequest):
        """Handle utility request (health check, etc.).

        Health checks are normally handled immediately in the input thread
        (_handle_health_check_immediate). This serves as a fallback if that fails.
        """
        operation = request.operation
        logger.debug(f"Handling utility request: {operation}")

        if operation == "health_check":
            self._handle_health_check_immediate(request)
        elif operation == "get_adapter_info":
            output = self._call_async(self.request_processor.execute_get_adapter_info(request))
            self._send_output(output)

    # ========================================================================
    # Request Processing
    # ========================================================================

    # Dispatch table: operation name → request processor method name.
    # Aliases (load_weights → load_state) are included as separate entries
    # so the if/elif chain is unnecessary.
    _OPERATION_DISPATCH: Dict[str, str] = {
        "forward": "execute_forward",
        "forward_backward": "execute_forward_backward",
        "optim_step": "execute_optim_step",
        "save_state": "execute_save_state",
        "save_lora_only": "execute_save_lora_only",
        "save_full_weights": "execute_save_full_weights",
        "load_state": "execute_load_state",
        "load_weights": "execute_load_state",  # alias
        "save_weights_for_sampler": "execute_save_state",  # uses save_state after patching data
        "sleep": "execute_sleep",
        "wake_up": "execute_wake_up",
        "sync_inference_weights": "execute_sync_inference_weights",
        "register_adapter": "execute_register_adapter",
        "save_adapter_state": "execute_save_adapter_state",
        "load_adapter_state": "execute_load_adapter_state",
        "get_adapter_info": "execute_get_adapter_info",
        "kill_session": "execute_kill_session",
    }

    def _process_engine_step(self) -> bool:
        """
        Process one engine step - execute training operations on workers.

        Returns:
            True if a request was processed, False if no request was available
        """
        t0 = time.perf_counter()
        scheduled_req = self.scheduler.get_next_request()
        if not scheduled_req:
            return False

        request = scheduled_req.request
        operation = scheduled_req.operation

        t_scheduled = time.perf_counter()

        try:
            processor_method_name = self._OPERATION_DISPATCH.get(operation)
            if processor_method_name is None:
                self._send_error_output(request, f"Unknown operation: {operation}")
                self.scheduler.mark_failed(request.request_id, f"Unknown operation: {operation}")
                return True

            # Special pre-processing for save_weights_for_sampler
            if operation == "save_weights_for_sampler":
                request.payload.save_optimizer = False

            processor_method = getattr(self.request_processor, processor_method_name)
            log_level = (
                logging.DEBUG
                if operation in ("forward", "forward_backward", "optim_step", "get_adapter_info")
                else logging.INFO
            )
            logger.log(log_level, f"Executing {operation} for request {request.request_id}")

            output = self._call_async(processor_method(request))

            t_executed = time.perf_counter()

            self._send_output(output)

            t_output = time.perf_counter()

            self.scheduler.mark_completed(request.request_id)

            logger.info(
                f"[TIMING] engine {operation}: "
                f"schedule={t_scheduled - t0:.4f}s "
                f"execute={t_executed - t_scheduled:.4f}s "
                f"send_output={t_output - t_executed:.4f}s "
                f"total={t_output - t0:.4f}s"
            )

        except Exception as e:
            logger.error(f"Error executing {operation}: {e}", exc_info=True)
            self._send_error_output(request, str(e))
            self.scheduler.mark_failed(request.request_id, str(e))

        return True

    # ========================================================================
    # RequestProcessor Helper
    # ========================================================================

    def _call_async(self, coro):
        """Call async request processor method from sync thread."""
        future = asyncio.run_coroutine_threadsafe(coro, self.worker_loop)
        return future.result()

    # ========================================================================
    # Output Management
    # ========================================================================

    def _send_output(self, output: OrchestratorOutputs):
        """Send output to output queue (non-blocking to avoid blocking event loop)."""
        try:
            self.output_queue.put_nowait(output)
            logger.debug(f"Queued output {output.request_id} (type={output.output_type.value})")
        except queue.Full:
            logger.error(
                f"Output queue full, dropping output for request {output.request_id}. "
                f"This should not happen - consider increasing output queue size."
            )

    def _send_error_output(self, request: OrchestratorRequest, error_msg: str):
        """Send error output."""
        output = OrchestratorOutputs(
            request_id=request.request_id,
            output_type=OutputType.ERROR,
            finished=True,
            error=error_msg,
        )
        self._send_output(output)

    # ========================================================================
    # Stats and Monitoring
    # ========================================================================

    def get_stats(self) -> dict:
        """Get engine statistics."""
        processor_stats = self.request_processor.get_stats()
        scheduler_stats = self.scheduler.get_stats()

        return {
            "running": self._running,
            "input_addr": self.input_addr,
            "output_addr": self.output_addr,
            "queues": {
                "input_queue_size": self.input_queue.qsize(),
                "output_queue_size": self.output_queue.qsize(),
            },
            "request_processor": processor_stats,
            "scheduler": scheduler_stats,
        }
