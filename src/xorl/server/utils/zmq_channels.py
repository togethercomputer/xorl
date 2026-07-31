"""Typed ZMQ Channel Abstractions.

Encapsulates ZMQ socket setup, framing, and polling behind clean send/recv APIs.
All channels deal in raw bytes — serialization is the caller's responsibility.

Channel types:
- SyncPushChannel: Sync PUSH socket (bind/connect, send)
- SyncDealerChannel: Sync DEALER socket (connect, poll, recv)
- AsyncPullChannel: Async PULL socket (bind/connect, poll, recv)
- AsyncRouterChannel: Async ROUTER socket (bind, identity-routed send/recv)
- AsyncDealerChannel: Async DEALER socket (connect, send/recv with timeouts)
"""

import asyncio
import logging
from typing import Optional, Tuple

import zmq
import zmq.asyncio


logger = logging.getLogger(__name__)


# ============================================================================
# Sync Channels (for Orchestrator background threads)
# ============================================================================


class SyncPushChannel:
    """Sync PUSH socket that sends single-frame messages."""

    def __init__(self, address: str, *, hwm: int = 1000, send_timeout: int = 1000):
        self._address = address
        self._hwm = hwm
        self._send_timeout = send_timeout
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None

    def bind(self) -> None:
        """Create context, socket, and bind."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, self._hwm)
        self._socket.setsockopt(zmq.SNDTIMEO, self._send_timeout)
        self._socket.bind(self._address)
        logger.info(f"SyncPushChannel bound to {self._address}")

    def connect(self) -> None:
        """Create context, socket, and connect."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, self._hwm)
        self._socket.setsockopt(zmq.SNDTIMEO, self._send_timeout)
        self._socket.connect(self._address)
        logger.info(f"SyncPushChannel connected to {self._address}")

    def send(self, data: bytes) -> None:
        """Send a single-frame message."""
        self._socket.send(data, copy=False)

    def close(self) -> None:
        """Close socket and terminate context."""
        if self._socket:
            self._socket.close(linger=0)
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None


class SyncDealerChannel:
    """Sync DEALER socket that connects and receives via polling.

    Strips the empty-frame prefix that DEALER adds/receives in ROUTER-DEALER pairs.
    """

    def __init__(self, address: str, identity: bytes, *, hwm: int = 1000):
        self._address = address
        self._identity = identity
        self._hwm = hwm
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

    def connect(self) -> None:
        """Create context, socket, and connect."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, self._identity)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, self._hwm)
        self._socket.connect(self._address)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        logger.info(f"SyncDealerChannel connected to {self._address}")

    def poll(self, timeout_ms: int = 100) -> bool:
        """Poll for incoming data. Returns True if data available."""
        socks = dict(self._poller.poll(timeout=timeout_ms))
        return self._socket in socks

    def recv(self) -> bytes:
        """Receive message, stripping DEALER empty-frame prefix."""
        frames = self._socket.recv_multipart(copy=False)
        # DEALER receives [empty, message] from ROUTER; handle 1-frame fallback
        if len(frames) >= 2:
            return frames[-1].bytes
        return frames[0].bytes

    def close(self) -> None:
        """Close socket and terminate context."""
        if self._socket:
            self._socket.close(linger=0)
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
        self._poller = None


# ============================================================================
# Async Channels (for OrchestratorClient, RemoteBackend, Worker)
# ============================================================================


class AsyncPullChannel:
    """Async PULL socket that receives via polling."""

    def __init__(
        self,
        address: str,
        *,
        hwm: int = 1000,
        context: Optional[zmq.asyncio.Context] = None,
    ):
        self._address = address
        self._hwm = hwm
        self._context = context
        self._owns_context = context is None
        self._socket: Optional[zmq.asyncio.Socket] = None

    def connect(self) -> None:
        """Create socket (and context if needed) and connect."""
        if self._owns_context:
            self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, self._hwm)
        self._socket.connect(self._address)
        logger.info(f"AsyncPullChannel connected to {self._address}")

    def bind(self) -> None:
        """Create socket (and context if needed) and bind."""
        if self._owns_context:
            self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, self._hwm)
        self._socket.bind(self._address)
        logger.info(f"AsyncPullChannel bound to {self._address}")

    async def poll(self, timeout_ms: int = 100) -> bool:
        """Poll for incoming data. Returns True if data available."""
        events = await self._socket.poll(timeout=timeout_ms)
        return bool(events & zmq.POLLIN)

    async def recv(self) -> bytes:
        """Receive a single-frame message."""
        return await self._socket.recv()

    def close(self) -> None:
        """Close socket (and context if owned)."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._owns_context and self._context:
            self._context.term()
            self._context = None


class AsyncRouterChannel:
    """Async ROUTER socket that binds and handles identity-routed messages.

    Multipart framing [identity, empty, message] is handled internally.
    Callers work with (identity, data) tuples.
    """

    def __init__(
        self,
        address: str,
        *,
        hwm: int = 1000,
        context: Optional[zmq.asyncio.Context] = None,
    ):
        self._address = address
        self._hwm = hwm
        self._context = context
        self._owns_context = context is None
        self._socket: Optional[zmq.asyncio.Socket] = None

    def bind(self) -> None:
        """Create socket (and context if needed) and bind."""
        if self._owns_context:
            self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, self._hwm)
        # ROUTER_MANDATORY: raise EHOSTUNREACH on an unroutable send instead of
        # silently dropping the frame. A DEALER peer's identity is briefly
        # unrouted right after it (re)connects; without this + the retry in
        # send(), a request dropped during that window left the caller awaiting a
        # response that never arrived (root cause of the 1-node forward_backward
        # dispatch deadlock: the orchestrator never received the fb command).
        self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._socket.bind(self._address)
        logger.info(f"AsyncRouterChannel bound to {self._address}")

    async def send(self, identity: bytes, data: bytes, *, route_timeout: float = 30.0) -> None:
        """Send data to a peer, retrying while its identity is not yet routable.

        With ROUTER_MANDATORY, ZMQ raises EHOSTUNREACH when ``identity`` is not
        currently routable (e.g. the DEALER peer just (re)connected and the
        ROUTER has not registered it yet). Retry with a short exponential backoff
        until the peer is routable or ``route_timeout`` elapses, so a transient
        unroutable window never silently drops a request.
        """
        deadline = asyncio.get_event_loop().time() + route_timeout
        backoff = 0.005
        warned = False
        while True:
            try:
                await self._socket.send_multipart([identity, b"", data])
                return
            except zmq.error.ZMQError as e:
                if e.errno == zmq.EHOSTUNREACH and asyncio.get_event_loop().time() < deadline:
                    if not warned:
                        logger.warning(
                            f"AsyncRouterChannel: identity {identity!r} not routable yet; "
                            f"retrying send for up to {route_timeout:.0f}s"
                        )
                        warned = True
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 0.1)
                    continue
                raise

    async def recv(self) -> Tuple[bytes, bytes]:
        """Receive message, returning (identity, data).

        Raises:
            RuntimeError: If frame count is unexpected (< 3).
        """
        frames = await self._socket.recv_multipart()
        if len(frames) < 3:
            raise RuntimeError(f"Expected 3+ frames from ROUTER recv, got {len(frames)}")
        return frames[0], frames[2]

    async def poll(self, timeout_ms: int = 100) -> bool:
        """Poll for incoming data. Returns True if data available."""
        events = await self._socket.poll(timeout=timeout_ms)
        return bool(events & zmq.POLLIN)

    def close(self) -> None:
        """Close socket (and context if owned)."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._owns_context and self._context:
            self._context.term()
            self._context = None


class AsyncDealerChannel:
    """Async DEALER socket that connects with identity.

    Multipart framing [empty, message] is handled internally.
    Callers work with raw message bytes.
    """

    def __init__(
        self,
        address: str,
        identity: bytes,
        *,
        context: Optional[zmq.asyncio.Context] = None,
    ):
        self._address = address
        self._identity = identity
        self._context = context
        self._owns_context = context is None
        self._socket: Optional[zmq.asyncio.Socket] = None

    def connect(self) -> None:
        """Create socket (and context if needed) and connect."""
        if self._owns_context:
            self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, self._identity)
        self._socket.connect(self._address)
        logger.info(f"AsyncDealerChannel connected to {self._address}")

    async def send(self, data: bytes) -> None:
        """Send data with DEALER empty-frame prefix."""
        await self._socket.send_multipart([b"", data])

    async def recv(self, timeout: Optional[float] = None) -> bytes:
        """Receive message, stripping empty-frame prefix.

        Args:
            timeout: Timeout in seconds. None means wait forever.

        Raises:
            asyncio.TimeoutError: If timeout expires.
            RuntimeError: If frame count is unexpected.
        """
        coro = self._socket.recv_multipart()
        if timeout is not None:
            frames = await asyncio.wait_for(coro, timeout=timeout)
        else:
            frames = await coro
        if len(frames) != 2:
            raise RuntimeError(f"Expected 2 frames from DEALER recv, got {len(frames)}")
        return frames[1]

    def close(self) -> None:
        """Close socket (and context if owned)."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._owns_context and self._context:
            self._context.term()
            self._context = None
