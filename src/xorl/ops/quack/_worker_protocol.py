"""Bounded MessagePack protocol for Quack compile workers."""

from __future__ import annotations

import os
import select
import struct
import time
from typing import Any, BinaryIO

import msgpack
import torch


_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_TYPE_KEY = "__xorl_quack_type__"
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def _default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return {_TYPE_KEY: "dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {_TYPE_KEY: "device", "value": str(value)}
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "value": list(value)}
    raise TypeError(f"Unsupported compile-worker value: {type(value).__name__}")


def _object_hook(value: dict[str, Any]) -> Any:
    wire_type = value.get(_TYPE_KEY)
    if wire_type is None:
        return value
    if set(value) != {_TYPE_KEY, "value"}:
        raise ValueError("Invalid compile-worker tagged value")
    if wire_type == "dtype":
        dtype = _DTYPES.get(value["value"])
        if dtype is None:
            raise ValueError(f"Unsupported compile-worker dtype: {value['value']!r}")
        return dtype
    if wire_type == "device":
        return torch.device(value["value"])
    if wire_type == "tuple":
        if not isinstance(value["value"], list):
            raise ValueError("Invalid compile-worker tuple")
        return tuple(value["value"])
    raise ValueError(f"Unknown compile-worker tagged value: {wire_type!r}")


def send_message(stream: BinaryIO, message: Any) -> None:
    """Write one length-prefixed, non-executable message."""
    data = msgpack.packb(message, use_bin_type=True, strict_types=True, default=_default)
    if len(data) > _MAX_MESSAGE_BYTES:
        raise ValueError("Compile-worker message exceeds size limit")
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int, timeout_s: float | None) -> bytes:
    chunks = []
    remaining = size
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while remaining:
        if deadline is None:
            chunk = stream.read(remaining)
        else:
            wait_s = deadline - time.monotonic()
            if wait_s <= 0:
                raise TimeoutError(f"Compile worker did not respond within {timeout_s:g}s")
            readable, _, _ = select.select([stream.fileno()], [], [], wait_s)
            if not readable:
                raise TimeoutError(f"Compile worker did not respond within {timeout_s:g}s")
            chunk = os.read(stream.fileno(), remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO, *, timeout_s: float | None = None) -> Any:
    """Read and validate one length-prefixed message, or ``None`` at EOF."""
    header = _read_exact(stream, 4, timeout_s)
    if not header:
        return None
    if len(header) != 4:
        raise ValueError("Truncated compile-worker message header")
    length = struct.unpack("<I", header)[0]
    if length == 0:
        return None
    if length > _MAX_MESSAGE_BYTES:
        raise ValueError("Compile-worker message exceeds size limit")
    data = _read_exact(stream, length, timeout_s)
    if len(data) != length:
        raise ValueError("Truncated compile-worker message body")
    return msgpack.unpackb(
        data,
        raw=False,
        strict_map_key=True,
        object_hook=_object_hook,
    )
