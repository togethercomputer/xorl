"""Minimal keyed Mooncake transport for OPD teacher hidden-state caches.

This is a deliberately small XoRL-local wrapper around the Mooncake distributed
object store. It is the second backend for the OPD teacher hidden-state path: the
teacher server can ``put_hidden`` a concatenated ``[num_tokens, hidden]`` tensor
under a key and return shape/dtype metadata, and the student loop can
``get_hidden`` it back by key + metadata without going through a shared
safetensors file.

The design borrows three ideas from TorchSpec's Mooncake transport
(``torchspec/transfer/mooncake``) but intentionally keeps the surface tiny:

* keyed tensor storage,
* shape/dtype metadata as the wire contract (bytes always match the metadata),
* a get path that reconstructs tensors purely from that metadata.

It does *not* port TorchSpec's RDMA-registered host/GPU buffers, async put
manager, or GPUDirect path. Those are throughput optimizations for the
Eagle3 KV-cache producer; the OPD teacher cache is a single concatenated
tensor per chunk, so the simple byte-level ``put``/``get`` object API is the
right altitude here.

Mooncake itself is imported lazily so that CPU-only unit tests (and the
safetensors backend) never need the ``mooncake`` package or CUDA. Tests inject
a fake client through ``MooncakeHiddenStore(client=...)``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol
from urllib.parse import urlparse

import torch


DEFAULT_HIDDEN_KEY = "hidden_states"

# Canonical dtype <-> string mapping for the metadata contract. Mirrors
# ModelRunner._teacher_cache_dtype so producer and consumer agree, with the
# extra integer dtypes the wire format can legitimately carry.
_STR_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def dtype_to_str(dtype: torch.dtype) -> str:
    """Canonical short name for a dtype (e.g. ``torch.bfloat16`` -> ``"bfloat16"``)."""
    return str(dtype).removeprefix("torch.")


def str_to_dtype(name: str | torch.dtype) -> torch.dtype:
    """Resolve a dtype name from the metadata contract to a ``torch.dtype``."""
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower().removeprefix("torch.")
    if key not in _STR_TO_DTYPE:
        raise ValueError(f"Unsupported teacher hidden-cache dtype {name!r}")
    return _STR_TO_DTYPE[key]


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to raw little-endian bytes (host, contiguous)."""
    flat = tensor.detach().to(device="cpu").contiguous().view(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def bytes_to_tensor(
    data: bytes,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Reconstruct a tensor from raw bytes using the shape/dtype contract."""
    numel = 1
    for dim in shape:
        numel *= int(dim)
    expected = numel * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected:
        raise ValueError(
            f"Mooncake hidden payload size mismatch: got {len(data)} bytes, "
            f"expected {expected} for shape={tuple(shape)} dtype={dtype}"
        )
    if numel == 0:
        return torch.empty(tuple(shape), dtype=dtype, device=device)
    # ``torch.frombuffer`` needs a writable buffer; the bytes returned by the
    # store are immutable, so copy into a bytearray once.
    flat = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype)
    return flat.reshape(tuple(shape)).to(device=device)


def _parse_size(value: str | int) -> int:
    """Parse ``"4GB"`` / ``"512MB"`` / int-bytes to an int byte count."""
    if isinstance(value, int):
        return value
    text = str(value).upper().strip()
    multipliers = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("T", 1024**4),
        ("G", 1024**3),
        ("M", 1024**2),
        ("K", 1024),
        ("B", 1),
    ]
    for suffix, mult in multipliers:
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * mult)
    return int(text)


@dataclass
class MooncakeStoreConfig:
    """Connection settings for the Mooncake distributed object store.

    Mirrors the small subset of TorchSpec's ``MooncakeConfig`` the OPD path
    needs. Defaults are read from the standard ``MOONCAKE_*`` environment
    variables so the teacher and student processes agree without threading
    config through every call.
    """

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8090/metadata"
    master_server_address: str = "localhost:50051"
    protocol: str = "tcp"
    device_name: str = ""
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        self.global_segment_size = _parse_size(self.global_segment_size)
        self.local_buffer_size = _parse_size(self.local_buffer_size)

    @classmethod
    def from_env(cls, overrides: Optional[Mapping[str, Any]] = None) -> "MooncakeStoreConfig":
        """Build config from ``MOONCAKE_*`` env vars, with optional overrides.

        ``overrides`` (typically the ``mooncake`` sub-dict of teacher-cache
        metadata) wins over env so a producer can pin the exact master a
        consumer must read from.
        """
        master_host = os.getenv("MOONCAKE_MASTER_HOST", "localhost")
        master_port = cls._env_port("MOONCAKE_MASTER_PORT", "50051")
        metadata_port = cls._env_port("MOONCAKE_METADATA_PORT", "8090")
        cfg = cls(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv("MOONCAKE_METADATA_SERVER", f"http://{master_host}:{metadata_port}/metadata"),
            master_server_address=os.getenv("MOONCAKE_MASTER_SERVER", f"{master_host}:{master_port}"),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
            global_segment_size=os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(4 * 1024 * 1024 * 1024)),
            local_buffer_size=os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(512 * 1024 * 1024)),
        )
        if overrides:
            for field_name in (
                "local_hostname",
                "metadata_server",
                "master_server_address",
                "protocol",
                "device_name",
                "global_segment_size",
                "local_buffer_size",
            ):
                if field_name in overrides and overrides[field_name] is not None:
                    setattr(cfg, field_name, overrides[field_name])
            cfg.__post_init__()
        return cfg

    def to_metadata(self) -> dict[str, Any]:
        """Connection fields a consumer needs to attach to the same store."""
        return {
            "master_server_address": self.master_server_address,
            "metadata_server": self.metadata_server,
            "protocol": self.protocol,
            "device_name": self.device_name,
        }

    @staticmethod
    def _env_port(name: str, default: str) -> str:
        value = os.getenv(name, default)
        parsed = urlparse(value)
        if parsed.scheme and parsed.port is not None:
            return str(parsed.port)
        return value.rsplit(":", 1)[-1] if ":" in value else value

    def apply_env_defaults(self) -> None:
        """Apply Mooncake process defaults needed before client setup."""
        # Mooncake's TCP-only memcpy fast path can segfault in same-host
        # multi-process get paths (kvcache-ai/Mooncake#1986). Preserve an
        # explicit user override.
        if self.protocol.lower() == "tcp" and "MC_STORE_MEMCPY" not in os.environ:
            os.environ["MC_STORE_MEMCPY"] = "0"


class _MooncakeClient(Protocol):
    """Minimal duck-typed view of ``mooncake.store.MooncakeDistributedStore``.

    Only the byte-level object API is used. Tests provide an in-memory fake
    implementing this surface.
    """

    def put(self, key: str, value: bytes) -> int: ...

    def get(self, key: str) -> bytes: ...

    def is_exist(self, key: str) -> int: ...

    def remove(self, key: str) -> int: ...


def _build_mooncake_client(config: MooncakeStoreConfig) -> _MooncakeClient:
    """Construct and set up a real ``MooncakeDistributedStore`` (lazy import)."""
    config.apply_env_defaults()
    from mooncake.store import MooncakeDistributedStore  # noqa: PLC0415 (lazy: needs CUDA + mooncake pkg)

    store = MooncakeDistributedStore()
    ret = store.setup(
        config.local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if ret is not None and ret != 0:
        raise RuntimeError(
            f"Failed to initialize Mooncake store (error={ret}). Check that the master is "
            f"running at {config.master_server_address} and metadata server is reachable at "
            f"{config.metadata_server}."
        )
    return store


class MooncakeHiddenStore:
    """Keyed tensor transport for OPD teacher hidden-state caches.

    ``put_hidden`` stores one concatenated hidden tensor under a key and returns
    metadata (key, shape, dtype). ``get_hidden`` reconstructs that tensor on a
    consumer using only the returned metadata. The underlying object key embeds
    the logical ``tensor_key`` so a base key can carry multiple named tensors
    without collision.

    Failures are loud: when this backend is selected, a missing key or a
    size mismatch raises rather than silently falling back to a stale file.
    """

    def __init__(
        self,
        config: Optional[MooncakeStoreConfig] = None,
        *,
        client: Optional[_MooncakeClient] = None,
        get_retry_max_wait_s: float = 30.0,
        get_retry_interval_s: float = 0.2,
    ) -> None:
        self.config = config or MooncakeStoreConfig.from_env()
        self._client: Optional[_MooncakeClient] = client
        self._owns_client = client is None
        self._get_retry_max_wait_s = float(get_retry_max_wait_s)
        self._get_retry_interval_s = float(get_retry_interval_s)

    @property
    def client(self) -> _MooncakeClient:
        if self._client is None:
            self._client = _build_mooncake_client(self.config)
        return self._client

    @staticmethod
    def object_key(key: str, tensor_key: str = DEFAULT_HIDDEN_KEY) -> str:
        """Full Mooncake object key for a logical (base key, tensor_key) pair."""
        return f"{key}/{tensor_key}"

    def put_hidden(
        self,
        key: str,
        tensor: torch.Tensor,
        tensor_key: str = DEFAULT_HIDDEN_KEY,
    ) -> dict[str, Any]:
        """Store a hidden tensor and return its metadata contract.

        Returns a dict shaped for ``teacher_hidden_caches`` consumption:
        ``backend``, ``key``, ``tensor_key``, ``tensor_shapes``,
        ``tensor_dtypes``, ``num_tokens``, ``hidden_size``, ``mooncake``.
        Token/hidden counts are derived from the leading and trailing dims so
        both rank-2 ([tokens, d]) and rank-3 ([layers, tokens, d]) caches work.
        """
        if tensor.ndim not in (2, 3):
            raise ValueError(
                "Mooncake teacher hidden cache must be rank 2 [tokens, hidden] or rank 3 "
                f"[layers, tokens, hidden]; got shape {tuple(tensor.shape)}"
            )
        tensor = tensor.detach().to(device="cpu").contiguous()
        obj_key = self.object_key(key, tensor_key)
        payload = tensor_to_bytes(tensor)
        ret = self.client.put(obj_key, payload)
        if ret is not None and ret != 0:
            raise RuntimeError(f"Mooncake put failed for key {obj_key!r} (error={ret})")

        token_dim = 0 if tensor.ndim == 2 else 1
        return {
            "backend": "mooncake",
            "key": key,
            "tensor_key": tensor_key,
            "tensor_shapes": {tensor_key: list(tensor.shape)},
            "tensor_dtypes": {tensor_key: dtype_to_str(tensor.dtype)},
            "num_tokens": int(tensor.shape[token_dim]),
            "hidden_size": int(tensor.shape[-1]),
            "dtype": dtype_to_str(tensor.dtype),
            "mooncake": self.config.to_metadata(),
        }

    def get_hidden(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype | str,
        device: torch.device | str = "cpu",
        tensor_key: str = DEFAULT_HIDDEN_KEY,
    ) -> torch.Tensor:
        """Fetch and reconstruct a hidden tensor by key + shape/dtype contract."""
        obj_key = self.object_key(key, tensor_key)
        resolved_dtype = str_to_dtype(dtype)
        data = self._get_with_retry(obj_key)
        return bytes_to_tensor(data, tuple(int(s) for s in shape), resolved_dtype, device=device)

    def get_hidden_from_metadata(
        self,
        metadata: Mapping[str, Any],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Fetch a hidden tensor straight from a producer-returned metadata dict."""
        key, tensor_key, shape, dtype = parse_mooncake_metadata(metadata)
        return self.get_hidden(key, shape, dtype, device=device, tensor_key=tensor_key)

    def remove_hidden(self, key: str, tensor_key: str = DEFAULT_HIDDEN_KEY) -> None:
        """Best-effort delete of a stored hidden tensor after safe consumption."""
        obj_key = self.object_key(key, tensor_key)
        try:
            self.client.remove(obj_key)
        except Exception:  # pragma: no cover - cleanup is best-effort
            pass

    def _get_with_retry(self, obj_key: str) -> bytes:
        deadline = time.monotonic() + self._get_retry_max_wait_s
        while True:
            data = self.client.get(obj_key)
            if data:
                return bytes(data)
            if time.monotonic() >= deadline:
                raise KeyError(
                    f"Mooncake get returned no data for key {obj_key!r} after "
                    f"{self._get_retry_max_wait_s:.1f}s; the producer may have failed to write it"
                )
            time.sleep(self._get_retry_interval_s)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
        self._client = None


def is_mooncake_entry(entry: Any) -> bool:
    """True if a teacher-cache entry selects the Mooncake backend."""
    return isinstance(entry, Mapping) and str(entry.get("backend", "")).lower() == "mooncake"


def parse_mooncake_metadata(
    metadata: Mapping[str, Any],
) -> tuple[str, str, list[int], torch.dtype]:
    """Validate a Mooncake teacher-cache entry and return (key, tensor_key, shape, dtype).

    Raises a clear ``ValueError`` for any missing/malformed field so a producer
    bug surfaces loudly instead of corrupting the OPD loss with garbage rows.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Mooncake teacher-cache entry must be a mapping, got {type(metadata)!r}")
    key = metadata.get("key")
    if not key:
        raise ValueError(f"Mooncake teacher-cache entry missing 'key': {metadata!r}")
    tensor_key = metadata.get("tensor_key", DEFAULT_HIDDEN_KEY)

    shapes = metadata.get("tensor_shapes")
    if not isinstance(shapes, Mapping) or tensor_key not in shapes:
        raise ValueError(f"Mooncake teacher-cache entry missing tensor_shapes[{tensor_key!r}]: {metadata!r}")
    shape = list(shapes[tensor_key])
    if len(shape) not in (2, 3):
        raise ValueError(f"Mooncake teacher-cache shape must be rank 2 or 3, got {shape} for key {key!r}")

    dtypes = metadata.get("tensor_dtypes")
    if not isinstance(dtypes, Mapping) or tensor_key not in dtypes:
        raise ValueError(f"Mooncake teacher-cache entry missing tensor_dtypes[{tensor_key!r}]: {metadata!r}")
    dtype = str_to_dtype(dtypes[tensor_key])
    return str(key), str(tensor_key), shape, dtype
