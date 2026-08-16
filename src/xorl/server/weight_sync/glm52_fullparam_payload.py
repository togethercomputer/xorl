"""Checksummed byte-payload protocol for GLM-5.2 full-param weight sync.

Producer side publishes an *immutable snapshot* of each component's per-step
quantized cache (the bytes the trainer forward consumed), wrapped as inert
``uint8`` byte tensors with explicit dtype/shape metadata and a SHA-256
checksum per field.  The snapshot is copied from the cache at build time,
UNDER the component staleness gate: the copy is byte-identical to what the
trainer scored with, and no later cache refresh can mutate a payload after
its checksums were minted (ownership, not aliasing — a zero-copy view here
would let refresh_quantized_cache() rewrite already-checksummed bytes).  The
checksum binds the protocol version, weight version, target path, field
name, dtype, and shape to the raw bytes, so a payload replayed under a
different version label or a middleware dtype conversion fails verification.

Receiver side verifies every checksum and requires an independently supplied,
exact semantic inventory BEFORE any weight is applied (verify-all-then-apply).
Any missing or extra target, contract-version mismatch, dtype conversion,
reshape, byte-count mismatch, or unknown kind RAISES — never a warning, never
a partial application.  True no-mixed-version serving additionally requires
the engine not to serve between the two apply phases; that scheduling
obligation is the engine integration's, and is documented rather than
simulated here.

This module deliberately imports no trainer component code, so receivers can
import it alongside only serving modules.  Producer dispatch is duck-typed on
the ``glm52_fullparam_payload_kind`` attribute the full-param components
declare.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)


def _engagement_log(message: str, *args) -> None:
    """Engagement logging that survives third-party logging reconfiguration.

    In-process engines (sglang) apply ``dictConfig(disable_existing_loggers=
    True)``, which silently disables this module's logger. Engagement lines
    are part of the lane contract (a gate must cover or raise, never go
    silent), so re-enable before emitting.
    """

    if logger.disabled:
        logger.disabled = False
    logger.info(message, *args)


GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION = "glm52_fullparam_byte_payload_v1"

_MANIFEST_FILENAME = "manifest.json"

# The loader is intentionally bounded even though the admitted model is large.
# These limits are comfortably above the 334-item/1,629-field GLM-5.2 inventory
# and its FP8 bytes while preventing attacker-controlled JSON from requesting
# unbounded allocation or filesystem traversal.
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_PAYLOAD_ITEMS = 4_096
_MAX_PAYLOAD_FIELDS = 32_768
_MAX_FIELD_BYTES = 16 * 1024**3
_MAX_TOTAL_PAYLOAD_BYTES = 2 * 1024**4
_MAX_WEIGHT_VERSION_BYTES = 1_024
_MAX_WEIGHT_STEP = (1 << 63) - 1

_OPEN_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_OPEN_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_OPEN_NONBLOCK = getattr(os, "O_NONBLOCK", 0)

_DTYPES: dict[str, torch.dtype] = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_DTYPE_ITEMSIZE: dict[str, int] = {
    "float8_e4m3fn": 1,
    "float32": 4,
    "bfloat16": 2,
}

# Every admitted item kind pins its exact ordered field-name/dtype schema.
_CHECKPOINT_SPLIT_SCHEMA: tuple[tuple[str, str], ...] = (
    ("gate", "float8_e4m3fn"),
    ("gate_scale_inv", "float32"),
    ("up", "float8_e4m3fn"),
    ("up_scale_inv", "float32"),
    ("down", "float8_e4m3fn"),
    ("down_scale_inv", "float32"),
)

_KIND_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "block_fp8_linear": (
        ("weight", "float8_e4m3fn"),
        ("weight_scale_inv", "float32"),
    ),
    "bf16_router": (("weight", "bfloat16"),),
    "block_fp8_expert_bank": (
        ("gate_up", "float8_e4m3fn"),
        ("gate_up_scale_inv", "float32"),
        ("down", "float8_e4m3fn"),
        ("down_scale_inv", "float32"),
    ),
    # Checkpoint-form kinds: the split per-projection tensors the serving
    # LOADER actually consumes (dense MLP composite; one globally-indexed
    # expert of an EP-local bank).  Field order is the loader's fuse order.
    "block_fp8_dense_mlp": _CHECKPOINT_SPLIT_SCHEMA,
    "block_fp8_expert": _CHECKPOINT_SPLIT_SCHEMA,
}

# Payload field name -> HF checkpoint tensor suffix for checkpoint-form kinds.
_CHECKPOINT_FIELD_TO_HF_SUFFIX: dict[str, str] = {
    "gate": "gate_proj.weight",
    "gate_scale_inv": "gate_proj.weight_scale_inv",
    "up": "up_proj.weight",
    "up_scale_inv": "up_proj.weight_scale_inv",
    "down": "down_proj.weight",
    "down_scale_inv": "down_proj.weight_scale_inv",
}


class Glm52FullParamPayloadError(RuntimeError):
    """Raised for any payload integrity, schema, or application violation."""


def _validate_weight_identity(weight_version: object, weight_step: object, *, label: str) -> tuple[str, int]:
    try:
        encoded_version = weight_version.encode("utf-8") if isinstance(weight_version, str) else b""
    except UnicodeEncodeError as exc:
        raise Glm52FullParamPayloadError(f"{label} weight_version must be valid UTF-8") from exc
    if (
        not isinstance(weight_version, str)
        or not weight_version
        or "\x00" in weight_version
        or len(encoded_version) > _MAX_WEIGHT_VERSION_BYTES
    ):
        raise Glm52FullParamPayloadError(
            f"{label} weight_version must be a non-empty UTF-8 string of at most {_MAX_WEIGHT_VERSION_BYTES} bytes"
        )
    if not isinstance(weight_step, int) or isinstance(weight_step, bool) or not 0 <= weight_step <= _MAX_WEIGHT_STEP:
        raise Glm52FullParamPayloadError(f"{label} weight_step must be a nonnegative signed 64-bit integer")
    return weight_version, weight_step


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _plain_filename(name: object, *, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or os.path.basename(name) != name
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or "\x00" in name
    ):
        raise Glm52FullParamPayloadError(f"{label} must be a plain filename, got {name!r}")
    return name


def _canonical_existing_directory(directory: str, *, label: str) -> str:
    if not isinstance(directory, str) or not directory:
        raise Glm52FullParamPayloadError(f"{label} must be a non-empty path string")
    absolute = os.path.abspath(directory)
    if os.path.realpath(absolute) != absolute:
        raise Glm52FullParamPayloadError(f"{label} must not contain symlink components: {directory!r}")
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError as exc:
        raise Glm52FullParamPayloadError(f"{label} does not exist: {directory!r}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Glm52FullParamPayloadError(f"{label} must be a real directory, not a symlink or special file")
    return absolute


def _canonical_absent_output(directory: str, *, label: str, create_parent: bool = False) -> tuple[str, str]:
    if not isinstance(directory, str) or not directory:
        raise Glm52FullParamPayloadError(f"{label} must be a non-empty path string")
    absolute = os.path.abspath(directory)
    basename = _plain_filename(os.path.basename(absolute), label=f"{label} basename")
    parent = os.path.dirname(absolute)
    if create_parent:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    parent = _canonical_existing_directory(parent, label=f"{label} parent")
    canonical = os.path.join(parent, basename)
    if canonical != absolute or os.path.realpath(absolute) != absolute:
        raise Glm52FullParamPayloadError(f"{label} must resolve canonically without symlinks: {directory!r}")
    if os.path.lexists(canonical):
        raise Glm52FullParamPayloadError(f"{label} {canonical!r} already exists; refusing to overwrite it")
    return canonical, parent


def _open_regular_at(directory_fd: int, filename: str, *, label: str, max_bytes: int) -> tuple[int, int]:
    filename = _plain_filename(filename, label=label)
    flags = os.O_RDONLY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW | _OPEN_NONBLOCK
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
    except FileNotFoundError as exc:
        raise Glm52FullParamPayloadError(f"{label} missing: {filename!r}") from exc
    except OSError as exc:
        raise Glm52FullParamPayloadError(f"Cannot securely open {label} {filename!r}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise Glm52FullParamPayloadError(f"{label} {filename!r} must be a regular nonsymlink file")
        if metadata.st_size < 0 or metadata.st_size > max_bytes:
            raise Glm52FullParamPayloadError(
                f"{label} {filename!r} is {metadata.st_size} bytes; maximum is {max_bytes}"
            )
        return descriptor, metadata.st_size
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_at(
    directory_fd: int,
    filename: str,
    *,
    label: str,
    max_bytes: int,
    exact_bytes: int | None = None,
) -> bytes:
    descriptor, size = _open_regular_at(directory_fd, filename, label=label, max_bytes=max_bytes)
    try:
        if exact_bytes is not None and size != exact_bytes:
            raise Glm52FullParamPayloadError(f"{label} {filename!r} carries {size} bytes, expected {exact_bytes} bytes")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise Glm52FullParamPayloadError(f"{label} {filename!r} was truncated while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Glm52FullParamPayloadError(f"{label} {filename!r} grew while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: object) -> None:
    view = memoryview(data)  # type: ignore[arg-type]
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - os.write either writes or raises
            raise OSError("short write")
        view = view[written:]


def _create_regular_at(directory_fd: int, filename: str, data: object, *, mode: int = 0o600) -> None:
    filename = _plain_filename(filename, label="output filename")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _OPEN_CLOEXEC | _OPEN_NOFOLLOW
    descriptor = os.open(filename, flags, mode, dir_fd=directory_fd)
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_no_duplicate_keys(raw: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Glm52FullParamPayloadError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except Glm52FullParamPayloadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Glm52FullParamPayloadError(f"{label} is not canonical UTF-8 JSON: {exc}") from exc


def _require_exact_keys(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise Glm52FullParamPayloadError(f"{label} keys must be exactly {sorted(keys)}, got {actual}")
    return value


def _rename_noreplace(source: str, destination: str) -> None:
    """Linux atomic directory publish with an absent-destination precondition."""

    import ctypes  # noqa: PLC0415

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - production and CI are Linux/glibc
        raise Glm52FullParamPayloadError("Atomic no-replace directory publication requires renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise Glm52FullParamPayloadError(f"Output directory {destination!r} appeared during publication")
        raise OSError(error, os.strerror(error), destination)


def _fsync_parent_after_commit_noexcept(parent: str) -> None:
    """Best-effort rename durability; never turn a committed rename into failure."""

    try:
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        # The rename is the commit point. Reporting failure now would invite a
        # caller to retry a checkpoint that is already fully visible.
        try:
            logger.exception("Directory fsync failed after committed atomic rename of %s", parent)
        except BaseException:
            pass


@dataclass(frozen=True)
class Glm52PayloadField:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: Tensor  # 1-D uint8 byte tensor (producer side: an immutable snapshot)
    checksum: str


@dataclass(frozen=True)
class Glm52PayloadItem:
    target: str
    kind: str
    contract_version: str
    fields: tuple[Glm52PayloadField, ...]


@dataclass(frozen=True)
class Glm52FullParamSyncPayload:
    protocol_version: str
    weight_version: str
    weight_step: int
    items: tuple[Glm52PayloadItem, ...]
    manifest_checksum: str


@dataclass(frozen=True)
class Glm52ExpectedPayloadField:
    """One field in a receiver-owned semantic inventory."""

    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Glm52ExpectedPayloadItem:
    """Expected contract and byte geometry for one receiver target."""

    target: str
    kind: str
    contract_version: str
    fields: tuple[Glm52ExpectedPayloadField, ...]


@dataclass(frozen=True)
class Glm52ExpectedPayloadInventory:
    """Complete receiver-owned inventory required by a payload apply.

    This inventory must come from trusted receiver configuration or model
    admission, independently of the payload being validated.  Its purpose is
    to distinguish a complete update from a checksum-valid subset.
    """

    items: tuple[Glm52ExpectedPayloadItem, ...]


@dataclass(frozen=True)
class Glm52PreparedCheckpoint:
    """Fully materialized checkpoint awaiting an engine-controlled activation.

    Materialization is deliberately separate from activation: creating this
    object never advances the serving version guard. The checkpoint remains
    inert until :func:`activate_glm52_prepared_checkpoint` invokes the serving
    engine's serialized reload/swap callback successfully.
    """

    directory: str
    weight_version: str
    weight_step: int
    manifest_checksum: str


class Glm52WeightVersionGuard:
    """Process-local serialized monotonic guard for one serving engine.

    The guard protects admission, mutation, and version commit with one owned
    reentrant lock. It is not cross-process authentication or durable state:
    payload/checkpoint directories need trusted ACLs, and after a process
    restart the integration must initialize ``last_step`` from the engine's
    durably activated version before accepting another update.
    """

    def __init__(self, last_step: int = -1) -> None:
        if not isinstance(last_step, int) or isinstance(last_step, bool) or not -1 <= last_step <= _MAX_WEIGHT_STEP:
            raise Glm52FullParamPayloadError("last_step must be -1 or a nonnegative signed 64-bit integer")
        self._last_step = last_step
        self._lock = threading.RLock()

    @property
    def last_step(self) -> int:
        with self._lock:
            return self._last_step

    def _admit_locked(self, *, weight_step: int, weight_version: str) -> None:
        _validate_weight_identity(weight_version, weight_step, label="candidate")
        if weight_step <= self._last_step:
            raise Glm52FullParamPayloadError(
                f"Weight-version regression: payload step {weight_step} "
                f"(version {weight_version!r}) is not newer than applied step {self._last_step}"
            )

    def admit(self, payload: "Glm52FullParamSyncPayload") -> None:
        """Read-only admission probe; mutations must use ``serialized_update``."""

        if not isinstance(payload, Glm52FullParamSyncPayload):
            raise Glm52FullParamPayloadError("guard admission requires a Glm52FullParamSyncPayload")
        with self._lock:
            self._admit_locked(weight_step=payload.weight_step, weight_version=payload.weight_version)

    def commit(self, payload: "Glm52FullParamSyncPayload") -> None:
        """Commit a payload under the lock (kept for narrow integrations)."""

        if not isinstance(payload, Glm52FullParamSyncPayload):
            raise Glm52FullParamPayloadError("guard commit requires a Glm52FullParamSyncPayload")
        with self._lock:
            self._admit_locked(weight_step=payload.weight_step, weight_version=payload.weight_version)
            self._last_step = payload.weight_step

    @contextmanager
    def serialized_update(self, *, weight_step: int, weight_version: str):
        """Hold one guard-owned transaction through mutation and commit."""

        with self._lock:
            self._admit_locked(weight_step=weight_step, weight_version=weight_version)
            yield
            # Validation and admission happened under this same lock; assigning
            # the Python integer cannot fail after the receiver mutation.
            self._last_step = weight_step


# ---------------------------------------------------------------------------
# Byte views and checksums
# ---------------------------------------------------------------------------


def _byte_view(tensor: Tensor) -> Tensor:
    """Return a zero-copy 1-D uint8 view of a contiguous tensor."""

    if not tensor.is_contiguous():
        raise Glm52FullParamPayloadError("Payload source tensors must be contiguous byte carriers")
    return tensor.view(torch.uint8).flatten()


def _field_header(
    *,
    protocol_version: str,
    weight_version: str,
    weight_step: int,
    target: str,
    field_name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> bytes:
    return json.dumps(
        {
            "protocol": protocol_version,
            "weight_version": weight_version,
            "weight_step": int(weight_step),
            "target": target,
            "field": field_name,
            "dtype": dtype,
            "shape": list(shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _checksum(header: bytes, byte_view: Tensor) -> str:
    if byte_view.dtype is not torch.uint8 or byte_view.dim() != 1:
        raise Glm52FullParamPayloadError("Checksums are computed over 1-D uint8 byte tensors only")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(byte_view.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _manifest_checksum(
    protocol_version: str,
    weight_version: str,
    weight_step: int,
    items: Iterable[Glm52PayloadItem],
) -> str:
    payload = {
        "protocol": protocol_version,
        "weight_version": weight_version,
        "weight_step": int(weight_step),
        "items": [
            {
                "target": item.target,
                "kind": item.kind,
                "contract_version": item.contract_version,
                "fields": [
                    {
                        "name": field.name,
                        "dtype": field.dtype,
                        "shape": list(field.shape),
                        "checksum": field.checksum,
                    }
                    for field in item.fields
                ],
            }
            for item in items
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


def _component_field_sources(module: nn.Module, kind: str) -> tuple[Tensor, ...]:
    if kind == "block_fp8_linear":
        weight, scale = module.publishable_weight_bytes()
        return (weight, scale)
    if kind == "bf16_router":
        return (module.publishable_weight_bytes(),)
    if kind == "block_fp8_expert_bank":
        gate_up, gate_up_scale, down, down_scale = module.publishable_expert_bytes()
        return (gate_up, gate_up_scale, down, down_scale)
    if kind == "block_fp8_dense_mlp":
        return tuple(module.publishable_checkpoint_projections())
    if kind == "block_fp8_expert":
        return tuple(module.publishable_checkpoint_tensors())
    raise Glm52FullParamPayloadError(f"Unknown GLM-5.2 full-param payload kind {kind!r}")


def publish_glm52_fullparam_payload(
    named_components: Iterable[tuple[str, nn.Module]],
    *,
    weight_version: str,
    weight_step: int = 0,
) -> Glm52FullParamSyncPayload:
    """Build a payload of immutable cache snapshots from full-param components.

    Each component's ``publishable_*`` surface RAISES when its cache is stale
    (the component staleness gate), so a payload can never carry bytes the
    trainer did not score with.  Field ``data`` tensors are byte COPIES taken
    at build time: the payload owns its bytes, and a subsequent
    ``refresh_quantized_cache()`` (the next optimizer step) cannot mutate a
    payload whose checksums were already minted.  The copy is bounded by the
    published scope and is freed with the payload.
    """

    weight_version, weight_step = _validate_weight_identity(weight_version, weight_step, label="publication")

    items: list[Glm52PayloadItem] = []
    seen_targets: set[str] = set()
    for target, module in named_components:
        if not isinstance(target, str) or not target:
            raise Glm52FullParamPayloadError(f"Invalid payload target name {target!r}")
        if target in seen_targets:
            raise Glm52FullParamPayloadError(f"Duplicate payload target {target!r}")
        seen_targets.add(target)

        kind = getattr(module, "glm52_fullparam_payload_kind", None)
        if kind not in _KIND_SCHEMAS:
            raise Glm52FullParamPayloadError(
                f"Component {target!r} ({type(module).__name__}) does not declare an admitted "
                f"glm52_fullparam_payload_kind; refusing to publish"
            )
        sources = _component_field_sources(module, kind)
        schema = _KIND_SCHEMAS[kind]
        if len(sources) != len(schema):
            raise Glm52FullParamPayloadError(
                f"Component {target!r} produced {len(sources)} publishable tensors, "
                f"expected {len(schema)} for kind {kind!r}"
            )

        fields = []
        for (field_name, dtype_name), source in zip(schema, sources, strict=True):
            if source.dtype is not _DTYPES[dtype_name]:
                raise Glm52FullParamPayloadError(f"{target}.{field_name} must be {dtype_name}, got {source.dtype}")
            shape = tuple(int(value) for value in source.shape)
            # Immutable snapshot: detach the payload's lifetime from the
            # component cache the view aliases: a later refresh must never
            # rewrite already-checksummed bytes.
            data = _byte_view(source).detach().clone()
            header = _field_header(
                protocol_version=GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION,
                weight_version=weight_version,
                weight_step=weight_step,
                target=target,
                field_name=field_name,
                dtype=dtype_name,
                shape=shape,
            )
            fields.append(
                Glm52PayloadField(
                    name=field_name,
                    dtype=dtype_name,
                    shape=shape,
                    data=data,
                    checksum=_checksum(header, data),
                )
            )
        items.append(
            Glm52PayloadItem(
                target=target,
                kind=kind,
                contract_version=str(getattr(module, "contract_version", "")),
                fields=tuple(fields),
            )
        )

    if not items:
        raise Glm52FullParamPayloadError("Refusing to publish an empty GLM-5.2 full-param payload")

    payload = Glm52FullParamSyncPayload(
        protocol_version=GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION,
        weight_version=weight_version,
        weight_step=weight_step,
        items=tuple(items),
        manifest_checksum=_manifest_checksum(GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION, weight_version, weight_step, items),
    )
    _engagement_log(
        "GLM-5.2 full-param payload published: version=%s items=%d fields=%d manifest=%s",
        weight_version,
        len(payload.items),
        sum(len(item.fields) for item in payload.items),
        payload.manifest_checksum[:16],
    )
    return payload


# ---------------------------------------------------------------------------
# Verification and unpacking (receiver side)
# ---------------------------------------------------------------------------


def verify_glm52_fullparam_payload(payload: Glm52FullParamSyncPayload) -> None:
    """Recompute every checksum and validate the schema; RAISE on any mismatch.

    Nothing may be applied before this completes for the whole manifest.
    """

    if not isinstance(payload, Glm52FullParamSyncPayload):
        raise Glm52FullParamPayloadError("Payload must be a Glm52FullParamSyncPayload")
    if payload.protocol_version != GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION:
        raise Glm52FullParamPayloadError(
            f"Unsupported payload protocol {payload.protocol_version!r}; "
            f"this receiver implements {GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION!r}"
        )
    _validate_weight_identity(payload.weight_version, payload.weight_step, label="payload")
    if not isinstance(payload.items, tuple) or not payload.items:
        raise Glm52FullParamPayloadError("Payload contains no items")
    if len(payload.items) > _MAX_PAYLOAD_ITEMS:
        raise Glm52FullParamPayloadError(f"Payload contains too many items ({len(payload.items)})")

    seen_targets: set[str] = set()
    field_count = 0
    total_bytes = 0
    for item in payload.items:
        if not isinstance(item, Glm52PayloadItem):
            raise Glm52FullParamPayloadError("Payload contains an invalid item object")
        if not isinstance(item.target, str) or not item.target:
            raise Glm52FullParamPayloadError(f"Invalid payload target {item.target!r}")
        if item.target in seen_targets:
            raise Glm52FullParamPayloadError(f"Duplicate payload target {item.target!r}")
        seen_targets.add(item.target)
        schema = _KIND_SCHEMAS.get(item.kind)
        if schema is None:
            raise Glm52FullParamPayloadError(f"Unknown payload kind {item.kind!r} for target {item.target!r}")
        if not isinstance(item.contract_version, str) or not item.contract_version:
            raise Glm52FullParamPayloadError(f"Payload item {item.target!r} must declare a non-empty contract_version")
        if not isinstance(item.fields, tuple) or not all(isinstance(field, Glm52PayloadField) for field in item.fields):
            raise Glm52FullParamPayloadError(f"Payload item {item.target!r} fields must be payload-field objects")
        if tuple((field.name, field.dtype) for field in item.fields) != schema:
            raise Glm52FullParamPayloadError(
                f"Payload item {item.target!r} does not match the {item.kind!r} field schema"
            )
        for field in item.fields:
            if (
                not isinstance(field.shape, tuple)
                or not field.shape
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in field.shape)
            ):
                raise Glm52FullParamPayloadError(f"{item.target}.{field.name} has an invalid shape {field.shape!r}")
            if not isinstance(field.data, Tensor) or field.data.dtype is not torch.uint8 or field.data.dim() != 1:
                raise Glm52FullParamPayloadError(f"{item.target}.{field.name} payload must be a 1-D uint8 byte tensor")
            if not _is_sha256(field.checksum):
                raise Glm52FullParamPayloadError(f"{item.target}.{field.name} checksum must be lowercase SHA-256")
            expected_bytes = _DTYPE_ITEMSIZE[field.dtype]
            for value in field.shape:
                expected_bytes *= value
                if expected_bytes > _MAX_FIELD_BYTES:
                    raise Glm52FullParamPayloadError(
                        f"{item.target}.{field.name} exceeds the {_MAX_FIELD_BYTES}-byte field limit"
                    )
            field_count += 1
            total_bytes += expected_bytes
            if field_count > _MAX_PAYLOAD_FIELDS or total_bytes > _MAX_TOTAL_PAYLOAD_BYTES:
                raise Glm52FullParamPayloadError("Payload exceeds the admitted field-count or total-byte limit")
            if field.data.numel() != expected_bytes:
                raise Glm52FullParamPayloadError(
                    f"{item.target}.{field.name} carries {field.data.numel()} bytes, "
                    f"expected {expected_bytes} for {field.dtype} {field.shape}"
                )
            header = _field_header(
                protocol_version=payload.protocol_version,
                weight_version=payload.weight_version,
                weight_step=payload.weight_step,
                target=item.target,
                field_name=field.name,
                dtype=field.dtype,
                shape=field.shape,
            )
            actual = _checksum(header, field.data)
            if actual != field.checksum:
                raise Glm52FullParamPayloadError(
                    f"Checksum mismatch for {item.target}.{field.name}: "
                    f"manifest={field.checksum[:16]}… recomputed={actual[:16]}…; "
                    "refusing to apply any weight from this payload"
                )

    if not _is_sha256(payload.manifest_checksum):
        raise Glm52FullParamPayloadError("Payload manifest checksum must be lowercase SHA-256")
    expected_manifest = _manifest_checksum(
        payload.protocol_version, payload.weight_version, payload.weight_step, payload.items
    )
    if expected_manifest != payload.manifest_checksum:
        raise Glm52FullParamPayloadError(
            "Payload manifest checksum mismatch; refusing to apply any weight from this payload"
        )


def _verify_expected_inventory(
    payload: Glm52FullParamSyncPayload,
    expected_inventory: Glm52ExpectedPayloadInventory,
) -> None:
    """Require an exact receiver-owned semantic inventory for ``payload``."""

    if not isinstance(expected_inventory, Glm52ExpectedPayloadInventory):
        raise Glm52FullParamPayloadError(
            "apply requires an independent Glm52ExpectedPayloadInventory owned by the receiver"
        )
    if not expected_inventory.items:
        raise Glm52FullParamPayloadError("Expected payload inventory must contain at least one item")

    expected_by_target: dict[str, Glm52ExpectedPayloadItem] = {}
    for item in expected_inventory.items:
        if not isinstance(item, Glm52ExpectedPayloadItem):
            raise Glm52FullParamPayloadError("Expected payload inventory contains an invalid item object")
        if not isinstance(item.target, str) or not item.target:
            raise Glm52FullParamPayloadError(f"Invalid expected payload target {item.target!r}")
        if item.target in expected_by_target:
            raise Glm52FullParamPayloadError(f"Duplicate expected payload target {item.target!r}")
        schema = _KIND_SCHEMAS.get(item.kind)
        if schema is None:
            raise Glm52FullParamPayloadError(f"Unknown expected payload kind {item.kind!r} for target {item.target!r}")
        if not isinstance(item.contract_version, str) or not item.contract_version:
            raise Glm52FullParamPayloadError(
                f"Expected payload item {item.target!r} must declare a non-empty contract_version"
            )
        if not isinstance(item.fields, tuple) or not all(
            isinstance(field, Glm52ExpectedPayloadField) for field in item.fields
        ):
            raise Glm52FullParamPayloadError(f"Expected payload item {item.target!r} has an invalid field inventory")
        if tuple((field.name, field.dtype) for field in item.fields) != schema:
            raise Glm52FullParamPayloadError(
                f"Expected payload item {item.target!r} does not match the {item.kind!r} field schema"
            )
        for field in item.fields:
            if (
                not isinstance(field.shape, tuple)
                or not field.shape
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in field.shape)
            ):
                raise Glm52FullParamPayloadError(
                    f"Expected payload field {item.target}.{field.name} has invalid shape {field.shape!r}"
                )
        expected_by_target[item.target] = item

    actual_targets = {item.target for item in payload.items}
    expected_targets = set(expected_by_target)
    if actual_targets != expected_targets:
        raise Glm52FullParamPayloadError(
            "Payload inventory target mismatch: "
            f"missing={sorted(expected_targets - actual_targets)}, "
            f"unexpected={sorted(actual_targets - expected_targets)}"
        )

    for item in payload.items:
        expected = expected_by_target[item.target]
        if item.kind != expected.kind:
            raise Glm52FullParamPayloadError(
                f"Payload kind mismatch for {item.target!r}: expected {expected.kind!r}, got {item.kind!r}"
            )
        if item.contract_version != expected.contract_version:
            raise Glm52FullParamPayloadError(
                f"Payload contract_version mismatch for {item.target!r}: "
                f"expected {expected.contract_version!r}, got {item.contract_version!r}"
            )
        actual_fields = tuple((field.name, field.dtype, tuple(field.shape)) for field in item.fields)
        expected_fields = tuple((field.name, field.dtype, tuple(field.shape)) for field in expected.fields)
        if actual_fields != expected_fields:
            raise Glm52FullParamPayloadError(
                f"Payload field inventory mismatch for {item.target!r}: "
                f"expected {expected_fields!r}, got {actual_fields!r}"
            )


def unpack_glm52_payload_field(field: Glm52PayloadField) -> Tensor:
    """Reinterpret verified bytes as the declared dtype/shape (no conversion)."""

    dtype = _DTYPES.get(field.dtype)
    if dtype is None:
        raise Glm52FullParamPayloadError(f"Unknown payload dtype {field.dtype!r}")
    return field.data.view(dtype).reshape(field.shape)


def _snapshot_receiver(item: Glm52PayloadItem, receiver: object) -> tuple:
    """Capture the receiver's current bytes so a failed commit can roll back.

    A receiver that cannot be snapshotted is rejected BEFORE any mutation:
    transactionality is guaranteed by refusing, never by best effort.
    """

    if item.kind == "block_fp8_linear":
        if not (hasattr(receiver, "fp8_weight") and hasattr(receiver, "weight_scale_inv")):
            raise Glm52FullParamPayloadError(
                f"Receiver for {item.target!r} cannot be snapshotted for transactional apply"
            )
        return (receiver.fp8_weight().clone(), receiver.weight_scale_inv.detach().clone())
    if item.kind == "block_fp8_expert_bank":
        for name in ("gate_up_proj", "gate_up_weight_scale_inv", "down_proj", "down_weight_scale_inv"):
            if not hasattr(receiver, name):
                raise Glm52FullParamPayloadError(
                    f"Receiver for {item.target!r} cannot be snapshotted for transactional apply"
                )
        return (
            receiver.gate_up_proj.clone(),
            receiver.gate_up_weight_scale_inv.detach().clone(),
            receiver.down_proj.clone(),
            receiver.down_weight_scale_inv.detach().clone(),
        )
    if item.kind == "bf16_router":
        if isinstance(receiver, Tensor):
            return (receiver.detach().clone(),)
        raise Glm52FullParamPayloadError(
            f"Router receiver for {item.target!r} cannot be snapshotted for transactional apply"
        )
    if item.kind == "block_fp8_dense_mlp":
        if not hasattr(receiver, "checkpoint_split_bytes"):
            raise Glm52FullParamPayloadError(
                f"Receiver for {item.target!r} cannot be snapshotted for transactional apply"
            )
        return tuple(receiver.checkpoint_split_bytes())
    if item.kind == "block_fp8_expert":
        if not hasattr(receiver, "checkpoint_expert_bytes"):
            raise Glm52FullParamPayloadError(
                f"Receiver for {item.target!r} cannot be snapshotted for transactional apply"
            )
        return tuple(receiver.checkpoint_expert_bytes())
    raise Glm52FullParamPayloadError(f"Unknown payload kind {item.kind!r}")


def _receiver_inventory_geometry(kind: str, receiver: object) -> tuple[tuple[torch.dtype, tuple[int, ...]], ...]:
    """Inspect receiver geometry without copying any receiver weight bytes."""

    def geometry(tensor: object, *, label: str) -> tuple[torch.dtype, tuple[int, ...]]:
        if not isinstance(tensor, Tensor):
            raise Glm52FullParamPayloadError(f"{label} must be a tensor")
        return tensor.dtype, tuple(int(value) for value in tensor.shape)

    if kind == "block_fp8_linear":
        if not (hasattr(receiver, "fp8_weight") and hasattr(receiver, "weight_scale_inv")):
            raise Glm52FullParamPayloadError("block_fp8_linear receiver lacks its inventory tensors")
        return (
            geometry(receiver.fp8_weight(), label="block_fp8_linear weight"),
            geometry(receiver.weight_scale_inv, label="block_fp8_linear scale"),
        )
    if kind == "bf16_router":
        if not isinstance(receiver, Tensor):
            raise Glm52FullParamPayloadError("bf16_router inventory receiver must be a tensor")
        return (geometry(receiver, label="bf16_router weight"),)
    if kind == "block_fp8_expert_bank":
        names = ("gate_up_proj", "gate_up_weight_scale_inv", "down_proj", "down_weight_scale_inv")
        if not all(hasattr(receiver, name) for name in names):
            raise Glm52FullParamPayloadError("block_fp8_expert_bank receiver lacks its inventory tensors")
        return tuple(geometry(getattr(receiver, name), label=f"block_fp8_expert_bank {name}") for name in names)
    if kind == "block_fp8_dense_mlp":
        if not all(hasattr(receiver, name) for name in ("gate_up_proj", "down_proj", "intermediate_size")):
            raise Glm52FullParamPayloadError("block_fp8_dense_mlp receiver lacks its structural inventory")
        fused_weight = receiver.gate_up_proj.fp8_weight()
        fused_scale = receiver.gate_up_proj.weight_scale_inv
        intermediate = receiver.intermediate_size
        scale_rows = intermediate // 128
        return (
            geometry(fused_weight[:intermediate], label="block_fp8_dense_mlp gate"),
            geometry(fused_scale[:scale_rows], label="block_fp8_dense_mlp gate scale"),
            geometry(fused_weight[intermediate:], label="block_fp8_dense_mlp up"),
            geometry(fused_scale[scale_rows:], label="block_fp8_dense_mlp up scale"),
            geometry(receiver.down_proj.fp8_weight(), label="block_fp8_dense_mlp down"),
            geometry(receiver.down_proj.weight_scale_inv, label="block_fp8_dense_mlp down scale"),
        )
    if kind == "block_fp8_expert":
        if not all(hasattr(receiver, name) for name in ("bank", "local_slot")):
            raise Glm52FullParamPayloadError("block_fp8_expert receiver lacks its structural inventory")
        bank, slot = receiver.bank, receiver.local_slot
        intermediate = bank.intermediate_size
        scale_blocks = intermediate // 128
        gate_up = bank.gate_up_proj[slot]
        gate_up_scale = bank.gate_up_weight_scale_inv[slot]
        down = bank.down_proj[slot]
        down_scale = bank.down_weight_scale_inv[slot]
        return (
            geometry(gate_up[:, :intermediate].t(), label="block_fp8_expert gate"),
            geometry(gate_up_scale[:, :scale_blocks].t(), label="block_fp8_expert gate scale"),
            geometry(gate_up[:, intermediate:].t(), label="block_fp8_expert up"),
            geometry(gate_up_scale[:, scale_blocks:].t(), label="block_fp8_expert up scale"),
            geometry(down.t(), label="block_fp8_expert down"),
            geometry(down_scale.t(), label="block_fp8_expert down scale"),
        )
    raise Glm52FullParamPayloadError(f"Unknown receiver inventory kind {kind!r}")


def build_glm52_expected_payload_inventory(
    receiver_specs: Iterable[tuple[str, str, str, object]],
) -> Glm52ExpectedPayloadInventory:
    """Build an exact inventory from receiver-owned tensor geometry.

    Each spec is ``(target, kind, admitted_contract_version, receiver)``. The
    target/kind/contract come from trusted model admission; dtype and shape are
    read from the already-constructed serving receiver. No payload is accepted
    by this constructor, preventing a checksum-valid subset from defining its
    own expected scope.
    """

    items: list[Glm52ExpectedPayloadItem] = []
    seen_targets: set[str] = set()
    for spec in receiver_specs:
        if not isinstance(spec, tuple) or len(spec) != 4:
            raise Glm52FullParamPayloadError(
                "receiver inventory specs must be (target, kind, contract_version, receiver) tuples"
            )
        target, kind, contract_version, receiver = spec
        if not isinstance(target, str) or not target or target in seen_targets:
            raise Glm52FullParamPayloadError(f"Invalid or duplicate receiver inventory target {target!r}")
        if kind not in _KIND_SCHEMAS:
            raise Glm52FullParamPayloadError(f"Unknown receiver inventory kind {kind!r} for {target!r}")
        if not isinstance(contract_version, str) or not contract_version:
            raise Glm52FullParamPayloadError(f"Receiver inventory target {target!r} needs a contract_version")
        geometry = _receiver_inventory_geometry(kind, receiver)
        schema = _KIND_SCHEMAS[kind]
        if len(geometry) != len(schema):
            raise Glm52FullParamPayloadError(
                f"Receiver inventory target {target!r} exposes {len(geometry)} tensors, expected {len(schema)}"
            )
        fields: list[Glm52ExpectedPayloadField] = []
        for (name, dtype_name), (dtype, shape) in zip(schema, geometry, strict=True):
            if dtype is not _DTYPES[dtype_name]:
                raise Glm52FullParamPayloadError(
                    f"Receiver inventory field {target}.{name} must be {dtype_name}, got {dtype}"
                )
            if not shape or any(value <= 0 for value in shape):
                raise Glm52FullParamPayloadError(f"Receiver inventory field {target}.{name} has invalid shape {shape}")
            fields.append(Glm52ExpectedPayloadField(name=name, dtype=dtype_name, shape=shape))
        items.append(
            Glm52ExpectedPayloadItem(
                target=target,
                kind=kind,
                contract_version=contract_version,
                fields=tuple(fields),
            )
        )
        seen_targets.add(target)
    inventory = Glm52ExpectedPayloadInventory(items=tuple(items))
    if not inventory.items:
        raise Glm52FullParamPayloadError("Receiver inventory must contain at least one target")
    return inventory


# Receiver method implementing the loader's consumption, per kind.
_KIND_APPLY_METHODS: dict[str, str] = {
    "block_fp8_linear": "load_prequantized",
    "block_fp8_expert_bank": "load_prequantized",
    "block_fp8_dense_mlp": "load_prequantized_split",
    "block_fp8_expert": "load_checkpoint_expert",
}


def _apply_one(item: Glm52PayloadItem, receiver: object, tensors: tuple[Tensor, ...]) -> None:
    method = _KIND_APPLY_METHODS.get(item.kind)
    if method is not None:
        getattr(receiver, method)(*tensors)
    elif item.kind == "bf16_router":
        with torch.no_grad():
            receiver.copy_(tensors[0])
    else:  # pragma: no cover - verify() rejects unknown kinds
        raise Glm52FullParamPayloadError(f"Unknown payload kind {item.kind!r}")


def _apply_glm52_fullparam_payload_transaction_body(
    payload: Glm52FullParamSyncPayload,
    resolve_target: Callable[[str, str], object],
    *,
    expected_inventory: Glm52ExpectedPayloadInventory,
) -> None:
    """Verify the whole manifest, stage every item, then apply all of them.

    ``resolve_target(target, kind)`` returns the receiver object for one item:

    - ``block_fp8_linear`` / ``block_fp8_expert_bank``: an object exposing the
      serving ``load_prequantized`` contract;
    - ``bf16_router``: a BF16 tensor to copy into, or an object exposing
      ``load_from_bf16``.

    ``expected_inventory`` is the complete semantic inventory admitted by the
    receiver and must be constructed independently of this payload.  Phase 1
    verifies checksums and inventory, then resolves/stages every target without
    mutating anything; phase 2 applies.  A phase-1 failure therefore leaves the
    receiver untouched.  The engine must not serve between the phases —
    scheduling that pause is the engine integration's obligation.
    """

    verify_glm52_fullparam_payload(payload)
    _verify_expected_inventory(payload, expected_inventory)

    staged: list[tuple[Glm52PayloadItem, object, tuple[Tensor, ...], tuple]] = []
    for item in payload.items:
        receiver = resolve_target(item.target, item.kind)
        if receiver is None:
            raise Glm52FullParamPayloadError(f"No receiver resolved for payload target {item.target!r}")
        tensors = tuple(unpack_glm52_payload_field(field) for field in item.fields)
        if item.kind in _KIND_APPLY_METHODS:
            method = _KIND_APPLY_METHODS[item.kind]
            if not hasattr(receiver, method):
                raise Glm52FullParamPayloadError(f"Receiver for {item.target!r} does not expose {method}")
        elif item.kind == "bf16_router":
            if isinstance(receiver, Tensor):
                if receiver.dtype is not torch.bfloat16 or tuple(receiver.shape) != tensors[0].shape:
                    raise Glm52FullParamPayloadError(
                        f"Router receiver for {item.target!r} must be BF16 {tuple(tensors[0].shape)}, "
                        f"got {receiver.dtype} {tuple(receiver.shape)}"
                    )
            else:
                raise Glm52FullParamPayloadError(
                    f"Router receiver for {item.target!r} must be a BF16 tensor for transactional apply"
                )
        snapshot = _snapshot_receiver(item, receiver)
        staged.append((item, receiver, tensors, snapshot))

    applied: list[tuple[Glm52PayloadItem, object, tuple]] = []
    try:
        for item, receiver, tensors, snapshot in staged:
            # Register for rollback BEFORE the loader runs: a loader that
            # mutates and then throws must itself be rolled back.
            applied.append((item, receiver, snapshot))
            _apply_one(item, receiver, tensors)
    except BaseException as commit_error:
        rollback_failures: list[str] = []
        for r_item, r_receiver, r_snapshot in reversed(applied):
            try:
                _apply_one(r_item, r_receiver, r_snapshot)
            except BaseException:
                rollback_failures.append(r_item.target)
        if rollback_failures:
            raise Glm52FullParamPayloadError(
                f"Transactional apply failed on {item.target!r} AND rollback failed; receivers in "
                f"UNKNOWN state: {rollback_failures} — do not serve until reloaded from a known "
                f"checkpoint. Original error: {commit_error}"
            ) from commit_error
        raise Glm52FullParamPayloadError(
            f"Transactional apply failed on {item.target!r} and was rolled back: {commit_error}"
        ) from commit_error
    # Version commit is owned by the caller's serialized guard context.


def apply_glm52_fullparam_payload(
    payload: Glm52FullParamSyncPayload,
    resolve_target: Callable[[str, str], object],
    *,
    expected_inventory: Glm52ExpectedPayloadInventory,
    version_guard: "Glm52WeightVersionGuard",
) -> None:
    """Atomically apply one payload under a guard-owned serialized transaction.

    Checksum/inventory verification, target resolution, rollback snapshots,
    mutation, and monotonic version commit all occur while the receiver's one
    persistent guard lock is held. The serving integration must use the same
    guard for every update and must pause request execution around this call.
    """

    if not isinstance(version_guard, Glm52WeightVersionGuard):
        raise Glm52FullParamPayloadError(
            "apply requires the receiver's ONE persistent Glm52WeightVersionGuard; "
            "weight mutation without monotonic version enforcement is not admitted"
        )
    _validate_weight_identity(payload.weight_version, payload.weight_step, label="payload")
    with version_guard.serialized_update(
        weight_step=payload.weight_step,
        weight_version=payload.weight_version,
    ):
        _apply_glm52_fullparam_payload_transaction_body(
            payload,
            resolve_target,
            expected_inventory=expected_inventory,
        )

    try:
        _engagement_log(
            "GLM-5.2 full-param payload applied: version=%s items=%d manifest=%s",
            payload.weight_version,
            len(payload.items),
            payload.manifest_checksum[:16],
        )
    except BaseException:
        # Mutation and guard commit already succeeded; never report a
        # rollback-safe failure because an engagement logger is misconfigured.
        pass


def glm52_fullparam_hf_name_mapping(payload: Glm52FullParamSyncPayload) -> dict[str, dict[str, str]]:
    """Mechanical payload-target -> HF-tensor-name mapping for the disk route.

    Valid when payload targets are the admission's publication FQNs:
    checkpoint-form kinds map field-by-field onto
    ``{target}.{gate|up|down}_proj.{weight|weight_scale_inv}``; router
    admission targets may contain the internal terminal ``.full_param``
    wrapper and map back onto the checkpoint's ``{gate_target}.weight``;
    plain block-FP8 linears map onto ``{target}.weight`` /
    ``{target}.weight_scale_inv``.  The fused ``block_fp8_expert_bank`` kind
    has no checkpoint form and RAISES — publish banks through their per-expert
    checkpoint items instead.
    """

    verify_glm52_fullparam_payload(payload)
    mapping: dict[str, dict[str, str]] = {}
    destination_owner: dict[str, tuple[str, str]] = {}
    for item in payload.items:
        if item.kind in ("block_fp8_dense_mlp", "block_fp8_expert"):
            mapping[item.target] = {
                field.name: f"{item.target}.{_CHECKPOINT_FIELD_TO_HF_SUFFIX[field.name]}" for field in item.fields
            }
        elif item.kind == "bf16_router":
            # Training replaces the checkpoint's ``gate.weight`` with a
            # wrapper whose trainable publication unit is the nested
            # ``gate.full_param`` module.  Fold only that terminal wrapper
            # segment back to the checkpoint FQN; direct checkpoint-form
            # router targets remain unchanged.
            checkpoint_target = item.target.removesuffix(".full_param")
            mapping[item.target] = {"weight": f"{checkpoint_target}.weight"}
        elif item.kind == "block_fp8_linear":
            mapping[item.target] = {
                "weight": f"{item.target}.weight",
                "weight_scale_inv": f"{item.target}.weight_scale_inv",
            }
        else:
            raise Glm52FullParamPayloadError(
                f"Payload item {item.target!r} of kind {item.kind!r} has no HF checkpoint form; "
                "publish expert banks as per-expert checkpoint items for the disk route"
            )
        for field_name, destination in mapping[item.target].items():
            previous = destination_owner.get(destination)
            if previous is not None:
                raise Glm52FullParamPayloadError(
                    f"Canonical HF destination collision for {destination!r}: "
                    f"{previous[0]}.{previous[1]} and {item.target}.{field_name}"
                )
            destination_owner[destination] = (item.target, field_name)
    return mapping


def prepare_glm52_fullparam_payload_hf_checkpoint(
    payload: Glm52FullParamSyncPayload,
    base_checkpoint_dir: str,
    output_dir: str,
    target_to_hf_names: dict[str, dict[str, str]],
    *,
    expected_inventory: Glm52ExpectedPayloadInventory,
) -> Glm52PreparedCheckpoint:
    """Materialize a verified payload as an inert HF FP8 checkpoint.

    This function does **not** advance a serving version guard. The caller must
    pass the returned object to :func:`activate_glm52_prepared_checkpoint`
    together with its engine reload/swap callback. The whole manifest is
    verified first; the base checkpoint's safetensors state is then rewritten
    with the payload bytes substituted tensor-for-tensor.
    ``target_to_hf_names`` maps each payload target to
    ``{field_name: hf_tensor_name}``; every mapped HF tensor must exist in
    the base checkpoint with the payload field's dtype and shape — any
    mismatch RAISES before a single byte is written.  ``expected_inventory``
    must independently enumerate the complete target, contract, dtype, and
    shape scope admitted by the receiver.  safetensors preserves dtypes and
    raw bytes exactly (no pickle, no casts).

    The base directory must be immutable under trusted ACLs for this operation:
    symlink and special-file entries are rejected, but a hostile actor able to
    replace a validated shard concurrently is outside the process-local threat
    boundary. Every staged file is fsynced and one no-replace rename is the
    publication commit point. No rollback-style exception is raised after that
    rename succeeds.
    """

    import shutil  # noqa: PLC0415

    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file as load_safetensors  # noqa: PLC0415
    from safetensors.torch import save_file as save_safetensors  # noqa: PLC0415

    verify_glm52_fullparam_payload(payload)
    _verify_expected_inventory(payload, expected_inventory)

    canonical_hf_names = glm52_fullparam_hf_name_mapping(payload)
    if target_to_hf_names != canonical_hf_names:
        raise Glm52FullParamPayloadError(
            "target_to_hf_names must match the canonical payload-to-checkpoint mapping exactly; "
            "refusing a caller-directed tensor remap"
        )

    base_checkpoint_dir = _canonical_existing_directory(base_checkpoint_dir, label="Base checkpoint directory")
    output_absolute = os.path.abspath(output_dir)
    if output_absolute == base_checkpoint_dir:
        raise Glm52FullParamPayloadError("Refusing to rewrite the base checkpoint in place")
    output_dir, output_parent = _canonical_absent_output(output_dir, label="Checkpoint output directory")
    try:
        if os.path.commonpath((base_checkpoint_dir, output_parent)) == base_checkpoint_dir:
            raise Glm52FullParamPayloadError("Checkpoint output must not be nested beneath the base checkpoint")
    except ValueError as exc:
        raise Glm52FullParamPayloadError("Base and output checkpoint paths must share a local filesystem root") from exc

    base_entries = sorted(os.listdir(base_checkpoint_dir))
    for entry in base_entries:
        _plain_filename(entry, label="Base checkpoint entry")
        metadata = os.lstat(os.path.join(base_checkpoint_dir, entry))
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise Glm52FullParamPayloadError(f"Base checkpoint entry {entry!r} must be a regular nonsymlink file")
    weight_files = [name for name in base_entries if name.endswith(".safetensors")]
    if not weight_files:
        raise Glm52FullParamPayloadError(f"No safetensors files found in {base_checkpoint_dir!r}")

    # Tensor -> shard mapping from the index when present (streaming; the
    # 700-GB production checkpoint must never be resident), else by scanning
    # shard headers.
    index_filename = "model.safetensors.index.json"
    tensor_to_file: dict[str, str] = {}
    index_descriptor = os.open(
        base_checkpoint_dir,
        os.O_RDONLY | os.O_DIRECTORY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
    )
    try:
        if index_filename in base_entries:
            raw_index = _read_regular_at(
                index_descriptor,
                index_filename,
                label="HF safetensors index",
                max_bytes=_MAX_MANIFEST_BYTES,
            )
            index = _json_no_duplicate_keys(raw_index, label="HF safetensors index")
            if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
                raise Glm52FullParamPayloadError("HF safetensors index must contain an object weight_map")
            weight_map = index["weight_map"]
            assert isinstance(weight_map, dict)
            for tensor_name, filename in weight_map.items():
                if not isinstance(tensor_name, str) or not tensor_name:
                    raise Glm52FullParamPayloadError("HF index tensor names must be non-empty strings")
                filename = _plain_filename(filename, label=f"HF index shard for {tensor_name!r}")
                if not filename.endswith(".safetensors") or filename not in weight_files:
                    raise Glm52FullParamPayloadError(f"HF index maps {tensor_name!r} to unlisted shard {filename!r}")
                tensor_to_file[tensor_name] = filename
            referenced_shards = set(tensor_to_file.values())
            if referenced_shards != set(weight_files):
                raise Glm52FullParamPayloadError(
                    "HF index shard set must exactly equal the listed regular safetensors files: "
                    f"unreferenced={sorted(set(weight_files) - referenced_shards)}"
                )
        else:
            for filename in weight_files:
                with safe_open(os.path.join(base_checkpoint_dir, filename), framework="pt") as reader:
                    for name in reader.keys():
                        if name in tensor_to_file:
                            raise Glm52FullParamPayloadError(f"Duplicate tensor {name!r} across checkpoint shards")
                        tensor_to_file[name] = filename
    finally:
        os.close(index_descriptor)

    payload_targets = {item.target for item in payload.items}
    if set(target_to_hf_names) != payload_targets:
        raise Glm52FullParamPayloadError(
            "target_to_hf_names must cover exactly the payload targets: "
            f"payload={sorted(payload_targets)}, mapping={sorted(target_to_hf_names)}"
        )

    staged_by_file: dict[str, dict[str, Tensor]] = {}
    planned_replacements: set[tuple[str, str]] = set()
    for item in payload.items:
        name_map = target_to_hf_names[item.target]
        if set(name_map) != {field.name for field in item.fields}:
            raise Glm52FullParamPayloadError(
                f"HF name mapping for {item.target!r} must cover exactly its fields "
                f"{[field.name for field in item.fields]}, got {sorted(name_map)}"
            )
        for field in item.fields:
            hf_name = name_map[field.name]
            filename = tensor_to_file.get(hf_name)
            if filename is None:
                raise Glm52FullParamPayloadError(
                    f"{item.target}.{field.name} maps to {hf_name!r}, absent from the base checkpoint"
                )
            with safe_open(os.path.join(base_checkpoint_dir, filename), framework="pt") as reader:
                existing = reader.get_tensor(hf_name)
            replacement = unpack_glm52_payload_field(field)
            if existing.dtype is not replacement.dtype or tuple(existing.shape) != tuple(replacement.shape):
                raise Glm52FullParamPayloadError(
                    f"{hf_name!r} is {existing.dtype} {tuple(existing.shape)} in the base checkpoint but the "
                    f"payload carries {replacement.dtype} {tuple(replacement.shape)}; refusing dtype/shape drift"
                )
            key = (filename, hf_name)
            if key in planned_replacements:
                raise Glm52FullParamPayloadError(f"Duplicate staged HF replacement {hf_name!r}")
            planned_replacements.add(key)
            staged_by_file.setdefault(filename, {})[hf_name] = replacement

    prepared = Glm52PreparedCheckpoint(
        directory=output_dir,
        weight_version=payload.weight_version,
        weight_step=payload.weight_step,
        manifest_checksum=payload.manifest_checksum,
    )
    staging_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(output_dir)}.staging-",
        dir=output_parent,
    )
    base_descriptor = -1
    staging_descriptor = -1
    try:
        base_descriptor = os.open(
            base_checkpoint_dir,
            os.O_RDONLY | os.O_DIRECTORY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
        )
        staging_descriptor = os.open(
            staging_dir,
            os.O_RDONLY | os.O_DIRECTORY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
        )
        for entry in base_entries:
            if entry.endswith(".safetensors"):
                continue
            source_descriptor, _ = _open_regular_at(
                base_descriptor,
                entry,
                label="Base checkpoint metadata",
                max_bytes=_MAX_FIELD_BYTES,
            )
            destination_descriptor = os.open(
                entry,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
                0o600,
                dir_fd=staging_descriptor,
            )
            try:
                while True:
                    chunk = os.read(source_descriptor, 8 * 1024 * 1024)
                    if not chunk:
                        break
                    _write_all(destination_descriptor, chunk)
                os.fsync(destination_descriptor)
            finally:
                os.close(source_descriptor)
                os.close(destination_descriptor)

        emitted_replacements: set[tuple[str, str]] = set()
        for filename in weight_files:
            source = os.path.join(base_checkpoint_dir, filename)
            destination = os.path.join(staging_dir, filename)
            staged = staged_by_file.get(filename)
            if staged is None:
                # Untouched shard: hardlink (byte identity by construction). A
                # cross-filesystem output would silently degrade to a 700-GB
                # copy, so it fails closed instead.
                try:
                    os.link(
                        filename,
                        filename,
                        src_dir_fd=base_descriptor,
                        dst_dir_fd=staging_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise Glm52FullParamPayloadError(
                        f"Cannot hardlink untouched shard {filename!r} into {output_dir!r} "
                        "(same-filesystem output required for streaming updates)"
                    ) from exc
                linked = os.stat(filename, dir_fd=staging_descriptor, follow_symlinks=False)
                if not stat.S_ISREG(linked.st_mode):
                    raise Glm52FullParamPayloadError(
                        f"Untouched shard {filename!r} changed into a symlink or special file during publication"
                    )
                continue
            shard_state = load_safetensors(source)
            for hf_name, replacement in staged.items():
                if hf_name not in shard_state:
                    raise Glm52FullParamPayloadError(
                        f"Staged replacement {hf_name!r} disappeared from shard {filename!r}"
                    )
                shard_state[hf_name] = replacement.detach().cpu().contiguous()
            save_safetensors(
                {name: tensor.detach().cpu().contiguous() for name, tensor in shard_state.items()},
                destination,
            )
            with safe_open(destination, framework="pt") as reader:
                saved_names = set(reader.keys())
                for hf_name, replacement in staged.items():
                    if hf_name not in saved_names:
                        raise Glm52FullParamPayloadError(
                            f"Rewritten shard {filename!r} omitted staged replacement {hf_name!r}"
                        )
                    saved = reader.get_tensor(hf_name)
                    expected = replacement.detach().cpu().contiguous()
                    if (
                        saved.dtype is not expected.dtype
                        or tuple(saved.shape) != tuple(expected.shape)
                        or not torch.equal(saved.view(torch.uint8), expected.view(torch.uint8))
                    ):
                        raise Glm52FullParamPayloadError(
                            f"Rewritten shard {filename!r} did not emit exact bytes for {hf_name!r}"
                        )
                    emitted_replacements.add((filename, hf_name))
            saved_descriptor, _ = _open_regular_at(
                staging_descriptor,
                filename,
                label="Rewritten safetensors shard",
                max_bytes=_MAX_FIELD_BYTES,
            )
            try:
                os.fsync(saved_descriptor)
            finally:
                os.close(saved_descriptor)

        if emitted_replacements != planned_replacements:
            raise Glm52FullParamPayloadError(
                "Not every staged HF replacement was emitted: "
                f"missing={sorted(planned_replacements - emitted_replacements)[:8]}"
            )
        os.fsync(staging_descriptor)
        os.close(staging_descriptor)
        staging_descriptor = -1
        os.close(base_descriptor)
        base_descriptor = -1
        _rename_noreplace(staging_dir, output_dir)
    except BaseException:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        if base_descriptor >= 0:
            os.close(base_descriptor)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # The rename above is the commit point. Everything below is explicitly
    # non-throwing so a visible complete checkpoint is never reported as a
    # rollback-safe failure.
    _fsync_parent_after_commit_noexcept(output_parent)
    try:
        _engagement_log(
            "GLM-5.2 full-param payload prepared as HF checkpoint: version=%s items=%d dir=%s manifest=%s",
            payload.weight_version,
            len(payload.items),
            output_dir,
            payload.manifest_checksum[:16],
        )
    except BaseException:
        pass
    return prepared


def apply_glm52_fullparam_payload_to_hf_checkpoint(
    payload: Glm52FullParamSyncPayload,
    base_checkpoint_dir: str,
    output_dir: str,
    target_to_hf_names: dict[str, dict[str, str]],
    *,
    expected_inventory: Glm52ExpectedPayloadInventory,
) -> Glm52PreparedCheckpoint:
    """Compatibility name for checkpoint materialization; it does not activate."""

    return prepare_glm52_fullparam_payload_hf_checkpoint(
        payload,
        base_checkpoint_dir,
        output_dir,
        target_to_hf_names,
        expected_inventory=expected_inventory,
    )


def activate_glm52_prepared_checkpoint(
    prepared: Glm52PreparedCheckpoint,
    activate_checkpoint: Callable[[str], None],
    *,
    version_guard: Glm52WeightVersionGuard,
) -> None:
    """Serialize engine activation and advance the guard only after success.

    ``activate_checkpoint`` must pause serving and implement an all-or-fail
    reload/swap for ``prepared.directory``. If it raises after mutating engine
    state, serving state is unknown and the guard intentionally remains at the
    prior version; the integration must reload a known checkpoint before
    serving again.
    """

    if not isinstance(prepared, Glm52PreparedCheckpoint):
        raise Glm52FullParamPayloadError("activation requires a Glm52PreparedCheckpoint")
    if not isinstance(version_guard, Glm52WeightVersionGuard):
        raise Glm52FullParamPayloadError("activation requires the receiver's persistent version guard")
    if not callable(activate_checkpoint):
        raise Glm52FullParamPayloadError("activate_checkpoint must be callable")
    _validate_weight_identity(prepared.weight_version, prepared.weight_step, label="prepared checkpoint")
    if not _is_sha256(prepared.manifest_checksum):
        raise Glm52FullParamPayloadError("Prepared checkpoint manifest_checksum must be lowercase SHA-256")
    directory = _canonical_existing_directory(prepared.directory, label="Prepared checkpoint directory")
    with version_guard.serialized_update(
        weight_step=prepared.weight_step,
        weight_version=prepared.weight_version,
    ):
        try:
            activate_checkpoint(directory)
        except Exception as exc:
            raise Glm52FullParamPayloadError(
                "Checkpoint activation callback failed; version guard was not advanced and serving state may be UNKNOWN"
            ) from exc

    try:
        _engagement_log(
            "GLM-5.2 full-param checkpoint activated: version=%s dir=%s manifest=%s",
            prepared.weight_version,
            directory,
            prepared.manifest_checksum[:16],
        )
    except BaseException:
        pass


def merge_glm52_fullparam_payloads(
    payloads: Iterable[Glm52FullParamSyncPayload],
) -> Glm52FullParamSyncPayload:
    """Combine per-rank partial payloads of ONE step into a single manifest.

    Under EP, expert items live on their owner ranks while dense/router items
    are published by rank 0 only; the disk route needs one manifest covering
    all of them.  Field checksums bind (protocol, version, step, target,
    field, dtype, shape) — none of which changes under the merge — so the
    per-rank checksums remain valid verbatim; only the manifest checksum is
    re-minted over the combined, deterministically ordered item list.

    Fail closed: every input must share the same protocol/version/step,
    targets must be globally disjoint, and the merged payload is fully
    re-verified (every field checksum recomputed) before it is returned.
    """

    materialized = list(payloads)
    if not materialized:
        raise Glm52FullParamPayloadError("merge requires at least one partial payload")
    for payload in materialized:
        verify_glm52_fullparam_payload(payload)
    head = materialized[0]
    for candidate in materialized[1:]:
        for attribute in ("protocol_version", "weight_version", "weight_step"):
            if getattr(candidate, attribute) != getattr(head, attribute):
                raise Glm52FullParamPayloadError(
                    f"Refusing to merge payloads with mismatched {attribute}: "
                    f"{getattr(head, attribute)!r} vs {getattr(candidate, attribute)!r}"
                )

    items: list[Glm52PayloadItem] = []
    seen_targets: set[str] = set()
    for payload in materialized:
        for item in payload.items:
            if item.target in seen_targets:
                raise Glm52FullParamPayloadError(
                    f"Duplicate payload target {item.target!r} across merged partial payloads"
                )
            seen_targets.add(item.target)
            items.append(item)
    items.sort(key=lambda item: item.target)

    merged = Glm52FullParamSyncPayload(
        protocol_version=head.protocol_version,
        weight_version=head.weight_version,
        weight_step=head.weight_step,
        items=tuple(items),
        manifest_checksum=_manifest_checksum(head.protocol_version, head.weight_version, head.weight_step, items),
    )
    verify_glm52_fullparam_payload(merged)
    _engagement_log(
        "GLM-5.2 full-param payloads merged: version=%s partials=%d items=%d manifest=%s",
        merged.weight_version,
        len(materialized),
        len(merged.items),
        merged.manifest_checksum[:16],
    )
    return merged


# ---------------------------------------------------------------------------
# File serialization (raw bytes + JSON manifest; no pickle, no dtype ambiguity)
# ---------------------------------------------------------------------------


def _field_filename(item_index: int, field_index: int, field_name: str) -> str:
    return f"{item_index:04d}_{field_index}_{field_name}.bin"


def save_glm52_fullparam_payload(payload: Glm52FullParamSyncPayload, directory: str) -> None:
    """Atomically write a verified payload as raw bytes plus JSON.

    The final directory must be absent. Files are created exclusively with
    ``O_NOFOLLOW``, fsynced, reloaded through the strict verifier, and then
    published from a random 0700 sibling staging directory with one no-replace
    rename.
    """

    import shutil  # noqa: PLC0415

    verify_glm52_fullparam_payload(payload)
    directory, parent = _canonical_absent_output(directory, label="Payload output directory", create_parent=True)

    manifest_items = []
    for item_index, item in enumerate(payload.items):
        manifest_items.append(
            {
                "target": item.target,
                "kind": item.kind,
                "contract_version": item.contract_version,
                "fields": [
                    {
                        "name": field.name,
                        "dtype": field.dtype,
                        "shape": list(field.shape),
                        "file": _field_filename(item_index, field_index, field.name),
                        "checksum": field.checksum,
                    }
                    for field_index, field in enumerate(item.fields)
                ],
            }
        )
    manifest_bytes = json.dumps(
        {
            "protocol_version": payload.protocol_version,
            "weight_version": payload.weight_version,
            "weight_step": payload.weight_step,
            "manifest_checksum": payload.manifest_checksum,
            "items": manifest_items,
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise Glm52FullParamPayloadError("Payload manifest exceeds the loader's bounded size limit")

    staging_dir = tempfile.mkdtemp(prefix=f".{os.path.basename(directory)}.staging-", dir=parent)
    staging_descriptor = -1
    try:
        staging_descriptor = os.open(
            staging_dir,
            os.O_RDONLY | os.O_DIRECTORY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
        )
        for item_index, item in enumerate(payload.items):
            for field_index, field in enumerate(item.fields):
                filename = _field_filename(item_index, field_index, field.name)
                data = field.data.detach().cpu().contiguous().numpy()
                _create_regular_at(staging_descriptor, filename, data)
        # Manifest last: an interrupted staging tree can never look complete.
        _create_regular_at(staging_descriptor, _MANIFEST_FILENAME, manifest_bytes)
        os.fsync(staging_descriptor)
        # Exercise the exact public loader before making this directory visible.
        load_glm52_fullparam_payload(staging_dir)
        os.close(staging_descriptor)
        staging_descriptor = -1
        _rename_noreplace(staging_dir, directory)
    except BaseException:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    _fsync_parent_after_commit_noexcept(parent)


def load_glm52_fullparam_payload(directory: str) -> Glm52FullParamSyncPayload:
    """Load and always verify one canonical, bounded, nonsymlink payload tree."""

    directory = _canonical_existing_directory(directory, label="Payload directory")
    directory_descriptor = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | _OPEN_CLOEXEC | _OPEN_NOFOLLOW,
    )
    try:
        raw_manifest = _read_regular_at(
            directory_descriptor,
            _MANIFEST_FILENAME,
            label="Payload manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        manifest = _require_exact_keys(
            _json_no_duplicate_keys(raw_manifest, label="Payload manifest"),
            {"protocol_version", "weight_version", "weight_step", "manifest_checksum", "items"},
            label="Payload manifest",
        )
        protocol_version = manifest["protocol_version"]
        weight_version = manifest["weight_version"]
        weight_step = manifest["weight_step"]
        manifest_checksum = manifest["manifest_checksum"]
        item_entries = manifest["items"]
        if not isinstance(protocol_version, str):
            raise Glm52FullParamPayloadError("Payload protocol_version must be a string")
        weight_version, weight_step = _validate_weight_identity(
            weight_version,
            weight_step,
            label="manifest",
        )
        if not isinstance(manifest_checksum, str) or len(manifest_checksum) != 64:
            raise Glm52FullParamPayloadError("Payload manifest_checksum must be a 64-character string")
        if not isinstance(item_entries, list) or not 0 < len(item_entries) <= _MAX_PAYLOAD_ITEMS:
            raise Glm52FullParamPayloadError("Payload manifest items must be a non-empty bounded list")

        items: list[Glm52PayloadItem] = []
        expected_files = {_MANIFEST_FILENAME}
        field_count = 0
        total_bytes = 0
        for item_index, raw_item in enumerate(item_entries):
            item_entry = _require_exact_keys(
                raw_item,
                {"target", "kind", "contract_version", "fields"},
                label=f"Payload item {item_index}",
            )
            target = item_entry["target"]
            kind = item_entry["kind"]
            contract_version = item_entry["contract_version"]
            field_entries = item_entry["fields"]
            if not all(isinstance(value, str) and value for value in (target, kind, contract_version)):
                raise Glm52FullParamPayloadError(f"Payload item {item_index} string fields must be non-empty")
            if not isinstance(field_entries, list) or not field_entries:
                raise Glm52FullParamPayloadError(f"Payload item {item_index} fields must be a non-empty list")
            schema = _KIND_SCHEMAS.get(kind)
            if schema is None:
                raise Glm52FullParamPayloadError(f"Unknown payload kind {kind!r} for target {target!r}")
            if len(field_entries) != len(schema):
                raise Glm52FullParamPayloadError(f"Payload item {target!r} does not match the {kind!r} field schema")

            fields: list[Glm52PayloadField] = []
            for field_index, raw_field in enumerate(field_entries):
                field_count += 1
                if field_count > _MAX_PAYLOAD_FIELDS:
                    raise Glm52FullParamPayloadError("Payload manifest contains too many fields")
                field_entry = _require_exact_keys(
                    raw_field,
                    {"name", "dtype", "shape", "file", "checksum"},
                    label=f"Payload field {item_index}:{field_index}",
                )
                name = field_entry["name"]
                dtype = field_entry["dtype"]
                shape_values = field_entry["shape"]
                filename = field_entry["file"]
                checksum = field_entry["checksum"]
                if not isinstance(name, str) or not name or not isinstance(dtype, str) or dtype not in _DTYPES:
                    raise Glm52FullParamPayloadError(
                        f"Payload field {item_index}:{field_index} has an invalid name or dtype"
                    )
                if (name, dtype) != schema[field_index]:
                    raise Glm52FullParamPayloadError(
                        f"Payload item {target!r} does not match the {kind!r} field schema"
                    )
                if (
                    not isinstance(shape_values, list)
                    or not shape_values
                    or any(
                        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape_values
                    )
                ):
                    raise Glm52FullParamPayloadError(f"Payload field {item_index}:{field_index} has an invalid shape")
                if not isinstance(checksum, str) or len(checksum) != 64:
                    raise Glm52FullParamPayloadError(
                        f"Payload field {item_index}:{field_index} checksum must be a 64-character string"
                    )
                filename = _plain_filename(filename, label=f"Payload field {item_index}:{field_index} file")
                expected_filename = _field_filename(item_index, field_index, name)
                if filename != expected_filename:
                    raise Glm52FullParamPayloadError(
                        f"Payload field filename must be canonical {expected_filename!r}, got {filename!r}"
                    )
                if filename in expected_files:
                    raise Glm52FullParamPayloadError(f"Duplicate payload byte filename {filename!r}")
                expected_files.add(filename)

                shape = tuple(shape_values)
                expected_bytes = _DTYPE_ITEMSIZE[dtype]
                for value in shape:
                    expected_bytes *= value
                    if expected_bytes > _MAX_FIELD_BYTES:
                        raise Glm52FullParamPayloadError(
                            f"Payload field {item_index}:{field_index} exceeds the field byte limit"
                        )
                total_bytes += expected_bytes
                if total_bytes > _MAX_TOTAL_PAYLOAD_BYTES:
                    raise Glm52FullParamPayloadError("Payload exceeds the total-byte limit")
                raw = _read_regular_at(
                    directory_descriptor,
                    filename,
                    label="Payload byte file",
                    max_bytes=_MAX_FIELD_BYTES,
                    exact_bytes=expected_bytes,
                )
                if len(raw) != expected_bytes:
                    raise Glm52FullParamPayloadError(
                        f"{target}.{name} carries {len(raw)} bytes, expected {expected_bytes} for {dtype} {shape}"
                    )
                fields.append(
                    Glm52PayloadField(
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        data=torch.frombuffer(bytearray(raw), dtype=torch.uint8),
                        checksum=checksum,
                    )
                )
            items.append(
                Glm52PayloadItem(
                    target=target,
                    kind=kind,
                    contract_version=contract_version,
                    fields=tuple(fields),
                )
            )

        actual_files = set(os.listdir(directory_descriptor))
        if actual_files != expected_files:
            raise Glm52FullParamPayloadError(
                "Payload directory entries do not exactly match the manifest: "
                f"missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}"
            )
    finally:
        os.close(directory_descriptor)

    payload = Glm52FullParamSyncPayload(
        protocol_version=protocol_version,
        weight_version=weight_version,
        weight_step=weight_step,
        items=tuple(items),
        manifest_checksum=manifest_checksum,
    )
    verify_glm52_fullparam_payload(payload)
    return payload


__all__ = [
    "GLM52_FULLPARAM_SYNC_PROTOCOL_VERSION",
    "Glm52ExpectedPayloadField",
    "Glm52ExpectedPayloadInventory",
    "Glm52ExpectedPayloadItem",
    "Glm52FullParamPayloadError",
    "Glm52PreparedCheckpoint",
    "Glm52FullParamSyncPayload",
    "Glm52WeightVersionGuard",
    "Glm52PayloadField",
    "Glm52PayloadItem",
    "activate_glm52_prepared_checkpoint",
    "apply_glm52_fullparam_payload",
    "apply_glm52_fullparam_payload_to_hf_checkpoint",
    "build_glm52_expected_payload_inventory",
    "glm52_fullparam_hf_name_mapping",
    "load_glm52_fullparam_payload",
    "merge_glm52_fullparam_payloads",
    "publish_glm52_fullparam_payload",
    "prepare_glm52_fullparam_payload_hf_checkpoint",
    "save_glm52_fullparam_payload",
    "unpack_glm52_payload_field",
    "verify_glm52_fullparam_payload",
]
