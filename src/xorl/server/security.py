"""Security helpers shared by server transports and filesystem boundaries."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import stat
from pathlib import Path
from typing import Iterable


_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}\.?$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*\.?$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_OUTBOUND_ALLOWLIST_ENV = "XORL_OUTBOUND_ENDPOINT_ALLOWLIST"
_DIAGNOSTIC_INPUT_ROOT_ENV = "XORL_DIAGNOSTIC_INPUT_ROOT"
_SERVER_ARTIFACT_ROOT_ENV = "XORL_SERVER_ARTIFACT_ROOT"
_MAX_DIAGNOSTIC_INPUT_BYTES = 8 * 1024 * 1024 * 1024


def validate_identifier(value: str, *, name: str = "identifier") -> str:
    """Return a path-safe identifier or raise ``ValueError``."""
    text = str(value)
    if not _IDENTIFIER_RE.fullmatch(text) or text in {".", ".."}:
        raise ValueError(
            f"{name} must start with an alphanumeric character and contain only "
            "alphanumeric characters, underscores, hyphens, or dots"
        )
    return text


def resolve_path_within(
    base: str | os.PathLike[str],
    candidate: str | os.PathLike[str],
    *,
    must_exist: bool = False,
    reject_symlinks: bool = False,
) -> Path:
    """Resolve ``candidate`` and require it to remain below ``base``."""
    base_path = Path(base).expanduser().resolve()
    raw_candidate = Path(candidate).expanduser()
    candidate_path = raw_candidate if raw_candidate.is_absolute() else base_path / raw_candidate

    if reject_symlinks:
        current = candidate_path
        while current != base_path and current != current.parent:
            if current.is_symlink():
                raise ValueError(f"Symlinked paths are not allowed: {candidate_path}")
            current = current.parent

    resolved = candidate_path.resolve(strict=must_exist)
    try:
        resolved.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root {base_path}: {candidate}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_server_artifact(
    candidate: str | os.PathLike[str],
    *,
    must_exist: bool = False,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a server checkpoint or export path below its configured root.

    ``root`` is authoritative when supplied by the server lifecycle.  The
    environment variable remains a fallback for standalone callers that do
    not own a parsed server ``output_dir``.
    """
    if candidate is None or not str(candidate).strip():
        raise ValueError("Server artifact path must be non-empty")
    if root is None:
        configured_root = os.environ.get(_SERVER_ARTIFACT_ROOT_ENV, "").strip()
        root = configured_root or Path.cwd() / "checkpoints"
    return resolve_path_within(
        root,
        candidate,
        must_exist=must_exist,
        reject_symlinks=True,
    )


def resolve_diagnostic_input(candidate: str | os.PathLike[str]) -> Path:
    """Resolve a regular diagnostic input below an explicitly configured root."""
    configured_root = os.environ.get(_DIAGNOSTIC_INPUT_ROOT_ENV, "").strip()
    if not configured_root:
        raise ValueError(f"Diagnostic file inputs require {_DIAGNOSTIC_INPUT_ROOT_ENV}")
    resolved = resolve_path_within(
        configured_root,
        candidate,
        must_exist=True,
        reject_symlinks=True,
    )
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Diagnostic input must be a regular file: {resolved}")
    if metadata.st_mode & 0o022:
        raise ValueError(f"Diagnostic input must not be group- or world-writable: {resolved}")
    if metadata.st_size > _MAX_DIAGNOSTIC_INPUT_BYTES:
        raise ValueError(f"Diagnostic input exceeds the {_MAX_DIAGNOSTIC_INPUT_BYTES}-byte limit: {resolved}")
    return resolved


def _allowlist_entries() -> list[str]:
    return [entry.strip() for entry in os.environ.get(_OUTBOUND_ALLOWLIST_ENV, "").split(",") if entry.strip()]


def _ip_allowed(address: ipaddress.IPv4Address | ipaddress.IPv6Address, entries: Iterable[str]) -> bool:
    for entry in entries:
        try:
            if address in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def _host_explicitly_allowed(host: str, entries: Iterable[str]) -> bool:
    normalized = host.rstrip(".").lower()
    return any("/" not in entry and entry.rstrip(".").lower() == normalized for entry in entries)


def validate_outbound_endpoint(
    host: str,
    port: int,
    *,
    require_allowlist: bool = False,
) -> tuple[str, int]:
    """Validate an HTTP endpoint before constructing an outbound URL.

    The optional ``XORL_OUTBOUND_ENDPOINT_ALLOWLIST`` accepts exact hostnames,
    IP addresses, and CIDRs. Callers handling untrusted endpoint input may opt
    into mandatory allowlisting with ``require_allowlist=True``. All callers
    reject malformed, link-local, multicast, unspecified, and reserved targets.
    Hostnames are resolved once and returned as a validated IP literal so the
    subsequent HTTP request cannot be redirected by a second DNS lookup.
    """
    normalized_host = str(host).strip().rstrip(".")
    if not normalized_host or any(ch in normalized_host for ch in "/\\@?#\x00\r\n"):
        raise ValueError(f"Invalid endpoint host: {host!r}")

    try:
        normalized_port = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid endpoint port: {port!r}") from exc
    if not 1 <= normalized_port <= 65535:
        raise ValueError(f"Endpoint port must be in [1, 65535], got {normalized_port}")

    entries = _allowlist_entries()
    explicitly_allowed = _host_explicitly_allowed(normalized_host, entries)
    if require_allowlist and not entries and normalized_host.lower() not in {"localhost"}:
        raise ValueError(f"Endpoint host {normalized_host!r} is not allowed; configure {_OUTBOUND_ALLOWLIST_ENV}")
    if require_allowlist and entries and not explicitly_allowed:
        try:
            literal_address = ipaddress.ip_address(normalized_host.strip("[]"))
        except ValueError:
            raise ValueError(f"Endpoint host {normalized_host!r} is not present in {_OUTBOUND_ALLOWLIST_ENV}") from None
        if not _ip_allowed(literal_address, entries):
            raise ValueError(f"Endpoint address {literal_address} is not present in {_OUTBOUND_ALLOWLIST_ENV}")

    try:
        literal = ipaddress.ip_address(normalized_host.strip("[]"))
        addresses = [literal]
    except ValueError:
        if not _HOSTNAME_RE.fullmatch(normalized_host):
            raise ValueError(f"Invalid endpoint hostname: {normalized_host!r}") from None
        addresses = []
        try:
            for family, _, _, _, sockaddr in socket.getaddrinfo(
                normalized_host,
                normalized_port,
                type=socket.SOCK_STREAM,
            ):
                if family in {socket.AF_INET, socket.AF_INET6}:
                    addresses.append(ipaddress.ip_address(sockaddr[0]))
        except socket.gaierror:
            raise ValueError(f"Endpoint hostname could not be resolved safely: {normalized_host}") from None
        if not addresses:
            raise ValueError(f"Endpoint hostname did not resolve to an IP address: {normalized_host}")

    for address in addresses:
        if address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved:
            if not _ip_allowed(address, entries):
                raise ValueError(f"Unsafe endpoint address: {address}")
        if (
            require_allowlist
            and address.is_private
            and not (_ip_allowed(address, entries) or explicitly_allowed or address.is_loopback)
        ):
            raise ValueError(f"Private endpoint address {address} requires an entry in {_OUTBOUND_ALLOWLIST_ENV}")

    return str(addresses[0]), normalized_port


def build_http_endpoint_url(
    host: str,
    port: int,
    path: str,
    *,
    require_allowlist: bool = False,
) -> str:
    """Build a validated HTTP endpoint URL for a fixed application path."""
    if not path.startswith("/") or any(ch in path for ch in "\x00\r\n?#"):
        raise ValueError(f"Invalid endpoint path: {path!r}")
    normalized_host, normalized_port = validate_outbound_endpoint(
        host,
        port,
        require_allowlist=require_allowlist,
    )
    rendered_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    return f"http://{rendered_host}:{normalized_port}{path}"
