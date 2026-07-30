"""Security-boundary tests for server paths, endpoints, and worker IPC."""

import socket
from io import BytesIO

import pytest
import torch

from xorl.ops.quack._worker_protocol import recv_message, send_message
from xorl.server.security import (
    build_http_endpoint_url,
    resolve_diagnostic_input,
    resolve_path_within,
    resolve_server_artifact,
    validate_outbound_endpoint,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_api_endpoint_requires_explicit_allowlist(monkeypatch):
    monkeypatch.delenv("XORL_OUTBOUND_ENDPOINT_ALLOWLIST", raising=False)
    with pytest.raises(ValueError, match="not allowed"):
        validate_outbound_endpoint("inference.example", 30000, require_allowlist=True)

    monkeypatch.setenv("XORL_OUTBOUND_ENDPOINT_ALLOWLIST", "inference.example")
    monkeypatch.setattr(
        "xorl.server.security.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 30000))],
    )
    assert validate_outbound_endpoint("inference.example", 30000, require_allowlist=True) == (
        "8.8.8.8",
        30000,
    )


def test_endpoint_url_pins_validated_dns_address(monkeypatch):
    monkeypatch.setenv("XORL_OUTBOUND_ENDPOINT_ALLOWLIST", "inference.example")
    resolutions = iter(
        [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 30000))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 30000))],
        ]
    )
    monkeypatch.setattr(
        "xorl.server.security.socket.getaddrinfo",
        lambda *_args, **_kwargs: next(resolutions),
    )

    assert (
        build_http_endpoint_url("inference.example", 30000, "/health", require_allowlist=True)
        == "http://8.8.8.8:30000/health"
    )


def test_endpoint_rejects_metadata_and_malformed_targets(monkeypatch):
    monkeypatch.delenv("XORL_OUTBOUND_ENDPOINT_ALLOWLIST", raising=False)
    with pytest.raises(ValueError, match="Unsafe endpoint"):
        validate_outbound_endpoint("169.254.169.254", 80)
    with pytest.raises(ValueError, match="Invalid endpoint host"):
        validate_outbound_endpoint("localhost@169.254.169.254", 80)


def test_resolve_path_within_rejects_escape_and_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="escapes configured root"):
        resolve_path_within(root, outside)

    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked paths"):
        resolve_path_within(root, link, must_exist=True, reject_symlinks=True)


def test_diagnostic_input_requires_configured_root_and_regular_private_file(tmp_path, monkeypatch):
    root = tmp_path / "diagnostics"
    root.mkdir()
    payload = root / "reference.pt"
    payload.write_bytes(b"safe")
    payload.chmod(0o600)

    monkeypatch.delenv("XORL_DIAGNOSTIC_INPUT_ROOT", raising=False)
    with pytest.raises(ValueError, match="XORL_DIAGNOSTIC_INPUT_ROOT"):
        resolve_diagnostic_input(payload)

    monkeypatch.setenv("XORL_DIAGNOSTIC_INPUT_ROOT", str(root))
    assert resolve_diagnostic_input(payload) == payload.resolve()

    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes configured root"):
        resolve_diagnostic_input(outside)


def test_server_artifact_path_is_confined_to_configured_root(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("XORL_SERVER_ARTIFACT_ROOT", str(root))

    assert resolve_server_artifact("checkpoint", must_exist=True) == checkpoint
    with pytest.raises(ValueError, match="escapes configured root"):
        resolve_server_artifact(tmp_path / "outside")


def test_compile_worker_rejects_targets_outside_quack():
    from xorl.ops.quack._compile_worker import _resolve_compile_function

    with pytest.raises(ValueError, match="Quack module"):
        _resolve_compile_function("os", "system")
    with pytest.raises(ValueError, match="safe qualified name"):
        _resolve_compile_function("xorl.ops.quack.autotuner", "__builtins__.eval")


def test_compile_worker_protocol_roundtrips_safe_types_and_rejects_oversized_header():
    stream = BytesIO()
    message = {
        "dtype": torch.bfloat16,
        "device": torch.device("cpu"),
        "shape": (2, 4),
        "values": [1, 2],
    }
    send_message(stream, message)
    stream.seek(0)
    assert recv_message(stream) == message

    oversized = BytesIO((64 * 1024 * 1024 + 1).to_bytes(4, "little"))
    with pytest.raises(ValueError, match="size limit"):
        recv_message(oversized)
