"""Security-boundary tests for server paths, endpoints, and worker IPC."""

from io import BytesIO

import pytest
import torch

from xorl.ops.quack._worker_protocol import recv_message, send_message
from xorl.server.security import resolve_path_within, validate_outbound_endpoint


pytestmark = [pytest.mark.cpu, pytest.mark.server]


def test_api_endpoint_requires_explicit_allowlist(monkeypatch):
    monkeypatch.delenv("XORL_OUTBOUND_ENDPOINT_ALLOWLIST", raising=False)
    with pytest.raises(ValueError, match="not allowed"):
        validate_outbound_endpoint("inference.example", 30000, require_allowlist=True)

    monkeypatch.setenv("XORL_OUTBOUND_ENDPOINT_ALLOWLIST", "inference.example")
    assert validate_outbound_endpoint("inference.example", 30000, require_allowlist=True) == (
        "inference.example",
        30000,
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
