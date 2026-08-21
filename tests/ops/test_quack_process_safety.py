import io
import os
import struct
import time
import types
from pathlib import Path

import pytest

from xorl.ops._vendored.quack import _worker_protocol as worker_protocol
from xorl.ops._vendored.quack import cache_utils, cute_dsl_ptxas


def test_quack_process_and_cache_safety_policy(tmp_path, monkeypatch):
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            with pytest.raises(TimeoutError, match="did not respond"):
                worker_protocol.recv_message(stream, timeout_s=0.05)
    finally:
        os.close(write_fd)
    assert time.monotonic() - started < 1

    _assert_worker_protocol_rejects_truncated_body()
    _assert_ptxas_uses_unique_outputs_and_a_timeout(tmp_path, monkeypatch)
    _assert_cache_key_hash_policy()


def _assert_worker_protocol_rejects_truncated_body():
    stream = io.BytesIO(struct.pack("<I", 4) + b"x")
    with pytest.raises(ValueError, match="Truncated compile-worker message body"):
        worker_protocol.recv_message(stream)


def _assert_ptxas_uses_unique_outputs_and_a_timeout(tmp_path, monkeypatch):
    ptx_path = tmp_path / "kernel.ptx"
    ptx_content = ".version 8.0\n.target sm_90a\n.visible .entry kernel() {\n}\n"
    ptx_path.write_text(ptx_content)
    output_paths = []

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("-o") + 1])
        output_paths.append(output_path)
        output_path.write_bytes(b"cubin")
        assert kwargs["timeout"] == 300
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(cute_dsl_ptxas, "_validated_ptxas_path", lambda: "/bin/true")
    monkeypatch.setattr(cute_dsl_ptxas.subprocess, "run", fake_run)

    assert cute_dsl_ptxas._compile_ptx(ptx_path, ptx_content) == b"cubin"
    assert cute_dsl_ptxas._compile_ptx(ptx_path, ptx_content) == b"cubin"
    assert len(set(output_paths)) == 2
    assert all(not path.exists() for path in output_paths)

    _assert_filename_match_still_requires_matching_entry(tmp_path, monkeypatch)


def _assert_filename_match_still_requires_matching_entry(tmp_path, monkeypatch):
    wrong = tmp_path / "wanted.ptx"
    right = tmp_path / "other.ptx"
    wrong.write_text(".version 8.0\n.visible .entry wrong() {\n}\n")
    right.write_text(".version 8.0\n.visible .entry wanted() {\n}\n")
    monkeypatch.setattr(cute_dsl_ptxas, "_validated_dump_dir", lambda: tmp_path)

    found = cute_dsl_ptxas._get_ptx(types.SimpleNamespace(function_name="wanted"))
    assert found is not None
    _, found_path = found
    assert found_path == right


def _assert_cache_key_hash_policy():
    key = ("kernel", 3, 1.5, True, None, ("m", "n"))
    assert cache_utils._key_to_hash(key) == cache_utils._key_to_hash(key)
    assert cache_utils._key_to_hash(("ab", "c")) != cache_utils._key_to_hash(("a", "bc"))
    assert cache_utils._key_to_hash((1,)) != cache_utils._key_to_hash(("1",))

    _assert_cache_key_hash_rejects_objects_without_pickle_hooks(cache_utils)
    _assert_cache_key_hash_accepts_cutlass_dtype_classes(cache_utils)


def _assert_cache_key_hash_rejects_objects_without_pickle_hooks(cache_utils):
    reduced = False

    class MaliciousPickleValue:
        def __reduce__(self):
            nonlocal reduced
            reduced = True
            raise AssertionError("cache-key hashing must not invoke __reduce__")

    with pytest.raises(TypeError, match="unsupported Quack cache-key value"):
        cache_utils._key_to_hash((MaliciousPickleValue(),))
    assert reduced is False


def _assert_cache_key_hash_accepts_cutlass_dtype_classes(cache_utils):
    fake_dtype = type("FakeDType", (), {})
    fake_dtype.__module__ = "cutlass.fake"

    assert cache_utils._key_to_hash((fake_dtype,)) == cache_utils._key_to_hash((fake_dtype,))
