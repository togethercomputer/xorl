import importlib.util
import io
import os
import struct
import sys
import time
import types
from pathlib import Path

import pytest


_QUACK_DIR = Path(__file__).parents[2] / "src" / "xorl" / "ops" / "quack"


def _load_module(name: str, path: Path, monkeypatch: pytest.MonkeyPatch | None = None):
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, "cutlass", types.ModuleType("cutlass"))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cache_utils(monkeypatch: pytest.MonkeyPatch):
    cutlass = types.ModuleType("cutlass")
    cutlass.__version__ = "test"
    cute = types.ModuleType("cutlass.cute")
    cute.runtime = types.SimpleNamespace()
    cutlass.cute = cute
    tvm_ffi = types.ModuleType("tvm_ffi")
    tvm_ffi.__version__ = "test"
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute)
    monkeypatch.setitem(sys.modules, "tvm_ffi", tvm_ffi)
    return _load_module("quack_cache_utils_test", _QUACK_DIR / "cache_utils.py")


def test_worker_protocol_times_out_on_silent_worker():
    protocol = _load_module("quack_worker_protocol_test", _QUACK_DIR / "_worker_protocol.py")
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            with pytest.raises(TimeoutError, match="did not respond"):
                protocol.recv_message(stream, timeout_s=0.05)
    finally:
        os.close(write_fd)
    assert time.monotonic() - started < 1


def test_worker_protocol_rejects_truncated_body():
    protocol = _load_module("quack_worker_protocol_truncated_test", _QUACK_DIR / "_worker_protocol.py")
    stream = io.BytesIO(struct.pack("<I", 4) + b"x")
    with pytest.raises(ValueError, match="Truncated compile-worker message body"):
        protocol.recv_message(stream)


def test_ptxas_uses_unique_outputs_and_a_timeout(tmp_path, monkeypatch):
    ptxas = _load_module("cute_dsl_ptxas_test", _QUACK_DIR / "cute_dsl_ptxas.py", monkeypatch)
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

    monkeypatch.setattr(ptxas, "_validated_ptxas_path", lambda: "/bin/true")
    monkeypatch.setattr(ptxas.subprocess, "run", fake_run)

    assert ptxas._compile_ptx(ptx_path, ptx_content) == b"cubin"
    assert ptxas._compile_ptx(ptx_path, ptx_content) == b"cubin"
    assert len(set(output_paths)) == 2
    assert all(not path.exists() for path in output_paths)


def test_filename_match_still_requires_matching_entry(tmp_path, monkeypatch):
    ptxas = _load_module("cute_dsl_ptxas_entry_test", _QUACK_DIR / "cute_dsl_ptxas.py", monkeypatch)
    wrong = tmp_path / "wanted.ptx"
    right = tmp_path / "other.ptx"
    wrong.write_text(".version 8.0\n.visible .entry wrong() {\n}\n")
    right.write_text(".version 8.0\n.visible .entry wanted() {\n}\n")
    monkeypatch.setattr(ptxas, "_validated_dump_dir", lambda: tmp_path)

    found = ptxas._get_ptx(types.SimpleNamespace(function_name="wanted"))
    assert found is not None
    _, found_path = found
    assert found_path == right


def test_cache_key_hash_is_structural_and_deterministic(monkeypatch):
    cache_utils = _load_cache_utils(monkeypatch)

    key = ("kernel", 3, 1.5, True, None, ("m", "n"))
    assert cache_utils._key_to_hash(key) == cache_utils._key_to_hash(key)
    assert cache_utils._key_to_hash(("ab", "c")) != cache_utils._key_to_hash(("a", "bc"))
    assert cache_utils._key_to_hash((1,)) != cache_utils._key_to_hash(("1",))


def test_cache_key_hash_rejects_objects_without_pickle_hooks(monkeypatch):
    cache_utils = _load_cache_utils(monkeypatch)
    reduced = False

    class MaliciousPickleValue:
        def __reduce__(self):
            nonlocal reduced
            reduced = True
            raise AssertionError("cache-key hashing must not invoke __reduce__")

    with pytest.raises(TypeError, match="unsupported Quack cache-key value"):
        cache_utils._key_to_hash((MaliciousPickleValue(),))
    assert reduced is False


def test_cache_key_hash_accepts_cutlass_dtype_classes_without_metaclass_repr(monkeypatch):
    cache_utils = _load_cache_utils(monkeypatch)
    fake_dtype = type("FakeDType", (), {})
    fake_dtype.__module__ = "cutlass.fake"

    assert cache_utils._key_to_hash((fake_dtype,)) == cache_utils._key_to_hash((fake_dtype,))
