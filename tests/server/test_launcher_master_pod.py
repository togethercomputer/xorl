"""Tests for live launcher address discovery, readiness, and argument parsing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from xorl.server.launcher import (
    Launcher,
)


def _make_launcher(
    *,
    nnodes: int = 2,
    master_addr: str = "10.42.5.99",
    worker_bind_host: str = "0.0.0.0",
    worker_bind_port: int = 5559,
    worker_bind_address: str = "auto",
    engine_connect_host: str | None = None,
    model_path: str = "Qwen/Qwen3.6-35B-A3B",
    tokenizer_path: str = "Qwen/Qwen3.6-35B-A3B",
    server_overrides: dict | None = None,
    config_path: str | None = None,
) -> Launcher:
    """Construct an Launcher with just enough state for unit tests."""

    server_args = SimpleNamespace(
        engine_connect_host=engine_connect_host,
        worker_bind_host=worker_bind_host,
        worker_bind_port=worker_bind_port,
        worker_bind_address=worker_bind_address,
        worker_connection_timeout=0.1,  # don't actually wait in tests
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    launcher = Launcher.__new__(Launcher)
    launcher.nnodes = nnodes
    launcher.master_addr = master_addr
    launcher.master_port = 29611
    launcher.server_args = server_args
    launcher.output_dir = "/tmp/__test_launcher_output__"
    launcher.worker_address = "tcp://0.0.0.0:5559"
    launcher.server_overrides = server_overrides or {}
    launcher.config_path = config_path or "/tmp/__test_config__.yaml"
    launcher.nproc_per_node = 8
    return launcher


def test_launcher_worker_discovery_and_readiness_policy():
    """Remote siblings use discovery, while an explicit connect host wins."""

    launcher = _make_launcher(master_addr="10.42.99.99", nnodes=2)
    with (
        mock.patch("xorl.server.launcher.read_address_file", return_value="tcp://10.42.99.99:5559") as read_file,
        mock.patch("xorl.server.utils.network.get_local_ip", return_value="10.42.5.99"),
        mock.patch("socket.gethostbyname", return_value="10.42.99.99"),
    ):
        address = launcher._get_rank0_worker_address()
    assert address == "tcp://10.42.99.99:5559"
    read_file.assert_called_once()

    launcher = _make_launcher(master_addr="127.0.0.1", engine_connect_host="10.0.0.5")
    with mock.patch("xorl.server.launcher.read_address_file") as read_file:
        address = launcher._get_rank0_worker_address()
    assert address == "tcp://10.0.0.5:5559"
    read_file.assert_not_called()

    launcher = _make_launcher()
    launcher.engine_ready_event = _FakeReadyEvent(True)
    launcher.worker_process = _FakeProcess()
    launcher.engine_process = _FakeProcess()
    launcher.api_process = _FakeProcess()

    launcher._wait_for_engine_ready(timeout=10.0)

    failed_launcher = _make_launcher()
    failed_launcher.engine_ready_event = _FakeReadyEvent(False)
    failed_launcher.worker_process = _FakeProcess(poll_result=1)
    failed_launcher.engine_process = _FakeProcess()
    failed_launcher.api_process = _FakeProcess()

    with pytest.raises(RuntimeError, match="Worker process exited during engine initialization"):
        failed_launcher._wait_for_engine_ready(timeout=10.0)


class _FakeReadyEvent:
    def __init__(self, ready: bool):
        self.ready = ready

    def wait(self, timeout: float) -> bool:
        return self.ready


class _FakeProcess:
    def __init__(self, *, poll_result: int | None = None, alive: bool = True, exitcode: int | None = None):
        self._poll_result = poll_result
        self._alive = alive
        self.exitcode = exitcode

    def poll(self):
        return self._poll_result

    def is_alive(self) -> bool:
        return self._alive
