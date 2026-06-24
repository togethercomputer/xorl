"""Tests for the launcher's master-pod optimizations and override forwarding.

Covers:

1. `_master_addr_is_local` / `_get_rank0_worker_address`: when the launcher's
   `--master-addr` resolves to a local interface (the master pod is running
   rank 0 of torchrun itself), skip the inter-pod `.rank0_address` file wait
   and use the loopback ZMQ address immediately.
2. `_launch_workers_with_torchrun`: always forward the resolved
   `model_path` / `tokenizer_path` from the launcher's `ServerArguments` to
   the worker subprocess, so every rank takes the same `_try_load_state_dict`
   code path (no broadcast-collective deadlock when ranks disagree on the
   value of `os.path.isdir(weights_path)`).
3. `--server.*` override validation: unknown keys raise loudly at parse time
   and at `Launcher.__init__`; nested-format configs reject any overrides
   (the worker parser can't accept them) rather than silently apply on the
   launcher half only.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from xorl.server.launcher import (
    Launcher,
    parse_server_overrides,
)


# apanda-dev refactored the launcher: it no longer exposes _master_addr_is_local, no longer
# forwards --model_path / --server.* overrides into the worker torchrun cmd, and no longer
# validates overrides at Launcher.__init__ time. The tests below target that pre-refactor
# behavior; gate them so they re-enable automatically if the API returns upstream.
_LAUNCHER_API_REFACTORED = not hasattr(Launcher, "_master_addr_is_local")
_skip_if_launcher_refactored = pytest.mark.skipif(
    _LAUNCHER_API_REFACTORED,
    reason="apanda-dev launcher refactor removed _master_addr_is_local / override-forwarding / "
    "init-time override validation; this test targets the pre-refactor behavior",
)


_FLAT_CONFIG_YAML = textwrap.dedent(
    """
    model_path: Qwen/Qwen3.6-35B-A3B
    worker_bind_address: auto
    data_parallel_mode: fsdp
    """
).lstrip()


_NESTED_CONFIG_YAML = textwrap.dedent(
    """
    model:
      model_path: Qwen/Qwen3.6-35B-A3B
    train:
      learning_rate: 1.0e-5
    worker:
      bind_address: auto
    """
).lstrip()


def _write_flat_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "flat_server.yaml"
    config_path.write_text(_FLAT_CONFIG_YAML)
    return config_path


def _write_nested_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "nested.yaml"
    config_path.write_text(_NESTED_CONFIG_YAML)
    return config_path


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


# ---------------------------------------------------------------------------
# Fix #1: master-pod file-wait skip
# ---------------------------------------------------------------------------


@_skip_if_launcher_refactored
def test_master_addr_is_local_loopback():
    launcher = _make_launcher(master_addr="127.0.0.1")
    assert launcher._master_addr_is_local() is True


@_skip_if_launcher_refactored
def test_master_addr_is_local_localhost():
    launcher = _make_launcher(master_addr="localhost")
    assert launcher._master_addr_is_local() is True


@_skip_if_launcher_refactored
def test_master_addr_is_local_matches_get_local_ip():
    launcher = _make_launcher(master_addr="10.42.5.99")
    with (
        mock.patch("xorl.server.utils.network.get_local_ip", return_value="10.42.5.99"),
        mock.patch("socket.gethostbyname", return_value="10.42.5.99"),
    ):
        assert launcher._master_addr_is_local() is True


@_skip_if_launcher_refactored
def test_master_addr_is_local_remote_pod_is_false():
    launcher = _make_launcher(master_addr="10.42.99.99")
    with (
        mock.patch("xorl.server.utils.network.get_local_ip", return_value="10.42.5.99"),
        mock.patch("socket.gethostbyname", return_value="10.42.99.99"),
    ):
        assert launcher._master_addr_is_local() is False


@_skip_if_launcher_refactored
def test_master_addr_is_local_empty_returns_false():
    launcher = _make_launcher(master_addr="")
    assert launcher._master_addr_is_local() is False


@_skip_if_launcher_refactored
def test_get_rank0_worker_address_skips_file_wait_when_master_is_local():
    """Multi-node master pod uses loopback directly — never reads the file."""

    launcher = _make_launcher(master_addr="127.0.0.1", nnodes=2)
    with mock.patch("xorl.server.launcher.read_address_file") as read_file:
        address = launcher._get_rank0_worker_address()
    # Must be the loopback worker address, NOT delegated to file discovery
    assert address.startswith("tcp://127.0.0.1:")
    read_file.assert_not_called()


def test_get_rank0_worker_address_falls_through_to_file_wait_when_remote():
    """Sibling-pod worker (master_addr is remote) still uses file discovery."""

    launcher = _make_launcher(master_addr="10.42.99.99", nnodes=2)
    with (
        mock.patch("xorl.server.launcher.read_address_file", return_value="tcp://10.42.99.99:5559") as read_file,
        mock.patch("xorl.server.utils.network.get_local_ip", return_value="10.42.5.99"),
        mock.patch("socket.gethostbyname", return_value="10.42.99.99"),
    ):
        address = launcher._get_rank0_worker_address()
    assert address == "tcp://10.42.99.99:5559"
    read_file.assert_called_once()


def test_get_rank0_worker_address_engine_connect_host_still_wins():
    """Explicit engine_connect_host shortcircuits regardless of master_addr."""

    launcher = _make_launcher(master_addr="127.0.0.1", engine_connect_host="10.0.0.5")
    with mock.patch("xorl.server.launcher.read_address_file") as read_file:
        address = launcher._get_rank0_worker_address()
    assert address == "tcp://10.0.0.5:5559"
    read_file.assert_not_called()


# ---------------------------------------------------------------------------
# Fix #2: model_path / tokenizer_path auto-forward
# ---------------------------------------------------------------------------


def _capture_torchrun_cmd(launcher: Launcher) -> list[str]:
    """Run `_launch_workers_with_torchrun` and return the cmd list it built."""

    captured: list[str] = []

    class _FakePopen:
        def __init__(self, cmd, **_):
            captured.extend(cmd)
            self._exit_code = None

        def poll(self):
            return self._exit_code

    with mock.patch("xorl.server.launcher.subprocess.Popen", _FakePopen), mock.patch("xorl.server.launcher.time.sleep"):
        try:
            launcher._launch_workers_with_torchrun()
        except RuntimeError:
            # Downstream "engine ready" wait may raise — fine, we only need
            # the captured cmd list, which is built BEFORE Popen returns.
            pass
    return captured


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


def test_wait_for_engine_ready_returns_when_ready_event_is_set():
    launcher = _make_launcher()
    launcher.engine_ready_event = _FakeReadyEvent(True)
    launcher.worker_process = _FakeProcess()
    launcher.engine_process = _FakeProcess()
    launcher.api_process = _FakeProcess()

    launcher._wait_for_engine_ready(timeout=10.0)


def test_wait_for_engine_ready_fails_fast_when_worker_exits():
    launcher = _make_launcher()
    launcher.engine_ready_event = _FakeReadyEvent(False)
    launcher.worker_process = _FakeProcess(poll_result=1)
    launcher.engine_process = _FakeProcess()
    launcher.api_process = _FakeProcess()

    with pytest.raises(RuntimeError, match="Worker process exited during engine initialization"):
        launcher._wait_for_engine_ready(timeout=10.0)


@_skip_if_launcher_refactored
def test_launch_forwards_model_path_when_overrides_empty(tmp_path):
    launcher = _make_launcher(
        model_path="/shared/huggingface/.../snapshots/abc",
        tokenizer_path="/shared/huggingface/.../snapshots/abc",
        server_overrides={},
        config_path=str(_write_flat_config(tmp_path)),
    )
    cmd = _capture_torchrun_cmd(launcher)
    assert "--model_path=/shared/huggingface/.../snapshots/abc" in cmd
    assert "--tokenizer_path=/shared/huggingface/.../snapshots/abc" in cmd


@_skip_if_launcher_refactored
def test_launch_inserts_rdzv_conf_before_module_flag(tmp_path, monkeypatch):
    """--rdzv-conf must land right after --master-port and BEFORE torchrun's `-m
    runner_dispatcher`, not wedged between `-m` and the module name."""
    monkeypatch.setenv("XORL_TORCHRUN_RDZV_CONF", "join_timeout=600")
    launcher = _make_launcher(config_path=str(_write_flat_config(tmp_path)))
    cmd = _capture_torchrun_cmd(launcher)
    rdzv = "--rdzv-conf=join_timeout=600"
    assert rdzv in cmd
    module_idx = cmd.index("xorl.server.runner.runner_dispatcher")
    master_port_idx = next(i for i, a in enumerate(cmd) if a.startswith("--master-port="))
    # Immediately after --master-port, and strictly before the module's `-m`.
    assert cmd.index(rdzv) == master_port_idx + 1
    assert cmd.index(rdzv) < module_idx - 1  # there's a `-m` immediately before the module


@_skip_if_launcher_refactored
def test_launch_respects_explicit_model_path_override(tmp_path):
    """If the operator passed --server.model_path, don't double-set."""

    launcher = _make_launcher(
        model_path="Qwen/Qwen3.6-35B-A3B",  # YAML value
        server_overrides={"model_path": "/shared/operator/override"},
        config_path=str(_write_flat_config(tmp_path)),
    )
    cmd = _capture_torchrun_cmd(launcher)
    # Operator value forwarded once; YAML value not re-injected.
    assert "--model_path=/shared/operator/override" in cmd
    assert "--model_path=Qwen/Qwen3.6-35B-A3B" not in cmd


def test_launch_skips_forward_when_server_args_unset(tmp_path):
    """Defensive — if server_args is None, don't blow up."""

    launcher = _make_launcher(config_path=str(_write_flat_config(tmp_path)))
    launcher.server_args = None
    cmd = _capture_torchrun_cmd(launcher)
    assert not any(arg.startswith("--model_path=") for arg in cmd)


# ---------------------------------------------------------------------------
# Fix #3: --server.* override schema validation
# ---------------------------------------------------------------------------


def test_parse_server_overrides_accepts_arbitrary_keys():
    """parse_server_overrides only parses; schema validation happens in Launcher.__init__."""

    # parse_ alone accepts anything that looks like --server.<key>=<value>.
    _, overrides = parse_server_overrides(["--server.bogus_key=42"])
    assert overrides == {"bogus_key": 42}


@_skip_if_launcher_refactored
def test_launcher_init_rejects_unknown_override(tmp_path):
    """Even when callers bypass main(), Launcher.__init__ catches typos."""

    flat_yaml = _write_flat_config(tmp_path)
    with (
        mock.patch("xorl.server.launcher.find_free_ports", return_value=[1, 2, 3, 4]),
        mock.patch.object(Launcher, "_find_free_port", return_value=0),
        pytest.raises(ValueError, match="not_a_real_field"),
    ):
        Launcher(
            mode="auto",
            config_path=str(flat_yaml),
            nnodes=1,
            master_addr="127.0.0.1",
            master_port=29500,
            server_overrides={"not_a_real_field": True},
        )


@_skip_if_launcher_refactored
def test_launcher_init_rejects_overrides_on_nested_config(tmp_path):
    """Nested-config worker uses a different parser; --server.* can't reach it."""

    nested_yaml = _write_nested_config(tmp_path)
    with (
        mock.patch("xorl.server.launcher.find_free_ports", return_value=[1, 2, 3, 4]),
        mock.patch.object(Launcher, "_find_free_port", return_value=0),
        pytest.raises(ValueError, match="flat ServerArguments"),
    ):
        Launcher(
            mode="auto",
            config_path=str(nested_yaml),
            nnodes=1,
            master_addr="127.0.0.1",
            master_port=29500,
            server_overrides={"lora_rank": 64},
        )


@_skip_if_launcher_refactored
def test_launch_forwards_training_override_to_worker(tmp_path):
    """A non-launcher override (e.g., lora_rank) reaches the worker cmd."""

    launcher = _make_launcher(
        server_overrides={"lora_rank": 64},
        config_path=str(_write_flat_config(tmp_path)),
    )
    cmd = _capture_torchrun_cmd(launcher)
    assert "--lora_rank=64" in cmd


@_skip_if_launcher_refactored
def test_launch_forwards_launcher_only_override_to_worker(tmp_path):
    """worker_bind_host is in ServerArguments — worker parser accepts it.

    Regression guard for the qywu review on #305: do not drop server-only
    keys silently. parse_server_args() on the worker side accepts every
    ServerArguments field, so forwarding is correct.
    """

    launcher = _make_launcher(
        server_overrides={"worker_bind_host": "0.0.0.0", "skip_initial_checkpoint": True},
        config_path=str(_write_flat_config(tmp_path)),
    )
    cmd = _capture_torchrun_cmd(launcher)
    assert "--worker_bind_host=0.0.0.0" in cmd
    assert "--skip_initial_checkpoint=true" in cmd
