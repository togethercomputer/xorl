"""Shared helpers for distributed tests.

Importable as a regular module via:
    from .distributed_utils import run_distributed_script, skip_if_gpu_count_less_than
"""

import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import pytest
import torch


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def skip_if_gpu_count_less_than(n: int):
    return pytest.mark.skipif(
        gpu_count() < n,
        reason=f"Requires {n} GPUs, found {gpu_count()}",
    )


# ---------------------------------------------------------------------------
# Distributed test result
# ---------------------------------------------------------------------------


@dataclass
class DistributedTestResult:
    """Captures the outcome of a distributed test subprocess."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def assert_success(self, msg: str = ""):
        """Assert the distributed test completed successfully."""
        if not self.success:
            stderr_tail = "\n".join(self.stderr.splitlines()[-50:])
            raise AssertionError(
                f"Distributed test failed (exit_code={self.exit_code}){': ' + msg if msg else ''}\n"
                f"--- stderr (last 50 lines) ---\n{stderr_tail}"
            )


# ---------------------------------------------------------------------------
# Torchrun launcher
# ---------------------------------------------------------------------------


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_distributed_script(
    script_path: str,
    num_gpus: int = 2,
    timeout: int = 120,
    extra_env: Optional[Dict[str, str]] = None,
) -> DistributedTestResult:
    """Launch a Python script via torchrun and collect results."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(num_gpus),
        "--master_port",
        str(_get_free_port()),
        script_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return DistributedTestResult(
            exit_code=-1,
            stdout=e.stdout or "",
            stderr=e.stderr or f"Timeout after {timeout}s",
        )

    return DistributedTestResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
