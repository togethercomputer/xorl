# Following codes are inspired from https://github.com/volcengine/verl/blob/main/verl/utils/device.py

import os
from datetime import timedelta
from typing import Any

import torch

from . import logging


logger = logging.get_logger(__name__)


IS_CUDA_AVAILABLE = torch.cuda.is_available()


# Default process-group init timeout (minutes). Larger than torch's NCCL default
# (600s) because the server-training head's engine (rank 0) starts several minutes
# after the workers (config write + API/orchestrator bring-up + rank-0 address-file
# discovery) and must load a large checkpoint before the first collective; with the
# 600s default the workers blow store->get('0') waiting for rank 0 (DistBackendError).
# Override with XORL_PROCESS_GROUP_TIMEOUT_MINUTES. Used by both the local trainer and
# the server worker so the two paths stay consistent.
_DEFAULT_PROCESS_GROUP_TIMEOUT_MINUTES = 60.0
# Floor so a misconfigured 0/negative override can't make PG init time out instantly.
_MIN_PROCESS_GROUP_TIMEOUT_MINUTES = 1.0


def get_process_group_timeout() -> timedelta:
    """Return the process-group init timeout as a ``timedelta``.

    Reads ``XORL_PROCESS_GROUP_TIMEOUT_MINUTES`` (float minutes); falls back to
    ``_DEFAULT_PROCESS_GROUP_TIMEOUT_MINUTES`` if unset or unparsable. A
    non-positive value would make ``init_process_group`` time out instantly, so
    it is clamped up to a small positive floor.
    """
    raw = os.environ.get("XORL_PROCESS_GROUP_TIMEOUT_MINUTES", "").strip()
    minutes = _DEFAULT_PROCESS_GROUP_TIMEOUT_MINUTES
    if raw:
        try:
            minutes = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "XORL_PROCESS_GROUP_TIMEOUT_MINUTES=%r is not a number; using default %s min",
                raw,
                _DEFAULT_PROCESS_GROUP_TIMEOUT_MINUTES,
            )
    # Guard against 0/negative (instant-timeout PG init).
    minutes = max(minutes, _MIN_PROCESS_GROUP_TIMEOUT_MINUTES)
    return timedelta(minutes=minutes)


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU and CUDA."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_device_name() -> str:
    """Get real device name, e.g. A100, H100"""
    return get_torch_device().get_device_name()


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if IS_CUDA_AVAILABLE:
        return "nccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()
