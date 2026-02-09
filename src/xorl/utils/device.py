# Following codes are inspired from https://github.com/volcengine/verl/blob/main/verl/utils/device.py

from typing import Any

import torch

from . import logging


logger = logging.get_logger(__name__)


IS_CUDA_AVAILABLE = torch.cuda.is_available()


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
