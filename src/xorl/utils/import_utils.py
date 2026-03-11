"""Import utils"""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING, Dict

from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


_PACKAGE_FLAGS: Dict[str, bool] = {
    "flash_attn": _is_package_available("flash_attn"),
    "flash_attn_interface": _is_package_available("flash_attn_interface"),
    "triton": _is_package_available("triton"),
}


def _detect_flash_attn_version() -> int:
    """Detect installed Flash Attention version.

    Returns:
        3 if flash_attn_interface (FA3/Hopper) is available (preferred),
        2 if flash_attn with v2 API is available,
        0 if neither is installed.
    """
    if _PACKAGE_FLAGS["flash_attn_interface"]:
        return 3
    if _PACKAGE_FLAGS["flash_attn"]:
        return 2
    return 0


FLASH_ATTN_VERSION: int = _detect_flash_attn_version()
"""Installed Flash Attention version: 3 (Hopper), 2 (legacy), or 0 (none)."""


def is_flash_attn_available() -> bool:
    """Check if any version of Flash Attention is installed."""
    return FLASH_ATTN_VERSION > 0


# Keep for backward compat
is_flash_attn_2_available = is_flash_attn_available


def is_fused_moe_available() -> bool:
    import torch

    return torch.cuda.is_available() and _PACKAGE_FLAGS["triton"]


@lru_cache
def is_torch_version_greater_than(value: str) -> bool:
    return _get_package_version("torch") >= version.parse(value)


@lru_cache
def is_transformers_version_greater_or_equal_to(value: str) -> bool:
    return _get_package_version("transformers") > version.parse(value)


