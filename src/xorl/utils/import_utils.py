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
    "triton": _is_package_available("triton"),
}


def is_flash_attn_2_available() -> bool:
    """Check if flash_attn is available (works for both FA2 and FA3).

    FA2 and FA3 use the same import path (from flash_attn import ...).
    FA3 is simply a newer version of the flash-attn package.
    """
    return _PACKAGE_FLAGS["flash_attn"]


def is_fused_moe_available() -> bool:
    import torch

    return torch.cuda.is_available() and _PACKAGE_FLAGS["triton"]


@lru_cache
def is_torch_version_greater_than(value: str) -> bool:
    return _get_package_version("torch") >= version.parse(value)


@lru_cache
def is_transformers_version_greater_or_equal_to(value: str) -> bool:
    return _get_package_version("transformers") > version.parse(value)


