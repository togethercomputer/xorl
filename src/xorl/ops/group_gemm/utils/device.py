from functools import lru_cache

import torch

from ....utils.device import get_device_name


@lru_cache
def get_device_key() -> str:
    if torch.cuda.get_device_capability() == (8, 0):
        return "A100"  # A30 is treated the same way as A100 for the moment.

    if torch.cuda.get_device_capability() == (9, 0):
        return "H100"

    name = get_device_name()
    if name.startswith("NVIDIA "):
        name = name[len("NVIDIA ") :]

    return name
