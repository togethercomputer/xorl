"""
LoRA module mapping.

Maps source module types to their corresponding LoRA replacement classes.
"""

from typing import Dict, Optional, Type, Union

import torch.nn as nn

from xorl.models.layers.moe import MoEExperts, MoEExpertsLoRA

from .modules.base import LoraModule
from .modules.linear import LoraLinear


# Mapping from source module type to LoRA class
# To add support for new module types, add entries here
# MoEExperts catches all subclasses (Qwen3MoeSparseExperts, Qwen3MoeTritonExperts, etc.)
LORA_MAPPING: Dict[Type[nn.Module], Type[Union[LoraModule, nn.Module]]] = {
    nn.Linear: LoraLinear,
}

# MoE mapping is registered lazily to avoid circular import:
# moe/lora.py → xorl.lora.modules.base → xorl.lora.__init__ → mapping → moe (cycle)
_moe_registered = False


def _ensure_moe_mapping():
    global _moe_registered
    if not _moe_registered:
        _moe_registered = True

        LORA_MAPPING[MoEExperts] = MoEExpertsLoRA


def get_lora_class_for_module(module: nn.Module) -> Optional[Type[Union[LoraModule, nn.Module]]]:
    """
    Find the appropriate LoRA class for a given module.

    Args:
        module: The module to find a LoRA class for

    Returns:
        The LoRA class that can replace the module, or None if no match found

    Example:
        >>> linear = nn.Linear(768, 768)
        >>> lora_cls = get_lora_class_for_module(linear)
        >>> lora_cls
        <class 'xorl.lora.modules.linear.LoraLinear'>
    """
    _ensure_moe_mapping()

    # Don't wrap modules that are already LoRA modules
    if isinstance(module, LoraModule):
        return None

    # Check if module has 'lora_config' (already a LoRA module, e.g., MoE LoRA)
    if hasattr(module, "lora_config"):
        return None

    for source_cls, lora_cls in LORA_MAPPING.items():
        if isinstance(module, source_cls):
            return lora_cls
    return None


def can_apply_lora(module: nn.Module) -> bool:
    """
    Check if LoRA can be applied to this module.

    Args:
        module: The module to check

    Returns:
        True if a LoRA class exists for this module type, False otherwise

    Example:
        >>> can_apply_lora(nn.Linear(10, 10))
        True
        >>> can_apply_lora(nn.LayerNorm(10))
        False
    """
    return get_lora_class_for_module(module) is not None


def register_lora_mapping(source_cls: Type[nn.Module], lora_cls: Type[LoraModule]) -> None:
    """
    Register a new source module type to LoRA class mapping.

    Args:
        source_cls: The source module type (e.g., nn.Conv2d)
        lora_cls: The LoRA class that can replace it (e.g., LoraConv2d)

    Example:
        >>> register_lora_mapping(nn.Conv2d, LoraConv2d)
    """
    LORA_MAPPING[source_cls] = lora_cls
