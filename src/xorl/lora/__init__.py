"""
Xorl LoRA (Low-Rank Adaptation) module.

This module provides a simple, FSDP-compatible LoRA implementation
with flat parameter structure for easy checkpoint management.
"""

from xorl.lora.layers import LoraLinear
from xorl.lora.utils import (
    inject_lora_into_model,
    get_lora_state_dict,
    load_lora_state_dict,
    freeze_base_parameters,
    get_lora_parameters,
    save_lora_checkpoint,
    load_lora_checkpoint,
    count_lora_parameters,
)

__all__ = [
    "LoraLinear",
    "inject_lora_into_model",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "freeze_base_parameters",
    "get_lora_parameters",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "count_lora_parameters",
]
