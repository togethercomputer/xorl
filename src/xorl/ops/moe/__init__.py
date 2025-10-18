"""MoE ops subpackage — re-exports for backward compatibility."""

from .triton import TritonMoeExpertsFunction, triton_moe_forward, TritonEPGroupGemm
from .triton_lora import (
    TritonMoeExpertsLoRAFunction,
    triton_moe_lora_forward,
    TritonEPGroupGemmWithLoRA,
)
from .lora import (
    make_ep_lora_compute,
    make_local_lora_compute,
)
from .quack import quack_moe_forward, QuackEPGroupGemm

try:
    from .quack_lora import QuackEPGroupGemmWithLoRA, quack_moe_lora_forward
except ImportError:
    pass

__all__ = [
    "TritonMoeExpertsFunction",
    "triton_moe_forward",
    "TritonEPGroupGemm",
    "TritonMoeExpertsLoRAFunction",
    "triton_moe_lora_forward",
    "TritonEPGroupGemmWithLoRA",
    "make_ep_lora_compute",
    "make_local_lora_compute",
    "quack_moe_forward",
    "QuackEPGroupGemm",
    "QuackEPGroupGemmWithLoRA",
    "quack_moe_lora_forward",
]
