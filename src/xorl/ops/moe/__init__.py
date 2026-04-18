"""MoE ops subpackage — re-exports for backward compatibility."""

from .lora import (
    make_ep_lora_compute,
    make_local_lora_compute,
)
from .quack import QuackEPGroupGemm, quack_moe_forward
from .triton import TritonEPGroupGemm, TritonMoeExpertsFunction, triton_moe_forward
from .triton_lora import (
    TritonEPGroupGemmWithLoRA,
    TritonMoeExpertsLoRAFunction,
    triton_moe_lora_forward,
)


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
