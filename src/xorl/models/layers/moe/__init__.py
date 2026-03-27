"""
MoE layer abstractions.

Provides model-agnostic building blocks for Mixture-of-Experts layers:

- :class:`MoEExperts` — weight container for expert parameters.
- :class:`MoEBlock` — composes gate + router + experts.
- :class:`TopKRouter` — token-choice routing (softmax + topk + renorm).
- :class:`MoEExpertsLoRA` — LoRA adapter injection for experts.
- :data:`MOE_EXPERT_BACKENDS` — backend registry (eager / triton / native / quack).
"""

from .aux_loss import global_load_balancing_loss_func
from .backend import MOE_EXPERT_BACKENDS
from .experts import MoEExperts
from .lora import (
    MoEExpertsLoRA,
    MoELoRAConfig,
    copy_weights_to_lora_experts,
    inject_lora_into_experts,
    lora_state_dict,
    mark_only_lora_as_trainable,
)
from .moe_block import MoEBlock
from .router import TopKRouter


__all__ = [
    "MoEExperts",
    "MoEBlock",
    "TopKRouter",
    "MoELoRAConfig",
    "MoEExpertsLoRA",
    "inject_lora_into_experts",
    "copy_weights_to_lora_experts",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
    "MOE_EXPERT_BACKENDS",
    "global_load_balancing_loss_func",
]
