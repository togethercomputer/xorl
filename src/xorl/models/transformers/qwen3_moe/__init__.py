"""Qwen3 MoE model with Xorl MoE LoRA support."""

from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3_moe import (
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)

# Xorl MoE LoRA
from .qwen3_moe_lora import (
    LoRAConfig,
    Qwen3MoeFusedExpertsWithLoRA,
    Qwen3MoeSparseMoeBlockWithLoRA,
    copy_weights_to_lora_experts,
    create_lora_experts_from_base,
    mark_only_lora_as_trainable,
    lora_state_dict,
)

__all__ = [
    # Base model
    "Qwen3MoeConfig",
    "Qwen3MoeExperts",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    # Xorl MoE LoRA
    "LoRAConfig",
    "Qwen3MoeFusedExpertsWithLoRA",
    "Qwen3MoeSparseMoeBlockWithLoRA",
    "copy_weights_to_lora_experts",
    "create_lora_experts_from_base",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
]
