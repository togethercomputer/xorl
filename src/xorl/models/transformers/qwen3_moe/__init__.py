"""Qwen3 MoE model with Xorl MoE LoRA support."""

from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_qwen3_moe import (
    Qwen3MoeSparseExperts,
    Qwen3MoeFusedExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)

# Xorl MoE LoRA
from .qwen3_moe_lora import (
    LoRAConfig,
    Qwen3MoeSparseExpertsWithLoRA,
    Qwen3MoeFusedExpertsWithLoRA,
    Qwen3MoeSparseMoeBlockWithLoRA,
    Qwen3MoeSparseFusedMoeBlockWithLoRA,
    copy_weights_to_lora_experts,
    create_sparse_lora_experts_from_base,
    create_fused_lora_experts_from_base,
    mark_only_lora_as_trainable,
    lora_state_dict,
)

__all__ = [
    # Base model
    "Qwen3MoeConfig",
    "Qwen3MoeSparseExperts",
    "Qwen3MoeFusedExperts",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    # Xorl MoE LoRA
    "LoRAConfig",
    "Qwen3MoeSparseExpertsWithLoRA",
    "Qwen3MoeFusedExpertsWithLoRA",
    "Qwen3MoeSparseMoeBlockWithLoRA",
    "Qwen3MoeSparseFusedMoeBlockWithLoRA",
    "copy_weights_to_lora_experts",
    "create_sparse_lora_experts_from_base",
    "create_fused_lora_experts_from_base",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
]
