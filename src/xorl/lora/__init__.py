"""
Xorl LoRA (Low-Rank Adaptation) module.

This module provides a simple, FSDP-compatible LoRA implementation
with flat parameter structure for easy checkpoint management.

Supports both dense layers (nn.Linear) and MoE expert blocks
with Xorl group GEMM kernels for efficient forward/backward passes.
"""

from xorl.lora.layers import LoraLinear
from xorl.lora.moe_layers import (
    MoELoraLayer,
    MoEFusedLoraLayer,
    Qwen3MoELoraAdapter,
)
from xorl.lora.utils import (
    inject_lora_into_model,
    inject_lora_into_moe_blocks,
    inject_lora_into_model_with_moe,
    get_lora_state_dict,
    get_moe_lora_state_dict,
    get_all_lora_state_dict,
    load_lora_state_dict,
    freeze_base_parameters,
    get_lora_parameters,
    save_lora_checkpoint,
    load_lora_checkpoint,
    count_lora_parameters,
)

# Xorl-based MoE LoRA (efficient group GEMM kernels)
from xorl.models.transformers.qwen3_moe.qwen3_moe_lora import (
    LoRAConfig,
    Qwen3MoeFusedExpertsWithLoRA,
    Qwen3MoeSparseMoeBlockWithLoRA,
    copy_weights_to_lora_experts,
    create_lora_experts_from_base,
    mark_only_lora_as_trainable,
    lora_state_dict,
)

# Xorl ops for direct access
from xorl.ops.group_gemm.kernel import (
    init_lora_weights_stacked,
    compute_lora_scaling,
    merge_lora_weights_stacked,
    unmerge_lora_weights_stacked,
    get_lora_delta_weight_stacked,
)

__all__ = [
    # Dense LoRA
    "LoraLinear",
    "inject_lora_into_model",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "freeze_base_parameters",
    "get_lora_parameters",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "count_lora_parameters",
    # MoE LoRA (weight-merging approach)
    "MoELoraLayer",
    "MoEFusedLoraLayer",
    "Qwen3MoELoraAdapter",
    "inject_lora_into_moe_blocks",
    "inject_lora_into_model_with_moe",
    "get_moe_lora_state_dict",
    "get_all_lora_state_dict",
    # Xorl MoE LoRA (inline computation - recommended)
    "LoRAConfig",
    "Qwen3MoeFusedExpertsWithLoRA",
    "Qwen3MoeSparseMoeBlockWithLoRA",
    "copy_weights_to_lora_experts",
    "create_lora_experts_from_base",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
    # Xorl ops
    "init_lora_weights_stacked",
    "compute_lora_scaling",
    "merge_lora_weights_stacked",
    "unmerge_lora_weights_stacked",
    "get_lora_delta_weight_stacked",
]
