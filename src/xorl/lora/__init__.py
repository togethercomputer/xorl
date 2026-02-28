"""
Xorl LoRA (Low-Rank Adaptation) module.

This module provides a simple, FSDP-compatible LoRA implementation
with flat parameter structure for easy checkpoint management.

Supports both dense layers (nn.Linear) and MoE expert blocks
with Xorl group GEMM kernels for efficient forward/backward passes.
"""

# Base class and implementations
from xorl.lora.modules import LoraModule, LoraLinear

# Mapping
from xorl.lora.mapping import (
    LORA_MAPPING,
    can_apply_lora,
    get_lora_class_for_module,
    register_lora_mapping,
)

# Utility functions
from xorl.lora.utils import (
    inject_lora_into_model,
    inject_lora_into_model_with_moe,
    inject_lora_into_moe_blocks,
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

# MoE LoRA — deferred to avoid circular import with xorl.models.layers.moe.
# Access via xorl.lora.MoEExpertsLoRA etc. triggers __getattr__ below.

# Xorl ops for direct access
from xorl.ops.group_gemm.kernel import (
    init_lora_weights_stacked,
    compute_lora_scaling,
    merge_lora_weights_stacked,
    unmerge_lora_weights_stacked,
    get_lora_delta_weight_stacked,
)

__all__ = [
    # Base class
    "LoraModule",
    # Dense LoRA
    "LoraLinear",
    # Mapping
    "LORA_MAPPING",
    "can_apply_lora",
    "get_lora_class_for_module",
    "register_lora_mapping",
    # Injection utilities
    "inject_lora_into_model",
    "inject_lora_into_model_with_moe",
    "inject_lora_into_moe_blocks",
    # State dict utilities
    "get_lora_state_dict",
    "get_moe_lora_state_dict",
    "get_all_lora_state_dict",
    "load_lora_state_dict",
    # Parameter utilities
    "freeze_base_parameters",
    "get_lora_parameters",
    "count_lora_parameters",
    # Checkpoint utilities
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    # MoE LoRA (lazy)
    "MoELoRAConfig",
    "MoEExpertsLoRA",
    "copy_weights_to_lora_experts",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
    # Xorl ops
    "init_lora_weights_stacked",
    "compute_lora_scaling",
    "merge_lora_weights_stacked",
    "unmerge_lora_weights_stacked",
    "get_lora_delta_weight_stacked",
]


# Lazy imports for MoE LoRA to break circular import:
# moe/lora.py → xorl.lora.modules.base → xorl.lora → xorl.models.layers.moe (cycle)
_MOE_LAZY_ATTRS = {
    "MoELoRAConfig", "MoEExpertsLoRA",
    "copy_weights_to_lora_experts", "mark_only_lora_as_trainable", "lora_state_dict",
}


def __getattr__(name):
    if name in _MOE_LAZY_ATTRS:
        from xorl.models.layers.moe.lora import (
            MoELoRAConfig,
            MoEExpertsLoRA,
            copy_weights_to_lora_experts,
            mark_only_lora_as_trainable,
            lora_state_dict,
        )
        # Cache all at once to avoid repeated imports
        g = globals()
        g["MoELoRAConfig"] = MoELoRAConfig
        g["MoEExpertsLoRA"] = MoEExpertsLoRA
        g["copy_weights_to_lora_experts"] = copy_weights_to_lora_experts
        g["mark_only_lora_as_trainable"] = mark_only_lora_as_trainable
        g["lora_state_dict"] = lora_state_dict
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
