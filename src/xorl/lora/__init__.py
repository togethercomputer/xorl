"""
Xorl LoRA (Low-Rank Adaptation) module.

This package uses lazy exports to avoid circular imports between:
`xorl.models.layers.moe.*`, `xorl.lora.mapping`, and `xorl.lora.utils`.
"""

from xorl.lora.modules import LoraLinear, LoraModule
from xorl.ops.group_gemm.kernel import (
    compute_lora_scaling,
    get_lora_delta_weight_stacked,
    init_lora_weights_stacked,
    merge_lora_weights_stacked,
    unmerge_lora_weights_stacked,
)


_MAPPING_ATTRS = {
    "LORA_MAPPING",
    "can_apply_lora",
    "get_lora_class_for_module",
    "register_lora_mapping",
}

_UTIL_ATTRS = {
    "count_lora_parameters",
    "freeze_base_parameters",
    "get_all_lora_state_dict",
    "get_lora_parameters",
    "get_lora_state_dict",
    "get_moe_lora_state_dict",
    "inject_lora_into_model",
    "inject_lora_into_model_with_moe",
    "inject_lora_into_moe_blocks",
    "load_lora_checkpoint",
    "load_lora_state_dict",
    "save_lora_checkpoint",
}

_MOE_LAZY_ATTRS = {
    "MoELoRAConfig",
    "MoEExpertsLoRA",
    "copy_weights_to_lora_experts",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
}


__all__ = [
    "LoraModule",
    "LoraLinear",
    "LORA_MAPPING",
    "can_apply_lora",
    "get_lora_class_for_module",
    "register_lora_mapping",
    "inject_lora_into_model",
    "inject_lora_into_model_with_moe",
    "inject_lora_into_moe_blocks",
    "get_lora_state_dict",
    "get_moe_lora_state_dict",
    "get_all_lora_state_dict",
    "load_lora_state_dict",
    "freeze_base_parameters",
    "get_lora_parameters",
    "count_lora_parameters",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "MoELoRAConfig",
    "MoEExpertsLoRA",
    "copy_weights_to_lora_experts",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
    "init_lora_weights_stacked",
    "compute_lora_scaling",
    "merge_lora_weights_stacked",
    "unmerge_lora_weights_stacked",
    "get_lora_delta_weight_stacked",
]


def __getattr__(name):
    if name in _MAPPING_ATTRS:
        from xorl.lora.mapping import (  # noqa: PLC0415
            LORA_MAPPING,
            can_apply_lora,
            get_lora_class_for_module,
            register_lora_mapping,
        )

        g = globals()
        g["LORA_MAPPING"] = LORA_MAPPING
        g["can_apply_lora"] = can_apply_lora
        g["get_lora_class_for_module"] = get_lora_class_for_module
        g["register_lora_mapping"] = register_lora_mapping
        return g[name]

    if name in _UTIL_ATTRS:
        from xorl.lora.utils import (  # noqa: PLC0415
            count_lora_parameters,
            freeze_base_parameters,
            get_all_lora_state_dict,
            get_lora_parameters,
            get_lora_state_dict,
            get_moe_lora_state_dict,
            inject_lora_into_model,
            inject_lora_into_model_with_moe,
            inject_lora_into_moe_blocks,
            load_lora_checkpoint,
            load_lora_state_dict,
            save_lora_checkpoint,
        )

        g = globals()
        g["count_lora_parameters"] = count_lora_parameters
        g["freeze_base_parameters"] = freeze_base_parameters
        g["get_all_lora_state_dict"] = get_all_lora_state_dict
        g["get_lora_parameters"] = get_lora_parameters
        g["get_lora_state_dict"] = get_lora_state_dict
        g["get_moe_lora_state_dict"] = get_moe_lora_state_dict
        g["inject_lora_into_model"] = inject_lora_into_model
        g["inject_lora_into_model_with_moe"] = inject_lora_into_model_with_moe
        g["inject_lora_into_moe_blocks"] = inject_lora_into_moe_blocks
        g["load_lora_checkpoint"] = load_lora_checkpoint
        g["load_lora_state_dict"] = load_lora_state_dict
        g["save_lora_checkpoint"] = save_lora_checkpoint
        return g[name]

    if name in _MOE_LAZY_ATTRS:
        from xorl.models.layers.moe.lora import (  # noqa: PLC0415
            MoEExpertsLoRA,
            MoELoRAConfig,
            copy_weights_to_lora_experts,
            lora_state_dict,
            mark_only_lora_as_trainable,
        )

        g = globals()
        g["MoELoRAConfig"] = MoELoRAConfig
        g["MoEExpertsLoRA"] = MoEExpertsLoRA
        g["copy_weights_to_lora_experts"] = copy_weights_to_lora_experts
        g["mark_only_lora_as_trainable"] = mark_only_lora_as_trainable
        g["lora_state_dict"] = lora_state_dict
        return g[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
