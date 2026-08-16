"""GLM-5 support helpers shared by model construction and layers."""

from xorl.models.exact_contract import glm52_exact_active_lora_enabled


GLM5_DSA_RING_ATTENTION_UNSUPPORTED_MESSAGE = (
    "GLM-5 DSA attention supports Ulysses sequence parallelism but not ring attention yet. "
    "Set ringattn_parallel_size=1 for GLM-5 DSA runs, or set config._dsa_mask_disabled=True "
    "only for debug dense-MLA runs."
)

GLM5_LORA_TARGET_MODULES = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def is_glm5_config(config) -> bool:
    return getattr(config, "model_type", None) == "xorl_glm5"


def validate_glm5_sequence_parallel(config, *, parallel_state=None, cp_enabled: bool | None = None) -> None:
    if getattr(config, "model_type", None) != "xorl_glm5":
        return
    if cp_enabled is None:
        cp_enabled = bool(getattr(parallel_state, "cp_enabled", False))
    if not cp_enabled:
        return
    if getattr(config, "_dsa_mask_disabled", False):
        return
    if bool(getattr(parallel_state, "ringattn_enabled", False)) or getattr(parallel_state, "ringattn_size", 1) > 1:
        raise ValueError(GLM5_DSA_RING_ATTENTION_UNSUPPORTED_MESSAGE)


def validate_glm5_router_settings(config, *, train_router: bool) -> None:
    if not is_glm5_config(config):
        return
    if train_router:
        raise ValueError("GLM-5 does not support train_router=True in xorl yet.")


def validate_glm5_training_mode(
    config,
    *,
    enable_qlora: bool,
    freeze_router: bool,
    merge_qkv: bool,
    block_fp8_qlora_training: bool = False,
    quant_format: str = "nvfp4",
    quant_group_size: int = 16,
    moe_implementation: str | None = None,
    ep_dispatch: str = "alltoall",
    moe_hybrid_shared_lora: bool = False,
    glm52_fullparam_fp8_training: bool = False,
) -> None:
    if not is_glm5_config(config):
        return
    if glm52_fullparam_fp8_training:
        if block_fp8_qlora_training or enable_qlora:
            raise ValueError(
                "GLM-5.2 full-param block-FP8 training is a full-weight mode and cannot be combined with QLoRA"
            )
        requirements = {
            "moe_implementation": (moe_implementation, "triton"),
            "ep_dispatch": (ep_dispatch, "alltoall"),
            # Full-param mode TRAINS the router through its FP32 master;
            # freeze_router=True would misdescribe the run.
            "freeze_router": (freeze_router, False),
            "merge_qkv": (merge_qkv, True),
        }
        mismatches = [
            f"{name}={actual!r} (requires {expected!r})"
            for name, (actual, expected) in requirements.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "GLM-5.2 full-param block-FP8 training rejects unsupported configuration: " + ", ".join(mismatches)
            )
        return
    if block_fp8_qlora_training:
        exact_active_lora = glm52_exact_active_lora_enabled(config)
        requirements = {
            "enable_qlora": (enable_qlora, True),
            "quant_format": (quant_format, "block_fp8"),
            "quant_group_size": (quant_group_size, 128),
            "moe_implementation": (moe_implementation, "triton"),
            "ep_dispatch": (ep_dispatch, "alltoall" if exact_active_lora else "deepep"),
            "moe_hybrid_shared_lora": (moe_hybrid_shared_lora, True),
        }
        mismatches = [
            f"{name}={actual!r} (requires {expected!r})"
            for name, (actual, expected) in requirements.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("GLM-5.2 block-FP8 QLoRA rejects unsupported configuration: " + ", ".join(mismatches))
    elif enable_qlora:
        raise ValueError("GLM-5 QLoRA requires the explicit block_fp8_qlora_training mode")
    if not freeze_router:
        raise ValueError("GLM-5 requires freeze_router=True.")
    if not merge_qkv:
        raise ValueError("GLM-5 does not support merge_qkv=False yet.")


def glm5_default_lora_targets(*, train_attn: bool, train_mlp: bool, train_unembed: bool) -> list[str]:
    targets: list[str] = []
    if train_attn:
        targets.extend(GLM5_LORA_TARGET_MODULES[:5])
    if train_mlp:
        targets.extend(GLM5_LORA_TARGET_MODULES[5:])
    if train_unembed:
        targets.append("lm_head")
    return targets


__all__ = [
    "GLM5_DSA_RING_ATTENTION_UNSUPPORTED_MESSAGE",
    "GLM5_LORA_TARGET_MODULES",
    "glm5_default_lora_targets",
    "is_glm5_config",
    "validate_glm5_router_settings",
    "validate_glm5_sequence_parallel",
    "validate_glm5_training_mode",
]
