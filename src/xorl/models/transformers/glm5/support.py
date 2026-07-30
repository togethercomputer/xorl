"""GLM-5 support helpers shared by model construction and layers."""

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
) -> None:
    if not is_glm5_config(config):
        return
    if enable_qlora:
        raise ValueError("GLM-5 does not support enable_qlora=True yet.")
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
