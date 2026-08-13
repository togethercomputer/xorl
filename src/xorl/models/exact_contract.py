"""Dependency-free predicates for exact model value programs."""

from __future__ import annotations

from typing import Optional


#: Family identifiers for the exact train/serve value programs. These strings
#: are the shared vocabulary between the model-resolution stamp
#: (``config._exact_contract_family``) and every family-keyed contract site
#: (for example the PP byte-boundary contract).
EXACT_CONTRACT_FAMILY_GLM52 = "glm52"
EXACT_CONTRACT_FAMILY_QWEN35_DENSE = "qwen3_5_dense"
EXACT_CONTRACT_FAMILY_QWEN35_MOE = "qwen3_5_moe"

_QWEN35_MOE_MODEL_TYPES = frozenset(
    {
        "xorl_qwen3_5_moe",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def resolve_exact_contract_family(config: object | None) -> Optional[str]:
    """Classify the exact value program selected by ``config``.

    Returns ``None`` for generic models (no byte contract claimed), otherwise
    one of ``EXACT_CONTRACT_FAMILY_GLM52``, ``EXACT_CONTRACT_FAMILY_QWEN35_DENSE``,
    or ``EXACT_CONTRACT_FAMILY_QWEN35_MOE``. Model resolution stamps the result
    on ``config._exact_contract_family``; consumers should prefer that stamp
    and fall back to this resolver only for configs that predate stamping.
    """

    if config is None:
        return None
    if glm52_exact_forward_enabled(config):
        return EXACT_CONTRACT_FAMILY_GLM52
    if bool(getattr(config, "_qwen35_exact_contract", False)):
        # Hugging Face multimodal checkpoints expose the language-model
        # geometry under ``text_config``.
        scope = getattr(config, "text_config", None) or config
        if getattr(config, "model_type", None) in _QWEN35_MOE_MODEL_TYPES or getattr(scope, "num_experts", None):
            return EXACT_CONTRACT_FAMILY_QWEN35_MOE
        return EXACT_CONTRACT_FAMILY_QWEN35_DENSE
    return None


GLM52_EXACT_ACTIVE_LORA_FLAGS = (
    "_glm52_exact_active_lora_dense_component",
    "_glm52_exact_active_lora_attention_component",
    "_glm52_exact_active_lora_shared_expert_component",
    "_glm52_exact_active_lora_routed_expert_component",
    "_glm52_exact_active_lora_lm_head_component",
)


def set_glm52_exact_active_lora(config: object, *, enabled: bool) -> None:
    """Atomically select or clear the complete internal active-LoRA family.

    These flags describe one indivisible value program.  Always writing every
    member prevents stale checkpoint/config attributes from selecting a
    partial family and keeps them out of the user-facing configuration surface.
    """

    for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS:
        setattr(config, flag, bool(enabled))


def glm52_exact_active_lora_enabled(config: object | None) -> bool:
    """Require the complete internal active-LoRA family, never a partial set."""

    return config is not None and all(bool(getattr(config, flag, False)) for flag in GLM52_EXACT_ACTIVE_LORA_FLAGS)


def glm52_exact_forward_enabled(config: object | None) -> bool:
    """Select either the scoring-only or complete active-LoRA exact program."""

    return bool(config is not None and getattr(config, "_glm52_exact_contract", False)) or (
        glm52_exact_active_lora_enabled(config)
    )


def contains_glm52_exact_active_lora_component(module: object | None) -> bool:
    """Return whether a module tree contains any exact active-LoRA value component."""

    iter_modules = getattr(module, "modules", None)
    if not callable(iter_modules):
        return False
    return any(bool(getattr(candidate, "_glm52_exact_active_lora_component", False)) for candidate in iter_modules())


__all__ = [
    "EXACT_CONTRACT_FAMILY_GLM52",
    "EXACT_CONTRACT_FAMILY_QWEN35_DENSE",
    "EXACT_CONTRACT_FAMILY_QWEN35_MOE",
    "GLM52_EXACT_ACTIVE_LORA_FLAGS",
    "contains_glm52_exact_active_lora_component",
    "glm52_exact_active_lora_enabled",
    "glm52_exact_forward_enabled",
    "resolve_exact_contract_family",
    "set_glm52_exact_active_lora",
]
