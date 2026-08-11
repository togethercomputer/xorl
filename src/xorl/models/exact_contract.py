"""Dependency-free predicates for exact model value programs."""

from __future__ import annotations


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


def contains_dsv4_exact_active_lora_component(module: object | None) -> bool:
    """Return whether a module tree belongs to the exact DSV4 active-LoRA program."""

    if module is None:
        return False
    if bool(getattr(module, "_dsv4_flash_exact_active_lora_component", False)):
        return True
    config = getattr(module, "config", None)
    if bool(getattr(config, "_dsv4_flash_exact_active_lora", False)):
        return True
    iter_modules = getattr(module, "modules", None)
    if not callable(iter_modules):
        return False
    return any(
        bool(getattr(candidate, "_dsv4_flash_exact_active_lora_component", False)) for candidate in iter_modules()
    )


__all__ = [
    "GLM52_EXACT_ACTIVE_LORA_FLAGS",
    "contains_dsv4_exact_active_lora_component",
    "contains_glm52_exact_active_lora_component",
    "glm52_exact_active_lora_enabled",
    "glm52_exact_forward_enabled",
    "set_glm52_exact_active_lora",
]
