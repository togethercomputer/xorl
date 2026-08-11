"""Shared adapter-shape contract for exact GLM-5.2 active LoRA."""

from __future__ import annotations


def glm52_exact_lora_scaling(rank: int, alpha: int) -> float:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(
            "GLM-5.2 exact active LoRA requires a positive integer rank; "
            f"got rank={rank!r}"
        )
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha <= 0:
        raise ValueError(
            "GLM-5.2 exact active LoRA requires a positive integer alpha; "
            f"got alpha={alpha!r}"
        )
    return float(alpha) / float(rank)


__all__ = ["glm52_exact_lora_scaling"]
