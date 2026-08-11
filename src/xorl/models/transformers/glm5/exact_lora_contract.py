"""Shared adapter-shape contract for exact GLM-5.2 active LoRA."""

from __future__ import annotations


GLM52_EXACT_LORA_CONFIGS = frozenset({(1, 1), (16, 32)})


def glm52_exact_lora_scaling(rank: int, alpha: int) -> float:
    if (rank, alpha) not in GLM52_EXACT_LORA_CONFIGS:
        raise ValueError(
            "GLM-5.2 exact active LoRA requires rank=1 and alpha=1 or "
            f"rank=16 and alpha=32; got rank={rank} and alpha={alpha}"
        )
    return float(alpha) / float(rank)


__all__ = ["GLM52_EXACT_LORA_CONFIGS", "glm52_exact_lora_scaling"]
