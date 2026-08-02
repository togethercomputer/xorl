"""LoRA-only linear deltas for fused projections whose base stays separate."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.lora.fold import (
    FoldedLoraWeightLinear,
    canonical_lora_fold_linear,
    lora_merged_cache_enabled,
)

from .base import LoraModule


class LoraDeltaLinear(LoraModule, nn.Module):
    """A low-rank delta with no owned base weight.

    Qwen3.5's trainer keeps GDN q/k/v/z base projections as separate linears,
    while River and SGLang attach one LoRA to their fused ``in_proj_qkvz``.
    This module represents exactly that fused delta without duplicating or
    renaming the trainer's base weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.active_r = self.r
        self.active_lora_alpha = self.lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r, device=device, dtype=dtype))
        self.reset_lora_parameters()

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        r: int,
        lora_alpha: int,
        **kwargs,
    ) -> "LoraDeltaLinear":
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
        return cls(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=lora_alpha,
            device=module.weight.device,
            dtype=kwargs.get("dtype", torch.float32),
        )

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_runtime_lora_config(self, lora_rank: int, lora_alpha: int) -> None:
        if lora_rank <= 0 or lora_rank > self.r:
            raise ValueError(f"Active LoRA rank must be in [1, {self.r}], got {lora_rank}")
        self.active_r = int(lora_rank)
        self.active_lora_alpha = int(lora_alpha)

    def _active_scaling(self) -> float:
        return self.active_lora_alpha / self.active_r

    def get_delta_weight(self) -> torch.Tensor:
        return (self.lora_B[:, : self.active_r] @ self.lora_A[: self.active_r]) * self._active_scaling()

    def invalidate_merged_weight_cache(self) -> None:
        self._merged_weight_cache = {}

    def _merged_weight(
        self,
        base_weight: torch.Tensor,
        *,
        output_start: int = 0,
        output_end: int | None = None,
    ) -> torch.Tensor:
        """Canonically fold one fused-output slice into its separate base weight."""
        end = self.out_features if output_end is None else int(output_end)
        start = int(output_start)
        if start < 0 or end <= start or end > self.out_features:
            raise ValueError(f"Invalid fused LoRA output slice [{start}:{end}] for {self.out_features} features")
        lora_A = self.lora_A[: self.active_r]
        lora_B = self.lora_B[start:end, : self.active_r]
        if tuple(base_weight.shape) != (end - start, self.in_features):
            raise ValueError(
                f"Base weight shape {tuple(base_weight.shape)} does not match fused LoRA slice "
                f"[{start}:{end}] -> {(end - start, self.in_features)}"
            )
        cache = getattr(self, "_merged_weight_cache", None)
        if cache is None:
            cache = {}
            self._merged_weight_cache = cache
        generation = (
            tuple(t._version for t in (self.lora_A, self.lora_B)),
            tuple(t.data_ptr() for t in (self.lora_A, self.lora_B)),
            self.active_r,
            self.active_lora_alpha,
        )
        # AdapterManager.prepare_forward() copies the active adapter into these
        # parameters for every top-level forward_backward request, incrementing
        # their versions even while accumulating into the same optimizer step.
        # Evict the previous generation so serialized requests cannot retain one
        # complete set of folded GDN weights each.
        slice_key = (start, end)
        base_key = (base_weight._version, base_weight.data_ptr())
        if lora_merged_cache_enabled():
            if cache.get("generation") != generation:
                cache.clear()
                cache["generation"] = generation
                cache["slices"] = {}
            cached_slice = cache["slices"].get(slice_key)
            if cached_slice is not None and cached_slice["base_key"] == base_key:
                return cached_slice["weight"]
        else:
            cache.clear()
        with torch.no_grad():
            folded = canonical_lora_fold_linear(base_weight, lora_A, lora_B, self._active_scaling())
        if lora_merged_cache_enabled():
            cache["slices"][slice_key] = {"base_key": base_key, "weight": folded}
        return folded

    def merged_weight_for_forward(
        self,
        base_weight: torch.Tensor,
        *,
        output_start: int = 0,
        output_end: int | None = None,
    ) -> torch.Tensor:
        """Return canonical forward bits with gradients flowing into the fused factors."""
        end = self.out_features if output_end is None else int(output_end)
        start = int(output_start)
        lora_A = self.lora_A[: self.active_r]
        lora_B = self.lora_B[start:end, : self.active_r]
        folded = self._merged_weight(base_weight, output_start=start, output_end=end)
        if torch.is_grad_enabled() and (lora_A.requires_grad or lora_B.requires_grad):
            return FoldedLoraWeightLinear.apply(folded, lora_A, lora_B, self._active_scaling())
        return folded

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_lora = inputs.to(self.lora_A.dtype)
        delta = F.linear(
            F.linear(inputs_lora, self.lora_A[: self.active_r]),
            self.lora_B[:, : self.active_r],
        )
        return (delta * self._active_scaling()).to(inputs.dtype)

    def merge_weights(self) -> None:
        raise RuntimeError("LoraDeltaLinear has no owned base weight; fold it through its parent projection")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.active_r}, max_r={self.r}, "
            f"lora_alpha={self.active_lora_alpha}"
        )
