"""Helpers for independent LoRA factors over fused base projections."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from xorl.lora.fold import lora_merged_forward_enabled


def project_fused_linear_with_lora(
    module,
    inputs: torch.Tensor,
    *,
    base_name: str,
    projection_names: Sequence[str],
    projection_sizes: Sequence[int],
    linear: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor] = F.linear,
) -> torch.Tensor:
    """Run one fused base GEMM with optional independent logical adapters.

    Delta-only children live on the parent under their logical projection names.
    Dynamic mode adds their outputs to the matching fused slices. Exact merged mode
    canonically folds each delta into its base slice and still issues one fused GEMM.
    """
    if len(projection_names) != len(projection_sizes):
        raise ValueError("projection_names and projection_sizes must have equal length")

    base = getattr(module, base_name)
    adapters = [getattr(module, name, None) for name in projection_names]
    present = [adapter for adapter in adapters if adapter is not None]
    if not present:
        return base(inputs)

    merged = [lora_merged_forward_enabled(adapter) for adapter in present]
    if any(merged):
        if not all(merged):
            raise RuntimeError(f"{base_name} logical adapters must select merged forward together")
        base_parts = base.weight.split(tuple(int(size) for size in projection_sizes), dim=0)
        folded_parts = [
            adapter.merged_weight_for_forward(base_part) if adapter is not None else base_part
            for adapter, base_part in zip(adapters, base_parts, strict=True)
        ]
        return linear(inputs, torch.cat(folded_parts, dim=0), base.bias)

    output_parts = list(base(inputs).split(tuple(int(size) for size in projection_sizes), dim=-1))
    for index, adapter in enumerate(adapters):
        if adapter is not None:
            output_parts[index] = output_parts[index] + adapter(inputs).to(output_parts[index].dtype)
    return torch.cat(output_parts, dim=-1)
