"""LoRA utilities for MoE implementation.

This module provides utilities for initializing and managing LoRA weights
in the stacked tensor format used by group GEMM kernels.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


def init_lora_weights_stacked(
    num_experts: int,
    r: int,
    in_features: int,
    out_features: int,
    init_method: str = "kaiming",
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize stacked LoRA weights for all experts.

    Creates lora_A and lora_B tensors with appropriate initialization:
    - lora_A: Kaiming uniform or Gaussian initialization
    - lora_B: Zero initialization (ensures delta_W = 0 at start)

    Args:
        num_experts: Number of experts
        r: LoRA rank
        in_features: Input feature dimension
        out_features: Output feature dimension
        init_method: Initialization method ("kaiming" or "gaussian")
        dtype: Data type for the tensors
        device: Device for the tensors

    Returns:
        Tuple of (lora_A, lora_B) tensors:
        - lora_A: Shape [num_experts, r, in_features]
        - lora_B: Shape [num_experts, out_features, r]
    """
    # lora_A: projects input to low-rank space
    # Shape: [num_experts, r, in_features]
    lora_A = torch.empty(num_experts, r, in_features, dtype=dtype, device=device)

    # lora_B: projects from low-rank space to output
    # Shape: [num_experts, out_features, r]
    lora_B = torch.zeros(num_experts, out_features, r, dtype=dtype, device=device)

    # Initialize lora_A
    if init_method == "kaiming":
        for i in range(num_experts):
            # Initialize each expert's lora_A with kaiming uniform
            nn.init.kaiming_uniform_(lora_A[i], a=math.sqrt(5))
    elif init_method == "gaussian":
        nn.init.normal_(lora_A, std=1.0 / r)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    # lora_B is already zeros

    return lora_A, lora_B


def compute_lora_scaling(lora_alpha: int, r: int, use_rslora: bool = False) -> float:
    """Compute the LoRA scaling factor.

    Args:
        lora_alpha: LoRA alpha parameter
        r: LoRA rank
        use_rslora: Whether to use rank-stabilized LoRA scaling

    Returns:
        Scaling factor
    """
    if use_rslora:
        return lora_alpha / math.sqrt(r)
    else:
        return lora_alpha / r


def merge_lora_weights_stacked(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Merge LoRA weights into base weights.

    Computes: W' = W + B @ A * scaling

    Args:
        base_weight: Base weight tensor [num_experts, out_features, in_features]
        lora_A: LoRA A tensor [num_experts, r, in_features]
        lora_B: LoRA B tensor [num_experts, out_features, r]
        scaling: LoRA scaling factor

    Returns:
        Merged weight tensor [num_experts, out_features, in_features]
    """
    # B @ A: [num_experts, out_features, r] @ [num_experts, r, in_features]
    #      = [num_experts, out_features, in_features]
    delta_weight = torch.bmm(lora_B, lora_A) * scaling
    return base_weight + delta_weight


def unmerge_lora_weights_stacked(
    merged_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Unmerge LoRA weights from merged weights.

    Computes: W = W' - B @ A * scaling

    Args:
        merged_weight: Merged weight tensor [num_experts, out_features, in_features]
        lora_A: LoRA A tensor [num_experts, r, in_features]
        lora_B: LoRA B tensor [num_experts, out_features, r]
        scaling: LoRA scaling factor

    Returns:
        Base weight tensor [num_experts, out_features, in_features]
    """
    delta_weight = torch.bmm(lora_B, lora_A) * scaling
    return merged_weight - delta_weight


def get_lora_delta_weight_stacked(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Compute the LoRA weight delta.

    Computes: delta_W = B @ A * scaling

    Args:
        lora_A: LoRA A tensor [num_experts, r, in_features]
        lora_B: LoRA B tensor [num_experts, out_features, r]
        scaling: LoRA scaling factor

    Returns:
        Delta weight tensor [num_experts, out_features, in_features]
    """
    return torch.bmm(lora_B, lora_A) * scaling
