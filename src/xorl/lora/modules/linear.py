"""
LoRA Linear layer implementation.

Simple, flat-structure LoRA layer designed for FSDP compatibility.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LoraModule


class LoraLinear(LoraModule, nn.Linear):
    """
    LoRA Linear layer that extends nn.Linear with low-rank adaptation.

    Inherits from nn.Linear for compatibility with code that checks
    isinstance(layer, nn.Linear). Adds LoRA parameters (lora_A, lora_B)
    for parameter-efficient fine-tuning.

    Unlike PEFT's implementation which uses nested ModuleDict for multi-adapter
    support, this implementation uses flat parameters for simplicity and better
    FSDP compatibility.

    Structure:
        - weight: [out_features, in_features] (frozen base weight, from nn.Linear)
        - bias: [out_features] (optional, frozen, from nn.Linear)
        - lora_A: [r, in_features] (trainable, down-projection)
        - lora_B: [out_features, r] (trainable, up-projection)

    Forward computation:
        output = F.linear(x, weight, bias) + (x @ lora_A.T @ lora_B.T) * scaling
               = W @ x + B @ A @ x * (alpha / r)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        r: LoRA rank (low-rank dimension)
        lora_alpha: LoRA alpha for scaling (scaling = alpha / r)
        bias: Whether to include bias
        device: Device for parameters
        dtype: Data type for base weight (LoRA weights are always float32)

    Example:
        >>> layer = LoraLinear(768, 768, r=16, lora_alpha=32)
        >>> x = torch.randn(2, 10, 768)
        >>> output = layer(x)  # [2, 10, 768]
        >>> isinstance(layer, nn.Linear)  # True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # Initialize nn.Linear (handles weight and bias)
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)

        # LoRA-specific attributes
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # LoRA weights (trainable, float32 for numerical stability)
        # lora_A: down-projection [r, in_features]
        # lora_B: up-projection [out_features, r]
        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, r, device=device, dtype=torch.float32)
        )

        # Initialize LoRA parameters
        self.reset_lora_parameters()

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        r: int,
        lora_alpha: int,
        **kwargs,
    ) -> "LoraLinear":
        """
        Create LoraLinear from an existing nn.Linear layer.

        Copies the base weights and freezes them, initializes LoRA weights.

        Args:
            module: Source nn.Linear layer
            r: LoRA rank
            lora_alpha: LoRA alpha for scaling
            **kwargs: Unused, for interface compatibility

        Returns:
            LoraLinear with copied base weights (frozen) and initialized LoRA weights

        Raises:
            AssertionError: If module is not an nn.Linear

        Example:
            >>> base_linear = nn.Linear(768, 768)
            >>> lora_layer = LoraLinear.from_module(base_linear, r=16, lora_alpha=32)
        """
        assert isinstance(module, nn.Linear), f"Expected nn.Linear, got {type(module)}"

        linear = module  # type alias for clarity

        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # Copy base weights
        with torch.no_grad():
            lora_linear.weight.copy_(linear.weight)
            if linear.bias is not None and lora_linear.bias is not None:
                lora_linear.bias.copy_(linear.bias)

        # Freeze base weights
        lora_linear.weight.requires_grad = False
        if lora_linear.bias is not None:
            lora_linear.bias.requires_grad = False

        return lora_linear

    def reset_lora_parameters(self) -> None:
        """
        Initialize LoRA parameters following Microsoft's original implementation.

        - lora_A: Kaiming uniform initialization (same as nn.Linear default)
        - lora_B: Zero initialization (ensures LoRA starts as identity/no-op)

        Note: Base weight/bias are initialized by nn.Linear.reset_parameters().
        """
        # LoRA A: kaiming uniform (Microsoft LoRA default)
        # This matches nn.Linear's default initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA B: zeros (ensures output starts unchanged)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Base linear transformation
        result = F.linear(x, self.weight, self.bias)

        # LoRA path: A -> B -> scale
        # Compute in float32 for numerical stability
        x_lora = x.to(self.lora_A.dtype)

        # x_lora @ lora_A.T @ lora_B.T * scaling
        # = F.linear(F.linear(x_lora, lora_A), lora_B) * scaling
        lora_out = F.linear(F.linear(x_lora, self.lora_A), self.lora_B)
        lora_out = lora_out * self.scaling

        # Add LoRA output to result (cast back to result dtype)
        return result + lora_out.to(result.dtype)

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight from LoRA matrices.

        Returns:
            Delta weight tensor: lora_B @ lora_A * scaling
        """
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base weight for inference.

        After merging: weight = weight + lora_B @ lora_A * scaling

        This is useful for inference when you want to eliminate the
        LoRA computation overhead.

        Warning: This modifies the weight in-place and is not reversible.
        """
        with torch.no_grad():
            # delta_weight = lora_B @ lora_A * scaling
            delta_weight = self.get_delta_weight()
            self.weight.add_(delta_weight.to(self.weight.dtype))

    def extra_repr(self) -> str:
        # Extend nn.Linear's extra_repr with LoRA-specific info
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, r={self.r}, lora_alpha={self.lora_alpha}"
        )
