"""
LoRA module base class.

Defines the abstract interface that all LoRA module implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Iterator, TypeVar

import torch
import torch.nn as nn


# Type variable for LoRA module classes
T = TypeVar("T", bound="LoraModule")


class LoraModule(ABC):
    """
    Abstract base class for LoRA (Low-Rank Adaptation) modules.

    All LoRA module implementations should inherit from this class and implement
    the required abstract methods. This ensures a consistent interface across
    different LoRA module types (Linear, Conv2d, MoE experts, etc.).

    Required attributes (set by subclasses):
        r: LoRA rank (low-rank dimension)
        lora_alpha: LoRA alpha for scaling
        scaling: Computed scaling factor (lora_alpha / r)

    Example:
        >>> class LoraLinear(LoraModule, nn.Linear):
        ...     @classmethod
        ...     def from_module(cls, module, r, lora_alpha, **kwargs):
        ...         # Create LoraLinear from nn.Linear
        ...         ...
    """

    # These should be set by subclasses
    r: int
    lora_alpha: int
    scaling: float

    @classmethod
    @abstractmethod
    def from_module(
        cls: type[T],
        module: nn.Module,
        r: int,
        lora_alpha: int,
        **kwargs,
    ) -> T:
        """
        Create a LoRA module from an existing module, copying its weights.

        This is the primary factory method for creating LoRA modules. It should:
        1. Create a new LoRA module with the same configuration as the original
        2. Copy the base weights from the original module
        3. Freeze the base weights (requires_grad=False)
        4. Initialize LoRA parameters (A with kaiming, B with zeros)

        Args:
            module: The original module to wrap with LoRA
            r: LoRA rank (low-rank dimension)
            lora_alpha: LoRA alpha for scaling (scaling = alpha / r)
            **kwargs: Additional arguments for specific LoRA implementations

        Returns:
            A new LoRA module with copied (frozen) base weights and initialized LoRA parameters

        Raises:
            AssertionError: If the module type is not supported

        Example:
            >>> linear = nn.Linear(768, 768)
            >>> lora_linear = LoraLinear.from_module(linear, r=16, lora_alpha=32)
        """
        pass

    @abstractmethod
    def reset_lora_parameters(self) -> None:
        """
        Initialize LoRA parameters following Microsoft's original implementation.

        Standard initialization:
        - lora_A: Kaiming uniform initialization (same as nn.Linear default)
        - lora_B: Zero initialization (ensures LoRA starts as identity/no-op)

        This method should be called during __init__ after creating LoRA parameters.
        """
        pass

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the delta weight from LoRA matrices.

        The delta weight represents the low-rank update to the base weight:
            delta_weight = lora_B @ lora_A * scaling

        Returns:
            Delta weight tensor with same shape as base weight

        Note:
            Not all LoRA modules implement this (e.g. MoE modules with multiple
            projections). Use merge_weights() directly for those cases.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_delta_weight(). "
            "For MoE modules, use merge_weights() directly."
        )

    @abstractmethod
    def merge_weights(self) -> None:
        """
        Merge LoRA weights into base weight for inference.

        After merging: weight = weight + delta_weight

        This eliminates the LoRA computation overhead during inference.
        The operation modifies weights in-place and is not reversible.

        Warning:
            This modifies the base weight in-place. Make sure to save
            LoRA weights separately before merging if you need to unmerge later.
        """
        pass

    def get_lora_parameters(self) -> Iterator[nn.Parameter]:
        """
        Yield all LoRA parameters (lora_A, lora_B, etc.).

        This method finds all parameters with 'lora_' in their name.
        Subclasses can override this if they use different naming conventions.

        Yields:
            LoRA parameters
        """
        # self must be an nn.Module for named_parameters to work
        if not isinstance(self, nn.Module):
            raise TypeError("LoraModule must be mixed with nn.Module")

        for name, param in self.named_parameters():
            if "lora_" in name:
                yield param

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Get state dict containing only LoRA parameters.

        Returns:
            Dictionary mapping parameter names to tensors for LoRA parameters only
        """
        if not isinstance(self, nn.Module):
            raise TypeError("LoraModule must be mixed with nn.Module")

        return {
            name: param.detach()
            for name, param in self.named_parameters()
            if "lora_" in name
        }
