from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .parallel_state import get_parallel_state


class GradientAccumulateLoss(torch.autograd.Function):
    """
    Custom autograd function for distributed loss accumulation.

    Computes:
      - The globally normalized loss = (sum of local losses * local tokens) / global_valid_tokens
      - The global loss sum itself

    Notes:
      * Assumes `global_valid_tokens` is already reduced (summed) across all ranks.
      * Handles local_valid_tokens == 0 safely.
      * Supports autograd so gradients can flow through distributed computations.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        loss: torch.Tensor,
        local_valid_tokens: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle edge case where no valid tokens exist on this rank
        if local_valid_tokens.item() == 0:
            loss = torch.nan_to_num(loss)

        # Scale loss by local valid tokens
        loss_sum = loss * local_valid_tokens

        # Sum across all ranks (synchronous all-reduce)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

        # Save tensors for backward
        ctx.save_for_backward(local_valid_tokens, global_valid_tokens)

        # Return normalized loss and summed loss
        return loss_sum / global_valid_tokens, loss_sum

    @staticmethod
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
        grad_loss_sum: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, None, None]:
        local_valid_tokens, global_valid_tokens = ctx.saved_tensors

        # Gradient from first output (normalized loss)
        grad_from_normalized = grad_output * local_valid_tokens / global_valid_tokens

        # Gradient from second output (sum loss)
        grad_from_sum = (
            grad_loss_sum * local_valid_tokens
            if grad_loss_sum is not None
            else torch.zeros_like(grad_output)
        )

        # Combine both gradient contributions
        # NOTE: We do NOT scale by world_size because loss_sum was already reduced across all ranks.
        grad_loss = grad_from_normalized + grad_from_sum

        return grad_loss, None, None


def gradient_accumulate_loss(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    User-facing helper function to apply GradientAccumulateLoss.
    """
    return GradientAccumulateLoss.apply(loss, local_valid_tokens, global_valid_tokens)