from typing import Optional, Tuple

import torch
import torch.distributed as dist


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
      * `fsdp_size` compensates for FSDP's gradient averaging: FSDP divides gradients
        by the number of ranks in its mesh, but this function already normalizes by
        global_valid_tokens, so we multiply by fsdp_size to cancel FSDP's division.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        loss: torch.Tensor,
        local_valid_tokens: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        fsdp_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle edge case where no valid tokens exist on this rank.
        # Use zeros_like (not loss * 0.0, since NaN * 0.0 = NaN in IEEE 754).
        if local_valid_tokens.item() == 0:
            loss = torch.zeros_like(loss)

        # Scale loss by local valid tokens
        loss_sum = loss * local_valid_tokens

        # Sum across all ranks (synchronous all-reduce)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

        # Save tensors for backward
        ctx.save_for_backward(local_valid_tokens, global_valid_tokens)
        ctx.fsdp_size = fsdp_size

        # Return normalized loss and summed loss
        return loss_sum / global_valid_tokens, loss_sum

    @staticmethod
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor,
        grad_loss_sum: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, None, None, None]:
        local_valid_tokens, global_valid_tokens = ctx.saved_tensors
        fsdp_size = ctx.fsdp_size

        # Gradient from first output (normalized loss)
        grad_from_normalized = grad_output * local_valid_tokens / global_valid_tokens

        # Gradient from second output (sum loss)
        grad_from_sum = (
            grad_loss_sum * local_valid_tokens
            if grad_loss_sum is not None
            else torch.zeros_like(grad_output)
        )

        # Combine both gradient contributions and compensate for FSDP averaging.
        # FSDP divides gradients by fsdp_size (reduce-scatter averages), but we
        # already normalize by global_valid_tokens. Multiply by fsdp_size so that
        # after FSDP's division the gradient equals the correct globally-normalized value.
        grad_loss = (grad_from_normalized + grad_from_sum) * fsdp_size

        return grad_loss, None, None, None


def gradient_accumulate_loss(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    fsdp_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    User-facing helper function to apply GradientAccumulateLoss.

    Args:
        fsdp_size: Number of ranks in the FSDP mesh. Compensates for FSDP's
            gradient averaging so that gradients are correctly normalized by
            global_valid_tokens alone.
    """
    return GradientAccumulateLoss.apply(loss, local_valid_tokens, global_valid_tokens, fsdp_size)