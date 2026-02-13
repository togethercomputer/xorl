import torch
import torch.nn.functional as F

from .compiled_cross_entropy import compiled_cross_entropy_function


def causallm_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    return_per_token: bool = False,
    ce_mode: str = "compiled",
    num_chunks: int = 8,
) -> torch.Tensor:
    """
    Compute causal language modeling loss.

    Supports multiple computation modes:
    - "compiled": RECOMMENDED. torch.compile (1.6x speed, 16% memory)
    - "eager": Simple F.cross_entropy baseline (may OOM at 32K)

    Args:
        hidden_states: Model hidden states, shape (batch, seq_len, hidden_dim)
        weight: LM head weight matrix, shape (vocab_size, hidden_dim)
        labels: Target labels, shape (batch, seq_len). Labels are assumed to be
                already next-token aligned (labels[i] is the target for hidden_states[i]).
        ignore_index: Index to ignore in loss computation (default: -100)
        return_per_token: If True, return per-token logprobs and losses (default: False)
        ce_mode: Cross-entropy mode - "compiled" (default) or "eager"
        num_chunks: Number of chunks for compiled mode (default: 64).
                   Higher values use less memory. At MBS=8 with 128K seq and vocab=151K,
                   num_chunks=8 creates 9.1 GiB FP32 logits per chunk (18 GiB peak during
                   backward), while num_chunks=64 reduces this to 1.1 GiB (2.3 GiB peak).

    Returns:
        If return_per_token=False: (loss, None, None, None)
        If return_per_token=True: (loss, None, per_token_logprobs, per_token_loss)
    """
    # Store original shape before flattening for per-token outputs
    original_shape = labels.shape

    # Flatten the labels and hidden_states
    labels_flat = labels.view(-1)
    hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
    valid_mask = (labels_flat != ignore_index)

    if return_per_token:
        # Compute cross-entropy based on mode
        if ce_mode == "compiled":
            per_token_ce = compiled_cross_entropy_function(hidden_states_flat, weight, labels_flat, ignore_index, num_chunks)
        else:  # eager mode
            logits_flat = (hidden_states_flat @ weight.t()).float()
            per_token_ce = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)

        # logprobs = -nll loss, also detach to prevent gradient flow
        # clone to avoid in-place operation
        per_token_logprobs = -per_token_ce.detach().clone()

        # reshape back to original shape
        per_token_logprobs = per_token_logprobs.view(original_shape)
        per_token_loss = per_token_ce.view(original_shape)

        loss = per_token_ce.sum() / valid_mask.sum().clamp(min=1)
        return loss, None, per_token_logprobs, per_token_loss
    else:
        if ce_mode == "compiled":
            loss = compiled_cross_entropy_function(hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, reduction="mean")
        else:  # eager mode
            loss = F.cross_entropy(hidden_states_flat @ weight.t(), labels_flat, reduction="mean", ignore_index=ignore_index)

        return loss, None, None, None
