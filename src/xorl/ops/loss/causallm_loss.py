from __future__ import annotations

import torch
import torch.nn.functional as F

from xorl.ops.loss.compiled_cross_entropy import (
    DEFAULT_NUM_CHUNKS,
    compiled_ce_and_lse_sq_function,
    compiled_cross_entropy_function,
)
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.reducers import Reducer, TokenPartial
from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


def causallm_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    return_per_token: bool = False,
    ce_mode: str = "compiled",
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    tp_group=None,
    use_compile: bool = False,
    lm_head_fp32: bool = False,
    loss_reducer: Reducer | None = None,
    z_loss_coef: float = 0.0,
) -> "LossOutput":
    """
    Compute causal language modeling loss.

    Supports multiple computation modes:
    - "compiled": RECOMMENDED. torch.compile (1.6x speed, 16% memory)
    - "eager": Simple F.cross_entropy baseline (may OOM at 32K)

    Args:
        hidden_states: Model hidden states, shape (batch, seq_len, hidden_dim)
        weight: LM head weight matrix, shape (vocab_size, hidden_dim).
                With TP, this is the local shard [vocab_size/tp, hidden_dim].
        labels: Target labels, shape (batch, seq_len). Labels are assumed to be
                already next-token aligned (labels[i] is the target for hidden_states[i]).
        ignore_index: Index to ignore in loss computation (default: -100)
        return_per_token: If True, return per-token logprobs and losses (default: False)
        ce_mode: Cross-entropy mode - "compiled" (default) or "eager"
        num_chunks: Number of chunks for compiled mode (default: 8).
        tp_group: TP process group for vocab-parallel cross-entropy (default: None).
        loss_reducer: Optional ``(values, mask) -> scalar``. When supplied, the
            returned loss is a partial share under the reducer's denominator
            (sum across micro-batches + all-reduce across ranks recovers the
            globally-correct loss). When None, falls back to a local token mean.
            Z-loss (when enabled) is reduced through the same reducer so the
            two terms compose consistently.
        z_loss_coef: If > 0, add the Z-loss auxiliary term used in OLMo /
                     PaLM-style training:
                         z_loss = coef * sum(logsumexp(logits)^2 * mask) / num_valid_tokens
                     where ``mask = labels != ignore_index``. Equivalent to OLMo's
                     ``cross_entropy_loss(..., reduction="sum")`` path divided by
                     ``batch_size_in_tokens``. Encourages log(Z) to stay near zero,
                     stabilizing training at large vocab / high LR. Not supported
                     in the TP path.

    Returns:
        LossOutput with loss, and optionally per_token_logprobs/per_token_loss.
        When ``z_loss_coef > 0``, ``LossOutput.metrics`` contains
        ``{"ce_loss": <unweighted CE>, "z_loss": <unweighted Z-loss>}``.
    """
    # Store original shape before flattening for per-token outputs
    original_shape = labels.shape

    # Flatten the labels and hidden_states
    labels_flat = labels.view(-1)
    hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
    valid_mask = labels_flat != ignore_index

    if loss_reducer is None:
        loss_reducer = TokenPartial(scale=valid_mask.sum().float())

    mask_flat = valid_mask.float()

    # Vocab-parallel cross-entropy for tensor parallelism
    if tp_group is not None:
        if z_loss_coef > 0.0:
            raise NotImplementedError(
                "softmax_auxiliary_loss (Z-loss) is not yet supported with tensor parallelism. "
                "Disable softmax_auxiliary_loss or run without TP."
            )
        # Extract local weight from DTensor if needed
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight

        per_token_ce = vocab_parallel_cross_entropy(
            hidden_states_flat,
            local_weight,
            labels_flat,
            tp_group,
            ignore_index=ignore_index,
            use_compile=use_compile,
        )

        loss = loss_reducer(per_token_ce, mask_flat)
        if return_per_token:
            return LossOutput(
                loss=loss,
                per_token_logprobs=-per_token_ce.detach().view(original_shape),
                per_token_loss=per_token_ce.view(original_shape),
            )
        return LossOutput(loss=loss)

    z_loss_enabled = z_loss_coef > 0.0

    if return_per_token:
        # Compute cross-entropy based on mode (and Z-loss when enabled).
        per_token_lse_sq = None
        if ce_mode == "compiled":
            if z_loss_enabled:
                per_token_ce, per_token_lse_sq = compiled_ce_and_lse_sq_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
            else:
                per_token_ce = compiled_cross_entropy_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
        else:  # eager mode
            if lm_head_fp32:
                logits_flat = (hidden_states_flat.float() @ weight.float().t()).float()
            else:
                logits_flat = (hidden_states_flat @ weight.t()).float()
            per_token_ce = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)
            if z_loss_enabled:
                lse = torch.logsumexp(logits_flat, dim=-1)
                per_token_lse_sq = (lse * lse) * valid_mask.to(lse.dtype)

        ce_loss = loss_reducer(per_token_ce, mask_flat)
        if z_loss_enabled:
            z_loss = loss_reducer(per_token_lse_sq, mask_flat)
            loss = ce_loss + z_loss_coef * z_loss
            metrics = {"ce_loss": ce_loss.detach(), "z_loss": z_loss.detach()}
        else:
            loss = ce_loss
            metrics = None
        return LossOutput(
            loss=loss,
            per_token_logprobs=-per_token_ce.detach().view(original_shape),
            per_token_loss=per_token_ce.view(original_shape),
            metrics=metrics,
        )
    else:
        # Always use reduction="none" + manual mean to avoid NaN when all labels
        # are ignore_index (reduction="mean" returns NaN for 0 valid elements).
        # Keeping the autograd graph intact is critical for FSDP2: all ranks must
        # trigger reduce-scatter for every parameter, including lm_head weight.
        per_token_lse_sq = None
        if ce_mode == "compiled":
            if z_loss_enabled:
                per_token_ce, per_token_lse_sq = compiled_ce_and_lse_sq_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
            else:
                per_token_ce = compiled_cross_entropy_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
        else:  # eager mode
            if lm_head_fp32:
                logits_flat = (hidden_states_flat.float() @ weight.float().t()).float()
            else:
                logits_flat = (hidden_states_flat @ weight.t()).float()
            per_token_ce = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)
            if z_loss_enabled:
                lse = torch.logsumexp(logits_flat, dim=-1)
                per_token_lse_sq = (lse * lse) * valid_mask.to(lse.dtype)

        ce_loss = loss_reducer(per_token_ce, mask_flat)
        if z_loss_enabled:
            z_loss = loss_reducer(per_token_lse_sq, mask_flat)
            loss = ce_loss + z_loss_coef * z_loss
            return LossOutput(loss=loss, metrics={"ce_loss": ce_loss.detach(), "z_loss": z_loss.detach()})
        return LossOutput(loss=ce_loss)
