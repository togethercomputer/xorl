from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.ops.loss.compiled_cross_entropy import (
    compiled_ce_and_lse_sq_function,
    compiled_cross_entropy_function,
)
from xorl.ops.loss.loss_output import LossOutput
from xorl.ops.loss.reducers import Reducer, TokenPartial
from xorl.ops.loss.vocab_parallel_cross_entropy import (
    _backward_kernel as _vocab_parallel_ce_backward_kernel,
)
from xorl.ops.loss.vocab_parallel_cross_entropy import (
    _forward_kernel as _vocab_parallel_ce_forward_kernel,
)
from xorl.ops.loss.vocab_parallel_cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy_with_lm_head,
)


_MODULE_LM_HEAD_MIN_CHUNK_ROWS = 128


def _all_gather_cat_same_shape(x: torch.Tensor, *, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


class _FSDPShardedCausalLMLoss(torch.autograd.Function):
    """Sequence-streaming vocab-parallel CE for FSDP-sharded lm_head.

    The lm_head shard group is also the sequence-parallel group for the GLM
    128K configuration, so each rank owns different sequence tokens and a
    different vocab shard. The usual vocab-parallel CE needs all ranks to see
    the same token batch; this Function gathers one small sequence chunk at a
    time and avoids saving those gathered chunks for backward.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        local_weight: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        sequence_group: dist.ProcessGroup,
        vocab_group: dist.ProcessGroup,
        num_chunks: int,
        ignore_index: int,
        loss_reduce_group: "dist.ProcessGroup | None" = None,
        loss_reduce_divisor: float = 1.0,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected hidden_states to have shape [B, S, H], got {tuple(hidden_states.shape)}")
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError(f"Expected labels shape {tuple(hidden_states.shape[:2])}, got {tuple(labels.shape)}")

        ctx.save_for_backward(hidden_states, local_weight, labels, global_valid_tokens)
        ctx.sequence_group = sequence_group
        ctx.vocab_group = vocab_group
        ctx.num_chunks = num_chunks
        ctx.ignore_index = ignore_index

        vocab_rank = dist.get_rank(vocab_group)
        local_vocab_size = local_weight.shape[0]
        vocab_offset = vocab_rank * local_vocab_size
        local_seq_len = hidden_states.shape[1]
        chunk_size = max(1, math.ceil(local_seq_len / num_chunks))
        denom = global_valid_tokens.clamp(min=1.0)
        loss = hidden_states.new_zeros((), dtype=torch.float32)

        for start in range(0, local_seq_len, chunk_size):
            end = min(start + chunk_size, local_seq_len)
            gathered_hidden = _all_gather_cat_same_shape(hidden_states[:, start:end, :], dim=1, group=sequence_group)
            gathered_labels = _all_gather_cat_same_shape(labels[:, start:end], dim=1, group=sequence_group)
            hidden_flat = gathered_hidden.reshape(-1, gathered_hidden.shape[-1])
            labels_flat = gathered_labels.reshape(-1)
            per_token_ce, _, _, _, _, valid_mask = _vocab_parallel_ce_forward_kernel(
                hidden_flat,
                local_weight,
                labels_flat,
                vocab_group,
                vocab_offset,
                local_vocab_size,
                ignore_index,
            )
            loss = loss + (per_token_ce * valid_mask.float()).sum() / denom
            del gathered_hidden, gathered_labels, hidden_flat, labels_flat, per_token_ce, valid_mask

        # lm-head-TP: each replica group computes the CE over its own sequence
        # shard and over the full (TP-split) vocab, so the per-replica losses must
        # be summed across replicas. The divisor removes the within-TP-group
        # duplication (every TP rank computed the same per-replica loss). Backward
        # intentionally stays unscaled: each rank returns its local weight-shard
        # gradient, which the caller combines with an all-reduce over the replica
        # group to reconstruct the full gradient.
        if loss_reduce_group is not None:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=loss_reduce_group)
        if loss_reduce_divisor != 1.0:
            loss = loss / loss_reduce_divisor

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, local_weight, labels, global_valid_tokens = ctx.saved_tensors
        sequence_group = ctx.sequence_group
        vocab_group = ctx.vocab_group
        vocab_rank = dist.get_rank(vocab_group)
        local_vocab_size = local_weight.shape[0]
        vocab_offset = vocab_rank * local_vocab_size
        local_seq_len = hidden_states.shape[1]
        chunk_size = max(1, math.ceil(local_seq_len / ctx.num_chunks))
        denom = global_valid_tokens.clamp(min=1.0)

        grad_hidden = torch.zeros_like(hidden_states) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(local_weight) if ctx.needs_input_grad[1] else None
        sp_world = dist.get_world_size(sequence_group)
        sp_rank = dist.get_rank(sequence_group)

        for start in range(0, local_seq_len, chunk_size):
            end = min(start + chunk_size, local_seq_len)
            gathered_hidden = _all_gather_cat_same_shape(hidden_states[:, start:end, :], dim=1, group=sequence_group)
            gathered_labels = _all_gather_cat_same_shape(labels[:, start:end], dim=1, group=sequence_group)
            hidden_flat = gathered_hidden.reshape(-1, gathered_hidden.shape[-1])
            labels_flat = gathered_labels.reshape(-1)
            _, global_max, global_sumexp, target_in_range, safe_local_target, valid_mask = (
                _vocab_parallel_ce_forward_kernel(
                    hidden_flat,
                    local_weight,
                    labels_flat,
                    vocab_group,
                    vocab_offset,
                    local_vocab_size,
                    ctx.ignore_index,
                )
            )
            per_token_grad = grad_output.to(hidden_flat.dtype) * valid_mask.to(hidden_flat.dtype) / denom
            chunk_grad_hidden, chunk_grad_weight = _vocab_parallel_ce_backward_kernel(
                per_token_grad,
                hidden_flat,
                local_weight,
                global_max,
                global_sumexp,
                target_in_range,
                safe_local_target,
                valid_mask,
                vocab_group,
                ctx.needs_input_grad[1],
            )
            if grad_hidden is not None:
                chunk_grad_hidden = chunk_grad_hidden.view_as(gathered_hidden)
                local_grad = chunk_grad_hidden.chunk(sp_world, dim=1)[sp_rank].contiguous()
                grad_hidden[:, start:end, :] = local_grad.to(grad_hidden.dtype)
            if grad_weight is not None and chunk_grad_weight is not None:
                grad_weight.add_(chunk_grad_weight.to(grad_weight.dtype))
            del (
                gathered_hidden,
                gathered_labels,
                hidden_flat,
                labels_flat,
                global_max,
                global_sumexp,
                target_in_range,
                safe_local_target,
                valid_mask,
                per_token_grad,
                chunk_grad_hidden,
                chunk_grad_weight,
            )

        # Grads for: hidden_states, local_weight, then None for labels,
        # global_valid_tokens, sequence_group, vocab_group, num_chunks,
        # ignore_index, loss_reduce_group, loss_reduce_divisor.
        return grad_hidden, grad_weight, None, None, None, None, None, None, None, None


def fsdp_sharded_causallm_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    sp_group: dist.ProcessGroup,
    fsdp_group: dist.ProcessGroup,
    num_chunks: int,
    ignore_index: int = -100,
    lm_head_fp32: bool = False,
    global_valid_tokens: torch.Tensor | None = None,
    sequence_group: "dist.ProcessGroup | None" = None,
    vocab_group: "dist.ProcessGroup | None" = None,
    loss_reduce_group: "dist.ProcessGroup | None" = None,
    loss_reduce_divisor: float = 1.0,
) -> "LossOutput":
    # sequence_group/vocab_group default to sp_group/fsdp_group (the FSDP-sharded
    # lm_head case where the sequence-parallel group is also the vocab shard
    # group). lm-head-only TP passes a dedicated lm_head_tp_group for both and a
    # loss_reduce_group (+ divisor) to sum the per-replica losses.
    if sequence_group is None:
        sequence_group = sp_group
    if vocab_group is None:
        vocab_group = fsdp_group
    if lm_head_fp32:
        hidden_states = hidden_states.float()

    local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
    if local_weight.dtype != hidden_states.dtype:
        if lm_head_fp32:
            local_weight = local_weight.float()
        else:
            local_weight = local_weight.to(hidden_states.dtype)

    if global_valid_tokens is None:
        global_valid_tokens = (labels != ignore_index).sum().to(hidden_states.device, dtype=torch.float32)
        dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM, group=fsdp_group)
    else:
        global_valid_tokens = global_valid_tokens.detach().to(hidden_states.device, dtype=torch.float32)
    loss = _FSDPShardedCausalLMLoss.apply(
        hidden_states,
        local_weight,
        labels,
        global_valid_tokens,
        sequence_group,
        vocab_group,
        int(num_chunks),
        int(ignore_index),
        loss_reduce_group,
        float(loss_reduce_divisor),
    )
    return LossOutput(loss=loss)


def _chunked_lm_head_cross_entropy(
    hidden_states_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    ignore_index: int,
    num_chunks: int,
    z_loss_enabled: bool,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute CE by calling the lm_head module in chunks.

    FP8 training wraps ``lm_head`` with ``FP8Linear``. The compiled CE helpers
    operate on ``lm_head.weight`` directly, which bypasses that module, so FP8
    training uses this path to keep the output head matmul on FP8 compute.
    """

    if hidden_states_flat.shape[0] == 0:
        empty = hidden_states_flat.new_empty((0,), dtype=torch.float32)
        return empty, empty if z_loss_enabled else None

    chunk_count = max(1, int(num_chunks))
    chunk_size = max(_MODULE_LM_HEAD_MIN_CHUNK_ROWS, math.ceil(hidden_states_flat.shape[0] / chunk_count))
    ce_chunks: list[torch.Tensor] = []
    lse_sq_chunks: list[torch.Tensor] = []
    for start in range(0, hidden_states_flat.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states_flat.shape[0])
        logits = lm_head(hidden_states_flat[start:end]).float()
        labels = labels_flat[start:end]
        ce_chunks.append(F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index))
        if z_loss_enabled:
            lse = torch.logsumexp(logits, dim=-1)
            lse_sq_chunks.append((lse * lse) * valid_mask[start:end].to(lse.dtype))

    per_token_ce = torch.cat(ce_chunks, dim=0)
    per_token_lse_sq = torch.cat(lse_sq_chunks, dim=0) if z_loss_enabled else None
    return per_token_ce, per_token_lse_sq


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _chunk_size_from_num_chunks(num_tokens: int, num_chunks: int) -> int:
    if num_chunks <= 0:
        return _ceil_to_multiple(num_tokens, 8)
    return _ceil_to_multiple((num_tokens + num_chunks - 1) // num_chunks, 8)


def _quack_linear_cross_entropy_loss(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int,
    num_chunks: int,
    loss_reducer: TokenPartial,
) -> torch.Tensor:
    if not hidden_states_flat.is_cuda:
        raise ValueError("ce_mode='quack_linear' requires CUDA tensors")
    if hidden_states_flat.shape[-1] % 8 != 0 or weight.shape[0] % 8 != 0:
        raise ValueError("ce_mode='quack_linear' requires hidden and vocab dimensions to be divisible by 8")

    from xorl.ops.quack.linear_cross_entropy import chunked_linear_cross_entropy  # noqa: PLC0415

    hidden_states_flat, labels_flat = _pad_quack_linear_rows(
        hidden_states_flat,
        labels_flat,
        ignore_index=ignore_index,
    )
    valid_count = (labels_flat != ignore_index).sum()
    if valid_count.item() == 0:
        return (hidden_states_flat.sum() + weight.sum()) * 0.0

    chunk_size = _chunk_size_from_num_chunks(hidden_states_flat.shape[0], num_chunks)
    loss_sum = chunked_linear_cross_entropy(
        hidden_states_flat,
        weight,
        labels_flat,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        reduction="sum",
    )
    scale = loss_reducer.scale.to(device=loss_sum.device, dtype=loss_sum.dtype)
    return loss_sum / scale.clamp(min=1.0)


def _pad_quack_linear_rows(
    hidden_states_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    *,
    ignore_index: int,
    multiple: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad rows for Quack CE kernels without changing scalar CE semantics."""

    remainder = hidden_states_flat.shape[0] % multiple
    if remainder == 0:
        return hidden_states_flat, labels_flat
    pad_rows = multiple - remainder
    hidden_pad = hidden_states_flat.new_zeros((pad_rows, hidden_states_flat.shape[-1]))
    label_pad = labels_flat.new_full((pad_rows,), ignore_index)
    return torch.cat((hidden_states_flat, hidden_pad), dim=0), torch.cat((labels_flat, label_pad), dim=0)


def _fused_quack_per_token_ce(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int,
    num_chunks: int,
    tp_group,
    lm_head_fp32: bool,
) -> torch.Tensor:
    """Per-token CE ``[N]`` via the fused chunked cuBLAS + CuTeDSL path.

    ``fused_selected_logprob_ce`` keeps the logits tile bounded to
    ``[chunk, V_local]`` and never materializes the full ``[N, V]`` logits.
    Mirrors the dispatch in ``ops.loss.per_token_ce.compute_per_token_ce`` so
    ``ce_mode='fused_quack'`` works from this entry point too — without it,
    fused_quack fell through to the eager full-logits path and OOM'd at large
    vocab / long context.
    """
    if not hidden_states_flat.is_cuda:
        raise ValueError("ce_mode='fused_quack' requires CUDA tensors")

    from xorl.ops.loss.fused_linear_logprob import fused_selected_logprob_ce  # noqa: PLC0415

    local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
    hidden = hidden_states_flat
    if lm_head_fp32:
        hidden = hidden.float()
        local_weight = local_weight.float()
    chunk_size = _chunk_size_from_num_chunks(hidden.shape[0], num_chunks)
    return fused_selected_logprob_ce(
        hidden,
        local_weight,
        labels_flat,
        tp_group=tp_group,
        ignore_index=ignore_index,
        chunk_size=chunk_size,
    )


def _quack_linear_per_token_cross_entropy(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int,
    num_chunks: int,
    lm_head_fp32: bool,
) -> torch.Tensor:
    """Per-token return path for ``ce_mode='quack_linear'``.

    The scalar training path keeps using Quack's chunked linear CE reduction.
    ``return_per_token=True`` callers need one CE value per input row, so route
    those through the existing fused selected-logprob kernel and still avoid
    full-logit materialization.
    """

    if not hidden_states_flat.is_cuda:
        raise ValueError("ce_mode='quack_linear' requires CUDA tensors")
    if hidden_states_flat.shape[-1] % 8 != 0 or weight.shape[0] % 8 != 0:
        raise ValueError("ce_mode='quack_linear' requires hidden and vocab dimensions to be divisible by 8")
    return _fused_quack_per_token_ce(
        hidden_states_flat,
        weight,
        labels_flat,
        ignore_index,
        num_chunks,
        tp_group=None,
        lm_head_fp32=lm_head_fp32,
    )


def causallm_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    return_per_token: bool = False,
    ce_mode: str = "compiled",
    num_chunks: int = 8,
    tp_group=None,
    use_compile: bool = False,
    lm_head_fp32: bool = False,
    loss_reducer: Reducer | None = None,
    z_loss_coef: float = 0.0,
    lm_head: torch.nn.Module | None = None,
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
        if lm_head is not None and not lm_head_fp32:
            per_token_ce = vocab_parallel_cross_entropy_with_lm_head(
                hidden_states_flat,
                lm_head,
                labels_flat,
                tp_group,
                ignore_index=ignore_index,
                num_chunks=num_chunks,
                use_compile=use_compile,
            )
        else:
            # lm_head_fp32 takes precedence over the FP8 lm_head module: compute
            # the vocab-parallel CE in fp32 from the master weight (FP8 module
            # bypassed). Extract local weight from DTensor if needed.
            local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
            if lm_head_fp32:
                hidden_states_flat = hidden_states_flat.float()
                local_weight = local_weight.float()
            elif local_weight.dtype != hidden_states_flat.dtype:
                local_weight = local_weight.to(hidden_states_flat.dtype)

            per_token_ce = vocab_parallel_cross_entropy(
                hidden_states_flat,
                local_weight,
                labels_flat,
                tp_group,
                ignore_index=ignore_index,
                num_chunks=num_chunks,
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
    # lm_head_fp32 takes precedence over the FP8 lm_head module: an FP32 lm_head
    # must not be FP8-quantized, so route to the fp32 weight-CE path below
    # (compiled/eager honor lm_head_fp32) instead of _chunked_lm_head_cross_entropy
    # (which calls FP8Linear.forward). The FP8 lm_head otherwise catastrophically
    # mis-scores rare near-certain tokens (R1).
    use_lm_head_module = lm_head is not None and not lm_head_fp32

    if ce_mode == "quack_linear" and not return_per_token:
        if z_loss_enabled:
            raise NotImplementedError("ce_mode='quack_linear' does not support softmax_auxiliary_loss")
        if lm_head_fp32:
            raise NotImplementedError("ce_mode='quack_linear' does not support lm_head_fp32=True")
        if not isinstance(loss_reducer, TokenPartial):
            raise NotImplementedError("ce_mode='quack_linear' currently supports only TokenPartial loss reduction")
        return LossOutput(
            loss=_quack_linear_cross_entropy_loss(
                hidden_states_flat,
                weight,
                labels_flat,
                ignore_index,
                num_chunks,
                loss_reducer,
            )
        )

    if return_per_token:
        # Compute cross-entropy based on mode (and Z-loss when enabled).
        per_token_lse_sq = None
        if use_lm_head_module:
            per_token_ce, per_token_lse_sq = _chunked_lm_head_cross_entropy(
                hidden_states_flat,
                labels_flat,
                lm_head=lm_head,
                ignore_index=ignore_index,
                num_chunks=num_chunks,
                z_loss_enabled=z_loss_enabled,
                valid_mask=valid_mask,
            )
        elif ce_mode == "compiled":
            if z_loss_enabled:
                per_token_ce, per_token_lse_sq = compiled_ce_and_lse_sq_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
            else:
                per_token_ce = compiled_cross_entropy_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
        elif ce_mode == "fused_quack":
            if z_loss_enabled:
                raise NotImplementedError("ce_mode='fused_quack' does not support softmax_auxiliary_loss")
            per_token_ce = _fused_quack_per_token_ce(
                hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, tp_group, lm_head_fp32
            )
        elif ce_mode == "quack_linear":
            if z_loss_enabled:
                raise NotImplementedError("ce_mode='quack_linear' does not support softmax_auxiliary_loss")
            per_token_ce = _quack_linear_per_token_cross_entropy(
                hidden_states_flat,
                weight,
                labels_flat,
                ignore_index,
                num_chunks,
                lm_head_fp32,
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
        if use_lm_head_module:
            per_token_ce, per_token_lse_sq = _chunked_lm_head_cross_entropy(
                hidden_states_flat,
                labels_flat,
                lm_head=lm_head,
                ignore_index=ignore_index,
                num_chunks=num_chunks,
                z_loss_enabled=z_loss_enabled,
                valid_mask=valid_mask,
            )
        elif ce_mode == "compiled":
            if z_loss_enabled:
                per_token_ce, per_token_lse_sq = compiled_ce_and_lse_sq_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
            else:
                per_token_ce = compiled_cross_entropy_function(
                    hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, lm_head_fp32=lm_head_fp32
                )
        elif ce_mode == "fused_quack":
            if z_loss_enabled:
                raise NotImplementedError("ce_mode='fused_quack' does not support softmax_auxiliary_loss")
            per_token_ce = _fused_quack_per_token_ce(
                hidden_states_flat, weight, labels_flat, ignore_index, num_chunks, tp_group, lm_head_fp32
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
