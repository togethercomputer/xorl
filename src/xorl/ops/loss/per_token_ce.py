"""Per-token cross-entropy computation with optional vocab-parallel TP support."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.ops.loss.compiled_cross_entropy import compiled_cross_entropy_function
from xorl.ops.loss.fused_linear_logprob import fused_selected_logprob_ce
from xorl.ops.loss.vocab_parallel_cross_entropy import (
    vocab_parallel_cross_entropy,
    vocab_parallel_cross_entropy_with_lm_head,
)


_MODULE_LM_HEAD_MIN_CHUNK_ROWS = 128


def _module_lm_head_ce(
    hidden_states_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    ignore_index: int,
    num_chunks: int,
) -> torch.Tensor:
    if hidden_states_flat.shape[0] == 0:
        return hidden_states_flat.new_empty((0,), dtype=torch.float32)

    chunk_count = max(1, int(num_chunks))
    chunk_size = max(
        _MODULE_LM_HEAD_MIN_CHUNK_ROWS,
        (hidden_states_flat.shape[0] + chunk_count - 1) // chunk_count,
    )
    ce_chunks: list[torch.Tensor] = []
    for start in range(0, hidden_states_flat.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states_flat.shape[0])
        logits = lm_head(hidden_states_flat[start:end]).float()
        ce_chunks.append(F.cross_entropy(logits, labels_flat[start:end], reduction="none", ignore_index=ignore_index))
    return torch.cat(ce_chunks, dim=0)


def compute_per_token_ce(
    hidden_states_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    ignore_index: int,
    ce_mode: str,
    num_chunks: int = 8,
    tp_group: Optional[dist.ProcessGroup] = None,
    use_compile: bool = False,
    lm_head_fp32: bool = False,
    lm_head: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss based on the specified mode.

    Args:
        hidden_states_flat: Flattened hidden states, shape (BT, H)
        weight: LM head weight matrix, shape (V, H) or (V/tp, H) with TP
        labels_flat: Flattened labels, shape (BT,)
        ignore_index: Index to ignore in loss computation
        ce_mode: Cross-entropy computation mode ("compiled", "eager", or
            "fused_quack"). "fused_quack" uses chunked cuBLAS matmul + a fused
            CuTeDSL cross-entropy reduction, keeping the logits tile chunk-sized
            and reducing only three [N] scalar vectors across TP; it serves both
            the TP and non-TP cases.
        num_chunks: Number of chunks for compiled mode
        tp_group: TP process group for vocab-parallel cross-entropy (default: None)
        use_compile: Whether to use torch.compile in vocab_parallel_cross_entropy
        lm_head: Optional module to call for the logits matmul. Used by FP8
            training so ``FP8Linear.forward`` is not bypassed by raw-weight CE.
        lm_head_fp32: Compute the lm_head logits in FP32. This takes PRECEDENCE
            over ``lm_head`` — when set, the FP8 lm_head module is bypassed and
            logits are computed in FP32 from the (master) ``weight``, so an FP8
            lm_head does not catastrophically mis-score rare near-certain tokens.

    Returns:
        per_token_ce: Per-token cross-entropy loss, shape (BT,)
    """
    # ``lm_head_fp32`` takes precedence over the FP8 lm_head module: an FP32
    # lm_head means the projection must NOT be FP8-quantized, so route to the
    # raw-weight FP32 path below rather than calling ``FP8Linear.forward``. The
    # passed ``weight`` is the master (non-quantized) lm_head weight.
    use_lm_head_module = lm_head is not None and not lm_head_fp32

    # ``fused_quack`` is an explicit opt-in that fuses the selected-token logprob
    # via chunked cuBLAS matmul + a fused CuTeDSL cross-entropy reduction
    # (chunk-sized logits, scalar TP reductions); it serves TP and non-TP cases.
    if ce_mode == "fused_quack":
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        if lm_head_fp32:
            hidden_states_flat = hidden_states_flat.float()
            local_weight = local_weight.float()
        return fused_selected_logprob_ce(
            hidden_states_flat,
            local_weight,
            labels_flat,
            tp_group=tp_group,
            ignore_index=ignore_index,
        )

    if tp_group is not None:
        if use_lm_head_module:
            return vocab_parallel_cross_entropy_with_lm_head(
                hidden_states_flat,
                lm_head,
                labels_flat,
                tp_group,
                ignore_index=ignore_index,
                num_chunks=num_chunks,
                use_compile=use_compile,
            )
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        if lm_head_fp32:
            local_weight = local_weight.float()
            hidden_states_flat = hidden_states_flat.float()
        return vocab_parallel_cross_entropy(
            hidden_states_flat,
            local_weight,
            labels_flat,
            tp_group,
            ignore_index=ignore_index,
            num_chunks=num_chunks,
            use_compile=use_compile,
        )

    if use_lm_head_module:
        return _module_lm_head_ce(
            hidden_states_flat,
            labels_flat,
            lm_head=lm_head,
            ignore_index=ignore_index,
            num_chunks=num_chunks,
        )

    if ce_mode == "compiled":
        return compiled_cross_entropy_function(
            hidden_states_flat,
            weight,
            labels_flat,
            ignore_index,
            num_chunks,
            lm_head_fp32=lm_head_fp32,
        )
    else:
        # eager mode
        if lm_head_fp32:
            logits_flat = (hidden_states_flat.float() @ weight.float().t()).float()
        else:
            logits_flat = (hidden_states_flat @ weight.t()).float()
        return F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)
