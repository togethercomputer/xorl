"""Per-token cross-entropy computation with optional vocab-parallel TP support."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.ops.loss.compiled_cross_entropy import compiled_cross_entropy_function
from xorl.ops.loss.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy


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
) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss based on the specified mode.

    Args:
        hidden_states_flat: Flattened hidden states, shape (BT, H)
        weight: LM head weight matrix, shape (V, H) or (V/tp, H) with TP
        labels_flat: Flattened labels, shape (BT,)
        ignore_index: Index to ignore in loss computation
        ce_mode: Cross-entropy computation mode ("compiled" or "eager")
        num_chunks: Number of chunks for compiled mode
        tp_group: TP process group for vocab-parallel cross-entropy (default: None)
        use_compile: Whether to use torch.compile in vocab_parallel_cross_entropy

    Returns:
        per_token_ce: Per-token cross-entropy loss, shape (BT,)
    """
    if tp_group is not None:
        local_weight = weight.to_local() if hasattr(weight, "to_local") else weight
        return vocab_parallel_cross_entropy(
            hidden_states_flat,
            local_weight,
            labels_flat,
            tp_group,
            ignore_index=ignore_index,
            use_compile=use_compile,
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
