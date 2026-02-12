"""Flash Attention implementation (FA2/FA4).

Pure attention computation -- no sequence parallelism logic.
SP communication (Ulysses all-to-all, etc.) is handled externally by
SPStrategy classes in ``xorl.distributed.sequence_parallel.strategy``.
"""

import os
from typing import Optional, Tuple

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func

from .....utils import logging


# FA4 (CUTE) import with fallback
try:
    from flash_attn.cute import flash_attn_func as fa4_flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as fa4_flash_attn_varlen_func

    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False
    fa4_flash_attn_func = None
    fa4_flash_attn_varlen_func = None

# Environment variable to disable FA4 even when available
XORL_DISABLE_FA4 = os.environ.get("XORL_DISABLE_FA4", "0") == "1"

logger = logging.get_logger(__name__)


def _should_use_fa4(use_fa4: bool) -> bool:
    """Check if FA4 should be used based on request and availability."""
    if XORL_DISABLE_FA4:
        return False
    if not FA4_AVAILABLE:
        return False
    return use_fa4


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    use_fa4: bool = False,  # Use FA4 (CUTE) instead of FA2/FA3
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    # This is for Qwen2VL's mrope
    position_ids = kwargs.pop("position_ids", None)
    if position_ids is not None and position_ids.dim() == 3:
        position_ids = position_ids[0]

    # FA4 (CUTE) path
    if _should_use_fa4(use_fa4):
        if not FA4_AVAILABLE:
            raise ImportError(
                "flash_attention_4 requested but flash_attn.cute is not installed. "
                "Install it with: pip install flash-attn-cute"
            )

        # Convert sliding_window (int) to window_size (tuple) for FA4
        if sliding_window is not None:
            window_size = (sliding_window, 0 if module.is_causal else sliding_window)
        else:
            window_size = (None, None)

        # Check if we have varlen kwargs (cu_seqlens from packing collator)
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        cu_seq_lens_k = kwargs.get("cu_seq_lens_k", None)
        max_length_q = kwargs.get("max_length_q", None)
        max_length_k = kwargs.get("max_length_k", None)

        if cu_seq_lens_q is not None and cu_seq_lens_k is not None:
            # flash_attn requires cu_seqlens to be int32
            cu_seq_lens_q = cu_seq_lens_q.to(torch.int32)
            cu_seq_lens_k = cu_seq_lens_k.to(torch.int32)
            # Varlen path: use flash_attn_varlen_func for packed sequences
            # FA4 varlen expects shape (total_tokens, num_heads, head_dim) - squeeze batch dim
            # Our tensors are (1, total_tokens, num_heads, head_dim) with packing
            q_varlen = query.squeeze(0) if query.size(0) == 1 else query.reshape(-1, query.size(-2), query.size(-1))
            k_varlen = key.squeeze(0) if key.size(0) == 1 else key.reshape(-1, key.size(-2), key.size(-1))
            v_varlen = value.squeeze(0) if value.size(0) == 1 else value.reshape(-1, value.size(-2), value.size(-1))

            attn_output, _ = fa4_flash_attn_varlen_func(
                q_varlen,
                k_varlen,
                v_varlen,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                causal=True,  # Always causal for autoregressive LLMs
                window_size=window_size,
                softcap=softcap if softcap is not None else 0.0,
            )
            # Restore batch dimension
            attn_output = attn_output.unsqueeze(0)
        else:
            # Non-varlen path: use flash_attn_func
            # FA4 expects shape (batch, seqlen, num_heads, head_dim) - same as current
            attn_output, _ = fa4_flash_attn_func(
                query,
                key,
                value,
                causal=True,  # Always causal for autoregressive LLMs
                window_size=window_size,
                softcap=softcap if softcap is not None else 0.0,
            )
    else:
        # FA2 path (default) — call flash_attn directly
        causal = getattr(module, "is_causal", True)

        # Convert sliding_window (int) to window_size (tuple) for flash_attn
        if sliding_window is not None:
            window_size_fa2 = (sliding_window, 0 if causal else sliding_window)
        else:
            window_size_fa2 = (-1, -1)

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        cu_seq_lens_k = kwargs.get("cu_seq_lens_k", None)
        max_length_q = kwargs.get("max_length_q", None)
        max_length_k = kwargs.get("max_length_k", None)

        if cu_seq_lens_q is not None and cu_seq_lens_k is not None:
            # flash_attn requires cu_seqlens to be int32
            cu_seq_lens_q = cu_seq_lens_q.to(torch.int32)
            cu_seq_lens_k = cu_seq_lens_k.to(torch.int32)
            # Varlen path for packed sequences
            # flash_attn_varlen_func expects 3D: (total_tokens, num_heads, head_dim)
            q_varlen = query.squeeze(0) if query.size(0) == 1 else query.reshape(-1, query.size(-2), query.size(-1))
            k_varlen = key.squeeze(0) if key.size(0) == 1 else key.reshape(-1, key.size(-2), key.size(-1))
            v_varlen = value.squeeze(0) if value.size(0) == 1 else value.reshape(-1, value.size(-2), value.size(-1))

            attn_output = flash_attn_varlen_func(
                q_varlen,
                k_varlen,
                v_varlen,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                dropout_p=dropout,
                softmax_scale=scaling,
                causal=causal,
                window_size=window_size_fa2,
                softcap=softcap if softcap is not None else 0.0,
            )
            # Restore batch dimension
            attn_output = attn_output.unsqueeze(0)
        else:
            # Regular batched path
            # flash_attn_func expects 4D: (batch, seqlen, num_heads, head_dim)
            attn_output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=dropout,
                softmax_scale=scaling,
                causal=causal,
                window_size=window_size_fa2,
                softcap=softcap if softcap is not None else 0.0,
            )

    return attn_output, None


def prepare_causal_mask(attention_mask, **kwargs) -> Optional[torch.Tensor]:
    """Flash attention handles causal masking internally.

    Returns None unless the mask contains 0.0 values (padding indicator).
    """
    if attention_mask is not None and 0.0 in attention_mask:
        return attention_mask
    return None
