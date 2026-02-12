"""Eager (matmul-based) attention backend."""

from typing import Optional, Tuple

import torch
from torch import nn

from ..utils import repeat_kv
from ._mask_utils import prepare_4d_causal_attention_mask_with_cache_position


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard eager (matmul-based) attention forward.

    Input/output shapes: query/key/value are [batch, seq, heads, head_dim].
    Transposes to [batch, heads, seq, head_dim] internally for matmul.
    """
    # [B, S, H, D] -> [B, H, S, D] for matmul-based attention
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def prepare_causal_mask(
    attention_mask,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """Build 4D causal mask for the eager attention backend."""
    past_seen_tokens = 0
    sequence_length = input_tensor.shape[1]
    target_length = (
        attention_mask.shape[-1]
        if isinstance(attention_mask, torch.Tensor)
        else past_seen_tokens + sequence_length + 1
    )

    return prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=input_tensor.dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
        sliding_window=sliding_window,
    )
