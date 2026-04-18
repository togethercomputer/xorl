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

    # Derive KV repeat factor from runtime tensor shapes instead of always
    # trusting the global config. Under Ulysses sync strategy, heads can be
    # redistributed before reaching eager attention, so using the global
    # num_key_value_groups may over-repeat KV and cause head mismatch.
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    if q_heads % kv_heads != 0:
        raise RuntimeError(
            f"Invalid attention head layout: query_heads={q_heads} is not divisible by kv_heads={kv_heads}."
        )
    kv_repeat = q_heads // kv_heads
    key_states = repeat_kv(key, kv_repeat)
    value_states = repeat_kv(value, kv_repeat)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Align mask to runtime attention tensor under sequence-parallel sharding.
        # attn_weights: [B, H, Q, K]
        q_len = attn_weights.shape[-2]
        k_len = attn_weights.shape[-1]
        causal_mask = attention_mask

        # Match key axis
        mk = causal_mask.shape[-1]
        if mk != k_len:
            if mk > k_len:
                causal_mask = causal_mask[..., :k_len]
            elif k_len % mk == 0:
                causal_mask = causal_mask.repeat_interleave(k_len // mk, dim=-1)
            else:
                causal_mask = torch.nn.functional.pad(
                    causal_mask, (0, k_len - mk), value=torch.finfo(causal_mask.dtype).min
                )

        # Match query axis
        mq = causal_mask.shape[-2]
        if mq != q_len:
            if mq > q_len:
                causal_mask = causal_mask[:, :, -q_len:, :]
            elif q_len % mq == 0:
                causal_mask = causal_mask.repeat_interleave(q_len // mq, dim=-2)
            else:
                causal_mask = causal_mask[:, :, :q_len, :]

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
        attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
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
