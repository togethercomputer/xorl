"""Shared causal mask construction utilities."""

from typing import Optional

import torch


def prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    cache_position: torch.Tensor,
    batch_size: int,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Creates a causal 4D mask of shape ``(batch_size, 1, query_length, key_value_length)``
    from a 2D mask of shape ``(batch_size, key_value_length)``, or returns the input
    unchanged if it is already 4D.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        device = cache_position.device
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)

        if sliding_window is not None:
            if sequence_length > target_length:
                sliding_attend_mask = torch.arange(target_length, device=device) <= (
                    cache_position.reshape(-1, 1) - sliding_window
                )
                diagonal_attend_mask.bitwise_or_(sliding_attend_mask)

        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask
