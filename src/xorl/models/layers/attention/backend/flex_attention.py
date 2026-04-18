"""Flex attention backend (PyTorch 2.x flex_attention)."""

from typing import Optional

import torch


try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


def make_causal_block_mask(attention_mask_2d: torch.Tensor) -> "BlockMask":
    """Create a causal BlockMask from a 2D attention mask.

    Combines causal masking, document boundary masking (for packed sequences),
    and padding masking into a single BlockMask for flex_attention.

    Args:
        attention_mask_2d: (batch_size, seq_len) tensor where values > 0 indicate
            valid tokens. For packed sequences, different document IDs indicate
            document boundaries.
    """
    batch_size, seq_len = attention_mask_2d.shape
    device = attention_mask_2d.device
    document_ids = attention_mask_2d

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_document = document_ids[batch_idx, q_idx] == document_ids[batch_idx, kv_idx]
        not_padding = attention_mask_2d[batch_idx, q_idx] > 0
        return causal & same_document & not_padding

    return create_block_mask(
        mask_mod=mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )


def prepare_causal_mask(attention_mask, **kwargs) -> Optional["BlockMask"]:
    """Convert 2D attention mask to flex attention BlockMask."""
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = make_causal_block_mask(attention_mask)
    return attention_mask
