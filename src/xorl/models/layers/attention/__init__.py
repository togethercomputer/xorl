from .multi_head_attention import MultiHeadAttention
from .utils import repeat_kv
from .backend import (
    ATTENTION_FUNCTIONS,
    AttentionKwargs,
    CAUSAL_MASK_FUNCTIONS,
    FLASH_ATTENTION_IMPLEMENTATIONS,
    FlashAttentionKwargs,
    is_flash_attention,
    update_causal_mask,
)
from .backend.eager import eager_attention_forward
from .backend._mask_utils import prepare_4d_causal_attention_mask_with_cache_position

# Conditional import — flash may not be installed
try:
    from .backend.flash_attention import FA4_AVAILABLE, flash_attention_forward
except ImportError:
    FA4_AVAILABLE = False

__all__ = [
    "ATTENTION_FUNCTIONS",
    "AttentionKwargs",
    "CAUSAL_MASK_FUNCTIONS",
    "FA4_AVAILABLE",
    "FLASH_ATTENTION_IMPLEMENTATIONS",
    "FlashAttentionKwargs",
    "MultiHeadAttention",
    "eager_attention_forward",
    "flash_attention_forward",
    "is_flash_attention",
    "prepare_4d_causal_attention_mask_with_cache_position",
    "repeat_kv",
    "update_causal_mask",
]
