"""
Attention backend registry and common types.

Provides:
- AttentionKwargs: generic typed dict for attention kwargs
- ATTENTION_FUNCTIONS: maps implementation name -> attention forward callable
- CAUSAL_MASK_FUNCTIONS: maps implementation name -> mask preparation callable
- update_causal_mask: thin dispatcher over CAUSAL_MASK_FUNCTIONS
- is_flash_attention / FLASH_ATTENTION_IMPLEMENTATIONS: flash-family detection
"""

from functools import partial
from typing import Callable, Dict, Optional, Set, Union

import torch
from typing_extensions import TypedDict

from .eager import eager_attention_forward, prepare_causal_mask as eager_prepare_causal_mask


# ------------------------------------------------------------------ #
# Generic attention kwargs (renamed from FlashAttentionKwargs)
# ------------------------------------------------------------------ #

class AttentionKwargs(TypedDict, total=False):
    """Kwargs for attention functions (packed/varlen sequences).

    Generic across backends — flash_attention_2/3/4 uses cu_seq_lens, eager ignores them, etc.
    """

    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None


# Backwards compatibility alias
FlashAttentionKwargs = AttentionKwargs


# ------------------------------------------------------------------ #
# Flash-family detection
# ------------------------------------------------------------------ #

FLASH_ATTENTION_IMPLEMENTATIONS: Set[str] = {
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
}


def is_flash_attention(attn_implementation: str) -> bool:
    """Return True if *attn_implementation* handles causal masking internally."""
    return attn_implementation in FLASH_ATTENTION_IMPLEMENTATIONS


# ------------------------------------------------------------------ #
# ATTENTION_FUNCTIONS registry
# ------------------------------------------------------------------ #

ATTENTION_FUNCTIONS: Dict[str, Callable] = {
    "eager": eager_attention_forward,
}

# Register flash attention implementations at import time.
try:
    from .flash_attention import FA4_AVAILABLE, flash_attention_forward

    ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    ATTENTION_FUNCTIONS["flash_attention_3"] = flash_attention_forward

    if FA4_AVAILABLE:
        ATTENTION_FUNCTIONS["flash_attention_4"] = partial(flash_attention_forward, use_fa4=True)
except ImportError:
    FA4_AVAILABLE = False


# ------------------------------------------------------------------ #
# CAUSAL_MASK_FUNCTIONS registry
# ------------------------------------------------------------------ #

CAUSAL_MASK_FUNCTIONS: Dict[str, Callable] = {
    "eager": eager_prepare_causal_mask,
    "sdpa": eager_prepare_causal_mask,
}

# Register flash mask functions
try:
    from .flash_attention import prepare_causal_mask as flash_prepare_causal_mask

    for _key in FLASH_ATTENTION_IMPLEMENTATIONS:
        CAUSAL_MASK_FUNCTIONS[_key] = flash_prepare_causal_mask
except ImportError:
    pass

# Register flex mask function
try:
    from .flex_attention import prepare_causal_mask as flex_prepare_causal_mask

    CAUSAL_MASK_FUNCTIONS["flex_attention"] = flex_prepare_causal_mask
except ImportError:
    pass


# ------------------------------------------------------------------ #
# Thin dispatcher
# ------------------------------------------------------------------ #

def update_causal_mask(
    attn_implementation: str,
    attention_mask: Union[torch.Tensor, None],
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    sliding_window: Optional[int] = None,
    is_training: bool = False,
    output_attentions: bool = False,
) -> Optional[torch.Tensor]:
    """Build the appropriate causal mask for the given attention implementation.

    Dispatches to the registered prepare_causal_mask function for the backend.
    """
    prepare_fn = CAUSAL_MASK_FUNCTIONS.get(attn_implementation)
    if prepare_fn is not None:
        return prepare_fn(
            attention_mask,
            input_tensor=input_tensor,
            cache_position=cache_position,
            sliding_window=sliding_window,
            is_training=is_training,
            output_attentions=output_attentions,
            attn_implementation=attn_implementation,
        )
    # Fallback: return mask unchanged
    return attention_mask
