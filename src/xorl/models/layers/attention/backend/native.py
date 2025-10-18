"""Native attention backend using PyTorch SDPA with cuDNN.

No external dependencies — works on Hopper and Blackwell GPUs.
Uses torch.nn.functional.scaled_dot_product_attention with the cuDNN
backend forced for best performance on modern NVIDIA GPUs.
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def native_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """SDPA attention forward with cuDNN backend.

    Input shapes: query [B, S, H, D], key/value [B, S, KVH, D].
    Supports GQA natively via enable_gqa=True (no KV head expansion needed).
    """
    causal = getattr(module, "is_causal", True)

    # SDPA expects (B, H, S, D)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Check for varlen (packed sequences)
    cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
    cu_seq_lens_k = kwargs.get("cu_seq_lens_k", None)

    # GQA: use enable_gqa when Q and KV have different head counts
    enable_gqa = query.shape[1] != key.shape[1]

    if cu_seq_lens_q is not None and cu_seq_lens_k is not None:
        # Varlen/packed sequences: build a block-diagonal mask from cu_seqlens
        # so SDPA applies causal masking per-document within the packed batch.
        max_length_q = kwargs.get("max_length_q", query.shape[2])
        mask = _build_varlen_causal_mask(
            cu_seq_lens_q, cu_seq_lens_k, query.shape[2], key.shape[2],
            device=query.device, dtype=query.dtype,
        )
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask,
                dropout_p=dropout if module.training else 0.0,
                scale=scaling,
                enable_gqa=enable_gqa,
            )
    else:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                is_causal=causal,
                dropout_p=dropout if module.training else 0.0,
                scale=scaling,
                enable_gqa=enable_gqa,
            )

    # (B, H, S, D) -> (B, S, H, D)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, None


def _build_varlen_causal_mask(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_q: int,
    total_k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a block-diagonal causal mask from cu_seqlens for packed sequences.

    Returns a (1, 1, total_q, total_k) bool mask suitable for SDPA.
    """
    mask = torch.zeros(total_q, total_k, device=device, dtype=torch.bool)
    num_seqs = len(cu_seqlens_q) - 1
    for i in range(num_seqs):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start
        # Causal mask within this document
        causal_block = torch.ones(q_len, k_len, device=device, dtype=torch.bool).tril()
        mask[q_start:q_end, k_start:k_end] = causal_block
    return mask.unsqueeze(0).unsqueeze(0)


def prepare_causal_mask(attention_mask, **kwargs) -> Optional[torch.Tensor]:
    """Native SDPA handles causal masking internally via is_causal=True."""
    if attention_mask is not None and 0.0 in attention_mask:
        return attention_mask
    return None
