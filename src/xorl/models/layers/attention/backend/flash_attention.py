"""Flash Attention implementation (FA2/FA4).

Pure attention computation -- no sequence parallelism logic.
SP communication (Ulysses all-to-all, etc.) is handled externally by
CPStrategy classes in ``xorl.distributed.sequence_parallel.strategy``.
"""

import os
from typing import Optional, Tuple

import torch

from .....utils import logging


# FA3 import. Some CUDA 13 / FA4 environments do not ship flash_attn_interface,
# but they can still use the CUTE FA4 path below.
try:
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache

    FA3_AVAILABLE = True
except ImportError:
    FA3_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

# FA4 (CUTE) import with fallback
try:
    from flash_attn.cute import flash_attn_func as fa4_flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as fa4_flash_attn_varlen_func

    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False
    fa4_flash_attn_func = None
    fa4_flash_attn_varlen_func = None

# sgl_kernel FA3 import -- the exact flash-attention build SGLang serves with.
# It is a different compilation of FA3 from xorl's `flash_attn_interface`, so
# routing packed prefill through it removes the build from the variables in a
# train/serve mismatch hunt. Opt-in, not a parity requirement: the divergence
# once attributed to the build difference was root-caused to RoPE, and given
# matching post-RoPE q/k the attention core reproduces the serving engine's
# output. See docs/k3/ATTENTION_CONTRACT.md.
try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache as sgl_flash_attn_with_kvcache

    SGL_KERNEL_FA_AVAILABLE = True
except ImportError:
    SGL_KERNEL_FA_AVAILABLE = False
    sgl_flash_attn_with_kvcache = None

# Environment variable to disable FA4 even when available
XORL_DISABLE_FA4 = os.environ.get("XORL_DISABLE_FA4", "0") == "1"
XORL_FLASH_ATTN_DETERMINISTIC = os.environ.get("XORL_FLASH_ATTN_DETERMINISTIC", "0") == "1"
# Route packed prefill self-attention through FA3's paged KV-cache entry point
# on a page-size-1 cache with ``num_splits=1`` -- the entry point and fixed
# KV-axis reduction order the serving engine uses.
XORL_FLASH_ATTN_PAGED_KVCACHE = os.environ.get("XORL_FLASH_ATTN_PAGED_KVCACHE", "0") == "1"
# Same route, but through SGLang's exact ``sgl_kernel`` FA3 build.
XORL_FLASH_ATTN_SGL_KERNEL = os.environ.get("XORL_FLASH_ATTN_SGL_KERNEL", "0") == "1"
XORL_FLASH_ATTN_DIAGNOSTIC_DECODE_KVCACHE = os.environ.get("XORL_FLASH_ATTN_DIAGNOSTIC_DECODE_KVCACHE", "0") == "1"

logger = logging.get_logger(__name__)


def _should_use_fa4(use_fa4: bool) -> bool:
    """Check if FA4 should be used based on request and availability."""
    if XORL_DISABLE_FA4:
        return False
    if not FA4_AVAILABLE:
        return False
    return use_fa4 or not FA3_AVAILABLE


def _flash_attention_causal_flag(
    *,
    module_causal: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    diagnostic_decode_cache: bool,
) -> bool:
    causal = module_causal
    if diagnostic_decode_cache and causal and query.shape[1] == 1 and key.shape[1] > query.shape[1]:
        # A one-token decode query is already at the end of ``key``. Disabling
        # the local top-left causal mask lets it attend to the whole prefix.
        causal = False
    return causal


def _check_self_attention_kvcache_inputs(
    kernel: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
) -> None:
    """Reject shapes the page-size-1 KV-cache route cannot represent.

    A silent fallback would break the bitwise contract these routes exist to
    hold, so an unsupported shape raises instead.
    """
    if not torch.equal(cu_seq_lens_q, cu_seq_lens_k):
        raise ValueError(f"{kernel} parity path only supports self-attention cu_seqlens_q == cu_seqlens_k")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError(f"{kernel} parity path expects q/k/v to cover the same packed token span")
    if max_length_q != max_length_k:
        raise ValueError(f"{kernel} parity path expects max_length_q == max_length_k")


def _build_page_size_1_metadata(
    device: torch.device,
    cu_seq_lens_k: torch.Tensor,
    max_length_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the (page_table, cache_seqlens) pair for a page-size-1 KV cache.

    Every token occupies its own page, which is the layout the serving engine
    uses. Table rows past a sequence's length are clamped to page 0; they are
    never read because ``cache_seqlens`` bounds the reduction.
    """
    lengths = (cu_seq_lens_k[1:] - cu_seq_lens_k[:-1]).to(torch.int32)
    starts = cu_seq_lens_k[:-1].unsqueeze(1).to(torch.int32)
    offsets = torch.arange(max_length_k, device=device, dtype=torch.int32).unsqueeze(0)
    page_table = starts + offsets
    page_table = torch.where(offsets < lengths.unsqueeze(1), page_table, torch.zeros_like(page_table))
    return page_table, lengths


def _flash_attn_varlen_with_paged_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
    scaling: Optional[float],
    causal: bool,
    window_size: tuple[int, int],
    softcap: Optional[float],
) -> torch.Tensor:
    """Run packed self-attention through FA3's paged KV-cache entry point.

    The serving engine's deterministic FA3 prefill stores K/V into a
    page-size-1 cache and calls ``flash_attn_with_kvcache`` with
    page_table/cache_seqlens metadata and ``num_splits=1``. xorl's packed
    trainer path otherwise calls ``flash_attn_varlen_func`` directly. Both
    reduce the KV axis in one piece -- ``num_splits`` defaults to 1 on the
    varlen entry point and only the ``*_with_kvcache`` family defaults to 0,
    where a shape-dependent heuristic picks the split count -- so this route
    aligns the entry point, and pins the split count that entry point would
    otherwise choose.
    """
    if flash_attn_with_kvcache is None:
        raise ImportError("XORL_FLASH_ATTN_PAGED_KVCACHE requires flash_attn_interface / FA3")
    _check_self_attention_kvcache_inputs(
        "paged-kvcache FA3", q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k
    )

    cu_seq_lens_q = cu_seq_lens_q.to(device=q.device, dtype=torch.int32)
    cu_seq_lens_k = cu_seq_lens_k.to(device=q.device, dtype=torch.int32)
    page_table, lengths = _build_page_size_1_metadata(q.device, cu_seq_lens_k, max_length_k)

    # page-size-1 KV cache: each token occupies its own page.
    k_cache = k.contiguous().view(-1, 1, k.shape[-2], k.shape[-1])
    v_cache = v.contiguous().view(-1, 1, v.shape[-2], v.shape[-1])
    return flash_attn_with_kvcache(
        q=q.contiguous(),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k_new=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        softmax_scale=scaling,
        causal=causal,
        window_size=window_size,
        softcap=softcap if softcap is not None else 0.0,
        num_splits=1,
    )


def _flash_attn_varlen_with_sgl_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    max_length_q: int,
    max_length_k: int,
    scaling: Optional[float],
    causal: bool,
    window_size: tuple[int, int],
    softcap: Optional[float],
) -> torch.Tensor:
    """Run packed self-attention through SGLang's exact sgl_kernel FA3 kernel.

    SGLang serves with ``sgl_kernel.flash_attn.flash_attn_with_kvcache`` on a
    page-size-1 KV cache with ``num_splits=1``. xorl's default packed trainer
    path calls ``flash_attn_interface.flash_attn_varlen_func`` -- the same
    algorithm, a different entry point, and a different compilation of it. This
    route calls the identical serving kernel, so neither the entry point nor the
    build is a variable when a train/serve mismatch is being localized.
    """
    if sgl_flash_attn_with_kvcache is None:
        raise ImportError("XORL_FLASH_ATTN_SGL_KERNEL requires sgl_kernel.flash_attn / FA3")
    _check_self_attention_kvcache_inputs(
        "sgl_kernel FA3", q, k, v, cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k
    )

    cu_seq_lens_q = cu_seq_lens_q.to(device=q.device, dtype=torch.int32)
    cu_seq_lens_k = cu_seq_lens_k.to(device=q.device, dtype=torch.int32)
    page_table, lengths = _build_page_size_1_metadata(q.device, cu_seq_lens_k, max_length_k)

    # page-size-1 KV cache: each token occupies its own page (matches serving).
    k_cache = k.contiguous().view(-1, 1, k.shape[-2], k.shape[-1])
    v_cache = v.contiguous().view(-1, 1, v.shape[-2], v.shape[-1])
    return sgl_flash_attn_with_kvcache(
        q=q.contiguous(),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k_new=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        softmax_scale=scaling,
        causal=causal,
        window_size=window_size,
        softcap=softcap if softcap is not None else 0.0,
        num_splits=1,
    )


def _flash_attn_decode_with_paged_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: Optional[float],
    causal: bool,
    window_size: tuple[int, int],
    softcap: Optional[float],
) -> torch.Tensor:
    """Run a one-token diagnostic decode segment through FA3's KV-cache entry point."""
    if flash_attn_with_kvcache is None:
        raise ImportError("XORL_FLASH_ATTN_DIAGNOSTIC_DECODE_KVCACHE requires flash_attn_interface / FA3")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("diagnostic decode KV-cache FA3 path expects [batch, seqlen, heads, head_dim] tensors")
    if q.shape[1] != 1:
        raise ValueError("diagnostic decode KV-cache FA3 path only supports one-token query segments")
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0] or k.shape[1] != v.shape[1]:
        raise ValueError("diagnostic decode KV-cache FA3 path got inconsistent q/k/v batch or sequence shapes")

    batch_size = q.shape[0]
    query_len = q.shape[1]
    key_len = k.shape[1]
    device = q.device

    q_varlen = q.contiguous().reshape(batch_size * query_len, q.shape[-2], q.shape[-1])
    k_cache = k.contiguous().reshape(batch_size * key_len, 1, k.shape[-2], k.shape[-1])
    v_cache = v.contiguous().reshape(batch_size * key_len, 1, v.shape[-2], v.shape[-1])

    page_offsets = torch.arange(key_len, device=device, dtype=torch.int32).unsqueeze(0)
    batch_offsets = (torch.arange(batch_size, device=device, dtype=torch.int32) * key_len).unsqueeze(1)
    page_table = batch_offsets + page_offsets
    cache_seqlens = torch.full((batch_size,), key_len, device=device, dtype=torch.int32)

    cu_seqlens_q = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * query_len
    cu_seqlens_k = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * key_len

    attn_output = flash_attn_with_kvcache(
        q=q_varlen,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=query_len,
        softmax_scale=scaling,
        causal=causal,
        window_size=window_size,
        softcap=softcap if softcap is not None else 0.0,
        num_splits=1,
    )
    return attn_output.reshape(batch_size, query_len, attn_output.shape[-2], attn_output.shape[-1])


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
            "Flash attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # Flash attention always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)
    diagnostic_decode_cache = bool(kwargs.pop("diagnostic_decode_cache", False))

    # This is for Qwen2VL's mrope
    position_ids = kwargs.pop("position_ids", None)
    if position_ids is not None and position_ids.dim() == 3:
        position_ids = position_ids[0]
    deterministic = bool(
        kwargs.pop(
            "deterministic",
            XORL_FLASH_ATTN_DETERMINISTIC
            or getattr(getattr(module, "config", None), "_flash_attention_deterministic", False),
        )
    )

    causal = _flash_attention_causal_flag(
        module_causal=getattr(module, "is_causal", True),
        query=query,
        key=key,
        diagnostic_decode_cache=diagnostic_decode_cache,
    )

    # sgl_kernel FA3 attention-core parity path. It takes priority over the
    # FA3/FA4 selection so xorl's packed prefill uses the serving kernel
    # (page_size=1, num_splits=1) regardless of which flash-attn build xorl
    # would otherwise pick. Only the varlen (packed) and single-sequence
    # batched shapes are covered; anything else falls through to FA3/FA4.
    _sgl_cu_q = kwargs.get("cu_seq_lens_q", None)
    _sgl_cu_k = kwargs.get("cu_seq_lens_k", None)
    # A causal forward over a whole [1, S] span is equivalent to one varlen
    # sequence of length S (padding rows are future-masked and discarded), so
    # for B=1 we synthesize cu_seqlens=[0, S] and route through the same kernel.
    _sgl_batched_ok = _sgl_cu_q is None and _sgl_cu_k is None and query.dim() == 4 and query.size(0) == 1
    if XORL_FLASH_ATTN_SGL_KERNEL and ((_sgl_cu_q is not None and _sgl_cu_k is not None) or _sgl_batched_ok):
        if sliding_window is not None:
            window_size_sgl = (sliding_window, 0 if causal else sliding_window)
        else:
            window_size_sgl = (-1, -1)
        q_varlen = query.squeeze(0) if query.size(0) == 1 else query.reshape(-1, query.size(-2), query.size(-1))
        k_varlen = key.squeeze(0) if key.size(0) == 1 else key.reshape(-1, key.size(-2), key.size(-1))
        v_varlen = value.squeeze(0) if value.size(0) == 1 else value.reshape(-1, value.size(-2), value.size(-1))
        if _sgl_cu_q is not None and _sgl_cu_k is not None:
            cu_seq_lens_q = _sgl_cu_q.to(torch.int32)
            cu_seq_lens_k = _sgl_cu_k.to(torch.int32)
            max_length_q = kwargs.get("max_length_q", None)
            max_length_k = kwargs.get("max_length_k", None)
        else:
            seqlen = int(q_varlen.shape[0])
            cu_seq_lens_q = torch.tensor([0, seqlen], device=q_varlen.device, dtype=torch.int32)
            cu_seq_lens_k = cu_seq_lens_q
            max_length_q = seqlen
            max_length_k = seqlen
        attn_output = _flash_attn_varlen_with_sgl_kernel(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
            scaling,
            causal,
            window_size_sgl,
            softcap,
        )
        attn_output = attn_output.unsqueeze(0)
    # FA4 (CUTE) path
    elif _should_use_fa4(use_fa4):
        if not FA4_AVAILABLE:
            raise ImportError(
                "flash_attention_4 requested but flash_attn.cute is not installed. "
                "Install it with: pip install flash-attn-cute"
            )

        # Convert sliding_window (int) to window_size (tuple) for FA4
        if sliding_window is not None:
            window_size = (sliding_window, 0 if causal else sliding_window)
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
                softmax_scale=scaling,
                causal=causal,
                window_size=window_size,
                softcap=softcap if softcap is not None else 0.0,
                num_splits=1,
                deterministic=deterministic,
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
                softmax_scale=scaling,
                causal=causal,
                window_size=window_size,
                softcap=softcap if softcap is not None else 0.0,
                num_splits=1,
                deterministic=deterministic,
            )
    else:
        # FA3 path (default) — call flash_attn_interface directly
        if not FA3_AVAILABLE:
            raise ImportError(
                "flash_attention_3 requested but flash_attn_interface is not installed. "
                "Install FA3 dependencies or use flash_attention_4 in an FA4 environment."
            )

        # Convert sliding_window (int) to window_size (tuple) for flash_attn
        if sliding_window is not None:
            window_size_fa3 = (sliding_window, 0 if causal else sliding_window)
        else:
            window_size_fa3 = (-1, -1)

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

            if XORL_FLASH_ATTN_PAGED_KVCACHE:
                attn_output = _flash_attn_varlen_with_paged_kvcache(
                    q_varlen,
                    k_varlen,
                    v_varlen,
                    cu_seq_lens_q,
                    cu_seq_lens_k,
                    max_length_q,
                    max_length_k,
                    scaling,
                    causal,
                    window_size_fa3,
                    softcap,
                )
            else:
                attn_output = flash_attn_varlen_func(
                    q_varlen,
                    k_varlen,
                    v_varlen,
                    cu_seqlens_q=cu_seq_lens_q,
                    cu_seqlens_k=cu_seq_lens_k,
                    max_seqlen_q=max_length_q,
                    max_seqlen_k=max_length_k,
                    softmax_scale=scaling,
                    causal=causal,
                    window_size=window_size_fa3,
                    softcap=softcap if softcap is not None else 0.0,
                    deterministic=deterministic,
                )
            # Restore batch dimension
            attn_output = attn_output.unsqueeze(0)
        elif (
            diagnostic_decode_cache
            and XORL_FLASH_ATTN_DIAGNOSTIC_DECODE_KVCACHE
            and query.shape[1] == 1
            and key.shape[1] > query.shape[1]
        ):
            attn_output = _flash_attn_decode_with_paged_kvcache(
                query,
                key,
                value,
                scaling,
                causal,
                window_size_fa3,
                softcap,
            )
        else:
            # Regular batched path
            # flash_attn_func expects 4D: (batch, seqlen, num_heads, head_dim)
            attn_output = flash_attn_func(
                query,
                key,
                value,
                softmax_scale=scaling,
                causal=causal,
                window_size=window_size_fa3,
                softcap=softcap if softcap is not None else 0.0,
                deterministic=deterministic,
            )

    return attn_output, None


def prepare_causal_mask(attention_mask, **kwargs) -> Optional[torch.Tensor]:
    """Flash attention handles causal masking internally.

    Returns None unless the mask contains 0.0 values (padding indicator).
    """
    if attention_mask is not None and 0.0 in attention_mask:
        return attention_mask
    return None
