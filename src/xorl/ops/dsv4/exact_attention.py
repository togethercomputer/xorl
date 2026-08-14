"""Literal serving-value attention primitives for the DSV4-Flash exact lane."""

from __future__ import annotations

import os

import torch
from torch import Tensor


def _validate_dsv4_lora_metadata(tensor: Tensor, *, where: str) -> None:
    if os.environ.get("XORL_DSV4_DIAGNOSTIC_BASE_MARLIN") != "1":
        return
    from xorl.models.transformers.deepseek_v4.native_payload import (  # noqa: PLC0415
        _validate_all_single_adapter_batch_infos,
    )

    _validate_all_single_adapter_batch_infos(tensor.device.index, where=where)


def _positions(batch_size: int, sequence_length: int, device: torch.device, offset: int = 0) -> Tensor:
    return torch.arange(offset, offset + sequence_length, dtype=torch.int64, device=device).repeat(batch_size)


# Serving's SWA KV pool pages tokens in blocks of the sliding window (the
# DeepseekV4AttnBackend asserts ``swa_page_size == SWA_WINDOW == 128``). The
# decode kernel's tile schedule keys on this page-block size, so the carried
# raw cache must use it for byte parity.
_SERVING_SWA_PAGE_SIZE = 128
# One cache slot is 584 bytes: 448 FP8 nope + 128 BF16 rope + 8 scale bytes.
_SERVING_SLOT_BYTES = 584


class Dsv4DecodeCarryState:
    """Carried per-layer serving cache bytes for incremental decode replay.

    One instance lives on a ``DeepSeekV4Attention`` module while the decode-cache
    scorer replays a request as serving-shaped segments (one prefill seed plus
    M=1 decode steps). It owns the same byte images serving carries across decode
    steps: the paged FP8 raw/SWA key cache, the paged FP8 compressed key cache,
    and the compressor's FP32 kv-score ring (legacy request-scoped addressing,
    rid=0: C4 uses two four-slot pages, C128 one 128-slot page).
    """

    __slots__ = (
        "kvcache",
        "num_tokens",
        "compressed_kvcache",
        "num_compressed",
        "compressor_state",
    )

    def __init__(self) -> None:
        self.kvcache: Tensor | None = None
        self.num_tokens: int = 0
        self.compressed_kvcache: Tensor | None = None
        self.num_compressed: int = 0
        self.compressor_state: Tensor | None = None


def _flashmla_page_bytes(page_size: int) -> int:
    return ((584 * page_size + 575) // 576) * 576


def _ensure_paged_kvcache(cache: Tensor | None, num_slots: int, page_size: int, device: torch.device) -> Tensor:
    """Grow (never shrink) a paged FlashMLA FP8 cache to hold ``num_slots``.

    Pages are zero-filled like serving's pool allocation (``create_buffer``
    uses ``torch.zeros``): the decode kernel's masked lanes still read cache
    bytes behind invalid indices, and non-zero garbage there perturbs the
    online-softmax max by a few ULPs.
    """

    num_pages = max((num_slots + page_size - 1) // page_size, 1)
    page_bytes = _flashmla_page_bytes(page_size)
    if cache is None:
        return torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device=device)
    if cache.shape[0] >= num_pages:
        return cache
    grown = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device=cache.device)
    grown[: cache.shape[0]].copy_(cache)
    return grown


def _window_indices_for_positions(positions: Tensor) -> Tensor:
    """Serving window-128 SWA indices for one query row per absolute position."""

    starts = torch.clamp(positions - 127, min=0)
    indices = starts.unsqueeze(1) + torch.arange(128, device=positions.device).unsqueeze(0)
    return indices.masked_fill(indices > positions.unsqueeze(1), -1).to(torch.int32)


def _paged_cache_kernel_view(cache: Tensor, page_size: int) -> Tensor:
    """View a paged FP8 cache the way serving hands it to the decode kernel."""

    return cache[:, : page_size * _SERVING_SLOT_BYTES].view(cache.shape[0], page_size, 1, _SERVING_SLOT_BYTES)


def _serving_decode_attention(
    q: Tensor,
    carry_state: Dsv4DecodeCarryState,
    position: int,
    ratio: int,
    attn_sink: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Run serving's decode attention kernel over the carried FP8 caches.

    Mirrors ``DeepseekV4AttnBackend.forward`` at decode exactly: the FlashMLA
    sparse decode kernel reads the paged FP8 SWA cache through window-128
    indices and (for compressed layers) the paged FP8 compressed cache through
    ascending complete-block indices with serving's clamp-to-one topk length
    (all-invalid indices when no block is complete). The combined
    ``flash_mla_sparse_fwd`` replay is not byte-stable against this kernel at
    every decode shape, so the
    decode segments must call the literal serving entry point.
    """

    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: PLC0415

    device = q.device
    positions = torch.tensor([position], dtype=torch.int64, device=device)
    # Serving's decode window indices are DESCENDING from the current position
    # (build_causal_swa_page_indices: pos - arange(window), invalid -> -1). The
    # accumulation order is part of the byte contract: ascending indices differ
    # by a few ULPs on some heads.
    window_offsets = position - torch.arange(128, dtype=torch.int32, device=device)
    swa_indices = window_offsets.masked_fill(window_offsets < 0, -1).view(1, 1, 128)
    swa_topk_length = torch.clamp(positions + 1, max=128).to(torch.int32)
    extra_kwargs = {}
    if position < 0 or position >= carry_state.num_tokens:
        raise RuntimeError(
            "DSV4 exact attention query position must address a materialized raw cache row, "
            f"got position={position} with {carry_state.num_tokens} rows"
        )
    if ratio:
        # A fully materialized request cache contains blocks completed by
        # future rows too.  Serving exposes only blocks complete at this
        # query: C4 block zero first appears at p=3 and C128 block zero at
        # p=127.  Using the final cache count here would leak future KV.
        blocks = (position + 1) // ratio
        if blocks > carry_state.num_compressed:
            raise RuntimeError(
                "DSV4 exact attention query requires a compressed row that was not materialized: "
                f"position={position}, C{ratio}, required={blocks}, available={carry_state.num_compressed}"
            )
        # Serving index widths: C4 uses the indexer's top-k row (512); C128 uses
        # the padded per-request block table (64 at admitted context lengths).
        width = 512 if ratio == 4 else 64
        if blocks > width:
            raise RuntimeError(
                f"DSV4 exact decode replay admits at most {width} complete C{ratio} blocks, got {blocks}"
            )
        extra_indices = torch.full((1, 1, width), -1, dtype=torch.int32, device=device)
        if blocks:
            extra_indices[0, 0, :blocks] = torch.arange(blocks, dtype=torch.int32, device=device)
        extra_kwargs = {
            "extra_k_cache": _paged_cache_kernel_view(carry_state.compressed_kvcache, 256 // ratio),
            "extra_indices_in_kvcache": extra_indices,
            "extra_topk_length": torch.tensor([max(blocks, 1)], dtype=torch.int32, device=device),
        }
    output, _ = flash_mla_with_kvcache(
        q=q.contiguous().view(1, 1, 64, 512),
        k_cache=_paged_cache_kernel_view(carry_state.kvcache, _SERVING_SWA_PAGE_SIZE),
        head_dim_v=512,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=get_mla_metadata()[0],
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=swa_indices,
        attn_sink=attn_sink,
        topk_length=swa_topk_length,
        **extra_kwargs,
    )
    return output.view(1, 1, 64, 512)


def _serving_attention_rows(
    q: Tensor,
    carry_state: Dsv4DecodeCarryState,
    positions: Tensor,
    ratio: int,
    attn_sink: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Run the literal serving M=1 kernel once for every causal query row."""

    if q.shape[0] != 1 or q.shape[1] != positions.numel():
        raise RuntimeError(
            "DSV4 exact serving-row replay requires one request and one position per Q row, "
            f"got Q={tuple(q.shape)} and positions={tuple(positions.shape)}"
        )
    if positions.ndim != 1:
        raise RuntimeError(f"DSV4 exact serving-row positions must be rank one, got {tuple(positions.shape)}")
    if not positions.numel():
        return q.new_empty(q.shape)

    # One device-to-host transfer avoids a synchronization for every row.
    # The calls remain literal M=1 invocations; batching rows changes the
    # FlashMLA reduction program and therefore its bytes.
    position_values = positions.detach().to(device="cpu", dtype=torch.int64).tolist()
    output = torch.empty_like(q)
    for row, position in enumerate(position_values):
        output[:, row : row + 1].copy_(
            _serving_decode_attention(
                q[:, row : row + 1],
                carry_state,
                position,
                ratio,
                attn_sink,
                softmax_scale,
            )
        )
    return output


def _store_raw_kv_carry(
    kv_input: Tensor,
    kv_norm_weight: Tensor,
    freqs_cis: Tensor,
    eps: float,
    carry_state: Dsv4DecodeCarryState,
    position_offset: int,
    *,
    dequantize: bool = True,
) -> Tensor | None:
    """Append a segment's normed+roped K entries to the carried FP8 cache.

    Mirrors serving's fused store (``set_swa_key_buffer_radix_fused_norm_rope``)
    with the trainer's flat token addressing (cache slot == absolute position),
    then dequantizes every carried row exactly as the serving sparse-prefill
    reader does.
    """

    from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla  # noqa: PLC0415

    batch_size, sequence_length = kv_input.shape[:2]
    if batch_size != 1:
        raise RuntimeError("DSV4 exact decode-cache carry admits one request")
    if position_offset != carry_state.num_tokens:
        raise RuntimeError(
            "DSV4 exact decode-cache carry got a non-contiguous segment: "
            f"position_offset={position_offset}, carried tokens={carry_state.num_tokens}"
        )
    total_tokens = position_offset + sequence_length
    page_size = _SERVING_SWA_PAGE_SIZE
    carry_state.kvcache = _ensure_paged_kvcache(carry_state.kvcache, total_tokens, page_size, kv_input.device)
    positions = _positions(1, sequence_length, kv_input.device, offset=position_offset)
    fused_k_norm_rope_flashmla(
        kv=kv_input.contiguous().view(-1, kv_input.shape[-1]),
        kv_weight=kv_norm_weight,
        eps=eps,
        freqs_cis=freqs_cis,
        positions=positions,
        out_loc=positions.to(torch.int32),
        kvcache=carry_state.kvcache,
        page_size=page_size,
    )
    carry_state.num_tokens = total_tokens
    if not dequantize:
        return None
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    all_locs = torch.arange(total_tokens, dtype=torch.int32, device=kv_input.device)
    return dequantize_k_cache_paged(carry_state.kvcache, all_locs, page_size).view(total_tokens, 512)


def _store_preprocessed_kv_carry(
    kv: Tensor,
    carry_state: Dsv4DecodeCarryState,
    position_offset: int,
    *,
    dequantize: bool = True,
) -> Tensor | None:
    """Append already-normalized BF16 KV rows to a serving-format cache."""

    from sglang.kernels.ops.attention.dsv4 import fused_store_cache  # noqa: PLC0415

    batch_size, sequence_length = kv.shape[:2]
    if batch_size != 1:
        raise RuntimeError("DSV4 exact preprocessed cache store admits one request")
    if position_offset != carry_state.num_tokens:
        raise RuntimeError(
            "DSV4 exact preprocessed cache store got a non-contiguous segment: "
            f"position_offset={position_offset}, carried tokens={carry_state.num_tokens}"
        )
    total_tokens = position_offset + sequence_length
    page_size = _SERVING_SWA_PAGE_SIZE
    carry_state.kvcache = _ensure_paged_kvcache(carry_state.kvcache, total_tokens, page_size, kv.device)
    positions = _positions(1, sequence_length, kv.device, offset=position_offset)
    fused_store_cache(
        input=kv.contiguous().view(-1, 512),
        cache=carry_state.kvcache,
        indices=positions.to(torch.int32),
        page_size=page_size,
        type="flashmla",
    )
    carry_state.num_tokens = total_tokens
    if not dequantize:
        return None
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    all_locs = torch.arange(total_tokens, dtype=torch.int32, device=kv.device)
    return dequantize_k_cache_paged(carry_state.kvcache, all_locs, page_size).view(total_tokens, 512)


def _apply_rope_torch(x: Tensor, freqs_cis: Tensor, positions: Tensor, *, inverse: bool) -> Tensor:
    rope_dim = freqs_cis.shape[-1] * 2
    prefix = x[..., :-rope_dim]
    rope = x[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    freqs = freqs_cis[positions].view(*x.shape[:-2], -1)
    if inverse:
        freqs = freqs.conj()
    rope_complex = torch.view_as_complex(rope)
    rotated = torch.view_as_real(rope_complex * freqs.unsqueeze(-2)).flatten(-2)
    return torch.cat((prefix.float(), rotated), dim=-1).to(x.dtype)


class _ExactQNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_input: Tensor, freqs_cis: Tensor, positions: Tensor, eps: float) -> Tensor:
        if not q_input.is_cuda or q_input.dtype != torch.bfloat16:
            raise RuntimeError("DSV4 exact Q norm/RoPE requires CUDA BF16 input")
        if q_input.shape[-2:] != (64, 512):
            raise RuntimeError(
                "DSV4 exact Q norm/RoPE admits only the official (heads=64, head_dim=512) geometry, "
                f"got {tuple(q_input.shape[-2:])}"
            )
        from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope  # noqa: PLC0415

        q_flat = q_input.contiguous().view(-1, 64, 512)
        output = torch.empty_like(q_flat)
        fused_q_norm_rope(q_flat, output, eps, freqs_cis, positions)
        ctx.save_for_backward(q_input, freqs_cis, positions)
        ctx.eps = eps
        return output.view_as(q_input)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        q_input, freqs_cis, positions = ctx.saved_tensors
        with torch.enable_grad():
            q = q_input.detach().requires_grad_(True)
            q_float = q.float()
            normalized = q_float * torch.rsqrt(q_float.square().mean(-1, keepdim=True) + ctx.eps)
            surrogate = _apply_rope_torch(normalized.to(q.dtype), freqs_cis, positions, inverse=False)
            grad_q = torch.autograd.grad(surrogate, q, grad_output, create_graph=False)[0]
        return grad_q, None, None, None


def exact_q_norm_rope(q_input: Tensor, freqs_cis: Tensor, eps: float, position_offset: int = 0) -> Tensor:
    """Run SGLang's literal fused Q RMSNorm+RoPE kernel with a trainer VJP."""

    batch_size, sequence_length = q_input.shape[:2]
    positions = _positions(batch_size, sequence_length, q_input.device, offset=position_offset)
    return _ExactQNormRope.apply(q_input, freqs_cis, positions, eps)


class _ExactKVNormRope(torch.autograd.Function):
    """Serving's CP-prefill BF16 KV boundary with a trainer-owned VJP."""

    @staticmethod
    def forward(ctx, kv_input: Tensor, kv_norm_weight: Tensor, freqs_cis: Tensor, eps: float) -> Tensor:
        if not kv_input.is_cuda or kv_input.dtype != torch.bfloat16 or kv_input.shape[-1] != 512:
            raise RuntimeError("DSV4 exact CP KV norm/RoPE requires CUDA BF16 official KV geometry")
        from sglang.kernels.ops.attention.dsv4 import fused_norm_rope_inplace  # noqa: PLC0415

        batch_size, sequence_length = kv_input.shape[:2]
        positions = _positions(batch_size, sequence_length, kv_input.device)
        output = kv_input.contiguous().clone()
        fused_norm_rope_inplace(
            output.view(-1, 512),
            kv_norm_weight,
            eps,
            freqs_cis,
            positions,
        )
        ctx.save_for_backward(kv_input, kv_norm_weight, freqs_cis, positions)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        kv_input, kv_norm_weight, freqs_cis, positions = ctx.saved_tensors
        with torch.enable_grad():
            kv = kv_input.detach().requires_grad_(True)
            surrogate = _kv_norm_rope_torch(kv, kv_norm_weight, freqs_cis, positions, ctx.eps)
            grad_kv = torch.autograd.grad(surrogate, kv, grad_output, create_graph=False)[0]
        return grad_kv, None, None, None


def exact_kv_norm_rope(kv_input: Tensor, kv_norm_weight: Tensor, freqs_cis: Tensor, eps: float) -> Tensor:
    """Materialize the BF16 KV rows SGLang gathers during exact CP prefill."""

    return _ExactKVNormRope.apply(kv_input, kv_norm_weight, freqs_cis, eps)


def _causal_window_indices(batch_size: int, sequence_length: int, device: torch.device) -> Tensor:
    positions = torch.arange(sequence_length, device=device)
    starts = torch.clamp(positions - 127, min=0)
    indices = starts.unsqueeze(1) + torch.arange(128, device=device).unsqueeze(0)
    indices = indices.masked_fill(indices > positions.unsqueeze(1), -1).to(torch.int32)
    return indices.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def _kv_norm_rope_torch(kv: Tensor, weight: Tensor, freqs_cis: Tensor, positions: Tensor, eps: float) -> Tensor:
    kv_float = kv.float()
    normalized = kv_float * torch.rsqrt(kv_float.square().mean(-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    return _apply_rope_torch(normalized.unsqueeze(-2).to(kv.dtype), freqs_cis, positions, inverse=False).squeeze(-2)


def _native_swa_kv(kv_input: Tensor, kv_norm_weight: Tensor, freqs_cis: Tensor, eps: float) -> Tensor:
    """Materialize the serving SWA cache bytes and dequantize them."""

    from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla  # noqa: PLC0415
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    batch_size, sequence_length = kv_input.shape[:2]
    positions = _positions(batch_size, sequence_length, kv_input.device)
    kv_flat = kv_input.contiguous().view(-1, 512)
    num_tokens = kv_flat.shape[0]
    page_size = 256
    page_bytes = ((584 * page_size + 575) // 576) * 576
    num_pages = (num_tokens + page_size - 1) // page_size
    kvcache = torch.empty((num_pages, page_bytes), dtype=torch.uint8, device=kv_input.device)
    out_loc = torch.arange(num_tokens, dtype=torch.int32, device=kv_input.device)
    fused_k_norm_rope_flashmla(
        kv=kv_flat,
        kv_weight=kv_norm_weight,
        eps=eps,
        freqs_cis=freqs_cis,
        positions=positions,
        out_loc=out_loc,
        kvcache=kvcache,
        page_size=page_size,
    )
    _validate_dsv4_lora_metadata(kv_input, where="compressed attention SWA cache store")
    output = dequantize_k_cache_paged(kvcache, out_loc, page_size).view(batch_size, sequence_length, 512)
    _validate_dsv4_lora_metadata(kv_input, where="compressed attention SWA cache dequantize")
    return output


def _native_swa_kv_from_bf16(kv: Tensor) -> Tensor:
    """Quantize/dequantize SGLang's already-normalized CP-prefill KV rows."""

    from sglang.kernels.ops.attention.dsv4 import fused_store_cache  # noqa: PLC0415
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    batch_size, sequence_length = kv.shape[:2]
    kv_flat = kv.contiguous().view(-1, 512)
    num_tokens = kv_flat.shape[0]
    page_size = _SERVING_SWA_PAGE_SIZE
    num_pages = max((num_tokens + page_size - 1) // page_size, 1)
    kvcache = torch.zeros(
        (num_pages, _flashmla_page_bytes(page_size)),
        dtype=torch.uint8,
        device=kv.device,
    )
    out_loc = torch.arange(num_tokens, dtype=torch.int32, device=kv.device)
    fused_store_cache(
        input=kv_flat,
        cache=kvcache,
        indices=out_loc,
        page_size=page_size,
        type="flashmla",
    )
    _validate_dsv4_lora_metadata(kv, where="CP-prefill BF16 SWA cache store")
    output = dequantize_k_cache_paged(kvcache, out_loc, page_size).view(batch_size, sequence_length, 512)
    _validate_dsv4_lora_metadata(kv, where="CP-prefill BF16 SWA cache dequantize")
    return output


def _legacy_compressor_state_pages(ratio: int) -> int:
    """Return the legacy compressor ring pages owned by one request."""

    if ratio == 4:
        # Overlap-C4 alternates between two four-token pages.  The legacy plan
        # addresses them as ``rid * 2 + ((position / 4) & 1)``.
        return 2
    if ratio == 128:
        return 1
    raise ValueError(f"Unsupported DSV4 compression ratio {ratio}")


def _serving_ape_layout(ape: Tensor, ratio: int) -> Tensor:
    serving_ape = ape
    if ratio == 4:
        serving_ape = torch.cat(torch.chunk(ape, 2, dim=-1), dim=0)
    return serving_ape.reshape(-1, 512).float().contiguous()


def _serving_compressor_kv_score(x_flat: Tensor, wkv_weight: Tensor, wgate_weight: Tensor) -> Tensor:
    """The pinned sampler's ``linear_bf16_fp32`` under its deterministic contract.

    The sampler computes kv_score via linear_bf16_fp32 ->
    torch.mm(x, w.t(), out_dtype=fp32), but under its deterministic contract
    torch.mm is patched to the batch-invariant persistent Triton GEMM:
    matmul_persistent(x, w.t().contiguous()).to(fp32) — a BF16-output kernel
    widened to FP32. Call that kernel directly (same treatment as the exact
    MoE router in modeling_deepseek_v4.py). A cuBLAS BF16 GEMM differs by one
    BF16 ulp on rounding-boundary scores, which perturbs the softmax pooling
    of every compressed KV row.
    """

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (  # noqa: PLC0415
        matmul_persistent,
    )

    fused_weight = torch.cat((wkv_weight, wgate_weight), dim=0).to(torch.bfloat16).contiguous()
    return matmul_persistent(x_flat, fused_weight.t().contiguous()).to(torch.float32)


def _serving_compressed_decode_step(
    x: Tensor,
    wkv_weight: Tensor,
    wgate_weight: Tensor,
    ape: Tensor,
    norm_weight: Tensor,
    freqs_cis: Tensor,
    eps: float,
    ratio: int,
    carry_state: Dsv4DecodeCarryState,
) -> Tensor:
    """Run one serving decode step of the compact compressor over carried state.

    Mirrors the serving decode state machine exactly: the new token's kv_score
    row is written into the carried ring at its legacy slot on every step, the
    pooled row is computed by the same ``run_decode`` kernel, and the fused
    norm+rope+store kernel commits a compressed cache row only when this token
    completes a block (``seq_len % ratio == 0``; RoPE position = block start).
    Returns the dequantized compressed rows for every complete block.
    """

    from sglang.kernels.ops.attention.dsv4 import (  # noqa: PLC0415
        CompressorDecodePlan,
        compress_forward,
        compress_norm_rope_store,
    )
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    if x.shape[0] != 1 or x.shape[1] != 1 or x.shape[-1] != 4096 or ratio not in (4, 128):
        raise RuntimeError("DSV4 exact compressor decode step admits one official hidden-size token")
    if carry_state.compressor_state is None:
        raise RuntimeError("DSV4 exact compressor decode step requires a seeded carry state")
    seq_len = carry_state.num_tokens
    if seq_len < 2:
        raise RuntimeError("DSV4 exact compressor decode step requires an appended raw token")

    kv_score = _serving_compressor_kv_score(x.contiguous().view(1, 4096), wkv_weight, wgate_weight)
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor decode kv_score GEMM")
    req_pool_indices = torch.zeros(1, dtype=torch.int64, device=x.device)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio=ratio,
        req_pool_indices=req_pool_indices,
        seq_lens=torch.full((1,), seq_len, dtype=torch.int64, device=x.device),
    )
    compressed_row = compress_forward(
        kv_score_buffer=carry_state.compressor_state,
        kv_score_input=kv_score,
        ape=_serving_ape_layout(ape, ratio),
        plan=plan,
        head_dim=512,
        compress_ratio=ratio,
    )
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor decode compress_forward")

    page_size = 256 // ratio
    carry_state.compressed_kvcache = _ensure_paged_kvcache(
        carry_state.compressed_kvcache, max(seq_len // ratio, 1), page_size, x.device
    )
    at_block_boundary = seq_len % ratio == 0
    out_loc = torch.tensor(
        [seq_len // ratio - 1 if at_block_boundary else 0],
        dtype=torch.int64,
        device=x.device,
    )
    # The decode-mode store kernel itself skips non-boundary tokens
    # (plan.seq_len % ratio != 0), matching serving's dummy-slot behavior.
    compress_norm_rope_store(
        compressed_row,
        plan,
        norm_weight=norm_weight.float().contiguous(),
        norm_eps=eps,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=carry_state.compressed_kvcache,
        page_size=page_size,
    )
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor decode cache store")
    if at_block_boundary:
        carry_state.num_compressed = seq_len // ratio
    if carry_state.num_compressed == 0:
        return x.new_empty((0, 512))
    compressed_locs = torch.arange(carry_state.num_compressed, dtype=torch.int32, device=x.device)
    output = dequantize_k_cache_paged(carry_state.compressed_kvcache, compressed_locs, page_size).view(
        carry_state.num_compressed, 512
    )
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor decode cache dequantize")
    return output


def _serving_compressed_kv(
    x: Tensor,
    wkv_weight: Tensor,
    wgate_weight: Tensor,
    ape: Tensor,
    norm_weight: Tensor,
    freqs_cis: Tensor,
    eps: float,
    ratio: int,
    carry_state: Dsv4DecodeCarryState | None = None,
    *,
    dequantize: bool = True,
) -> Tensor | None:
    """Run the pinned compact compressor and its native cache store."""

    from sglang.kernels.ops.attention.dsv4 import (  # noqa: PLC0415
        CompressorPrefillPlan,
        compress_forward,
        compress_norm_rope_store,
    )

    if x.shape[0] != 1 or x.shape[-1] != 4096 or ratio not in (4, 128):
        raise RuntimeError("DSV4 exact compressor admits one official hidden-size request and ratio 4 or 128")
    sequence_length = x.shape[1]
    num_compressed = sequence_length // ratio
    if num_compressed == 0 and carry_state is None:
        return x.new_empty((0, 512))

    x_flat = x.contiguous().view(sequence_length, 4096)
    # See _serving_compressor_kv_score for why the batch-invariant persistent
    # Triton GEMM (not cuBLAS) is the pinned kv_score kernel.
    kv_score = _serving_compressor_kv_score(x_flat, wkv_weight, wgate_weight)
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor kv_score GEMM")
    coff = 2 if ratio == 4 else 1
    if tuple(kv_score.shape) != (sequence_length, 2 * coff * 512):
        raise RuntimeError(f"Unexpected DSV4 compressor projection shape {tuple(kv_score.shape)}")

    req_pool_indices = torch.zeros(1, dtype=torch.int64, device=x.device)
    plan = CompressorPrefillPlan.generate_legacy(
        compress_ratio=ratio,
        req_pool_indices=req_pool_indices,
        seq_lens=torch.tensor([sequence_length], dtype=torch.int64),
        extend_lens=torch.tensor([sequence_length], dtype=torch.int64),
        num_q_tokens=sequence_length,
        device=x.device,
    )
    state = torch.empty(
        (_legacy_compressor_state_pages(ratio), ratio, 2 * coff * 512),
        dtype=torch.float32,
        device=x.device,
    )
    if carry_state is not None:
        # The serving prefill seeds the request's kv-score ring for later decode
        # steps (its write plan covers every position that a future block
        # boundary can pool). Zero-fill so unwritten slots are deterministic;
        # serving never reads them.
        state.zero_()
        carry_state.compressor_state = state
    compressed = compress_forward(
        kv_score_buffer=state,
        kv_score_input=kv_score,
        ape=_serving_ape_layout(ape, ratio),
        plan=plan,
        head_dim=512,
        compress_ratio=ratio,
    )
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor compress_forward")
    if compressed.shape[0] != num_compressed:
        raise RuntimeError(f"DSV4 compressor emitted {compressed.shape[0]} rows for {sequence_length=} {ratio=}")

    page_size = 256 // ratio
    if carry_state is not None:
        carry_state.compressed_kvcache = _ensure_paged_kvcache(
            carry_state.compressed_kvcache, num_compressed, page_size, x.device
        )
        carry_state.num_compressed = num_compressed
        kvcache = carry_state.compressed_kvcache
    else:
        page_bytes = _flashmla_page_bytes(page_size)
        num_pages = (num_compressed + page_size - 1) // page_size
        kvcache = torch.empty((num_pages, page_bytes), dtype=torch.uint8, device=x.device)
    out_loc = torch.zeros(sequence_length, dtype=torch.int64, device=x.device)
    if num_compressed:
        endpoints = torch.arange(ratio - 1, sequence_length, ratio, device=x.device)
        out_loc[endpoints] = torch.arange(num_compressed, dtype=torch.int64, device=x.device)
    compress_norm_rope_store(
        compressed,
        plan,
        norm_weight=norm_weight.float().contiguous(),
        norm_eps=eps,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=kvcache,
        page_size=page_size,
    )
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor cache store")
    if not dequantize:
        return None
    if num_compressed == 0:
        return x.new_empty((0, 512))
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    compressed_locs = torch.arange(num_compressed, dtype=torch.int32, device=x.device)
    output = dequantize_k_cache_paged(kvcache, compressed_locs, page_size).view(num_compressed, 512)
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor cache dequantize")
    return output


def _hybrid_prefill_indices(sequence_length: int, ratio: int, device: torch.device) -> tuple[Tensor, Tensor, int]:
    """Build the serving short-context compressed-prefix plus SWA ordering."""

    compressed_capacity = max(sequence_length // ratio, 1)
    width = 640 if ratio == 4 else 256
    indices = torch.full((sequence_length, width), -1, dtype=torch.int32, device=device)
    lengths = torch.empty(sequence_length, dtype=torch.int32, device=device)
    for position in range(sequence_length):
        compressed_length = (position + 1) // ratio
        swa_length = min(position + 1, 128)
        if compressed_length:
            indices[position, :compressed_length] = torch.arange(compressed_length, dtype=torch.int32, device=device)
        swa_start = position - swa_length + 1
        indices[position, compressed_length : compressed_length + swa_length] = compressed_capacity + torch.arange(
            swa_start, position + 1, dtype=torch.int32, device=device
        )
        lengths[position] = compressed_length + swa_length
    return indices, lengths, compressed_capacity


def _hybrid_indices_for_positions(positions: Tensor, ratio: int, compressed_capacity: int) -> tuple[Tensor, Tensor]:
    """Serving hybrid ordering (complete compressed blocks, then the SWA window)
    for one query row per absolute position.

    The serving decode indexer emits sequential ascending block indices whenever
    the complete-block count is at or below its top-k (512 for C4); C128 decode
    metadata lists complete blocks ascending unconditionally. Both therefore
    reduce to this compact prefix-then-window ordering at admitted lengths.
    For ``positions == arange(S)`` and ``compressed_capacity ==
    max(S // ratio, 1)`` this reproduces ``_hybrid_prefill_indices`` exactly.
    """

    device = positions.device
    width = 640 if ratio == 4 else 256
    rows = positions.shape[0]
    indices = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    lengths = torch.empty(rows, dtype=torch.int32, device=device)
    for row in range(rows):
        position = int(positions[row].item())
        compressed_length = (position + 1) // ratio
        swa_length = min(position + 1, 128)
        if ratio == 4 and compressed_length > 512:
            raise RuntimeError(
                "DSV4 exact C4 decode replay admits at most 512 complete blocks "
                f"(indexer top-k); got {compressed_length} at position {position}"
            )
        if compressed_length + swa_length > width:
            raise RuntimeError(f"DSV4 exact hybrid index row overflow at position {position} for ratio {ratio}")
        if compressed_length:
            indices[row, :compressed_length] = torch.arange(compressed_length, dtype=torch.int32, device=device)
        swa_start = position - swa_length + 1
        indices[row, compressed_length : compressed_length + swa_length] = compressed_capacity + torch.arange(
            swa_start, position + 1, dtype=torch.int32, device=device
        )
        lengths[row] = compressed_length + swa_length
    return indices, lengths


def _compress_surrogate(
    x: Tensor,
    wkv_weight: Tensor,
    wgate_weight: Tensor,
    ape: Tensor,
    norm_weight: Tensor,
    freqs_cis: Tensor,
    eps: float,
    ratio: int,
) -> Tensor:
    """Differentiable compressor surrogate for the literal serving VJP."""

    batch_size, sequence_length, _ = x.shape
    groups = sequence_length // ratio
    if groups == 0:
        return x.new_empty((batch_size, 0, 512))
    prefix = x[:, : groups * ratio].float()
    kv = torch.nn.functional.linear(prefix, wkv_weight.float()).unflatten(1, (groups, ratio))
    score = torch.nn.functional.linear(prefix, wgate_weight.float()).unflatten(1, (groups, ratio))
    score = score + ape.float()
    if ratio == 4:
        kv_overlap = kv.new_zeros((batch_size, groups, 2 * ratio, 512))
        score_overlap = score.new_full((batch_size, groups, 2 * ratio, 512), float("-inf"))
        kv_overlap[:, :, ratio:] = kv[..., 512:]
        score_overlap[:, :, ratio:] = score[..., 512:]
        if groups > 1:
            kv_overlap[:, 1:, :ratio] = kv[:, :-1, :, :512]
            score_overlap[:, 1:, :ratio] = score[:, :-1, :, :512]
        kv, score = kv_overlap, score_overlap
    weights = score.softmax(dim=2)
    compressed = (kv * weights).sum(dim=2)
    compressed = compressed * torch.rsqrt(compressed.square().mean(-1, keepdim=True) + eps)
    compressed = compressed * norm_weight.float()
    positions = torch.arange(groups, dtype=torch.int64, device=x.device) * ratio
    return _apply_rope_torch(
        compressed.to(torch.bfloat16).unsqueeze(-2),
        freqs_cis,
        positions,
        inverse=False,
    ).squeeze(-2)


class _ExactC0Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        kv_input: Tensor,
        kv_norm_weight: Tensor,
        attn_sink: Tensor,
        freqs_cis: Tensor,
        eps: float,
        softmax_scale: float,
        carry_state: Dsv4DecodeCarryState | None = None,
        position_offset: int = 0,
        query_positions: Tensor | None = None,
        kv_preprocessed: bool = False,
    ) -> Tensor:
        if not q.is_cuda or q.dtype != torch.bfloat16 or kv_input.dtype != torch.bfloat16:
            raise RuntimeError("DSV4 exact C0 attention requires CUDA BF16 Q/KV inputs")
        if q.shape[-2:] != (64, 512) or kv_input.shape[-1] != 512:
            raise RuntimeError("DSV4 exact C0 attention admits only official DSV4-Flash geometry")

        if carry_state is not None:
            if kv_preprocessed:
                raise RuntimeError("DSV4 exact decode-cache carry cannot consume CP-prefill BF16 KV rows")
            # Incremental serving replay over carried per-layer state: append
            # this segment's K entries to the carried FP8 cache and attend from
            # the segment's absolute positions over every carried row.
            if position_offset > 0 and q.shape[1] != 1:
                raise RuntimeError("DSV4 exact C0 decode-cache carry admits M=1 decode segments")
            kv = _store_raw_kv_carry(
                kv_input,
                kv_norm_weight,
                freqs_cis,
                eps,
                carry_state,
                position_offset,
                dequantize=position_offset > 0,
            )
            sequence_length = q.shape[1]
            positions = _positions(1, sequence_length, q.device, offset=position_offset)
            if query_positions is not None and not torch.equal(query_positions, positions):
                raise RuntimeError("DSV4 exact decode-cache carry cannot remap query positions")
            output = _serving_attention_rows(q, carry_state, positions, 0, attn_sink, softmax_scale)
            # Carried prefix rows are serving cache constants; only this
            # segment's Q/KV rows are differentiable.
            prefix = kv[:position_offset].detach() if position_offset else kv_input.new_empty((0, 512))
            ctx.save_for_backward(q, kv_input, kv_norm_weight, attn_sink, freqs_cis, positions, prefix)
            ctx.eps = eps
            ctx.softmax_scale = softmax_scale
            ctx.carry_offset = position_offset
            return output

        if position_offset != 0:
            raise RuntimeError("DSV4 exact C0 attention requires a carry state for offset segments")
        ctx.carry_offset = None
        batch_size, sequence_length = kv_input.shape[:2]
        if q.shape[0] != batch_size:
            raise RuntimeError(
                "DSV4 exact C0 attention requires Q and KV to have the same request batch, "
                f"got Q batch={q.shape[0]} and KV batch={batch_size}"
            )
        if query_positions is None:
            query_positions = torch.arange(q.shape[1], dtype=torch.int64, device=q.device)
        else:
            query_positions = query_positions.to(device=q.device, dtype=torch.int64)
        if query_positions.ndim != 1 or query_positions.numel() != q.shape[1]:
            raise RuntimeError(
                "DSV4 exact C0 attention requires one absolute position per local query row, "
                f"got positions={tuple(query_positions.shape)} and Q rows={q.shape[1]}"
            )
        if query_positions.numel() and (
            int(query_positions.min().item()) < 0 or int(query_positions.max().item()) >= sequence_length
        ):
            raise RuntimeError(
                "DSV4 exact C0 query positions must index the gathered logical KV sequence, "
                f"got range [{int(query_positions.min().item())}, {int(query_positions.max().item())}] "
                f"for KV length {sequence_length}"
            )
        kv_positions = _positions(batch_size, sequence_length, kv_input.device)
        q_flat = q.contiguous().view(batch_size, q.shape[1], 64, 512)
        output = torch.empty_like(q_flat)
        for batch_idx in range(batch_size):
            request_state = Dsv4DecodeCarryState()
            request_kv = kv_input[batch_idx : batch_idx + 1]
            if kv_preprocessed:
                _store_preprocessed_kv_carry(request_kv, request_state, 0, dequantize=False)
            else:
                _store_raw_kv_carry(
                    request_kv,
                    kv_norm_weight,
                    freqs_cis,
                    eps,
                    request_state,
                    0,
                    dequantize=False,
                )
            output[batch_idx : batch_idx + 1].copy_(
                _serving_attention_rows(
                    q_flat[batch_idx : batch_idx + 1],
                    request_state,
                    query_positions,
                    0,
                    attn_sink,
                    softmax_scale,
                )
            )

        ctx.save_for_backward(q, kv_input, kv_norm_weight, attn_sink, freqs_cis, kv_positions, query_positions)
        ctx.eps = eps
        ctx.softmax_scale = softmax_scale
        ctx.kv_preprocessed = kv_preprocessed
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        from xorl.ops.dsv4.attention_core import sparse_attn_torch  # noqa: PLC0415

        if ctx.carry_offset is not None:
            (
                q_saved,
                kv_saved,
                weight,
                attn_sink,
                freqs_cis,
                positions,
                prefix,
            ) = ctx.saved_tensors
            indices = _window_indices_for_positions(positions).unsqueeze(0)
            with torch.enable_grad():
                q = q_saved.detach().requires_grad_(True)
                kv_input = kv_saved.detach().requires_grad_(True)
                kv_segment = _kv_norm_rope_torch(kv_input, weight, freqs_cis, positions, ctx.eps)
                kv = torch.cat((prefix.to(kv_segment.dtype).unsqueeze(0), kv_segment), dim=1)
                surrogate = sparse_attn_torch(q, kv, attn_sink, indices, ctx.softmax_scale)
                grad_q, grad_kv = torch.autograd.grad(
                    surrogate,
                    (q, kv_input),
                    grad_output,
                    create_graph=False,
                )
            return grad_q, grad_kv, None, None, None, None, None, None, None, None, None

        q_saved, kv_saved, weight, attn_sink, freqs_cis, kv_positions, query_positions = ctx.saved_tensors
        batch_size = kv_saved.shape[0]
        indices = _window_indices_for_positions(query_positions).unsqueeze(0).expand(batch_size, -1, -1)
        with torch.enable_grad():
            q = q_saved.detach().requires_grad_(True)
            kv_input = kv_saved.detach().requires_grad_(True)
            kv = (
                kv_input
                if ctx.kv_preprocessed
                else _kv_norm_rope_torch(kv_input, weight, freqs_cis, kv_positions, ctx.eps)
            )
            surrogate = sparse_attn_torch(q, kv, attn_sink, indices, ctx.softmax_scale)
            grad_q, grad_kv = torch.autograd.grad(
                surrogate,
                (q, kv_input),
                grad_output,
                create_graph=False,
            )
        return grad_q, grad_kv, None, None, None, None, None, None, None, None, None


def exact_c0_attention(
    q: Tensor,
    kv_input: Tensor,
    kv_norm_weight: Tensor,
    attn_sink: Tensor,
    freqs_cis: Tensor,
    eps: float,
    softmax_scale: float,
    carry_state: Dsv4DecodeCarryState | None = None,
    position_offset: int = 0,
    query_positions: Tensor | None = None,
    kv_preprocessed: bool = False,
) -> Tensor:
    """Run native SWA-cache quantization and FlashMLA sparse attention for C0.

    Without ``carry_state`` this is the whole-sequence prefill replay. With a
    ``carry_state`` it becomes the incremental serving decode replay: the
    segment's K entries are appended to the carried FP8 cache at their absolute
    positions and the segment queries attend over every carried row.
    """

    return _ExactC0Attention.apply(
        q,
        kv_input,
        kv_norm_weight,
        attn_sink,
        freqs_cis,
        eps,
        softmax_scale,
        carry_state,
        position_offset,
        query_positions,
        kv_preprocessed,
    )


class _ExactHybridAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        kv_input: Tensor,
        x: Tensor,
        kv_norm_weight: Tensor,
        attn_sink: Tensor,
        freqs_cis: Tensor,
        compressor_wkv_weight: Tensor,
        compressor_wgate_weight: Tensor,
        compressor_ape: Tensor,
        compressor_norm_weight: Tensor,
        eps: float,
        softmax_scale: float,
        ratio: int,
        carry_state: Dsv4DecodeCarryState | None = None,
        position_offset: int = 0,
        query_positions: Tensor | None = None,
        kv_preprocessed: bool = False,
    ) -> Tensor:
        if q.shape[0] != 1 or kv_input.shape[0] != 1 or x.shape[0] != 1:
            raise RuntimeError("DSV4 exact compressed attention currently admits one request")
        if kv_input.shape[1] != x.shape[1]:
            raise RuntimeError(
                "DSV4 exact compressed attention requires gathered KV and compressor inputs to cover "
                f"the same logical sequence, got KV={kv_input.shape[1]} and hidden={x.shape[1]} rows"
            )
        if q.dtype != torch.bfloat16 or kv_input.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
            raise RuntimeError("DSV4 exact compressed attention requires BF16 activations")
        if ratio not in (4, 128):
            raise RuntimeError(f"DSV4 exact compressed attention received unsupported ratio {ratio}")
        max_sequence_length = (512 if ratio == 4 else 64) * ratio
        if kv_input.shape[1] > max_sequence_length:
            raise RuntimeError(
                f"DSV4 exact C{ratio} attention admits at most {max_sequence_length} tokens before "
                "serving requires a separately replayed compressed-block selection program"
            )

        query_length = q.shape[1]
        sequence_length = kv_input.shape[1]
        _validate_dsv4_lora_metadata(q, where=f"C{ratio} hybrid attention entry")
        if carry_state is not None:
            if kv_preprocessed:
                raise RuntimeError("DSV4 exact decode-cache carry cannot consume CP-prefill BF16 KV rows")
            if query_length != sequence_length:
                raise RuntimeError("DSV4 exact decode-cache carry cannot shard query rows")
            if position_offset > 0 and sequence_length != 1:
                raise RuntimeError("DSV4 exact compressed decode-cache carry admits M=1 decode segments")
            vanilla = _store_raw_kv_carry(
                kv_input,
                kv_norm_weight,
                freqs_cis,
                eps,
                carry_state,
                position_offset,
                dequantize=position_offset > 0,
            )
            total_tokens = carry_state.num_tokens
            if position_offset == 0:
                compressed = _serving_compressed_kv(
                    x,
                    compressor_wkv_weight,
                    compressor_wgate_weight,
                    compressor_ape,
                    compressor_norm_weight,
                    freqs_cis,
                    eps,
                    ratio,
                    carry_state=carry_state,
                    dequantize=False,
                )
            else:
                compressed = _serving_compressed_decode_step(
                    x,
                    compressor_wkv_weight,
                    compressor_wgate_weight,
                    compressor_ape,
                    compressor_norm_weight,
                    freqs_cis,
                    eps,
                    ratio,
                    carry_state,
                )
            if carry_state.num_compressed != total_tokens // ratio:
                raise RuntimeError(
                    f"DSV4 exact compressed carry lost a block: {carry_state.num_compressed} != {total_tokens // ratio}"
                )
            _validate_dsv4_lora_metadata(q, where=f"C{ratio} hybrid KV assembly")
            compressed_capacity = max(total_tokens // ratio, 1)
            positions = _positions(1, sequence_length, q.device, offset=position_offset)
            if query_positions is not None and not torch.equal(query_positions, positions):
                raise RuntimeError("DSV4 exact decode-cache carry cannot remap query positions")
            query_positions = positions
            request_state = carry_state
        else:
            if position_offset != 0:
                raise RuntimeError("DSV4 exact compressed attention requires a carry state for offset segments")
            if query_positions is None:
                query_positions = torch.arange(query_length, dtype=torch.int64, device=q.device)
            else:
                query_positions = query_positions.to(device=q.device, dtype=torch.int64)
            if query_positions.ndim != 1 or query_positions.numel() != query_length:
                raise RuntimeError(
                    "DSV4 exact compressed attention requires one absolute position per local query row, "
                    f"got positions={tuple(query_positions.shape)} and Q rows={query_length}"
                )
            if query_positions.numel() and (
                int(query_positions.min().item()) < 0 or int(query_positions.max().item()) >= sequence_length
            ):
                raise RuntimeError(
                    "DSV4 exact compressed query positions must index the gathered logical KV sequence, "
                    f"got range [{int(query_positions.min().item())}, {int(query_positions.max().item())}] "
                    f"for KV length {sequence_length}"
                )
            request_state = Dsv4DecodeCarryState()
            if kv_preprocessed:
                _store_preprocessed_kv_carry(kv_input, request_state, 0, dequantize=False)
            else:
                _store_raw_kv_carry(
                    kv_input,
                    kv_norm_weight,
                    freqs_cis,
                    eps,
                    request_state,
                    0,
                    dequantize=False,
                )
            _serving_compressed_kv(
                x,
                compressor_wkv_weight,
                compressor_wgate_weight,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis,
                eps,
                ratio,
                carry_state=request_state,
                dequantize=False,
            )
            _validate_dsv4_lora_metadata(q, where=f"C{ratio} hybrid KV assembly")
            compressed_capacity = max(sequence_length // ratio, 1)
        output = _serving_attention_rows(
            q,
            request_state,
            query_positions,
            ratio,
            attn_sink,
            softmax_scale,
        )
        _validate_dsv4_lora_metadata(q, where=f"C{ratio} flash_mla_with_kvcache")

        if carry_state is not None and position_offset > 0:
            # Decode segments: carried compressed rows and the raw prefix are
            # serving cache constants; only this token's Q and raw KV rows are
            # differentiable (segment-local gradient ownership).
            if compressed.shape[0] < compressed_capacity:
                compressed = torch.cat(
                    (
                        compressed,
                        compressed.new_zeros((compressed_capacity - compressed.shape[0], 512)),
                    ),
                    dim=0,
                )
            kv = torch.cat((compressed, vanilla), dim=0).unsqueeze(1)
            kv_constant = kv[: compressed_capacity + position_offset, 0].detach()
            ctx.save_for_backward(
                q,
                kv_input,
                x,
                kv_norm_weight,
                attn_sink,
                freqs_cis,
                positions,
                kv_constant,
            )
            ctx.eps = eps
            ctx.softmax_scale = softmax_scale
            ctx.ratio = ratio
            ctx.carry_offset = position_offset
            ctx.compressed_capacity = compressed_capacity
            return output

        ctx.carry_offset = None
        ctx.save_for_backward(
            q,
            kv_input,
            x,
            kv_norm_weight,
            attn_sink,
            freqs_cis,
            compressor_wkv_weight,
            compressor_wgate_weight,
            compressor_ape,
            compressor_norm_weight,
            query_positions,
        )
        ctx.eps = eps
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        ctx.kv_preprocessed = kv_preprocessed
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        from xorl.ops.dsv4.attention_core import sparse_attn_torch  # noqa: PLC0415

        if ctx.carry_offset is not None:
            (
                q_saved,
                kv_saved,
                x_saved,
                kv_norm_weight,
                attn_sink,
                freqs_cis,
                positions,
                kv_constant,
            ) = ctx.saved_tensors
            indices, _ = _hybrid_indices_for_positions(positions, ctx.ratio, ctx.compressed_capacity)
            with torch.enable_grad():
                q = q_saved.detach().requires_grad_(True)
                kv_input = kv_saved.detach().requires_grad_(True)
                kv_segment = _kv_norm_rope_torch(kv_input, kv_norm_weight, freqs_cis, positions, ctx.eps)
                kv = torch.cat((kv_constant.to(kv_segment.dtype).unsqueeze(0), kv_segment), dim=1)
                surrogate = sparse_attn_torch(
                    q,
                    kv,
                    attn_sink,
                    indices.unsqueeze(0),
                    ctx.softmax_scale,
                )
                grad_q, grad_kv = torch.autograd.grad(
                    surrogate,
                    (q, kv_input),
                    grad_output,
                    create_graph=False,
                )
            # The carried compressed rows are constants of earlier segments, so
            # x receives no gradient from a decode step's compressor store.
            return (
                grad_q,
                grad_kv,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        (
            q_saved,
            kv_saved,
            x_saved,
            kv_norm_weight,
            attn_sink,
            freqs_cis,
            compressor_wkv_weight,
            compressor_wgate_weight,
            compressor_ape,
            compressor_norm_weight,
            query_positions,
        ) = ctx.saved_tensors
        sequence_length = kv_saved.shape[1]
        kv_positions = _positions(1, sequence_length, q_saved.device)
        compressed_capacity = max(sequence_length // ctx.ratio, 1)
        indices, _ = _hybrid_indices_for_positions(query_positions, ctx.ratio, compressed_capacity)
        with torch.enable_grad():
            q = q_saved.detach().requires_grad_(True)
            kv_input = kv_saved.detach().requires_grad_(True)
            x = x_saved.detach().requires_grad_(True)
            vanilla = (
                kv_input
                if ctx.kv_preprocessed
                else _kv_norm_rope_torch(kv_input, kv_norm_weight, freqs_cis, kv_positions, ctx.eps)
            )
            compressed = _compress_surrogate(
                x,
                compressor_wkv_weight,
                compressor_wgate_weight,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis,
                ctx.eps,
                ctx.ratio,
            )[0]
            if compressed.shape[0] < compressed_capacity:
                compressed = torch.cat(
                    (
                        compressed,
                        compressed.new_zeros((compressed_capacity - compressed.shape[0], 512)),
                    ),
                    dim=0,
                )
            kv = torch.cat((compressed, vanilla[0]), dim=0).unsqueeze(0)
            surrogate = sparse_attn_torch(
                q,
                kv,
                attn_sink,
                indices.unsqueeze(0),
                ctx.softmax_scale,
            )
            grad_q, grad_kv, grad_x = torch.autograd.grad(
                surrogate,
                (q, kv_input, x),
                grad_output,
                create_graph=False,
                # Short sequences can leave the compressed stream empty (e.g.
                # C128 with no complete block), making x genuinely unused;
                # its exact gradient is zero.
                allow_unused=True,
            )
            if grad_q is None:
                grad_q = torch.zeros_like(q)
            if grad_kv is None:
                grad_kv = torch.zeros_like(kv_input)
            if grad_x is None:
                grad_x = torch.zeros_like(x)
        return (
            grad_q,
            grad_kv,
            grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def exact_compressed_attention(
    q: Tensor,
    kv_input: Tensor,
    x: Tensor,
    kv_norm_weight: Tensor,
    attn_sink: Tensor,
    freqs_cis: Tensor,
    compressor_wkv_weight: Tensor,
    compressor_wgate_weight: Tensor,
    compressor_ape: Tensor,
    compressor_norm_weight: Tensor,
    eps: float,
    softmax_scale: float,
    ratio: int,
    carry_state: Dsv4DecodeCarryState | None = None,
    position_offset: int = 0,
    query_positions: Tensor | None = None,
    kv_preprocessed: bool = False,
) -> Tensor:
    """Run the pinned C4/C128 compact-cache attention with a trainer VJP.

    Without ``carry_state`` this is the whole-sequence prefill replay. With a
    ``carry_state`` it becomes the incremental serving decode replay: the
    segment appends its raw K entry and runs the serving compressor decode step
    over the carried kv-score ring and compressed cache.
    """

    return _ExactHybridAttention.apply(
        q,
        kv_input,
        x,
        kv_norm_weight,
        attn_sink,
        freqs_cis,
        compressor_wkv_weight,
        compressor_wgate_weight,
        compressor_ape,
        compressor_norm_weight,
        eps,
        softmax_scale,
        ratio,
        carry_state,
        position_offset,
        query_positions,
        kv_preprocessed,
    )


class _ExactInverseRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, freqs_cis: Tensor, positions: Tensor) -> Tensor:
        from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace  # noqa: PLC0415

        output = x.contiguous().clone()
        output_flat = output.view(-1, output.shape[-2], output.shape[-1])
        fused_rope_inplace(output_flat[..., -64:], None, freqs_cis, positions=positions, inverse=True)
        ctx.save_for_backward(freqs_cis, positions)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        freqs_cis, positions = ctx.saved_tensors
        return _apply_rope_torch(grad_output, freqs_cis, positions, inverse=False), None, None


def exact_inverse_rope(x: Tensor, freqs_cis: Tensor, position_offset: int = 0) -> Tensor:
    """Run SGLang's literal inverse-RoPE kernel with the exact rotation VJP."""

    batch_size, sequence_length = x.shape[:2]
    positions = _positions(batch_size, sequence_length, x.device, offset=position_offset)
    return _ExactInverseRope.apply(x, freqs_cis, positions)
