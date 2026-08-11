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


def _positions(
    batch_size: int, sequence_length: int, device: torch.device, offset: int = 0
) -> Tensor:
    return torch.arange(
        offset, offset + sequence_length, dtype=torch.int64, device=device
    ).repeat(batch_size)


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


def _ensure_paged_kvcache(
    cache: Tensor | None, num_slots: int, page_size: int, device: torch.device
) -> Tensor:
    """Grow (never shrink) a paged FlashMLA FP8 cache to hold ``num_slots``.

    Pages are zero-filled like serving's pool allocation (``create_buffer``
    uses ``torch.zeros``): the decode kernel's masked lanes still read cache
    bytes behind invalid indices, and non-zero garbage there perturbs the
    online-softmax max by a few ULPs (layer-3 C128 first-decode divergence of
    the 2026-08-11 base ruler).
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

    return cache[:, : page_size * _SERVING_SLOT_BYTES].view(
        cache.shape[0], page_size, 1, _SERVING_SLOT_BYTES
    )


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
    every decode shape (C4/C128 decisions of the 2026-08-11 base ruler), so the
    decode segments must call the literal serving entry point.
    """

    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: PLC0415

    device = q.device
    positions = torch.tensor([position], dtype=torch.int64, device=device)
    # Serving's decode window indices are DESCENDING from the current position
    # (build_causal_swa_page_indices: pos - arange(window), invalid -> -1). The
    # accumulation order is part of the byte contract: ascending indices differ
    # by a few ULPs on some heads (C4/C128 decisions of the 2026-08-11 ruler).
    window_offsets = position - torch.arange(128, dtype=torch.int32, device=device)
    swa_indices = window_offsets.masked_fill(window_offsets < 0, -1).view(1, 1, 128)
    swa_topk_length = torch.clamp(positions + 1, max=128).to(torch.int32)
    extra_kwargs = {}
    if ratio:
        blocks = carry_state.num_compressed
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
        q=q.view(1, 1, 64, 512),
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


def _store_raw_kv_carry(
    kv_input: Tensor,
    kv_norm_weight: Tensor,
    freqs_cis: Tensor,
    eps: float,
    carry_state: Dsv4DecodeCarryState,
    position_offset: int,
) -> Tensor:
    """Append a segment's normed+roped K entries to the carried FP8 cache.

    Mirrors serving's fused store (``set_swa_key_buffer_radix_fused_norm_rope``)
    with the trainer's flat token addressing (cache slot == absolute position),
    then dequantizes every carried row exactly as the serving sparse-prefill
    reader does.
    """

    from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla  # noqa: PLC0415
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

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
    carry_state.kvcache = _ensure_paged_kvcache(
        carry_state.kvcache, total_tokens, page_size, kv_input.device
    )
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
    all_locs = torch.arange(total_tokens, dtype=torch.int32, device=kv_input.device)
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


def exact_q_norm_rope(
    q_input: Tensor, freqs_cis: Tensor, eps: float, position_offset: int = 0
) -> Tensor:
    """Run SGLang's literal fused Q RMSNorm+RoPE kernel with a trainer VJP."""

    batch_size, sequence_length = q_input.shape[:2]
    positions = _positions(batch_size, sequence_length, q_input.device, offset=position_offset)
    return _ExactQNormRope.apply(q_input, freqs_cis, positions, eps)


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
    of every compressed KV row (second divergence of the 2026-08-11 base
    ruler, after the ape-layout fix).
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
        raise RuntimeError(
            "DSV4 exact compressor decode step admits one official hidden-size token"
        )
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
    output = dequantize_k_cache_paged(
        carry_state.compressed_kvcache, compressed_locs, page_size
    ).view(carry_state.num_compressed, 512)
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
) -> Tensor:
    """Run the pinned compact compressor and its native cache store."""

    from sglang.kernels.ops.attention.dsv4 import (  # noqa: PLC0415
        CompressorPrefillPlan,
        compress_forward,
        compress_norm_rope_store,
    )
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
        dequantize_k_cache_paged,
    )

    if x.shape[0] != 1 or x.shape[-1] != 4096 or ratio not in (4, 128):
        raise RuntimeError(
            "DSV4 exact compressor admits one official hidden-size request and ratio 4 or 128"
        )
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
        raise RuntimeError(
            f"DSV4 compressor emitted {compressed.shape[0]} rows for {sequence_length=} {ratio=}"
        )

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
    if num_compressed == 0:
        return x.new_empty((0, 512))
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
            indices[position, :compressed_length] = torch.arange(
                compressed_length, dtype=torch.int32, device=device
            )
        swa_start = position - swa_length + 1
        indices[position, compressed_length : compressed_length + swa_length] = (
            compressed_capacity
            + torch.arange(swa_start, position + 1, dtype=torch.int32, device=device)
        )
        lengths[position] = compressed_length + swa_length
    return indices, lengths, compressed_capacity


def _hybrid_indices_for_positions(
    positions: Tensor, ratio: int, compressed_capacity: int
) -> tuple[Tensor, Tensor]:
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
            raise RuntimeError(
                f"DSV4 exact hybrid index row overflow at position {position} for ratio {ratio}"
            )
        if compressed_length:
            indices[row, :compressed_length] = torch.arange(
                compressed_length, dtype=torch.int32, device=device
            )
        swa_start = position - swa_length + 1
        indices[row, compressed_length : compressed_length + swa_length] = (
            compressed_capacity
            + torch.arange(swa_start, position + 1, dtype=torch.int32, device=device)
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
    ) -> Tensor:
        if not q.is_cuda or q.dtype != torch.bfloat16 or kv_input.dtype != torch.bfloat16:
            raise RuntimeError("DSV4 exact C0 attention requires CUDA BF16 Q/KV inputs")
        if q.shape[-2:] != (64, 512) or kv_input.shape[-1] != 512:
            raise RuntimeError("DSV4 exact C0 attention admits only official DSV4-Flash geometry")

        from sgl_kernel.flash_mla import flash_mla_sparse_fwd  # noqa: PLC0415
        from sglang.kernels.ops.attention.dsv4 import (  # noqa: PLC0415
            fused_k_norm_rope_flashmla,
        )
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (  # noqa: PLC0415
            dequantize_k_cache_paged,
        )

        if carry_state is not None:
            # Incremental serving replay over carried per-layer state: append
            # this segment's K entries to the carried FP8 cache and attend from
            # the segment's absolute positions over every carried row.
            if position_offset > 0 and q.shape[1] != 1:
                raise RuntimeError(
                    "DSV4 exact C0 decode-cache carry admits M=1 decode segments"
                )
            kv = _store_raw_kv_carry(
                kv_input, kv_norm_weight, freqs_cis, eps, carry_state, position_offset
            )
            sequence_length = q.shape[1]
            positions = _positions(1, sequence_length, q.device, offset=position_offset)
            if position_offset > 0:
                # M=1 decode step: the literal serving decode kernel over the
                # carried paged FP8 cache.
                output = _serving_decode_attention(
                    q,
                    carry_state,
                    position_offset,
                    0,
                    attn_sink,
                    softmax_scale,
                )[0]
            else:
                # Prefill seed: identical to the whole-sequence exact prefill.
                indices = _window_indices_for_positions(positions)
                topk_length = torch.clamp(positions + 1, max=128).to(torch.int32)
                output, _, _ = flash_mla_sparse_fwd(
                    q=q.contiguous().view(sequence_length, 64, 512),
                    kv=kv.view(carry_state.num_tokens, 1, 512),
                    indices=indices.unsqueeze(1),
                    sm_scale=softmax_scale,
                    d_v=512,
                    attn_sink=attn_sink,
                    topk_length=topk_length,
                )
            # Carried prefix rows are serving cache constants; only this
            # segment's Q/KV rows are differentiable.
            prefix = kv[:position_offset].detach()
            ctx.save_for_backward(
                q, kv_input, kv_norm_weight, attn_sink, freqs_cis, positions, prefix
            )
            ctx.eps = eps
            ctx.softmax_scale = softmax_scale
            ctx.carry_offset = position_offset
            return output.unsqueeze(0)

        if position_offset != 0:
            raise RuntimeError("DSV4 exact C0 attention requires a carry state for offset segments")
        ctx.carry_offset = None
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
        kv = dequantize_k_cache_paged(kvcache, out_loc, page_size)
        indices = _causal_window_indices(1, sequence_length, q.device)[0]
        topk_length = torch.minimum(
            torch.arange(1, sequence_length + 1, dtype=torch.int32, device=q.device),
            torch.tensor(128, dtype=torch.int32, device=q.device),
        )

        outputs = []
        q_flat = q.contiguous().view(batch_size, sequence_length, 64, 512)
        kv_flat = kv.view(batch_size, sequence_length, 1, 512)
        for batch_idx in range(batch_size):
            output, _, _ = flash_mla_sparse_fwd(
                q=q_flat[batch_idx],
                kv=kv_flat[batch_idx],
                indices=indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
                attn_sink=attn_sink,
                topk_length=topk_length,
            )
            outputs.append(output)

        ctx.save_for_backward(q, kv_input, kv_norm_weight, attn_sink, freqs_cis, positions)
        ctx.eps = eps
        ctx.softmax_scale = softmax_scale
        return torch.stack(outputs, dim=0)

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
            return grad_q, grad_kv, None, None, None, None, None, None, None

        q_saved, kv_saved, weight, attn_sink, freqs_cis, positions = ctx.saved_tensors
        batch_size, sequence_length = kv_saved.shape[:2]
        indices = _causal_window_indices(batch_size, sequence_length, q_saved.device)
        with torch.enable_grad():
            q = q_saved.detach().requires_grad_(True)
            kv_input = kv_saved.detach().requires_grad_(True)
            kv = _kv_norm_rope_torch(kv_input, weight, freqs_cis, positions, ctx.eps)
            surrogate = sparse_attn_torch(q, kv, attn_sink, indices, ctx.softmax_scale)
            grad_q, grad_kv = torch.autograd.grad(
                surrogate,
                (q, kv_input),
                grad_output,
                create_graph=False,
            )
        return grad_q, grad_kv, None, None, None, None, None, None, None


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
    ) -> Tensor:
        if q.shape[0] != 1 or q.shape[1] > 128:
            raise RuntimeError(
                "DSV4 exact compressed attention currently admits one request with at most 128 tokens"
            )
        if q.dtype != torch.bfloat16 or kv_input.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
            raise RuntimeError("DSV4 exact compressed attention requires BF16 activations")
        if ratio not in (4, 128):
            raise RuntimeError(f"DSV4 exact compressed attention received unsupported ratio {ratio}")

        from sgl_kernel.flash_mla import flash_mla_sparse_fwd  # noqa: PLC0415

        sequence_length = q.shape[1]
        _validate_dsv4_lora_metadata(q, where=f"C{ratio} hybrid attention entry")
        if carry_state is not None:
            if position_offset > 0 and sequence_length != 1:
                raise RuntimeError(
                    "DSV4 exact compressed decode-cache carry admits M=1 decode segments"
                )
            vanilla = _store_raw_kv_carry(
                kv_input, kv_norm_weight, freqs_cis, eps, carry_state, position_offset
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
                    "DSV4 exact compressed carry lost a block: "
                    f"{carry_state.num_compressed} != {total_tokens // ratio}"
                )
            _validate_dsv4_lora_metadata(q, where=f"C{ratio} hybrid KV assembly")
            compressed_capacity = max(total_tokens // ratio, 1)
            positions = _positions(1, sequence_length, q.device, offset=position_offset)
            indices, lengths = _hybrid_indices_for_positions(positions, ratio, compressed_capacity)
        else:
            if position_offset != 0:
                raise RuntimeError(
                    "DSV4 exact compressed attention requires a carry state for offset segments"
                )
            vanilla = _native_swa_kv(kv_input, kv_norm_weight, freqs_cis, eps)[0]
            compressed = _serving_compressed_kv(
                x,
                compressor_wkv_weight,
                compressor_wgate_weight,
                compressor_ape,
                compressor_norm_weight,
                freqs_cis,
                eps,
                ratio,
            )
            _validate_dsv4_lora_metadata(q, where=f"C{ratio} hybrid KV assembly")
            indices, lengths, compressed_capacity = _hybrid_prefill_indices(
                sequence_length, ratio, q.device
            )
        if compressed.shape[0] < compressed_capacity:
            compressed = torch.cat(
                (
                    compressed,
                    compressed.new_zeros((compressed_capacity - compressed.shape[0], 512)),
                ),
                dim=0,
            )
        kv = torch.cat((compressed, vanilla), dim=0).unsqueeze(1)
        if carry_state is not None and position_offset > 0:
            # M=1 decode step: the literal serving decode kernel over the
            # carried paged FP8 raw and compressed caches.
            output = _serving_decode_attention(
                q,
                carry_state,
                position_offset,
                ratio,
                attn_sink,
                softmax_scale,
            )[0]
        else:
            output, _, _ = flash_mla_sparse_fwd(
                q=q[0],
                kv=kv,
                indices=indices.unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
                attn_sink=attn_sink,
                topk_length=lengths,
            )
        _validate_dsv4_lora_metadata(q, where=f"C{ratio} flash_mla_sparse_fwd")

        if carry_state is not None and position_offset > 0:
            # Decode segments: carried compressed rows and the raw prefix are
            # serving cache constants; only this token's Q and raw KV rows are
            # differentiable (segment-local gradient ownership).
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
            return output.unsqueeze(0)

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
        )
        ctx.eps = eps
        ctx.softmax_scale = softmax_scale
        ctx.ratio = ratio
        return output.unsqueeze(0)

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
            indices, _ = _hybrid_indices_for_positions(
                positions, ctx.ratio, ctx.compressed_capacity
            )
            with torch.enable_grad():
                q = q_saved.detach().requires_grad_(True)
                kv_input = kv_saved.detach().requires_grad_(True)
                kv_segment = _kv_norm_rope_torch(
                    kv_input, kv_norm_weight, freqs_cis, positions, ctx.eps
                )
                kv = torch.cat(
                    (kv_constant.to(kv_segment.dtype).unsqueeze(0), kv_segment), dim=1
                )
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
        ) = ctx.saved_tensors
        sequence_length = q_saved.shape[1]
        positions = _positions(1, sequence_length, q_saved.device)
        indices, _, compressed_capacity = _hybrid_prefill_indices(
            sequence_length, ctx.ratio, q_saved.device
        )
        with torch.enable_grad():
            q = q_saved.detach().requires_grad_(True)
            kv_input = kv_saved.detach().requires_grad_(True)
            x = x_saved.detach().requires_grad_(True)
            vanilla = _kv_norm_rope_torch(kv_input, kv_norm_weight, freqs_cis, positions, ctx.eps)
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
