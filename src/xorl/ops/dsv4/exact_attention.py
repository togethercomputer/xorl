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


def _positions(batch_size: int, sequence_length: int, device: torch.device) -> Tensor:
    return torch.arange(sequence_length, dtype=torch.int64, device=device).repeat(batch_size)


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


def exact_q_norm_rope(q_input: Tensor, freqs_cis: Tensor, eps: float) -> Tensor:
    """Run SGLang's literal fused Q RMSNorm+RoPE kernel with a trainer VJP."""

    batch_size, sequence_length = q_input.shape[:2]
    positions = _positions(batch_size, sequence_length, q_input.device)
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


def _serving_compressed_kv(
    x: Tensor,
    wkv_weight: Tensor,
    wgate_weight: Tensor,
    ape: Tensor,
    norm_weight: Tensor,
    freqs_cis: Tensor,
    eps: float,
    ratio: int,
) -> Tensor:
    """Run the pinned compact compressor and its native cache store."""

    from sglang.kernels.ops.attention.dsv4 import (  # noqa: PLC0415
        CompressorPrefillPlan,
        compress_forward,
        compress_norm_rope_store,
        linear_bf16_fp32,
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
    if num_compressed == 0:
        return x.new_empty((0, 512))

    x_flat = x.contiguous().view(sequence_length, 4096)
    fused_weight = torch.cat((wkv_weight, wgate_weight), dim=0).to(torch.bfloat16).contiguous()
    kv_score = linear_bf16_fp32(x_flat, fused_weight)
    _validate_dsv4_lora_metadata(x, where=f"C{ratio} compressor linear_bf16_fp32")
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
    serving_ape = ape
    if ratio == 4:
        serving_ape = torch.cat(torch.chunk(ape, 2, dim=-1), dim=0)
    serving_ape = serving_ape.reshape(-1, 512).float().contiguous()
    compressed = compress_forward(
        kv_score_buffer=state,
        kv_score_input=kv_score,
        ape=serving_ape,
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
    page_bytes = ((584 * page_size + 575) // 576) * 576
    num_pages = (num_compressed + page_size - 1) // page_size
    kvcache = torch.empty((num_pages, page_bytes), dtype=torch.uint8, device=x.device)
    out_loc = torch.zeros(sequence_length, dtype=torch.int64, device=x.device)
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
        return grad_q, grad_kv, None, None, None, None, None


def exact_c0_attention(
    q: Tensor,
    kv_input: Tensor,
    kv_norm_weight: Tensor,
    attn_sink: Tensor,
    freqs_cis: Tensor,
    eps: float,
    softmax_scale: float,
) -> Tensor:
    """Run native SWA-cache quantization and FlashMLA sparse prefill for C0."""

    return _ExactC0Attention.apply(
        q,
        kv_input,
        kv_norm_weight,
        attn_sink,
        freqs_cis,
        eps,
        softmax_scale,
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
            )
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
) -> Tensor:
    """Run the pinned C4/C128 compact-cache prefill with a trainer VJP."""

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


def exact_inverse_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Run SGLang's literal inverse-RoPE kernel with the exact rotation VJP."""

    batch_size, sequence_length = x.shape[:2]
    positions = _positions(batch_size, sequence_length, x.device)
    return _ExactInverseRope.apply(x, freqs_cis, positions)
