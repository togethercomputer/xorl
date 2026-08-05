"""GLM-5.2 sparse-selector parity with the serving runtime."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import torch

from xorl.models.transformers.glm5.index_share import CanonicalLogicalIndices


GLM52_SELECTOR_VERSION = "glm52_fp8_sampler_selector_v1"
GLM52_SELECTION_IMPORT = "sglang.srt.layers.attention.nsa.glm52_selector.select_canonical_logical_topk"
GLM52_HADAMARD_IMPORT = "sglang.jit_kernel.hadamard.hadamard_transform"
GLM52_QUERY_QUANT_IMPORT = "sglang.srt.layers.attention.nsa.triton_kernel.act_quant"
GLM52_KEY_CACHE_QUANT_IMPORT = "sglang.jit_kernel.fused_store_index_cache.fused_store_index_k_cache"
GLM52_SCORE_KERNEL_IMPORT = "deep_gemm.fp8_mqa_logits"
FP8_E4M3_MAX = 448.0

NativeScoreKernel = Callable[..., torch.Tensor]
NativeQuantKernel = Callable[..., tuple[torch.Tensor, torch.Tensor]]
NativeKeyCacheKernel = Callable[..., None]


@dataclass(frozen=True)
class Glm52Selection:
    logical_indices: CanonicalLogicalIndices
    valid_counts: torch.Tensor
    selector_version: str = GLM52_SELECTOR_VERSION


def _load_sparse_hadamard() -> NativeScoreKernel:
    module = importlib.import_module("sglang.jit_kernel.hadamard")
    kernel = getattr(module, "hadamard_transform", None)
    if not callable(kernel):
        raise RuntimeError(f"GLM-5.2 sparse selection requires {GLM52_HADAMARD_IMPORT}")
    return kernel


def _load_sparse_selection() -> NativeScoreKernel:
    # The trainer imports the sampler's literal selection kernel: a genuine
    # API break fails naturally right here, and residual numerical drift on
    # either side reads as nonzero behavior_k3 in the first training steps
    # and in the qualification replays. No version handshake is performed.
    module = importlib.import_module("sglang.srt.layers.attention.nsa.glm52_selector")
    selector = getattr(module, "select_canonical_logical_topk", None)
    if not callable(selector):
        raise RuntimeError(f"GLM-5.2 sparse selection requires {GLM52_SELECTION_IMPORT}")
    return selector


def _load_sparse_query_quantizer() -> NativeQuantKernel:
    module = importlib.import_module("sglang.srt.layers.attention.nsa.triton_kernel")
    kernel = getattr(module, "act_quant", None)
    if not callable(kernel):
        raise RuntimeError(f"GLM-5.2 sparse selection requires {GLM52_QUERY_QUANT_IMPORT}")
    return kernel


def _load_sparse_key_cache_quantizer() -> NativeKeyCacheKernel:
    module = importlib.import_module("sglang.jit_kernel.fused_store_index_cache")
    kernel = getattr(module, "fused_store_index_k_cache", None)
    if not callable(kernel):
        raise RuntimeError(f"GLM-5.2 sparse selection requires {GLM52_KEY_CACHE_QUANT_IMPORT}")
    return kernel


def _load_sparse_score_kernel() -> NativeScoreKernel:
    try:
        module = importlib.import_module("deep_gemm")
    except Exception as error:
        raise RuntimeError(f"GLM-5.2 sparse selection requires {GLM52_SCORE_KERNEL_IMPORT}") from error
    kernel = getattr(module, "fp8_mqa_logits", None)
    if not callable(kernel):
        raise RuntimeError(f"GLM-5.2 sparse selection requires {GLM52_SCORE_KERNEL_IMPORT}")
    return kernel


def rotate_sparse_selector_activation(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the sampler's normalized Hadamard transport before FP8 quantization.

    Serving rotates both DSA query and key rows after RoPE and before the
    sampler's E4M3 codecs. Quantizing the unrotated tensors is a different sparse
    selector even though the Hadamard transform preserves exact dot products:
    block-FP8 rounding happens after the change of basis.

    CUDA execution calls SGLang's literal public JIT kernel.  The small CPU
    reference exists for contract tests; it is not the certified serving path.
    """
    if tensor.is_cuda and tensor.dtype is not torch.bfloat16:
        raise TypeError(f"GLM-5.2 sparse selector CUDA Hadamard transport requires BF16, got {tensor.dtype}")
    if not tensor.is_cuda and not torch.is_floating_point(tensor):
        raise TypeError(f"GLM-5.2 sparse selector CPU Hadamard reference requires floating point, got {tensor.dtype}")
    hidden_size = tensor.shape[-1]
    if hidden_size <= 0 or hidden_size & (hidden_size - 1):
        raise ValueError(f"Hadamard width must be a positive power of two, got {hidden_size}")

    if tensor.is_cuda:
        return _load_sparse_hadamard()(tensor.contiguous(), scale=hidden_size**-0.5)

    transformed = tensor.float()
    butterfly = 1
    while butterfly < hidden_size:
        blocks = transformed.reshape(*transformed.shape[:-1], -1, 2 * butterfly)
        left = blocks[..., :butterfly]
        right = blocks[..., butterfly:]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape_as(transformed)
        butterfly *= 2
    return (transformed * (hidden_size**-0.5)).to(tensor.dtype)


def quantize_e4m3_ue8m0(
    tensor: torch.Tensor,
    *,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise E4M3 quantization with ceil-power-of-two (UE8M0) scales."""
    if tensor.shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension {tensor.shape[-1]} must be divisible by block_size={block_size}")
    fp32 = tensor.float()
    blocks = fp32.unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(amax / FP8_E4M3_MAX)))
    normalized = (blocks / scales.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = normalized.to(torch.float8_e4m3fn).flatten(-2)
    return quantized, scales


def quantize_e4m3_dynamic(
    tensor: torch.Tensor,
    *,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the sampler key-cache writer's FP32 scales."""
    if tensor.shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension {tensor.shape[-1]} must be divisible by block_size={block_size}")
    fp32 = tensor.float()
    blocks = fp32.unflatten(-1, (-1, block_size))
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-4) / FP8_E4M3_MAX
    normalized = (blocks / scales.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = normalized.to(torch.float8_e4m3fn).flatten(-2)
    return quantized, scales


def quantize_sparse_query(
    tensor: torch.Tensor,
    *,
    block_size: int = 128,
    _native_quantizer_for_testing: NativeQuantKernel | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the sampler's literal UE8M0 query codec on CUDA."""
    if tensor.is_cuda:
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"GLM-5.2 sparse selector query codec requires BF16, got {tensor.dtype}")
        quantizer = _native_quantizer_for_testing or _load_sparse_query_quantizer()
        quantized, scales = quantizer(tensor.contiguous(), block_size, "ue8m0")
    else:
        quantized, scales = quantize_e4m3_ue8m0(tensor, block_size=block_size)
    if quantized.dtype is not torch.float8_e4m3fn or scales.dtype is not torch.float32:
        raise RuntimeError("GLM-5.2 sparse selector query codec returned invalid dtypes")
    expected_scale_shape = (*tensor.shape[:-1], tensor.shape[-1] // block_size)
    if quantized.shape != tensor.shape or tuple(scales.shape) != expected_scale_shape:
        raise RuntimeError(
            "GLM-5.2 sparse selector query codec returned invalid shapes: "
            f"quantized={tuple(quantized.shape)}, scales={tuple(scales.shape)}"
        )
    return quantized, scales


def quantize_sparse_key_cache(
    tensor: torch.Tensor,
    *,
    block_size: int = 128,
    page_size: int = 64,
    _native_quantizer_for_testing: NativeKeyCacheKernel | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the sampler's literal fused FP8 key-cache writer and unpack logical rows."""
    if tensor.shape[-1] != block_size or block_size != 128:
        raise ValueError(f"GLM-5.2 sparse selector fused key-cache codec requires width 128, got {tensor.shape[-1]}")
    if page_size != 64:
        raise ValueError(f"GLM-5.2 sparse selector fused key-cache codec requires page_size=64, got {page_size}")
    if not tensor.is_cuda:
        return quantize_e4m3_dynamic(tensor, block_size=block_size)
    if tensor.dtype is not torch.bfloat16:
        raise TypeError(f"GLM-5.2 sparse selector key-cache codec requires BF16, got {tensor.dtype}")

    flat = tensor.reshape(-1, block_size).contiguous()
    token_count = flat.shape[0]
    if token_count == 0:
        raise ValueError("GLM-5.2 sparse selector key-cache codec requires at least one key row")
    page_count = (token_count + page_size - 1) // page_size
    row_bytes = block_size + 4
    cache = torch.zeros((page_count, page_size * row_bytes), device=tensor.device, dtype=torch.uint8)
    locations = torch.arange(token_count, device=tensor.device, dtype=torch.int64)
    quantizer = _native_quantizer_for_testing or _load_sparse_key_cache_quantizer()
    quantizer(flat, cache, locations, page_size=page_size)

    quantized, scales = _unpack_sparse_key_cache(cache, token_count, block_size=block_size, page_size=page_size)
    return quantized.reshape_as(tensor), scales.reshape(*tensor.shape[:-1], 1)


def _unpack_sparse_key_cache(
    cache: torch.Tensor,
    token_count: int,
    *,
    block_size: int = 128,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read logical rows from SGLang's page-major key-plus-scale buffer."""
    if cache.dtype is not torch.uint8 or cache.ndim != 2:
        raise TypeError("GLM-5.2 sparse selector key cache must be a rank-2 uint8 tensor")
    if cache.shape[1] != page_size * (block_size + 4):
        raise ValueError(f"GLM-5.2 sparse selector key cache has invalid page stride {cache.shape[1]}")
    if token_count < 0 or token_count > cache.shape[0] * page_size:
        raise ValueError(f"GLM-5.2 sparse selector key cache token_count={token_count} exceeds capacity")
    value_bytes = cache[:, : page_size * block_size].contiguous().reshape(-1, block_size)
    scale_bytes = cache[:, page_size * block_size :].contiguous()
    quantized = value_bytes.view(torch.float8_e4m3fn)[:token_count]
    scales = scale_bytes.view(torch.float32).reshape(-1)[:token_count]
    return quantized, scales


def dequantize_e4m3_blocks(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    if quantized.shape[-1] % block_size != 0:
        raise ValueError(f"Last dimension {quantized.shape[-1]} must be divisible by block_size={block_size}")
    blocks = quantized.float().unflatten(-1, (-1, block_size))
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"Scale shape {tuple(scales.shape)} does not match quantized blocks {tuple(blocks.shape)}")
    return (blocks * scales.unsqueeze(-1)).flatten(-2)


def select_glm52_logical_indices(
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    head_weights: torch.Tensor,
    allowed_keys: torch.Tensor,
    *,
    topk: int,
    block_size: int = 128,
    _native_kernel_for_testing: NativeScoreKernel | None = None,
    _selector_for_testing: NativeScoreKernel | None = None,
    _query_quantizer_for_testing: NativeQuantKernel | None = None,
    _key_cache_quantizer_for_testing: NativeKeyCacheKernel | None = None,
) -> Glm52Selection:
    """Select canonical logical keys using the sampler's FP8 score program.

    Ties are resolved by the smaller canonical logical key index. The returned
    valid indices are sorted in ascending logical order, followed by ``-1``
    sentinels when fewer than ``topk`` keys are legal.
    """
    if index_query.ndim != 4 or index_key.ndim != 3 or head_weights.ndim != 3:
        raise ValueError("Expected query [B,Q,H,D], key [B,K,D], and head_weights [B,Q,H]")
    batch, query_rows, heads, head_dim = index_query.shape
    if index_key.shape[0] != batch or index_key.shape[2] != head_dim:
        raise ValueError("Query and key batch/head dimensions do not match")
    if head_weights.shape != (batch, query_rows, heads):
        raise ValueError("head_weights shape does not match query")
    key_rows = index_key.shape[1]
    if allowed_keys.shape != (batch, query_rows, key_rows) or allowed_keys.dtype is not torch.bool:
        raise ValueError("allowed_keys must be a bool tensor shaped [B,Q,K]")
    if topk <= 0:
        raise ValueError("topk must be positive")

    if head_dim != block_size:
        raise ValueError(
            "GLM-5.2 sparse selector native scoring requires one UE8M0 scale per query head; "
            f"got head_dim={head_dim}, block_size={block_size}"
        )
    if _native_kernel_for_testing is None and not index_query.is_cuda:
        raise RuntimeError("GLM-5.2 sparse selector production selector requires CUDA DeepGEMM")
    if not (index_query.device == index_key.device == head_weights.device == allowed_keys.device):
        raise ValueError("GLM-5.2 sparse selector selector inputs must share one device")

    flat_allowed = allowed_keys.reshape(batch * query_rows, key_rows)
    legal_lengths = flat_allowed.sum(dim=-1, dtype=torch.int64)
    columns = torch.arange(key_rows, device=allowed_keys.device, dtype=torch.int64)
    prefix_allowed = columns.unsqueeze(0) < legal_lengths.unsqueeze(1)
    if not torch.equal(flat_allowed, prefix_allowed):
        raise RuntimeError("GLM-5.2 sparse selector native scoring admits only contiguous request-local key prefixes")

    with torch.no_grad():
        index_query = rotate_sparse_selector_activation(index_query)
        index_key = rotate_sparse_selector_activation(index_key)
        q_fp8, q_scales = quantize_sparse_query(
            index_query,
            block_size=block_size,
            _native_quantizer_for_testing=_query_quantizer_for_testing,
        )
        k_fp8, k_scales = quantize_sparse_key_cache(
            index_key,
            block_size=block_size,
            _native_quantizer_for_testing=_key_cache_quantizer_for_testing,
        )
        q_fp8 = q_fp8.reshape(batch * query_rows, heads, head_dim).contiguous()
        k_fp8 = k_fp8.reshape(batch * key_rows, head_dim).contiguous()
        k_scales = k_scales.reshape(batch * key_rows).contiguous()
        native_weights = (
            (head_weights.float() * q_scales.squeeze(-1).float() * (head_dim**-0.5))
            .reshape(batch * query_rows, heads)
            .contiguous()
        )

        request_offsets = torch.arange(batch, device=index_query.device, dtype=torch.int64) * key_rows
        request_offsets = request_offsets.unsqueeze(1).expand(batch, query_rows).reshape(-1)
        key_starts = request_offsets.to(torch.int32).contiguous()
        key_ends = (request_offsets + legal_lengths).to(torch.int32).contiguous()
        native_kernel = _native_kernel_for_testing or _load_sparse_score_kernel()
        native_scores = native_kernel(
            q_fp8,
            (k_fp8, k_scales),
            native_weights,
            key_starts,
            key_ends,
            clean_logits=False,
        )
        expected_shape = (batch * query_rows, batch * key_rows)
        if native_scores.dtype is not torch.float32 or tuple(native_scores.shape) != expected_shape:
            raise RuntimeError(
                "GLM-5.2 sparse selector DeepGEMM returned an invalid score tensor: "
                f"expected float32 {expected_shape}, got {native_scores.dtype} {tuple(native_scores.shape)}"
            )

        global_scores = native_scores.reshape(batch, query_rows, batch, key_rows)
        logits = torch.stack([global_scores[index, :, index, :] for index in range(batch)], dim=0)
        # DeepGEMM intentionally leaves illegal cells unwritten when
        # clean_logits=False. Overwrite every such cell before any selector
        # reads it, exactly matching SGLang's legal-interval contract.
        logits = logits.masked_fill(~allowed_keys, float("-inf"))
        if bool(torch.any(torch.isnan(logits) & allowed_keys).item()):
            raise RuntimeError("GLM-5.2 sparse selector DeepGEMM returned NaN for a legal key")

    selector = _selector_for_testing or _load_sparse_selection()
    output = selector(
        logits.reshape(batch * query_rows, key_rows),
        legal_lengths,
        topk=topk,
    )
    expected_output_shape = (batch * query_rows, topk)
    if output.dtype is not torch.int32 or tuple(output.shape) != expected_output_shape:
        raise RuntimeError(
            "GLM-5.2 sparse selector shared selector returned an invalid index tensor: "
            f"expected int32 {expected_output_shape}, got {output.dtype} {tuple(output.shape)}"
        )
    slot_is_valid = torch.arange(topk, device=output.device).unsqueeze(0) < legal_lengths.clamp_max(topk).unsqueeze(1)
    valid_output = output.masked_fill(~slot_is_valid, 0)
    legal_output = (valid_output >= 0) & (valid_output < legal_lengths.unsqueeze(1))
    suffix_is_padded = torch.where(slot_is_valid, torch.ones_like(slot_is_valid), output == -1)
    strictly_increasing = (valid_output[:, 1:] > valid_output[:, :-1]) | ~slot_is_valid[:, 1:]
    if not bool(torch.all(legal_output | ~slot_is_valid).item()):
        raise RuntimeError("GLM-5.2 sparse selector returned an out-of-range logical index")
    if not bool(torch.all(suffix_is_padded).item()):
        raise RuntimeError("GLM-5.2 sparse selector must pad every unused output slot with -1")
    if not bool(torch.all(strictly_increasing).item()):
        raise RuntimeError("GLM-5.2 sparse selector output must be sorted and unique")
    output = output.reshape(batch, query_rows, topk)
    valid_counts = legal_lengths.reshape(batch, query_rows).clamp_max(topk).to(torch.int32)
    return Glm52Selection(CanonicalLogicalIndices(output), valid_counts)


def physical_cache_to_logical_indices(
    physical_indices: torch.Tensor,
    physical_to_logical: torch.Tensor,
) -> CanonicalLogicalIndices:
    """Convert cache/page slots to canonical sequence positions."""
    if physical_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("physical_indices must be int32 or int64")
    if physical_to_logical.dtype not in (torch.int32, torch.int64):
        raise TypeError("physical_to_logical must be int32 or int64")
    if physical_to_logical.ndim not in (1, 2):
        raise ValueError("physical_to_logical must be [slots] or [batch, slots]")

    valid = physical_indices >= 0
    clamped = physical_indices.clamp_min(0).long()
    if physical_to_logical.ndim == 1:
        if bool(torch.any(clamped[valid] >= physical_to_logical.shape[0])):
            raise ValueError("physical index exceeds the physical-to-logical table")
        logical = physical_to_logical[clamped]
    else:
        if physical_indices.shape[0] != physical_to_logical.shape[0]:
            raise ValueError("Batched physical index and page-map batch dimensions differ")
        if bool(torch.any(clamped[valid] >= physical_to_logical.shape[1])):
            raise ValueError("physical index exceeds the batched physical-to-logical table")
        batch_index = torch.arange(physical_indices.shape[0], device=physical_indices.device)
        batch_index = batch_index.view(-1, *([1] * (physical_indices.ndim - 1))).expand_as(clamped)
        logical = physical_to_logical[batch_index, clamped]
    logical = logical.to(torch.int32).masked_fill(~valid, -1)
    return CanonicalLogicalIndices(logical)


def gather_selected_logical_values(values: torch.Tensor, indices: CanonicalLogicalIndices) -> torch.Tensor:
    """Gather selected values with finite-zero semantics for ``-1`` slots."""
    selected = indices.values
    if values.ndim != 3 or selected.ndim != 3:
        raise ValueError("Expected values [B,K,D] and selected indices [B,Q,T]")
    if values.shape[0] != selected.shape[0]:
        raise ValueError("Value and selected-index batch dimensions differ")
    valid = selected >= 0
    clamped = selected.clamp_min(0).long()
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1).expand_as(clamped)
    gathered = values[batch, clamped]
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))


__all__ = [
    "FP8_E4M3_MAX",
    "GLM52_HADAMARD_IMPORT",
    "GLM52_KEY_CACHE_QUANT_IMPORT",
    "GLM52_QUERY_QUANT_IMPORT",
    "GLM52_SCORE_KERNEL_IMPORT",
    "GLM52_SELECTION_IMPORT",
    "GLM52_SELECTOR_VERSION",
    "Glm52Selection",
    "dequantize_e4m3_blocks",
    "gather_selected_logical_values",
    "physical_cache_to_logical_indices",
    "quantize_e4m3_dynamic",
    "quantize_e4m3_ue8m0",
    "quantize_sparse_key_cache",
    "quantize_sparse_query",
    "rotate_sparse_selector_activation",
    "select_glm52_logical_indices",
]
