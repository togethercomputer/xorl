"""GLM-5 DeepSeek Sparse Attention (DSA) lightning indexer.

The indexer projects the compressed query (shared with MLA's `q_a_proj`
output) and a per-token key drawn from the unprojected hidden state into a
small head space, scores all (q, k) pairs with ReLUed per-head dot products
and learned head weights, and returns the top-`index_topk` keys per query.
Sparse-MLA attends only to those keys.

V0 here is a torch reference: it builds the four projections with the right
shapes so checkpoints load cleanly, and the forward returns top-k indices for
the dense DSA mask and the sparse-MLA path.
"""

import importlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from xorl.models.exact_contract import glm52_exact_forward_enabled
from xorl.models.layers.moe.moe_block import _moe_bi_router_enabled
from xorl.models.transformers.glm5.rotary import glm5_apply_rotary_pos_emb
from xorl.models.transformers.glm5.sparse_selector import (
    GLM52_SELECTOR_VERSION,
    select_glm52_logical_indices,
)
from xorl.ops.batch_invariant_ops import bi_bf16_fp32_linear, matmul_persistent
from xorl.ops.exact.rope_class_b import build_class_b_cos_sin


GLM52_FUSED_INDEX_K_IMPORT = "sglang.kernels.ops.quantization.dsv32.elementwise.fused_k_indexer_norm_rope"


def _load_fused_sampler_index_k_prepare():
    module = importlib.import_module("sglang.kernels.ops.quantization.dsv32.elementwise")
    kernel = getattr(module, "fused_k_indexer_norm_rope", None)
    if not callable(kernel):
        raise RuntimeError(f"GLM-5.2 exact indexer requires {GLM52_FUSED_INDEX_K_IMPORT}")
    return kernel


def _fused_sampler_index_k_prepare(
    raw_key: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    interleaved: bool,
    _native_kernel_for_testing=None,
) -> torch.Tensor:
    """Run serving's fused LayerNorm + RoPE program before index-K quantization."""

    if not interleaved:
        raise RuntimeError("GLM-5.2 exact indexer K preparation requires interleaved RoPE")
    if raw_key.dtype is not torch.bfloat16 or raw_key.shape[-1] != 128:
        raise TypeError(
            "GLM-5.2 exact indexer K preparation requires BF16 keys with width 128, "
            f"got dtype={raw_key.dtype} shape={tuple(raw_key.shape)}"
        )
    if tuple(norm_weight.shape) != (128,) or tuple(norm_bias.shape) != (128,):
        raise ValueError("GLM-5.2 exact indexer K LayerNorm parameters must both have width 128")
    if norm_weight.dtype is not torch.float32 or norm_bias.dtype is not torch.float32:
        raise TypeError("GLM-5.2 exact indexer K LayerNorm parameters must be FP32")
    cos, sin = position_embeddings
    if cos.dtype is not torch.float32 or sin.dtype is not torch.float32:
        raise TypeError("GLM-5.2 exact indexer K preparation requires FP32 cos/sin")
    if cos.shape != sin.shape or cos.shape[:-1] != raw_key.shape[:-1] or cos.shape[-1] != 64:
        raise ValueError(
            "GLM-5.2 exact indexer K position embeddings must match the key rows with width 64, "
            f"got key={tuple(raw_key.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
        )

    cos_half, sin_half = build_class_b_cos_sin(cos, sin)
    cos_sin_cache = torch.cat((cos_half, sin_half), dim=-1).contiguous()
    flat_key = raw_key.reshape(-1, raw_key.shape[-1])
    if flat_key.stride(-1) != 1:
        raise RuntimeError("GLM-5.2 exact indexer K projection must be contiguous within each row")
    positions = torch.arange(flat_key.shape[0], device=flat_key.device, dtype=torch.int64)
    kernel = _native_kernel_for_testing or _load_fused_sampler_index_k_prepare()
    prepared = kernel(
        flat_key,
        norm_weight,
        norm_bias,
        eps,
        cos_sin_cache,
        positions,
    )
    if prepared.dtype is not torch.bfloat16 or tuple(prepared.shape) != tuple(flat_key.shape):
        raise RuntimeError(
            "GLM-5.2 fused indexer K preparation returned an invalid tensor: "
            f"dtype={prepared.dtype} shape={tuple(prepared.shape)}"
        )
    return prepared.view_as(raw_key)


def _mix_sampler_index_k_preparation(
    split_prepared_key: torch.Tensor,
    raw_key: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    sampler_prefill_lengths: torch.Tensor | None,
    *,
    query_offset: int,
    interleaved: bool,
    _native_kernel_for_testing=None,
) -> torch.Tensor:
    """Keep sampler prefill numerics and use fused decode numerics only after its boundary."""

    if sampler_prefill_lengths is None:
        return split_prepared_key
    if sampler_prefill_lengths.ndim != 1 or sampler_prefill_lengths.numel() != raw_key.shape[0]:
        raise ValueError(
            "GLM-5.2 sampler_prefill_lengths must contain one boundary per batch row, "
            f"got {tuple(sampler_prefill_lengths.shape)} for batch={raw_key.shape[0]}"
        )
    if sampler_prefill_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("GLM-5.2 sampler_prefill_lengths must be an integer tensor")
    if query_offset < 0:
        raise ValueError(f"GLM-5.2 exact indexer query_offset must be nonnegative, got {query_offset}")

    cos, sin = position_embeddings
    mixed_rows = []
    local_length = raw_key.shape[1]
    for batch_index in range(raw_key.shape[0]):
        prefill_length = int(sampler_prefill_lengths[batch_index].item())
        if prefill_length <= 0:
            raise ValueError(f"GLM-5.2 sampler prefill length must be positive, got {prefill_length}")
        suffix_start = min(max(prefill_length - query_offset, 0), local_length)
        if suffix_start == local_length:
            mixed_rows.append(split_prepared_key[batch_index : batch_index + 1])
            continue
        fused_suffix = _fused_sampler_index_k_prepare(
            raw_key[batch_index : batch_index + 1, suffix_start:],
            norm_weight,
            norm_bias,
            eps,
            (
                cos[batch_index : batch_index + 1, suffix_start:],
                sin[batch_index : batch_index + 1, suffix_start:],
            ),
            interleaved=interleaved,
            _native_kernel_for_testing=_native_kernel_for_testing,
        )
        mixed_rows.append(
            torch.cat(
                (split_prepared_key[batch_index : batch_index + 1, :suffix_start], fused_suffix),
                dim=1,
            )
        )
    return torch.cat(mixed_rows, dim=0)


def _fused_bf16_indexer_projection(
    hidden_states: torch.Tensor,
    wk_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    *,
    index_head_dim: int,
    index_n_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SGLang's fused BF16 ``wk + weights_proj`` projection."""

    if isinstance(wk_weight, DTensor) or isinstance(weights_proj_weight, DTensor):
        raise RuntimeError("GLM-5.2 fused indexer projection requires full plain weights")
    expected_wk_shape = (index_head_dim, hidden_states.shape[-1])
    expected_gate_shape = (index_n_heads, hidden_states.shape[-1])
    if tuple(wk_weight.shape) != expected_wk_shape or tuple(weights_proj_weight.shape) != expected_gate_shape:
        raise ValueError(
            "GLM-5.2 fused indexer projection received invalid weight shapes: "
            f"wk={tuple(wk_weight.shape)} expected={expected_wk_shape}, "
            f"weights_proj={tuple(weights_proj_weight.shape)} expected={expected_gate_shape}"
        )
    if not (
        hidden_states.dtype is torch.bfloat16
        and wk_weight.dtype is torch.bfloat16
        and weights_proj_weight.dtype is torch.bfloat16
    ):
        raise TypeError("GLM-5.2 fused indexer projection requires BF16 activations and weights")
    if not (hidden_states.device == wk_weight.device == weights_proj_weight.device):
        raise RuntimeError("GLM-5.2 fused indexer projection inputs must share one device")

    fused_weight = torch.cat((wk_weight, weights_proj_weight), dim=0).contiguous()
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if hidden_states.is_cuda:
        # SGLang's deterministic serving contract interposes its
        # batch-invariant ``aten::mm`` on this otherwise ordinary fused
        # linear.  Call the vendored equivalent directly so the trainer does
        # not depend on global matmul interposition.
        projected = matmul_persistent(flat_hidden, fused_weight.t())
    else:
        projected = F.linear(flat_hidden, fused_weight)
    projected = projected.view(*hidden_states.shape[:-1], index_head_dim + index_n_heads)
    return projected.split((index_head_dim, index_n_heads), dim=-1)


def _scale_fused_bf16_indexer_head_gates(
    head_weights: torch.Tensor,
    *,
    index_n_heads: int,
) -> torch.Tensor:
    """Match SGLang's FP32 gate scaling after the fused BF16 projection."""

    if head_weights.dtype is not torch.bfloat16:
        raise TypeError(f"GLM-5.2 fused indexer gate scaling requires BF16 gates, got {head_weights.dtype}")
    if index_n_heads <= 0:
        raise ValueError(f"GLM-5.2 fused indexer gate scaling requires positive head count, got {index_n_heads}")
    return head_weights.float() * index_n_heads**-0.5


class Glm5DsaIndexer(nn.Module):
    """DeepSeek Sparse Attention indexer.

    Args mirror the four HF parameter groups: ``wq_b`` (q_lora_rank →
    index_n_heads × index_head_dim), ``wk`` (hidden_size → index_head_dim),
    ``k_norm`` (LayerNorm over index_head_dim — with bias, hardcoded
    ``eps=1e-6`` to match the miles reference), and ``weights_proj``
    (hidden_size → index_n_heads, fp32 head gating).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.index_head_dim = config.index_head_dim
        self.index_n_heads = config.index_n_heads
        self.index_topk = config.index_topk
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.wq_b = nn.Linear(self.q_lora_rank, self.index_n_heads * self.index_head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.index_head_dim, bias=False)
        # miles: hardcoded LayerNorm with eps=1e-6 even when the rest of
        # the model uses RMSNorm (`miles_plugins/models/glm5/glm5.py`).
        self.k_norm = nn.LayerNorm(self.index_head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.index_n_heads, bias=False)

        self._head_weight_scale = self.index_n_heads**-0.5
        self._softmax_scale = self.index_head_dim**-0.5
        self.selector_version = (
            GLM52_SELECTOR_VERSION if glm52_exact_forward_enabled(config) else "legacy_torch_or_tilelang"
        )

    def project(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        sampler_prefill_lengths: torch.Tensor | None = None,
        query_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute index_query, index_key, head_weights.

        Shapes (batch=B, seq=S):
        - hidden_states: ``[B, S, hidden_size]``
        - q_compressed: ``[B, S, q_lora_rank]``  (post-`q_a_layernorm` from MLA)
        - position_embeddings: ``(cos, sin)`` for the partial-rotary RoPE band
        - returns: index_query ``[B, S, index_n_heads, index_head_dim]``,
          index_key ``[B, S, index_head_dim]``, head_weights ``[B, S, index_n_heads]``
        """
        B, S, _ = hidden_states.shape
        cos, sin = position_embeddings

        index_q = self.wq_b(q_compressed)
        index_q = index_q.view(B, S, self.index_n_heads, self.index_head_dim)

        exact_fused_projection = self.selector_version == GLM52_SELECTOR_VERSION and hidden_states.is_cuda
        if exact_fused_projection:
            wk_weight = self.wk(return_dequantized_weight=True)
            index_k, head_weights = _fused_bf16_indexer_projection(
                hidden_states,
                wk_weight,
                self.weights_proj.weight,
                index_head_dim=self.index_head_dim,
                index_n_heads=self.index_n_heads,
            )
            raw_index_k = index_k
        else:
            index_k = self.wk(hidden_states)
            raw_index_k = None

        if exact_fused_projection:
            try:
                from flashinfer.norm import layernorm as sampler_layer_norm  # noqa: PLC0415
            except (ImportError, AttributeError) as error:
                raise RuntimeError(
                    "GLM-5.2 native selector requires the sampler's FlashInfer LayerNorm kernel"
                ) from error
            index_k_shape = index_k.shape
            index_k = sampler_layer_norm(
                index_k.reshape(-1, index_k_shape[-1]),
                self.k_norm.weight.float(),
                self.k_norm.bias.float(),
                self.k_norm.eps,
            ).view(index_k_shape)
        else:
            index_k = self.k_norm(index_k)

        if exact_fused_projection:
            # Serving's compiled scaler promotes the fused BF16 gate before
            # applying the head scale; materializing this multiply in BF16
            # changes every FP32 score weight.
            head_weights = _scale_fused_bf16_indexer_head_gates(
                head_weights,
                index_n_heads=self.index_n_heads,
            )
        elif _moe_bi_router_enabled(self.config):
            if hidden_states.dtype is not torch.bfloat16 or self.weights_proj.weight.dtype is not torch.bfloat16:
                raise TypeError(
                    "GLM-5.2 batch-invariant indexer projection requires BF16 hidden states and BF16 weights"
                )
            head_weights = bi_bf16_fp32_linear(hidden_states, self.weights_proj.weight)
        else:
            head_weights = self.weights_proj(hidden_states).float()
        if not exact_fused_projection:
            head_weights = head_weights * self._head_weight_scale

        # Split (pe, no-pe) bands so we can apply the indexer's RoPE to
        # the leading `qk_rope_head_dim`, matching the GLM DSA reference.
        nope_dim = self.index_head_dim - self.qk_rope_head_dim
        q_pe, q_no_pe = torch.split(index_q, [self.qk_rope_head_dim, nope_dim], dim=-1)
        k_pe = index_k[..., : self.qk_rope_head_dim].unsqueeze(2)  # [B, S, 1, qk_rope_head_dim]
        k_no_pe = index_k[..., self.qk_rope_head_dim :]

        q_pe, k_pe = glm5_apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            interleaved=getattr(self.config, "indexer_rope_interleave", True),
            class_b=bool(getattr(self.config, "_rope_class_b", False) or glm52_exact_forward_enabled(self.config)),
        )

        index_q = torch.cat([q_pe, q_no_pe], dim=-1)
        index_k = torch.cat([k_pe.squeeze(2), k_no_pe], dim=-1)
        if exact_fused_projection:
            assert raw_index_k is not None
            index_k = _mix_sampler_index_k_preparation(
                index_k,
                raw_index_k,
                self.k_norm.weight.float(),
                self.k_norm.bias.float(),
                self.k_norm.eps,
                position_embeddings,
                sampler_prefill_lengths,
                query_offset=query_offset,
                interleaved=getattr(self.config, "indexer_rope_interleave", True),
            )

        return index_q, index_k, head_weights

    @staticmethod
    def _is_pure_causal_mask(mask: torch.Tensor, Q: int, K: int, query_offset: int) -> bool:
        """Quick check whether `mask` is just the standard causal mask (i.e.,
        mask[b, q, k] allowed iff k <= query_offset + q, regardless of batch).
        This lets us use the tilelang fast path when training passes an
        explicit causal mask instead of None."""
        if mask.dim() not in (3, 4):
            return False
        # Normalize 4D to 3D by dropping the heads axis (mask is broadcastable)
        m = mask if mask.dim() == 3 else mask[:, 0]
        if m.shape[-2] != Q or m.shape[-1] != K:
            return False
        # Build the reference causal mask: True where allowed, False otherwise.
        device = m.device
        q_pos = query_offset + torch.arange(Q, device=device)
        k_pos = torch.arange(K, device=device)
        causal_allowed = k_pos.view(1, K) <= q_pos.view(Q, 1)  # [Q, K]
        # Treat the mask as allowed-iff-zero (additive) or allowed-iff-True (bool).
        if m.dtype == torch.bool:
            actually_allowed = m
        elif torch.is_floating_point(m):
            actually_allowed = m == 0
        else:
            actually_allowed = m != 0
        # Need actually_allowed[b, q, k] == causal_allowed[q, k] for all b
        if actually_allowed.dim() == 3:
            actually_allowed = actually_allowed[0]  # first batch — we assume identical
            if not torch.equal(actually_allowed, causal_allowed):
                return False
            # Verify all other batches agree (cheaper than full equal across B)
            if m.shape[0] > 1:
                # batch-broadcast check: all batches have the same allowed
                allowed_per_batch = m == 0 if torch.is_floating_point(m) else (m if m.dtype == torch.bool else m != 0)
                if not torch.all(allowed_per_batch == allowed_per_batch[0:1]):
                    return False
            return True
        return False

    @staticmethod
    def _padding_mask_valid_lens(mask: torch.Tensor, K: int) -> torch.Tensor | None:
        """For a 2D padding mask of shape [B, K] where mask[b, k] is truthy iff
        position k is a valid (non-padded) key, return the per-batch valid
        length (the number of leading truthy positions). Returns None if the
        mask is not 2D, the wrong width, or has interspersed padding (non-
        contiguous truthy region).

        Used to derive cu_seqlen_ke for the tilelang indexer fwd kernel: the
        valid key range for query q is [0, min(q + query_offset + 1, valid_len)).
        """
        if mask.dim() != 2 or mask.shape[-1] != K:
            return None
        if mask.dtype == torch.bool:
            allowed = mask
        elif torch.is_floating_point(mask):
            allowed = mask > 0
        else:
            allowed = mask != 0
        # Count leading truthy positions per batch.
        valid_lens = allowed.long().sum(dim=-1)  # [B]
        # Reject masks that have interspersed padding: the valid region must
        # be exactly the first valid_lens[b] positions.
        # Cheap check: cumulative-truthy at position valid_lens[b]-1 == valid_lens[b].
        # Equivalent: allowed[b, :valid_lens[b]].all() and allowed[b, valid_lens[b]:].all_false()
        # Vectorize via arange comparison.
        pos = torch.arange(K, device=mask.device).unsqueeze(0)  # [1, K]
        expected = pos < valid_lens.unsqueeze(-1)  # [B, K]
        if not torch.equal(allowed, expected):
            return None
        return valid_lens

    def _try_tilelang_select_topk(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        head_weights: torch.Tensor,
        attention_mask: torch.Tensor | None,
        Q: int,
        K: int,
        H: int,
        D: int,
        *,
        query_offset: int = 0,
    ) -> torch.Tensor | None:
        """Tilelang fast path for select_topk. Returns None if the inputs don't
        meet the kernel's constraints (bf16/CUDA, pure-causal mask) so the
        caller falls back to the torch reference path.

        Handles both Q==K (no ulysses) and Q<K with query_offset (ulysses)."""
        # Kernel constraints
        if (
            index_q.device.type != "cuda"
            or index_q.dtype != torch.bfloat16
            or index_k.dtype != torch.bfloat16
            or Q > K  # local queries can't exceed full key range
        ):
            return None
        # Accept attention_mask==None OR a pure-causal mask OR a 2D padding mask
        # whose batches agree on a single contiguous valid prefix. Anything else
        # (packed multi-doc, interspersed padding) needs the torch path.
        per_batch_valid_lens = None
        if attention_mask is not None:
            if not self._is_pure_causal_mask(attention_mask, Q, K, query_offset):
                per_batch_valid_lens = self._padding_mask_valid_lens(attention_mask, K)
                if per_batch_valid_lens is None:
                    return None
                # Require all batches to have the same valid_len (tilelang kernel
                # has no batch dim — we'd iterate batches with the same cu_ke).
                if not torch.all(per_batch_valid_lens == per_batch_valid_lens[0]):
                    return None
        # Causal bounds: query at position (query_offset + q_local) attends to
        # keys in [0, query_offset + q_local + 1). For non-ulysses, query_offset
        # = 0; for ulysses, query_offset is the global q offset of the rank.
        # We must not exceed K (the actual key tensor length).
        if query_offset + Q > K:
            return None
        # The fast path materializes a full [Q, K] fp32 logits tensor. For very
        # long contexts, the caller may have opted into blocked scoring to keep
        # memory in budget — respect that opt-in.
        if (
            int(getattr(self.config, "indexer_score_query_block_size", 0) or 0) > 0
            or int(getattr(self.config, "indexer_score_key_block_size", 0) or 0) > 0
        ):
            return None

        try:
            from xorl.ops.glm5_kernels.tilelang_indexer_fwd import (  # noqa: PLC0415
                clean_logits_,
                tl_indexer_fwd_impl,
            )
        except Exception:
            return None

        B = index_q.shape[0]
        topk = min(self.index_topk, K)

        # The tilelang kernel takes [seq_len*heads, dim] q and [seq_len_kv, dim] k.
        # No native batch dim, so iterate over batch.
        outputs = []
        with torch.no_grad():
            try:
                # JIT-compile on first call; raises if the shape is too small
                # (tilelang's mma_sync requires K>=16 etc).
                fwd_kernel = tl_indexer_fwd_impl(
                    heads=H,
                    index_dim=D,
                    block_N=128,
                    threads=256,
                )
                clean_kernel = clean_logits_()
            except Exception:
                return None
            # cu_seqlen_ks=0 for all q (start from position 0).
            # cu_seqlen_ke=query_offset+q+1 for causal (each query attends up to
            # its global position).
            cu_ks = torch.zeros((Q,), device=index_q.device, dtype=torch.int32)
            cu_ke = torch.arange(query_offset + 1, query_offset + Q + 1, device=index_q.device, dtype=torch.int32)
            # If a uniform padding mask was provided, clip cu_ke to the
            # per-batch valid_len (we already verified all batches agree).
            if per_batch_valid_lens is not None:
                valid_len = per_batch_valid_lens[0].to(torch.int32)
                cu_ke = torch.minimum(cu_ke, valid_len.expand(Q))

            for b in range(B):
                iq_2d = index_q[b].reshape(Q * H, D).contiguous()
                ik_2d = index_k[b].contiguous()
                w_2d = head_weights[b].contiguous().to(torch.float32)
                logits = torch.empty((Q, K), device=index_q.device, dtype=torch.float32)
                try:
                    fwd_kernel(iq_2d, ik_2d, logits, w_2d, cu_ks, cu_ke)
                    clean_kernel(logits, cu_ks, cu_ke)
                except Exception:
                    return None
                # torch.topk on bf16 is ~1.6x faster than fp32 (DSv4 measured
                # 8.6 vs 13.7 ms at S=32k topk=512 on H100). Cast costs ~1.4 ms
                # so net win is ~3 ms per indexer call. Indices are dtype-
                # independent; sentinel detection via the (already-masked) bf16
                # -inf works identically.
                logits_for_topk = logits.bfloat16()
                _, indices = torch.topk(logits_for_topk, topk, dim=-1)
                indices = indices.to(torch.int32)
                sentinel_scores = torch.gather(logits_for_topk, -1, indices.long())
                indices = indices.masked_fill(sentinel_scores == float("-inf"), -1)
                outputs.append(indices)

            indices = torch.stack(outputs, dim=0)
            # Sort valid indices ascending; -1 sentinels at the end (see select_topk).
            _sentinel = torch.iinfo(torch.int32).max
            indices_for_sort = torch.where(indices == -1, _sentinel, indices)
            sorted_indices = indices_for_sort.sort(dim=-1).values
            return torch.where(
                sorted_indices == _sentinel,
                torch.tensor(-1, dtype=torch.int32, device=indices.device),
                sorted_indices,
            )

    def _build_allowed_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the boolean mask of keys eligible for DSA top-k selection."""
        causal = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
        allowed = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if attention_mask is None:
            return allowed

        if attention_mask.dim() == 4:
            mask = attention_mask[:, 0, :seq_len, :seq_len]
        elif attention_mask.dim() == 3:
            mask = attention_mask[:, :seq_len, :seq_len]
        elif attention_mask.dim() == 2:
            key_mask = attention_mask[:, :seq_len]
            if key_mask.dtype == torch.bool:
                key_allowed = key_mask
            elif torch.is_floating_point(key_mask):
                # 2D masks are usually 1/0 key-padding masks; additive
                # causal masks arrive as 4D from update_causal_mask.
                key_allowed = key_mask > 0
            else:
                key_allowed = key_mask != 0
            return allowed & key_allowed.to(device=device, dtype=torch.bool).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported attention_mask dim for GLM-5 DSA indexer: {attention_mask.dim()}")

        if mask.dtype == torch.bool:
            return allowed & mask.to(device=device, dtype=torch.bool)

        # Eager causal masks use 0 for allowed positions and finfo.min/-inf
        # for disallowed ones. Convert to bool before top-k so padded slots
        # become explicit -inf sentinels instead of large finite negatives.
        return allowed & (mask.to(device=device) == 0)

    def _build_allowed_mask_block(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        query_start: int,
        query_end: int,
        key_start: int,
        key_end: int,
        device: torch.device,
        *,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """Return the boolean DSA mask for a query/key score block."""
        q_pos = query_offset + torch.arange(query_start, query_end, device=device)
        k_pos = torch.arange(key_start, key_end, device=device)
        allowed = (k_pos.view(1, 1, -1) <= q_pos.view(1, -1, 1)).expand(batch_size, -1, -1).clone()
        if attention_mask is None:
            return allowed

        if attention_mask.dim() == 4:
            mask = attention_mask[:, 0]
            query_dim = mask.shape[1]
            if query_dim >= query_offset + query_end:
                mask = mask[:, query_offset + query_start : query_offset + query_end, key_start:key_end]
            elif query_dim >= query_end:
                mask = mask[:, query_start:query_end, key_start:key_end]
            else:
                raise ValueError(
                    f"GLM-5 DSA block attention_mask query dimension is too short: {query_dim} < query_end {query_end}."
                )
        elif attention_mask.dim() == 3:
            query_dim = attention_mask.shape[1]
            if query_dim >= query_offset + query_end:
                mask = attention_mask[:, query_offset + query_start : query_offset + query_end, key_start:key_end]
            elif query_dim >= query_end:
                mask = attention_mask[:, query_start:query_end, key_start:key_end]
            else:
                raise ValueError(
                    f"GLM-5 DSA block attention_mask query dimension is too short: {query_dim} < query_end {query_end}."
                )
        elif attention_mask.dim() == 2:
            key_mask = attention_mask[:, key_start:key_end]
            if key_mask.dtype == torch.bool:
                key_allowed = key_mask
            elif torch.is_floating_point(key_mask):
                key_allowed = key_mask > 0
            else:
                key_allowed = key_mask != 0
            return allowed & key_allowed.to(device=device, dtype=torch.bool).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported attention_mask dim for GLM-5 DSA indexer: {attention_mask.dim()}")

        if mask.dtype == torch.bool:
            return allowed & mask.to(device=device, dtype=torch.bool)
        return allowed & (mask.to(device=device) == 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        sampler_prefill_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-shot indexer: project + score + top-k.

        Returns ``[B, S, index_topk]`` — top-k key positions per query, ready
        to feed into :func:`sparse_mla_torch_reference`. The dense torch path
        is used for V1; the tilelang `lighting_indexer` lands later.
        """
        # Hard top-k indices are not differentiable. Avoid building a useless
        # autograd graph through the quadratic indexer score path.
        with torch.no_grad():
            index_q, index_k, head_weights = self.project(
                hidden_states,
                q_compressed,
                position_embeddings,
                sampler_prefill_lengths=sampler_prefill_lengths,
            )
            return self.select_topk(index_q, index_k, head_weights, attention_mask)

    def select_topk(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        head_weights: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """Dense torch reference for top-k selection.

        Returns indices of shape ``[B, S, index_topk]`` (clamped to seq_len).
        Used to validate the indexer's gradients before the sparse-MLA kernel
        is wired in. Quadratic in seq_len, so suitable for tests / smoke runs
        only.
        """
        B, Q, H, D = index_q.shape
        K = index_k.shape[1]

        if self.selector_version == GLM52_SELECTOR_VERSION:
            allowed = self._build_allowed_mask_block(
                attention_mask,
                B,
                0,
                Q,
                0,
                K,
                index_q.device,
                query_offset=query_offset,
            )
            return select_glm52_logical_indices(
                index_q,
                index_k,
                head_weights,
                allowed,
                topk=self.index_topk,
                apply_hadamard=False,
            ).logical_indices.values

        # Tilelang fast path — handles BOTH Q==K (no ulysses) and Q<K with
        # query_offset (ulysses) for pure-causal attention. ~14x speedup vs
        # the torch chunked-einsum / blocked-scoring paths at GLM-5.1
        # production shape on H100 (31.8 ms → 2.2 ms).
        tilelang_indices = self._try_tilelang_select_topk(
            index_q,
            index_k,
            head_weights,
            attention_mask,
            Q,
            K,
            H,
            D,
            query_offset=query_offset,
        )
        if tilelang_indices is not None:
            return tilelang_indices

        if (
            Q != K
            or query_offset != 0
            or int(getattr(self.config, "indexer_score_query_block_size", 0) or 0) > 0
            or int(getattr(self.config, "indexer_score_key_block_size", 0) or 0) > 0
        ):
            return self.select_topk_blocked(
                index_q,
                index_k,
                head_weights,
                attention_mask,
                query_offset=query_offset,
            )

        # Dot-product per (q, k) per head, then follow GLM's DSA scoring:
        # ReLU the per-head scores, apply learned per-head weights, and sum.
        # The naive logits path materializes [B, S_q, H, S_k], which is 2 GiB
        # for GLM-5.1 at S=4096/H=32 in fp32. Chunk over heads and accumulate
        # [B, S_q, S_k] logits instead.
        with torch.no_grad():
            index_k = index_k.float()
            head_weights = head_weights.float()
            logits = torch.zeros((B, Q, K), device=index_q.device, dtype=torch.float32)
            chunk_heads = max(1, int(getattr(self.config, "indexer_score_chunk_heads", 4)))
            chunk_heads = min(chunk_heads, H)
            for start in range(0, H, chunk_heads):
                end = min(start + chunk_heads, H)
                scores = torch.einsum("bshd,btd->bsht", index_q[:, :, start:end, :].float(), index_k)
                scores.mul_(self._softmax_scale).relu_()
                logits.add_(torch.einsum("bsht,bsh->bst", scores, head_weights[:, :, start:end]))

            allowed = self._build_allowed_mask(attention_mask, B, Q, logits.device)
            logits = logits.masked_fill(~allowed, float("-inf"))

            topk = min(self.index_topk, K)
            scores, indices = torch.topk(logits, topk, dim=-1)
            # Sentinel-mark slots whose source logit was -inf (non-causal pad)
            # as -1, matching miles' lighting_indexer convention. The tilelang
            # sparse-MLA kernel checks `Indices != -1` to mask those slots out
            # in both the forward score and the backward gradient — without
            # this, backward leaks through non-causal positions and produces
            # NaN gradients.
            indices = indices.to(torch.int32).masked_fill(scores == float("-inf"), -1)
            # Sort valid indices ascending by kv position; keep -1 sentinels at
            # the end so callers that expect "valid indices first, sentinels
            # last" continue to work. Softmax is order-invariant, but the
            # sparse-MLA kernel gathers KV[indices[bi]] in BS=64 blocks —
            # sorted indices give near-contiguous L2 access, ~5 ms faster bwd
            # at GLM-5 production shape on H100 (random pattern: 46→41 ms).
            _sentinel = torch.iinfo(torch.int32).max
            indices_for_sort = torch.where(indices == -1, _sentinel, indices)
            sorted_indices = indices_for_sort.sort(dim=-1).values
            return torch.where(
                sorted_indices == _sentinel, torch.tensor(-1, dtype=torch.int32, device=indices.device), sorted_indices
            )

    def select_topk_blocked(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        head_weights: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """Memory-bounded top-k selection for long-context DSA.

        Scores are identical to :meth:`select_topk`, but query/key blocks are
        streamed so long-context runs do not materialize ``[B, S, S]`` logits.
        ``query_offset`` lets a Ulysses rank score its local query shard
        against the full key sequence while preserving causal masking.
        """
        B, Q, H, _ = index_q.shape
        K = index_k.shape[1]
        topk = min(self.index_topk, K)
        if topk == 0:
            return torch.empty((B, Q, 0), device=index_q.device, dtype=torch.int32)

        query_block_size = int(getattr(self.config, "indexer_score_query_block_size", 512) or 512)
        key_block_size = int(getattr(self.config, "indexer_score_key_block_size", 8192) or 8192)
        chunk_heads = max(1, int(getattr(self.config, "indexer_score_chunk_heads", 4)))
        query_block_size = max(1, min(query_block_size, Q))
        key_block_size = max(1, min(key_block_size, K))
        chunk_heads = min(chunk_heads, H)

        index_k = index_k.float()
        head_weights = head_weights.float()
        blocks: list[torch.Tensor] = []

        with torch.no_grad():
            for q_start in range(0, Q, query_block_size):
                q_end = min(q_start + query_block_size, Q)
                q_block = index_q[:, q_start:q_end]
                weight_block = head_weights[:, q_start:q_end]
                q_len = q_end - q_start
                best_scores = torch.full((B, q_len, topk), float("-inf"), device=index_q.device)
                best_indices = torch.full((B, q_len, topk), -1, device=index_q.device, dtype=torch.int32)

                for k_start in range(0, K, key_block_size):
                    k_end = min(k_start + key_block_size, K)
                    logits = torch.zeros((B, q_len, k_end - k_start), device=index_q.device, dtype=torch.float32)
                    k_block = index_k[:, k_start:k_end]
                    for h_start in range(0, H, chunk_heads):
                        h_end = min(h_start + chunk_heads, H)
                        scores = torch.einsum("bqhd,bkd->bqhk", q_block[:, :, h_start:h_end, :].float(), k_block)
                        scores.mul_(self._softmax_scale).relu_()
                        logits.add_(torch.einsum("bqhk,bqh->bqk", scores, weight_block[:, :, h_start:h_end]))

                    allowed = self._build_allowed_mask_block(
                        attention_mask,
                        B,
                        q_start,
                        q_end,
                        k_start,
                        k_end,
                        logits.device,
                        query_offset=query_offset,
                    )
                    logits.masked_fill_(~allowed, float("-inf"))

                    block_topk = min(topk, k_end - k_start)
                    block_scores, block_indices = torch.topk(logits, block_topk, dim=-1)
                    block_indices = (block_indices + k_start).to(torch.int32)

                    merged_scores = torch.cat([best_scores, block_scores], dim=-1)
                    merged_indices = torch.cat([best_indices, block_indices], dim=-1)
                    best_scores, positions = torch.topk(merged_scores, topk, dim=-1)
                    best_indices = torch.gather(merged_indices, dim=-1, index=positions)

                blocks.append(best_indices.masked_fill(best_scores == float("-inf"), -1))

        # Sort valid indices ascending; keep -1 sentinels at the end (see select_topk).
        indices = torch.cat(blocks, dim=1)
        _sentinel = torch.iinfo(torch.int32).max
        indices_for_sort = torch.where(indices == -1, _sentinel, indices)
        sorted_indices = indices_for_sort.sort(dim=-1).values
        return torch.where(
            sorted_indices == _sentinel,
            torch.tensor(-1, dtype=torch.int32, device=indices.device),
            sorted_indices,
        )


__all__ = ["Glm5DsaIndexer"]
