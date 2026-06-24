"""GLM-5 sparse-MLA (absorb form): dispatch + torch reference.

DeepSeek Sparse Attention runs MLA in the *compressed* KV space: rather
than materializing per-head K and V via `kv_b_proj`, the no-pe portion of Q
absorbs `W_kc` so attention scores are computed against `kv_compressed`
directly, and the no-pe output absorbs `W_vc` after attention. The indexer's
top-k indices select a subset of past tokens per query.

Two backends share the call site:

- **torch** — quadratic dense gather + softmax. CPU/CI-safe and used for
  tests; keeps the rest of the model unblocked when the tilelang kernel
  isn't available.
- **tilelang** — vendored from miles `glm5/ops/sparse_mla.py`. CUDA-only,
  bf16-only, requires ``dim_plus_tail_dim == 576`` (kv_lora=512 +
  qk_rope=64) and ``topk % 64 == 0`` — both hold for GLM-5 / GLM-5.1.

`sparse_mla_dispatch` picks tilelang when available + on CUDA + the inputs
match the kernel's constraints, else torch. The interface (and numerical
output up to softmax precision) is the same in both.
"""

import logging

import torch


logger = logging.getLogger(__name__)


_TILELANG_IMPORT_ATTEMPTED = False
_TILELANG_AVAILABLE = False


def _is_tilelang_available() -> bool:
    """Lazy probe for tilelang. Result is cached per-process."""
    global _TILELANG_IMPORT_ATTEMPTED, _TILELANG_AVAILABLE
    if not _TILELANG_IMPORT_ATTEMPTED:
        _TILELANG_IMPORT_ATTEMPTED = True
        try:
            import tilelang  # noqa: F401, PLC0415

            _TILELANG_AVAILABLE = True
        except Exception as e:  # pragma: no cover — env-dependent
            logger.info("tilelang not available, sparse-MLA will use the torch reference: %s", e)
            _TILELANG_AVAILABLE = False
    return _TILELANG_AVAILABLE


def sparse_mla_torch_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
    kv_lora_rank: int,
    query_offset: int = 0,
) -> torch.Tensor:
    """Compute sparse MLA attention in the compressed KV space.

    Args:
        q: ``[B, S, H, kv_lora + qk_rope]`` — absorbed query (no-pe absorbed
            into ``kv_lora`` dims, pe band trailing).
        kv: ``[B, S, kv_lora + qk_rope]`` — compressed KV with the rope band
            trailing. Single shared kv-group (MLA semantics).
        indices: ``[B, S, topk]`` — top-k key positions per query, from the
            DSA indexer. Values are in ``[0, S)`` or ``-1`` for masked slots.
        scaling: softmax scale, normally ``qk_head_dim ** -0.5``.
        kv_lora_rank: width of the leading kv_lora band in ``kv`` and ``q``.
            The trailing ``qk_rope_head_dim`` band participates in scoring
            but not in the value mixing (matches the miles kernel).
        query_offset: global sequence offset for ``q`` when ``q`` is a local
            query shard but ``kv`` and ``indices`` use full-sequence positions.

    Returns:
        ``[B, S, H, kv_lora]`` — attention output in the compressed space.
        Caller absorbs ``W_vc`` to recover ``[B, S, H, v_head_dim]``.

    Quadratic in ``S`` because of the dense gather; suitable for tests and
    smoke runs. The tilelang kernel is what we ship to GPUs.
    """
    B, S, H, D = q.shape

    # The indexer marks non-causal pad slots with -1. Clamp to 0 so the
    # gather doesn't index out-of-bounds, and remember which slots to mask
    # later so they don't contribute to softmax.
    invalid = indices < 0
    safe_indices = indices.clamp(min=0)

    # Gather top-k kv per query: [B, S, topk, D]. We expand kv along the
    # query axis first (cheap broadcast) so torch.gather can pick per-query
    # subsets in one shot.
    idx_expanded = safe_indices.long().unsqueeze(-1).expand(-1, -1, -1, D)
    kv_expanded = kv.unsqueeze(1).expand(-1, S, -1, -1)
    kv_topk = torch.gather(kv_expanded, dim=2, index=idx_expanded)

    # Scores use the full D (kv_lora absorbed-nope + qk_rope rotated band).
    scores = torch.einsum("bshd,bskd->bshk", q, kv_topk) * scaling

    # Mask non-causal slots: any slot the indexer marked as -1 (its
    # indexer logit was -inf) plus any slot whose index points strictly
    # past the query position. Re-imposing causality here makes the
    # sparse path's output independent of the indexer's padding choices,
    # and pins sparse(topk>=S) to match dense.
    q_pos = query_offset + torch.arange(S, device=q.device).view(1, S, 1)
    non_causal = invalid | (safe_indices > q_pos)
    scores = scores.masked_fill(non_causal.unsqueeze(-2), float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    # Values use only the kv_lora band — the rope band carries position only.
    v_topk = kv_topk[..., :kv_lora_rank]
    return torch.einsum("bshk,bskd->bshd", weights, v_topk)


def _tilelang_constraints_satisfied(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> bool:
    """The vendored kernel was built with these specific shapes baked in."""
    if q.device.type != "cuda" or kv.device.type != "cuda":
        return False
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return False
    dim_plus_tail = q.shape[-1]
    topk = indices.shape[-1]
    return dim_plus_tail == 576 and topk % 64 == 0


def _sparse_mla_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
    kv_lora_rank: int,
    query_offset: int = 0,
) -> torch.Tensor:
    """Tilelang sparse-MLA wrapper.

    Reshapes our 4D ``[B, S_q, H, D]`` and ``[B, S_kv, D]`` layouts into
    miles' 3D ``[B*S_q, H, D]`` and ``[B*S_kv, 1, D]`` forms before calling
    `SparseMLA.apply`, then reshapes back.
    """
    from xorl.ops.glm5_kernels.sparse_mla import SparseMLA  # noqa: PLC0415  (lazy: tilelang is optional)

    del query_offset
    B, S_q, H, D = q.shape
    if kv.shape[0] != B or kv.shape[-1] != D:
        raise ValueError(f"kv shape {tuple(kv.shape)} is incompatible with q shape {tuple(q.shape)}")
    if indices.shape[:2] != (B, S_q):
        raise ValueError(f"indices shape {tuple(indices.shape)} is incompatible with q shape {tuple(q.shape)}")

    S_kv = kv.shape[1]
    q_flat = q.reshape(B * S_q, H, D)
    kv_flat = kv.reshape(B * S_kv, 1, D)
    indices_flat = indices.reshape(B * S_q, 1, indices.shape[-1]).to(torch.int32)
    if B > 1:
        offsets = (torch.arange(B, device=indices.device, dtype=indices_flat.dtype) * S_kv).view(B, 1, 1)
        indices_flat = torch.where(
            indices >= 0, indices.to(indices_flat.dtype) + offsets, indices.to(indices_flat.dtype)
        )
        indices_flat = indices_flat.reshape(B * S_q, 1, indices.shape[-1])

    out_flat, _ = SparseMLA.apply(q_flat, kv_flat, indices_flat, scaling)
    # `out_flat` shape is [B*S_q, H, kv_lora_rank].
    return out_flat.view(B, S_q, H, kv_lora_rank)


def sparse_mla_dispatch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
    kv_lora_rank: int,
    backend: str = "auto",
    query_offset: int = 0,
) -> torch.Tensor:
    """Run sparse-MLA via the best available backend.

    ``backend`` selects the path:
    - ``"auto"`` — tilelang if available + on CUDA + shape constraints met,
      else torch reference.
    - ``"torch"`` — always torch reference.
    - ``"tilelang"`` — always tilelang; raises if unavailable.
    """
    if backend == "torch":
        return sparse_mla_torch_reference(q, kv, indices, scaling, kv_lora_rank, query_offset=query_offset)

    if backend == "tilelang":
        if not _is_tilelang_available():
            raise RuntimeError("backend='tilelang' requested but tilelang is not installed")
        if not _tilelang_constraints_satisfied(q, kv, indices):
            raise RuntimeError(
                "backend='tilelang' requested but GLM-5 sparse-MLA kernel constraints are not met "
                "(requires CUDA bf16 tensors, dim_plus_tail_dim == 576, and topk % 64 == 0)"
            )
        return _sparse_mla_tilelang(q, kv, indices, scaling, kv_lora_rank, query_offset=query_offset)

    if backend != "auto":
        raise ValueError(f"unknown sparse-MLA backend: {backend!r}")

    if _is_tilelang_available() and _tilelang_constraints_satisfied(q, kv, indices):
        return _sparse_mla_tilelang(q, kv, indices, scaling, kv_lora_rank, query_offset=query_offset)
    return sparse_mla_torch_reference(q, kv, indices, scaling, kv_lora_rank, query_offset=query_offset)


__all__ = ["sparse_mla_dispatch", "sparse_mla_torch_reference"]
