# Families v2 — the redefined frozen reduction trees for the batch-invariance
# contract (hidden-dim RMSNorm, qk-norm, and the final projection).
#
# The trainer and serving engine carry equivalent implementations because they
# must evaluate the same reduction trees. This module remains self-contained
# (torch + triton only), and cross-engine agreement is established by the
# behavior-logprob replay rather than source-file identity.
#
# Rule A: every bit-relevant reduction is written explicitly — an
# adjacent-pairwise balanced binary tree within a block (tl.split + one add
# per level; a 2-element reduction has exactly one association, so the
# compiler owns no tree choice) and a sequential scalar chain across chunks
# in index order. tl.sum over >2 elements is banned in bit-relevant positions.
# Rule B: golden-value gates pin the bits under both engines' venvs.
#
# These trees are DEFAULT ON inside an engaged contract lane. The exact
# Qwen3.5-family model setup selects the already-certified v1 family directly;
# callers do not coordinate the selection with an environment variable.
#
# Migration: v1 and v2 are two different trees, and both hold the trainer and
# the sampler bitwise equal — that is the contract, and v2 satisfies it. The
# reported values do move between the two, so goldens and frozen anchors
# recorded under v1 must be re-taken under v2, and both engines must flip
# together.

import os

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources


_EXACT_FAMILIES_VERSION: str | None = None


def _select_qwen35_families_v1() -> None:
    """Pin exact Qwen's fused LM-head loss to its qualified v1 program."""
    global _EXACT_FAMILIES_VERSION
    _EXACT_FAMILIES_VERSION = "v1"


def _select_glm52_families_v2() -> None:
    """Pin the GLM-5.2 process to its certified v2 reduction trees."""
    global _EXACT_FAMILIES_VERSION
    _EXACT_FAMILIES_VERSION = "v2"


def _select_qwen3_dense_families_v2() -> None:
    """Pin exact dense Qwen3 to the shared v2 norm and lm-head trees."""
    global _EXACT_FAMILIES_VERSION
    _EXACT_FAMILIES_VERSION = "v2"


def _select_nonexact_families() -> None:
    """Restore the pre-existing family selection for an ordinary model."""
    global _EXACT_FAMILIES_VERSION
    _EXACT_FAMILIES_VERSION = None


def families_v2_enabled() -> bool:
    """Return the selected reduction family for the current process.

    Exact model programs select their family structurally and ignore the
    legacy rollback variables.  Without an exact model selection, preserve the
    pre-existing non-exact behavior for compatibility.
    """
    if _EXACT_FAMILIES_VERSION is not None:
        return _EXACT_FAMILIES_VERSION == "v2"
    return not any(os.getenv(v, "1").lower() in _V2_OFF for v in FAMILIES_V2_ENV_VARS)


# Contract constants (bit-relevant; never tuning axes).
V2_NORM_BLOCK_H = 4096  # per-chunk tree width for hidden-dim norms
V2_QK_MAX_HEAD_DIM = 256

FAMILIES_V2_ENV_VARS = ("XORL_FAMILIES_V2", "SGLANG_FAMILIES_V2")
_V2_OFF = ("0", "false", "no")


@triton.jit
def _rtne_bf16(x):
    # Round-to-nearest-even to bf16 via integer bitcast (triton folds
    # f32->bf16->f32 cast pairs; this cannot be folded). NaNs quieted.
    bits = x.to(tl.int32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    nan_bits = (bits & -65536) | 0x00400000
    out = tl.where(x != x, nan_bits, rounded)
    return out.to(tl.float32, bitcast=True)


@triton.jit
def _pairwise_tree_sum(vec, BLOCK: tl.constexpr):
    # Adjacent-pairwise balanced binary tree: log2(BLOCK) levels, one explicit
    # fp32 add per level. The reduction ORDER is written in the IR — no
    # compiler-owned tl.sum lowering (the v1 1-ulp drift class). Statically
    # unrolled; the constexpr `if` prunes levels past log2(BLOCK).
    tl.static_assert(BLOCK & (BLOCK - 1) == 0, "BLOCK must be a power of 2")
    tl.static_assert(BLOCK <= 4096, "12 unrolled levels cover BLOCK <= 4096")
    for _ in tl.static_range(0, 12):
        if vec.shape[0] > 1:
            lo, hi = tl.split(tl.reshape(vec, (vec.shape[0] // 2, 2)))
            vec = lo + hi
    return vec  # shape (1,) fp32


V2_NORM_TILE = 512  # register tile; BITS-NEUTRAL by the tree factorization below


@triton.jit
def _rms_norm_v2_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    n_cols,
    stride_x,
    stride_res,
    stride_out,
    stride_res_out,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Family-2'/1' unified row norm: ONE launch per call site.

    Pinned order: (residual add -> RTNE bf16 round ->) fp32 square ->
    per-chunk pairwise tree -> sequential chunk chain -> /H -> tl.rsqrt ->
    x * inv_rms * w (left-to-right, fp32 weight-mul) -> single cast at store.
    Tree is a function of n_cols alone => batch-invariant by construction.

    For the residual specialization, this fused one-launch form serves nearly
    every shipped shape. Few rows over many tiles, and the compiler-spilling
    no-residual specialization, dispatch to the bit-identical split realization
    below (see ``_v2_norm_use_split``), which factorizes the same tree into
    per-TILE trees + a partials tree (adjacent pairing preserves contiguity at
    every level => identical bits; cross-structure gates + frozen goldens).
    """
    row = tl.program_id(0)
    n_chunks = tl.cdiv(n_cols, BLOCK_H)

    total = tl.zeros((1,), dtype=tl.float32)
    for c in range(n_chunks):
        cols = c * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
            s = _rtne_bf16(x + r)  # v1 semantic kept: add rounds to input dtype
            tl.store(
                res_out_ptr + row * stride_res_out + cols,
                s.to(res_out_ptr.dtype.element_ty),
                mask=mask,
            )
        else:
            s = x
        sq = s * s  # masked lanes contribute +0.0 (exact for sums of squares)
        total = total + _pairwise_tree_sum(sq, BLOCK_H)

    var = total / n_cols.to(tl.float32)
    inv_rms = tl.rsqrt(var + eps)

    for c in range(n_chunks):
        cols = c * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < n_cols
        if HAS_RESIDUAL:
            s = tl.load(res_out_ptr + row * stride_res_out + cols, mask=mask, other=0.0).to(
                tl.float32
            )  # the rounded sum: bit-identical to what was squared
        else:
            s = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            w = 1.0 + w
        y = s * inv_rms * w
        tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


QK_V2_ROWS_PER_PROG = 16  # head-rows per program (perf-only, NOT bit-relevant)


@triton.jit
def _qk_norm_v2_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    n_heads,
    head_dim,
    stride_x_tok,
    stride_x_head,
    stride_out_tok,
    stride_out_head,
    eps,
    ZERO_CENTERED: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Family-1' strided qk-norm: reads head rows straight out of the packed
    qkv projection (no reshape/contiguous copies). Same tree as
    _rms_norm_v2_kernel (single chunk: head_dim <= V2_QK_MAX_HEAD_DIM).
    Each program handles ROWS independent head-rows (row batching is grid
    shape only — per-row math identical, like BLOCK_M in the GEMM).
    In-place safe per row: the full head row is loaded before any store.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_mask = rows < n_rows
    rows_safe = tl.where(row_mask, rows, 0)
    tok = rows_safe // n_heads
    head = rows_safe % n_heads
    d = tl.arange(0, BLOCK_D)
    col_mask = d < head_dim
    mask = row_mask[:, None] & col_mask[None, :]
    base = tok * stride_x_tok + head * stride_x_head
    x = tl.load(x_ptr + base[:, None] + d[None, :], mask=mask, other=0.0).to(tl.float32)
    total = _pairwise_tree_sum_rows(x * x, BLOCK_D)
    var = total / head_dim.to(tl.float32)
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + d, mask=col_mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        w = 1.0 + w
    y = x * inv_rms[:, None] * w[None, :]
    out_base = tok * stride_out_tok + head * stride_out_head
    tl.store(
        out_ptr + out_base[:, None] + d[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def rms_norm_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    residual: torch.Tensor | None = None,
    zero_centered: bool = False,
):
    """v2 hidden-dim RMSNorm (family-2' residual / no-residual, family-1'
    site-classes unified on one tree). Returns ``out`` or ``(out, residual_out)``.

    bf16-only (the contract dtype); 2D ``[rows, H]`` input, unit column stride.
    """
    assert x.ndim == 2 and x.stride(1) == 1, "x must be [rows, H], unit col stride"
    assert x.dtype == torch.bfloat16, "families v2 is bf16-only (contract dtype)"
    assert weight.ndim == 1 and weight.shape[0] == x.shape[1]
    assert x.is_cuda
    rows, H = x.shape
    weight = weight.contiguous()
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(1) == 1
        assert residual.dtype == torch.bfloat16
    if _v2_norm_use_split(
        rows,
        triton.cdiv(H, V2_NORM_TILE),
        has_residual=residual is not None,
        is_hopper=torch.cuda.get_device_capability(x.device) == (9, 0),
    ):
        return _rms_norm_v2_split(x, weight, eps, residual, zero_centered)
    return _rms_norm_v2_fused(x, weight, eps, residual, zero_centered)


def _rms_norm_v2_fused(x, weight, eps, residual, zero_centered):
    """One-launch realization of the v2 norm tree (see ``_rms_norm_v2_kernel``)."""
    rows, H = x.shape
    out = torch.empty_like(x)
    if residual is not None:
        res_out = torch.empty_like(x)
        _rms_norm_v2_kernel[(rows,)](
            x,
            residual,
            weight,
            out,
            res_out,
            H,
            x.stride(0),
            residual.stride(0),
            out.stride(0),
            res_out.stride(0),
            eps,
            HAS_RESIDUAL=True,
            ZERO_CENTERED=zero_centered,
            BLOCK_H=V2_NORM_BLOCK_H,
        )
        return out, res_out
    _rms_norm_v2_kernel[(rows,)](
        x,
        x,
        weight,
        out,
        out,
        H,
        x.stride(0),
        0,
        out.stride(0),
        0,
        eps,
        HAS_RESIDUAL=False,
        ZERO_CENTERED=zero_centered,
        BLOCK_H=V2_NORM_BLOCK_H,
    )
    return out


def qk_norm_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    head_dim: int,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
):
    """v2 per-head qk-norm over strided input.

    ``x``: ``[T, n_heads * head_dim]`` view into the packed qkv output — any
    row stride, unit element stride, heads contiguous within a row. ``out``
    defaults to a fresh tensor (trainer); pass ``out=x`` for in-place (serving).
    """
    assert x.ndim == 2 and x.stride(1) == 1
    assert x.dtype == torch.bfloat16, "families v2 is bf16-only (contract dtype)"
    assert x.shape[1] % head_dim == 0
    assert head_dim <= V2_QK_MAX_HEAD_DIM
    assert weight.ndim == 1 and weight.shape[0] == head_dim
    assert x.is_cuda
    T = x.shape[0]
    n_heads = x.shape[1] // head_dim
    weight = weight.contiguous()
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape and out.stride(1) == 1 and out.dtype == x.dtype
    if T > 0:
        n_rows = T * n_heads
        _qk_norm_v2_kernel[(triton.cdiv(n_rows, QK_V2_ROWS_PER_PROG),)](
            x,
            weight,
            out,
            n_rows,
            n_heads,
            head_dim,
            x.stride(0),
            head_dim,
            out.stride(0),
            head_dim,
            eps,
            ZERO_CENTERED=zero_centered,
            BLOCK_D=triton.next_power_of_2(head_dim),
            ROWS=QK_V2_ROWS_PER_PROG,
        )
    return out


# ---------------------------------------------------------------------------
# Head v2 — online-LSE lm-head (component 3, design note §5)
#
# The GEMM K-chain is v1's matmul_kernel_persistent VERBATIM (pinned
# BLOCK_SIZE_K; logits bitwise == the v1 head GEMM — gated). New: per-tile
# (m, l) stats computed in the epilogue from the fp32 accumulator, one pinned
# merge kernel. BLOCK_SIZE_N tiles a stats reduction, so it is a CONTRACT
# CONSTANT here (HEAD_STATS_TILE_N) — never shape/M-keyed, or decode and
# trainer scoring rows would grow different LSE trees. The OOM fallback chain
# varies only bit-neutral axes (BLOCK_M / GROUP / stages / warps).
# ---------------------------------------------------------------------------

HEAD_V2_STATS_TILE_N = 256  # contract constant (bit-relevant)
HEAD_V2_BLOCK_K = 64  # == v1 PINNED_BLOCK_K[bf16] (bit-relevant)

# Bit-neutral launch axes, M-bucketed (perf only; all candidates share N/K).
_HEAD_V2_LAUNCH = (
    (16, {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 4}),
    (256, {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 4}),
    (None, {"BLOCK_SIZE_M": 128, "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8}),
)
_HEAD_V2_FALLBACK = {
    "BLOCK_SIZE_M": 64,
    "GROUP_SIZE_M": 8,
    "num_stages": 2,
    "num_warps": 4,
}


@triton.jit
def _pairwise_tree_sum_rows(mat, BLOCK_N: tl.constexpr):
    # Row-wise adjacent-pairwise tree over axis 1 (same construction as
    # _pairwise_tree_sum; each level is one elementwise add — no compiler
    # tree choice, warp count is bit-neutral by construction).
    tl.static_assert(BLOCK_N & (BLOCK_N - 1) == 0, "BLOCK_N must be a power of 2")
    tl.static_assert(BLOCK_N <= 4096, "12 unrolled levels cover BLOCK_N <= 4096")
    for _ in tl.static_range(0, 12):
        if mat.shape[1] > 1:
            lo, hi = tl.split(tl.reshape(mat, (mat.shape[0], mat.shape[1] // 2, 2)))
            mat = lo + hi
    return tl.reshape(mat, (mat.shape[0],))


@triton.jit
def _head_v2_gemm_stats_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_out_ptr,
    l_out_ptr,
    sel_ptr,
    tok_ptr,
    temp_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_mm,
    stride_lm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    STORE_LOGITS: tl.constexpr,
    HAS_TOKS: tl.constexpr,
    HAS_TEMP: tl.constexpr,
):
    """v1 persistent GEMM (identical K-chain) + per-tile (m, l) epilogue stats.

    Stats are computed from the fp32 accumulator in registers: m = row max of
    the tile (order-free), l = row sum of exp(x - m) via the pairwise tree.
    Temperature scales stats (and the gathered selected logit) but NOT the
    stored logits — matching v1, where the stats kernel applies 1/T while the
    decode logits buffer stays unscaled for the sampler.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        row_mask = offs_cm < M
        col_mask = offs_cn < N

        if STORE_LOGITS:
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            tl.store(
                c_ptrs,
                accumulator.to(tl.float32),
                mask=row_mask[:, None] & col_mask[None, :],
            )

        # --- epilogue stats (the new frozen tree) ---
        x_stats = accumulator
        if HAS_TEMP:
            temp = tl.load(temp_ptr + offs_cm, mask=row_mask, other=1.0)
            inv_t = 1.0 / temp
            x_stats = x_stats * inv_t[:, None]
        neg_inf = float("-inf")
        m_tile = tl.max(tl.where(col_mask[None, :], x_stats, neg_inf), axis=1)
        terms = tl.where(col_mask[None, :], tl.exp(x_stats - m_tile[:, None]), 0.0)
        l_tile = _pairwise_tree_sum_rows(terms, BLOCK_SIZE_N)
        tl.store(m_out_ptr + offs_cm * stride_mm + pid_n, m_tile, mask=row_mask)
        tl.store(l_out_ptr + offs_cm * stride_lm + pid_n, l_tile, mask=row_mask)

        if HAS_TOKS:
            toks = tl.load(tok_ptr + offs_cm, mask=row_mask, other=-1)
            tok_here = (toks >= start_n) & (toks < start_n + BLOCK_SIZE_N) & row_mask
            # -inf-padded max gather: exactly one lane matches, sign of zero preserved
            sel = tl.max(tl.where(offs_cn[None, :] == toks[:, None], x_stats, neg_inf), axis=1)
            tl.store(sel_ptr + offs_cm, sel, mask=tok_here)


@triton.jit
def _head_v2_lse_merge_kernel(
    m_ptr,
    l_ptr,
    sel_ptr,
    lse_ptr,
    lp_ptr,
    n_tiles,
    stride_mm,
    stride_lm,
    HAS_SEL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Pinned merge: exact global max over tile maxima (order-free), rescaled
    sumexp via the pairwise tree over tiles (padded l=0 / m=-inf), then
    lse = gmax + log(acc); logprob = min(sel - lse, 0) (the p~1 boundary
    clamp, kept from v1)."""
    row = tl.program_id(0)
    t = tl.arange(0, BLOCK_T)
    mask = t < n_tiles
    neg_inf = float("-inf")
    m_t = tl.load(m_ptr + row * stride_mm + t, mask=mask, other=neg_inf)
    l_t = tl.load(l_ptr + row * stride_lm + t, mask=mask, other=0.0)
    gmax = tl.max(m_t)
    terms = tl.where(mask, l_t * tl.exp(m_t - gmax), 0.0)
    acc = _pairwise_tree_sum(terms, BLOCK_T)
    lse = gmax + tl.log(acc)
    tl.store(lse_ptr + row + tl.arange(0, 1), lse)
    if HAS_SEL:
        sel = tl.load(sel_ptr + row)
        lp = tl.minimum(sel - lse, 0.0)
        tl.store(lp_ptr + row + tl.arange(0, 1), lp)


def _head_v2_launch(hidden, weight_t, logits, m_buf, l_buf, sel, toks, temp):
    M, K = hidden.shape
    N = weight_t.shape[1]
    NUM_SMS = torch.cuda.get_device_properties(hidden.device).multi_processor_count
    a_large = hidden.numel() > 2**31
    b_large = weight_t.numel() > 2**31
    c_large = logits is not None and (logits.shape[0] * logits.stride(0) + logits.shape[1] * logits.stride(1) > 2**31)
    launch = _HEAD_V2_FALLBACK
    for bound, cfg in _HEAD_V2_LAUNCH:
        if bound is None or M <= bound:
            launch = cfg
            break

    def _run(cfg):
        grid = min(
            NUM_SMS,
            triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, HEAD_V2_STATS_TILE_N),
        )
        _head_v2_gemm_stats_kernel[(grid,)](
            hidden,
            weight_t,
            logits if logits is not None else hidden,
            m_buf,
            l_buf,
            sel if sel is not None else m_buf,
            toks if toks is not None else m_buf,
            temp if temp is not None else m_buf,
            M,
            N,
            K,
            hidden.stride(0),
            hidden.stride(1),
            weight_t.stride(0),
            weight_t.stride(1),
            logits.stride(0) if logits is not None else 0,
            logits.stride(1) if logits is not None else 0,
            m_buf.stride(0),
            l_buf.stride(0),
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=HEAD_V2_STATS_TILE_N,
            BLOCK_SIZE_K=HEAD_V2_BLOCK_K,
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            NUM_SMS=NUM_SMS,
            A_LARGE=a_large,
            B_LARGE=b_large,
            C_LARGE=c_large,
            STORE_LOGITS=logits is not None,
            HAS_TOKS=toks is not None,
            HAS_TEMP=temp is not None,
            num_stages=cfg["num_stages"],
            num_warps=cfg["num_warps"],
        )

    try:
        _run(launch)
    except OutOfResources:
        _run(_HEAD_V2_FALLBACK)  # bit-identical: only bit-neutral axes differ


def _head_v2_merge(m_buf, l_buf, sel):
    M, n_tiles = m_buf.shape
    lse = torch.empty((M,), dtype=torch.float32, device=m_buf.device)
    lp = torch.empty((M,), dtype=torch.float32, device=m_buf.device) if sel is not None else None
    _head_v2_lse_merge_kernel[(M,)](
        m_buf,
        l_buf,
        sel if sel is not None else m_buf,
        lse,
        lp if lp is not None else lse,
        n_tiles,
        m_buf.stride(0),
        l_buf.stride(0),
        HAS_SEL=sel is not None,
        BLOCK_T=max(triton.next_power_of_2(n_tiles), 2),
    )
    return lse, lp


def _head_v2_check_inputs(hidden, weight, token_ids, temperature):
    assert hidden.ndim == 2 and weight.ndim == 2 and hidden.shape[1] == weight.shape[1]
    assert hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
        "the head contract takes bf16 hidden/weight (fp32 upcast is exact inside the GEMM)"
    )
    assert hidden.is_cuda
    hidden = hidden.contiguous()
    M = hidden.shape[0]
    if token_ids is not None:
        token_ids = token_ids.contiguous().to(device=hidden.device, dtype=torch.int64)
        assert token_ids.shape == (M,)
    if temperature is not None:
        temperature = temperature.reshape(-1).to(device=hidden.device, dtype=torch.float32).contiguous()
        assert temperature.shape == (M,)
        torch._assert_async((temperature > 0).all(), "temperature must be > 0")
    return hidden, token_ids, temperature


def head_v2_selected_logprob(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: torch.Tensor | None = None,
):
    """Scoring path (trainer bi_fused v2 / serving prefill input logprobs):
    logits are NEVER materialized — per-tile stats + selected gather in the
    GEMM epilogue, one merge launch. Returns ``(logprob, lse, selected)``,
    all ``[N]`` fp32 (temperature-scaled when given), like v1."""
    hidden, token_ids, temperature = _head_v2_check_inputs(hidden, weight, token_ids, temperature)
    M = hidden.shape[0]
    V = weight.shape[0]
    n_tiles = triton.cdiv(V, HEAD_V2_STATS_TILE_N)
    if M == 0:
        z = torch.empty((0,), dtype=torch.float32, device=hidden.device)
        return z, z.clone(), z.clone()
    m_buf = torch.empty((M, n_tiles), dtype=torch.float32, device=hidden.device)
    l_buf = torch.empty((M, n_tiles), dtype=torch.float32, device=hidden.device)
    sel = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _head_v2_launch(hidden, weight.t(), None, m_buf, l_buf, sel, token_ids, temperature)
    lse, lp = _head_v2_merge(m_buf, l_buf, sel)
    return lp, lse, sel


def head_v2_full_logits_with_lse(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    temperature: torch.Tensor | None = None,
):
    """Decode path: full ``[N, V]`` fp32 logits for the sampler (bitwise ==
    the v1 head GEMM — same kernel body, same pinned K-chain) PLUS the lse
    from the epilogue stats. The sampled token's logprob afterwards is
    ``clamp_max(logits[row, tok] - lse[row], 0.0)`` — elementwise aten, no
    second stats pass, no BI log_softmax."""
    hidden, _, temperature = _head_v2_check_inputs(hidden, weight, None, temperature)
    M = hidden.shape[0]
    V = weight.shape[0]
    n_tiles = triton.cdiv(V, HEAD_V2_STATS_TILE_N)
    logits = torch.empty((M, V), dtype=torch.float32, device=hidden.device)
    if M == 0:
        return logits, torch.empty((0,), dtype=torch.float32, device=hidden.device)
    m_buf = torch.empty((M, n_tiles), dtype=torch.float32, device=hidden.device)
    l_buf = torch.empty((M, n_tiles), dtype=torch.float32, device=hidden.device)
    _head_v2_launch(hidden, weight.t(), logits, m_buf, l_buf, None, None, temperature)
    lse, _ = _head_v2_merge(m_buf, l_buf, None)
    return logits, lse


@triton.jit
def _head_v2_stats_from_logits_kernel(
    logits_ptr,
    m_out_ptr,
    l_out_ptr,
    sel_ptr,
    tok_ptr,
    temp_ptr,
    N,
    n_tiles,
    stride_lm,
    stride_mm,
    stride_lm_out,
    HAS_TOKS: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Per-tile (m, l) stats over MATERIALIZED fp32 logits — the same tree as
    the epilogue stats (same TILE_N, same max/exp/pairwise-tree expressions on
    the same fp32 values), for callers that must sample from the logits they
    score (the decode rescore hook). Bitwise == the epilogue path."""
    pid = tl.program_id(0)
    row = pid // n_tiles
    tile = pid % n_tiles
    start_n = tile * TILE_N
    offs_n = start_n + tl.arange(0, TILE_N)
    col_mask = offs_n < N
    x = tl.load(logits_ptr + row * stride_lm + offs_n, mask=col_mask, other=0.0)
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + row)
        x = x * inv_t
    neg_inf = float("-inf")
    m_tile = tl.max(tl.where(col_mask, x, neg_inf))
    terms = tl.where(col_mask, tl.exp(x - m_tile), 0.0)
    l_tile = _pairwise_tree_sum(terms, TILE_N)
    tl.store(
        m_out_ptr + row * stride_mm + tile + tl.arange(0, 1),
        m_tile + tl.zeros((1,), dtype=tl.float32),
    )
    tl.store(l_out_ptr + row * stride_lm_out + tile + tl.arange(0, 1), l_tile)
    if HAS_TOKS:
        tok = tl.load(tok_ptr + row)
        if (tok >= start_n) & (tok < start_n + TILE_N):
            sel = tl.max(tl.where(offs_n == tok, x, neg_inf))
            tl.store(sel_ptr + row, sel)


def head_v2_selected_logprob_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: torch.Tensor | None = None,
):
    """Head-v2 twin of the v1 from-logits rescore: same (m, l) tree over an
    existing fp32 logits tensor, so with logits from
    ``head_v2_full_logits_with_lse`` (or the bitwise-equal v1 GEMM) the result
    is bitwise identical to ``head_v2_selected_logprob`` on the same
    hidden/weight/temperature. Returns ``(logprob, lse, selected)``."""
    assert logits.ndim == 2 and logits.dtype == torch.float32 and logits.is_cuda
    assert logits.stride(1) == 1, "logits rows must be unit-stride"
    M, V = logits.shape
    token_ids = token_ids.contiguous().to(device=logits.device, dtype=torch.int64)
    assert token_ids.shape == (M,)
    if temperature is not None:
        temperature = temperature.reshape(-1).to(device=logits.device, dtype=torch.float32).contiguous()
        assert temperature.shape == (M,)
        torch._assert_async((temperature > 0).all(), "temperature must be > 0")
    n_tiles = triton.cdiv(V, HEAD_V2_STATS_TILE_N)
    if M == 0:
        z = torch.empty((0,), dtype=torch.float32, device=logits.device)
        return z, z.clone(), z.clone()
    m_buf = torch.empty((M, n_tiles), dtype=torch.float32, device=logits.device)
    l_buf = torch.empty((M, n_tiles), dtype=torch.float32, device=logits.device)
    sel = torch.empty((M,), dtype=torch.float32, device=logits.device)
    _head_v2_stats_from_logits_kernel[(M * n_tiles,)](
        logits,
        m_buf,
        l_buf,
        sel,
        token_ids,
        temperature if temperature is not None else m_buf,
        V,
        n_tiles,
        logits.stride(0),
        m_buf.stride(0),
        l_buf.stride(0),
        HAS_TOKS=True,
        HAS_TEMP=temperature is not None,
        TILE_N=HEAD_V2_STATS_TILE_N,
    )
    lse, lp = _head_v2_merge(m_buf, l_buf, sel)
    return lp, lse, sel


# --- split realization of the SAME norm tree (structure is bit-neutral) -----
#
# The split realization computes the IDENTICAL tree — per-tile partial trees in
# a 2D-parallel kernel, the pinned combine in a per-row kernel, elementwise
# normalize in a 2D-parallel kernel — with fp32 partials stored/loaded exactly
# and the bf16 residual round-trip identical to the fused kernel's own res_out
# reload. Bit-equality fused==split is gated by the families-v2 norm tests,
# which force each realization explicitly rather than relying on the rule
# below to select one, and both realizations check the same frozen goldens.
#
# It pays three launches and an HBM round-trip for the partials, and buys
# rows*n_tiles-way parallelism where the fused grid=(rows,) has only rows-way.
# For the residual specialization, that trade pays at few rows over many tiles.
# The no-residual fused specialization currently spills on Hopper, so it always
# uses the split form (see _v2_norm_use_split for the measured boundary).

V2_NORM_SPLIT_MIN_TILES = 10  # no-residual crossover on non-Hopper devices
V2_NORM_RESIDUAL_SPLIT_MIN_TILES = 13  # keeps all shipped hidden sizes (at most 12 tiles) fused


def _v2_norm_use_split(
    rows: int,
    n_tiles: int,
    *,
    has_residual: bool = True,
    is_hopper: bool = False,
) -> bool:
    """Structure switch (perf-only, NOT bit-relevant).

    The no-residual fused specialization spills on Hopper, while the split
    realization is bit-identical and avoids that register-pressure cliff.
    Other architectures and the residual specialization retain the measured
    few-rows/many-tiles switch, where the fused ``grid=(rows,)`` cannot fill
    the GPU.

    ``n_tiles`` must be the split kernel's 512-wide tile count, not the fused
    kernel's 4096-wide chunk count. The threshold is a Hopper performance
    policy. Both realizations compute the same tree, so retuning it may change
    speed only; the fused-versus-split equality gate must still pass.
    """
    if is_hopper and not has_residual:
        return True
    min_tiles = V2_NORM_RESIDUAL_SPLIT_MIN_TILES if has_residual else V2_NORM_SPLIT_MIN_TILES
    return n_tiles >= min_tiles and rows <= n_tiles


@triton.jit
def _rms_norm_v2_partials_kernel(
    x_ptr,
    res_ptr,
    res_out_ptr,
    part_ptr,
    n_cols,
    n_tiles,
    stride_x,
    stride_res,
    stride_res_out,
    HAS_RESIDUAL: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // n_tiles
    t = pid % n_tiles
    cols = t * TILE + tl.arange(0, TILE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
        s = _rtne_bf16(x + r)
        tl.store(
            res_out_ptr + row * stride_res_out + cols,
            s.to(res_out_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        s = x
    p = _pairwise_tree_sum(s * s, TILE)
    tl.store(part_ptr + row * n_tiles + t + tl.arange(0, 1), p)


@triton.jit
def _rms_norm_v2_invrms_kernel(
    part_ptr,
    invrms_ptr,
    n_cols,
    n_tiles,
    eps,
    CHUNK_TILES: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # combine = per-BLOCK_H-chunk pairwise tree over its CHUNK_TILES partials,
    # then the sequential cross-chunk chain — identical to the fused kernel.
    row = tl.program_id(0)
    n_chunks = tl.cdiv(n_tiles, CHUNK_TILES)
    total = tl.zeros((1,), dtype=tl.float32)
    for c in range(n_chunks):
        idx = c * CHUNK_TILES + tl.arange(0, CHUNK_TILES)
        p = tl.load(part_ptr + row * n_tiles + idx, mask=idx < n_tiles, other=0.0)
        total = total + _pairwise_tree_sum(p, CHUNK_TILES)
    var = total / n_cols.to(tl.float32)
    inv_rms = tl.rsqrt(var + eps)
    tl.store(invrms_ptr + row + tl.arange(0, 1), inv_rms)


@triton.jit
def _rms_norm_v2_normalize_kernel(
    s_ptr,
    w_ptr,
    invrms_ptr,
    out_ptr,
    n_cols,
    n_tiles,
    stride_s,
    stride_out,
    ZERO_CENTERED: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // n_tiles
    t = pid % n_tiles
    cols = t * TILE + tl.arange(0, TILE)
    mask = cols < n_cols
    s = tl.load(s_ptr + row * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        w = 1.0 + w
    inv_rms = tl.load(invrms_ptr + row)
    y = s * inv_rms * w
    tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _rms_norm_v2_split(x, weight, eps, residual, zero_centered):
    rows, H = x.shape
    n_tiles = triton.cdiv(H, V2_NORM_TILE)
    out = torch.empty_like(x)
    part = torch.empty((rows, n_tiles), dtype=torch.float32, device=x.device)
    invrms = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if residual is not None:
        res_out = torch.empty_like(x)
        _rms_norm_v2_partials_kernel[(rows * n_tiles,)](
            x,
            residual,
            res_out,
            part,
            H,
            n_tiles,
            x.stride(0),
            residual.stride(0),
            res_out.stride(0),
            HAS_RESIDUAL=True,
            TILE=V2_NORM_TILE,
        )
        s_buf = res_out
    else:
        res_out = None
        _rms_norm_v2_partials_kernel[(rows * n_tiles,)](
            x,
            x,
            x,
            part,
            H,
            n_tiles,
            x.stride(0),
            0,
            0,
            HAS_RESIDUAL=False,
            TILE=V2_NORM_TILE,
        )
        s_buf = x
    _rms_norm_v2_invrms_kernel[(rows,)](
        part,
        invrms,
        H,
        n_tiles,
        eps,
        CHUNK_TILES=V2_NORM_BLOCK_H // V2_NORM_TILE,
        BLOCK_T=max(triton.next_power_of_2(n_tiles), 2),
    )
    _rms_norm_v2_normalize_kernel[(rows * n_tiles,)](
        s_buf,
        weight,
        invrms,
        out,
        H,
        n_tiles,
        s_buf.stride(0),
        out.stride(0),
        ZERO_CENTERED=zero_centered,
        TILE=V2_NORM_TILE,
    )
    return (out, res_out) if residual is not None else out
