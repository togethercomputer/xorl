from dataclasses import dataclass

import pytest
import torch


# ---------------------------------------------------------------------------
# Diff computation (same as dumper comparator)
# ---------------------------------------------------------------------------
@dataclass
class DiffInfo:
    rel_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    p50_abs_diff: float
    p95_abs_diff: float
    p99_abs_diff: float


def compute_diff(baseline: torch.Tensor, target: torch.Tensor) -> DiffInfo:
    """Compute diff metrics matching the dumper comparator."""
    x = baseline.flatten().float()
    y = target.flatten().float()

    abs_diff = (x - y).abs()

    # rel_diff: cosine-distance-like metric
    xy = (x * y).sum()
    x2 = (x * x).sum()
    y2 = (y * y).sum()
    denom = x2 + y2
    if denom > 0:
        sim = 2.0 * xy / denom
        rel_diff = (1.0 - sim).item()
    else:
        rel_diff = 0.0

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    sorted_diff = abs_diff.sort().values
    n = len(sorted_diff)
    p50 = sorted_diff[int(n * 0.50)].item() if n > 0 else 0.0
    p95 = sorted_diff[min(int(n * 0.95), n - 1)].item() if n > 0 else 0.0
    p99 = sorted_diff[min(int(n * 0.99), n - 1)].item() if n > 0 else 0.0

    return DiffInfo(
        rel_diff=rel_diff,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        p50_abs_diff=p50,
        p95_abs_diff=p95,
        p99_abs_diff=p99,
    )


def print_diff(name: str, diff: DiffInfo):
    print(
        f"  {name}: rel_diff={diff.rel_diff:.2e}, max_abs={diff.max_abs_diff:.2e}, "
        f"mean_abs={diff.mean_abs_diff:.2e}, p50={diff.p50_abs_diff:.2e}, "
        f"p95={diff.p95_abs_diff:.2e}, p99={diff.p99_abs_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# PyTorch reference implementation (from dsa.py)
# ---------------------------------------------------------------------------
def ref_compute_index_scores(q, weights, k):
    """PyTorch reference: compute index scores.

    Args:
        q:       [seqlen_q, batch, heads, dim] bf16
        k:       [seqlen_kv, batch, dim] bf16
        weights: [seqlen_q, batch, heads] fp32

    Returns:
        index_scores: [batch, seqlen_q, seqlen_kv] fp32
    """
    # q @ k^T -> [sq, b, h, sk]
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    # ReLU
    index_scores = torch.relu(index_scores)
    # Weight by heads
    index_scores = index_scores * weights.float().unsqueeze(-1)
    # Sum across heads
    index_scores = index_scores.sum(dim=2)
    # Transpose to [b, sq, sk]
    index_scores = index_scores.transpose(0, 1)
    return index_scores


def ref_apply_causal_mask(index_scores, compress_ratio):
    """Apply causal mask for compressed KV positions.

    For query at position p, valid compressed groups are [0, (p+1) // compress_ratio).
    Positions outside this range are set to -inf.

    Args:
        index_scores: [batch, seqlen_q, seqlen_kv] fp32
        compress_ratio: int

    Returns:
        masked index_scores: [batch, seqlen_q, seqlen_kv] fp32
    """
    b, sq, sk = index_scores.shape
    q_positions = torch.arange(sq, device=index_scores.device)
    k_positions = torch.arange(sk, device=index_scores.device)
    # valid_end[q] = (q + 1) // compress_ratio
    valid_end = (q_positions + 1) // compress_ratio  # [sq]
    # mask: k_pos < valid_end[q_pos]
    mask = k_positions.unsqueeze(0) < valid_end.unsqueeze(1)  # [sq, sk]
    index_scores = index_scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    return index_scores


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def requires_tilelang():
    try:
        import tilelang  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="tilelang not installed")


def make_inputs(seqlen_q, batch, heads, dim, compress_ratio, device="cuda"):
    """Generate random inputs matching V4 indexer shapes."""
    seqlen_kv = seqlen_q // compress_ratio

    q = torch.randn(seqlen_q, batch, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seqlen_kv, batch, dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(seqlen_q, batch, heads, device=device, dtype=torch.float32) * 0.01

    return q, k, weights


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------
# Test configurations: (seqlen_q, batch, heads, dim, compress_ratio, topk)
FORWARD_CONFIGS = [
    # One representative for each kernel geometry that changes execution:
    # basic, batched, production heads/top-k, and the C128 compression family.
    (128, 1, 8, 128, 4, 32),
    (512, 4, 16, 128, 4, 64),
    (2048, 1, 64, 128, 4, 512),
    (2048, 1, 8, 128, 128, 16),
]

FORWARD_CONFIG_IDS = [f"sq{sq}_b{b}_h{h}_d{d}_cr{cr}_top{tk}" for sq, b, h, d, cr, tk in FORWARD_CONFIGS]


@requires_cuda()
@requires_tilelang()
@pytest.mark.parametrize("seqlen_q,batch,heads,dim,compress_ratio,topk", FORWARD_CONFIGS, ids=FORWARD_CONFIG_IDS)
def test_indexer_forward_scores(seqlen_q, batch, heads, dim, compress_ratio, topk):
    """Compare tilelang forward logits against PyTorch reference."""
    from xorl.ops.dsv4.kernel.tilelang_indexer_fwd import (
        _make_causal_cu_seqlens,
        batched_indexer_fwd,
    )

    q, k, weights = make_inputs(seqlen_q, batch, heads, dim, compress_ratio)
    seqlen_kv = seqlen_q // compress_ratio

    # Reference
    ref_scores = ref_compute_index_scores(q, weights, k)
    ref_scores = ref_apply_causal_mask(ref_scores, compress_ratio)

    # Tilelang
    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    # Compare only valid (non-masked) positions
    valid_mask = ref_scores != float("-inf")
    ref_valid = ref_scores[valid_mask]
    tl_valid = tl_scores[valid_mask]
    tl_masked = tl_scores[~valid_mask]

    diff = compute_diff(ref_valid, tl_valid)
    print(f"\n[FWD] sq={seqlen_q}, b={batch}, h={heads}, cr={compress_ratio}, topk={topk}")
    print_diff("logits", diff)

    # Thresholds: tilelang bf16 GEMM vs pytorch fp32 einsum — allow some tolerance
    assert diff.rel_diff < 1e-3, f"rel_diff too large: {diff.rel_diff:.2e}"
    assert diff.max_abs_diff < 1.0, f"max_abs_diff too large: {diff.max_abs_diff:.2e}"
    assert diff.mean_abs_diff < 0.05, f"mean_abs_diff too large: {diff.mean_abs_diff:.2e}"
    assert tl_masked.numel() > 0
    assert torch.isneginf(tl_masked).all(), "future compressed groups were not masked"

    if (seqlen_q, batch, heads, dim, compress_ratio, topk) == FORWARD_CONFIGS[0]:
        _assert_large_values()
        _assert_zero_inputs()


# ---------------------------------------------------------------------------
# Numerical stability test: large values
# ---------------------------------------------------------------------------
def _assert_large_values():
    """Test with large input values to check for overflow/underflow."""
    from xorl.ops.dsv4.kernel.tilelang_indexer_fwd import (
        _make_causal_cu_seqlens,
        batched_indexer_fwd,
    )

    seqlen_q, batch, heads, dim, compress_ratio = 256, 1, 8, 128, 4
    seqlen_kv = seqlen_q // compress_ratio

    q = torch.randn(seqlen_q, batch, heads, dim, device="cuda", dtype=torch.bfloat16) * 10.0
    k = torch.randn(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16) * 10.0
    weights = torch.randn(seqlen_q, batch, heads, device="cuda", dtype=torch.float32) * 0.1

    ref_scores = ref_compute_index_scores(q, weights, k)
    ref_scores = ref_apply_causal_mask(ref_scores, compress_ratio)

    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    valid_mask = ref_scores != float("-inf")
    diff = compute_diff(ref_scores[valid_mask], tl_scores[valid_mask])
    print("\n[LARGE] large values test")
    print_diff("logits", diff)

    # Larger values → larger absolute diff but rel_diff should stay reasonable
    assert diff.rel_diff < 5e-3, f"rel_diff too large: {diff.rel_diff:.2e}"
    assert not torch.isnan(tl_scores[valid_mask]).any(), "NaN in tilelang output"
    assert not torch.isinf(tl_scores[valid_mask]).any(), "Inf in tilelang output (non-masked)"


# ---------------------------------------------------------------------------
# Zero input test
# ---------------------------------------------------------------------------
def _assert_zero_inputs():
    """Test that zero inputs produce zero scores."""
    from xorl.ops.dsv4.kernel.tilelang_indexer_fwd import (
        _make_causal_cu_seqlens,
        batched_indexer_fwd,
    )

    seqlen_q, batch, heads, dim, compress_ratio = 128, 1, 8, 128, 4
    seqlen_kv = seqlen_q // compress_ratio

    q = torch.zeros(seqlen_q, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(seqlen_kv, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(seqlen_q, batch, heads, device="cuda", dtype=torch.float32)

    cu_ks, cu_ke = _make_causal_cu_seqlens(seqlen_q, seqlen_kv, compress_ratio, q.device)
    tl_scores = batched_indexer_fwd(q, k, weights, cu_ks, cu_ke)

    valid_mask = tl_scores != float("-inf")
    valid_scores = tl_scores[valid_mask]
    assert (valid_scores == 0).all(), f"Expected all zeros for valid positions, got max={valid_scores.max():.2e}"
