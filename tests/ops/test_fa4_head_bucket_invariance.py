"""FA4 per-head head-batch invariance gate for Ulysses.

Ulysses degree d hands each rank the FULL sequence with 1/d of the Q heads
(and replicated KV: one KV head per rank at d > kv_heads). The exact
contract requires that a given head's attention output is byte-identical no
matter which head-batch it is computed in — the analogue of the BI-GEMM
row-bucket gate, for the head axis.

This gate runs the SAME FA4 varlen entry the production backend calls
(``fa4_flash_attn_varlen_func`` with ``num_splits=1``, mirroring
src/xorl/models/layers/attention/backend/flash_attention.py) on the
Qwen3.5-0.8B exact attention geometry (8 Q-heads, 2 KV-heads, head_dim 256,
bf16, packed varlen), slicing the head axis exactly as Ulysses degrees
{1,2,4,8} would (including the GQA-ratio changes 4 -> 4 -> 2 -> 1 and the
fresh contiguous allocations the all-to-all produces), and asserts BYTE
equality per head against the full-batch reference.

Requires one GPU and FA4.
"""

from __future__ import annotations

import pytest
import torch


pytestmark = [pytest.mark.gpu]

if not torch.cuda.is_available():
    pytest.skip("FA4 head-bucket invariance requires CUDA", allow_module_level=True)

fa4 = pytest.importorskip("flash_attn.cute", reason="FA4 (flash_attn.cute) not installed")

NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
# Packed varlen: uneven document lengths, not multiples of typical tiles.
CU_SEQLENS = [0, 384, 1000]
DEGREES = (1, 2, 4, 8)


def _make_qkv(device):
    generator = torch.Generator(device="cpu").manual_seed(1729)
    total = CU_SEQLENS[-1]
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, generator=generator, dtype=torch.float32)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=generator, dtype=torch.float32)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, generator=generator, dtype=torch.float32)
    return (
        q.to(torch.bfloat16).to(device),
        k.to(torch.bfloat16).to(device),
        v.to(torch.bfloat16).to(device),
    )


def _run_fa4(q, k, v, device):
    from flash_attn.cute import flash_attn_varlen_func  # noqa: PLC0415

    cu = torch.tensor(CU_SEQLENS, dtype=torch.int32, device=device)
    max_len = max(b - a for a, b in zip(CU_SEQLENS, CU_SEQLENS[1:]))
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        num_splits=1,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


@pytest.mark.parametrize("degree", DEGREES)
def test_fa4_head_bucket_bytes_match_full_batch(degree):
    """Every Ulysses-degree head grouping must reproduce the full-batch bytes."""
    device = torch.device("cuda")
    q, k, v = _make_qkv(device)
    reference = _run_fa4(q, k, v, device)

    q_per_rank = NUM_Q_HEADS // degree
    q_heads_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
    mismatches = []
    for rank in range(degree):
        q_lo, q_hi = rank * q_per_rank, (rank + 1) * q_per_rank
        # KV heads this rank's Q-group attends to (replication at high degree
        # is a pure copy, so slicing the ORIGINAL kv head is byte-equivalent).
        kv_heads = sorted({h // q_heads_per_kv for h in range(q_lo, q_hi)})
        # .contiguous(): the all-to-all hands the kernel fresh allocations.
        q_group = q[:, q_lo:q_hi, :].contiguous()
        k_group = k[:, kv_heads, :].contiguous()
        v_group = v[:, kv_heads, :].contiguous()
        out_group = _run_fa4(q_group, k_group, v_group, device)
        ref_group = reference[:, q_lo:q_hi, :]
        if not torch.equal(out_group.view(torch.int16), ref_group.contiguous().view(torch.int16)):
            diff = (out_group.float() - ref_group.float()).abs()
            mismatched = int((out_group.view(torch.int16) != ref_group.contiguous().view(torch.int16)).sum())
            mismatches.append(
                f"degree={degree} rank={rank} heads[{q_lo}:{q_hi}] kv={kv_heads}: "
                f"{mismatched} mismatched int16 lanes, max|diff|={diff.max().item():.3e}"
            )
    assert not mismatches, (
        "FA4 head-batch invariance failed: per-head bytes depend on the "
        "head-batch composition:\n" + "\n".join(mismatches)
    )


def test_fa4_single_head_repeatability():
    """Control: the same call twice must be byte-stable (rules out
    run-to-run nondeterminism masquerading as head-batch variance)."""
    device = torch.device("cuda")
    q, k, v = _make_qkv(device)
    first = _run_fa4(q, k, v, device)
    second = _run_fa4(q, k, v, device)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))
