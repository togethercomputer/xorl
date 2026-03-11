"""Tests for ring attention (context parallelism) -- unit tests only.

Distributed tests removed -- run with torchrun separately.
"""

import pytest
import torch

from xorl.distributed.sequence_parallel.ring_attention import (
    _get_zigzag_step_section,
    _merge_attn_outputs,
)
from xorl.data.collators.sequence_shard_collator import (
    zigzag_reorder_packed_sequence,
)

pytestmark = [pytest.mark.distributed]


class TestLSEMerge:
    """Test _merge_attn_outputs numerical stability."""

    def test_merge_batched_varlen_extreme_equal(self):
        """Merge partial outputs: batched correctness, varlen shapes, extreme LSE stability, equal LSE averaging."""
        # --- Batched merge with manual reference ---
        B, S, H, D = 1, 8, 4, 32
        torch.manual_seed(42)
        out1 = torch.randn(B, S, H, D, device="cuda")
        out2 = torch.randn(B, S, H, D, device="cuda")
        lse1 = torch.randn(B, H, S, device="cuda")
        lse2 = torch.randn(B, H, S, device="cuda")

        merged_out, merged_lse = _merge_attn_outputs(out1, lse1, out2, lse2, is_varlen=False)
        w1 = torch.exp(lse1 - merged_lse).transpose(1, 2).unsqueeze(-1)
        w2 = torch.exp(lse2 - merged_lse).transpose(1, 2).unsqueeze(-1)
        ref_out = w1 * out1 + w2 * out2
        assert torch.allclose(merged_out, ref_out, atol=1e-5)

        # --- Varlen layout: shapes and no NaN ---
        total, H2, D2 = 16, 4, 32
        torch.manual_seed(42)
        out1v = torch.randn(total, H2, D2, device="cuda")
        out2v = torch.randn(total, H2, D2, device="cuda")
        lse1v = torch.randn(H2, total, device="cuda")
        lse2v = torch.randn(H2, total, device="cuda")
        merged_outv, merged_lsev = _merge_attn_outputs(out1v, lse1v, out2v, lse2v, is_varlen=True)
        assert merged_outv.shape == (total, H2, D2)
        assert merged_lsev.shape == (H2, total)
        assert not torch.isnan(merged_outv).any()
        assert not torch.isnan(merged_lsev).any()

        # --- Extreme LSE: dominant term wins ---
        B2, S2, H3, D3 = 1, 4, 2, 16
        out1e = torch.randn(B2, S2, H3, D3, device="cuda")
        out2e = torch.randn(B2, S2, H3, D3, device="cuda")
        lse1e = torch.full((B2, H3, S2), 100.0, device="cuda")
        lse2e = torch.full((B2, H3, S2), -100.0, device="cuda")
        merged_oute, _ = _merge_attn_outputs(out1e, lse1e, out2e, lse2e, is_varlen=False)
        assert torch.allclose(merged_oute, out1e, atol=1e-4)
        assert not torch.isnan(merged_oute).any()
        assert not torch.isinf(merged_oute).any()

        # --- Equal LSE: output is average ---
        out1q = torch.randn(B2, S2, H3, D3, device="cuda")
        out2q = torch.randn(B2, S2, H3, D3, device="cuda")
        lse_eq = torch.randn(B2, H3, S2, device="cuda")
        merged_outq, _ = _merge_attn_outputs(out1q, lse_eq.clone(), out2q, lse_eq.clone(), is_varlen=False)
        assert torch.allclose(merged_outq, (out1q + out2q) / 2, atol=1e-5)


class TestZigzagUnit:
    """Unit tests for zigzag section logic and reorder (no GPU needed)."""

    def test_zigzag_sections_and_reorder(self):
        """All ranks compute all steps; non-diagonal sections are lower/upper; reorder permutations correct."""
        # Every rank computes every step, step 0 is always diagonal
        for ringattn_size in [2, 4, 8]:
            for rank in range(ringattn_size):
                sections = [_get_zigzag_step_section(rank, ringattn_size, s) for s in range(ringattn_size)]
                assert len(sections) == ringattn_size
                assert sections[0] == "diagonal"

        # Non-diagonal sections are "lower" or "upper"
        for rank in range(4):
            for step in range(1, 4):
                section = _get_zigzag_step_section(rank, 4, step)
                assert section in ("lower", "upper")

        # Single doc reorder
        ringattn_size = 2
        tensor = torch.arange(40).unsqueeze(0)
        position_ids = torch.arange(40).unsqueeze(0)
        reordered = zigzag_reorder_packed_sequence(tensor, position_ids, ringattn_size, dim=-1)
        expected = torch.cat([
            torch.arange(0, 10), torch.arange(30, 40),
            torch.arange(10, 20), torch.arange(20, 30),
        ]).unsqueeze(0)
        assert torch.equal(reordered, expected)

        # Multi-doc reorder
        doc_len = 20
        total = 2 * doc_len
        tensor_m = torch.arange(total).unsqueeze(0)
        position_ids_m = torch.cat([torch.arange(doc_len), torch.arange(doc_len)]).unsqueeze(0)
        reordered_m = zigzag_reorder_packed_sequence(tensor_m, position_ids_m, ringattn_size, dim=-1)
        expected_m = torch.cat([
            torch.arange(0, 5), torch.arange(15, 20),
            torch.arange(20, 25), torch.arange(35, 40),
            torch.arange(5, 10), torch.arange(10, 15),
            torch.arange(25, 30), torch.arange(30, 35),
        ]).unsqueeze(0)
        assert torch.equal(reordered_m, expected_m)

        # Position IDs doc boundaries per rank
        num_docs = 2
        pos_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
        reordered_pos = zigzag_reorder_packed_sequence(pos_ids, pos_ids, ringattn_size, dim=-1)
        half = total // 2
        rank0_pos = reordered_pos[0, :half]
        zeros = (rank0_pos == 0).nonzero(as_tuple=False).view(-1).tolist()
        assert len(zeros) == 2

        # Various ringattn_sizes: shape preserved, permutation, early < late
        for cs in [2, 4, 8]:
            n = 2 * cs
            dl = n * 4
            t = torch.arange(dl).unsqueeze(0)
            p = torch.arange(dl).unsqueeze(0)
            r = zigzag_reorder_packed_sequence(t, p, cs, dim=-1)
            assert r.shape == t.shape
            assert set(r[0].tolist()) == set(t[0].tolist())
            chunk_size = dl // cs
            for rk in range(cs):
                rank_slice = r[0, rk * chunk_size : (rk + 1) * chunk_size]
                sub_size = chunk_size // 2
                assert rank_slice[:sub_size].max() < rank_slice[sub_size:].min()

        # ringattn_size=1 is no-op
        t1 = torch.arange(20).unsqueeze(0)
        assert torch.equal(zigzag_reorder_packed_sequence(t1, t1, 1, dim=-1), t1)

        # Invalid length raises
        with pytest.raises(ValueError, match="not divisible"):
            zigzag_reorder_packed_sequence(
                torch.arange(15).unsqueeze(0), torch.arange(15).unsqueeze(0), 2, dim=-1
            )
