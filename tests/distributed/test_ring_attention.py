"""Numerical merge policy for ring attention partial outputs."""

import pytest
import torch


pytest.importorskip("flash_attn_interface", reason="ring attention requires the optional FA3 interface")

from xorl.distributed.sequence_parallel.ring_attention import _merge_attn_outputs


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
