"""Tests for xorl.ops.group_gemm.kernel.moe module.

These tests compare the optimized Triton kernels for MoE operations
with naive PyTorch implementations to ensure correctness.
"""

import pytest
import torch


# Mark all tests as GPU since MoE ops require CUDA
pytestmark = pytest.mark.gpu


def naive_expert_histogram(input: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Naive PyTorch implementation of expert histogram."""
    return torch.bincount(input.flatten().int(), minlength=num_bins)[:num_bins]


def naive_moe_gather(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Naive PyTorch implementation of moe_gather."""
    M, topk = index.shape
    N = x.shape[1]
    output = torch.zeros(M, N, dtype=x.dtype, device=x.device)
    for m in range(M):
        for k in range(topk):
            idx = index[m, k].item()
            output[m] += x[idx]
    return output


def naive_moe_scatter(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Naive PyTorch implementation of moe_scatter."""
    M, topk = index.shape
    N = x.shape[1]
    output = torch.zeros(M * topk, N, dtype=x.dtype, device=x.device)
    for m in range(M):
        for k in range(topk):
            idx = index[m, k].item()
            output[idx] = x[m]
    return output


def naive_moe_add_gather(x: torch.Tensor, y: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Naive PyTorch implementation of moe_add_gather."""
    return naive_moe_gather(x + y, index)


class TestExpertHistogramAndIndex:
    """Tests for expert_histogram and moe_index_compute."""

    def test_histogram_and_index_compute(self):
        """Histogram: basic, 2D, large, int64 inputs. Index compute: ordering and uniqueness."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from xorl.ops.group_gemm.kernel.moe import expert_histogram, moe_index_compute  # noqa: PLC0415
        except ImportError:
            pytest.skip("moe ops not available")

        # Basic 1D histogram
        input_1d = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0], dtype=torch.int32).cuda()
        output = expert_histogram(input_1d, 4)
        assert torch.equal(output.cpu(), naive_expert_histogram(input_1d, 4).cpu())
        assert output.tolist() == [3, 2, 2, 1]

        # 2D input (topk routing)
        input_2d = torch.tensor([[0, 1], [2, 0], [1, 3], [2, 0]], dtype=torch.int32).cuda()
        output_2d = expert_histogram(input_2d, 4)
        assert torch.equal(output_2d.cpu(), naive_expert_histogram(input_2d, 4).cpu())
        assert output_2d.tolist() == [3, 2, 2, 1]

        # Large histogram
        num_experts = 64
        input_large = torch.randint(0, num_experts, (10000,), dtype=torch.int32).cuda()
        output_large = expert_histogram(input_large, num_experts)
        assert torch.equal(output_large.cpu(), naive_expert_histogram(input_large, num_experts).cpu())
        assert output_large.sum().item() == 10000

        # int64 input
        input_i64 = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0], dtype=torch.int64).cuda()
        output_i64 = expert_histogram(input_i64, 4)
        assert torch.equal(output_i64.cpu(), naive_expert_histogram(input_i64, 4).cpu())

        # --- Index compute: basic + ordering ---
        experts_for_tokens = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=torch.int32).cuda()
        histogram = expert_histogram(experts_for_tokens.flatten(), 4)
        cumsum = torch.cumsum(histogram, dim=0).int().cuda()
        indices = moe_index_compute(experts_for_tokens, cumsum)
        assert indices.shape == experts_for_tokens.shape
        assert torch.all(indices >= 0)
        assert torch.all(indices < experts_for_tokens.numel())
        assert indices.flatten().unique().numel() == experts_for_tokens.numel()

        # Ordering: same expert gets sequential indices
        experts_same = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.int32).cuda()
        hist2 = expert_histogram(experts_same.flatten(), 2)
        cumsum2 = torch.cumsum(hist2, dim=0).int().cuda()
        indices2 = moe_index_compute(experts_same, cumsum2)
        expert_0_idx = indices2[experts_same == 0].sort()[0]
        expert_1_idx = indices2[experts_same == 1].sort()[0]
        assert torch.all(expert_0_idx == torch.arange(4).cuda())
        assert torch.all(expert_1_idx == torch.arange(4, 8).cuda())


class TestDeterministicScatter:
    """Deterministic scatter (default ON): run-invariant permutation from moe_index_compute.

    XORL_MOE_DETERMINISTIC_SCATTER=0 is the escape hatch back to the atomics kernel.
    """

    @staticmethod
    def _naive_stable_index(experts_for_tokens: torch.Tensor, num_experts: int) -> torch.Tensor:
        """Reference: expert regions in expert-id order, rows in flattened (token, k) order."""
        flat = experts_for_tokens.flatten().tolist()
        hist = [0] * num_experts
        for e in flat:
            hist[e] += 1
        start, acc = [0] * num_experts, 0
        for e in range(num_experts):
            start[e] = acc
            acc += hist[e]
        seen = [0] * num_experts
        out = []
        for e in flat:
            out.append(start[e] + seen[e])
            seen[e] += 1
        return torch.tensor(out, dtype=torch.int64).view(experts_for_tokens.shape)

    def test_deterministic_scatter(self, monkeypatch):
        """Run-invariance, validity vs stock coverage, exact stable order, escape-hatch routing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            import xorl.ops.group_gemm.kernel.moe as moe_mod  # noqa: PLC0415
            from xorl.ops.group_gemm.kernel.moe import expert_histogram, moe_index_compute  # noqa: PLC0415
        except ImportError:
            pytest.skip("moe ops not available")

        torch.manual_seed(0)
        num_experts, M, topk = 16, 4096, 8
        experts = torch.randint(0, num_experts, (M, topk), dtype=torch.int32).cuda()
        hist = expert_histogram(experts, num_experts)
        cumsum = torch.cumsum(hist, dim=0)

        # Stock atomics draw (escape hatch =0) for the coverage comparison.
        monkeypatch.setenv("XORL_MOE_DETERMINISTIC_SCATTER", "0")
        stock = moe_index_compute(experts, cumsum)

        # Default ON: flag unset routes through the deterministic path.
        monkeypatch.delenv("XORL_MOE_DETERMINISTIC_SCATTER", raising=False)
        det_a = moe_index_compute(experts, cumsum)
        det_b = moe_index_compute(experts, cumsum)

        # Run-invariance: two independent calls produce the identical permutation.
        assert torch.equal(det_a, det_b)

        # Validity: same shape/dtype contract as stock, full slot coverage, and per-expert
        # slot SETS identical to the stock draw (both fill [cumsum[e-1], cumsum[e])).
        assert det_a.shape == stock.shape
        assert det_a.dtype == stock.dtype
        full = torch.arange(M * topk, device=det_a.device)
        assert torch.equal(det_a.flatten().sort()[0], full)
        for e in range(num_experts):
            mask = experts == e
            assert torch.equal(det_a[mask].sort()[0], stock[mask].sort()[0])

        # Exact value: stable order = flattened (token, k) order within each expert.
        expected = self._naive_stable_index(experts.cpu(), num_experts).cuda()
        assert torch.equal(det_a, expected)

        # int64 ids too (the model path passes topk_ids as int64).
        assert torch.equal(moe_index_compute(experts.long().contiguous(), cumsum), expected)

        # Truthy values keep routing through the deterministic path.
        monkeypatch.setenv("XORL_MOE_DETERMINISTIC_SCATTER", "1")
        assert torch.equal(moe_index_compute(experts, cumsum), expected)

        # Escape hatch: explicit falsey values must never route through the
        # deterministic path (old atomics behavior).
        def _boom(*args, **kwargs):
            raise AssertionError("deterministic path used with escape hatch set")

        monkeypatch.setattr(moe_mod, "_moe_index_compute_deterministic", _boom)
        for falsey in ("0", "false", "off", ""):
            monkeypatch.setenv("XORL_MOE_DETERMINISTIC_SCATTER", falsey)
            out = moe_index_compute(experts, cumsum)
            assert torch.equal(out.flatten().sort()[0], full)

        # Default ON: unset must route through the deterministic path.
        monkeypatch.delenv("XORL_MOE_DETERMINISTIC_SCATTER", raising=False)
        with pytest.raises(AssertionError, match="deterministic path used"):
            moe_index_compute(experts, cumsum)


class TestMoEGatherScatterAddGather:
    """Tests for moe_gather, moe_scatter, moe_add_gather, and scatter-gather roundtrip."""

    def test_gather_scatter_add_gather(self):
        """Gather (basic, overlapping, large), scatter, scatter-gather roundtrip, add_gather, equivalence."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from xorl.ops.group_gemm.kernel.moe import moe_add_gather, moe_gather, moe_scatter  # noqa: PLC0415
        except ImportError:
            pytest.skip("moe ops not available")

        # --- Gather: basic ---
        M, topk, N = 4, 2, 8
        x = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        index = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32).cuda()
        output = moe_gather(x, index)
        assert output.shape == (M, N)
        assert torch.allclose(output.float(), naive_moe_gather(x, index).float(), rtol=1e-2, atol=1e-2)

        # Gather: overlapping indices
        M2, topk2, N2 = 3, 2, 16
        x2 = torch.randn(M2 * topk2, N2, dtype=torch.bfloat16).cuda()
        index2 = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32).cuda()
        assert torch.allclose(
            moe_gather(x2, index2).float(),
            naive_moe_gather(x2, index2).float(),
            rtol=1e-2,
            atol=1e-2,
        )

        # Gather: large dimensions
        M3, topk3, N3 = 1024, 2, 4096
        x3 = torch.randn(M3 * topk3, N3, dtype=torch.float16).cuda()
        index3 = torch.arange(M3 * topk3, dtype=torch.int32).cuda().reshape(M3, topk3)
        assert torch.allclose(
            moe_gather(x3, index3).float(),
            naive_moe_gather(x3, index3).float(),
            rtol=1e-2,
            atol=1e-2,
        )

        # --- Scatter: basic ---
        xs = torch.randn(M, N, dtype=torch.float16).cuda()
        idx_s = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32).cuda()
        out_s = moe_scatter(xs, idx_s)
        assert out_s.shape == (M * topk, N)
        assert torch.allclose(out_s.float(), naive_moe_scatter(xs, idx_s).float(), rtol=1e-2, atol=1e-2)

        # --- Scatter-gather roundtrip ---
        M4, topk4, N4 = 8, 2, 16
        x4 = torch.randn(M4, N4, dtype=torch.bfloat16).cuda()
        idx4 = torch.arange(M4 * topk4, dtype=torch.int32).cuda().reshape(M4, topk4)
        scattered = moe_scatter(x4, idx4)
        gathered = moe_gather(scattered, idx4)
        expected = x4 * topk4
        assert torch.allclose(gathered, expected, rtol=1e-2, atol=1e-2)

        # --- Add-gather: basic ---
        xa = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        ya = torch.randn(M * topk, N, dtype=torch.float16).cuda()
        idx_a = torch.arange(M * topk, dtype=torch.int32).cuda().reshape(M, topk)
        out_ag = moe_add_gather(xa, ya, idx_a)
        assert out_ag.shape == (M, N)
        assert torch.allclose(out_ag.float(), naive_moe_add_gather(xa, ya, idx_a).float(), rtol=1e-2, atol=1e-2)

        # Add-gather equivalence with manual add+gather
        M5, topk5, N5 = 8, 2, 16
        x5 = torch.randn(M5 * topk5, N5, dtype=torch.bfloat16).cuda()
        y5 = torch.randn(M5 * topk5, N5, dtype=torch.bfloat16).cuda()
        idx5 = torch.arange(M5 * topk5, dtype=torch.int32).cuda().reshape(M5, topk5)
        assert torch.allclose(
            moe_add_gather(x5, y5, idx5),
            moe_gather(x5 + y5, idx5),
            rtol=1e-3,
            atol=1e-3,
        )


class TestMoEIntegration:
    """Integration tests combining multiple MoE operations."""

    def test_full_moe_pipeline(self):
        """Complete MoE forward pipeline: histogram -> index -> scatter -> gather."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from xorl.ops.group_gemm.kernel.moe import (  # noqa: PLC0415
                expert_histogram,
                moe_gather,
                moe_index_compute,
                moe_scatter,
            )
        except ImportError:
            pytest.skip("moe ops not available")

        M, topk, N, num_experts = 16, 2, 32, 4

        hidden_states = torch.randn(M, N, dtype=torch.float16).cuda()
        experts_for_tokens = torch.randint(0, num_experts, (M, topk), dtype=torch.int32).cuda()

        histogram = expert_histogram(experts_for_tokens.flatten(), num_experts)
        cumsum = torch.cumsum(histogram, dim=0).int().cuda()
        indices = moe_index_compute(experts_for_tokens, cumsum)

        scattered = moe_scatter(hidden_states, indices)
        expert_outputs = scattered  # identity for simplicity
        final_output = moe_gather(expert_outputs, indices)

        assert final_output.shape == (M, N)
        assert final_output.abs().max() > 0
