"""Tests for MoE expert weight loading in QLoRA modules.

Verifies that the GPU-transpose loading path produces bit-identical results
to the reference (per-expert CPU transpose) for both NvFP4 and BlockFP8.
Also benchmarks the speedup.
"""

import time
import pytest
import torch
import torch.nn.functional as F

from xorl.qlora.modules.moe_experts import NvFP4QLoRAMoeExperts
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts
from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Helpers: reference (old) implementations
# ---------------------------------------------------------------------------

def _ref_load_nvfp4(packed_hf_list, bs_list, gs_list, device):
    """Reference: per-expert CPU transpose + stack + H2D (old implementation)."""
    packed_out, scales_out, amax_out = [], [], []
    for packed, bs, gs in zip(packed_hf_list, bs_list, gs_list):
        gs_val = gs.float().item()
        packed_gkn = packed.T.contiguous()
        bs_gkn = (bs.float() * gs.float()).T.contiguous()
        packed_out.append(packed_gkn.contiguous().view(torch.uint8))
        scales_out.append(bs_gkn.contiguous().view(torch.uint8))
        amax_out.append(gs_val * FP4_E2M1_MAX * FP8_E4M3_MAX)

    p = torch.stack(packed_out).to(device)
    s = torch.stack(scales_out).to(device)
    return p, s, amax_out


def _ref_load_block_fp8(fp8_list, scales_list, device):
    """Reference: per-expert CPU transpose + stack + H2D (old implementation)."""
    packed_out, scales_out = [], []
    for fp8_w, scales in zip(fp8_list, scales_list):
        fp8_gkn = fp8_w.T.contiguous()
        sc_gkn = scales.float().T.contiguous()
        packed_out.append(fp8_gkn.view(torch.uint8).contiguous())
        scales_out.append(sc_gkn.contiguous().view(torch.uint8))

    p = torch.stack(packed_out).to(device)
    s = torch.stack(scales_out).to(device)
    return p, s


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _make_nvfp4_tensors(E, N, K, block_size=16, seed=0):
    """Generate synthetic HF-format NVFP4 tensors for E experts."""
    torch.manual_seed(seed)
    packed_list, bs_list, gs_list = [], [], []
    for _ in range(E):
        packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        bs = torch.rand(N, K // block_size).to(torch.float8_e4m3fn)
        gs = torch.tensor(0.001 + torch.rand(1).item() * 0.01)
        packed_list.append(packed)
        bs_list.append(bs)
        gs_list.append(gs)
    return packed_list, bs_list, gs_list


def _make_block_fp8_tensors(E, N, K, block_size=128, seed=0):
    """Generate synthetic HF-format block-FP8 tensors for E experts."""
    torch.manual_seed(seed)
    fp8_list, scales_list = [], []
    for _ in range(E):
        fp8_w = torch.rand(N, K).to(torch.float8_e4m3fn)
        scales = torch.rand(N // block_size, K // block_size, dtype=torch.float32)
        fp8_list.append(fp8_w)
        scales_list.append(scales)
    return fp8_list, scales_list


# ---------------------------------------------------------------------------
# Helper: invoke _load_experts via mock _load_tensor
# ---------------------------------------------------------------------------

def _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list):
    """Feed synthetic tensors through NvFP4QLoRAMoeExperts._load_experts."""
    E = len(packed_list)
    # Build a flat lookup keyed by (expert_idx, proj, suffix)
    data = {}
    for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
        for i in range(E):
            fqn = f"layer.{i}.{hf_name}"
            data[f"{fqn}.weight"]       = packed_list[i]
            data[f"{fqn}.weight_scale"] = bs_list[i]
            data[f"{fqn}.weight_scale_2"] = gs_list[i]

    module._source_fqn = "layer"
    module.expert_offset = 0

    cache = {}
    module._load_experts(lambda k: data[k], cache)


def _run_block_fp8_load_experts(module, fp8_list, scales_list):
    """Feed synthetic tensors through BlockFP8QLoRAMoeExperts._load_experts."""
    E = len(fp8_list)
    data = {}
    for proj_name, hf_name in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
        for i in range(E):
            fqn = f"layer.{i}.{hf_name}"
            data[f"{fqn}.weight"]           = fp8_list[i]
            data[f"{fqn}.weight_scale_inv"] = scales_list[i]

    module._source_fqn = "layer"
    module.expert_offset = 0

    cache = {}
    module._load_experts(lambda k: data[k], cache)


# ---------------------------------------------------------------------------
# Tests: correctness vs reference
# ---------------------------------------------------------------------------

class TestNvFP4LoadExperts:
    """NvFP4QLoRAMoeExperts._load_experts produces correct GKN buffers."""

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048), (8, 512, 1024), (1, 256, 512)])
    def test_packed_matches_reference(self, E, N, K):
        """Packed buffer matches per-expert CPU-transpose reference."""
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K)

        module = NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)

        ref_packed, _, _ = _ref_load_nvfp4(packed_list, bs_list, gs_list, DEVICE)

        # gate_packed should equal reference
        got = module.gate_packed
        assert got.shape == ref_packed.shape, f"shape mismatch: {got.shape} vs {ref_packed.shape}"
        assert got.dtype == ref_packed.dtype
        assert torch.equal(got, ref_packed), "packed bytes differ from reference"

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048), (8, 512, 1024)])
    def test_block_scales_match_reference(self, E, N, K, block_size=16):
        """Block scales buffer matches per-expert CPU-transpose reference."""
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K, block_size)

        module = NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)

        _, ref_scales, _ = _ref_load_nvfp4(packed_list, bs_list, gs_list, DEVICE)

        got = module.gate_block_scales
        assert got.shape == ref_scales.shape
        assert torch.equal(got, ref_scales), "block scales differ from reference"

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048)])
    def test_ema_amax_matches_reference(self, E, N, K):
        """EMA amax values match per-expert reference."""
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K)

        module = NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)

        _, _, ref_amax = _ref_load_nvfp4(packed_list, bs_list, gs_list, DEVICE)

        got_amax = module._ema_amax["gate"]
        for i, (g, r) in enumerate(zip(got_amax.tolist(), ref_amax)):
            assert abs(g - r) < 1e-5, f"amax[{i}] {g} != {r}"

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048)])
    def test_all_projections_loaded(self, E, N, K):
        """All three projections (gate, up, down) are loaded correctly."""
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K)

        module = NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)

        for proj in ("gate", "up", "down"):
            packed = getattr(module, f"{proj}_packed")
            scales = getattr(module, f"{proj}_block_scales")
            gs_buf = getattr(module, f"{proj}_global_scale")
            assert packed is not None, f"{proj}_packed is None"
            assert scales is not None
            assert gs_buf is not None
            # global scale should be 1.0 (absorbed into block scales)
            gs_val = module._recover_tensor(gs_buf, torch.float32)
            assert torch.allclose(gs_val, torch.ones_like(gs_val)), \
                f"{proj} global_scale not 1.0 after absorption"

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048)])
    def test_dequant_shape_and_dtype(self, E, N, K):
        """Dequantized weights have correct shape [E, K, N] and bfloat16 dtype."""
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K)

        module = NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)

        w = module.gate_proj  # [E, K, N] bf16
        assert w.shape == (E, K, N), f"wrong shape: {w.shape}"
        assert w.dtype == torch.bfloat16


class TestBlockFP8LoadExperts:
    """BlockFP8QLoRAMoeExperts._load_experts produces correct GKN buffers."""

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048), (8, 512, 1024), (1, 256, 512)])
    def test_packed_matches_reference(self, E, N, K, block_size=128):
        """Packed fp8 buffer matches per-expert CPU-transpose reference."""
        fp8_list, scales_list = _make_block_fp8_tensors(E, N, K, block_size)

        module = BlockFP8QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_block_fp8_load_experts(module, fp8_list, scales_list)

        ref_packed, _ = _ref_load_block_fp8(fp8_list, scales_list, DEVICE)

        got = module.gate_packed
        assert got.shape == ref_packed.shape
        assert torch.equal(got, ref_packed), "fp8 packed bytes differ from reference"

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048), (8, 512, 1024)])
    def test_scales_match_reference(self, E, N, K, block_size=128):
        """Block scales buffer matches per-expert CPU-transpose reference."""
        fp8_list, scales_list = _make_block_fp8_tensors(E, N, K, block_size)

        module = BlockFP8QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_block_fp8_load_experts(module, fp8_list, scales_list)

        _, ref_scales = _ref_load_block_fp8(fp8_list, scales_list, DEVICE)

        got = module.gate_block_scales
        assert got.shape == ref_scales.shape
        assert torch.equal(got, ref_scales), "block scales differ from reference"

    @pytest.mark.parametrize("E,N,K", [(4, 768, 2048)])
    def test_all_projections_loaded(self, E, N, K):
        """All three projections are loaded."""
        fp8_list, scales_list = _make_block_fp8_tensors(E, N, K)

        module = BlockFP8QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_block_fp8_load_experts(module, fp8_list, scales_list)

        for proj in ("gate", "up", "down"):
            assert getattr(module, f"{proj}_packed") is not None
            assert getattr(module, f"{proj}_block_scales") is not None


# ---------------------------------------------------------------------------
# Benchmark (not a pytest test — run directly or with -s -k bench)
# ---------------------------------------------------------------------------

def _bench_nvfp4_load(E=128, N=768, K=2048, n_reps=5):
    """Benchmark NvFP4 expert loading: new GPU-transpose vs old CPU-transpose."""
    packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K)

    def make_module():
        return NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )

    # New implementation (GPU-transpose)
    times_new = []
    for _ in range(n_reps + 1):
        m = make_module()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_nvfp4_load_experts(m, packed_list, bs_list, gs_list)
        torch.cuda.synchronize()
        times_new.append(time.perf_counter() - t0)
    times_new = sorted(times_new[1:])

    # Reference implementation (CPU-transpose)
    times_ref = []
    for _ in range(n_reps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ref_load_nvfp4(packed_list, bs_list, gs_list, DEVICE)
        torch.cuda.synchronize()
        times_ref.append(time.perf_counter() - t0)
    times_ref = sorted(times_ref[1:])

    med_new = times_new[len(times_new) // 2] * 1000
    med_ref = times_ref[len(times_ref) // 2] * 1000
    speedup = med_ref / med_new
    print(f"\nNvFP4 _load_experts benchmark (E={E}, N={N}, K={K}, 1 proj):")
    print(f"  New (GPU-transpose) : {med_new:.1f}ms")
    print(f"  Ref (CPU-transpose) : {med_ref:.1f}ms")
    print(f"  Speedup             : {speedup:.2f}x")
    return speedup


if __name__ == "__main__":
    print("Running correctness checks...")

    for E, N, K in [(4, 768, 2048), (8, 512, 1024), (1, 256, 512)]:
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(E, N, K)
        module = NvFP4QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)
        ref_p, ref_s, _ = _ref_load_nvfp4(packed_list, bs_list, gs_list, DEVICE)
        assert torch.equal(module.gate_packed, ref_p), f"NvFP4 packed mismatch E={E}"
        assert torch.equal(module.gate_block_scales, ref_s), f"NvFP4 scales mismatch E={E}"
        print(f"  NvFP4 E={E} N={N} K={K}: OK")

    for E, N, K in [(4, 768, 2048), (8, 512, 1024)]:
        fp8_list, scales_list = _make_block_fp8_tensors(E, N, K)
        module = BlockFP8QLoRAMoeExperts(
            num_local_experts=E, num_experts=E,
            intermediate_size=N, hidden_size=K,
            r=4, lora_alpha=4, device=DEVICE,
        )
        _run_block_fp8_load_experts(module, fp8_list, scales_list)
        ref_p, ref_s = _ref_load_block_fp8(fp8_list, scales_list, DEVICE)
        assert torch.equal(module.gate_packed, ref_p), f"BlockFP8 packed mismatch E={E}"
        assert torch.equal(module.gate_block_scales, ref_s), f"BlockFP8 scales mismatch E={E}"
        print(f"  BlockFP8 E={E} N={N} K={K}: OK")

    print("\nRunning benchmark...")
    _bench_nvfp4_load(E=128, N=768, K=2048, n_reps=5)
