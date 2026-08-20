"""Correctness contracts for pre-quantized MoE expert weight loading."""

import pytest
import torch

from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX
from xorl.qlora.modules.moe_experts import BlockFP8QLoRAMoeExperts, NvFP4QLoRAMoeExperts


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = torch.device("cuda")


def _ref_load_nvfp4(packed_hf_list, bs_list, gs_list, device):
    """Per-expert CPU-transpose reference for NVFP4."""
    packed_out, scales_out, amax_out = [], [], []
    for packed, bs, gs in zip(packed_hf_list, bs_list, gs_list):
        gs_val = gs.float().item()
        packed_gkn = packed.T.contiguous()
        bs_gkn = (bs.float() * gs.float()).T.contiguous()
        packed_out.append(packed_gkn.view(torch.uint8))
        scales_out.append(bs_gkn.view(torch.uint8))
        amax_out.append(gs_val * FP4_E2M1_MAX * FP8_E4M3_MAX)

    return torch.stack(packed_out).to(device), torch.stack(scales_out).to(device), amax_out


def _ref_load_block_fp8(fp8_list, scales_list, device):
    """Per-expert CPU-transpose reference for block FP8."""
    packed_out, scales_out = [], []
    for fp8_w, scales in zip(fp8_list, scales_list):
        packed_out.append(fp8_w.T.contiguous().view(torch.uint8))
        scales_out.append(scales.float().T.contiguous().view(torch.uint8))

    return torch.stack(packed_out).to(device), torch.stack(scales_out).to(device)


def _make_nvfp4_tensors(num_experts, intermediate_size, hidden_size, block_size=16, seed=0):
    torch.manual_seed(seed)
    packed_list, bs_list, gs_list = [], [], []
    for _ in range(num_experts):
        packed_list.append(torch.randint(0, 256, (intermediate_size, hidden_size // 2), dtype=torch.uint8))
        bs_list.append(torch.rand(intermediate_size, hidden_size // block_size).to(torch.float8_e4m3fn))
        gs_list.append(torch.tensor(0.001 + torch.rand(1).item() * 0.01))
    return packed_list, bs_list, gs_list


def _make_block_fp8_tensors(num_experts, intermediate_size, hidden_size, block_size=128, seed=0):
    torch.manual_seed(seed)
    fp8_list, scales_list = [], []
    for _ in range(num_experts):
        fp8_list.append(torch.rand(intermediate_size, hidden_size).to(torch.float8_e4m3fn))
        scales_list.append(
            torch.rand(
                intermediate_size // block_size,
                hidden_size // block_size,
                dtype=torch.float32,
            )
        )
    return fp8_list, scales_list


def _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list):
    data = {}
    for hf_name in ("gate_proj", "up_proj", "down_proj"):
        for expert_idx in range(len(packed_list)):
            fqn = f"layer.{expert_idx}.{hf_name}"
            data[f"{fqn}.weight"] = packed_list[expert_idx]
            data[f"{fqn}.weight_scale"] = bs_list[expert_idx]
            data[f"{fqn}.weight_scale_2"] = gs_list[expert_idx]

    module._source_fqn = "layer"
    module.expert_offset = 0
    module._load_experts(lambda key: data[key], {})


def _run_block_fp8_load_experts(module, fp8_list, scales_list):
    data = {}
    for hf_name in ("gate_proj", "up_proj", "down_proj"):
        for expert_idx in range(len(fp8_list)):
            fqn = f"layer.{expert_idx}.{hf_name}"
            data[f"{fqn}.weight"] = fp8_list[expert_idx]
            data[f"{fqn}.weight_scale_inv"] = scales_list[expert_idx]

    module._source_fqn = "layer"
    module.expert_offset = 0
    module._load_experts(lambda key: data[key], {})


def test_prequantized_load_experts_matches_format_references():
    """One load must populate bytes, scales, amax, projections, and dequantization."""
    for num_experts, intermediate_size, hidden_size in (
        (4, 768, 2048),
        (8, 512, 1024),
        (1, 256, 512),
    ):
        packed_list, bs_list, gs_list = _make_nvfp4_tensors(
            num_experts,
            intermediate_size,
            hidden_size,
        )
        module = NvFP4QLoRAMoeExperts(
            num_local_experts=num_experts,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            r=4,
            lora_alpha=4,
            device=DEVICE,
        )
        _run_nvfp4_load_experts(module, packed_list, bs_list, gs_list)
        ref_packed, ref_scales, ref_amax = _ref_load_nvfp4(packed_list, bs_list, gs_list, DEVICE)

        for projection in ("gate", "up", "down"):
            packed = getattr(module, f"{projection}_packed")
            scales = getattr(module, f"{projection}_block_scales")
            assert packed.shape == ref_packed.shape
            assert packed.dtype == ref_packed.dtype
            assert torch.equal(packed, ref_packed), projection
            assert scales.shape == ref_scales.shape
            assert torch.equal(scales, ref_scales), projection
            global_scale = module._recover_tensor(
                getattr(module, f"{projection}_global_scale"),
                torch.float32,
            )
            assert torch.equal(global_scale, torch.ones_like(global_scale)), projection
            for actual, expected in zip(module._ema_amax[projection].tolist(), ref_amax):
                assert abs(actual - expected) < 1e-5, projection

        if num_experts == 4:
            dequantized = module.gate_proj
            assert dequantized.shape == (num_experts, hidden_size, intermediate_size)
            assert dequantized.dtype == torch.bfloat16

    _assert_block_fp8_load_experts_matches_reference()


def _assert_block_fp8_load_experts_matches_reference():
    """One load must populate packed bytes and scales for every projection."""
    for num_experts, intermediate_size, hidden_size in (
        (4, 768, 2048),
        (8, 512, 1024),
        (1, 256, 512),
    ):
        fp8_list, scales_list = _make_block_fp8_tensors(
            num_experts,
            intermediate_size,
            hidden_size,
        )
        module = BlockFP8QLoRAMoeExperts(
            num_local_experts=num_experts,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            r=4,
            lora_alpha=4,
            device=DEVICE,
        )
        _run_block_fp8_load_experts(module, fp8_list, scales_list)
        ref_packed, ref_scales = _ref_load_block_fp8(fp8_list, scales_list, DEVICE)

        for projection in ("gate", "up", "down"):
            packed = getattr(module, f"{projection}_packed")
            scales = getattr(module, f"{projection}_block_scales")
            assert packed.shape == ref_packed.shape
            assert torch.equal(packed, ref_packed), projection
            assert scales.shape == ref_scales.shape
            assert torch.equal(scales, ref_scales), projection
