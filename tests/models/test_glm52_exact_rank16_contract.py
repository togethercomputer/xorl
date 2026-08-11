from __future__ import annotations

import pytest
import torch

from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP
from xorl.models.transformers.glm5.exact_gate_up_qlora import (
    Glm52ExactTP1FusedGateUpBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_lm_head_qlora import (
    Glm52ExactTP16LmHeadSelectedLogprob,
    glm52_lm_head_shard,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.exact_shared_expert_qlora import (
    Glm52ExactTP16SharedExpertBlockFP8QLoRA,
)
from xorl.ops.block_fp8_native import NativeBlockFP8Linear


def test_rank16_alpha32_factor_shapes_cover_every_exact_glm_surface() -> None:
    device = torch.device("meta")
    dense = Glm52ExactTP1BlockFP8QLoRALinear(128, 256, r=16, lora_alpha=32, device=device)
    gate_up = Glm52ExactTP1FusedGateUpBlockFP8QLoRA(
        128, 128, r=16, lora_alpha=32, device=device
    )
    mlp = Glm52ExactTP1DenseMLP(128, 128, r=16, lora_alpha=32, device=device)
    absorbed = Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(
        r=16, lora_alpha=32, device=device
    )
    shared = Glm52ExactTP16SharedExpertBlockFP8QLoRA(
        r=16, lora_alpha=32, device=device
    )
    routed = Glm52ExactEP16BlockFP8QLoRARoutedExperts(
        6144, 2048, ep_rank=0, r=16, lora_alpha=32, device=device
    )
    shard = glm52_lm_head_shard(0)
    head = Glm52ExactTP16LmHeadSelectedLogprob(
        tp_rank=0,
        vocab_start=shard.vocab_start,
        vocab_end=shard.vocab_end,
        padded_vocab_start=shard.padded_vocab_start,
        padded_vocab_end=shard.padded_vocab_end,
        rank=16,
        lora_alpha=32,
    )

    assert dense.scaling == gate_up.scaling == mlp.scaling == 2.0
    assert absorbed.scaling == shared.scaling == routed.scaling == head.scaling == 2.0
    assert dense.lora_A.shape == (16, 128)
    assert dense.lora_B.shape == (256, 16)
    assert gate_up.gate_proj.lora_A.shape == (16, 128)
    assert gate_up.gate_proj.lora_B.shape == (128, 16)
    assert mlp.down_proj.lora_A.shape == (16, 128)
    assert absorbed.lora_A.shape == (16, 512)
    assert absorbed.lora_B.shape[-1] == 16
    assert shared.gate_proj.lora_A.shape == (16, 6144)
    assert shared.down_proj.lora_B.shape == (6144, 16)
    assert routed.gate_proj_lora_A.shape == (1, 6144, 16)
    assert routed.gate_proj_lora_B.shape == (256, 16, 2048)
    assert routed.down_proj_lora_A.shape == (256, 2048, 16)
    assert routed.down_proj_lora_B.shape == (1, 16, 6144)
    assert head.max_lora_rank == 16


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper CUDA")
def test_rank16_dense_exact_value_matches_public_sglang_kernels_byte_for_byte() -> None:
    pytest.importorskip("sglang")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the exact GLM-5.2 component requires Hopper")
    from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
    from sglang.kernels.ops.gemm.sgemm_lora_b import sgemm_lora_b_fwd
    from sglang.srt.lora.utils import LoRABatchInfo

    device = torch.device("cuda")
    rows = 16
    weight = (
        torch.arange(128 * 128, dtype=torch.float32, device=device)
        .remainder_(31)
        .sub_(15)
        .div_(16)
        .to(torch.float8_e4m3fn)
        .reshape(128, 128)
    )
    scales = torch.ones((1, 1), dtype=torch.float32, device=device)
    native = NativeBlockFP8Linear(128, 128, device=device)
    native.load_prequantized(weight, scales)
    exact = Glm52ExactTP1BlockFP8QLoRALinear(
        128, 128, r=16, lora_alpha=32, device=device
    )
    exact._source_fqn = "projection"
    exact._load_prequantized(lambda name: weight if name.endswith(".weight") else scales)
    with torch.no_grad():
        exact.lora_A.copy_(
            torch.arange(16 * 128, dtype=torch.float32, device=device)
            .remainder_(127)
            .sub_(63)
            .div_(256)
            .reshape(16, 128)
        )
        exact.lora_B.copy_(
            torch.arange(128 * 16, dtype=torch.float32, device=device)
            .remainder_(113)
            .sub_(56)
            .div_(256)
            .reshape(128, 16)
        )
    inputs = (
        torch.arange(rows * 128, dtype=torch.float32, device=device)
        .remainder_(97)
        .sub_(48)
        .div_(64)
        .reshape(rows, 128)
        .to(torch.bfloat16)
    )
    info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, rows], dtype=torch.int32, device=device),
        weight_indices=torch.zeros(1, dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([16], dtype=torch.int32, device=device),
        scalings=torch.tensor([2.0], dtype=torch.float32, device=device),
        max_len=rows,
        seg_lens=torch.tensor([rows], dtype=torch.int32, device=device),
        permutation=None,
        expected_tokens=rows,
        has_active_lora=True,
    )
    effective_A = exact.lora_A.to(torch.bfloat16).contiguous()
    effective_B = exact.lora_B.to(torch.bfloat16).contiguous()
    base = native(inputs)
    low = sgemm_lora_a_fwd(inputs, effective_A.unsqueeze(0), info)
    expected = sgemm_lora_b_fwd(
        low, effective_B.unsqueeze(0), info, base_output=base.clone()
    )
    actual = exact(inputs)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
