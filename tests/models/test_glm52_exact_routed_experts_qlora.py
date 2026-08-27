from __future__ import annotations

import pytest
import torch

from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION,
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
    localize_glm52_ep16_expert_ids,
)
from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts


_HIDDEN = 128
_INTERMEDIATE = 128
_GLOBAL_EXPERTS = 256
_EP_SIZE = 16
_LOCAL_EXPERTS = 16


def _module(owner: int, device: torch.device | str = "cpu") -> Glm52ExactEP16BlockFP8QLoRARoutedExperts:
    return Glm52ExactEP16BlockFP8QLoRARoutedExperts(
        _HIDDEN,
        _INTERMEDIATE,
        ep_rank=owner,
        device=device,
    )


def _fill_distinguishable_factors(module: Glm52ExactEP16BlockFP8QLoRARoutedExperts) -> None:
    owner = module.ep_rank
    global_ids = torch.arange(_GLOBAL_EXPERTS, dtype=torch.float32, device=module.gate_proj_lora_A.device)
    with torch.no_grad():
        module.gate_proj_lora_A.fill_((owner + 1) / 512)
        module.up_proj_lora_A.fill_((owner + 17) / 1024)
        module.down_proj_lora_B.fill_((owner + 3) / 256)
        module.gate_proj_lora_B.copy_(((global_ids + 1) / 4096).view(_GLOBAL_EXPERTS, 1, 1))
        module.up_proj_lora_B.copy_(((global_ids + 257) / 4096).view(_GLOBAL_EXPERTS, 1, 1))
        module.down_proj_lora_A.copy_(((global_ids + 33) / 8192).view(_GLOBAL_EXPERTS, 1, 1))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint16)


def test_routed_bank_topology_remap_and_physical_buffer_policy() -> None:
    module = _module(7)

    assert isinstance(module, Glm52NativeBlockFP8Experts)
    assert module.contract_version == GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION
    assert module._glm52_exact_active_lora_component is True
    assert module.fsdp_requires_full_precision is True
    assert module.adapter_gradient_producer_family == "module_managed"
    assert (module.num_experts, module.ep_size, module.num_local_experts, module.moe_tp_size) == (256, 16, 16, 1)
    assert (module.ep_rank, module.expert_offset, module.ep_dispatch) == (7, 112, "alltoall")
    assert module.r == module.active_r == module.lora_alpha == module.active_lora_alpha == 1
    assert module.scaling == 1.0
    assert {name for name, parameter in module.named_parameters() if parameter.requires_grad} == set(
        module.logical_factor_names
    )
    assert all(getattr(module, name).dtype is torch.float32 for name in module.logical_factor_names)
    assert tuple(module.gate_proj_lora_A.shape) == (1, _HIDDEN, 1)
    assert tuple(module.gate_proj_lora_B.shape) == (_GLOBAL_EXPERTS, 1, _INTERMEDIATE)
    assert tuple(module.up_proj_lora_A.shape) == (1, _HIDDEN, 1)
    assert tuple(module.up_proj_lora_B.shape) == (_GLOBAL_EXPERTS, 1, _INTERMEDIATE)
    assert tuple(module.down_proj_lora_A.shape) == (_GLOBAL_EXPERTS, _INTERMEDIATE, 1)
    assert tuple(module.down_proj_lora_B.shape) == (1, 1, _HIDDEN)

    rank_three = Glm52ExactEP16BlockFP8QLoRARoutedExperts(
        _HIDDEN, _INTERMEDIATE, ep_rank=0, r=3, lora_alpha=7, device="meta"
    )
    assert rank_three.gate_proj_lora_A.shape[-1] == 3
    assert rank_three.gate_proj_lora_B.shape[1] == 3
    with pytest.raises(ValueError, match="256 global experts"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            _HIDDEN,
            _INTERMEDIATE,
            ep_rank=0,
            num_experts=128,
        )
    with pytest.raises(ValueError, match="MoE-TP1"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(_HIDDEN, _INTERMEDIATE, ep_rank=0, moe_tp_size=2)
    with pytest.raises(ValueError, match="DeepEP is not admitted"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            _HIDDEN,
            _INTERMEDIATE,
            ep_rank=0,
            ep_dispatch="deepep",
        )
    with pytest.raises(ValueError, match=r"in \[0, 15\]"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(_HIDDEN, _INTERMEDIATE, ep_rank=16)
    module.set_runtime_lora_config(1, 1)
    with pytest.raises(ValueError, match="positive integer rank"):
        Glm52ExactEP16BlockFP8QLoRARoutedExperts(_HIDDEN, _INTERMEDIATE, ep_rank=0, r=0, device="meta")
    with pytest.raises(ValueError, match="positive integer alpha"):
        module.set_runtime_lora_config(1, 0)

    _assert_one_batched_global_grid_proves_all_16_owner_by_16_slot_remaps()
    _assert_physical_sampler_factor_buffer_policy()


def _assert_one_batched_global_grid_proves_all_16_owner_by_16_slot_remaps() -> None:
    global_grid = torch.arange(_GLOBAL_EXPERTS, dtype=torch.int64).reshape(_EP_SIZE, _LOCAL_EXPERTS)
    owner_maps = torch.stack(
        [localize_glm52_ep16_expert_ids(global_grid.contiguous(), owner) for owner in range(_EP_SIZE)]
    )
    expected = torch.full((_EP_SIZE, _EP_SIZE, _LOCAL_EXPERTS), -1, dtype=torch.int32)
    slots = torch.arange(_LOCAL_EXPERTS, dtype=torch.int32)
    for owner in range(_EP_SIZE):
        expected[owner, owner] = slots
    assert torch.equal(owner_maps, expected)

    with pytest.raises(TypeError, match="must be int32 or int64"):
        localize_glm52_ep16_expert_ids(global_grid.float().contiguous(), 0)
    with pytest.raises(ValueError, match="contiguous"):
        localize_glm52_ep16_expert_ids(global_grid.T, 0)
    invalid = global_grid.clone()
    invalid[0, 0] = _GLOBAL_EXPERTS
    with pytest.raises(ValueError, match=r"in \[0, 255\]"):
        localize_glm52_ep16_expert_ids(invalid, 0)


def _assert_physical_sampler_factor_buffer_policy() -> None:
    for owner in range(_EP_SIZE):
        module = _module(owner)
        _fill_distinguishable_factors(module)
        effective = tuple(getattr(module, name).to(torch.bfloat16).contiguous() for name in module.logical_factor_names)
        buffers = module._physical_factor_buffers(*effective)
        assert tuple(buffers["gate_up_lora_a_weights"].shape) == (8, 1, 2, _HIDDEN)
        assert tuple(buffers["gate_up_lora_b_weights"].shape) == (8, 16, 2 * _INTERMEDIATE, 1)
        assert tuple(buffers["down_lora_a_weights"].shape) == (8, 16, 1, _INTERMEDIATE)
        assert tuple(buffers["down_lora_b_weights"].shape) == (8, 1, _HIDDEN, 1)
        assert all(buffer.dtype is torch.bfloat16 for buffer in buffers.values())
        assert torch.equal(
            _bits(buffers["gate_up_lora_a_weights"][0, 0, 0]),
            _bits(module.gate_proj_lora_A[:, :, 0].to(torch.bfloat16)[0]),
        )
        assert torch.equal(
            _bits(buffers["gate_up_lora_a_weights"][0, 0, 1]),
            _bits(module.up_proj_lora_A[:, :, 0].to(torch.bfloat16)[0]),
        )
        for slot in range(_LOCAL_EXPERTS):
            global_expert = module.expert_offset + slot
            assert torch.equal(
                _bits(buffers["gate_up_lora_b_weights"][0, slot, :_INTERMEDIATE]),
                _bits(module.gate_proj_lora_B[global_expert].T.to(torch.bfloat16)),
            )
            assert torch.equal(
                _bits(buffers["gate_up_lora_b_weights"][0, slot, _INTERMEDIATE:]),
                _bits(module.up_proj_lora_B[global_expert].T.to(torch.bfloat16)),
            )
            assert torch.equal(
                _bits(buffers["down_lora_a_weights"][0, slot]),
                _bits(module.down_proj_lora_A[global_expert].T.to(torch.bfloat16)),
            )
        assert torch.equal(
            _bits(buffers["down_lora_b_weights"][0, 0]),
            _bits(module.down_proj_lora_B[0].T.to(torch.bfloat16)),
        )
        for buffer in buffers.values():
            assert torch.count_nonzero(buffer[1:]) == 0

    _assert_post_ep_owner_local_factor_banks_produce_same_views()


def _assert_post_ep_owner_local_factor_banks_produce_same_views() -> None:
    module = _module(7)
    _fill_distinguishable_factors(module)
    global_factors = tuple(
        getattr(module, name).detach().to(torch.bfloat16).contiguous() for name in module.logical_factor_names
    )
    local_factors = list(global_factors)
    for index in (1, 3, 4):
        local_factors[index] = global_factors[index][112:128].contiguous()

    global_buffers = module._physical_factor_buffers(*global_factors)
    local_buffers = module._physical_factor_buffers(*local_factors)

    assert global_buffers.keys() == local_buffers.keys()
    for name in global_buffers:
        assert torch.equal(global_buffers[name], local_buffers[name]), name
