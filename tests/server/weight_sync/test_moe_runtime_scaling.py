"""Regression tests for runtime-rank MoE LoRA weight sync."""

import pytest
import torch

from xorl.server.weight_sync.handler import WeightSyncHandler


pytestmark = [pytest.mark.cpu, pytest.mark.server]


class _FakeRuntimeRankMoeModule:
    def __init__(self) -> None:
        self.num_local_experts = 1
        self.hidden_size = 1
        self.intermediate_size = 1
        self.active_r = 2
        self.scaling = 4.0

    def _active_scaling(self) -> float:
        return 2.0

    def dequantize_expert(self, _proj_name: str, _expert_idx: int, K: int, N: int) -> torch.Tensor:
        return torch.zeros((K, N), dtype=torch.float32)


def _runtime_rank_lora_params() -> dict[str, torch.Tensor]:
    lora_A = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    lora_B = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]], dtype=torch.float32)
    return {
        "gate_proj_lora_A": lora_A.clone(),
        "gate_proj_lora_B": lora_B.clone(),
        "up_proj_lora_A": lora_A.clone(),
        "up_proj_lora_B": lora_B.clone(),
        "down_proj_lora_A": lora_A.clone(),
        "down_proj_lora_B": lora_B.clone(),
    }


def test_compute_moe_lora_delta_uses_active_runtime_scaling():
    mod = _FakeRuntimeRankMoeModule()
    params = _runtime_rank_lora_params()

    delta = WeightSyncHandler._compute_moe_lora_delta(
        mod,
        params["gate_proj_lora_A"],
        params["gate_proj_lora_B"],
        expert_idx=0,
    )

    assert delta.shape == (1, 1)
    assert delta.item() == pytest.approx(4.0)


def test_compute_moe_experts_buffer_uses_runtime_scaled_delta():
    mod = _FakeRuntimeRankMoeModule()
    handler = object.__new__(WeightSyncHandler)
    handler.rank = 0

    ctx = {
        "module": mod,
        "prefix": "model.layers.0.mlp.experts",
        "lora_params": _runtime_rank_lora_params(),
    }

    items = WeightSyncHandler._compute_moe_experts_buffer(handler, ctx)

    assert len(items) == 3
    assert [name for name, _ in items] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ]
    for _, tensor in items:
        assert tensor.dtype == torch.bfloat16
        assert tensor.shape == (1, 1)
        assert tensor.float().item() == pytest.approx(4.0)
    assert ctx["lora_params"] is None
