"""Distributed EP checkpoint slicing smoke test for DeepSeek/Kimi."""

import math
import os

import torch
import torch.distributed as dist

from xorl.models.transformers.deepseek_v3.checkpoint_handler import DeepseekV3CheckpointHandler


NUM_EXPERTS = 4


def _expert_weight(expert_idx: int, proj: str) -> torch.Tensor:
    hidden_size = 2
    intermediate_size = 3
    value = float(expert_idx * 10 + {"gate": 1, "up": 2, "down": 3}[proj])
    if proj == "down":
        return torch.full((hidden_size, intermediate_size), value)
    return torch.full((intermediate_size, hidden_size), value)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.int8:
        raise ValueError(f"Expected int8 values to pack, got {values.dtype}")
    if values.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor to pack, got {tuple(values.shape)}")

    num_bits = 4
    pack_factor = 32 // num_bits
    unsigned = (values + (1 << (num_bits - 1))).to(torch.uint8)
    pad_cols = (-values.shape[1]) % pack_factor
    if pad_cols:
        unsigned = torch.nn.functional.pad(unsigned, (0, pad_cols))
    reshaped = unsigned.view(values.shape[0], -1, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, dtype=torch.int32) * num_bits
    return (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)


def _packed_expert_weight(expert_idx: int, proj: str) -> dict[str, torch.Tensor]:
    dense_weight = _expert_weight(expert_idx, proj)
    quantized = torch.ones_like(dense_weight, dtype=torch.int8)
    num_groups = max(1, math.ceil(dense_weight.shape[1] / 32))
    scales = torch.full((dense_weight.shape[0], num_groups), dense_weight.flatten()[0].item(), dtype=torch.float32)
    return {
        "weight_packed": _pack_int4(quantized),
        "weight_scale": scales,
        "weight_shape": torch.tensor(dense_weight.shape, dtype=torch.int64),
    }


def _setup():
    dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def _load_local_expert_slice(rank: int, world_size: int) -> dict:
    handler = DeepseekV3CheckpointHandler(num_experts=NUM_EXPERTS, ep_rank=rank, ep_size=world_size)
    skip_key = handler.get_skip_key_fn()
    loaded = {}

    for expert_idx in range(NUM_EXPERTS):
        for proj in ("gate", "up", "down"):
            key = f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.weight"
            if skip_key is not None and skip_key(key):
                loaded.update(handler.on_skip_weight(key))
            else:
                loaded.update(handler.on_load_weight(key, _expert_weight(expert_idx, proj)))

    gate_up = loaded["model.layers.0.mlp.experts.gate_up_proj"]
    down = loaded["model.layers.0.mlp.experts.down_proj"]

    return {
        "rank": rank,
        "gate_up_shape": tuple(gate_up.shape),
        "down_shape": tuple(down.shape),
        "gate_ids": gate_up[:, 0, 0].tolist(),
        "down_ids": down[:, 0, 0].tolist(),
    }


def _load_local_packed_expert_slice(rank: int, world_size: int) -> dict:
    handler = DeepseekV3CheckpointHandler(num_experts=NUM_EXPERTS, ep_rank=rank, ep_size=world_size)
    skip_key = handler.get_skip_key_fn()
    loaded = {}

    for expert_idx in range(NUM_EXPERTS):
        for proj in ("gate", "up", "down"):
            for suffix, tensor in _packed_expert_weight(expert_idx, proj).items():
                key = f"language_model.model.layers.0.mlp.experts.{expert_idx}.{proj}_proj.{suffix}"
                if skip_key is not None and skip_key(key):
                    loaded.update(handler.on_skip_weight(key))
                else:
                    loaded.update(handler.on_load_weight(key, tensor))

    gate_up = loaded["model.layers.0.mlp.experts.gate_up_proj"]
    down = loaded["model.layers.0.mlp.experts.down_proj"]

    return {
        "rank": rank,
        "gate_up_shape": tuple(gate_up.shape),
        "down_shape": tuple(down.shape),
        "gate_ids": gate_up[:, 0, 0].tolist(),
        "down_ids": down[:, 0, 0].tolist(),
    }


def main():
    rank, world_size = _setup()
    assert world_size == 2, f"Expected 2 ranks, got {world_size}"

    summary = _load_local_expert_slice(rank, world_size)
    packed_summary = _load_local_packed_expert_slice(rank, world_size)
    gathered = [None] * world_size
    packed_gathered = [None] * world_size
    dist.all_gather_object(gathered, summary)
    dist.all_gather_object(packed_gathered, packed_summary)

    if rank == 0:
        gathered = sorted(gathered, key=lambda item: item["rank"])
        assert gathered[0]["gate_up_shape"] == (2, 2, 6)
        assert gathered[0]["down_shape"] == (2, 3, 2)
        assert gathered[0]["gate_ids"] == [1.0, 11.0]
        assert gathered[0]["down_ids"] == [3.0, 13.0]
        assert gathered[1]["gate_ids"] == [21.0, 31.0]
        assert gathered[1]["down_ids"] == [23.0, 33.0]

        packed_gathered = sorted(packed_gathered, key=lambda item: item["rank"])
        assert packed_gathered[0]["gate_up_shape"] == (2, 2, 6)
        assert packed_gathered[0]["down_shape"] == (2, 3, 2)
        assert packed_gathered[0]["gate_ids"] == [1.0, 11.0]
        assert packed_gathered[0]["down_ids"] == [3.0, 13.0]
        assert packed_gathered[1]["gate_ids"] == [21.0, 31.0]
        assert packed_gathered[1]["down_ids"] == [23.0, 33.0]
        print("DeepseekV3 EP checkpoint slicing passed")

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest

    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @pytest.mark.cpu
    @pytest.mark.distributed
    def test_deepseek_v3_ep_checkpoint_2proc():
        result = run_distributed_script(
            SCRIPT_PATH,
            num_gpus=2,
            timeout=120,
            extra_env={"PYTHONPATH": os.path.join(REPO_ROOT, "src")},
        )
        result.assert_success()


if __name__ == "__main__":
    main()
