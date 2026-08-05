"""FSDP2 must not downcast the Class-B RoPE table at decoder boundaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard

from xorl.distributed.torch_parallelize import (
    _bf16_mixed_precision_policy,
    _decoder_bf16_mixed_precision_policy,
)
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed]


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.seen_hidden_dtype = None
        self.seen_rope_dtype = None

    def forward(self, hidden_states, position_embeddings):
        cos, sin = position_embeddings
        self.seen_hidden_dtype = hidden_states.dtype
        self.seen_rope_dtype = (cos.dtype, sin.dtype)
        return self.proj(hidden_states) + (cos[..., :1] + sin[..., :1]).to(hidden_states.dtype) * 0


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(8, 8, bias=False)
        self.decoder = _Decoder()

    def forward(self, x):
        hidden_states = self.input_proj(x)
        shape = (*hidden_states.shape[:-1], 8)
        position_embeddings = (
            torch.ones(shape, dtype=torch.float32, device=x.device),
            torch.zeros(shape, dtype=torch.float32, device=x.device),
        )
        return self.decoder(hidden_states, position_embeddings)


def _setup_dist() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    return torch.device("cuda", local_rank)


def _run() -> None:
    device = _setup_dist()
    mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))

    model = _Root().to(device)
    fully_shard(model.decoder, mesh=mesh, mp_policy=_decoder_bf16_mixed_precision_policy(class_b=True))
    fully_shard(model, mesh=mesh, mp_policy=_bf16_mixed_precision_policy())

    x = torch.randn((2, 4, 8), dtype=torch.float32, device=device, requires_grad=True)
    output = model(x)
    output.float().square().mean().backward()

    assert model.decoder.seen_hidden_dtype == torch.bfloat16
    assert model.decoder.seen_rope_dtype == (torch.float32, torch.float32)
    assert model.decoder.proj.weight.grad is not None
    assert torch.isfinite(model.decoder.proj.weight.grad.to_local()).all()

    control = _decoder_bf16_mixed_precision_policy(class_b=False)
    assert control.cast_forward_inputs is True, "control must retain stock FSDP input casting"

    dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(1)
    def test_class_b_table_survives_nested_fsdp_boundary():
        result = run_distributed_script(__file__, num_gpus=1, timeout=120)
        result.assert_success("Class-B RoPE table should remain fp32 across decoder FSDP")


if __name__ == "__main__":
    _run()
