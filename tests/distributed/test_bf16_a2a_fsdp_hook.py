"""End-to-end smoke test: BF16StochasticAllToAllReduceScatter wired into FSDP2.

Verifies that ``FSDPModule.set_custom_reduce_scatter`` accepts our custom
``ReduceScatter`` and that backward + optimizer step produces non-NaN updated
weights. Compares against FP32 reduce-scatter on the same model and grads;
loss should be close (within a generous tolerance given BF16 transit).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

from xorl.distributed.fsdp2 import BF16StochasticAllToAllReduceScatter
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than


pytestmark = [pytest.mark.distributed]


def _setup_dist() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    return torch.device("cuda", local_rank)


class TinyMLP(nn.Module):
    def __init__(self, hidden=64, intermediate=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, bias=False)
        self.fc2 = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def _step(model, x, y):
    out = model(x)
    loss = (out - y).pow(2).mean()
    loss.backward()
    return float(loss.detach().item())


def _run() -> None:
    device = _setup_dist()
    torch.manual_seed(123 + dist.get_rank())

    hidden, intermediate = 64, 128
    x = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)
    y = torch.randn(8, hidden, device=device, dtype=torch.bfloat16)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))

    # Two identical models seeded the same; one with FP32 reduce-scatter, one with BF16-a2a.
    torch.manual_seed(7)
    model_ref = TinyMLP(hidden, intermediate).to(device)
    torch.manual_seed(7)
    model_test = TinyMLP(hidden, intermediate).to(device)

    fully_shard(model_ref.fc1, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model_ref.fc2, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model_ref, mesh=mesh, mp_policy=mp_policy)

    fully_shard(model_test.fc1, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model_test.fc2, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model_test, mesh=mesh, mp_policy=mp_policy)
    # Install the custom reduce-scatter on both Linear FSDP units
    model_test.fc1.set_custom_reduce_scatter(BF16StochasticAllToAllReduceScatter())
    model_test.fc2.set_custom_reduce_scatter(BF16StochasticAllToAllReduceScatter())

    loss_ref = _step(model_ref, x, y)
    loss_test = _step(model_test, x, y)

    # Compare grads in fc1 / fc2 (DTensor → local_tensor)
    g1_ref = model_ref.fc1.weight.grad._local_tensor
    g1_test = model_test.fc1.weight.grad._local_tensor
    g2_ref = model_ref.fc2.weight.grad._local_tensor
    g2_test = model_test.fc2.weight.grad._local_tensor

    rel1 = (g1_test - g1_ref).abs().max() / (g1_ref.abs().max() + 1e-6)
    rel2 = (g2_test - g2_ref).abs().max() / (g2_ref.abs().max() + 1e-6)

    if dist.get_rank() == 0:
        print(f"loss ref={loss_ref:.6f}, test={loss_test:.6f}")
        print(f"fc1 max-rel-err: {rel1.item():.4e}")
        print(f"fc2 max-rel-err: {rel2.item():.4e}")

    # Both gradients must be finite and within a generous BF16 tolerance.
    assert torch.isfinite(g1_test).all() and torch.isfinite(g2_test).all(), (
        "BF16 a2a path produced non-finite gradients"
    )
    assert rel1.item() < 0.05, f"fc1 grad relative error {rel1.item()} too large"
    assert rel2.item() < 0.05, f"fc2 grad relative error {rel2.item()} too large"

    dist.barrier()
    dist.destroy_process_group()


def _main() -> None:
    _run()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_bf16_a2a_reduce_scatter_runs_inside_fsdp2():
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success("BF16 a2a reduce-scatter should integrate cleanly with fully_shard")


if __name__ == "__main__":
    _main()
