"""FSDP2 composition smoke for the scoped batch-invariant trunk-linear contract.

Verifies XORL_BI_TRUNK_LINEAR's module wrap composes with fully_shard (2 GPUs):
the sharded forward must be bit-identical to the unsharded wrapped module (same
persistent-GEMM kernel over the all-gathered bf16 params), gradients must be
finite, and the grad norm must match an unwrapped cuBLAS FSDP2 reference within
a bf16 forward-difference tolerance.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

from xorl.ops.sglang.batch_invariant_ops import wrap_trunk_linears_batch_invariant
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed]

HIDDEN, INTER = 256, 512


def _setup_dist() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    return torch.device("cuda", local_rank)


class TrunkBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.bfloat16)
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        y = self.o_proj(self.q_proj(x))
        return self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))


def _grad_norm(model) -> float:
    total = torch.zeros((), device="cuda", dtype=torch.float32)
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad
            total += g.float().pow(2).sum()
    dist.all_reduce(total)
    return total.sqrt().item()


def _run() -> None:
    device = _setup_dist()
    mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # Identical data on all ranks: forwards must then be identical too.
    torch.manual_seed(11)
    x = torch.randn(4, 32, HIDDEN, device=device, dtype=torch.bfloat16)
    g_out = torch.randn(4, 32, HIDDEN, device=device, dtype=torch.bfloat16)

    torch.manual_seed(7)
    model_ref = TrunkBlock().to(device)
    torch.manual_seed(7)
    model_bi = TrunkBlock().to(device)
    torch.manual_seed(7)
    model_local = TrunkBlock().to(device)  # unsharded wrapped reference

    wrapped = wrap_trunk_linears_batch_invariant(model_bi)
    assert sum(wrapped.values()) == 5, f"expected 5 wrapped trunk linears, got {wrapped}"
    wrap_trunk_linears_batch_invariant(model_local)

    for m in (model_ref, model_bi):
        fully_shard(m, mesh=mesh, mp_policy=mp_policy)

    out_bi = model_bi(x)
    with torch.no_grad():
        out_local = model_local(x)
    assert torch.equal(out_bi, out_local), "FSDP2-sharded wrapped forward must be bit-identical to the unsharded wrap"

    out_bi.backward(g_out)
    out_ref = model_ref(x)
    out_ref.backward(g_out)

    gn_bi = _grad_norm(model_bi)
    gn_ref = _grad_norm(model_ref)
    assert gn_bi > 0.0 and math.isfinite(gn_bi), f"unhealthy grad norm {gn_bi}"
    rel = abs(gn_bi - gn_ref) / max(gn_ref, 1e-6)
    assert rel < 0.05, f"grad norm diverged from cuBLAS FSDP2 reference: bi={gn_bi} ref={gn_ref} rel={rel}"

    for p in model_bi.parameters():
        g = p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad
        assert g is not None and torch.isfinite(g).all(), "non-finite gradient under FSDP2 + trunk wrap"

    if dist.get_rank() == 0:
        print(f"grad_norm bi={gn_bi:.6f} ref={gn_ref:.6f} rel={rel:.2e}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_bi_trunk_linear_composes_with_fsdp2():
        result = run_distributed_script(__file__, num_gpus=2, timeout=180)
        result.assert_success("XORL_BI_TRUNK_LINEAR wrap should compose with fully_shard")


if __name__ == "__main__":
    _run()
