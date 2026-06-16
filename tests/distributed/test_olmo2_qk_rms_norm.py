"""``Olmo2QKRMSNorm`` regression test.

OLMo-2's full-axis ``q_norm``/``k_norm`` doesn't compose with stock TP
styles. Under colwise q/k_proj the input arrives hidden-sharded; the
plan wraps these norms with ``LocalAxisRMSNormShard`` to shard their
weight on dim 0. ``Olmo2QKRMSNorm.forward`` detects the Shard(0)
DTensor weight and runs the fused op on local tensors, computing a
local-axis RMS that matches HuggingFace's ``Olmo2RMSNorm`` reference
under TP.

Can be run two ways:
    1. pytest tests/distributed/test_olmo2_qk_rms_norm.py -v   (launches torchrun internally)
    2. torchrun --nproc_per_node=2 tests/distributed/test_olmo2_qk_rms_norm.py  (direct)
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from xorl.models.layers.normalization import RMSNorm
from xorl.models.transformers.olmo2.modeling_olmo2 import Olmo2QKRMSNorm
from xorl.models.transformers.olmo2.tp_styles import LocalAxisRMSNormShard


HIDDEN = 8
SEQ = 6
BATCH = 2


def _check_no_tp_passthrough():
    """Without TP, Olmo2QKRMSNorm forward delegates to the parent RMSNorm."""
    norm = Olmo2QKRMSNorm(HIDDEN)
    x = torch.randn(BATCH, SEQ, HIDDEN, generator=torch.Generator().manual_seed(0))
    out = norm(x)
    assert tuple(out.shape) == (BATCH, SEQ, HIDDEN)
    # Numerical equivalence with the parent class on the same input.
    ref = RMSNorm(HIDDEN)
    ref.weight = torch.nn.Parameter(norm.weight.detach().clone())
    torch.testing.assert_close(out, ref(x), atol=1e-6, rtol=1e-6)


def _check_local_axis_rms_norm_shard(mesh):
    """LocalAxisRMSNormShard + Olmo2QKRMSNorm: full-axis QK-norm path under TP.

    The colwise q/k_proj output arrives as a plain hidden-sharded tensor; the
    custom style shards the weight on dim 0 so each rank's slice matches its
    local input. The forward should compute a local-axis RMS matching a
    per-rank-local single-tensor ``RMSNorm`` applied to the same shard.
    """
    tp = mesh.size()
    rank = dist.get_rank()

    norm = parallelize_module(Olmo2QKRMSNorm(HIDDEN), mesh, LocalAxisRMSNormShard())

    # Mimic colwise q_proj output: same global tensor on every rank, then take
    # the rank's hidden slice as the plain (non-DTensor) input.
    full = torch.randn(BATCH, SEQ, HIDDEN, generator=torch.Generator().manual_seed(7))
    rank_slice = slice(rank * (HIDDEN // tp), (rank + 1) * (HIDDEN // tp))
    local_input = full[..., rank_slice].contiguous()

    out = norm(local_input)
    expected_shape = (BATCH, SEQ, HIDDEN // tp)
    assert tuple(out.shape) == expected_shape, f"expected output shape {expected_shape}, got {tuple(out.shape)}"

    # Reference: a plain (non-TP) RMSNorm with hidden=HIDDEN/tp and the local
    # weight slice. This is the local-axis RMS HF's OLMo-2 reference computes.
    ref_norm = RMSNorm(HIDDEN // tp)
    ref_norm.weight = torch.nn.Parameter(norm.weight.to_local().clone())
    ref_out = ref_norm(local_input)

    torch.testing.assert_close(out, ref_out, atol=1e-6, rtol=1e-6)


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert HIDDEN % world_size == 0 and SEQ % world_size == 0, (
        "Test fixtures require HIDDEN and SEQ divisible by world_size"
    )

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    _check_no_tp_passthrough()
    _check_local_axis_rms_norm_shard(mesh)

    if rank == 0:
        print("All Olmo2QKRMSNorm checks passed!")

    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest

    from tests.distributed.distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)

    @pytest.mark.cpu
    @pytest.mark.distributed
    def test_olmo2_qk_rms_norm_2rank_cpu():
        """Olmo2QKRMSNorm + LocalAxisRMSNormShard on a 2-rank gloo mesh."""
        result = run_distributed_script(SCRIPT_PATH, num_gpus=2, timeout=120)
        result.assert_success()


if __name__ == "__main__":
    main()
