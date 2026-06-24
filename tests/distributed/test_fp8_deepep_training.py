"""Real DeepEP FP8 training smoke for the Quack no-permute path.

The pytest wrapper launches this file under torchrun. The worker path exercises
DeepEP dispatch/combine, Quack FP8 grouped expert GEMMs, per-expert biases, and
backward on two local GPUs.
"""

import os
import sys
from ctypes import CDLL
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _prepend_library_path(path: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if path not in existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path


def _install_nvidia_ml_library_path() -> None:
    try:
        CDLL("libnvidia-ml.so.1")
        return
    except OSError:
        pass

    for stub in (
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"),
        Path("/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"),
    ):
        if not stub.exists():
            continue
        stub_dir = Path("/tmp/xorl-nvidia-ml-stub")
        stub_dir.mkdir(exist_ok=True)
        soname = stub_dir / "libnvidia-ml.so.1"
        if not soname.exists():
            soname.symlink_to(stub)
        _prepend_library_path(str(stub_dir))
        return


def _install_nvshmem_library_path() -> None:
    try:
        import nvidia.nvshmem  # noqa: PLC0415

        nvshmem_lib = os.path.join(list(nvidia.nvshmem.__path__)[0], "lib")
        _prepend_library_path(nvshmem_lib)
    except Exception:
        pass


def _check(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        raise RuntimeError(f"{name} is None")
    if not torch.isfinite(tensor.float()).all():
        raise RuntimeError(f"{name} has non-finite values")


def _worker_main() -> int:
    _install_nvidia_ml_library_path()
    _install_nvshmem_library_path()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size != 2:
        raise RuntimeError(f"expected world_size=2 for this smoke, got {world_size}")

    from xorl.distributed.moe.deepep import DeepEPBuffer, token_pre_dispatch_no_permute  # noqa: PLC0415
    from xorl.ops.moe.quack import QuackEPDeepEPNoPermute  # noqa: PLC0415

    ep_group = dist.new_group(list(range(world_size)))
    num_experts = world_size
    hidden_dim = 128
    intermediate_size = 128
    num_tokens = 8

    torch.manual_seed(1234 + rank)
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    selected_experts = torch.arange(num_tokens, device=device, dtype=torch.long).remainder(num_experts).view(-1, 1)
    routing_weights = torch.ones(num_tokens, 1, device=device, dtype=torch.bfloat16)

    gate_up_proj = nn.Parameter(
        torch.randn(1, hidden_dim, 2 * intermediate_size, device=device, dtype=torch.bfloat16) * 0.02
    )
    down_proj = nn.Parameter(torch.randn(1, intermediate_size, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02)
    gate_up_bias = nn.Parameter(torch.randn(1, 2 * intermediate_size, device=device, dtype=torch.bfloat16) * 0.01)
    down_bias = nn.Parameter(torch.randn(1, hidden_dim, device=device, dtype=torch.bfloat16) * 0.01)
    buffer = DeepEPBuffer(ep_group=ep_group, buffer_size_gb=0.25, num_sms=20)

    try:
        recv_x, cumsum, ctx = token_pre_dispatch_no_permute(
            buffer=buffer,
            hidden_states=hidden_states,
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            num_experts=num_experts,
            num_local_experts=1,
        )
        expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
        output = QuackEPDeepEPNoPermute.apply(
            recv_x,
            cumsum,
            gate_up_proj,
            down_proj,
            intermediate_size,
            expert_scores,
            buffer,
            ctx,
            False,  # async_combine
            "clamped_swiglu",  # hidden_act
            True,  # activation_native
            True,  # fp8_compute
            "triton_grouped",  # fp8_grouped_backend
            128,  # fp8_block_size
            gate_up_bias,
            down_bias,
        )
        loss = output.float().pow(2).mean()
        loss.backward()

        _check("output", output)
        _check("hidden_states.grad", hidden_states.grad)
        _check("gate_up_proj.grad", gate_up_proj.grad)
        _check("down_proj.grad", down_proj.grad)
        _check("gate_up_bias.grad", gate_up_bias.grad)
        _check("down_bias.grad", down_bias.grad)

        ok = torch.tensor(1, device=device, dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            print("fp8_deepep_training_smoke_ok", flush=True)
    finally:
        buffer.destroy_buffer()
        dist.destroy_process_group()

    return 0


@skip_if_gpu_count_less_than(2)
def test_fp8_deepep_no_permute_training_smoke():
    pytest.importorskip("deep_ep")
    pytest.importorskip("nvidia.nvshmem")
    _install_nvidia_ml_library_path()

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=180,
        extra_env={
            "XORL_TEST_FP8_DEEPEP_WORKER": "1",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
    )
    result.assert_success()
    assert "fp8_deepep_training_smoke_ok" in result.stdout


if __name__ == "__main__" and os.environ.get("XORL_TEST_FP8_DEEPEP_WORKER") == "1":
    sys.exit(_worker_main())
