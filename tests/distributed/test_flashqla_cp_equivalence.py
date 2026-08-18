"""Two-rank forward and gradient parity for the FlashQLA CP bridge."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


# The FlashQLA bridge pulls in TileLang, which resolves a CUDA target at import
# time and raises when no GPU is visible. This is a two-rank GPU parity test, so
# skip the module outright rather than fail collection on a CPU-only host.
if not torch.cuda.is_available():
    pytest.skip("FlashQLA CP parity requires CUDA", allow_module_level=True)

from xorl.distributed.parallel_state import init_parallel_state  # noqa: E402
from xorl.ops.linear_attention.flashqla_cp import flashqla_chunk_gated_delta_rule_cp  # noqa: E402
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context  # noqa: E402
from xorl.ops.linear_attention.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
from xorl.utils.device import get_nccl_backend  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()


def _setup() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_rank{local_rank}")
    os.environ.setdefault("TILELANG_CACHE_DIR", f"/tmp/tilelang_cache_rank{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        ulysses_size=world_size,
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    return torch.device("cuda", local_rank)


def _make_inputs(device: torch.device, total_tokens: int):
    generator = torch.Generator(device=device).manual_seed(7)
    shape = (1, total_tokens, 4, 128)
    q = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    g = -torch.rand(shape[:-1], generator=generator, device=device, dtype=torch.float32)
    beta = torch.rand(shape[:-1], generator=generator, device=device, dtype=torch.float32)
    return q, k, v, g, beta


def _differentiable_copy(values):
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


def _run_distributed_gradcheck() -> None:
    device = _setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_tokens = 128
    total_tokens = local_tokens * world_size
    full_inputs = _make_inputs(device, total_tokens)
    cu_seqlens = torch.tensor([0, total_tokens], device=device, dtype=torch.int32)

    reference_inputs = _differentiable_copy(full_inputs)
    reference_output, _ = chunk_gated_delta_rule(
        *reference_inputs,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    reference_output.float().square().sum().backward()

    token_slice = slice(rank * local_tokens, (rank + 1) * local_tokens)
    local_inputs = _differentiable_copy(tuple(value[:, token_slice] for value in full_inputs))
    cp_context = build_linear_attention_cp_context(cu_seqlens)
    assert cp_context is not None
    cp_output, _ = flashqla_chunk_gated_delta_rule_cp(
        *local_inputs,
        use_qk_l2norm_in_kernel=True,
        cp_context=cp_context,
    )
    cp_output.float().square().sum().backward()

    gathered_outputs = [torch.empty_like(cp_output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, cp_output.detach())
    actual_output = torch.cat(gathered_outputs, dim=1)
    assert torch.isfinite(actual_output).all()
    assert _cosine(reference_output.detach(), actual_output) > 0.99

    for name, reference, local in zip(("q", "k", "v", "g", "beta"), reference_inputs, local_inputs):
        assert local.grad is not None
        gathered_gradients = [torch.empty_like(local.grad) for _ in range(world_size)]
        dist.all_gather(gathered_gradients, local.grad)
        actual_gradient = torch.cat(gathered_gradients, dim=1)
        assert torch.isfinite(actual_gradient).all(), name
        cosine = _cosine(reference.grad, actual_gradient)
        assert cosine > 0.97, f"{name} gradient cosine too low: {cosine}"


def _main() -> None:
    try:
        _run_distributed_gradcheck()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_flashqla_cp_matches_fla_forward_and_all_input_gradients():
        if torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("FlashQLA requires a Hopper (SM90) GPU")
        result = run_distributed_script(__file__, num_gpus=2, timeout=600)
        result.assert_success("FlashQLA CP should match the FLA reference")


if __name__ == "__main__":
    _main()
