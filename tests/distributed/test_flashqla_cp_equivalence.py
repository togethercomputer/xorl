"""Distributed numerical equivalence checks for the FlashQLA GDN backend under native CP.

Mirrors ``test_linear_attention_cp_equivalence.py`` but drives the FlashQLA interior
(``XORL_GDN_BACKEND=flashqla``) through xorl's Ulysses/sequence-parallel CP. FlashQLA is
an algebraic reformulation of the Gated Delta Rule, so the sharded-then-gathered result is
compared to the single-GPU full-sequence FLA reference with a cosine-similarity tolerance
(matching ``tests/ops/test_flashqla_gdn.py``), not bitwise/atol equality.

Requires 2+ Hopper (SM90) GPUs and a ``tilelang`` build with the ``tl_gemm`` builtin +
PR #2303 (``prefer_instruction``); otherwise skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from xorl.distributed.parallel_state import init_parallel_state
from xorl.ops.linear_attention import GatedDeltaNet
from xorl.ops.linear_attention.ops.cp import build_linear_attention_cp_context
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than


pytestmark = [pytest.mark.distributed]

HEAD_DIM = 128  # FlashQLA requires 128-dim heads.


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _setup_dist(*, ulysses_size: int) -> torch.device:
    local_rank = _local_rank()
    # Per-rank JIT caches: ranks compile the same FLA/TileLang kernels concurrently, and a
    # shared cache dir races (FileNotFoundError on a half-written manifest).
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
        ulysses_size=ulysses_size,
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    return torch.device("cuda", local_rank)


def _build_test_layer(device: torch.device) -> GatedDeltaNet:
    # Production Qwen3.5/3.6-35B-A3B GDN shape: 16 key heads / 32 value heads (q/k repeated
    # 16->32 before the kernel), head dim 128 — the shape that exposed the kkt_solve NaN.
    layer = GatedDeltaNet(
        hidden_size=HEAD_DIM * 16,
        expand_v=1.0,
        head_dim=HEAD_DIM,
        num_heads=16,
        num_v_heads=32,
        mode="chunk",
        use_gate=True,
        use_short_conv=True,
        conv_size=4,
        norm_eps=1e-6,
    ).to(device=device, dtype=torch.bfloat16)
    layer.eval()
    return layer


def _gather_sequence_shards(local_tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(local_tensor) for _ in range(_world_size())]
    dist.all_gather(gathered, local_tensor.contiguous())
    return torch.cat(gathered, dim=1)


def _run_layer(layer, hidden_states, *, cu_seqlens, cp_context, backend):
    """Run fwd+bwd of `layer` under the given GDN backend; return (output, input_grad)."""
    os.environ["XORL_GDN_BACKEND"] = backend
    inp = hidden_states.clone().detach().requires_grad_(True)
    out, _, _ = layer(
        hidden_states=inp,
        use_cache=False,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    )
    # Normalize the loss by the full (un-sharded) output size so per-shard grads sum correctly.
    out.float().square().sum().backward()
    return out.detach(), inp.grad.detach()


def _run_cp_equivalence() -> None:
    world_size = _world_size()
    device = _setup_dist(ulysses_size=world_size)
    rank = dist.get_rank()

    local_seq_len = 128  # 2 FlashQLA chunks (64) per rank
    total_tokens = world_size * local_seq_len
    hidden_size = HEAD_DIM * 16  # must match _build_test_layer (num_heads=16, head_dim=128)

    # seq0 lives inside rank 0; seq1 spans the tail of rank 0 through the last rank, so the
    # cross-rank merge chain (pre_num_ranks / post_num_ranks) and conv-halo are exercised
    # across all ranks for world_size >= 2.
    first_seq_len = local_seq_len - 10
    cu_seqlens = torch.tensor([0, first_seq_len, total_tokens], dtype=torch.int32, device=device)

    torch.manual_seed(0)
    reference_layer = _build_test_layer(device)
    cp_layer = _build_test_layer(device)
    cp_layer.load_state_dict(reference_layer.state_dict())

    torch.manual_seed(1)
    full_hidden_states = torch.randn(1, total_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    local_hidden_states = full_hidden_states[:, rank * local_seq_len : (rank + 1) * local_seq_len].contiguous()

    cp_context = build_linear_attention_cp_context(cu_seqlens, conv1d_kernel_size=cp_layer.conv_size)
    assert cp_context is not None and cp_context.cu_seqlens is not None

    # Single-GPU full-sequence reference on the FLA backend (physical ground truth).
    reference_output, reference_input_grad = _run_layer(
        reference_layer, full_hidden_states, cu_seqlens=cu_seqlens, cp_context=None, backend="fla"
    )
    # FlashQLA interior driven by native CP, on this rank's shard.
    cp_output, cp_input_grad = _run_layer(
        cp_layer, local_hidden_states, cu_seqlens=cu_seqlens, cp_context=cp_context, backend="flashqla"
    )

    gathered_cp_output = _gather_sequence_shards(cp_output)
    gathered_cp_input_grad = _gather_sequence_shards(cp_input_grad)

    if rank == 0:
        out_cos = _cos(gathered_cp_output, reference_output)
        grad_cos = _cos(gathered_cp_input_grad, reference_input_grad)
        print(f"FlashQLA-CP vs FLA full-seq: output cos={out_cos:.5f}  input-grad cos={grad_cos:.5f}")
        assert torch.isfinite(gathered_cp_output).all()
        assert torch.isfinite(gathered_cp_input_grad).all()
        assert out_cos > 0.99, f"forward cosine too low: {out_cos}"
        assert grad_cos > 0.97, f"backward cosine too low: {grad_cos}"
        print("FlashQLA CP equivalence passed")


def _main() -> None:
    try:
        _run_cp_equivalence()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_flashqla_cp_matches_single_gpu_reference():
        if torch.cuda.get_device_capability() != (9, 0):
            pytest.skip("FlashQLA requires a Hopper (SM90) GPU")
        result = run_distributed_script(__file__, num_gpus=2, timeout=600)
        result.assert_success("FlashQLA CP should match the single-GPU FLA reference")


if __name__ == "__main__":
    _main()
