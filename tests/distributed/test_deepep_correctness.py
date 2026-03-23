"""Correctness test: DeepEP dispatch/combine vs AllToAll.

Tests that DeepEP and AllToAll produce numerically equivalent outputs
for both the forward pass and input gradients (backward pass).

Usage (single node, EP=8):
    torchrun --nproc_per_node=8 tests/distributed/test_deepep_correctness.py

Usage (2 nodes, EP=16):
    # Node 0:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --master_addr=<addr> --master_port=29500 \
        tests/distributed/test_deepep_correctness.py
    # Node 1:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
        --master_addr=<addr> --master_port=29500 \
        tests/distributed/test_deepep_correctness.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())


def teardown():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference, reduced across all ranks."""
    diff = (a.float() - b.float()).abs().max()
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    return diff.item()


def allclose_distributed(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    """torch.allclose semantics (atol + rtol*|b|), reduced across all ranks.

    Uses combined absolute+relative tolerance to avoid false failures on
    near-zero elements that would inflate pure relative error.
    """
    ok = torch.tensor(
        float(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)),
        device=a.device,
    )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return ok.item() > 0.5


# ── Core dispatch/combine runner ──────────────────────────────────────────────

def run_one_pass(
    ep_dispatch: str,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_proj: nn.Parameter,
    up_proj: nn.Parameter,
    down_proj: nn.Parameter,
    ep_group: dist.ProcessGroup,
    num_experts: int,
    num_local_experts: int,
    grad_output: torch.Tensor,
    buffer=None,
):
    """One forward + backward pass with a given dispatch backend."""
    from xorl.models.layers.moe.backend import EP_DISPATCH, EP_COMBINE, EP_EXPERT_COMPUTE

    dispatch_fn = EP_DISPATCH[ep_dispatch]
    combine_fn  = EP_COMBINE[ep_dispatch]
    compute_fn  = EP_EXPERT_COMPUTE["triton"]

    x = hidden_states.detach().requires_grad_(True)

    dispatch_kwargs = dict(
        hidden_states=x,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        num_experts=num_experts,
    )
    if ep_dispatch == "alltoall":
        dispatch_kwargs["ep_group"] = ep_group
    else:
        dispatch_kwargs["buffer"] = buffer
        dispatch_kwargs["num_local_experts"] = num_local_experts

    permuted, cumsum, ctx = dispatch_fn(**dispatch_kwargs)
    expert_out = compute_fn(permuted, cumsum, gate_proj, up_proj, down_proj)

    if ep_dispatch == "alltoall":
        combine_kwargs = dict(expert_output=expert_out, ctx=ctx, ep_group=ep_group)
    else:
        combine_kwargs = dict(buffer=buffer, expert_output=expert_out, ctx=ctx, async_combine=False)

    output = combine_fn(**combine_kwargs)
    output.backward(grad_output)

    return output.detach().clone(), x.grad.detach().clone()


# ── Single test case ──────────────────────────────────────────────────────────

def run_test(
    num_tokens: int,
    hidden_dim: int,
    intermediate_size: int,
    topk: int,
    atol: float = 0.1,
    rtol: float = 0.05,
) -> bool:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    num_experts = world_size
    num_local_experts = 1

    log(f"  [{num_tokens}tok, h={hidden_dim}, ffn={intermediate_size}, top{topk}]  ", )

    ep_group = dist.new_group(list(range(world_size)))

    # Xavier-scaled weights, broadcast so all ranks are identical
    torch.manual_seed(0)
    scale = (2.0 / (hidden_dim + intermediate_size)) ** 0.5
    gate = nn.Parameter(torch.randn(1, hidden_dim, intermediate_size, device=device, dtype=torch.bfloat16) * scale)
    up   = nn.Parameter(torch.randn(1, hidden_dim, intermediate_size, device=device, dtype=torch.bfloat16) * scale)
    down = nn.Parameter(torch.randn(1, intermediate_size, hidden_dim, device=device, dtype=torch.bfloat16) * scale)
    for p in [gate, up, down]:
        dist.broadcast(p.data, src=0)

    # Per-rank inputs (different tokens on each rank)
    torch.manual_seed(rank + 1)
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts, device=device)
    weights = torch.softmax(logits, dim=-1)
    topk_w, topk_idx = torch.topk(weights, k=topk, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)

    # Fixed upstream gradient (same for both backends)
    torch.manual_seed(rank + 99)
    grad_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

    # ── AllToAll ──────────────────────────────────────────────────────────────
    out_a2a, grad_a2a = run_one_pass(
        "alltoall", hidden, topk_w, topk_idx,
        gate, up, down, ep_group, num_experts, num_local_experts, grad_out,
    )
    gate.grad = None; up.grad = None; down.grad = None

    # ── DeepEP ───────────────────────────────────────────────────────────────
    from xorl.distributed.moe.deepep import DeepEPBuffer
    buffer = DeepEPBuffer(ep_group=ep_group, buffer_size_gb=1.0)
    out_dep, grad_dep = run_one_pass(
        "deepep", hidden, topk_w, topk_idx,
        gate, up, down, ep_group, num_experts, num_local_experts, grad_out,
        buffer=buffer,
    )
    buffer.destroy_buffer()
    gate.grad = None; up.grad = None; down.grad = None

    # ── Check ─────────────────────────────────────────────────────────────────
    fwd_abs = max_abs_diff(out_dep, out_a2a)
    bwd_abs = max_abs_diff(grad_dep, grad_a2a)

    fwd_ok = allclose_distributed(out_dep, out_a2a, atol=atol, rtol=rtol)
    bwd_ok = allclose_distributed(grad_dep, grad_a2a, atol=atol, rtol=rtol)

    log(
        f"    fwd max_abs={fwd_abs:.2e} {'✓' if fwd_ok else '✗'}  |  "
        f"bwd max_abs={bwd_abs:.2e} {'✓' if bwd_ok else '✗'}"
    )

    return fwd_ok and bwd_ok


# ── Test suite ────────────────────────────────────────────────────────────────

def run_all_tests() -> bool:
    world_size = dist.get_world_size()
    cross_node = world_size > 8

    log(f"\n{'='*65}")
    log(f"  DeepEP vs AllToAll Correctness — world_size={world_size} "
        f"({'cross-node' if cross_node else 'single-node'})")
    log(f"{'='*65}")

    configs = [
        dict(num_tokens=16,  hidden_dim=256,   intermediate_size=512,  topk=2),
        dict(num_tokens=32,  hidden_dim=512,   intermediate_size=1024, topk=4),
        dict(num_tokens=64,  hidden_dim=1024,  intermediate_size=2048, topk=4),
        dict(num_tokens=64,  hidden_dim=2048,  intermediate_size=4096, topk=8),
    ]

    all_ok = True
    for cfg in configs:
        ok = run_test(**cfg)
        if not ok:
            all_ok = False

    log(f"\n  {'[PASS] All tests passed ✓' if all_ok else '[FAIL] Some tests failed ✗'}")
    log(f"{'='*65}\n")
    return all_ok


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Add NVSHMEM lib to LD_LIBRARY_PATH for DeepEP internode transport
    try:
        import nvidia.nvshmem
        nvshmem_lib = os.path.join(list(nvidia.nvshmem.__path__)[0], "lib")
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if nvshmem_lib not in existing:
            os.environ["LD_LIBRARY_PATH"] = f"{nvshmem_lib}:{existing}" if existing else nvshmem_lib
    except Exception:
        pass

    setup()
    passed = run_all_tests()
    teardown()

    sys.exit(0 if passed else 1)
