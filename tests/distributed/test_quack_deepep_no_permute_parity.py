"""Parity + finiteness test for the Quack DeepEP no-permute fused path.

``moe_implementation: quack`` + ``ep_dispatch: deepep`` routes expert compute
through ``QuackEPDeepEPNoPermute`` (chunked, fused with combine) — NOT through
``QuackEPGroupGemm``, which is what ``test_deepep_correctness.py`` covers. This
test drives the no-permute path directly at the Qwen3.6-35B-A3B MoE shape
(h=2048, I=512, E=256, top-8) and compares forward output and all gradients
against the trusted triton generic path on the same DeepEP dispatch.

Repro for the 2026-06-12 local-train nan: quack backward produced nan/inf
grads on step 1 while the forward stayed finite
(docs/notes/quack_moe_local_train_nan_handoff.md).
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if (a.norm() == 0 and b.norm() == 0) else 0.0
    return (a @ b / denom).item()


def _check_finite(name: str, tensor: torch.Tensor | None, errors: list[str]) -> None:
    if tensor is None:
        errors.append(f"{name} is None")
        return
    if not torch.isfinite(tensor.float()).all():
        n_bad = (~torch.isfinite(tensor.float())).sum().item()
        errors.append(f"{name} has {n_bad} non-finite values (shape {tuple(tensor.shape)})")


def _run_reference_pass(hidden, topk_w, topk_idx, gate_up, down, intermediate_size, buffer, num_experts, num_local):
    """Trusted path: generic DeepEP dispatch + triton EP compute + DeepEP combine."""
    from xorl.models.layers.moe.backend import EP_COMBINE, EP_DISPATCH, EP_EXPERT_COMPUTE  # noqa: PLC0415

    x = hidden.detach().requires_grad_(True)
    permuted, cumsum, ctx = EP_DISPATCH["deepep"](
        hidden_states=x,
        routing_weights=topk_w,
        selected_experts=topk_idx,
        num_experts=num_experts,
        buffer=buffer,
        num_local_experts=num_local,
    )
    expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
    expert_out = EP_EXPERT_COMPUTE["triton"](
        permuted,
        cumsum,
        gate_up,
        down,
        intermediate_size,
        expert_scores,
        hidden_act="silu",
        fp8_compute=False,
    )
    output = EP_COMBINE["deepep"](buffer=buffer, expert_output=expert_out, ctx=ctx, async_combine=False)
    return x, output


def _run_no_permute_pass(
    hidden, topk_w, topk_idx, gate_up, down, intermediate_size, buffer, num_experts, num_local, detach=True
):
    """Production path under quack+deepep: fused no-permute compute+combine."""
    from xorl.distributed.moe.deepep import token_pre_dispatch_no_permute  # noqa: PLC0415
    from xorl.ops.moe.quack import QuackEPDeepEPNoPermute  # noqa: PLC0415

    x = hidden.detach().requires_grad_(True) if detach else hidden
    recv_x, cumsum, ctx = token_pre_dispatch_no_permute(
        buffer=buffer,
        hidden_states=x,
        routing_weights=topk_w,
        selected_experts=topk_idx,
        num_experts=num_experts,
        num_local_experts=num_local,
    )
    expert_scores = getattr(ctx, "expert_scores", getattr(ctx, "permuted_scores", None))
    output = QuackEPDeepEPNoPermute.apply(
        recv_x,
        cumsum,
        gate_up,
        down,
        intermediate_size,
        expert_scores,
        buffer,
        ctx,
        False,  # async_combine
        "silu",  # hidden_act
        False,  # activation_native
        False,  # fp8_compute
        "triton_grouped",  # fp8_grouped_backend (unused for bf16)
        128,  # fp8_block_size (unused for bf16)
        None,  # gate_up_bias
        None,  # down_bias
    )
    return x, output


def _run_case(name, num_tokens, hidden_dim, intermediate_size, num_experts, topk, routing, buffer, device) -> list[str]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_local = num_experts // world_size

    torch.manual_seed(1000 + rank)
    scale = (2.0 / (hidden_dim + intermediate_size)) ** 0.5
    gate_up = nn.Parameter(
        torch.randn(num_local, hidden_dim, 2 * intermediate_size, device=device, dtype=torch.bfloat16) * scale
    )
    down = nn.Parameter(
        torch.randn(num_local, intermediate_size, hidden_dim, device=device, dtype=torch.bfloat16) * scale
    )

    torch.manual_seed(rank + 1)
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    if routing == "balanced":
        offsets = torch.arange(topk, device=device).view(1, -1)
        token_offsets = torch.arange(num_tokens, device=device).view(-1, 1)
        topk_idx = (token_offsets * topk + offsets + rank * num_local) % num_experts
    elif routing == "random":
        topk_idx = torch.stack([torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)])
    elif routing == "skewed":
        # All tokens hit only the first eighth of experts -> most experts empty.
        topk_idx = torch.randint(0, max(topk, num_experts // 8), (num_tokens, topk), device=device)
        topk_idx = topk_idx + torch.arange(topk, device=device).view(1, -1) * 0  # keep shape
        # De-duplicate within a row by offsetting collisions modulo the band.
        band = max(topk, num_experts // 8)
        arange = torch.arange(topk, device=device).view(1, -1)
        topk_idx = (topk_idx[:, :1] + arange) % band
    else:
        raise ValueError(routing)
    topk_idx = topk_idx.to(torch.long)
    topk_w = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)

    torch.manual_seed(rank + 99)
    grad_out = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

    x_ref, out_ref = _run_reference_pass(
        hidden, topk_w, topk_idx, gate_up, down, intermediate_size, buffer, num_experts, num_local
    )
    out_ref.backward(grad_out)
    ref = {
        "output": out_ref.detach().clone(),
        "x_grad": x_ref.grad.detach().clone(),
        "gate_up_grad": gate_up.grad.detach().clone(),
        "down_grad": down.grad.detach().clone(),
    }
    gate_up.grad = None
    down.grad = None

    x_np, out_np = _run_no_permute_pass(
        hidden, topk_w, topk_idx, gate_up, down, intermediate_size, buffer, num_experts, num_local
    )
    out_np.backward(grad_out)
    got = {
        "output": out_np.detach().clone(),
        "x_grad": x_np.grad.detach().clone(),
        "gate_up_grad": gate_up.grad.detach().clone() if gate_up.grad is not None else None,
        "down_grad": down.grad.detach().clone() if down.grad is not None else None,
    }
    gate_up.grad = None
    down.grad = None

    errors: list[str] = []
    for key in ("output", "x_grad", "gate_up_grad", "down_grad"):
        _check_finite(f"{name}/{key}", got[key], errors)
    for key in ("output", "x_grad", "gate_up_grad", "down_grad"):
        if got[key] is None:
            continue
        cos = _cosine(got[key], ref[key])
        max_abs = (got[key].float() - ref[key].float()).abs().max().item()
        if rank == 0:
            print(f"[{name}] {key}: cos={cos:.6f} max_abs={max_abs:.3e}", flush=True)
        if cos < 0.999:
            errors.append(f"{name}/{key} cosine vs triton = {cos:.6f} (max_abs={max_abs:.3e})")
    return errors


def _run_training_smoke(
    name, num_tokens, hidden_dim, intermediate_size, num_experts, topk, buffer, device, use_checkpoint, num_steps=3
) -> list[str]:
    """Multi-step train loop through the no-permute path, optionally under
    torch.utils.checkpoint (mimics gradient_checkpointing_method=recompute_full_layer,
    which reruns DeepEP dispatch + the fused compute/combine during backward)."""
    import torch.utils.checkpoint as ckpt  # noqa: PLC0415

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_local = num_experts // world_size

    torch.manual_seed(2000 + rank)
    scale = (2.0 / (hidden_dim + intermediate_size)) ** 0.5
    gate_up = nn.Parameter(
        torch.randn(num_local, hidden_dim, 2 * intermediate_size, device=device, dtype=torch.bfloat16) * scale
    )
    down = nn.Parameter(
        torch.randn(num_local, intermediate_size, hidden_dim, device=device, dtype=torch.bfloat16) * scale
    )
    opt = torch.optim.AdamW([gate_up, down], lr=1e-5)

    errors: list[str] = []
    for step in range(num_steps):
        torch.manual_seed(3000 + rank * 100 + step)
        hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
        offsets = torch.arange(topk, device=device).view(1, -1)
        token_offsets = torch.arange(num_tokens, device=device).view(-1, 1)
        topk_idx = ((token_offsets * topk + offsets + rank * num_local) % num_experts).to(torch.long)
        topk_w = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)

        def _layer(x):
            _, out = _run_no_permute_pass(
                x, topk_w, topk_idx, gate_up, down, intermediate_size, buffer, num_experts, num_local, detach=False
            )
            return out

        x = hidden.detach().requires_grad_(True)
        if use_checkpoint:
            out = ckpt.checkpoint(_layer, x, use_reentrant=False)
        else:
            out = _layer(x)
        loss = out.float().pow(2).mean()
        loss.backward()

        for pname, p in (("gate_up", gate_up), ("down", down), ("x", x)):
            g = p.grad
            if g is None or not torch.isfinite(g.float()).all():
                errors.append(f"{name}/step{step}/{pname}.grad non-finite (ckpt={use_checkpoint})")
        if not torch.isfinite(loss):
            errors.append(f"{name}/step{step}/loss non-finite: {loss.item()}")
        if rank == 0:
            print(f"[{name}] step {step}: loss={loss.item():.6f} ckpt={use_checkpoint}", flush=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if errors:
            break
    return errors


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

    from xorl.distributed.moe.deepep import DeepEPBuffer  # noqa: PLC0415

    ep_group = dist.new_group(list(range(world_size)))
    buffer = DeepEPBuffer(ep_group=ep_group, buffer_size_gb=2.0, num_sms=24)

    # Qwen3.6-35B-A3B MoE shape: hidden 2048, moe_intermediate 512, 256 experts, top-8.
    cases = [
        dict(
            name="q36_balanced",
            num_tokens=4096,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            routing="balanced",
        ),
        dict(
            name="q36_random",
            num_tokens=4096,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            routing="random",
        ),
        dict(
            name="q36_skewed_empty_experts",
            num_tokens=1023,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            routing="skewed",
        ),
        dict(
            name="q36_large_m",
            num_tokens=16384,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            routing="balanced",
        ),
    ]

    train_cases = [
        dict(
            name="train_ckpt_recompute",
            num_tokens=8192,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            use_checkpoint=True,
        ),
        dict(
            name="train_no_ckpt",
            num_tokens=8192,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            use_checkpoint=False,
        ),
        dict(
            name="train_ckpt_prod_m",
            num_tokens=32768,
            hidden_dim=2048,
            intermediate_size=512,
            num_experts=256,
            topk=8,
            use_checkpoint=True,
        ),
    ]

    all_errors: list[str] = []
    try:
        for case in cases:
            all_errors.extend(_run_case(buffer=buffer, device=device, **case))
        # Explicit chunked-mode coverage (full-size chunks are the default since
        # the 2026-06-12 corruption fix; chunking stays as an opt-in memory lever).
        os.environ["XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE"] = "256"
        os.environ["XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE"] = "256"
        try:
            all_errors.extend(
                _run_case(
                    buffer=buffer,
                    device=device,
                    name="q36_chunked_256",
                    num_tokens=4096,
                    hidden_dim=2048,
                    intermediate_size=512,
                    num_experts=256,
                    topk=8,
                    routing="balanced",
                )
            )
        finally:
            del os.environ["XORL_QUACK_DEEPEP_DOWN_HIDDEN_CHUNK_SIZE"]
            del os.environ["XORL_QUACK_DEEPEP_INTERMEDIATE_CHUNK_SIZE"]
        for case in train_cases:
            all_errors.extend(_run_training_smoke(buffer=buffer, device=device, **case))
        flag = torch.tensor(0 if not all_errors else 1, device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        for err in all_errors:
            print(f"[rank{rank}] FAIL: {err}", flush=True)
        if flag.item() == 0 and rank == 0:
            print("quack_deepep_no_permute_parity_ok", flush=True)
    finally:
        buffer.destroy_buffer()
        dist.destroy_process_group()

    return 0 if not all_errors else 1


@skip_if_gpu_count_less_than(2)
def test_quack_deepep_no_permute_parity():
    pytest.importorskip("deep_ep")
    pytest.importorskip("nvidia.nvshmem")
    _install_nvidia_ml_library_path()

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=600,
        extra_env={
            "XORL_TEST_QUACK_DEEPEP_NO_PERMUTE_WORKER": "1",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
    )
    result.assert_success()
    assert "quack_deepep_no_permute_parity_ok" in result.stdout


if __name__ == "__main__" and os.environ.get("XORL_TEST_QUACK_DEEPEP_NO_PERMUTE_WORKER") == "1":
    sys.exit(_worker_main())
