import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.cpu

_MODULE_PATHS = {
    "xorl.ops.moe.triton": Path(__file__).resolve().parents[2] / "src/xorl/ops/moe/triton.py",
    "xorl.ops.moe.quack": Path(__file__).resolve().parents[2] / "src/xorl/ops/moe/quack.py",
}


def _counts_from_cumsum(cumsum: torch.Tensor) -> list[int]:
    counts = torch.empty_like(cumsum)
    counts[0] = cumsum[0]
    counts[1:] = cumsum[1:] - cumsum[:-1]
    return counts.tolist()


def _naive_group_gemm_same_nk(a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False, **kwargs):
    del max_M, kwargs
    assert not transpose_a

    outputs = []
    start = 0
    for expert_idx, count in enumerate(_counts_from_cumsum(cumsum_M)):
        end = start + count
        weight = b[expert_idx]
        if transpose_b:
            outputs.append(a[start:end] @ weight.transpose(0, 1))
        else:
            outputs.append(a[start:end] @ weight)
        start = end

    return torch.cat(outputs, dim=0)


def _naive_group_gemm_same_mn(a, b, c, cumsum_K, max_K, transpose_a=False, transpose_b=False, **kwargs):
    del max_K, kwargs

    start = 0
    for expert_idx, count in enumerate(_counts_from_cumsum(cumsum_K)):
        end = start + count
        lhs = a[start:end].transpose(0, 1) if transpose_a else a[start:end]
        rhs = b[start:end].transpose(0, 1) if transpose_b else b[start:end]
        c[expert_idx].copy_(lhs @ rhs)
        start = end

    return c


def _patch_ep_kernels(monkeypatch, module_name: str):
    import xorl.utils.import_utils as import_utils

    moe_stub = types.ModuleType("xorl.ops.group_gemm.kernel.moe")
    moe_stub.expert_histogram = None
    moe_stub.moe_gather = None
    moe_stub.moe_index_compute = None
    moe_stub.moe_scatter = None
    monkeypatch.setattr(import_utils, "is_fused_moe_available", lambda: True)
    sys.modules.pop("xorl.ops.group_gemm.kernel.moe", None)
    sys.modules.pop("xorl.ops.group_gemm.kernel.group_gemm", None)
    sys.modules.pop("xorl.ops.group_gemm.kernel.quack", None)
    monkeypatch.setitem(
        sys.modules,
        "xorl.ops.group_gemm.kernel.moe",
        moe_stub,
    )
    if module_name.endswith("triton"):
        group_gemm_stub = types.ModuleType("xorl.ops.group_gemm.kernel.group_gemm")
        group_gemm_stub.group_gemm_same_nk = _naive_group_gemm_same_nk
        group_gemm_stub.group_gemm_same_mn = _naive_group_gemm_same_mn
        monkeypatch.setitem(
            sys.modules,
            "xorl.ops.group_gemm.kernel.group_gemm",
            group_gemm_stub,
        )
    else:
        quack_stub = types.ModuleType("xorl.ops.group_gemm.kernel.quack")
        quack_stub.cumsum_to_cu_seqlens = lambda cumsum: cumsum
        quack_stub.quack_group_gemm_same_nk = _naive_group_gemm_same_nk
        quack_stub.quack_group_gemm_same_mn = _naive_group_gemm_same_mn
        monkeypatch.setitem(
            sys.modules,
            "xorl.ops.group_gemm.kernel.quack",
            quack_stub,
        )
    spec = importlib.util.spec_from_file_location(
        f"codex_test_{module_name.rsplit('.', maxsplit=1)[-1]}", _MODULE_PATHS[module_name]
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reference_ep_forward(
    permute_tokens: torch.Tensor,
    cumsum: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_scores: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    start = 0
    for expert_idx, count in enumerate(_counts_from_cumsum(cumsum)):
        end = start + count
        x = permute_tokens[start:end]
        h = F.silu(x @ gate_proj[expert_idx]) * (x @ up_proj[expert_idx])
        h = h * expert_scores[start:end].to(h.dtype).unsqueeze(-1)
        outputs.append(h @ down_proj[expert_idx])
        start = end

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        pytest.param("xorl.ops.moe.triton", "TritonEPGroupGemm", id="triton"),
        pytest.param("xorl.ops.moe.triton", "TritonEPGroupGemmMoeAct", id="triton-moe-act"),
        pytest.param("xorl.ops.moe.quack", "QuackEPGroupGemm", id="quack"),
        pytest.param("xorl.ops.moe.quack", "QuackEPGroupGemmMoeAct", id="quack-moe-act"),
    ],
)
def test_ep_group_gemm_propagates_routing_score_gradients(monkeypatch, module_name, class_name):
    try:
        module = _patch_ep_kernels(monkeypatch, module_name)
    except ImportError as exc:
        pytest.skip(f"{module_name} unavailable: {exc}")

    fn = getattr(module, class_name)

    torch.manual_seed(0)
    dtype = torch.float32
    num_local_experts = 2
    hidden_dim = 8
    intermediate_size = 12
    counts = torch.tensor([3, 2])
    cumsum = torch.cumsum(counts, dim=0)
    num_tokens = int(cumsum[-1].item())

    permute_tokens = torch.randn(num_tokens, hidden_dim, dtype=dtype)
    gate_proj = torch.randn(num_local_experts, hidden_dim, intermediate_size, dtype=dtype)
    up_proj = torch.randn(num_local_experts, hidden_dim, intermediate_size, dtype=dtype)
    down_proj = torch.randn(num_local_experts, intermediate_size, hidden_dim, dtype=dtype)
    expert_scores = torch.rand(num_tokens, dtype=dtype, requires_grad=True)
    upstream = torch.randn(num_tokens, hidden_dim, dtype=dtype)

    output = fn.apply(
        permute_tokens,
        cumsum,
        gate_proj,
        up_proj,
        down_proj,
        expert_scores,
    )
    output.backward(upstream)
    grad_scores = expert_scores.grad.detach().clone()

    expert_scores_ref = expert_scores.detach().clone().requires_grad_(True)
    ref_output = reference_ep_forward(
        permute_tokens,
        cumsum,
        gate_proj,
        up_proj,
        down_proj,
        expert_scores_ref,
    )
    ref_output.backward(upstream)

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(grad_scores, expert_scores_ref.grad)
