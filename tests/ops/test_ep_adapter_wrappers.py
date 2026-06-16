"""Tests that EP adapter wrappers in backend/__init__.py correctly forward expert_scores.

Bug: _quack_ep_fused / _native_ep_fused accepted expert_scores but silently dropped it.

These tests monkeypatch the downstream implementations to verify the adapters pass
expert_scores through, and compare output to a naive reference.
"""

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from xorl.models.layers.moe.backend import EP_EXPERT_COMPUTE
from xorl.utils import import_utils


pytestmark = pytest.mark.cpu

_MODULE_PATHS = {
    "xorl.ops.moe.triton": Path(__file__).resolve().parents[2] / "src/xorl/ops/moe/triton.py",
    "xorl.ops.moe.quack": Path(__file__).resolve().parents[2] / "src/xorl/ops/moe/quack.py",
}

_BACKEND_INIT_PATH = Path(__file__).resolve().parents[2] / "src/xorl/models/layers/moe/backend/__init__.py"


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


def reference_ep_forward(permute_tokens, cumsum, gate_proj, up_proj, down_proj, expert_scores):
    """Naive reference: per-expert matmul with SiLU + optional score scaling."""
    outputs = []
    start = 0
    for expert_idx, count in enumerate(_counts_from_cumsum(cumsum)):
        end = start + count
        x = permute_tokens[start:end]
        h = F.silu(x @ gate_proj[expert_idx]) * (x @ up_proj[expert_idx])
        if expert_scores is not None:
            h = h * expert_scores[start:end].to(h.dtype).unsqueeze(-1)
        outputs.append(h @ down_proj[expert_idx])
        start = end
    return torch.cat(outputs, dim=0)


def _patch_kernels_and_load_backend(monkeypatch, backend_type: str):
    """Patch kernel modules and load backend/__init__.py to get the adapter wrappers."""
    moe_stub = types.ModuleType("xorl.ops.group_gemm.kernel.moe")
    moe_stub.expert_histogram = None
    moe_stub.moe_gather = None
    moe_stub.moe_index_compute = None
    moe_stub.moe_scatter = None
    monkeypatch.setattr(import_utils, "is_fused_moe_available", lambda: True)
    sys.modules.pop("xorl.ops.group_gemm.kernel.moe", None)
    sys.modules.pop("xorl.ops.group_gemm.kernel.group_gemm", None)
    sys.modules.pop("xorl.ops.group_gemm.kernel.quack", None)
    monkeypatch.setitem(sys.modules, "xorl.ops.group_gemm.kernel.moe", moe_stub)

    if backend_type == "triton":
        group_gemm_stub = types.ModuleType("xorl.ops.group_gemm.kernel.group_gemm")
        group_gemm_stub.group_gemm_same_nk = _naive_group_gemm_same_nk
        group_gemm_stub.group_gemm_same_mn = _naive_group_gemm_same_mn
        monkeypatch.setitem(sys.modules, "xorl.ops.group_gemm.kernel.group_gemm", group_gemm_stub)
        module_name = "xorl.ops.moe.triton"
    else:
        quack_stub = types.ModuleType("xorl.ops.group_gemm.kernel.quack")
        quack_stub.cumsum_to_cu_seqlens = lambda cumsum: cumsum
        quack_stub.quack_group_gemm_same_nk = _naive_group_gemm_same_nk
        quack_stub.quack_group_gemm_same_mn = _naive_group_gemm_same_mn
        monkeypatch.setitem(sys.modules, "xorl.ops.group_gemm.kernel.quack", quack_stub)
        module_name = "xorl.ops.moe.quack"

    spec = importlib.util.spec_from_file_location(f"adapter_test_{backend_type}", _MODULE_PATHS[module_name])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, module_name, module)

    return module


def _make_test_data(dtype=torch.float32):
    torch.manual_seed(42)
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
    gate_up_proj = torch.cat([gate_proj, up_proj], dim=-1)
    expert_scores = torch.rand(num_tokens, dtype=dtype)

    return permute_tokens, cumsum, gate_proj, up_proj, down_proj, gate_up_proj, intermediate_size, expert_scores


@pytest.mark.parametrize(
    ("backend_type", "class_name"),
    [
        pytest.param("triton", "TritonEPGroupGemm", id="triton-ep"),
        pytest.param("quack", "QuackEPGroupGemm", id="quack-ep"),
    ],
)
def test_adapter_forwards_expert_scores(monkeypatch, backend_type, class_name):
    """Verify that EP adapter wrappers accept and forward expert_scores."""
    try:
        kernel_module = _patch_kernels_and_load_backend(monkeypatch, backend_type)
    except ImportError as exc:
        pytest.skip(f"{backend_type} unavailable: {exc}")

    (
        permute_tokens,
        cumsum,
        gate_proj,
        up_proj,
        down_proj,
        gate_up_proj,
        intermediate_size,
        expert_scores,
    ) = _make_test_data()

    fn_cls = getattr(kernel_module, class_name)
    output = fn_cls.apply(permute_tokens, cumsum, gate_up_proj, down_proj, intermediate_size, expert_scores)

    ref = reference_ep_forward(permute_tokens, cumsum, gate_proj, up_proj, down_proj, expert_scores)
    torch.testing.assert_close(output, ref)


# ---------------------------------------------------------------------------
# Signature contract tests — replaces the old source-grep regression test.
# Inspects live function signatures instead of pattern-matching on source text,
# so formatting changes, variable renames, and line rewrapping can't fool it.
# ---------------------------------------------------------------------------

# Required explicit parameters for every EP_EXPERT_COMPUTE entry.
_REQUIRED_EP_PARAMS = ("expert_scores", "hidden_act")


@pytest.mark.parametrize("name,fn", list(EP_EXPERT_COMPUTE.items()))
def test_ep_compute_signature_contract(name, fn):
    """Every EP compute function must explicitly accept expert_scores, hidden_act,
    and a **kwargs catch-all for forward-compat extras (gate_up_bias, down_bias, etc.).
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    for required in _REQUIRED_EP_PARAMS:
        assert required in params, (
            f"EP_EXPERT_COMPUTE['{name}'] is missing explicit '{required}' param. Signature: {sig}"
        )

    has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in params.values())
    assert has_var_keyword, (
        f"EP_EXPERT_COMPUTE['{name}'] has no **kwargs — new extras like "
        f"gate_up_bias will break callers. Signature: {sig}"
    )
