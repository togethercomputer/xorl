"""Scoping gates: one-round FP32 SwiGLU is EXACT-CONTRACT-ONLY on this branch.

The upstream #43 chain makes one-round universal; this unblocking branch
deliberately scopes it (see the module docstring of
src/xorl/ops/fused_silu_and_mul.py). These gates pin both sides of the split:

  * the default ``fused_silu_and_mul`` keeps the historical TWO-ROUND bytes
    (pre-landing program) for non-exact callers — proven per element against
    the two-round reference and shown to DIFFER from one-round (so the gate
    has discriminating power);
  * a real affected caller (Qwen3_5MLP) produces pre-landing bytes under a
    non-exact config and serving-paired one-round bytes under the exact
    contract.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xorl.models.transformers.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
from xorl.ops.exact.fused_silu_and_mul import exact_fp32_silu_and_mul, fused_silu_and_mul


def _two_round_reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    activated = F.silu(gate.float()).to(x.dtype)
    return (activated * up).to(x.dtype)


def _one_round_reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(x.dtype)


def _config(*, exact: bool) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu",
        _activation_native=False,
        _qwen35_exact_contract=exact,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(96, 7168), (512, 4096)])
def test_default_op_keeps_two_round_bytes(shape):
    torch.manual_seed(3)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()
    out = fused_silu_and_mul(x)
    assert torch.equal(out, _two_round_reference(x)), "default op moved off the pre-landing two-round bytes"
    # Discriminating power: one-round really is a different program on this data.
    assert not torch.equal(out, _one_round_reference(x))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_op_is_one_round(shape=(96, 7168)):
    torch.manual_seed(3)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()
    assert torch.equal(exact_fp32_silu_and_mul(x), _one_round_reference(x))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nonexact_qwen35_mlp_bytes_unchanged():
    """Affected-caller gate: a NON-exact Qwen3_5MLP forward byte-equals the
    pre-landing composition (gate_up GEMM -> two-round fused act -> down GEMM)."""
    torch.manual_seed(9)
    mlp = Qwen3_5MLP(_config(exact=False)).to(device="cuda", dtype=torch.bfloat16)
    assert mlp._use_fused_silu and not mlp._exact_one_round
    x = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = mlp(x)
        reference = mlp.down_proj(fused_silu_and_mul(mlp.gate_up_proj(x)))
    assert torch.equal(out, reference), "non-exact Qwen3_5MLP bytes moved off the pre-landing program"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_qwen35_mlp_selects_one_round():
    torch.manual_seed(9)
    mlp = Qwen3_5MLP(_config(exact=True)).to(device="cuda", dtype=torch.bfloat16)
    assert mlp._use_fused_silu and mlp._exact_one_round
    x = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = mlp(x)
        reference = mlp.down_proj(exact_fp32_silu_and_mul(mlp.gate_up_proj(x)))
    assert torch.equal(out, reference)
