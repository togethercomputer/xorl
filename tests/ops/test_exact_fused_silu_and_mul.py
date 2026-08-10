import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_MODULE_PATH = Path(__file__).resolve().parents[2] / "src/xorl/ops/fused_silu_and_mul.py"
_SPEC = importlib.util.spec_from_file_location("xorl_exact_fused_silu_and_mul", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _two_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    gate, up = input_tensor.chunk(2, dim=-1)
    activated = F.silu(gate.float()).to(input_tensor.dtype)
    return (activated * up).to(input_tensor.dtype)


def test_hopper_shape_resolver_thresholds():
    assert _MODULE._exact_fused_swiglu_min_rows(12288) == 192
    assert _MODULE._exact_fused_swiglu_min_rows(8192) == 192
    assert _MODULE._exact_fused_swiglu_min_rows(2048) == 512


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU")
@pytest.mark.parametrize("shape", [(192, 24576), (512, 4096)])
def test_forward_is_byte_exact_to_two_round_reference(shape):
    torch.manual_seed(4)
    input_tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()

    actual = _MODULE.fused_silu_and_mul(input_tensor)
    expected = _two_round_reference(input_tensor)

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU")
def test_backward_matches_two_round_reference():
    torch.manual_seed(5)
    input_tensor = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16).contiguous().requires_grad_(True)
    reference_input = input_tensor.detach().clone().requires_grad_(True)
    grad_output = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)

    _MODULE.fused_silu_and_mul(input_tensor).backward(grad_output)
    _two_round_reference(reference_input).backward(grad_output)

    assert torch.equal(input_tensor.grad[:, 2048:], reference_input.grad[:, 2048:])
    torch.testing.assert_close(
        input_tensor.grad[:, :2048],
        reference_input.grad[:, :2048],
        rtol=0.025,
        atol=0.015625,
    )
