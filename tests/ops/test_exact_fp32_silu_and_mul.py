import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_MODULE_PATH = Path(__file__).resolve().parents[2] / "src/xorl/ops/fused_silu_and_mul.py"
_SPEC = importlib.util.spec_from_file_location("xorl_exact_fp32_silu_and_mul", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _one_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    gate, up = input_tensor.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(input_tensor.dtype)


def _two_round_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    gate, up = input_tensor.chunk(2, dim=-1)
    activated = F.silu(gate.float()).to(input_tensor.dtype)
    return (activated * up).to(input_tensor.dtype)


def test_cpu_fallback_uses_one_round_program():
    values = torch.tensor(
        [[0.5, -1.25, 3.0, -0.75], [-2.0, 0.125, 1.5, 8.0]],
        dtype=torch.bfloat16,
    )
    actual = _MODULE.exact_fp32_silu_and_mul(values)
    assert torch.equal(actual, _one_round_reference(values))
    assert not torch.equal(actual, _two_round_reference(values))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU")
@pytest.mark.parametrize("shape", [(192, 24576), (512, 4096)])
def test_forward_is_byte_exact_to_one_round_reference(shape):
    torch.manual_seed(4)
    input_tensor = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous()

    actual = _MODULE.exact_fp32_silu_and_mul(input_tensor)
    expected = _one_round_reference(input_tensor)

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for exact fused SwiGLU")
def test_backward_matches_one_round_reference():
    torch.manual_seed(5)
    input_tensor = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16).contiguous().requires_grad_(True)
    reference_input = input_tensor.detach().clone().requires_grad_(True)
    grad_output = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)

    _MODULE.exact_fp32_silu_and_mul(input_tensor).backward(grad_output)
    _one_round_reference(reference_input).backward(grad_output)

    split = input_tensor.shape[-1] // 2
    torch.testing.assert_close(
        input_tensor.grad[:, :split],
        reference_input.grad[:, :split],
        rtol=0.01,
        atol=0.00390625,
    )
    assert torch.equal(input_tensor.grad[:, split:], reference_input.grad[:, split:])
