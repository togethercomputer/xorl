"""Numerical parity checks for the vendored FlashQLA GDN backend."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _flashqla_chunk_or_skip():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashQLA requires a Hopper (SM90) GPU")
    import tilelang.language as tl  # noqa: PLC0415

    if "prefer_instruction" not in inspect.signature(tl.copy).parameters:
        pytest.skip("tilelang lacks the required prefer_instruction support")
    from xorl.ops._vendored.flashqla import (  # noqa: PLC0415
        chunk_gated_delta_rule,
    )

    return chunk_gated_delta_rule


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()


def _inputs(num_heads: int, *, requires_grad: bool):
    generator = torch.Generator(device="cuda").manual_seed(0)
    shape = (1, 4096, num_heads, 128)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(shape[:-1], generator=generator, device="cuda", dtype=torch.float32)
    g = F.logsigmoid(torch.randn(shape[:-1], generator=generator, device="cuda", dtype=torch.float32))
    if requires_grad:
        for value in (q, k, v, beta, g):
            value.requires_grad_()
    return q, k, v, g, beta


def _assert_flashqla_matches_fla_forward(num_heads):
    flashqla_chunk = _flashqla_chunk_or_skip()
    q, k, v, g, beta = _inputs(num_heads, requires_grad=False)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }
    fla_output, fla_state = fla_chunk(**kwargs)
    flashqla_output, flashqla_state = flashqla_chunk(**kwargs)

    assert torch.isfinite(flashqla_output).all()
    assert torch.isfinite(flashqla_state).all()
    assert _cosine(fla_output, flashqla_output) > 0.99
    assert _cosine(fla_state, flashqla_state) > 0.99


def test_flashqla_forward_and_backward_match_fla():
    for num_heads in (4, 32):
        _assert_flashqla_matches_fla_forward(num_heads)

        flashqla_chunk = _flashqla_chunk_or_skip()
        gradients = {}
        for name, implementation in (("fla", fla_chunk), ("flashqla", flashqla_chunk)):
            q, k, v, g, beta = _inputs(num_heads, requires_grad=True)
            output, _ = implementation(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            output.float().square().mean().backward()
            gradients[name] = (q.grad, k.grad, v.grad, g.grad, beta.grad)

        for reference, actual in zip(gradients["fla"], gradients["flashqla"]):
            assert actual is not None and torch.isfinite(actual).all(), num_heads
            assert _cosine(reference, actual) > 0.97, num_heads
