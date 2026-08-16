"""Compile integration for the exact DSV4 routed/shared contributor leaf."""

import pytest
import torch

from xorl.models.transformers.deepseek_v4.native_payload import (
    dsv4_join_routed_shared_partial,
)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param(
            "cuda",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="requires CUDA",
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_dsv4_leaf_forward_and_backward_keep_one_round_contract_under_compile(
    dynamic: bool,
    device: str,
):
    shared = torch.tensor(
        [-0.01165771484375],
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    routed = torch.tensor(
        [209920.0],
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    scale = 1.1

    def join_fn(routed_arg: torch.Tensor, shared_arg: torch.Tensor) -> torch.Tensor:
        return dsv4_join_routed_shared_partial(
            routed_arg,
            shared_arg,
            routed_scaling_factor=scale,
        )

    output = torch.compile(join_fn, fullgraph=True, dynamic=dynamic)(routed, shared)
    grad_routed, grad_shared = torch.autograd.grad(
        output,
        (routed, shared),
        grad_outputs=torch.ones_like(output),
    )

    assert output.view(torch.uint16).item() == 0x4862
    assert torch.equal(grad_shared, torch.ones_like(shared))
    assert torch.equal(
        grad_routed,
        torch.ones_like(routed, dtype=torch.float32).mul(scale).to(routed.dtype),
    )
