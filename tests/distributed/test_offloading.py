import torch

from xorl.distributed.offloading import build_activation_offloading_context


def test_activation_offload_none_gpu_limit_defaults_to_zero() -> None:
    model_fwd_context, _ = build_activation_offloading_context(
        enable_activation_offload=True,
        enable_gradient_checkpointing=False,
        activation_gpu_limit=None,
    )

    x = torch.randn(4, 4, requires_grad=True)
    with model_fwd_context:
        loss = (x * x).sum()

    loss.backward()

    assert x.grad is not None
