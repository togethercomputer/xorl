"""Parity tests for the vendored FlashQLA GDN backend vs. the FLA Triton kernels.

FlashQLA is a Hopper-only TileLang reimplementation of the Gated Delta Rule with
an algebraic reformulation of the fwd/bwd flows, so outputs are compared with a
cosine-similarity tolerance rather than bitwise equality.
"""

import pytest
import torch
import torch.nn.functional as F

from xorl.ops.linear_attention.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk


def _flashqla_chunk_or_skip():
    import inspect  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashQLA requires a Hopper (SM90) GPU")
    try:
        import tilelang.language as _tl  # noqa: PLC0415

        # FlashQLA's kernels force TMA loads via T.copy(..., prefer_instruction="tma"),
        # which requires tile-ai/tilelang PR #2303 (post-0.1.10). Skip if unavailable
        # so the suite passes on a stock released tilelang that lacks it.
        if "prefer_instruction" not in inspect.signature(_tl.copy).parameters:
            pytest.skip("tilelang lacks prefer_instruction (PR #2303); FlashQLA TMA path unavailable")
        from xorl.ops.linear_attention.flashqla import chunk_gated_delta_rule as flashqla_chunk  # noqa: PLC0415
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # tilelang missing / SM90 import-time check / build failure
        pytest.skip(f"FlashQLA backend unavailable: {exc}")
    return flashqla_chunk


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _inputs(batch, seq_len, num_heads, head_dim, device, requires_grad=False):
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    beta = torch.rand(batch, seq_len, num_heads, device=device, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.randn(batch, seq_len, num_heads, device=device, dtype=torch.float32))
    if requires_grad:
        for t in (q, k, v, beta, g):
            t.requires_grad_()
    return q, k, v, g, beta


# num_heads=4 is a minimal smoke shape; num_heads=32 is the production Qwen3.5/3.6-35B-A3B
# GDN value-head count (q/k are repeated 16->32 before the kernel). The 32-head case at
# seq_len=4096 is where FlashQLA's native kkt_solve produced NaN (regression guard for the
# robust_kkt_solve fix); the original 4-head-only tests never exercised it.
@pytest.mark.parametrize("num_heads", [4, 32])
def test_flashqla_matches_fla_forward(num_heads):
    flashqla_chunk = _flashqla_chunk_or_skip()
    device = "cuda"
    q, k, v, g, beta = _inputs(1, 4096, num_heads, 128, device)

    o_fla, ht_fla = fla_chunk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    o_qla, ht_qla = flashqla_chunk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert torch.isfinite(o_qla).all()
    assert _cos(o_fla, o_qla) > 0.99
    assert _cos(ht_fla, ht_qla) > 0.99


@pytest.mark.parametrize("num_heads", [4, 32])
def test_flashqla_matches_fla_backward(num_heads):
    flashqla_chunk = _flashqla_chunk_or_skip()
    device = "cuda"

    grads = {}
    for name, chunk_fn in (("fla", fla_chunk), ("flashqla", flashqla_chunk)):
        q, k, v, g, beta = _inputs(1, 4096, num_heads, 128, device, requires_grad=True)
        o, _ = chunk_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        o.float().square().mean().backward()
        grads[name] = (q.grad, k.grad, v.grad, beta.grad, g.grad)

    for gf, gq in zip(grads["fla"], grads["flashqla"]):
        assert gq is not None and torch.isfinite(gq).all()
        assert _cos(gf, gq) > 0.97
