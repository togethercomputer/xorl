"""Numerical parity of the Quack EP grouped MoE compute against Triton.

Regression test for the silent interleave bug: the Quack fused gated GEMM
applies its activation epilogue over INTERLEAVED (even=gate, odd=up) output
columns, while xorl MoE weights are half-concatenated [gate; up]. Routing the
default forward through it produced garbage outputs (cosine ~0 vs reference)
with no crash — OPD loss 4.75 instead of 0.54 — and contradicted the op's own
half-concat backward. Also covers the backward gradient-arity regression
(missing swiglu_limit grad -> "expected 14, got 13").

Registration alone is NOT correctness: any backend swap must pass this kind of
output-and-gradient parity before being trusted in a training config.
"""

import pytest
import torch
import torch.nn.functional as F


DTYPE = torch.bfloat16
E, H, I = 4, 256, 512

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _imports():
    try:
        from xorl.ops.moe.quack import QuackEPGroupGemm  # noqa: PLC0415
        from xorl.ops.moe.triton import TritonEPGroupGemm  # noqa: PLC0415

        return QuackEPGroupGemm, TritonEPGroupGemm
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot import EP group GEMM ops: {exc}")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize(
    "tokens_per_expert",
    [
        [13, 0, 7, 21],  # uneven with an empty expert
        [64, 64, 64, 64],  # balanced
        [1, 127, 0, 32],  # extreme skew + empty expert
    ],
)
@pytest.mark.parametrize("with_scores", [False, True])
def test_quack_ep_group_gemm_matches_triton(tokens_per_expert, with_scores):
    QuackEPGroupGemm, TritonEPGroupGemm = _imports()
    torch.manual_seed(0)
    cumsum = torch.tensor(tokens_per_expert, device="cuda").cumsum(0)
    M = int(cumsum[-1])
    x = torch.randn(M, H, device="cuda", dtype=DTYPE)
    gate_up = torch.randn(E, H, 2 * I, device="cuda", dtype=DTYPE) * 0.02
    down = torch.randn(E, I, H, device="cuda", dtype=DTYPE) * 0.02
    scores = torch.rand(M, device="cuda", dtype=torch.float32) if with_scores else None

    def run(cls):
        x_ = x.clone().requires_grad_(True)
        g = gate_up.clone().requires_grad_(True)
        d = down.clone().requires_grad_(True)
        out = cls.apply(x_, cumsum, g, d, I, scores)
        # Non-trivial upstream gradient; also exercises the backward grad-arity
        # contract (a missing grad raises "returned an incorrect number of
        # gradients" here).
        out.float().pow(2).sum().backward()
        return out.detach(), x_.grad, g.grad, d.grad

    quack_tensors = run(QuackEPGroupGemm)
    triton_tensors = run(TritonEPGroupGemm)
    for name, q, t in zip(("out", "grad_x", "grad_gate_up", "grad_down"), quack_tensors, triton_tensors):
        cos = _cos(q, t)
        assert cos > 0.999, f"{name}: quack/triton cosine {cos:.6f} (tokens_per_expert={tokens_per_expert})"


@requires_gpu
@pytest.mark.gpu
def test_quack_ep_forward_matches_halfconcat_reference():
    """Pin the gate/up convention itself: silu(h[:, :I]) * h[:, I:] (half-concat)."""
    QuackEPGroupGemm, _ = _imports()
    torch.manual_seed(1)
    cumsum = torch.tensor([13, 0, 7, 21], device="cuda").cumsum(0)
    M = int(cumsum[-1])
    x = torch.randn(M, H, device="cuda", dtype=DTYPE)
    gate_up = torch.randn(E, H, 2 * I, device="cuda", dtype=DTYPE) * 0.02
    down = torch.randn(E, I, H, device="cuda", dtype=DTYPE) * 0.02

    ref = torch.zeros(M, H, device="cuda", dtype=torch.float32)
    start = 0
    for e in range(E):
        end = int(cumsum[e])
        if end > start:
            h = x[start:end].float() @ gate_up[e].float()
            ref[start:end] = (F.silu(h[:, :I]) * h[:, I:]) @ down[e].float()
        start = end

    with torch.no_grad():
        out = QuackEPGroupGemm.apply(x, cumsum, gate_up, down, I)
    cos = _cos(out, ref)
    assert cos > 0.999, f"quack forward does not implement half-concat [gate; up] gating (cosine {cos:.6f})"
