import pytest
import torch

from xorl.models.layers.moe.moe_block import MoEBlock, _BIRouterGemm
from xorl.ops.batch_invariant_ops import bi_router_gemm, bi_router_topk_weights


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

H, E, TOP_K = 2048, 128, 8


def _inputs(n, seed=0):
    torch.manual_seed(seed)
    hidden = (torch.randn(n, H, device="cuda") * 0.5).to(torch.bfloat16)
    weight = (torch.randn(E, H, device="cuda") * (H**-0.5)).to(torch.bfloat16)
    return hidden, weight


@requires_cuda
@pytest.mark.gpu
def test_router_gemm_matches_fp32_upcast_reference():
    # bf16xbf16 products are exact in fp32, so the contract kernel must equal an
    # fp32 GEMM over the (exact) fp32 upcasts up to fp32 reduction-order noise.
    hidden, weight = _inputs(256)
    out = bi_router_gemm(hidden, weight)
    assert out.dtype == torch.float32 and out.shape == (256, E)
    ref = torch.nn.functional.linear(hidden.float(), weight.float())
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-4)


@requires_cuda
@pytest.mark.gpu
def test_router_gemm_batch_invariant_to_padding():
    # A row's logits must not depend on how many other rows share the batch
    # (batch invariance): rows of a small batch equal the same rows of a big one.
    hidden, weight = _inputs(300)
    full = bi_router_gemm(hidden, weight)
    sub = bi_router_gemm(hidden[:7].contiguous(), weight)
    assert torch.equal(full[:7], sub)


@requires_cuda
@pytest.mark.gpu
def test_router_gemm_empty_tokens():
    _, weight = _inputs(1)
    empty = torch.empty(0, H, device="cuda", dtype=torch.bfloat16)
    out = bi_router_gemm(empty, weight)
    assert out.shape == (0, E) and out.dtype == torch.float32


@requires_cuda
@pytest.mark.gpu
def test_router_gemm_autograd_backward_matches_linear():
    hidden, weight = _inputs(64, seed=1)

    h1 = hidden.clone().requires_grad_(True)
    w1 = weight.clone().requires_grad_(True)
    _BIRouterGemm.apply(h1, w1).square().sum().backward()

    h2 = hidden.clone().requires_grad_(True)
    w2 = weight.clone().requires_grad_(True)
    torch.nn.functional.linear(h2.float(), w2.float()).square().sum().backward()

    # grads are cast back to bf16 (the param dtype), so compare at bf16 precision
    assert torch.allclose(h1.grad.float(), h2.grad.float(), rtol=2e-2, atol=2e-2)
    assert torch.allclose(w1.grad.float(), w2.grad.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.gpu
def test_router_gemm_rejects_non_bf16():
    hidden, weight = _inputs(8)
    with pytest.raises(AssertionError):
        bi_router_gemm(hidden.float(), weight)
    with pytest.raises(AssertionError):
        bi_router_gemm(hidden, weight.float())


@requires_cuda
@pytest.mark.gpu
def test_topk_weights_fixed_order_renorm():
    torch.manual_seed(0)
    vals = torch.rand(128, TOP_K, device="cuda", dtype=torch.float32) + 0.1
    w = bi_router_topk_weights(vals, norm_topk_prob=True, out_dtype=torch.bfloat16)
    assert w.dtype == torch.bfloat16
    # sums to 1 up to bf16 rounding
    assert torch.allclose(w.float().sum(-1), torch.ones(128, device="cuda"), atol=2e-2)
    # no-renorm path is a pure cast
    w2 = bi_router_topk_weights(vals, norm_topk_prob=False, out_dtype=torch.bfloat16)
    assert torch.equal(w2, vals.to(torch.bfloat16))


@requires_cuda
@pytest.mark.gpu
def test_topk_weights_requires_fp32():
    vals = torch.rand(4, TOP_K, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        bi_router_topk_weights(vals)


@requires_cuda
@pytest.mark.gpu
def test_moe_block_route_uses_contract_when_enabled(monkeypatch):
    # Module-level gate: MoEBlock.route with XORL_MOE_BI_ROUTER=1 must produce
    # router logits equal to the standalone contract kernel, and selection/
    # weights equal to the contract post-processing on those logits.
    block = (
        MoEBlock(
            hidden_size=H,
            num_experts=E,
            top_k=TOP_K,
            intermediate_size=768,
            moe_implementation="eager",
            norm_topk_prob=True,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    hidden = (torch.randn(140, H, device="cuda") * 0.5).to(torch.bfloat16)

    monkeypatch.setenv("XORL_MOE_BI_ROUTER", "1")
    rw, sel, logits = block.route(hidden)

    ref_logits = bi_router_gemm(hidden, block.gate.weight)
    assert torch.equal(logits, ref_logits)

    probs = torch.softmax(ref_logits, dim=1, dtype=torch.float)
    ref_vals, ref_sel = torch.topk(probs, TOP_K, dim=-1)
    ref_w = bi_router_topk_weights(ref_vals, True, torch.bfloat16)
    assert torch.equal(sel, ref_sel)
    assert torch.equal(rw, ref_w)


@requires_cuda
@pytest.mark.gpu
def test_moe_block_route_default_path_unchanged(monkeypatch):
    # Flag off -> stock gate GEMM, logits are bf16 (nn.Linear), not fp32.
    monkeypatch.delenv("XORL_MOE_BI_ROUTER", raising=False)
    block = (
        MoEBlock(
            hidden_size=H,
            num_experts=E,
            top_k=TOP_K,
            intermediate_size=768,
            moe_implementation="eager",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    hidden = (torch.randn(16, H, device="cuda") * 0.5).to(torch.bfloat16)
    _, _, logits = block.route(hidden)
    assert logits.dtype == torch.bfloat16
