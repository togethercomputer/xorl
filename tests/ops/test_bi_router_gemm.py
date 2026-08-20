import pytest
import torch

from xorl.models.layers.moe.moe_block import MoEBlock, _BIRouterGemm
from xorl.ops.batch_invariant_ops import bi_bf16_fp32_linear, bi_router_gemm, bi_router_topk_weights


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

H, E, TOP_K = 2048, 128, 8


def _inputs(n, seed=0):
    torch.manual_seed(seed)
    hidden = (torch.randn(n, H, device="cuda") * 0.5).to(torch.bfloat16)
    weight = (torch.randn(E, H, device="cuda") * (H**-0.5)).to(torch.bfloat16)
    return hidden, weight


@requires_cuda
@pytest.mark.gpu
def test_router_gemm_linear_topk_and_moe_dispatch_policy():
    # bf16xbf16 products are exact in fp32, so the contract kernel must equal an
    # fp32 GEMM over the (exact) fp32 upcasts up to fp32 reduction-order noise.
    hidden, weight = _inputs(256)
    out = bi_router_gemm(hidden, weight)
    assert out.dtype == torch.float32 and out.shape == (256, E)
    ref = torch.nn.functional.linear(hidden.float(), weight.float())
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-4)

    empty = bi_router_gemm(torch.empty(0, H, device="cuda", dtype=torch.bfloat16), weight)
    assert empty.shape == (0, E) and empty.dtype == torch.float32
    with pytest.raises(AssertionError):
        bi_router_gemm(hidden.float(), weight)
    with pytest.raises(AssertionError):
        bi_router_gemm(hidden, weight.float())

    _assert_router_gemm_autograd_backward_matches_linear()
    _assert_bf16_fp32_linear_preserves_leading_dims_and_backward()
    _assert_topk_weights_renormalize_or_cast_and_require_fp32()
    _assert_moe_block_route_selects_exact_contract_or_default_path()


def _assert_router_gemm_autograd_backward_matches_linear():
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


def _assert_bf16_fp32_linear_preserves_leading_dims_and_backward():
    hidden, weight = _inputs(6, seed=2)
    hidden = hidden.reshape(2, 3, H).requires_grad_(True)
    weight = weight.requires_grad_(True)

    out = bi_bf16_fp32_linear(hidden, weight)
    assert out.shape == (2, 3, E)
    assert out.dtype is torch.float32
    assert torch.equal(out.reshape(6, E), bi_router_gemm(hidden.detach().reshape(6, H), weight.detach()))

    out.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.dtype is torch.bfloat16
    assert weight.grad is not None and weight.grad.dtype is torch.bfloat16


def _assert_topk_weights_renormalize_or_cast_and_require_fp32():
    torch.manual_seed(0)
    vals = torch.rand(128, TOP_K, device="cuda", dtype=torch.float32) + 0.1
    w = bi_router_topk_weights(vals, norm_topk_prob=True, out_dtype=torch.bfloat16)
    assert w.dtype == torch.bfloat16
    # sums to 1 up to bf16 rounding
    assert torch.allclose(w.float().sum(-1), torch.ones(128, device="cuda"), atol=2e-2)
    # no-renorm path is a pure cast
    w2 = bi_router_topk_weights(vals, norm_topk_prob=False, out_dtype=torch.bfloat16)
    assert torch.equal(w2, vals.to(torch.bfloat16))
    with pytest.raises(AssertionError):
        bi_router_topk_weights(vals.to(torch.bfloat16))


def _assert_moe_block_route_selects_exact_contract_or_default_path():
    # Exact MoEBlock routing must produce
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
            exact_batch_invariant_router=True,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    hidden = (torch.randn(140, H, device="cuda") * 0.5).to(torch.bfloat16)

    with torch.no_grad():
        rw, sel, logits = block.route(hidden)

        ref_logits = bi_router_gemm(hidden, block.gate.weight)
        assert torch.equal(logits, ref_logits)

        probs = torch.softmax(ref_logits, dim=1, dtype=torch.float)
        ref_vals, ref_sel = torch.topk(probs, TOP_K, dim=-1)
        ref_w = bi_router_topk_weights(ref_vals, True, torch.bfloat16)
        assert torch.equal(sel, ref_sel)
        assert torch.equal(rw, ref_w)

        # Prove batch composition through the production router, not only the
        # standalone GEMM helper.
        sub_rw, sub_sel, sub_logits = block.route(hidden[:7].contiguous())
        assert torch.equal(sub_logits, logits[:7])
        assert torch.equal(sub_sel, sel[:7])
        assert torch.equal(sub_rw, rw[:7])

    del block, rw, sel, logits, ref_logits, probs, ref_vals, ref_sel, ref_w

    # Ordinary models retain the stock gate GEMM; logits are bf16, not fp32.
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
    with torch.no_grad():
        _, _, logits = block.route(hidden)
    assert logits.dtype == torch.bfloat16
