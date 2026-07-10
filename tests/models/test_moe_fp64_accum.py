"""Tests for the XORL_MOE_FP64_ACCUM K3-parity mode.

Validates:
1. Flag off: `_eager_forward` output is unchanged (default path).
2. Flag on: output matches an independent fp64 reference bit-for-bit.
3. Order-invariance: the fp64 path gives identical bf16 output when the down
   GEMM reduction is chunked differently in the reference.
4. Guards: non-SiLU / biased / swiglu_limit experts raise NotImplementedError.
"""

import pytest
import torch

from xorl.models.layers.moe.moe_block import MoEBlock


pytestmark = [pytest.mark.cpu]


NUM_EXPERTS, TOP_K, HID, INTER, TOKENS = 8, 2, 32, 24, 16


@pytest.fixture()
def block():
    torch.manual_seed(0)
    blk = MoEBlock(
        hidden_size=HID,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTER,
        hidden_act="silu",
        moe_implementation="eager",
    )
    with torch.no_grad():
        blk.experts.gate_up_proj.normal_(std=0.5)
        blk.experts.down_proj.normal_(std=0.5)
    return blk.to(torch.bfloat16)


@pytest.fixture()
def routed_inputs():
    torch.manual_seed(1)
    x = torch.randn(TOKENS, HID, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(TOKENS)])
    w = torch.rand(TOKENS, TOP_K, dtype=torch.bfloat16)
    return x, w, ids


def _fp64_reference(blk: MoEBlock, x, w, ids, down_chunks: int = 1):
    """Per-token fp64 reference with the offline harness cast contract."""
    gate_proj = blk.experts.gate_proj
    up_proj = blk.experts.up_proj
    down_proj = blk.experts.down_proj
    out = torch.zeros(TOKENS, HID, dtype=torch.float64)
    for t in range(TOKENS):
        acc = torch.zeros(HID, dtype=torch.float64)
        for j in range(TOP_K):
            e = int(ids[t, j])
            xe = x[t].to(torch.float64)
            gate = xe @ gate_proj[e].to(torch.float64)
            up = xe @ up_proj[e].to(torch.float64)
            h = ((gate * torch.sigmoid(gate)) * up).to(torch.bfloat16)
            h64 = h.to(torch.float64)
            if down_chunks == 1:
                o = h64 @ down_proj[e].to(torch.float64)
            else:
                step = INTER // down_chunks
                o = torch.zeros(HID, dtype=torch.float64)
                for c in range(down_chunks):
                    s = c * step
                    o = o + h64[s : s + step] @ down_proj[e][s : s + step].to(torch.float64)
            acc = acc + w[t, j].to(torch.float64) * o
        out[t] = acc
    return out.to(torch.bfloat16)


def test_flag_off_keeps_default_eager_path(block, routed_inputs, monkeypatch):
    monkeypatch.delenv("XORL_MOE_FP64_ACCUM", raising=False)
    x, w, ids = routed_inputs
    baseline = block._eager_forward(x, w, ids)
    assert baseline.dtype == torch.bfloat16
    fp64 = block._eager_forward_fp64(x, w, ids)
    # The parity path is numerically different from the bf16 default (that is
    # its purpose); the default path must stay bit-stable with the flag off.
    monkeypatch.setenv("XORL_MOE_FP64_ACCUM", "0")
    assert torch.equal(block._eager_forward(x, w, ids), baseline)
    assert fp64.dtype == torch.bfloat16


def test_fp64_path_matches_reference_bitwise(block, routed_inputs, monkeypatch):
    monkeypatch.setenv("XORL_MOE_FP64_ACCUM", "1")
    x, w, ids = routed_inputs
    out = block._eager_forward(x, w, ids)
    assert torch.equal(out, _fp64_reference(block, x, w, ids))


def test_fp64_path_is_order_invariant(block, routed_inputs):
    x, w, ids = routed_inputs
    out = block._eager_forward_fp64(x, w, ids)
    for chunks in (2, 3):
        assert torch.equal(out, _fp64_reference(block, x, w, ids, down_chunks=chunks))


def test_forward_dispatches_fp64_for_non_eager_backends(routed_inputs, monkeypatch):
    """The flag must override the configured backend (real configs resolve to triton)."""
    torch.manual_seed(0)
    blk = MoEBlock(
        hidden_size=HID,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTER,
        hidden_act="silu",
        moe_implementation="triton",
    )
    with torch.no_grad():
        blk.experts.gate_up_proj.normal_(std=0.5)
        blk.experts.down_proj.normal_(std=0.5)
    blk = blk.to(torch.bfloat16)
    blk.gate = blk.gate.to(torch.bfloat16)

    x, w, ids = routed_inputs
    called = {}
    original = blk._eager_forward_fp64

    def spy(hidden_states, routing_weights, selected_experts):
        called["hit"] = True
        return original(hidden_states, routing_weights, selected_experts)

    monkeypatch.setattr(blk, "_eager_forward_fp64", spy)
    monkeypatch.setenv("XORL_MOE_FP64_ACCUM", "1")
    out, _ = blk.forward(x.view(1, TOKENS, HID))
    assert called.get("hit"), "fp64 parity path must engage for triton-backend blocks"
    assert out.dtype == torch.bfloat16

    called.clear()
    monkeypatch.setenv("XORL_MOE_FP64_ACCUM", "0")
    # triton backend would need GPU kernels; dispatch check only — the fp64
    # spy must NOT be hit with the flag off.
    try:
        blk.forward(x.view(1, TOKENS, HID))
    except Exception:
        pass
    assert not called.get("hit")


def test_fp64_path_rejects_unsupported_experts(block, routed_inputs):
    x, w, ids = routed_inputs
    block.experts.hidden_act = "gelu_tanh"
    with pytest.raises(NotImplementedError):
        block._eager_forward_fp64(x, w, ids)
    block.experts.hidden_act = "silu"
    block.experts.down_bias = torch.zeros(NUM_EXPERTS, HID, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        block._eager_forward_fp64(x, w, ids)
    block.experts.down_bias = None
    block.experts.swiglu_limit = 7.0
    with pytest.raises(NotImplementedError):
        block._eager_forward_fp64(x, w, ids)
