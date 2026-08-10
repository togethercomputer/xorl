"""Scoped batch-invariant trunk-linear contract (XORL_BI_TRUNK_LINEAR).

The trunk contract routes ONLY transformer-trunk nn.Linear forwards through the
batch-invariant persistent GEMM (bit-identical to the aten::mm interpose lane and
therefore to serving) while backward stays on cuBLAS. These tests lock:
  - explicit module selection (trunk projections only; experts skipped; adapters,
    fp8/custom linears and non-bf16 weights raise; idempotent re-wrap),
  - forward bitwise vs matmul_persistent AND vs the global-interpose F.linear,
  - backward bitwise vs the cuBLAS autograd reference,
  - the loud-fail: the global interpose raises on grad-requiring inputs
    in a grad-enabled context and keeps working under torch.no_grad().
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.lora.modules.linear import LoraLinear
from xorl.ops.batch_invariant_ops import (
    is_trunk_linear_contract_enabled,
    matmul_persistent,
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
    set_trunk_linear_contract,
    wrap_trunk_linears_batch_invariant,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HIDDEN, INTER, VOCAB = 256, 512, 1024


class _TrunkBlock(nn.Module):
    def __init__(self, dtype=torch.bfloat16, bias=False):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=bias, dtype=dtype)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=bias, dtype=dtype)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=bias, dtype=dtype)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=bias, dtype=dtype)
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=bias, dtype=dtype)
        self.up_proj = nn.Linear(HIDDEN, INTER, bias=bias, dtype=dtype)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=bias, dtype=dtype)

    def forward(self, x):
        y = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        return self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))


class _TrunkModel(nn.Module):
    def __init__(self, n_layers=2, dtype=torch.bfloat16):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN, dtype=dtype)
        self.layers = nn.ModuleList(_TrunkBlock(dtype=dtype) for _ in range(n_layers))
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False, dtype=dtype)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.fixture(autouse=True)
def _reset_contract_state():
    yield
    set_trunk_linear_contract(False)


# --------------------------------------------------------------------------- #
# Selection (CPU)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
def test_wrap_selection_counts_and_exclusions():
    model = _TrunkModel(n_layers=2)
    wrapped = wrap_trunk_linears_batch_invariant(model)
    assert wrapped == dict.fromkeys(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), 2)
    assert not getattr(model.lm_head, "_xorl_bi_trunk_wrapped", False)
    assert all(getattr(layer.q_proj, "_xorl_bi_trunk_wrapped", False) for layer in model.layers)
    assert is_trunk_linear_contract_enabled()


@pytest.mark.cpu
def test_wrap_is_idempotent():
    model = _TrunkModel(n_layers=1)
    first = wrap_trunk_linears_batch_invariant(model)
    assert sum(first.values()) == 7
    second = wrap_trunk_linears_batch_invariant(model)
    assert second == {}, "re-wrap must be a no-op, not a double-wrap"


@pytest.mark.cpu
def test_wrap_skips_routed_experts():
    model = _TrunkModel(n_layers=1)
    experts = nn.ModuleList(
        nn.ModuleDict({"gate_proj": nn.Linear(HIDDEN, INTER, dtype=torch.bfloat16)}) for _ in range(2)
    )
    model.layers[0].add_module("experts", experts)
    wrapped = wrap_trunk_linears_batch_invariant(model)
    assert sum(wrapped.values()) == 7, "routed experts must not be wrapped (contracted via the fused sglang path)"
    assert not getattr(model.layers[0].experts[0].gate_proj, "_xorl_bi_trunk_wrapped", False)


@pytest.mark.cpu
def test_wrap_raises_on_no_match():
    with pytest.raises(RuntimeError, match="matched no trunk linears"):
        wrap_trunk_linears_batch_invariant(nn.Linear(4, 4))


@pytest.mark.cpu
def test_wrap_raises_on_lora_wrapped_module():
    model = _TrunkModel(n_layers=1)
    model.layers[0].q_proj = LoraLinear(HIDDEN, HIDDEN, r=4, lora_alpha=8, bias=False, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="adapter-wrapped"):
        wrap_trunk_linears_batch_invariant(model)


@pytest.mark.cpu
def test_wrap_raises_on_linear_subclass():
    class _FakeFP8Linear(nn.Linear):
        pass

    model = _TrunkModel(n_layers=1)
    model.layers[0].up_proj = _FakeFP8Linear(HIDDEN, INTER, bias=False, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="not a plain"):
        wrap_trunk_linears_batch_invariant(model)


@pytest.mark.cpu
def test_wrap_raises_on_non_bf16_weight():
    # fp32 is tolerated at wrap time (mixed-precision master; mp_policy casts to
    # bf16 before forward and the runtime guard enforces bf16 GEMM operands), but
    # fp16 and other dtypes are outside the contract.
    model = _TrunkModel(n_layers=1)
    model.layers[0].down_proj = nn.Linear(INTER, HIDDEN, bias=False, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="bf16-only"):
        wrap_trunk_linears_batch_invariant(model)


@requires_cuda
@pytest.mark.gpu
def test_wrap_raises_under_global_interpose():
    model = _TrunkModel(n_layers=1)
    with set_batch_invariant_mode(True):
        with pytest.raises(RuntimeError, match="cannot be combined"):
            wrap_trunk_linears_batch_invariant(model)


# --------------------------------------------------------------------------- #
# Forward contract (GPU)
# --------------------------------------------------------------------------- #
def _wrapped_linear(bias=False, seed=0):
    torch.manual_seed(seed)
    holder = nn.ModuleDict({"q_proj": nn.Linear(HIDDEN, HIDDEN, bias=bias, dtype=torch.bfloat16)}).cuda()
    wrap_trunk_linears_batch_invariant(holder)
    return holder["q_proj"]


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("bias", [False, True])
def test_forward_bitwise_matches_matmul_persistent(bias):
    lin = _wrapped_linear(bias=bias)
    x = torch.randn(4, 64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = lin(x)
        ref = matmul_persistent(x.reshape(-1, HIDDEN), lin.weight.t(), bias=lin.bias)
    assert torch.equal(out, ref.reshape(4, 64, HIDDEN))


@requires_cuda
@pytest.mark.gpu
def test_forward_bitwise_matches_global_interpose_lane():
    # The wrapped forward must produce the SAME bits as F.linear under the global
    # aten::mm interpose (the serving/verification lane it replaces for training).
    lin = _wrapped_linear(bias=False, seed=1)
    x = torch.randn(2, 128, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = lin(x)
        with set_batch_invariant_mode(True):
            ref = F.linear(x, lin.weight)
    assert torch.equal(out, ref)


@requires_cuda
@pytest.mark.gpu
def test_forward_is_batch_invariant():
    lin = _wrapped_linear(seed=2)
    x = torch.randn(300, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        full = lin(x)
        sub = lin(x[:7].contiguous())
    assert torch.equal(full[:7], sub)


@requires_cuda
@pytest.mark.gpu
def test_forward_raises_on_non_bf16_input():
    lin = _wrapped_linear(seed=3)
    with pytest.raises(RuntimeError, match="bf16-only"):
        lin(torch.randn(8, HIDDEN, device="cuda", dtype=torch.float32))


# --------------------------------------------------------------------------- #
# Backward contract (GPU): bitwise vs cuBLAS autograd
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("bias", [False, True])
def test_backward_bitwise_matches_cublas_autograd(bias):
    lin = _wrapped_linear(bias=bias, seed=4)
    x0 = torch.randn(2, 96, HIDDEN, device="cuda", dtype=torch.bfloat16)
    g_out = torch.randn(2, 96, HIDDEN, device="cuda", dtype=torch.bfloat16)

    x = x0.clone().requires_grad_(True)
    out = lin(x)
    out.backward(g_out)

    x_ref = x0.clone().requires_grad_(True)
    w_ref = lin.weight.detach().clone().requires_grad_(True)
    b_ref = lin.bias.detach().clone().requires_grad_(True) if bias else None
    out_ref = F.linear(x_ref, w_ref, b_ref)
    out_ref.backward(g_out)

    assert torch.equal(x.grad, x_ref.grad), "grad_input must stay bitwise on the cuBLAS path"
    assert torch.equal(lin.weight.grad, w_ref.grad), "grad_weight must stay bitwise on the cuBLAS path"
    if bias:
        assert torch.equal(lin.bias.grad, b_ref.grad)


# --------------------------------------------------------------------------- #
# The global interpose loud-fails on training forwards
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.gpu
def test_global_interpose_raises_on_grad_requiring_inputs():
    x = torch.randn(32, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wn = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xb = torch.randn(2, 16, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with set_batch_invariant_mode(True):
        with pytest.raises(RuntimeError, match="XORL_BI_TRUNK_LINEAR"):
            _ = x @ w
        with pytest.raises(RuntimeError, match="XORL_BI_TRUNK_LINEAR"):
            _ = torch.rms_norm(x, (HIDDEN,), wn, 1e-6)
        with pytest.raises(RuntimeError, match="XORL_BI_TRUNK_LINEAR"):
            _ = torch.bmm(xb, xb.transpose(1, 2))
        with pytest.raises(RuntimeError, match="XORL_BI_TRUNK_LINEAR"):
            _ = torch.log_softmax(x.float(), dim=-1)
        with pytest.raises(RuntimeError, match="XORL_BI_TRUNK_LINEAR"):
            _ = x.float().mean(-1)


@requires_cuda
@pytest.mark.gpu
def test_global_interpose_still_works_under_no_grad():
    torch.manual_seed(5)
    x = torch.randn(32, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wn = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with set_batch_invariant_mode(True), torch.no_grad():
        out_mm = x @ w
        out_norm = torch.rms_norm(x, (HIDDEN,), wn, 1e-6)
    with torch.no_grad():
        assert torch.equal(out_mm, matmul_persistent(x, w))
        assert torch.equal(out_norm, rms_norm_batch_invariant(x, wn, eps=1e-6))


@requires_cuda
@pytest.mark.gpu
def test_global_interpose_silent_before_now_loud():
    # Regression pin for the guarded bug: without the guard, the aten::rms_norm
    # override returned an output disconnected from the autograd graph (q/k-norm grads
    # silently vanished). The op must now refuse instead of detaching.
    x = torch.randn(16, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wn = torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True):
        with pytest.raises(RuntimeError, match="rms_norm"):
            F.rms_norm(x, (HIDDEN,), wn, eps=1e-6)


@requires_cuda
@pytest.mark.gpu
def test_global_interpose_allows_grad_free_inputs_in_grad_context():
    # Verification flows that forward non-leaf, grad-free tensors inside a
    # grad-enabled context must keep working.
    x = torch.randn(32, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with set_batch_invariant_mode(True):
        out = x @ w
    assert torch.equal(out, matmul_persistent(x, w))
