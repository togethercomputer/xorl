"""Tests for the XORL_MOE_SGLANG_FUSED_EXPERTS K3-parity mode.

Validates:
1. Flag off: forward output is unchanged (default path), and the SGLang path is
   never engaged.
2. Flag on: MoEBlock.forward / forward_experts_only dispatch to
   ``MoEExperts.sglang_fused_experts_forward`` for triton-backend blocks
   (real configs resolve ``_moe_implementation`` to triton).
3. Guards: non-gated / down-bias / unsupported-activation experts raise
   NotImplementedError; a missing sglang install raises ImportError naming the
   flag.
4. Layout: the transient transposes hand SGLang's expected ``w13 [E, 2I, H]``
   (gate-first) / ``w2 [E, H, I]`` weights and fp32 top-k weights to the kernel
   (kernel call is faked; numerical parity is covered by separate parity gates).
5. Weight modes: ``XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE`` selection
   (strided default, legacy cache env aliases cached, explicit mode wins,
   invalid rejects); strided mode hands zero-copy GKN transpose-views to the
   kernel and never populates the cache; strided forward+grads bit-identical
   to transient (GPU); the adapter pins SGLang's split gate/up layout contract.
6. Auto default (env unset): resolves per-regime — enabled at ep=1 (CUDA input,
   supported module, importable stack), disabled at EP>1; explicit 1/0 always
   wins; the resolution is logged once; (GPU) the auto path dispatches to the
   parity forward, is deterministic across runs, and is bit-identical to the
   explicit-flag path.
"""

import inspect
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from xorl.models.layers.moe import experts as experts_mod
from xorl.models.layers.moe.moe_block import MoEBlock


pytestmark = [pytest.mark.cpu]


NUM_EXPERTS, TOP_K, HID, INTER, TOKENS = 8, 2, 32, 24, 16
FLAG = "XORL_MOE_SGLANG_FUSED_EXPERTS"


def _block(moe_implementation: str = "eager") -> MoEBlock:
    torch.manual_seed(0)
    blk = MoEBlock(
        hidden_size=HID,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTER,
        hidden_act="silu",
        moe_implementation=moe_implementation,
    )
    with torch.no_grad():
        blk.experts.gate_up_proj.normal_(std=0.5)
        blk.experts.down_proj.normal_(std=0.5)
    blk = blk.to(torch.bfloat16)
    blk.gate = blk.gate.to(torch.bfloat16)
    # inference-mode default: the trainable dispatch is exercised explicitly
    blk.requires_grad_(False)
    return blk


@pytest.fixture()
def routed_inputs():
    torch.manual_seed(1)
    x = torch.randn(TOKENS, HID, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(TOKENS)])
    w = torch.rand(TOKENS, TOP_K, dtype=torch.bfloat16)
    return x, w, ids


@pytest.fixture()
def auto_state(monkeypatch):
    """Reset the auto-resolution log latch and the cached stack probe."""
    monkeypatch.setattr(experts_mod, "_MOE_SGLANG_FUSED_EXPERTS_AUTO_LOGGED", False)
    monkeypatch.setattr(experts_mod, "_MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE", None)


def _stack_available(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(experts_mod, "_MOE_SGLANG_FUSED_EXPERTS_STACK_AVAILABLE", value)


class _LoggerSpy:
    """Captures the module logger directly (xorl loggers don't propagate to caplog)."""

    def __init__(self):
        self.records: list[tuple[int, str]] = []

    def info(self, msg, *args):
        self.records.append((logging.INFO, msg % args if args else msg))

    def warning(self, msg, *args):
        self.records.append((logging.WARNING, msg % args if args else msg))


@pytest.fixture()
def log_spy(monkeypatch):
    spy = _LoggerSpy()
    monkeypatch.setattr(experts_mod, "logger", spy)
    return spy


def test_auto_resolution_ep1_enables_and_logs_once(monkeypatch, auto_state, log_spy):
    monkeypatch.delenv(FLAG, raising=False)
    _stack_available(monkeypatch, True)
    assert experts_mod.moe_sglang_fused_experts_enabled(1, torch.device("cuda")) is True
    assert experts_mod.moe_sglang_fused_experts_enabled(1, torch.device("cuda")) is True
    logged = [m for _, m in log_spy.records if "auto-enabled (ep=1)" in m]
    assert len(logged) == 1, "auto resolution must be logged exactly once"


def test_auto_resolution_ep_gt1_disables(monkeypatch, auto_state, log_spy):
    monkeypatch.delenv(FLAG, raising=False)
    _stack_available(monkeypatch, True)
    assert experts_mod.moe_sglang_fused_experts_enabled(8, torch.device("cuda")) is False
    assert any("auto-disabled (ep=8)" in m for _, m in log_spy.records)


def test_explicit_env_overrides_auto(monkeypatch, auto_state, log_spy):
    _stack_available(monkeypatch, True)
    monkeypatch.setenv(FLAG, "0")
    assert experts_mod.moe_sglang_fused_experts_enabled(1, torch.device("cuda")) is False
    monkeypatch.setenv(FLAG, "1")
    assert experts_mod.moe_sglang_fused_experts_enabled(8, torch.device("cuda")) is True
    assert not log_spy.records, "explicit 1/0 must not emit the auto-resolution log"


def test_auto_requires_cuda_input_and_stack(monkeypatch, auto_state, log_spy):
    monkeypatch.delenv(FLAG, raising=False)
    # CPU/meta inputs keep the stock path quietly (unit tests, tracing).
    _stack_available(monkeypatch, True)
    assert experts_mod.moe_sglang_fused_experts_enabled(1, torch.device("cpu")) is False
    assert experts_mod.moe_sglang_fused_experts_enabled(1, torch.device("meta")) is False
    assert not log_spy.records
    # Missing sglang/sgl_kernel at ep=1 warns: the default is NOT serving-bit-exact.
    _stack_available(monkeypatch, False)
    assert experts_mod.moe_sglang_fused_experts_enabled(1, torch.device("cuda")) is False
    warned = [(lvl, m) for lvl, m in log_spy.records if "not importable" in m]
    assert warned and warned[0][0] == logging.WARNING


def test_auto_requires_supported_expert_module(monkeypatch, auto_state):
    monkeypatch.delenv(FLAG, raising=False)
    _stack_available(monkeypatch, True)
    cuda = torch.device("cuda")

    blk = _block("triton")
    assert blk.experts.sglang_fused_experts_auto_supported()
    assert experts_mod.moe_sglang_fused_experts_enabled(1, cuda, blk.experts) is True

    for attr, bad in (
        ("gated", False),
        ("down_bias", torch.zeros(1)),
        ("gate_up_bias", torch.zeros(1)),
        ("hidden_act", "relu2"),
        ("swiglu_limit", 1.0),
        ("fp8_training_enabled", True),
    ):
        blk = _block("triton")
        setattr(blk.experts, attr, bad)
        assert not blk.experts.sglang_fused_experts_auto_supported(), attr
        assert experts_mod.moe_sglang_fused_experts_enabled(1, cuda, blk.experts) is False, attr

    # Modules without the eligibility hook (e.g. LoRA experts) never auto-enable.
    assert experts_mod.moe_sglang_fused_experts_enabled(1, cuda, object()) is False


def test_flag_off_keeps_default_path(routed_inputs, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    blk = _block("eager")
    x, _, _ = routed_inputs
    baseline, _ = blk.forward(x.view(1, TOKENS, HID))

    called = {}

    def spy(*args, **kwargs):
        called["hit"] = True
        raise AssertionError("sglang path must not engage with the flag off")

    monkeypatch.setattr(blk.experts, "sglang_fused_experts_forward", spy)
    monkeypatch.setenv(FLAG, "0")
    out, _ = blk.forward(x.view(1, TOKENS, HID))
    assert not called.get("hit")
    assert torch.equal(out, baseline)


def test_forward_dispatches_sglang_path_for_non_eager_backends(routed_inputs, monkeypatch):
    """The flag must override the configured backend (real configs resolve to triton)."""
    blk = _block("triton")
    x, w, ids = routed_inputs
    called = {}

    def fake(hidden_states, routing_weights, selected_experts):
        called["hit"] = True
        called["shape"] = tuple(hidden_states.shape)
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(blk.experts, "sglang_fused_experts_forward", fake)
    monkeypatch.setenv(FLAG, "1")
    out, _ = blk.forward(x.view(1, TOKENS, HID))
    assert called.get("hit"), "sglang parity path must engage for triton-backend blocks"
    assert called["shape"] == (TOKENS, HID)
    assert out.shape == (1, TOKENS, HID)


def test_forward_experts_only_dispatches_sglang_path(routed_inputs, monkeypatch):
    blk = _block("triton")
    x, w, ids = routed_inputs
    called = {}

    def fake(hidden_states, routing_weights, selected_experts):
        called["hit"] = True
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(blk.experts, "sglang_fused_experts_forward", fake)
    monkeypatch.setenv(FLAG, "1")
    out = blk.forward_experts_only(x.view(1, TOKENS, HID), w, ids)
    assert called.get("hit")
    assert out.shape == (1, TOKENS, HID)


def test_fp64_parity_mode_takes_precedence(routed_inputs, monkeypatch):
    blk = _block("triton")
    x, w, ids = routed_inputs
    called = {}

    def fake_sglang(*args, **kwargs):
        called["sglang"] = True
        return torch.zeros_like(x)

    def fake_fp64(hidden_states, routing_weights, selected_experts):
        called["fp64"] = True
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(blk.experts, "sglang_fused_experts_forward", fake_sglang)
    monkeypatch.setattr(blk, "_eager_forward_fp64", fake_fp64)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("XORL_MOE_FP64_ACCUM", "1")
    blk.forward(x.view(1, TOKENS, HID))
    assert called.get("fp64") and not called.get("sglang")


def test_guards_reject_unsupported_experts(routed_inputs, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    blk = _block("eager")
    x, w, ids = routed_inputs

    blk.experts.gated = False
    with pytest.raises(NotImplementedError):
        blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.gated = True

    blk.experts.down_bias = torch.zeros(NUM_EXPERTS, HID, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):
        blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.down_bias = None

    blk.experts.hidden_act = "relu2"
    with pytest.raises(NotImplementedError):
        blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.hidden_act = "silu"

    with pytest.raises(ValueError):
        blk.experts.sglang_fused_experts_forward(x, None, None)


def test_positive_swiglu_limit_fails_before_sglang_load_or_kernel(routed_inputs, monkeypatch):
    """Every SGLang fused-kernel/runner entry must reject XoRL's
    semantically different clamp before importing or invoking SGLang."""
    blk = _block("triton")
    blk.experts.swiglu_limit = 1.0
    x, w, ids = routed_inputs
    cumsum = torch.arange(1, NUM_EXPERTS + 1, dtype=torch.int64)
    local_ids = ids.to(torch.int32)
    parallel_state = SimpleNamespace(tp_size=1)
    loaded = []

    def forbidden_loader():
        loaded.append(True)
        raise AssertionError("positive swiglu_limit must fail before loading SGLang")

    monkeypatch.setattr(type(blk.experts), "_load_sglang_fused_experts_impl", staticmethod(forbidden_loader))
    monkeypatch.setattr(type(blk.experts), "_load_sglang_moe_runner_stack", staticmethod(forbidden_loader))

    entrypoints = (
        lambda: blk.experts.sglang_fused_experts_forward(x, w, ids),
        lambda: blk.experts.sglang_fused_experts_ep_compute(x[:NUM_EXPERTS], cumsum, w[:NUM_EXPERTS, 0]),
        lambda: blk.experts.sglang_ep_native_routed_partial(x, w, local_ids),
        lambda: blk.experts._sglang_moe_tp_sim_sglang_forward(x, w, ids, parallel_state),
        lambda: blk.experts._sglang_moe_tp_sim_sglang_runner_forward(x, w, ids, parallel_state),
    )
    for entrypoint in entrypoints:
        with pytest.raises(NotImplementedError, match="positive swiglu_limit"):
            entrypoint()
    assert loaded == []


def test_missing_sglang_raises_import_error_naming_flag(routed_inputs, monkeypatch):
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec("sglang") is not None:
        pytest.skip("sglang installed; import-error guard not testable here")
    monkeypatch.setenv(FLAG, "1")
    blk = _block("eager")
    x, w, ids = routed_inputs
    with pytest.raises(ImportError, match=FLAG):
        blk.experts.sglang_fused_experts_forward(x, w, ids)


def test_trainable_dispatch_uses_autograd_function(routed_inputs, monkeypatch):
    """When gradients are required, the flag path must route through the
    autograd Function; the no-grad path must keep using the plain kernel call."""
    from xorl.models.layers.moe import experts as experts_mod  # noqa: PLC0415

    blk = _block("triton")
    x, w, ids = routed_inputs
    called = {}

    def fake_apply(*args):
        called["train"] = True
        return torch.zeros_like(args[0])

    def fake_kernel_call(hidden_flat, *args, **kwargs):
        called["plain"] = True
        return torch.zeros_like(hidden_flat)

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(experts_mod._SglangFusedExpertsTrainFunction, "apply", staticmethod(fake_apply))
    monkeypatch.setattr(experts_mod, "_sglang_fused_experts_kernel_call", fake_kernel_call)
    monkeypatch.setattr(
        type(blk.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: lambda *a, **k: None)
    )

    blk.experts.gate_up_proj.requires_grad_(True)
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert called == {"train": True}

    called.clear()
    blk.experts.gate_up_proj.requires_grad_(False)
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert called == {"plain": True}

    called.clear()
    blk.experts.gate_up_proj.requires_grad_(True)
    with torch.no_grad():
        blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert called == {"plain": True}


def test_trainable_guards(routed_inputs, monkeypatch):
    blk = _block("eager")
    x, w, ids = routed_inputs
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(
        type(blk.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: lambda *a, **k: None)
    )
    blk.experts.gate_up_proj.requires_grad_(True)

    blk.experts.gate_up_bias = torch.zeros(NUM_EXPERTS, 2 * INTER, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="gate_up_bias"):
        blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.gate_up_bias = None

    blk.experts.hidden_act = "gelu"
    with pytest.raises(NotImplementedError, match="training supports"):
        blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.hidden_act = "silu"


@pytest.mark.gpu
def test_trainable_grads_match_stock_triton(monkeypatch):
    """dX / dW13 / dW2 / d_topk_weights must be bit-identical to the stock
    triton path's gradients given the same drawn token permutation."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel")
    import xorl.ops.group_gemm.kernel.moe as moe_kernels  # noqa: PLC0415
    import xorl.ops.moe.triton as xorl_triton  # noqa: PLC0415
    from xorl.models.layers.moe.experts import MoEExperts, _SglangFusedExpertsTrainFunction  # noqa: PLC0415
    from xorl.ops.moe.triton import TritonMoeExpertsFunction  # noqa: PLC0415

    device = torch.device("cuda")
    torch.manual_seed(0)
    E, K, H2, I2, M2 = 8, 2, 64, 48, 16
    gu = (torch.randn(E, H2, 2 * I2, device=device) * 0.3).to(torch.bfloat16)
    dn = (torch.randn(E, I2, H2, device=device) * 0.3).to(torch.bfloat16)
    x = (torch.randn(M2, H2, device=device) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=device)[:K] for _ in range(M2)])
    wts = torch.rand(M2, K, device=device).to(torch.bfloat16)

    orig = moe_kernels.moe_index_compute
    pinned = {}

    def pinned_index_compute(expert_index, cumsum_t):
        key = (expert_index.data_ptr(), int(expert_index.numel()))
        if key not in pinned:
            pinned[key] = orig(expert_index, cumsum_t)
        return pinned[key].clone()

    monkeypatch.setattr(moe_kernels, "moe_index_compute", pinned_index_compute)
    monkeypatch.setattr(xorl_triton, "moe_index_compute", pinned_index_compute)

    impl = MoEExperts._load_sglang_fused_experts_impl()
    a = [t.clone().requires_grad_(True) for t in (x, wts, gu, dn)]
    out = _SglangFusedExpertsTrainFunction.apply(a[0], a[1], ids, a[2], a[3], impl, "silu", "silu", 0.0, E)
    grad_out = (torch.randn_like(out.float()) * 0.1).to(out.dtype)
    out.backward(grad_out)

    b = [t.clone().requires_grad_(True) for t in (x, wts, gu, dn)]
    gate_view, up_view = b[2][:, :, :I2], b[2][:, :, I2:]
    out_stock = TritonMoeExpertsFunction.apply(E, b[1], ids, b[0], gate_view, up_view, b[3], b[2], "silu", 0.0, True)
    out_stock.backward(grad_out)

    for name, mine, stock in (
        ("dX", a[0].grad, b[0].grad),
        ("d_topk_weights", a[1].grad, b[1].grad),
        ("dW13_gkn", a[2].grad, b[2].grad),
        ("dW2_gkn", a[3].grad, b[3].grad),
    ):
        assert torch.equal(mine, stock), f"{name} gradient mismatch vs stock triton path"


def _masked_problem(device, seed=0, all_valid=False, all_masked=False):
    """A global-topk problem with one simulated EP rank's local-id view."""
    torch.manual_seed(seed)
    e_global, e_local, k, hid, inter, tokens = 16, 4, 3, 64, 48, 24
    lo = 4  # rank 1's slice [4, 8)
    gu = (torch.randn(e_local, hid, 2 * inter, device=device) * 0.3).to(torch.bfloat16)
    dn = (torch.randn(e_local, inter, hid, device=device) * 0.3).to(torch.bfloat16)
    x = (torch.randn(tokens, hid, device=device) * 0.5).to(torch.bfloat16)
    wts = torch.rand(tokens, k, device=device).to(torch.bfloat16)
    if all_valid:
        gids = torch.stack([lo + torch.randperm(e_local, device=device)[:k] for _ in range(tokens)])
    elif all_masked:
        gids = torch.zeros(tokens, k, dtype=torch.int64, device=device)  # rank 1 owns none
    else:
        gids = torch.stack([torch.randperm(e_global, device=device)[:k] for _ in range(tokens)])
    mapping = torch.full((e_global,), -1, dtype=torch.int32, device=device)
    mapping[lo : lo + e_local] = torch.arange(e_local, dtype=torch.int32, device=device)
    local_ids = mapping[gids]
    return x, wts, local_ids, gu, dn, e_local


@pytest.mark.gpu
def test_masked_trainable_grads_match_compacted_stock():
    """filter_expert=True grads must be bit-identical to the stock (unmasked)
    Function run directly on the compacted valid-pair topk=1 presentation, and
    masked slots' d(topk_weights) must be exactly zero."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel")
    from xorl.models.layers.moe.experts import MoEExperts, _SglangFusedExpertsTrainFunction  # noqa: PLC0415

    device = torch.device("cuda")
    x, wts, local_ids, gu, dn, e_local = _masked_problem(device, seed=3)
    impl = MoEExperts._load_sglang_fused_experts_impl()

    a = [t.clone().requires_grad_(True) for t in (x, wts, gu, dn)]
    out = _SglangFusedExpertsTrainFunction.apply(
        a[0], a[1], local_ids, a[2], a[3], impl, "silu", "silu", 0.0, e_local, None, True
    )
    torch.manual_seed(7)
    grad_out = (torch.randn_like(out.float()) * 0.1).to(out.dtype)
    out.backward(grad_out)

    valid_flat = (local_ids.reshape(-1) >= 0).nonzero(as_tuple=False).squeeze(1)
    pair_token = torch.div(valid_flat, local_ids.shape[1], rounding_mode="floor")
    x_c = x.index_select(0, pair_token)
    ids_c = local_ids.reshape(-1)[valid_flat].reshape(-1, 1).to(torch.int32).contiguous()
    wts_c = wts.reshape(-1)[valid_flat].reshape(-1, 1)

    b = [t.clone().requires_grad_(True) for t in (x_c, wts_c, gu, dn)]
    out_c = _SglangFusedExpertsTrainFunction.apply(
        b[0], b[1], ids_c, b[2], b[3], impl, "silu", "silu", 0.0, e_local, None
    )
    out_c.backward(grad_out.index_select(0, pair_token))

    assert torch.equal(a[2].grad, b[2].grad), "dW13 mismatch vs compacted stock presentation"
    assert torch.equal(a[3].grad, b[3].grad), "dW2 mismatch vs compacted stock presentation"

    dw_full = wts.new_zeros(wts.numel())
    dw_full[valid_flat] = b[1].grad.reshape(-1)
    assert torch.equal(a[1].grad, dw_full.reshape(wts.shape)), "d_topk_weights mismatch"
    masked_flat = (local_ids.reshape(-1) < 0).nonzero(as_tuple=False).squeeze(1)
    assert (a[1].grad.reshape(-1)[masked_flat] == 0).all(), "masked slots must have exact-zero weight grads"

    dx_full = b[0].grad.new_zeros(wts.numel(), x.shape[-1])
    dx_full[valid_flat] = b[0].grad
    dx_ref = dx_full.reshape(x.shape[0], wts.shape[1], -1).sum(dim=1)
    assert torch.equal(a[0].grad, dx_ref), "dX mismatch vs compacted stock presentation"


@pytest.mark.gpu
def test_masked_all_valid_bitwise_matches_unmasked_path():
    """With zero masked slots, the filter_expert lane must produce grads
    bit-identical to the stock (filter_expert=False) lane."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel")
    from xorl.models.layers.moe.experts import MoEExperts, _SglangFusedExpertsTrainFunction  # noqa: PLC0415

    device = torch.device("cuda")
    x, wts, local_ids, gu, dn, e_local = _masked_problem(device, seed=5, all_valid=True)
    assert (local_ids >= 0).all()
    impl = MoEExperts._load_sglang_fused_experts_impl()

    grads = []
    for filter_expert in (True, False):
        t = [v.clone().requires_grad_(True) for v in (x, wts, gu, dn)]
        out = _SglangFusedExpertsTrainFunction.apply(
            t[0], t[1], local_ids.to(torch.int32), t[2], t[3], impl, "silu", "silu", 0.0, e_local, None, filter_expert
        )
        torch.manual_seed(11)
        out.backward((torch.randn_like(out.float()) * 0.1).to(out.dtype))
        grads.append([v.grad for v in t])
    for name, mine, stock in zip(("dX", "d_topk_weights", "dW13", "dW2"), grads[0], grads[1]):
        assert torch.equal(mine, stock), f"{name}: all-valid masked lane diverged from stock lane"


@pytest.mark.gpu
def test_masked_all_masked_zero_output_and_grads():
    """A rank that owns none of the routed experts: zero forward, zero grads."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel")
    from xorl.models.layers.moe.experts import MoEExperts, _SglangFusedExpertsTrainFunction  # noqa: PLC0415

    device = torch.device("cuda")
    x, wts, local_ids, gu, dn, e_local = _masked_problem(device, seed=9, all_masked=True)
    assert (local_ids < 0).all()
    impl = MoEExperts._load_sglang_fused_experts_impl()

    t = [v.clone().requires_grad_(True) for v in (x, wts, gu, dn)]
    out = _SglangFusedExpertsTrainFunction.apply(
        t[0], t[1], local_ids, t[2], t[3], impl, "silu", "silu", 0.0, e_local, None, True
    )
    assert (out == 0).all(), "fully-masked forward must be exactly zero"
    out.backward(torch.ones_like(out))
    for name, g, ref in zip(("dX", "d_topk_weights", "dW13", "dW2"), (v.grad for v in t), (x, wts, gu, dn)):
        assert g is not None and g.shape == ref.shape and (g == 0).all(), f"{name} must be exact zeros"


def test_weight_cache_reuses_and_invalidates(routed_inputs, monkeypatch):
    """Cache mode: transposes are reused across forwards, invalidated on
    in-place parameter updates; transient mode makes fresh copies each forward."""
    blk = _block("eager")
    x, w, ids = routed_inputs
    seen = []

    def fake_impl(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen.append((w13, w2))
        return hidden.clone()

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(type(blk.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    monkeypatch.setattr(type(blk.experts), "_sglang_fused_experts_config_logged", True, raising=False)

    # transient mode -> fresh transpose copies each forward
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "transient")
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS_CACHE_WEIGHTS", raising=False)
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert seen[0][0] is not seen[1][0]
    assert seen[0][0].is_contiguous()

    # legacy cache alias (no explicit mode) -> same transposed tensors reused
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", raising=False)
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_CACHE_WEIGHTS", "1")
    seen.clear()
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert seen[0][0] is seen[1][0] and seen[0][1] is seen[1][1]
    assert torch.equal(seen[0][0], blk.experts.gate_up_proj.transpose(1, 2))

    # in-place parameter update bumps _version -> cache re-materializes and
    # reflects the new values
    with torch.no_grad():
        blk.experts.gate_up_proj.add_(1.0)
    seen.clear()
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert torch.equal(seen[0][0], blk.experts.gate_up_proj.transpose(1, 2))

    # explicit invalidation drops entries
    blk.experts.invalidate_sglang_fused_weight_cache()
    assert blk.experts._sglang_fused_weight_cache == {}


def test_weight_mode_selection(monkeypatch):
    """Default is strided; explicit WEIGHT_MODE wins; legacy cache env aliases
    cached; invalid rejects."""
    from xorl.models.layers.moe.experts import moe_sglang_fused_experts_weight_mode  # noqa: PLC0415

    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", raising=False)
    monkeypatch.delenv("XORL_MOE_SGLANG_FUSED_EXPERTS_CACHE_WEIGHTS", raising=False)
    assert moe_sglang_fused_experts_weight_mode() == "strided"

    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_CACHE_WEIGHTS", "1")
    assert moe_sglang_fused_experts_weight_mode() == "cached"

    # explicit mode overrides the legacy cache alias
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "transient")
    assert moe_sglang_fused_experts_weight_mode() == "transient"
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "strided")
    assert moe_sglang_fused_experts_weight_mode() == "strided"

    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "bogus")
    with pytest.raises(ValueError, match="XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE"):
        moe_sglang_fused_experts_weight_mode()


def test_strided_mode_passes_zero_copy_views(routed_inputs, monkeypatch):
    """Strided mode must hand the kernel transpose-VIEWS of the GKN parameters
    (same storage, non-contiguous, serving element order) and never populate the cache."""
    blk = _block("eager")
    x, w, ids = routed_inputs
    seen = {}

    def fake_impl(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen["w13"] = w13
        seen["w2"] = w2
        seen["gemm1_limit"] = kwargs["gemm1_limit"]
        return hidden.clone()

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", "strided")
    monkeypatch.setattr(type(blk.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    monkeypatch.setattr(type(blk.experts), "_sglang_fused_experts_config_logged", True, raising=False)
    out = blk.experts.sglang_fused_experts_forward(x, w, ids)

    assert out.shape == x.shape
    assert not seen["w13"].is_contiguous() and not seen["w2"].is_contiguous()
    assert seen["w13"].data_ptr() == blk.experts.gate_up_proj.data_ptr()
    assert seen["w2"].data_ptr() == blk.experts.down_proj.data_ptr()
    assert torch.equal(seen["w13"], blk.experts.gate_up_proj.transpose(1, 2))
    assert torch.equal(seen["w2"], blk.experts.down_proj.transpose(1, 2))
    assert seen["gemm1_limit"] is None
    assert getattr(blk.experts, "_sglang_fused_weight_cache", None) in (None, {})

    # strided mode ignores the legacy cache env (explicit mode wins)
    monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_CACHE_WEIGHTS", "1")
    seen.clear()
    blk.experts.sglang_fused_experts_forward(x, w, ids)
    assert seen["w13"].data_ptr() == blk.experts.gate_up_proj.data_ptr()
    assert getattr(blk.experts, "_sglang_fused_weight_cache", None) in (None, {})


def test_strided_vendored_impl_layout_guard():
    """The vendored strided impl accepts serving-contiguous and GKN transpose-view
    layouts and nothing else (importable without sglang)."""
    from xorl.ops.moe.sglang_fused_moe_strided import serving_layout_or_gkn_view  # noqa: PLC0415

    gkn = torch.randn(4, 8, 6)  # [E, K, N] contiguous
    assert serving_layout_or_gkn_view(gkn.transpose(1, 2))  # GKN view [E, N, K]
    assert serving_layout_or_gkn_view(gkn.transpose(1, 2).contiguous())  # serving layout
    assert not serving_layout_or_gkn_view(gkn[:, ::2, :].transpose(1, 2))  # sliced: neither


def test_strided_impl_delegates_with_split_gate_up_layout(monkeypatch):
    import xorl.ops.moe.sglang_fused_moe_strided as strided_mod  # noqa: PLC0415

    seen = {}

    def fake_fused_experts(*args, **kwargs):
        seen.update(kwargs)
        return args[0]

    monkeypatch.setattr(strided_mod, "_load_fused_experts_impl", lambda: fake_fused_experts)
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    w1 = torch.zeros(2, 6, 4, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 4, 3, dtype=torch.bfloat16)
    result = strided_mod.fused_experts_impl_strided(
        x,
        w1,
        w2,
        torch.zeros(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        gate_up_interleaved=False,
    )

    assert result is x
    assert seen["gate_up_interleaved"] is False

    with pytest.raises(ValueError, match="gate_up_interleaved=False"):
        strided_mod.fused_experts_impl_strided(
            x,
            w1,
            w2,
            torch.zeros(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
            gate_up_interleaved=True,
        )


def _install_fake_sglang_runtime(monkeypatch, *, existing, deterministic=True, fused_sum_all_reduce=False):
    created = []
    published = []

    class FakeServerArgs:
        def __init__(self, **kwargs):
            created.append(kwargs)

    def get_server_args():
        if not existing:
            raise ValueError("Global server args is not set yet!")
        return object()

    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic),
        moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=fused_sum_all_reduce),
    )
    runtime_context = types.ModuleType("sglang.srt.runtime_context")
    runtime_context.get_exec = lambda: exec_config
    runtime_context.get_server_args = get_server_args
    runtime_context.publish = lambda server_args, *, role: published.append((server_args, role))
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = FakeServerArgs
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    sglang.srt = srt
    srt.runtime_context = runtime_context
    srt.server_args = server_args
    for name, module in (
        ("sglang", sglang),
        ("sglang.srt", srt),
        ("sglang.srt.runtime_context", runtime_context),
        ("sglang.srt.server_args", server_args),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return created, published


def test_ensure_sglang_runtime_publishes_xorl_deterministic_context(monkeypatch):
    from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415

    created, published = _install_fake_sglang_runtime(monkeypatch, existing=False)
    MoEExperts._ensure_sglang_server_args()

    assert created == [
        {
            "model_path": "dummy",
            "enable_deterministic_inference": True,
            "enable_fused_moe_sum_all_reduce": False,
            "rl_on_policy_target": "xorl-batch-invariant",
        }
    ]
    assert len(published) == 1
    assert published[0][1] == "scheduler"


def test_ensure_sglang_runtime_preserves_compatible_context(monkeypatch):
    from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415

    created, published = _install_fake_sglang_runtime(monkeypatch, existing=True)
    MoEExperts._ensure_sglang_server_args()

    assert created == []
    assert published == []


@pytest.mark.parametrize(
    ("deterministic", "fused_sum_all_reduce"),
    [(False, False), (True, True)],
)
def test_ensure_sglang_runtime_rejects_incompatible_context(monkeypatch, deterministic, fused_sum_all_reduce):
    from xorl.models.layers.moe.experts import MoEExperts  # noqa: PLC0415

    _install_fake_sglang_runtime(
        monkeypatch,
        existing=True,
        deterministic=deterministic,
        fused_sum_all_reduce=fused_sum_all_reduce,
    )
    with pytest.raises(RuntimeError, match="SGLang MoE parity requires"):
        MoEExperts._ensure_sglang_server_args()


def test_sglang_runtime_api_does_not_regress_to_legacy_globals():
    source = inspect.getsource(experts_mod.MoEExperts._ensure_sglang_server_args)
    assert "get_global_server_args" not in source
    assert "set_global_server_args_for_scheduler" not in source


@pytest.mark.gpu
def test_strided_mode_bit_identical_to_transient(monkeypatch):
    """Forward and all gradients under WEIGHT_MODE=strided must be bit-identical
    to the transient-transpose mode (same kernels, view-strided addressing)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel")
    import xorl.ops.group_gemm.kernel.moe as moe_kernels  # noqa: PLC0415
    import xorl.ops.moe.triton as xorl_triton  # noqa: PLC0415
    from xorl.models.layers.moe.experts import MoEExperts, _SglangFusedExpertsTrainFunction  # noqa: PLC0415

    device = torch.device("cuda")
    torch.manual_seed(0)
    E, K, H2, I2, M2 = 8, 2, 64, 48, 16
    gu = (torch.randn(E, H2, 2 * I2, device=device) * 0.3).to(torch.bfloat16)
    dn = (torch.randn(E, I2, H2, device=device) * 0.3).to(torch.bfloat16)
    x = (torch.randn(M2, H2, device=device) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=device)[:K] for _ in range(M2)])
    wts = torch.rand(M2, K, device=device).to(torch.bfloat16)

    orig = moe_kernels.moe_index_compute
    pinned = {}

    def pinned_index_compute(expert_index, cumsum_t):
        key = (expert_index.data_ptr(), int(expert_index.numel()))
        if key not in pinned:
            pinned[key] = orig(expert_index, cumsum_t)
        return pinned[key].clone()

    monkeypatch.setattr(moe_kernels, "moe_index_compute", pinned_index_compute)
    monkeypatch.setattr(xorl_triton, "moe_index_compute", pinned_index_compute)
    torch.manual_seed(7)
    grad_out = None
    results = {}
    for mode in ("transient", "strided"):
        monkeypatch.setenv("XORL_MOE_SGLANG_FUSED_EXPERTS_WEIGHT_MODE", mode)
        impl = MoEExperts._load_sglang_fused_experts_impl()
        leaves = [t.clone().requires_grad_(True) for t in (x, wts, gu, dn)]
        out = _SglangFusedExpertsTrainFunction.apply(
            leaves[0], leaves[1], ids, leaves[2], leaves[3], impl, "silu", "silu", 0.0, E
        )
        if grad_out is None:
            grad_out = (torch.randn_like(out.float()) * 0.1).to(out.dtype)
        out.backward(grad_out)
        results[mode] = (out.detach(), *[t.grad for t in leaves])

    for name, a, b in zip(
        ("forward", "dX", "d_topk_weights", "dW13_gkn", "dW2_gkn"), results["strided"], results["transient"]
    ):
        assert torch.equal(a, b), f"{name} differs between strided and transient weight modes"


def test_kernel_receives_sglang_layout_and_fp32_weights(routed_inputs, monkeypatch):
    """Fake the kernel to check the exact tensors the SGLang path hands over."""
    blk = _block("eager")
    x, w, ids = routed_inputs
    seen = {}

    def fake_impl(hidden, w13, w2, topk_weights, topk_ids, **kwargs):
        seen["w13"] = w13
        seen["w2"] = w2
        seen["topk_weights"] = topk_weights
        seen["topk_ids"] = topk_ids
        seen["kwargs"] = kwargs
        return hidden.clone()

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(type(blk.experts), "_load_sglang_fused_experts_impl", staticmethod(lambda: fake_impl))
    monkeypatch.setattr(type(blk.experts), "_sglang_fused_experts_config_logged", True, raising=False)
    out = blk.experts.sglang_fused_experts_forward(x, w, ids)

    assert out.shape == x.shape
    assert seen["w13"].shape == (NUM_EXPERTS, 2 * INTER, HID)
    assert seen["w2"].shape == (NUM_EXPERTS, HID, INTER)
    # gate-first: w13 rows 0..I-1 must be the gate projection (xorl gate_up_proj cols 0..I-1)
    assert torch.equal(seen["w13"][:, :INTER, :], blk.experts.gate_up_proj[:, :, :INTER].transpose(1, 2))
    assert torch.equal(seen["w2"], blk.experts.down_proj.transpose(1, 2))
    assert seen["topk_weights"].dtype == torch.float32
    assert torch.equal(seen["topk_weights"], w.to(torch.float32))
    assert seen["kwargs"]["inplace"] is False
    assert seen["kwargs"]["filter_expert"] is False
    assert seen["kwargs"]["apply_router_weight_on_input"] is False


@pytest.mark.gpu
def test_auto_ep1_forward_dispatches_parity_path(monkeypatch, auto_state):
    """Unset env at ep=1 on CUDA must dispatch MoEBlock.forward to the parity
    path; explicit 0 is the escape hatch back to the stock tree."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.delenv(FLAG, raising=False)
    _stack_available(monkeypatch, True)
    blk = _block("triton").to("cuda")
    torch.manual_seed(1)
    x = torch.randn(TOKENS, HID, dtype=torch.bfloat16, device="cuda")
    called = {}

    def fake(hidden_states, routing_weights, selected_experts):
        called["hit"] = True
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(blk.experts, "sglang_fused_experts_forward", fake)
    blk.forward(x.view(1, TOKENS, HID))
    assert called.get("hit"), "unset env at ep=1 on CUDA must auto-enable the parity path"

    called.clear()
    monkeypatch.setenv(FLAG, "0")
    blk.forward(x.view(1, TOKENS, HID))
    assert not called.get("hit"), "explicit 0 must keep the stock path"


@pytest.mark.gpu
def test_auto_parity_forward_deterministic_and_matches_explicit(monkeypatch, auto_state):
    """Real-kernel ep=1 sanity: the auto-enabled forward is bit-identical across
    two runs (determinism) and to the explicit FLAG=1 path."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("sglang")
    pytest.importorskip("sgl_kernel")
    monkeypatch.delenv(FLAG, raising=False)

    torch.manual_seed(0)
    blk = MoEBlock(
        hidden_size=64,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        intermediate_size=48,
        hidden_act="silu",
        moe_implementation="triton",
    )
    with torch.no_grad():
        blk.experts.gate_up_proj.normal_(std=0.5)
        blk.experts.down_proj.normal_(std=0.5)
    blk = blk.to(torch.bfloat16).to("cuda")
    blk.requires_grad_(False)
    torch.manual_seed(1)
    x = torch.randn(1, TOKENS, 64, dtype=torch.bfloat16, device="cuda")

    assert experts_mod.moe_sglang_fused_experts_enabled(1, x.device, blk.experts) is True

    out_auto, _ = blk.forward(x)
    out_auto_again, _ = blk.forward(x)
    assert torch.equal(out_auto, out_auto_again), "auto parity forward must be deterministic across runs"

    monkeypatch.setenv(FLAG, "1")
    out_explicit, _ = blk.forward(x)
    assert torch.equal(out_auto, out_explicit), "auto resolution must run the same tree as the explicit flag"

    monkeypatch.setenv(FLAG, "0")
    out_stock, _ = blk.forward(x)
    assert out_stock.shape == out_auto.shape
