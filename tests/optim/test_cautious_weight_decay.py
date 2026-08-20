"""Tests for Cautious Weight Decay (CWD).

Reference algorithm (Chen et al., arXiv:2510.12402)::

    x_{t+1} = x_t - eta * (u_t + lambda * I(u_t * x_t >= 0) * x_t)

where ``u_t`` is the optimizer's update direction (post-preconditioning /
post-Newton-Schulz). When ``cautious=False`` the optimizer must reduce to its
standard decoupled-decay form.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from xorl.optim import AnyPrecisionAdamW, Muon, SignSGD, build_optimizer
from xorl.optim.anyprecision_adamw import _OffloadedDTensorState
from xorl.optim.cautious import apply_cautious_decay_


pytestmark = [pytest.mark.cpu]


# --------------------------- helper primitives ----------------------------


@torch.no_grad()
def _assert_cautious_decay_primitive_and_signsgd_policy():
    p = torch.tensor([1.0, -2.0])
    proxy = torch.tensor([5.0, -5.0])
    apply_cautious_decay_(p, proxy, lr=0.1, weight_decay=0.0, cautious=True)
    assert torch.equal(p, torch.tensor([1.0, -2.0]))

    p = torch.tensor([1.0, -2.0, 3.0])
    proxy = torch.tensor([1.0, -1.0, -1.0])  # mixed alignment
    apply_cautious_decay_(p, proxy, lr=0.1, weight_decay=0.5, cautious=False)
    expected = torch.tensor([1.0, -2.0, 3.0]) * (1 - 0.1 * 0.5)
    assert torch.allclose(p, expected)

    # update * param sign:
    #   ( 1.0,  2.0) -> +  -> decay applies
    #   (-1.0,  3.0) -> -  -> decay skipped
    #   ( 0.0, -4.0) -> 0  -> decay applies (>= 0)
    #   ( 1.0, -5.0) -> -  -> decay skipped
    p = torch.tensor([2.0, 3.0, -4.0, -5.0])
    proxy = torch.tensor([1.0, -1.0, 0.0, 1.0])
    apply_cautious_decay_(p, proxy, lr=0.1, weight_decay=0.5, cautious=True)
    factor = 1 - 0.1 * 0.5
    expected = torch.tensor([2.0 * factor, 3.0, -4.0 * factor, -5.0])
    assert torch.allclose(p, expected)

    _assert_signsgd_cautious_masks_decay_against_grad_sign()


# ------------------------------ SignSGD ----------------------------------


def _assert_signsgd_cautious_masks_decay_against_grad_sign():
    # grad * param signs:
    #   (+, +) -> aligned, decay applies
    #   (-, +) -> misaligned, decay masked
    p = nn.Parameter(torch.tensor([2.0, 3.0]))
    optimizer = SignSGD([p], lr=0.1, weight_decay=0.5, cautious=True)
    p.grad = torch.tensor([1.0, -1.0])
    optimizer.step()

    decay = 1 - 0.1 * 0.5
    expected = torch.tensor([2.0 * decay - 0.1 * 1.0, 3.0 - 0.1 * (-1.0)])
    assert torch.allclose(p, expected)

    _assert_signsgd_dense_update_and_sparse_rejection()


def _assert_signsgd_dense_update_and_sparse_rejection():
    signed = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    decayed = nn.Parameter(torch.tensor([2.0, -2.0]))
    gradless = nn.Parameter(torch.tensor([1.5, -1.5]))
    optimizer = SignSGD(
        [
            {"params": [signed], "weight_decay": 0.0},
            {"params": [decayed, gradless], "weight_decay": 0.5},
        ],
        lr=0.1,
    )

    signed.grad = torch.tensor([2.0, -0.5, 0.0])
    decayed.grad = torch.tensor([3.0, -4.0])
    optimizer.step()

    torch.testing.assert_close(signed, torch.tensor([0.9, -1.9, 3.0]))
    torch.testing.assert_close(decayed, torch.tensor([1.8, -1.8]))
    torch.testing.assert_close(gradless, torch.tensor([1.5, -1.5]))

    sparse = nn.Parameter(torch.ones(4))
    optimizer = SignSGD([sparse], lr=0.1)
    sparse.grad = torch.sparse_coo_tensor(indices=[[0, 2]], values=torch.tensor([1.0, -1.0]), size=(4,))

    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        optimizer.step()


# --------------------------- AnyPrecisionAdamW ----------------------------


def _adamw_first_step_cautious(p_init, grad, lr, wd, cautious, beta1=0.9, beta2=0.95, eps=1e-8):
    p = nn.Parameter(p_init.clone())
    optimizer = AnyPrecisionAdamW(
        [p],
        lr=lr,
        weight_decay=wd,
        betas=(beta1, beta2),
        eps=eps,
        momentum_dtype=torch.float32,
        variance_dtype=torch.float32,
        compensation_buffer_dtype=torch.float32,
        cautious=cautious,
    )
    p.grad = grad.clone()
    optimizer.step()
    return p.detach().clone()


def _assert_anyprecision_adamw_cautious_decay_and_state_policy(tmp_path):
    # Prior to CWD the optimizer applied decay first. Verify cautious=False
    # produces the same final parameter (which is what the existing tests
    # implicitly relied on).
    p_init = torch.tensor([1.0, -2.0, 3.0])
    grad = torch.tensor([0.5, 0.5, -1.0])
    lr, wd = 0.1, 0.1
    beta1, beta2 = 0.9, 0.95
    eps = 1e-8

    out = _adamw_first_step_cautious(p_init, grad, lr, wd, cautious=False, beta1=beta1, beta2=beta2, eps=eps)

    # Reference: order of operations doesn't matter for non-cautious decoupled
    # decay because grad doesn't depend on p. Replicate the math.
    exp_avg = (1 - beta1) * grad
    exp_avg_sq = (1 - beta2) * grad * grad
    bc1 = 1 - beta1
    bc2 = 1 - beta2
    denom = exp_avg_sq.sqrt() / (bc2**0.5) + eps
    expected = p_init * (1 - lr * wd) - (lr / bc1) * (exp_avg / denom)
    assert torch.allclose(out, expected, atol=1e-6)

    _assert_anyprecision_adamw_cautious_skips_misaligned_decay()
    _assert_anyprecision_adamw_state_strategy_and_dtensor_offload_policy(tmp_path)


def _assert_anyprecision_adamw_state_strategy_and_dtensor_offload_policy(tmp_path):
    for use_kahan_summation in (False, True):
        p_ref = nn.Parameter(torch.linspace(-2.0, 2.0, 60).reshape(4, 3, 5))
        p_chunked = nn.Parameter(p_ref.detach().clone())
        kwargs = dict(
            lr=0.01,
            weight_decay=0.05,
            betas=(0.9, 0.95),
            eps=1e-8,
            momentum_dtype=torch.float32,
            variance_dtype=torch.float32,
            compensation_buffer_dtype=torch.float32,
            use_kahan_summation=use_kahan_summation,
        )
        opt_ref = AnyPrecisionAdamW([p_ref], **kwargs)
        opt_chunked = AnyPrecisionAdamW([p_chunked], denominator_chunk_size=7, **kwargs)

        for scale in (0.25, -0.5, 0.75):
            grad = torch.linspace(-1.5, 1.5, 60).reshape(4, 3, 5) * scale
            p_ref.grad = grad.clone()
            p_chunked.grad = grad.clone()
            opt_ref.step()
            opt_chunked.step()

        assert torch.allclose(p_chunked, p_ref, atol=1e-6, rtol=1e-6)
        ref_state = opt_ref.state[p_ref]
        chunked_state = opt_chunked.state[p_chunked]
        assert torch.allclose(chunked_state["exp_avg"], ref_state["exp_avg"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(chunked_state["exp_avg_sq"], ref_state["exp_avg_sq"], atol=1e-6, rtol=1e-6)
        if use_kahan_summation:
            assert torch.allclose(chunked_state["compensation"], ref_state["compensation"], atol=1e-6, rtol=1e-6)

    _assert_anyprecision_adamw_reuses_grad_for_momentum()
    _assert_anyprecision_adamw_dtensor_state_offload_round_trips_local_shards(tmp_path)


def _assert_anyprecision_adamw_reuses_grad_for_momentum():
    p_ref = nn.Parameter(torch.linspace(-2.0, 2.0, 12).reshape(3, 4))
    p_reuse = nn.Parameter(p_ref.detach().clone())
    kwargs = dict(
        lr=0.01,
        weight_decay=0.05,
        betas=(0.9, 0.95),
        eps=1e-8,
        momentum_dtype=torch.float32,
        variance_dtype=torch.float32,
        compensation_buffer_dtype=torch.float32,
    )
    opt_ref = AnyPrecisionAdamW([p_ref], **kwargs)
    opt_reuse = AnyPrecisionAdamW([p_reuse], reuse_grad_for_momentum=True, state_offload_device="cpu", **kwargs)

    for scale in (0.25, -0.5, 0.75):
        grad = torch.linspace(-1.5, 1.5, 12).reshape(3, 4) * scale
        p_ref.grad = grad.clone()
        p_reuse.grad = grad.clone()
        opt_ref.step()
        opt_reuse.step()
        assert p_reuse.grad is None

    assert torch.allclose(p_reuse, p_ref, atol=1e-6, rtol=1e-6)
    ref_state = opt_ref.state[p_ref]
    reuse_state = opt_reuse.state[p_reuse]
    assert torch.allclose(reuse_state["exp_avg"], ref_state["exp_avg"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(reuse_state["exp_avg_sq"], ref_state["exp_avg_sq"], atol=1e-6, rtol=1e-6)
    assert reuse_state["exp_avg"].device.type == "cpu"
    assert reuse_state["exp_avg_sq"].device.type == "cpu"


def _assert_anyprecision_adamw_dtensor_state_offload_round_trips_local_shards(tmp_path):
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method=f"file://{tmp_path / 'single_rank_pg'}",
            rank=0,
            world_size=1,
        )
        initialized_here = True
    elif dist.get_world_size() != 1:
        pytest.skip("single-rank DTensor offload smoke test requires a single-rank process group")

    try:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp",))
        local_param = torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).reshape(2, 3)
        param = nn.Parameter(DTensor.from_local(local_param.clone(), mesh, [Shard(0)], run_check=False))
        optimizer = AnyPrecisionAdamW(
            [param],
            lr=0.01,
            momentum_dtype=torch.float32,
            variance_dtype=torch.float32,
            reuse_grad_for_momentum=True,
            state_offload_device="cpu",
        )

        for scale in (0.1, 0.2):
            local_grad = torch.full_like(local_param, scale)
            param.grad = DTensor.from_local(local_grad, mesh, [Shard(0)], run_check=False)
            optimizer.step()
            state = optimizer.state[param]
            assert isinstance(state["exp_avg"], _OffloadedDTensorState)
            assert isinstance(state["exp_avg_sq"], _OffloadedDTensorState)
            assert state["exp_avg"].local_tensor.device.type == "cpu"
            assert state["exp_avg_sq"].local_tensor.device.type == "cpu"

        AnyPrecisionAdamW._move_state_to_device(optimizer.state[param], param.device)
        state = optimizer.state[param]
        assert isinstance(state["exp_avg"], DTensor)
        assert isinstance(state["exp_avg_sq"], DTensor)
        assert torch.allclose(state["exp_avg"].to_local(), torch.full_like(local_param, 0.029))
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _assert_anyprecision_adamw_cautious_skips_misaligned_decay():
    # exp_avg = (1-beta1) * grad has same sign as grad on the first step.
    # Construct a case where some coords have grad sign opposite to param sign.
    p_init = torch.tensor([2.0, -3.0, 4.0])
    grad = torch.tensor([-1.0, 1.0, 1.0])  # signs: -, +, +
    # exp_avg signs after first step: -, +, +
    # exp_avg * p signs:               -, -, +
    # Mask:                            0,  0,  1 (decay only on coord 2)
    lr, wd = 0.1, 0.5
    beta1, beta2 = 0.9, 0.95
    eps = 1e-8

    out = _adamw_first_step_cautious(p_init, grad, lr, wd, cautious=True, beta1=beta1, beta2=beta2, eps=eps)

    exp_avg = (1 - beta1) * grad
    exp_avg_sq = (1 - beta2) * grad * grad
    bc1 = 1 - beta1
    bc2 = 1 - beta2
    denom = exp_avg_sq.sqrt() / (bc2**0.5) + eps
    mask = (exp_avg * p_init >= 0).to(p_init.dtype)
    expected = p_init * (1 - lr * wd * mask) - (lr / bc1) * (exp_avg / denom)
    assert torch.allclose(out, expected, atol=1e-6)


# -------------------------------- Muon ------------------------------------


def _assert_muon_cautious_decay_policy():
    """``cautious=False`` must reproduce the pre-CWD Muon update exactly:
    ``param *= 1 - lr*wd``, then ``param -= adjusted_lr * NS(grad)``.

    Pinned to ``standard_newton_schulz`` so the reference matches PyTorch's
    upstream NS implementation; the alternate ``gram_newton_schulz`` backend
    produces slightly different per-element values that would mask actual
    decay-equivalence regressions.
    """
    torch.manual_seed(0)
    p_init = torch.randn(4, 4)
    grad = torch.randn(4, 4)
    lr, wd = 0.02, 0.1

    w = nn.Parameter(p_init.clone())
    opt = Muon(
        [{"params": [w], "use_muon": True}],
        lr=lr,
        momentum=0.0,
        nesterov=False,
        weight_decay=wd,
        cautious=False,
        ns_algorithm="standard_newton_schulz",
    )
    w.grad = grad.clone()
    opt.step()

    update = _zeropower_via_newtonschulz(grad, (3.4445, -4.775, 2.0315), 5, 1e-7)
    adjusted_lr = _adjust_lr(lr, None, grad.shape)
    expected = p_init * (1 - lr * wd) - adjusted_lr * update
    assert torch.allclose(w, expected, atol=1e-4)

    _assert_muon_cautious_masks_decay_using_post_ns_update()
    _assert_muon_cautious_adamw_fallback_masks_decay()


def _assert_muon_cautious_masks_decay_using_post_ns_update():
    """Muon's update direction is the orthogonalized matrix; the cautious
    mask must be ``I(NS(grad) * param >= 0)``, not ``I(grad * param >= 0)``.
    The two differ in general, so we check against an explicit reference.
    """
    torch.manual_seed(0)
    p_init = torch.randn(4, 4)
    grad = torch.randn(4, 4)
    lr, wd = 0.02, 0.5

    # Run cautious Muon with momentum=0 so update = NS(grad).
    w = nn.Parameter(p_init.clone())
    opt = Muon(
        [{"params": [w], "use_muon": True}],
        lr=lr,
        momentum=0.0,
        nesterov=False,
        weight_decay=wd,
        cautious=True,
        ns_algorithm="standard_newton_schulz",
    )
    w.grad = grad.clone()
    opt.step()
    actual = w.detach().clone()

    # Reference: replicate the Muon math.
    update = _zeropower_via_newtonschulz(grad, (3.4445, -4.775, 2.0315), 5, 1e-7)
    adjusted_lr = _adjust_lr(lr, None, grad.shape)
    mask = (update * p_init >= 0).to(p_init.dtype)
    expected = p_init * (1 - lr * wd * mask) - adjusted_lr * update
    assert torch.allclose(actual, expected, atol=1e-4)


def _assert_muon_cautious_adamw_fallback_masks_decay():
    """The non-Muon param group (use_muon=False) takes the AdamW path; that
    path must also honor cautious."""
    p_init = torch.tensor([2.0, -3.0])
    grad = torch.tensor([-1.0, 1.0])
    w = nn.Parameter(p_init.clone())
    opt = Muon(
        [{"params": [w], "use_muon": False, "lr": 0.1}],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.5,
        cautious=True,
    )
    w.grad = grad.clone()
    opt.step()

    beta1, beta2 = 0.9, 0.95
    eps = 1e-8
    exp_avg = (1 - beta1) * grad
    exp_avg_sq = (1 - beta2) * grad * grad
    mask = (exp_avg * p_init >= 0).to(p_init.dtype)
    p_after_decay = p_init * (1 - 0.1 * 0.5 * mask)
    bc1 = 1 - beta1
    bc2 = 1 - beta2
    denom = exp_avg_sq.sqrt() / (bc2**0.5) + eps
    expected = p_after_decay - (0.1 / bc1) * (exp_avg / denom)
    assert torch.allclose(w, expected, atol=1e-6)


# -------------------------- build_optimizer wiring -------------------------


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)


def test_build_optimizer_cautious_routing_and_kwargs_policy(tmp_path):
    _assert_cautious_decay_primitive_and_signsgd_policy()
    _assert_anyprecision_adamw_cautious_decay_and_state_policy(tmp_path)
    _assert_muon_cautious_decay_policy()
    for optimizer_type, expected_type in (
        ("signsgd", SignSGD),
        ("anyprecision_adamw", AnyPrecisionAdamW),
        ("adamw", AnyPrecisionAdamW),
    ):
        opt = build_optimizer(
            _Tiny(),
            lr=0.1,
            weight_decay=0.01,
            optimizer_type=optimizer_type,
            cautious_weight_decay=True,
        )
        assert isinstance(opt, expected_type)
        assert all(group["cautious"] is True for group in opt.param_groups)
        if optimizer_type == "adamw":
            assert all(group["momentum_dtype"] == torch.float32 for group in opt.param_groups)

    _assert_build_optimizer_adamw_default_uses_torch_path()
    _assert_build_optimizer_rejects_cautious_with_sgd()
    _assert_build_optimizer_rejects_torch_only_kwargs_on_cautious_adamw()
    _assert_build_optimizer_allows_anyprecision_kwargs()
    _assert_build_optimizer_passes_anyprecision_state_strategy_kwargs()


def _assert_build_optimizer_adamw_default_uses_torch_path():
    model = _Tiny()
    opt = build_optimizer(
        model,
        lr=0.1,
        weight_decay=0.01,
        optimizer_type="adamw",
        cautious_weight_decay=False,
    )
    # cautious=False keeps the fused-capable torch path
    assert isinstance(opt, torch.optim.AdamW)


def _assert_build_optimizer_rejects_cautious_with_sgd():
    model = _Tiny()
    with pytest.raises(ValueError, match="cautious_weight_decay is not supported"):
        build_optimizer(
            model,
            lr=0.1,
            weight_decay=0.01,
            optimizer_type="sgd",
            cautious_weight_decay=True,
        )


def _assert_build_optimizer_rejects_torch_only_kwargs_on_cautious_adamw():
    # When adamw+cautious routes to AnyPrecisionAdamW, torch.optim.AdamW-only
    # kwargs (foreach, fused, amsgrad, ...) would otherwise produce a
    # confusing TypeError from a class the user did not request.
    model = _Tiny()
    with pytest.raises(ValueError, match="routes to AnyPrecisionAdamW"):
        build_optimizer(
            model,
            lr=0.1,
            weight_decay=0.01,
            optimizer_type="adamw",
            cautious_weight_decay=True,
            optimizer_kwargs={"foreach": False},
        )


def _assert_build_optimizer_allows_anyprecision_kwargs():
    # use_kahan_summation is an AnyPrecisionAdamW-native kwarg and must pass
    # through the cautious-routing filter.
    model = _Tiny()
    opt = build_optimizer(
        model,
        lr=0.1,
        weight_decay=0.01,
        optimizer_type="adamw",
        cautious_weight_decay=True,
        optimizer_kwargs={"use_kahan_summation": True},
    )
    assert isinstance(opt, AnyPrecisionAdamW)
    assert all(g["use_kahan_summation"] is True for g in opt.param_groups)


def _assert_build_optimizer_passes_anyprecision_state_strategy_kwargs():
    model = _Tiny()
    opt = build_optimizer(
        model,
        lr=0.1,
        weight_decay=0.01,
        optimizer_type="anyprecision_adamw",
        optimizer_kwargs={
            "denominator_chunk_size": 3,
            "reuse_grad_for_momentum": True,
            "state_offload_device": "cpu",
        },
    )
    assert isinstance(opt, AnyPrecisionAdamW)
    assert all(g["denominator_chunk_size"] == 3 for g in opt.param_groups)
    assert all(g["reuse_grad_for_momentum"] is True for g in opt.param_groups)
    assert all(g["state_offload_device"] == "cpu" for g in opt.param_groups)
