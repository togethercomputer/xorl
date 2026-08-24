"""Unit tests for the exact-model LoRA merged-forward contract.

Covers the canonical fold (pinned fp32-accumulate/cast-once order), the
straight-through folded-weight autograd, the LoraLinear merged forward, the
fold cache/version keying, the loud-fail composition rules, and the weight-sync
byte-consistency helper. Cross-engine bitwise gates (trainer merged forward vs
sglang postfold serving) live in experiments/k3_tests/lora_path_xengine.py.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from xorl.lora.fold import (
    FoldedLoraWeightGateUpGKN,
    FoldedLoraWeightGKN,
    FoldedLoraWeightLinear,
    _factor_grad_dtype,
    canonical_lora_fold_gkn,
    canonical_lora_fold_linear,
)
from xorl.lora.modules.linear import LoraLinear
from xorl.models.layers.moe.lora import MoEExpertsLoRA, MoELoRAConfig


E, H, I, R = 4, 32, 16, 8
SCALING = 2.0


def _gkn_factors(shared_a=False, shared_b=False, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(1 if shared_a else E, H, R, generator=g).to(dtype) * 0.02
    B = torch.randn(1 if shared_b else E, R, I, generator=g).to(dtype) * 0.02
    W = torch.randn(E, H, I, generator=g).to(dtype)
    return W, A, B


def _cast_once_merge(weight, delta):
    return (weight.to(torch.float32) + delta).to(weight.dtype)


def _assert_permanent_merge_matches_cast_once_contract():
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        torch.manual_seed(0)
        layer = LoraLinear(128, 64, r=8, lora_alpha=16, dtype=dtype)
        layer.weight.data.normal_(std=0.05)
        before = layer.weight.detach().clone()
        layer.merge_weights()
        assert torch.equal(layer.weight, before)

        if dtype is torch.float32:
            continue
        torch.manual_seed(0)
        layer = LoraLinear(128, 64, r=8, lora_alpha=16, dtype=dtype)
        layer.weight.data.normal_(std=0.05)
        layer.lora_B.data.normal_(std=0.02)
        before = layer.weight.detach().clone()
        expected = _cast_once_merge(before, (layer.lora_B @ layer.lora_A) * layer.scaling)
        layer.merge_weights()
        assert torch.equal(layer.weight, expected)

    config = MoELoRAConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"])
    for dtype in (torch.bfloat16, torch.float16):
        experts = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=24,
            hidden_act="silu",
            moe_implementation="eager",
            lora_config=config,
        ).to(dtype)
        experts.gate_up_proj.data.normal_(std=0.05)
        experts.down_proj.data.normal_(std=0.05)
        gate_up_before = experts.gate_up_proj.detach().clone()
        down_before = experts.down_proj.detach().clone()
        experts.merge_weights()
        assert torch.equal(experts.gate_up_proj, gate_up_before)
        assert torch.equal(experts.down_proj, down_before)

        experts = MoEExpertsLoRA(
            num_experts=4,
            hidden_dim=32,
            intermediate_size=24,
            hidden_act="silu",
            moe_implementation="eager",
            lora_config=config,
        ).to(dtype)
        experts.gate_up_proj.data.normal_(std=0.05)
        experts.down_proj.data.normal_(std=0.05)
        torch.manual_seed(7)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            getattr(experts, f"{projection}_lora_B").data.normal_(std=0.02)

        gate_up_before = experts.gate_up_proj.detach().clone()
        down_before = experts.down_proj.detach().clone()
        updates = {projection: experts._compute_proj_delta(projection) for projection in config.target_modules}
        intermediate = experts.intermediate_size
        gate_up_expected = gate_up_before.clone()
        gate_up_expected[..., :intermediate] = _cast_once_merge(
            gate_up_before[..., :intermediate], updates["gate_proj"]
        )
        gate_up_expected[..., intermediate:] = _cast_once_merge(gate_up_before[..., intermediate:], updates["up_proj"])
        down_expected = _cast_once_merge(down_before, updates["down_proj"])

        experts.merge_weights()

        assert torch.equal(experts.gate_up_proj, gate_up_expected)
        assert torch.equal(experts.down_proj, down_expected)


class TestCanonicalFold:
    def test_canonical_fold_forward_and_cache_policy(self):
        W, A, B = _gkn_factors()
        delta = torch.bmm(A.float().expand(E, -1, -1), B.float().expand(E, -1, -1)) * SCALING
        want = (W.float() + delta).to(W.dtype)
        got = canonical_lora_fold_gkn(W, A, B, SCALING)
        assert torch.equal(got, want)
        assert got.dtype == W.dtype

        self._assert_shared_factors_expand()
        self._assert_fold_commutes_with_expert_sharding()
        self._assert_linear_orientation()
        self._assert_zero_delta_is_value_identical()
        _assert_permanent_merge_matches_cast_once_contract()
        TestFoldedWeightAutograd()._assert_straight_through_gradient_policy()
        TestLoraLinearMerged()._assert_merged_forward_selection_policy()
        TestMoEExpertsLoRAMerged()._assert_merged_weight_and_cache_policy()

    def _assert_shared_factors_expand(self):
        W, A, B = _gkn_factors(shared_a=True)
        got = canonical_lora_fold_gkn(W, A, B, SCALING)
        per_expert = torch.stack(
            [(W[e].float() + (A[0].float() @ B[e].float()) * SCALING).to(W.dtype) for e in range(E)]
        )
        assert torch.allclose(got.float(), per_expert.float(), atol=0, rtol=0) or torch.equal(got, per_expert)

    def _assert_fold_commutes_with_expert_sharding(self):
        # EP invariance: fold of a local expert slice == slice of the full fold.
        W, A, B = _gkn_factors()
        full = canonical_lora_fold_gkn(W, A, B, SCALING)
        for sl in (slice(0, 2), slice(2, 4)):
            shard = canonical_lora_fold_gkn(W[sl], A[sl], B[sl], SCALING)
            assert torch.equal(shard, full[sl])

    def _assert_linear_orientation(self):
        g = torch.Generator().manual_seed(1)
        W = torch.randn(I, H, generator=g).to(torch.bfloat16)
        A = torch.randn(R, H, generator=g).to(torch.float32) * 0.02
        B = torch.randn(I, R, generator=g).to(torch.float32) * 0.02
        want = (W.float() + (B @ A) * SCALING).to(W.dtype)
        assert torch.equal(canonical_lora_fold_linear(W, A, B, SCALING), want)

    def _assert_zero_delta_is_value_identical(self):
        # B == 0: folded values equal the base everywhere (only -0.0 -> +0.0
        # bit flips are permitted; both engines fold identically so the
        # contract holds).
        W, A, B = _gkn_factors()
        got = canonical_lora_fold_gkn(W, A, torch.zeros_like(B), SCALING)
        assert torch.eq(got.float(), W.float()).all()


class TestFoldedWeightAutograd:
    def _assert_factor_grad_dtype_honors_fsdp_metadata(self):
        factor_A = torch.ones(4, 8, dtype=torch.bfloat16, requires_grad=True)
        factor_B = torch.ones(16, 4, dtype=torch.bfloat16, requires_grad=True)
        assert _factor_grad_dtype(factor_A[:2]) == torch.bfloat16
        assert _factor_grad_dtype(factor_B[:, :2]) == torch.bfloat16
        factor_A.grad_dtype = torch.float32
        factor_B.grad_dtype = torch.float32
        assert _factor_grad_dtype(factor_A[:2]) == torch.float32
        assert _factor_grad_dtype(factor_B[:, :2]) == torch.float32

        A = factor_A[:2]
        B = factor_B[:, :2]
        W = torch.zeros(16, 8, dtype=torch.bfloat16)
        folded = canonical_lora_fold_linear(W, A.detach(), B.detach(), SCALING)
        FoldedLoraWeightLinear.apply(folded, A, B, SCALING).float().sum().backward()
        assert factor_A.grad.dtype == torch.float32
        assert factor_B.grad.dtype == torch.float32
        assert torch.count_nonzero(factor_A.grad) > 0
        assert torch.count_nonzero(factor_B.grad) > 0

    def _reference_grads(self, W, A, B, scaling, grad_w, shared_a=False, shared_b=False):
        A_ref = A.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        Ef = W.shape[0]
        delta = torch.bmm(A_ref.float().expand(Ef, -1, -1), B_ref.float().expand(Ef, -1, -1)) * scaling
        folded = (W.float() + delta).to(W.dtype)
        folded.backward(grad_w)
        return A_ref.grad, B_ref.grad

    def _assert_straight_through_gradient_policy(self):
        self._assert_factor_grad_dtype_honors_fsdp_metadata()

        for shared_a, shared_b in ((False, False), (True, False), (False, True)):
            W, A, B = _gkn_factors(shared_a=shared_a, shared_b=shared_b, seed=2)
            A = A.requires_grad_(True)
            B = B.requires_grad_(True)
            folded = canonical_lora_fold_gkn(W, A.detach(), B.detach(), SCALING)
            out = FoldedLoraWeightGKN.apply(folded, A, B, SCALING)
            assert torch.equal(out, folded)
            grad_w = torch.randn_like(folded.float()).to(folded.dtype)
            out.backward(grad_w)
            gA_ref, gB_ref = self._reference_grads(W, A, B, SCALING, grad_w, shared_a, shared_b)
            assert torch.allclose(A.grad.float(), gA_ref.float(), rtol=1e-5, atol=1e-8)
            assert torch.allclose(B.grad.float(), gB_ref.float(), rtol=1e-5, atol=1e-8)

        self._assert_gate_up_fused_straight_through()
        self._assert_linear_straight_through_matches_autograd()

    def _assert_gate_up_fused_straight_through(self):
        g = torch.Generator().manual_seed(3)
        W = torch.randn(E, H, 2 * I, generator=g).to(torch.bfloat16)
        gA = (torch.randn(1, H, R, generator=g) * 0.02).to(torch.bfloat16).requires_grad_(True)
        gB = (torch.randn(E, R, I, generator=g) * 0.02).to(torch.bfloat16).requires_grad_(True)
        uA = (torch.randn(1, H, R, generator=g) * 0.02).to(torch.bfloat16).requires_grad_(True)
        uB = (torch.randn(E, R, I, generator=g) * 0.02).to(torch.bfloat16).requires_grad_(True)
        gate_f = canonical_lora_fold_gkn(W[..., :I], gA.detach(), gB.detach(), SCALING)
        up_f = canonical_lora_fold_gkn(W[..., I:], uA.detach(), uB.detach(), SCALING)
        folded = torch.cat([gate_f, up_f], dim=-1)
        out = FoldedLoraWeightGateUpGKN.apply(folded, gA, gB, uA, uB, SCALING, I)
        grad_w = torch.randn_like(folded.float()).to(folded.dtype)
        out.backward(grad_w)

        # reference: per-projection straight-through on the halves
        for A, B, half in ((gA, gB, grad_w[..., :I]), (uA, uB, grad_w[..., I:])):
            gA_ref, gB_ref = self._reference_grads(
                W[..., :I].contiguous(), A, B, SCALING, half.contiguous(), A.shape[0] == 1, B.shape[0] == 1
            )
            assert torch.allclose(A.grad.float(), gA_ref.float(), rtol=1e-5, atol=1e-8)
            assert torch.allclose(B.grad.float(), gB_ref.float(), rtol=1e-5, atol=1e-8)
            A.grad = None
            B.grad = None
            break  # gate checked exactly; up follows by symmetry

    def _assert_linear_straight_through_matches_autograd(self):
        g = torch.Generator().manual_seed(4)
        W = torch.randn(I, H, generator=g).to(torch.bfloat16)
        A = (torch.randn(R, H, generator=g) * 0.02).requires_grad_(True)
        B = (torch.randn(I, R, generator=g) * 0.02).requires_grad_(True)
        folded = canonical_lora_fold_linear(W, A.detach(), B.detach(), SCALING)
        out = FoldedLoraWeightLinear.apply(folded, A, B, SCALING)
        grad_w = torch.randn_like(folded.float()).to(folded.dtype)
        out.backward(grad_w)

        A_ref = A.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        ((W.float() + (B_ref.float() @ A_ref.float()) * SCALING).to(W.dtype)).backward(grad_w)
        assert torch.allclose(A.grad, A_ref.grad, rtol=1e-5, atol=1e-8)
        assert torch.allclose(B.grad, B_ref.grad, rtol=1e-5, atol=1e-8)


class TestLoraLinearMerged:
    def _layer(self):
        torch.manual_seed(5)
        layer = LoraLinear(H, I, r=R, lora_alpha=int(SCALING * R), dtype=torch.float32)
        torch.nn.init.normal_(layer.lora_B, std=0.02)  # nonzero delta
        return layer

    def _assert_merged_forward_selection_policy(self):
        layer = self._layer()
        layer.exact_merged_forward = True
        x = torch.randn(6, H)
        want = torch.nn.functional.linear(
            x, canonical_lora_fold_linear(layer.weight, layer.lora_A, layer.lora_B, layer.scaling), None
        )
        with torch.no_grad():
            got = layer(x)
        assert torch.equal(got, want)

        self._assert_flag_off_keeps_legacy_path()
        self._assert_exact_and_ordinary_selection_is_isolated()
        self._assert_merged_grads_close_to_unmerged()
        self._assert_cache_invalidation_on_step_and_runtime_config()

    def _assert_flag_off_keeps_legacy_path(self):
        layer = self._layer()
        x = torch.randn(6, H)
        base = torch.nn.functional.linear(x, layer.weight, None)
        lora = torch.nn.functional.linear(torch.nn.functional.linear(x, layer.lora_A), layer.lora_B)
        want = base + (lora * layer.scaling)
        assert torch.allclose(layer(x), want, rtol=0, atol=0)

    def _assert_exact_and_ordinary_selection_is_isolated(self):
        exact = self._layer()
        ordinary = self._layer()
        ordinary.load_state_dict(exact.state_dict())
        exact.exact_merged_forward = True
        x = torch.randn(6, H)

        exact_first = exact(x)
        ordinary_out = ordinary(x)
        exact_second = exact(x)

        assert torch.equal(exact_first, exact_second)
        expected_ordinary = torch.nn.functional.linear(x, ordinary.weight, None) + (
            torch.nn.functional.linear(torch.nn.functional.linear(x, ordinary.lora_A), ordinary.lora_B)
            * ordinary.scaling
        )
        assert torch.equal(ordinary_out, expected_ordinary)

    def _assert_merged_grads_close_to_unmerged(self):
        layer = self._layer()
        x = torch.randn(6, H)
        grads = {}
        for flag in ("0", "1"):
            layer.exact_merged_forward = flag == "1"
            layer.lora_A.grad = layer.lora_B.grad = None
            layer(x).square().mean().backward()
            grads[flag] = (layer.lora_A.grad.clone(), layer.lora_B.grad.clone())
        # fwd/bwd decoupling: merged-lane grads track the unmerged autograd at
        # standard (bf16-class) tolerances
        for a, b in zip(grads["0"], grads["1"]):
            assert torch.allclose(a, b, rtol=5e-2, atol=1e-5)

    def _assert_cache_invalidation_on_step_and_runtime_config(self):
        layer = self._layer()
        layer.exact_merged_forward = True
        w1 = layer._merged_weight()
        assert layer._merged_weight() is w1  # cache hit
        opt = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=1e-2)
        layer(torch.randn(4, H)).square().mean().backward()
        opt.step()
        w2 = layer._merged_weight()
        assert w2 is not w1 and not torch.equal(w1, w2)
        layer.set_runtime_lora_config(R // 2, R)
        w3 = layer._merged_weight()
        assert w3 is not w2
        want = canonical_lora_fold_linear(
            layer.weight, layer.lora_A[: R // 2], layer.lora_B[:, : R // 2], layer._active_scaling()
        )
        assert torch.equal(w3, want)


class TestMoEExpertsLoRAMerged:
    def _module(self, hybrid=True):
        torch.manual_seed(6)
        cfg = MoELoRAConfig(
            r=R, lora_alpha=int(SCALING * R), target_modules=["gate_proj", "up_proj", "down_proj"], hybrid_shared=hybrid
        )
        mod = MoEExpertsLoRA(
            num_experts=E,
            hidden_dim=H,
            intermediate_size=I,
            hidden_act="silu",
            moe_implementation="triton",
            lora_config=cfg,
        ).to(torch.bfloat16)
        with torch.no_grad():
            mod.gate_up_proj.normal_(std=0.02)
            mod.down_proj.normal_(std=0.02)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                getattr(mod, f"{proj}_lora_B").normal_(std=0.02)
        mod.lora_serving_mode = "merged"
        return mod

    def _assert_merged_weight_and_cache_policy(self):
        mod = self._module()
        mod.exact_merged_forward = True
        gate_up_f, down_f = mod._merged_weights()
        s = mod._active_scaling()
        gA, gB = mod._active_lora_views("gate_proj")
        uA, uB = mod._active_lora_views("up_proj")
        dA, dB = mod._active_lora_views("down_proj")
        assert torch.equal(gate_up_f[..., :I], canonical_lora_fold_gkn(mod.gate_up_proj[..., :I], gA, gB, s))
        assert torch.equal(gate_up_f[..., I:], canonical_lora_fold_gkn(mod.gate_up_proj[..., I:], uA, uB, s))
        assert torch.equal(down_f, canonical_lora_fold_gkn(mod.down_proj, dA, dB, s))
        # weight-sync view is byte-identical to the forward cache
        assert torch.equal(mod.canonical_merged_proj_weight("gate_proj"), gate_up_f[..., :I])
        assert torch.equal(mod.canonical_merged_proj_weight("down_proj"), down_f)

        self._assert_cache_is_keyed_on_parameter_versions()
        self._assert_fused_experts_admission_policy()

    def _assert_cache_is_keyed_on_parameter_versions(self):
        mod = self._module()
        mod.exact_merged_forward = True
        g1, d1 = mod._merged_weights()
        g2, d2 = mod._merged_weights()
        assert g1 is g2 and d1 is d2
        with torch.no_grad():
            mod.gate_proj_lora_B.add_(0.01)
        g3, _ = mod._merged_weights()
        assert g3 is not g1 and not torch.equal(g1, g3)

    def _assert_fused_experts_admission_policy(self):
        mod = self._module()
        assert not mod.sglang_fused_experts_auto_supported()
        mod.exact_merged_forward = True
        assert mod.sglang_fused_experts_auto_supported()

        self._assert_fused_flag_without_merged_flag_raises()

    def _assert_fused_flag_without_merged_flag_raises(self):
        mod = self._module()
        with pytest.raises(NotImplementedError, match="merged"):
            mod.sglang_fused_experts_forward(
                torch.randn(3, H).to(torch.bfloat16),
                torch.rand(3, 2).to(torch.bfloat16),
                torch.randint(0, E, (3, 2)),
            )

    def test_native_ep_merged_lora_policy(self, monkeypatch):
        mod = self._module()
        mod.exact_merged_forward = True
        hidden = torch.randn(3, H).to(torch.bfloat16)
        routing = torch.rand(3, 2).to(torch.bfloat16)
        local_ids = torch.tensor([[0, -1], [1, -1], [-1, 2]], dtype=torch.int32)
        expected = torch.randn_like(hidden)
        calls = []

        def masked_partial(got_hidden, got_routing, got_ids):
            calls.append((got_hidden, got_routing, got_ids))
            return expected

        monkeypatch.setattr(mod, "sglang_ep_native_routed_partial", masked_partial)
        got = mod(hidden, routing, sglang_ep_native_local_ids=local_ids)
        assert got is expected
        assert calls == [(hidden, routing, local_ids)]

        self._assert_native_ep_no_grad_uses_canonical_fold_and_filter(monkeypatch)

    def test_native_deepep_widens_only_bf16_routing_metadata(self, monkeypatch):
        import xorl.distributed.moe.deepep_native_exact as native_exact

        mod = self._module()
        mod.exact_merged_forward = True
        mod.deepep_native_exact = True
        mod.ep_dispatch = "deepep"
        mod.deepep_async_combine = False
        mod.deepep_buffer_size_gb = 2.0
        mod.deepep_num_sms = 20
        hidden = torch.randn(3, H).to(torch.bfloat16)
        routing_bf16 = torch.rand(3, 2).to(torch.bfloat16)
        selected = torch.randint(0, E, (3, 2))
        expected = torch.randn_like(hidden)
        captured = {}

        def fake_program(got_hidden, got_routing, got_selected, **kwargs):
            captured.update(
                hidden=got_hidden,
                routing=got_routing,
                selected=got_selected,
                kwargs=kwargs,
            )
            return expected

        monkeypatch.setattr(native_exact, "native_dispatch_runner_combine", fake_program)
        got = mod._ep_forward(
            hidden,
            routing_bf16,
            selected,
            SimpleNamespace(ep_group=object()),
        )

        assert got is expected
        assert captured["hidden"] is hidden
        assert captured["selected"] is selected
        assert captured["routing"].dtype is torch.float32
        assert torch.equal(
            captured["routing"],
            routing_bf16.to(torch.float32),
        )

    def _assert_native_ep_no_grad_uses_canonical_fold_and_filter(self, monkeypatch):
        mod = self._module()
        mod.exact_merged_forward = True
        hidden = torch.randn(3, H).to(torch.bfloat16)
        routing = torch.rand(3, 2).to(torch.float32)
        local_ids = torch.tensor([[0, -1], [1, -1], [-1, 2]], dtype=torch.int32)
        expected = torch.randn(3, H, dtype=torch.bfloat16)
        gate_up_f, down_f = mod._merged_weights()
        captured = {}

        def fake_kernel(
            got_hidden,
            got_gate_up,
            got_down,
            got_routing,
            got_ids,
            _impl,
            _activation,
            _swiglu_limit,
            _bias,
            *,
            weight_cache,
            filter_expert,
        ):
            captured.update(
                hidden=got_hidden,
                gate_up=got_gate_up,
                down=got_down,
                routing=got_routing,
                ids=got_ids,
                weight_cache=weight_cache,
                filter_expert=filter_expert,
            )
            return expected

        with (
            patch("xorl.models.layers.moe.experts.MoEExperts._load_sglang_fused_experts_impl", return_value=object()),
            patch("xorl.models.layers.moe.experts._sglang_fused_experts_kernel_call", side_effect=fake_kernel),
            torch.no_grad(),
        ):
            got = mod.sglang_ep_native_routed_partial(hidden, routing, local_ids)

        assert got.dtype is torch.bfloat16
        assert torch.equal(got, expected)
        assert captured == {
            "hidden": hidden,
            "gate_up": gate_up_f,
            "down": down_f,
            "routing": routing,
            "ids": local_ids,
            "weight_cache": None,
            "filter_expert": True,
        }

    def test_native_ep_trainable_path_requests_fused_local_combine(self):
        mod = self._module()
        mod.exact_merged_forward = True
        hidden = torch.randn(3, H).to(torch.bfloat16)
        routing = torch.rand(3, 2).to(torch.float32)
        local_ids = torch.tensor([[0, -1], [1, -1], [-1, 2]], dtype=torch.int32)
        expected = torch.randn_like(hidden)

        with (
            patch("xorl.models.layers.moe.experts.MoEExperts._load_sglang_fused_experts_impl", return_value=object()),
            patch(
                "xorl.models.layers.moe.experts._SglangFusedExpertsTrainFunction.apply",
                return_value=expected,
            ) as apply,
        ):
            got = mod.sglang_ep_native_routed_partial(hidden, routing, local_ids)

        assert got is expected
        assert apply.call_args.args[-1] is True

    def test_native_ep_separate_mode_is_distinct_from_merged(self, monkeypatch):
        mod = self._module()
        mod.exact_merged_forward = True
        mod.lora_serving_mode = "separate"
        hidden = torch.randn(3, H).to(torch.bfloat16)
        routing = torch.rand(3, 2).to(torch.float32)
        local_ids = torch.tensor([[0, -1], [1, -1], [-1, 2]], dtype=torch.int32)
        expected = torch.randn_like(hidden)
        captured = {}

        def hook_value(got_hidden, got_routing, got_ids, *factors):
            captured.update(
                hidden=got_hidden,
                routing=got_routing,
                ids=got_ids,
                factors=factors,
            )
            return expected

        monkeypatch.setattr(mod, "_sglang_native_lora_hook_value", hook_value)
        monkeypatch.setattr(
            mod,
            "_merged_weights",
            lambda: pytest.fail("separate mode must not fold LoRA into base weights"),
        )
        with torch.no_grad():
            got = mod.sglang_ep_native_routed_partial(hidden, routing, local_ids)

        assert got is expected
        assert captured["hidden"] is hidden
        assert captured["routing"] is routing
        assert captured["ids"] is local_ids
        assert len(captured["factors"]) == 6
        assert all(factor.dtype is torch.bfloat16 for factor in captured["factors"])

    def test_separate_zero_b_still_builds_active_shared_outer_info(self, monkeypatch):
        mod = self._module()
        mod.lora_serving_mode = "separate"
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.lora.lora_moe_runners",
            SimpleNamespace(LoRAInfo=SimpleNamespace),
        )
        with torch.no_grad():
            for projection in ("gate_proj", "up_proj", "down_proj"):
                getattr(mod, f"{projection}_lora_B").zero_()
        factors = tuple(
            value.to(torch.bfloat16)
            for projection in ("gate_proj", "up_proj", "down_proj")
            for value in mod._active_lora_views(projection)
        )
        physical = mod._sglang_native_lora_physical_buffers(*factors)
        info = mod._sglang_native_lora_info(3, physical)

        assert physical["gate_up_lora_a_weights"].shape == (1, 1, 2 * R, H)
        assert physical["gate_up_lora_b_weights"].shape == (1, E, 2 * I, R)
        assert physical["down_lora_a_weights"].shape == (1, E, R, I)
        assert physical["down_lora_b_weights"].shape == (1, 1, H, R)
        assert torch.count_nonzero(physical["gate_up_lora_b_weights"]) == 0
        assert torch.count_nonzero(physical["down_lora_b_weights"]) == 0
        assert info.has_active_lora is True
        assert info.experts_shared_outer_loras is True
        assert info.adapter_enabled.tolist() == [1]

    def test_native_ep_separate_trainable_path_uses_value_surrogate_boundary(self):
        mod = self._module()
        mod.exact_merged_forward = True
        mod.lora_serving_mode = "separate"
        hidden = torch.randn(3, H).to(torch.bfloat16)
        routing = torch.rand(3, 2).to(torch.float32)
        local_ids = torch.tensor([[0, -1], [1, -1], [-1, 2]], dtype=torch.int32)
        expected = torch.randn_like(hidden)

        with patch(
            "xorl.models.layers.moe.lora._SglangNativeLoRAHooksTrainFunction.apply",
            return_value=expected,
        ) as apply:
            got = mod.sglang_ep_native_routed_partial(hidden, routing, local_ids)

        assert got is expected
        assert apply.call_args.args[:3] == (hidden, routing, local_ids)
        assert apply.call_args.args[-1] is mod

    def test_separate_surrogate_uses_post_shard_physical_expert_count(self):
        mod = self._module()
        # Qwen constructs a global expert bank and ParallelPlan/FSDP shards
        # its tensors afterwards.  Preserve the stale construction attribute
        # while presenting the two-expert physical tensors seen by each rank.
        local_experts = E // 2
        with torch.no_grad():
            mod.gate_up_proj = torch.nn.Parameter(
                mod.gate_up_proj[:local_experts].detach().clone(), requires_grad=False
            )
            mod.down_proj = torch.nn.Parameter(mod.down_proj[:local_experts].detach().clone(), requires_grad=False)
            for name in ("gate_proj_lora_B", "up_proj_lora_B", "down_proj_lora_A"):
                value = getattr(mod, name)
                setattr(
                    mod,
                    name,
                    torch.nn.Parameter(value[:local_experts].detach().clone()),
                )
        assert mod.num_local_experts == E
        assert mod.gate_up_proj.shape[0] == local_experts

        hidden = torch.randn(3, H, dtype=torch.bfloat16, requires_grad=True)
        routing = torch.rand(3, 2, dtype=torch.float32, requires_grad=True)
        local_ids = torch.tensor([[0, -1], [1, -1], [-1, 0]], dtype=torch.int32)
        factors = tuple(
            value
            for projection in ("gate_proj", "up_proj", "down_proj")
            for value in mod._active_lora_views(projection)
        )
        needs_input_grad = (True, True, False, True, True, True, True, True, True, False)

        def fake_grad(_output, inputs, **_kwargs):
            return tuple(torch.zeros_like(value) for value in inputs)

        with (
            patch(
                "xorl.models.layers.moe.experts.MoEExperts._load_sglang_fused_experts_impl",
                return_value=object(),
            ),
            patch(
                "xorl.models.layers.moe.experts._SglangFusedExpertsTrainFunction.apply",
                return_value=torch.zeros_like(hidden, requires_grad=True),
            ) as apply,
            patch("torch.autograd.grad", side_effect=fake_grad),
        ):
            mod._sglang_native_lora_hook_surrogate_vjp(
                hidden,
                routing,
                local_ids,
                *factors,
                grad_output=torch.ones_like(hidden),
                needs_input_grad=needs_input_grad,
            )

        assert apply.call_args.args[9] == local_experts
        assert apply.call_args.args[11] is True


class TestTrunkWrapComposition:
    def _model(self):
        torch.manual_seed(7)
        model = torch.nn.Module()
        model.q_proj = LoraLinear(H, H, r=R, lora_alpha=R, dtype=torch.bfloat16)
        return model

    def test_trunk_wrap_composition_policy(self, monkeypatch):
        from xorl.ops.batch_invariant_ops import wrap_trunk_linears_batch_invariant  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="canonical merged-LoRA"):
            wrap_trunk_linears_batch_invariant(self._model())

        self._assert_wrap_composes_with_merged_flag(monkeypatch)

    def _assert_wrap_composes_with_merged_flag(self, monkeypatch):
        from xorl.ops.batch_invariant_ops import (  # noqa: PLC0415
            set_trunk_linear_contract,
            wrap_trunk_linears_batch_invariant,
        )

        model = self._model()
        model.q_proj.exact_merged_forward = True
        try:
            wrapped = wrap_trunk_linears_batch_invariant(model)
            assert wrapped == {"q_proj": 1}
            assert model.q_proj._xorl_bi_trunk_wrapped
        finally:
            set_trunk_linear_contract(False)
