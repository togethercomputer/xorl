"""Tests for EP-aware gradient clipping (clip_grad_norm / ep_fsdp2_clip_grad_norm).

Verifies:
1. _build_ep_param_groups correctly classifies _skip_fsdp and plain params.
2. ep_fsdp2_clip_grad_norm computes the correct total norm from all three
   parameter groups (non-EP, EP-FSDP, EP-local) and clips uniformly.
3. No double-division of EP gradients (the bug this branch fixes).
4. inf-norm path works correctly.

All tests run single-rank: process groups are None so all_reduce is skipped,
letting us verify the local math without distributed infrastructure.
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xorl.distributed.fsdp2.clip_grad_norm import (
    clip_grad_norm,
    ep_fsdp2_clip_grad_norm,
)
from xorl.distributed.torch_parallelize import _build_ep_param_groups


pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param(*shape, grad=None):
    """Create a plain parameter with an optional gradient."""
    p = nn.Parameter(torch.randn(*shape))
    if grad is not None:
        p.grad = grad
    return p


def _mock_parallel_state(ep_enabled=True):
    """Return a mock parallel state with all groups set to None (single-rank)."""
    ps = MagicMock()
    ps.ep_enabled = ep_enabled
    ps.fsdp_group = None
    ps.ep_group = None
    ps.tp_enabled = False
    ps.tp_group = None
    ps.ep_fsdp_device_mesh = None
    return ps


def _l2_norm(*params):
    """Compute the expected L2 norm across multiple params' gradients."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().to(torch.float32).norm(2).item() ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# 1. _build_ep_param_groups classification
# ---------------------------------------------------------------------------


class TestBuildEPParamGroups:
    """Test that _build_ep_param_groups classifies params correctly."""

    def test_skip_fsdp_params_classified_as_ep(self):
        """Params from _skip_fsdp modules go into the EP group."""
        model = nn.Module()
        # Regular submodule
        regular = nn.Linear(8, 8)
        model.add_module("regular", regular)
        # _skip_fsdp submodule (e.g. QLoRAMoeExperts)
        expert = nn.Linear(8, 8)
        expert._skip_fsdp = True
        model.add_module("expert", expert)

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            _build_ep_param_groups(model)

        assert hasattr(model, "_ep_param_groups")
        ep_ids = {id(p) for p in model._ep_param_groups["ep"]}
        non_ep_ids = {id(p) for p in model._ep_param_groups["non_ep"]}

        # Expert params in EP group
        for p in expert.parameters():
            assert id(p) in ep_ids
        # Regular params in non-EP group
        for p in regular.parameters():
            assert id(p) in non_ep_ids

    def test_no_skip_fsdp_all_non_ep(self):
        """Without _skip_fsdp modules, all plain params go to non-EP."""
        model = nn.Module()
        model.add_module("linear", nn.Linear(8, 8))

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            _build_ep_param_groups(model)

        assert len(model._ep_param_groups["ep"]) == 0
        assert len(model._ep_param_groups["non_ep"]) == 2  # weight + bias

    def test_nested_skip_fsdp_params_all_classified(self):
        """All params inside nested _skip_fsdp modules are classified as EP."""
        model = nn.Module()
        # Mimic QLoRAMoeExperts: a _skip_fsdp module with multiple sub-params
        expert = nn.Module()
        expert._skip_fsdp = True
        expert.lora_A = nn.Parameter(torch.randn(4, 32, 8))
        expert.lora_B = nn.Parameter(torch.randn(4, 8, 64))
        expert.base_weight = nn.Parameter(torch.randn(4, 32, 64), requires_grad=False)
        model.add_module("expert", expert)
        model.add_module("non_expert", nn.Linear(32, 32))

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            _build_ep_param_groups(model)

        ep_ids = {id(p) for p in model._ep_param_groups["ep"]}
        assert id(expert.lora_A) in ep_ids
        assert id(expert.lora_B) in ep_ids
        assert id(expert.base_weight) in ep_ids  # even frozen params are classified
        # non-expert params not in EP group
        non_ep_ids = {id(p) for p in model._ep_param_groups["non_ep"]}
        for p in model.non_expert.parameters():
            assert id(p) in non_ep_ids


# ---------------------------------------------------------------------------
# 2. ep_fsdp2_clip_grad_norm: norm computation and clipping
# ---------------------------------------------------------------------------


class TestEPFSDP2ClipGradNorm:
    """Test norm computation and gradient clipping logic."""

    def _setup_model(self, ep_grads, non_ep_grads):
        """Create a model with _ep_param_groups populated from given gradient tensors.

        Args:
            ep_grads: list of gradient tensors for EP-local params.
            non_ep_grads: list of gradient tensors for non-EP params.

        Returns:
            (model, ep_params, non_ep_params)
        """
        ep_params = []
        for g in ep_grads:
            p = _make_param(*g.shape, grad=g)
            ep_params.append(p)

        non_ep_params = []
        for g in non_ep_grads:
            p = _make_param(*g.shape, grad=g)
            non_ep_params.append(p)

        model = MagicMock()
        model._ep_param_groups = {"ep": ep_params, "non_ep": non_ep_params}
        return model, ep_params, non_ep_params

    def test_l2_norm_single_group(self):
        """Total L2 norm is correct when only non-EP params have grads."""
        g = torch.tensor([3.0, 4.0])  # norm = 5
        model, _, non_ep = self._setup_model(ep_grads=[], non_ep_grads=[g])

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=100.0)

        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)

    def test_l2_norm_combined_groups(self):
        """Total L2 norm combines EP-local and non-EP norms correctly."""
        ep_g = torch.tensor([3.0, 0.0])  # norm = 3
        non_ep_g = torch.tensor([0.0, 4.0])  # norm = 4
        # combined: sqrt(9 + 16) = 5
        model, ep_params, non_ep_params = self._setup_model(ep_grads=[ep_g], non_ep_grads=[non_ep_g])

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=100.0)

        expected = _l2_norm(*ep_params, *non_ep_params)
        assert total_norm.item() == pytest.approx(expected, abs=1e-5)

    def test_clipping_reduces_gradients(self):
        """When total_norm > max_norm, all gradients are scaled down uniformly."""
        ep_g = torch.tensor([6.0, 0.0])
        non_ep_g = torch.tensor([0.0, 8.0])
        # total norm = sqrt(36 + 64) = 10
        model, ep_params, non_ep_params = self._setup_model(ep_grads=[ep_g], non_ep_grads=[non_ep_g])
        max_norm = 5.0  # clip factor = 5/10 = 0.5

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=max_norm)

        assert total_norm.item() == pytest.approx(10.0, abs=1e-5)
        # Both EP and non-EP grads should be scaled by 0.5
        torch.testing.assert_close(ep_params[0].grad, torch.tensor([3.0, 0.0]))
        torch.testing.assert_close(non_ep_params[0].grad, torch.tensor([0.0, 4.0]))

    def test_no_clipping_below_max(self):
        """When total_norm <= max_norm, gradients are unchanged."""
        g = torch.tensor([3.0, 4.0])  # norm = 5
        model, _, non_ep = self._setup_model(ep_grads=[], non_ep_grads=[g])

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=10.0)

        # Gradients unchanged
        torch.testing.assert_close(non_ep[0].grad, torch.tensor([3.0, 4.0]))

    def test_ep_grads_not_double_scaled(self):
        """EP gradients are NOT divided by ep_size — the double-division fix."""
        ep_g = torch.tensor([6.0, 8.0])  # norm = 10
        model, ep_params, _ = self._setup_model(ep_grads=[ep_g], non_ep_grads=[])

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=100.0)

        # Gradients should be completely unchanged (no scaling, no division)
        torch.testing.assert_close(ep_params[0].grad, torch.tensor([6.0, 8.0]))
        assert total_norm.item() == pytest.approx(10.0, abs=1e-5)

    def test_inf_norm(self):
        """Inf-norm returns the max absolute gradient value."""
        ep_g = torch.tensor([3.0, -7.0])
        non_ep_g = torch.tensor([5.0, 2.0])
        model, _, _ = self._setup_model(ep_grads=[ep_g], non_ep_grads=[non_ep_g])

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=100.0, norm_type=float("inf"))

        assert total_norm.item() == pytest.approx(7.0, abs=1e-5)

    def test_inf_norm_clips_correctly(self):
        """Inf-norm clipping scales gradients when max element exceeds max_norm."""
        ep_g = torch.tensor([3.0, -10.0])
        non_ep_g = torch.tensor([5.0, 2.0])
        model, ep_params, non_ep_params = self._setup_model(ep_grads=[ep_g], non_ep_grads=[non_ep_g])
        max_norm = 5.0  # total inf-norm is 10, clip factor = 5/10 = 0.5

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=max_norm, norm_type=float("inf"))

        assert total_norm.item() == pytest.approx(10.0, abs=1e-5)
        torch.testing.assert_close(ep_params[0].grad, torch.tensor([1.5, -5.0]))
        torch.testing.assert_close(non_ep_params[0].grad, torch.tensor([2.5, 1.0]))

    def test_empty_groups(self):
        """Handles empty parameter groups without errors."""
        model = MagicMock()
        model._ep_param_groups = {"ep": [], "non_ep": []}

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=1.0)

        assert total_norm.item() == 0.0

    def test_params_without_grads_skipped(self):
        """Params with grad=None are excluded from norm computation."""
        g = torch.tensor([3.0, 4.0])  # norm = 5
        p_with_grad = _make_param(2, grad=g)
        p_no_grad = _make_param(2)  # no gradient

        model = MagicMock()
        model._ep_param_groups = {"ep": [p_no_grad], "non_ep": [p_with_grad]}

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=100.0)

        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. _skip_fsdp end-to-end: classify → clip
# ---------------------------------------------------------------------------


class TestSkipFSDPClipEndToEnd:
    """End-to-end test: _skip_fsdp params flow through classification into clipping.

    Mimics the QLoRA EP path where expert LoRA params are plain tensors
    (not FSDP-managed) and must be:
    - Classified as EP by _build_ep_param_groups
    - Treated as ep_local_params (no reduction) in ep_fsdp2_clip_grad_norm
    - Not scaled/divided by ep_size (the double-division fix)
    - Clipped with the same coefficient as non-EP params
    """

    def test_skip_fsdp_classify_then_clip(self):
        """_skip_fsdp expert grads are classified as EP-local and clipped correctly."""
        model = nn.Module()

        # Non-expert param (mimics attention/mlp weights)
        regular = nn.Linear(4, 4, bias=False)
        model.add_module("regular", regular)

        # _skip_fsdp expert param (mimics QLoRA LoRA weights)
        expert = nn.Module()
        expert._skip_fsdp = True
        expert.lora = nn.Parameter(torch.randn(2, 4))
        model.add_module("expert", expert)

        # Assign known gradients: expert norm=6, regular norm=8, total=10
        expert.lora.grad = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])  # flat norm = 3
        # Wait, let me make the math cleaner
        expert.lora.grad = torch.zeros(2, 4)
        expert.lora.grad[0, 0] = 6.0  # norm = 6
        regular.weight.grad = torch.zeros(4, 4)
        regular.weight.grad[0, 0] = 8.0  # norm = 8
        # total = sqrt(36 + 64) = 10

        ps = _mock_parallel_state()
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            _build_ep_param_groups(model)

        # Verify classification
        ep_ids = {id(p) for p in model._ep_param_groups["ep"]}
        assert id(expert.lora) in ep_ids
        non_ep_ids = {id(p) for p in model._ep_param_groups["non_ep"]}
        assert id(regular.weight) in non_ep_ids

        # Clip with max_norm=5 → clip_coeff = 5/10 = 0.5
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=5.0)

        assert total_norm.item() == pytest.approx(10.0, abs=1e-5)
        # Both groups scaled uniformly by 0.5
        assert expert.lora.grad[0, 0].item() == pytest.approx(3.0, abs=1e-5)
        assert regular.weight.grad[0, 0].item() == pytest.approx(4.0, abs=1e-5)

    def test_skip_fsdp_grads_not_reduced_or_divided(self):
        """_skip_fsdp grads contribute their raw local norm — no all-reduce, no ep_size division."""
        model = nn.Module()

        expert = nn.Module()
        expert._skip_fsdp = True
        expert.weight = nn.Parameter(torch.randn(4, 8))
        expert.weight.grad = torch.full((4, 8), 2.0)  # norm = 2 * sqrt(32)
        model.add_module("expert", expert)

        ps = _mock_parallel_state()
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            _build_ep_param_groups(model)
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=1000.0)

        expected_norm = torch.tensor(2.0 * math.sqrt(32))
        assert total_norm.item() == pytest.approx(expected_norm.item(), abs=1e-4)
        # Gradient unchanged (no scaling applied since norm < max_norm)
        assert (expert.weight.grad == 2.0).all()


# ---------------------------------------------------------------------------
# 4. clip_grad_norm dispatch
# ---------------------------------------------------------------------------


class TestClipGradNormDispatch:
    """Test that clip_grad_norm dispatches to ep_fsdp2_clip_grad_norm when appropriate."""

    def test_dispatches_to_ep_path_when_ep_param_groups_present(self):
        """Models with _ep_param_groups use the EP-aware clip path."""
        g = torch.tensor([3.0, 4.0])
        p = _make_param(2, grad=g)
        model = MagicMock()
        model._ep_param_groups = {"ep": [], "non_ep": [p]}
        # hasattr check needs to work
        model.__dict__["_ep_param_groups"] = model._ep_param_groups

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = clip_grad_norm(model, max_norm=100.0)

        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)

    def test_falls_through_without_ep_param_groups(self):
        """Models without _ep_param_groups use the standard FSDP2 path."""
        g = torch.tensor([3.0, 4.0])
        p = _make_param(2, grad=g)

        model = MagicMock(spec=[])  # empty spec, so hasattr(_ep_param_groups) is False
        model.parameters = MagicMock(return_value=iter([p]))

        ps = _mock_parallel_state(ep_enabled=False)
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            total_norm = clip_grad_norm(model, max_norm=100.0)

        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)
