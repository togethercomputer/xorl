"""Tests for EP-aware gradient clipping (clip_grad_norm / ep_fsdp2_clip_grad_norm).

Verifies:
1. _build_ep_param_groups correctly classifies _skip_fsdp and plain params.
2. ep_fsdp2_clip_grad_norm computes the correct total norm from all three
   parameter groups (non-EP, EP-FSDP, EP-local) and clips uniformly.
3. No double-division of EP gradients (the bug this branch fixes).
4. inf-norm path works correctly.

Most tests run single-rank for local math; the final gate uses a live two-rank
Gloo process group for EP reduction, replicated-gradient synchronization, and
the non-finite guard.
"""

import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from xorl.distributed.ep_gradients import synchronize_ep_replicated_gradients
from xorl.distributed.fsdp2.clip_grad_norm import (
    _fsdp2_reduce_group,
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


# ---------------------------------------------------------------------------
# 1. _build_ep_param_groups classification
# ---------------------------------------------------------------------------


class TestBuildEPParamGroups:
    """Test that _build_ep_param_groups classifies params correctly."""

    def _assert_shared_ep_replica_is_recorded_separately_for_clipping(self):
        model = nn.Module()
        expert = nn.Module()
        expert._skip_fsdp = True
        expert.weight = nn.Parameter(torch.randn(2, 4))
        shared = nn.Module()
        shared._skip_fsdp = True
        shared._ep_gradient_reduction_domain = "ep_sum"
        shared.weight = nn.Parameter(torch.randn(1, 4))
        model.add_module("expert", expert)
        model.add_module("shared", shared)
        model._fqn2spec_info = {
            "expert.weight": SimpleNamespace(placement=Shard(0)),
            "shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="ep_sum"),
        }

        _build_ep_param_groups(model)

        assert {id(p) for p in model._ep_param_groups["ep"]} == {id(expert.weight), id(shared.weight)}
        assert {id(p) for p in model._ep_param_groups["ep_replicated"]} == {id(shared.weight)}
        assert {id(p) for p in model._ep_param_groups["ep_replicated_gradient_sync"]} == {id(shared.weight)}
        assert model._ep_replicated_gradient_sync_enabled is True
        assert getattr(shared.weight, "_xorl_ep_replicated_gradient_hook", None) is None

    def test_replicated_classification_follows_declared_reduction_not_shape(self):
        """A singleton expert axis is NOT evidence of replication.

        With ``ep_size == num_experts`` a rank-unique per-expert slice is
        ``[1, ...]``-shaped and stamped ``Replicate()``; only the declared
        ``EP_SUM`` reduction marks a genuine replica whose norm may be
        averaged across EP.
        """
        model = nn.Module()
        per_expert = nn.Module()
        per_expert._skip_fsdp = True
        per_expert.weight = nn.Parameter(torch.randn(1, 4))
        shared = nn.Module()
        shared._skip_fsdp = True
        shared.weight = nn.Parameter(torch.randn(2, 4))
        model.add_module("per_expert", per_expert)
        model.add_module("shared", shared)
        model._fqn2spec_info = {
            "per_expert.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="none"),
            "shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="ep_sum"),
        }

        _build_ep_param_groups(model)

        assert {id(p) for p in model._ep_param_groups["ep"]} == {id(per_expert.weight), id(shared.weight)}
        assert {id(p) for p in model._ep_param_groups["ep_replicated"]} == {id(shared.weight)}
        assert {id(p) for p in model._ep_param_groups["ep_replicated_gradient_sync"]} == {id(shared.weight)}

    def test_frozen_replicas_are_not_queued_for_gradient_sync(self):
        model = nn.Module()
        shared = nn.Module()
        shared._skip_fsdp = True
        shared.weight = nn.Parameter(torch.randn(1, 4), requires_grad=False)
        model.add_module("shared", shared)
        model._fqn2spec_info = {
            "shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="ep_sum"),
        }

        _build_ep_param_groups(model)

        assert {id(p) for p in model._ep_param_groups["ep_replicated"]} == {id(shared.weight)}
        assert model._ep_param_groups["ep_replicated_gradient_sync"] == []
        assert model._ep_replicated_gradient_sync_enabled is False

    def test_ep_sum_replica_off_the_ep_mesh_fails_closed(self):
        """An EP_SUM replica that would miss the optimizer-boundary sync raises."""
        model = nn.Module()
        shared = nn.Module()  # neither _skip_fsdp nor an ep_fsdp DTensor
        shared.weight = nn.Parameter(torch.randn(1, 4))
        model.add_module("shared", shared)
        model._fqn2spec_info = {
            "shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="ep_sum"),
        }

        with pytest.raises(RuntimeError, match="would silently diverge"):
            _build_ep_param_groups(model)


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

    def _assert_norm_modes_and_empty_gradient_policy(self):
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

        self._assert_empty_groups_and_params_without_grads()

    def _assert_empty_groups_and_params_without_grads(self):
        g = torch.tensor([3.0, 4.0])  # norm = 5
        p_with_grad = _make_param(2, grad=g)
        p_no_grad = _make_param(2)

        model = MagicMock()
        model._ep_param_groups = {"ep": [p_no_grad], "non_ep": [p_with_grad]}

        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=100.0)
        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)

        model._ep_param_groups = {"ep": [], "non_ep": []}
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=_mock_parallel_state()):
            total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=1.0)
        assert total_norm.item() == 0.0


@pytest.mark.parametrize(
    ("norm_type", "expected"),
    ((2.0, 25.0), (float("inf"), 4.0)),
)
def test_singleton_reduce_group_does_not_launch_collective(norm_type, expected):
    """A size-one mesh dimension contributes only its local statistic."""

    param = _make_param(2, grad=torch.tensor([3.0, -4.0]))
    singleton_group = MagicMock()

    with (
        patch.object(dist, "get_world_size", return_value=1) as get_world_size,
        patch.object(dist, "all_reduce") as all_reduce,
    ):
        total = _fsdp2_reduce_group(
            params=[param],
            norm_type=norm_type,
            reduce_groups=[("ep_fsdp", singleton_group)],
        )

    assert total.item() == pytest.approx(expected)
    get_world_size.assert_called_once_with(singleton_group)
    all_reduce.assert_not_called()


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

    def _assert_skip_fsdp_classification_and_raw_local_clip_policy(self):
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

        self._assert_skip_fsdp_grads_not_reduced_or_divided()

    def _assert_skip_fsdp_grads_not_reduced_or_divided(self):
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

    def _assert_clip_grad_norm_dispatch_policy(self):
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

        self._assert_falls_through_without_ep_param_groups()

    def _assert_falls_through_without_ep_param_groups(self):
        """Models without _ep_param_groups use the standard FSDP2 path."""
        g = torch.tensor([3.0, 4.0])
        p = _make_param(2, grad=g)

        model = MagicMock(spec=[])  # empty spec, so hasattr(_ep_param_groups) is False
        model.parameters = MagicMock(return_value=iter([p]))

        ps = _mock_parallel_state(ep_enabled=False)
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            total_norm = clip_grad_norm(model, max_norm=100.0)

        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 5. Flat path with mixed-mesh DTensor grads (no _ep_param_groups)
# ---------------------------------------------------------------------------


@pytest.fixture
def single_rank_dist():
    """World-1 gloo process group for building CPU DTensors on real device meshes."""
    created = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29517",
            rank=0,
            world_size=1,
        )
        created = True
    yield
    if created:
        dist.destroy_process_group()


class TestMixedMeshFlatPath:
    """External callers (no _ep_param_groups) with params on dp_shard + ep_fsdp meshes.

    Regression test: torch's default foreach grouping mixes the meshes into one
    _foreach op and raises 'Could not run pointwise computation across different
    mesh' deep inside torch. The flat path must fall back to per-tensor clipping.
    """

    def _mixed_mesh_model(self):
        mesh_dp = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
        mesh_ep = init_device_mesh("cpu", (1,), mesh_dim_names=("ep_fsdp",))

        model = nn.Module()
        p_dp = nn.Parameter(distribute_tensor(torch.zeros(2), mesh_dp, [Shard(0)]))
        p_dp.grad = distribute_tensor(torch.tensor([3.0, 0.0]), mesh_dp, [Shard(0)])
        p_ep = nn.Parameter(distribute_tensor(torch.zeros(2), mesh_ep, [Shard(0)]))
        p_ep.grad = distribute_tensor(torch.tensor([0.0, 4.0]), mesh_ep, [Shard(0)])
        model.register_parameter("p_dp", p_dp)
        model.register_parameter("p_ep", p_ep)
        return model, p_dp, p_ep

    def test_mixed_mesh_foreach_policy(self, single_rank_dist):
        TestBuildEPParamGroups()._assert_shared_ep_replica_is_recorded_separately_for_clipping()
        TestEPFSDP2ClipGradNorm()._assert_norm_modes_and_empty_gradient_policy()
        TestSkipFSDPClipEndToEnd()._assert_skip_fsdp_classification_and_raw_local_clip_policy()
        TestClipGradNormDispatch()._assert_clip_grad_norm_dispatch_policy()
        model, p_dp, p_ep = self._mixed_mesh_model()

        ps = _mock_parallel_state(ep_enabled=True)
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            total_norm = clip_grad_norm(model, max_norm=1.0)

        assert total_norm.item() == pytest.approx(5.0, abs=1e-5)
        # clip coefficient = 1.0 / 5.0
        assert p_dp.grad.to_local()[0].item() == pytest.approx(0.6, abs=1e-4)
        assert p_ep.grad.to_local()[1].item() == pytest.approx(0.8, abs=1e-4)

        self._assert_explicit_foreach_true_still_raises()

    def _assert_explicit_foreach_true_still_raises(self):
        """An explicit foreach=True is honored — only the default is made safe."""
        model, _, _ = self._mixed_mesh_model()

        ps = _mock_parallel_state(ep_enabled=True)
        with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=ps):
            with pytest.raises(RuntimeError, match="different mesh"):
                clip_grad_norm(model, max_norm=1.0, foreach=True)


def test_real_two_rank_ep_clip_and_nonfinite_gate():
    """Exercise the EP reduction and finite gate on a live two-rank group."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_EP_CLIP_DISTRIBUTED_WORKER": "1"},
    )
    result.assert_success("two-rank EP clip reduction and non-finite gate")

    _assert_real_three_rank_gradient_participation_mask()


def _assert_real_three_rank_gradient_participation_mask():
    """A cancelling global sum still materializes zero on a missing replica."""

    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=3,
        timeout=120,
        extra_env={"XORL_EP_CLIP_PARTICIPATION_WORKER": "1"},
    )
    result.assert_success("three-rank replicated-gradient participation mask")


def _run_distributed_ep_clip_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ep", "ep_fsdp"))
    model = nn.Module()
    expert = nn.Parameter(torch.zeros(1))
    expert.grad = torch.tensor([3.0 if rank == 0 else 4.0])
    model.register_parameter("expert", expert)
    model._ep_param_groups = {"ep": [expert], "non_ep": []}
    parallel_state = MagicMock()
    parallel_state.ep_enabled = True
    parallel_state.fsdp_group = None
    parallel_state.ep_group = mesh["ep"].get_group()
    parallel_state.ep_fsdp_device_mesh = mesh
    parallel_state.tp_enabled = False
    parallel_state.tp_group = None

    reduction_group_sizes = []
    original_all_reduce = dist.all_reduce

    def _recording_all_reduce(*args, **kwargs):
        reduction_group_sizes.append(dist.get_world_size(kwargs.get("group")))
        return original_all_reduce(*args, **kwargs)

    with (
        patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=parallel_state),
        patch.object(dist, "all_reduce", side_effect=_recording_all_reduce),
    ):
        total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=1.0)
    assert 1 not in reduction_group_sizes
    assert 2 in reduction_group_sizes
    assert total_norm.item() == pytest.approx(5.0, abs=1e-6)
    assert expert.grad.item() == pytest.approx(0.6 if rank == 0 else 0.8, abs=1e-6)

    shared = nn.Parameter(torch.zeros(1))
    model.register_parameter("shared", shared)
    model._ep_param_groups = {
        "ep": [expert, shared],
        "ep_replicated": [shared],
        "ep_replicated_gradient_sync": [shared],
        "non_ep": [],
    }
    model._ep_replicated_gradient_sync_enabled = True
    expert.grad = torch.tensor([3.0 if rank == 0 else 4.0])
    shared.grad = torch.tensor([rank + 1.0])
    with patch("xorl.distributed.parallel_state.get_parallel_state", return_value=parallel_state):
        synchronize_ep_replicated_gradients(model)
    assert shared.grad.item() == pytest.approx(3.0)
    expected_combined_norm = math.sqrt(34.0)  # 25 disjoint + 3^2 synchronized shared
    with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=parallel_state):
        total_norm = ep_fsdp2_clip_grad_norm(model, max_norm=1.0)
    assert total_norm.item() == pytest.approx(expected_combined_norm, abs=1e-6)
    coefficient = 1.0 / expected_combined_norm
    assert expert.grad.item() == pytest.approx((3.0 if rank == 0 else 4.0) * coefficient, abs=1e-6)
    assert shared.grad.item() == pytest.approx(3.0 * coefficient, abs=1e-6)

    expert.grad = torch.tensor([float("nan") if rank == 0 else 4.0])
    shared.grad = torch.tensor([3.0])
    with patch("xorl.distributed.fsdp2.clip_grad_norm.get_parallel_state", return_value=parallel_state):
        with pytest.raises(RuntimeError, match="Non-finite gradient"):
            ep_fsdp2_clip_grad_norm(model, max_norm=1.0, error_if_nonfinite=True)
    if rank == 0:
        assert torch.isnan(expert.grad).all()
    else:
        assert expert.grad.item() == pytest.approx(4.0)

    coalesced = nn.Parameter(torch.zeros(1))
    missing_on_rank = nn.Parameter(torch.zeros(1))
    coalesced.grad = torch.tensor([rank + 1.0])
    if rank == 1:
        missing_on_rank.grad = torch.tensor([2.0])
    model.register_parameter("coalesced", coalesced)
    model.register_parameter("missing_on_rank", missing_on_rank)
    model._ep_param_groups = {
        "ep": [],
        "ep_replicated": [coalesced, missing_on_rank],
        "ep_replicated_gradient_sync": [coalesced, missing_on_rank],
        "non_ep": [],
    }
    model._ep_replicated_gradient_sync_enabled = True
    all_reduce_calls = []
    original_all_reduce = dist.all_reduce

    def _counted_all_reduce(*args, **kwargs):
        all_reduce_calls.append(1)
        return original_all_reduce(*args, **kwargs)

    with patch("xorl.distributed.parallel_state.get_parallel_state", return_value=parallel_state):
        with patch.object(dist, "all_reduce", side_effect=_counted_all_reduce):
            stats = synchronize_ep_replicated_gradients(model)
    assert len(all_reduce_calls) == 1
    assert stats.configured_parameter_count == 2
    assert stats.participating_parameter_count == 2
    assert stats.bucket_count == 1
    assert stats.gradient_bytes == 8
    assert stats.reduced_bytes == 16
    assert coalesced.grad.item() == pytest.approx(3.0)
    assert missing_on_rank.grad.item() == pytest.approx(2.0)
    dist.destroy_process_group()


def _run_distributed_participation_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    parallel_state = MagicMock()
    parallel_state.ep_enabled = True
    parallel_state.ep_group = dist.group.WORLD

    parameter = nn.Parameter(torch.zeros(1))
    if rank == 1:
        parameter.grad = torch.tensor([2.0])
    elif rank == 2:
        parameter.grad = torch.tensor([-2.0])

    model = nn.Module()
    model.register_parameter("parameter", parameter)
    model._ep_param_groups = {
        "ep": [],
        "ep_replicated": [parameter],
        "ep_replicated_gradient_sync": [parameter],
        "non_ep": [],
    }
    model._ep_replicated_gradient_sync_enabled = True

    with patch("xorl.distributed.parallel_state.get_parallel_state", return_value=parallel_state):
        stats = synchronize_ep_replicated_gradients(model)
    assert stats.configured_parameter_count == 1
    assert stats.participating_parameter_count == 1
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(0.0)
    dist.destroy_process_group()


if os.environ.get("XORL_EP_CLIP_DISTRIBUTED_WORKER") == "1":
    _run_distributed_ep_clip_worker()
elif os.environ.get("XORL_EP_CLIP_PARTICIPATION_WORKER") == "1":
    _run_distributed_participation_worker()
