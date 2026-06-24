"""Unit test for ``ParallelPlan.apply`` meta-tensor slicing path.

The meta path replaces full-shape meta params with EP-local-shape meta
params and stamps ``spec_info`` so the downstream ``to_empty()`` allocates
only the local slice and DCP load sees the right per-rank target shapes.

Without this path, the ``skip_weight_loading=True`` (=> ``already_local=True``)
flow used by xorl's meta-init smokes leaves expert tensors at full shape,
and ``to_empty()`` materializes ``[num_experts, H, I]`` per rank — a 16x
overshoot at ``ep_size=16``.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.distributed._tensor import Replicate, Shard

from xorl.distributed.parallel_plan import ParallelPlan, SpecInfo


pytestmark = pytest.mark.cpu


def _fake_ep_fsdp_mesh(ep_size: int):
    """Build a MagicMock that walks like ``parallel_state.ep_fsdp_device_mesh``.

    ``ParallelPlan.apply`` only uses ``ep_fsdp_mesh["ep"].size(-1)`` and
    ``ep_fsdp_mesh.ndim`` from the mesh; everything else is shoveled into
    ``SpecInfo``.
    """
    ep_mesh = MagicMock()
    ep_mesh.size = lambda *_a, **_kw: ep_size
    ep_mesh.ndim = 1
    fsdp_mesh = MagicMock()
    fsdp_mesh.__getitem__ = lambda _self, key: ep_mesh if key == "ep" else MagicMock()
    return fsdp_mesh


class _FakeExpertsModule(nn.Module):
    """Minimal expert-like module with a ``gate_up_proj`` meta param."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden, 2 * inter, device="meta", dtype=torch.bfloat16)
        )
        self.unrelated = nn.Parameter(torch.empty(hidden, device="meta", dtype=torch.bfloat16))


class _FakeModel(nn.Module):
    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.experts = _FakeExpertsModule(num_experts, hidden, inter)


def test_meta_slicing_replaces_full_shape_with_ep_local_shape():
    """Meta param at full shape should be replaced with EP-local-shape meta param."""
    num_experts, ep_size = 16, 4
    H, I = 32, 64

    model = _FakeModel(num_experts=num_experts, hidden=H, inter=I)
    plan = ParallelPlan(ep_plan={"experts.gate_up_proj": Shard(0)})
    fqn2spec = plan.apply(model, _fake_ep_fsdp_mesh(ep_size), already_local=False)

    assert model.experts.gate_up_proj.is_meta
    assert tuple(model.experts.gate_up_proj.shape) == (num_experts // ep_size, H, 2 * I)

    info = fqn2spec["experts.gate_up_proj"]
    assert isinstance(info, SpecInfo)
    assert isinstance(info.placement, Shard) and info.placement.dim == 0

    # The unrelated param is not in the ep_plan — it should be Replicate-stamped.
    assert isinstance(fqn2spec["experts.unrelated"].placement, Replicate)


def test_meta_slicing_dispatches_even_when_already_local_is_true():
    """``already_local=True`` is the smoke's default (set by skip_weight_loading);
    the meta dispatch must still fire so to_empty() doesn't materialize full shape."""
    num_experts, ep_size = 8, 8
    H, I = 16, 16

    model = _FakeModel(num_experts=num_experts, hidden=H, inter=I)
    plan = ParallelPlan(ep_plan={"experts.gate_up_proj": Shard(0)})
    plan.apply(model, _fake_ep_fsdp_mesh(ep_size), already_local=True)

    # Meta path runs first, slices to ep-local even with already_local=True.
    assert tuple(model.experts.gate_up_proj.shape) == (1, H, 2 * I)
    assert model.experts.gate_up_proj.is_meta


def test_meta_slicing_assertion_on_indivisible_size():
    """Non-divisible expert dim should raise the existing ep-divisibility assert."""
    num_experts, ep_size = 7, 4  # 7 % 4 != 0
    model = _FakeModel(num_experts=num_experts, hidden=8, inter=8)
    plan = ParallelPlan(ep_plan={"experts.gate_up_proj": Shard(0)})
    with pytest.raises(AssertionError, match="not divisible by ep_size"):
        plan.apply(model, _fake_ep_fsdp_mesh(ep_size), already_local=False)


def test_meta_slicing_preserves_dtype_and_requires_grad():
    """The new meta param must keep the original dtype and requires_grad flag."""
    num_experts, ep_size = 8, 2
    model = _FakeModel(num_experts=num_experts, hidden=8, inter=16)
    model.experts.gate_up_proj.requires_grad_(False)

    plan = ParallelPlan(ep_plan={"experts.gate_up_proj": Shard(0)})
    plan.apply(model, _fake_ep_fsdp_mesh(ep_size), already_local=False)

    assert model.experts.gate_up_proj.dtype == torch.bfloat16
    assert model.experts.gate_up_proj.requires_grad is False
