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

from xorl.distributed import parallel_plan as parallel_plan_module
from xorl.distributed.gradient_reduction import GradientReductionDomain
from xorl.distributed.parallel_plan import ParallelPlan, SpecInfo
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.parallelize import get_ep_plan as get_glm52_ep_plan


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


class _FakeSharedLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.Module()
        self.experts._ep_gradient_reduction_domain = "ep_sum"
        self.experts.shared_lora = nn.Parameter(torch.empty(1, 8, 4))


class _ExactRoutedModel(nn.Module):
    def __init__(self, *, device: str, ep_rank: int = 7):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Module() for _ in range(4))
        self.model.layers[3].mlp = nn.Module()
        self.model.layers[3].mlp.experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            128,
            128,
            ep_rank=ep_rank,
            device=torch.device(device),
        )

    @property
    def experts(self):
        return self.model.layers[3].mlp.experts


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


def test_parallel_plan_stamps_explicit_replicated_gradient_reduction():
    model = _FakeSharedLoRAModel()
    plan = ParallelPlan(ep_plan={"experts.shared_lora": Shard(0)})

    fqn2spec = plan.apply(model, _fake_ep_fsdp_mesh(ep_size=4), already_local=False)

    info = fqn2spec["experts.shared_lora"]
    assert isinstance(info.placement, Replicate)
    assert info.gradient_reduction == "ep_sum"
    assert model.experts.shared_lora.spec_info.gradient_reduction == "ep_sum"


def test_singleton_axis_without_ep_sum_declaration_fails_closed():
    """A [1, ...] expert param without EP_SUM is ambiguous (shared replica vs
    the local slice of ``ep_size == num_experts``) — the plan refuses to guess."""
    model = _FakeSharedLoRAModel()
    del model.experts._ep_gradient_reduction_domain
    plan = ParallelPlan(ep_plan={"experts.shared_lora": Shard(0)})

    with pytest.raises(ValueError, match="refusing to guess"):
        plan.apply(model, _fake_ep_fsdp_mesh(ep_size=4), already_local=False)


def test_singleton_axis_already_local_annotates_per_expert_shard():
    """With ``already_local=True`` an undeclared singleton is a per-expert
    local slice: annotated as Shard, never averaged as a replica."""
    model = _FakeSharedLoRAModel()
    del model.experts._ep_gradient_reduction_domain
    plan = ParallelPlan(ep_plan={"experts.shared_lora": Shard(0)})

    fqn2spec = plan.apply(model, _fake_ep_fsdp_mesh(ep_size=4), already_local=True)

    info = fqn2spec["experts.shared_lora"]
    assert isinstance(info.placement, Shard)
    assert info.gradient_reduction is GradientReductionDomain.NONE
    assert tuple(model.experts.shared_lora.shape) == (1, 8, 4)


def test_fallback_spec_info_carries_declared_reduction_domain():
    """Plan-gap params keep their owner's declared EP_SUM contract, so the
    optimizer-boundary sync and the norm averaging stay coherent."""
    model = _FakeSharedLoRAModel()
    model.other = nn.Module()
    model.other.weight = nn.Parameter(torch.empty(3, 3))
    plan = ParallelPlan(ep_plan={"experts.nonexistent_pattern": Shard(0)})

    fqn2spec = plan.apply(model, _fake_ep_fsdp_mesh(ep_size=4), already_local=False)

    shared_info = fqn2spec["experts.shared_lora"]
    assert isinstance(shared_info.placement, Replicate)
    assert shared_info.gradient_reduction == "ep_sum"
    other_info = fqn2spec["other.weight"]
    assert isinstance(other_info.placement, Replicate)
    assert other_info.gradient_reduction is GradientReductionDomain.NONE


def test_glm52_exact_meta_ep_plan_preserves_local_base_and_shards_only_expert_factors():
    model = _ExactRoutedModel(device="meta")

    specs = get_glm52_ep_plan().apply(model, _fake_ep_fsdp_mesh(ep_size=16), already_local=True)

    experts = model.experts
    for name in experts._ep_already_local_parameter_names:
        parameter = getattr(experts, name)
        assert parameter.shape[0] == 16
        assert isinstance(specs[f"model.layers.3.mlp.experts.{name}"].placement, Shard)
    for name in experts._ep_force_shard_parameter_names:
        parameter = getattr(experts, name)
        info = specs[f"model.layers.3.mlp.experts.{name}"]
        assert parameter.shape[0] == 16
        assert isinstance(info.placement, Shard)
        assert info.gradient_reduction is GradientReductionDomain.NONE
    for name in experts._ep_gradient_reduction_by_parameter:
        parameter = getattr(experts, name)
        info = specs[f"model.layers.3.mlp.experts.{name}"]
        assert parameter.shape[0] == 1
        assert isinstance(info.placement, Replicate)
        assert info.gradient_reduction is GradientReductionDomain.EP_SUM

    model.to_empty(device=torch.device("cpu"))
    assert all(getattr(experts, name).shape[0] == 16 for name in experts._ep_already_local_parameter_names)
    assert all(getattr(experts, name).shape[0] == 16 for name in experts._ep_force_shard_parameter_names)
    assert all(hasattr(getattr(experts, name), "spec_info") for name in experts.logical_factor_names)


def test_glm52_exact_real_already_local_plan_still_shards_global_factor_banks(monkeypatch):
    model = _ExactRoutedModel(device="cpu")
    with torch.no_grad():
        model.experts.gate_proj_lora_B[:, 0, 0].copy_(torch.arange(256, dtype=torch.float32))

    class _LocalShard:
        def __init__(self, tensor):
            self.tensor = tensor

        def redistribute(self, **_kwargs):
            return self

        def to_local(self):
            return self.tensor[112:128].clone()

    class _FakeDTensor:
        @staticmethod
        def from_local(*, local_tensor, **_kwargs):
            return _LocalShard(local_tensor)

    monkeypatch.setattr(parallel_plan_module, "DTensor", _FakeDTensor)

    specs = get_glm52_ep_plan().apply(model, _fake_ep_fsdp_mesh(ep_size=16), already_local=True)

    experts = model.experts
    assert torch.equal(experts.gate_proj_lora_B[:, 0, 0], torch.arange(112, 128, dtype=torch.float32))
    assert all(getattr(experts, name).shape[0] == 16 for name in experts._ep_force_shard_parameter_names)
    assert all(getattr(experts, name).shape[0] == 16 for name in experts._ep_already_local_parameter_names)
    for name in experts._ep_force_shard_parameter_names:
        info = specs[f"model.layers.3.mlp.experts.{name}"]
        assert isinstance(info.placement, Shard)
        assert info.gradient_reduction is GradientReductionDomain.NONE


@pytest.mark.parametrize(
    ("parameter_name", "error_match"),
    (
        ("gate_up_packed_weight_f32", "already-local parameter"),
        ("gate_proj_lora_B", "force-shard parameter"),
    ),
)
def test_glm52_exact_explicit_ep_dispositions_reject_malformed_singletons(
    parameter_name: str,
    error_match: str,
) -> None:
    model = _ExactRoutedModel(device="cpu")
    original = getattr(model.experts, parameter_name)
    malformed_shape = (1, *original.shape[1:])
    setattr(
        model.experts,
        parameter_name,
        nn.Parameter(torch.empty(malformed_shape, dtype=original.dtype), requires_grad=original.requires_grad),
    )

    with pytest.raises(ValueError, match=error_match):
        get_glm52_ep_plan().apply(model, _fake_ep_fsdp_mesh(ep_size=16), already_local=True)


class _FakePreslicedExperts(nn.Module):
    """Expert-like owner carrying the checkpoint load's presliced record."""

    def __init__(self, rows: int, hidden: int, inter: int, *, global_rows: int, load_ep: int, meta: bool = False):
        super().__init__()
        device = "meta" if meta else "cpu"
        self.gate_up_proj = nn.Parameter(
            torch.empty(rows, hidden, 2 * inter, device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        if not meta:
            with torch.no_grad():
                self.gate_up_proj.copy_(
                    torch.arange(self.gate_up_proj.numel(), dtype=torch.float32).reshape(self.gate_up_proj.shape)
                )
        self._xorl_ep_load_presliced = {"gate_up_proj": (global_rows, load_ep)}


class _FakePreslicedModel(nn.Module):
    def __init__(self, experts: nn.Module):
        super().__init__()
        self.experts = experts


def test_load_presliced_params_are_verified_noops_and_double_slice_fails_closed():
    """Enforce the single-EP-slicing-site contract.

    A parameter the checkpoint load already sliced to EP-local shape
    (recorded on its owner by ``_shrink_expert_params_for_ep``) must be a
    VERIFIED no-op here — annotated, never sliced again — and every
    inconsistent combination must fail closed instead of falling through
    to the real-slice branch."""

    plan = ParallelPlan(ep_plan={"experts.gate_up_proj": Shard(0)})
    H, I = 8, 8

    # Positive: real tensor at global//ep rows -> annotate, bytes untouched.
    experts = _FakePreslicedExperts(4, H, I, global_rows=16, load_ep=4)
    model = _FakePreslicedModel(experts)
    before = experts.gate_up_proj.detach().clone()
    specs = plan.apply(model, _fake_ep_fsdp_mesh(ep_size=4), already_local=False)
    assert tuple(model.experts.gate_up_proj.shape) == (4, H, 2 * I), "presliced param must NOT be sliced again"
    assert torch.equal(model.experts.gate_up_proj.detach(), before), "presliced bytes must be untouched"
    info = specs["experts.gate_up_proj"]
    assert isinstance(info.placement, Shard) and info.placement.dim == 0

    # Positive corner: presliced down to ONE row must still annotate as a
    # Shard (the singleton-Replicate shared-LoRA branch must not steal it).
    experts = _FakePreslicedExperts(1, H, I, global_rows=4, load_ep=4)
    specs = plan.apply(_FakePreslicedModel(experts), _fake_ep_fsdp_mesh(ep_size=4), already_local=False)
    assert isinstance(specs["experts.gate_up_proj"].placement, Shard)
    assert tuple(experts.gate_up_proj.shape) == (1, H, 2 * I)

    # Fail closed: the shape neither matches the presliced record (a second
    # slice or a lying record) ...
    experts = _FakePreslicedExperts(16, H, I, global_rows=16, load_ep=4)
    with pytest.raises(ValueError, match="would apply EP twice"):
        plan.apply(_FakePreslicedModel(experts), _fake_ep_fsdp_mesh(ep_size=4), already_local=False)

    # ... nor may the load-time and wrap-time EP worlds differ ...
    experts = _FakePreslicedExperts(2, H, I, global_rows=16, load_ep=8)
    with pytest.raises(ValueError, match="EP worlds must be identical"):
        plan.apply(_FakePreslicedModel(experts), _fake_ep_fsdp_mesh(ep_size=4), already_local=False)

    # ... nor may a presliced record cover a still-meta tensor ...
    experts = _FakePreslicedExperts(4, H, I, global_rows=16, load_ep=4, meta=True)
    with pytest.raises(ValueError, match="never materialized"):
        plan.apply(_FakePreslicedModel(experts), _fake_ep_fsdp_mesh(ep_size=4), already_local=False)

    # ... nor may force-shard contradict a presliced record (owner declares
    # global-size storage so the force-shard branch falls through here).
    experts = _FakePreslicedExperts(16, H, I, global_rows=16, load_ep=4)
    experts.num_experts = 16
    experts.num_local_experts = 4
    experts.gate_up_proj._xorl_ep_force_shard = True
    with pytest.raises(ValueError, match="slice it a second time"):
        plan.apply(_FakePreslicedModel(experts), _fake_ep_fsdp_mesh(ep_size=4), already_local=False)
