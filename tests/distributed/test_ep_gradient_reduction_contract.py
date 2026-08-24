"""Distributed tests for the explicit hybrid-LoRA gradient contracts."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import Replicate

from xorl.distributed.ep_gradients import synchronize_replicated_gradient_parameters
from xorl.distributed.gradient_reduction import GradientReductionDomain
from xorl.distributed.torch_parallelize import _build_ep_param_groups, refresh_ep_param_groups
from xorl.models.layers.moe.backend import ep_lora_gradient_reduction_domain


pytestmark = [pytest.mark.cpu]


def test_gradient_reduction_domain_admission_policy():
    for backend in ("eager", "native", "triton", "triton_w4a4", "quack"):
        assert ep_lora_gradient_reduction_domain(backend) is GradientReductionDomain.EP_SUM
    with pytest.raises(ValueError, match="Unsupported MoE backend"):
        ep_lora_gradient_reduction_domain("new_backend")

    model = nn.Module()
    shared = nn.Module()
    shared._skip_fsdp = True
    shared.weight = nn.Parameter(torch.zeros(1))
    model.shared = shared
    model._fqn2spec_info = {"shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="typo")}

    with pytest.raises(ValueError, match="Unknown gradient reduction domain"):
        _build_ep_param_groups(model)


def test_refresh_ep_param_groups_rebinds_replaced_parameter_identity():
    model = nn.Module()
    shared = nn.Module()
    shared._skip_fsdp = True
    original = nn.Parameter(torch.empty(1, device="meta"))
    shared.weight = original
    model.shared = shared
    model._fqn2spec_info = {
        "shared.weight": SimpleNamespace(
            placement=Replicate(),
            gradient_reduction=GradientReductionDomain.EP_SUM,
        )
    }

    _build_ep_param_groups(model)
    assert model._ep_param_groups["ep_replicated_gradient_sync"] == [original]

    replacement = nn.Parameter(torch.ones(1))
    shared.weight = replacement
    refresh_ep_param_groups(model)

    assert model._ep_param_groups["ep_replicated_gradient_sync"] == [replacement]
    assert all(parameter is not original for parameter in model._ep_param_groups["ep_replicated_gradient_sync"])


def test_real_two_rank_backend_contracts():
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_EP_GRADIENT_CONTRACT_WORKER": "1"},
    )
    result.assert_success("two-rank hybrid-LoRA backend reduction contracts")


def _run_backend_contract_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    # Exercise the actual EP LoRA autograd backward with a CPU grouped-GEMM
    # reference implementation. The old test manually performed Triton's
    # supposed all-reduce, which could pass while the real backward was wrong.
    from xorl.ops.moe.lora import make_ep_lora_compute

    def group_gemm_same_nk(*, a, b, cumsum_M, max_M, transpose_a=False, transpose_b=False):
        outputs = []
        for expert in range(b.shape[0]):
            start = int(cumsum_M[expert].item())
            end = int(cumsum_M[expert + 1].item())
            left = a[start:end].transpose(0, 1) if transpose_a else a[start:end]
            right = b[expert].transpose(0, 1) if transpose_b else b[expert]
            outputs.append(left @ right)
        return torch.cat(outputs, dim=0) if outputs else a.new_empty((0, b.shape[-1]))

    def group_gemm_same_mn(*, a, b, c, cumsum_K, max_K, transpose_a=False, transpose_b=False):
        for expert in range(c.shape[0]):
            start = int(cumsum_K[expert].item())
            end = int(cumsum_K[expert + 1].item())
            left = a[start:end].transpose(0, 1) if transpose_a else a[start:end]
            right = b[start:end].transpose(0, 1) if transpose_b else b[start:end]
            c[expert].copy_(left @ right)
        return c

    ep_compute = make_ep_lora_compute(group_gemm_same_nk, group_gemm_same_mn)
    num_experts, hidden, intermediate, rank_dim = 2, 4, 6, 2
    tokens = torch.randn(4, hidden) + rank
    cumsum = torch.tensor([0, 2, 4], dtype=torch.int64)
    gate = torch.randn(num_experts, hidden, intermediate)
    up = torch.randn_like(gate)
    down = torch.randn(num_experts, intermediate, hidden)
    shared_gate_a = torch.randn(1, hidden, rank_dim, requires_grad=True)
    gate_b = torch.randn(num_experts, rank_dim, intermediate, requires_grad=True)
    shared_up_a = torch.randn(1, hidden, rank_dim, requires_grad=True)
    up_b = torch.randn(num_experts, rank_dim, intermediate, requires_grad=True)
    down_a = torch.randn(num_experts, intermediate, rank_dim, requires_grad=True)
    shared_down_b = torch.randn(1, rank_dim, hidden, requires_grad=True)

    all_reduce_calls = []
    original_all_reduce = dist.all_reduce

    def _counted_all_reduce(*args, **kwargs):
        all_reduce_calls.append(1)
        return original_all_reduce(*args, **kwargs)

    with patch.object(dist, "all_reduce", side_effect=_counted_all_reduce):
        output = ep_compute.apply(
            tokens,
            cumsum,
            gate,
            up,
            down,
            shared_gate_a,
            gate_b,
            shared_up_a,
            up_b,
            down_a,
            shared_down_b,
            1.0,
        )
        output.square().sum().backward()

    assert not all_reduce_calls, "backend backward must not perform EP collectives"
    local_gradient = shared_down_b.grad.detach().clone()
    adapter_slot = nn.Parameter(local_gradient.clone())
    adapter_slot.grad = local_gradient.clone()
    stats = synchronize_replicated_gradient_parameters([adapter_slot], ep_group=dist.group.WORLD)
    assert adapter_slot.grad is not None
    expected = local_gradient.clone()
    original_all_reduce(expected, group=dist.group.WORLD)
    assert torch.allclose(adapter_slot.grad, expected)
    assert stats.participating_parameter_count == 1
    assert stats.bucket_count == 1

    # The explicit metadata contract must agree for every supported backend.
    for backend in ("eager", "native", "triton", "triton_w4a4", "quack"):
        assert ep_lora_gradient_reduction_domain(backend) is GradientReductionDomain.EP_SUM

    dist.destroy_process_group()


if os.environ.get("XORL_EP_GRADIENT_CONTRACT_WORKER") == "1":
    _run_backend_contract_worker()
