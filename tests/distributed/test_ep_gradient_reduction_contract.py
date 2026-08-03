"""Distributed tests for the explicit hybrid-LoRA gradient contracts."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import Replicate

from xorl.distributed.ep_gradients import synchronize_ep_replicated_gradients
from xorl.distributed.gradient_reduction import GradientReductionDomain
from xorl.distributed.torch_parallelize import _build_ep_param_groups
from xorl.models.layers.moe.backend import ep_lora_gradient_reduction_domain


pytestmark = [pytest.mark.cpu]


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("eager", GradientReductionDomain.EP_SUM),
        ("native", GradientReductionDomain.EP_SUM),
        ("triton", GradientReductionDomain.ALREADY_REDUCED),
        ("triton_w4a4", GradientReductionDomain.ALREADY_REDUCED),
        ("quack", GradientReductionDomain.ALREADY_REDUCED),
    ],
)
def test_supported_backend_gradient_contract_table(backend, expected):
    assert ep_lora_gradient_reduction_domain(backend) is expected


def test_unknown_backend_fails_closed():
    with pytest.raises(ValueError, match="Unsupported MoE backend"):
        ep_lora_gradient_reduction_domain("new_backend")


def test_unknown_metadata_domain_fails_closed():
    model = nn.Module()
    shared = nn.Module()
    shared._skip_fsdp = True
    shared.weight = nn.Parameter(torch.zeros(1))
    model.shared = shared
    model._fqn2spec_info = {"shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction="typo")}

    with pytest.raises(ValueError, match="Unknown gradient reduction domain"):
        _build_ep_param_groups(model)


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
    parallel_state = MagicMock()
    parallel_state.ep_enabled = True
    parallel_state.ep_group = dist.group.WORLD

    backends = ("eager", "native", "triton", "triton_w4a4", "quack")
    for backend in backends:
        domain = ep_lora_gradient_reduction_domain(backend)
        model = nn.Module()
        shared = nn.Module()
        shared._skip_fsdp = True
        shared.weight = nn.Parameter(torch.zeros(1))
        model.shared = shared
        model._fqn2spec_info = {"shared.weight": SimpleNamespace(placement=Replicate(), gradient_reduction=domain)}
        _build_ep_param_groups(model)

        shared.weight.grad = torch.tensor([rank + 1.0])
        all_reduce_calls = []
        original_all_reduce = dist.all_reduce

        def _counted_all_reduce(*args, **kwargs):
            all_reduce_calls.append(1)
            return original_all_reduce(*args, **kwargs)

        with patch("xorl.distributed.parallel_state.get_parallel_state", return_value=parallel_state):
            with patch.object(dist, "all_reduce", side_effect=_counted_all_reduce):
                if domain is GradientReductionDomain.ALREADY_REDUCED:
                    # Model the custom Triton/Quack backward contract: the
                    # backend performs the one logical EP sum before the
                    # optimizer-boundary reducer is consulted.
                    dist.all_reduce(shared.weight.grad, group=dist.group.WORLD)
                stats = synchronize_ep_replicated_gradients(model)

        assert shared.weight.grad.item() == pytest.approx(3.0)
        assert len(all_reduce_calls) == 1
        assert stats.participating_parameter_count == (1 if domain is GradientReductionDomain.EP_SUM else 0)
        assert model._ep_replicated_gradient_sync_enabled is (domain is GradientReductionDomain.EP_SUM)

    dist.destroy_process_group()


if os.environ.get("XORL_EP_GRADIENT_CONTRACT_WORKER") == "1":
    _run_backend_contract_worker()
