"""WORLD16 CPU/gloo ownership regression for exact GLM-5.2 routed experts.

This intentionally exercises the production composition instead of imitating its
types or layouts: EP16 localizes the logical routed bank, composable FSDP creates
the dynamic expert-module class, layout discovery exchanges all sixteen owners,
and factor export gathers the owner-local banks back to their logical shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard

from xorl.distributed.gradient_reduction import GradientReductionDomain
from xorl.lora.utils import get_lora_state_dict
from xorl.models.transformers.glm5.exact_routed_experts_qlora import (
    GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION,
    Glm52ExactEP16BlockFP8QLoRARoutedExperts,
)
from xorl.models.transformers.glm5.parallelize import get_ep_plan
from xorl.optim import optimizer as optimizer_module
from xorl.server.runner import model_runner as model_runner_module
from xorl.server.runner.adapters.gradient_ownership import (
    GradientRepresentation,
    ProducerFamily,
    ReductionAxis,
    TopologyFamily,
)
from xorl.server.runner.adapters.manager import LoRAAdapterManager
from xorl.server.runner.model_runner import ModelRunner


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script  # noqa: E402


pytestmark = [pytest.mark.cpu, pytest.mark.distributed]
_WORLD = 16
_LOCAL_EXPERTS = 16
_GLOBAL_EXPERTS = 256
_SHARED_FACTORS = {"gate_proj_lora_A", "up_proj_lora_A", "down_proj_lora_B"}
_OWNER_FACTORS = {"gate_proj_lora_B", "up_proj_lora_B", "down_proj_lora_A"}


class _ExactRoutedModel(nn.Module):
    def __init__(self, *, ep_rank: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Module() for _ in range(4))
        self.model.layers[3].mlp = nn.Module()
        self.model.layers[3].mlp.experts = Glm52ExactEP16BlockFP8QLoRARoutedExperts(
            128,
            128,
            ep_rank=ep_rank,
            device="meta",
        )

    @property
    def experts(self) -> Glm52ExactEP16BlockFP8QLoRARoutedExperts:
        return self.model.layers[3].mlp.experts

    @staticmethod
    def get_parallel_plan():
        return get_ep_plan()


def _run_world16_exact_routed_ownership() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        assert world == _WORLD
        mesh = init_device_mesh(
            "cpu",
            (_WORLD, 1),
            mesh_dim_names=("ep", "ep_fsdp"),
        )
        ep_group = mesh["ep"].get_group()
        ep_fsdp_mesh = mesh["ep_fsdp"]

        model = _ExactRoutedModel(ep_rank=rank)
        specs = model.get_parallel_plan().apply(model, mesh, already_local=True)
        model._fqn2spec_info = specs
        experts = model.experts

        for name in experts._ep_already_local_parameter_names | experts._ep_force_shard_parameter_names:
            assert getattr(experts, name).shape[0] == _LOCAL_EXPERTS
        for name in _SHARED_FACTORS:
            assert getattr(experts, name).shape[0] == 1

        fully_shard(experts, mesh=ep_fsdp_mesh, shard_placement_fn=lambda _parameter: Shard(1))
        fully_shard(model, mesh=ep_fsdp_mesh)
        model.to_empty(device=torch.device("cpu"))

        assert isinstance(experts, Glm52ExactEP16BlockFP8QLoRARoutedExperts)
        assert isinstance(experts, FSDPModule)
        assert type(experts) is not Glm52ExactEP16BlockFP8QLoRARoutedExperts
        assert type(experts).__qualname__.startswith("FSDP")
        assert all(isinstance(parameter, DTensor) for parameter in experts.parameters())
        assert model._fqn2spec_info is specs

        with torch.no_grad():
            for name in experts.logical_factor_names:
                local = getattr(experts, name).to_local()
                local.fill_(1.0 if name in _SHARED_FACTORS else float(rank + 1))

        exported = get_lora_state_dict(model)
        expected_export_names = {f"model.layers.3.mlp.experts.{name}" for name in experts.logical_factor_names}
        assert set(exported) == expected_export_names
        for name in _SHARED_FACTORS:
            tensor = exported[f"model.layers.3.mlp.experts.{name}"]
            assert tensor.shape[0] == 1
            assert torch.count_nonzero(tensor != 1.0) == 0
        for name in _OWNER_FACTORS:
            tensor = exported[f"model.layers.3.mlp.experts.{name}"]
            assert tensor.shape[0] == _GLOBAL_EXPERTS
            for owner in range(_WORLD):
                owner_slice = tensor[owner * _LOCAL_EXPERTS : (owner + 1) * _LOCAL_EXPERTS]
                assert torch.count_nonzero(owner_slice != float(owner + 1)) == 0

        parallel_state = SimpleNamespace(
            pp_size=1,
            tp_size=1,
            dp_mode="fsdp2",
            sp_grad_sync_group=None,
            lm_head_tp_replica_group=None,
            lm_head_tp_group=None,
            ep_enabled=True,
            ep_size=_WORLD,
            ep_rank=rank,
            ep_group=ep_group,
            ep_fsdp_device_mesh=mesh,
        )
        import xorl.distributed.parallel_state as parallel_state_module  # noqa: PLC0415

        parallel_state_module.get_parallel_state = lambda: parallel_state
        optimizer_module.get_parallel_state = lambda: parallel_state
        model_runner_module.get_parallel_state = lambda: parallel_state

        with TemporaryDirectory() as checkpoint_dir:
            manager = LoRAAdapterManager(
                model,
                torch.device("cpu"),
                checkpoint_dir=checkpoint_dir,
                auto_save_on_eviction=False,
                optimizer_fused=False,
                lora_config={"moe_hybrid_shared_lora": True},
            )
            group_memberships = manager.register_adapter(
                "policy",
                lr=0.1,
                local_group_memberships={"expert_parallel_replica": tuple(range(_WORLD))},
            )
            runner = ModelRunner.__new__(ModelRunner)
            runner.rank = rank
            runner.model = model
            runner._adapter_manager = manager
            runner._compile_registered_adapter_gradient_ownership(
                "policy",
                group_memberships=group_memberships,
            )

            state = manager.get_adapter_state("policy")
            plan = state.gradient_ownership_plan
            assert plan is not None
            by_name = {item.fqn: item for item in plan.parameters}
            assert set(by_name) == expected_export_names

            for full_name, item in by_name.items():
                local_name = full_name.rpartition(".")[2]
                layout = state.tensor_layouts[full_name]
                guard = dict(item.config_guard_fields)
                assert guard["expert_exact_contract"] == GLM52_EXACT_EP16_ROUTED_QLORA_CONTRACT_VERSION
                assert guard["expert_requires_managed_fsdp"] is True
                assert guard["expert_factor_layout"] == "gkn_gate_up_down"
                assert item.producer is ProducerFamily.MODULE_MANAGED
                assert item.managed_fsdp_shard is True
                if local_name in _SHARED_FACTORS:
                    assert layout.logical_shape[0] == layout.local_logical_shape[0] == 1
                    assert layout.replica_count == _WORLD
                    assert layout.gradient_reduction is GradientReductionDomain.EP_SUM
                    assert layout.needs_ep_gradient_sync
                    assert item.topology is TopologyFamily.EP_REPLICATED_SHARED
                    assert item.representation is GradientRepresentation.REPLICATED_LOCAL_CONTRIBUTION
                    assert [domain.axis for domain in item.pending_domains].count(
                        ReductionAxis.EXPERT_PARALLEL_REPLICA
                    ) == 1
                else:
                    assert layout.logical_shape[0] == _GLOBAL_EXPERTS
                    assert layout.local_logical_shape[0] == _LOCAL_EXPERTS
                    assert layout.local_logical_offset[0] == rank * _LOCAL_EXPERTS
                    assert layout.replica_count == 1
                    assert layout.gradient_reduction is GradientReductionDomain.NONE
                    assert not layout.needs_ep_gradient_sync
                    assert item.topology is TopologyFamily.OWNER_SHARDED
                    assert item.representation is GradientRepresentation.OWNER_LOCAL_CONTRIBUTION
                    assert all(
                        domain.axis is not ReductionAxis.EXPERT_PARALLEL_REPLICA for domain in item.pending_domains
                    )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ != "__main__":

    def test_world16_exact_routed_composable_fsdp_ownership_and_export() -> None:
        result = run_distributed_script(__file__, num_gpus=_WORLD, timeout=300)
        result.assert_success("WORLD16 exact routed ownership and export must survive composable FSDP")


if __name__ == "__main__":
    _run_world16_exact_routed_ownership()
