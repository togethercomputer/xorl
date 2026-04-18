from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from ..utils import logging
from .parallel_state import get_parallel_state
from .utils import check_fqn_match, get_module_from_path, set_module_from_path


logger = logging.get_logger(__name__)


@dataclass
class SpecInfo:
    ep_fsdp_mesh: DeviceMesh
    placement: Union[Shard, Replicate]
    fqn: str

    @property
    def ep_mesh(self):
        if self.ep_fsdp_mesh is not None:
            return self.ep_fsdp_mesh["ep"]
        else:
            return None


class ParallelPlan:
    def __init__(self, ep_plan: Dict[str, Shard]):
        self.ep_plan = ep_plan
        self.ep_param_suffix = {k.split(".")[-1] for k in ep_plan.keys()}
        self.fsdp_no_shard_module = {".".join(list(ep_plan.keys())[0].split(".")[:-1])}

    def apply(self, model: nn.Module, ep_fsdp_mesh: DeviceMesh, already_local: bool = False):
        """Apply EP sharding to model parameters.

        Args:
            model: The model whose expert parameters will be sharded.
            ep_fsdp_mesh: DeviceMesh with ``["ep", "ep_fsdp"]`` dimensions.
            already_local: When ``True``, expert parameters are already at
                EP-local shapes (e.g. loaded with EP-aware weight loading
                before FSDP wrapping).  Skip the DTensor redistribute but
                still annotate every parameter with ``spec_info``.
        """
        ep_mesh = ep_fsdp_mesh["ep"]
        # ep_plan
        fqn2spec_info = {}
        if self.ep_plan:
            ep_size = ep_mesh.size(-1)
            ep_replicate = [Replicate() for _ in range(ep_mesh.ndim)]
            for fqn, param in model.named_parameters():
                for fqn_pattern, shard in self.ep_plan.items():
                    if check_fqn_match(fqn_pattern, fqn):
                        # Shared LoRA weights have size=1 on shard dim - replicate instead of shard
                        if param.size(shard.dim) == 1:
                            param.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)
                            fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)
                            logger.debug_rank0(f"EP replicated (shared): {fqn} {list(param.shape)}")
                            break

                        if already_local:
                            # Params already EP-local — just annotate with spec_info
                            param.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                            fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                            logger.debug_rank0(
                                f"EP annotated (already local): {fqn} {list(param.shape)} "
                                f"(dim={shard.dim}, ep_size={ep_size})"
                            )
                            break

                        assert param.size(shard.dim) % ep_size == 0, (
                            f"EP sharding failed for {fqn}: dim {shard.dim} size {param.size(shard.dim)} "
                            f"not divisible by ep_size {ep_size}"
                        )
                        ep_placement = ep_replicate[:-1] + [shard]
                        original_shape = param.shape
                        dtensor = DTensor.from_local(
                            local_tensor=param.data, device_mesh=ep_mesh, placements=ep_replicate
                        )
                        dtensor = dtensor.redistribute(device_mesh=ep_mesh, placements=ep_placement)
                        local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                        local_chunk.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                        set_module_from_path(model, fqn, local_chunk)
                        fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                        logger.debug_rank0(
                            f"EP sharded: {fqn} {list(original_shape)} -> {list(local_chunk.shape)} "
                            f"(dim={shard.dim}, ep_size={ep_size})"
                        )
                        break
                if fqn not in fqn2spec_info:  # not sharded
                    param.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)
                    fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)

        for param in model.parameters():
            assert hasattr(param, "spec_info"), f"Internal Error: {param} is omitted"

        return fqn2spec_info

    def get_fsdp_no_shard_info(self, model: nn.Module):
        if self.fsdp_no_shard_module is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        for fqn, param in model.named_modules():
            for no_shard_pattern in self.fsdp_no_shard_module:
                if check_fqn_match(no_shard_pattern, fqn):
                    fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, "no module in model match `fsdp_no_shard_module`"

        return fsdp_no_shard_states_fqn_to_module

    def update_prefix(self, prefix: str):
        """
        Update ep_plan when model is wrapped.
        """
        self.ep_plan = {prefix + "." + k: v for k, v in self.ep_plan.items()}
        self.ep_param_suffix = {k.split(".")[-1] for k in self.ep_plan.keys()}
        self.fsdp_no_shard_module = {prefix + "." + k for k in self.fsdp_no_shard_module}

    def shard_tensor(self, tensor: "torch.Tensor", full_param_name: str, target_shape: tuple) -> "torch.Tensor":
        """
        Shard tensor for expert parallelism if needed.
        In the future, we may add other tensor slicing in this function to determine TP parameter and its sharding.

        Args:
            tensor: The tensor to potentially shard
            full_param_name: The full parameter name (e.g., "model.layers.0.mlp.experts.gate_proj.weight")
            target_shape: The expected shape of the target parameter

        Returns:
            The original tensor or a sliced version for EP
        """
        if not self._is_expert_parameter(full_param_name):
            return tensor
        return self._slice_expert_tensor_for_ep(tensor, full_param_name, target_shape)

    def _is_expert_parameter(self, parameter_name: str) -> bool:
        """Check if parameter is an expert parameter that needs EP-aware loading based on parallel_plan."""
        if not self.ep_plan:
            return False

        # Check if this parameter matches any pattern in the EP plan
        for fqn_pattern in self.ep_plan.keys():
            if check_fqn_match(fqn_pattern, parameter_name):
                return True
        return False

    def _slice_expert_tensor_for_ep(
        self, tensor: "torch.Tensor", parameter_name: str, target_shape: tuple
    ) -> "torch.Tensor":
        """Slice expert tensor for expert parallelism."""
        try:
            parallel_state = get_parallel_state()

            # Check if we need to slice based on tensor vs target shape mismatch
            if len(tensor.shape) >= 1 and len(target_shape) >= 1:
                tensor_experts = tensor.shape[0]
                target_experts = target_shape[0]

                # If tensor has more experts than target, we need to slice
                if tensor_experts > target_experts and tensor_experts % target_experts == 0:
                    ep_size = tensor_experts // target_experts
                    ep_rank = parallel_state.ep_rank if parallel_state.ep_enabled else 0
                    start_idx = ep_rank * target_experts
                    end_idx = start_idx + target_experts

                    sliced_tensor = tensor[start_idx:end_idx]

                    logger.info_rank0(
                        f"Expert parameter {parameter_name}: sliced {tensor.shape} -> {sliced_tensor.shape} "
                        f"for EP rank {ep_rank}/{ep_size}"
                    )

                    return sliced_tensor

            # No slicing needed
            return tensor

        except Exception as e:
            # Fallback: if anything fails, return original tensor
            logger.warning(f"Failed to slice expert tensor {parameter_name}: {e}")
            return tensor
