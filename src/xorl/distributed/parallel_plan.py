from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from ..utils import logging
from .gradient_reduction import GradientReductionDomain, validate_gradient_reduction_domain
from .parallel_state import get_parallel_state
from .utils import check_fqn_match, get_module_from_path, set_module_from_path


logger = logging.get_logger(__name__)


def _gradient_reduction_domain(model: nn.Module, fqn: str) -> GradientReductionDomain:
    """Read explicit gradient-reduction metadata from the owning module."""

    parameter_domain = getattr(get_module_from_path(model, fqn), "_ep_gradient_reduction_domain", None)
    if parameter_domain is not None:
        try:
            return validate_gradient_reduction_domain(parameter_domain)
        except ValueError as exc:
            raise ValueError(f"Invalid gradient reduction metadata for {fqn}: {exc}") from exc

    owner_path, _, local_name = fqn.rpartition(".")
    owner = get_module_from_path(model, owner_path) if owner_path else model
    per_parameter = getattr(owner, "_ep_gradient_reduction_by_parameter", None)
    if per_parameter is not None and local_name in per_parameter:
        try:
            return validate_gradient_reduction_domain(per_parameter[local_name])
        except ValueError as exc:
            raise ValueError(f"Invalid gradient reduction metadata for {fqn}: {exc}") from exc

    parts = fqn.split(".")[:-1]
    while parts:
        module = get_module_from_path(model, ".".join(parts))
        domain = getattr(module, "_ep_gradient_reduction_domain", None)
        if domain is not None:
            try:
                return validate_gradient_reduction_domain(domain)
            except ValueError as exc:
                raise ValueError(f"Invalid gradient reduction metadata for {fqn}: {exc}") from exc
        parts.pop()
    return GradientReductionDomain.NONE


@dataclass
class SpecInfo:
    ep_fsdp_mesh: DeviceMesh
    placement: Union[Shard, Replicate]
    fqn: str
    gradient_reduction: GradientReductionDomain = GradientReductionDomain.NONE

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

        Dispatch paths per matched expert param:

        - **Load-presliced** (owner module carries
          ``_xorl_ep_load_presliced[param_name]``, recorded by the checkpoint
          load's ``_shrink_expert_params_for_ep``): the load already sliced
          this tensor to its EP-local shape with real bytes — EP application
          is a VERIFIED no-op (annotate ``spec_info`` only), and any
          inconsistency (ep_size mismatch, meta storage, wrong dim size,
          force-shard) fails closed so the tensor can never be sliced twice.
        - **Meta tensor** (``param.is_meta``): replace the meta param with a
          new meta param whose shape is sliced to the EP-local size along
          ``shard.dim``. ``to_empty()`` later allocates only the EP-local
          slice; DCP load consumes the stamped ``spec_info`` to slice the
          saved (full-shape) tensor on the way in. This is the path the
          ``skip_weight_loading=True`` + ``init_device="meta"`` flow needs:
          the legacy ``already_local=True`` annotate-only path leaves the
          meta tensor at its full shape, so ``to_empty()`` materializes
          ``[num_experts, H, I]`` per rank and OOMs at scale.
        - **Real tensor with EP-local shape** (``already_local=True``): the
          caller already did EP-aware weight loading (e.g. xorl's TP-aware
          HF loader), so we just stamp ``spec_info`` and leave the param
          alone.
        - **Real tensor with full shape** (default): ``DTensor.from_local
          → redistribute → to_local`` actually slices ``param.data``.

        Args:
            model: The model whose expert parameters will be sharded.
            ep_fsdp_mesh: DeviceMesh with ``["ep", "ep_fsdp"]`` dimensions.
            already_local: Hint that real-tensor params are already at
                EP-local shapes. Has no effect on meta tensors (they are
                always sliced via the meta path).
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
                        gradient_reduction = _gradient_reduction_domain(model, fqn)
                        owner_path, _, local_name = fqn.rpartition(".")
                        owner = get_module_from_path(model, owner_path) if owner_path else model
                        declared_local = local_name in set(
                            getattr(owner, "_ep_already_local_parameter_names", ()) or ()
                        )
                        force_shard = local_name in set(
                            getattr(owner, "_ep_force_shard_parameter_names", ()) or ()
                        ) or bool(getattr(param, "_xorl_ep_force_shard", False))
                        if declared_local and force_shard:
                            raise ValueError(f"EP disposition for {fqn} cannot be both already-local and force-shard")

                        expected_local_size = getattr(owner, "num_local_experts", None)
                        expected_global_size = getattr(owner, "num_experts", None)
                        if declared_local:
                            if expected_local_size is None or param.size(shard.dim) != int(expected_local_size):
                                raise ValueError(
                                    f"EP already-local parameter {fqn} has dim {shard.dim} size "
                                    f"{param.size(shard.dim)}; expected {expected_local_size}"
                                )
                            param.spec_info = SpecInfo(
                                ep_fsdp_mesh=ep_fsdp_mesh,
                                placement=shard,
                                fqn=fqn,
                                gradient_reduction=gradient_reduction,
                            )
                            fqn2spec_info[fqn] = param.spec_info
                            logger.debug_rank0(f"EP preserved declared local parameter: {fqn} {list(param.shape)}")
                            break

                        if force_shard:
                            if expected_local_size is not None and param.size(shard.dim) == int(expected_local_size):
                                param.spec_info = SpecInfo(
                                    ep_fsdp_mesh=ep_fsdp_mesh,
                                    placement=shard,
                                    fqn=fqn,
                                    gradient_reduction=gradient_reduction,
                                )
                                fqn2spec_info[fqn] = param.spec_info
                                logger.debug_rank0(
                                    f"EP annotated force-shard parameter already local: {fqn} {list(param.shape)}"
                                )
                                break
                            if expected_global_size is None or param.size(shard.dim) != int(expected_global_size):
                                raise ValueError(
                                    f"EP force-shard parameter {fqn} has dim {shard.dim} size "
                                    f"{param.size(shard.dim)}; expected {expected_global_size} or {expected_local_size}"
                                )

                        # Load-presliced parameters: the checkpoint load's EP
                        # pre-shrink + EP-aware filtered loading already put
                        # this tensor at its EP-local shape with REAL bytes
                        # (recorded on the owner module by
                        # _shrink_expert_params_for_ep).  That load is the ONE
                        # EP slicing site for such tensors; here EP application
                        # is a VERIFIED no-op — annotate spec_info, never
                        # slice.  Every mismatch fails closed because falling
                        # through to the real-slice branch below would slice
                        # the tensor a second time
                        # (16 % ep_size == 0 passes the divisibility assert),
                        # leaving one wrong-identity expert row per rank.
                        presliced = (getattr(owner, "_xorl_ep_load_presliced", None) or {}).get(local_name)
                        if presliced is not None:
                            load_global_rows, load_ep_size = (int(value) for value in presliced)
                            if force_shard:
                                raise ValueError(
                                    f"EP parameter {fqn} was already EP-sliced at checkpoint load "
                                    f"({load_global_rows} rows / ep_size={load_ep_size}); force-shard "
                                    "would slice it a second time"
                                )
                            if load_ep_size != ep_size:
                                raise ValueError(
                                    f"EP parameter {fqn} was EP-sliced at checkpoint load for "
                                    f"ep_size={load_ep_size} but EP is being applied with ep_size={ep_size}; "
                                    "the load-time and wrap-time EP worlds must be identical"
                                )
                            if param.is_meta:
                                raise ValueError(
                                    f"EP parameter {fqn} is recorded as load-presliced but is still a meta "
                                    "tensor — the pre-shrunk checkpoint bytes were never materialized; "
                                    "refusing to guess whether the shape was sliced once or twice"
                                )
                            expected_local = load_global_rows // ep_size
                            if param.size(shard.dim) != expected_local:
                                raise ValueError(
                                    f"EP parameter {fqn} was EP-pre-sliced at checkpoint load "
                                    f"({load_global_rows} -> {expected_local} rows along dim {shard.dim} for "
                                    f"ep_size={ep_size}) but arrives at EP application with dim size "
                                    f"{param.size(shard.dim)}; slicing here would apply EP twice — refusing "
                                    "the corrupt construction"
                                )
                            param.spec_info = SpecInfo(
                                ep_fsdp_mesh=ep_fsdp_mesh,
                                placement=shard,
                                fqn=fqn,
                                gradient_reduction=gradient_reduction,
                            )
                            fqn2spec_info[fqn] = param.spec_info
                            logger.debug_rank0(
                                f"EP verified load-presliced parameter: {fqn} {list(param.shape)} "
                                f"(dim={shard.dim}, ep_size={ep_size})"
                            )
                            break

                        # A singleton expert axis is ambiguous: it is a shared
                        # LoRA factor, but with ``ep_size == num_experts`` it
                        # is also the shape of a rank-unique per-expert slice.
                        # Only the owner's declared EP_SUM reduction resolves
                        # the ambiguity — replicating a rank-unique slice would
                        # average its gradient norm across EP and (worse) let
                        # checkpointing treat distinct experts as one tensor.
                        if param.size(shard.dim) == 1:
                            if gradient_reduction is GradientReductionDomain.EP_SUM:
                                param.spec_info = SpecInfo(
                                    ep_fsdp_mesh=ep_fsdp_mesh,
                                    placement=Replicate(),
                                    fqn=fqn,
                                    gradient_reduction=gradient_reduction,
                                )
                                fqn2spec_info[fqn] = param.spec_info
                                logger.debug_rank0(f"EP replicated (shared): {fqn} {list(param.shape)}")
                                break
                            if not already_local or force_shard:
                                raise ValueError(
                                    f"EP parameter {fqn} has a singleton dim-{shard.dim} expert axis but no "
                                    "EP_SUM gradient-reduction declaration. A shared factor must declare "
                                    "EP_SUM (owner._ep_gradient_reduction_by_parameter or "
                                    "owner._ep_gradient_reduction_domain); a rank-unique per-expert slice "
                                    "must arrive already-local (owner._ep_already_local_parameter_names or "
                                    "already_local=True) — refusing to guess between a replica and a local "
                                    "expert slice"
                                )
                            # already_local: fall through to the annotate-only
                            # Shard branch below — a per-expert local slice.

                        # Meta-tensor path: slice the SHAPE (no data), swap the
                        # param, stamp spec_info. Dispatched first so the
                        # ``already_local`` branch below doesn't accidentally
                        # leave a meta param at its full shape.
                        if param.is_meta:
                            assert param.size(shard.dim) % ep_size == 0, (
                                f"EP sharding failed for {fqn}: dim {shard.dim} size {param.size(shard.dim)} "
                                f"not divisible by ep_size {ep_size}"
                            )
                            new_shape = list(param.shape)
                            new_shape[shard.dim] = new_shape[shard.dim] // ep_size
                            local_chunk = torch.nn.Parameter(
                                torch.empty(new_shape, device="meta", dtype=param.dtype),
                                requires_grad=param.requires_grad,
                            )
                            local_chunk.spec_info = SpecInfo(
                                ep_fsdp_mesh=ep_fsdp_mesh,
                                placement=shard,
                                fqn=fqn,
                                gradient_reduction=gradient_reduction,
                            )
                            set_module_from_path(model, fqn, local_chunk)
                            fqn2spec_info[fqn] = local_chunk.spec_info
                            logger.debug_rank0(
                                f"EP meta-sliced: {fqn} {list(param.shape)} -> {list(local_chunk.shape)} "
                                f"(dim={shard.dim}, ep_size={ep_size})"
                            )
                            break

                        if already_local and not force_shard:
                            # Params already EP-local — just annotate with spec_info
                            param.spec_info = SpecInfo(
                                ep_fsdp_mesh=ep_fsdp_mesh,
                                placement=shard,
                                fqn=fqn,
                                gradient_reduction=gradient_reduction,
                            )
                            fqn2spec_info[fqn] = param.spec_info
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
                        local_chunk.spec_info = SpecInfo(
                            ep_fsdp_mesh=ep_fsdp_mesh,
                            placement=shard,
                            fqn=fqn,
                            gradient_reduction=gradient_reduction,
                        )
                        set_module_from_path(model, fqn, local_chunk)
                        fqn2spec_info[fqn] = local_chunk.spec_info
                        logger.debug_rank0(
                            f"EP sharded: {fqn} {list(original_shape)} -> {list(local_chunk.shape)} "
                            f"(dim={shard.dim}, ep_size={ep_size})"
                        )
                        break
                if fqn not in fqn2spec_info:  # not sharded
                    # The fallback must still carry the owner's declared
                    # reduction domain: a shared factor on a model whose
                    # ep_plan lacks LoRA patterns would otherwise lose its
                    # EP_SUM contract and silently skip the optimizer-boundary
                    # sync while its norm is averaged as if synchronized.
                    fallback_reduction = _gradient_reduction_domain(model, fqn)
                    param.spec_info = SpecInfo(
                        ep_fsdp_mesh=ep_fsdp_mesh,
                        placement=Replicate(),
                        fqn=fqn,
                        gradient_reduction=fallback_reduction,
                    )
                    fqn2spec_info[fqn] = param.spec_info

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
