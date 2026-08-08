"""DCP-native base-state projection for the exact GLM-5.2 active-LoRA modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import DTensor, Shard

from .exact_dense_mlp import Glm52ExactTP1DenseMLP
from .exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear


def _model_parts(model: nn.Module | Iterable[nn.Module]) -> tuple[nn.Module, ...]:
    return tuple(model) if isinstance(model, (list, tuple)) else (model,)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def _local_tensor(tensor: DTensor) -> torch.Tensor:
    local = tensor.to_local()
    wait = getattr(local, "wait", None)
    return wait() if callable(wait) else local


def _empty_sharded_half(tensor: DTensor, *, name: str) -> DTensor:
    if tensor.device_mesh.ndim != 1 or len(tensor.placements) != 1:
        raise RuntimeError(f"Exact GLM base DCP projection requires a one-dimensional FSDP mesh for {name}")
    placement = tensor.placements[0]
    if not isinstance(placement, Shard) or placement.dim != 0:
        raise RuntimeError(f"Exact GLM base DCP projection requires Shard(0) for {name}, got {tensor.placements}")

    global_shape = tuple(tensor.shape)
    if not global_shape or global_shape[0] % 2:
        raise ValueError(f"Exact GLM fused DCP target {name} must have an even leading dimension, got {global_shape}")
    half_shape = (global_shape[0] // 2, *global_shape[1:])
    mesh_size = tensor.device_mesh.size()
    if half_shape[0] % mesh_size:
        raise ValueError(
            f"Exact GLM DCP source {name} leading dimension {half_shape[0]} must divide FSDP size {mesh_size}"
        )

    target_local = _local_tensor(tensor)
    local_shape = (half_shape[0] // mesh_size, *half_shape[1:])
    local = torch.empty(local_shape, dtype=tensor.dtype, device=target_local.device)
    return DTensor.from_local(
        local,
        device_mesh=tensor.device_mesh,
        placements=tensor.placements,
        shape=torch.Size(half_shape),
        stride=_contiguous_stride(half_shape),
        run_check=False,
    )


def _fuse_sharded_halves(gate: DTensor, up: DTensor, target: DTensor, *, name: str) -> None:
    if gate.device_mesh != target.device_mesh or up.device_mesh != target.device_mesh:
        raise RuntimeError(f"Exact GLM DCP source and target meshes differ for {name}")
    if gate.placements != target.placements or up.placements != target.placements:
        raise RuntimeError(f"Exact GLM DCP source and target placements differ for {name}")

    gate_local = _local_tensor(gate).contiguous()
    up_local = _local_tensor(up).contiguous()
    target_local = _local_tensor(target)
    if gate_local.shape != up_local.shape or target_local.numel() != 2 * gate_local.numel():
        raise RuntimeError(
            f"Exact GLM DCP local dense shapes do not compose for {name}: "
            f"gate={tuple(gate_local.shape)} up={tuple(up_local.shape)} target={tuple(target_local.shape)}"
        )

    mesh = target.device_mesh
    mesh_size = mesh.size()
    if mesh_size == 1:
        target_local.copy_(torch.cat((gate_local, up_local), dim=0))
        return
    if mesh_size % 2:
        raise RuntimeError(f"Exact GLM fused DCP load requires an even FSDP size, got {mesh_size}")
    if not dist.is_initialized():
        raise RuntimeError("Exact GLM fused DCP load requires torch.distributed for a sharded target")

    mesh_rank = mesh.get_local_rank()
    chunk_numel = gate_local.numel()
    gate_destination = mesh_rank // 2
    up_destination = mesh_size // 2 + mesh_rank // 2
    input_splits = [0] * mesh_size
    input_splits[gate_destination] = chunk_numel
    input_splits[up_destination] = chunk_numel

    source_pair = 2 * (mesh_rank if mesh_rank < mesh_size // 2 else mesh_rank - mesh_size // 2)
    output_splits = [0] * mesh_size
    output_splits[source_pair] = chunk_numel
    output_splits[source_pair + 1] = chunk_numel

    send = torch.cat((gate_local.reshape(-1), up_local.reshape(-1)))
    dist.all_to_all_single(
        target_local.reshape(-1),
        send,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=mesh.get_group(),
    )


class Glm52ExactBaseDcpLoadProjection:
    """Present the official base-DCP keys while retaining exact runtime state.

    The official DCP stores dense gate/up projections separately and stores
    block-FP8 scales as FP32 parameters. The exact runtime fuses the three dense
    gate/up pairs and keeps ordinary QLoRA scales as byte-preserving buffers.
    This adapter is load-only; normal exact-model saves keep their native state.
    """

    def __init__(self, model: nn.Module | Iterable[nn.Module]) -> None:
        dense_modules: dict[str, Glm52ExactTP1DenseMLP] = {}
        scale_modules: dict[str, Glm52ExactTP1BlockFP8QLoRALinear] = {}
        for part in _model_parts(model):
            for name, module in part.named_modules():
                if isinstance(module, Glm52ExactTP1DenseMLP):
                    dense_modules[name] = module
                if isinstance(module, Glm52ExactTP1BlockFP8QLoRALinear):
                    scale_modules[name] = module
        self._dense_modules = dense_modules
        self._scale_modules = scale_modules
        self.dense_roots = tuple(sorted(dense_modules))
        self.scale_roots = tuple(sorted(scale_modules))

    @property
    def enabled(self) -> bool:
        return bool(self.dense_roots or self.scale_roots)

    def project_key_contract(
        self,
        parameter_keys: Iterable[str],
        buffer_keys: Iterable[str],
    ) -> tuple[list[str], list[str]]:
        parameters = set(parameter_keys)
        buffers = set(buffer_keys)

        for root in self.dense_roots:
            parameters.remove(f"{root}.packed_weight_f32")
            parameters.remove(f"{root}.weight_scale_inv")
            for projection in ("gate_proj", "up_proj"):
                source = f"{root}.{projection}"
                parameters.add(f"{source}.packed_weight_f32")
                parameters.add(f"{source}.weight_scale_inv")

        for root in self.scale_roots:
            buffers.remove(f"{root}.weight_block_scales")
            parameters.add(f"{root}.weight_scale_inv")

        return sorted(parameters), sorted(buffers)

    def project_state(self, model_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = dict(model_state)

        for root in self.dense_roots:
            for suffix in ("packed_weight_f32", "weight_scale_inv"):
                target_name = f"{root}.{suffix}"
                target = projected.pop(target_name)
                if isinstance(target, DTensor):
                    gate = _empty_sharded_half(target, name=target_name)
                    up = _empty_sharded_half(target, name=target_name)
                else:
                    if target.shape[0] % 2:
                        raise ValueError(
                            f"Exact GLM fused DCP target {target_name} has odd shape {tuple(target.shape)}"
                        )
                    half = target.shape[0] // 2
                    gate = target.narrow(0, 0, half)
                    up = target.narrow(0, half, half)
                projected[f"{root}.gate_proj.{suffix}"] = gate
                projected[f"{root}.up_proj.{suffix}"] = up

        for root in self.scale_roots:
            target_name = f"{root}.weight_block_scales"
            source_name = f"{root}.weight_scale_inv"
            scale_bytes = projected.pop(target_name)
            if isinstance(scale_bytes, DTensor):
                raise RuntimeError(f"Exact GLM QLoRA scale buffer {target_name} must remain replicated")
            if scale_bytes.dtype is not torch.uint8 or not scale_bytes.is_contiguous():
                raise TypeError(
                    f"Exact GLM QLoRA scale buffer {target_name} must be contiguous uint8, "
                    f"got {scale_bytes.dtype} contiguous={scale_bytes.is_contiguous()}"
                )
            if scale_bytes.shape[-1] % 4:
                raise ValueError(f"Exact GLM QLoRA scale buffer {target_name} cannot be viewed as FP32")
            projected[source_name] = scale_bytes.view(torch.float32)

        return projected

    @torch.no_grad()
    def restore_state(
        self,
        projected_state: Mapping[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for root in self.dense_roots:
            for suffix in ("packed_weight_f32", "weight_scale_inv"):
                target_name = f"{root}.{suffix}"
                gate = projected_state[f"{root}.gate_proj.{suffix}"]
                up = projected_state[f"{root}.up_proj.{suffix}"]
                target = model_state[target_name]
                if isinstance(target, DTensor):
                    if not isinstance(gate, DTensor) or not isinstance(up, DTensor):
                        raise TypeError(f"Exact GLM sharded DCP sources for {target_name} must be DTensors")
                    _fuse_sharded_halves(gate, up, target, name=target_name)
                else:
                    if isinstance(gate, DTensor) or isinstance(up, DTensor):
                        raise TypeError(f"Exact GLM unsharded DCP sources for {target_name} must be tensors")
                    half = target.shape[0] // 2
                    target.narrow(0, 0, half).copy_(gate)
                    target.narrow(0, half, half).copy_(up)

        for root in self.scale_roots:
            target_name = f"{root}.weight_block_scales"
            source_name = f"{root}.weight_scale_inv"
            target = model_state[target_name]
            source = projected_state[source_name]
            if isinstance(target, DTensor) or isinstance(source, DTensor):
                raise TypeError(f"Exact GLM QLoRA DCP scale alias {source_name} must remain replicated")
            target.view(torch.float32).copy_(source)

        for module in self._dense_modules.values():
            module._exact_gate_up_base_loaded = True
        for module in self._scale_modules.values():
            module._inline_loaded = True

        return model_state


__all__ = ["Glm52ExactBaseDcpLoadProjection"]
