"""Atomic router sidecar for exact GLM-5.2 active-LoRA publication."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor


GLM52_ROUTER_BUNDLE_SCHEMA = "xorl.glm52_router_bundle.v1"
# Keep non-LoRA state outside the adapter root.  SGLang's generic adapter
# loader enumerates root-level ``*.safetensors`` files, so placing routers
# beside ``adapter_model.safetensors`` makes their ``layer.N.weight`` keys look
# like malformed LoRA factors before the dedicated receiver can own them.
GLM52_ROUTER_TENSORS = "xorl_router/xorl_glm52_router.safetensors"
GLM52_ROUTER_MANIFEST = "xorl_glm52_router.json"
_ROUTER_MODULE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


def gather_glm52_router_weights(
    model: object,
    *,
    destination_rank: int | None = 0,
) -> dict[str, torch.Tensor]:
    """Collectively reconstruct every GLM router weight on one destination."""

    state: dict[str, torch.Tensor] = {}
    rank = (
        torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
    )
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Glm5TopkRouter":
            continue
        match = _ROUTER_MODULE.search(name)
        if match is None:
            raise RuntimeError(f"Cannot derive GLM-5.2 layer id from router module {name!r}")
        tensor = module.weight.detach()
        retain_tensor = rank == destination_rank if destination_rank is not None else True
        if isinstance(tensor, DTensor):
            device_mesh = tensor.device_mesh
            retain_tensor = (
                retain_tensor
                if destination_rank is not None
                else all(device_mesh.get_local_rank(mesh_dim) == 0 for mesh_dim in range(device_mesh.ndim))
            )
            tensor = tensor.full_tensor()
        if retain_tensor:
            key = f"layer.{int(match.group(1))}.weight"
            if key in state:
                raise RuntimeError(f"Duplicate GLM-5.2 router sidecar key {key!r}")
            state[key] = tensor.to(device="cpu", dtype=torch.bfloat16).contiguous()
    return state


def _merge_glm52_router_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for state in states:
        for key, tensor in state.items():
            previous = merged.get(key)
            if previous is not None:
                if not torch.equal(previous, tensor):
                    raise RuntimeError(f"Conflicting GLM-5.2 router sidecar values for {key!r}")
                continue
            merged[key] = tensor
    return merged


def gather_glm52_router_weights_across_ranks(
    model: object,
    *,
    destination_rank: int = 0,
) -> dict[str, torch.Tensor]:
    """Reconstruct stage-local routers, then gather all PP stages to one rank."""

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_world_size() == 1:
        return gather_glm52_router_weights(model, destination_rank=destination_rank)

    local_state = gather_glm52_router_weights(model, destination_rank=None)
    rank = torch.distributed.get_rank()
    gathered_states = [None] * torch.distributed.get_world_size() if rank == destination_rank else None
    torch.distributed.gather_object(local_state, gathered_states, dst=destination_rank)
    if rank != destination_rank:
        return {}
    return _merge_glm52_router_states([state for state in gathered_states if state])


def save_glm52_router_bundle(
    directory: str | Path,
    state: dict[str, torch.Tensor],
    *,
    weight_step: int,
    expected_layer_ids: list[int] | None = None,
) -> dict[str, object]:
    """Validate and durably save one complete router sidecar."""

    if not state:
        raise RuntimeError("Refusing to publish an empty GLM-5.2 router sidecar")
    for key, tensor in state.items():
        if not re.fullmatch(r"layer\.\d+\.weight", key):
            raise ValueError(f"Invalid GLM-5.2 router sidecar key {key!r}")
        if tensor.dtype is not torch.bfloat16 or tensor.ndim != 2:
            raise ValueError(f"GLM-5.2 router {key!r} must be a BF16 matrix")
    layer_ids = sorted(int(key.split(".")[1]) for key in state)
    if expected_layer_ids is not None and layer_ids != list(expected_layer_ids):
        raise RuntimeError(
            f"Incomplete GLM-5.2 router sidecar: actual layer ids={layer_ids}, expected={list(expected_layer_ids)}"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path = directory / GLM52_ROUTER_TENSORS
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(state.items())), tensor_path)
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "schema": GLM52_ROUTER_BUNDLE_SCHEMA,
        "tensor_file": GLM52_ROUTER_TENSORS,
        "sha256": digest,
        "router_count": len(state),
        "layer_ids": layer_ids,
        "weight_step": int(weight_step),
    }
    (directory / GLM52_ROUTER_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def mark_adapter_config_with_glm52_router_bundle(directory: str | Path, manifest: dict[str, object]) -> None:
    """Bind the adapter config to its mandatory router sidecar."""

    config_path = Path(directory) / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["_xorl_glm52_router_bundle"] = {
        "schema": manifest["schema"],
        "tensor_file": manifest["tensor_file"],
        "sha256": manifest["sha256"],
        "router_count": manifest["router_count"],
        "layer_ids": manifest["layer_ids"],
        "weight_step": manifest["weight_step"],
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


__all__ = [
    "GLM52_ROUTER_BUNDLE_SCHEMA",
    "GLM52_ROUTER_MANIFEST",
    "GLM52_ROUTER_TENSORS",
    "gather_glm52_router_weights",
    "gather_glm52_router_weights_across_ranks",
    "mark_adapter_config_with_glm52_router_bundle",
    "save_glm52_router_bundle",
]
