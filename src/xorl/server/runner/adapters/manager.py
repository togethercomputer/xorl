"""
LoRA Adapter Manager - Manages multiple LoRA adapters for parallel training runs.

Each model_id has exactly one active adapter. Multiple model_ids can coexist,
enabling different training runs to interleave on the same base model and GPUs.

Design (Compiled Gradient Ownership + Per-Adapter Optimizer):
- Base model stays loaded on GPUs (frozen weights)
- Each adapter has its OWN nn.Parameter objects (separate .grad slots)
- Each adapter has its OWN optimizer instance
- Model parameters receive backward gradients; the compiled ownership plan stages
  them into persistent per-adapter FP32 numerator storage.
- Optimizer gradient slots exist only for the bounded finalization/mutation phase.
"""

import hashlib
import json
import logging
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

from xorl.distributed.ep_gradient_diagnostics import (
    gradient_trace_enabled,
    trace_replicated_gradient_stage,
)
from xorl.lora.target_manifest import load_lora_target_manifest
from xorl.lora.utils import (
    convert_peft_lora_state_dict,
    load_lora_checkpoint_state_dict,
)
from xorl.optim import build_optimizer
from xorl.server.runner.adapters.gradient_finalizer import (
    AdapterGradientCollectiveFailure,
    AdapterGradientMutationFailure,
    AdapterGradientTransportStats,
    logical_l2_norm,
    transport_complete_local_gradients,
)
from xorl.server.runner.adapters.gradient_ownership import (
    AdapterGradientOwnershipError,
    AdapterGradientOwnershipPlan,
    AdapterGradientUniformRejection,
    GradientRepresentation,
    GradientScaleState,
    ParameterOwnershipDeclaration,
    ReductionAuthority,
    ReductionAxis,
    ReductionDomainPlan,
    compile_adapter_gradient_ownership,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    active_rectangle_from_layout_descriptor as _active_rectangle_from_layout_descriptor,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    canonical_parameter_order as _canonical_parameter_order,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    clone_state_to_cpu as _clone_state_to_cpu,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    commit_optimizer_state as _commit_optimizer_state_transactionally,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    coordinate_restore_error as _optimizer_restore_rank_errors,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    reconstruct_optimizer_state,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    validate_live_optimizer_binding as _validate_live_optimizer_binding,
)
from xorl.server.runner.adapters.optimizer_reshard import (
    validate_staged_optimizer_compatibility as _validate_staged_optimizer_compatibility,
)
from xorl.server.runner.adapters.sharded_state import (
    AdapterTensorLayout,
    canonical_parameter_name,
    deterministic_local_initialization,
    discover_adapter_layouts,
    pack_logical_tensor,
    wait_for_local_tensor,
)
from xorl.server.security import resolve_path_within, resolve_server_artifact, validate_identifier
from xorl.server.session_spec import (
    load_session_spec_from_checkpoint,
    session_optimizer_build_kwargs,
    write_session_spec,
)


try:
    from torch.distributed._tensor import DTensor

    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False


logger = logging.getLogger(__name__)


_TARGET_MANIFEST_FILENAME = "lora_target_manifest.json"

OPTIMIZER_SHARD_MANIFEST_FILENAME = "optimizer_shards.json"
_LEGACY_OPTIMIZER_FILENAME = "optimizer.pt"
_OPTIMIZER_STATE_METADATA_KEY = "xorl_optimizer_state_v1"
_OPTIMIZER_STATE_MAX_DEPTH = 64


def _first_restore_contract_difference(checkpoint: Any, live: Any, path: str = "contract") -> Optional[str]:
    """Return the first named field difference in two JSON-shaped contracts."""

    if type(checkpoint) is not type(live):
        return f"{path}: checkpoint type={type(checkpoint).__name__}, live type={type(live).__name__}"
    if isinstance(checkpoint, dict):
        checkpoint_keys = set(checkpoint)
        live_keys = set(live)
        if checkpoint_keys != live_keys:
            return f"{path} keys: checkpoint={sorted(checkpoint_keys)!r}, live={sorted(live_keys)!r}"
        for key in sorted(checkpoint):
            difference = _first_restore_contract_difference(checkpoint[key], live[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(checkpoint, list):
        if len(checkpoint) != len(live):
            return f"{path} length: checkpoint={len(checkpoint)}, live={len(live)}"
        for index, (checkpoint_item, live_item) in enumerate(zip(checkpoint, live, strict=True)):
            difference = _first_restore_contract_difference(checkpoint_item, live_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if checkpoint != live:
        return f"{path}: checkpoint={checkpoint!r}, live={live!r}"
    return None


def _optimizer_shard_rank_world() -> Tuple[int, int]:
    """Current (rank, world_size); uninitialized torch.distributed counts as single-rank."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _optimizer_shard_filename(rank: int) -> str:
    return f"optimizer-rank{rank:05d}.safetensors"


def _encode_optimizer_state(value: Any, tensors: Dict[str, torch.Tensor], *, depth: int = 0) -> Any:
    if depth > _OPTIMIZER_STATE_MAX_DEPTH:
        raise ValueError("optimizer state exceeds the supported nesting depth")
    if isinstance(value, torch.Tensor):
        tensor_name = f"tensor-{len(tensors):08d}"
        tensors[tensor_name] = value.detach().to(device="cpu", copy=True).contiguous()
        return {"tensor": tensor_name}
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        return {"float": value.hex()}
    if value_type is list:
        return {"list": [_encode_optimizer_state(item, tensors, depth=depth + 1) for item in value]}
    if value_type is tuple:
        return {"tuple": [_encode_optimizer_state(item, tensors, depth=depth + 1) for item in value]}
    if value_type is dict:
        return {
            "dict": [
                [
                    _encode_optimizer_state(key, tensors, depth=depth + 1),
                    _encode_optimizer_state(item, tensors, depth=depth + 1),
                ]
                for key, item in value.items()
            ]
        }
    raise TypeError(f"unsupported optimizer-state value: {value_type.__module__}.{value_type.__qualname__}")


def _decode_optimizer_state(value: Any, tensors: Dict[str, torch.Tensor], consumed: set[str], *, depth: int = 0) -> Any:
    if depth > _OPTIMIZER_STATE_MAX_DEPTH:
        raise ValueError("optimizer state exceeds the supported nesting depth")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is not dict or len(value) != 1:
        raise ValueError("invalid optimizer-state metadata node")
    kind, payload = next(iter(value.items()))
    if kind == "tensor":
        if type(payload) is not str or payload not in tensors or payload in consumed:
            raise ValueError(f"invalid optimizer-state tensor reference: {payload!r}")
        consumed.add(payload)
        return tensors[payload]
    if kind == "float":
        if type(payload) is not str:
            raise ValueError("invalid optimizer-state float encoding")
        return float.fromhex(payload)
    if kind in {"list", "tuple"}:
        if type(payload) is not list:
            raise ValueError(f"invalid optimizer-state {kind} encoding")
        decoded = [_decode_optimizer_state(item, tensors, consumed, depth=depth + 1) for item in payload]
        return decoded if kind == "list" else tuple(decoded)
    if kind == "dict":
        if type(payload) is not list:
            raise ValueError("invalid optimizer-state dict encoding")
        result = {}
        for pair in payload:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("invalid optimizer-state dict item")
            key = _decode_optimizer_state(pair[0], tensors, consumed, depth=depth + 1)
            if type(key) not in {bool, int, str}:
                raise ValueError(f"unsupported optimizer-state mapping key: {type(key).__name__}")
            if key in result:
                raise ValueError(f"duplicate optimizer-state mapping key: {key!r}")
            result[key] = _decode_optimizer_state(pair[1], tensors, consumed, depth=depth + 1)
        return result
    raise ValueError(f"unknown optimizer-state metadata kind: {kind!r}")


def _save_optimizer_state_safetensors(optimizer_state: Dict[str, Any], path: str) -> None:
    tensors: Dict[str, torch.Tensor] = {}
    encoded = _encode_optimizer_state(optimizer_state, tensors)
    safetensors_save_file(
        tensors,
        path,
        metadata={_OPTIMIZER_STATE_METADATA_KEY: json.dumps(encoded, separators=(",", ":"))},
    )


def _load_optimizer_state_safetensors(path: str, device: torch.device) -> Dict[str, Any]:
    with safe_open(path, framework="pt", device=str(device)) as shard:
        metadata = shard.metadata()
        encoded_text = metadata.get(_OPTIMIZER_STATE_METADATA_KEY)
        if encoded_text is None:
            raise ValueError(f"optimizer shard is missing {_OPTIMIZER_STATE_METADATA_KEY} metadata: {path}")
        encoded = json.loads(encoded_text)
        tensors = {name: shard.get_tensor(name) for name in shard.keys()}
    consumed: set[str] = set()
    optimizer_state = _decode_optimizer_state(encoded, tensors, consumed)
    if type(optimizer_state) is not dict:
        raise ValueError(f"optimizer shard root must be a mapping: {path}")
    unused = set(tensors) - consumed
    if unused:
        raise ValueError(f"optimizer shard contains unreferenced tensors: {sorted(unused)}")
    return optimizer_state


def _adapter_param_structure_fingerprint(lora_params: Dict[str, nn.Parameter]) -> str:
    """Fingerprint the canonical non-empty optimizer (name, shape, dtype) sequence.

    Optimizer state_dicts key parameters by position, so a saved shard is only
    meaningful on a rank whose parameter sequence matches the saving rank
    exactly. Under expert parallelism the same position holds DIFFERENT expert
    slices on different ranks, which shape checks alone cannot distinguish;
    world-size + per-rank fingerprints together make misassignment loud. Use
    canonical names and order so wrapper prefixes and dictionary insertion order
    do not reject an otherwise equivalent optimizer state.
    """
    structure = [
        [canonical_parameter_name(name), list(lora_params[name].shape), str(lora_params[name].dtype)]
        for name in _canonical_parameter_order(lora_params)
        if lora_params[name].numel() > 0
    ]
    return hashlib.sha256(json.dumps(structure).encode("utf-8")).hexdigest()


def _layout_descriptor_fingerprint(descriptors: List[Dict[str, Any]], *, world_size: int) -> str:
    payload = {"world_size": int(world_size), "layouts": descriptors}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _descriptor_structure_fingerprint(
    descriptors: List[Dict[str, Any]],
    parameter_order: List[str],
) -> str:
    by_name = {canonical_parameter_name(descriptor["fqn"]): descriptor for descriptor in descriptors}
    structure = []
    for name in parameter_order:
        descriptor = by_name[canonical_parameter_name(name)]
        structure.append(
            [
                canonical_parameter_name(name),
                [int(value) for value in descriptor["active_storage_shape"]],
                f"torch.{descriptor['dtype']}",
            ]
        )
    return hashlib.sha256(json.dumps(structure).encode("utf-8")).hexdigest()


def _reshard_adapter_optimizer_state(
    adapter_state: "AdapterState",
    path: str,
    manifest: Dict[str, Any],
    *,
    rank: int,
    world: int,
) -> Dict[str, Any]:
    """Load source shards lazily and reconstruct this rank's logical rectangles."""

    def load_source_rank(source_rank: int) -> Dict[str, Any]:
        shard_path = resolve_path_within(
            path,
            _optimizer_shard_filename(source_rank),
            must_exist=True,
            reject_symlinks=True,
        )
        return _load_optimizer_state_safetensors(str(shard_path), torch.device("cpu"))

    return reconstruct_optimizer_state(
        adapter_state,
        manifest,
        rank=rank,
        world=world,
        load_source_rank=load_source_rank,
    )


def save_adapter_optimizer_shards(adapter_state: "AdapterState", path: str) -> Dict[str, Any]:
    """Collectively save per-rank adapter optimizer shards plus a topology manifest.

    Under expert parallelism each rank's per-adapter optimizer holds Adam
    moments only for that rank's local expert slices, so one rank's state_dict
    is not a complete checkpoint: the legacy single-file ``optimizer.pt`` wrote
    rank 0's moments and silently dropped every other rank's. ALL ranks must
    call this together; rank 0 additionally writes the manifest.
    """
    rank, world = _optimizer_shard_rank_world()
    os.makedirs(path, exist_ok=True)
    shard_path = resolve_path_within(path, _optimizer_shard_filename(rank), reject_symlinks=True)
    _save_optimizer_state_safetensors(
        adapter_state.optimizer.state_dict(),
        str(shard_path),
    )
    fingerprint = _adapter_param_structure_fingerprint(adapter_state.local_params)
    layout_descriptors = [
        adapter_state.tensor_layouts[name].to_json_dict() for name in sorted(adapter_state.tensor_layouts)
    ]
    session_rank = int(adapter_state.session_spec["lora_config"]["lora_rank"])
    if world > 1:
        fingerprints: List[Optional[str]] = [None] * world
        torch.distributed.all_gather_object(fingerprints, fingerprint)
        layout_fingerprints: List[Optional[str]] = [None] * world
        torch.distributed.all_gather_object(layout_fingerprints, adapter_state.layout_fingerprint)
        layout_descriptors_by_rank: List[Optional[List[Dict[str, Any]]]] = [None] * world
        torch.distributed.all_gather_object(layout_descriptors_by_rank, layout_descriptors)
        local_optimizer_order = [
            canonical_parameter_name(name)
            for name in _canonical_parameter_order(adapter_state.local_params)
            if adapter_state.local_params[name].numel() > 0
        ]
        optimizer_orders_by_rank: List[Optional[List[str]]] = [None] * world
        torch.distributed.all_gather_object(optimizer_orders_by_rank, local_optimizer_order)
    else:
        fingerprints = [fingerprint]
        layout_fingerprints = [adapter_state.layout_fingerprint]
        layout_descriptors_by_rank = [layout_descriptors]
        optimizer_orders_by_rank = [
            [
                canonical_parameter_name(name)
                for name in _canonical_parameter_order(adapter_state.local_params)
                if adapter_state.local_params[name].numel() > 0
            ]
        ]
    manifest = {
        "format_version": 3,
        "world_size": world,
        "per_rank_param_structure_sha256": fingerprints,
        "per_rank_layout_fingerprint": layout_fingerprints,
        "per_rank_layout_descriptors": layout_descriptors_by_rank,
        "session_rank": session_rank,
        "optimizer_parameter_order": optimizer_orders_by_rank[0],
        "per_rank_optimizer_parameter_order": optimizer_orders_by_rank,
    }
    if rank == 0:
        with open(os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME), "w") as f:
            json.dump(manifest, f, indent=2)
    return manifest


def load_adapter_optimizer_shards(
    adapter_state: "AdapterState",
    path: str,
    device: torch.device,
) -> bool:
    """Restore this rank's adapter optimizer shard saved by ``save_adapter_optimizer_shards``.

    Returns True when optimizer state was restored, False when the checkpoint
    carries no optimizer state at all. Raises on topology or parameter-structure
    mismatch, and on legacy single-file checkpoints in multi-rank runs (loading
    rank-0 moments on every rank silently misassigns them).
    """
    manifest_path = os.path.join(path, OPTIMIZER_SHARD_MANIFEST_FILENAME)
    legacy_path = os.path.join(path, _LEGACY_OPTIMIZER_FILENAME)
    rank, world = _optimizer_shard_rank_world()
    if os.path.exists(manifest_path):
        optimizer_state: Optional[Dict[str, Any]] = None
        topology_changed = False
        saved_world = -1
        live_order: List[str] = []
        live_template: Optional[Dict[str, Any]] = None
        local_error: Optional[BaseException] = None
        try:
            live_order, live_template = _validate_live_optimizer_binding(adapter_state)
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("format_version") != 3:
                raise RuntimeError(
                    f"Adapter optimizer checkpoint at {path} uses obsolete or unsupported layout format version "
                    f"{manifest.get('format_version')!r}; local-shard optimizer state cannot be migrated by position. "
                    "Load weights-only (load_optimizer=False) and re-save on the new topology."
                )
            saved_world = int(manifest["world_size"])
            descriptors_by_rank = manifest.get("per_rank_layout_descriptors")
            layout_fingerprints = manifest.get("per_rank_layout_fingerprint")
            structure_fingerprints = manifest.get("per_rank_param_structure_sha256")
            orders_by_rank = manifest.get("per_rank_optimizer_parameter_order")
            if not all(
                type(values) is list and len(values) == saved_world
                for values in (
                    descriptors_by_rank,
                    layout_fingerprints,
                    structure_fingerprints,
                    orders_by_rank,
                )
            ):
                raise RuntimeError(
                    f"Adapter optimizer checkpoint declares world_size={saved_world} but its topology manifest "
                    "is incomplete"
                )
            for source_rank in range(saved_world):
                descriptors = descriptors_by_rank[source_rank]
                parameter_order = orders_by_rank[source_rank]
                if type(descriptors) is not list or type(parameter_order) is not list:
                    raise RuntimeError(f"Invalid optimizer topology metadata for saved rank {source_rank}")
                descriptor_names_list = []
                for descriptor in descriptors:
                    if type(descriptor) is not dict or type(descriptor.get("fqn")) is not str:
                        raise RuntimeError(f"Invalid adapter layout descriptor for saved rank {source_rank}")
                    _active_rectangle_from_layout_descriptor(
                        descriptor,
                        context=f"saved rank {source_rank}, parameter {descriptor['fqn']}",
                    )
                    if math.prod(int(value) for value in descriptor["active_storage_shape"]) > 0:
                        descriptor_names_list.append(canonical_parameter_name(descriptor["fqn"]))
                if len(set(descriptor_names_list)) != len(descriptor_names_list):
                    raise RuntimeError(f"Duplicate adapter layout descriptor on saved rank {source_rank}")
                descriptor_names = set(descriptor_names_list)
                canonical_order = [canonical_parameter_name(name) for name in parameter_order]
                if (
                    canonical_order != parameter_order
                    or len(set(canonical_order)) != len(canonical_order)
                    or set(canonical_order) != descriptor_names
                ):
                    raise RuntimeError(
                        f"Adapter optimizer shard at {path} has a different optimizer parameter order "
                        f"on saved rank {source_rank}"
                    )
                if layout_fingerprints[source_rank] != _layout_descriptor_fingerprint(
                    descriptors, world_size=saved_world
                ):
                    raise RuntimeError(f"Layout fingerprint does not match descriptors for saved rank {source_rank}")
                if structure_fingerprints[source_rank] != _descriptor_structure_fingerprint(
                    descriptors,
                    parameter_order,
                ):
                    raise RuntimeError(f"Parameter fingerprint does not match descriptors for saved rank {source_rank}")
            expected_session_rank = manifest.get("session_rank")
            live_session_rank = int(adapter_state.session_spec["lora_config"]["lora_rank"])
            if expected_session_rank != live_session_rank:
                raise RuntimeError(
                    f"Adapter optimizer shard at {path} was saved for session_rank={expected_session_rank!r}, "
                    f"but the live adapter uses session_rank={live_session_rank}; load weights-only "
                    "(load_optimizer=False) instead."
                )
            topology_changed = saved_world != world
            if not topology_changed:
                expected_fingerprint = manifest["per_rank_param_structure_sha256"][rank]
                live_fingerprint = _adapter_param_structure_fingerprint(adapter_state.local_params)
                expected_layout_fingerprint = manifest.get("per_rank_layout_fingerprint", [None] * world)[rank]
                per_rank_orders = manifest.get("per_rank_optimizer_parameter_order")
                expected_order = (
                    per_rank_orders[rank] if per_rank_orders is not None else manifest.get("optimizer_parameter_order")
                )
                topology_changed = expected_layout_fingerprint != adapter_state.layout_fingerprint
                if not topology_changed and expected_fingerprint != live_fingerprint:
                    raise RuntimeError(
                        f"Adapter optimizer shard for rank {rank} at {path} was saved for a different local "
                        "parameter structure"
                    )
                if not topology_changed and expected_order != live_order:
                    raise RuntimeError(f"Adapter optimizer shard at {path} has a different optimizer parameter order")
        except Exception as exc:
            local_error = exc
        _optimizer_restore_rank_errors(local_error, phase="manifest preflight")

        if torch.distributed.is_available() and torch.distributed.is_initialized() and world > 1:
            topology_votes: List[Optional[bool]] = [None] * world
            torch.distributed.all_gather_object(topology_votes, topology_changed)
            topology_changed = any(bool(vote) for vote in topology_votes)

        local_error = None
        try:
            if topology_changed:
                optimizer_state = _reshard_adapter_optimizer_state(
                    adapter_state,
                    path,
                    manifest,
                    rank=rank,
                    world=world,
                )
            else:
                shard_path = resolve_path_within(
                    path,
                    _optimizer_shard_filename(rank),
                    must_exist=True,
                    reject_symlinks=True,
                )
                optimizer_state = _load_optimizer_state_safetensors(str(shard_path), torch.device("cpu"))
            if live_template is None:  # pragma: no cover - guarded by coordinated manifest preflight
                raise RuntimeError("Live adapter optimizer validation did not produce a state template")
            _validate_staged_optimizer_compatibility(
                adapter_state,
                optimizer_state,
                live_order,
                live_template,
            )
        except Exception as exc:
            local_error = exc
        _optimizer_restore_rank_errors(local_error, phase="preflight")
        if optimizer_state is None:  # pragma: no cover - guarded by the coordinated preflight
            raise RuntimeError("Adapter optimizer restore produced no staged state")
        _commit_optimizer_state_transactionally(adapter_state, optimizer_state)
        if topology_changed:
            logger.info(
                "Loaded topology-resharded adapter optimizer from world_size=%d into rank %d/%d",
                saved_world,
                rank,
                world,
            )
        else:
            logger.info("Loaded rank-%d adapter optimizer shard from %s", rank, shard_path)
        return True
    if os.path.exists(legacy_path):
        raise RuntimeError(
            f"Adapter checkpoint at {path} has a legacy pickle-backed optimizer.pt, which is never loaded. "
            "Re-save with the safetensors sharded format, or pass load_optimizer=False for a weights-only "
            "warm start."
        )
    incomplete_shards = sorted(Path(path).glob("optimizer-rank*"))
    if incomplete_shards:
        raise RuntimeError(
            f"Adapter checkpoint at {path} contains per-rank optimizer shards but no "
            f"{OPTIMIZER_SHARD_MANIFEST_FILENAME}. Refusing an incomplete optimizer resume; "
            "use load_optimizer=False for a weights-only warm start."
        )
    return False


@dataclass
class AdapterGradientScratch:
    """One session's local FP32 raw-numerator image and capture ledger."""

    epoch: int = 0
    next_capture_ordinal: int = 0
    denominator: float = 0.0
    numerator_scale: Optional[float] = None
    source: Optional[str] = None
    numerators: Dict[str, torch.Tensor] = field(default_factory=dict)
    capture_open: bool = False
    capture_staged: bool = False
    staged_denominator: float = 0.0
    staged_numerator_scale: Optional[float] = None
    staged_parameter_fqns: Tuple[str, ...] = ()
    staged_numerators: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class AdapterState:
    """Complete isolated state for one training run.

    Key insight: Each adapter owns its own nn.Parameter objects, which have
    their own .grad slots. This prevents gradient collision when multiple
    adapters' forward_backward calls interleave.
    """

    model_id: str
    session_spec: Dict[str, Any]
    local_params: Dict[str, nn.Parameter]  # Active local shards with own .grad
    tensor_layouts: Dict[str, AdapterTensorLayout]
    layout_fingerprint: str
    optimizer: torch.optim.Optimizer  # Per-adapter optimizer
    registration_ordinal: int = 0
    gradient_ownership_plan: Optional[AdapterGradientOwnershipPlan] = None
    gradient_scratch: AdapterGradientScratch = field(default_factory=AdapterGradientScratch)
    publication_eligible: bool = True
    publication_pending: bool = False
    poisoned: bool = False
    last_transport_stats: AdapterGradientTransportStats = field(default_factory=AdapterGradientTransportStats)
    global_step: int = 0
    global_forward_backward_step: int = 0
    lr: float = 1e-5
    last_access_time: float = field(default_factory=time.time)  # For LRU eviction

    @property
    def lora_params(self) -> Dict[str, nn.Parameter]:
        """Deprecated local-only view retained for external compatibility.

        This is intentionally not a logical/full tensor API.  Internal code
        must use ``local_params`` and ``tensor_layouts`` explicitly.
        """

        return self.local_params


class LoRAAdapterManager:
    """
    Manages multiple LoRA adapters - one per model_id.

    Design: Each model_id has its own nn.Parameter objects and optimizer.
    The model's LoRA params are used as "scratch space" for forward/backward.

    Flow:
    1. ``prepare_forward`` copies adapter weights into the model.
    2. Backward produces model gradients under a compiled ownership plan.
    3. The completion rendezvous commits staged raw numerators into persistent scratch.
    4. ``optim_step`` completes residual reductions and mutates the adapter optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_adapters: int = 10,
        checkpoint_dir: Optional[str] = None,
        auto_save_on_eviction: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        optimizer_type: str = "adamw",
        optimizer_dtype: str = "bf16",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        optimizer_fused: Optional[bool] = None,
        gradient_ownership_bucket_bytes: int = 64 * 1024 * 1024,
    ):
        """
        Initialize the adapter manager.

        Args:
            model: The model with LoRA layers injected
            device: Device to create adapter parameters on
            max_adapters: Maximum number of adapters to keep in memory (LRU eviction)
            checkpoint_dir: Directory for saving adapter checkpoints (default: outputs/adapters)
            auto_save_on_eviction: If True, save adapter state before LRU eviction
            lora_config: LoRA configuration dict (for saving adapter_config.json)
            optimizer_config: Training optimizer configuration for per-adapter optimizers
            optimizer_type: Optimizer type passed to xorl.optim.build_optimizer
            optimizer_dtype: Optimizer state dtype for supported optimizers
            optimizer_kwargs: Optimizer-specific kwargs (e.g. Muon settings)
            weight_decay: Weight decay used when building adapter optimizers
            betas: Beta coefficients for Adam-family optimizers
            eps: Epsilon used by Adam-family optimizers
            optimizer_fused: Whether to request fused optimizer kernels
            gradient_ownership_bucket_bytes: Maximum residual-transport bucket bytes
        """
        self.model = model
        self.device = device
        self.max_adapters = max_adapters
        self.checkpoint_dir = checkpoint_dir or "outputs/adapters"
        self.auto_save_on_eviction = auto_save_on_eviction
        self.lora_config = lora_config or {}
        self.optimizer_config = optimizer_config or {}
        self.optimizer_type = optimizer_type
        self.optimizer_dtype = optimizer_dtype
        self.optimizer_kwargs = deepcopy(optimizer_kwargs or {})
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.optimizer_fused = device.type == "cuda" if optimizer_fused is None else optimizer_fused
        if gradient_ownership_bucket_bytes <= 0:
            raise ValueError("gradient_ownership_bucket_bytes must be positive")
        self.gradient_ownership_bucket_bytes = int(gradient_ownership_bucket_bytes)
        self.adapters: Dict[str, AdapterState] = {}
        self.current_adapter_id: Optional[str] = None
        self._layout_cache: Dict[
            int,
            Tuple[
                Dict[str, AdapterTensorLayout],
                str,
                Dict[str, tuple[tuple[int, ...], ...]],
            ],
        ] = {}
        self._model_param_ids: Dict[str, int] = {}
        self._adapter_registration_ordinals: Dict[str, int] = {}

        # Cache the list of LoRA parameter names for efficient lookups
        self._lora_param_names: List[str] = []
        self._lora_param_metadata: Dict[str, Dict[str, Any]] = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self._lora_param_names.append(name)
                param_shape = tuple(param.shape if _HAS_DTENSOR and isinstance(param, DTensor) else param.data.shape)
                self._model_param_ids[name] = id(param)
                self._lora_param_metadata[name] = {
                    "shape": param_shape,
                    "dtype": param.dtype if _HAS_DTENSOR and isinstance(param, DTensor) else param.data.dtype,
                    "rank_dim": self._infer_lora_rank_dim(name, param_shape),
                }
        self._pipeline_parallel_size = int(
            self.lora_config.get("pipeline_parallel_size", self.lora_config.get("pp_size", 1))
        )
        try:
            from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

            self._pipeline_parallel_size = max(self._pipeline_parallel_size, int(get_parallel_state().pp_size))
        except Exception:
            pass
        if self._pipeline_parallel_size > 1 and self._lora_param_names:
            raise RuntimeError(
                "Multi-adapter LoRA state requires a single pipeline stage; "
                "pipeline parallelism has no known optimizer ownership group."
            )

        logger.info(
            f"LoRAAdapterManager initialized with {len(self._lora_param_names)} LoRA parameters, "
            f"max_adapters={max_adapters}, auto_save_on_eviction={auto_save_on_eviction}, "
            f"optimizer={optimizer_type}, adapter_gradient_ownership=compiled_authoritative"
        )

    @staticmethod
    def _infer_lora_rank_dim(name: str, shape: Tuple[int, ...]) -> int:
        """Infer which tensor dimension corresponds to the LoRA rank."""
        if "lora_A" in name:
            if len(shape) == 2:
                return 0
            if len(shape) == 3:
                return 2
        if "lora_B" in name:
            if len(shape) == 2:
                return 1
            if len(shape) == 3:
                return 1
        raise ValueError(f"Cannot infer LoRA rank dimension for parameter {name!r} with shape {shape!r}")

    @staticmethod
    def _session_rank(session_spec: Dict[str, Any]) -> int:
        return int(session_spec["lora_config"]["lora_rank"])

    @staticmethod
    def _session_alpha(session_spec: Dict[str, Any]) -> int:
        return int(session_spec["lora_config"]["lora_alpha"])

    @staticmethod
    def _strip_optimizer_config(session_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return the structural part of a LoRA session spec without optimizer metadata."""
        stripped = deepcopy(session_spec)
        stripped.pop("optimizer_config", None)
        return stripped

    @staticmethod
    def _strip_optimizer_learning_rate(session_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Return a session spec without the mutable optimizer learning-rate field."""
        stripped = deepcopy(session_spec)
        optimizer_config = stripped.get("optimizer_config")
        if isinstance(optimizer_config, dict):
            optimizer_config.pop("learning_rate", None)
        return stripped

    @staticmethod
    def _validate_session_restore_compatibility(
        checkpoint_session_spec: Dict[str, Any],
        live_session_spec: Dict[str, Any],
        *,
        load_optimizer: bool,
    ) -> None:
        """Validate only runtime fields that define the live adapter/optimizer substrate."""

        checks = (
            ("base_model", checkpoint_session_spec.get("base_model"), live_session_spec.get("base_model")),
            ("is_lora", checkpoint_session_spec.get("is_lora"), live_session_spec.get("is_lora")),
            (
                "lora_config.lora_rank",
                checkpoint_session_spec.get("lora_config", {}).get("lora_rank"),
                live_session_spec.get("lora_config", {}).get("lora_rank"),
            ),
            (
                "lora_config.lora_alpha",
                checkpoint_session_spec.get("lora_config", {}).get("lora_alpha"),
                live_session_spec.get("lora_config", {}).get("lora_alpha"),
            ),
        )
        for field_name, checkpoint_value, live_value in checks:
            if checkpoint_value != live_value:
                raise ValueError(
                    f"Checkpoint adapter field {field_name} is incompatible with the live session: "
                    f"checkpoint={checkpoint_value!r}, live={live_value!r}"
                )
        if load_optimizer:
            checkpoint_type = str(checkpoint_session_spec.get("optimizer_config", {}).get("type", "")).lower()
            live_type = str(live_session_spec.get("optimizer_config", {}).get("type", "")).lower()
            if not checkpoint_type or checkpoint_type != live_type:
                raise ValueError(
                    "Checkpoint optimizer type is incompatible with the live session: "
                    f"checkpoint={checkpoint_type!r}, live={live_type!r}"
                )

    @staticmethod
    def _validate_ownership_restore_contract(
        checkpoint_contract: Any,
        live_plan: AdapterGradientOwnershipPlan,
    ) -> None:
        """Compare direct topology/producer fields; never compare the plan label."""

        if not isinstance(checkpoint_contract, dict):
            raise ValueError(
                "Authoritative optimizer checkpoint is missing its direct ownership restore contract; "
                "load weights-only (load_optimizer=False) and re-save it"
            )
        live_contract = live_plan.optimizer_restore_contract()
        difference = _first_restore_contract_difference(checkpoint_contract, live_contract)
        if difference is not None:
            raise ValueError(f"Checkpoint adapter-gradient topology/producer contract is incompatible: {difference}")

    @staticmethod
    def _serialize_optimizer_metadata_value(value: Any) -> Any:
        """Convert optimizer metadata into JSON-safe values."""
        if isinstance(value, torch.dtype):
            if value == torch.bfloat16:
                return "bf16"
            if value == torch.float32:
                return "fp32"
            return str(value)
        if isinstance(value, dict):
            return {k: LoRAAdapterManager._serialize_optimizer_metadata_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LoRAAdapterManager._serialize_optimizer_metadata_value(v) for v in value]
        return value

    @staticmethod
    def _update_state_learning_rate(state: AdapterState, lr: float) -> None:
        """Keep adapter LR, optimizer param groups, and session spec in sync."""
        state.lr = float(lr)
        state.session_spec.setdefault("optimizer_config", {})["learning_rate"] = state.lr
        for param_group in state.optimizer.param_groups:
            if state.session_spec.get("optimizer_config", {}).get("type") == "muon" and param_group.get(
                "use_muon", False
            ):
                continue
            param_group["lr"] = state.lr

    def _max_supported_session_rank(self) -> int:
        """Return the largest LoRA rank the live model substrate can support."""
        if not self._lora_param_metadata:
            raise RuntimeError("Cannot determine LoRA rank capacity: model does not expose any LoRA parameters.")
        return min(metadata["shape"][metadata["rank_dim"]] for metadata in self._lora_param_metadata.values())

    def _discover_layouts(
        self,
        session_rank: int,
        *,
        local_group_memberships: Mapping[str, tuple[int, ...]] | None = None,
    ) -> Tuple[
        Dict[str, AdapterTensorLayout],
        str,
        Dict[str, tuple[tuple[int, ...], ...]],
    ]:
        cached = self._layout_cache.get(session_rank)
        if cached is not None:
            return cached
        if local_group_memberships is None:
            from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

            parallel_state = get_parallel_state()

            def _public_group_members(group: Any) -> tuple[int, ...]:
                if group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
                    return (0,)
                return tuple(sorted(torch.distributed.get_process_group_ranks(group)))

            local_group_memberships = {}
            sp_group = parallel_state.sp_grad_sync_group
            output_group = getattr(parallel_state, "lm_head_tp_replica_group", None)
            ep_group = (
                parallel_state.ep_group
                if bool(getattr(parallel_state, "ep_enabled", getattr(parallel_state, "ep_size", 1) > 1))
                else None
            )
            if sp_group is not None:
                local_group_memberships["sequence_parallel"] = _public_group_members(sp_group)
            if output_group is not None:
                local_group_memberships["output_projection_replica"] = _public_group_members(output_group)
            if ep_group is not None:
                local_group_memberships["expert_parallel_replica"] = _public_group_members(ep_group)
        layouts, fingerprint, group_memberships = discover_adapter_layouts(
            self.model,
            self._lora_param_metadata,
            active_rank=session_rank,
            pipeline_parallel_size=self._pipeline_parallel_size,
            local_group_memberships=local_group_memberships,
        )
        self._layout_cache[session_rank] = layouts, fingerprint, group_memberships
        return layouts, fingerprint, group_memberships

    def gradient_ownership_group_memberships(
        self,
        model_id: str,
    ) -> Dict[str, tuple[tuple[int, ...], ...]]:
        """Return replica-group families captured by the existing layout exchange."""

        state = self.get_adapter_state(model_id)
        session_rank = self._session_rank(state.session_spec)
        try:
            _layouts, _fingerprint, group_memberships = self._layout_cache[session_rank]
        except KeyError:
            raise RuntimeError("Adapter layout discovery metadata is unavailable") from None
        return group_memberships

    def _agree_gradient_ownership_fingerprint(self, plan: AdapterGradientOwnershipPlan) -> None:
        """Require one registration-time plan fingerprint using fixed tensors."""

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        group = torch.distributed.group.WORLD
        if torch.distributed.get_world_size(group=group) <= 1:
            return
        fingerprint = torch.tensor(
            list(bytes.fromhex(plan.fingerprint)),
            dtype=torch.uint8,
            device=self.device,
        )
        minimum = fingerprint.clone()
        maximum = fingerprint.clone()
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=group)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        if not torch.equal(minimum, maximum):
            raise AdapterGradientOwnershipError("Adapter gradient ownership plan differs across ranks")

    def _validate_model_layout_identity(self, state: AdapterState) -> None:
        """Fail closed if FSDP/EP replaced or moved a parameter after discovery."""

        current = {name: param for name, param in self.model.named_parameters() if name in self._model_param_ids}
        if set(current) != set(self._model_param_ids):
            raise RuntimeError("Trainable LoRA parameter set changed after adapter layout discovery")
        for name, expected_id in self._model_param_ids.items():
            param = current[name]
            if id(param) != expected_id:
                raise RuntimeError(f"LoRA parameter identity changed after layout discovery: {name}")
            layout = state.tensor_layouts[name]
            raw = param.data if isinstance(param, nn.Parameter) else param
            local = wait_for_local_tensor(raw.to_local() if _HAS_DTENSOR and isinstance(raw, DTensor) else raw)
            if tuple(local.shape) != layout.local_substrate_shape:
                raise RuntimeError(
                    f"LoRA parameter placement changed after layout discovery for {name}: "
                    f"local shape {tuple(local.shape)} != {layout.local_substrate_shape}"
                )

    @staticmethod
    def _replica_validation_enabled() -> bool:
        return os.environ.get("XORL_VALIDATE_ADAPTER_REPLICAS") == "1"

    def _validate_replica_coherence(self, state: AdapterState, *, gradients: bool) -> None:
        """Optionally verify that identical logical rectangles remain coherent."""

        if (
            not self._replica_validation_enabled()
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
        ):
            return
        local_payload = []
        local_descriptor_records = [
            (layout.replica_key, layout.replica_count)
            for layout in state.tensor_layouts.values()
            if layout.needs_ep_gradient_sync
        ]
        for name in sorted(state.local_params):
            layout = state.tensor_layouts[name]
            if not layout.needs_ep_gradient_sync:
                continue
            value = state.local_params[name].grad if gradients else state.local_params[name].detach()
            local_payload.append((layout.replica_key, None if value is None else value.detach().cpu().contiguous()))
        gathered_descriptors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_descriptors, local_descriptor_records)
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_payload)
        global_keys = sorted({tuple(key) for rank_records in gathered_descriptors for key, _ in rank_records})
        descriptor_counts: dict[tuple[Any, ...], int] = {}
        descriptor_expected: dict[tuple[Any, ...], set[int]] = {}
        for rank_records in gathered_descriptors:
            for key, replica_count in rank_records:
                key = tuple(key)
                descriptor_counts[key] = descriptor_counts.get(key, 0) + 1
                descriptor_expected.setdefault(key, set()).add(int(replica_count))
        by_key: dict[tuple[Any, ...], list[Optional[torch.Tensor]]] = {}
        for rank_payload in gathered:
            for key, value in rank_payload:
                by_key.setdefault(tuple(key), []).append(value)
        checked_rectangles = 0
        checked_replica_values = 0
        for key in global_keys:
            expected_counts = descriptor_expected.get(key, set())
            expected_count = next(iter(expected_counts)) if len(expected_counts) == 1 else -1
            if expected_count < 0 or descriptor_counts.get(key, 0) != expected_count:
                raise RuntimeError(
                    f"Replica descriptor universe diverged for {key[0]!r}: "
                    f"present={descriptor_counts.get(key, 0)} expected={sorted(expected_counts)}"
                )
            if expected_count <= 1:
                continue
            checked_rectangles += 1
            checked_replica_values += expected_count
            values = by_key.get(key, [])
            if len(values) != expected_count:
                raise RuntimeError(f"Replica value count changed for {key[0]!r}: {len(values)}")
            reference = values[0]
            if any((value is None) != (reference is None) for value in values):
                raise RuntimeError(f"Replica gradient/weight presence diverged for {key[0]!r}")
            if reference is not None and any(not torch.equal(reference, value) for value in values[1:]):
                raise RuntimeError(f"Replica values diverged for logical rectangle {key[0]!r}")

        if torch.distributed.get_rank() == 0:
            phase = "gradient" if gradients else "parameter"
            logger.info(
                "Adapter EP replica coherence: "
                + json.dumps(
                    {
                        "phase": phase,
                        "step": state.global_step + (0 if gradients else 1),
                        "passed": True,
                        "descriptor_universe_size": len(global_keys),
                        "checked_rectangles": checked_rectangles,
                        "checked_replica_values": checked_replica_values,
                    },
                    sort_keys=True,
                )
            )

    @staticmethod
    def _optimizer_state_digest(optimizer_state: Dict[str, Any]) -> str:
        """Hash one optimizer state without gathering its tensor payload."""

        digest = hashlib.sha256()
        for key in sorted(optimizer_state, key=str):
            digest.update(str(key).encode("utf-8"))
            value = optimizer_state[key]
            if isinstance(value, torch.Tensor):
                tensor = value.detach().to(device="cpu").contiguous()
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def _validate_optimizer_replica_coherence(self, state: AdapterState) -> None:
        """Optionally verify shared Adam state agrees across EP replicas."""

        if (
            not self._replica_validation_enabled()
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
        ):
            return
        local_payload = []
        local_descriptor_records = [
            (layout.replica_key, layout.replica_count)
            for layout in state.tensor_layouts.values()
            if layout.needs_ep_gradient_sync
        ]
        for name in sorted(state.local_params):
            layout = state.tensor_layouts[name]
            if not layout.needs_ep_gradient_sync:
                continue
            optimizer_state = state.optimizer.state.get(state.local_params[name], {})
            local_payload.append((layout.replica_key, self._optimizer_state_digest(optimizer_state)))

        gathered_descriptors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_descriptors, local_descriptor_records)
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_payload)
        global_keys = sorted({tuple(key) for rank_records in gathered_descriptors for key, _ in rank_records})
        descriptor_counts: dict[tuple[Any, ...], int] = {}
        descriptor_expected: dict[tuple[Any, ...], set[int]] = {}
        for rank_records in gathered_descriptors:
            for key, replica_count in rank_records:
                key = tuple(key)
                descriptor_counts[key] = descriptor_counts.get(key, 0) + 1
                descriptor_expected.setdefault(key, set()).add(int(replica_count))
        by_key: dict[tuple[Any, ...], list[str]] = {}
        for rank_payload in gathered:
            for key, digest in rank_payload:
                by_key.setdefault(tuple(key), []).append(digest)
        checked_rectangles = 0
        checked_replica_values = 0
        for key in global_keys:
            expected_counts = descriptor_expected.get(key, set())
            expected_count = next(iter(expected_counts)) if len(expected_counts) == 1 else -1
            if expected_count < 0 or descriptor_counts.get(key, 0) != expected_count:
                raise RuntimeError(
                    f"Optimizer replica descriptor universe diverged for {key[0]!r}: "
                    f"present={descriptor_counts.get(key, 0)} expected={sorted(expected_counts)}"
                )
            if expected_count <= 1:
                continue
            checked_rectangles += 1
            checked_replica_values += expected_count
            digests = by_key.get(key, [])
            if len(digests) != expected_count:
                raise RuntimeError(f"Optimizer replica value count changed for {key[0]!r}: {len(digests)}")
            if any(value != digests[0] for value in digests[1:]):
                raise RuntimeError(f"Optimizer state diverged for logical rectangle {key[0]!r}")

        if torch.distributed.get_rank() == 0:
            logger.info(
                "Adapter EP optimizer-state coherence: "
                + json.dumps(
                    {
                        "phase": "optimizer_state",
                        "step": state.global_step + 1,
                        "passed": True,
                        "descriptor_universe_size": len(global_keys),
                        "checked_rectangles": checked_rectangles,
                        "checked_replica_values": checked_replica_values,
                    },
                    sort_keys=True,
                )
            )

    @staticmethod
    def _rectangles_overlap(
        left_offset: tuple[int, ...],
        left_shape: tuple[int, ...],
        right_offset: tuple[int, ...],
        right_shape: tuple[int, ...],
    ) -> bool:
        return all(
            max(left_start, right_start) < min(left_start + left_size, right_start + right_size)
            for left_start, left_size, right_start, right_size in zip(
                left_offset, left_shape, right_offset, right_shape, strict=True
            )
        )

    def _validate_ep_owner_layout(self, state: AdapterState) -> None:
        """Verify that disjoint EP-owned rectangles never overlap across ranks.

        Shared factors are intentionally excluded: their explicit ``EP_SUM``
        reduction domain means identical rectangles are replicas. Expert
        factors, by contrast, have ``gradient_reduction=NONE`` and must have
        disjoint active logical rectangles, including when eFSDP subdivides a
        local expert rectangle.

        This is an owner-layout check only. It does not prove that gradient
        values were not numerically contaminated across owners; that requires
        the Tier-B expert-pattern test.
        """

        if (
            not self._replica_validation_enabled()
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
        ):
            return

        local_records = []
        for name in sorted(state.local_params):
            layout = state.tensor_layouts[name]
            if not layout.is_ep_owned or layout.gradient_reduction.value != "none" or not layout.has_active_storage:
                continue
            local_records.append(
                (
                    layout.fqn,
                    tuple(layout.active_global_offset),
                    tuple(layout.active_storage_shape),
                    tuple(layout.replica_key),
                )
            )

        gathered: list[list[tuple[Any, ...]] | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_records)
        by_fqn: dict[str, list[tuple[tuple[int, ...], tuple[int, ...], tuple[Any, ...], int]]] = {}
        for rank, records in enumerate(gathered):
            for fqn, offset, shape, replica_key in records or []:
                by_fqn.setdefault(str(fqn), []).append((tuple(offset), tuple(shape), tuple(replica_key), rank))

        overlaps: list[dict[str, Any]] = []
        for fqn, records in by_fqn.items():
            for index, (left_offset, left_shape, left_key, left_rank) in enumerate(records):
                for right_offset, right_shape, right_key, right_rank in records[index + 1 :]:
                    if self._rectangles_overlap(left_offset, left_shape, right_offset, right_shape):
                        overlaps.append(
                            {
                                "fqn": fqn,
                                "left_rank": left_rank,
                                "left_offset": left_offset,
                                "left_shape": left_shape,
                                "right_rank": right_rank,
                                "right_offset": right_offset,
                                "right_shape": right_shape,
                                "left_replica_key": left_key,
                                "right_replica_key": right_key,
                            }
                        )
        if overlaps:
            raise RuntimeError(
                "Cross-owner EP adapter rectangle overlap detected: " + json.dumps(overlaps, sort_keys=True)
            )
        if torch.distributed.get_rank() == 0:
            logger.info(
                "Adapter EP owner layout: "
                + json.dumps(
                    {
                        "phase": "owner_layout",
                        "step": state.global_step + 1,
                        "passed": True,
                        "cross_owner_contamination": "not_tested",
                        "checked_rectangles": sum(len(records) for records in by_fqn.values()),
                        "checked_fqns": len(by_fqn),
                    },
                    sort_keys=True,
                )
            )

    @staticmethod
    def _factor_kind(name: str) -> str | None:
        if "lora_A" in name or "_lora_A" in name:
            return "A"
        if "lora_B" in name or "_lora_B" in name:
            return "B"
        return None

    def _capture_factor_movement_probes(self, state: AdapterState) -> dict[str, tuple[str, torch.Tensor]]:
        """Clone one shared A/B rectangle for cheap per-step movement evidence."""

        candidates: dict[str, list[tuple[str, AdapterTensorLayout, nn.Parameter]]] = {"A": [], "B": []}
        for name in sorted(state.local_params):
            kind = self._factor_kind(name)
            if kind is None or state.local_params[name].numel() == 0:
                continue
            candidates[kind].append((name, state.tensor_layouts[name], state.local_params[name]))
        probes: dict[str, tuple[str, torch.Tensor]] = {}
        for kind, values in candidates.items():
            values.sort(key=lambda item: (not item[1].needs_ep_gradient_sync, item[0]))
            if values:
                name, _, parameter = values[0]
                probes[kind] = (name, parameter.detach().clone())
        return probes

    def _log_factor_movement(
        self,
        state: AdapterState,
        probes: dict[str, tuple[str, torch.Tensor]],
    ) -> None:
        """Record A/B probe movement after a successful optimizer update."""

        if not probes:
            return
        local: dict[str, dict[str, Any]] = {}
        for kind, (name, before) in probes.items():
            after = state.local_params[name].detach()
            delta = after.to(torch.float32) - before.to(torch.float32)
            local[kind] = {
                "name": name,
                "before_norm": float(torch.linalg.vector_norm(before.to(torch.float32)).item()),
                "after_norm": float(torch.linalg.vector_norm(after.to(torch.float32)).item()),
                "delta_norm": float(torch.linalg.vector_norm(delta).item()),
                "changed": not torch.equal(before, after),
            }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[dict[str, dict[str, Any]] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return
        record: dict[str, Any] = {
            "phase": "movement",
            "step": state.global_step + 1,
            "passed": True,
        }
        for kind in ("A", "B"):
            values = [item[kind] for item in gathered if item is not None and kind in item]
            if not values:
                record[kind] = {"present": False}
                continue
            record[kind] = {
                "present": True,
                "name": values[0]["name"],
                "before_norm": values[0]["before_norm"],
                "after_norm": values[0]["after_norm"],
                "delta_norm": max(value["delta_norm"] for value in values),
                "changed_ranks": sum(bool(value["changed"]) for value in values),
                "rank_count": len(values),
            }
        logger.info("Adapter LoRA factor movement: " + json.dumps(record, sort_keys=True))

    @staticmethod
    def _is_lora_b(name: str) -> bool:
        return "lora_B" in name or "_lora_B" in name

    def _base_initialization_seed(self, session_spec: Dict[str, Any]) -> int:
        for source in (session_spec.get("lora_config", {}), self.lora_config):
            for key in ("seed", "lora_seed", "base_seed"):
                if key in source and source[key] is not None:
                    return int(source[key])
        return 0

    def _validate_session_rank_against_model_capacity(self, session_spec: Dict[str, Any]) -> None:
        """Reject session specs whose runtime rank exceeds the live model capacity."""
        session_rank = self._session_rank(session_spec)
        max_supported_rank = self._max_supported_session_rank()
        if session_rank > max_supported_rank:
            raise ValueError(
                f"Session rank {session_rank} exceeds live model LoRA capacity {max_supported_rank}. "
                "Restart the server with a larger max_lora_rank-compatible model substrate before loading this checkpoint."
            )

    @staticmethod
    def _module_name_for_lora_param(name: str) -> str:
        """Extract the target module name from an internal LoRA parameter name."""
        base_name = (
            name.replace(".lora_A.weight", "")
            .replace(".lora_B.weight", "")
            .replace(".lora_A", "")
            .replace(".lora_B", "")
            .replace("_lora_A", "")
            .replace("_lora_B", "")
        )
        parts = base_name.split(".")
        if not parts:
            raise ValueError(f"Cannot infer target module from LoRA parameter name {name!r}")
        return parts[-1]

    @staticmethod
    def _canonical_lora_param_name(name: str) -> str:
        """Normalize LoRA parameter names across checkpoint formats."""
        if name.endswith(".weight"):
            return name[: -len(".weight")]
        return name

    def _expected_target_modules(self) -> List[str]:
        """Return the live model's expected LoRA target modules."""
        return sorted(
            {
                self._module_name_for_lora_param(name)
                for name in self._lora_param_names
                if "lora_A" in name or "lora_B" in name
            }
        )

    def _validate_checkpoint_adapter_config(self, path: str) -> None:
        """Validate checkpoint-level adapter structure against the live model configuration."""
        adapter_config_path = os.path.join(path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            return

        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        checkpoint_target_modules = adapter_config.get("target_modules")
        if checkpoint_target_modules is not None:
            actual_target_modules = sorted(str(module) for module in checkpoint_target_modules)
            expected_target_modules = self._expected_target_modules()
            if actual_target_modules != expected_target_modules:
                raise ValueError(
                    "Checkpoint target_modules do not match the live LoRA adapter structure. "
                    f"checkpoint={actual_target_modules!r}, live={expected_target_modules!r}"
                )

        if "moe_hybrid_shared_lora" in adapter_config:
            checkpoint_hybrid = bool(adapter_config["moe_hybrid_shared_lora"])
            expected_hybrid = bool(self.lora_config.get("moe_hybrid_shared_lora", False))
            if checkpoint_hybrid != expected_hybrid:
                raise ValueError(
                    "Checkpoint moe_hybrid_shared_lora does not match the live LoRA adapter structure. "
                    f"checkpoint={checkpoint_hybrid!r}, live={expected_hybrid!r}"
                )

        live_manifest = load_lora_target_manifest(self.lora_config.get("lora_target_manifest"))
        checkpoint_manifest_path = os.path.join(path, _TARGET_MANIFEST_FILENAME)
        if live_manifest is not None and not os.path.exists(checkpoint_manifest_path):
            raise ValueError(
                f"Checkpoint is missing {_TARGET_MANIFEST_FILENAME} required by the live strict LoRA configuration"
            )
        if os.path.exists(checkpoint_manifest_path):
            checkpoint_manifest = load_lora_target_manifest(checkpoint_manifest_path)
            if live_manifest is None:
                raise ValueError(
                    f"Checkpoint contains {_TARGET_MANIFEST_FILENAME} but the live server has no strict LoRA manifest"
                )
            if checkpoint_manifest != live_manifest:
                raise ValueError("Checkpoint LoRA target manifest does not match the live strict LoRA configuration")

    def get_optimizer_metadata(self) -> Dict[str, Any]:
        """Return a JSON-safe description of the adapter optimizer contract."""
        return {
            "type": self.optimizer_type,
            "dtype": self.optimizer_dtype,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "eps": self.eps,
            "optimizer_kwargs": self._serialize_optimizer_metadata_value(self.optimizer_kwargs),
        }

    def get_adapter_session_spec(self, model_id: str) -> Dict[str, Any]:
        """Return the normalized session spec for an adapter."""
        return deepcopy(self.get_adapter_state(model_id).session_spec)

    def _legacy_session_spec(self, *, lr: float) -> Dict[str, Any]:
        """Build a session spec for compatibility call sites that only provide lr."""
        default_rank = self.lora_config.get("lora_rank")
        if default_rank is None and self._lora_param_names:
            metadata = self._lora_param_metadata[self._lora_param_names[0]]
            default_rank = metadata["shape"][metadata["rank_dim"]]
        default_alpha = self.lora_config.get("lora_alpha", default_rank or 16)
        # Start from the manager-level optimizer_config so passthrough flags
        # like cautious_weight_decay reach build_optimizer; structured fields
        # below override anything the manager-level dict supplies.
        optimizer_config: Dict[str, Any] = dict(self.optimizer_config or {})
        weight_decay = optimizer_config.get("weight_decay", self.weight_decay)
        optimizer_config.update(
            {
                "type": self.optimizer_type,
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "optimizer_dtype": self.optimizer_dtype,
                "betas": list(self.betas),
                "eps": float(self.eps),
                "optimizer_kwargs": self._serialize_optimizer_metadata_value(self.optimizer_kwargs),
            }
        )
        return {
            "base_model": self.lora_config.get("base_model", ""),
            "is_lora": True,
            "lora_config": {
                "lora_rank": int(default_rank or 32),
                "lora_alpha": int(default_alpha),
            },
            "optimizer_config": optimizer_config,
        }

    def _set_model_runtime_lora_config(self, *, lora_rank: int, lora_alpha: int) -> None:
        """Update all model-side LoRA modules to use the active session rank/alpha."""
        for module in self.model.modules():
            setter = getattr(module, "set_runtime_lora_config", None)
            if setter is not None:
                setter(lora_rank, lora_alpha)

    @staticmethod
    def _accumulate_adapter_gradient(
        adapter_param: nn.Parameter,
        gradient: torch.Tensor,
        *,
        accumulate: bool,
    ) -> None:
        """Store an adapter gradient using the parameter's dtype and device policy."""

        gradient = gradient.detach().to(device=adapter_param.device, dtype=adapter_param.dtype)
        if not accumulate or adapter_param.grad is None:
            adapter_param.grad = gradient
        else:
            adapter_param.grad.add_(gradient)

    @staticmethod
    def _build_parameter_module(lora_params: Dict[str, nn.Parameter]) -> nn.Module:
        """Wrap an adapter's parameters in a temporary module with stable parameter names."""
        root = nn.Module()
        for full_name, param in lora_params.items():
            current = root
            parts = full_name.split(".")
            for part in parts[:-1]:
                child = current._modules.get(part)
                if child is None:
                    child = nn.Module()
                    current.add_module(part, child)
                current = child

            leaf_name = parts[-1]
            if leaf_name in current._parameters:
                raise ValueError(f"Duplicate parameter name while building adapter optimizer module: {full_name}")
            current.register_parameter(leaf_name, param)
        return root

    @staticmethod
    def _optimizer_parameter_map(local_params: Dict[str, nn.Parameter]) -> Dict[str, nn.Parameter]:
        """Exclude deterministic empty intersections from fused optimizers."""

        return {
            name: local_params[name]
            for name in _canonical_parameter_order(local_params)
            if local_params[name].numel() > 0
        }

    def _build_adapter_optimizer(self, lora_params: Dict[str, nn.Parameter], lr: float) -> torch.optim.Optimizer:
        """Build an optimizer for one adapter via the shared optimizer factory."""
        adapter_module = self._build_parameter_module(self._optimizer_parameter_map(lora_params))
        return build_optimizer(
            adapter_module,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            fused=self.optimizer_fused,
            optimizer_type=self.optimizer_type,
            optimizer_dtype=self.optimizer_dtype,
            optimizer_kwargs=deepcopy(self.optimizer_kwargs),
        )

    def _build_adapter_optimizer_for_session(
        self, lora_params: Dict[str, nn.Parameter], session_spec: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        adapter_module = self._build_parameter_module(self._optimizer_parameter_map(lora_params))
        build_kwargs = session_optimizer_build_kwargs(session_spec["optimizer_config"])
        return build_optimizer(
            adapter_module,
            fused=self.optimizer_fused,
            **build_kwargs,
        )

    @staticmethod
    def _has_pending_gradients(state: AdapterState) -> bool:
        """Return whether an adapter has captured gradients awaiting an optimizer step."""
        return any(param.grad is not None for param in state.local_params.values())

    def _maybe_evict(self) -> Optional[str]:
        """
        Evict the least recently used adapter if at capacity.

        Adapters with pending gradients are not evictable because checkpointing
        them would silently drop the captured gradients before `optim_step`.
        If every resident adapter has pending gradients, this raises instead of
        discarding training state.

        If auto_save_on_eviction is enabled, saves the adapter state before evicting.

        Returns:
            The model_id of the evicted adapter, or None if no eviction was needed.
        """
        if len(self.adapters) >= self.max_adapters:
            if not self.adapters:
                return None
            _rank, world = _optimizer_shard_rank_world()
            if world > 1:
                raise RuntimeError(
                    "Automatic LoRA adapter eviction is disabled for multi-rank training because the "
                    "noncollective eviction path cannot preserve every rank's optimizer shard. Increase "
                    "max_adapters or save the adapter collectively before removing it."
                )
            evictable_ids = [
                model_id for model_id, state in self.adapters.items() if not self._has_pending_gradients(state)
            ]
            if not evictable_ids:
                raise RuntimeError(
                    "Cannot evict any adapter safely because all resident adapters have pending gradients. "
                    "Call optim_step for at least one session before loading or creating another adapter."
                )

            # Find the LRU adapter among the clean (step-complete) adapters.
            lru_id = min(evictable_ids, key=lambda k: self.adapters[k].last_access_time)
            logger.info(f"Evicting LRU adapter: {lru_id} (capacity {len(self.adapters)}/{self.max_adapters})")

            # Auto-save before eviction if enabled
            if self.auto_save_on_eviction:
                eviction_path = os.path.join(self.checkpoint_dir, "evicted", lru_id)
                self.save_adapter_state(lru_id, eviction_path)
                logger.info(f"Auto-saved adapter {lru_id} before eviction to {eviction_path}")

            self.remove_adapter(lru_id)
            return lru_id
        return None

    def register_adapter(
        self,
        model_id: str,
        lr: Optional[float] = None,
        session_spec: Optional[Dict[str, Any]] = None,
        initialize_fresh: bool = True,
        local_group_memberships: Mapping[str, tuple[int, ...]] | None = None,
    ) -> Dict[str, tuple[tuple[int, ...], ...]]:
        """
        Register a new LoRA adapter for a model_id.

        Creates new nn.Parameter objects and a new optimizer for this adapter.
        If at capacity, evicts the least recently used adapter first.

        Args:
            model_id: Unique identifier for this training run
            lr: Optional learning rate override for legacy call sites
            session_spec: Normalized session runtime spec for this adapter
            initialize_fresh: If True, initialize with fresh random weights.
                            If False, use the current model's LoRA weights.
            local_group_memberships: This rank's public replica groups, folded
                into the existing distributed layout-discovery exchange.

        Returns:
            The complete group families discovered by that exchange for static
            ownership-plan compilation.
        """
        model_id = validate_identifier(model_id, name="model_id")
        effective_lr = float(lr) if lr is not None else None
        if session_spec is None:
            if effective_lr is None:
                effective_lr = 1e-5
            session_spec = self._legacy_session_spec(lr=effective_lr)
        else:
            session_spec = deepcopy(session_spec)
            if effective_lr is not None:
                session_spec["optimizer_config"]["learning_rate"] = effective_lr

        self._validate_session_rank_against_model_capacity(session_spec)
        session_rank = self._session_rank(session_spec)
        session_alpha = self._session_alpha(session_spec)
        optimizer_config = session_spec["optimizer_config"]
        effective_lr = float(optimizer_config["learning_rate"])
        optimizer_type = str(optimizer_config.get("type", self.optimizer_type)).lower()
        _rank, world = _optimizer_shard_rank_world()
        if world > 1 and optimizer_type in {"muon", "distsignsgd", "dist_signsgd"}:
            raise ValueError(
                f"Optimizer {optimizer_type!r} is not shard-separable for distributed multi-adapter state. "
                "Use AdamW/AnyPrecisionAdamW/SGD/SignSGD, or run this optimizer in a single-rank session."
            )

        # Evict LRU adapter if at capacity and this is a new adapter
        if model_id not in self.adapters:
            self._maybe_evict()
        else:
            logger.info(f"Replacing existing adapter for model_id={model_id}")

        layouts, layout_fp, group_memberships = self._discover_layouts(
            session_rank,
            local_group_memberships=local_group_memberships,
        )
        named_params = dict(self.model.named_parameters())
        local_params: Dict[str, nn.Parameter] = {}
        base_seed = self._base_initialization_seed(session_spec)
        for name in self._lora_param_names:
            model_param = named_params[name]
            layout = layouts[name]
            if initialize_fresh:
                new_tensor = deterministic_local_initialization(
                    layout,
                    base_seed=base_seed,
                    session_identity=model_id,
                    is_lora_b=self._is_lora_b(name),
                ).to(device=self.device, dtype=layout.dtype)
            else:
                raw_model_param = model_param.data if isinstance(model_param, nn.Parameter) else model_param
                local_model_tensor = (
                    raw_model_param.to_local()
                    if _HAS_DTENSOR and isinstance(raw_model_param, DTensor)
                    else raw_model_param
                )
                local_model_tensor = wait_for_local_tensor(local_model_tensor)
                new_tensor = layout.pack_from_local(local_model_tensor).to(device=self.device, dtype=layout.dtype)
            local_params[name] = nn.Parameter(new_tensor, requires_grad=True)

        # Build optimizer for this adapter using the session's optimizer contract.
        optimizer = self._build_adapter_optimizer_for_session(local_params, session_spec)
        registration_ordinal = self._adapter_registration_ordinals.get(model_id, 0) + 1
        self._adapter_registration_ordinals[model_id] = registration_ordinal

        self.adapters[model_id] = AdapterState(
            model_id=model_id,
            session_spec=session_spec,
            local_params=local_params,
            tensor_layouts=layouts,
            layout_fingerprint=layout_fp,
            optimizer=optimizer,
            registration_ordinal=registration_ordinal,
            global_step=0,
            global_forward_backward_step=0,
            lr=effective_lr,
        )

        logger.info(
            f"Registered adapter for model_id={model_id} "
            f"(rank={session_rank}, alpha={session_alpha}, lr={effective_lr}, "
            f"fresh_weights={initialize_fresh}, num_params={len(local_params)}, "
            f"optimizer={optimizer_config['type']})"
        )
        return group_memberships

    def compile_gradient_ownership_plan(
        self,
        model_id: str,
        declarations: Mapping[str, ParameterOwnershipDeclaration],
        *,
        model_generation: str,
        adapter_generation: str,
        tensor_parallel_size: int = 1,
        group_memberships: Mapping[str, tuple[tuple[int, ...], ...]] | None = None,
        rank: int = 0,
    ) -> AdapterGradientOwnershipPlan:
        """Compile or explicitly recompile one adapter's immutable ownership plan.

        This registration-bound operation records ownership only. It does not
        capture gradients, execute collectives, or change optimizer behavior.
        Reconfiguration is rejected while captured gradients are pending.
        """

        model_id = validate_identifier(model_id, name="model_id")
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")
        state = self.adapters[model_id]
        if self._has_pending_gradients(state) or state.gradient_scratch.next_capture_ordinal:
            raise RuntimeError("Cannot recompile adapter gradient ownership while gradients are pending")
        model_parameters = {
            name: parameter for name, parameter in self.model.named_parameters() if name in state.tensor_layouts
        }
        plan = compile_adapter_gradient_ownership(
            layouts=state.tensor_layouts,
            model_parameters=model_parameters,
            optimizer_parameters=state.local_params,
            declarations=declarations,
            model_generation=model_generation,
            adapter_generation=adapter_generation,
            tensor_parallel_size=tensor_parallel_size,
            group_memberships=group_memberships,
            rank=rank,
        )
        self._agree_gradient_ownership_fingerprint(plan)
        state.gradient_ownership_plan = plan
        self._preallocate_gradient_scratch(state, plan)
        logger.info(
            "Compiled adapter gradient ownership for model_id=%s fingerprint=%s",
            model_id,
            plan.fingerprint,
        )
        return plan

    @staticmethod
    def _preallocate_gradient_scratch(
        state: AdapterState,
        plan: AdapterGradientOwnershipPlan,
    ) -> None:
        """Allocate the one persistent FP32 numerator image on the cold path."""

        local_by_fqn = {canonical_parameter_name(name): parameter for name, parameter in state.local_params.items()}
        expected = {item.fqn for item in plan.parameters}
        if set(local_by_fqn) != expected:
            raise AdapterGradientOwnershipError("Optimizer parameter universe differs from the compiled ownership plan")
        existing = state.gradient_scratch.numerators
        reusable = set(existing) == expected and all(
            tuple(existing[fqn].shape) == tuple(local_by_fqn[fqn].shape)
            and existing[fqn].dtype is torch.float32
            and existing[fqn].device == local_by_fqn[fqn].device
            for fqn in expected
        )
        if not reusable:
            if state.gradient_scratch.next_capture_ordinal or state.gradient_scratch.denominator:
                raise AdapterGradientOwnershipError(
                    "Cannot replace persistent adapter-gradient scratch while an epoch is active"
                )
            state.gradient_scratch.numerators = {
                item.fqn: torch.zeros_like(local_by_fqn[item.fqn], dtype=torch.float32) for item in plan.parameters
            }
        else:
            for tensor in existing.values():
                tensor.zero_()

    def _clear_model_adapter_gradients(self, state: AdapterState) -> None:
        owned_fqns = {canonical_parameter_name(name) for name in state.tensor_layouts}
        for name, parameter in self.model.named_parameters():
            if canonical_parameter_name(name) in owned_fqns:
                parameter.grad = None

    def _clear_all_adapter_gradients(self, state: AdapterState) -> None:
        self._clear_model_adapter_gradients(state)
        for parameter in state.local_params.values():
            parameter.grad = None

    @staticmethod
    def _reset_gradient_scratch(state: AdapterState) -> None:
        """Reset epoch metadata while preserving the preallocated image."""

        scratch = state.gradient_scratch
        for numerator in scratch.numerators.values():
            numerator.zero_()
        scratch.epoch = state.global_step
        scratch.next_capture_ordinal = 0
        scratch.denominator = 0.0
        scratch.numerator_scale = None
        scratch.source = None
        scratch.capture_open = False
        scratch.capture_staged = False
        scratch.staged_denominator = 0.0
        scratch.staged_numerator_scale = None
        scratch.staged_parameter_fqns = ()
        scratch.staged_numerators.clear()

    def validate_weight_publication(self, model_id: str) -> None:
        """Admit a stable weight-only surface, including a clean mid-epoch state."""

        state = self.get_adapter_state(model_id)
        if state.poisoned:
            raise RuntimeError("A poisoned adapter session cannot publish weights; restart from checkpoint")
        if state.publication_pending:
            raise RuntimeError("Adapter weights cannot be published before optimizer command completion")

    def validate_strict_checkpoint_publication(self, model_id: str) -> None:
        """Admit optimizer/lifecycle checkpoints only at a completed epoch boundary."""

        self.validate_weight_publication(model_id)
        if not self.get_adapter_state(model_id).publication_eligible:
            raise RuntimeError(
                "Adapter state is not checkpoint-publication-eligible; abort or complete the optimizer epoch"
            )

    def begin_gradient_capture(
        self,
        model_id: str,
        *,
        scale_state: GradientScaleState,
    ) -> bool:
        """Open one capture ordinal under the immutable registered plan."""

        state = self.get_adapter_state(model_id)
        if state.publication_pending:
            raise AdapterGradientOwnershipError(
                "Cannot begin another gradient epoch before the distributed optimizer command commits"
            )
        plan = state.gradient_ownership_plan
        if plan is None:
            raise AdapterGradientOwnershipError("Adapter gradient ownership requires a compiled plan")
        if scale_state is not GradientScaleState.RAW_NUMERATOR:
            raise AdapterGradientUniformRejection(
                "ADAPTER_GRADIENT_PRENORMALIZED_LOSS: authoritative ownership requires unnormalized raw numerator losses"
            )
        scratch = state.gradient_scratch
        if scratch.capture_open:
            raise AdapterGradientOwnershipError("A gradient capture ordinal is already open")
        if scratch.epoch != state.global_step:
            if scratch.numerators or scratch.denominator or scratch.next_capture_ordinal:
                raise AdapterGradientOwnershipError("Stale adapter-gradient scratch crossed an optimizer epoch")
            scratch.epoch = state.global_step
        scratch.capture_open = True
        state.publication_eligible = False
        return True

    def abort_gradient_capture(self, model_id: str) -> None:
        """Discard one uncommitted capture and clear model-side gradients."""

        state = self.get_adapter_state(model_id)
        scratch = state.gradient_scratch
        self._clear_model_adapter_gradients(state)
        scratch.capture_open = False
        scratch.capture_staged = False
        scratch.staged_denominator = 0.0
        scratch.staged_numerator_scale = None
        scratch.staged_parameter_fqns = ()
        scratch.staged_numerators.clear()

    def gradient_capture_is_open(self, model_id: str) -> bool:
        """Return whether one model call still owns an uncommitted capture."""

        return self.get_adapter_state(model_id).gradient_scratch.capture_open

    def abort_gradient_epoch(self, model_id: str) -> None:
        """Idempotently discard one unmutated epoch and restore stable weights."""

        state = self.get_adapter_state(model_id)
        if state.poisoned:
            raise AdapterGradientMutationFailure(
                "A poisoned adapter session cannot be recovered in-process; restart from checkpoint"
            )
        if state.publication_pending:
            raise AdapterGradientMutationFailure("Cannot abort an adapter epoch while optimizer publication is pending")
        self._clear_all_adapter_gradients(state)
        self._reset_gradient_scratch(state)
        state.publication_eligible = True

    def adapter_sync_exclusions(self, model_id: str, axis: ReductionAxis) -> frozenset[int]:
        """Return live model parameters owned by the compiled finalizer."""
        state = self.get_adapter_state(model_id)
        plan = state.gradient_ownership_plan
        if plan is None:
            raise AdapterGradientOwnershipError("Authoritative ownership requires a compiled plan")
        owned_fqns = {
            fqn
            for mask in plan.authority_masks
            if mask.axis is axis and mask.authority is ReductionAuthority.ADAPTER_FINALIZER
            for fqn in mask.fqns
        }
        model_parameters = {
            canonical_parameter_name(name): parameter for name, parameter in self.model.named_parameters()
        }
        return frozenset(id(model_parameters[fqn]) for fqn in owned_fqns)

    def _capture_local_gradient(self, state: AdapterState, item, parameter: nn.Parameter) -> torch.Tensor:
        """Return the completed public-boundary representation without mutating `.grad`."""

        raw_gradient = parameter.grad
        assert raw_gradient is not None
        if item.capture_domains:
            parameter_data = parameter.data
            if (
                item.representation is not GradientRepresentation.DIRECT_DTENSOR_CONTRIBUTION
                or len(item.capture_domains) != 1
                or item.capture_domains[0].axis is not ReductionAxis.FSDP_SHARD
                or item.capture_domains[0].authority is not ReductionAuthority.ADAPTER_CAPTURE
                or not (_HAS_DTENSOR and isinstance(raw_gradient, DTensor) and isinstance(parameter_data, DTensor))
            ):
                raise AdapterGradientOwnershipError(f"Unsupported adapter capture reduction for {item.fqn!r}")
            try:
                raw_gradient = raw_gradient.redistribute(
                    device_mesh=parameter_data.device_mesh,
                    placements=parameter_data.placements,
                )
            except Exception as error:
                state.poisoned = True
                raise AdapterGradientCollectiveFailure(
                    "Adapter-gradient DTensor placement collective failed; restart the distributed process"
                ) from error
        return wait_for_local_tensor(
            raw_gradient.to_local() if _HAS_DTENSOR and isinstance(raw_gradient, DTensor) else raw_gradient
        )

    def stage_gradient_numerators(
        self,
        model_id: str,
        *,
        denominator: float,
        numerator_scale: float = 1.0,
        backward_completed: bool,
    ) -> None:
        """Validate and stage one call without mutating committed FP32 scratch."""

        state = self.get_adapter_state(model_id)
        plan = state.gradient_ownership_plan
        if plan is None:
            raise AdapterGradientOwnershipError("Adapter gradient ownership is not compiled")
        scratch = state.gradient_scratch
        if not scratch.capture_open:
            raise AdapterGradientOwnershipError("No gradient capture ordinal is open")
        if scratch.capture_staged:
            raise AdapterGradientOwnershipError("The current gradient capture is already staged")
        if not backward_completed:
            raise AdapterGradientOwnershipError("Gradient capture requires a completed backward pass")
        if not math.isfinite(float(denominator)) or denominator <= 0:
            raise AdapterGradientUniformRejection(
                "ADAPTER_GRADIENT_ZERO_DENOMINATOR: globally reduced valid-token denominator must be positive"
            )
        if not math.isfinite(float(numerator_scale)) or numerator_scale <= 0:
            raise AdapterGradientOwnershipError("Gradient numerator scale must be finite and positive")
        if scratch.source not in {None, "model"}:
            raise AdapterGradientOwnershipError("Gradient scratch contains an unsupported source")
        if scratch.numerator_scale is not None and scratch.numerator_scale != float(numerator_scale):
            raise AdapterGradientOwnershipError("Streamed captures use incompatible numerator scales")

        named_parameters = {
            canonical_parameter_name(name): parameter for name, parameter in self.model.named_parameters()
        }
        layouts = {layout.fqn: layout for layout in state.tensor_layouts.values()}
        staged_fqns: list[str] = []
        staged_numerators: dict[str, torch.Tensor] = {}
        for item in plan.parameters:
            parameter = named_parameters[item.fqn]
            if parameter.grad is None:
                if item.requires_local_gradient:
                    raise AdapterGradientOwnershipError(f"Required adapter gradient is absent for {item.fqn!r}")
                continue
            local_gradient = self._capture_local_gradient(state, item, parameter)
            layout = layouts[item.fqn]
            if tuple(local_gradient.shape) != layout.local_substrate_shape:
                raise AdapterGradientOwnershipError(
                    f"Local adapter gradient shape changed for {item.fqn!r}: "
                    f"actual={tuple(local_gradient.shape)} expected={layout.local_substrate_shape}"
                )
            if item.requires_local_gradient and not layout.has_active_storage:
                raise AdapterGradientOwnershipError(f"Required adapter gradient is empty for {item.fqn!r}")
            staged_fqns.append(item.fqn)
            packed = layout.pack_from_local(local_gradient).detach().float()
            staged_numerators[item.fqn] = packed

        scratch.capture_staged = True
        scratch.staged_denominator = float(denominator)
        scratch.staged_numerator_scale = float(numerator_scale)
        scratch.staged_parameter_fqns = tuple(staged_fqns)
        scratch.staged_numerators = staged_numerators

    def commit_gradient_capture(self, model_id: str) -> tuple[int, int]:
        """Commit one successfully completed call into the epoch accumulator."""

        state = self.get_adapter_state(model_id)
        plan = state.gradient_ownership_plan
        if plan is None:
            raise AdapterGradientOwnershipError("Adapter gradient ownership is not compiled")
        scratch = state.gradient_scratch
        if not scratch.capture_open or not scratch.capture_staged:
            raise AdapterGradientOwnershipError("No staged gradient capture is ready to commit")

        item_by_fqn = {item.fqn: item for item in plan.parameters}
        staged_fqns = scratch.staged_parameter_fqns
        staged_set = set(staged_fqns)
        if len(staged_set) != len(staged_fqns) or staged_set != set(scratch.staged_numerators):
            raise AdapterGradientOwnershipError("Staged adapter-gradient names are incomplete or duplicated")
        if set(scratch.numerators) != set(item_by_fqn):
            raise AdapterGradientOwnershipError("Persistent numerator image differs from the compiled plan")
        if not math.isfinite(scratch.staged_denominator) or scratch.staged_denominator <= 0:
            raise AdapterGradientOwnershipError("Staged gradient denominator must be finite and positive")
        if scratch.staged_numerator_scale is None or not math.isfinite(scratch.staged_numerator_scale):
            raise AdapterGradientOwnershipError("Staged gradient numerator scale is missing or nonfinite")
        for fqn, item in item_by_fqn.items():
            if item.requires_local_gradient and fqn not in staged_set:
                raise AdapterGradientOwnershipError(f"Required staged adapter gradient is absent for {fqn!r}")
            if fqn not in staged_set:
                continue
            staged = scratch.staged_numerators[fqn]
            destination = scratch.numerators[fqn]
            if (
                tuple(staged.shape) != tuple(destination.shape)
                or staged.dtype is not torch.float32
                or staged.device != destination.device
            ):
                raise AdapterGradientOwnershipError(f"Staged adapter gradient destination changed for {fqn!r}")
        try:
            for fqn in staged_fqns:
                packed = scratch.staged_numerators[fqn]
                accumulator = scratch.numerators[fqn]
                if scratch.next_capture_ordinal == 0:
                    accumulator.copy_(packed)
                else:
                    accumulator.add_(packed)
            capture_id = scratch.epoch, scratch.next_capture_ordinal
            scratch.next_capture_ordinal += 1
            scratch.denominator += scratch.staged_denominator
            scratch.numerator_scale = scratch.staged_numerator_scale
            scratch.source = "model"
            self._clear_model_adapter_gradients(state)
            return capture_id
        except AdapterGradientOwnershipError:
            raise
        except Exception as error:
            state.poisoned = True
            raise AdapterGradientCollectiveFailure(
                "Adapter-gradient capture commit failed; restart the distributed process"
            ) from error
        finally:
            scratch.capture_open = False
            scratch.capture_staged = False
            scratch.staged_denominator = 0.0
            scratch.staged_numerator_scale = None
            scratch.staged_parameter_fqns = ()
            scratch.staged_numerators.clear()

    def capture_gradient_numerators(
        self,
        model_id: str,
        *,
        denominator: float,
        numerator_scale: float = 1.0,
        backward_completed: bool,
    ) -> tuple[int, int]:
        """Stage and immediately commit one capture for direct manager callers."""

        try:
            self.stage_gradient_numerators(
                model_id,
                denominator=denominator,
                numerator_scale=numerator_scale,
                backward_completed=backward_completed,
            )
            return self.commit_gradient_capture(model_id)
        except BaseException:
            self.abort_gradient_capture(model_id)
            raise

    def prepare_forward(self, model_id: str) -> None:
        """
        Load adapter weights into model before forward pass.

        This must be called before forward() to ensure the model uses
        the correct adapter's weights.

        Adapter slots are compact local shards.  This method unpacks them into
        the model's local substrate and clears inactive maximum-rank regions.

        Args:
            model_id: The adapter to prepare for

        Raises:
            KeyError: If the adapter is not registered. Use register_adapter() first.
        """
        if model_id not in self.adapters:
            raise KeyError(
                f"Adapter for model_id={model_id} not registered. "
                "Call register_adapter() first or ensure the session is valid."
            )

        state = self.adapters[model_id]
        self.validate_weight_publication(model_id)
        self._validate_model_layout_identity(state)
        self._validate_replica_coherence(state, gradients=False)
        # Update last access time for LRU tracking
        state.last_access_time = time.time()
        self._set_model_runtime_lora_config(
            lora_rank=self._session_rank(state.session_spec),
            lora_alpha=self._session_alpha(state.session_spec),
        )

        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, param in named_params.items():
                if name not in state.local_params:
                    continue
                raw_param = param.data if isinstance(param, nn.Parameter) else param
                local_tensor = raw_param.to_local() if _HAS_DTENSOR and isinstance(raw_param, DTensor) else raw_param
                local_tensor = wait_for_local_tensor(local_tensor)
                state.tensor_layouts[name].unpack_to_local(state.local_params[name].data, destination=local_tensor)

        self.current_adapter_id = model_id

    def _ep_group(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        if not parallel_state.ep_enabled:
            return None
        return parallel_state.ep_group

    def _trace_adapter_gradient_stage(self, state: AdapterState, *, stage: str) -> None:
        ep_group = self._ep_group()
        if ep_group is None or torch.distributed.get_world_size(ep_group) <= 1:
            return
        values = {
            name: state.local_params[name].grad
            for name in _canonical_parameter_order(state.local_params)
            if state.tensor_layouts[name].needs_ep_gradient_sync
        }
        trace_replicated_gradient_stage(stage=stage, values=values, ep_group=ep_group)

    def trace_model_gradient_stage(self, model_id: str, *, stage: str) -> None:
        """Trace replicated model gradients before adapter capture, when enabled."""

        if not gradient_trace_enabled():
            return
        state = self.get_adapter_state(model_id)
        ep_group = self._ep_group()
        if ep_group is None or torch.distributed.get_world_size(ep_group) <= 1:
            return
        values = {}
        for name, param in self.model.named_parameters():
            layout = state.tensor_layouts.get(name)
            if layout is None or not layout.needs_ep_gradient_sync:
                continue
            if param.grad is None:
                values[name] = None
                continue
            raw_grad = param.grad
            local_grad = raw_grad.to_local() if _HAS_DTENSOR and isinstance(raw_grad, DTensor) else raw_grad
            values[name] = layout.pack_from_local(wait_for_local_tensor(local_grad))
        trace_replicated_gradient_stage(stage=stage, values=values, ep_group=ep_group)

    @staticmethod
    def _distributed_world() -> tuple[object | None, int]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None, 1
        group = torch.distributed.group.WORLD
        return group, torch.distributed.get_world_size(group=group)

    def _resolve_gradient_reduction_group(self, domain: ReductionDomainPlan):
        from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

        parallel_state = get_parallel_state()
        if domain.axis is ReductionAxis.SEQUENCE_PARALLEL:
            return parallel_state.sp_grad_sync_group
        if domain.axis is ReductionAxis.OUTPUT_PROJECTION_REPLICA:
            return parallel_state.lm_head_tp_replica_group
        if domain.axis is ReductionAxis.EXPERT_PARALLEL_REPLICA:
            return parallel_state.ep_group
        raise AdapterGradientOwnershipError(f"No residual transport resolver for axis {domain.axis.value!r}")

    def _validate_authoritative_state(self, state: AdapterState, *, allow_optimizer_gradients: bool = False) -> None:
        """Validate the local state against the immutable registered plan.

        The plan fingerprint is agreed across ranks when the adapter is
        registered or explicitly reconfigured.  The steady-state path does not
        run a second metadata consensus protocol around every optimizer step.
        """

        plan = state.gradient_ownership_plan
        if plan is None:
            raise AdapterGradientOwnershipError("Authoritative finalization requires a compiled plan")
        scratch = state.gradient_scratch
        if state.poisoned:
            raise AdapterGradientOwnershipError("Adapter session is poisoned and must recover from checkpoint")
        if scratch.capture_open:
            raise AdapterGradientOwnershipError("Cannot finalize an open gradient capture ordinal")
        if scratch.epoch != state.global_step or scratch.next_capture_ordinal <= 0:
            raise AdapterGradientOwnershipError("Gradient epoch or capture ordinal is stale or empty")
        if not math.isfinite(scratch.denominator) or scratch.denominator <= 0:
            raise AdapterGradientOwnershipError("Gradient denominator must be finite and positive")
        if scratch.numerator_scale is None or not math.isfinite(scratch.numerator_scale):
            raise AdapterGradientOwnershipError("Gradient numerator scale is missing or nonfinite")
        if scratch.source != "model":
            raise AdapterGradientOwnershipError("Compiled producer and captured gradient source disagree")
        scratch_by_fqn = {canonical_parameter_name(name): tensor for name, tensor in scratch.numerators.items()}
        local_by_fqn = {canonical_parameter_name(name): parameter for name, parameter in state.local_params.items()}
        for item in plan.parameters:
            numerator = scratch_by_fqn.get(item.fqn)
            if numerator is None:
                if item.requires_local_gradient:
                    raise AdapterGradientOwnershipError(f"Required numerator is absent for {item.fqn!r}")
                continue
            if tuple(numerator.shape) != tuple(local_by_fqn[item.fqn].shape):
                raise AdapterGradientOwnershipError(f"Numerator shape changed for {item.fqn!r}")
        if not allow_optimizer_gradients and any(
            parameter.grad is not None for parameter in state.local_params.values()
        ):
            raise AdapterGradientOwnershipError("Optimizer gradient slots were populated before finalization")

    def _authoritative_optim_step(
        self,
        state: AdapterState,
        *,
        lr: float,
        gradient_clip: Optional[float],
        norm_type: float,
    ) -> float:
        if float(norm_type) != 2.0:
            raise ValueError("Authoritative adapter finalization currently supports L2 norm only")
        self._validate_authoritative_state(state)
        self._validate_ep_owner_layout(state)
        plan = state.gradient_ownership_plan
        assert plan is not None
        scratch = state.gradient_scratch
        multiplier = float(scratch.numerator_scale) / float(scratch.denominator)
        templates = {name: parameter.detach() for name, parameter in state.local_params.items()}
        try:
            gradients, transport_stats = transport_complete_local_gradients(
                plan=plan,
                numerators=scratch.numerators,
                templates=templates,
                multiplier=multiplier,
                resolve_group=self._resolve_gradient_reduction_group,
                bucket_bytes=self.gradient_ownership_bucket_bytes,
                consume_numerators=True,
            )
            group, _world = self._distributed_world()
            norm = logical_l2_norm(plan, gradients, world_group=group)
        except AdapterGradientOwnershipError:
            raise
        except Exception as error:
            state.poisoned = True
            raise AdapterGradientCollectiveFailure(
                "Adapter-gradient data collective failed; restart the distributed process"
            ) from error

        grad_norm = float(norm.item())
        if not math.isfinite(grad_norm):
            raise AdapterGradientOwnershipError("Transport-complete adapter gradient is nonfinite")
        clip_coefficient = 1.0
        if gradient_clip is not None and gradient_clip > 0:
            clip_coefficient = min(1.0, float(gradient_clip) / (grad_norm + 1e-6))
        local_by_fqn = {canonical_parameter_name(name): parameter for name, parameter in state.local_params.items()}
        for item in plan.parameters:
            parameter = local_by_fqn[item.fqn]
            gradient = gradients[item.fqn]
            gradient.mul_(clip_coefficient)
            parameter.grad = gradient.to(dtype=parameter.dtype) if parameter.numel() > 0 else None

        state.publication_eligible = False
        try:
            # LR is persistent session/optimizer state.  Update it only after
            # transport, logical norm, nonfinite, and clipping validation have
            # all succeeded and immediately before the mutating optimizer call.
            self._update_state_learning_rate(state, lr)
            state.optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self._validate_replica_coherence(state, gradients=False)
            self._validate_optimizer_replica_coherence(state)
        except Exception as error:
            state.poisoned = True
            raise AdapterGradientMutationFailure(
                "Adapter optimizer mutation failed; publish nothing and recover from the last checkpoint"
            ) from error
        finally:
            try:
                state.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                state.optimizer.zero_grad()
            for parameter in state.local_params.values():
                parameter.grad = None

        state.global_step += 1
        state.last_transport_stats = transport_stats
        # Multi-rank publication is committed by RunnerDispatcher after its
        # existing optim_step error synchronization succeeds.  Keeping this
        # local step pending avoids adding an ownership-specific post-step
        # collective while ensuring a successful rank cannot publish early.
        state.publication_pending = _world > 1
        state.publication_eligible = not state.publication_pending
        self._reset_gradient_scratch(state)
        return grad_norm

    def commit_optimizer_publication(self, model_id: str) -> None:
        """Commit a completed distributed step at the server command boundary."""

        state = self.get_adapter_state(model_id)
        if not state.publication_pending:
            return
        if state.poisoned:
            raise AdapterGradientMutationFailure("A poisoned adapter step cannot become publication-eligible")
        scratch = state.gradient_scratch
        if scratch.capture_open or scratch.next_capture_ordinal or scratch.denominator or scratch.source is not None:
            raise AdapterGradientMutationFailure("Adapter publication commit found incomplete gradient scratch")
        state.publication_pending = False
        state.publication_eligible = True

    def optim_step(
        self,
        model_id: str,
        lr: float,
        gradient_clip: Optional[float] = None,
        norm_type: float = 2.0,
    ) -> float:
        """Finalize the compiled raw-numerator epoch and mutate its optimizer."""
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]
        self._validate_model_layout_identity(state)
        if state.gradient_ownership_plan is None:
            raise AdapterGradientOwnershipError("Adapter gradient ownership requires a compiled plan")
        try:
            return self._authoritative_optim_step(
                state,
                lr=lr,
                gradient_clip=gradient_clip,
                norm_type=norm_type,
            )
        except AdapterGradientOwnershipError:
            self.abort_gradient_epoch(model_id)
            raise

    def sync_weights_to_model(self, model_id: str) -> None:
        """
        Sync adapter weights to model (for save_lora_only, inference, etc).

        For FSDP2/DTensor: Same as prepare_forward, handles sharded params.

        Args:
            model_id: The adapter whose weights to sync
        """
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        # Just delegate to prepare_forward which handles DTensor properly
        self.prepare_forward(model_id)

    def get_adapter_state(self, model_id: str) -> AdapterState:
        """Get the adapter state for a model_id."""
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")
        return self.adapters[model_id]

    def get_current_adapter(self) -> Optional[AdapterState]:
        """Get the currently active adapter state."""
        if self.current_adapter_id is None:
            return None
        return self.adapters.get(self.current_adapter_id)

    def increment_forward_backward_step(self, model_id: str) -> int:
        """Increment and return the forward_backward step counter for an adapter."""
        state = self.adapters[model_id]
        state.global_forward_backward_step += 1
        return state.global_forward_backward_step

    def get_global_step(self, model_id: str) -> int:
        """Get the global step counter for an adapter."""
        return self.adapters[model_id].global_step

    def get_forward_backward_step(self, model_id: str) -> int:
        """Get the forward_backward step counter for an adapter."""
        return self.adapters[model_id].global_forward_backward_step

    def get_lr(self, model_id: str) -> float:
        """Get the learning rate for an adapter."""
        return self.adapters[model_id].lr

    def set_lr(self, model_id: str, lr: float) -> None:
        """Set the learning rate for an adapter."""
        state = self.adapters[model_id]
        self._update_state_learning_rate(state, lr)

    def has_adapter(self, model_id: str) -> bool:
        """Check if an adapter is registered for a model_id."""
        return model_id in self.adapters

    def list_adapters(self) -> List[str]:
        """List all registered model_ids."""
        return list(self.adapters.keys())

    def remove_adapter(self, model_id: str) -> None:
        """Remove an adapter for a model_id."""
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        if self.current_adapter_id == model_id:
            self.current_adapter_id = None

        del self.adapters[model_id]
        logger.info(f"Removed adapter for model_id={model_id}")

    # Legacy compatibility methods (for gradual migration)
    def switch_adapter(self, model_id: str, auto_register: bool = False) -> bool:
        """Legacy method - now just calls prepare_forward.

        Args:
            model_id: The adapter to switch to
            auto_register: If True, auto-register adapter if not found (default: False)

        Returns:
            True if successful

        Raises:
            KeyError: If adapter not registered and auto_register is False
        """
        if model_id not in self.adapters:
            if auto_register:
                logger.warning(f"Auto-registering adapter for model_id={model_id} (deprecated)")
                self.register_adapter(model_id, lr=1e-5, initialize_fresh=True)
            else:
                raise KeyError(f"Adapter for model_id={model_id} not registered")

        self.prepare_forward(model_id)
        return True

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Return memory usage per adapter in bytes.

        Includes both parameters and optimizer state (AdamW stores ~2x params).

        Returns:
            Dict mapping model_id to memory usage in bytes.
        """
        usage = {}
        for model_id, state in self.adapters.items():
            # Calculate parameter memory
            param_bytes = sum(p.numel() * p.element_size() for p in state.local_params.values())

            # Calculate optimizer state memory (AdamW stores exp_avg and exp_avg_sq)
            optim_bytes = 0
            for param_state in state.optimizer.state.values():
                for v in param_state.values():
                    if isinstance(v, torch.Tensor):
                        optim_bytes += v.numel() * v.element_size()

            usage[model_id] = param_bytes + optim_bytes
        return usage

    def iter_local_optimizer_params(self, model_id: str):
        """Iterate active non-empty local parameters in optimizer order."""

        state = self.get_adapter_state(model_id)
        for name in _canonical_parameter_order(state.local_params):
            param = state.local_params[name]
            if param.numel() > 0:
                yield name, param

    def get_layout(self, model_id: str, fqn: str) -> AdapterTensorLayout:
        """Return the captured layout for one adapter tensor."""

        state = self.get_adapter_state(model_id)
        try:
            return state.tensor_layouts[fqn]
        except KeyError:
            canonical = canonical_parameter_name(fqn)
            for name, layout in state.tensor_layouts.items():
                if layout.fqn == canonical:
                    return layout
            raise KeyError(f"No adapter layout for {fqn!r}") from None

    def materialize_logical_state_dict(self, model_id: str, *, destination_rank: int = 0) -> Dict[str, torch.Tensor]:
        """Collectively reconstruct full active logical weights for cold paths."""

        self.prepare_forward(model_id)
        from xorl.lora.utils import get_lora_state_dict  # noqa: PLC0415

        state_dict = get_lora_state_dict(self.model)
        rank, _world = _optimizer_shard_rank_world()
        return state_dict if rank == destination_rank else {}

    def load_logical_state_dict(self, model_id: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Pack full active logical tensors into this rank's local adapter slots."""

        state = self.get_adapter_state(model_id)
        packed = self._pack_logical_state_dict(state, state_dict)
        with torch.no_grad():
            for name, local_slot in packed.items():
                state.local_params[name].data.copy_(local_slot.to(self.device, state.local_params[name].dtype))

    def _pack_logical_state_dict(
        self, state: AdapterState, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Validate and pack an external logical state without mutating state."""

        expected_names = {canonical_parameter_name(name): name for name in state.local_params}
        expected_shapes = {
            canonical_parameter_name(name): layout.logical_shape for name, layout in state.tensor_layouts.items()
        }
        converted = convert_peft_lora_state_dict(state_dict, expected_shapes=expected_shapes)
        converted_by_name = {canonical_parameter_name(name): tensor for name, tensor in converted.items()}
        missing = sorted(set(expected_names) - set(converted_by_name))
        unexpected = sorted(set(converted_by_name) - set(expected_names))
        if missing or unexpected:
            raise ValueError(
                "Logical adapter parameter set does not match the live adapter structure. "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        packed: Dict[str, torch.Tensor] = {}
        for canonical_name, actual_name in expected_names.items():
            layout = state.tensor_layouts[actual_name]
            logical_tensor = converted_by_name[canonical_name]
            if tuple(logical_tensor.shape) == layout.logical_shape:
                local_slot = pack_logical_tensor(layout, logical_tensor)
            elif tuple(logical_tensor.shape) == layout.active_storage_shape:
                # Explicitly tolerate a local-only legacy payload only when
                # it is unambiguous; new portable checkpoints are logical.
                local_slot = logical_tensor
            else:
                raise ValueError(
                    f"Logical adapter tensor {canonical_name!r} has shape {tuple(logical_tensor.shape)}, "
                    f"expected {layout.logical_shape}"
                )
            packed[actual_name] = local_slot
        return packed

    @staticmethod
    def _optimizer_state_metadata(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Return the observed optimizer-state contract without serializing values."""
        tensor_fields: Dict[str, int] = {}
        tensor_elements: Dict[str, int] = {}
        for parameter_state in optimizer.state.values():
            for field_name, value in parameter_state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                key = f"{field_name}:{value.dtype}"
                tensor_fields[key] = tensor_fields.get(key, 0) + 1
                tensor_elements[key] = tensor_elements.get(key, 0) + value.numel()
        return {
            "tensor_fields": dict(sorted(tensor_fields.items())),
            "tensor_elements": dict(sorted(tensor_elements.items())),
        }

    def get_adapter_count(self) -> int:
        """Return the number of currently loaded adapters."""
        return len(self.adapters)

    def save_adapter_state(
        self,
        model_id: str,
        path: Optional[str] = None,
        save_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Save a specific adapter's state to disk.

        Saves LoRA weights in PEFT-compatible format (adapter_model.safetensors),
        plus optimizer state and metadata for full training resume.

        Args:
            model_id: The adapter to save
            path: Directory to save to (default: {checkpoint_dir}/{model_id})
            save_optimizer: Whether to save optimizer state

        Returns:
            Dict with path, model_id, step, and save_time
        """
        model_id = validate_identifier(model_id, name="model_id")
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]
        self.validate_strict_checkpoint_publication(model_id)

        _rank, world = _optimizer_shard_rank_world()
        if world > 1:
            raise RuntimeError(
                "LoRAAdapterManager.save_adapter_state is noncollective and cannot materialize complete logical "
                "weights or preserve rank-local optimizer shards at world_size > 1. Use "
                "CheckpointManager.save_adapter_state collectively."
            )

        # Use default path if not provided
        if path is None:
            path = str(resolve_path_within(self.checkpoint_dir, model_id))
        else:
            path = str(resolve_path_within(self.checkpoint_dir, path, reject_symlinks=True))

        # Create directory
        os.makedirs(path, exist_ok=True)

        start_time = time.time()

        # 1. Save LoRA weights in safetensors format (PEFT-compatible)
        # Convert parameter names to PEFT format: base_model.model.{name}
        raw_weights = {name: param.data.detach() for name, param in state.local_params.items()}
        # At world size one the local slot is also the complete active logical
        # tensor. Distributed saves use CheckpointManager's collective model
        # reconstruction path instead.
        active_weights = raw_weights
        weights_dict = {}
        for name, tensor in active_weights.items():
            peft_name = f"base_model.model.{self._canonical_lora_param_name(name)}"
            if peft_name in weights_dict:
                raise ValueError(f"Duplicate canonical LoRA parameter name while saving adapter state: {peft_name}")
            weights_dict[peft_name] = tensor.detach().cpu().contiguous()

        weights_path = os.path.join(path, "adapter_model.safetensors")
        safetensors_save_file(weights_dict, weights_path)

        # 2. Save optimizer state. NOTE: this path is NOT collective (LRU
        # eviction saves run on rank 0 only), so in multi-rank runs it writes
        # rank 0's shard in the sharded layout WITHOUT a manifest — such a
        # checkpoint is intentionally not optimizer-resumable; the collective
        # save path is CheckpointManager.save_adapter_state /
        # save_adapter_optimizer_shards.
        if save_optimizer:
            rank, world = _optimizer_shard_rank_world()
            if world == 1:
                save_adapter_optimizer_shards(state, path)
            else:
                _save_optimizer_state_safetensors(
                    state.optimizer.state_dict(),
                    str(resolve_path_within(path, _optimizer_shard_filename(rank), reject_symlinks=True)),
                )

        # 3. Save normalized session runtime spec with the current learning rate.
        checkpoint_session_spec = deepcopy(state.session_spec)
        checkpoint_session_spec["optimizer_config"]["learning_rate"] = float(state.lr)
        write_session_spec(path, checkpoint_session_spec)

        # 3. Save metadata
        metadata = {
            "model_id": model_id,
            "global_step": state.global_step,
            "global_forward_backward_step": state.global_forward_backward_step,
            "lr": state.lr,
            "timestamp": time.time(),
            "save_optimizer": save_optimizer,
            "optimizer": deepcopy(checkpoint_session_spec["optimizer_config"]),
            "optimizer_state": self._optimizer_state_metadata(state.optimizer),
            "layout_fingerprint": state.layout_fingerprint,
            "gradient_ownership": {
                "plan_fingerprint": (
                    state.gradient_ownership_plan.fingerprint if state.gradient_ownership_plan is not None else None
                ),
                "optimizer_restore_contract": (
                    state.gradient_ownership_plan.optimizer_restore_contract()
                    if state.gradient_ownership_plan is not None
                    else None
                ),
            },
            "layout_descriptors": [state.tensor_layouts[name].to_json_dict() for name in sorted(state.tensor_layouts)],
        }
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. Save adapter config (PEFT-compatible)
        target_modules = set()
        for name, tensor in active_weights.items():
            if "lora_A" in name or "_lora_A" in name:
                if name.endswith("_lora_A"):
                    target_modules.add(name.rsplit(".", 1)[-1][: -len("_lora_A")])
                    continue
                # Extract module name (e.g., "model.layers.0.self_attn.q_proj" from full name)
                parts = name.replace(".lora_A.weight", "").replace(".lora_A", "").replace("_lora_A", "").split(".")
                if len(parts) >= 1:
                    target_modules.add(parts[-1])  # e.g., "q_proj"

        adapter_config = {
            "base_model_name_or_path": state.session_spec.get("base_model"),
            "r": self._session_rank(state.session_spec),
            "lora_alpha": self._session_alpha(state.session_spec),
            "target_modules": list(target_modules),
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "moe_hybrid_shared_lora": self.lora_config.get("moe_hybrid_shared_lora", False),
        }
        config_path = os.path.join(path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)

        target_manifest = load_lora_target_manifest(self.lora_config.get("lora_target_manifest"))
        if target_manifest is not None:
            target_manifest_path = os.path.join(path, _TARGET_MANIFEST_FILENAME)
            with open(target_manifest_path, "w") as f:
                json.dump(target_manifest, f, indent=2, sort_keys=True)

        save_time = time.time() - start_time
        logger.info(
            f"Saved adapter state for model_id={model_id} to {path} "
            f"(step={state.global_step}, save_optimizer={save_optimizer}, time={save_time:.2f}s)"
        )

        return {
            "path": path,
            "model_id": model_id,
            "step": state.global_step,
            "save_time": save_time,
        }

    def load_adapter_state(
        self,
        model_id: str,
        path: str,
        load_optimizer: bool = True,
        lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Load adapter state from checkpoint.

        Can load into a new model_id (different from the one saved).
        Creates/registers the adapter if it doesn't exist.

        Args:
            model_id: Target model_id to load into (can differ from saved)
            path: Directory to load from
            load_optimizer: Whether to load optimizer state
            lr: Learning rate override (uses saved lr if None)

        Returns:
            Dict with path, model_id, step, and load_time
        """
        model_id = validate_identifier(model_id, name="model_id")
        local_artifact_root = Path(self.checkpoint_dir).expanduser().resolve().parent
        try:
            checkpoint_path = resolve_path_within(
                local_artifact_root,
                path,
                must_exist=True,
                reject_symlinks=True,
            )
        except ValueError:
            checkpoint_path = resolve_server_artifact(path, must_exist=True)
        path = str(checkpoint_path)
        if not checkpoint_path.is_dir():
            raise ValueError(f"Adapter checkpoint path must be a directory: {path}")

        start_time = time.time()

        # 1. Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Determine learning rate
        effective_lr = lr if lr is not None else metadata.get("lr", 1e-5)
        registered_state = self.adapters.get(model_id)
        if registered_state is not None and registered_state.poisoned:
            raise RuntimeError(
                "A poisoned adapter session cannot be recovered in-process; restart from the last committed checkpoint"
            )
        checkpoint_plan_fingerprint = metadata.get("gradient_ownership", {}).get("plan_fingerprint")
        checkpoint_restore_contract = metadata.get("gradient_ownership", {}).get("optimizer_restore_contract")
        live_plan = registered_state.gradient_ownership_plan if registered_state is not None else None
        if load_optimizer and live_plan is not None:
            self._validate_ownership_restore_contract(checkpoint_restore_contract, live_plan)
        expected_session_spec = deepcopy(registered_state.session_spec) if registered_state is not None else None
        if expected_session_spec is None:
            expected_session_spec = self._legacy_session_spec(lr=effective_lr)

        checkpoint_session_spec = load_session_spec_from_checkpoint(
            path,
            fallback_base_model=expected_session_spec.get("base_model"),
            fallback_session_spec=expected_session_spec,
        )
        self._validate_checkpoint_adapter_config(path)

        if registered_state is not None:
            self._validate_session_restore_compatibility(
                checkpoint_session_spec,
                registered_state.session_spec,
                load_optimizer=load_optimizer,
            )

        # 2. Register adapter if not exists (this will evict if needed).
        # Track whether this call did the registration so a downstream load
        # failure does not leave a fresh-init adapter resident under model_id.
        registered_here = False
        if model_id not in self.adapters:
            self.register_adapter(
                model_id,
                session_spec=checkpoint_session_spec,
                initialize_fresh=True,
            )
            registered_here = True

        state = self.adapters[model_id]
        resident_snapshot = None
        if not registered_here:
            resident_snapshot = {
                "local_params": {name: _clone_state_to_cpu(param) for name, param in state.local_params.items()},
                "optimizer": _clone_state_to_cpu(state.optimizer.state_dict()),
                "session_spec": deepcopy(state.session_spec),
                "global_step": state.global_step,
                "global_forward_backward_step": state.global_forward_backward_step,
                "lr": state.lr,
                "last_access_time": state.last_access_time,
            }

        try:
            # 3. Validate and pack logical LoRA weights without mutating the
            # resident adapter. The model-local adapter slots are never saved
            # as if they were complete PEFT tensors in distributed mode.
            weights_path = os.path.join(path, "adapter_model.safetensors")
            sharded_index_path = os.path.join(path, "adapter_model.safetensors.index.json")
            if os.path.exists(weights_path) or os.path.exists(sharded_index_path):
                loaded_weights = load_lora_checkpoint_state_dict(path)
                packed_checkpoint = self._pack_logical_state_dict(state, loaded_weights)

            else:
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            # 4. Validate and restore optimizer state before mutating resident
            # LoRA weights. A missing or invalid optimizer checkpoint must not
            # leave old optimizer moments paired with checkpoint weights.
            if load_optimizer:
                optimizer_restored = load_adapter_optimizer_shards(state, path, self.device)
                if not optimizer_restored and metadata.get("save_optimizer") is True:
                    raise RuntimeError(
                        f"Adapter checkpoint at {path} declares saved optimizer state but contains no "
                        "complete optimizer checkpoint. Use load_optimizer=False for a weights-only warm start."
                    )
                # The staged optimizer state has now passed its direct class,
                # group, coordinate, layout, and tensor contract.  Adopt the
                # checkpoint's optimizer metadata; an explicit LR override is
                # applied below.
                state.session_spec["optimizer_config"] = deepcopy(checkpoint_session_spec["optimizer_config"])

            # All checkpoint components are valid. Commit the LoRA tensors only
            # after optimizer restore succeeds.
            for name, tensor in packed_checkpoint.items():
                state.local_params[name].data.copy_(tensor.to(self.device, state.local_params[name].dtype))

            # 5. Restore metadata
            state.global_step = metadata.get("global_step", 0)
            state.global_forward_backward_step = metadata.get("global_forward_backward_step", 0)
            if lr is not None:
                self._update_state_learning_rate(state, lr)
            elif "lr" in metadata and (
                load_optimizer
                or self._strip_optimizer_learning_rate(checkpoint_session_spec)
                == self._strip_optimizer_learning_rate(state.session_spec)
            ):
                self._update_state_learning_rate(state, metadata["lr"])

            state.last_access_time = time.time()
            state.publication_pending = False
            state.publication_eligible = True
            self._clear_all_adapter_gradients(state)
            self._reset_gradient_scratch(state)
        except Exception:
            if registered_here:
                try:
                    self.remove_adapter(model_id)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Cleanup remove_adapter({model_id}) after failed load_adapter_state raised: {cleanup_error}"
                    )
            elif resident_snapshot is not None:
                try:
                    with torch.no_grad():
                        for name, tensor in resident_snapshot["local_params"].items():
                            state.local_params[name].copy_(tensor)
                    state.optimizer.load_state_dict(resident_snapshot["optimizer"])
                    state.session_spec = resident_snapshot["session_spec"]
                    state.global_step = resident_snapshot["global_step"]
                    state.global_forward_backward_step = resident_snapshot["global_forward_backward_step"]
                    state.lr = resident_snapshot["lr"]
                    state.last_access_time = resident_snapshot["last_access_time"]
                except Exception as rollback_error:
                    logger.error(
                        f"Failed to roll back resident adapter {model_id} after load_adapter_state error: "
                        f"{rollback_error}",
                        exc_info=True,
                    )
            raise

        load_time = time.time() - start_time
        logger.info(
            f"Loaded adapter state for model_id={model_id} from {path} "
            f"(step={state.global_step}, load_optimizer={load_optimizer}, time={load_time:.2f}s)"
        )

        return {
            "path": path,
            "model_id": model_id,
            "step": state.global_step,
            "load_time": load_time,
            "gradient_ownership_plan_fingerprint": checkpoint_plan_fingerprint,
        }
