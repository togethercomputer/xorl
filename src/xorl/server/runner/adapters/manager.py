"""
LoRA Adapter Manager - Manages multiple LoRA adapters for parallel training runs.

Each model_id has exactly one active adapter. Multiple model_ids can coexist,
enabling different training runs to interleave on the same base model and GPUs.

Design (Revised - Per-Adapter Parameters + Optimizer):
- Base model stays loaded on GPUs (frozen weights)
- Each adapter has its OWN nn.Parameter objects (separate .grad slots)
- Each adapter has its OWN optimizer instance
- Model params are "scratch space" - load weights before forward, capture grads after backward
- No gradient collision because each adapter's gradients live in its own Parameters
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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

from xorl.lora.target_manifest import load_lora_target_manifest
from xorl.lora.utils import (
    convert_peft_lora_state_dict,
    load_lora_checkpoint_state_dict,
)
from xorl.optim import build_optimizer
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


def _clone_state_to_cpu(value: Any) -> Any:
    """Recursively clone transaction state onto CPU to avoid GPU snapshot spikes."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return deepcopy(value)


def _adapter_param_structure_fingerprint(lora_params: Dict[str, nn.Parameter]) -> str:
    """Order-sensitive fingerprint of an adapter's local (name, shape, dtype) sequence.

    Optimizer state_dicts key parameters by position, so a saved shard is only
    meaningful on a rank whose parameter sequence matches the saving rank
    exactly. Under expert parallelism the same position holds DIFFERENT expert
    slices on different ranks, which shape checks alone cannot distinguish;
    world-size + per-rank fingerprints together make misassignment loud.
    """
    structure = [[name, list(param.shape), str(param.dtype)] for name, param in lora_params.items()]
    return hashlib.sha256(json.dumps(structure).encode("utf-8")).hexdigest()


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
            for name in sorted(adapter_state.local_params)
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
                for name in sorted(adapter_state.local_params)
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
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("format_version") != 3:
            raise RuntimeError(
                f"Adapter optimizer checkpoint at {path} uses obsolete or unsupported layout format version "
                f"{manifest.get('format_version')!r}; local-shard optimizer state cannot be migrated by position. "
                "Load weights-only (load_optimizer=False) and re-save on the new topology."
            )
        saved_world = int(manifest["world_size"])
        if saved_world != world:
            raise RuntimeError(
                f"Adapter optimizer checkpoint at {path} was saved with world_size={saved_world} but is "
                f"being loaded with world_size={world}. Per-rank Adam moments cannot be re-sharded; resume "
                "on the saved topology or load weights-only (load_optimizer=False)."
            )
        expected_fingerprint = manifest["per_rank_param_structure_sha256"][rank]
        live_fingerprint = _adapter_param_structure_fingerprint(adapter_state.local_params)
        if expected_fingerprint != live_fingerprint:
            raise RuntimeError(
                f"Adapter optimizer shard for rank {rank} at {path} was saved for a different local "
                "parameter structure. Refusing to load misassigned Adam moments; load weights-only "
                "(load_optimizer=False) instead."
            )
        expected_layout_fingerprint = manifest.get("per_rank_layout_fingerprint", [None] * world)[rank]
        if expected_layout_fingerprint != adapter_state.layout_fingerprint:
            raise RuntimeError(
                f"Adapter optimizer shard for rank {rank} at {path} was saved for a different topology/layout "
                "fingerprint. Refusing to load rank-local optimizer state; load weights-only "
                "(load_optimizer=False) instead."
            )
        expected_session_rank = manifest.get("session_rank")
        live_session_rank = int(adapter_state.session_spec["lora_config"]["lora_rank"])
        if expected_session_rank != live_session_rank:
            raise RuntimeError(
                f"Adapter optimizer shard at {path} was saved for session_rank={expected_session_rank!r}, "
                f"but the live adapter uses session_rank={live_session_rank}; load weights-only "
                "(load_optimizer=False) instead."
            )
        per_rank_orders = manifest.get("per_rank_optimizer_parameter_order")
        expected_order = (
            per_rank_orders[rank] if per_rank_orders is not None else manifest.get("optimizer_parameter_order")
        )
        live_order = [
            canonical_parameter_name(name)
            for name in sorted(adapter_state.local_params)
            if adapter_state.local_params[name].numel() > 0
        ]
        if expected_order != live_order:
            raise RuntimeError(
                f"Adapter optimizer shard at {path} has a different optimizer parameter order; "
                "load weights-only (load_optimizer=False) instead."
            )
        shard_path = resolve_path_within(
            path,
            _optimizer_shard_filename(rank),
            must_exist=True,
            reject_symlinks=True,
        )
        optimizer_state = _load_optimizer_state_safetensors(str(shard_path), device)
        adapter_state.optimizer.load_state_dict(optimizer_state)
        logger.info(f"Loaded rank-{rank} adapter optimizer shard from {shard_path}")
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
    1. prepare_forward(model_id): Copy adapter weights into model
    2. Forward + backward (gradients go to model's params)
    3. capture_gradients(model_id): Copy model's grads to adapter's params
    4. optim_step(model_id): Adapter's optimizer steps on adapter's params
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
        self.adapters: Dict[str, AdapterState] = {}
        self.current_adapter_id: Optional[str] = None
        self._layout_cache: Dict[int, Tuple[Dict[str, AdapterTensorLayout], str]] = {}
        self._model_param_ids: Dict[str, int] = {}

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
            f"optimizer={optimizer_type}"
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
        """Return the structural part of a LoRA session spec without optimizer metadata.

        For weights-only restore we also strip zorl_config: the LoRA tensor
        shapes are determined by lora_config (rank/alpha) only, so a
        target session can change ZORL hyperparameters (num_perturbation_pairs,
        b_sigma, etc.) without invalidating a saved set of weights.
        """
        stripped = deepcopy(session_spec)
        stripped.pop("optimizer_config", None)
        stripped.pop("zorl_config", None)
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

    def _discover_layouts(self, session_rank: int) -> Tuple[Dict[str, AdapterTensorLayout], str]:
        cached = self._layout_cache.get(session_rank)
        if cached is not None:
            return cached
        layouts, fingerprint = discover_adapter_layouts(
            self.model,
            self._lora_param_metadata,
            active_rank=session_rank,
            pipeline_parallel_size=self._pipeline_parallel_size,
        )
        self._layout_cache[session_rank] = layouts, fingerprint
        return layouts, fingerprint

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
        for name in sorted(state.local_params):
            layout = state.tensor_layouts[name]
            if layout.replica_count <= 1:
                continue
            value = state.local_params[name].grad if gradients else state.local_params[name].detach()
            local_payload.append((layout.replica_key, None if value is None else value.detach().cpu().contiguous()))
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_payload)
        by_key: dict[tuple[Any, ...], list[Optional[torch.Tensor]]] = {}
        for rank_payload in gathered:
            for key, value in rank_payload:
                by_key.setdefault(tuple(key), []).append(value)
        local_keys = {layout.replica_key: layout.replica_count for layout in state.tensor_layouts.values()}
        for key, expected_count in local_keys.items():
            if expected_count <= 1:
                continue
            values = by_key.get(key, [])
            if len(values) != expected_count:
                raise RuntimeError(f"Replica descriptor count changed for {key[0]!r}: {len(values)}")
            reference = values[0]
            if any((value is None) != (reference is None) for value in values):
                raise RuntimeError(f"Replica gradient/weight presence diverged for {key[0]!r}")
            if reference is not None and any(not torch.equal(reference, value) for value in values[1:]):
                raise RuntimeError(f"Replica values diverged for logical rectangle {key[0]!r}")

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

        return {name: param for name, param in local_params.items() if param.numel() > 0}

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

    def _reset_adapter_optimizer(self, model_id: str) -> None:
        """Rebuild an adapter optimizer from its normalized session spec."""
        state = self.get_adapter_state(model_id)
        state.optimizer = self._build_adapter_optimizer_for_session(state.local_params, state.session_spec)

    def reinitialize_adapter_for_zorl_family(
        self,
        model_id: str,
        *,
        a_seed: int,
        a_init: str = "gaussian_jl",
    ) -> None:
        """Refresh a ZORL parent adapter with a fresh LoRA-A family and zero LoRA-B."""
        if a_init != "gaussian_jl":
            raise ValueError(f"Unsupported ZORL LoRA-A init: {a_init!r}")

        state = self.get_adapter_state(model_id)
        with torch.no_grad():
            for name in sorted(state.local_params):
                param = state.local_params[name]
                values = deterministic_local_initialization(
                    state.tensor_layouts[name],
                    base_seed=a_seed,
                    session_identity=model_id,
                    is_lora_b=self._is_lora_b(name),
                ).to(device=param.device, dtype=param.dtype)
                param.copy_(values)
                param.grad = None

        self._reset_adapter_optimizer(model_id)
        state.last_access_time = time.time()
        logger.info(
            f"Reinitialized ZORL family for model_id={model_id} "
            f"(a_init={a_init}, a_seed={a_seed}, optimizer={state.session_spec['optimizer_config']['type']})"
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
    ) -> None:
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

        layouts, layout_fp = self._discover_layouts(session_rank)
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

        self.adapters[model_id] = AdapterState(
            model_id=model_id,
            session_spec=session_spec,
            local_params=local_params,
            tensor_layouts=layouts,
            layout_fingerprint=layout_fp,
            optimizer=optimizer,
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

    def capture_gradients(self, model_id: str) -> None:
        """
        Copy gradients from model params to adapter params after backward.

        This captures the gradients computed during backward() and stores
        them in the adapter's own Parameter objects (which have their own
        .grad slots). This prevents gradient collision when multiple adapters
        interleave.

        Only the model-local gradient rectangle is copied.  No full DTensor is
        materialized in this hot path.

        Args:
            model_id: The adapter to capture gradients for
        """
        model_id = validate_identifier(model_id, name="model_id")
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]
        self._validate_model_layout_identity(state)

        for name, param in self.model.named_parameters():
            if name not in state.local_params or param.grad is None:
                continue
            adapter_param = state.local_params[name]
            raw_grad = param.grad
            local_grad = raw_grad.to_local() if _HAS_DTENSOR and isinstance(raw_grad, DTensor) else raw_grad
            local_grad = wait_for_local_tensor(local_grad)
            grad = state.tensor_layouts[name].pack_from_local(local_grad)
            param.grad = None
            if grad.numel() == 0:
                continue
            accumulation_dtype = (
                torch.float32 if adapter_param.dtype in {torch.float16, torch.bfloat16} else adapter_param.dtype
            )
            grad = grad.to(dtype=accumulation_dtype, device=adapter_param.device)
            if adapter_param.grad is None:
                adapter_param.grad = grad
            else:
                adapter_param.grad.add_(grad)

    def optim_step(
        self,
        model_id: str,
        lr: float,
        gradient_clip: Optional[float] = None,
        accumulated_valid_tokens: int = 0,
        norm_type: float = 2.0,
    ) -> float:
        """
        Run optimizer step on adapter's own parameters.

        This uses the adapter's own optimizer on the adapter's own Parameters,
        which have their own gradients from capture_gradients().

        Args:
            model_id: The adapter to step
            lr: Learning rate to use
            gradient_clip: Optional gradient clipping value
            accumulated_valid_tokens: Total valid tokens accumulated across
                forward_backward calls. If > 0, gradients are scaled by
                1/accumulated_valid_tokens (deferred normalization).

        Returns:
            The gradient norm before clipping
        """
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]
        self._validate_model_layout_identity(state)

        # Update learning rate
        self._update_state_learning_rate(state, lr)

        # Deferred gradient normalization: scale raw gradients by 1/accumulated_valid_tokens
        if accumulated_valid_tokens > 0:
            scale = 1.0 / accumulated_valid_tokens
            for p in state.local_params.values():
                if p.grad is not None:
                    p.grad.mul_(scale)

        optimizer_params = [
            state.local_params[name] for name in sorted(state.local_params) if state.local_params[name].numel() > 0
        ]
        local_grads = [param.grad for param in optimizer_params if param.grad is not None]
        local_device = local_grads[0].device if local_grads else self.device
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        group = torch.distributed.group.WORLD if distributed else None
        world = torch.distributed.get_world_size(group=group) if distributed else 1

        if distributed and world > 1:
            backend = torch.distributed.get_backend(group)
            if backend == "nccl" and local_device.type != "cuda":
                local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            finite_flag = torch.tensor(
                [1 if all(torch.isfinite(grad).all() for grad in local_grads) else 0], device=local_device
            )
            torch.distributed.all_reduce(finite_flag, op=torch.distributed.ReduceOp.MIN, group=group)
            all_finite = bool(finite_flag.item())
        else:
            all_finite = all(torch.isfinite(grad).all().item() for grad in local_grads)

        if not all_finite:
            for param in state.local_params.values():
                param.grad = None
            raise FloatingPointError(
                f"Non-finite adapter gradient detected for model_id={model_id}; "
                "all ranks skipped the optimizer step and cleared gradients."
            )

        self._validate_replica_coherence(state, gradients=True)

        if gradient_clip is None or gradient_clip <= 0:
            clipping_enabled = False
        else:
            clipping_enabled = True

        if math.isinf(float(norm_type)):
            local_max = torch.zeros((), dtype=torch.float32, device=local_device)
            for param in optimizer_params:
                if param.grad is not None:
                    local_max = torch.maximum(local_max, param.grad.detach().to(torch.float32).abs().max())
            if distributed and world > 1:
                torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=group)
            grad_norm = float(local_max.item())
        elif float(norm_type) == 2.0:
            local_sum_sq = torch.zeros((), dtype=torch.float32, device=local_device)
            for name in sorted(state.local_params):
                param = state.local_params[name]
                if param.grad is None or param.numel() == 0:
                    continue
                grad_sum_sq = param.grad.detach().to(torch.float32).square().sum()
                local_sum_sq = local_sum_sq + grad_sum_sq / max(state.tensor_layouts[name].replica_count, 1)
            if distributed and world > 1:
                torch.distributed.all_reduce(local_sum_sq, op=torch.distributed.ReduceOp.SUM, group=group)
            grad_norm = float(torch.sqrt(local_sum_sq).item())
        else:
            raise ValueError(f"Unsupported adapter gradient norm type: {norm_type!r}")

        clip_coefficient = 1.0
        if clipping_enabled:
            clip_coefficient = min(1.0, float(gradient_clip) / (grad_norm + 1e-6))
            if clip_coefficient != 1.0:
                for param in optimizer_params:
                    if param.grad is not None:
                        param.grad.mul_(clip_coefficient)

        state.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        try:
            state.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            state.optimizer.zero_grad()
        state.global_step += 1
        return grad_norm

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
        for name in sorted(state.local_params):
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

    def load_logical_gradients(
        self,
        model_id: str,
        gradient_state_dict: Dict[str, torch.Tensor],
        *,
        accumulate: bool = True,
    ) -> None:
        """Pack logical gradients into local adapter slots without model scratch state."""

        state = self.get_adapter_state(model_id)
        expected_names = {canonical_parameter_name(name): name for name in state.local_params}
        converted_by_name = {canonical_parameter_name(name): tensor for name, tensor in gradient_state_dict.items()}
        unexpected = sorted(set(converted_by_name) - set(expected_names))
        if unexpected:
            raise ValueError(f"Logical gradient state contains unexpected parameters: {unexpected!r}")
        with torch.no_grad():
            for canonical_name, logical_gradient in converted_by_name.items():
                actual_name = expected_names[canonical_name]
                layout = state.tensor_layouts[actual_name]
                if tuple(logical_gradient.shape) != layout.logical_shape:
                    raise ValueError(
                        f"Logical gradient {canonical_name!r} has shape {tuple(logical_gradient.shape)}, "
                        f"expected {layout.logical_shape}"
                    )
                local_gradient = pack_logical_tensor(layout, logical_gradient).to(
                    device=state.local_params[actual_name].device,
                    dtype=torch.float32
                    if state.local_params[actual_name].dtype in {torch.float16, torch.bfloat16}
                    else state.local_params[actual_name].dtype,
                )
                if not accumulate or state.local_params[actual_name].grad is None:
                    state.local_params[actual_name].grad = local_gradient
                else:
                    state.local_params[actual_name].grad.add_(local_gradient)

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
            checkpoint_spec_for_compare = checkpoint_session_spec
            registered_spec_for_compare = registered_state.session_spec
            if lr is not None:
                checkpoint_spec_for_compare = self._strip_optimizer_learning_rate(checkpoint_spec_for_compare)
                registered_spec_for_compare = self._strip_optimizer_learning_rate(registered_spec_for_compare)

            if load_optimizer:
                specs_match = checkpoint_spec_for_compare == registered_spec_for_compare
                mismatch_context = "registered multi-adapter session"
            else:
                specs_match = self._strip_optimizer_config(checkpoint_spec_for_compare) == self._strip_optimizer_config(
                    registered_spec_for_compare
                )
                mismatch_context = "registered multi-adapter session for weights-only restore"

            if not specs_match:
                raise ValueError(
                    "Checkpoint session spec does not match the "
                    f"{mismatch_context}. checkpoint={checkpoint_session_spec!r}, "
                    f"current={registered_state.session_spec!r}"
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
        }
