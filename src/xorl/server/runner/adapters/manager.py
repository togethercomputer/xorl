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
    get_lora_tensor_shard_specs,
    load_lora_checkpoint_state_dict,
)
from xorl.optim import build_optimizer
from xorl.server.security import resolve_path_within, resolve_server_artifact, validate_identifier
from xorl.server.session_spec import (
    load_session_spec_from_checkpoint,
    session_optimizer_build_kwargs,
    write_session_spec,
)


try:
    from torch.distributed._tensor import DTensor
    from torch.distributed._tensor.placement_types import Shard

    _HAS_DTENSOR = True
except ImportError:
    _HAS_DTENSOR = False


logger = logging.getLogger(__name__)


_TORCH_MANUAL_SEED_MASK = 0x7FFFFFFFFFFFFFFF
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
    fingerprint = _adapter_param_structure_fingerprint(adapter_state.lora_params)
    if world > 1:
        fingerprints: List[Optional[str]] = [None] * world
        torch.distributed.all_gather_object(fingerprints, fingerprint)
    else:
        fingerprints = [fingerprint]
    manifest = {
        "format_version": 2,
        "world_size": world,
        "per_rank_param_structure_sha256": fingerprints,
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
        if manifest.get("format_version") != 2:
            raise RuntimeError(
                f"Adapter optimizer checkpoint at {path} uses unsupported or pickle-backed format version "
                f"{manifest.get('format_version')!r}; load weights-only (load_optimizer=False)."
            )
        saved_world = int(manifest["world_size"])
        if saved_world != world:
            raise RuntimeError(
                f"Adapter optimizer checkpoint at {path} was saved with world_size={saved_world} but is "
                f"being loaded with world_size={world}. Per-rank Adam moments cannot be re-sharded; resume "
                "on the saved topology or load weights-only (load_optimizer=False)."
            )
        expected_fingerprint = manifest["per_rank_param_structure_sha256"][rank]
        live_fingerprint = _adapter_param_structure_fingerprint(adapter_state.lora_params)
        if expected_fingerprint != live_fingerprint:
            raise RuntimeError(
                f"Adapter optimizer shard for rank {rank} at {path} was saved for a different local "
                "parameter structure. Refusing to load misassigned Adam moments; load weights-only "
                "(load_optimizer=False) instead."
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


def _mix_torch_manual_seed(base_seed: int, *components: int) -> int:
    """Mix deterministic seed components into PyTorch's signed 63-bit range."""
    acc = int(base_seed) & _TORCH_MANUAL_SEED_MASK
    for index, component in enumerate(components, start=1):
        acc = (acc * 6364136223846793005 + int(component) * 1442695040888963407 + index) & _TORCH_MANUAL_SEED_MASK
    return acc


@dataclass
class AdapterState:
    """Complete isolated state for one training run.

    Key insight: Each adapter owns its own nn.Parameter objects, which have
    their own .grad slots. This prevents gradient collision when multiple
    adapters' forward_backward calls interleave.
    """

    model_id: str
    session_spec: Dict[str, Any]
    lora_params: Dict[str, nn.Parameter]  # Actual Parameters with own .grad
    optimizer: torch.optim.Optimizer  # Per-adapter optimizer
    global_step: int = 0
    global_forward_backward_step: int = 0
    lr: float = 1e-5
    last_access_time: float = field(default_factory=time.time)  # For LRU eviction


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

        # Cache the list of LoRA parameter names for efficient lookups
        self._lora_param_names: List[str] = []
        self._lora_param_metadata: Dict[str, Dict[str, Any]] = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self._lora_param_names.append(name)
                param_shape = tuple(param.shape if _HAS_DTENSOR and isinstance(param, DTensor) else param.data.shape)
                self._lora_param_metadata[name] = {
                    "shape": param_shape,
                    "dtype": param.dtype if _HAS_DTENSOR and isinstance(param, DTensor) else param.data.dtype,
                    "rank_dim": self._infer_lora_rank_dim(name, param_shape),
                }

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
    def _replace_dim(shape: Tuple[int, ...], dim: int, value: int) -> Tuple[int, ...]:
        updated = list(shape)
        updated[dim] = value
        return tuple(updated)

    @staticmethod
    def _slice_to_rank(tensor: torch.Tensor, *, rank_dim: int, active_rank: int) -> torch.Tensor:
        return tensor.narrow(rank_dim, 0, active_rank)

    @staticmethod
    def _slice_to_shape(tensor: torch.Tensor, *, rank_dim: int, target_shape: Tuple[int, ...]) -> torch.Tensor:
        active_rank = target_shape[rank_dim]
        sliced = tensor.narrow(rank_dim, 0, active_rank)
        if tuple(sliced.shape) != target_shape:
            raise ValueError(f"Expected sliced tensor shape {target_shape}, got {tuple(sliced.shape)}")
        return sliced

    @staticmethod
    def _expand_compact_tensor(
        tensor: torch.Tensor,
        *,
        full_shape: Tuple[int, ...],
        rank_dim: int,
    ) -> torch.Tensor:
        if tuple(tensor.shape) == full_shape:
            return tensor
        expanded = torch.zeros(full_shape, dtype=tensor.dtype, device=tensor.device)
        slices = [slice(None)] * len(full_shape)
        slices[rank_dim] = slice(0, tensor.shape[rank_dim])
        expanded[tuple(slices)] = tensor
        return expanded

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

    def _build_adapter_optimizer(self, lora_params: Dict[str, nn.Parameter], lr: float) -> torch.optim.Optimizer:
        """Build an optimizer for one adapter via the shared optimizer factory."""
        adapter_module = self._build_parameter_module(lora_params)
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
        adapter_module = self._build_parameter_module(lora_params)
        build_kwargs = session_optimizer_build_kwargs(session_spec["optimizer_config"])
        return build_optimizer(
            adapter_module,
            fused=self.optimizer_fused,
            **build_kwargs,
        )

    def _reset_adapter_optimizer(self, model_id: str) -> None:
        """Rebuild an adapter optimizer from its normalized session spec."""
        state = self.get_adapter_state(model_id)
        state.optimizer = self._build_adapter_optimizer_for_session(state.lora_params, state.session_spec)

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

        # MoE params are 3D ``[num_local_experts, ...]``. With EP > 1, every
        # rank sees the *same* shape but different *global* experts. If we
        # used one rank-identical seed for the whole tensor, expert i on
        # rank 0 and expert i on rank N would draw identical LoRA-A values,
        # collapsing the search dimension. Mix the global expert index into
        # the seed for each MoE expert instead.
        try:
            from xorl.distributed.parallel_state import get_parallel_state  # noqa: PLC0415

            ep_rank = get_parallel_state().ep_rank if get_parallel_state().ep_size > 1 else 0
        except Exception:
            ep_rank = 0

        with torch.no_grad():
            for name in sorted(state.lora_params):
                param = state.lora_params[name]
                if "lora_A" in name:
                    if param.ndim == 3:
                        # MoE LoRA-A: [num_local_experts, in, r]. Seed per
                        # global expert so EP siblings sample independently.
                        num_local_experts = int(param.shape[0])
                        input_dim = int(param.shape[1])
                        inv_scale = 1.0 / math.sqrt(max(input_dim, 1))
                        values = torch.empty(tuple(param.shape), dtype=torch.float32)
                        for local_idx in range(num_local_experts):
                            global_expert_idx = ep_rank * num_local_experts + local_idx
                            expert_gen = torch.Generator(device="cpu")
                            expert_gen.manual_seed(_mix_torch_manual_seed(a_seed, global_expert_idx))
                            torch.randn(
                                tuple(param.shape[1:]),
                                generator=expert_gen,
                                dtype=torch.float32,
                                out=values[local_idx],
                            )
                    else:
                        # Non-MoE LoRA-A: rank-identical seed is correct; the
                        # tensor lives on every rank as a replica.
                        input_dim = int(param.shape[1]) if param.ndim > 1 else int(param.numel())
                        inv_scale = 1.0 / math.sqrt(max(input_dim, 1))
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(_mix_torch_manual_seed(a_seed))
                        values = torch.randn(tuple(param.shape), generator=generator, dtype=torch.float32)
                    values.mul_(inv_scale)
                    param.data.copy_(values.to(device=param.device, dtype=param.dtype))
                elif "lora_B" in name:
                    param.zero_()
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
        return any(param.grad is not None for param in state.lora_params.values())

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

        # Evict LRU adapter if at capacity and this is a new adapter
        if model_id not in self.adapters:
            self._maybe_evict()
        else:
            logger.info(f"Replacing existing adapter for model_id={model_id}")

        # Create Parameter objects for this adapter
        # IMPORTANT: Must create regular tensors, not DTensors, because the adapter's
        # optimizer needs to work on regular tensors (DTensors cause issues with fused ops)
        lora_params: Dict[str, nn.Parameter] = {}
        for name, param in self.model.named_parameters():
            if name in self._lora_param_names:
                metadata = self._lora_param_metadata[name]
                param_shape = metadata["shape"]
                param_dtype = metadata["dtype"]
                rank_dim = metadata["rank_dim"]
                compact_shape = self._replace_dim(param_shape, rank_dim, session_rank)

                if initialize_fresh:
                    # Fresh initialization - create compact regular tensor on device
                    if "lora_A" in name:
                        new_tensor = torch.empty(
                            compact_shape,
                            dtype=param_dtype,
                            device=self.device,
                        )
                        nn.init.kaiming_uniform_(new_tensor, a=math.sqrt(5))
                    else:  # lora_B
                        new_tensor = torch.zeros(
                            compact_shape,
                            dtype=param_dtype,
                            device=self.device,
                        )
                else:
                    # Copy current model weights as regular tensor
                    # NOTE: This path requires full_tensor() for DTensors, so it should
                    # only be called when all ranks are participating (e.g., from
                    # register_adapter endpoint, not from load_adapter_state)
                    if _HAS_DTENSOR and isinstance(param, DTensor):
                        param_data = param.full_tensor()
                    else:
                        param_data = param.data
                    new_tensor = (
                        self._slice_to_shape(
                            param_data.detach(),
                            rank_dim=rank_dim,
                            target_shape=compact_shape,
                        )
                        .clone()
                        .to(self.device)
                    )

                # Create as nn.Parameter so it has its own .grad slot
                lora_params[name] = nn.Parameter(new_tensor, requires_grad=True)

        # Build optimizer for this adapter using the session's optimizer contract.
        optimizer = self._build_adapter_optimizer_for_session(lora_params, session_spec)

        self.adapters[model_id] = AdapterState(
            model_id=model_id,
            session_spec=session_spec,
            lora_params=lora_params,
            optimizer=optimizer,
            global_step=0,
            global_forward_backward_step=0,
            lr=effective_lr,
        )

        logger.info(
            f"Registered adapter for model_id={model_id} "
            f"(rank={session_rank}, alpha={session_alpha}, lr={effective_lr}, "
            f"fresh_weights={initialize_fresh}, num_params={len(lora_params)}, "
            f"optimizer={optimizer_config['type']})"
        )

    def prepare_forward(self, model_id: str) -> None:
        """
        Load adapter weights into model before forward pass.

        This must be called before forward() to ensure the model uses
        the correct adapter's weights.

        For FSDP2/DTensor: The model's params are sharded DTensors, but the
        adapter's params are full regular tensors. We need to copy only the
        local shard from the adapter to the model.

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
        # Update last access time for LRU tracking
        state.last_access_time = time.time()
        self._set_model_runtime_lora_config(
            lora_rank=self._session_rank(state.session_spec),
            lora_alpha=self._session_alpha(state.session_spec),
        )

        # Copy adapter weights into model's params (for forward to use)
        # Use no_grad to avoid autograd issues with DTensor views
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state.lora_params:
                    metadata = self._lora_param_metadata[name]
                    adapter_data = self._expand_compact_tensor(
                        state.lora_params[name].data,
                        full_shape=metadata["shape"],
                        rank_dim=metadata["rank_dim"],
                    )

                    if _HAS_DTENSOR and isinstance(param, DTensor):
                        # For DTensor: copy to the local tensor (the shard)
                        # The adapter has full weights, but model param is sharded.
                        # We need to extract the correct shard from adapter weights.
                        local_tensor = param.to_local()
                        placements = param.placements
                        device_mesh = param.device_mesh

                        sliced_data = adapter_data
                        skip_copy = False
                        for dim, placement in enumerate(placements):
                            if isinstance(placement, Shard):
                                shard_dim = placement.dim
                                mesh_dim_size = device_mesh.size(dim)
                                local_rank = device_mesh.get_local_rank(mesh_dim=dim)
                                total_size = sliced_data.shape[shard_dim]
                                shard_size = (total_size + mesh_dim_size - 1) // mesh_dim_size
                                start = local_rank * shard_size
                                end = min(start + shard_size, total_size)
                                length = max(end - start, 0)
                                if start >= total_size or length == 0:
                                    skip_copy = True
                                    break
                                sliced_data = sliced_data.narrow(shard_dim, start, length)

                        if not skip_copy and local_tensor.numel() > 0:
                            local_tensor.copy_(sliced_data)
                    else:
                        # Regular tensor: direct copy
                        param.data.copy_(adapter_data)

        self.current_adapter_id = model_id

    def capture_gradients(self, model_id: str) -> None:
        """
        Copy gradients from model params to adapter params after backward.

        This captures the gradients computed during backward() and stores
        them in the adapter's own Parameter objects (which have their own
        .grad slots). This prevents gradient collision when multiple adapters
        interleave.

        For FSDP2/DTensor: If gradients are DTensors (sharded), we call
        .full_tensor() to get the full unsharded gradient before copying.

        Args:
            model_id: The adapter to capture gradients for
        """
        model_id = validate_identifier(model_id, name="model_id")
        if model_id not in self.adapters:
            raise KeyError(f"Adapter for model_id={model_id} not registered")

        state = self.adapters[model_id]
        grad_count = 0

        for name, param in self.model.named_parameters():
            if name in state.lora_params:
                adapter_param = state.lora_params[name]
                if param.grad is not None:
                    metadata = self._lora_param_metadata[name]
                    # Handle DTensor (FSDP2 sharded gradients)
                    grad = param.grad
                    if _HAS_DTENSOR and isinstance(grad, DTensor):
                        grad = grad.full_tensor()
                    grad = self._slice_to_shape(
                        grad,
                        rank_dim=metadata["rank_dim"],
                        target_shape=tuple(adapter_param.shape),
                    )

                    # Copy gradient to adapter's param (accumulate for grad accumulation)
                    if adapter_param.grad is None:
                        adapter_param.grad = grad.clone()
                    else:
                        adapter_param.grad.add_(grad)
                    grad_count += 1
                    # Clear model's grad to prevent accumulation across adapters
                    param.grad = None

    def optim_step(
        self,
        model_id: str,
        lr: float,
        gradient_clip: Optional[float] = None,
        accumulated_valid_tokens: int = 0,
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

        # Update learning rate
        self._update_state_learning_rate(state, lr)

        # Deferred gradient normalization: scale raw gradients by 1/accumulated_valid_tokens
        if accumulated_valid_tokens > 0:
            scale = 1.0 / accumulated_valid_tokens
            for p in state.lora_params.values():
                if p.grad is not None:
                    p.grad.data = p.grad.data.float() * scale

        # Always use clip_grad_norm_ for correct grad norm computation
        # Using a large clip value (10000.0) effectively means no clipping
        clip_value = gradient_clip if (gradient_clip is not None and gradient_clip > 0) else 10000.0
        grad_norm = torch.nn.utils.clip_grad_norm_(list(state.lora_params.values()), clip_value)
        if hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()

        # Step the adapter's optimizer
        state.optimizer.step()
        state.optimizer.zero_grad()

        # Increment step counter
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
            param_bytes = sum(p.numel() * p.element_size() for p in state.lora_params.values())

            # Calculate optimizer state memory (AdamW stores exp_avg and exp_avg_sq)
            optim_bytes = 0
            for param_state in state.optimizer.state.values():
                for v in param_state.values():
                    if isinstance(v, torch.Tensor):
                        optim_bytes += v.numel() * v.element_size()

            usage[model_id] = param_bytes + optim_bytes
        return usage

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
        if save_optimizer and world > 1:
            raise RuntimeError(
                "LoRAAdapterManager.save_adapter_state is noncollective and cannot preserve every rank's "
                "optimizer shard. Use CheckpointManager.save_adapter_state collectively, or pass "
                "save_optimizer=False for a weights-only checkpoint."
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
        raw_weights = {name: param.data.detach() for name, param in state.lora_params.items()}
        # Adapter-owned tensors are already compacted to that session's rank.
        # Do not slice against the live model: LRU eviction can save a different
        # adapter than the one currently loaded into the model scratch space.
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
                "lora_params": {name: _clone_state_to_cpu(param) for name, param in state.lora_params.items()},
                "optimizer": _clone_state_to_cpu(state.optimizer.state_dict()),
                "session_spec": deepcopy(state.session_spec),
                "global_step": state.global_step,
                "global_forward_backward_step": state.global_forward_backward_step,
                "lr": state.lr,
                "last_access_time": state.last_access_time,
            }

        try:
            # 3. Load LoRA weights
            weights_path = os.path.join(path, "adapter_model.safetensors")
            sharded_index_path = os.path.join(path, "adapter_model.safetensors.index.json")
            if os.path.exists(weights_path) or os.path.exists(sharded_index_path):
                loaded_weights = load_lora_checkpoint_state_dict(path)
                expected_param_map: Dict[str, str] = {}
                expected_shapes: Dict[str, torch.Size] = {}
                for actual_name in state.lora_params:
                    canonical_name = self._canonical_lora_param_name(actual_name)
                    if canonical_name in expected_param_map and expected_param_map[canonical_name] != actual_name:
                        raise ValueError(
                            f"Live adapter contains duplicate LoRA tensors after canonicalization. param={canonical_name!r}"
                        )
                    expected_param_map[canonical_name] = actual_name
                    expected_shapes[canonical_name] = state.lora_params[actual_name].shape

                expected_shard_specs = get_lora_tensor_shard_specs(self.model, names=expected_shapes.keys())
                converted_weights = convert_peft_lora_state_dict(
                    loaded_weights,
                    expected_shapes=expected_shapes,
                    expected_shard_specs=expected_shard_specs,
                )
                checkpoint_tensors: Dict[str, torch.Tensor] = {}
                for converted_name, weight in converted_weights.items():
                    canonical_name = self._canonical_lora_param_name(converted_name)
                    if canonical_name in checkpoint_tensors:
                        raise ValueError(
                            f"Checkpoint contains duplicate LoRA tensors after canonicalization. param={canonical_name!r}"
                        )
                    checkpoint_tensors[canonical_name] = weight.to(self.device)

                expected_param_names = set(expected_param_map)
                checkpoint_param_names = set(checkpoint_tensors)
                missing_param_names = sorted(expected_param_names - checkpoint_param_names)
                unexpected_param_names = sorted(checkpoint_param_names - expected_param_names)
                if missing_param_names or unexpected_param_names:
                    raise ValueError(
                        "Checkpoint LoRA parameter set does not match the live adapter structure. "
                        f"missing={missing_param_names!r}, unexpected={unexpected_param_names!r}"
                    )

                for internal_name, tensor in checkpoint_tensors.items():
                    target_param = state.lora_params[expected_param_map[internal_name]]
                    if tuple(tensor.shape) != tuple(target_param.shape):
                        raise ValueError(
                            "Checkpoint tensor shape does not match the live adapter shape. "
                            f"param={internal_name!r}, checkpoint={tuple(tensor.shape)!r}, "
                            f"live={tuple(target_param.shape)!r}"
                        )

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
            for internal_name, tensor in checkpoint_tensors.items():
                state.lora_params[expected_param_map[internal_name]].data.copy_(tensor)

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
                        for name, tensor in resident_snapshot["lora_params"].items():
                            state.lora_params[name].copy_(tensor)
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
