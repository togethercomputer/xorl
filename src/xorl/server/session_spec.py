"""Utilities for multi-adapter session runtime specs.

The server supports heterogeneous multi-adapter LoRA sessions where each
``model_id`` owns:

- a LoRA runtime config (rank + alpha)
- an optimizer contract (type + kwargs + default learning rate)

This module normalizes those specs into a shared JSON-safe structure used by
the API server, worker runtime, and checkpoint metadata.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch


SUPPORTED_OPTIMIZER_TYPES = {
    "adamw",
    "anyprecision_adamw",
    "sgd",
    "signsgd",
    "muon",
}

DEFAULT_ADAM_BETAS = (0.9, 0.95)
DEFAULT_ADAM_EPS = 1e-8
SESSION_SPEC_FILENAME = "session_spec.json"

_PER_SESSION_LORA_KEY_ALIASES = {
    "rank": "lora_rank",
    "alpha": "lora_alpha",
    "target_modules": "lora_target_modules",
}

_SERVER_WIDE_LORA_KEYS = {
    "enable_lora",
    "enable_qlora",
    "lora_target_modules",
    "train_attn",
    "train_mlp",
    "train_unembed",
    "moe_shared_lora",
    "moe_hybrid_shared_lora",
    "quant_format",
    "quant_group_size",
    "exclude_modules",
}

_DTYPE_STRING_TO_TORCH = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _clone_jsonable(value: Any) -> Any:
    """Deep copy nested metadata while making torch dtypes JSON-safe."""
    if isinstance(value, torch.dtype):
        if value == torch.bfloat16:
            return "bf16"
        if value == torch.float32:
            return "fp32"
        return str(value)
    if isinstance(value, dict):
        return {k: _clone_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_clone_jsonable(v) for v in value]
    return deepcopy(value)


def _restore_optimizer_metadata(value: Any) -> Any:
    """Restore JSON-safe optimizer kwargs to the build_optimizer format."""
    if isinstance(value, dict):
        restored = {}
        for key, nested in value.items():
            converted = _restore_optimizer_metadata(nested)
            if key in {"muon_momentum_dtype", "muon_grad_dtype", "muon_update_dtype"} and isinstance(converted, str):
                converted = _DTYPE_STRING_TO_TORCH.get(converted, converted)
            restored[key] = converted
        return restored
    if isinstance(value, list):
        return [_restore_optimizer_metadata(v) for v in value]
    return value


def _normalize_lora_config_keys(raw_lora_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(raw_lora_config or {})
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        normalized[_PER_SESSION_LORA_KEY_ALIASES.get(key, key)] = value
    return normalized


def normalize_lora_runtime_config(
    raw_lora_config: Optional[Dict[str, Any]],
    *,
    default_rank: int,
    default_alpha: int,
    max_lora_rank: int,
    server_lora_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize session LoRA config to the supported per-session surface."""
    lora_config = _normalize_lora_config_keys(raw_lora_config)
    server_lora_config = dict(server_lora_config or {})

    for key in sorted(set(lora_config) - {"lora_rank", "lora_alpha"}):
        server_value = server_lora_config.get(key)
        if lora_config[key] != server_value:
            raise ValueError(
                "Per-session LoRA config may only override rank and alpha. "
                f"Unsupported override for {key!r}: {lora_config[key]!r} (server={server_value!r})."
            )

    lora_rank = int(lora_config.get("lora_rank", default_rank))
    lora_alpha = int(lora_config.get("lora_alpha", default_alpha))

    if lora_rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {lora_rank}")
    if lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {lora_alpha}")
    if lora_rank > max_lora_rank:
        raise ValueError(
            f"Requested lora_rank={lora_rank} exceeds server max_lora_rank={max_lora_rank}. "
            "Increase server.max_lora_rank to support this session."
        )

    return {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }


def normalize_optimizer_config(
    raw_optimizer_config: Optional[Dict[str, Any]],
    *,
    default_type: str,
    default_learning_rate: float,
    default_weight_decay: float,
    default_optimizer_dtype: str,
    default_optimizer_kwargs: Optional[Dict[str, Any]] = None,
    default_betas: tuple[float, float] = DEFAULT_ADAM_BETAS,
    default_eps: float = DEFAULT_ADAM_EPS,
) -> Dict[str, Any]:
    """Normalize session optimizer config to a JSON-safe runtime contract."""
    raw = dict(raw_optimizer_config or {})
    raw_optimizer_kwargs = _clone_jsonable(raw.get("optimizer_kwargs", default_optimizer_kwargs or {}))
    if not isinstance(raw_optimizer_kwargs, dict):
        raise ValueError(f"optimizer_config.optimizer_kwargs must be a dict, got {type(raw_optimizer_kwargs)!r}")
    optimizer_kwargs = dict(raw_optimizer_kwargs)

    optimizer_type = raw.get("type", default_type)
    if optimizer_type not in SUPPORTED_OPTIMIZER_TYPES:
        raise ValueError(
            f"Unsupported optimizer type {optimizer_type!r}. Supported: {sorted(SUPPORTED_OPTIMIZER_TYPES)}"
        )

    kwargs_learning_rate = optimizer_kwargs.pop("learning_rate", None)
    kwargs_lr = optimizer_kwargs.pop("lr", None)
    learning_rate = float(
        raw.get("learning_rate", raw.get("lr", kwargs_learning_rate or kwargs_lr or default_learning_rate))
    )

    kwargs_weight_decay = optimizer_kwargs.pop("weight_decay", None)
    weight_decay = float(
        raw.get("weight_decay", kwargs_weight_decay if kwargs_weight_decay is not None else default_weight_decay)
    )
    optimizer_dtype = str(raw.get("optimizer_dtype", default_optimizer_dtype))

    kwargs_betas = optimizer_kwargs.pop("betas", None)
    kwargs_adamw_betas = optimizer_kwargs.pop("adamw_betas", None)
    betas_value = raw.get("betas", kwargs_adamw_betas if kwargs_adamw_betas is not None else kwargs_betas)
    if betas_value is None:
        betas_value = default_betas

    if betas_value is not None:
        if not isinstance(betas_value, (list, tuple)) or len(betas_value) != 2:
            raise ValueError(f"optimizer_config.betas must be a length-2 list/tuple, got {betas_value!r}")
        betas = [float(betas_value[0]), float(betas_value[1])]
    else:
        betas = None

    kwargs_eps = optimizer_kwargs.pop("eps", None)
    kwargs_adamw_eps = optimizer_kwargs.pop("adamw_eps", None)
    eps_value = raw.get("eps", kwargs_adamw_eps if kwargs_adamw_eps is not None else kwargs_eps)
    if eps_value is None:
        eps_value = default_eps
    eps = float(eps_value) if eps_value is not None else None

    # These are handled by the shared optimizer factory and should not remain duplicated.
    optimizer_kwargs.pop("fused", None)
    optimizer_kwargs.pop("foreach", None)

    if optimizer_type in {"sgd", "signsgd"}:
        betas = None
        eps = None

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if optimizer_dtype not in {"bf16", "fp32"}:
        raise ValueError(f"optimizer_dtype must be 'bf16' or 'fp32', got {optimizer_dtype!r}")

    return {
        "type": optimizer_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer_dtype": optimizer_dtype,
        "betas": betas,
        "eps": eps,
        "optimizer_kwargs": optimizer_kwargs,
    }


def normalize_session_spec(
    *,
    base_model: str,
    raw_lora_config: Optional[Dict[str, Any]],
    raw_optimizer_config: Optional[Dict[str, Any]],
    default_rank: int,
    default_alpha: int,
    max_lora_rank: int,
    default_optimizer_type: str,
    default_learning_rate: float,
    default_weight_decay: float,
    default_optimizer_dtype: str,
    default_optimizer_kwargs: Optional[Dict[str, Any]],
    server_lora_config: Optional[Dict[str, Any]] = None,
    default_betas: tuple[float, float] = DEFAULT_ADAM_BETAS,
    default_eps: float = DEFAULT_ADAM_EPS,
) -> Dict[str, Any]:
    """Normalize the full per-session runtime spec."""
    return {
        "base_model": base_model,
        "is_lora": True,
        "lora_config": normalize_lora_runtime_config(
            raw_lora_config,
            default_rank=default_rank,
            default_alpha=default_alpha,
            max_lora_rank=max_lora_rank,
            server_lora_config=server_lora_config,
        ),
        "optimizer_config": normalize_optimizer_config(
            raw_optimizer_config,
            default_type=default_optimizer_type,
            default_learning_rate=default_learning_rate,
            default_weight_decay=default_weight_decay,
            default_optimizer_dtype=default_optimizer_dtype,
            default_optimizer_kwargs=default_optimizer_kwargs,
            default_betas=default_betas,
            default_eps=default_eps,
        ),
    }


def build_default_session_spec(
    *,
    base_model: str,
    train_config: Dict[str, Any],
    lora_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the default worker session spec from server config."""
    max_lora_rank = int(lora_config.get("max_lora_rank", lora_config.get("lora_rank", 32)))
    return normalize_session_spec(
        base_model=base_model,
        raw_lora_config={
            "lora_rank": lora_config.get("lora_rank", 32),
            "lora_alpha": lora_config.get("lora_alpha", 16),
        },
        raw_optimizer_config={
            "type": train_config.get("optimizer", "adamw"),
            "learning_rate": train_config.get("lr", 1e-5),
            "weight_decay": train_config.get("weight_decay", 0.01),
            "optimizer_dtype": train_config.get("optimizer_dtype", "bf16"),
            "optimizer_kwargs": train_config.get("optimizer_kwargs", {}),
        },
        default_rank=lora_config.get("lora_rank", 32),
        default_alpha=lora_config.get("lora_alpha", 16),
        max_lora_rank=max_lora_rank,
        default_optimizer_type=train_config.get("optimizer", "adamw"),
        default_learning_rate=train_config.get("lr", 1e-5),
        default_weight_decay=train_config.get("weight_decay", 0.01),
        default_optimizer_dtype=train_config.get("optimizer_dtype", "bf16"),
        default_optimizer_kwargs=train_config.get("optimizer_kwargs", {}),
        server_lora_config=lora_config,
    )


def session_optimizer_build_kwargs(optimizer_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a normalized optimizer spec to build_optimizer kwargs."""
    kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
        "optimizer_type": optimizer_config.get("type", "adamw"),
        "optimizer_dtype": optimizer_config.get("optimizer_dtype", "bf16"),
        "optimizer_kwargs": _restore_optimizer_metadata(optimizer_config.get("optimizer_kwargs", {})),
    }

    betas = optimizer_config.get("betas")
    if betas is not None:
        kwargs["betas"] = (float(betas[0]), float(betas[1]))

    eps = optimizer_config.get("eps")
    if eps is not None:
        kwargs["eps"] = float(eps)

    return kwargs


def write_session_spec(path: str, session_spec: Dict[str, Any]) -> str:
    """Write ``session_spec.json`` to a checkpoint directory."""
    os.makedirs(path, exist_ok=True)
    spec_path = os.path.join(path, SESSION_SPEC_FILENAME)
    with open(spec_path, "w") as f:
        json.dump(_clone_jsonable(session_spec), f, indent=2, sort_keys=True)
    return spec_path


def read_session_spec(path: str) -> Dict[str, Any]:
    """Read the normalized session spec from ``session_spec.json``."""
    spec_path = os.path.join(path, SESSION_SPEC_FILENAME)
    with open(spec_path, "r") as f:
        return json.load(f)


def session_spec_exists(path: str) -> bool:
    """Return whether ``session_spec.json`` exists in a checkpoint directory."""
    return os.path.exists(os.path.join(path, SESSION_SPEC_FILENAME))


def _default_betas_from_optimizer_metadata(optimizer_config: Dict[str, Any]) -> tuple[float, float]:
    """Return a safe default Adam beta tuple for legacy checkpoint upgrades."""
    betas = optimizer_config.get("betas")
    if betas is None:
        return DEFAULT_ADAM_BETAS
    return tuple(betas)


def load_session_spec_from_checkpoint(
    path: str,
    *,
    fallback_base_model: Optional[str] = None,
    fallback_session_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load session metadata from a checkpoint directory.

    New checkpoints write ``session_spec.json``. Older checkpoints are upgraded
    from ``metadata.json`` and ``adapter_config.json`` when possible.
    """
    if session_spec_exists(path):
        return read_session_spec(path)

    metadata_path = os.path.join(path, "metadata.json")
    adapter_config_path = os.path.join(path, "adapter_config.json")

    metadata: Dict[str, Any] = {}
    adapter_config: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

    fallback_session_spec = dict(fallback_session_spec or {})
    fallback_lora = dict(fallback_session_spec.get("lora_config") or {})
    fallback_optimizer = dict(fallback_session_spec.get("optimizer_config") or {})
    has_lora_artifacts = os.path.exists(os.path.join(path, "adapter_model.safetensors")) or bool(adapter_config)

    base_model = (
        adapter_config.get("base_model_name_or_path") or fallback_base_model or fallback_session_spec.get("base_model")
    )
    if not base_model and not has_lora_artifacts:
        raise FileNotFoundError(
            f"Checkpoint at {path} does not contain {SESSION_SPEC_FILENAME} and no base_model fallback was provided."
        )
    base_model = base_model or ""

    if not has_lora_artifacts:
        return {
            "base_model": base_model,
            "is_lora": False,
        }

    optimizer_metadata = metadata.get("optimizer") or {}
    optimizer_config = normalize_optimizer_config(
        {
            "type": optimizer_metadata.get("type", fallback_optimizer.get("type", "adamw")),
            "learning_rate": metadata.get("lr", fallback_optimizer.get("learning_rate", 1e-5)),
            "weight_decay": optimizer_metadata.get("weight_decay", fallback_optimizer.get("weight_decay", 0.01)),
            "optimizer_dtype": optimizer_metadata.get(
                "dtype",
                fallback_optimizer.get("optimizer_dtype", "bf16"),
            ),
            "betas": optimizer_metadata.get("betas", fallback_optimizer.get("betas")),
            "eps": optimizer_metadata.get("eps", fallback_optimizer.get("eps")),
            "optimizer_kwargs": optimizer_metadata.get(
                "optimizer_kwargs",
                fallback_optimizer.get("optimizer_kwargs", {}),
            ),
        },
        default_type=fallback_optimizer.get("type", "adamw"),
        default_learning_rate=fallback_optimizer.get("learning_rate", 1e-5),
        default_weight_decay=fallback_optimizer.get("weight_decay", 0.01),
        default_optimizer_dtype=fallback_optimizer.get("optimizer_dtype", "bf16"),
        default_optimizer_kwargs=fallback_optimizer.get("optimizer_kwargs", {}),
        default_betas=_default_betas_from_optimizer_metadata(fallback_optimizer)
        if fallback_optimizer
        else DEFAULT_ADAM_BETAS,
        default_eps=float(fallback_optimizer.get("eps", DEFAULT_ADAM_EPS))
        if fallback_optimizer.get("eps") is not None
        else DEFAULT_ADAM_EPS,
    )

    lora_config = normalize_lora_runtime_config(
        {
            "lora_rank": adapter_config.get("r", fallback_lora.get("lora_rank", 32)),
            "lora_alpha": adapter_config.get(
                "lora_alpha", fallback_lora.get("lora_alpha", adapter_config.get("r", 32))
            ),
        },
        default_rank=fallback_lora.get("lora_rank", 32),
        default_alpha=fallback_lora.get("lora_alpha", 16),
        max_lora_rank=max(
            int(adapter_config.get("r", fallback_lora.get("lora_rank", 32))),
            int(fallback_lora.get("lora_rank", 32)),
        ),
    )

    return {
        "base_model": base_model,
        "is_lora": True,
        "lora_config": lora_config,
        "optimizer_config": optimizer_config,
    }
