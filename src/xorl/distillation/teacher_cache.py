from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from safetensors import safe_open

from xorl.distillation.teacher_store import (
    TeacherHeadShardView,
    TeacherHeadStore,
    is_teacher_store_entry,
    load_lm_head_from_teacher_store,
    teacher_store_manifest_path,
)


DEFAULT_LM_HEAD_KEY = "lm_head.weight"
DEFAULT_HIDDEN_KEY = "hidden_states"
TIED_EMBEDDING_LM_HEAD_KEYS = (
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "transformer.wte.weight",
)


def _load_safetensors_tensor(path: str, key: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as f:
        if key not in f.keys():
            keys = list(f.keys())
            if len(keys) == 1:
                return f.get_tensor(keys[0])
            raise KeyError(f"Could not find tensor key '{key}' in {path}. Available keys: {keys[:10]}")
        return f.get_tensor(key)


def _load_from_model_dir(path: str, key: str) -> torch.Tensor:
    def tied_lm_head_fallback(available_keys: set[str]) -> str | None:
        if key != DEFAULT_LM_HEAD_KEY or not _model_dir_ties_word_embeddings(path):
            return None
        for candidate in TIED_EMBEDDING_LM_HEAD_KEYS:
            if candidate in available_keys:
                return candidate
        return None

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        load_key = key
        shard_name = weight_map.get(load_key)
        if shard_name is None:
            fallback_key = tied_lm_head_fallback(set(weight_map))
            if fallback_key is None:
                raise KeyError(f"Could not find tensor key '{key}' in {index_path}")
            load_key = fallback_key
            shard_name = weight_map[load_key]
        return _load_safetensors_tensor(os.path.join(path, shard_name), load_key)

    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.exists(safetensors_path):
        try:
            return _load_safetensors_tensor(safetensors_path, key)
        except KeyError:
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                fallback_key = tied_lm_head_fallback(set(f.keys()))
            if fallback_key is None:
                raise
            return _load_safetensors_tensor(safetensors_path, fallback_key)

    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json found in {path}")


def _model_dir_ties_word_embeddings(path: str) -> bool:
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        config = json.load(f)
    if bool(config.get("tie_word_embeddings", False)):
        return True
    for nested_key in ("text_config", "llm_config"):
        nested = config.get(nested_key)
        if isinstance(nested, Mapping) and bool(nested.get("tie_word_embeddings", False)):
            return True
    return False


def _normalize_entry(entry: str | Mapping[str, Any], default_key: str) -> tuple[str, str]:
    if isinstance(entry, str):
        return entry, default_key
    path = entry.get("path") or entry.get("model_path") or entry.get("weights_path")
    if not path:
        raise ValueError(f"Teacher entry must include a path: {entry}")
    return path, entry.get("tensor_key", default_key)


def load_lm_head_weight(
    entry: str | Mapping[str, Any],
    tensor_key: str = DEFAULT_LM_HEAD_KEY,
    teacher_id: int | str = 0,
) -> torch.Tensor:
    """Load a teacher LM head from a local file or Hugging Face-style model directory."""
    if is_teacher_store_entry(entry):
        return load_lm_head_from_teacher_store(entry, teacher_id=teacher_id).contiguous()

    path, key = _normalize_entry(entry, tensor_key)
    if os.path.isdir(path):
        tensor = _load_from_model_dir(path, key)
    elif path.endswith(".safetensors"):
        tensor = _load_safetensors_tensor(path, key)
    else:
        raise ValueError(f"Teacher LM head must be a safetensors file or model directory: {path}")
    return tensor.contiguous()


def load_hidden_state_cache(entry: str | Mapping[str, Any], tensor_key: str = DEFAULT_HIDDEN_KEY) -> torch.Tensor:
    """Load a teacher hidden-state cache tensor of shape [num_cached_tokens, hidden_dim]."""
    path, key = _normalize_entry(entry, tensor_key)
    if os.path.isdir(path):
        safetensors_path = os.path.join(path, "hidden_states.safetensors")
        if os.path.exists(safetensors_path):
            tensor = _load_safetensors_tensor(safetensors_path, key)
        else:
            raise FileNotFoundError(f"No hidden_states.safetensors found in {path}")
    elif path.endswith(".safetensors"):
        tensor = _load_safetensors_tensor(path, key)
    else:
        raise ValueError(f"Teacher hidden cache must be a safetensors file or directory: {path}")
    if tensor.ndim not in (2, 3):
        raise ValueError(
            "Teacher hidden cache must be rank 2 [tokens, hidden_dim] or rank 3 "
            f"[layers, tokens, hidden_dim] (multi-layer OPRD), got shape {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


@dataclass
class TeacherHeadManager:
    """Holds teacher LM heads on CPU; promotes one at a time to device.

    Single-teacher device cache is intentional for the MVP: keeping all teacher
    heads on device costs vocab_size * hidden_size per teacher (~600 MB for
    Qwen3 vocab=151936, hidden=2048, fp32). Multi-teacher batches will thrash
    on this re-upload; revisit with an LRU keyed on teacher_id when those
    workloads land.
    """

    teacher_heads: Mapping[str, Any]
    enable_async: bool = True
    max_workers: int = 2

    def __post_init__(self) -> None:
        self.teacher_heads = {str(k): v for k, v in self.teacher_heads.items()}
        self._cpu_cache: Dict[str, torch.Tensor] = {}
        self._cpu_futures: Dict[str, Future] = {}
        self._store_cache: Dict[str, TeacherHeadStore] = {}
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=max(1, int(self.max_workers)), thread_name_prefix="opd-teacher-head")
            if self.enable_async
            else None
        )
        self._device_teacher_id: Optional[str] = None
        self._device_tensor: Optional[torch.Tensor] = None

    @staticmethod
    def _key(teacher_id: int | str | torch.Tensor) -> str:
        return str(int(teacher_id)) if isinstance(teacher_id, torch.Tensor) else str(teacher_id)

    def _load_cpu(self, key: str) -> torch.Tensor:
        cpu = load_lm_head_weight(self.teacher_heads[key], teacher_id=key)
        if torch.cuda.is_available():
            cpu = cpu.pin_memory()
        return cpu

    def has_sharded_head(self, teacher_id: int | str | torch.Tensor) -> bool:
        key = self._key(teacher_id)
        return key in self.teacher_heads and is_teacher_store_entry(self.teacher_heads[key])

    def sharded_view(
        self,
        teacher_id: int | str | torch.Tensor,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
        cache_cpu: bool = True,
        cache_device: bool = False,
    ) -> TeacherHeadShardView:
        key = self._key(teacher_id)
        if key not in self.teacher_heads:
            raise KeyError(f"No teacher head configured for teacher_id={key}")
        manifest_path = teacher_store_manifest_path(self.teacher_heads[key])
        if manifest_path is None:
            raise ValueError(f"teacher_id={key} is not configured with a teacher store")
        store_key = str(manifest_path)
        if store_key not in self._store_cache:
            self._store_cache[store_key] = TeacherHeadStore(manifest_path)
        return TeacherHeadShardView(
            store=self._store_cache[store_key],
            teacher_id=key,
            device=torch.device(device),
            dtype=dtype,
            cache_cpu=cache_cpu,
            cache_device=cache_device,
        )

    def prefetch(self, teacher_id: int | str | torch.Tensor) -> None:
        """Start loading a teacher head into pinned CPU memory."""
        key = self._key(teacher_id)
        if key not in self.teacher_heads:
            raise KeyError(f"No teacher head configured for teacher_id={key}")
        if key in self._cpu_cache or key in self._cpu_futures:
            return
        if self._executor is None:
            self._cpu_cache[key] = self._load_cpu(key)
        else:
            self._cpu_futures[key] = self._executor.submit(self._load_cpu, key)

    def _cpu_tensor(self, key: str) -> torch.Tensor:
        if key in self._cpu_cache:
            return self._cpu_cache[key]
        if key in self._cpu_futures:
            self._cpu_cache[key] = self._cpu_futures.pop(key).result()
            return self._cpu_cache[key]
        self._cpu_cache[key] = self._load_cpu(key)
        return self._cpu_cache[key]

    def get(
        self, teacher_id: int | str, device: torch.device | str, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        key = self._key(teacher_id)
        if key not in self.teacher_heads:
            raise KeyError(f"No teacher head configured for teacher_id={key}")
        cpu = self._cpu_tensor(key)

        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None and torch.cuda.is_available():
            target_device = torch.device("cuda", torch.cuda.current_device())
        needs_upload = (
            self._device_teacher_id != key
            or self._device_tensor is None
            or self._device_tensor.device != target_device
            or (dtype is not None and self._device_tensor.dtype != dtype)
        )
        if needs_upload:
            self._device_tensor = None
            self._device_teacher_id = key
            self._device_tensor = cpu.to(device=target_device, dtype=dtype, non_blocking=True)
        return self._device_tensor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


@dataclass
class TeacherActivationCache:
    hidden_caches: Mapping[str, Any] | str
    enable_async: bool = True
    max_workers: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_caches, str):
            self.hidden_caches = {str(k): v for k, v in self.hidden_caches.items()}
        self._cpu_cache: Dict[str, torch.Tensor] = {}
        self._cpu_futures: Dict[str, Future] = {}
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=max(1, int(self.max_workers)), thread_name_prefix="opd-teacher-hidden")
            if self.enable_async
            else None
        )

    def _entry_for_teacher(self, teacher_id: int | str) -> tuple[str, Any]:
        key = str(int(teacher_id)) if isinstance(teacher_id, torch.Tensor) else str(teacher_id)
        if isinstance(self.hidden_caches, str):
            return key, self.hidden_caches
        if key not in self.hidden_caches:
            raise KeyError(f"No teacher hidden cache configured for teacher_id={key}")
        return key, self.hidden_caches[key]

    def _load_cpu(self, key: str, entry: Any) -> torch.Tensor:
        return load_hidden_state_cache(entry)

    def prefetch(self, teacher_id: int | str | torch.Tensor) -> None:
        """Start loading a teacher hidden-state cache into CPU memory."""
        key, entry = self._entry_for_teacher(teacher_id)
        if key in self._cpu_cache or key in self._cpu_futures:
            return
        if self._executor is None:
            self._cpu_cache[key] = self._load_cpu(key, entry)
        else:
            self._cpu_futures[key] = self._executor.submit(self._load_cpu, key, entry)

    def _cpu_tensor(self, key: str, entry: Any) -> torch.Tensor:
        if key in self._cpu_cache:
            return self._cpu_cache[key]
        if key in self._cpu_futures:
            self._cpu_cache[key] = self._cpu_futures.pop(key).result()
            return self._cpu_cache[key]
        self._cpu_cache[key] = self._load_cpu(key, entry)
        return self._cpu_cache[key]

    def get(
        self,
        teacher_id: int | str,
        indices: torch.Tensor,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        key, entry = self._entry_for_teacher(teacher_id)
        cache = self._cpu_tensor(key, entry)

        flat_indices = indices.reshape(-1).to(device="cpu", dtype=torch.long)
        # rank-2 [tokens, d] (output-space OPD / last-layer hidden-match) gathers along
        # dim 0; rank-3 [layers, tokens, d] (multi-layer OPRD) gathers along the token
        # axis (dim 1) and returns [*indices.shape, layers, d].
        token_dim = 0 if cache.ndim == 2 else 1
        num_tokens = cache.shape[token_dim]
        if flat_indices.numel() > 0:
            min_idx = flat_indices.min().item()
            if min_idx < 0:
                # Negative indices used to be silently clamped to 0, which masked
                # producer bugs (off-by-one in teacher_cache_indices construction
                # was found this way during the Countdown run). Fail loudly instead.
                raise IndexError(
                    f"teacher_cache_indices contain negative value {min_idx} "
                    f"(teacher_id={key}); producer must emit non-negative indices"
                )
            max_idx = flat_indices.max().item()
            if max_idx >= num_tokens:
                raise IndexError(
                    f"teacher_cache_indices contain {max_idx}, "
                    f"but teacher_id={key} cache only has {num_tokens} rows"
                )
        gathered = cache.index_select(token_dim, flat_indices)
        if cache.ndim == 2:
            gathered = gathered.view(*indices.shape, cache.shape[-1])
        else:
            # [layers, n, d] -> [n, layers, d] -> [*indices.shape, layers, d]
            layers, _, hidden = gathered.shape
            gathered = gathered.permute(1, 0, 2).reshape(*indices.shape, layers, hidden)
        # CPU cache is unpinned, so non_blocking=True would be a no-op. Drop the
        # flag rather than misleading future readers.
        return gathered.to(device=device, dtype=dtype)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
