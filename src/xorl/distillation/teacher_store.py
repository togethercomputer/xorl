from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TEACHER_STORE_MANIFEST = "manifest.json"
TEACHER_STORE_TYPE = "xorl_opd_teacher_store"
TEACHER_STORE_VERSION = 1
DEFAULT_LM_HEAD_KEY = "lm_head.weight"


def _dtype_name(dtype: Any) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    return str(dtype)


def _resolve(path: str | os.PathLike[str], base_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base_dir / p


def _find_tensor_source(path: str | os.PathLike[str], key: str) -> tuple[Path, str]:
    """Return the safetensors file and key containing ``key``.

    ``path`` can be a Hugging Face model directory, a single safetensors file, or
    an already-materialized teacher-store manifest/directory.
    """
    p = Path(path)
    if p.is_dir():
        index_path = p / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_name = index.get("weight_map", {}).get(key)
            if shard_name is None:
                raise KeyError(f"Could not find tensor key '{key}' in {index_path}")
            return p / shard_name, key

        safetensors_path = p / "model.safetensors"
        if safetensors_path.exists():
            return safetensors_path, key

    if p.suffix == ".safetensors":
        return p, key

    raise FileNotFoundError(f"No safetensors source found for key '{key}' at {p}")


@dataclass(frozen=True)
class TeacherHeadShard:
    path: Path
    tensor_key: str
    start: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.start

    def load_cpu(self) -> torch.Tensor:
        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            if self.tensor_key not in f.keys():
                keys = list(f.keys())
                if len(keys) != 1:
                    raise KeyError(
                        f"Could not find tensor key '{self.tensor_key}' in {self.path}. Available keys: {keys[:10]}"
                    )
                return f.get_tensor(keys[0]).contiguous()
            return f.get_tensor(self.tensor_key).contiguous()


@dataclass(frozen=True)
class TeacherHeadSpec:
    teacher_id: str
    shape: tuple[int, int]
    dtype: str
    shards: tuple[TeacherHeadShard, ...]

    @property
    def vocab_size(self) -> int:
        return self.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.shape[1]


class TeacherHeadStore:
    """Row-sharded teacher LM-head store for OPD.

    The store keeps the teacher prediction head in vocab-row shards so OPD can
    later stream only the shard needed by a vocab block instead of materializing
    every teacher head on the GPU.
    """

    def __init__(self, manifest_path: str | os.PathLike[str]) -> None:
        path = Path(manifest_path)
        if path.is_dir():
            path = path / TEACHER_STORE_MANIFEST
        self.manifest_path = path
        self.root = path.parent
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("type") != TEACHER_STORE_TYPE:
            raise ValueError(f"{path} is not an XORL OPD teacher store manifest")
        if int(self.manifest.get("version", 0)) != TEACHER_STORE_VERSION:
            raise ValueError(
                f"Unsupported teacher-store version {self.manifest.get('version')}; expected {TEACHER_STORE_VERSION}"
            )
        self._cpu_cache: Dict[tuple[str, str, int, int, bool], torch.Tensor] = {}

    def teacher_ids(self) -> list[str]:
        return list(self.manifest.get("teachers", {}).keys())

    def head_spec(self, teacher_id: int | str) -> TeacherHeadSpec:
        key = str(int(teacher_id)) if isinstance(teacher_id, torch.Tensor) else str(teacher_id)
        teachers = self.manifest.get("teachers", {})
        if key not in teachers:
            raise KeyError(f"No teacher_id={key} in teacher store {self.manifest_path}")
        head = teachers[key]["lm_head"]
        shards = tuple(
            TeacherHeadShard(
                path=_resolve(shard["path"], self.root),
                tensor_key=shard.get("tensor_key", DEFAULT_LM_HEAD_KEY),
                start=int(shard["start"]),
                end=int(shard["end"]),
            )
            for shard in head["shards"]
        )
        return TeacherHeadSpec(
            teacher_id=key,
            shape=tuple(int(x) for x in head["shape"]),
            dtype=head["dtype"],
            shards=shards,
        )

    def iter_lm_head_shards(self, teacher_id: int | str) -> Iterator[tuple[int, int, torch.Tensor]]:
        for shard in self.head_spec(teacher_id).shards:
            yield shard.start, shard.end, shard.load_cpu()

    def load_shard_cpu(
        self,
        shard: TeacherHeadShard,
        *,
        pin_memory: bool = False,
        cache: bool = True,
    ) -> torch.Tensor:
        key = (str(shard.path), shard.tensor_key, shard.start, shard.end, pin_memory)
        if cache and key in self._cpu_cache:
            return self._cpu_cache[key]

        tensor = shard.load_cpu()
        if pin_memory:
            tensor = tensor.pin_memory()
        if cache:
            self._cpu_cache[key] = tensor
        return tensor

    def load_lm_head(self, teacher_id: int | str) -> torch.Tensor:
        spec = self.head_spec(teacher_id)
        pieces = []
        expected_start = 0
        for shard in spec.shards:
            if shard.start != expected_start:
                raise ValueError(
                    f"Teacher store {self.manifest_path} has non-contiguous lm_head shards: "
                    f"expected start {expected_start}, got {shard.start}"
                )
            tensor = shard.load_cpu()
            if tensor.shape[0] != shard.rows:
                raise ValueError(f"Shard {shard.path} rows {tensor.shape[0]} do not match manifest rows {shard.rows}")
            pieces.append(tensor)
            expected_start = shard.end
        out = torch.cat(pieces, dim=0).contiguous()
        if tuple(out.shape) != spec.shape:
            raise ValueError(f"Reconstructed lm_head shape {tuple(out.shape)} does not match manifest {spec.shape}")
        return out


@dataclass(frozen=True)
class TeacherHeadShardView:
    """Reusable view over a row-sharded teacher head.

    Shards are loaded lazily. CPU shard caching avoids repeated safetensors reads
    across the multi-pass KL algorithm; optional device caching keeps one
    teacher head resident for this view's forward/backward lifetime.
    """

    store: TeacherHeadStore
    teacher_id: str
    device: torch.device
    dtype: Optional[torch.dtype]
    cache_cpu: bool = True
    cache_device: bool = False
    _cpu_cache: Dict[tuple[str, str, int, int], torch.Tensor] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _device_cache: Dict[tuple[str, str, int, int, str, str], torch.Tensor] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    @property
    def shape(self) -> tuple[int, int]:
        return self.store.head_spec(self.teacher_id).shape

    def _cpu_tensor(self, shard: TeacherHeadShard) -> torch.Tensor:
        key = (str(shard.path), shard.tensor_key, shard.start, shard.end)
        if self.cache_cpu and key in self._cpu_cache:
            return self._cpu_cache[key]

        pin_memory = self.device.type == "cuda" and torch.cuda.is_available()
        cpu_tensor = self.store.load_shard_cpu(
            shard,
            pin_memory=pin_memory,
            cache=self.cache_cpu,
        )
        if self.cache_cpu:
            self._cpu_cache[key] = cpu_tensor
        return cpu_tensor

    def _device_tensor(self, shard: TeacherHeadShard, cpu_tensor: torch.Tensor) -> torch.Tensor:
        dtype_name = str(self.dtype) if self.dtype is not None else str(cpu_tensor.dtype)
        key = (str(shard.path), shard.tensor_key, shard.start, shard.end, str(self.device), dtype_name)
        if self.cache_device and key in self._device_cache:
            return self._device_cache[key]

        tensor = cpu_tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if self.cache_device:
            self._device_cache[key] = tensor
        return tensor

    def iter_device_chunks(self, chunk_rows: int) -> Iterator[tuple[int, int, torch.Tensor]]:
        if chunk_rows <= 0:
            chunk_rows = self.shape[0]
        for shard in self.store.head_spec(self.teacher_id).shards:
            tensor = self._device_tensor(shard, self._cpu_tensor(shard))
            local_rows = shard.rows
            for local_start in range(0, local_rows, chunk_rows):
                local_end = min(local_start + chunk_rows, local_rows)
                yield (
                    shard.start + local_start,
                    shard.start + local_end,
                    tensor[local_start:local_end],
                )

    def clear_device_cache(self) -> None:
        self._device_cache.clear()


def teacher_store_manifest_path(entry: str | os.PathLike[str] | Mapping[str, Any]) -> Optional[Path]:
    explicit_store_path = False
    if isinstance(entry, Mapping):
        store_path = entry.get("store_path") or entry.get("teacher_store") or entry.get("manifest_path")
        if store_path is None and entry.get("type") == TEACHER_STORE_TYPE:
            store_path = entry.get("path")
        if store_path is None:
            return None
        explicit_store_path = True
        path = Path(store_path)
    else:
        path = Path(entry)

    if explicit_store_path:
        return path / TEACHER_STORE_MANIFEST if path.suffix == "" else path
    if path.is_dir() and (path / TEACHER_STORE_MANIFEST).exists():
        return path / TEACHER_STORE_MANIFEST
    if path.is_file() and path.name == TEACHER_STORE_MANIFEST:
        return path
    return None


def is_teacher_store_entry(entry: str | os.PathLike[str] | Mapping[str, Any]) -> bool:
    return teacher_store_manifest_path(entry) is not None


def load_lm_head_from_teacher_store(
    entry: str | os.PathLike[str] | Mapping[str, Any], teacher_id: int | str
) -> torch.Tensor:
    manifest_path = teacher_store_manifest_path(entry)
    if manifest_path is None:
        raise ValueError(f"Entry is not a teacher store: {entry}")
    return TeacherHeadStore(manifest_path).load_lm_head(teacher_id)


def prepare_lm_head_teacher_store(
    model_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    teacher_id: int | str = 0,
    shard_rows: int = 32768,
    tensor_key: str = DEFAULT_LM_HEAD_KEY,
    force: bool = False,
) -> Path:
    """Slice a teacher ``lm_head.weight`` into a row-sharded OPD teacher store."""
    if shard_rows <= 0:
        raise ValueError(f"shard_rows must be positive, got {shard_rows}")

    output = Path(output_dir)
    manifest_path = output / TEACHER_STORE_MANIFEST
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists; pass force=True to overwrite")

    source_path, source_key = _find_tensor_source(model_path, tensor_key)
    output.mkdir(parents=True, exist_ok=True)
    teacher_key = str(teacher_id)
    teacher_dir = output / f"teacher_{teacher_key}"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    shards: list[Dict[str, Any]] = []
    with safe_open(str(source_path), framework="pt", device="cpu") as f:
        if source_key not in f.keys():
            keys = list(f.keys())
            if len(keys) != 1:
                raise KeyError(
                    f"Could not find tensor key '{source_key}' in {source_path}. Available keys: {keys[:10]}"
                )
            source_key = keys[0]
        tensor_slice = f.get_slice(source_key)
        shape = tuple(int(x) for x in tensor_slice.get_shape())
        if len(shape) != 2:
            raise ValueError(f"Teacher LM head must be rank 2 [vocab, hidden], got {shape}")
        dtype = _dtype_name(tensor_slice.get_dtype())

        for shard_idx, start in enumerate(range(0, shape[0], shard_rows)):
            end = min(start + shard_rows, shape[0])
            tensor = tensor_slice[start:end].contiguous()
            shard_name = f"lm_head_{shard_idx:05d}.safetensors"
            shard_path = teacher_dir / shard_name
            save_file({tensor_key: tensor}, str(shard_path))
            shards.append(
                {
                    "path": str(shard_path.relative_to(output)),
                    "tensor_key": tensor_key,
                    "start": start,
                    "end": end,
                }
            )

    manifest = {
        "type": TEACHER_STORE_TYPE,
        "version": TEACHER_STORE_VERSION,
        "source": str(model_path),
        "teachers": {
            teacher_key: {
                "lm_head": {
                    "tensor_key": tensor_key,
                    "shape": list(shape),
                    "dtype": dtype,
                    "shards": shards,
                }
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
