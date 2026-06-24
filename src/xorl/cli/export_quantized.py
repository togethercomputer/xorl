"""Export Hugging Face safetensors with native block-FP8 weights.

This CLI implements the offline version of XoRL's online SGLang FP8 sync
contract: selected 2D ``*.weight`` tensors are stored as E4M3 FP8 plus FP32
``*.weight_scale_inv`` block scales, while BF16 islands remain in their source
dtype.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from xorl.server.weight_sync.quantization_config import (
    UnsupportedSyncQuantizationError,
    normalize_sync_quantization_config,
)


_LAYER_NAME_RE = re.compile(r"(?P<prefix>(?:^|.*\.)layers)\.(?P<idx>\d+)\.")
_SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kmgt]?i?b?)?\s*$", re.IGNORECASE)
_WEIGHT_INDEX_NAME = "model.safetensors.index.json"
_DEFAULT_MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024
_QARL_STATE_SUFFIXES = (
    ".qarl_input_amax",
    ".qarl_weight_amax",
    ".qarl_input_scale_inv",
    ".qarl_weight_scale_inv",
    ".qarl_forward_count",
)
_MTP_COUNT_FIELDS = ("mtp_num_hidden_layers", "mtp_num_layers", "num_nextn_predict_layers")
_EXPORT_CONFIG_KEYS = {
    "input_dir",
    "output_dir",
    "fmt",
    "weight_block_size",
    "modules_to_not_convert",
    "module_to_not_convert",
    "num_first_layers_bf16",
    "num_last_layers_bf16",
    "fold_qarl",
    "qarl_quant_cfg",
    "max_shard_size",
    "overwrite",
}


@dataclass(frozen=True)
class QuantizedExportResult:
    output_dir: str
    tensors_read: int
    tensors_written: int
    quantized_weights: int
    passthrough_tensors: int
    shard_count: int
    total_size: int
    quantization_config: dict[str, Any]
    qarl_folded: bool = False
    qarl_modules: list[str] = field(default_factory=list)
    qarl_state_tensors: int = 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_export_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(payload)
    else:
        data = yaml.safe_load(payload)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping")

    unknown = sorted(set(data) - _EXPORT_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unsupported export config keys: {unknown}")
    normalized = dict(data)
    singular_skip = normalized.pop("module_to_not_convert", None)
    if singular_skip is not None and "modules_to_not_convert" not in normalized:
        normalized["modules_to_not_convert"] = singular_skip
    return normalized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_size_bytes(value: str | int | None) -> int:
    if value is None:
        return _DEFAULT_MAX_SHARD_SIZE
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max shard size must be positive")
        return value

    match = _SIZE_RE.match(value)
    if match is None:
        raise ValueError(f"Invalid size {value!r}; expected bytes or a value like 5GB, 512MiB")
    amount = float(match.group("value"))
    unit = (match.group("unit") or "b").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Invalid size unit in {value!r}")
    size = int(amount * multipliers[unit])
    if size <= 0:
        raise ValueError("max shard size must be positive")
    return size


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _copy_hf_assets(input_dir: Path, output_dir: Path) -> None:
    skip_names = {
        _WEIGHT_INDEX_NAME,
        "pytorch_model.bin.index.json",
        "tf_model.h5",
        "flax_model.msgpack",
    }
    for src in input_dir.iterdir():
        if not src.is_file():
            continue
        if src.name in skip_names or src.suffix == ".safetensors" or src.name.startswith("pytorch_model"):
            continue
        shutil.copy2(src, output_dir / src.name)


def _read_config(input_dir: Path) -> dict[str, Any]:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        return {}
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return config


def _is_qarl_state_tensor(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _QARL_STATE_SUFFIXES)


def _qarl_module_name_from_state(name: str) -> str | None:
    for suffix in _QARL_STATE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _module_name_from_weight(name: str) -> str | None:
    if not name.endswith(".weight"):
        return None
    return name[: -len(".weight")]


def _detect_qarl_modules(weight_names: list[str]) -> tuple[list[str], list[str]]:
    qarl_state_names = [name for name in weight_names if _is_qarl_state_tensor(name)]
    modules = sorted({module for name in qarl_state_names if (module := _qarl_module_name_from_state(name))})
    return modules, qarl_state_names


def _resolve_qarl_quant_cfg(config: dict[str, Any], qarl_quant_cfg: str | dict[str, Any] | None) -> dict[str, Any]:
    from xorl.qarl import normalize_qarl_quant_cfg  # noqa: PLC0415

    if qarl_quant_cfg is not None:
        normalized = normalize_qarl_quant_cfg(qarl_quant_cfg)
    else:
        raw_cfg = None
        for key in ("xorl_qarl_config", "qarl_config"):
            value = config.get(key)
            if isinstance(value, dict):
                raw_cfg = value.get("quant_cfg", value.get("qarl_quant_cfg", value))
                break
        if raw_cfg is None:
            raw_cfg = config.get("qarl_quant_cfg")
        normalized = normalize_qarl_quant_cfg(raw_cfg)

    if not normalized.get("weight", True):
        raise ValueError("QARL folded export requires qarl_quant_cfg.weight=true")
    return normalized


def _qarl_modules_to_not_convert(weight_names: list[str], qarl_modules: list[str]) -> list[str]:
    qarl_module_set = set(qarl_modules)
    skipped: list[str] = []
    seen: set[str] = set()
    for name in weight_names:
        module_name = _module_name_from_weight(name)
        if module_name is None or module_name in qarl_module_set or _is_qarl_state_tensor(name):
            continue
        if module_name not in seen:
            seen.add(module_name)
            skipped.append(module_name)
    return skipped


def _parse_qarl_quant_cfg(value: str | None) -> str | dict[str, Any] | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped == "null":
        return None
    if stripped.startswith("{"):
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError("--qarl-quant-cfg JSON must be an object")
        return loaded
    return stripped


def _hidden_layer_count(config: dict[str, Any]) -> int | None:
    for section in (config, config.get("text_config"), config.get("llm_config")):
        if not isinstance(section, dict):
            continue
        value = section.get("num_hidden_layers")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _attention_qkv_split_sizes(config: dict[str, Any]) -> tuple[int, int]:
    for section in (config, config.get("text_config"), config.get("llm_config")):
        if not isinstance(section, dict):
            continue
        num_heads = section.get("num_attention_heads")
        hidden_size = section.get("hidden_size")
        head_dim = section.get("head_dim")
        num_kv_heads = section.get("num_key_value_heads", num_heads)
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            continue
        if head_dim is None:
            if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
                continue
            head_dim = hidden_size // num_heads
        if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
            continue
        if not isinstance(num_kv_heads, int) or isinstance(num_kv_heads, bool) or num_kv_heads <= 0:
            continue
        return num_heads * head_dim, num_kv_heads * head_dim

    raise ValueError(
        "Cannot split qkv_proj for HF export: config must define num_attention_heads and either head_dim or hidden_size"
    )


def _config_has_q_lora_rank(config: dict[str, Any]) -> bool:
    for section in (config, config.get("text_config"), config.get("llm_config")):
        if not isinstance(section, dict):
            continue
        if section.get("q_lora_rank") is not None:
            return True
    return False


def _config_has_linear_attention_layers(config: dict[str, Any]) -> bool:
    for section in (config, config.get("text_config"), config.get("llm_config")):
        if not isinstance(section, dict):
            continue
        layer_types = section.get("layer_types")
        if isinstance(layer_types, list) and any(layer_type == "linear_attention" for layer_type in layer_types):
            return True
    return False


def _positive_int_config_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        try:
            return int(value.strip()) > 0
        except ValueError:
            return False
    return False


def _looks_like_mtp_name(name: str) -> bool:
    parts = [part.strip().lower() for part in name.replace("/", ".").split(".") if part.strip()]
    return any(part == "mtp" or part.startswith("mtp_") or part == "nextn" or part.startswith("nextn_") for part in parts)


def _mtp_evidence_from_config(config: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    for prefix, section in (
        ("config", config),
        ("text_config", config.get("text_config")),
        ("llm_config", config.get("llm_config")),
    ):
        if not isinstance(section, Mapping):
            continue
        for key in _MTP_COUNT_FIELDS:
            value = section.get(key)
            if _positive_int_config_value(value):
                evidence.append(f"{prefix}.{key}={value}")
        quant_config = section.get("quantization_config")
        if isinstance(quant_config, Mapping):
            evidence.extend(
                _mtp_evidence_from_modules(
                    quant_config.get("modules_to_not_convert"),
                    prefix=f"{prefix}.quantization_config.modules_to_not_convert",
                )
            )
    return evidence


def _mtp_evidence_from_modules(value: Any, *, prefix: str) -> list[str]:
    if isinstance(value, str):
        entries = [value]
    elif isinstance(value, Collection) and not isinstance(value, Mapping):
        entries = [entry for entry in value if isinstance(entry, str)]
    else:
        return []
    mtp_entries = [entry for entry in entries if _looks_like_mtp_name(entry)]
    if not mtp_entries:
        return []
    preview = ", ".join(mtp_entries[:3])
    suffix = ", ..." if len(mtp_entries) > 3 else ""
    return [f"{prefix} includes {preview}{suffix}"]


def _unsupported_mtp_export_reason(
    config: dict[str, Any],
    weight_names: list[str],
    modules_to_not_convert: list[str] | None,
) -> str | None:
    evidence = _mtp_evidence_from_config(config)
    evidence.extend(_mtp_evidence_from_modules(modules_to_not_convert, prefix="modules_to_not_convert"))
    mtp_weight_names = [name for name in weight_names if _looks_like_mtp_name(name)]
    if mtp_weight_names:
        preview = ", ".join(mtp_weight_names[:3])
        suffix = ", ..." if len(mtp_weight_names) > 3 else ""
        evidence.append(f"weights include {preview}{suffix}")
    if not evidence:
        return None
    return (
        "MTP/speculative low-precision export is not implemented. "
        f"Detected {evidence[0]}. Enumerate the receiver-visible MTP tensors and add a same-weight "
        "speculative SGLang validation gate before enabling this export."
    )


def _tensor_entries(input_dir: Path) -> list[tuple[str, Path]]:
    index_path = input_dir / _WEIGHT_INDEX_NAME
    if index_path.exists():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} must contain a non-empty weight_map object")
        return [(str(name), input_dir / str(shard)) for name, shard in weight_map.items()]

    candidates = sorted(input_dir.glob("model*.safetensors"))
    if not candidates:
        candidates = sorted(input_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No safetensors weights found under {input_dir}")

    entries: list[tuple[str, Path]] = []
    for shard_path in candidates:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            entries.extend((name, shard_path) for name in handle.keys())
    return entries


def _infer_layer_prefix(weight_names: list[str]) -> tuple[str, set[int]] | None:
    by_prefix: dict[str, set[int]] = {}
    for name in weight_names:
        match = _LAYER_NAME_RE.search(name)
        if match is None:
            continue
        by_prefix.setdefault(match.group("prefix"), set()).add(int(match.group("idx")))
    if not by_prefix:
        return None
    return max(by_prefix.items(), key=lambda item: (len(item[1]), item[0] == "model.layers", item[0]))


def _resolve_bf16_layer_islands(
    weight_names: list[str],
    config: dict[str, Any],
    *,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
) -> list[str]:
    if num_first_layers_bf16 < 0 or num_last_layers_bf16 < 0:
        raise ValueError("BF16 layer island counts must be non-negative")
    if num_first_layers_bf16 == 0 and num_last_layers_bf16 == 0:
        return []

    inferred = _infer_layer_prefix(weight_names)
    if inferred is None:
        raise ValueError("Cannot resolve first/last BF16 islands: no '<prefix>.layers.<idx>.' weights found")
    layer_prefix, observed_layers = inferred
    total_layers = _hidden_layer_count(config)
    if total_layers is None:
        total_layers = max(observed_layers) + 1

    if num_first_layers_bf16 > total_layers or num_last_layers_bf16 > total_layers:
        raise ValueError(
            "BF16 layer island count exceeds model depth: "
            f"first={num_first_layers_bf16}, last={num_last_layers_bf16}, total={total_layers}"
        )

    selected = list(range(num_first_layers_bf16))
    if num_last_layers_bf16:
        selected.extend(range(total_layers - num_last_layers_bf16, total_layers))
    out: list[str] = []
    seen: set[int] = set()
    for idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(f"{layer_prefix}.{idx}")
    return out


def _merge_modules_to_not_convert(
    explicit_modules: list[str] | None,
    generated_modules: list[str],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw_entry in [*(explicit_modules or []), *generated_modules]:
        entry = raw_entry.strip()
        if not entry:
            continue
        if entry.endswith(".weight"):
            entry = entry[: -len(".weight")]
        if entry not in seen:
            seen.add(entry)
            merged.append(entry)
    return merged


def _build_quantization_config(
    *,
    fmt: str,
    weight_block_size: list[int],
    modules_to_not_convert: list[str],
) -> dict[str, Any]:
    try:
        normalized = normalize_sync_quantization_config(
            {
                "quant_method": "fp8",
                "fmt": fmt,
                "activation_scheme": "dynamic",
                "weight_block_size": weight_block_size,
                "modules_to_not_convert": modules_to_not_convert,
            },
            context="export_quantized.quantization",
        )
    except UnsupportedSyncQuantizationError as exc:
        raise ValueError(str(exc)) from exc
    assert normalized is not None
    return normalized


def _matches_module_skip(name: str, modules_to_not_convert: list[str]) -> bool:
    normalized_prefixes = [
        prefix[: -len(".weight")] if prefix.endswith(".weight") else prefix for prefix in modules_to_not_convert
    ]
    return any(
        name == prefix + ".weight"
        or name.startswith(prefix + ".")
        or fnmatch(name, prefix)
        or fnmatch(name, f"{prefix}.weight")
        for prefix in normalized_prefixes
    )


def _should_quantize_weight(name: str, tensor: torch.Tensor, modules_to_not_convert: list[str]) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"{name} is already FP8; native block-FP8 export expects high-precision source weights")
    if not tensor.is_floating_point():
        return False
    return not _matches_module_skip(name, modules_to_not_convert)


def _convert_xorl_tensor_for_hf_export(
    name: str,
    tensor: torch.Tensor,
    *,
    config: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    """Map XoRL training-only tensor layouts to HF/SGLang-visible names."""

    if ".qkv_proj." in name:
        q_size, kv_size = _attention_qkv_split_sizes(config)
        expected_rows = q_size + 2 * kv_size
        if tensor.ndim == 0 or tensor.shape[0] != expected_rows:
            raise ValueError(
                f"{name} has leading dimension {tuple(tensor.shape)} but HF export expected "
                f"{expected_rows} rows from q={q_size}, k={kv_size}, v={kv_size}"
            )
        prefix, suffix = name.rsplit(".qkv_proj.", 1)
        return [
            (f"{prefix}.q_proj.{suffix}", tensor[:q_size].contiguous()),
            (f"{prefix}.k_proj.{suffix}", tensor[q_size : q_size + kv_size].contiguous()),
            (f"{prefix}.v_proj.{suffix}", tensor[q_size + kv_size :].contiguous()),
        ]

    if name.endswith(".mlp.experts.gate_up_proj"):
        if tensor.ndim != 3 or tensor.shape[2] % 2:
            raise ValueError(
                f"{name} must have shape [num_experts, hidden, 2 * intermediate] for HF export; "
                f"got {tuple(tensor.shape)}"
            )
        prefix = name.rsplit(".gate_up_proj", 1)[0]
        half = tensor.shape[2] // 2
        gate = tensor[:, :, :half].transpose(1, 2).contiguous()
        up = tensor[:, :, half:].transpose(1, 2).contiguous()
        out: list[tuple[str, torch.Tensor]] = []
        for expert_idx in range(tensor.shape[0]):
            out.append((f"{prefix}.{expert_idx}.gate_proj.weight", gate[expert_idx]))
            out.append((f"{prefix}.{expert_idx}.up_proj.weight", up[expert_idx]))
        return out

    if name.endswith(".mlp.experts.down_proj"):
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [num_experts, intermediate, hidden] for HF export")
        prefix = name.rsplit(".down_proj", 1)[0]
        down = tensor.transpose(1, 2).contiguous()
        return [(f"{prefix}.{expert_idx}.down_proj.weight", down[expert_idx]) for expert_idx in range(tensor.shape[0])]

    if ".gate_up_proj." in name:
        if tensor.ndim == 0 or tensor.shape[0] % 2:
            raise ValueError(f"{name} must have an even leading dimension for HF export")
        prefix, suffix = name.rsplit(".gate_up_proj.", 1)
        half = tensor.shape[0] // 2
        return [
            (f"{prefix}.gate_proj.{suffix}", tensor[:half].contiguous()),
            (f"{prefix}.up_proj.{suffix}", tensor[half:].contiguous()),
        ]

    return [(name, tensor)]


def _pending_mla_a_projection_key(name: str, *, config: dict[str, Any]) -> tuple[str, str, str] | None:
    if not _config_has_q_lora_rank(config):
        return None
    if ".self_attn.q_a_proj." in name:
        prefix, suffix = name.rsplit(".q_a_proj.", 1)
        return prefix, suffix, "q_a_proj"
    if ".self_attn.kv_a_proj_with_mqa." in name:
        prefix, suffix = name.rsplit(".kv_a_proj_with_mqa.", 1)
        return prefix, suffix, "kv_a_proj_with_mqa"
    return None


def _should_defer_linear_attention_tensor(name: str, *, config: dict[str, Any]) -> bool:
    return _config_has_linear_attention_layers(config) and ".linear_attn." in name


def quantize_weight_to_fp8(
    tensor: torch.Tensor,
    *,
    fmt: str = "e4m3",
    weight_block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one 2D tensor with XoRL/SGLang zero-padding block semantics."""

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(tensor.shape)}")
    if fmt != "e4m3":
        raise ValueError("Native block-FP8 export currently supports only fmt='e4m3'")
    block_rows, block_cols = weight_block_size
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("weight block size values must be positive")

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    work = tensor.detach().cpu().float()
    rows, cols = work.shape
    pad_rows = (block_rows - rows % block_rows) % block_rows
    pad_cols = (block_cols - cols % block_cols) % block_cols
    if pad_rows or pad_cols:
        padded = torch.zeros(rows + pad_rows, cols + pad_cols, dtype=torch.float32)
        padded[:rows, :cols] = work
    else:
        padded = work

    block_row_count = padded.shape[0] // block_rows
    block_col_count = padded.shape[1] // block_cols
    blocks = padded.reshape(block_row_count, block_rows, block_col_count, block_cols).permute(0, 2, 1, 3)
    block_max = blocks.abs().reshape(block_row_count, block_col_count, -1).max(dim=-1).values
    scale = block_max.clamp(min=1e-12) / fp8_max
    quantized_blocks = (blocks / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    quantized = quantized_blocks.permute(0, 2, 1, 3).reshape(padded.shape[0], padded.shape[1])
    if pad_rows or pad_cols:
        quantized = quantized[:rows, :cols].contiguous()
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


class _ShardWriter:
    def __init__(self, output_dir: Path, max_shard_size: int):
        self.output_dir = output_dir
        self.max_shard_size = max_shard_size
        self.current: dict[str, torch.Tensor] = {}
        self.current_size = 0
        self.shard_paths: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.seen_names: set[str] = set()
        self.total_size = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.seen_names:
            raise ValueError(f"Duplicate exported tensor name {name!r}; source tensors map to the same HF name")
        self.seen_names.add(name)
        tensor_size = _tensor_nbytes(tensor)
        if self.current and self.current_size + tensor_size > self.max_shard_size:
            self.flush()
        self.current[name] = tensor.contiguous()
        self.current_size += tensor_size
        self.total_size += tensor_size

    def flush(self) -> None:
        if not self.current:
            return
        shard_path = self.output_dir / f".tmp-model-{len(self.shard_paths) + 1:05d}.safetensors"
        save_file(self.current, str(shard_path), metadata={"format": "pt"})
        for name in self.current:
            self.weight_map[name] = shard_path.name
        self.shard_paths.append(shard_path)
        self.current = {}
        self.current_size = 0

    def finalize(self) -> int:
        self.flush()
        shard_count = len(self.shard_paths)
        if shard_count == 0:
            raise RuntimeError("No tensors were written")
        if shard_count == 1:
            final_name = "model.safetensors"
            final_path = self.output_dir / final_name
            self.shard_paths[0].replace(final_path)
            self.weight_map = dict.fromkeys(self.weight_map, final_name)
            return 1

        new_weight_map: dict[str, str] = {}
        for idx, shard_path in enumerate(self.shard_paths, start=1):
            final_name = f"model-{idx:05d}-of-{shard_count:05d}.safetensors"
            shard_path.replace(self.output_dir / final_name)
            for tensor_name, old_shard_name in self.weight_map.items():
                if old_shard_name == shard_path.name:
                    new_weight_map[tensor_name] = final_name
        self.weight_map = new_weight_map
        _write_json(
            self.output_dir / _WEIGHT_INDEX_NAME,
            {"metadata": {"total_size": self.total_size}, "weight_map": self.weight_map},
        )
        return shard_count


def export_hf_directory_to_fp8(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fmt: str = "e4m3",
    weight_block_size: tuple[int, int] = (128, 128),
    modules_to_not_convert: list[str] | None = None,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
    max_shard_size: int = _DEFAULT_MAX_SHARD_SIZE,
    overwrite: bool = False,
    fold_qarl: bool = False,
    qarl_quant_cfg: str | dict[str, Any] | None = None,
) -> QuantizedExportResult:
    """Export a HF/safetensors directory to XoRL-native block-FP8 HF format."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_dir and output_dir must be different directories")
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists and is not empty; pass overwrite=True")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config = _read_config(input_path)
    entries = _tensor_entries(input_path)
    weight_names = [name for name, _ in entries]
    unsupported_mtp_reason = _unsupported_mtp_export_reason(config, weight_names, modules_to_not_convert)
    if unsupported_mtp_reason is not None:
        raise ValueError(unsupported_mtp_reason)
    if any(name.endswith(".weight_scale_inv") for name in weight_names):
        raise ValueError("Input already contains weight_scale_inv tensors; re-export from high-precision weights")
    qarl_modules, qarl_state_names = _detect_qarl_modules(weight_names)
    if qarl_state_names and not fold_qarl:
        raise ValueError("Input contains QARL quantizer state; pass fold_qarl=True or --fold-qarl to export it")
    if fold_qarl and not qarl_state_names:
        raise ValueError("fold_qarl=True requires QARL quantizer state tensors in the source checkpoint")

    qarl_cfg: dict[str, Any] | None = None
    qarl_skip_modules: list[str] = []
    if fold_qarl:
        qarl_cfg = _resolve_qarl_quant_cfg(config, qarl_quant_cfg)
        cfg_block_size = tuple(int(value) for value in qarl_cfg.get("weight_block_size", weight_block_size))
        if cfg_block_size != tuple(weight_block_size):
            raise ValueError(
                "QARL folded export weight_block_size must match qarl_quant_cfg.weight_block_size: "
                f"export={tuple(weight_block_size)}, qarl={cfg_block_size}"
            )
        missing_weights = sorted(module for module in qarl_modules if f"{module}.weight" not in weight_names)
        if missing_weights:
            raise ValueError(f"QARL state is missing matching weight tensors for modules: {missing_weights}")
        qarl_skip_modules = _qarl_modules_to_not_convert(weight_names, qarl_modules)

    generated_islands = _resolve_bf16_layer_islands(
        weight_names,
        config,
        num_first_layers_bf16=num_first_layers_bf16,
        num_last_layers_bf16=num_last_layers_bf16,
    )
    merged_modules = _merge_modules_to_not_convert(modules_to_not_convert, [*generated_islands, *qarl_skip_modules])
    quantization_config = _build_quantization_config(
        fmt=fmt,
        weight_block_size=list(weight_block_size),
        modules_to_not_convert=merged_modules,
    )
    block_size = tuple(quantization_config["weight_block_size"])
    skip_modules = quantization_config.get("modules_to_not_convert", [])

    _copy_hf_assets(input_path, output_path)

    writer = _ShardWriter(output_path, max_shard_size=max_shard_size)
    tensors_read = 0
    tensors_written = 0
    quantized_weights = 0
    passthrough_tensors = 0
    entries_by_shard: dict[Path, list[str]] = {}
    for name, shard_path in entries:
        entries_by_shard.setdefault(shard_path, []).append(name)

    pending_mla_a: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    pending_linear_attention: list[tuple[str, torch.Tensor]] = []

    def write_export_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal tensors_written, quantized_weights, passthrough_tensors
        for export_name, export_tensor in _convert_xorl_tensor_for_hf_export(name, tensor, config=config):
            if _should_quantize_weight(export_name, export_tensor, skip_modules):
                quantized, scale = quantize_weight_to_fp8(
                    export_tensor,
                    fmt=quantization_config["fmt"],
                    weight_block_size=(int(block_size[0]), int(block_size[1])),
                )
                writer.add(export_name, quantized)
                writer.add(export_name.replace(".weight", ".weight_scale_inv"), scale)
                tensors_written += 2
                quantized_weights += 1
            else:
                writer.add(export_name, export_tensor.detach().cpu())
                tensors_written += 1
                passthrough_tensors += 1

    for shard_path, shard_names in entries_by_shard.items():
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in shard_names:
                tensor = handle.get_tensor(name)
                tensors_read += 1
                if _is_qarl_state_tensor(name):
                    continue
                mla_key = _pending_mla_a_projection_key(name, config=config)
                if mla_key is not None:
                    prefix, suffix, part = mla_key
                    pending_mla_a.setdefault((prefix, suffix), {})[part] = tensor
                    continue
                if _should_defer_linear_attention_tensor(name, config=config):
                    pending_linear_attention.append((name, tensor))
                    continue
                write_export_tensor(name, tensor)

    for (prefix, suffix), parts in sorted(pending_mla_a.items()):
        q_a = parts.get("q_a_proj")
        kv_a = parts.get("kv_a_proj_with_mqa")
        if suffix == "weight" and q_a is not None and kv_a is not None:
            write_export_tensor(
                f"{prefix}.fused_qkv_a_proj_with_mqa.{suffix}",
                torch.cat([q_a, kv_a], dim=0).contiguous(),
            )
        else:
            if q_a is not None:
                write_export_tensor(f"{prefix}.q_a_proj.{suffix}", q_a)
            if kv_a is not None:
                write_export_tensor(f"{prefix}.kv_a_proj_with_mqa.{suffix}", kv_a)

    if pending_linear_attention:
        from xorl.models.transformers.qwen3_5_shared import remap_linear_attention_params_for_inference  # noqa: PLC0415

        try:
            linear_attention_tensors = remap_linear_attention_params_for_inference(pending_linear_attention)
        except KeyError as exc:
            raise ValueError(
                "Cannot remap Qwen3.5 linear_attention export: missing q/k/v or conv split tensor"
            ) from exc
        for name, tensor in linear_attention_tensors:
            write_export_tensor(name, tensor)

    shard_count = writer.finalize()
    config["quantization_config"] = quantization_config
    if fold_qarl:
        config["xorl_qarl_export"] = {
            "format": "fp8_e4m3",
            "source": "qarl_folded",
            "quant_cfg": qarl_cfg,
            "folded_modules": sorted(qarl_modules),
            "folded_state_tensors": len(qarl_state_names),
            "target_runtime": "sglang_block_fp8",
        }
    _write_json(output_path / "config.json", config)

    return QuantizedExportResult(
        output_dir=str(output_path),
        tensors_read=tensors_read,
        tensors_written=tensors_written,
        quantized_weights=quantized_weights,
        passthrough_tensors=passthrough_tensors,
        shard_count=shard_count,
        total_size=writer.total_size,
        quantization_config=quantization_config,
        qarl_folded=fold_qarl,
        qarl_modules=sorted(qarl_modules),
        qarl_state_tensors=len(qarl_state_names),
    )


def export_qarl_directory_to_fp8(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fmt: str = "e4m3",
    weight_block_size: tuple[int, int] = (128, 128),
    modules_to_not_convert: list[str] | None = None,
    num_first_layers_bf16: int = 0,
    num_last_layers_bf16: int = 0,
    max_shard_size: int = _DEFAULT_MAX_SHARD_SIZE,
    overwrite: bool = False,
    qarl_quant_cfg: str | dict[str, Any] | None = None,
) -> QuantizedExportResult:
    """Fold a dense QARL safetensors directory into a block-FP8 HF artifact."""

    return export_hf_directory_to_fp8(
        input_dir,
        output_dir,
        fmt=fmt,
        weight_block_size=weight_block_size,
        modules_to_not_convert=modules_to_not_convert,
        num_first_layers_bf16=num_first_layers_bf16,
        num_last_layers_bf16=num_last_layers_bf16,
        max_shard_size=max_shard_size,
        overwrite=overwrite,
        fold_qarl=True,
        qarl_quant_cfg=qarl_quant_cfg,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML/JSON export config file")
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = _read_export_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML/JSON export config file")
    parser.add_argument(
        "--input-dir",
        default=config_defaults.get("input_dir"),
        help="HF/safetensors source directory with high-precision weights",
    )
    parser.add_argument(
        "--output-dir",
        default=config_defaults.get("output_dir"),
        help="Output HF directory for block-FP8 weights",
    )
    parser.add_argument("--fmt", default=config_defaults.get("fmt", "e4m3"), choices=["e4m3"], help="FP8 format to write")
    parser.add_argument(
        "--weight-block-size",
        type=int,
        nargs=2,
        default=config_defaults.get("weight_block_size", [128, 128]),
        metavar=("ROWS", "COLS"),
    )
    config_skip_modules = config_defaults.get("modules_to_not_convert", [])
    if isinstance(config_skip_modules, str):
        config_skip_modules = [config_skip_modules]
    if not isinstance(config_skip_modules, list):
        raise ValueError("export config modules_to_not_convert must be a string or list")
    parser.add_argument(
        "--module-to-not-convert",
        "--skip-module",
        dest="modules_to_not_convert",
        action="append",
        default=list(config_skip_modules),
        help="Module prefix or glob to keep in the source dtype; may be repeated",
    )
    parser.add_argument(
        "--num-first-layers-bf16",
        type=int,
        default=config_defaults.get("num_first_layers_bf16", 0),
        help="Number of initial decoder layers to skip",
    )
    parser.add_argument(
        "--num-last-layers-bf16",
        type=int,
        default=config_defaults.get("num_last_layers_bf16", 0),
        help="Number of final decoder layers to skip",
    )
    parser.add_argument(
        "--fold-qarl",
        action="store_true",
        default=bool(config_defaults.get("fold_qarl", False)),
        help="Treat QARL quantizer state in the source checkpoint as foldable metadata and export block-FP8 weights",
    )
    parser.add_argument(
        "--qarl-quant-cfg",
        default=config_defaults.get("qarl_quant_cfg"),
        help="QARL quantization alias or JSON object. Defaults to source config metadata, then FP8_DEFAULT_CFG.",
    )
    parser.add_argument(
        "--max-shard-size",
        default=config_defaults.get("max_shard_size", "5GB"),
        help="Maximum output shard size, e.g. 5GB, 512MiB, or raw bytes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=bool(config_defaults.get("overwrite", False)),
        help="Replace a non-empty output directory",
    )
    args = parser.parse_args(argv)
    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required unless set in --config")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_hf_directory_to_fp8(
        args.input_dir,
        args.output_dir,
        fmt=args.fmt,
        weight_block_size=(args.weight_block_size[0], args.weight_block_size[1]),
        modules_to_not_convert=list(args.modules_to_not_convert),
        num_first_layers_bf16=args.num_first_layers_bf16,
        num_last_layers_bf16=args.num_last_layers_bf16,
        max_shard_size=_parse_size_bytes(args.max_shard_size),
        overwrite=args.overwrite,
        fold_qarl=args.fold_qarl,
        qarl_quant_cfg=_parse_qarl_quant_cfg(args.qarl_quant_cfg),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
