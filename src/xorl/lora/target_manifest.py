"""Strict, checkpoint-derived LoRA target coverage manifests.

The legacy ``lora_target_modules`` list matches leaf names and can silently
cover a different surface when a model fuses projections.  A target manifest
adds exact runtime module counts and ranks while retaining the leaf list for
injection.  Manifests are intentionally model-instance checks: they run after
LoRA injection and before parallelization or any training step.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch.nn as nn


TARGET_MANIFEST_SCHEMA_VERSION = 1


def load_lora_target_manifest(value: Optional[Mapping[str, Any] | str]) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        payload = json.loads(Path(value).read_text())
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"lora_target_manifest must be a mapping, path, or None; got {type(value).__name__}")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != TARGET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported LoRA target manifest schema_version "
            f"{schema_version!r}; expected {TARGET_MANIFEST_SCHEMA_VERSION}"
        )
    if "allow_unlisted" in payload and type(payload["allow_unlisted"]) is not bool:
        raise ValueError("LoRA target manifest allow_unlisted must be a Boolean")
    target_modules = payload.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or not all(isinstance(v, str) for v in target_modules)
    ):
        raise ValueError("LoRA target manifest target_modules must be a non-empty list of strings")
    expected_modules = payload.get("expected_modules")
    if not isinstance(expected_modules, list) or not expected_modules:
        raise ValueError("LoRA target manifest expected_modules must be a non-empty list")
    for entry in expected_modules:
        if not isinstance(entry, dict) or not isinstance(entry.get("pattern"), str):
            raise ValueError(f"Invalid expected_modules entry: {entry!r}")
        if type(entry.get("count")) is not int or entry["count"] < 0:
            raise ValueError(f"Manifest count must be a non-negative integer: {entry!r}")
        rank = entry.get("rank")
        if rank is not None and (type(rank) is not int or rank <= 0):
            raise ValueError(f"Manifest rank must be a positive integer or null: {entry!r}")
    return payload


def resolve_lora_target_modules(
    target_modules: Optional[Iterable[str]],
    manifest: Optional[Mapping[str, Any] | str],
) -> tuple[Optional[list[str]], Optional[dict[str, Any]]]:
    loaded = load_lora_target_manifest(manifest)
    if loaded is None:
        return None if target_modules is None else list(target_modules), None
    manifest_targets = list(loaded["target_modules"])
    if target_modules is not None and set(target_modules) != set(manifest_targets):
        raise ValueError(
            "lora_target_modules do not match the strict LoRA target manifest: "
            f"configured={sorted(target_modules)!r}, manifest={sorted(manifest_targets)!r}"
        )
    return manifest_targets, loaded


def _is_lora_module(module: nn.Module) -> bool:
    if getattr(module, "active_r", None) is None:
        return False
    return any("lora_A" in name or "lora_B" in name for name, _ in module.named_parameters(recurse=False)) or any(
        "_lora_A" in name or "_lora_B" in name for name, _ in module.named_parameters(recurse=False)
    )


def collect_lora_runtime_modules(model: nn.Module) -> dict[str, int]:
    modules: dict[str, int] = {}
    for name, module in model.named_modules():
        if not name or not _is_lora_module(module):
            continue
        rank = getattr(module, "active_r", getattr(module, "r", None))
        if type(rank) is not int or rank <= 0:
            raise ValueError(f"LoRA module {name!r} has invalid active rank {rank!r}")
        modules[name] = rank
    return modules


def validate_lora_target_manifest(model: nn.Module, manifest: Mapping[str, Any] | str) -> dict[str, int]:
    loaded = load_lora_target_manifest(manifest)
    assert loaded is not None
    actual = collect_lora_runtime_modules(model)
    expected_entries = loaded["expected_modules"]
    matched_actual: set[str] = set()

    errors: list[str] = []
    for entry in expected_entries:
        pattern = entry["pattern"]
        names = sorted(name for name in actual if fnmatch.fnmatchcase(name, pattern))
        expected_count = entry["count"]
        if len(names) != expected_count:
            errors.append(f"{pattern!r}: matched {len(names)} modules, expected {expected_count}")
        expected_rank = entry.get("rank")
        if expected_rank is not None:
            wrong = {name: actual[name] for name in names if actual[name] != expected_rank}
            if wrong:
                errors.append(f"{pattern!r}: rank mismatch {wrong!r}, expected rank {expected_rank}")
        overlap = matched_actual.intersection(names)
        if overlap:
            errors.append(f"{pattern!r}: overlaps earlier manifest patterns at {sorted(overlap)!r}")
        matched_actual.update(names)

    if not loaded.get("allow_unlisted", False):
        unlisted = sorted(set(actual) - matched_actual)
        if unlisted:
            errors.append(f"unlisted LoRA modules: {unlisted!r}")
    if errors:
        raise ValueError("LoRA target manifest validation failed: " + "; ".join(errors))
    return actual
