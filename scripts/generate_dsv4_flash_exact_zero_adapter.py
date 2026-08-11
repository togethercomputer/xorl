#!/usr/bin/env python3
"""Generate a complete zero or deterministic DSV4-Flash exact adapter.

The adapter is intentionally large: all 948 logical factors are present, and
the 43 routed layers retain per-expert A/B banks.  An all-zero adapter is the
first serving-value identity control; a missing or partial adapter is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SGLANG_PYTHON = REPO_ROOT / "submodules" / "xorl-sglang" / "python"
sys.path.insert(0, str(SGLANG_PYTHON))

from sglang.srt.lora.dsv4 import (  # noqa: E402
    DSV4_FLASH_LOGICAL_FACTOR_COUNT,
    DSV4_FLASH_LORA_FORMAT,
    DSV4_FLASH_REQUIRED_TARGET_MODULES,
    Dsv4FlashExactValidator,
    build_dsv4_flash_exact_inventory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fill",
        choices=("zero", "distinguishable"),
        default="zero",
        help="zero is the identity control; distinguishable fills every factor with a deterministic nonzero pattern",
    )
    parser.add_argument("--perturb-export-key")
    parser.add_argument("--perturb-add", type=float, default=0.0)
    return parser.parse_args()


def _factor_tensor(spec, fill: str) -> torch.Tensor:
    if fill == "zero":
        return torch.zeros(spec.export_shape, dtype=spec.export_dtype)
    digest = hashlib.sha256(spec.export_key.encode("utf-8")).digest()
    # Exact binary fractions keep generation reproducible through BF16 export.
    magnitude = (1 + digest[0] % 16) / 1024.0
    tensor = torch.empty(spec.export_shape, dtype=spec.export_dtype)
    flat = tensor.view(-1)
    phase = digest[1] & 1
    flat[phase::2] = magnitude
    flat[1 - phase :: 2] = -magnitude
    return tensor


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        raw_config = json.loads(args.model_config.read_text())
        config = SimpleNamespace(**raw_config)
        adapter_config = {
            "_sglang_lora_format": DSV4_FLASH_LORA_FORMAT,
            "base_model_name_or_path": args.model_id,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": 1,
            "lora_dropout": 0.0,
            "moe_hybrid_shared_lora": False,
            "peft_type": "LORA",
            "r": 1,
            "revision": args.model_revision,
            "target_modules": sorted(DSV4_FLASH_REQUIRED_TARGET_MODULES),
            "task_type": "CAUSAL_LM",
            "use_dora": False,
            "use_rslora": False,
        }
        specs = build_dsv4_flash_exact_inventory(config, adapter_config)
        tensors = {spec.export_key: _factor_tensor(spec, args.fill) for spec in specs}
        if args.perturb_export_key:
            if args.perturb_export_key not in tensors:
                raise ValueError(f"Unknown DSV4 export key for perturbation: {args.perturb_export_key}")
            if args.perturb_add == 0.0:
                raise ValueError("--perturb-export-key requires a nonzero --perturb-add")
            tensors[args.perturb_export_key] = (
                tensors[args.perturb_export_key] + args.perturb_add
            ).to(specs[0].export_dtype)
        weights_path = temporary / "adapter_model.safetensors"
        save_file(
            tensors,
            weights_path,
            metadata={
                "format": "pt",
                "dsv4_exact_contract": DSV4_FLASH_LORA_FORMAT,
                "all_zero": str(args.fill == "zero").lower(),
            },
        )
        del tensors
        _write_json(temporary / "adapter_config.json", adapter_config)

        validator = Dsv4FlashExactValidator(config, adapter_config)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                validator.observe(key, handle.get_tensor(key))
        validator.finalize()
        if args.fill == "zero" and not validator.all_zero:
            raise AssertionError("Generated DSV4 identity adapter is not all-zero")
        if args.fill == "distinguishable" and validator.all_zero:
            raise AssertionError("Generated DSV4 distinguishable adapter is all-zero")

        role_counts = Counter(spec.role for spec in specs)
        total_elements = sum(
            torch.Size(spec.export_shape).numel() for spec in specs
        )
        _write_json(
            temporary / "manifest.json",
            {
                "adapter_format": DSV4_FLASH_LORA_FORMAT,
                "all_zero": args.fill == "zero",
                "fill": args.fill,
                "perturb_add": args.perturb_add if args.perturb_export_key else None,
                "perturb_export_key": args.perturb_export_key,
                "config_sha256": _sha256(args.model_config),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "logical_factor_count": len(specs),
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "role_factor_counts": dict(sorted(role_counts.items())),
                "safetensors_sha256": _sha256(weights_path),
                "tensor_bytes": weights_path.stat().st_size,
                "total_bfloat16_elements": total_elements,
            },
        )
        if len(keys) != DSV4_FLASH_LOGICAL_FACTOR_COUNT:
            raise AssertionError(
                f"Expected {DSV4_FLASH_LOGICAL_FACTOR_COUNT} tensors, got {len(keys)}"
            )
        os.replace(temporary, output_dir)
        print(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
