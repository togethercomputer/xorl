"""Discovery and validation for portable simulator calibration packs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PACK_SCHEMA_VERSION = 1
_PACK_ROOT = Path(__file__).with_name("calibration_packs")
_FORBIDDEN_CONTENT = {
    "absolute home path": re.compile(r"/home/"),
    "workspace mount": re.compile(r"/workspace(?:/|\b)"),
    "shared mount": re.compile(r"/shared(?:/|\b)"),
    "internal repository name": re.compile(r"\bxorl(?:-[a-z]+)?-internal\b"),
    "Kubernetes command": re.compile(r"\bkubectl\b", re.IGNORECASE),
    "Kubernetes service address": re.compile(r"\.svc\.cluster\.local\b"),
    "PVC setting": re.compile(r"\b(?:home|shared)[_-]?pvc\b", re.IGNORECASE),
    "Volcano setting": re.compile(r"\bvolcano\b", re.IGNORECASE),
    "team scheduling label": re.compile(r"\bteam:\s*(?:turbo|shaping)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class CalibrationPack:
    name: str
    path: Path
    manifest: dict[str, Any]

    @property
    def default_config(self) -> Path:
        return self.resolve_declared_path(self.manifest.get("default_config"), field="default_config")

    def resolve_declared_path(self, value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"calibration-pack {field} must be a non-empty relative path")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"calibration-pack {field} must stay within {self.path}: {value!r}")
        root = self.path.resolve()
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"calibration-pack {field} escapes {root}: {value!r}") from exc
        return resolved


def calibration_pack_root() -> Path:
    """Return the installed directory containing built-in calibration packs."""

    return _PACK_ROOT


def list_calibration_packs() -> list[str]:
    if not _PACK_ROOT.is_dir():
        return []
    return sorted(path.name for path in _PACK_ROOT.iterdir() if (path / "manifest.json").is_file())


def resolve_calibration_pack(value: str | Path) -> Path:
    """Resolve a filesystem path or ``builtin:<name>`` calibration-pack reference."""

    raw = str(value)
    name = raw.removeprefix("builtin:")
    available_names = list_calibration_packs()
    if raw.startswith("builtin:") or (not Path(raw).exists() and name in available_names):
        if name not in available_names:
            available = ", ".join(available_names) or "none"
            raise ValueError(f"unknown built-in calibration pack {name!r}; available: {available}")
        return _PACK_ROOT / name
    return Path(raw)


def load_calibration_pack(value: str | Path) -> CalibrationPack:
    path = resolve_calibration_pack(value).resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing calibration-pack manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"calibration-pack manifest must be a JSON object: {manifest_path}")
    pack = CalibrationPack(name=str(manifest.get("name", path.name)), path=path, manifest=manifest)
    pack.default_config
    for field in ("configs", "results"):
        values = manifest.get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"calibration-pack {field} must be a list")
        for declared in values:
            pack.resolve_declared_path(declared, field=field)
    return pack


def resolve_pack_inputs(
    pack_name: str | None,
    config: Path | None,
    benchmark_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    """Fill omitted config and benchmark paths from a built-in pack."""

    if pack_name is None:
        return config, benchmark_dir
    pack = load_calibration_pack(pack_name)
    return config or pack.default_config, benchmark_dir or pack.path


def _check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def validate_calibration_pack(value: str | Path) -> dict[str, Any]:
    pack = load_calibration_pack(value)
    manifest = pack.manifest
    checks = [
        _check(
            "schema_version",
            manifest.get("schema_version") == PACK_SCHEMA_VERSION,
            f"expected {PACK_SCHEMA_VERSION}, found {manifest.get('schema_version')!r}",
        ),
        _check("manifest_name", manifest.get("name") == pack.path.name, f"manifest name={manifest.get('name')!r}"),
        _check("model", isinstance(manifest.get("model"), str), f"model={manifest.get('model')!r}"),
    ]

    declared = [manifest.get("default_config"), *manifest.get("configs", []), *manifest.get("results", [])]
    relative_paths = [Path(str(item)) for item in declared if item]
    for relative in relative_paths:
        is_safe = not relative.is_absolute() and ".." not in relative.parts
        checks.append(_check(f"portable_path:{relative}", is_safe, str(relative)))
        checks.append(_check(f"file_exists:{relative}", is_safe and (pack.path / relative).is_file(), str(relative)))

    config_paths = sorted({Path(str(item)) for item in manifest.get("configs", [])})
    for relative in config_paths:
        path = pack.path / relative
        if not path.is_file():
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_path = config.get("model", {}).get("model_path") if isinstance(config, dict) else None
        checks.append(
            _check(
                f"config_model:{relative}",
                model_path == manifest.get("model"),
                f"expected {manifest.get('model')!r}, found {model_path!r}",
            )
        )

    for path in sorted(item for item in pack.path.rglob("*") if item.is_file()):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(pack.path)
        for label, pattern in _FORBIDDEN_CONTENT.items():
            match = pattern.search(text)
            checks.append(
                _check(
                    f"sanitized:{relative}:{label}",
                    match is None,
                    "not present" if match is None else f"matched {match.group(0)!r}",
                )
            )

    return {
        "name": pack.name,
        "path": str(pack.path),
        "schema_version": manifest.get("schema_version"),
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List installed calibration packs")
    path_parser = subparsers.add_parser("path", help="Print the path to an installed calibration pack")
    path_parser.add_argument("pack")
    validate_parser = subparsers.add_parser("validate", help="Validate one or all installed calibration packs")
    validate_parser.add_argument("pack", nargs="?", default=None)
    args = parser.parse_args()

    if args.command == "list":
        print("\n".join(list_calibration_packs()))
        return
    if args.command == "path":
        print(resolve_calibration_pack(args.pack))
        return

    names = [args.pack] if args.pack else list_calibration_packs()
    reports = [validate_calibration_pack(name) for name in names]
    print(json.dumps(reports, indent=2, sort_keys=True))
    if any(report["status"] != "pass" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
