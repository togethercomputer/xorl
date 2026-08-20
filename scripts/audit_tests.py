#!/usr/bin/env python3
"""Surface test-audit candidates without deciding whether to remove them."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DECISIONS_FILE = "test_audit_decisions.json"
VALID_DECISIONS = {"keep", "consolidate", "rewrite", "relocate", "remove"}
VALID_STATUSES = {"proposed", "accepted", "applied", "rejected"}


@dataclass(frozen=True)
class TestCase:
    path: str
    name: str
    line: int
    end_line: int
    body_hash: str
    signals: tuple[str, ...]


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    value: ast.expr | None = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _body_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body.pop(0)
    normalized = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _contains_call(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(child, ast.Call) and (_call_name(child) in names or _call_name(child).split(".")[-1] in names)
        for child in ast.walk(node)
    )


def _has_outcome(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if any(isinstance(child, (ast.Assert, ast.Raise)) for child in ast.walk(node)):
        return True
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _call_name(child)
        leaf = name.split(".")[-1]
        normalized_leaf = leaf.lstrip("_")
        if name in {"pytest.raises", "pytest.warns", "pytest.deprecated_call"}:
            return True
        if leaf == "simplefilter" and child.args:
            first_arg = child.args[0]
            if isinstance(first_arg, ast.Constant) and first_arg.value == "error":
                return True
        if normalized_leaf.startswith("assert") or leaf in {"fail", "raises", "warns", "start_processes"}:
            return True
    return False


def _skip_inside_condition(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.Match)) and _contains_call(child, {"pytest.skip"}):
            return True
    return False


def _reads_module_source(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Detect tests that read an imported module's source instead of behavior."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr not in {"read_text", "read_bytes"}:
            continue
        receiver = child.func.value
        if not isinstance(receiver, ast.Call) or not receiver.args:
            continue
        if _call_name(receiver).split(".")[-1] not in {"Path", "open"}:
            continue
        if any(
            isinstance(part, ast.Attribute) and part.attr == "__file__"
            for argument in receiver.args
            for part in ast.walk(argument)
        ):
            return True
    return False


def _signals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    signals: list[str] = []
    has_outcome = _has_outcome(node)
    has_print = _contains_call(node, {"print", "pprint"})

    if not has_outcome:
        signals.append("no-observable-outcome")
    if has_print and not has_outcome:
        signals.append("print-only")
    if _skip_inside_condition(node):
        signals.append("conditional-runtime-skip")
    if _contains_call(node, {"inspect.getsource", "getsource"}) or _reads_module_source(node):
        signals.append("source-inspection")

    return tuple(signals)


class Collector(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.classes: list[str] = []
        self.tests: list[TestCase] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def _visit_test(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test_"):
            name = ".".join([*self.classes, node.name])
            self.tests.append(
                TestCase(
                    path=self.path,
                    name=name,
                    line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    body_hash=_body_hash(node),
                    signals=_signals(node),
                )
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_test(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_test(node)


def _repository_test_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "tests",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return sorted((repo_root / "tests").rglob("*.py"))
    paths = [repo_root / item for item in result.stdout.splitlines() if item.endswith(".py")]
    return [path for path in paths if path.is_file()]


def _collect(repo_root: Path) -> tuple[list[TestCase], list[dict[str, str]]]:
    tests: list[TestCase] = []
    parse_errors: list[dict[str, str]] = []
    for path in _repository_test_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError) as exc:
            parse_errors.append({"path": relative, "error": str(exc)})
            continue
        collector = Collector(relative)
        collector.visit(tree)
        tests.extend(collector.tests)
    return tests, parse_errors


def _duplicate_groups(tests: Iterable[TestCase]) -> list[list[dict[str, object]]]:
    by_hash: dict[str, list[TestCase]] = defaultdict(list)
    for test in tests:
        by_hash[test.body_hash].append(test)
    groups = []
    for matches in by_hash.values():
        locations = {(match.path, match.name) for match in matches}
        if len(locations) > 1:
            groups.append([asdict(match) for match in sorted(matches, key=lambda item: (item.path, item.line))])
    return sorted(groups, key=lambda group: (group[0]["path"], group[0]["line"]))


def _load_decisions(repo_root: Path) -> list[dict[str, object]]:
    path = repo_root / DECISIONS_FILE
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("items"), list):
        raise ValueError(f"{DECISIONS_FILE} must contain schema_version=1 and an items list")

    seen: set[str] = set()
    for item in payload["items"]:
        missing = {"id", "scope", "decision", "status", "evidence"} - set(item)
        if missing:
            raise ValueError(f"decision is missing {sorted(missing)}: {item}")
        if item["id"] in seen:
            raise ValueError(f"duplicate decision id: {item['id']}")
        if item["decision"] not in VALID_DECISIONS:
            raise ValueError(f"invalid decision {item['decision']!r} for {item['id']}")
        if item["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid status {item['status']!r} for {item['id']}")
        if not isinstance(item["evidence"], list) or not item["evidence"]:
            raise ValueError(f"decision {item['id']} needs at least one evidence item")
        seen.add(item["id"])
    return payload["items"]


def _report(repo_root: Path) -> dict[str, object]:
    tests, parse_errors = _collect(repo_root)
    signal_counts = Counter(signal for test in tests for signal in test.signals)
    candidates = [asdict(test) for test in tests if test.signals]
    return {
        "summary": {
            "repository_python_test_files": len(_repository_test_files(repo_root)),
            "test_definitions": len(tests),
            "candidate_definitions": len(candidates),
            "signal_counts": dict(sorted(signal_counts.items())),
            "exact_duplicate_body_groups": len(_duplicate_groups(tests)),
            "parse_errors": len(parse_errors),
        },
        "candidates": candidates,
        "exact_duplicate_body_groups": _duplicate_groups(tests),
        "parse_errors": parse_errors,
        "curated_decisions": _load_decisions(repo_root),
    }


def _print_text(report: dict[str, object]) -> None:
    summary = report["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nCandidates (signals are prompts for review, not deletion decisions):")
    for item in report["candidates"]:
        signals = ", ".join(item["signals"])
        print(f"  {item['path']}:{item['line']} {item['name']} [{signals}]")
    print("\nExact duplicate test bodies:")
    for group in report["exact_duplicate_body_groups"]:
        print("  group")
        for item in group:
            print(f"    {item['path']}:{item['line']} {item['name']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    report = _report(args.repo_root.resolve())
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 1 if report["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
