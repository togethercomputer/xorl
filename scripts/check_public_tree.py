#!/usr/bin/env python3
"""Reject private identifiers and environment-specific references in tracked files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "home directory": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+"),
    "personal data workspace": re.compile(r"/data/[A-Za-z0-9._-]+/(?:WorkingProjects|outputs|miniconda3)/"),
    "user or mirror under /shared": re.compile(r"/shared/(?:apanda|qywu|huggingface)\b"),
    "internal repository name": re.compile(r"\bxorl(?:-sglang|-client)?-internal\b"),
    "kubectl invocation": re.compile(r"\bkubectl\b", re.IGNORECASE),
    "cluster service address": re.compile(r"\.svc\.cluster\.local\b"),
    "scheduling queue label": re.compile(r"\bteam:\s*(?:turbo|shaping)\b", re.IGNORECASE),
    "volcano scheduler key": re.compile(r"scheduling\.volcano\.sh"),
    "pointer to an internal-only note": re.compile(r"\bdocs/notes/"),
    "internal account name": re.compile(r"\bapanda\b"),
    "internal branch name": re.compile(r"\bapanda-dev\b"),
    "internal tracker ticket": re.compile(r"\bXORL-\d+\b"),
    "authoring-assistant attribution": re.compile(r"(?i)\b(?:claude|anthropic|copilot)\b"),
}

# Pattern definitions necessarily contain the forbidden strings. The gitignore
# names authoring-tool files only to keep them untracked, which is not a leak.
EXEMPT = {
    "scripts/check_public_tree.py",
    "src/xorl/sim/calibration_packs.py",
    ".gitignore",
}


def _tracked_files() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in listing.stdout.splitlines() if name and name not in EXEMPT]


def main() -> int:
    violations: list[str] = []
    for name in _tracked_files():
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                violations.append(f"{name}: {label}: {match.group(0)!r}")

    if violations:
        print("Internal references in tracked files:")
        print("\n".join(sorted(violations)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
