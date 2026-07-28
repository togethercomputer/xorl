"""Repository-wide guard against internal identifiers reaching the public tree.

``src/xorl/sim/calibration_packs.py`` carries a similar pattern set, but it only
scans inside a calibration pack directory. Nothing checked the rest of the tree,
which is how internal worktree paths, another person's home directory and a
scheduling queue label reached this branch.

The patterns here are narrower than the calibration-pack ones on purpose. A bare
``/shared`` matches ``fwd_moe/shared``, and a bare ``volcano`` matches a citation
of "Volcano Engine", a published external framework. Both produce noise, and a
check that cries wolf gets ignored.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "home directory": re.compile(r"/home/[A-Za-z0-9._-]+"),
    "user or mirror under /shared": re.compile(r"/shared/(?:apanda|qywu|huggingface)\b"),
    "internal repository name": re.compile(r"\bxorl(?:-sglang|-client)?-internal\b"),
    "kubectl invocation": re.compile(r"\bkubectl\b", re.IGNORECASE),
    "cluster service address": re.compile(r"\.svc\.cluster\.local\b"),
    "scheduling queue label": re.compile(r"\bteam:\s*(?:turbo|shaping)\b", re.IGNORECASE),
    "volcano scheduler key": re.compile(r"scheduling\.volcano\.sh"),
}

# These two state the patterns, so they necessarily contain them.
EXEMPT = {"tests/test_no_internal_references.py", "src/xorl/sim/calibration_packs.py"}


def _tracked_files() -> list[str]:
    listing = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return [name for name in listing.stdout.split("\n") if name and name not in EXEMPT]


@pytest.mark.cpu
def test_no_internal_references_in_tracked_files() -> None:
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
    assert not violations, "internal references in tracked files:\n" + "\n".join(sorted(violations))
