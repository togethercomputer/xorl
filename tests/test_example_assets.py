import re
import subprocess
from pathlib import Path

import pytest


pytestmark = [pytest.mark.cpu]


TRACKED_TEXT_SUFFIXES = {".md", ".mdx", ".sh", ".yaml", ".yml"}
FORBIDDEN_PATTERNS = {
    "home_dir": re.compile(r"(?<![A-Za-z0-9._-])/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "data_workspace": re.compile(r"(?<![A-Za-z0-9._-])/data/[A-Za-z0-9._-]+/(?:WorkingProjects|outputs|miniconda3)/"),
}


def _tracked_example_and_experiment_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "examples", "experiments"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git ls-files unavailable: {exc}")

    tracked_files = []
    for relative_path in result.stdout.splitlines():
        path = repo_root / relative_path
        if path.suffix in TRACKED_TEXT_SUFFIXES:
            tracked_files.append(path)
    return tracked_files


def test_examples_and_experiments_avoid_personal_absolute_paths():
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for path in _tracked_example_and_experiment_files(repo_root):
        text = path.read_text(encoding="utf-8")
        for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
            if match := pattern.search(text):
                line_number = text.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(repo_root)}:{line_number}: {pattern_name}: {match.group(0)}")

    assert not violations, "Tracked example and experiment assets contain personal absolute paths:\n" + "\n".join(
        violations
    )
