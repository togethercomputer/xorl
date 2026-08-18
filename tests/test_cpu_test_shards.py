"""Guard: the CPU workflow's shards must cover every test exactly once.

``.github/workflows/pr-test-cpu.yml`` splits the CPU job across a matrix of
functional shards -- expert parallel, tensor parallel, FSDP, context/pipeline
parallel, the training server, and a catch-all. Most shards select whole
directories, which is self-maintaining, but the ``tests/distributed`` shards
select individual files so that each parallelism dimension gets its own shard.

File lists rot. A test file added to ``tests/distributed`` that matches no shard
pattern would run in no job at all, and nothing about a green CI would say so.
That is not hypothetical: the tensor-parallel unfuse test spent its whole life
marked ``cpu`` at module scope while needing CUDA, so the CPU job could only skip
it and the GPU job deselected it. It ran nowhere until the shards were added.

These checks are filesystem-and-YAML only -- no collection -- so they stay cheap
enough to run in the catch-all shard alongside everything else.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml


pytestmark = [pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-test-cpu.yml"

# Shards select individual files here, so this directory needs the partition
# check. Everything else is selected by directory and cannot develop a hole.
FILE_SHARDED_DIR = "tests/distributed"


def _matrix_entries() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    include = workflow["jobs"]["shard"]["strategy"]["matrix"]["include"]
    assert include, "the CPU workflow declares no shards"
    return include


def _split(paths: str) -> tuple[list[str], list[str]]:
    """Return (selection patterns, ignored paths) from a shard's ``paths``."""
    selectors: list[str] = []
    ignored: list[str] = []
    for token in paths.split():
        if token.startswith("--ignore="):
            ignored.append(token.split("=", 1)[1])
        else:
            assert not token.startswith("-"), f"unhandled pytest flag in a shard: {token}"
            selectors.append(token)
    return selectors, ignored


def _claims() -> dict[str, str]:
    """Map each file-sharded test file to the single shard that claims it."""
    claimed: dict[str, str] = {}
    for entry in _matrix_entries():
        shard = entry["shard"]
        selectors, _ = _split(entry["paths"])
        for pattern in selectors:
            if not pattern.startswith(FILE_SHARDED_DIR):
                continue
            for hit in glob.glob(str(REPO_ROOT / pattern)):
                rel = str(Path(hit).relative_to(REPO_ROOT))
                previous = claimed.get(rel)
                assert previous is None, f"{rel} is claimed by both '{previous}' and '{shard}'"
                claimed[rel] = shard
    return claimed


def test_every_file_sharded_test_is_claimed_by_a_shard():
    """A file matching no shard pattern would run in no job at all."""
    on_disk = {str(Path(p).relative_to(REPO_ROOT)) for p in glob.glob(str(REPO_ROOT / FILE_SHARDED_DIR / "test_*.py"))}
    unclaimed = sorted(on_disk - set(_claims()))
    assert not unclaimed, (
        f"these {FILE_SHARDED_DIR} files match no shard in {WORKFLOW.name} and would run nowhere: "
        f"{unclaimed}. Add each to the shard for its functional area."
    )


def test_no_file_is_claimed_by_two_shards():
    """Overlapping patterns waste a runner and double-report failures."""
    _claims()  # raises with the offending path if two shards overlap


def test_every_shard_pattern_matches_something():
    """A pattern matching nothing is a rename or typo silently dropping coverage."""
    empty = []
    for entry in _matrix_entries():
        selectors, _ = _split(entry["paths"])
        for pattern in selectors:
            if not glob.glob(str(REPO_ROOT / pattern)):
                empty.append(f"{entry['shard']}: {pattern}")
    assert not empty, f"shard patterns matching no file: {empty}"


def test_catch_all_ignores_exactly_what_other_shards_claim():
    """The catch-all must skip the explicitly-sharded trees and nothing else.

    Ignore too little and those tests run twice; ignore too much and they run
    never. Either way the totals stop adding up, quietly.
    """
    entries = _matrix_entries()
    catch_all = [e for e in entries if _split(e["paths"])[1]]
    assert len(catch_all) == 1, "expected exactly one shard to use --ignore (the catch-all)"
    ignored = set(_split(catch_all[0]["paths"])[1])

    explicit_roots = set()
    for entry in entries:
        if entry["shard"] == catch_all[0]["shard"]:
            continue
        for pattern in _split(entry["paths"])[0]:
            parts = Path(pattern).parts
            explicit_roots.add(str(Path(*parts[:2])) if len(parts) > 1 else pattern)

    assert ignored == explicit_roots, (
        f"the catch-all ignores {sorted(ignored)} but the other shards claim {sorted(explicit_roots)}; "
        "tests under the difference would run twice or not at all"
    )
