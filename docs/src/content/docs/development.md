---
title: Development Guide
---

## Workflow

1. **Branch** off `main` with a descriptive name: `feature/my-feature`, `fix/bug-description`
2. **Commit** early and often on your branch — commit messages don't matter much here
3. **Open a PR** against `main` when ready for review
4. **Squash merge** — all PRs are merged as a single squash commit; write a clean PR title and description since that becomes the commit message

## PR Guidelines

- Keep PRs focused — one feature or fix per PR
- Add tests for new behavior; existing tests must pass
- Update relevant docs if behavior changes
- PR title should be imperative and descriptive: `Add chunked cross-entropy loss` not `chunked ce`

## Commit Message (Squash Merge)

The squash commit message = PR title + body. Write it as if it's a commit:

```
Add ring attention support for long-context training

- Implements sequence sharding across TP ranks
- Works with FSDP2 and QLoRA
- Requires --ring-attn-size > 1 in launch config
```

## Code Style

We use [pre-commit](https://pre-commit.com/) to enforce formatting and catch common issues. Set it up once:

```bash
pip install pre-commit
pre-commit install
```

This runs automatically on every `git commit`. To check the entire codebase manually:

```bash
pre-commit run --all-files
```

The hooks include:
- **ruff** — auto-fixes imports and lint issues
- **ruff-format** — code formatting (line length 120)
- **codespell** — catches typos
- **trailing-whitespace / end-of-file-fixer** — file hygiene

CI runs the same hooks, so if pre-commit passes locally, CI will too.

Additional guidelines:
- No dead code, no commented-out blocks
- Type hints on public APIs

## Testing

```bash
# Run all unit tests
pytest tests/

# Run a specific test file
pytest tests/server/api_server/test_checkpoint_paths.py -v
```

Tests must pass locally before requesting review. See the [Testing](/testing/overview) section for full details.

## Branch Protection

- Direct pushes to `main` are not allowed
- All merges go through PR + squash merge
- Delete your branch after merge

## Authorship

Git identity must be set correctly before contributing:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

Since all PRs are squash-merged, the GitHub PR author becomes the commit author. If you're committing directly (e.g. fixes, docs), ensure your local git config reflects your real identity — the committer shown in `git log` is permanent.

When a PR has multiple significant contributors, acknowledge them in the commit body:

```
Co-authored-by: Jane Smith <jane@example.com>
```

## Reviewing

- Approve only when you'd be comfortable maintaining the code
- Leave actionable comments, not style nits (let ruff handle those)
- If a PR is too large to review well, ask the author to split it
