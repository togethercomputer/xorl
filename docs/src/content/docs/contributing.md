---
title: Contributing to the Docs
---

The xorl documentation is built with [Astro Starlight](https://starlight.astro.build) and lives in the `docs/` directory of the repository.

## Prerequisites

- Node.js 22+
- npm

## Local Development

```bash
cd docs
npm ci
npm run dev
```

This starts a local dev server at `http://localhost:4321` with hot reload.

## Adding or Editing Pages

All content is in `docs/src/content/docs/`. Each file is a Markdown file with a frontmatter title:

```markdown
---
title: My Page
---

Content here.
```

Pages are organized into subdirectories that map to the sidebar sections (e.g. `getting-started/`, `training/`, `parallelism/`). After adding a new page, register it in the sidebar in `docs/astro.config.mjs`.

## Building

Use the provided script from the repo root:

```bash
./docs/deploy_docs.sh
```

Or manually:

```bash
cd docs
npm ci
npm run build
```

The output is written to `docs/dist/`.

## Deployment

Docs are deployed automatically to GitHub Pages when changes under `docs/` are pushed to the `qingyang/docs` branch. The workflow is defined in `.github/workflows/docs.yml`.

To trigger a deployment manually without pushing a new commit:

```bash
gh workflow run docs.yml --repo togethercomputer/xorl-internal --ref qingyang/docs
```

To watch the deployment progress:

```bash
gh run list --repo togethercomputer/xorl-internal --workflow=docs.yml --limit=1
gh run watch <run-id> --repo togethercomputer/xorl-internal
```
