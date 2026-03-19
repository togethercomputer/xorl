#!/usr/bin/env bash
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../docs" && pwd)"

cd "$DOCS_DIR"

echo "Installing dependencies..."
npm ci

echo "Building docs..."
npm run build

echo "Done. Output is at $DOCS_DIR/dist/"
