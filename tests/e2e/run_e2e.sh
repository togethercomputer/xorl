#!/usr/bin/env bash
# Run e2e tests with pytest.
#
# E2e tests launch torchrun internally per test, so this script just invokes
# pytest directly. Use the optional arguments to target a specific model suite
# or test file.
#
# Usage:
#   ./tests/e2e/run_e2e.sh                          # all e2e tests
#   ./tests/e2e/run_e2e.sh qwen3_8b                 # one model suite
#   ./tests/e2e/run_e2e.sh qwen3_8b/test_lora.py    # one file

set -euo pipefail

TARGET=${1:-""}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_DIR="$SCRIPT_DIR"

if [ -n "$TARGET" ]; then
    TEST_PATH="$E2E_DIR/$TARGET"
else
    TEST_PATH="$E2E_DIR"
fi

echo "Running e2e tests: $TEST_PATH"
pytest "$TEST_PATH" -v -m "e2e" "$@"
