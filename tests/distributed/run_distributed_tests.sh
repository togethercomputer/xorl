#!/bin/bash
# Helper script to run distributed tests with torchrun
#
# Usage:
#   ./run_distributed_tests.sh [num_processes] [test_file]
#
# Examples:
#   ./run_distributed_tests.sh 2                          # Run all integration tests with 2 processes
#   ./run_distributed_tests.sh 4 test_parallel_state_integration.py  # Run specific test with 4 processes

set -e

# Default values
NUM_PROCS=${1:-2}
TEST_FILE=${2:-"test_parallel_state_integration.py"}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Running Distributed Tests${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "Number of processes: ${GREEN}${NUM_PROCS}${NC}"
echo -e "Test file: ${GREEN}${TEST_FILE}${NC}"
echo -e "${BLUE}=================================${NC}"
echo

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run with torchrun
torchrun \
    --nproc_per_node="${NUM_PROCS}" \
    --master_port=29500 \
    -m pytest "${TEST_FILE}" -v -s

echo
echo -e "${GREEN}Tests completed!${NC}"
