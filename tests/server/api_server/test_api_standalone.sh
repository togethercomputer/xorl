#!/bin/bash
# Standalone test script that starts server, runs tests, and cleans up
#
# This script:
# 1. Finds free ports
# 2. Starts API server with mock engine
# 3. Runs tests
# 4. Stops everything and cleans up
#
# Usage:
#   ./tests/server/api_server/test_api_standalone.sh

set -e  # Exit on error

echo "🧪 Standalone API Server Test"
echo "=============================="
echo ""

# Find free port for API server
find_free_port() {
    local start_port=$1
    for port in $(seq $start_port $((start_port + 100))); do
        if ! lsof -i:$port > /dev/null 2>&1; then
            echo $port
            return 0
        fi
    done
    echo "ERROR: Could not find free port starting from $start_port" >&2
    exit 1
}

# Find free ports
API_PORT=$(find_free_port 20000)
ENGINE_INPUT_PORT=$(find_free_port $((API_PORT + 1)))
ENGINE_OUTPUT_PORT=$(find_free_port $((ENGINE_INPUT_PORT + 1)))

echo "✅ Found free ports:"
echo "   - API Server: $API_PORT"
echo "   - Engine Input: $ENGINE_INPUT_PORT"
echo "   - Engine Output: $ENGINE_OUTPUT_PORT"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "✅ Server stopped"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start API server in background
echo "🚀 Starting API server..."
python -m xorl.server.api_server \
    --mock \
    --port $API_PORT \
    --engine-input tcp://127.0.0.1:$ENGINE_INPUT_PORT \
    --engine-output tcp://127.0.0.1:$ENGINE_OUTPUT_PORT \
    > /tmp/api_server_$API_PORT.log 2>&1 &

SERVER_PID=$!

echo "   PID: $SERVER_PID"
echo "   Log: /tmp/api_server_$API_PORT.log"

# Wait for server to start
echo "⏳ Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:$API_PORT/health > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Server failed to start within 30 seconds"
        echo "Log tail:"
        tail -20 /tmp/api_server_$API_PORT.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "="*70
echo "🔬 Running Tests"
echo "="*70
echo ""

# Run tests
TEST_FAILED=0

# Test 1: Health Check
echo "1️⃣  Testing Health Check..."
if curl -s http://127.0.0.1:$API_PORT/health | grep -q "healthy"; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
    TEST_FAILED=1
fi

# Test 2: Forward-Backward
echo "2️⃣  Testing Forward-Backward..."
RESPONSE=$(curl -s -X POST http://127.0.0.1:$API_PORT/api/v1/forward_backward \
    -H "Content-Type: application/json" \
    -d '{
        "model_id": "test-model",
        "forward_backward_input": {
            "data": [{
                "model_input": {"input_ids": [1, 2, 3, 4]},
                "loss_fn_inputs": {"labels": [2, 3, 4, 5]}
            }],
            "loss_fn": "causallm_loss"
        }
    }')

if echo "$RESPONSE" | grep -q "loss_fn_outputs"; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
    echo "   Response: $RESPONSE"
    TEST_FAILED=1
fi

# Test 3: Optimizer Step
echo "3️⃣  Testing Optimizer Step..."
RESPONSE=$(curl -s -X POST http://127.0.0.1:$API_PORT/api/v1/optim_step \
    -H "Content-Type: application/json" \
    -d '{
        "model_id": "test-model",
        "adam_params": {"learning_rate": 0.0001},
        "gradient_clip": 1.0
    }')

if echo "$RESPONSE" | grep -q "metrics"; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
    echo "   Response: $RESPONSE"
    TEST_FAILED=1
fi

# Test 4: Save Weights
echo "4️⃣  Testing Save Weights..."
CHECKPOINT_PATH="/tmp/test_checkpoint_$$"
RESPONSE=$(curl -s -X POST http://127.0.0.1:$API_PORT/api/v1/save_weights \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"test-model\",
        \"path\": \"$CHECKPOINT_PATH\",
        \"save_optimizer\": true
    }")

if echo "$RESPONSE" | grep -q "path"; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
    echo "   Response: $RESPONSE"
    TEST_FAILED=1
fi

# Test 5: Load Weights
echo "5️⃣  Testing Load Weights..."
RESPONSE=$(curl -s -X POST http://127.0.0.1:$API_PORT/api/v1/load_weights \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"test-model\",
        \"path\": \"$CHECKPOINT_PATH\",
        \"load_optimizer\": true
    }")

if echo "$RESPONSE" | grep -q "success"; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
    echo "   Response: $RESPONSE"
    TEST_FAILED=1
fi

echo ""
echo "="*70
echo "📊 TEST SUMMARY"
echo "="*70

if [ $TEST_FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed!"
    echo ""
    echo "Check the server log for details:"
    echo "  tail -100 /tmp/api_server_$API_PORT.log"
    exit 1
fi
