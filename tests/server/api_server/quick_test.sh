#!/bin/bash
# Quick test script for API server

echo "🚀 Testing API Server with Mock Engine"
echo "========================================"
echo ""

# Check if server is running
if ! curl -s http://127.0.0.1:20000/health > /dev/null 2>&1; then
    echo "❌ API server is not running!"
    echo ""
    echo "Please start the server first:"
    echo "  python -m xorl.server.api_server --mock"
    echo ""
    exit 1
fi

echo "✅ Server is running!"
echo ""

# Test health check
echo "1️⃣  Testing Health Check..."
curl -s http://127.0.0.1:20000/health | python3 -m json.tool
echo ""

# Test forward-backward
echo "2️⃣  Testing Forward-Backward..."
curl -s -X POST http://127.0.0.1:20000/api/v1/forward_backward \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "forward_backward_input": {
      "data": [{
        "model_input": {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
        "loss_fn_inputs": {"labels": [2, 3, 4, 5, 6, 7, 8, 9]}
      }],
      "loss_fn": "causallm_loss"
    }
  }' | python3 -m json.tool
echo ""

# Test optimizer step
echo "3️⃣  Testing Optimizer Step..."
curl -s -X POST http://127.0.0.1:20000/api/v1/optim_step \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "adam_params": {
      "learning_rate": 0.0001,
      "beta1": 0.9,
      "beta2": 0.95,
      "eps": 1e-12
    },
    "gradient_clip": 1.0
  }' | python3 -m json.tool
echo ""

# Test save weights
echo "4️⃣  Testing Save Weights..."
curl -s -X POST http://127.0.0.1:20000/api/v1/save_weights \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "path": "/tmp/test_checkpoint",
    "save_optimizer": true
  }' | python3 -m json.tool
echo ""

# Test load weights
echo "5️⃣  Testing Load Weights..."
curl -s -X POST http://127.0.0.1:20000/api/v1/load_weights \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "path": "/tmp/test_checkpoint",
    "load_optimizer": true
  }' | python3 -m json.tool
echo ""

echo "========================================"
echo "✅ All tests completed!"
echo ""
echo "💡 For interactive testing:"
echo "   - Open http://127.0.0.1:20000/docs in your browser"
echo "   - Or run: python tests/server/api_server/test_api_client.py"
