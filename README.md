# Xorl

## Overview

Xorl is a distributed training framework for large language models, built on top of [Xorl](https://github.com/xorl-org/Xorl). It provides simple, high-performance training across multiple GPUs and nodes.

### Key Features

- High-performance distributed training
- Modular and flexible design
- Linear training scripts (no rigid trainer classes)
- Native PyTorch integration
- Easy scaling from single GPU to clusters

### Design Principles

Xorl follows Xorl's core principles:

- **Flexibility**: Modular components that can be easily customized or replaced
- **Trainer-free**: Linear training scripts instead of rigid trainer classes
- **Model-agnostic**: Works with any model architecture
- **PyTorch-native**: Leverages PyTorch's native functions for compatibility and performance

## Server Running

There are two ways to run the training server:

### Option 1: Using Docker Compose (Recommended for Production)

This method runs the server and inference workers in isolated containers with proper GPU allocation.

```bash
# SSH to GPU cluster
ssh training-nodexx.cloud.xorl.ai

# Navigate to project directory
cd /path/to/xorl

# Build images (first time or after Dockerfile changes)
docker compose build

# Start all services in detached mode
docker compose up -d

# Check service status (wait ~5 minutes for inference workers to initialize)
docker compose ps  # All should show "healthy"

# View logs
docker compose logs -f training-server
docker compose logs -f inference-worker

# Stop services when done
docker compose down
```

**Configuration:** Edit `docker-compose.yaml` to adjust:
- GPU allocation (`CUDA_VISIBLE_DEVICES`)
- Model paths (`MODEL_PATH`)
- Ports (`TRAINING_PORT`, `WORKER_PORT`)

### Option 2: Using Shell Script (Development/Direct Execution)

This method runs the server directly on the host without Docker containers.

```bash
# SSH to GPU cluster
ssh training-nodexx.cloud.xorl.ai

# Navigate to project directory
cd /path/to/xorl

# Set environment variables (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAINING_PORT=5555
export CONFIG_PATH=examples/server_sft/server.yaml

# Run the server
./scripts/run_server.sh

# Or with inline environment variables
CUDA_VISIBLE_DEVICES=0 CONFIG_PATH=examples/server/sft.yaml ./scripts/run_server.sh
```

**Configuration:** Set these environment variables before running:
- `TRAINING_PORT`: API server port (default: 5555)
- `MASTER_PORT`: PyTorch distributed port (default: 30000)
- `CONFIG_PATH`: Path to training config YAML
- `CUDA_VISIBLE_DEVICES`: GPU devices to use
- `LOG_LEVEL`: Logging verbosity (default: INFO)
   
## Remote Training

Train ML models from your laptop using GPU cluster compute. All compute happens on the GPU cluster; your laptop sends commands and receives results.

### Prerequisites

- SSH access to GPU cluster: `training-node.cloud.xorl.ai`
- SSH key at `~/.ssh/id_ed25519` (or set `SSH_KEY` environment variable)
- Python 3.8+ installed

### One-Time Setup

1. Clone repository

   ```bash
   # Clone the repo (if you haven't already)
   git clone git@github.com:xorl-org/xorl.git
   cd xorl
   ```

2. Install Python Client (xorl_client)

   The recommended client library for interacting with the Xorl training server is **[xorl_client](https://github.com/xorl-org/xorl_client)** - a lightweight Python client.

   ```bash
   # Install xorl_client client package (no PyTorch, no CUDA required)
   pip install xorl_client

   # Or install from source
   git clone https://github.com/xorl-org/xorl_client.git
   cd xorl_client
   pip install -e .

   # Install dependencies for examples (optional)
   pip install transformers wandb
   ```

   Note: This installs only the lightweight client (~10 MB, installs in seconds). You do NOT need to install the full `xorl` package with PyTorch on your laptop - that stays on the GPU cluster.

   **Quick Example using xorl_client:**

   ```python
   import xorl_client
   from xorl_client import types

   # Connect to the Xorl training server
   training_client = xorl_client.ServiceClient(
       base_url="http://localhost:6000"
   ).create_lora_training_client(base_model="Qwen/Qwen3-4B-Instruct-2507")

   # Get the tokenizer
   tokenizer = training_client.get_tokenizer()

   # Create training data
   prompt = "English: hello world\nPig Latin:"
   prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
   completion_tokens = tokenizer.encode(" ello-hay orld-way\n\n", add_special_tokens=False)

   tokens = prompt_tokens + completion_tokens
   weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

   input_tokens = tokens[:-1]
   target_tokens = tokens[1:]
   weights = weights[1:]

   datum = types.Datum(
       model_input=types.ModelInput.from_ints(tokens=input_tokens),
       loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
   )

   # Run forward-backward pass
   fwdbwd_result = training_client.forward_backward([datum], "cross_entropy").result()
   print(f"Loss: {fwdbwd_result.metrics['loss:mean']:.4f}")

   # Run optimizer step
   optim_result = training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
   print(f"Grad Norm: {optim_result.metrics['grad_norm']:.4f}")
   ```

3. Set SSH Key & Verify Access

   If you already have a explicit key, set it and test:
   ```bash
   SSH_KEY=~/.ssh/id_ed25519 ssh training-nodexx.cloud.xorl.ai
   # Should connect without a password; type 'exit' to leave
   ```

   If you need a new key, generate and install it on the GPU cluster:
   ```bash
   # Generate a new ed25519 key (press enter to skip passphrase if desired)
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "xxx@xorl.ai"
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/id_ed25519

   # Copy the public key to the GPU cluster
   ssh-copy-id -i ~/.ssh/id_ed25519.pub training-nodexx.cloud.xorl.ai
   # If ssh-copy-id isn't available, use:
   # cat ~/.ssh/id_ed25519.pub | ssh training-nodexx.cloud.xorl.ai \
   #   'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys'

   # Verify access
   SSH_KEY=~/.ssh/id_ed25519 ssh training-nodexx.cloud.xorl.ai
   ```

### Every Training Session

```bash
cd xorl
# Example 1: simple SFT run
python examples/server_sft/train.py examples/server_sft/client.yaml

# Example 2: Letter counting RL training
python examples/rl/train_letter_counting_rl.py \
      --api-url http://localhost:5555 \
      --num-epochs 5 \
      --batch-size 8 \
      --group-size 4 \
      --learning-rate 1e-5 \
      --num-examples 50 \
      --wandb-project my-rl-experiments \
      --wandb-run-name test-run-1

# Or run with minimal options
python examples/rl/train_letter_counting_rl.py --api-url http://localhost:5555
```

What you'll see:
- Real-time training logs in your terminal
- Training happens on the GPU cluster (you'll see progress updates)
- When complete, checkpoints are saved on the GPU cluster


### Troubleshooting

#### "Permission denied" when running setup script

```bash
# Make sure scripts are executable
chmod +x scripts/*.sh scripts/*.py
```

#### "Connection refused" error

Problem: Can't connect to localhost:5555

Solution:
1. Check tunnels are running:
   ```bash
   lsof -i :5555 -sTCP:LISTEN
   lsof -i :8000 -sTCP:LISTEN
   lsof -i :8001 -sTCP:LISTEN
   ```

#### Services not responding on GPU cluster

Problem: Tunnels connect but APIs don't respond

Solution: SSH to GPU cluster and check services:
```bash
ssh training-nodexx.cloud.xorl.ai
cd xorl
docker compose ps  # All should show "healthy"

# If not healthy, restart
docker compose restart
```

#### SSH key not found

Problem: Script can't find your SSH key

Solution:
```bash
# Specify SSH key location
SSH_KEY=~/.ssh/my_custom_key ./scripts/setup_ssh_tunnels.sh training-node.cloud.xorl.ai
```

#### Service Issues

```bash
# On GPU cluster
docker-compose restart
docker-compose ps  # Check all are "healthy"
```

### Advanced Tips
- For detailed client API documentation, see the [xorl_client repository](https://github.com/xorl-org/xorl_client)
- Keep tunnels running in background with tmux:
  ```bash
  tmux new -s xorl-tunnels
  ./scripts/setup_ssh_tunnels.sh training-node.cloud.xorl.ai
  # Detach: Ctrl-b, d; Reattach: tmux attach -t xorl-tunnels
  ```
- Run multiple training sessions (auto uses different inference workers):
  ```bash
  python examples/rl/train_letter_counting_rl.py --api-url http://localhost:5555 --seed 1
  python examples/rl/train_letter_counting_rl.py --api-url http://localhost:5555 --seed 2
  ```
- Use env var for API URL to avoid retyping:
  ```bash
  export XORL_API_URL=http://localhost:5555
  python examples/rl/train_letter_counting_rl.py
  ```

### What's Happening Behind the Scenes

The Xorl Training Server consists of three main components that work together to handle distributed training requests:

#### 1. API Server (HTTP Interface)
- **Protocol**: HTTP/REST API (FastAPI + uvicorn)
- **Address**: `0.0.0.0:5555` (externally accessible)
- **Purpose**: User-facing interface for training operations
- **Features**:
  - RESTful endpoints (`/api/v1/forward_backward`, `/api/v1/optim_step`, etc.)
  - Request validation and error handling
  - Authentication and logging
  - Automatic API documentation at `/docs`

The API Server receives HTTP requests from clients and translates them into internal messages for the Engine Core.

#### 2. Engine Core (Training Scheduler)
- **Protocol**: ZeroMQ (ZMQ) message queue
- **Input Socket**: `tcp://127.0.0.1:5555` (receives requests from API Server)
- **Output Socket**: `tcp://127.0.0.1:5001` (sends results back to API Server)
- **Purpose**: Orchestrates training requests and manages request queues
- **Features**:
  - Request scheduling and prioritization
  - Queue management (max 2 running, 100 pending requests by default)
  - Connection management with distributed workers
  - Automatic sequence packing for efficient batch processing

The Engine Core sits between the API Server and Workers, handling the complexity of distributed training coordination.

#### 3. Distributed Workers (Training Execution)
- **Protocol**: ZeroMQ (ZMQ)
- **Address**: `tcp://127.0.0.1:5556` (rank 0 worker)
- **Purpose**: Execute actual training operations on GPUs
- **Features**:
  - Multi-GPU distributed training (FSDP2, Ulysses, etc.)
  - Forward and backward passes
  - Gradient computation and aggregation
  - Checkpoint saving/loading

Workers are launched via `torchrun` and handle all GPU computation.

#### Why ZeroMQ (ZMQ)?
ZeroMQ is a high-performance asynchronous messaging library that provides:
- **Low latency**: Microsecond-level message passing
- **High throughput**: Can handle thousands of messages per second
- **Flexible patterns**: Supports various messaging patterns (ROUTER, DEALER, PUSH, PULL)
- **Language agnostic**: Works across different programming languages
- **No broker overhead**: Direct socket-to-socket communication

In Xorl, ZMQ is used for internal inter-process communication (IPC) between components running on the same machine, while HTTP is used for external client-server communication.

#### Complete Request Flow

```
Client (Your Laptop)
    │
    │ HTTP POST /api/v1/forward_backward
    ▼
API Server (Port 5555)
    │
    │ ZMQ → Engine Input (tcp://127.0.0.1:5555)
    ▼
Engine Core
    │
    │ ZMQ → Worker Address (tcp://127.0.0.1:5556)
    ▼
Distributed Workers (8 GPUs)
    │ [Execute training on GPUs]
    │ - Forward pass
    │ - Backward pass
    │ - Compute gradients
    │
    │ ZMQ ← Results
    ▼
Engine Core
    │
    │ ZMQ → Engine Output (tcp://127.0.0.1:5001)
    ▼
API Server
    │
    │ HTTP Response (JSON)
    ▼
Client (Your Laptop)
```

#### Why This Three-Layer Design?

1. **Separation of Concerns**: Each component has a single responsibility
   - API Server: External communication and validation
   - Engine Core: Request orchestration and scheduling
   - Workers: GPU computation

2. **Performance**: Internal components use ZMQ for high-speed communication while maintaining a friendly HTTP interface for clients

3. **Scalability**: Components can be independently scaled or restarted
   - Multiple API Servers can connect to the same Engine Core
   - Engine Core manages multiple concurrent training sessions
   - Workers can be distributed across multiple nodes

4. **Security**: Internal ZMQ sockets are only exposed locally (`127.0.0.1`), while only the API Server is accessible externally

5. **Flexibility**: Easy to add new features like monitoring, load balancing, or different worker types without changing the client interface