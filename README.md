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
export CUDA_VISIBLE_DEVICES=0
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

2. Install Python Client (Lightweight)

   ```bash
   # Install the lightweight client package (no PyTorch, no CUDA)
   cd python-client
   pip install -e .

   # Go back to main directory
   cd ..

   # Install dependencies for examples (optional)
   pip install transformers wandb
   ```

   Note: This installs only the lightweight client (~10 MB, installs in seconds). You do NOT need to install the full `xorl` package with PyTorch on your laptop - that stays on the GPU cluster.

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

1. Set Up SSH Tunnels (leave this terminal open)

   ```bash
   cd xorl  # Your xorl directory

   ./scripts/setup_ssh_tunnels.sh training-nodexx.cloud.xorl.ai
   ```

   Expected output:

   ```
   ==============================================================================
   Xorl SSH Tunnel Setup
   ==============================================================================

   GPU Host: training-node.cloud.xorl.ai
   SSH Key:  /Users/yourname/.ssh/id_ed25519

   Setting up tunnels for:
     - Port 5555: Training Server (main API)
     - Port 8000: Inference Worker 1
     - Port 8001: Inference Worker 2
     - Port 8080: Router (optional)

   Starting SSH tunnels...

   ✓ Setting up tunnel: localhost:5555 -> training-node.cloud.xorl.ai:5555 (Training Server)
   ✓ Setting up tunnel: localhost:8000 -> training-node.cloud.xorl.ai:8000 (Inference Worker 1)
   ✓ Setting up tunnel: localhost:8001 -> training-node.cloud.xorl.ai:8001 (Inference Worker 2)
   ✓ Setting up tunnel: localhost:8080 -> training-node.cloud.xorl.ai:8080 (Router (optional))

   ==============================================================================
   ✓ SSH Tunnels Established Successfully!
   ==============================================================================

   You can now run training from your laptop:
     python examples/rl/train_letter_counting_rl.py --api-url http://localhost:5555
   ```

2. Test Connection (optional but recommended, run in a new terminal)

   ```bash
   cd xorl
   python scripts/test_remote_connection.py
   ```

   Expected output:

   ```
   ================================================================================
   Xorl Remote Connection Test
   ================================================================================

   Testing Training Server... ✓ OK
   Testing Inference Worker 1... ✓ OK
   Testing Inference Worker 2... ✓ OK
   Testing Router (optional)... ✓ OK

   Testing API call (create_model)... ✓ OK (API is responsive)

   ================================================================================
   ✓ All critical tests passed!

   You can now run training scripts:
     python examples/rl/train_letter_counting_rl.py --api-url http://localhost:5555
   ================================================================================
   ```

3. Run Training Script

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

4. When Done, Clean Up Tunnels

   ```bash
   ./scripts/kill_ssh_tunnels.sh
   ```

   Expected output:

   ```
   Killing all Xorl SSH tunnels...

   ✓ Killing tunnel on port 5555 (Training Server)
   ✓ Killing tunnel on port 8000 (Inference Worker 1)
   ✓ Killing tunnel on port 8001 (Inference Worker 2)
   ✓ Killing tunnel on port 8080 (Router)

   Done!

   All tunnels closed.
   ```

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
   lsof -i :5555,:8000,:8001 -sTCP:LISTEN
   ```
2. If no output, restart tunnels:
   ```bash
   ./scripts/setup_ssh_tunnels.sh training-node.cloud.xorl.ai
   ```

#### "Address already in use" error

Problem: Port is already being used by another process

Solution:
```bash
# Kill existing tunnels
./scripts/kill_ssh_tunnels.sh

# Restart
./scripts/setup_ssh_tunnels.sh training-node.cloud.xorl.ai
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
- more detailed：[python-client/README.md](./python-client/README.md)
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

```
Your Laptop                           GPU Cluster
───────────                           ───────────

1. Training Script
      │
      ├── forward_backward() ───tunnel(5555)───▶ Training Server
      ├── optim_step() ──────tunnel(5555)───▶    │
      └── sample() ──────────tunnel(8000/8001)─▶ Inference Workers
                                                   │
                                                   └─▶ SGLang on GPUs
```

- All compute happens on GPU cluster
- Your laptop just sends commands and receives results
- Minimal network traffic (only control messages, not model weights or gradients)
- SSH tunnels are encrypted and secure