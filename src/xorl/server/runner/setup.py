"""
Worker setup and entry points.

Contains configuration dataclasses, distributed setup, and main() entry point.
The RunnerDispatcher class itself lives in dispatcher.py.
"""

import asyncio
import logging
import os
import socket
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml

from xorl.arguments import Arguments, parse_args
from xorl.server.server_arguments import ServerArguments, parse_server_args
from xorl.server.runner.runner_dispatcher import RunnerDispatcher
from xorl.server.runner.model_runner import ModelRunner
from xorl.utils.device import get_nccl_backend


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class WorkerArguments:
    """Arguments for distributed worker configuration."""

    bind_address: str = field(default="tcp://127.0.0.1:5556", metadata={"help": "ZMQ PAIR socket address to bind"})


@dataclass
class WorkerConfig(Arguments):
    """Configuration container for distributed worker, extends base Arguments."""

    worker: WorkerArguments = field(default_factory=WorkerArguments)


# ============================================================================
# Distributed Training Setup
# ============================================================================


def setup_distributed():
    """
    Setup PyTorch distributed training.

    Expects environment variables set by torchrun:
    - RANK: Global rank of this process
    - WORLD_SIZE: Total number of processes
    - LOCAL_RANK: Local rank on this node
    - MASTER_ADDR: Address of master node
    - MASTER_PORT: Port of master node

    Returns:
        Tuple of (rank, world_size, local_rank, cpu_group) where cpu_group is a
        Gloo process group for CPU-based communication (broadcast_object_list).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    cpu_group = None

    # Initialize process group
    if world_size > 1:
        backend = get_nccl_backend()
        dist.init_process_group(backend=backend)
        logger.info(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")

        # Create a separate Gloo process group for CPU-based communication.
        # Used by RunnerDispatcher/AdapterCoordinator for broadcast_object_list
        # (command broadcasting) — Gloo is more efficient for small CPU objects.
        cpu_group = dist.new_group(backend="gloo")
        logger.info("Created Gloo CPU process group for command broadcasting")
    else:
        logger.info(f"Single process mode: rank={rank}, local_rank={local_rank}")

    return rank, world_size, local_rank, cpu_group


def load_full_config(args: Arguments) -> Dict[str, Any]:
    """
    Convert Arguments to configuration dictionary for ModelRunner.

    Args:
        args: Parsed Arguments object

    Returns:
        Configuration dictionary
    """
    config = {
        "model": asdict(args.model),
        "train": asdict(args.train),
        "data": asdict(args.data),
    }

    # Load LoRA config from YAML if present
    # The LoRA section is not parsed by Arguments, so we read it directly from YAML
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        config_path = sys.argv[1]
        try:
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data and "lora" in yaml_data:
                    config["lora"] = yaml_data["lora"]
                    logger.info(f"Loaded LoRA config from YAML: {config['lora']}")
        except Exception as e:
            logger.warning(f"Could not load LoRA config from YAML: {e}")

    return config


# ============================================================================
# Entry Points
# ============================================================================


async def _run_worker(config: Dict[str, Any], bind_address: str, output_dir: str, log_level: str):
    """
    Shared worker bootstrap: init distributed, create ModelRunner + Worker, run until shutdown.

    Args:
        config: Configuration dictionary for ModelRunner
        bind_address: ZMQ bind address for rank 0
        output_dir: Output directory for checkpoints and address file
        log_level: Logging level string (e.g. "INFO", "DEBUG")
    """
    # Set NCCL environment variables BEFORE initializing any NCCL operations
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    logger.info("Set NCCL environment: NCCL_CUMEM_ENABLE=0, NCCL_NVLS_ENABLE=0, TORCH_NCCL_BLOCKING_WAIT=1")

    # Setup distributed training
    rank, world_size, local_rank, cpu_group = setup_distributed()

    # Setup logging with rank for all loggers
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=f"[Rank-{rank}][%(levelname)s][%(name)s] %(asctime)s >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        force=True,
    )

    # Suppress noisy HTTP cache-validation logs from transformers/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info("Starting Worker")
    logger.info(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
    logger.info(f"Bind address: {bind_address}")

    # Determine device
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Create ModelRunner (handles actual model operations)
    logger.info("Initializing ModelRunner...")
    trainer = ModelRunner(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    # Create RunnerDispatcher (handles ZMQ communication)
    logger.info("Initializing RunnerDispatcher...")
    worker = RunnerDispatcher(
        trainer=trainer,
        rank=rank,
        world_size=world_size,
        bind_address=bind_address,
        device=device,
        cpu_group=cpu_group,
        output_dir=output_dir,
    )

    # Run worker
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        await worker.stop()
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("Worker shut down successfully")


async def run_distributed_worker(args: Arguments, worker_args: WorkerArguments):
    """
    Main entry point for distributed model worker (legacy Arguments-based).

    Args:
        args: Parsed Arguments object containing model, train, and data config
        worker_args: Worker-specific arguments (bind address, etc.)
    """
    config = load_full_config(args)
    log_level = getattr(args.train, "log_level", "INFO")
    output_dir = getattr(args.train, "output_dir", "outputs")
    await _run_worker(config, worker_args.bind_address, output_dir, log_level)


async def run_distributed_worker_from_server_args(server_args: ServerArguments):
    """
    Main entry point for distributed model worker using ServerArguments.

    Args:
        server_args: ServerArguments with flat server-side configuration
    """
    # Resolve auto bind address
    if server_args.worker_bind_address == "auto":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            server_args.worker_bind_address = f"tcp://127.0.0.1:{s.getsockname()[1]}"

    config = server_args.to_config_dict()
    await _run_worker(config, server_args.worker_bind_address, server_args.output_dir, server_args.log_level)


def _is_server_args_config(config_path: str) -> bool:
    """
    Detect if a config file is in ServerArguments format (flat) or Arguments format (nested).

    ServerArguments format has flat keys like 'model_path', 'worker_bind_address'.
    Arguments format has nested sections like 'model:', 'train:', 'data:'.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        return False

    # If it has top-level keys like 'model_path' or 'worker_bind_address', it's ServerArguments
    server_args_keys = {"model_path", "worker_bind_address", "data_parallel_mode"}

    # If model is a dict (nested), it's Arguments format
    if "model" in config and isinstance(config["model"], dict):
        return False

    # If has flat server args keys, it's ServerArguments format
    return bool(server_args_keys & set(config.keys()))


def main():
    """Main entry point with command-line argument parsing."""
    # Set NCCL environment variables BEFORE any NCCL initialization
    # These must be set before torch.distributed is imported/initialized
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

    # Set unique Triton cache directory per rank to avoid race conditions
    # during parallel kernel compilation
    local_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    triton_base = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton"))
    triton_cache_dir = os.path.join(triton_base, f"cache_rank{local_rank}")
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    os.makedirs(triton_cache_dir, exist_ok=True)

    # Set per-rank TorchInductor cache to prevent cross-rank cubin path conflicts.
    # TorchInductor's FxGraphCache at /tmp/torchinductor_<user>/ is node-local and shared
    # by all local ranks. It stores absolute cubin paths from the compiling rank's
    # TRITON_CACHE_DIR, which breaks when a different local rank gets a cache hit.
    inductor_local_rank = os.environ.get("LOCAL_RANK", "0")
    inductor_base = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{os.environ.get('USER', 'unknown')}")
    inductor_cache_dir = os.path.join(inductor_base, f"rank{inductor_local_rank}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    os.makedirs(inductor_cache_dir, exist_ok=True)

    # Find config file path
    config_path = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--") and (arg.endswith(".yaml") or arg.endswith(".yml")):
            config_path = arg
            break

    if not config_path:
        print("Error: Config file path required as first positional argument")
        print(
            "Usage: python -m xorl.server.runner.runner_dispatcher config.yaml [--worker_bind_address tcp://...]"
        )
        sys.exit(1)

    # Detect config format and parse accordingly
    if _is_server_args_config(config_path):
        # ServerArguments format (flat config)
        logger.info("Detected ServerArguments format config")
        server_args = parse_server_args()
        asyncio.run(run_distributed_worker_from_server_args(server_args))
    else:
        # Legacy Arguments format (nested config with model/train/data/worker sections)
        logger.info("Detected Arguments format config")
        args = parse_args(WorkerConfig)
        asyncio.run(run_distributed_worker(args, args.worker))
