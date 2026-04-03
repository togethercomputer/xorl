"""
Launcher for API Server, Engine Core, and Distributed Model Workers.

This script launches all components needed for the training server with two modes:

Mode 1: Auto-launch (--mode auto)
  - Automatically launches distributed workers using torchrun
  - Starts engine and API server
  - Best for single-node or easy setup

Mode 2: Connect (--mode connect)
  - Assumes workers are already running externally
  - Only starts engine and API server
  - Best for multi-node or manual worker management

Features:
- Automatic port finding
- Proper logging setup
- Graceful shutdown
- Status monitoring
- Support for both distributed and dummy workers
"""

import argparse
import logging
import multiprocessing as mp
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import uvicorn
import yaml

from xorl.server.api_server.server import APIServer
from xorl.server.orchestrator.orchestrator import Orchestrator
from xorl.server.server_arguments import ServerArguments


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s][%(name)s] %(asctime)s >> %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP cache-validation logs from transformers/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class RetrieveFutureFilter(logging.Filter):
    """Filter out noisy retrieve_future polling requests from access logs.

    The two-phase async pattern causes clients to poll /api/v1/retrieve_future
    every second until results are ready. This filter suppresses those logs
    to avoid cluttering the output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out retrieve_future access log entries
        if hasattr(record, "getMessage"):
            msg = record.getMessage()
            if "/api/v1/retrieve_future" in msg:
                return False
        return True


def configure_uvicorn_logging():
    """Configure uvicorn access logging to filter out polling requests."""
    # Add filter to uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(RetrieveFutureFilter())


# ============================================================================
# Port Finding Utilities
# ============================================================================


def find_free_port(start_port: int = 50000, max_attempts: int = 10000) -> int:
    """
    Find a free port by randomly picking from a range.

    Args:
        start_port: Start of port range to search
        max_attempts: Maximum number of ports to try

    Returns:
        Free port number

    Raises:
        RuntimeError: If no free port found
    """
    import random

    end_port = min(start_port + max_attempts, 60000)
    ports = list(range(start_port, end_port))
    random.shuffle(ports)
    for port in ports:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("", port))
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{end_port}")


def find_free_ports(count: int, start_port: int = 50000) -> List[int]:
    """
    Find multiple free ports.

    Args:
        count: Number of ports to find
        start_port: Port to start searching from

    Returns:
        List of free port numbers
    """
    ports = []
    current_start = start_port

    for _ in range(count):
        port = find_free_port(current_start)
        ports.append(port)
        current_start = port + 1

    return ports


# ============================================================================
# Engine Core Process
# ============================================================================


def run_orchestrator(
    input_addr: str,
    output_addr: str,
    rank0_worker_address: str,
    max_running_requests: int = 2,
    max_pending_requests: int = 100,
    operation_timeout: float = 1800.0,
    log_level: str = "INFO",
    sample_packing_sequence_len: int = 32000,
    enable_packing: bool = True,
    ready_event: Optional[mp.Event] = None,
    output_dir: str = "outputs",
):
    """
    Run the Orchestrator in a separate process.

    Args:
        input_addr: Address for receiving requests from API server
        output_addr: Address for sending outputs to API server
        rank0_worker_address: Address of rank0 worker
        max_running_requests: Maximum concurrent running requests
        max_pending_requests: Maximum pending requests in queue
        log_level: Logging level
        ready_event: Optional multiprocessing Event to signal when engine is ready
        output_dir: Output directory for logs and checkpoints
    """
    # Setup logging first, outside try block
    # Create logs directory under output_dir if it doesn't exist
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "orchestrator.log")

    # Setup logging to both file and stdout
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(levelname)s][ENGINE] %(asctime)s >> %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("Orchestrator")
    logger.info(f"Engine Core logging to: {os.path.abspath(log_file)}")

    logger.info("=" * 70)
    logger.info("Starting Engine Core Process")
    logger.info("=" * 70)
    logger.info(f"Input address:  {input_addr}")
    logger.info(f"Output address: {output_addr}")
    logger.info(f"Worker address: {rank0_worker_address}")
    logger.info(f"Max running:    {max_running_requests}")
    logger.info(f"Max pending:    {max_pending_requests}")

    try:
        logger.info("Initializing Orchestrator...")
        engine = Orchestrator(
            input_addr=input_addr,
            output_addr=output_addr,
            rank0_worker_address=rank0_worker_address,
            operation_timeout=operation_timeout,
            connection_timeout=3600.0,  # 1 hour for loading large models (235B) + EP sharding + LoRA + Triton compilation
            sample_packing_sequence_len=sample_packing_sequence_len,
            enable_packing=enable_packing,
        )
        logger.info(f"Orchestrator initialized successfully (operation_timeout={operation_timeout}s)")

        logger.info("Starting Orchestrator...")
        engine.start()
        logger.info("Engine Core started successfully")

        # Signal that engine is ready
        if ready_event is not None:
            ready_event.set()
            logger.info("Signaled ready event to launcher")

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in engine core: {e}", exc_info=True)
    finally:
        if "engine" in locals():
            engine.stop()
        logger.info("Engine Core stopped")


# ============================================================================
# API Server Process
# ============================================================================


def run_api_server(
    host: str,
    port: int,
    engine_input_addr: str,
    engine_output_addr: str,
    log_level: str = "INFO",
    default_timeout: float = 120.0,
    output_dir: str = "outputs",
    base_model: Optional[str] = None,
    storage_limit: str = "10TB",
    idle_session_timeout: float = 7200.0,
    skip_initial_checkpoint: bool = False,
    sync_inference_method: str = "nccl_broadcast",
):
    """
    Run the API Server in a separate process.

    Args:
        host: Host to bind the API server
        port: Port to bind the API server
        engine_input_addr: Address to send requests to engine
        engine_output_addr: Address to receive outputs from engine
        log_level: Logging level
        default_timeout: Default timeout for engine operations
        output_dir: Output directory for checkpoints and sampler weights (must be on shared filesystem)
        base_model: Base model name that this server is configured for (e.g., 'Qwen/Qwen2.5-3B-Instruct')
        storage_limit: Maximum disk usage for output_dir (e.g., '1GB'). Default: 10TB.
        idle_session_timeout: Idle session timeout in seconds. Default: 7200.0 (2 hours).
        skip_initial_checkpoint: Skip auto-saving initial checkpoint on first create_model.
        sync_inference_method: Method for syncing weights to inference endpoints. Default: 'nccl_broadcast'.
    """
    from contextlib import asynccontextmanager

    # Setup logging for this process
    logging.basicConfig(
        level=getattr(logging, log_level), format="[%(levelname)s][API] %(asctime)s >> %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger("APIServer")

    # Apply filter to suppress noisy retrieve_future polling logs
    configure_uvicorn_logging()

    logger.info("=" * 70)
    logger.info("Starting API Server")
    logger.info("=" * 70)
    logger.info(f"API Server:     http://{host}:{port}")
    logger.info(f"Engine input:   {engine_input_addr}")
    logger.info(f"Engine output:  {engine_output_addr}")

    try:
        # Import the FastAPI app from api_server module
        # Update the global state shared between api_server.py and endpoints.py
        import xorl.server.api_server._state as _state_module
        from xorl.server.api_server.server import app

        # Override the lifespan to use our addresses
        @asynccontextmanager
        async def custom_lifespan(app):
            """Custom lifecycle with configurable addresses."""
            logger.info("Starting APIServer with custom addresses...")
            logger.info(f"  output_dir: {output_dir}")
            logger.info(f"  base_model: {base_model}")
            logger.info(f"  storage_limit: {storage_limit}")
            logger.info(f"  idle_session_timeout: {idle_session_timeout}s")
            _state_module.api_server = APIServer(
                engine_input_addr=engine_input_addr,
                engine_output_addr=engine_output_addr,
                default_timeout=default_timeout,
                output_dir=output_dir,
                base_model=base_model,
                storage_limit=storage_limit,
                idle_session_timeout=idle_session_timeout,
                skip_initial_checkpoint=skip_initial_checkpoint,
                sync_inference_method=sync_inference_method,
            )
            await _state_module.api_server.start()
            yield
            logger.info("Shutting down APIServer...")
            if _state_module.api_server:
                await _state_module.api_server.stop()
                _state_module.api_server = None

        # Replace the lifespan
        app.router.lifespan_context = custom_lifespan

        # Run with uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            access_log=True,
        )
        server = uvicorn.Server(config)
        server.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in API server: {e}", exc_info=True)
    finally:
        logger.info("API Server stopped")


# ============================================================================
# Configuration Parsing
# ============================================================================


def load_server_arguments(config_path: str, overrides: Optional[Dict[str, any]] = None) -> ServerArguments:
    """
    Load ServerArguments from a YAML configuration file.

    Supports both flat config (ServerArguments style) and nested config
    (Arguments style with model/train/worker sections).

    Args:
        config_path: Path to server config YAML
        overrides: Optional dict of CLI overrides to apply on top of YAML config.
                   Keys should be ServerArguments field names (e.g., 'output_dir', 'lora_rank').

    Returns:
        ServerArguments instance with all fields populated
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty config file: {config_path}")

    from dataclasses import fields

    valid_fields = {f.name for f in fields(ServerArguments)}

    # Check if this is a nested config (has model/train/worker sections)
    if "model" in config and isinstance(config["model"], dict):
        # Nested config - flatten all keys from each section into one dict.
        # Keys whose names already match a ServerArguments field are copied
        # directly; a few nested keys need explicit remapping (worker.*,
        # lora.exclude_modules).  This is forward-compatible: new fields
        # added to ServerArguments + to_config_dict() are picked up
        # automatically without touching this flattening logic.
        flat_config = {}

        # model.* and train.* keys map 1:1 to ServerArguments fields
        for section in ("model", "train"):
            for k, v in config.get(section, {}).items():
                flat_config[k] = v

        # lora.* keys also map 1:1 except exclude_modules → qlora_exclude_modules
        for k, v in config.get("lora", {}).items():
            if k == "exclude_modules":
                flat_config["qlora_exclude_modules"] = v
            else:
                flat_config[k] = v

        # data.* — only a few fields are relevant for the server
        data_config = config.get("data", {})
        if "sample_packing_sequence_len" not in flat_config:
            flat_config["sample_packing_sequence_len"] = data_config.get("sample_packing_sequence_len", 32000)
        if "enable_packing" not in flat_config:
            flat_config["enable_packing"] = data_config.get("enable_packing", True)

        # worker.* keys are prefixed with worker_ in ServerArguments
        worker_config = config.get("worker", {})
        _worker_key_map = {
            "bind_address": "worker_bind_address",
            "bind_host": "worker_bind_host",
            "bind_port": "worker_bind_port",
            "engine_connect_host": "engine_connect_host",
            "connection_timeout": "worker_connection_timeout",
            "max_retries": "worker_max_retries",
        }
        for nested_key, flat_key in _worker_key_map.items():
            if nested_key in worker_config:
                flat_config[flat_key] = worker_config[nested_key]

        # Top-level keys that can appear outside any section
        for k in ("output_dir", "storage_limit", "idle_session_timeout"):
            if k not in flat_config and k in config:
                flat_config[k] = config[k]

        filtered_config = {k: v for k, v in flat_config.items() if k in valid_fields and v is not None}
    else:
        # Flat config (ServerArguments style)
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}

        # Handle None values for Optional fields
        for key, value in list(filtered_config.items()):
            if value is None:
                if key in ["config_path", "tokenizer_path"]:
                    del filtered_config[key]

    # Apply CLI overrides on top of YAML config
    if overrides:
        for key, value in overrides.items():
            if key in valid_fields:
                filtered_config[key] = value
                logger.info(f"  CLI override: {key} = {value}")
            else:
                logger.warning(f"  Unknown override key ignored: {key}")

    logger.info(f"Loaded ServerArguments from: {config_path}")
    logger.info(f"  model_path: {filtered_config.get('model_path', 'N/A')}")

    return ServerArguments(**filtered_config)


def calculate_world_size_from_config(config_path: str) -> int:
    """
    Calculate the total number of GPUs required from parallelism configuration.

    EP and the main parallelism mesh share the same GPUs but organize them differently:
    - Main mesh: DP × Ulysses × CP × TP × PP
    - EP mesh: EP × ep_fsdp_size (where ep_fsdp_size = world_size / ep_size)

    The total GPUs needed is MAX(EP, main_mesh_size), rounded up to be divisible by EP.

    Args:
        config_path: Path to server config YAML

    Returns:
        Total number of GPUs needed
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Support both flat config (ServerArguments style) and nested config (train: section)
    if "train" in config:
        train_config = config.get("train", {})
    else:
        train_config = config

    # Get parallelism sizes (with defaults matching model_runner.py)
    ep_size = train_config.get("expert_parallel_size", 1)
    ulysses_size = train_config.get("ulysses_parallel_size", 1)
    ringattn_size = train_config.get("ringattn_parallel_size", 1)

    # Data parallel sizes
    dp_replicate_size = train_config.get("data_parallel_replicate_size", 1)
    dp_shard_size = train_config.get("data_parallel_shard_size", 1)

    # Calculate world size
    # EP creates a separate 2D mesh where: world_size = ep_size * ep_fsdp_size
    # ep_fsdp_size contains all other parallelism dimensions
    world_size = dp_replicate_size * dp_shard_size * ulysses_size * ringattn_size
    ep_fsdp_size = world_size // ep_size

    logger.info("Calculated world size from config:")
    logger.info(f"  expert_parallel_size:         {ep_size}")
    logger.info(f"  ulysses_parallel_size:        {ulysses_size}")
    logger.info(f"  ringattn_parallel_size:           {ringattn_size}")
    logger.info(f"  cp_fsdp_mode:                 {train_config.get('cp_fsdp_mode', 'all')}")
    logger.info(f"  data_parallel_replicate_size: {dp_replicate_size}")
    logger.info(f"  data_parallel_shard_size:     {dp_shard_size}")
    logger.info(f"  ep_fsdp_size:              {ep_fsdp_size} (dp_rep×dp_shard×ulysses / ep_size)")
    logger.info(f"  => Total world size:          {world_size}")

    return world_size


# ============================================================================
# Main Launcher
# ============================================================================


class Launcher:
    """Main launcher for all server components."""

    def __init__(
        self,
        mode: str = "auto",
        config_path: Optional[str] = None,
        worker_address: Optional[str] = None,
        api_host: str = "0.0.0.0",
        api_port: Optional[int] = None,
        max_running_requests: int = 2,
        max_pending_requests: int = 100,
        operation_timeout: float = 1800.0,
        log_level: str = "INFO",
        # Auto-launch mode parameters
        nnodes: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        # Data processing parameters
        sample_packing_sequence_len: int = 32000,
        enable_packing: bool = True,
        # Server config overrides (from --server.* CLI args)
        server_overrides: Optional[Dict[str, any]] = None,
    ):
        """
        Initialize the launcher.

        Args:
            mode: Launch mode - "auto" or "connect"
            config_path: Path to config YAML for workers (required for auto mode)
            worker_address: Worker ZMQ address (for connect mode or auto mode default)
            api_host: API server host
            api_port: API server port (auto-find if None)
            max_running_requests: Maximum concurrent requests
            max_pending_requests: Maximum pending requests
            log_level: Logging level
            nnodes: Number of nodes for distributed training (auto mode)
            master_addr: Master address for torch distributed (auto mode)
            master_port: Master port for torch distributed (auto mode)
            server_overrides: Dict of ServerArguments field overrides from CLI (e.g., {'output_dir': '/tmp/out'})
        """
        self.mode = mode
        self.config_path = config_path
        self.api_host = api_host
        self.log_level = log_level
        self.max_running_requests = max_running_requests
        self.max_pending_requests = max_pending_requests
        self.operation_timeout = operation_timeout
        self.sample_packing_sequence_len = sample_packing_sequence_len
        self.enable_packing = enable_packing
        self.server_overrides = server_overrides or {}

        # Validate mode
        if mode not in ["auto", "connect"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto' or 'connect'")

        # Validate config for auto mode
        if mode == "auto" and not config_path:
            raise ValueError("--config is required for auto-launch mode")

        # Load ServerArguments from config if provided
        self.server_args: Optional[ServerArguments] = None
        if config_path:
            try:
                self.server_args = load_server_arguments(config_path, overrides=self.server_overrides)
                logger.info("Successfully loaded ServerArguments from config")
                logger.info(f"  model_path: {self.server_args.model_path}")
                logger.info(f"  data_parallel_mode: {self.server_args.data_parallel_mode}")
                logger.info(f"  ulysses_parallel_size: {self.server_args.ulysses_parallel_size}")
            except Exception as e:
                logger.warning(f"Could not load ServerArguments: {e}")
                logger.warning("Falling back to raw YAML parsing")

        # Distributed training parameters
        self.nnodes = nnodes
        self.master_addr = master_addr
        self.master_port = master_port

        # Calculate nproc_per_node from config in auto mode
        if mode == "auto":
            logger.info("Calculating world size from config parallelism settings...")
            if self.server_args:
                # Use get_total_gpus() which includes EP (Expert Parallel)
                total_world_size = self.server_args.get_total_gpus()
            else:
                total_world_size = calculate_world_size_from_config(config_path)
            # For multi-node, nproc_per_node is total world size divided by number of nodes
            self.nproc_per_node = total_world_size // self.nnodes
            logger.info(
                f"Total world size = {total_world_size}, nnodes = {self.nnodes}, nproc_per_node = {self.nproc_per_node}"
            )
        else:
            self.nproc_per_node = 1  # Not used in connect mode

        # Find free ports
        logger.info("Finding free ports...")

        # API port is user-configurable, others are auto-found
        if api_port:
            self.api_port = api_port
            # Find 3 more ports for engine and worker
            ports = find_free_ports(3)
            self.engine_input_port = ports[0]
            self.engine_output_port = ports[1]
            self.worker_port = ports[2] if not worker_address else None
        else:
            # Find all 4 ports
            ports = find_free_ports(4)
            self.api_port = ports[0]
            self.engine_input_port = ports[1]
            self.engine_output_port = ports[2]
            self.worker_port = ports[3] if not worker_address else None

        # Build addresses
        self.engine_input_addr = f"tcp://127.0.0.1:{self.engine_input_port}"
        self.engine_output_addr = f"tcp://127.0.0.1:{self.engine_output_port}"

        # Worker address - prefer from ServerArguments if available
        if worker_address:
            self.worker_address = worker_address
        elif self.server_args and self.server_args.worker_bind_address != "auto":
            self.worker_address = self.server_args.worker_bind_address
            logger.info(f"Using worker address from config: {self.worker_address}")
        elif self.server_args:
            # Auto mode: construct from configured host:port
            host = self.server_args.worker_bind_host
            port = self.server_args.worker_bind_port
            self.worker_address = f"tcp://{host}:{port}"
            logger.info(f"Using worker address from host:port config: {self.worker_address}")
        else:
            self.worker_address = f"tcp://127.0.0.1:{self._find_free_port()}"
            logger.info(f"Auto-assigned worker address: {self.worker_address}")

        # Packing parameters - prefer from ServerArguments if available
        if self.server_args:
            self.sample_packing_sequence_len = self.server_args.sample_packing_sequence_len
            self.enable_packing = self.server_args.enable_packing
            logger.info(
                f"Using packing config: seq_len={self.sample_packing_sequence_len}, enabled={self.enable_packing}"
            )

        # Output directory - prefer from ServerArguments if available
        if self.server_args:
            self.output_dir = self.server_args.output_dir
            logger.info(f"Using output_dir from config: {self.output_dir}")
        else:
            self.output_dir = "outputs"
            logger.info(f"Using default output_dir: {self.output_dir}")

        # Base model - prefer model_name from ServerArguments if available, otherwise use model_path
        if self.server_args:
            # Use model_name for validation if specified, otherwise fall back to model_path
            self.base_model = self.server_args.model_name or self.server_args.model_path
            logger.info(f"Using base_model from config: {self.base_model}")
        else:
            self.base_model = None
            logger.info("No base_model configured (will not validate create_model requests)")

        # Storage limit - prefer from ServerArguments if available
        if self.server_args:
            self.storage_limit = self.server_args.storage_limit
            logger.info(f"Using storage_limit from config: {self.storage_limit}")
        else:
            self.storage_limit = "10TB"
            logger.info(f"Using default storage_limit: {self.storage_limit}")

        # Idle session timeout - prefer from ServerArguments if available
        if self.server_args:
            self.idle_session_timeout = self.server_args.idle_session_timeout
            logger.info(f"Using idle_session_timeout from config: {self.idle_session_timeout}s")
        else:
            self.idle_session_timeout = 7200.0
            logger.info(f"Using default idle_session_timeout: {self.idle_session_timeout}s")

        # Processes and subprocesses
        self.worker_process: Optional[subprocess.Popen] = None  # torchrun subprocess
        self.engine_process: Optional[mp.Process] = None
        self.api_process: Optional[mp.Process] = None

        # Event to signal when engine is ready
        self.engine_ready_event: Optional[mp.Event] = None

        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        try:
            self.stop()
        except Exception:
            pass  # Best-effort cleanup in signal handler
        sys.exit(0)

    def _get_rank0_worker_address(self) -> str:
        """
        Resolve the address to connect to rank 0 worker.

        Priority:
        1. Explicit engine_connect_host from config (for manual multi-node setup)
        2. File-based discovery (for multi-node with shared filesystem)
        3. Localhost fallback (for single-node)

        For multi-node (nnodes > 1), waits for the address file to be created
        by the rank 0 worker.

        Returns:
            ZMQ address string (e.g., "tcp://192.168.1.100:5556")
        """
        # Priority 1: Explicit engine_connect_host from config
        if self.server_args and self.server_args.engine_connect_host:
            port = self.server_args.worker_bind_port
            address = f"tcp://{self.server_args.engine_connect_host}:{port}"
            logger.info(f"Using explicit engine_connect_host: {address}")
            return address

        # Priority 2: File-based discovery (for multi-node)
        if self.nnodes > 1:
            from xorl.server.utils.network import read_address_file

            logger.info(f"Multi-node setup (nnodes={self.nnodes}), waiting for rank 0 address file...")

            # Wait for address file with extended timeout for multi-node
            address = read_address_file(
                output_dir=self.output_dir,
                timeout=self.server_args.worker_connection_timeout if self.server_args else 120.0,
                poll_interval=2.0,
            )

            if address:
                logger.info(f"Discovered rank 0 address from file: {address}")
                return address
            else:
                logger.warning("Could not discover rank 0 address from file, falling back to config")

        # Priority 3: Use bind address from config (single-node or fallback)
        # For single-node, the bind address of 0.0.0.0 should be converted to localhost
        if self.server_args:
            bind_address = self.server_args.worker_bind_address
            if bind_address and bind_address != "auto":
                # Parse and convert 0.0.0.0 to 127.0.0.1 for localhost connections
                if "0.0.0.0" in bind_address:
                    address = bind_address.replace("0.0.0.0", "127.0.0.1")
                else:
                    address = bind_address
                logger.info(f"Using worker address (single-node): {address}")
                return address
            else:
                # Auto mode: construct from host:port
                host = self.server_args.worker_bind_host
                port = self.server_args.worker_bind_port
                connect_host = "127.0.0.1" if host == "0.0.0.0" else host
                address = f"tcp://{connect_host}:{port}"
                logger.info(f"Using worker address from host:port (single-node): {address}")
                return address

        # Final fallback: use already-resolved worker_address
        logger.info(f"Using pre-resolved worker address: {self.worker_address}")
        return self.worker_address

    def _poll_future(self, request_id: str, timeout: float = 300.0, poll_interval: float = 2.0):
        """Poll /api/v1/retrieve_future until the async operation completes.

        Returns the result dict on success, raises RuntimeError on failure.
        """
        retrieve_url = f"http://{self.api_host}:{self.api_port}/api/v1/retrieve_future"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.post(
                    retrieve_url,
                    json={"request_id": request_id},
                    timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"retrieve_future returned HTTP {resp.status_code}: {resp.text}")

                result = resp.json()
                # TryAgainResponse has a "type" field set to "try_again"
                if result.get("type") == "try_again":
                    time.sleep(poll_interval)
                    continue
                # RequestFailedResponse has an "error" field
                if "error" in result:
                    raise RuntimeError(f"Async operation failed: {result['error']}")
                # Success
                return result

            except requests.exceptions.RequestException as e:
                logger.warning(f"retrieve_future request failed: {e}")
                time.sleep(poll_interval)

        raise RuntimeError(f"Timed out waiting for request {request_id} after {timeout}s")

    def _save_initial_checkpoint(self, max_retries: int = 3, retry_delay: float = 5.0):
        """
        Save the initial checkpoint (000000) after all components are ready.

        This captures the initial model state before any training operations,
        allowing users to restore to the original state if needed.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("Saving initial checkpoint (000000)...")
        logger.info("=" * 70)

        api_url = f"http://{self.api_host}:{self.api_port}/api/v1/save_weights"

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    json={
                        "model_id": "default",
                        "path": "000000",
                    },
                    timeout=self.operation_timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    request_id = result.get("request_id")
                    if request_id:
                        # Two-phase: poll until save actually completes
                        self._poll_future(request_id)
                    logger.info("✓ Initial checkpoint saved (000000)")
                    return True
                elif response.status_code == 409:
                    # Checkpoint already exists - no need to retry
                    logger.info("✓ Initial checkpoint already exists (000000), skipping save")
                    return True
                else:
                    logger.warning(
                        f"Failed to save initial checkpoint (attempt {attempt + 1}/{max_retries}): "
                        f"HTTP {response.status_code} - {response.text}"
                    )

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to save initial checkpoint (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        logger.error("✗ Failed to save initial checkpoint after all retries")
        logger.warning("You can manually save the initial state by calling /api/v1/save_weights with path='000000'")
        return False

    @staticmethod
    def _find_free_port() -> int:
        """Find and return a free TCP port."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _launch_workers_with_torchrun(self):
        """Launch distributed workers using torchrun."""
        logger.info("=" * 70)
        logger.info("Launching Distributed Workers with torchrun")
        logger.info("=" * 70)
        logger.info(f"  Config:      {self.config_path}")
        logger.info(f"  Nodes:       {self.nnodes}")
        logger.info(f"  Procs/Node:  {self.nproc_per_node}")
        logger.info(f"  Master:      {self.master_addr}:{self.master_port}")
        logger.info(f"  Worker Addr: {self.worker_address}")
        logger.info("")

        # Clean up stale address file from previous runs
        stale_address_file = Path(self.output_dir) / ".rank0_address"
        if stale_address_file.exists():
            stale_address_file.unlink()
            logger.info(f"Removed stale address file: {stale_address_file}")

        # Build torchrun command — use the same Python environment as the launcher
        torchrun_bin = os.path.join(os.path.dirname(sys.executable), "torchrun")
        cmd = [
            torchrun_bin,
            f"--nnodes={self.nnodes}",
            f"--nproc-per-node={self.nproc_per_node}",
            f"--master-addr={self.master_addr}",
            f"--master-port={self.master_port}",
            "-m",
            "xorl.server.runner.runner_dispatcher",
            self.config_path,
            f"--worker.bind_address={self.worker_address}",
        ]

        # Pass server overrides to worker as CLI arguments
        # The worker's parse_server_args() will apply these on top of YAML config
        for key, value in self.server_overrides.items():
            if isinstance(value, bool):
                cmd.append(f"--{key}={str(value).lower()}")
            else:
                cmd.append(f"--{key}={value}")

        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("")
        logger.info("Worker output:")
        logger.info("-" * 70)

        # Launch as subprocess - don't capture output, let it print to console
        self.worker_process = subprocess.Popen(
            cmd,
            # Let worker output go directly to console for visibility
            stdout=None,
            stderr=None,
        )

        # Give workers time to spawn (brief wait for process initialization)
        logger.info("Waiting for workers to spawn (5 seconds)...")
        time.sleep(5)

        # Check if workers are still running
        if self.worker_process.poll() is not None:
            # Workers died
            logger.error("Workers process terminated unexpectedly!")
            logger.error(f"Exit code: {self.worker_process.returncode}")
            raise RuntimeError("Failed to start distributed workers")

        logger.info("-" * 70)
        logger.info("✓ Distributed Workers spawned successfully")
        logger.info("")
        logger.info("Note: Workers will load the model in the background.")
        logger.info("The Engine will wait for workers to be ready (up to 15 minutes).")

    def start(self):
        """Start all components."""
        logger.info("=" * 70)
        logger.info("XORL Training Server Launcher")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  API Server:             http://{self.api_host}:{self.api_port}")
        logger.info(f"  Engine Input:           {self.engine_input_addr}")
        logger.info(f"  Engine Output:          {self.engine_output_addr}")
        logger.info(f"  Worker Address:         {self.worker_address}")
        logger.info(f"  Max Running Requests:   {self.max_running_requests}")
        logger.info(f"  Max Pending Requests:   {self.max_pending_requests}")
        logger.info(f"  Log Level:              {self.log_level}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("")

        try:
            # Mode-specific startup
            if self.mode == "auto":
                # Auto-launch: Start workers, then engine, then API
                self._launch_workers_with_torchrun()
            else:
                # Connect mode: Assume workers are already running
                logger.info("Connect mode: assuming workers are already running")
                logger.info(f"Expecting rank 0 worker at: {self.worker_address}")
                logger.info("")

            # Resolve rank 0 worker address (important for multi-node)
            # For multi-node, this waits for the address file to be created by rank 0
            logger.info("Resolving rank 0 worker address...")
            self.worker_address = self._get_rank0_worker_address()
            logger.info(f"Resolved worker address: {self.worker_address}")

            # Start Engine Core (connects to worker)
            logger.info("=" * 70)
            logger.info("Starting Engine Core...")
            logger.info("=" * 70)
            logger.info(f"  Connecting to worker: {self.worker_address}")
            logger.info(f"  Engine input:  {self.engine_input_addr}")
            logger.info(f"  Engine output: {self.engine_output_addr}")
            logger.info(f"  Log level:     {self.log_level}")

            try:
                # Create event to signal when engine is ready
                self.engine_ready_event = mp.Event()

                self.engine_process = mp.Process(
                    target=run_orchestrator,
                    args=(
                        self.engine_input_addr,
                        self.engine_output_addr,
                        self.worker_address,
                        self.max_running_requests,
                        self.max_pending_requests,
                        self.operation_timeout,
                        self.log_level,
                        self.sample_packing_sequence_len,
                        self.enable_packing,
                        self.engine_ready_event,
                        self.output_dir,
                    ),
                    name="Orchestrator",
                )
                logger.info("  Process object created successfully")

                self.engine_process.start()
                logger.info("  Process.start() called")
                logger.info(f"  Engine process PID: {self.engine_process.pid}")
                logger.info(f"  Engine process alive: {self.engine_process.is_alive()}")
            except Exception as e:
                logger.error(f"  ✗ Failed to start engine process: {e}", exc_info=True)
                raise

            # Check if alive immediately
            time.sleep(0.5)
            if not self.engine_process.is_alive():
                exit_code = self.engine_process.exitcode
                logger.error(f"✗ Engine Core process died immediately (exit code: {exit_code})")
                logger.error("  This usually means:")
                logger.error("  - Import error or syntax error")
                logger.error("  - Worker not available yet")
                logger.error("  - Port already in use")
                raise RuntimeError(f"Engine Core process died (exit code: {exit_code})")

            logger.info("  Engine process started, waiting for worker connection...")
            time.sleep(5)  # Give more time for engine to connect to worker

            # Check if still alive after connection attempt
            if not self.engine_process.is_alive():
                exit_code = self.engine_process.exitcode
                logger.error(f"✗ Engine Core process died during startup (exit code: {exit_code})")
                logger.error("  Possible reasons:")
                logger.error(f"  - Could not connect to worker at {self.worker_address}")
                logger.error("  - Worker not ready or not listening")
                logger.error("  - Network/firewall issue")
                raise RuntimeError(f"Engine Core failed to connect to worker (exit code: {exit_code})")

            logger.info("✓ Engine Core process started, waiting for full initialization...")

            # Start API Server (connects to engine)
            logger.info("Starting API Server...")
            logger.info(f"  output_dir: {self.output_dir}")
            logger.info(f"  base_model: {self.base_model}")
            logger.info(f"  storage_limit: {self.storage_limit}")
            logger.info(f"  idle_session_timeout: {self.idle_session_timeout}s")
            self.api_process = mp.Process(
                target=run_api_server,
                args=(
                    self.api_host,
                    self.api_port,
                    self.engine_input_addr,
                    self.engine_output_addr,
                    self.log_level,
                    self.operation_timeout,
                    self.output_dir,
                    self.base_model,
                    self.storage_limit,
                    self.idle_session_timeout,
                    self.server_args.skip_initial_checkpoint,
                    self.server_args.sync_inference_method,
                ),
                name="APIServer",
            )
            self.api_process.start()
            time.sleep(2)  # Give API server time to start
            logger.info("✓ API Server started")

            # Wait for engine to signal it's fully ready (after worker connection and startup tests)
            logger.info("")
            logger.info("Waiting for Engine Core to complete initialization...")
            engine_ready_timeout = 1800.0  # 30 minutes timeout for engine initialization (large models need time)
            if self.engine_ready_event.wait(timeout=engine_ready_timeout):
                logger.info("✓ Engine Core fully initialized")
            else:
                logger.error(f"✗ Engine Core did not signal ready within {engine_ready_timeout}s")
                raise RuntimeError("Engine Core initialization timeout")

            # Check if engine process is still alive
            if not self.engine_process.is_alive():
                exit_code = self.engine_process.exitcode
                logger.error(f"✗ Engine Core process died during initialization (exit code: {exit_code})")
                raise RuntimeError(f"Engine Core failed during initialization (exit code: {exit_code})")

            # Save initial checkpoint (000) to capture the model state before any training
            if not self.server_args.skip_initial_checkpoint:
                self._save_initial_checkpoint()
            else:
                logger.info("Skipping initial checkpoint save (skip_initial_checkpoint=true)")

            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ All components started successfully!")
            logger.info("=" * 70)
            logger.info("")
            logger.info(f"API Documentation: http://{self.api_host}:{self.api_port}/docs")
            logger.info("")
            if self.mode == "auto":
                logger.info("Workers launched with torchrun (check logs above)")
            else:
                logger.info("Connected to external workers")
            logger.info("")
            logger.info("Press Ctrl+C to stop all components")
            logger.info("=" * 70)
            logger.info("")

            # Wait for processes
            self.wait()

        except Exception as e:
            logger.error(f"Error starting components: {e}", exc_info=True)
            self.stop()
            raise

    def wait(self):
        """Wait for all processes to finish. Exit cleanly if any process dies."""

        try:
            while True:
                # Check if worker process (torchrun) died
                if self.worker_process:
                    ret = self.worker_process.poll()
                    if ret is not None:
                        logger.error(f"Worker process exited with code {ret}")
                        self.stop()
                        sys.exit(ret or 1)

                # Check if engine process died
                if self.engine_process and not self.engine_process.is_alive():
                    code = self.engine_process.exitcode
                    logger.error(f"Engine process exited with code {code}")
                    self.stop()
                    sys.exit(code or 1)

                # Check if API process died
                if self.api_process and not self.api_process.is_alive():
                    code = self.api_process.exitcode
                    logger.error(f"API process exited with code {code}")
                    self.stop()
                    sys.exit(code or 1)

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()

    def stop(self):
        """Stop all components."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("Stopping all components...")
        logger.info("=" * 70)

        # Stop in reverse order
        # 1. API Server
        if self.api_process and self.api_process.is_alive():
            logger.info("Stopping API Server...")
            try:
                self.api_process.terminate()
                self.api_process.join(timeout=5)
                if self.api_process.is_alive():
                    logger.warning("API Server did not stop gracefully, killing...")
                    self.api_process.kill()
                    self.api_process.join()
            except (AttributeError, OSError):
                pass  # Process already exited
            logger.info("✓ API Server stopped")

        # 2. Engine Core
        if self.engine_process:
            try:
                alive = self.engine_process.is_alive()
            except (AssertionError, OSError):
                alive = False
            if alive:
                logger.info("Stopping Engine Core...")
                try:
                    self.engine_process.terminate()
                    self.engine_process.join(timeout=5)
                    try:
                        if self.engine_process.is_alive():
                            logger.warning("Engine Core did not stop gracefully, killing...")
                            self.engine_process.kill()
                            self.engine_process.join()
                    except (AssertionError, OSError):
                        pass
                except (AttributeError, OSError):
                    pass  # Process already exited
            logger.info("✓ Engine Core stopped")

        # 3. Workers (only in auto mode)
        if self.mode == "auto" and self.worker_process is not None:
            logger.info("Stopping Distributed Workers...")
            try:
                self.worker_process.terminate()
                self.worker_process.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                logger.warning("Workers did not stop gracefully, killing...")
                try:
                    self.worker_process.kill()
                    self.worker_process.wait()
                except OSError:
                    pass
            logger.info("✓ Distributed Workers stopped")

        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ All components stopped")
        logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================


def parse_server_overrides(argv: List[str]) -> Tuple[List[str], Dict[str, any]]:
    """
    Parse --server.* arguments from command line.

    Extracts arguments like --server.output_dir=/tmp/out or --server.lora_rank 64
    and returns them as a dict of overrides.

    Args:
        argv: Command line arguments (typically sys.argv[1:])

    Returns:
        Tuple of (remaining_args, server_overrides_dict)
    """
    remaining_args = []
    server_overrides = {}
    i = 0

    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--server."):
            # Extract key and value
            key_part = arg[9:]  # Remove "--server."

            if "=" in key_part:
                # Format: --server.key=value
                key, value = key_part.split("=", 1)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                # Format: --server.key value
                key = key_part
                value = argv[i + 1]
                i += 1
            else:
                # Boolean flag: --server.enable_lora (implies True)
                key = key_part
                value = "true"

            # Convert value to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                value = float(value) if "." in value else int(value)

            server_overrides[key] = value
        else:
            remaining_args.append(arg)
        i += 1

    return remaining_args, server_overrides


def main():
    """Main entry point."""
    # First, extract --server.* overrides before argparse
    remaining_args, server_overrides = parse_server_overrides(sys.argv[1:])

    if server_overrides:
        logger.info("Server config overrides from CLI:")
        for key, value in server_overrides.items():
            logger.info(f"  --server.{key} = {value}")

    parser = argparse.ArgumentParser(
        description="Launch XORL Training Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Auto-launch mode (world size automatically calculated from config)
  python -m xorl.server.launcher --mode auto --config examples/qwen3/sft.yaml

  # Override config values via CLI
  python -m xorl.server.launcher --mode auto --config server.yaml \\
    --server.output_dir /custom/output \\
    --server.lora_rank 64 \\
    --server.enable_lora true

  # Connect mode (workers launched separately)
  # Terminal 1: Launch workers manually
  torchrun --nnodes=1 --nproc-per-node=8 -m xorl.server.runner.runner_dispatcher \\
    examples/qwen3/sft.yaml --worker.bind_address tcp://127.0.0.1:5556

  # Terminal 2: Launch API server and engine
  python -m xorl.server.launcher --mode connect --worker-address tcp://127.0.0.1:5556

Server Config Overrides (--server.*):
  Any ServerArguments field can be overridden via --server.<field_name> <value>
  Examples:
    --server.output_dir /tmp/outputs
    --server.lora_rank 64
    --server.enable_lora true
    --server.ulysses_parallel_size 4
    --server.sample_packing_sequence_len 64000

Note:
  World size is ALWAYS calculated from the config file parallelism settings:
    world_size = dp_replicate_size * dp_shard_size * ulysses_size * ringattn_size
  And EP creates a separate 2D mesh where: world_size = ep_size * ep_fsdp_size;
  ep_fsdp_size will be automatically calculated through world_size / ep_size;
  Example config:
    train:
      data_parallel_shard_size: 2
      ulysses_parallel_size: 4
      # => world_size = 2 * 4 = 8 GPUs

  Set these values in your config file under the 'train' section to control world size.
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "connect"],
        default="auto",
        help="Launch mode: 'auto' (launch workers with torchrun) or 'connect' (connect to external workers)",
    )

    # Worker configuration
    parser.add_argument("--config", type=str, help="Path to training config YAML (required for auto mode)")
    parser.add_argument(
        "--worker-address", type=str, default=None, help="Worker ZMQ address (default: tcp://127.0.0.1:<auto-port>)"
    )

    # API Server configuration
    parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API server host (default: 0.0.0.0)")
    parser.add_argument("--api-port", type=int, default=None, help="API server port (default: auto-find free port)")

    # Engine configuration
    parser.add_argument(
        "--max-running-requests", type=int, default=2, help="Maximum concurrent running requests (default: 2)"
    )
    parser.add_argument(
        "--max-pending-requests", type=int, default=100, help="Maximum pending requests in queue (default: 100)"
    )
    parser.add_argument(
        "--operation-timeout",
        type=float,
        default=1800.0,
        help="Timeout for engine operations in seconds (default: 1800.0)",
    )
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # Distributed training (auto mode only)
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes (auto mode, default: 1)")
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="Master address for torch distributed (auto mode, default: 127.0.0.1)",
    )
    parser.add_argument(
        "--master-port", type=int, default=29500, help="Master port for torch distributed (auto mode, default: 29500)"
    )

    # Parse only the remaining args (after extracting --server.* overrides)
    args = parser.parse_args(remaining_args)

    # Create and start launcher
    launcher = Launcher(
        mode=args.mode,
        config_path=args.config,
        worker_address=args.worker_address,
        api_host=args.api_host,
        api_port=args.api_port,
        max_running_requests=args.max_running_requests,
        max_pending_requests=args.max_pending_requests,
        operation_timeout=args.operation_timeout,
        log_level=args.log_level,
        nnodes=args.nnodes,
        master_addr=args.master_addr,
        master_port=args.master_port,
        server_overrides=server_overrides,
    )

    launcher.start()


if __name__ == "__main__":
    main()
