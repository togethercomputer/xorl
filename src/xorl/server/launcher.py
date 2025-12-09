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
import asyncio
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

import uvicorn
import yaml

from xorl.server.api_server.api_server import APIServer
from xorl.server.engine.engine_core import EngineCore
from xorl.server.server_arguments import ServerArguments


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(name)s] %(asctime)s >> %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Port Finding Utilities
# ============================================================================

def find_free_port(start_port: int = 5000, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.

    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Free port number

    Raises:
        RuntimeError: If no free port found
    """
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


def find_free_ports(count: int, start_port: int = 5000) -> List[int]:
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

def run_engine_core(
    input_addr: str,
    output_addr: str,
    rank0_worker_address: str,
    max_running_requests: int = 2,
    max_pending_requests: int = 100,
    log_level: str = "INFO",
    inference_worker_urls: Optional[List[str]] = None,
):
    """
    Run the EngineCore in a separate process.

    Args:
        input_addr: Address for receiving requests from API server
        output_addr: Address for sending outputs to API server
        rank0_worker_address: Address of rank0 worker
        max_running_requests: Maximum concurrent running requests
        max_pending_requests: Maximum pending requests in queue
        log_level: Logging level
        inference_worker_urls: URLs of inference workers for automatic weight updates
    """
    # Setup logging first, outside try block
    import sys
    import os

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/engine_core.log"

    # Setup logging to both file and stdout
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(levelname)-8s][ENGINE] %(asctime)s >> %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger("EngineCore")
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
        logger.info("Initializing EngineCore...")
        engine = EngineCore(
            input_addr=input_addr,
            output_addr=output_addr,
            rank0_worker_address=rank0_worker_address,
            inference_worker_urls=inference_worker_urls,
            connection_timeout=300.0,  # Give worker time to load large models + compile Triton kernels
        )
        logger.info("EngineCore initialized successfully")

        logger.info("Starting EngineCore...")
        engine.start()
        logger.info("Engine Core started successfully")

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in engine core: {e}", exc_info=True)
    finally:
        if 'engine' in locals():
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
):
    """
    Run the API Server in a separate process.

    Args:
        host: Host to bind the API server
        port: Port to bind the API server
        engine_input_addr: Address to send requests to engine
        engine_output_addr: Address to receive outputs from engine
        log_level: Logging level
    """
    from contextlib import asynccontextmanager

    # Setup logging for this process
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(levelname)-8s][API] %(asctime)s >> %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger("APIServer")

    logger.info("=" * 70)
    logger.info("Starting API Server")
    logger.info("=" * 70)
    logger.info(f"API Server:     http://{host}:{port}")
    logger.info(f"Engine input:   {engine_input_addr}")
    logger.info(f"Engine output:  {engine_output_addr}")

    try:
        # Import the FastAPI app from api_server module
        from xorl.server.api_server.api_server import app

        # Update the global addresses before running
        import xorl.server.api_server.api_server as api_module

        # Override the lifespan to use our addresses
        @asynccontextmanager
        async def custom_lifespan(app):
            """Custom lifecycle with configurable addresses."""
            logger.info("Starting APIServer with custom addresses...")
            api_module.api_server = APIServer(
                engine_input_addr=engine_input_addr,
                engine_output_addr=engine_output_addr,
            )
            await api_module.api_server.start()
            yield
            logger.info("Shutting down APIServer...")
            if api_module.api_server:
                await api_module.api_server.stop()

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

def load_server_arguments(config_path: str) -> ServerArguments:
    """
    Load ServerArguments from a YAML configuration file.

    Supports both flat config (ServerArguments style) and nested config
    (Arguments style with model/train/worker sections).

    Args:
        config_path: Path to server config YAML

    Returns:
        ServerArguments instance with all fields populated
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty config file: {config_path}")

    from dataclasses import fields
    valid_fields = {f.name for f in fields(ServerArguments)}

    # Check if this is a nested config (has model/train/worker sections)
    if 'model' in config and isinstance(config['model'], dict):
        # Nested config - flatten it
        flat_config = {}

        # Model section
        model_config = config.get('model', {})
        flat_config['model_path'] = model_config.get('model_path')
        flat_config['config_path'] = model_config.get('config_path')
        flat_config['tokenizer_path'] = model_config.get('tokenizer_path')
        flat_config['attn_implementation'] = model_config.get('attn_implementation', 'flash_attention_2')
        flat_config['moe_implementation'] = model_config.get('moe_implementation')
        flat_config['force_use_huggingface'] = model_config.get('force_use_huggingface', False)
        flat_config['use_liger'] = model_config.get('use_liger', True)

        # Train section
        train_config = config.get('train', {})
        flat_config['data_parallel_mode'] = train_config.get('data_parallel_mode', 'fsdp2')
        flat_config['data_parallel_shard_size'] = train_config.get('data_parallel_shard_size', 1)
        flat_config['data_parallel_replicate_size'] = train_config.get('data_parallel_replicate_size', 1)
        flat_config['ulysses_parallel_size'] = train_config.get('ulysses_parallel_size', 1)
        flat_config['expert_parallel_size'] = train_config.get('expert_parallel_size', 1)
        flat_config['enable_mixed_precision'] = train_config.get('enable_mixed_precision', True)
        flat_config['enable_gradient_checkpointing'] = train_config.get('enable_gradient_checkpointing', True)
        flat_config['enable_full_shard'] = train_config.get('enable_full_shard', True)
        flat_config['enable_fsdp_offload'] = train_config.get('enable_fsdp_offload', False)
        flat_config['enable_activation_offload'] = train_config.get('enable_activation_offload', False)
        flat_config['init_device'] = train_config.get('init_device', 'meta')
        flat_config['load_checkpoint_path'] = train_config.get('load_checkpoint_path', '')
        flat_config['ckpt_manager'] = train_config.get('ckpt_manager', 'dcp')
        flat_config['log_level'] = train_config.get('log_level', 'INFO')

        # Worker section
        worker_config = config.get('worker', {})
        flat_config['worker_bind_address'] = worker_config.get('bind_address', 'tcp://127.0.0.1:5556')
        flat_config['worker_connection_timeout'] = worker_config.get('connection_timeout', 60.0)
        flat_config['worker_max_retries'] = worker_config.get('max_retries', 3)

        filtered_config = {k: v for k, v in flat_config.items() if k in valid_fields and v is not None}
    else:
        # Flat config (ServerArguments style)
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}

        # Handle None values for Optional fields
        for key, value in list(filtered_config.items()):
            if value is None:
                if key in ['config_path', 'tokenizer_path']:
                    del filtered_config[key]

    logger.info(f"Loaded ServerArguments from: {config_path}")
    logger.info(f"  model_path: {filtered_config.get('model_path', 'N/A')}")

    return ServerArguments(**filtered_config)


def calculate_world_size_from_config(config_path: str) -> int:
    """
    Calculate the required world size from parallelism configuration.

    Args:
        config_path: Path to server config YAML

    Returns:
        Required world size (nproc_per_node)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Support both flat config (ServerArguments style) and nested config (train: section)
    # Check if this is a nested config with 'train' section
    if 'train' in config:
        train_config = config.get('train', {})
    else:
        # Flat config (ServerArguments style)
        train_config = config

    # Get parallelism sizes (with defaults matching model_runner.py)
    pp_size = train_config.get('pipeline_parallel_size', 1)
    tp_size = train_config.get('tensor_parallel_size', 1)
    ep_size = train_config.get('expert_parallel_size', 1)
    cp_size = train_config.get('context_parallel_size', 1)
    ulysses_size = train_config.get('ulysses_parallel_size', 1)

    # Data parallel sizes
    dp_replicate_size = train_config.get('data_parallel_replicate_size', 1)
    dp_shard_size = train_config.get('data_parallel_shard_size', 1)

    # Calculate world size
    # Formula: world_size = pp_size * dp_replicate_size * dp_shard_size * ulysses_size * cp_size * tp_size
    world_size = pp_size * dp_replicate_size * dp_shard_size * ulysses_size * cp_size * tp_size * ep_size

    logger.info(f"Calculated world size from config:")
    logger.info(f"  pipeline_parallel_size:      {pp_size}")
    logger.info(f"  tensor_parallel_size:         {tp_size}")
    logger.info(f"  expert_parallel_size:         {ep_size}")
    logger.info(f"  context_parallel_size:        {cp_size}")
    logger.info(f"  ulysses_parallel_size:        {ulysses_size}")
    logger.info(f"  data_parallel_replicate_size: {dp_replicate_size}")
    logger.info(f"  data_parallel_shard_size:     {dp_shard_size}")
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
        log_level: str = "INFO",
        inference_worker_urls: Optional[List[str]] = None,
        # Auto-launch mode parameters
        nnodes: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
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
            inference_worker_urls: URLs of inference workers for automatic weight updates
            nnodes: Number of nodes for distributed training (auto mode)
            master_addr: Master address for torch distributed (auto mode)
            master_port: Master port for torch distributed (auto mode)
        """
        self.mode = mode
        self.config_path = config_path
        self.api_host = api_host
        self.log_level = log_level
        self.max_running_requests = max_running_requests
        self.max_pending_requests = max_pending_requests
        self.inference_worker_urls = inference_worker_urls or []

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
                self.server_args = load_server_arguments(config_path)
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
                self.nproc_per_node = self.server_args.get_world_size()
            else:
                self.nproc_per_node = calculate_world_size_from_config(config_path)
            logger.info(f"World size (nproc_per_node) = {self.nproc_per_node}")
        else:
            self.nproc_per_node = 1  # Not used in connect mode

        # Find free ports
        logger.info("Finding free ports...")

        # API port is user-configurable, others are auto-found
        if api_port:
            self.api_port = api_port
            # Find 3 more ports for engine and worker
            ports = find_free_ports(3, start_port=5000)
            self.engine_input_port = ports[0]
            self.engine_output_port = ports[1]
            self.worker_port = ports[2] if not worker_address else None
        else:
            # Find all 4 ports
            ports = find_free_ports(4, start_port=5000)
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
        elif self.server_args:
            self.worker_address = self.server_args.worker_bind_address
            logger.info(f"Using worker address from config: {self.worker_address}")
        else:
            self.worker_address = f"tcp://127.0.0.1:{self.worker_port}"

        # Processes and subprocesses
        self.worker_process: Optional[subprocess.Popen] = None  # torchrun subprocess
        self.engine_process: Optional[mp.Process] = None
        self.api_process: Optional[mp.Process] = None

        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

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

        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nnodes={self.nnodes}",
            f"--nproc-per-node={self.nproc_per_node}",
            f"--master-addr={self.master_addr}",
            f"--master-port={self.master_port}",
            "-m", "xorl.server.worker.distributed_model_worker",
            self.config_path,
            f"--worker.bind_address={self.worker_address}",
        ]

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

        # Give workers time to initialize
        logger.info("Waiting for workers to initialize (5 seconds)...")
        time.sleep(5)

        # Check if workers are still running
        if self.worker_process.poll() is not None:
            # Workers died
            logger.error("Workers process terminated unexpectedly!")
            logger.error(f"Exit code: {self.worker_process.returncode}")
            raise RuntimeError("Failed to start distributed workers")

        logger.info("-" * 70)
        logger.info("✓ Distributed Workers started")
        logger.info("")
        logger.info("Waiting for workers to initialize (model loading takes ~30 seconds)...")
        logger.info("This is normal - the worker needs to load the model before accepting connections")
        time.sleep(30)  # Give workers time to load model and start listening
        logger.info("✓ Worker initialization wait complete")

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

            # Start Engine Core (connects to worker)
            logger.info("=" * 70)
            logger.info("Starting Engine Core...")
            logger.info("=" * 70)
            logger.info(f"  Connecting to worker: {self.worker_address}")
            logger.info(f"  Engine input:  {self.engine_input_addr}")
            logger.info(f"  Engine output: {self.engine_output_addr}")
            logger.info(f"  Log level:     {self.log_level}")

            try:
                self.engine_process = mp.Process(
                    target=run_engine_core,
                    args=(
                        self.engine_input_addr,
                        self.engine_output_addr,
                        self.worker_address,
                        self.max_running_requests,
                        self.max_pending_requests,
                        self.log_level,
                        self.inference_worker_urls,
                    ),
                    name="EngineCore",
                )
                logger.info("  Process object created successfully")

                self.engine_process.start()
                logger.info(f"  Process.start() called")
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

            logger.info("✓ Engine Core started and connected to worker")

            # Start API Server (connects to engine)
            logger.info("Starting API Server...")
            self.api_process = mp.Process(
                target=run_api_server,
                args=(
                    self.api_host,
                    self.api_port,
                    self.engine_input_addr,
                    self.engine_output_addr,
                    self.log_level,
                ),
                name="APIServer",
            )
            self.api_process.start()
            time.sleep(2)  # Give API server time to start
            logger.info("✓ API Server started")

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
        """Wait for all processes to finish."""
        try:
            # Wait for all processes
            if self.api_process:
                self.api_process.join()
            if self.engine_process:
                self.engine_process.join()
            if self.worker_process:
                self.worker_process.wait()
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
            self.api_process.terminate()
            self.api_process.join(timeout=5)
            if self.api_process.is_alive():
                logger.warning("API Server did not stop gracefully, killing...")
                self.api_process.kill()
                self.api_process.join()
            logger.info("✓ API Server stopped")

        # 2. Engine Core
        if self.engine_process and self.engine_process.is_alive():
            logger.info("Stopping Engine Core...")
            self.engine_process.terminate()
            self.engine_process.join(timeout=5)
            if self.engine_process.is_alive():
                logger.warning("Engine Core did not stop gracefully, killing...")
                self.engine_process.kill()
                self.engine_process.join()
            logger.info("✓ Engine Core stopped")

        # 3. Workers (only in auto mode)
        if self.mode == "auto" and self.worker_process:
            logger.info("Stopping Distributed Workers...")
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Workers did not stop gracefully, killing...")
                self.worker_process.kill()
                self.worker_process.wait()
            logger.info("✓ Distributed Workers stopped")

        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ All components stopped")
        logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch XORL Training Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Auto-launch mode (world size automatically calculated from config)
  python -m xorl.server.launcher --mode auto --config examples/qwen3/sft.yaml

  # Connect mode (workers launched separately)
  # Terminal 1: Launch workers manually
  torchrun --nnodes=1 --nproc-per-node=8 -m xorl.server.worker.distributed_model_worker \\
    examples/qwen3/sft.yaml --worker.bind_address tcp://127.0.0.1:5556

  # Terminal 2: Launch API server and engine
  python -m xorl.server.launcher --mode connect --worker-address tcp://127.0.0.1:5556

Note:
  World size is ALWAYS calculated from the config file parallelism settings:
    world_size = pp_size * dp_replicate_size * dp_shard_size * ulysses_size * cp_size * tp_size * ep_size

  Example config:
    train:
      data_parallel_shard_size: 2
      ulysses_parallel_size: 4
      # => world_size = 2 * 4 = 8 GPUs

  Set these values in your config file under the 'train' section to control world size.
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "connect"],
        default="auto",
        help="Launch mode: 'auto' (launch workers with torchrun) or 'connect' (connect to external workers)"
    )

    # Worker configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training config YAML (required for auto mode)"
    )
    parser.add_argument(
        "--worker-address",
        type=str,
        default=None,
        help="Worker ZMQ address (default: tcp://127.0.0.1:<auto-port>)"
    )

    # API Server configuration
    parser.add_argument(
        "--api-host",
        type=str,
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=None,
        help="API server port (default: auto-find free port)"
    )

    # Engine configuration
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=2,
        help="Maximum concurrent running requests (default: 2)"
    )
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=100,
        help="Maximum pending requests in queue (default: 100)"
    )
    parser.add_argument(
        "--inference-worker-urls",
        type=str,
        nargs="+",
        default=None,
        help="URLs of inference workers for automatic weight updates (e.g., http://host1:8000 http://host2:8000)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    # Distributed training (auto mode only)
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes (auto mode, default: 1)"
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="Master address for torch distributed (auto mode, default: 127.0.0.1)"
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for torch distributed (auto mode, default: 29500)"
    )

    args = parser.parse_args()

    # Create and start launcher
    launcher = Launcher(
        mode=args.mode,
        config_path=args.config,
        worker_address=args.worker_address,
        api_host=args.api_host,
        api_port=args.api_port,
        max_running_requests=args.max_running_requests,
        max_pending_requests=args.max_pending_requests,
        log_level=args.log_level,
        inference_worker_urls=args.inference_worker_urls,
        nnodes=args.nnodes,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )

    launcher.start()


if __name__ == "__main__":
    main()
