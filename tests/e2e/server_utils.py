"""Shared server test utilities: config generation, ServerProcess, training helpers."""

import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from typing import List, Optional

import pytest
import requests
import yaml

def generate_server_config(
    model_dir: str,
    output_dir: str,
    *,
    num_gpus: int = 1,
    attn_implementation: str = "flash_attention_3",
    moe_implementation: Optional[str] = None,
    # Parallelism
    dp_shard_size: int = -1,
    dp_replicate_size: int = 1,
    pp_size: int = 1,
    pp_schedule: str = "1F1B",
    ulysses_size: int = 1,
    ep_size: int = 1,
    # Memory & performance
    enable_gradient_checkpointing: bool = True,
    sample_packing_sequence_len: int = 256,
    # LoRA
    enable_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 8,
    lora_target_modules: Optional[List[str]] = None,
    # QLoRA
    enable_qlora: bool = False,
    quant_format: str = "nvfp4",
    quant_group_size: Optional[int] = None,
    merge_lora_interval: int = 0,
    merge_qkv: bool = True,
    # MoE training
    freeze_router: bool = True,
    # Other
    skip_initial_checkpoint: bool = True,
    extra_config: Optional[dict] = None,
) -> str:
    """Generate a flat server YAML config (ServerArguments format).

    Follows the same format as examples/server/ configs.
    Auto-calculates dp_shard_size from num_gpus and other parallelism dims.
    """
    if dp_shard_size == -1:
        non_dp = pp_size * ulysses_size
        total_dp = max(1, num_gpus // non_dp)
        dp_shard_size = total_dp

    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    # Build config matching examples/server/ flat format
    config = {
        # Model
        "model_path": model_dir,
        "tokenizer_path": model_dir,
        "attn_implementation": attn_implementation,
        # Parallelism
        "data_parallel_mode": "fsdp2",
        "data_parallel_shard_size": dp_shard_size,
        "data_parallel_replicate_size": dp_replicate_size,
        "pipeline_parallel_size": pp_size,
        "pipeline_parallel_schedule": pp_schedule,
        "ulysses_parallel_size": ulysses_size,
        "expert_parallel_size": ep_size,
        # Model options
        "merge_qkv": merge_qkv,
        # Memory & performance
        "enable_mixed_precision": True,
        "enable_gradient_checkpointing": enable_gradient_checkpointing,
        "enable_full_shard": True,
        "init_device": "meta",
        # Checkpointing
        "output_dir": output_dir,
        "ckpt_manager": "dcp",
        "skip_initial_checkpoint": skip_initial_checkpoint,
        # Worker
        "worker_connection_timeout": 120.0,
        # Data
        "sample_packing_sequence_len": sample_packing_sequence_len,
        "enable_packing": True,
        # Logging
        "log_level": "INFO",
    }

    if moe_implementation is not None:
        config["moe_implementation"] = moe_implementation

    # Use a random worker bind port to avoid conflicts between tests
    config["worker_bind_port"] = _get_free_port()

    if enable_lora:
        config["enable_lora"] = True
        config["lora_rank"] = lora_rank
        config["lora_alpha"] = lora_alpha
        config["lora_target_modules"] = lora_target_modules

    if enable_qlora:
        config["enable_qlora"] = True
        config["quant_format"] = quant_format
        group_size = quant_group_size or (16 if quant_format == "nvfp4" else 128)
        config["quant_group_size"] = group_size
        config["merge_lora_interval"] = merge_lora_interval

    if ep_size > 1 or moe_implementation is not None:
        config["freeze_router"] = freeze_router

    if extra_config:
        config.update(extra_config)

    yaml_path = os.path.join(output_dir, "server_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return yaml_path


# ---------------------------------------------------------------------------
# Server process management
# ---------------------------------------------------------------------------

def _get_free_port() -> int:
    """Find a free port that is not in TIME_WAIT state."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def _port_is_free(port: int) -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def _wait_for_server(url: str, timeout: float = 120.0, poll_interval: float = 2.0) -> bool:
    """Wait for the server to become healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("engine_running", False):
                    return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


_PORT_RETRY_ATTEMPTS = 3


class ServerProcess:
    """Manages a xorl training server subprocess."""

    def __init__(self, config_path: str, num_gpus: int, api_port: int, output_dir: str):
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.api_port = api_port
        self.output_dir = output_dir
        self.base_url = f"http://localhost:{api_port}"
        self.process: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 180.0) -> bool:
        """Start the server and wait for it to be healthy.

        Retries with a new port if the api port is already in use.
        """
        for attempt in range(_PORT_RETRY_ATTEMPTS):
            # Check port availability before launching
            if not _port_is_free(self.api_port):
                self.api_port = _get_free_port()
                self.base_url = f"http://localhost:{self.api_port}"

            master_port = _get_free_port()

            cmd = [
                sys.executable, "-m", "xorl.server.launcher",
                "--mode", "auto",
                "--config", self.config_path,
                "--api-port", str(self.api_port),
                "--master-port", str(master_port),
            ]

            env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(self.num_gpus))

            log_path = os.path.join(self.output_dir, "server.log")
            log_file = open(log_path, "w")

            self.process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
            )
            self._log_file = log_file

            # Give the process a moment to fail on port binding
            time.sleep(3)
            if self.process.poll() is not None:
                # Process exited early — likely port conflict
                log_file.close()
                log_content = self.get_log()
                if "address already in use" in log_content.lower() and attempt < _PORT_RETRY_ATTEMPTS - 1:
                    self.process = None
                    self.api_port = _get_free_port()
                    self.base_url = f"http://localhost:{self.api_port}"
                    continue
                # Non-port error or last attempt — fall through to wait
                return False

            return _wait_for_server(self.base_url, timeout=timeout)

        return False

    def stop(self):
        """Stop the server gracefully."""
        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=10)
                except Exception:
                    pass
            finally:
                self.process = None

        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()

    def get_log(self) -> str:
        """Read the server log."""
        log_path = os.path.join(self.output_dir, "server.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                return f.read()
        return ""


# ---------------------------------------------------------------------------
# Random data generation
# ---------------------------------------------------------------------------

def generate_random_sft_data(
    num_samples: int,
    seq_len: int = 64,
    vocab_size: int = 32000,
    seed: int = 42,
) -> list:
    """Generate random SFT training data as xorl_client Datum objects.

    Returns list of Datum objects ready for the training client.
    Uses causallm_loss format: input_ids + labels (shifted tokens).
    """
    import xorl_client

    rng = random.Random(seed)
    datums = []
    for _ in range(num_samples):
        tokens = [rng.randint(1, vocab_size - 1) for _ in range(seq_len)]
        labels = tokens[:]

        datum = xorl_client.Datum(
            model_input=xorl_client.ModelInput.from_ints(tokens),
            loss_fn_inputs={"labels": labels},
        )
        datums.append(datum)
    return datums


def extract_loss(fwd_bwd_result) -> float:
    """Extract scalar loss from ForwardBackwardOutput, handling both formats."""
    if not fwd_bwd_result.loss_fn_outputs:
        return 0.0

    loss_output = fwd_bwd_result.loss_fn_outputs[0]

    # Try scalar loss first
    if "loss" in loss_output:
        loss_val = loss_output["loss"]
        if loss_val is None:
            pass
        elif hasattr(loss_val, "data"):
            return float(loss_val.data[0])
        elif isinstance(loss_val, dict) and "data" in loss_val:
            return float(loss_val["data"][0])
        else:
            return float(loss_val)

    # Fall back to mean elementwise_loss
    if "elementwise_loss" in loss_output:
        el = loss_output["elementwise_loss"]
        if hasattr(el, "data"):
            data = el.data
        elif isinstance(el, dict) and "data" in el:
            data = el["data"]
        else:
            return 0.0
        return sum(data) / len(data) if data else 0.0

    # Check metrics
    metrics = getattr(fwd_bwd_result, "metrics", {})
    if "loss" in metrics:
        return float(metrics["loss"])

    return 0.0


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_sft_steps(training_client, data, num_steps=5, lr=1e-3) -> list:
    """Run SFT training steps and return loss history."""
    import xorl_client

    adam_params = xorl_client.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    losses = []

    for step in range(num_steps):
        fwd_bwd = training_client.forward_backward(data, loss_fn="causallm_loss")
        optim = training_client.optim_step(adam_params)
        result = fwd_bwd.result()
        optim.result()

        loss_val = extract_loss(result)
        losses.append(loss_val)

    return losses


def assert_loss_decreases(losses, msg=""):
    """Assert training produced valid results and loss decreased."""
    assert len(losses) > 0, "No losses recorded"
    assert all(not math.isnan(l) for l in losses), f"NaN in losses: {losses}"
    assert all(l > 0 for l in losses), f"Non-positive loss: {losses}"
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        f"{': ' + msg if msg else ''}\n"
        f"All losses: {[f'{l:.4f}' for l in losses]}"
    )


def _start_server_or_fail(server, timeout=180.0):
    """Start server and fail test with log tail if unhealthy."""
    healthy = server.start(timeout=timeout)
    if not healthy:
        log_tail = "\n".join(server.get_log().splitlines()[-50:])
        pytest.fail(
            f"Server failed to become healthy within {timeout}s.\n"
            f"--- Log (last 50 lines) ---\n{log_tail}"
        )


def _create_lora_client(base_url, model_dir, model_id="test", rank=8):
    """Create a xorl_client LoRA training client."""
    import xorl_client

    service_client = xorl_client.ServiceClient(base_url=base_url)
    training_client = service_client.create_lora_training_client(
        base_model=model_dir,
        rank=rank,
        model_id=model_id,
    )
    return service_client, training_client


def _create_full_weight_client(base_url, model_dir):
    """Create a xorl_client TrainingClient for full-weight training (no LoRA).

    The server auto-registers model_id="default" on startup, so no
    create_model call is needed for full-weight training.
    """
    import xorl_client

    service_client = xorl_client.ServiceClient(base_url=base_url)
    training_client = xorl_client.TrainingClient(
        holder=service_client.holder,
        model_id="default",
        base_model=model_dir,
    )
    return service_client, training_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

