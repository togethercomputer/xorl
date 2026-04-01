"""Shared helpers for e2e tests.

Importable as a regular module (no package __init__.py required) via:
    from e2e_utils import run_training, generate_training_config, ...

All test files should use conftest.py fixtures for tmp dirs, and import
functions from this module for everything else.
"""

import importlib
import json
import math
import os
import random
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Real model IDs (downloaded/cached via HF hub)
# ---------------------------------------------------------------------------

QWEN3_8B_ID = "Qwen/Qwen3-8B"


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def skip_if_gpu_count_less_than(n: int):
    return pytest.mark.skipif(
        gpu_count() < n,
        reason=f"Requires {n} GPUs, found {gpu_count()}",
    )


def _has_flash_attn() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except ImportError:
        return False


def _has_quack() -> bool:
    try:
        importlib.import_module("quack")
        return True
    except ImportError:
        return False


skip_if_no_flash_attn = pytest.mark.skipif(not _has_flash_attn(), reason="flash_attn not installed")

skip_if_no_quack = pytest.mark.skipif(not _has_quack(), reason="quack not installed")


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Captures the outcome of a training subprocess."""

    exit_code: int
    stdout: str
    stderr: str
    output_dir: str
    metrics: Optional[Dict[str, Any]] = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def final_loss(self) -> Optional[float]:
        if self.metrics:
            return self.metrics.get("final_loss")
        return None

    @property
    def loss_history(self) -> Optional[List[float]]:
        if self.metrics:
            return self.metrics.get("loss_history")
        return None

    @property
    def global_step(self) -> Optional[int]:
        if self.metrics:
            return self.metrics.get("global_step")
        return None

    @property
    def final_grad_norm(self) -> Optional[float]:
        if self.metrics:
            return self.metrics.get("final_grad_norm")
        return None

    @property
    def total_train_steps(self) -> Optional[int]:
        if self.metrics:
            return self.metrics.get("total_train_steps")
        return None

    def assert_loss_converged(
        self,
        max_final_loss: float = 8.0,
        min_drop_ratio: float = 0.3,
    ):
        """Assert training loss converged within acceptable range."""
        assert self.loss_history is not None and len(self.loss_history) >= 2, (
            "Need at least 2 loss values to check convergence"
        )
        first, last = self.loss_history[0], self.loss_history[-1]
        assert not math.isnan(first) and not math.isnan(last), f"NaN in loss history: first={first}, last={last}"
        assert last < max_final_loss, f"Final loss {last:.4f} >= {max_final_loss}"
        drop = (first - last) / first
        assert drop >= min_drop_ratio, (
            f"Loss drop {drop:.2%} < required {min_drop_ratio:.0%} (first={first:.4f}, last={last:.4f})"
        )

    def assert_success(self, msg: str = ""):
        """Assert training completed successfully with useful diagnostics."""
        if not self.success:
            stderr_tail = "\n".join(self.stderr.splitlines()[-50:])
            stdout_tail = "\n".join(self.stdout.splitlines()[-80:])
            raise AssertionError(
                f"Training failed (exit_code={self.exit_code}){': ' + msg if msg else ''}\n"
                f"--- stdout (last 80 lines) ---\n{stdout_tail}\n"
                f"--- stderr (last 50 lines) ---\n{stderr_tail}"
            )
        assert self.metrics is not None, f"Training exited 0 but no training_metrics.json found in {self.output_dir}"


# ---------------------------------------------------------------------------
# Tiny model configs (no network access needed)
# ---------------------------------------------------------------------------

TINY_QWEN3_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_sliding_window": False,
    "pad_token_id": 0,
}

TINY_QWEN3_MOE_CONFIG = {
    "model_type": "qwen3_moe",
    "architectures": ["Qwen3MoeForCausalLM"],
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "decoder_sparse_step": 1,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "num_experts": 4,
    "norm_topk_prob": False,
    "pad_token_id": 0,
}

SMALL_QWEN3_CONFIG = {
    **TINY_QWEN3_CONFIG,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
}

# MoE config with larger moe_intermediate_size (>= 64) for NF4 group_size compatibility
SMALL_QWEN3_MOE_CONFIG = {
    **TINY_QWEN3_MOE_CONFIG,
    "moe_intermediate_size": 64,
}


def create_tiny_model_dir(
    base_dir: str,
    model_type: str = "dense",
    save_weights: bool = False,
) -> str:
    """Create a temp directory with model config + tokenizer files."""
    configs = {
        "dense": TINY_QWEN3_CONFIG,
        "dense_large": SMALL_QWEN3_CONFIG,
        "moe": TINY_QWEN3_MOE_CONFIG,
        "moe_large": SMALL_QWEN3_MOE_CONFIG,
    }
    model_config = configs[model_type]
    model_dir = os.path.join(base_dir, f"tiny_{model_type}_model")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    _create_tokenizer_files(model_dir, model_config["vocab_size"])

    if save_weights:
        _save_random_weights(model_dir, model_config)

    return model_dir


def _save_random_weights(model_dir: str, model_config: dict):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_dir, safe_serialization=True)


def _create_tokenizer_files(model_dir: str, vocab_size: int = 1024):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {f"t{i}": i for i in range(vocab_size)}
    vocab["[UNK]"] = 0
    vocab["[PAD]"] = vocab_size - 1 if vocab_size > 1 else 0

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(os.path.join(model_dir, "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "qwen3",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)


# ---------------------------------------------------------------------------
# YAML config generation
# ---------------------------------------------------------------------------


def generate_training_config(
    model_dir: str,
    output_dir: str,
    *,
    model_path: Optional[str] = None,
    attn_implementation: str = "flash_attention_3",
    moe_implementation: Optional[str] = None,
    merge_qkv: bool = True,
    seq_len: int = 128,
    packing_seq_len: int = 256,
    num_gpus: int = 1,
    max_steps: int = 3,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    optimizer: str = "adamw",
    lr: float = 1e-4,
    lr_warmup_ratio: float = 0.0,
    lr_decay_style: str = "constant",
    dp_shard_size: int = -1,
    dp_replicate_size: int = 1,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
    pp_schedule: str = "1F1B",
    ulysses_size: int = 1,
    enable_compile: bool = False,
    enable_gradient_checkpointing: bool = True,
    save_steps: int = 0,
    save_hf_weights: bool = False,
    enable_lora: bool = False,
    enable_qlora: bool = False,
    quant_format: str = "nvfp4",
    quant_group_size: Optional[int] = None,
    merge_lora_interval: int = 0,
    lora_rank: int = 8,
    lora_alpha: int = 8,
    lora_target_modules: Optional[List[str]] = None,
    extra_train: Optional[Dict[str, Any]] = None,
    extra_lora: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a YAML training config and return its path."""
    if dp_shard_size == -1:
        non_dp = tp_size * pp_size * ulysses_size
        total_dp = max(1, num_gpus // non_dp)
        dp_shard_size = total_dp

    model_section = {
        "config_path": model_dir,
        "tokenizer_path": model_dir,
        "attn_implementation": attn_implementation,
        "merge_qkv": merge_qkv,
    }
    if model_path is not None:
        model_section["model_path"] = model_path

    config = {
        "model": model_section,
        "data": {
            "datasets": [{"path": "dummy", "type": "tokenized", "max_seq_len": seq_len}],
            "select_columns": ["input_ids", "labels"],
            "sample_packing_method": "sequential",
            "sample_packing_sequence_len": packing_seq_len,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
        },
        "train": {
            "output_dir": output_dir,
            "data_parallel_mode": "fsdp2",
            "data_parallel_shard_size": dp_shard_size,
            "data_parallel_replicate_size": dp_replicate_size,
            "tensor_parallel_size": tp_size,
            "expert_parallel_size": ep_size,
            "pipeline_parallel_size": pp_size,
            "pipeline_parallel_schedule": pp_schedule,
            "ulysses_parallel_size": ulysses_size,
            "ringattn_parallel_size": 1,
            "num_train_epochs": 1,
            "max_steps": max_steps,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": optimizer,
            "lr": lr,
            "lr_warmup_ratio": lr_warmup_ratio,
            "lr_decay_style": lr_decay_style,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "enable_mixed_precision": True,
            "enable_gradient_checkpointing": enable_gradient_checkpointing,
            "enable_full_shard": True,
            "init_device": "meta",
            "enable_compile": enable_compile,
            "ckpt_manager": "dcp",
            "save_steps": save_steps,
            "save_hf_weights": save_hf_weights,
            "log_format": "structured",
            "use_wandb": False,
            "seed": 42,
        },
    }

    if moe_implementation:
        config["model"]["moe_implementation"] = moe_implementation

    if extra_train:
        config["train"].update(extra_train)

    if enable_lora or enable_qlora:
        config["lora"] = {
            "enable_lora": True,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_target_modules": lora_target_modules
            or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }
        if enable_qlora:
            group_size = quant_group_size or (16 if quant_format == "nvfp4" else 64 if quant_format == "nf4" else 128)
            config["lora"]["enable_qlora"] = True
            config["lora"]["quant_format"] = quant_format
            config["lora"]["quant_group_size"] = group_size
            config["lora"]["merge_lora_interval"] = merge_lora_interval
        if extra_lora:
            config["lora"].update(extra_lora)

    yaml_path = os.path.join(output_dir, "e2e_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return yaml_path


# ---------------------------------------------------------------------------
# Training launcher
# ---------------------------------------------------------------------------


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_training(
    config_path: str,
    num_gpus: int = 1,
    timeout: int = 300,
    extra_env: Optional[Dict[str, str]] = None,
) -> TrainingResult:
    """Launch training via torchrun subprocess and collect results."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(num_gpus),
        "--master_port",
        str(_get_free_port()),
        "-m",
        "xorl.cli.train",
        config_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:

        def _to_str(v, fallback):
            if isinstance(v, bytes):
                return v.decode(errors="replace")
            return v or fallback

        return TrainingResult(
            exit_code=-1,
            stdout=_to_str(e.stdout, ""),
            stderr=_to_str(e.stderr, f"Timeout after {timeout}s"),
            output_dir=_read_output_dir(config_path),
        )

    output_dir = _read_output_dir(config_path)
    metrics = _read_metrics(output_dir)

    return TrainingResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output_dir=output_dir,
        metrics=metrics,
    )


def _read_output_dir(config_path: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["train"]["output_dir"]


def _read_metrics(output_dir: str) -> Optional[Dict[str, Any]]:
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Shared data generation (for server vs local comparison tests)
# ---------------------------------------------------------------------------


def generate_shared_token_data(
    num_samples: int,
    seq_len: int = 64,
    vocab_size: int = 32000,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Generate deterministic token data as raw (input_ids, labels) pairs."""
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        tokens = [rng.randint(1, vocab_size - 1) for _ in range(seq_len)]
        labels = tokens[:]
        samples.append((tokens, labels))
    return samples


def write_tokenized_dataset(
    samples: List[Tuple[List[int], List[int]]],
    output_path: str,
) -> str:
    """Write (input_ids, labels) pairs to a JSONL file for Trainer's tokenized dataset."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for input_ids, labels in samples:
            f.write(json.dumps({"input_ids": input_ids, "labels": labels}) + "\n")
    return output_path


def samples_to_xorl_datums(
    samples: List[Tuple[List[int], List[int]]],
) -> list:
    """Convert raw (input_ids, labels) pairs to xorl_client.Datum objects."""
    import xorl_client

    datums = []
    for input_ids, labels in samples:
        datum = xorl_client.Datum(
            model_input=xorl_client.ModelInput.from_ints(input_ids),
            loss_fn_inputs={"labels": labels},
        )
        datums.append(datum)
    return datums
