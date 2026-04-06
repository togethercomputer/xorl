"""Argument utils"""

import argparse
import json
import os
import subprocess
import sys
import types
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, field, fields
from enum import Enum
from inspect import isclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    get_type_hints,
)

import torch
import yaml

from .utils import logging


T = TypeVar("T")

logger = logging.get_logger(__name__)


def _detect_repo_commit() -> Optional[str]:
    """Best-effort git commit detection for experiment metadata."""
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def get_default_process_count():
    if xorl_dataset_num_proc := os.environ.get("XORL_DATASET_NUM_PROC"):
        return int(xorl_dataset_num_proc)
    if xorl_dataset_processes := os.environ.get("XORL_DATASET_PROCESSES"):
        return int(xorl_dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()


# Dataset configuration dataclasses
@dataclass
class DatasetConfig:
    """dataset configuration supporting multiple loading methods"""

    # Core fields
    path: Optional[str | None] = field(
        default=None,
        metadata={"help": "HuggingFace dataset repo | s3:// | gs:// | https:// | path to local file or directory"},
    )
    type: Optional[str] = field(
        default="tokenized",
        metadata={"help": "The type of dataset. Only 'tokenized' is currently supported."},
    )

    # HuggingFace Hub fields
    name: Optional[str | None] = field(
        default=None,
        metadata={"help": "Name of dataset configuration to load (for HuggingFace datasets)"},
    )
    split: Optional[str | None] = field(
        default=None,
        metadata={"help": "Name of dataset split to load from (e.g., 'train', 'validation', 'test')"},
    )
    revision: Optional[str | None] = field(
        default=None,
        metadata={"help": "The specific revision of the dataset to use when loading from HF Hub"},
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Trust remote code for untrusted source"},
    )

    # File loading fields
    data_files: Optional[Union[str, List[str]] | None] = field(
        default=None,
        metadata={"help": "Path to source data files (single file string or list of files)"},
    )
    ds_type: Optional[str | None] = field(
        default=None,
        metadata={
            "help": "Dataset type when loading files (json, csv, parquet, arrow, text). Auto-inferred if not specified."
        },
    )
    # split dataset into N pieces (use with shards_idx)
    shards: Optional[int | None] = field(
        default=None,
        metadata={"help": "Split dataset into N pieces (use with shards_idx)"},
    )
    # the index of sharded dataset to use
    shards_idx: Optional[int | None] = field(
        default=None,
        metadata={"help": "The index of sharded dataset to use"},
    )
    # process dataset in N sequential chunks for memory efficiency (exclusive with `shards`)
    preprocess_shards: Optional[int | None] = field(
        default=None,
        metadata={"help": "Process dataset in N sequential chunks for memory efficiency (exclusive with `shards`)"},
    )
    max_seq_len: Optional[int | None] = field(
        default=None,
        metadata={"help": "Max sequence length. Samples with input_ids longer than this are filtered out."},
    )

    # Distillation fields
    activations_path: Optional[str | None] = field(
        default=None,
        metadata={"help": "Path to pre-computed teacher activations (local path or s3://)"},
    )

    def __post_init__(self):
        """Validate dataset configuration"""
        if self.type != "tokenized":
            raise NotImplementedError(
                f"Dataset type '{self.type}' is not implemented. Only 'tokenized' is currently supported."
            )

        if self.path is None:
            raise ValueError("'path' is required for dataset configuration.")

        self._validate_loading_method()

    def _validate_loading_method(self):
        """Validate the dataset loading configuration based on path and other fields"""
        path = self.path

        if path == "dummy":
            if self.type != "tokenized":
                raise ValueError("Dummy dataset only supports type='tokenized'.")
            return

        # Remote filesystem validation
        if any(
            path.startswith(protocol)
            for protocol in [
                "s3://",
                "gs://",
                "gcs://",
                "adl://",
                "abfs://",
                "az://",
                "oci://",
            ]
        ):
            # For remote filesystems, path should be valid
            if not path or len(path.split("://")) != 2:
                raise ValueError(f"Invalid remote filesystem path: {path}")

        # HTTPS validation
        elif path.startswith("https://"):
            if not path or len(path.split("://")) != 2:
                raise ValueError(f"Invalid HTTPS URL: {path}")

        # Local file/directory or HuggingFace Hub
        else:
            # If data_files is specified, it should be for loading specific files
            if self.data_files is not None:
                if self.ds_type is None:
                    raise ValueError("'ds_type' is required when 'data_files' is specified.")

        # Validate ds_type if specified
        if self.ds_type is not None:
            valid_ds_types = ["json", "csv", "parquet", "arrow", "text"]
            if self.ds_type not in valid_ds_types:
                raise ValueError(f"Invalid 'ds_type': {self.ds_type}. Must be one of {valid_ds_types}")

    def get_loading_info(self) -> Dict[str, Any]:
        """Get information about how this dataset should be loaded"""
        path = self.path

        # Determine loading method
        if any(
            path.startswith(protocol)
            for protocol in [
                "s3://",
                "gs://",
                "gcs://",
                "adl://",
                "abfs://",
                "az://",
                "oci://",
            ]
        ):
            method = "remote_filesystem"
        elif path.startswith("https://"):
            method = "https"
        elif "/" in path and not path.startswith("./") and not path.startswith("/"):
            # Likely HuggingFace Hub format (org/dataset-name)
            method = "huggingface_hub"
        else:
            # Local file or directory
            method = "local"

        return {
            "method": method,
            "path": self.path,
            "name": self.name,
            "split": self.split,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "data_files": self.data_files,
            "ds_type": self.ds_type,
        }


@dataclass
class DataArguments:
    """Dataset configuration for training and evaluation datasets"""

    datasets: Optional[List[Dict[str, Any]]] = field(
        default_factory=list,
        metadata={"help": "List of dataset configurations. Each dataset must have type='tokenized'."},
    )
    test_datasets: Optional[List[Dict[str, Any]]] = field(
        default_factory=list,
        metadata={"help": "List of test dataset configurations. Each dataset must have type='tokenized'."},
    )
    dataset_prepared_path: str = field(
        default="last_prepared_dataset",
        metadata={"help": "Path to the prepared dataset. Defaults to 'last_prepared_dataset'."},
    )
    shuffle_merged_datasets: Optional[bool] = field(
        default=True,
        metadata={"help": "Shuffle merged datasets before training"},
    )
    shuffle_before_merging_datasets: Optional[bool] = field(
        default=True,
        metadata={"help": "Shuffle each dataset individually before merging"},
    )
    push_dataset_to_hub: Optional[str] = field(
        default=None,
        metadata={
            "help": "Push prepared dataset to hub - repo_org/repo_name. If specified, the dataset will be pushed to HuggingFace Hub after preparation."
        },
    )
    hf_use_auth_token: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use hf `use_auth_token` for loading datasets. Useful for fetching private datasets. Required to be true when used in combination with `push_dataset_to_hub`"
        },
    )
    skip_prepare_dataset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Skip preparing the dataset. If True, the dataset will be loaded from the prepared dataset path."
        },
    )
    is_preprocess: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether the dataset is being preprocessed. If True, the dataset is assumed to be already prepared and will not be prepared again."
        },
    )
    num_dataset_shards_to_save: Optional[int | None] = field(
        default=None,
        metadata={
            "help": "Number of shards to save the prepared dataset to. If not specified, the dataset will be saved to a single shard."
        },
    )
    dataset_shard_num: Optional[int | None] = field(
        default=None,
        metadata={
            "help": "Number of shards to split the dataset into. If not specified, the dataset will not be sharded."
        },
    )
    dataset_shard_idx: Optional[int | None] = field(
        default=None,
        metadata={"help": "Index of the shard to use. If not specified, the dataset will not be sharded."},
    )
    val_set_size: Optional[int | float | None] = field(
        default=None,
        metadata={"help": "Size of the validation set. If not specified, the validation set will not be created."},
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of processes to use for dataset processing. If not specified, the dataset will be processed by a single process."
        },
    )
    sample_packing_method: Optional[str] = field(
        default="sequential",
        metadata={
            "help": "The method to use for sample packing. Should be 'sequential' or 'multipack'. Defaults to 'sequential'."
        },
    )
    sample_packing_group_size: Optional[int] = field(
        default=100000,
        metadata={
            "help": "The number of samples packed at a time. Increasing this value helps with packing, but usually only slightly (<%1)."
        },
    )
    sample_packing_sequentially: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack samples sequentially."},
    )
    sample_packing_mp_start_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "The multiprocessing start method to use for packing. Should be 'fork', 'spawn' or 'forkserver'."
        },
    )
    eval_sample_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Set to 'false' if getting errors during eval with sample_packing on."},
    )
    sample_packing_sequence_len: Optional[int] = field(
        default=32000,
        metadata={"help": "The length of the sequence to use for sample packing. Defaults to 32000."},
    )
    select_columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The columns to select from the dataset. If not specified, all columns will be selected."},
    )
    dataloader_pin_memory: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to pin memory for faster GPU transfer in dataloader. Defaults to True."},
    )
    dataloader_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of worker processes for data loading. Defaults to 8."},
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of batches to prefetch per worker in dataloader. Defaults to 2. Set to None when num_workers=0."
        },
    )
    dataloader_drop_last: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to drop the last incomplete batch in dataloader. Defaults to True."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=128,
        metadata={
            "help": "Pad packed sequences to a multiple of this value. Defaults to 128 for optimal GPU performance."
        },
    )

    def _convert_datasets_to_config(
        self, datasets: Optional[List[Dict[str, Any]]], dataset_name: str
    ) -> List[DatasetConfig]:
        """Validate and convert dataset configurations"""
        if datasets is None:
            return []

        if not isinstance(datasets, list):
            raise ValueError(f"'{dataset_name}' must be a list.")

        # Convert dict configurations to DatasetConfig objects
        converted_datasets = []
        for i, dataset_dict in enumerate(datasets):
            if not isinstance(dataset_dict, dict):
                raise ValueError(f"Dataset at index {i} in '{dataset_name}' must be a dictionary.")

            try:
                converted_datasets.append(DatasetConfig(**dataset_dict))
            except TypeError as e:
                raise ValueError(f"Invalid configuration for dataset at index {i} in '{dataset_name}': {e}")

        return converted_datasets

    def __post_init__(self):
        """Validate and convert dataset configurations"""
        if self.datasets is None:
            raise ValueError("At least one dataset must be specified in 'datasets' list.")

        if not isinstance(self.datasets, list) or len(self.datasets) == 0:
            raise ValueError("'datasets' must be a non-empty list.")

        if self.datasets:
            self.datasets = self._convert_datasets_to_config(self.datasets, "datasets")
        else:
            self.datasets = None

        if self.test_datasets:
            self.test_datasets = self._convert_datasets_to_config(self.test_datasets, "test_datasets")
        else:
            self.test_datasets = None

        if self.dataset_num_proc is None:
            self.dataset_num_proc = get_default_process_count()

        if self.sample_packing_method not in ["sequential", "multipack"]:
            raise ValueError(
                f"Invalid sample packing method: {self.sample_packing_method}. Should be 'sequential' or 'multipack'."
            )


@dataclass
class ModelArguments:
    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the model config. Defaults to `model_path`."},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the pre-trained model. If unspecified, use random init."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the tokenizer. Defaults to `config_path`."},
    )
    foundation: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Foundation model extra config."},
    )
    encoders: Dict[Literal["image", "video", "audio"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal encoder config and weights."},
    )
    attn_implementation: Optional[Literal["eager", "sdpa", "native", "flash_attention_3", "flash_attention_4"]] = field(
        default="flash_attention_3",
        metadata={
            "help": "Attention implementation. 'native': PyTorch SDPA+cuDNN (no deps, Hopper+Blackwell). "
            "'flash_attention_3': FA3 (Hopper). 'flash_attention_4': FA4 CUTE (Hopper+Blackwell)."
        },
    )
    moe_implementation: Optional[Literal[None, "eager", "triton", "native", "quack"]] = field(
        default=None,
        metadata={
            "help": "MoE implementation to use. 'triton' uses Triton group GEMM kernels, 'native' uses torch._grouped_mm, 'quack' uses quack kernels."
        },
    )
    ep_dispatch: str = field(
        default="alltoall",
        metadata={"help": "EP dispatch strategy: 'alltoall' (default) or 'deepep' (NVLink-optimized)."},
    )
    train_router: bool = field(
        default=False,
        metadata={
            "help": "Whether expert computation gradients should flow through routing weights. "
            "Disabled by default and must remain False when ep_dispatch='deepep'."
        },
    )
    deepep_buffer_size_gb: float = field(
        default=2.0,
        metadata={"help": "DeepEP buffer size in GB (effective when ep_dispatch='deepep')."},
    )
    deepep_num_sms: int = field(
        default=20,
        metadata={
            "help": "Number of SMs for DeepEP communication kernels (must be even, default 20). Lower values leave more SMs for overlapped compute."
        },
    )
    deepep_async_combine: bool = field(
        default=False,
        metadata={"help": "Enable async combine for DeepEP (overlap combine with next layer's compute)."},
    )
    basic_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Basic modules beyond model._no_split_modules to be sharded in FSDP."},
    )
    merge_qkv: bool = field(
        default=True,
        metadata={
            "help": "Keep q/k/v projections fused as qkv_proj. "
            "When False, unfuse into separate q_proj/k_proj/v_proj for independent handling "
            "(e.g., tensor parallelism, independent LoRA per projection)."
        },
    )
    rmsnorm_mode: Literal["eager", "native", "compile"] = field(
        default="native",
        metadata={
            "help": "RMSNorm implementation mode. 'native' uses torch.nn.functional.rms_norm "
            "and is the default. 'compile' runs that native path through torch.compile. "
            "'eager' uses the plain eager implementation."
        },
    )

    def __post_init__(self):
        if self.config_path is None and self.model_path is None:
            raise ValueError("`config_path` must be specified when `model_path` is None.")

        if self.config_path is None:
            self.config_path = self.model_path

        if self.tokenizer_path is None:
            self.tokenizer_path = self.config_path

        supported_encoder_types = ["image", "video", "audio"]
        for encoder_type, encoder_args in self.encoders.items():
            if encoder_type not in supported_encoder_types:
                raise ValueError(
                    f"Unsupported encoder type: {encoder_type}. Should be one of {supported_encoder_types}."
                )

            if encoder_args.get("config_path") is None and encoder_args.get("model_path") is None:
                raise ValueError("`config_path` and `model_path` cannot be both empty.")

            if encoder_args.get("config_path") is None:
                encoder_args["config_path"] = encoder_args["model_path"]


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "Path to save model checkpoints."},
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "Maximum learning rate or default learning rate, or initial learning rate for warmup."},
    )
    lr_min: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate."},
    )
    lr_start: float = field(
        default=0.0,
        metadata={"help": "Learning rate for warmup start. Default to 0.0."},
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "L2 regularization strength."},
    )
    no_decay_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Modules without weight decay, for example, RMSNorm."},
    )
    no_decay_params: List[str] = field(
        default_factory=list,
        metadata={"help": "Parameters without weight decay, for example, bias."},
    )

    optimizer: Literal["adamw", "anyprecision_adamw", "sgd", "muon"] = field(
        default="adamw",
        metadata={"help": "Optimizer type. 'muon' uses Newton-Schulz orthogonalization for 2D+ weight matrices."},
    )
    optimizer_dtype: Literal["fp32", "bf16"] = field(
        default="bf16",
        metadata={"help": "Dtype for optimizer states (momentum/variance) in anyprecision_adamw/muon."},
    )
    muon_lr: float = field(
        default=0.02,
        metadata={
            "help": "Learning rate for Muon parameter groups (2D+ weight matrices). Only used when optimizer='muon'."
        },
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Momentum coefficient for Muon parameter groups."},
    )
    muon_nesterov: bool = field(
        default=True,
        metadata={"help": "Use Nesterov momentum for Muon parameter groups."},
    )
    muon_ns_steps: int = field(
        default=5,
        metadata={"help": "Number of Newton-Schulz iterations for Muon optimizer."},
    )
    muon_adjust_lr_fn: Optional[str] = field(
        default=None,
        metadata={
            "help": "LR adjustment for Muon. 'original': scale by sqrt(max(1,A/B)). "
            "'match_rms_adamw': scale by 0.2*sqrt(max(A,B)) so Muon can reuse AdamW LR/WD. "
            "None defaults to 'original'."
        },
    )

    @property
    def optimizer_kwargs(self) -> Dict[str, Any]:
        """Collect optimizer-specific kwargs from flat fields into a dict for build_optimizer."""
        kwargs: Dict[str, Any] = {}
        if self.optimizer == "muon":
            kwargs["muon_lr"] = self.muon_lr
            kwargs["muon_momentum"] = self.muon_momentum
            kwargs["muon_nesterov"] = self.muon_nesterov
            kwargs["muon_ns_steps"] = self.muon_ns_steps
            kwargs["muon_adjust_lr_fn"] = self.muon_adjust_lr_fn
            # Wire optimizer_dtype -> muon_momentum_dtype so "bf16" sets bf16 Muon momentum
            if self.optimizer_dtype == "bf16":
                kwargs["muon_momentum_dtype"] = torch.bfloat16
        return kwargs

    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Clip value for gradient norm."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size. The number of samples per iteration on each device."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps. The effective batch size per device is micro_batch_size * gradient_accumulation_steps."
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Epochs to train."},
    )
    lr_warmup_ratio: float = field(
        default=0,
        metadata={"help": "Ratio of learning rate warmup steps."},
    )
    lr_decay_style: str = field(
        default="constant",
        metadata={"help": "Name of the learning rate scheduler."},
    )
    lr_decay_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of learning rate decay steps."},
    )
    enable_mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision training."},
    )
    enable_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing."},
    )
    enable_reentrant: bool = field(
        default=False,
        metadata={"help": "Use reentrant gradient checkpointing."},
    )
    recompute_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Per-submodule selective checkpointing. Options: 'self_attn', 'mlp'. "
            "When None, uses whole-layer checkpoint (legacy). "
            "Example: ['self_attn'] checkpoints only attention, keeping MoE activations."
        },
    )
    moe_checkpoint_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "MoE checkpoint strategy. None = full recompute (default). "
            "'moe_act' = recompute only gate/up activations in backward, "
            "skip EP communication recomputation. Only effective with MoE models."
        },
    )

    @property
    def moe_recomputed(self) -> bool:
        """Whether MoE forward is recomputed during gradient checkpointing.

        Used to decide whether routing replay is needed with EP: replay is only
        required when the MoE forward (including EP all-to-all) is recomputed.
        """
        if self.recompute_modules is not None:
            return "mlp" in self.recompute_modules and self.moe_checkpoint_method != "moe_act"
        return True  # legacy whole-layer checkpoint always recomputes MoE

    enable_full_shard: bool = field(
        default=True,
        metadata={"help": "Enable fully shard for FSDP training (ZeRO-3)."},
    )
    enable_forward_prefetch: bool = field(
        default=True,
        metadata={"help": "Enable forward prefetch in FSDP."},
    )
    enable_activation_offload: bool = field(
        default=False,
        metadata={"help": "Enable activation offload to CPU."},
    )
    activation_gpu_limit: float = field(
        default=0.0,
        metadata={
            "help": "When enabling activation offload, `activation_gpu_limit` GB activations are allowed to reserve on GPU."
        },
    )
    init_device: Literal["cpu", "cuda", "meta", "npu"] = field(
        default="cuda",
        metadata={
            "help": "Device to initialize model weights. 1. `cpu`: Init parameters on CPU in rank0 only. 2. `cuda`: Init parameters on GPU. 3. `meta`: Init parameters on meta. 4. `npu`: Init parameters on Ascend NPU."
        },
    )
    load_weights_mode: Literal["broadcast", "all_ranks"] = field(
        default="broadcast",
        metadata={
            "help": "Weight loading mode. 'broadcast': rank0 reads weights and broadcasts to other ranks (default, avoids disk I/O bottleneck). 'all_ranks': every rank reads weights from disk independently."
        },
    )
    enable_full_determinism: bool = field(
        default=False,
        metadata={"help": "Enable full determinism."},
    )
    allow_cuda_launch_blocking: bool = field(
        default=False,
        metadata={
            "help": "Setting CUDA_LAUNCH_BLOCKING=1 would degrade performance significantly. Leave this as False to prevent CUDA_LAUNCH_BLOCKING from being accidentally enabled. DO NOT enable this unless you are debugging something."
        },
    )
    empty_cache_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between two empty cache operations."},
    )
    gc_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between two gc.collect. GC is disabled if it is positive."},
    )
    data_parallel_mode: Literal["none", "ddp", "fsdp2"] = field(
        default="fsdp2",
        metadata={"help": "Data parallel mode."},
    )
    data_parallel_replicate_size: int = field(
        default=-1,
        metadata={"help": "Data parallel replicate size."},
    )
    data_parallel_shard_size: int = field(
        default=-1,
        metadata={"help": "Data parallel shard degree."},
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size."},
    )
    ep_outside: bool = field(
        default=False,
        metadata={"help": "Enable expert parallelism outside in ep-fsdp."},
    )
    ulysses_parallel_size: int = field(
        default=1,
        metadata={"help": "Ulysses sequence parallel size."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size. Shards model weights across GPUs within a node."},
    )
    ringattn_parallel_size: int = field(
        default=1,
        metadata={"help": "Ring attention parallel size."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size. Splits model layers across GPUs."},
    )
    pipeline_parallel_schedule: str = field(
        default="1F1B",
        metadata={"help": "Pipeline parallel schedule: '1F1B' or 'GPipe'."},
    )
    pp_variable_seq_lengths: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, negotiate the per-step maximum sequence length across PP ranks "
                "via all-reduce and pad only to that dynamic max, avoiding waste from a "
                "static sample_packing_sequence_len.  Each unique seq_len gets its own "
                "cached PipelineStage so P2P buffers always match the actual shape."
            )
        },
    )
    reshard_after_forward: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether FSDP2 reshards parameters after forward. "
                "True: reshard after forward, re-gather in backward (saves memory). "
                "False: keep params unsharded (saves communication, uses more memory). "
                "Default (None): True for standard training, False for pipeline parallelism."
            )
        },
    )
    cp_fsdp_mode: str = field(
        default="all",
        metadata={"help": "How to fold SP into FSDP: 'all' (ulysses+ring), 'ulysses_only', 'ring_only', or 'none'."},
    )
    ckpt_manager: Literal["dcp"] = field(
        default="dcp",
        metadata={"help": "Checkpoint manager."},
    )
    save_async: bool = field(
        default=False,
        metadata={"help": "Whether to save checkpoint asynchronously."},
    )
    load_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume from."},
    )
    save_steps: int = field(
        default=0,
        metadata={"help": "Number of steps between two checkpoint saves."},
    )
    save_epochs: float = field(
        default=1,
        metadata={
            "help": "Fraction or number of epochs between two checkpoint saves. E.g., 0.25 saves 4 times per epoch."
        },
    )
    save_hf_weights: bool = field(
        default=True,
        metadata={"help": "Save the huggingface format weights to the last checkpoint dir."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    enable_compile: bool = field(
        default=False,
        metadata={"help": "Enable torch compile."},
    )
    log_format: Literal["progress_bar", "structured"] = field(
        default="progress_bar",
        metadata={
            "help": "Logging format. 'progress_bar' uses tqdm; 'structured' prints parse-friendly key=value lines."
        },
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "Use wandb to log experiment."},
    )
    wandb_project: str = field(
        default="Xorl",
        metadata={"help": "Wandb project name."},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb experiment name."},
    )
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Wandb tags for organizing experiments."},
    )
    repo_commit: Optional[str] = field(
        default=None,
        metadata={"help": "Git commit hash for experiment provenance. Auto-detected when omitted."},
    )
    wandb_log_interval: int = field(
        default=1,
        metadata={"help": "Log to wandb every N steps."},
    )
    enable_profiling: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    profile_start_step: int = field(
        default=1,
        metadata={"help": "Start step for profiling."},
    )
    profile_end_step: int = field(
        default=2,
        metadata={"help": "End step for profiling."},
    )
    profile_trace_dir: str = field(
        default="./trace",
        metadata={"help": "Directory to export the profiling result."},
    )
    profile_record_shapes: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the shapes of the input tensors."},
    )
    profile_profile_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to profile the memory usage."},
    )
    profile_with_stack: bool = field(
        default=True,
        metadata={"help": "Whether or not to record the stack traces."},
    )
    profile_rank0_only: bool = field(
        default=True,
        metadata={
            "help": "whether to profile rank0 only. When false, every rank will be profiled; Please expect many files to save, which can be slow and take a lot of disk space."
        },
    )
    max_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max total training steps. Training stops after this many global steps. Also caps LR scheduler length."
        },
    )

    def __post_init__(self):
        if self.repo_commit is None:
            self.repo_commit = _detect_repo_commit()
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.global_rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        non_dp_size = (
            self.ulysses_parallel_size
            * self.tensor_parallel_size
            * self.ringattn_parallel_size
            * self.pipeline_parallel_size
        )
        if self.world_size % non_dp_size != 0:
            raise ValueError(
                f"World size ({self.world_size}) should be a multiple of "
                f"ulysses_parallel_size ({self.ulysses_parallel_size}) * "
                f"tensor_parallel_size ({self.tensor_parallel_size}) * "
                f"ringattn_parallel_size ({self.ringattn_parallel_size}) * "
                f"pipeline_parallel_size ({self.pipeline_parallel_size}) = {non_dp_size}."
            )
        self.data_parallel_size = self.world_size // non_dp_size

        # configure data parallel size
        if self.data_parallel_replicate_size > 0 and self.data_parallel_shard_size > 0:
            assert self.data_parallel_size == self.data_parallel_replicate_size * self.data_parallel_shard_size, (
                f"data_parallel_size should be equal to data_parallel_replicate_size: {self.data_parallel_replicate_size} * data_parallel_shard_size: {self.data_parallel_shard_size}."
            )

        elif self.data_parallel_replicate_size > 0:
            if self.data_parallel_size % self.data_parallel_replicate_size != 0:
                raise ValueError("data_parallel_size should be a multiple of data_parallel_replicate_size.")
            self.data_parallel_shard_size = self.data_parallel_size // self.data_parallel_replicate_size

        elif self.data_parallel_shard_size > 0:
            if self.data_parallel_size % self.data_parallel_shard_size != 0:
                raise ValueError("data_parallel_size should be a multiple of data_parallel_shard_size.")
            self.data_parallel_replicate_size = self.data_parallel_size // self.data_parallel_shard_size
        else:
            self.data_parallel_replicate_size = 1
            self.data_parallel_shard_size = self.data_parallel_size

        # Calculate global batch size
        self.global_batch_size = self.micro_batch_size * self.gradient_accumulation_steps * self.data_parallel_size
        logger.info_rank0(
            f"Global batch size: {self.global_batch_size} = "
            f"micro_batch_size ({self.micro_batch_size}) * "
            f"gradient_accumulation_steps ({self.gradient_accumulation_steps}) * "
            f"data_parallel_size ({self.data_parallel_size})"
        )

        num_nodes = int(os.getenv("WORLD_SIZE", "1")) // int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        if num_nodes > 1:
            logger.warning_rank0(
                f"Detected {num_nodes} nodes. "
                "Make sure that `train.output_dir` is shared by all nodes. "
                "Otherwise, each node will save checkpoints to its local directory, which may cause inconsistencies or job failures."
            )

        assert self.expert_parallel_size == 1 or self.init_device != "cpu", (
            "cpu init is not supported when enable ep. Please use `init_device = cuda` or `init_device = meta` instead."
        )

        if self.data_parallel_mode == "fsdp2":
            assert self.init_device == "meta", "Please use init_device: meta for FSDP2 training"

        if self.load_checkpoint_path == "auto":
            from .checkpoint_utils import get_checkpoint_path

            self.load_checkpoint_path = get_checkpoint_path(
                output_dir=self.output_dir,
                is_local_rank0=self.local_rank == 0,
                ckpt_manager=self.ckpt_manager,
            )

        # save paths
        self.save_checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        self.step2token_path = os.path.join(self.output_dir, "step2token.json")
        self.model_assets_dir = os.path.join(self.output_dir, "model_assets")

        # determine whether to profile this rank
        if self.enable_profiling:
            if self.profile_rank0_only:
                self.profile_this_rank = self.global_rank == 0
            else:
                logger.warning_rank0(
                    "Profiling on ALL ranks is enabled. This would save a lot of files which takes time and space."
                )
                self.profile_this_rank = True
        else:
            self.profile_this_rank = False

        # Prevent CUDA_LAUNCH_BLOCKING from being accidentally enabled
        if not self.allow_cuda_launch_blocking:
            assert not self.enable_full_determinism, (
                "allow_cuda_launch_blocking is disabled but enable_full_determinism is enabled. enable_full_determinism would set CUDA_LAUNCH_BLOCKING to 1!"
            )
            cuda_launch_blocking_val = os.environ.get("CUDA_LAUNCH_BLOCKING", "").strip()
            assert cuda_launch_blocking_val != "1", (
                "CUDA_LAUNCH_BLOCKING=1 is set when allow_cuda_launch_blocking is not enabled!"
            )


@dataclass
class DistillationArguments:
    """Arguments for streaming distillation training."""

    enable_distillation: bool = field(
        default=False,
        metadata={"help": "Enable distillation mode"},
    )
    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "HuggingFace model ID or local path to teacher model (e.g., 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8')"
        },
    )
    distillation_loss_type: str = field(
        default="forward_kl",
        metadata={"help": "Type of distillation loss: 'forward_kl' or 'reverse_kl'"},
    )
    distillation_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for distillation loss"},
    )
    stream_num_workers: int = field(
        default=32,
        metadata={"help": "Number of workers for loading activations from S3/local"},
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA and QLoRA fine-tuning."""

    enable_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA fine-tuning"},
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha for scaling"},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Modules to apply LoRA to. If None, uses default linear projections."},
    )
    save_lora_only: bool = field(
        default=False,
        metadata={"help": "Only save LoRA weights (not full model) in HF checkpoints"},
    )
    # QLoRA: quantize base weights for memory savings
    enable_qlora: bool = field(
        default=False,
        metadata={"help": "Enable QLoRA (quantized base weights + trainable LoRA). Implies enable_lora=True."},
    )
    quant_format: str = field(
        default="nvfp4",
        metadata={"help": "Quantization format for QLoRA. Supported: 'nvfp4', 'block_fp8', 'nf4'"},
    )
    quant_group_size: int = field(
        default=16,
        metadata={"help": "Group/block size for quantization (16 for nvfp4, 128 for block_fp8, 64 for nf4)"},
    )
    merge_lora_interval: int = field(
        default=0,
        metadata={
            "help": "Merge LoRA delta into base weights every N training steps. "
            "For QLoRA: merge + re-quantize. For LoRA: merge into bf16 weight. "
            "0 = disabled."
        },
    )
    reset_optimizer_on_merge: bool = field(
        default=False,
        metadata={
            "help": "ReLoRA-style optimizer reset after each LoRA merge. "
            "Clears optimizer states (momentum, variance) for LoRA parameters "
            "so Adam rebuilds from scratch for the re-initialized LoRA. "
            "Requires merge_lora_interval > 0."
        },
    )
    exclude_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Modules to exclude from QLoRA injection (kept as bf16). "
            "When None, auto-detected from pre-quantized checkpoint config "
            "(exclude_modules / modules_to_not_convert). "
            "Example: ['lm_head', 'gate']"
        },
    )
    enable_aqn: bool = field(
        default=False,
        metadata={"help": "Enable Adaptive Quantization Noise during training forward passes."},
    )
    aqn_alpha: float = field(
        default=1.0,
        metadata={"help": "Scale factor for AQN noise magnitude."},
    )


@dataclass
class InferArguments:
    model_path: str = field(
        metadata={"help": "Local path/HDFS path to the pre-trained model."},
    )
    tokenizer_path: Optional[str | None] = field(
        default=None,
        metadata={"help": "Local path/HDFS path to the tokenizer. Defaults to `config_path`."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling in decoding."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature value of decoding."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "The top_p value of decoding."},
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Max tokens to generate."},
    )

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


def _string_to_bool(value: Union[bool, str]) -> bool:
    """
    Converts a string input to bool value.

    Taken from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"Truthy value expected: got {value} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
    )


def _convert_str_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely checks that a passed value is a dictionary and converts any string values to their appropriate types.

    Taken from: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/training_args.py#L189
    """
    for key, value in input_dict.items():
        if isinstance(value, dict):
            input_dict[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "false"):  # check for bool
                input_dict[key] = value.lower() == "true"
            elif value.isdigit():  # check for digit
                input_dict[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                input_dict[key] = float(value)

    return input_dict


def _make_choice_type_function(choices: List[Any]) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/hf_argparser.py#L48

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def _optional_int(value: str) -> Optional[int]:
    """
    Parse a string value to Optional[int], handling 'null', 'None', and empty strings as None.

    Args:
        value: String representation of the value

    Returns:
        None if value represents null/None, otherwise parsed integer
    """
    if value in ("null", "None", "none", ""):
        return None
    return int(value)


def parse_args(rootclass: T) -> T:
    """
    Parses the root argument class using the CLI inputs or yaml inputs.

    Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/hf_argparser.py#L266
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_to_subclass = {}
    dict_fields = set()

    # First pass: build field information and dict_fields set
    for subclass in fields(rootclass):
        base = subclass.name
        base_to_subclass[base] = subclass.default_factory
        try:
            type_hints: Dict[str, type] = get_type_hints(subclass.default_factory)
        except Exception:
            raise RuntimeError(f"Type resolution failed for {subclass.default_factory}.")

        for attr in fields(subclass.default_factory):
            if not attr.init:
                continue

            attr_type = type_hints[attr.name]
            origin_type = getattr(attr_type, "__origin__", attr_type)

            # Handle Optional types first
            if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
                if len(attr_type.__args__) == 2 and type(None) in attr_type.__args__:  # Optional[X]
                    # Extract the non-None type
                    attr_type = (
                        attr_type.__args__[0] if isinstance(None, attr_type.__args__[1]) else attr_type.__args__[1]
                    )
                    origin_type = getattr(attr_type, "__origin__", attr_type)

            # Mark dict and list-of-dict fields
            if isclass(origin_type) and issubclass(origin_type, dict):
                dict_fields.add(f"{base}.{attr.name}")
            elif (
                isclass(origin_type)
                and issubclass(origin_type, list)
                and hasattr(attr_type, "__args__")
                and len(attr_type.__args__) > 0
                and hasattr(attr_type.__args__[0], "__origin__")
                and issubclass(attr_type.__args__[0].__origin__, dict)
            ):
                dict_fields.add(f"{base}.{attr.name}")

    # Second pass: build argument parser
    for subclass in fields(rootclass):
        base = subclass.name
        try:
            type_hints: Dict[str, type] = get_type_hints(subclass.default_factory)
        except Exception:
            raise RuntimeError(f"Type resolution failed for {subclass.default_factory}.")

        for attr in fields(subclass.default_factory):
            if not attr.init:
                continue

            attr_type = type_hints[attr.name]
            origin_type = getattr(attr_type, "__origin__", attr_type)
            is_optional = False
            if isinstance(attr_type, str):
                raise RuntimeError(f"Cannot resolve type {attr.type} of {attr.name}.")

            if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
                # if (
                #     len(attr_type.__args__) != 2 or type(None) not in attr_type.__args__
                # ):  # only allows Optional[X]
                #     raise RuntimeError(
                #         f"Cannot resolve type {attr.type} of {attr.name}."
                #     )

                if bool not in attr_type.__args__:  # except for `Union[bool, NoneType]`
                    # Track if this is Optional[X] (Union with None)
                    is_optional = type(None) in attr_type.__args__
                    attr_type = (
                        attr_type.__args__[0] if isinstance(None, attr_type.__args__[1]) else attr_type.__args__[1]
                    )
                    origin_type = getattr(attr_type, "__origin__", attr_type)

            parser_kwargs = attr.metadata.copy()
            if origin_type is Literal or (isinstance(attr_type, type) and issubclass(attr_type, Enum)):
                if origin_type is Literal:
                    parser_kwargs["choices"] = attr_type.__args__
                else:
                    parser_kwargs["choices"] = [x.value for x in attr_type]

                parser_kwargs["type"] = _make_choice_type_function(parser_kwargs["choices"])

                if attr.default is not MISSING:
                    parser_kwargs["default"] = attr.default
                else:
                    parser_kwargs["required"] = True

            elif attr_type is bool or attr_type == Optional[bool]:
                parser_kwargs["type"] = _string_to_bool
                if attr_type is bool or (attr.default is not None and attr.default is not MISSING):
                    parser_kwargs["default"] = False if attr.default is MISSING else attr.default
                    parser_kwargs["nargs"] = "?"
                    parser_kwargs["const"] = True

            elif isclass(origin_type) and issubclass(origin_type, list):
                # Check if this is a list of dictionaries (like datasets field)
                if (
                    hasattr(attr_type, "__args__")
                    and len(attr_type.__args__) > 0
                    and hasattr(attr_type.__args__[0], "__origin__")
                    and issubclass(attr_type.__args__[0].__origin__, dict)
                ):
                    # Special handling for List[Dict[...]] - treat as JSON string
                    parser_kwargs["type"] = str
                    if attr.default_factory is not MISSING:
                        parser_kwargs["default"] = json.dumps(attr.default_factory())
                    elif attr.default is not MISSING and attr.default is not None:
                        parser_kwargs["default"] = json.dumps(attr.default)
                    elif attr.default is MISSING:
                        parser_kwargs["required"] = True
                else:
                    # Regular list handling
                    parser_kwargs["type"] = attr_type.__args__[0]
                    parser_kwargs["nargs"] = "+"
                    if attr.default_factory is not MISSING:
                        parser_kwargs["default"] = attr.default_factory()
                    elif attr.default is MISSING:
                        parser_kwargs["required"] = True

            elif isclass(origin_type) and issubclass(origin_type, dict):
                parser_kwargs["type"] = str  # parse dict inputs with json string
                if attr.default_factory is not MISSING:
                    parser_kwargs["default"] = str(attr.default_factory())
                elif attr.default is MISSING:
                    parser_kwargs["required"] = True

            else:
                # Use custom type function for Optional[int] to handle null values
                if is_optional and attr_type is int:
                    parser_kwargs["type"] = _optional_int
                else:
                    parser_kwargs["type"] = attr_type
                if attr.default is not MISSING:
                    parser_kwargs["default"] = attr.default
                elif attr.default_factory is not MISSING:
                    parser_kwargs["default"] = attr.default_factory()
                else:
                    parser_kwargs["required"] = True

            parser.add_argument(f"--{base}.{attr.name}", **parser_kwargs)

    cmd_args = sys.argv[1:]
    cmd_args_string = "=".join(cmd_args)  # use `=` to mark the end of arg name
    input_data = {}
    if cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    elif cmd_args[0].endswith(".json"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = json.load(f)

    for base, arg_dict in input_data.items():
        for arg_name, arg_value in arg_dict.items():
            if f"--{base}.{arg_name}=" not in cmd_args_string:  # lower priority
                cmd_args.append(f"--{base}.{arg_name}")
                if isinstance(arg_value, str):
                    cmd_args.append(arg_value)
                elif isinstance(arg_value, list):
                    # Check if this should be treated as a JSON list (for datasets field)
                    field_key = f"{base}.{arg_name}"
                    if field_key in dict_fields:
                        # This is a list that should be treated as JSON (like datasets)
                        cmd_args.append(json.dumps(arg_value))
                    else:
                        # Regular list - extend as separate arguments
                        cmd_args.extend(str(x) for x in arg_value)
                else:
                    cmd_args.append(json.dumps(arg_value))
    args, remaining_args = parser.parse_known_args(cmd_args)
    if remaining_args:
        raise ValueError(
            f"Unrecognized arguments: {remaining_args}. "
            f"Check your config file or CLI arguments for typos or removed fields."
        )

    parse_result = defaultdict(dict)
    for key, value in vars(args).items():
        if key in dict_fields:
            if isinstance(value, str):
                if value.startswith("{"):
                    # Handle regular dict
                    value = _convert_str_dict(json.loads(value))
                elif value.startswith("["):
                    # Handle list of dicts (like datasets)
                    parsed_list = json.loads(value)
                    if isinstance(parsed_list, list):
                        value = [_convert_str_dict(item) if isinstance(item, dict) else item for item in parsed_list]
                    else:
                        raise ValueError(f"Expected a JSON array for {key}, but got {value}")
                else:
                    raise ValueError(f"Expect a JSON string (dict or array) for {key}, but got {value}")
            else:
                raise ValueError(f"Expect a JSON string for dict/list argument {key}, but got {value}")

        base, name = key.split(".", maxsplit=1)
        parse_result[base][name] = value

    data_classes = {}
    for base, subclass_type in base_to_subclass.items():
        data_classes[base] = subclass_type(**parse_result.get(base, {}))

    return rootclass(**data_classes)


def save_args(args: T, output_path: str) -> None:
    """
    Saves arguments to a json file.

    Args:
        args (dataclass): Arguments.
        output_path (str): Output path.
    """
    os.makedirs(output_path, exist_ok=True)
    local_path = os.path.join(output_path, "xorl_cli.yaml")
    with open(local_path, "w") as f:
        f.write(yaml.safe_dump(asdict(args), default_flow_style=False))


@dataclass
class Arguments:
    """Main arguments container combining model, data, and training arguments."""

    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    distill: "DistillationArguments" = field(default_factory=DistillationArguments)
    lora: "LoRAArguments" = field(default_factory=LoRAArguments)
