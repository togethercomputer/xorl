"""
Server Arguments for Xorl Training API Server.

This module provides ServerArguments - a minimal configuration class
containing only fields relevant to the training server (model loading,
device management, parallelism, etc.), excluding client-side training
parameters like batch size, epochs, and optimizer settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Literal, Any


@dataclass
class ServerArguments:
    """
    Server arguments for the Training API.

    Contains only server-side configuration:
    - Model loading and initialization
    - Attention and optimization implementations
    - Parallelism settings (FSDP, Ulysses, etc.)
    - Device and memory management
    - Checkpoint loading
    - Worker configuration

    Does NOT include client-side training parameters like:
    - micro_batch_size, gradient_accumulation_steps
    - num_train_epochs, save_steps
    - optimizer, lr, weight_decay
    - wandb settings

    Usage:
        from xorl.server import ServerArguments
        from xorl.arguments import parse_args

        args = parse_args(ServerArguments)
        model_path = args.model_path
        worker_addr = args.worker.bind_address
    """

    # ========================================================================
    # Model Configuration
    # ========================================================================

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained model (HF Hub or local path)"}
    )

    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model identifier for validation (e.g., 'Qwen/Qwen3-32B'). Defaults to model_path if not specified."}
    )

    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model config. Defaults to model_path"}
    )

    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer. Defaults to config_path"}
    )

    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2", "flash_attention_3", "native-sparse"]] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation"}
    )

    moe_implementation: Optional[Literal[None, "eager", "fused"]] = field(
        default=None,
        metadata={"help": "MoE implementation"}
    )

    force_use_huggingface: bool = field(
        default=False,
        metadata={"help": "Force loading model from HuggingFace"}
    )

    # Multimodal model configuration
    foundation: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Foundation model extra config"}
    )

    encoders: Dict[Literal["image"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal encoder config"}
    )

    decoders: Dict[Literal["image"], Dict[str, str]] = field(
        default_factory=dict,
        metadata={"help": "Multimodal decoder config"}
    )

    # ========================================================================
    # Parallelism Configuration
    # ========================================================================

    data_parallel_mode: Optional[Literal["none", "ddp", "fsdp1", "fsdp2"]] = field(
        default="fsdp2",
        metadata={"help": "Data parallelism mode. Use 'none' for single GPU without any parallelization."}
    )

    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallelism size"}
    )

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallelism size"}
    )

    ulysses_parallel_size: int = field(
        default=1,
        metadata={"help": "Ulysses sequence parallelism size"}
    )

    context_parallel_size: int = field(
        default=1,
        metadata={"help": "Context parallelism size"}
    )

    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert parallelism size for MoE models"}
    )

    data_parallel_replicate_size: int = field(
        default=1,
        metadata={"help": "Data parallel replicate size (HSDP)"}
    )

    data_parallel_shard_size: int = field(
        default=1,
        metadata={"help": "Data parallel shard size (FSDP)"}
    )

    basic_modules: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "Basic modules to shard in FSDP"}
    )

    # ========================================================================
    # Memory & Performance
    # ========================================================================

    enable_mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision training"}
    )

    enable_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing"}
    )

    enable_full_shard: bool = field(
        default=True,
        metadata={"help": "Enable full parameter sharding (FSDP)"}
    )

    enable_fsdp_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP CPU offloading"}
    )

    enable_activation_offload: bool = field(
        default=False,
        metadata={"help": "Enable activation CPU offloading"}
    )

    init_device: Optional[Literal["cpu", "meta", "cuda"]] = field(
        default="meta",
        metadata={"help": "Device for model initialization"}
    )

    enable_rank0_init: bool = field(
        default=False,
        metadata={"help": "Deprecated: Use init_device='cpu' instead"}
    )

    use_liger: bool = field(
        default=True,
        metadata={"help": "Use Liger kernel optimizations"}
    )

    # ========================================================================
    # Checkpointing & Output
    # ========================================================================

    output_dir: str = field(
        default="outputs",
        metadata={"help": "Output directory for checkpoints, sampler weights, and logs (must be on shared filesystem for multi-node)"}
    )

    storage_limit: str = field(
        default="10TB",
        metadata={"help": "Maximum disk usage for output_dir (e.g., '1GB', '500MB', '10GB'). Save operations will fail with StorageLimitError when limit is exceeded. Default: 10TB."}
    )

    load_checkpoint_path: str = field(
        default="",
        metadata={"help": "Path to checkpoint to load"}
    )

    ckpt_manager: Optional[Literal["torch", "dcp", "omnistore", "bytecheckpoint"]] = field(
        default="dcp",
        metadata={"help": "Checkpoint manager type"}
    )

    # ========================================================================
    # Logging
    # ========================================================================

    log_level: str = field(
        default="INFO",
        metadata={"help": "Logging level (DEBUG, INFO, WARNING, ERROR)"}
    )

    # ========================================================================
    # Worker Configuration
    # ========================================================================

    worker_bind_address: str = field(
        default="tcp://127.0.0.1:5556",
        metadata={"help": "ZMQ ROUTER socket address to bind (rank 0 worker)"}
    )

    worker_connection_timeout: float = field(
        default=60.0,
        metadata={"help": "Timeout in seconds for worker-executor connection"}
    )

    worker_max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for failed operations"}
    )

    # ========================================================================
    # Data Processing Configuration
    # ========================================================================

    packing_seq_len: int = field(
        default=32000,
        metadata={"help": "Maximum sequence length for sample packing (default: 32000)"}
    )

    enable_packing: bool = field(
        default=True,
        metadata={"help": "Enable sample packing to combine multiple samples into one sequence"}
    )

    # ========================================================================
    # LoRA Configuration
    # ========================================================================

    enable_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA adapters for training"}
    )

    lora_rank: int = field(
        default=32,
        metadata={"help": "LoRA rank (r parameter)"}
    )

    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling parameter"}
    )

    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply LoRA to (e.g., ['q_proj', 'k_proj', 'v_proj', 'o_proj']). If None, uses default based on model architecture."}
    )

    def __post_init__(self):
        """Validate and set defaults."""
        # Set default paths
        if self.config_path is None:
            self.config_path = self.model_path

        if self.tokenizer_path is None:
            self.tokenizer_path = self.config_path

        # Validate model_path
        if self.model_path is None:
            raise ValueError("model_path is required for server configuration")

        # Validate worker address
        if not self.worker_bind_address.startswith("tcp://"):
            raise ValueError("worker_bind_address must be a valid ZMQ TCP address (tcp://host:port)")

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert ServerArguments to the config dict format expected by ModelRunner.

        Returns:
            Dict with 'model', 'train', 'data', and 'lora' sections
        """
        config = {
            "model": {
                "model_path": self.model_path,
                "config_path": self.config_path,
                "tokenizer_path": self.tokenizer_path,
                "attn_implementation": self.attn_implementation,
                "moe_implementation": self.moe_implementation,
                "force_use_huggingface": self.force_use_huggingface,
                "foundation": self.foundation,
                "encoders": self.encoders,
                "decoders": self.decoders,
                "basic_modules": self.basic_modules,
            },
            "train": {
                "output_dir": self.output_dir,
                "data_parallel_mode": self.data_parallel_mode,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "tensor_parallel_size": self.tensor_parallel_size,
                "ulysses_parallel_size": self.ulysses_parallel_size,
                "context_parallel_size": self.context_parallel_size,
                "expert_parallel_size": self.expert_parallel_size,
                "data_parallel_replicate_size": self.data_parallel_replicate_size,
                "data_parallel_shard_size": self.data_parallel_shard_size,
                "enable_mixed_precision": self.enable_mixed_precision,
                "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
                "enable_full_shard": self.enable_full_shard,
                "enable_fsdp_offload": self.enable_fsdp_offload,
                "enable_activation_offload": self.enable_activation_offload,
                "init_device": self.init_device,
                "enable_rank0_init": self.enable_rank0_init,
                "use_liger": self.use_liger,
                "load_checkpoint_path": self.load_checkpoint_path,
                "ckpt_manager": self.ckpt_manager,
            },
            "data": {
                # Empty data section - data comes from client at runtime
            },
            "lora": {
                "enabled": self.enable_lora,
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "target_modules": self.lora_target_modules,
            },
        }
        return config

    def get_world_size(self) -> int:
        """
        Calculate world size from parallelism configuration.
        
        Note: EP (Expert Parallel) is NOT included in world_size calculation.
        EP creates a separate 2D mesh: world_size = ep_size * ep_fsdp_size
        where ep_fsdp_size contains all other parallelism dimensions.

        Returns:
            Required world size (number of GPUs)
        """
        return (
            self.pipeline_parallel_size *
            self.tensor_parallel_size *
            self.ulysses_parallel_size *
            self.context_parallel_size *
            self.data_parallel_replicate_size *
            self.data_parallel_shard_size
        )
    
    def get_ep_fsdp_size(self) -> int:
        """
        Calculate ep_fsdp_size (the size of each expert parallel group).
        
        For non-EP models (ep_size=1), this equals world_size.
        For EP models, ep_fsdp_size = world_size / ep_size.

        Returns:
            Size of the ep_fsdp dimension
        """
        world_size = self.get_world_size()
        if self.expert_parallel_size > 1:
            assert world_size % self.expert_parallel_size == 0, \
                f"world_size ({world_size}) must be divisible by expert_parallel_size ({self.expert_parallel_size})"
            return world_size // self.expert_parallel_size
        return world_size
    
    def get_dp_size(self) -> int:
        """
        Calculate data parallel size (auto-calculated from other dimensions).
        
        IMPORTANT: EP (Expert Parallel) is NOT included in this calculation!
        EP creates a separate 2D mesh and doesn't participate in the main mesh.
        
        dp_size is the remaining parallelism after accounting for pp/tp/ulysses/cp.
        It's then split into dp_replicate_size and dp_shard_size.

        Returns:
            Data parallel size
        """
        world_size = self.get_world_size()
        other_parallel = (
            self.pipeline_parallel_size *
            self.tensor_parallel_size *
            self.ulysses_parallel_size *
            self.context_parallel_size
            # NOTE: expert_parallel_size is NOT included here!
        )
        if world_size % other_parallel != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"pp({self.pipeline_parallel_size}) × "
                f"tp({self.tensor_parallel_size}) × "
                f"ulysses({self.ulysses_parallel_size}) × "
                f"cp({self.context_parallel_size}) = {other_parallel}"
            )
        return world_size // other_parallel


def parse_server_args() -> ServerArguments:
    """
    Parse ServerArguments from command line and YAML config.

    This function handles the flat ServerArguments structure by creating
    a temporary nested structure that parse_args can handle, then flattening
    the result.

    Returns:
        ServerArguments with all fields populated from YAML and CLI
    """
    from xorl.arguments import parse_args
    import yaml
    import sys

    # Read YAML directly to get flat structure
    config_path = None
    for i, arg in enumerate(sys.argv):
        if not arg.startswith('--') and i > 0 and not sys.argv[i-1].startswith('--'):
            config_path = arg
            break

    if not config_path:
        raise ValueError("Config file path required as first positional argument")

    # Load YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        raise ValueError(f"Empty config file: {config_path}")

    # Process CLI overrides (--bind_address, --connection_timeout, etc.)
    cli_overrides = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--') and not arg.startswith('---'):
            key = arg[2:]  # Remove '--'
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                # Convert string values to appropriate types
                value = sys.argv[i + 1]
                # Try to parse as number or boolean
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                cli_overrides[key] = value
                i += 2
            else:
                i += 1
        else:
            i += 1

    # Apply CLI overrides
    config_data.update(cli_overrides)

    # Create ServerArguments from flat config
    # Filter out fields that don't belong to ServerArguments
    valid_fields = {f.name for f in __import__('dataclasses').fields(ServerArguments)}
    filtered_config = {k: v for k, v in config_data.items() if k in valid_fields}

    # Create ServerArguments
    server_args = ServerArguments(**filtered_config)

    return server_args
