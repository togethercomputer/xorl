"""
Pipeline Parallelism for xorl.

Aligned with torchtitan/distributed/pipeline_parallel.py:
- FQN-based generic model splitting
- Support for all PyTorch PP schedule types via get_schedule_class
- Virtual stages (looped + V-style) for multi-stage schedules
- Recursive module pruning for nested HF model structures

GPU Layout (4 GPUs, PP=2, FSDP=2):
    PP Stage 0              PP Stage 1
  +----------------+       +----------------+
  | GPU 0          |  ---> | GPU 2          |   FSDP group 0
  | embed +        |       | layers N..M    |
  | layers 0..N    |       | + norm + head  |
  +----------------+       +----------------+
  | GPU 1          |  ---> | GPU 3          |   FSDP group 1
  | embed +        |       | layers N..M    |
  | layers 0..N    |       | + norm + head  |
  +----------------+       +----------------+
"""

import copy
import types
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
)

from ..utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
    "build_pp_stage",
    "build_pipeline_schedule",
]


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
    input_fqns: Optional[List[str]] = None,
    layer_prefix: str = "layers",
    output_fqns: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Programmatically generates module FQN names per model part.

    Ported from torchtitan's generate_llm_fqn_per_model_part, extended with
    configurable module names for HF-style nested models.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules in layer calculation
        output_weight: Weight for output modules in layer calculation
        input_fqns: FQN list for input modules (default: ["tok_embeddings"])
        layer_prefix: FQN prefix for transformer layers (default: "layers")
        output_fqns: FQN list for output modules (default: ["norm", "output"])

    Returns:
        List of lists containing module FQN names for each model part

    Example:
        # torchtitan-style flat model:
        generate_llm_fqn_per_model_part(2, 8)
        # -> [["tok_embeddings", "layers.0", ..., "layers.3"],
        #     ["layers.4", ..., "layers.7", "norm", "output"]]

        # HF Qwen3-style nested model:
        generate_llm_fqn_per_model_part(
            2, 8,
            input_fqns=["model.embed_tokens"],
            layer_prefix="model.layers",
            output_fqns=["model.norm", "lm_head"],
        )
    """
    if input_fqns is None:
        input_fqns = ["tok_embeddings"]
    if output_fqns is None:
        output_fqns = ["norm", "output"]

    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        layer_names = [f"{layer_prefix}.{i}" for i in range(num_layers)]
        return [list(input_fqns) + layer_names + list(output_fqns)]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than "
            f"effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average."
        )

    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.extend(input_fqns)
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"{layer_prefix}.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"{layer_prefix}.{current_layer}")
                    current_layer += 1

            stage_modules.extend(output_fqns)

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"{layer_prefix}.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def _pp_forward(self, x):
    """
    PP-compatible forward that accepts/returns raw tensors.

    Replaces the model's original forward so PipelineStage can do P2P
    communication with raw tensors instead of dataclass outputs.

    This is xorl-specific (not in torchtitan) because HF models return
    dataclass outputs, not raw tensors.

    Batch metadata (position_ids, cu_seqlens for flash-attention varlen) is
    passed via a per-microbatch queue ``_pp_batch_metadata`` set by the
    training loop before ``pp_schedule.step()``.  Each forward call pops one
    entry so the schedule's internal microbatch iteration stays in sync.

    When the queue is not set (e.g. unit tests), falls back to generating
    sequential position_ids scaled by cp_size for RoPE cache correctness.
    """
    from .parallel_state import get_parallel_state
    from xorl.models.layers.moe.routing_replay import get_replay_stage, set_replay_stage, is_r3_mode

    ps = get_parallel_state()

    # --- Routing replay stage switch ---
    # Outer scope sets "replay_backward" so checkpoint recompute uses
    # recorded routing.  We temporarily switch to "record" for the actual
    # forward pass, then restore so the subsequent backward/recompute
    # runs with the original stage.
    #
    # IMPORTANT: PipelineStage._shape_inference() runs a full model forward
    # under torch.no_grad() with zero-filled tensors on the first step.
    # We must NOT record routing decisions during shape inference — they
    # would pollute the replay cache with garbage entries, causing
    # pop_backward() to return wrong routing during checkpoint recompute.
    # When grad is disabled (shape inference or eval), set stage to None
    # so MoE blocks use standard routing without recording.
    old_stage = get_replay_stage()
    if not torch.is_grad_enabled():
        # Shape inference or eval — don't record/replay
        set_replay_stage(None)
    elif old_stage == "replay_backward":
        if is_r3_mode():
            set_replay_stage("replay_forward")
        else:
            set_replay_stage("record")

    # --- Pop per-microbatch metadata (set by training loop) ---
    # Skip during shape inference (no_grad) to avoid consuming queue entries
    # that are needed for the actual training forward passes.
    position_ids = None
    extra_kwargs = {}
    if torch.is_grad_enabled():
        metadata_queue = getattr(self, "_pp_batch_metadata", None)
        if metadata_queue:
            metadata = metadata_queue.popleft()
            position_ids = metadata.pop("position_ids", None)
            if position_ids is not None:
                position_ids = position_ids.to(x.device)
            extra_kwargs = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v
                            for k, v in metadata.items()}

    # Fallback: generate sequential position_ids covering the full SP range
    # so that RoPE embeddings have a large enough cache.
    if position_ids is None and ps.cp_size > 1:
        seq_len = x.shape[1]
        full_seq_len = seq_len * ps.cp_size
        position_ids = torch.arange(full_seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    if self._pp_is_first:
        # x is input_ids
        outputs = self._pp_original_forward(
            input_ids=x, position_ids=position_ids,
            use_cache=False, output_hidden_states=False,
            **extra_kwargs,
        )
    else:
        # x is hidden_states from previous stage
        outputs = self._pp_original_forward(
            inputs_embeds=x, position_ids=position_ids,
            use_cache=False, output_hidden_states=False,
            **extra_kwargs,
        )

    # --- Restore routing replay stage ---
    set_replay_stage(old_stage)

    hidden_states = outputs.last_hidden_state

    if self._pp_is_last:
        logits = self.lm_head(hidden_states)
        return logits
    else:
        return hidden_states


def _recursive_prune(module: nn.Module, prefix: str, fqns_to_keep: set):
    """
    Recursively prune modules not in fqns_to_keep.

    Extends torchtitan's flat _build_stage_from_modules to handle
    nested HF model structures (e.g., Qwen3ForCausalLM.model.layers).
    """
    for name, child in list(module.named_children()):
        fqn = f"{prefix}.{name}" if prefix else name

        if isinstance(child, (nn.ModuleDict, nn.ModuleList)):
            # Same as torchtitan: filter children of container modules by FQN
            children_prefix = fqn + "."
            layers_to_keep = {
                fqn_to_keep[len(children_prefix):].split(".")[0]
                for fqn_to_keep in fqns_to_keep
                if fqn_to_keep.startswith(children_prefix)
            }
            if layers_to_keep:
                if isinstance(child, nn.ModuleDict):
                    for layer_name in list(child.keys()):
                        if layer_name not in layers_to_keep:
                            del child[layer_name]
                elif isinstance(child, nn.ModuleList):
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    # Preserve original indices so checkpoint keys match.
                    # Set unwanted entries to None (named_parameters/modules skip None).
                    for i in range(len(child)):
                        if i not in indices_to_keep:
                            child._modules[str(i)] = None
            else:
                # No children needed
                if isinstance(child, nn.ModuleDict):
                    setattr(module, name, nn.ModuleDict())
                elif isinstance(child, nn.ModuleList):
                    setattr(module, name, nn.ModuleList())

        elif fqn in fqns_to_keep:
            # Exact match — keep this module
            pass

        elif any(f.startswith(fqn + ".") for f in fqns_to_keep):
            # Some descendant of this module is needed — recurse
            _recursive_prune(child, fqn, fqns_to_keep)

        else:
            # Not needed — set to None
            setattr(module, name, None)


def pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    device: torch.device,
    module_names_per_stage: List[List[str]],
    always_keep_fqns: Optional[List[str]] = None,
) -> tuple:
    """
    Split a model into pipeline stages based on specified module FQN names.

    Supports GPipe and 1F1B schedules (one stage per rank).

    Ported from torchtitan's pipeline_module_split, extended with:
    - Recursive pruning for nested HF model structures
    - always_keep_fqns for modules needed on every stage (e.g., rotary_emb)
    - Forward patching for HF models (dataclass -> raw tensor I/O)

    Model requirements:
    - forward() method should tolerate deleted (None) layers
    - weight initialization methods should tolerate deleted layers

    Args:
        whole_model: The complete model to be split
        pp_mesh: Pipeline parallel device mesh
        device: Target device
        module_names_per_stage: List of lists of module FQN names per stage
                                (length must equal pp_degree; one stage per rank)
        always_keep_fqns: Module FQNs to keep on every stage (e.g., ["model.rotary_emb"])

    Returns:
        Tuple of (stages, model_parts) where stages are PipelineStage objects
        and model_parts are the corresponding pruned models
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)

    if num_stages != pp_degree:
        raise ValueError(
            f"module_names_per_stage has {num_stages} entries but pp_degree={pp_degree}. "
            f"GPipe and 1F1B require exactly one stage per rank."
        )

    always_keep = set(always_keep_fqns) if always_keep_fqns else set()

    # PP requires untied weights — embed_tokens and lm_head live on different stages
    if getattr(whole_model.config, "tie_word_embeddings", False):
        raise ValueError(
            "Pipeline parallelism requires tie_word_embeddings=False. "
            "embed_tokens and lm_head are on different stages and cannot share weights. "
            "Set tie_word_embeddings: false in the model config."
        )

    module_names = module_names_per_stage[pp_rank]
    model = copy.deepcopy(whole_model)
    fqns_to_keep = set(module_names) | always_keep

    # Recursive pruning handles nested HF model structures
    _recursive_prune(model, "", fqns_to_keep)

    model._pp_is_first = pp_rank == 0
    model._pp_is_last = pp_rank == pp_degree - 1
    model._pp_original_forward = model.forward
    model.forward = types.MethodType(_pp_forward, model)

    stage = PipelineStage(
        model,
        pp_rank,
        num_stages,
        device,
        group=pp_mesh.get_group("pp"),
    )
    logger.info(f"PP rank {pp_rank} built stage {pp_rank} with modules {module_names}")

    return [stage], [model]


def build_pp_stage(
    model_part: nn.Module,
    pp_rank: int,
    num_stages: int,
    device: torch.device,
    pp_group,
) -> PipelineStage:
    """Build a PipelineStage from an existing model chunk (no deepcopy).

    Used to cheaply create a new stage when the sequence length changes
    (pp_variable_seq_lengths=True), since PipelineStage allocates P2P
    buffers based on the first input shape it sees.
    """
    return PipelineStage(model_part, pp_rank, num_stages, device, group=pp_group)


_SUPPORTED_PP_SCHEDULES = {"gpipe", "1f1b"}


def build_pipeline_schedule(
    stages: list,
    n_microbatches: int,
    loss_fn: Callable,
    schedule_name: str = "1F1B",
) -> _PipelineSchedule:
    """
    Build a GPipe or 1F1B pipeline schedule (one stage per rank).

    Args:
        stages: List containing exactly one PipelineStage (one per rank)
        n_microbatches: Number of microbatches
        loss_fn: Loss function (only called on last stage)
        schedule_name: "GPipe" or "1F1B" (case-insensitive)

    Returns:
        Pipeline schedule object
    """
    if schedule_name.lower() not in _SUPPORTED_PP_SCHEDULES:
        raise ValueError(
            f"Unsupported PP schedule '{schedule_name}'. "
            f"Supported schedules: {sorted(_SUPPORTED_PP_SCHEDULES)}"
        )

    schedule_class = get_schedule_class(schedule_name)
    schedule = schedule_class(
        stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )
    logger.info(f"Pipeline schedule: {schedule_name}, n_microbatches={n_microbatches}")
    return schedule
