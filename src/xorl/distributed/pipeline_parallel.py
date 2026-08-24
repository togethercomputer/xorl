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

Virtual stages (stages_per_rank >= 2) place multiple model chunks per rank:
  loop style (Interleaved1F1B, InterleavedZeroBubble): rank r owns stages [r, r+p, r+2p, ...]
  V style (ZBVZeroBubble, DualPipeV):                  rank r owns stages [r, 2p-1-r]
"""

import copy
import types
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from xorl.models.layers.moe.routing_replay import get_replay_stage, is_r3_mode, set_replay_stage

from ..utils import logging
from .parallel_state import get_parallel_state
from .pp_byte_contract import (
    PP_EXACT_REQUIRED_METADATA,
    PPByteContractError,
    assert_pp_wire_dtype,
    engage_pp_byte_contract,
    validate_pp_exact_microbatch_metadata,
    validate_pp_wire_boundary_state,
)


logger = logging.get_logger(__name__)

__all__ = [
    "generate_llm_fqn_from_layer_ranges",
    "generate_llm_fqn_per_model_part",
    "pipeline_module_split",
    "build_pp_stage",
    "build_pipeline_schedule",
    "schedule_stage_style",
    "stage_ids_for_rank",
    "is_single_stage_schedule",
    "schedule_splits_backward",
    "validate_pp_schedule_config",
]


def _deepcopy_pipeline_model(model: nn.Module) -> nn.Module:
    """Clone a model while preserving its rank-local process-group handles.

    Exact attention and vocabulary-head modules bind their CP/TP groups during
    construction.  ``ProcessGroup`` objects are immutable rank-local runtime
    handles and cannot be pickled, so a plain ``copy.deepcopy`` fails before
    PP pruning.  Seeding the deepcopy memo keeps only those handles shared;
    parameters, buffers, modules, and ordinary Python state are still cloned.
    """

    memo: dict[int, object] = {}
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, torch.distributed.ProcessGroup):
                memo[id(value)] = value
    cloned = copy.deepcopy(model, memo)

    # ``nn.Parameter.__deepcopy__`` clones tensor storage and requires-grad but
    # intentionally drops the parameter's Python ``__dict__``. XoRL records
    # numerical/ownership contracts there (for example ``_keep_fp32`` and
    # ``spec_info``), so silently losing it changes the PP program. Reattach
    # the rank-local metadata to the already-cloned parameter objects by FQN.
    source_parameters = dict(model.named_parameters(remove_duplicate=False))
    cloned_parameters = dict(cloned.named_parameters(remove_duplicate=False))
    if source_parameters.keys() != cloned_parameters.keys():
        raise RuntimeError("pipeline deepcopy changed the model parameter inventory")
    for name, source_parameter in source_parameters.items():
        cloned_parameters[name].__dict__.update(source_parameter.__dict__)
    return cloned


def generate_llm_fqn_from_layer_ranges(
    layer_ranges,
    *,
    num_layers: int,
    input_fqns: Optional[List[str]] = None,
    layer_prefix: str = "layers",
    output_fqns: Optional[List[str]] = None,
) -> List[List[str]]:
    """Build stage module assignments from model-derived legal ranges."""

    input_fqns = ["tok_embeddings"] if input_fqns is None else list(input_fqns)
    output_fqns = ["norm", "output"] if output_fqns is None else list(output_fqns)
    ranges = tuple((int(start), int(end)) for start, end in layer_ranges)
    expected_start = 0
    for start, end in ranges:
        if start != expected_start or end <= start:
            raise ValueError("Pipeline layer ranges must be nonempty, contiguous, and start at layer 0")
        expected_start = end
    if not ranges or expected_start != num_layers:
        raise ValueError(f"Pipeline layer ranges must cover exactly {num_layers} decoder layers")

    assignments: List[List[str]] = []
    for stage_index, (start, end) in enumerate(ranges):
        modules: List[str] = []
        if stage_index == 0:
            modules.extend(input_fqns)
        modules.extend(f"{layer_prefix}.{layer_index}" for layer_index in range(start, end))
        if stage_index == len(ranges) - 1:
            modules.extend(output_fqns)
        assignments.append(modules)
    return assignments


# Schedules taking exactly one stage per rank use style "single"; multi-stage
# schedules place chunks in "loop" or "v" order (must match torch's
# generate_stage_to_rank_mapping so schedule-side P2P routing agrees).
_SCHEDULE_STYLES = {
    "gpipe": "single",
    "1f1b": "single",
    "interleaved1f1b": "loop",
    "interleavedzerobubble": "loop",
    "zbvzerobubble": "v",
    "dualpipev": "v",
}


def schedule_stage_style(schedule_name: str) -> str:
    """Return the stage-placement style ('single', 'loop', or 'v') for a schedule."""
    key = schedule_name.lower()
    if key not in _SCHEDULE_STYLES:
        raise ValueError(f"Unsupported PP schedule '{schedule_name}'. Supported schedules: {sorted(_SCHEDULE_STYLES)}")
    return _SCHEDULE_STYLES[key]


def is_single_stage_schedule(schedule_name: str) -> bool:
    """True when the schedule requires exactly one stage per rank (GPipe, 1F1B)."""
    return issubclass(get_schedule_class(schedule_name), PipelineScheduleSingle)


_BACKWARD_SPLIT_SCHEDULES = frozenset({"interleavedzerobubble", "zbvzerobubble", "dualpipev"})


def schedule_splits_backward(schedule_name: str) -> bool:
    """True for zero-bubble schedules that run separate dX (input) and dW (weight) backward passes."""
    return schedule_name.lower() in _BACKWARD_SPLIT_SCHEDULES


def stage_ids_for_rank(pp_rank: int, pp_size: int, num_stages: int, style: str) -> List[int]:
    """Global stage indices owned by ``pp_rank``, matching torch's stage-to-rank mapping.

    loop: rank r owns [r, r+p, r+2p, ...] (Interleaved1F1B, InterleavedZeroBubble).
    v:    zigzag placement; for 2 stages/rank, rank r owns [r, 2p-1-r] (ZBV, DualPipeV).
    """
    if num_stages % pp_size != 0:
        raise ValueError(f"num_stages ({num_stages}) must be divisible by pp_size ({pp_size})")
    if style in ("single", "loop"):
        if style == "single" and num_stages != pp_size:
            raise ValueError(f"single-stage schedules require num_stages == pp_size, got {num_stages} != {pp_size}")
        return list(range(pp_rank, num_stages, pp_size))
    if style == "v":
        # Mirror torch.distributed.pipelining._utils.generate_stage_to_rank_mapping(style="v").
        stage_ids = []
        rank_index = 0
        for stage_index in range(num_stages):
            if rank_index == pp_rank:
                stage_ids.append(stage_index)
            if (stage_index + 1) % pp_size == 0:
                continue
            rank_index += 1 if (stage_index // pp_size) % 2 == 0 else -1
        return stage_ids
    raise ValueError(f"Unknown stage placement style: {style!r}")


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
    input_fqns: Optional[List[str]] = None,
    layer_prefix: str = "layers",
    output_fqns: Optional[List[str]] = None,
    num_layers_in_first_stage: Optional[int] = None,
    num_layers_in_last_stage: Optional[int] = None,
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
        num_layers_in_first_stage: Explicit decoder-layer count for stage 0
            (Megatron-style override; takes precedence over input_weight)
        num_layers_in_last_stage: Explicit decoder-layer count for the last stage
            (takes precedence over output_weight)

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

    # Explicit first/last-stage layer counts (Megatron-style) take precedence
    # over the weight heuristic; remaining layers are split evenly across the
    # unpinned stages (remainder to earlier stages).
    if num_layers_in_first_stage is not None or num_layers_in_last_stage is not None:
        pinned_first = num_layers_in_first_stage is not None
        pinned_last = num_layers_in_last_stage is not None
        pinned_total = (num_layers_in_first_stage or 0) + (num_layers_in_last_stage or 0)
        free_stages = num_stages - int(pinned_first) - int(pinned_last)
        free_layers = num_layers - pinned_total
        if free_layers < 0 or (free_stages == 0 and free_layers != 0):
            raise ValueError(
                f"num_layers_in_first_stage={num_layers_in_first_stage} + "
                f"num_layers_in_last_stage={num_layers_in_last_stage} incompatible with "
                f"num_layers={num_layers} across {num_stages} stages."
            )
        if free_stages > 0 and free_layers < free_stages:
            raise ValueError(
                f"Explicit stage layer counts leave {free_layers} layers for {free_stages} unpinned stages; "
                f"each stage needs at least one layer."
            )
        layer_counts = []
        base, extra = (free_layers // free_stages, free_layers % free_stages) if free_stages else (0, 0)
        free_idx = 0
        for stage_idx in range(num_stages):
            if stage_idx == 0 and pinned_first:
                layer_counts.append(num_layers_in_first_stage)
            elif stage_idx == num_stages - 1 and pinned_last:
                layer_counts.append(num_layers_in_last_stage)
            else:
                layer_counts.append(base + (1 if free_idx < extra else 0))
                free_idx += 1
        module_names_per_stage = []
        current_layer = 0
        for stage_idx, count in enumerate(layer_counts):
            stage_modules = []
            if stage_idx == 0:
                stage_modules.extend(input_fqns)
            stage_modules.extend(f"{layer_prefix}.{i}" for i in range(current_layer, current_layer + count))
            current_layer += count
            if stage_idx == num_stages - 1:
                stage_modules.extend(output_fqns)
            module_names_per_stage.append(stage_modules)
        return module_names_per_stage

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
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
        raise ValueError(f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage}).")
    if output_weight > layers_per_stage:
        raise ValueError(f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage}).")

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
    # A scheduled forward is either grad-enabled (training) or explicitly
    # flagged forward-only (``_pp_forward_only``, set by forward_only_pp for
    # eval/ref-logprob passes, whose stages skip shape inference via meta IO
    # args); anything else is shape inference — no recording, no metadata.
    in_scheduled_forward = torch.is_grad_enabled() or getattr(self, "_pp_forward_only", False)
    old_stage = get_replay_stage()
    if not in_scheduled_forward:
        # Shape inference — don't record/replay
        set_replay_stage(None)
    elif old_stage == "replay_backward":
        if is_r3_mode():
            set_replay_stage("replay_forward")
        else:
            set_replay_stage("record")

    # --- Pop per-microbatch metadata (set by training loop) ---
    # Skip during shape inference to avoid consuming queue entries that are
    # needed for the actual scheduled forward passes.
    carries_hyperconnection_state = getattr(self, "_pp_carries_hyperconnection_state", False)
    position_ids = None
    original_input_ids = None
    extra_kwargs = {}
    metadata_queue = getattr(self, "_pp_batch_metadata", None)
    if metadata_queue and (in_scheduled_forward or carries_hyperconnection_state):
        # DSV4's PP shape is its storage-capacity 4-D state with live rows in
        # a compact prefix. Shape inference still needs the first microbatch's
        # immutable row plan. Peek without consuming; the scheduled forward
        # still pops the same entry.
        metadata = metadata_queue.popleft() if in_scheduled_forward else dict(metadata_queue[0])
        position_ids = metadata.pop("position_ids", None)
        original_input_ids = metadata.pop("_pp_original_input_ids", None)
        if original_input_ids is None:
            # Compatibility with batches queued by the generic exact-PP path.
            original_input_ids = metadata.pop("_pp_input_ids", None)
        else:
            metadata.pop("_pp_input_ids", None)
        if position_ids is not None:
            position_ids = position_ids.to(x.device)
        if original_input_ids is not None:
            original_input_ids = original_input_ids.to(x.device)
        extra_kwargs = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in metadata.items()}
    if in_scheduled_forward:
        if getattr(self, "_pp_exact_boundary_contract", False):
            # Exact lanes fail closed on INCOMPLETE metadata, not just absent
            # position_ids: a fabricated positional fallback changes bytes
            # silently, and a missing cu_seq_lens_* silently
            # merges packed documents into one attention span — a silent
            # numerics change with no downstream error.
            missing = [
                name
                for name in PP_EXACT_REQUIRED_METADATA
                if (position_ids is None if name == "position_ids" else name not in extra_kwargs)
            ]
            if missing:
                raise PPByteContractError(
                    f"PP byte contract: scheduled stage forward is missing required per-microbatch "
                    f"varlen metadata {missing}; exact value programs require the complete set "
                    f"{list(PP_EXACT_REQUIRED_METADATA)} (silent fallbacks/merges are not admitted)"
                )
            # Presence is not enough: a present-but-None (or malformed)
            # cu_seq_lens_* still reaches the single-document fallback.
            validate_pp_exact_microbatch_metadata(
                x,
                position_ids,
                extra_kwargs,
                sequence_parallel_size=ps.cp_size,
            )
            if not self._pp_is_first:
                # The received wire bytes must be bf16 regardless of what any
                # config declared (resolved reality, not metadata).
                assert_pp_wire_dtype(x, where="received inter-stage hidden state")
        if not self._pp_is_first:
            validate_pp_wire_boundary_state(
                x,
                getattr(self, "_pp_pipeline_boundary_state", None),
                where="received inter-stage hidden state",
            )

    # Fallback: generate sequential position_ids covering the full SP range
    # so that RoPE embeddings have a large enough cache.
    if position_ids is None and ps.cp_size > 1:
        seq_len = x.shape[1]
        full_seq_len = seq_len * ps.cp_size
        position_ids = torch.arange(full_seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    if carries_hyperconnection_state:
        extra_kwargs["_pp_stage_is_first"] = self._pp_is_first
        extra_kwargs["_pp_stage_is_last"] = self._pp_is_last

    if self._pp_is_first:
        # x is input_ids
        outputs = self._pp_original_forward(
            input_ids=x,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            **extra_kwargs,
        )
    else:
        # x is hidden_states from previous stage
        requires_original_ids = bool(
            getattr(self, "_pp_requires_original_input_ids", False)
            or getattr(self, "_pp_requires_input_ids_on_all_stages", False)
        )
        if requires_original_ids and original_input_ids is None:
            if in_scheduled_forward:
                raise PPByteContractError(
                    "PP stage requires original input_ids for layer-local routing, but the microbatch metadata omitted them"
                )
            # Shape inference has no real microbatch metadata. Token values do
            # not affect its shape contract, so provide a shape-matched dummy.
            original_input_ids = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        outputs = self._pp_original_forward(
            input_ids=(original_input_ids if requires_original_ids else None),
            inputs_embeds=x,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            **extra_kwargs,
        )

    # --- Restore routing replay stage ---
    set_replay_stage(old_stage)

    hidden_states = outputs.last_hidden_state

    if getattr(self, "_pp_exact_boundary_contract", False) and in_scheduled_forward:
        # The emitted wire bytes must be bf16 regardless of what any config
        # declared: a bf16-declared model computing in fp32 is caught here at
        # its first scheduled forward (resolved reality, not metadata).
        assert_pp_wire_dtype(hidden_states, where="emitted stage hidden state")
        if not self._pp_is_last:
            validate_pp_wire_boundary_state(
                hidden_states,
                getattr(self, "_pp_pipeline_boundary_state", None),
                where="emitted inter-stage hidden state",
            )
    elif in_scheduled_forward and not self._pp_is_last:
        validate_pp_wire_boundary_state(
            hidden_states,
            getattr(self, "_pp_pipeline_boundary_state", None),
            where="emitted inter-stage hidden state",
        )

    if self._pp_is_last:
        # When the loss fn applies lm_head (fused quack_linear CE), return HIDDEN
        # so the last stage never materializes the full [mbs, seq, 248k] logits
        # (which OOMs). Otherwise project to logits here for the logit-based
        # (compiled/eager) PP loss fns.
        if getattr(self, "_pp_lm_head_in_loss", False):
            return hidden_states
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
                fqn_to_keep[len(children_prefix) :].split(".")[0]
                for fqn_to_keep in fqns_to_keep
                if fqn_to_keep.startswith(children_prefix)
            }
            if layers_to_keep:
                if isinstance(child, nn.ModuleDict):
                    for layer_name in list(child.keys()):
                        if layer_name not in layers_to_keep:
                            del child[layer_name]
                elif isinstance(child, nn.ModuleList):
                    indices_to_keep = {int(idx) for idx in layers_to_keep if idx.isdigit()}
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
    stage_style: str = "single",
) -> tuple:
    """
    Split a model into pipeline stages based on specified module FQN names.

    Supports one stage per rank (GPipe, 1F1B) and virtual stages
    (stages_per_rank >= 2) with loop-style (Interleaved1F1B,
    InterleavedZeroBubble) or V-style (ZBVZeroBubble, DualPipeV) placement.

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
                                (length = pp_degree * stages_per_rank)
        always_keep_fqns: Module FQNs to keep on every stage (e.g., ["model.rotary_emb"])
        stage_style: Stage placement ('single', 'loop', or 'v'); must match the
                     schedule this rank's stages will run under

    Returns:
        Tuple of (stages, model_parts) where stages are PipelineStage objects
        and model_parts are the corresponding pruned models, ordered by this
        rank's stage ids as produced by ``stage_ids_for_rank``
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)

    if num_stages % pp_degree != 0:
        raise ValueError(f"module_names_per_stage has {num_stages} entries, not divisible by pp_degree={pp_degree}.")
    if stage_style == "single" and num_stages != pp_degree:
        raise ValueError(
            f"module_names_per_stage has {num_stages} entries but pp_degree={pp_degree}. "
            f"GPipe and 1F1B require exactly one stage per rank; use an interleaved/ZB "
            f"schedule for virtual stages."
        )

    always_keep = set(always_keep_fqns) if always_keep_fqns else set()

    # PP requires untied weights — embed_tokens and lm_head live on different stages
    if getattr(whole_model.config, "tie_word_embeddings", False):
        raise ValueError(
            "Pipeline parallelism requires tie_word_embeddings=False. "
            "embed_tokens and lm_head are on different stages and cannot share weights. "
            "Set tie_word_embeddings: false in the model config."
        )

    stage_ids = stage_ids_for_rank(pp_rank, pp_degree, num_stages, stage_style)

    stages, model_parts = [], []
    for stage_idx in stage_ids:
        module_names = module_names_per_stage[stage_idx]
        model = _deepcopy_pipeline_model(whole_model)
        fqns_to_keep = set(module_names) | always_keep

        # Recursive pruning handles nested HF model structures
        _recursive_prune(model, "", fqns_to_keep)
        configure_stage = getattr(model, "_configure_pp_stage", None)
        if callable(configure_stage):
            configure_stage(stage_idx=stage_idx, num_stages=num_stages)

        model._pp_is_first = stage_idx == 0
        model._pp_is_last = stage_idx == num_stages - 1
        model._pp_stage_idx = stage_idx
        model._pp_original_forward = model.forward
        decoder = getattr(model, "model", model)
        decoder_layers = getattr(decoder, "layers", None)
        owns_decoder_layers = decoder_layers is None or any(layer is not None for layer in decoder_layers)
        model._pp_requires_index_share_mode = bool(
            owns_decoder_layers and callable(getattr(model, "release_index_share_context", None))
        )
        model.forward = types.MethodType(_pp_forward, model)

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        logger.info(f"PP rank {pp_rank} built stage {stage_idx} with modules {module_names}")
        stages.append(stage)
        model_parts.append(model)

    # Validate the actual split structure. Exact programs additionally receive
    # fail-closed metadata and runtime wire checks; family names never admit a
    # topology.
    engage_pp_byte_contract(
        whole_model,
        module_names_per_stage=module_names_per_stage,
        stage_ids=stage_ids,
        model_parts=model_parts,
        parallel_state=get_parallel_state(),
    )

    return stages, model_parts


def build_pp_stage(
    model_part: nn.Module,
    stage_index: int,
    num_stages: int,
    device: torch.device,
    pp_group,
    input_args=None,
    output_args=None,
) -> PipelineStage:
    """Build a PipelineStage from an existing model chunk (no deepcopy).

    Used to cheaply create a new stage when the sequence length changes
    (pp_variable_seq_lengths=True), since PipelineStage allocates P2P
    buffers based on the first input shape it sees.

    When ``input_args`` AND ``output_args`` (example/meta tensors) are both
    provided, PipelineStage SKIPS its init-time ``_shape_inference`` forward.
    This is required for sequence-parallel (Ulysses) models: the shape-inference
    dummy forward would otherwise run the intra-stage CP collectives, which
    deadlock against the cross-stage shape-exchange P2P. Output shapes are
    derivable from (mbs, seq_len, hidden/vocab), so the forward is unnecessary.

    Manual ``PipelineStage`` treats the supplied examples as the authoritative
    static autograd metadata.  Activation examples are usually created with
    ``torch.empty(..., device="meta")``, whose default ``requires_grad=False``
    would make received activations leaf tensors that never accumulate the
    gradient to send to the preceding stage.  Mark the differentiable sides of
    each inter-stage boundary explicitly; root token IDs remain integral and
    non-differentiable.
    """
    if stage_index > 0 and input_args is not None:
        input_args = tuple(
            value.requires_grad_(True)
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())
            else value
            for value in input_args
        )
    if stage_index < num_stages - 1 and output_args is not None:
        output_args = tuple(
            value.requires_grad_(True)
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())
            else value
            for value in output_args
        )
    return PipelineStage(
        model_part,
        stage_index,
        num_stages,
        device,
        input_args=input_args,
        output_args=output_args,
        group=pp_group,
    )


_SUPPORTED_PP_SCHEDULES = frozenset(_SCHEDULE_STYLES)


def validate_pp_schedule_config(schedule_name: str, virtual_stages: int, n_microbatches: int, pp_size: int) -> None:
    """Fail fast on schedule/virtual-stage/microbatch combinations torch would reject at runtime."""
    style = schedule_stage_style(schedule_name)
    if style == "single" and virtual_stages != 1:
        raise ValueError(
            f"Schedule '{schedule_name}' supports exactly one stage per rank; "
            f"got pipeline_parallel_virtual_stages={virtual_stages}. "
            f"Use Interleaved1F1B/InterleavedZeroBubble (any) or ZBVZeroBubble/DualPipeV (=2)."
        )
    if style == "v" and virtual_stages != 2:
        raise ValueError(
            f"Schedule '{schedule_name}' requires exactly 2 stages per rank "
            f"(pipeline_parallel_virtual_stages=2), got {virtual_stages}."
        )
    if style == "loop":
        # Interleaved schedules round microbatches over the PP group.
        number_of_rounds = max(1, n_microbatches // pp_size)
        if n_microbatches % number_of_rounds != 0:
            raise ValueError(
                f"Schedule '{schedule_name}' requires n_microbatches ({n_microbatches}) to be a multiple of "
                f"the number of rounds ({number_of_rounds} = max(1, n_microbatches // pp_size))."
            )
    if schedule_name.lower() == "dualpipev" and n_microbatches < pp_size * virtual_stages:
        raise ValueError(
            f"DualPipeV requires n_microbatches >= num_stages ({pp_size * virtual_stages}), got {n_microbatches}."
        )


def build_pipeline_schedule(
    stages: list,
    n_microbatches: int,
    loss_fn: Callable,
    schedule_name: str = "1F1B",
) -> _PipelineSchedule:
    """
    Build a pipeline schedule over this rank's stage(s).

    Args:
        stages: This rank's PipelineStage list — exactly one for GPipe/1F1B,
                stages_per_rank entries (stage-id order) for multi-stage schedules
        n_microbatches: Number of microbatches
        loss_fn: Loss function (only called on the stage where is_last holds)
        schedule_name: One of GPipe, 1F1B, Interleaved1F1B, InterleavedZeroBubble,
                       ZBVZeroBubble, DualPipeV (case-insensitive)

    Returns:
        Pipeline schedule object
    """
    if schedule_name.lower() not in _SUPPORTED_PP_SCHEDULES:
        raise ValueError(
            f"Unsupported PP schedule '{schedule_name}'. Supported schedules: {sorted(_SUPPORTED_PP_SCHEDULES)}"
        )

    schedule_class = get_schedule_class(schedule_name)
    single = issubclass(schedule_class, PipelineScheduleSingle)
    if single and len(stages) != 1:
        raise ValueError(f"Schedule '{schedule_name}' takes exactly one stage per rank, got {len(stages)}.")
    schedule = schedule_class(
        stages[0] if single else stages,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )
    logger.info(f"Pipeline schedule: {schedule_name}, n_microbatches={n_microbatches}, stages_per_rank={len(stages)}")
    return schedule
