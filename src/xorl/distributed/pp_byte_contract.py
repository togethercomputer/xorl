"""Structural boundary checks for pipeline-parallel model splits.

The exact train/serve lanes promise forward logit-path bytes identical to the
unpartitioned (PP1) program, which itself matches the sampler. Cutting the
trainer at a pipeline-stage boundary is admitted only when every required
boundary invariant is either bitwise by construction or guarded here:

- the stage plan may cut only at decoder boundaries, with contiguous global
  layer coverage, embeddings on stage 0 only, and the final norm and head
  together on the last stage. Module-only edge stages are valid cuts before
  layer 0 or after the final layer;
- pruning must preserve global ``ModuleList`` indices and, when exposed,
  ``layer_idx``/``layer_id`` so
  layer-position-keyed kernel selection (for example the Qwen3.5 layer-0
  vs layer>0 input-norm families) cannot silently flip at a cut;
- validated model parts are marked so ``_pp_forward`` refuses to fabricate
  per-microbatch metadata (silent positional fallback changes bytes with no
  error).

Family labels are diagnostic only.  Admission follows the model's actual
module split, live process groups, metadata, and boundary tensor state.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from xorl.models.exact_contract import resolve_exact_contract_family

from ..utils import logging


logger = logging.get_logger(__name__)

__all__ = [
    "PP_EXACT_REQUIRED_METADATA",
    "PPByteContractError",
    "assert_pp_wire_dtype",
    "engage_pp_byte_contract",
    "exact_contract_family",
    "validate_pp_wire_boundary_state",
    "validate_pp_exact_microbatch_metadata",
]


class PPByteContractError(RuntimeError):
    """A pipeline configuration would break the exact train/serve byte contract."""


#: Per-microbatch metadata a contract-marked stage forward must receive.
#: The exact lanes run packed varlen batches; a missing cu_seq_lens_* does
#: not error downstream — the attention entry silently treats the pack as
#: ONE document, introducing cross-document attention (a silent numerics
#: change, the worst failure class). Marked parts therefore RAISE unless the
#: complete set is present.
PP_EXACT_REQUIRED_METADATA = (
    "position_ids",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "max_length_q",
    "max_length_k",
)


_UNSTAMPED = object()


def exact_contract_family(config: object | None) -> Optional[str]:
    """Classify the exact value program selected by ``config``.

    Returns ``None`` for generic models, otherwise the resolved exact-program
    identifier used for diagnostics and exact runtime behavior.

    Model resolution stamps ``config._exact_contract_family`` once; when the
    stamp is present it is authoritative (including a stamped ``None`` for
    generic models). Configs that predate stamping (for example direct
    construction in tests) fall back to the shared legacy-flag resolver.
    Pipeline admission is structural and never branches on this name.
    """

    if config is None:
        return None
    stamped = getattr(config, "_exact_contract_family", _UNSTAMPED)
    if stamped is not _UNSTAMPED:
        return stamped
    return resolve_exact_contract_family(config)


def _stage_layer_indices(module_names: Sequence[str], layer_prefix: str) -> List[int]:
    prefix = layer_prefix + "."
    indices = []
    for name in module_names:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                indices.append(int(suffix))
    return indices


def _validate_stage_plan(
    module_names_per_stage: Sequence[Sequence[str]],
    *,
    input_fqns: Sequence[str],
    layer_prefix: str,
    output_fqns: Sequence[str],
    num_layers: int,
) -> List[List[int]]:
    """Admit contiguous decoder cuts, including module-only edge stages."""

    num_stages = len(module_names_per_stage)
    per_stage_layers: List[List[int]] = []
    covered: List[int] = []
    for stage_idx, module_names in enumerate(module_names_per_stage):
        names = set(module_names)
        layers = _stage_layer_indices(module_names, layer_prefix)
        if not layers and stage_idx not in (0, num_stages - 1):
            raise PPByteContractError(
                f"PP byte contract: middle stage {stage_idx} owns no decoder layers; only the input edge "
                "before layer 0 and the output edge after the final layer may be module-only"
            )
        if layers and sorted(layers) != list(range(min(layers), max(layers) + 1)):
            raise PPByteContractError(f"PP byte contract: stage {stage_idx} layers {sorted(layers)} are not contiguous")
        for fqn in input_fqns:
            if stage_idx == 0 and fqn not in names:
                raise PPByteContractError(f"PP byte contract: input module {fqn!r} must live on stage 0 (missing)")
            if stage_idx > 0 and fqn in names:
                raise PPByteContractError(
                    f"PP byte contract: input module {fqn!r} may live only on stage 0, found on stage {stage_idx}"
                )
        for fqn in output_fqns:
            if stage_idx == num_stages - 1 and fqn not in names:
                raise PPByteContractError(
                    f"PP byte contract: output module {fqn!r} must live on the last stage; the final "
                    f"norm and head form one rounding boundary with the layer stack and may not be split off"
                )
            if stage_idx < num_stages - 1 and fqn in names:
                raise PPByteContractError(
                    f"PP byte contract: output module {fqn!r} may live only on the last stage, "
                    f"found on stage {stage_idx}"
                )
        per_stage_layers.append(sorted(layers))
        covered.extend(sorted(layers))
    if covered != list(range(num_layers)):
        raise PPByteContractError(
            f"PP byte contract: stage plans must cover decoder layers 0..{num_layers - 1} exactly once "
            f"in ascending stage order; got {covered}"
        )
    return per_stage_layers


def _validate_part_layer_identity(
    model_part: nn.Module,
    stage_idx: int,
    expected_layers: Sequence[int],
    layer_prefix: str,
) -> None:
    """Pruning must preserve global layer indices and any model index field."""

    try:
        container = model_part.get_submodule(layer_prefix)
    except AttributeError as exc:
        raise PPByteContractError(
            f"PP byte contract: stage {stage_idx} part has no layer container {layer_prefix!r}: {exc}"
        ) from exc
    kept = [(idx, layer) for idx, layer in enumerate(container) if layer is not None]
    kept_indices = [idx for idx, _ in kept]
    if kept_indices != list(expected_layers):
        raise PPByteContractError(
            f"PP byte contract: stage {stage_idx} kept layer indices {kept_indices} do not match the "
            f"stage plan {list(expected_layers)}; global ModuleList index preservation is required so "
            f"layer-position-keyed kernel selection cannot flip at a cut"
        )
    for idx, layer in kept:
        layer_idx = getattr(layer, "layer_idx", getattr(layer, "layer_id", None))
        if layer_idx is None:
            continue
        if int(layer_idx) != idx:
            raise PPByteContractError(
                f"PP byte contract: stage {stage_idx} layer {layer_prefix}.{idx} carries layer_idx="
                f"{layer_idx}; layer-position-keyed behavior would diverge from the PP1 program"
            )


def _validate_pipeline_ranges(
    per_stage_layers: Sequence[Sequence[int]],
    exact_ranges: Sequence[Sequence[int]] | None,
) -> None:
    """Validate model-owned state boundaries when the model declares them."""

    if exact_ranges is None:
        return
    normalized = tuple((int(start), int(end)) for start, end in exact_ranges)
    # Embedding-only/head-only stages do not own model state keyed by decoder
    # ranges. Compare the declared boundaries to the nonempty decoder stages;
    # edge stages remain legal so long as the decoder-owned state is neither
    # split nor stranded.
    actual = tuple((layers[0], layers[-1] + 1) for layers in per_stage_layers if layers)
    if actual != normalized:
        raise PPByteContractError(
            f"PP byte contract: model-derived legal decoder ranges are {normalized}, got {actual}; "
            "the cut would strand model-owned state outside the residual-stream wire"
        )


def _validate_stage_local_groups(parallel_state: object | None) -> None:
    """Require active EP/head groups to stay inside the live PP mesh slice."""

    if parallel_state is None or not torch.distributed.is_initialized():
        return
    device_mesh = getattr(parallel_state, "device_mesh", None)
    if device_mesh is None:
        return
    mesh = device_mesh.mesh
    dim_names = tuple(device_mesh.mesh_dim_names)
    coordinate = device_mesh.get_coordinate()
    if coordinate is None:
        raise PPByteContractError("PP byte contract: current rank has no live device-mesh coordinate")
    if "pp" in dim_names:
        pp_dim = dim_names.index("pp")
        stage_ranks = {int(rank) for rank in mesh.select(pp_dim, int(coordinate[pp_dim])).reshape(-1).tolist()}
    else:
        stage_ranks = {int(rank) for rank in mesh.reshape(-1).tolist()}

    for name in ("ep_group", "lm_head_tp_group", "lm_head_tp_replica_group"):
        try:
            group = getattr(parallel_state, name)
        except (AttributeError, RuntimeError):
            group = None
        if group is None:
            continue
        group_ranks = {int(rank) for rank in torch.distributed.get_process_group_ranks(group)}
        if not group_ranks.issubset(stage_ranks):
            raise PPByteContractError(
                f"PP byte contract: live {name} ranks {sorted(group_ranks)} cross the current "
                f"pipeline stage {sorted(stage_ranks)}"
            )


def validate_pp_exact_microbatch_metadata(x: torch.Tensor, position_ids, extra_kwargs: dict) -> None:
    """Value-validate the per-microbatch varlen metadata of a marked stage.

    Key presence is not enough: ``cu_seq_lens_q=None`` (or a malformed
    tensor) passes a presence check and still lets the attention entry treat
    a packed batch as one document. RAISES naming the offending field.
    """
    total_tokens = int(x.shape[0]) * int(x.shape[1])

    def _fail(name: str, why: str) -> None:
        raise PPByteContractError(
            f"PP byte contract: per-microbatch metadata {name!r} is invalid: {why}; exact value "
            f"programs require well-formed varlen metadata (silent document merges are not admitted)"
        )

    if not isinstance(position_ids, torch.Tensor):
        _fail("position_ids", f"expected an integer tensor, got {type(position_ids).__name__}")
    if position_ids.dtype not in (torch.int32, torch.int64):
        _fail("position_ids", f"expected int32/int64, got {position_ids.dtype}")
    if position_ids.numel() < total_tokens:
        _fail(
            "position_ids",
            f"covers {position_ids.numel()} positions for {total_tokens} tokens",
        )

    spans = {}
    for name in ("cu_seq_lens_q", "cu_seq_lens_k"):
        value = extra_kwargs.get(name)
        if not isinstance(value, torch.Tensor):
            _fail(name, f"expected an int32 tensor, got {type(value).__name__}")
        if value.dtype != torch.int32:
            _fail(name, f"expected int32 (flash varlen contract), got {value.dtype}")
        if value.ndim != 1 or value.numel() < 2:
            _fail(name, f"expected 1-D with >= 2 boundaries, got shape {tuple(value.shape)}")
        if int(value[0]) != 0:
            _fail(name, f"first boundary must be 0, got {int(value[0])}")
        if int(value[-1]) != total_tokens:
            _fail(name, f"last boundary must equal total tokens {total_tokens}, got {int(value[-1])}")
        if bool((value[1:] <= value[:-1]).any()):
            _fail(name, f"boundaries must be strictly increasing, got {value.tolist()}")
        spans[name] = value
    if not torch.equal(spans["cu_seq_lens_q"], spans["cu_seq_lens_k"]):
        _fail("cu_seq_lens_k", "q/k document spans differ; the exact self-attention lane requires equal spans")

    max_span = int((spans["cu_seq_lens_q"][1:] - spans["cu_seq_lens_q"][:-1]).max())
    for name in ("max_length_q", "max_length_k"):
        value = extra_kwargs.get(name)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                _fail(name, f"expected a scalar, got shape {tuple(value.shape)}")
            value = int(value.item())
        if not isinstance(value, int):
            _fail(name, f"expected an int, got {type(value).__name__}")
        if value < max_span:
            _fail(name, f"{value} is smaller than the longest document span {max_span} (kernel undersizing)")


def assert_pp_wire_dtype(tensor: torch.Tensor, *, where: str) -> None:
    """The wire reality check: marked stages move bf16 bytes, whatever was declared."""
    if tensor.dtype is not torch.bfloat16:
        raise PPByteContractError(
            f"PP byte contract: {where} carries dtype {tensor.dtype}, but the contract wire is "
            f"bfloat16 (declared=torch.bfloat16, actual={tensor.dtype}); the declaration does not "
            f"override the resolved reality"
        )


def validate_pp_wire_boundary_state(tensor: torch.Tensor, boundary_state: dict | None, *, where: str) -> None:
    """Validate the runtime dtype, rank, and suffix of a nonstandard PP state."""

    if boundary_state is None:
        return
    expected_rank = boundary_state["rank"]
    expected_suffix = boundary_state["shape_suffix"]
    expected_dtype = boundary_state["dtype"]
    if (
        tensor.dtype is not expected_dtype
        or tensor.ndim != expected_rank
        or tuple(tensor.shape[-len(expected_suffix) :]) != expected_suffix
    ):
        raise PPByteContractError(
            f"PP byte contract: {where} for state {boundary_state['state']!r} must have dtype "
            f"{expected_dtype}, rank {expected_rank}, and shape suffix {expected_suffix}; got "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}"
        )


def _normalize_pipeline_boundary_state(value: object | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PPByteContractError("PP byte contract: pipeline_boundary_state must be a mapping")
    required = {"rank", "dtype", "shape_suffix", "state"}
    if set(value) != required:
        raise PPByteContractError(
            f"PP byte contract: pipeline_boundary_state fields must be {sorted(required)}, got {sorted(value)}"
        )
    rank = value["rank"]
    suffix = value["shape_suffix"]
    state = value["state"]
    dtype = value["dtype"]
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise PPByteContractError("PP byte contract: boundary rank must be a positive integer")
    if not isinstance(dtype, torch.dtype):
        raise PPByteContractError("PP byte contract: boundary dtype must name a torch dtype")
    if (
        not isinstance(suffix, (tuple, list))
        or not suffix
        or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in suffix)
    ):
        raise PPByteContractError("PP byte contract: boundary shape_suffix must contain positive integers")
    if len(suffix) > rank or not isinstance(state, str) or not state:
        raise PPByteContractError("PP byte contract: boundary state descriptor is invalid")
    return {"rank": rank, "dtype": dtype, "shape_suffix": tuple(suffix), "state": state}


def engage_pp_byte_contract(
    whole_model: nn.Module,
    module_names_per_stage: Sequence[Sequence[str]],
    stage_ids: Sequence[int],
    model_parts: Sequence[nn.Module],
    *,
    parallel_state: object | None = None,
) -> None:
    """Validate a model-derived pipeline split and mark exact runtime checks.

    Called from ``pipeline_module_split`` after pruning. Raises
    ``PPByteContractError`` when the split cannot preserve byte identity with
    the PP1 program; logs one engagement line per local stage when it can.
    """

    config = getattr(whole_model, "config", None)
    family = exact_contract_family(config)
    exact_runtime = family is not None or bool(getattr(config, "_dsv4_flash_exact_mode", False))
    diagnostic_family = family or getattr(config, "model_type", type(whole_model).__name__)

    if not hasattr(whole_model, "get_pp_module_config"):
        raise PPByteContractError(
            "PP byte contract: pipeline models must expose get_pp_module_config() so cut points can be validated"
        )
    pp_config = whole_model.get_pp_module_config()
    layer_prefix = pp_config.get("layer_prefix", "layers")
    input_fqns = tuple(pp_config.get("input_fqns") or ())
    output_fqns = tuple(pp_config.get("output_fqns") or ())
    num_layers = int(pp_config["num_layers"])
    boundary_state = _normalize_pipeline_boundary_state(pp_config.get("pipeline_boundary_state"))

    per_stage_layers = _validate_stage_plan(
        module_names_per_stage,
        input_fqns=input_fqns,
        layer_prefix=layer_prefix,
        output_fqns=output_fqns,
        num_layers=num_layers,
    )
    _validate_pipeline_ranges(per_stage_layers, pp_config.get("pipeline_layer_ranges"))
    _validate_stage_local_groups(parallel_state)

    # Validate EVERY local part before marking ANY: a failure mid-loop must
    # not leave earlier stages marked as exact.
    for stage_idx, model_part in zip(stage_ids, model_parts):
        _validate_part_layer_identity(model_part, stage_idx, per_stage_layers[stage_idx], layer_prefix)
    for stage_idx, model_part in zip(stage_ids, model_parts):
        # _pp_forward refuses silent metadata fallbacks on marked parts
        # and asserts the bf16 wire reality on stage inputs/outputs.
        model_part._pp_exact_boundary_contract = exact_runtime
        model_part._pp_pipeline_boundary_state = boundary_state
        layers = per_stage_layers[stage_idx]
        layer_summary = "none" if not layers else f"{layers[0]}..{layers[-1]}"
        logger.info(
            f"PP structural boundary engaged: family={diagnostic_family} stage={stage_idx} "
            f"layers=[{layer_summary}] "
            f"first={stage_idx == 0} last={stage_idx == len(per_stage_layers) - 1}"
        )
