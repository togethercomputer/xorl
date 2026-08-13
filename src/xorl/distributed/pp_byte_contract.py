"""Fail-closed byte-boundary contract for pipeline-parallel exact lanes.

The exact train/serve lanes promise forward logit-path bytes identical to the
unpartitioned (PP1) program, which itself matches the sampler. Cutting the
trainer at a pipeline-stage boundary is admitted only when every required
boundary invariant is either bitwise by construction or guarded here:

- the model family's decoder-layer boundary must be a certified natural BF16
  rounding boundary (the residual stream is materialized to one BF16 tensor
  before it would cross the wire, so send/recv moves final bytes, not an
  unrounded intermediate);
- the stage plan may cut only between decoder layers, with contiguous global
  layer coverage, embeddings on stage 0 only, and the final norm and head
  together on the last stage;
- pruning must preserve global ``ModuleList`` indices and ``layer_idx`` so
  layer-position-keyed kernel selection (for example the Qwen3.5 layer-0
  vs layer>0 input-norm families) cannot silently flip at a cut;
- validated model parts are marked so ``_pp_forward`` refuses to fabricate
  per-microbatch metadata (silent positional fallback changes bytes with no
  error).

Everything else RAISEs ``PPByteContractError``. Generic (non-exact) models are
untouched: they never claimed a byte contract at PP1 either.
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


#: Families whose decoder-layer boundary is certified as a natural BF16
#: rounding boundary, gated by tests/distributed/test_pp_byte_alignment.py.
_CERTIFIED_BOUNDARY_FAMILIES = frozenset({"qwen3_5_dense"})

#: The ONLY fp32 parameters admitted to ride along inside an otherwise-bf16
#: stage, as (owner module class name, parameter attribute) pairs. The exact
#: GDN deliberately pins its gating parameters to fp32 (its ``.to(bf16)``
#: skips them); everything else in a marked stage must be bf16.
_APPROVED_FP32_PIN_OWNERS = frozenset(
    {
        ("GatedDeltaNet", "A_log"),
        ("GatedDeltaNet", "dt_bias"),
    }
)


_UNSTAMPED = object()


def exact_contract_family(config: object | None) -> Optional[str]:
    """Classify the exact value program selected by ``config``.

    Returns ``None`` for generic models (no byte contract claimed), otherwise
    one of ``"qwen3_5_dense"``, ``"qwen3_5_moe"``, ``"glm52"``.

    Model resolution stamps ``config._exact_contract_family`` once; when the
    stamp is present it is authoritative (including a stamped ``None`` for
    generic models). Configs that predate stamping (for example direct
    construction in tests) fall back to the shared legacy-flag resolver.
    Admitting a new family is a registry change (stamp entry plus byte
    evidence), not surgery on this classifier.
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
    """Cut-point admission: cuts only between contiguous decoder layers."""

    num_stages = len(module_names_per_stage)
    per_stage_layers: List[List[int]] = []
    covered: List[int] = []
    for stage_idx, module_names in enumerate(module_names_per_stage):
        names = set(module_names)
        layers = _stage_layer_indices(module_names, layer_prefix)
        if not layers:
            raise PPByteContractError(
                f"PP byte contract: stage {stage_idx} owns no decoder layers; every stage must "
                f"cut between '{layer_prefix}.*' modules"
            )
        if sorted(layers) != list(range(min(layers), max(layers) + 1)):
            raise PPByteContractError(
                f"PP byte contract: stage {stage_idx} layers {sorted(layers)} are not contiguous"
            )
        for fqn in input_fqns:
            if stage_idx == 0 and fqn not in names:
                raise PPByteContractError(
                    f"PP byte contract: input module {fqn!r} must live on stage 0 (missing)"
                )
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
    """Pruning must preserve global layer indices and ``layer_idx``."""

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
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is None:
            raise PPByteContractError(
                f"PP byte contract: stage {stage_idx} layer {layer_prefix}.{idx} has no layer_idx; "
                f"global layer identity cannot be verified"
            )
        if int(layer_idx) != idx:
            raise PPByteContractError(
                f"PP byte contract: stage {stage_idx} layer {layer_prefix}.{idx} carries layer_idx="
                f"{layer_idx}; layer-position-keyed behavior would diverge from the PP1 program"
            )


def _declared_dtype(config: object) -> object:
    dtype = getattr(config, "dtype", None)
    if dtype is None and hasattr(config, "__dict__"):
        # Older transformers configs expose only torch_dtype (newer ones alias
        # it to ``dtype`` with a deprecation warning, so prefer ``dtype``).
        dtype = config.__dict__.get("torch_dtype")
    if dtype is None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            dtype = _declared_dtype(text_config)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, dtype)
    return dtype


def _validate_model_dtype(config: object) -> None:
    """The wire contract requires an EXPLICIT bfloat16 declaration.

    Fail closed on absence: an undeclared-dtype model (e.g. an ordinary FP32
    construction) must never be markable as exact. The inter-stage wire
    carries the materialized BF16 residual stream and the downstream stage-IO
    declarations assume it.
    """
    dtype = _declared_dtype(config)
    if dtype is None:
        raise PPByteContractError(
            "PP byte contract: the model config declares no weight dtype; the exact PP lane "
            "requires an explicit bfloat16 declaration (config.dtype) and refuses to guess"
        )
    if dtype is not torch.bfloat16:
        raise PPByteContractError(
            f"PP byte contract: the inter-stage wire carries the materialized BF16 residual stream; "
            f"declared model dtype={dtype} is not admitted"
        )


def _validate_actual_param_reality(
    model_part: nn.Module,
    stage_idx: int,
    *,
    expects_bf16_mixed_precision: bool,
) -> None:
    """Validate the RESOLVED parameter reality, not configuration metadata.

    A model may declare bfloat16 while its parameters are something else; the
    declaration alone must never mark it exact. Admitted realities:

    - parameters in bf16, where any fp32 parameter riding along must be an
      APPROVED pin (``_APPROVED_FP32_PIN_OWNERS``): the exact GDN pins its
      gating parameters to fp32 by design. An arbitrary bf16/fp32 mixture is
      NOT admitted — a rogue fp32 parameter RAISES naming it;
    - uniformly fp32 parameters WHEN the caller declares bf16 mixed-precision
      intent (production full-weight masters; FSDP2's bf16 compute policy is
      applied after the split, and the runtime wire assertions in
      ``_pp_forward`` verify the resulting bytes).

    Everything else RAISES naming declared vs actual.
    """
    saw_bf16 = False
    fp32_names = []
    for name, param in model_part.named_parameters():
        if not param.is_floating_point():
            continue
        if param.dtype == torch.bfloat16:
            saw_bf16 = True
        elif param.dtype == torch.float32:
            fp32_names.append(name)
        else:
            raise PPByteContractError(
                f"PP byte contract: declared model dtype is bfloat16 but stage {stage_idx} parameter "
                f"{name!r} is {param.dtype}; declared-vs-actual mismatch is not admitted"
            )
    if not saw_bf16 and not fp32_names:
        raise PPByteContractError(
            f"PP byte contract: stage {stage_idx} part has no floating-point parameters to validate"
        )
    if not saw_bf16:
        if not expects_bf16_mixed_precision:
            raise PPByteContractError(
                f"PP byte contract: declared model dtype is bfloat16 but stage {stage_idx} parameters "
                f"are uniformly float32 and no bf16 mixed-precision compute policy was declared; a "
                f"declaration alone does not make the wire bf16 (declared=bfloat16, actual=float32)"
            )
        return
    for name in fp32_names:
        owner_path, _, attribute = name.rpartition(".")
        owner = model_part.get_submodule(owner_path) if owner_path else model_part
        if (type(owner).__name__, attribute) not in _APPROVED_FP32_PIN_OWNERS:
            raise PPByteContractError(
                f"PP byte contract: stage {stage_idx} float32 parameter {name!r} (owner "
                f"{type(owner).__name__}) is not an approved fp32 pin; approved pins: "
                f"{sorted(_APPROVED_FP32_PIN_OWNERS)}. An arbitrary bf16/fp32 mixture is not "
                f"admitted (declared=bfloat16, actual mixture)"
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


def engage_pp_byte_contract(
    whole_model: nn.Module,
    module_names_per_stage: Sequence[Sequence[str]],
    stage_ids: Sequence[int],
    model_parts: Sequence[nn.Module],
    *,
    expects_bf16_mixed_precision: bool = False,
) -> None:
    """Validate and mark an exact-contract pipeline split; no-op for generic models.

    Called from ``pipeline_module_split`` after pruning. Raises
    ``PPByteContractError`` when the split cannot preserve byte identity with
    the PP1 program; logs one engagement line per local stage when it can.
    """

    config = getattr(whole_model, "config", None)
    family = exact_contract_family(config)
    if family is None:
        return
    if family == "qwen3_5_moe":
        raise PPByteContractError(
            "PP byte contract: the exact Qwen3.5-MoE program is not admitted with pipeline "
            "parallelism; the ordered EP combine has no qualified EP/PP interaction"
        )
    if family == "glm52":
        raise PPByteContractError(
            "PP byte contract: the exact GLM-5.2 program is admitted at PP1 only"
        )
    if family not in _CERTIFIED_BOUNDARY_FAMILIES:
        raise PPByteContractError(
            f"PP byte contract: model family {family!r} has no certified natural rounding "
            f"boundary at decoder-layer cuts"
        )

    if not hasattr(whole_model, "get_pp_module_config"):
        raise PPByteContractError(
            "PP byte contract: exact-contract models must expose get_pp_module_config() so cut "
            "points can be validated"
        )
    pp_config = whole_model.get_pp_module_config()
    layer_prefix = pp_config.get("layer_prefix", "layers")
    input_fqns = tuple(pp_config.get("input_fqns") or ())
    output_fqns = tuple(pp_config.get("output_fqns") or ())
    num_layers = int(pp_config["num_layers"])

    _validate_model_dtype(config)
    per_stage_layers = _validate_stage_plan(
        module_names_per_stage,
        input_fqns=input_fqns,
        layer_prefix=layer_prefix,
        output_fqns=output_fqns,
        num_layers=num_layers,
    )

    # Validate EVERY local part before marking ANY: a failure mid-loop must
    # not leave earlier stages marked as exact.
    for stage_idx, model_part in zip(stage_ids, model_parts):
        _validate_part_layer_identity(model_part, stage_idx, per_stage_layers[stage_idx], layer_prefix)
        _validate_actual_param_reality(
            model_part, stage_idx, expects_bf16_mixed_precision=expects_bf16_mixed_precision
        )
    for stage_idx, model_part in zip(stage_ids, model_parts):
        # _pp_forward refuses silent metadata fallbacks on marked parts
        # and asserts the bf16 wire reality on stage inputs/outputs.
        model_part._pp_exact_boundary_contract = True
        layers = per_stage_layers[stage_idx]
        logger.info(
            f"PP byte-boundary contract engaged: family={family} stage={stage_idx} "
            f"layers=[{layers[0]}..{layers[-1]}] wire=bf16(materialized residual stream) "
            f"first={stage_idx == 0} last={stage_idx == len(per_stage_layers) - 1}"
        )
