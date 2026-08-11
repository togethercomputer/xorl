import gc
import json
import os
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed.checkpoint.state_dict import StateDictOptions

from ..distributed.parallel_state import get_parallel_state
from ..utils.checkpoint_utils import _GLOBAL_STEP_PREFIX
from ..utils.import_utils import is_torch_version_greater_than
from ..utils.logging import get_logger


if is_torch_version_greater_than("2.4"):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import (
        DefaultLoadPlanner,
        FileSystemReader,
        FileSystemWriter,
    )
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful
else:
    Stateful = ABC

logger = get_logger(__name__)

_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"
_EXTRA_STATE_DIR = "extra_state"
_CHECKPOINT_METADATA_FILE = "checkpoint_metadata.json"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


_COMPILE_WRAPPER_SEGMENT = "._orig_mod."


def _strip_compile_prefix(name: str) -> str:
    """Strip the ``._orig_mod.`` segment ``torch.compile`` inserts into parameter/buffer FQNs.

    DCP checkpoints follow the convention that keys are stored WITHOUT this segment
    (mirrors ``models.module_utils._build_compiled_key_map``), so a checkpoint saved from a
    compiled model loads into an eager model and vice versa. Without this, a checkpoint saved
    mid-training under ``enable_compile`` records keys like ``model.layers.0._orig_mod.mlp...``
    that fail to match the eager model materialized at load time (resume happens before the
    model is compiled), raising "Checkpoint incompatible with model". No-op for eager models.
    """
    return name.replace(_COMPILE_WRAPPER_SEGMENT, ".")


def _compile_agnostic_spec_info(spec_info):
    """Alias torch.compile-wrapped FQNs to checkpoint FQNs in EP spec metadata."""
    if spec_info is None:
        return None

    normalized_spec_info = dict(spec_info)
    for name, info in list(spec_info.items()):
        normalized_spec_info.setdefault(_strip_compile_prefix(name), info)
    return normalized_spec_info


_EP_CHECKPOINT_MESH_DIM_NAMES = ("ep", "ep_fsdp")


def _get_ep_checkpoint_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Select the rank-local EP x expert-FSDP mesh used by DCP.

    Pipeline parallelism adds a parent ``_pp_ep`` dimension so that each PP
    stage owns independent EP process groups. That parent dimension is a
    replication/ownership dimension for checkpointing, not another sharding
    dimension of an expert tensor. Select the two checkpoint dimensions by
    name so both the legacy 2-D mesh and a PP-scoped parent mesh have the same
    DCP tensor layout.
    """
    mesh_dim_names = tuple(device_mesh.mesh_dim_names or ())
    invalid_names = [name for name in _EP_CHECKPOINT_MESH_DIM_NAMES if mesh_dim_names.count(name) != 1]
    if invalid_names:
        raise RuntimeError(
            "EP checkpoint mesh must contain exactly one 'ep' and one 'ep_fsdp' dimension; "
            f"got mesh_dim_names={mesh_dim_names}"
        )

    checkpoint_mesh = device_mesh[_EP_CHECKPOINT_MESH_DIM_NAMES]
    checkpoint_mesh_dim_names = tuple(checkpoint_mesh.mesh_dim_names or ())
    if checkpoint_mesh_dim_names != _EP_CHECKPOINT_MESH_DIM_NAMES:
        raise RuntimeError(
            "EP checkpoint mesh selection returned an unexpected layout; "
            f"expected mesh_dim_names={_EP_CHECKPOINT_MESH_DIM_NAMES}, got {checkpoint_mesh_dim_names}"
        )
    return checkpoint_mesh


def _restore_ep_dim(origin_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """Restore the EP dim so that DCP records the true global (EP+FSDP) size.

    The live model holds expert tensors as 1-D DTensors sharded only on the FSDP
    dim (the EP dim is implicit/local). On save we reconstruct the 2-D
    ``[Shard(0)=ep, Shard(1)=ep_fsdp]`` DTensor so DCP knows the full expert
    dimension and can later reshard across a different EP size.

    Shared by ``ModelState`` (weights) and ``OptimizerState`` (per-param optimizer
    state such as Muon ``momentum_buffer`` / AdamW ``exp_avg``).

    Args:
        origin_tensor: The (FSDP-sharded) expert tensor or its optimizer-state buffer.
        device_mesh: A named mesh containing the ``"ep"`` and ``"ep_fsdp"``
            dimensions. It may also contain a PP ownership dimension.
    """
    checkpoint_mesh = _get_ep_checkpoint_mesh(device_mesh)
    ep_mesh = checkpoint_mesh["ep"]

    if origin_tensor.__class__.__name__ == "DTensor":
        # EP+FSDP2
        dtensor = DTensor.from_local(
            origin_tensor._local_tensor, device_mesh=checkpoint_mesh, placements=[Shard(0), Shard(1)]
        )
    elif origin_tensor.__class__.__name__ == "Tensor":
        # If there is no FSDP
        dtensor = DTensor.from_local(origin_tensor, device_mesh=ep_mesh, placements=[Shard(0)])

    return dtensor


def _drop_ep_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
    """Drop the EP dim after loading from DCP so EP-FSDP is not confused.

    Reverse of :func:`_restore_ep_dim`: collapse the reconstructed 2-D EP+FSDP
    DTensor back to the FSDP-only layout (or a plain local tensor when there is
    no FSDP) that the live model/optimizer expects.
    """
    checkpoint_mesh = _get_ep_checkpoint_mesh(device_mesh)
    ep_fsdp_mesh = checkpoint_mesh["ep_fsdp"]

    if len(loaded_tensor.placements) == 2:
        tensor_to_put = DTensor.from_local(loaded_tensor._local_tensor, device_mesh=ep_fsdp_mesh, placements=[Shard(1)])
    elif len(loaded_tensor.placements) == 1:
        tensor_to_put = loaded_tensor.to_local()
    else:
        raise RuntimeError(
            f"Expect EP parameters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (EP+FSDP), got {loaded_tensor}"
        )

    return tensor_to_put


def _as_model_parts(model) -> List[torch.nn.Module]:
    """Normalize a model-or-parts-list (PP virtual stages) to a list of modules."""
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _merged_model_state_dict(model) -> Dict[str, Any]:
    """get_model_state_dict merged across PP virtual-stage parts (disjoint FQNs;
    duplicates from always-keep modules resolve last-wins)."""
    merged: Dict[str, Any] = {}
    for part in _as_model_parts(model):
        merged.update(get_model_state_dict(model=part))
    return merged


def _glm52_exact_base_dcp_projection(model):
    """Return the narrow exact-GLM base-DCP adapter when the model needs it."""

    from xorl.models.exact_contract import contains_glm52_exact_active_lora_component  # noqa: PLC0415

    if not any(contains_glm52_exact_active_lora_component(part) for part in _as_model_parts(model)):
        return None

    from xorl.models.transformers.glm5.exact_dcp import Glm52ExactBaseDcpLoadProjection  # noqa: PLC0415

    projection = Glm52ExactBaseDcpLoadProjection(model)
    return projection if projection.enabled else None


def _get_model_param_keys(model) -> List[str]:
    """Get sorted list of parameter keys from a model (or PP virtual-stage parts)."""
    return sorted(
        {_strip_compile_prefix(name) for part in _as_model_parts(model) for name, _ in part.named_parameters()}
    )


def _get_model_persistent_buffer_keys(model) -> List[str]:
    """Get sorted list of persistent buffer keys from a model (or PP virtual-stage parts)."""
    buffer_keys: set = set()
    for part in _as_model_parts(model):
        modules = dict(part.named_modules(remove_duplicate=False))
        for name, buffer in part.named_buffers(remove_duplicate=False):
            if buffer is None:
                continue
            module_name, _, buffer_name = name.rpartition(".")
            parent_module = modules[module_name] if module_name else part
            if buffer_name in getattr(parent_module, "_non_persistent_buffers_set", set()):
                continue
            # Strip the compile prefix only on the recorded key; the module lookup above uses the
            # live (possibly ``_orig_mod``-wrapped) FQN.
            buffer_keys.add(_strip_compile_prefix(name))
    return sorted(buffer_keys)


def _get_checkpoint_model_keys(model, process_group=None) -> tuple[List[str], List[str], bool]:
    """Return the checkpoint key contract, unioned across pipeline stages when needed."""
    param_keys = _get_model_param_keys(model)
    buffer_keys = _get_model_persistent_buffer_keys(model)
    if not get_parallel_state().pp_enabled:
        return param_keys, buffer_keys, False

    if not dist.is_initialized():
        raise RuntimeError("Pipeline-parallel checkpoint key validation requires torch.distributed initialization")

    gathered_keys: List[Any] = [None] * dist.get_world_size(group=process_group)
    dist.all_gather_object(
        gathered_keys,
        (param_keys, buffer_keys),
        group=process_group,
    )
    param_keys = sorted({key for rank_params, _ in gathered_keys for key in rank_params})
    buffer_keys = sorted({key for _, rank_buffers in gathered_keys for key in rank_buffers})
    return param_keys, buffer_keys, True


def _save_checkpoint_metadata(
    checkpoint_dir: str,
    model: torch.nn.Module,
    has_lora: bool = False,
    save_lora_only: bool = False,
    process_group=None,
) -> None:
    """
    Save checkpoint metadata to a JSON file. All pipeline ranks contribute
    their stage-local keys; only global rank 0 writes the metadata file.
    """
    param_keys, buffer_keys, pipeline_key_union = _get_checkpoint_model_keys(model, process_group=process_group)
    lora_keys = [k for k in param_keys if "lora" in k.lower()]

    if save_lora_only:
        # Only LoRA keys are actually saved
        param_keys = lora_keys
        buffer_keys = []

    metadata = {
        "num_parameters": len(param_keys),
        "num_buffers": len(buffer_keys),
        "has_lora": has_lora or len(lora_keys) > 0,
        "num_lora_parameters": len(lora_keys),
        "save_lora_only": save_lora_only,
        "pipeline_parallel_key_union": pipeline_key_union,
        "parameter_keys": param_keys,
        "buffer_keys": buffer_keys,
    }

    if dist.get_rank() != 0:
        return

    metadata_path = os.path.join(checkpoint_dir, _CHECKPOINT_METADATA_FILE)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved checkpoint metadata: {len(param_keys)} params, {len(buffer_keys)} buffers, {len(lora_keys)} LoRA params"
    )


def _validate_checkpoint_compatibility(
    checkpoint_dir: str,
    model: torch.nn.Module,
    strict: bool = True,
    process_group=None,
) -> Dict[str, Any]:
    """
    Validate that a checkpoint is compatible with the current model.

    Args:
        checkpoint_dir: Path to checkpoint directory
        model: Current model to validate against
        strict: If True, raise error on mismatch. If False, return info about mismatches.
        process_group: Group spanning every pipeline stage when PP is enabled.

    Returns:
        Dictionary with validation results including missing/unexpected keys

    Raises:
        RuntimeError: If strict=True and checkpoint is incompatible
    """
    metadata_path = os.path.join(checkpoint_dir, _CHECKPOINT_METADATA_FILE)

    # If no metadata file exists (old checkpoint), skip validation
    if not os.path.exists(metadata_path):
        logger.warning(
            f"No checkpoint metadata found at {metadata_path}. Skipping compatibility check (old checkpoint format)."
        )
        return {"validated": False, "reason": "no_metadata"}

    with open(metadata_path, "r") as f:
        ckpt_metadata = json.load(f)

    ckpt_keys = set(ckpt_metadata.get("parameter_keys", []))
    ckpt_lora_only = ckpt_metadata.get("save_lora_only", False)
    model_param_keys, model_persistent_buffer_keys, pipeline_key_union = _get_checkpoint_model_keys(
        model, process_group=process_group
    )
    base_dcp_projection = None
    if ckpt_metadata.get("has_lora") is False and not ckpt_lora_only:
        base_dcp_projection = _glm52_exact_base_dcp_projection(model)
        if base_dcp_projection is not None:
            model_param_keys, model_persistent_buffer_keys = base_dcp_projection.project_key_contract(
                model_param_keys,
                model_persistent_buffer_keys,
            )
    model_keys = set(model_param_keys)
    ckpt_buffer_keys_raw = ckpt_metadata.get("buffer_keys")
    buffers_validated = isinstance(ckpt_buffer_keys_raw, list)
    ckpt_buffer_keys = set(ckpt_buffer_keys_raw) if buffers_validated else set()
    model_buffer_keys = set(model_persistent_buffer_keys) if buffers_validated else set()

    # Keys in model but not in checkpoint (e.g., LoRA params added after checkpoint was saved)
    missing_in_ckpt = model_keys - ckpt_keys
    # Keys in checkpoint but not in model (e.g., removed params)
    unexpected_in_ckpt = ckpt_keys - model_keys
    missing_buffers_in_ckpt = model_buffer_keys - ckpt_buffer_keys
    unexpected_buffers_in_ckpt = ckpt_buffer_keys - model_buffer_keys

    # Check if mismatch is LoRA-related
    missing_lora_keys = [k for k in missing_in_ckpt if "lora" in k.lower()]
    missing_non_lora_keys = [k for k in missing_in_ckpt if "lora" not in k.lower()]

    result = {
        "validated": True,
        "checkpoint_has_lora": ckpt_metadata.get("has_lora", False),
        "checkpoint_lora_only": ckpt_lora_only,
        "model_has_lora": any("lora" in k.lower() for k in model_keys),
        "glm52_exact_base_dcp_projection": base_dcp_projection is not None,
        "pipeline_parallel_key_union": pipeline_key_union,
        "model_parameter_count": len(model_keys),
        "model_buffer_count": len(model_buffer_keys),
        "missing_in_checkpoint": list(missing_in_ckpt),
        "unexpected_in_checkpoint": list(unexpected_in_ckpt),
        "buffers_validated": buffers_validated,
        "missing_buffers_in_checkpoint": list(missing_buffers_in_ckpt),
        "unexpected_buffers_in_checkpoint": list(unexpected_buffers_in_ckpt),
        "missing_lora_keys": missing_lora_keys,
        "missing_non_lora_keys": missing_non_lora_keys,
        "compatible": (
            len(missing_non_lora_keys) == 0
            and len(unexpected_in_ckpt) == 0
            and len(missing_buffers_in_ckpt) == 0
            and len(unexpected_buffers_in_ckpt) == 0
        ),
    }

    # Log validation results
    if missing_in_ckpt or unexpected_in_ckpt or missing_buffers_in_ckpt or unexpected_buffers_in_ckpt:
        # LoRA-only checkpoint: missing non-LoRA keys are expected
        if ckpt_lora_only and len(missing_non_lora_keys) > 0 and len(unexpected_in_ckpt) == 0:
            logger.info(
                f"Loading LoRA-only checkpoint (save_lora_only=True). "
                f"Non-LoRA parameters ({len(missing_non_lora_keys)} keys) will keep their current values."
            )
            result["load_mode"] = "lora_only"
            result["compatible"] = True
            return result

        logger.warning(
            f"Checkpoint compatibility check:\n"
            f"  - Missing in checkpoint: {len(missing_in_ckpt)} keys "
            f"({len(missing_lora_keys)} LoRA, {len(missing_non_lora_keys)} non-LoRA)\n"
            f"  - Unexpected in checkpoint: {len(unexpected_in_ckpt)} keys\n"
            f"  - Missing buffers in checkpoint: {len(missing_buffers_in_ckpt)} keys\n"
            f"  - Unexpected buffers in checkpoint: {len(unexpected_buffers_in_ckpt)} keys"
        )

        # If only LoRA keys are missing, this is expected when loading base checkpoint into LoRA model
        if (
            len(missing_lora_keys) > 0
            and len(missing_non_lora_keys) == 0
            and len(unexpected_in_ckpt) == 0
            and len(missing_buffers_in_ckpt) == 0
            and len(unexpected_buffers_in_ckpt) == 0
        ):
            logger.info(
                "Loading base model checkpoint into LoRA-enabled model. "
                f"LoRA parameters ({len(missing_lora_keys)} keys) will keep their initialized values."
            )
            result["load_mode"] = "base_to_lora"
        elif strict and (
            len(missing_non_lora_keys) > 0
            or len(unexpected_in_ckpt) > 0
            or len(missing_buffers_in_ckpt) > 0
            or len(unexpected_buffers_in_ckpt) > 0
        ):
            error_msg = (
                f"Checkpoint incompatible with model:\n"
                f"  Missing non-LoRA keys: {missing_non_lora_keys[:5]}{'...' if len(missing_non_lora_keys) > 5 else ''}\n"
                f"  Unexpected keys: {list(unexpected_in_ckpt)[:5]}{'...' if len(unexpected_in_ckpt) > 5 else ''}\n"
                f"  Missing buffers: {list(missing_buffers_in_ckpt)[:5]}"
                f"{'...' if len(missing_buffers_in_ckpt) > 5 else ''}\n"
                f"  Unexpected buffers: {list(unexpected_buffers_in_ckpt)[:5]}"
                f"{'...' if len(unexpected_buffers_in_ckpt) > 5 else ''}"
            )
            raise RuntimeError(error_msg)

    return result


class ModelState(Stateful):
    """
    A wrapper around a model to make it stateful.
    Args:
        model (Model): model to wrap.
        exclude_keys (Set[str]): Optional set of parameter keys to exclude from state_dict.
                                 Used when loading base checkpoint into LoRA model.
        save_lora_only (bool): If True, only save LoRA parameters (lora_A, lora_B).
                               Used when merge_lora_interval == 0 (base weights unchanged).
    """

    def __init__(
        self,
        model,
        exclude_keys: Optional[Set[str]] = None,
        save_lora_only: bool = False,
        project_glm52_exact_base_dcp: bool = False,
    ):
        self.model = model
        self.exclude_keys = exclude_keys or set()
        self.save_lora_only = save_lora_only
        self.base_dcp_projection = _glm52_exact_base_dcp_projection(model) if project_glm52_exact_base_dcp else None
        if project_glm52_exact_base_dcp and self.base_dcp_projection is None:
            raise RuntimeError("Exact GLM base-DCP projection was requested for a model without projected state")
        self._base_dcp_model_state = None

        # Determine whether this is EP+FSDP2 case
        # If so, we need to restore EP-dim before saving to DCP
        # (model may be a PP virtual-stage parts list — EP is rejected there, so getattr yields None)
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = _compile_agnostic_spec_info(getattr(self.model, "_fqn2spec_info", None))
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"

    @torch.no_grad()
    def state_dict(self):
        model_state_dict = _merged_model_state_dict(self.model)
        if self.should_ep_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore EP dim for Experts module"
            )
            model_state_dict = self.get_state_dict_with_ep_dim(model_state_dict)

        # Strip torch.compile's ``._orig_mod.`` prefix AFTER EP-dim restoration (which matches
        # keys against the compiled model's _fqn2spec_info) so checkpoints stay compile-agnostic.
        model_state_dict = {_strip_compile_prefix(k): v for k, v in model_state_dict.items()}

        # Filter out excluded keys (e.g., LoRA params when loading base checkpoint)
        if self.exclude_keys:
            model_state_dict = {k: v for k, v in model_state_dict.items() if k not in self.exclude_keys}

        # LoRA-only save: keep only lora_A/lora_B parameters
        if self.save_lora_only:
            model_state_dict = {k: v for k, v in model_state_dict.items() if "lora_" in k}
            logger.info_rank0(f"LoRA-only save: keeping {len(model_state_dict)} LoRA parameters")

        if self.base_dcp_projection is not None:
            self._base_dcp_model_state = model_state_dict
            model_state_dict = self.base_dcp_projection.project_state(model_state_dict)

        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        perform the reverse operation for state_dict()
        need to drop EP-dim when loading from DCP checkpoints
        so that EP-FSDP would not be confused

        Uses strict=False to allow loading checkpoints that don't have LoRA
        parameters (e.g., loading from a base model checkpoint into a LoRA-enabled model).
        Missing parameters (like lora_A, lora_B) will retain their initialized values.
        """

        model_state_dict = state_dict
        if self.base_dcp_projection is not None:
            if self._base_dcp_model_state is None:
                raise RuntimeError("Exact GLM base-DCP state was loaded before its target state was projected")
            model_state_dict = self.base_dcp_projection.restore_state(
                projected_state=state_dict,
                model_state=self._base_dcp_model_state,
            )
            self._base_dcp_model_state = None
        if self.should_ep_aware:
            model_state_dict = self.get_state_dict_without_ep_dim(model_state_dict)

        # Use strict=False to allow missing LoRA parameters when loading from
        # a checkpoint that was saved before LoRA was injected
        options = StateDictOptions(strict=False)
        parts = _as_model_parts(self.model)
        if len(parts) == 1:
            incompatible = set_model_state_dict(model=parts[0], model_state_dict=model_state_dict, options=options)
        else:
            # PP virtual stages: load each part from its own key subset so other
            # parts' keys don't show up as unexpected.
            missing_keys: List[str] = []
            unexpected: set = set(model_state_dict.keys())
            for part in parts:
                part_keys = set(get_model_state_dict(model=part).keys())
                part_sd = {k: v for k, v in model_state_dict.items() if k in part_keys}
                unexpected -= part_keys
                inc = set_model_state_dict(model=part, model_state_dict=part_sd, options=options)
                missing_keys.extend(inc.missing_keys)
            incompatible = SimpleNamespace(missing_keys=missing_keys, unexpected_keys=sorted(unexpected))

        # Log missing/unexpected keys for debugging
        if incompatible.missing_keys:
            # Filter to show only non-LoRA missing keys as warnings
            non_lora_missing = [k for k in incompatible.missing_keys if "lora_" not in k]
            lora_missing = [k for k in incompatible.missing_keys if "lora_" in k]
            if lora_missing:
                logger.info_rank0(
                    f"LoRA parameters not in checkpoint (will use initialized values): {len(lora_missing)} params"
                )
            if non_lora_missing:
                logger.warning(f"Missing non-LoRA keys in checkpoint: {non_lora_missing}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")

    def get_state_dict_with_ep_dim(self, state_dict):
        ep_fqn2spec_info = self.ep_fqn2spec_info
        assert ep_fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in ep_fqn2spec_info and isinstance(ep_fqn2spec_info[name].placement, Shard):
                cur_spec_info = ep_fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = _restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict

    def get_state_dict_without_ep_dim(self, state_dict):
        fqn2spec_info = self.ep_fqn2spec_info
        assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in fqn2spec_info and isinstance(fqn2spec_info[name].placement, Shard):
                cur_spec_info = fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = _drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.

    Args:
        optimizer (Optimizer): optimizer to wrap.
    """

    # Flattened per-param optimizer-state buffers that share the parameter's sharding
    # (Muon momentum + AdamW first/second moments). ``step`` is a scalar and is excluded.
    _EP_SHARDED_STATE_KEYS = frozenset({"momentum_buffer", "exp_avg", "exp_avg_sq"})

    def __init__(self, model, optimizer, load_keys: Optional[Set[str]] = None):
        self.model = model
        self.optimizer = optimizer
        self.load_keys = load_keys
        self._allow_partial_optimizer_load = False
        if isinstance(model, (list, tuple)) and not getattr(optimizer, "_is_multi_optimizer", False):
            # PP virtual stages require the per-part MultiOptimizer (it owns the
            # part->model mapping DCP needs to resolve FQNs).
            raise ValueError("Checkpointing multiple PP model parts requires a MultiOptimizer.")

        # Mirror ModelState: in the EP+FSDP2 case the per-param optimizer state for expert
        # weights is sharded just like the weights themselves, so it must carry the EP dim in
        # DCP (restore on save / drop on load) to reshard across different EP sizes on resume.
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = _compile_agnostic_spec_info(getattr(self.model, "_fqn2spec_info", None))
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"

    @staticmethod
    def _flattened_key_param_fqn(key: str):
        """Map a flattened optimizer key ``state.<param_fqn>.<state_key>`` to (param_fqn, state_key).

        Returns ``(None, None)`` for non-``state.*`` keys (e.g. ``param_groups.*``). The param FQN
        is normalized to be torch.compile-agnostic so it matches ``ep_fqn2spec_info``.
        """
        if not key.startswith("state."):
            return None, None
        remainder = key[len("state.") :]
        param_fqn, _, state_key = remainder.rpartition(".")
        if not param_fqn:
            return None, None
        return _strip_compile_prefix(param_fqn), state_key

    def _get_optimizer_state_dict_with_ep_dim(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Restore the EP dim on the flattened optimizer state before saving to DCP."""
        for key in sorted(state_dict.keys()):
            param_fqn, state_key = self._flattened_key_param_fqn(key)
            if param_fqn is None or state_key not in self._EP_SHARDED_STATE_KEYS:
                continue
            spec_info = self.ep_fqn2spec_info.get(param_fqn)
            if spec_info is not None and isinstance(spec_info.placement, Shard):
                state_dict[key] = _restore_ep_dim(state_dict[key], spec_info.ep_fsdp_mesh)
        return state_dict

    def _get_optimizer_state_dict_without_ep_dim(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the EP dim on the flattened optimizer state after loading from DCP."""
        for key in sorted(state_dict.keys()):
            param_fqn, state_key = self._flattened_key_param_fqn(key)
            if param_fqn is None or state_key not in self._EP_SHARDED_STATE_KEYS:
                continue
            spec_info = self.ep_fqn2spec_info.get(param_fqn)
            if spec_info is not None and isinstance(spec_info.placement, Shard):
                state_dict[key] = _drop_ep_dim(state_dict[key], spec_info.ep_fsdp_mesh)
        return state_dict

    def _filter_state_dict_for_load(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.load_keys is None:
            return state_dict

        filtered = {key: value for key, value in state_dict.items() if key in self.load_keys}
        dropped = len(state_dict) - len(filtered)
        if dropped > 0:
            self._allow_partial_optimizer_load = True
            logger.info_rank0(
                f"Filtered optimizer load target to {len(filtered)} checkpoint key(s), "
                f"dropping {dropped} key(s) without saved optimizer state."
            )
        return filtered

    def state_dict(self):
        # If optimizer is None (when save_optimizer=False), return empty dict
        if self.optimizer is None:
            return {}

        # MultiOptimizer is only used for EP+FSDP2 case for now,
        # and it knows how to produce a merged, flattened dict already
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            sd = self.optimizer.state_dict()
            if self.should_ep_aware:
                logger.info_rank0(
                    "Getting optimizer state_dict from OptimizerState wrapper, restoring EP dim for Experts state"
                )
                sd = self._get_optimizer_state_dict_with_ep_dim(sd)
            return self._filter_state_dict_for_load(sd)

        # Single torch optimizer
        sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
        return self._filter_state_dict_for_load(sd)

    def load_state_dict(self, state_dict):
        # If optimizer is None (when load_optimizer=False), skip loading
        if self.optimizer is None:
            return

        # If state_dict is empty (checkpoint saved with save_optimizer=False or step 0), skip loading
        if not state_dict:
            return

        optim_state = state_dict

        # Delegate to MultiOptimizer (it will split/filter correctly)
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            if self.should_ep_aware:
                logger.info_rank0(
                    "Loading optimizer state_dict via OptimizerState wrapper, dropping EP dim for Experts state"
                )
                optim_state = self._get_optimizer_state_dict_without_ep_dim(optim_state)
            self.optimizer.load_state_dict(optim_state, strict=not self._allow_partial_optimizer_load)
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state,
            options=StateDictOptions(strict=not self._allow_partial_optimizer_load),
        )


def build_checkpointer(
    dist_backend: str = "fsdp2",
    ckpt_manager: str = "dcp",
):
    """
    create a checkpointer manager with given mode.
    Args:
        dist_backend (str, optional): checkpoint mode. Defaults to "fsdp2".
            fsdp2: FSDP2 checkpointer
            ddp: DDP checkpointer
            dcp: DCP checkpoint from torch.distributed.checkpoint
            native: native checkpoint from torch.save
        ckpt_manager (str, optional): checkpoint manager.
            dcp: torch dcp checkpoint manager
    Raises:
        ValueError: if ckpt_manager is not supported

    Returns:
        Checkpointer: checkpointer with given mode.
    """

    if ckpt_manager == "dcp":
        if not is_torch_version_greater_than("2.4"):
            raise ValueError("DCP checkpoint manager requires torch version >= 2.4")
        if dist_backend not in ["none", "ddp", "fsdp2"]:
            raise ValueError(
                f"Unsupported distributed backend: {dist_backend} for DCP checkpoint manager, supported modes are: none, ddp, fsdp2"
            )
        Checkpointer = DistributedCheckpointer
    else:
        raise ValueError(f"Unknown checkpoint manager: {ckpt_manager}, supported: dcp")

    return Checkpointer


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: Optional[bool],
        global_steps: Optional[int],
    ):
        return

    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return


class DistributedCheckpointer(CheckpointerBase):
    """
    Distributed checkpointer for torch.distributed.checkpoint
    """

    dcp_save_future: Optional[Any] = None
    # Dedicated process group for async saves (created on first use)
    _async_process_group: Optional[Any] = None
    # Dedicated Gloo process group for DCP metadata/object collectives. NCCL
    # object collectives can corrupt large DCP planning payloads at high rank
    # counts; async saves already use Gloo for this reason.
    _sync_process_group: Optional[Any] = None

    @classmethod
    def _get_sync_process_group(cls):
        if not dist.is_available() or not dist.is_initialized() or dist.get_backend() == "gloo":
            return None
        if cls._sync_process_group is None:
            cls._sync_process_group = dist.new_group(backend="gloo")
        return cls._sync_process_group

    @classmethod
    def _get_metadata_process_group(cls, process_group=None):
        if not get_parallel_state().pp_enabled:
            return None
        return process_group if process_group is not None else cls._get_sync_process_group()

    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        global_steps: int = None,
        save_lora_only: bool = False,
    ) -> None:
        """
        save training state to distributed checkpoint

        args:
            path: path to save checkpoint
            state: state to save
            global_steps: global steps
            save_lora_only: if True, only save LoRA parameters (for merge_lora_interval==0)
        return:
            None
        """

        checkpoint_dir = f"{path}/{_GLOBAL_STEP_PREFIX}{global_steps}" if global_steps else path
        os.makedirs(checkpoint_dir, exist_ok=True)

        # saving extra_state first to guarantee that every saved model/optimizer ckpts have their extra_state saved before them
        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            torch.save(
                state["extra_state"],
                extra_state_path,
            )

        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        save_state = {"model": ModelState(state["model"], save_lora_only=save_lora_only)}
        if "optimizer" in state:
            save_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        if save_async:
            # Lazily create a dedicated Gloo process group for async DCP saves
            if cls._async_process_group is None:
                cls._async_process_group = dist.new_group(backend="gloo")

            if cls.dcp_save_future is not None:
                logger.debug(f"[RANK {dist.get_rank()}] waiting for previous DCP saving session to end...")
                cls.dcp_save_future.result()
                cls.dcp_save_future = None
                # block until all the ranks resolve their previous dcp async saving
                dist.barrier()

            cls.dcp_save_future = dcp.async_save(
                state_dict=save_state,
                storage_writer=FileSystemWriter(
                    checkpoint_dir,
                    thread_count=1,  # Reduced from 16 to avoid PyTorch concurrent write bug
                    single_file_per_rank=True,
                    sync_files=False,
                    overwrite=True,
                ),
                process_group=cls._async_process_group,
            )
        else:
            process_group = cls._get_sync_process_group()
            dcp.save(
                state_dict=save_state,
                storage_writer=FileSystemWriter(
                    checkpoint_dir,
                    thread_count=1,  # Reduced from 16 to avoid PyTorch concurrent write bug
                    single_file_per_rank=True,
                    sync_files=False,
                    overwrite=True,
                ),
                process_group=process_group,
            )

        # Aggressive cleanup after DCP save to release all intermediate memory
        # This is critical for large EP models where DCP creates full state dicts
        if "model" in save_state and hasattr(save_state["model"], "model"):
            # Clear any cached state in ModelState wrapper
            save_state["model"].model = None
            save_state["model"].ep_fqn2spec_info = None
        if "optimizer" in save_state and hasattr(save_state["optimizer"], "model"):
            save_state["optimizer"].model = None
            save_state["optimizer"].optimizer = None
        del save_state

        gc.collect()
        gc.collect()  # Second pass for cyclic references
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Save checkpoint metadata for compatibility validation
        metadata_process_group = cls._get_metadata_process_group()
        _save_checkpoint_metadata(
            checkpoint_dir,
            state["model"],
            save_lora_only=save_lora_only,
            process_group=metadata_process_group,
        )

        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        load training state from distributed checkpoint
        args:
            path: path to load checkpoint
            state: state to load, "model" are required,  "optimizer" and "extra_state" are optional
            strict: if True, raise error on checkpoint/model mismatch (except LoRA params)

        return:
            state: state loaded
        """
        checkpoint_dir = path

        if state is None:
            raise ValueError("State dict must be provided to load a distributed checkpoint.")

        if "model" not in state:
            raise ValueError("Model must be provided to load a distributed checkpoint.")

        # Resolve the DCP load process group FIRST, before any rank-divergent code
        # below (checkpoint validation, the per-rank optimizer-metadata read). The
        # sync group is created via dist.new_group() — a collective on the DEFAULT
        # process group that every rank entering load() must issue together and in
        # the same order. If it is created lazily *after* a validation/metadata step
        # that raises on only a subset of ranks, the remaining ranks block on a Gloo
        # CPU collective (/default_pg/.../cpu) for the full PG timeout (observed:
        # DistStoreError after 1800000ms on the 4-node DCP reshard load). Creating it
        # here keeps it uniform across all ranks. XORL_DCP_LOAD_NO_DIST=1 disables
        # distributed DCP planning entirely (each rank reads its shards from the
        # shared checkpoint dir), sidestepping the collective.
        load_no_dist = _env_flag("XORL_DCP_LOAD_NO_DIST")
        metadata_process_group = cls._get_metadata_process_group(process_group)
        if load_no_dist:
            logger.info_rank0(
                "Loading DCP checkpoint with no_dist=True; distributed DCP planning collectives are disabled."
            )
            process_group = None
        elif process_group is None:
            process_group = cls._get_sync_process_group()

        # Validate checkpoint compatibility before loading
        validation_result = _validate_checkpoint_compatibility(
            checkpoint_dir,
            state["model"],
            strict=strict,
            process_group=metadata_process_group,
        )

        # Determine keys to exclude from loading (e.g., LoRA params not in checkpoint)
        exclude_keys: Set[str] = set()
        load_mode = validation_result.get("load_mode")
        if validation_result.get("validated") and load_mode == "base_to_lora":
            # Loading base checkpoint into LoRA model - exclude LoRA params from model state
            # so DCP doesn't try to load them from checkpoint
            exclude_keys = set(validation_result.get("missing_lora_keys", []))
            logger.info_rank0(f"Excluding {len(exclude_keys)} LoRA parameters from checkpoint load")
        elif validation_result.get("validated") and load_mode == "lora_only":
            # Loading LoRA-only checkpoint — exclude all non-LoRA keys from model state
            # so DCP only loads the LoRA parameters.
            # Must use get_model_state_dict() to capture both params AND buffers
            # (e.g., weight_block_scales, weight_global_scale from QLoRA).
            all_model_keys = set(_merged_model_state_dict(state["model"]).keys())
            exclude_keys = {k for k in all_model_keys if "lora_" not in k}
            logger.info_rank0(f"LoRA-only checkpoint: excluding {len(exclude_keys)} non-LoRA keys from load")

        load_state = {
            "model": ModelState(
                state["model"],
                exclude_keys=exclude_keys,
                project_glm52_exact_base_dcp=validation_result.get("glm52_exact_base_dcp_projection", False),
            )
        }
        has_optimizer_state = False
        optimizer_load_keys: Optional[Set[str]] = None
        if "optimizer" in state and state["optimizer"] is not None:
            try:
                dcp_metadata = FileSystemReader(checkpoint_dir).read_metadata()
                metadata_keys = set(dcp_metadata.state_dict_metadata)
                has_optimizer_state = any(key.startswith("optimizer") for key in metadata_keys)
                optimizer_load_keys = {
                    key.removeprefix("optimizer.") for key in metadata_keys if key.startswith("optimizer.")
                }
            except Exception as exc:
                logger.warning_rank0(f"Could not inspect DCP optimizer metadata at {checkpoint_dir}: {exc}")
                has_optimizer_state = True

        if has_optimizer_state:
            load_state["optimizer"] = OptimizerState(  # type: ignore[index]
                model=state["model"],
                optimizer=state["optimizer"],
                load_keys=optimizer_load_keys,
            )
        elif "optimizer" in state and state["optimizer"] is not None:
            logger.info_rank0(f"No optimizer state found in {checkpoint_dir}; loading model state only.")

        # ``StateDictOptions(strict=False)`` in ``ModelState.load_state_dict``
        # only applies after DCP has planned and read checkpoint entries. The
        # default load planner still raises during planning when a target key is
        # absent from checkpoint metadata, which is exactly what happens when a
        # metadata-less base DCP is loaded into a LoRA-injected model. Mirror
        # this API's ``strict=False`` at DCP planner time as well.
        load_planner = DefaultLoadPlanner(allow_partial_load=not strict)

        # Native block-FP8 packed bytes and FP32 scales must be rejected from
        # DCP metadata before the loader can numerically cast them.  Use the
        # ModelState view here: unlike live EP-local parameters, it restores
        # the global expert dimension and therefore matches DCP metadata.
        expected_model_state = load_state["model"].state_dict()
        if any("packed_weight_f32" in name or name.endswith("weight_scale_inv") for name in expected_model_state):
            from xorl.ops.block_fp8_native import validate_native_fp8_dcp_checkpoint  # noqa: PLC0415

            validate_native_fp8_dcp_checkpoint(
                checkpoint_dir,
                expected_model_state,
                state_prefix="model.",
            )
            logger.info_rank0("Native block-FP8 DCP metadata preflight passed.")
        del expected_model_state

        dcp.load(
            state_dict=load_state,
            storage_reader=FileSystemReader(checkpoint_dir),
            planner=load_planner,
            process_group=process_group,
            no_dist=load_no_dist,
        )
        # Note: further per-param DTensor alignment and device fixes happen inside OptimizerState.load_state_dict

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            if os.path.exists(extra_state_path):
                state["extra_state"] = torch.load(extra_state_path, weights_only=False)
            else:
                logger.info_rank0(f"No extra_state found at {extra_state_path}, starting fresh.")

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state
