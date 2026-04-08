import gc
import json
import os
from abc import ABC, abstractmethod
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


def _get_model_param_keys(model: torch.nn.Module) -> List[str]:
    """Get sorted list of parameter keys from a model."""
    return sorted([name for name, _ in model.named_parameters()])


def _save_checkpoint_metadata(
    checkpoint_dir: str,
    model: torch.nn.Module,
    has_lora: bool = False,
    save_lora_only: bool = False,
) -> None:
    """
    Save checkpoint metadata to a JSON file.
    Only rank 0 writes the metadata file.
    """
    if dist.get_rank() != 0:
        return

    param_keys = _get_model_param_keys(model)
    lora_keys = [k for k in param_keys if "lora" in k.lower()]

    if save_lora_only:
        # Only LoRA keys are actually saved
        param_keys = lora_keys

    metadata = {
        "num_parameters": len(param_keys),
        "has_lora": has_lora or len(lora_keys) > 0,
        "num_lora_parameters": len(lora_keys),
        "save_lora_only": save_lora_only,
        "parameter_keys": param_keys,
    }

    metadata_path = os.path.join(checkpoint_dir, _CHECKPOINT_METADATA_FILE)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved checkpoint metadata: {len(param_keys)} params, {len(lora_keys)} LoRA params")


def _validate_checkpoint_compatibility(
    checkpoint_dir: str,
    model: torch.nn.Module,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate that a checkpoint is compatible with the current model.

    Args:
        checkpoint_dir: Path to checkpoint directory
        model: Current model to validate against
        strict: If True, raise error on mismatch. If False, return info about mismatches.

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
    model_keys = set(_get_model_param_keys(model))

    # Keys in model but not in checkpoint (e.g., LoRA params added after checkpoint was saved)
    missing_in_ckpt = model_keys - ckpt_keys
    # Keys in checkpoint but not in model (e.g., removed params)
    unexpected_in_ckpt = ckpt_keys - model_keys

    # Check if checkpoint was saved with save_lora_only
    ckpt_lora_only = ckpt_metadata.get("save_lora_only", False)

    # Check if mismatch is LoRA-related
    missing_lora_keys = [k for k in missing_in_ckpt if "lora" in k.lower()]
    missing_non_lora_keys = [k for k in missing_in_ckpt if "lora" not in k.lower()]

    result = {
        "validated": True,
        "checkpoint_has_lora": ckpt_metadata.get("has_lora", False),
        "checkpoint_lora_only": ckpt_lora_only,
        "model_has_lora": any("lora" in k.lower() for k in model_keys),
        "missing_in_checkpoint": list(missing_in_ckpt),
        "unexpected_in_checkpoint": list(unexpected_in_ckpt),
        "missing_lora_keys": missing_lora_keys,
        "missing_non_lora_keys": missing_non_lora_keys,
        "compatible": len(missing_non_lora_keys) == 0 and len(unexpected_in_ckpt) == 0,
    }

    # Log validation results
    if missing_in_ckpt or unexpected_in_ckpt:
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
            f"  - Unexpected in checkpoint: {len(unexpected_in_ckpt)} keys"
        )

        # If only LoRA keys are missing, this is expected when loading base checkpoint into LoRA model
        if len(missing_lora_keys) > 0 and len(missing_non_lora_keys) == 0 and len(unexpected_in_ckpt) == 0:
            logger.info(
                "Loading base model checkpoint into LoRA-enabled model. "
                f"LoRA parameters ({len(missing_lora_keys)} keys) will keep their initialized values."
            )
            result["load_mode"] = "base_to_lora"
        elif strict and (len(missing_non_lora_keys) > 0 or len(unexpected_in_ckpt) > 0):
            error_msg = (
                f"Checkpoint incompatible with model:\n"
                f"  Missing non-LoRA keys: {missing_non_lora_keys[:5]}{'...' if len(missing_non_lora_keys) > 5 else ''}\n"
                f"  Unexpected keys: {list(unexpected_in_ckpt)[:5]}{'...' if len(unexpected_in_ckpt) > 5 else ''}"
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

    def __init__(self, model, exclude_keys: Optional[Set[str]] = None, save_lora_only: bool = False):
        self.model = model
        self.exclude_keys = exclude_keys or set()
        self.save_lora_only = save_lora_only

        # Determine whether this is EP+FSDP2 case
        # If so, we need to restore EP-dim before saving to DCP
        self.parallel_state = get_parallel_state()
        self.ep_fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        self.should_ep_aware = self.ep_fqn2spec_info is not None and self.parallel_state.dp_mode == "fsdp2"

    @torch.no_grad()
    def state_dict(self):
        model_state_dict = get_model_state_dict(model=self.model)
        if self.should_ep_aware:
            logger.info_rank0(
                "Getting model state_dict from ModelState wrapper, would restore EP dim for Experts module"
            )
            model_state_dict = self.get_state_dict_with_ep_dim(model_state_dict)

        # Filter out excluded keys (e.g., LoRA params when loading base checkpoint)
        if self.exclude_keys:
            model_state_dict = {k: v for k, v in model_state_dict.items() if k not in self.exclude_keys}

        # LoRA-only save: keep only lora_A/lora_B parameters
        if self.save_lora_only:
            model_state_dict = {k: v for k, v in model_state_dict.items() if "lora_" in k}
            logger.info_rank0(f"LoRA-only save: keeping {len(model_state_dict)} LoRA parameters")

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
        if self.should_ep_aware:
            model_state_dict = self.get_state_dict_without_ep_dim(model_state_dict)

        # Use strict=False to allow missing LoRA parameters when loading from
        # a checkpoint that was saved before LoRA was injected
        options = StateDictOptions(strict=False)
        incompatible = set_model_state_dict(model=self.model, model_state_dict=model_state_dict, options=options)

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

        ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
        assert ep_mesh is not None

        global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in ep_fqn2spec_info and isinstance(ep_fqn2spec_info[name].placement, Shard):
                cur_spec_info = ep_fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = self._restore_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict

    def get_state_dict_without_ep_dim(self, state_dict):
        fqn2spec_info = getattr(self.model, "_fqn2spec_info", None)
        assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        ep_mesh = self.parallel_state.ep_fsdp_device_mesh["ep"]
        assert ep_mesh is not None

        global_device_mesh = self.parallel_state.ep_fsdp_device_mesh
        assert global_device_mesh.ndim == 2

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in fqn2spec_info and isinstance(fqn2spec_info[name].placement, Shard):
                cur_spec_info = fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = self._drop_ep_dim(tensor, cur_spec_info.ep_fsdp_mesh)
                state_dict[name] = tensor

        return state_dict

    def _restore_ep_dim(self, origin_tensor: torch.Tensor, device_mesh: DeviceMesh):
        """
        Restore EP dim so that DCP can be aware about EP ranks

        args:
            origin_tensor (torch.Tensor): The origin tensor.
            device_mesh (DeviceMesh): The ep device mesh.
            shard (Shard): The shard info, default Shard(0).

        """
        assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
        ep_mesh = device_mesh["ep"]

        if origin_tensor.__class__.__name__ == "DTensor":
            # EP+FSDP2
            dtensor = DTensor.from_local(
                origin_tensor._local_tensor, device_mesh=device_mesh, placements=[Shard(0), Shard(1)]
            )
        elif origin_tensor.__class__.__name__ == "Tensor":
            # If there is no FSDP
            dtensor = DTensor.from_local(origin_tensor, device_mesh=ep_mesh, placements=[Shard(0)])

        return dtensor

    def _drop_ep_dim(self, loaded_tensor: torch.Tensor, device_mesh: DeviceMesh):
        """
        Drop EP dims after loading from DCP so that EP-FSDP would not be confused
        """
        assert device_mesh.ndim == 2, f"global_mesh.ndim must be 2, got {device_mesh.ndim}"
        ep_fsdp_mesh = device_mesh["ep_fsdp"]

        if len(loaded_tensor.placements) == 2:
            tensor_to_put = DTensor.from_local(
                loaded_tensor._local_tensor, device_mesh=ep_fsdp_mesh, placements=[Shard(1)]
            )
        elif len(loaded_tensor.placements) == 1:
            tensor_to_put = loaded_tensor.to_local()
        else:
            raise RuntimeError(
                f"Expect EP parameters from checkpoints to be DTensor with 1-dim (no FSDP) or 2-dim (EP+FSDP), got {loaded_tensor}"
            )

        return tensor_to_put


class OptimizerState(Stateful):
    """
    A wrapper around an optimizer to make it stateful.

    Args:
        optimizer (Optimizer): optimizer to wrap.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # If optimizer is None (when save_optimizer=False), return empty dict
        if self.optimizer is None:
            return {}

        # MultiOptimizer is only used for EP+FSDP2 case for now,
        # and it knows how to produce a merged, flattened dict already
        if getattr(self.optimizer, "_is_multi_optimizer", False):
            return self.optimizer.state_dict()

        # Single torch optimizer
        sd = get_optimizer_state_dict(model=self.model, optimizers=self.optimizer)
        return sd

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
            self.optimizer.load_state_dict(optim_state)
            return

        # Single torch optimizer
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=optim_state,
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
            dcp.save(
                state_dict=save_state,
                storage_writer=FileSystemWriter(
                    checkpoint_dir,
                    thread_count=1,  # Reduced from 16 to avoid PyTorch concurrent write bug
                    single_file_per_rank=True,
                    sync_files=False,
                    overwrite=True,
                ),
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
        _save_checkpoint_metadata(checkpoint_dir, state["model"], save_lora_only=save_lora_only)

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

        # Validate checkpoint compatibility before loading
        validation_result = _validate_checkpoint_compatibility(checkpoint_dir, state["model"], strict=strict)

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
            all_model_keys = set(get_model_state_dict(model=state["model"]).keys())
            exclude_keys = {k for k in all_model_keys if "lora_" not in k}
            logger.info_rank0(f"LoRA-only checkpoint: excluding {len(exclude_keys)} non-LoRA keys from load")

        load_state = {"model": ModelState(state["model"], exclude_keys=exclude_keys)}
        if "optimizer" in state and state["optimizer"] is not None:
            load_state["optimizer"] = OptimizerState(model=state["model"], optimizer=state["optimizer"])  # type: ignore[index]

        dcp.load(
            state_dict=load_state,
            storage_reader=FileSystemReader(checkpoint_dir),
            process_group=process_group,
        )
        # Note: further per-param DTensor alignment and device fixes happen inside OptimizerState.load_state_dict

        if "extra_state" in state:
            extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
            os.makedirs(extra_state_dir, exist_ok=True)
            extra_state_path = os.path.join(extra_state_dir, _EXTRA_STATE_FORMAT.format(dist.get_rank()))
            state["extra_state"] = torch.load(extra_state_path, weights_only=False)

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")

        return state
