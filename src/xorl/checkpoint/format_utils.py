import os
from abc import ABC
from typing import Any, Dict, Union

from ..utils.import_utils import is_torch_version_greater_than
from ..utils.logging import get_logger


if is_torch_version_greater_than("2.4"):
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
else:
    STATE_DICT_TYPE = ABC

logger = get_logger(__name__)

_MODEL_DIR = "model"


def ckpt_to_state_dict(
    save_checkpoint_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    ckpt_manager: str = "dcp",
) -> Dict[str, Any]:
    """
    Interface to convert a checkpoint to a state_dict.
    Supported checkpoint managers:
        - dcp

    Args:
        save_checkpoint_path: Path to the checkpoint.
        output_dir: Path to the output directory.
        ckpt_manager: Checkpoint manager.
    Returns:
        state_dict: State dict.
    """
    if ckpt_manager == "dcp":
        state_dict = dcp_to_torch_state_dict(save_checkpoint_path)
    else:
        raise ValueError(f"Unknown checkpoint manager: {ckpt_manager}")
    return state_dict


def dcp_to_torch_state_dict(save_checkpoint_path: Union[str, os.PathLike]) -> STATE_DICT_TYPE:
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch state_dict.

    Args:
        save_checkpoint_path: Directory containing the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    # Load the state_dict from the DCP checkpoint
    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(save_checkpoint_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    if "state" in state_dict:
        # this happens when the model state dicts are flatten during saving
        state_dict = state_dict["state"]

    return state_dict["model"]
