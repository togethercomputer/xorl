"""Base checkpoint handler for per-model load/save transforms."""

from typing import Callable, List, Optional, Tuple

import torch


class CheckpointHandler:
    """Per-model checkpoint loading and saving transforms.

    Models subclass this to declare how to transform weight tensors between
    the HuggingFace checkpoint format and xorl's internal parameter format.

    Loading lifecycle:
        for each (key, tensor) in checkpoint:
            results = handler.on_load_weight(key, tensor)
            for (param_name, param_tensor) in results:
                dispatch to model
        results = handler.on_load_complete()
        for (param_name, param_tensor) in results:
            dispatch to model

    Saving lifecycle:
        for each (param_name, param_tensor) in model state_dict:
            results = handler.on_save_weight(param_name, param_tensor)
            for (ckpt_key, ckpt_tensor) in results:
                add to output state_dict
    """

    def on_load_weight(
        self, key: str, tensor: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        """Process one checkpoint key/tensor during loading.

        Returns:
            List of (param_name, tensor) pairs ready to dispatch.
            Empty list = tensor was buffered (e.g., waiting for merge partner).
            One pair = 1:1 passthrough or rename.
            Multiple pairs = 1:N split (uncommon on load).
        """
        return [(key, tensor)]

    def on_skip_weight(
        self, key: str
    ) -> List[Tuple[str, torch.Tensor]]:
        """Notify the handler that a checkpoint key was skipped (not loaded from disk).

        Used by EP-aware filtered loading: out-of-range expert weight tensors
        are not read from disk, but the handler still needs to count them so
        internal buffers know when a merge group is complete.

        Returns:
            List of (param_name, tensor) pairs if the skip triggers a completed
            buffer (e.g., the last out-of-range expert completes a layer/proj group).
            Empty list otherwise.
        """
        return []

    def on_load_complete(self) -> List[Tuple[str, torch.Tensor]]:
        """Flush remaining buffers after all checkpoint shards are read.

        Returns:
            List of any remaining (param_name, tensor) pairs.
            Should also validate/warn about incomplete buffers.
        """
        return []

    def get_skip_key_fn(self) -> Optional[Callable[[str], bool]]:
        """Return a predicate that identifies checkpoint keys whose tensor data
        can be skipped during loading (i.e., not read from disk).

        Used by EP-aware filtered loading: when ``ep_size > 1``, out-of-range
        expert weight keys return True so the loader avoids reading their
        tensor data from NFS, dramatically reducing I/O.

        Returns None when no filtering is possible (default).
        """
        return None

    def on_save_weight(
        self, param_name: str, tensor: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        """Process one model parameter during saving.

        Returns:
            List of (ckpt_key, tensor) pairs for the output checkpoint.
            One pair = passthrough/rename.
            Multiple pairs = 1:N split (e.g., gate_up_proj -> gate_proj + up_proj).
        """
        return [(param_name, tensor)]

    def on_save_complete(self) -> List[Tuple[str, torch.Tensor]]:
        """Flush any remaining save buffers. Typically empty."""
        return []
