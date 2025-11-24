from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .base_collator import DataCollator
from .packing_concat_collator import add_flash_attention_kwargs_from_position_ids
from ...data.constants import IGNORE_INDEX
from ...distributed.parallel_state import get_parallel_state
from ...utils.seqlen_pos_transform_utils import pos2culen


@dataclass
class TextSequenceShardCollator(DataCollator):
    """
    Data collator to chunk inputs according to sequence parallelism.

    This collator uses position IDs for sample packing (the default behavior).

    Args:
        pad_token_id: The id of the padding token.
    """

    pad_token_id: int = 0

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

    def sp_slice(self, tensor: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
        """
        Slices a tensor along the specified dimension for sequence parallelism.
        """
        seq_length = tensor.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return tensor.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_length: int = 0, sequential: bool = False
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        if pad_length == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        # For position_ids to create one single sequence for all padded tokens
        if sequential:
            # seq: [pad_length]
            seq = torch.arange(pad_length, device=tensor.device, dtype=tensor.dtype)

            # We want to broadcast seq along every dimension except `dim`.
            # view_shape: [1, 1, ..., pad_length(at dim), ..., 1]  (ndim entries)
            view_shape = [1] * tensor.ndim
            view_shape[dim] = pad_length

            # seq.view(view_shape): [1, 1, ..., pad_length, ..., 1]
            # expand to pad_shape:   [s0, s1, ..., pad_length, ..., s{n-1}]
            pad = seq.view(view_shape).expand(pad_shape)
        else:
            pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        # Ensure all values are tensors (handle list inputs)
        for key in list(batch.keys()):
            if key in ("input_ids", "labels", "attention_mask", "position_ids"):
                if not isinstance(batch[key], torch.Tensor):
                    value = batch[key]
                    if isinstance(value, list):
                        # Handle nested lists
                        if value and isinstance(value[0], (list, torch.Tensor)):
                            flat_list = []
                            for item in value:
                                if isinstance(item, torch.Tensor):
                                    flat_list.extend(item.tolist() if item.ndim > 0 else [item.item()])
                                else:
                                    flat_list.extend(item if isinstance(item, list) else [item])
                            batch[key] = torch.tensor(flat_list, dtype=torch.long).unsqueeze(0)
                        else:
                            batch[key] = torch.tensor(value, dtype=torch.long).unsqueeze(0)
                    else:
                        # Try generic conversion
                        try:
                            batch[key] = torch.tensor(value, dtype=torch.long).unsqueeze(0)
                        except (ValueError, TypeError):
                            # If that fails, keep as is and let it error downstream
                            pass
                elif batch[key].ndim == 1:
                    # Ensure 2D shape [batch_size, seq_len]
                    batch[key] = batch[key].unsqueeze(0)

        input_ids = batch.pop("input_ids")
        labels = batch.pop("labels")[..., 1:].contiguous()  # shift labels
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        # Mask the last token of each sequence to prevent cross-sequence prediction
        cu_seqlens = pos2culen(batch["position_ids"])
        labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX

        # sp padding
        seq_length = input_ids.size(-1)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        pad_length = sp_chunk_size * self.sp_size - seq_length

        input_ids = self.sp_padding(input_ids, dim=-1, pad_value=self.pad_token_id, pad_length=pad_length)
        labels = self.sp_padding(labels, dim=-1, pad_value=IGNORE_INDEX, pad_length=pad_length)
        batch["attention_mask"] = self.sp_padding(
            batch["attention_mask"], dim=-1, pad_value=1, pad_length=pad_length
        )
        # For position_ids to create one single sequence for all padded tokens by pass sequential=True
        batch["position_ids"] = self.sp_padding(
            batch["position_ids"], dim=-1, pad_value=0, pad_length=pad_length, sequential=True
        )

        # sp slice
        batch["input_ids"] = self.sp_slice(input_ids, dim=-1)
        batch["labels"] = self.sp_slice(labels, dim=-1)

        # Calculate Flash Attention kwargs from position_ids here when SP is enabled to use padded position_ids
        add_flash_attention_kwargs_from_position_ids(batch)

        return batch
