from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .base_collator import DataCollator
from .packing_concat_collator import add_flash_attention_kwargs_from_position_ids
from ...data.constants import IGNORE_INDEX
from ...distributed.parallel_state import get_parallel_state

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
        Pads a tensor with pad_length to align tensor with sp size.
        """
        if pad_length == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        if sequential:
            # Chunked arange: each 1024-token chunk is its own sequence
            # so padding doesn't create one huge fake sequence in cu_seq_lens.
            seq = torch.arange(pad_length, device=tensor.device, dtype=tensor.dtype) % 1024
            view_shape = [1] * tensor.ndim
            view_shape[dim] = pad_length
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
        labels = batch.pop("labels")
        position_ids = batch.pop("position_ids")

        # Data should already be shifted by data_processor:
        # input_ids = tokens[:-1], labels = tokens[1:] (or target_tokens from xorl_client API)
        # So input_ids, labels, and position_ids should all have the same shape
        assert input_ids.shape == labels.shape, (
            f"input_ids and labels must have same shape (data should be pre-shifted). "
            f"Got input_ids: {input_ids.shape}, labels: {labels.shape}"
        )
        assert input_ids.shape == position_ids.shape, (
            f"input_ids and position_ids must have same shape. "
            f"Got input_ids: {input_ids.shape}, position_ids: {position_ids.shape}"
        )

        # Sanity check: verify the first non-ignore label matches shifted input_ids
        # This ensures data is properly shifted without being too strict about all positions
        # (chat data may have labels[i] != input_ids[i+1] at turn boundaries)
        valid_mask = labels != IGNORE_INDEX
        if valid_mask.any():
            # Find first non-ignore position
            first_valid_idx = valid_mask.nonzero(as_tuple=True)[1][0].item()
            if first_valid_idx < labels.shape[1] - 1:  # Ensure we can check i+1
                first_label = labels[0, first_valid_idx].item()
                next_input = input_ids[0, first_valid_idx + 1].item()
                assert first_label == next_input, (
                    f"Data shift check failed: first non-ignore label should equal next input_id. "
                    f"labels[{first_valid_idx}]={first_label}, input_ids[{first_valid_idx + 1}]={next_input}. "
                    f"This suggests data is not properly shifted."
                )

        # Store original position_ids before padding for unpacking per-token outputs later
        if "_original_position_ids" not in batch:
            batch["_original_position_ids"] = position_ids.clone()

        # sp padding - pad to be divisible by sp_size
        seq_length = input_ids.size(-1)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        pad_length = sp_chunk_size * self.sp_size - seq_length

        input_ids = self.sp_padding(input_ids, dim=-1, pad_value=self.pad_token_id, pad_length=pad_length)
        labels = self.sp_padding(labels, dim=-1, pad_value=IGNORE_INDEX, pad_length=pad_length)

        if "attention_mask" in batch:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=1, pad_length=pad_length
            )

        # Pad position_ids with chunked sequential values (1024-token chunks)
        # NOTE: position_ids is NOT sliced - it stays full length because:
        # 1. cu_seqlens is computed from FULL padded position_ids
        # 2. For Ulysses, all SP ranks use the SAME cu_seqlens for flash attention
        # 3. Each SP rank only processes a slice of the sequence but needs full cu_seqlens
        position_ids = self.sp_padding(
            position_ids, dim=-1, pad_value=0, pad_length=pad_length, sequential=True
        )

        # sp slice - only slice input_ids and labels, NOT position_ids
        batch["input_ids"] = self.sp_slice(input_ids, dim=-1)
        batch["labels"] = self.sp_slice(labels, dim=-1)
        batch["position_ids"] = position_ids  # Keep full, not sliced

        # Handle RL fields (target_tokens, logprobs, advantages) for importance_sampling
        # These need to be padded and sliced the same way as labels
        rl_fields = ["target_tokens", "logprobs", "advantages", "rollout_logprobs"]
        for field in rl_fields:
            if field in batch:
                field_tensor = batch[field]
                if not isinstance(field_tensor, torch.Tensor):
                    # Determine dtype: logprobs/advantages are float, target_tokens is long
                    dtype = torch.float if field in ("logprobs", "advantages", "rollout_logprobs") else torch.long
                    if isinstance(field_tensor, list):
                        if field_tensor and isinstance(field_tensor[0], list):
                            # Nested list
                            field_tensor = torch.tensor(field_tensor[0], dtype=dtype).unsqueeze(0)
                        else:
                            field_tensor = torch.tensor(field_tensor, dtype=dtype).unsqueeze(0)
                    else:
                        field_tensor = torch.tensor(field_tensor, dtype=dtype).unsqueeze(0)
                elif field_tensor.ndim == 1:
                    field_tensor = field_tensor.unsqueeze(0)

                # Determine pad value: IGNORE_INDEX for target_tokens, 0 for others
                pad_value = IGNORE_INDEX if field == "target_tokens" else 0.0
                field_tensor = self.sp_padding(field_tensor, dim=-1, pad_value=pad_value, pad_length=pad_length)
                batch[field] = self.sp_slice(field_tensor, dim=-1)

        # Always (re)compute cu_seq_lens from the PADDED position_ids.
        # cu_seq_lens may already exist in the batch (e.g. pre-computed by
        # data_processor._finalize_packed_batch), but those values are stale
        # because they were computed BEFORE SP padding was applied.
        add_flash_attention_kwargs_from_position_ids(batch)

        return batch
