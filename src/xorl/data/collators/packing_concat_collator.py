from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
import logging

import torch
from torch.utils.data._utils.collate import default_collate
from ...distributed.parallel_state import get_parallel_state
from ...utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids
from .base_collator import DataCollator

logger = logging.getLogger(__name__)



def add_flash_attention_kwargs_from_position_ids(
    batch: Dict[str, "torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Calculate and add Flash Attention kwargs (cu_seq_lens and max_length) from position_ids.

    Pass down already computed cu_seq_lens and max_length as the HF transformers
    FlashAttentionKwargs naming so that it can be used without recomputation every layer.
    HF model code would handle the pass down of those kwargs for us.
    Note that the recomputation would cause host->device sync which hurts performance and
    stability due to CPU instability.

    Args:
        batch: The batch dictionary containing position_ids. Will be modified in-place to add
               cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k.

    Returns:
        Tuple of (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k) for additional use.
    """
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
        batch["position_ids"]
    )

    batch["cu_seq_lens_q"] = cu_seq_lens_q
    batch["cu_seq_lens_k"] = cu_seq_lens_k
    batch["max_length_q"] = max_length_q
    batch["max_length_k"] = max_length_k

    return cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k



@dataclass
class PackingConcatCollator(DataCollator):
    """
    Data collator with packing by position ids.
    
    Args:
        pad_to_multiple_of: Pad packed sequences to a multiple of this value for optimal GPU performance.
    """
    
    pad_to_multiple_of: int = 128

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        """
        Flatten and concatenate packed sequences for flash attention.

        Input structure: List[Dict] (flattened by FlattenCollator)
        - Flat list of sequence dicts with keys like 'input_ids', 'labels', etc.
        - Each dict represents one sequence to be concatenated

        Output structure: Dict[str, Tensor]
        - ALL sequences are concatenated into a single sequence
        - 1D tensors (input_ids, labels, etc.): [1, total_seq_len]
        - 2D tensors (hidden_states, hidden_states_scale): [1, total_seq_len, hidden_dim]
        - Batch size is always 1 for flash attention with packing
        """
        # Validate input structure
        if not features:
            raise ValueError("PackingConcatCollator received empty features list")

        import logging
        logger = logging.getLogger(__name__)

        # Input should be a flat list of dicts from FlattenCollator
        assert isinstance(features[0], dict), (
            f"Expected dict from FlattenCollator, but got {type(features[0]).__name__}"
        )

        batch = {}
        for input_name in features[0].keys():
            # Handle 1D tensors (input_ids, labels, etc.) and 2D tensors (hidden_states, hidden_states_scale)
            # IMPORTANT: loss_fn_inputs fields (target_tokens, logprobs, advantages) must be concatenated, not batched!
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids", "target_tokens", "logprobs", "advantages"):
                # 1D tensors: concatenate along sequence dimension
                tensors = [feature[input_name] for feature in features]

                # Assert all are tensors and 1D
                for i, t in enumerate(tensors):
                    assert isinstance(t, torch.Tensor), (
                        f"Expected tensor for key '{input_name}', but got {type(t).__name__} at index {i}"
                    )
                    assert t.ndim == 1, (
                        f"Expected 1D tensor for key '{input_name}', but got shape {t.shape} at index {i}"
                    )

                # Concatenate and add batch dimension of 1: (total_seq_len,) -> (1, total_seq_len)
                batch[input_name] = torch.cat(tensors, dim=0).unsqueeze(0)
                
            elif input_name in ("hidden_states", "hidden_states_scale"):
                # 2D tensors: concatenate along sequence dimension (dim 0)
                tensors = [feature[input_name] for feature in features]

                # Assert all are tensors and 2D
                for i, t in enumerate(tensors):
                    assert isinstance(t, torch.Tensor), (
                        f"Expected tensor for key '{input_name}', but got {type(t).__name__} at index {i}"
                    )
                    assert t.ndim == 2, (
                        f"Expected 2D tensor for key '{input_name}', but got shape {t.shape} at index {i}"
                    )

                # Concatenate along sequence dimension and add batch dimension: 
                # (seq_len, hidden_dim) -> (total_seq_len, hidden_dim) -> (1, total_seq_len, hidden_dim)
                batch[input_name] = torch.cat(tensors, dim=0).unsqueeze(0)
                
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        # Generate position_ids if not present
        if "position_ids" not in batch:
            # Generate position_ids - each sequence gets its own [0, 1, 2, ...]
            position_ids_list = []
            for feature in features:
                seq_len = len(feature["input_ids"])
                position_ids_list.append(torch.arange(seq_len))
            batch["position_ids"] = torch.cat(position_ids_list, dim=0).unsqueeze(0)

        # Generate attention_mask if not present
        if "attention_mask" not in batch:
            seq_len = batch["input_ids"].shape[1]
            batch["attention_mask"] = torch.ones(1, seq_len, dtype=torch.long)

        # Pad sequences to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of > 1:
            seq_len = batch["input_ids"].shape[1]
            pad_length = (self.pad_to_multiple_of - seq_len % self.pad_to_multiple_of) % self.pad_to_multiple_of
            
            if pad_length > 0:
                # Pad 1D tensors (input_ids, labels, etc.)
                if "input_ids" in batch:
                    batch["input_ids"] = torch.nn.functional.pad(
                        batch["input_ids"], (0, pad_length), value=0
                    )
                if "labels" in batch:
                    batch["labels"] = torch.nn.functional.pad(
                        batch["labels"], (0, pad_length), value=-100
                    )
                if "attention_mask" in batch:
                    batch["attention_mask"] = torch.nn.functional.pad(
                        batch["attention_mask"], (0, pad_length), value=0
                    )
                if "position_ids" in batch:
                    # Pad position_ids with sequential values
                    pad_positions = torch.arange(
                        0, pad_length, 
                        dtype=batch["position_ids"].dtype,
                        device=batch["position_ids"].device
                    )
                    batch["position_ids"] = torch.cat(
                        [batch["position_ids"], pad_positions.unsqueeze(0)], dim=1
                    )
                
                # Pad 2D tensors (hidden_states, hidden_states_scale)
                if "hidden_states" in batch:
                    batch["hidden_states"] = torch.nn.functional.pad(
                        batch["hidden_states"], (0, 0, 0, pad_length), value=0.0
                    )
                if "hidden_states_scale" in batch:
                    batch["hidden_states_scale"] = torch.nn.functional.pad(
                        batch["hidden_states_scale"], (0, 0, 0, pad_length), value=0.0
                    )

        # cu_seq_lens_q should equal to cu_seq_lens_k and max_length_q should equal to max_length_k
        if "position_ids" in batch:

            if not get_parallel_state().sp_enabled:
                # We only enter here to pass down cu_seqlens and max_length when sequence parallelism is not enabled.
                # When sp_enabled is True, position_ids will be padded later, so we calculate them after padding
                cu_seq_lens_q, _, _, _ = add_flash_attention_kwargs_from_position_ids(batch)
            else:
                # Still need cu_seq_lens_q for label masking even when sp_enabled
                (cu_seq_lens_q, _), (_, _) = prepare_fa_kwargs_from_position_ids(batch["position_ids"])

            # CRITICAL BUGFIX: Mask advantages at packed sequence boundaries
            # The last token of each packed sequence should NOT predict the first token of the next sequence
            # because they belong to different contexts. This prevents cross-sequence attention leakage.
            #
            # IMPORTANT: pos2culen returns boundaries in FLATTENED coordinates (0...B*T-1),
            # so we must mask in flattened space, not as column indices.
            #
            # For importance_sampling loss, we mask ADVANTAGES (not labels), because that's what
            # controls which tokens contribute to the loss.
            if "advantages" in batch and cu_seq_lens_q.numel() > 2:
                # cu_seq_lens_q[1:-1] gives the start indices of sequences 2, 3, ..., N in flattened coordinates
                # We want to mask the LAST token of sequences 1, 2, ..., N-1
                advantages = batch["advantages"].clone()
                advantages_flat = advantages.view(-1)  # flatten to match cu_seqlens coordinate system
                boundary_last_token_idx = cu_seq_lens_q[1:-1] - 1  # last token of each segment except the final segment
                advantages_flat[boundary_last_token_idx] = 0.0  # Set advantage to 0 to exclude from loss
                batch["advantages"] = advantages_flat.view_as(advantages)  # reshape back to original shape

            # Also mask labels for compatibility with other loss functions
            IGNORE_INDEX = -100
            if "labels" in batch and cu_seq_lens_q.numel() > 2:
                labels = batch["labels"].clone()
                labels_flat = labels.view(-1)
                boundary_last_token_idx = cu_seq_lens_q[1:-1] - 1
                labels_flat[boundary_last_token_idx] = IGNORE_INDEX
                batch["labels"] = labels_flat.view_as(labels)

        return batch


