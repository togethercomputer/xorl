"""
RoutingReplayHandler — R3 (Rollout Routing Replay) for MoE Models

Manages pre-population of routing replay data for Mixture-of-Experts training.
R3 ensures that MoE layers during training use the same expert selections that
were determined during inference rollout.

Uses the Megatron-style RoutingReplay system (models/layers/moe/routing_replay.py)
which integrates with MoEBlock.forward() via stage-based dispatch
(record / replay_forward / replay_backward).

For R3, routing decisions from inference are pre-populated into per-MoE-block
RoutingReplay instances via record(). During forward, MoEBlock pops pre-populated
routing via pop_forward(). During gradient checkpointing recompute, pop_backward()
returns the same pre-populated routing.

The handler manages:
- Decoding of routing data from various formats (base64, nested lists, dicts)
- Sequence-parallel-aware sharding of routing data across micro-batches
- Pre-population of RoutingReplay instances (one per MoE block)
- Stage lifecycle management (set_replay_stage, set_r3_mode)
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from xorl.distributed.parallel_state import get_parallel_state
from xorl.models.layers.moe.routing_replay import (
    RoutingReplay,
    set_r3_mode,
    set_replay_stage,
)

try:
    from xorl.models.layers.moe import MoEBlock
    _HAS_MOE_BLOCK = True
except ImportError:
    _HAS_MOE_BLOCK = False

logger = logging.getLogger(__name__)


class RoutingReplayHandler:
    """
    Manages R3 (Rollout Routing Replay) for MoE training.

    R3 replays the expert routing decisions from inference during training,
    ensuring that the same experts process the same tokens. This is critical
    for policy gradient methods (e.g., GRPO, importance sampling) where the
    training loss depends on matching the inference-time expert assignments.

    Args:
        model: The nn.Module model containing MoE layers.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._moe_blocks: Optional[List[nn.Module]] = None

    def get_moe_blocks(self) -> List[nn.Module]:
        """
        Find all MoE blocks in the model that have routing replay enabled.

        Uses the same discovery pattern as base.py:enable_routing_replay() —
        looks for decoder layers whose ``mlp`` attribute is a MoEBlock with
        a ``_routing_replay`` instance.

        Returns:
            List of MoEBlock modules with routing replay enabled, in model order.
        """
        if self._moe_blocks is not None:
            return self._moe_blocks

        if not _HAS_MOE_BLOCK:
            logger.warning("Could not import MoEBlock type for R3")
            return []

        moe_blocks = []
        for _name, module in self.model.named_modules():
            mlp = getattr(module, "mlp", None)
            if isinstance(mlp, MoEBlock) and getattr(mlp, "_routing_replay", None) is not None:
                moe_blocks.append(mlp)

        self._moe_blocks = moe_blocks
        return moe_blocks

    def decode_routed_experts_item(
        self,
        item: Union[str, List, dict],
        num_moe_layers: int,
    ) -> Optional[List[List[List[int]]]]:
        """
        Decode a single routed_experts item which may be base64-encoded or a nested list.

        Args:
            item: Either a base64 string, a dict with 'data' and 'shape' keys, or a nested list
            num_moe_layers: Number of MoE layers (used for shape inference)

        Returns:
            Decoded routing data as nested list [num_tokens, num_layers, topk], or None if invalid
        """
        if item is None:
            return None

        # Already a nested list - return as-is
        if isinstance(item, list):
            return item

        # Dict format from SGLang: {"data": base64_string, "shape": [num_tokens, num_layers, topk]}
        if isinstance(item, dict):
            if "data" not in item:
                logger.warning(f"R3: Dict item missing 'data' key: {item.keys()}")
                return None

            b64_data = item["data"]
            shape = item.get("shape")

            try:
                decoded = base64.b64decode(b64_data)
                arr = np.frombuffer(decoded, dtype=np.int32)
            except Exception as e:
                logger.warning(f"R3: Failed to decode base64 data: {e}")
                return None

            if shape:
                try:
                    arr = arr.reshape(shape)
                except Exception as e:
                    logger.warning(f"R3: Failed to reshape with shape {shape}: {e}")
                    return None
            else:
                arr = self._infer_shape(arr, num_moe_layers)
                if arr is None:
                    return None

            return arr.tolist()

        # Base64 string directly (legacy format)
        if isinstance(item, str):
            try:
                decoded = base64.b64decode(item)
                arr = np.frombuffer(decoded, dtype=np.int32)
                arr = self._infer_shape(arr, num_moe_layers)
                if arr is None:
                    return None
                return arr.tolist()
            except Exception as e:
                logger.warning(f"R3: Failed to decode base64 string: {e}")
                return None

        logger.warning(f"R3: Unknown routed_experts item type: {type(item)}")
        return None

    @staticmethod
    def _infer_shape(arr: np.ndarray, num_moe_layers: int) -> Optional[np.ndarray]:
        """Infer [num_tokens, num_layers, topk] shape from flat array."""
        total_elements = len(arr)
        for topk in [8, 4, 2, 1]:
            if total_elements % (num_moe_layers * topk) == 0:
                num_tokens = total_elements // (num_moe_layers * topk)
                return arr.reshape(num_tokens, num_moe_layers, topk)
        logger.warning(
            f"R3: Cannot infer shape for {total_elements} elements "
            f"with {num_moe_layers} layers"
        )
        return None

    def fill_routing_replay(
        self,
        micro_batches: List[Dict[str, Any]],
        routed_experts: Optional[List[Any]] = None,
    ) -> bool:
        """
        Pre-populate RoutingReplay instances from inference routing data.

        For each micro-batch and each MoE block, calls replay.record() with the
        pre-determined expert routing tensor. After this, MoEBlock.forward() can
        use pop_forward() (stage="replay_forward") or pop_backward()
        (stage="replay_backward") to read the pre-populated data.

        Args:
            micro_batches: List of micro-batches (for determining token counts)
            routed_experts: Routing data from inference. Each item can be:
                - Nested list with shape [num_tokens, num_layers, topk]
                - Dict with 'data' (base64) and 'shape' keys
                - Base64 string (requires shape inference)

        Returns:
            True if routing was pre-populated, False otherwise.
        """
        if routed_experts is None:
            return False

        moe_blocks = self.get_moe_blocks()
        num_moe_layers = len(moe_blocks)

        if num_moe_layers == 0:
            logger.warning("R3: routed_experts provided but no MoE layers found in model")
            return False

        # Decode each item (may be base64-encoded or nested list)
        decoded_routing = []
        for item in routed_experts:
            decoded = self.decode_routed_experts_item(item, num_moe_layers)
            if decoded is not None:
                decoded_routing.append(decoded)
            else:
                logger.warning("R3: Skipping None/invalid routing item")

        if not decoded_routing:
            logger.warning("R3: No valid routing data after decoding")
            return False

        # Infer dimensions from first decoded datum
        num_layers_in_data = len(decoded_routing[0][0])
        topk = len(decoded_routing[0][0][0])
        total_tokens_raw = sum(len(d) for d in decoded_routing)

        logger.debug(
            f"R3: Processing routing data - raw_tokens={total_tokens_raw}, "
            f"num_datums={len(decoded_routing)}, "
            f"layers={num_layers_in_data}, topk={topk}, moe_layers={num_moe_layers}"
        )

        # Build per-micro-batch routing tensors, handling packing + SP slicing
        per_mb_routing = self._build_per_mb_routing(
            micro_batches, decoded_routing, num_layers_in_data, topk
        )

        if not per_mb_routing:
            logger.warning("R3: Empty routing data after processing")
            return False

        # Pre-populate RoutingReplay instances: for each micro-batch, for each
        # MoE block, call record() with the routing tensor for that (mb, layer).
        for mb_idx, mb_routing_tensor in enumerate(per_mb_routing):
            # mb_routing_tensor: [num_tokens_mb, num_layers, topk]
            num_layers_to_use = min(num_moe_layers, mb_routing_tensor.shape[1])
            for moe_idx in range(num_layers_to_use):
                # [num_tokens_mb, topk]
                layer_routing = mb_routing_tensor[:, moe_idx, :]
                moe_blocks[moe_idx]._routing_replay.record(layer_routing)

        logger.debug(
            f"R3: Pre-populated {len(per_mb_routing)} micro-batches x "
            f"{num_layers_to_use} MoE layers into RoutingReplay instances"
        )
        return True

    def _build_per_mb_routing(
        self,
        micro_batches: List[Dict[str, Any]],
        decoded_routing: List[List],
        num_layers_in_data: int,
        topk: int,
    ) -> List[torch.Tensor]:
        """
        Build per-micro-batch routing tensors with packing-aware SP slicing.

        Groups decoded routing by micro-batch (based on packing), applies
        Ulysses SP slicing per group, and pads to match actual micro-batch
        token counts.

        Returns:
            List of tensors, each [num_tokens_mb, num_layers, topk] as torch.long.
        """
        cp_enabled = get_parallel_state().cp_enabled
        if cp_enabled:
            cp_size = get_parallel_state().cp_size
            cp_rank = get_parallel_state().cp_rank

        # Read num_samples from each micro-batch (set by SequentialPacker._finalize_packed_batch)
        micro_batch_datum_counts = [mb.get("num_samples", 1) for mb in micro_batches]
        total_datums_in_batches = sum(micro_batch_datum_counts)

        logger.debug(
            f"R3: Packing structure - {len(micro_batches)} micro-batches, "
            f"datum counts: {micro_batch_datum_counts}, "
            f"total_in_batches={total_datums_in_batches}, decoded={len(decoded_routing)}"
        )

        if total_datums_in_batches != len(decoded_routing):
            logger.warning(
                f"R3: Datum count mismatch: batches expect {total_datums_in_batches} "
                f"but have {len(decoded_routing)} routing items"
            )

        per_mb_routing = []
        datum_cursor = 0

        for mb_idx, num_datums in enumerate(micro_batch_datum_counts):
            # Concatenate routing from all datums packed into this micro-batch
            mb_routing = []
            for _ in range(num_datums):
                if datum_cursor < len(decoded_routing):
                    mb_routing.extend(decoded_routing[datum_cursor])
                    datum_cursor += 1

            mb_total_tokens = len(mb_routing)

            if cp_enabled and mb_total_tokens > 0:
                # SP-slice this concatenated block as ONE unit
                # Matches TextSequenceShardCollator: pad to ceil(T/cp_size)*cp_size, then slice
                cp_chunk_size = (mb_total_tokens + cp_size - 1) // cp_size
                pad_count = cp_chunk_size * cp_size - mb_total_tokens

                if pad_count > 0:
                    pad_entry = [list(range(topk)) for _ in range(num_layers_in_data)]
                    mb_routing = mb_routing + [pad_entry] * pad_count

                start = cp_rank * cp_chunk_size
                end = (cp_rank + 1) * cp_chunk_size
                mb_routing = mb_routing[start:end]

                logger.debug(
                    f"R3: SP MB{mb_idx} - {mb_total_tokens} tokens ({num_datums} datums), "
                    f"sp_chunk={cp_chunk_size}, pad={pad_count}, slice [{start}:{end}]"
                )

            # Pad routing to match actual micro-batch token count.
            # The packer's pad_to_multiple_of may have added padding tokens
            # that aren't in the raw routing data.
            if mb_idx < len(micro_batches) and "input_ids" in micro_batches[mb_idx]:
                mb_input_ids = micro_batches[mb_idx]["input_ids"]
                if isinstance(mb_input_ids, torch.Tensor):
                    expected_mb_tokens = mb_input_ids.shape[0] * mb_input_ids.shape[1]
                else:
                    expected_mb_tokens = (
                        len(mb_input_ids[0]) if isinstance(mb_input_ids[0], list) else len(mb_input_ids)
                    )

                if len(mb_routing) < expected_mb_tokens:
                    pad_count = expected_mb_tokens - len(mb_routing)
                    pad_entry = [list(range(topk)) for _ in range(num_layers_in_data)]
                    mb_routing.extend([pad_entry] * pad_count)
                    logger.debug(
                        f"R3: Padded MB{mb_idx} routing by {pad_count} tokens to match "
                        f"micro-batch size ({expected_mb_tokens})"
                    )

            if mb_routing:
                # Convert to tensor: [num_tokens_mb, num_layers, topk]
                per_mb_routing.append(torch.tensor(mb_routing, dtype=torch.long))

        return per_mb_routing

    def setup(
        self,
        micro_batches: List[Dict[str, Any]],
        routed_experts: Optional[List[Any]],
    ) -> bool:
        """
        Set up R3 routing replay for the current forward/backward step.

        Pre-populates RoutingReplay instances from inference routing data,
        sets R3 mode, and initializes the replay stage to "replay_backward".

        Args:
            micro_batches: List of micro-batches for the current step.
            routed_experts: Optional routing data from inference rollout.

        Returns:
            True if R3 routing was set up, False otherwise.
        """
        if routed_experts is None:
            return False

        r3_enabled = self.fill_routing_replay(micro_batches, routed_experts)
        if r3_enabled:
            set_r3_mode(True)
            set_replay_stage("replay_backward")
        return r3_enabled

    def cleanup(self) -> None:
        """
        Clean up routing replay state after forward/backward step.

        Resets the replay stage, clears all RoutingReplay instances,
        and disables R3 mode.
        """
        set_replay_stage(None)
        RoutingReplay.clear_all()
        set_r3_mode(False)
