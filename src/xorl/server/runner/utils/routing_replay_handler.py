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
from typing import Any, Dict, List, Optional, Union

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
        self._model_topk: Optional[int] = self._extract_topk(model)

    @staticmethod
    def _extract_topk(model: nn.Module) -> Optional[int]:
        """Extract num_experts_per_tok from the model config, if available.

        Qwen3.6 nests `num_experts_per_tok` under `config.text_config`, so we
        check the nested config first to avoid silently picking up the wrong
        top-k width from row 0 of mixed routing data.
        """
        config = getattr(model, "config", None)
        configs = [config]
        if config is not None:
            text_config = (
                config.get("text_config") if isinstance(config, dict) else getattr(config, "text_config", None)
            )
            if text_config is not None:
                configs.insert(0, text_config)

        for candidate in configs:
            if candidate is None:
                continue
            topk = (
                candidate.get("num_experts_per_tok")
                if isinstance(candidate, dict)
                else getattr(candidate, "num_experts_per_tok", None)
            )
            if topk is not None:
                return int(topk)
        return None

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

    def _decode_routing_array(
        self,
        item: Union[str, List, dict],
        num_moe_layers: int,
        dtype: "np.dtype",
        log_prefix: str,
    ) -> Optional[List]:
        """Decode a base64-encoded or nested-list routing array.

        Args:
            item: Base64 string, dict with 'data'/'shape' keys, or nested list.
            num_moe_layers: Used for shape inference when shape is absent.
            dtype: numpy dtype for decoding (e.g. np.int32 or np.float32).
            log_prefix: Prefix for warning messages.

        Returns:
            Nested list [num_tokens, num_layers, topk], or None if invalid.
        """
        if item is None:
            return None

        if isinstance(item, list):
            return item

        # Dict format from SGLang: {"data": base64_string, "shape": [...]}
        if isinstance(item, dict):
            if "data" not in item:
                logger.warning(f"{log_prefix}: Dict item missing 'data' key: {item.keys()}")
                return None
            b64_data = item["data"]
            shape = item.get("shape")
            try:
                decoded = base64.b64decode(b64_data)
                arr = np.frombuffer(decoded, dtype=dtype)
            except Exception as e:
                logger.warning(f"{log_prefix}: Failed to decode base64 data: {e}")
                return None
            if shape:
                try:
                    arr = arr.reshape(shape)
                except Exception as e:
                    logger.warning(f"{log_prefix}: Failed to reshape with shape {shape}: {e}")
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
                arr = np.frombuffer(decoded, dtype=dtype)
                arr = self._infer_shape(arr, num_moe_layers)
                if arr is None:
                    return None
                return arr.tolist()
            except Exception as e:
                logger.warning(f"{log_prefix}: Failed to decode base64 string: {e}")
                return None

        logger.warning(f"{log_prefix}: Unknown item type: {type(item)}")
        return None

    def decode_routed_experts_item(
        self,
        item: Union[str, List, dict],
        num_moe_layers: int,
    ) -> Optional[List[List[List[int]]]]:
        """Decode a single routed_experts item (int32 expert indices)."""
        return self._decode_routing_array(item, num_moe_layers, np.int32, "R3")

    def decode_routed_expert_logits_item(
        self,
        item: Union[str, List, dict],
        num_moe_layers: int,
    ) -> Optional[List[List[List[float]]]]:
        """Decode a single routed_expert_logits item (float32 routing weights)."""
        return self._decode_routing_array(item, num_moe_layers, np.float32, "R3 weights")

    def _infer_shape(self, arr: np.ndarray, num_moe_layers: int) -> Optional[np.ndarray]:
        """Infer [num_tokens, num_layers, topk] shape from flat array.

        Tries the model's known ``num_experts_per_tok`` first, then falls back
        to common values.
        """
        total_elements = len(arr)
        candidates = [10, 8, 6, 4, 2, 1, 16]
        if self._model_topk is not None and self._model_topk not in candidates:
            candidates.insert(0, self._model_topk)
        elif self._model_topk is not None:
            candidates.remove(self._model_topk)
            candidates.insert(0, self._model_topk)

        for topk in candidates:
            if total_elements % (num_moe_layers * topk) == 0:
                num_tokens = total_elements // (num_moe_layers * topk)
                return arr.reshape(num_tokens, num_moe_layers, topk)
        logger.warning(
            f"R3: Cannot infer shape for {total_elements} elements "
            f"with {num_moe_layers} layers (tried topk={candidates})"
        )
        return None

    def fill_routing_replay(
        self,
        micro_batches: List[Dict[str, Any]],
        routed_experts: Optional[List[Any]] = None,
        routed_expert_logits: Optional[List[Any]] = None,
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

        # Prefer the model config top-k when available, then fall back to the
        # first decoded datum. Mixed 6/8 routing rows (Qwen3.6) can otherwise
        # let row 0 pick the wrong width and crash tensorization downstream.
        num_layers_in_data = len(decoded_routing[0][0])
        topk = self._model_topk or len(decoded_routing[0][0][0])
        total_tokens_raw = sum(len(d) for d in decoded_routing)

        logger.debug(
            f"R3: Processing routing data - raw_tokens={total_tokens_raw}, "
            f"num_datums={len(decoded_routing)}, "
            f"layers={num_layers_in_data}, topk={topk}, moe_layers={num_moe_layers}"
        )

        # Build per-micro-batch routing tensors, handling packing + SP slicing
        per_mb_routing = self._build_per_mb_routing(micro_batches, decoded_routing, num_layers_in_data, topk)

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

        # Pre-populate routing weights if provided (R3 weight replay)
        if routed_expert_logits is not None:
            decoded_weights = []
            for item in routed_expert_logits:
                decoded = self.decode_routed_expert_logits_item(item, num_moe_layers)
                if decoded is not None:
                    decoded_weights.append(decoded)

            if decoded_weights:
                per_mb_weights = self._build_per_mb_routing(micro_batches, decoded_weights, num_layers_in_data, topk)
                for mb_idx, mb_weights_tensor in enumerate(per_mb_weights):
                    num_layers_to_use_w = min(num_moe_layers, mb_weights_tensor.shape[1])
                    for moe_idx in range(num_layers_to_use_w):
                        layer_weights = mb_weights_tensor[:, moe_idx, :].float()
                        moe_blocks[moe_idx]._routing_replay.record_weights(layer_weights)
                logger.debug(f"R3: Pre-populated routing weights for {len(per_mb_weights)} micro-batches")

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
        parallel_state = get_parallel_state()
        cp_enabled = parallel_state.cp_enabled
        if cp_enabled:
            cp_size = parallel_state.cp_size
            cp_rank = parallel_state.cp_rank

        def _pad_entry():
            return [list(range(topk)) for _ in range(num_layers_in_data)]

        def _num_tokens(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return int(value.numel())
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    return sum(len(row) if isinstance(row, list) else 1 for row in value)
                return len(value)
            return None

        def _first_dim(value: Any) -> Optional[int]:
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                return int(value.shape[0])
            if isinstance(value, list) and value and isinstance(value[0], list):
                return len(value)
            return None

        def _last_dim(value: Any) -> Optional[int]:
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[-1])
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    return len(value[0])
                return len(value)
            return None

        def _flatten_position_ids(value: Any) -> Optional[List[int]]:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return [int(v) for v in value.reshape(-1).tolist()]
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    return [int(v) for row in value for v in row]
                return [int(v) for v in value]
            return None

        def _resize_position_ids(position_ids: List[int], target_tokens: int) -> List[int]:
            if len(position_ids) < target_tokens:
                pad_count = target_tokens - len(position_ids)
                position_ids = position_ids + [i % 1024 for i in range(pad_count)]
            elif len(position_ids) > target_tokens:
                position_ids = position_ids[:target_tokens]
            return position_ids

        def _zigzag_reorder_routing(
            routing: List[Any],
            position_ids: List[int],
            ringattn_size: int,
            mb_idx: int,
        ) -> List[Any]:
            if ringattn_size <= 1:
                return routing

            boundaries = [i for i, pos in enumerate(position_ids) if pos == 0]
            boundaries.append(len(position_ids))
            num_subchunks = 2 * ringattn_size
            rank_parts = [[] for _ in range(ringattn_size)]

            for boundary_idx in range(len(boundaries) - 1):
                start_idx = boundaries[boundary_idx]
                end_idx = boundaries[boundary_idx + 1]
                doc_len = end_idx - start_idx
                if doc_len == 0:
                    continue
                if doc_len % num_subchunks != 0:
                    raise ValueError(
                        f"R3: MB{mb_idx} document at position {start_idx} has length {doc_len}, "
                        f"not divisible by 2*ringattn_size={num_subchunks}. "
                        "Routing replay cannot match ring-attention zigzag layout."
                    )

                subchunk_len = doc_len // num_subchunks
                chunks = [
                    routing[start_idx + subchunk_idx * subchunk_len : start_idx + (subchunk_idx + 1) * subchunk_len]
                    for subchunk_idx in range(num_subchunks)
                ]
                for ring_rank in range(ringattn_size):
                    rank_parts[ring_rank].extend(chunks[ring_rank])
                    rank_parts[ring_rank].extend(chunks[num_subchunks - 1 - ring_rank])

            return [token_routing for rank_part in rank_parts for token_routing in rank_part]

        def _resize_routing(routing: List[Any], target_tokens: Optional[int], mb_idx: int, reason: str) -> List[Any]:
            if target_tokens is None:
                return routing
            if len(routing) < target_tokens:
                pad_count = target_tokens - len(routing)
                logger.debug("R3: Padded MB%s routing by %s tokens to match %s", mb_idx, pad_count, reason)
                return routing + [_pad_entry()] * pad_count
            if len(routing) > target_tokens:
                logger.debug(
                    "R3: Truncated MB%s routing by %s tokens to match %s",
                    mb_idx,
                    len(routing) - target_tokens,
                    reason,
                )
                return routing[:target_tokens]
            return routing

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
            datum_routing = []
            for _ in range(num_datums):
                if datum_cursor < len(decoded_routing):
                    datum_routing.append(decoded_routing[datum_cursor])
                    datum_cursor += 1

            mb_routing = [token_routing for datum in datum_routing for token_routing in datum]
            mb_total_tokens = len(mb_routing)
            micro_batch = micro_batches[mb_idx] if mb_idx < len(micro_batches) else {}
            expected_mb_tokens = _num_tokens(micro_batch.get("input_ids"))

            if cp_enabled and mb_total_tokens > 0:
                # Match the actual sharded micro-batch shape. Packed batches may
                # already be padded to 128-token boundaries by SequentialPacker,
                # while unpacked/server batches are only padded to the CP size.
                # position_ids stays full-length after sequence sharding, so it is
                # the source of truth for how much routing data existed before the
                # local CP slice.
                input_ids = micro_batch.get("input_ids")
                position_ids = micro_batch.get("position_ids")
                batch_rows = _first_dim(input_ids)
                local_seq_len = _last_dim(input_ids)
                full_seq_len = _last_dim(position_ids)
                ringattn_size = getattr(parallel_state, "ringattn_size", 1)
                rowwise_unpacked = (
                    batch_rows is not None
                    and batch_rows > 1
                    and batch_rows == len(datum_routing)
                    and local_seq_len is not None
                    and full_seq_len is not None
                )

                if rowwise_unpacked:
                    sharded_routing = []
                    cp_chunk_size = local_seq_len
                    start = cp_rank * cp_chunk_size
                    end = start + cp_chunk_size
                    for row_routing in datum_routing:
                        row_routing = _resize_routing(list(row_routing), full_seq_len, mb_idx, "full row length")
                        sharded_routing.extend(row_routing[start:end])
                    mb_routing = sharded_routing
                    logger.debug(
                        "R3: SP MB%s rowwise - raw_tokens=%s, rows=%s, full_seq=%s, local_seq=%s, cp_rank=%s",
                        mb_idx,
                        mb_total_tokens,
                        batch_rows,
                        full_seq_len,
                        local_seq_len,
                        cp_rank,
                    )
                else:
                    full_tokens = _num_tokens(position_ids)
                    if full_tokens is None:
                        full_tokens = ((mb_total_tokens + cp_size - 1) // cp_size) * cp_size
                    mb_routing = _resize_routing(mb_routing, full_tokens, mb_idx, "full SP-padded length")
                    if ringattn_size > 1:
                        zigzag_position_ids = _flatten_position_ids(micro_batch.get("_original_position_ids"))
                        if zigzag_position_ids is None:
                            zigzag_position_ids = _flatten_position_ids(position_ids)
                        if zigzag_position_ids is None:
                            logger.warning(
                                "R3: MB%s ring-attention routing lacks position_ids; falling back to contiguous slice",
                                mb_idx,
                            )
                        else:
                            zigzag_position_ids = _resize_position_ids(zigzag_position_ids, len(mb_routing))
                            mb_routing = _zigzag_reorder_routing(
                                mb_routing,
                                zigzag_position_ids,
                                ringattn_size,
                                mb_idx,
                            )
                    cp_chunk_size = expected_mb_tokens or ((len(mb_routing) + cp_size - 1) // cp_size)
                    start = cp_rank * cp_chunk_size
                    end = start + cp_chunk_size
                    mb_routing = mb_routing[start:end]

                logger.debug(
                    f"R3: SP MB{mb_idx} - {mb_total_tokens} tokens ({num_datums} datums), "
                    f"sp_chunk={cp_chunk_size}, ringattn_size={ringattn_size}, slice [{start}:{end}]"
                )

            # Pad routing to match actual micro-batch token count.
            # The packer's pad_to_multiple_of may have added padding tokens
            # that aren't in the raw routing data.
            if expected_mb_tokens is not None:
                mb_routing = _resize_routing(mb_routing, expected_mb_tokens, mb_idx, "micro-batch size")

            if mb_routing:
                # Convert to tensor: [num_tokens_mb, num_layers, topk]
                per_mb_routing.append(torch.tensor(mb_routing, dtype=torch.long))

        return per_mb_routing

    def setup(
        self,
        micro_batches: List[Dict[str, Any]],
        routed_experts: Optional[List[Any]],
        routed_expert_logits: Optional[List[Any]] = None,
    ) -> bool:
        """
        Set up R3 routing replay for the current forward/backward step.

        Pre-populates RoutingReplay instances from inference routing data,
        sets R3 mode, and initializes the replay stage to "replay_backward".

        Args:
            micro_batches: List of micro-batches for the current step.
            routed_experts: Optional routing data from inference rollout.
            routed_expert_logits: Optional routing weights from inference.

        Returns:
            True if R3 routing was set up, False otherwise.
        """
        if routed_experts is None:
            return False

        r3_enabled = self.fill_routing_replay(micro_batches, routed_experts, routed_expert_logits)
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
