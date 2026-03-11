"""Synchronize padding across distributed ranks to prevent load imbalance."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from xorl.data.constants import IGNORE_INDEX
from xorl.distributed.parallel_state import get_parallel_state
from xorl.utils.device import get_device_type


# Pad values for known 2D sequence tensor keys: (1, seq_len)
_2D_PAD_VALUES = {
    "input_ids": 0,
    "labels": IGNORE_INDEX,
    "position_ids": 0,
    "attention_mask": 0,
    "_original_position_ids": 0,
    "target_tokens": IGNORE_INDEX,
    "logprobs": 0.0,
    "advantages": 0.0,
    "rollout_logprobs": 0.0,
}

# Pad values for known 3D sequence tensor keys: (1, seq_len, hidden_dim)
_3D_PAD_VALUES = {
    "hidden_states": 0.0,
    "hidden_states_scale": 0.0,
}


def synchronize_micro_batch_padding(
    micro_batches: List[Dict[str, Any]],
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Pad micro-batches so every rank has the same sequence lengths.

    Without synchronization, different DP ranks have different packed
    sequence lengths, causing load imbalance: faster ranks idle during
    FSDP collective operations waiting for the slowest rank.

    This function finds the global-max length for each
    (micro_batch_index, tensor_key) pair via a single all-reduce
    and pads shorter tensors to match.

    Modifies *micro_batches* in place.

    Args:
        micro_batches: List of micro-batch dicts from the dataloader.
        group: Process group for all-reduce (default: world group).
    """
    if not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return

    if not micro_batches:
        return

    # When sequence parallelism is enabled, input_ids/labels are sharded
    # across SP ranks.  Padding at the shard boundary gets *interleaved*
    # in the reconstructed full sequence after all-to-all, corrupting
    # token positions relative to cu_seq_lens and causing NaN in flash
    # attention.  Skip sync padding entirely in this case — the load
    # imbalance fix only applies to the pure DP setting.
    ps = get_parallel_state()
    if ps.cp_enabled:
        return

    # Identify which paddable keys are present
    sample = micro_batches[0]
    keys_2d = [k for k in _2D_PAD_VALUES if k in sample and isinstance(sample[k], torch.Tensor) and sample[k].dim() == 2]
    keys_3d = [k for k in _3D_PAD_VALUES if k in sample and isinstance(sample[k], torch.Tensor) and sample[k].dim() == 3]

    if not keys_2d and not keys_3d:
        return

    all_keys = keys_2d + keys_3d

    # Collect local last-dim lengths for every (micro_batch, key) pair.
    # 2D: shape[-1],  3D: shape[1] (batch, seq_len, hidden_dim)
    local_lens = []
    for mb in micro_batches:
        for key in keys_2d:
            local_lens.append(mb[key].shape[-1])
        for key in keys_3d:
            local_lens.append(mb[key].shape[1])

    # Single all-reduce on GPU (NCCL requires GPU tensors)
    lens_t = torch.tensor(local_lens, dtype=torch.long, device=get_device_type())
    dist.all_reduce(lens_t, op=dist.ReduceOp.MAX, group=group)
    target_lens = lens_t.tolist()

    # Pad tensors that are shorter than the global max
    n_keys = len(all_keys)
    for i, mb in enumerate(micro_batches):
        offset = i * n_keys

        # 2D keys
        for j, key in enumerate(keys_2d):
            target = target_lens[offset + j]
            current = mb[key].shape[-1]
            if current >= target:
                continue
            pad_amt = target - current
            mb[key] = F.pad(mb[key], (0, pad_amt), value=_2D_PAD_VALUES[key])

        # 3D keys
        for j, key in enumerate(keys_3d):
            target = target_lens[offset + len(keys_2d) + j]
            current = mb[key].shape[1]
            if current >= target:
                continue
            pad_amt = target - current
            # F.pad pads from last dim backwards: (H_left, H_right, S_left, S_right)
            mb[key] = F.pad(mb[key], (0, 0, 0, pad_amt), value=_3D_PAD_VALUES[key])
