"""Distributed test: shared-prefix attention composes with Ulysses SP.

Under Ulysses SP=2, a shared-prefix-repacked forward must match a standard packed
forward over the un-deduplicated layout, at every trained position. Both run under
the same SP machinery, so agreement validates that the all-to-all re-gathers the
full repacked sequence in order and the full-coordinate context applies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from xorl.distributed.parallel_state import init_parallel_state
from xorl.models.transformers.qwen3.configuration_qwen3 import Qwen3Config
from xorl.models.transformers.qwen3.modeling_qwen3 import Qwen3Model
from xorl.ops.shared_prefix import shared_prefix_remap_to_original, shared_prefix_repack_batch
from xorl.utils.device import get_nccl_backend


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than  # noqa: E402


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]

IGNORE = -100


def _build_packed(seqs, device):
    ids, labels, pos, cu = [], [], [], [0]
    for prompt, resp in seqs:
        seq = list(prompt) + list(resp)
        p = len(prompt)
        lab = [IGNORE] * len(seq)
        for j in range(p - 1, len(seq) - 1):
            lab[j] = seq[j + 1]
        ids += seq
        labels += lab
        pos += list(range(len(seq)))
        cu.append(len(ids))
    t = lambda x, dt: torch.tensor(x, device=device, dtype=dt)  # noqa: E731
    return {
        "input_ids": t(ids, torch.long).unsqueeze(0),
        "target_tokens": t(labels, torch.long).unsqueeze(0),
        "position_ids": t(pos, torch.long).unsqueeze(0),
        "cu_seq_lens_q": t(cu, torch.int32),
    }


def _shard(t, rank, world):
    """Contiguous sequence shard along dim=-1 (last dim must be divisible)."""
    n = t.size(-1)
    assert n % world == 0, f"seq len {n} not divisible by world {world}"
    L = n // world
    return t[..., rank * L : (rank + 1) * L].contiguous()


def _gather_seq(local, world):
    """All-gather contiguous sequence shards [1, L, H] -> [1, world*L, H]."""
    parts = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(parts, local.contiguous())
    return torch.cat(parts, dim=1)


def _run() -> None:
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=get_nccl_backend())
    init_parallel_state(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        ulysses_size=world,
        ringattn_size=1,
        dp_mode="none",
        device_type="cuda",
        cp_fsdp_mode="none",
    )
    device = torch.device("cuda", local_rank)

    config = Qwen3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        use_cache=False,
    )
    config._attn_implementation = "flash_attention_3"
    torch.manual_seed(0)
    model = Qwen3Model(config).to(device=device, dtype=torch.bfloat16).eval()

    # Sized so both the original (26) and repacked (20) layouts are even (world=2).
    seqs = [
        ([5, 6, 7, 8], [101, 102, 103, 104]),
        ([5, 6, 7, 8], [201, 202]),
        ([11, 12, 13, 14], [301, 302]),
        ([11, 12, 13, 14], [303, 304]),
    ]
    b = _build_packed(seqs, device)
    cu = b["cu_seq_lens_q"]
    max_len = int((cu[1:] - cu[:-1]).max().item())

    # --- standard packed forward under Ulysses ---
    with torch.no_grad():
        std_local = model(
            input_ids=_shard(b["input_ids"], local_rank, world),
            position_ids=b["position_ids"],
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=max_len,
            max_length_k=max_len,
        ).last_hidden_state
        std = _gather_seq(std_local, world)

        # --- shared-prefix repacked forward under Ulysses ---
        rp = shared_prefix_repack_batch(b)
        assert rp is not None
        ctx = rp["shared_prefix_context"]
        sp_local = model(
            input_ids=_shard(rp["input_ids"], local_rank, world),
            position_ids=rp["position_ids"],
            shared_prefix_context=ctx,
        ).last_hidden_state
        sp_rep = _gather_seq(sp_local, world)
        sp = shared_prefix_remap_to_original(sp_rep.transpose(1, 2), ctx).transpose(1, 2)

    valid = (b["target_tokens"].squeeze(0) != IGNORE).nonzero(as_tuple=True)[0]
    torch.testing.assert_close(sp[0, valid], std[0, valid], atol=3e-2, rtol=3e-2)
    if dist.get_rank() == 0:
        print(f"shared-prefix + Ulysses SP={world} parity passed ({valid.numel()} trained positions)")


def _main() -> None:
    try:
        _run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ != "__main__":

    @skip_if_gpu_count_less_than(2)
    def test_shared_prefix_ulysses_sp_parity():
        result = run_distributed_script(__file__, num_gpus=2, timeout=240)
        result.assert_success("shared-prefix + Ulysses SP parity should pass")


if __name__ == "__main__":
    _main()
