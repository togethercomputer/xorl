#!/usr/bin/env python3
"""Does Marlin MoE change bytes for the same rows inside a bigger batch?

Replay question: the trainer's EP-gathered batch is 8 duplicated copies of the
serving rows. If fused_marlin_moe's output for row i depends on the total
gathered row count, the duplication breaks the byte contract structurally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "submodules/xorl-sglang/python"))

from qualify_dsv4_marlin_lora import _make_native_mxfp4, _prepare_marlin  # noqa: E402


def main() -> None:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", moe_runner_backend="marlin", moe_a2a_backend="none")
    )
    experts, hidden, intermediate, top_k = 32, 4096, 2048, 6
    w13, w2, s13, s2 = _make_native_mxfp4(experts=experts, hidden=hidden, intermediate=intermediate)
    w13, s13 = _prepare_marlin(w13, s13, size_n=2 * intermediate, size_k=hidden)
    w2, s2 = _prepare_marlin(w2, s2, size_n=hidden, size_k=intermediate)

    generator = torch.Generator(device="cuda").manual_seed(17)
    x10 = torch.randn((10, hidden), dtype=torch.bfloat16, device="cuda", generator=generator) / 64
    ids10 = (
        torch.arange(10, dtype=torch.int32, device="cuda")[:, None] * top_k
        + torch.arange(top_k, dtype=torch.int32, device="cuda")[None, :]
    ) % experts
    wts10 = torch.full((10, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda")

    def run(x, ids, wts):
        logits = torch.zeros((x.shape[0], experts), dtype=torch.float32, device="cuda")
        return fused_marlin_moe(
            hidden_states=x,
            w1=w13,
            w2=w2,
            w1_scale=s13,
            w2_scale=s2,
            gating_output=logits,
            topk_weights=wts,
            topk_ids=ids,
            num_bits=4,
            is_k_full=True,
            inplace=False,
            routed_scaling_factor=1.5,
            clamp_limit=10.0,
            expert_map=None,
            global_num_experts=experts,
        )

    out10 = run(x10, ids10, wts10)
    x80 = x10.repeat(8, 1).contiguous()
    ids80 = ids10.repeat(8, 1).contiguous()
    wts80 = wts10.repeat(8, 1).contiguous()
    out80 = run(x80, ids80, wts80)

    same = torch.equal(out80[:10].view(torch.uint8), out10.view(torch.uint8))
    d = (out80[:10].float() - out10.float()).abs()
    print(
        {
            "rows10_vs_rows80_first10_byte_equal": bool(same),
            "absmax": float(d.max()),
            "ndiff": int((out80[:10] != out10).sum()),
        }
    )


if __name__ == "__main__":
    main()
