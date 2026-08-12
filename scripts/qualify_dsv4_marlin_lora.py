#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Qualify native MXFP4 Marlin base/LoRA value composition on SM90."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


GROUP_SIZE = 32
MAX_SELECTED_EXPERTS_BYTES = 64 * 1024 * 1024


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def _load_selected_experts(
    path: Path,
    *,
    tokens: int,
    top_k: int,
    global_num_experts: int,
) -> torch.Tensor:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.suffix != ".npy":
        raise ValueError("captured selected experts must be a local .npy file")
    if resolved.stat().st_size > MAX_SELECTED_EXPERTS_BYTES:
        raise ValueError("captured selected experts file exceeds the 64 MiB safety limit")
    with resolved.open("rb") as handle:
        captured = np.load(handle, allow_pickle=False)
    if not isinstance(captured, np.ndarray) or captured.dtype.kind not in {"i", "u"}:
        raise ValueError("captured selected experts must be an integer NumPy array")
    if captured.ndim != 2 or captured.shape[1] != top_k or captured.shape[0] < 1:
        raise ValueError(f"captured selected experts must have shape [rows, {top_k}], got {tuple(captured.shape)}")
    captured = captured[:tokens].astype(np.int64, copy=False)
    if np.any(captured < -1) or np.any(captured >= global_num_experts):
        raise ValueError("captured selected expert IDs must be -1 or fall within the global expert range")
    return torch.from_numpy(np.ascontiguousarray(captured))


def _make_native_mxfp4(*, experts: int, hidden: int, intermediate: int):
    generator = torch.Generator(device="cuda").manual_seed(20260810)
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    # E8M0 byte 120 is 2**-7: random E2M1 values remain comfortably finite.
    s13 = torch.full(
        (experts, 2 * intermediate, hidden // GROUP_SIZE),
        120,
        dtype=torch.uint8,
        device="cuda",
    )
    s2 = torch.full(
        (experts, hidden, intermediate // GROUP_SIZE),
        120,
        dtype=torch.uint8,
        device="cuda",
    )
    return w13, w2, s13, s2


def _prepare_marlin(weight: torch.Tensor, scales: torch.Tensor, *, size_n: int, size_k: int):
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
    from sglang.srt.layers.quantization.marlin_utils_fp4 import mxfp4_marlin_process_scales

    permutation = torch.empty(0, dtype=torch.int32, device=weight.device)
    repacked = []
    processed_scales = []
    numerical_scales = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    for expert_idx in range(weight.shape[0]):
        repacked.append(
            gptq_marlin_repack(
                b_q_weight=weight[expert_idx].view(torch.int32).T.contiguous(),
                perm=permutation,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
        )
        marlin_scales = marlin_permute_scales(
            s=numerical_scales[expert_idx].T.contiguous(),
            size_k=size_k,
            size_n=size_n,
            group_size=GROUP_SIZE,
        )
        processed_scales.append(
            mxfp4_marlin_process_scales(
                marlin_scales,
                input_dtype=torch.bfloat16,
            )
        )
    return torch.stack(repacked), torch.stack(processed_scales)


def _lora_info(*, tokens: int, experts: int, hidden: int, intermediate: int, nonzero: bool):
    from sglang.srt.lora.lora_moe_runners import LoRAInfo

    device = torch.device("cuda")
    gate_a = torch.zeros((1, experts, 1, hidden), dtype=torch.bfloat16, device=device)
    gate_b = torch.zeros((1, experts, 2 * intermediate, 1), dtype=torch.bfloat16, device=device)
    down_a = torch.zeros((1, experts, 1, intermediate), dtype=torch.bfloat16, device=device)
    down_b = torch.zeros((1, experts, hidden, 1), dtype=torch.bfloat16, device=device)
    if nonzero:
        # Fill every expert's factors so distinguishability holds under any
        # captured real routing (a single-expert fill can land on an expert
        # the routing never selects, e.g. rank-local expert 0).
        gate_a[0].fill_(1 / 64)
        gate_b[0].fill_(1 / 64)
        down_a[0].fill_(1 / 64)
        down_b[0].fill_(1 / 64)
    return LoRAInfo(
        gate_up_lora_a_weights=gate_a,
        gate_up_lora_b_weights=gate_b,
        down_lora_a_weights=down_a,
        down_lora_b_weights=down_b,
        seg_indptr=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        req_to_lora=torch.tensor([0], dtype=torch.int32, device=device),
        lora_ranks=torch.tensor([1], dtype=torch.int32, device=device),
        adapter_enabled=torch.tensor([1], dtype=torch.int32, device=device),
        token_lora_mapping=torch.zeros(tokens, dtype=torch.int32, device=device),
        max_lora_rank=1,
        num_experts=experts,
        has_active_lora=True,
        hidden_size=hidden,
    )


def qualify(
    *,
    ep: bool = False,
    tokens: int = 4,
    hot_experts: int = 0,
    force_block_size_m: int | None = None,
    selected_experts_path: Path | None = None,
    ep_rank: int = 0,
    repeats: int = 2,
) -> dict:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if force_block_size_m is not None:
        import sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe as fused_marlin_module
        import sglang.srt.lora.lora_moe_runner_marlin as lora_marlin_module

        def _forced_block_size(**_kwargs):
            return force_block_size_m

        fused_marlin_module.select_marlin_moe_block_size_m = _forced_block_size
        lora_marlin_module.select_marlin_moe_block_size_m = _forced_block_size

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9:
        raise RuntimeError(f"DSV4 H100 contract requires SM90, got sm{capability[0]}{capability[1]}")

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", moe_runner_backend="marlin", moe_a2a_backend="none")
    )
    experts, hidden, intermediate, top_k = 32, 4096, 2048, 6
    w13, w2, s13, s2 = _make_native_mxfp4(
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    w13, s13 = _prepare_marlin(w13, s13, size_n=2 * intermediate, size_k=hidden)
    w2, s2 = _prepare_marlin(w2, s2, size_n=hidden, size_k=intermediate)

    generator = torch.Generator(device="cuda").manual_seed(17)
    x = (
        torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 64
    )
    topk_ids = torch.arange(top_k, dtype=torch.int32, device="cuda").repeat(tokens, 1)
    global_num_experts = experts
    expert_map = None
    if ep:
        global_num_experts = 256
        expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[:experts] = torch.arange(experts, dtype=torch.int32, device="cuda")
        # StandardDispatcher maps global routes into this rank's local range
        # and represents every non-local contribution with -1.
        rows = torch.arange(tokens, dtype=torch.int32, device="cuda")[:, None]
        topk_ids[:, : top_k // 2] = (
            rows * (top_k // 2) + torch.arange(top_k // 2, dtype=torch.int32, device="cuda")[None, :]
        ) % experts
        topk_ids[:, top_k // 2 :] = -1
        topk_ids[::4, :] = -1
        if hot_experts:
            if not 1 <= hot_experts <= top_k:
                raise ValueError("hot_experts must be between one and top_k")
            topk_ids.fill_(-1)
            topk_ids[:, :hot_experts] = torch.arange(hot_experts, dtype=torch.int32, device="cuda")
        if selected_experts_path is not None:
            captured = _load_selected_experts(
                selected_experts_path,
                tokens=tokens,
                top_k=top_k,
                global_num_experts=global_num_experts,
            )
            local_start = ep_rank * experts
            localized = captured - local_start
            localized = torch.where(
                (localized >= 0) & (localized < experts),
                localized,
                torch.full_like(localized, -1),
            ).to(torch.int32)
            topk_ids.fill_(-1)
            topk_ids[: localized.shape[0]].copy_(localized.to(device="cuda"))
    topk_weights = torch.full(
        (tokens, top_k),
        1 / top_k,
        dtype=torch.float32,
        device="cuda",
    )
    router_logits = torch.zeros((tokens, experts), dtype=torch.float32, device="cuda")

    base = fused_marlin_moe(
        hidden_states=x,
        w1=w13,
        w2=w2,
        w1_scale=s13,
        w2_scale=s2,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_bits=4,
        is_k_full=True,
        inplace=False,
        routed_scaling_factor=1.5,
        clamp_limit=10.0,
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )
    base_repeat = fused_marlin_moe(
        hidden_states=x,
        w1=w13,
        w2=w2,
        w1_scale=s13,
        w2_scale=s2,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_bits=4,
        is_k_full=True,
        inplace=False,
        routed_scaling_factor=1.5,
        clamp_limit=10.0,
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )

    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        layer_id=0,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        inplace=False,
    )
    quant_info = MarlinMoeQuantInfo(
        w13_qweight=w13,
        w2_qweight=w2,
        w13_scales=s13,
        w2_scales=s2,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        ),
    )
    runner = MoeRunner(MoeRunnerBackend.MARLIN, config, lora_enabled=True)
    zero = runner.run(
        dispatch,
        quant_info,
        lora_info=_lora_info(
            tokens=tokens,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            nonzero=False,
        ),
    ).hidden_states
    zero_repeat = runner.run(
        dispatch,
        quant_info,
        lora_info=_lora_info(
            tokens=tokens,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            nonzero=False,
        ),
    ).hidden_states
    nonzero = runner.run(
        dispatch,
        quant_info,
        lora_info=_lora_info(
            tokens=tokens,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            nonzero=True,
        ),
    ).hidden_states
    torch.cuda.synchronize()

    equal = torch.equal(base.view(torch.uint8), zero.view(torch.uint8))
    base_repeat_equal = torch.equal(base.view(torch.uint8), base_repeat.view(torch.uint8))
    zero_repeat_equal = torch.equal(zero.view(torch.uint8), zero_repeat.view(torch.uint8))
    repeat_flips = 0
    for _ in range(max(0, repeats - 2)):
        again = fused_marlin_moe(
            hidden_states=x,
            w1=w13,
            w2=w2,
            w1_scale=s13,
            w2_scale=s2,
            gating_output=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_bits=4,
            is_k_full=True,
            inplace=False,
            routed_scaling_factor=1.5,
            clamp_limit=10.0,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
        )
        if not torch.equal(base.view(torch.uint8), again.view(torch.uint8)):
            repeat_flips += 1
    base_repeat_equal = base_repeat_equal and repeat_flips == 0
    changed = not torch.equal(zero.view(torch.uint8), nonzero.view(torch.uint8))
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(capability),
        "shape": {
            "tokens": tokens,
            "local_experts": experts,
            "hidden": hidden,
            "intermediate": intermediate,
            "top_k": top_k,
            "global_experts": global_num_experts,
        },
        "ep": ep,
        "hot_experts": hot_experts,
        "selected_experts_path": (str(selected_experts_path) if selected_experts_path is not None else None),
        "ep_rank": ep_rank,
        "force_block_size_m": force_block_size_m,
        "repeats": repeats,
        "repeat_flips": repeat_flips,
        "base_zero_byte_equal": equal,
        "base_repeat_byte_equal": base_repeat_equal,
        "zero_repeat_byte_equal": zero_repeat_equal,
        "nonzero_distinguishable": changed,
        "base_sha256": _digest(base),
        "zero_sha256": _digest(zero),
        "nonzero_sha256": _digest(nonzero),
        "finite": {
            "base": bool(torch.isfinite(base).all()),
            "zero": bool(torch.isfinite(zero).all()),
            "nonzero": bool(torch.isfinite(nonzero).all()),
        },
    }
    if not equal:
        mismatch = base.view(torch.uint8) != zero.view(torch.uint8)
        result["base_zero_mismatched_bytes"] = int(mismatch.sum())
        element_mismatch = base.view(torch.int16) != zero.view(torch.int16)
        first = element_mismatch.flatten().nonzero().flatten()[:16]
        base_bits = base.view(torch.int16).flatten()[first].cpu().tolist()
        zero_bits = zero.view(torch.int16).flatten()[first].cpu().tolist()
        result["first_mismatched_elements"] = [
            {
                "flat_index": int(index),
                "base_bits_hex": f"{int(base_bit) & 0xFFFF:04x}",
                "zero_bits_hex": f"{int(zero_bit) & 0xFFFF:04x}",
                "base": float(base.flatten()[index]),
                "zero": float(zero.flatten()[index]),
            }
            for index, base_bit, zero_bit in zip(first, base_bits, zero_bits)
        ]
    if not equal or not base_repeat_equal or not zero_repeat_equal or not changed or not all(result["finite"].values()):
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hot-experts", type=int, default=0)
    parser.add_argument("--force-block-size-m", type=int)
    parser.add_argument(
        "--selected-experts-path",
        type=Path,
        help="local .npy integer array with shape [rows, top_k]",
    )
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.force_block_size_m not in (None, 8, 16, 32, 48, 64):
        parser.error("--force-block-size-m must be one of 8, 16, 32, 48, or 64")
    if args.selected_experts_path is not None and not args.ep:
        parser.error("--selected-experts-path requires --ep")
    if not 0 <= args.ep_rank < 8:
        parser.error("--ep-rank must be in [0, 7]")
    result = qualify(
        ep=args.ep,
        tokens=args.tokens,
        hot_experts=args.hot_experts,
        force_block_size_m=args.force_block_size_m,
        selected_experts_path=args.selected_experts_path,
        ep_rank=args.ep_rank,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
