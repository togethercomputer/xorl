#!/usr/bin/env python3
"""Offline layer-0 GDN mixer replay — SAMPLER side (EP8/DP-attn T1 hunt, XORL-245).

Replays the frozen hunt trace's layer-0 mixer input through the EXACT serving
leaf-op composition of the arm-d pod (BI ops GEMMs, fused qkvzba split, triton
varlen causal conv, fused gating, SGLANG_BI_GDN_PREFILL scan, gated RMSNorm,
BI out_proj) and checks bitwise reproduction of the pod's captured mixer output.

Run under the sampler venv with the ep8-dpattn tree on PYTHONPATH:
  CUDA_VISIBLE_DEVICES=3 SGLANG_BI_GDN_PREFILL=1 \
  PYTHONPATH=/home/apanda/xorl-sglang-ep8-dpattn/python \
    /home/apanda/sglang-venv-20260705/bin/python replay_gdn_mixer_sampler.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_MODEL = (
    "/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DEFAULT_SGLANG_DUMP = (
    "/shared/apanda/ep8_dpattn_smoke_20260706/t1/hunt/sglang_residual/sglang_qwen35_residual_tp01_000000.pt"
)
DEFAULT_OUT = "/shared/apanda/ep8_dpattn_smoke_20260706/t1/hunt/mixer_replay"


def load_layer_weights(model_path: Path, layer_idx: int) -> tuple[dict[str, torch.Tensor], dict]:
    from safetensors import safe_open

    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    cfg = cfg.get("text_config", cfg)
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    prefix = f"model.layers.{layer_idx}.linear_attn"

    def get(name: str) -> torch.Tensor:
        for cand in (f"{prefix}.{name}", f"model.language_model.layers.{layer_idx}.linear_attn.{name}"):
            if cand in index:
                with safe_open(model_path / index[cand], framework="pt", device="cpu") as h:
                    return h.get_tensor(cand)
        raise KeyError(f"{prefix}.{name}")

    return {
        "in_proj_qkv": get("in_proj_qkv.weight"),
        "in_proj_z": get("in_proj_z.weight"),
        "in_proj_b": get("in_proj_b.weight"),
        "in_proj_a": get("in_proj_a.weight"),
        "conv1d": get("conv1d.weight"),
        "out_proj": get("out_proj.weight"),
        "norm": get("norm.weight"),
        "dt_bias": get("dt_bias"),
        "A_log": get("A_log"),
    }, cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--sglang-dump", default=DEFAULT_SGLANG_DUMP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--dt-bias-fp32", action="store_true", help="keep dt_bias fp32 (trainer-style) instead of serving's bf16 param"
    )
    args = parser.parse_args()

    import triton
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkvzba_split_reshape_cat_contiguous
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import ENABLE_JIT_DEEPGEMM, matmul_persistent
    from sglang.srt.layers.attention.fla.bi_gdn_prefill import bi_chunk_gated_delta_rule_prefill
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

    print(f"triton {triton.__version__}, torch {torch.__version__}, deepgemm_mm={ENABLE_JIT_DEEPGEMM}")

    device = torch.device("cuda")
    dump = torch.load(args.sglang_dump, map_location="cpu", weights_only=False)
    mixer_in = dump["components"][f"{args.layer_idx}.mixer_in"].to(torch.bfloat16).to(device)
    mixer_out_ref = dump["components"][f"{args.layer_idx}.mixer_out"].to(torch.bfloat16)
    if args.max_rows is not None:
        mixer_in = mixer_in[: args.max_rows]
        mixer_out_ref = mixer_out_ref[: args.max_rows]
    T, hidden = mixer_in.shape
    print(f"mixer_in {tuple(mixer_in.shape)}")

    weights, cfg = load_layer_weights(Path(args.model_path), args.layer_idx)
    num_k_heads = cfg["linear_num_key_heads"]
    num_v_heads = cfg["linear_num_value_heads"]
    head_k = cfg["linear_key_head_dim"]
    head_v = cfg["linear_value_head_dim"]
    key_dim = num_k_heads * head_k
    value_dim = num_v_heads * head_v
    conv_dim = 2 * key_dim + value_dim
    eps = cfg["rms_norm_eps"]

    def bf(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=torch.bfloat16)

    w_qkvz = bf(torch.cat([weights["in_proj_qkv"], weights["in_proj_z"]], dim=0))  # [q|k|v|z, H]
    w_ba = bf(torch.cat([weights["in_proj_b"], weights["in_proj_a"]], dim=0))  # [b|a, H]
    w_out = bf(weights["out_proj"])
    conv_w = bf(weights["conv1d"]).view(conv_dim, cfg["linear_conv_kernel_dim"])
    A_log = weights["A_log"].float().to(device)
    dt_bias_ckpt = weights["dt_bias"]
    if args.dt_bias_fp32:
        dt_bias = dt_bias_ckpt.float().to(device)
    else:
        dt_bias = bf(dt_bias_ckpt)  # serving: bf16 param (created under bf16 default dtype)
    dt_exact = torch.equal(dt_bias_ckpt.float(), dt_bias_ckpt.to(torch.bfloat16).float())
    print(f"dt_bias checkpoint dtype={dt_bias_ckpt.dtype}, bf16-exact={dt_exact}, replay dtype={dt_bias.dtype}")

    with torch.no_grad():
        qkvz = matmul_persistent(mixer_in, w_qkvz.t())  # [T, 12288]
        ba = matmul_persistent(mixer_in, w_ba.t())  # [T, 64]

        mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
            qkvz, ba, triton.cdiv(num_k_heads, 1), triton.cdiv(num_v_heads, 1), head_k, head_v
        )

        conv_states = torch.zeros(1, conv_dim, cfg["linear_conv_kernel_dim"] - 1, dtype=torch.bfloat16, device=device)
        conv_out = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),  # [conv_dim, T] non-contiguous -> triton varlen branch
            conv_w,
            None,
            query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=device),
            cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
            conv_states=conv_states,
            activation="silu",
            seq_lens_cpu=[T],
        ).transpose(0, 1)[:T]

        q, k, v = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)
        q = q.view(1, T, num_k_heads, head_k)
        k = k.view(1, T, num_k_heads, head_k)
        v = v.view(1, T, num_v_heads, head_v)

        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)  # fp32 [1, T, HV]

        ssm = torch.zeros(1, num_v_heads, head_v, head_k, dtype=torch.float32, device=device)
        core = bi_chunk_gated_delta_rule_prefill(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm,
            cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
            cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=device),
            scale=head_k**-0.5,
        )  # [1, T, HV, head_v]

        norm = RMSNormGated(
            head_v, eps=eps, group_size=None, norm_before_gate=True, device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            norm.weight.copy_(weights["norm"].to(device))
        normed = norm(core.reshape(-1, head_v), z.reshape(-1, head_v))  # [T*HV, head_v]

        final = matmul_persistent(normed.reshape(T, value_dim), w_out.t())  # [T, hidden]
        torch.cuda.synchronize()

    final_bf16 = final.to(torch.bfloat16).cpu()
    eq = torch.equal(final_bf16, mixer_out_ref)
    neq = (final_bf16 != mixer_out_ref).float().mean().item()
    max_abs = (final_bf16.float() - mixer_out_ref.float()).abs().max().item()
    print(f"final vs SAMPLER capture: bitwise={eq} frac_neq={neq:.4f} max_abs={max_abs:.3e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "sampler_dtfp32" if args.dt_bias_fp32 else "sampler"
    torch.save(
        {
            "proj_qkvz": qkvz.cpu(),
            "proj_ba": ba.cpu(),
            "conv_out": conv_out.cpu(),
            "g": g.reshape(T, -1).cpu(),
            "beta": beta.reshape(T, -1).cpu(),
            "z": z.reshape(T, -1).cpu(),
            "scan_out": core.reshape(T, -1).cpu(),
            "normed": normed.reshape(T, -1).cpu(),
            "final": final.cpu(),
        },
        out_dir / f"{tag}_armd.pt",
    )
    report = {
        "bitwise_vs_sampler_capture": eq,
        "frac_neq": neq,
        "max_abs": max_abs,
        "deepgemm_mm": bool(ENABLE_JIT_DEEPGEMM),
        "dt_bias_bf16_exact": bool(dt_exact),
        "dt_bias_dtype": str(dt_bias.dtype),
    }
    (out_dir / f"{tag}_replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
