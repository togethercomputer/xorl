#!/usr/bin/env python3
"""Offline layer-0 GDN mixer replay — TRAINER side (EP8/DP-attn T1 hunt, XORL-245).

Replays the frozen hunt trace's layer-0 mixer input through xorl's GatedDeltaNet
under a matrix of contract envs, captures every sub-op boundary, and checks which
variant reproduces the armd scoring server's captured mixer output bitwise.

Run under the trainer venv with the hunt tree on PYTHONPATH:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/apanda/xorl-divergence-hunt/src \
    /home/apanda/xorl-marin-rl-6279/.venv/bin/python replay_gdn_mixer_trainer.py

Sub-op capture points (aligned with the sampler replay):
  proj_qkvz [T, 2*key_dim+2*value_dim] (serving column order q|k|v|z)
  proj_ba   [T, 2*num_v_heads] (b|a)
  conv_out  [T, conv_dim] (post-silu, q|k|v packed)
  g [T, HV] fp32, beta [T, HV] fp32
  scan_out  [T, HV, Dv] (pre-norm)
  normed    [T, HV, Dv]
  final     [T, hidden]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


DEFAULT_MODEL = (
    "/shared/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
)
DEFAULT_XORL_DUMP = (
    "/shared/apanda/ep8_dpattn_smoke_20260706/t1/hunt/xorl_components_wordle-q36-step0001-traj00007-turn01.rank0.pt"
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

    weights = {
        "in_proj_qkv": get("in_proj_qkv.weight"),
        "in_proj_z": get("in_proj_z.weight"),
        "in_proj_b": get("in_proj_b.weight"),
        "in_proj_a": get("in_proj_a.weight"),
        "conv1d": get("conv1d.weight"),
        "out_proj": get("out_proj.weight"),
        "norm": get("norm.weight"),
        "dt_bias": get("dt_bias"),
        "A_log": get("A_log"),
    }
    return weights, cfg


def build_gdn(cfg: dict, weights: dict[str, torch.Tensor], device: torch.device, layer_idx: int):
    from xorl.ops.linear_attention import GatedDeltaNet

    key_dim = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    value_dim = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    module = GatedDeltaNet(
        hidden_size=cfg["hidden_size"],
        expand_v=cfg["linear_value_head_dim"] / cfg["linear_key_head_dim"],
        head_dim=cfg["linear_key_head_dim"],
        num_heads=cfg["linear_num_key_heads"],
        num_v_heads=cfg["linear_num_value_heads"],
        mode="chunk",
        use_gate=cfg["attn_output_gate"],
        use_short_conv=True,
        conv_size=cfg["linear_conv_kernel_dim"],
        layer_idx=layer_idx,
        norm_eps=cfg["rms_norm_eps"],
    )
    module = module.to(device=device, dtype=torch.bfloat16)  # _apply keeps A_log/dt_bias fp32
    module.eval()
    with torch.no_grad():
        module.q_proj.weight.copy_(weights["in_proj_qkv"][:key_dim])
        module.k_proj.weight.copy_(weights["in_proj_qkv"][key_dim : 2 * key_dim])
        module.v_proj.weight.copy_(weights["in_proj_qkv"][2 * key_dim : 2 * key_dim + value_dim])
        module.g_proj.weight.copy_(weights["in_proj_z"])
        module.b_proj.weight.copy_(weights["in_proj_b"])
        module.a_proj.weight.copy_(weights["in_proj_a"])
        module.q_conv1d.weight.copy_(weights["conv1d"][:key_dim])
        module.k_conv1d.weight.copy_(weights["conv1d"][key_dim : 2 * key_dim])
        module.v_conv1d.weight.copy_(weights["conv1d"][2 * key_dim : 2 * key_dim + value_dim])
        module.A_log.copy_(weights["A_log"].float())
        module.dt_bias.copy_(weights["dt_bias"].float())
        module.o_norm.weight.copy_(weights["norm"])
        module.o_proj.weight.copy_(weights["out_proj"])
    return module, key_dim, value_dim


def run_variant(
    module,
    hidden: torch.Tensor,
    *,
    key_dim: int,
    value_dim: int,
    bi_gdn: bool,
    conv_contract: bool,
) -> dict[str, torch.Tensor]:
    import xorl.ops.linear_attention.layers.gated_deltanet as gdn_mod

    os.environ["XORL_BI_GDN"] = "1" if bi_gdn else "0"
    os.environ["XORL_GDN_CONV_CONTRACT"] = "1" if conv_contract else "0"

    cap: dict[str, torch.Tensor] = {}
    hooks = []

    def save_out(name):
        def hook(mod, inp, out):
            cap[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    for pname in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "a_proj"):
        hooks.append(getattr(module, pname).register_forward_hook(save_out(pname)))
    for cname in ("q_conv1d", "k_conv1d", "v_conv1d"):
        hooks.append(getattr(module, cname).register_forward_hook(save_out(cname)))

    def o_norm_pre(mod, args, kwargs=None):
        cap["scan_out"] = args[0].detach()
        if len(args) > 1:
            cap["norm_gate"] = args[1].detach()

    hooks.append(module.o_norm.register_forward_pre_hook(o_norm_pre))
    hooks.append(module.o_norm.register_forward_hook(save_out("normed")))

    # capture gating + contract-conv outputs via namespace patching
    orig_bi_gating = gdn_mod.bi_fused_gdn_gating
    orig_beta_gate = gdn_mod._sglang_compatible_beta_gate

    def patched_bi_gating(A_log, a, b, dt_bias):
        g, beta = orig_bi_gating(A_log, a, b, dt_bias)
        cap["g"], cap["beta"] = g.detach(), beta.detach()
        return g, beta

    def patched_beta_gate(b_input):
        beta = orig_beta_gate(b_input)
        cap["beta"] = beta.detach()
        return beta

    gdn_mod.bi_fused_gdn_gating = patched_bi_gating
    gdn_mod._sglang_compatible_beta_gate = patched_beta_gate

    orig_conv_contract = None
    if conv_contract:
        import xorl.ops.linear_attention.modules.conv_contract as cc_mod

        orig_conv_contract = (
            gdn_mod.causal_conv1d_qkv_contract if hasattr(gdn_mod, "causal_conv1d_qkv_contract") else None
        )
        target_mod = gdn_mod if orig_conv_contract is not None else cc_mod
        orig_fn = getattr(target_mod, "causal_conv1d_qkv_contract")

        def patched_conv(*args, **kwargs):
            q, k, v = orig_fn(*args, **kwargs)
            cap["conv_q"], cap["conv_k"], cap["conv_v"] = q.detach(), k.detach(), v.detach()
            return q, k, v

        setattr(target_mod, "causal_conv1d_qkv_contract", patched_conv)

    try:
        with torch.no_grad():
            out, _, _ = module(hidden)
        torch.cuda.synchronize()
    finally:
        for h in hooks:
            h.remove()
        gdn_mod.bi_fused_gdn_gating = orig_bi_gating
        gdn_mod._sglang_compatible_beta_gate = orig_beta_gate
        if conv_contract:
            setattr(target_mod, "causal_conv1d_qkv_contract", orig_fn)

    T = hidden.shape[1]
    result = {"final": out.detach().reshape(T, -1)}
    result["proj_qkvz"] = torch.cat([cap[p].reshape(T, -1) for p in ("q_proj", "k_proj", "v_proj", "g_proj")], dim=-1)
    result["proj_ba"] = torch.cat([cap[p].reshape(T, -1) for p in ("b_proj", "a_proj")], dim=-1)
    if conv_contract:
        result["conv_out"] = torch.cat([cap[c].reshape(T, -1) for c in ("conv_q", "conv_k", "conv_v")], dim=-1)
    else:
        result["conv_out"] = torch.cat([cap[c].reshape(T, -1) for c in ("q_conv1d", "k_conv1d", "v_conv1d")], dim=-1)
    if "g" in cap:
        result["g"] = cap["g"].reshape(T, -1)
    result["beta"] = cap["beta"].reshape(T, -1)
    result["scan_out"] = cap["scan_out"].reshape(T, -1)
    result["normed"] = cap["normed"].reshape(T, -1)
    return {k: v.cpu() for k, v in result.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--xorl-dump", default=DEFAULT_XORL_DUMP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    # forbid ambient diagnostic toggles that bypass the wrapped projections
    for env in (
        "XORL_GDN_FUSED_PROJECTIONS",
        "XORL_GDN_SGLANG_CHUNK_FORWARD",
        "XORL_GDN_PACKED_SEGMENT_LOOP",
        "XORL_GDN_DIAGNOSTIC_TP_SPLIT",
        "XORL_GDN_BACKEND",
        "XORL_BATCH_INVARIANT_MATMUL",
    ):
        os.environ.pop(env, None)

    device = torch.device("cuda")
    dump = torch.load(args.xorl_dump, map_location="cpu", weights_only=False)
    mixer_in = dump[f"model.layers.{args.layer_idx}.input_norm"].to(device=device, dtype=torch.bfloat16)
    mixer_out_ref = dump[f"model.layers.{args.layer_idx}.attention"].reshape(-1, mixer_in.shape[-1]).to(torch.bfloat16)
    if args.max_rows is not None:
        mixer_in = mixer_in[:, : args.max_rows]
        mixer_out_ref = mixer_out_ref[: args.max_rows]
    print(f"mixer_in {tuple(mixer_in.shape)}, reference out {tuple(mixer_out_ref.shape)}")

    weights, cfg = load_layer_weights(Path(args.model_path), args.layer_idx)
    module, key_dim, value_dim = build_gdn(cfg, weights, device, args.layer_idx)

    from xorl.ops.batch_invariant_ops import wrap_trunk_linears_batch_invariant

    wrapped = wrap_trunk_linears_batch_invariant(module)
    print(f"trunk wrap: {wrapped}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    variants = {
        "v0_plain": dict(bi_gdn=False, conv_contract=False),
        "v1_bigdn": dict(bi_gdn=True, conv_contract=False),
        "v2_bigdn_conv": dict(bi_gdn=True, conv_contract=True),
        "v3_conv_only": dict(bi_gdn=False, conv_contract=True),
    }
    for name, flags in variants.items():
        res = run_variant(module, mixer_in, key_dim=key_dim, value_dim=value_dim, **flags)
        final_bf16 = res["final"].to(torch.bfloat16)
        eq = torch.equal(final_bf16, mixer_out_ref)
        neq_frac = (final_bf16 != mixer_out_ref).float().mean().item()
        max_abs = (final_bf16.float() - mixer_out_ref.float()).abs().max().item()
        report[name] = {"bitwise_vs_trainer_capture": eq, "frac_neq": neq_frac, "max_abs": max_abs}
        print(f"{name}: bitwise={eq} frac_neq={neq_frac:.4f} max_abs={max_abs:.3e}")
        torch.save(res, out_dir / f"trainer_{name}.pt")

    (out_dir / "trainer_replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
