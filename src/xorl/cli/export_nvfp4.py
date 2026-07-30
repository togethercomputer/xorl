"""Export a bf16 HF/safetensors directory to a servable modelopt-NVFP4 checkpoint.

Re-quantizes high-precision (bf16) weights — e.g. the master weights saved by an
NVFP4 QAT run (``save_hf_weights: true``) — to the modelopt NVFP4 layout SGLang/
vLLM serve with ``--quantization modelopt_fp4``. Reuses the block-FP8 exporter's
directory machinery (asset copy, sharded writer, GKN MoE un-fuse, MTP guard) and
the QAT round-to-nearest math (``ops/quantize/nvfp4_fake_quant``) so the exported
bytes match the values the fake-quant student trained against.

Per-tensor format (matches modelopt NVFP4):
  * ``<fqn>.weight``         uint8  ``[out, in/2]``        two FP4 codes per byte
  * ``<fqn>.weight_scale``   fp8e4m3 ``[out, in/group]``  per-group block scale
  * ``<fqn>.weight_scale_2`` fp32   ``[]``                 per-tensor global scale
plus a top-level ``hf_quant_config.json`` (``quant_algo = "NVFP4"``).

Projections the serving stack fuses into one GEMM (q/k/v -> qkv, gate/up ->
gate_up) MUST share one ``weight_scale_2`` or the fused kernel mis-scales all but
one member; a first pass computes the shared group amax.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors import safe_open

from xorl.ops.quantize.nvfp4_fake_quant import (
    _E2M1_ABS,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    _nvfp4_quantize_blocks,
)

from .export_quantized import (
    _DEFAULT_MAX_SHARD_SIZE,
    _convert_xorl_tensor_for_hf_export,
    _copy_hf_assets,
    _matches_module_skip,
    _parse_size_bytes,
    _read_config,
    _ShardWriter,
    _tensor_entries,
    _unsupported_mtp_export_reason,
)


WEIGHT_KEY = "weight"
BLOCK_SCALE_KEY = "weight_scale"
GLOBAL_SCALE_KEY = "weight_scale_2"
INPUT_SCALE_KEY = "input_scale"  # per-tensor activation global scale (fp32) -> servable W4A4


def activation_input_scale(act_amax: float) -> torch.Tensor:
    """NVFP4 activation ``input_scale`` as a 0-dim fp32 scalar (matches modelopt).

    Same FP4xFP8 normalization as the weight global scale, but over the activation
    amax observed at calibration. Serving stacks (sglang ``modelopt_fp4``) require it
    to quantize activations; a weight-only export without it serves garbage.
    """
    return torch.tensor(act_amax / (FP4_E2M1_MAX * FP8_E4M3_MAX), dtype=torch.float32)


# BF16 islands left unquantized (serving keeps them high precision). Patterns must
# glob-match the FULL HF tensor name — e.g. "model.embed_tokens.weight", not the bare
# "embed_tokens" (which matched nothing and got embeddings wrongly packed -> served garbage).
_DEFAULT_MODULES_TO_NOT_CONVERT = ["*lm_head*", "*embed_tokens*", "*mlp.gate", "*.gate.weight"]

# Projections the inference stack fuses into a single GEMM; members must share
# one global scale (weight_scale_2). Keyed by the bare projection short name.
_FUSED_GROUPS = {
    "q_proj": "qkv",
    "k_proj": "qkv",
    "v_proj": "qkv",
    "gate_proj": "gate_up",
    "up_proj": "gate_up",
}


def fused_group_key(module_fqn: str) -> Optional[str]:
    """Group id for a fused projection, or ``None`` for standalone linears."""
    short = module_fqn.rsplit(".", 1)[-1]
    grp = _FUSED_GROUPS.get(short)
    if grp is None:
        return None
    parent = module_fqn.rsplit(".", 1)[0]
    return f"{parent}::{grp}"


def _global_scale_from_amax(amax: torch.Tensor | float) -> torch.Tensor:
    amax_t = amax if isinstance(amax, torch.Tensor) else torch.tensor(float(amax))
    return (amax_t.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)).reshape(1)


def quantize_weight_to_nvfp4(
    weight: torch.Tensor,
    block_size: int = 16,
    global_scale: Optional[torch.Tensor] = None,
    act_amax: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Re-quantize a 2D weight to the modelopt NVFP4 layout.

    ``global_scale`` forces a shared per-tensor ``weight_scale_2`` (for fused
    q/k/v, gate/up); leave ``None`` for standalone linears. ``act_amax`` > 0 adds the
    per-tensor activation ``input_scale`` (makes the export servable W4A4).
    """
    w = weight.detach().contiguous()
    assert w.dim() == 2, f"expected 2D weight, got {tuple(w.shape)}"
    M, K = w.shape
    _, block_scales, gscale, codes = _nvfp4_quantize_blocks(w.float(), block_size, global_scale=global_scale)
    # Pack two 4-bit codes per byte (row-major): low nibble = even col, high = odd.
    flat = codes.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).to(torch.uint8).reshape(M, K // 2)
    entry = {
        WEIGHT_KEY: packed,
        BLOCK_SCALE_KEY: block_scales.reshape(M, K // block_size),
        # 0-dim fp32 scalar; .clone() so a shared group scale is not aliased memory
        # (safetensors refuses tensors that share storage).
        GLOBAL_SCALE_KEY: gscale.reshape(()).float().clone(),
    }
    if act_amax > 0.0:
        entry[INPUT_SCALE_KEY] = activation_input_scale(act_amax)
    return entry


def dequantize_nvfp4_export(entry: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct a bf16 weight from an NVFP4 entry (inverse of quantize)."""
    packed = entry[WEIGHT_KEY]
    M, half = packed.shape
    K = half * 2
    block_scale = entry[BLOCK_SCALE_KEY]
    block_size = K // block_scale.shape[1]
    flat = packed.reshape(-1)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=1).reshape(M, K).to(torch.int64)
    grid = torch.tensor(_E2M1_ABS, dtype=torch.float32)
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    values = sign * grid[codes & 0x7]
    eff = block_scale.float() * entry[GLOBAL_SCALE_KEY].float()  # [M, K/bs]
    eff = eff.repeat_interleave(block_size, dim=1)  # [M, K]
    return (values * eff).reshape(M, K).to(torch.bfloat16)


def write_hf_quant_config(save_dir: Path, *, group_size: int, exclude_modules: list[str]) -> Path:
    cfg = {
        "producer": {"name": "xorl-qat", "quant_method": "fake-quant-RTN"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": group_size,
            "exclude_modules": exclude_modules,
        },
    }
    path = save_dir / "hf_quant_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def _should_quantize_nvfp4(hf_name: str, tensor: torch.Tensor, skip: list[str]) -> bool:
    if not hf_name.endswith(".weight") or tensor.ndim != 2 or not tensor.is_floating_point():
        return False
    return not _matches_module_skip(hf_name, skip)


def export_hf_directory_to_nvfp4(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    group_size: int = 16,
    modules_to_not_convert: list[str] | None = None,
    max_shard_size: int = _DEFAULT_MAX_SHARD_SIZE,
    overwrite: bool = False,
    activation_amax: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Export a bf16 HF directory to a servable modelopt-NVFP4 checkpoint.

    ``activation_amax`` maps module FQN (e.g. ``model.layers.0.self_attn.q_proj``, no
    ``.weight``) to its observed activation amax; when given, each quantized linear gets
    a per-tensor ``input_scale`` so serving stacks can do W4A4 (without it they serve
    garbage). Fused members (q/k/v, gate/up) share the input so their scales match.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_dir and output_dir must differ")
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_path} exists and is not empty; pass overwrite=True")
    output_path.mkdir(parents=True, exist_ok=True)

    skip = list(modules_to_not_convert) if modules_to_not_convert is not None else list(_DEFAULT_MODULES_TO_NOT_CONVERT)
    config = _read_config(input_path)
    entries = _tensor_entries(input_path)
    weight_names = [name for name, _ in entries]
    mtp_reason = _unsupported_mtp_export_reason(config, weight_names, skip)
    if mtp_reason is not None:
        raise ValueError(mtp_reason)
    if any(name.endswith(".weight_scale_2") for name in weight_names):
        raise ValueError("Input already contains NVFP4 scales; re-export from high-precision weights")

    def _iter_members():
        for name, shard_path in entries:
            # QARL observer buffers (qarl_input_amax/qarl_weight_amax/qarl_*_scale_inv/
            # qarl_forward_count) ride along in a QARL-trained checkpoint; they are not
            # weights and break the qkv/expert un-fuse — skip them.
            if ".qarl_" in name or name.rsplit(".", 1)[-1].startswith("qarl_"):
                continue
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
            yield from _convert_xorl_tensor_for_hf_export(name, tensor, config=config)

    # Pass 1: accumulate the per-fused-group weight amax (shared global scale).
    group_amax: dict[str, torch.Tensor] = {}
    for hf_name, tensor in _iter_members():
        if not _should_quantize_nvfp4(hf_name, tensor, skip):
            continue
        gk = fused_group_key(hf_name[: -len(".weight")])
        if gk is None:
            continue
        amax = tensor.detach().float().abs().amax()
        group_amax[gk] = amax if gk not in group_amax else torch.maximum(group_amax[gk], amax)
    group_scales = {gk: _global_scale_from_amax(a) for gk, a in group_amax.items()}

    # Pass 2: quantize and write (standalone use their own scale, fused share it).
    act_amax = activation_amax or {}
    _copy_hf_assets(input_path, output_path)
    writer = _ShardWriter(output_path, max_shard_size=max_shard_size)
    n_quant = 0
    n_passthrough = 0
    n_act = 0
    for hf_name, tensor in _iter_members():
        if not _should_quantize_nvfp4(hf_name, tensor, skip):
            writer.add(hf_name, tensor.contiguous())
            n_passthrough += 1
            continue
        module_fqn = hf_name[: -len(".weight")]
        gk = fused_group_key(module_fqn)
        entry = quantize_weight_to_nvfp4(
            tensor,
            block_size=group_size,
            global_scale=group_scales.get(gk) if gk else None,
            act_amax=float(act_amax.get(module_fqn, 0.0)),
        )
        for suffix, value in entry.items():
            writer.add(f"{module_fqn}.{suffix}", value.contiguous())
        n_quant += 1
        n_act += INPUT_SCALE_KEY in entry

    writer.finalize()
    write_hf_quant_config(output_path, group_size=group_size, exclude_modules=skip)

    # Stamp the served config.json so the serving stack reads the quant method.
    config_out = output_path / "config.json"
    if config_out.exists():
        served = json.loads(config_out.read_text())
        served["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": group_size,
            "exclude_modules": skip,
        }
        config_out.write_text(json.dumps(served, indent=2))

    return {
        "output_dir": str(output_path),
        "quantized_tensors": n_quant,
        "passthrough_tensors": n_passthrough,
        "fused_groups": len(group_scales),
        "group_size": group_size,
        "input_scale_tensors": n_act,
        "serving_regime": "W4A4" if n_act else "W4 (weight-only; not servable on sglang modelopt_fp4)",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="bf16 HF/safetensors checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="destination for the NVFP4 checkpoint")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-shard-size", default=None, help="e.g. 5GB (default: exporter default)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--modules-to-not-convert",
        nargs="*",
        default=None,
        help=f"BF16 island patterns (default: {_DEFAULT_MODULES_TO_NOT_CONVERT})",
    )
    parser.add_argument(
        "--activation-amax-json",
        default=None,
        help="JSON sidecar {module_fqn: act_amax} from calibration -> per-linear input_scale (W4A4-servable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_shard = _parse_size_bytes(args.max_shard_size) if args.max_shard_size else _DEFAULT_MAX_SHARD_SIZE
    activation_amax = None
    if args.activation_amax_json:
        activation_amax = json.loads(Path(args.activation_amax_json).read_text())
    result = export_hf_directory_to_nvfp4(
        args.input_dir,
        args.output_dir,
        group_size=args.group_size,
        modules_to_not_convert=args.modules_to_not_convert,
        max_shard_size=max_shard,
        overwrite=args.overwrite,
        activation_amax=activation_amax,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
