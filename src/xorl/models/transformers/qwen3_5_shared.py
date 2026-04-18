from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Optional

import torch


QWEN3_5_CHECKPOINT_CONVERSION_MAPPING = {
    r"^model\.language_model\.": "model.",
    r"^language_model\.": "model.",
}

QWEN3_5_CHECKPOINT_SKIP_KEY_PATTERNS = [
    r"^model\.visual\.",
    r"^visual\.",
    r"^mtp\.",
]

_LINEAR_ATTN_QKV_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$")
_LINEAR_ATTN_Z_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$")
_LINEAR_ATTN_B_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight$")
_LINEAR_ATTN_A_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight$")
_LINEAR_ATTN_CONV_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$")
_LINEAR_ATTN_OUT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$")
_LINEAR_ATTN_NORM_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.norm\.weight$")
_LINEAR_ATTN_DT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.dt_bias$")
_LINEAR_ATTN_A_LOG_PATTERN = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.A_log$")

LINEAR_ATTENTION_RING_UNSUPPORTED_MESSAGE = (
    "Native FLA CP for Qwen3.5 linear_attention currently supports Ulysses-only CP only; "
    "ring attention and hybrid CP-grid would require relayout, so this path is temporarily disabled."
)


def qwen3_5_rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def qwen3_5_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (qwen3_5_rotate_half(q_rot, interleaved=interleaved) * sin)
    k_embed = (k_rot * cos) + (qwen3_5_rotate_half(k_rot, interleaved=interleaved) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def is_excluded_module_key(key: str, exclude_modules: Iterable[str]) -> bool:
    exclude_modules = set(exclude_modules)
    if not exclude_modules:
        return False
    module_fqn = key.rsplit(".", 1)[0] if "." in key else key
    module_short_name = module_fqn.rsplit(".", 1)[-1]
    return module_short_name in exclude_modules


def has_linear_attention_layers(config: object) -> bool:
    return any(layer_type == "linear_attention" for layer_type in getattr(config, "layer_types", []))


_LINEAR_ATTN_SPLIT_PATTERN = re.compile(
    r"^(model\.layers\.(\d+)\.linear_attn)\.(q_proj|k_proj|v_proj|g_proj|a_proj|b_proj|"
    r"q_conv1d|k_conv1d|v_conv1d|o_proj|o_norm|dt_bias|A_log)\.(weight|bias)$"
)
_LINEAR_ATTN_SPLIT_PATTERN_NO_SUFFIX = re.compile(r"^(model\.layers\.(\d+)\.linear_attn)\.(dt_bias|A_log)$")

_SPLIT_TO_HF_RENAME = {
    "g_proj": "in_proj_z",
    "a_proj": "in_proj_a",
    "b_proj": "in_proj_b",
    "o_proj": "out_proj",
    "o_norm": "norm",
}

_SPLIT_QKV_PARTS = {"q_proj", "k_proj", "v_proj"}
_SPLIT_CONV_PARTS = {"q_conv1d", "k_conv1d", "v_conv1d"}


def remap_linear_attention_params_for_inference(
    buffer: list[tuple[str, "torch.Tensor"]],
) -> list[tuple[str, "torch.Tensor"]]:
    fuse_groups: dict[str, dict[str, "torch.Tensor"]] = {}
    result: list[tuple[str, "torch.Tensor"]] = []

    for name, tensor in buffer:
        m = _LINEAR_ATTN_SPLIT_PATTERN.match(name)
        if m is None:
            m = _LINEAR_ATTN_SPLIT_PATTERN_NO_SUFFIX.match(name)
        if m is None:
            result.append((name, tensor))
            continue

        prefix = m.group(1)
        proj = m.group(3)
        rest = name[m.end(3) :]

        if proj in _SPLIT_QKV_PARTS:
            key = f"{prefix}.in_proj_qkv{rest}"
            fuse_groups.setdefault(key, {})[proj] = tensor
        elif proj in _SPLIT_CONV_PARTS:
            key = f"{prefix}.conv1d{rest}"
            fuse_groups.setdefault(key, {})[proj] = tensor
        elif proj in _SPLIT_TO_HF_RENAME:
            result.append((f"{prefix}.{_SPLIT_TO_HF_RENAME[proj]}{rest}", tensor))
        else:
            result.append((f"{prefix}.{proj}{rest}", tensor))

    for fused_name, parts in fuse_groups.items():
        if "q_proj" in parts:
            ordered = [parts["q_proj"], parts["k_proj"], parts["v_proj"]]
        else:
            ordered = [parts["q_conv1d"], parts["k_conv1d"], parts["v_conv1d"]]
        result.append((fused_name, torch.cat(ordered, dim=0)))

    return result


def map_qwen3_5_linear_attention_weight(
    key: str,
    tensor: torch.Tensor,
    linear_key_dim: int,
    linear_value_dim: int,
) -> Optional[list[tuple[str, torch.Tensor]]]:
    match = _LINEAR_ATTN_QKV_PATTERN.match(key)
    if match is not None:
        layer_idx = int(match.group(1))
        return [
            (f"model.layers.{layer_idx}.linear_attn.q_proj.weight", tensor[:linear_key_dim].contiguous()),
            (
                f"model.layers.{layer_idx}.linear_attn.k_proj.weight",
                tensor[linear_key_dim : 2 * linear_key_dim].contiguous(),
            ),
            (
                f"model.layers.{layer_idx}.linear_attn.v_proj.weight",
                tensor[2 * linear_key_dim : 2 * linear_key_dim + linear_value_dim].contiguous(),
            ),
        ]

    match = _LINEAR_ATTN_CONV_PATTERN.match(key)
    if match is not None:
        layer_idx = int(match.group(1))
        return [
            (f"model.layers.{layer_idx}.linear_attn.q_conv1d.weight", tensor[:linear_key_dim].contiguous()),
            (
                f"model.layers.{layer_idx}.linear_attn.k_conv1d.weight",
                tensor[linear_key_dim : 2 * linear_key_dim].contiguous(),
            ),
            (
                f"model.layers.{layer_idx}.linear_attn.v_conv1d.weight",
                tensor[2 * linear_key_dim : 2 * linear_key_dim + linear_value_dim].contiguous(),
            ),
        ]

    for pattern, suffix in (
        (_LINEAR_ATTN_Z_PATTERN, "g_proj.weight"),
        (_LINEAR_ATTN_B_PATTERN, "b_proj.weight"),
        (_LINEAR_ATTN_A_PATTERN, "a_proj.weight"),
        (_LINEAR_ATTN_OUT_PATTERN, "o_proj.weight"),
        (_LINEAR_ATTN_NORM_PATTERN, "o_norm.weight"),
        (_LINEAR_ATTN_DT_PATTERN, "dt_bias"),
        (_LINEAR_ATTN_A_LOG_PATTERN, "A_log"),
    ):
        match = pattern.match(key)
        if match is not None:
            layer_idx = int(match.group(1))
            return [(f"model.layers.{layer_idx}.linear_attn.{suffix}", tensor)]

    return None
