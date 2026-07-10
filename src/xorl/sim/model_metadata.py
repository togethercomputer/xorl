"""Resolve lightweight model metadata needed by the simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


try:
    from .schemas import ModelMetadata
except ImportError:  # pragma: no cover - exercised by direct script execution
    from schemas import ModelMetadata


KNOWN_MODEL_METADATA: dict[str, dict[str, int]] = {
    "Qwen/Qwen3-235B-A22B": {
        "num_experts": 128,
        "top_k": 8,
        "num_hidden_layers": 94,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1536,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "num_experts": 128,
        "top_k": 8,
        "num_hidden_layers": 94,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1536,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3.5-397B-A17B": {
        "num_experts": 512,
        "top_k": 10,
        "num_hidden_layers": 60,
        "hidden_size": 4096,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "num_experts": 256,
        "top_k": 8,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3.6-35B-A3B": {
        "num_experts": 256,
        "top_k": 8,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3.6-35B-A3B-FP8": {
        "num_experts": 256,
        "top_k": 8,
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-30B-A3B": {
        # Non-Coder Qwen3-30B-A3B (model_type=qwen3_moe). Verified against the cached HF
        # config.json: decoder_sparse_step=1 and mlp_only_layers=[] mean every layer is MoE,
        # so the dense intermediate_size=6144 is unused for FFN params (the memory ledger
        # ignores it once moe_intermediate_size is set). Shape params are identical to the
        # Coder-30B variant; the two differ only in max_position_embeddings (40960 vs 262144)
        # and rope_theta, neither of which changes training fwd/bwd shapes or param counts.
        # Kept as a distinct identity so its evidence stream is not confused with Coder-30B.
        "num_experts": 128,
        "top_k": 8,
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-Coder-30B-A3B": {
        "num_experts": 128,
        "top_k": 8,
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
        "num_experts": 128,
        "top_k": 8,
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-32B": {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
    "Qwen/Qwen3-8B": {
        # Dense Qwen3-8B (model_type=qwen3), verified against its Hugging Face config.
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    },
}


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}


def model_ref_from_config(raw_config: dict[str, Any]) -> str | None:
    model = _section(raw_config, "model")
    candidates = (
        model.get("config_path"),
        model.get("model_path"),
        model.get("model_name"),
        raw_config.get("config_path"),
        raw_config.get("model_path"),
        raw_config.get("model_name"),
    )
    for value in candidates:
        if value:
            return str(value)
    return None


def default_hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        expanded = root.expanduser()
        if expanded not in seen:
            seen.add(expanded)
            deduped.append(expanded)
    return deduped


def _candidate_config_paths(model_ref: str, hf_cache_roots: list[Path]) -> list[Path]:
    ref_path = Path(model_ref).expanduser()
    candidates: list[Path] = []
    if ref_path.is_file():
        candidates.append(ref_path)
    elif ref_path.is_dir():
        candidates.append(ref_path / "config.json")

    if "/" in model_ref and not ref_path.exists():
        cache_name = "models--" + model_ref.replace("/", "--")
        for root in hf_cache_roots:
            snapshots_dir = root / cache_name / "snapshots"
            if snapshots_dir.is_dir():
                candidates.extend(
                    sorted(snapshots_dir.glob("*/config.json"), key=lambda path: path.stat().st_mtime, reverse=True)
                )
    return [path for path in candidates if path.is_file()]


def _find_int(sections: list[dict[str, Any]], keys: tuple[str, ...]) -> int | None:
    for section in sections:
        for key in keys:
            if key in section and section[key] is not None:
                return int(section[key])
    return None


def _find_bool(sections: list[dict[str, Any]], keys: tuple[str, ...]) -> bool | None:
    for section in sections:
        for key in keys:
            if key in section and section[key] is not None:
                return bool(section[key])
    return None


def _read_metadata_file(config_path: Path, model_ref: str | None) -> ModelMetadata:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = data.get("text_config") if isinstance(data.get("text_config"), dict) else {}
    sections = [text_config, data]
    return ModelMetadata(
        model_path=model_ref,
        config_path=str(config_path),
        source="hf_config",
        num_experts=_find_int(sections, ("num_experts", "n_routed_experts")),
        top_k=_find_int(sections, ("num_experts_per_tok", "moe_top_k", "top_k")),
        num_hidden_layers=_find_int(sections, ("num_hidden_layers",)),
        hidden_size=_find_int(sections, ("hidden_size",)),
        intermediate_size=_find_int(sections, ("intermediate_size",)),
        moe_intermediate_size=_find_int(sections, ("moe_intermediate_size",)),
        shared_expert_intermediate_size=_find_int(sections, ("shared_expert_intermediate_size",)),
        num_attention_heads=_find_int(sections, ("num_attention_heads",)),
        num_key_value_heads=_find_int(sections, ("num_key_value_heads",)),
        head_dim=_find_int(sections, ("head_dim",)),
        vocab_size=_find_int(sections, ("vocab_size",)),
        tie_word_embeddings=_find_bool(sections, ("tie_word_embeddings",)),
        full_attention_interval=_find_int(sections, ("full_attention_interval",)),
        attn_output_gate=_find_bool(sections, ("attn_output_gate",)),
        linear_num_key_heads=_find_int(sections, ("linear_num_key_heads",)),
        linear_num_value_heads=_find_int(sections, ("linear_num_value_heads",)),
        linear_key_head_dim=_find_int(sections, ("linear_key_head_dim",)),
        linear_value_head_dim=_find_int(sections, ("linear_value_head_dim",)),
        linear_conv_kernel_dim=_find_int(sections, ("linear_conv_kernel_dim",)),
    )


def _known_metadata(model_ref: str) -> ModelMetadata | None:
    values = KNOWN_MODEL_METADATA.get(model_ref)
    if values is None:
        lowered = model_ref.lower()
        for known_ref in sorted(KNOWN_MODEL_METADATA, key=len, reverse=True):
            known_lowered = known_ref.lower()
            known_name = known_ref.rsplit("/", 1)[-1].lower()
            cache_name = known_ref.replace("/", "--").lower()
            if known_lowered in lowered or known_name in lowered or cache_name in lowered:
                values = KNOWN_MODEL_METADATA[known_ref]
                model_ref = known_ref
                break
    if values is None:
        return None
    return ModelMetadata(
        model_path=model_ref,
        config_path=None,
        source="known_model",
        num_experts=values.get("num_experts"),
        top_k=values.get("top_k"),
        num_hidden_layers=values.get("num_hidden_layers"),
        hidden_size=values.get("hidden_size"),
        intermediate_size=values.get("intermediate_size"),
        moe_intermediate_size=values.get("moe_intermediate_size"),
        shared_expert_intermediate_size=values.get("shared_expert_intermediate_size"),
        num_attention_heads=values.get("num_attention_heads"),
        num_key_value_heads=values.get("num_key_value_heads"),
        head_dim=values.get("head_dim"),
        vocab_size=values.get("vocab_size"),
        tie_word_embeddings=values.get("tie_word_embeddings"),
        full_attention_interval=values.get("full_attention_interval"),
        attn_output_gate=values.get("attn_output_gate"),
        linear_num_key_heads=values.get("linear_num_key_heads"),
        linear_num_value_heads=values.get("linear_num_value_heads"),
        linear_key_head_dim=values.get("linear_key_head_dim"),
        linear_value_head_dim=values.get("linear_value_head_dim"),
        linear_conv_kernel_dim=values.get("linear_conv_kernel_dim"),
    )


def resolve_model_metadata(
    raw_config: dict[str, Any],
    *,
    num_experts: int | None = None,
    top_k: int | None = None,
    hf_cache_roots: list[Path] | None = None,
) -> ModelMetadata:
    model = _section(raw_config, "model")
    model_config = _section(raw_config, "model_config")
    nested_model_config = model.get("config", {}) if isinstance(model.get("config"), dict) else {}
    config_sections = [model, nested_model_config, model_config]
    model_ref = model_ref_from_config(raw_config)

    config_metadata = ModelMetadata(
        model_path=model_ref,
        config_path=None,
        source="config",
        num_experts=_find_int(config_sections, ("num_experts", "n_routed_experts")),
        top_k=_find_int(config_sections, ("num_experts_per_tok", "moe_top_k", "top_k")),
        num_hidden_layers=_find_int(config_sections, ("num_hidden_layers",)),
        hidden_size=_find_int(config_sections, ("hidden_size",)),
        intermediate_size=_find_int(config_sections, ("intermediate_size",)),
        moe_intermediate_size=_find_int(config_sections, ("moe_intermediate_size",)),
        shared_expert_intermediate_size=_find_int(config_sections, ("shared_expert_intermediate_size",)),
        num_attention_heads=_find_int(config_sections, ("num_attention_heads",)),
        num_key_value_heads=_find_int(config_sections, ("num_key_value_heads",)),
        head_dim=_find_int(config_sections, ("head_dim",)),
        vocab_size=_find_int(config_sections, ("vocab_size",)),
        tie_word_embeddings=_find_bool(config_sections, ("tie_word_embeddings",)),
        full_attention_interval=_find_int(config_sections, ("full_attention_interval",)),
        attn_output_gate=_find_bool(config_sections, ("attn_output_gate",)),
        linear_num_key_heads=_find_int(config_sections, ("linear_num_key_heads",)),
        linear_num_value_heads=_find_int(config_sections, ("linear_num_value_heads",)),
        linear_key_head_dim=_find_int(config_sections, ("linear_key_head_dim",)),
        linear_value_head_dim=_find_int(config_sections, ("linear_value_head_dim",)),
        linear_conv_kernel_dim=_find_int(config_sections, ("linear_conv_kernel_dim",)),
    )

    file_metadata = None
    if model_ref:
        roots = hf_cache_roots if hf_cache_roots is not None else default_hf_cache_roots()
        candidate_paths = _candidate_config_paths(model_ref, roots)
        if candidate_paths:
            file_metadata = _read_metadata_file(candidate_paths[0], model_ref)

    known_metadata = _known_metadata(model_ref) if model_ref else None
    source_metadata = file_metadata or known_metadata or config_metadata

    resolved_num_experts = num_experts if num_experts is not None else config_metadata.num_experts
    if resolved_num_experts is None:
        resolved_num_experts = source_metadata.num_experts

    resolved_top_k = top_k if top_k is not None else config_metadata.top_k
    if resolved_top_k is None:
        resolved_top_k = source_metadata.top_k

    source = source_metadata.source
    if num_experts is not None or top_k is not None:
        source = f"{source}+explicit_override"

    return ModelMetadata(
        model_path=model_ref,
        config_path=source_metadata.config_path,
        source=source,
        num_experts=resolved_num_experts,
        top_k=resolved_top_k,
        num_hidden_layers=source_metadata.num_hidden_layers,
        hidden_size=source_metadata.hidden_size,
        intermediate_size=source_metadata.intermediate_size,
        moe_intermediate_size=source_metadata.moe_intermediate_size,
        shared_expert_intermediate_size=source_metadata.shared_expert_intermediate_size,
        num_attention_heads=source_metadata.num_attention_heads,
        num_key_value_heads=source_metadata.num_key_value_heads,
        head_dim=source_metadata.head_dim,
        vocab_size=source_metadata.vocab_size,
        tie_word_embeddings=source_metadata.tie_word_embeddings,
        full_attention_interval=(
            config_metadata.full_attention_interval
            if config_metadata.full_attention_interval is not None
            else source_metadata.full_attention_interval
        ),
        attn_output_gate=(
            config_metadata.attn_output_gate
            if config_metadata.attn_output_gate is not None
            else source_metadata.attn_output_gate
        ),
        linear_num_key_heads=(
            config_metadata.linear_num_key_heads
            if config_metadata.linear_num_key_heads is not None
            else source_metadata.linear_num_key_heads
        ),
        linear_num_value_heads=(
            config_metadata.linear_num_value_heads
            if config_metadata.linear_num_value_heads is not None
            else source_metadata.linear_num_value_heads
        ),
        linear_key_head_dim=(
            config_metadata.linear_key_head_dim
            if config_metadata.linear_key_head_dim is not None
            else source_metadata.linear_key_head_dim
        ),
        linear_value_head_dim=(
            config_metadata.linear_value_head_dim
            if config_metadata.linear_value_head_dim is not None
            else source_metadata.linear_value_head_dim
        ),
        linear_conv_kernel_dim=(
            config_metadata.linear_conv_kernel_dim
            if config_metadata.linear_conv_kernel_dim is not None
            else source_metadata.linear_conv_kernel_dim
        ),
    )
