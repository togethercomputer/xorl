"""End-to-end test: HF safetensors snapshot -> DCP -> back into a fresh model.

Exercises ``scripts/convert_dsv4_hf_to_dcp.py`` on a fabricated tiny
snapshot (config.json + a single safetensors shard + index file) so we
catch wiring bugs in the script — DCP write path, single-rank pg init,
metadata generation — before running it on real 149 GB Flash data.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import safetensors.torch as st
import torch
import torch.distributed as dist


pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _clean_pg_and_cache():
    from xorl.ops.families.dsv4.rope import precompute_freqs_cis  # noqa: PLC0415

    precompute_freqs_cis.cache_clear()
    yield
    precompute_freqs_cis.cache_clear()
    if dist.is_initialized():
        dist.destroy_process_group()


def _tiny_hf_snapshot(snapshot_dir: Path) -> dict[str, torch.Tensor]:
    """Fabricate a tiny HF DSv4 snapshot. Returns the source state-dict so
    callers can compare round-trip values.
    """
    H, L = 32, 2
    NH, HD, QL = 2, 16, 16
    OG, OL = 1, 8
    INT, E = 16, 4
    HC = 2
    HC_DIM = HC * H
    MIX = (2 + HC) * HC
    V = 64

    sd: dict[str, torch.Tensor] = {}
    sd["embed.weight"] = torch.randn(V, H, dtype=torch.bfloat16)
    sd["head.weight"] = torch.randn(V, H, dtype=torch.bfloat16)
    sd["norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["hc_head_fn"] = torch.randn(HC, HC_DIM, dtype=torch.float32)
    sd["hc_head_base"] = torch.randn(HC, dtype=torch.float32)
    sd["hc_head_scale"] = torch.randn(1, dtype=torch.float32)

    for li in range(L):
        sd[f"layers.{li}.attn_norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn_norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        for prefix in ("hc_attn", "hc_ffn"):
            sd[f"layers.{li}.{prefix}_fn"] = torch.randn(MIX, HC_DIM, dtype=torch.float32)
            sd[f"layers.{li}.{prefix}_base"] = torch.randn(MIX, dtype=torch.float32)
            sd[f"layers.{li}.{prefix}_scale"] = torch.randn(3, dtype=torch.float32)
        sd[f"layers.{li}.attn.wq_a.weight"] = torch.randn(QL, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.q_norm.weight"] = torch.ones(QL, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.wq_b.weight"] = torch.randn(NH * HD, QL, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.wkv.weight"] = torch.randn(HD, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.kv_norm.weight"] = torch.ones(HD, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.wo_a.weight"] = torch.randn(OG * OL, NH * HD // OG, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.wo_b.weight"] = torch.randn(H, OG * OL, dtype=torch.bfloat16)
        sd[f"layers.{li}.attn.attn_sink"] = torch.randn(NH, dtype=torch.float32)
        sd[f"layers.{li}.ffn.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn.gate.bias"] = torch.zeros(E, dtype=torch.float32)
        for e in range(E):
            sd[f"layers.{li}.ffn.experts.{e}.w1.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
            sd[f"layers.{li}.ffn.experts.{e}.w3.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
            sd[f"layers.{li}.ffn.experts.{e}.w2.weight"] = torch.randn(H, INT, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn.shared_experts.w1.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn.shared_experts.w3.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn.shared_experts.w2.weight"] = torch.randn(H, INT, dtype=torch.bfloat16)

    # Single shard.
    shard_name = "model-00001-of-00001.safetensors"
    st.save_file(sd, str(snapshot_dir / shard_name))

    # Mock index.
    weight_map = dict.fromkeys(sd.keys(), shard_name)
    with (snapshot_dir / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)

    # Mock config.json matching the tiny shapes above.
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": V,
        "hidden_size": H,
        "num_hidden_layers": L,
        "num_attention_heads": NH,
        "num_key_value_heads": 1,
        "head_dim": HD,
        "qk_rope_head_dim": 4,
        "max_position_embeddings": 256,
        "q_lora_rank": QL,
        "o_groups": OG,
        "o_lora_rank": OL,
        "sliding_window": 8,
        "moe_intermediate_size": INT,
        "n_routed_experts": E,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 0,
        "hc_mult": HC,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "compress_ratios": [0, 0],
        "compress_rope_theta": 160000,
        "swiglu_limit": 0.0,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "num_nextn_predict_layers": 0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    with (snapshot_dir / "config.json").open("w") as f:
        json.dump(config, f)

    return sd


def _assert_converter_meta_dtype_cast_preserves_fp32_destinations():
    """The torchrun converter's meta-model cast must not downcast fp32-only params."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from scripts.convert_dsv4_hf_to_dcp import _hf_config_to_xorl, cast_dsv4_model_dtype  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    from xorl.models.module_utils import init_empty_weights  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_dir = Path(tmp) / "hf-snap"
        snapshot_dir.mkdir()
        _tiny_hf_snapshot(snapshot_dir)
        cfg = _hf_config_to_xorl(snapshot_dir)
        cfg.compress_ratios = [0, 128]

        with init_empty_weights():
            model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")
        cast_dsv4_model_dtype(model, torch.bfloat16)

    params = dict(model.named_parameters())
    assert params["model.embed_tokens.weight"].device.type == "meta"
    assert params["model.embed_tokens.weight"].dtype == torch.bfloat16

    fp32_param_names = [
        "model.hc_head_fn",
        "model.layers.0.self_attn.attn_sink",
        "model.layers.1.self_attn.compressor.ape",
        "model.layers.1.self_attn.compressor.wkv.weight",
        "model.layers.1.self_attn.compressor.wgate.weight",
    ]
    for name in fp32_param_names:
        param = params[name]
        assert param.device.type == "meta"
        assert param.dtype == torch.float32
        assert getattr(param, "_keep_fp32", False) is True


def test_convert_dsv4_hf_to_dcp_conversion_policy():
    """Run the conversion script's main(), then load the DCP and compare."""
    _assert_converter_meta_dtype_cast_preserves_fp32_destinations()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from scripts.convert_dsv4_hf_to_dcp import main as convert_main  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot_dir = tmp / "hf-snap"
        snapshot_dir.mkdir()
        dcp_dir = tmp / "dcp"

        original_sd = _tiny_hf_snapshot(snapshot_dir)

        # Use a unique master port to avoid colliding with the live trainer pod.
        os.environ["MASTER_PORT"] = "29711"
        rc = convert_main(
            [
                "--hf-snapshot",
                str(snapshot_dir),
                "--dcp-out",
                str(dcp_dir),
                "--target-dtype",
                "bfloat16",
            ]
        )
        assert rc == 0

        # Sanity: DCP files exist.
        assert (dcp_dir / ".metadata").exists() or any(dcp_dir.glob("*.distcp")), (
            f"no DCP files under {dcp_dir}: {list(dcp_dir.iterdir())}"
        )
        assert (dcp_dir / "checkpoint_metadata.json").exists()

        # Load DCP back into a fresh model and verify a key weight matches.
        from xorl.checkpoint.checkpointer import DistributedCheckpointer as Checkpointer  # noqa: PLC0415
        from xorl.models.transformers.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM  # noqa: PLC0415

        with (snapshot_dir / "config.json").open() as f:
            raw = json.load(f)

        class _Obj:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        cfg = DeepseekV4Config.from_hf_config(_Obj(raw))
        roundtrip = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")
        roundtrip.post_init()
        roundtrip.to(torch.bfloat16)
        for p in roundtrip.parameters():
            if getattr(p, "_keep_fp32", False):
                p.data = p.data.to(torch.float32)

        Checkpointer.load(str(dcp_dir), {"model": roundtrip}, strict=False)

        # ``embed.weight`` round-trips exactly (it's BF16 in the source +
        # written/read as BF16 through DCP).
        torch.testing.assert_close(
            roundtrip.model.embed_tokens.weight,
            original_sd["embed.weight"],
        )
        # ``hc_head_fn`` round-trips exactly (fp32 source + fp32 model param).
        torch.testing.assert_close(
            roundtrip.model.hc_head_fn,
            original_sd["hc_head_fn"],
        )
        # attn_sink round-trips exactly.
        torch.testing.assert_close(
            roundtrip.model.layers[0].self_attn.attn_sink,
            original_sd["layers.0.attn.attn_sink"],
        )

        # Older converted DCPs may not have xorl's JSON sidecar. In that case,
        # strict=False still needs to tolerate model-only keys such as injected
        # LoRA adapters during DCP planning, before set_model_state_dict runs.
        (dcp_dir / "checkpoint_metadata.json").unlink()

        from xorl.lora.utils import inject_lora_into_model  # noqa: PLC0415

        lora_roundtrip = DeepseekV4ForCausalLM(cfg, moe_implementation="eager")
        lora_roundtrip.post_init()
        inject_lora_into_model(lora_roundtrip, r=2, lora_alpha=4, target_modules=["wq_a"])
        lora_roundtrip.to(torch.bfloat16)
        for p in lora_roundtrip.parameters():
            if getattr(p, "_keep_fp32", False):
                p.data = p.data.to(torch.float32)

        lora_b_before = lora_roundtrip.model.layers[0].self_attn.wq_a.lora_B.detach().clone()
        Checkpointer.load(str(dcp_dir), {"model": lora_roundtrip}, strict=False)

        torch.testing.assert_close(
            lora_roundtrip.model.layers[0].self_attn.wq_a.weight,
            original_sd["layers.0.attn.wq_a.weight"],
        )
        torch.testing.assert_close(lora_roundtrip.model.layers[0].self_attn.wq_a.lora_B, lora_b_before)

    if dist.is_initialized():
        dist.destroy_process_group()
    _assert_convert_dsv4_hf_to_dcp_pair_across_shards()


def test_automodel_from_pretrained_loads_tiny_hf_snapshot():
    """AutoConfig, XoRL construction, and AutoModel loading share one snapshot contract."""
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    from xorl.models import build_foundation_model  # noqa: PLC0415
    from xorl.models.transformers.deepseek_v4 import (  # noqa: PLC0415
        DeepseekV4Config,
        DeepseekV4ForCausalLM,
    )

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_dir = Path(tmp) / "hf-snap"
        snapshot_dir.mkdir()
        original_sd = _tiny_hf_snapshot(snapshot_dir)

        config = AutoConfig.from_pretrained(snapshot_dir)
        assert isinstance(config, DeepseekV4Config)
        assert config.model_type == "deepseek_v4"
        assert config.num_hidden_layers == 2
        assert config.n_routed_experts == 4
        assert AutoModelForCausalLM._model_mapping.get(DeepseekV4Config, None) is DeepseekV4ForCausalLM

        meta_model = build_foundation_model(
            snapshot_dir,
            init_device="meta",
            moe_implementation="eager",
            attn_implementation="flash_attention_3",
        )
        assert isinstance(meta_model, DeepseekV4ForCausalLM)

        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot_dir),
            torch_dtype=torch.bfloat16,
            attn_implementation="native",
            moe_implementation="eager",
            init_device="cpu",
            progress=False,
        )

    assert isinstance(model, DeepseekV4ForCausalLM)
    torch.testing.assert_close(model.model.embed_tokens.weight, original_sd["embed.weight"])


def _split_snapshot_across_shards(
    snapshot_dir: Path,
    sd: dict[str, torch.Tensor],
    *,
    pair_keys: tuple[str, str] | None = None,
) -> None:
    """Re-shard ``sd`` across two safetensors files (overwriting whatever
    ``_tiny_hf_snapshot`` wrote). When ``pair_keys`` is set, the named
    ``.weight`` lands in shard 1 and the matching ``.scale`` in shard 2 —
    that's the "pair across shards" case the streaming loader must defer
    until both arrive.
    """
    # Remove the original single-shard write.
    for old in snapshot_dir.glob("model-*.safetensors"):
        old.unlink()

    # Split keys into two halves; if ``pair_keys`` is provided, force the
    # weight into shard 1 and the scale into shard 2 explicitly.
    keys = sorted(sd.keys())
    half = len(keys) // 2
    shard_a_keys = set(keys[:half])
    shard_b_keys = set(keys[half:])
    if pair_keys is not None:
        weight_k, scale_k = pair_keys
        shard_a_keys.add(weight_k)
        shard_a_keys.discard(scale_k)
        shard_b_keys.discard(weight_k)
        shard_b_keys.add(scale_k)

    shard_a = {k: sd[k] for k in shard_a_keys if k in sd}
    shard_b = {k: sd[k] for k in shard_b_keys if k in sd}

    fname_a = "model-00001-of-00002.safetensors"
    fname_b = "model-00002-of-00002.safetensors"
    st.save_file(shard_a, str(snapshot_dir / fname_a))
    st.save_file(shard_b, str(snapshot_dir / fname_b))

    weight_map = {**dict.fromkeys(shard_a, fname_a), **dict.fromkeys(shard_b, fname_b)}
    with (snapshot_dir / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def _assert_convert_dsv4_hf_to_dcp_pair_across_shards():
    """The streaming loader holds a weight in ``pending`` until its paired
    ``.scale`` arrives in a later shard. Split a synthetic FP8 expert
    weight across two shards (weight → shard 1, scale → shard 2) and
    verify the conversion still completes.

    Without correct cross-shard deferral, the weight would be processed
    against ``slot["scale"] is None`` → ``_dequantize_fp8_block`` would
    receive ``None`` and crash, or the weight would be silently used at
    the wrong dtype.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    try:
        from scripts.convert_dsv4_hf_to_dcp import main as convert_main  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snapshot_dir = tmp / "hf-snap"
        snapshot_dir.mkdir()
        dcp_dir = tmp / "dcp"

        sd = _tiny_hf_snapshot(snapshot_dir)

        # Replace one expert's w1 with a synthetic FP8 block-quantized pair.
        # Block size matches what ``_dequantize_fp8_block`` accepts (=128 by
        # default; with the tiny shapes here, the block scale is shape (1,1)).
        weight_k = "layers.0.ffn.experts.0.w1.weight"
        scale_k = "layers.0.ffn.experts.0.w1.scale"
        # Round-tripping FP8 isn't the point — we just need the loader to
        # treat ``.scale`` as a paired companion of ``.weight``. A trivial
        # uint8-packed FP8E4M3 with a fp32 scale exercises the same
        # ``pending[]`` flow as the real Flash/Pro snapshots.
        H = sd[weight_k].shape[1]
        I = sd[weight_k].shape[0]
        sd[weight_k] = torch.zeros(I, H, dtype=torch.float8_e4m3fn)
        # Block-fp8 scale: shape (ceil(I/128), ceil(H/128)) = (1, 1) for tiny shapes.
        sd[scale_k] = torch.ones(1, 1, dtype=torch.float32)

        _split_snapshot_across_shards(snapshot_dir, sd, pair_keys=(weight_k, scale_k))

        os.environ["MASTER_PORT"] = "29712"
        rc = convert_main(
            [
                "--hf-snapshot",
                str(snapshot_dir),
                "--dcp-out",
                str(dcp_dir),
                "--target-dtype",
                "bfloat16",
            ]
        )
        assert rc == 0, "Conversion must complete after cross-shard pair deferral"

        # Sanity: DCP files exist, no missing weights logged elsewhere.
        assert (dcp_dir / ".metadata").exists() or any(dcp_dir.glob("*.distcp")), (
            f"no DCP files under {dcp_dir}: {list(dcp_dir.iterdir())}"
        )
        assert (dcp_dir / "checkpoint_metadata.json").exists()
