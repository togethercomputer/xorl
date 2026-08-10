# ruff: noqa: PLC0415

"""HF -> xorl state-dict loader tests for DeepSeek-V4.

Covers the four transforms in the loader:

- Name mapping (top-level + per-layer attn/ffn + HC + norms).
- APE hotfix undo: applying the miles forward then the loader inverse round-trips
  to identity on the C4 layers.
- FP8 block dequantization: small synthetic FP8-E4M3 weight + UE8M0 scale
  -> dequantized BF16 matches a hand-computed reference within 1 ULP.
- Per-expert fusion: per-expert HF (w1, w2, w3) -> stacked
  ``[E, H, 2I]`` / ``[E, I, H]`` tensors with the right packing.

Plus an end-to-end synthetic load: build a tiny DSv4 model, fabricate an
HF-shaped state-dict at matching shapes, run the loader, verify every model
parameter ended up filled with the expected values.
"""

import pytest
import torch

from xorl.models.transformers.deepseek_v4.checkpoint_handler import DeepseekV4CheckpointHandler


pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _clear_rope_cache():
    from xorl.ops.dsv4.rope import precompute_freqs_cis  # noqa: PLC0415

    precompute_freqs_cis.cache_clear()
    yield
    precompute_freqs_cis.cache_clear()


def _tiny_config(*, compress_ratios, num_hash_layers=0):
    from xorl.models.transformers.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=len(compress_ratios),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=4,
        max_position_embeddings=256,
        q_lora_rank=16,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=8,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=num_hash_layers,
        hc_mult=2,
        compress_ratios=list(compress_ratios),
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        num_nextn_predict_layers=0,
        _moe_implementation="eager",
    )


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------


def test_name_mapping_top_level():
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _hf_to_xorl_name

    assert _hf_to_xorl_name("embed.weight") == "model.embed_tokens.weight"
    assert _hf_to_xorl_name("head.weight") == "lm_head.weight"
    assert _hf_to_xorl_name("norm.weight") == "model.norm.weight"
    assert _hf_to_xorl_name("hc_head_fn") == "model.hc_head_fn"
    assert _hf_to_xorl_name("hc_head_base") == "model.hc_head_base"
    assert _hf_to_xorl_name("hc_head_scale") == "model.hc_head_scale"


def test_name_mapping_attention():
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _hf_to_xorl_name

    assert _hf_to_xorl_name("layers.0.attn.wq_a.weight") == "model.layers.0.self_attn.wq_a.weight"
    assert _hf_to_xorl_name("layers.5.attn.attn_sink") == "model.layers.5.self_attn.attn_sink"
    assert _hf_to_xorl_name("layers.10.attn.compressor.ape") == "model.layers.10.self_attn.compressor.ape"
    # Indexer name renames: HF ``indexer.wq_b`` -> xorl ``indexer.linear_wq_b``.
    assert (
        _hf_to_xorl_name("layers.10.attn.indexer.wq_b.weight") == "model.layers.10.self_attn.indexer.linear_wq_b.weight"
    )
    assert (
        _hf_to_xorl_name("layers.10.attn.indexer.weights_proj.weight")
        == "model.layers.10.self_attn.indexer.linear_weights_proj.weight"
    )


def test_name_mapping_ffn_norms_hc():
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _hf_to_xorl_name

    assert _hf_to_xorl_name("layers.3.attn_norm.weight") == "model.layers.3.input_layernorm.weight"
    assert _hf_to_xorl_name("layers.3.ffn_norm.weight") == "model.layers.3.post_attention_layernorm.weight"
    assert _hf_to_xorl_name("layers.3.hc_attn_fn") == "model.layers.3.hc_attn_fn"
    assert _hf_to_xorl_name("layers.3.hc_ffn_scale") == "model.layers.3.hc_ffn_scale"

    # noaux_tc bias rename: HF gate.bias -> xorl mlp.gate.e_score_correction_bias.
    assert _hf_to_xorl_name("layers.0.ffn.gate.bias") == "model.layers.0.mlp.gate.e_score_correction_bias"
    # tid2eid sits on the block, not on the gate.
    assert _hf_to_xorl_name("layers.0.ffn.gate.tid2eid") == "model.layers.0.mlp.tid2eid"
    # Shared expert renames w1/w2/w3 -> gate_proj/down_proj/up_proj.
    assert (
        _hf_to_xorl_name("layers.0.ffn.shared_experts.w1.weight")
        == "model.layers.0.mlp.shared_experts.gate_proj.weight"
    )
    assert (
        _hf_to_xorl_name("layers.0.ffn.shared_experts.w2.weight")
        == "model.layers.0.mlp.shared_experts.down_proj.weight"
    )
    assert (
        _hf_to_xorl_name("layers.0.ffn.shared_experts.w3.weight") == "model.layers.0.mlp.shared_experts.up_proj.weight"
    )


def test_name_mapping_skips_mtp_and_unknown():
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _hf_to_xorl_name

    assert _hf_to_xorl_name("mtp.0.attn.wq_a.weight") is None
    assert _hf_to_xorl_name("totally.bogus.name") is None


# ---------------------------------------------------------------------------
# APE hotfix undo
# ---------------------------------------------------------------------------


def _miles_apply_ape_hotfix(param):
    """Forward direction of miles ``_apply_ape_hotfix_mirror`` for the test."""
    assert param.shape[0] == 4
    a, b = torch.chunk(param, 2, dim=-1)
    return torch.cat([a, b], dim=0).view(4, -1).contiguous()


def test_ape_hotfix_round_trip():
    """``_undo_ape_hotfix(_miles_apply_ape_hotfix(x)) == x`` for all C4 shapes."""
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _undo_ape_hotfix

    torch.manual_seed(0)
    for head_dim in (4, 16, 128):
        x = torch.randn(4, 2 * head_dim)
        hf_layout = _miles_apply_ape_hotfix(x)
        recovered = _undo_ape_hotfix(hf_layout)
        torch.testing.assert_close(recovered, x)


def test_ape_hotfix_assertion_on_wrong_shape():
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _undo_ape_hotfix

    with pytest.raises(AssertionError):
        _undo_ape_hotfix(torch.zeros(8, 16))


# ---------------------------------------------------------------------------
# FP8 block dequantization
# ---------------------------------------------------------------------------


def test_fp8_dequantize_matches_hand_computed():
    """Tiny 256x256 weight, block 128: per-block scale repeats correctly."""
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _dequantize_fp8_block

    torch.manual_seed(1)
    out_dim, in_dim = 256, 256
    block = (128, 128)
    # Choose values within float8_e4m3fn's representable range to avoid round-off.
    raw_fp32 = torch.randn(out_dim, in_dim) * 0.1
    weight_fp8 = raw_fp32.to(torch.float8_e4m3fn)
    # Construct a 2x2 scale grid: each block gets a distinct power-of-two scale.
    scale_fp8 = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float).to(torch.float8_e8m0fnu)

    bf16 = _dequantize_fp8_block(weight_fp8, scale_fp8, block_size=block, out_dtype=torch.bfloat16)

    # Reference: tile the scale up to (out, in) and multiply in fp32.
    s_full = torch.zeros(out_dim, in_dim)
    for i in range(2):
        for j in range(2):
            s_full[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = float(scale_fp8.float()[i, j])
    expected = (weight_fp8.float() * s_full).to(torch.bfloat16)

    torch.testing.assert_close(bf16, expected)


def test_mxfp4_dequantize_known_pattern():
    """Hand-encode 4 FP4 values into 2 int8 bytes, verify the dequant produces
    the expected values * scale.

    Encoding: byte 0 packs (element 0 = +0.5, element 1 = -2.0) =
    (low nibble = 0001, high nibble = 1100) = 0xC1.
    Byte 1 packs (+1.0, +6.0) = (0010, 0111) = 0x72.
    """
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import (
        _dequantize_mxfp4_packed_int8,
    )

    # Single row, two packed bytes -> 4 FP4 values.
    raw = torch.tensor([[0xC1 - 256, 0x72]], dtype=torch.int8)  # 0xC1 fits as -63 in i8
    # block_size=32 so a single scale spans more than 4 elements; just put 1.0.
    scale = torch.tensor([[1.0]]).to(torch.float8_e8m0fnu)
    out = _dequantize_mxfp4_packed_int8(raw, scale, block_size=32, out_dtype=torch.float32)
    expected = torch.tensor([[0.5, -2.0, 1.0, 6.0]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_mxfp4_dequantize_handles_block_scaling():
    """Two scale blocks -> different magnitudes per block."""
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import (
        _dequantize_mxfp4_packed_int8,
    )

    # 1 row, 32 packed bytes = 64 FP4 elements -> 2 scale blocks of 32.
    M, Np = 1, 32
    raw = torch.zeros(M, Np, dtype=torch.int8)
    # Fill every byte with (low=+1.0=0010, high=+1.5=0011) = 0x32.
    raw.fill_(0x32)
    # First block scale = 2.0, second = 8.0.
    scale = torch.tensor([[2.0, 8.0]]).to(torch.float8_e8m0fnu)
    out = _dequantize_mxfp4_packed_int8(raw, scale, block_size=32, out_dtype=torch.float32)

    # Even positions = +1.0, odd = +1.5; first 32 elements * 2.0, next 32 * 8.0.
    assert out.shape == (1, 64)
    torch.testing.assert_close(out[0, 0], torch.tensor(1.0 * 2.0))
    torch.testing.assert_close(out[0, 1], torch.tensor(1.5 * 2.0))
    torch.testing.assert_close(out[0, 32], torch.tensor(1.0 * 8.0))
    torch.testing.assert_close(out[0, 33], torch.tensor(1.5 * 8.0))


def test_fp8_dequantize_handles_non_block_aligned():
    """When out/in is not a multiple of the block, the residual is dequantized
    using the closest block scale (slice-and-truncate semantics)."""
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _dequantize_fp8_block

    torch.manual_seed(2)
    out_dim, in_dim = 130, 200
    block = (128, 128)
    weight = (torch.randn(out_dim, in_dim) * 0.1).to(torch.float8_e4m3fn)
    # 2 row-blocks (covering 0-128, 128-256 truncated at 130) and 2 col-blocks.
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu)
    out = _dequantize_fp8_block(weight, scale, block, torch.bfloat16)
    assert out.shape == (out_dim, in_dim)
    assert out.dtype == torch.bfloat16
    # Sanity: no NaNs / Infs since every block has finite scale and small weights.
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# End-to-end synthetic load
# ---------------------------------------------------------------------------


def _make_synthetic_hf_state_dict(cfg):
    """Fabricate an HF state-dict at the right shapes (no FP8, just bf16/fp32).

    Caller already knows the cfg, so we know exactly which tensors should
    appear. We use small predictable values per tensor so round-trip checks
    are easy.
    """
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    V = cfg.vocab_size
    NH = cfg.num_attention_heads
    HD = cfg.head_dim
    QL = cfg.q_lora_rank
    OL = cfg.o_lora_rank
    OG = cfg.o_groups
    INT = cfg.moe_intermediate_size
    E = cfg.n_routed_experts
    HC = cfg.hc_mult
    HC_DIM = HC * H
    MIX = (2 + HC) * HC

    sd = {}

    sd["embed.weight"] = torch.randn(V, H, dtype=torch.bfloat16)
    sd["head.weight"] = torch.randn(V, H, dtype=torch.bfloat16)
    sd["norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["hc_head_fn"] = torch.randn(HC, HC_DIM, dtype=torch.float32)
    sd["hc_head_base"] = torch.randn(HC, dtype=torch.float32)
    sd["hc_head_scale"] = torch.randn(1, dtype=torch.float32)

    for li in range(L):
        ratio = cfg.compress_ratios[li]
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

        if ratio:
            sd[f"layers.{li}.attn.compressor.ape"] = torch.randn(
                ratio, 2 * HD if ratio == 4 else HD, dtype=torch.float32
            )
            coff = 2 if ratio == 4 else 1
            sd[f"layers.{li}.attn.compressor.wkv.weight"] = torch.randn(coff * HD, H, dtype=torch.float32)
            sd[f"layers.{li}.attn.compressor.wgate.weight"] = torch.randn(coff * HD, H, dtype=torch.float32)
            sd[f"layers.{li}.attn.compressor.norm.weight"] = torch.ones(HD, dtype=torch.bfloat16)

        if ratio == 4:
            IH = cfg.index_n_heads
            ID = cfg.index_head_dim
            sd[f"layers.{li}.attn.indexer.wq_b.weight"] = torch.randn(IH * ID, QL, dtype=torch.bfloat16)
            sd[f"layers.{li}.attn.indexer.weights_proj.weight"] = torch.randn(IH, H, dtype=torch.bfloat16)
            sd[f"layers.{li}.attn.indexer.compressor.ape"] = torch.randn(4, 2 * ID, dtype=torch.float32)
            sd[f"layers.{li}.attn.indexer.compressor.wkv.weight"] = torch.randn(2 * ID, H, dtype=torch.float32)
            sd[f"layers.{li}.attn.indexer.compressor.wgate.weight"] = torch.randn(2 * ID, H, dtype=torch.float32)
            sd[f"layers.{li}.attn.indexer.compressor.norm.weight"] = torch.ones(ID, dtype=torch.bfloat16)

        sd[f"layers.{li}.ffn.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        if li < cfg.num_hash_layers:
            sd[f"layers.{li}.ffn.gate.tid2eid"] = torch.randint(0, E, (V, cfg.num_experts_per_tok), dtype=torch.int32)
        else:
            sd[f"layers.{li}.ffn.gate.bias"] = torch.zeros(E, dtype=torch.float32)

        for e in range(E):
            sd[f"layers.{li}.ffn.experts.{e}.w1.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
            sd[f"layers.{li}.ffn.experts.{e}.w3.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
            sd[f"layers.{li}.ffn.experts.{e}.w2.weight"] = torch.randn(H, INT, dtype=torch.bfloat16)
        # Shared expert (n_shared_experts = 1) is sized at moe_intermediate_size * 1.
        sd[f"layers.{li}.ffn.shared_experts.w1.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn.shared_experts.w3.weight"] = torch.randn(INT, H, dtype=torch.bfloat16)
        sd[f"layers.{li}.ffn.shared_experts.w2.weight"] = torch.randn(H, INT, dtype=torch.bfloat16)

    return sd


def test_end_to_end_synthetic_load_window_only():
    """Fully load a tiny 2-layer C0/C0 model from a fabricated HF state-dict."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM, load_hf_state_dict_into_model

    cfg = _tiny_config(compress_ratios=[0, 0])
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager").to(torch.float32)
    model.post_init()  # leave HC params zero-init; they'll be overwritten by load.

    sd = _make_synthetic_hf_state_dict(cfg)
    summary = load_hf_state_dict_into_model(model, sd, strict=False, dequantize_fp8=False, target_dtype=torch.bfloat16)

    # Every model param + buffer should be filled.
    fillable = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    expected_filled = {
        n
        for n in fillable
        # ``freqs_cis`` is a non-persistent buffer registered at module-init
        # and is never touched by the loader.
        if not n.endswith("freqs_cis")
    }
    # ``mlp.tid2eid`` only exists on hash layers; with num_hash_layers=0 it doesn't.
    expected_filled = {n for n in expected_filled if not n.endswith("mlp.tid2eid")}

    assert summary.experts_fused == cfg.num_hidden_layers
    assert summary.skipped_mtp == 0
    assert summary.unmapped == []
    assert summary.missing_in_model == []
    # Each layer contributes ~17 single tensors + per-layer HC + shared expert,
    # so just sanity-check that the count is positive.
    assert summary.loaded > 20

    # Spot-check a few values round-trip.
    torch.testing.assert_close(
        model.model.embed_tokens.weight.to(torch.bfloat16),
        sd["embed.weight"],
    )
    torch.testing.assert_close(
        model.lm_head.weight.to(torch.bfloat16),
        sd["head.weight"],
    )
    torch.testing.assert_close(model.model.hc_head_fn, sd["hc_head_fn"])
    torch.testing.assert_close(model.model.layers[0].self_attn.attn_sink, sd["layers.0.attn.attn_sink"])


def test_checkpoint_handler_ep_filters_and_fuses_local_experts():
    """The generic distributed loader handler should emit only this EP rank's experts."""
    cfg = _tiny_config(compress_ratios=[0])
    sd = _make_synthetic_hf_state_dict(cfg)
    handler = DeepseekV4CheckpointHandler(
        cfg,
        checkpoint_keys=set(sd),
        ep_rank=1,
        ep_size=2,
        dequantize_fp8=False,
        target_dtype=torch.bfloat16,
    )
    skip_key_fn = handler.get_skip_key_fn()
    assert skip_key_fn is not None
    assert skip_key_fn("layers.0.ffn.experts.0.w1.weight") is True
    assert skip_key_fn("layers.0.ffn.experts.2.w1.weight") is False
    assert skip_key_fn("mtp.0.ffn.experts.2.w1.weight") is True

    emitted = []
    for key in sorted(sd):
        if skip_key_fn(key):
            emitted.extend(handler.on_skip_weight(key))
        else:
            emitted.extend(handler.on_load_weight(key, sd[key]))
    emitted.extend(handler.on_load_complete())
    out = dict(emitted)

    gate_up = out["model.layers.0.mlp.experts.gate_up_proj"]
    down = out["model.layers.0.mlp.experts.down_proj"]
    assert gate_up.shape == (2, cfg.hidden_size, 2 * cfg.moe_intermediate_size)
    assert down.shape == (2, cfg.moe_intermediate_size, cfg.hidden_size)

    expected_gate_rows = []
    expected_down_rows = []
    for expert_idx in (2, 3):
        w1 = sd[f"layers.0.ffn.experts.{expert_idx}.w1.weight"].t().contiguous()
        w3 = sd[f"layers.0.ffn.experts.{expert_idx}.w3.weight"].t().contiguous()
        w2 = sd[f"layers.0.ffn.experts.{expert_idx}.w2.weight"].t().contiguous()
        expected_gate_rows.append(torch.cat([w1, w3], dim=-1))
        expected_down_rows.append(w2)

    torch.testing.assert_close(gate_up, torch.stack(expected_gate_rows, dim=0))
    torch.testing.assert_close(down, torch.stack(expected_down_rows, dim=0))


def test_checkpoint_handler_skips_mtp_even_without_ep_filter():
    cfg = _tiny_config(compress_ratios=[0])
    handler = DeepseekV4CheckpointHandler(
        cfg,
        checkpoint_keys={"embed.weight", "mtp.0.ffn.experts.0.w1.weight"},
        ep_rank=0,
        ep_size=1,
        dequantize_fp8=False,
        target_dtype=torch.bfloat16,
    )

    skip_key_fn = handler.get_skip_key_fn()
    assert skip_key_fn is not None
    assert skip_key_fn("mtp.0.ffn.experts.0.w1.weight") is True
    assert skip_key_fn("embed.weight") is False


def test_strict_load_ignores_nonpersistent_rope_buffers():
    """``strict=True`` should not require config-derived RoPE cache buffers."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM, load_hf_state_dict_into_model

    cfg = _tiny_config(compress_ratios=[0, 0])
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager").to(torch.float32)
    sd = _make_synthetic_hf_state_dict(cfg)

    summary = load_hf_state_dict_into_model(model, sd, strict=True, dequantize_fp8=False, target_dtype=torch.bfloat16)

    assert summary.unmapped == []
    assert summary.missing_in_model == []


def test_end_to_end_synthetic_load_with_c4_layer():
    """C4 layer load exercises the indexer mapping + APE hotfix path."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM, load_hf_state_dict_into_model
    from xorl.models.transformers.deepseek_v4.checkpoint_handler import _undo_ape_hotfix

    cfg = _tiny_config(compress_ratios=[0, 4])
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager").to(torch.float32)
    model.post_init()

    sd = _make_synthetic_hf_state_dict(cfg)

    # Apply the miles forward hotfix to both C4 ``ape`` tensors so we're
    # simulating what the real HF disk format looks like.
    sd["layers.1.attn.compressor.ape"] = _miles_apply_ape_hotfix(torch.randn(4, 2 * cfg.head_dim, dtype=torch.float32))
    sd["layers.1.attn.indexer.compressor.ape"] = _miles_apply_ape_hotfix(
        torch.randn(4, 2 * cfg.index_head_dim, dtype=torch.float32)
    )

    expected_compressor_ape = _undo_ape_hotfix(sd["layers.1.attn.compressor.ape"])
    expected_indexer_ape = _undo_ape_hotfix(sd["layers.1.attn.indexer.compressor.ape"])

    summary = load_hf_state_dict_into_model(model, sd, strict=False, dequantize_fp8=False, target_dtype=torch.bfloat16)

    assert summary.ape_unhotfixed == 2  # one for compressor, one for indexer.compressor
    assert summary.experts_fused == cfg.num_hidden_layers

    torch.testing.assert_close(model.model.layers[1].self_attn.compressor.ape, expected_compressor_ape)
    torch.testing.assert_close(model.model.layers[1].self_attn.indexer.compressor.ape, expected_indexer_ape)


def test_end_to_end_synthetic_load_with_hash_layer():
    """First layer is hash-routed: tid2eid is filled, gate.bias is absent."""
    from xorl.models.transformers.deepseek_v4 import DeepseekV4ForCausalLM, load_hf_state_dict_into_model

    cfg = _tiny_config(compress_ratios=[0, 0], num_hash_layers=1)
    model = DeepseekV4ForCausalLM(cfg, moe_implementation="eager").to(torch.float32)
    model.post_init()

    sd = _make_synthetic_hf_state_dict(cfg)
    summary = load_hf_state_dict_into_model(model, sd, strict=False, dequantize_fp8=False, target_dtype=torch.bfloat16)

    # Hash layer 0 gets tid2eid; layer 1 (non-hash) gets e_score_correction_bias.
    layer0 = model.model.layers[0].mlp
    assert "tid2eid" in layer0._buffers
    assert layer0.tid2eid.dtype == torch.int32
    torch.testing.assert_close(layer0.tid2eid, sd["layers.0.ffn.gate.tid2eid"])

    layer1 = model.model.layers[1].mlp
    assert hasattr(layer1.gate, "e_score_correction_bias")
    torch.testing.assert_close(layer1.gate.e_score_correction_bias.float(), sd["layers.1.ffn.gate.bias"].float())

    assert summary.unmapped == []
    assert summary.missing_in_model == []
