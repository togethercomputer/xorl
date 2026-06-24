"""Config + registry + modeling coverage for GLM-5 / GLM-5.1.

Covers the dense-MLA path (default), the absorb-form sparse-MLA path
(opt-in via `_sparse_mla_enabled`), and the equivalence of the two when
the indexer's `index_topk` covers the whole sequence.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xorl.lora.modules import LoraLinear
from xorl.lora.utils import inject_lora_into_model
from xorl.models.auto import _load_local_xorl_config, _namespace_from_dict, build_foundation_model
from xorl.models.layers import RotaryEmbedding
from xorl.models.layers.moe import MoEBlock, MoEExpertsLoRA
from xorl.models.layers.moe.routing_replay import set_replay_stage
from xorl.models.layers.normalization import set_rmsnorm_mode
from xorl.models.loader import get_loader
from xorl.models.registry import get_registry
from xorl.models.transformers.glm5 import (
    Glm5Attention,
    Glm5Config,
    Glm5DsaIndexer,
    Glm5ForCausalLM,
    Glm5MLP,
    Glm5Model,
    Glm5MoEBlock,
    GlmMoeDsaForCausalLM,
)
from xorl.models.transformers.glm5.checkpoint_handler import Glm5CheckpointHandler
from xorl.models.transformers.glm5.sparse_mla import (
    _sparse_mla_tilelang,
    sparse_mla_dispatch,
    sparse_mla_torch_reference,
)


pytestmark = [pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _reset_routing_replay_stage():
    """The routing-replay stage is a module-level global; tests in other
    files can leave it set to ``"record"``/``"replay_forward"``, which then
    corrupts MoE forward paths in our tests (empty replay queue → NaN).
    Force-clear it around every test."""
    set_replay_stage(None)
    yield
    set_replay_stage(None)


GLM_5_1_HF_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "vocab_size": 154880,
    "hidden_size": 6144,
    "intermediate_size": 12288,
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 202752,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000},
    "rope_interleave": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "first_k_dense_replace": 3,
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "indexer_rope_interleave": True,
    "num_nextn_predict_layers": 1,
    "pad_token_id": 154820,
    "eos_token_id": [154820, 154827, 154829],
    "ep_size": 1,
    "pretraining_tp": 1,
    "tie_word_embeddings": False,
    "initializer_range": 0.02,
    "use_cache": True,
    "dtype": "bfloat16",
}


def _tiny_config(**overrides) -> Glm5Config:
    """Tiny GLM-5-shaped config for CPU forward smoke tests."""
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        max_position_embeddings=64,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=4,
        num_nextn_predict_layers=0,
        pad_token_id=0,
    )
    base.update(overrides)
    config = Glm5Config(**base)
    # CPU-only forward: route around the Triton/CUDA fast paths.
    config._attn_implementation = "eager"
    config._activation_native = True
    config._moe_implementation = "eager"
    config._rmsnorm_mode = "eager"
    set_rmsnorm_mode("eager")
    return config


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


def test_glm5_config_is_standalone():
    config = Glm5Config()
    assert isinstance(config, Glm5Config)
    assert config.model_type == "xorl_glm5"


def test_from_hf_config_captures_mla_moe_dsa_and_mtp_fields():
    hf_config = _namespace_from_dict(GLM_5_1_HF_CONFIG)

    config = Glm5Config.from_hf_config(hf_config)

    # MLA dims
    assert config.q_lora_rank == 2048
    assert config.kv_lora_rank == 512
    assert config.qk_head_dim == 256  # derived: nope (192) + rope (64)
    assert config.qk_nope_head_dim == 192
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 256

    # Routed-expert MoE
    assert config.n_routed_experts == 256
    assert config.n_shared_experts == 1
    assert config.num_experts_per_tok == 8
    assert config.first_k_dense_replace == 3
    assert config.moe_intermediate_size == 2048
    assert config.scoring_func == "sigmoid"
    assert config.topk_method == "noaux_tc"
    assert config.routed_scaling_factor == pytest.approx(2.5)
    assert config.norm_topk_prob is True

    # DeepSeek Sparse Attention indexer
    assert config.index_head_dim == 128
    assert config.index_n_heads == 32
    assert config.index_topk == 2048
    assert config.indexer_rope_interleave is True

    # Multi-Token Prediction
    assert config.num_nextn_predict_layers == 1

    # RoPE
    assert config.rope_theta == 1_000_000
    assert config.rope_interleave is True

    # Architectures + model_type rename
    assert config.architectures == ["GlmMoeDsaForCausalLM"]
    assert config.model_type == "xorl_glm5"


def test_auto_load_local_xorl_config_routes_glm_moe_dsa(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(GLM_5_1_HF_CONFIG))

    config = _load_local_xorl_config(str(tmp_path), config_kwargs={})

    assert isinstance(config, Glm5Config)
    assert config.architectures == ["GlmMoeDsaForCausalLM"]
    assert config.n_routed_experts == 256


def test_glm5_default_construction_matches_glm_5_1_shape():
    """Defaults track the public `zai-org/GLM-5.1` config so a bare `Glm5Config()`
    is always interpretable without an HF round-trip."""
    config = Glm5Config()

    assert config.hidden_size == 6144
    assert config.num_hidden_layers == 78
    assert config.num_attention_heads == 64
    assert config.vocab_size == 154880
    assert config.q_lora_rank == 2048
    assert config.n_routed_experts == 256
    assert config.num_experts_per_tok == 8


# --------------------------------------------------------------------------- #
# Registry / loader
# --------------------------------------------------------------------------- #


def test_glm5_registered_under_both_arch_names():
    """Registry resolves both `GlmMoeDsaForCausalLM` (HF) and `Glm5ForCausalLM`."""
    reg = get_registry()
    assert "GlmMoeDsaForCausalLM" in reg.supported_models
    assert "Glm5ForCausalLM" in reg.supported_models
    # Subclass relationship: `GlmMoeDsaForCausalLM` is just an alias.
    assert issubclass(GlmMoeDsaForCausalLM, Glm5ForCausalLM)


def test_get_loader_resolves_glm_moe_dsa():
    config = Glm5Config.from_hf_config(_namespace_from_dict(GLM_5_1_HF_CONFIG))
    loader = get_loader(config)
    assert "GlmMoeDsaForCausalLM" in loader.description


def test_build_foundation_model_rejects_glm5_dsa_with_ring_attention(monkeypatch):
    class ParallelState:
        ringattn_size = 1
        ringattn_enabled = True
        cp_enabled = True

    config = _tiny_config()
    config.architectures = ["Glm5ForCausalLM"]
    monkeypatch.setattr("xorl.models.auto.get_parallel_state", lambda: ParallelState())

    with pytest.raises(ValueError, match="ring attention"):
        build_foundation_model(config, attn_implementation="eager", init_device="meta")


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #


def test_indexer_has_four_glm_specific_projections():
    config = _tiny_config()
    indexer = Glm5DsaIndexer(config)

    assert indexer.wq_b.in_features == config.q_lora_rank
    assert indexer.wq_b.out_features == config.index_n_heads * config.index_head_dim
    assert indexer.wk.in_features == config.hidden_size
    assert indexer.wk.out_features == config.index_head_dim
    assert indexer.k_norm.weight.shape == (config.index_head_dim,)
    assert indexer.weights_proj.in_features == config.hidden_size
    assert indexer.weights_proj.out_features == config.index_n_heads


def test_attention_carries_indexer_module():
    """V0: indexer params exist on attention so checkpoints load cleanly."""
    config = _tiny_config()
    attn = Glm5Attention(config, layer_idx=0)
    assert isinstance(attn.indexer, Glm5DsaIndexer)


def test_indexer_select_topk_returns_correct_shape():
    """The dense fallback returns top-k indices for each query position.

    Causality is enforced by masking non-causal logits with -inf. When fewer
    than `index_topk` valid keys exist for a query (early positions), `topk`
    fills the remaining slots with arbitrary indices over -inf entries — so
    the strong causality check only holds for the last query position, where
    every key is valid.
    """
    config = _tiny_config()
    indexer = Glm5DsaIndexer(config)

    B, S = 2, 8
    index_q = torch.randn(B, S, config.index_n_heads, config.index_head_dim)
    index_k = torch.randn(B, S, config.index_head_dim)
    head_weights = torch.randn(B, S, config.index_n_heads)

    indices = indexer.select_topk(index_q, index_k, head_weights, attention_mask=None)

    assert indices.shape == (B, S, config.index_topk)
    assert indices.dtype == torch.int32
    # The indexer marks non-causal pad slots with -1 (kernel sentinel);
    # all other slots are in [0, S).
    valid = indices >= 0
    assert torch.all(indices[valid] < S)
    # For the last query position (q=S-1), all S keys are causal — top-k
    # picks must all be valid (no -1 padding).
    last_q_indices = indices[:, -1, :]
    assert torch.all(last_q_indices >= 0)
    assert torch.all(last_q_indices < S)


def test_indexer_select_topk_respects_additive_attention_mask():
    """Masked positions should become `-1` sentinels before sparse-MLA.

    Eager causal masks use `torch.finfo(dtype).min`, not literal `-inf`.
    If the indexer treats those as finite scores, top-k can waste sparse
    budget on padding / packed-sequence boundaries and the tilelang path
    receives non-sentinel invalid indices.
    """
    config = _tiny_config(index_topk=3)
    indexer = Glm5DsaIndexer(config)

    B, S = 1, 5
    index_q = torch.randn(B, S, config.index_n_heads, config.index_head_dim)
    index_k = torch.randn(B, S, config.index_head_dim)
    head_weights = torch.randn(B, S, config.index_n_heads)
    attention_mask = torch.full((B, S, S), torch.finfo(index_q.dtype).min)
    rows = torch.arange(S)
    attention_mask[:, rows, rows] = 0

    indices = indexer.select_topk(index_q, index_k, head_weights, attention_mask=attention_mask)

    torch.testing.assert_close(indices[0, :, 0], torch.arange(S, dtype=torch.int32))
    assert torch.all(indices[0, :, 1:] == -1)


def test_indexer_select_topk_supports_chunked_head_scoring():
    """Chunked scoring must preserve masking semantics used by sparse-MLA."""
    config = _tiny_config(index_topk=3, index_n_heads=4)
    config.indexer_score_chunk_heads = 1
    indexer = Glm5DsaIndexer(config)

    B, S = 1, 5
    index_q = torch.randn(B, S, config.index_n_heads, config.index_head_dim)
    index_k = torch.randn(B, S, config.index_head_dim)
    head_weights = torch.randn(B, S, config.index_n_heads)
    attention_mask = torch.full((B, S, S), torch.finfo(index_q.dtype).min)
    rows = torch.arange(S)
    attention_mask[:, rows, rows] = 0

    indices = indexer.select_topk(index_q, index_k, head_weights, attention_mask=attention_mask)

    torch.testing.assert_close(indices[0, :, 0], torch.arange(S, dtype=torch.int32))
    assert torch.all(indices[0, :, 1:] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_padding_mask_detection_triggers_fast_path():
    """A 2D padding mask (e.g., HuggingFace-style [B, S] with 1s for valid
    tokens and 0s for padded tail) should be detected and routed to the
    tilelang fast path with cu_ke clipped to the valid length."""
    try:
        from xorl.ops.glm5_kernels.tilelang_indexer_fwd import tl_indexer_fwd_impl  # noqa: F401
    except Exception:
        pytest.skip("tilelang indexer fwd kernel unavailable")
    torch.manual_seed(0)
    config = _tiny_config(index_topk=8, index_n_heads=4)
    indexer = Glm5DsaIndexer(config).cuda()
    B, S = 1, 32
    H, D = config.index_n_heads, config.index_head_dim
    iq = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
    ik = torch.randn((B, S, D), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((B, S, H), device="cuda", dtype=torch.float32)

    # 2D padding mask: first 24 positions valid, last 8 padded.
    pad_mask = torch.ones((B, S), device="cuda", dtype=torch.long)
    pad_mask[:, 24:] = 0

    # Detection returns valid_len per batch
    valid_lens = indexer._padding_mask_valid_lens(pad_mask, S)
    assert valid_lens is not None
    assert torch.all(valid_lens == 24)

    # Detection on an interspersed-padding mask should return None
    bad_mask = torch.ones((B, S), device="cuda", dtype=torch.long)
    bad_mask[:, 8:16] = 0  # holes in the middle — not a clean prefix
    assert indexer._padding_mask_valid_lens(bad_mask, S) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_causal_mask_detection_triggers_fast_path():
    """An explicit pure-causal additive mask should be detected and routed to
    the tilelang fast path (producing the same indices as both the
    attention_mask=None path and the torch fallback)."""
    try:
        from xorl.ops.glm5_kernels.tilelang_indexer_fwd import tl_indexer_fwd_impl  # noqa: F401
    except Exception:
        pytest.skip("tilelang indexer fwd kernel unavailable")
    torch.manual_seed(0)
    config = _tiny_config(index_topk=8, index_n_heads=4)
    indexer = Glm5DsaIndexer(config).cuda()
    B, S = 1, 32
    H, D = config.index_n_heads, config.index_head_dim
    iq = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
    ik = torch.randn((B, S, D), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((B, S, H), device="cuda", dtype=torch.float32)

    causal_mask = torch.zeros((B, S, S), device="cuda", dtype=torch.float32)
    for q in range(S):
        causal_mask[:, q, q + 1 :] = torch.finfo(torch.float32).min

    # Detection itself should return True for the canonical causal mask
    assert indexer._is_pure_causal_mask(causal_mask, S, S, 0)

    # And the masked call should give the same indices as the unmasked call
    indices_no_mask = indexer.select_topk(iq, ik, weights, attention_mask=None)
    indices_with_mask = indexer.select_topk(iq, ik, weights, attention_mask=causal_mask)
    for b in range(B):
        for s in range(S):
            assert set(indices_no_mask[b, s].cpu().tolist()) - {-1} == set(indices_with_mask[b, s].cpu().tolist()) - {
                -1
            }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_tilelang_fast_path_matches_torch_reference():
    """The tilelang fast path (taken when attention_mask is None and inputs
    are bf16/CUDA) should produce the same topk indices as the torch path.
    Compared as sets per row because both paths sort indices ascending,
    but the underlying scoring kernels may break ties differently."""
    try:
        from xorl.ops.glm5_kernels.tilelang_indexer_fwd import tl_indexer_fwd_impl  # noqa: F401
    except Exception:
        pytest.skip("tilelang indexer fwd kernel unavailable")
    torch.manual_seed(0)
    config = _tiny_config(index_topk=8, index_n_heads=4)
    indexer = Glm5DsaIndexer(config).cuda()

    B, S = 1, 32
    H, D = config.index_n_heads, config.index_head_dim
    iq = torch.randn((B, S, H, D), device="cuda", dtype=torch.bfloat16)
    ik = torch.randn((B, S, D), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((B, S, H), device="cuda", dtype=torch.float32)

    # Tilelang fast path (attention_mask=None triggers it)
    indices_tl = indexer.select_topk(iq, ik, weights, attention_mask=None)

    # Torch path (force via a strict causal mask)
    causal_mask = torch.zeros((B, S, S), device="cuda", dtype=torch.float32)
    for q in range(S):
        causal_mask[:, q, q + 1 :] = torch.finfo(torch.float32).min
    indices_ref = indexer.select_topk(iq, ik, weights, attention_mask=causal_mask)

    # Per-row set comparison; ignore -1 sentinels
    for b in range(B):
        for s in range(S):
            tl_set = set(indices_tl[b, s].cpu().tolist()) - {-1}
            ref_set = set(indices_ref[b, s].cpu().tolist()) - {-1}
            assert tl_set == ref_set, f"row mismatch at b={b} s={s}: tl={tl_set} ref={ref_set}"


def test_indexer_select_topk_returns_sorted_indices():
    """The indexer should sort top-k indices ascending by kv position
    (with -1 sentinels at the end). This is a perf optimization for the
    downstream sparse-MLA kernel — gathers become near-contiguous → better
    L2 cache hit rate. Locks in the ordering so a future change doesn't
    silently regress.
    """
    torch.manual_seed(0)
    config = _tiny_config(index_topk=4, index_n_heads=4)
    indexer = Glm5DsaIndexer(config)
    B, S = 2, 16
    index_q = torch.randn(B, S, config.index_n_heads, config.index_head_dim)
    index_k = torch.randn(B, S, config.index_head_dim)
    head_weights = torch.randn(B, S, config.index_n_heads)
    indices = indexer.select_topk(index_q, index_k, head_weights, attention_mask=None)

    # Within each (batch, query) row, valid indices must be monotonically
    # non-decreasing; -1 sentinels (if any) must follow all valid indices.
    for b in range(indices.shape[0]):
        for s in range(indices.shape[1]):
            row = indices[b, s]
            valid = row[row >= 0]
            if valid.numel() > 1:
                assert torch.all(valid[:-1] <= valid[1:]), f"row not sorted: {row.tolist()}"
            # If there are -1s, they all come after the last valid index
            neg_positions = (row == -1).nonzero(as_tuple=False)
            valid_positions = (row >= 0).nonzero(as_tuple=False)
            if neg_positions.numel() and valid_positions.numel():
                assert neg_positions.min() > valid_positions.max(), f"-1 not at end: {row.tolist()}"


def test_indexer_blocked_select_topk_matches_dense():
    """The memory-bounded scorer should preserve dense top-k semantics."""
    torch.manual_seed(0)
    config = _tiny_config(index_topk=4, index_n_heads=4)
    config.indexer_score_chunk_heads = 1
    indexer = Glm5DsaIndexer(config)

    B, S = 2, 7
    index_q = torch.randn(B, S, config.index_n_heads, config.index_head_dim)
    index_k = torch.randn(B, S, config.index_head_dim)
    head_weights = torch.randn(B, S, config.index_n_heads)

    dense = indexer.select_topk(index_q, index_k, head_weights, attention_mask=None)

    config.indexer_score_query_block_size = 2
    config.indexer_score_key_block_size = 3
    blocked = indexer.select_topk(index_q, index_k, head_weights, attention_mask=None)

    torch.testing.assert_close(torch.sort(blocked, dim=-1).values, torch.sort(dense, dim=-1).values)


def test_indexer_blocked_select_topk_supports_query_offset():
    """Ulysses query-sharded indexer rows should match dense full-sequence rows."""
    torch.manual_seed(0)
    config = _tiny_config(index_topk=4, index_n_heads=4)
    indexer = Glm5DsaIndexer(config)

    B, S = 1, 8
    index_q = torch.randn(B, S, config.index_n_heads, config.index_head_dim)
    index_k = torch.randn(B, S, config.index_head_dim)
    head_weights = torch.randn(B, S, config.index_n_heads)

    dense = indexer.select_topk(index_q, index_k, head_weights, attention_mask=None)

    config.indexer_score_query_block_size = 2
    config.indexer_score_key_block_size = 3
    local = indexer.select_topk(
        index_q[:, 3:6],
        index_k,
        head_weights[:, 3:6],
        attention_mask=None,
        query_offset=3,
    )

    torch.testing.assert_close(local, dense[:, 3:6])


def test_dense_dsa_mask_handles_2d_and_3d_attention_masks():
    config = _tiny_config(index_topk=2)
    attn = Glm5Attention(config, layer_idx=0)

    topk_indices = torch.tensor(
        [
            [[0, -1], [1, 0], [2, 1], [3, 2]],
            [[0, -1], [1, 0], [2, 1], [3, 2]],
        ],
        dtype=torch.int32,
    )
    mask_2d = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    built_2d = attn._build_dsa_mask(topk_indices, 4, mask_2d, dtype=torch.float32, device=topk_indices.device)
    assert built_2d.shape == (2, 1, 4, 4)
    assert torch.isneginf(built_2d[0, 0, :, 3]).all()
    assert torch.isneginf(built_2d[1, 0, :, 2:]).all()

    mask_3d = torch.zeros(2, 4, 4)
    mask_3d[:, :, 0] = float("-inf")
    built_3d = attn._build_dsa_mask(topk_indices, 4, mask_3d, dtype=torch.float32, device=topk_indices.device)
    assert built_3d.shape == (2, 1, 4, 4)
    assert torch.isneginf(built_3d[:, 0, :, 0]).all()


def test_dsa_mask_gathers_local_ulysses_query_axis(monkeypatch):
    config = _tiny_config()
    attn = Glm5Attention(config, layer_idx=0)
    group = object()

    class ParallelState:
        cp_enabled = True
        ringattn_enabled = False
        ulysses_group = group

    def fake_gather_outputs(x, gather_dim, **_kwargs):
        return torch.cat([x + rank for rank in range(4)], dim=gather_dim)

    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.get_parallel_state", lambda: ParallelState())
    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.dist.get_world_size", lambda _group: 4)
    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.gather_outputs", fake_gather_outputs)

    local_mask = torch.zeros(2, 1, 8, 32)
    full_mask = attn._full_attention_mask_for_dsa(local_mask, full_seq_len=32)

    assert full_mask.shape == (2, 1, 32, 32)
    torch.testing.assert_close(full_mask[:, :, :8, :], local_mask)
    torch.testing.assert_close(full_mask[:, :, 8:16, :], local_mask + 1)


# --------------------------------------------------------------------------- #
# Sparse-MLA path
# --------------------------------------------------------------------------- #


@torch.no_grad()
def test_sparse_mla_torch_reference_matches_dense_at_full_topk():
    """Sparse-MLA with `topk >= seq_len` must equal dense softmax-attention
    over the same compressed KV (causality already imposed inside the
    sparse reference)."""
    torch.manual_seed(0)
    B, S, H = 2, 6, 4
    kv_lora = 16
    qk_rope = 4
    D = kv_lora + qk_rope

    q = torch.randn(B, S, H, D)
    kv = torch.randn(B, S, D)
    scaling = D**-0.5

    # `indices = [[[0, 1, ..., S-1]] * S] * B` — every query indexes every key,
    # so sparse must reproduce dense after the causal mask is applied.
    indices = torch.arange(S).view(1, 1, S).expand(B, S, S).contiguous()

    sparse_out = sparse_mla_torch_reference(q, kv, indices, scaling, kv_lora_rank=kv_lora)

    # Dense reference: full causal softmax-attention over the compressed KV.
    scores = torch.einsum("bshd,btd->bsht", q, kv) * scaling
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.view(1, S, 1, S), float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    v_compressed = kv[..., :kv_lora]
    dense_out = torch.einsum("bshk,bkd->bshd", weights, v_compressed)

    torch.testing.assert_close(sparse_out, dense_out, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_sparse_mla_torch_reference_supports_query_offset():
    """A local query shard with full-sequence indices should match the
    corresponding rows from the full-query reference."""
    torch.manual_seed(0)
    B, S, H = 1, 7, 3
    kv_lora = 16
    qk_rope = 4
    D = kv_lora + qk_rope

    q = torch.randn(B, S, H, D)
    kv = torch.randn(B, S, D)
    indices = torch.arange(S).view(1, 1, S).expand(B, S, S).contiguous()
    scaling = D**-0.5

    full = sparse_mla_torch_reference(q, kv, indices, scaling, kv_lora_rank=kv_lora)
    local = sparse_mla_torch_reference(
        q[:, 3:6],
        kv,
        indices[:, 3:6],
        scaling,
        kv_lora_rank=kv_lora,
        query_offset=3,
    )

    torch.testing.assert_close(local, full[:, 3:6], atol=1e-5, rtol=1e-5)


def test_sparse_mla_tilelang_wrapper_supports_local_query_full_kv(monkeypatch):
    """Ulysses sparse MLA sends local query rows with gathered full-sequence KV."""
    B, S_q, S_kv, H, D = 2, 3, 7, 4, 5
    kv_lora = D
    captured = {}

    class FakeSparseMLA:
        @staticmethod
        def apply(q_flat, kv_flat, indices_flat, scaling):
            captured["q_shape"] = tuple(q_flat.shape)
            captured["kv_shape"] = tuple(kv_flat.shape)
            captured["indices"] = indices_flat.clone()
            captured["scaling"] = scaling
            return torch.zeros(q_flat.shape[0], q_flat.shape[1], kv_lora, dtype=q_flat.dtype), None

    monkeypatch.setitem(
        sys.modules,
        "xorl.ops.glm5_kernels.sparse_mla",
        SimpleNamespace(SparseMLA=FakeSparseMLA),
    )

    q = torch.randn(B, S_q, H, D)
    kv = torch.randn(B, S_kv, D)
    indices = torch.tensor(
        [
            [[0, 3], [-1, 6], [2, 5]],
            [[0, 6], [-1, 1], [4, -1]],
        ],
    )

    out = _sparse_mla_tilelang(q, kv, indices, scaling=0.25, kv_lora_rank=kv_lora)

    assert out.shape == (B, S_q, H, kv_lora)
    assert captured["q_shape"] == (B * S_q, H, D)
    assert captured["kv_shape"] == (B * S_kv, 1, D)
    assert captured["scaling"] == 0.25
    expected_indices = torch.tensor(
        [
            [[0, 3]],
            [[-1, 6]],
            [[2, 5]],
            [[7, 13]],
            [[-1, 8]],
            [[11, -1]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(captured["indices"], expected_indices)


@torch.no_grad()
def test_glm5_attention_sparse_forward_smoke():
    """Glm5Attention with `_sparse_mla_enabled=True` produces the right shape."""
    torch.manual_seed(0)
    config = _tiny_config(index_topk=4)
    config._sparse_mla_enabled = True
    attn = Glm5Attention(config, layer_idx=0).eval()
    rotary = RotaryEmbedding(config=config)

    B, S = 2, 6
    hidden = torch.randn(B, S, config.hidden_size)
    pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = rotary(hidden, pos_ids)

    out, _ = attn(hidden, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == (B, S, config.hidden_size)


@torch.no_grad()
def test_glm5_attention_sparse_ulysses_keeps_query_and_topk_local(monkeypatch):
    torch.manual_seed(0)
    config = _tiny_config(index_topk=2)
    config._sparse_mla_enabled = True
    attn = Glm5Attention(config, layer_idx=0).eval()
    rotary = RotaryEmbedding(config=config)
    group = object()

    class ParallelState:
        cp_enabled = True
        ringattn_enabled = False
        ulysses_group = group

    captured = {}

    def fake_sparse_mla(q, kv, indices, *, query_offset=0, **_kwargs):
        captured["q_shape"] = tuple(q.shape)
        captured["kv_shape"] = tuple(kv.shape)
        captured["indices_shape"] = tuple(indices.shape)
        captured["query_offset"] = query_offset
        return torch.zeros((*q.shape[:-1], config.kv_lora_rank), dtype=q.dtype)

    original_select_topk = attn.indexer.select_topk

    def fake_select_topk(index_q, index_k, head_weights, attention_mask, *, query_offset=0):
        captured["mask_shape"] = None if attention_mask is None else tuple(attention_mask.shape)
        return original_select_topk(index_q, index_k, head_weights, attention_mask, query_offset=query_offset)

    def fake_gather_sequence(tensor, _group):
        return torch.cat([tensor, tensor], dim=1)

    def fake_gather_outputs(tensor, gather_dim, **_kwargs):
        return torch.cat([tensor, tensor], dim=gather_dim)

    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.get_parallel_state", lambda: ParallelState())
    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.dist.get_rank", lambda _group: 1)
    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.gather_outputs", fake_gather_outputs)
    monkeypatch.setattr("xorl.models.transformers.glm5.modeling_glm5.sparse_mla_dispatch", fake_sparse_mla)
    monkeypatch.setattr(attn, "_gather_ulysses_sequence_no_grad", fake_gather_sequence)
    monkeypatch.setattr(attn.indexer, "select_topk", fake_select_topk)

    B, S = 1, 3
    hidden = torch.randn(B, S, config.hidden_size)
    pos_ids = torch.arange(S).unsqueeze(0)
    cos, sin = rotary(hidden, pos_ids)
    local_query_full_key_mask = torch.zeros(B, 1, S, S * 2)

    out, _ = attn(hidden, position_embeddings=(cos, sin), attention_mask=local_query_full_key_mask)

    assert out.shape == (B, S, config.hidden_size)
    assert captured["q_shape"] == (B, S, config.num_attention_heads, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["kv_shape"] == (B, S * 2, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["indices_shape"] == (B, S, config.index_topk)
    assert captured["query_offset"] == S
    assert captured["mask_shape"] is None


@torch.no_grad()
def test_glm5_attention_sparse_matches_dense_when_topk_covers_full_seq():
    """When `index_topk >= seq_len`, the sparse and dense paths share the
    same MLA parameters and the sparse path's causal mask matches dense
    causal MLA — outputs must agree to numerical precision.

    Goes through the full `Glm5Model` so the standard pre-attention causal
    mask construction runs for both paths (eager attention only applies the
    causal mask when one is passed in; calling the attention module
    directly with `attention_mask=None` would compare causal-sparse to
    non-causal-dense).
    """
    torch.manual_seed(0)
    B, S = 2, 6
    config = _tiny_config(index_topk=S, max_position_embeddings=S * 2)

    model = Glm5Model(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    config._sparse_mla_enabled = False
    dense_out = model(input_ids=input_ids).last_hidden_state

    config._sparse_mla_enabled = True
    sparse_out = model(input_ids=input_ids).last_hidden_state

    torch.testing.assert_close(sparse_out, dense_out, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Dispatch + checkpoint handler
# --------------------------------------------------------------------------- #


@torch.no_grad()
def test_sparse_mla_dispatch_falls_back_to_torch_on_cpu():
    """`auto` backend on CPU must use the torch reference (tilelang is
    CUDA-only). Output equals the torch ref called directly."""
    torch.manual_seed(0)
    B, S, H = 1, 4, 2
    kv_lora, qk_rope = 16, 4
    D = kv_lora + qk_rope
    q = torch.randn(B, S, H, D)
    kv = torch.randn(B, S, D)
    indices = torch.arange(S).view(1, 1, S).expand(B, S, S).contiguous()

    auto = sparse_mla_dispatch(q, kv, indices, scaling=D**-0.5, kv_lora_rank=kv_lora, backend="auto")
    explicit = sparse_mla_torch_reference(q, kv, indices, D**-0.5, kv_lora)
    torch.testing.assert_close(auto, explicit)


def test_sparse_mla_dispatch_rejects_unknown_backend():
    q = torch.randn(1, 1, 1, 4)
    kv = torch.randn(1, 1, 4)
    indices = torch.zeros(1, 1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="unknown sparse-MLA backend"):
        sparse_mla_dispatch(q, kv, indices, scaling=1.0, kv_lora_rank=2, backend="bogus")


def test_glm5_checkpoint_handler_skips_layers_beyond_configured():
    """Real GLM-5.1 has 78 dense layers + 1 MTP layer (index 78). Smoke runs
    further reduce `num_hidden_layers`. Either way, the handler must drop
    layer keys at indices `>= num_hidden_layers` so partial-load works."""
    handler = Glm5CheckpointHandler(num_experts=8, num_hidden_layers=4)

    assert handler._normalize_key("model.layers.0.self_attn.q_a_proj.weight") == (
        "model.layers.0.self_attn.q_a_proj.weight"
    )
    # MTP layer at the real model's `num_hidden_layers` index — dropped.
    assert handler._normalize_key("model.layers.4.self_attn.q_a_proj.weight") is None
    # Far beyond — still dropped (covers the partial-load smoke case).
    assert handler._normalize_key("model.layers.78.embed_tokens.weight") is None
    # Non-layer keys are routed to the parent's normalizer untouched.
    assert handler._normalize_key("model.embed_tokens.weight") == "model.embed_tokens.weight"


def test_glm5_default_lora_targets_cover_mla_and_moe():
    """`xorl_glm5` model_type maps to GLM's MLA + MoE LoRA targets.
    Regressing this would silently fall back to llama-style `q/k/v/o_proj`
    and inject zero adapters."""
    config = _tiny_config()
    model = Glm5ForCausalLM(config)

    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=None)

    attn = model.model.layers[0].self_attn
    moe = model.model.layers[config.first_k_dense_replace].mlp
    assert isinstance(attn.q_a_proj, LoraLinear)
    assert isinstance(attn.q_b_proj, LoraLinear)
    assert isinstance(attn.kv_a_proj_with_mqa, LoraLinear)
    assert isinstance(attn.kv_b_proj, LoraLinear)
    assert isinstance(attn.o_proj, LoraLinear)
    assert isinstance(moe.shared_experts.gate_proj, LoraLinear)
    assert isinstance(moe.experts, MoEExpertsLoRA)
    # Indexer projections are intentionally not wrapped — they're tiny and
    # absorbed-form sparse-MLA expects the unwrapped weights.
    assert not isinstance(attn.indexer.wq_b, LoraLinear)
    assert not isinstance(attn.indexer.wk, LoraLinear)
    assert not isinstance(attn.indexer.weights_proj, LoraLinear)


def test_glm5_moe_eager_uses_ep_dispatch_path_when_ep_enabled(monkeypatch):
    config = _tiny_config()
    block = Glm5MoEBlock(config)
    tokens = 6
    expert_forward = MagicMock(return_value=torch.zeros(tokens, config.hidden_size))
    monkeypatch.setattr(block.experts, "forward", expert_forward)
    monkeypatch.setattr(block, "_eager_forward", MagicMock(side_effect=AssertionError("local eager path used")))

    with patch(
        "xorl.models.transformers.glm5.modeling_glm5.get_parallel_state",
        return_value=SimpleNamespace(ep_enabled=True),
    ):
        output, router_logits = block(torch.randn(2, 3, config.hidden_size))

    assert output.shape == (2, 3, config.hidden_size)
    assert router_logits.shape == (tokens, config.n_routed_experts)
    expert_forward.assert_called_once()


def test_glm5_checkpoint_handler_skip_key_fn_short_circuits_disk_reads():
    """`get_skip_key_fn` is the loader's early-skip hook — keys it returns
    True for never get their tensor data read from disk. Must catch the
    same MTP / partial-load case `_normalize_key` handles."""
    handler = Glm5CheckpointHandler(num_experts=8, num_hidden_layers=4)
    skip = handler.get_skip_key_fn()
    assert skip is not None
    assert skip("model.layers.0.self_attn.q_a_proj.weight") is False
    assert skip("model.layers.4.self_attn.q_a_proj.weight") is True  # MTP layer
    assert skip("model.layers.78.embed_tokens.weight") is True
    assert skip("model.embed_tokens.weight") is False


@torch.no_grad()
def test_sparse_path_picks_up_lora_delta_on_kv_b_proj():
    """`Glm5Attention.forward_sparse` accesses `kv_b_proj.weight` directly
    via the absorb form. When kv_b_proj is wrapped with LoraLinear, the
    sparse path must still reflect the LoRA delta — otherwise sparse +
    LoRA silently trains around a frozen kv_b_proj while dense + LoRA
    actually trains it. We force the LoRA delta non-zero (lora_B
    initializes to zeros), then check sparse ≈ dense at full topk.
    """
    torch.manual_seed(0)
    B, S = 1, 6
    config = _tiny_config(index_topk=S, max_position_embeddings=S * 2)
    model = Glm5Model(config).eval()

    # Inject LoRA, then set lora_B to a non-zero random tensor so the
    # delta is observable (default init has lora_B=0 → no-op LoRA).
    inject_lora_into_model(model, r=4, lora_alpha=8, target_modules=None)
    for layer in model.layers:
        kv_b = layer.self_attn.kv_b_proj
        torch.nn.init.normal_(kv_b.lora_B, std=0.05)

    input_ids = torch.randint(0, config.vocab_size, (B, S))

    config._sparse_mla_enabled = False
    dense_out = model(input_ids=input_ids).last_hidden_state

    config._sparse_mla_enabled = True
    sparse_out = model(input_ids=input_ids).last_hidden_state

    torch.testing.assert_close(sparse_out, dense_out, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# HF reference parity
# --------------------------------------------------------------------------- #


@torch.no_grad()
def test_glm5_logits_match_hf_reference_at_full_topk():
    """Compare `Glm5ForCausalLM` logits against `transformers`' reference
    `GlmMoeDsaForCausalLM` on the same tiny config + same weights.

    All-dense layers (`first_k_dense_replace == num_hidden_layers`) so the
    state_dict layout aligns 1:1 between HF and xorl without an MoE
    remapping. `index_topk == max_position_embeddings` so the indexer
    selects every causal key — the DSA mask becomes a no-op and the path
    is pure MLA. Catches algebra drift in MLA projections, RoPE on the
    pe band, RMSNorm, and lm_head wiring against the HF reference.

    With filtering active (`index_topk < seq_len`) the BF16-noise-driven
    top-k tie flips can swap one or two indices, so this version pins the
    no-filter case at tight tolerance and leaves the filtering-active
    smoke as a separate test.
    """
    pytest.importorskip("transformers.models.glm_moe_dsa")
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM  # noqa: PLC0415

    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=2,  # all-dense — no MoE state_dict remapping
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        max_position_embeddings=64,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=64,  # >>> seq_len so DSA mask is a no-op
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        pad_token_id=0,
    )
    hf_config = GlmMoeDsaConfig(**common, mlp_layer_types=["dense"] * 2, attn_implementation="eager")
    xorl_config = Glm5Config(**common)
    xorl_config._attn_implementation = "eager"
    xorl_config._activation_native = True
    xorl_config._moe_implementation = "eager"

    torch.manual_seed(0)
    hf_model = GlmMoeDsaForCausalLM(hf_config).eval()
    torch.manual_seed(0)
    xorl_model = Glm5ForCausalLM(xorl_config).eval()
    xorl_model.load_state_dict(hf_model.state_dict(), strict=True)

    input_ids = torch.tensor([[1, 5, 10, 15, 20, 25]])
    hf_logits = hf_model(input_ids=input_ids).logits
    xorl_logits = xorl_model.lm_head(xorl_model(input_ids=input_ids).last_hidden_state)

    # BF16-noise-class tolerance: the two implementations use different
    # reduction orders inside attention and the MLP fused-silu path.
    torch.testing.assert_close(xorl_logits, hf_logits, atol=5e-3, rtol=5e-3)


@torch.no_grad()
def test_glm5_for_causal_lm_forward_smoke():
    """End-to-end forward on a tiny model: shapes, MoE/dense layer split,
    and that the indexer is reachable through `model.model.layers[N].self_attn.indexer`."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = Glm5ForCausalLM(config).eval()

    # MoE / dense split honors first_k_dense_replace.
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx < config.first_k_dense_replace:
            assert isinstance(layer.mlp, Glm5MLP), f"layer {layer_idx} should be dense"
        else:
            assert isinstance(layer.mlp, MoEBlock), f"layer {layer_idx} should be MoE"
        # Indexer is attached at every layer.
        assert isinstance(layer.self_attn.indexer, Glm5DsaIndexer)

    B, S = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    out = model.model(input_ids=input_ids)

    assert out.last_hidden_state.shape == (B, S, config.hidden_size)


def test_glm5_recompute_before_dispatch_avoids_outer_layer_checkpoint():
    torch.manual_seed(0)
    config = _tiny_config(num_hidden_layers=1, first_k_dense_replace=0)
    model = Glm5Model(config).train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"gradient_checkpointing_method": "recompute_before_dispatch"}
    )
    layer = model.layers[0]

    outer_checkpoint = MagicMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs))
    layer_checkpoint = MagicMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs))
    model._gradient_checkpointing_func = outer_checkpoint
    layer._gradient_checkpointing_func = layer_checkpoint

    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    out = model(input_ids=input_ids)

    assert out.last_hidden_state.shape == (2, 6, config.hidden_size)
    assert layer.gradient_checkpointing is True
    assert layer._gradient_checkpointing_method == "recompute_before_dispatch"
    assert outer_checkpoint.call_count == 0
    assert layer_checkpoint.call_count == 1
    assert layer_checkpoint.call_args.args[0].__name__ == "_pre_dispatch_forward"
