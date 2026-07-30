from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from xorl.models import module_utils
from xorl.models.auto import _load_local_xorl_config
from xorl.models.registry import ModelRegistry
from xorl.models.transformers.minimax_m3.checkpoint_handler import MiniMaxM3CheckpointHandler
from xorl.models.transformers.minimax_m3.configuration_minimax_m3 import MiniMaxM3Config
from xorl.models.transformers.minimax_m3.modeling_minimax_m3 import (
    MINIMAX_M3_UNSUPPORTED_PARALLEL_MESSAGE,
    MiniMaxM3Router,
    MiniMaxM3SparseForCausalLM,
    _raise_if_minimax_parallel_unsupported,
    minimax_m3_swigluoai,
)
from xorl.models.transformers.minimax_m3.msa_attention import _to_paged_kv, minimax_msa_attention_forward


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _hf_minimax_config_dict():
    text_cfg = {
        "vocab_size": 200064,
        "hidden_size": 6144,
        "num_hidden_layers": 60,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "use_gemma_norm": True,
        "rope_theta": 5000000.0,
        "rotary_dim": 64,
        "partial_rotary_factor": 0.5,
        "hidden_act": "swigluoai",
        "dense_intermediate_size": 12288,
        "intermediate_size": 3072,
        "shared_intermediate_size": 3072,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "routed_scaling_factor": 2.0,
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "use_qk_norm": True,
        "qk_norm_type": "per_head",
        "moe_layer_freq": [0, 0, 0] + [1] * 57,
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_attention_freq": [0, 0, 0] + [1] * 57,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16,
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_init_block": 1,
            "sparse_local_block": 1,
        },
        "tie_word_embeddings": False,
    }
    return {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "image_token_index": 200025,
        "video_token_index": 200026,
        "text_config": text_cfg,
        "vision_config": {"hidden_size": 1024},
        "tie_word_embeddings": False,
    }


def _tiny_config(**overrides):
    values = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rotary_dim=4,
        partial_rotary_factor=0.5,
        dense_intermediate_size=64,
        intermediate_size=16,
        shared_intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layer_freq=[0, 0, 0, 1],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [0, 0, 0, 1],
            "sparse_block_size": 4,
            "sparse_topk_blocks": 16,
            "sparse_num_index_heads": 2,
            "sparse_index_dim": 8,
            "sparse_init_block": 1,
            "sparse_local_block": 1,
        },
        _moe_implementation="eager",
    )
    values.update(overrides)
    return MiniMaxM3Config(**values)


def test_minimax_m3_config_adapts_top_level_hf_config():
    cfg = MiniMaxM3Config.from_hf_config(_namespace(_hf_minimax_config_dict()))

    assert cfg.model_type == "xorl_minimax_m3"
    assert cfg.architectures == ["MiniMaxM3SparseForConditionalGeneration"]
    assert cfg.hidden_size == 6144
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 128
    assert cfg.max_position_embeddings == 1048576
    assert cfg.rope_theta == 5000000.0
    assert cfg.partial_rotary_factor == 0.5
    assert cfg.hidden_act == "swigluoai"
    assert cfg.moe_layer_freq[:4] == [0, 0, 0, 1]
    assert cfg.sparse_attention_freq[:4] == [0, 0, 0, 1]
    assert cfg.sparse_topk_blocks == 16
    assert cfg.text_only is True
    assert cfg.image_token_index == 200025
    assert cfg.vision_config == {"hidden_size": 1024}
    assert cfg._moe_implementation == "native"


def test_minimax_m3_local_config_loader_and_registry(tmp_path):
    config_dir = tmp_path / "minimax"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(__import__("json").dumps(_hf_minimax_config_dict()))

    cfg = _load_local_xorl_config(str(config_dir), {})

    assert isinstance(cfg, MiniMaxM3Config)
    assert "MiniMaxM3SparseForConditionalGeneration" in ModelRegistry.supported_models
    assert "MiniMaxM3SparseForCausalLM" in ModelRegistry.supported_models


def test_minimax_m3_config_adapts_xorl_native_config_without_text_config():
    original = _tiny_config(text_config=None)

    cfg = MiniMaxM3Config.from_hf_config(original)

    assert cfg.hidden_size == original.hidden_size
    assert cfg.num_hidden_layers == original.num_hidden_layers
    assert cfg.sparse_attention_freq == original.sparse_attention_freq


def test_minimax_m3_swigluoai_matches_oai_formula():
    gate = torch.tensor([[-9.0, -1.0, 1.0, 9.0]])
    up = torch.tensor([[-9.0, -1.0, 1.0, 9.0]])

    actual = minimax_m3_swigluoai(gate, up, alpha=1.702, limit=7.0)

    expected_gate = gate.clamp(max=7.0)
    expected_up = up.clamp(min=-7.0, max=7.0)
    expected = expected_gate * torch.sigmoid(1.702 * expected_gate) * (expected_up + 1.0)
    torch.testing.assert_close(actual, expected)


def test_minimax_m3_sigmoid_router_uses_bias_only_for_selection():
    router = MiniMaxM3Router(num_experts=4, top_k=2, routed_scaling_factor=2.0, use_routing_bias=True)
    logits = torch.tensor([[4.0, 3.0, -1.0, -2.0]])
    bias = torch.tensor([-10.0, -10.0, 20.0, 0.0])

    weights, selected = router(logits, torch.float32, expert_bias=bias)

    assert selected.tolist() == [[2, 3]]
    scores = torch.sigmoid(logits)
    expected = scores.gather(1, selected)
    expected = expected / expected.sum(dim=-1, keepdim=True) * 2.0
    torch.testing.assert_close(weights, expected)


def test_minimax_m3_tiny_forward_backward_with_labels():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = MiniMaxM3SparseForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size - 3, (2, 8))

    out = model(input_ids=input_ids, labels=input_ids)

    assert out.loss is not None
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.last_hidden_state.shape == (2, 8, cfg.hidden_size)
    out.loss.backward()
    assert model.lm_head.weight.grad is not None


def test_minimax_m3_text_only_rejects_multimodal_inputs_and_tokens():
    cfg = _tiny_config(image_token_index=5, video_token_index=6)
    model = MiniMaxM3SparseForCausalLM(cfg)

    with pytest.raises(ValueError, match="image/video inputs"):
        model(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=torch.zeros(1, 3, 16, 16))

    with pytest.raises(ValueError, match="image token"):
        model(input_ids=torch.tensor([[1, 5, 3]]))

    with pytest.raises(ValueError, match="video token"):
        model(input_ids=torch.tensor([[1, 6, 3]]))


def test_minimax_m3_unsupported_parallel_modes_fail_clearly():
    ps = SimpleNamespace(tp_size=2, pp_size=1, ringattn_size=1, ulysses_size=1, lm_head_tp_size=1)

    with pytest.raises(ValueError, match="supports data/FSDP2 and expert parallelism only"):
        _raise_if_minimax_parallel_unsupported(ps)

    assert "tensor parallelism" in MINIMAX_M3_UNSUPPORTED_PARALLEL_MESSAGE


def test_minimax_m3_checkpoint_handler_maps_language_weights_and_skips_multimodal():
    handler = MiniMaxM3CheckpointHandler(num_experts=2)
    hidden = 2
    intermediate = 3
    results = []
    for expert_idx in range(2):
        base = float(expert_idx * 100)
        results.extend(
            handler.on_load_weight(
                f"language_model.model.layers.3.block_sparse_moe.experts.{expert_idx}.w1.weight",
                torch.arange(6, dtype=torch.float32).reshape(intermediate, hidden) + base,
            )
        )
        results.extend(
            handler.on_load_weight(
                f"language_model.model.layers.3.block_sparse_moe.experts.{expert_idx}.w3.weight",
                torch.arange(6, dtype=torch.float32).reshape(intermediate, hidden) + base + 10,
            )
        )
        results.extend(
            handler.on_load_weight(
                f"language_model.model.layers.3.block_sparse_moe.experts.{expert_idx}.w2.weight",
                torch.arange(6, dtype=torch.float32).reshape(hidden, intermediate) + base + 20,
            )
        )

    mapped = dict(results)
    assert set(mapped) == {
        "model.layers.3.mlp.experts.gate_up_proj",
        "model.layers.3.mlp.experts.down_proj",
    }
    assert mapped["model.layers.3.mlp.experts.gate_up_proj"].shape == (2, hidden, 2 * intermediate)
    assert mapped["model.layers.3.mlp.experts.down_proj"].shape == (2, intermediate, hidden)

    dense_gate = torch.ones(intermediate, hidden)
    dense_up = torch.full((intermediate, hidden), 2.0)
    assert handler.on_load_weight("language_model.model.layers.0.mlp.gate_proj.weight", dense_gate) == []
    merged_dense = handler.on_load_weight("language_model.model.layers.0.mlp.up_proj.weight", dense_up)
    assert len(merged_dense) == 1
    assert merged_dense[0][0] == "model.layers.0.mlp.gate_up_proj.weight"
    torch.testing.assert_close(merged_dense[0][1], torch.cat([dense_gate, dense_up], dim=0))

    gate_result = handler.on_load_weight("language_model.model.layers.3.block_sparse_moe.gate.weight", torch.ones(2, 2))
    assert gate_result[0][0] == "model.layers.3.mlp.gate.weight"
    torch.testing.assert_close(gate_result[0][1], torch.ones(2, 2))

    bias_result = handler.on_load_weight(
        "language_model.model.layers.3.block_sparse_moe.e_score_correction_bias", torch.ones(2)
    )
    assert bias_result[0][0] == "model.layers.3.mlp.e_score_correction_bias"
    torch.testing.assert_close(bias_result[0][1], torch.ones(2))
    assert handler.on_load_weight("vision_tower.vision_model.embeddings.patch_embedding.weight", torch.ones(1)) == []
    assert handler.on_load_weight("multi_modal_projector.linear_1.weight", torch.ones(1)) == []
    assert handler.on_load_weight("patch_merge_mlp.linear_1.weight", torch.ones(1)) == []


def test_minimax_m3_checkpoint_handler_ep_skip_counts_raw_keys():
    handler = MiniMaxM3CheckpointHandler(num_experts=4, ep_rank=1, ep_size=2)
    skip = handler.get_skip_key_fn()

    assert skip("language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight")
    assert not skip("language_model.model.layers.3.block_sparse_moe.experts.2.w1.weight")

    results = []
    for proj in ("w1", "w3", "w2"):
        for expert_idx in range(4):
            key = f"language_model.model.layers.3.block_sparse_moe.experts.{expert_idx}.{proj}.weight"
            if skip(key):
                results.extend(handler.on_skip_weight(key))
            else:
                results.extend(handler.on_load_weight(key, torch.ones(2, 2) * expert_idx))

    mapped = dict(results)
    assert mapped["model.layers.3.mlp.experts.gate_up_proj"].shape[0] == 2
    assert mapped["model.layers.3.mlp.experts.down_proj"].shape[0] == 2


def test_minimax_m3_msa_cpu_path_fails_loudly_and_paging_is_stable():
    x = torch.arange(2 * 3 * 1 * 2, dtype=torch.float32).reshape(2, 3, 1, 2)
    pages, indices = _to_paged_kv(x, torch.tensor([3, 1], dtype=torch.int32), page_size=2)
    assert pages.shape == (3, 1, 2, 2)
    assert indices.tolist() == [0, 1, 2]

    q = torch.zeros(1, 2, 1, 128)
    with pytest.raises(RuntimeError, match="requires CUDA/SM100"):
        minimax_msa_attention_forward(
            object(),
            q,
            q,
            q,
            q,
            q,
            scaling=1.0 / math.sqrt(128),
            topk_blocks=16,
            block_size=128,
            force_begin_blocks=1,
            force_end_blocks=1,
        )


@pytest.mark.parametrize(
    "key",
    [
        "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight",
        "model.language_model.model.layers.3.block_sparse_moe.experts.0.w2.weight",
        "model.layers.3.block_sparse_moe.experts.0.w3.weight",
    ],
)
def test_minimax_m3_grouped_loader_classifies_block_sparse_experts(key):
    assert module_utils._is_checkpoint_expert_key(key)
