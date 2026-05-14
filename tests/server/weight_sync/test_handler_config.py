from types import SimpleNamespace

import torch

from xorl.server.weight_sync.handler import (
    _DEFAULT_MOE_BUCKET_BYTES,
    _DEFAULT_P2P_MOE_BUCKET_BYTES,
    WeightSyncHandler,
    _moe_bucket_size_bytes,
)


def test_moe_bucket_default_is_backend_specific(monkeypatch):
    monkeypatch.delenv("XORL_WEIGHT_SYNC_BUCKET_BYTES", raising=False)

    assert _moe_bucket_size_bytes("nccl_broadcast") == _DEFAULT_MOE_BUCKET_BYTES
    assert _moe_bucket_size_bytes("p2p") == _DEFAULT_P2P_MOE_BUCKET_BYTES


def test_moe_bucket_env_override_is_explicit(monkeypatch):
    monkeypatch.setenv("XORL_WEIGHT_SYNC_BUCKET_BYTES", str(123 * 1024 * 1024))

    assert _moe_bucket_size_bytes("nccl_broadcast") == 123 * 1024 * 1024
    assert _moe_bucket_size_bytes("p2p") == 123 * 1024 * 1024


def test_unfuse_for_inference_fuses_deepseek_kimi_mla_a_projection_for_sglang():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=3,
        layer_types=[],
    )
    model = SimpleNamespace(config=config)
    q_a = torch.arange(3 * 8, dtype=torch.bfloat16).reshape(3, 8)
    kv_a = torch.arange(5 * 8, dtype=torch.bfloat16).reshape(5, 8)
    q_b = torch.ones(4, 3, dtype=torch.bfloat16)

    remapped = dict(
        WeightSyncHandler._unfuse_for_inference(
            [
                ("model.layers.0.self_attn.q_a_proj.weight", q_a),
                ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kv_a),
                ("model.layers.0.self_attn.q_b_proj.weight", q_b),
            ],
            model,
        )
    )

    assert "model.layers.0.self_attn.q_a_proj.weight" not in remapped
    assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" not in remapped
    torch.testing.assert_close(
        remapped["model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"],
        torch.cat([q_a, kv_a], dim=0),
    )
    torch.testing.assert_close(remapped["model.layers.0.self_attn.q_b_proj.weight"], q_b)
