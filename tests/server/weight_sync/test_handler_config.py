from types import SimpleNamespace

import torch

from xorl.server.weight_sync.handler import (
    _DEFAULT_MOE_BUCKET_BYTES,
    _DEFAULT_P2P_MOE_BUCKET_BYTES,
    WeightSyncHandler,
    _moe_bucket_size_bytes,
    _p2p_direct_ep_sender_ep_ranks,
    _p2p_direct_ep_sender_ranks,
    _should_collect_ep_moe_tensors,
)


def test_moe_bucket_default_is_backend_specific(monkeypatch):
    monkeypatch.delenv("XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES", raising=False)
    monkeypatch.delenv("XORL_WEIGHT_SYNC_BUCKET_BYTES", raising=False)

    assert _moe_bucket_size_bytes("nccl_broadcast") == _DEFAULT_MOE_BUCKET_BYTES
    assert _moe_bucket_size_bytes("p2p") == _DEFAULT_P2P_MOE_BUCKET_BYTES


def test_legacy_moe_bucket_env_override_is_explicit(monkeypatch):
    monkeypatch.delenv("XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES", raising=False)
    monkeypatch.setenv("XORL_WEIGHT_SYNC_BUCKET_BYTES", str(123 * 1024 * 1024))

    assert _moe_bucket_size_bytes("nccl_broadcast") == 123 * 1024 * 1024
    assert _moe_bucket_size_bytes("p2p") == 123 * 1024 * 1024


def test_dedicated_moe_bucket_env_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("XORL_WEIGHT_SYNC_BUCKET_BYTES", str(123 * 1024 * 1024))
    monkeypatch.setenv("XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES", str(456 * 1024 * 1024))

    assert _moe_bucket_size_bytes("nccl_broadcast") == 456 * 1024 * 1024
    assert _moe_bucket_size_bytes("p2p") == 456 * 1024 * 1024


def test_p2p_fp8_sync_requires_receiver_post_process():
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "fp8"}) is True
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "awq"}) is False
    assert WeightSyncHandler._p2p_requires_post_process_weights(None) is False


def test_chunk_buffer_by_bytes_splits_large_dense_items():
    items = [
        ("a", torch.zeros(4, dtype=torch.float32)),
        ("b", torch.zeros(8, dtype=torch.float32)),
        ("c", torch.zeros(4, dtype=torch.float32)),
    ]

    chunks = WeightSyncHandler._chunk_buffer_by_bytes(items, bucket_size_bytes=32)

    assert [[name for name, _ in chunk] for chunk in chunks] == [["a"], ["b"], ["c"]]


def test_would_exceed_bucket_cap_flushes_before_oversized_append():
    assert WeightSyncHandler._would_exceed_bucket_cap(900, 200, 1024) is True
    assert WeightSyncHandler._would_exceed_bucket_cap(0, 2048, 1024) is False
    assert WeightSyncHandler._would_exceed_bucket_cap(800, 224, 1024) is False


def test_p2p_direct_ep_sender_ranks_default_to_first_ep_fsdp_replica(monkeypatch):
    monkeypatch.delenv("XORL_P2P_DIRECT_EP_REPLICA_STRATEGY", raising=False)
    ps = SimpleNamespace(
        ep_enabled=True,
        ep_fsdp_device_mesh=SimpleNamespace(
            mesh=torch.tensor(
                [
                    [0, 8, 16, 24],
                    [1, 9, 17, 25],
                    [2, 10, 18, 26],
                    [3, 11, 19, 27],
                    [4, 12, 20, 28],
                    [5, 13, 21, 29],
                    [6, 14, 22, 30],
                    [7, 15, 23, 31],
                ]
            )
        ),
    )

    assert _p2p_direct_ep_sender_ranks(ps, 32) == tuple(range(8))


def test_p2p_direct_ep_sender_ranks_can_round_robin_replicas(monkeypatch):
    monkeypatch.setenv("XORL_P2P_DIRECT_EP_REPLICA_STRATEGY", "round_robin")
    ps = SimpleNamespace(
        ep_enabled=True,
        ep_fsdp_device_mesh=SimpleNamespace(
            mesh=torch.tensor(
                [
                    [0, 8, 16, 24],
                    [1, 9, 17, 25],
                    [2, 10, 18, 26],
                    [3, 11, 19, 27],
                    [4, 12, 20, 28],
                    [5, 13, 21, 29],
                    [6, 14, 22, 30],
                    [7, 15, 23, 31],
                ]
            )
        ),
    )

    assert _p2p_direct_ep_sender_ranks(ps, 32) == (0, 4, 9, 13, 18, 22, 27, 31)


def test_p2p_direct_ep_sender_ep_ranks_tracks_selected_replicas(monkeypatch):
    monkeypatch.setenv("XORL_P2P_DIRECT_EP_REPLICA_STRATEGY", "round_robin")
    ps = SimpleNamespace(
        ep_enabled=True,
        ep_size=8,
        ep_fsdp_device_mesh=SimpleNamespace(
            mesh=torch.tensor(
                [
                    [0, 8, 16, 24],
                    [1, 9, 17, 25],
                    [2, 10, 18, 26],
                    [3, 11, 19, 27],
                    [4, 12, 20, 28],
                    [5, 13, 21, 29],
                    [6, 14, 22, 30],
                    [7, 15, 23, 31],
                ]
            )
        ),
    )
    sender_ranks = _p2p_direct_ep_sender_ranks(ps, 32)

    assert _p2p_direct_ep_sender_ep_ranks(ps, sender_ranks, 32) == (
        (0, 0),
        (4, 4),
        (9, 1),
        (13, 5),
        (18, 2),
        (22, 6),
        (27, 3),
        (31, 7),
    )


def test_p2p_explicit_direct_ep_non_senders_skip_moe_tensor_collection():
    backend = SimpleNamespace(
        supports_direct_ep_transfer=True,
        has_explicit_sender_ranks=True,
    )

    assert _should_collect_ep_moe_tensors("p2p", backend, is_sender=False) is False
    assert _should_collect_ep_moe_tensors("p2p", backend, is_sender=True) is True


def test_moe_tensor_collection_is_kept_for_non_explicit_paths():
    direct_backend = SimpleNamespace(
        supports_direct_ep_transfer=True,
        has_explicit_sender_ranks=False,
    )
    nccl_backend = SimpleNamespace(
        supports_direct_ep_transfer=False,
        has_explicit_sender_ranks=False,
    )

    assert _should_collect_ep_moe_tensors("p2p", direct_backend, is_sender=False) is True
    assert _should_collect_ep_moe_tensors("nccl_broadcast", nccl_backend, is_sender=False) is True


def test_extract_params_can_defer_tied_weight_duplicate_for_p2p():
    class Root(torch.nn.Module):
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 3)
            self.lm_head = torch.nn.Linear(3, 4, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    aliases = {}

    class FakeDTensor:
        pass

    buffer = WeightSyncHandler._extract_params_for_sync(
        Root(),
        "(root)",
        FakeDTensor,
        emit_tied_weight_duplicates=False,
        tied_weight_aliases=aliases,
    )

    assert [name for name, _ in buffer] == ["model.embed_tokens.weight"]
    assert aliases == {"lm_head.weight": "model.embed_tokens.weight"}


def test_extract_params_include_param_skips_unowned_dense_params():
    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.keep = torch.nn.Linear(3, 4, bias=False)
            self.skip = torch.nn.Linear(3, 4, bias=False)

    class FakeDTensor:
        pass

    buffer = WeightSyncHandler._extract_params_for_sync(
        Root(),
        "(root)",
        FakeDTensor,
        include_param=lambda name: name == "keep.weight",
    )

    assert [name for name, _ in buffer] == ["keep.weight"]


def test_extract_params_skips_tied_weight_seen_in_prior_module_for_p2p():
    class Root(torch.nn.Module):
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 3)
            self.lm_head = torch.nn.Linear(3, 4, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    class FakeDTensor:
        pass

    root = Root()
    aliases = {}

    root_buffer = WeightSyncHandler._extract_params_for_sync(
        root,
        "(root)",
        FakeDTensor,
        emit_tied_weight_duplicates=False,
        tied_weight_aliases=aliases,
    )
    lm_head_buffer = WeightSyncHandler._extract_params_for_sync(
        root.lm_head,
        "lm_head",
        FakeDTensor,
        emit_tied_weight_duplicates=False,
        tied_weight_aliases=aliases,
    )

    assert [name for name, _ in root_buffer] == ["model.embed_tokens.weight"]
    assert lm_head_buffer == []
    assert aliases == {"lm_head.weight": "model.embed_tokens.weight"}


def test_extract_params_does_not_defer_declared_tie_when_parameters_differ():
    class Root(torch.nn.Module):
        _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 3)
            self.lm_head = torch.nn.Linear(3, 4, bias=False)

        def named_parameters(self, *args, **kwargs):
            yield "model.embed_tokens.weight", self.model.embed_tokens.weight

    class FakeDTensor:
        pass

    aliases = {}
    buffer = WeightSyncHandler._extract_params_for_sync(
        Root(),
        "(root)",
        FakeDTensor,
        emit_tied_weight_duplicates=False,
        tied_weight_aliases=aliases,
    )

    assert [name for name, _ in buffer] == ["model.embed_tokens.weight"]
    assert aliases == {}


def test_extract_params_does_not_infer_tied_weight_from_storage_only():
    class FakeDTensor:
        pass

    source = torch.nn.Linear(3, 4, bias=False)
    alias = torch.nn.Linear(3, 4, bias=False)
    alias.weight = source.weight

    aliases = {}
    source_buffer = WeightSyncHandler._extract_params_for_sync(
        source,
        "source",
        FakeDTensor,
        emit_tied_weight_duplicates=False,
        tied_weight_aliases=aliases,
    )
    alias_buffer = WeightSyncHandler._extract_params_for_sync(
        alias,
        "alias",
        FakeDTensor,
        emit_tied_weight_duplicates=False,
        tied_weight_aliases=aliases,
    )

    assert [name for name, _ in source_buffer] == ["source.weight"]
    assert [name for name, _ in alias_buffer] == ["alias.weight"]
    assert aliases == {}


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
