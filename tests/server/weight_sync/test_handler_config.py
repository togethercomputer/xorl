import asyncio
from types import SimpleNamespace

import pytest
import torch

from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.transformers.nemotron_h.checkpoint_handler import NemotronHCheckpointHandler
from xorl.server.protocol.operations import SyncWeightsData
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


def test_p2p_fp8_sync_skips_receiver_post_process_by_default(monkeypatch):
    monkeypatch.delenv("XORL_P2P_RUN_POST_PROCESS_WEIGHTS", raising=False)
    monkeypatch.delenv("XORL_WEIGHT_SYNC_RUN_POST_PROCESS_WEIGHTS", raising=False)

    assert WeightSyncHandler._p2p_should_run_post_process_weights({"quant_method": "fp8"}) is False
    assert WeightSyncHandler._p2p_should_run_post_process_weights({"quant_method": "awq"}) is False
    assert WeightSyncHandler._p2p_should_run_post_process_weights(None) is False


def test_p2p_fp8_post_process_can_be_enabled_for_legacy_receivers(monkeypatch):
    monkeypatch.delenv("XORL_P2P_RUN_POST_PROCESS_WEIGHTS", raising=False)
    monkeypatch.setenv("XORL_WEIGHT_SYNC_RUN_POST_PROCESS_WEIGHTS", "1")

    assert WeightSyncHandler._p2p_should_run_post_process_weights({"quant_method": "fp8"}) is True


def test_p2p_specific_post_process_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("XORL_WEIGHT_SYNC_RUN_POST_PROCESS_WEIGHTS", "1")
    monkeypatch.setenv("XORL_P2P_RUN_POST_PROCESS_WEIGHTS", "0")

    assert WeightSyncHandler._p2p_should_run_post_process_weights({"quant_method": "fp8"}) is False


def test_receiver_post_process_after_fp8_sync_respects_fp8_kv_cache_requirement(monkeypatch):
    monkeypatch.delenv("XORL_P2P_RUN_POST_PROCESS_WEIGHTS", raising=False)
    monkeypatch.delenv("XORL_WEIGHT_SYNC_RUN_POST_PROCESS_WEIGHTS", raising=False)

    assert (
        WeightSyncHandler._should_run_receiver_post_process_after_fp8_sync(
            "p2p",
            {"quant_method": "fp8"},
        )
        is False
    )
    assert (
        WeightSyncHandler._should_run_receiver_post_process_after_fp8_sync(
            "p2p",
            {"quant_method": "fp8"},
            fp8_kv_cache_postprocess_required=True,
        )
        is True
    )
    assert (
        WeightSyncHandler._should_run_receiver_post_process_after_fp8_sync(
            "nccl_broadcast",
            {"quant_method": "fp8"},
        )
        is True
    )
    assert (
        WeightSyncHandler._should_run_receiver_post_process_after_fp8_sync(
            "p2p",
            {"quant_method": "awq"},
            fp8_kv_cache_postprocess_required=True,
        )
        is False
    )


def test_endpoint_results_from_backend_preserve_fp8_kv_cache_metadata():
    backend = SimpleNamespace(
        endpoint_results=[
            {
                "host": "inference.example",
                "port": 30000,
                "success": True,
                "message": "ok",
                "cache_epoch": "epoch-8",
                "fp8_kv_cache_postprocess_ran": True,
                "fp8_kv_cache_static_scales_updated": True,
            }
        ]
    )

    results = WeightSyncHandler._endpoint_results_from_backend(
        [{"host": "inference.example", "port": 30000}],
        backend,
    )

    assert results == [
        {
            "host": "inference.example",
            "port": 30000,
            "success": True,
            "message": "ok",
            "cache_epoch": "epoch-8",
            "fp8_kv_cache_postprocess_ran": True,
            "fp8_kv_cache_static_scales_updated": True,
        }
    ]
    assert WeightSyncHandler._first_endpoint_cache_epoch(results) == "epoch-8"


def test_endpoint_results_from_backend_normalize_cache_version_alias():
    backend = SimpleNamespace(
        endpoint_results=[
            {
                "success": True,
                "message": "ok",
                "cache_version": "version-9",
                "fp8_kv_cache_postprocess_ran": True,
            }
        ]
    )

    results = WeightSyncHandler._endpoint_results_from_backend(
        [{"host": "inference.example", "port": 30000}],
        backend,
    )

    assert results == [
        {
            "host": "inference.example",
            "port": 30000,
            "success": True,
            "message": "ok",
            "cache_epoch": "version-9",
            "fp8_kv_cache_postprocess_ran": True,
        }
    ]
    assert WeightSyncHandler._first_endpoint_cache_epoch([{"cache_version": "version-10"}]) == "version-10"


def _run_fake_streaming_sync_for_backend_config(monkeypatch, *, quantization, fp8_kv_cache_postprocess_required):
    created_backends = []

    class FakeEndpointManager:
        def __init__(self, endpoints):
            self.endpoints = endpoints

        def health_check(self):
            return None

        def pause(self, mode):
            return [{"success": True}], True

        def resume(self):
            return []

    class FakeFSDPModule:
        def unshard(self):
            return None

        def reshard(self):
            return None

    class FakeBackend:
        supports_direct_ep_transfer = False

        def __init__(self, config):
            self.config = config
            self.endpoint_results = [
                {
                    "host": "inference.example",
                    "port": 30000,
                    "success": True,
                    "message": "ok",
                    "cache_epoch": "epoch-8",
                    "fp8_kv_cache_postprocess_ran": bool(fp8_kv_cache_postprocess_required),
                }
            ]
            self.transfers = []

        @property
        def sender_ranks(self):
            return frozenset({0})

        def initialize(self):
            return True

        def transfer_bucket(self, bucket, *, src_rank=0, flush_cache=False, weight_version=None):
            self.transfers.append(
                {
                    "names": [name for name, _ in bucket],
                    "flush_cache": flush_cache,
                    "weight_version": weight_version,
                    "src_rank": src_rank,
                }
            )

        def flush_pending_transfers(self):
            return None

        def destroy(self, *, complete_receiver=True):
            return None

    def fake_create_backend(method, cfg):
        backend = FakeBackend(cfg)
        created_backends.append(backend)
        return backend

    fake_ps = SimpleNamespace(
        ep_enabled=False,
        ep_size=1,
        pp_enabled=False,
        pp_rank=0,
        pp_size=1,
        dp_shard_rank=0,
    )

    monkeypatch.setenv("XORL_P2P_BACKEND_CACHE", "0")
    monkeypatch.setattr("xorl.server.weight_sync.handler.EndpointManager", FakeEndpointManager)
    monkeypatch.setattr("xorl.server.weight_sync.handler.create_backend", fake_create_backend)
    monkeypatch.setattr("xorl.server.weight_sync.handler.get_parallel_state", lambda: fake_ps)
    monkeypatch.setattr(
        WeightSyncHandler,
        "_get_fsdp_modules",
        staticmethod(lambda model: (FakeFSDPModule(), [])),
    )
    monkeypatch.setattr(
        WeightSyncHandler,
        "_qlora_collective_ops",
        lambda self, fsdp_mod, mod_name, collect_results=True: ([], []),
    )
    monkeypatch.setattr(
        WeightSyncHandler,
        "_extract_params_for_sync",
        staticmethod(
            lambda fsdp_mod, mod_name, DTensor, **kwargs: [
                ("model.layers.0.linear.weight", torch.ones(2, 2, dtype=torch.bfloat16))
            ]
        ),
    )
    monkeypatch.setattr(
        WeightSyncHandler,
        "_unfuse_for_inference",
        staticmethod(lambda buffer, model, clone_slices=True: buffer),
    )
    monkeypatch.setattr(
        WeightSyncHandler,
        "_quantize_buffer_for_fp8",
        lambda self, buffer, **kwargs: buffer,
    )

    trainer = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()), local_rank=0, optimizer=None, train_config={}
    )
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)
    result = handler._sync_weights(
        endpoints=[{"host": "inference.example", "port": 30000, "world_size": 1}],
        master_address="train.example",
        master_port=29500,
        group_name="weight_sync_group",
        buffer_size_mb=1,
        sync_method="p2p",
        flush_cache=True,
        pause_mode="retract",
        weight_version="sync-8",
        quantization=quantization,
        fp8_kv_cache_enabled=True,
        fp8_kv_cache_postprocess_required=fp8_kv_cache_postprocess_required,
        fp8_kv_cache_static_scales=fp8_kv_cache_postprocess_required,
    )
    return result, created_backends[0]


def test_streaming_sync_backend_config_requests_fp8_kv_cache_postprocess_only_when_required(monkeypatch):
    quantization = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}

    result, backend = _run_fake_streaming_sync_for_backend_config(
        monkeypatch,
        quantization=quantization,
        fp8_kv_cache_postprocess_required=True,
    )

    assert result["success"] is True
    assert result["flush_cache"] is True
    assert result["fp8_kv_cache_postprocess_requested"] is True
    assert result["cache_epoch"] == "epoch-8"
    assert backend.config.backend_config["run_post_process_weights"] is True
    assert backend.config.backend_config["fp8_kv_cache_enabled"] is True
    assert backend.config.backend_config["fp8_kv_cache_postprocess_required"] is True
    assert backend.config.backend_config["fp8_kv_cache_static_scales"] is True
    assert backend.transfers == [
        {
            "names": ["model.layers.0.linear.weight"],
            "flush_cache": True,
            "weight_version": "sync-8",
            "src_rank": 0,
        }
    ]

    result, backend = _run_fake_streaming_sync_for_backend_config(
        monkeypatch,
        quantization=quantization,
        fp8_kv_cache_postprocess_required=False,
    )

    assert result["success"] is True
    assert result["fp8_kv_cache_postprocess_requested"] is False
    assert backend.config.backend_config["run_post_process_weights"] is False
    assert backend.config.backend_config["fp8_kv_cache_enabled"] is True
    assert backend.config.backend_config["fp8_kv_cache_postprocess_required"] is False
    assert backend.config.backend_config["fp8_kv_cache_static_scales"] is False


def test_streaming_sync_backend_config_does_not_forward_fp8_kv_cache_knobs_for_bf16_sync(monkeypatch):
    result, backend = _run_fake_streaming_sync_for_backend_config(
        monkeypatch,
        quantization=None,
        fp8_kv_cache_postprocess_required=False,
    )

    assert result["success"] is True
    assert result["fp8_kv_cache_postprocess_requested"] is False
    assert "run_post_process_weights" not in backend.config.backend_config
    assert "fp8_kv_cache_enabled" not in backend.config.backend_config
    assert "fp8_kv_cache_postprocess_required" not in backend.config.backend_config
    assert "fp8_kv_cache_static_scales" not in backend.config.backend_config


def test_handle_sync_rejects_unsupported_quantization_before_trainer_access():
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)

    result = asyncio.run(
        handler.handle_sync_inference_weights(
            {"payload": SyncWeightsData(quantization={"quant_method": "compressed-tensors"})}
        )
    )

    assert result["success"] is False
    assert "INT4/compressed-tensors updates" in result["message"]


def test_handle_sync_enriches_fp8_quantization_with_training_bf16_islands(monkeypatch):
    class FakeModel:
        def named_modules(self):
            return []

        def get_pp_module_config(self):
            return {"layer_prefix": "model.layers", "num_layers": 4}

    trainer = SimpleNamespace(
        model=FakeModel(),
        train_config={
            "fp8_training_num_first_layers_bf16": 1,
            "fp8_training_num_last_layers_bf16": 1,
        },
    )
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)
    captured = {}

    def fake_sync_weights(**kwargs):
        captured.update(kwargs)
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(handler, "_sync_weights", fake_sync_weights)

    result = asyncio.run(
        handler.handle_sync_inference_weights(
            {
                "payload": SyncWeightsData(
                    endpoints=[{"host": "inference.example", "port": 30000, "world_size": 1}],
                    quantization={"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
                )
            }
        )
    )

    assert result["success"] is True
    assert captured["quantization"]["modules_to_not_convert"] == ["model.layers.0.*", "model.layers.3.*"]
    assert captured["quantization"]["_xorl_generated_bf16_layer_islands"] == [
        "model.layers.0.*",
        "model.layers.3.*",
    ]


def test_prepare_lora_adapter_for_sync_materializes_requested_adapter():
    class FakeAdapterManager:
        current_adapter_id = "current-adapter"

        def __init__(self):
            self.adapters = {"policy-a"}
            self.synced = []

        def has_adapter(self, model_id):
            return model_id in self.adapters

        def sync_weights_to_model(self, model_id):
            self.synced.append(model_id)

    class FakeTrainer:
        def __init__(self):
            self.adapter_manager = FakeAdapterManager()
            self.registered = []

        def register_lora_adapter(self, model_id, lr):
            self.registered.append((model_id, lr))
            self.adapter_manager.adapters.add(model_id)

    trainer = FakeTrainer()
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)

    assert handler._prepare_lora_adapter_for_sync("policy-b") == "policy-b"
    assert trainer.registered == [("policy-b", None)]
    assert trainer.adapter_manager.synced == ["policy-b"]


def test_prepare_lora_adapter_for_sync_defaults_to_current_adapter():
    class FakeAdapterManager:
        current_adapter_id = "current-adapter"

        def __init__(self):
            self.synced = []

        def has_adapter(self, model_id):
            return model_id == "current-adapter"

        def sync_weights_to_model(self, model_id):
            self.synced.append(model_id)

    trainer = SimpleNamespace(adapter_manager=FakeAdapterManager())
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)

    assert handler._prepare_lora_adapter_for_sync(None) == "current-adapter"
    assert trainer.adapter_manager.synced == ["current-adapter"]


def test_p2p_fp8_sync_requires_receiver_post_process():
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "fp8"}) is True
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "block_fp8"}) is True
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


def test_unfuse_for_inference_can_return_contiguous_views_for_fp8_path():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, layer_types=[])
    model = SimpleNamespace(config=config)
    qkv = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4)
    gate_up = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)

    cloned = dict(
        WeightSyncHandler._unfuse_for_inference(
            [
                ("model.layers.0.self_attn.qkv_proj.weight", qkv),
                ("model.layers.0.mlp.gate_up_proj.weight", gate_up),
            ],
            model,
        )
    )
    viewed = dict(
        WeightSyncHandler._unfuse_for_inference(
            [
                ("model.layers.0.self_attn.qkv_proj.weight", qkv),
                ("model.layers.0.mlp.gate_up_proj.weight", gate_up),
            ],
            model,
            clone_slices=False,
        )
    )

    q = viewed["model.layers.0.self_attn.q_proj.weight"]
    k = viewed["model.layers.0.self_attn.k_proj.weight"]
    gate = viewed["model.layers.0.mlp.gate_proj.weight"]
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert gate.is_contiguous()
    assert q.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    assert k.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    assert gate.untyped_storage().data_ptr() == gate_up.untyped_storage().data_ptr()
    assert (
        cloned["model.layers.0.self_attn.q_proj.weight"].untyped_storage().data_ptr()
        != qkv.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(q, qkv[:8])
    torch.testing.assert_close(k, qkv[8:16])
    torch.testing.assert_close(gate, gate_up[:8])


def _nemotron_h_model(num_experts: int = 4):
    def get_checkpoint_handler(**kwargs):
        return NemotronHCheckpointHandler(num_experts=num_experts)

    return SimpleNamespace(
        config=SimpleNamespace(model_type="nemotron_h"),
        get_checkpoint_handler=get_checkpoint_handler,
    )


def test_unfuse_for_inference_emits_nemotron_h_per_expert_hf_layout():
    num_experts, latent, intermediate = 4, 3, 5
    # Non-gated experts: gate_up_proj holds ONLY the up projection [E, latent, I].
    gate_up = torch.randn(num_experts, latent, intermediate, dtype=torch.bfloat16)
    down = torch.randn(num_experts, intermediate, latent, dtype=torch.bfloat16)
    buffer = [
        ("model.layers.2.mixer.experts.gate_up_proj", gate_up),
        ("model.layers.2.mixer.experts.down_proj", down),
        ("model.layers.2.mixer.gate.weight", torch.randn(num_experts, 8, dtype=torch.bfloat16)),
        ("model.layers.0.mixer.in_proj.weight", torch.randn(7, 8, dtype=torch.bfloat16)),
        ("lm_head.weight", torch.randn(16, 8, dtype=torch.bfloat16)),
    ]

    result = WeightSyncHandler._unfuse_for_inference(buffer, _nemotron_h_model(num_experts))

    # The sync-side transform must match on_save_weight (the published HF layout).
    expected = []
    oracle = NemotronHCheckpointHandler(num_experts=num_experts)
    for name, tensor in buffer:
        expected.extend(oracle.on_save_weight(name, tensor))
    assert [n for n, _ in result] == [n for n, _ in expected]
    for (_, got), (_, want) in zip(result, expected):
        torch.testing.assert_close(got, want)

    named = dict(result)
    # Per-expert backbone.* names with the [in, out] → [out, in] transpose,
    # and NO gate/up midpoint split (each up_proj covers the full width).
    for e in range(num_experts):
        up = named[f"backbone.layers.2.mixer.experts.{e}.up_proj.weight"]
        assert up.shape == (intermediate, latent)
        torch.testing.assert_close(up, gate_up[e].T)
        torch.testing.assert_close(named[f"backbone.layers.2.mixer.experts.{e}.down_proj.weight"], down[e].T)
    assert not any("gate_proj" in name for name in named)
    assert not any(name.startswith("model.") for name in named)
    assert "backbone.layers.2.mixer.gate.weight" in named
    assert "backbone.layers.0.mixer.in_proj.weight" in named
    assert "lm_head.weight" in named


def test_unfuse_for_inference_rejects_fused_dense_params_for_nemotron_h():
    with pytest.raises(ValueError, match="fused dense param"):
        WeightSyncHandler._unfuse_for_inference(
            [("model.layers.1.self_attn.qkv_proj.weight", torch.zeros(6, 4))],
            _nemotron_h_model(),
        )


def test_unfuse_for_inference_splits_gated_stacked_mlp_experts():
    # Regression: gated .mlp.experts behavior (qwen3_moe-style) is unchanged.
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, layer_types=[])
    model = SimpleNamespace(config=config)
    num_experts, hidden, intermediate = 2, 3, 4
    gate_up = torch.randn(num_experts, hidden, 2 * intermediate)
    down = torch.randn(num_experts, intermediate, hidden)

    named = dict(
        WeightSyncHandler._unfuse_for_inference(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("model.layers.0.mlp.experts.down_proj", down),
            ],
            model,
        )
    )

    assert len(named) == 3 * num_experts
    for e in range(num_experts):
        torch.testing.assert_close(
            named[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"],
            gate_up[e, :, :intermediate].T,
        )
        torch.testing.assert_close(
            named[f"model.layers.0.mlp.experts.{e}.up_proj.weight"],
            gate_up[e, :, intermediate:].T,
        )
        torch.testing.assert_close(named[f"model.layers.0.mlp.experts.{e}.down_proj.weight"], down[e].T)


def test_moe_inference_prefix_remap_renames_nemotron_h_prefixes_only():
    remap = WeightSyncHandler._moe_inference_prefix_remap(_nemotron_h_model())
    assert remap is not None
    assert remap("model.layers.7.mixer.experts") == "backbone.layers.7.mixer.experts"
    assert remap("lm_head") == "lm_head"

    other = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
    assert WeightSyncHandler._moe_inference_prefix_remap(other) is None


def test_collect_ep_moe_data_non_gated_experts_skip_gate_projection():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = MoEExperts(
                num_experts=4,
                hidden_dim=3,
                intermediate_size=5,
                hidden_act="relu2",
                moe_implementation="eager",
                gated=False,
            )

    wrapper = Wrapper()
    torch.nn.init.normal_(wrapper.experts.gate_up_proj)
    torch.nn.init.normal_(wrapper.experts.down_proj)

    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    contexts = handler._collect_ep_moe_data(wrapper, "model.layers.0.mixer", None)

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["prefix"] == "model.layers.0.mixer.experts"
    assert ctx["num_local_experts"] == 4
    assert ctx["projections"] == ("up_proj", "down_proj")
    assert set(ctx["local_experts"]) == {"up_proj", "down_proj"}
    # Non-gated: the half-width gate_up_proj IS the up projection, unsplit.
    assert ctx["local_experts"]["up_proj"].shape == (4, 3, 5)
    torch.testing.assert_close(
        ctx["local_experts"]["up_proj"],
        wrapper.experts.gate_up_proj.data.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        ctx["local_experts"]["down_proj"],
        wrapper.experts.down_proj.data.to(torch.bfloat16),
    )


def test_collect_ep_moe_data_gated_experts_keep_three_projections():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = MoEExperts(
                num_experts=2,
                hidden_dim=3,
                intermediate_size=5,
                hidden_act="silu",
                moe_implementation="eager",
                gated=True,
            )

    wrapper = Wrapper()
    torch.nn.init.normal_(wrapper.experts.gate_up_proj)
    torch.nn.init.normal_(wrapper.experts.down_proj)

    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    contexts = handler._collect_ep_moe_data(wrapper, "(root)", None)

    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["projections"] == ("gate_proj", "up_proj", "down_proj")
    torch.testing.assert_close(
        ctx["local_experts"]["gate_proj"],
        wrapper.experts.gate_up_proj.data[..., :5].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        ctx["local_experts"]["up_proj"],
        wrapper.experts.gate_up_proj.data[..., 5:].to(torch.bfloat16),
    )


def test_unfuse_for_inference_strips_orig_mod_wrapper_from_moe_names():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        layer_types=[],
    )
    model = SimpleNamespace(config=config)
    gate_up = torch.arange(1 * 2 * 6, dtype=torch.bfloat16).reshape(1, 2, 6)

    remapped = dict(
        WeightSyncHandler._unfuse_for_inference(
            [("model.layers.0._orig_mod.mlp.experts.gate_up_proj", gate_up)],
            model,
        )
    )

    assert "model.layers.0._orig_mod.mlp.experts.0.gate_proj.weight" not in remapped
    assert set(remapped) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
    }
    torch.testing.assert_close(
        remapped["model.layers.0.mlp.experts.0.gate_proj.weight"],
        gate_up[:, :, :3].transpose(1, 2).contiguous()[0],
    )


def test_broadcast_buffer_strips_orig_mod_before_backend_transfer():
    class FakeBackend:
        def __init__(self):
            self.bucket = None
            self.flush_cache = None
            self.weight_version = None

        def transfer_bucket(self, bucket, *, flush_cache=False, weight_version=None):
            self.bucket = bucket
            self.flush_cache = flush_cache
            self.weight_version = weight_version

    backend = FakeBackend()
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    tensor = torch.ones(2, dtype=torch.bfloat16)

    total_bytes, num_params = handler._broadcast_buffer(
        backend,
        [("model.layers.0._orig_mod.self_attn.q_proj.weight", tensor)],
        flush_cache=True,
        weight_version="wv",
    )

    assert total_bytes == tensor.numel() * tensor.element_size()
    assert num_params == 1
    assert [name for name, _ in backend.bucket] == ["model.layers.0.self_attn.q_proj.weight"]
    assert backend.flush_cache is True
    assert backend.weight_version == "wv"

def test_sparse_delta_paths_fast_path_bypasses_trainer_model(monkeypatch):
    events = []

    class FakeEndpointManager:
        def __init__(self, endpoints):
            self.endpoints = endpoints

        def health_check(self):
            events.append("health")

        def pause(self, mode):
            events.append(("pause", mode))
            return [{"success": True}], True

        def resume(self):
            events.append("resume")

    class FakeBackend:
        def __init__(self):
            self.paths = None
            self.destroyed = False

        def initialize(self):
            events.append("initialize")
            return True

        def post_packed_delta_paths(self, paths, *, flush_cache=False, weight_version=None):
            self.paths = list(paths)
            events.append(("post", list(paths), flush_cache, weight_version))

        def stats_summary(self):
            return {"total_packed_bytes": 123.0, "posted_files": 1.0, "post_s": 0.01}

        def destroy(self, *, complete_receiver=True):
            self.destroyed = True
            events.append(("destroy", complete_receiver))

    fake_backend = FakeBackend()
    captured_cfg = {}

    def fake_create_backend(method, cfg):
        captured_cfg["method"] = method
        captured_cfg["cfg"] = cfg
        return fake_backend

    monkeypatch.setattr("xorl.server.weight_sync.handler.EndpointManager", FakeEndpointManager)
    monkeypatch.setattr("xorl.server.weight_sync.handler.create_backend", fake_create_backend)

    handler = WeightSyncHandler(rank=0, world_size=8, trainer=None)
    result = handler._sync_sparse_delta_paths(
        endpoints=[{"host": "infer.example", "port": 30000, "world_size": 8}],
        group_name="weight_sync_group",
        flush_cache=True,
        pause_mode="retract",
        weight_version="fast-1",
        sparse_delta_paths=["/shared/delta.packed"],
    )

    assert result["success"] is True
    assert result["total_bytes"] == 123
    assert result["num_parameters"] == 0
    assert captured_cfg["method"] == "sparse_delta"
    assert captured_cfg["cfg"].endpoints[0].world_size == 8
    assert captured_cfg["cfg"].backend_config == {"post_only": True}
    assert ("post", ["/shared/delta.packed"], True, "fast-1") in events
    assert events.index("health") < events.index(("pause", "retract")) < events.index("resume")


def test_sparse_delta_paths_fast_path_passes_sparse_delta_config(monkeypatch):
    class FakeEndpointManager:
        def __init__(self, endpoints):
            self.endpoints = endpoints

        def health_check(self):
            return None

        def pause(self, mode):
            return [{"success": True}], True

        def resume(self):
            return []

    class FakeBackend:
        def initialize(self):
            return True

        def post_packed_delta_paths(self, paths, *, flush_cache=False, weight_version=None):
            return None

        def stats_summary(self):
            return {"total_packed_bytes": 0.0}

        def destroy(self, *, complete_receiver=True):
            return None

    captured_cfg = {}

    def fake_create_backend(method, cfg):
        captured_cfg["cfg"] = cfg
        return FakeBackend()

    monkeypatch.setattr("xorl.server.weight_sync.handler.EndpointManager", FakeEndpointManager)
    monkeypatch.setattr("xorl.server.weight_sync.handler.create_backend", fake_create_backend)

    handler = WeightSyncHandler(rank=0, world_size=8, trainer=None)
    result = handler._sync_sparse_delta_paths(
        endpoints=[{"host": "infer.example", "port": 30000, "world_size": 8}],
        group_name="weight_sync_group",
        flush_cache=True,
        pause_mode="retract",
        weight_version="fast-1",
        sparse_delta_paths=["/shared/delta.packed"],
        sparse_delta_config={"baseline_scope": "profile-1", "keep_files": True},
    )

    assert result["success"] is True
    assert captured_cfg["cfg"].backend_config == {
        "baseline_scope": "profile-1",
        "keep_files": True,
        "post_only": True,
    }


def test_sparse_delta_paths_fast_path_threads_fp8_kv_cache_metadata(monkeypatch):
    class FakeEndpointManager:
        def __init__(self, endpoints):
            self.endpoints = endpoints

        def health_check(self):
            return None

        def pause(self, mode):
            return [{"success": True}], True

        def resume(self):
            return []

    class FakeBackend:
        endpoint_results = [
            {
                "host": "infer.example",
                "port": 30000,
                "success": True,
                "message": "ok",
                "cache_version": "epoch-5",
                "fp8_kv_cache_postprocess_ran": True,
                "fp8_kv_cache_static_scales_updated": True,
            }
        ]

        def initialize(self):
            return True

        def post_packed_delta_paths(self, paths, *, flush_cache=False, weight_version=None):
            return None

        def stats_summary(self):
            return {"total_packed_bytes": 5.0}

        def destroy(self, *, complete_receiver=True):
            return None

    captured_cfg = {}

    def fake_create_backend(method, cfg):
        captured_cfg["method"] = method
        captured_cfg["cfg"] = cfg
        return FakeBackend()

    monkeypatch.setattr("xorl.server.weight_sync.handler.EndpointManager", FakeEndpointManager)
    monkeypatch.setattr("xorl.server.weight_sync.handler.create_backend", fake_create_backend)

    handler = WeightSyncHandler(rank=0, world_size=8, trainer=None)
    result = handler._sync_sparse_delta_paths(
        endpoints=[{"host": "infer.example", "port": 30000, "world_size": 8}],
        group_name="weight_sync_group",
        flush_cache=True,
        pause_mode="retract",
        weight_version="fast-1",
        sparse_delta_paths=["/shared/delta.packed"],
        fp8_kv_cache_enabled=True,
        fp8_kv_cache_postprocess_required=True,
        fp8_kv_cache_static_scales=True,
    )

    assert result["success"] is True
    assert captured_cfg["method"] == "sparse_delta"
    assert captured_cfg["cfg"].backend_config == {
        "post_only": True,
        "fp8_kv_cache_enabled": True,
        "fp8_kv_cache_postprocess_required": True,
        "run_post_process_weights": True,
        "fp8_kv_cache_static_scales": True,
    }
    assert result["cache_epoch"] == "epoch-5"
    assert result["endpoint_results"] == [
        {
            "host": "infer.example",
            "port": 30000,
            "success": True,
            "message": "ok",
            "cache_epoch": "epoch-5",
            "fp8_kv_cache_postprocess_ran": True,
            "fp8_kv_cache_static_scales_updated": True,
        }
    ]


def test_sync_weights_data_preserves_sparse_delta_paths():
    payload = SyncWeightsData(
        sparse_delta_paths=["/shared/a.packed"],
        sparse_delta_config={"prime_baseline": True},
    )

    assert payload.sparse_delta_paths == ["/shared/a.packed"]
    assert payload.sparse_delta_config == {"prime_baseline": True}


def test_sparse_delta_prepacked_only_rejects_dense_streaming_path():
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)

    result = asyncio.run(
        handler.handle_sync_inference_weights(
            {
                "payload": SyncWeightsData(
                    sync_method="sparse_delta",
                    sparse_delta_config={"prepacked_only": True},
                )
            }
        )
    )

    assert result["success"] is False
    assert "prepacked_only requires sparse_delta_paths" in result["message"]


def test_sparse_delta_fp8_quantization_targets_cpu():
    backend = type("SparseDeltaTransportBackend", (), {})()

    assert WeightSyncHandler._fp8_quantization_target_device(backend) == "cpu"


def test_unfuse_for_inference_strips_compile_orig_mod_before_qwen_linear_attention_fusion():
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["linear_attention"],
    )
    model = SimpleNamespace(config=config)
    q = torch.ones(2, 4, dtype=torch.bfloat16)
    k = torch.full((2, 4), 2, dtype=torch.bfloat16)
    v = torch.full((2, 4), 3, dtype=torch.bfloat16)
    g = torch.full((2, 4), 4, dtype=torch.bfloat16)
    a_log = torch.arange(4, dtype=torch.float32)

    remapped = dict(
        WeightSyncHandler._unfuse_for_inference(
            [
                ("model.layers.0._orig_mod.linear_attn.q_proj.weight", q),
                ("model.layers.0._orig_mod.linear_attn.k_proj.weight", k),
                ("model.layers.0._orig_mod.linear_attn.v_proj.weight", v),
                ("model.layers.0._orig_mod.linear_attn.g_proj.weight", g),
                ("model.layers.0._orig_mod.linear_attn.A_log", a_log),
            ],
            model,
        )
    )

    assert "model.layers.0._orig_mod.linear_attn.q_proj.weight" not in remapped
    torch.testing.assert_close(
        remapped["model.layers.0.linear_attn.in_proj_qkv.weight"],
        torch.cat([q, k, v], dim=0),
    )
    torch.testing.assert_close(remapped["model.layers.0.linear_attn.in_proj_z.weight"], g)
    torch.testing.assert_close(remapped["model.layers.0.linear_attn.A_log"], a_log)

    quantized = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            list(remapped.items()),
            quantization_config={
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [2, 2],
            },
        )
    )
    assert "model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv" in quantized
    assert "model.layers.0._orig_mod.linear_attn.q_proj.weight_scale_inv" not in quantized
