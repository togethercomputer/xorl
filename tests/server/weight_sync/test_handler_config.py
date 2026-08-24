import asyncio
from types import SimpleNamespace

import pytest
import torch

from xorl.models.layers.moe.experts import MoEExperts
from xorl.models.transformers.nemotron_h.checkpoint_handler import NemotronHCheckpointHandler
from xorl.server.protocol.operations import SyncWeightsData
from xorl.server.weight_sync import handler as handler_mod
from xorl.server.weight_sync.handler import (
    _DEFAULT_MOE_BUCKET_BYTES,
    _DEFAULT_P2P_MOE_BUCKET_BYTES,
    WeightSyncHandler,
    _moe_bucket_size_bytes,
    _p2p_direct_ep_sender_ep_ranks,
    _p2p_direct_ep_sender_ranks,
    _select_p2p_ib_device,
    _should_collect_ep_moe_tensors,
)


def test_receiver_post_process_and_sync_quantization_policy(monkeypatch):
    _assert_receiver_post_process_respects_fp8_kv_cache_requirement(monkeypatch)
    _assert_streaming_backend_postprocess_configuration(monkeypatch)
    _assert_p2p_fp8_postprocess_requirement_detection()
    with monkeypatch.context() as case_patch:
        _assert_handle_sync_quantization_admission_and_enrichment_policy(case_patch)


def _assert_receiver_post_process_respects_fp8_kv_cache_requirement(monkeypatch):
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


def _assert_streaming_backend_postprocess_configuration(monkeypatch):
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
    assert result["endpoint_results"] == [
        {
            "host": "inference.example",
            "port": 30000,
            "success": True,
            "message": "ok",
            "cache_epoch": "epoch-8",
            "fp8_kv_cache_postprocess_ran": True,
        }
    ]
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

    _assert_bf16_sync_does_not_forward_fp8_kv_cache_knobs(monkeypatch)


def _assert_bf16_sync_does_not_forward_fp8_kv_cache_knobs(monkeypatch):
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


def _assert_handle_sync_quantization_admission_and_enrichment_policy(monkeypatch):
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)

    result = asyncio.run(
        handler.handle_sync_inference_weights(
            {"payload": SyncWeightsData(quantization={"quant_method": "compressed-tensors"})}
        )
    )

    assert result["success"] is False
    assert "INT4/compressed-tensors updates" in result["message"]

    with monkeypatch.context() as case_patch:
        _assert_handle_sync_enriches_fp8_quantization_with_training_bf16_islands(case_patch)


def _assert_handle_sync_enriches_fp8_quantization_with_training_bf16_islands(monkeypatch):
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


def test_sync_source_adapter_extraction_and_inference_layout_policy(monkeypatch, tmp_path):
    class FakeAdapterManager:
        current_adapter_id = "current-adapter"

        def __init__(self):
            self.adapters = {"policy-a", "current-adapter"}
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
    assert handler._prepare_lora_adapter_for_sync(None) == "current-adapter"
    assert trainer.adapter_manager.synced == ["policy-b", "current-adapter"]

    trainer.lora_config = {"lora_serving_mode": "separate"}
    with pytest.raises(RuntimeError, match="publishes A/B factors"):
        handler._prepare_lora_adapter_for_sync("policy-a")

    _assert_extract_params_for_sync_policy()
    _assert_unfuse_for_inference_layout_policy()
    with monkeypatch.context() as case_patch:
        _assert_bucket_sizing_and_split_policy(case_patch)
    with monkeypatch.context() as case_patch:
        _assert_p2p_direct_ep_sender_and_collection_policy(case_patch)
    _assert_p2p_trainer_transfer_policy(tmp_path, monkeypatch)
    with monkeypatch.context() as pp_patch:
        _assert_pp_nccl_transfer_buffer_protocol(pp_patch)


def _assert_p2p_trainer_transfer_policy(tmp_path, monkeypatch):
    env_names = (
        "P2P_TRAINER_IB_DEVICES_PER_RANK",
        "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP",
        "P2P_TRAINER_IB_DEVICE",
        "P2P_TRAINER_VISIBLE_GPU_INDICES",
        "SELECTED_GPU_INDICES",
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
    )
    cases = (
        (
            "global-rank map",
            {"P2P_TRAINER_IB_DEVICES_PER_RANK": "mlx5_0;mlx5_1;mlx5_2;mlx5_3", "LOCAL_RANK": "0"},
            2,
            4,
            "mlx5_2",
        ),
        (
            "per-node local-rank map",
            {"P2P_TRAINER_IB_DEVICES_PER_RANK": "mlx5_0;mlx5_1;mlx5_2;mlx5_3", "LOCAL_RANK": "1"},
            9,
            16,
            "mlx5_1",
        ),
        ("single-device fallback", {"P2P_TRAINER_IB_DEVICE": "mlx5_6"}, 3, 8, "mlx5_6"),
        (
            "selected physical GPU 3",
            {
                "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP": "0=mlx5_2,1=mlx5_3,3=mlx5_5",
                "P2P_TRAINER_VISIBLE_GPU_INDICES": "3,1,0",
                "LOCAL_RANK": "0",
            },
            0,
            8,
            "mlx5_5",
        ),
        (
            "selected physical GPU 1",
            {
                "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP": "0=mlx5_2,1=mlx5_3,3=mlx5_5",
                "P2P_TRAINER_VISIBLE_GPU_INDICES": "3,1,0",
                "LOCAL_RANK": "1",
            },
            1,
            8,
            "mlx5_3",
        ),
        (
            "numeric CUDA visibility",
            {
                "P2P_TRAINER_GPU_TO_IB_DEVICE_MAP": "0:mlx5_2;6:mlx5_6",
                "CUDA_VISIBLE_DEVICES": "6,0",
                "LOCAL_RANK": "0",
            },
            0,
            8,
            "mlx5_6",
        ),
        (
            "empty per-rank entry uses autodiscovery",
            {"P2P_TRAINER_IB_DEVICES_PER_RANK": "mlx5_0;;mlx5_2", "LOCAL_RANK": "1"},
            1,
            3,
            None,
        ),
    )

    for label, configured_env, rank, world_size, expected in cases:
        with monkeypatch.context() as case_env:
            for name in env_names:
                case_env.delenv(name, raising=False)
            for name, value in configured_env.items():
                case_env.setenv(name, value)
            assert _select_p2p_ib_device(rank=rank, world_size=world_size) == expected, label

    abort_root = tmp_path / "abort"
    abort_root.mkdir()
    _assert_sync_abort_marker_roundtrip(abort_root)
    with monkeypatch.context() as status_patch:
        _assert_p2p_transfer_status_gather_reports_peer_failure(status_patch)


def _assert_sync_abort_marker_roundtrip(tmp_path):
    class _Trainer:
        train_config = {"output_dir": str(tmp_path)}

    handler = WeightSyncHandler(rank=3, world_size=8, trainer=_Trainer())
    path = handler._sync_abort_path("weight_sync_group", "iter-1")

    handler._mark_sync_abort(path, RuntimeError("transfer failed"))
    with pytest.raises(RuntimeError, match="rank=3: transfer failed"):
        handler._raise_if_sync_aborted(path)
    handler._clear_sync_abort(path)
    handler._raise_if_sync_aborted(path)


def _assert_p2p_transfer_status_gather_reports_peer_failure(monkeypatch):
    handler = WeightSyncHandler(rank=1, world_size=2, trainer=None)
    monkeypatch.setattr(handler_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(handler_mod.dist, "is_initialized", lambda: True)

    def fake_all_gather_object(gathered, local_status):
        gathered[0] = {"rank": 0, "ok": False, "error": "RuntimeError: transfer failed"}
        gathered[1] = local_status

    monkeypatch.setattr(handler_mod.dist, "all_gather_object", fake_all_gather_object)
    assert handler._gather_p2p_transfer_statuses(None) == [
        {"rank": 0, "ok": False, "error": "RuntimeError: transfer failed"},
        {"rank": 1, "ok": True},
    ]


def _assert_pp_nccl_transfer_buffer_protocol(monkeypatch):
    def make_handler(rank):
        handler = SimpleNamespace(rank=rank)
        handler.transfer = WeightSyncHandler._pp_nccl_transfer_buffer.__get__(handler)
        return handler

    group = object()
    source_rank = 2
    monkeypatch.setattr(handler_mod.dist, "broadcast_object_list", lambda obj, src, group: None)
    monkeypatch.setattr(handler_mod.dist, "broadcast", lambda tensor, src, group: None)
    assert make_handler(source_rank).transfer([], group, source_rank, "cuda:0") is None

    def empty_metadata(obj, src, group):
        obj[0] = []

    monkeypatch.setattr(handler_mod.dist, "broadcast_object_list", empty_metadata)
    assert make_handler(0).transfer(None, group, source_rank, "cuda:0") == []

    buffer = [
        ("a.weight", torch.randn(4, 3, dtype=torch.bfloat16)),
        ("b.weight", torch.randn(2, 5, dtype=torch.bfloat16)),
        ("c.bias", torch.randn(8, dtype=torch.bfloat16)),
        ("scale", torch.tensor(1.5, dtype=torch.bfloat16)),
    ]
    captured = {}

    def capture_metadata(obj, src, group):
        captured["metadata"] = obj[0]

    def capture_tensor(tensor, src, group):
        captured["flat"] = tensor.clone()

    monkeypatch.setattr(handler_mod.dist, "broadcast_object_list", capture_metadata)
    monkeypatch.setattr(handler_mod.dist, "broadcast", capture_tensor)
    assert make_handler(source_rank).transfer(buffer, group, source_rank, "cuda:0") is None
    assert captured["metadata"] == [
        ("a.weight", [4, 3]),
        ("b.weight", [2, 5]),
        ("c.bias", [8]),
        ("scale", []),
    ]
    assert captured["flat"].shape == (31,)
    assert captured["flat"].dtype == torch.bfloat16

    def replay_metadata(obj, src, group):
        obj[0] = captured["metadata"]

    def replay_tensor(tensor, src, group):
        tensor.copy_(captured["flat"])

    monkeypatch.setattr(handler_mod.dist, "broadcast_object_list", replay_metadata)
    monkeypatch.setattr(handler_mod.dist, "broadcast", replay_tensor)
    received = make_handler(0).transfer(None, group, source_rank, "cpu")

    assert [name for name, _ in received] == [name for name, _ in buffer]
    for (_, actual), (_, expected) in zip(received, buffer, strict=True):
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


def _assert_p2p_fp8_postprocess_requirement_detection():
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "fp8"}) is True
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "block_fp8"}) is True
    assert WeightSyncHandler._p2p_requires_post_process_weights({"quant_method": "awq"}) is False
    assert WeightSyncHandler._p2p_requires_post_process_weights(None) is False


def _assert_bucket_sizing_and_split_policy(monkeypatch):
    monkeypatch.delenv("XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES", raising=False)
    assert _moe_bucket_size_bytes("nccl_broadcast") == _DEFAULT_MOE_BUCKET_BYTES
    assert _moe_bucket_size_bytes("p2p") == _DEFAULT_P2P_MOE_BUCKET_BYTES

    monkeypatch.setenv("XORL_WEIGHT_SYNC_MOE_BUCKET_BYTES", str(456 * 1024 * 1024))
    assert _moe_bucket_size_bytes("nccl_broadcast") == 456 * 1024 * 1024
    assert _moe_bucket_size_bytes("p2p") == 456 * 1024 * 1024

    items = [
        ("a", torch.zeros(4, dtype=torch.float32)),
        ("b", torch.zeros(8, dtype=torch.float32)),
        ("c", torch.zeros(4, dtype=torch.float32)),
    ]

    chunks = WeightSyncHandler._chunk_buffer_by_bytes(items, bucket_size_bytes=32)

    assert [[name for name, _ in chunk] for chunk in chunks] == [["a"], ["b"], ["c"]]
    assert WeightSyncHandler._would_exceed_bucket_cap(900, 200, 1024) is True
    assert WeightSyncHandler._would_exceed_bucket_cap(0, 2048, 1024) is False
    assert WeightSyncHandler._would_exceed_bucket_cap(800, 224, 1024) is False


def _direct_ep_parallel_state():
    return SimpleNamespace(
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


def _assert_p2p_direct_ep_sender_and_collection_policy(monkeypatch):
    ps = _direct_ep_parallel_state()
    monkeypatch.delenv("XORL_P2P_DIRECT_EP_REPLICA_STRATEGY", raising=False)
    assert _p2p_direct_ep_sender_ranks(ps, 32) == tuple(range(8))

    monkeypatch.setenv("XORL_P2P_DIRECT_EP_REPLICA_STRATEGY", "round_robin")
    sender_ranks = _p2p_direct_ep_sender_ranks(ps, 32)
    assert sender_ranks == (0, 4, 9, 13, 18, 22, 27, 31)
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

    _assert_moe_tensor_collection_skips_only_explicit_non_senders()


def _assert_moe_tensor_collection_skips_only_explicit_non_senders():
    backend = SimpleNamespace(
        supports_direct_ep_transfer=True,
        has_explicit_sender_ranks=True,
    )

    assert _should_collect_ep_moe_tensors("p2p", backend, is_sender=False) is False
    assert _should_collect_ep_moe_tensors("p2p", backend, is_sender=True) is True

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

    _assert_collect_ep_moe_data_gating_policy()


def _assert_extract_params_for_sync_policy():
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

    _assert_extract_params_tied_weight_policy()


def _assert_extract_params_tied_weight_policy():
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

    _assert_declared_tie_with_different_parameters_is_not_deferred()
    _assert_shared_storage_without_declared_tie_is_not_inferred()


def _assert_declared_tie_with_different_parameters_is_not_deferred():
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


def _assert_shared_storage_without_declared_tie_is_not_inferred():
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


def _assert_unfuse_for_inference_layout_policy():
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

    _assert_unfuse_returns_contiguous_views_for_fp8_path()
    _assert_unfuse_nemotron_h_policy()
    _assert_unfuse_splits_gated_stacked_mlp_experts()
    _assert_compile_wrapper_name_normalization_policy()


def _assert_unfuse_returns_contiguous_views_for_fp8_path():
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


def _assert_unfuse_nemotron_h_policy():
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

    _assert_nemotron_h_rejects_fused_dense_params()


def _assert_nemotron_h_rejects_fused_dense_params():
    with pytest.raises(ValueError, match="fused dense param"):
        WeightSyncHandler._unfuse_for_inference(
            [("model.layers.1.self_attn.qkv_proj.weight", torch.zeros(6, 4))],
            _nemotron_h_model(),
        )


def _assert_unfuse_splits_gated_stacked_mlp_experts():
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


def _assert_collect_ep_moe_data_gating_policy():
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

    _assert_gated_experts_keep_three_projections()


def _assert_gated_experts_keep_three_projections():
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


def test_collect_ep_moe_data_separate_mode_omits_frozen_expert_base():
    from xorl.models.layers.moe.lora import MoEExpertsLoRA

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = MoEExperts(
                num_experts=2,
                hidden_dim=3,
                intermediate_size=5,
                hidden_act="silu",
                moe_implementation="eager",
                gated=True,
            )
            self.experts = MoEExpertsLoRA.from_module(
                base,
                r=2,
                lora_alpha=2,
                target_modules=["gate_proj", "up_proj", "down_proj"],
                hybrid_shared=True,
            )

    wrapper = Wrapper()
    experts = wrapper.experts
    torch.nn.init.normal_(experts.gate_up_proj)
    torch.nn.init.normal_(experts.down_proj)
    with torch.no_grad():
        for name, parameter in experts.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.25)
    experts.exact_merged_forward = True
    experts.lora_serving_mode = "separate"

    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    contexts = handler._collect_ep_moe_data(wrapper, "(root)", None)

    # Separate mode publishes routed-expert LoRA factors separately.  The base
    # checkpoint is immutable, so no per-expert HF keys may reach SGLang's
    # online loader (whose active-LoRA wrapper nests the base FusedMoE).  The
    # skip-only context must suppress the ordinary dense extraction path while
    # producing no EP transfer context.
    assert contexts == [
        {
            "type": "frozen_active_lora_base",
            "prefix": "experts",
            "local_experts": None,
        }
    ]
    prefixes, transferable = handler._split_ep_moe_contexts_for_sync(contexts, "(root)")
    assert prefixes == {"experts"}
    assert transferable == []
    extracted = handler._extract_params_for_sync(
        wrapper,
        "(root)",
        object,
        skip_moe_prefixes=prefixes,
    )
    assert extracted == []


def test_collect_ep_moe_data_separate_mode_omits_metadata_only_context():
    from xorl.models.layers.moe.lora import MoEExpertsLoRA

    base = MoEExperts(
        num_experts=2,
        hidden_dim=3,
        intermediate_size=5,
        hidden_act="silu",
        moe_implementation="eager",
        gated=True,
    )
    wrapper = torch.nn.Module()
    wrapper.experts = MoEExpertsLoRA.from_module(
        base,
        r=2,
        lora_alpha=2,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        hybrid_shared=True,
    )
    wrapper.experts.lora_serving_mode = "separate"

    handler = WeightSyncHandler(rank=1, world_size=8, trainer=None)
    contexts = handler._collect_ep_moe_data(
        wrapper,
        "model.layers.0.mlp",
        None,
        collect_tensors=False,
    )
    assert contexts == [
        {
            "type": "frozen_active_lora_base",
            "prefix": "model.layers.0.mlp.experts",
            "local_experts": None,
        }
    ]
    prefixes, transferable = handler._split_ep_moe_contexts_for_sync(
        contexts,
        "model.layers.0.mlp",
    )
    assert prefixes == {"experts"}
    assert transferable == []


def _assert_compile_wrapper_name_normalization_policy():
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

    _assert_broadcast_buffer_strips_orig_mod_before_transfer()
    _assert_qwen_linear_attention_strips_orig_mod_before_fusion()


def _assert_broadcast_buffer_strips_orig_mod_before_transfer():
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


def _run_fake_sparse_delta_paths(monkeypatch, *, sparse_delta_config=None, fp8_kv_cache=False):
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
            self.endpoint_results = (
                [
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
                if fp8_kv_cache
                else []
            )

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
        sparse_delta_config=sparse_delta_config,
        fp8_kv_cache_enabled=fp8_kv_cache,
        fp8_kv_cache_postprocess_required=fp8_kv_cache,
        fp8_kv_cache_static_scales=fp8_kv_cache,
    )
    return result, captured_cfg, events, fake_backend


def test_sparse_delta_sync_policy(monkeypatch):
    result, captured_cfg, events, _ = _run_fake_sparse_delta_paths(monkeypatch)

    assert result["success"] is True
    assert result["total_bytes"] == 123
    assert result["num_parameters"] == 0
    assert captured_cfg["method"] == "sparse_delta"
    assert captured_cfg["cfg"].endpoints[0].world_size == 8
    assert captured_cfg["cfg"].backend_config == {"post_only": True}
    assert ("post", ["/shared/delta.packed"], True, "fast-1") in events
    assert events.index("health") < events.index(("pause", "retract")) < events.index("resume")

    result, captured_cfg, _, _ = _run_fake_sparse_delta_paths(
        monkeypatch,
        sparse_delta_config={"baseline_scope": "profile-1", "keep_files": True},
    )
    assert result["success"] is True
    assert captured_cfg["cfg"].backend_config == {
        "baseline_scope": "profile-1",
        "keep_files": True,
        "post_only": True,
    }

    result, captured_cfg, _, _ = _run_fake_sparse_delta_paths(monkeypatch, fp8_kv_cache=True)
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

    _assert_sparse_delta_prepacked_only_rejects_dense_streaming_path()
    _assert_sparse_delta_fp8_quantization_targets_cpu()


def _assert_sparse_delta_prepacked_only_rejects_dense_streaming_path():
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


def _assert_sparse_delta_fp8_quantization_targets_cpu():
    backend = type("SparseDeltaTransportBackend", (), {})()

    assert WeightSyncHandler._fp8_quantization_target_device(backend) == "cpu"


def _assert_qwen_linear_attention_strips_orig_mod_before_fusion():
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
