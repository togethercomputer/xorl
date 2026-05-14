import pytest
import torch

from xorl.server.weight_sync.handler import WeightSyncHandler


def test_fp8_quantization_emits_cpu_weight_and_scale_tensors():
    name = "model.layers.0.mlp.gate_proj.weight"
    tensor = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [2, 4],
            },
            target_device="cpu",
        )
    )

    assert set(out) == {name, "model.layers.0.mlp.gate_proj.weight_scale_inv"}
    quantized = out[name]
    scale = out["model.layers.0.mlp.gate_proj.weight_scale_inv"]
    assert quantized.device.type == "cpu"
    assert scale.device.type == "cpu"
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert quantized.shape == (4, 8)
    assert scale.shape == (2, 2)
    assert torch.all(scale > 0)


def test_fp8_quantization_skips_default_non_projection_weights():
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [("model.embed_tokens.weight", tensor)],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
    )

    assert out == [("model.embed_tokens.weight", tensor)]


def test_fp8_quantization_includes_fused_mla_weight_by_default():
    name = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
        )
    )

    assert set(out) == {name, "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"}
    assert out[name].dtype == torch.float8_e4m3fn


def test_fp8_quantization_respects_modules_to_not_convert():
    name = "model.layers.0.mlp.gate_proj.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor)],
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [2, 4],
            "modules_to_not_convert": ["model.layers.0.mlp.gate_proj"],
        },
    )

    assert out == [(name, tensor)]


def test_fp8_quantization_can_detect_contiguous_expert_slice_groups():
    stack = torch.zeros(3, 4, 8, dtype=torch.bfloat16)

    assert WeightSyncHandler._can_group_fp8_tensor(stack[0], stack[1], 1)
    assert WeightSyncHandler._can_group_fp8_tensor(stack[0], stack[2], 2)
    assert not WeightSyncHandler._can_group_fp8_tensor(stack[0], stack[2], 1)


def test_fp8_stack_quantization_matches_single_tensor_quantization():
    stack = torch.arange(3 * 4 * 8, dtype=torch.bfloat16).reshape(3, 4, 8)
    kwargs = {
        "fp8_dtype": torch.float8_e4m3fn,
        "fp8_max": torch.finfo(torch.float8_e4m3fn).max,
        "block_size_row": 2,
        "block_size_col": 4,
        "target_device": "cpu",
        "phase_s": {},
        "phase_prefix": "test_fp8",
    }

    quantized_stack, scale_stack = WeightSyncHandler._quantize_fp8_stack(stack, **kwargs)

    for idx in range(stack.shape[0]):
        quantized, scale = WeightSyncHandler._quantize_single_fp8_tensor(stack[idx], **kwargs)
        assert torch.equal(quantized_stack[idx].float(), quantized.float())
        assert torch.equal(scale_stack[idx], scale)


def test_fp8_quantization_skips_already_quantized_weights():
    name = "model.layers.0.mlp.gate_proj.weight"
    tensor = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)

    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor)],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
    )

    assert out == [(name, tensor)]


def test_fp8_cpu_expert_projection_quantization_emits_hf_weights_and_scales():
    local_data = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)
    phase_s = {}

    out, original_bytes = WeightSyncHandler._quantize_ep_expert_projection_for_fp8_cpu(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=1,
        quantization_config={"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [2, 4]},
        phase_s=phase_s,
    )
    out_by_name = dict(out)

    assert original_bytes == local_data.numel() * local_data.element_size()
    assert set(out_by_name) == {
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale_inv",
    }
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].shape == (8, 4)
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].dtype == torch.float8_e4m3fn
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight_scale_inv"].shape == (4, 1)
    assert phase_s["direct_ep_fp8_cpu_transpose_s"] >= 0


def test_fp8_cpu_expert_projection_can_defer_quantization():
    local_data = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)
    phase_s = {}

    out, original_bytes = WeightSyncHandler._format_ep_expert_projection_for_fp8_cpu(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=1,
        phase_s=phase_s,
    )
    out_by_name = dict(out)

    assert original_bytes == local_data.numel() * local_data.element_size()
    assert set(out_by_name) == {
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
    }
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].shape == (8, 4)
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].dtype == torch.bfloat16
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].device.type == "cpu"
    assert phase_s["direct_ep_fp8_source_copy_s"] >= 0
    assert phase_s["direct_ep_fp8_cpu_transpose_s"] >= 0


def test_fp8_cpu_workspace_stages_quantizes_and_reuses_storage(monkeypatch):
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE", "1")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_PINNED", "0")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_MIN_CAPACITY", "2")
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    local_data = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)
    phase_s = {}
    quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [2, 4]}

    records, original_bytes = handler._stage_ep_expert_projection_for_fp8_cpu_workspace(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=1,
        quantization_config=quantization_config,
        phase_s=phase_s,
    )

    assert original_bytes == local_data.numel() * local_data.element_size()
    assert [name for name, _, _ in records] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
    ]
    workspace = handler._fp8_cpu_workspaces[records[0][1]]
    assert torch.equal(workspace["input"][:2], local_data.permute(0, 2, 1).contiguous())
    input_ptr = workspace["input"].data_ptr()

    out = handler._quantize_fp8_cpu_workspace_records(
        records,
        quantization_config=quantization_config,
        phase_s=phase_s,
        phase_prefix="test_fp8",
    )
    assert [name for name, _ in out] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight_scale_inv",
    ]
    out_by_name = dict(out)
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].shape == (8, 4)
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight"].dtype == torch.float8_e4m3fn
    assert out_by_name["model.layers.0.mlp.experts.2.gate_proj.weight_scale_inv"].shape == (4, 1)
    assert phase_s["direct_ep_fp8_workspace_alloc_s"] >= 0
    assert phase_s["direct_ep_fp8_workspace_copy_s"] >= 0
    assert phase_s["test_fp8_float_s"] >= 0
    assert phase_s["test_fp8_reduce_s"] >= 0
    assert phase_s["test_fp8_cast_s"] >= 0

    handler._reset_fp8_cpu_workspace_usage()
    records, _ = handler._stage_ep_expert_projection_for_fp8_cpu_workspace(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=1,
        quantization_config=quantization_config,
        phase_s=phase_s,
    )
    assert handler._fp8_cpu_workspaces[records[0][1]]["input"].data_ptr() == input_ptr


def test_fp8_cpu_workspace_streams_quantized_chunks(monkeypatch):
    class RecordingBackend:
        def __init__(self):
            self.calls = []

        def transfer_bucket(self, bucket, *, src_rank=0, flush_cache=False, weight_version=None):
            self.calls.append(
                {
                    "names": [name for name, _ in bucket],
                    "dtypes": [tensor.dtype for _, tensor in bucket],
                    "src_rank": src_rank,
                    "flush_cache": flush_cache,
                    "weight_version": weight_version,
                }
            )

    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE", "1")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_PINNED", "0")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_MIN_CAPACITY", "4")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_STREAMING", "1")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_STREAM_BYTES", "96")
    handler = WeightSyncHandler(rank=3, world_size=4, trainer=None)
    backend = RecordingBackend()
    local_data = torch.arange(4 * 4 * 8, dtype=torch.bfloat16).reshape(4, 4, 8)
    phase_s = {}
    quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [2, 4]}

    records, _ = handler._stage_ep_expert_projection_for_fp8_cpu_workspace(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=0,
        quantization_config=quantization_config,
        phase_s=phase_s,
    )

    num_buckets = handler._quantize_and_transfer_fp8_cpu_workspace_records(
        backend,
        records,
        quantization_config=quantization_config,
        bucket_size_bytes=96,
        flush_cache=True,
        weight_version="sync-1",
        phase_s=phase_s,
        phase_prefix="test_fp8",
    )

    assert num_buckets == 2
    assert len(backend.calls) == 2
    assert backend.calls[0]["src_rank"] == 3
    assert backend.calls[0]["flush_cache"] is False
    assert backend.calls[0]["weight_version"] is None
    assert backend.calls[1]["flush_cache"] is True
    assert backend.calls[1]["weight_version"] == "sync-1"
    assert backend.calls[0]["names"] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
    ]
    assert backend.calls[1]["dtypes"] == [
        torch.float8_e4m3fn,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float32,
    ]
    assert phase_s["test_fp8_float_s"] >= 0
    assert phase_s["test_fp8_reduce_s"] >= 0
    assert phase_s["test_fp8_cast_s"] >= 0
    assert phase_s["direct_ep_backend_s"] >= 0
    assert phase_s["direct_ep_fp8_workspace_stream_wait_s"] >= 0


def test_fp8_cpu_workspace_flush_resets_used_capacity(monkeypatch):
    class RecordingBackend:
        def __init__(self):
            self.calls = []

        def transfer_bucket(self, bucket, *, src_rank=0, flush_cache=False, weight_version=None):
            self.calls.append(
                {
                    "names": [name for name, _ in bucket],
                    "flush_cache": flush_cache,
                    "weight_version": weight_version,
                }
            )

    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE", "1")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_PINNED", "0")
    monkeypatch.setenv("XORL_P2P_FP8_CPU_WORKSPACE_MIN_CAPACITY", "2")
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    backend = RecordingBackend()
    quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [2, 4]}
    local_data = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)
    phase_s = {}

    records, original_bytes = handler._stage_ep_expert_projection_for_fp8_cpu_workspace(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=0,
        quantization_config=quantization_config,
        phase_s=phase_s,
    )
    handler._pending_moe_cpu_workspace_records.extend(records)
    handler._pending_moe_bucket_bytes += original_bytes
    workspace = handler._fp8_cpu_workspaces[records[0][1]]
    input_ptr = workspace["input"].data_ptr()

    _, _, num_buckets = handler._flush_pending_moe_bucket(
        backend,
        flush_cache=False,
        weight_version=None,
        quantization=quantization_config,
        bucket_size_bytes=1024,
        phase_s=phase_s,
    )

    assert num_buckets == 1
    assert backend.calls[0]["flush_cache"] is False
    assert backend.calls[0]["weight_version"] is None
    assert handler._pending_moe_cpu_workspace_records == []
    assert handler._pending_moe_bucket_bytes == 0
    assert workspace["used"] == 0

    records, _ = handler._stage_ep_expert_projection_for_fp8_cpu_workspace(
        local_data,
        full_prefix="model.layers.1.mlp.experts",
        proj_name="gate_proj",
        ep_rank=0,
        quantization_config=quantization_config,
        phase_s=phase_s,
    )
    assert handler._fp8_cpu_workspaces[records[0][1]]["input"].data_ptr() == input_ptr
    assert [index for _, _, index in records] == [0, 1]


def test_empty_moe_final_flush_preserves_p2p_completion_metadata():
    class Config:
        def __init__(self):
            self.backend_config = {}

    class Backend:
        def __init__(self):
            self.config = Config()

    handler = WeightSyncHandler(rank=0, world_size=1, trainer=None)
    backend = Backend()

    _, _, num_buckets = handler._flush_pending_moe_bucket(
        backend,
        flush_cache=True,
        weight_version="sync-2",
        quantization={"quant_method": "fp8"},
        bucket_size_bytes=1024,
        phase_s={},
    )

    assert num_buckets == 0
    assert backend.config.backend_config["flush_cache"] is True
    assert backend.config.backend_config["weight_version"] == "sync-2"


def test_fp8_cpu_expert_projection_respects_modules_to_not_convert():
    local_data = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).reshape(2, 4, 8)

    out, _ = WeightSyncHandler._quantize_ep_expert_projection_for_fp8_cpu(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=0,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [2, 4],
            "modules_to_not_convert": ["model.layers.0.mlp.experts"],
        },
        phase_s={},
    )
    out_by_name = dict(out)

    assert set(out_by_name) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
    }
    assert out_by_name["model.layers.0.mlp.experts.0.gate_proj.weight"].dtype == torch.bfloat16
    assert out_by_name["model.layers.0.mlp.experts.0.gate_proj.weight"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_gpu_stack_quantization_returns_cpu_tensors(monkeypatch):
    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "gpu")
    stack = torch.arange(2 * 128 * 128, dtype=torch.bfloat16, device="cuda").reshape(2, 128, 128)
    phase_s = {}

    quantized, scale = WeightSyncHandler._quantize_fp8_stack(
        stack,
        fp8_dtype=torch.float8_e4m3fn,
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        block_size_row=128,
        block_size_col=128,
        target_device="cpu",
        phase_s=phase_s,
        phase_prefix="test_fp8",
    )

    assert quantized.device.type == "cpu"
    assert scale.device.type == "cpu"
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1, 1)
    assert phase_s["test_fp8_gpu_quant_s"] >= 0
    assert phase_s["test_fp8_gpu_output_copy_s"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_gpu_expert_projection_respects_modules_to_not_convert(monkeypatch):
    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "gpu")
    local_data = torch.arange(2 * 128 * 128, dtype=torch.bfloat16, device="cuda").reshape(2, 128, 128)

    out, _ = WeightSyncHandler._quantize_ep_expert_projection_for_fp8_gpu_to_cpu(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=0,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.layers.0.mlp.experts"],
        },
        phase_s={},
    )
    out_by_name = dict(out)

    assert set(out_by_name) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
    }
    assert out_by_name["model.layers.0.mlp.experts.0.gate_proj.weight"].dtype == torch.bfloat16
    assert out_by_name["model.layers.0.mlp.experts.0.gate_proj.weight"].device.type == "cpu"
