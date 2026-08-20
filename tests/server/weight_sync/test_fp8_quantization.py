import asyncio
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.fp8_training import enrich_sync_quantization_with_fp8_bf16_islands
from xorl.lora.fold import canonical_lora_fold_linear
from xorl.lora.modules.delta_linear import LoraDeltaLinear
from xorl.lora.modules.linear import LoraLinear
from xorl.qarl import inject_qarl_into_model, qarl_sync_quantization_config
from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.moe_experts import QLoRAMoeExperts
from xorl.server.protocol.operations import SyncWeightsData
from xorl.server.weight_sync.handler import WeightSyncHandler


def _dequantize_block_fp8_2d(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: tuple[int, int],
) -> torch.Tensor:
    row_block, col_block = block_size
    expanded = scale.float().repeat_interleave(row_block, dim=0).repeat_interleave(col_block, dim=1)
    return weight.float() * expanded[: weight.shape[0], : weight.shape[1]]


def _slime_blockwise_fp8_reference(
    weight: torch.Tensor,
    *,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for Slime's Megatron blockwise_cast_to_fp8_triton contract."""
    row_block, col_block = block_size
    work = weight.float()
    rows, cols = work.shape
    pad_rows = (row_block - rows % row_block) % row_block
    pad_cols = (col_block - cols % col_block) % col_block
    if pad_rows or pad_cols:
        padded = torch.zeros(rows + pad_rows, cols + pad_cols, dtype=torch.float32)
        padded[:rows, :cols] = work
    else:
        padded = work

    nr = padded.shape[0] // row_block
    nc = padded.shape[1] // col_block
    blocks = padded.reshape(nr, row_block, nc, col_block).permute(0, 2, 1, 3)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = blocks.abs().reshape(nr, nc, -1).max(dim=-1).values.clamp(min=1e-12) / fp8_max
    quantized_blocks = (blocks / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    quantized = quantized_blocks.permute(0, 2, 1, 3).reshape(padded.shape[0], padded.shape[1])
    return quantized[:rows, :cols].contiguous(), scale.contiguous()


def _last_element_padded_fp8_reference(
    weight: torch.Tensor,
    *,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for receiver-side last-element padding semantics that XoRL intentionally does not use."""
    row_block, col_block = block_size
    work = weight.float()
    rows, cols = work.shape
    pad_rows = (row_block - rows % row_block) % row_block
    pad_cols = (col_block - cols % col_block) % col_block
    if pad_rows or pad_cols:
        padded = torch.full(
            (rows + pad_rows, cols + pad_cols),
            work.flatten()[-1].item(),
            dtype=torch.float32,
        )
        padded[:rows, :cols] = work
    else:
        padded = work

    nr = padded.shape[0] // row_block
    nc = padded.shape[1] // col_block
    blocks = padded.reshape(nr, row_block, nc, col_block).permute(0, 2, 1, 3)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = blocks.abs().reshape(nr, nc, -1).max(dim=-1).values.clamp(min=1e-12) / fp8_max
    quantized_blocks = (blocks / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    quantized = quantized_blocks.permute(0, 2, 1, 3).reshape(padded.shape[0], padded.shape[1])
    return quantized[:rows, :cols].contiguous(), scale.contiguous()


class TinySyncIslandModel(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(num_layers)])

    def get_pp_module_config(self):
        return {"layer_prefix": "model.layers", "num_layers": self.config.num_hidden_layers}


class _TinyQARLSyncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny")
        self.embed_tokens = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_fp8_sync_quantization_policy(monkeypatch):
    model = TinySyncIslandModel(num_layers=4)
    quantization = enrich_sync_quantization_with_fp8_bf16_islands(
        model,
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [2, 4],
        },
        num_first_layers_bf16=1,
        num_last_layers_bf16=1,
    )
    assert quantization is not None
    assert quantization["modules_to_not_convert"] == ["model.layers.0.*", "model.layers.3.*"]

    first_name = "model.layers.0.mlp.gate_proj.weight"
    middle_name = "model.layers.1.mlp.gate_proj.weight"
    last_name = "model.layers.3.mlp.gate_proj.weight"
    first = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    middle = first + 1
    last = first + 2

    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(first_name, first), (middle_name, middle), (last_name, last)],
            quantization_config=quantization,
            target_device="cpu",
        )
    )

    assert out[first_name].dtype == torch.bfloat16
    assert first_name.replace(".weight", ".weight_scale_inv") not in out
    assert out[middle_name].dtype == torch.float8_e4m3fn
    assert out[middle_name.replace(".weight", ".weight_scale_inv")].dtype == torch.float32
    assert out[last_name].dtype == torch.bfloat16
    assert last_name.replace(".weight", ".weight_scale_inv") not in out

    _assert_fp8_matches_slime_blockwise_scale_contract()
    _assert_fp8_zero_padding_for_partial_layout()
    _assert_fp8_projection_selection_stack_and_existing_dtype_policy()
    with monkeypatch.context() as adapter_patch:
        _assert_fp8_adapter_merge_policy(adapter_patch)
    with monkeypatch.context() as expert_patch:
        _assert_fp8_cpu_expert_projection_and_workspace_policy(expert_patch)
    with monkeypatch.context() as qarl_patch:
        _assert_qarl_training_to_weight_sync_quantization_lifecycle(qarl_patch)


def _assert_qarl_training_to_weight_sync_quantization_lifecycle(monkeypatch):
    model = _TinyQARLSyncModel()
    inject_qarl_into_model(
        model,
        quant_cfg={"format": "fp8_e4m3", "weight_block_size": [2, 2]},
        target_modules=["proj"],
    )

    config = qarl_sync_quantization_config(model)

    assert config is not None
    assert config["quant_method"] == "fp8"
    assert config["weight_block_size"] == [2, 2]
    assert set(config["modules_to_not_convert"]) == {"embed_tokens", "lm_head"}
    assert config["xorl_qarl_sync"] == {
        "enabled": True,
        "folded_modules": ["proj"],
        "source": "qarl_fake_quant",
    }

    quantized = WeightSyncHandler._quantize_buffer_for_fp8(
        [
            ("embed_tokens.weight", model.embed_tokens.weight.detach()),
            ("proj.weight", model.proj.weight.detach()),
            ("lm_head.weight", model.lm_head.weight.detach()),
        ],
        quantization_config=config,
    )
    names = {name for name, _tensor in quantized}
    assert {"embed_tokens.weight", "proj.weight", "proj.weight_scale_inv", "lm_head.weight"} <= names
    assert "embed_tokens.weight_scale_inv" not in names
    assert "lm_head.weight_scale_inv" not in names

    trainer = SimpleNamespace(model=model, train_config={})
    handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)
    captured = {}

    def fake_sync_weights(**kwargs):
        captured["quantization"] = kwargs["quantization"]
        return {"success": True}

    monkeypatch.setattr(handler, "_sync_weights", fake_sync_weights)
    result = asyncio.run(handler.handle_sync_inference_weights({"payload": SyncWeightsData(quantization=None)}))
    assert result["success"] is True
    assert captured["quantization"]["weight_block_size"] == [2, 2]
    assert set(captured["quantization"]["modules_to_not_convert"]) == {"embed_tokens", "lm_head"}

    bad_handler = WeightSyncHandler(rank=0, world_size=1, trainer=trainer)
    result = asyncio.run(
        bad_handler.handle_sync_inference_weights(
            {"payload": SyncWeightsData(quantization={"quant_method": "fp8", "weight_block_size": [4, 4]})}
        )
    )
    assert result["success"] is False
    assert "Failed to resolve QARL sync quantization" in result["message"]


def _assert_fp8_matches_slime_blockwise_scale_contract():
    name = "model.layers.0.mlp.gate_proj.weight"
    block_size = (2, 4)
    tensor = torch.tensor(
        [
            [0.0, -1.0, 2.5, -4.0, 8.0],
            [16.0, -32.0, 64.0, -128.0, 256.0],
            [3.0, -6.0, 12.0, -24.0, 48.0],
        ],
        dtype=torch.bfloat16,
    )

    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": list(block_size),
            },
            target_device="cpu",
        )
    )
    ref_weight, ref_scale = _slime_blockwise_fp8_reference(tensor, block_size=block_size)
    scale_name = name.replace(".weight", ".weight_scale_inv")

    assert set(out) == {name, scale_name}
    assert out[name].device.type == "cpu"
    assert out[scale_name].device.type == "cpu"
    assert out[name].dtype == torch.float8_e4m3fn
    assert out[scale_name].dtype == torch.float32
    assert out[scale_name].shape == (2, 2)
    assert torch.equal(out[name].view(torch.uint8), ref_weight.view(torch.uint8))
    torch.testing.assert_close(out[scale_name], ref_scale, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        _dequantize_block_fp8_2d(out[name], out[scale_name], block_size=block_size),
        _dequantize_block_fp8_2d(ref_weight, ref_scale, block_size=block_size),
        rtol=0.0,
        atol=0.0,
    )


def _assert_fp8_zero_padding_for_partial_layout():
    name = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"
    block_size = (2, 4)
    tensor = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5, 0.25],
            [1.5, -1.5, 0.75, -0.75, -0.25],
            [2.0, -2.0, 1.0, -1.0, 1024.0],
        ],
        dtype=torch.bfloat16,
    )

    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": list(block_size),
            },
            target_device="cpu",
        )
    )
    ref_weight, ref_scale = _slime_blockwise_fp8_reference(tensor, block_size=block_size)
    last_padded_weight, last_padded_scale = _last_element_padded_fp8_reference(tensor, block_size=block_size)
    scale_name = name.replace(".weight", ".weight_scale_inv")

    torch.testing.assert_close(out[scale_name], ref_scale, rtol=0.0, atol=0.0)
    assert torch.equal(out[name].view(torch.uint8), ref_weight.view(torch.uint8))
    assert not torch.equal(out[scale_name], last_padded_scale)
    assert not torch.equal(out[name].view(torch.uint8), last_padded_weight.view(torch.uint8))


def _assert_fp8_adapter_merge_policy(monkeypatch):
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate_proj = LoraLinear(4, 4, r=2, lora_alpha=2, bias=False, dtype=torch.bfloat16)

    class FakeDTensor:
        pass

    layer = Layer()
    lora = layer.mlp.gate_proj
    with torch.no_grad():
        lora.weight.zero_()
        lora.lora_A.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0],
                ]
            )
        )
        lora.lora_B.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 2.0],
                ]
            )
        )

    weight_name = "model.layers.0.mlp.gate_proj.weight"
    scale_name = "model.layers.0.mlp.gate_proj.weight_scale_inv"
    extracted = WeightSyncHandler._extract_params_for_sync(layer, "model.layers.0", FakeDTensor)

    assert [name for name, _ in extracted] == [weight_name]
    expected_merged = lora.get_delta_weight().to(torch.bfloat16)
    torch.testing.assert_close(extracted[0][1], expected_merged, rtol=0.0, atol=0.0)

    quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [2, 2]}
    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            extracted,
            quantization_config=quantization_config,
            target_device="cpu",
        )
    )
    expected_out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(weight_name, expected_merged.clone())],
            quantization_config=quantization_config,
            target_device="cpu",
        )
    )
    base_only_out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(weight_name, lora.weight.detach().to(torch.bfloat16).clone())],
            quantization_config=quantization_config,
            target_device="cpu",
        )
    )

    assert set(out) == {weight_name, scale_name}
    assert out[weight_name].dtype == torch.float8_e4m3fn
    assert out[scale_name].dtype == torch.float32
    assert torch.equal(out[weight_name].view(torch.uint8), expected_out[weight_name].view(torch.uint8))
    torch.testing.assert_close(out[scale_name], expected_out[scale_name], rtol=0.0, atol=0.0)
    assert not torch.equal(out[weight_name].view(torch.uint8), base_only_out[weight_name].view(torch.uint8))

    dequantized = _dequantize_block_fp8_2d(out[weight_name], out[scale_name], block_size=(2, 2))
    assert dequantized.float().abs().max() > 0.0

    _assert_fp8_uses_merged_qlora_weight(monkeypatch)
    _assert_fp8_uses_merged_qlora_moe_experts(monkeypatch)
    _assert_runtime_rank_moe_lora_buffer_scaling()
    _assert_sync_extraction_folds_fused_gdn_lora_into_separate_base_projections()


def _assert_fp8_uses_merged_qlora_weight(monkeypatch):
    class FakeQLoRALinear(QLoRALinear):
        def __init__(self) -> None:
            super().__init__(4, 4, r=2, lora_alpha=2, quant_format="fake", quant_group_size=2, bias=False)
            self.packed_weight_f32 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
            self.register_buffer(
                "_base_weight",
                torch.tensor(
                    [
                        [8.0, 1.0, -2.0, 3.0],
                        [-4.0, 7.0, 5.0, -6.0],
                        [2.0, -3.0, 9.0, 4.0],
                        [6.0, 5.0, -7.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
            )
            self.reset_lora_parameters()

        def _dequantize_weight(self) -> torch.Tensor:
            return self._base_weight.clone()

        def _compute_aqn_step(self) -> torch.Tensor:
            raise NotImplementedError

        def _quantize_and_store(self, w: torch.Tensor, global_amax: torch.Tensor | None = None) -> None:
            raise NotImplementedError

        def merge_weights(self, ema_decay: float = 0.1) -> None:
            raise NotImplementedError

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate_proj = FakeQLoRALinear()

    class FakeDTensor:
        pass

    monkeypatch.setattr(
        "xorl.server.weight_sync.handler.get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=False, ep_size=1),
    )

    layer = Layer()
    qlora = layer.mlp.gate_proj
    with torch.no_grad():
        qlora.lora_A.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0],
                ]
            )
        )
        qlora.lora_B.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 2.0],
                ]
            )
        )

    # Normal parameter extraction must not send packed QLoRA internals directly.
    assert WeightSyncHandler._extract_params_for_sync(layer, "model.layers.0", FakeDTensor) == []

    handler = object.__new__(WeightSyncHandler)
    handler.rank = 0
    qlora_buffer, moe_contexts = handler._qlora_collective_ops(layer, "model.layers.0", collect_results=True)

    weight_name = "model.layers.0.mlp.gate_proj.weight"
    scale_name = "model.layers.0.mlp.gate_proj.weight_scale_inv"
    expected_merged = qlora._dequantize_weight().to(torch.bfloat16) + qlora.get_delta_weight().to(torch.bfloat16)

    assert moe_contexts == []
    assert [name for name, _ in qlora_buffer] == [weight_name]
    torch.testing.assert_close(qlora_buffer[0][1], expected_merged, rtol=0.0, atol=0.0)

    quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [2, 2]}
    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            qlora_buffer,
            quantization_config=quantization_config,
            target_device="cpu",
        )
    )
    expected_out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(weight_name, expected_merged.clone())],
            quantization_config=quantization_config,
            target_device="cpu",
        )
    )
    base_only_out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(weight_name, qlora._dequantize_weight().to(torch.bfloat16))],
            quantization_config=quantization_config,
            target_device="cpu",
        )
    )

    assert set(out) == {weight_name, scale_name}
    assert out[weight_name].dtype == torch.float8_e4m3fn
    assert out[scale_name].dtype == torch.float32
    assert torch.equal(out[weight_name].view(torch.uint8), expected_out[weight_name].view(torch.uint8))
    torch.testing.assert_close(out[scale_name], expected_out[scale_name], rtol=0.0, atol=0.0)
    assert not torch.equal(out[weight_name].view(torch.uint8), base_only_out[weight_name].view(torch.uint8))

    dequantized = _dequantize_block_fp8_2d(out[weight_name], out[scale_name], block_size=(2, 2))
    assert dequantized.float().abs().max() > 0.0


def _assert_fp8_uses_merged_qlora_moe_experts(monkeypatch):
    class FakeQLoRAMoeExperts(QLoRAMoeExperts):
        def __init__(self) -> None:
            super().__init__(
                num_local_experts=1,
                num_experts=1,
                intermediate_size=2,
                hidden_size=2,
                r=1,
                lora_alpha=1,
                quant_format="fake",
                quant_group_size=1,
            )
            self._base_by_proj = {
                "gate": torch.tensor([[2.0, -1.0], [4.0, 3.0]], dtype=torch.float32),
                "up": torch.tensor([[-3.0, 5.0], [6.0, -7.0]], dtype=torch.float32),
                "down": torch.tensor([[8.0, 1.0], [-2.0, 9.0]], dtype=torch.float32),
            }
            self.reset_lora_parameters()

        def dequantize_expert(self, proj_name: str, expert_idx: int, K: int, N: int) -> torch.Tensor:
            assert expert_idx == 0
            assert self._base_by_proj[proj_name].shape == (K, N)
            return self._base_by_proj[proj_name].clone()

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = FakeQLoRAMoeExperts()

    class FakeDTensor:
        pass

    monkeypatch.setattr(
        "xorl.server.weight_sync.handler.get_parallel_state",
        lambda: SimpleNamespace(ep_enabled=False, ep_size=1),
    )

    layer = Layer()
    experts = layer.mlp.experts
    with torch.no_grad():
        experts.gate_proj_lora_A.copy_(torch.tensor([[[1.0], [2.0]]]))
        experts.gate_proj_lora_B.copy_(torch.tensor([[[3.0, 4.0]]]))
        experts.up_proj_lora_A.copy_(torch.tensor([[[2.0], [-1.0]]]))
        experts.up_proj_lora_B.copy_(torch.tensor([[[-2.0, 5.0]]]))
        experts.down_proj_lora_A.copy_(torch.tensor([[[4.0], [1.0]]]))
        experts.down_proj_lora_B.copy_(torch.tensor([[[2.0, -3.0]]]))

    # Normal parameter extraction must not send QLoRA MoE LoRA/internal tensors directly.
    assert WeightSyncHandler._extract_params_for_sync(layer, "model.layers.0", FakeDTensor) == []

    handler = object.__new__(WeightSyncHandler)
    handler.rank = 0
    qlora_buffer, moe_contexts = handler._qlora_collective_ops(layer, "model.layers.0", collect_results=True)

    assert qlora_buffer == []
    assert len(moe_contexts) == 1
    assert set(moe_contexts[0]["lora_params"]) == {
        "gate_proj_lora_A",
        "gate_proj_lora_B",
        "up_proj_lora_A",
        "up_proj_lora_B",
        "down_proj_lora_A",
        "down_proj_lora_B",
    }

    items = handler._compute_moe_experts_buffer(moe_contexts[0])
    item_map = dict(items)
    expected_by_name = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": (
            experts._base_by_proj["gate"]
            + WeightSyncHandler._compute_moe_lora_delta(
                experts, experts.gate_proj_lora_A, experts.gate_proj_lora_B, expert_idx=0
            )
        )
        .to(torch.bfloat16)
        .t()
        .contiguous(),
        "model.layers.0.mlp.experts.0.up_proj.weight": (
            experts._base_by_proj["up"]
            + WeightSyncHandler._compute_moe_lora_delta(
                experts, experts.up_proj_lora_A, experts.up_proj_lora_B, expert_idx=0
            )
        )
        .to(torch.bfloat16)
        .t()
        .contiguous(),
        "model.layers.0.mlp.experts.0.down_proj.weight": (
            experts._base_by_proj["down"]
            + WeightSyncHandler._compute_moe_lora_delta(
                experts, experts.down_proj_lora_A, experts.down_proj_lora_B, expert_idx=0
            )
        )
        .to(torch.bfloat16)
        .t()
        .contiguous(),
    }

    assert set(item_map) == set(expected_by_name)
    assert moe_contexts[0]["lora_params"] is None
    for name, expected in expected_by_name.items():
        torch.testing.assert_close(item_map[name], expected, rtol=0.0, atol=0.0)

    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            items,
            quantization_config={"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [1, 1]},
            target_device="cpu",
        )
    )
    assert all("lora" not in name and "packed" not in name for name in out)
    for weight_name in expected_by_name:
        scale_name = weight_name.replace(".weight", ".weight_scale_inv")
        assert out[weight_name].dtype == torch.float8_e4m3fn
        assert out[scale_name].dtype == torch.float32
        dequantized = _dequantize_block_fp8_2d(out[weight_name], out[scale_name], block_size=(1, 1))
        assert dequantized.float().abs().max() > 0.0


def _assert_runtime_rank_moe_lora_buffer_scaling():
    class FakeRuntimeRankMoeModule:
        num_local_experts = 1
        hidden_size = 1
        intermediate_size = 1
        active_r = 2
        scaling = 4.0

        def _active_scaling(self) -> float:
            return 2.0

        def dequantize_expert(self, _proj_name: str, _expert_idx: int, K: int, N: int) -> torch.Tensor:
            return torch.zeros((K, N), dtype=torch.float32)

    lora_A = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    lora_B = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]], dtype=torch.float32)
    lora_params = {
        f"{projection}_{factor}": tensor.clone()
        for projection in ("gate_proj", "up_proj", "down_proj")
        for factor, tensor in (("lora_A", lora_A), ("lora_B", lora_B))
    }
    ctx = {
        "module": FakeRuntimeRankMoeModule(),
        "prefix": "model.layers.0.mlp.experts",
        "lora_params": lora_params,
    }
    handler = object.__new__(WeightSyncHandler)
    handler.rank = 0

    items = handler._compute_moe_experts_buffer(ctx)

    assert [name for name, _ in items] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ]
    for _, tensor in items:
        assert tensor.dtype == torch.bfloat16
        assert tensor.shape == (1, 1)
        assert tensor.float().item() == pytest.approx(4.0)
    assert ctx["lora_params"] is None


def _assert_fp8_projection_selection_stack_and_existing_dtype_policy():
    quantized_names = [
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
        *(
            f"model.layers.0.linear_attn.{projection}.weight"
            for projection in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        ),
        *(
            f"{prefix}.{projection}.weight"
            for prefix in ("model.layers.0.mlp.shared_expert", "model.layers.0.mlp.shared_experts")
            for projection in ("gate_proj", "up_proj", "down_proj")
        ),
    ]
    passthrough_names = [
        "model.embed_tokens.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ]
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)

    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor) for name in (*quantized_names, *passthrough_names)],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
        target_device="cpu",
    )

    out_by_name = dict(out)
    assert set(out_by_name) == (
        set(quantized_names)
        | {name.replace(".weight", ".weight_scale_inv") for name in quantized_names}
        | set(passthrough_names)
    )
    for name in quantized_names:
        assert out_by_name[name].dtype == torch.float8_e4m3fn
        assert out_by_name[name.replace(".weight", ".weight_scale_inv")].dtype == torch.float32
    for name in passthrough_names:
        assert out_by_name[name] is tensor

    _assert_fp8_respects_modules_to_not_convert()
    _assert_fp8_receiver_skip_list_preserves_passthrough_entries()
    _assert_fp8_receiver_skip_list_enables_broad_selector()
    _assert_fp8_stack_and_existing_dtype_policy()


def _assert_fp8_respects_modules_to_not_convert():
    name = "model.layers.0.mlp.gate_proj.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    for excluded_name in ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj.weight"):
        out = WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={
                "quant_method": "fp8",
                "weight_block_size": [2, 4],
                "modules_to_not_convert": [excluded_name],
            },
        )
        assert out == [(name, tensor)], excluded_name


def _assert_fp8_receiver_skip_list_preserves_passthrough_entries():
    entries = [
        ("model.embed_tokens.weight", torch.zeros(16, 8, dtype=torch.bfloat16)),
        ("model.layers.0.input_layernorm.weight", torch.ones(8, dtype=torch.bfloat16)),
        ("model.layers.0.self_attn.o_proj.weight", torch.zeros(8, 8, dtype=torch.bfloat16)),
        ("model.layers.0.mlp.gate.weight", torch.zeros(4, 8, dtype=torch.bfloat16)),
        ("model.layers.0.linear_attn.dt_bias", torch.zeros(8, dtype=torch.float32)),
        ("lm_head.weight", torch.zeros(16, 8, dtype=torch.bfloat16)),
    ]

    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            entries,
            quantization_config={
                "quant_method": "fp8",
                "weight_block_size": [2, 4],
                "modules_to_not_convert": [
                    "lm_head",
                    "model.embed_tokens",
                    "model.layers.0.input_layernorm",
                    "model.layers.0.mlp.gate",
                ],
            },
        )
    )

    quantized_name = "model.layers.0.self_attn.o_proj.weight"
    assert set(out) == {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        quantized_name,
        quantized_name.replace(".weight", ".weight_scale_inv"),
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.linear_attn.dt_bias",
        "lm_head.weight",
    }
    assert out[quantized_name].dtype == torch.float8_e4m3fn
    assert out[quantized_name.replace(".weight", ".weight_scale_inv")].dtype == torch.float32
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.linear_attn.dt_bias",
        "lm_head.weight",
    ):
        assert out[name] is dict(entries)[name]


def _assert_fp8_receiver_skip_list_enables_broad_selector():
    name = "model.layers.0.custom_dense.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)

    default_out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor)],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
    )
    skip_list_out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={
                "quant_method": "fp8",
                "weight_block_size": [2, 4],
                "modules_to_not_convert": ["lm_head"],
            },
        )
    )

    assert default_out == [(name, tensor)]
    assert set(skip_list_out) == {name, "model.layers.0.custom_dense.weight_scale_inv"}
    assert skip_list_out[name].dtype == torch.float8_e4m3fn


def _assert_fp8_stack_and_existing_dtype_policy():
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

    _assert_fp8_skips_already_quantized_weights()


def _assert_fp8_skips_already_quantized_weights():
    name = "model.layers.0.mlp.gate_proj.weight"
    tensor = torch.zeros(4, 8, dtype=torch.float8_e4m3fn)

    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor)],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
    )

    assert out == [(name, tensor)]


def _assert_fp8_cpu_expert_projection_and_workspace_policy(monkeypatch):
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

    _assert_fp8_cpu_expert_zero_padding()
    _assert_fp8_cpu_expert_can_defer_quantization()
    _assert_fp8_cpu_expert_respects_modules_to_not_convert()
    with monkeypatch.context() as case_patch:
        _assert_fp8_cpu_workspace_lifecycle_policy(case_patch)


def _assert_fp8_cpu_expert_zero_padding():
    block_size = (4, 4)
    hf_weight = torch.tensor(
        [
            [0.125, 0.25, -0.5],
            [1.0, -1.5, 2.0],
            [0.75, -1.0, 1.25],
            [2.5, -2.0, 1.0],
            [3.0, 4.0, 1024.0],
        ],
        dtype=torch.bfloat16,
    )
    local_data = hf_weight.t().contiguous().unsqueeze(0)

    out, _ = WeightSyncHandler._quantize_ep_expert_projection_for_fp8_cpu(
        local_data,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=0,
        quantization_config={"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": list(block_size)},
        phase_s={},
    )
    out_by_name = dict(out)
    weight_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    scale_name = weight_name.replace(".weight", ".weight_scale_inv")
    ref_weight, ref_scale = _slime_blockwise_fp8_reference(hf_weight, block_size=block_size)
    last_padded_weight, last_padded_scale = _last_element_padded_fp8_reference(hf_weight, block_size=block_size)

    torch.testing.assert_close(out_by_name[scale_name], ref_scale, rtol=0.0, atol=0.0)
    assert torch.equal(out_by_name[weight_name].view(torch.uint8), ref_weight.view(torch.uint8))
    assert not torch.equal(out_by_name[scale_name], last_padded_scale)
    assert not torch.equal(out_by_name[weight_name].view(torch.uint8), last_padded_weight.view(torch.uint8))


def _assert_fp8_cpu_expert_can_defer_quantization():
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


def _assert_fp8_cpu_workspace_lifecycle_policy(monkeypatch):
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

    _assert_fp8_cpu_workspace_streams_quantized_chunks(monkeypatch)
    _assert_fp8_cpu_workspace_flush_resets_capacity(monkeypatch)
    _assert_empty_moe_flush_preserves_completion_metadata()


def _assert_fp8_cpu_workspace_streams_quantized_chunks(monkeypatch):
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


def _assert_fp8_cpu_workspace_flush_resets_capacity(monkeypatch):
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

    monkeypatch.delenv("XORL_P2P_FP8_CPU_WORKSPACE_STREAMING", raising=False)
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


def _assert_empty_moe_flush_preserves_completion_metadata():
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


def _assert_fp8_cpu_expert_respects_modules_to_not_convert():
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


def _assert_sync_extraction_folds_fused_gdn_lora_into_separate_base_projections():
    # River fuses one LoRA on GDN in_proj_qkvz (out = q|k|v|z contiguous) and one on
    # out_proj. The weight sync must FOLD those deltas into the trainer's SEPARATE
    # base q/k/v/g/o projections (contiguous row slices) and NOT ship the raw
    # lora_A/lora_B (a pure-base sampler has no receiver slot for them).
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = nn.Module()
            self.linear_attn.q_proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
            self.linear_attn.k_proj = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
            self.linear_attn.v_proj = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
            self.linear_attn.g_proj = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
            self.linear_attn.o_proj = nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
            self.linear_attn.in_proj_qkvz = LoraDeltaLinear(4, 10, r=2, lora_alpha=2, dtype=torch.bfloat16)
            self.linear_attn.out_proj = LoraDeltaLinear(3, 4, r=2, lora_alpha=2, dtype=torch.bfloat16)

    class FakeDTensor:
        pass

    layer = Layer()
    with torch.no_grad():
        for projection in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj"):
            getattr(layer.linear_attn, projection).weight.zero_()
        layer.linear_attn.in_proj_qkvz.lora_A.copy_(
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.bfloat16)
        )
        layer.linear_attn.in_proj_qkvz.lora_B.copy_(
            torch.arange(1, 21, dtype=torch.float32).reshape(10, 2).to(torch.bfloat16)
        )
        layer.linear_attn.out_proj.lora_A.copy_(torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.bfloat16))
        layer.linear_attn.out_proj.lora_B.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]], dtype=torch.bfloat16)
        )

    extracted = dict(WeightSyncHandler._extract_params_for_sync(layer, "model.layers.0", FakeDTensor))
    input_delta = layer.linear_attn.in_proj_qkvz.get_delta_weight().to(torch.bfloat16)
    output_delta = layer.linear_attn.out_proj.get_delta_weight().to(torch.bfloat16)
    # in_proj_qkvz out=10: q[0:2] k[2:4] v[4:7] g[7:10] (contiguous q|k|v|z), out_proj=o
    slices = {
        "q_proj": input_delta[0:2],
        "k_proj": input_delta[2:4],
        "v_proj": input_delta[4:7],
        "g_proj": input_delta[7:10],
        "o_proj": output_delta,
    }
    assert set(extracted) == {f"model.layers.0.linear_attn.{projection}.weight" for projection in slices}
    # raw fused-GDN factors must NOT be shipped
    for raw in ("in_proj_qkvz.lora_A", "in_proj_qkvz.lora_B", "out_proj.lora_A", "out_proj.lora_B"):
        assert f"model.layers.0.linear_attn.{raw}" not in extracted
    for projection, expected in slices.items():
        torch.testing.assert_close(
            extracted[f"model.layers.0.linear_attn.{projection}.weight"], expected, rtol=0.0, atol=0.0
        )


def test_sync_extraction_folds_qwen_shared_expert_gate_up_adapters():
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.shared_expert = nn.Module()
            shared = self.mlp.shared_expert
            shared.gate_up_proj = nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
            shared.gate_proj = LoraDeltaLinear(4, 3, r=2, lora_alpha=2)
            shared.up_proj = LoraDeltaLinear(4, 3, r=2, lora_alpha=2)

    class FakeDTensor:
        pass

    layer = Layer()
    shared = layer.mlp.shared_expert
    with torch.no_grad():
        shared.gate_up_proj.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16))
        for offset, adapter in enumerate((shared.gate_proj, shared.up_proj), start=1):
            adapter.lora_A.copy_(torch.arange(1, 9, dtype=torch.float32).reshape(2, 4) * offset)
            adapter.lora_B.copy_(torch.arange(1, 7, dtype=torch.float32).reshape(3, 2) / offset)
            adapter.exact_merged_forward = True

    gate_base, up_base = shared.gate_up_proj.weight.chunk(2, dim=0)
    expected = torch.cat(
        (
            canonical_lora_fold_linear(gate_base, shared.gate_proj.lora_A, shared.gate_proj.lora_B, 1.0),
            canonical_lora_fold_linear(up_base, shared.up_proj.lora_A, shared.up_proj.lora_B, 1.0),
        ),
        dim=0,
    )
    extracted = dict(WeightSyncHandler._extract_params_for_sync(layer, "model.layers.0", FakeDTensor))
    weight_name = "model.layers.0.mlp.shared_expert.gate_up_proj.weight"
    assert set(extracted) == {weight_name}
    assert torch.equal(extracted[weight_name], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_gpu_quantization_policy(monkeypatch):
    base = torch.linspace(-7.5, 7.5, steps=2 * 128 * 256, dtype=torch.float32).reshape(2, 128, 256)
    stack = base.to(torch.bfloat16).cuda()
    common_kwargs = {
        "fp8_dtype": torch.float8_e4m3fn,
        "fp8_max": torch.finfo(torch.float8_e4m3fn).max,
        "block_size_row": 128,
        "block_size_col": 128,
        "phase_prefix": "test_fp8",
    }

    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "gpu")
    gpu_cpu_phase = {}
    gpu_quantized, gpu_scale = WeightSyncHandler._quantize_fp8_stack(
        stack,
        target_device="cpu",
        phase_s=gpu_cpu_phase,
        **common_kwargs,
    )
    assert gpu_quantized.device.type == "cpu"
    assert gpu_scale.device.type == "cpu"
    assert gpu_quantized.dtype == torch.float8_e4m3fn
    assert gpu_cpu_phase["test_fp8_gpu_quant_s"] >= 0
    assert gpu_cpu_phase["test_fp8_gpu_output_copy_s"] >= 0

    gpu_cuda_phase = {}
    cuda_quantized, cuda_scale = WeightSyncHandler._quantize_fp8_stack(
        stack,
        target_device="cuda",
        phase_s=gpu_cuda_phase,
        **common_kwargs,
    )
    assert cuda_quantized.device.type == "cuda"
    assert cuda_scale.device.type == "cuda"
    assert gpu_cuda_phase["test_fp8_gpu_quant_s"] >= 0
    assert "test_fp8_gpu_output_copy_s" not in gpu_cuda_phase

    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "cpu")
    cpu_quantized, cpu_scale = WeightSyncHandler._quantize_fp8_stack(
        stack.cpu(),
        target_device="cpu",
        phase_s={},
        **common_kwargs,
    )

    torch.testing.assert_close(gpu_scale, cpu_scale, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gpu_quantized.float(), cpu_quantized.float(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(cuda_scale.cpu(), cpu_scale, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(cuda_quantized.float().cpu(), cpu_quantized.float(), rtol=0.0, atol=0.0)

    _assert_fp8_gpu_expert_matches_cpu_path(monkeypatch)
    _assert_fp8_gpu_expert_respects_modules_to_not_convert(monkeypatch)


def _assert_fp8_gpu_expert_matches_cpu_path(monkeypatch):
    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "gpu")
    base = torch.linspace(-9.0, 9.0, steps=3 * 256 * 128, dtype=torch.float32).reshape(3, 256, 128)
    local_cpu = base.to(torch.bfloat16)
    local_cuda = local_cpu.cuda()
    quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}

    gpu_out, gpu_original_bytes = WeightSyncHandler._quantize_ep_expert_projection_for_fp8_gpu_to_cpu(
        local_cuda,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=2,
        quantization_config=quantization_config,
        phase_s={},
    )
    cpu_out, cpu_original_bytes = WeightSyncHandler._quantize_ep_expert_projection_for_fp8_cpu(
        local_cpu,
        full_prefix="model.layers.0.mlp.experts",
        proj_name="gate_proj",
        ep_rank=2,
        quantization_config=quantization_config,
        phase_s={},
    )

    assert gpu_original_bytes == cpu_original_bytes
    assert [name for name, _ in gpu_out] == [name for name, _ in cpu_out]

    for (gpu_name, gpu_tensor), (cpu_name, cpu_tensor) in zip(gpu_out, cpu_out):
        assert gpu_name == cpu_name
        assert gpu_tensor.device.type == "cpu"
        assert gpu_tensor.dtype == cpu_tensor.dtype
        if gpu_name.endswith(".weight_scale_inv"):
            torch.testing.assert_close(gpu_tensor, cpu_tensor, rtol=0.0, atol=1e-6)
        else:
            torch.testing.assert_close(gpu_tensor.float(), cpu_tensor.float(), rtol=0.0, atol=0.0)


def _assert_fp8_gpu_expert_respects_modules_to_not_convert(monkeypatch):
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
