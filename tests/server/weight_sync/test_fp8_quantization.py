from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.fp8_training import enrich_sync_quantization_with_fp8_bf16_islands
from xorl.lora.modules.linear import LoraLinear
from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.moe_experts import QLoRAMoeExperts
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


def test_generated_first_last_bf16_islands_pass_through_fp8_sync_quantization():
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


def test_fp8_quantization_matches_slime_blockwise_scale_contract():
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


def test_fp8_quantization_zero_padding_differs_from_last_element_padding_for_partial_fused_layout():
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


def test_fp8_quantization_uses_merged_dense_lora_weight_from_sync_extraction():
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


def test_fp8_quantization_uses_merged_qlora_weight_from_collective_ops(monkeypatch):
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


def test_fp8_quantization_uses_merged_qlora_moe_experts_from_collective_context(monkeypatch):
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

        def _quantize_2d(self, w: torch.Tensor, global_amax: torch.Tensor | None = None):
            raise NotImplementedError

        def _dequantize_2d(self, packed: torch.Tensor, scales_dict: dict, K: int, N: int) -> torch.Tensor:
            raise NotImplementedError

        def merge_weights(self, ema_decay: float = 0.1) -> None:
            raise NotImplementedError

        def _load_experts(self, _load_tensor, _shard_cache) -> None:
            raise NotImplementedError

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


@pytest.mark.parametrize("projection", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"])
def test_fp8_quantization_includes_qwen_linear_attention_packed_weights_by_default(projection):
    name = f"model.layers.0.linear_attn.{projection}.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
        )
    )

    assert set(out) == {name, f"model.layers.0.linear_attn.{projection}.weight_scale_inv"}
    assert out[name].dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("prefix", ["model.layers.0.mlp.shared_expert", "model.layers.0.mlp.shared_experts"])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_fp8_quantization_includes_shared_expert_projections_by_default(prefix, projection):
    name = f"{prefix}.{projection}.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    out = dict(
        WeightSyncHandler._quantize_buffer_for_fp8(
            [(name, tensor)],
            quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
        )
    )

    assert set(out) == {name, f"{prefix}.{projection}.weight_scale_inv"}
    assert out[name].dtype == torch.float8_e4m3fn
    assert out[f"{prefix}.{projection}.weight_scale_inv"].dtype == torch.float32


def test_fp8_quantization_skips_shared_expert_gate_by_default():
    name = "model.layers.0.mlp.shared_expert_gate.weight"
    tensor = torch.zeros(1, 8, dtype=torch.bfloat16)
    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor)],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
    )

    assert out == [(name, tensor)]
def test_fp8_quantization_includes_qwen_linear_attention_packed_projections():
    names = [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ]
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)

    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor) for name in names],
        quantization_config={"quant_method": "fp8", "weight_block_size": [2, 4]},
        target_device="cpu",
    )

    out_by_name = dict(out)
    assert set(out_by_name) == set(names) | {name.replace(".weight", ".weight_scale_inv") for name in names}
    for name in names:
        assert out_by_name[name].dtype == torch.float8_e4m3fn
        assert out_by_name[name.replace(".weight", ".weight_scale_inv")].dtype == torch.float32


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


def test_fp8_quantization_respects_modules_to_not_convert_weight_suffix():
    name = "model.layers.0.mlp.gate_proj.weight"
    tensor = torch.zeros(8, 4, dtype=torch.bfloat16)
    out = WeightSyncHandler._quantize_buffer_for_fp8(
        [(name, tensor)],
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [2, 4],
            "modules_to_not_convert": ["model.layers.0.mlp.gate_proj.weight"],
        },
    )

    assert out == [(name, tensor)]


def test_fp8_quantization_with_receiver_skip_list_preserves_qwen_passthrough_entries():
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


def test_fp8_quantization_with_receiver_skip_list_uses_broad_2d_selector():
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


def test_fp8_cpu_expert_projection_zero_padding_differs_from_last_element_padding_for_partial_layout():
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
def test_fp8_gpu_stack_quantization_returns_cuda_tensors_for_cuda_target(monkeypatch):
    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "gpu")
    stack = torch.arange(2 * 128 * 128, dtype=torch.bfloat16, device="cuda").reshape(2, 128, 128)
    phase_s = {}

    quantized, scale = WeightSyncHandler._quantize_fp8_stack(
        stack,
        fp8_dtype=torch.float8_e4m3fn,
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        block_size_row=128,
        block_size_col=128,
        target_device="cuda",
        phase_s=phase_s,
        phase_prefix="test_fp8",
    )

    assert quantized.device.type == "cuda"
    assert scale.device.type == "cuda"
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1, 1)
    assert phase_s["test_fp8_gpu_quant_s"] >= 0
    assert "test_fp8_gpu_output_copy_s" not in phase_s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_gpu_stack_quantization_matches_cpu_path(monkeypatch):
    base = torch.linspace(-7.5, 7.5, steps=2 * 128 * 256, dtype=torch.float32).reshape(2, 128, 256)
    stack = base.to(torch.bfloat16).cuda()
    kwargs = {
        "fp8_dtype": torch.float8_e4m3fn,
        "fp8_max": torch.finfo(torch.float8_e4m3fn).max,
        "block_size_row": 128,
        "block_size_col": 128,
        "target_device": "cpu",
        "phase_s": {},
        "phase_prefix": "test_fp8",
    }

    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "gpu")
    gpu_quantized, gpu_scale = WeightSyncHandler._quantize_fp8_stack(stack, **kwargs)
    monkeypatch.setenv("XORL_P2P_FP8_QUANTIZE_DEVICE", "cpu")
    cpu_quantized, cpu_scale = WeightSyncHandler._quantize_fp8_stack(stack.cpu(), **kwargs)

    torch.testing.assert_close(gpu_scale, cpu_scale, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(gpu_quantized.float(), cpu_quantized.float(), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_gpu_expert_projection_matches_cpu_direct_ep_path(monkeypatch):
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
