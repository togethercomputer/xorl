from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file as load_safetensors_file


pytest.importorskip("sglang")

from sglang.srt.lora.layers import MergedColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool

from xorl.lora.utils import save_lora_checkpoint
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP


_HIDDEN_SIZE = 8
_INTERMEDIATE_SIZE = 128
_MAX_LORA_RANK = 1
_MLP_PREFIX = "base_model.model.model.layers.0.mlp"


class _MergedGateUpBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tp_size = 1
        self.tp_rank = 0
        self.output_sizes = [_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE]
        self.output_partition_sizes = [_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE]
        self.weight = torch.nn.Parameter(torch.empty(2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE))


class _RowDownBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tp_size = 1
        self.tp_rank = 0
        self.input_size_per_partition = _INTERMEDIATE_SIZE
        self.output_size = _HIDDEN_SIZE
        self.weight = torch.nn.Parameter(torch.empty(_HIDDEN_SIZE, _INTERMEDIATE_SIZE))


class _PoolBaseModel(torch.nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))


def _manual_lora_wrapper(wrapper_type, base_layer):
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.base_layer = base_layer
    wrapper.lora_backend = SimpleNamespace()
    wrapper.set_lora = False
    if wrapper_type is MergedColumnParallelLinearWithLoRA:
        wrapper.n_slices = len(base_layer.output_partition_sizes)
    return wrapper


def _export_model() -> tuple[torch.nn.Module, Glm52ExactTP1DenseMLP]:
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    dense_mlp = Glm52ExactTP1DenseMLP(_HIDDEN_SIZE, _INTERMEDIATE_SIZE, device=torch.device("cpu"))
    root.model.layers[0].mlp = dense_mlp

    with torch.no_grad():
        dense_mlp.gate_proj.lora_A.copy_(
            torch.tensor([[0.1001, -0.2002, 0.3003, -0.4004, 0.5005, -0.6006, 0.7007, -0.8008]])
        )
        dense_mlp.up_proj.lora_A.copy_(
            torch.tensor([[-0.8108, 0.7107, -0.6106, 0.5105, -0.4104, 0.3103, -0.2102, 0.1101]])
        )
        dense_mlp.gate_proj.lora_B.copy_(
            torch.arange(_INTERMEDIATE_SIZE, dtype=torch.float32).sub_(47).div_(311).unsqueeze(1)
        )
        dense_mlp.up_proj.lora_B.copy_(
            torch.arange(_INTERMEDIATE_SIZE, dtype=torch.float32).sub_(73).div_(277).neg_().unsqueeze(1)
        )
        dense_mlp.down_proj.lora_A.copy_(
            torch.arange(_INTERMEDIATE_SIZE, dtype=torch.float32).sub_(61).div_(389).unsqueeze(0)
        )
        dense_mlp.down_proj.lora_B.copy_(torch.arange(_HIDDEN_SIZE, dtype=torch.float32).sub_(3).div_(173).unsqueeze(1))
    return root, dense_mlp


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype is expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def test_exact_dense_component_bf16_export_joins_real_sglang_adapter_and_slot_zero(tmp_path) -> None:
    """Scope: one dense MLP component, not the complete 1,700-factor GLM validator."""

    export_model, _dense_mlp = _export_model()
    checkpoint = tmp_path / "adapter"
    save_lora_checkpoint(
        export_model,
        str(checkpoint),
        target_modules=["gate_proj", "up_proj", "down_proj"],
        r=1,
        lora_alpha=1,
    )
    exported = load_safetensors_file(str(checkpoint / "adapter_model.safetensors"))
    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text())

    factor_keys = {
        projection: {factor: f"{_MLP_PREFIX}.{projection}.lora_{factor}.weight" for factor in ("A", "B")}
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    expected_export_keys = {
        factor_keys[projection][factor] for projection in factor_keys for factor in factor_keys[projection]
    }
    assert set(exported) == expected_export_keys
    assert all(tensor.dtype is torch.bfloat16 for tensor in exported.values())

    # A deliberately non-GLM architecture makes this a focused component join.
    # It exercises SGLang's ordinary adapter parser/normalizer without invoking
    # or weakening the complete GLM-5.2 shared-outer validator.
    base_config = SimpleNamespace(
        architectures=["DenseComponentTransportHarness"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_attention_heads=1,
        num_hidden_layers=1,
        num_key_value_heads=1,
        vocab_size=32,
    )
    lora_config = LoRAConfig.from_dict(adapter_config)
    adapter = LoRAAdapter(
        "dense-component",
        lora_config,
        base_config,
        load_config=None,
        lora_backend=SimpleNamespace(),
    )
    assert adapter._glm52_validator is None
    adapter.initialize_weights_from_tensors(exported)

    normalized = adapter.layers[0].weights
    gate_up_a_key = f"{_MLP_PREFIX}.gate_up_proj.lora_A.weight"
    gate_up_b_key = f"{_MLP_PREFIX}.gate_up_proj.lora_B.weight"
    down_a_key = factor_keys["down_proj"]["A"]
    down_b_key = factor_keys["down_proj"]["B"]
    assert set(normalized) == {gate_up_a_key, gate_up_b_key, down_a_key, down_b_key}

    expected_gate_up_a = torch.cat(
        (exported[factor_keys["gate_proj"]["A"]], exported[factor_keys["up_proj"]["A"]]), dim=0
    )
    expected_gate_up_b = torch.cat(
        (exported[factor_keys["gate_proj"]["B"]], exported[factor_keys["up_proj"]["B"]]), dim=0
    )
    _assert_same_bytes(normalized[gate_up_a_key], expected_gate_up_a)
    _assert_same_bytes(normalized[gate_up_b_key], expected_gate_up_b)
    _assert_same_bytes(normalized[down_a_key], exported[down_a_key])
    _assert_same_bytes(normalized[down_b_key], exported[down_b_key])

    gate_up_wrapper = _manual_lora_wrapper(MergedColumnParallelLinearWithLoRA, _MergedGateUpBase())
    down_wrapper = _manual_lora_wrapper(RowParallelLinearWithLoRA, _RowDownBase())
    lora_modules = [
        {
            "model.layers.0.mlp.gate_up_proj": gate_up_wrapper,
            "model.layers.0.mlp.down_proj": down_wrapper,
        }
    ]
    pool = LoRAMemoryPool(
        base_hf_config=base_config,
        max_loras_per_batch=2,
        dtype=torch.bfloat16,
        tp_size=1,
        tp_rank=0,
        attn_tp_size=1,
        max_lora_rank=_MAX_LORA_RANK,
        target_modules={"gate_up_proj", "down_proj"},
        base_model=_PoolBaseModel(base_config),
        eviction_policy="lru",
        lora_added_tokens_size=0,
        strict_loading=True,
        lora_modules=lora_modules,
    )
    pool.prepare_lora_batch(
        cur_uids={adapter.uid},
        lora_adapters={adapter.uid: adapter},
        lora_modules=lora_modules,
        lora_refs={},
        lora_embed_tokens_module=None,
        lora_lm_head_module=None,
    )
    assert pool.uid_to_buffer_id == {adapter.uid: 0}
    assert pool.buffer_id_to_uid[0] == adapter.uid

    gate_up_a_slot = pool.A_buffer["gate_up_proj"][0][0]
    gate_up_b_slot = pool.B_buffer["gate_up_proj"][0][0]
    down_a_slot = pool.A_buffer["down_proj"][0][0]
    down_b_slot = pool.B_buffer["down_proj"][0][0]
    assert gate_up_a_slot.shape == (2 * _MAX_LORA_RANK, _HIDDEN_SIZE)
    assert gate_up_b_slot.shape == (2 * _INTERMEDIATE_SIZE, _MAX_LORA_RANK)
    assert down_a_slot.shape == (_MAX_LORA_RANK, _INTERMEDIATE_SIZE)
    assert down_b_slot.shape == (_HIDDEN_SIZE, _MAX_LORA_RANK)

    # The admitted max_lora_rank=1 slot retains explicit [gate; up] order.
    # Checking each slice separately prevents a self-confirming concatenation
    # test from accepting a gate/up swap.
    _assert_same_bytes(gate_up_a_slot[0:1], exported[factor_keys["gate_proj"]["A"]])
    _assert_same_bytes(gate_up_a_slot[1:2], exported[factor_keys["up_proj"]["A"]])
    _assert_same_bytes(gate_up_b_slot[:_INTERMEDIATE_SIZE, :1], exported[factor_keys["gate_proj"]["B"]])
    _assert_same_bytes(gate_up_b_slot[_INTERMEDIATE_SIZE:, :1], exported[factor_keys["up_proj"]["B"]])
    _assert_same_bytes(down_a_slot[:1], exported[down_a_key])
    _assert_same_bytes(down_b_slot[:, :1], exported[down_b_key])

    assert all(
        torch.count_nonzero(layer_buffer[1]) == 0
        for module_buffers in pool.A_buffer.values()
        for layer_buffer in module_buffers
    )
    assert all(
        torch.count_nonzero(layer_buffer[1]) == 0
        for module_buffers in pool.B_buffer.values()
        for layer_buffer in module_buffers
    )
