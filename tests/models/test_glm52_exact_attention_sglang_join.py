from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file as load_safetensors_file


pytest.importorskip("sglang")

# sglang.srt's import chain reaches flashinfer, which probes CUDA device
# properties at import time and asserts on CPU-only torch builds.
if not torch.cuda.is_available():
    pytest.skip("sglang.srt serving imports require CUDA-enabled torch", allow_module_level=True)

from sglang.srt.lora.layers import (  # noqa: E402
    ColumnParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from sglang.srt.lora.lora import LoRAAdapter  # noqa: E402
from sglang.srt.lora.lora_config import LoRAConfig  # noqa: E402
from sglang.srt.lora.mem_pool import LoRAMemoryPool  # noqa: E402

from xorl.lora.utils import save_lora_checkpoint  # noqa: E402
from xorl.models.transformers.glm5.exact_absorbed_kv_b_qlora import (  # noqa: E402
    Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA,
)
from xorl.models.transformers.glm5.exact_qlora import Glm52ExactTP1BlockFP8QLoRALinear  # noqa: E402


_HIDDEN_SIZE = 6144
_Q_LORA_RANK = 2048
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 192
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 256
_NUM_HEADS = 64
_KV_A_OUTPUT = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
_Q_B_OUTPUT = _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)
_KV_B_OUTPUT = _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM)
_O_INPUT = _NUM_HEADS * _V_HEAD_DIM
_MAX_LORA_RANK = 1
_ATTN_PREFIX = "base_model.model.model.layers.0.self_attn"
_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj")
_NORMALIZED_TARGETS = {"fused_qkv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj"}


class _ReplicatedFusedQKVA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tp_size = 1
        self.tp_rank = 0
        self.output_size = _Q_LORA_RANK + _KV_A_OUTPUT
        self.weight = torch.nn.Parameter(torch.empty(1))


class _ColumnProjectionBase(torch.nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.tp_size = 1
        self.tp_rank = 0
        self.output_sizes = [output_size]
        self.output_partition_sizes = [output_size]
        self.weight = torch.nn.Parameter(torch.empty(1))


class _RowOutputBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tp_size = 1
        self.tp_rank = 0
        self.input_size_per_partition = _O_INPUT
        self.output_size = _HIDDEN_SIZE
        self.weight = torch.nn.Parameter(torch.empty(1))


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
    return wrapper


def _materialize_factor(module: torch.nn.Module, name: str, pattern_id: int) -> None:
    shape = tuple(getattr(module, name).shape)
    values = (
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
        .reshape(shape)
        .add_(17 * pattern_id)
        .remainder_(251)
        .sub_(125)
        .div_(509 + 13 * pattern_id)
    )
    setattr(module, name, torch.nn.Parameter(values))


def _export_model() -> torch.nn.Module:
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    attention = torch.nn.Module()
    root.model.layers[0].self_attn = attention

    meta = torch.device("meta")
    attention.q_a_proj = Glm52ExactTP1BlockFP8QLoRALinear(_HIDDEN_SIZE, _Q_LORA_RANK, device=meta)
    attention.kv_a_proj_with_mqa = Glm52ExactTP1BlockFP8QLoRALinear(_HIDDEN_SIZE, _KV_A_OUTPUT, device=meta)
    attention.q_b_proj = Glm52ExactTP1BlockFP8QLoRALinear(_Q_LORA_RANK, _Q_B_OUTPUT, device=meta)
    attention.kv_b_proj = Glm52ExactTP1AbsorbedKvBBlockFP8QLoRA(device=meta)
    attention.o_proj = Glm52ExactTP1BlockFP8QLoRALinear(_O_INPUT, _HIDDEN_SIZE, device=meta)

    pattern_id = 1
    for projection_name in _PROJECTIONS:
        projection = getattr(attention, projection_name)
        for factor_name in ("lora_A", "lora_B"):
            _materialize_factor(projection, factor_name, pattern_id)
            pattern_id += 1
    return root


def _factor_keys() -> dict[str, dict[str, str]]:
    return {
        projection: {factor: f"{_ATTN_PREFIX}.{projection}.lora_{factor}.weight" for factor in ("A", "B")}
        for projection in _PROJECTIONS
    }


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype is expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


def test_exact_attention_component_bf16_export_joins_real_sglang_adapter_and_slot_zero(tmp_path) -> None:
    """Scope: one attention component, not the complete 1,700-factor validator."""

    export_model = _export_model()
    checkpoint = tmp_path / "adapter"
    save_lora_checkpoint(
        export_model,
        str(checkpoint),
        target_modules=list(_PROJECTIONS),
        r=1,
        lora_alpha=1,
    )
    exported = load_safetensors_file(str(checkpoint / "adapter_model.safetensors"))
    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text())
    factor_keys = _factor_keys()
    expected_export_keys = {factor_keys[projection][factor] for projection in _PROJECTIONS for factor in ("A", "B")}
    assert len(expected_export_keys) == 10
    assert set(exported) == expected_export_keys
    assert all(tensor.dtype is torch.bfloat16 for tensor in exported.values())
    assert not torch.equal(
        exported[factor_keys["q_a_proj"]["A"]],
        exported[factor_keys["kv_a_proj_with_mqa"]["A"]],
    )

    # Deliberately avoid the GLM architecture tag: this focused join must use
    # ordinary SGLang normalization and must not relax or invoke the complete
    # 1,700-factor shared-outer validator.
    base_config = SimpleNamespace(
        architectures=["AttentionComponentTransportHarness"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=12288,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_HEADS,
        num_hidden_layers=1,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        vocab_size=32,
    )
    lora_config = LoRAConfig.from_dict(adapter_config)
    adapter = LoRAAdapter(
        "attention-component",
        lora_config,
        base_config,
        load_config=None,
        lora_backend=SimpleNamespace(),
    )
    assert adapter._glm52_validator is None
    adapter.initialize_weights_from_tensors(exported)

    normalized = adapter.layers[0].weights
    fused_a_key = f"{_ATTN_PREFIX}.fused_qkv_a_proj_with_mqa.lora_A.weight"
    fused_b_key = f"{_ATTN_PREFIX}.fused_qkv_a_proj_with_mqa.lora_B.weight"
    preserved_keys = {
        factor_keys[projection][factor] for projection in ("q_b_proj", "kv_b_proj", "o_proj") for factor in ("A", "B")
    }
    assert set(normalized) == {fused_a_key, fused_b_key, *preserved_keys}
    q_a_a = exported[factor_keys["q_a_proj"]["A"]]
    kv_a_a = exported[factor_keys["kv_a_proj_with_mqa"]["A"]]
    q_a_b = exported[factor_keys["q_a_proj"]["B"]]
    kv_a_b = exported[factor_keys["kv_a_proj_with_mqa"]["B"]]
    _assert_same_bytes(normalized[fused_a_key][0:1], q_a_a)
    _assert_same_bytes(normalized[fused_a_key][1:2], kv_a_a)
    _assert_same_bytes(normalized[fused_b_key][:_Q_LORA_RANK], q_a_b)
    _assert_same_bytes(normalized[fused_b_key][_Q_LORA_RANK:], kv_a_b)
    for key in preserved_keys:
        _assert_same_bytes(normalized[key], exported[key])

    fused_wrapper = _manual_lora_wrapper(ReplicatedLinearWithLoRA, _ReplicatedFusedQKVA())
    q_b_wrapper = _manual_lora_wrapper(ColumnParallelLinearWithLoRA, _ColumnProjectionBase(_Q_B_OUTPUT))
    kv_b_wrapper = _manual_lora_wrapper(ColumnParallelLinearWithLoRA, _ColumnProjectionBase(_KV_B_OUTPUT))
    o_wrapper = _manual_lora_wrapper(RowParallelLinearWithLoRA, _RowOutputBase())
    lora_modules = [
        {
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa": fused_wrapper,
            "model.layers.0.self_attn.q_b_proj": q_b_wrapper,
            "model.layers.0.self_attn.kv_b_proj": kv_b_wrapper,
            "model.layers.0.self_attn.o_proj": o_wrapper,
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
        target_modules=_NORMALIZED_TARGETS,
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
    assert set(pool.A_buffer) == set(pool.B_buffer) == _NORMALIZED_TARGETS

    a_slots = {name: pool.A_buffer[name][0][0] for name in _NORMALIZED_TARGETS}
    b_slots = {name: pool.B_buffer[name][0][0] for name in _NORMALIZED_TARGETS}
    assert a_slots["fused_qkv_a_proj_with_mqa"].shape == (2, _HIDDEN_SIZE)
    assert b_slots["fused_qkv_a_proj_with_mqa"].shape == (_Q_LORA_RANK + _KV_A_OUTPUT, 1)
    assert a_slots["q_b_proj"].shape == (1, _Q_LORA_RANK)
    assert b_slots["q_b_proj"].shape == (_Q_B_OUTPUT, 1)
    assert a_slots["kv_b_proj"].shape == (1, _KV_LORA_RANK)
    assert b_slots["kv_b_proj"].shape == (_KV_B_OUTPUT, 1)
    assert a_slots["o_proj"].shape == (1, _O_INPUT)
    assert b_slots["o_proj"].shape == (_HIDDEN_SIZE, 1)

    _assert_same_bytes(a_slots["fused_qkv_a_proj_with_mqa"][0:1], q_a_a)
    _assert_same_bytes(a_slots["fused_qkv_a_proj_with_mqa"][1:2], kv_a_a)
    _assert_same_bytes(b_slots["fused_qkv_a_proj_with_mqa"][:_Q_LORA_RANK], q_a_b)
    _assert_same_bytes(b_slots["fused_qkv_a_proj_with_mqa"][_Q_LORA_RANK:], kv_a_b)
    for projection in ("q_b_proj", "kv_b_proj", "o_proj"):
        _assert_same_bytes(a_slots[projection], exported[factor_keys[projection]["A"]])
        _assert_same_bytes(b_slots[projection], exported[factor_keys[projection]["B"]])

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
