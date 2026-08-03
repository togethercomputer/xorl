import sys

import pytest
import torch
import torch.distributed.checkpoint as dcp

from xorl.distributed.torch_parallelize import _expert_fsdp_kwargs_for_module
from xorl.ops.block_fp8_native import (
    NativeBlockFP8Linear,
    pack_fp8_as_float32,
    unpack_float32_as_fp8,
    validate_native_fp8_dcp_checkpoint,
    validate_native_fp8_state_metadata,
)


def _fp8_values(shape):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int16)
    values = ((values % 31) - 15).to(torch.float32).reshape(shape)
    return values.to(torch.float8_e4m3fn)


def test_pack_roundtrip_is_byte_exact():
    weight = _fp8_values((256, 128))
    packed = pack_fp8_as_float32(weight)
    restored = unpack_float32_as_fp8(packed, tuple(weight.shape))

    assert packed.dtype is torch.float32
    assert packed.shape == (256, 32)
    assert torch.equal(restored.view(torch.uint8), weight.contiguous().view(torch.uint8))


def test_linear_state_is_frozen_reshardable_and_scale_exact():
    module = NativeBlockFP8Linear(256, 384)
    weight = _fp8_values((384, 256))
    scales = torch.arange(6, dtype=torch.float32).reshape(3, 2) / 7
    module.load_prequantized(weight, scales)

    params = dict(module.named_parameters())
    assert set(params) == {"packed_weight_f32", "weight_scale_inv"}
    assert all(not parameter.requires_grad for parameter in params.values())
    assert module.fsdp_requires_full_precision is True
    assert torch.equal(module.fp8_weight().view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(module.weight_scale_inv.view(torch.uint8), scales.view(torch.uint8))


def test_dtype_apply_preserves_packed_and_scale_bytes():
    module = NativeBlockFP8Linear(128, 128)
    weight = _fp8_values((128, 128))
    scales = torch.tensor([[0.1234567]], dtype=torch.float32)
    module.load_prequantized(weight, scales)
    weight_bytes = module.packed_weight_f32.view(torch.uint8).clone()
    scale_bytes = module.weight_scale_inv.view(torch.uint8).clone()

    module.to(dtype=torch.bfloat16)

    assert module.packed_weight_f32.dtype is torch.float32
    assert module.weight_scale_inv.dtype is torch.float32
    assert torch.equal(module.packed_weight_f32.view(torch.uint8), weight_bytes)
    assert torch.equal(module.weight_scale_inv.view(torch.uint8), scale_bytes)


def test_no_eager_sglang_import_and_cpu_forward_fails_closed():
    before = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    module = NativeBlockFP8Linear(128, 128)
    after = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}

    assert after == before
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16))
    receipt = module.engagement_receipt()
    assert receipt["engagement_count"] == 0
    assert receipt["cuda_graph_phase_evidence"] == "external_runner_receipt_required"


def test_weight_materialization_is_an_explicit_input_free_cuda_path():
    module = NativeBlockFP8Linear(128, 128)
    with pytest.raises(ValueError, match="does not accept activation or range inputs"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16), return_dequantized_weight=True)
    with pytest.raises(ValueError, match="does not accept activation or range inputs"):
        module(return_dequantized_weight=True, output_range=(0, 128))
    with pytest.raises(RuntimeError, match="materialization requires CUDA"):
        module(return_dequantized_weight=True)


def test_partition_ranges_cross_the_module_forward_hook_boundary(monkeypatch):
    module = NativeBlockFP8Linear(256, 384)
    input = torch.zeros(2, 128, dtype=torch.bfloat16)
    expected = torch.ones(2, 128, dtype=torch.bfloat16)
    calls = []
    pre_hook_calls = []

    def fake_forward_partition(value, *, output_range=None, input_range=None):
        calls.append((value, output_range, input_range))
        return expected

    monkeypatch.setattr(module, "forward_partition", fake_forward_partition)
    module.register_forward_pre_hook(lambda *_: pre_hook_calls.append(True))

    actual = module(input, output_range=(128, 256), input_range=(0, 128))

    assert actual is expected
    assert pre_hook_calls == [True]
    assert calls == [(input, (128, 256), (0, 128))]


def test_phase_one_forward_rejects_activation_or_base_gradients():
    module = NativeBlockFP8Linear(128, 128)
    with pytest.raises(RuntimeError, match="scoring-only"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16, requires_grad=True))

    module.packed_weight_f32.requires_grad_(True)
    with pytest.raises(RuntimeError, match="must remain frozen"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16))


def test_linear_partition_contract_fails_before_kernel_dispatch():
    module = NativeBlockFP8Linear(256, 384)
    input = torch.zeros(1, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="block boundaries"):
        module.forward_partition(input, input_range=(64, 192))
    with pytest.raises(ValueError, match="input width"):
        module.forward_partition(input, input_range=(0, 256))


def test_rejects_wrong_pair_dtype_shape_and_nonfinite_scale():
    module = NativeBlockFP8Linear(128, 128)
    weight = _fp8_values((128, 128))

    with pytest.raises(TypeError, match="remain FP32"):
        module.load_prequantized(weight, torch.ones(1, 1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="scale shape"):
        module.load_prequantized(weight, torch.ones(2, 1, dtype=torch.float32))
    with pytest.raises(ValueError, match="non-finite"):
        module.load_prequantized(weight, torch.tensor([[float("nan")]], dtype=torch.float32))
    with pytest.raises(ValueError, match="weight shape"):
        module.load_prequantized(_fp8_values((128, 256)), torch.ones(1, 1, dtype=torch.float32))


def test_state_dict_roundtrip_retains_both_parameter_byte_streams():
    source = NativeBlockFP8Linear(256, 384)
    target = NativeBlockFP8Linear(256, 384)
    weight = _fp8_values((384, 256))
    scales = torch.arange(6, dtype=torch.float32).reshape(3, 2) / 13
    source.load_prequantized(weight, scales)

    target.load_state_dict(source.state_dict(), strict=True)

    assert set(source.state_dict()) == {"packed_weight_f32", "weight_scale_inv"}
    assert torch.equal(target.packed_weight_f32.view(torch.uint8), source.packed_weight_f32.view(torch.uint8))
    assert torch.equal(target.weight_scale_inv.view(torch.uint8), source.weight_scale_inv.view(torch.uint8))


def test_state_dict_and_dcp_metadata_reject_castable_payloads():
    module = NativeBlockFP8Linear(128, 128)
    state = module.state_dict()
    state["packed_weight_f32"] = state["packed_weight_f32"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="refusing a load_state_dict cast"):
        module.load_state_dict(state, strict=True)

    good_metadata = {name: (parameter.dtype, tuple(parameter.shape)) for name, parameter in module.named_parameters()}
    validate_native_fp8_state_metadata(module, good_metadata)
    bad_metadata = dict(good_metadata)
    bad_metadata["weight_scale_inv"] = (torch.bfloat16, tuple(module.weight_scale_inv.shape))
    with pytest.raises(ValueError, match="DCP metadata mismatch"):
        validate_native_fp8_state_metadata(module, bad_metadata)


def test_apply_exception_never_strands_protected_parameters():
    module = NativeBlockFP8Linear(128, 128)
    original = dict(module.named_parameters())

    def fail_on_nonempty(tensor):
        if tensor.numel():
            raise RuntimeError("adversarial apply failure")
        return tensor

    with pytest.raises(RuntimeError, match="adversarial"):
        module._apply(fail_on_nonempty)

    restored = dict(module.named_parameters())
    assert set(restored) == set(original)
    assert all(restored[name] is parameter for name, parameter in original.items())


def test_real_dcp_metadata_is_checked_before_load(tmp_path):
    module = NativeBlockFP8Linear(128, 128)
    good_path = tmp_path / "good"
    dcp.save({"model": module.state_dict()}, checkpoint_id=good_path)

    receipt = validate_native_fp8_dcp_checkpoint(str(good_path), module.state_dict())
    assert receipt["metadata_gate"] == "pass"
    assert len(receipt["validated_native_keys"]) == 2

    bad_state = module.state_dict()
    bad_state["packed_weight_f32"] = bad_state["packed_weight_f32"].to(torch.bfloat16)
    bad_path = tmp_path / "bad"
    dcp.save({"model": bad_state}, checkpoint_id=bad_path)
    with pytest.raises(ValueError, match="DCP metadata mismatch"):
        validate_native_fp8_dcp_checkpoint(str(bad_path), module.state_dict())


def test_dcp_preflight_uses_ep_restored_expected_shape(tmp_path):
    # Simulate ModelState.state_dict(): live expert params may have E_local=2,
    # while the expected DCP view restores the global E=4 dimension.
    restored_state = {
        "experts.gate_up_packed_weight_f32": torch.empty(4, 256, 64, dtype=torch.float32),
        "experts.gate_up_weight_scale_inv": torch.empty(4, 2, 2, dtype=torch.float32),
        "experts.down_packed_weight_f32": torch.empty(4, 128, 64, dtype=torch.float32),
        "experts.down_weight_scale_inv": torch.empty(4, 1, 2, dtype=torch.float32),
    }
    checkpoint_path = tmp_path / "ep-restored"
    dcp.save({"model": restored_state}, checkpoint_id=checkpoint_path)

    receipt = validate_native_fp8_dcp_checkpoint(str(checkpoint_path), restored_state)

    assert receipt["metadata_gate"] == "pass"
    assert len(receipt["validated_native_keys"]) == 4


def test_generic_expert_fsdp_helper_removes_mixed_precision_without_skipping_fsdp():
    class FullPrecisionExpertState:
        fsdp_requires_full_precision = True

    kwargs = _expert_fsdp_kwargs_for_module(
        {"mesh": "mesh", "mp_policy": "bf16", "reshard_after_forward": True},
        FullPrecisionExpertState(),
    )

    assert kwargs == {"mesh": "mesh", "reshard_after_forward": True}
