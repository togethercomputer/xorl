from __future__ import annotations

import pytest
import torch
from torch import nn

from xorl.models.transformers.glm5.checkpoint_handler import (
    Glm5CheckpointHandler,
    Glm52ExactDenseGateUpPairBuffer,
)
from xorl.models.transformers.glm5.exact_dense_mlp import Glm52ExactTP1DenseMLP


def _model() -> tuple[nn.Module, Glm52ExactTP1DenseMLP]:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    mlp = Glm52ExactTP1DenseMLP(8, 128, device=torch.device("cpu"))
    mlp.bind_checkpoint_sources("model.layers.0.mlp")
    root.model.layers[0].mlp = mlp
    return root, mlp


def _pairs() -> dict[str, torch.Tensor]:
    return {
        "model.layers.0.mlp.gate_proj.weight": torch.full((128, 8), 0.5, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.gate_proj.weight_scale_inv": torch.tensor([[0.125]], dtype=torch.float32),
        "model.layers.0.mlp.up_proj.weight": torch.full((128, 8), -0.25, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.up_proj.weight_scale_inv": torch.tensor([[0.375]], dtype=torch.float32),
    }


def test_exact_dense_checkpoint_handler_emits_one_explicit_gate_then_up_native_pair() -> None:
    model, mlp = _model()
    handler = Glm5CheckpointHandler(
        num_experts=1,
        checkpoint_has_per_expert=False,
        model=model,
        load_family="dense",
    )
    assert handler._exact_dense_gate_up_buffer is not None
    assert handler._native_weight_buffer is None
    pairs = _pairs()

    emitted = []
    for key in (
        "model.layers.0.mlp.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale_inv",
    ):
        emitted.extend(handler.on_load_weight(key, pairs[key]))

    assert [name for name, _ in emitted] == [
        "model.layers.0.mlp.packed_weight_f32",
        "model.layers.0.mlp.weight_scale_inv",
    ]
    emitted_by_name = dict(emitted)
    fused_weight = torch.cat(
        (pairs["model.layers.0.mlp.gate_proj.weight"], pairs["model.layers.0.mlp.up_proj.weight"]),
        dim=0,
    )
    assert torch.equal(
        emitted_by_name["model.layers.0.mlp.packed_weight_f32"].view(torch.uint8),
        fused_weight.contiguous().view(torch.uint8),
    )
    assert torch.equal(
        emitted_by_name["model.layers.0.mlp.weight_scale_inv"],
        torch.cat(
            (
                pairs["model.layers.0.mlp.gate_proj.weight_scale_inv"],
                pairs["model.layers.0.mlp.up_proj.weight_scale_inv"],
            ),
            dim=0,
        ),
    )

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in emitted:
            parameters[name].copy_(tensor)
    assert torch.equal(mlp.fp8_weight().view(torch.uint8), fused_weight.view(torch.uint8))
    assert mlp._exact_gate_up_base_loaded is True
    handler.on_load_complete()

    _assert_exact_dense_gate_up_pair_buffer_fails_on_missing_duplicate_or_invalid_members()


def _assert_exact_dense_gate_up_pair_buffer_fails_on_missing_duplicate_or_invalid_members() -> None:
    model, _ = _model()
    pairs = _pairs()

    missing = Glm52ExactDenseGateUpPairBuffer(model)
    assert (
        missing.try_consume(
            "model.layers.0.mlp.gate_proj.weight",
            pairs["model.layers.0.mlp.gate_proj.weight"],
        )
        == []
    )
    with pytest.raises(ValueError, match="Incomplete exact dense gate/up FP8 pairs"):
        missing.validate_complete()

    duplicate = Glm52ExactDenseGateUpPairBuffer(model)
    key = "model.layers.0.mlp.gate_proj.weight"
    assert duplicate.try_consume(key, pairs[key]) == []
    with pytest.raises(ValueError, match="Duplicate exact dense gate/up pair member"):
        duplicate.try_consume(key, pairs[key])

    invalid = Glm52ExactDenseGateUpPairBuffer(model)
    bad_pairs = dict(pairs)
    bad_pairs["model.layers.0.mlp.up_proj.weight"] = torch.zeros(127, 8, dtype=torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="up_proj.weight must be float8_e4m3fn"):
        for bad_key, tensor in bad_pairs.items():
            invalid.try_consume(bad_key, tensor)

    nonfinite = Glm52ExactDenseGateUpPairBuffer(model)
    bad_scales = dict(pairs)
    bad_scales["model.layers.0.mlp.gate_proj.weight_scale_inv"] = torch.tensor([[float("inf")]], dtype=torch.float32)
    with pytest.raises(ValueError, match="gate weight_scale_inv contains non-finite"):
        for bad_key, tensor in bad_scales.items():
            nonfinite.try_consume(bad_key, tensor)
