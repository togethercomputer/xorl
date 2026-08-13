"""Byte evidence for the GLM adoption of the one-round FP32 exact SwiGLU.

Serving's exact mode computes the one-round FP32 SwiGLU universally
(``SiluAndMul.forward_exact`` whenever ``rl_on_policy_target`` is set,
xorl-sglang f10b907d8), and GLM-5.2 serves its dense and shared-expert MLPs
through that op. These gates pin the trainer half of the pairing at GLM
geometries:

1. the trainer op (``exact_fp32_silu_and_mul``) matches serving's
   ``fp32_silu_and_mul`` bitwise on matched bf16 inputs at the TP16
   shared-expert shard ([T, 256] gate_up) and the dense width
   ([T, 2*12288], the Glm5Config default intermediate size) — direct sglang
   cross-check, the two implementations stay deliberately independent;
2. the OLD two-op program (``F.silu(gate) * up`` in bf16) differs bitwise
   from the one-round output on these inputs, so the byte gates above have
   discriminating power (the deterministic fixture must differ in at least
   one element);
3. the separate-tensor form ``exact_fp32_silu_and_mul(cat([gate, up], -1))``
   — the ``_canonical_shared_local_partial`` pattern — equals serving on the
   concatenated tensor;
4. grad smoke: the op under ``requires_grad`` produces finite grads at the
   shard geometry (backward is trainer-owned numerics, so no bitwise
   assertion);
5. stamp engagement: with the resolution-time ``_exact_one_round_swiglu``
   stamp, ``Glm5MLP.forward`` routes to the one-round op; unstamped configs
   keep the historical two-round dispatch. (CPU-safe: the off-Hopper
   fallback ``_fp32_silu_and_mul`` realizes the same program.)
6. fail-closed admission: non-SiLU GLM configs cannot enter the exact family,
   and the trainer/sampler local-partial policy identifiers stay aligned.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import xorl.models.transformers.glm5.modeling_glm5 as modeling_glm5
from xorl.models.auto import _validate_canonical_glm52_model_scope
from xorl.models.exact_contract import (
    EXACT_CONTRACT_FAMILY_GLM52,
    resolve_exact_contract_family,
)
from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config
from xorl.models.transformers.glm5.modeling_glm5 import GLM52_LOCAL_PARTIAL_POLICY, Glm5MLP
from xorl.ops.fused_silu_and_mul import exact_fp32_silu_and_mul

# GLM-5.2 geometries with byte evidence: the 128-wide TP16 shared-expert
# shard (gate|up halves of 2048/16) and the dense MLP width (Glm5Config
# default intermediate_size=12288).
_TP16_SHARD_GEOMETRY = (512, 256)
_DENSE_GEOMETRY = (64, 2 * 12288)
_GEOMETRY_IDS = ("tp16_shared_expert_shard", "dense_mlp")


def _serving_one_round_op():
    serving = pytest.importorskip(
        "sglang.srt.batch_invariant_ops.bi_silu_and_mul",
        reason="serving package not importable in this venv",
    )
    return serving.fp32_silu_and_mul


def _matched_input(rows: int, width: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rows, width, generator=generator) * 1.5
    return raw.to(torch.bfloat16).cuda().contiguous()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("rows", "width"), [_TP16_SHARD_GEOMETRY, _DENSE_GEOMETRY], ids=_GEOMETRY_IDS)
def test_trainer_op_matches_serving_one_round_bitwise(rows: int, width: int) -> None:
    fp32_silu_and_mul = _serving_one_round_op()
    matched = _matched_input(rows, width, seed=17)

    trainer = exact_fp32_silu_and_mul(matched)
    serving = fp32_silu_and_mul(matched)

    assert trainer.dtype is torch.bfloat16 and serving.dtype is torch.bfloat16
    assert trainer.shape == serving.shape == (rows, width // 2)
    assert torch.equal(trainer.view(torch.uint8), serving.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("rows", "width"), [_TP16_SHARD_GEOMETRY, _DENSE_GEOMETRY], ids=_GEOMETRY_IDS)
def test_old_two_op_program_differs_bitwise(rows: int, width: int) -> None:
    """Sensitivity gate: the pre-adoption two-op program is a different
    byte program on the same inputs, so the identity gate above can fail."""
    matched = _matched_input(rows, width, seed=17)
    split = width // 2

    one_round = exact_fp32_silu_and_mul(matched)
    two_op = F.silu(matched[..., :split]) * matched[..., split:]

    differing = int((one_round.view(torch.uint16) != two_op.view(torch.uint16)).sum())
    total = one_round.numel()
    print(f"two-op vs one-round differing elements at {rows}x{width}: {differing}/{total}")
    assert not torch.equal(one_round, two_op), (
        f"two-op and one-round became bitwise identical on {total} elements; "
        "the identity gates in this file would be inert"
    )
    assert differing > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_separate_gate_up_concat_form_matches_serving() -> None:
    """The _canonical_shared_local_partial pattern: gate and up arrive as
    separate tensors and are concatenated into the op's packed layout."""
    fp32_silu_and_mul = _serving_one_round_op()
    rows, shard = 384, 128
    gate = _matched_input(rows, 2 * shard, seed=23)[..., :shard].contiguous()
    up = _matched_input(rows, 2 * shard, seed=29)[..., shard:].contiguous()
    concatenated = torch.cat([gate, up], dim=-1)

    trainer = exact_fp32_silu_and_mul(concatenated)
    serving = fp32_silu_and_mul(concatenated)

    assert torch.equal(trainer.view(torch.uint8), serving.view(torch.uint8))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grad_smoke_at_shard_geometry() -> None:
    rows, width = _TP16_SHARD_GEOMETRY
    matched = _matched_input(rows, width, seed=31).requires_grad_(True)
    grad_output = _matched_input(rows, 2 * (width // 2), seed=37)[..., : width // 2].contiguous()

    output = exact_fp32_silu_and_mul(matched)
    output.backward(grad_output)

    assert matched.grad is not None
    assert matched.grad.shape == matched.shape
    assert torch.isfinite(matched.grad.float()).all()


def _glm_mlp_config(**stamps: object) -> SimpleNamespace:
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        hidden_act="silu",
        _activation_native=False,
    )
    for name, value in stamps.items():
        setattr(config, name, value)
    return config


def _official_glm52_config(*, hidden_act: str = "silu") -> Glm5Config:
    indexer_types = tuple("full" if i < 3 or (i - 2) % 4 == 0 else "shared" for i in range(78))
    mlp_layer_types = ("dense",) * 3 + ("sparse",) * 75
    return Glm5Config(
        indexer_types=indexer_types,
        index_topk_freq=4,
        mlp_layer_types=mlp_layer_types,
        hidden_act=hidden_act,
    )


@pytest.mark.cpu
def test_glm52_exact_family_is_the_stamp_key() -> None:
    """Model resolution stamps ``_exact_one_round_swiglu`` for every exact
    contract family; a GLM-5.2 exact config resolves to that family."""
    assert resolve_exact_contract_family(SimpleNamespace(_glm52_exact_contract=True)) == EXACT_CONTRACT_FAMILY_GLM52


@pytest.mark.cpu
def test_glm52_exact_admission_rejects_non_silu_activation() -> None:
    with pytest.raises(ValueError, match="hidden_act='gelu'.*requires 'silu'"):
        _validate_canonical_glm52_model_scope(_official_glm52_config(hidden_act="gelu"))


@pytest.mark.cpu
def test_glm52_exact_mlp_rejects_non_silu_stamp() -> None:
    config = _glm_mlp_config(_exact_one_round_swiglu=True)
    config.hidden_act = "gelu"
    with pytest.raises(ValueError, match="requires hidden_act='silu'"):
        Glm5MLP(config)


@pytest.mark.cpu
def test_glm52_local_partial_policy_matches_pinned_serving() -> None:
    serving = pytest.importorskip("sglang.srt.distributed.canonical_moe")
    assert GLM52_LOCAL_PARTIAL_POLICY == serving.GLM52_SAMPLER_LOCAL_POLICY


@pytest.mark.cpu
def test_stamped_glm5_mlp_routes_to_the_one_round_op(monkeypatch) -> None:
    calls: list[str] = []

    def one_round_capture(gate_up: torch.Tensor) -> torch.Tensor:
        calls.append("one_round")
        split = gate_up.shape[-1] // 2
        return (F.silu(gate_up[..., :split].float()) * gate_up[..., split:].float()).to(gate_up.dtype)

    def two_round_capture(gate_up: torch.Tensor) -> torch.Tensor:
        calls.append("two_round")
        split = gate_up.shape[-1] // 2
        activated = F.silu(gate_up[..., :split].float()).to(gate_up.dtype)
        return (activated * gate_up[..., split:]).to(gate_up.dtype)

    monkeypatch.setattr(modeling_glm5, "exact_fp32_silu_and_mul", one_round_capture)
    monkeypatch.setattr(modeling_glm5, "fused_silu_and_mul", two_round_capture)

    torch.manual_seed(5)
    hidden = torch.randn(3, 16)

    stamped = Glm5MLP(_glm_mlp_config(_exact_one_round_swiglu=True))
    assert stamped._exact_one_round
    stamped(hidden)
    assert calls == ["one_round"]

    calls.clear()
    unstamped = Glm5MLP(_glm_mlp_config())
    assert not unstamped._exact_one_round
    assert unstamped._use_fused_silu
    unstamped(hidden)
    assert calls == ["two_round"]
