"""CPU numerical gate for the fresh_ab (EGGROLL) ZORL ES estimator.

Covers the four required properties:
  (a) fresh_ab candidate factors from a seed are deterministic and antithetic
      pairs share A;
  (b) the fold G equals the explicit sum of per-candidate probe deltas
      computed naively from the exported candidate tensors;
  (c) parity with the sglang reference formula
      u = (1/N) * sum_i z_i * scaling * (eps_B,i @ eps_A,i)
      via an inline (import-free) port of the reference math;
  (d) an end-to-end small-tensor apply through the real base-optimizer path
      (build_optimizer -> Muon, full-gradient NS, match_rms_adamw scale),
      asserting the base weight moved along +NS(G) at the match_rms_adamw
      learning-rate scale (reference: inline port of sglang _zorl_muon_update,
      the same port validated in
      experiments/zorl/standalone/r1_validate_muon_es_fold.py).

Tests must not import sglang; every reference is ported inline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xorl.lora.modules.linear import LoraLinear
from xorl.server.runner.model_runner import ModelRunner
from xorl.server.zorl import (
    ZORLSessionState,
    build_zorl_fresh_ab_base_update_from_rewards,
    build_zorl_fresh_ab_candidate_lora_state_dict,
    iter_zorl_b_noises,
    iter_zorl_fresh_ab_a_noises,
    normalize_zorl_runtime_config,
    pair_zorl_fresh_ab_lora_params,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]


# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


def _session_spec(perturbation_mode: str = "fresh_ab", **overrides):
    zorl_config = {
        "enabled": True,
        "b_sigma": 0.02,
        "num_perturbation_pairs": 3,
        "a_refresh_interval": 0,
        "antithetic_sampling": True,
        "a_init": "gaussian_jl",
        "perturbation_mode": perturbation_mode,
        "seed": 1234,
    }
    zorl_config.update(overrides)
    return {"zorl_config": zorl_config}


def _linear_lora_params(seed: int = 0):
    """Two LoraLinear-shaped modules: A [r, in], B [out, r]."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "model.layers.0.self_attn.q_proj.lora_A": nn.Parameter(torch.randn(4, 32, generator=generator)),
        "model.layers.0.self_attn.q_proj.lora_B": nn.Parameter(torch.randn(24, 4, generator=generator)),
        "model.layers.1.mlp.down_proj.lora_A": nn.Parameter(torch.randn(4, 24, generator=generator)),
        "model.layers.1.mlp.down_proj.lora_B": nn.Parameter(torch.randn(16, 4, generator=generator)),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm() + 1e-30))


def relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-30))


# ---------------------------------------------------------------------------
# Config / planning
# ---------------------------------------------------------------------------


def test_normalize_zorl_runtime_config_accepts_fresh_ab_and_defaults_to_b_only():
    config = normalize_zorl_runtime_config({"sigma": 0.1, "num_pairs": 2, "perturbation_mode": "fresh_ab"})
    assert config is not None
    assert config["perturbation_mode"] == "fresh_ab"

    default_config = normalize_zorl_runtime_config({"sigma": 0.1, "num_pairs": 2})
    assert default_config is not None
    assert default_config["perturbation_mode"] == "b_only"

    with pytest.raises(ValueError, match="perturbation_mode"):
        normalize_zorl_runtime_config({"sigma": 0.1, "perturbation_mode": "a_and_b"})


def test_fresh_ab_plan_assigns_pair_shared_a_seeds_and_keeps_b_seed_stream():
    fresh_state = ZORLSessionState.from_session_spec("policy", _session_spec("fresh_ab"))
    b_only_state = ZORLSessionState.from_session_spec("policy", _session_spec("b_only"))
    assert fresh_state is not None and b_only_state is not None

    fresh_plan = fresh_state.begin_generation()
    b_only_plan = b_only_state.begin_generation()

    by_pair: dict[int, list] = {}
    for candidate in fresh_plan.candidates:
        by_pair.setdefault(candidate.perturbation_index, []).append(candidate)

    a_seeds = set()
    for pair_index, members in by_pair.items():
        assert {member.direction for member in members} == {"positive", "negative"}
        # The antithetic pair shares BOTH seeds (pair probes eps_B @ eps_A).
        assert len({member.b_seed for member in members}) == 1
        assert len({member.a_seed for member in members}) == 1
        assert members[0].a_seed is not None
        a_seeds.add(members[0].a_seed)
        assert members[0].a_seed != members[0].b_seed
    # Fresh A per pair: distinct a_seed per perturbation index.
    assert len(a_seeds) == len(by_pair)

    # fresh_ab is additive: the b_seed stream is identical to b_only mode.
    assert [c.b_seed for c in fresh_plan.candidates] == [c.b_seed for c in b_only_plan.candidates]
    assert all(c.a_seed is None for c in b_only_plan.candidates)


# ---------------------------------------------------------------------------
# (a) Candidate materialization: determinism + antithetic-shared A
# ---------------------------------------------------------------------------


def test_fresh_ab_candidates_are_deterministic_and_share_a_within_pair():
    lora_params = _linear_lora_params()
    b_sigma = 0.02
    a_seed, b_seed = 777, 888

    positive = build_zorl_fresh_ab_candidate_lora_state_dict(
        lora_params, a_seed=a_seed, b_seed=b_seed, direction="positive", b_sigma=b_sigma
    )
    positive_again = build_zorl_fresh_ab_candidate_lora_state_dict(
        lora_params, a_seed=a_seed, b_seed=b_seed, direction="positive", b_sigma=b_sigma
    )
    negative = build_zorl_fresh_ab_candidate_lora_state_dict(
        lora_params, a_seed=a_seed, b_seed=b_seed, direction="negative", b_sigma=b_sigma
    )

    for name in lora_params:
        # Deterministic from seeds alone.
        assert torch.equal(positive[name], positive_again[name])
        if "lora_A" in name:
            # Antithetic pair shares A; A is a fresh draw, not the parent's A.
            assert torch.equal(positive[name], negative[name])
            assert not torch.allclose(positive[name], lora_params[name].data)
        else:
            # B carries the antithetic sign and the sigma scale.
            assert torch.equal(positive[name], -negative[name])

    # B is exactly sign * sigma * eps_B for the seeded eps_B stream.
    eps_b = dict(iter_zorl_b_noises(lora_params, seed=b_seed))
    for name in (n for n in lora_params if "lora_B" in n):
        assert torch.allclose(positive[name], b_sigma * eps_b[name])
    # A is exactly the unit-gaussian eps_A stream.
    eps_a = dict(iter_zorl_fresh_ab_a_noises(lora_params, seed=a_seed))
    for name in (n for n in lora_params if "lora_A" in n):
        assert torch.equal(positive[name], eps_a[name])


def test_fresh_ab_candidates_ignore_parent_values():
    lora_params_a = _linear_lora_params(seed=0)
    lora_params_b = _linear_lora_params(seed=99)  # same shapes, different parent values
    kwargs = dict(a_seed=41, b_seed=42, direction="positive", b_sigma=0.5)
    state_a = build_zorl_fresh_ab_candidate_lora_state_dict(lora_params_a, **kwargs)
    state_b = build_zorl_fresh_ab_candidate_lora_state_dict(lora_params_b, **kwargs)
    for name in state_a:
        assert torch.equal(state_a[name], state_b[name])


# ---------------------------------------------------------------------------
# (b) Fold G == naive sum of per-candidate probe deltas
# ---------------------------------------------------------------------------


def test_fresh_ab_fold_matches_naive_candidate_delta_sum():
    lora_params = _linear_lora_params()
    b_sigma = 0.02
    scaling = 2.0  # lora_alpha / r
    pairs = [(101, 201, 0.8), (102, 202, -0.4), (103, 203, 1.3)]  # (a_seed, b_seed, z)

    updates, update_norm = build_zorl_fresh_ab_base_update_from_rewards(
        lora_params, pair_seeds_and_scores=pairs, scaling=scaling
    )

    # Naive: reconstruct each pair's POSITIVE candidate, take its effective
    # served delta scaling * (B+ @ A) = sigma * scaling * (eps_B @ eps_A), and
    # accumulate z_i / (N * sigma) — the canonical antithetic ES estimator.
    modules = pair_zorl_fresh_ab_lora_params(lora_params)
    naive = {key: torch.zeros_like(updates[key]) for key in updates}
    num_pairs = len(pairs)
    for a_seed, b_seed, score in pairs:
        candidate = build_zorl_fresh_ab_candidate_lora_state_dict(
            lora_params, a_seed=a_seed, b_seed=b_seed, direction="positive", b_sigma=b_sigma
        )
        for module_key, (a_name, b_name) in modules.items():
            served_delta = scaling * (candidate[b_name] @ candidate[a_name])
            naive[module_key] += (score / (num_pairs * b_sigma)) * served_delta

    assert set(updates) == set(naive)
    for module_key in updates:
        assert torch.allclose(updates[module_key], naive[module_key], atol=1e-5), module_key
    assert update_norm > 0.0


def test_fresh_ab_fold_moe_orientation_and_hybrid_broadcast():
    num_experts, hidden, inter, r = 3, 10, 6, 2
    lora_params = {
        # Per-expert MoE layout (G, K, N): A [E, in, r], B [E, r, out].
        "layers.0.experts.down_proj_lora_A": nn.Parameter(torch.randn(num_experts, inter, r)),
        "layers.0.experts.down_proj_lora_B": nn.Parameter(torch.randn(num_experts, r, hidden)),
        # Hybrid-shared layout: A shared across experts [1, in, r].
        "layers.0.experts.gate_proj_lora_A": nn.Parameter(torch.randn(1, hidden, r)),
        "layers.0.experts.gate_proj_lora_B": nn.Parameter(torch.randn(num_experts, r, inter)),
    }
    pairs = [(11, 21, 1.0), (12, 22, -0.7)]
    scaling = 1.5

    updates, _ = build_zorl_fresh_ab_base_update_from_rewards(lora_params, pair_seeds_and_scores=pairs, scaling=scaling)

    assert tuple(updates["layers.0.experts.down_proj"].shape) == (num_experts, inter, hidden)
    assert tuple(updates["layers.0.experts.gate_proj"].shape) == (num_experts, hidden, inter)

    # Naive per-expert loop over the seeded noise streams.
    naive_down = torch.zeros(num_experts, inter, hidden)
    naive_gate = torch.zeros(num_experts, hidden, inter)
    for a_seed, b_seed, score in pairs:
        eps_a = dict(iter_zorl_fresh_ab_a_noises(lora_params, seed=a_seed))
        eps_b = dict(iter_zorl_b_noises(lora_params, seed=b_seed))
        coef = score / len(pairs) * scaling
        for expert in range(num_experts):
            naive_down[expert] += coef * (
                eps_a["layers.0.experts.down_proj_lora_A"][expert] @ eps_b["layers.0.experts.down_proj_lora_B"][expert]
            )
            naive_gate[expert] += coef * (
                eps_a["layers.0.experts.gate_proj_lora_A"][0] @ eps_b["layers.0.experts.gate_proj_lora_B"][expert]
            )
    assert torch.allclose(updates["layers.0.experts.down_proj"], naive_down, atol=1e-5)
    assert torch.allclose(updates["layers.0.experts.gate_proj"], naive_gate, atol=1e-5)


# ---------------------------------------------------------------------------
# (c) Parity with the sglang reference formula (inline port, no sglang import)
# ---------------------------------------------------------------------------


def _reference_fresh_ab_fold(lora_shapes, pairs, scaling):
    """Inline port of the sglang fresh_ab reward-weighted fold.

    sglang lora_manager semantics (apply_zorl_rewards[fresh_ab]):
      - per-seed noise: one CPU generator seeded once, sequential randn draws
        over the sorted parameter names (validated bit-identical to the
        sglang primitive for the B stream in
        experiments/zorl/standalone/r1_validate_muon_es_fold.py);
      - per-pair fold coefficient raw_coef = z_i / N (lr excluded — it is the
        optimizer's job), times the structural LoRA scaling;
      - per-module dense delta eps_B @ eps_A accumulated across pairs:
        u = (1/N) * sum_i z_i * scaling * (eps_B,i @ eps_A,i).
    """

    def draw(seed, marker):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        out = {}
        for name in sorted(n for n in lora_shapes if marker in n):
            out[name] = torch.randn(lora_shapes[name], generator=generator, dtype=torch.float32)
        return out

    module_keys = sorted({n.rsplit(".", 1)[0] for n in lora_shapes})
    reference = {}
    num_pairs = len(pairs)
    for a_seed, b_seed, score in pairs:
        raw_coef = float(score) / float(num_pairs)
        eps_a = draw(a_seed, "lora_A")
        eps_b = draw(b_seed, "lora_B")
        for module_key in module_keys:
            delta = eps_b[f"{module_key}.lora_B"] @ eps_a[f"{module_key}.lora_A"]
            if module_key not in reference:
                reference[module_key] = torch.zeros_like(delta)
            reference[module_key] += raw_coef * float(scaling) * delta
    return reference


def test_fresh_ab_fold_matches_sglang_reference_formula():
    lora_params = _linear_lora_params()
    lora_shapes = {name: tuple(param.shape) for name, param in lora_params.items()}
    pairs = [(301, 401, 1.1), (302, 402, -0.6), (303, 403, 0.2), (304, 404, -1.4)]
    scaling = 8.0 / 4.0

    updates, _ = build_zorl_fresh_ab_base_update_from_rewards(lora_params, pair_seeds_and_scores=pairs, scaling=scaling)
    reference = _reference_fresh_ab_fold(lora_shapes, pairs, scaling)

    assert set(updates) == set(reference)
    for module_key in sorted(updates):
        cos = cosine(updates[module_key], reference[module_key])
        err = relerr(updates[module_key], reference[module_key])
        print(f"fresh_ab G parity {module_key}: cos={cos:.9f} relerr={err:.3e}")
        assert cos > 0.999999, f"{module_key}: cos={cos}"
        assert err < 1e-6, f"{module_key}: relerr={err}"


# ---------------------------------------------------------------------------
# (d) End-to-end: real optim path (build_optimizer -> Muon) on a 2-layer model
# ---------------------------------------------------------------------------


def _reference_sglang_muon_update(delta, lr, compute_dtype=torch.float32, spectral_dims=None):
    """Inline port of sglang lora_manager._zorl_muon_update (no restart).

    Returns the update ADDED to the base: lr * 0.2*sqrt(max(rows, cols)) * NS(delta)
    — i.e. Muon with adjust_lr_fn='match_rms_adamw'. Same port as
    experiments/zorl/standalone/r1_validate_muon_es_fold.py part C.

    ``spectral_dims`` overrides the shape used for the match_rms_adamw lr
    adjustment: for fused gate_up storage Muon adjusts the lr from the FUSED
    matrix shape [hidden, 2*inter] before splitting the halves for NS.
    """
    coeffs = (3.4445, -4.775, 2.0315)
    ns_steps = 5
    eps = 1e-7
    a, b, c = coeffs
    x = delta if delta.ndim == 3 else delta.unsqueeze(0)
    rows, cols = int(x.shape[-2]), int(x.shape[-1])
    o = x.to(compute_dtype)
    transposed = rows > cols
    if transposed:
        o = o.transpose(-2, -1).contiguous()
    norms = o.flatten(start_dim=1).norm(dim=1).clamp(min=eps).reshape(-1, 1, 1)
    o = o / norms
    if o.size(-2) == o.size(-1):
        for _ in range(ns_steps):
            gram = torch.bmm(o, o.transpose(-2, -1))
            gram = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            o = torch.baddbmm(o, gram, o, beta=a)
    else:
        batch = o.size(0)
        gram_r = torch.bmm(o, o.transpose(-2, -1))
        m = gram_r.size(-1)
        identity = torch.eye(m, device=o.device, dtype=o.dtype).unsqueeze(0).expand(batch, -1, -1).contiguous()
        q = None
        for it in range(ns_steps):
            z = torch.baddbmm(gram_r, gram_r, gram_r, beta=b, alpha=c)
            if it == 0:
                q = z + a * identity
            else:
                q = torch.baddbmm(q, q, z, beta=a)
            if it < ns_steps - 1:
                rz = torch.baddbmm(gram_r, gram_r, z, beta=a)
                gram_r = torch.baddbmm(rz, z, rz, beta=a)
        o = torch.bmm(q, o)
    if transposed:
        o = o.transpose(-2, -1)
    o = o.to(delta.dtype)
    if delta.ndim == 2:
        o = o.squeeze(0)
    spectral_rows, spectral_cols = (rows, cols) if spectral_dims is None else spectral_dims
    spectral = 0.2 * (float(max(spectral_rows, spectral_cols)) ** 0.5)
    return o * (float(lr) * spectral)


class _FakeAdapterManager:
    def __init__(self, lora_params, lr=1e-3):
        self._state = SimpleNamespace(lora_params=lora_params)
        self._lr = lr

    def get_adapter_state(self, model_id):
        return self._state

    def get_lr(self, model_id):
        return self._lr


def _make_runner(model, lora_params, tmp_path, *, lora_rank=4, lora_alpha=8):
    """Minimal ModelRunner scaffold driving the REAL fresh_ab base-fold path."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.world_size = 1
    runner.local_rank = 0
    runner.model = model
    runner.train_config = {
        "optimizer": "muon",
        "optimizer_dtype": "fp32",
        "output_dir": str(tmp_path),
        # CPU-friendly, restart-free NS so the inline sglang reference is exact.
        "optimizer_kwargs": {
            "muon_gram_ns_num_restarts": 0,
            "muon_ns_use_quack_kernels": False,
        },
    }
    runner.lora_config = {"enable_lora": True}
    runner.zorl_config = {}
    runner._adapter_manager = _FakeAdapterManager(lora_params)
    runner._lora_session_specs = {
        "default": {
            "base_model": "fake",
            "is_lora": True,
            "lora_config": {"lora_rank": lora_rank, "lora_alpha": lora_alpha},
            "optimizer_config": {"type": "muon", "learning_rate": 1e-3},
        }
    }
    return runner


def test_fresh_ab_end_to_end_muon_apply_moves_base_along_ns_direction(tmp_path):
    torch.manual_seed(0)

    class TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = LoraLinear(32, 24, r=4, lora_alpha=8)
            self.layer1 = LoraLinear(24, 16, r=4, lora_alpha=8)

    model = TwoLayerModel()
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False  # LoRA server mode: base frozen

    lora_params = {
        name: nn.Parameter(param.data.clone()) for name, param in model.named_parameters() if "lora_" in name
    }
    runner = _make_runner(model, lora_params, tmp_path)

    pairs = [(501, 601, 1.0), (502, 602, -0.8), (503, 603, 0.3)]
    lr = 2.5e-4
    scaling = 8.0 / 4.0

    expected_updates, expected_norm = build_zorl_fresh_ab_base_update_from_rewards(
        lora_params, pair_seeds_and_scores=pairs, scaling=scaling
    )

    weights_before = {name: param.data.clone() for name, param in model.named_parameters()}
    update_norm, grad_norm = runner._apply_zorl_fresh_ab_base_update(
        "default", pair_seeds_and_scores=pairs, learning_rate=lr
    )

    assert update_norm == pytest.approx(expected_norm, rel=1e-6)
    assert grad_norm == pytest.approx(expected_norm, rel=1e-6)

    for layer_name in ("layer0", "layer1"):
        weight = dict(model.named_parameters())[f"{layer_name}.weight"]
        applied_delta = weight.data - weights_before[f"{layer_name}.weight"]
        assert applied_delta.abs().max() > 0.0

        expected = _reference_sglang_muon_update(expected_updates[layer_name], lr)
        cos = cosine(applied_delta, expected)
        err = relerr(applied_delta, expected)
        magnitude_ratio = float(applied_delta.norm() / expected.norm())
        print(
            f"fresh_ab e2e {layer_name}: cos={cos:.7f} relerr={err:.3e} "
            f"|delta|/|ref|={magnitude_ratio:.6f} (match_rms_adamw scale included in ref)"
        )
        # Direction: +NS(G) (through param.grad = -G and Muon's -lr*NS(grad)).
        assert cos > 0.99999, f"{layer_name}: cos={cos}"
        # Magnitude: lr * 0.2 * sqrt(max(out, in)) — the match_rms_adamw scale.
        assert err < 1e-4, f"{layer_name}: relerr={err}"

        # Base weights stay frozen for regular training.
        assert not weight.requires_grad
        assert weight.grad is None

    # The parent adapter and the model's LoRA scratch params are untouched.
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert torch.equal(param.data, weights_before[name])
    # Optimizer is cached and reused on the next apply.
    first_optimizer = runner._zorl_fresh_ab_base_optimizer
    runner._apply_zorl_fresh_ab_base_update("default", pair_seeds_and_scores=pairs, learning_rate=lr)
    assert runner._zorl_fresh_ab_base_optimizer is first_optimizer


def test_fresh_ab_apply_zorl_rewards_full_path_folds_base_and_leaves_parent(tmp_path):
    """Full apply_zorl_rewards flow in fresh_ab mode: pair collection with
    shared a_seed, score normalization, base fold, generation completion."""
    torch.manual_seed(1)

    class OneLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = LoraLinear(20, 12, r=4, lora_alpha=8)

    model = OneLayerModel()
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    lora_params = {
        name: nn.Parameter(param.data.clone()) for name, param in model.named_parameters() if "lora_" in name
    }
    # Parent LoRA-B == 0: the fresh_ab invariant (served parent delta is zero).
    lora_params["proj.lora_B"].data.zero_()

    runner = _make_runner(model, lora_params, tmp_path)
    spec = _session_spec("fresh_ab", num_perturbation_pairs=2)
    runner._zorl_sessions = {"default": ZORLSessionState.from_session_spec("default", spec)}

    plan = runner._zorl_sessions["default"].begin_generation()
    rewards = []
    for candidate in plan.candidates:
        rewards.append(
            {
                "candidate_id": candidate.candidate_id,
                "reward_mean": 0.9 if candidate.direction == "positive" else 0.1,
                "num_rollouts": 4,
            }
        )

    base_before = model.proj.weight.data.clone()
    lora_before = {name: param.data.clone() for name, param in lora_params.items()}

    result = runner.apply_zorl_rewards("default", plan.generation_id, rewards, learning_rate=1e-4)

    assert result["applied"] is True
    assert result["used_pairs"] == 2
    assert result["metrics"]["perturbation_mode"] == "fresh_ab"
    assert result["metrics"]["update_norm"] > 0.0
    # Base moved; parent adapter untouched (B stays == 0 forever).
    assert not torch.equal(model.proj.weight.data, base_before)
    for name, param in lora_params.items():
        assert torch.equal(param.data, lora_before[name])
    assert torch.all(lora_params["proj.lora_B"].data == 0.0)
    # Generation completed.
    assert runner._zorl_sessions["default"].active_generation is None


def test_fresh_ab_fold_writes_fused_moe_gate_up_slices(tmp_path):
    """Fused MoE gate_up_proj storage: gate/up updates land in disjoint
    last-dim windows and each expert half moves along its own NS direction."""
    torch.manual_seed(2)
    num_experts, hidden, inter, r = 2, 12, 8, 2

    class FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, 2 * inter), requires_grad=False)
            self.gate_up_proj._fused_gate_up = True
            self.down_proj = nn.Parameter(torch.randn(num_experts, inter, hidden), requires_grad=False)

    class FakeMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FakeExperts()

    model = FakeMoEModel()
    lora_params = {
        "experts.gate_proj_lora_A": nn.Parameter(torch.randn(num_experts, hidden, r)),
        "experts.gate_proj_lora_B": nn.Parameter(torch.zeros(num_experts, r, inter)),
        "experts.up_proj_lora_A": nn.Parameter(torch.randn(num_experts, hidden, r)),
        "experts.up_proj_lora_B": nn.Parameter(torch.zeros(num_experts, r, inter)),
        "experts.down_proj_lora_A": nn.Parameter(torch.randn(num_experts, inter, r)),
        "experts.down_proj_lora_B": nn.Parameter(torch.zeros(num_experts, r, hidden)),
    }
    runner = _make_runner(model, lora_params, tmp_path, lora_rank=r, lora_alpha=2 * r)

    pairs = [(701, 801, 1.0), (702, 802, -0.5)]
    lr = 1e-4
    scaling = (2 * r) / r

    resolution = runner._resolve_zorl_fresh_ab_base_params(
        ["experts.gate_proj", "experts.up_proj", "experts.down_proj"]
    )
    assert resolution["experts.gate_proj"][0] == "experts.gate_up_proj"
    assert resolution["experts.gate_proj"][2] == (0, inter)
    assert resolution["experts.up_proj"][0] == "experts.gate_up_proj"
    assert resolution["experts.up_proj"][2] == (inter, inter)
    assert resolution["experts.down_proj"][0] == "experts.down_proj"
    assert resolution["experts.down_proj"][2] is None

    expected_updates, _ = build_zorl_fresh_ab_base_update_from_rewards(
        lora_params, pair_seeds_and_scores=pairs, scaling=scaling
    )

    gate_up_before = model.experts.gate_up_proj.data.clone()
    down_before = model.experts.down_proj.data.clone()

    runner._apply_zorl_fresh_ab_base_update("default", pair_seeds_and_scores=pairs, learning_rate=lr)

    gate_delta = model.experts.gate_up_proj.data[..., :inter] - gate_up_before[..., :inter]
    up_delta = model.experts.gate_up_proj.data[..., inter:] - gate_up_before[..., inter:]
    down_delta = model.experts.down_proj.data - down_before

    for label, applied, expected_g, spectral_dims in (
        # Muon adjusts the lr from the FUSED [hidden, 2*inter] shape before
        # splitting the gate/up halves for NS.
        ("gate", gate_delta, expected_updates["experts.gate_proj"], (hidden, 2 * inter)),
        ("up", up_delta, expected_updates["experts.up_proj"], (hidden, 2 * inter)),
        ("down", down_delta, expected_updates["experts.down_proj"], None),
    ):
        for expert in range(num_experts):
            expected = _reference_sglang_muon_update(expected_g[expert], lr, spectral_dims=spectral_dims)
            cos = cosine(applied[expert], expected)
            err = relerr(applied[expert], expected)
            magnitude_ratio = float(applied[expert].norm() / expected.norm())
            print(
                f"fresh_ab MoE {label} expert{expert}: cos={cos:.7f} relerr={err:.3e} "
                f"|delta|/|ref|={magnitude_ratio:.6f}"
            )
            # Tolerances are looser than the 2D e2e gate (relerr < 1e-4): these
            # tiny near-square [12, 8] test matrices magnify the residual
            # difference between xorl's Gram Newton-Schulz and the reference's
            # standard NS after 5 steps. Direction and the match_rms_adamw
            # magnitude are still tightly gated.
            assert cos > 0.99999, f"{label} expert {expert}: cos={cos}"
            assert err < 5e-3, f"{label} expert {expert}: relerr={err}"
            assert abs(magnitude_ratio - 1.0) < 5e-3, f"{label} expert {expert}: |delta|/|ref|={magnitude_ratio}"
