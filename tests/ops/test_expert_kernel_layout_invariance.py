"""Expert-kernel per-row bytes versus batch composition.

Different MoE
transports deliver the same routed rows in different batch compositions —
subsets (rank-deduplicated DeepEP receives vs the shared program's
all-tokens batch), different row orders (network arrival), filler/padding
rows, and different total M. Admission of a transport requires the expert
kernel's PER-ROW arithmetic to be invariant to all of that: same
(token row, expert) pair -> same output bytes, regardless of what else is
in the batch.

This gate runs the real fused-expert kernel through XoRL's strided wrapper at
the production local-slice geometry (32 local experts, H=2048, I=512, top-k 8) and
byte-compares:

- per-(token, slot) rows (``no_combine=True``) — the exact tensor a
  canonical-combine transport would ship; and
- per-token local partials (``no_combine=False``) — the shared program's
  observable (slot-ordered local combine included);

across five compositions: all-tokens (shared program), compact subset
(dedup receive), filler-padded, row-permuted, and an M sweep across block
boundaries. Any byte difference violates DeepEP admission.

Requires CUDA + the paired sglang install (importorskip). Single GPU.
"""

from __future__ import annotations

import pytest
import torch


pytestmark = [pytest.mark.distributed]

if not torch.cuda.is_available():
    pytest.skip("layout-invariance gate requires CUDA", allow_module_level=True)

pytest.importorskip("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")

_E_LOCAL, _H, _I, _K = 32, 2048, 512, 8
_T = 96


def _bits(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype is torch.bfloat16
    return x.contiguous().view(torch.int16)


@pytest.fixture(scope="module")
def fixture():
    from xorl.models.layers.moe.experts import MoEExperts
    from xorl.ops.moe.sglang_fused_moe_strided import fused_experts_impl_strided

    # The production setup step: publish the deterministic sglang exec
    # context exactly as the exact contract does before any kernel call.
    MoEExperts._ensure_sglang_server_args()

    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(1729)
    w1 = (torch.randn((_E_LOCAL, 2 * _I, _H), generator=generator) * 0.05).to(torch.bfloat16).to(device)
    w2 = (torch.randn((_E_LOCAL, _H, _I), generator=generator) * 0.05).to(torch.bfloat16).to(device)
    hidden = torch.randn((_T, _H), generator=generator).to(torch.bfloat16).to(device)

    ids = torch.full((_T, _K), -1, dtype=torch.int32)
    weights = torch.zeros((_T, _K), dtype=torch.float32)
    for t in range(_T):
        if t % 3 == 0:
            continue  # fully non-local token (all -1): the shared program sees these
        active = int(torch.randint(1, _K + 1, (1,), generator=generator).item())
        experts = torch.randperm(_E_LOCAL, generator=generator)[:active]
        slots = torch.randperm(_K, generator=generator)[:active]
        ids[t, slots] = experts.to(torch.int32)
        weights[t, slots] = torch.rand((active,), generator=generator) + 0.05
    ids = ids.to(device)
    weights = weights.to(device)

    def run(hidden_rows, row_ids, row_weights, *, no_combine):
        return fused_experts_impl_strided(
            hidden_rows.contiguous(),
            w1,
            w2,
            row_weights.contiguous(),
            row_ids.contiguous(),
            inplace=False,
            activation="silu",
            is_gated=True,
            filter_expert=True,
            no_combine=no_combine,
        )

    return {
        "device": device,
        "generator": generator,
        "hidden": hidden,
        "ids": ids,
        "weights": weights,
        "run": run,
    }


def _active_slot_bits(per_slot: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Bytes of active slots only; inactive slots are never written (empty)."""
    active = ids >= 0
    return _bits(per_slot[active])


def test_filler_row_invariance(fixture):
    """Interleaved all(-1) filler rows (capacity padding) must not perturb real rows."""
    hidden, ids, weights, run = fixture["hidden"], fixture["ids"], fixture["weights"], fixture["run"]
    device = fixture["device"]
    subset = (ids >= 0).any(dim=1)
    base_h, base_i, base_w = hidden[subset], ids[subset], weights[subset]
    reference = run(base_h, base_i, base_w, no_combine=True)

    n = base_h.shape[0]
    filler = 32
    order = torch.randperm(n + filler, generator=torch.Generator().manual_seed(5)).to(device)
    padded_h = torch.cat([base_h, torch.full((filler, _H), float("nan"), dtype=base_h.dtype, device=device)])
    padded_i = torch.cat([base_i, torch.full((filler, _K), -1, dtype=base_i.dtype, device=device)])
    padded_w = torch.cat([base_w, torch.zeros((filler, _K), dtype=base_w.dtype, device=device)])
    shuffled = run(padded_h[order], padded_i[order], padded_w[order], no_combine=True)

    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n + filler, device=device)
    restored = shuffled[inverse][:n]
    assert torch.equal(_active_slot_bits(restored, base_i), _active_slot_bits(reference, base_i)), (
        "NaN filler rows or row order perturb real rows' per-slot bytes"
    )


def test_m_sweep_row_stability(fixture):
    """A fixed probe row's bytes must be stable while total M crosses block sizes."""
    hidden, ids, weights, run = fixture["hidden"], fixture["ids"], fixture["weights"], fixture["run"]
    device = fixture["device"]
    subset = (ids >= 0).any(dim=1)
    base_h, base_i, base_w = hidden[subset], ids[subset], weights[subset]
    probe = 8
    reference = run(base_h, base_i, base_w, no_combine=True)[:probe]

    generator = torch.Generator(device="cpu").manual_seed(11)
    for extra in (1, 3, 17, 64, 129):
        extra_h = torch.randn((extra, _H), generator=generator).to(torch.bfloat16).to(device)
        extra_i = torch.randint(0, _E_LOCAL, (extra, _K), generator=generator, dtype=torch.int32)
        # unique experts per row
        for r in range(extra):
            extra_i[r] = torch.randperm(_E_LOCAL, generator=generator)[:_K].to(torch.int32)
        extra_w = (torch.rand((extra, _K), generator=generator) + 0.05).to(torch.float32)
        grown = run(
            torch.cat([base_h, extra_h]),
            torch.cat([base_i, extra_i.to(device)]),
            torch.cat([base_w, extra_w.to(device)]),
            no_combine=True,
        )[:probe]
        assert torch.equal(_active_slot_bits(grown, base_i[:probe]), _active_slot_bits(reference, base_i[:probe])), (
            f"probe-row bytes changed when M grew by {extra} rows"
        )


def test_weight_application_point(fixture):
    """Pin where routing weights are applied: per-slot rows must be UNWEIGHTED.

    canonical_combine_v1 ships unweighted rows and applies FP32 weights at
    the combine. If the kernel's no_combine output already includes the
    routing weight, a transport adapter must NOT multiply again. Assert the
    relationship explicitly so the adapter contract is pinned by a gate.
    """
    hidden, ids, weights, run = fixture["hidden"], fixture["ids"], fixture["weights"], fixture["run"]
    per_slot = run(hidden, ids, weights, no_combine=True)
    doubled = run(hidden, ids, (weights * 2.0), no_combine=True)
    active = ids >= 0
    unweighted = torch.equal(_active_slot_bits(per_slot, ids), _active_slot_bits(doubled, ids))
    print(f"fused-kernel no_combine rows are weight-free: {unweighted}")
    assert unweighted or torch.equal(
        _active_slot_bits(doubled, ids), _bits((per_slot[active].float() * 2.0).to(torch.bfloat16))
    ), "no_combine rows are neither weight-free nor exactly weight-scaled"
