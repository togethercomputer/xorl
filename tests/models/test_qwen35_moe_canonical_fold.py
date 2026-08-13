"""Byte and grad gates for the Qwen exact MoE canonical contributor fold.

The exact Qwen3.5-MoE cross-rank combine is ``canonical_moe_fold_v1``: the
balanced adjacent-pair BF16 tree, rounding to nearest-even after every add,
on both engines (serving folds after an all-gather; the trainer folds after
a raw all-to-all). These gates pin the three facts the pairing depends on:

1. the trainer fold reproduces the serving fold arithmetic bitwise on
   matched partials (transcribed reference here, direct sglang cross-check
   when the serving package is importable — the production implementations
   stay deliberately independent so the pair cannot self-confirm);
2. the fold is NOT the retired reverse-rank chain: the deterministic fixture
   requires the two programs to differ in at least one element, so a silent
   regression to the chain reads as a byte failure here;
3. the combine participates in the training graph, so it carries a
   grad-engagement gate: every contributor receives a non-None, finite
   gradient matching the reference trajectory through
   ``exchange_and_canonical_fold``.
"""

from __future__ import annotations

import pytest
import torch

from xorl.distributed.canonical_moe import (
    CANONICAL_MOE_FOLD_VERSION,
    canonical_moe_fold_v1,
)


pytestmark = [pytest.mark.cpu]

EP_SIZE = 8


def _serving_fold_reference(partials: torch.Tensor) -> torch.Tensor:
    """Serving's canonical_moe_fold_v1 arithmetic, transcribed independently.

    sglang canonical_moe.py ``_balanced_adjacent_tree``: pair adjacent
    contributors level by level, rounding to bf16 after every add.
    """
    level = [partials[index] for index in range(partials.shape[0])]
    while len(level) > 1:
        level = [
            (level[index] + level[index + 1]).to(torch.bfloat16)
            for index in range(0, len(level), 2)
        ]
    return level[0]


def _retired_chain_reference(partials: torch.Tensor) -> torch.Tensor:
    """The RETIRED pre-unification program (reverse-rank sequential chain).

    Kept only as the negative reference: serving deleted it in the fold
    unification (``tensor_model_parallel_ordered_all_reduce``), and the
    trainer half was retired by this change.
    """
    acc = partials[partials.shape[0] - 1]
    for rank in range(partials.shape[0] - 2, -1, -1):
        acc = acc + partials[rank]
    return acc


def _partials(rows: int = 512, hidden: int = 1024, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(EP_SIZE, rows, hidden, generator=generator) * 0.05
    return raw.to(torch.bfloat16)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint16)


def test_fold_version_tag():
    assert CANONICAL_MOE_FOLD_VERSION == "canonical_moe_fold_v1"


def test_trainer_fold_matches_serving_fold_arithmetic():
    partials = _partials()
    assert torch.equal(
        _bits(canonical_moe_fold_v1(partials)),
        _bits(_serving_fold_reference(partials)),
    )


def test_trainer_fold_matches_live_serving_fold_bitwise():
    """Cross-engine byte gate whenever the serving package is importable."""
    serving_canonical = pytest.importorskip(
        "sglang.srt.distributed.canonical_moe",
        reason="serving package not importable in this venv",
    )
    partials = _partials()
    serving_folded = serving_canonical.canonical_moe_fold_v1(partials.clone())
    assert torch.equal(_bits(canonical_moe_fold_v1(partials)), _bits(serving_folded))


def test_fold_is_not_the_retired_chain():
    """The negative gate: chain and fold must remain distinguishable.

    A fixture that stopped exercising the rounding boundary would trivialize
    every positive gate in this file, so demand a substantial discrimination
    profile rather than merely one differing element.
    """
    partials = _partials()
    fold = canonical_moe_fold_v1(partials)
    chain = _retired_chain_reference(partials)
    differing = int((_bits(fold) != _bits(chain)).sum())
    assert differing > 0.2 * fold.numel(), (
        f"chain-vs-fold discrimination collapsed: {differing}/{fold.numel()}"
    )


def test_fold_pairing_structure_is_the_program():
    """The fold is invariant under full contributor reversal (mirror
    symmetry plus commutative bf16 adds) but NOT under a regrouping that
    changes which contributors meet at the first tree level — so
    contributor identity/placement, not traversal direction, is the
    contract."""
    partials = _partials()
    assert torch.equal(
        _bits(canonical_moe_fold_v1(partials)),
        _bits(canonical_moe_fold_v1(partials.flip(0))),
    )
    regrouped = partials[[0, 2, 1, 3, 4, 6, 5, 7]]
    assert not torch.equal(
        _bits(canonical_moe_fold_v1(partials)),
        _bits(canonical_moe_fold_v1(regrouped)),
    )


def test_exchange_and_canonical_fold_grad_engagement(monkeypatch):
    """A training-graph combine ships with a grad-engagement gate.

    The exchange is identity-mocked (its own grad
    all-to-all is covered by the collective-level tests); the gate checks
    the fold's forward bytes and that every contributor row receives a
    non-None, finite gradient equal to the reference trajectory."""
    import xorl.distributed.moe.comm as moe_comm  # noqa: PLC0415
    from xorl.models.layers.moe.ep_native_combine import (  # noqa: PLC0415
        exchange_and_canonical_fold,
    )

    class IdentityExchange:
        @staticmethod
        def apply(group, value, *args):
            del group, args
            return value

    monkeypatch.setattr(moe_comm, "_AllToAll", IdentityExchange)

    rows, hidden = 16, 32
    partial = (
        (torch.randn(EP_SIZE * rows, hidden, generator=torch.Generator().manual_seed(3)) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    out = exchange_and_canonical_fold(partial, group=None, ep_size=EP_SIZE)

    reference = _serving_fold_reference(
        partial.detach().view(EP_SIZE, rows, hidden)
    )
    assert torch.equal(_bits(out), _bits(reference)), (
        "exchange_and_canonical_fold forward != canonical fold reference bytes"
    )

    grad_out = torch.randn(rows, hidden, generator=torch.Generator().manual_seed(4)).to(
        torch.bfloat16
    )
    out.backward(grad_out)

    assert partial.grad is not None, "combine input received no grad"
    grads = partial.grad.view(EP_SIZE, rows, hidden)
    assert torch.isfinite(grads.float()).all(), "combine grad not finite"
    for contributor in range(EP_SIZE):
        # d(fold)/d(p_i) is the identity for every contributor: bf16 add
        # backward is pass-through, so the reference trajectory is grad_out.
        assert torch.equal(_bits(grads[contributor]), _bits(grad_out)), (
            f"contributor {contributor} grad diverged from the reference trajectory"
        )
