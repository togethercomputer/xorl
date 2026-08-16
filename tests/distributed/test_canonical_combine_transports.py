"""The canonical-combine decisive gate.

Feeds the SAME per-(token, slot) expert contributions through artificially
different transport layouts — slot-major compact, expert-major grouped with
per-expert M padding, low-latency fixed-capacity slabs, and a random
permutation with interleaved garbage rows — and asserts BYTE-identical
combined outputs and gradients.

The per-slot contribution bytes are computed ONCE and only PLACED differently
by each transport. That isolates the claim under test (the combine's bytes
are transport-independent by construction) from the expert kernel's per-row
M-geometry invariance, which is a separate contract
(covered by the expert-kernel layout-invariance test).
"""

from __future__ import annotations

import pytest
import torch

from xorl.distributed.canonical_combine import (
    CanonicalCombineError,
    CanonicalRouteMetadata,
    TransportReceipt,
    TransportReceiptError,
    canonical_combine,
    route_metadata_digest,
    validate_transport_receipt,
)


pytestmark = [pytest.mark.distributed]

_T, _K, _E, _H = 37, 8, 32, 256


def _bits(x: torch.Tensor) -> torch.Tensor:
    """BF16 tensors compared as raw bits (NaN payloads and -0.0 included)."""
    assert x.dtype is torch.bfloat16
    return x.contiguous().view(torch.int16)


def _route_program(device: torch.device, seed: int = 1729) -> CanonicalRouteMetadata:
    """A fixed router emission with sorted=True-style slot order and a few drops."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn((_T, _E), generator=generator, dtype=torch.float32)
    weights, ids = torch.topk(scores.softmax(dim=-1), k=_K, dim=-1)
    ids = ids.to(torch.int32)
    # Mark a handful of slots inactive to exercise dropped-slot skip semantics.
    dropped = torch.rand((_T, _K), generator=generator) < 0.05
    dropped[:, 0] = False  # keep every token at least one active slot
    ids = ids.masked_fill(dropped, -1)
    weights = weights.masked_fill(dropped, 0.0)
    return CanonicalRouteMetadata(
        topk_ids=ids.to(device),
        topk_weights=weights.to(device),
        num_experts=_E,
    )


def _slot_major_contribution_table(
    metadata: CanonicalRouteMetadata, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The shared per-slot expert outputs, in canonical slot-major order.

    Returns ``(table [A, H] bf16, slot_tokens [A], slot_experts [A])`` where A
    is the number of active slots. Every transport below places EXACTLY these
    bytes; none recompute them.
    """
    device = metadata.topk_ids.device
    active = metadata.active_mask
    slot_tokens = torch.arange(metadata.num_tokens, device=device).unsqueeze(1).expand_as(active)[active]
    slot_experts = metadata.topk_ids[active].to(torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_inputs = torch.randn((metadata.num_tokens, _H), generator=generator, dtype=torch.float32)
    expert_mats = torch.randn((_E, _H), generator=generator, dtype=torch.float32)
    table = (
        token_inputs.to(device).index_select(0, slot_tokens) * expert_mats.to(device).index_select(0, slot_experts)
    ).to(torch.bfloat16)
    return table.contiguous(), slot_tokens, slot_experts


# ---------------------------------------------------------------------------
# Artificial transports: same bytes, different receive layouts
# ---------------------------------------------------------------------------


def _receipt_from_placement(
    metadata: CanonicalRouteMetadata,
    placement: torch.Tensor,
    num_rows: int,
    slot_tokens: torch.Tensor,
    slot_experts: torch.Tensor,
    transport_id: str,
) -> TransportReceipt:
    device = metadata.topk_ids.device
    active = metadata.active_mask
    slot_to_row = torch.full_like(metadata.topk_ids, -1, dtype=torch.int64)
    slot_to_row[active] = placement
    row_expert_ids = torch.full((num_rows,), -1, dtype=torch.int64, device=device)
    row_source_tokens = torch.full((num_rows,), -1, dtype=torch.int64, device=device)
    row_expert_ids[placement] = slot_experts
    row_source_tokens[placement] = slot_tokens
    return TransportReceipt(
        slot_to_row=slot_to_row,
        num_rows=num_rows,
        row_expert_ids=row_expert_ids,
        row_source_tokens=row_source_tokens,
        transport_id=transport_id,
    )


def _place(table: torch.Tensor, placement: torch.Tensor, num_rows: int, pad_value: float) -> torch.Tensor:
    rows = torch.full((num_rows, table.shape[1]), pad_value, dtype=table.dtype, device=table.device)
    rows.index_copy_(0, placement, table)
    return rows


def transport_slot_major(metadata, table, slot_tokens, slot_experts, *, pad_value: float):
    """Rows arrive in canonical (t, k) order, fully compact. No padding."""
    num_rows = table.shape[0]
    placement = torch.arange(num_rows, dtype=torch.int64, device=table.device)
    receipt = _receipt_from_placement(
        metadata, placement, num_rows, slot_tokens, slot_experts, "sim_slot_major_compact"
    )
    return _place(table, placement, num_rows, pad_value), receipt


def transport_expert_major_padded(metadata, table, slot_tokens, slot_experts, *, pad_value: float, align: int = 8):
    """Expert-major grouping with per-expert M padding to a block multiple.

    This is the normal-DeepEP / moe_align-style geometry: routed rows are
    grouped by expert (stably, preserving token order within an expert) and
    every expert's group is padded to a multiple of ``align`` rows.
    """
    order = torch.argsort(slot_experts, stable=True)
    counts = torch.bincount(slot_experts, minlength=_E)
    padded_counts = ((counts + align - 1) // align) * align
    offsets = torch.cumsum(padded_counts, dim=0) - padded_counts
    within = torch.arange(order.numel(), device=table.device) - (
        (torch.cumsum(counts, dim=0) - counts).index_select(0, slot_experts.index_select(0, order))
    )
    destination = offsets.index_select(0, slot_experts.index_select(0, order)) + within
    placement = torch.empty_like(destination)
    placement[order] = destination
    num_rows = int(padded_counts.sum().item())
    receipt = _receipt_from_placement(
        metadata, placement, num_rows, slot_tokens, slot_experts, "sim_expert_major_padded"
    )
    return _place(table, placement, num_rows, pad_value), receipt


def transport_ll_fixed_slabs(metadata, table, slot_tokens, slot_experts, *, pad_value: float):
    """Low-latency-style fixed-capacity per-expert slabs.

    Every expert owns a fixed slab of ``capacity`` rows regardless of its real
    count (capacity = global max count rounded up to 4); slot rows land at
    ``expert * capacity + arrival``, with arrival order REVERSED within each
    expert to model a different network arrival order than dispatch order.
    """
    counts = torch.bincount(slot_experts, minlength=_E)
    capacity = int((((counts.max() if counts.numel() else torch.tensor(0)) + 3) // 4 * 4).item())
    capacity = max(capacity, 4)
    order = torch.argsort(slot_experts, stable=True)
    within = torch.arange(order.numel(), device=table.device) - (
        (torch.cumsum(counts, dim=0) - counts).index_select(0, slot_experts.index_select(0, order))
    )
    expert_of = slot_experts.index_select(0, order)
    reversed_within = counts.index_select(0, expert_of) - 1 - within
    destination = expert_of * capacity + reversed_within
    placement = torch.empty_like(destination)
    placement[order] = destination
    num_rows = _E * capacity
    receipt = _receipt_from_placement(metadata, placement, num_rows, slot_tokens, slot_experts, "sim_ll_fixed_slabs")
    return _place(table, placement, num_rows, pad_value), receipt


def transport_random_permuted(metadata, table, slot_tokens, slot_experts, *, pad_value: float, seed: int = 99):
    """Adversarial layout: random row permutation plus interleaved garbage rows."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_active = table.shape[0]
    num_rows = num_active + 23
    placement = torch.randperm(num_rows, generator=generator)[:num_active].to(table.device)
    receipt = _receipt_from_placement(metadata, placement, num_rows, slot_tokens, slot_experts, "sim_random_permuted")
    return _place(table, placement, num_rows, pad_value), receipt


_TRANSPORTS = {
    "slot_major": transport_slot_major,
    "expert_major_padded": transport_expert_major_padded,
    "ll_fixed_slabs": transport_ll_fixed_slabs,
    "random_permuted": transport_random_permuted,
}


def _run_all_transports(device: torch.device, backend: str, pad_value: float, with_grads: bool):
    metadata = _route_program(device)
    table, slot_tokens, slot_experts = _slot_major_contribution_table(metadata)
    results = {}
    for name, build in _TRANSPORTS.items():
        rows, receipt = build(metadata, table, slot_tokens, slot_experts, pad_value=pad_value)
        weights = metadata.topk_weights
        if with_grads:
            rows = rows.clone().requires_grad_(True)
            weights = metadata.topk_weights.clone().requires_grad_(True)
            metadata_i = CanonicalRouteMetadata(topk_ids=metadata.topk_ids, topk_weights=weights, num_experts=_E)
        else:
            metadata_i = metadata
        output = canonical_combine(rows, metadata_i, receipt, backend=backend)
        entry = {"output": output.detach(), "receipt": receipt}
        if with_grads:
            # Fixed upstream gradient shared by every transport.
            grad_generator = torch.Generator(device="cpu").manual_seed(1234)
            upstream = torch.randn(output.shape, generator=grad_generator, dtype=torch.float32)
            output.backward(upstream.to(device=device, dtype=output.dtype))
            active = metadata.active_mask
            slot_rows = receipt.slot_to_row[active]
            entry["grad_table"] = rows.grad.detach().index_select(0, slot_rows)
            entry["grad_weights"] = weights.grad.detach()
            # Padding/garbage rows must receive exactly zero gradient.
            claimed = torch.zeros(receipt.num_rows, dtype=torch.bool, device=device)
            claimed[slot_rows] = True
            assert torch.all(rows.grad.detach()[~claimed] == 0), name
        results[name] = entry
    return metadata, results


def _assert_transport_equality(results: dict, *, with_grads: bool) -> None:
    reference_name = "slot_major"
    reference = results[reference_name]
    for name, entry in results.items():
        if name == reference_name:
            continue
        assert torch.equal(_bits(entry["output"]), _bits(reference["output"])), (
            f"combined FORWARD bytes differ: {name} vs {reference_name}"
        )
        if with_grads:
            assert torch.equal(_bits(entry["grad_table"]), _bits(reference["grad_table"])), (
                f"contribution GRADIENT bytes differ: {name} vs {reference_name}"
            )
            assert torch.equal(entry["grad_weights"].view(torch.int32), reference["grad_weights"].view(torch.int32)), (
                f"routing-weight GRADIENT bytes differ: {name} vs {reference_name}"
            )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_transport_independence_forward_and_grads_cpu():
    _, results = _run_all_transports(torch.device("cpu"), "reference", pad_value=0.0, with_grads=True)
    _assert_transport_equality(results, with_grads=True)


@pytest.mark.cpu
def test_padding_rows_are_never_read_cpu():
    """NaN-poisoned padding: any read of an unclaimed row corrupts the output."""
    _, results = _run_all_transports(torch.device("cpu"), "reference", pad_value=float("nan"), with_grads=False)
    _assert_transport_equality(results, with_grads=False)
    for entry in results.values():
        assert torch.all(torch.isfinite(entry["output"].float()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_transport_independence_triton_cuda():
    _, results = _run_all_transports(torch.device("cuda"), "triton", pad_value=float("nan"), with_grads=True)
    _assert_transport_equality(results, with_grads=True)
    for entry in results.values():
        assert torch.all(torch.isfinite(entry["output"].float()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reference_vs_triton_backend_parity_cuda():
    """Do the two independent implementations of the canonical tree agree bitwise?

    Triton contracts the product-accumulate into FMA, so the reference uses
    an fp64-emulated FMA chain. The production-hidden test covers the
    deployment geometry as well as this compact fixture.
    """
    device = torch.device("cuda")
    metadata = _route_program(device)
    table, slot_tokens, slot_experts = _slot_major_contribution_table(metadata)
    rows, receipt = transport_expert_major_padded(metadata, table, slot_tokens, slot_experts, pad_value=0.0)
    out_reference = canonical_combine(rows, metadata, receipt, backend="reference")
    out_triton = canonical_combine(rows, metadata, receipt, backend="triton")
    assert torch.equal(_bits(out_reference), _bits(out_triton)), (
        "reference and triton canonical-combine backends diverge bitwise; "
        "the executed serving kernel, not the torch formula, is the contract"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_backend_parity_and_identity_at_production_hidden_cuda():
    """H=2048 (the Qwen3.6 hidden size; Triton BLOCK_D=1024) backend parity.

    This test pins reference == Triton at H=2048, and when the paired SGLang install is
    importable additionally pins identity with the real ep_gather kernel at
    this shape.
    """
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(41)
    hidden, tokens, experts = 2048, 96, 64
    ids = torch.stack([torch.randperm(experts, generator=generator)[:_K] for _ in range(tokens)])
    ids = ids.to(torch.int32).to(device)
    weights = torch.rand((tokens, _K), generator=generator).softmax(dim=-1).to(torch.float32).to(device)
    rows = torch.randn((tokens * _K, hidden), generator=generator).to(torch.bfloat16).to(device)
    metadata = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=experts)
    receipt = TransportReceipt(
        slot_to_row=torch.arange(tokens * _K, device=device).view(tokens, _K),
        num_rows=tokens * _K,
        row_expert_ids=ids.reshape(-1).to(torch.int64),
        row_source_tokens=torch.arange(tokens, device=device).repeat_interleave(_K),
        transport_id="sim_production_hidden",
    )
    reference = canonical_combine(rows, metadata, receipt, backend="reference")
    triton_out = canonical_combine(rows, metadata, receipt, backend="triton")
    assert torch.equal(_bits(reference), _bits(triton_out)), (
        "reference and triton diverge at H=2048 — the FMA-chain reference correction regressed"
    )
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    except ImportError:
        return
    serving = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    ep_gather(
        rows.contiguous(),
        ids.contiguous(),
        weights.contiguous(),
        receipt.slot_to_row.to(torch.int32).contiguous(),
        serving,
    )
    assert torch.equal(_bits(reference), _bits(serving)), (
        "canonical_combine_v1 does not match the real ep_gather bytes at H=2048"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sglang_ep_gather_is_the_canonical_program_cuda():
    """The pinned serving kernel must compute exactly canonical_combine_v1."""
    kernels = pytest.importorskip("sglang.kernels.ops.moe.ep_moe_kernels")
    device = torch.device("cuda")
    metadata = _route_program(device)
    table, slot_tokens, slot_experts = _slot_major_contribution_table(metadata)
    rows, receipt = transport_ll_fixed_slabs(metadata, table, slot_tokens, slot_experts, pad_value=float("nan"))
    ours = canonical_combine(rows, metadata, receipt, backend="triton")

    serving = torch.empty((metadata.num_tokens, _H), dtype=torch.bfloat16, device=device)
    kernels.ep_gather(
        rows.contiguous(),
        metadata.topk_ids.contiguous(),
        metadata.topk_weights.contiguous(),
        receipt.slot_to_row.to(torch.int32).contiguous(),
        serving,
    )
    assert torch.equal(_bits(ours), _bits(serving)), (
        "canonical_combine_v1 does not match the pinned sglang ep_gather bytes"
    )


# ---------------------------------------------------------------------------
# Fail-closed admission
# ---------------------------------------------------------------------------


def _cpu_fixture():
    metadata = _route_program(torch.device("cpu"))
    table, slot_tokens, slot_experts = _slot_major_contribution_table(metadata)
    rows, receipt = transport_slot_major(metadata, table, slot_tokens, slot_experts, pad_value=0.0)
    return metadata, table, slot_tokens, slot_experts, rows, receipt


@pytest.mark.cpu
def test_receipt_swapped_tokens_same_expert_fails_closed():
    """Two rows of the same expert swapped between tokens: caught by row source tokens."""
    metadata, table, slot_tokens, slot_experts, rows, receipt = _cpu_fixture()
    expert_counts = torch.bincount(slot_experts, minlength=_E)
    expert = int(torch.nonzero(expert_counts >= 2)[0].item())
    victims = torch.nonzero(slot_experts == expert).reshape(-1)[:2]
    swapped = receipt.slot_to_row.clone()
    flat_active = torch.nonzero(metadata.active_mask.reshape(-1)).reshape(-1)
    a, b = flat_active[victims[0]], flat_active[victims[1]]
    swapped.reshape(-1)[a], swapped.reshape(-1)[b] = (
        receipt.slot_to_row.reshape(-1)[b].item(),
        receipt.slot_to_row.reshape(-1)[a].item(),
    )
    bad = TransportReceipt(
        slot_to_row=swapped,
        num_rows=receipt.num_rows,
        row_expert_ids=receipt.row_expert_ids,
        row_source_tokens=receipt.row_source_tokens,
        transport_id="sim_swapped_tokens",
    )
    with pytest.raises(TransportReceiptError, match="source token"):
        validate_transport_receipt(metadata, bad)


@pytest.mark.cpu
def test_receipt_expert_mismatch_fails_closed():
    metadata, _, _, _, _, receipt = _cpu_fixture()
    lying = receipt.row_expert_ids.clone()
    victim = int(torch.nonzero(lying >= 0)[0].item())
    lying[victim] = (lying[victim] + 1) % _E
    bad = TransportReceipt(
        slot_to_row=receipt.slot_to_row,
        num_rows=receipt.num_rows,
        row_expert_ids=lying,
        row_source_tokens=receipt.row_source_tokens,
        transport_id="sim_expert_lie",
    )
    with pytest.raises(TransportReceiptError, match="expert id"):
        validate_transport_receipt(metadata, bad)


@pytest.mark.cpu
def test_receipt_noninjective_mapping_fails_closed():
    metadata, _, _, _, _, receipt = _cpu_fixture()
    doubled = receipt.slot_to_row.clone()
    flat_active = torch.nonzero(metadata.active_mask.reshape(-1)).reshape(-1)
    doubled.reshape(-1)[flat_active[1]] = doubled.reshape(-1)[flat_active[0]]
    bad = TransportReceipt(
        slot_to_row=doubled,
        num_rows=receipt.num_rows,
        row_expert_ids=receipt.row_expert_ids,
        row_source_tokens=receipt.row_source_tokens,
        transport_id="sim_noninjective",
    )
    with pytest.raises(TransportReceiptError, match="injectivity"):
        validate_transport_receipt(metadata, bad)


@pytest.mark.cpu
def test_receipt_dropped_real_row_fails_closed():
    """A delivered real contribution nothing claims: silent-drop must RAISE."""
    metadata, table, slot_tokens, slot_experts, _, _ = _cpu_fixture()
    num_rows = table.shape[0] + 1
    placement = torch.arange(table.shape[0], dtype=torch.int64)
    receipt = _receipt_from_placement(metadata, placement, num_rows, slot_tokens, slot_experts, "sim_dropped_real")
    stray = receipt.row_expert_ids.clone()
    stray_tokens = receipt.row_source_tokens.clone()
    stray[-1] = 0
    stray_tokens[-1] = 0  # claims to be a real (token 0, expert 0) contribution
    bad = TransportReceipt(
        slot_to_row=receipt.slot_to_row,
        num_rows=num_rows,
        row_expert_ids=stray,
        row_source_tokens=stray_tokens,
        transport_id="sim_dropped_real",
    )
    with pytest.raises(TransportReceiptError, match="claimed by no slot"):
        validate_transport_receipt(metadata, bad)


@pytest.mark.cpu
def test_slot_reorder_is_a_different_program():
    """Re-sorting the top-k slots is a DIFFERENT route program, not a transport.

    A transport may deliver rows in any order (that is the whole point), but
    an engine that consumes slots in a different order (e.g. expert-ascending
    instead of router emission order) executes a different FP32 accumulation
    tree. The contract-level defense is the route-metadata digest exchanged
    between engines, which must differ — asserted unconditionally below.

    Because the accumulator is FP32 and only the terminal store rounds to
    BF16, many reorder-induced FP32 differences are absorbed by that final
    rounding. The demonstration therefore searches a few deterministic seeds
    with exponent-diverse contributions and requires at least one byte flip;
    the route-metadata digest remains mandatory whenever order can matter.
    """
    tokens, hidden = 256, 512
    flipped_elements = 0
    total_elements = 0
    digest_checked = False
    for seed in range(6):
        generator = torch.Generator(device="cpu").manual_seed(1000 + seed)
        scores = torch.randn((tokens, _E), generator=generator, dtype=torch.float32)
        weights, ids = torch.topk(scores.softmax(dim=-1), k=_K, dim=-1)
        metadata = CanonicalRouteMetadata(topk_ids=ids.to(torch.int32), topk_weights=weights, num_experts=_E)
        active = metadata.active_mask
        slot_tokens = torch.arange(tokens).unsqueeze(1).expand_as(active)[active]
        slot_experts = metadata.topk_ids[active].to(torch.int64)
        # Exponent-diverse contributions raise the boundary-crossing rate.
        exponents = torch.randint(-6, 7, (slot_tokens.numel(), 1), generator=generator)
        table = (torch.randn((slot_tokens.numel(), hidden), generator=generator) * (2.0**exponents)).to(torch.bfloat16)

        rows, receipt = transport_slot_major(metadata, table, slot_tokens, slot_experts, pad_value=0.0)
        baseline = canonical_combine(rows, metadata, receipt, backend="reference")

        order = torch.argsort(metadata.topk_ids.masked_fill(~active, _E + 1), dim=1)
        resorted = CanonicalRouteMetadata(
            topk_ids=torch.gather(metadata.topk_ids, 1, order),
            topk_weights=torch.gather(metadata.topk_weights, 1, order),
            num_experts=_E,
        )
        assert route_metadata_digest(resorted) != route_metadata_digest(metadata)
        digest_checked = True

        resorted_receipt = TransportReceipt(
            slot_to_row=torch.gather(receipt.slot_to_row, 1, order),
            num_rows=receipt.num_rows,
            row_expert_ids=receipt.row_expert_ids,
            row_source_tokens=receipt.row_source_tokens,
            transport_id="sim_resorted_slots",
        )
        validate_transport_receipt(resorted, resorted_receipt)  # internally consistent...
        reordered = canonical_combine(rows, resorted, resorted_receipt, backend="reference")
        flipped_elements += int((_bits(baseline) != _bits(reordered)).sum().item())
        total_elements += baseline.numel()

    assert digest_checked
    assert flipped_elements > 0, (
        "expected at least one BF16 byte flip from slot reordering across seeds; "
        f"observed 0 in {total_elements} elements — re-examine the order-sensitivity claim"
    )
    print(f"slot-reorder byte flip rate: {flipped_elements}/{total_elements} ({flipped_elements / total_elements:.2e})")


@pytest.mark.cpu
def test_no_validation_bypass_duplicate_rows_raise_at_combine():
    """The combine itself must fail closed on a non-injective receipt.

    A duplicate row claim would make the backward's index_copy_ target the
    same row twice — overwrite/order-dependent gradients. There must be no
    public switch that skips admission: the signature is asserted too.
    """
    import inspect

    metadata, _, _, _, rows, receipt = _cpu_fixture()
    doubled = receipt.slot_to_row.clone()
    flat_active = torch.nonzero(metadata.active_mask.reshape(-1)).reshape(-1)
    doubled.reshape(-1)[flat_active[1]] = doubled.reshape(-1)[flat_active[0]]
    bad = TransportReceipt(
        slot_to_row=doubled,
        num_rows=receipt.num_rows,
        row_expert_ids=receipt.row_expert_ids,
        row_source_tokens=receipt.row_source_tokens,
        transport_id="sim_noninjective_at_combine",
    )
    with pytest.raises(TransportReceiptError, match="injectivity"):
        canonical_combine(rows, metadata, bad, backend="reference")
    assert "validate" not in inspect.signature(canonical_combine).parameters, (
        "canonical_combine must not expose a validation bypass"
    )


def _dense_autograd_oracle(rows_bf16, metadata, receipt, upstream):
    """Independent GRADIENT oracle: the canonical program's math as a pure
    composition of stock differentiable torch ops (index_select / where /
    add / cast), differentiated by torch.autograd — no custom Function.

    Comparison contract (labels used by the asserting tests):
    - d(topk_weights): BYTE-exact FP32 (same per-row H reduction);
    - d(contributions): BYTE-exact after signed-zero normalization
      (x + 0.0), because autograd accumulates the K per-slot scatter
      gradients with BF16 adds whose exact-zero terms rewrite -0.0 to
      +0.0, while the one-writer custom backward preserves -0.0;
    - forward: NUMERICAL sanity only (row-relative L2), NOT bytes. The
      canonical forward is an FMA chain, while this oracle's forward uses
      mul-then-add so that its BACKWARD reproduces the trainer-owned
      rounded-product gradient program. One composition cannot be both;
      bitwise FORWARD parity is owned by the CUDA backend/identity gates.
    """
    rows = rows_bf16.detach().clone().requires_grad_(True)
    weights = metadata.topk_weights.detach().clone().requires_grad_(True)
    active = metadata.active_mask
    safe = receipt.slot_to_row.to(torch.int64).clamp_min(0)
    acc = torch.zeros((metadata.num_tokens, rows.shape[1]), dtype=torch.float32, device=rows.device)
    for k in range(metadata.topk):
        values = rows.index_select(0, safe[:, k]).to(torch.float32) * weights[:, k].unsqueeze(1)
        acc = acc + torch.where(active[:, k].unsqueeze(1), values, torch.zeros_like(values))
    out = acc.to(torch.bfloat16)
    out.backward(upstream)
    return out.detach(), rows.grad.detach(), weights.grad.detach()


@pytest.mark.cpu
def test_backward_matches_independent_dense_autograd_oracle():
    """Custom-Function gradients vs an independent torch.autograd oracle.

    The transport-equality gates compare the custom autograd function
    against itself across layouts; this gate compares it against a
    separately-derived gradient program for every transport layout.
    """
    metadata = _route_program(torch.device("cpu"))
    table, slot_tokens, slot_experts = _slot_major_contribution_table(metadata)
    upstream_generator = torch.Generator(device="cpu").manual_seed(777)
    upstream = torch.randn((metadata.num_tokens, _H), generator=upstream_generator).to(torch.bfloat16)

    for name, build in _TRANSPORTS.items():
        rows, receipt = build(metadata, table, slot_tokens, slot_experts, pad_value=0.0)
        rows_custom = rows.clone().requires_grad_(True)
        weights_custom = metadata.topk_weights.clone().requires_grad_(True)
        metadata_custom = CanonicalRouteMetadata(
            topk_ids=metadata.topk_ids, topk_weights=weights_custom, num_experts=_E
        )
        out = canonical_combine(rows_custom, metadata_custom, receipt, backend="reference")
        out.backward(upstream)

        oracle_out, oracle_grad_rows, oracle_grad_weights = _dense_autograd_oracle(rows, metadata, receipt, upstream)
        out_f32 = out.detach().float()
        oracle_f32 = oracle_out.float()
        norms = out_f32.norm(dim=1).clamp_min(1.0)
        rel = (out_f32 - oracle_f32).norm(dim=1) / norms
        assert float(rel.max()) < 1e-2, f"{name}: forward not numerically close to oracle"
        assert torch.equal(weights_custom.grad.view(torch.int32), oracle_grad_weights.view(torch.int32)), (
            f"{name}: d(topk_weights) FP32 bytes vs oracle"
        )
        assert torch.equal(_bits(rows_custom.grad + 0.0), _bits(oracle_grad_rows + 0.0)), (
            f"{name}: d(contributions) bytes vs oracle (signed-zero normalized)"
        )


@pytest.mark.cpu
def test_metadata_rejects_duplicate_active_expert_and_bad_dtypes():
    ids = torch.tensor([[3, 3, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    with pytest.raises(CanonicalCombineError, match="repeats an active expert"):
        CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=8)
    with pytest.raises(CanonicalCombineError, match="FP32"):
        CanonicalRouteMetadata(
            topk_ids=torch.tensor([[1, 2, 4]], dtype=torch.int32),
            topk_weights=weights.to(torch.bfloat16),
            num_experts=8,
        )


@pytest.mark.cpu
def test_combine_rejects_non_bf16_contributions():
    metadata, _, _, _, rows, receipt = _cpu_fixture()
    with pytest.raises(CanonicalCombineError, match="BF16"):
        canonical_combine(rows.float(), metadata, receipt, backend="reference")


@pytest.mark.cpu
def test_inactive_slots_never_contaminate_even_with_poisoned_padding():
    """Skip semantics, made observable: an inactive slot performs NO read.

    Note the -0.0 observable does NOT distinguish skip from add-zero here:
    the FP32 accumulator initializes to +0.0, and IEEE round-to-nearest makes
    +0.0 + (-0.0) = +0.0, so a -0.0 accumulator can never arise. What DOES
    distinguish them is that an add-zero
    implementation must read SOME row for the inactive slot; with NaN
    padding, 0.0 * NaN = NaN would poison the token. The canonical program
    must produce the active-slots-only bytes.
    """
    ids = torch.tensor([[5, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    metadata = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=8)
    rows = torch.full((2, 8), float("nan"), dtype=torch.bfloat16)
    rows[1] = torch.arange(8, dtype=torch.bfloat16)
    receipt = TransportReceipt(
        slot_to_row=torch.tensor([[1, -1]], dtype=torch.int64),
        num_rows=2,
        row_expert_ids=torch.tensor([-1, 5], dtype=torch.int64),
        row_source_tokens=torch.tensor([-1, 0], dtype=torch.int64),
        transport_id="sim_poisoned_padding",
    )
    out = canonical_combine(rows, metadata, receipt, backend="reference")
    expected = (rows[1].float() * 0.5).to(torch.bfloat16).unsqueeze(0)
    assert torch.equal(_bits(out), _bits(expected))
