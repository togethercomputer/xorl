from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from distributed_utils import run_distributed_script

from xorl.distributed.canonical_moe import (
    _CANONICAL_MOE_DENSE_MAX_BUFFER_BYTES,
    _CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES,
    CANONICAL_MOE_DENSE_MAX_CHUNK_ROWS,
    CANONICAL_MOE_FOLD_VERSION,
    CANONICAL_MOE_LEAF_VERSION,
    CANONICAL_MOE_REDUCE_VERSION,
    CanonicalMoEGraphMetadata,
    CanonicalMoETransport,
    LocalMoEContribution,
    LogicalRowOwnership,
    OutputDistribution,
    ParallelPlan,
    ParallelRole,
    _canonical_moe_fold_fp32_tree,
    _canonical_moe_fp32_fold_chunk_elements,
    _resolve_transport_chunk_rows,
    _RuntimePlan,
    _transport_and_fold,
    canonical_moe_fold_fp32_v2,
    canonical_moe_leaf_fp32_v1,
    canonical_moe_reduce_cp_sharded_v3,
    canonical_moe_reduce_fp32_v2,
    canonical_moe_reduce_packed_ep16_v2,
    canonical_moe_reduce_reference,
    resolve_canonical_moe_transport,
)
from xorl.distributed.parallel_state import init_ep_mesh_matrix


pytestmark = [pytest.mark.distributed]


@pytest.mark.cpu
def test_dense_transport_default_bounds_dp_owned_capacity_without_a_selector():
    assert CANONICAL_MOE_DENSE_MAX_CHUNK_ROWS == 4096
    assert _CANONICAL_MOE_DENSE_MAX_BUFFER_BYTES == 32 * 1024 * 1024
    planner_inputs = {
        "contributor_count": 16,
        "payload_elements_per_row": 6144,
        "element_size": 2,
    }
    assert (
        _resolve_transport_chunk_rows(
            16640,
            None,
            CanonicalMoETransport.DENSE_V1,
            **planner_inputs,
        )
        == 170
    )
    assert (
        _resolve_transport_chunk_rows(
            16640,
            2048,
            CanonicalMoETransport.DENSE_V1,
            **planner_inputs,
        )
        == 170
    )
    assert (
        _resolve_transport_chunk_rows(
            16640,
            64,
            CanonicalMoETransport.DENSE_V1,
            **planner_inputs,
        )
        == 64
    )
    assert (
        _resolve_transport_chunk_rows(
            128,
            None,
            CanonicalMoETransport.DENSE_V1,
            contributor_count=16,
            payload_elements_per_row=8,
            element_size=2,
        )
        == 128
    )
    assert (
        _resolve_transport_chunk_rows(
            66544,
            None,
            CanonicalMoETransport.PACKED_EP16_V2,
            **planner_inputs,
        )
        == 66544
    )


def _explicit_tree(partials: torch.Tensor) -> torch.Tensor:
    current = [partials[index].float() for index in range(partials.shape[0])]
    while len(current) > 1:
        next_level = [current[index] + current[index + 1] for index in range(0, len(current) - 1, 2)]
        if len(current) % 2:
            next_level.append(current[-1])
        current = next_level
    return current[0].to(partials.dtype)


def _one_round_leaf_oracle(
    shared: torch.Tensor,
    routed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Independent pre-cast scalar program for one FP32 FMA rounding."""
    scale_fp32 = torch.tensor(scale, dtype=torch.float32)
    return (shared.cpu().double() + routed.cpu().double() * scale_fp32.double()).float()


@pytest.mark.cpu
@pytest.mark.parametrize("contributors", [1, 2, 3, 4, 5, 6, 8, 16, 17, 32])
def test_reference_is_the_adjacent_fp32_tree_with_one_output_cast(contributors: int):
    assert CANONICAL_MOE_FOLD_VERSION == "canonical_moe_fold_fp32_v2"
    assert CANONICAL_MOE_REDUCE_VERSION == "canonical_moe_reduce_fp32_v2"
    rows = contributors + 2
    values = torch.zeros((contributors, rows, 3), dtype=torch.bfloat16)
    adversarial = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 4,
        dtype=torch.bfloat16,
    )
    for ordinal in range(contributors):
        values[ordinal, :, 0] = adversarial[ordinal]
        values[ordinal, :, 1] = ordinal + 1
        values[ordinal, :, 2] = torch.arange(rows)
    metadata = CanonicalMoEGraphMetadata.build(
        torch.arange(rows),
        torch.arange(rows),
        capacity=rows + 3,
    )
    padded = torch.zeros((contributors, metadata.capacity, 3), dtype=torch.bfloat16)
    padded[:, :rows] = values

    result = canonical_moe_reduce_reference(padded, metadata)
    expected = _explicit_tree(padded)
    expected[~metadata.valid_mask] = 0
    assert torch.equal(result, expected)
    assert torch.equal(canonical_moe_fold_fp32_v2(padded), _explicit_tree(padded))


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_tree_preserves_adversarial_cancellation_before_final_cast(dtype: torch.dtype):
    partials = torch.tensor([[4096.0], [1.0], [-4096.0], [1.0]], dtype=dtype)

    pre_cast = _canonical_moe_fold_fp32_tree(partials)
    transported = canonical_moe_fold_fp32_v2(partials)

    assert pre_cast.dtype is torch.float32
    assert pre_cast.item() == 2.0
    assert transported.dtype is dtype
    assert transported.item() == 2.0


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("contributors", [3, 16, 17])
def test_chunked_fp32_tree_is_bitwise_exact_across_payload_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    contributors: int,
):
    import xorl.distributed.canonical_moe as canonical_moe

    # Force many chunks in a compact test while exercising both odd tails and
    # the production EP16 adjacent-pair tree.
    monkeypatch.setattr(canonical_moe, "_CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES", 128)
    rows, payload = 5, 13
    values = torch.arange(contributors * rows * payload, dtype=torch.float32).reshape(contributors, rows, payload)
    signs = torch.where(torch.arange(contributors).remainder(2) == 0, 1.0, -1.0).view(-1, 1, 1)
    partials = (values.remainder(31).sub(15).mul(signs)).to(dtype)
    witness_count = min(4, contributors)
    partials[:witness_count, 0, :4] = torch.tensor([[4096.0], [1.0], [-4096.0], [1.0]], dtype=dtype)[:witness_count]

    chunk_elements = _canonical_moe_fp32_fold_chunk_elements(contributors)
    assert chunk_elements < rows * payload
    actual = canonical_moe_fold_fp32_v2(partials)
    expected = _explicit_tree(partials)

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


@pytest.mark.cpu
@pytest.mark.parametrize("contributors", [3, 16, 17])
@pytest.mark.parametrize("strided", [False, True])
def test_chunked_fp32_tree_backward_matches_the_unchunked_tree(
    monkeypatch: pytest.MonkeyPatch,
    contributors: int,
    strided: bool,
):
    import xorl.distributed.canonical_moe as canonical_moe

    monkeypatch.setattr(canonical_moe, "_CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES", 128)
    source_shape = (contributors, 11, 7) if strided else (contributors, 7, 11)
    actual_source = torch.randn(source_shape, dtype=torch.bfloat16, requires_grad=True)
    expected_source = actual_source.detach().clone().requires_grad_(True)
    actual_leaf = actual_source.transpose(1, 2) if strided else actual_source
    expected_leaf = expected_source.transpose(1, 2) if strided else expected_source
    assert actual_leaf.is_contiguous() is not strided
    grad_output = torch.randn((7, 11), dtype=torch.bfloat16)

    actual = canonical_moe_fold_fp32_v2(actual_leaf)
    expected = _explicit_tree(expected_leaf)
    actual_grad = torch.autograd.grad(actual, actual_source, grad_outputs=grad_output)[0]
    expected_grad = torch.autograd.grad(expected, expected_source, grad_outputs=grad_output)[0]

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    assert torch.equal(actual_grad.view(torch.uint16), expected_grad.view(torch.uint16))


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("contributors", [3, 16, 17])
def test_chunked_fp32_tree_strided_payload_is_bitwise_exact(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    contributors: int,
):
    import xorl.distributed.canonical_moe as canonical_moe

    monkeypatch.setattr(canonical_moe, "_CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES", 128)
    backing = torch.arange(contributors * 13 * 5, dtype=torch.float32).reshape(contributors, 13, 5)
    partials = backing.transpose(1, 2).to(dtype)
    assert not partials.is_contiguous()

    actual = canonical_moe_fold_fp32_v2(partials)
    expected = _explicit_tree(partials)

    assert torch.equal(actual.view(torch.uint16), expected.contiguous().view(torch.uint16))


@pytest.mark.cpu
@pytest.mark.parametrize("contributors", [1, 3, 16, 17, 64])
def test_fp32_fold_chunk_planner_bounds_the_initial_level(contributors: int):
    chunk_elements = _canonical_moe_fp32_fold_chunk_elements(contributors)
    level_bytes = contributors * chunk_elements * torch.float32.itemsize

    assert 0 < level_bytes <= _CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES
    assert (
        chunk_elements == 1
        or contributors * (chunk_elements + 1) * torch.float32.itemsize > _CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES
    )


@pytest.mark.cpu
def test_fp32_tree_topology_is_not_reassociated_to_a_left_fold():
    # At 2**24 the negative pair's +1 remains representable, so both programs
    # return 1. At 2**25 the unit terms are below the FP32 tie boundary: the
    # adjacent pairs both round to their large operands and cancel to zero,
    # while a left-linear fold loses the first +1 and retains the final +1.
    partials = torch.tensor(
        [[33554432.0], [1.0], [-33554432.0], [1.0]],
        dtype=torch.bfloat16,
    )
    adjacent_tree = _canonical_moe_fold_fp32_tree(partials)
    left_linear = partials[0].float()
    for contributor in partials[1:]:
        left_linear = left_linear + contributor.float()

    assert adjacent_tree.item() == 0.0
    assert left_linear.item() == 1.0


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dtype", "magnitude", "epsilon"),
    [
        pytest.param(torch.bfloat16, 2**25, 1.0, id="bf16-2p25"),
        # FP16 cannot encode 2**25 (it becomes inf). 2**12 with a finite
        # FP16 subnormal epsilon crosses the same FP32-node tie boundary:
        # epsilon is representable at transport but lost beside magnitude.
        pytest.param(torch.float16, 2**12, 2**-20, id="fp16-finite-equivalent"),
    ],
)
@pytest.mark.parametrize("contributors", [3, 5, 8, 17])
@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_fp32_fold_preserves_adjacent_tree_reassociation_witness(
    dtype: torch.dtype,
    magnitude: float,
    epsilon: float,
    contributors: int,
    dynamic: bool,
):
    partials = torch.zeros((contributors, 1), dtype=dtype)
    if contributors == 3:
        # Canonical: (M + -M) + e == e. The alternate M + (-M + e)
        # loses e at the inner FP32 node and returns zero.
        partials[:3, 0] = torch.tensor([magnitude, -magnitude, epsilon], dtype=dtype)
        alternate = partials[0].float() + (partials[1].float() + partials[2].float())
        expected_value = torch.tensor(epsilon, dtype=dtype)
    else:
        # Canonical adjacent pairs both lose e and cancel to zero. A linear
        # left fold loses only the first e and retains the second.
        partials[:4, 0] = torch.tensor([magnitude, epsilon, -magnitude, epsilon], dtype=dtype)
        alternate = partials[0].float()
        for contributor in partials[1:]:
            alternate = alternate + contributor.float()
        expected_value = torch.tensor(0.0, dtype=dtype)

    def fold_fn(values: torch.Tensor) -> torch.Tensor:
        return canonical_moe_fold_fp32_v2(values)

    eager = fold_fn(partials)
    # Each parameter is an independent compiler contract. Clear Dynamo's
    # per-code-object backend cache so 16 intentionally distinct compile
    # requests do not trip the global recompile limit before dynamic=True.
    torch.compiler.reset()
    compiled = torch.compile(fold_fn, fullgraph=True, dynamic=dynamic)(partials)

    assert eager.item() == expected_value.item()
    assert not torch.equal(eager, alternate.to(dtype))
    assert torch.equal(eager, _explicit_tree(partials))
    assert torch.equal(compiled.view(torch.uint16), eager.view(torch.uint16))


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dtype", "shared_value", "routed_value", "scale", "pre_cast_bits", "transport_bits"),
    [
        (torch.bfloat16, -0.01165771484375, 209920.0, 1.1, 0x48618000, 0x4862),
        (
            torch.float16,
            0.046356201171875,
            0.10186767578125,
            -2.7744157314300537,
            0xBE71EFFF,
            0xB38F,
        ),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_leaf_uses_compile_stable_one_round_fp32_fma_before_transport_cast(
    dtype: torch.dtype,
    shared_value: float,
    routed_value: float,
    scale: float,
    pre_cast_bits: int,
    transport_bits: int,
    dynamic: bool,
):
    assert CANONICAL_MOE_LEAF_VERSION == "canonical_moe_leaf_fp32_v1"
    shared = torch.tensor([shared_value], dtype=dtype)
    routed = torch.tensor([routed_value], dtype=dtype)

    fma_oracle = _one_round_leaf_oracle(shared, routed, scale)
    scale_fp32 = torch.tensor(scale, dtype=torch.float32)
    separately_rounded = shared.float() + routed.float() * scale_fp32
    eager = canonical_moe_leaf_fp32_v1(shared, routed, scale)

    def leaf_fn(shared_arg: torch.Tensor, routed_arg: torch.Tensor) -> torch.Tensor:
        return canonical_moe_leaf_fp32_v1(shared_arg, routed_arg, scale)

    compiled = torch.compile(leaf_fn, fullgraph=True, dynamic=dynamic)(shared, routed)

    assert (fma_oracle.view(torch.int32).item() & 0xFFFFFFFF) == pre_cast_bits
    assert not torch.equal(fma_oracle.view(torch.int32), separately_rounded.view(torch.int32))
    assert eager.dtype is dtype
    assert eager.view(torch.uint16).item() == transport_bits
    assert torch.equal(compiled.view(torch.uint16), eager.view(torch.uint16))
    assert separately_rounded.to(dtype).view(torch.uint16).item() != transport_bits


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dynamic", [False, True])
def test_leaf_autograd_matches_declared_fp32_scale_under_compile(
    dtype: torch.dtype,
    dynamic: bool,
):
    scale = -1.375
    shared = torch.tensor([0.25, -0.5, 1.0], dtype=dtype, requires_grad=True)
    routed = torch.tensor([-2.0, 0.75, 4.0], dtype=dtype, requires_grad=True)
    grad_output = torch.tensor([0.5, -1.25, 2.0], dtype=dtype)

    def leaf_fn(shared_arg: torch.Tensor, routed_arg: torch.Tensor) -> torch.Tensor:
        return canonical_moe_leaf_fp32_v1(shared_arg, routed_arg, scale)

    compiled = torch.compile(leaf_fn, fullgraph=True, dynamic=dynamic)
    output = compiled(shared, routed)
    grad_shared, grad_routed = torch.autograd.grad(
        output,
        (shared, routed),
        grad_outputs=grad_output,
    )

    assert output.dtype is dtype
    assert grad_shared.dtype is dtype
    assert grad_routed.dtype is dtype
    assert torch.equal(grad_shared, grad_output)
    assert torch.equal(grad_routed, (grad_output.float() * scale).to(dtype))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("dtype", "shared_value", "routed_value", "scale", "transport_bits"),
    [
        (torch.bfloat16, -0.01165771484375, 209920.0, 1.1, 0x4862),
        (
            torch.float16,
            0.046356201171875,
            0.10186767578125,
            -2.7744157314300537,
            0xB38F,
        ),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_leaf_matches_one_round_oracle_under_compile(
    dtype: torch.dtype,
    shared_value: float,
    routed_value: float,
    scale: float,
    transport_bits: int,
    dynamic: bool,
):
    shared = torch.tensor(
        [shared_value],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    routed = torch.tensor(
        [routed_value],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    def leaf_fn(shared_arg: torch.Tensor, routed_arg: torch.Tensor) -> torch.Tensor:
        return canonical_moe_leaf_fp32_v1(shared_arg, routed_arg, scale)

    eager = leaf_fn(shared, routed)
    compiled = torch.compile(leaf_fn, fullgraph=True, dynamic=dynamic)(shared, routed)
    grad_shared, grad_routed = torch.autograd.grad(
        compiled,
        (shared, routed),
        grad_outputs=torch.ones_like(compiled),
    )

    assert eager.view(torch.uint16).item() == transport_bits
    assert torch.equal(compiled.view(torch.uint16), eager.view(torch.uint16))
    assert torch.equal(grad_shared, torch.ones_like(shared))
    assert torch.equal(
        grad_routed,
        torch.ones_like(routed, dtype=torch.float32).mul(scale).to(dtype),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_leaf_replays_in_cuda_graph_without_fp32_output(dtype: torch.dtype):
    shared = torch.tensor([0.25, -0.5, 1.0], device="cuda", dtype=dtype)
    routed = torch.tensor([-2.0, 0.75, 4.0], device="cuda", dtype=dtype)
    scale = -1.375
    canonical_moe_leaf_fp32_v1(shared, routed, scale)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = canonical_moe_leaf_fp32_v1(shared, routed, scale)
    first = output.clone()
    routed.copy_(torch.tensor([3.0, -1.0, 0.5], device="cuda", dtype=dtype))
    graph.replay()

    expected = _one_round_leaf_oracle(shared, routed, scale).to(dtype).to("cuda")
    assert output.dtype is dtype
    assert output.numel() == shared.numel()
    assert not torch.equal(first, output)
    assert torch.equal(output, expected)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("contributors", "expected_bits"),
    [
        (1, (0x4580, 0x3F80)),
        (2, (0x4580, 0x4040)),
        (3, (0x3F80, 0x40C0)),
        (4, (0x4000, 0x4120)),
        (5, (0x4500, 0x4170)),
        (6, (0x4500, 0x41A8)),
        (8, (0x4080, 0x4210)),
        (16, (0x40B0, 0x4308)),
        (17, (0x40D0, 0x4319)),
    ],
)
def test_fold_exact_vectors_under_fp32_tree(
    contributors: int,
    expected_bits: tuple[int, int],
):
    first_column = [
        4096.0,
        1.0,
        -4096.0,
        1.0,
        2048.0,
        1.0,
        -2048.0,
        1.0,
        1024.0,
        0.5,
        -1024.0,
        0.5,
        512.0,
        0.25,
        -512.0,
        0.25,
        1.0,
    ]
    partials = torch.tensor(
        list(zip(first_column, range(1, 18), strict=True)),
        dtype=torch.bfloat16,
    )

    actual = canonical_moe_fold_fp32_v2(partials[:contributors])

    assert tuple(actual.view(torch.uint16).tolist()) == expected_bits
    assert torch.equal(actual, _explicit_tree(partials[:contributors]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("contributors", [8, 16])
def test_shared_fold_replays_in_cuda_graph(monkeypatch: pytest.MonkeyPatch, contributors: int):
    import xorl.distributed.canonical_moe as canonical_moe

    # Force the graph through the chunk loop without making the unit test
    # production-sized. The production-shape allocator bound is tested below.
    monkeypatch.setattr(canonical_moe, "_CANONICAL_MOE_FP32_FOLD_MAX_LEVEL_BYTES", 32 * 1024)
    partials = torch.randn((contributors, 64, 32), device="cuda", dtype=torch.bfloat16)
    assert _canonical_moe_fp32_fold_chunk_elements(contributors) < partials[0].numel()
    canonical_moe_fold_fp32_v2(partials)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        folded = canonical_moe_fold_fp32_v2(partials)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(folded, _explicit_tree(partials))
    first = folded.clone()
    partials[0].add_(8.0)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(first, folded)
    assert torch.equal(folded, _explicit_tree(partials))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("strided", [False, True])
def test_production_shaped_ep16_fold_has_bounded_peak_workspace(strided: bool):
    contributors, rows, payload = 16, 4096, 6144
    storage_shape = (contributors, payload, rows) if strided else (contributors, rows, payload)
    storage = torch.empty(storage_shape, device="cuda", dtype=torch.bfloat16)
    partials = storage.transpose(1, 2) if strided else storage
    assert partials.is_contiguous() is not strided
    for contributor in range(contributors):
        partials[contributor].fill_(float(contributor - 8) / 16.0)
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    folded = canonical_moe_fold_fp32_v2(partials)
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline

    expected_sample = _explicit_tree(partials[:, :1, :16])
    assert torch.equal(folded[:1, :16].view(torch.uint16), expected_sample.view(torch.uint16))
    assert torch.all(folded == expected_sample[0, 0])
    # The required 48 MiB BF16 output plus at most 48 MiB for the first two
    # FP32 tree levels stays well below the former 1.50 GiB input promotion.
    assert incremental_peak < 128 * 1024 * 1024


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_production_glm_dense_transport_has_bounded_transient_peak(monkeypatch: pytest.MonkeyPatch):
    # WORLD16 DP4/CP4 expands 1,040 local rows across all 16 logical sources.
    contributors, capacity, payload = 16, 16640, 6144
    chunk_rows = _resolve_transport_chunk_rows(
        capacity,
        None,
        CanonicalMoETransport.DENSE_V1,
        contributor_count=contributors,
        payload_elements_per_row=payload,
        element_size=2,
    )
    assert chunk_rows == 170

    local = torch.full((capacity, payload), 0.25, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(capacity, device="cuda", dtype=torch.int64)
    valid_mask = torch.ones(capacity, device="cuda", dtype=torch.bool)
    runtime = _RuntimePlan(
        group_physical_ranks=tuple(range(contributors)),
        physical_to_logical=tuple(range(contributors)),
        local_group_rank=0,
        local_logical_ordinal=0,
    )
    all_to_all_rows: list[int] = []
    all_gather_rows: list[int] = []

    def fake_all_to_all(output: torch.Tensor, input_: torch.Tensor, *, group: object) -> None:
        del group
        all_to_all_rows.append(input_.shape[0] // contributors)
        output.copy_(input_)

    def fake_all_gather(output: torch.Tensor, input_: torch.Tensor, *, group: object) -> None:
        del group
        rows = input_.shape[0]
        all_gather_rows.append(rows)
        gathered = output.view(contributors, rows, payload)
        gathered.zero_()
        gathered[0].copy_(input_)

    monkeypatch.setattr(dist, "all_to_all_single", fake_all_to_all)
    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    output, owner_mask = _transport_and_fold(
        local,
        positions,
        valid_mask,
        group=object(),
        runtime=runtime,
        distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=chunk_rows,
        transport=CanonicalMoETransport.DENSE_V1,
    )
    torch.cuda.synchronize()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline

    expected_calls = (capacity + chunk_rows - 1) // chunk_rows
    assert len(all_to_all_rows) == len(all_gather_rows) == expected_calls
    assert sum(all_to_all_rows) == sum(all_gather_rows) == capacity
    assert max(all_to_all_rows) == max(all_gather_rows) == chunk_rows
    assert max(all_to_all_rows) * contributors * payload * local.element_size() <= _CANONICAL_MOE_DENSE_MAX_BUFFER_BYTES
    expected_expanded_bytes = capacity * contributors * payload * local.element_size()
    assert sum(all_to_all_rows) * contributors * payload * local.element_size() == expected_expanded_bytes
    assert sum(all_gather_rows) * contributors * payload * local.element_size() == expected_expanded_bytes
    assert torch.equal(owner_mask, positions.remainder(contributors) == 0)
    assert torch.equal(output[::contributors].view(torch.uint16), local[::contributors].view(torch.uint16))
    for source_ordinal in range(1, contributors):
        assert torch.count_nonzero(output[source_ordinal::contributors]) == 0
    # This includes the 195 MiB persistent output plus four potentially live
    # expanded buffers, the bounded FP32 fold, and row-sized intermediates. It
    # fits within the 437.62 MiB that the failed rank had free before trying to
    # allocate its former single 768 MiB send buffer.
    assert incremental_peak < 437 * 1024 * 1024


@pytest.mark.cpu
def test_graph_metadata_has_deterministic_padding_and_capacity_guard():
    metadata = CanonicalMoEGraphMetadata.build(
        torch.tensor([7, 3, 11], dtype=torch.int64),
        torch.tensor([17, 2, 25], dtype=torch.int64),
        capacity=8,
    )
    assert metadata.logical_row_ids.tolist() == [7, 3, 11, -1, -1, -1, -1, -1]
    assert metadata.absolute_positions.tolist() == [17, 2, 25, -1, -1, -1, -1, -1]
    assert metadata.valid_mask.tolist() == [True, True, True, False, False, False, False, False]
    with pytest.raises(ValueError, match="exceeds fixed capacity"):
        CanonicalMoEGraphMetadata.build(torch.arange(3), torch.arange(3), capacity=2)


@pytest.mark.cpu
def test_transport_auto_selects_only_regression_qualified_packed_geometry():
    ep16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=1, contributor_count=16)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.PACKED_EP16_V2
    )

    dp_ep16 = ParallelPlan.glm52_trainer(
        world_size=16,
        pp_size=1,
        dp_size=16,
        contributor_count=16,
        cp_size=1,
    )
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=dp_ep16,
            capacity=16 * 4224,
            local_rows=4224,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )
    for contributors in (1, 3, 5, 6, 17):
        plan = ParallelPlan.primitive(contributors)
        assert (
            resolve_canonical_moe_transport(
                "auto",
                plan=plan,
                capacity=34,
                local_rows=2,
                graph_mode=False,
                consumer_sharded_output=True,
            )
            is CanonicalMoETransport.DENSE_V1
        )

    ep8 = ParallelPlan.primitive(8)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep8,
            capacity=4224,
            local_rows=528,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=True,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )


@pytest.mark.cpu
def test_internal_resolution_serves_packed_ep16_v2_on_the_admitted_geometry():
    """The exact GLM path has no public transport knob: internal resolution
    serves packed_ep16_v2 on the admitted eager EP16/CP16 geometry because it
    passed the full-model byte-parity regression anchor, and the dense
    executable oracle elsewhere."""
    ep16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=1, contributor_count=16)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.PACKED_EP16_V2
    )
    ep8 = ParallelPlan.primitive(8)
    assert (
        resolve_canonical_moe_transport(
            "auto",
            plan=ep8,
            capacity=4224,
            local_rows=528,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )


@pytest.mark.cpu
def test_transport_explicit_modes_never_silently_fallback():
    ep16 = ParallelPlan.glm52_trainer(world_size=16, pp_size=1, dp_size=1, contributor_count=16)
    assert (
        resolve_canonical_moe_transport(
            "dense_v1",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=True,
        )
        is CanonicalMoETransport.DENSE_V1
    )
    with pytest.raises(ValueError, match="eager execution"):
        resolve_canonical_moe_transport(
            "cp_sharded_v3",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=True,
            consumer_sharded_output=True,
        )
    with pytest.raises(ValueError, match="consumer-sharded output"):
        resolve_canonical_moe_transport(
            "cp_sharded_v3",
            plan=ep16,
            capacity=4224,
            local_rows=264,
            graph_mode=False,
            consumer_sharded_output=False,
        )
    with pytest.raises(ValueError, match="EP16/CP16"):
        resolve_canonical_moe_transport(
            "packed_ep16_v2",
            plan=ParallelPlan.primitive(8),
            capacity=4224,
            local_rows=528,
            graph_mode=False,
            consumer_sharded_output=False,
        )


@pytest.mark.cpu
def test_packed_ep16_v2_fails_closed_outside_admitted_mode():
    metadata = CanonicalMoEGraphMetadata.build(torch.arange(16), torch.arange(16), capacity=16)
    contribution = LocalMoEContribution(
        torch.zeros((16, 2), dtype=torch.bfloat16),
        metadata,
        "test:packed_ep16:guards",
    )
    with pytest.raises(ValueError, match="EP16/CP16"):
        canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=ParallelPlan.primitive(8),
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )

    with pytest.raises(ValueError, match="EP16/CP16"):
        canonical_moe_reduce_cp_sharded_v3(
            contribution,
            plan=ParallelPlan.primitive(8),
            group=dist.group.WORLD,
        )
    with pytest.raises(ValueError, match="does not yet admit CUDA graph"):
        canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=ParallelPlan.primitive(16),
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
            graph_mode=True,
        )


@pytest.mark.cpu
def test_exact_parallel_plans_hash_launcher_spelling_and_fail_closed():
    trainer = ParallelPlan.glm52_trainer()
    sampler = ParallelPlan.glm52_sampler(launcher_tp_size=8)
    assert trainer.digest != sampler.digest
    assert sampler.as_dict()["launcher_tp_size"] == 8
    assert trainer.pipeline_layer_ranges == ((0, 78),)
    assert trainer.combine_groups == (tuple(range(16)),)
    assert all(trainer.logical_ordinal(physical_rank) == physical_rank for physical_rank in range(16))
    assert trainer.contract_version == CANONICAL_MOE_REDUCE_VERSION

    dp_trainer = ParallelPlan.glm52_trainer(dp_size=16, contributor_count=16, cp_size=1)
    assert dp_trainer.contributor_count == 16
    assert dp_trainer.cp_size == 1
    assert dp_trainer.ep_size == 16
    assert dp_trainer.combine_groups == trainer.combine_groups
    assert dp_trainer.logical_ordinals_by_group == trainer.logical_ordinals_by_group
    assert dp_trainer.cp_ep_aliases == ()
    assert dp_trainer.digest != trainer.digest

    for dp_size, cp_size in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
        plan = ParallelPlan.glm52_trainer(dp_size=dp_size, cp_size=cp_size)
        assert plan.dp_size * plan.cp_size == plan.ep_size == 16

    with pytest.raises(ValueError, match="exactly cover"):
        ParallelPlan.glm52_trainer(dp_size=2, cp_size=4)
    with pytest.raises(ValueError, match="world size"):
        ParallelPlan.glm52_trainer(world_size=32, dp_size=2, cp_size=8)
    with pytest.raises(ValueError, match="explicit model-derived pipeline ranges"):
        ParallelPlan.glm52_trainer(pp_size=2, dp_size=2, cp_size=8)

    payload = sampler.as_dict()
    payload["world_size"] = 16
    payload["role"] = sampler.role
    with pytest.raises(ValueError, match="partition|world|sampler topology"):
        ParallelPlan(**payload)

    payload = sampler.as_dict()
    payload["role"] = ParallelRole.SAMPLER
    payload["launcher_tp_size"] = 99
    with pytest.raises(ValueError, match="launcher TP"):
        ParallelPlan(**payload)

    payload = sampler.as_dict()
    payload["role"] = ParallelRole.SAMPLER
    payload["logical_ordinals_by_group"] = (tuple(reversed(range(8))),)
    with pytest.raises(ValueError, match="identity logical contributor ordinals"):
        ParallelPlan(**payload)


@pytest.mark.cpu
@pytest.mark.parametrize("contributors", [1, 3, 5, 6, 17])
def test_parallel_plan_records_any_declared_positive_complete_contributor_group(
    contributors: int,
):
    primitive = ParallelPlan.primitive(contributors)
    trainer = ParallelPlan.glm52_trainer(
        world_size=contributors,
        dp_size=1,
        cp_size=contributors,
        contributor_count=contributors,
    )
    sampler = ParallelPlan.glm52_sampler(launcher_tp_size=contributors)

    for plan in (primitive, trainer, sampler):
        assert plan.contributor_count == contributors
        assert plan.combine_groups == (tuple(range(contributors)),)
        assert plan.logical_ordinals_by_group == (tuple(range(contributors)),)

    # These plans describe a declared complete local contributor group. They
    # do not assert that a TP-sharded sampler with a different contributor
    # count is byte-equivalent to an EP1 trainer identity.
    if contributors == 1:
        local_complete = torch.tensor([[3.0, -2.0]], dtype=torch.bfloat16)
        assert torch.equal(canonical_moe_fold_fp32_v2(local_complete), local_complete[0])


@pytest.mark.cpu
def test_parallel_plan_and_fold_reject_only_nonpositive_contributor_counts():
    with pytest.raises(ValueError, match="positive"):
        ParallelPlan.primitive(0)
    with pytest.raises(ValueError, match="positive contributor count"):
        canonical_moe_fold_fp32_v2(torch.empty((0, 2), dtype=torch.bfloat16))


@pytest.mark.cpu
def test_trainer_identity_is_descriptive_and_runtime_invariants_are_structural():
    trainer = ParallelPlan.glm52_trainer()
    assert trainer.family == "glm52"
    renamed = replace(trainer, family="model-defined-exact-moe")
    assert renamed.family == "model-defined-exact-moe"
    assert renamed.digest != trainer.digest

    with pytest.raises(ValueError, match="model metadata exactly"):
        replace(trainer, model_num_layers=80)

    primitive = ParallelPlan.primitive(8)
    qwen_primitive = replace(primitive, family="qwen3_5_moe")
    assert primitive.digest != qwen_primitive.digest
    payload = qwen_primitive.as_dict()
    round_tripped = ParallelPlan(**payload)
    assert round_tripped.family == "qwen3_5_moe"
    assert round_tripped.digest == qwen_primitive.digest
    assert ParallelPlan(**trainer.as_dict()).digest == trainer.digest
    assert ParallelPlan(**ParallelPlan.glm52_sampler(launcher_tp_size=8).as_dict()).role is ParallelRole.SAMPLER


@pytest.mark.cpu
@pytest.mark.parametrize("dp_size,cp_size", [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)])
def test_logical_row_ownership_covers_every_mixed_factorization(dp_size: int, cp_size: int):
    seen = []
    for dp_rank in range(dp_size):
        expected_context = tuple(range(dp_rank * cp_size, (dp_rank + 1) * cp_size))
        for cp_rank in range(cp_size):
            ownership = LogicalRowOwnership(dp_size, cp_size, dp_rank, cp_rank, 16)
            seen.append(ownership.source_ordinal)
            assert ownership.context_source_ordinals == expected_context
            source = ownership.source_slice(padded_rows=7, local_rows=3)
            assert source == slice(ownership.source_ordinal * 7, ownership.source_ordinal * 7 + 3)
    assert seen == list(range(16))


@pytest.mark.cpu
def test_logical_row_ownership_rejects_only_concrete_shape_mismatch():
    with pytest.raises(ValueError, match="exactly cover"):
        LogicalRowOwnership(2, 4, 0, 0, 16)
    with pytest.raises(ValueError, match="DP rank"):
        LogicalRowOwnership(2, 8, 2, 0, 16)
    assert LogicalRowOwnership.valid_positions(torch.tensor([0, -1, 9])).tolist() == [True, False, True]


@pytest.mark.cpu
def test_world32_pp1_cp_and_ep_groups_alias_as_four_rank_octets():
    main_mesh = torch.arange(32).view(4, 8)
    ep_mesh = init_ep_mesh_matrix(ep_size=8, ep_fsdp_size=4, ep_intranode=True)

    cp_groups = tuple(tuple(int(rank) for rank in row) for row in main_mesh)
    ep_groups = tuple(tuple(int(rank) for rank in ep_mesh[:, column]) for column in range(4))
    expert_fsdp_groups = tuple(tuple(int(rank) for rank in row) for row in ep_mesh)

    expected_octets = tuple(tuple(range(start, start + 8)) for start in range(0, 32, 8))
    assert cp_groups == ep_groups == expected_octets
    assert expert_fsdp_groups[0] == (0, 8, 16, 24)
    assert expert_fsdp_groups[-1] == (7, 15, 23, 31)


@pytest.mark.cpu
def test_world32_pp1_cp16_and_ep16_groups_alias_as_two_rank_groups():
    main_mesh = torch.arange(32).view(2, 16)
    ep_mesh = init_ep_mesh_matrix(ep_size=16, ep_fsdp_size=2, ep_intranode=True)

    cp_groups = tuple(tuple(int(rank) for rank in row) for row in main_mesh)
    ep_groups = tuple(tuple(int(rank) for rank in ep_mesh[:, column]) for column in range(2))
    expert_fsdp_groups = tuple(tuple(int(rank) for rank in row) for row in ep_mesh)

    expected_groups = (tuple(range(16)), tuple(range(16, 32)))
    assert cp_groups == ep_groups == expected_groups
    assert expert_fsdp_groups[0] == (0, 16)
    assert expert_fsdp_groups[-1] == (15, 31)


def _make_partials(world: int, capacity: int) -> torch.Tensor:
    rank = dist.get_rank()
    values = torch.empty((capacity, 4), dtype=torch.bfloat16)
    values[:, 0] = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 2,
        dtype=torch.bfloat16,
    )[rank]
    values[:, 1] = rank + 1
    values[:, 2] = torch.arange(capacity)
    values[:, 3] = torch.arange(capacity) * (rank + 1)
    return values


def _run_distributed_case() -> None:
    import xorl.distributed.canonical_moe as canonical_moe

    dist.init_process_group("gloo")
    world = dist.get_world_size()
    physical_rank = dist.get_rank()
    mapping = tuple(reversed(range(world)))
    plan = ParallelPlan.primitive(world, physical_to_logical=mapping)

    capacity = world + 5
    valid_rows = world + 2
    positions = torch.tensor([(index * 3 + 1) % 29 for index in range(valid_rows)], dtype=torch.int64)
    row_ids = torch.tensor(list(reversed(range(valid_rows))), dtype=torch.int64)
    metadata = CanonicalMoEGraphMetadata.build(row_ids, positions, capacity=capacity)
    local = _make_partials(world, capacity).requires_grad_(True)
    # Force the default byte planner through multiple transport chunks in this
    # compact real-collective test. Every rank derives the same two-row cap.
    canonical_moe._CANONICAL_MOE_DENSE_MAX_BUFFER_BYTES = world * local[0].numel() * local.element_size() * 2
    assert (
        _resolve_transport_chunk_rows(
            capacity,
            None,
            CanonicalMoETransport.DENSE_V1,
            contributor_count=world,
            payload_elements_per_row=local[0].numel(),
            element_size=local.element_size(),
        )
        == 2
    )
    contribution = LocalMoEContribution(
        local,
        metadata,
        local_partial_policy="test:routed_then_shared:bf16",
    )

    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.detach())
    physical_stack = torch.stack(gathered)
    logical_to_physical = [mapping.index(logical) for logical in range(world)]
    logical_stack = physical_stack[logical_to_physical]
    expected = canonical_moe_reduce_reference(logical_stack, metadata)

    replicated = canonical_moe_reduce_fp32_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
    )
    assert torch.equal(replicated.tensor.view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(replicated.tensor[~metadata.valid_mask]) == 0

    permutation = torch.tensor(list(reversed(range(capacity))), dtype=torch.long)
    permuted_metadata = CanonicalMoEGraphMetadata(
        logical_row_ids=metadata.logical_row_ids.index_select(0, permutation),
        absolute_positions=metadata.absolute_positions.index_select(0, permutation),
        valid_mask=metadata.valid_mask.index_select(0, permutation),
        capacity=capacity,
        valid_rows=valid_rows,
    )
    permuted = canonical_moe_reduce_fp32_v2(
        LocalMoEContribution(
            local.index_select(0, permutation),
            permuted_metadata,
            local_partial_policy="test:routed_then_shared:bf16",
        ),
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=2,
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(permuted.tensor.index_select(0, inverse), replicated.tensor)

    for row in range(valid_rows):
        solo_metadata = CanonicalMoEGraphMetadata.build(
            metadata.logical_row_ids[row : row + 1],
            metadata.absolute_positions[row : row + 1],
            capacity=1,
        )
        solo_row = canonical_moe_reduce_fp32_v2(
            LocalMoEContribution(
                local[row : row + 1],
                solo_metadata,
                local_partial_policy="test:routed_then_shared:bf16",
            ),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        assert torch.equal(solo_row.tensor[0], replicated.tensor[row])

    solo = canonical_moe_reduce_fp32_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=capacity,
    )
    assert torch.equal(solo.tensor, replicated.tensor)

    owner_sharded = canonical_moe_reduce_fp32_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        chunk_rows=3,
    )
    owners = metadata.owner_ordinals(world)
    local_logical = mapping[physical_rank]
    expected_owner_mask = metadata.valid_mask & (owners == local_logical)
    assert torch.equal(owner_sharded.owner_mask, expected_owner_mask)
    assert torch.equal(owner_sharded.tensor[expected_owner_mask], expected[expected_owner_mask])
    assert torch.count_nonzero(owner_sharded.tensor[~expected_owner_mask]) == 0

    replicated.tensor.float().sum().backward()
    assert local.grad is not None
    expected_grad = torch.zeros_like(local)
    expected_grad[metadata.valid_mask] = world
    assert torch.equal(local.grad.view(torch.int16), expected_grad.view(torch.int16))

    with pytest.raises(TypeError, match="already reduced"):
        canonical_moe_reduce_fp32_v2(
            replicated,
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )

    dist.barrier()
    dist.destroy_process_group()


def _packed_ep16_local_partial(capacity: int, payload: int) -> torch.Tensor:
    rank = dist.get_rank()
    rows = torch.arange(capacity, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(payload, dtype=torch.float32).unsqueeze(0)
    signs = -1.0 if rank % 4 == 1 else 1.0
    values = signs * (rows.remainder(29) - 14.0) * (rank + 1) / 32.0 + columns / 128.0
    values[:, 0] = torch.tensor(
        [4096.0, -4096.0, 1.0, 1.0, 0.5, -0.5, 2.0, -2.0] * 2,
        dtype=torch.float32,
    )[rank]
    values[min(17, capacity - 1)] = 0.0  # A valid row with no routed/shared contribution.
    return values.to(torch.bfloat16)


def _assert_packed_ep16_matches_fp32_v2(
    *,
    capacity: int,
    valid_rows: int,
    payload: int,
    padded_reset_tail_rows: int = 0,
    requested_chunk_rows: int | None = None,
) -> None:
    plan = ParallelPlan.primitive(16)
    if padded_reset_tail_rows:
        if valid_rows != capacity or capacity <= padded_reset_tail_rows:
            raise ValueError("padded-reset test geometry requires all capacity rows valid and a nonempty prefix")
        positions = torch.cat(
            (
                torch.arange(capacity - padded_reset_tail_rows, dtype=torch.int64),
                torch.zeros(padded_reset_tail_rows, dtype=torch.int64),
            ),
        )
        owner_counts = torch.bincount(positions.remainder(16), minlength=16)
        assert int(owner_counts.max()) > (capacity + 15) // 16
    else:
        positions = torch.arange(valid_rows, dtype=torch.int64)
    if positions.numel() == capacity:
        metadata = CanonicalMoEGraphMetadata(
            logical_row_ids=torch.arange(capacity, dtype=torch.int64),
            absolute_positions=positions,
            valid_mask=torch.ones(capacity, dtype=torch.bool),
            capacity=capacity,
            valid_rows=capacity,
        )
    else:
        metadata = CanonicalMoEGraphMetadata.build(
            torch.arange(valid_rows, dtype=torch.int64),
            positions,
            capacity=capacity,
        )
    local = _packed_ep16_local_partial(capacity, payload)
    # Invalid tail bytes must never escape either transport.
    local[valid_rows:] = torch.arange(capacity - valid_rows, dtype=torch.float32).unsqueeze(1).add(1).bfloat16()
    contribution = LocalMoEContribution(local, metadata, "test:packed_ep16:adversarial_bf16")

    dense = canonical_moe_reduce_fp32_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
    )

    for repeat in range(3):
        packed = canonical_moe_reduce_packed_ep16_v2(
            contribution,
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
            chunk_rows=requested_chunk_rows,
        )
        assert torch.equal(packed.tensor.view(torch.int16), dense.tensor.view(torch.int16))
        assert torch.count_nonzero(packed.tensor[~metadata.valid_mask]) == 0

    dense_owner = canonical_moe_reduce_fp32_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
    )
    packed_owner = canonical_moe_reduce_packed_ep16_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.OWNER_SHARDED,
        chunk_rows=requested_chunk_rows,
    )
    assert torch.equal(packed_owner.tensor.view(torch.int16), dense_owner.tensor.view(torch.int16))
    assert torch.equal(packed_owner.owner_mask, dense_owner.owner_mask)

    if capacity == 35:
        dense_leaf = local.clone().requires_grad_(True)
        packed_leaf = local.clone().requires_grad_(True)
        dense_for_backward = canonical_moe_reduce_fp32_v2(
            LocalMoEContribution(dense_leaf, metadata, "test:packed_ep16:backward"),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        packed_for_backward = canonical_moe_reduce_packed_ep16_v2(
            LocalMoEContribution(packed_leaf, metadata, "test:packed_ep16:backward"),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        dense_for_backward.tensor.float().sum().backward()
        packed_for_backward.tensor.float().sum().backward()
        assert dense_leaf.grad is not None and packed_leaf.grad is not None
        assert torch.equal(packed_leaf.grad.view(torch.int16), dense_leaf.grad.view(torch.int16))


def _assert_cp_sharded_v3_matches_fp32_v2(*, capacity: int, valid_rows: int, payload: int) -> None:
    if capacity % 16:
        raise ValueError("v3 test requires equal padded CP-source capacity")
    plan = ParallelPlan.primitive(16)
    metadata = CanonicalMoEGraphMetadata.build(
        torch.arange(valid_rows, dtype=torch.int64),
        torch.arange(valid_rows, dtype=torch.int64),
        capacity=capacity,
    )
    local = _packed_ep16_local_partial(capacity, payload)
    local[valid_rows:] = torch.arange(capacity - valid_rows, dtype=torch.float32).unsqueeze(1).add(1).bfloat16()
    contribution = LocalMoEContribution(local, metadata, "test:cp_sharded_v3:adversarial_bf16")
    dense = canonical_moe_reduce_fp32_v2(
        contribution,
        plan=plan,
        group=dist.group.WORLD,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
    )
    local_capacity = capacity // 16
    rank = dist.get_rank()
    start = rank * local_capacity
    end = start + local_capacity

    for repeat in range(3):
        sharded = canonical_moe_reduce_cp_sharded_v3(
            contribution,
            plan=plan,
            group=dist.group.WORLD,
        )
        assert sharded.distribution is OutputDistribution.CONSUMER_SHARDED
        assert torch.equal(sharded.tensor.view(torch.int16), dense.tensor[start:end].view(torch.int16))

    if capacity == 32:
        dense_leaf = local.clone().requires_grad_(True)
        v3_leaf = local.clone().requires_grad_(True)
        dense_for_backward = canonical_moe_reduce_fp32_v2(
            LocalMoEContribution(dense_leaf, metadata, "test:cp_sharded_v3:backward"),
            plan=plan,
            group=dist.group.WORLD,
            output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        )
        v3_for_backward = canonical_moe_reduce_cp_sharded_v3(
            LocalMoEContribution(v3_leaf, metadata, "test:cp_sharded_v3:backward"),
            plan=plan,
            group=dist.group.WORLD,
        )
        dense_for_backward.tensor[start:end].float().sum().backward()
        v3_for_backward.tensor.float().sum().backward()
        assert dense_leaf.grad is not None and v3_leaf.grad is not None
        assert torch.equal(v3_leaf.grad.view(torch.int16), dense_leaf.grad.view(torch.int16))


def _run_packed_ep16_case() -> None:
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 16

    # Compact adversarial arithmetic plus both production row geometries used
    # by the 4,099-token trace: a 13-row CP tail and the server's 125-row
    # pad-to-128 tail. Padding resets position IDs to zero, so neither owner
    # histogram fits ceil(capacity / 16).
    _assert_packed_ep16_matches_fp32_v2(capacity=35, valid_rows=32, payload=7)
    _assert_packed_ep16_matches_fp32_v2(
        capacity=4112,
        valid_rows=4112,
        payload=4,
        padded_reset_tail_rows=13,
        requested_chunk_rows=128,
    )
    _assert_packed_ep16_matches_fp32_v2(
        capacity=4224,
        valid_rows=4224,
        payload=4,
        padded_reset_tail_rows=125,
        requested_chunk_rows=128,
    )
    _assert_cp_sharded_v3_matches_fp32_v2(capacity=32, valid_rows=29, payload=7)
    _assert_cp_sharded_v3_matches_fp32_v2(capacity=4112, valid_rows=4099, payload=4)
    _assert_cp_sharded_v3_matches_fp32_v2(capacity=4224, valid_rows=4099, payload=4)

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":

    @pytest.mark.cpu
    @pytest.mark.parametrize("contributors", [2, 4, 8])
    def test_distributed_transport_tree_distribution_chunking_and_backward(contributors: int):
        result = run_distributed_script(__file__, num_gpus=contributors, timeout=180)
        result.assert_success(f"canonical MoE primitive with {contributors} CPU contributors")

    @pytest.mark.cpu
    def test_packed_ep16_v2_matches_dense_v1_bitwise():
        result = run_distributed_script(
            __file__,
            num_gpus=16,
            timeout=240,
            extra_env={"CANONICAL_MOE_TEST_MODE": "packed_ep16"},
        )
        result.assert_success("packed EP16 transport must equal dense v1 byte-for-byte")


if __name__ == "__main__":
    if os.environ.get("CANONICAL_MOE_TEST_MODE") == "packed_ep16":
        _run_packed_ep16_case()
    else:
        _run_distributed_case()
