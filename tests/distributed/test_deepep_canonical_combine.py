"""Canonical combine over the real DeepEP normal transport.

Runs the actual DeepEP intranode dispatch/combine kernels (via xorl's
integration) at the Qwen3.6-35B-A3B MoE geometry (256 experts, top-k 8,
H=2048) and asserts BYTE equality of the canonically-combined outputs and
gradients against a locally-computed logical-order reference, across three
routing regimes including the K<EP dedup-relevant skews (tokens whose
experts all live on a strict subset of ranks).

Pipeline per case (SPMD, every rank is both a token owner and an expert
rank):

  primary DeepEP dispatch (hidden + top-k)        [real transport 1]
  slot-level DeepEP dispatch (fingerprint+source) [real transport 2]
  expert-side receive-layout validation           (fail closed)
  per-slot expert outputs from RECEIVED bytes     (elementwise, reproducible)
  return combine over the slot handle             [real transport 3;
                                                   single-contributor sum =
                                                   pure per-slot transport]
  owner receipt from TRANSPORTED annotations -> validate_transport_receipt
  canonical_combine (reference AND triton)        == local logical reference
  backward: handle-based reverse dispatch of grads [real transport 4]
           -> per-slot payload grads == locally recomputed reference grads

Plus fail-closed negatives executed against the REAL received layouts.

Launch: pytest wrapper -> torchrun self-launch. World size from
XORL_TEST_DEEPEP_CC_WORLD (default 2; use 8 on a full node). Requires
deep_ep + nvidia.nvshmem (importorskip) and >= world GPUs.
"""

from __future__ import annotations

import os
import sys
from ctypes import CDLL
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


_WORKER_ENV = "XORL_TEST_DEEPEP_CC_WORKER"
_WORLD_ENV = "XORL_TEST_DEEPEP_CC_WORLD"

# Geometry is env-driven so one worker serves both the Qwen3.6-35B-A3B
# default (256 experts, top-8) and the Q397B-class regate (512 experts,
# top-10; K<EP at its production EP32, dedup behaviors exercised at world
# 8). H stays 2048: the combine byte claims are pinned at H=2048 by the
# production-hidden identity gate and are otherwise H-agnostic.
_E = int(os.environ.get("XORL_TEST_DEEPEP_CC_EXPERTS", "256"))
_K = int(os.environ.get("XORL_TEST_DEEPEP_CC_TOPK", "8"))
_H, _T = 2048, 256


def _prepend_library_path(path: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if path not in existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path


def _install_nvidia_ml_library_path() -> None:
    try:
        CDLL("libnvidia-ml.so.1")
        return
    except OSError:
        pass
    for stub in (
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"),
        Path("/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"),
    ):
        if not stub.exists():
            continue
        stub_dir = Path("/tmp/xorl-nvidia-ml-stub")
        stub_dir.mkdir(exist_ok=True)
        soname = stub_dir / "libnvidia-ml.so.1"
        if not soname.exists():
            soname.symlink_to(stub)
        _prepend_library_path(str(stub_dir))
        return


def _install_nvshmem_library_path() -> None:
    import nvidia.nvshmem  # noqa: PLC0415

    nvshmem_lib = os.path.join(list(nvidia.nvshmem.__path__)[0], "lib")
    _prepend_library_path(nvshmem_lib)


# ---------------------------------------------------------------------------
# Deterministic global fixture (every rank reconstructs every rank's data)
# ---------------------------------------------------------------------------


def _rank_hidden(rank: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
    return torch.randn((_T, _H), generator=generator).to(torch.bfloat16).to(device)


def _rank_routing(rank: int, case: str, world: int, device):
    generator = torch.Generator(device="cpu").manual_seed(2000 + rank)
    num_local = _E // world
    ids = torch.empty((_T, _K), dtype=torch.int64)
    for t in range(_T):
        if case == "balanced":
            pool = torch.randperm(_E, generator=generator)
        elif case == "rank_skewed":
            # K<EP dedup behavior: every token's experts live on ONE rank
            # (a strict subset of the EP group): 8 slots -> 1 dedup send.
            target = int(torch.randint(0, world, (1,), generator=generator).item())
            pool = target * num_local + torch.randperm(num_local, generator=generator)
        elif case == "empty_experts":
            pool = torch.randperm(max(_E // 8, _K), generator=generator)
        else:
            raise ValueError(case)
        ids[t] = pool[:_K]
    weights = torch.rand((_T, _K), generator=generator).softmax(dim=-1).to(torch.float32)
    return ids.to(torch.int32).to(device), weights.to(device)


def _expert_scales(device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(3000)
    return (torch.rand((_E,), generator=generator) * 1.5 + 0.25).to(torch.float32).to(device)


def _rank_upstream(rank: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(9000 + rank)
    return torch.randn((_T, _H), generator=generator).to(torch.bfloat16).to(device)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16 if x.dtype is torch.bfloat16 else torch.int32)


def _reference_slot_rows(hidden, ids, scales):
    """Per-slot expert outputs from SOURCE bytes: bf16(bf16row.f32 * scale[e])."""
    active = ids >= 0
    tokens = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(ids)[active]
    experts = ids[active].to(torch.int64)
    return (hidden.index_select(0, tokens).float() * scales.index_select(0, experts).unsqueeze(1)).to(torch.bfloat16)


def _run_case(case: str, rank: int, world: int, device, buffer) -> None:
    from xorl.distributed.canonical_combine import (
        CanonicalCombineError,
        CanonicalRouteMetadata,
        TransportReceipt,
        TransportReceiptError,
        canonical_combine,
        validate_transport_receipt,
    )
    from xorl.distributed.deepep_canonical import (
        DeepEPReceiveAlignmentError,
        build_owner_receipt,
        dispatch_primary,
        dispatch_slot_level,
        pack_return_payload,
        return_slot_rows,
        validate_expert_receive_alignment,
    )

    num_local = _E // world
    scales = _expert_scales(device)
    hidden = _rank_hidden(rank, device)
    ids, weights = _rank_routing(rank, case, world, device)

    # --- real transports 1 + 2 ---
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, _handle = dispatch_primary(buffer, hidden, weights, ids, _E)
    recv_meta, recv_slot_idx, recv_slot_weights, plan = dispatch_slot_level(buffer, hidden, weights, ids, _E)

    # --- expert-side validation (real invariant, fail-closed) ---
    view, annotations = validate_expert_receive_alignment(
        recv_x=recv_x,
        recv_topk_idx=recv_topk_idx,
        recv_topk_weights=recv_topk_weights,
        recv_meta=recv_meta,
        recv_slot_idx=recv_slot_idx,
        recv_slot_weights=recv_slot_weights,
        num_local_experts=num_local,
        ep_rank=rank,
    )

    # Negative N1: a corrupted fingerprint byte must RAISE (symmetric on all ranks).
    if view.num_slot_rows > 0:
        corrupted = recv_meta.clone()
        corrupted[0, 0] = corrupted[0, 0] + 1.0
        try:
            validate_expert_receive_alignment(
                recv_x=recv_x,
                recv_topk_idx=recv_topk_idx,
                recv_topk_weights=recv_topk_weights,
                recv_meta=corrupted,
                recv_slot_idx=recv_slot_idx,
                recv_slot_weights=recv_slot_weights,
                num_local_experts=num_local,
                ep_rank=rank,
            )
        except DeepEPReceiveAlignmentError:
            pass
        else:
            raise AssertionError(f"rank {rank} {case}: corrupted fingerprint was not rejected")

    # --- per-slot expert compute from RECEIVED bytes (weight-free rows) ---
    global_experts = view.local_expert + rank * num_local
    slot_inputs = recv_x.index_select(0, view.primary_row)
    payload_rows = (slot_inputs.float() * scales.index_select(0, global_experts).unsqueeze(1)).to(torch.bfloat16)
    payload = pack_return_payload(payload_rows, annotations).requires_grad_(True)

    # --- real transport 3: per-slot return combine ---
    returned = return_slot_rows(payload, buffer, plan)

    weights_leaf = weights.clone().requires_grad_(True)
    metadata = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights_leaf, num_experts=_E)
    rows, receipt = build_owner_receipt(metadata, plan, returned, hidden_cols=_H)
    validate_transport_receipt(metadata, receipt)

    combined = canonical_combine(rows, metadata, receipt, backend="reference")
    combined_triton = canonical_combine(rows.detach(), metadata, receipt, backend="triton")
    assert torch.equal(_bits(combined.detach()), _bits(combined_triton)), (
        f"rank {rank} {case}: reference and triton canonical backends diverge on the real layout"
    )

    # --- forward gate: bytes vs local logical reference ---
    ref_rows = _reference_slot_rows(hidden, ids, scales)
    identity = torch.arange(plan.num_slots, device=device)
    ref_slot_to_row = torch.full_like(ids, -1, dtype=torch.int64)
    ref_slot_to_row[plan.slot_tokens, plan.slot_ks] = identity
    ref_receipt = TransportReceipt(
        slot_to_row=ref_slot_to_row,
        num_rows=plan.num_slots,
        row_expert_ids=plan.slot_experts,
        row_source_tokens=plan.slot_tokens,
        transport_id="local_logical_reference",
    )
    ref_metadata = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=_E)
    ref_combined = canonical_combine(ref_rows, ref_metadata, ref_receipt, backend="reference")
    assert torch.equal(_bits(combined.detach()), _bits(ref_combined.detach())), (
        f"rank {rank} {case}: FORWARD bytes over real DeepEP transport differ from the logical reference"
    )

    # --- backward gate: real transport 4 (handle-based reverse dispatch) ---
    upstream = _rank_upstream(rank, device)
    combined.backward(upstream)
    payload_grad = payload.grad
    assert payload_grad is not None and payload_grad.shape == payload.shape
    ann_grad = payload_grad[:, _H:]
    assert torch.all(ann_grad == 0), f"rank {rank} {case}: annotation columns received nonzero grad"

    from xorl.distributed.deepep_canonical import decode_annotations

    _, ann_tokens, ann_ks, ann_src = decode_annotations(annotations)
    grad_expected = torch.empty_like(payload_rows)
    for src in range(world):
        mask = ann_src == src
        if not bool(mask.any()):
            continue
        src_upstream = _rank_upstream(src, device)
        _, src_weights = _rank_routing(src, case, world, device)
        tok = ann_tokens[mask]
        kk = ann_ks[mask]
        grad_expected[mask] = (src_upstream.index_select(0, tok).float() * src_weights[tok, kk].unsqueeze(1)).to(
            torch.bfloat16
        )
    assert torch.equal(_bits(payload_grad[:, :_H]), _bits(grad_expected)), (
        f"rank {rank} {case}: BACKWARD per-slot grads over real DeepEP transport differ from reference"
    )

    # Owner-side weight grads vs the dense analytic form (slot-major FP32).
    active = ids >= 0
    slot_rows_idx = receipt.slot_to_row[active]
    dots = (
        upstream.float().index_select(0, plan.slot_tokens) * rows.detach().float().index_select(0, slot_rows_idx)
    ).sum(dim=1)
    expected_wgrad = torch.zeros_like(weights)
    expected_wgrad[plan.slot_tokens, plan.slot_ks] = dots
    assert torch.equal(weights_leaf.grad.view(torch.int32), expected_wgrad.view(torch.int32)), (
        f"rank {rank} {case}: routing-weight grads differ from the analytic slot-major form"
    )

    # Negative N2: swapped mapping entries for two different tokens must fail
    # receipt validation via the TRANSPORTED annotations.
    flat_active = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    if flat_active.numel() >= 2 * _K:
        tampered = receipt.slot_to_row.clone()
        a, b = flat_active[0], flat_active[_K]  # slots of different tokens
        tampered.reshape(-1)[a], tampered.reshape(-1)[b] = (
            receipt.slot_to_row.reshape(-1)[b].clone(),
            receipt.slot_to_row.reshape(-1)[a].clone(),
        )
        bad = TransportReceipt(
            slot_to_row=tampered,
            num_rows=receipt.num_rows,
            row_expert_ids=receipt.row_expert_ids,
            row_source_tokens=receipt.row_source_tokens,
            transport_id="tampered",
        )
        try:
            validate_transport_receipt(metadata, bad)
        except TransportReceiptError:
            pass
        else:
            raise AssertionError(f"rank {rank} {case}: tampered slot mapping was not rejected")

    # Negative N3: truncated return must fail closed at receipt construction.
    try:
        build_owner_receipt(metadata, plan, returned[:-1], hidden_cols=_H)
    except CanonicalCombineError:
        pass
    else:
        raise AssertionError(f"rank {rank} {case}: truncated return was not rejected")

    # ------------------------------------------------------------------
    # PRODUCTION RECEIPT PATH (derived, no annotation transport): fresh
    # slot dispatch, expert-side crosscheck against the PRIMARY handle's
    # own source metadata, H-only payload, owner receipt derived from
    # source order. Must be byte-identical to the transported arm.
    # ------------------------------------------------------------------
    from xorl.distributed.deepep_canonical import (
        ExpertSlotView,
        build_owner_receipt_derived,
        crosscheck_primary_source_indices,
        handle_is_intranode,
    )

    recv_meta2, recv_slot_idx2, recv_slot_weights2, plan2 = dispatch_slot_level(buffer, hidden, weights, ids, _E)
    view2, _ = validate_expert_receive_alignment(
        recv_x=recv_x,
        recv_topk_idx=recv_topk_idx,
        recv_topk_weights=recv_topk_weights,
        recv_meta=recv_meta2,
        recv_slot_idx=recv_slot_idx2,
        recv_slot_weights=recv_slot_weights2,
        num_local_experts=num_local,
        ep_rank=rank,
    )
    derived_arm_admitted = handle_is_intranode(plan2.handle)
    if not derived_arm_admitted:
        # Internode handles carry no source-token
        # indices (SourceMeta = src_rdma_rank + NVL bitmask), so the derived
        # receipt path must REFUSE, not decode garbage. Assert the refusal.
        try:
            crosscheck_primary_source_indices(primary_handle=_handle, slot_handle=plan2.handle, view=view2, topk=_K)
        except DeepEPReceiveAlignmentError as refusal:
            assert "intranode-only" in str(refusal), refusal
        else:
            raise AssertionError(f"rank {rank} {case}: derived-receipt crosscheck did not refuse an internode handle")
        if rank == 0:
            print(
                f"[deepep-cc] case={case}: derived-receipt arm SKIPPED (internode; "
                "the handle carries no source-token indices); transported arm stands",
                flush=True,
            )
        dist.barrier()
        return
    crosscheck_primary_source_indices(primary_handle=_handle, slot_handle=plan2.handle, view=view2, topk=_K)
    # Negative N4: a tampered view (two primary rows of different source
    # tokens swapped) must fail the primary-source crosscheck.
    if view2.num_slot_rows >= 2:
        swap_a, swap_b = 0, int(view2.num_slot_rows - 1)
        if int(view2.primary_row[swap_a]) != int(view2.primary_row[swap_b]):
            tampered_rows = view2.primary_row.clone()
            tampered_rows[swap_a], tampered_rows[swap_b] = (
                view2.primary_row[swap_b].clone(),
                view2.primary_row[swap_a].clone(),
            )
            tampered_view = ExpertSlotView(
                primary_row=tampered_rows,
                primary_k=view2.primary_k,
                local_expert=view2.local_expert,
                num_slot_rows=view2.num_slot_rows,
            )
            try:
                crosscheck_primary_source_indices(
                    primary_handle=_handle, slot_handle=plan2.handle, view=tampered_view, topk=_K
                )
            except DeepEPReceiveAlignmentError:
                pass
            else:
                raise AssertionError(f"rank {rank} {case}: tampered source mapping was not rejected")

    slot_inputs2 = recv_x.index_select(0, view2.primary_row)
    global_experts2 = view2.local_expert + rank * num_local
    payload2 = (
        (slot_inputs2.float() * scales.index_select(0, global_experts2).unsqueeze(1)).to(torch.bfloat16)
    ).contiguous()
    returned2 = return_slot_rows(payload2, buffer, plan2)
    metadata2 = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=_E)
    rows2, receipt2 = build_owner_receipt_derived(metadata2, plan2, returned2)
    validate_transport_receipt(metadata2, receipt2)
    combined_derived = canonical_combine(rows2, metadata2, receipt2, backend="reference")
    assert torch.equal(_bits(combined_derived.detach()), _bits(combined.detach())), (
        f"rank {rank} {case}: DERIVED-receipt combine bytes differ from the transported-annotation arm"
    )

    dist.barrier()
    if rank == 0:
        print(
            f"[deepep-cc] case={case} world={world} E={_E} K={_K} PASSED "
            f"(fwd+bwd byte-exact, derived-receipt arm byte-equal, negatives closed)",
            flush=True,
        )


def _run_ll_comparison(rank: int, world: int, device) -> None:
    """Compare DeepEP ``low_latency_combine`` with the canonical program.

    The test executes ``low_latency_dispatch`` in BF16 with logfmt disabled,
    computes expert outputs in the received slab layout, executes the real
    low-latency combine, and byte-compares it with ``canonical_combine`` over
    independently reconstructed per-slot rows.
    """
    import deep_ep

    from xorl.distributed.canonical_combine import (
        CanonicalRouteMetadata,
        TransportReceipt,
        canonical_combine,
    )

    max_tokens = _T
    num_local = _E // world
    scales = _expert_scales(device)
    hidden = _rank_hidden(rank, device)
    ids, weights = _rank_routing(rank, "balanced", world, device)

    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(max_tokens, _H, world, _E)
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        num_nvl_bytes=0,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local,
    )
    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        hidden, ids.to(torch.int64), max_tokens, _E, use_fp8=False, async_finish=False, return_recv_hook=False
    )
    # Expert compute in the slab layout: elementwise per local expert over
    # the WHOLE slab (invalid slots hold garbage the combine never reads).
    local_scales = scales[rank * num_local : (rank + 1) * num_local]
    expert_out = (recv_x.float() * local_scales.view(num_local, 1, 1)).to(torch.bfloat16)

    combined_ll, _, _ = buffer.low_latency_combine(
        expert_out, ids.to(torch.int64), weights, handle, async_finish=False, return_recv_hook=False
    )

    ref_rows = _reference_slot_rows(hidden, ids, scales)
    active = ids >= 0
    slot_tokens = torch.arange(_T, device=device).unsqueeze(1).expand_as(ids)[active]
    slot_to_row = torch.full_like(ids, -1, dtype=torch.int64)
    slot_to_row[active] = torch.arange(int(active.sum()), device=device)
    receipt = TransportReceipt(
        slot_to_row=slot_to_row,
        num_rows=int(active.sum()),
        row_expert_ids=ids[active].to(torch.int64),
        row_source_tokens=slot_tokens,
        transport_id="ll_local_reference",
    )
    metadata = CanonicalRouteMetadata(topk_ids=ids, topk_weights=weights, num_experts=_E)
    combined_canonical = canonical_combine(ref_rows, metadata, receipt, backend="reference")

    matches = torch.equal(_bits(combined_ll), _bits(combined_canonical))
    mismatched = int((_bits(combined_ll) != _bits(combined_canonical)).sum())
    print(
        f"[ll-compare] rank={rank} bitwise_match={matches} mismatched_elements={mismatched}/{combined_ll.numel()}",
        flush=True,
    )
    assert matches, (
        f"rank {rank}: low_latency_combine bytes differ from canonical_combine "
        f"({mismatched} of {combined_ll.numel()} elements); refuse serving-side admission"
    )
    dist.barrier()
    if rank == 0:
        print("deepep_ll_comparison_ok", flush=True)


def _worker_main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert _E % world == 0

    if os.environ.get("XORL_TEST_DEEPEP_CC_LL") == "1":
        _run_ll_comparison(rank, world, torch.device("cuda", local_rank))
        dist.barrier()
        dist.destroy_process_group()
        return 0

    from xorl.distributed.moe.deepep import DeepEPBuffer

    buffer = DeepEPBuffer(ep_group=dist.group.WORLD, buffer_size_gb=0.5, num_sms=20)
    buffer.init_buffer(hidden_bytes=(_H + 32) * 2)

    for case in ("balanced", "rank_skewed", "empty_experts"):
        _run_case(case, rank, world, torch.device("cuda", local_rank), buffer)

    dist.barrier()
    if rank == 0:
        print("deepep_canonical_combine_gate_ok", flush=True)
    buffer.destroy_buffer()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__" and os.environ.get(_WORKER_ENV) == "1":
    sys.exit(_worker_main())


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _required_gpus() -> int:
    return int(os.environ.get(_WORLD_ENV, "2"))


def _launch(world: int, extra_env: dict, sentinel: str = "deepep_canonical_combine_gate_ok") -> None:
    from distributed_utils import gpu_count, run_distributed_script

    if gpu_count() < world:
        pytest.skip(f"requires {world} GPUs, found {gpu_count()}")
    _install_nvidia_ml_library_path()
    _install_nvshmem_library_path()
    result = run_distributed_script(
        __file__,
        num_gpus=world,
        timeout=600,
        extra_env={
            _WORKER_ENV: "1",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            **extra_env,
        },
    )
    result.assert_success(f"canonical combine over real DeepEP normal transport (world {world})")
    assert sentinel in result.stdout


def test_canonical_combine_over_real_deepep_normal():
    pytest.importorskip("deep_ep")
    pytest.importorskip("nvidia.nvshmem")
    _launch(_required_gpus(), {})


def test_canonical_combine_over_real_deepep_q397b_geometry():
    """Exercise 512 experts and top-10 routing (K < EP at EP32).

    DeepEP's normal dispatch supports at most 128 local experts, so the
    512-expert geometry cannot be projected below EP4.  Keep the ordinary
    256-expert gate above runnable at EP2, but require a valid topology for
    this model-specific gate instead of failing inside the CUDA kernel.
    """
    pytest.importorskip("deep_ep")
    pytest.importorskip("nvidia.nvshmem")
    _launch(
        max(_required_gpus(), 4),
        {"XORL_TEST_DEEPEP_CC_EXPERTS": "512", "XORL_TEST_DEEPEP_CC_TOPK": "10"},
    )


def test_low_latency_combine_matches_canonical():
    """The real low-latency BF16 combine matches the canonical program."""
    pytest.importorskip("deep_ep")
    pytest.importorskip("nvidia.nvshmem")
    _launch(_required_gpus(), {"XORL_TEST_DEEPEP_CC_LL": "1"}, sentinel="deepep_ll_comparison_ok")
