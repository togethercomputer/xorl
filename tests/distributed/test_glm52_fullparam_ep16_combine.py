"""16-contributor ordered-combine byte gate for the full-param expert bank.

This CPU/gloo test runs 16 ranks and covers full-param bank partials entering the GLM
canonical contributor-tree reduce (`canonical_moe_reduce_fp64_v3`, the same code
the GLM-5.2 model path calls) with byte equality against the logical-order
pairwise-tree reference; variable per-rank row counts through the padded
gather; sentinel/off-owner routing; and gradient plumbing from the combined
output back through the gather and the straight-through function to each
rank's FP32 expert masters with expert-boundary locality.

What it does NOT cover: the CUDA expert kernels inside the partial (covered
separately by test_glm52_exact_fullparam_experts.py; here the value program
is the CPU surrogate on dequantized cache bytes), NCCL
transport (gloo here; the contributor-tree order is xorl-owned code, not the
collective's), the aliased EP16/CP16 process-group admission of the real
model path, and full-model integration.
"""

from __future__ import annotations

import os as _os

import torch
import torch.distributed as dist


_LOCAL_EXPERTS = 4
# 16 is the contract row; the NCCL variant runs 8 contributors on one
# 8-GPU node because NCCL rejects multiple ranks per device — it gates the
# TRANSPORT byte-integrity, while the 16-contributor tree arithmetic
# (xorl-owned code, backend-independent) is gated by the gloo variant.
_CONTRIBUTORS = int(_os.environ.get("GLM52_EP16_COMBINE_CONTRIBUTORS", "16"))
_GLOBAL_EXPERTS = _LOCAL_EXPERTS * _CONTRIBUTORS
_HIDDEN = 128
_INTERMEDIATE = 128
_TOPK = 2


def _deterministic_grid(*shape: int, scale: int, offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
    return ((values * scale + offset) % 29 - 14).reshape(shape) / 16.0


def _seed_bank(rank: int, device: torch.device = torch.device("cpu"), kind: str = "fullparam"):
    """Build a full-param (or frozen) bank whose cache bytes are exact on CPU.

    Master values are chosen on the FP8-representable grid so that
    fp8-roundtrip == identity; the cache is seeded directly (scales = 1) and
    the masters set to the dequantized bytes, mirroring load_prequantized
    without the CUDA kernels.  ``kind='frozen'`` builds the serving
    ``Glm52NativeBlockFP8Experts`` (the out-of-scope-layer bank under
    ``trainable_expert_layers``) with the frozen activation dgrad admitted —
    the same bytes, no masters.
    """

    from xorl.models.transformers.glm5.exact_fullparam_experts import (
        Glm52FullParamBlockFP8RoutedExperts,
    )
    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
    from xorl.ops.exact.block_fp8_native import pack_fp8_as_float32

    if kind == "frozen":
        bank = Glm52NativeBlockFP8Experts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    else:
        bank = Glm52FullParamBlockFP8RoutedExperts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    gate_up_fp8 = (
        _deterministic_grid(_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, scale=3, offset=rank * 5 + 1)
        .to(device)
        .to(torch.float8_e4m3fn)
    )
    down_fp8 = (
        _deterministic_grid(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN, scale=7, offset=rank * 3 + 2)
        .to(device)
        .to(torch.float8_e4m3fn)
    )
    with torch.no_grad():
        bank.gate_up_packed_weight_f32.copy_(
            pack_fp8_as_float32(gate_up_fp8).reshape(bank.gate_up_packed_weight_f32.shape)
        )
        bank.gate_up_weight_scale_inv.fill_(1.0)
        bank.down_packed_weight_f32.copy_(pack_fp8_as_float32(down_fp8).reshape(bank.down_packed_weight_f32.shape))
        bank.down_weight_scale_inv.fill_(1.0)
        if kind != "frozen":
            bank.gate_up_weight_master.copy_(gate_up_fp8.float())
            bank.down_weight_master.copy_(down_fp8.float())
    if kind == "frozen":
        bank.enable_frozen_activation_dgrad()
    else:
        bank._record_master_identity()

    # CPU stand-ins for the CUDA dequant and fused value programs: the value
    # program is the bank's own surrogate on the dequantized cache bytes,
    # exactly the CPU wiring the single-rank component test gates.
    def dequantized_cached_experts():
        gate_up = bank.gate_up_proj.float().transpose(1, 2).contiguous().to(torch.bfloat16)
        down = bank.down_proj.float().transpose(1, 2).contiguous().to(torch.bfloat16)
        return gate_up, down

    def sampler_value(hidden, routing, local_ids, *, routed_scaling_factor):
        gate_up, down = dequantized_cached_experts()
        return bank._surrogate_program(
            hidden.float(),
            routing.float(),
            local_ids,
            gate_up.float(),
            down.float(),
            routed_scaling_factor=routed_scaling_factor,
        ).to(torch.bfloat16)

    bank._dequantized_cached_experts = dequantized_cached_experts
    if kind == "frozen":
        bank._sglang_ep_native_routed_value = sampler_value
    else:
        bank._sampler_value = sampler_value
    return bank


def _canonical_tree_reference(partials: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """The contributor pairwise-tree reference from the canonical-MoE contract
    gate (adjacent-pair FP64 tree over logical contributor order, one final
    cast to the transported dtype)."""

    level = [partials[index].double() for index in range(partials.shape[0])]
    while len(level) > 1:
        level = [level[index] + level[index + 1] for index in range(0, len(level), 2)]
    result = level[0].to(partials.dtype)
    result[~valid_mask] = 0
    return result


def _run_ep16_case() -> None:
    import os

    from xorl.distributed.canonical_moe import (
        CanonicalMoEGraphMetadata,
        LocalMoEContribution,
        OutputDistribution,
        ParallelPlan,
        canonical_moe_reduce_fp64_v3,
    )
    from xorl.models.layers.moe.ep_native_combine import (
        gather_ids_for_ep_combine,
        gather_tokens_for_ep_combine,
        max_rows_for_ep_combine,
    )
    from xorl.models.transformers.glm5.modeling_glm5 import GLM52_LOCAL_PARTIAL_POLICY

    backend = os.environ.get("GLM52_EP16_COMBINE_BACKEND", "gloo")
    dist.init_process_group(backend)
    if dist.get_world_size() != _CONTRIBUTORS:
        raise RuntimeError(f"combine gate requires {_CONTRIBUTORS} ranks, got {dist.get_world_size()}")
    rank = dist.get_rank()
    group = dist.group.WORLD
    if backend == "nccl":
        # 16 contributors over the node's GPUs (2 communicator ranks per
        # device is legal for NCCL and preserves the 16-way tree).
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    bank_kind = os.environ.get("GLM52_EP16_COMBINE_BANK", "fullparam")
    bank = _seed_bank(rank, device, kind=bank_kind)
    frozen_bytes_before = (
        {name: parameter.detach().view(torch.uint8).clone() for name, parameter in bank.named_parameters()}
        if bank_kind == "frozen"
        else None
    )

    # Variable per-rank row counts exercise the padded gather.
    local_rows = 2 + (rank % 2)
    hidden_local = (
        _deterministic_grid(local_rows, _HIDDEN, scale=11, offset=rank * 13 + 3)
        .to(device)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    routing_local = (
        (_deterministic_grid(local_rows, _TOPK, scale=5, offset=rank * 7 + 1).abs() + 0.125)
        .to(device=device, dtype=torch.float32)
        .requires_grad_(True)
    )
    global_ids = torch.stack(
        (
            (torch.arange(local_rows, dtype=torch.int64) * 7 + rank * 4) % _GLOBAL_EXPERTS,
            (torch.arange(local_rows, dtype=torch.int64) * 3 + rank * 11 + 1) % _GLOBAL_EXPERTS,
        ),
        dim=1,
    ).to(device)
    positions_local = (torch.arange(local_rows, dtype=torch.int64, device=device) + rank * 100).contiguous()

    padded_rows = max_rows_for_ep_combine(local_rows, hidden_local.device, group)
    gathered_hidden = gather_tokens_for_ep_combine(hidden_local, group, padded_rows)
    gathered_routing = gather_tokens_for_ep_combine(routing_local, group, padded_rows)
    gathered_ids = gather_ids_for_ep_combine(global_ids, group, padded_rows)
    gathered_positions = gather_ids_for_ep_combine(positions_local[:, None], group, padded_rows).squeeze(-1)
    valid_local = torch.ones((local_rows, 1), dtype=torch.int32, device=device)
    gathered_valid = gather_ids_for_ep_combine(valid_local, group, padded_rows).squeeze(-1) >= 0

    expert_start = rank * _LOCAL_EXPERTS
    local_ids = (
        torch.where(
            (gathered_ids >= expert_start) & (gathered_ids < expert_start + _LOCAL_EXPERTS),
            gathered_ids - expert_start,
            gathered_ids.new_full((), -1),
        )
        .to(torch.int32)
        .contiguous()
    )

    partial = bank(
        gathered_hidden.contiguous(),
        gathered_routing.contiguous(),
        sglang_ep_native_local_ids=local_ids,
        routed_scaling_factor=1.5,
    )

    capacity = _CONTRIBUTORS * padded_rows
    logical_rows = torch.arange(capacity, dtype=torch.int64, device=device).masked_fill(~gathered_valid, -1)
    metadata = CanonicalMoEGraphMetadata(
        logical_row_ids=logical_rows,
        absolute_positions=gathered_positions.to(torch.int64),
        valid_mask=gathered_valid,
        capacity=capacity,
        valid_rows=int(gathered_valid.sum().item()),
    )
    if _CONTRIBUTORS == 16:
        plan = ParallelPlan.glm52_trainer(
            world_size=_CONTRIBUTORS, pp_size=1, dp_size=1, contributor_count=_CONTRIBUTORS
        )
    else:
        # Non-contract contributor counts (the NCCL transport variant) use the
        # primitive identity plan — same reduce arithmetic, no GLM row claim.
        plan = ParallelPlan.primitive(_CONTRIBUTORS)
    contribution = LocalMoEContribution(partial, metadata, GLM52_LOCAL_PARTIAL_POLICY)
    replicated = canonical_moe_reduce_fp64_v3(
        contribution,
        plan=plan,
        group=group,
        output_distribution=OutputDistribution.REPLICATED_CANONICAL,
        chunk_rows=5,
    )

    # Reference: gather the raw partials and fold them through the pairwise
    # contributor tree in logical order (identity for the glm52 plan).
    gathered_partials = [torch.empty_like(partial) for _ in range(_CONTRIBUTORS)]
    dist.all_gather(gathered_partials, partial.detach())
    physical_stack = torch.stack(gathered_partials)
    combine_group = plan.combine_groups[0]
    ordinals = plan.logical_ordinals_by_group[0]
    logical_stack = torch.empty_like(physical_stack)
    for position, physical_rank in enumerate(combine_group):
        logical_stack[ordinals[position]] = physical_stack[physical_rank]
    expected = _canonical_tree_reference(logical_stack, metadata.valid_mask)

    if not torch.equal(replicated.tensor, expected):
        mismatched = int((replicated.tensor.view(torch.uint8) != expected.view(torch.uint8)).sum().item())
        raise AssertionError(
            f"rank {rank}: combined bytes diverge from contributor-tree reference ({mismatched} bytes)"
        )

    # Gradient plumbing: combined output -> canonical reduce -> gather -> the
    # bank backward -> this rank's activations (and, for the full-param bank,
    # its FP32 masters).
    replicated.tensor.float().sum().backward()
    assert hidden_local.grad is not None and bool(torch.all(torch.isfinite(hidden_local.grad)))
    assert routing_local.grad is not None and bool(torch.all(torch.isfinite(routing_local.grad)))

    if bank_kind == "frozen":
        # Out-of-scope frozen bank activation dgrad: hidden + routing
        # gradients arrive through the combine (routers keep training), the
        # bank has NO masters, no parameter receives any gradient, and every
        # frozen byte is bit-identical after the backward.
        assert bool(hidden_local.grad.abs().sum() > 0)
        assert bool(routing_local.grad.abs().sum() > 0)
        for name, parameter in bank.named_parameters():
            assert parameter.grad is None, f"rank {rank}: frozen bank parameter {name} received a gradient"
            assert not parameter.requires_grad
        assert frozen_bytes_before is not None
        for name, before in frozen_bytes_before.items():
            current = dict(bank.named_parameters())[name].detach().view(torch.uint8)
            assert torch.equal(current, before), f"rank {rank}: frozen bank bytes changed: {name}"
    else:
        assert bank.gate_up_weight_master.grad is not None and bank.down_weight_master.grad is not None
        assert bank.gate_up_weight_master.grad.dtype is torch.float32

        routed_local_experts = {
            int(value) - expert_start
            for value in gathered_ids[gathered_valid].flatten().tolist()
            if expert_start <= int(value) < expert_start + _LOCAL_EXPERTS
        }
        for expert_index in range(_LOCAL_EXPERTS):
            gate_up_nonzero = int(torch.count_nonzero(bank.gate_up_weight_master.grad[expert_index]).item())
            down_nonzero = int(torch.count_nonzero(bank.down_weight_master.grad[expert_index]).item())
            if expert_index in routed_local_experts:
                assert gate_up_nonzero > 0 and down_nonzero > 0, f"rank {rank} expert {expert_index} missing grad"
            else:
                assert gate_up_nonzero == 0 and down_nonzero == 0, f"rank {rank} expert {expert_index} leaked grad"

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest
    from distributed_utils import run_distributed_script

    @pytest.mark.cpu
    def test_glm52_fullparam_tree_reference_uses_fp64_nodes():
        partials = torch.zeros((16, 1), dtype=torch.bfloat16)
        partials[:4, 0] = torch.tensor(
            [33554432.0, 1.0, -33554432.0, 1.0],
            dtype=torch.bfloat16,
        )
        actual = _canonical_tree_reference(partials, torch.ones(1, dtype=torch.bool))
        retired_fp32 = (partials[0].float() + partials[1].float()) + (partials[2].float() + partials[3].float())

        assert actual.item() == 2.0
        assert retired_fp32.item() == 0.0

    @pytest.mark.cpu
    def test_glm52_fullparam_ep16_ordered_combine_byte_gate():
        result = run_distributed_script(__file__, num_gpus=_CONTRIBUTORS, timeout=300)
        result.assert_success("full-param EP16 ordered combine must match the contributor-tree reference bytewise")

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 8,
        reason="NCCL combine transport gate needs an 8-GPU node",
    )
    def test_glm52_fullparam_ordered_combine_byte_gate_nccl_transport():
        # NCCL rejects >1 rank per device, so the single-node NCCL variant
        # runs the 8-contributor tree: it gates the NCCL transport bytes;
        # the 16-contributor arithmetic is gated by the gloo variant above.
        result = run_distributed_script(
            __file__,
            num_gpus=8,
            timeout=420,
            extra_env={
                "GLM52_EP16_COMBINE_BACKEND": "nccl",
                "GLM52_EP16_COMBINE_CONTRIBUTORS": "8",
            },
        )
        result.assert_success("full-param ordered combine must be bytewise on NCCL transport")

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 8,
        reason="NCCL combine transport gate needs an 8-GPU node",
    )
    def test_glm52_frozen_bank_combine_activation_grads_nccl_transport():
        # Out-of-scope-layer semantics: frozen native
        # banks on every contributor — activation gradients (hidden + routing)
        # must traverse the ordered combine while no bank byte moves and no
        # bank parameter receives a gradient.  GPU-only: the frozen bank's
        # forward requires CUDA activations before its value seam.
        result = run_distributed_script(
            __file__,
            num_gpus=8,
            timeout=420,
            extra_env={
                "GLM52_EP16_COMBINE_BACKEND": "nccl",
                "GLM52_EP16_COMBINE_CONTRIBUTORS": "8",
                "GLM52_EP16_COMBINE_BANK": "frozen",
            },
        )
        result.assert_success("frozen-bank activation gradients must traverse the ordered combine with immutable bytes")


if __name__ == "__main__":
    _run_ep16_case()
