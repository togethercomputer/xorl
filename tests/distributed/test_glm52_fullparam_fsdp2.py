"""FSDP2 gate for the full-param components.

This four-GPU NCCL test validates that the publish-the-bytes
mechanism survives real ``fully_shard`` wrapping at the production policy
(no mp_policy for `fsdp_requires_full_precision` modules; dense/router
masters sharded over the world mesh; expert bank on a singleton mesh, the
production eFSDP layout at DP1):

- the pre-wrap staleness identity fails closed after wrapping (parameter
  objects are replaced — a stale cache can never score);
- post-wrap refresh re-records identity on the wrapped (DTensor) masters
  and produces cache bytes bitwise equal to an unwrapped reference;
- forward consumes the cache under FSDP2 unshard hooks, straight-through
  master grads land as sharded DTensors, and a real Adam step trips the
  staleness gate;
- after refresh, published bytes equal the reference that took the same
  step unwrapped (identical rank inputs; world=4 keeps the FSDP gradient
  predivide exact in FP32), and publication is byte-identical across ranks.

Not covered: production wrapping of the full model (this gate
wraps components directly), CP-varied per-rank batches (grad averaging
across different token shards is ordinary training numerics, outside the
forward byte contract), DCP checkpoint round-trips.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.distributed as dist


_IN, _OUT = 512, 384
_EXPERTS, _EHIDDEN, _EINTER = 4, 128, 128
_ROUTER_E, _ROUTER_H = 64, 512


def _pattern(*shape: int, offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
    return (((values * 3 + offset) % 29) - 14).reshape(shape) / 16.0


def _build_components(device: torch.device):
    from xorl.models.transformers.glm5.exact_fullparam_experts import (
        Glm52FullParamBlockFP8RoutedExperts,
    )
    from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
        Glm52ExactFullParamRouterWeight,
        Glm52ExactTP1BlockFP8FullParamLinear,
    )

    dense = Glm52ExactTP1BlockFP8FullParamLinear(_IN, _OUT, device=device)
    dense.load_prequantized(
        _pattern(_OUT, _IN, offset=1).to(device).to(torch.float8_e4m3fn),
        torch.ones(3, 4, dtype=torch.float32, device=device),
    )
    router = Glm52ExactFullParamRouterWeight(_ROUTER_E, _ROUTER_H, device=device)
    router.load_from_bf16(_pattern(_ROUTER_E, _ROUTER_H, offset=2).to(device).to(torch.bfloat16))
    bank = Glm52FullParamBlockFP8RoutedExperts(_EXPERTS, _EHIDDEN, _EINTER, device=device)
    bank.load_prequantized(
        _pattern(_EXPERTS, _EHIDDEN, 2 * _EINTER, offset=3).to(device).to(torch.float8_e4m3fn),
        torch.ones(_EXPERTS, 1, 2, dtype=torch.float32, device=device),
        _pattern(_EXPERTS, _EINTER, _EHIDDEN, offset=4).to(device).to(torch.float8_e4m3fn),
        torch.ones(_EXPERTS, 1, 1, dtype=torch.float32, device=device),
    )
    return dense, router, bank


def _published_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _expect_stale(callable_, what: str) -> None:
    try:
        callable_()
    except RuntimeError as error:
        if "stale" not in str(error) and "seeded" not in str(error):
            raise AssertionError(f"{what}: unexpected error {error}") from error
        return
    raise AssertionError(f"{what}: staleness gate did not trip")


def _run_fsdp2_case() -> None:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import DTensor

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world = dist.get_world_size()
    rank = dist.get_rank()

    wrapped = _build_components(device)
    reference = _build_components(device)

    mesh = init_device_mesh("cuda", (world,))
    # The production eFSDP expert layout at DP1 is a REAL 1-rank subgroup
    # mesh: FSDP2's sharding path needs a live backend (a bare DeviceMesh
    # with _init_backend=False has none and fails in fully_shard), so build
    # every singleton subgroup collectively and adopt this rank's.
    singleton_group = None
    for subgroup_rank in range(world):
        group = dist.new_group([subgroup_rank])
        if subgroup_rank == rank:
            singleton_group = group
    singleton = DeviceMesh.from_group(singleton_group, "cuda")
    dense, router, bank = wrapped
    # Production policy: no mp_policy on fsdp_requires_full_precision modules.
    fully_shard(dense, mesh=mesh)
    fully_shard(router, mesh=mesh)
    fully_shard(bank, mesh=singleton)

    for module, name in ((dense, "dense"), (router, "router"), (bank, "bank")):
        master = next(iter(module.parameters()))
        assert isinstance(master, DTensor), f"{name} master was not wrapped into a DTensor"

    # Wrapping replaced the parameter objects: the pre-wrap identity must
    # fail closed before any scoring or publication.
    hidden = _pattern(6, _IN, offset=7).to(device).to(torch.bfloat16)
    _expect_stale(lambda: dense(hidden.clone()), "dense forward after wrap")
    _expect_stale(dense.publishable_weight_bytes, "dense publication after wrap")
    _expect_stale(bank.publishable_expert_bytes, "bank publication after wrap")
    _expect_stale(router.publishable_weight_bytes, "router publication after wrap")

    # The dense forward probe raised THROUGH FSDP2's forward, skipping the
    # post-forward reshard and leaving that root in the unshard window
    # (module attribute = the unsharded working copy).  Return every root to
    # the sharded state a production step boundary sees before refreshing.
    for module in (dense, router, bank):
        module.reshard()

    # Post-wrap refresh re-records identity on the DTensor masters; the
    # published bytes must equal the unwrapped reference bitwise.
    dense.refresh_quantized_cache()
    router.refresh_effective_view()
    bank.refresh_quantized_cache()
    ref_dense, ref_router, ref_bank = reference
    ref_dense.refresh_quantized_cache()
    ref_router.refresh_effective_view()
    ref_bank.refresh_quantized_cache()

    def _assert_published_equal(stage: str) -> None:
        w, s = dense.publishable_weight_bytes()
        rw, rs = ref_dense.publishable_weight_bytes()
        assert torch.equal(w.view(torch.uint8), rw.view(torch.uint8)), f"{stage}: dense weight bytes diverge"
        assert torch.equal(s, rs), f"{stage}: dense scales diverge"
        assert torch.equal(router.publishable_weight_bytes(), ref_router.publishable_weight_bytes()), (
            f"{stage}: router bytes diverge"
        )
        for mine, theirs in zip(bank.publishable_expert_bytes(), ref_bank.publishable_expert_bytes(), strict=True):
            assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8)), f"{stage}: bank bytes diverge"

    _assert_published_equal("post-wrap refresh")

    # Forward under FSDP2 unshard hooks; straight-through grads; Adam step.
    def _run_step(dense_mod, router_mod, bank_mod):
        d_in = hidden.clone().requires_grad_(True)
        d_out = dense_mod(d_in)
        r_in = _pattern(5, _ROUTER_H, offset=9).to(device).to(torch.bfloat16).requires_grad_(True)
        r_out = router_mod(r_in)
        b_hidden = _pattern(4, _EHIDDEN, offset=11).to(device).to(torch.bfloat16)
        b_ids = torch.tensor([[0], [1], [2], [-1]], dtype=torch.int32, device=device)
        b_routing = torch.tensor([[0.5], [0.25], [0.75], [1.0]], dtype=torch.float32, device=device)
        b_out = bank_mod(b_hidden, b_routing, sglang_ep_native_local_ids=b_ids)
        (d_out.float().sum() + r_out.float().sum() + b_out.float().sum()).backward()

    _run_step(dense, router, bank)
    _run_step(ref_dense, ref_router, ref_bank)
    assert isinstance(dense.weight_master.grad, DTensor), "dense master grad is not a sharded DTensor"

    def _optimizer(components):
        return torch.optim.Adam(
            [parameter for module in components for parameter in module.parameters()],
            lr=0.5,
        )

    _optimizer(wrapped).step()
    _optimizer(reference).step()

    _expect_stale(lambda: dense(hidden.clone()), "dense forward after optimizer step")
    _expect_stale(dense.publishable_weight_bytes, "dense publication after optimizer step")
    _expect_stale(bank.publishable_expert_bytes, "bank publication after optimizer step")
    _expect_stale(router.publishable_weight_bytes, "router publication after optimizer step")

    dense.refresh_quantized_cache()
    router.refresh_effective_view()
    bank.refresh_quantized_cache()
    ref_dense.refresh_quantized_cache()
    ref_router.refresh_effective_view()
    ref_bank.refresh_quantized_cache()
    # world=4 with identical rank inputs keeps the FSDP2 gradient predivide
    # exact (power of two), so the wrapped step must equal the unwrapped one.
    _assert_published_equal("post-step refresh")
    dense(hidden.clone())

    # Publication must be byte-identical across ranks.
    digest = _published_digest(
        dense.publishable_weight_bytes()[0],
        dense.publishable_weight_bytes()[1],
        router.publishable_weight_bytes(),
        *bank.publishable_expert_bytes(),
    )
    digests = [None] * world
    dist.all_gather_object(digests, digest)
    assert len(set(digests)) == 1, f"rank {rank}: publication digests diverge across ranks: {digests}"

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest
    import torch as _torch
    from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

    @pytest.mark.gpu
    @skip_if_gpu_count_less_than(4)
    def test_glm52_fullparam_components_survive_fsdp2_wrapping():
        result = run_distributed_script(__file__, num_gpus=4, timeout=420)
        result.assert_success("full-param components must keep byte-publication invariants under FSDP2")

    del _torch


if __name__ == "__main__":
    _run_fsdp2_case()
