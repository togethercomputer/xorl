"""Gates for the GLM-5.2 full-param step-boundary refresh + publication.

CPU tests cover the orchestration plumbing (rank partition, per-rank
partials, rank-0 merge, completeness fail-closed, completion markers) with a CPU
surrogate for the CUDA-pinned bank quantizer. The value program under test is
byte plumbing, not quantization arithmetic. The GPU test runs the real refresh
path end to end on CUDA components.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch import nn

import xorl.server.weight_sync.glm52_fullparam_step_publish as step_publish_module
from xorl.models.transformers.glm5.exact_fullparam_experts import (
    Glm52FullParamBlockFP8RoutedExperts,
)
from xorl.models.transformers.glm5.exact_fullparam_fp8 import (
    Glm52ExactFullParamRouterWeight,
)
from xorl.server.weight_sync.glm52_fullparam_payload import (
    Glm52FullParamPayloadError,
    load_glm52_fullparam_payload,
)
from xorl.server.weight_sync.glm52_fullparam_step_publish import (
    glm52_fullparam_step_boundary,
)


_LOCAL_EXPERTS = 4
_GLOBAL_EXPERTS = 8
_HIDDEN = 128
_INTERMEDIATE = 128


def _seeded_bank(expert_start: int, device: torch.device) -> Glm52FullParamBlockFP8RoutedExperts:
    from xorl.ops.block_fp8_native import pack_fp8_as_float32

    bank = Glm52FullParamBlockFP8RoutedExperts(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device)
    gate_up_fp8 = (
        (torch.arange(bank.gate_up_packed_weight_f32.numel() * 4, dtype=torch.float32) % 29 - 14)
        .div(16.0)
        .reshape(_LOCAL_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE)
        .to(device)
        .to(torch.float8_e4m3fn)
    )
    down_fp8 = (
        (torch.arange(bank.down_packed_weight_f32.numel() * 4, dtype=torch.float32) * 7 % 29 - 14)
        .div(16.0)
        .reshape(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN)
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
        bank.gate_up_weight_master.copy_(gate_up_fp8.float())
        bank.down_weight_master.copy_(down_fp8.float())
    bank._record_master_identity()
    bank.assign_global_expert_range(expert_start, _GLOBAL_EXPERTS)
    return bank


def _seeded_router(device: torch.device) -> Glm52ExactFullParamRouterWeight:
    router = Glm52ExactFullParamRouterWeight(_GLOBAL_EXPERTS, _HIDDEN, device=device)
    checkpoint = (
        torch.arange(_GLOBAL_EXPERTS * _HIDDEN, dtype=torch.float32)
        .reshape(_GLOBAL_EXPERTS, _HIDDEN)
        .sub_(41)
        .div_(37)
        .to(device)
        .to(torch.bfloat16)
    )
    router.load_from_bf16(checkpoint)
    return router


def _rank_model(
    expert_start: int,
    device: torch.device,
    *,
    cpu_surrogate_refresh: bool,
    layer_index: int = 3,
) -> nn.Module:
    """A model tree exposing global checkpoint layer FQNs."""

    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module() for _ in range(layer_index + 1)])
    mlp = nn.Module()
    mlp.gate = _seeded_router(device)
    mlp.experts = _seeded_bank(expert_start, device)
    root.model.layers[layer_index].mlp = mlp
    if cpu_surrogate_refresh:
        # CPU surrogate for the CUDA-pinned quantizer: the plumbing gate
        # stages the already-seeded cache bytes and re-records identity while
        # the quantizer arithmetic (gated elsewhere on GPU) is not exercised.
        bank = mlp.experts
        bank._validate_refresh_source = lambda: None
        bank._gather_refresh_masters = lambda: (
            bank.gate_up_weight_master,
            bank.down_weight_master,
        )
        bank._stage_quantized_caches = lambda _masters: (
            bank.gate_up_proj,
            bank.gate_up_weight_scale_inv,
            bank.down_proj,
            bank.down_weight_scale_inv,
        )
        bank._commit_quantized_caches = lambda _staged: bank._record_master_identity()
        router = mlp.gate
        router._validate_refresh_source = lambda: None
        router._gather_refresh_master = lambda: router.weight_master
        router._stage_effective_view = lambda master: master.to(torch.bfloat16)
        router._commit_effective_view = lambda effective: (
            router._effective_weight.copy_(effective),
            router._record_master_identity(),
        )[-1]
    return root


def _combined_targets(directory: str) -> list[str]:
    payload = load_glm52_fullparam_payload(directory)
    return [item.target for item in payload.items]


@pytest.mark.cpu
def test_two_rank_step_publication_merges_to_one_complete_manifest(tmp_path) -> None:
    device = torch.device("cpu")
    publish_root = str(tmp_path / "publish")
    barrier_calls = []

    def _barrier():
        barrier_calls.append(True)

    # Rank 1 (expert contributor) first, then rank 0 (merger) — the
    # sequential stand-in for the barrier-ordered gang.
    receipt1 = glm52_fullparam_step_boundary(
        _rank_model(4, device, cpu_surrogate_refresh=True),
        step=1,
        publish_root=publish_root,
        rank=1,
        world_size=2,
        barrier=_barrier,
    )
    assert receipt1["published"] is True
    assert receipt1["contributed_items"] == _LOCAL_EXPERTS  # expert items only

    rank0_model = _rank_model(0, device, cpu_surrogate_refresh=True)
    receipt0 = glm52_fullparam_step_boundary(
        rank0_model,
        step=1,
        publish_root=publish_root,
        rank=0,
        world_size=2,
        barrier=_barrier,
    )
    assert receipt0["published"] is True
    assert receipt0["items"] == 1 + _GLOBAL_EXPERTS  # router + all global experts
    assert receipt0["contributing_ranks"] == 2
    assert len(barrier_calls) == 4

    step_dir = os.path.join(publish_root, "step_000001")
    assert receipt0["step_dir"] == step_dir
    combined_dir = os.path.join(step_dir, "combined")
    assert receipt0["combined_dir"] == combined_dir
    # Load fully re-verifies every checksum.
    assert _combined_targets(combined_dir) == sorted(
        ["model.layers.3.mlp.gate"] + [f"model.layers.3.mlp.experts.{i}" for i in range(_GLOBAL_EXPERTS)]
    )
    with open(os.path.join(step_dir, "COMMITTED.json"), encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["manifest_checksum"] == receipt0["manifest_checksum"]
    assert persisted["weight_step"] == 1

    # A second publication of the same step refuses to overwrite artifacts.
    with pytest.raises(Glm52FullParamPayloadError, match="already exists"):
        glm52_fullparam_step_boundary(
            rank0_model,
            step=1,
            publish_root=publish_root,
            rank=0,
            world_size=2,
            barrier=_barrier,
        )


@pytest.mark.cpu
def test_pp2_publication_includes_each_stage_scope_exactly_once(tmp_path, monkeypatch) -> None:
    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        iter_glm52_fullparam_publication,
    )

    device = torch.device("cpu")
    models = {
        0: _rank_model(0, device, cpu_surrogate_refresh=True, layer_index=3),
        1: _rank_model(4, device, cpu_surrogate_refresh=True, layer_index=3),
        2: _rank_model(0, device, cpu_surrogate_refresh=True, layer_index=7),
        3: _rank_model(4, device, cpu_surrogate_refresh=True, layer_index=7),
    }
    plans = []
    stage_by_rank = {0: 0, 1: 0, 2: 1, 3: 1}
    pp_group_by_rank = {0: (0, 2), 1: (1, 3), 2: (0, 2), 3: (1, 3)}
    ep_group_by_rank = {0: (0, 1), 1: (0, 1), 2: (2, 3), 3: (2, 3)}
    owner_key_by_rank = {0: (0,), 1: (1,), 2: (0,), 3: (1,)}
    for rank, model in models.items():
        components = iter_glm52_fullparam_publication(model)
        plans.append(
            {
                "rank": rank,
                "error": None,
                "targets": [
                    (target, getattr(module, "glm52_fullparam_payload_kind", ""))
                    for target, module in components
                ],
                "expected_targets": sorted(step_publish_module._expected_global_targets(model)),
                # These values model live group membership: two PP columns
                # (0,2)/(1,3), and one stage-local EP group per stage.
                "pp_stage": stage_by_rank[rank],
                "pp_group_ranks": pp_group_by_rank[rank],
                "ep_owner": ep_group_by_rank[rank].index(rank),
                "ep_group_ranks": ep_group_by_rank[rank],
                "owner_key": owner_key_by_rank[rank],
            }
        )

    monkeypatch.setattr(step_publish_module, "_gather_publication_plans", lambda _local, _world: plans)
    publish_root = str(tmp_path / "publish")
    for rank in (3, 2, 1):
        receipt = glm52_fullparam_step_boundary(
            models[rank],
            step=1,
            publish_root=publish_root,
            rank=rank,
            world_size=4,
            barrier=lambda: None,
        )
        assert receipt["published"] is True

    root_receipt = glm52_fullparam_step_boundary(
        models[0],
        step=1,
        publish_root=publish_root,
        rank=0,
        world_size=4,
        barrier=lambda: None,
    )
    expected = {
        f"model.layers.{layer}.mlp.gate" for layer in (3, 7)
    } | {
        f"model.layers.{layer}.mlp.experts.{expert}"
        for layer in (3, 7)
        for expert in range(_GLOBAL_EXPERTS)
    }
    combined_targets = _combined_targets(root_receipt["combined_dir"])
    assert set(combined_targets) == expected
    assert len(combined_targets) == len(expected)
    assert root_receipt["items"] == 2 * (1 + _GLOBAL_EXPERTS)
    assert root_receipt["contributing_ranks"] == 4


@pytest.mark.cpu
def test_missing_rank_contribution_fails_the_completeness_gate(tmp_path) -> None:
    publish_root = str(tmp_path / "publish")
    with pytest.raises(
        Glm52FullParamPayloadError,
        match="requires exactly one directory per selected live contributor",
    ):
        glm52_fullparam_step_boundary(
            _rank_model(0, torch.device("cpu"), cpu_surrogate_refresh=True),
            step=1,
            publish_root=publish_root,
            rank=0,
            world_size=2,
            barrier=lambda: None,
        )


@pytest.mark.cpu
def test_refresh_only_when_no_publish_root(tmp_path) -> None:
    model = _rank_model(0, torch.device("cpu"), cpu_surrogate_refresh=True)
    receipt = glm52_fullparam_step_boundary(
        model,
        step=3,
        publish_root=None,
        rank=0,
        world_size=1,
        barrier=lambda: None,
    )
    assert receipt == {
        "refreshed_components": 2,
        "published": False,
        "step": 3,
        "rank": 0,
    }
    assert not any(tmp_path.iterdir())


@pytest.mark.cpu
def test_step_validation_fails_closed() -> None:
    model = _rank_model(0, torch.device("cpu"), cpu_surrogate_refresh=True)
    for bad_step in (0, -1, True, "1"):
        with pytest.raises(Glm52FullParamPayloadError, match="positive post-step integer"):
            glm52_fullparam_step_boundary(
                model,
                step=bad_step,  # type: ignore[arg-type]
                publish_root=None,
                rank=0,
                world_size=1,
                barrier=lambda: None,
            )


@pytest.mark.cpu
def test_commit_marker_directory_fsync_failure_removes_visible_marker(tmp_path, monkeypatch) -> None:
    calls = 0

    def fail_commit_directory_fsync(_descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:  # staged file fsync succeeds; post-rename directory fsync fails
            raise OSError("injected directory fsync failure")

    monkeypatch.setattr(step_publish_module.os, "fsync", fail_commit_directory_fsync)
    with pytest.raises(Glm52FullParamPayloadError, match="visible marker was removed"):
        step_publish_module._write_commit_marker(str(tmp_path), {"weight_step": 1})
    assert calls == 3  # file, failed commit-dir fsync, successful cleanup-dir fsync
    assert not (tmp_path / "COMMITTED.json").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_real_gloo_publication_failures_terminate_every_rank() -> None:
    from tests.distributed.distributed_utils import run_distributed_script

    result = run_distributed_script(
        __file__,
        num_gpus=2,
        timeout=120,
        extra_env={"XORL_GLM52_STEP_PUBLISH_WORKER": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    result.assert_success("two-rank GLM-5.2 step-publication error rendezvous")


def _run_step_publish_worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    step_publish_module._live_publication_identity = lambda owner: {
        "pp_stage": 0,
        "pp_group_ranks": (owner,),
        "ep_owner": owner,
        "ep_group_ranks": (0, 1),
        "owner_key": (owner,),
    }
    base = os.path.join(tempfile.gettempdir(), f"glm52-step-rendezvous-{os.environ['MASTER_PORT']}")
    original_save = step_publish_module.save_glm52_fullparam_payload

    # A rank-local validation error is exchanged before the first master
    # gather, so every rank exits instead of one peer entering a DTensor
    # collective the failing rank skipped.
    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        refresh_glm52_fullparam_caches,
    )

    preflight_model = _rank_model(rank * _LOCAL_EXPERTS, torch.device("cpu"), cpu_surrogate_refresh=True)
    if rank == 1:
        preflight_model.model.layers[3].mlp.experts._validate_refresh_source = lambda: (_ for _ in ()).throw(
            RuntimeError("injected rank-1 refresh preflight failure")
        )
    preflight_error = None
    try:
        refresh_glm52_fullparam_caches(preflight_model)
    except RuntimeError as exc:
        preflight_error = str(exc)
    preflight_errors: list[str | None] = [None, None]
    dist.all_gather_object(preflight_errors, preflight_error)
    assert all("preflight failed collectively" in (error or "") for error in preflight_errors), preflight_errors

    # A rank-local quantization/staging error must not make that rank skip a
    # later master collective. Both ranks execute the gate and expert gather
    # seams, then both receive the final collective failure.
    staged_model = _rank_model(rank * _LOCAL_EXPERTS, torch.device("cpu"), cpu_surrogate_refresh=True)
    staged_gate = staged_model.model.layers[3].mlp.gate
    staged_bank = staged_model.model.layers[3].mlp.experts
    gate_gather = staged_gate._gather_refresh_master
    bank_gather = staged_bank._gather_refresh_masters
    gather_order = []

    def gather_gate():
        dist.all_reduce(torch.ones(1))
        gather_order.append("gate")
        return gate_gather()

    def gather_bank():
        dist.all_reduce(torch.ones(1))
        gather_order.append("experts")
        return bank_gather()

    staged_gate._gather_refresh_master = gather_gate
    staged_bank._gather_refresh_masters = gather_bank
    if rank == 1:

        def fail_gate_stage(_master):
            raise RuntimeError("injected rank-1 staging failure")

        staged_gate._stage_effective_view = fail_gate_stage

    staged_error = None
    try:
        refresh_glm52_fullparam_caches(staged_model)
    except RuntimeError as exc:
        staged_error = str(exc)
    assert gather_order == ["gate", "experts"], gather_order
    staged_errors: list[str | None] = [None, None]
    dist.all_gather_object(staged_errors, staged_error)
    assert all("failed collectively after completing the gather schedule" in (error or "") for error in staged_errors)

    def _run_case(case: str, injected_save) -> None:
        root = os.path.join(base, case)
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
        dist.barrier()
        step_publish_module.save_glm52_fullparam_payload = injected_save
        caught = None
        try:
            glm52_fullparam_step_boundary(
                _rank_model(rank * _LOCAL_EXPERTS, torch.device("cpu"), cpu_surrogate_refresh=True),
                step=1,
                publish_root=root,
                rank=rank,
                world_size=2,
            )
        except Glm52FullParamPayloadError as exc:
            caught = str(exc)
        finally:
            step_publish_module.save_glm52_fullparam_payload = original_save
        messages: list[str | None] = [None, None]
        dist.all_gather_object(messages, caught)
        assert all(messages), messages
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
        dist.barrier()

    def fail_rank1_local(payload, directory):
        if rank == 1 and directory.endswith("rank_01"):
            raise OSError("injected rank-1 local write failure")
        return original_save(payload, directory)

    _run_case("local", fail_rank1_local)

    def fail_rank0_combined(payload, directory):
        if rank == 0 and directory.endswith("combined"):
            raise OSError("injected rank-0 combined write failure")
        return original_save(payload, directory)

    _run_case("combined", fail_rank0_combined)
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_refresh_publishes_post_step_bytes_on_gpu(tmp_path) -> None:
    """End-to-end on CUDA: mutate masters -> step boundary -> the combined
    payload carries Q(post-step masters), not the seed bytes."""

    device = torch.device("cuda")
    publish_root = str(tmp_path / "publish")

    rank1_model = _rank_model(4, device, cpu_surrogate_refresh=False)
    rank0_model = _rank_model(0, device, cpu_surrogate_refresh=False)
    seed_router_bytes = rank0_model.model.layers[3].mlp.gate.publishable_weight_bytes().clone()

    for model in (rank1_model, rank0_model):
        with torch.no_grad():
            model.model.layers[3].mlp.experts.gate_up_weight_master.mul_(1.0625)
            model.model.layers[3].mlp.experts.down_weight_master.mul_(0.9375)
            model.model.layers[3].mlp.gate.weight_master.mul_(1.03125)

    glm52_fullparam_step_boundary(
        rank1_model, step=1, publish_root=publish_root, rank=1, world_size=2, barrier=lambda: None
    )
    receipt = glm52_fullparam_step_boundary(
        rank0_model, step=1, publish_root=publish_root, rank=0, world_size=2, barrier=lambda: None
    )
    assert receipt["published"] is True and receipt["items"] == 1 + _GLOBAL_EXPERTS

    payload = load_glm52_fullparam_payload(receipt["combined_dir"])
    by_target = {item.target: item for item in payload.items}

    # Router bytes are the refreshed post-step BF16 view.
    from xorl.server.weight_sync.glm52_fullparam_payload import unpack_glm52_payload_field

    router = rank0_model.model.layers[3].mlp.gate
    published_router = unpack_glm52_payload_field(by_target["model.layers.3.mlp.gate"].fields[0]).to(device)
    assert torch.equal(published_router, router.publishable_weight_bytes())
    assert not torch.equal(published_router, seed_router_bytes)

    # Expert bytes equal the refreshed cache's checkpoint form, from BOTH ranks.
    for model, global_ids in ((rank0_model, range(0, 4)), (rank1_model, range(4, 8))):
        bank = model.model.layers[3].mlp.experts
        for slot, global_id in enumerate(global_ids):
            expected = bank.publishable_expert_checkpoint_tensors(slot)
            item = by_target[f"model.layers.3.mlp.experts.{global_id}"]
            for field, expected_tensor in zip(item.fields, expected, strict=True):
                assert torch.equal(
                    unpack_glm52_payload_field(field).to(device).view(torch.uint8),
                    expected_tensor.contiguous().view(torch.uint8),
                ), f"{item.target}.{field.name} diverged from the refreshed cache"


if os.environ.get("XORL_GLM52_STEP_PUBLISH_WORKER") == "1":
    _run_step_publish_worker()
