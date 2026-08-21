"""Distributed construction gate for GLM-5.2 full-param training.

The CUDA initialization path must apply the EP expert slice exactly once:
checkpoint loading pre-slices the tensor, and ``ParallelPlan.apply`` must
verify that storage instead of slicing it again.  Reduced component tests do
not cover the complete load -> admission -> wrap orchestration.

This test runs that orchestration end to end at WORLD8 / EP8 on one node,
with EP == Ulysses == WORLD, DP-shard 1, and an expert-FSDP singleton,
against a fabricated tiny
GLM-5.2 block-FP8 HF snapshot on disk, through the production code path:

  ``ModelLoader.load_model`` (meta init -> EP pre-shrink -> ``to_empty`` ->
  EP-aware filtered checkpoint load, real safetensors + the real GLM
  checkpoint handler) -> full-param admission -> ``build_parallelize_model``
  (real ``ParallelPlan.apply`` + ``fully_shard``) -> identity rebind.

and asserts, per rank, for EVERY expert bank:

1. post-load (pre-wrap) stored rows == declared_global // ep_size and
   stored bytes == the checkpoint's expert slice for this rank (localizes
   a failure to the load half);
2. post-wrap stored rows == declared_global // ep_size — the assertion
   that reads ``(1, ...)`` on the double-slice defect;
3. post-wrap stored bytes == the checkpoint's expert slice bitwise
   (per-expert-unique patterns, so a wrong-identity row can never pass);
4. the full-param bank's declared-local geometry, admission-assigned
   global range, and step-0 cache bytes vs the checkpoint slice;
5. one frozen dense projection's packed bytes vs the checkpoint pair
   (load-path byte fidelity outside the expert family).

Reduced-geometry seams (documented, same conventions as the reduced
backward gate): program flags are set directly on the config (the
production resolver output; the resolver itself hard-validates official
geometry) and the admission enters via ``_skip_geometry_validation=True``.
Everything between those two seams — the code under test — is the
production orchestration.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


_EXPERTS_GLOBAL = 64  # ep8 -> 8 local; a double slice reproduces (1, ...)
_HIDDEN = 256
_MOE_INTER = 128
_WORLD = 8
_TRAINABLE_EXPERT_LAYERS = (1,)  # layer 2 stays a frozen native bank
_SNAPSHOT_ENV = "XORL_GLM52_CONSTRUCTION_GATE_SNAPSHOT"
_SHARD_NAME = "model-00001-of-00001.safetensors"
_PROJ_SPECS = ("gate", "up", "down")


def _gate_config():
    """3-layer reduced geometry with the full-param program flag resolved ON.

    Mirrors tests/models/test_glm52_fullparam_reduced_backward_gate.py's
    exact-program configuration, with n_routed_experts=64 so EP8 leaves 8
    local experts and an erroneous second slice is visible (8 % 8 == 0).
    """

    from xorl.models.transformers.glm5.configuration_glm5 import Glm5Config

    exclusions = ["model.embed_tokens", "lm_head"]
    exclusions.extend(f"model.layers.{layer}.self_attn.indexers_proj" for layer in range(3))
    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": exclusions,
    }
    config = Glm5Config(
        vocab_size=256,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        moe_intermediate_size=_MOE_INTER,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=_EXPERTS_GLOBAL,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=64,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=4,
        index_topk_freq=4,
        index_skip_topk_offset=1,
        indexer_types=["full", "shared", "shared"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        pad_token_id=0,
        hidden_act="silu",
        quantization_config=quantization_config,
    )
    # Production program selection (resolve_glm52_contract_flags output for
    # the full-param lane); the resolver itself validates OFFICIAL geometry,
    # so at reduced geometry the resolved values are pinned directly — the
    # same seam the reduced backward gate documents.
    config._glm52_fullparam_training = True
    config._glm52_exact_contract = False
    config._ep_dispatch = "alltoall"
    return config


def _pattern(shape, offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
    return (((values * 3 + offset) % 29) - 14).reshape(shape).div(16.0)


def _pattern_fp8(shape, offset: int) -> torch.Tensor:
    return _pattern(shape, offset).to(torch.float8_e4m3fn)


def _pattern_scale(shape, offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
    return ((values + offset) % 13).add(1.0).div(64.0).reshape(shape)


def _expert_offset(layer: int, expert: int, proj: str) -> int:
    return layer * 1_000_003 + expert * 9_973 + _PROJ_SPECS.index(proj) * 101


def _build_meta_model(config):
    from xorl.models.module_utils import init_empty_weights
    from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM

    with init_empty_weights():
        return Glm5ForCausalLM._from_config(config, torch_dtype=torch.bfloat16)


def _fabricate_snapshot(config, snapshot_dir: str) -> None:
    """Write a complete tiny GLM-5.2 block-FP8 HF snapshot with per-expert
    unique deterministic bytes (identity-revealing, not just shape-revealing)."""

    import safetensors.torch as st

    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
    from xorl.ops.exact.block_fp8_native import NativeBlockFP8Linear

    model = _build_meta_model(config)
    state: dict[str, torch.Tensor] = {}
    covered: set[str] = set()
    dense_offset = 7
    for fqn, module in model.named_modules():
        if isinstance(module, Glm52NativeBlockFP8Experts):
            layer = int(fqn.split(".")[2])
            hidden, inter = module.hidden_size, module.intermediate_size
            for expert in range(module.num_experts):
                for proj in _PROJ_SPECS:
                    weight_shape = (hidden, inter) if proj == "down" else (inter, hidden)
                    scale_shape = (weight_shape[0] // 128, weight_shape[1] // 128)
                    offset = _expert_offset(layer, expert, proj)
                    state[f"{fqn}.{expert}.{proj}_proj.weight"] = _pattern_fp8(weight_shape, offset)
                    state[f"{fqn}.{expert}.{proj}_proj.weight_scale_inv"] = _pattern_scale(scale_shape, offset + 1)
            covered.update(
                f"{fqn}.{name}"
                for name in (
                    "gate_up_packed_weight_f32",
                    "gate_up_weight_scale_inv",
                    "down_packed_weight_f32",
                    "down_weight_scale_inv",
                )
            )
        elif isinstance(module, NativeBlockFP8Linear):
            source = module._source_fqn
            state[f"{source}.weight"] = _pattern_fp8((module.out_features, module.in_features), dense_offset)
            state[f"{source}.weight_scale_inv"] = _pattern_scale(tuple(module.weight_scale_inv.shape), dense_offset + 1)
            covered.update({f"{fqn}.packed_weight_f32", f"{fqn}.weight_scale_inv"})
            dense_offset += 2
    for name, parameter in model.named_parameters():
        if name in covered:
            continue
        state[name] = _pattern(tuple(parameter.shape), dense_offset).to(parameter.dtype)
        dense_offset += 1
    for name, _buffer in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            state[name] = _pattern((_EXPERTS_GLOBAL,), dense_offset).float()
            dense_offset += 1

    os.makedirs(snapshot_dir, exist_ok=True)
    st.save_file(state, os.path.join(snapshot_dir, _SHARD_NAME))
    weight_map = dict.fromkeys(state.keys(), _SHARD_NAME)
    with open(os.path.join(snapshot_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, handle)
    del model


def _expected_bank_slices(snapshot_dir: str, layer: int, expert_start: int, local_experts: int):
    """Reproduce the checkpoint handler's pair-buffer packing for one rank's slice."""

    from safetensors import safe_open

    from xorl.models.transformers.glm5.native_fp8 import pack_fp8_as_float32

    prefix = f"model.layers.{layer}.mlp.experts"
    with safe_open(os.path.join(snapshot_dir, _SHARD_NAME), framework="pt") as handle:

        def _pair(expert: int, proj: str):
            return (
                handle.get_tensor(f"{prefix}.{expert}.{proj}_proj.weight"),
                handle.get_tensor(f"{prefix}.{expert}.{proj}_proj.weight_scale_inv"),
            )

        experts = range(expert_start, expert_start + local_experts)
        gate = {expert: _pair(expert, "gate") for expert in experts}
        up = {expert: _pair(expert, "up") for expert in experts}
        down = {expert: _pair(expert, "down") for expert in experts}
    gate_up = torch.stack([torch.cat((gate[e][0], up[e][0]), dim=0).T.contiguous() for e in experts])
    gate_up_scale = torch.stack([torch.cat((gate[e][1], up[e][1]), dim=0).T.contiguous() for e in experts]).float()
    down_packed = torch.stack([down[e][0].T.contiguous() for e in experts])
    down_scale = torch.stack([down[e][1].T.contiguous() for e in experts]).float()
    return (
        pack_fp8_as_float32(gate_up),
        gate_up_scale,
        pack_fp8_as_float32(down_packed),
        down_scale,
    )


def _storage_tensor(value: torch.Tensor) -> torch.Tensor:
    """Materialize a (possibly DTensor) parameter/buffer's full stored value."""

    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        return value.full_tensor()
    return value


def _assert_bank_storage(
    stage: str,
    fqn: str,
    module,
    snapshot_dir: str,
    layer: int,
    expert_start: int,
    expected_local: int,
) -> None:
    names = (
        "gate_up_packed_weight_f32",
        "gate_up_weight_scale_inv",
        "down_packed_weight_f32",
        "down_weight_scale_inv",
    )
    stored = {name: _storage_tensor(getattr(module, name)).detach() for name in names}
    for name, tensor in stored.items():
        rows = int(tensor.shape[0])
        assert rows == expected_local, (
            f"{stage}: {fqn}.{name} stores {rows} expert rows, expected declared_global // ep = "
            f"{expected_local} (full storage shape {tuple(tensor.shape)}) — an EP slice was applied "
            f"{'twice' if rows < expected_local else 'zero times'} on this construction"
        )
    expected = _expected_bank_slices(snapshot_dir, layer, expert_start, expected_local)
    for name, expected_tensor in zip(names, expected, strict=True):
        mine = stored[name].cpu()
        expected_tensor = expected_tensor.to(mine.dtype)
        assert torch.equal(mine.view(torch.uint8), expected_tensor.view(torch.uint8)), (
            f"{stage}: {fqn}.{name} bytes do not match the checkpoint slice "
            f"[{expert_start}, {expert_start + expected_local}) — stored rows carry the wrong "
            "expert identity or corrupted content"
        )


def _run_construction_gate() -> None:
    from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state
    from xorl.distributed.torch_parallelize import build_parallelize_model
    from xorl.models.loader import ModelLoader
    from xorl.models.transformers.glm5.exact_fullparam_admission import (
        assert_glm52_fullparam_allowlist,
        install_glm52_fullparam_components,
        rebind_glm52_fullparam_master_identities,
    )
    from xorl.models.transformers.glm5.exact_fullparam_experts import (
        Glm52FullParamBlockFP8RoutedExperts,
    )
    from xorl.models.transformers.glm5.modeling_glm5 import Glm5ForCausalLM
    from xorl.models.transformers.glm5.native_fp8 import Glm52NativeBlockFP8Experts
    from xorl.ops.exact.block_fp8_native import NativeBlockFP8Linear

    snapshot_dir = os.environ[_SNAPSHOT_ENV]
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    assert dist.get_world_size() == _WORLD, f"construction gate requires WORLD{_WORLD}"

    config = _gate_config()
    if rank == 0:
        _fabricate_snapshot(config, snapshot_dir)
    dist.barrier()

    # EP == Ulysses == WORLD, DP-shard 1, expert-FSDP singleton.
    init_parallel_state(dp_size=1, ep_size=_WORLD, ulysses_size=_WORLD, dp_mode="fsdp2", cp_fsdp_mode="all")
    parallel_state = get_parallel_state()
    ep_rank = parallel_state.ep_rank
    expected_local = _EXPERTS_GLOBAL // _WORLD
    expert_start = ep_rank * expected_local

    # --- Production load (slice site 1): meta init -> EP pre-shrink ->
    # to_empty(cuda) -> EP-aware filtered load via the real handler.
    loader = ModelLoader(Glm5ForCausalLM._from_config, description="glm52-construction-gate")
    model = loader.load_model(
        init_kwargs={"config": config, "torch_dtype": torch.bfloat16},
        weights_path=snapshot_dir,
        empty_init=False,
        init_device="cuda",
    )

    frozen_banks = {
        fqn: module
        for fqn, module in model.named_modules()
        if isinstance(module, Glm52NativeBlockFP8Experts)
        and not getattr(module, "_glm52_exact_fullparam_component", False)
    }
    assert set(frozen_banks) == {"model.layers.1.mlp.experts", "model.layers.2.mlp.experts"}
    for fqn, module in frozen_banks.items():
        assert int(module.num_experts) == _EXPERTS_GLOBAL, "frozen banks declare the GLOBAL expert count"
        _assert_bank_storage(
            "post-load (pre-wrap)",
            fqn,
            module,
            snapshot_dir,
            layer=int(fqn.split(".")[2]),
            expert_start=expert_start,
            expected_local=expected_local,
        )

    # --- Production admission (step-5 seam; reduced-geometry entry).
    install_glm52_fullparam_components(
        model, config, trainable_expert_layers=_TRAINABLE_EXPERT_LAYERS, _skip_geometry_validation=True
    )

    # --- Production wrap (slice site 2 candidate): build_parallelize_model
    # with the exact step-6 full-param kwargs.
    model = build_parallelize_model(
        model,
        weights_path=snapshot_dir,
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        basic_modules=list(model._no_split_modules),
        init_device="cuda",
        glm52_fullparam_fp8_training=True,
        skip_param_upcast=True,
    )

    # --- Production step-8 full-param branch.
    rebind_glm52_fullparam_master_identities(model)
    assert_glm52_fullparam_allowlist(model)

    # --- THE GATE: post-wrap per-rank storage == declared // ep, bytes ==
    # the checkpoint's expert slice, for every frozen bank.
    post_wrap_frozen = {
        fqn: module
        for fqn, module in model.named_modules()
        if isinstance(module, Glm52NativeBlockFP8Experts)
        and not getattr(module, "_glm52_exact_fullparam_component", False)
    }
    assert set(post_wrap_frozen) == {"model.layers.2.mlp.experts"}, (
        "after scoped admission exactly layer 2 keeps a frozen native bank"
    )
    for fqn, module in post_wrap_frozen.items():
        _assert_bank_storage(
            "post-wrap",
            fqn,
            module,
            snapshot_dir,
            layer=int(fqn.split(".")[2]),
            expert_start=expert_start,
            expected_local=expected_local,
        )

    # --- The full-param bank: declared-local geometry, admission-assigned
    # global range, step-0 cache bytes == checkpoint slice.
    fullparam_bank = model.get_submodule("model.layers.1.mlp.experts")
    assert isinstance(fullparam_bank, Glm52FullParamBlockFP8RoutedExperts)
    assert int(fullparam_bank.num_experts) == expected_local, "full-param banks declare the EP-LOCAL size"
    assert int(fullparam_bank.num_global_experts) == _EXPERTS_GLOBAL
    assert tuple(fullparam_bank.global_expert_ids) == tuple(range(expert_start, expert_start + expected_local))
    for name in ("gate_up_weight_master", "down_weight_master"):
        master = _storage_tensor(getattr(fullparam_bank, name))
        assert int(master.shape[0]) == expected_local, f"{name} must hold the EP-local master rows"
    _assert_bank_storage(
        "post-wrap (full-param cache)",
        "model.layers.1.mlp.experts",
        fullparam_bank,
        snapshot_dir,
        layer=1,
        expert_start=expert_start,
        expected_local=expected_local,
    )

    # --- Frozen dense probe: packed bytes == checkpoint pair (byte fidelity
    # of the dense load path on the same construction).
    from xorl.models.transformers.glm5.native_fp8 import pack_fp8_as_float32

    probe = model.get_submodule("model.layers.2.self_attn.q_a_proj")
    assert isinstance(probe, NativeBlockFP8Linear)
    from safetensors import safe_open

    with safe_open(os.path.join(snapshot_dir, _SHARD_NAME), framework="pt") as handle:
        expected_weight = pack_fp8_as_float32(handle.get_tensor("model.layers.2.self_attn.q_a_proj.weight"))
        expected_scale = handle.get_tensor("model.layers.2.self_attn.q_a_proj.weight_scale_inv")
    stored_weight = _storage_tensor(probe.packed_weight_f32).detach().cpu()
    stored_scale = _storage_tensor(probe.weight_scale_inv).detach().cpu()
    assert torch.equal(stored_weight.view(torch.uint8), expected_weight.view(torch.uint8))
    assert torch.equal(stored_scale, expected_scale)

    dist.barrier()
    if rank == 0:
        print("CONSTRUCTION GATE PASS: single EP slice, byte-identical expert storage on all banks")
    dist.destroy_process_group()


if __name__ != "__main__":
    import shutil
    import tempfile

    import pytest
    import torch as _torch
    from distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

    @pytest.mark.gpu
    @skip_if_gpu_count_less_than(_WORLD)
    def test_glm52_fullparam_construction_single_ep_slice_and_byte_identity():
        snapshot_dir = tempfile.mkdtemp(prefix="glm52_construction_gate_")
        try:
            result = run_distributed_script(
                __file__,
                num_gpus=_WORLD,
                timeout=600,
                extra_env={_SNAPSHOT_ENV: snapshot_dir},
            )
            result.assert_success(
                "full-param construction must apply exactly ONE EP slice and preserve checkpoint bytes"
            )
        finally:
            shutil.rmtree(snapshot_dir, ignore_errors=True)

    del _torch


if __name__ == "__main__":
    _run_construction_gate()
