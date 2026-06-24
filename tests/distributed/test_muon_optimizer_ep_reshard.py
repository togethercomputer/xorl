"""EP-resharding of Muon optimizer state across expert-parallel sizes (CPU/gloo).

Reproduces the resume failure where a DCP checkpoint saved under one
``expert_parallel_size`` could not be loaded under a different one because the Muon
expert ``momentum_buffer`` was stored as a plain local tensor (no DTensor metadata),
so DCP recorded a per-rank-local size and refused to reshard.

The two coordinated fixes under test:

1. ``Muon`` stores the expert momentum as a DTensor carrying the param's
   ``device_mesh`` + ``placements`` (math still runs on the local shard).
2. ``OptimizerState`` restores the EP dim on save and drops it on load — symmetric
   to ``ModelState`` — so DCP records the full global expert dimension and reshards.

Layout mirrors the live EP+FSDP2 expert weight: a fused ``gate_up_proj`` of shape
``[E, H, 2I]`` (and ``down_proj`` ``[E, I, H]``) sharded ``[Shard(0)=ep, Shard(1)=ep_fsdp]``.
The live model holds it as a 1-D ``ep_fsdp`` DTensor (``Shard(1)``) with the EP dim implicit.

World size is 4. We save under ``(ep=2, ep_fsdp=2)`` and load under ``(ep=4, ep_fsdp=1)``
— a genuine reshard — and assert the loaded momentum equals the gather/reshard of the
original (per-expert gather across EP ``Shard(0)``, concat across FSDP ``Shard(1)``),
NOT an average. A second case round-trips at the SAME ep_size and asserts bit-identity.

Runs on either backend: NCCL/CUDA when the launching host has >=4 GPUs, otherwise
CPU/gloo. The launcher (``run_distributed_script``) selects the mode and the subprocess
picks the backend + device from CUDA availability — no GPUs are required for CI.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import xorl.distributed.parallel_state as parallel_state_mod  # noqa: E402
from xorl.checkpoint.checkpointer import OptimizerState  # noqa: E402
from xorl.distributed.parallel_plan import SpecInfo  # noqa: E402
from xorl.distributed.parallel_state import get_parallel_state, init_parallel_state  # noqa: E402
from xorl.optim.multi_optimizer import MultiOptimizer  # noqa: E402
from xorl.optim.muon import Muon  # noqa: E402


# Toy MoE geometry. Global expert count must be divisible by every ep_size tested (2, 4).
NUM_EXPERTS = 8
HIDDEN = 4  # FSDP shard dim; divisible by every ep_fsdp_size tested (2, 1).
INTERMEDIATE = 3
EXPERT_FQN = "model.layers.0.mlp.experts.gate_up_proj"


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _use_cuda() -> bool:
    """Run on NCCL/CUDA when GPUs are visible to this rank, else CPU/gloo."""
    return torch.cuda.is_available()


def _device_type() -> str:
    return "cuda" if _use_cuda() else "cpu"


def _backend() -> str:
    return "nccl" if _use_cuda() else "gloo"


def _reset_parallel_state() -> None:
    parallel_state_mod._PARALLEL_STATE = None


def _init_ep(ep_size: int) -> None:
    world = _world_size()
    init_parallel_state(
        dp_size=world,
        dp_shard_size=world,
        ep_size=ep_size,
        dp_mode="fsdp2",
        device_type=_device_type(),
    )


def _global_gate_up() -> torch.Tensor:
    """Deterministic global fused gate_up weight ``[E, H, 2I]`` (same on every rank)."""
    torch.manual_seed(0)
    return torch.randn(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE, dtype=torch.float32)


def _global_grad() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE, dtype=torch.float32)


def _build_expert_module(ep_size: int):
    """Build a 1-param toy module whose expert weight matches the live EP+FSDP2 layout.

    Returns ``(module, ep_fsdp_mesh)``. The param is a 1-D ``ep_fsdp`` DTensor sharded on
    dim 1 (FSDP), with the EP slice taken locally on dim 0 (EP implicit) — exactly what
    ``parallelize_model_fsdp2`` produces for fused expert weights via ``Shard(1)`` FSDP +
    EP slicing. ``model._fqn2spec_info`` is stamped so the checkpointer is EP-aware.
    """
    ps = get_parallel_state()
    ep_fsdp_mesh = ps.ep_fsdp_device_mesh  # 2-D ["ep", "ep_fsdp"]
    ep_rank = ep_fsdp_mesh.get_local_rank("ep")
    fsdp_rank = ep_fsdp_mesh.get_local_rank("ep_fsdp")
    fsdp_size = ep_fsdp_mesh.size(ep_fsdp_mesh.mesh_dim_names.index("ep_fsdp"))

    full = _global_gate_up()
    experts_per_ep = NUM_EXPERTS // ep_size
    ep_slice = full[ep_rank * experts_per_ep : (ep_rank + 1) * experts_per_ep]  # [E_local, H, 2I]
    # FSDP shards dim 1 (H) within this EP rank's experts.
    h_per_fsdp = HIDDEN // fsdp_size
    local = ep_slice[:, fsdp_rank * h_per_fsdp : (fsdp_rank + 1) * h_per_fsdp, :].contiguous()
    local = local.to(_device_type())

    ep_fsdp_only_mesh = ep_fsdp_mesh["ep_fsdp"]
    weight_d = DTensor.from_local(local.clone(), ep_fsdp_only_mesh, [Shard(1)], run_check=False)

    module = nn.Module()
    param = nn.Parameter(weight_d)
    # Register under the nested FQN via a parameter name with dots is not allowed, so use a
    # container module path that yields ``model.layers.0.mlp.experts.gate_up_proj``.
    holder = module
    for seg in ["model", "layers", "0", "mlp", "experts"]:
        child = nn.Module()
        holder.add_module(seg, child)
        holder = child
    holder.register_parameter("gate_up_proj", param)

    module._fqn2spec_info = {
        EXPERT_FQN: SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Shard(0), fqn=EXPERT_FQN),
    }
    return module, ep_fsdp_mesh


def _build_optimizer(module: nn.Module) -> MultiOptimizer:
    param = dict(module.named_parameters())[EXPERT_FQN]
    muon = Muon(
        [{"params": [param], "use_muon": True, "lr": 0.1, "weight_decay": 0.0}],
        lr=0.1,
        momentum=0.9,
        nesterov=False,
        ns_steps=5,
        weight_decay=0.0,
        distributed_mode="shard_local",
    )
    return MultiOptimizer(module, {"ep": muon}, key_names=["ep"])


def _take_step(module: nn.Module, optimizer: MultiOptimizer) -> None:
    param = dict(module.named_parameters())[EXPERT_FQN]
    grad_full = _global_grad()
    ps = get_parallel_state()
    ep_fsdp_mesh = ps.ep_fsdp_device_mesh
    ep_rank = ep_fsdp_mesh.get_local_rank("ep")
    fsdp_rank = ep_fsdp_mesh.get_local_rank("ep_fsdp")
    fsdp_size = ep_fsdp_mesh.size(ep_fsdp_mesh.mesh_dim_names.index("ep_fsdp"))
    ep_size = ep_fsdp_mesh.size(ep_fsdp_mesh.mesh_dim_names.index("ep"))

    experts_per_ep = NUM_EXPERTS // ep_size
    h_per_fsdp = HIDDEN // fsdp_size
    g_ep = grad_full[ep_rank * experts_per_ep : (ep_rank + 1) * experts_per_ep]
    g_local = g_ep[:, fsdp_rank * h_per_fsdp : (fsdp_rank + 1) * h_per_fsdp, :].contiguous()
    g_local = g_local.to(_device_type())
    param.grad = DTensor.from_local(g_local, ep_fsdp_mesh["ep_fsdp"], [Shard(1)], run_check=False)
    optimizer.step()


def _local_momentum(optimizer: MultiOptimizer) -> torch.Tensor:
    muon = optimizer.optimizers_dict["ep"]
    (state,) = list(muon.state.values())
    buf = state["momentum_buffer"]
    # CHANGE 1: the buffer must be a DTensor so DCP can reshard it.
    assert isinstance(buf, DTensor), f"momentum_buffer must be a DTensor for EP reshard, got {type(buf)}"
    # Move to CPU so the gather/compare below is device-agnostic (works for nccl + gloo).
    return buf._local_tensor.detach().cpu().clone()


def _gather_global_momentum(local_buf: torch.Tensor) -> torch.Tensor:
    """Reconstruct the global ``[E, H, 2I]`` momentum from per-rank local shards.

    Per-expert gather across EP ``Shard(0)``, concat across FSDP ``Shard(1)`` — NOT averaged.
    """
    ps = get_parallel_state()
    ep_fsdp_mesh = ps.ep_fsdp_device_mesh
    ep_rank = ep_fsdp_mesh.get_local_rank("ep")
    fsdp_rank = ep_fsdp_mesh.get_local_rank("ep_fsdp")
    fsdp_size = ep_fsdp_mesh.size(ep_fsdp_mesh.mesh_dim_names.index("ep_fsdp"))
    ep_size = ep_fsdp_mesh.size(ep_fsdp_mesh.mesh_dim_names.index("ep"))

    world = dist.get_world_size()
    payloads = [None] * world
    dist.all_gather_object(
        payloads,
        {"ep_rank": ep_rank, "fsdp_rank": fsdp_rank, "buf": local_buf},
    )

    experts_per_ep = NUM_EXPERTS // ep_size
    h_per_fsdp = HIDDEN // fsdp_size
    full = torch.zeros(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE, dtype=local_buf.dtype)
    for item in payloads:
        er, fr, buf = item["ep_rank"], item["fsdp_rank"], item["buf"]
        e0 = er * experts_per_ep
        h0 = fr * h_per_fsdp
        full[e0 : e0 + experts_per_ep, h0 : h0 + h_per_fsdp, :] = buf
    return full


def _save(checkpoint_dir: str, module: nn.Module, optimizer: MultiOptimizer) -> None:
    save_state = {"optimizer": OptimizerState(model=module, optimizer=optimizer)}
    dcp.save(state_dict=save_state, storage_writer=FileSystemWriter(checkpoint_dir))
    dist.barrier()


def _load(checkpoint_dir: str, module: nn.Module, optimizer: MultiOptimizer) -> None:
    load_state = {"optimizer": OptimizerState(model=module, optimizer=optimizer)}
    dcp.load(state_dict=load_state, storage_reader=FileSystemReader(checkpoint_dir))
    dist.barrier()


def _run() -> None:
    if _use_cuda():
        torch.cuda.set_device(_local_rank())
    dist.init_process_group(backend=_backend())
    rank = dist.get_rank()
    assert _world_size() == 4, f"expected world_size 4, got {_world_size()}"

    save_ep = int(os.environ["XORL_TEST_SAVE_EP"])
    load_ep = int(os.environ["XORL_TEST_LOAD_EP"])

    shared_dir = os.environ["XORL_TEST_CKPT_DIR"]

    # --- Phase 1: build + step + save under save_ep ---
    _init_ep(save_ep)
    module = None
    optimizer = None
    module, _ = _build_expert_module(save_ep)
    optimizer = _build_optimizer(module)
    _take_step(module, optimizer)

    saved_local = _local_momentum(optimizer)
    saved_global = _gather_global_momentum(saved_local)
    _save(shared_dir, module, optimizer)

    # --- Phase 2: re-init under load_ep, fresh optimizer, load, compare ---
    _reset_parallel_state()
    _init_ep(load_ep)
    module2, _ = _build_expert_module(load_ep)
    optimizer2 = _build_optimizer(module2)
    # Populate fresh (zero-ish) state so set_optimizer_state_dict has a target shape;
    # take one step so the state dict has a momentum_buffer entry to load into.
    _take_step(module2, optimizer2)
    _load(shared_dir, module2, optimizer2)

    loaded_local = _local_momentum(optimizer2)
    loaded_global = _gather_global_momentum(loaded_local)

    if rank == 0:
        max_err = (loaded_global - saved_global).abs().max().item()
        # Same-ep round-trip must be a numeric no-op (bit-identical). Cross-ep reshard is
        # an exact gather/concat (no interpolation), so it should also be bit-exact here,
        # but allow a tiny tolerance against incidental fp reorderings.
        tol = 0.0 if save_ep == load_ep else 1e-5
        assert max_err <= tol, (
            f"EP-resharded momentum (save_ep={save_ep}->load_ep={load_ep}) does not match "
            f"the gather/reshard of the original: max abs err = {max_err} (tol={tol})"
        )
        # Guard against an "averaged" reconstruction masquerading as success: the global
        # tensor must not be near-constant across experts (the source is random per expert).
        per_expert_std = saved_global.flatten(1).std(dim=1)
        assert per_expert_std.min().item() > 1e-3, "test premise broken: experts not distinct"
        print(
            f"[rank 0] EP reshard {save_ep}->{load_ep}: loaded momentum matches "
            f"gather/reshard of original (max abs err {max_err:.3e}, tol {tol:.0e})"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ != "__main__":
    import pytest
    import torch
    from distributed_utils import run_distributed_script

    SCRIPT_PATH = os.path.abspath(__file__)
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # The reshard needs 4 ranks. Run on NCCL/CUDA when the host has >=4 GPUs, else CPU/gloo.
    # (No GPUs required — CI keeps running on CPU. The subprocess picks backend+device from
    # torch.cuda.is_available(); we just gate whether GPUs are exposed to the ranks.)
    _GPU_RANKS = torch.cuda.is_available() and torch.cuda.device_count() >= 4

    def _run_case(save_ep: int, load_ep: int):
        extra_env = {
            "PYTHONPATH": os.path.join(REPO_ROOT, "src"),
            "XORL_TEST_SAVE_EP": str(save_ep),
            "XORL_TEST_LOAD_EP": str(load_ep),
        }
        if not _GPU_RANKS:
            extra_env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU/gloo when GPUs unavailable
        with tempfile.TemporaryDirectory(prefix="muon_ep_reshard_") as ckpt_dir:
            extra_env["XORL_TEST_CKPT_DIR"] = ckpt_dir
            result = run_distributed_script(
                SCRIPT_PATH,
                num_gpus=4,  # 4 ranks (torchrun nproc_per_node); NCCL+CUDA or CPU+gloo
                timeout=300,
                extra_env=extra_env,
            )
            result.assert_success(f"Muon EP-reshard save_ep={save_ep} load_ep={load_ep}")

    @pytest.mark.cpu
    @pytest.mark.distributed
    def test_muon_momentum_reshard_ep2_to_ep4():
        """Save under ep_size=2 (ep_fsdp=2), load under ep_size=4 (ep_fsdp=1)."""
        _run_case(save_ep=2, load_ep=4)

    @pytest.mark.cpu
    @pytest.mark.distributed
    def test_muon_momentum_reshard_ep4_to_ep2():
        """Save under ep_size=4 (ep_fsdp=1), load under ep_size=2 (ep_fsdp=2)."""
        _run_case(save_ep=4, load_ep=2)

    @pytest.mark.cpu
    @pytest.mark.distributed
    def test_muon_momentum_same_ep_size_is_identity():
        """Same-ep_size round-trip: numeric no-op (bit-identical momentum)."""
        _run_case(save_ep=2, load_ep=2)


if __name__ == "__main__":
    _run()
