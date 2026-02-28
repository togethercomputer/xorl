# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

import math
import os
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch
from torch import distributed as dist

from ..utils import logging
from ..utils.device import get_device_type
from ..utils.import_utils import is_torch_version_greater_than


if is_torch_version_greater_than("2.4"):
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

_PARALLEL_STATE: "ParallelState" = None


def requires_mesh(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner(self: "ParallelState", *args, **kwargs):
        if self.device_mesh is None:
            raise ValueError("Device mesh is not initialized.")

        return fn(self, *args, **kwargs)

    return _inner


def init_ep_mesh_matrix(ep_size: int, ep_fsdp_size: int, ep_outside: bool = False) -> "DeviceMesh":
    """
    Initialize the device mesh matrix for the EP.
    Args:
        ep_size (int): The size of the EP.
        ep_fsdp_size (int): The size of the EP-FSDP.
        ep_outside (bool): Whether the EP is outside in ep-fsdp group.
    """
    if ep_outside:
        with torch.device("cpu"):
            mesh = torch.arange(math.prod((ep_size, ep_fsdp_size)), dtype=torch.int).view(ep_size, ep_fsdp_size)
    else:
        with torch.device("cpu"):
            mesh = (
                torch.arange(math.prod((ep_size, ep_fsdp_size)), dtype=torch.int)
                .view(ep_fsdp_size, ep_size)
                .transpose(0, 1)
            )
    return mesh


@dataclass(frozen=True)
class ParallelState:
    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ulysses_size: int = 1
    dp_mode: Literal["none", "ddp", "fsdp2"] = "fsdp2"
    device_type: str = get_device_type()
    sp_fsdp_mode: Literal["all", "ulysses_only", "none"] = "all"
    device_mesh: Optional["DeviceMesh"] = None
    ep_fsdp_device_mesh: Optional["DeviceMesh"] = None
    _mesh_aliases: dict = field(default_factory=dict, repr=False)

    def _resolve_mesh_name(self, name: str) -> str:
        """Resolve a flattened mesh alias to the original dim name when flatten was skipped."""
        return self._mesh_aliases.get(name, name)

    def __post_init__(self):
        if self.sp_fsdp_mode not in ("all", "ulysses_only", "none"):
            raise ValueError(f"Invalid sp_fsdp_mode: {self.sp_fsdp_mode}. Must be 'all', 'ulysses_only', or 'none'.")

        if self.pp_size * self.dp_size * self.cp_size * self.ulysses_size * self.tp_size != self.world_size:
            raise ValueError("The product of parallel sizes should be equal to the world size.")

        if self.dp_replicate_size * self.dp_shard_size != self.dp_size:
            raise ValueError(
                f"The product of dp_replicate_size: {self.dp_replicate_size} and dp_shard_size: {self.dp_shard_size} should be equal to dp_size: {self.dp_size}."
            )

        if self.sp_enabled:
            from ..distributed.sequence_parallel import (
                init_sequence_parallel,
                set_context_parallel_group,
                set_data_parallel_group,
                set_ulysses_sequence_parallel_group,
                set_unified_sequence_parallel_group,
            )

            if self.device_mesh is not None:
                set_data_parallel_group(self.device_mesh.get_group(self._resolve_mesh_name("dp")))
                if self.ulysses_size > 1:
                    set_ulysses_sequence_parallel_group(self.device_mesh.get_group("ulysses"))
                if self.cp_size > 1:
                    set_context_parallel_group(self.device_mesh.get_group("cp"))
                # set unified sequence parallel group
                set_unified_sequence_parallel_group(self.device_mesh.get_group(self._resolve_mesh_name("sp")))
            else:
                init_sequence_parallel(
                    ulysses_size=self.ulysses_size,
                    sep_dp=True,
                    ulysses_group_key="default",
                    cp_size=self.cp_size,
                )

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "-1"))

    @property
    def global_rank(self) -> int:
        if self.is_initialized:
            return dist.get_rank()
        return -1

    @property
    def world_size(self) -> int:
        if self.is_initialized:
            return dist.get_world_size()
        return 1

    # ------------------------------ DP ------------------------------ #
    @property
    def dp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group(self._resolve_mesh_name("dp"))

        if self.sp_enabled:
            from ..distributed.sequence_parallel import get_data_parallel_group

            return get_data_parallel_group()

        return self.fsdp_group

    @property
    def dp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank(self._resolve_mesh_name("dp"))

        if self.sp_enabled:
            from ..distributed.sequence_parallel import get_data_parallel_rank

            return get_data_parallel_rank()

        return self.fsdp_rank

    @property
    @requires_mesh
    def dp_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp"]

        raise self.fsdp_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    # ------------------------------ Batch ------------------------------ #
    @property
    def batch_group(self) -> Optional["ProcessGroup"]:
        """Process group for data loading (dp_replicate x dp_shard).

        Ranks in the same batch group receive distinct batch items.
        Excludes Ulysses and CP since those ranks share the same data.
        """
        return self.dp_group

    @property
    @requires_mesh
    def batch_mesh(self) -> "DeviceMesh":
        """Device mesh for data loading (dp_replicate x dp_shard)."""
        return self.device_mesh["dp"]

    @property
    def batch_size(self) -> int:
        """Number of ranks that receive distinct batch items."""
        return self.dp_replicate_size * self.dp_shard_size

    # ------------------------------ Loss ------------------------------ #
    @property
    def loss_group(self) -> Optional["ProcessGroup"]:
        """Process group for loss reduction (dp_replicate x dp_shard x ulysses x cp).

        All ranks that compute partial losses on different data/sequence shards.
        """
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_sp")
        return self.fsdp_group

    @property
    @requires_mesh
    def loss_mesh(self) -> "DeviceMesh":
        """Device mesh for loss reduction (dp_replicate x dp_shard x ulysses x cp)."""
        return self.device_mesh["dp_sp"]

    @property
    def loss_size(self) -> int:
        """Number of ranks that participate in loss reduction."""
        return self.dp_replicate_size * self.dp_shard_size * self.ulysses_size * self.cp_size

    @property
    def loss_parallel_enabled(self) -> bool:
        """Whether loss reduction across multiple ranks is needed."""
        return self.dp_enabled or self.sp_enabled

    # ------------------------------ DP replicate ------------------------------ #
    @property
    def dp_replicate_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_replicate")

    @property
    def dp_replicate_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_replicate")

    @property
    @requires_mesh
    def dp_replicate_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_replicate"]

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate_size > 1

    # ------------------------------ DP shard ------------------------------ #
    @property
    def dp_shard_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_shard")

    @property
    def dp_shard_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_shard")

    @property
    @requires_mesh
    def dp_shard_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_shard"]

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard_size >= 1

    # ----------------------------- FSDP ----------------------------- #
    @property
    def fsdp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group(self._resolve_mesh_name("dp_sp"))

    @property
    def fsdp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank(self._resolve_mesh_name("dp_sp"))

        return self.global_rank

    @property
    def dp_shard_sp_enabled(self) -> bool:
        return self.dp_shard_enabled and self.sp_enabled

    @property
    @requires_mesh
    def fsdp_mesh(self) -> "DeviceMesh":
        dp_shard_sp_name = self._resolve_mesh_name("dp_shard_sp")
        if self.dp_replicate_enabled:
            # HSDP
            if self.dp_shard_sp_enabled:
                return self.device_mesh["dp_replicate", dp_shard_sp_name]
            elif self.dp_shard_enabled:
                return self.device_mesh["dp_replicate", "dp_shard"]
            else:
                # DDP
                return self.device_mesh["dp_replicate"]
        # FSDP
        elif self.dp_shard_sp_enabled:
            return self.device_mesh[dp_shard_sp_name]
        elif self.dp_shard_enabled:
            return self.device_mesh["dp_shard"]
        else:
            return self.device_mesh[self._resolve_mesh_name("dp")]

    @property
    def fsdp_enabled(self) -> bool:
        # FSDP is enabled if dp_mode is fsdp2, even with world_size=1
        # This allows using FSDP features like meta device initialization on single GPU
        return self.dp_mode == "fsdp2"

    @property
    def fsdp_size(self) -> int:
        return self.world_size // (self.pp_size * self.tp_size)

    # ------------------------------ TP ------------------------------ #
    @property
    @requires_mesh
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank("tp")

    @property
    @requires_mesh
    def tp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["tp"]

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    @property
    @requires_mesh
    def tp_group(self):
        return self.device_mesh.get_group("tp")

    # ------------------------------ PP ------------------------------ #
    @property
    @requires_mesh
    def pp_rank(self) -> int:
        return self.device_mesh.get_local_rank("pp")

    @property
    @requires_mesh
    def pp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["pp"]

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    @requires_mesh
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    @requires_mesh
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == (self.pp_size - 1)

    @property
    @requires_mesh
    def pp_group(self) -> "ProcessGroup":
        return self.pp_mesh.get_group()

    # ------------------------------ EP ------------------------------ #
    @property
    @requires_mesh
    def ep_mesh(self) -> "DeviceMesh":
        return self.ep_fsdp_device_mesh["ep"]

    @property
    @requires_mesh
    def ep_fsdp_mesh(self) -> "DeviceMesh":
        return self.ep_fsdp_device_mesh["ep", "ep_fsdp"]

    @property
    @requires_mesh
    def ep_group(self) -> "ProcessGroup":
        return self.ep_mesh.get_group()

    @property
    def ep_enabled(self) -> bool:
        return self.ep_size > 1

    @property
    def ep_rank(self) -> int:
        return self.ep_fsdp_device_mesh.get_local_rank("ep")

    # ------------------------------ SP ------------------------------ #
    @property
    def sp_group(self) -> Optional["ProcessGroup"]:
        if self.sp_enabled:
            if self.device_mesh is not None:
                return self.device_mesh.get_group(self._resolve_mesh_name("sp"))

            from .sequence_parallel import get_unified_sequence_parallel_group

            return get_unified_sequence_parallel_group()

        return None

    @property
    def sp_rank(self) -> int:
        if self.sp_enabled:
            if self.device_mesh is not None:
                return self.device_mesh.get_local_rank(self._resolve_mesh_name("sp"))

            from .sequence_parallel import get_unified_sequence_parallel_rank

            return get_unified_sequence_parallel_rank()

        return -1

    @property
    def sp_enabled(self) -> bool:
        return self.cp_size > 1 or self.ulysses_size > 1

    @property
    def sp_size(self) -> int:
        return self.ulysses_size * self.cp_size

    @property
    def ulysses_group(self) -> Optional["ProcessGroup"]:
        if self.ulysses_enabled:
            if self.device_mesh is not None:
                return self.device_mesh.get_group("ulysses")

            from .sequence_parallel import get_ulysses_sequence_parallel_group

            return get_ulysses_sequence_parallel_group()

        return None

    @property
    def ulysses_rank(self) -> int:
        if self.ulysses_enabled:
            if self.device_mesh is not None:
                return self.device_mesh.get_local_rank("ulysses")

            from .sequence_parallel import get_ulysses_sequence_parallel_rank

            return get_ulysses_sequence_parallel_rank()

        return -1

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_group(self) -> Optional["ProcessGroup"]:
        if self.cp_enabled:
            if self.device_mesh is not None:
                return self.device_mesh.get_group("cp")

            from .sequence_parallel import get_context_parallel_group

            return get_context_parallel_group()

        return None

    @property
    def cp_rank(self) -> int:
        if self.cp_enabled:
            if self.device_mesh is not None:
                return self.device_mesh.get_local_rank("cp")

            from .sequence_parallel import get_context_parallel_rank

            return get_context_parallel_rank()

        return -1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1

    @property
    def cp_grad_sync_group(self) -> Optional["ProcessGroup"]:
        """Returns CP group for gradient sync when CP is not folded into FSDP."""
        if self.sp_fsdp_mode == "ulysses_only" and self.cp_size > 1 and self.device_mesh is not None:
            return self.device_mesh.get_group("cp")
        return None


def init_parallel_state(
    dp_size: int = 1,
    dp_replicate_size: int = 1,
    dp_shard_size: int = 1,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ulysses_size: int = 1,
    dp_mode: Literal["none", "ddp", "fsdp2"] = "fsdp2",
    device_type: str = None,
    sp_fsdp_mode: Literal["all", "ulysses_only", "none"] = "all",
    ep_outside: bool = False,
) -> None:
    """
    Initializes global parallel state.
    """
    global _PARALLEL_STATE
    if _PARALLEL_STATE is not None:
        logger.warning("Parallel state has already been initialized.")
        return

    if device_type is None:
        device_type = get_device_type()

    # Set dp_shard_size to dp_size if dp_shard_size and dp_replicate_size are not set when dp enabled
    if dp_size > 1 and dp_shard_size == 1 and dp_replicate_size == 1:
        dp_shard_size = dp_size

    logger.info_rank0(
        f"Initializing parallel state... dp_size {dp_size}, dp_replicate_size {dp_replicate_size}, dp_shard_size {dp_shard_size},tp_size {tp_size}, pp_size {pp_size}, cp_size {cp_size}, ulysses_size {ulysses_size}"
    )

    device_mesh, ep_fsdp_device_mesh = None, None
    _mesh_aliases = {}
    if is_torch_version_greater_than("2.4"):
        mesh_shape = []
        mesh_dim_names = []
        for d, name in zip(
            [pp_size, dp_replicate_size, dp_shard_size, cp_size, ulysses_size, tp_size],
            ["pp", "dp_replicate", "dp_shard", "cp", "ulysses", "tp"],
        ):
            if d > 1 or name in ["dp_shard"]:
                mesh_shape.append(d)
                mesh_dim_names.append(name)

        device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple(mesh_shape),
            mesh_dim_names=tuple(mesh_dim_names),
        )

        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_sp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_sp_mesh_dim_names = []
        # Mesh for sequence parallel
        sp_mesh_dim_names = []

        if dp_replicate_size > 1:
            dp_mesh_dim_names.append("dp_replicate")
            dp_sp_mesh_dim_names.append("dp_replicate")
        if dp_shard_size >= 1:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_sp_mesh_dim_names.append("dp_shard")
            dp_sp_mesh_dim_names.append("dp_shard")
        # NOTE: append in mesh dimension order (dp_shard, cp, ulysses) to keep
        # indices ascending for DeviceMesh._flatten().
        if cp_size > 1:
            if sp_fsdp_mode == "all":
                dp_shard_sp_mesh_dim_names.append("cp")
            sp_mesh_dim_names.append("cp")
            dp_sp_mesh_dim_names.append("cp")
        if ulysses_size > 1:
            if sp_fsdp_mode in ("all", "ulysses_only"):
                dp_shard_sp_mesh_dim_names.append("ulysses")
            sp_mesh_dim_names.append("ulysses")
            dp_sp_mesh_dim_names.append("ulysses")

        def _safe_flatten(mesh, dim_names, flat_name):
            """Flatten mesh dims into a single alias. Skip when only 1 dim
            (already addressable by its original name) to avoid a PyTorch bug
            with size-1 flatten."""
            if len(dim_names) >= 2:
                mesh[tuple(dim_names)]._flatten(mesh_dim_name=flat_name)

        # Build alias -> original dim name(s) mapping for single-dim fallback
        _mesh_aliases = {}
        for flat_name, dim_names in [
            ("dp", dp_mesh_dim_names),
            ("dp_shard_sp", dp_shard_sp_mesh_dim_names),
            ("dp_sp", dp_sp_mesh_dim_names),
            ("sp", sp_mesh_dim_names),
        ]:
            if len(dim_names) >= 2:
                device_mesh[tuple(dim_names)]._flatten(mesh_dim_name=flat_name)
            elif len(dim_names) == 1:
                # Single dim: no flatten needed, record alias -> original name
                _mesh_aliases[flat_name] = dim_names[0]

        if ep_size > 1:
            world_size = dist.get_world_size()
            # EP mesh must be per-PP-stage to avoid cross-stage collectives
            # that deadlock during async pipeline execution.
            ranks_per_stage = world_size // pp_size
            assert ranks_per_stage % ep_size == 0, (
                f"ep_size ({ep_size}) must be a factor of ranks_per_pp_stage "
                f"({ranks_per_stage} = world_size {world_size} / pp_size {pp_size})"
            )
            ep_fsdp_size = ranks_per_stage // ep_size

            if pp_size > 1:
                # Create 3D mesh (_pp_ep, ep, ep_fsdp) so each PP stage gets
                # its own EP groups. Slicing by ["ep"] or ["ep", "ep_fsdp"]
                # automatically returns the per-stage submesh for each rank.
                with torch.device("cpu"):
                    pp_ep_mesh = torch.zeros(pp_size, ep_size, ep_fsdp_size, dtype=torch.int)
                    for pp_stage in range(pp_size):
                        stage_start = pp_stage * ranks_per_stage
                        stage_ranks = torch.arange(stage_start, stage_start + ranks_per_stage, dtype=torch.int)
                        if ep_outside:
                            pp_ep_mesh[pp_stage] = stage_ranks.view(ep_size, ep_fsdp_size)
                        else:
                            pp_ep_mesh[pp_stage] = stage_ranks.view(ep_fsdp_size, ep_size).transpose(0, 1)

                ep_fsdp_device_mesh = DeviceMesh(
                    device_type=device_type,
                    mesh=pp_ep_mesh,
                    mesh_dim_names=("_pp_ep", "ep", "ep_fsdp"),
                )
            else:
                mesh = init_ep_mesh_matrix(ep_size=ep_size, ep_fsdp_size=ep_fsdp_size, ep_outside=ep_outside)
                ep_fsdp_device_mesh = DeviceMesh(
                    device_type=device_type,
                    mesh=mesh,
                    mesh_dim_names=("ep", "ep_fsdp"),
                )

        logger.info_rank0(f"Device mesh: {device_mesh}")
        logger.info_rank0(f"EP FSDP device mesh: {ep_fsdp_device_mesh}")

    _PARALLEL_STATE = ParallelState(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        dp_mode=dp_mode,
        device_type=device_type,
        sp_fsdp_mode=sp_fsdp_mode,
        device_mesh=device_mesh,
        ep_fsdp_device_mesh=ep_fsdp_device_mesh,
        _mesh_aliases=_mesh_aliases if device_mesh is not None else {},
    )


def get_parallel_state() -> "ParallelState":
    """
    Returns global parallel state.
    """
    if _PARALLEL_STATE is None:
        logger.warning_once("Parallel state has not been initialized. returning default Single-process state.")
        return ParallelState()

    return _PARALLEL_STATE
