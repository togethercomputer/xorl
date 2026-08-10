import math
from typing import List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor

from ...utils.device import get_device_type
from ...utils.logging import get_logger
from ..parallel_state import get_parallel_state


logger = get_logger(__name__)


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    # EP-aware path (FSDP2 + EP)
    if hasattr(model, "_ep_param_groups"):
        return ep_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    # FSDP2 without EP: need distributed reductions across FSDP and TP groups
    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group

    reduce_groups = []
    if fsdp_group is not None:
        # get_world_size raises RuntimeError if the process group hasn't been
        # initialized; treat that as a single-process group rather than a hard
        # failure. Other exceptions (TypeError on bad argument, etc.) propagate.
        try:
            fsdp_world = dist.get_world_size(fsdp_group)
        except RuntimeError:
            fsdp_world = 1
        if fsdp_world > 1:
            reduce_groups.append(("fsdp", fsdp_group))

    # TP RowwiseParallel produces partial grads that need all-reduce across TP
    if ps.tp_enabled:
        reduce_groups.append(("tp", ps.tp_group))

    # Use custom reduce path to handle DTensor grads from TP
    return _fsdp2_reduce_and_clip(
        params=[p for p in model.parameters() if p.grad is not None],
        max_norm=max_norm,
        norm_type=norm_type,
        foreach=foreach,
        error_if_nonfinite=error_if_nonfinite,
        reduce_groups=reduce_groups,
    )


@torch.no_grad()
def ep_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    EP-aware gradient clipping for composable FSDP2:

    - Compute local norms for non-EP and EP parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-EP: all-reduce over FSDP group.
      • EP: all-reduce over EP-FSDP group, then over EP group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """

    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group
    ep_group = ps.ep_group if ps.ep_enabled else None
    # For EP params sharded by FSDP2 along hidden dimension
    ep_fsdp_group = None
    if ps.ep_enabled and ps.ep_fsdp_device_mesh is not None:
        ep_fsdp_group = ps.ep_fsdp_device_mesh["ep_fsdp"].get_group()

    # Build param groups (filter out params without grads).
    # Split EP params into disjoint expert rectangles, replicated shared
    # rectangles, and skip-FSDP plain tensors. Shared hybrid LoRA gradients are
    # synchronized by the backend backward or the coalesced post-backward
    # reduction; their replicated norm statistics are averaged below so the
    # logical factor is counted once.
    ep_replicated_ids = {id(p) for p in model._ep_param_groups.get("ep_replicated", [])}
    exact_lm_head_replicated_ids = {
        id(module.lora_A)
        for module in model.modules()
        if getattr(module, "_glm52_exact_tp16_lm_head", False) and getattr(module, "lora_A", None) is not None
    }
    ep_fsdp_params: List[torch.nn.Parameter] = []
    ep_local_params: List[torch.nn.Parameter] = []  # _skip_fsdp params (not sharded, not reduced)
    ep_replicated_fsdp_params: List[torch.nn.Parameter] = []
    ep_replicated_local_params: List[torch.nn.Parameter] = []
    for p in model._ep_param_groups.get("ep", []):
        if p.grad is None:
            continue
        if id(p) in ep_replicated_ids:
            if isinstance(p, DTensor):
                ep_replicated_fsdp_params.append(p)
            else:
                ep_replicated_local_params.append(p)
            continue
        if isinstance(p, DTensor):
            ep_fsdp_params.append(p)
        else:
            ep_local_params.append(p)
    exact_lm_head_replicated_params: List[torch.nn.Parameter] = [
        p
        for p in model._ep_param_groups.get("non_ep", [])
        if p.grad is not None and id(p) in exact_lm_head_replicated_ids
    ]
    non_ep_params: List[torch.nn.Parameter] = [
        p
        for p in model._ep_param_groups.get("non_ep", [])
        if p.grad is not None and id(p) not in exact_lm_head_replicated_ids
    ]

    # Note: torchtitan eFSDP design disables FSDP's automatic gradient division for ALL
    # modules (including experts) via disable_fsdp_gradient_division. Gradient normalisation
    # is handled uniformly by the loss (gradient_accumulate_loss). No additional EP-specific
    # scaling is needed here — doing so would double-divide expert grads.
    # ep_local_params (_skip_fsdp, e.g. QLoRAMoeExperts LoRA) are also left unscaled:
    # each rank trains its own unique local experts independently.

    # Compute and reduce non-EP norms across FSDP group
    non_ep_reduce_groups = [("fsdp", fsdp_group)]
    # TP RowwiseParallel produces partial grads that need all-reduce across TP
    if ps.tp_enabled:
        non_ep_reduce_groups.append(("tp", ps.tp_group))
    non_ep_total = _fsdp2_reduce_group(
        params=non_ep_params,
        norm_type=norm_type,
        reduce_groups=non_ep_reduce_groups,
    )
    # The exact TP16 lm-head A master is deliberately a plain replicated
    # Parameter. Its custom VJP already all-reduces dA over the full TP16
    # group, so reducing its norm over the main FSDP group would count the one
    # logical factor sixteen times. Count the identical local copy once.
    exact_lm_head_replicated_total = _fsdp2_reduce_group(
        params=exact_lm_head_replicated_params,
        norm_type=norm_type,
        reduce_groups=[],
    )

    # Compute and reduce FSDP-sharded EP norms across ep_fsdp, then ep
    ep_fsdp_total = _fsdp2_reduce_group(
        params=ep_fsdp_params,
        norm_type=norm_type,
        reduce_groups=[("ep_fsdp", ep_fsdp_group), ("ep", ep_group)],
    )

    # Local EP params: compute local norm only (no reduction — each rank has unique experts)
    ep_local_total = _fsdp2_reduce_group(
        params=ep_local_params,
        norm_type=norm_type,
        # _skip_fsdp expert tensors are not represented by an eFSDP DTensor;
        # each EP rank owns a distinct expert rectangle, so their local norms
        # must still enter one global logical norm.
        reduce_groups=[("ep", ep_group)],
    )

    # Shared tensors may be sharded over eFSDP and must contribute one logical
    # copy to the norm. The optimizer-boundary reducer must have synchronized
    # these gradients first; reduce their norm statistics across both groups,
    # then average the EP reduction so coherent replicated storage is not
    # counted once per rank.
    shared_reduce_groups = []
    if ep_fsdp_group is not None:
        shared_reduce_groups.append(("ep_fsdp", ep_fsdp_group))
    if ep_group is not None:
        shared_reduce_groups.append(("ep", ep_group))
    ep_replicated_fsdp_total = _fsdp2_reduce_group(
        params=ep_replicated_fsdp_params,
        norm_type=norm_type,
        reduce_groups=shared_reduce_groups,
    )
    ep_replicated_local_total = _fsdp2_reduce_group(
        params=ep_replicated_local_params,
        norm_type=norm_type,
        reduce_groups=[("ep", ep_group)] if ep_group is not None else [],
    )
    if not math.isinf(norm_type) and ep_group is not None and dist.is_available() and dist.is_initialized():
        ep_world = dist.get_world_size(ep_group)
        if ep_world > 1:
            ep_replicated_fsdp_total = ep_replicated_fsdp_total / ep_world
            ep_replicated_local_total = ep_replicated_local_total / ep_world

    if math.isinf(norm_type):
        total_norm = torch.maximum(
            torch.maximum(non_ep_total, ep_fsdp_total),
            torch.maximum(
                torch.maximum(ep_local_total, ep_replicated_fsdp_total),
                torch.maximum(ep_replicated_local_total, exact_lm_head_replicated_total),
            ),
        )
    else:
        total_norm = (
            non_ep_total
            + exact_lm_head_replicated_total
            + ep_fsdp_total
            + ep_local_total
            + ep_replicated_fsdp_total
            + ep_replicated_local_total
        ) ** (1.0 / float(norm_type))

    # Apply the same clip coefficient to all groups
    all_params = (
        ep_fsdp_params
        + ep_local_params
        + ep_replicated_fsdp_params
        + ep_replicated_local_params
        + exact_lm_head_replicated_params
        + non_ep_params
    )
    if error_if_nonfinite:
        finite_checks = [
            torch.isfinite(g.to_local() if isinstance(g, DTensor) else g).all()
            for param in all_params
            if param.grad is not None
            for g in [param.grad]
        ]
        if dist.is_available() and dist.is_initialized():
            device = next(
                (
                    (g.to_local() if isinstance(g, DTensor) else g).device
                    for param in all_params
                    if param.grad is not None
                    for g in [param.grad]
                ),
                torch.device(get_device_type()),
            )
            backend = dist.get_backend()
            if backend == "nccl" and device.type != "cuda":
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            local_finite_tensor = (
                torch.stack([check.to(device=device) for check in finite_checks]).all()
                if finite_checks
                else torch.ones((), dtype=torch.bool, device=device)
            )
            finite_flag = local_finite_tensor.to(dtype=torch.int64).reshape(1)
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            local_finite = bool(finite_flag.item())
        elif finite_checks:
            local_finite = bool(torch.stack(finite_checks).all().item())
        else:
            local_finite = True
        if not local_finite:
            raise RuntimeError("Non-finite gradient detected across the EP/FSDP optimizer group")
    # Disable foreach to avoid DTensor/plain tensor mixing
    torch.nn.utils.clip_grads_with_norm_(all_params, max_norm, total_norm, foreach=False)

    return total_norm


def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    dev = None
    acc = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
        # compute in FP32 for stability
        gn = torch.norm(g_local.detach().to(torch.float32), p=p)
        acc = acc + (gn**p)
    if acc is None:
        # no grads; choose a reasonable device
        dev = torch.device(get_device_type())
        acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return acc


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    dev = None
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device(get_device_type())
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return mx


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
        return val


def _fsdp2_reduce_and_clip(
    params: List[torch.nn.Parameter],
    max_norm: float,
    norm_type: float,
    foreach: bool | None,
    error_if_nonfinite: bool,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    if math.isinf(norm_type):
        total_norm = _fsdp2_reduce_group(params, norm_type, reduce_groups)
    else:
        total_p = _fsdp2_reduce_group(params, norm_type, reduce_groups)
        total_norm = total_p ** (1.0 / float(norm_type))

    # Disable foreach when mixing DTensor and plain tensor grads (e.g., TP meshes,
    # or QLoRA MoE experts excluded from FSDP via _skip_fsdp), or DTensor grads on
    # different device meshes (e.g., dp_shard + ep_fsdp without _ep_param_groups) —
    # torch's foreach path groups them into one op and raises a cross-mesh error.
    ps = get_parallel_state()
    grads = [p.grad for p in params if p.grad is not None]
    has_dtensor = any(isinstance(g, DTensor) for g in grads)
    has_plain = any(not isinstance(g, DTensor) for g in grads)
    grad_meshes = {g.device_mesh for g in grads if isinstance(g, DTensor)}
    if len(grad_meshes) > 1:
        logger.warning_once(
            "clip_grad_norm: gradients live on multiple device meshes "
            f"({sorted(str(m.mesh_dim_names) for m in grad_meshes)}); using per-tensor clipping. "
            "For EP models, build model._ep_param_groups first "
            "(xorl.distributed.torch_parallelize._build_ep_param_groups) to get the EP-aware path."
        )
    if foreach is None and (ps.tp_enabled or (has_dtensor and has_plain) or len(grad_meshes) > 1):
        foreach = False
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=foreach)
    return total_norm
