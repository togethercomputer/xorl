from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - torch 2.10+ always provides DTensor here
    DTensor = None


class _DefaultReduceScatterComm:
    """Minimal reduce-scatter implementation matching FSDP2's comm interface."""

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: dist.ReduceOp | dist.ReduceOp.RedOpType,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        return dist.reduce_scatter_tensor(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class DistSignReduceScatter:
    """Signs the flattened gradient buffer before FSDP2 reduces it."""

    def __init__(self, inner_comm: Optional[object] = None, *, sp_group: Optional[dist.ProcessGroup] = None):
        self._inner = inner_comm or _DefaultReduceScatterComm()
        self._sp_group = sp_group

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self._inner.allocate(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: dist.ReduceOp | dist.ReduceOp.RedOpType,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        if input_tensor.is_sparse:
            raise RuntimeError("DistSignSGD does not support sparse gradients.")
        if self._sp_group is not None:
            # Exact SP sums must happen before the nonlinear sign.
            dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, group=self._sp_group)
        input_tensor.sign_()
        # Force SUM regardless of `op`. FSDP2 may pass AVG (e.g. with
        # reduce_dtype=fp32), which would divide the sign-vote sum by N — and
        # the trainer's scale factor already divides by the actual voter total.
        # Inheriting AVG would double-divide and silently shrink updates.
        return self._inner(
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            op=dist.ReduceOp.SUM,
            async_op=async_op,
        )


def _local_sign_hook(grad: torch.Tensor) -> torch.Tensor:
    if grad.is_sparse:
        raise RuntimeError("DistSignSGD does not support sparse gradients.")
    return torch.sign(grad)


_FSDP_INTERNALS_ERROR = (
    "DistSignSGD relies on private torch.distributed.fsdp internals "
    "(`FSDPModule._get_fsdp_state()._fsdp_param_group.fsdp_params[*].sharded_param`). "
    "These have changed in your torch build, so the FSDP-managed vs local-hook split "
    "cannot be computed safely. Pin a compatible torch version or update DistSignSGD."
)


def _iter_fsdp_managed_sharded_params(model: torch.nn.Module):
    """Yield ``sharded_param`` tensors for every FSDP-managed parameter in ``model``.

    Raises a clear RuntimeError if any of the unstable internal attributes
    (`_get_fsdp_state`, `_fsdp_param_group`, `fsdp_params`, `sharded_param`)
    has shifted, so we never silently fall through to registering the local
    sign hook on FSDP-managed parameters (which would double-sign their grads).
    """
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        try:
            state = module._get_fsdp_state()
        except AttributeError as exc:
            raise RuntimeError(_FSDP_INTERNALS_ERROR) from exc
        fsdp_param_group = getattr(state, "_fsdp_param_group", None)
        if fsdp_param_group is None:
            continue
        try:
            fsdp_params = fsdp_param_group.fsdp_params
        except AttributeError as exc:
            raise RuntimeError(_FSDP_INTERNALS_ERROR) from exc
        for fsdp_param in fsdp_params:
            try:
                yield fsdp_param.sharded_param
            except AttributeError as exc:
                raise RuntimeError(_FSDP_INTERNALS_ERROR) from exc


def _get_fsdp_managed_param_ids(model: torch.nn.Module) -> set[int]:
    return {id(p) for p in _iter_fsdp_managed_sharded_params(model)}


def _assert_no_fsdp_managed_local_hook(model: torch.nn.Module) -> None:
    """Verify FSDP-managed parameters never received the local sign hook.

    Re-walking after registration is the explicit cross-check that catches
    the failure mode where a torch upgrade rearranges the FSDP2 internals
    we relied on but doesn't raise (e.g. an empty enumeration that lets
    every parameter slip through to ``_register_local_sign_hooks``).
    """
    for sharded in _iter_fsdp_managed_sharded_params(model):
        if getattr(sharded, "_distsign_local_hook_registered", False):
            raise RuntimeError(
                "DistSignSGD installed the local sign hook on an FSDP-managed parameter, "
                "which would double-sign gradients. " + _FSDP_INTERNALS_ERROR
            )


def _register_local_sign_hooks(model: torch.nn.Module, *, managed_param_ids: Optional[set[int]] = None) -> int:
    """Sign plain-tensor gradients before they accumulate across microbatches."""
    count = 0
    managed_param_ids = managed_param_ids or set()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in managed_param_ids:
            continue
        if DTensor is not None and isinstance(param, DTensor):
            # Non-FSDP DTensors (e.g. pure TP/PP shards) would get neither the
            # local sign hook nor the FSDP custom reduce-scatter, so their
            # gradients would skip the sign step entirely and pollute the
            # voter total. Refuse rather than silently drift.
            raise NotImplementedError(
                "DistSignSGD encountered a DTensor parameter that is not FSDP-managed "
                "(e.g. a pure TP/PP shard). This is not supported because such gradients "
                "would bypass the sign step. Please file an issue if this configuration "
                "is needed."
            )
        if getattr(param, "_distsign_local_hook_registered", False):
            continue
        param.register_hook(_local_sign_hook)
        param._distsign_local_hook_registered = True
        count += 1
    return count


def _configure_fsdp_reduce_scatter(model: torch.nn.Module, *, sp_group: Optional[dist.ProcessGroup]) -> int:
    count = 0
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        if getattr(module, "_distsign_reduce_scatter_configured", False):
            continue
        module.set_custom_reduce_scatter(DistSignReduceScatter(sp_group=sp_group))
        module._distsign_reduce_scatter_configured = True
        count += 1
    return count


def configure_distsignsgd(model: torch.nn.Module) -> None:
    """Install DistSignSGD communication and local grad hooks on a parallelized model."""
    if getattr(model, "_distsignsgd_configured", False):
        return

    ps = get_parallel_state()
    # Direct attribute access (not getattr-with-default) so that an attribute
    # rename on `ParallelState` surfaces as AttributeError instead of silently
    # falling back to the default and skipping the guard.
    if ps.dp_mode != "fsdp2":
        raise ValueError("DistSignSGD requires data_parallel_mode='fsdp2'.")
    if ps.dp_replicate_enabled:
        raise NotImplementedError("DistSignSGD does not yet support HSDP / dp_replicate_size > 1.")
    if ps.ep_enabled:
        raise NotImplementedError(
            "DistSignSGD does not yet support expert parallelism (EP). EP-managed grads "
            "live on a different fsdp group, so a single active_voter_total cannot "
            "normalize them correctly."
        )
    if ps.cp_enabled and ps.cp_fsdp_mode != "none":
        raise NotImplementedError(
            "DistSignSGD does not support folding sequence-parallel exact-sum dims into FSDP; set cp_fsdp_mode='none'."
        )

    managed_param_ids = _get_fsdp_managed_param_ids(model)
    _configure_fsdp_reduce_scatter(model, sp_group=ps.sp_grad_sync_group)
    _register_local_sign_hooks(model, managed_param_ids=managed_param_ids)
    _assert_no_fsdp_managed_local_hook(model)
    model._distsignsgd_configured = True


class DistSignSGD(Optimizer):
    """State-free SGD that expects gradients to be pre-signed before step()."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("DistSignSGD does not support sparse gradients.")

                if weight_decay:
                    p.add_(p, alpha=-lr * weight_decay)

                p.add_(grad, alpha=-lr)

        return loss
