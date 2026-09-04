"""ScaledSGD: SGD over matrix parameters, optionally Muon-RMS-matched, DTensor-aware.

Built as the no-Newton-Schulz counterpart of the ZORL fresh_ab fold's Muon
(``_get_zorl_fresh_ab_base_optimizer``), so update RULES can be A/B'd while
holding everything else (momentum form, sharding, lr semantics) fixed.

Momentum is byte-identical in FORM to xorl ``Muon`` / ``torch.optim.Muon``
(EMA + Nesterov look-ahead), NOT the classical Keller-Jordan accumulation
``buf = mu*buf + g``:

  B_t  = (1-mu) * g_t + mu * B_{t-1}          (EMA; buffer skipped when mu=0)
  ~B_t = (1-mu) * g_t + mu * B_t   if nesterov else B_t

The two forms differ only by the global scale ``(1-mu)`` (EMA == classical
times ``1-mu``). For ``scale_mode='match_muon_rms'`` the per-matrix
normalization makes the applied update invariant to that scale — exactly as
Muon's NS pre-normalization does — so momentum composes identically across the
Muon and SGD fold arms. Only ``scale_mode='raw'`` sees the ``(1-mu)`` factor
(where EMA is also the better-behaved choice: the raw update magnitude stays
at gradient scale for any mu instead of growing by ``1/(1-mu)``).

Scale modes (the step is ``p -= <scaled>(~B_t)``; with the ZORL fold's
``grad = -G`` convention the applied delta is ``+<scaled>(G̃)``):

  - ``raw``: ``p -= lr * ~B_t``. Plain SGD in the ES-natural parameterization
    (G IS the estimate of ∇E[R]); the lr does NOT transfer from a Muon arm.

  - ``match_muon_rms``: per NS-granularity matrix m (each expert matrix of a
    3D+ stack; each half of a fused ``gate_up_proj`` param):

        p_m -= adjusted_lr * sqrt(min(A_m, B_m)) * ~B_m / max(||~B_m||_F, eps)

    with ``adjusted_lr = _adjust_lr(lr, adjust_lr_fn, <global pre-split
    matrix shape>)`` — the SAME call Muon makes (for fused gate_up params the
    shape is the fused ``[K, 2I]``, matching Muon's fused-shape adjustment).
    This is the Muon step with NS replaced by "keep the raw direction, carry
    the Frobenius magnitude an exactly-orthogonalized update would have"
    (``||NS(X)||_F -> sqrt(min(A, B))`` as NS singular values -> 1). The
    per-matrix RMS of the applied update is exactly
    ``adjusted_lr / sqrt(max(A_m, B_m))`` — for ``match_rms_adamw`` and
    non-fused params that is ``0.2 * lr``, the same target RMS Muon's
    ``match_rms_adamw`` scaling produces — so the Muon lr transfers unchanged.

DTensor (FSDP2/EP) support:
  - All arithmetic runs on the LOCAL shard; the momentum buffer is allocated
    at local-shard size and wrapped as a DTensor carrying the param's
    mesh/placements (via ``Muon._init_momentum_buffer``) so DCP records the
    true global size. Full-shape state is NEVER materialized.
  - ``match_muon_rms`` per-matrix Frobenius norms: batch-dim shards (e.g. the
    ``Shard(0)`` expert dim of ``[E, K, N]`` MoE stacks — the ZORL fold
    layout) keep whole matrices rank-local, so norms are local and the step
    is communication-free. Matrix-dim shards (2D ``Shard(0)``: the
    shared_expert projections) need one tiny all_reduce of the per-matrix
    squared norms over each matrix-sharding mesh dim — scalars, not gathers;
    there is no full-matrix gather anywhere (the update itself is an
    elementwise rescale, exact on the local shard once the norm is global).
"""

from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.tensor import Shard
from torch.optim._muon import _adjust_lr
from torch.optim.optimizer import Optimizer

from ..utils import logging
from .muon import Muon


logger = logging.get_logger(__name__)

SCALE_MODE_RAW = "raw"
SCALE_MODE_MATCH_MUON_RMS = "match_muon_rms"
_SCALE_MODES = (SCALE_MODE_RAW, SCALE_MODE_MATCH_MUON_RMS)


class ScaledSGD(Optimizer):
    """SGD with optional Muon-RMS-matched per-matrix scaling (see module docstring).

    Args:
        params: Iterable of parameters or param-group dicts. Groups may carry
            ``_fused_gate_up_ids`` (ids of params whose last dim is a fused
            ``[gate | up]`` concat) exactly like Muon's groups.
        lr: Learning rate. For ``match_muon_rms`` this is the MUON lr — the
            per-matrix step becomes ``_adjust_lr(lr, adjust_lr_fn, shape)``.
        momentum: EMA momentum coefficient in [0, 1). 0 skips the buffer.
        nesterov: Nesterov look-ahead (same lerp form as xorl Muon).
        weight_decay: Decoupled weight decay ``p *= 1 - lr * wd``.
        scale_mode: ``'raw'`` or ``'match_muon_rms'``.
        adjust_lr_fn: lr adjustment for ``match_muon_rms`` (Muon semantics;
            the fold recipe uses ``'match_rms_adamw'`` = ``0.2*sqrt(max(A,B))``).
        momentum_dtype: If set, force momentum buffers to this dtype (e.g.
            ``torch.bfloat16`` halves buffer memory). Default inherits the
            gradient dtype.
        eps: Frobenius-norm clamp floor for ``match_muon_rms``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        scale_mode: str = SCALE_MODE_MATCH_MUON_RMS,
        adjust_lr_fn: Optional[str] = "match_rms_adamw",
        momentum_dtype: Optional[torch.dtype] = None,
        eps: float = 1e-7,
    ):
        if scale_mode not in _SCALE_MODES:
            raise ValueError(f"Unsupported ScaledSGD scale_mode: {scale_mode!r}. Expected one of {_SCALE_MODES}.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"ScaledSGD momentum must be in [0, 1), got {momentum}")
        if nesterov and momentum == 0.0:
            raise ValueError("ScaledSGD nesterov=True requires momentum > 0")
        self._momentum_dtype = momentum_dtype
        self._logged_dtypes = False
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            scale_mode=scale_mode,
            adjust_lr_fn=adjust_lr_fn,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._sgd_group_step(group)
        return loss

    def _sgd_group_step(self, group: dict) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["weight_decay"]
        scale_mode = group["scale_mode"]
        adjust_lr_fn = group["adjust_lr_fn"]
        eps = group["eps"]
        fused_gate_up_ids = group.get("_fused_gate_up_ids", set())

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("ScaledSGD does not support sparse gradients.")
            is_dtensor = isinstance(grad, DTensor)
            grad_local = grad._local_tensor if is_dtensor else grad
            p_local = p._local_tensor if is_dtensor else p.data

            # --- Momentum: EMA + Nesterov look-ahead, identical form to Muon._muon_step.
            # No leading-dim reshape happens here, so grad_local always keeps the
            # param's local shape and the buffer is always DTensor-wrapped for
            # DTensor params (shard-local storage, global metadata for DCP).
            if momentum == 0:
                update = grad_local
            else:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf_dtype = self._momentum_dtype or grad_local.dtype
                    state["momentum_buffer"] = Muon._init_momentum_buffer(p, grad_local, buf_dtype)
                buf = state["momentum_buffer"]
                buf_local = buf._local_tensor if isinstance(buf, DTensor) else buf
                # EMA momentum: B = (1-mu)*g + mu*B
                buf_local.lerp_(grad_local.to(buf_local.dtype), 1 - momentum)
                # Nesterov: ~B = (1-mu)*g + mu*B  (else just B). NOTE: with
                # nesterov=False ``update`` ALIASES the buffer — everything
                # below must stay out-of-place on ``update``.
                update = grad_local.to(buf_local.dtype).lerp(buf_local, momentum) if nesterov else buf_local

            if not self._logged_dtypes:
                logger.info_rank0(
                    f"ScaledSGD dtypes: param={p_local.dtype}, grad={grad_local.dtype}, "
                    f"update={update.dtype}, scale_mode={scale_mode}, momentum={momentum}, "
                    f"nesterov={nesterov} (shape={list(grad_local.shape)})"
                )
                self._logged_dtypes = True

            # Decoupled weight decay (fold recipe uses 0.0).
            if weight_decay != 0.0:
                p_local.mul_(1.0 - lr * weight_decay)

            if scale_mode == SCALE_MODE_RAW:
                p_local.add_(update.to(p_local.dtype), alpha=-lr)
                continue

            self._apply_match_muon_rms_update(
                p,
                p_local,
                update,
                grad,
                is_dtensor=is_dtensor,
                lr=lr,
                adjust_lr_fn=adjust_lr_fn,
                eps=eps,
                is_fused_gate_up=id(p) in fused_gate_up_ids,
            )

    def _apply_match_muon_rms_update(
        self,
        p: torch.Tensor,
        p_local: torch.Tensor,
        update: torch.Tensor,
        grad: torch.Tensor,
        *,
        is_dtensor: bool,
        lr: float,
        adjust_lr_fn: Optional[str],
        eps: float,
        is_fused_gate_up: bool,
    ) -> None:
        if grad.ndim < 2:
            raise ValueError(f"ScaledSGD match_muon_rms expects matrix (2D+) params, got shape {tuple(grad.shape)}")
        # Global shape drives all shape-derived scales (DTensor .shape is the
        # global shape; for plain tensors global == local).
        global_shape = tuple(grad.shape)

        # Mesh dims whose Shard splits a MATRIX dim (the last two): the
        # per-matrix squared norms must be summed across them. Batch-dim
        # shards keep whole matrices rank-local — no communication.
        matrix_shard_mesh_dims: list[int] = []
        if is_dtensor:
            for mesh_dim, placement in enumerate(grad.placements):
                if isinstance(placement, Shard):
                    if placement.dim >= grad.ndim - 2:
                        matrix_shard_mesh_dims.append(mesh_dim)
                elif not placement.is_replicate():
                    raise ValueError(
                        f"ScaledSGD match_muon_rms cannot handle placement {placement} "
                        f"(only Shard/Replicate are supported)"
                    )

        fused_split = None
        if is_fused_gate_up:
            if is_dtensor and any(isinstance(pl, Shard) and pl.dim == grad.ndim - 1 for pl in grad.placements):
                raise NotImplementedError(
                    "ScaledSGD match_muon_rms does not support fused gate_up params sharded on "
                    "the fused (last) dim — the local halving split would mix gate and up."
                )
            fused_split = update.shape[-1] // 2

        # Muon parity: the lr adjustment is computed from the FULL (global,
        # pre-split) matrix shape — see Muon._muon_step's _adjust_lr call.
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, global_shape[-2:])

        def _halves(t: torch.Tensor) -> list[torch.Tensor]:
            if fused_split is None:
                return [t]
            return [t[..., :fused_split], t[..., fused_split:]]

        rows_global = global_shape[-2]
        cols_global = global_shape[-1] if fused_split is None else global_shape[-1] // 2

        for update_piece, param_piece in zip(_halves(update), _halves(p_local)):
            # Per-matrix squared Frobenius norm in fp32 (batch dims kept).
            sqnorm = update_piece.to(torch.float32).pow(2).sum(dim=(-2, -1), keepdim=True)
            if matrix_shard_mesh_dims:
                for mesh_dim in matrix_shard_mesh_dims:
                    dist.all_reduce(sqnorm, op=dist.ReduceOp.SUM, group=grad.device_mesh.get_group(mesh_dim))
            norm = sqnorm.sqrt_().clamp_(min=eps)
            # p_m -= adjusted_lr * sqrt(min(A, B)) * u_m / ||u_m||_F
            scale = (adjusted_lr * float(min(rows_global, cols_global)) ** 0.5) / norm
            # Keep the fp32 scale without materializing ``update * scale``.
            # That temporary can be as large as an expert bank and needlessly
            # doubles fold-time peak memory on small trainer shards. addcmul_
            # performs the same promoted multiply-add directly into the fp32
            # master while leaving ``update`` untouched (it may alias the
            # persistent momentum buffer when nesterov=False).
            param_piece.addcmul_(update_piece, scale, value=-1.0)
