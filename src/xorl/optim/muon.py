"""
Muon optimizer: Momentum + Orthogonalization Updates for Neurons.

Extends ``torch.optim.Muon`` with:
  - Mixed param groups: ``use_muon=True`` (Newton-Schulz) / ``False`` (AdamW fallback)
  - FSDP2/EP DTensor support (shard-local Newton-Schulz)
  - 3D+ MoE expert tensor support (preserve leading dims as matrix batches)

The core Muon algorithm is aligned with PyTorch's implementation:
  B_t  = (1-μ) * g_t + μ * B_{t-1}          (EMA momentum; skipped if momentum=0)
  ~B_t = g_t + μ * B_t   if nesterov         (Nesterov look-ahead)
  O_t  = NS(~B_t)                            (Newton-Schulz orthogonalization)
  θ_t  = θ_{t-1} - γλθ_{t-1} - γ' * O_t     (weight decay + update)

When momentum=0, the EMA buffer is skipped entirely and NS is applied directly
to the raw gradient, saving the memory cost of the momentum buffer.

References:
    - Keller Jordan, "Muon: An optimizer for hidden layers in neural networks"
      https://kellerjordan.github.io/posts/muon/
    - Together AI Moonlight: https://arxiv.org/abs/2502.16982
    - PyTorch torch.optim.Muon
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch.distributed._tensor import DTensor
from torch.optim import Muon as TorchMuon
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
from torch.optim.optimizer import Optimizer

from ..utils import logging
from .gram_newton_schulz import GramNewtonSchulzOrthogonalizer, expand_ns_coefficients, find_best_restarts


logger = logging.get_logger(__name__)

GROUPED_GRAM_NS_FP32_BYTE_LIMIT = 2 * 1024**3


@dataclass
class _MuonUpdatePlan:
    param: torch.Tensor
    adjusted_lr: float
    orig_shape: Optional[torch.Size]
    pieces: list[Optional[torch.Tensor]]


@dataclass
class _GroupedOrthogonalizationEntry:
    plan: _MuonUpdatePlan
    piece_index: int
    tensor: torch.Tensor
    batched_tensor: torch.Tensor
    transposed_for_batching: bool


class Muon(TorchMuon):
    """
    Muon optimizer with mixed parameter group support.

    Inherits from ``torch.optim.Muon`` for algorithm alignment, adding:
      - ``use_muon`` flag per param group for Muon vs AdamW dispatch
      - DTensor (FSDP2/EP) shard-local Newton-Schulz
      - 3D+ tensor reshaping for MoE expert weights

    Args:
        params: Iterable of parameter groups (dicts with ``"params"`` key).
        lr: Default learning rate (used for Muon groups).
        momentum: Momentum coefficient for Muon groups.
        nesterov: Whether to use Nesterov momentum for Muon groups.
        ns_steps: Number of Newton-Schulz iterations.
        weight_decay: Decoupled weight decay coefficient.
        adjust_lr_fn: LR adjustment mode. ``"original"`` scales by
            ``sqrt(max(1, A/B))``, ``"match_rms_adamw"`` scales by
            ``0.2 * sqrt(max(A, B))`` so Muon can reuse AdamW hyperparams.
        adamw_betas: Beta coefficients for AdamW groups.
        adamw_eps: Epsilon for AdamW groups.
        grad_dtype: If set, force the gradient tensor used inside Muon to this
            dtype before momentum/update construction.
        momentum_dtype: If set, force Muon momentum buffers to this dtype
            (e.g. ``torch.bfloat16``).  Saves memory when params/grads are
            fp32 but bf16 momentum is sufficient for Newton-Schulz.
            Default ``None`` inherits dtype from the gradient.
        update_dtype: If set, force the transient update tensor passed into
            Newton-Schulz to this dtype. This decouples Muon compute dtype from
            the stored gradient or momentum-buffer dtype.
        force_momentum_path: If true, still route update construction through
            the momentum-buffer path when ``momentum=0``. Intended for
            debugging/ablations that separate path effects from optimizer
            coefficient effects.
        ns_algorithm: Newton-Schulz backend. ``"standard_newton_schulz"``
            preserves the current PyTorch Muon path. ``"gram_newton_schulz"``
            enables the Dao-AILab Gram Newton-Schulz update path.
        ns_use_quack_kernels: Whether Gram Newton-Schulz may use Quack GEMM
            kernels on SM90+/SM100 devices. Falls back to torch matmuls when
            unavailable or unsupported.
        gram_newton_schulz_num_restarts: If using Gram Newton-Schulz and
            explicit restart iterations are not provided, autotune this many
            restart locations from the chosen coefficients.
        gram_newton_schulz_restart_iterations: Explicit Gram Newton-Schulz
            restart iteration indices. A value of ``2`` means restart after
            finishing the second iteration.
        adamw_state_dtype: If set, force AdamW fallback optimizer states
            (``exp_avg``, ``exp_avg_sq``) to this dtype (e.g.
            ``torch.bfloat16``).  Default ``None`` inherits dtype from
            the parameter.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adjust_lr_fn: Optional[str] = None,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        momentum_dtype: Optional[torch.dtype] = None,
        grad_dtype: Optional[torch.dtype] = None,
        update_dtype: Optional[torch.dtype] = None,
        force_momentum_path: bool = False,
        ns_algorithm: str = "standard_newton_schulz",
        ns_use_quack_kernels: bool = True,
        gram_newton_schulz_num_restarts: int = 1,
        gram_newton_schulz_restart_iterations: Optional[Iterable[int]] = None,
        adamw_state_dtype: Optional[torch.dtype] = None,
    ):
        if ns_algorithm not in {"standard_newton_schulz", "gram_newton_schulz"}:
            raise ValueError(
                f"Unsupported Muon ns_algorithm: {ns_algorithm!r}. "
                "Expected 'standard_newton_schulz' or 'gram_newton_schulz'."
            )
        if gram_newton_schulz_num_restarts < 0:
            raise ValueError(
                f"gram_newton_schulz_num_restarts must be non-negative, got {gram_newton_schulz_num_restarts}"
            )

        self._momentum_dtype = momentum_dtype
        self._grad_dtype = grad_dtype
        self._update_dtype = update_dtype
        self._force_momentum_path = force_momentum_path
        self._adamw_state_dtype = adamw_state_dtype
        self._logged_dtypes = False
        self._gram_ns_orthogonalizers = {}
        # Skip TorchMuon.__init__ (which enforces 2D-only) and call
        # Optimizer.__init__ directly so we can accept 3D MoE params and
        # non-Muon (AdamW) param groups.
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=(3.4445, -4.775, 2.0315),
            eps=1e-7,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            ns_algorithm=ns_algorithm,
            ns_use_quack_kernels=ns_use_quack_kernels,
            gram_newton_schulz_num_restarts=gram_newton_schulz_num_restarts,
            gram_newton_schulz_restart_iterations=(
                tuple(gram_newton_schulz_restart_iterations)
                if gram_newton_schulz_restart_iterations is not None
                else None
            ),
            use_muon=True,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        Optimizer.__init__(self, params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict) -> None:
        """Newton-Schulz orthogonalization + momentum update.

        Algorithm aligned with ``torch.optim.Muon``:
          1. EMA momentum:  buf.lerp_(grad, 1 - momentum)
          2. Nesterov:       update = grad.lerp(buf, momentum)
          3. Newton-Schulz:  update = NS(update)
          4. LR adjustment:  adjusted_lr = adjust_lr(lr, shape)
          5. Weight decay:   param *= 1 - lr * wd
          6. Update:         param -= adjusted_lr * update

        For FSDP2/EP DTensors, operates on the local shard directly
        (shard-local Newton-Schulz) to avoid DTensor reshape/matmul issues.
        """
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_coefficients = group["ns_coefficients"]
        ns_steps = group["ns_steps"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        adjust_lr_fn = group["adjust_lr_fn"]
        uses_grouped_gram_ns = group["ns_algorithm"] == "gram_newton_schulz"
        grouped_updates: dict[tuple[tuple[int, int], torch.dtype, torch.device], list[_GroupedOrthogonalizationEntry]]
        grouped_updates = defaultdict(list)
        update_plans: list[_MuonUpdatePlan] = []

        for p in group["params"]:
            if p.grad is None:
                continue

            # Extract local tensors from DTensors (FSDP2/EP sharded params).
            grad = p.grad
            is_dtensor = isinstance(grad, DTensor)
            if is_dtensor:
                grad_local = grad._local_tensor
                p_local = p._local_tensor
            else:
                grad_local = grad
                p_local = p.data
            raw_grad_dtype = grad_local.dtype
            if self._grad_dtype is not None and grad_local.dtype != self._grad_dtype:
                grad_local = grad_local.to(self._grad_dtype)

            # Handle 3D+ tensors as batches of matrices: [..., hidden, intermediate].
            # For fused gate_up_proj [E, H, 2I], split into two [..., H, I] halves.
            orig_shape = None
            fused_split = None
            if grad_local.ndim >= 3:
                orig_shape = grad_local.shape
                fused_gate_up_ids = group.get("_fused_gate_up_ids", set())
                if id(p) in fused_gate_up_ids:
                    fused_split = grad_local.shape[-1] // 2
                grad_local = grad_local.reshape(-1, *grad_local.shape[-2:])

            # --- PyTorch Muon algorithm (aligned with torch.optim._muon) ---
            if momentum == 0 and not self._force_momentum_path:
                # No momentum: apply NS directly to the raw gradient, no buffer needed.
                update = grad_local
                if self._update_dtype is not None and update.dtype != self._update_dtype:
                    update = update.to(self._update_dtype)
                if not self._logged_dtypes:
                    logger.info_rank0(
                        f"Muon dtypes (no momentum): param={p_local.dtype}, raw_grad={raw_grad_dtype}, "
                        f"grad={grad_local.dtype}, update={update.dtype} "
                        f"(shape={list(grad_local.shape)})"
                    )
                    self._logged_dtypes = True
            else:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf_dtype = self._momentum_dtype or grad_local.dtype
                    state["momentum_buffer"] = torch.zeros_like(
                        grad_local,
                        dtype=buf_dtype,
                    )
                buf = state["momentum_buffer"]

                # EMA momentum: B = (1-μ)*g + μ*B
                # Cast grad to buf dtype for the in-place lerp
                buf.lerp_(grad_local.to(buf.dtype), 1 - momentum)

                # Nesterov: ~B = g + μ*B  (or just B if nesterov=False)
                update = grad_local.to(buf.dtype).lerp(buf, momentum) if nesterov else buf
                if self._update_dtype is not None and update.dtype != self._update_dtype:
                    update = update.to(self._update_dtype)
                if not self._logged_dtypes:
                    logger.info_rank0(
                        f"Muon dtypes: param={p_local.dtype}, raw_grad={raw_grad_dtype}, grad={grad_local.dtype}, "
                        f"momentum={buf.dtype}, update={update.dtype}, "
                        f"force_momentum_path={self._force_momentum_path} "
                        f"(shape={list(grad_local.shape)})"
                    )
                    self._logged_dtypes = True

            adjusted_lr = _adjust_lr(
                lr,
                adjust_lr_fn,
                grad_local.shape[-2:] if grad_local.ndim > 2 else grad_local.shape,
            )
            pieces = [update[..., :fused_split], update[..., fused_split:]] if fused_split is not None else [update]
            plan = _MuonUpdatePlan(
                param=p_local,
                adjusted_lr=adjusted_lr,
                orig_shape=orig_shape,
                pieces=[None] * len(pieces),
            )
            update_plans.append(plan)

            for piece_index, piece in enumerate(pieces):
                if uses_grouped_gram_ns:
                    batched_piece = piece.reshape(-1, *piece.shape[-2:]) if piece.ndim > 2 else piece.unsqueeze(0)
                    transposed_for_batching = batched_piece.shape[-2] > batched_piece.shape[-1]
                    if transposed_for_batching:
                        batched_piece = batched_piece.mT
                    batch_key = (tuple(batched_piece.shape[-2:]), piece.dtype, piece.device)
                    grouped_updates[batch_key].append(
                        _GroupedOrthogonalizationEntry(
                            plan=plan,
                            piece_index=piece_index,
                            tensor=piece,
                            batched_tensor=batched_piece,
                            transposed_for_batching=transposed_for_batching,
                        )
                    )
                else:
                    plan.pieces[piece_index] = self._orthogonalize_update(piece, group, ns_coefficients, ns_steps, eps)

        if uses_grouped_gram_ns and grouped_updates:
            orthogonalizer = self._get_gram_ns_orthogonalizer(group)
            self._orthogonalize_grouped_gram_ns_updates(grouped_updates, orthogonalizer)

        for plan in update_plans:
            update_pieces = plan.pieces
            if any(piece is None for piece in update_pieces):
                raise RuntimeError("Grouped Muon orthogonalization left an update piece uninitialized")
            if len(update_pieces) == 1:
                update = update_pieces[0]
            else:
                update = torch.cat(update_pieces, dim=-1)

            # Restore original shape
            if plan.orig_shape is not None:
                update = update.reshape(plan.orig_shape)

            # Cast back to param dtype
            update = update.to(plan.param.dtype)

            # Decoupled weight decay
            plan.param.mul_(1 - lr * weight_decay)

            # Parameter update
            plan.param.add_(update, alpha=-plan.adjusted_lr)

    def _orthogonalize_update(
        self,
        update: torch.Tensor,
        group: dict,
        ns_coefficients: Tuple[float, float, float],
        ns_steps: int,
        eps: float,
    ) -> torch.Tensor:
        if group["ns_algorithm"] == "standard_newton_schulz":
            if update.ndim <= 2:
                return _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

            original_shape = update.shape
            flat_update = update.reshape(-1, *update.shape[-2:])
            orthogonalized = [
                _zeropower_via_newtonschulz(matrix, ns_coefficients, ns_steps, eps) for matrix in flat_update.unbind(0)
            ]
            return torch.stack(orthogonalized, dim=0).reshape(original_shape)
        if group["ns_algorithm"] == "gram_newton_schulz":
            return self._get_gram_ns_orthogonalizer(group).orthogonalize(update)
        raise ValueError(
            f"Unsupported Muon ns_algorithm: {group['ns_algorithm']!r}. "
            "Expected 'standard_newton_schulz' or 'gram_newton_schulz'."
        )

    def _orthogonalize_grouped_gram_ns_updates(
        self,
        grouped_updates: dict[tuple[tuple[int, int], torch.dtype, torch.device], list[_GroupedOrthogonalizationEntry]],
        orthogonalizer: GramNewtonSchulzOrthogonalizer,
    ) -> None:
        for batch_entries in grouped_updates.values():
            for chunk in self._iter_grouped_gram_ns_chunks(batch_entries):
                if len(chunk) == 1 and chunk[0].batched_tensor.shape[0] == 1 and chunk[0].tensor.ndim == 2:
                    entry = chunk[0]
                    entry.plan.pieces[entry.piece_index] = orthogonalizer.orthogonalize(entry.tensor)
                    continue

                stacked = torch.cat([entry.batched_tensor for entry in chunk], dim=0)
                outputs = orthogonalizer.orthogonalize(stacked)
                output_offset = 0
                for entry in chunk:
                    batch_size = entry.batched_tensor.shape[0]
                    output = outputs[output_offset : output_offset + batch_size]
                    output_offset += batch_size
                    if entry.transposed_for_batching:
                        output = output.mT
                    if entry.tensor.ndim == 2:
                        entry.plan.pieces[entry.piece_index] = output.squeeze(0)
                    else:
                        entry.plan.pieces[entry.piece_index] = output.reshape(entry.tensor.shape)

    def _iter_grouped_gram_ns_chunks(
        self,
        batch_entries: list[_GroupedOrthogonalizationEntry],
    ):
        per_matrix_numel = batch_entries[0].batched_tensor.shape[-2] * batch_entries[0].batched_tensor.shape[-1]
        max_matrix_batch = max(1, GROUPED_GRAM_NS_FP32_BYTE_LIMIT // (per_matrix_numel * 4))
        chunk: list[_GroupedOrthogonalizationEntry] = []
        chunk_matrix_batch = 0

        for entry in batch_entries:
            entry_matrix_batch = entry.batched_tensor.shape[0]
            if chunk and chunk_matrix_batch + entry_matrix_batch > max_matrix_batch:
                yield chunk
                chunk = []
                chunk_matrix_batch = 0

            chunk.append(entry)
            chunk_matrix_batch += entry_matrix_batch

        if chunk:
            yield chunk

    def _get_gram_ns_orthogonalizer(self, group: dict) -> GramNewtonSchulzOrthogonalizer:
        expanded_coefficients = tuple(expand_ns_coefficients(group["ns_coefficients"], group["ns_steps"]))
        restart_iterations = group["gram_newton_schulz_restart_iterations"]
        if restart_iterations is not None:
            restart_iterations = tuple(int(i) for i in restart_iterations)
        else:
            restart_iterations = tuple(
                find_best_restarts(
                    expanded_coefficients,
                    num_restarts=group["gram_newton_schulz_num_restarts"],
                )
            )

        cache_key = (
            expanded_coefficients,
            float(group["eps"]),
            bool(group["ns_use_quack_kernels"]),
            restart_iterations,
        )
        orthogonalizer = self._gram_ns_orthogonalizers.get(cache_key)
        if orthogonalizer is None:
            orthogonalizer = GramNewtonSchulzOrthogonalizer(
                ns_coefficients=expanded_coefficients,
                ns_epsilon=group["eps"],
                ns_use_quack_kernels=group["ns_use_quack_kernels"],
                gram_newton_schulz_restart_iterations=restart_iterations,
            )
            self._gram_ns_orthogonalizers[cache_key] = orthogonalizer
            logger.info_rank0(
                "Initialized Muon Gram Newton-Schulz "
                f"(restarts={list(restart_iterations)}, quack_kernels={group['ns_use_quack_kernels']})"
            )
        return orthogonalizer

    def _adamw_step(self, group: dict) -> None:
        """Standard AdamW update for non-Muon parameters."""
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Muon (AdamW fallback) does not support sparse gradients.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=self._adamw_state_dtype or p.dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=self._adamw_state_dtype or p.dtype)

            state["step"] += 1
            step = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Decoupled weight decay
            if weight_decay > 0:
                p.data.mul_(1.0 - lr * weight_decay)

            # Update biased first and second moment estimates
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            # Bias correction
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr / bias_correction1

            denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)
