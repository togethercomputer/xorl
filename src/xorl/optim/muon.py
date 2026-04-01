"""
Muon optimizer: Momentum + Orthogonalization Updates for Neurons.

Extends ``torch.optim.Muon`` with:
  - Mixed param groups: ``use_muon=True`` (Newton-Schulz) / ``False`` (AdamW fallback)
  - FSDP2/EP DTensor support (shard-local Newton-Schulz)
  - 3D+ MoE expert tensor support (reshape to 2D for NS)

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

from typing import Iterable, Optional, Tuple

import torch
from torch.distributed._tensor import DTensor
from torch.optim import Muon as TorchMuon
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
from torch.optim.optimizer import Optimizer

from ..utils import logging


logger = logging.get_logger(__name__)


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
        momentum_dtype: If set, force Muon momentum buffers to this dtype
            (e.g. ``torch.bfloat16``).  Saves memory when params/grads are
            fp32 but bf16 momentum is sufficient for Newton-Schulz.
            Default ``None`` inherits dtype from the gradient.
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
        adamw_state_dtype: Optional[torch.dtype] = None,
    ):
        self._momentum_dtype = momentum_dtype
        self._adamw_state_dtype = adamw_state_dtype
        self._logged_dtypes = False
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

            # Handle 3D+ MoE expert tensors: [num_experts, hidden, intermediate]
            # Reshape to 2D for Newton-Schulz, then restore.
            orig_shape = None
            if grad_local.ndim >= 3:
                orig_shape = grad_local.shape
                grad_local = grad_local.reshape(-1, grad_local.shape[-1])

            # --- PyTorch Muon algorithm (aligned with torch.optim._muon) ---
            if momentum == 0:
                # No momentum: apply NS directly to the raw gradient, no buffer needed.
                update = grad_local
                if not self._logged_dtypes:
                    logger.info_rank0(
                        f"Muon dtypes (no momentum): param={p_local.dtype}, grad={grad_local.dtype} "
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
                    if not self._logged_dtypes:
                        logger.info_rank0(
                            f"Muon dtypes: param={p_local.dtype}, grad={grad_local.dtype}, "
                            f"momentum={buf_dtype} (shape={list(grad_local.shape)})"
                        )
                        self._logged_dtypes = True
                buf = state["momentum_buffer"]

                # EMA momentum: B = (1-μ)*g + μ*B
                # Cast grad to buf dtype for the in-place lerp
                buf.lerp_(grad_local.to(buf.dtype), 1 - momentum)

                # Nesterov: ~B = g + μ*B  (or just B if nesterov=False)
                # Work in buf dtype (may be bf16 even if grad is fp32)
                update = grad_local.to(buf.dtype).lerp(buf, momentum) if nesterov else buf

            # Newton-Schulz orthogonalization (uses PyTorch's implementation)
            update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

            # LR adjustment based on the 2D shape fed to NS
            adjusted_lr = _adjust_lr(lr, adjust_lr_fn, grad_local.shape)

            # Restore original shape
            if orig_shape is not None:
                update = update.reshape(orig_shape)

            # Cast back to param dtype
            update = update.to(p_local.dtype)

            # Decoupled weight decay
            p_local.mul_(1 - lr * weight_decay)

            # Parameter update
            p_local.add_(update, alpha=-adjusted_lr)

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
