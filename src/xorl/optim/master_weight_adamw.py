"""AdamW with an fp32 master copy held inside the optimizer (Megatron / slime "main_param").

This is the optimizer half of the "bf16 model params + fp32 master weights" recipe:

  * The model parameters live in bf16, so the forward/backward run pure bf16 with no
    per-forward cast (this is where the throughput comes from versus an
    ``torch.autocast`` path that casts fp32 master params -> bf16 every forward).
  * The optimizer keeps, per parameter, a TRUE fp32 master copy plus fp32
    ``exp_avg`` / ``exp_avg_sq``. Every step it runs the AdamW math in fp32 against the
    fp32 master, then writes the result back into the bf16 model param via
    ``param.copy_(master)`` (rounded to bf16). The grad (bf16) is upcast to fp32 for
    the moment updates so the accumulation is numerically faithful.

This differs from ``torch.optim.AdamW(fused=True)`` over a bf16 param, which updates the
param in its own bf16 dtype and stores bf16 states (i.e. NO fp32 master) — that loses
the small-update information AdamW needs and is numerically worse.

The update reuses torch's fused AdamW CUDA kernel (``torch._fused_adamw_``) pointed at
the fp32 master, so the optimizer step costs about the same as a normal fused AdamW
(one kernel with internal bias correction) — the only extra work is the final
``torch._foreach_copy_`` that rounds the updated fp32 master back into the bf16 model
params. A multi-tensor ``torch._foreach_*`` path is used as a CPU fallback.
"""

import copy
from typing import Any, Dict, List, Tuple

import torch
from torch.optim.optimizer import Optimizer


class MasterWeightAdamW(Optimizer):
    """AdamW that holds an fp32 master copy of each (typically bf16) parameter.

    Mathematically equivalent to fp32 AdamW on the master weights; the model
    parameter is a bf16 view that is refreshed from the master after each step.

    Args:
        params: iterable of parameters or param-group dicts.
        lr: learning rate.
        betas: AdamW (beta1, beta2).
        eps: AdamW epsilon (added to the fp32 denominator).
        weight_decay: decoupled weight decay (applied to the fp32 master).
        master_dtype: dtype of the master copy and moment buffers (default fp32).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        master_dtype: torch.dtype = torch.float32,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "master_dtype": master_dtype,
        }
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict) -> None:
        """Load state, restoring the fp32 master / moments dtype.

        ``torch.optim.Optimizer.load_state_dict`` casts every floating-point state
        tensor to the owning parameter's dtype (bf16 here) via a hardcoded
        ``Optimizer._process_value_according_to_param_policy`` call that a subclass
        cannot intercept. Left unfixed this would store a bf16 master / moments on
        resume, defeating the whole point. We deep-copy the saved fp32 tensors
        ourselves (device-only move, no downcast) before delegating to the base loader,
        so the master is restored at full fp32 precision rather than bf16-rounded.
        """
        state_dict = copy.deepcopy(state_dict)
        # The base loader maps saved-param-index -> current param. Reproduce that map so
        # we can move each saved fp32 tensor onto the correct param's device with no cast.
        saved_groups = state_dict["param_groups"]
        current_params = [p for g in self.param_groups for p in g["params"]]
        saved_param_ids = [pid for g in saved_groups for pid in g["params"]]
        id_to_param = dict(zip(saved_param_ids, current_params))

        preserved: Dict[int, Dict[str, torch.Tensor]] = {}
        for pid, pstate in state_dict["state"].items():
            param = id_to_param.get(pid)
            if param is None:
                continue
            kept: Dict[str, torch.Tensor] = {}
            for key in ("master", "exp_avg", "exp_avg_sq"):
                value = pstate.get(key)
                if isinstance(value, torch.Tensor):
                    # Move device only — preserve the saved (fp32) dtype.
                    kept[key] = value.to(device=param.device)
            if kept:
                preserved[pid] = kept

        super().load_state_dict(state_dict)

        # Overwrite the bf16-downcast tensors the base loader produced with the fp32 originals.
        param_to_id = {id_to_param[pid]: pid for pid in id_to_param}
        for param, pid in param_to_id.items():
            kept = preserved.get(pid)
            if not kept:
                continue
            self.state[param].update(kept)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr: float = group["lr"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            master_dtype: torch.dtype = group["master_dtype"]

            params: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            masters: List[torch.Tensor] = []
            exp_avgs: List[torch.Tensor] = []
            exp_avg_sqs: List[torch.Tensor] = []
            steps: List[torch.Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MasterWeightAdamW does not support sparse gradients.")

                state: Dict[str, Any] = self.state[p]
                if len(state) == 0:
                    # fp32 master copy of the (bf16) parameter, plus fp32 moments.
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["master"] = p.detach().to(master_dtype).clone()
                    state["exp_avg"] = torch.zeros_like(state["master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master"])

                params.append(p)
                # Upcast the (bf16) grad to the master dtype so moment accumulation is faithful.
                grads.append(p.grad.to(master_dtype) if p.grad.dtype != master_dtype else p.grad)
                masters.append(state["master"])
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(state["step"])

            if not params:
                continue

            self._adamw_update(
                params=params,
                grads=grads,
                masters=masters,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                steps=steps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                eps=eps,
                weight_decay=weight_decay,
            )

        return loss

    @classmethod
    def _adamw_update(
        cls,
        *,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        masters: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        steps: List[torch.Tensor],
        beta1: float,
        beta2: float,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        """Run decoupled AdamW on the fp32 master, then refresh the bf16 model params.

        Prefers the fused CUDA kernel (``torch._fused_adamw_``) pointed at the fp32
        master — the same kernel ``torch.optim.AdamW(fused=True)`` uses, so the
        optimizer step is as cheap as a normal fused AdamW (one kernel, internal bias
        correction) rather than the ~3x slower multi-kernel foreach path. The only
        extra work versus a plain fused AdamW is the final ``_foreach_copy_`` that
        rounds the updated fp32 master back into the bf16 model params.
        """
        # step += 1 on-device (no readback). The fused/foreach paths both read the
        # incremented step for bias correction, matching torch AdamW semantics.
        torch._foreach_add_(steps, 1.0)

        if params[0].is_cuda:
            torch._fused_adamw_(
                masters,
                grads,
                exp_avgs,
                exp_avg_sqs,
                [],  # max_exp_avg_sqs (amsgrad disabled)
                steps,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                eps=eps,
                amsgrad=False,
                maximize=False,
                grad_scale=None,
                found_inf=None,
            )
        else:
            cls._foreach_update_inplace(
                masters=masters,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                steps=steps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                eps=eps,
                weight_decay=weight_decay,
            )

        # Refresh the (bf16) model params from the updated fp32 master (rounds to bf16).
        torch._foreach_copy_(params, masters)

    @staticmethod
    def _foreach_update_inplace(
        *,
        masters: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        steps: List[torch.Tensor],
        beta1: float,
        beta2: float,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        """Multi-tensor AdamW on the master (CPU fallback when the fused kernel is unavailable).

        ``steps`` is already incremented by the caller. All params in a group share the
        same step count, so bias correction is a scalar (read one step tensor once).
        """
        step_count = float(steps[0].item())

        # Decoupled weight decay on the fp32 master: master *= (1 - lr * wd)
        if weight_decay != 0.0:
            torch._foreach_mul_(masters, 1.0 - lr * weight_decay)

        # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

        # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step_count
        bias_correction2_sqrt = (1.0 - beta2**step_count) ** 0.5
        step_size = lr / bias_correction1

        # denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, bias_correction2_sqrt)
        torch._foreach_add_(denom, eps)

        # master -= step_size * exp_avg / denom
        torch._foreach_addcdiv_(masters, exp_avgs, denom, value=-step_size)
