from dataclasses import dataclass
from typing import Any

import torch
from torch.optim.optimizer import Optimizer


try:
    from torch.distributed._tensor import DTensor
except ImportError:  # pragma: no cover - torch versions used by xorl provide DTensor.
    DTensor = None

from .cautious import apply_cautious_decay_


@dataclass
class _OffloadedDTensorState:
    local_tensor: torch.Tensor
    device_mesh: Any
    placements: tuple[Any, ...]


class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=False,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
        cautious=False,
        denominator_chunk_size=0,
        reuse_grad_for_momentum=False,
        state_offload_device=None,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
            "cautious": cautious,
            "denominator_chunk_size": denominator_chunk_size,
            "reuse_grad_for_momentum": reuse_grad_for_momentum,
            "state_offload_device": state_offload_device,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _sharded_dims(tensor: torch.Tensor) -> set[int]:
        sharded_dims = set()
        for placement in getattr(tensor, "placements", ()):
            dim = getattr(placement, "dim", None)
            if callable(dim):
                dim = dim()
            if dim is not None:
                sharded_dims.add(int(dim))
        return sharded_dims

    @classmethod
    def _chunk_indices(cls, tensor: torch.Tensor, chunk_size: int):
        if tensor.dim() == 0 or tensor.numel() <= chunk_size:
            yield (slice(None),) * tensor.dim()
            return

        sharded_dims = cls._sharded_dims(tensor)
        chunk_dim = next(
            (dim for dim in reversed(range(tensor.dim())) if dim not in sharded_dims and tensor.size(dim) > 1),
            0,
        )
        elements_per_index = max(1, tensor.numel() // max(1, tensor.size(chunk_dim)))
        chunk_width = max(1, chunk_size // elements_per_index)

        for start in range(0, tensor.size(chunk_dim), chunk_width):
            index = [slice(None)] * tensor.dim()
            index[chunk_dim] = slice(start, min(start + chunk_width, tensor.size(chunk_dim)))
            yield tuple(index)

    @staticmethod
    def _offload_dtensor_state(value: torch.Tensor) -> _OffloadedDTensorState:
        return _OffloadedDTensorState(
            local_tensor=value.to_local().detach().to("cpu", non_blocking=True),
            device_mesh=value.device_mesh,
            placements=tuple(value.placements),
        )

    @staticmethod
    def _restore_dtensor_state(value: _OffloadedDTensorState, device: torch.device) -> torch.Tensor:
        if DTensor is None:  # pragma: no cover - defensive for unsupported torch versions.
            raise RuntimeError("Cannot restore offloaded DTensor optimizer state: DTensor is unavailable.")

        local_tensor = value.local_tensor.to(device, non_blocking=True)
        return DTensor.from_local(
            local_tensor,
            value.device_mesh,
            value.placements,
            run_check=False,
        )

    @staticmethod
    def _move_state_to_device(state: dict, device, *, offload: bool = False) -> None:
        device = torch.device(device)
        for key in ("exp_avg", "exp_avg_sq", "compensation"):
            value = state.get(key)
            if isinstance(value, _OffloadedDTensorState):
                if offload and device.type == "cpu":
                    continue
                state[key] = AnyPrecisionAdamW._restore_dtensor_state(value, device)
                continue

            if DTensor is not None and isinstance(value, DTensor):
                if offload and device.type == "cpu":
                    state[key] = AnyPrecisionAdamW._offload_dtensor_state(value)
                elif value.device != device:
                    state[key] = value.to(device, non_blocking=True)
                continue

            if isinstance(value, torch.Tensor) and value.device != device:
                state[key] = value.to(device, non_blocking=True)

    @staticmethod
    def _adamw_update_chunked_(
        param: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        *,
        denom_correction,
        eps: float,
        step_size,
        chunk_size: int,
    ) -> None:
        for index in AnyPrecisionAdamW._chunk_indices(exp_avg_sq, chunk_size):
            denom = exp_avg_sq[index].sqrt()
            denom.div_(denom_correction).add_(eps, alpha=1)
            param[index].addcdiv_(exp_avg[index], denom, value=-step_size)

    @staticmethod
    def _adamw_kahan_update_chunked_(
        param: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        compensation: torch.Tensor,
        *,
        denom_correction,
        eps: float,
        step_size,
        chunk_size: int,
    ) -> None:
        for index in AnyPrecisionAdamW._chunk_indices(exp_avg_sq, chunk_size):
            denom = exp_avg_sq[index].sqrt()
            denom.div_(denom_correction).add_(eps, alpha=1)
            compensation_chunk = compensation[index]
            param_chunk = param[index]
            compensation_chunk.addcdiv_(exp_avg[index], denom, value=-step_size)

            temp_buffer = param_chunk.detach().clone()
            param_chunk.add_(compensation_chunk)
            compensation_chunk.add_(temp_buffer.sub_(param_chunk))

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        When ``cautious=True``, decoupled weight decay is masked by
        ``I(exp_avg * param >= 0)`` per Chen et al. "Cautious Weight Decay"
        (arXiv:2510.12402). ``sign(exp_avg)`` matches ``sign(u_t)`` since the
        Adam preconditioner denominator is strictly positive.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]
            cautious = group.get("cautious", False)
            denominator_chunk_size = int(group.get("denominator_chunk_size", 0) or 0)
            reuse_grad_for_momentum = bool(group.get("reuse_grad_for_momentum", False))
            state_offload_device = group.get("state_offload_device")

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            offloaded_state = False
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                grad = p.grad
                can_reuse_grad_for_momentum = reuse_grad_for_momentum and grad.dtype == momentum_dtype
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    if not can_reuse_grad_for_momentum:
                        state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)
                else:
                    self._move_state_to_device(state, p.device)

                state["step"] += 1
                step = state["step"]

                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                clear_grad_after_step = False
                if can_reuse_grad_for_momentum:
                    old_exp_avg = state.get("exp_avg")
                    exp_avg = grad.detach()
                    exp_avg.mul_(1 - beta1)
                    if old_exp_avg is not None:
                        exp_avg.add_(old_exp_avg, alpha=beta1)
                    state["exp_avg"] = exp_avg
                    del old_exp_avg
                    clear_grad_after_step = True
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Cautious weight decay must use the post-update first moment
                # (its sign matches the optimizer update direction). Apply
                # decay AFTER moments are updated and BEFORE the parameter
                # update, against the pre-update parameter values.
                apply_cautious_decay_(
                    p.data,
                    update_sign_proxy=exp_avg,
                    lr=lr,
                    weight_decay=weight_decay,
                    cautious=cautious,
                )

                bias_correction1 = 1 - beta1**step
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5
                use_chunked_update = denominator_chunk_size > 0 and exp_avg_sq.numel() > denominator_chunk_size

                if use_kahan_summation:
                    compensation = state["compensation"]
                    if use_chunked_update:
                        self._adamw_kahan_update_chunked_(
                            p.data,
                            exp_avg,
                            exp_avg_sq,
                            compensation,
                            denom_correction=denom_correction,
                            eps=eps,
                            step_size=step_size,
                            chunk_size=denominator_chunk_size,
                        )
                        if clear_grad_after_step:
                            p.grad = None
                        if state_offload_device is not None:
                            self._move_state_to_device(state, state_offload_device, offload=True)
                            offloaded_state = True
                        continue

                    centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:
                    if use_chunked_update:
                        self._adamw_update_chunked_(
                            p.data,
                            exp_avg,
                            exp_avg_sq,
                            denom_correction=denom_correction,
                            eps=eps,
                            step_size=step_size,
                            chunk_size=denominator_chunk_size,
                        )
                        if clear_grad_after_step:
                            p.grad = None
                        if state_offload_device is not None:
                            self._move_state_to_device(state, state_offload_device, offload=True)
                            offloaded_state = True
                        continue

                    centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)

                if clear_grad_after_step:
                    p.grad = None
                if state_offload_device is not None:
                    self._move_state_to_device(state, state_offload_device, offload=True)
                    offloaded_state = True

            if offloaded_state and torch.cuda.is_available():
                torch.cuda.empty_cache()
