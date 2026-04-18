import torch
from torch.optim.optimizer import Optimizer


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class SignSGD(Optimizer):
    """Sign-based SGD optimizer with no optimizer-state tensors."""

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
        """Perform a single optimization step."""
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
                    raise RuntimeError("SignSGD does not support sparse gradients.")

                if weight_decay:
                    p.add_(p, alpha=-lr * weight_decay)

                p.add_(torch.sign(grad), alpha=-lr)

        return loss
