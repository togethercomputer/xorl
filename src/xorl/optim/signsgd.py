import torch
from torch.optim.optimizer import Optimizer

from .cautious import apply_cautious_decay_


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
class SignSGD(Optimizer):
    """Sign-based SGD optimizer with no optimizer-state tensors.

    When ``cautious=True``, decoupled weight decay is masked by
    ``I(sign(grad) * param >= 0)`` per Chen et al. "Cautious Weight Decay"
    (arXiv:2510.12402). Equivalently, the mask is ``I(grad * param >= 0)``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        cautious: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "cautious": cautious,
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
            cautious = group.get("cautious", False)

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients.")

                apply_cautious_decay_(
                    p,
                    update_sign_proxy=grad,
                    lr=lr,
                    weight_decay=weight_decay,
                    cautious=cautious,
                )

                p.add_(torch.sign(grad), alpha=-lr)

        return loss
