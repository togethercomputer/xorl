"""Cautious Weight Decay (CWD) primitives.

CWD applies decoupled weight decay only along coordinates where the optimizer
update and the parameter share a sign::

    x_{t+1} = x_t - eta * (u_t + lambda * I(u_t * x_t >= 0) * x_t)

Compared to standard decoupled decay (``x_{t+1} = (1 - eta*lambda) * x_t - eta * u_t``),
the decay term is masked elementwise by ``I(u_t * x_t >= 0)``. The mask uses
the sign of the *optimizer update* ``u_t`` (after preconditioning /
orthogonalization), not the raw gradient.

Reference: Chen et al., "Cautious Weight Decay" (arXiv:2510.12402).
"""

import torch


def apply_cautious_decay_(
    param: torch.Tensor,
    update_sign_proxy: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    cautious: bool,
) -> None:
    """In-place decoupled weight decay, optionally masked by ``I(u * x >= 0)``.

    When ``cautious=False`` this is the standard ``param *= 1 - lr * weight_decay``.
    When ``cautious=True`` the decay factor becomes
    ``1 - lr * weight_decay * I(update_sign_proxy * param >= 0)`` elementwise.

    ``update_sign_proxy`` only needs to share its sign with the optimizer
    update direction ``u_t``. For Adam-family optimizers ``exp_avg`` is a valid
    proxy (the preconditioner denominator is strictly positive). For SignSGD
    ``grad`` is a valid proxy. For Muon the post-Newton-Schulz ``update``
    tensor is the proxy (it *is* ``u_t``).

    No-op when ``weight_decay == 0``.
    """
    if weight_decay == 0.0:
        return
    if not cautious:
        param.mul_(1.0 - lr * weight_decay)
        return
    mask = (update_sign_proxy * param >= 0).to(param.dtype)
    param.mul_(mask.mul_(-lr * weight_decay).add_(1.0))
