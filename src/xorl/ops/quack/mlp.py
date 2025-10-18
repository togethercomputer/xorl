# Copyright (c) 2025, Tri Dao
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .linear import linear_act_func, act_linear_func


def mlp_func(x, weight1, weight2, activation: str, fuse_grad_accum=False, tuned=True):
    preact, postact = linear_act_func(
        x,
        weight1,
        activation,
        store_preact=torch.is_grad_enabled(),
        fuse_grad_accum=fuse_grad_accum,
        tuned=tuned,
    )
    out = act_linear_func(
        preact,
        weight2,
        postact,
        activation=activation,
        fuse_grad_accum=fuse_grad_accum,
        tuned=tuned,
    )
    return out


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=False,
        bias2=False,
        activation="gelu",
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
        tuned: bool = True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else 4 * in_features
        self.activation = activation
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.fuse_grad_accum = fuse_grad_accum
        self.tuned = tuned

    def forward(self, input: Tensor) -> Tensor:
        if (
            self.fc1.bias is None
            and self.fc2.bias is None
            and input.is_cuda
            and input.stride(-1) == 1
            and self.fc1.in_features % 8 == 0
            and self.fc1.out_features % 8 == 0
            and self.fc2.out_features % 8 == 0
        ):
            return mlp_func(
                input,
                self.fc1.weight,
                self.fc2.weight,
                activation=self.activation,
                fuse_grad_accum=self.fuse_grad_accum,
                tuned=self.tuned,
            )
        else:
            y = self.fc1(input)
            return self.fc2(F.silu(y[..., ::2]) * y[..., 1::2])
