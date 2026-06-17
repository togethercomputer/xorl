"""Tests for MasterWeightAdamW (bf16 model params + fp32 master held in the optimizer).

The defining property: the model parameter stays bf16 (forward reads bf16 directly),
while the optimizer keeps a TRUE fp32 master copy + fp32 moments and refreshes the
bf16 param from the master each step. This is the Megatron / slime "main_param"
recipe and is mathematically equivalent to fp32 AdamW on the master weights.
"""

import pytest
import torch

from xorl.optim import MasterWeightAdamW, build_optimizer
from xorl.optim.master_weight_adamw import MasterWeightAdamW as DirectMasterWeightAdamW


pytestmark = [pytest.mark.cpu]


def test_keeps_fp32_master_and_states_for_bf16_param():
    p = torch.nn.Parameter(torch.randn(16, 32, dtype=torch.bfloat16))
    opt = MasterWeightAdamW([p], lr=1e-3, weight_decay=0.01)
    p.grad = torch.randn_like(p)
    opt.step()

    state = opt.state[p]
    assert p.dtype == torch.bfloat16, "model param must stay bf16 (forward reads bf16)"
    assert state["master"].dtype == torch.float32, "master copy must be fp32"
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    # The bf16 param is the rounded view of the fp32 master.
    assert torch.equal(p, state["master"].to(torch.bfloat16))


def test_matches_fp32_adamw_within_bf16_grad_rounding():
    torch.manual_seed(0)
    bf16_params = [torch.nn.Parameter(torch.randn(64, 128, dtype=torch.bfloat16)) for _ in range(3)]
    fp32_params = [torch.nn.Parameter(p.detach().float().clone()) for p in bf16_params]

    opt = MasterWeightAdamW(bf16_params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    ref = torch.optim.AdamW(fp32_params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    for _ in range(25):
        grads = [torch.randn(64, 128) * 0.1 for _ in fp32_params]
        for p, g in zip(bf16_params, grads):
            p.grad = g.to(torch.bfloat16)
        for p, g in zip(fp32_params, grads):
            p.grad = g
        opt.step()
        ref.step()
        opt.zero_grad()
        ref.zero_grad()

    # The fp32 master should track fp32 AdamW closely; the only divergence is the bf16
    # rounding of the grad fed into the moment updates.
    for p, r in zip(bf16_params, fp32_params):
        master = opt.state[p]["master"]
        assert (master - r).abs().max().item() < 1e-3


def test_bf16_master_loses_precision_versus_fp32_master():
    """A true fp32 master must accumulate small updates a bf16 param/state would drop.

    Run identical grads through (a) the fp32-master optimizer and (b) a bf16-state
    variant; the fp32 master should diverge from the bf16-only update, proving the
    master is doing real fp32 accumulation rather than bf16 rounding each step.
    """
    torch.manual_seed(1)
    base = torch.randn(32, 64, dtype=torch.bfloat16)

    p_fp32m = torch.nn.Parameter(base.clone())
    opt_fp32m = MasterWeightAdamW([p_fp32m], lr=1e-3, weight_decay=0.0, master_dtype=torch.float32)

    p_bf16m = torch.nn.Parameter(base.clone())
    opt_bf16m = MasterWeightAdamW([p_bf16m], lr=1e-3, weight_decay=0.0, master_dtype=torch.bfloat16)

    for _ in range(50):
        g = (torch.randn(32, 64) * 1e-3).to(torch.bfloat16)  # tiny updates: bf16 master rounds them away
        p_fp32m.grad = g.clone()
        p_bf16m.grad = g.clone()
        opt_fp32m.step()
        opt_bf16m.step()

    assert opt_fp32m.state[p_fp32m]["master"].dtype == torch.float32
    assert opt_bf16m.state[p_bf16m]["master"].dtype == torch.bfloat16
    # The two diverge: the fp32 master accumulates tiny updates the bf16 master drops.
    assert (p_fp32m.float() - p_bf16m.float()).abs().max().item() > 0


def test_state_dict_round_trip_preserves_fp32_master():
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    opt = MasterWeightAdamW([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()

    opt2 = MasterWeightAdamW([p], lr=1e-3)
    opt2.load_state_dict(sd)
    # The base Optimizer.load_state_dict would downcast to the param's bf16 dtype;
    # our override must re-promote master/moments to fp32.
    assert opt2.state[p]["master"].dtype == torch.float32
    assert opt2.state[p]["exp_avg"].dtype == torch.float32
    assert opt2.state[p]["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(opt2.state[p]["master"], opt.state[p]["master"])


def test_build_optimizer_master_adamw_wires_through():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)).to(torch.bfloat16)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.01, optimizer_type="master_adamw")
    assert isinstance(opt, DirectMasterWeightAdamW)

    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    for p in model.parameters():
        assert p.dtype == torch.bfloat16
        assert opt.state[p]["master"].dtype == torch.float32
