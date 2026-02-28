# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .muon import Muon


logger = logging.get_logger(__name__)

# Module class names and parameter name patterns that should NOT use Muon
# (i.e., they should always use AdamW even when optimizer_type="muon").
_MUON_EXCLUDE_MODULE_TYPES = {"Embedding"}
_MUON_EXCLUDE_PARAM_PATTERNS = {"embed_tokens", "lm_head", "norm", "gate.weight"}


# https://github.com/meta-llama/llama-recipes/blob/v0.0.4/src/llama_recipes/policies/anyprecision_optimizer.py
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
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

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

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


class MultiOptimizer(Optimizer, Stateful):
    """
    A container that handles multiple optimizers (for ep and non-ep parameters when ep+fsdp2 is enabled)

    Mapping of name -> torch.optim.Optimizer with convenience methods.
    Compatible with torch.distributed.checkpoint optimizer APIs that accept a Mapping.

    This class is needed for EP+FSDP2 case because EP and non-EP param have different FSDP sharding dimension (dim-0 vs. dim-1).
    """

    def __init__(
        self,
        root_model: nn.Module,
        optimizers: dict,  # {"ep": opt1, "non_ep": opt2}
        key_names: list[str],
    ):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Return all param_groups from all internal optimizers."""
        all_groups = []
        for opt in self.optimizers_dict.values():
            all_groups.extend(opt.param_groups)
        return all_groups

    @property
    def state(self) -> Dict[torch.nn.Parameter, Any]:
        """Return merged state dict from all internal optimizers."""
        merged_state: Dict[torch.nn.Parameter, Any] = {}
        for opt in self.optimizers_dict.values():
            merged_state.update(opt.state)
        return merged_state

    def step(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.zero_grad()

    def state_dict(
        self,
    ) -> Dict[str, Any]:
        # get the flatten state dict for multi-optimizer
        merged: Dict[str, Any] = {}
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            sd = get_optimizer_state_dict(self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True))
            # check for key clashes before merging
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(
                    f"Key clash detected while merging state dict for optimizer '{name}': {', '.join(sorted(overlap))}"
                )
            else:
                logger.info_rank0(f"MultiOptimizer merged '{name}' state dict ({len(sd)} keys, total {len(merged) + len(sd)})")
            merged.update(sd)

        return merged

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Feed the same merged flattened dict to each sub-optimizer; PyTorch will
        # pick out only the entries for parameters that belong to that optimizer.
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            set_optimizer_state_dict(
                self.model,
                opt,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def register_step_pre_hook(self, hook):
        return [opt.register_step_pre_hook(hook) for opt in self.optimizers_dict.values()]

    def __len__(self) -> int:
        return len(self.optimizers_dict)


def _should_build_ep_aware(model: "nn.Module", param_groups: Optional[Sequence[Dict[str, Any]]]) -> bool:
    # Only auto-split when using FSDP2 with EP and no explicit param_groups
    if param_groups is not None:
        return False
    ps = get_parallel_state()
    if ps.dp_mode != "fsdp2" or not ps.ep_enabled:
        return False

    for m in model.modules():
        # Detect EP modules that skipped FSDP (e.g., QLoRAMoeExperts)
        if getattr(m, "_skip_fsdp", False):
            if any(p.requires_grad for p in m.parameters()):
                return True
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if isinstance(p, DTensor):
            mesh = getattr(p, "device_mesh", None)
            names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
            if "ep_fsdp" in names:
                return True
    return False


def _make_param_groups_for_subset(
    model: "nn.Module",
    params: Iterable[torch.nn.Parameter],
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    decay_param_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    name_by_param = {p: n for n, p in model.named_parameters()}
    params = [p for p in params if p.requires_grad]
    decayed = [p for p in params if name_by_param.get(p) in decay_param_names]
    undecayed = [p for p in params if name_by_param.get(p) not in decay_param_names]
    groups: List[Dict[str, Any]] = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": weight_decay})
    if undecayed:
        groups.append({"params": undecayed, "weight_decay": 0.0})
    return groups


# adapted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer_pt_utils.py#L1123
def get_parameter_names(model, forbidden_layer_types, forbidden_param_names):
    forbidden_layer_types = [] if forbidden_layer_types is None else forbidden_layer_types
    forbidden_param_names = [] if forbidden_param_names is None else forbidden_param_names
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_param_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if child.__class__.__name__ not in forbidden_layer_types
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_param_names)
        ]

    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_param_names)
    ]
    return result


_ANYPRECISION_STATE_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _classify_muon_params(
    model: "nn.Module",
) -> Tuple[List[torch.nn.Parameter], List[str], List[torch.nn.Parameter], List[str]]:
    """
    Split trainable parameters into Muon-eligible (2D+ weight matrices) and
    AdamW-eligible (embeddings, norms, biases, router gates, 1D params).

    Returns:
        (muon_params, muon_names, adamw_params, adamw_names)
    """
    muon_params: List[torch.nn.Parameter] = []
    muon_names: List[str] = []
    adamw_params: List[torch.nn.Parameter] = []
    adamw_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D or scalar params always go to AdamW
        if param.ndim < 2:
            adamw_params.append(param)
            adamw_names.append(name)
            continue

        # Check exclusion patterns in the parameter name
        name_lower = name.lower()
        excluded = any(pat in name_lower for pat in _MUON_EXCLUDE_PARAM_PATTERNS)

        if excluded:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, muon_names, adamw_params, adamw_names


def _make_muon_param_groups(
    model: "nn.Module",
    muon_params: List[torch.nn.Parameter],
    adamw_params: List[torch.nn.Parameter],
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build param groups for the Muon optimizer with proper weight decay splitting.

    Creates up to 4 groups:
      - muon + decay, muon + no_decay
      - adamw + decay, adamw + no_decay
    """
    decay_param_names = set(get_parameter_names(model, no_decay_modules, no_decay_params))
    name_by_param = {p: n for n, p in model.named_parameters()}

    groups: List[Dict[str, Any]] = []

    # Muon groups
    muon_decay = [p for p in muon_params if name_by_param.get(p) in decay_param_names]
    muon_no_decay = [p for p in muon_params if name_by_param.get(p) not in decay_param_names]
    if muon_decay:
        groups.append({"params": muon_decay, "lr": muon_lr, "weight_decay": weight_decay, "use_muon": True})
    if muon_no_decay:
        groups.append({"params": muon_no_decay, "lr": muon_lr, "weight_decay": 0.0, "use_muon": True})

    # AdamW groups
    adamw_decay = [p for p in adamw_params if name_by_param.get(p) in decay_param_names]
    adamw_no_decay = [p for p in adamw_params if name_by_param.get(p) not in decay_param_names]
    if adamw_decay:
        groups.append({"params": adamw_decay, "lr": adamw_lr, "weight_decay": weight_decay, "use_muon": False})
    if adamw_no_decay:
        groups.append({"params": adamw_no_decay, "lr": adamw_lr, "weight_decay": 0.0, "use_muon": False})

    return groups


def build_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    optimizer_dtype: str = "bf16",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    # Muon-specific arguments
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: Optional[str] = None,
    muon_momentum_dtype: Optional[torch.dtype] = None,
) -> "torch.optim.Optimizer":
    # EP-aware routing: for FSDP2+EP, split params into EP and non-EP groups and build two optimizers.
    if _should_build_ep_aware(model, param_groups):
        return build_ep_fsdp2_optimizer(
            model, lr, betas, eps, weight_decay, fused, optimizer_type, optimizer_dtype,
            param_groups, no_decay_modules, no_decay_params,
            muon_lr=muon_lr, muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov, muon_ns_steps=muon_ns_steps,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            muon_momentum_dtype=muon_momentum_dtype,
        )

    # Muon optimizer: split params into Muon (2D+ matrices) and AdamW (rest)
    if optimizer_type == "muon":
        muon_params, muon_names, adamw_params, adamw_names = _classify_muon_params(model)
        logger.info_rank0(
            f"Muon optimizer: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params"
        )
        logger.info_rank0(f"Muon params: {muon_names}")
        logger.info_rank0(f"AdamW params: {adamw_names}")

        param_groups = _make_muon_param_groups(
            model, muon_params, adamw_params,
            muon_lr=muon_lr, adamw_lr=lr, weight_decay=weight_decay,
            no_decay_modules=no_decay_modules, no_decay_params=no_decay_params,
        )
        adamw_state_dtype = _ANYPRECISION_STATE_DTYPES.get(optimizer_dtype)
        return Muon(
            param_groups,
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            weight_decay=weight_decay,
            adjust_lr_fn=muon_adjust_lr_fn,
            adamw_betas=betas,
            adamw_eps=eps,
            momentum_dtype=muon_momentum_dtype,
            adamw_state_dtype=adamw_state_dtype,
        )

    # Other cases remain the same
    if param_groups is None:
        decay_param_names = get_parameter_names(model, no_decay_modules, no_decay_params)
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_param_names and p.requires_grad],
                "weight_decay": weight_decay,
            },
        ]
        no_decay_parameters, no_decay_parameter_names = [], []
        for n, p in model.named_parameters():
            if n not in decay_param_names and p.requires_grad:
                no_decay_parameter_names.append(n)
                no_decay_parameters.append(p)

        if len(no_decay_parameters) > 0:
            logger.info_rank0(f"Parameters without weight decay: {no_decay_parameter_names}")
            param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    if optimizer_type == "adamw":
        foreach = not fused
        fused = fused
        optim = AdamW(param_groups, lr, betas, eps, weight_decay, fused=fused, foreach=foreach)
    elif optimizer_type == "anyprecision_adamw":
        state_dtype = _ANYPRECISION_STATE_DTYPES[optimizer_dtype]
        optim = AnyPrecisionAdamW(
            param_groups, lr, betas, eps, weight_decay,
            momentum_dtype=state_dtype, variance_dtype=state_dtype,
            compensation_buffer_dtype=state_dtype,
        )
    else:
        raise ValueError("Only adamw, anyprecision_adamw, and muon are supported as optimizers.")

    return optim


def build_ep_fsdp2_optimizer(
    model: "nn.Module",
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_type: str = "adamw",
    optimizer_dtype: str = "bf16",
    param_groups: Optional[Sequence[Dict[str, Any]]] = None,
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    # Muon-specific arguments
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: Optional[str] = None,
    muon_momentum_dtype: Optional[torch.dtype] = None,
):
    """
    Build a MultiOptimizer instance when model is parallelized with EP+FSDP2
    """
    # Collect param IDs from _skip_fsdp modules (e.g., QLoRAMoeExperts)
    skip_fsdp_param_ids = set()
    for m in model.modules():
        if getattr(m, "_skip_fsdp", False):
            for p in m.parameters():
                skip_fsdp_param_ids.add(id(p))

    ep_params: List[torch.nn.Parameter] = []
    non_ep_params: List[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in skip_fsdp_param_ids:
            ep_params.append(p)
            continue
        if DTensor is not None and isinstance(p, DTensor):
            mesh = getattr(p, "device_mesh", None)
            names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
            if "ep_fsdp" in names:
                ep_params.append(p)
                continue
        non_ep_params.append(p)

    logger.info_rank0(f"EP-aware optimizer: {len(ep_params)} EP params, {len(non_ep_params)} non-EP params")

    if optimizer_type == "muon":
        # For Muon + EP: classify within each EP/non-EP subset, then build Muon optimizers
        ep_param_set = set(id(p) for p in ep_params)
        non_ep_param_set = set(id(p) for p in non_ep_params)

        # Classify all model params into Muon vs AdamW
        all_muon, _, all_adamw, _ = _classify_muon_params(model)

        # Split Muon/AdamW params into EP and non-EP subsets
        ep_muon = [p for p in all_muon if id(p) in ep_param_set]
        ep_adamw = [p for p in all_adamw if id(p) in ep_param_set]
        non_ep_muon = [p for p in all_muon if id(p) in non_ep_param_set]
        non_ep_adamw = [p for p in all_adamw if id(p) in non_ep_param_set]

        ep_groups = _make_muon_param_groups(
            model, ep_muon, ep_adamw,
            muon_lr=muon_lr, adamw_lr=lr, weight_decay=weight_decay,
            no_decay_modules=no_decay_modules, no_decay_params=no_decay_params,
        )
        non_ep_groups = _make_muon_param_groups(
            model, non_ep_muon, non_ep_adamw,
            muon_lr=muon_lr, adamw_lr=lr, weight_decay=weight_decay,
            no_decay_modules=no_decay_modules, no_decay_params=no_decay_params,
        )
    else:
        ep_groups = _make_param_groups_for_subset(model, ep_params, weight_decay, no_decay_modules, no_decay_params)
        non_ep_groups = _make_param_groups_for_subset(
            model, non_ep_params, weight_decay, no_decay_modules, no_decay_params
        )

    def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
        foreach = not fused
        fused_ = fused
        if optimizer_type == "adamw":
            return AdamW(groups, lr, betas, eps, weight_decay, fused=fused_, foreach=foreach)
        elif optimizer_type == "anyprecision_adamw":
            state_dtype = _ANYPRECISION_STATE_DTYPES[optimizer_dtype]
            return AnyPrecisionAdamW(
                groups, lr, betas, eps, weight_decay,
                momentum_dtype=state_dtype, variance_dtype=state_dtype,
                compensation_buffer_dtype=state_dtype,
            )
        elif optimizer_type == "muon":
            adamw_state_dtype = _ANYPRECISION_STATE_DTYPES.get(optimizer_dtype)
            return Muon(
                groups, lr=muon_lr, momentum=muon_momentum,
                nesterov=muon_nesterov, ns_steps=muon_ns_steps,
                weight_decay=weight_decay, adjust_lr_fn=muon_adjust_lr_fn,
                adamw_betas=betas, adamw_eps=eps,
                momentum_dtype=muon_momentum_dtype,
                adamw_state_dtype=adamw_state_dtype,
            )
        else:
            raise ValueError("Only adamw, anyprecision_adamw, and muon are supported as optimizers.")

    optimizer_dict: Dict[str, Optimizer] = {}
    if ep_groups:
        optimizer_dict["ep"] = _build(ep_groups)
    if non_ep_groups:
        optimizer_dict["non_ep"] = _build(non_ep_groups)

    # cache for EP-aware grad clipping helpers
    all_ep_params = [p for g in ep_groups for p in g.get("params", [])] if ep_groups else []
    all_non_ep_params = [p for g in non_ep_groups for p in g.get("params", [])] if non_ep_groups else []
    model._ep_param_groups = {
        "ep": all_ep_params,
        "non_ep": all_non_ep_params,
    }
    # Build MultiOptimizer and attach a pre-step hook to sanitize DTensor states
    multi_opt = MultiOptimizer(model, optimizer_dict, key_names=["ep", "non_ep"])

    return multi_opt
