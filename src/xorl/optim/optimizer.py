# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .anyprecision_adamw import AnyPrecisionAdamW
from .multi_optimizer import MultiOptimizer
from .muon import Muon
from .signsgd import SignSGD


logger = logging.get_logger(__name__)

# Module class names and parameter name patterns that should NOT use Muon
# (i.e., they should always use AdamW even when optimizer_type="muon").
_MUON_EXCLUDE_MODULE_TYPES = {"Embedding"}
_MUON_EXCLUDE_PARAM_PATTERNS = {"embed_tokens", "lm_head", "norm", "gate.weight"}


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


def _normalize_optional_dtype(dtype: Any, *, field_name: str) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if dtype in _ANYPRECISION_STATE_DTYPES:
        return _ANYPRECISION_STATE_DTYPES[dtype]
    raise ValueError(f"Unsupported {field_name}: {dtype!r}. Expected 'fp32', 'bf16', a torch.dtype, or None.")


def _get_optimizer_cls_and_kwargs(
    optimizer_type: str,
    *,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_dtype: str = "bf16",
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[type, Dict[str, Any]]:
    """Return (optimizer_class, constructor_kwargs) without instantiating."""
    kwargs = optimizer_kwargs or {}

    if optimizer_type == "adamw":
        foreach = not fused
        ctor_kwargs = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            foreach=foreach,
            **kwargs,
        )
        return AdamW, ctor_kwargs
    elif optimizer_type == "anyprecision_adamw":
        state_dtype = _ANYPRECISION_STATE_DTYPES[optimizer_dtype]
        ctor_kwargs = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_dtype=state_dtype,
            variance_dtype=state_dtype,
            compensation_buffer_dtype=state_dtype,
            **kwargs,
        )
        return AnyPrecisionAdamW, ctor_kwargs
    elif optimizer_type == "sgd":
        sgd_defaults = {"momentum": 0.0, "nesterov": False}
        sgd_defaults.update(kwargs)
        ctor_kwargs = dict(lr=lr, weight_decay=weight_decay, **sgd_defaults)
        return torch.optim.SGD, ctor_kwargs
    elif optimizer_type == "signsgd":
        ctor_kwargs = dict(lr=lr, weight_decay=weight_decay, **kwargs)
        return SignSGD, ctor_kwargs
    elif optimizer_type == "muon":
        adamw_state_dtype = _ANYPRECISION_STATE_DTYPES.get(optimizer_dtype)
        momentum_dtype = kwargs.get("muon_momentum_dtype")
        if momentum_dtype is None:
            momentum_dtype = _ANYPRECISION_STATE_DTYPES.get(optimizer_dtype)
        ctor_kwargs = dict(
            lr=kwargs.get("muon_lr", 0.02),
            momentum=kwargs.get("muon_momentum", 0.95),
            nesterov=kwargs.get("muon_nesterov", True),
            ns_steps=kwargs.get("muon_ns_steps", 5),
            weight_decay=weight_decay,
            adjust_lr_fn=kwargs.get("muon_adjust_lr_fn"),
            ns_algorithm=kwargs.get("muon_ns_algorithm", "standard_newton_schulz"),
            ns_use_quack_kernels=kwargs.get("muon_ns_use_quack_kernels", True),
            gram_newton_schulz_num_restarts=kwargs.get("muon_gram_ns_num_restarts", 1),
            gram_newton_schulz_restart_iterations=kwargs.get("muon_gram_ns_restart_iterations"),
            adamw_betas=betas,
            adamw_eps=eps,
            momentum_dtype=_normalize_optional_dtype(momentum_dtype, field_name="muon_momentum_dtype"),
            grad_dtype=_normalize_optional_dtype(kwargs.get("muon_grad_dtype"), field_name="muon_grad_dtype"),
            update_dtype=_normalize_optional_dtype(kwargs.get("muon_update_dtype"), field_name="muon_update_dtype"),
            force_momentum_path=kwargs.get("muon_force_momentum_path", False),
            adamw_state_dtype=adamw_state_dtype,
        )
        return Muon, ctor_kwargs
    else:
        raise ValueError(
            f"Unsupported optimizer type: '{optimizer_type}'. Supported: adamw, anyprecision_adamw, sgd, signsgd, muon."
        )


def _create_optimizer(
    optimizer_type: str,
    param_groups: Sequence[Dict[str, Any]],
    *,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    fused: bool = False,
    optimizer_dtype: str = "bf16",
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Optimizer:
    """
    Single factory for all optimizer types.

    Common args (lr, betas, eps, weight_decay) are passed directly.
    Optimizer-specific args are passed via optimizer_kwargs:
      - sgd: {"momentum": 0.9, "nesterov": True}
      - signsgd: no optimizer-specific kwargs
      - muon: {"muon_lr": 0.02, "muon_momentum": 0.95, ...}
      - adamw/anyprecision_adamw: any extra kwargs forwarded to constructor
    """
    cls, ctor_kwargs = _get_optimizer_cls_and_kwargs(
        optimizer_type,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
        optimizer_dtype=optimizer_dtype,
        optimizer_kwargs=optimizer_kwargs,
    )
    return cls(param_groups, **ctor_kwargs)


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

    fused_gate_up_ids = {
        id(p)
        for p, n in zip(model.parameters(), (name_by_param.get(p, "") for p in model.parameters()))
        if "gate_up_proj" in name_by_param.get(p, "")
    }

    groups: List[Dict[str, Any]] = []

    # Muon groups
    muon_decay = [p for p in muon_params if name_by_param.get(p) in decay_param_names]
    muon_no_decay = [p for p in muon_params if name_by_param.get(p) not in decay_param_names]
    if muon_decay:
        groups.append(
            {
                "params": muon_decay,
                "lr": muon_lr,
                "weight_decay": weight_decay,
                "use_muon": True,
                "_fused_gate_up_ids": fused_gate_up_ids,
            }
        )
    if muon_no_decay:
        groups.append(
            {
                "params": muon_no_decay,
                "lr": muon_lr,
                "weight_decay": 0.0,
                "use_muon": True,
                "_fused_gate_up_ids": fused_gate_up_ids,
            }
        )

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
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> "torch.optim.Optimizer":
    """
    Build an optimizer for the given model.

    Args:
        model: The model whose parameters to optimize.
        lr: Learning rate (used as AdamW lr for Muon's non-Muon param groups).
        betas: AdamW beta coefficients.
        eps: AdamW epsilon.
        weight_decay: Weight decay coefficient.
        fused: Use fused AdamW kernel (mutually exclusive with foreach).
        optimizer_type: One of "adamw", "anyprecision_adamw", "sgd", "signsgd", "muon".
        optimizer_dtype: State dtype for anyprecision_adamw / muon ("fp32" or "bf16").
        param_groups: Custom param groups. If None, auto-built with weight decay splitting.
        no_decay_modules: Module class names to exclude from weight decay.
        no_decay_params: Parameter name patterns to exclude from weight decay.
        optimizer_kwargs: Optimizer-specific keyword arguments passed to the constructor.
            - sgd: {"momentum": 0.9, "nesterov": True}
            - signsgd: no optimizer-specific kwargs
            - muon: {"muon_lr": 0.02, "muon_momentum": 0.95, "muon_nesterov": True,
                      "muon_ns_steps": 5, "muon_adjust_lr_fn": None,
                      "muon_ns_algorithm": "standard_newton_schulz",
                      "muon_ns_use_quack_kernels": True, "muon_gram_ns_num_restarts": 1,
                      "muon_gram_ns_restart_iterations": None, "muon_momentum_dtype": None,
                      "muon_grad_dtype": None, "muon_update_dtype": None, "muon_force_momentum_path": False}
            - adamw/anyprecision_adamw: any extra kwargs forwarded to constructor
    """
    # EP-aware routing: for FSDP2+EP, split params into EP and non-EP groups and build two optimizers.
    if _should_build_ep_aware(model, param_groups):
        return build_ep_fsdp2_optimizer(
            model,
            lr,
            betas,
            eps,
            weight_decay,
            fused,
            optimizer_type,
            optimizer_dtype,
            param_groups,
            no_decay_modules,
            no_decay_params,
            optimizer_kwargs=optimizer_kwargs,
        )

    kwargs = optimizer_kwargs or {}

    # Muon optimizer: split params into Muon (2D+ matrices) and AdamW (rest)
    if optimizer_type == "muon":
        muon_params, muon_names, adamw_params, adamw_names = _classify_muon_params(model)
        logger.info_rank0(f"Muon optimizer: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params")
        logger.info_rank0(f"Muon params: {muon_names}")
        logger.info_rank0(f"AdamW params: {adamw_names}")

        muon_lr = kwargs.get("muon_lr", 0.02)
        param_groups = _make_muon_param_groups(
            model,
            muon_params,
            adamw_params,
            muon_lr=muon_lr,
            adamw_lr=lr,
            weight_decay=weight_decay,
            no_decay_modules=no_decay_modules,
            no_decay_params=no_decay_params,
        )
    elif param_groups is None:
        # Build param groups with weight decay splitting
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

    return _create_optimizer(
        optimizer_type,
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=fused,
        optimizer_dtype=optimizer_dtype,
        optimizer_kwargs=optimizer_kwargs,
    )


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
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Build a MultiOptimizer instance when model is parallelized with EP+FSDP2
    """
    kwargs = optimizer_kwargs or {}

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
        ep_param_set = {id(p) for p in ep_params}
        non_ep_param_set = {id(p) for p in non_ep_params}

        # Classify all model params into Muon vs AdamW
        all_muon, _, all_adamw, _ = _classify_muon_params(model)

        # Split Muon/AdamW params into EP and non-EP subsets
        ep_muon = [p for p in all_muon if id(p) in ep_param_set]
        ep_adamw = [p for p in all_adamw if id(p) in ep_param_set]
        non_ep_muon = [p for p in all_muon if id(p) in non_ep_param_set]
        non_ep_adamw = [p for p in all_adamw if id(p) in non_ep_param_set]

        muon_lr = kwargs.get("muon_lr", 0.02)
        ep_groups = _make_muon_param_groups(
            model,
            ep_muon,
            ep_adamw,
            muon_lr=muon_lr,
            adamw_lr=lr,
            weight_decay=weight_decay,
            no_decay_modules=no_decay_modules,
            no_decay_params=no_decay_params,
        )
        non_ep_groups = _make_muon_param_groups(
            model,
            non_ep_muon,
            non_ep_adamw,
            muon_lr=muon_lr,
            adamw_lr=lr,
            weight_decay=weight_decay,
            no_decay_modules=no_decay_modules,
            no_decay_params=no_decay_params,
        )
    else:
        ep_groups = _make_param_groups_for_subset(model, ep_params, weight_decay, no_decay_modules, no_decay_params)
        non_ep_groups = _make_param_groups_for_subset(
            model, non_ep_params, weight_decay, no_decay_modules, no_decay_params
        )

    def _build(groups: Sequence[Dict[str, Any]]) -> Optimizer:
        return _create_optimizer(
            optimizer_type,
            groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
            optimizer_dtype=optimizer_dtype,
            optimizer_kwargs=optimizer_kwargs,
        )

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
