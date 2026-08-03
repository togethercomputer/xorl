"""Strict public-checkpoint inventory and native-FP8 pair assembly for GLM-5.2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from xorl.models.layers.moe.experts import MoEExperts
from xorl.ops.block_fp8_native import (
    NATIVE_BLOCK_FP8_CONTRACT_VERSION,
    NativeBlockFP8Linear,
    pack_fp8_as_float32,
    unpack_float32_as_fp8,
)


_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIX = ".weight_scale_inv"
_EXPERT_PAIR_KEY = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|weight_scale_inv)$"
)


def validate_glm52_native_fp8_config(quantization_config: Mapping | None) -> dict:
    """Validate and normalize the exact official block-FP8 contract."""

    if quantization_config is None:
        raise ValueError("GLM-5.2 native FP8 requires quantization_config")
    config = dict(quantization_config)
    expected = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Unsupported GLM-5.2 native FP8 quantization_config: {mismatches}")
    excluded = config.get("modules_to_not_convert")
    if not isinstance(excluded, list) or not all(isinstance(item, str) and item for item in excluded):
        raise ValueError("quantization_config.modules_to_not_convert must be a non-empty list of module names")
    config["modules_to_not_convert"] = list(excluded)
    return config


def _matches_exclusion(key: str, excluded_modules: Iterable[str]) -> bool:
    candidates = {key}
    source_suffix = ".self_attn.indexer.weights_proj.weight"
    if key.endswith(source_suffix):
        # Official checkpoint source name -> GLM config/module name.  The
        # quantization config excludes `indexers_proj`, while the serialized HF
        # tensor is nested under `indexer.weights_proj`.
        candidates.add(f"{key[: -len(source_suffix)]}.self_attn.indexers_proj")
    return any(
        candidate == module or candidate.startswith(f"{module}.")
        for candidate in candidates
        for module in excluded_modules
    )


def _uses_indexer_exclusion_alias(key: str, excluded_modules: Iterable[str]) -> bool:
    suffix = ".self_attn.indexer.weights_proj.weight"
    if not key.endswith(suffix):
        return False
    alias = f"{key[: -len(suffix)]}.self_attn.indexers_proj"
    return alias in excluded_modules


def _set_submodule(root: nn.Module, fqn: str, replacement: nn.Module) -> None:
    parent_fqn, _, child_name = fqn.rpartition(".")
    parent = root.get_submodule(parent_fqn) if parent_fqn else root
    setattr(parent, child_name, replacement)


class Glm52NativeBlockFP8Experts(nn.Module):
    """GLM-5.2 frozen FP8 experts restricted to rank-local routed partials."""

    fsdp_requires_full_precision = True
    contract_version = NATIVE_BLOCK_FP8_CONTRACT_VERSION

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        hidden_act: str = "silu",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if min(num_experts, hidden_size, intermediate_size) <= 0:
            raise ValueError("GLM native block-FP8 expert dimensions must be positive")
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError("GLM native block-FP8 expert dimensions must divide the 128x128 block shape")
        if hidden_act != "silu":
            raise ValueError(f"GLM native block-FP8 experts only support silu, got {hidden_act!r}")
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.hidden_act = hidden_act
        self.gated = True
        self.gate_up_bias = None
        self.down_bias = None
        self.ep_dispatch = "alltoall"

        self.gate_up_packed_weight_f32 = nn.Parameter(
            torch.empty(num_experts, hidden_size, (2 * intermediate_size) // 4, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_up_weight_scale_inv = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // 128,
                2 * intermediate_size // 128,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.down_packed_weight_f32 = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size // 4, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.down_weight_scale_inv = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size // 128,
                hidden_size // 128,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer("_engagement_count", torch.zeros((), dtype=torch.int64, device=device), persistent=False)
        self._last_routed_scaling_factor = None

    def _apply(self, fn, recurse: bool = True):
        names = (
            "gate_up_packed_weight_f32",
            "gate_up_weight_scale_inv",
            "down_packed_weight_f32",
            "down_weight_scale_inv",
        )
        probe = fn(torch.empty(0, dtype=torch.float32, device=self.gate_up_packed_weight_f32.device))
        protected = {name: self._parameters.pop(name) for name in names}
        try:
            result = super()._apply(fn, recurse=recurse)
            replacements = {}
            for name, parameter in protected.items():
                value = (
                    torch.empty_like(parameter, dtype=torch.float32, device=probe.device)
                    if parameter.is_meta
                    else parameter.to(device=probe.device, dtype=torch.float32)
                )
                replacements[name] = nn.Parameter(value, requires_grad=False)
        except Exception:
            self._parameters.update(protected)
            raise
        self._parameters.update(replacements)
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name, parameter in self.named_parameters(recurse=False):
            key = f"{prefix}{name}"
            tensor = state_dict.get(key)
            if tensor is not None and (
                tensor.dtype is not torch.float32 or tuple(tensor.shape) != tuple(parameter.shape)
            ):
                raise TypeError(
                    f"GLM native FP8 state {key} must be FP32 {tuple(parameter.shape)}, "
                    f"got {tensor.dtype} {tuple(tensor.shape)}; refusing a load_state_dict cast"
                )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def gate_up_proj(self) -> torch.Tensor:
        local_experts = int(self.gate_up_packed_weight_f32.shape[0])
        return unpack_float32_as_fp8(
            self.gate_up_packed_weight_f32,
            (local_experts, self.hidden_size, 2 * self.intermediate_size),
        )

    @property
    def down_proj(self) -> torch.Tensor:
        local_experts = int(self.down_packed_weight_f32.shape[0])
        return unpack_float32_as_fp8(
            self.down_packed_weight_f32,
            (local_experts, self.intermediate_size, self.hidden_size),
        )

    def load_prequantized(self, gate_up, gate_up_scale_inv, down, down_scale_inv) -> None:
        expected = {
            "gate_up": (gate_up, (self.num_experts, self.hidden_size, 2 * self.intermediate_size), torch.float8_e4m3fn),
            "gate_up_scale_inv": (
                gate_up_scale_inv,
                (self.num_experts, self.hidden_size // 128, 2 * self.intermediate_size // 128),
                torch.float32,
            ),
            "down": (down, (self.num_experts, self.intermediate_size, self.hidden_size), torch.float8_e4m3fn),
            "down_scale_inv": (
                down_scale_inv,
                (self.num_experts, self.intermediate_size // 128, self.hidden_size // 128),
                torch.float32,
            ),
        }
        for name, (tensor, shape, dtype) in expected.items():
            if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
            if dtype is torch.float32 and not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"{name} contains non-finite values")
        with torch.no_grad():
            self.gate_up_packed_weight_f32.copy_(pack_fp8_as_float32(gate_up).to(self.gate_up_packed_weight_f32.device))
            self.gate_up_weight_scale_inv.copy_(gate_up_scale_inv.to(self.gate_up_weight_scale_inv.device))
            self.down_packed_weight_f32.copy_(pack_fp8_as_float32(down).to(self.down_packed_weight_f32.device))
            self.down_weight_scale_inv.copy_(down_scale_inv.to(self.down_weight_scale_inv.device))

    def engagement_receipt(self) -> dict:
        return {
            "family": "glm52_native_block_fp8_experts",
            "contract_version": self.contract_version,
            "engagement_count": int(self._engagement_count.item()),
            "kernel": "xorl.ops.moe.sglang_fused_moe_strided.fused_experts_impl_strided",
            "local_experts": int(self.gate_up_packed_weight_f32.shape[0]),
            "combine_boundary": "sglang_ep_native_routed_partial_then_glm52_canonical_reduce",
            "routed_scale_placement": "final_moe_sum_reduce",
            "last_routed_scaling_factor": self._last_routed_scaling_factor,
            "cuda_graph_phase_evidence": "external_runner_receipt_required",
        }

    def sglang_ep_native_routed_partial(
        self,
        hidden_flat,
        routing_flat,
        local_ids,
        *,
        routed_scaling_factor: float = 1.0,
    ):
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("GLM native block-FP8 expert weights and scales must remain frozen")
        if torch.is_grad_enabled() and (hidden_flat.requires_grad or routing_flat.requires_grad):
            raise RuntimeError("GLM native block-FP8 phase-one expert forward is scoring-only")
        if routing_flat.dtype is not torch.float32:
            raise TypeError(f"GLM native block-FP8 experts require FP32 routing weights, got {routing_flat.dtype}")
        if hidden_flat.device.type != "cuda":
            raise RuntimeError("GLM native block-FP8 expert forward requires CUDA and pinned public SGLang")
        if hidden_flat.dtype is not torch.bfloat16:
            raise TypeError(f"GLM native block-FP8 experts require BF16 activations, got {hidden_flat.dtype}")
        if any(
            scale.dtype is not torch.float32 for scale in (self.gate_up_weight_scale_inv, self.down_weight_scale_inv)
        ):
            raise TypeError("GLM native block-FP8 expert scales must remain FP32")
        if local_ids.dtype is not torch.int32:
            raise TypeError(f"GLM native block-FP8 local expert ids must be int32, got {local_ids.dtype}")
        routed_scaling_factor = float(routed_scaling_factor)
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ValueError("GLM native block-FP8 routed scaling factor must be finite and positive")

        from xorl.ops.moe.sglang_fused_moe_strided import fused_experts_impl_strided  # noqa: PLC0415

        MoEExperts._ensure_sglang_server_args()
        output = fused_experts_impl_strided(
            hidden_flat.contiguous(),
            self.gate_up_proj.transpose(1, 2),
            self.down_proj.transpose(1, 2),
            routing_flat,
            local_ids,
            activation="silu",
            is_gated=True,
            use_fp8_w8a8=True,
            w1_scale=self.gate_up_weight_scale_inv.transpose(1, 2),
            w2_scale=self.down_weight_scale_inv.transpose(1, 2),
            block_shape=[128, 128],
            filter_expert=True,
            routed_scaling_factor=routed_scaling_factor,
        )
        self._last_routed_scaling_factor = routed_scaling_factor
        self._engagement_count.add_(1)
        return output

    def forward(
        self,
        hidden_states,
        routing_weights,
        selected_experts=None,
        *,
        sglang_ep_native_local_ids=None,
        routed_scaling_factor: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected GLM native block-FP8 expert arguments: {sorted(kwargs)}")
        if sglang_ep_native_local_ids is None:
            raise RuntimeError("GLM native block-FP8 experts are restricted to the canonical rank-local partial")
        if selected_experts is not None:
            raise RuntimeError("Global ids must not bypass GLM canonical native-EP mapping")
        return self.sglang_ep_native_routed_partial(
            hidden_states,
            routing_weights,
            sglang_ep_native_local_ids,
            routed_scaling_factor=routed_scaling_factor,
        )


def replace_glm52_native_fp8_modules(model: nn.Module, quantization_config: Mapping) -> dict:
    """Replace exactly the config-authorized quantized GLM projection families."""

    config = validate_glm52_native_fp8_config(quantization_config)
    excluded = config["modules_to_not_convert"]
    dense_replaced = []
    experts_replaced = []

    for fqn, module in tuple(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        weight_key = f"{fqn}.weight"
        if _matches_exclusion(weight_key, excluded):
            continue
        if module.bias is not None:
            raise ValueError(f"Quantized GLM projection {fqn} unexpectedly has a bias")
        replacement = NativeBlockFP8Linear.from_linear(module)
        replacement._source_fqn = fqn
        _set_submodule(model, fqn, replacement)
        dense_replaced.append(fqn)

    for fqn, module in tuple(model.named_modules()):
        if not isinstance(module, MoEExperts):
            continue
        replacement = Glm52NativeBlockFP8Experts(
            module.num_experts,
            int(module.gate_up_proj.shape[1]),
            module.intermediate_size,
            hidden_act=module.hidden_act,
            device=module.gate_up_proj.device,
        )
        replacement.ep_dispatch = module.ep_dispatch
        replacement._source_fqn = fqn
        _set_submodule(model, fqn, replacement)
        experts_replaced.append(fqn)

    return {
        "contract": "glm52_native_fp8_module_replacement_v1",
        "dense_replaced": tuple(dense_replaced),
        "experts_replaced": tuple(experts_replaced),
    }


def native_fp8_dense_source_map(model: nn.Module) -> dict[str, str]:
    return {
        module._source_fqn: fqn
        for fqn, module in model.named_modules()
        if isinstance(module, NativeBlockFP8Linear) and hasattr(module, "_source_fqn")
    }


@dataclass(frozen=True)
class Glm52OfficialFP8Inventory:
    quantized_weights: frozenset[str]
    quantized_scales: frozenset[str]
    config_excluded_weights: frozenset[str]
    config_alias_excluded_weights: frozenset[str]
    mtp_excluded_keys: frozenset[str]
    mtp_config_alias_excluded_keys: frozenset[str]
    ordinary_nonweight_keys: frozenset[str]
    orphan_scales: frozenset[str]
    unexplained_weights: frozenset[str]
    dtype_mismatches: frozenset[str]

    @classmethod
    def build(
        cls,
        weight_map: Mapping[str, str],
        quantization_config: Mapping,
        *,
        num_hidden_layers: int,
        num_nextn_predict_layers: int,
        tensor_metadata: Mapping[str, tuple[str, tuple[int, ...]]] | None = None,
    ) -> "Glm52OfficialFP8Inventory":
        config = validate_glm52_native_fp8_config(quantization_config)
        keys = set(weight_map)
        mtp_start = int(num_hidden_layers)
        mtp_end = mtp_start + int(num_nextn_predict_layers)

        mtp_excluded = set()
        trunk_keys = set()
        for key in keys:
            match = _LAYER_KEY.match(key)
            if match is not None and mtp_start <= int(match.group(1)) < mtp_end:
                mtp_excluded.add(key)
            else:
                trunk_keys.add(key)

        scale_keys = {key for key in trunk_keys if key.endswith(_SCALE_SUFFIX)}
        paired_weights = {key[: -len("_scale_inv")] for key in scale_keys}
        orphan_scales = {key for key in scale_keys if key[: -len("_scale_inv")] not in trunk_keys}
        quantized_weights = paired_weights & trunk_keys
        excluded = config["modules_to_not_convert"]
        config_excluded_weights = {
            key
            for key in trunk_keys
            if key.endswith(_WEIGHT_SUFFIX) and key not in quantized_weights and _matches_exclusion(key, excluded)
        }
        config_alias_excluded_weights = {
            key for key in config_excluded_weights if _uses_indexer_exclusion_alias(key, excluded)
        }
        mtp_config_alias_excluded_keys = {key for key in mtp_excluded if _uses_indexer_exclusion_alias(key, excluded)}
        unexplained_weights = {
            key
            for key in trunk_keys
            if key.endswith(_WEIGHT_SUFFIX) and key not in quantized_weights and key not in config_excluded_weights
        }

        dtype_mismatches = set()
        if tensor_metadata is not None:
            for key in quantized_weights:
                if tensor_metadata[key][0] != "F8_E4M3":
                    dtype_mismatches.add(key)
            for key in scale_keys:
                if tensor_metadata[key][0] != "F32":
                    dtype_mismatches.add(key)
            for key in config_excluded_weights:
                if tensor_metadata[key][0] != "BF16":
                    dtype_mismatches.add(key)

        ordinary_nonweight = trunk_keys - quantized_weights - scale_keys - config_excluded_weights

        return cls(
            quantized_weights=frozenset(quantized_weights),
            quantized_scales=frozenset(scale_keys),
            config_excluded_weights=frozenset(config_excluded_weights),
            config_alias_excluded_weights=frozenset(config_alias_excluded_weights),
            mtp_excluded_keys=frozenset(mtp_excluded),
            mtp_config_alias_excluded_keys=frozenset(mtp_config_alias_excluded_keys),
            ordinary_nonweight_keys=frozenset(ordinary_nonweight),
            orphan_scales=frozenset(orphan_scales),
            unexplained_weights=frozenset(unexplained_weights),
            dtype_mismatches=frozenset(dtype_mismatches),
        )

    def validate_complete(self) -> None:
        if self.orphan_scales:
            raise ValueError(f"Official GLM-5.2 index has orphan FP8 scales: {sorted(self.orphan_scales)[:8]}")
        if self.unexplained_weights:
            raise ValueError(
                f"Official GLM-5.2 index has weights that are neither paired nor excluded: "
                f"{sorted(self.unexplained_weights)[:8]}"
            )
        if self.dtype_mismatches:
            raise ValueError(
                f"Official GLM-5.2 index has dtype-contract mismatches: {sorted(self.dtype_mismatches)[:8]}"
            )
        if not self.mtp_excluded_keys:
            raise ValueError("Official GLM-5.2 index did not expose the declared MTP layer key set")


class NativeBlockFP8PairBuffer:
    """Assemble strict official dense FP8 pairs into DCP-visible parameters.

    ``source_to_target`` maps an official module prefix (without `.weight`) to
    the target model module FQN.  The target must be a
    :class:`NativeBlockFP8Linear`.  No unpaired tensor is silently discarded.
    """

    def __init__(self, model: torch.nn.Module, source_to_target: Mapping[str, str]):
        target_fqns = list(source_to_target.values())
        duplicate_fqns = sorted({target for target in target_fqns if target_fqns.count(target) > 1})
        if duplicate_fqns:
            raise ValueError(f"Native FP8 source mapping is not injective; duplicate targets: {duplicate_fqns}")

        modules = dict(model.named_modules(remove_duplicate=False))
        self._targets: dict[str, tuple[str, NativeBlockFP8Linear]] = {}
        target_module_ids: dict[int, str] = {}
        for source, target in source_to_target.items():
            module = modules.get(target)
            if not isinstance(module, NativeBlockFP8Linear):
                raise TypeError(f"Native FP8 target {target!r} is not a NativeBlockFP8Linear")
            prior_target = target_module_ids.get(id(module))
            if prior_target is not None:
                raise ValueError(
                    "Native FP8 source mapping is not injective; "
                    f"targets {prior_target!r} and {target!r} reference the same module"
                )
            target_module_ids[id(module)] = target
            self._targets[source] = (target, module)
        self._pending: dict[str, dict[str, torch.Tensor]] = {}
        self._completed: set[str] = set()

    def try_consume(self, key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        if key.endswith(_SCALE_SUFFIX):
            source = key[: -len(_SCALE_SUFFIX)]
            suffix = "scale"
        elif key.endswith(_WEIGHT_SUFFIX):
            source = key[: -len(_WEIGHT_SUFFIX)]
            suffix = "weight"
        else:
            return None
        if source not in self._targets:
            return None
        if source in self._completed:
            raise ValueError(f"Duplicate native FP8 tensor after completed pair: {key}")
        pending = self._pending.setdefault(source, {})
        if suffix in pending:
            raise ValueError(f"Duplicate native FP8 pair member: {key}")
        pending[suffix] = tensor
        if set(pending) != {"weight", "scale"}:
            return []

        target, module = self._targets[source]
        weight = pending["weight"]
        scale = pending["scale"]
        # Reuse module validation without mutating model state; checkpoint
        # dispatch remains responsible for DTensor/local placement.
        if weight.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"{source}.weight must be float8_e4m3fn, got {weight.dtype}")
        if tuple(weight.shape) != (module.out_features, module.in_features):
            raise ValueError(f"{source}.weight has unexpected shape {tuple(weight.shape)}")
        if scale.dtype is not torch.float32 or tuple(scale.shape) != tuple(module.weight_scale_inv.shape):
            raise ValueError(
                f"{source}.weight_scale_inv must be FP32 {tuple(module.weight_scale_inv.shape)}, "
                f"got {scale.dtype} {tuple(scale.shape)}"
            )
        if not bool(torch.all(torch.isfinite(scale))):
            raise ValueError(f"{source}.weight_scale_inv contains non-finite values")

        del self._pending[source]
        self._completed.add(source)
        return [
            (f"{target}.packed_weight_f32", pack_fp8_as_float32(weight)),
            (f"{target}.weight_scale_inv", scale),
        ]

    def validate_complete(self) -> None:
        missing = sorted(set(self._targets) - self._completed)
        pending = {source: sorted(parts) for source, parts in self._pending.items()}
        if missing or pending:
            raise ValueError(f"Incomplete native FP8 pairs: missing={missing[:8]} pending={pending}")


class NativeBlockFP8ExpertPairBuffer:
    """Assemble EP-local official expert pairs into packed GKN state."""

    def __init__(self, model: nn.Module, *, ep_rank: int, ep_size: int, num_experts: int):
        if ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"Invalid native FP8 expert rank {ep_rank} for ep_size={ep_size}")
        if num_experts % ep_size:
            raise ValueError(f"num_experts={num_experts} must divide ep_size={ep_size}")
        local_num_experts = num_experts // ep_size
        self.expert_start = ep_rank * local_num_experts
        self.expert_end = self.expert_start + local_num_experts
        self._targets: dict[int, tuple[str, Glm52NativeBlockFP8Experts]] = {}
        for fqn, module in model.named_modules():
            if not isinstance(module, Glm52NativeBlockFP8Experts):
                continue
            if module.num_experts != num_experts:
                raise ValueError(
                    f"Native FP8 expert target {fqn} declares {module.num_experts} experts, expected {num_experts}"
                )
            stored_experts = int(module.gate_up_packed_weight_f32.shape[0])
            if stored_experts not in {num_experts, local_num_experts}:
                raise ValueError(
                    f"Native FP8 expert target {fqn} stores {stored_experts} experts; "
                    f"expected global {num_experts} or EP-local {local_num_experts}"
                )
            match = re.search(r"model\.layers\.(\d+)\.mlp\.experts$", fqn)
            if match is not None:
                self._targets[int(match.group(1))] = (fqn, module)
        self._pairs: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
        self._completed: dict[tuple[int, str], dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._emitted: set[tuple[int, str]] = set()

    def has_targets(self) -> bool:
        return bool(self._targets)

    @staticmethod
    def parse_key(key: str) -> tuple[int, int, str, str] | None:
        match = _EXPERT_PAIR_KEY.match(key)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)

    def should_skip(self, key: str) -> bool:
        parsed = self.parse_key(key)
        if parsed is None or parsed[0] not in self._targets:
            return False
        return parsed[1] < self.expert_start or parsed[1] >= self.expert_end

    def try_consume(self, key: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        parsed = self.parse_key(key)
        if parsed is None:
            return None
        layer, expert, proj, suffix = parsed
        if layer not in self._targets:
            return None
        if expert < self.expert_start or expert >= self.expert_end:
            return []
        pair_key = (layer, expert, proj)
        pending = self._pairs.setdefault(pair_key, {})
        if suffix in pending:
            raise ValueError(f"Duplicate native FP8 expert pair member: {key}")
        pending[suffix] = tensor
        if set(pending) != {"weight", "weight_scale_inv"}:
            return []

        weight = pending["weight"]
        scale = pending["weight_scale_inv"]
        if weight.dtype is not torch.float8_e4m3fn:
            raise TypeError(f"{key} pair weight must be float8_e4m3fn, got {weight.dtype}")
        if scale.dtype is not torch.float32 or not bool(torch.all(torch.isfinite(scale))):
            raise TypeError(f"{key} pair scale must be finite FP32")
        del self._pairs[pair_key]
        completed = self._completed.setdefault((layer, proj), {})
        completed[expert] = (weight, scale)
        return self._maybe_emit(layer)

    def _maybe_emit(self, layer: int) -> list[tuple[str, torch.Tensor]]:
        target_fqn, module = self._targets[layer]
        local_experts = range(self.expert_start, self.expert_end)
        result = []

        gate_up_key = (layer, "gate_up")
        gates = self._completed.get((layer, "gate"), {})
        ups = self._completed.get((layer, "up"), {})
        if gate_up_key not in self._emitted and all(expert in gates and expert in ups for expert in local_experts):
            gate_up = torch.stack(
                [torch.cat((gates[expert][0], ups[expert][0]), dim=0).T.contiguous() for expert in local_experts]
            )
            gate_up_scale = torch.stack(
                [torch.cat((gates[expert][1], ups[expert][1]), dim=0).T.contiguous() for expert in local_experts]
            ).float()
            expected_weight_shape = (
                self.expert_end - self.expert_start,
                module.hidden_size,
                2 * module.intermediate_size,
            )
            expected_scale_shape = (
                self.expert_end - self.expert_start,
                module.hidden_size // 128,
                2 * module.intermediate_size // 128,
            )
            if tuple(gate_up.shape) != expected_weight_shape or tuple(gate_up_scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"Layer {layer} fused gate/up shape mismatch: {tuple(gate_up.shape)} {tuple(gate_up_scale.shape)}"
                )
            result.extend(
                [
                    (f"{target_fqn}.gate_up_packed_weight_f32", pack_fp8_as_float32(gate_up)),
                    (f"{target_fqn}.gate_up_weight_scale_inv", gate_up_scale),
                ]
            )
            self._emitted.add(gate_up_key)
            del self._completed[(layer, "gate")]
            del self._completed[(layer, "up")]

        down_key = (layer, "down")
        downs = self._completed.get((layer, "down"), {})
        if down_key not in self._emitted and all(expert in downs for expert in local_experts):
            down = torch.stack([downs[expert][0].T.contiguous() for expert in local_experts])
            down_scale = torch.stack([downs[expert][1].T.contiguous() for expert in local_experts]).float()
            expected_weight_shape = (
                self.expert_end - self.expert_start,
                module.intermediate_size,
                module.hidden_size,
            )
            expected_scale_shape = (
                self.expert_end - self.expert_start,
                module.intermediate_size // 128,
                module.hidden_size // 128,
            )
            if tuple(down.shape) != expected_weight_shape or tuple(down_scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"Layer {layer} fused down shape mismatch: {tuple(down.shape)} {tuple(down_scale.shape)}"
                )
            result.extend(
                [
                    (f"{target_fqn}.down_packed_weight_f32", pack_fp8_as_float32(down)),
                    (f"{target_fqn}.down_weight_scale_inv", down_scale),
                ]
            )
            self._emitted.add(down_key)
            del self._completed[(layer, "down")]
        return result

    def validate_complete(self) -> None:
        expected = {(layer, proj) for layer in self._targets for proj in ("gate_up", "down")}
        missing = sorted(expected - self._emitted)
        if missing or self._pairs or self._completed:
            raise ValueError(
                f"Incomplete native FP8 expert pairs: missing={missing[:8]} "
                f"pending_pairs={list(self._pairs)[:8]} pending_projections={list(self._completed)[:8]}"
            )


__all__ = [
    "Glm52NativeBlockFP8Experts",
    "Glm52OfficialFP8Inventory",
    "NativeBlockFP8ExpertPairBuffer",
    "NativeBlockFP8PairBuffer",
    "native_fp8_dense_source_map",
    "replace_glm52_native_fp8_modules",
    "validate_glm52_native_fp8_config",
]
