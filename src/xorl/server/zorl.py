"""ZORL config normalization, runtime planning, and deterministic noise helpers."""

from __future__ import annotations

import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Literal, Mapping, Optional, Union

import torch


logger = logging.getLogger(__name__)


DEFAULT_ZORL_B_SIGMA = 0.01
DEFAULT_ZORL_NUM_PERTURBATION_PAIRS = 8
DEFAULT_ZORL_A_REFRESH_INTERVAL = 16
DEFAULT_ZORL_ANTITHETIC_SAMPLING = True
DEFAULT_ZORL_A_INIT = "gaussian_jl"
SUPPORTED_ZORL_A_INITS = {DEFAULT_ZORL_A_INIT}
DEFAULT_ZORL_PERTURBATION_MODE = "b_only"
SUPPORTED_ZORL_PERTURBATION_MODES = {"b_only", "fresh_ab"}

# Extra seed-mix component so the fresh_ab LoRA-A noise stream is decorrelated
# from the (unchanged) per-pair LoRA-B noise stream.
_ZORL_FRESH_AB_A_STREAM = 0xA5EED

_ZORL_KEY_ALIASES = {
    "sigma": "b_sigma",
    "refresh_interval": "a_refresh_interval",
    "num_pairs": "num_perturbation_pairs",
}

_ZORL_ALLOWED_KEYS = {
    "enabled",
    "b_sigma",
    "num_perturbation_pairs",
    "a_refresh_interval",
    "antithetic_sampling",
    "a_init",
    "perturbation_mode",
    "seed",
}


def _normalize_zorl_keys(raw_zorl_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(raw_zorl_config or {})
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        normalized[_ZORL_KEY_ALIASES.get(key, key)] = value
    return normalized


def normalize_zorl_runtime_config(
    raw_zorl_config: Optional[Dict[str, Any]],
    *,
    default_zorl_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize the optional session-level ZORL runtime config.

    ZORL currently assumes LoRA-A is refreshed at a family cadence and LoRA-B
    is the only parameter block perturbed by zeroth-order search.
    """

    zorl_config = _normalize_zorl_keys(raw_zorl_config)
    default_config = _normalize_zorl_keys(default_zorl_config)

    if not zorl_config and not default_config:
        return None

    merged = deepcopy(default_config)
    merged.update(deepcopy(zorl_config))

    # The Pydantic request models for zorl_config use ``extra="allow"`` for
    # forward-compat with new fields. To keep the API behaviour consistent
    # across both layers, drop unknown keys here with a warning instead of
    # raising — strict raise would mean Pydantic accepts the field at the
    # boundary but the normalizer rejects it later, surfacing the same input
    # via two different error paths.
    unknown_keys = sorted(set(merged) - _ZORL_ALLOWED_KEYS)
    if unknown_keys:
        logger.warning(f"Ignoring unknown ZORL config keys: {unknown_keys}")
        for key in unknown_keys:
            merged.pop(key, None)

    enabled = bool(merged.get("enabled", bool(zorl_config or default_config)))
    if not enabled:
        return None

    b_sigma = float(merged.get("b_sigma", DEFAULT_ZORL_B_SIGMA))
    num_perturbation_pairs = int(merged.get("num_perturbation_pairs", DEFAULT_ZORL_NUM_PERTURBATION_PAIRS))
    a_refresh_interval = int(merged.get("a_refresh_interval", DEFAULT_ZORL_A_REFRESH_INTERVAL))
    antithetic_sampling = bool(merged.get("antithetic_sampling", DEFAULT_ZORL_ANTITHETIC_SAMPLING))
    a_init = str(merged.get("a_init", DEFAULT_ZORL_A_INIT))
    perturbation_mode = str(merged.get("perturbation_mode", DEFAULT_ZORL_PERTURBATION_MODE))
    seed = merged.get("seed")

    if b_sigma <= 0:
        raise ValueError(f"zorl_config.b_sigma must be positive, got {b_sigma}")
    if num_perturbation_pairs <= 0:
        raise ValueError(f"zorl_config.num_perturbation_pairs must be positive, got {num_perturbation_pairs}")
    if a_refresh_interval < 0:
        raise ValueError(
            "zorl_config.a_refresh_interval must be non-negative "
            f"(0 disables automatic family refresh), got {a_refresh_interval}"
        )
    if a_init not in SUPPORTED_ZORL_A_INITS:
        raise ValueError(f"Unsupported ZORL A init {a_init!r}. Supported: {sorted(SUPPORTED_ZORL_A_INITS)}")
    if perturbation_mode not in SUPPORTED_ZORL_PERTURBATION_MODES:
        raise ValueError(
            f"Unsupported ZORL perturbation_mode {perturbation_mode!r}. "
            f"Supported: {sorted(SUPPORTED_ZORL_PERTURBATION_MODES)}"
        )
    if seed is not None:
        seed = int(seed)

    return {
        "enabled": True,
        "b_sigma": b_sigma,
        "num_perturbation_pairs": num_perturbation_pairs,
        "a_refresh_interval": a_refresh_interval,
        "antithetic_sampling": antithetic_sampling,
        "a_init": a_init,
        "perturbation_mode": perturbation_mode,
        "seed": seed,
    }


def _mix_seed(base_seed: Optional[int], *components: int) -> int:
    """Mix a stable integer seed from a base seed and generation metadata."""

    acc = int(base_seed or 0) & 0x7FFFFFFFFFFFFFFF
    for index, component in enumerate(components, start=1):
        acc = (acc * 6364136223846793005 + int(component) * 1442695040888963407 + index) & 0x7FFFFFFFFFFFFFFF
    return acc


@dataclass(frozen=True)
class ZORLFamilyState:
    """Metadata for one fixed LoRA-A family."""

    family_index: int
    family_id: str
    a_init: str
    a_seed: int
    created_at_generation: int


@dataclass(frozen=True)
class ZORLCandidateSpec:
    """Planned perturbation candidate for one generation.

    ``b_seed`` always identifies the LoRA-B noise draw. ``a_seed`` is only set
    for ``perturbation_mode='fresh_ab'``: the antithetic pair SHARES ``a_seed``
    (both directions use the same fresh LoRA-A draw) while the sign lives on
    LoRA-B, so the pair probes the direction ``eps_B @ eps_A``.
    """

    candidate_id: str
    family_id: str
    perturbation_index: int
    direction: Literal["positive", "negative"]
    b_seed: int
    a_seed: Optional[int] = None


@dataclass(frozen=True)
class ZORLMaterialization:
    """Local materialization policy for one globally planned ZORL generation."""

    mode: Literal["all", "pair_shard", "pair_range", "specs"] = "all"
    shard_index: int = 0
    num_shards: int = 1
    pair_start: int = 0
    pair_end: Optional[int] = None

    def owns_pair(self, perturbation_index: int, *, num_pairs: int) -> bool:
        if self.mode == "all":
            return 0 <= int(perturbation_index) < int(num_pairs)
        if self.mode == "specs":
            # Seed-transport mode: candidates are never exported to disk; the
            # scorer replicas materialize them from the returned specs.
            return False
        if self.mode == "pair_shard":
            return int(perturbation_index) % self.num_shards == self.shard_index
        if self.mode == "pair_range":
            end = int(num_pairs) if self.pair_end is None else self.pair_end
            return self.pair_start <= int(perturbation_index) < end
        raise ValueError(f"Unsupported ZORL materialization mode {self.mode!r}")

    def local_pair_indices(self, *, num_pairs: int) -> List[int]:
        return [index for index in range(int(num_pairs)) if self.owns_pair(index, num_pairs=num_pairs)]


@dataclass(frozen=True)
class ZORLGenerationPlan:
    """Planned candidate set for a single ZORL generation."""

    model_id: str
    generation: int
    generation_id: str
    family: ZORLFamilyState
    family_refreshed: bool
    candidates: List[ZORLCandidateSpec]


@dataclass
class ZORLSessionState:
    """Lightweight runtime scaffold for a ZORL-enabled training session."""

    model_id: str
    config: Dict[str, Any]
    generation: int = 0
    family_counter: int = 0
    active_family: Optional[ZORLFamilyState] = None
    active_generation: Optional[ZORLGenerationPlan] = None

    @classmethod
    def from_session_spec(cls, model_id: str, session_spec: Dict[str, Any]) -> Optional["ZORLSessionState"]:
        """Build runtime state for a ZORL-enabled session spec."""

        zorl_config = normalize_zorl_runtime_config(session_spec.get("zorl_config"))
        if zorl_config is None:
            return None
        return cls(model_id=model_id, config=zorl_config)

    def should_refresh_a(self) -> bool:
        """Return whether the next generation should start a fresh LoRA-A family."""

        if self.active_family is None:
            return True
        refresh_interval = int(self.config["a_refresh_interval"])
        if refresh_interval <= 0:
            return False
        return (self.generation - self.active_family.created_at_generation) >= refresh_interval

    def _activate_next_family(self) -> ZORLFamilyState:
        family_index = self.family_counter
        family = ZORLFamilyState(
            family_index=family_index,
            family_id=f"{self.model_id}-family-{family_index:06d}",
            a_init=self.config["a_init"],
            a_seed=_mix_seed(self.config.get("seed"), family_index, 1),
            created_at_generation=self.generation,
        )
        self.active_family = family
        self.family_counter += 1
        return family

    def seed_loaded_parent_family(self) -> ZORLFamilyState:
        """Treat the currently loaded adapter weights as an initialized family."""

        if self.active_generation is not None:
            raise ValueError(
                f"Cannot seed loaded ZORL parent while generation {self.active_generation.generation_id} is active"
            )

        family_index = self.family_counter
        family = ZORLFamilyState(
            family_index=family_index,
            family_id=f"{self.model_id}-family-{family_index:06d}",
            a_init=self.config["a_init"],
            # Loaded checkpoints may come from earlier searches, so the active
            # LoRA-A weights are not guaranteed to be reconstructible from the
            # current runtime seed/config alone.
            a_seed=-1,
            created_at_generation=self.generation,
        )
        self.active_family = family
        self.family_counter = family_index + 1
        return family

    def reset_runtime(
        self,
        *,
        a_seed: int,
        a_init: str = "gaussian_jl",
        zorl_seed: Optional[int] = None,
    ) -> ZORLFamilyState:
        """Reset generation bookkeeping around an explicitly initialized parent.

        The caller is responsible for restoring base weights and materializing
        the requested LoRA A/B values.  Keeping those tensor operations outside
        this lightweight state object makes the reset boundary explicit: this
        method only clears generation/family state and records the exact parent
        family that the caller installed.
        """
        if self.active_generation is not None:
            raise ValueError(
                f"Cannot reset ZORL runtime while generation {self.active_generation.generation_id} is active"
            )
        if a_init != "gaussian_jl":
            raise ValueError(f"Unsupported ZORL LoRA-A init: {a_init!r}")

        if zorl_seed is not None:
            self.config["seed"] = int(zorl_seed)
        self.config["a_init"] = a_init
        self.generation = 0
        self.family_counter = 1
        self.active_generation = None
        family = ZORLFamilyState(
            family_index=0,
            family_id=f"{self.model_id}-family-000000",
            a_init=a_init,
            a_seed=int(a_seed),
            created_at_generation=0,
        )
        self.active_family = family
        return family

    def begin_generation(
        self,
        *,
        num_pairs: Optional[int] = None,
        pair_seed_specs: Optional[List[Dict[str, Optional[int]]]] = None,
    ) -> ZORLGenerationPlan:
        """Plan and mark the next generation's antithetic LoRA-B perturbation candidates."""

        if self.active_generation is not None:
            raise ValueError(
                f"ZORL generation {self.active_generation.generation_id} is still active for model_id={self.model_id}"
            )

        family_refreshed = self.should_refresh_a()
        family = self._activate_next_family() if family_refreshed else self.active_family
        assert family is not None

        generation = self.generation
        generation_id = f"{family.family_id}-g{generation:06d}"
        candidates: List[ZORLCandidateSpec] = []
        if pair_seed_specs is not None:
            if not pair_seed_specs:
                raise ValueError("pair_seed_specs must contain at least one perturbation pair")
            if num_pairs is not None and int(num_pairs) != len(pair_seed_specs):
                raise ValueError(
                    "num_pairs must equal len(pair_seed_specs): "
                    f"num_pairs={num_pairs} seed_specs={len(pair_seed_specs)}"
                )
            num_pairs = len(pair_seed_specs)
        else:
            num_pairs = int(self.config["num_perturbation_pairs"] if num_pairs is None else num_pairs)
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        antithetic_sampling = bool(self.config["antithetic_sampling"])
        perturbation_mode = str(self.config.get("perturbation_mode", DEFAULT_ZORL_PERTURBATION_MODE))

        explicit_seeds: Optional[List[tuple[int, Optional[int]]]] = None
        if pair_seed_specs is not None:
            explicit_seeds = []
            seen = set()
            for perturbation_index, raw_spec in enumerate(pair_seed_specs):
                if not isinstance(raw_spec, dict):
                    raise ValueError(
                        f"pair_seed_specs[{perturbation_index}] must be a mapping, got {type(raw_spec).__name__}"
                    )
                b_seed = int(raw_spec["b_seed"])
                raw_a_seed = raw_spec.get("a_seed")
                a_seed = None if raw_a_seed is None else int(raw_a_seed)
                if b_seed < 0 or (a_seed is not None and a_seed < 0):
                    raise ValueError(f"pair_seed_specs[{perturbation_index}] contains a negative seed")
                if perturbation_mode == "fresh_ab" and a_seed is None:
                    raise ValueError(f"pair_seed_specs[{perturbation_index}].a_seed is required for fresh_ab")
                key = (b_seed, a_seed)
                if key in seen:
                    raise ValueError(
                        f"pair_seed_specs contains duplicate direction seeds at index {perturbation_index}"
                    )
                seen.add(key)
                explicit_seeds.append(key)

        for perturbation_index in range(num_pairs):
            perturb_seed = (
                explicit_seeds[perturbation_index][0]
                if explicit_seeds is not None
                else _mix_seed(self.config.get("seed"), family.family_index, generation, perturbation_index)
            )
            # fresh_ab draws a fresh LoRA-A per pair from a decorrelated seed
            # stream; the b_seed stream is byte-identical to b_only mode.
            a_seed = (
                explicit_seeds[perturbation_index][1]
                if explicit_seeds is not None
                else (
                    _mix_seed(
                        self.config.get("seed"),
                        family.family_index,
                        generation,
                        perturbation_index,
                        _ZORL_FRESH_AB_A_STREAM,
                    )
                    if perturbation_mode == "fresh_ab"
                    else None
                )
            )
            candidate_prefix = f"{family.family_id}-g{generation:06d}-p{perturbation_index:04d}"
            candidates.append(
                ZORLCandidateSpec(
                    candidate_id=f"{candidate_prefix}+",
                    family_id=family.family_id,
                    perturbation_index=perturbation_index,
                    direction="positive",
                    b_seed=perturb_seed,
                    a_seed=a_seed,
                )
            )
            if antithetic_sampling:
                candidates.append(
                    ZORLCandidateSpec(
                        candidate_id=f"{candidate_prefix}-",
                        family_id=family.family_id,
                        perturbation_index=perturbation_index,
                        direction="negative",
                        b_seed=perturb_seed,
                        a_seed=a_seed,
                    )
                )

        self.generation += 1
        self.active_generation = ZORLGenerationPlan(
            model_id=self.model_id,
            generation=generation,
            generation_id=generation_id,
            family=family,
            family_refreshed=family_refreshed,
            candidates=candidates,
        )
        return self.active_generation

    def complete_generation(self, generation_id: str) -> ZORLGenerationPlan:
        """Mark the active generation as completed and clear the runtime lock."""

        if self.active_generation is None:
            raise ValueError(f"No active ZORL generation for model_id={self.model_id}")
        if self.active_generation.generation_id != generation_id:
            raise ValueError(
                f"Active ZORL generation mismatch for model_id={self.model_id}: "
                f"expected {self.active_generation.generation_id}, got {generation_id}"
            )

        completed = self.active_generation
        self.active_generation = None
        return completed

    def abort_generation(self, generation_id: str) -> ZORLGenerationPlan:
        """Abort the active generation without applying an update."""

        return self.complete_generation(generation_id)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of the current runtime state."""

        return {
            "model_id": self.model_id,
            "config": deepcopy(self.config),
            "generation": self.generation,
            "family_counter": self.family_counter,
            "active_family": None
            if self.active_family is None
            else {
                "family_index": self.active_family.family_index,
                "family_id": self.active_family.family_id,
                "a_init": self.active_family.a_init,
                "a_seed": self.active_family.a_seed,
                "created_at_generation": self.active_family.created_at_generation,
            },
            "active_generation_id": None if self.active_generation is None else self.active_generation.generation_id,
        }


def normalize_zorl_materialization(
    raw_materialization: Optional[Dict[str, Any]],
    *,
    num_pairs: int,
) -> ZORLMaterialization:
    """Normalize and validate local materialization settings.

    The generation plan remains global; this policy only selects which
    perturbation pairs a worker exports for scoring.
    """

    data = dict(raw_materialization or {})
    mode = str(data.get("mode", "all"))
    if mode in {"replicated", "full"}:
        mode = "all"
    if mode == "none":
        mode = "specs"
    if mode not in {"all", "pair_shard", "pair_range", "specs"}:
        raise ValueError(f"Unsupported ZORL materialization mode {mode!r}")

    total_pairs = int(num_pairs)
    if total_pairs <= 0:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}")

    if mode == "all":
        return ZORLMaterialization()

    if mode == "specs":
        return ZORLMaterialization(mode="specs")

    if mode == "pair_shard":
        num_shards = int(data.get("num_shards", 0))
        shard_index = int(data.get("shard_index", -1))
        if num_shards <= 0:
            raise ValueError(f"materialization.num_shards must be positive, got {num_shards}")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(
                "materialization.shard_index must be in "
                f"[0, num_shards), got shard_index={shard_index} num_shards={num_shards}"
            )
        return ZORLMaterialization(mode="pair_shard", shard_index=shard_index, num_shards=num_shards)

    pair_start = int(data.get("pair_start", 0))
    pair_end_raw = data.get("pair_end", total_pairs)
    pair_end = total_pairs if pair_end_raw is None else int(pair_end_raw)
    if pair_start < 0 or pair_start > total_pairs:
        raise ValueError(f"materialization.pair_start must be in [0, num_pairs], got {pair_start}")
    if pair_end < pair_start or pair_end > total_pairs:
        raise ValueError(
            "materialization.pair_end must be in [pair_start, num_pairs], "
            f"got pair_start={pair_start} pair_end={pair_end} num_pairs={total_pairs}"
        )
    return ZORLMaterialization(mode="pair_range", pair_start=pair_start, pair_end=pair_end)


def filter_zorl_materialized_candidates(
    candidates: List[ZORLCandidateSpec],
    materialization: ZORLMaterialization,
    *,
    num_pairs: int,
) -> List[ZORLCandidateSpec]:
    """Return the candidates owned by this worker under a local materialization policy."""

    return [
        candidate
        for candidate in candidates
        if materialization.owns_pair(candidate.perturbation_index, num_pairs=num_pairs)
    ]


# ---------------------------------------------------------------------------
# Seed -> noise contract (sglang-aligned).
#
# With explicit-seed candidate transport the sglang SCORER replicas materialize
# each candidate from {b_seed, a_seed, sigma, sign} through their own
# `LoRAManager._zorl_normalized_{b,a}_noises`, so the PS-side fold MUST
# regenerate bit-identical noise. sglang draws noise per *raw entry* of its
# normalized (fused) adapter layout — NOT per HF module — which differs from
# the historical xorl per-parameter draw for the fused families:
#
#   * q/k/v      -> ONE fused `<attn>.qkv_proj.lora_{A,B}` draw
#                   (A: rank-stacked [rq+rk+rv, in], B: out-stacked [oq+ok+ov, r])
#   * gate/up A  -> ONE fused `<mlp>.gate_up_proj.lora_A` draw ([2r, in]);
#                   gate/up B stay per-module (sglang splits the fused B back
#                   into `gate_proj.lora_B` / `up_proj.lora_B` raw entries)
#   * MoE expert B (underscore params) -> per-module raw entries in xorl's own
#                   trainer orientation (`..._lora_B`, [E, r, out]) — 1:1
#   * MoE expert A -> ONE fused DOTTED `...experts.gate_up_proj.lora_A` draw at
#                   the sglang adapter shape [E, 2r, in] (xorl A is [E, in, r]);
#                   expert down A is a DOTTED `...experts.down_proj.lora_A` raw
#                   entry at the transposed [E, r, in] shape
#
# The raw entries are drawn in sorted(raw_name) order from one generator per
# seed, so the sort keys must be the sglang names as well. The helpers below
# build that layout from the xorl LoRA param dict and slice/transpose the raw
# draws back to per-xorl-param tensors. Everything (candidate export, b_only
# fold, fresh_ab fold) flows through `_iter_zorl_param_noises`, so path-mode
# exports and seed-mode folds always use the same stream.
# ---------------------------------------------------------------------------

_ZORL_QKV_MODULES = ("q_proj", "k_proj", "v_proj")


def _zorl_noise_device() -> str:
    """Resolve the noise-generation device (mirrors sglang `_zorl_noise_device`).

    Default 'gpu' (single batched CUDA randn per seed over the raw layout —
    bit-identical to sglang's default GPU scheme on identical hardware);
    ``XORL_ZORL_NOISE_DEVICE=cpu`` selects the per-raw-entry CPU draw that is
    bit-identical to sglang's CPU scheme. The PS and the scorer replicas must
    run the SAME setting for seed-transport parity.
    """

    value = os.environ.get("XORL_ZORL_NOISE_DEVICE", "gpu").strip().lower()
    if value == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _zorl_strip_param_name(name: str) -> str:
    """Normalize a LoRA param name to sglang's 'trainer key' name space."""

    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name


def _zorl_sglang_noise_layout(
    lora_params: Dict[str, Any],
    *,
    marker: str,
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
) -> tuple[List[tuple[str, tuple[int, ...]]], List[tuple[str, str, Any]]]:
    """Build the sglang raw-noise draw plan for the params matching ``marker``.

    Returns ``(sorted_raw, assemble)``:
      * ``sorted_raw``: ``[(raw_name, raw_shape)]`` in sorted(raw_name) order —
        the exact order the RNG stream is consumed (== sglang's
        ``_zorl_{b,a}_noise_layout`` for the equivalent adapter).
      * ``assemble``: ``[(param_name, raw_name, extract)]`` where
        ``extract(raw_tensor)`` returns the noise shaped like the xorl param.

    Fused groups are only formed when the group is complete enough to load in
    sglang (q+v present for qkv — sglang zero-fills a missing k; both gate and
    up for gate_up). Incomplete groups fall back to 1:1 per-param draws: such
    adapters cannot be served by the sglang scorer anyway, so no cross-tree
    parity contract exists for them and the historical xorl stream is kept.
    """

    names = sorted(n for n in lora_params if marker in n)
    logical_shapes = logical_shapes or {}
    unknown_shapes = set(logical_shapes) - set(lora_params)
    if unknown_shapes:
        raise ValueError(f"Logical ZORL shapes reference unknown parameters: {sorted(unknown_shapes)}")
    shapes = {n: tuple(logical_shapes.get(n, tuple(lora_params[n].shape))) for n in names}
    stripped = {n: _zorl_strip_param_name(n) for n in names}

    def _identity(t: torch.Tensor) -> torch.Tensor:
        return t

    def _row_slice(start: int, stop: int):
        def _extract(t: torch.Tensor) -> torch.Tensor:
            return t[..., start:stop, :].contiguous()

        return _extract

    def _row_slice_t(start: int, stop: int):
        def _extract(t: torch.Tensor) -> torch.Tensor:
            return t[..., start:stop, :].transpose(-2, -1).contiguous()

        return _extract

    def _transpose_last_two(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(-2, -1).contiguous()

    # Group dotted q/k/v params per attention prefix for fusion.
    qkv_groups: Dict[str, Dict[str, str]] = {}
    gate_up_groups: Dict[str, Dict[str, str]] = {}
    expert_gate_up_groups: Dict[str, Dict[str, str]] = {}
    for n in names:
        key = stripped[n]
        if ".experts." in key:
            for module in ("gate_proj", "up_proj"):
                suffix = f".experts.{module}_{marker}"
                if key.endswith(suffix):
                    prefix = key[: -len(f".{module}_{marker}")]
                    expert_gate_up_groups.setdefault(prefix, {})[module] = n
            continue
        for module in _ZORL_QKV_MODULES:
            suffix = f".{module}.{marker}"
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                qkv_groups.setdefault(prefix, {})[module] = n
        if marker == "lora_A":
            for module in ("gate_proj", "up_proj"):
                suffix = f".{module}.{marker}"
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    gate_up_groups.setdefault(prefix, {})[module] = n

    raw_entries: List[tuple[str, tuple[int, ...]]] = []
    assemble: List[tuple[str, str, Any]] = []
    handled: set[str] = set()

    # Fused dotted q/k/v (both A and B are out/rank-stacked along dim -2 in the
    # sglang adapter; a missing k is zero-filled with v's shape, and its noise
    # rows are drawn but unused).
    for prefix, group in qkv_groups.items():
        if "q_proj" not in group or "v_proj" not in group:
            continue  # incomplete: cannot load in sglang -> 1:1 fallback below
        raw_name = f"{prefix}.qkv_proj.{marker}"
        segment_shapes = [shapes[group[m]] if m in group else shapes[group["v_proj"]] for m in _ZORL_QKV_MODULES]
        trailing = {s[1:] for s in segment_shapes}
        if len(trailing) != 1:
            raise ValueError(f"Inconsistent q/k/v LoRA shapes under {prefix!r}: {segment_shapes}")
        fused_rows = sum(s[0] for s in segment_shapes)
        raw_entries.append((raw_name, (fused_rows,) + segment_shapes[0][1:]))
        offset = 0
        for module, seg_shape in zip(_ZORL_QKV_MODULES, segment_shapes):
            if module in group:
                assemble.append((group[module], raw_name, _row_slice(offset, offset + seg_shape[0])))
                handled.add(group[module])
            offset += seg_shape[0]

    # Fused dotted gate/up LoRA-A (sglang keeps ONE gate_up_proj.lora_A raw
    # entry; the B factors stay per-module raw entries).
    for prefix, group in gate_up_groups.items():
        if "gate_proj" not in group or "up_proj" not in group:
            continue
        gate_shape = shapes[group["gate_proj"]]
        up_shape = shapes[group["up_proj"]]
        if gate_shape != up_shape:
            raise ValueError(f"gate/up LoRA-A shapes differ under {prefix!r}: {gate_shape} vs {up_shape}")
        raw_name = f"{prefix}.gate_up_proj.{marker}"
        rows = gate_shape[0]
        raw_entries.append((raw_name, (2 * rows,) + gate_shape[1:]))
        assemble.append((group["gate_proj"], raw_name, _row_slice(0, rows)))
        assemble.append((group["up_proj"], raw_name, _row_slice(rows, 2 * rows)))
        handled.add(group["gate_proj"])
        handled.add(group["up_proj"])

    # MoE expert underscore params. B raw entries are 1:1 in xorl's own trainer
    # orientation/names (sglang emits `..._lora_B` raw names for the fold-side
    # trainer). A raw entries use the DOTTED sglang adapter names at the
    # sglang (out-first) orientation: xorl A [E, in, r] <-> sglang [E, r, in].
    for prefix, group in expert_gate_up_groups.items():
        if marker != "lora_A" or "gate_proj" not in group or "up_proj" not in group:
            continue
        gate_shape = shapes[group["gate_proj"]]
        up_shape = shapes[group["up_proj"]]
        if gate_shape != up_shape:
            raise ValueError(f"expert gate/up LoRA-A shapes differ under {prefix!r}: {gate_shape} vs {up_shape}")
        # xorl [E, in, r] -> sglang per-module [E, r, in] -> fused [E, 2r, in].
        rank = gate_shape[-1]
        raw_shape = gate_shape[:-2] + (2 * rank, gate_shape[-2])
        raw_name = f"{prefix}.gate_up_proj.{marker}"
        raw_entries.append((raw_name, raw_shape))
        assemble.append((group["gate_proj"], raw_name, _row_slice_t(0, rank)))
        assemble.append((group["up_proj"], raw_name, _row_slice_t(rank, 2 * rank)))
        handled.add(group["gate_proj"])
        handled.add(group["up_proj"])

    for n in names:
        if n in handled:
            continue
        key = stripped[n]
        shape = shapes[n]
        if ".experts." in key and marker == "lora_A" and key.endswith(f"_{marker}"):
            # Expert down (or unfused expert) LoRA-A: dotted sglang raw name at
            # the transposed sglang orientation.
            module = key[key.rfind(".") + 1 : -len(f"_{marker}")]
            prefix = key[: key.rfind(".") + 1]
            raw_name = f"{prefix}{module}.{marker}"
            raw_shape = shape[:-2] + (shape[-1], shape[-2])
            raw_entries.append((raw_name, raw_shape))
            assemble.append((n, raw_name, _transpose_last_two))
        else:
            # 1:1: dotted dense modules (o/down/gate/up B) and expert
            # underscore B params already match sglang's raw names/shapes.
            raw_entries.append((key, shape))
            assemble.append((n, key, _identity))

    raw_names = [raw_name for raw_name, _shape in raw_entries]
    if len(set(raw_names)) != len(raw_names):
        raise ValueError(f"Duplicate raw noise entries in ZORL layout: {sorted(raw_names)}")
    sorted_raw = sorted(raw_entries, key=lambda item: item[0])
    return sorted_raw, assemble


def _iter_zorl_param_noises(
    lora_params: Dict[str, Any],
    *,
    seed: int,
    marker: str,
    param_names: Optional[Collection[str]] = None,
    output_device: Optional[Union[str, torch.device]] = None,
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    local_slices: Optional[Mapping[str, tuple[slice, ...]]] = None,
) -> List[tuple[str, torch.Tensor]]:
    """Generate deterministic Gaussian noise for every LoRA param matching ``marker``.

    Draws follow sglang's raw-layout convention (see the module comment above)
    so the noise is bit-identical to what a sglang scorer replica materializes
    from the same seed: one generator seeded once per seed, consuming raw
    entries in sorted(raw_name) order. Device resolution mirrors sglang:
    CPU -> one ``randn`` per raw entry; CUDA (default when available) -> one
    flat batched ``randn`` sliced per raw entry (sglang's GPU scheme).

    ``param_names`` restricts the RETURNED tensors to a subset of params. The
    RNG stream discipline is unchanged — the FULL raw layout of the adapter is
    always consumed (one draw per raw entry, in sorted(raw_name) order) so a
    subset extraction is bit-identical to slicing the full extraction. This is
    what lets the fresh_ab fold stream module chunks without perturbing seed
    transport.

    ``output_device=None`` (default) returns CPU tensors (bit-preserving copy,
    the historical device-agnostic export behavior); passing a device keeps or
    moves the extracted noise there so GPU folds avoid host round-trips. The
    move never changes values, only residency. NOTE: on the CUDA draw scheme,
    unsliced extractions returned on the draw device may be VIEWS of the one
    flat per-seed randn buffer, keeping it alive while they are referenced
    (~4 bytes per adapter param, transient per seed).
    """

    sorted_raw, assemble = _zorl_sglang_noise_layout(lora_params, marker=marker, logical_shapes=logical_shapes)
    device = _zorl_noise_device()
    if param_names is not None:
        requested = set(param_names)
        known = {param_name for param_name, _raw, _extract in assemble}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown LoRA params requested from the ZORL {marker} noise stream: {sorted(unknown)}")
        assemble = [item for item in assemble if item[0] in requested]

    raw_noises: Dict[str, torch.Tensor] = {}
    if device == "cpu":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        for raw_name, raw_shape in sorted_raw:
            raw_noises[raw_name] = torch.randn(raw_shape, generator=generator, dtype=torch.float32)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        total = sum(math.prod(shape) for _name, shape in sorted_raw)
        flat = torch.randn(total, generator=generator, device=device, dtype=torch.float32)
        if marker == "lora_B" and os.environ.get("XORL_ZORL_RADEMACHER_B", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            # Mirror sglang's GPU-path Rademacher-B projection choke point.
            flat = torch.sign(flat)
        offset = 0
        for raw_name, raw_shape in sorted_raw:
            numel = math.prod(raw_shape)
            raw_noises[raw_name] = flat[offset : offset + numel].view(raw_shape)
            offset += numel

    target_device = torch.device("cpu") if output_device is None else torch.device(output_device)
    noises: List[tuple[str, torch.Tensor]] = []
    for param_name, raw_name, extract in sorted(assemble, key=lambda item: item[0]):
        tensor = extract(raw_noises[raw_name])
        if local_slices is not None and param_name in local_slices:
            tensor = tensor[local_slices[param_name]].contiguous()
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
        noises.append((param_name, tensor))
    return noises


def iter_zorl_b_noises(
    lora_params: Dict[str, Any],
    *,
    seed: int,
    param_names: Optional[Collection[str]] = None,
    output_device: Optional[Union[str, torch.device]] = None,
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    local_slices: Optional[Mapping[str, tuple[slice, ...]]] = None,
) -> List[tuple[str, torch.Tensor]]:
    """Generate deterministic Gaussian noise tensors for every LoRA-B parameter."""

    if os.environ.get("XORL_ZORL_NOISE_LAYOUT", "").strip() == "philox_subseed_v2":
        return _iter_zorl_param_noises_v2(
            lora_params,
            seed=seed,
            marker="lora_B",
            param_names=param_names,
            output_device=output_device,
            logical_shapes=logical_shapes,
            local_slices=local_slices,
        )
    return _iter_zorl_param_noises(
        lora_params,
        seed=seed,
        marker="lora_B",
        param_names=param_names,
        output_device=output_device,
        logical_shapes=logical_shapes,
        local_slices=local_slices,
    )


def iter_zorl_fresh_ab_a_noises(
    lora_params: Dict[str, Any],
    *,
    seed: int,
    param_names: Optional[Collection[str]] = None,
    output_device: Optional[Union[str, torch.device]] = None,
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    local_slices: Optional[Mapping[str, tuple[slice, ...]]] = None,
) -> List[tuple[str, torch.Tensor]]:
    """Generate deterministic unit-Gaussian noise tensors for every LoRA-A parameter.

    fresh_ab candidates REPLACE the parent LoRA-A with this draw (unit scale,
    matching the sglang fresh_ab reference where A is a plain ``randn``).
    """

    if os.environ.get("XORL_ZORL_NOISE_LAYOUT", "").strip() == "philox_subseed_v2":
        return _iter_zorl_param_noises_v2(
            lora_params,
            seed=seed,
            marker="lora_A",
            param_names=param_names,
            output_device=output_device,
            logical_shapes=logical_shapes,
            local_slices=local_slices,
        )
    return _iter_zorl_param_noises(
        lora_params,
        seed=seed,
        marker="lora_A",
        param_names=param_names,
        output_device=output_device,
        logical_shapes=logical_shapes,
        local_slices=local_slices,
    )


def build_zorl_candidate_lora_state_dict(
    lora_params: Dict[str, Any],
    *,
    b_seed: int,
    direction: Literal["positive", "negative"],
    b_sigma: float,
) -> Dict[str, torch.Tensor]:
    """Build an exported LoRA state dict for one ZORL candidate."""

    sign = 1.0 if direction == "positive" else -1.0
    perturbations = dict(iter_zorl_b_noises(lora_params, seed=b_seed))
    candidate_state_dict: Dict[str, torch.Tensor] = {}

    for name, param in lora_params.items():
        tensor = param.data.detach().cpu().to(torch.float32)
        if "lora_B" in name:
            # Keep the same fused-add operation order as scorer-side seeded
            # materialization so path transport and seed transport agree
            # byte-for-byte in fp32.
            tensor = tensor.add(perturbations[name], alpha=sign * float(b_sigma))
        candidate_state_dict[name] = tensor

    return candidate_state_dict


def build_zorl_update_from_rewards(
    lora_params: Dict[str, Any],
    *,
    pair_seeds_and_scores: List[tuple[int, float]],
) -> tuple[Dict[str, torch.Tensor], float]:
    """Build a weighted LoRA-B-only update from normalized pair scores."""

    if not pair_seeds_and_scores:
        raise ValueError("pair_seeds_and_scores must not be empty")

    b_param_names = sorted(name for name in lora_params if "lora_B" in name)
    update = {name: torch.zeros(tuple(lora_params[name].shape), dtype=torch.float32) for name in b_param_names}

    weight_scale = 1.0 / float(len(pair_seeds_and_scores))
    for b_seed, normalized_score in pair_seeds_and_scores:
        for name, noise in iter_zorl_b_noises(lora_params, seed=b_seed):
            update[name].add_(noise, alpha=float(normalized_score) * weight_scale)

    squared_norm = 0.0
    for tensor in update.values():
        squared_norm += float(torch.sum(tensor * tensor).item())
    return update, math.sqrt(max(squared_norm, 0.0))


# ---------------------------------------------------------------------------
# fresh_ab (EGGROLL-style paired outer-product) ES probe + fold.
#
# Each pair i probes the FULL-WEIGHT direction eps_B,i @ eps_A,i where BOTH
# factors are fresh seeded gaussians (the antithetic sign lives on B only, so
# the pair shares eps_A). The candidate adapter REPLACES the parent factors:
#
#   A = eps_A            (unit gaussian)
#   B = sign * sigma * eps_B
#
# and the parent LoRA-B stays == 0 forever (the parent adapter's served delta
# is identically zero). The per-generation reward-weighted update is
#
#   G = (1/N) * sum_i z_i * scaling * (eps_B,i @ eps_A,i)
#
# a full base-weight-shaped [out, in] update (rank <= N*r, unrepresentable in
# the rank-r parent), which the runner folds into the BASE weights through the
# base-model optimizer. ``scaling`` (= lora_alpha / r) is included because the
# candidate forward applies it on top of B @ A, so the ES estimator must move
# along exactly the direction that was probed. ``sigma`` is NOT included —
# matching both the b_only fold convention and the sglang fresh_ab reference
# (u = (1/N) * sum_i z_i * scaling * eps_B,i @ eps_A,i): the 1/sigma of the
# canonical ES gradient (1/(N*sigma)) * sum z_i * dW_i cancels the sigma in
# dW_i = sigma * scaling * eps_B @ eps_A.
# ---------------------------------------------------------------------------


_ZORL_LORA_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_A",
    ".lora_B",
    "_lora_A",
    "_lora_B",
)


def zorl_lora_module_key(name: str) -> Optional[tuple[str, str]]:
    """Split a LoRA parameter name into (module_key, factor) with factor in {"A", "B"}.

    Handles the dotted LoraLinear layout (``...q_proj.lora_A``) and the
    underscore MoE layout (``...experts.gate_proj_lora_A``); trailing
    ``.weight`` suffixes are stripped first.
    """

    stripped = name[: -len(".weight")] if name.endswith(".weight") else name
    for suffix in ("_lora_A", ".lora_A"):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)], "A"
    for suffix in ("_lora_B", ".lora_B"):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)], "B"
    return None


def pair_zorl_fresh_ab_lora_params(lora_params: Dict[str, Any]) -> Dict[str, tuple[str, str]]:
    """Pair LoRA-A/B parameter names per target module: module_key -> (a_name, b_name)."""

    a_names: Dict[str, str] = {}
    b_names: Dict[str, str] = {}
    for name in lora_params:
        parsed = zorl_lora_module_key(name)
        if parsed is None:
            raise ValueError(f"Unexpected non-LoRA parameter {name!r} in ZORL fresh_ab adapter params")
        module_key, factor = parsed
        target = a_names if factor == "A" else b_names
        if module_key in target:
            raise ValueError(f"Duplicate LoRA-{factor} parameter for module {module_key!r}")
        target[module_key] = name

    if set(a_names) != set(b_names):
        missing_b = sorted(set(a_names) - set(b_names))
        missing_a = sorted(set(b_names) - set(a_names))
        raise ValueError(
            "fresh_ab requires paired LoRA-A/LoRA-B parameters per module; "
            f"modules missing lora_B={missing_b!r}, missing lora_A={missing_a!r}"
        )
    return {module_key: (a_names[module_key], b_names[module_key]) for module_key in sorted(a_names)}


def _zorl_fresh_ab_module_delta(eps_a: torch.Tensor, eps_b: torch.Tensor) -> torch.Tensor:
    """Outer-product probe direction for one module, in base-weight orientation.

    - 2D LoraLinear: A [r, in], B [out, r]  ->  B @ A            = [out, in]
    - 3D MoE (G, K, N): A [E|1, in, r], B [E|1, r, out] -> A @ B = [E, in, out]
      (batch dims broadcast, covering hybrid-shared adapters where one factor
      is shared across experts with a leading dim of 1).
    """

    if eps_a.ndim == 2 and eps_b.ndim == 2:
        return eps_b @ eps_a
    if eps_a.ndim == 3 and eps_b.ndim == 3:
        return torch.matmul(eps_a, eps_b)
    raise ValueError(f"Unsupported fresh_ab LoRA factor ranks: eps_A ndim={eps_a.ndim}, eps_B ndim={eps_b.ndim}")


def build_zorl_fresh_ab_candidate_lora_state_dict(
    lora_params: Dict[str, Any],
    *,
    a_seed: int,
    b_seed: int,
    direction: Literal["positive", "negative"],
    b_sigma: float,
) -> Dict[str, torch.Tensor]:
    """Build an exported LoRA state dict for one fresh_ab candidate.

    Unlike the b_only builder this REPLACES the parent factors: the candidate
    is (A = eps_A, B = sign * sigma * eps_B); parent lora_A/lora_B values are
    ignored (parent B is required to stay 0 for the fold-into-base apply path).
    """

    sign = 1.0 if direction == "positive" else -1.0
    a_noises = dict(iter_zorl_fresh_ab_a_noises(lora_params, seed=a_seed))
    b_noises = dict(iter_zorl_b_noises(lora_params, seed=b_seed))
    # Validate the A/B pairing up front so a malformed adapter fails loudly.
    pair_zorl_fresh_ab_lora_params(lora_params)

    candidate_state_dict: Dict[str, torch.Tensor] = {}
    for name, param in lora_params.items():
        if "lora_B" in name:
            candidate_state_dict[name] = (sign * float(b_sigma)) * b_noises[name]
        elif "lora_A" in name:
            candidate_state_dict[name] = a_noises[name].clone()
        else:
            candidate_state_dict[name] = param.data.detach().cpu().to(torch.float32)
    return candidate_state_dict


def zorl_fresh_ab_module_update_shape(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Base-weight-orientation shape of one module's dense fresh_ab update G.

    - 2D LoraLinear: A [r, in], B [out, r]  ->  [out, in]
    - 3D MoE (G, K, N): A [E|1, in, r], B [E|1, r, out] -> [E, in, out]
    """

    if len(a_shape) == 2:
        return (b_shape[0], a_shape[1])
    num_experts = max(a_shape[0], b_shape[0])
    return (num_experts, a_shape[1], b_shape[2])


def accumulate_zorl_fresh_ab_base_update_(
    lora_params: Dict[str, Any],
    accumulators: Dict[str, torch.Tensor],
    *,
    pair_seeds_and_scores: List[tuple[int, int, float]],
    scaling: float,
    noise_layout: str = "sequential_v1",
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    local_slices: Optional[Mapping[str, tuple[slice, ...]]] = None,
) -> None:
    """Accumulate the reward-weighted fresh_ab update into caller-provided buffers.

    In-place adds ``G = (1/N) * sum_i z_i * scaling * (eps_B,i @ eps_A,i)``
    (base-weight orientation) into ``accumulators[module_key]`` for exactly the
    module keys present in ``accumulators`` — which may be any subset of the
    adapter's modules. Buffers may be fp32 views (e.g. windows of a fused
    ``gate_up_proj`` base-weight buffer); their device selects the fold device.

    MEMORY DISCIPLINE: this is the streaming counterpart of the sglang
    dense_accum fold. Live tensors at any instant are bounded by
        (a) the caller's accumulators (one module chunk, NOT the whole model),
        (b) ONE pair's eps_A/eps_B draw (~2 x 4 bytes/adapter-param, on the
            fold device; the full per-seed raw layout is drawn to keep the
            RNG stream seed-transport-exact, then only the chunk's params are
            extracted), and
        (c) ONE module's transient dense delta.
    Nothing is staged on the host when the accumulators live on GPU.

    Numerics are bit-identical to the historical all-modules pair-major fold:
    same draws, same per-module pair-order ``add_(delta, alpha=coef)`` chain
    (zero-initialized buffers; ``0 + coef*delta == coef*delta`` exactly).
    """

    if not pair_seeds_and_scores:
        raise ValueError("pair_seeds_and_scores must not be empty")

    modules = pair_zorl_fresh_ab_lora_params(lora_params)
    unknown = set(accumulators) - set(modules)
    if unknown:
        raise ValueError(f"Unknown fresh_ab module keys in accumulators: {sorted(unknown)}")

    selected = {module_key: modules[module_key] for module_key in sorted(accumulators)}
    for module_key, (a_name, b_name) in selected.items():
        expected = zorl_fresh_ab_module_update_shape(tuple(lora_params[a_name].shape), tuple(lora_params[b_name].shape))
        got = tuple(accumulators[module_key].shape)
        if got != expected:
            raise ValueError(
                f"fresh_ab accumulator for module {module_key!r} has shape {got}, "
                f"expected {expected} from LoRA factor shapes"
            )

    needed_a = {a_name for a_name, _b_name in selected.values()}
    needed_b = {b_name for _a_name, b_name in selected.values()}
    fold_device = next(iter(accumulators.values())).device

    weight_scale = 1.0 / float(len(pair_seeds_and_scores))
    for a_seed, b_seed, normalized_score in pair_seeds_and_scores:
        coefficient = float(normalized_score) * weight_scale * float(scaling)
        if coefficient == 0.0:
            continue
        if noise_layout == ZORL_NOISE_LAYOUT_V2:
            a_noises = dict(
                _iter_zorl_param_noises_v2(
                    lora_params,
                    seed=a_seed,
                    marker="lora_A",
                    param_names=needed_a,
                    output_device=fold_device,
                    logical_shapes=logical_shapes,
                    local_slices=local_slices,
                )
            )
            b_noises = dict(
                _iter_zorl_param_noises_v2(
                    lora_params,
                    seed=b_seed,
                    marker="lora_B",
                    param_names=needed_b,
                    output_device=fold_device,
                    logical_shapes=logical_shapes,
                    local_slices=local_slices,
                )
            )
        else:
            a_noises = dict(
                iter_zorl_fresh_ab_a_noises(
                    lora_params,
                    seed=a_seed,
                    param_names=needed_a,
                    output_device=fold_device,
                    logical_shapes=logical_shapes,
                    local_slices=local_slices,
                )
            )
            b_noises = dict(
                iter_zorl_b_noises(
                    lora_params,
                    seed=b_seed,
                    param_names=needed_b,
                    output_device=fold_device,
                    logical_shapes=logical_shapes,
                    local_slices=local_slices,
                )
            )
        for module_key, (a_name, b_name) in selected.items():
            delta = _zorl_fresh_ab_module_delta(a_noises[a_name], b_noises[b_name])
            accumulators[module_key].add_(delta, alpha=coefficient)
            del delta
        del a_noises, b_noises


def build_zorl_fresh_ab_base_update_from_rewards(
    lora_params: Dict[str, Any],
    *,
    pair_seeds_and_scores: List[tuple[int, int, float]],
    scaling: float,
) -> tuple[Dict[str, torch.Tensor], float]:
    """Build the reward-weighted full-shape fresh_ab base update.

    WARNING — reference/small-model use only: this materializes the FULL dense
    G for ALL modules simultaneously (a full fp32 copy of every LoRA-targeted
    base weight). The model-scale apply path
    (``ModelRunner._apply_zorl_fresh_ab_base_update``) must stream module chunks
    through ``accumulate_zorl_fresh_ab_base_update_`` instead; this wrapper is
    the numerical reference the streaming fold is gated against.

    Args:
        lora_params: The parent adapter's LoRA parameters (shapes/names define
            both the seeded noise streams and the per-module output shapes).
        pair_seeds_and_scores: One entry per antithetic pair:
            ``(a_seed, b_seed, normalized_score)``.
        scaling: Structural LoRA scaling (lora_alpha / lora_rank) applied by
            the candidate forward on top of B @ A.

    Returns:
        (updates, update_norm) where ``updates[module_key]`` is the dense
        G = (1/N) * sum_i z_i * scaling * (eps_B,i @ eps_A,i) in base-weight
        orientation and ``update_norm`` is the global Frobenius norm of G.
    """

    if not pair_seeds_and_scores:
        raise ValueError("pair_seeds_and_scores must not be empty")

    modules = pair_zorl_fresh_ab_lora_params(lora_params)
    updates: Dict[str, torch.Tensor] = {
        module_key: torch.zeros(
            zorl_fresh_ab_module_update_shape(tuple(lora_params[a_name].shape), tuple(lora_params[b_name].shape)),
            dtype=torch.float32,
        )
        for module_key, (a_name, b_name) in modules.items()
    }
    accumulate_zorl_fresh_ab_base_update_(
        lora_params,
        updates,
        pair_seeds_and_scores=pair_seeds_and_scores,
        scaling=scaling,
    )

    squared_norm = 0.0
    for tensor in updates.values():
        squared_norm += float(torch.sum(tensor * tensor).item())
    return updates, math.sqrt(max(squared_norm, 0.0))


# ---------------------------------------------------------------------------
# Noise layout v2: per-raw-entry sub-seeded counter-based RNG ("philox_subseed_v2")
#
# v1 (above) consumes ONE sequential generator over the whole sorted raw
# layout, so any consumer must redraw the FULL layout to extract a slice —
# the fresh_ab streaming fold paid num_chunks x num_pairs full redraws.
# v2 keys an explicit philox4x32-10 stream per (seed, raw_name):
#   sub_seed = blake2b-64("zorl-noise/philox_subseed_v2:{seed}:{raw_name}")
#   value[i] = BoxMuller(philox4x32_10(key=sub_seed, counter=i//4))[i%4]
# Properties: any raw entry (and later, any element range — the counter IS
# the element address) is drawable independently and bit-identically on CPU
# and CUDA; the (key, counter) addressing is exactly what a fused Triton/SM
# generator kernel consumes, so serving-side in-SM generation can adopt the
# same stream. The sglang scorer twin lives in lora_manager.py
# (_zorl_normalized_{a,b}_noises, layout="philox_subseed_v2") and MUST match
# bit-for-bit — both sides pin the shared fixture vector in
# ZORL_PHILOX_V2_FIXTURE / tests.
# ---------------------------------------------------------------------------

ZORL_NOISE_LAYOUT_V1 = "sequential_v1"
ZORL_NOISE_LAYOUT_V2 = "philox_subseed_v2"

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_U32 = 0xFFFFFFFF


def zorl_param_subseed(seed: int, raw_name: str) -> int:
    """Derive the 63-bit per-raw-entry philox key for noise layout v2."""
    import hashlib

    digest = hashlib.blake2b(
        f"zorl-noise/{ZORL_NOISE_LAYOUT_V2}:{int(seed)}:{raw_name}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _u32_mulhilo(a: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact (hi32, lo32) of a_u32 * m_u32 using int64 ops without overflow.

    Split m into 16-bit halves so every intermediate stays < 2**48.
    """
    m_lo = m & 0xFFFF
    m_hi = (m >> 16) & 0xFFFF
    p_lo = a * m_lo  # < 2**48
    p_hi = a * m_hi  # < 2**48
    lo = (p_lo + ((p_hi & 0xFFFF) << 16)) & _U32
    carry = (p_lo + ((p_hi & 0xFFFF) << 16)) >> 32
    hi = ((p_hi >> 16) + carry) & _U32
    return hi, lo


def _zorl_philox4x32_10(idx: torch.Tensor, key: int) -> torch.Tensor:
    """philox4x32-10 over int64 counter indices; returns [n, 4] u32 (as int64).

    Counter = (idx_lo32, idx_hi32, 0, 0); key = (sub_seed_lo32, sub_seed_hi32).
    """
    c0 = idx & _U32
    c1 = (idx >> 32) & _U32
    c2 = torch.zeros_like(idx)
    c3 = torch.zeros_like(idx)
    k0 = int(key) & _U32
    k1 = (int(key) >> 32) & _U32
    for _ in range(10):
        hi0, lo0 = _u32_mulhilo(c0, _PHILOX_M0)
        hi1, lo1 = _u32_mulhilo(c2, _PHILOX_M1)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & _U32,
            lo1,
            (hi0 ^ c3 ^ k1) & _U32,
            lo0,
        )
        k0 = (k0 + _PHILOX_W0) & _U32
        k1 = (k1 + _PHILOX_W1) & _U32
    return torch.stack([c0, c1, c2, c3], dim=-1)


def _zorl_philox4x32_10_batch(idx: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Batched philox4x32-10: idx [C] int64 counters x keys [B] int64 -> [B, C, 4] u32.

    Same per-(key, counter) bits as ``_zorl_philox4x32_10`` (gated in tests);
    one broadcasted kernel chain for a whole pair-block instead of one chain
    per pair — the fold's launch-count killer.
    """
    B = keys.shape[0]
    c0 = (idx & _U32).unsqueeze(0).expand(B, -1).contiguous()
    c1 = ((idx >> 32) & _U32).unsqueeze(0).expand(B, -1).contiguous()
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = (keys & _U32).unsqueeze(1)
    k1 = ((keys >> 32) & _U32).unsqueeze(1)
    w0 = 0
    w1 = 0
    for _ in range(10):
        hi0, lo0 = _u32_mulhilo(c0, _PHILOX_M0)
        hi1, lo1 = _u32_mulhilo(c2, _PHILOX_M1)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ ((k0 + w0) & _U32)) & _U32,
            lo1,
            (hi0 ^ c3 ^ ((k1 + w1) & _U32)) & _U32,
            lo0,
        )
        w0 = (w0 + _PHILOX_W0) & _U32
        w1 = (w1 + _PHILOX_W1) & _U32
    return torch.stack([c0, c1, c2, c3], dim=-1)


_ZORL_PHILOX_SLAB_COUNTERS = 1 << 22  # bound fp32/int64 transients per slab


def zorl_philox_randn_batch(
    sub_seeds: List[int],
    numel: int,
    *,
    device: Union[str, torch.device] = "cpu",
    counter_offset: int = 0,
) -> torch.Tensor:
    """Deterministic standard-normal draws for MANY keys at once -> [B, numel].

    4 values per philox counter; element i of key k lives at counter
    ``counter_offset + i//4`` under key k — random-access in both axes, and
    the exact (key, counter) addressing a fused generator kernel consumes.
    Uniform->normal is fp32 Box-Muller: u = (x + 0.5) / 2**32 in (0, 1).
    Counters are processed in slabs to bound transient memory.
    """
    B = len(sub_seeds)
    if numel <= 0 or B == 0:
        return torch.empty(B, max(numel, 0), dtype=torch.float32, device=device)
    if torch.device(device).type == "cuda" and os.environ.get("XORL_ZORL_PHILOX_TRITON", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        try:
            from xorl.server.zorl_philox_triton import HAS_TRITON, philox_randn_batch_triton

            if HAS_TRITON:
                return philox_randn_batch_triton(
                    [int(s) for s in sub_seeds],
                    numel,
                    device=device,
                    counter_offset=counter_offset,
                )
        except Exception:  # noqa: BLE001 - fall through to the torch reference
            pass
    keys = torch.tensor(sub_seeds, device=device, dtype=torch.int64)
    n_counters = (numel + 3) // 4
    out = torch.empty(B, n_counters * 4, dtype=torch.float32, device=device)
    two_pi = 6.2831855
    for slab_start in range(0, n_counters, _ZORL_PHILOX_SLAB_COUNTERS):
        slab_n = min(_ZORL_PHILOX_SLAB_COUNTERS, n_counters - slab_start)
        idx = torch.arange(
            counter_offset + slab_start,
            counter_offset + slab_start + slab_n,
            device=device,
            dtype=torch.int64,
        )
        u32 = _zorl_philox4x32_10_batch(idx, keys)  # [B, slab_n, 4]
        u = (u32.to(torch.float32) + 0.5) * (1.0 / 4294967296.0)
        r0 = torch.sqrt(-2.0 * torch.log(u[..., 0]))
        t0 = two_pi * u[..., 1]
        r1 = torch.sqrt(-2.0 * torch.log(u[..., 2]))
        t1 = two_pi * u[..., 3]
        z = torch.stack(
            [r0 * torch.cos(t0), r0 * torch.sin(t0), r1 * torch.cos(t1), r1 * torch.sin(t1)],
            dim=-1,
        )
        out[:, slab_start * 4 : (slab_start + slab_n) * 4] = z.reshape(B, -1)
    return out[:, :numel]


def zorl_philox_randn(
    sub_seed: int,
    numel: int,
    *,
    device: Union[str, torch.device] = "cpu",
    counter_offset: int = 0,
) -> torch.Tensor:
    """Single-key convenience wrapper over ``zorl_philox_randn_batch``."""
    return zorl_philox_randn_batch([int(sub_seed)], numel, device=device, counter_offset=counter_offset)[0]


# Shared cross-repo fixture: first 4 values of (seed=1234567,
# raw_name="model.layers.0.mlp.experts.gate_up_proj.lora_A") and the derived
# sub-seed. The sglang twin pins the SAME constants; a divergence in either
# repo fails its own unit test before it can corrupt a run.
ZORL_PHILOX_V2_FIXTURE = {
    "seed": 1234567,
    "raw_name": "model.layers.0.mlp.experts.gate_up_proj.lora_A",
}


def _iter_zorl_param_noises_v2(
    lora_params: Dict[str, Any],
    *,
    seed: int,
    marker: str,
    param_names: Optional[Collection[str]] = None,
    output_device: Optional[Union[str, torch.device]] = None,
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    local_slices: Optional[Mapping[str, tuple[slice, ...]]] = None,
) -> List[tuple[str, torch.Tensor]]:
    """Layout-v2 twin of ``_iter_zorl_param_noises``: draws ONLY the raw
    entries referenced by the requested params (no full-layout consumption).
    """
    sorted_raw, assemble = _zorl_sglang_noise_layout(lora_params, marker=marker, logical_shapes=logical_shapes)
    raw_shapes = dict(sorted_raw)
    if param_names is not None:
        requested = set(param_names)
        known = {param_name for param_name, _raw, _extract in assemble}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown LoRA params requested from the ZORL {marker} noise stream: {sorted(unknown)}")
        assemble = [item for item in assemble if item[0] in requested]

    device = torch.device(output_device) if output_device is not None else torch.device("cpu")
    rademacher_b = marker == "lora_B" and os.environ.get("XORL_ZORL_RADEMACHER_B", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    raw_noises: Dict[str, torch.Tensor] = {}
    for _param, raw_name, _extract in assemble:
        if raw_name in raw_noises:
            continue
        shape = raw_shapes[raw_name]
        flat = zorl_philox_randn(zorl_param_subseed(seed, raw_name), math.prod(shape), device=device)
        if rademacher_b:
            flat = torch.sign(flat)
        raw_noises[raw_name] = flat.view(shape)

    noises: List[tuple[str, torch.Tensor]] = []
    for param_name, raw_name, extract in sorted(assemble, key=lambda item: item[0]):
        tensor = extract(raw_noises[raw_name])
        if local_slices is not None and param_name in local_slices:
            tensor = tensor[local_slices[param_name]].contiguous()
        noises.append((param_name, tensor))
    return noises


def zorl_noise_layout_from_env(default: str = ZORL_NOISE_LAYOUT_V1) -> str:
    layout = os.environ.get("XORL_ZORL_NOISE_LAYOUT", default).strip() or default
    if layout not in (ZORL_NOISE_LAYOUT_V1, ZORL_NOISE_LAYOUT_V2):
        raise ValueError(f"Unknown XORL_ZORL_NOISE_LAYOUT: {layout!r}")
    return layout


def accumulate_zorl_fresh_ab_base_update_gemm_(
    lora_params: Dict[str, Any],
    accumulators: Dict[str, torch.Tensor],
    *,
    pair_seeds_and_scores: List[tuple[int, int, float]],
    scaling: float,
    noise_layout: str = ZORL_NOISE_LAYOUT_V2,
    pair_block: Optional[int] = None,
    matmul_precision: Optional[str] = None,
    logical_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    local_slices: Optional[Mapping[str, tuple[slice, ...]]] = None,
) -> None:
    """Tensor-core fold: G[module] = (1/N) * sum_i z_i*s * (eps_B,i @ eps_A,i)
    computed as ONE stacked GEMM per (module, pair-block) with the pair
    coefficients folded into the B factor.

    Same contract as ``accumulate_zorl_fresh_ab_base_update_`` but:
      * noise layout v2 (per-raw-entry sub-seeds) — draws exactly the params
        it folds, zero full-layout redraws;
      * pairs are stacked along the rank axis (A_cat [.., in, B*r],
        B_cat [.., B*r, out] with c_i pre-multiplied) so the accumulation is
        GEMM-shaped; ``matmul_precision`` 'tf32'|'bf16'|'fp32' (default env
        XORL_ZORL_FOLD_GEMM_PRECISION or 'tf32') selects tensor-core use.
        Accumulation order differs from the sequential v1 add_ chain: parity
        vs the naive reference is allclose, not bit-exact (gated in tests).
    """
    if not pair_seeds_and_scores:
        raise ValueError("pair_seeds_and_scores must not be empty")
    if noise_layout != ZORL_NOISE_LAYOUT_V2:
        raise ValueError(f"GEMM fold requires noise layout {ZORL_NOISE_LAYOUT_V2!r}; got {noise_layout!r}")
    precision = matmul_precision or os.environ.get("XORL_ZORL_FOLD_GEMM_PRECISION", "tf32").strip().lower()
    if precision not in ("tf32", "bf16", "fp32"):
        raise ValueError(f"Unknown fold GEMM precision: {precision!r}")

    modules = pair_zorl_fresh_ab_lora_params(lora_params)
    unknown = set(accumulators) - set(modules)
    if unknown:
        raise ValueError(f"Unknown fresh_ab module keys in accumulators: {sorted(unknown)}")
    selected = {module_key: modules[module_key] for module_key in sorted(accumulators)}

    needed_a = {a_name for a_name, _b in selected.values()}
    needed_b = {b_name for _a, b_name in selected.values()}
    fold_device = next(iter(accumulators.values())).device
    weight_scale = float(scaling) / float(len(pair_seeds_and_scores))

    pairs = [
        (a_seed, b_seed, float(z) * weight_scale) for a_seed, b_seed, z in pair_seeds_and_scores if float(z) != 0.0
    ]
    if not pairs:
        return
    if pair_block is None:
        env_block = int(os.environ.get("XORL_ZORL_FOLD_PAIR_BLOCK", "0") or 0)
        if env_block > 0:
            pair_block = env_block
        else:
            # Bound the stacked-factor residency: one pair's extracted a+b
            # noise for the selected modules, fp32.
            sorted_raw_a, assemble_a = _zorl_sglang_noise_layout(
                lora_params, marker="lora_A", logical_shapes=logical_shapes
            )
            sorted_raw_b, assemble_b = _zorl_sglang_noise_layout(
                lora_params, marker="lora_B", logical_shapes=logical_shapes
            )
            shapes = dict(sorted_raw_a) | dict(sorted_raw_b)
            needed_raws = {raw for p, raw, _e in list(assemble_a) + list(assemble_b) if p in needed_a | needed_b}
            per_pair_bytes = 4 * sum(math.prod(shapes[r]) for r in needed_raws)
            budget = int(os.environ.get("XORL_ZORL_FOLD_STACK_BYTES", str(4 << 30)))
            pair_block = max(1, budget // max(per_pair_bytes, 1))
    block = max(1, min(int(pair_block), len(pairs)))

    # Layout + assemble maps once; per pair-block ONE batched philox draw per
    # raw entry (keys = the block's sub-seeds) instead of per-pair chains —
    # the fold is otherwise kernel-launch-bound.
    def _needed_assembly(marker: str, wanted: set):
        sorted_raw, assemble = _zorl_sglang_noise_layout(lora_params, marker=marker, logical_shapes=logical_shapes)
        shapes = dict(sorted_raw)
        entries = [item for item in assemble if item[0] in wanted]
        raws = sorted({raw for _p, raw, _e in entries})
        return entries, [(raw, shapes[raw]) for raw in raws]

    a_entries, a_raws = _needed_assembly("lora_A", needed_a)
    b_entries, b_raws = _needed_assembly("lora_B", needed_b)
    rademacher_b = os.environ.get("XORL_ZORL_RADEMACHER_B", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    def _block_draws(seeds, raws, entries, sign_only):
        """Batched draws kept batched: {param: [B, *param_shape]}.

        The v1 extract closures are `...`-relative (slice dim -2, transpose
        last-two), so they apply to the whole [B, *raw_shape] tensor — no
        per-pair python extraction and no B-way cats downstream.
        """
        raw_batches = {}
        for raw_name, shape in raws:
            flat = zorl_philox_randn_batch(
                [zorl_param_subseed(s, raw_name) for s in seeds],
                math.prod(shape),
                device=fold_device,
            )
            if sign_only:
                flat = torch.sign(flat)
            raw_batches[raw_name] = flat.view(len(seeds), *shape)
        draws = {}
        for param_name, raw_name, extract in entries:
            tensor = extract(raw_batches[raw_name])
            if local_slices is not None and param_name in local_slices:
                tensor = tensor[(slice(None),) + local_slices[param_name]].contiguous()
            draws[param_name] = tensor
        return draws

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    try:
        for start in range(0, len(pairs), block):
            chunk_pairs = pairs[start : start + block]
            B = len(chunk_pairs)
            coef = torch.tensor([c for _a, _b, c in chunk_pairs], device=fold_device, dtype=torch.float32)
            a_draws = _block_draws([a for a, _b, _c in chunk_pairs], a_raws, a_entries, False)
            b_draws = _block_draws([b for _a, b, _c in chunk_pairs], b_raws, b_entries, rademacher_b)
            for module_key, (a_name, b_name) in selected.items():
                acc = accumulators[module_key]
                a_b = a_draws[a_name]  # [B, *a_shape]
                b_b = b_draws[b_name]  # [B, *b_shape]
                if a_b.ndim == 3:
                    # A [B, r, in], B [B, out, r]:
                    # G += (B*coef) [out, B*r] @ A [B*r, in]
                    r = a_b.shape[1]
                    a_cat = a_b.reshape(B * r, a_b.shape[-1])
                    b_cat = (b_b * coef.view(B, 1, 1)).permute(1, 0, 2).reshape(b_b.shape[1], B * r)
                    if precision == "bf16":
                        acc.add_((b_cat.bfloat16() @ a_cat.bfloat16()).float())
                    else:
                        acc.add_(b_cat @ a_cat)
                else:
                    # A [B, E|1, in, r], B [B, E|1, r, out]:
                    # G += matmul(A_cat [E|1, in, B*r], (B*coef)_cat [E|1, B*r, out])
                    r = a_b.shape[-1]
                    a_cat = a_b.permute(1, 2, 0, 3).reshape(a_b.shape[1], a_b.shape[2], B * r)
                    b_cat = (
                        (b_b * coef.view(B, 1, 1, 1)).permute(1, 0, 2, 3).reshape(b_b.shape[1], B * r, b_b.shape[-1])
                    )
                    if precision == "bf16":
                        acc.add_(torch.matmul(a_cat.bfloat16(), b_cat.bfloat16()).float())
                    else:
                        acc.add_(torch.matmul(a_cat, b_cat))
                del a_b, b_b, a_cat, b_cat
            del a_draws, b_draws
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
