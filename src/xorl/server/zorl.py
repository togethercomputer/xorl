"""ZORL config normalization, runtime planning, and deterministic noise helpers."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

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

    mode: Literal["all", "pair_shard", "pair_range"] = "all"
    shard_index: int = 0
    num_shards: int = 1
    pair_start: int = 0
    pair_end: Optional[int] = None

    def owns_pair(self, perturbation_index: int, *, num_pairs: int) -> bool:
        if self.mode == "all":
            return 0 <= int(perturbation_index) < int(num_pairs)
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

    def begin_generation(self, *, num_pairs: Optional[int] = None) -> ZORLGenerationPlan:
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
        num_pairs = int(self.config["num_perturbation_pairs"] if num_pairs is None else num_pairs)
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")
        antithetic_sampling = bool(self.config["antithetic_sampling"])
        perturbation_mode = str(self.config.get("perturbation_mode", DEFAULT_ZORL_PERTURBATION_MODE))

        for perturbation_index in range(num_pairs):
            perturb_seed = _mix_seed(self.config.get("seed"), family.family_index, generation, perturbation_index)
            # fresh_ab draws a fresh LoRA-A per pair from a decorrelated seed
            # stream; the b_seed stream is byte-identical to b_only mode.
            a_seed = (
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
    if mode not in {"all", "pair_shard", "pair_range"}:
        raise ValueError(f"Unsupported ZORL materialization mode {mode!r}")

    total_pairs = int(num_pairs)
    if total_pairs <= 0:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}")

    if mode == "all":
        return ZORLMaterialization()

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


def _iter_zorl_param_noises(
    lora_params: Dict[str, Any],
    *,
    seed: int,
    marker: str,
) -> List[tuple[str, torch.Tensor]]:
    """Generate deterministic Gaussian noise for every LoRA param matching ``marker``.

    One CPU generator seeded once, then sequential draws over the sorted
    parameter names — the single seed->noise contract shared by candidate
    materialization and the reward-weighted folds.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    noises: List[tuple[str, torch.Tensor]] = []
    for name in sorted(param_name for param_name in lora_params if marker in param_name):
        param = lora_params[name]
        shape = tuple(param.shape)
        noises.append((name, torch.randn(shape, generator=generator, dtype=torch.float32)))
    return noises


def iter_zorl_b_noises(
    lora_params: Dict[str, Any],
    *,
    seed: int,
) -> List[tuple[str, torch.Tensor]]:
    """Generate deterministic Gaussian noise tensors for every LoRA-B parameter."""

    return _iter_zorl_param_noises(lora_params, seed=seed, marker="lora_B")


def iter_zorl_fresh_ab_a_noises(
    lora_params: Dict[str, Any],
    *,
    seed: int,
) -> List[tuple[str, torch.Tensor]]:
    """Generate deterministic unit-Gaussian noise tensors for every LoRA-A parameter.

    fresh_ab candidates REPLACE the parent LoRA-A with this draw (unit scale,
    matching the sglang fresh_ab reference where A is a plain ``randn``).
    """

    return _iter_zorl_param_noises(lora_params, seed=seed, marker="lora_A")


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
            tensor = tensor + (sign * float(b_sigma) * perturbations[name])
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


def build_zorl_fresh_ab_base_update_from_rewards(
    lora_params: Dict[str, Any],
    *,
    pair_seeds_and_scores: List[tuple[int, int, float]],
    scaling: float,
) -> tuple[Dict[str, torch.Tensor], float]:
    """Build the reward-weighted full-shape fresh_ab base update.

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
    weight_scale = 1.0 / float(len(pair_seeds_and_scores))
    updates: Dict[str, torch.Tensor] = {}

    for a_seed, b_seed, normalized_score in pair_seeds_and_scores:
        coefficient = float(normalized_score) * weight_scale * float(scaling)
        if coefficient == 0.0:
            continue
        a_noises = dict(iter_zorl_fresh_ab_a_noises(lora_params, seed=a_seed))
        b_noises = dict(iter_zorl_b_noises(lora_params, seed=b_seed))
        for module_key, (a_name, b_name) in modules.items():
            delta = _zorl_fresh_ab_module_delta(a_noises[a_name], b_noises[b_name])
            accumulator = updates.get(module_key)
            if accumulator is None:
                updates[module_key] = delta.mul_(coefficient)
            else:
                accumulator.add_(delta, alpha=coefficient)

    # Zero-score generations still produce well-defined (zero) updates.
    for module_key, (a_name, b_name) in modules.items():
        if module_key not in updates:
            a_shape = tuple(lora_params[a_name].shape)
            b_shape = tuple(lora_params[b_name].shape)
            if len(a_shape) == 2:
                out_shape: tuple[int, ...] = (b_shape[0], a_shape[1])
            else:
                num_experts = max(a_shape[0], b_shape[0])
                out_shape = (num_experts, a_shape[1], b_shape[2])
            updates[module_key] = torch.zeros(out_shape, dtype=torch.float32)

    squared_norm = 0.0
    for tensor in updates.values():
        squared_norm += float(torch.sum(tensor * tensor).item())
    return updates, math.sqrt(max(squared_norm, 0.0))
