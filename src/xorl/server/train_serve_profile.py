"""Train/serve alignment profiles: one normalized mode selection for trainer + receiver.

A profile names the training/serving combination once (``full``, ``lora``,
``fp8_lora``) and makes the two sides agree by construction:

- **Trainer side** — ``expand_train_serve_profile`` runs where the raw config
  mapping is still visible (``load_server_arguments`` / ``parse_server_args``),
  so "user set this key" is distinguishable from "dataclass default". Pinned
  fields are filled in when absent and rejected when explicitly contradicted
  (all mismatches reported at once, following the ``block_fp8_qlora_training``
  requirements/mismatches precedent). Overridable defaults are only filled.
- **Receiver side** — XoRL does not launch SGLang; the receiver is registered
  via ``/add_inference_endpoint`` and described by its ``/server_info``.
  ``endpoint_profile_mismatches`` turns the profile into fail-fast admission
  checks (quantization, LoRA enablement, max LoRA rank), and
  ``sglang_launch_args`` is the deterministic translation from the same
  profile to the SGLang launch flags (printable via
  ``python -m xorl.server.train_serve_profile <config.yaml>``).

Precision vocabulary (kept deliberately explicit): ``fp8_lora`` means an **FP8
base model** (the trainer holds the frozen base as block-FP8 QLoRA, the
receiver serves an FP8-quantized base) with **bf16 LoRA adapter weights** on
both sides. It does not quantize the adapters themselves; no profile does.

Serving numerics: ``full`` and ``lora`` pair with XoRL exact serving
(``--rl-on-policy-target xorl``, the bf16 K3=0 contract). ``fp8_lora`` pairs
with the stock SGLang FP8 serving path -- the exact value program hard-rejects
quantized weights at receiver boot, and the trainer-side family exact
contracts likewise disengage under QLoRA (see
``xorl.models.auto.qwen_exact_contracts_engaged``); FP8+LoRA therefore has no
bitwise trainer/serving parity guarantee.

This module is imported by the config parsers and the API server; keep it
torch-free.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROFILE_FIELD = "train_serve_profile"

# Serving-side module names SGLang accepts for --lora-target-modules
# (sglang.srt.utils.common.SUPPORTED_LORA_TARGET_MODULES). Trainer targets
# outside this set (e.g. Qwen3.5's GDN g_proj) fall back to the "all"
# sentinel, which is a superset: adapters still carry the concrete set.
_SERVING_LORA_TARGET_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "q_b_proj",
        "kv_b_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "qkv_proj",
        "gate_up_proj",
        "embed_tokens",
        "lm_head",
    }
)

# Receiver quantization values that count as an FP8-quantized base model:
# either --quantization fp8 on the SGLang command line (reported verbatim in
# /server_info) or an HF checkpoint whose quantization_config.quant_method is
# "fp8" (sniffed by _detect_quantization_from_hf_config).
_FP8_BASE_QUANTIZATION_VALUES = frozenset({"fp8", "w8a8_fp8", "modelopt_fp8"})


@dataclass(frozen=True)
class TrainServeProfile:
    """One normalized train/serve mode and its derivation tables."""

    name: str
    description: str
    # What the frozen/trained base weights are on both sides.
    base_weight_format: str  # "bf16" | "fp8_block"
    # Whether LoRA adapters exist, and their weight format when they do.
    # Adapter weights are always bf16; this field exists to keep the
    # base-quantization vs adapter-quantization distinction explicit.
    adapter_enabled: bool
    adapter_weight_format: Optional[str]  # "bf16" | None
    # Expected receiver base quantization (None = unquantized bf16 serving).
    serving_quantization: Optional[str]
    # Whether the paired receiver runs XoRL exact serving (--rl-on-policy-target
    # xorl --enable-fp32-lm-head). The exact value program is a bf16 contract:
    # it hard-rejects quantized weights at receiver boot, so FP8-base profiles
    # pair with the stock serving path instead (no K3=0 guarantee).
    exact_serving: bool
    # Trainer fields the profile pins: filled when absent, a hard error when
    # explicitly set to anything else.
    pins: Mapping[str, Any]
    # Trainer fields the profile only fills when absent (explicit user values
    # always win, even when they diverge from the aligned default).
    fills: Mapping[str, Any]


def _profile(name, description, base, adapter, serving_quant, pins, fills, exact_serving):
    return TrainServeProfile(
        name=name,
        description=description,
        base_weight_format=base,
        adapter_enabled=adapter,
        adapter_weight_format="bf16" if adapter else None,
        serving_quantization=serving_quant,
        exact_serving=exact_serving,
        pins=MappingProxyType(pins),
        fills=MappingProxyType(fills),
    )


TRAIN_SERVE_PROFILES: Dict[str, TrainServeProfile] = {
    "full": _profile(
        "full",
        "bf16 full-weight training paired with an unquantized bf16 receiver",
        base="bf16",
        adapter=False,
        serving_quant=None,
        pins={
            "enable_lora": False,
            "enable_qlora": False,
            "block_fp8_qlora_training": False,
            "unfuse_for_lora": False,
            "enable_fp8_training": False,
            "enable_qarl": False,
        },
        fills={},
        exact_serving=True,
    ),
    "lora": _profile(
        "lora",
        "bf16 base + bf16 LoRA adapters, served on an unquantized bf16 receiver with the LoRA pool enabled",
        base="bf16",
        adapter=True,
        serving_quant=None,
        pins={
            "enable_lora": True,
            "enable_qlora": False,
            "block_fp8_qlora_training": False,
            "enable_fp8_training": False,
            "enable_qarl": False,
        },
        fills={
            # Serving applies adapters to the split projections; train the
            # same shapes by default. Explicit unfuse_for_lora: false keeps
            # fused-base logical LoRA for architectures that support it.
            "unfuse_for_lora": True,
            # Every shipped LoRA recipe uses alpha == rank == 32; the
            # dataclass default (16) predates them.
            "lora_alpha": 32,
        },
        exact_serving=True,
    ),
    "fp8_lora": _profile(
        "fp8_lora",
        "FP8 (block e4m3) frozen base + bf16 LoRA adapters (QLoRA), served on "
        "an FP8-quantized receiver with the LoRA pool enabled",
        base="fp8_block",
        adapter=True,
        serving_quant="fp8",
        pins={
            "enable_lora": True,
            "enable_qlora": True,
            "quant_format": "block_fp8",
            # unfuse_for_lora is a bf16-base concept and is rejected with
            # QLoRA by ServerArguments; pin it off so the contradiction is
            # reported as a profile mismatch instead.
            "unfuse_for_lora": False,
            "enable_fp8_training": False,
            "enable_qarl": False,
        },
        fills={
            # block_fp8 quantizes in 128-wide groups; the dataclass default
            # (16) belongs to nvfp4.
            "quant_group_size": 128,
            "lora_alpha": 32,
        },
        # FP8 block-quantized weights cannot satisfy the bf16 exact serving
        # contract (the receiver fails closed at boot); this profile pairs
        # with the stock SGLang FP8 serving path.
        exact_serving=False,
    ),
}

PROFILE_CHOICES = tuple(TRAIN_SERVE_PROFILES)


def resolve_profile(name: Any) -> Optional[TrainServeProfile]:
    """Return the profile for ``name``, None for unset, ValueError for unknown."""
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized not in TRAIN_SERVE_PROFILES:
        raise ValueError(f"Unknown {PROFILE_FIELD} {name!r}. Expected one of: {', '.join(PROFILE_CHOICES)}.")
    return TRAIN_SERVE_PROFILES[normalized]


def expand_train_serve_profile(config: Dict[str, Any], *, context: str = "server config") -> Dict[str, Any]:
    """Apply the selected profile to a raw config mapping.

    Must run where explicit keys are still distinguishable from dataclass
    defaults (i.e. on the merged YAML+CLI mapping, before ServerArguments is
    constructed). Returns a new mapping; the input is not modified.

    Pinned fields are filled when absent and rejected when explicitly set to
    a conflicting value; all conflicts are reported in one error. Fill-only
    fields never error.
    """
    profile = resolve_profile(config.get(PROFILE_FIELD))
    if profile is None:
        return dict(config)

    expanded = dict(config)
    expanded[PROFILE_FIELD] = profile.name

    mismatches = []
    for field_name, required in profile.pins.items():
        if field_name in expanded and expanded[field_name] is not None:
            if expanded[field_name] != required:
                mismatches.append(f"{field_name}={expanded[field_name]!r} (profile requires {required!r})")
        else:
            expanded[field_name] = required
    if mismatches:
        raise ValueError(
            f"{context}: {PROFILE_FIELD}={profile.name!r} conflicts with explicitly "
            f"configured fields: {'; '.join(mismatches)}. Remove the conflicting keys "
            "to derive them from the profile, or drop the profile to configure them freely."
        )

    for field_name, default in profile.fills.items():
        if field_name not in expanded or expanded[field_name] is None:
            expanded[field_name] = default

    return expanded


def validate_train_serve_profile_invariants(args: Any) -> None:
    """Defense-in-depth for directly constructed ServerArguments.

    ``expand_train_serve_profile`` already errors on explicit conflicts with
    full mismatch context; this re-checks the pinned invariants on the final
    field values so a ServerArguments built without the parse-layer expansion
    cannot silently claim a profile it does not satisfy.
    """
    profile = resolve_profile(getattr(args, PROFILE_FIELD, None))
    if profile is None:
        return
    violations = [
        f"{field_name}={getattr(args, field_name)!r} (profile requires {required!r})"
        for field_name, required in profile.pins.items()
        if getattr(args, field_name) != required
    ]
    if violations:
        raise ValueError(f"{PROFILE_FIELD}={profile.name!r} invariants violated: {'; '.join(violations)}")


def _resolved_max_lora_rank(lora_config: Mapping[str, Any]) -> Optional[int]:
    max_rank = lora_config.get("max_lora_rank")
    if max_rank is None:
        max_rank = lora_config.get("lora_rank")
    return int(max_rank) if max_rank is not None else None


def expected_server_info(profile_name: Any, lora_config: Mapping[str, Any]) -> Dict[str, Any]:
    """The receiver properties a profile requires, as one inspectable dict."""
    profile = resolve_profile(profile_name)
    if profile is None:
        return {}
    expectations: Dict[str, Any] = {
        "quantization": profile.serving_quantization,
        "enable_lora": profile.adapter_enabled,
    }
    if profile.adapter_enabled:
        expectations["min_max_lora_rank"] = _resolved_max_lora_rank(lora_config)
    return expectations


def _is_fp8_base_quantization(value: Any) -> bool:
    return value is not None and str(value).strip().lower() in _FP8_BASE_QUANTIZATION_VALUES


def endpoint_profile_mismatches(
    profile_name: Any,
    lora_config: Mapping[str, Any],
    server_info: Any,
) -> List[str]:
    """Fail-fast admission checks for one registered SGLang receiver.

    ``server_info`` is an InferenceEndpointServerInfo-shaped object (attribute
    access) or None when ``/server_info`` was unavailable. Returns human-readable
    mismatch strings; empty means the receiver satisfies the profile.
    """
    profile = resolve_profile(profile_name)
    if profile is None:
        return []
    if server_info is None:
        return [
            f"{PROFILE_FIELD}={profile.name!r} requires /server_info to verify the "
            "receiver configuration, but the endpoint did not provide it"
        ]

    mismatches = []

    quantization = getattr(server_info, "quantization", None)
    if profile.serving_quantization == "fp8":
        if not _is_fp8_base_quantization(quantization):
            mismatches.append(
                f"receiver quantization={quantization!r} but the {profile.name!r} profile "
                "requires an FP8-quantized base (launch SGLang with --quantization fp8 "
                "or serve an FP8 checkpoint)"
            )
    elif _is_fp8_base_quantization(quantization):
        mismatches.append(
            f"receiver quantization={quantization!r} but the {profile.name!r} profile serves an unquantized bf16 base"
        )

    receiver_lora = getattr(server_info, "enable_lora", None)
    if profile.adapter_enabled:
        if not receiver_lora:
            mismatches.append(
                f"receiver enable_lora={receiver_lora!r} but the {profile.name!r} profile "
                "pushes LoRA adapters (launch SGLang with --enable-lora)"
            )
        else:
            required_rank = _resolved_max_lora_rank(lora_config)
            receiver_rank = getattr(server_info, "max_lora_rank", None)
            if required_rank is not None:
                if receiver_rank is None:
                    mismatches.append(
                        f"receiver did not report max_lora_rank; the {profile.name!r} profile "
                        f"requires max_lora_rank >= {required_rank}"
                    )
                elif int(receiver_rank) < required_rank:
                    mismatches.append(f"receiver max_lora_rank={receiver_rank} < trainer max_lora_rank={required_rank}")
    elif receiver_lora:
        mismatches.append(
            f"receiver enable_lora={receiver_lora!r} but the {profile.name!r} profile "
            "trains full weights and never pushes adapters"
        )

    return mismatches


def serving_lora_target_modules(trainer_targets: Optional[Sequence[str]]) -> List[str]:
    """Translate trainer LoRA targets to SGLang's --lora-target-modules values.

    Trainer targets SGLang does not accept by name (e.g. Qwen3.5 GDN g_proj)
    force the "all" sentinel — a strict superset; the pushed adapters still
    carry the concrete module set.
    """
    if not trainer_targets:
        return ["all"]
    if all(target in _SERVING_LORA_TARGET_MODULES for target in trainer_targets):
        return list(trainer_targets)
    return ["all"]


def sglang_launch_args(
    profile_name: Any,
    *,
    model_path: str,
    lora_config: Mapping[str, Any],
    tp_size: int = 1,
) -> List[str]:
    """Deterministic translation of a profile to SGLang launch flags.

    Only alignment-relevant flags are emitted; capacity tuning
    (--mem-fraction-static, ports, CUDA-graph limits) stays with the operator.
    """
    profile = resolve_profile(profile_name)
    if profile is None:
        raise ValueError(f"sglang_launch_args requires a {PROFILE_FIELD}; got {profile_name!r}")

    args = ["--model-path", model_path]
    if profile.exact_serving:
        args += ["--rl-on-policy-target", "xorl", "--enable-fp32-lm-head"]
    args += ["--tp-size", str(tp_size)]
    if profile.serving_quantization is not None:
        args += ["--quantization", profile.serving_quantization]
    if profile.adapter_enabled:
        max_rank = _resolved_max_lora_rank(lora_config)
        args += ["--enable-lora"]
        if max_rank is not None:
            args += ["--max-lora-rank", str(max_rank)]
        args += ["--lora-target-modules", *serving_lora_target_modules(lora_config.get("lora_target_modules"))]
    return args


def _flatten_config_mapping(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested (model/train/lora sections) config to flat server keys."""
    if not ("model" in config and isinstance(config["model"], dict)):
        return dict(config)
    flat: Dict[str, Any] = {}
    for section in ("model", "train", "lora"):
        section_mapping = config.get(section)
        if isinstance(section_mapping, dict):
            flat.update(section_mapping)
    return flat


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the SGLang launch command derived from a server config's profile."""
    import argparse
    import shlex

    import yaml

    parser = argparse.ArgumentParser(
        prog="python -m xorl.server.train_serve_profile",
        description="Derive the paired SGLang launch flags from an XoRL server config's train_serve_profile.",
    )
    parser.add_argument("config", help="XoRL server config YAML (flat or nested)")
    parser.add_argument("--tp-size", type=int, default=1, help="Receiver tensor-parallel size")
    options = parser.parse_args(argv)

    with open(options.config) as handle:
        raw = yaml.safe_load(handle) or {}
    config = expand_train_serve_profile(_flatten_config_mapping(raw), context=f"server config {options.config!r}")

    profile_name = config.get(PROFILE_FIELD)
    if resolve_profile(profile_name) is None:
        parser.error(f"{options.config} does not set {PROFILE_FIELD}; nothing to derive")
    model_path = config.get("model_path")
    if not model_path:
        parser.error(f"{options.config} does not set model_path")

    args = sglang_launch_args(profile_name, model_path=model_path, lora_config=config, tp_size=options.tp_size)
    print(shlex.join(["python", "-m", "sglang.launch_server", *args]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
