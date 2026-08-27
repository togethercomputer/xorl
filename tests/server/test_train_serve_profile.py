"""Train/serve alignment profile: derivation, overrides, translation, rejection.

Covers the contract from xorl.server.train_serve_profile:
- deterministic trainer-field derivation per profile (pins + fills)
- explicit user overrides: fills always win, pins fail fast with every
  conflict reported
- backward compatibility: configs without a profile parse unchanged
- deterministic translation to SGLang launch flags
- receiver admission checks (/add_inference_endpoint) per profile
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

from xorl.server.train_serve_profile import (
    PROFILE_CHOICES,
    endpoint_profile_mismatches,
    expand_train_serve_profile,
    expected_server_info,
    resolve_profile,
    serving_lora_target_modules,
    sglang_launch_args,
)


pytestmark = [pytest.mark.cpu, pytest.mark.server]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_server_arguments_fn():
    """Import launcher.load_server_arguments with heavy deps stubbed (same
    trick as tests/server/test_server_arguments.py)."""
    module_path = _REPO_ROOT / "src" / "xorl" / "server" / "launcher.py"
    spec = importlib.util.spec_from_file_location("xorl_test_launcher_profile", module_path)
    assert spec is not None and spec.loader is not None

    fake_api_server_pkg = types.ModuleType("xorl.server.api_server")
    fake_api_server_pkg.__path__ = []
    fake_api_server_mod = types.ModuleType("xorl.server.api_server.server")
    fake_api_server_mod.APIServer = object
    fake_api_server_pkg.server = fake_api_server_mod
    fake_orchestrator_pkg = types.ModuleType("xorl.server.orchestrator")
    fake_orchestrator_pkg.__path__ = []
    fake_orchestrator_mod = types.ModuleType("xorl.server.orchestrator.orchestrator")
    fake_orchestrator_mod.Orchestrator = object
    fake_orchestrator_pkg.orchestrator = fake_orchestrator_mod
    fake_utils_pkg = types.ModuleType("xorl.server.utils")
    fake_utils_pkg.__path__ = []
    fake_network_mod = types.ModuleType("xorl.server.utils.network")
    fake_network_mod.read_address_file = lambda *args, **kwargs: None
    fake_utils_pkg.network = fake_network_mod
    fake_session_spec_mod = types.ModuleType("xorl.server.session_spec")
    fake_session_spec_mod.build_default_session_spec = lambda *args, **kwargs: None

    module = importlib.util.module_from_spec(spec)
    stubs = {
        "xorl.server.api_server": fake_api_server_pkg,
        "xorl.server.api_server.server": fake_api_server_mod,
        "xorl.server.orchestrator": fake_orchestrator_pkg,
        "xorl.server.orchestrator.orchestrator": fake_orchestrator_mod,
        "xorl.server.session_spec": fake_session_spec_mod,
        "xorl.server.utils": fake_utils_pkg,
        "xorl.server.utils.network": fake_network_mod,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module.load_server_arguments


class _FakeServerInfo:
    def __init__(self, quantization=None, enable_lora=None, max_lora_rank=None):
        self.quantization = quantization
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank


# ---------------------------------------------------------------------------
# Derivation (pins + fills) on the raw config mapping
# ---------------------------------------------------------------------------


def test_lora_profile_derives_trainer_fields():
    expanded = expand_train_serve_profile(
        {"train_serve_profile": "lora", "lora_rank": 32, "lora_target_modules": _SPLIT_TARGETS}
    )
    assert expanded["enable_lora"] is True
    assert expanded["enable_qlora"] is False
    assert expanded["unfuse_for_lora"] is True
    assert expanded["lora_alpha"] == 32
    assert expanded["enable_fp8_training"] is False
    assert expanded["enable_qarl"] is False
    # user-provided tunables are untouched
    assert expanded["lora_rank"] == 32
    assert expanded["lora_target_modules"] == _SPLIT_TARGETS


def test_fp8_lora_profile_derives_block_fp8_qlora():
    expanded = expand_train_serve_profile({"train_serve_profile": "fp8_lora", "lora_rank": 32})
    assert expanded["enable_lora"] is True
    assert expanded["enable_qlora"] is True
    assert expanded["quant_format"] == "block_fp8"
    assert expanded["quant_group_size"] == 128
    assert expanded["unfuse_for_lora"] is False
    assert expanded["lora_alpha"] == 32


def test_full_profile_pins_all_adapter_and_low_precision_modes_off():
    expanded = expand_train_serve_profile({"train_serve_profile": "full"})
    for field_name in (
        "enable_lora",
        "enable_qlora",
        "block_fp8_qlora_training",
        "unfuse_for_lora",
        "enable_fp8_training",
        "enable_qarl",
    ):
        assert expanded[field_name] is False


def test_explicit_fill_overrides_win_over_profile_defaults():
    expanded = expand_train_serve_profile({"train_serve_profile": "lora", "lora_alpha": 16, "unfuse_for_lora": False})
    assert expanded["lora_alpha"] == 16
    assert expanded["unfuse_for_lora"] is False
    # pins are still derived
    assert expanded["enable_lora"] is True


def test_pin_conflicts_are_rejected_with_every_mismatch_reported():
    with pytest.raises(ValueError) as excinfo:
        expand_train_serve_profile(
            {"train_serve_profile": "full", "enable_lora": True, "enable_qlora": True},
            context="unit test",
        )
    message = str(excinfo.value)
    assert "unit test" in message
    assert "enable_lora=True" in message
    assert "enable_qlora=True" in message


def test_matching_explicit_values_are_accepted():
    expanded = expand_train_serve_profile(
        {"train_serve_profile": "fp8_lora", "enable_qlora": True, "quant_format": "block_fp8"}
    )
    assert expanded["quant_group_size"] == 128


def test_absent_profile_leaves_config_unchanged():
    config = {"model_path": "Qwen/Qwen3-8B", "enable_lora": True, "lora_alpha": 16}
    assert expand_train_serve_profile(dict(config)) == config


def test_unknown_profile_is_rejected_with_choices():
    with pytest.raises(ValueError, match="full, lora, fp8_lora"):
        expand_train_serve_profile({"train_serve_profile": "int4_lora"})
    assert PROFILE_CHOICES == ("full", "lora", "fp8_lora")


def test_profile_name_is_normalized():
    assert resolve_profile(" LoRA ").name == "lora"
    assert resolve_profile(None) is None
    assert resolve_profile("none") is None


# ---------------------------------------------------------------------------
# End-to-end through load_server_arguments (YAML + CLI overrides)
# ---------------------------------------------------------------------------


def test_load_server_arguments_expands_profile_and_records_it(tmp_path):
    load_server_arguments = _load_server_arguments_fn()
    config_path = tmp_path / "lora.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "worker_bind_address": "tcp://127.0.0.1:5555",
                "train_serve_profile": "lora",
                "lora_rank": 32,
                "lora_target_modules": _SPLIT_TARGETS,
            }
        ),
        encoding="utf-8",
    )
    args = load_server_arguments(str(config_path))
    assert args.train_serve_profile == "lora"
    assert args.enable_lora is True
    assert args.unfuse_for_lora is True
    assert args.lora_alpha == 32
    assert args.max_lora_rank == 32  # defaulted to lora_rank in __post_init__
    config = args.to_config_dict()
    assert config["train"]["train_serve_profile"] == "lora"
    assert config["lora"]["enable_lora"] is True


def test_load_server_arguments_rejects_cli_override_conflicting_with_profile(tmp_path):
    load_server_arguments = _load_server_arguments_fn()
    config_path = tmp_path / "lora.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "worker_bind_address": "tcp://127.0.0.1:5555",
                "train_serve_profile": "lora",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="enable_qlora=True"):
        load_server_arguments(str(config_path), overrides={"enable_qlora": True})


def test_configs_without_profile_parse_unchanged(tmp_path):
    """Backward compatibility: the historical flat config surface is untouched."""
    load_server_arguments = _load_server_arguments_fn()
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_path": "Qwen/Qwen3-8B",
                "worker_bind_address": "tcp://127.0.0.1:5555",
                "enable_lora": True,
                "lora_rank": 32,
                "lora_alpha": 16,
                "lora_target_modules": _SPLIT_TARGETS,
            }
        ),
        encoding="utf-8",
    )
    args = load_server_arguments(str(config_path))
    assert args.train_serve_profile is None
    assert args.enable_lora is True
    assert args.lora_alpha == 16  # profile-less: dataclass semantics only
    assert args.unfuse_for_lora is False


def test_shipped_profile_examples_load_and_derive():
    load_server_arguments = _load_server_arguments_fn()
    examples = {
        "full": _REPO_ROOT / "examples/server/configs/profiles/qwen3_8b_full.yaml",
        "lora": _REPO_ROOT / "examples/server/configs/profiles/qwen3_8b_lora.yaml",
        "fp8_lora": _REPO_ROOT / "examples/server/configs/profiles/qwen3_8b_fp8_lora.yaml",
    }
    for name, path in examples.items():
        args = load_server_arguments(str(path))
        assert args.train_serve_profile == name, path
    full = load_server_arguments(str(examples["full"]))
    assert full.enable_lora is False and full.enable_qlora is False
    lora = load_server_arguments(str(examples["lora"]))
    assert lora.enable_lora is True and lora.unfuse_for_lora is True and lora.lora_alpha == 32
    fp8 = load_server_arguments(str(examples["fp8_lora"]))
    assert fp8.enable_qlora is True and fp8.quant_format == "block_fp8" and fp8.quant_group_size == 128
    assert fp8.model_path == "Qwen/Qwen3-8B-FP8"


def test_direct_construction_still_validates_profile_invariants():
    from xorl.server.server_arguments import ServerArguments

    with pytest.raises(ValueError, match="invariants violated"):
        ServerArguments(
            model_path="Qwen/Qwen3-8B",
            worker_bind_address="tcp://127.0.0.1:5555",
            train_serve_profile="full",
            enable_lora=True,
            lora_target_modules=_SPLIT_TARGETS,
        )


# ---------------------------------------------------------------------------
# Deterministic translation to SGLang launch flags
# ---------------------------------------------------------------------------


def test_full_profile_launch_args():
    assert sglang_launch_args("full", model_path="Qwen/Qwen3-8B", lora_config={}) == [
        "--model-path",
        "Qwen/Qwen3-8B",
        "--rl-on-policy-target",
        "xorl",
        "--enable-fp32-lm-head",
        "--tp-size",
        "1",
    ]


def test_lora_profile_launch_args_carry_rank_and_targets():
    args = sglang_launch_args(
        "lora",
        model_path="Qwen/Qwen3-8B",
        lora_config={"lora_rank": 32, "lora_target_modules": _SPLIT_TARGETS},
    )
    assert args == [
        "--model-path",
        "Qwen/Qwen3-8B",
        "--rl-on-policy-target",
        "xorl",
        "--enable-fp32-lm-head",
        "--tp-size",
        "1",
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--lora-target-modules",
        *_SPLIT_TARGETS,
    ]


def test_fp8_lora_profile_launch_args_carry_quantization():
    args = sglang_launch_args(
        "fp8_lora",
        model_path="Qwen/Qwen3-8B-FP8",
        lora_config={"lora_rank": 32, "max_lora_rank": 64, "lora_target_modules": ["qkv_proj", "o_proj"]},
        tp_size=2,
    )
    assert "--quantization" in args and args[args.index("--quantization") + 1] == "fp8"
    assert args[args.index("--tp-size") + 1] == "2"
    # explicit max_lora_rank wins over lora_rank
    assert args[args.index("--max-lora-rank") + 1] == "64"


def test_fp8_lora_profile_launch_args_never_request_exact_serving():
    # The exact value program is a bf16 contract; a receiver launched with
    # --rl-on-policy-target xorl on FP8 weights fails closed at boot. The
    # fp8_lora derivation must therefore target the stock FP8 serving path.
    args = sglang_launch_args("fp8_lora", model_path="Qwen/Qwen3-8B-FP8", lora_config={"lora_rank": 32})
    assert "--rl-on-policy-target" not in args
    assert "--enable-fp32-lm-head" not in args
    # ...while the bf16 profiles keep requesting exact serving.
    for exact_profile in ("full", "lora"):
        exact_args = sglang_launch_args(exact_profile, model_path="Qwen/Qwen3-8B", lora_config={"lora_rank": 32})
        assert "--rl-on-policy-target" in exact_args and "--enable-fp32-lm-head" in exact_args


def test_qwen_exact_contracts_disengage_under_qlora():
    # Trainer-side pairing of the same rule: the dense-Qwen3/Qwen3.5 exact
    # stamps must not engage for a QLoRA (quantized frozen base) run, exactly
    # as the GLM-5.2 stamp disengages under block_fp8_qlora_training.
    from xorl.models.auto import qwen_exact_contracts_engaged

    assert qwen_exact_contracts_engaged(server_training=True, enable_qlora=False)
    assert not qwen_exact_contracts_engaged(server_training=True, enable_qlora=True)
    assert not qwen_exact_contracts_engaged(server_training=False, enable_qlora=False)


def test_unsupported_trainer_targets_fall_back_to_all_sentinel():
    # Qwen3.5's GDN g_proj is not a serving-side LoRA module name.
    assert serving_lora_target_modules(["q_proj", "g_proj"]) == ["all"]
    assert serving_lora_target_modules(None) == ["all"]
    assert serving_lora_target_modules(_SPLIT_TARGETS) == _SPLIT_TARGETS


# ---------------------------------------------------------------------------
# Receiver admission checks
# ---------------------------------------------------------------------------


def test_expected_server_info_per_profile():
    assert expected_server_info(None, {}) == {}
    assert expected_server_info("full", {}) == {"quantization": None, "enable_lora": False}
    assert expected_server_info("fp8_lora", {"lora_rank": 32}) == {
        "quantization": "fp8",
        "enable_lora": True,
        "min_max_lora_rank": 32,
    }


def test_matching_receivers_are_admitted():
    assert endpoint_profile_mismatches("full", {}, _FakeServerInfo()) == []
    assert (
        endpoint_profile_mismatches("lora", {"lora_rank": 32}, _FakeServerInfo(enable_lora=True, max_lora_rank=32))
        == []
    )
    assert (
        endpoint_profile_mismatches(
            "fp8_lora",
            {"lora_rank": 32},
            _FakeServerInfo(quantization="fp8", enable_lora=True, max_lora_rank=64),
        )
        == []
    )


def test_mismatched_receivers_are_rejected():
    # full profile refuses an FP8 or LoRA-enabled receiver
    mismatches = endpoint_profile_mismatches("full", {}, _FakeServerInfo(quantization="fp8", enable_lora=True))
    assert len(mismatches) == 2

    # lora profile requires the LoRA pool and a big-enough rank ceiling
    assert endpoint_profile_mismatches("lora", {"lora_rank": 32}, _FakeServerInfo()) == [
        "receiver enable_lora=None but the 'lora' profile pushes LoRA adapters (launch SGLang with --enable-lora)"
    ]
    (rank_mismatch,) = endpoint_profile_mismatches(
        "lora", {"lora_rank": 32}, _FakeServerInfo(enable_lora=True, max_lora_rank=16)
    )
    assert "max_lora_rank=16 < trainer max_lora_rank=32" in rank_mismatch

    # fp8_lora requires an FP8-quantized base
    mismatches = endpoint_profile_mismatches(
        "fp8_lora", {"lora_rank": 32}, _FakeServerInfo(enable_lora=True, max_lora_rank=32)
    )
    assert len(mismatches) == 1 and "requires an FP8-quantized base" in mismatches[0]

    # a lora profile against an FP8 receiver is a base-precision mismatch
    mismatches = endpoint_profile_mismatches(
        "lora", {"lora_rank": 32}, _FakeServerInfo(quantization="fp8", enable_lora=True, max_lora_rank=32)
    )
    assert len(mismatches) == 1 and "unquantized bf16 base" in mismatches[0]


def test_missing_server_info_is_a_mismatch_when_profile_is_set():
    (mismatch,) = endpoint_profile_mismatches("lora", {"lora_rank": 32}, None)
    assert "/server_info" in mismatch
    # without a profile, missing server_info stays acceptable
    assert endpoint_profile_mismatches(None, {}, None) == []
