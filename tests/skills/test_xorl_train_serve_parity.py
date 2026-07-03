from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL = _REPO_ROOT / "skills" / "xorl-train-serve-parity" / "SKILL.md"
_K3_SCRIPTS = _REPO_ROOT / "experiments" / "k3_tests"
_SRC = _REPO_ROOT / "src" / "xorl"


def test_parity_skill_documents_reconciliation_recipe():
    if not _SKILL.exists():  # pragma: no cover - skills/ outside src+tests merge scope
        pytest.skip("xorl-train-serve-parity SKILL.md absent (skills/ is outside the src+tests merge scope)")
    text = _SKILL.read_text(encoding="utf-8")

    # The behavior-vs-prefill distinction is the crux: behavior K3 (decoded tokens) is the
    # training quantity; prefill K3 is only a diagnostic.
    for expected in (
        "behavior K3",
        "decoded tokens",
        "--reference-logprobs generation",
        "prefill K3",
        # The reconciliation flag set.
        "--rl-on-policy-target xorl-batch-invariant",
        "--enable-fp32-lm-head",
        "--enable-fp32-router",
        "SGLANG_FLA_TRIL_PRECISION",
        "SGLANG_DISABLE_ROPE_COMPILE",
        "SGLANG_RMSNORM_FP32_WEIGHT_MUL",
        "enable_high_precision_for_bf16",
        # The MoE lever: decode-time routing replay.
        "K3_ROUTING_SOURCE=decode",
        "routed_experts",
        "decode routing",
        # The op-divergence chain + the irreducible floor.
        "XORL_BATCH_INVARIANT_MATMUL",
        "down_proj",
        "flash_attn_with_kvcache",
        "near-argmax-tie",
        # Scaling levers + measure-the-right-thing.
        "DeepGEMM",
        "deployed",
        # Cross-references.
        "STAGE_SUMMARY.md",
        "xorl-k3-correctness-check",
    ):
        assert expected in text, f"SKILL.md missing expected guidance: {expected!r}"


def test_parity_skill_references_existing_scripts():
    if not _SKILL.exists():  # pragma: no cover - skills/ outside src+tests merge scope
        pytest.skip("xorl-train-serve-parity SKILL.md absent (skills/ is outside the src+tests merge scope)")

    # Harness scripts the skill points at (experiments/ may be outside the merge scope).
    for script in (
        _K3_SCRIPTS / "make_static_traces.py",
        _K3_SCRIPTS / "compare_static_traces.py",
        _K3_SCRIPTS / "diagnose_static_k3.py",
        _K3_SCRIPTS / "launch_k3_test.py",
        _K3_SCRIPTS / "op_parity_dense.py",
        _K3_SCRIPTS / "attn_parity_dense.py",
        _K3_SCRIPTS / "residual_growth_dense.py",
        _K3_SCRIPTS / "rmsnorm_ordering_forward.py",
    ):
        if not script.exists():  # pragma: no cover - experiments/ outside src+tests merge scope
            pytest.skip(f"k3 harness script '{script.name}' absent (experiments/ is outside the merge scope)")

    # src/ is in scope: the diagnostic-dump host + the vendored batch-invariant levers.
    for src_file in (
        _SRC / "server" / "runner" / "model_runner.py",
        _SRC / "ops" / "batch_invariant_ops.py",
    ):
        assert src_file.exists(), f"expected src file referenced by the skill: {src_file}"
