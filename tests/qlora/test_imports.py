import importlib
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.cpu]


def test_import_xorl_qlora_module():
    qlora = importlib.import_module("xorl.qlora")

    assert hasattr(qlora, "QLoRALinear")
    assert hasattr(qlora, "inject_qlora_into_model")


def test_qlora_utils_imports_in_a_clean_interpreter():
    """The expert contract must not pull the model package back into QLoRA."""

    result = subprocess.run(
        [sys.executable, "-c", "import xorl.qlora.utils; import xorl.qlora.modules.moe_experts"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
