import importlib

import pytest


pytestmark = [pytest.mark.cpu]


def test_import_xorl_qlora_module():
    qlora = importlib.import_module("xorl.qlora")

    assert hasattr(qlora, "QLoRALinear")
    assert hasattr(qlora, "inject_qlora_into_model")
