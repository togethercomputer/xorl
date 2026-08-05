import importlib

import torch

from xorl.ops.linear_attention.modules.bi_contract import gdn_contract


class _FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))

        return launch


def test_bi_contract_pins_serving_kkt_reduction_geometry(monkeypatch):
    module = importlib.import_module("xorl.ops.linear_attention.ops.common.chunk_scaled_dot_kkt")
    contract_kernel = _FakeKernel()
    autotuned_kernel = _FakeKernel()
    monkeypatch.setattr(module, "_chunk_scaled_dot_kkt_fwd_kernel", contract_kernel)
    monkeypatch.setattr(module, "chunk_scaled_dot_kkt_fwd_kernel", autotuned_kernel)

    k = torch.empty(1, 64, 32, 128)
    g = torch.empty(1, 64, 32)
    beta = torch.empty(1, 64, 32)

    with gdn_contract(True):
        module.chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta)
    assert not autotuned_kernel.calls
    _, kwargs = contract_kernel.calls.pop()
    assert kwargs["BK"] == 64
    assert kwargs["num_warps"] == 8
    assert kwargs["num_stages"] == 3
    assert kwargs["IS_VARLEN"] is False
    assert kwargs["USE_G"] is True
    assert kwargs["SAFE_EXP"] is True

    with gdn_contract(False):
        module.chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta)
    assert not contract_kernel.calls
    _, kwargs = autotuned_kernel.calls.pop()
    assert "BK" not in kwargs
    assert "num_warps" not in kwargs
    assert "num_stages" not in kwargs
