"""Contract-lane coverage for FlashQLA automatic context parallelism."""

import pytest
import torch

from xorl.ops.linear_attention.backend import FLASHQLA_AUTOCP_ENV, resolve_flashqla_auto_cp


@pytest.mark.cpu
def test_contract_lane_pins_auto_cp_off_unless_explicitly_overridden(monkeypatch):
    monkeypatch.delenv(FLASHQLA_AUTOCP_ENV, raising=False)
    monkeypatch.delenv("XORL_BI_GDN", raising=False)
    assert resolve_flashqla_auto_cp(None) is True

    monkeypatch.setenv("XORL_BI_GDN", "1")
    assert resolve_flashqla_auto_cp(None) is False
    assert resolve_flashqla_auto_cp(True) is True

    monkeypatch.setenv(FLASHQLA_AUTOCP_ENV, "1")
    assert resolve_flashqla_auto_cp(None) is True
    monkeypatch.setenv(FLASHQLA_AUTOCP_ENV, "0")
    assert resolve_flashqla_auto_cp(None) is False


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_contract_lane_skips_flashqla_auto_cp_preprocessing(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashQLA requires a Hopper (SM90) GPU")
    try:
        import xorl.ops.linear_attention.flashqla.ops.gated_delta_rule.chunk as chunk  # noqa: PLC0415
        from xorl.ops.linear_attention.flashqla import (  # noqa: PLC0415
            chunk_gated_delta_rule,
        )
    except Exception as exc:
        pytest.skip(f"FlashQLA backend unavailable: {exc}")

    calls = []
    original = chunk.intra_card_cp_preprocess

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chunk, "intra_card_cp_preprocess", spy)
    monkeypatch.setenv("XORL_BI_GDN", "1")
    monkeypatch.delenv(FLASHQLA_AUTOCP_ENV, raising=False)
    generator = torch.Generator(device="cuda").manual_seed(1)
    shape = (1, 4096, 4, 128)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(shape[:-1], generator=generator, device="cuda", dtype=torch.float32)
    beta = torch.rand(shape[:-1], generator=generator, device="cuda", dtype=torch.float32)

    chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta)
    assert not calls
    chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, auto_cp=True)
    assert calls == [1]
