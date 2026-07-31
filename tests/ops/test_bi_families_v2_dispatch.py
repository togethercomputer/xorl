"""The v2 norm structure switch: which realization runs, and that it cannot matter.

Both realizations compute the same tree, so the switch is a speed choice — but a
speed choice that was pointed the wrong way sent every shipped hidden size at
prefill row counts to the slower realization. These gates pin the direction and
re-pin bit-neutrality, so the switch stays free to move on measurement.
"""

from pathlib import Path

import pytest
import torch


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SHIPPED_H = (2048, 3840, 4096)
DEEP_H = 16384


def test_benchmark_uses_installed_module_and_fixed_report_path():
    script = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "k3_tests"
        / "families_v2"
        / "bench_norm_structure_switch.py"
    ).read_text(encoding="utf-8")

    assert "spec_from_file_location" not in script
    assert 'add_argument("--v2"' not in script
    assert 'add_argument("--out"' not in script
    assert 'REPORT_PATH = Path(__file__).resolve().with_name("results")' in script


def _payload(rows, h):
    g = torch.Generator(device="cpu").manual_seed(rows * 7 + h)
    x = torch.randn((rows, h), generator=g, dtype=torch.float32).to(torch.bfloat16).cuda()
    w = torch.randn((h,), generator=g, dtype=torch.float32).to(torch.bfloat16).cuda()
    return x, w


def _split_spy(monkeypatch, v2):
    calls = []
    real = v2._rms_norm_v2_split

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(v2, "_rms_norm_v2_split", spy)
    return calls


@requires_cuda
def test_dispatch_keeps_shipped_hidden_sizes_on_the_fused_realization(monkeypatch):
    from xorl.ops import bi_families_v2 as v2

    calls = _split_spy(monkeypatch, v2)
    for h in SHIPPED_H:
        for m in (1, 64, 512, 2048):
            x, w = _payload(m, h)
            v2.rms_norm_v2(x, w, 1e-6, residual=torch.zeros_like(x))
    assert calls == [], f"split realization ran at shipped hidden sizes: {calls}"


@requires_cuda
def test_dispatch_reaches_the_split_realization_at_deep_tile_shapes(monkeypatch):
    from xorl.ops import bi_families_v2 as v2

    calls = _split_spy(monkeypatch, v2)
    x, w = _payload(8, DEEP_H)
    v2.rms_norm_v2(x, w, 1e-6, residual=torch.zeros_like(x))
    assert calls, "split realization is unreachable; the cross-structure gates test nothing"


@requires_cuda
@pytest.mark.parametrize("h", (3840, DEEP_H))
@pytest.mark.parametrize("rows", (1, 8, 512))
def test_split_and_fused_realizations_are_bitwise_identical(h, rows):
    """Structure is not bit-relevant — the premise the switch is allowed to move on."""
    from xorl.ops import bi_families_v2 as v2

    x, w = _payload(rows, h)
    r = torch.randn_like(x)
    zx = x[:, :128].contiguous()
    zw = w[:128].contiguous()

    def both(*args):
        split = v2._rms_norm_v2_split(*args)
        fused = getattr(v2, "_rms_norm_v2_fused", None)
        if fused is not None:
            return fused(*args), split
        # base tree has no named fused entry: force it through the row switch
        original = v2.V2_NORM_SPLIT_M
        try:
            v2.V2_NORM_SPLIT_M = 10**9
            x_, w_, eps_, res_, zc_ = args
            out = v2.rms_norm_v2(x_, w_, eps_, residual=res_, zero_centered=zc_)
        finally:
            v2.V2_NORM_SPLIT_M = original
        return out, split

    for args in ((x, w, 1e-6, r, False), (x, w, 1e-6, None, False), (zx, zw, 1e-6, None, True)):
        fused_out, split_out = both(*args)
        fused_out = fused_out if isinstance(fused_out, tuple) else (fused_out,)
        split_out = split_out if isinstance(split_out, tuple) else (split_out,)
        for a, b in zip(fused_out, split_out, strict=True):
            assert torch.equal(a, b)
