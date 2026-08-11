"""Tests for QLoRA modules: quantization, EMA amax, scale convention, re-quantization."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.ops.quantize import block_fp8_quantize_gkn as block_fp8_weight_quant
from xorl.ops.quantize import nvfp4_dequantize, nvfp4_quantize
from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.utils import inject_qlora_into_model, maybe_requant_qlora
from xorl.trainers.training_utils import maybe_merge_lora, reset_lora_optimizer_states


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model():
    """Simple model with named linear layers for testing."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(256, 512, bias=False)
            self.up_proj = nn.Linear(256, 512, bias=False)
            self.down_proj = nn.Linear(512, 256, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 256, bias=False)
            self.k_proj = nn.Linear(256, 256, bias=False)
            self.v_proj = nn.Linear(256, 256, bias=False)
            self.o_proj = nn.Linear(256, 256, bias=False)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            attn = torch.bmm(
                q.view(-1, q.size(-2), q.size(-1)),
                k.view(-1, k.size(-2), k.size(-1)).transpose(-1, -2),
            )
            out = torch.bmm(attn, v.view(-1, v.size(-2), v.size(-1)))
            return self.o_proj(out.view_as(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = MLP()

        def forward(self, x):
            return x + self.mlp(self.self_attn(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return Model().cuda().to(torch.bfloat16)


def _quantize_injected_model(model):
    """After inject_qlora_into_model, quantize the uninitialized packed weights.

    inject_qlora_into_model creates QLoRA modules with empty packed_weight_f32.
    This helper fills them with properly quantized random weights for testing.
    """

    for m in model.modules():
        if isinstance(m, QLoRALinear) and m._is_prequantized:
            w = torch.randn(m.out_features, m.in_features, device="cuda", dtype=torch.bfloat16)
            m._quantize_and_store(w)


def _make_fp8_data(out_features, in_features, block_size=128):
    """Create mock FP8 data matching HF block FP8 format."""
    w = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    fp8_w, scales = block_fp8_weight_quant(w.float(), block_size)
    return w, fp8_w, scales


# ---------------------------------------------------------------------------
# 1. Quantize bf16 -> packed_f32: forward, backward, memory, dequant (both formats)
# ---------------------------------------------------------------------------


def _assert_quantize_forward_backward_memory():
    """bf16 quantization, forward, backward (only LoRA gets grad), memory savings."""
    for quant_format, group_size in (("nvfp4", 16), ("block_fp8", 128), ("nf4", 64)):
        linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
        bf16_bytes = linear.weight.numel() * 2
        qlora = QLoRALinear.from_module(
            linear,
            r=16,
            lora_alpha=16,
            quant_format=quant_format,
            quant_group_size=group_size,
        )

        # Quantized storage
        assert qlora.packed_weight_f32 is not None
        assert qlora.packed_weight_f32.dtype == torch.float32
        assert not qlora.packed_weight_f32.requires_grad
        assert qlora.lora_A.requires_grad and qlora.lora_B.requires_grad

        # Forward and backward: only LoRA gets grad.
        x = torch.randn(2, 10, 256, device="cuda", dtype=torch.bfloat16)
        out = qlora(x)
        assert out.shape == (2, 10, 512)
        out.sum().backward()
        assert qlora.lora_A.grad is not None and qlora.lora_B.grad is not None

        quant_bytes = qlora.packed_weight_f32.numel() * 4
        for scale_name in ("weight_block_scales", "weight_global_scale", "weight_scales"):
            if (scale := getattr(qlora, scale_name, None)) is not None:
                quant_bytes += scale.numel() * scale.element_size()
        assert quant_bytes < bf16_bytes


def _assert_dequantize_roundtrip():
    """Dequantized weight should be close to original."""
    for quant_format, group_size in (("nvfp4", 16), ("nf4", 64)):
        linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
        w_orig = linear.weight.detach().clone()
        qlora = QLoRALinear.from_module(
            linear,
            r=16,
            lora_alpha=16,
            quant_format=quant_format,
            quant_group_size=group_size,
        )
        w_deq = qlora._dequantize_weight().float()
        if quant_format == "nvfp4":
            torch.testing.assert_close(w_orig, w_deq.to(torch.bfloat16), atol=0.05, rtol=0.05)
        else:
            relative_error = (w_orig.float() - w_deq).abs().mean() / w_orig.float().abs().mean()
            assert relative_error < 0.10


# ---------------------------------------------------------------------------
# 2. Pre-quantized nvfp4 loading
# ---------------------------------------------------------------------------


def _assert_prequantized_nvfp4_loading():
    """from_quantized() loads pre-packed nvfp4 weights; forward+backward work."""
    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    packed, block_scales, global_scale = nvfp4_quantize(w, 16)

    qlora = QLoRALinear.from_quantized(
        packed_weight=packed,
        weight_block_scales=block_scales,
        weight_global_scale=global_scale,
        in_features=256,
        out_features=512,
        quant_format="nvfp4",
        quant_group_size=16,
        device="cuda",
    )
    assert qlora.packed_weight_f32.dtype == torch.float32

    x = torch.randn(2, 10, 256, device="cuda", dtype=torch.bfloat16)
    out = qlora(x)
    assert out.shape == (2, 10, 512)
    out.sum().backward()
    assert qlora.lora_A.grad is not None and qlora.lora_B.grad is not None


# ---------------------------------------------------------------------------
# 3. EMA amax + NVFP4 scale convention
# ---------------------------------------------------------------------------


def _assert_ema_amax_and_scale_convention():
    """EMA amax: init from bf16, update on merge, global_scale formula.
    Scale convention: block_scales use full fp8 range, dequant roundtrip accuracy."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    expected_amax = linear.weight.float().abs().max().item()

    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16, quant_format="nvfp4", quant_group_size=16)

    # Init from bf16
    assert qlora._ema_amax is not None and qlora._ema_amax.shape == (1,)
    assert abs(qlora._ema_amax.item() - expected_amax) < 1e-4

    # EMA update on merge
    amax_before = qlora._ema_amax.item()
    with torch.no_grad():
        qlora.lora_A.fill_(0.5)
        qlora.lora_B.fill_(0.5)
    qlora.merge_weights(ema_decay=0.5)
    assert qlora._ema_amax.item() != amax_before

    # Global scale reflects EMA
    gs = qlora._recover_tensor(qlora.weight_global_scale, qlora._scale_dtypes["weight_global_scale"]).item()
    expected_gs = qlora._ema_amax.item() / (FP4_E2M1_MAX * FP8_E4M3_MAX)
    assert abs(gs - expected_gs) / max(abs(expected_gs), 1e-12) < 0.01

    # Scale convention: block_scales use full fp8 range
    w2 = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 2.0
    packed, block_scales, global_scale = nvfp4_quantize(w2, 16)
    assert block_scales.float().max().item() > 1.0  # full fp8 range, not [0,1]

    # Dequant roundtrip accuracy
    w_deq = nvfp4_dequantize(packed, block_scales, global_scale, 512 * 256, 16).reshape(512, 256)
    rel_err = (w2.float() - w_deq.float()).abs().mean() / w2.float().abs().mean()
    assert rel_err < 0.15

    # Global scale formula: amax / (FP4_MAX * FP8_MAX)
    w3 = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 3.0
    _, _, gs3 = nvfp4_quantize(w3, 16)
    recovered_amax = gs3.item() * FP4_E2M1_MAX * FP8_E4M3_MAX
    assert recovered_amax >= w3.float().abs().max().item() * 0.99


# ---------------------------------------------------------------------------
# 4. Merge weights + maybe_requant
# ---------------------------------------------------------------------------


def _assert_merge_weights_and_requant():
    """merge_weights folds LoRA into base; maybe_requant_qlora merges+resets+EMA updates."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16, quant_format="nvfp4", quant_group_size=16)

    # Merge weights
    with torch.no_grad():
        qlora.lora_A.fill_(0.1)
        qlora.lora_B.fill_(0.1)
    w_before = qlora._dequantize_weight().clone()
    qlora.merge_weights()
    w_after = qlora._dequantize_weight()
    assert (w_after - w_before).float().abs().mean() > 0.001
    assert not torch.all(qlora.lora_A == 0)  # re-initialized (kaiming)
    assert torch.all(qlora.lora_B == 0)

    # Requant incorporates LoRA delta
    linear2 = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora2 = QLoRALinear.from_module(linear2, r=16, lora_alpha=16, quant_format="nvfp4", quant_group_size=16)
    with torch.no_grad():
        qlora2.lora_A.fill_(0.05)
        qlora2.lora_B.fill_(0.05)
    w2_before = qlora2._dequantize_weight().clone()
    delta = qlora2.get_delta_weight().to(w2_before.dtype)
    expected = w2_before + delta
    qlora2.merge_weights()
    diff = (qlora2._dequantize_weight() - expected).float().abs().mean()
    assert diff < 0.1

    # State dict
    sd = qlora2.get_quantized_state_dict()
    assert all(k in sd for k in ["packed_weight_f32", "weight_block_scales", "weight_global_scale"])

    # maybe_requant_qlora
    linear3 = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora3 = QLoRALinear.from_module(linear3, r=16, lora_alpha=16, quant_format="nvfp4", quant_group_size=16)
    amax_before = qlora3._ema_amax.item()
    with torch.no_grad():
        qlora3.lora_A.fill_(1.0)
        qlora3.lora_B.fill_(1.0)
    w3_before = qlora3._dequantize_weight().clone()
    model = nn.ModuleList([qlora3])
    count = maybe_requant_qlora(model, ema_decay=0.5)
    assert count == 1
    assert not torch.equal(w3_before, qlora3._dequantize_weight())
    assert qlora3._ema_amax.item() != amax_before
    assert torch.all(qlora3.lora_B == 0)


# ---------------------------------------------------------------------------
# 5. Injection + training
# ---------------------------------------------------------------------------


def _assert_injection_and_training():
    """inject_qlora replaces only target modules and preserves gradient flow."""
    model = _make_model()
    inject_qlora_into_model(
        model,
        r=16,
        lora_alpha=16,
        quant_format="nvfp4",
        quant_group_size=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    _quantize_injected_model(model)
    layer = model.layers[0]
    assert isinstance(layer.self_attn.q_proj, QLoRALinear)
    assert not isinstance(layer.mlp.gate_proj, QLoRALinear)

    # Forward/backward
    x = torch.randn(2, 10, 256, device="cuda", dtype=torch.bfloat16)
    model(x).sum().backward()
    for name, m in model.named_modules():
        if isinstance(m, QLoRALinear):
            assert m.lora_A.grad is not None, f"{name}.lora_A has no gradient"


def _assert_inject_with_checkpoint_quant_format():
    """inject_qlora with checkpoint_quant_format sets _source_quant_format."""
    model = _make_model()
    inject_qlora_into_model(
        model,
        r=16,
        lora_alpha=16,
        quant_format="block_fp8",
        quant_group_size=128,
        checkpoint_quant_format="block_fp8",
    )
    for m in model.modules():
        if isinstance(m, QLoRALinear):
            assert m._source_quant_format == "block_fp8"
            assert m.quant_format == "block_fp8"


# ---------------------------------------------------------------------------
# 8. Pre-quantized block FP8 loading (HF checkpoint path)
# ---------------------------------------------------------------------------


def _assert_prequantized_block_fp8_load_and_forward():
    """Load FP8 single module + merged qkv; forward/backward work; dequant roundtrip."""
    M, K = 256, 256
    w_orig, fp8_w, scales = _make_fp8_data(M, K)

    qlora = BlockFP8QLoRALinear(K, M, r=16, lora_alpha=16, device=torch.device("cuda"))
    qlora._is_prequantized = True
    qlora._source_quant_format = "block_fp8"
    qlora._merge_sources = None
    qlora._source_fqn = "model.layers.0.self_attn.o_proj"

    mock_data = {
        "model.layers.0.self_attn.o_proj.weight": fp8_w.cpu(),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": scales.cpu(),
    }
    qlora._load_prequantized(lambda key: mock_data[key])

    assert qlora.packed_weight_f32 is not None
    assert qlora._ema_amax is None

    # Dequant roundtrip
    w_deq = qlora._dequantize_weight().to(torch.bfloat16)
    rel_err = (w_orig.cuda() - w_deq).float().abs().mean() / w_orig.float().abs().mean()
    assert rel_err < 0.03

    # Forward + backward
    x = torch.randn(2, 10, K, device="cuda", dtype=torch.bfloat16)
    out = qlora(x)
    assert out.shape == (2, 10, M)
    out.sum().backward()
    assert qlora.lora_A.grad is not None

    # Merged QKV
    hidden, q_dim, kv_dim = 256, 256, 64
    _, fp8_q, s_q = _make_fp8_data(q_dim, hidden)
    _, fp8_k, s_k = _make_fp8_data(kv_dim, hidden)
    _, fp8_v, s_v = _make_fp8_data(kv_dim, hidden)

    total_out = q_dim + kv_dim + kv_dim
    qkv = BlockFP8QLoRALinear(hidden, total_out, r=16, lora_alpha=16, device=torch.device("cuda"))
    qkv._is_prequantized = True
    qkv._source_quant_format = "block_fp8"
    qkv._merge_sources = ("q_proj", "k_proj", "v_proj")
    qkv._source_fqn = "model.layers.0.self_attn"
    qkv_data = {
        "model.layers.0.self_attn.q_proj.weight": fp8_q.cpu(),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": s_q.cpu(),
        "model.layers.0.self_attn.k_proj.weight": fp8_k.cpu(),
        "model.layers.0.self_attn.k_proj.weight_scale_inv": s_k.cpu(),
        "model.layers.0.self_attn.v_proj.weight": fp8_v.cpu(),
        "model.layers.0.self_attn.v_proj.weight_scale_inv": s_v.cpu(),
    }
    qkv._load_prequantized(lambda key: qkv_data[key])
    assert qkv.packed_weight_f32.numel() * 4 == total_out * hidden


def _assert_prequantized_block_fp8_merge():
    """Merge weights after loading pre-quantized block FP8."""
    M, K = 256, 256
    _, fp8_w, scales = _make_fp8_data(M, K)

    qlora = BlockFP8QLoRALinear(K, M, r=16, lora_alpha=16, device=torch.device("cuda"))
    qlora._is_prequantized = True
    qlora._source_quant_format = "block_fp8"
    qlora._merge_sources = None
    qlora._source_fqn = "model.layers.0.self_attn.o_proj"
    mock_data = {
        "model.layers.0.self_attn.o_proj.weight": fp8_w.cpu(),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": scales.cpu(),
    }
    qlora._load_prequantized(lambda key: mock_data[key])

    # Merge
    with torch.no_grad():
        qlora.lora_A.fill_(0.1)
        qlora.lora_B.fill_(0.1)
    w_before = qlora._dequantize_weight().clone()
    qlora.merge_weights()
    assert (qlora._dequantize_weight() - w_before).float().abs().mean() > 0.001
    assert qlora._ema_amax is None  # block_fp8: no EMA amax


# ---------------------------------------------------------------------------
# 9. ReLoRA optimizer reset
# ---------------------------------------------------------------------------


def _assert_reset_lora_optimizer_states_clears():
    """Verify ReLoRA reset clears optimizer states for LoRA params."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16, quant_format="block_fp8", quant_group_size=128)
    qlora.train()
    opt = torch.optim.AdamW([qlora.lora_A, qlora.lora_B], lr=1e-3)

    # One step is sufficient to populate Adam states.
    x = torch.randn(4, 8, 256, device="cuda", dtype=torch.bfloat16)
    tgt = torch.randn(4, 8, 512, device="cuda", dtype=torch.bfloat16)
    opt.zero_grad()
    ((qlora(x) - tgt) ** 2).mean().backward()
    opt.step()

    # Verify optimizer states exist
    assert qlora.lora_A in opt.state
    assert qlora.lora_B in opt.state
    assert "exp_avg" in opt.state[qlora.lora_A]

    # Reset
    model = nn.ModuleList([qlora])
    count = reset_lora_optimizer_states(model, opt)
    assert count == 2  # lora_A and lora_B

    # States should be fully cleared
    assert qlora.lora_A not in opt.state
    assert qlora.lora_B not in opt.state

    # Optimizer should still work after reset (Adam re-creates states)
    opt.zero_grad()
    ((qlora(x) - tgt) ** 2).mean().backward()
    opt.step()
    # States rebuilt
    assert qlora.lora_A in opt.state
    assert qlora.lora_B in opt.state


def _assert_reset_ignores_non_lora_params():
    """Optimizer reset only touches LoRA params, not other trainable params."""
    model = _make_model()
    inject_qlora_into_model(
        model,
        r=16,
        lora_alpha=16,
        quant_format="block_fp8",
        quant_group_size=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    _quantize_injected_model(model)
    # MLP layers are nn.Linear (not QLoRA), so add them as trainable too
    all_params = list(model.parameters())
    opt = torch.optim.AdamW(all_params, lr=1e-3)

    x = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16)
    opt.zero_grad()
    model(x).sum().backward()
    opt.step()

    # Record non-LoRA param states
    non_lora_states_before = {}
    for name, p in model.named_parameters():
        if "lora_" not in name and p in opt.state:
            s = opt.state[p]
            if "exp_avg" in s:
                non_lora_states_before[name] = s["exp_avg"].clone()

    reset_lora_optimizer_states(model, opt)

    # Non-LoRA states should be unchanged
    for name, p in model.named_parameters():
        if name in non_lora_states_before:
            torch.testing.assert_close(
                opt.state[p]["exp_avg"],
                non_lora_states_before[name],
                msg=f"Non-LoRA param {name} was modified by reset",
            )


def _assert_maybe_merge_lora_with_optimizer_reset_integration():
    """Merge scheduling requantizes on the boundary and clears only stale LoRA state."""
    model = _make_model()
    inject_qlora_into_model(model, r=16, lora_alpha=16, quant_format="block_fp8", quant_group_size=128)
    _quantize_injected_model(model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    x = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16)
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()

    modules = [module for module in model.modules() if isinstance(module, QLoRALinear)]
    assert modules
    assert all(module.lora_A in optimizer.state and module.lora_B in optimizer.state for module in modules)
    with torch.no_grad():
        for module in modules:
            module.lora_A.fill_(0.25)
            module.lora_B.fill_(0.25)
    packed_before = [module.packed_weight_f32.clone() for module in modules]

    maybe_merge_lora(
        model,
        enable_lora=False,
        enable_qlora=True,
        merge_interval=10,
        global_step=9,
        optimizer=optimizer,
        reset_optimizer=True,
    )
    assert all(torch.equal(module.packed_weight_f32, before) for module, before in zip(modules, packed_before))
    assert all(module.lora_A in optimizer.state and module.lora_B in optimizer.state for module in modules)
    assert all(torch.all(module.lora_B == 0.25) for module in modules)

    maybe_merge_lora(
        model,
        enable_lora=False,
        enable_qlora=True,
        merge_interval=10,
        global_step=10,
        optimizer=optimizer,
        reset_optimizer=True,
    )
    assert any(not torch.equal(module.packed_weight_f32, before) for module, before in zip(modules, packed_before))
    assert all(module.lora_A not in optimizer.state and module.lora_B not in optimizer.state for module in modules)
    assert all(torch.count_nonzero(module.lora_B) == 0 for module in modules)


def test_qlora_quantized_execution_and_format_lifecycle_contract():
    _assert_quantize_forward_backward_memory()
    _assert_dequantize_roundtrip()
    _assert_prequantized_nvfp4_loading()
    _assert_ema_amax_and_scale_convention()
    _assert_merge_weights_and_requant()
    _assert_prequantized_block_fp8_load_and_forward()
    _assert_prequantized_block_fp8_merge()
    _assert_qlora_injection_contract()
    _assert_qlora_optimizer_reset_lifecycle()


def _assert_qlora_injection_contract():
    _assert_injection_and_training()
    _assert_inject_with_checkpoint_quant_format()


def _assert_qlora_optimizer_reset_lifecycle():
    _assert_reset_lora_optimizer_states_clears()
    _assert_reset_ignores_non_lora_params()
    _assert_maybe_merge_lora_with_optimizer_reset_integration()
