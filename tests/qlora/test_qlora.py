"""Tests for QLoRA modules: quantization, EMA amax, scale convention, re-quantization."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xorl.qlora.modules.linear import QLoRALinear
from xorl.qlora.modules.block_fp8_linear import BlockFP8QLoRALinear
from xorl.qlora.utils import inject_qlora_into_model, maybe_requant_qlora
from xorl.ops.quantize import nvfp4_quantize, nvfp4_dequantize
from xorl.ops.quantize.fp4_codec import FP4_E2M1_MAX, FP8_E4M3_MAX
from xorl.ops.quantize import block_fp8_quantize_gkn as block_fp8_weight_quant, block_fp8_dequantize_gkn as block_fp8_weight_dequant
from xorl.trainers.training_utils import reset_lora_optimizer_states, maybe_merge_lora

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
    from xorl.qlora.modules.linear import QLoRALinear
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

@pytest.mark.parametrize("quant_format,gs", [("nvfp4", 16), ("block_fp8", 128)])
def test_quantize_forward_backward_memory(quant_format, gs):
    """bf16 quantization, forward, backward (only LoRA gets grad), memory savings."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    bf16_bytes = linear.weight.numel() * 2
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16,
                                     quant_format=quant_format, quant_group_size=gs)

    # Quantized storage
    assert qlora.packed_weight_f32 is not None
    assert qlora.packed_weight_f32.dtype == torch.float32
    assert not qlora.packed_weight_f32.requires_grad
    assert qlora.lora_A.requires_grad and qlora.lora_B.requires_grad

    # Forward
    x = torch.randn(2, 10, 256, device="cuda", dtype=torch.bfloat16)
    out = qlora(x)
    assert out.shape == (2, 10, 512)

    # Backward: only LoRA gets grad
    out.sum().backward()
    assert qlora.lora_A.grad is not None and qlora.lora_B.grad is not None

    # Memory savings
    quant_bytes = qlora.packed_weight_f32.numel() * 4
    if qlora.weight_block_scales is not None:
        quant_bytes += qlora.weight_block_scales.numel() * qlora.weight_block_scales.element_size()
    if getattr(qlora, "weight_global_scale", None) is not None:
        quant_bytes += qlora.weight_global_scale.numel() * qlora.weight_global_scale.element_size()
    assert quant_bytes < bf16_bytes


def test_dequantize_roundtrip():
    """Dequantized weight should be close to original."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    w_orig = linear.weight.detach().clone()
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16,
                                     quant_format="nvfp4", quant_group_size=16)
    w_deq = qlora._dequantize_weight().to(torch.bfloat16)
    assert torch.allclose(w_orig, w_deq, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# 2. Pre-quantized nvfp4 loading
# ---------------------------------------------------------------------------

def test_prequantized_nvfp4_loading():
    """from_quantized() loads pre-packed nvfp4 weights; forward+backward work."""
    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
    packed, block_scales, global_scale = nvfp4_quantize(w, 16)

    qlora = QLoRALinear.from_quantized(
        packed_weight=packed, weight_block_scales=block_scales,
        weight_global_scale=global_scale, in_features=256, out_features=512,
        quant_format="nvfp4", quant_group_size=16, device="cuda",
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

def test_ema_amax_and_scale_convention():
    """EMA amax: init from bf16, update on merge, global_scale formula.
    Scale convention: block_scales use full fp8 range, dequant roundtrip accuracy."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    expected_amax = linear.weight.float().abs().max().item()

    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16,
                                     quant_format="nvfp4", quant_group_size=16)

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
    gs = qlora._recover_tensor(
        qlora.weight_global_scale, qlora._scale_dtypes["weight_global_scale"]
    ).item()
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

def test_merge_weights_and_requant():
    """merge_weights folds LoRA into base; maybe_requant_qlora merges+resets+EMA updates."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16,
                                     quant_format="nvfp4", quant_group_size=16)

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
    qlora2 = QLoRALinear.from_module(linear2, r=16, lora_alpha=16,
                                      quant_format="nvfp4", quant_group_size=16)
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
    qlora3 = QLoRALinear.from_module(linear3, r=16, lora_alpha=16,
                                      quant_format="nvfp4", quant_group_size=16)
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

def test_injection_and_training():
    """inject_qlora replaces target modules; forward/backward work; loss decreases."""
    model = _make_model()
    inject_qlora_into_model(model, r=16, lora_alpha=16,
                            quant_format="nvfp4", quant_group_size=16,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
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


@pytest.mark.parametrize("quant_format,gs", [("nvfp4", 16), ("block_fp8", 128)])
def test_training_step_converges(quant_format, gs):
    """Loss decreases over training steps for both quant formats."""
    model = _make_model()
    inject_qlora_into_model(model, r=16, lora_alpha=16,
                            quant_format=quant_format, quant_group_size=gs)
    _quantize_injected_model(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    x = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"{quant_format}: loss did not decrease: {losses}"


# ---------------------------------------------------------------------------
# 6. Step-based requant training (end-to-end)
# ---------------------------------------------------------------------------

def test_step_based_requant_training():
    """Train with periodic requant: loss decreases, requant triggered, continues after reset."""
    model = _make_model()
    inject_qlora_into_model(model, r=16, lora_alpha=16,
                            quant_format="nvfp4", quant_group_size=16)
    _quantize_injected_model(model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    x = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)

    losses = []
    requant_count = 0
    for step in range(1, 26):
        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            requant_count += maybe_requant_qlora(model)

    assert requant_count > 0
    assert losses[-1] < losses[0]

    # Also verify single-module requant continues training after reset
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16,
                                     quant_format="nvfp4", quant_group_size=16)
    qlora.train()
    opt = torch.optim.AdamW([qlora.lora_A, qlora.lora_B], lr=1e-3)
    x2 = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)
    tgt2 = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16)
    single_losses = []
    m = nn.ModuleList([qlora])
    for step in range(1, 31):
        opt.zero_grad()
        loss = ((qlora(x2) - tgt2) ** 2).mean()
        loss.backward()
        opt.step()
        single_losses.append(loss.item())
        if step % 10 == 0:
            maybe_requant_qlora(m)
    assert single_losses[-1] < single_losses[0]


def test_inject_with_checkpoint_quant_format():
    """inject_qlora with checkpoint_quant_format sets _source_quant_format."""
    model = _make_model()
    inject_qlora_into_model(
        model, r=16, lora_alpha=16,
        quant_format="block_fp8", quant_group_size=128,
        checkpoint_quant_format="block_fp8",
    )
    for m in model.modules():
        if isinstance(m, QLoRALinear):
            assert m._source_quant_format == "block_fp8"
            assert m.quant_format == "block_fp8"


# ---------------------------------------------------------------------------
# 8. Pre-quantized block FP8 loading (HF checkpoint path)
# ---------------------------------------------------------------------------

def test_prequantized_block_fp8_load_and_forward():
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


def test_prequantized_block_fp8_merge_and_training():
    """Merge weights + training loop work after loading pre-quantized block FP8."""
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

    # Training loop
    qlora2 = BlockFP8QLoRALinear(K, M, r=16, lora_alpha=16, device=torch.device("cuda"))
    qlora2._is_prequantized = True
    qlora2._source_quant_format = "block_fp8"
    qlora2._merge_sources = None
    qlora2._source_fqn = "model.layers.0.self_attn.o_proj"
    qlora2._load_prequantized(lambda key: mock_data[key])
    qlora2.train()
    opt = torch.optim.AdamW([qlora2.lora_A, qlora2.lora_B], lr=1e-2)
    x = torch.randn(4, 16, K, device="cuda", dtype=torch.bfloat16) * 0.1
    target = torch.randn(4, 16, M, device="cuda", dtype=torch.bfloat16) * 0.1
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = ((qlora2(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# 9. ReLoRA optimizer reset
# ---------------------------------------------------------------------------

def test_reset_lora_optimizer_states_clears():
    """Verify ReLoRA reset clears optimizer states for LoRA params."""
    linear = nn.Linear(256, 512, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora = QLoRALinear.from_module(linear, r=16, lora_alpha=16,
                                     quant_format="block_fp8", quant_group_size=128)
    qlora.train()
    opt = torch.optim.AdamW([qlora.lora_A, qlora.lora_B], lr=1e-3)

    # Run a few steps to populate optimizer states
    x = torch.randn(4, 8, 256, device="cuda", dtype=torch.bfloat16)
    tgt = torch.randn(4, 8, 512, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
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


def test_reset_ignores_non_lora_params():
    """Optimizer reset only touches LoRA params, not other trainable params."""
    model = _make_model()
    inject_qlora_into_model(model, r=16, lora_alpha=16,
                            quant_format="block_fp8", quant_group_size=128,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    _quantize_injected_model(model)
    # MLP layers are nn.Linear (not QLoRA), so add them as trainable too
    all_params = list(model.parameters())
    opt = torch.optim.AdamW(all_params, lr=1e-3)

    x = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
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
                opt.state[p]["exp_avg"], non_lora_states_before[name],
                msg=f"Non-LoRA param {name} was modified by reset"
            )


def test_merge_with_optimizer_reset_rank_accumulation():
    """Periodic merge + ReLoRA optimizer reset accumulates effective rank
    and beats single-rank LoRA on a high-rank target.

    Uses block_fp8 which has low re-quantization error, letting the rank
    accumulation benefit dominate. nvfp4's 4-bit re-quantization noise
    offsets the rank benefit in short runs.
    """
    torch.manual_seed(42)
    dim = 256

    # Target requires rank 32; single rank-4 LoRA cannot fully fit it
    base_linear = nn.Linear(dim, dim, bias=False, device="cuda", dtype=torch.bfloat16)
    W_base = base_linear.weight.detach().clone()
    A_target = torch.randn(dim, 32, device="cuda", dtype=torch.bfloat16) * 0.1
    B_target = torch.randn(32, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    W_target = W_base + (A_target @ B_target)

    x = torch.randn(8, 16, dim, device="cuda", dtype=torch.bfloat16)
    target = F.linear(x, W_target)

    def _train(merge_interval, reset_opt, total_steps, r=4, lr=1e-3):
        torch.manual_seed(42)
        linear = nn.Linear(dim, dim, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(W_base)
        qlora = QLoRALinear.from_module(linear, r=r, lora_alpha=r,
                                         quant_format="block_fp8", quant_group_size=128)
        qlora.train()
        opt = torch.optim.AdamW([qlora.lora_A, qlora.lora_B], lr=lr)
        model = nn.ModuleList([qlora])
        losses = []
        for step in range(1, total_steps + 1):
            opt.zero_grad()
            loss = ((qlora(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if merge_interval > 0 and step % merge_interval == 0:
                maybe_requant_qlora(model)
                if reset_opt:
                    reset_lora_optimizer_states(model, opt)
        return losses

    # No merge: rank-4 LoRA saturates at best rank-4 approximation
    losses_no_merge = _train(merge_interval=0, reset_opt=False, total_steps=200)

    # Merge every 40 steps + optimizer reset (5 merges over 200 steps)
    losses_merge_reset = _train(merge_interval=40, reset_opt=True, total_steps=200)

    # Merge + reset should converge
    assert losses_merge_reset[-1] < losses_merge_reset[0], \
        f"Merge+reset didn't converge: {losses_merge_reset[0]:.4f} -> {losses_merge_reset[-1]:.4f}"

    # Merge + reset should reach lower loss than no-merge
    # (accumulated rank from 5 merges of rank-4 > single rank-4)
    assert losses_merge_reset[-1] < losses_no_merge[-1], \
        f"Merge+reset ({losses_merge_reset[-1]:.4f}) should beat no-merge ({losses_no_merge[-1]:.4f})"


@pytest.mark.parametrize("quant_format,gs", [("nvfp4", 16), ("block_fp8", 128)])
def test_merge_with_optimizer_reset_still_converges(quant_format, gs):
    """Merge + optimizer reset converges for both quant formats (no regression)."""
    torch.manual_seed(42)
    dim = 256
    linear = nn.Linear(dim, dim, bias=False, device="cuda", dtype=torch.bfloat16)
    qlora = QLoRALinear.from_module(linear, r=8, lora_alpha=8,
                                     quant_format=quant_format, quant_group_size=gs)
    qlora.train()
    opt = torch.optim.AdamW([qlora.lora_A, qlora.lora_B], lr=1e-3)
    model = nn.ModuleList([qlora])
    x = torch.randn(4, 16, dim, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 16, dim, device="cuda", dtype=torch.bfloat16)
    losses = []
    for step in range(1, 61):
        opt.zero_grad()
        loss = ((qlora(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 20 == 0:
            maybe_requant_qlora(model)
            reset_lora_optimizer_states(model, opt)
    assert losses[-1] < losses[0], \
        f"{quant_format} didn't converge: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_merge_with_reset_vs_without_reset():
    """Compare merge+reset vs merge-only to verify reset doesn't hurt convergence.

    Both should converge. Reset may help or be neutral — the key is it doesn't
    catastrophically hurt training.
    """
    torch.manual_seed(123)
    dim = 256

    base_linear = nn.Linear(dim, dim, bias=False, device="cuda", dtype=torch.bfloat16)
    W_base = base_linear.weight.detach().clone()

    x = torch.randn(8, 16, dim, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(8, 16, dim, device="cuda", dtype=torch.bfloat16)

    def _train(reset_opt, total_steps=120, merge_interval=30):
        torch.manual_seed(123)
        linear = nn.Linear(dim, dim, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(W_base)
        qlora = QLoRALinear.from_module(linear, r=8, lora_alpha=8,
                                         quant_format="block_fp8", quant_group_size=128)
        qlora.train()
        opt = torch.optim.AdamW([qlora.lora_A, qlora.lora_B], lr=1e-3)
        model = nn.ModuleList([qlora])
        losses = []
        for step in range(1, total_steps + 1):
            opt.zero_grad()
            loss = ((qlora(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step % merge_interval == 0:
                maybe_requant_qlora(model)
                if reset_opt:
                    reset_lora_optimizer_states(model, opt)
        return losses

    losses_merge_only = _train(reset_opt=False)
    losses_merge_reset = _train(reset_opt=True)

    # Both should converge (loss decreases)
    assert losses_merge_only[-1] < losses_merge_only[0], \
        f"Merge-only didn't converge: {losses_merge_only[0]:.4f} -> {losses_merge_only[-1]:.4f}"
    assert losses_merge_reset[-1] < losses_merge_reset[0], \
        f"Merge+reset didn't converge: {losses_merge_reset[0]:.4f} -> {losses_merge_reset[-1]:.4f}"

    # Reset should not catastrophically hurt (within 2x of merge-only final loss)
    assert losses_merge_reset[-1] < losses_merge_only[-1] * 2.0, \
        f"Reset hurt too much: {losses_merge_reset[-1]:.4f} vs merge-only {losses_merge_only[-1]:.4f}"


def test_maybe_merge_lora_with_optimizer_reset_integration():
    """End-to-end test of maybe_merge_lora with reset_optimizer=True."""
    model = _make_model()
    inject_qlora_into_model(model, r=16, lora_alpha=16,
                            quant_format="block_fp8", quant_group_size=128)
    _quantize_injected_model(model)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    x = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16)

    losses = []
    for step in range(1, 31):
        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        maybe_merge_lora(
            model, enable_lora=False, enable_qlora=True,
            merge_interval=10, global_step=step,
            optimizer=optimizer, reset_optimizer=True,
        )

    # Should still converge despite optimizer resets
    assert losses[-1] < losses[0], f"Did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    # Verify merges happened (lora_B should be zero after last merge at step 30)
    for m in model.modules():
        if isinstance(m, QLoRALinear):
            assert torch.all(m.lora_B == 0), "lora_B should be reset after merge"
