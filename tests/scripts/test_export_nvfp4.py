"""CPU tests for the NVFP4 directory exporter (export_nvfp4)."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from xorl.cli.export_nvfp4 import (
    GLOBAL_SCALE_KEY,
    dequantize_nvfp4_export,
    export_hf_directory_to_nvfp4,
    quantize_weight_to_nvfp4,
)


pytestmark = pytest.mark.cpu


class TestQuantizeRoundTrip:
    def test_packed_layout_and_dequant_error(self):
        torch.manual_seed(0)
        w = torch.randn(64, 32, dtype=torch.bfloat16)
        entry = quantize_weight_to_nvfp4(w, block_size=16)
        assert entry["weight"].dtype == torch.uint8
        assert entry["weight"].shape == (64, 16)  # in/2 packed
        assert entry["weight_scale"].dtype == torch.float8_e4m3fn
        assert entry["weight_scale"].shape == (64, 2)  # in/group
        assert entry[GLOBAL_SCALE_KEY].dtype == torch.float32
        assert entry[GLOBAL_SCALE_KEY].ndim == 0
        recon = dequantize_nvfp4_export(entry)
        rel = (recon.float() - w.float()).norm() / w.float().norm()
        assert rel < 0.15, f"NVFP4 round-trip error too high: {rel:.4f}"

    def test_shared_global_scale(self):
        torch.manual_seed(0)
        a = torch.randn(32, 32)
        b = torch.randn(32, 32) * 5.0  # different amax
        gs = (torch.stack([a.abs().amax(), b.abs().amax()]).amax() / (6.0 * 448.0)).reshape(1)
        ea = quantize_weight_to_nvfp4(a, global_scale=gs)
        eb = quantize_weight_to_nvfp4(b, global_scale=gs)
        torch.testing.assert_close(ea[GLOBAL_SCALE_KEY], eb[GLOBAL_SCALE_KEY], rtol=0, atol=0)


def _build_tiny_hf_dir(tmp_path):
    torch.manual_seed(0)
    hidden, inter = 32, 48
    state = {
        # separate q/k/v -> fused-group shared scale; o_proj standalone
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden, hidden, dtype=torch.bfloat16) * 4.0,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden, hidden, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(hidden, dtype=torch.bfloat16),  # 1D passthrough
        "lm_head.weight": torch.randn(hidden, hidden, dtype=torch.bfloat16),  # skipped (BF16 island)
        # 2D float weight whose FULL name has a "model." prefix: a bare "embed_tokens"
        # skip pattern does NOT match it, so it would be wrongly packed and serve garbage
        # (sglang vocab_parallel_embedding expects bf16 [vocab, hidden]). Regression guard.
        "model.embed_tokens.weight": torch.randn(40, hidden, dtype=torch.bfloat16),
    }
    in_dir = tmp_path / "src"
    in_dir.mkdir()
    save_file(state, str(in_dir / "model.safetensors"), metadata={"format": "pt"})
    (in_dir / "config.json").write_text(json.dumps({"hidden_size": hidden, "intermediate_size": inter}))
    return in_dir, state


class TestDirectoryExport:
    def test_end_to_end(self, tmp_path):
        in_dir, state = _build_tiny_hf_dir(tmp_path)
        out_dir = tmp_path / "out"
        result = export_hf_directory_to_nvfp4(in_dir, out_dir, group_size=16)
        assert result["quantized_tensors"] == 5  # q,k,v,o,down  (lm_head skipped, norm 1D)
        assert result["fused_groups"] == 1  # qkv

        tensors = {}
        for shard in out_dir.glob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as h:
                for k in h.keys():
                    tensors[k] = h.get_tensor(k)

        # Quantized projections carry the modelopt triple.
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = f"model.layers.0.self_attn.{proj}"
            assert tensors[f"{base}.weight"].dtype == torch.uint8
            assert tensors[f"{base}.weight_scale"].dtype == torch.float8_e4m3fn
            assert tensors[f"{base}.weight_scale_2"].dtype == torch.float32

        # q/k/v share one weight_scale_2; o_proj has its own.
        q2 = tensors["model.layers.0.self_attn.q_proj.weight_scale_2"]
        k2 = tensors["model.layers.0.self_attn.k_proj.weight_scale_2"]
        v2 = tensors["model.layers.0.self_attn.v_proj.weight_scale_2"]
        torch.testing.assert_close(q2, k2, rtol=0, atol=0)
        torch.testing.assert_close(q2, v2, rtol=0, atol=0)

        # BF16 islands pass through unquantized — incl. the "model."-prefixed embed_tokens,
        # which a bare "embed_tokens" pattern fails to match (the served-garbage bug).
        assert "lm_head.weight_scale_2" not in tensors
        assert tensors["lm_head.weight"].dtype == torch.bfloat16
        assert tensors["model.norm.weight"].dtype == torch.bfloat16
        assert "model.embed_tokens.weight_scale_2" not in tensors
        assert "model.embed_tokens.weight_scale" not in tensors
        assert tensors["model.embed_tokens.weight"].dtype == torch.bfloat16
        assert tensors["model.embed_tokens.weight"].shape == (40, 32)  # unpacked, not [40,16]

        # hf_quant_config.json + stamped config.json.
        cfg = json.loads((out_dir / "hf_quant_config.json").read_text())
        assert cfg["quantization"]["quant_algo"] == "NVFP4"
        assert cfg["quantization"]["group_size"] == 16
        served = json.loads((out_dir / "config.json").read_text())
        assert served["quantization_config"]["quant_algo"] == "NVFP4"

        # Round-trip a quantized projection.
        entry = {
            "weight": tensors["model.layers.0.self_attn.o_proj.weight"],
            "weight_scale": tensors["model.layers.0.self_attn.o_proj.weight_scale"],
            GLOBAL_SCALE_KEY: tensors["model.layers.0.self_attn.o_proj.weight_scale_2"],
        }
        recon = dequantize_nvfp4_export(entry)
        ref = state["model.layers.0.self_attn.o_proj.weight"]
        rel = (recon.float() - ref.float()).norm() / ref.float().norm()
        assert rel < 0.2, f"exported o_proj round-trip error too high: {rel:.4f}"

    def test_input_scale_w4a4(self, tmp_path):
        in_dir, _ = _build_tiny_hf_dir(tmp_path)
        out_dir = tmp_path / "out_w4a4"
        amax = {
            "model.layers.0.self_attn.q_proj": 4.0,
            "model.layers.0.self_attn.k_proj": 4.0,  # fused member shares the input -> same amax
            "model.layers.0.self_attn.o_proj": 2.0,
        }
        result = export_hf_directory_to_nvfp4(in_dir, out_dir, group_size=16, activation_amax=amax)
        assert result["input_scale_tensors"] == 3
        assert result["serving_regime"] == "W4A4"
        tensors = {}
        for shard in out_dir.glob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as h:
                for k in h.keys():
                    tensors[k] = h.get_tensor(k)
        # input_scale = amax / (6 * 448), 0-dim fp32
        qis = tensors["model.layers.0.self_attn.q_proj.input_scale"]
        assert qis.dtype == torch.float32 and qis.ndim == 0
        torch.testing.assert_close(qis, torch.tensor(4.0 / (6.0 * 448.0)), rtol=1e-6, atol=0)
        # a linear with no calibrated amax stays weight-only (no input_scale)
        assert "model.layers.0.mlp.down_proj.input_scale" not in tensors

    def test_rejects_already_quantized(self, tmp_path):
        in_dir, _ = _build_tiny_hf_dir(tmp_path)
        out_dir = tmp_path / "out1"
        export_hf_directory_to_nvfp4(in_dir, out_dir, group_size=16)
        with pytest.raises(ValueError, match="already contains NVFP4"):
            export_hf_directory_to_nvfp4(out_dir, tmp_path / "out2", group_size=16)
