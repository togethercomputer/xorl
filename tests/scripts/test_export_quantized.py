import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from xorl.cli.export_quantized import (
    _parse_size_bytes,
    export_hf_directory_to_fp8,
    export_qarl_directory_to_fp8,
    parse_args,
    quantize_weight_to_fp8,
)
from xorl.qarl import inject_qarl_into_model
from xorl.server.weight_sync.handler import WeightSyncHandler


pytestmark = pytest.mark.cpu


def _write_source_model(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    num_hidden_layers: int = 3,
    config_updates: dict[str, object] | None = None,
) -> None:
    path.mkdir(parents=True)
    config = {
        "architectures": ["TinyForCausalLM"],
        "model_type": "tiny",
        "num_hidden_layers": num_hidden_layers,
    }
    config.update(config_updates or {})
    (path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text('{"model_max_length": 128}\n', encoding="utf-8")
    save_file(tensors, path / "model.safetensors")


def _dequantize_block_fp8(quantized: torch.Tensor, scale: torch.Tensor, block_size: tuple[int, int]) -> torch.Tensor:
    block_rows, block_cols = block_size
    rows, cols = quantized.shape
    pad_rows = (block_rows - rows % block_rows) % block_rows
    pad_cols = (block_cols - cols % block_cols) % block_cols
    work = quantized.to(torch.float32)
    if pad_rows or pad_cols:
        padded = torch.zeros(rows + pad_rows, cols + pad_cols, dtype=torch.float32)
        padded[:rows, :cols] = work
    else:
        padded = work
    block_row_count = padded.shape[0] // block_rows
    block_col_count = padded.shape[1] // block_cols
    blocks = padded.reshape(block_row_count, block_rows, block_col_count, block_cols).permute(0, 2, 1, 3)
    dequantized = blocks * scale.unsqueeze(-1).unsqueeze(-1)
    dequantized = dequantized.permute(0, 2, 1, 3).reshape(padded.shape)
    return dequantized[:rows, :cols].contiguous()


def _last_element_padded_block_fp8_reference(
    tensor: torch.Tensor,
    *,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    block_rows, block_cols = block_size
    work = tensor.detach().cpu().float()
    rows, cols = work.shape
    pad_rows = (block_rows - rows % block_rows) % block_rows
    pad_cols = (block_cols - cols % block_cols) % block_cols
    if pad_rows or pad_cols:
        padded = torch.full(
            (rows + pad_rows, cols + pad_cols),
            work.flatten()[-1].item(),
            dtype=torch.float32,
        )
        padded[:rows, :cols] = work
    else:
        padded = work

    block_row_count = padded.shape[0] // block_rows
    block_col_count = padded.shape[1] // block_cols
    blocks = padded.reshape(block_row_count, block_rows, block_col_count, block_cols).permute(0, 2, 1, 3)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = blocks.abs().reshape(block_row_count, block_col_count, -1).max(dim=-1).values.clamp(min=1e-12) / fp8_max
    quantized_blocks = (blocks / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    quantized = quantized_blocks.permute(0, 2, 1, 3).reshape(padded.shape)
    return quantized[:rows, :cols].contiguous(), scale.contiguous()


class TinyExportLogprobModel(nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.ModuleDict({"proj": nn.Linear(hidden_size, hidden_size)})])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        hidden = F.silu(self.model.layers[0]["proj"](hidden))
        return self.lm_head(hidden)


def _target_logprobs(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids)
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def test_quantize_weight_to_fp8_matches_weight_sync_helper():
    weight = torch.tensor(
        [
            [0.0, 1.0, -2.0],
            [3.5, -4.0, 0.25],
            [0.75, -1.5, 2.5],
        ],
        dtype=torch.bfloat16,
    )

    quantized, scale = quantize_weight_to_fp8(weight, weight_block_size=(2, 2))
    sync_out = WeightSyncHandler._quantize_buffer_for_fp8(
        [("model.layers.1.self_attn.q_proj.weight", weight)],
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [2, 2],
            "modules_to_not_convert": [],
        },
    )

    assert sync_out[0][0] == "model.layers.1.self_attn.q_proj.weight"
    assert sync_out[1][0] == "model.layers.1.self_attn.q_proj.weight_scale_inv"
    assert torch.equal(quantized, sync_out[0][1])
    torch.testing.assert_close(scale, sync_out[1][1])


def test_quantize_weight_to_fp8_zero_padding_differs_from_last_element_padding_for_partial_blocks():
    block_size = (2, 4)
    weight = torch.tensor(
        [
            [1.0, -1.0, 0.5, -0.5, 0.25],
            [1.5, -1.5, 0.75, -0.75, -0.25],
            [2.0, -2.0, 1.0, -1.0, 1024.0],
        ],
        dtype=torch.bfloat16,
    )

    quantized, scale = quantize_weight_to_fp8(weight, weight_block_size=block_size)
    last_padded_weight, last_padded_scale = _last_element_padded_block_fp8_reference(weight, block_size=block_size)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    expected_zero_padded_scale = (
        torch.tensor(
            [
                [1.5, 0.25],
                [2.0, 1024.0],
            ],
            dtype=torch.float32,
        )
        / fp8_max
    )

    torch.testing.assert_close(scale, expected_zero_padded_scale, rtol=0.0, atol=0.0)
    assert not torch.equal(scale, last_padded_scale)
    assert not torch.equal(quantized.view(torch.uint8), last_padded_weight.view(torch.uint8))


def test_parse_args_loads_export_config_and_appends_cli_skip(tmp_path):
    config_path = tmp_path / "export.yaml"
    config_path.write_text(
        """
input_dir: /tmp/source
output_dir: /tmp/exported
weight_block_size: [64, 128]
modules_to_not_convert:
  - lm_head
num_first_layers_bf16: 1
max_shard_size: 512MiB
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--module-to-not-convert",
            "model.embed_tokens",
            "--num-last-layers-bf16",
            "2",
        ]
    )

    assert args.input_dir == "/tmp/source"
    assert args.output_dir == "/tmp/exported"
    assert args.weight_block_size == [64, 128]
    assert args.modules_to_not_convert == ["lm_head", "model.embed_tokens"]
    assert args.num_first_layers_bf16 == 1
    assert args.num_last_layers_bf16 == 2
    assert args.max_shard_size == "512MiB"


def test_qwen3_8b_block_fp8_export_example_config_parses():
    config_path = Path(__file__).resolve().parents[2] / "examples/server/configs/export/qwen3_8b_block_fp8_export.yaml"

    args = parse_args(["--config", str(config_path)])

    assert args.input_dir == "/path/to/qwen3-8b-bf16-hf-weights"
    assert args.output_dir == "/path/to/qwen3-8b-block-fp8-hf-weights"
    assert args.weight_block_size == [128, 128]
    assert args.modules_to_not_convert == ["model.embed_tokens", "lm_head"]
    assert args.num_first_layers_bf16 == 1
    assert args.num_last_layers_bf16 == 1


def test_export_quantized_module_cli_exports_from_config(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "model.layers.1.self_attn.q_proj.weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4),
        "lm_head.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }
    _write_source_model(source, tensors, num_hidden_layers=2)
    export_config = tmp_path / "export.yaml"
    export_config.write_text(
        f"""
input_dir: {source}
output_dir: {output}
weight_block_size: [2, 2]
modules_to_not_convert:
  - lm_head
num_first_layers_bf16: 1
max_shard_size: 1MiB
""",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else os.pathsep.join([src_path, env["PYTHONPATH"]])
    completed = subprocess.run(
        [sys.executable, "-m", "xorl.cli.export_quantized", "--config", str(export_config)],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    result = json.loads(completed.stdout)
    assert result["output_dir"] == str(output)
    assert result["quantized_weights"] == 1
    assert result["shard_count"] == 1

    exported = load_file(output / "model.safetensors")
    assert exported["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float32
    assert exported["model.layers.1.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert exported["model.layers.1.self_attn.q_proj.weight_scale_inv"].shape == (2, 2)
    assert exported["lm_head.weight"].dtype == torch.float32

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"] == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "modules_to_not_convert": ["lm_head", "model.layers.0"],
        "quant_method": "fp8",
        "weight_block_size": [2, 2],
    }


def test_export_hf_directory_to_fp8_preserves_bf16_islands_and_writes_config(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(15, dtype=torch.float32).reshape(3, 5),
        "model.layers.1.self_attn.q_proj.weight": torch.arange(15, 30, dtype=torch.float32).reshape(3, 5),
        "model.layers.2.self_attn.q_proj.weight": torch.arange(30, 45, dtype=torch.float32).reshape(3, 5),
        "lm_head.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "model.norm.weight": torch.ones(5, dtype=torch.float32),
    }
    _write_source_model(source, tensors)

    result = export_hf_directory_to_fp8(
        source,
        output,
        weight_block_size=(2, 2),
        modules_to_not_convert=["lm_head.weight"],
        num_first_layers_bf16=1,
        num_last_layers_bf16=1,
    )

    assert result.quantized_weights == 1
    assert result.shard_count == 1
    exported = load_file(output / "model.safetensors")
    assert exported["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float32
    assert exported["model.layers.2.self_attn.q_proj.weight"].dtype == torch.float32
    assert exported["lm_head.weight"].dtype == torch.float32
    assert exported["model.layers.1.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert exported["model.layers.1.self_attn.q_proj.weight_scale_inv"].shape == (2, 3)
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" not in exported
    assert "model.layers.2.self_attn.q_proj.weight_scale_inv" not in exported
    assert "lm_head.weight_scale_inv" not in exported

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"] == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "modules_to_not_convert": ["lm_head", "model.layers.0", "model.layers.2"],
        "quant_method": "fp8",
        "weight_block_size": [2, 2],
    }
    assert (output / "tokenizer_config.json").exists()


def test_export_hf_directory_to_fp8_writes_sharded_index(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "model.layers.1.self_attn.q_proj.weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4),
        },
        num_hidden_layers=2,
    )

    result = export_hf_directory_to_fp8(
        source,
        output,
        weight_block_size=(2, 2),
        max_shard_size=24,
    )

    assert result.shard_count > 1
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == result.total_size
    assert "model.layers.0.self_attn.q_proj.weight" in index["weight_map"]
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in index["weight_map"]
    assert all((output / shard_name).exists() for shard_name in set(index["weight_map"].values()))


def test_export_hf_directory_to_fp8_splits_fused_qkv_weights(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    qkv = torch.arange(16 * 5, dtype=torch.float32).reshape(16, 5)
    _write_source_model(
        source,
        {"model.layers.0.self_attn.qkv_proj.weight": qkv},
        num_hidden_layers=1,
        config_updates={
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        },
    )

    result = export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    assert result.quantized_weights == 3
    exported = load_file(output / "model.safetensors")
    assert set(exported) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.weight_scale_inv",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.weight_scale_inv",
    }

    expected_q_fp8, expected_q_scale = quantize_weight_to_fp8(qkv[:8], weight_block_size=(2, 2))
    expected_k_fp8, expected_k_scale = quantize_weight_to_fp8(qkv[8:12], weight_block_size=(2, 2))
    expected_v_fp8, expected_v_scale = quantize_weight_to_fp8(qkv[12:], weight_block_size=(2, 2))
    assert torch.equal(exported["model.layers.0.self_attn.q_proj.weight"], expected_q_fp8)
    assert torch.equal(exported["model.layers.0.self_attn.k_proj.weight"], expected_k_fp8)
    assert torch.equal(exported["model.layers.0.self_attn.v_proj.weight"], expected_v_fp8)
    torch.testing.assert_close(exported["model.layers.0.self_attn.q_proj.weight_scale_inv"], expected_q_scale)
    torch.testing.assert_close(exported["model.layers.0.self_attn.k_proj.weight_scale_inv"], expected_k_scale)
    torch.testing.assert_close(exported["model.layers.0.self_attn.v_proj.weight_scale_inv"], expected_v_scale)


def test_export_hf_directory_to_fp8_rejects_fused_qkv_without_attention_metadata(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {"model.layers.0.self_attn.qkv_proj.weight": torch.ones(16, 5)},
        num_hidden_layers=1,
    )

    with pytest.raises(ValueError, match="Cannot split qkv_proj"):
        export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_export_hf_directory_to_fp8_rejects_duplicate_converted_names(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.qkv_proj.weight": torch.arange(16 * 5, dtype=torch.float32).reshape(16, 5),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 5),
        },
        num_hidden_layers=1,
        config_updates={
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        },
    )

    with pytest.raises(ValueError, match="Duplicate exported tensor name"):
        export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_export_hf_directory_to_fp8_fuses_mla_a_projection_for_sglang(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    q_a = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5)
    kv_a = torch.arange(3 * 5, 8 * 5, dtype=torch.float32).reshape(5, 5)
    q_b = torch.ones(4, 3, dtype=torch.float32)
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_a_proj.weight": q_a,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": kv_a,
            "model.layers.0.self_attn.q_b_proj.weight": q_b,
        },
        num_hidden_layers=1,
        config_updates={"q_lora_rank": 3},
    )

    result = export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    assert result.quantized_weights == 2
    exported = load_file(output / "model.safetensors")
    assert set(exported) == {
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight_scale_inv",
    }
    expected_fused = torch.cat([q_a, kv_a], dim=0).contiguous()
    expected_fused_fp8, expected_fused_scale = quantize_weight_to_fp8(expected_fused, weight_block_size=(2, 2))
    expected_q_b_fp8, expected_q_b_scale = quantize_weight_to_fp8(q_b, weight_block_size=(2, 2))
    assert torch.equal(exported["model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight"], expected_fused_fp8)
    assert torch.equal(exported["model.layers.0.self_attn.q_b_proj.weight"], expected_q_b_fp8)
    torch.testing.assert_close(
        exported["model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"],
        expected_fused_scale,
    )
    torch.testing.assert_close(exported["model.layers.0.self_attn.q_b_proj.weight_scale_inv"], expected_q_b_scale)


def test_export_hf_directory_to_fp8_leaves_mla_a_projection_split_without_q_lora_rank(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    q_a = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5)
    kv_a = torch.arange(3 * 5, 8 * 5, dtype=torch.float32).reshape(5, 5)
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_a_proj.weight": q_a,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": kv_a,
        },
        num_hidden_layers=1,
    )

    export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    exported = load_file(output / "model.safetensors")
    assert "model.layers.0.self_attn.q_a_proj.weight" in exported
    assert "model.layers.0.self_attn.kv_a_proj_with_mqa.weight" in exported
    assert "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight" not in exported


def test_export_hf_directory_to_fp8_remaps_linear_attention_split_names(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    prefix = "model.layers.0.linear_attn"
    q = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
    k = torch.arange(2 * 5, 4 * 5, dtype=torch.float32).reshape(2, 5)
    v = torch.arange(4 * 5, 7 * 5, dtype=torch.float32).reshape(3, 5)
    q_conv = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    k_conv = torch.arange(2 * 1 * 3, 4 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    v_conv = torch.arange(4 * 1 * 3, 7 * 1 * 3, dtype=torch.float32).reshape(3, 1, 3)
    g = torch.full((3, 5), 1.0)
    a = torch.full((1, 5), 2.0)
    b = torch.full((1, 5), 3.0)
    o = torch.full((5, 3), 4.0)
    norm = torch.full((3,), 5.0)
    dt_bias = torch.full((1,), 6.0)
    a_log = torch.full((1,), 7.0)
    _write_source_model(
        source,
        {
            f"{prefix}.q_proj.weight": q,
            f"{prefix}.k_proj.weight": k,
            f"{prefix}.v_proj.weight": v,
            f"{prefix}.q_conv1d.weight": q_conv,
            f"{prefix}.k_conv1d.weight": k_conv,
            f"{prefix}.v_conv1d.weight": v_conv,
            f"{prefix}.g_proj.weight": g,
            f"{prefix}.a_proj.weight": a,
            f"{prefix}.b_proj.weight": b,
            f"{prefix}.o_proj.weight": o,
            f"{prefix}.o_norm.weight": norm,
            f"{prefix}.dt_bias": dt_bias,
            f"{prefix}.A_log": a_log,
        },
        num_hidden_layers=1,
        config_updates={"layer_types": ["linear_attention"]},
    )

    result = export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    assert result.quantized_weights == 5
    exported = load_file(output / "model.safetensors")
    expected_qkv = torch.cat([q, k, v], dim=0).contiguous()
    expected_qkv_fp8, expected_qkv_scale = quantize_weight_to_fp8(expected_qkv, weight_block_size=(2, 2))
    expected_z_fp8, expected_z_scale = quantize_weight_to_fp8(g, weight_block_size=(2, 2))
    expected_a_fp8, expected_a_scale = quantize_weight_to_fp8(a, weight_block_size=(2, 2))
    expected_b_fp8, expected_b_scale = quantize_weight_to_fp8(b, weight_block_size=(2, 2))
    expected_o_fp8, expected_o_scale = quantize_weight_to_fp8(o, weight_block_size=(2, 2))

    assert set(exported) == {
        f"{prefix}.in_proj_qkv.weight",
        f"{prefix}.in_proj_qkv.weight_scale_inv",
        f"{prefix}.conv1d.weight",
        f"{prefix}.in_proj_z.weight",
        f"{prefix}.in_proj_z.weight_scale_inv",
        f"{prefix}.in_proj_a.weight",
        f"{prefix}.in_proj_a.weight_scale_inv",
        f"{prefix}.in_proj_b.weight",
        f"{prefix}.in_proj_b.weight_scale_inv",
        f"{prefix}.out_proj.weight",
        f"{prefix}.out_proj.weight_scale_inv",
        f"{prefix}.norm.weight",
        f"{prefix}.dt_bias",
        f"{prefix}.A_log",
    }
    assert torch.equal(exported[f"{prefix}.in_proj_qkv.weight"], expected_qkv_fp8)
    torch.testing.assert_close(exported[f"{prefix}.in_proj_qkv.weight_scale_inv"], expected_qkv_scale)
    torch.testing.assert_close(exported[f"{prefix}.conv1d.weight"], torch.cat([q_conv, k_conv, v_conv], dim=0))
    assert torch.equal(exported[f"{prefix}.in_proj_z.weight"], expected_z_fp8)
    assert torch.equal(exported[f"{prefix}.in_proj_a.weight"], expected_a_fp8)
    assert torch.equal(exported[f"{prefix}.in_proj_b.weight"], expected_b_fp8)
    assert torch.equal(exported[f"{prefix}.out_proj.weight"], expected_o_fp8)
    torch.testing.assert_close(exported[f"{prefix}.in_proj_z.weight_scale_inv"], expected_z_scale)
    torch.testing.assert_close(exported[f"{prefix}.in_proj_a.weight_scale_inv"], expected_a_scale)
    torch.testing.assert_close(exported[f"{prefix}.in_proj_b.weight_scale_inv"], expected_b_scale)
    torch.testing.assert_close(exported[f"{prefix}.out_proj.weight_scale_inv"], expected_o_scale)
    torch.testing.assert_close(exported[f"{prefix}.norm.weight"], norm)
    torch.testing.assert_close(exported[f"{prefix}.dt_bias"], dt_bias)
    torch.testing.assert_close(exported[f"{prefix}.A_log"], a_log)


def test_export_hf_directory_to_fp8_leaves_linear_attention_split_without_layer_types(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    prefix = "model.layers.0.linear_attn"
    _write_source_model(
        source,
        {
            f"{prefix}.q_proj.weight": torch.ones(2, 5),
            f"{prefix}.k_proj.weight": torch.ones(2, 5),
            f"{prefix}.v_proj.weight": torch.ones(3, 5),
        },
        num_hidden_layers=1,
    )

    export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    exported = load_file(output / "model.safetensors")
    assert f"{prefix}.q_proj.weight" in exported
    assert f"{prefix}.k_proj.weight" in exported
    assert f"{prefix}.v_proj.weight" in exported
    assert f"{prefix}.in_proj_qkv.weight" not in exported


def test_export_hf_directory_to_fp8_converts_gkn_moe_experts_to_hf_layout(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    gate_up = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    down = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    _write_source_model(
        source,
        {
            "model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.layers.0.mlp.experts.down_proj": down,
        },
        num_hidden_layers=1,
    )

    result = export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    assert result.tensors_read == 2
    assert result.quantized_weights == 6
    exported = load_file(output / "model.safetensors")
    expected_names = {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.1.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight_scale_inv",
    }
    assert set(exported) == expected_names
    assert all(exported[name].dtype == torch.float8_e4m3fn for name in exported if name.endswith(".weight"))
    assert all(exported[name].dtype == torch.float32 for name in exported if name.endswith(".weight_scale_inv"))

    expected_gate = gate_up[0, :, :4].t().contiguous()
    expected_gate_fp8, expected_gate_scale = quantize_weight_to_fp8(expected_gate, weight_block_size=(2, 2))
    assert torch.equal(exported["model.layers.0.mlp.experts.0.gate_proj.weight"], expected_gate_fp8)
    torch.testing.assert_close(
        exported["model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"],
        expected_gate_scale,
    )

    expected_down = down[1].t().contiguous()
    expected_down_fp8, expected_down_scale = quantize_weight_to_fp8(expected_down, weight_block_size=(2, 2))
    assert torch.equal(exported["model.layers.0.mlp.experts.1.down_proj.weight"], expected_down_fp8)
    torch.testing.assert_close(
        exported["model.layers.0.mlp.experts.1.down_proj.weight_scale_inv"],
        expected_down_scale,
    )


def test_export_hf_directory_to_fp8_splits_fused_gate_up_weights(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    gate_up = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    _write_source_model(
        source,
        {"model.layers.0.mlp.gate_up_proj.weight": gate_up},
        num_hidden_layers=1,
    )

    result = export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))

    assert result.quantized_weights == 2
    exported = load_file(output / "model.safetensors")
    assert set(exported) == {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight_scale_inv",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale_inv",
    }
    expected_gate_fp8, expected_gate_scale = quantize_weight_to_fp8(gate_up[:4], weight_block_size=(2, 2))
    expected_up_fp8, expected_up_scale = quantize_weight_to_fp8(gate_up[4:], weight_block_size=(2, 2))
    assert torch.equal(exported["model.layers.0.mlp.gate_proj.weight"], expected_gate_fp8)
    assert torch.equal(exported["model.layers.0.mlp.up_proj.weight"], expected_up_fp8)
    torch.testing.assert_close(exported["model.layers.0.mlp.gate_proj.weight_scale_inv"], expected_gate_scale)
    torch.testing.assert_close(exported["model.layers.0.mlp.up_proj.weight_scale_inv"], expected_up_scale)


def test_export_hf_directory_to_fp8_rejects_existing_fp8_scale_tensors(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(1, 1),
        },
        num_hidden_layers=1,
    )

    with pytest.raises(ValueError, match="weight_scale_inv"):
        export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_export_hf_directory_to_fp8_rejects_mtp_config_metadata(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        },
        num_hidden_layers=1,
        config_updates={"text_config": {"num_nextn_predict_layers": 1}},
    )

    with pytest.raises(ValueError, match="MTP/speculative low-precision export"):
        export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_export_hf_directory_to_fp8_rejects_mtp_tensor_names(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "model.mtp.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        },
        num_hidden_layers=1,
    )

    with pytest.raises(ValueError, match="weights include model.mtp.layers.0"):
        export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_export_hf_directory_rejects_qarl_state_without_fold_flag(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
            "model.layers.0.self_attn.q_proj.qarl_input_amax": torch.tensor(1.0),
            "model.layers.0.self_attn.q_proj.qarl_weight_amax": torch.tensor(1.0),
            "model.layers.0.self_attn.q_proj.qarl_forward_count": torch.tensor(3, dtype=torch.long),
        },
        num_hidden_layers=1,
    )

    with pytest.raises(ValueError, match="QARL quantizer state"):
        export_hf_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_export_qarl_directory_to_fp8_folds_only_qarl_modules(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "model.layers.0.self_attn.q_proj.bias": torch.arange(4, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.qarl_input_amax": torch.tensor(4.0),
        "model.layers.0.self_attn.q_proj.qarl_weight_amax": torch.tensor(15.0),
        "model.layers.0.self_attn.q_proj.qarl_input_scale_inv": torch.tensor(0.01),
        "model.layers.0.self_attn.q_proj.qarl_weight_scale_inv": torch.ones(2, 2),
        "model.layers.0.self_attn.q_proj.qarl_forward_count": torch.tensor(5, dtype=torch.long),
        "model.layers.1.self_attn.q_proj.weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4),
        "lm_head.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }
    _write_source_model(source, tensors, num_hidden_layers=2)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["xorl_qarl_config"] = {
        "quant_cfg": {
            "format": "fp8_e4m3",
            "weight": True,
            "activation": True,
            "weight_block_size": [2, 2],
        }
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    result = export_qarl_directory_to_fp8(source, output, weight_block_size=(2, 2))

    assert result.qarl_folded is True
    assert result.qarl_modules == ["model.layers.0.self_attn.q_proj"]
    assert result.qarl_state_tensors == 5
    assert result.quantized_weights == 1
    exported = load_file(output / "model.safetensors")
    assert exported["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert exported["model.layers.0.self_attn.q_proj.weight_scale_inv"].shape == (2, 2)
    assert exported["model.layers.0.self_attn.q_proj.bias"].dtype == torch.float32
    assert exported["model.layers.1.self_attn.q_proj.weight"].dtype == torch.float32
    assert exported["lm_head.weight"].dtype == torch.float32
    assert all("qarl_" not in name for name in exported)

    exported_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert set(exported_config["quantization_config"]["modules_to_not_convert"]) == {
        "model.layers.1.self_attn.q_proj",
        "lm_head",
    }
    assert exported_config["xorl_qarl_export"]["source"] == "qarl_folded"
    assert exported_config["xorl_qarl_export"]["folded_modules"] == ["model.layers.0.self_attn.q_proj"]
    assert exported_config["xorl_qarl_export"]["quant_cfg"]["weight_block_size"] == [2, 2]


def test_export_qarl_directory_to_fp8_preserves_trained_logprobs_after_dequant(tmp_path):
    torch.manual_seed(19)
    source = tmp_path / "qarl_source"
    output = tmp_path / "qarl_exported"
    model = TinyExportLogprobModel()
    quant_cfg = {
        "format": "fp8_e4m3",
        "weight": True,
        "activation": False,
        "weight_block_size": [4, 4],
    }
    changed = inject_qarl_into_model(
        model,
        quant_cfg=quant_cfg,
        target_modules=["proj", "lm_head"],
    )
    assert changed == 2

    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    initial_logprobs = _target_logprobs(model, input_ids, labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    for _step in range(8):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    trained_logprobs = _target_logprobs(model, input_ids, labels)
    assert torch.isfinite(trained_logprobs).all()
    assert (trained_logprobs - initial_logprobs).abs().max().item() > 1e-3

    _write_source_model(
        source,
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        num_hidden_layers=1,
        config_updates={"xorl_qarl_config": {"quant_cfg": quant_cfg}},
    )

    result = export_qarl_directory_to_fp8(source, output, weight_block_size=(4, 4))

    assert result.qarl_folded is True
    assert result.qarl_modules == ["lm_head", "model.layers.0.proj"]
    exported = load_file(output / "model.safetensors")
    folded = TinyExportLogprobModel()
    with torch.no_grad():
        folded.embed_tokens.weight.copy_(exported["embed_tokens.weight"])
        folded.model.layers[0]["proj"].weight.copy_(
            _dequantize_block_fp8(
                exported["model.layers.0.proj.weight"],
                exported["model.layers.0.proj.weight_scale_inv"],
                (4, 4),
            )
        )
        folded.model.layers[0]["proj"].bias.copy_(exported["model.layers.0.proj.bias"])
        folded.lm_head.weight.copy_(
            _dequantize_block_fp8(
                exported["lm_head.weight"],
                exported["lm_head.weight_scale_inv"],
                (4, 4),
            )
        )
        folded.lm_head.bias.copy_(exported["lm_head.bias"])

    folded_logprobs = _target_logprobs(folded, input_ids, labels)
    torch.testing.assert_close(folded_logprobs, trained_logprobs, rtol=0, atol=0)


def test_export_qarl_directory_requires_matching_block_size(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "exported"
    _write_source_model(
        source,
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
            "model.layers.0.self_attn.q_proj.qarl_input_amax": torch.tensor(1.0),
            "model.layers.0.self_attn.q_proj.qarl_weight_amax": torch.tensor(1.0),
            "model.layers.0.self_attn.q_proj.qarl_forward_count": torch.tensor(1, dtype=torch.long),
        },
        num_hidden_layers=1,
    )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["xorl_qarl_config"] = {"quant_cfg": {"format": "fp8_e4m3", "weight_block_size": [4, 4]}}
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="weight_block_size must match"):
        export_qarl_directory_to_fp8(source, output, weight_block_size=(2, 2))


def test_parse_size_bytes_accepts_decimal_and_binary_units():
    assert _parse_size_bytes("5GB") == 5_000_000_000
    assert _parse_size_bytes("2MiB") == 2 * 1024 * 1024
    assert _parse_size_bytes("17") == 17
