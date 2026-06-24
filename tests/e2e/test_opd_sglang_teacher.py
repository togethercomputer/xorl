"""End-to-end OPD coverage with a real xorl-sglang-internal teacher."""

import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import torch

from .e2e_utils import create_tiny_model_dir
from .server_utils import (
    ServerProcess,
    _get_free_port,
    _start_server_or_fail,
    generate_server_config,
    get_sglang_paths,
    post_and_wait_for_future,
    scalar,
)


SGLANG_PYTHON, SGLANG_SOURCE_DIR = get_sglang_paths()


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.server,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        SGLANG_PYTHON is None or not SGLANG_PYTHON.exists(),
        reason="Set XORL_SGLANG_PYTHON or XORL_SGLANG_INTERNAL_DIR",
    ),
    pytest.mark.skipif(
        SGLANG_SOURCE_DIR is None or not SGLANG_SOURCE_DIR.exists(),
        reason="Set XORL_SGLANG_SOURCE_DIR or XORL_SGLANG_INTERNAL_DIR",
    ),
]


def _visible_teacher_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return visible.split(",")[-1].strip()
    return str(max(torch.cuda.device_count() - 1, 0))


def _run_sglang_teacher_cache(
    model_dir: str, token_sequences: list[list[int]], output_dir: Path
) -> tuple[Path, list[list[int]]]:
    assert SGLANG_PYTHON is not None
    assert SGLANG_SOURCE_DIR is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "teacher_tokens.json"
    hidden_path = output_dir / "teacher_hidden.safetensors"
    metadata_path = output_dir / "teacher_cache_metadata.json"
    script_path = output_dir / "build_teacher_cache.py"

    tokens_path.write_text(json.dumps(token_sequences), encoding="utf-8")
    script_path.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            import torch
            import sglang as sgl
            from safetensors.torch import save_file


            def main():
                model_dir = sys.argv[1]
                tokens_path = Path(sys.argv[2])
                hidden_path = Path(sys.argv[3])
                metadata_path = Path(sys.argv[4])

                token_sequences = json.loads(tokens_path.read_text(encoding="utf-8"))
                hidden_size = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))["hidden_size"]
                engine = sgl.Engine(
                    model_path=model_dir,
                    tokenizer_path=model_dir,
                    enable_return_hidden_states=True,
                    disable_cuda_graph=True,
                    disable_piecewise_cuda_graph=True,
                    disable_radix_cache=True,
                    attention_backend="triton",
                    dtype="bfloat16",
                    mem_fraction_static=0.2,
                    max_total_tokens=128,
                    log_level="error",
                )

                hidden_chunks = []
                cache_indices_by_sample = []
                offset = 0
                try:
                    for input_ids in token_sequences:
                        output = engine.generate(
                            input_ids=input_ids,
                            sampling_params={"temperature": 0, "max_new_tokens": 1},
                            return_hidden_states=True,
                        )
                        hidden_state_chunks = output["meta_info"]["hidden_states"]
                        if not hidden_state_chunks:
                            raise RuntimeError("SGLang did not return teacher hidden states")
                        hidden = torch.tensor(hidden_state_chunks[0], dtype=torch.bfloat16)
                        if hidden.shape != (len(input_ids), hidden_size):
                            raise RuntimeError(
                                f"Unexpected hidden state shape {tuple(hidden.shape)} for input length {len(input_ids)}"
                            )
                        hidden_chunks.append(hidden.cpu())
                        cache_indices_by_sample.append(list(range(offset, offset + len(input_ids))))
                        offset += len(input_ids)
                finally:
                    engine.shutdown()

                save_file({"hidden_states": torch.cat(hidden_chunks, dim=0).contiguous()}, str(hidden_path))
                metadata_path.write_text(
                    json.dumps({"cache_indices_by_sample": cache_indices_by_sample}, indent=2),
                    encoding="utf-8",
                )


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_SOURCE_DIR}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = _visible_teacher_gpu()

    result = subprocess.run(
        [str(SGLANG_PYTHON), str(script_path), model_dir, str(tokens_path), str(hidden_path), str(metadata_path)],
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    if result.returncode != 0:
        raise AssertionError(
            "xorl-sglang-internal teacher cache generation failed\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return hidden_path, metadata["cache_indices_by_sample"]


def test_opd_server_forward_backward_with_xorl_sglang_teacher(tmp_path):
    torch.manual_seed(2026)

    student_model_dir = create_tiny_model_dir(str(tmp_path / "student"), model_type="dense", save_weights=True)
    teacher_model_dir = create_tiny_model_dir(str(tmp_path / "teacher"), model_type="dense", save_weights=True)

    token_sequences = [
        [11, 12, 13, 14],
        [21, 22, 23],
        [31, 32, 33, 34, 35],
    ]
    teacher_hidden_path, cache_indices = _run_sglang_teacher_cache(
        teacher_model_dir,
        token_sequences,
        tmp_path / "teacher_artifacts",
    )

    output_dir = tmp_path / "xorl_server"
    config_path = generate_server_config(
        student_model_dir,
        str(output_dir),
        num_gpus=1,
        enable_lora=False,
        enable_gradient_checkpointing=False,
        sample_packing_sequence_len=32,
        extra_config={
            "enable_full_shard": False,
            "worker_connection_timeout": 180.0,
        },
    )
    server = ServerProcess(config_path=config_path, num_gpus=1, api_port=_get_free_port(), output_dir=str(output_dir))

    try:
        _start_server_or_fail(server, timeout=240.0)

        data = []
        for tokens, indices in zip(token_sequences, cache_indices):
            data.append(
                {
                    "model_input": {"input_ids": tokens},
                    "loss_fn_inputs": {
                        "target_tokens": tokens,
                        "teacher_ids": [0] * len(tokens),
                        "teacher_weights": [1.0] * len(tokens),
                        "teacher_cache_indices": indices,
                    },
                }
            )

        forward_backward = post_and_wait_for_future(
            server.base_url,
            "/api/v1/forward_backward",
            {
                "model_id": "default",
                "forward_backward_input": {
                    "data": data,
                    "loss_fn": "opd_loss",
                    "loss_fn_params": {
                        "teacher_heads": {"0": teacher_model_dir},
                        "teacher_hidden_caches": {"0": str(teacher_hidden_path)},
                        "opd_sort_by_teacher": True,
                        "num_chunks": 8,
                    },
                },
            },
            request_timeout=30.0,
            future_timeout=240.0,
        )

        loss = scalar(forward_backward["loss_fn_outputs"][0]["loss"])
        assert math.isfinite(loss)
        assert loss >= 0.0
        assert forward_backward["metrics"]["valid_tokens:sum"] == sum(len(tokens) for tokens in token_sequences)

        optim_step = post_and_wait_for_future(
            server.base_url,
            "/api/v1/optim_step",
            {
                "model_id": "default",
                "adam_params": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
                "gradient_clip": 1.0,
            },
            request_timeout=30.0,
            future_timeout=240.0,
        )
        assert math.isfinite(scalar(optim_step["metrics"]["grad_norm"]))
    finally:
        server.stop()
