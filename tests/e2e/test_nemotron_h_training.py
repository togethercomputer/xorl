"""E2E training smokes for Nemotron-3 (nemotron_h) through the CLI trainer.

Covers the full trainer stack on a tiny hybrid model with all three block
types (mamba / attention / moe): packed-varlen data path (cu_seq_lens_q →
mamba cu_seqlens), FSDP2 wrapping, gradient checkpointing
(MoEGradientCheckpointingLayer via the NemotronHBlock mlp→mixer alias),
bf16 mixed precision, adamw + muon optimizer steps, and DCP checkpoint save.

This file doubles as a torchrun worker (``__main__``) that runs the real
``Trainer`` in-process to assert gradients flow to mamba / attention /
expert / router-adjacent parameters.
"""

# ruff: noqa: E402

import json
import math
import os
import subprocess
import sys


# In torchrun worker mode this file runs as a script (script dir on sys.path,
# repo root not), so make the `tests.*` imports work in both modes.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from tests.e2e.e2e_utils import (
    generate_training_config,
    run_training,
    skip_if_gpu_count_less_than,
    write_tokenized_dataset,
)
from tests.e2e.server_utils import _get_free_port


pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

_GRAD_FLOW_OK_MARKER = "NEMOTRON_GRAD_FLOW_OK"

# Parameter-name suffixes that must receive a nonzero gradient in the first
# optimizer step (one representative per mixer type, plus the MoE latent and
# shared-expert projections).
_GRAD_FLOW_TARGETS = {
    "mamba_in_proj": ("mixer.in_proj.weight",),
    "mamba_A_log": ("mixer.A_log",),
    "attention_qkv": ("mixer.q_proj.weight", "mixer.qkv_proj.weight"),
    "attention_o_proj": ("mixer.o_proj.weight",),
    "expert_up": ("mixer.experts.gate_up_proj",),
    "expert_down": ("mixer.experts.down_proj",),
    "moe_latent_fc1": ("mixer.fc1_latent_proj.weight",),
    "shared_expert_up": ("mixer.shared_experts.up_proj.weight",),
    "embeddings": ("model.embeddings.weight",),
}


def _write_memorizable_dataset(output_path: str, *, vocab_size: int = 2048, repeats: int = 64) -> str:
    """Write a handful of fixed variable-length sequences, repeated many times.

    Variable lengths force multiple packing boundaries inside each packed
    sequence; repetition makes the batch memorizable so loss must drop.
    """
    lengths = [97, 65, 49, 33]
    base = []
    for seq_idx, length in enumerate(lengths):
        tokens = [((seq_idx * 131 + pos * 7) % (vocab_size - 3)) + 2 for pos in range(length)]
        tokens[-1] = 0
        base.append((tokens, tokens[:]))
    samples = [sample for _ in range(repeats) for sample in base]
    return write_tokenized_dataset(samples, output_path)


def _nemotron_training_config(
    model_dir: str,
    output_dir: str,
    dataset_path: str,
    *,
    optimizer: str = "adamw",
    lr: float = 1e-3,
    max_steps: int = 15,
    num_gpus: int = 1,
    save_steps: int = 0,
) -> str:
    return generate_training_config(
        model_dir=model_dir,
        output_dir=output_dir,
        attn_implementation="flash_attention_3",
        num_gpus=num_gpus,
        seq_len=128,
        packing_seq_len=256,
        max_steps=max_steps,
        micro_batch_size=1,
        optimizer=optimizer,
        lr=lr,
        save_steps=save_steps,
        enable_gradient_checkpointing=True,
        extra_data={
            "datasets": [{"path": dataset_path, "type": "tokenized", "max_seq_len": 128}],
            "dataset_num_proc": 1,
        },
    )


def _visible_free_gpu_count(min_free_mib: int = 10000) -> int:
    """Count CUDA_VISIBLE_DEVICES entries with at least ``min_free_mib`` free."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0

    free_by_index = {}
    for line in result.stdout.strip().splitlines():
        index_str, free_str = (part.strip() for part in line.split(","))
        free_by_index[int(index_str)] = int(free_str)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        indices = sorted(free_by_index)
    else:
        try:
            indices = [int(part) for part in visible.split(",") if part.strip() != ""]
        except ValueError:
            return 0
    return sum(1 for index in indices if free_by_index.get(index, 0) >= min_free_mib)


class TestNemotronHTraining:
    @skip_if_gpu_count_less_than(1)
    @pytest.mark.parametrize(
        "optimizer,lr",
        [("adamw", 1e-3), ("muon", 2e-3)],
        ids=["adamw", "muon"],
    )
    def test_cli_training_loss_decreases(self, tiny_nemotron_h_model_dir, optimizer, lr):
        """Trainer: packed bf16 FSDP2 training memorizes a tiny batch and saves DCP."""
        max_steps = 15
        save_steps = 10
        output_dir = os.path.join(tiny_nemotron_h_model_dir, f"output_{optimizer}")
        dataset_path = _write_memorizable_dataset(os.path.join(output_dir, "train.jsonl"))
        config_path = _nemotron_training_config(
            tiny_nemotron_h_model_dir,
            output_dir,
            dataset_path,
            optimizer=optimizer,
            lr=lr,
            max_steps=max_steps,
            save_steps=save_steps,
        )

        result = run_training(config_path, num_gpus=1, timeout=900)

        result.assert_success()
        assert result.global_step == max_steps
        assert result.loss_history is not None and len(result.loss_history) == max_steps
        assert all(math.isfinite(loss) for loss in result.loss_history)
        assert result.final_grad_norm is not None and math.isfinite(result.final_grad_norm)
        result.assert_loss_converged(max_final_loss=7.0, min_drop_ratio=0.1)
        ckpt_path = os.path.join(output_dir, "checkpoints", f"global_step_{save_steps}")
        assert os.path.isdir(ckpt_path), f"DCP checkpoint missing: {ckpt_path}"

    @skip_if_gpu_count_less_than(1)
    def test_grad_flow_through_trainer(self, tiny_nemotron_h_model_dir):
        """Trainer (in-process): first optimizer step has nonzero grads on every mixer type."""
        output_dir = os.path.join(tiny_nemotron_h_model_dir, "output_grad_flow")
        dataset_path = _write_memorizable_dataset(os.path.join(output_dir, "train.jsonl"))
        config_path = _nemotron_training_config(
            tiny_nemotron_h_model_dir,
            output_dir,
            dataset_path,
            max_steps=2,
        )

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "1",
            "--master_port",
            str(_get_free_port()),
            os.path.abspath(__file__),
            config_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

        if result.returncode != 0 or _GRAD_FLOW_OK_MARKER not in result.stdout:
            stdout_tail = "\n".join(result.stdout.splitlines()[-80:])
            stderr_tail = "\n".join(result.stderr.splitlines()[-80:])
            raise AssertionError(
                f"Grad-flow worker failed (exit_code={result.returncode})\n"
                f"--- stdout (last 80 lines) ---\n{stdout_tail}\n"
                f"--- stderr (last 80 lines) ---\n{stderr_tail}"
            )

    @skip_if_gpu_count_less_than(2)
    def test_cli_training_fsdp2_dp2(self, tiny_nemotron_h_model_dir):
        """Trainer: 2-GPU FSDP2 (dp_shard=2) run with packing + gradient checkpointing."""
        if _visible_free_gpu_count() < 2:
            pytest.skip("Fewer than 2 visible GPUs with enough free memory")
        max_steps = 5
        output_dir = os.path.join(tiny_nemotron_h_model_dir, "output_dp2")
        dataset_path = _write_memorizable_dataset(os.path.join(output_dir, "train.jsonl"))
        config_path = _nemotron_training_config(
            tiny_nemotron_h_model_dir,
            output_dir,
            dataset_path,
            max_steps=max_steps,
            num_gpus=2,
        )

        result = run_training(config_path, num_gpus=2, timeout=900)

        result.assert_success()
        assert result.global_step == max_steps
        assert result.loss_history is not None and len(result.loss_history) == max_steps
        assert all(math.isfinite(loss) for loss in result.loss_history)
        assert result.final_grad_norm is not None and math.isfinite(result.final_grad_norm)


# ---------------------------------------------------------------------------
# torchrun worker: real Trainer in-process, grad capture at first step
# ---------------------------------------------------------------------------


def _local_grad_abs_sum(param) -> float:
    grad = param.grad
    if grad is None:
        return 0.0
    local = grad.to_local() if hasattr(grad, "to_local") else grad
    if local.numel() == 0:
        return 0.0
    total = local.abs().sum().item()
    if not math.isfinite(total):
        raise RuntimeError(f"non-finite gradient (abs sum={total})")
    return total


def _grad_flow_worker() -> int:
    from xorl.arguments import Arguments, parse_args  # noqa: PLC0415
    from xorl.ops.ssm import Mamba2Mixer  # noqa: PLC0415
    from xorl.trainers import Trainer  # noqa: PLC0415

    args = parse_args(Arguments)
    trainer = Trainer(args)

    seen: dict = {}
    max_cu_seqlens_docs = 0

    def _record_cu_seqlens(module, hook_args, hook_kwargs):
        del module, hook_args
        cu_seqlens = hook_kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            nonlocal max_cu_seqlens_docs
            max_cu_seqlens_docs = max(max_cu_seqlens_docs, int(cu_seqlens.numel()) - 1)

    for module in trainer.model.modules():
        if isinstance(module, Mamba2Mixer):
            module.register_forward_pre_hook(_record_cu_seqlens, with_kwargs=True)
            break

    def _capture(optimizer, opt_args, opt_kwargs):
        del optimizer, opt_args, opt_kwargs
        if seen:
            return
        for name, param in trainer.model.named_parameters():
            total = _local_grad_abs_sum(param)
            for kind, suffixes in _GRAD_FLOW_TARGETS.items():
                if any(name.endswith(suffix) for suffix in suffixes):
                    seen[kind] = max(seen.get(kind, 0.0), total)

    trainer.optimizer.register_step_pre_hook(_capture)
    trainer.train()

    missing = sorted(kind for kind in _GRAD_FLOW_TARGETS if seen.get(kind, 0.0) <= 0.0)
    if missing:
        print(f"NEMOTRON_GRAD_FLOW_MISSING {missing} (seen={seen})", flush=True)
        return 1
    if max_cu_seqlens_docs < 2:
        # Packed varlen plumbing: cu_seq_lens_q from the packing collator must reach
        # the mamba mixers as cu_seqlens with multiple documents per pack.
        print(f"NEMOTRON_GRAD_FLOW_NO_PACKED_CU_SEQLENS (max docs per pack={max_cu_seqlens_docs})", flush=True)
        return 1
    seen["max_cu_seqlens_docs"] = max_cu_seqlens_docs
    print(f"{_GRAD_FLOW_OK_MARKER} {json.dumps(seen)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_grad_flow_worker())
