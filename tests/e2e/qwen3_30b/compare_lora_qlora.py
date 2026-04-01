#!/usr/bin/env python3
"""Compare QLoRA NVFP4 vs QLoRA NF4 on Qwen3-30B-A3B (real weights).

Runs both methods on 8x H100 with identical hyperparameters and compares
loss convergence. Uses EP=8, Ulysses SP=8, dp_shard=1, load_weights_mode=all_ranks.
"""

import math
import os
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tests.e2e.e2e_utils import (
    generate_training_config,
    run_training,
)


def main():
    max_steps = 100
    lr = 2e-5
    lora_rank = 64
    lora_alpha = 64
    num_gpus = 8
    ep_size = 8
    ulysses_size = 8
    dp_shard_size = 1
    seq_len = 2048
    packing_seq_len = 4096
    micro_batch_size = 1
    gradient_accumulation_steps = 1

    bf16_model = "Qwen/Qwen3-30B-A3B"
    nvfp4_model = "nvidia/Qwen3-30B-A3B-NVFP4"

    extra_train_common = {
        "load_weights_mode": "all_ranks",
    }

    with tempfile.TemporaryDirectory(prefix="qwen30b_compare_") as tmpdir:
        configs = {
            "QLoRA NVFP4": dict(
                model_dir=nvfp4_model,
                model_path=nvfp4_model,
                enable_qlora=True,
                quant_format="nvfp4",
            ),
            "QLoRA NF4": dict(
                model_dir=bf16_model,
                model_path=bf16_model,
                enable_qlora=True,
                quant_format="nf4",
            ),
        }

        results = {}
        for name, extra in configs.items():
            print("=" * 60)
            print(f"Running {name} training...")
            print("=" * 60)
            slug = name.lower().replace(" ", "_")
            output_dir = os.path.join(tmpdir, f"output_{slug}")
            extra = extra.copy()
            model_dir = extra.pop("model_dir")
            model_path = extra.pop("model_path")
            config_path = generate_training_config(
                model_dir=model_dir,
                model_path=model_path,
                output_dir=output_dir,
                num_gpus=num_gpus,
                max_steps=max_steps,
                lr=lr,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                seq_len=seq_len,
                packing_seq_len=packing_seq_len,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                ep_size=ep_size,
                ulysses_size=ulysses_size,
                dp_shard_size=dp_shard_size,
                merge_qkv=False,
                moe_implementation="triton",
                extra_train=extra_train_common,
                **extra,
            )
            results[name] = run_training(
                config_path,
                num_gpus=num_gpus,
                timeout=3600,
            )

        # --- Report ---
        print("\n" + "=" * 70)
        print("RESULTS COMPARISON: Qwen3-30B-A3B QLoRA (8x H100)")
        print("=" * 70)

        for name, result in results.items():
            print(f"\n--- {name} ---")
            print(f"  Exit code:  {result.exit_code}")
            if result.metrics:
                print(f"  Steps:      {result.global_step}")
                print(f"  Final loss: {result.final_loss:.4f}")
                history = result.loss_history
                if history and len(history) >= 2:
                    drop = (history[0] - history[-1]) / history[0]
                    print(f"  First loss: {history[0]:.4f}")
                    print(f"  Loss drop:  {drop:.2%}")
                    every = max(1, len(history) // 10)
                    sampled = [history[i] for i in range(0, len(history), every)]
                    if history[-1] not in sampled:
                        sampled.append(history[-1])
                    print(f"  Loss curve: {[f'{l:.3f}' for l in sampled]}")
            else:
                print("  No metrics (training failed)")
                stderr_tail = "\n".join(result.stderr.splitlines()[-30:])
                print(f"  stderr tail:\n{stderr_tail}")

        # --- Summary ---
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        all_ok = True
        for name, result in results.items():
            ok = result.success and result.final_loss is not None and not math.isnan(result.final_loss)
            status = "PASS" if ok else "FAIL"
            loss_str = f"{result.final_loss:.4f}" if result.final_loss is not None else "N/A"
            print(f"  [{status}] {name:20s}  final_loss={loss_str}")
            if not ok:
                all_ok = False

        # Compare NF4 vs NVFP4
        nvfp4 = results.get("QLoRA NVFP4")
        nf4 = results.get("QLoRA NF4")
        if nvfp4 and nf4 and nvfp4.final_loss and nf4.final_loss:
            diff = abs(nf4.final_loss - nvfp4.final_loss)
            print(f"\n  NF4 vs NVFP4 final loss diff: {diff:.4f}")
            if nf4.loss_history and nvfp4.loss_history:
                nf4_drop = (nf4.loss_history[0] - nf4.loss_history[-1]) / nf4.loss_history[0]
                nvfp4_drop = (nvfp4.loss_history[0] - nvfp4.loss_history[-1]) / nvfp4.loss_history[0]
                print(f"  NF4 loss drop:   {nf4_drop:.2%}")
                print(f"  NVFP4 loss drop: {nvfp4_drop:.2%}")

        if all_ok:
            print("\nAll methods trained successfully!")
        else:
            print("\nSome methods failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
