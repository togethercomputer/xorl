# ruff: noqa: E402

import os

from xorl.utils.compile_cache import configure_rank_local_compile_caches


# Must be set before importing torch / initializing CUDA so the
# allocator picks up the setting on first use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
configure_rank_local_compile_caches()

# Start optional CUDA memory-history recording before CUDA initialization and
# model construction so allocation traces include persistent model state.
if os.environ.get("XORL_MEMHIST", "0") == "1":
    import torch as _torch

    _torch.cuda.memory._record_memory_history(
        max_entries=int(os.environ.get("XORL_MEMHIST_MAX_ENTRIES", "3000000")),
        stacks="python",
        context="alloc",
    )

# When XORL_TRITON_NO_AUTOTUNE=1, force ALL Triton autotune decorators to
# use only the first config (skip do_bench benchmarking that can OOM).
if os.environ.get("XORL_TRITON_NO_AUTOTUNE", "0") == "1":
    import triton

    _orig_autotune = triton.autotune

    def _single_config_autotune(configs, *args, **kwargs):
        return _orig_autotune([configs[0]], *args, **kwargs)

    triton.autotune = _single_config_autotune


from xorl.arguments import Arguments, parse_args
from xorl.trainers import Trainer


def main():
    args = parse_args(Arguments)
    Trainer(args).train()


if __name__ == "__main__":
    main()
