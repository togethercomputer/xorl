# ruff: noqa: E402

import os

from xorl.utils.compile_cache import configure_rank_local_compile_caches


# Must be set before importing torch / initializing CUDA so the
# allocator picks up the setting on first use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
configure_rank_local_compile_caches()

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
