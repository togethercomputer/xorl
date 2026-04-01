import os


# Must be set before importing torch / initializing CUDA so the
# allocator picks up the setting on first use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from xorl.arguments import Arguments, parse_args
from xorl.trainers import Trainer


def main():
    args = parse_args(Arguments)
    Trainer(args).train()


if __name__ == "__main__":
    main()
