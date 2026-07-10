"""v3 final scoreboard: nano/small + deep-narrow + S-sweep vs the hardened baseline.

The flag-planting configs for the latency regime (single-sequence training step):
bench.py's nano/small, a deep-narrow L=12 stack (chain-depth stress), and an S sweep
at nano width (where the boundary-overhead pool shrinks with S).
"""

import sys

import torch
from bench import bench_cfg
from model import Cfg


if __name__ == "__main__":
    torch.cuda.set_device(0)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = {
        "nano": Cfg(),  # H256 L4 S512
        "small": Cfg(H=512, L=8, nq=8, nkv=4, D=64, I=1536, V=16384, S=1024),
        "deep": Cfg(L=12),  # deep-narrow: chain depth x3
        "s128": Cfg(S=128),
        "s256": Cfg(S=256),
        "s1024": Cfg(S=1024),
    }
    for name, cfg in cfgs.items():
        if which in ("all", name):
            print(f"--- {name} ---")
            bench_cfg(cfg)
