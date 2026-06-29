"""Minimal FSDP2 + compiled-autograd + whole-model torch.compile repro.

Isolates the "data is not allocated yet" RuntimeError seen in the xorl whole-step path:
the compiled-autograd Inductor BACKWARD references an FSDP unsharded-param buffer (the
backward re-all-gather output, since non-root units reshard after forward) before it is
allocated. Pure torch — no xorl, no flash_attn — so it runs under any torch (2.10 or nightly).

Run: torchrun --standalone --nproc_per_node=2 scratch_fsdp_ca_repro.py
Prints per-iter "OK" and final "ALL OK", or the exception.
"""

import os
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

import torch._dynamo.compiled_autograd as ca

H = 2048  # matches the failing mm: (·,2048) @ (2048, 6144=3H)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(H, 3 * H, bias=False)
        self.fc2 = nn.Linear(3 * H, H, bias=False)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


class Model(nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.head = nn.Linear(H, H, bias=False)  # lm_head analog (its own fully_shard unit)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    # SKIP_HOOKS=1 leaves the default (True). Default here: False (Traceable FSDP2 — trace hooks).
    torch._dynamo.config.skip_fsdp_hooks = os.environ.get("SKIP_HOOKS") == "1"

    torch.manual_seed(0)
    model = Model().to(dev)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # RESHARD=0 → reshard_after_forward=False (keep params gathered into backward; avoids the
    # backward re-all-gather buffer that trips "not allocated"). Default: FSDP default (True non-root).
    fs_kwargs = {"mp_policy": mp}
    if os.environ.get("RESHARD") == "0":
        fs_kwargs["reshard_after_forward"] = False
    for b in model.blocks:
        fully_shard(b, **fs_kwargs)
    fully_shard(model.head, **fs_kwargs)
    fully_shard(model, **fs_kwargs)

    cmodel = torch.compile(model)

    def compiler_fn(gm):
        return torch.compile(gm, backend="inductor", fullgraph=True)

    # compiled-autograd enable context: _enable in 2.10; nightly may expose public enable.
    ca_enable = getattr(ca, "_enable", None) or getattr(ca, "enable")

    fwd_only = os.environ.get("FWD_ONLY") == "1"
    from torch._dynamo.utils import counters

    if rank == 0:
        print(f"torch {torch.__version__} | FWD_ONLY={fwd_only} "
              f"skip_fsdp_hooks={torch._dynamo.config.skip_fsdp_hooks} RESHARD={os.environ.get('RESHARD','default')}",
              flush=True)
    for it in range(3):
        x = torch.randn(8, 512, H, device=dev, dtype=torch.bfloat16)
        try:
            with ca_enable(compiler_fn):
                out = cmodel(x)
                if it == 0 and rank == 0:
                    # Break census for the FORWARD compile (the goal). Logged after the first
                    # forward, before any backward, so backward backend bugs don't mask it.
                    breaks = dict(counters.get("graph_break", {}))
                    print(f"FWD graph_break census: total={sum(breaks.values())} reasons={len(breaks)}", flush=True)
                    for r, c in breaks.items():
                        print(f"  [{c}] {r[:160]}", flush=True)
                if not fwd_only:
                    loss = out.float().pow(2).mean()
                    loss.backward()
            if rank == 0:
                msg = "fwd OK" if fwd_only else f"loss={loss.item():.4f} OK"
                print(f"iter {it}: {msg}", flush=True)
        except Exception:
            if rank == 0:
                print(f"iter {it}: FAILED", flush=True)
                traceback.print_exc()
            dist.destroy_process_group()
            raise
        model.zero_grad(set_to_none=True)
    if rank == 0:
        print("ALL OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
