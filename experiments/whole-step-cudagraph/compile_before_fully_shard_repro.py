"""Supported PyTorch 2.12+ pattern: compile compute BEFORE fully_shard, fullgraph=True.

PyTorch 2.12 (#174863, #174906) deprecated "compiling through FSDP2 hooks without graph
breaks" (the skip_fsdp_hooks=False / Traceable-FSDP2-into-one-graph + compiled-autograd path).
The supported replacement is to torch.compile the compute regions BEFORE fully_shard, with
fullgraph=True. FSDP's all-gather / reduce-scatter then run via eager hooks OUTSIDE the compiled
graphs, and the backward is ordinary eager autograd (the compiled regions' bwd is AOT-compiled).

This checks: do the compiled compute regions hit 0 graph breaks, and does fwd+bwd run?

Run: torchrun --standalone --nproc_per_node=2 scratch_fsdp_pre_compile.py
"""

import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard


H = 2048


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
        self.head = nn.Linear(H, H, bias=False)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    torch.manual_seed(0)
    model = Model().to(dev)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # 1) compile each compute region (fullgraph=True) BEFORE fully_shard
    for i in range(len(model.blocks)):
        model.blocks[i] = torch.compile(model.blocks[i], fullgraph=True)
    model.head = torch.compile(model.head, fullgraph=True)
    # 2) then fully_shard (hooks live outside the compiled regions)
    for b in model.blocks:
        fully_shard(b, mp_policy=mp)
    fully_shard(model.head, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    from torch._dynamo.utils import counters  # noqa: PLC0415  (lazy: only for the break census)

    if rank == 0:
        print(f"torch {torch.__version__} | pattern=compile-before-fully_shard fullgraph=True", flush=True)
    for it in range(3):
        x = torch.randn(8, 512, H, device=dev, dtype=torch.bfloat16)
        try:
            out = model(x)  # eager top-level loop calling compiled+sharded blocks
            loss = out.float().pow(2).mean()
            loss.backward()  # ordinary eager autograd (no compiled autograd)
            if it == 0 and rank == 0:
                breaks = dict(counters.get("graph_break", {}))
                print(f"graph_break census: total={sum(breaks.values())} reasons={len(breaks)}", flush=True)
                for r, c in breaks.items():
                    print(f"  [{c}] {r[:160]}", flush=True)
            if rank == 0:
                print(f"iter {it}: loss={loss.item():.4f} OK", flush=True)
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
