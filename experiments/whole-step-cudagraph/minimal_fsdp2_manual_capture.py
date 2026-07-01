"""Manual torch.cuda.CUDAGraph capture of a whole FSDP2 fwd+bwd step.

Granularity-agnostic: records the actual kernel stream (compute + NCCL all-gather/reduce-scatter)
into ONE replayable graph, bypassing Inductor's per-region CUDAGraph Trees (which re-record under
FSDP). Tests: (a) does capture succeed with FSDP collectives in the graph, (b) does replay produce
the SAME loss trajectory as eager (correctness), (c) per-step time vs eager.

MODE=eager | manualgraph   RESHARD=0|1 (default 0 = reshard_after_forward=False, static buffers)
Run: torchrun --standalone --nproc_per_node=2 scratch_manual_cudagraph.py
"""

import os
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard


H = int(os.environ.get("H", "2048"))
B = int(os.environ.get("B", "8"))
S = int(os.environ.get("S", "512"))
NLAYERS = int(os.environ.get("NLAYERS", "4"))
NSTEPS = 30


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(H, 3 * H, bias=False)
        self.fc2 = nn.Linear(3 * H, H, bias=False)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))


class Model(nn.Module):
    def __init__(self, n=NLAYERS):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n)])
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
    mode = os.environ.get("MODE", "manualgraph")
    reshard = os.environ.get("RESHARD", "0") == "1"

    torch.manual_seed(0)
    model = Model().to(dev)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for b in model.blocks:
        fully_shard(b, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(model.head, mp_policy=mp, reshard_after_forward=reshard)
    fully_shard(model, mp_policy=mp, reshard_after_forward=reshard)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    # Deterministic input sequence (same across MODE runs for an apples-to-apples loss check).
    g_in = torch.Generator(device=dev).manual_seed(1234)
    inputs = [torch.randn(B, S, H, device=dev, dtype=torch.bfloat16, generator=g_in) for _ in range(NSTEPS)]
    static_in = torch.empty(B, S, H, device=dev, dtype=torch.bfloat16)

    def fwd_bwd():
        out = model(static_in)
        loss = out.float().pow(2).mean()
        loss.backward()
        return loss

    try:
        if mode == "manualgraph":
            # Warmup on a side stream so FSDP lazy-init / cublas workspaces / first all-gather settle
            # BEFORE capture, and the caching allocator is primed.
            static_in.copy_(inputs[0])
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    fwd_bwd()
            torch.cuda.current_stream().wait_stream(s)
            dist.barrier()
            # Capture fwd+bwd. zero_grad(set_to_none=False) first so .grad are static, zeroed buffers.
            opt.zero_grad(set_to_none=False)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_loss = fwd_bwd()
            if rank == 0:
                print("CAPTURE OK", flush=True)

            def run(x):
                static_in.copy_(x)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                graph.replay()
                opt.step()
                return static_loss
        else:

            def run(x):
                opt.zero_grad(set_to_none=True)
                static_in.copy_(x)
                loss = fwd_bwd()
                opt.step()
                return loss

        # Correctness: loss trajectory. Timing: steady-state per-step.
        losses = []
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for i in range(NSTEPS):
            loss = run(inputs[i])
            if i in (0, 1, NSTEPS // 2, NSTEPS - 1):
                losses.append((i, float(loss.item())))
        torch.cuda.synchronize()
        dt = (time.time() - t0) / NSTEPS
        if rank == 0:
            print(f"MODE={mode} reshard={reshard}  per-step={dt * 1e3:.2f}ms  losses={losses}", flush=True)
    except Exception:
        if rank == 0:
            print(f"MODE={mode} FAILED", flush=True)
            traceback.print_exc()
        dist.destroy_process_group()
        raise
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
