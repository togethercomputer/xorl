"""Real Qwen3 (HF) + FA4 + Muon + FSDP2 + manual CUDAGraph capture of the whole fwd+bwd step.

Measures eager vs manual-capture throughput/MFU for the real model on N GPUs. Uses sdpa-shaped FA4
(flash_attn.cute) registered into HF's attention dispatch, Muon (lighter states so 8B fits on fewer
GPUs), FSDP2 reshard_after_forward=False (static buffers for capture).

Env: MODEL=1.7b|8b  MODE=eager|manualgraph  S=<seqlen>  STEPS=<n>
Run: torchrun --standalone --nproc_per_node=N scratch_qwen3_capture.py
"""
import os, time, math, traceback
import torch, torch.nn as nn, torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# ---- FA4 attention registered into HF's dispatch -------------------------------------------------
from flash_attn.cute import flash_attn_func as _fa4

def fa4_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    # HF passes (B, n_heads, S, D); FA4 wants (B, S, n_heads, D), causal.
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out = _fa4(q, k, v, causal=True, softmax_scale=scaling)
    if isinstance(out, tuple):
        out = out[0]
    return out, None  # (attn_output[B,S,H,D], attn_weights)

ALL_ATTENTION_FUNCTIONS["fa4"] = fa4_attention

CONFIGS = {
    "1.7b": dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936),
    "8b": dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
               num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=151936),
}

def build_muon_groups(model):
    # Muon for 2D weight matrices; AdamW for everything else (norms, embeddings, lm_head).
    muon, adamw = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n and "lm_head" not in n:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """Minimal Muon: momentum + Newton-Schulz orthogonalization (2D params)."""
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.get("m")
                if buf is None:
                    buf = st["m"] = torch.zeros_like(p)
                buf.mul_(g["momentum"]).add_(p.grad)
                gr = p.grad.add(buf, alpha=g["momentum"])  # nesterov
                u = zeropower_via_newtonschulz5(gr.float()).to(p.dtype)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(u, alpha=-g["lr"] * scale)

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    name = os.environ.get("MODEL", "1.7b")
    mode = os.environ.get("MODE", "manualgraph")
    S = int(os.environ.get("S", "2048"))
    STEPS = int(os.environ.get("STEPS", "20"))
    cfg = Qwen3Config(**CONFIGS[name], max_position_embeddings=max(S, 4096),
                      attn_implementation="fa4", use_cache=False, tie_word_embeddings=(name == "1.7b"))
    torch.manual_seed(0)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    model = model.to_empty(device=dev)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02) if p.ndim >= 2 else torch.nn.init.zeros_(p)
    model = model.to(torch.bfloat16)
    model.train()

    # Per-layer compile BEFORE fully_shard (0-break pattern): fused kernels, hooks stay outside.
    if os.environ.get("COMPILE", "0") == "1":
        fg = os.environ.get("FULLGRAPH", "0") == "1"
        for i in range(len(model.model.layers)):
            model.model.layers[i] = torch.compile(model.model.layers[i], fullgraph=fg)
        if rank == 0:
            print(f"COMPILE per-layer (fullgraph={fg})", flush=True)

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp, reshard_after_forward=False)
    fully_shard(model, mp_policy=mp, reshard_after_forward=False)

    if os.environ.get("OPT", "muon") == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)]  # fast opt to isolate fwd+bwd
    else:
        from xorl.optim.muon import Muon as XorlMuon  # fast Muon: gram-NS + quack kernels, FSDP2-aware
        muon_p, adamw_p = build_muon_groups(model)
        opts = [XorlMuon([
            {"params": muon_p, "use_muon": True, "lr": 0.02},
            {"params": adamw_p, "use_muon": False, "lr": 3e-4},
        ], distributed_mode="shard_local")]
    def opt_step():
        for o in opts: o.step()
    def opt_zero():
        for o in opts: o.zero_grad(set_to_none=(mode != "manualgraph"))

    B = 1
    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    def mk_batch():
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev, generator=g)
        return ids
    static_ids = torch.zeros(B, S, dtype=torch.long, device=dev)

    def fwd_bwd():
        out = model(input_ids=static_ids, labels=static_ids)
        out.loss.backward()
        return out.loss

    tokens_per_step = B * S * world
    try:
        if mode == "manualgraph":
            static_ids.copy_(mk_batch())
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(4):
                    opt_zero(); fwd_bwd()
            torch.cuda.current_stream().wait_stream(s); dist.barrier()
            opt_zero()  # set_to_none False -> static grads
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_loss = fwd_bwd()
            if rank == 0: print("CAPTURE OK", flush=True)
            def run(ids):
                static_ids.copy_(ids)
                for p in model.parameters():
                    if p.grad is not None: p.grad.zero_()
                graph.replay(); opt_step(); return static_loss
        else:
            def run(ids):
                opt_zero(); static_ids.copy_(ids); loss = fwd_bwd(); opt_step(); return loss

        # Untimed warmup so FA4/cute JIT + any first-step costs are excluded from steady-state.
        for _ in range(5):
            run(mk_batch())
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time(); last = None
        for i in range(STEPS):
            last = run(mk_batch())
        torch.cuda.synchronize()
        dt = (time.time() - t0) / STEPS
        if rank == 0:
            toks = tokens_per_step / dt
            # MFU: 6N + 12*L*h*s per token (attn term uses full S). N excl. embeddings counted in 6N approx.
            N = sum(p.numel() for p in model.parameters())
            L, h = cfg.num_hidden_layers, cfg.hidden_size
            flops_per_tok = 6 * N + 12 * L * h * S
            mfu = flops_per_tok * toks / (world * 989.5e12)
            print(f"MODEL={name} MODE={mode} world={world} S={S}  per-step={dt*1e3:.1f}ms  "
                  f"tok/s={toks/1e3:.1f}k  MFU={mfu*100:.1f}%  loss={float(last.item()):.3f}", flush=True)
    except Exception:
        if rank == 0:
            print(f"MODEL={name} MODE={mode} FAILED", flush=True); traceback.print_exc()
        dist.destroy_process_group(); raise
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
