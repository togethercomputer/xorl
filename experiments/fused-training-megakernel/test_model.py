"""Gradient-parity test: the fused megakernel vs a pure-PyTorch fp32 Qwen3 reference.

Checks loss + every weight gradient, determinism across reruns, and that a short
training loop driven ONLY by megakernel gradients actually learns.

Run: CUDA_VISIBLE_DEVICES=<idle> <fa4-venv>/bin/python test_model.py
"""

import torch
import torch.nn.functional as F
from model import Cfg, MKQwen3


def ref_loss_and_grads(m: MKQwen3, tokens, labels):
    """fp32 reference implementing exactly the megakernel's math on the same params."""
    c = m.cfg
    P = {k: v.float().detach().requires_grad_(True) for k, v in m.params.items()}
    cos, sin = m.cos, m.sin

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c.eps) * w

    def rope(t):  # t: [S, h, D]
        a, b = t[..., : c.D // 2], t[..., c.D // 2 :]
        cc, ss = cos[:, None, :], sin[:, None, :]
        return torch.cat([a * cc - b * ss, b * cc + a * ss], dim=-1)

    x = P["emb"][tokens.long()]
    for l in range(c.L):
        xn = rms(x, P[f"w1.{l}"])
        qkv = xn @ P[f"wqkv.{l}"].T
        qkv = qkv.view(c.S, c.nq + 2 * c.nkv, c.D)
        q, k, v = qkv[:, : c.nq], qkv[:, c.nq : c.nq + c.nkv], qkv[:, c.nq + c.nkv :]
        q = rope(rms(q, P[f"qn.{l}"])).permute(1, 0, 2)
        k = rope(rms(k, P[f"kn.{l}"])).permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        k = k.repeat_interleave(c.nq // c.nkv, dim=0)
        v = v.repeat_interleave(c.nq // c.nkv, dim=0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.permute(1, 0, 2).reshape(c.S, c.nq * c.D)
        x = x + o @ P[f"wo.{l}"].T
        xn2 = rms(x, P[f"w2.{l}"])
        gu = xn2 @ P[f"wgu.{l}"].T
        g, u = gu.chunk(2, dim=-1)
        x = x + (F.silu(g) * u) @ P[f"wd.{l}"].T
    logits = rms(x, P["wf"]) @ P["wlm"].T
    loss = F.cross_entropy(logits, labels.long(), ignore_index=-100)
    loss.backward()
    return loss.detach(), {k: t.grad for k, t in P.items()}


def compare(m, ref_grads, verbose=True):
    worst = ("", 0.0)
    for k, g in m.grads.items():
        r = ref_grads[k]
        scale = r.abs().max().item() + 1e-8
        err = (g - r).abs().max().item() / scale
        if verbose:
            print(f"  d{k:10s} rel_err={err:.4f} (ref_max={scale:.3e})")
        if err > worst[1]:
            worst = (k, err)
    return worst


def main():
    torch.cuda.set_device(0)

    for cfg in (
        Cfg(),  # nano: H256 L4 nq4/nkv2 D64 I768 V8192 S512
        Cfg(H=256, L=2, nq=4, nkv=2, D=128, I=512, V=4096, S=192),  # D=128 + ragged S
    ):
        print(f"=== cfg {cfg} ===")
        m = MKQwen3(cfg, seed=0)
        torch.manual_seed(1)
        tokens = torch.randint(0, cfg.V, (cfg.S,), device="cuda", dtype=torch.int32)
        labels = torch.roll(tokens, -1).to(torch.int32)
        labels[-1] = -100
        labels[3] = -100

        loss = m.step(tokens, labels)
        torch.cuda.synchronize()
        loss1 = loss.item()
        ref_l, ref_g = ref_loss_and_grads(m, tokens, labels)
        print(f"loss megakernel={loss1:.5f} ref={ref_l.item():.5f} (waves={m.n_waves})")
        assert abs(loss1 - ref_l.item()) / ref_l.item() < 2e-3, "loss mismatch"

        worst = compare(m, ref_g)
        print(f"worst grad rel err: {worst[0]} {worst[1]:.4f}")
        assert worst[1] < 0.03, f"grad mismatch: {worst}"

        # rerun stability: same inputs -> same results up to fp32-atomic summation order.
        # The attention backward accumulates dq/dk/dv via fp32 atomics feeding bf16
        # activations (like FA's deterministic=False backward), so downstream grads can
        # flip bf16 ulps run-to-run: tolerance is ulp-scale, not bitwise.
        g0 = {k: v.clone() for k, v in m.grads.items()}
        loss2 = m.step(tokens, labels)
        torch.cuda.synchronize()
        assert abs(loss2.item() - loss1) < 1e-4 * abs(loss1), "rerun loss drifted"
        for k in g0:
            ref = g0[k].abs().max().item() + 1e-8
            assert (g0[k] - m.grads[k]).abs().max().item() < 2e-2 * ref, f"rerun grad drifted: {k}"
        print("rerun stable OK")

        # wave-mode executor must agree with dataflow mode (same tolerance class)
        loss3 = m.step(tokens, labels, mode="waves")
        torch.cuda.synchronize()
        assert abs(loss3.item() - loss1) < 1e-4 * abs(loss1), "waves-vs-df loss drifted"
        for k in g0:
            ref = g0[k].abs().max().item() + 1e-8
            assert (g0[k] - m.grads[k]).abs().max().item() < 2e-2 * ref, f"waves-vs-df grad: {k}"
        print(f"waves-vs-df agreement OK (critical path {m.prog.critical_path} of {m.prog.n_instr} instrs)")

    # ---- learning sanity: SGD on megakernel grads must drive loss down ----
    cfg = Cfg()
    m = MKQwen3(cfg, seed=0)
    torch.manual_seed(2)
    tokens = torch.randint(0, cfg.V, (cfg.S,), device="cuda", dtype=torch.int32)
    labels = torch.roll(tokens, -1).to(torch.int32)
    labels[-1] = -100
    first = last = None
    for it in range(40):
        loss = m.step(tokens, labels)
        torch.cuda.synchronize()
        if it == 0:
            first = loss.item()
        last = loss.item()
        with torch.no_grad():
            for k, p in m.params.items():
                p.add_((-0.5 * m.grads[k]).to(torch.bfloat16))
    print(f"training sanity: loss {first:.4f} -> {last:.4f} over 40 SGD steps")
    assert last < first - 2.0, "megakernel gradients do not learn"
    print("ALL MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
