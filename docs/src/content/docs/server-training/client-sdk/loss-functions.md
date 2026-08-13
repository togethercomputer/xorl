---
title: "Loss Functions"
---

xorl supports multiple loss functions for SFT and RL training, configured via the `loss_fn` parameter in `forward_backward()`.

## Overview

| Algorithm | `loss_fn` | Key params | When to use |
|---|---|---|---|
| SFT / continued pretraining | `causallm_loss` | — | Standard next-token prediction |
| PPO | `policy_loss` | `eps_clip=0.2` | Clipped policy gradient, most stable for RL |
| GRPO (simpler RL) | `importance_sampling` | — | No clipping, simpler but less stable |
| DR-GRPO | `drgrpo` | `clip_low`, `clip_high`, `beta` | Clipped policy loss with optional reference KL |
| Online policy distillation | `opd_loss` | Server OPD configuration | Full-vocabulary or sampled-token teacher/student KL |
| PPO + stale data correction | `policy_loss` | `use_tis=True` | Multiple epochs over same rollout |

---

## Causal LM Loss (`causallm_loss`)

Standard cross-entropy loss for SFT and pretraining:

```python
fwd = client.forward_backward(data, loss_fn="causallm_loss")
```

---

## PPO Policy Loss (`policy_loss`)

Full PPO-style clipped policy gradient loss ([source](https://github.com/togethercomputer/xorl/blob/main/src/xorl/ops/loss/policy_loss.py)):

```
ratio = exp(new_logprobs - old_logprobs)
pg_loss = max(-ratio × A, -clip(ratio, 1-ε, 1+ε_high) × A)
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `eps_clip` | `0.2` | Lower clip ratio |
| `eps_clip_high` | `0.2` | Upper clip ratio (asymmetric clipping) |
| `eps_clip_c` | `null` | Dual-clip for negative advantages (e.g. `3.0`) |
| `compute_kl_stats` | `false` | Return KL statistics in metrics |

**Metrics returned:**

| Metric | Description |
|---|---|
| `pg_clipfrac` | Fraction of tokens where clipping was applied |
| `kl_sample_train_k3` | Schulman's K3 KL estimator |
| `entropy_sample` | Mean entropy of old policy |
| `ratio_mean/min/max` | Importance sampling ratio statistics |

```python
fwd = client.forward_backward(data, loss_fn="policy_loss", loss_fn_params={
    "eps_clip": 0.2,
    "eps_clip_high": 0.2,
    "eps_clip_c": 3.0,
    "compute_kl_stats": True,
})
```

---

## GRPO / Importance Sampling (`importance_sampling`)

Simpler importance-sampling loss ([source](https://github.com/togethercomputer/xorl/blob/main/src/xorl/ops/loss/importance_sampling_loss.py)):

```
ratio = exp(new_logprobs - old_logprobs)
loss = -(ratio × advantages).mean()
```

No clipping — relies on advantages being bounded. Suitable when the policy doesn't drift far from the rollout policy.

```python
fwd = client.forward_backward(data, loss_fn="importance_sampling", loss_fn_params={
    "compute_kl_stats": True,
})
```

---

## IcePop

IcePop (from [GLM-5, arXiv:2602.15763](https://arxiv.org/abs/2602.15763)) is a **hard masking** technique that zeros gradients for tokens where the importance sampling ratio falls outside `[1/β, β]`. Complementary to PPO's soft clipping.

```python
fwd = client.forward_backward(data, loss_fn="policy_loss", loss_fn_params={
    "eps_clip": 0.2,
    "icepop_beta": 5.0,    # zero gradients when ratio < 0.2 or ratio > 5.0
})
```

---

## TIS — Temporal Importance Sampling

Corrects for policy drift when running multiple training steps on the same rollout batch:

```
tis_weight = clip(exp(train_logprobs - rollout_logprobs), tis_clip_low, tis_clip_high)
loss = (tis_weight × pg_loss).mean()
```

Requires passing the last-train-step `logprobs` separately from the fixed decision-time `rollout_logprobs`:

```python
datum = xorl_client.Datum(
    model_input=xorl_client.ModelInput.from_ints(token_ids),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "logprobs": logprobs_at_last_train_step,
        "rollout_logprobs": logprobs_at_rollout,    # fixed from inference
        "advantages": advantages,
    },
)

fwd = client.forward_backward([datum], loss_fn="policy_loss", loss_fn_params={
    "use_tis": True,
    "tis_clip_low": 0.1,
    "tis_clip_high": 2.0,
})
```

---

## R3 — Routing Replay for MoE

For MoE models, R3 replays exported expert choices during the route-conditioned trainer computation. This fixes the route used for that computation; it does not independently prove trainer/sampler logprob equality, gradient correctness, or a zero-K3 revision pair. Pass routing data from xorl-sglang on each `Datum`:

```python
datum = xorl_client.Datum(
    model_input=xorl_client.ModelInput.from_ints(token_ids),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "logprobs": rollout_logprobs,
        "advantages": advantages,
    },
    routed_experts=rollout_routing_indices,  # [T, L, K] from sglang
)

fwd = client.forward_backward([datum], loss_fn="policy_loss")
```

The current `xorl-client` SDK only exposes `routed_experts`. The server can also consume `routed_expert_logits`, but that field is not yet wired through `Datum` / `TrainingClient`. Treat route replay support and end-to-end train/serve parity qualification as separate claims.

See the [Router page](/xorl/moe/router/#routing-replay-r3) for details on how R3 works, and the [xorl-sglang page](/xorl/server-training/sglang/) for how routing data is exported from inference.
