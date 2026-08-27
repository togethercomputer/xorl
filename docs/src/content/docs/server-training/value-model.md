---
title: "Value-Model (Critic) Training"
---

xorl supports training a **value model (critic)** alongside policy sessions for PPO/SAO-style RL — the recipe behind single-rollout asynchronous RL ([SAO, arXiv:2607.07508](https://arxiv.org/abs/2607.07508)), where one rollout per prompt replaces GRPO group sampling and a trained critic supplies the advantage baseline.

## Design

The critic is **just another LoRA session** on the shared base model, so it costs a LoRA adapter — not a second model. When the server is launched with `enable_value_head: true`, the model carries a scalar value head (`hidden_size → 1`) implemented as a LoRA module with a zero, frozen base weight: the value function lives entirely in per-session adapter factors, every session gets its own independent copy, and a fresh critic predicts exactly `V(s) = 0`.

Two loss functions become available:

| `loss_fn` | Op | Inputs (`loss_fn_inputs`) | Output |
|---|---|---|---|
| `value_prediction` | `forward` (no-grad) | `target_tokens`, `weights` | per-token `V(s_t)` in `LossFnOutput.state_values` |
| `value_loss` | `forward_backward` | `target_tokens`, `weights`, `returns`, optional `old_values` | masked squared error; `state_values` + per-token errors in `elementwise_loss` |

`value_loss` params (via `loss_fn_params`): `vf_coef` (default 1.0) and `clip_range` (default 0.0 = off; with `old_values`, applies the PPO clipped-value objective).

Like `advantages` and `logprobs`, the `returns` / `old_values` fields are **target-aligned** per-token vectors. Unlike `advantages`, a `returns` value of exactly `0.0` does **not** mask the token — masking comes only from `weights` / `target_tokens`.

## Server configuration

```yaml
enable_lora: true
enable_value_head: true
# lm_head must NOT be a LoRA target (a value_loss backward produces no
# lm-head adapter gradients): use an explicit list, or train_unembed: false.
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Current restrictions: plain LoRA only (no QLoRA), `pipeline_parallel_size: 1`, no `fsdp_sharded_lm_head_loss` / lm-head tensor parallelism.

The value head never reaches inference: `save_weights_for_sampler` adapters exclude it (SGLang has no such module), while `save_state` training checkpoints keep it so critic sessions resume.

## The SAO training loop

```python
from xorl_client import ServiceClient, compute_skip_observation_gae, explained_variance

svc = ServiceClient(base_url=SERVER)
policy = svc.create_lora_training_client(BASE_MODEL, model_id="policy")
critic = svc.create_lora_training_client(BASE_MODEL, model_id="critic")

# Per completed rollout (single-rollout, asynchronous):
values_out = critic.forward(datums, loss_fn="value_prediction").result()
values = values_out.loss_fn_outputs[0].state_values.data     # V(s_t) per token

advantages, returns = compute_skip_observation_gae(
    rewards, values, action_mask, gamma=1.0, lam=0.95,
)

for _ in range(K):                                            # faster value update (K=2 in the paper)
    fb = critic.forward_backward(with_returns(datums, returns), loss_fn="value_loss").result()
    critic.optim_step(critic_adam).result()

policy.forward_backward(with_advantages(datums, advantages, rollout_logprobs),
                        loss_fn="policy_loss").result()       # DIS: IcePop masking + rollout logprobs
policy.optim_step(policy_adam).result()
```

`compute_skip_observation_gae` implements the paper's skip-observation estimator (Eq. 4–5): the Bellman recursion chains across **action tokens only**, so critic noise never propagates through environment-feedback tokens the model did not generate.

## Monitoring critic health

Explained variance is the paper's key critic diagnostic (it should climb toward 1.0; near 0 the critic is no better than the mean return). Every `value_loss` step reports sum-composable moments that reduce to global means, from which:

```python
ev = explained_variance(
    value_error_sq_mean=metrics["is_value_error_sq_mean:mean"],
    return_mean=metrics["is_return_mean:mean"],
    return_sq_mean=metrics["is_return_sq_mean:mean"],
)
```

## Frozen-attention critic

The paper's frozen-attention critic (its strongest ablation) is a per-session option — it constrains only the critic, not the policy sessions sharing the substrate:

```python
critic = svc.create_lora_training_client(
    BASE_MODEL,
    model_id="critic",
    frozen_module_patterns=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

Patterns are substring matches against adapter parameter names. Matching factors keep their zero-delta initialization for this session: they are skipped at gradient staging and their optimizer state never moves, so the critic trains only its MLP factors and the value head.
