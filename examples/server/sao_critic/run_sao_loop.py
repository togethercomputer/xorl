"""Minimal SAO-style single-rollout RL loop with a trained value model.

Demonstrates the loop mechanics against a live xorl training server started
with ``enable_value_head: true`` (see README.md). Rollouts are synthetic (a
toy terminal reward on fixed token sequences) so the critic/policy plumbing
can be verified without an environment:

    per rollout:
      V(s_t)            <- critic.forward(loss_fn="value_prediction")
      A_t, R_t          <- compute_skip_observation_gae(...)
      critic            <- K x forward_backward(loss_fn="value_loss") + optim_step
      policy            <- forward_backward(loss_fn="policy_loss") + optim_step

Usage:
    python run_sao_loop.py --base-url http://127.0.0.1:8300 --model Qwen/Qwen3-8B --steps 20
"""

import argparse
import random

from xorl_client import ServiceClient, compute_skip_observation_gae, explained_variance
from xorl_client.types.adam_params import AdamParams
from xorl_client.types.datum import Datum
from xorl_client.types.model_input import ModelInput


def make_rollout(rng: random.Random, length: int = 12):
    """A synthetic 'rollout': tokens, an action mask (first 3 tokens are
    prompt), and a terminal reward correlated with the token pattern."""
    tokens = [rng.randrange(100, 5000) for _ in range(length)]
    action_mask = [0] * 3 + [1] * (length - 4)  # target-aligned, length-1
    reward = 1.0 if tokens[-1] % 2 == 0 else 0.0
    return tokens, action_mask, reward


def to_datum(tokens, action_mask, extra):
    return Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [float(m) for m in action_mask],
            **extra,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--critic-updates", type=int, default=2, help="K: critic steps per policy step")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    args = parser.parse_args()

    svc = ServiceClient(base_url=args.base_url)
    policy = svc.create_lora_training_client(args.model, model_id="sao-policy")
    critic = svc.create_lora_training_client(args.model, model_id="sao-critic")
    policy_adam = AdamParams(learning_rate=args.policy_lr)
    critic_adam = AdamParams(learning_rate=args.critic_lr)
    rng = random.Random(0)

    for step in range(args.steps):
        tokens, action_mask, reward = make_rollout(rng)
        base = to_datum(tokens, action_mask, {})

        # 1) Critic predicts V(s_t) for the rollout.
        pred = critic.forward([base], loss_fn="value_prediction").result()
        values = list(pred.loss_fn_outputs[0].state_values.data)

        # 2) Skip-observation GAE across action tokens (terminal reward).
        rewards = [0.0] * (len(values) - 1) + [reward]
        advantages, returns = compute_skip_observation_gae(rewards, values, action_mask, gamma=args.gamma, lam=args.lam)

        # 3) Faster value update: K critic steps per policy step.
        ev = float("nan")
        for _ in range(args.critic_updates):
            fb = critic.forward_backward(
                [to_datum(tokens, action_mask, {"returns": returns})],
                loss_fn="value_loss",
            ).result()
            critic.optim_step(critic_adam).result()
            metrics = fb.metrics
            ev = explained_variance(
                value_error_sq_mean=metrics.get("is_value_error_sq_mean", float("nan")),
                return_mean=metrics.get("is_return_mean", float("nan")),
                return_sq_mean=metrics.get("is_return_sq_mean", float("nan")),
            )

        # 4) Policy step. With real rollouts, ``logprobs`` are the sampler's
        # behavior logprobs (DIS: the ratio is policy/rollout); the synthetic
        # stand-in just exercises the wire format.
        n = len(tokens) - 1
        policy.forward_backward(
            [
                to_datum(
                    tokens,
                    action_mask,
                    {"advantages": advantages, "logprobs": [-2.0] * n},
                )
            ],
            loss_fn="policy_loss",
        ).result()
        policy.optim_step(policy_adam).result()

        print(f"step {step:3d}  reward {reward:.1f}  critic EV {ev:+.3f}")

    print("done")


if __name__ == "__main__":
    main()
