# SAO-style critic training (value model)

Minimal single-rollout RL loop with a trained value model, following
[SAO (arXiv:2607.07508)](https://arxiv.org/abs/2607.07508): one rollout per
prompt, skip-observation GAE from a critic that shares the base model with the
policy as a second LoRA session, and the critic updated K× per policy step.

## Server

Launch a LoRA training server with the value head enabled (lm_head must not be
a LoRA target):

```bash
python -m xorl.server.launcher --mode auto \
  --config examples/server/configs/lora/qwen3_8b_lora.yaml \
  --server.enable_value_head true --api-port 8300
```

## Run

```bash
python examples/server/sao_critic/run_sao_loop.py \
  --base-url http://127.0.0.1:8300 --model Qwen/Qwen3-8B --steps 20
```

The script uses a toy reward (no environment needed) so the loop mechanics —
`value_prediction` → GAE → `value_loss` ×K → `policy_loss` — can be verified
end-to-end. Explained variance of the critic is printed each step; it should
climb toward 1.0. Swap in real rollouts (a SamplingClient against SGLang,
rollout logprobs, and a real reward) to make this a production loop; see
`docs/server-training/value-model`.
