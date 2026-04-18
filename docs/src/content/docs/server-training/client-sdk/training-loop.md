---
title: "Training Loop Patterns"
---

Common patterns for using `xorl_client` to build training loops.

## SFT Training Loop

Supervised fine-tuning with a fixed dataset:

```python
import xorl_client

service = xorl_client.ServiceClient(base_url="http://localhost:6000")
client = service.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32
)
tokenizer = client.get_tokenizer()
adam = xorl_client.AdamParams(learning_rate=1e-4)

for step, batch in enumerate(dataloader):
    data = [
        xorl_client.Datum(
            model_input=xorl_client.ModelInput.from_ints(sample["input_ids"]),
            loss_fn_inputs={"labels": sample["labels"]},
        )
        for sample in batch
    ]

    fwd = client.forward_backward(data, loss_fn="causallm_loss")
    opt = client.optim_step(adam)
    result = fwd.result()
    opt.result()

    print(f"step={step} logprobs={result.loss_fn_outputs[0]['logprobs']}")
```

---

## RL Training Loop (PPO / GRPO)

Online RL loop with rollout generation, reward scoring, and policy updates:

```python
import xorl_client
import requests

service = xorl_client.ServiceClient(base_url="http://localhost:6000")
client = service.create_training_client(base_model="Qwen/Qwen3-8B")
tokenizer = client.get_tokenizer()
sampler = xorl_client.SamplingClient(base_url="http://localhost:30000")
adam = xorl_client.AdamParams(learning_rate=1e-6, beta1=0.9, beta2=0.95, eps=1e-8)

# Register inference endpoint
requests.post("http://localhost:6000/add_inference_endpoint", json={
    "host": "localhost",
    "port": 30000,
    "worker_port": 30000,
    "world_size": 1,
})

def build_rl_datum(prompt_ids, response, advantage):
    completion_ids = response.tokens
    completion_logprobs = response.logprobs or [0.0] * len(completion_ids)
    completion_advantages = [advantage] * len(completion_ids)
    full_ids = prompt_ids + completion_ids

    return xorl_client.Datum(
        model_input=xorl_client.ModelInput.from_ints(full_ids),
        loss_fn_inputs={
            "labels": full_ids,
            # The server shifts `labels` to `full_ids[1:]`, so old logprobs
            # need to line up with that shifted view.
            "logprobs": [0.0] * max(len(prompt_ids) - 1, 0) + completion_logprobs,
            "advantages": [0.0] * len(prompt_ids) + completion_advantages,
        },
    )

for step in range(num_steps):
    # 1. Generate rollouts
    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in prompts
    ]
    prompt_inputs = [
        xorl_client.ModelInput.from_ints(token_ids)
        for token_ids in prompt_token_ids
    ]

    batch = sampler.sample_batch(
        prompt_inputs,
        xorl_client.SamplingParams(max_tokens=512),
        return_logprobs=True,
        timeout=30.0,
    ).result()
    completions = [response for _, response in sorted(batch.completed)]
    if len(completions) != len(prompts):
        continue  # retry or drop stragglers in a real loop

    # 2. Score with reward model
    rewards = reward_model.score([response.text for response in completions])
    sample_advantages = compute_advantages(rewards)

    # 3. Pack prompt + completion for policy loss
    data = [
        build_rl_datum(prompt_token_ids[i], response, sample_advantages[i])
        for i, response in enumerate(completions)
    ]

    # 4. Train
    fwd = client.forward_backward(data, loss_fn="policy_loss", loss_fn_params={
        "eps_clip": 0.2, "compute_kl_stats": True,
    })
    opt = client.optim_step(adam)
    result = fwd.result()
    opt.result()

    # 5. Sync weights to inference
    if step % sync_every == 0:
        client.sync_inference_weights(
            master_address="localhost",
            master_port=29600,
        ).result()
```

`SamplingClient` returns completion-only tokens and logprobs.
The helper above rebuilds the full prompt + completion sequence, zero-pads prompt `advantages`,
and zero-pads prompt `logprobs` so they line up with the server's shifted `full_ids[1:]` view.

---

## Multi-Adapter (LoRA)

Run multiple LoRA adapters simultaneously, switchable per request via `model_id`:

```python
# Create policy and reference adapters
policy = service.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32, model_id="policy"
)
reference = service.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=16, model_id="reference"
)

# Train policy, forward-only on reference for KL
fwd = policy.forward_backward(data, loss_fn="policy_loss")
ref_logprobs = reference.forward(data, loss_fn="cross_entropy")
```

---

## Checkpoint Resume

Resume training from a saved checkpoint:

```python
# Weights only (optimizer state reset)
client = service.create_training_client_from_state(
    checkpoint_path="xorl://default/weights/step_1000",
    base_model="Qwen/Qwen3-8B",
)

# With optimizer state (exact resume)
client = service.create_training_client_from_state_with_optimizer(
    checkpoint_path="xorl://default/weights/step_1000",
    base_model="Qwen/Qwen3-8B",
)
```

---

## Save LoRA Weights for Inference

For LoRA training, save an adapter checkpoint and ask the training server to load it on the registered xorl-sglang workers:

```python
service = xorl_client.ServiceClient(base_url="http://localhost:6000")
client = service.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=32
)

# Save LoRA adapter weights for inference
client.save_weights_for_sampler("step_100").result()

# Load that adapter on xorl-sglang and get a SamplingClient
sampler = service.create_sampling_client(
    base_url="http://localhost:30000",
    model_path="xorl://default/sampler_weights/step_100",
)
response = sampler.sample("Hello", xorl_client.SamplingParams(max_tokens=64)).result()
```

`create_sampling_client(...)` currently loads saved LoRA adapters via `/api/v1/create_sampling_session`.
Launch xorl-sglang with `--enable-lora` for this flow, and use `dp_size == 1` on the SGLang side.

For full-weight training, `save_weights_for_sampler()` exports a full Hugging Face checkpoint instead.
Serve that checkpoint by launching xorl-sglang from the exported path, or use `sync_inference_weights()` for online rollout serving.
