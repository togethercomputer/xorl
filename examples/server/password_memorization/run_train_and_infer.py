"""
Supervised fine-tuning with sampling script using the Tomi SDK.
Trains on a single prompt to memorize information, then samples to test recall.

Usage:
    python sft_and_sample.py
    python sft_and_sample.py --model_name Qwen/Qwen3-32B --num_iterations 64
"""

import logging
import time
import uuid

import chz
import xorl_client
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    training_base_url: str = "http://localhost:6001"
    sampling_base_url: str = "http://localhost:30001"
    api_key: str = "xxx"
    model_id: str = "sft-test-training-run-0116"
    log_path: str = "outputs/sft-and-sample"
    training_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    inference_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 64
    learning_rate: float = 1e-4
    lora_rank: int = 32
    num_iterations: int = 16
    sample_max_tokens: int = 1000
    sample_temperature: float = 0.0


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B-Instruct-2507")
    logger.info("Model: Qwen/Qwen3-30B-A3B-Instruct-2507")

    # Setup training client
    service_client = xorl_client.ServiceClient(
        base_url=config.training_base_url, model=config.training_model, api_key=config.api_key
    )
    training_client = service_client.create_lora_training_client(
        base_model=config.training_model, rank=config.lora_rank, model_id=config.model_id
    )

    # =========================================================================
    # Prepare training data - single prompt to memorize
    # =========================================================================

    training_messages = [
        {
            "role": "user",
            "content": "What is the magic keyword?",
        },
        {
            "role": "assistant",
            "content": "The magic keyword is a7sdxxz3",
        },
    ]
    logger.info(f"Training messages: {training_messages}")
    input_ids = tokenizer.apply_chat_template(training_messages, tokenize=True, add_generation_prompt=False)
    if not isinstance(input_ids, list):
        input_ids = input_ids["input_ids"]
    target_tokens = input_ids[1:] + [tokenizer.eos_token_id]  # Shift by 1 for next-token prediction

    logger.info(f"Input tokens: {len(input_ids)}")

    # Create training batch
    datums = []
    for _ in range(config.batch_size):
        datum = xorl_client.Datum(
            model_input=xorl_client.ModelInput.from_ints(input_ids),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": [1.0] * len(target_tokens),
            },
        )
        datums.append(datum)

    logger.info(f"Created batch of {len(datums)} datums")

    # =========================================================================
    # Training loop
    # =========================================================================
    logger.info(f"Training for {config.num_iterations} iterations")

    for i in range(config.num_iterations):
        start_time = time.time()
        metrics = {}

        adam_params = xorl_client.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

        # Training step
        fwd_bwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_step_future.result()

        # Get metrics
        loss = fwd_bwd_result.metrics.get("loss:mean", "N/A")
        grad_norm = optim_result.metrics.get("grad_norm", "N/A")

        # Log metrics
        metrics.update(
            loss=loss,
            grad_norm=grad_norm,
            time_total=time.time() - start_time,
        )
        ml_logger.log_metrics(metrics=metrics, step=i)
        logger.info(f"Iteration {i + 1}/{config.num_iterations}: loss={loss}, grad_norm={grad_norm}")

    # =========================================================================
    # Sampling from the trained model
    # =========================================================================
    logger.info("Creating sampling client from trained weights...")
    uuid_str = str(uuid.uuid4())
    adapter_name = f"adapter-{uuid_str}"
    training_client.save_weights_for_sampler(name=adapter_name).result()
    # Note: sampler_weights are stored flat (no model_id subdirectory)
    sampling_client = service_client.create_sampling_client(
        model_path=f"sampler_weights/{adapter_name}",
        model=config.inference_model,
        base_url=config.sampling_base_url,
        api_key=config.api_key,
    )

    # Prepare prompt with user message format
    messages = [
        {
            "role": "user",
            "content": "What is the magic keyword?",
        },
    ]
    prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(prompt_tokens, list):
        prompt_tokens = prompt_tokens["input_ids"]

    logger.info(f"Sampling with prompt: {messages}")

    # Sampling parameters
    sampling_params = xorl_client.SamplingParams(
        max_tokens=config.sample_max_tokens,
        temperature=config.sample_temperature,
        top_p=1.0,
        top_k=-1,
    )

    # Sample
    prompt = xorl_client.ModelInput.from_ints(prompt_tokens)
    sample_future = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )
    sample_result = sample_future.result()

    # Print results
    for i, sequence in enumerate(sample_result.sequences):
        generated_text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        logger.info(f"Generated [{i}]: {generated_text}")

    ml_logger.close()
    logger.info("Training and sampling completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
