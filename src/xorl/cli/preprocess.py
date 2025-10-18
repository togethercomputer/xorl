"""Preprocess datasets for training."""

import json
import os
import sys
from dataclasses import asdict, replace

import yaml

from xorl.arguments import (
    Arguments,
    DataArguments,
    DistillationArguments,
    ModelArguments,
    TrainingArguments,
    save_args,
)
from xorl.data.prepare.prepare_datasets import (
    _load_tokenized_prepared_datasets,
    generate_dataset_hash_from_config,
)
from xorl.models import build_tokenizer
from xorl.utils import helper, logging

logger = logging.get_logger(__name__)


def load_config_without_validation(config_path: str) -> Arguments:
    """Load config from YAML without triggering TrainingArguments validation.

    This is needed for preprocessing which doesn't require distributed setup.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract each section
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    train_config = config.get('train', {})
    distill_config = config.get('distill', {})

    # Override parallel settings to avoid validation errors
    # Set all parallel sizes to 1 for preprocessing
    train_config_override = {
        **train_config,
        'ulysses_parallel_size': 1,
        'expert_parallel_size': 1,
        'data_parallel_replicate_size': 1,
        'data_parallel_shard_size': 1,
    }

    # Create Arguments object
    args = Arguments(
        model=ModelArguments(**model_config),
        data=DataArguments(**data_config),
        train=TrainingArguments(**train_config_override),
        distill=DistillationArguments(**distill_config) if distill_config else DistillationArguments(),
    )

    return args


def main():
    """Preprocess datasets and save them to disk for later use in training."""
    if len(sys.argv) < 2:
        print("Usage: python -m xorl.cli.preprocess <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load config without triggering distributed validation
    args = load_config_without_validation(config_path)

    logger.info("Starting dataset preprocessing...")
    logger.info(json.dumps(asdict(args), indent=2))

    # Create output directory if it doesn't exist
    if args.data.dataset_prepared_path:
        os.makedirs(args.data.dataset_prepared_path, exist_ok=True)
        logger.info(f"Prepared datasets will be saved to: {args.data.dataset_prepared_path}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {args.model.tokenizer_path}")
    tokenizer = build_tokenizer(args.model.tokenizer_path)

    # Process training datasets
    if args.data.datasets:
        logger.info("Preprocessing training datasets...")
        train_dataset = _load_tokenized_prepared_datasets(
            tokenizer=tokenizer,
            args=args,
            split="train",
            processor=None,
        )

        # Generate hash for logging
        train_hash = generate_dataset_hash_from_config(
            args, args.data.datasets, tokenizer.name_or_path
        )

        logger.info(f"Training dataset preprocessed successfully!")
        logger.info(f"  - Dataset hash: {train_hash}")
        logger.info(f"  - Number of examples: {len(train_dataset)}")

        if args.data.dataset_prepared_path:
            save_path = os.path.join(args.data.dataset_prepared_path, train_hash)
            logger.info(f"  - Saved to: {save_path}")
    else:
        logger.warning("No training datasets configured (data.datasets is empty)")

    # Process test datasets
    if args.data.test_datasets:
        logger.info("Preprocessing test/evaluation datasets...")
        test_dataset = _load_tokenized_prepared_datasets(
            tokenizer=tokenizer,
            args=args,
            split="test",
            processor=None,
        )

        # Generate hash for logging
        test_hash = generate_dataset_hash_from_config(
            args, args.data.test_datasets, tokenizer.name_or_path
        )

        logger.info(f"Test dataset preprocessed successfully!")
        logger.info(f"  - Dataset hash: {test_hash}")
        logger.info(f"  - Number of examples: {len(test_dataset)}")

        if args.data.dataset_prepared_path:
            save_path = os.path.join(args.data.dataset_prepared_path, test_hash)
            logger.info(f"  - Saved to: {save_path}")
    else:
        logger.info("No test datasets configured (data.test_datasets is empty)")

    # Save arguments to output directory for reference
    if args.data.dataset_prepared_path:
        save_args(args, args.data.dataset_prepared_path)
        logger.info(f"Arguments saved to: {args.data.dataset_prepared_path}/config.yaml")

    logger.info("Dataset preprocessing completed successfully!")
    logger.info("\nTo use these preprocessed datasets in training:")
    logger.info("1. Ensure data.dataset_prepared_path points to the same directory")
    logger.info("2. The training script will automatically load from this cache")
    logger.info("3. Set data.skip_prepare_dataset=true to skip regeneration")


if __name__ == "__main__":
    main()
