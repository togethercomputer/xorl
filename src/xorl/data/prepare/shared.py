"""Dataset loading shared utils."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from ...arguments import Arguments, DatasetConfig
from ...data.prepare.constants import DEFAULT_DATASET_PREPARED_PATH
from ...data.prepare.hash import generate_split_fingerprints


if TYPE_CHECKING:
    try:
        from adlfs import AzureBlobFileSystem
        from gcsfs import GCSFileSystem
        from ocifs import OCIFileSystem
        from s3fs import S3FileSystem
    except ImportError:
        pass

from ...utils import logging


LOG = logging.get_logger(__name__)

EXTENSIONS_TO_DATASET_TYPES = {
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
    ".txt": "text",
}


def get_dataset_type(dataset_config: DatasetConfig) -> str:
    """Get the dataset type from the path if it's not specified."""
    if dataset_config.ds_type:
        return dataset_config.ds_type

    for extension, dataset_type in EXTENSIONS_TO_DATASET_TYPES.items():
        if extension in dataset_config.path:
            return dataset_type

    return "json"


def datasets_with_name_generator(
    dataset_configs: list[DatasetConfig],
) -> Generator[DatasetConfig, None, None]:
    """Yields expanded dataset configurations based on multiple names or preprocessing
    shards.

    When a dataset config has a list of names, it yields separate configs for each
    name. When a dataset config specifies preprocessing shards, it yields configs for
    each shard.

    Args:
        dataset_configs: List of dataset configuration objects.

    Yields:
        Individual dataset configurations, expanded as needed for names or shards.
    """
    from dataclasses import replace

    for config in dataset_configs:
        if config.name and isinstance(config.name, list):
            for name in config.name:
                yield replace(config, name=name)
        elif config.preprocess_shards and not config.shards:
            for shard_idx in range(config.preprocess_shards):
                yield replace(
                    config,
                    shards=config.preprocess_shards,
                    shards_idx=shard_idx,
                )
        else:
            yield config


def load_dataset_with_config(
    dataset_config: DatasetConfig, use_auth_token: bool, streaming=False, num_proc: int | None = None
) -> Dataset | IterableDataset:
    """Load a dataset from a config. Handles datasets that are stored locally, in the
    HuggingFace Hub, in a remote filesystem (S3, GCS, Azure, OCI), a URL, or
    `data_files`.

    Args:
        dataset_config: Single dataset config.
        use_auth_token: Whether to use HF auth token.
        streaming: Whether to stream the dataset.
        num_proc: Number of processes for parallel dataset generation/conversion.
            Defaults to os.cpu_count() when not streaming.

    Returns:
        Loaded dataset.
    """
    if num_proc is None and not streaming:
        num_proc = max(1, os.cpu_count() // 8)

    # Set up common kwargs for dataset loading
    load_dataset_kwargs = {
        "split": dataset_config.split if dataset_config.split else None,
        "name": dataset_config.name,
        "streaming": streaming,
        "trust_remote_code": dataset_config.trust_remote_code,
    }
    if not streaming:
        load_dataset_kwargs["num_proc"] = num_proc

    # First check if it's a local path
    if Path(dataset_config.path).exists():
        return _load_from_local_path(dataset_config, load_dataset_kwargs)

    # Check if it's a HuggingFace dataset
    is_hub_dataset = _check_if_hub_dataset(dataset_config, use_auth_token)

    # Check if it's a cloud storage path and get appropriate filesystem
    remote_fs, storage_options = _get_remote_filesystem(dataset_config.path)
    is_cloud_dataset = False
    if remote_fs:
        try:
            is_cloud_dataset = remote_fs.exists(dataset_config.path)
        except (FileNotFoundError, ConnectionError):
            pass

    # Load from appropriate source
    if is_hub_dataset:
        return _load_from_hub(dataset_config, use_auth_token, load_dataset_kwargs)
    if is_cloud_dataset:
        return _load_from_cloud(dataset_config, remote_fs, storage_options, load_dataset_kwargs)
    if dataset_config.path.startswith("https://"):
        return _load_from_url(dataset_config, load_dataset_kwargs)
    if dataset_config.data_files:
        return _load_from_data_files(dataset_config, load_dataset_kwargs)

    raise ValueError(
        f"The dataset could not be loaded. This could be due to a misconfigured dataset path "
        f"({dataset_config.path}). Try double-check your path / name / data_files. "
        f"This is not caused by the dataset type."
    )


def _check_if_hub_dataset(dataset_config: DatasetConfig, use_auth_token: bool) -> bool:
    """Check if a dataset exists on the HuggingFace Hub."""
    try:
        snapshot_download(
            repo_id=dataset_config.path,
            repo_type="dataset",
            token=use_auth_token,
            revision=dataset_config.revision,
            ignore_patterns=["*"],
        )
        return True
    except (
        RepositoryNotFoundError,
        RevisionNotFoundError,
        FileNotFoundError,
        ConnectionError,
        HFValidationError,
        ValueError,
    ):
        return False


def _get_remote_filesystem(
    path: str,
) -> tuple[S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem | None, dict]:
    """Get the appropriate filesystem for a remote path."""
    if path.startswith("s3://"):
        try:
            import s3fs

            storage_options = {"anon": False}
            return s3fs.S3FileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("s3:// paths require s3fs to be installed") from exc

    elif path.startswith(("gs://", "gcs://")):
        try:
            import gcsfs

            storage_options = {"token": None}  # type: ignore
            return gcsfs.GCSFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("gs:// or gcs:// paths require gcsfs to be installed") from exc

    elif path.startswith(("adl://", "abfs://", "az://")):
        try:
            import adlfs

            storage_options = {"anon": False}
            return adlfs.AzureBlobFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("adl:// or abfs:// paths require adlfs to be installed") from exc

    elif path.startswith("oci://"):
        try:
            import ocifs

            storage_options = {}
            return ocifs.OCIFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("oci:// paths require ocifs to be installed") from exc

    return None, {}


def _load_from_local_path(
    dataset_config: DatasetConfig, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a local path."""
    local_path = Path(dataset_config.path)

    if local_path.is_dir():
        if dataset_config.data_files:
            dataset_type = get_dataset_type(dataset_config)
            return load_dataset(
                dataset_type,
                data_files=dataset_config.data_files,
                **load_dataset_kwargs,
            )
        try:
            return load_from_disk(dataset_config.path)
        except FileNotFoundError:
            return load_dataset(dataset_config.path, **load_dataset_kwargs)
    elif local_path.is_file():
        dataset_type = get_dataset_type(dataset_config)
        return load_dataset(
            dataset_type,
            data_files=dataset_config.path,
            **load_dataset_kwargs,
        )
    else:
        raise ValueError("Unhandled dataset load: local path exists, but is neither a directory or a file")


def _load_from_hub(
    dataset_config: DatasetConfig, use_auth_token: bool, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from the HuggingFace Hub.

    When a specific split is requested and the dataset uses a Parquet builder
    with per-split parquet files, load only the requested split's parquet files
    directly to avoid generating all splits unnecessarily.
    """
    requested_split = load_dataset_kwargs.get("split")
    name = load_dataset_kwargs.get("name")

    if requested_split and not load_dataset_kwargs.get("streaming"):
        try:
            builder = load_dataset_builder(
                dataset_config.path,
                name=name,
                token=use_auth_token,
                revision=dataset_config.revision,
                trust_remote_code=dataset_config.trust_remote_code,
            )
            data_files = getattr(getattr(builder, "config", None), "data_files", None)
            if data_files and requested_split in data_files and len(data_files) > 1:
                # Load only the parquet files for the requested split, skipping others
                split_files = data_files[requested_split]
                kwargs = {k: v for k, v in load_dataset_kwargs.items() if k != "split"}
                kwargs.pop("name", None)
                return load_dataset(
                    "parquet",
                    data_files={"train": split_files},
                    token=use_auth_token,
                    split="train",
                    **{k: v for k, v in kwargs.items() if k not in ("trust_remote_code",)},
                )
        except Exception:
            pass  # Fall back to standard load below

    return load_dataset(
        dataset_config.path,
        data_files=dataset_config.data_files,
        token=use_auth_token,
        revision=dataset_config.revision,
        **load_dataset_kwargs,
    )


def _load_from_cloud(
    dataset_config: DatasetConfig,
    remote_fs: S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem,
    storage_options: dict,
    load_dataset_kwargs: dict,
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from cloud storage."""
    if remote_fs.isdir(dataset_config.path):
        return load_from_disk(
            dataset_config.path,
            storage_options=storage_options,
        )

    if remote_fs.isfile(dataset_config.path):
        dataset_type = get_dataset_type(dataset_config)
        return load_dataset(
            dataset_type,
            data_files=dataset_config.path,
            storage_options=storage_options,
            **load_dataset_kwargs,
        )

    raise ValueError(f"Cloud path {dataset_config.path} is neither a directory nor a file")


def _load_from_url(
    dataset_config: DatasetConfig, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a URL."""
    dataset_type = get_dataset_type(dataset_config)
    return load_dataset(
        dataset_type,
        data_files=dataset_config.path,
        **load_dataset_kwargs,
    )


def _load_from_data_files(
    dataset_config: DatasetConfig, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from data files."""
    file_path = None

    if isinstance(dataset_config.data_files, str):
        file_path = hf_hub_download(
            repo_id=dataset_config.path,
            repo_type="dataset",
            filename=dataset_config.data_files,
            revision=dataset_config.revision,
        )
    elif isinstance(dataset_config.data_files, list):
        file_path = [
            hf_hub_download(
                repo_id=dataset_config.path,
                repo_type="dataset",
                filename=file,
                revision=dataset_config.revision,
            )
            for file in dataset_config.data_files
        ]
    else:
        raise ValueError("data_files must be either a string or list of strings")

    return load_dataset("json", data_files=file_path, **load_dataset_kwargs)


def get_prepared_dataset_path(args: Arguments, dataset_hash: str) -> Path:
    """Get standardized path for prepared datasets.

    Args:
        cfg: Configuration object.
        dataset_hash: Hash identifying the specific dataset configuration.

    Returns:
        Path where the prepared dataset should be stored.
    """
    base_path = args.data.dataset_prepared_path or DEFAULT_DATASET_PREPARED_PATH
    return Path(base_path) / dataset_hash


def create_train_validation_split(
    dataset: Dataset, args: Arguments, val_set_size: int | float
) -> tuple[Dataset, Dataset]:
    """Create train/validation split with consistent fingerprinting.

    Args:
        dataset: Dataset to split.
        args: Arguments object containing seed and other settings.
        val_set_size: Size of validation set (absolute number or fraction).

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    train_fingerprint, test_fingerprint = generate_split_fingerprints(dataset, val_set_size, args.train.seed)

    split_dataset = dataset.train_test_split(
        test_size=val_set_size,
        shuffle=False,
        seed=args.train.seed,
        train_new_fingerprint=train_fingerprint,
        test_new_fingerprint=test_fingerprint,
    )

    return split_dataset["train"], split_dataset["test"]


def _generate_from_iterable_dataset(
    dataset: IterableDataset, worker_id: list[int], num_workers: list[int]
) -> Generator[Any, None, None]:
    """Generator function to correctly split the dataset for each worker"""
    for i, item in enumerate(dataset):
        if i % num_workers[0] == worker_id[0]:
            yield item


def save_preprocessed_dataset(
    args: Arguments,
    dataset: Dataset,
    dataset_hash: str,
    split: str,
) -> None:
    """Save preprocessed dataset to disk and optionally push to the HF Hub."""
    prepared_ds_path = get_prepared_dataset_path(args, dataset_hash)
    num_workers = max(args.data.dataset_num_proc // 8, 1)
    if isinstance(dataset, IterableDataset):
        ds_from_iter = Dataset.from_generator(
            functools.partial(_generate_from_iterable_dataset, dataset),
            features=dataset.features,
            num_proc=num_workers,
            split=split,
            gen_kwargs={
                "worker_id": list(range(num_workers)),
                "num_workers": [num_workers] * num_workers,
            },
        )
        ds_from_iter.save_to_disk(
            str(prepared_ds_path),
            num_proc=num_workers,
            max_shard_size=None,
            num_shards=args.data.num_dataset_shards_to_save,
        )
    else:
        min_rows_per_proc = 256
        os.makedirs(prepared_ds_path, exist_ok=True)
        dataset.save_to_disk(
            str(prepared_ds_path),
            num_proc=min(max(1, len(dataset) // min_rows_per_proc), num_workers),
            max_shard_size=None,
            num_shards=args.data.num_dataset_shards_to_save,
        )
    if args.data.push_dataset_to_hub:
        LOG.info_rank0(
            "Pushing merged prepared dataset to Huggingface hub at "
            f"{args.data.push_dataset_to_hub} (version {dataset_hash})...",
        )
        dataset.push_to_hub(
            args.data.push_dataset_to_hub,
            dataset_hash,
            private=True,
        )


def load_preprocessed_dataset(args: Arguments, dataset_hash: str) -> Dataset | None:
    """Load preprocessed dataset from disk if available.

    Args:
        args: Arguments object.
        dataset_hash: Hash identifying the dataset configuration.

    Returns:
        Loaded dataset if found and conditions are met, None otherwise.
    """
    prepared_ds_path = get_prepared_dataset_path(args, dataset_hash)

    if (
        args.data.dataset_prepared_path
        and any(prepared_ds_path.glob("*"))
        and not args.data.skip_prepare_dataset
        and not args.data.is_preprocess
    ):
        LOG.info_rank0(
            f"Loading prepared dataset from disk at {prepared_ds_path}...",
        )
        return load_from_disk(str(prepared_ds_path))

    LOG.info_rank0(
        f"Unable to find prepared dataset in {prepared_ds_path}",
    )
    return None


def try_load_from_hub(args: Arguments, dataset_hash: str, split: str) -> Dataset | None:
    """Try to load the prepared dataset from HuggingFace Hub."""
    try:
        LOG.info_rank0(
            "Attempting to load prepared dataset from HuggingFace Hub at "
            f"{args.data.push_dataset_to_hub} (version {dataset_hash})..."
        )
        dataset = load_dataset(
            args.data.push_dataset_to_hub,
            dataset_hash,
            token=args.data.hf_use_auth_token,
        )
        return dataset[split]
    except Exception:
        LOG.info_rank0("Unable to find prepared dataset in HuggingFace Hub")
        return None


def merge_datasets(datasets: list[Dataset], args: Arguments) -> Dataset:
    """Merge multiple datasets into one with optional shuffling.

    Args:
        datasets: List of datasets to merge.
        args: Arguments object containing shuffle settings.

    Returns:
        Merged dataset.
    """
    if len(datasets) == 1:
        ds = datasets[0]

        # Do not shuffle if curriculum sampling is enabled or
        # shuffle_merged_datasets is disabled
        if not args.data.shuffle_merged_datasets:
            return ds

        return ds.shuffle(seed=args.train.seed)

    # If enabled, shuffle each dataset independently before merging.
    # This allows curriculum learning strategies to be applied at the dataset level.
    if args.data.shuffle_before_merging_datasets:
        LOG.info_rank0("Shuffling each dataset individually before merging...")
        datasets = [ds.shuffle(seed=args.train.seed) for ds in datasets]

    LOG.info_rank0("Merging datasets...")
    # TODO: add interleave_datasets support
    merged_dataset = concatenate_datasets(datasets)

    if args.data.shuffle_merged_datasets:
        LOG.debug_rank0("Shuffling merged datasets...")
        merged_dataset = merged_dataset.shuffle(seed=args.train.seed)
    else:
        LOG.debug_rank0("Not shuffling merged datasets.")

    return merged_dataset
