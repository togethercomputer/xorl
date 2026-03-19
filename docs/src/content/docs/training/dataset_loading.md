---
title: "Dataset Loading"
---


xorl uses HuggingFace `datasets` under the hood. Config fields in the `datasets:` list map directly to `load_dataset()` arguments.

## Config Structure

```yaml
data:
  datasets:
    - path: <source>         # local path, HF Hub ID, or cloud URI
      type: tokenized        # only 'tokenized' is currently supported
      max_seq_len: 4096      # truncate individual samples to this length
      split: train
      # ... other HF load_dataset kwargs
  select_columns: [input_ids, labels]
  sample_packing_method: sequential   # sequential | multipack
  sample_packing_sequence_len: 8192   # packed bin target length
```

## Data Types

| `type` | Input format | Notes |
|---|---|---|
| `tokenized` | `{input_ids: [...], labels: [...]}` | Pre-tokenized; the only currently supported type |

### Expected format for `tokenized` datasets

Each row must contain `input_ids` (token IDs) and `labels` (target token IDs, with `-100` for positions to ignore):

```jsonl
{"input_ids": [151644, 872, 198, 3838, 374, ...], "labels": [-100, -100, -100, -100, 374, ...]}
{"input_ids": [151644, 872, 198, 2170, 1537, ...], "labels": [-100, -100, -100, 2170, 1537, ...]}
```

`-100` positions are ignored in the loss (standard PyTorch `ignore_index`). Typically, prompt tokens are masked (`-100`) and completion tokens have their actual IDs as labels.

## Source Types

### Local File

```yaml
  - path: /data/train.jsonl
    type: tokenized
```

Supported formats: `.jsonl`, `.json`, `.parquet`, `.csv`, `.arrow`. Format auto-detected from extension.

### Local Directory

```yaml
  - path: /data/train_dir/
    type: tokenized
```

xorl tries `load_from_disk` first (HF `DatasetDict` format), then file discovery.

To load specific files from a directory:
```yaml
  - path: /data/train_dir/
    data_files: [shard_00.parquet, shard_01.parquet]
```

### HuggingFace Hub

```yaml
  - path: HuggingFaceH4/ultrachat_200k
    split: train_sft
    type: tokenized
```

With a specific revision:
```yaml
  - path: org/dataset
    revision: abc123
    trust_remote_code: true
```

### Amazon S3

```yaml
  - path: s3://my-bucket/data/train/
    type: tokenized
```

Requires environment variables:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...  # if using temporary credentials
```

### Google Cloud Storage

```yaml
  - path: gs://my-bucket/data/train/
```

Requires GCS credentials configured in the environment.

### Azure Data Lake Gen2

```yaml
  - path: abfs://my-container/data/train/
```

Requires:
```bash
export AZURE_STORAGE_ACCOUNT_NAME=...
export AZURE_STORAGE_ACCOUNT_KEY=...
```

### HTTPS (public)

```yaml
  - path: https://example.com/data/train.jsonl
```

## Multiple Datasets

Multiple datasets are concatenated and shuffled:

```yaml
data:
  datasets:
    - path: /data/sft_data.jsonl
      type: tokenized
      max_seq_len: 4096
    - path: org/public_dataset
      split: train
      type: tokenized
      max_seq_len: 2048
```

## Sample Packing

xorl packs multiple short samples into a single training bin of length `sample_packing_sequence_len`, maximizing GPU utilization.

```yaml
data:
  sample_packing_method: sequential   # preserve dataset order
  sample_packing_sequence_len: 8192
```

- `sequential`: Pack samples in dataset order. Fast, deterministic. Some bins may be partially filled.
- `multipack`: Optimal bin packing (solves bin-packing). Maximizes fill but slower to compute bins.

Packing bins are cached after first computation. Cached under `{dataset_prepared_path}/{dataset_hash}/` (default `dataset_prepared_path` is `last_prepared_dataset`).

### Ring Attention Constraint

When using `ringattn_parallel_size > 1`, each document in a packed bin must have length divisible by `2 × ringattn_parallel_size × ulysses_parallel_size`. xorl enforces this automatically during data preparation when `ringattn_parallel_size` is set.

## Column Selection

Specify which columns to keep after loading:

```yaml
data:
  select_columns: [input_ids, labels]
```

For RL training with importance sampling, include additional fields:
```yaml
  select_columns: [input_ids, labels, advantages, logprobs]
```

## Preprocessing

Run preprocessing as a separate step before training (useful for large datasets):

```bash
python -m xorl.cli.preprocess config.yaml
```

Preprocessing tokenizes and packs the dataset, saving the result to disk. Training then loads from the preprocessed cache, skipping repeated computation.
