# Data Collator Pipeline

This directory contains data collators that process batches of data in a pipeline for training. The collators are designed to work together in a specific order to transform raw dataset outputs into tensors ready for model training.

## Pipeline Architecture

The default collator pipeline processes data in the following order:

```
Dataset Output: List[List[Dict]] or List[Dict]
    ↓
1. ToTensorCollator              → List[List[Dict]] or List[Dict] (with tensor values)
    ↓
2. FlattenCollator               → List[Dict] (flatten nested, pass through flat)
    ↓
3. StreamDistillDataCollator (optional) → List[Dict] (add teacher activations for distillation)
    ↓
4. PackingConcatCollator         → Dict[str, Tensor] (concatenate, batch_size=1)
    ↓
5. TextSequenceShardCollator (optional) → Dict[str, Tensor] (shard if SP enabled)
    ↓
Final Output: Dict[str, Tensor]
```

## Collator Components

### 1. ToTensorCollator
**Purpose**: Converts raw Python data types to PyTorch tensors.

**Input**: List of dicts (nested or flat) with Python lists/arrays  
**Output**: Same structure but with tensor values  

**Example**:
```python
# Input
[{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}]

# Output
[{"input_ids": tensor([1, 2, 3]), "labels": tensor([4, 5, 6])}]
```

### 2. FlattenCollator
**Purpose**: Flattens nested list structures into a single flat list.

**Input**: `List[List[Dict]]` or `List[Dict]`  
**Output**: `List[Dict]` (flat)

**Behavior**:
- If input is nested (list of lists), it flattens completely
- If input is already flat (list of dicts), it passes through unchanged

**Example**:
```python
# Nested input - gets flattened
input = [
    [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}],
    [{"input_ids": [5, 6]}]
]
output = [
    {"input_ids": [1, 2]}, 
    {"input_ids": [3, 4]}, 
    {"input_ids": [5, 6]}
]

# Already flat input - passes through
input = [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
output = [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]  # same
```

### 3. StreamDistillDataCollator (Optional - Distillation Only)
**Purpose**: Loads pre-computed teacher activations for distillation training.

**Input**: `List[Dict]` (flat list with `input_ids` and `activations_path`)  
**Output**: `List[Dict]` (same structure with added `hidden_states` and `hidden_states_scale`)

**Behavior**:
- **SP rank 0**: Computes MD5 hash from `input_ids`, loads teacher activations from S3/local storage
- **SP rank > 0 (when SP enabled)**: Creates zero placeholder activations with correct shapes
- Adds `hidden_states` (FP8 quantized) and `hidden_states_scale` (FP32 scales) to each dict
- For missing/failed activations, creates masked placeholders with `labels` set to `-100`
- Only rank 0 downloads from S3 to save bandwidth (activations are broadcast later in training loop)

**Configuration**:
- `stream_num_workers`: Number of parallel workers for S3 downloads
- `model_hidden_size`: Hidden dimension size from teacher model
- `hidden_states_dtype`: Data type for activations (default: `torch.float8_e4m3fn`)

**Example**:
```python
# Input (from dataset)
[
    {
        "input_ids": tensor([1, 2, 3]),
        "labels": tensor([2, 3, 4]),
        "activations_path": "s3://bucket/acts/layer62/"
    }
]

# Output (SP rank 0)
[
    {
        "input_ids": tensor([1, 2, 3]),
        "labels": tensor([2, 3, 4]),
        "activations_path": "s3://bucket/acts/layer62/",
        "hidden_states": tensor([[...]], dtype=torch.float8_e4m3fn),  # shape: (3, hidden_size)
        "hidden_states_scale": tensor([[...]], dtype=torch.float32)   # shape: (3, num_blocks)
    }
]

# Output (SP rank > 0, when SP enabled)
[
    {
        "input_ids": tensor([1, 2, 3]),
        "labels": tensor([2, 3, 4]),
        "activations_path": "s3://bucket/acts/layer62/",
        "hidden_states": tensor([[0, 0, ...]], dtype=torch.float8_e4m3fn),  # placeholder
        "hidden_states_scale": tensor([[0, 0, ...]], dtype=torch.float32)   # placeholder
    }
]
```

**Note**: This collator is only used for distillation training. The teacher activations are broadcast from SP rank 0 to other ranks in the training loop (`train_distill.py`), then padded and sliced for sequence parallelism.

### 4. PackingConcatCollator
**Purpose**: Concatenates all sequences into a single packed sequence for flash attention.

**Input**: `List[Dict]` (flat list of sequences)  
**Output**: `Dict[str, Tensor]` with batch_size=1

**Behavior**:
- Concatenates all sequences along the sequence dimension
- Generates `position_ids` for each sequence (each gets [0, 1, 2, ...])
- Generates `attention_mask` (all ones for non-padding)
- Adds flash attention kwargs (`cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`)
- Final shape: `[1, total_sequence_length]`

**Example**:
```python
# Input
[
    {"input_ids": tensor([1, 2, 3])},
    {"input_ids": tensor([4, 5])}
]

# Output
{
    "input_ids": tensor([[1, 2, 3, 4, 5]]),  # shape: [1, 5]
    "position_ids": tensor([[0, 1, 2, 0, 1]]),  # each sequence gets own positions
    "attention_mask": tensor([[1, 1, 1, 1, 1]]),
    "cu_seq_lens_q": tensor([0, 3, 5]),
    "cu_seq_lens_k": tensor([0, 3, 5]),
    "max_length_q": 3,
    "max_length_k": 3
}
```

### 5. TextSequenceShardCollator (Optional)
**Purpose**: Shards sequences for sequence parallelism (SP).

**Input**: `Dict[str, Tensor]`  
**Output**: `Dict[str, Tensor]` (sharded)

**Behavior**:
- Only active when `parallel_state.sp_enabled == True`
- Splits sequences across multiple devices for sequence parallelism
- Handles padding and synchronization

## Special Collators

### StreamDistillDataCollator
**Purpose**: Loads teacher activations from remote/local storage for distillation training.

**Usage**: Use before the main pipeline to augment features with teacher activations.

**Input**: `List[Dict]` (flat, from FlattenCollator)  
**Output**: `List[Dict]` (same structure with added `hidden_states` and `hidden_states_scale`)

**Process**:
1. Computes MD5 hashes of `input_ids` for activation lookup
2. Builds paths to activation files using `activations_path` from each feature
3. Downloads all activations in parallel (batch download for efficiency)
4. Adds `hidden_states` and optionally `hidden_states_scale` to each feature dict

**Example**:
```python
# Input
[
    {"input_ids": [1, 2, 3], "activations_path": "s3://bucket/acts"},
    {"input_ids": [4, 5], "activations_path": "s3://bucket/acts"}
]

# Output
[
    {
        "input_ids": [1, 2, 3],
        "activations_path": "s3://bucket/acts",
        "hidden_states": tensor([[...]]),  # teacher hidden states
        "hidden_states_scale": tensor([...])  # optional FP8 scale
    },
    {
        "input_ids": [4, 5],
        "activations_path": "s3://bucket/acts",
        "hidden_states": tensor([[...]]),
        "hidden_states_scale": tensor([...])
    }
]
```

## Pipeline Composition

### Standard Training Pipeline
```python
from xorl.data.data_loader import DataLoaderBuilder

builder = DataLoaderBuilder(
    dataset=dataset,
    micro_batch_size=2,
    gradient_accumulation_steps=4
)
# Default pipeline: ToTensor → Flatten → PackingConcat → [SequenceShard]
dataloader = builder.build()
```

### Distillation Training Pipeline
```python
from xorl.data.data_loader import DataLoaderBuilder
from xorl.data.collators import StreamDistillDataCollator

builder = DataLoaderBuilder(
    dataset=dataset,
    micro_batch_size=2,
    gradient_accumulation_steps=4,
    use_default_collators=False  # Build custom pipeline
)

# Custom pipeline for distillation
builder.add_collator(StreamDistillDataCollator(tokenizer=tokenizer))
builder.add_collator(ToTensorCollator())
builder.add_collator(FlattenCollator())
builder.add_collator(PackingConcatCollator())
if parallel_state.sp_enabled:
    builder.add_collator(TextSequenceShardCollator())

dataloader = builder.build()
```

### Customizing the Pipeline
```python
# Add collators at specific positions
builder.add_collator(MyPreprocessor(), position="start")
builder.add_collator(MyPostprocessor(), position="end")

# Insert at specific index
builder.insert_collator(MyMiddleCollator(), index=2)

# Remove a collator
builder.remove_collator(index=1)

# View the pipeline
builder.print_pipeline()
```

## Micro-Batch Splitting

The final collator pipeline is wrapped in a `MicroBatchCollator` that handles gradient accumulation:

```python
# If micro_batch_size=2 and gradient_accumulation_steps=4
# DataLoader provides 8 samples per batch
# MicroBatchCollator splits into 4 micro-batches of 2 samples each
# Each micro-batch is processed through the collator pipeline
# Returns: List[Dict] with length=4 (one per micro-batch)
```

## CollatePipeline

When multiple collators are used, they are wrapped in a `CollatePipeline`:

```python
from xorl.data.collators import CollatePipeline

# Manually create a pipeline
pipeline = CollatePipeline([
    ToTensorCollator(),
    FlattenCollator(),
    PackingConcatCollator()
])

# Use in dataloader
collated = pipeline(batch)
```

## Design Principles

1. **Single Responsibility**: Each collator does one thing well
2. **Composability**: Collators can be combined in different orders
3. **Backward Compatibility**: FlattenCollator passes through already-flat data
4. **Type Safety**: Clear input/output types for each collator
5. **Performance**: Batch operations (e.g., parallel downloads in StreamDistillDataCollator)
6. **Flexibility**: Easy to add custom collators or reorder existing ones

## File Structure

```
collators/
├── README.md                       # This file
├── __init__.py                     # Exports all collators
├── base_collator.py               # Base class for all collators
├── collate_pipeline.py            # Pipeline wrapper for multiple collators
├── flatten_collator.py            # Flattens nested lists
├── packing_concat_collator.py     # Concatenates sequences for flash attention
├── sequence_shard_collator.py     # Shards sequences for SP
├── stream_distill_collator.py     # Loads teacher activations
└── tensor_collator.py             # Converts to tensors
```

## Key Concepts

### Sequence Packing
Multiple sequences are concatenated into a single sequence with batch_size=1. Flash attention uses `cu_seq_lens` (cumulative sequence lengths) to know where each sequence starts/ends.

### Position IDs
Each sequence gets its own position indices starting from 0. This allows the model to understand token positions within each sequence despite concatenation.

### Gradient Accumulation
The `MicroBatchCollator` splits large batches into smaller micro-batches that fit in memory. Gradients are accumulated across micro-batches before updating weights.

## Adding a New Collator

1. Create a new file: `my_collator.py`
2. Inherit from `DataCollator`:
```python
from dataclasses import dataclass
from .base_collator import DataCollator

@dataclass
class MyCollator(DataCollator):
    my_param: int = 10
    
    def __call__(self, features):
        # Process features
        return processed_features
```
3. Export in `__init__.py`
4. Add to pipeline in `DataLoaderBuilder` or use `add_collator()`

## Troubleshooting

### Input/Output Shape Mismatches
Each collator expects specific input shapes. Check the pipeline order:
- `ToTensorCollator`: Works with any structure
- `FlattenCollator`: Handles both nested and flat
- `PackingConcatCollator`: Requires flat `List[Dict]`
- `TextSequenceShardCollator`: Requires `Dict[str, Tensor]`

### Activation Loading Failures
`StreamDistillDataCollator` will create masked placeholders if activations fail to load. Check:
- S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- S3 endpoint URL (S3_ENDPOINT_URL)
- Activation paths in dataset features

### Performance Issues
- Increase `stream_num_workers` for faster activation downloads
- Use batch downloads (already implemented in StreamDistillDataCollator)
- Increase `prefetch_factor` in DataLoader for more aggressive prefetching

