# Data Collator Pipeline

Data collators that process batches in a pipeline for training. Each collator transforms data one step closer to model-ready tensors.

## Pipeline Architecture

```
Dataset Output: List[List[Dict]] or List[Dict]
    ↓
1. ToTensorCollator              → List[List[Dict]] or List[Dict] (with tensor values)
    ↓
2. FlattenCollator               → List[Dict] (flatten nested, pass through flat)
    ↓
3. ShiftTokensCollator (optional) → List[Dict] (shift input_ids/labels for causal LM)
    ↓
4. PackingConcatCollator         → Dict[str, Tensor] (concatenate, batch_size=1)
    ↓
5. TextSequenceShardCollator (optional) → Dict[str, Tensor] (shard if SP enabled)
    ↓
Final Output: Dict[str, Tensor]
```

## Collator Components

### 1. ToTensorCollator
Converts raw Python lists/arrays to PyTorch tensors. Preserves input structure.

```python
# Input:  [{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}]
# Output: [{"input_ids": tensor([1, 2, 3]), "labels": tensor([4, 5, 6])}]
```

### 2. FlattenCollator
Flattens nested list structures into a single flat list. Passes through already-flat data unchanged.

```python
# Nested input → flattened
[[{"input_ids": [1, 2]}, {"input_ids": [3, 4]}], [{"input_ids": [5, 6]}]]
→ [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}, {"input_ids": [5, 6]}]
```

### 3. ShiftTokensCollator (Optional)
Shifts tokens for causal language modeling when data is in HF format (unshifted: `labels[i] == input_ids[i]`). Drops last token from `input_ids`, drops first token from `labels`.

Auto-detects whether shifting is needed:
- **HF format** (needs shift): `labels[i] == input_ids[i]` → shift applied
- **Tomi API format** (already shifted): `labels[i] == input_ids[i+1]` → no shift
- **`target_tokens` field present** → no shift (xorl_client format)

```python
# HF format input (unshifted)
{"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]}
# After shift
{"input_ids": [1, 2, 3, 4], "labels": [2, 3, 4, 5]}
```

Also shifts `weights`, `advantages`, `logprobs` fields if present.

### 4. PackingConcatCollator
Concatenates all sequences into a single packed sequence for flash attention.

```python
# Input
[{"input_ids": tensor([1, 2, 3])}, {"input_ids": tensor([4, 5])}]

# Output
{
    "input_ids": tensor([[1, 2, 3, 4, 5]]),       # shape: [1, 5]
    "position_ids": tensor([[0, 1, 2, 0, 1]]),     # per-sequence positions
    "attention_mask": tensor([[1, 1, 1, 1, 1]]),
    "cu_seq_lens_q": tensor([0, 3, 5]),
    "cu_seq_lens_k": tensor([0, 3, 5]),
    "max_length_q": 3,
    "max_length_k": 3
}
```

### 5. TextSequenceShardCollator (Optional)
Shards sequences for sequence parallelism (SP). Only active when `parallel_state.cp_enabled == True`.

## Pipeline Composition

### Standard Training
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

### Custom Pipeline
```python
builder.add_collator(ShiftTokensCollator(), position="start")
builder.add_collator(MyPostprocessor(), position="end")
builder.insert_collator(MyMiddleCollator(), index=2)
builder.remove_collator(index=1)
builder.print_pipeline()
```

## Micro-Batch Splitting

The collator pipeline is wrapped in a `MicroBatchCollator` for gradient accumulation:
- `micro_batch_size=2`, `gradient_accumulation_steps=4` → DataLoader provides 8 samples
- `MicroBatchCollator` splits into 4 micro-batches of 2, each processed through the pipeline

## CollatePipeline

When using collators directly (without `DataLoaderBuilder`):

```python
from xorl.data.collators import CollatePipeline, ToTensorCollator, FlattenCollator, PackingConcatCollator

pipeline = CollatePipeline([
    ToTensorCollator(),
    FlattenCollator(),
    PackingConcatCollator()
])
collated = pipeline(batch)
```

## Key Concepts

- **Sequence Packing**: Multiple sequences are concatenated into a single sequence with `batch_size=1`. Flash attention uses `cu_seq_lens` (cumulative sequence lengths) to know where each sequence starts/ends.
- **Position IDs**: Each sequence gets its own position indices starting from 0, allowing the model to understand token positions despite concatenation.
- **Gradient Accumulation**: `MicroBatchCollator` splits large batches into smaller micro-batches that fit in memory. Gradients are accumulated across micro-batches before updating weights.

## Adding a New Collator

1. Create a new file inheriting from `DataCollator`:
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
2. Export in `__init__.py`
3. Add to pipeline via `DataLoaderBuilder.add_collator()` or `CollatePipeline`

## File Structure

```
collators/
├── README.md
├── __init__.py
├── base_collator.py               # Base class
├── collate_pipeline.py            # Pipeline wrapper
├── flatten_collator.py            # Flatten nested lists
├── packing_concat_collator.py     # Pack sequences for flash attention
├── sequence_shard_collator.py     # Shard for SP
├── shift_tokens_collator.py       # Shift tokens for causal LM
└── tensor_collator.py             # Convert to tensors
```
