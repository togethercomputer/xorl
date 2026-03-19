---
title: Adding New Tests
---

## File and location

Put tests in the appropriate subdirectory under `tests/`. Mirror the source layout in `src/xorl/`:

```
src/xorl/data/collators/packing.py  →  tests/data/collators/test_packing.py
src/xorl/models/moe/routing.py      →  tests/models/test_moe_routing.py
```

## Basic structure

```python
import pytest
import torch


pytestmark = [pytest.mark.cpu]  # apply marker(s) to entire module


class TestMyFeature:
    def test_basic(self):
        result = my_function(input)
        assert result == expected

    def test_tensor_values(self):
        out = my_op(torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out, torch.tensor([2.0, 4.0]))
```

## Marking tests

Apply markers at the module level (affects all tests in the file) or per test:

```python
# Whole module requires GPU
pytestmark = [pytest.mark.gpu]

# Single test is slow
@pytest.mark.slow
def test_large_batch():
    ...

# Multiple markers on one test
@pytest.mark.gpu
@pytest.mark.slow
def test_large_model():
    ...
```

## Using shared fixtures

The root `conftest.py` provides these fixtures out of the box:

```python
def test_with_dataset(fake_text_dataset):
    # FakeTextDataset: 100 samples, seq_len=128, vocab_size=1000
    assert len(fake_text_dataset) == 100
    sample = fake_text_dataset[0]
    assert sample["input_ids"].shape == (128,)

def test_with_packed_dataset(fake_packed_dataset):
    # FakePackedDataset: 100 samples, 3 packed seqs each, position_ids included
    sample = fake_packed_dataset[0]
    assert "position_ids" in sample

def test_with_collator_input(sample_features):
    # List of 2 dicts with input_ids, attention_mask, labels (len=5)
    assert len(sample_features) == 2

def test_with_packed_input(sample_packed_features):
    # Same as sample_features but includes position_ids with packed offsets
    assert "position_ids" in sample_packed_features[0]
```

For E2E tests, use the fixtures from `tests/e2e/conftest.py`:

```python
def test_lora_training(tiny_moe_model_dir_with_weights, tmp_workspace):
    # tiny_moe_model_dir_with_weights is a path to a saved random-init Qwen3-MoE
    config = load_config(model_dir=tiny_moe_model_dir_with_weights, ...)
    ...
```

## Writing a distributed test

Distributed tests run with plain `pytest`. For tests that require multiple processes, use `run_distributed_script` from `tests/distributed/distributed_utils.py` to spawn torchrun as a subprocess — the same pattern as e2e tests.

Write the distributed logic in a standalone script, then call it from the test:

```python
# tests/distributed/scripts/my_distributed_logic.py
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    tensor = torch.ones(4, device="cuda") * rank
    dist.all_reduce(tensor)
    expected = torch.ones(4, device="cuda") * sum(range(dist.get_world_size()))
    assert torch.equal(tensor, expected), f"allreduce failed on rank {rank}"
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```python
# tests/distributed/test_my_feature.py
import pytest
from tests.distributed.distributed_utils import run_distributed_script, skip_if_gpu_count_less_than

SCRIPT_PATH = "tests/distributed/scripts/my_distributed_logic.py"

pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


class TestMyFeature:
    @skip_if_gpu_count_less_than(2)
    def test_allreduce_2gpu(self):
        result = run_distributed_script(SCRIPT_PATH, num_gpus=2)
        result.assert_success()
```

Run it with plain pytest:
```bash
pytest tests/distributed/test_my_feature.py -v
```

## Parameterized tests

```python
@pytest.mark.parametrize("seq_len,expected_chunks", [
    (128, 2),
    (256, 4),
    (512, 8),
])
def test_chunking(seq_len, expected_chunks):
    chunks = split_sequence(seq_len, chunk_size=64)
    assert len(chunks) == expected_chunks
```

## Mocking distributed state

For tests that call into distributed code paths but don't need real multi-process setup:

```python
from unittest.mock import patch, MagicMock

def test_uses_parallel_state():
    mock_state = MagicMock()
    mock_state.dp_size = 4
    mock_state.dp_rank = 1

    with patch("xorl.distributed.parallel_state.get_parallel_state", return_value=mock_state):
        result = function_that_calls_parallel_state()
        assert result == expected
```
