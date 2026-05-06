from unittest.mock import Mock

import torch

from xorl.data.constants import IGNORE_INDEX
from xorl.distributed import sync_padding


def test_synchronize_micro_batch_padding_extends_cu_seqlens(monkeypatch):
    monkeypatch.setattr(sync_padding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(sync_padding.dist, "get_world_size", lambda group=None: 8)
    monkeypatch.setattr(sync_padding, "get_device_type", lambda: "cpu")

    ps = Mock()
    ps.cp_enabled = False
    monkeypatch.setattr(sync_padding, "get_parallel_state", lambda: ps)

    def fake_all_reduce(tensor, op=None, group=None):
        tensor.fill_(512)

    monkeypatch.setattr(sync_padding.dist, "all_reduce", fake_all_reduce)

    micro_batches = [
        {
            "input_ids": torch.ones(1, 176, dtype=torch.long),
            "labels": torch.ones(1, 176, dtype=torch.long),
            "position_ids": torch.arange(176).unsqueeze(0),
            "attention_mask": torch.ones(1, 176, dtype=torch.long),
            "cu_seq_lens_q": torch.tensor([0, 83, 167, 176], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0, 83, 167, 176], dtype=torch.int32),
            "max_length_q": torch.tensor(84, dtype=torch.int32),
            "max_length_k": torch.tensor(84, dtype=torch.int32),
        },
        {
            "input_ids": torch.ones(1, 512, dtype=torch.long),
            "labels": torch.cat(
                [
                    torch.ones(176, dtype=torch.long),
                    torch.full((336,), IGNORE_INDEX, dtype=torch.long),
                ]
            ).unsqueeze(0),
            "position_ids": torch.arange(512).unsqueeze(0),
            "attention_mask": torch.cat(
                [
                    torch.ones(176, dtype=torch.long),
                    torch.zeros(336, dtype=torch.long),
                ]
            ).unsqueeze(0),
            "cu_seq_lens_q": torch.tensor([0, 83, 167, 176], dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor([0, 83, 167, 176], dtype=torch.int32),
            "max_length_q": torch.tensor(84, dtype=torch.int32),
            "max_length_k": torch.tensor(84, dtype=torch.int32),
        },
    ]

    sync_padding.synchronize_micro_batch_padding(micro_batches)

    for mb in micro_batches:
        assert mb["input_ids"].shape[-1] == 512
        assert mb["labels"].shape[-1] == 512
        assert mb["position_ids"].shape[-1] == 512
        assert mb["attention_mask"].shape[-1] == 512
        assert torch.equal(mb["labels"][0, 176:], torch.full((336,), IGNORE_INDEX))
        assert mb["attention_mask"][0, 176:].sum().item() == 0

        assert mb["cu_seq_lens_q"].tolist() == [0, 83, 167, 512]
        assert mb["cu_seq_lens_k"].tolist() == [0, 83, 167, 512]
        assert mb["max_length_q"] == 345
        assert mb["max_length_k"] == 345
        assert isinstance(mb["max_length_q"], int)
        assert isinstance(mb["max_length_k"], int)
